"""分片样本源的并行读取 helper(profile / baseline 的 loader 加速)。

背景(实测,2026-10-25,req-t7 400w×1513 二分类):

    profile 总墙钟 15.3 min 里,``load_data()`` 占 5.9 min(38%)。拆开是
    「8 个分片串行读 → 逐分片 to_pandas → 逐列 pd.to_numeric 处理 decimal
    退化列 → pd.concat 8 大帧」四段。四段都有结构性浪费:

    | 段 | 原实现 | 实测 |
    |---|---|---|
    | 读+to_pandas | 逐分片串行 | 229.4 s |
    | decimal 治理 | pandas object → to_numeric | 38.5 s |
    | 拼装 | ``pd.concat`` 8 个宽帧 | 59.3 s |
    | 合计 | | **329.3 s** |

    本模块的替代路径:**多线程并行读分片 → 在 Arrow 层 ``pc.cast`` 把
    decimal 列直接转 float64 → ``pa.concat_tables``(零拷贝)→ 一次
    ``to_pandas``**。同一份数据实测 **173.6 s**(保留全部 1513 列,内存
    100.3 GB 与旧路径同量级),省 48%。

为什么 decimal 必须在 Arrow 层转:
    ``decimal128`` 落到 pandas 是 object;逐列 ``pd.to_numeric`` 要先物化
    Python 对象再解析。Arrow 侧 ``cast`` 是向量化的,且能顺手把
    ``to_pandas`` 的 object 列数从 16 降到 5(实测 dtype 分布完全一致)。

为什么快照一致性要求"一次 to_pandas":
    分块 ``to_pandas`` 再拼接会走 ``pd.concat`` 的大帧路径(59 s 是这个
    代价);``pa.concat_tables`` 之后一次性转换可以省掉它。并行分块
    ``to_pandas`` 实测 180.9 s(转 92 s + concat 39 s),不优于单次 173.6 s。

不改变的语义:
    - 行序 = 分片文件名排序序(与 ``sorted(glob(...))`` 完全一致);
    - 列序 = 首个非空分片的 schema 列序;
    - 空分片(``metadata.num_rows == 0``)跳过;
    - ``use_threads`` 仍交给 pyarrow,本模块只并行**分片**这一层。

约定:本模块只做"读 + dtype 治理",不做切分/采样/特征工程——那是
Harness 的职责,loader 保持纯读取。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

#: 并行读分片的默认线程数。分片读取是 IO + 解压混合负载,实测 4 路
#: 已达 1.53×(8 分片 229.4 s → 150.1 s);更高并发收益递减(解压吃 CPU,
#: 而调用方通常还要把它和其它分析抢核)。可用 ``threads`` 覆盖。
DEFAULT_SHARD_THREADS = 4

#: Arrow 里会退化成 pandas object 的数值类型。decimal 一律转 float64:
#: 金额列的小数位在 float32 下会丢精度(实测 8 位有效数字时 float32 已
#: 不可分辨),而这些列在建模里要么被 banned、要么只做分箱,float64 的
#: 内存代价(相比 float32)远小于精度损失的代价。
_DECIMAL_PREFIX = "decimal"


def _is_decimal_type(dtype) -> bool:
    return str(dtype).startswith(_DECIMAL_PREFIX)


def read_parquet_shards(
    paths: Sequence[str | Path],
    *,
    threads: int = DEFAULT_SHARD_THREADS,
    cast_decimals: bool = True,
    post: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    skip_empty: bool = True,
) -> pd.DataFrame:
    """读入多个 parquet 分片,返回合并后的单个 DataFrame。

    参数
    ----
    paths : 分片路径;内部按字符串排序(与 ``sorted(glob(...))`` 同序)。
    threads : 并行读分片的线程数;``<=1`` 时退化为串行。
    cast_decimals : Arrow 层把 decimal 列转 float64(默认开)。
    post : 可选的行内后处理(如 object 列降 float32)。**必须**是逐列
        就地变换,不得改变行数/行序——本函数已把分片拼好,后处理再动行序
        会让调用方拿到的行序失去"分片名排序"这个可预期语义。
    skip_empty : 跳过 0 行分片(常见于占位的 ``part-000N`` 空分片)。

    返回
    ----
    合并后的 DataFrame;全部分片为空时抛 ``ValueError``。
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    ordered = sorted(str(p) for p in paths)
    if not ordered:
        raise ValueError("read_parquet_shards: 没有输入分片")

    def _read_one(path: str) -> "pa.Table | None":
        pf = pq.ParquetFile(path)
        if skip_empty and pf.metadata.num_rows == 0:
            return None
        table = pf.read(use_threads=True)
        if cast_decimals:
            names = table.schema.names
            cols = []
            changed = False
            for i, name in enumerate(names):
                col = table.column(i)
                if _is_decimal_type(col.type):
                    cols.append(pc.cast(col, pa.float64()))
                    changed = True
                else:
                    cols.append(col)
            if changed:
                table = pa.Table.from_arrays(cols, names=names)
        return table

    if threads and threads > 1 and len(ordered) > 1:
        with ThreadPoolExecutor(max_workers=min(int(threads), len(ordered))) as ex:
            parts = list(ex.map(_read_one, ordered))
    else:
        parts = [_read_one(p) for p in ordered]
    parts = [p for p in parts if p is not None]
    if not parts:
        raise ValueError("read_parquet_shards: 全部分片为空")

    table = parts[0] if len(parts) == 1 else pa.concat_tables(parts)
    del parts
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    if post is not None:
        df = post(df)
    return df


def shrink_object_columns(
    frame: pd.DataFrame,
    *,
    keep_as_is: Iterable[str] = (),
    skip_prefixes: Iterable[str] = (),
    min_parse_ratio: float = 0.95,
    min_coverage: float = 0.5,
) -> pd.DataFrame:
    """把 object 形态的数值列降为 float32(就地,返回同一帧)。

    这是"decimal 已由 Arrow 处理完"之后仍然残留的 object 列治理:上游
    若还有以字符串承载的数值(numpy 数值被读成 string),这里统一处理。
    判据两道,全部通过才降 dtype:

    - ``min_parse_ratio``:非空值里能解析成数字的比例(``safe_to_numeric``
      的口径,默认 0.95)——同时用它的预检路径避免长 object 列上的递归
      爆栈(见 ``runtime.safe_dtype`` 的实测事故);
    - ``min_coverage``:解析出的非空值占**全部行**的比例(默认 0.5,与历史
      loader 的 ``notna().sum() >= 0.5 * len`` 一致),防"稀疏但可解析"的
      列被误当数值。
    """
    from .safe_dtype import safe_to_numeric

    keep = set(keep_as_is)
    prefixes = tuple(skip_prefixes)
    for name, dtype in frame.dtypes.items():
        if dtype != object:
            continue
        if name in keep:
            continue
        if prefixes and name.startswith(prefixes):
            continue
        numeric = safe_to_numeric(frame[name], max_rows=None,
                                  min_parse_ratio=min_parse_ratio)
        if numeric is None:
            continue
        if numeric.notna().sum() >= min_coverage * len(numeric):
            frame[name] = numeric.astype("float32")
    return frame
