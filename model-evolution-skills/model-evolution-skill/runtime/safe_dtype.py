"""安全 dtype 治理:把 object/string 列数值化,但不进入 pandas 的递归解析路径。

背景(实测事故,2026-09-18):
    `pd.to_numeric(series, errors="coerce")` 在**含非数字值的长 object 列**
    上会走 `pandas._libs.lib.maybe_convert_numeric` 的失败分支。pandas 2.3.3
    的实现是递归的,每层吃掉一块栈;实测在「连续失败段 ≥ 1978 行」时栈耗尽
    并 **SIGSEGV(exit 139)** —— `errors="coerce"` 只处理解析失败,**不防爆栈**。

    复现(默认 12.5MB 栈):
        pd.to_numeric(pd.Series([md5(str(i)) for i in range(1978)]),
                      errors="coerce")   # → 段错误
    阈值与栈上限成比例(44638 行需要 ~220MB 栈才不崩),所以"加大栈"只是
    掩盖:容器默认栈就会踩,且评测侧无法控制执行环境。

    更隐蔽的是**静默数据污染**:同一个解析器会把 24 位十六进制样本键
    当成合法数字,`"000000000000000000000000"` → `0.0`,于是
    `notna().sum() >= 0.95 * n` 这类"是否可数值化"的判据会**误判为真**,
    把键列/ID 列放进模型面。

本模块的对策:先做一次全列向量化正则预检,把"整列都是数字字面量"与
"含非数字值"分开处理,两条路都不触发递归失败分支:
    - 全部匹配  → 交给 `pd.to_numeric`(无失败 ⇒ 无递归,快且语义一致);
    - 存在不匹配 → 逐元素 Python 解析(显式循环,无递归,语义一致)。

调用方拿到 `None` 即表示"该列不是数值列",按既有的类别/跳过分支处理,
不再需要自己写 `notna().sum() >= 0.95 * n` 这类会被污染的判据。
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

#: 数字字面量(与 pandas 的常规解析口径一致:整数、小数、科学计数法、正负号)。
#: 刻意不含 inf/nan 等特殊词——它们在 pandas 里是"解析成功但非数字字面量",
#: 放进特征面只会制造 inf/nan 噪声;需要时由调用方显式处理。
_NUM_STR_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

#: 非特征列:绝不参与数值化/画像。样本键是哈希身份、时间列是切分坐标、
#: 实体列是 ID、目标是标签本体——它们都不该出现在任何"特征统计"里。
NON_FEATURE_COLUMNS = frozenset({"__mvo_sample_key"})


def safe_to_numeric(
    s: pd.Series,
    *,
    max_rows: int | None = None,
    min_parse_ratio: float = 0.95,
) -> pd.Series | None:
    """把 `s` 数值化;不是数值列时返回 `None`。

    Parameters
    ----------
    s : pd.Series
        待治理列。已是数值 dtype 时原样返回(不复制)。
    max_rows : int | None
        仅用于超大列的显式短路(超过则直接判非数值,避免逐元素慢路径)。
        默认 None = 不限。
    min_parse_ratio : float
        混合列中"可解析值占比"的下限。低于它即判非数值列。默认 0.95,
        与调用方历史上的 `notna().sum() >= 0.95 * n` 口径一致。

    Returns
    -------
    pd.Series | None
        数值化结果(dtype 与 `pd.to_numeric` 一致);不可数值化时为 `None`。
    """
    if pd.api.types.is_numeric_dtype(s):
        return s

    if s.dtype != object and not pd.api.types.is_string_dtype(s):
        return None

    nn = s.dropna()
    if nn.empty:
        return None

    try:
        as_str = nn.astype(str)
    except Exception:  # noqa: BLE001
        return None

    try:
        all_numeric = bool(as_str.str.match(_NUM_STR_RE).all())
    except Exception:  # noqa: BLE001
        return None

    if all_numeric:
        # 无失败值 ⇒ pandas 不进入递归失败分支,可安全调用。
        return pd.to_numeric(s, errors="coerce")

    if max_rows is not None and len(s) > max_rows:
        return None

    # 混合列:逐元素解析。显式循环、无递归,不会爆栈。
    values = nn.to_numpy(dtype=object)
    parsed = np.full(len(values), np.nan, dtype="float64")
    for i, v in enumerate(values):
        if isinstance(v, (int, float, np.integer, np.floating)):
            parsed[i] = float(v)
            continue
        try:
            parsed[i] = float(v)
        except (TypeError, ValueError):
            continue

    # 关键:必须要求**绝大多数**非空值可解析,而不是"至少有一个"。
    # 十六进制键列(如 `f"{i:024x}"`)里前段恰好是纯十进制数字,逐元素
    # 解析会成功一部分、失败一部分 —— 只看 `any()` 就会把它当成数值列,
    # 正是本模块要消灭的"静默污染"。阈值沿用调用方的历史口径(95%)。
    ratio = float(np.isfinite(parsed).mean())
    if ratio < min_parse_ratio:
        return None

    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[nn.index] = parsed
    return out


def is_feature_column(col: str, *, extra_exclude=()) -> bool:
    """该列是否允许进入特征统计/画像(排除样本键、时间、实体、目标)。"""
    if col in NON_FEATURE_COLUMNS or col in extra_exclude:
        return False
    return not str(col).startswith("__mvo_")
