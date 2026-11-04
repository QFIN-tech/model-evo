"""data-loader contract 与 split protocol。

data-loader.py 是任务工作区中唯一针对任务定制的 Harness 输入(§7.3):
它必须暴露 ``load_data() -> pandas.DataFrame``。切分逻辑全部固化在
本模块,Agent / Trial 代码不可修改(§2.1)。
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .logging_utils import get_logger, now_iso, dump_json
from .spec import TaskSpec

logger = get_logger(__name__)

SPLIT_NAMES = ("train", "validation", "shadow_test", "final_test")


# ---------------------------------------------------------------------------
# data-loader contract
# ---------------------------------------------------------------------------

def load_dataframe(spec: TaskSpec, run_dir: Path | None = None) -> pd.DataFrame:
    """通过任务工作区 data-loader.py 加载原始样本。

    契约:data-loader.py 暴露 ``load_data() -> pd.DataFrame``;
    环境变量 MODELEVO_TASK_SPEC / MODELEVO_RUN_DIR 提供上下文。

    v2.17.4:执行 loader 前把 **skill 根** 放进 ``sys.path``,这样 loader
    可以直接 ``from runtime.loaders import read_parquet_shards`` 复用
    Harness 的分片并行读取,而不必在任务目录里复制一份实现(或退回串行
    的 ``for f in sorted(glob(...))``)。实测 req-t7 400w×1513 的分片读取
    串行 329 s、并行 174 s。注入是幂等的,且只加一次。
    """
    loader_path = spec.data.get("data_loader")
    if loader_path is None:
        # TaskSpec 的规范位置是 ``data.data_loader``;顶层 ``data_loader``
        # 只是旧式写法。相对路径按 run_dir 解析(loader 通常就在 run 根)。
        nested = (spec.data.get("data") or {}).get("data_loader")
        if nested:
            cand = Path(nested)
            if run_dir is not None and not cand.is_absolute():
                cand = Path(run_dir) / cand
            loader_path = str(cand)
    if loader_path is None and run_dir is not None:
        candidate = Path(run_dir) / "data-loader.py"
        loader_path = str(candidate) if candidate.exists() else None
    if loader_path is None:
        raise ValueError("TaskSpec 未提供 data.data_loader,且 run 目录无 data-loader.py")

    path = Path(loader_path)
    if not path.exists():
        raise FileNotFoundError(f"data-loader 不存在: {path}")

    import sys

    skill_root = str(Path(__file__).resolve().parent.parent)
    if skill_root not in sys.path:
        sys.path.insert(0, skill_root)

    spec_mod = importlib.util.spec_from_file_location("task_data_loader", path)
    module = importlib.util.module_from_spec(spec_mod)
    os.environ.setdefault("MODELEVO_TASK_SPEC", str(getattr(spec, "_spec_path", "")))
    if run_dir is not None:
        os.environ.setdefault("MODELEVO_RUN_DIR", str(run_dir))
    try:
        spec_mod.loader.exec_module(module)
        df = module.load_data()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"data-loader 执行失败({path}): {exc}") from exc

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"data-loader 必须返回 DataFrame,得到 {type(df)}")
    if df.index.has_duplicates:
        df = df.reset_index(drop=True)
    return df


def resolve_feature_columns(df: pd.DataFrame, spec: TaskSpec) -> list[str]:
    """按 TaskSpec 白名单/排除列解析特征列;目标、时间、ID、treatment 列剔除。

    v13:banned 与 reparse 列都从通用特征面剔除(reparse 走独立访问器
    reparse_frames(),不进通用面);banned 的主防御不在这里——
    materialize_splits 落盘时物理不写入,这里只是第二道名字过滤。
    """
    drop = {spec.target_name, spec.time_column, spec.id_column,
            SAMPLE_KEY_COLUMN}
    drop.discard(None)
    treatment = (spec.data.get("treatment") or {}).get("column")
    if treatment:
        drop.add(treatment)
    drop.update(spec.excluded_columns)

    if spec.feature_columns:
        missing = [c for c in spec.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"特征白名单中的列不存在: {missing[:5]}")
        return [c for c in spec.feature_columns if c not in drop]
    return [c for c in df.columns if c not in drop]


# ---------------------------------------------------------------------------
# v13/v21 四产物物理隔离(banned 列在任何数据产物中不存在)
# ---------------------------------------------------------------------------

LABELS_SUBDIR = "labels"
REPARSE_SUBDIR = "reparse"
IDENTITY_SUBDIR = "identity"
SAMPLE_KEY_COLUMN = "__mvo_sample_key"


def _split_face_frames(chunk: pd.DataFrame, spec: TaskSpec
                       ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """把一个 split 帧切成三个可消费面:特征 / 标签 / 解析源。

    第四个 identity 面由写入方通过 :func:`sample_keys` 单独生成，只供
    框架对齐和完整性校验，Trial 的变换代码不消费它。

    - 特征面:全部列 − banned − reparse − 目标/时间/ID/treatment。
      时间/ID/treatment 保留在标签面(行对齐与审计需要),特征面剔除;
    - 标签面:目标 + treatment(存在时)+ 时间/ID(对齐键);
    - 解析源:reparse 原列 + 时间/ID(对齐键)。

    banned 列不写入任何一个数据面——它从此只存在于 load_data() 的
    原始输出与分类审计内部,任何下游 transform 物理接触不到。
    """
    banned = set(spec.banned_columns)
    reparse = set(spec.reparse_columns)
    key_cols = [c for c in (spec.time_column, spec.id_column) if c]
    label_cols = [c for c in ([spec.target_name, spec.treatment_column]
                              + key_cols) if c and c in chunk.columns]
    drop = (banned | reparse | {spec.target_name, spec.time_column,
                                spec.id_column, spec.treatment_column,
                                SAMPLE_KEY_COLUMN})
    drop.discard(None)
    drop -= {c for c in drop if c not in chunk.columns}
    face = chunk.drop(columns=sorted(drop))
    labels = chunk[label_cols]
    reparse_cols = [c for c in reparse if c in chunk.columns]
    if reparse_cols:
        rep = chunk[reparse_cols + key_cols]
    else:
        # 无 reparse 列时返回零列帧(实测事故回写:iloc[0:0] 切的是行,
        # shape[1] 仍是全列数且恒 >0,三个生产方都会误触发落盘 →
        # reparse/ 目录因 spec.reparse_columns 为空未创建 → OSError;
        # 更糟的分支:spec 声明了 reparse 但列不在 chunk 时,会把含
        # banned+目标列的全 schema 0 行 parquet 写进 reparse/)
        rep = chunk.iloc[:, 0:0]
    if spec.reparse_columns and not reparse_cols:
        # 声明了 reparse 列却在数据中不存在:显式报错而不是静默落
        # 空产物(has_reparse_dir 会谎报存在合法解析源)
        raise ValueError(
            f"reparse 列 {sorted(spec.reparse_columns)} 不在数据中"
            f"(可用列示例: {list(chunk.columns)[:8]});请核对 task-spec")
    return face, labels, rep


def sample_keys(frame: pd.DataFrame, spec: TaskSpec, *,
                require_unique: bool = True) -> pd.Series:
    """构造不进入模型面的稳定样本键。

    键只使用 TaskSpec 声明的时间列与实体列。跨视图比较必须拥有同一组
    唯一键；如果任务没有足够的业务键，宁可把比较声明为不可比，也不
    能退化成按行位置配对。返回值使用固定 SHA-256 摘要，避免在预测
    artifact 中暴露原始实体标识。
    """
    if SAMPLE_KEY_COLUMN in frame.columns:
        keys = frame[SAMPLE_KEY_COLUMN].astype(str).reset_index(drop=True)
        keys.name = SAMPLE_KEY_COLUMN
        if require_unique and keys.duplicated().any():
            raise ValueError("继承的样本键不唯一")
        return keys

    key_cols = [c for c in (spec.time_column, spec.id_column)
                if c and c in frame.columns]
    if key_cols:
        raw = frame[key_cols[0]].astype(str)
        for col in key_cols[1:]:
            raw = raw.str.cat(frame[col].astype(str), sep="\x1f")
    else:
        # random/group 任务可能没有业务 ID。使用非标签行内容的稳定
        # 指纹，而不是行号；跨视图若改变了内容，键集合会自然不一致，
        # evaluator 会拒绝比较，不会产生“看似可比”的伪证据。
        safe_cols = [c for c in frame.columns
                     if c not in {spec.target_name, spec.treatment_column}]
        if not safe_cols:
            raise ValueError(
                "无法构造样本身份:既无 time/entity 键，也无非标签列")
        raw = pd.util.hash_pandas_object(
            frame[safe_cols], index=False).astype(str)
        key_cols = safe_cols
    # 业务键不唯一时用非标签行内容补强。仍完全相同的重复行在数据上
    # 不可区分，只能附确定性 occurrence；这类键跨重排视图会不匹配并
    # 被完整性门拒绝，绝不回退到位置配对。
    if raw.duplicated(keep=False).any():
        safe_cols = [c for c in frame.columns
                     if c not in {spec.target_name, spec.treatment_column}]
        if safe_cols:
            row_hash = pd.util.hash_pandas_object(
                frame[safe_cols], index=False).astype(str)
            raw = raw.str.cat(row_hash, sep="\x1f")
    if raw.duplicated(keep=False).any():
        occurrence = raw.groupby(raw, sort=False).cumcount().astype(str)
        raw = raw.str.cat(occurrence, sep="\x1f")
    keys = raw.map(lambda value: hashlib.sha256(
        value.encode("utf-8")).hexdigest()[:24])
    keys.name = SAMPLE_KEY_COLUMN
    if require_unique and keys.duplicated().any():
        n_dup = int(keys.duplicated(keep=False).sum())
        raise ValueError(
            f"样本键不唯一:{n_dup}/{len(keys)} 行的 {key_cols} 组合重复")
    return keys.reset_index(drop=True)


def identity_frame(frame: pd.DataFrame, spec: TaskSpec) -> pd.DataFrame:
    """无标签对齐面：时间坐标 + 不可逆样本键。

    该面服务于 data view 的行序保持和预测绑定，不含 target/treatment，
    因而可覆盖 final_test。样本键若已由父视图携带则原样保留，禁止因
    特征变换而重新计算身份。原始实体 ID 也不写入此面：视图对齐以
    哈希键完成，final_test 的视图脚本不需要接触可外查的业务标识。
    """
    cols = [c for c in (spec.time_column,) if c and c in frame.columns]
    out = frame[cols].reset_index(drop=True).copy()
    if SAMPLE_KEY_COLUMN in frame.columns:
        keys = frame[SAMPLE_KEY_COLUMN].astype(str).reset_index(drop=True)
        if keys.duplicated().any():
            raise ValueError("identity 面的样本键重复")
    else:
        keys = sample_keys(frame, spec)
    out[SAMPLE_KEY_COLUMN] = keys.values
    return out


def assert_face_clean(face_dir: Path, spec: TaskSpec) -> list[str]:
    """数据面后置断言:任何 parquet 不得含 banned 列(schema 级读取,
    零 IO 成本,毫秒级)。

    对产物的校验比对脚本行为的约束可靠一个数量级——view 构建脚本
    即使重新调 load_data() 聚合,产物也过不了这道断言。返回违规
    清单(空 = 通过),调用方据此 raise。
    """
    import pyarrow.parquet as pq

    face_dir = Path(face_dir)
    banned = set(spec.banned_columns)
    violations: list[str] = []
    if not banned:
        return violations
    for pq_path in sorted(face_dir.glob("*.parquet")):
        try:
            names = set(pq.read_schema(pq_path).names)
        except Exception as exc:  # noqa: BLE001
            violations.append(f"{pq_path.name}: schema 读取失败 {exc}")
            continue
        hit = banned & names
        if hit:
            violations.append(
                f"{pq_path.name}: 泄漏列渗入数据面 {sorted(hit)}")
    return violations


def _assert_view_clean(view_dir: Path, spec: TaskSpec) -> None:
    """四产物目录断言(特征/标签/解析源/身份面);违例 raise。

    v14:旧实现只扫 view_dir 顶层的 *.parquet(glob 非递归),
    labels/ 与 reparse/ 子目录从不打开——reparse/ 里的 banned 列
    schema 泄漏漏检(审计实测:0 行全 schema reparse 产物可带 banned
    列名通过)。数据产物目录逐层断言。
    """
    view_dir = Path(view_dir)
    violations = list(assert_face_clean(view_dir, spec))
    for sub in (LABELS_SUBDIR, REPARSE_SUBDIR):
        if (view_dir / sub).is_dir():
            violations += [f"{sub}/{v}"
                           for v in assert_face_clean(view_dir / sub, spec)]
    if violations:
        raise ValueError(
            "数据面泄漏断言失败(banned 列必须物理不存在,见"
            f"leak-classification): {'; '.join(violations)}")

    # identity 是受保护的非模型面，只允许时间坐标与哈希样本键。
    identity_dir = view_dir / IDENTITY_SUBDIR
    allowed = {SAMPLE_KEY_COLUMN, spec.time_column} - {None}
    if identity_dir.is_dir():
        import pyarrow.parquet as pq

        for path in identity_dir.glob("*.parquet"):
            extra = set(pq.read_schema(path).names) - allowed
            if extra:
                raise ValueError(
                    f"identity/{path.name} 含非对齐列:{sorted(extra)}")


def load_labels(run_dir: Path, splits_dir: str = "sample-profile/splits",
                splits: tuple[str, ...] | None = None) -> dict[str, pd.DataFrame]:
    """读取标签面(splits-labels/):目标 + treatment + 对齐键。

    Trial 的 y/w 装配与评估器都从这里取标签;标签独立存文件使
    "手写特征面带回 label 同族列"从根上不可能(历史事故:AUC=1.0)。
    """
    base = _prescreen_run_dir(run_dir) / splits_dir
    out: dict[str, pd.DataFrame] = {}
    for name in (splits or SPLIT_NAMES):
        p = base / LABELS_SUBDIR / f"{name}.parquet"
        if p.exists():
            out[name] = pd.read_parquet(p)
    return out


def load_identities(
    run_dir: Path, splits_dir: str = "sample-profile/splits",
    splits: tuple[str, ...] | None = None,
) -> dict[str, pd.DataFrame]:
    """读取无标签 identity 面；final_test 也始终可用。"""
    base = _prescreen_run_dir(run_dir) / splits_dir / IDENTITY_SUBDIR
    out: dict[str, pd.DataFrame] = {}
    for name in (splits or SPLIT_NAMES):
        path = base / f"{name}.parquet"
        if path.exists():
            out[name] = pd.read_parquet(path)
    return out


def _pos_fallback(rp: pd.DataFrame, chunk: pd.DataFrame) -> pd.DataFrame | None:
    """按位置对齐的保底:行数一致才返回,否则 None(宁可缺失也不错行)。"""
    return rp if len(rp) == len(chunk) else None


def _align_reparse_to_chunk(rp: pd.DataFrame, chunk: pd.DataFrame,
                            spec: TaskSpec) -> pd.DataFrame | None:
    """把父视图的 reparse 面按**样本键**对齐到已重排的 chunk。

    v2.13.1:父 reparse 面是父视图的落盘序,而视图落盘前 chunk 会被
    `_sort_split_frame` 重排。洗牌键若退化成"行位置 permutation",
    视图与父视图在同一天内的落盘序就会不同(实测 final_test
    42685/42687 行错位,per-day 集合相同但组内序不同),而配对比较
    按行位置配对。这里按 `SAMPLE_KEY_COLUMN` 对齐 —— 它绑定 time+ID、
    两端都有、且按位置单义(time_column 在同一天内有上千行重复,不能
    当键)。
    """
    if SAMPLE_KEY_COLUMN in rp.columns and SAMPLE_KEY_COLUMN in chunk.columns:
        rk = rp[SAMPLE_KEY_COLUMN].astype(str)
        ck = chunk[SAMPLE_KEY_COLUMN].astype(str)
        if rk.is_unique and ck.is_unique:
            aligned = rp.set_index(rk)
            if not ck[~ck.isin(aligned.index)].any():
                return aligned.reindex(ck).reset_index(drop=True)
    return _pos_fallback(rp, chunk)


def _load_parent_reparse(run_dir: Path, parent_dir: Path, name: str,
                         spec: TaskSpec) -> pd.DataFrame | None:
    """取父视图某个 split 的 reparse 原文面(供视图落盘时拼回)。

    v2.13.1:视图脚本按契约消费 reparse 源后会把原文从输出里剔除,于是
    `_split_face_frames` 的"声明了 reparse 却不在数据中"硬报错会把任何
    真实消费 reparse 的视图挡在门外。这里按父视图 + split 名读回原文,
    由调用方按样本键对齐后拼进 chunk。
    """
    if not spec.reparse_columns:
        return None
    rel = parent_dir.relative_to(Path(run_dir)) / REPARSE_SUBDIR / f"{name}.parquet"
    p = Path(run_dir) / rel
    if not p.exists():
        return None
    return pd.read_parquet(p)


def reparse_frames(run_dir: Path, splits_dir: str = "sample-profile/splits",
                   splits: tuple[str, ...] | None = None,
                   columns: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """读取解析源(splits-reparse/):reparse 原列 + 对齐键。

    解析类 transform 的唯一合法入口(特征面物理没有这些列);
    产物必须带解析前缀(seq_* 等),行序与特征面/标签面严格一致。
    """
    base = _prescreen_run_dir(run_dir) / splits_dir
    out: dict[str, pd.DataFrame] = {}
    for name in (splits or SPLIT_NAMES):
        p = base / REPARSE_SUBDIR / f"{name}.parquet"
        if not p.exists():
            continue
        frame = pd.read_parquet(p)
        if columns:
            missing = [c for c in columns if c not in frame.columns]
            if missing:
                raise KeyError(
                    f"reparse 源 {name} 不含列 {missing}"
                    f"(可用: {list(frame.columns)})")
            frame = frame[columns]
        out[name] = frame
    return out


# ---------------------------------------------------------------------------
# split protocol(§12.1)— Immutable
# ---------------------------------------------------------------------------

def _parse_date_token(token: str, *, is_end: bool) -> tuple[pd.Timestamp, str]:
    """把窗口端点归一为绝对日期(闭区间语义),返回 (日期, 粒度)。

    粒度 `"month"`(端点写作 `YYYY-MM` / `YYYYMM`)时,`is_end=True`
    取该月最后一天;粒度 `"day"` 时端点即当天。无法识别为日期则抛
    ValueError,由调用方降级为前缀匹配。

    `YYYYMMDD` / `YYYYMM` 紧凑格式与短横线格式等价——load_data() 里
    的 `str.replace` 归一化在按行混排(部分行带短横线、部分不带)时
    整体失败并保留原样,这种残余紧凑格式不应让整个 split 报错。
    """
    text = token.strip()
    for fmt, gran in (("%Y-%m-%d", "day"), ("%Y%m%d", "day"),
                      ("%Y-%m", "month"), ("%Y%m", "month")):
        try:
            ts = pd.to_datetime(text, format=fmt, errors="raise")
        except (ValueError, TypeError):
            continue
        if gran == "month" and is_end:
            ts = ts + pd.offsets.MonthEnd(0)
        return ts, gran
    raise ValueError(f"无法解析的时间窗端点: {token!r}")


def _match_time_window(values: pd.Series, window: str) -> np.ndarray:
    """时间窗匹配:支持 `YYYY-MM`、`YYYY-MM~YYYY-MM`、`YYYY-MM-DD`。

    区间语义(v19.1):

    - `A~B` 为**闭区间**:按端点自身粒度,`A` 从首个单位起算、`B`
      到最后一个单位为止(`YYYY-MM-DD` 含当天,`YYYY-MM` 含整月)。
      旧实现用 `t < end` 比较,把 `B` 自身那一格静默排除在外——实测
      `train: 2026-01-11~2026-04-26` 丢掉 2026-04-26 全天,又把
      `validation: 2026-05-04~2026-05-20` 末日的 11,135 行放回 train
      (端点日恰好是别的窗口边界时,还会伪装成"窗口重叠")。
    - 单端点(`YYYY-MM`,无 `~`):**含整月**(前缀匹配,与旧行为一致;
      `validation: 2026-05` 覆盖 5 月全部日期,而非仅 05-01)。

    v19.2:端点不可解析时**不再静默降级为前缀匹配**。降级会让一个
    被注解污染的窗口(如 `2026-01~2026-04(20260111~...,16 期,2,461,339 行)`)
    只匹配到某几个日期前缀、其余行落进 train 兜底——split 构成被
    静默改写而无人知晓(实测:train 少 4 个快照日,validation 少 4 个,
    shadow/final_test 整体平移一格)。现在由 `_parse_date_token` 抛出的
    ValueError 直接冒泡到 spec 校验层(见 `spec.validate_split_window_syntax`),
    在 profile 之前就报错。
    """
    text = values.astype(str)
    if "~" in window:
        start, end = (w.strip() for w in window.split("~", 1))
        start_lo, start_gran = _parse_date_token(start, is_end=False)
        end_hi, end_gran = _parse_date_token(end, is_end=True)
        if start_gran == "month" and end_gran == "month":
            # 两端都是"月"粒度:按月份前缀比较,让该月**全部日期**命中
            # (直接拿 'YYYY-MM' 与整行做字符串比较会把 '2026-06-30'
            # 判为大于 '2026-06',等于又把末日排除了)。
            return ((text.str[:7] >= start_lo.strftime("%Y-%m"))
                    & (text.str[:7] <= end_hi.strftime("%Y-%m"))).values
        return ((text >= start_lo.strftime("%Y-%m-%d"))
                & (text <= end_hi.strftime("%Y-%m-%d"))).values
    return text.str.startswith(window.strip()).values


def _month_name(text: str) -> str:
    """归一到 'YYYY-MM'(供互斥优先级排序)。"""
    return text[:7]


def split_data(df: pd.DataFrame, spec: TaskSpec) -> dict[str, pd.Index]:
    """按 TaskSpec.split_policy 计算每个 split 的行索引。

    返回 {split_name: Index};不复制数据,只产生归属,保证与
    SampleProfile / Trial 使用完全一致的数据面。
    """
    policy = spec.data.get("split_policy") or {}
    stype = policy["type"]
    n = len(df)

    if stype == "time_based":
        tcol = spec.time_column
        if not tcol:
            raise ValueError("time_based split 需要 time.column")
        if tcol not in df.columns:
            raise ValueError(f"时间列不存在: {tcol}")
        tval = df[tcol]
        if np.issubdtype(tval.dtype, np.datetime64):
            tval = tval.dt.strftime("%Y-%m-%d")
        # [FIX-2026-09-01] 时间格式显式报错:原样字符串若为 YYYYMMDD
        # 8 位数字(无分隔符),与窗口的 YYYY-MM/YYYY-MM-DD 匹配对不上,
        # 旧行为是"所有行落窗口之外 → 静默全归 train",验证面缺失、
        # 单视图跑完才发现。实测事故:req-t7-drawdown 30 万行 pday=
        # '20260220' 全进 train,validation/shadow/final_test 恒空。
        # 现在直接抛错并给出归一化指引,不让错误口径静默跑下去。
        if isinstance(tval.dtype, object) or str(tval.dtype) == "object":
            _s = tval.astype(str).str.strip()
            _compact = _s.str.fullmatch(r"\d{8}")
            if _compact.sum() > 0 and _compact.sum() == len(_s):
                raise ValueError(
                    "time_based split:时间列是 YYYYMMDD 8 位紧凑格式,"
                    "与 split 窗口的 YYYY-MM/YYYY-MM-DD 匹配不兼容。"
                    "请在 data-loader.load_data() 里归一化为 ISO 格式"
                    "(如 df['pday']=df['pday'].astype(str).str.replace("
                    "r'^(\\d{4})(\\d{2})(\\d{2})$', r'\\1-\\2-\\3'))")
        masks = {}
        for name in ("train", "validation", "shadow_test", "final_test"):
            window = policy.get(name)
            if not window:
                masks[name] = np.zeros(n, dtype=bool)
            else:
                masks[name] = _match_time_window(tval, window)
        # 窗口重叠冲突:按 train > validation > shadow_test > final_test
        # 优先级互斥分配(§12.1 split 必须无重叠)
        priority = ["train", "validation", "shadow_test", "final_test"]
        for i, hi in enumerate(priority):
            for lo in priority[i + 1:]:
                overlap = masks[hi] & masks[lo]
                if overlap.any():
                    logger.warning("split 窗口重叠:%s 与 %s 交叠 %d 行,"
                                   "按 %s 优先保留", hi, lo, int(overlap.sum()), hi)
                    masks[lo] &= ~overlap
        assigned = np.logical_or.reduce([m for m in masks.values()])
        if not assigned.any():
            # v19.2:零命中 = 窗口语法必然有误(或被注解污染后解析落空)。
            # 若放任兜底,全部数据会被塞进 train、其余 split 静默为空——
            # 口径错误伪装成"split 构成正常"(实测 train 少 4 个快照日、
            # validation 少 4 个、shadow/final_test 整体平移一格)。
            # spec 侧 validate_split_policy_windows 已做前置拦截,这里
            # 是数据面的最后一道保险(绕过 validate 直调 profile 的场景)。
            raise ValueError(
                "time_based split:四个窗口没有任何一行命中(0 命中)。"
                "窗口语法可能有误(注解污染/格式不匹配),请在 "
                "task-spec.yaml 的 split_policy 里使用纯区间"
                f"(如 `2026-01~2026-04`),注解放 YAML 注释;当前: {policy!r}")
        if not assigned.all():
            # 部分命中:窗口只覆盖数据的一部分,其余行归 train(既有语义,
            # 保留但显式告警)。**此路径与"零命中"必须可区分**——前者是
            # 正常兜底,后者是口径错误。
            unassigned = int((~assigned).sum())
            logger.warning("split 归属: %d 行落在所有窗口之外,归入 train"
                           "(命中 %d 行)", unassigned, int(assigned.sum()))
            masks["train"] |= ~assigned
        return {k: df.index[v] for k, v in masks.items() if v.any()}

    if stype == "random":
        ratios = policy.get("ratios") or {
            "train": 0.6, "validation": 0.15, "shadow_test": 0.1, "final_test": 0.15
        }
        rng = np.random.default_rng(policy.get("seed", 20260814))
        perm = rng.permutation(n)
        cuts, acc = {}, 0
        names = ["train", "validation", "shadow_test", "final_test"]
        fracs = np.array([ratios.get(k, 0.0) for k in names], dtype=float)
        fracs = fracs / fracs.sum()
        bounds = np.cumsum(fracs) * n
        for i, name in enumerate(names):
            hi = int(bounds[i]) if i < len(names) - 1 else n
            cuts[name] = df.index[perm[acc:hi]]
            acc = hi
        return {k: v for k, v in cuts.items() if len(v)}

    if stype == "group_based":
        gcol = (policy.get("group_column")
                or (spec.data.get("entity") or {}).get("id_column"))
        if not gcol or gcol not in df.columns:
            raise ValueError("group_based split 需要 split_policy.group_column")
        rng = np.random.default_rng(policy.get("seed", 20260814))
        groups = pd.unique(df[gcol])
        rng.shuffle(groups)
        ratios = policy.get("ratios") or {
            "train": 0.6, "validation": 0.15, "shadow_test": 0.1, "final_test": 0.15
        }
        names = ["train", "validation", "shadow_test", "final_test"]
        out: dict[str, pd.Index] = {}
        group_df = pd.Series(groups)
        acc = 0
        total = len(groups)
        for i, name in enumerate(names):
            frac = ratios.get(name, 0.0)
            hi = int(total * np.cumsum([ratios.get(k, 0.0) for k in names])[i]) \
                if i < 3 else total
            sel = set(group_df.iloc[acc:hi])
            out[name] = df.index[df[gcol].isin(sel)]
            acc = hi
        return {k: v for k, v in out.items() if len(v)}

    raise ValueError(f"未知 split 类型: {stype}")


def _sort_split_frame(frame: pd.DataFrame, spec: TaskSpec) -> \
        tuple[pd.DataFrame, str]:
    """split 落盘前按 time_column 升序排序(稳定;P0-1 修复),
    天内按标签洗牌(v17.1)。

    历史事故(seq-v3-3):落盘保留 loader 原始行序,train 前缀 63,854 行
    标签有序——capacity.py 用 ``X_tr.iloc[:k]`` 取前缀、hpo 的
    inner_split 取尾部、halving quick 轮取前缀,三处机制全部踩中
    "前缀标签分布 ≠ 全体"的坑,产出 AUC=0.5 的伪学习曲线,误导 45%
    算力投向。落盘前排序一次,三个机制同时修好。

    v17.1(第二层事故,drawdown-t7 实测):时间排序本身不安全——
    上游 loader 按天落表时天内行序天然按 label 聚集(实测单日
    runs=['1'*4151, '0'*8238],validation 前缀 73%~100% 正样本而
    全体 33%)。merge sort 保持天内原序,于是 "sorted_by:pday" 的
    前缀/尾部切片仍然单类化,manifest 的 prefix_rate_rel_dev 高达
    2.07,v12 防线(ordering 判定)对这种数据形同虚设。修复:排序
    key 升级为 (time, within-day-shuffle)——洗牌 seed 由行数+固定
    盐派生(确定性,同数据同序);天内洗牌不打乱跨天时间语义
    (天界仍是硬边界),但让任意前缀/尾部切片的标签分布与全体
    一致,capacity 样本量曲线、hpo inner-split、smoke/quick 轮
    的采样同时回归无偏。

    无时间列(或不支持排序)时返回原帧并记录 ordering 状态,调用方
    把它写进 manifest,前缀采样类机制据此降级(分层随机)。

    v19(req-t7-drawdown 实测事故回写):v17.1 的洗牌 key 是**输入帧
    位置**的 permutation(seed 只含行数)——write_view_splits 的
    build_view(df) 输入行序与 v1 不同时,同一天内两视图的落盘行序
    不同(实测 v1 vs v-rec validation 44638 行中同位置用户重合率
    0.0001),而配对 bootstrap 按行位置配对 → 跨视图 delta 伪影
    (trial-009 实测 +0.2844,CI [+0.277,+0.292])。修复:洗牌 key
    改**数据派生**——sha1(时间列值 + ID 列值) 的十六进制序,行序
    成为行内容的函数(与输入帧序无关),任何视图对同一行集落盘出
    函数式一致的行序;前缀采样无偏性不变(天内仍是均匀散布)。
    ID 列缺失时回退 v17.1 行为(行数 seed)并保留 ordering 标注。
    """
    tcol = spec.time_column
    if not tcol or tcol not in frame.columns or len(frame) == 0:
        return frame, "original_rows_no_time_column"
    try:
        s = frame[tcol]
        if np.issubdtype(s.dtype, np.datetime64):
            tval = s
        else:
            tval = s.astype(str)
        # v19:天内洗牌 key 改数据派生(确定性:同内容同序,与输入
        # 帧序无关)——跨视图行序函数式一致,配对比较不再错位
        icol = getattr(spec, "id_column", None)
        if icol and icol in frame.columns:
            _key = (tval.astype(str).str.cat(
                frame[icol].astype(str), sep="\x1f"))
            _shuf = pd.util.hash_array(
                _key.to_numpy(dtype=object, copy=False))
        else:
            # 无 ID 列:回退 v17.1 行为(行位置 permutation),ordering
            # 标注照旧,跨视图防线由 runner/evaluator 的 key-join 兜底
            rng = np.random.default_rng(20260814 + len(frame))
            _shuf = pd.Series(rng.permutation(len(frame)))
        out = frame.assign(
            __tval=tval.values, __shuf=np.asarray(_shuf)).sort_values(
            ["__tval", "__shuf"], kind="mergesort").drop(
            columns=["__tval", "__shuf"])
        return out.reset_index(drop=True), f"sorted_by:{tcol}"
    except Exception:  # noqa: BLE001  排序失败保底:保留原序并标注
        return frame, "sort_failed_fallback_original"
def _prefix_label_check(frame: pd.DataFrame, spec: TaskSpec,
                        ordering: str) -> dict:
    """前缀标签分布断言(P0-1):落盘 manifest 记录前 12.5% 行的标签
    统计,供 capacity/hpo/halving 的前缀采样类机制核对是否与全体同
    分布;二分类时直接给出前缀正样本率与全体的相对偏差。"""
    n = len(frame)
    # v21:目标列在四平面布局下位于 labels/(特征面不含目标列);_split_face_frames
    # 内部已先切出 labels 再调用本函数,故 frame 为 labels 面时含目标列。
    if n == 0 or spec.target_name not in frame.columns:
        return {"ordering": ordering}
    k = max(1, int(n * 0.125))
    out: dict = {"ordering": ordering, "prefix_rows": int(k)}
    y = frame[spec.target_name]
    if spec.model_type in ("binary_classification", "uplift"):
        full = float((y == 1).mean())
        pref = float((y.iloc[:k] == 1).mean())
        out["prefix_positive_rate"] = pref
        out["full_positive_rate"] = full
        if full > 0:
            out["prefix_rate_rel_dev"] = round(
                abs(pref - full) / full, 6)
    else:
        out["prefix_target_mean"] = float(y.iloc[:k].mean())
        out["full_target_mean"] = float(y.mean())
    return out


class _HashingWriter:
    """写盘时顺带累计 sha256 的文件对象(供 parquet 序列化器写入)。

    ``pyarrow`` / ``pandas.to_parquet`` 接受 file-like 对象;这里把每次
    ``write()`` 的字节同时喂给摘要,落盘结束时直接拿 ``hexdigest``,
    省掉"写完再 read_bytes 回读"的那一遍 IO。

    实测(req-t7 400w,该 NFS 单流 ~90 MB/s):

    | 产物 | 大小 | 回读 | 写入 |
    |---|---|---|---|
    | train 特征面 | 4.39 GB | 9.5 s | 118.5 s |
    | reparse/train | 8.78 GB | 13.9 s | 150.9 s |

    回读约占写盘耗时的 8%,且是完全的额外 IO;流式摘要把这个成本降到 0。
    产物字节与不经包装的写入**完全一致**(已逐字节比对),因此 manifest
    里的 ``sha256`` 与历史 run 同口径。
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._h = hashlib.sha256()
        self._f = open(self._path, "wb")
        self.name = str(self._path)

    # -- 序列化器需要的文件协议面 --
    def write(self, data) -> int:
        self._h.update(data)
        return self._f.write(data)

    def flush(self) -> None:
        self._f.flush()

    def tell(self) -> int:
        return self._f.tell()

    def seek(self, *args, **kwargs):
        return self._f.seek(*args, **kwargs)

    def truncate(self, *args):
        return self._f.truncate(*args)

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return True

    @property
    def closed(self) -> bool:
        return self._f.closed

    def close(self) -> None:
        if self._f.closed:
            return
        self._f.flush()
        os.fsync(self._f.fileno())
        self._f.close()

    @property
    def hexdigest(self) -> str:
        return self._h.hexdigest()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _write_parquet_hashed(frame: pd.DataFrame, path: Path) -> str:
    """落盘 parquet 并返回 sha256 前 16 位(与历史 manifest 同口径)。

    失败时删除半成品,避免下游读到截断的 parquet(旧实现
    ``path.read_bytes()`` 失败会直接抛错,但那时文件已经写完;这里在
    写入中途失败必须清理)。
    """
    path = Path(path)
    writer = _HashingWriter(path)
    try:
        frame.to_parquet(writer, index=False)
    except BaseException:
        writer.close()
        path.unlink(missing_ok=True)
        raise
    finally:
        writer.close()
    return writer.hexdigest[:16]


def write_splits(
    df: pd.DataFrame,
    spec: TaskSpec,
    out_dir: Path,
    split_index: dict[str, pd.Index] | None = None,
    *,
    workers: int | None = None,
) -> dict:
    """把切分结果落盘到 out_dir(独立目录,如 sample-profile/splits/)。

    v12(P0-1):每个 split 落盘前按 time_column 排序 + 前缀标签分布
    断言,manifest 记录 ordering;前缀采样类机制(capacity/hpo/
    halving)从"假设有序"变为"读 manifest 确认"。

    v13/v21(四产物物理隔离):每个 split 落盘四份——
    - ``{name}.parquet``        特征面(banned/reparse/目标/键 全部不在);
    - ``labels/{name}.parquet`` 目标 + treatment + 对齐键;
    - ``reparse/{name}.parquet``reparse 原列 + 对齐键(无则整目录跳过)。
    - ``identity/{name}.parquet``时间坐标 + 哈希样本键(final 也存在)。
    banned 列不写入任何产物;落盘后 assert_face_clean 断言。

    v21:``final_test`` 标签不再物化到搜索工作区。final 特征面仍可在
    搜索冻结后的重训阶段打分，但目标/treatment 仅由 evaluate-final
    在 gate acquire 之后从原始 data-loader 临时加载，且不落盘。

    v2.17.4(split 级并发写):各 split 彼此独立(各自的排序、切面、
    写盘、manifest 条目),原先却是一个 for 循环串行落盘。实测 req-t7
    400w×1493:train 特征面 4.39GB / reparse 8.78GB,其余三个 split
    各约 0.8GB;串行时单流稳定在 85~90 MB/s,而该 NFS 上 4 条并发流
    可跑到 340 MB/s、8 条 673 MB/s —— 瓶颈在单流而非介质。改成
    split 级线程池后,小 split 的写盘完全被 train 的大写盘吸收:

    | 场景 | 串行 | 并发 |
    |---|---|---|
    | 4×120k 行 split(同尺寸) | 28.0 s | 7.9 s(3.55×) |
    | train 特征面 + reparse(真实尺寸) | 269.4 s | 156.1 s(1.73×) |

    ``workers=None`` 时按 (a) 环境变量 ``MODELEVO_SPLIT_WRITE_WORKERS``、
    (b) 可用内存自动降级依次判定;显式传 1 可完全回到旧的串行行为。
    内存守卫是必须的:每个并发 split 会同时持有"排序后的 chunk + 特征面
    副本",req-t7 的 train 单 split 峰值约 32GB,4 路并发理论上需要 ~50GB;
    守卫按 psutil 可用内存估算,不足时降为串行而不是赌 OOM。

    manifest 的 ``splits`` 顺序严格保持 ``split_index`` 的插入序(与并发
    无关),因此同一份输入的 manifest 与串行路径逐字节一致。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LABELS_SUBDIR).mkdir(exist_ok=True)
    (out_dir / IDENTITY_SUBDIR).mkdir(exist_ok=True)
    # 升级旧 run 时先移除曾经物化的盲测标签；不能让一次重跑 profile
    # 因“本轮不写”而保留历史危险文件。
    (out_dir / LABELS_SUBDIR / "final_test.parquet").unlink(missing_ok=True)
    if spec.reparse_columns:
        (out_dir / REPARSE_SUBDIR).mkdir(exist_ok=True)
    if split_index is None:
        split_index = split_data(df, spec)

    manifest: dict = {
        "created_at": now_iso(),
        "split_policy": spec.data.get("split_policy"),
        "leak_manifest_hash": spec.leak_manifest_hash,
        "data_face_hash": spec.data_face_hash,
        "banned_columns": spec.banned_columns,
        "reparse_columns": spec.reparse_columns,
        "splits": {},
    }
    tcol = spec.treatment_column

    def _write_one(name: str, idx) -> tuple[str, dict, bool]:
        """落盘单个 split 的四个面,返回 (name, manifest 条目, 是否有 reparse)。"""
        path = out_dir / f"{name}.parquet"
        chunk, ordering = _sort_split_frame(df.loc[idx], spec)
        face, labels, rep = _split_face_frames(chunk, spec)
        # v2.17.4:写盘时同步算 sha256,免去落盘后的 read_bytes 回读
        # (train 特征面 4.39GB 回读 9.5s、reparse 8.78GB 13.9s)。
        face_sha = _write_parquet_hashed(face, path)
        identity = identity_frame(chunk, spec)
        identity.to_parquet(
            out_dir / IDENTITY_SUBDIR / f"{name}.parquet", index=False)
        # 所有可搜索 split 的标签面都携带稳定样本键。预测 artifact
        # 由 runner 用该键补全，父/子两侧分别读取后再做真实 key join。
        if name != "final_test":
            labels = labels.reset_index(drop=True)
            labels[SAMPLE_KEY_COLUMN] = sample_keys(chunk, spec).values
            labels.to_parquet(out_dir / LABELS_SUBDIR / f"{name}.parquet",
                              index=False)
        has_rep = rep.shape[1] > 0
        if has_rep:
            rep.to_parquet(out_dir / REPARSE_SUBDIR / f"{name}.parquet",
                           index=False)
        entry = {
            "rows": int(len(face)),
            "file": path.name,
            "sha256": face_sha,
            "ordering": ordering,
            "identity_file": f"{IDENTITY_SUBDIR}/{name}.parquet",
        }
        if name == "final_test":
            entry.update({
                "labels_materialized": False,
                "labels_access": "finalizer_only_after_gate",
            })
        else:
            entry["labels_file"] = f"{LABELS_SUBDIR}/{name}.parquet"
            entry.update(_prefix_label_check(labels, spec, ordering))
            pos = None
            if spec.model_type in ("binary_classification", "uplift"):
                y = labels[spec.target_name]
                pos = float((y == 1).mean()) if len(y) else None
            tgt = labels[spec.target_name]
            if pos is not None:
                entry["positive_rate"] = pos
            elif len(tgt):
                entry["target_mean"] = float(tgt.mean())
                entry["target_std"] = float(tgt.std())
            if tcol and tcol in labels.columns:
                sw = labels[tcol]
                entry["treatment_rate"] = float((sw == 1).mean()) \
                    if len(sw) else None
                t_mask = (sw == 1).values
                c_mask = (sw == 0).values
                tgt_idx = labels[spec.target_name]
                if t_mask.sum() and c_mask.sum():
                    entry["response_rate_treated"] = float(
                        (tgt_idx[t_mask] == 1).mean())
                    entry["response_rate_control"] = float(
                        (tgt_idx[c_mask] == 1).mean())
        return name, entry, has_rep

    items = list(split_index.items())
    n_workers = _split_write_workers(items, df, workers)
    if n_workers > 1 and len(items) > 1:
        logger.info("write_splits: %d 个 split 并发落盘(workers=%d)",
                    len(items), n_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(lambda it: _write_one(*it), items))
    else:
        results = [_write_one(*it) for it in items]

    has_reparse = False
    for name, entry, has_rep in results:
        manifest["splits"][name] = entry
        has_reparse = has_reparse or has_rep
    manifest["has_reparse_dir"] = has_reparse
    dump_json(out_dir / "manifest.json", manifest)
    # v13:后置断言——banned 列在任何产物中物理不存在
    _assert_view_clean(out_dir, spec)
    return manifest


def splits_reusable(out_dir: Path, spec: TaskSpec,
                    split_index: dict[str, pd.Index] | None = None,
                    *, df: pd.DataFrame | None = None) -> tuple[bool, str]:
    """判断 ``out_dir`` 里已有的四平面能否直接复用,免掉一次全量重写。

    [FIX-2026-10-29] 背景:profile 每次调用都无条件 ``write_splits``,重写
    train/validation/shadow/final 四份特征面 + labels + reparse。实测
    (run4 req-t7 run1, 4M 行 × 1493 列 + 2 列长文本序列)单次 402~406 s、
    20 GB;确认门裁决后按协议必须重跑同一 ``profile`` 命令,于是同一份
    **数据面完全没变**的四平面被原样重写一遍。

    复用条件(全部满足才复用):
    1. ``manifest.json`` 存在且可解析;
    2. ``data_face_hash`` 与当前 spec 一致(只含 banned/reparse ——
       ``leak_confirmation`` 那两档按协议不改数据面,因此裁决后重跑命中);
    3. ``split_policy`` 一致(窗口变更 = 样本面变更);
    4. 每个 split 的特征面文件存在,且逐个 sha256 与 manifest 记录一致
       (整文件摘要,不是抽样 —— 读到截断/被篡改的产物必须重写而非复用);
    5. labels/identity 文件齐全;``has_reparse_dir`` 与当前 reparse 声明一致;
    6. 传入 ``df`` 时额外校验每个 split 的行数与 manifest 一致。

    返回 ``(可复用, 原因)``;原因直接进 profile 日志,便于审计为何重写。
    """
    import hashlib

    out_dir = Path(out_dir)
    mpath = out_dir / "manifest.json"
    if not mpath.is_file():
        return False, "无 manifest.json"
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, f"manifest.json 不可解析: {exc}"
    recorded = manifest.get("data_face_hash")
    if not recorded:
        return False, "manifest 无 data_face_hash(旧产物)"
    if recorded != spec.data_face_hash:
        return False, (f"数据面指纹变化 {recorded} → {spec.data_face_hash}"
                       "(banned/reparse 已变更,必须重写)")
    if (manifest.get("split_policy") or {}) != (spec.data.get("split_policy") or {}):
        return False, "split_policy 变更"
    splits = manifest.get("splits") or {}
    if not splits:
        return False, "manifest 无 splits 条目"
    expected = set((split_index or {}).keys()) or set(splits.keys())
    if set(splits.keys()) != expected:
        return False, "split 集合与当前切分不一致"
    has_reparse = bool(spec.reparse_columns)
    if bool(manifest.get("has_reparse_dir")) != has_reparse:
        return False, "reparse 目录声明与当前 spec 不一致"
    for name, entry in splits.items():
        face = out_dir / str(entry.get("file") or f"{name}.parquet")
        if not face.is_file():
            return False, f"{name} 特征面缺失"
        h = hashlib.sha256()
        try:
            with open(face, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
        except OSError as exc:
            return False, f"{name} 特征面不可读: {exc}"
        if h.hexdigest()[:16] != entry.get("sha256"):
            return False, f"{name} 特征面 sha256 与 manifest 不符"
        idf = out_dir / str(entry.get("identity_file") or
                            f"{IDENTITY_SUBDIR}/{name}.parquet")
        if not idf.is_file():
            return False, f"{name} identity 面缺失"
        if name != "final_test":
            lab = out_dir / str(entry.get("labels_file") or
                                f"{LABELS_SUBDIR}/{name}.parquet")
            if not lab.is_file():
                return False, f"{name} labels 面缺失"
            if has_reparse and not (out_dir / REPARSE_SUBDIR
                                    / f"{name}.parquet").is_file():
                return False, f"{name} reparse 面缺失"
        if df is not None and split_index is not None:
            want = len(split_index.get(name, ()))
            if int(entry.get("rows", -1)) != want:
                return False, (f"{name} 行数 {entry.get('rows')} ≠ 当前切分 {want}")
    return True, "数据面未变(指纹一致、产物校验通过)"


def _split_write_workers(items: list, df: pd.DataFrame,
                         workers: int | None) -> int:
    """决定 split 级并发数(1 = 旧串行行为)。

    优先级:显式入参 > 环境变量 ``MODELEVO_SPLIT_WRITE_WORKERS`` > 自动。

    自动路径先按 split 数取上限 4,再按**最大** split 的峰值内存做守卫。
    每个并发 split 的峰值约为其 chunk 的两倍(排序结果 + 特征面副本);
    req-t7 400w 的 train 单 split 实测约 32GB,而 profile 进程本身还持有
    全量 df(100GB)与若干分析副本。预算按 ``最大 split 峰值 × 2`` 留
    余量,可用内存不足时直接降为串行 —— 宁可慢,也不赌 OOM(profile
    一旦 OOM,前面的 load_data 与全部分析都要重跑)。
    """
    n_splits = len(items)
    if n_splits <= 1:
        return 1
    if workers is None:
        env = os.environ.get("MODELEVO_SPLIT_WRITE_WORKERS")
        if env:
            try:
                workers = int(env)
            except (TypeError, ValueError):
                workers = None
    if workers is not None:
        return max(1, min(int(workers), n_splits))

    want = min(4, n_splits)
    if want <= 1:
        return 1
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:  # noqa: BLE001  探测失败:保守串行
        return 1
    per_row_bytes = 8 * max(int(df.shape[1]), 1)
    try:
        max_rows = max(int(len(idx)) for _, idx in items)
    except Exception:  # noqa: BLE001  索引形态异常:整帧行数当上界
        max_rows = int(df.shape[0])
    # 最大 split 的 chunk 峰值(浅口径 ×2:排序结果 + 特征面副本)
    peak_gb = max_rows * per_row_bytes / (1024 ** 3) * 2.0
    # 多个小 split 并发不会超过"最大 split + 其余之和";用 1.5 倍余量覆盖
    budget_gb = peak_gb * 1.5
    if avail_gb < budget_gb:
        return 1
    return want



def load_final_test_labels(spec: TaskSpec, run_dir: Path) -> pd.DataFrame:
    """在 FinalTestGate 之后临时加载 final_test 标签，永不落盘。

    该函数只允许交付命令调用。普通 Trial 的数据面不包含 final 标签；
    调用方必须先持有 ``FinalTestGate``，函数本身故意不提供任何缓存或
    旁路，也不会把标签写回 Trial 可见的数据产物。
    """
    df = load_dataframe(spec, run_dir)
    idx = split_data(df, spec).get("final_test")
    if idx is None or not len(idx):
        raise ValueError("原始样本中没有 final_test 行")
    chunk, _ = _sort_split_frame(df.loc[idx], spec)
    _, labels, _ = _split_face_frames(chunk, spec)
    labels = labels.reset_index(drop=True)
    labels[SAMPLE_KEY_COLUMN] = sample_keys(chunk, spec).values
    return labels


def load_split_frames(
    run_dir: Path, splits_dir: str = "sample-profile/splits"
) -> dict[str, pd.DataFrame]:
    """从落盘切分读取 DataFrame(Trial / Evaluator 使用)。

    v2.17.0:prescreen 沙箱内(MODELEVO_PRESCREEN_ROOT)按沙箱重映射。"""
    base = _prescreen_run_dir(run_dir) / splits_dir
    out = {}
    for name in SPLIT_NAMES:
        p = base / f"{name}.parquet"
        if p.exists():
            out[name] = pd.read_parquet(p)
    return out


SAMPLE_VIEW_ID = "sample"  # 特征探索抽样面(feature-exploration.md)


def write_sample_splits(
    df: pd.DataFrame,
    spec: TaskSpec,
    run_dir: Path,
    *,
    max_rows: int = 200_000,
    seed: int = 20260814,
    note: str = "",
) -> dict:
    """落盘特征探索抽样面(splits-sample/,协议 v10)。

    抽样规则(feature-exploration.md「20 万抽样契约」):
    - 各 split 按行数比例等距抽样(固定 seed),train/validation/
      shadow_test 的正样本率/时间分布与全量一致;
    - final_test **不抽样不落盘**——盲测面永远不动,sample 面上的
      trial 无法产出 final_test 分数(评估照常,不涉及盲测);
    - 行→split 归属沿用 v1:先 split_data(与全量同口径),再在各
      split 内部等距抽样,实体/时间覆盖均匀。

    幂等:重复调用覆盖落盘(manifest 记录抽样参数供复现)。

    v13/v21:与 write_splits 同构的四产物落盘
    (特征面/labels//reparse//identity/),
    banned 列物理不存在;抽样的行选择在四个物理面间完全一致(同一
    sampled_idx 切分,行序对齐)。
    """
    run_dir = Path(run_dir)
    if max_rows < 1000:
        raise ValueError(f"max_rows={max_rows} 过小(特征探索面需 ≥1000)")
    split_index = split_data(df, spec)
    total = int(sum(len(v) for v in split_index.values()))
    rng = np.random.default_rng(seed)
    out_dir = view_splits_dir(run_dir, SAMPLE_VIEW_ID)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LABELS_SUBDIR).mkdir(exist_ok=True)
    (out_dir / IDENTITY_SUBDIR).mkdir(exist_ok=True)
    (out_dir / LABELS_SUBDIR / "final_test.parquet").unlink(missing_ok=True)
    if spec.reparse_columns:
        (out_dir / REPARSE_SUBDIR).mkdir(exist_ok=True)

    manifest: dict = {
        "created_at": now_iso(),
        "view_id": SAMPLE_VIEW_ID,
        "parent_view": DEFAULT_VIEW_ID,
        "note": note or "特征探索抽样面(feature-exploration.md)",
        "sampled": True,
        "max_rows": int(max_rows),
        "seed": int(seed),
        "n_rows_original": total,
        "split_policy": spec.data.get("split_policy"),
        "leak_manifest_hash": spec.leak_manifest_hash,
        "banned_columns": spec.banned_columns,
        "reparse_columns": spec.reparse_columns,
        "splits": {},
    }
    # 各 split 按行数比例分摊 max_rows,保底 1 行;final_test 跳过
    alloc = {}
    eligible = [k for k in SPLIT_NAMES
                if k != "final_test" and k in split_index]
    eligible_rows = sum(len(split_index[k]) for k in eligible)
    if eligible_rows == 0:
        raise ValueError("无可抽样的 split(缺 train/validation)")
    for k in eligible:
        n_k = max(1, int(round(max_rows * len(split_index[k])
                               / eligible_rows)))
        alloc[k] = min(n_k, len(split_index[k]))
    has_reparse = False
    for name in SPLIT_NAMES:
        if name == "final_test":
            continue  # 盲测面不进抽样
        if name not in split_index:
            continue
        idx = split_index[name]
        n = alloc.get(name, len(idx))
        if n >= len(idx):
            sampled_idx = idx
        else:
            # 等距骨架 + 随机抖动:时间分布均匀,避免周期性对齐
            stride = len(idx) / n
            base = (np.arange(n) * stride).astype(np.int64)
            jitter = rng.integers(0, max(int(stride), 1), size=n)
            sampled_idx = idx[np.unique(np.clip(base + jitter, 0,
                                                len(idx) - 1))]
        chunk, ordering = _sort_split_frame(
            df.loc[sampled_idx], spec)
        chunk = chunk.reset_index(drop=True)
        face, labels, rep = _split_face_frames(chunk, spec)
        path = out_dir / f"{name}.parquet"
        face.to_parquet(path, index=False)
        identity_frame(chunk, spec).to_parquet(
            out_dir / IDENTITY_SUBDIR / f"{name}.parquet", index=False)
        labels = labels.reset_index(drop=True)
        labels[SAMPLE_KEY_COLUMN] = sample_keys(chunk, spec).values
        labels.to_parquet(out_dir / LABELS_SUBDIR / f"{name}.parquet",
                          index=False)
        if rep.shape[1] > 0:
            has_reparse = True
            rep.to_parquet(out_dir / REPARSE_SUBDIR / f"{name}.parquet",
                           index=False)
        entry = {
            "rows": int(len(face)),
            "rows_original": int(len(idx)),
            "file": path.name,
            "labels_file": f"{LABELS_SUBDIR}/{name}.parquet",
            "identity_file": f"{IDENTITY_SUBDIR}/{name}.parquet",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            **_prefix_label_check(labels, spec, ordering),
        }
        if spec.target_name in labels.columns and len(labels):
            y = labels[spec.target_name]
            if spec.model_type in ("binary_classification", "uplift"):
                entry["positive_rate"] = float((y == 1).mean())
            else:
                entry["target_mean"] = float(y.mean())
        manifest["splits"][name] = entry
    manifest["has_reparse_dir"] = has_reparse
    tcol = spec.treatment_column
    if tcol and tcol in df.columns:
        manifest["treatment_column"] = tcol
    # ---- G4 功效声明(v12.1):抽样面能可靠检出的最小效应量 ----
    # 依据(v3-4 实测):15 万行抽样面上 Δ+0.0005 的真实增量被噪声淹没,
    # 阴性结论被误采信。效应量可检性由 validation 行数与标签方差决定,
    # AUC 量纲下配对 bootstrap SE ≈ sqrt(AUC*(1-AUC)/n_eff) 的一阶近似;
    # 这里给出保守上界(plan 时用于判断"该面只能测 ≥X 的效应"):
    try:
        va_entry = manifest["splits"].get("validation") or {}
        n_va = int(va_entry.get("rows") or 0)
        pos = float(va_entry.get("positive_rate")
                    or va_entry.get("target_mean") or 0)
        if n_va > 30 and 0 < pos < 1:
            auc0 = 0.78   # 经验锚:行为二分类典型区间中部(仅做量纲参考)
            # AUC 的 SE 近似(Hanley-McNeil 简化):SE ≈ sqrt(Q1 - AUC^2)/n_pos + (Q2 - AUC^2)/n_neg 的开方;取工程简化式
            q1 = auc0 / (2 - auc0)
            q2 = 2 * auc0 * auc0 / (1 + auc0)
            se_va = ((q1 - auc0 ** 2) / max(n_va * pos, 1)
                     + (q2 - auc0 ** 2) / max(n_va * (1 - pos), 1)) ** 0.5
            # 配对比较的相关增益按 ~40% 折减(同模型跨特征面相关性高)
            se_delta = se_va * 0.6
            detectable = round(float(se_delta) * 3.92, 5)  # 95% CI 全宽
            manifest["power_statement"] = {
                "validation_rows": n_va,
                "auc_se_approx": round(float(se_va), 6),
                "delta_ci_full_width_approx": detectable,
                "min_reliably_detectable_delta": round(detectable, 5),
                "note": (
                    f"本抽样面({n_va} 行)配对差值 95%CI 全宽约 "
                    f"{detectable:.4f};联合增量 < 该值时阴性不可信,"
                    "方向裁决必须全量 confirmation(G1/G3/协议 v10 复核条款)。"
                    "近似基于 Hanley-McNeil 0.78 锚,经验折减系数 0.6。"),
            }
    except Exception as _e:  # noqa: BLE001  功效声明失败不阻塞落盘
        manifest["power_statement_error"] = str(_e)[:120]
    dump_json(out_dir / "manifest.json", manifest)
    # v13:后置断言——抽样面同样不得含 banned 列
    _assert_view_clean(out_dir, spec)
    return manifest


# ---------------------------------------------------------------------------
# 数据视图(data view,协议 v7)— per-view splits 版本化
# ---------------------------------------------------------------------------

DEFAULT_VIEW_ID = "v1"


def view_splits_dir(run_dir: Path, view_id: str = DEFAULT_VIEW_ID) -> Path:
    """视图 splits 目录:v1 保持旧路径(向后兼容),其余为 splits-{view}。"""
    run_dir = Path(run_dir)
    if view_id in ("", DEFAULT_VIEW_ID, "default"):
        return run_dir / "sample-profile" / "splits"
    return run_dir / "sample-profile" / f"splits-{view_id}"


def write_view_splits(
    df_view: pd.DataFrame,
    spec: TaskSpec,
    run_dir: Path,
    view_id: str,
    *,
    parent_splits_dir: Path | None = None,
    note: str = "",
    source_script: Path | None = None,
) -> dict:
    """把数据视图 df 沿用 v1 的行→split 归属落盘(协议 v7)。

    契约(防口径污染):
    - df_view 行数/行序必须与 v1 splits 的合并行集完全一致(视图只替换
      特征面,不改样本面);受保护样本键必须原样带回；可搜索 split 的
      目标/时间/ID/treatment 原样带回，final 的目标与 ID 是不可用占位;
    - split 归属不重新随机切分:直接按 v1 各 split 的行序切 df_view,
      保证逐 split 实体集合与 v1 完全一致,跨视图 trial 可比;
    - manifest 追加 view_id / parent_view / n_features / note,
      特征列集合 hash 落盘 feature_sha256(视图指纹);
    - source_script(v10):构建脚本路径与内容 sha256 记入 manifest——
      脚本本身按惯例存 run 目录 views/{id}.py(不随 harness 拷贝),
      manifest 记路径+指纹保证视图可追溯到构建代码、可从原始数据重推。

    v13/v21(四产物):视图同样落特征面/labels//reparse//identity/。
    视图输入 df_view 由 build_view 产出——标签与解析源不是视图要替换
    的东西；final 的目标占位不会写入 labels，identity 样本键原样继承；
    特征面按本函数切分落盘。构建脚本即使把 banned 列带进 df_view,
    落盘断言也会拒绝(对产物的校验,不约束脚本行为)。
    """
    run_dir = Path(run_dir)
    if view_id in ("", DEFAULT_VIEW_ID):
        raise ValueError(f"view_id={view_id!r} 保留给默认视图,不能覆写")
    out_dir = view_splits_dir(run_dir, view_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LABELS_SUBDIR).mkdir(exist_ok=True)
    (out_dir / IDENTITY_SUBDIR).mkdir(exist_ok=True)
    (out_dir / LABELS_SUBDIR / "final_test.parquet").unlink(missing_ok=True)
    if spec.reparse_columns:
        (out_dir / REPARSE_SUBDIR).mkdir(exist_ok=True)
    parent_dir = Path(parent_splits_dir) if parent_splits_dir \
        else view_splits_dir(run_dir)
    parent = load_split_frames(run_dir,
                               str(parent_dir.relative_to(run_dir)))
    if not parent:
        raise FileNotFoundError(f"父视图 splits 不存在: {parent_dir}")

    parent_rows = [parent[name] for name in SPLIT_NAMES if name in parent]
    parent_concat = pd.concat(
        parent_rows, ignore_index=True, copy=False)
    if len(df_view) != len(parent_concat):
        raise ValueError(
            f"数据视图行数 {len(df_view)} 与父视图 {len(parent_concat)} 不一致:"
            "视图只允许替换特征面,不允许改样本面(增删行/重排)")
    for col in spec.target_name, spec.id_column:
        if col is not None and col not in df_view.columns:
            raise ValueError(f"数据视图缺少必需列: {col}")
    # 行对齐:ID 序列必须与父视图按 SPLIT_NAMES 拼接序逐位置一致。
    # write-splits-view 传给 build_view 的输入就是该拼接序(见 CLI),
    # 脚本作者原样保留行序即可;ID 唯一时也按位置判(简单无歧义)。
    # v13:ID 在标签面,父视图拼接序的对齐键从 labels/ 取(特征面
    # 物理不含 id_column,旧守卫因此恒 False,视图行序错位漏检)
    parent_identity = load_identities(
        run_dir, str(parent_dir.relative_to(run_dir)))
    _key_parts = [parent_identity[n][SAMPLE_KEY_COLUMN]
                  for n in SPLIT_NAMES if n in parent_identity]
    if not _key_parts:
        raise ValueError("父视图缺 identity 样本键；请重新 profile")
    parent_keys = pd.concat(_key_parts, ignore_index=True).astype(str)
    if SAMPLE_KEY_COLUMN not in df_view.columns:
        raise ValueError(
            f"数据视图丢失受保护列 {SAMPLE_KEY_COLUMN}；build_view 必须"
            "保持行身份和行序")
    view_keys = df_view[SAMPLE_KEY_COLUMN].astype(str).reset_index(drop=True)
    if len(parent_keys) != len(view_keys) or not parent_keys.equals(view_keys):
        raise ValueError("数据视图样本键/行序与父视图不一致")
    # 哈希样本键比原始 ID 更严格：它绑定 time+ID，并且能区分同实体的
    # 多个时点。上面的逐位置等值检查已覆盖旧 ID 守卫，且避免把 final
    # 的原始业务 ID 暴露给 view 脚本。

    manifest: dict = {
        "created_at": now_iso(),
        "view_id": view_id,
        "parent_view": DEFAULT_VIEW_ID,
        "note": note,
        "split_policy": spec.data.get("split_policy"),
        "leak_manifest_hash": spec.leak_manifest_hash,
        "banned_columns": spec.banned_columns,
        "reparse_columns": spec.reparse_columns,
        "splits": {},
    }
    if source_script is not None:
        sp = Path(source_script)
        manifest["source_script"] = str(sp)
        try:
            manifest["source_script_sha256"] = hashlib.sha256(
                sp.read_bytes()).hexdigest()[:16]
        except OSError:
            logger.warning("视图构建脚本不可读,跳过指纹: %s", sp)
    feat_cols = resolve_feature_columns(df_view, spec)
    # v19.3 产物级泄漏断言(视图派生列不得携带保护列)：
    # 视图脚本是唯一"用户可写代码进数据面"的通路,而 recency/active/kwin
    # 三类模板都是"特征列 = 原列 × 系数/重聚合"的形态 —— 脚本若把
    # 目标/ID/时间列也纳入数值列集合,会产出 w30_<target> / act_<target>
    # 这类派生列,其 train 单变量 AUC≈1(实测 w30_label_req_t7 AUC=1.0),
    # 而 _split_face_frames 只按**精确列名**剔目标列,banned 断言也只查
    # banned 集合 —— 派生列从两道防线之间穿过去。
    # 判定口径:特征列名以 "_<保护列名>" 结尾(或与保护列同名)即视为
    # 该保护列的派生形态。放在产物侧而非脚本侧,理由与 assert_face_clean
    # 相同:对产物的校验比对脚本行为的约束可靠一个数量级。
    _guard_cols = [c for c in (spec.target_name, spec.id_column,
                               spec.time_column, spec.treatment_column)
                   if c]
    _leaked: list[tuple[str, str]] = []
    for _fc in feat_cols:
        for _gc in _guard_cols:
            if _fc == _gc or _fc.endswith("_" + _gc):
                _leaked.append((_fc, _gc))
                break
    if _leaked:
        raise ValueError(
            "数据视图派生列携带保护列(标签/ID/时间泄漏): "
            + "; ".join(f"{f} ← {g}" for f, g in _leaked[:8])
            + f"(共 {len(_leaked)} 列)。视图只允许对特征列做变换 —— "
            "检查构建脚本的数值列集合是否排除了 target/ID/time/"
            "treatment/banned/reparse(recency/active 模板由 CLI 注入的 "
            "_PROTECTED_COLS 承担该排除)。"
        )
    feat_hash = hashlib.sha256(
        "\n".join(sorted(feat_cols)).encode("utf-8")).hexdigest()[:16]
    manifest["n_features"] = len(feat_cols)
    manifest["feature_sha256"] = feat_hash

    has_reparse = False
    offset = 0
    for name in SPLIT_NAMES:
        if name not in parent:
            continue
        n = len(parent[name])
        chunk = df_view.iloc[offset:offset + n].reset_index(drop=True)
        offset += n
        # v12(P0-1)排序在视图路径**不重复施加**:df_view 由父视图各面按
        # SPLIT_NAMES 拼接而成,父视图落盘时已排序,且视图契约要求行序
        # 不变(上方逐位置样本键等值检查强制)。v2.13.1 实测事故:此处
        # 重复排序时,`_sort_split_frame` 的洗牌键在 final_test 上会退化
        # 成"行位置 permutation"(v21 四产物下 final_test 帧不含真实
        # 实体 ID,id_column 缺失),于是视图与父视图同一天内落盘序不同
        # —— 视图 final_test 全 42687 行错位;改用样本键当键又会让 v1
        # 自身行序改变(破坏既有 baseline 可比性)。故正确修法是视图
        # 路径**保持入序**。
        chunk = chunk.reset_index(drop=True)
        ordering = "sorted_by:%s" % spec.time_column if spec.time_column \
            else "original_rows_no_time_column"
        # v2.13.1:视图脚本按契约消费 reparse 源后会把**原文剔除**
        # (只允许替换特征列,原始长文本不得留在特征面)。此前这里直接
        # 调 _split_face_frames,撞上"声明了 reparse 却不在数据中"的
        # 硬报错 —— 任何真实消费 reparse 的视图都落不了盘。修法:从
        # **父视图**的 reparse/ 面拼回原文,让解析源面照常切出(此时
        # chunk 未重排,逐位置即逐样本)。
        _rp = _load_parent_reparse(run_dir, parent_dir, name, spec)
        if _rp is not None and len(_rp) == len(chunk):
            _rp = _align_reparse_to_chunk(_rp, chunk, spec)
            if _rp is not None:
                for _c in spec.reparse_columns:
                    if _c in _rp.columns and _c not in chunk.columns:
                        chunk[_c] = _rp[_c].values
        face, labels, rep = _split_face_frames(chunk, spec)
        path = out_dir / f"{name}.parquet"
        face.to_parquet(path, index=False)
        identity_frame(chunk, spec).to_parquet(
            out_dir / IDENTITY_SUBDIR / f"{name}.parquet", index=False)
        if name != "final_test":
            labels = labels.reset_index(drop=True)
            labels[SAMPLE_KEY_COLUMN] = sample_keys(chunk, spec).values
            labels.to_parquet(out_dir / LABELS_SUBDIR / f"{name}.parquet",
                              index=False)
        if rep.shape[1] > 0:
            has_reparse = True
            rep.to_parquet(out_dir / REPARSE_SUBDIR / f"{name}.parquet",
                           index=False)
        entry = {
            "rows": int(len(face)),
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            "ordering": ordering,
            "identity_file": f"{IDENTITY_SUBDIR}/{name}.parquet",
        }
        if name == "final_test":
            entry.update({"labels_materialized": False,
                          "labels_access": "finalizer_only_after_gate"})
        else:
            entry["labels_file"] = f"{LABELS_SUBDIR}/{name}.parquet"
            entry.update(_prefix_label_check(labels, spec, ordering))
        if name != "final_test" and spec.target_name in labels.columns:
            y = labels[spec.target_name]
            if spec.model_type in ("binary_classification", "uplift"):
                entry["positive_rate"] = float((y == 1).mean()) if len(y) else None
            else:
                entry["target_mean"] = float(y.mean()) if len(y) else None
        manifest["splits"][name] = entry
    manifest["has_reparse_dir"] = has_reparse
    dump_json(out_dir / "manifest.json", manifest)
    # v13:后置断言——视图产物不得含 banned 列(构建脚本绕不过)
    _assert_view_clean(out_dir, spec)
    return manifest


# ---------------------------------------------------------------------------
# v2.17.0(#12):流式数据视图 —— 按 split × 分块构建,内存有界
# ---------------------------------------------------------------------------
# 事故(run4 regression.2):write-splits-view 先把 v1 四个 split 的特征面 +
# 标签面 + identity + reparse 全量拼成 df_v1(4M 行 × 1513 列)再调 build_view,
# 3.5 min RSS 涨到 215 GB 仍在增长,人工 kill;第二个数据视图始终无法登记,
# stop_reason 一直挂着 single_data_view_insufficient。
#
# 流式契约(references/task-and-data.md「数据视图」节需同步):
# - build_view(df) 必须是**行内变换**:输出第 i 行只依赖输入第 i 行(与冻结
#   常量)。按块调用时,块内行集是同一 split 的连续行段,index 为该段在
#   v1 合并序中的全局行位置(与全量模式一致);
# - 自检:落盘前在每个 split 的首/末行组各取一小段,比对"单段独立调用"与
#   "跨 split 拼接调用"的输出(逐位相等);不等 = 块间存在依赖(如 recency
#   模板的逐 split 参考日)→ 抛 ViewNotRowLocalError,CLI 按内存预算回退
#   全量模式或拒绝;
# - 每块输入的列 dtype 统一到"全量拼接时 pandas 会给出的 dtype"(按 parquet
#   元数据的 null 计数推断 int→float64 / bool→object,跨 split 取公共类型),
#   因此行内变换在流式与全量两种模式下逐位一致;
# - 行数/行序/样本键与 v1 逐位一致(逐块校验),manifest 字段与全量模式相同,
#   构建方式另记 build-info.json(不改 manifest 字段集合)。
import os  # noqa: E402  (流式视图的原子落盘)

BUILD_INFO_FILE = "build-info.json"


class ViewNotRowLocalError(ValueError):
    """build_view 不是行内变换(块间存在依赖),流式模式不可用。"""


def _rss_gb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        return None
    return None


class _RowReader:
    """按需取 n 行的 parquet 顺序读取器(跨 batch/行组拼接,内存 ~ n 行)。"""

    def __init__(self, path: Path, columns: list[str] | None = None,
                 batch_rows: int = 65536):
        import pyarrow.parquet as pq

        self.path = Path(path)
        self.pf = pq.ParquetFile(self.path)
        self.num_rows = int(self.pf.metadata.num_rows)
        self._it = self.pf.iter_batches(batch_size=int(batch_rows),
                                        columns=columns)
        self._buf: list = []
        self._buf_rows = 0
        self.schema = self.pf.schema_arrow if columns is None else \
            __import__("pyarrow").schema(
                [self.pf.schema_arrow.field(c) for c in columns])

    def take(self, n: int):
        import pyarrow as pa

        while self._buf_rows < n:
            try:
                b = next(self._it)
            except StopIteration:
                break
            self._buf.append(b)
            self._buf_rows += b.num_rows
        if not self._buf:
            return pa.Table.from_batches([], schema=self.schema)
        tbl = pa.Table.from_batches(self._buf, schema=self._buf[0].schema)
        out, rest = tbl.slice(0, n), tbl.slice(n)
        self._buf = rest.to_batches() if rest.num_rows else []
        self._buf_rows = rest.num_rows
        return out


def _null_counts(pf, name: str) -> int | None:
    """parquet 元数据里某列的 null 总数;任一行组缺统计返回 None。"""
    md = pf.metadata
    try:
        j = pf.schema_arrow.get_field_index(name)
        leaf = [i for i in range(md.num_columns)
                if md.schema.column(i).path.split(".")[0] == name]
        if j < 0 or not leaf:
            return None
        total = 0
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(leaf[0]).statistics
            if st is None or not st.has_null_count:
                return None
            total += int(st.null_count)
        return total
    except Exception:  # noqa: BLE001
        return None


def _file_pandas_dtypes(path: Path) -> dict:
    """整文件 pd.read_parquet 时各列会得到的 pandas dtype(不读数据)。

    int 列含 null → float64;bool 含 null → object;dictionary → 读该单列
    取全量类别(单列,内存有界)。统计缺失时读单列数 null。
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    base = schema.empty_table().to_pandas().dtypes.to_dict()
    out: dict = {}
    for field in schema:
        name = field.name
        t = field.type
        dt = base.get(name)
        if pa.types.is_dictionary(t):
            col = pd.read_parquet(path, columns=[name])[name]
            out[name] = col.dtype
            continue
        if pa.types.is_integer(t) or pa.types.is_boolean(t):
            nulls = _null_counts(pf, name)
            if nulls is None:
                nulls = int(pq.read_table(path, columns=[name])
                            .column(0).null_count)
            if nulls > 0:
                dt = np.dtype("float64") if pa.types.is_integer(t) \
                    else np.dtype("object")
        out[name] = dt
    return out


def _common_dtype(dtypes: list):
    """多个 split 的同名列在 pd.concat 后的 dtype(pandas 公共类型规则)。"""
    uniq = []
    for d in dtypes:
        if d is not None and not any(str(d) == str(u) for u in uniq):
            uniq.append(d)
    if not uniq:
        return None
    if len(uniq) == 1:
        return uniq[0]
    try:
        from pandas.core.dtypes.cast import find_common_type

        return find_common_type(uniq)
    except Exception:  # noqa: BLE001
        return np.dtype("object")


def _cast_frame(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    for c in df.columns:
        want = dtypes.get(c)
        if want is None or str(df[c].dtype) == str(want):
            continue
        try:
            df[c] = df[c].astype(want)
        except Exception:  # noqa: BLE001  无法对齐时保持原 dtype
            pass
    return df


class _ChunkWriter:
    """分块写 parquet:首块定 schema,后块 cast 对齐;cast 失败时转分片模式,
    收尾按 unify_schemas 合并(块间 dtype 漂移也不丢数据、不爆内存)。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + f".tmp{os.getpid()}")
        self.writer = None
        self.schema = None
        self.parts: list[Path] = []
        self.rows = 0

    def write_table(self, tbl) -> None:
        import pyarrow.parquet as pq

        if self.schema is None:
            self.schema = tbl.schema
            self.writer = pq.ParquetWriter(self.tmp, self.schema)
        elif self.parts:
            p = self.tmp.with_name(self.tmp.name + f".part{len(self.parts)}")
            pq.write_table(tbl, p)
            self.parts.append(p)
            self.rows += tbl.num_rows
            return
        else:
            if tbl.schema.names != self.schema.names:
                if sorted(tbl.schema.names) != sorted(self.schema.names):
                    raise ValueError(
                        "视图输出列在块间不一致: "
                        f"{sorted(set(tbl.schema.names) ^ set(self.schema.names))[:6]}")
                tbl = tbl.select(self.schema.names)
            if not tbl.schema.equals(self.schema, check_metadata=False):
                try:
                    tbl = tbl.cast(self.schema)
                except Exception:  # noqa: BLE001  → 分片模式
                    self.writer.close()
                    self.writer = None
                    p0 = self.tmp.with_name(self.tmp.name + ".part0")
                    os.replace(self.tmp, p0)
                    self.parts.append(p0)
                    p = self.tmp.with_name(self.tmp.name + ".part1")
                    pq.write_table(tbl, p)
                    self.parts.append(p)
                    self.rows += tbl.num_rows
                    return
        self.writer.write_table(tbl)
        self.rows += tbl.num_rows

    def write_frame(self, df: pd.DataFrame) -> None:
        import pyarrow as pa

        self.write_table(pa.Table.from_pandas(df, preserve_index=False))

    def close(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.parts:
            schemas = [pq.read_schema(p) for p in self.parts]
            try:
                uni = pa.unify_schemas(schemas, promote_options="permissive")
            except TypeError:  # 旧 pyarrow 无 promote_options
                uni = pa.unify_schemas(schemas)
            uni = pa.schema([uni.field(n) for n in schemas[0].names],
                            metadata=schemas[0].metadata)
            with pq.ParquetWriter(self.tmp, uni) as w:
                for p in self.parts:
                    for b in pq.ParquetFile(p).iter_batches(batch_size=65536):
                        t = pa.Table.from_batches([b]).select(uni.names)
                        w.write_table(t.cast(uni))
            for p in self.parts:
                p.unlink(missing_ok=True)
            self.parts = []
        if self.schema is None:
            raise ValueError(f"分块写入器未收到任何数据: {self.path}")
        os.replace(self.tmp, self.path)


def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(8 << 20), b""):
            h.update(buf)
    return h.hexdigest()[:16]


def _view_dirs(run_dir: Path, view_id: str, spec: TaskSpec) -> Path:
    if view_id in ("", DEFAULT_VIEW_ID):
        raise ValueError(f"view_id={view_id!r} 保留给默认视图,不能覆写")
    out_dir = view_splits_dir(run_dir, view_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LABELS_SUBDIR).mkdir(exist_ok=True)
    (out_dir / IDENTITY_SUBDIR).mkdir(exist_ok=True)
    (out_dir / LABELS_SUBDIR / "final_test.parquet").unlink(missing_ok=True)
    if spec.reparse_columns:
        (out_dir / REPARSE_SUBDIR).mkdir(exist_ok=True)
    return out_dir


def _view_manifest_head(spec: TaskSpec, view_id: str, note: str,
                        source_script: Path | None) -> dict:
    manifest: dict = {
        "created_at": now_iso(),
        "view_id": view_id,
        "parent_view": DEFAULT_VIEW_ID,
        "note": note,
        "split_policy": spec.data.get("split_policy"),
        "leak_manifest_hash": spec.leak_manifest_hash,
        "banned_columns": spec.banned_columns,
        "reparse_columns": spec.reparse_columns,
        "splits": {},
    }
    if source_script is not None:
        sp = Path(source_script)
        manifest["source_script"] = str(sp)
        try:
            manifest["source_script_sha256"] = hashlib.sha256(
                sp.read_bytes()).hexdigest()[:16]
        except OSError:
            logger.warning("视图构建脚本不可读,跳过指纹: %s", sp)
    return manifest


def _guard_derived_columns(feat_cols: list[str], spec: TaskSpec) -> None:
    """与 write_view_splits 同口径的派生泄漏列断言(列名以 _<保护列> 结尾)。"""
    guard = [c for c in (spec.target_name, spec.id_column, spec.time_column,
                         spec.treatment_column) if c]
    leaked = []
    for fc in feat_cols:
        for gc in guard:
            if fc == gc or fc.endswith("_" + gc):
                leaked.append((fc, gc))
                break
    if leaked:
        raise ValueError(
            "数据视图派生列携带保护列(标签/ID/时间泄漏): "
            + "; ".join(f"{f} ← {g}" for f, g in leaked[:8])
            + f"(共 {len(leaked)} 列)。视图只允许对特征列做变换。")


def _view_split_entry(name: str, path: Path, labels: pd.DataFrame | None,
                      spec: TaskSpec, rows: int) -> dict:
    """与 write_view_splits 同字段的 split 条目。"""
    ordering = "sorted_by:%s" % spec.time_column if spec.time_column \
        else "original_rows_no_time_column"
    entry = {
        "rows": int(rows),
        "file": path.name,
        "sha256": _sha16(path),
        "ordering": ordering,
        "identity_file": f"{IDENTITY_SUBDIR}/{name}.parquet",
    }
    if name == "final_test":
        entry.update({"labels_materialized": False,
                      "labels_access": "finalizer_only_after_gate"})
    else:
        entry["labels_file"] = f"{LABELS_SUBDIR}/{name}.parquet"
        if labels is not None:
            entry.update(_prefix_label_check(labels, spec, ordering))
    if name != "final_test" and labels is not None \
            and spec.target_name in labels.columns:
        y = labels[spec.target_name]
        if spec.model_type in ("binary_classification", "uplift"):
            entry["positive_rate"] = float((y == 1).mean()) if len(y) else None
        else:
            entry["target_mean"] = float(y.mean()) if len(y) else None
    return entry


class _ParentPlan:
    """父视图(默认 v1)各 split 的四产物与"全量拼接 dtype"计划。"""

    def __init__(self, spec: TaskSpec, run_dir: Path,
                 parent_dir: Path | None = None):
        self.spec = spec
        self.run_dir = Path(run_dir)
        self.parent_dir = Path(parent_dir) if parent_dir else \
            view_splits_dir(self.run_dir)
        rel = str(self.parent_dir.relative_to(self.run_dir))
        self.names = [n for n in SPLIT_NAMES
                      if (self.parent_dir / f"{n}.parquet").exists()]
        if not self.names:
            raise FileNotFoundError(f"父视图 splits 不存在: {self.parent_dir}")
        self.labels = load_labels(self.run_dir, rel)
        self.identities = load_identities(self.run_dir, rel)
        import pyarrow.parquet as pq

        self.rows = {n: int(pq.ParquetFile(self.parent_dir / f"{n}.parquet")
                            .metadata.num_rows) for n in self.names}
        for n in self.names:
            if n not in self.identities:
                raise ValueError(
                    f"identity 面缺失 {IDENTITY_SUBDIR}/{n}.parquet;请重新 profile")
            if len(self.identities[n]) != self.rows[n]:
                raise ValueError(f"{n} 的特征面与 identity 面行数不一致")
            if n != "final_test" and n not in self.labels:
                raise ValueError(f"标签面缺失: {LABELS_SUBDIR}/{n}.parquet")
        self.offsets: dict[str, int] = {}
        acc = 0
        for n in self.names:
            self.offsets[n] = acc
            acc += self.rows[n]
        self.total_rows = acc

    def reparse_path(self, name: str) -> Path | None:
        if not self.spec.reparse_columns:
            return None
        p = self.parent_dir / REPARSE_SUBDIR / f"{name}.parquet"
        return p if p.exists() else None

    def label_part(self, name: str) -> pd.DataFrame:
        """可搜索 split = 真实标签面;final_test = 结构占位(与 CLI 同口径)。"""
        memo = self.__dict__.setdefault("_label_memo", {})
        if name not in memo:
            memo[name] = self._label_part(name)
        return memo[name]

    def _label_part(self, name: str) -> pd.DataFrame:
        spec = self.spec
        if name in self.labels:
            return self.labels[name].reset_index(drop=True)
        ident = self.identities[name].reset_index(drop=True)
        ref = next((self.labels[s] for s in SPLIT_NAMES if s in self.labels),
                   None)
        stub = pd.DataFrame(index=range(len(ident)))
        cols = list(ref.columns) if ref is not None else [
            c for c in (spec.target_name, spec.treatment_column,
                        spec.id_column) if c]
        keys = ident[SAMPLE_KEY_COLUMN].astype(str).values
        for c in cols:
            if c in (spec.target_name, spec.treatment_column):
                stub[c] = float("nan")
            elif c == spec.time_column:
                stub[c] = (ident[c].values if c in ident.columns else pd.NaT)
            else:
                stub[c] = keys
        return stub

    def unified_dtypes(self) -> dict:
        """各列在全量模式 df_v1(四 split 拼接 + 首次出现去重)中的 dtype。"""
        per_col: dict[str, list] = {}
        for n in self.names:
            parts: list[tuple[str, object]] = []
            for c, d in _file_pandas_dtypes(
                    self.parent_dir / f"{n}.parquet").items():
                parts.append((c, d))
            for c, d in self.label_part(n).dtypes.items():
                parts.append((c, d))
            for c, d in self.identities[n].dtypes.items():
                parts.append((c, d))
            rp = self.reparse_path(n)
            if rp is not None:
                for c, d in _file_pandas_dtypes(rp).items():
                    parts.append((c, d))
            seen = set()
            for c, d in parts:
                if c in seen:
                    continue
                seen.add(c)
                per_col.setdefault(c, []).append(d)
        n_splits = len(self.names)
        out = {}
        for c, ds in per_col.items():
            if len(ds) < n_splits:
                # 某 split 缺该列 → outer concat 以 NA 填充
                ds = list(ds) + [np.dtype("float64")
                                 if any(str(d).startswith(("int", "uint"))
                                        for d in ds)
                                 else ds[0]]
            out[c] = _common_dtype(ds)
        return out

    def assemble(self, name: str, face: pd.DataFrame, start: int,
                 stop: int, reparse: pd.DataFrame | None,
                 dtypes: dict) -> pd.DataFrame:
        """拼出 [start, stop) 行段的 build_view 输入(与 CLI 全量拼接同口径)。"""
        cols = [face.reset_index(drop=True),
                self.label_part(name).iloc[start:stop].reset_index(drop=True),
                self.identities[name].iloc[start:stop].reset_index(drop=True)]
        if reparse is not None:
            cols.append(reparse.reset_index(drop=True))
        df = pd.concat(cols, axis=1)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()
        df = _cast_frame(df, dtypes)
        off = self.offsets[name] + start
        df.index = pd.RangeIndex(off, off + len(df))
        return df

    def _file_row_bytes(self, path: Path) -> float:
        """单行内存宽度估计:max(parquet 未压缩字节/行, schema 定宽估计)。

        parquet 的 total_byte_size 是编码后(字典/RLE)大小,对稀疏宽表会把
        pandas 内存低估一个数量级(run4:4M×1513 列 float64 ≈ 48 GB/份,
        而未压缩字节只有 ~8 GB);schema 估计按 64 位数值 8 B、32 位 4 B、
        bool 1 B、字符串/对象 64 B 计。
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        md = pf.metadata
        enc = (sum(md.row_group(i).total_byte_size
                   for i in range(md.num_row_groups)) / md.num_rows) \
            if md.num_rows else 0.0
        width = 0
        for f in pf.schema_arrow:
            t = f.type
            if pa.types.is_boolean(t):
                width += 1
            elif pa.types.is_dictionary(t):
                width += 8
            else:
                try:
                    width += max(1, int(t.bit_width) // 8)
                except (ValueError, AttributeError, TypeError):
                    width += 64      # 变长(字符串/二进制/嵌套)
        return float(max(enc, width))

    def estimate_full_gb(self) -> float:
        """全量模式峰值内存粗估:行宽 × 总行数 × 4
        (各 split 拼接件 + df_v1 + build_view 输出副本 + 切面副本)。"""
        total = 0.0
        for n in self.names:
            w = self._file_row_bytes(self.parent_dir / f"{n}.parquet")
            rp = self.reparse_path(n)
            if rp is not None:
                w += self._file_row_bytes(rp)
            total += w * self.rows[n]
        return total * 4 / 1e9

    def bytes_per_row(self) -> float:
        best = 0.0
        for n in self.names:
            w = self._file_row_bytes(self.parent_dir / f"{n}.parquet")
            rp = self.reparse_path(n)
            if rp is not None:
                w += self._file_row_bytes(rp)
            best = max(best, w)
        return max(best, 64.0)


def _read_head(path: Path, n: int, *, last_group: bool = False,
               columns: list[str] | None = None) -> tuple[pd.DataFrame, int]:
    """文件首行组(或末行组)开头的 n 行 → (DataFrame, 起始行位置)。"""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    md = pf.metadata
    rg = md.num_row_groups - 1 if last_group else 0
    start = sum(md.row_group(i).num_rows for i in range(rg))
    it = pf.iter_batches(batch_size=max(1, int(n)), row_groups=[rg],
                         columns=columns)
    try:
        b = next(it)
    except StopIteration:
        return pd.DataFrame(), start
    return b.to_pandas(), start


def _read_rows(path: Path, start: int, stop: int) -> pd.DataFrame:
    """顺序跳读 [start, stop) 行(从包含 start 的行组起读,内存 ~ 一个 batch)。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    md = pf.metadata
    acc, rg0 = 0, 0
    for i in range(md.num_row_groups):
        n = md.row_group(i).num_rows
        if acc + n > start:
            rg0 = i
            break
        acc += n
    pos = acc
    out = []
    for b in pf.iter_batches(batch_size=65536,
                             row_groups=list(range(rg0, md.num_row_groups))):
        lo, hi = max(start, pos), min(stop, pos + b.num_rows)
        if hi > lo:
            out.append(b.slice(lo - pos, hi - lo))
        pos += b.num_rows
        if pos >= stop:
            break
    if not out:
        return pf.schema_arrow.empty_table().to_pandas()
    return pa.Table.from_batches(out).to_pandas()


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> str | None:
    """逐位比较;相等返回 None,否则返回差异说明。"""
    if list(a.columns) != list(b.columns):
        return (f"列不同: {sorted(set(a.columns) ^ set(b.columns))[:5]}"
                or "列序不同")
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=True, check_exact=True)
        return None
    except AssertionError as exc:
        return str(exc).splitlines()[0][:300] if str(exc) else "值不同"


def check_row_local(build_view, plan: _ParentPlan, dtypes: dict,
                    probe_rows: int = 128) -> dict:
    """行内变换自检:每个 split 首/末行组各取 probe_rows 行;
    (a) 单段独立调用 vs (b) 全部段拼接后调用,输出逐位相等才算行内。"""
    import pyarrow.parquet as pq

    segs = []
    for n in plan.names:
        p = plan.parent_dir / f"{n}.parquet"
        n_rg = pq.ParquetFile(p).metadata.num_row_groups
        for last in ([False, True] if n_rg > 1 else [False]):
            face, start = _read_head(p, probe_rows, last_group=last)
            if not len(face):
                continue
            stop = start + len(face)
            rp = plan.reparse_path(n)
            rep = None
            if rp is not None:
                rep_all, rstart = _read_head(rp, probe_rows, last_group=last)
                if rstart == start and len(rep_all) == len(face):
                    rep = rep_all
                else:  # 行组边界不同 → 顺序跳读到同一行位置(内存有界)
                    rep = _read_rows(rp, start, stop)
            segs.append((n, start, stop,
                         plan.assemble(n, face, start, stop, rep, dtypes)))
    if not segs:
        return {"passed": True, "segments": 0}
    full_in = pd.concat([s[3] for s in segs], axis=0)
    try:
        full_out = build_view(full_in.copy())
    except Exception as exc:  # noqa: BLE001
        raise ViewNotRowLocalError(f"自检:build_view 在拼接探针上失败: {exc}")
    if not isinstance(full_out, pd.DataFrame) or len(full_out) != len(full_in):
        raise ViewNotRowLocalError("自检:build_view 输出行数与输入不一致")
    pos = 0
    for n, start, stop, seg in segs:
        k = len(seg)
        try:
            alone = build_view(seg.copy())
        except Exception as exc:  # noqa: BLE001
            raise ViewNotRowLocalError(
                f"自检:build_view 在 {n}[{start}:{stop}] 单段上失败: {exc}")
        ref = full_out.iloc[pos:pos + k]
        pos += k
        diff = _frames_equal(alone, ref)
        if diff is not None:
            raise ViewNotRowLocalError(
                f"自检失败:{n}[{start}:{stop}] 单段调用与跨 split 拼接调用的"
                f"输出不一致({diff})—— build_view 不是行内变换(块间存在依赖,"
                "如逐 split 参考日/全局统计),不能流式构建")
    return {"passed": True, "segments": len(segs), "probe_rows": probe_rows}


def write_view_splits_streaming(
    build_view,
    spec: TaskSpec,
    run_dir: Path,
    view_id: str,
    *,
    chunk_rows: int | None = None,
    max_mem_gb: float | None = None,
    parent_splits_dir: Path | None = None,
    note: str = "",
    source_script: Path | None = None,
    self_check: bool = True,
) -> dict:
    """流式版 write-splits-view:按 split × 分块调 build_view 并追加落盘。

    产物与 write_view_splits(全量模式)逐位一致(行内变换前提下):同样的
    四产物、行序、样本键、manifest 字段;构建方式另记 build-info.json。
    max_mem_gb:按父视图未压缩行宽估算块大小(×6 安全系数),并在每块后
    检查 RSS,超过上限 85% 时块大小减半("超限前分块落盘")。
    """
    run_dir = Path(run_dir)
    plan = _ParentPlan(spec, run_dir, parent_splits_dir)
    dtypes = plan.unified_dtypes()
    check = check_row_local(build_view, plan, dtypes) if self_check \
        else {"passed": None, "skipped": True}
    bpr = plan.bytes_per_row()
    if chunk_rows is None:
        chunk_rows = 200_000
    if max_mem_gb:
        budget = float(max_mem_gb) * 1e9 * 0.8
        chunk_rows = min(int(chunk_rows), int(budget / (bpr * 6)))
    chunk_rows = max(1000, int(chunk_rows))
    out_dir = _view_dirs(run_dir, view_id, spec)
    manifest = _view_manifest_head(spec, view_id, note, source_script)
    has_reparse = False
    feat_cols_ref: list[str] | None = None
    peak_rss = _rss_gb() or 0.0
    n_chunks = 0
    shrinks = 0
    t0 = __import__("time").time()
    for name in plan.names:
        n_rows = plan.rows[name]
        face_rd = _RowReader(plan.parent_dir / f"{name}.parquet",
                             batch_rows=min(chunk_rows, 65536))
        rp_path = plan.reparse_path(name)
        rp_rd = _RowReader(rp_path, batch_rows=min(chunk_rows, 65536)) \
            if rp_path is not None else None
        w_face = _ChunkWriter(out_dir / f"{name}.parquet")
        w_ident = _ChunkWriter(out_dir / IDENTITY_SUBDIR / f"{name}.parquet")
        w_lab = _ChunkWriter(out_dir / LABELS_SUBDIR / f"{name}.parquet") \
            if name != "final_test" else None
        w_rep = None
        label_parts: list[pd.DataFrame] = []
        parent_keys = plan.identities[name][SAMPLE_KEY_COLUMN].astype(str) \
            .reset_index(drop=True)
        start = 0
        while start < n_rows:
            stop = min(n_rows, start + chunk_rows)
            face = face_rd.take(stop - start).to_pandas()
            rep = rp_rd.take(stop - start).to_pandas() if rp_rd else None
            chunk_in = plan.assemble(name, face, start, stop, rep, dtypes)
            del face
            out = build_view(chunk_in)
            if not isinstance(out, pd.DataFrame):
                raise TypeError(
                    f"build_view 必须返回 DataFrame,得到 {type(out)}")
            if len(out) != len(chunk_in):
                raise ValueError(
                    f"数据视图 {name}[{start}:{stop}] 行数 {len(out)} != "
                    f"{len(chunk_in)}:视图只允许替换特征面,不允许增删行")
            if SAMPLE_KEY_COLUMN not in out.columns:
                raise ValueError(
                    f"数据视图丢失受保护列 {SAMPLE_KEY_COLUMN};build_view 必须"
                    "保持行身份和行序")
            vk = out[SAMPLE_KEY_COLUMN].astype(str).reset_index(drop=True)
            if not vk.equals(parent_keys.iloc[start:stop]
                             .reset_index(drop=True)):
                raise ValueError("数据视图样本键/行序与父视图不一致")
            for col in (spec.target_name, spec.id_column):
                if col is not None and col not in out.columns:
                    raise ValueError(f"数据视图缺少必需列: {col}")
            feat_cols = resolve_feature_columns(out, spec)
            if feat_cols_ref is None:
                _guard_derived_columns(feat_cols, spec)
                feat_cols_ref = feat_cols
            elif feat_cols != feat_cols_ref:
                raise ValueError("数据视图特征列在块间不一致(build_view 输出"
                                 "列集合必须与数据内容无关)")
            chunk = out.reset_index(drop=True)
            del out, chunk_in
            if spec.reparse_columns and rep is not None:
                for _c in spec.reparse_columns:
                    if _c in rep.columns and _c not in chunk.columns:
                        chunk[_c] = rep[_c].values
            f_face, f_lab, f_rep = _split_face_frames(chunk, spec)
            w_face.write_frame(f_face)
            w_ident.write_frame(identity_frame(chunk, spec))
            if w_lab is not None:
                f_lab = f_lab.reset_index(drop=True)
                f_lab[SAMPLE_KEY_COLUMN] = sample_keys(chunk, spec).values
                w_lab.write_frame(f_lab)
                label_parts.append(f_lab)
            if f_rep.shape[1] > 0:
                has_reparse = True
                if w_rep is None:
                    w_rep = _ChunkWriter(out_dir / REPARSE_SUBDIR
                                         / f"{name}.parquet")
                w_rep.write_frame(f_rep)
            del chunk, f_face, f_rep, rep
            n_chunks += 1
            rss = _rss_gb()
            if rss is not None:
                peak_rss = max(peak_rss, rss)
                if max_mem_gb and rss > 0.85 * float(max_mem_gb) \
                        and chunk_rows > 1000:
                    chunk_rows = max(1000, chunk_rows // 2)
                    shrinks += 1
                    logger.warning("流式视图 RSS %.1f GB 逼近上限 %.1f GB,"
                                   "块大小减半为 %d 行", rss,
                                   float(max_mem_gb), chunk_rows)
            start = stop
        w_face.close()
        w_ident.close()
        if w_lab is not None:
            w_lab.close()
        if w_rep is not None:
            w_rep.close()
        labels_full = pd.concat(label_parts, ignore_index=True) \
            if label_parts else None
        manifest["splits"][name] = _view_split_entry(
            name, out_dir / f"{name}.parquet", labels_full, spec,
            w_face.rows)
        if w_face.rows != n_rows:
            raise RuntimeError(f"{name} 落盘行数 {w_face.rows} != 父视图 {n_rows}")
    feat_cols_ref = feat_cols_ref or []
    manifest["n_features"] = len(feat_cols_ref)
    manifest["feature_sha256"] = hashlib.sha256(
        "\n".join(sorted(feat_cols_ref)).encode("utf-8")).hexdigest()[:16]
    manifest["has_reparse_dir"] = has_reparse
    dump_json(out_dir / "manifest.json", manifest)
    _assert_view_clean(out_dir, spec)
    dump_json(out_dir / BUILD_INFO_FILE, {
        "mode": "stream", "created_at": now_iso(),
        "chunk_rows_final": int(chunk_rows), "n_chunks": n_chunks,
        "chunk_shrinks": shrinks,
        "max_mem_gb": float(max_mem_gb) if max_mem_gb else None,
        "peak_rss_gb": round(float(peak_rss), 3),
        "est_bytes_per_row": round(float(bpr), 1),
        "self_check": check,
        "elapsed_sec": round(__import__("time").time() - t0, 2),
        "contract": "build_view 必须是行内变换(输出第 i 行只依赖输入第 i 行)",
    })
    return manifest


# ---------------------------------------------------------------------------
# v2.17.0(#12):--from-feature-block —— 已缓存/可重算特征块直接材料化为视图
# ---------------------------------------------------------------------------
def _find_block_functions(module, block: str) -> list:
    """trial 模块里调用 feature_block(<block>, ...) 的 split→DataFrame 函数。

    block 可以是 feature_block 的块名(如 amt_v1),也可以直接是函数名
    (如 _amt_block)。按字节码常量识别,不执行任何代码。
    """
    import inspect
    import types as _types

    fn = getattr(module, block, None)
    if inspect.isfunction(fn) and getattr(fn, "__module__", None) == \
            module.__name__:
        return [fn]

    def _consts(code):
        out = set()
        for c in code.co_consts:
            if isinstance(c, str):
                out.add(c)
            elif isinstance(c, _types.CodeType):
                out |= _consts(c)
        return out

    def _names(code):
        out = set(code.co_names)
        for c in code.co_consts:
            if isinstance(c, _types.CodeType):
                out |= _names(c)
        return out

    hits = []
    for name, obj in vars(module).items():
        if not inspect.isfunction(obj) or \
                getattr(obj, "__module__", None) != module.__name__:
            continue
        code = obj.__code__
        if "feature_block" in _names(code) and block in _consts(code):
            hits.append(obj)
    return hits


def _block_names_in(module) -> list[str]:
    import inspect

    names = []
    for _n, obj in vars(module).items():
        if inspect.isfunction(obj) and \
                getattr(obj, "__module__", None) == module.__name__ and \
                "feature_block" in obj.__code__.co_names:
            names.append(_n)
    return sorted(names)


def write_view_from_feature_blocks(
    blocks: list[tuple[Path, str]],
    spec: TaskSpec,
    run_dir: Path,
    view_id: str,
    *,
    chunk_rows: int | None = None,
    max_mem_gb: float | None = None,
    parent_splits_dir: Path | None = None,
    note: str = "",
    verify_rows: int = 2048,
) -> dict:
    """把 trial 的特征块(缓存命中或按内核重算)追加到父视图特征面,落为新视图。

    - 父视图特征面按 arrow 批原样拷贝(不经 pandas,原列逐位不变),块列以
      float32 右接;labels/identity/reparse 按字节复制父视图(行集/行序/样本键
      与父视图逐位一致);
    - 逐值一致性校验:(a) 落盘后逐批回读块列,与块数据逐位比对(NaN 视为
      相等);(b) 缓存命中时用当前内核在每个 split 首 verify_rows 行上重算,
      与缓存逐位比对(防"内核已改、缓存仍旧"的错用);
    - manifest 字段与全量模式一致:source_script = 块所在 trial 代码。
    """
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    from . import feature_cache as FC
    from .group_importance import _load_trial_code_module

    run_dir = Path(run_dir)
    plan = _ParentPlan(spec, run_dir, parent_splits_dir)
    parent_face_cols = set(pq.read_schema(
        plan.parent_dir / f"{plan.names[0]}.parquet").names)
    # ---- 1) 每个 (trial, block) 在每个 split 上取块(缓存命中/重算)----
    captured: list[dict] = []
    new_cols: list[str] = []
    for tdir, block in blocks:
        tdir = Path(tdir)
        module = _load_trial_code_module(tdir)
        fns = _find_block_functions(module, block)
        if not fns:
            raise ValueError(
                f"{tdir.name} 中找不到特征块 {block!r}(按 feature_block 块名或"
                f"函数名识别);可用的块函数: {_block_names_in(module)}")
        if len(fns) > 1:
            raise ValueError(f"{tdir.name} 中块 {block!r} 对应多个函数 "
                             f"{[f.__name__ for f in fns]},请改用函数名指定")
        fn = fns[0]
        t_splits = Path(getattr(module, "SPLITS", plan.parent_dir)).resolve()
        if t_splits != plan.parent_dir.resolve():
            # 视图与 v1 行集/行序一致(契约)→ 块可对齐;抽样面等子集视图除外
            t_ident = load_identities(run_dir, str(
                t_splits.relative_to(run_dir.resolve())))
            for n in plan.names:
                a = t_ident.get(n)
                b = plan.identities[n]
                if a is None or len(a) != len(b) or not a[SAMPLE_KEY_COLUMN] \
                        .astype(str).reset_index(drop=True).equals(
                            b[SAMPLE_KEY_COLUMN].astype(str)
                            .reset_index(drop=True)):
                    raise ValueError(
                        f"{tdir.name} 的取数目录 {t_splits} 与父视图在 {n} 上"
                        "行集/行序不一致(抽样面等子集视图上的块不能材料化)")
        per_split: dict[str, dict] = {}
        for n in plan.names:
            seen: list[dict] = []
            FC._BLOCK_HOOK = seen.append
            try:
                frame = fn(n)
            finally:
                FC._BLOCK_HOOK = None
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{fn.__name__}({n!r}) 未返回 DataFrame")
            if len(frame) != plan.rows[n]:
                raise ValueError(f"块 {block} 在 {n} 上 {len(frame)} 行 != "
                                 f"父视图 {plan.rows[n]} 行")
            info = seen[-1] if seen else {}
            cache_file = info.get("cache_file")
            entry = {"info": info, "columns": list(frame.columns)}
            if cache_file and Path(cache_file).exists():
                entry["cache_file"] = Path(cache_file)
                FC._MEMO.pop(info.get("key"), None)  # 内存有界:改为流式回读
            else:
                entry["frame"] = frame.astype("float32").reset_index(drop=True)
            per_split[n] = entry
            del frame
        cols = per_split[plan.names[0]]["columns"]
        for n in plan.names:
            if per_split[n]["columns"] != cols:
                raise ValueError(f"块 {block} 的列在 split 间不一致")
        clash = sorted((set(cols) & parent_face_cols) | (set(cols) &
                                                        set(new_cols)))
        if clash:
            raise ValueError(f"块列与父视图特征/其它块重名: {clash[:6]}"
                             "(视图只增列,不替换)")
        new_cols += cols
        captured.append({"trial_dir": tdir, "block": block, "fn": fn,
                         "columns": cols, "per_split": per_split})
    _guard_derived_columns(new_cols, spec)

    # ---- 2) 缓存命中 → 用当前内核抽查重算(首 verify_rows 行)----
    spot: list[dict] = []
    for cap in captured:
        for n, ent in cap["per_split"].items():
            info = ent["info"]
            if not info or "cache_file" not in ent or verify_rows <= 0:
                continue
            src = Path(info["source_path"])
            pf = pq.ParquetFile(src)
            b = next(pf.iter_batches(batch_size=int(verify_rows),
                                     columns=list(info["sources"])))
            got = FC._as_block(
                info["fn"](FC._batch_payload(b, list(info["sources"]),
                                             info["batch_format"])),
                b.num_rows, len(cap["columns"]))
            cached = next(pq.ParquetFile(ent["cache_file"]).iter_batches(
                batch_size=b.num_rows)).to_pandas().to_numpy(dtype=np.float32)
            ok = bool(np.array_equal(got, cached[:b.num_rows], equal_nan=True))
            spot.append({"block": cap["block"], "split": n,
                         "rows": int(b.num_rows), "equal": ok})
            if not ok:
                raise ValueError(
                    f"块 {cap['block']} 在 {n} 上缓存值与当前内核重算不一致"
                    "(内核已改而缓存未失效?)—— 拒绝材料化")

    # ---- 3) 流式落盘:父特征面原样 + 块列 ----
    out_dir = _view_dirs(run_dir, view_id, spec)
    src_script = Path(captured[0]["trial_dir"]) / "code" / "train.py"
    manifest = _view_manifest_head(
        spec, view_id,
        (note + "; " if note else "") + "from-feature-block " + ",".join(
            f"{Path(c['trial_dir']).name}:{c['block']}" for c in captured),
        src_script if src_script.exists() else None)
    if chunk_rows is None:
        chunk_rows = 200_000
    if max_mem_gb:
        bpr = plan.bytes_per_row() + 4 * len(new_cols)
        chunk_rows = min(int(chunk_rows),
                         int(float(max_mem_gb) * 1e9 * 0.8 / (bpr * 4)))
    chunk_rows = max(1000, int(chunk_rows))
    has_reparse = False
    value_check = {"columns": len(new_cols), "rows": 0, "mismatches": 0}
    for n in plan.names:
        face_rd = _RowReader(plan.parent_dir / f"{n}.parquet",
                             batch_rows=min(chunk_rows, 65536))
        blk_rds = []
        for cap in captured:
            ent = cap["per_split"][n]
            blk_rds.append(_RowReader(ent["cache_file"]) if "cache_file" in ent
                           else ent["frame"])
        w = _ChunkWriter(out_dir / f"{n}.parquet")
        start, total = 0, plan.rows[n]
        while start < total:
            stop = min(total, start + chunk_rows)
            tbl = face_rd.take(stop - start)
            for cap, rd in zip(captured, blk_rds):
                part = rd.take(stop - start) if isinstance(rd, _RowReader) \
                    else pa.Table.from_pandas(rd.iloc[start:stop],
                                              preserve_index=False)
                for c in cap["columns"]:
                    tbl = tbl.append_column(
                        pa.field(c, pa.float32()),
                        part.column(c).cast(pa.float32()))
            w.write_table(tbl)
            start = stop
        w.close()
        # labels / identity / reparse:父视图原样(字节复制)
        for sub in (LABELS_SUBDIR, IDENTITY_SUBDIR, REPARSE_SUBDIR):
            sp = plan.parent_dir / sub / f"{n}.parquet"
            if sub == LABELS_SUBDIR and n == "final_test":
                continue
            if sp.exists():
                (out_dir / sub).mkdir(exist_ok=True)
                shutil.copyfile(sp, out_dir / sub / f"{n}.parquet")
                if sub == REPARSE_SUBDIR:
                    has_reparse = True
        # 逐值一致性:回读落盘块列 vs 块源(逐批、逐位)
        rd_out = _RowReader(out_dir / f"{n}.parquet", columns=new_cols)
        src_rds = [(_RowReader(cap["per_split"][n]["cache_file"])
                    if "cache_file" in cap["per_split"][n]
                    else cap["per_split"][n]["frame"]) for cap in captured]
        start = 0
        while start < total:
            stop = min(total, start + 65536)
            got = rd_out.take(stop - start)
            for cap, rd in zip(captured, src_rds):
                ref = rd.take(stop - start).to_pandas() \
                    if isinstance(rd, _RowReader) else rd.iloc[start:stop]
                for c in cap["columns"]:
                    a = got.column(c).to_numpy(zero_copy_only=False)
                    b = ref[c].to_numpy(dtype=np.float32)
                    if not np.array_equal(a.astype(np.float32), b,
                                          equal_nan=True):
                        value_check["mismatches"] += 1
            start = stop
        value_check["rows"] += int(total)
        manifest["splits"][n] = _view_split_entry(
            n, out_dir / f"{n}.parquet", plan.labels.get(n), spec, w.rows)
    if value_check["mismatches"]:
        raise RuntimeError(f"逐值一致性校验失败: {value_check}")
    face_cols = pq.read_schema(out_dir / f"{plan.names[0]}.parquet").names
    feat_cols = resolve_feature_columns(pd.DataFrame(columns=face_cols), spec)
    manifest["n_features"] = len(feat_cols)
    manifest["feature_sha256"] = hashlib.sha256(
        "\n".join(sorted(feat_cols)).encode("utf-8")).hexdigest()[:16]
    manifest["has_reparse_dir"] = has_reparse
    dump_json(out_dir / "manifest.json", manifest)
    _assert_view_clean(out_dir, spec)
    dump_json(out_dir / BUILD_INFO_FILE, {
        "mode": "from_feature_block", "created_at": now_iso(),
        "chunk_rows": int(chunk_rows),
        "blocks": [{
            "trial": Path(c["trial_dir"]).name, "block": c["block"],
            "function": c["fn"].__name__, "columns": len(c["columns"]),
            "column_head": c["columns"][:5],
            "per_split": {n: {"how": (e["info"] or {}).get("how"),
                              "key": (e["info"] or {}).get("key"),
                              "cache_file": str(e.get("cache_file") or "")}
                          for n, e in c["per_split"].items()},
        } for c in captured],
        "value_check": value_check,
        "spot_recompute": spot,
    })
    return manifest


# ---------------------------------------------------------------------------
# v2.17.0(#16):prescreen —— train 内时间切片的沙箱数据面
# ---------------------------------------------------------------------------
# trial 代码的实际取数路径(templates.py 与 run4 真实 trial 核对):
#   - 特征面:pd.read_parquet(SPLITS / "{split}.parquet"),SPLITS =
#     Path(__file__).parents[3] / "sample-profile/<view>";
#   - 标签面:模板内 load_labels(split) → SPLITS / "labels/{split}.parquet";
#   - 解析源:模板内 reparse_frame(split) → SPLITS / "reparse/...";
#   - 特征块:runtime.feature_cache.feature_block(splits_dir=SPLITS);
#   - 少数手写 trial 直接调本模块 load_split_frames/load_labels/
#     load_identities/reparse_frames(run_dir)。
# 因此 prescreen 把 trial 代码复制进一个**镜像 run 布局的沙箱**
# (<sandbox>/trials/<id>/code/train.py),沙箱的 sample-profile/<view>/ 里:
#   train      = train 的拟合段 + 早停段(时间序,早停段在尾部);
#   validation = train 的留出段(时间最晚的 holdout_frac);
#   shadow_test= 留出段前若干行(只为让 shadow 打分代码路径照常执行);
#   final_test 不存在。
# SPLITS 相对 __file__ 解析,天然落进沙箱;另以环境变量
# MODELEVO_PRESCREEN_ROOT 让本模块的访问函数与 feature_block 把沙箱外的
# run_dir/splits_dir 重映射进沙箱(手写 trial 用绝对路径时兜底)。
# 沙箱只由真实 run 的 **train** 四产物生成;validation/shadow/final 的标签
# 绝不读取(build_prescreen_splits 的访问日志逐条记录并断言)。
PRESCREEN_ROOT_ENV = "MODELEVO_PRESCREEN_ROOT"


def _prescreen_run_dir(run_dir: Path) -> Path:
    """prescreen 沙箱内:把沙箱外的 run_dir 重映射到沙箱根(同相对布局)。"""
    root = os.environ.get(PRESCREEN_ROOT_ENV)
    if not root:
        return Path(run_dir)
    root_p = Path(root).resolve()
    rd = Path(run_dir).resolve()
    if rd == root_p or root_p in rd.parents:
        return Path(run_dir)
    logger.info("prescreen 沙箱:数据访问 %s 重映射到 %s", run_dir, root_p)
    return root_p


class PrescreenAccessLog(list):
    """harness 侧读取清单;只允许 train 的四产物(绝不读 validation/shadow/
    final 的任何面,尤其是标签)。"""

    def read(self, path: Path, **kw) -> pd.DataFrame:
        self.check(path)
        self.append(str(Path(path)))
        return pd.read_parquet(path, **kw)

    def check(self, path: Path) -> None:
        p = Path(path)
        if p.stem in ("validation", "shadow_test", "final_test"):
            raise PermissionError(
                f"prescreen 禁止读取 {p}(只用 train 面;validation/shadow/"
                "final_test 的数据与标签不得进入预筛)")


def _segment_bounds(t_sorted: np.ndarray, n: int, holdout_frac: float,
                    es_frac: float) -> tuple[int, int]:
    """(拟合段末, 早停段末);边界尽量吸附到时间值变化处(同一时点不跨段)。"""
    n_ho = max(1, int(round(n * holdout_frac)))
    n_es = max(1, int(round(n * es_frac))) if es_frac > 0 else 0
    b2 = n - n_ho
    b1 = b2 - n_es

    def _snap(b: int, slack: int) -> int:
        if b <= 0 or b >= n or t_sorted is None:
            return b
        if t_sorted[b] != t_sorted[b - 1]:
            return b
        j = b
        while j < min(n, b + slack) and t_sorted[j] == t_sorted[j - 1]:
            j += 1
        if j < n and j < b + slack and t_sorted[j] != t_sorted[j - 1]:
            return j
        j = b
        while j > max(1, b - slack) and t_sorted[j] == t_sorted[j - 1]:
            j -= 1
        return j if t_sorted[j] != t_sorted[j - 1] else b

    b2 = _snap(b2, max(1, n_ho // 2))
    if n_es:
        b1 = min(b2 - 1, _snap(b2 - n_es, max(1, n_es // 2)))
    else:
        b1 = b2
    return max(1, b1), b2


def build_prescreen_splits(
    run_dir: Path,
    spec: TaskSpec,
    out_root: Path,
    *,
    view_id: str = DEFAULT_VIEW_ID,
    holdout_frac: float = 0.15,
    es_frac: float = 0.05,
    max_rows: int | None = None,
    shadow_rows: int = 1000,
    access_log: PrescreenAccessLog | None = None,
) -> dict:
    """由真实 run 的 train 面生成 prescreen 沙箱的数据面(流式,内存有界)。

    out_root 是沙箱 run 根;产物落 out_root/<view 相对路径>/(与 trial 的
    SPLITS 相对路径一致)。同参数 + 同源(train 文件内容 sha)时复用。
    """
    import pyarrow.parquet as pq

    log = access_log if access_log is not None else PrescreenAccessLog()
    run_dir = Path(run_dir)
    src = view_splits_dir(run_dir, view_id)
    rel = src.relative_to(run_dir)
    dst = Path(out_root) / rel
    face_p = src / "train.parquet"
    lab_p = src / LABELS_SUBDIR / "train.parquet"
    id_p = src / IDENTITY_SUBDIR / "train.parquet"
    rp_p = src / REPARSE_SUBDIR / "train.parquet"
    for p in (face_p, lab_p, id_p):
        if not p.exists():
            raise FileNotFoundError(f"prescreen 需要 train 四产物: {p}")
    log.check(face_p)
    from .feature_cache import content_sha

    src_sha = content_sha(face_p)[0]   # manifest 记录优先(免读大文件)
    params = {"view_id": view_id, "holdout_frac": float(holdout_frac),
              "es_frac": float(es_frac), "max_rows": int(max_rows or 0),
              "shadow_rows": int(shadow_rows), "source_train_sha256": src_sha}
    info_p = dst / "prescreen-splits.json"
    if info_p.exists() and (dst / "manifest.json").exists():
        try:
            old = __import__("json").loads(info_p.read_text(encoding="utf-8"))
            if old.get("params") == params:
                old["reused"] = True
                return old
        except Exception:  # noqa: BLE001
            pass
    labels = log.read(lab_p)
    ident = log.read(id_p)
    n_all = len(ident)
    if len(labels) != n_all or int(pq.ParquetFile(face_p).metadata.num_rows) \
            != n_all:
        raise ValueError("train 四产物行数不一致")
    tcol = spec.time_column
    if tcol and tcol in ident.columns:
        tv = ident[tcol].astype(str).to_numpy()
        order = np.argsort(tv, kind="stable")
        ordering = f"sorted_by:{tcol}"
    else:
        tv = None
        order = np.arange(n_all)
        ordering = "file_order_no_time_column"
    if max_rows and n_all > int(max_rows):
        pick = np.unique(np.linspace(0, n_all - 1, int(max_rows))
                         .round().astype(np.int64))
        order = order[pick]
    n = len(order)
    if n < 50:
        raise ValueError(f"train 行数过少({n}),无法做 prescreen 切片")
    t_sorted = tv[order] if tv is not None else None
    b1, b2 = _segment_bounds(t_sorted, n, holdout_frac, es_frac)
    fit_pos = np.sort(order[:b1])
    es_pos = np.sort(order[b1:b2])
    ho_pos = np.sort(order[b2:])
    if not len(ho_pos) or not len(fit_pos):
        raise ValueError("prescreen 切片为空(检查 holdout/es 比例)")
    if dst.exists():
        import shutil

        shutil.rmtree(dst)
    (dst / LABELS_SUBDIR).mkdir(parents=True)
    (dst / IDENTITY_SUBDIR).mkdir()

    def _stream_select(path: Path, segments: dict[str, np.ndarray],
                       writers: dict) -> None:
        """一次顺序扫描,把各段行(文件序)写进各自 writer。"""
        pf = pq.ParquetFile(path)
        off = 0
        for b in pf.iter_batches(batch_size=65536):
            k = b.num_rows
            for seg, pos in segments.items():
                lo, hi = np.searchsorted(pos, [off, off + k])
                if hi > lo:
                    idx = pos[lo:hi] - off
                    writers[seg].write_table(
                        __import__("pyarrow").Table.from_batches([b])
                        .take(idx))
            off += k

    def _plane(path: Path, out_name: str) -> None:
        """train = 拟合段 + 早停段(早停段接在尾部);validation = 留出段。"""
        tmp_es = dst / (".es-" + out_name.format(split="train")
                        .replace("/", "_") + ".tmp.parquet")
        w_tr = _ChunkWriter(dst / out_name.format(split="train"))
        w_es = _ChunkWriter(tmp_es)
        w_va = _ChunkWriter(dst / out_name.format(split="validation"))
        segs = {"fit": fit_pos, "es": es_pos, "ho": ho_pos}
        _stream_select(path, {k: v for k, v in segs.items() if len(v)},
                       {"fit": w_tr, "es": w_es, "ho": w_va})
        if len(es_pos):
            w_es.close()
            for b in pq.ParquetFile(tmp_es).iter_batches(batch_size=65536):
                w_tr.write_table(__import__("pyarrow").Table.from_batches([b]))
            tmp_es.unlink(missing_ok=True)
        w_tr.close()
        w_va.close()

    log.append(str(face_p))
    _plane(face_p, "{split}.parquet")
    if rp_p.exists():
        log.check(rp_p)
        log.append(str(rp_p))
        (dst / REPARSE_SUBDIR).mkdir()
        _plane(rp_p, REPARSE_SUBDIR + "/{split}.parquet")
    tr_idx = np.concatenate([fit_pos, es_pos])
    for frame, sub in ((labels, LABELS_SUBDIR), (ident, IDENTITY_SUBDIR)):
        frame.iloc[tr_idx].reset_index(drop=True).to_parquet(
            dst / sub / "train.parquet", index=False)
        frame.iloc[ho_pos].reset_index(drop=True).to_parquet(
            dst / sub / "validation.parquet", index=False)
    # shadow_test:留出段前 shadow_rows 行(沙箱内部复制,只为代码路径完整)
    k_sh = int(min(max(0, shadow_rows), len(ho_pos)))
    if k_sh:
        for sub in ("", LABELS_SUBDIR, IDENTITY_SUBDIR, REPARSE_SUBDIR):
            sp = dst / sub / "validation.parquet"
            if sp.exists():
                pq.write_table(pq.read_table(sp).slice(0, k_sh),
                               dst / sub / "shadow_test.parquet")
    manifest = {
        "created_at": now_iso(),
        "view_id": f"prescreen:{view_id}",
        "prescreen": True,
        "split_policy": spec.data.get("split_policy"),
        "leak_manifest_hash": spec.leak_manifest_hash,
        "banned_columns": spec.banned_columns,
        "reparse_columns": spec.reparse_columns,
        "splits": {},
        "has_reparse_dir": rp_p.exists(),
    }
    for name in ("train", "validation", "shadow_test"):
        p = dst / f"{name}.parquet"
        if not p.exists():
            continue
        manifest["splits"][name] = {
            "rows": int(pq.ParquetFile(p).metadata.num_rows),
            "file": p.name, "sha256": _sha16(p), "ordering": ordering,
            "identity_file": f"{IDENTITY_SUBDIR}/{name}.parquet",
            "labels_file": f"{LABELS_SUBDIR}/{name}.parquet",
        }
    dump_json(dst / "manifest.json", manifest)
    _assert_view_clean(dst, spec)

    def _tval(pos_arr, which):
        if tv is None or not len(pos_arr):
            return None
        return str(tv[pos_arr].min() if which == "min" else tv[pos_arr].max())

    info = {
        "params": params,
        "splits_dir": str(dst),
        "relative_splits_dir": str(rel),
        "ordering": ordering,
        "n_train_source": int(n_all),
        "n_used": int(n),
        "n_fit": int(len(fit_pos)), "n_es": int(len(es_pos)),
        "n_holdout": int(len(ho_pos)), "n_shadow_stub": k_sh,
        "es_fraction_of_sandbox_train": round(
            len(es_pos) / max(1, len(fit_pos) + len(es_pos)), 6),
        "fit_time_range": [_tval(fit_pos, "min"), _tval(fit_pos, "max")],
        "es_time_range": [_tval(es_pos, "min"), _tval(es_pos, "max")],
        "holdout_time_range": [_tval(ho_pos, "min"), _tval(ho_pos, "max")],
        "created_at": now_iso(),
        "reused": False,
    }
    dump_json(info_p, info)
    return info


# ---------------------------------------------------------------------------
# v2.17.0(#19):抽样面噪声预检
# ---------------------------------------------------------------------------
def _bootstrap_metric_se(metric: str, y: np.ndarray, s: np.ndarray, *,
                         treatment: np.ndarray | None = None,
                         n_boot: int = 200, seed: int = 20260814) -> float:
    from .metrics import REGRESSION_METRICS, compute_metric

    rng = np.random.default_rng(seed)
    n = len(y)
    reg = metric in REGRESSION_METRICS
    vals = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        ss = s[idx]
        try:
            vals.append(float(compute_metric(
                metric, y[idx], ss if reg else None, score=ss,
                treatment=None if treatment is None else treatment[idx])))
        except Exception:  # noqa: BLE001  退化重采样(单类)跳过
            continue
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")


def sample_noise_precheck(run_dir: Path, spec: TaskSpec, config, *,
                          n_boot: int = 200) -> dict:
    """write-splits-sample 之后:baseline 的 v1 validation 预测在抽样面子集与
    全量面上的主指标 bootstrap SE,对比 MDE,给出是否直接用 v1 面的建议。

    依据(run4 regression.2):26k 行抽样验证面上原值 rmse 的配对 SE 约
    50~89,全量面 12~21,MDE=10 —— 抽样面对 MDE 量级完全没有分辨力,而全量
    trial 只需约 6 min。判据:SE_sample > 2×MDE 且 baseline 运行时长 <
    feature_exploration.sample_precheck_max_full_minutes(默认 30)→
    recommend_v1=true。SE_sample 优先用"配对 SE 估计"(已评估 v1 trial 的
    配对 bootstrap SE 中位数 × se_ratio,可得时),否则用 baseline 主指标的
    bootstrap SE(偏保守)。结果写 splits-sample/manifest.json 的
    noise_precheck(WP2 planner-context 读取 recommend_v1/se_ratio/se_sample/
    se_full/mde/reason)。
    """
    from .logging_utils import load_json as _lj, load_yaml as _ly
    from .metrics import align_score_by_keys

    run_dir = Path(run_dir)
    out: dict = {"recommend_v1": False, "se_ratio": None, "se_sample": None,
                 "se_full": None, "mde": None, "reason": "",
                 "computed_at": now_iso(), "metric": spec.primary_metric}

    def _skip(reason: str) -> dict:
        out.update({"skipped": True, "reason": reason})
        return out

    try:
        st = _lj(run_dir / "run-state.json") if (
            run_dir / "run-state.json").exists() else {}
    except Exception:  # noqa: BLE001
        st = {}
    bid = (st or {}).get("baseline_id")
    if not bid:
        return _skip("run-state 无 baseline_id(baseline 未跑),跳过噪声预检")
    bdir = run_dir / "trials" / str(bid)
    rec = (_ly(bdir / "trial.yaml") or {}) if (bdir / "trial.yaml").exists() \
        else {}
    view = str(rec.get("data_view") or "v1")
    score_p = bdir / "validation_score.parquet"
    if view != DEFAULT_VIEW_ID or not score_p.exists():
        return _skip(f"baseline {bid} 无 v1 validation 预测(view={view},"
                     f"score={'有' if score_p.exists() else '无'}),跳过噪声预检")
    v1_lab_p = view_splits_dir(run_dir) / LABELS_SUBDIR / "validation.parquet"
    s_lab_p = view_splits_dir(run_dir, SAMPLE_VIEW_ID) / LABELS_SUBDIR / \
        "validation.parquet"
    if not v1_lab_p.exists() or not s_lab_p.exists():
        return _skip("v1 或抽样面 validation 标签面缺失,跳过噪声预检")
    lab = pd.read_parquet(v1_lab_p)
    sdf = pd.read_parquet(score_p)
    y = lab[spec.target_name].to_numpy()
    s = sdf["score"].to_numpy(dtype=float)
    if SAMPLE_KEY_COLUMN in lab.columns and SAMPLE_KEY_COLUMN in sdf.columns:
        s = align_score_by_keys(s, lab[SAMPLE_KEY_COLUMN].astype(str).values,
                                sdf[SAMPLE_KEY_COLUMN].astype(str).values)
    elif len(s) != len(y):
        return _skip("baseline 预测与 v1 validation 行数不一致,跳过噪声预检")
    tcol = spec.treatment_column
    tr = lab[tcol].to_numpy() if tcol and tcol in lab.columns else None
    skeys = set(pd.read_parquet(s_lab_p, columns=[SAMPLE_KEY_COLUMN])[
        SAMPLE_KEY_COLUMN].astype(str))
    mask = lab[SAMPLE_KEY_COLUMN].astype(str).isin(skeys).to_numpy()
    if mask.sum() < 30:
        return _skip(f"抽样面与 v1 validation 交集仅 {int(mask.sum())} 行")
    metric = spec.primary_metric
    se_full = _bootstrap_metric_se(metric, y, s, treatment=tr, n_boot=n_boot)
    se_samp = _bootstrap_metric_se(
        metric, y[mask], s[mask],
        treatment=None if tr is None else tr[mask], n_boot=n_boot)
    ratio = se_samp / se_full if se_full and np.isfinite(se_full) \
        and se_full > 0 else None
    # 配对 SE 参照:已评估的 v1 trial 的配对 bootstrap SE(noise)中位数
    paired = []
    for ev_p in sorted((run_dir / "trials").glob("*/evaluation.json")):
        try:
            ty = _ly(ev_p.parent / "trial.yaml") or {}
            if str(ty.get("data_view") or "v1") != DEFAULT_VIEW_ID:
                continue
            nz = (_lj(ev_p) or {}).get("noise")
            if nz is not None and np.isfinite(float(nz)) and float(nz) > 0:
                paired.append(float(nz))
        except Exception:  # noqa: BLE001
            continue
    paired_ref = float(np.median(paired)) if paired else None
    paired_est = paired_ref * ratio if (paired_ref is not None and ratio) \
        else None
    try:
        from .mde import effective_mde

        mde = float(effective_mde(config, run_dir=run_dir))
    except Exception:  # noqa: BLE001
        mde = None
    rt = (rec.get("result") or {}).get("runtime_sec")
    rt_min = round(float(rt) / 60.0, 2) if rt else None
    max_min = float(config.get(
        "feature_exploration.sample_precheck_max_full_minutes", 30) or 30)
    se_dec = paired_est if paired_est is not None else se_samp
    basis = "paired_estimate" if paired_est is not None \
        else "metric_bootstrap_se"
    noisy = bool(mde is not None and se_dec is not None
                 and np.isfinite(se_dec) and se_dec > 2 * mde)
    cheap = rt_min is not None and rt_min < max_min
    rec_v1 = bool(noisy and cheap)
    if mde is None:
        reason = "MDE 不可解析,不做建议"
    elif not noisy:
        reason = (f"抽样面 SE={se_dec:.4g}({basis})≤ 2×MDE={2 * mde:.4g},"
                  "抽样面对 MDE 量级仍有分辨力,按协议先用抽样面筛选")
    elif not cheap:
        reason = (f"抽样面 SE={se_dec:.4g}({basis})> 2×MDE={2 * mde:.4g},"
                  f"但 baseline 全量运行 {rt_min if rt_min is not None else '未知'}"
                  f" min ≥ 阈值 {max_min:g} min(或未知),仍用抽样面但阴性"
                  "结论不可信、方向须回 v1 复核")
    else:
        reason = (f"抽样面 SE={se_dec:.4g}({basis})> 2×MDE={2 * mde:.4g}:"
                  f"抽样面对 MDE 量级无分辨力;baseline 全量仅 {rt_min} min"
                  f" < {max_min:g} min —— 建议特征探索直接在 v1 全量面进行"
                  "(留痕本预检)")
    out.update({
        "recommend_v1": rec_v1,
        "se_ratio": None if ratio is None else round(float(ratio), 6),
        "se_sample": round(float(se_samp), 8),
        "se_full": round(float(se_full), 8),
        "mde": mde,
        "reason": reason,
        "basis": basis,
        "se_sample_paired_est": None if paired_est is None
        else round(float(paired_est), 8),
        "paired_se_full_ref": paired_ref,
        "n_paired_ref_trials": len(paired),
        "n_sample": int(mask.sum()), "n_full": int(len(y)),
        "baseline_id": str(bid),
        "baseline_runtime_min": rt_min,
        "max_full_minutes": max_min,
        "n_boot": int(n_boot),
        "skipped": False,
    })
    return out
