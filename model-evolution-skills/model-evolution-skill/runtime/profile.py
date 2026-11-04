"""SampleProfile:固定分析与数据审计(§7)。

拿到样本后首先执行,不由 LLM 决定是否执行;各分析并行执行;
代码固化在 Skill 内,仅 data-loader 针对任务填写(§7.3)。
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (resolve_feature_columns, split_data, splits_reusable,
                   write_splits)
from .logging_utils import _CNFormatter, get_logger, now_iso, dump_json, \
    dump_yaml
from .metrics import ks_stat, roc_auc
from .safe_dtype import safe_to_numeric
from .spec import TaskSpec

logger = get_logger(__name__)

LEAKAGE_HIGH_UNIVARIATE_AUC = 0.85
LEAKAGE_ID_LIKE_UNIQUE_RATIO = 0.95
#: v22.1:复合文本字段的形态判据阈值。真 ID 长度恒定、无分隔符、无词表复用;
#: 事件序列/JSON/编码串三项全反。详见 `_text_shape_evidence`。
TEXT_SHAPE_LEN_CV_MIN = 0.25      # 长度变异系数下限(定长 ID 的 CV≈0)
TEXT_SHAPE_VOCAB_REUSE_MIN = 50   # token 语料/词表 复用倍数下限
TEXT_SHAPE_DELIM_MIN = 0.5        # 平均分隔符个数下限
TEXT_SHAPE_SCAN_ROWS = 20000      # 形态扫描行数上限(逐行字符串运算较贵)
#: v2.17.0(D9):id_leakage 判据——列名标识符词提示与标识符 token 形态
ID_NAME_HINT_RE = re.compile(
    r"(?i)(?:^|[_\W])(?:id|ids|uid|uuid|guid|key|no|sn|code|hash|token|"
    r"account|acct|phone|mobile|card|serial)(?:[_\W]|$)|[a-z]id$|"
    r"编号|号码|账号|卡号|单号")
ID_TOKEN_RE = r"[A-Za-z0-9_\-:.#/]+"
ID_SHAPE_SCAN_ROWS = 20000        # 标识符形态扫描行数上限
PSI_BANDS = [(0.1, "high"), (0.25, "severe")]  # PSI>0.1 高漂移,>0.25 严重
PROFILE_SAMPLE_CAP = 1_000_000  # 分析样本上限(超限确定性下采样;2M→1M 提速)
#: v2.17.2(E2):reparse/文本列 token 级漂移(按时间列月份)。事故:run4
#: req-t7-400w 的 LOAN_MARKTING 事件在 2026-02 之后断档,drift-profile 只有
#: 数值列 PSI,靠 gen-03 手写探针才发现。下列缺省值均可经 config
#: `profile.token_drift_*` 覆盖(enabled/top_k/rows_per_month/sep/head_sep/
#: min_baseline_coverage/collapse_ratio/edge_frac/edge_collapse_ratio/
#: max_extra_columns)。
TOKEN_DRIFT_TOP_K = 50                 # 按 train 行覆盖率取头部 token
TOKEN_DRIFT_ROWS_PER_MONTH = 5000      # 每月确定性等距抽样行数上限(成本有界)
TOKEN_DRIFT_MIN_MONTH_ROWS = 100       # 与跨 date PSI 同口径:不足 100 行的月不比
TOKEN_DRIFT_SEP = ","                  # token 分隔字符集(每个字符都是分隔符)
TOKEN_DRIFT_HEAD_SEP = "@"             # 取首个该字符之前的头部作 token 名;"" 关闭
TOKEN_DRIFT_MIN_BASELINE_COVERAGE = 0.05   # 基线覆盖率 ≥5% 才判塌缩
TOKEN_DRIFT_COLLAPSE_RATIO = 0.10      # 某月覆盖率 < 基线×10% 即塌缩
#: 序列近端窗口:每行首/尾各 10% token 内的覆盖率。长回溯序列(seq_d365 =
#: 决策日前 365 天)里某事件停止落日志后,整行覆盖率只缓慢下降(run4 实测
#: LOAN_MARKTING 0.95→0.81,整行口径永远不会塌缩),近端窗口则陡降
#: (尾窗 0.56→0.075)。窗口被轻度用户的长跨度稀释,阈值取基线×50%;
#: 实测 41 个基线≥5% 的事件只命中 LOAN_MARKTING。
TOKEN_DRIFT_EDGE_FRAC = 0.10
TOKEN_DRIFT_EDGE_COLLAPSE_RATIO = 0.50
TOKEN_DRIFT_MAX_EXTRA_COLUMNS = 10     # reparse 之外的文本特征列上限
TOKEN_DRIFT_SHAPE_ROWS = 2000          # 文本特征列形态初筛行数

# 信息压缩检测(v8):loader 源码压缩模式(正则, 语义标签)
COMPRESSION_SCAN_PATTERNS = [
    (r"\.last\(", "last"),
    (r"\.tail\(\s*\d*\s*\)", "tail"),
    (r"\.iloc\[\s*-\d+", "iloc_neg"),
    (r"\.nth\(\s*-?\d+", "nth"),
    (r"drop_duplicates\(", "drop_duplicates"),
    (r"groupby\([^)]*\)\s*\.\s*agg\(", "groupby().agg"),
]
COMPRESSION_SCAN_MAX_ENTITIES = 200  # 组内零方差抽样的实体上限
COMPRESSION_SCAN_MAX_COLS = 50       # 抽样的数值特征上限


def raw_columns_fingerprint(columns, spec: TaskSpec) -> dict:
    """数据底座指纹(v2.15.10):loader 返回列 − spec 声明的派生列。

    `data.derived_columns` 列出 data-loader 在原始源之上**派生**的列(典型:
    标签变换 loan_amt_t7_log1p = log1p(loan_amt_t7))。它们不属于数据底座,
    不进指纹 —— 否则同一份数据只因标签口径不同就算出不同的 col_hash
    (实测 req-t7 run4 regression:1513 原生列 bc5329dc,+1 派生列变成
    e58b4b22,二分类 run 在同一数据上的经验对回归 run 完全不可见)。
    未声明 derived_columns 时与旧口径逐位一致(向后兼容)。
    """
    import hashlib

    derived = {str(c) for c in
               ((spec.data.get("data") or {}).get("derived_columns") or [])}
    cols = sorted(str(c) for c in columns if str(c) not in derived)
    out = {
        "raw_columns_hash": hashlib.sha256(
            "|".join(cols).encode("utf-8")).hexdigest()[:8],
        "raw_columns_count": int(len(cols)),
    }
    if derived:
        out["derived_columns_excluded"] = sorted(
            c for c in derived if c in {str(x) for x in columns})
    return out


def _scan_loader_source(loader_path: Path | None) -> list[dict]:
    """逐行正则扫描 data-loader.py 源码压缩模式;无文件/失败返回空。"""
    if loader_path is None or not Path(loader_path).exists():
        return []
    try:
        text = Path(loader_path).read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return []
    hits: list[dict] = []
    seen: set[str] = set()
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("#"):
            continue
        for pat, label in COMPRESSION_SCAN_PATTERNS:
            if label not in seen and re.search(pat, s):
                seen.add(label)
                hits.append({"pattern": label, "line": i,
                             "text": s[:120]})
    return hits


def _downsample_for_profile(df: pd.DataFrame, spec: TaskSpec,
                            cap: int = PROFILE_SAMPLE_CAP) -> tuple[pd.DataFrame,
                                                                    int | None]:
    """分析样本超过 cap 时下采样到 cap(固定 seed,可复现)。

    只影响 profile 分析与写盘的分桶视图规模;splits 落盘仍用全量
    (训练数据面不受影响)。按行序等距抽样保证时间/实体覆盖均匀。
    返回 (df_used, n_original);未超限时返回 (原 df, None)。
    """
    n = len(df)
    if n <= cap:
        return df, None
    rng = np.random.default_rng(20260814)
    # 等距骨架 + 随机抖动:时间分布均匀,又避免周期性对齐
    stride = n / cap
    base = (np.arange(cap) * stride).astype(np.int64)
    jitter = rng.integers(0, max(int(stride), 1), size=cap)
    idx = np.unique(np.clip(base + jitter, 0, n - 1))
    sampled = df.iloc[idx]
    logger.info("profile 下采样: %d → %d 行(分析口径;split 落盘仍全量)",
                n, len(sampled))
    return sampled, n


def _govern_analysis_dtypes(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """分析面 dtype 治理:可数值化的 object 列转 float64。

    历史教训(与 baseline.py 的 dtype 治理同源):parquet 读入的
    decimal 金额列为 object dtype,is_numeric_dtype 判否,IV/WOE 会
    走类别分支分桶(top N + __other__),高基数金额列爆出几十个伪
    类别桶;质量统计/单变量 AUC/PSI 也全部跳过。规则与 baseline
    一致:非空值 ≥95% 可数值化才转,否则保持类别口径。
    只作用于分析副本(浅拷贝后覆写列),落盘 splits 用的 df_full
    保持原始 dtype——训练数据面不变,trial 侧自行治理。

    v2.12.2.8:数值化改走 `safe_to_numeric`。旧实现的
    `notna().sum() >= 0.95*n` 判据在十六进制键列上会被
    `pd.to_numeric` 的**部分解析**骗过(24 位 hex 会被解析成数字),
    把 ID 列当成数值特征送进 IV/PSI;且长非数字列会直接段错误。
    """
    out = df.copy(deep=False)
    for col in cols:
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        conv = safe_to_numeric(out[col])
        if conv is not None:
            out[col] = conv.astype("float64")
    return out


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10,
         eps: float = 1e-6) -> float:
    """Population Stability Index;分位数分箱,NaN 归入独立箱。"""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    e_nan, a_nan = np.isnan(expected), np.isnan(actual)
    e, a = expected[~e_nan], actual[~a_nan]
    if len(e) < 10 or len(a) < 10:
        return float("nan")
    qs = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)[1:-1]))
    if len(qs) == 0:
        e_ratio = np.array([1.0])
        a_ratio = np.array([float(len(a)) / max(len(a), 1)])
    else:
        e_cuts = np.concatenate([[-np.inf], qs, [np.inf]])
        e_ratio = np.histogram(e, e_cuts)[0] / len(e) + eps
        a_ratio = np.histogram(a, e_cuts)[0] / len(a) + eps
        if e_nan.any() or a_nan.any():
            e_ratio = np.append(e_ratio, e_nan.mean() + eps)
            a_ratio = np.append(a_ratio, a_nan.mean() + eps)
    return float(np.sum((a_ratio - e_ratio) * np.log(a_ratio / e_ratio)))


def _iv_woe(y: np.ndarray, x: np.ndarray, bins: int = 10,
            eps: float = 0.5, max_categories: int = 50) -> tuple[float, list]:
    """单特征 IV 与分箱 WOE;二分类专用。

    类别列超过 max_categories 时只取 top N + __other__ 桶,
    防止高基数类别列产出上万 WOE 行。
    """
    y = np.asarray(y)
    x = np.asarray(x)
    pos, neg = y == 1, y != 1
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), []
    if pd.api.types.is_numeric_dtype(pd.Series(x)):
        xv = pd.to_numeric(pd.Series(x), errors="coerce").values.astype(float)
        nan_mask = np.isnan(xv)
        edges = np.unique(np.nanquantile(xv[~nan_mask], np.linspace(0, 1, bins + 1)))
        edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]]) if len(edges) > 2 \
            else np.array([-np.inf, np.inf])
        binned = np.digitize(xv, edges[1:-1], right=True)
        groups: list[tuple[str, np.ndarray]] = [
            (f"[{edges[i]:.4g},{edges[i + 1]:.4g})", binned == i)
            for i in range(len(edges) - 1)
        ]
        groups.append(("__nan__", nan_mask))
    else:
        s = pd.Series(x).astype(str)
        vc = s.value_counts()
        if len(vc) > max_categories:
            top = vc.index[:max_categories]
            groups = [(str(k), (s == k).values) for k in top]
            other_mask = ~s.isin(top).values
            if other_mask.any():
                groups.append(("__other__", other_mask))
        else:
            groups = [(str(k), (s == k).values) for k in vc.index]
    iv, table = 0.0, []
    for label, mask in groups:
        if not mask.any():
            continue
        g_pos, g_neg = int((mask & pos).sum()), int((mask & neg).sum())
        woe = np.log(((g_pos + eps) / n_pos) / ((g_neg + eps) / n_neg))
        part = ((g_pos + eps) / n_pos - (g_neg + eps) / n_neg) * woe
        iv += part
        table.append({"bin": label, "count": int(mask.sum()),
                      "pos": g_pos, "neg": g_neg,
                      "pos_rate": g_pos / max(mask.sum(), 1),
                      "woe": float(woe), "iv": float(part)})
    return float(iv), table


def _spearman_corr(y: np.ndarray, x: np.ndarray) -> float:
    """单特征与回归目标的 |Spearman corr|(方向归一)。

    NaN 成对剔除;样本 <10 或退化常数列返回 nan。类别列先按频序编码,
    高基数时只取 top50 + __other__(与 IV 的 max_categories 口径一致)。
    """
    x = np.asarray(x)
    m = ~np.isnan(np.asarray(x, dtype=float)) if \
        pd.api.types.is_numeric_dtype(pd.Series(x)) else None
    if m is not None:
        y_v, x_v = y[m], x[m].astype(float)
    else:
        s = pd.Series(x).astype(str)
        vc = s.value_counts()
        if len(vc) > 50:
            top = vc.index[:50]
            s = s.where(s.isin(top), "__other__")
        codes = pd.Series(s).astype("category").cat.codes
        y_v, x_v = y[codes >= 0], codes[codes >= 0].values.astype(float)
    if len(x_v) < 10:
        return float("nan")
    sx = pd.Series(x_v)
    if sx.nunique() <= 1:
        return float("nan")
    return float(abs(sx.corr(pd.Series(y_v), method="spearman")))


def _token_head_regex(sep: str = TOKEN_DRIFT_SEP,
                      head_sep: str = TOKEN_DRIFT_HEAD_SEP) -> re.Pattern:
    """v2.17.2(E2):token 头部抽取正则(一次 findall 完成切分 + 取头)。

    `sep` 中每个字符都是 token 分隔符;`head_sep` 非空时取 token 中首个该
    字符之前的部分(`LOAN_MARKTING@HR_noon@AMT_lt1k` → `LOAN_MARKTING`)。
    每个 token 产出一个头部;头部为空(以 head_sep 开头的位置标记,如
    `@DB_351`,或空 token)时产出 "",调用方丢弃 ""——它们不携带事件身份。
    "逐 token 吞掉"的写法比"(?:^|,)" 锚定快约 1.5 倍(不在每个字符位置
    重试分支)。
    """
    seps = "".join(re.escape(c) for c in (sep or TOKEN_DRIFT_SEP))
    heads = "".join(re.escape(c) for c in (head_sep or ""))
    return re.compile(r"([^%s%s]*)[^%s]*[%s]?" % (seps, heads, seps, seps))


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon 散度(log2,∈[0,1]);空分布返回 nan。"""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray) -> float:
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / m[nz])))

    return float(0.5 * _kl(p) + 0.5 * _kl(q))


class SampleProfile:
    """固定快速 Profile(§7.1):分布、特征质量、leakage audit。"""

    def __init__(self, df: pd.DataFrame, spec: TaskSpec, out_dir: Path,
                 sample_cap: int = PROFILE_SAMPLE_CAP,
                 loader_path: Path | None = None):
        # v19:PSI 列数上限(默认 2000;旧硬编码 300/150 把 1193 列截
        # 成 NaN 且偏好列序靠前的基础列)。config 缺省读取,读不到回退。
        # [FIX-2026-10-29] 必须带上 run 级 modelevo-config.yaml:旧版用
        # ``Config.load()`` 只读 skill 级,于是 profile 内部的泄漏对账
        # (``_cross_check_experience``)绕过 run 的 library.* 直连真实服务 ——
        # 实测 run config 写了 enabled=false/uri=:9,日志里仍是
        # ``LIBRARY_RESOLVED enabled=true uri=http://127.0.0.1:18765``,
        # 真实库视图照样被取回。``out_dir`` 是 ``<run>/sample-profile``,
        # 其父目录即 run 根。
        try:
            from .config import Config as _Config

            _run_cfg = Path(out_dir).parent / "modelevo-config.yaml"
            self.config = _Config.load(run_config=_run_cfg if _run_cfg.exists()
                                       else None)
        except Exception:  # noqa: BLE001
            class _C:  # noqa: SIM108
                @staticmethod
                def get(*_a, **_k):
                    return _k.get("default")
            self.config = _C()
        self.spec = spec
        self.out_dir = Path(out_dir)
        self.loader_path = Path(loader_path) if loader_path else None
        # v21:先切分、再建立搜索画像。final_test 特征与标签都不进入
        # profile 反馈；监督型画像(IV/AUC/相关性/标签分布)进一步只使用
        # train。validation/shadow 标签只允许固定 evaluator 消费。
        self.df_full = df
        self.n_original, self.sampled_from = len(df), None
        self.full_split_index = split_data(df, spec)
        search_parts = [self.full_split_index[n]
                        for n in ("train", "validation", "shadow_test")
                        if n in self.full_split_index]
        if not search_parts:
            raise ValueError("profile 没有 train/validation/shadow 搜索样本")
        search_index = search_parts[0]
        for part in search_parts[1:]:
            search_index = search_index.append(part)
        search_df = df.loc[search_index]
        self.df, sampled_from = _downsample_for_profile(
            search_df, spec, sample_cap)
        if sampled_from is not None:
            self.sampled_from = sampled_from
        # v13 phase-1 泄漏分类的审计范围是**全部列**(不只特征面——
        # 被 exclude 的列才是最需要审计的对象;旧口径审计范围=特征面,
        # 恰好漏掉 banned 候选)。分类证据的逐列计算在 phase-1 内完成;
        # phase-2 的特征面统计沿用旧口径。
        self.audit_cols = [c for c in self.df.columns
                          if c not in {spec.target_name, spec.time_column,
                                       spec.id_column, spec.treatment_column}]
        # 分析面 dtype 治理先于特征解析:decimal 金额列(object)转数值,
        # 否则 IV/WOE 按类别分桶、质量统计缺 min/max。仅分析口径,
        # splits 落盘仍用 df_full(原始 dtype)
        self.feature_cols = resolve_feature_columns(self.df, spec)
        self.df = _govern_analysis_dtypes(self.df, self.feature_cols)
        # 不能在下采样后的帧上重新随机/group 切分；必须沿用全量帧已经
        # 冻结的 split 归属，再与分析样本求交。
        self.split_index = {
            name: self.df.index.intersection(idx, sort=False)
            for name, idx in self.full_split_index.items()
            if name != "final_test"
            and len(self.df.index.intersection(idx, sort=False))
        }
        train_idx = self.split_index.get("train")
        if train_idx is None or not len(train_idx):
            raise ValueError("profile 的 train split 为空")
        self.supervised_df = self.df.loc[train_idx].copy(deep=False)

    # ---- 主入口:并行执行各分析 ----
    def _profile_logger(self) -> tuple[logging.Logger, logging.FileHandler]:
        """专用 logger:同时输出到 stderr 与 logs/profile.log(§7.3)。

        独立命名空间(不挂在 modelevo 根 logger 下),避免把 CLI 其他
        模块的日志也灌进 profile.log;handler 用完即摘除,重复调用
        run() 不叠加。
        """
        logs_dir = self.out_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        name = f"modelevo.profile.{id(self)}"
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = False
        fh = logging.FileHandler(logs_dir / "profile.log", encoding="utf-8")
        fh.setFormatter(_CNFormatter())
        sh = logging.StreamHandler()
        sh.setFormatter(_CNFormatter())
        lg.handlers = [fh, sh]
        return lg, fh

    def run(self, max_workers: int = 4) -> dict:
        started = now_iso()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "splits").mkdir(exist_ok=True)
        lg, fh = self._profile_logger()
        lg.info("profile start: rows=%d features=%d workers=%d",
                len(self.df), len(self.feature_cols), max_workers)

        def _timed(name, fn, *a, **kw):
            t0 = time.time()
            try:
                out = fn(*a, **kw)
                lg.info("%s ok in %.2fs", name, time.time() - t0)
                return out
            except Exception as exc:  # noqa: BLE001
                lg.error("%s FAILED in %.2fs: %s", name, time.time() - t0,
                         exc, exc_info=True)
                raise

        # v2.13.5:外层池至少 6 路(write_splits + 三个并行审计 + 两个分析)
        with cf.ThreadPoolExecutor(max_workers=max(6, max_workers)) as pool:
            fut_sample = pool.submit(_timed, "sample_distribution",
                                     self._sample_distribution)
            fut_splits = pool.submit(_timed, "splits_summary",
                                     self._splits_summary)
            # write_splits 涉及 IO,一并并行;用全量数据(df_full),
            # 下采样只作用于分析口径。
            # [FIX-2026-10-29] 先判能否复用:确认门裁决(kept_ambiguous /
            # reinstated_banned)按协议**不改数据面**,旧实现却把 20 GB 四平面
            # 原样重写一遍(实测 402~406 s)。复用判定要求 data_face_hash、
            # split_policy、全部特征面 sha256、labels/identity 齐全且行数一致。
            _sidx = split_data(self.df_full, self.spec)
            _splits_out = self.out_dir / "splits"
            _reuse, _why = splits_reusable(_splits_out, self.spec, _sidx,
                                           df=self.df_full)
            if _reuse:
                lg.info("write_splits 复用已有四平面(%s),跳过重写", _why)
                fut_write = pool.submit(_timed, "write_splits_reuse",
                                        lambda: json.loads(
                                            (_splits_out / "manifest.json")
                                            .read_text(encoding="utf-8")))
            else:
                lg.info("write_splits 重写四平面:%s", _why)
                fut_write = pool.submit(
                    _timed, "write_splits", write_splits, self.df_full, self.spec,
                    _splits_out, _sidx)
            # 逐特征分析(质量/IV/PSI)在独立子线程池内按特征并行,
            # 与外层 pool 隔离,避免同池嵌套等待死锁
            sub_workers = max(2, min(max_workers, 16))

            def _feature_block():
                with cf.ThreadPoolExecutor(max_workers=sub_workers) as sub:
                    feat = _timed("feature_quality", self._feature_quality, sub)
                    iv = _timed("iv_woe_table", self._iv_woe_table, sub)
                    drift = _timed("drift_profile", self._drift_profile, sub)
                return feat, iv, drift

            fut_feat = pool.submit(_feature_block)
            sample = fut_sample.result()
            splits_meta = fut_splits.result()
            feat, iv, drift = fut_feat.result()
            # v2.13.5:三个审计互不依赖、也不读 write_splits 的产物 ——
            # 旧序"等 write_splits(559s)→ leak_classify(295s)→
            # leakage_audit(214s)→ compression"全串行,profile 18 min 里
            # 有 ~9 min 是排队;现在与 write_splits 重叠并互相并行。
            # v13 phase-1:全列泄漏分类(替代旧 leakage-audit 的列分类
            # 部分;证据引用已算中间件,机械规则见 _leak_classify)
            fut_leak_cls = pool.submit(
                _timed, "leak_classify", self._leak_classify,
                iv_summary=iv["summary"], feat_summary=feat["summary"])
            # 旧 leakage-audit 保留为数据面体检(group/duplicate/label_shift
            # 等非列分类项);列分类 critical 已由 phase-1 承担
            fut_leak = pool.submit(
                _timed, "leakage_audit", self._leakage_audit,
                iv_summary=iv["summary"], feat_summary=feat["summary"],
                splits_summary=splits_meta, drift_summary=drift["summary"])
            # 信息压缩检测(v8):data_view 候选自动生成的依据
            fut_comp = pool.submit(
                _timed, "compression_audit", self._compression_audit)
            leak_class = fut_leak_cls.result()
            leakage = fut_leak.result()
            compression = fut_comp.result()
            manifest = fut_write.result()

        lg.info("profile finished: rows=%d features=%d",
                len(self.df), len(self.feature_cols))
        fh.flush()
        lg.handlers.clear()  # 摘除 handler,防止后续调用重复写文件

        profile_json = {
            "created_at": started,
            "finished_at": now_iso(),
            "task_id": self.spec.task_id,
            "n_rows": int(self.n_original),
            "n_rows_profiled": int(len(self.df)),
            "sampled_from": self.sampled_from,
            "n_features": len(self.feature_cols),
            # v2.15 抽象任务指纹：data_loader 加载后 df_full.columns 的排序 hash。
            # 用于 derive_abstract_task_id —— 同一份数据底座的所有 run 得到
            # 相同的 raw_columns_hash，不依赖 feature-quality.csv 产出时机，
            # 也不受特征面探索过程中新增列的影响。
            # 口径 = 原生全量面(df_full，含 identity/time/被排除列)，
            # 刻意不用 resolve_feature_columns 后的建模面——原生面才是
            # "数据底座"的定义；建模面会随 banned/exclude 演进漂移。
            # v2.15.10:loader 派生列(spec data.derived_columns)不算底座,
            # 见 raw_columns_fingerprint。
            **raw_columns_fingerprint(self.df_full.columns, self.spec),
            "sample_summary": sample,
            "splits_summary": splits_meta,
            "feature_summary": feat["summary"],
            "iv_summary": iv["summary"],
            "drift_summary": drift["summary"],
            "leak_classification": leak_class,
            "leakage": leakage,
            "splits_manifest": manifest,
        }
        # compression_audit 独立成键:disabled 时直接无键(向后兼容)
        if compression.get("enabled", True):
            profile_json["compression_audit"] = compression
        dump_json(self.out_dir / "sample-profile.json", profile_json)
        # feature-quality.csv:质量 + IV/单变量AUC + 跨split PSI 单表视图
        quality = self._merge_quality_table(feat["table"], iv["table"],
                                            drift["table"])
        quality = self._annotate_iv_psi_levels(quality)
        profile_json["quality_levels"] = {
            "iv_level_counts": quality["iv_level"].value_counts().to_dict(),
            "stability_flag_counts":
                quality["stability_flag"].value_counts().to_dict(),
            "suspicious_features": quality.loc[
                quality["iv_level"] == "suspicious", "feature"].tolist(),
            "drift_features": quality.loc[
                quality["stability_flag"] != "OK", "feature"].tolist(),
        }
        quality.to_csv(self.out_dir / "feature-quality.csv", index=False)
        quality.to_parquet(self.out_dir / "feature-quality.parquet",
                           index=False)
        # WOE 分箱明细(长表:feature × bin)
        if iv.get("woe_rows"):
            pd.DataFrame(iv["woe_rows"]).to_csv(
                self.out_dir / "feature-woe.csv", index=False)
        drift["table"].to_parquet(self.out_dir / "drift-profile.parquet",
                                  index=False)
        drift["table"].to_csv(self.out_dir / "drift-profile.csv", index=False)
        (self.out_dir / "leakage-audit.md").write_text(
            _render_leakage_md(leakage), encoding="utf-8")
        # v19.2:摘要里显式记录"分析面基于下采样",并在两处口径数字不一致
        # 时标红。历史事故:splits-summary 报 train 1,117,911 行(2M
        # 采样面)、splits/manifest 报 2,235,937 行(全量落盘面)、调用方
        # 又各自带着自己算的行数——三个互不相等的数字同时出现在产物里,
        # 而且没有一个能在 split 构成异常时报警(实体重叠率 6.52% 反被
        # 读成"泄漏")。
        splits_meta["analysis_frame"] = {
            "sampled_from": self.sampled_from,
            "rows_analyzed": int(len(self.df)),
            "note": ("splits-summary 的行数/正率基于分析面(可能已下采样);"
                     "splits/ 与 manifest.json 的行数才是训练数据面(全量)"
                     if self.sampled_from else
                     "分析面 = 全量,与 splits/ 行数一致"),
        }
        (self.out_dir / "splits-summary.json").write_text(
            __import__("json").dumps(splits_meta, ensure_ascii=False, indent=2,
                                     default=str), encoding="utf-8")
        (self.out_dir / "splits-summary.md").write_text(
            _render_splits_md(splits_meta), encoding="utf-8")
        (self.out_dir / "sample-profile.md").write_text(
            _render_profile_md(profile_json), encoding="utf-8")
        # v18/P1-1:信息源清单——"信息已榨干"的反向对账面。reparse 源
        # 与强 IV 族的消费状态 + 粒度空间 checklist,planner-context
        # 第 12 节呈现,information_channels_exhausted 停止出口对账。
        try:
            from .inventory import build_inventory

            inventory = build_inventory(self.out_dir.parent, self.spec)
            dump_yaml(self.out_dir / "information-inventory.yaml", inventory)
            profile_json["information_inventory"] = inventory.get("summary")
        except Exception as exc:  # noqa: BLE001
            logger.warning("information-inventory 产出失败(不阻塞 profile): %s",
                           exc)
        return profile_json

    def _annotate_iv_psi_levels(self, quality: pd.DataFrame) -> pd.DataFrame:
        """IV 等级 + PSI 稳定性判别(§7.1 扩展,规则透明可审计)。

        - iv_level: useless(<0.02) / weak(0.02~0.1) / medium(0.1~0.3) /
          strong(0.3~0.5) / suspicious(≥0.5,反泄漏复核清单)
          回归任务 iv 列为 |Spearman corr|,套用同一等级刻度
          (corr 与 IV 皆为 [0,1] 无向强度,量级可比)。
        - {psi_col}_level: stable(<0.02) / watch(0.02~0.1) /
          unstable(0.1~0.25) / broken(≥0.25)
        - stability_flag: OK / DRIFT_VALIDATION / DRIFT_FINAL_TEST /
          DRIFT_ADJACENT(任一口径 unstable/broken 即标记)
        """
        out = quality.copy()

        def _iv_level(v: float) -> str:
            if pd.isna(v):
                return "no_value"
            if v >= 0.5:
                return "suspicious"
            if v >= 0.3:
                return "strong"
            if v >= 0.1:
                return "medium"
            if v >= 0.02:
                return "weak"
            return "useless"

        def _psi_level(v: float) -> str:
            if pd.isna(v):
                return "no_value"
            if v >= 0.25:
                return "broken"
            if v >= 0.1:
                return "unstable"
            if v >= 0.02:
                return "watch"
            return "stable"

        out["iv_level"] = out["iv"].map(_iv_level)
        psi_cols = [c for c in ("psi_train_vs_validation",
                                "psi_train_vs_final_test",
                                "psi_max_adjacent_months")
                    if c in out.columns]
        for c in psi_cols:
            out[f"{c}_level"] = out[c].map(_psi_level)

        def _flag(r) -> str:
            flags = []
            if str(r.get("psi_train_vs_validation_level", "")) in (
                    "unstable", "broken"):
                flags.append("DRIFT_VALIDATION")
            if str(r.get("psi_train_vs_final_test_level", "")) in (
                    "unstable", "broken"):
                flags.append("DRIFT_FINAL_TEST")
            if str(r.get("psi_max_adjacent_months_level", "")) in (
                    "unstable", "broken"):
                flags.append("DRIFT_ADJACENT")
            return "|".join(flags) if flags else "OK"

        out["stability_flag"] = out.apply(_flag, axis=1)
        return out

    def _merge_quality_table(self, quality: pd.DataFrame, iv: pd.DataFrame,
                             drift: pd.DataFrame) -> pd.DataFrame:
        """单表视图:质量列 + iv/univariate_auc + 各 split PSI + 跨月 PSI。

        drift 行形如 (feature, comparison, psi);透视成
        psi_train_vs_validation / psi_train_vs_shadow_test / ... 与
        psi_max_across_months(跨 date 漂移峰值)。
        """
        out = quality.copy()
        if len(iv):
            out = out.merge(iv[["feature", "iv", "univariate_auc"]],
                            on="feature", how="left")
        else:
            out["iv"] = float("nan")
            out["univariate_auc"] = float("nan")
        if len(drift):
            d = drift.copy()
            # 跨 date PSI:每特征峰值(区分 first/adjacent 口径)
            date_rows = d[d["kind"].astype(str).str.startswith("date",
                                                               na=False)] \
                if "kind" in d.columns else pd.DataFrame()
            if len(date_rows):
                for kind, col_name in (
                        ("date_first", "psi_max_vs_train_first_month"),
                        ("date_adjacent", "psi_max_adjacent_months")):
                    sub = date_rows[date_rows["kind"] == kind]
                    if len(sub):
                        peak = sub.groupby("feature")["psi"].max().rename(
                            col_name).reset_index()
                        out = out.merge(peak, on="feature", how="left")
            # 跨 split PSI:透视成列
            split_rows = d[d["kind"] == "split"] if "kind" in d.columns else d
            if len(split_rows):
                piv = split_rows.pivot_table(
                    index="feature", columns="comparison", values="psi",
                    aggfunc="first")
                piv.columns = [f"psi_{c}" for c in piv.columns]
                out = out.merge(piv, on="feature", how="left")
        return out

    # ---- 样本分布 ----
    def _sample_distribution(self) -> dict:
        # 任何目标统计都只基于 train；其余 split 只公开行数/时间范围。
        df, spec = self.supervised_df, self.spec
        target = spec.target_name
        is_cls = spec.model_type in ("binary_classification", "multiclass",
                                     "ordinal", "uplift")
        out: dict = {"total_rows": int(len(df))}
        if is_cls:
            vc = df[target].value_counts(dropna=False)
            out["label_counts"] = {str(k): int(v) for k, v in vc.items()}
            out["label_rate"] = float((df[target] == 1).mean()) \
                if spec.model_type in ("binary_classification", "uplift") else None
        else:
            yv = pd.to_numeric(df[target], errors="coerce")
            desc = yv.describe()
            out["target_stats"] = {k: float(v) for k, v in desc.items()}
            # 重尾诊断(回归):分位数刻度差异大时提示对数/分段建模
            q01, q50, q99 = yv.quantile([0.01, 0.5, 0.99])
            if all(np.isfinite(v) for v in (q01, q50, q99)) and q99 > q50:
                out["target_skew_hint"] = {
                    "q01": float(q01), "q50": float(q50), "q99": float(q99),
                    "q99_over_q50": float(q99 / q50) if q50 > 0 else None,
                }
        tcol = spec.time_column
        if tcol and tcol in df.columns:
            by_date = df.groupby(df[tcol].astype(str).str[:7]).size()
            out["rows_by_month"] = {str(k): int(v) for k, v in by_date.items()}
            if is_cls:
                out["label_rate_by_month"] = {
                    str(m): float((g == 1).mean())
                    for m, g in df.groupby(df[tcol].astype(str).str[:7])[target]
                }
            else:
                yv = pd.to_numeric(df[target], errors="coerce")
                out["target_mean_by_month"] = {
                    str(m): float(g.mean())
                    for m, g in yv.groupby(df[tcol].astype(str).str[:7])
                }
        if spec.id_column and spec.id_column in df.columns:
            out["unique_entities"] = int(df[spec.id_column].nunique())
            out["entity_dup_rate"] = float(
                1 - df[spec.id_column].nunique() / max(len(df), 1))
        out["duplicate_row_rate"] = float(df.duplicated().mean())
        # uplift:treatment 分布与 ATE 审计(随机实验缺失时 ATE 解读要小心)
        tcol_u = spec.treatment_column
        if tcol_u and tcol_u in df.columns:
            w = df[tcol_u]
            out["treatment_summary"] = {
                "column": tcol_u,
                "treatment_rate": float((w == 1).mean()),
                "n_treated": int((w == 1).sum()),
                "n_control": int((w == 0).sum()),
                "non_binary_values": int(
                    len(w) - (w == 1).sum() - (w == 0).sum()
                    - w.isna().sum()),
            }
            if (w == 1).sum() and (w == 0).sum():
                y_u = df[target]
                out["treatment_summary"]["ate_observed"] = float(
                    (y_u[w == 1] == 1).mean() - (y_u[w == 0] == 1).mean())
                out["treatment_summary"]["response_rate_treated"] = float(
                    (y_u[w == 1] == 1).mean())
                out["treatment_summary"]["response_rate_control"] = float(
                    (y_u[w == 0] == 1).mean())
        by_split = {}
        for name, idx in self.split_index.items():
            sub = self.df.loc[idx]
            entry = {"rows": int(len(sub))}
            if name == "train" and is_cls:
                entry["label_rate"] = float((sub[target] == 1).mean()) \
                    if spec.model_type in ("binary_classification", "uplift") else None
            elif name == "train" and not is_cls:
                yv = pd.to_numeric(sub[target], errors="coerce")
                entry["target_mean"] = float(yv.mean()) if len(yv) else None
            else:
                entry["labels_hidden_from_profile"] = True
            # uplift:分 split 的处置占比与组内响应率(split 间实验条件
            # 是否一致,直接决定 uplift 指标可比性)
            if name == "train" and tcol_u and tcol_u in sub.columns:
                sw = sub[tcol_u]
                entry["treatment_rate"] = float((sw == 1).mean()) \
                    if len(sw) else None
                if (sw == 1).sum() and (sw == 0).sum():
                    entry["response_rate_treated"] = float(
                        (sub.loc[sw == 1, target] == 1).mean())
                    entry["response_rate_control"] = float(
                        (sub.loc[sw == 0, target] == 1).mean())
            by_split[name] = entry
        out["by_split"] = by_split
        return out

    # ---- 切分摘要 ----
    def _splits_summary(self) -> dict:
        df, spec = self.df_full, self.spec
        tcol = spec.time_column
        out: dict = {"policy": spec.data.get("split_policy")}
        rows = []
        for name in ("train", "validation", "shadow_test", "final_test"):
            idx = self.full_split_index.get(name)
            if idx is None or not len(idx):
                continue
            sub = df.loc[idx]
            entry = {"split": name, "rows": int(len(sub))}
            if name == "train" and spec.model_type in (
                    "binary_classification", "uplift"):
                entry["positive_rate"] = float((sub[spec.target_name] == 1).mean())
            elif name == "train" and spec.model_type == "regression":
                yv = pd.to_numeric(sub[spec.target_name], errors="coerce")
                entry["target_mean"] = float(yv.mean())
                entry["target_std"] = float(yv.std())
            else:
                entry["labels_hidden_from_profile"] = True
            if tcol and tcol in sub.columns:
                tv = sub[tcol].astype(str)
                entry["time_min"], entry["time_max"] = str(tv.min()), str(tv.max())
            rows.append(entry)
        out["splits"] = rows
        if tcol and tcol in df.columns and len(rows) >= 2:
            # 实体跨 split 泄漏(group leakage)
            idc = spec.id_column
            if idc and idc in df.columns:
                overlap = {}
                names = [r["split"] for r in rows]
                for i, a in enumerate(names):
                    for b in names[i + 1:]:
                        sa = set(df.loc[self.full_split_index[a], idc])
                        sb = set(df.loc[self.full_split_index[b], idc])
                        if sa and sb:
                            overlap[f"{a}∩{b}"] = len(sa & sb) / max(len(sa | sb), 1)
                out["entity_overlap_rate"] = overlap
        return out

    # ---- 特征质量:逐特征并行 ----
    def _feature_quality(self, pool: cf.Executor | None = None) -> dict:
        records = self._map_features(self._quality_one_col, pool)
        table = pd.DataFrame(records)
        summary = {
            "n_features": len(records),
            "constant_features": table.loc[table["is_constant"], "feature"].tolist(),
            "high_missing_features": table.loc[
                table["missing_rate"] > 0.5, "feature"].tolist(),
            "high_cardinality_features": table.loc[
                table["high_cardinality"], "feature"].tolist(),
            # v2.17.0(D9):id_leakage 的报警集(高基数 ∧ 整数/字符串型 ∧ 像标识符)
            "id_like_features": table.loc[
                table["id_like"], "feature"].tolist()
            if "id_like" in table.columns else [],
        }
        return {"table": table, "summary": summary}

    def _quality_one_col(self, col: str) -> dict:
        df = self.df
        s = df[col]
        n = len(s)
        rec: dict = {
            "feature": col,
            "dtype": str(s.dtype),
            "coverage": float(s.notna().mean()),
            "missing_rate": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
            "unique_ratio": float(s.nunique(dropna=True) / max(n, 1)),
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            rec.update({
                "min": float(desc["min"]), "max": float(desc["max"]),
                "mean": float(desc["mean"]), "std": float(desc["std"]),
            })
            q1, q99 = s.quantile([0.01, 0.99])
            rec["outlier_ratio"] = float(
                ((s < q1 - 3 * (q99 - q1)) | (s > q99 + 3 * (q99 - q1))).mean())
            rec["is_constant"] = bool(rec["n_unique"] <= 1)
        else:
            rec["is_constant"] = bool(rec["n_unique"] <= 1)
            rec["top_value_share"] = float(
                s.value_counts(normalize=True, dropna=True).iloc[0]
                if rec["n_unique"] else 0.0)
        rec["high_cardinality"] = bool(
            rec["unique_ratio"] > LEAKAGE_ID_LIKE_UNIQUE_RATIO and rec["n_unique"] > 100)
        # v2.17.0(D9):"疑似 ID 列"只看整数型/字符串型、高唯一率且像标识符
        # 的列;非整数值的连续浮点列唯一率≈1 是常态(金额/评分),不是 ID
        # (实测 23 个连续浮点特征被报 id_leakage warning,全部是噪声)
        rec["id_like"] = bool(rec["high_cardinality"]
                              and self._looks_like_identifier(col, s))
        return rec

    @staticmethod
    def _looks_like_identifier(col: str, s: pd.Series) -> bool:
        """[v2.17.0 D9] 高唯一率列是否"像标识符"(整数型或字符串型才可能是)。

        - 浮点列:非整数值 → 否(连续值);全为整数值(含 NaN 导致的 float 存储)
          按整数处理;
        - 整数型:列名含标识符词(id/no/key/uuid/code/编号…)或数位长度集中
          (≥80% 行同一位数,编码号段的典型形态);
        - 字符串型:非复合文本(_text_shape_evidence.looks_composite 为假)、
          非数值字面量的小数串,且 ≥90% 抽样值是无空白的标识符字符串
          ([A-Za-z0-9_-:.#/]);或列名含标识符词;
        - 其它 dtype(bool/datetime/category…)只认列名提示。
        """
        name_hint = bool(ID_NAME_HINT_RE.search(str(col)))
        try:
            if pd.api.types.is_bool_dtype(s):
                return False
            if pd.api.types.is_numeric_dtype(s):
                v = pd.to_numeric(s, errors="coerce").dropna()
                if len(v) > ID_SHAPE_SCAN_ROWS:
                    v = v.iloc[:ID_SHAPE_SCAN_ROWS]
                if not len(v):
                    return False
                arr = v.to_numpy(dtype=float)
                if pd.api.types.is_float_dtype(s) and not bool(
                        np.all(np.isfinite(arr) & (np.mod(arr, 1.0) == 0))):
                    return False                    # 非整数值的连续浮点列
                if name_hint:
                    return True
                digits = pd.Series(np.floor(np.log10(np.abs(arr) + 1.0))
                                   ).value_counts(normalize=True)
                return bool(len(digits) and float(digits.iloc[0]) >= 0.8)
            if pd.api.types.is_object_dtype(s) or \
                    pd.api.types.is_string_dtype(s):
                txt = s.dropna().astype(str)
                if len(txt) > ID_SHAPE_SCAN_ROWS:
                    txt = txt.iloc[:ID_SHAPE_SCAN_ROWS]
                if not len(txt):
                    return False
                num = pd.to_numeric(txt, errors="coerce")
                if float(num.notna().mean()) >= 0.9:
                    arr = num.dropna().to_numpy(dtype=float)
                    if not bool(np.all(np.isfinite(arr)
                                       & (np.mod(arr, 1.0) == 0))):
                        return False                # 小数字面量串 = 连续值
                if SampleProfile._text_shape_evidence(s).get(
                        "looks_composite"):
                    return False                    # 复合文本走 reparse 口径
                if name_hint:
                    return True
                return bool(float(txt.str.fullmatch(ID_TOKEN_RE).mean())
                            >= 0.9)
        except Exception:  # noqa: BLE001 —— 判定失败按"不像 ID"降级
            return False
        return name_hint

    # ---- IV/WOE 或回归相关性:逐特征并行 ----
    def _iv_woe_table(self, pool: cf.Executor | None = None) -> dict:
        spec = self.spec
        if spec.model_type == "regression":
            results = self._map_features(self._reg_one_col, pool)
            rows = [{"feature": r["feature"], "iv": r["corr"],
                     "univariate_auc": float("nan")}
                    for r in results]
            table = pd.DataFrame(rows).sort_values("iv", ascending=False)
            summary = {
                "supported": True,
                "mode": "regression_correlation",
                "top_features": table.head(20)[["feature", "iv"]]
                    .rename(columns={"iv": "corr"}).to_dict("records"),
                "corr_gt_03": table.loc[
                    table["iv"].abs() > 0.3, "feature"].tolist()[:50],
                "corr_gt_095_leakage": table.loc[
                    table["iv"].abs() > 0.95, "feature"].tolist()[:50],
                "suspect_univariate_auc_gt_085": [],
            }
            return {"table": table, "summary": summary, "woe_rows": []}
        if spec.model_type not in ("binary_classification", "uplift"):
            return {"table": pd.DataFrame(), "summary": {"supported": False},
                    "woe_rows": []}
        if spec.model_type == "uplift":
            return self._uplift_iv_table(pool)
        results = self._map_features(self._iv_one_col, pool)
        rows = [r["row"] for r in results]
        woe_rows = []
        for r in results:
            for bin_row in r["woe_table"]:
                woe_rows.append({"feature": r["row"]["feature"], **bin_row})
        table = pd.DataFrame(rows).sort_values("iv", ascending=False)
        summary = {
            "supported": True,
            "top_features": table.head(20)[["feature", "iv", "univariate_auc"]]
            .to_dict("records"),
            "iv_gt_02": table.loc[table["iv"] > 0.2, "feature"].tolist()[:50],
            "suspect_univariate_auc_gt_085": table.loc[
                table["univariate_auc"] > LEAKAGE_HIGH_UNIVARIATE_AUC, "feature"
            ].tolist(),
        }
        return {"table": table, "summary": summary, "woe_rows": woe_rows}

    def _iv_one_col(self, col: str) -> dict:
        spec = self.spec
        df = self.supervised_df
        y = (df[spec.target_name] == 1).values.astype(int)
        try:
            iv, woe_table = _iv_woe(y, df[col].values)
        except Exception as exc:  # noqa: BLE001
            logger.warning("IV 计算失败 %s: %s", col, exc)
            iv, woe_table = float("nan"), []
        univ_auc = float("nan")
        if pd.api.types.is_numeric_dtype(df[col]):
            m = df[col].notna()
            if m.sum() > 100 and y[m].sum() > 0 and (1 - y[m]).sum() > 0:
                univ_auc = roc_auc(y[m], df.loc[m, col].values)
                if univ_auc < 0.5:
                    univ_auc = 1 - univ_auc  # 方向归一
        return {"row": {"feature": col, "iv": iv, "univariate_auc": univ_auc,
                        "coverage": float(df[col].notna().mean())},
                "woe_table": woe_table}

    def _uplift_iv_table(self, pool: cf.Executor | None = None) -> dict:
        """uplift:IV 按 treated/control 分别计算,iv_diff 找调节特征。

        - iv_treated / iv_control:各组内对 target 的判别力;
        - iv = 全样本 IV(转化率视角,与 binary 同口径);
        - iv_diff = |iv_treated − iv_control|:处置改变了特征与转化的
          关系 → uplift 调节(moderator)候选,Planner 优先造交互特征。
        单变量 AUC 口径同 binary(全样本对 target)。
        """
        results = [r for r in self._map_features(self._uplift_iv_one_col, pool)
                   if r is not None]
        woe_rows = []
        for r in results:
            for bin_row in r.pop("woe_table"):
                woe_rows.append({"feature": r["feature"], **bin_row})
        table = pd.DataFrame(results)
        if len(table):
            table = table.sort_values("iv_diff", ascending=False)
        summary: dict = {"supported": True, "mode": "uplift_split_iv",
                         "top_moderators": [], "top_iv_features": [],
                         "iv_diff_gt_01": [],
                         "suspect_univariate_auc_gt_085": []}
        if len(table):
            summary["top_moderators"] = table.head(20)[
                ["feature", "iv_treated", "iv_control", "iv_diff"]
            ].to_dict("records")
            summary["top_iv_features"] = table.sort_values(
                "iv", ascending=False).head(20)[
                ["feature", "iv", "univariate_auc"]].to_dict("records")
            summary["iv_diff_gt_01"] = table.loc[
                table["iv_diff"] > 0.1, "feature"].tolist()[:50]
            summary["suspect_univariate_auc_gt_085"] = table.loc[
                table["univariate_auc"] > LEAKAGE_HIGH_UNIVARIATE_AUC,
                "feature"].tolist()
        return {"table": table, "summary": summary, "woe_rows": woe_rows}

    def _uplift_iv_one_col(self, col: str) -> dict | None:
        spec = self.spec
        df = self.supervised_df
        tcol = spec.treatment_column
        w = (df[tcol] == 1).values
        y = (df[spec.target_name] == 1).values.astype(int)
        x = df[col].values
        try:
            iv_all, woe_table = _iv_woe(y, x)
            iv_t, _ = _iv_woe(y[w], x[w])
            iv_c, _ = _iv_woe(y[~w], x[~w])
        except Exception as exc:  # noqa: BLE001
            logger.warning("uplift IV 计算失败 %s: %s", col, exc)
            return None
        univ_auc = float("nan")
        if pd.api.types.is_numeric_dtype(df[col]):
            m = df[col].notna()
            if m.sum() > 100 and y[m].sum() > 0 and (1 - y[m]).sum() > 0:
                univ_auc = roc_auc(y[m], df.loc[m, col].values)
                if univ_auc < 0.5:
                    univ_auc = 1 - univ_auc  # 方向归一
        iv_diff = abs(iv_t - iv_c) if (np.isfinite(iv_t) and
                                       np.isfinite(iv_c)) else float("nan")
        return {"feature": col, "iv": iv_all, "iv_treated": iv_t,
                "iv_control": iv_c, "iv_diff": iv_diff,
                "univariate_auc": univ_auc,
                "coverage": float(df[col].notna().mean()),
                "woe_table": woe_table}

    def _reg_one_col(self, col: str) -> dict:
        """回归任务的单特征相关性(替代 IV):|Spearman corr|。"""
        df = self.supervised_df
        y = pd.to_numeric(df[self.spec.target_name],
                          errors="coerce").values.astype(float)
        y = np.nan_to_num(y, nan=np.nanmedian(y))
        try:
            corr = _spearman_corr(y, df[col].values)
        except Exception as exc:  # noqa: BLE001
            logger.warning("回归相关性计算失败 %s: %s", col, exc)
            corr = float("nan")
        return {"feature": col, "corr": corr,
                "coverage": float(df[col].notna().mean())}

    # ---- 逐特征并行工具:无 pool 时同步循环(小样本不值得开线程)----
    def _map_features(self, fn, pool: cf.Executor | None) -> list:
        if pool is not None and len(self.feature_cols) >= 8:
            return list(pool.map(fn, self.feature_cols))
        return [fn(c) for c in self.feature_cols]

    # ---- 漂移 PSI:跨 split 与跨 date,比较粒度并行 ----
    def _drift_profile(self, pool: cf.Executor | None = None) -> dict:
        """两类比较(§7.1 不同日期 PSI / 不同 split PSI):

        - 跨 split:train vs validation/shadow_test/final_test(基准=train);
        - 跨 date:以 train 时间窗内各月为基准序列,滚动对比后续月份
          (相邻月 adjacent + 相对 train 首月 first,反映时间趋势而非
          单点差异)。
        每个 (comparison × feature) 组为独立工作项,并行计算。
        """
        df, spec = self.df, self.spec
        tcol = spec.time_column
        # v19(req-t7-drawdown 核查回写):旧硬编码 [:300]/[:150] 把
        # 1493 特征的 1193 列 PSI 截成 NaN,且按列序截断=系统性偏好
        # 排在前面的基础列。修复:上限提额(默认 2000,config 可调
        # profile.psi_max_columns)并按 IV 降序取头部(IV 缺失时回退
        # 列序);被截断清单记 manifest,截断不再静默。
        _psi_cap = int(self.config.get("profile.psi_max_columns", 2000)
                       or 2000)
        numeric_all = [c for c in self.feature_cols
                       if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_all) > _psi_cap:
            _iv_col = None
            try:
                _fq = self.out_dir / "feature-quality.csv"
                if _fq.exists():
                    import pandas as _pd

                    _fq_df = _pd.read_csv(_fq)
                    if {"feature", "iv"} <= set(_fq_df.columns):
                        _iv_map = dict(zip(_fq_df["feature"],
                                           _pd.to_numeric(_fq_df["iv"],
                                                          errors="coerce")
                                          .fillna(-1)))
                        numeric_all.sort(key=lambda c: -_iv_map.get(c, -1))
                        _iv_col = "iv"
            except Exception:  # noqa: BLE001
                pass
            numeric_cols = numeric_all[:_psi_cap]
            logger.warning(
                "PSI 列数上限 %d(实际 %d,按 %s 降序取头部;被截断 %d 列"
                "不计入 drift-profile,可用 profile.psi_max_columns 调额)",
                _psi_cap, len(numeric_all), _iv_col or "column_order",
                len(numeric_all) - _psi_cap)
        else:
            numeric_cols = numeric_all
        train_idx = self.split_index.get("train")
        base = df.loc[train_idx, numeric_cols] if train_idx is not None else df

        # ---- 构造工作项:(comparison, kind, base_values, cur_values, cols) ----
        jobs: list[dict] = []
        for name in ("validation", "shadow_test", "final_test"):
            idx = self.split_index.get(name)
            if idx is None or not len(idx):
                continue
            cur = df.loc[idx]
            jobs.append({"comparison": f"train_vs_{name}", "kind": "split",
                         "base": base, "cur": cur, "cols": numeric_cols})

        if tcol and tcol in df.columns:
            months = sorted(df[tcol].astype(str).str[:7].unique())
            if 2 <= len(months) <= 60:
                month_data = {m: df[df[tcol].astype(str).str[:7] == m]
                              for m in months}
                train_months = sorted({
                    str(v)[:7] for v in df.loc[
                        train_idx, tcol].astype(str)}) if train_idx is not None \
                    else months[:1]
                first_m = train_months[0] if train_months else months[0]
                cols_date = numeric_cols  # v19:与 split PSI 同额(不再二次截 150)
                ref_first = month_data.get(first_m)
                for m in months:
                    cur = month_data[m]
                    if len(cur) < 100:
                        continue
                    if ref_first is not None and len(ref_first) >= 100 \
                            and m != first_m:
                        jobs.append({
                            "comparison": f"{first_m}_vs_{m}",
                            "kind": "date_first",
                            "base": ref_first, "cur": cur, "cols": cols_date})
                    mi = months.index(m)
                    if mi > 0:
                        prev = month_data.get(months[mi - 1])
                        if prev is not None and len(prev) >= 100:
                            jobs.append({
                                "comparison": f"{months[mi - 1]}_vs_{m}",
                                "kind": "date_adjacent",
                                "base": prev, "cur": cur, "cols": cols_date})

        def _run_job(job: dict) -> list[dict]:
            out = []
            for col in job["cols"]:
                out.append({
                    "feature": col, "comparison": job["comparison"],
                    "psi": _psi(job["base"][col].values, job["cur"][col].values),
                    "kind": job["kind"]})
            return out

        if pool is not None and len(jobs) >= 4:
            chunks = list(pool.map(_run_job, jobs))
            rows = [r for chunk in chunks for r in chunk]
        else:
            rows = [r for job in jobs for r in _run_job(job)]

        table = pd.DataFrame(rows)
        summary: dict = {"n_comparisons": int(len(table))}
        if len(table):
            sev = table.dropna(subset=["psi"])
            summary["features_psi_gt_010_vs_validation"] = sev.loc[
                (sev["comparison"] == "train_vs_validation") & (sev["psi"] > 0.1),
                "feature"].tolist()[:50]
            summary["max_psi"] = float(sev["psi"].max()) if len(sev) else None
            # 跨 date 峰值(供 feature-quality 单表视图)
            date_rows = sev[sev["kind"].str.startswith("date")]
            if len(date_rows):
                peak = date_rows.groupby("feature")["psi"].max()
                summary["features_psi_gt_010_across_dates"] = peak[
                    peak > 0.1].index.tolist()[:50]
                summary["max_psi_across_dates"] = float(peak.max())
        # v2.17.2(E2):reparse/文本列 token 级漂移,kind="token" 行追加在
        # 数值 PSI 行之后(既有列与 n_comparisons 口径不变;token 行 psi=NaN,
        # 不进上面的 PSI 汇总与 feature-quality 透视)
        tok = self._token_drift()
        summary["token_drift"] = tok["summary"]
        collapsed = tok["summary"].get("collapsed_tokens") or []
        if collapsed:
            summary["tokens_coverage_collapsed"] = list(dict.fromkeys(
                f"{c['column']}:{c['token']}" for c in collapsed))[:50]
        if tok["rows"]:
            tok_df = pd.DataFrame(tok["rows"])
            table = tok_df if not len(table) else pd.concat(
                [table, tok_df], ignore_index=True, sort=False)
        return {"table": table, "summary": summary}

    # ---- v2.17.2(E2):token 级漂移(reparse / 文本列 × 时间列月份)----
    def _cfg(self, key: str, default):
        try:
            v = self.config.get(key, default)
        except Exception:  # noqa: BLE001
            v = None
        return default if v is None else v

    def _token_drift_columns(self, rx: re.Pattern, sep: str) -> list[str]:
        """spec reparse 列(文本型)+ 形态像 token 序列的文本特征列(有上限)。

        文本特征列的初筛只看头部 TOKEN_DRIFT_SHAPE_ROWS 行:平均分隔符数
        > TEXT_SHAPE_DELIM_MIN 且 token 复用倍数 > TEXT_SHAPE_VOCAB_REUSE_MIN
        (与 _text_shape_evidence 同判据,分隔符改用 token 漂移的分隔字符集),
        普通类别列/自由文本不进。
        """
        df = self.df

        def _is_text(c) -> bool:
            return pd.api.types.is_object_dtype(df[c]) \
                or pd.api.types.is_string_dtype(df[c])

        cols = [c for c in (self.spec.reparse_columns or [])
                if c in df.columns and _is_text(c)]
        cap = int(self._cfg("profile.token_drift_max_extra_columns",
                            TOKEN_DRIFT_MAX_EXTRA_COLUMNS))
        if cap <= 0:
            return cols
        sep_cls = "[%s]" % "".join(re.escape(c) for c in sep)
        extra: list[str] = []
        for c in self.feature_cols:
            if len(extra) >= cap:
                break
            if c in cols or c not in df.columns or not _is_text(c):
                continue
            try:
                txt = df[c].iloc[:TOKEN_DRIFT_SHAPE_ROWS].dropna().astype(str)
                if not len(txt) or float(txt.str.count(sep_cls).mean()) \
                        <= TEXT_SHAPE_DELIM_MIN:
                    continue
                heads = [h for v in txt for h in rx.findall(v) if h]
                vocab = len(set(heads))
                if vocab and len(heads) / vocab > TEXT_SHAPE_VOCAB_REUSE_MIN:
                    extra.append(c)
            except Exception:  # noqa: BLE001  初筛失败只是不纳入
                continue
        return cols + extra

    def _token_drift(self) -> dict:
        """reparse/文本列按月的 token 级漂移(kind="token" 行 + 摘要)。

        口径(基准 = train 行;比较 = 时间列每个月,含 validation/shadow 月):
        - 切分:`sep` 字符集切 token,取 `head_sep` 之前的头部作 token 名
          (与 _text_shape_evidence 的逗号切分一致;事件 token 取 '@' 前事件名);
        - 成本:每月确定性等距抽样 ≤ rows_per_month 行(与分析面总行数无关,
          1M 行 × 2 万字符的序列列也只扫 月数×5000 行;单次 findall 在 C 层
          完成切分与取头,Counter 计数同在 C 层);
        - token 行:train 行覆盖率 top-K token × 月,写 coverage(该月含此
          token 的行占比)/ baseline_coverage / coverage_ratio;edge_frac>0 时
          另写 tail_coverage / head_coverage(每行尾/首 edge_frac 比例 token
          内的覆盖,序列近端窗口——方向未知时两端都看)与各自 train 基线;
        - `__vocab__` 行:每月 tokens_per_row、非空行占比(coverage)、相对
          train 的 JSD(top-K + __other__ 的出现频次分布)、新 token 出现份额
          (train 词表外)、消失 token 份额(train 出现质量中本月为 0 的部分);
        - 塌缩(逐口径 scope=row/tail/head):基线覆盖率 ≥ min_baseline_coverage,
          且在出现过(≥ 基线×ratio)的月份之后某月跌破 基线×ratio(row 用
          collapse_ratio=0.1,窗口用 edge_collapse_ratio=0.5)→ flag=
          collapse / collapse_tail / collapse_head,进摘要 collapsed_tokens
          与 leakage-audit 的 token_drift 告警。新出现的 token(早月为 0)
          不算塌缩,由 new_token_share 体现。
        任何异常只降级为空结果(写 summary.error),不阻塞 profile。
        """
        from collections import Counter

        sep = str(self._cfg("profile.token_drift_sep", TOKEN_DRIFT_SEP)
                  or TOKEN_DRIFT_SEP)
        head_sep = str(self._cfg("profile.token_drift_head_sep",
                                 TOKEN_DRIFT_HEAD_SEP))
        params = {
            "top_k": int(self._cfg("profile.token_drift_top_k",
                                   TOKEN_DRIFT_TOP_K)),
            "rows_per_month": int(self._cfg("profile.token_drift_rows_per_month",
                                            TOKEN_DRIFT_ROWS_PER_MONTH)),
            "min_baseline_coverage": float(self._cfg(
                "profile.token_drift_min_baseline_coverage",
                TOKEN_DRIFT_MIN_BASELINE_COVERAGE)),
            "collapse_ratio": float(self._cfg(
                "profile.token_drift_collapse_ratio",
                TOKEN_DRIFT_COLLAPSE_RATIO)),
            "edge_frac": float(self._cfg("profile.token_drift_edge_frac",
                                         TOKEN_DRIFT_EDGE_FRAC)),
            "edge_collapse_ratio": float(self._cfg(
                "profile.token_drift_edge_collapse_ratio",
                TOKEN_DRIFT_EDGE_COLLAPSE_RATIO)),
            "sep": sep, "head_sep": head_sep,
        }
        summary: dict = {"columns": [], "params": params, "n_rows": 0,
                         "n_collapsed": 0, "collapsed_tokens": []}
        rows: list[dict] = []
        if not self._cfg("profile.token_drift_enabled", True):
            summary["skipped"] = "disabled"
            return {"rows": rows, "summary": summary}
        t0 = time.time()
        try:
            df, tcol = self.df, self.spec.time_column
            if not tcol or tcol not in df.columns:
                summary["skipped"] = "no_time_column"
                return {"rows": rows, "summary": summary}
            rx = _token_head_regex(sep, head_sep)
            cols = self._token_drift_columns(rx, sep)
            if not cols:
                summary["skipped"] = "no_text_columns"
                return {"rows": rows, "summary": summary}
            codes, months = pd.factorize(
                df[tcol].astype(str).str[:7], sort=True)
            months = [str(m) for m in months]
            if not 2 <= len(months) <= 60:
                summary["skipped"] = f"months={len(months)}"
                return {"rows": rows, "summary": summary}
            train_idx = self.split_index.get("train")
            is_train = df.index.isin(train_idx) if train_idx is not None \
                else np.zeros(len(df), dtype=bool)
            per = max(1, params["rows_per_month"])
            pos_by_month: dict[str, np.ndarray] = {}
            for k, m in enumerate(months):
                pos = np.flatnonzero(codes == k)
                if len(pos) > per:  # 确定性等距抽样(时间/实体覆盖均匀)
                    pos = pos[np.unique(np.linspace(
                        0, len(pos) - 1, per).astype(np.int64))]
                pos_by_month[m] = pos
            n_seps = len(sep)
            for col in cols:
                col_rows, col_collapsed, info = self._token_drift_one(
                    col, df[col].to_numpy(), pos_by_month, is_train, rx,
                    n_seps, params, Counter)
                if info.get("skipped"):
                    summary.setdefault("skipped_columns", {})[col] = \
                        info["skipped"]
                    continue
                summary["columns"].append(col)
                summary.setdefault("max_jsd", {})[col] = info.get("max_jsd")
                rows.extend(col_rows)
                summary["collapsed_tokens"].extend(col_collapsed)
            summary["months"] = [m for m in months
                                 if len(pos_by_month.get(m, ())) >=
                                 TOKEN_DRIFT_MIN_MONTH_ROWS]
        except Exception as exc:  # noqa: BLE001  补充审计,不阻塞 profile
            logger.warning("token 漂移审计失败(不阻塞 profile): %s", exc)
            summary["error"] = str(exc)[:300]
            return {"rows": [], "summary": summary}
        summary["n_rows"] = len(rows)
        summary["n_collapsed"] = len(summary["collapsed_tokens"])
        summary["collapsed_tokens"] = summary["collapsed_tokens"][:100]
        summary["elapsed_sec"] = round(time.time() - t0, 2)
        if summary["n_collapsed"]:
            logger.warning(
                "token 漂移:%d 条 token 月覆盖率塌缩(基线≥%.0f%%;整行 <基线×"
                "%.0f%% / 首尾窗 <基线×%.0f%%): %s", summary["n_collapsed"],
                params["min_baseline_coverage"] * 100,
                params["collapse_ratio"] * 100,
                params["edge_collapse_ratio"] * 100,
                ", ".join(f"{c['column']}:{c['token']}[{c['scope']}]"
                          f"@{c['collapse_month']}"
                          for c in summary["collapsed_tokens"][:10]))
        return {"rows": rows, "summary": summary}

    @staticmethod
    def _token_drift_one(col: str, values: np.ndarray,
                         pos_by_month: dict, is_train: np.ndarray,
                         rx: re.Pattern, n_seps: int, params: dict,
                         Counter) -> tuple[list, list, dict]:
        """单列 token 漂移:逐月抽样行一次 findall,Counter 累计行覆盖与频次。

        每行只更新一组计数器(按是否 train 分桶),月度与 train 基线在月末
        由桶合并得到(合并成本 O(词表),不随行数增长)。edge_frac>0 时另记
        每行首/尾 edge_frac 比例 token 内的覆盖(序列近端窗口,见 _token_drift)。
        """
        seps, frac = params["sep"], float(params.get("edge_frac") or 0.0)
        # 计数器槽位:row 覆盖 / 出现频次 / 尾窗覆盖 / 首窗覆盖
        D, O, T, H = 3, 4, 5, 6
        stats: dict[str, dict] = {}
        tr = [0, 0, 0, Counter(), Counter(), Counter(), Counter()]
        for m, pos in pos_by_month.items():
            # 桶 0 = 非 train 行,桶 1 = train 行:[n, ntok, nonempty, D, O, T, H]
            acc = [[0, 0, 0, Counter(), Counter(), Counter(), Counter()]
                   for _ in range(2)]
            for v, is_tr in zip(values[pos], is_train[pos]):
                a = acc[1 if is_tr else 0]
                a[0] += 1
                if not isinstance(v, str):
                    if v is None or v is pd.NA or (isinstance(v, float)
                                                   and v != v):
                        continue
                    v = str(v)
                if not v:
                    continue
                a[1] += (sum(v.count(c) for c in seps) if n_seps > 1
                         else v.count(seps)) + 1
                a[2] += 1
                heads = rx.findall(v)
                if heads and heads[-1] == "":
                    heads.pop()  # 正则在串尾的空匹配
                a[O].update(heads)
                a[D].update(set(heads))
                if frac > 0 and heads:
                    w = max(1, int(np.ceil(len(heads) * frac)))
                    a[T].update(set(heads[-w:]))
                    a[H].update(set(heads[:w]))
            for a in acc:  # 空头部(位置标记/空 token)不进词表
                for k in (D, O, T, H):
                    del a[k][""]
            a0, a1 = acc
            for k in range(3):
                tr[k] += a1[k]
            for k in (D, O, T, H):
                tr[k].update(a1[k])
            stats[m] = {"n": a0[0] + a1[0], "ntok": a0[1] + a1[1],
                        "nonempty": a0[2] + a1[2], "doc": a0[D] + a1[D],
                        "occ": a0[O] + a1[O], "tail": a0[T] + a1[T],
                        "head": a0[H] + a1[H]}
        tr_rows, tr_tok, tr_nonempty = tr[0], tr[1], tr[2]
        tr_doc, tr_occ = tr[D], tr[O]
        if tr_rows < TOKEN_DRIFT_MIN_MONTH_ROWS or not tr_doc:
            return [], [], {"skipped": f"train_rows={tr_rows}"}
        top = sorted(tr_doc, key=lambda t: (-tr_doc[t], t))[:params["top_k"]]
        base = {"row": {t: tr_doc[t] / tr_rows for t in top}}
        scopes = [("row", "doc", params["collapse_ratio"])]
        if frac > 0:
            base["tail"] = {t: tr[T][t] / tr_rows for t in top}
            base["head"] = {t: tr[H][t] / tr_rows for t in top}
            scopes += [("tail", "tail", params["edge_collapse_ratio"]),
                       ("head", "head", params["edge_collapse_ratio"])]
        tr_occ_total = sum(tr_occ.values())
        tr_top_occ = np.array([tr_occ[t] for t in top], dtype=float)
        q = np.append(tr_top_occ, max(tr_occ_total - tr_top_occ.sum(), 0.0))
        months = [m for m in sorted(stats)
                  if stats[m]["n"] >= TOKEN_DRIFT_MIN_MONTH_ROWS]
        rows: list[dict] = []
        cov: dict[str, dict] = {s: {t: {} for t in top} for s, _, _ in scopes}
        max_jsd = float("nan")
        for m in months:
            st = stats[m]
            n, occ = st["n"], st["occ"]
            occ_total = sum(occ.values())
            p_top = np.array([occ[t] for t in top], dtype=float)
            p = np.append(p_top, max(occ_total - p_top.sum(), 0.0))
            jsd = _jsd(p, q)
            if jsd == jsd:
                max_jsd = jsd if max_jsd != max_jsd else max(max_jsd, jsd)
            new_share = (sum(c for t, c in occ.items() if t not in tr_occ)
                         / occ_total) if occ_total else float("nan")
            vanished = (sum(c for t, c in tr_occ.items() if occ.get(t, 0) == 0)
                        / tr_occ_total) if tr_occ_total else float("nan")
            rows.append({
                "feature": col, "comparison": f"train_vs_{m}", "psi": np.nan,
                "kind": "token", "token": "__vocab__", "period": m,
                "n_rows": int(n),
                "coverage": st["nonempty"] / n if n else np.nan,
                "baseline_coverage": tr_nonempty / tr_rows,
                "coverage_ratio": np.nan,
                "tokens_per_row": st["ntok"] / n if n else np.nan,
                "baseline_tokens_per_row": tr_tok / tr_rows,
                "jsd": jsd, "new_token_share": new_share,
                "vanished_token_share": vanished, "flag": ""})
            for t in top:
                for s, key, _ in scopes:
                    cov[s][t][m] = st[key].get(t, 0) / n if n else 0.0
                b = base["row"][t]
                r = {"feature": col, "comparison": f"train_vs_{m}",
                     "psi": np.nan, "kind": "token", "token": t, "period": m,
                     "n_rows": int(n), "coverage": cov["row"][t][m],
                     "baseline_coverage": b,
                     "coverage_ratio": cov["row"][t][m] / b if b > 0
                     else np.nan, "flag": ""}
                if frac > 0:
                    for s in ("tail", "head"):
                        r[f"{s}_coverage"] = cov[s][t][m]
                        r[f"baseline_{s}_coverage"] = base[s][t]
                rows.append(r)
        # 塌缩判定(逐口径):出现过(≥ 基线×ratio)之后某月跌破 基线×ratio
        collapsed: list[dict] = []
        flagged: dict[tuple[str, str], list[str]] = {}
        for scope, _, ratio in scopes:
            for t in top:
                b = base[scope][t]
                if b < params["min_baseline_coverage"]:
                    continue
                seen, first, low = False, None, []
                for m in months:
                    c = cov[scope][t].get(m, 0.0)
                    if c >= ratio * b:
                        seen = True
                    elif seen:
                        first = first or m
                        low.append(m)
                        flagged.setdefault((t, m), []).append(
                            "collapse" if scope == "row"
                            else f"collapse_{scope}")
                if first is not None:
                    after = [m for m in months if m >= first]
                    collapsed.append({
                        "column": col, "token": t, "scope": scope,
                        "threshold_ratio": ratio,
                        "baseline_coverage": round(b, 4),
                        "collapse_month": first,
                        "collapsed_months": low,
                        "min_coverage_ratio": round(min(
                            cov[scope][t].get(m, 0.0) / b for m in low), 4),
                        "persistent": len(low) == len(after),
                    })
        if flagged:
            for r in rows:
                f = flagged.get((r["token"], r["period"]))
                if f:
                    r["flag"] = "|".join(f)
        return rows, collapsed, {"max_jsd": None if max_jsd != max_jsd
                                 else round(max_jsd, 4)}

    # ---- 信息压缩检测(v8):data_view 候选自动生成 ----
    def _compression_audit(self) -> dict:
        """检测 init 聚合口径是否压缩了信息,产出 data_view 候选。

        双侧启发式:
        - 数据侧:实体多行结构(id 分组 rows_per_entity>1 → 原始数据本有
          多期,若特征面未利用组内时序即提示);抽样实体组内零方差列占比
          (占比低 = 特征面几乎不含组内变化信息);
        - loader 侧:逐行正则扫描 data-loader.py 源码中的压缩模式
          (.last()/tail/iloc[-n]/nth/drop_duplicates/groupby().agg)。

        risk_level: none / suspect / high;仅 suspect/high 产出候选。
        候选供 Planner 直接读取(write-splits-view 落地,见
        task-and-data.md「数据视图」)。
        """
        enabled = True
        try:
            from .config import Config  # noqa: F401 (profile 不持 config)
        except Exception:
            pass
        out = {"enabled": enabled, "risk_level": "none",
               "rows_per_entity": None, "zero_variance_col_share": None,
               "loader_patterns": [], "data_view_candidates": []}
        df, spec = self.df, self.spec
        id_col = spec.id_column
        tcol = spec.time_column

        # ---- 数据侧:实体多行结构 + 组内零方差 ----
        data_signal = False
        if id_col and id_col in df.columns:
            n_ent = int(df[id_col].nunique())
            if n_ent > 0:
                rpe = float(len(df) / n_ent)
                out["rows_per_entity"] = round(rpe, 3)
                if rpe > 1.3:
                    data_signal = True
                # 抽样实体 × 数值列的组内零方差占比(信息量代理)
                try:
                    num_cols = [c for c in self.feature_cols
                                if pd.api.types.is_numeric_dtype(df[c])
                                ][:COMPRESSION_SCAN_MAX_COLS]
                    if num_cols and n_ent > 0:
                        sample_ids = pd.Series(df[id_col].unique()).sample(
                            n=min(n_ent, COMPRESSION_SCAN_MAX_ENTITIES),
                            random_state=0)
                        sub = df[df[id_col].isin(sample_ids)]
                        grouped = sub.groupby(id_col)[num_cols].nunique(
                            dropna=False)
                        zero_share = float((grouped <= 1).mean().mean())
                        out["zero_variance_col_share"] = round(zero_share, 3)
                        if rpe > 1.3 and zero_share > 0.7:
                            data_signal = True  # 多行但特征面几乎无组内变化
                except Exception:  # noqa: BLE001  抽样失败不阻塞
                    pass

        # ---- loader 侧:源码压缩模式扫描 ----
        patterns = _scan_loader_source(self.loader_path)
        out["loader_patterns"] = patterns
        strong = {p["pattern"] for p in patterns}

        loader_signal = len(strong) >= 2 or (
            len(strong) >= 1 and {"groupby().agg", "last",
                                  "tail"}.intersection(strong))
        if data_signal and loader_signal:
            out["risk_level"] = "high"
        elif data_signal or loader_signal:
            out["risk_level"] = "suspect"

        # ---- 候选生成(仅 suspect/high)----
        if out["risk_level"] in ("suspect", "high"):
            cands = []
            if data_signal:
                cands.append({
                    "view_id_hint": "v-window-k",
                    "trigger": (f"rows_per_entity={out['rows_per_entity']}"
                                + (f", zero_variance_col_share="
                                   f"{out['zero_variance_col_share']}"
                                   if out["zero_variance_col_share"]
                                   is not None else "")),
                    "rationale": "原始数据实体多行(多期),冻结宽表口径可能"
                                 "压缩了组内时序信息(Home Credit 教训:"
                                 "bureau_balance 状态流被压到末行)",
                    "suggested_action": "write-splits-view 重做聚合:按实体"
                                        "保留近 K 期窗口/趋势斜率/事件计数,"
                                        "而非仅末行或均值",
                })
            if patterns:
                pat_names = ", ".join(sorted(strong)) or \
                    ", ".join(p["pattern"] for p in patterns[:3])
                cands.append({
                    "view_id_hint": "v-reagg",
                    "trigger": f"data-loader 命中压缩模式: {pat_names}",
                    "rationale": "init 聚合口径决定特征天花板(v7 data_view "
                                 "动机),loader 的压缩模式提示聚合层之下"
                                 "可能有未利用信号",
                    "suggested_action": "调整聚合口径(新近度加权/active "
                                        "子集统计/多窗口)重建特征面",
                })
            if tcol and id_col and id_col in df.columns \
                    and tcol in df.columns:
                cands.append({
                    "view_id_hint": "v-recent-bias",
                    "trigger": "时间列存在时的通用候选",
                    "rationale": "时间衰减加权/近 K 期子集往往优于全历史"
                                 "等权聚合",
                    "suggested_action": "近 K 期子集统计 + 全历史统计双口径"
                                        "对照",
                })
            out["data_view_candidates"] = cands
        return out

    # ---- Leakage audit(§7.1)----
    # ------------------------------------------------------------------
    # v13 phase-1:全列泄漏分类(毒性轴:banned / reparse / ambiguous)
    # ------------------------------------------------------------------

    _WINDOW_SUFFIX_RE = re.compile(
        r"^(?P<stem>.+?)[_\-](?:t(?P<tw>\d+)|(?P<dw>\d+)d)$")

    @staticmethod
    def _text_shape_evidence(s: pd.Series) -> dict:
        """非数值列的**形态证据**:区分「实体 ID」与「复合文本字段」。

        背景(实测事故,2026-09-22,req-t7-propensity run):
            规则 3 原实现只看「非数值 + 唯一率>0.95」就判 ``entity_identifier``
            并 banned-certain。但**逐人唯一的复合文本字段**同样唯一率≈1:
            本 run 的 ``seq_d365``(逗号分隔的行为事件序列,中位 342 个 token)
            与 ``seq_gap_d365`` 唯一率均为 0.9976,被误封为实体标识 → 直接
            丢弃该数据面最强信息通道(全局经验库 exp-202609-8a4fe068 标注
            seq reparse 加宽可带来 ~+0.0035 AUC)。同数据源的三个历史 run
            均正确判为 reparse,只有本 run 误判——说明原判据不可靠。

        三项判据(任一命中即视为"疑似复合文本字段"):
            - 长度变异系数 CV:真 ID 定长(CV≈0);序列文本长度随活跃度剧烈变化;
            - token 语料/词表复用倍数:token 会被千万级语料复用上万次,ID 无词表;
            - 平均分隔符个数:复合编码天然含分隔符,ID 通常没有。

        全部为只读统计,失败静默降级(返回空证据),不影响既有分类路径。
        """
        out: dict = {}
        try:
            if not pd.api.types.is_object_dtype(s) \
                    and not pd.api.types.is_string_dtype(s):
                return out
            txt = s.dropna().astype(str)
            if not len(txt):
                return out
            # 逐行字符串运算较贵 → 确定性截断(seed 无关,取前 N 行)
            if len(txt) > TEXT_SHAPE_SCAN_ROWS:
                txt = txt.iloc[:TEXT_SHAPE_SCAN_ROWS]
            lens = txt.str.len()
            mean_len = float(lens.mean())
            out["len_cv"] = round(
                float(lens.std()) / mean_len, 3) if mean_len > 0 else 0.0
            out["delim_density"] = round(
                float(txt.str.count(r"[,;|]").mean()), 2)
            # token 词表复用:序列字段的 token 种类远少于行数 × token 数
            tokens = txt.str.split(r"[,;|]").explode()
            vocab = int(tokens.nunique())
            out["token_vocab"] = vocab
            out["token_reuse"] = round(
                len(tokens) / vocab, 1) if vocab else 0.0
            out["looks_composite"] = bool(
                out["len_cv"] > TEXT_SHAPE_LEN_CV_MIN
                or out["token_reuse"] > TEXT_SHAPE_VOCAB_REUSE_MIN
                or out["delim_density"] > TEXT_SHAPE_DELIM_MIN)
        except Exception:  # noqa: BLE001  形态证据是补充,不得影响主判定
            return {}
        return out

    def _cross_check_experience(self, proposals: list[dict],
                                spec: TaskSpec) -> list[dict]:
        """把机械提案与**全局经验库**对账,命中"曾被证明有价值"的列时降档。

        背景(实测事故,2026-09-22,req-t7-propensity run):
            规则 3 把 ``seq_d365``/``seq_gap_d365`` 误判为 entity_identifier
            并封禁。而全局库里的高置信度经验 exp-202609-8a4fe068 明确写着
            「seq reparse 加宽在 30 万行信贷行为时序表上带来 ~+0.0035 AUC,
            是该数据面最强信息通道」——**经验库当时已经在磁盘上,但
            phase-1 决策完全不读它**。检索能力只接在 Planner(演进期),
            泄漏分类这条更早、更硬的路径上是断的。

        机制:对每个 banned-certain 提案,用「列名 + 数据语义」做一次
        全局库检索;命中经验卡的语句时,把该列降档为 **ambiguous** 并
        附上命中卡 id 作为证据——强制进入人工裁决,而不是被机械规则
        静默封禁。降档方向是 fail-safe 的:ambiguous 不会放行任何列
        (数据面仍由用户裁决决定),只是把"机械判决"变成"待复核"。

        任何异常都静默降级为空结果:经验库是**补充证据**,不得让
        profile 因它失败。检索用 n-gram 路径(embedding 后端缺失时自动
        回退),不引入额外依赖。
        """
        hits: list[dict] = []
        if not proposals:
            return hits
        # v2.18.0:经验来自本 run 固定水位的经验库视图(首次调用时固定;服务不可达
        # 且无本机缓存时为空视图)。只扫**正向**原则(direction=+1)且口径同类型;
        # "避免 / 无增益"原则命中词根会被误报成"曾有正增益"(v2.15.8 教训)。
        try:
            from . import library_view

            view = library_view.ensure(self.out_dir.parent, spec=spec,
                                       config=getattr(self, "config", None))
        except Exception:  # noqa: BLE001  经验库不可用时不影响分类
            return hits
        _mt = library_view.spec_scope(spec).get("model_type")
        _conf_rank = {"high": 0, "medium": 1, "low": 2}
        pool = []
        for pr in view.principles:
            if pr.get("direction") != 1 or pr.get("category") == "engineering":
                continue
            if _mt and (pr.get("scope") or {}).get("model_type") not in (None, _mt):
                continue
            pool.append({"experience_id": pr["principle_id"], "claim": pr.get("claim"),
                         "objects": pr.get("objects") or [],
                         "condition": pr.get("condition") or {},
                         "limitations": pr.get("limitations") or [],
                         "confidence": (view.stats.get(pr["principle_id"]) or {}).get(
                             "confidence") or pr.get("confidence")})
        pool.sort(key=lambda c: _conf_rank.get(str(c.get("confidence")), 3))
        for p in proposals:
            if p["verdict"] != "banned-certain":
                continue
            matched = []
            for c in pool:
                # v2.17.0:单张畸形卡只跳过该卡,不得让 profile 失败
                try:
                    if _experience_supports_column(c, p["column"]):
                        matched.append(c)
                except Exception:  # noqa: BLE001
                    continue
            if not matched:
                continue
            card = matched[0]
            _claim = str(card.get("claim") or card.get("statement") or "")
            p["verdict"] = "ambiguous"
            p["reason"] = "experience_conflict"
            p["evidence"] = (
                f"{p['evidence']};**与经验库冲突**:{card.get('experience_id')}"
                f" 记录该列/同族列曾有正增益"
                f"({_claim[:80]}…),"
                "降档待人工裁决")
            p["experience_id"] = card.get("experience_id")
            p["experience_statement"] = _claim[:400]
            hits.append({
                "column": p["column"],
                "experience_id": card.get("experience_id"),
                "confidence": card.get("confidence"),
                "statement": _claim[:400],
            })
        return hits

    def _leak_classify(self, *, feat_summary: dict | None = None,
                       iv_summary: dict | None = None) -> dict:
        """全列三档分类提案(banned-certain / ambiguous / reparse-candidate)。

        审计范围 = 全部列(audit_cols),不只特征面——被排除的列才是
        最需要审计的对象(旧 leakage-audit 只扫 feature-quality 覆盖列,
        banned 候选根本不在表里)。证据来自 profile 已算的中间件引用
        + 本方法的机械规则,不重复昂贵计算:

        机械规则(优先级从高到低):
        1. stem 窗口:与 target 同 stem 且窗口 > performance_window →
           future_window(banned-certain;窗口越过决策时点);
        2. 单变量 AUC > 0.85 → label_equivalent(banned-certain);
        3. 唯一率≈1 且非数值 → **形态裁决**(v22.1 修正):
           - n_unique 超过实体数 → 不可能是实体标识(实证否决);
           - 形态像复合文本字段(长度 CV 大 / 分隔符密集 / token 词表高复用)
             → reparse-candidate(信息合法但形态不可直用,走 plan 解析);
           - 其余(定长、无分隔、无词表)→ entity_identifier(banned-certain);
           唯一率≈1 且数值连续 → reparse-candidate(分箱去 ID 化);
        4. object / 大文本高基数 → reparse-candidate(形态不可用);
        5. AUC ∈ (0.7, 0.85) 或 stem 邻近 target → ambiguous(人裁)。

        **banned-certain 不是终审**(v22.1):它只是机械提案,可能误伤
        高价值列。其形态证据(unique_ratio / n_entities / len_cv /
        delim_density / token_reuse)必须逐列落进提案表供人复核;复核后
        判为合法的列走 ``features.leak_confirmation.reinstated_banned``
        撤销封禁——该通道缺位时,错误封禁**只能**靠新建 run 才能挽回。

        产出 proposals + 已分类现状(spec 已有 banned/reparse 时标注
        confirmed,增量列才进 proposals)。用户逐档确认后由 CLI 写回
        task-spec.yaml,phase-2 材料化据此物理隔离。
        """
        spec = self.spec
        df = self.supervised_df
        tname = spec.target_name

        # target 的 stem(如 req_cnt_t7 → req_cnt)与 performance 窗口
        m = self._WINDOW_SUFFIX_RE.match(tname)
        target_stem = m.group("stem") if m else tname
        target_win = int(m.group("tw") or m.group("dw") or 0) if m else 0
        try:
            pw = re.search(r"T\+?(\d+)",
                           str((spec.data.get("target") or {})
                               .get("performance_window") or ""))
            perf_win = int(pw.group(1)) if pw else target_win
        except Exception:  # noqa: BLE001
            perf_win = target_win

        # 已算中间件:特征面质量/IV 表(引用,不重算);审计列超出
        # 特征面时补算该列的单变量 AUC / 唯一率(便宜,逐列一次)
        known_auc: dict[str, float] = {}
        if iv_summary and isinstance(iv_summary.get("top_features"), list):
            pass  # top-20 不够全列覆盖;统一走下方逐列补算
        y_bin = None
        if spec.model_type in ("binary_classification", "uplift"):
            y_bin = (df[tname] == 1).values.astype(int)

        proposals: list[dict] = []
        confirmed_banned = set(spec.banned_columns)
        confirmed_reparse = set(spec.reparse_columns)
        # v2.12.2.9:用户已裁决"确认保留"的 ambiguous 列。它们不再进
        # proposals —— 确认门的语义是"还有未裁决的 ambiguous 才要人裁",
        # 已裁决项重复出现在提案表里会让门永远关不上(旧实现的 bug:
        # 只要 proposals 非空且 spec 没有 banned/reparse,门恒为 true,
        # 正确裁决"保留"反而无法关上门)。
        confirmed_kept = set(spec.kept_ambiguous_columns)
        # v22.1:复核后撤销的机械封禁列。与 confirmed_* 同样从审计面
        # 剔除——否则每次 profile 都会把已裁决的列重新提一遍。
        reinstated = set(spec.reinstated_banned_columns)
        # 实体数:判 entity_identifier 的**对齐基准**。真 ID 的唯一值数
        # 不可能超过实体数;超过则必然是复合文本字段。
        id_col = spec.id_column
        n_entities = None
        if id_col and id_col in df.columns:
            try:
                n_entities = int(df[id_col].nunique(dropna=True))
            except Exception:  # noqa: BLE001
                n_entities = None

        for col in self.audit_cols:
            # 已裁决的列不再进提案面。v22.1:被 reinstated 的列是例外——
            # 撤销封禁正是为了让该列**重新出现在复核清单里**待重新裁决。
            if (col in confirmed_banned and col not in reinstated) \
                    or col in confirmed_reparse or col in confirmed_kept:
                continue
            s = df[col]
            n_unique = int(s.nunique(dropna=True))
            unique_ratio = n_unique / max(len(s), 1)
            is_numeric = pd.api.types.is_numeric_dtype(s)
            auc = None
            if y_bin is not None and is_numeric and n_unique > 1:
                try:
                    auc = float(roc_auc(y_bin,
                                        pd.to_numeric(s, errors="coerce")
                                        .fillna(0).values))
                    auc = max(auc, 1 - auc)  # 方向修正
                except Exception:  # noqa: BLE001
                    auc = None

            # 规则 1:同 stem 且窗口越过 performance_window
            mc = self._WINDOW_SUFFIX_RE.match(col)
            reason, evidence, verdict = None, "", None
            if mc and mc.group("stem") == target_stem:
                win = int(mc.group("tw") or mc.group("dw") or 0)
                if win > perf_win:
                    reason = "future_window"
                    evidence = (f"与 target 同 stem({target_stem}),窗口 "
                                f"{win} > performance {perf_win}"
                                + (f";AUC={auc:.3f}" if auc else ""))
                    verdict = "banned-certain"
            # 规则 2:AUC 哨兵
            if verdict is None and auc is not None \
                    and auc > LEAKAGE_HIGH_UNIVARIATE_AUC:
                reason = "label_equivalent"
                evidence = f"单变量 AUC={auc:.3f} > {LEAKAGE_HIGH_UNIVARIATE_AUC}"
                verdict = "banned-certain"
            # 规则 3:ID-like。数值连续列(浮点唯一率=1 是常态,如金额/
            # 评分)不算 ID——树模型天然可分箱;只有离散整数且值域与
            # 行数同量级才疑似实体编码。
            #
            # v22.1 修正:非数值近唯一列**不再直接判 entity_identifier**。
            # 原实现在此把逐人唯一的复合文本字段(行为序列/JSON/编码串)
            # 一律封禁,而 BANNED_REASONS 里 entity_identifier 的定义是
            # 「唯一率≈1 **且与实体数对齐**」——代码只实现了前半句。
            # 现在补上后半句:先查实体数对齐与形态证据,再定档。
            shape: dict = {}
            if verdict is None and unique_ratio > LEAKAGE_ID_LIKE_UNIQUE_RATIO \
                    and n_unique > 100:
                if is_numeric and not pd.api.types.is_float_dtype(s):
                    verdict = "reparse-candidate"
                    reason = "form_unusable"
                    evidence = (f"整数近唯一(唯一率 {unique_ratio:.3f}),"
                                "疑似实体编码;分箱/秩变换后信息合法")
                elif not is_numeric:
                    shape = self._text_shape_evidence(s)
                    over_entities = (n_entities is not None
                                     and n_unique > n_entities)
                    if shape.get("looks_composite") or over_entities:
                        # 信息合法、只是形态不可直用 → 走 reparse plan。
                        # 注意 BANNED_REASONS 无 form_unusable,硬判 banned
                        # 会连 validate 都过不了:语义上本就该去 reparse。
                        verdict = "reparse-candidate"
                        reason = "form_unusable"
                        why = []
                        if over_entities:
                            why.append(
                                f"唯一值数 {n_unique} > 实体数 {n_entities}")
                        if shape.get("len_cv", 0) > TEXT_SHAPE_LEN_CV_MIN:
                            why.append(f"长度CV={shape['len_cv']}")
                        if shape.get("delim_density", 0) > \
                                TEXT_SHAPE_DELIM_MIN:
                            why.append(
                                f"分隔符密度={shape['delim_density']}")
                        if shape.get("token_reuse", 0) > \
                                TEXT_SHAPE_VOCAB_REUSE_MIN:
                            why.append(
                                f"token复用={shape['token_reuse']}x/"
                                f"{shape.get('token_vocab')}词表")
                        evidence = ("疑似复合文本字段("
                                    + ";".join(why)
                                    + "),非实体标识;须补 reparse plan")
                    else:
                        reason = "entity_identifier"
                        evidence = (f"非数值且唯一率 {unique_ratio:.3f}≈1,"
                                    "疑似实体标识"
                                    + (f";n_unique {n_unique} ≤ 实体数 "
                                       f"{n_entities}" if n_entities else "")
                                    + (f";长度CV={shape['len_cv']}"
                                       if shape.get("len_cv") is not None
                                       else ""))
                        verdict = "banned-certain"
            # 规则 4:形态不可用(object/大文本高基数)。注意只对真正
            # 的"不可直用"形态收案:object/字符串/混合 dtype;数值列
            # 即使高基数也可直用(f1/f2 这类连续特征不该进 reparse)
            if verdict is None and not is_numeric:
                verdict = "reparse-candidate"
                reason = "form_unusable"
                evidence = (f"dtype={s.dtype},基数 {n_unique};"
                            "需解析为数值族后入面")
            # 规则 5:语义模糊
            if verdict is None:
                near = bool(mc and mc.group("stem") == target_stem)
                if (auc is not None and auc > 0.70) or near:
                    verdict = "ambiguous"
                    reason = "label_proxy_family" if near else "user_confirmed"
                    evidence = (f"AUC={auc:.3f}" if auc else ""
                                ) + (";与 target 同族业务口径" if near else "")
            if verdict is None:
                continue
            proposals.append({
                "column": col, "verdict": verdict,
                "reason": reason, "evidence": evidence,
                "univariate_auc": auc,
                "unique_ratio": round(unique_ratio, 4),
                # v22.1:banned-certain 是**机械提案**不是终审,复核所需的
                # 形态证据必须逐列落表(n_unique vs 实体数是 entity_identifier
                # 的核心判据;文本形态三项是复合字段的识别依据)。
                "n_unique": n_unique,
                "n_entities": n_entities,
                "len_cv": shape.get("len_cv"),
                "delim_density": shape.get("delim_density"),
                "token_vocab": shape.get("token_vocab"),
                "token_reuse": shape.get("token_reuse"),
                "dtype": str(s.dtype),
            })

        order = {"banned-certain": 0, "ambiguous": 1, "reparse-candidate": 2}
        # v22.1:机械提案与经验库对账(见 _cross_check_experience)
        exp_hits = self._cross_check_experience(proposals, spec)
        proposals.sort(key=lambda p: (order[p["verdict"]],
                                      -(p.get("univariate_auc") or 0)))
        out = {
            "checked_at": now_iso(),
            "target_stem": target_stem,
            "performance_window": perf_win,
            "confirmed": {
                "banned": sorted(confirmed_banned),
                "reparse": sorted(confirmed_reparse),
                "reparse_plans": spec.reparse_plans,
                # v2.12.2.9:确认门裁决留痕(逐列 + 用户说明)
                "kept_ambiguous": sorted(confirmed_kept),
                # v22.1:复核后撤销的机械封禁(留痕,进 leak-classification.md)
                "reinstated_banned": sorted(reinstated),
                "n_entities": n_entities,
                "confirmation_note": spec.leak_confirmation_note,
            },
            "proposals": proposals,
            "counts": {v: sum(1 for p in proposals if p["verdict"] == v)
                       for v in order},
            # v22.1:与全局经验库对账的结果(命中列已降档为 ambiguous)
            "experience_cross_check": exp_hits,
            # v2.12.2.9:门只看**未裁决**的 ambiguous。已确认保留的列
            # 已从 proposals 中剔除(见上方 confirmed_kept 过滤),因此
            # 这里只需判"是否还有 ambiguous 提案"。旧实现额外加了
            # `not confirmed_banned and not confirmed_reparse` 的兜底,
            # 那会让"spec 只声明了保留、没声明排除"的合法形态永远
            # 关不上门——正是本次修复要消除的缺口。
            "needs_user_confirmation": bool(
                any(p["verdict"] == "ambiguous" for p in proposals)),
        }
        (self.out_dir / "leak-classification.md").write_text(
            _render_leak_classification_md(out), encoding="utf-8")
        return out

    def _leakage_audit(self, *, iv_summary: dict | None = None,
                       feat_summary: dict | None = None,
                       splits_summary: dict | None = None,
                       drift_summary: dict | None = None) -> dict:
        """泄漏审计;接受已算好的 summary 复用,不重复计算。"""
        df, spec = self.df, self.spec
        findings: list[dict] = []
        severity_cnt = {"critical": 0, "warning": 0, "info": 0}

        def add(severity: str, kind: str, detail: str, cols: list | None = None):
            findings.append({"severity": severity, "kind": kind,
                             "detail": detail, "columns": cols or []})
            severity_cnt[severity] += 1

        tname = spec.target_name
        # 1) label 派生/标签相关字段名
        target_tokens = set(str(tname).lower().replace("_", " ").split())
        for col in self.feature_cols:
            ctok = set(col.lower().replace("_", " ").split())
            if target_tokens and target_tokens.issubset(ctok) and col != tname:
                add("critical", "label_leakage",
                    f"特征名包含目标字段 token: {col}", [col])
        # 2) 异常高单变量 AUC(复用 IV 表结果;无则现算)
        #    回归无 AUC 口径,跳过(回归泄漏靠相关性与目标漂移检测)
        try:
            if iv_summary is None:
                iv_summary = self._iv_woe_table()["summary"]
            for rec in iv_summary.get("suspect_univariate_auc_gt_085", []):
                add("critical", "label_leakage",
                    f"单变量 AUC > {LEAKAGE_HIGH_UNIVARIATE_AUC}: {rec}", [rec])
        except Exception as exc:  # noqa: BLE001
            logger.warning("leakage: 单变量 AUC 检查失败: %s", exc)
        # 2b) 回归:异常高单特征相关(|corr|>0.95)视为泄漏候选
        if spec.model_type == "regression":
            try:
                if iv_summary is None:
                    iv_summary = self._iv_woe_table()["summary"]
                for rec in iv_summary.get("corr_gt_095_leakage", []):
                    add("critical", "label_leakage",
                        f"单特征 |Spearman corr| > 0.95: {rec}", [rec])
            except Exception as exc:  # noqa: BLE001
                logger.warning("leakage: 回归相关性检查失败: %s", exc)
        # 3) ID-like 特征(复用质量表结果;无则现算)
        try:
            if feat_summary is None:
                feat_summary = self._feature_quality()["summary"]
            # v2.17.0(D9):只报整数型/字符串型且像标识符的高唯一率列
            # (id_like_features);旧调用方传入的摘要无该键时沿用高基数口径
            _id_cols = feat_summary.get("id_like_features")
            if _id_cols is None:
                _id_cols = feat_summary["high_cardinality_features"]
            for col in _id_cols:
                add("warning", "id_leakage",
                    f"唯一率≈1、整数/字符串型且形似标识符,疑似 ID 列: {col}",
                    [col])
        except Exception as exc:  # noqa: BLE001
            logger.warning("leakage: 高基数检查失败: %s", exc)
        # 4) 跨 split 实体重叠(group leakage;复用 splits 摘要)
        splits = splits_summary if splits_summary is not None \
            else self._splits_summary()
        overlap = splits.get("entity_overlap_rate") or {}
        for pair, rate in overlap.items():
            if float(rate) > 0.0:
                add("warning", "group_leakage",
                    f"实体跨 split 重叠 {pair}: {rate:.2%}", [])
        # 5) time leakage:特征时间戳晚于预测时点(TaskSpec 显式声明)
        for bad in (spec.data.get("leakage_flags") or {}).get(
                "features_after_decision_time", []):
            add("critical", "future_leakage",
                f"TaskSpec 声明决策时点后的特征: {bad}", [bad])
        # 6) duplicate leakage
        dup = float(df.duplicated().mean())
        if dup > 0.01:
            add("warning", "duplicate_leakage",
                f"完全重复行占比 {dup:.2%},可能造成 split 间复制", [])
        # validation/shadow 的标签统计不进入 profile 反馈。目标漂移只在
        # 固定 evaluator/confirmation 阶段衡量；这里保留无监督 PSI。
        # 7) v2.17.2(E2):reparse/文本列 token 月覆盖率塌缩(数据面体检:
        #    日志断档/口径变更会让依赖该 token 的派生特征在塌缩月后静默失真;
        #    复用 drift 摘要,每列一条 warning)
        td = (drift_summary or {}).get("token_drift") or {}
        by_col: dict[str, list] = {}
        for c in td.get("collapsed_tokens") or []:
            by_col.setdefault(c["column"], []).append(c)
        _scope_cn = {"row": "整行", "tail": "尾窗", "head": "首窗"}
        for col, items in by_col.items():
            desc = "; ".join(
                f"{c['token']}[{_scope_cn.get(c.get('scope'), '整行')}]"
                f"(基线 {c['baseline_coverage']:.1%},{c['collapse_month']} 起"
                f" ≤{c['min_coverage_ratio']:.0%} 基线"
                f"{',持续' if c.get('persistent') else ''})"
                for c in items[:8])
            _p = td.get("params") or {}
            add("warning", "token_drift",
                f"{col}: {len(items)} 条 token 月覆盖率塌缩(基线≥"
                f"{_p.get('min_baseline_coverage', 0.05):.0%};整行 <基线×"
                f"{_p.get('collapse_ratio', 0.1):.0%},序列首/尾 "
                f"{_p.get('edge_frac', 0.1):.0%} token 窗 <基线×"
                f"{_p.get('edge_collapse_ratio', 0.5):.0%}):{desc}。"
                "疑似日志断档/口径变更,依赖这些 token 的派生特征(尤其近因/"
                "短窗计数)在塌缩月之后失真;明细见 drift-profile(kind=token)",
                [col])

        return {
            "checked_at": now_iso(),
            "severity_counts": severity_cnt,
            "findings": findings,
        }


# ---------------------------------------------------------------------------
# Markdown 渲染
# ---------------------------------------------------------------------------

def _card_text_fields(card: dict) -> list[str]:
    """经验卡里参与"讲的是哪一列"匹配的全部文本(v2.16.3,D-014)。

    字段一律按**任意类型**降级取值:共享库里有 20+ 条 ``context`` 是字符串、
    上百条 ``limitations`` 是字符串,旧实现 ``(card.get("context") or {}).get(...)``
    在单条坏数据上就抛 AttributeError,把整个泄漏分类(以及 8 个回归用例)打成
    失败。现在非 dict 的 ``context`` / ``condition`` 按原文整体并入 blob ——
    信息不丢(字符串里出现的列名照样能命中),类型也不再是崩溃点。
    """
    out: list[str] = []
    for key in ("claim", "statement", "recommended_action", "target"):
        v = card.get(key)
        if v:
            out.append(str(v))
    ctx = card.get("context")
    if isinstance(ctx, dict):
        # objective_description(旧 dict 卡)/ note(v2.17.0 起字符串 context
        # 的归一布局 {"note": 原文})
        for k in ("objective_description", "note"):
            if ctx.get(k):
                out.append(str(ctx[k]))
    elif ctx:
        out.append(str(ctx))
    if card.get("context_text"):
        out.append(str(card["context_text"]))
    cond = card.get("condition")
    if not isinstance(cond, dict) and cond:
        out.append(str(cond))
    lims = card.get("limitations")
    if isinstance(lims, list):
        out.extend(str(x) for x in lims if x)
    elif lims:
        out.append(str(lims))
    return out


def _experience_supports_column(card: dict, column: str) -> bool:
    """经验卡是否在讲**这一列**(或同族列)。

    判据刻意保守:必须出现列名本身,或列名的语义词根(去掉窗口/来源前缀后
    的骨干,如 ``seq_d365`` → ``seq``)。只看任务级语义(如"信贷行为时序")
    会把任意卡片都算命中,反而制造噪声——实测本任务检索 top-3 里有 2 张
    讲的是别的列族。

    命中即为"该列值得人工复核",不是"该列一定合法";因此宁松勿严:漏判
    退化成当前的机械封禁(可被 reinstated_banned 挽回),误判只是多一条
    待裁决项。

    v2.16.3(D-014):字段取值经 :func:`_card_text_fields` 统一降级 ——
    ``context`` 为字符串时按原文并入 blob(旧实现直接 ``.get`` → AttributeError)。
    """
    if not column or not isinstance(card, dict):
        return False
    # 词根: seq_d365 → seq;offline_inner_jtbeh_draw_x → jtbeh
    stem = re.split(r"[_\-]", column)[0]
    # v2.15.8:结构化 target 优先(新 schema 的作用对象字段);v2.18.0 原则的作用
    # 对象是 objects 列表(列名或列前缀)
    tgt = str(card.get("target") or "")
    if tgt and tgt in (column, stem):
        return True
    for obj in card.get("objects") or []:
        o = str(obj)
        if o and (o in (column, stem) or o.rstrip("_") in (column, stem)
                  or (o.endswith("_") and column.startswith(o))):
            return True
    # v2.17.0:context 在全局库里有三种形态 —— dict(top-trial 配方卡)、
    # str(四字段简写原样入库,库内实有 27 张)、None/缺失;旧实现无条件
    # ``.get`` → str 卡抛 AttributeError(tests/test_v22_1 的 8 个失败)。
    # v2.16.3 批次:所有字段(claim/statement/recommended_action/target/
    # context[objective_description|note]/context_text/非 dict condition/
    # limitations)统一经 _card_text_fields 降级取文本并入 blob。
    blob = " ".join(_card_text_fields(card))
    if column in blob:
        return True
    # 词根按**词**匹配(不做子串):防 "seq" 命中 "sequence_len" 之类
    words = set(re.findall(r"[a-z0-9]+", blob.lower()))
    return bool(stem) and len(stem) >= 3 and stem.lower() in words


def _render_leak_classification_md(cls: dict) -> str:
    lines = ["# Leak Classification(v13 phase-1)", ""]
    c = cls.get("counts") or {}
    lines.append(
        f"三档提案:banned-certain={c.get('banned-certain', 0)}, "
        f"ambiguous={c.get('ambiguous', 0)}, "
        f"reparse-candidate={c.get('reparse-candidate', 0)};确认后写回 "
        "task-spec.yaml(features.banned / features.reparse / "
        "features.leak_confirmation.kept_ambiguous),"
        "phase-2 材料化按此物理隔离。")
    lines.append("")
    conf = cls.get("confirmed") or {}
    if conf.get("banned") or conf.get("reparse"):
        lines.append("## 已确认(spec 现状)")
        lines.append(f"- banned: {', '.join(conf['banned']) or '-'}")
        lines.append(f"- reparse: {', '.join(conf['reparse']) or '-'}")
        lines.append("")
    # v2.12.2.9:确认门裁决留痕。"保留"是合法裁决,必须与 banned/reparse
    # 并列展示——否则用户看到门关了却查不到"哪几列被确认保留了"。
    if conf.get("kept_ambiguous"):
        lines.append("## 已确认保留(features.leak_confirmation)")
        for col in conf["kept_ambiguous"]:
            lines.append(f"- {col}")
        note = conf.get("confirmation_note")
        if note:
            lines.append("")
            lines.append(f"> 裁决说明: {note}")
        lines.append("")
    # v22.1:撤销机械封禁的留痕。与 kept_ambiguous 同理——裁决有去向才不会被
    # 迫选择"新建 run"。
    if conf.get("reinstated_banned"):
        lines.append("## 已撤销机械封禁(features.leak_confirmation"
                     ".reinstated_banned)")
        for col in conf["reinstated_banned"]:
            lines.append(f"- {col}")
        lines.append("")
    gate = cls.get("needs_user_confirmation")
    lines.append(f"**needs_user_confirmation = {gate}**"
                 + ("(仍有未裁决的 ambiguous 列,需用户逐列拍板)"
                    if gate else "(全部 ambiguous 已裁决)"))
    lines.append("")
    # v22.1:机械提案复核清单。banned-certain 由正则/统计规则产生,**不是
    # 人工裁决**;把它的形态证据单独成表,让"高价值列被机械规则误封"这一
    # 类错误在复核时可见(实测事故:seq_d365 唯一率 0.9976 被误判实体标识,
    # 而同数据源三个历史 run 均正确判为 reparse)。
    banned_props = [p for p in cls.get("proposals") or []
                    if p["verdict"] == "banned-certain"]
    if banned_props:
        n_ent = cls.get("confirmed", {}).get("n_entities")
        lines.append("## 机械提案复核清单(banned-certain,非终审)")
        lines.append("")
        lines.append(
            "以下封禁由机械规则产生,**未经人工复核**。请对照形态证据确认;"
            "若判定误封,写入 `features.leak_confirmation.reinstated_banned` "
            "撤销(该通道只撤销封禁,不改数据面)。"
            + (f"对照基准:实体数 {n_ent}(真 ID 的唯一值数不可能超过它)。"
               if n_ent else ""))
        lines.append("")
        lines.append("| 列 | reason | n_unique | 实体数 | 长度CV | 分隔符密度 "
                     "| token复用 | 证据 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for p in banned_props:
            def _f(k, fmt="{:.2f}"):
                v = p.get(k)
                return "-" if v is None else (
                    fmt.format(v) if isinstance(v, (int, float)) else str(v))
            lines.append(
                f"| {p['column']} | {p['reason']} | {p.get('n_unique', '-')} "
                f"| {p.get('n_entities') or '-'} | {_f('len_cv', '{:.3f}')} "
                f"| {_f('delim_density')} | {_f('token_reuse', '{:.0f}')}x "
                f"| {p['evidence']} |")
        lines.append("")
    # v22.1:与经验库对账的留痕。"机械规则判封禁、但经验库说这列有价值"
    # 是本轮修复的核心场景,必须显式可见——否则降档原因只存在于 json 里。
    xc = cls.get("experience_cross_check") or []
    if xc:
        lines.append("## 经验库对账(机械封禁与已证经验冲突)")
        lines.append("")
        lines.append(
            "以下列被机械规则判为封禁,但全局经验库中已有记录表明"
            "**该列/同族列曾带来正增益**,已自动降档为 ambiguous 待人工裁决。")
        lines.append("")
        lines.append("| 列 | 经验卡 | 置信度 | 经验摘要 |")
        lines.append("|---|---|---|---|")
        for h in xc:
            lines.append(
                f"| {h['column']} | {h.get('experience_id') or '-'} "
                f"| {h.get('confidence') or '-'} "
                f"| {str(h.get('statement') or '')[:160]} |")
        lines.append("")
    if not cls.get("proposals"):
        lines.append("无新增提案(全部列已有归属或干净)。")
        return "\n".join(lines) + "\n"
    lines.append("| 档位 | 列 | reason | 证据 | AUC | 唯一率 | dtype |")
    lines.append("|---|---|---|---|---|---|---|")
    for p in cls["proposals"]:
        auc = p.get("univariate_auc")
        lines.append(
            f"| {p['verdict']} | {p['column']} | {p['reason']} "
            f"| {p['evidence']} | {'-' if auc is None else round(auc, 3)} "
            f"| {p.get('unique_ratio', '-')} | {p.get('dtype', '-')} |")
    lines.append("")
    lines.append(
        "> ambiguous 档需要用户逐列拍板(建议判据:窗口是否越过决策时点、"
        "与 target 是否同族业务口径);"
        "**banned-certain 与 reparse-candidate 是机械提案,不是终审**——"
        "复核后判为误封的走 `features.leak_confirmation.reinstated_banned`;"
        "reparse-candidate 需补 plan(解析去向);"
        "**裁决为保留**的列写入 `features.leak_confirmation.kept_ambiguous`。"
        "写入任一裁决后重跑 profile 即关闭确认门"
        "(无需把合法列误标为 banned,也无需为撤销封禁新建 run)。")
    return "\n".join(lines) + "\n"


def _render_leakage_md(leak: dict) -> str:
    lines = ["# Leakage Audit", ""]
    cnt = leak["severity_counts"]
    lines.append(
        f"发现 critical={cnt['critical']}, warning={cnt['warning']}, "
        f"info={cnt['info']}")
    lines.append("")
    if not leak["findings"]:
        lines.append("未发现明显泄漏风险。")
        return "\n".join(lines) + "\n"
    lines.append("| 严重度 | 类型 | 说明 | 列 |")
    lines.append("|---|---|---|---|")
    for f in leak["findings"]:
        lines.append(f"| {f['severity']} | {f['kind']} | {f['detail']} "
                     f"| {', '.join(f['columns']) or '-'} |")
    return "\n".join(lines) + "\n"


def _render_splits_md(splits: dict) -> str:
    af = splits.get("analysis_frame") or {}
    lines = ["# Splits Summary", ""]
    if af.get("sampled_from"):
        lines += [
            f"> ⚠️ **本表行数基于分析面(下采样 {af['sampled_from']} → "
            f"{af.get('rows_analyzed')} 行)**,不是训练数据面。",
            "> 训练数据面(全量)的行数以 `splits/manifest.json` 为准;"
            "两者不一致是预期行为。",
            "",
        ]
    lines += [
        "| split | rows | positive_rate / target_mean | time range |",
        "|---|---|---|---|",
    ]
    for r in splits.get("splits", []):
        rate = r.get("positive_rate")
        if rate is None:
            rate = r.get("target_mean")
        lines.append(
            f"| {r['split']} | {r['rows']} | "
            f"{'-' if rate is None else round(float(rate), 4)} | "
            f"{r.get('time_min', '-')} ~ {r.get('time_max', '-')} |")
    ov = splits.get("entity_overlap_rate") or {}
    if ov:
        lines += ["", "## 实体跨 split 重叠率", ""]
        lines += [f"- {k}: {v:.2%}" for k, v in ov.items()]
        lines += [
            "",
            "> 实体跨 split 重叠是时间切分的正常代价(老用户同时出现在"
            "多期快照),**不是列级泄漏**;但若某个 split 的窗口被写错"
            "(注解污染/格式不匹配),重叠率会异常升高——请对照 "
            "`splits/manifest.json` 的窗口构成确认。",
        ]
    return "\n".join(lines) + "\n"


def _render_profile_md(p: dict) -> str:
    s = p["sample_summary"]
    lines = [
        "# Sample Profile", "",
        f"- task: {p['task_id']}",
        f"- rows: {p['n_rows']}, features: {p['n_features']}",
    ]
    if p.get("sampled_from"):
        lines.append(f"- ⚠️ 分析样本下采样: {p['sampled_from']} → "
                     f"{p.get('n_rows_profiled')} 行(profile 统计口径;"
                     "split 落盘仍为全量)")
    if s.get("label_rate") is not None:
        lines.append(f"- label_rate: {s['label_rate']:.4f}")
    if s.get("target_stats") is not None:
        ts = s["target_stats"]
        lines.append(f"- target: mean={ts.get('mean', float('nan')):.4f} "
                     f"std={ts.get('std', float('nan')):.4f} "
                     f"min={ts.get('min', float('nan')):.4g} "
                     f"max={ts.get('max', float('nan')):.4g}")
        hint = s.get("target_skew_hint") or {}
        if hint.get("q99_over_q50") is not None and hint["q99_over_q50"] > 10:
            lines.append(f"- ⚠️ 目标重尾(q99/q50={hint['q99_over_q50']:.1f}):"
                         "评估 log1p/分段/加权口径是否更贴业务决策")
    if s.get("unique_entities") is not None:
        lines.append(f"- unique_entities: {s['unique_entities']} "
                     f"(dup_rate={s.get('entity_dup_rate', 0):.2%})")
    lines.append(f"- duplicate_row_rate: {s.get('duplicate_row_rate', 0):.2%}")
    ts = s.get("treatment_summary")
    if ts:
        lines.append(f"- treatment({ts['column']}): treated={ts['n_treated']} "
                     f"control={ts['n_control']} "
                     f"rate={ts['treatment_rate']:.2%}")
        if ts.get("ate_observed") is not None:
            lines.append(f"- ATE(观测): {ts['ate_observed']:.4f} "
                         f"(treated={ts['response_rate_treated']:.4f} vs "
                         f"control={ts['response_rate_control']:.4f})")
    fs = p["feature_summary"]
    lines += [
        "", "## 特征质量摘要", "",
        f"- constant: {len(fs['constant_features'])}",
        f"- missing>50%: {len(fs['high_missing_features'])}",
        f"- high-cardinality: {len(fs['high_cardinality_features'])}",
    ]
    ivs = p["iv_summary"]
    if ivs.get("supported") and ivs.get("mode") == "regression_correlation":
        lines += ["", "## Top 相关特征(回归,|Spearman corr|)", "",
                  "| feature | corr |", "|---|---|"]
        for r in ivs["top_features"][:15]:
            c = r.get("corr")
            c = f"{c:.4f}" if isinstance(c, float) and np.isfinite(c) else "-"
            lines.append(f"| {r['feature']} | {c} |")
    elif ivs.get("supported") and ivs.get("mode") == "uplift_split_iv":
        lines += ["", "## Top uplift 调节特征(|iv_treated − iv_control|)", "",
                  "| feature | iv_treated | iv_control | iv_diff |",
                  "|---|---|---|---|"]
        for r in ivs.get("top_moderators", [])[:15]:
            def _f(v):
                return f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else "-"
            lines.append(f"| {r['feature']} | {_f(r.get('iv_treated'))} "
                         f"| {_f(r.get('iv_control'))} | {_f(r.get('iv_diff'))} |")
    elif ivs.get("supported"):
        lines += ["", "## Top IV 特征", "",
                  "| feature | iv | univariate_auc |", "|---|---|---|"]
        for r in ivs["top_features"][:15]:
            auc = r.get("univariate_auc")
            auc = f"{auc:.4f}" if isinstance(auc, float) and np.isfinite(auc) else "-"
            lines.append(f"| {r['feature']} | {r['iv']:.4f} | {auc} |")
    qs = p.get("quality_levels") or {}
    if qs:
        iv_label = ("corr 等级(回归)"
                    if (p.get("iv_summary") or {}).get("mode")
                    == "regression_correlation" else "IV 等级")
        lines += ["", f"## {iv_label} × 稳定性判别", ""]
        vc = qs.get("iv_level_counts") or {}
        lines += ["| iv_level | 数量 |", "|---|---|"]
        for k in ("suspicious", "strong", "medium", "weak", "useless",
                  "no_value"):
            lines.append(f"| {k} | {vc.get(k, 0)} |")
        sc = qs.get("stability_flag_counts") or {}
        lines += ["", "| stability_flag | 数量 |", "|---|---|"]
        for k, v in sorted(sc.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v} |")
        lines.append("")
        if (p.get("iv_summary") or {}).get("mode") == "regression_correlation":
            lines.append("- suspicious(|corr|≥0.5)进入反泄漏复核清单;"
                         "stability_flag≠OK 进入漂移处置清单")
        else:
            lines.append("- suspicious(IV≥0.5)进入反泄漏复核清单;"
                         "stability_flag≠OK 进入漂移处置清单")
        lines.append("- 明细见 feature-quality.csv 的 iv_level / *_level / "
                     "stability_flag 列")
    ds = p["drift_summary"]
    lines += ["", "## 漂移(PSI)", "",
              f"- comparisons: {ds.get('n_comparisons', 0)}",
              f"- PSI>0.1(vs validation)特征数: "
              f"{len(ds.get('features_psi_gt_010_vs_validation', []))}"]
    # v2.17.2(E2):token 级漂移(reparse/文本列 × 月)
    td = ds.get("token_drift") or {}
    if td.get("columns"):
        tp = td["params"]
        lines += ["", "## token 漂移(reparse/文本列 × 月,v2.17.2)", "",
                  f"- 列: {', '.join(td['columns'])}"
                  f"(每列 train 覆盖率 top-{tp['top_k']} token;"
                  f"每月抽样 ≤{tp['rows_per_month']} 行)",
                  f"- 覆盖率塌缩: {td.get('n_collapsed', 0)} 条"
                  f"(基线≥{tp['min_baseline_coverage']:.0%};整行某月"
                  f" <基线×{tp['collapse_ratio']:.0%},或序列首/尾 "
                  f"{tp.get('edge_frac', 0):.0%} token 窗某月"
                  f" <基线×{tp.get('edge_collapse_ratio', 0):.0%})"]
        if td.get("collapsed_tokens"):
            lines += ["", "| column | token | 口径 | 基线覆盖率 | 塌缩起始月 | "
                          "最低比值 | 持续 |", "|---|---|---|---|---|---|---|"]
            for c in td["collapsed_tokens"][:20]:
                lines.append(
                    f"| {c['column']} | {c['token']} | {c.get('scope', 'row')} "
                    f"| {c['baseline_coverage']:.2%} | {c['collapse_month']} "
                    f"| {c['min_coverage_ratio']:.2f} "
                    f"| {'是' if c.get('persistent') else '否'} |")
            lines.append("- ⚠️ 疑似日志断档/口径变更:依赖这些 token 的派生特征"
                         "(尤其近因/短窗计数)在塌缩月之后失真;明细见 "
                         "drift-profile.csv(kind=token)")
    lc = p["leakage"]["severity_counts"]
    lines += ["", "## Leakage", "",
              f"- critical: {lc['critical']}, warning: {lc['warning']}, "
              f"info: {lc['info']}",
              f"- 明细见 [leakage-audit.md](leakage-audit.md)"]
    ca = p.get("compression_audit")
    if ca:
        lines += ["", "## 信息压缩检测(data_view 候选,v8)", "",
                  f"- risk_level: **{ca.get('risk_level')}**"]
        if ca.get("rows_per_entity") is not None:
            lines.append(f"- rows_per_entity: {ca['rows_per_entity']}")
        if ca.get("zero_variance_col_share") is not None:
            lines.append(f"- zero_variance_col_share(抽样): "
                         f"{ca['zero_variance_col_share']}")
        if ca.get("loader_patterns"):
            lines += ["", "**data-loader 压缩模式命中**:", "",
                      "| pattern | line | code |", "|---|---|---|"]
            for pat in ca["loader_patterns"][:10]:
                lines.append(f"| {pat['pattern']} | {pat['line']} "
                             f"| `{pat['text'][:80]}` |")
        if ca.get("data_view_candidates"):
            lines += ["", "**data_view 候选**(经 write-splits-view 落地):",
                      ""]
            for c in ca["data_view_candidates"]:
                lines += [f"- `{c['view_id_hint']}`: {c['suggested_action']}",
                          f"  - trigger: {c['trigger']}",
                          f"  - rationale: {c['rationale']}"]
        else:
            lines.append("- 无候选(risk_level 未达 suspect)")
    return "\n".join(lines) + "\n"
