"""v2.18.1:多维任务画像与任务相似度(分级检索的 L1–L3 依据)。

atid 只能回答"是不是同一个任务";介于"同一个"与"无关"之间的**相关任务**(同业务、
同目标、列集部分重合)需要多维画像才能识别。画像描述数据与任务本身,不分命名空间,
由 run 在固定视图时从 task-spec 与原生列算出,随快照提交,服务端按 atid 追加保存
(``fingerprint_profiles``,读侧取最新)。

画像字段(全部可缺省,缺的维度不参与相似度):
- ``model_type`` / ``target_name`` / ``target_base``(去掉 _log1p 等变换后缀)
- ``business_domains`` / ``objective`` / ``label``(正样本定义、观测 / 表现窗口)
- ``split_type`` / ``temporal``(样本空间的切分方式)
- ``sample_scale`` / ``n_columns`` / ``col_minhash``(列名 MinHash)/ ``col_families``
- ``metric`` / ``metric_family``

本模块是纯函数,不依赖 runtime 其它模块(spec / profile 由调用方传入)。
"""
from __future__ import annotations

import hashlib
import re

from . import domains as D

PROFILE_SCHEMA = "modelevo.task-profile/1"
MINHASH_K = 64
_P = (1 << 61) - 1
#: 目标变换后缀(与 evidence._TRANSFORMS 同口径,多一个 boxcox)
TRANSFORM_SUFFIXES = ("_log1p", "_log", "_sqrt", "_boxcox")
#: 指标族:同族指标的结论可在放宽档互通(排序类 auc/gini/ks 单调相关)
METRIC_FAMILY = {
    "auc": "rank", "roc_auc": "rank", "gini": "rank", "ks": "rank", "pr_auc": "rank",
    "ap": "rank", "average_precision": "rank",
    "logloss": "prob", "log_loss": "prob", "brier": "prob",
    "rmse": "reg_error", "mse": "reg_error", "mae": "reg_error", "rmsle": "reg_error",
    "mape": "reg_error", "r2": "reg_fit",
    "qini": "uplift", "auuc": "uplift", "uplift_auc": "uplift",
}
#: 相似度各维度权重(缺失维度不计入分母)。objects 是"原则 objects 推出的列族"重合度,
#: 只覆盖列集的一小部分 —— 单独权重低,且不足以把任务判成"近似任务"。
SIM_WEIGHTS = {"columns": 0.35, "target": 0.25, "domain": 0.25, "objects": 0.10,
               "scale": 0.10, "sample": 0.05}
_SCALES = ("lt10w", "10w_50w", "50w_200w", "200w_1000w", "ge1000w")


def metric_family(metric: str | None) -> str | None:
    m = str(metric or "").lower()
    return METRIC_FAMILY.get(m, m or None)


def target_base(target: str | None) -> str:
    t = str(target or "")
    for suf in TRANSFORM_SUFFIXES:
        if t.endswith(suf):
            return t[: -len(suf)]
    return t


def _h64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


_COEF = [((_h64(f"a{i}") % (_P - 1)) + 1, _h64(f"b{i}") % _P) for i in range(MINHASH_K)]


def minhash(columns) -> list[int] | None:
    """列名集合的 MinHash 签名(K=64,估计 Jaccard 的标准误约 0.06)。"""
    hs = [_h64(str(c)) % _P for c in set(columns or [])]
    if not hs:
        return None
    return [min((a * h + b) % _P for h in hs) for a, b in _COEF]


def minhash_jaccard(a, b) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


_FAMILY_RE = re.compile(r"^([a-z]+(?:_[a-z]+)?)_")


def column_families(columns, *, limit: int = 200) -> list[str]:
    """列名前缀族(如 ``seq_gap_d365_cnt`` → ``seq_gap_``),列 MinHash 缺失时的退路,
    也用于展示"列族重合"。只保留出现 ≥2 次的前缀。"""
    cnt: dict[str, int] = {}
    for c in columns or []:
        m = _FAMILY_RE.match(str(c).lower())
        if m:
            cnt[m.group(1) + "_"] = cnt.get(m.group(1) + "_", 0) + 1
    fams = sorted((f for f, n in cnt.items() if n >= 2), key=lambda f: (-cnt[f], f))
    return sorted(fams[:limit])


def build_profile(*, spec_data: dict | None = None, columns=None, atid: str | None = None,
                  sample_scale: str | None = None) -> dict:
    """由 task-spec 的 ``data``(dict)与原生列名构造画像。"""
    d = spec_data or {}
    tgt = d.get("target") if isinstance(d.get("target"), dict) else {}
    tname = str(tgt.get("name") or "")
    if not sample_scale and atid:
        parts = str(atid).rsplit("-", 2)
        if len(parts) == 3 and parts[1] in _SCALES:
            sample_scale = parts[1]
    split = d.get("split_policy") if isinstance(d.get("split_policy"), dict) else {}
    doms = d.get("business_domains") or []
    if isinstance(doms, str):
        doms = [doms]
    derived = {str(c) for c in ((d.get("data") or {}).get("derived_columns") or [])} \
        if isinstance(d.get("data"), dict) else set()
    cols = [str(c) for c in (columns or []) if str(c) not in derived]
    out = {
        "schema": PROFILE_SCHEMA,
        "model_type": d.get("model_type"),
        "target_name": tname or None,
        "target_base": target_base(tname) or None,
        "business_domains": sorted(D.normalize(doms)),
        "objective": str(d.get("objective_description") or "")[:300] or None,
        "label": {k: tgt.get(k) for k in ("positive_definition", "observation_window",
                                          "performance_window") if tgt.get(k)} or None,
        "split_type": split.get("type"),
        "temporal": bool(d.get("time_column") or split.get("time_column")) or None,
        "metric": str(d.get("primary_metric") or "").lower() or None,
        "metric_family": metric_family(d.get("primary_metric")),
        "sample_scale": sample_scale,
        "n_columns": len(set(cols)) or None,
        "col_minhash": minhash(cols),
        "col_families": column_families(cols) or None,
    }
    return {k: v for k, v in out.items() if v not in (None, [], {})}


def _jac(a, b) -> float | None:
    a, b = set(a or []), set(b or [])
    if not a or not b:
        return None
    return len(a & b) / len(a | b)


def _overlap(a, b) -> float | None:
    """重合系数 |A∩B| / min(|A|,|B|):一侧是另一侧的子集时为 1。

    用于"是否共享"语义的维度 —— 业务域:存量任务的业务域是其全部原则 condition 的
    并集(常有 4–6 个),与新任务的 1 个业务域算 Jaccard 只有 0.2 左右,"同业务"信号
    被稀释;对象列族:原则 objects 只是列族的一小部分,同样只能看是否被包含。"""
    a, b = set(a or []), set(b or [])
    if not a or not b:
        return None
    return len(a & b) / min(len(a), len(b))


def similarity(a: dict | None, b: dict | None) -> tuple[float, dict]:
    """两个画像的加权相似度 ∈ [0,1] 与各维度读数。缺失维度不计入分母;
    双方一个可比维度都没有时返回 0。"""
    a, b = a or {}, b or {}
    dims: dict[str, float] = {}
    # 列:双方都有 MinHash 时用它(真实列集 Jaccard);否则真实列族 Jaccard。
    # 原则 objects 推出的列族(存量任务没有画像时的退路)单独作 objects 维度
    col = minhash_jaccard(a.get("col_minhash"), b.get("col_minhash"))
    if col is None:
        col = _jac(a.get("col_families"), b.get("col_families"))
    if col is not None:
        dims["columns"] = round(col, 3)
    obj = _overlap(a.get("object_families"), b.get("object_families"))
    if obj is not None:
        dims["objects"] = round(obj, 3)
    if a.get("target_base") and b.get("target_base"):
        dims["target"] = 1.0 if a["target_base"] == b["target_base"] else 0.0
    # v2.18.2:业务域按受控词表归一并按层级比较(domains.py);无层级时即重合系数
    dom = D.similarity(a.get("business_domains"), b.get("business_domains"))
    if dom is not None:
        dims["domain"] = round(dom, 3)
    sa, sb = a.get("sample_scale"), b.get("sample_scale")
    if sa in _SCALES and sb in _SCALES:
        gap = abs(_SCALES.index(sa) - _SCALES.index(sb))
        dims["scale"] = 1.0 if gap == 0 else (0.5 if gap == 1 else 0.0)
    if a.get("split_type") and b.get("split_type"):
        dims["sample"] = 1.0 if a["split_type"] == b["split_type"] else 0.0
    den = sum(SIM_WEIGHTS[k] for k in dims)
    if not den:
        return 0.0, dims
    return round(sum(SIM_WEIGHTS[k] * v for k, v in dims.items()) / den, 4), dims


def profile_from_atid(atid: str | None) -> dict:
    """只有 atid 时的最小画像(目标、量级)。"""
    if not atid or "-" not in str(atid):
        return {}
    mt, rest = str(atid).split("-", 1)
    parts = rest.rsplit("-", 2)
    if len(parts) != 3:
        return {}
    out = {"target_name": parts[0], "target_base": target_base(parts[0])}
    if parts[1] in _SCALES:
        out["sample_scale"] = parts[1]
    return out


def describe(dims: dict) -> str:
    """维度读数 → 卡片上的简短说明。"""
    names = {"columns": "列重合", "target": "同目标", "domain": "业务域", "scale": "量级",
             "sample": "切分", "objects": "对象重合"}
    bits = []
    for k in ("target", "columns", "domain", "scale"):
        if k not in dims:
            continue
        if k == "target":
            bits.append("同目标" if dims[k] else "异目标")
        else:
            bits.append(f"{names[k]} {dims[k]:.2f}")
    return "·".join(bits)
