"""v2.18.2:业务域受控词表(dev 文档 §22)。

业务域原先是自由文本,按字符串精确比较:正式库里同属信贷的写法有 credit /
credit_loan / consumer_credit / loan_drawdown / credit_risk 五种,把 ``user_operation``
写成 ``user_ops`` 就会让相关任务在检索里整体落到"通用"档。本模块把写法归一到
``business_domains.yaml`` 的层级代码,并按层级给出相似度。

归一只在**比较与写入时**做,不改已入库的原文:存量画像与原则立即受益,词表以后改了
也不需要迁移数据。不认识的写法原样保留(只做精确比较),由 ``library domains`` 列出。
"""
from __future__ import annotations

import functools
import re
from pathlib import Path

import yaml

VOCAB_FILE = Path(__file__).with_name("business_domains.yaml")
#: 层级关系的相似度
SAME, LINEAGE, SIBLING = 1.0, 0.7, 0.5


def _key(s) -> str:
    return re.sub(r"[\s\-/]+", "_", str(s or "").strip().lower())


@functools.lru_cache(maxsize=1)
def vocabulary() -> dict[str, dict]:
    data = yaml.safe_load(VOCAB_FILE.read_text(encoding="utf-8")) or {}
    out = {}
    for code, spec in (data.get("domains") or {}).items():
        spec = spec or {}
        out[str(code)] = {"code": str(code), "label": spec.get("label") or "",
                          "parent": str(code).rsplit(".", 1)[0] if "." in str(code) else None,
                          "aliases": [str(a) for a in spec.get("aliases") or []]}
    return out


@functools.lru_cache(maxsize=1)
def _resolver() -> dict[str, str]:
    res: dict[str, str] = {}
    for code, d in vocabulary().items():
        # 中文名可能写成"授信申请 / 获客",两半各自可用
        for name in [code, *d["aliases"], *re.split(r"\s*/\s*", d["label"])]:
            if name:
                res.setdefault(_key(name), code)
    return res


def resolve(value) -> tuple[str, bool]:
    """→ (代码或归一后的原文, 是否在词表中)。"""
    k = _key(value)
    code = _resolver().get(k)
    return (code, True) if code else (k, False)


def normalize(values) -> list[str]:
    """归一并去重(保持顺序);空值丢弃。"""
    if isinstance(values, str):
        values = [values]
    out: list[str] = []
    for v in values or []:
        code, _known = resolve(v)
        if code and code not in out:
            out.append(code)
    return out


def unknown(values) -> list[str]:
    if isinstance(values, str):
        values = [values]
    return [_key(v) for v in values or [] if _key(v) and not resolve(v)[1]]


def relation(a: str, b: str) -> float:
    """两个已归一的业务域之间的相似度。"""
    if a == b:
        return SAME
    voc = vocabulary()
    if a not in voc or b not in voc:
        return 0.0
    if b.startswith(a + ".") or a.startswith(b + "."):
        return LINEAGE
    pa, pb = voc[a]["parent"], voc[b]["parent"]
    if pa and pa == pb:
        return SIBLING
    return 0.0


def similarity(a, b) -> float | None:
    """两组业务域的相似度:较小一组里每个域取它在另一组中的最佳匹配,再取平均。

    没有层级时退化为重合系数 |A∩B| / min(|A|,|B|)(v2.18.1 口径):存量任务的业务域
    是其全部原则 condition 的并集,与新任务的 1 个业务域算 Jaccard 会稀释"同业务"信号。
    任一侧为空返回 None(该维度不参与相似度)。"""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return None
    small, big = (na, nb) if len(na) <= len(nb) else (nb, na)
    return sum(max(relation(x, y) for y in big) for x in small) / len(small)
