"""v2.18.0 M3:证据关联与证据加权(dev 文档 §7.4,Q19)。

- :func:`infer_trial_topics`:trial 的主题由 LLM 声明的 ``topics``、``family_hints``、
  ``changed_modules`` 与 action 推断(Q6:经验与 trial 两边都打主题);
- :func:`link_snapshot` / :func:`relink`:按「主题(含父子)+ 口径」把 trial 结果
  关联到原则,记 support / refute / neutral。显式依据(``--basis``)记 explicit,其余
  passive。``relink`` 用于新原则出现后回补历史 trial 的证据;
- :func:`principle_stats`:按证据加权汇总每条原则的支持度与置信度,替代旧的
  Beta 后验(只看显式引用、长期为 0)。

权重:explicit 1.0 / passive 0.5;同任务 1.0 / 同底座 0.7 / 跨底座与未知 0.3;
被动证据只认主题精确命中(主主题 1.0、其余 0.6);时间衰减半衰期 60 天。
"""
from __future__ import annotations

import datetime as _dt
import re
from collections import defaultdict

from sqlalchemy import and_, select

from . import schema as S
from .curation import _topic_resolver, principles_current, topics_current
from .snapshot import parse_atid
from .store import _rows, clean_json

#: prefilled(v2.18.6):trial 原样沿用了 plan 预填的依据 —— 程序按主题命中给的引用,
#: 不代表 Agent 的判断,按被动权重计,也不计入 n_explicit
BINDING_W = {"explicit": 1.0, "prefilled": 0.5, "passive": 0.5}
BASE_W = {"same_task": 1.0, "same_base": 0.7, "cross_base": 0.3, "unknown": 0.3}
HALF_LIFE_DAYS = 60.0
#: 被动匹配只用于"建模动作"类原则:trial 的 Δ 量的是这类动作的效果;评估方法与工程类
#: 原则的对错不能由某个 trial 的 Δ 判读,只接受显式引用
PASSIVE_CATEGORIES = ("feature_engineering", "architecture", "hpo")
#: family_hints 中 info:<列>:<维度> 的维度 → 主题
DIM_TOPICS = {"window": "feature.seq.window_count", "aggregation": "feature.seq.window_count",
              "recency": "feature.seq.recency", "sequence_structure": "feature.seq.funnel",
              "bucketing": "feature.seq.periodicity", "periodicity": "feature.seq.periodicity",
              "derived_interaction": "feature.base.interaction",
              "missing": "feature.missing"}
MODULE_TOPICS = {"hpo": "hpo.search", "hyperparameters": "hpo.search", "loss": "model.loss",
                 "model": "model.gbm"}
ACTION_TOPICS = {"hpo": "hpo.search", "confirm": "eval.confirm"}
_MODEL_TYPES = {"binary": "binary_classification", "reg": "regression", "uplift": "uplift"}
_TRANSFORMS = (("_log1p", "log1p"), ("_log", "log"), ("_sqrt", "sqrt"))


# --------------------------------------------------------------------------
# trial 主题与口径
# --------------------------------------------------------------------------

def _hint_candidates(hint: str) -> list[str]:
    h = str(hint or "").strip()
    out = [h]
    stripped = re.sub(r"[_\d]+$", "", h)
    if stripped and stripped != h:
        out += [stripped, stripped + "_"]
    return out


def infer_trial_topics(row: dict, resolver: dict[str, str]) -> tuple[list[str], list[str]]:
    """返回 (topics, objects)。topics 去掉被更具体子主题覆盖的父主题,最多 4 个。"""
    found: list[str] = []
    objects: list[str] = []

    def add(code):
        if code and code not in found:
            found.append(code)

    for t in row.get("topics") or []:
        add(resolver.get(str(t).lower()))
    # 联合面 trial 的 family_hints 先列已在面上的族、最后才是本次新增的族:倒序读,
    # 让上限截断时保留的是本次改动
    for hint in reversed(row.get("family_hints") or []):
        h = str(hint)
        if h.startswith("info:"):
            parts = h.split(":")
            col = parts[1] if len(parts) > 1 else ""
            dim = parts[-1] if len(parts) > 2 else ""
            if col and col not in objects:
                objects.append(col)
            col_topic = next((resolver[c.lower()] for c in _hint_candidates(col)
                              if c.lower() in resolver), None)
            add(col_topic if col_topic and col_topic.count(".") >= 2 else
                DIM_TOPICS.get(dim) or col_topic)
            continue
        hit = next((resolver[c.lower()] for c in _hint_candidates(h) if c.lower() in resolver), None)
        if hit:
            add(hit)
            if h.endswith("_") and h not in objects:
                objects.append(h)
    for m in row.get("changed_modules") or []:
        add(MODULE_TOPICS.get(str(m)))
    add(ACTION_TOPICS.get(str(row.get("action") or "")))
    found = [c for c in found if c]
    specific = [c for c in found if not any(o != c and o.startswith(c + ".") for o in found)]
    return specific[:4], objects[:10]


def trial_scope(row: dict) -> dict | None:
    p = parse_atid(row.get("atid"))
    if not p:
        return None
    mt = _MODEL_TYPES.get(p["model_type"])
    target = p["target_name"] or ""
    tt = next((v for suf, v in _TRANSFORMS if target.endswith(suf)), "raw")
    return {"model_type": mt, "primary_metric": str(row.get("primary_metric") or "").lower(),
            "target_transform": tt}


WILDCARD = "*"
_SCOPE_KEYS = ("model_type", "primary_metric", "target_transform")


def _scope_tuple(s: dict) -> tuple:
    return (s.get("model_type"), str(s.get("primary_metric") or "").lower(),
            str(s.get("target_transform") or "raw").lower())


def scope_matches(a: dict | None, b: dict | None) -> bool:
    """口径严格一致(证据关联与"同口径"判定用)。任一侧写 ``*`` 的字段视为匹配
    (v2.18.1:评估 / 工程类原则可声明与指标、目标变换无关)。"""
    if not a or not b:
        return False
    return all(x == y or WILDCARD in (x, y) for x, y in zip(_scope_tuple(a), _scope_tuple(b)))


#: 口径无关的原则类别:评估方法与工程踩坑与具体指标无关;engineering 连任务类型都无关
SCOPE_AGNOSTIC = {"evaluation": ("model_type",), "engineering": ()}


def scope_level(task: dict | None, principle: dict | None,
                category: str | None = None) -> str | None:
    """检索用的分级口径匹配(v2.18.1):

    - ``exact``:严格一致(含通配);
    - ``metric_family``:任务类型与目标变换一致、主指标同族(auc/gini/ks 同属排序类);
    - ``agnostic``:评估类只要求任务类型一致,工程类不要求 —— 它们按设计与指标无关;
    - ``None``:不可用。
    """
    from .task_profile import metric_family

    if not task or not principle:
        return None
    if scope_matches(task, principle):
        return "exact"
    tm, tp, tt = _scope_tuple(task)
    pm, pp, pt = _scope_tuple(principle)
    if tm == pm and (tt == pt or WILDCARD in (tt, pt)) and tp and pp and \
            metric_family(tp) == metric_family(pp):
        return "metric_family"
    keys = SCOPE_AGNOSTIC.get(str(category or ""))
    if keys is not None and all((task.get(k) == principle.get(k)) or
                                principle.get(k) == WILDCARD for k in keys):
        return "agnostic"
    return None


def topic_relevance(q_topics, p_topics) -> float:
    """精确 1.0;一方是另一方的子主题 0.6(只在两层及以上的主题间,顶层词不参与)。"""
    best = 0.0
    for q in q_topics or []:
        for p in p_topics or []:
            if q == p:
                return 1.0
            if "." in q and "." in p and (p.startswith(q + ".") or q.startswith(p + ".")):
                best = max(best, 0.6)
    return best


# --------------------------------------------------------------------------
# 关联
# --------------------------------------------------------------------------

def _gain_ci(row: dict) -> tuple[float | None, float | None, float | None]:
    lo, hi, d = row.get("delta_parent_ci_low"), row.get("delta_parent_ci_high"), \
        row.get("delta_parent")
    if row.get("higher_is_better") is False:
        lo, hi, d = (-hi if hi is not None else None), (-lo if lo is not None else None), \
            (-d if d is not None else None)
    return d, lo, hi


def _relation_from_ci(direction: int, lo: float | None, hi: float | None) -> str | None:
    """按方向与增益 CI 判定支持 / 反驳 / 中性;缺 CI 时返回 None(由调用方兜底)。"""
    if lo is None or hi is None:
        return None
    if direction == 1:
        return "support" if lo > 0 else ("refute" if hi < 0 else "neutral")
    if direction == 0:
        # 「无增益」原则要有中性带:CI 跨越 0 的 trial 什么也没证明,既不是支持也不是
        # 反驳。旧实现把 lo<=0 一律记 support,会让没有分辨力的 trial 抬高后验(正式库
        # 311 条 direction=0 的 support 里 187 条 CI 跨 0),进而影响置信度与避免清单。
        return "refute" if lo > 0 else ("support" if hi < 0 else "neutral")
    return "support" if hi < 0 else ("refute" if lo > 0 else "neutral")


def relation_for(direction: int, row: dict) -> str | None:
    """trial 读数对原则方向的支持 / 反驳 / 中性;无法判读时返回 None。"""
    d, lo, hi = _gain_ci(row)
    rel = _relation_from_ci(direction, lo, hi)
    if rel is not None:
        return rel
    v = row.get("verdict")
    if v in ("accepted", "confirmed"):
        lo, hi = 1e-12, 1.0
    elif v == "rejected":
        lo, hi = -1.0, 0.0
    else:
        return None
    return _relation_from_ci(direction, lo, hi)


def base_relation(trial_atid: str | None, principle: dict, canon) -> str:
    t = canon(trial_atid) if trial_atid else None
    ev = principle.get("evidence") if isinstance(principle.get("evidence"), dict) else {}
    atids = {canon(a) for a in (ev.get("abstract_task_ids") or []) if a}
    if not t or not atids:
        return "unknown"
    if t in atids:
        return "same_task"
    tb = (parse_atid(t) or {}).get("data_base")
    bases = {(parse_atid(a) or {}).get("data_base") for a in atids} - {None}
    if tb and tb in bases:
        return "same_base"
    return "cross_base" if tb and bases else "unknown"


def _explicit_refs(row: dict) -> dict[str, str]:
    """trial 显式依据 → {principle_id: declared relation}。"""
    out: dict[str, str] = {}
    extra = row.get("extra") or {}
    basis = extra.get("basis") or []
    if isinstance(basis, dict):          # new-trial 写入的 {novel, refs}
        basis = basis.get("refs") or []
    for item in basis:
        if isinstance(item, dict) and item.get("id"):
            out[str(item["id"])] = str(item.get("relation") or "follows")
    for eid in row.get("experience_ids") or []:
        out.setdefault(str(eid), "follows")
    return out


def basis_prefilled(row: dict) -> bool:
    """依据是否原样沿用 plan 预填值(v2.18.6 起 new-trial 记 ``basis.source``)。"""
    b = (row.get("extra") or {}).get("basis") or row.get("basis")
    return isinstance(b, dict) and b.get("source") == "prefilled"


# P1-C: changed_modules → 主题类别映射
# trial 声明的 changed_modules 严格枚举 feature/sample/model/loss/data;
# 把它们映射到原则 topics 的顶层分类,用于被动关联的交叉权重折扣。
_MODULE_CATEGORY: dict[str, set[str]] = {
    "feature": {"feature"},
    "model":   {"model", "hpo"},
    "loss":    {"model"},
    "sample":  {"eval", "engineering"},
    "data":    {"eval", "engineering"},
}


def _changed_modules_weight(changed_modules, action: str | None,
                            p_topics: list[str]) -> float:
    """P1-C:按 changed_modules 与原则主题的模块类别交叉校验被动关联权重。

    - 若 trial 未声明 changed_modules:返回 1.0(无信息,不折扣)。
    - 若原则主题顶层类别与 trial 的模块类别**全不相交**:返回 0.1(几乎不相关)。
    - 若有交叉(trial 确实改动了原则所涉模块):返回 1.0(不折扣)。
    action 为 "hpo" 时自动把 "model" 纳入模块集合。
    """
    if not changed_modules:
        return 1.0
    # 收集 trial 影响的模块类别
    trial_cats: set[str] = set()
    for m in (changed_modules if isinstance(changed_modules, list) else [changed_modules]):
        trial_cats.update(_MODULE_CATEGORY.get(str(m).lower(), set()))
    if action and str(action).lower() == "hpo":
        trial_cats.update({"model", "hpo"})
    # 提取原则主题的顶层类别(第一段,如 "feature.seq.window_count" → "feature")
    p_cats: set[str] = set()
    for pt in p_topics:
        top = str(pt).split(".")[0].lower()
        p_cats.add(top)
    # 两边都有信息时做交叉校验
    if trial_cats and p_cats and trial_cats.isdisjoint(p_cats):
        return 0.1  # 跨模块类别,几乎不相关,显著折扣
    return 1.0


def build_links(trials: list[dict], principles: list[dict], resolver: dict[str, str],
                canon, *, trial_times: dict | None = None) -> list[dict]:
    links = []
    by_id = {p["principle_id"]: p for p in principles}
    for t in trials:
        scope = trial_scope(t)
        topics, objects = infer_trial_topics(t, resolver)
        explicit = _explicit_refs(t)
        binding = "prefilled" if basis_prefilled(t) else "explicit"
        d, lo, hi = _gain_ci(t)
        at = (trial_times or {}).get((t["run_id"], t["trial_id"]))
        seen = set()
        for pid, declared in explicit.items():
            p = by_id.get(pid)
            if p is None:
                continue
            rel = relation_for(p["direction"], t)
            if rel is None:
                continue
            br = base_relation(t.get("atid"), p, canon)
            links.append({"principle_id": pid, "run_id": t["run_id"], "trial_id": t["trial_id"],
                          "binding": binding, "relation": rel, "declared": declared,
                          "base_relation": br, "match": {"topics": topics},
                          "gain": d, "ci_low": lo, "ci_high": hi,
                          "weight": BINDING_W[binding] * BASE_W[br], "trial_at": at})
            seen.add(pid)
        if not topics or not scope:
            continue
        for p in principles:
            if p["principle_id"] in seen or p.get("category") not in PASSIVE_CATEGORIES \
                    or not scope_matches(scope, p["scope"]):
                continue
            # 被动证据只认主题精确命中:父子命中只用于检索相关度。原则主题常是通道级 /
            # 条件性的结论,子方向 trial 的读数对它只有间接意义(实测父子命中让"后续延伸
            # 无增益"的 trial 大量反驳"首轮加宽有效")。主主题(第一个)权重 1.0,其余 0.6。
            # 原则主题先过 resolver:trial 主题来自 `--topics`(已归一),原则若还写旧词,
            # 精确命中会失败。
            p_topics = [resolver.get(str(pt).lower(), pt) for pt in p["topics"]]
            exact = [i for i, pt in enumerate(p_topics) if pt in topics]
            if not exact:
                continue
            rel_t = 1.0 if exact[0] == 0 else 0.6
            br = base_relation(t.get("atid"), p, canon)
            obj_hit = sorted(set(objects) & set(p.get("objects") or []))
            # 同底座且双方都有具体对象却不相交:同类动作、不同对象,证据权重减半
            obj_w = 0.5 if (br in ("same_task", "same_base") and p.get("objects")
                            and objects and not obj_hit) else 1.0
            # P1-C: changed_modules 交叉校验 —— trial 改动的模块类别若与原则主题对应
            # 的模块类别不相交,说明这个 trial 与该原则所描述的变更方向无关,降权避免
            # 假反驳(例如改了 sample 权重的 trial 反驳了 feature_engineering 原则)。
            cm_w = _changed_modules_weight(t.get("changed_modules"), t.get("action"), p_topics)
            rel = relation_for(p["direction"], t)
            if rel is None:
                continue
            links.append({"principle_id": p["principle_id"], "run_id": t["run_id"],
                          "trial_id": t["trial_id"], "binding": "passive", "relation": rel,
                          "declared": None, "base_relation": br,
                          "match": {"topics": topics, "relevance": rel_t, "objects": obj_hit},
                          "gain": d, "ci_low": lo, "ci_high": hi,
                          "weight": BINDING_W["passive"] * BASE_W[br] * rel_t * obj_w * cm_w,
                          "trial_at": at})
    return links


def _context(store, namespace: str, at: int):
    topics = topics_current(store, namespace, at=at)
    resolver = _topic_resolver(topics)
    with store._read() as conn:
        amap = store._alias_map(conn, at)
    return resolver, (lambda a: store._canonical(a, amap) if a else a)


def _trial_times(store, namespace: str, at: int) -> dict:
    """(run_id, trial_id) → 时间:快照 run.created_at,缺失时取提交时间。"""
    with store._read() as conn:
        snaps = _rows(conn.execute(
            select(S.snapshots.c.snapshot_id, S.snapshots.c.meta, S.commits.c.created_at)
            .join(S.commits, S.commits.c.seq == S.snapshots.c.commit_seq)
            .where(and_(S.snapshots.c.namespace == namespace,
                        S.snapshots.c.commit_seq <= at))))
    when = {s["snapshot_id"]: ((s["meta"] or {}).get("run") or {}).get("created_at")
            or s["created_at"] for s in snaps}
    out = {}
    for t in store.trial_outcomes(namespace, at=at):
        out[(t["run_id"], t["trial_id"])] = when.get(t["snapshot_id"])
    return out


def _existing_keys(store, namespace: str) -> set:
    with store._read() as conn:
        return {tuple(r) for r in conn.execute(select(
            S.evidence_links.c.principle_id, S.evidence_links.c.run_id,
            S.evidence_links.c.trial_id, S.evidence_links.c.binding).where(
            S.evidence_links.c.namespace == namespace))}


def _write_links(store, namespace: str, links: list[dict], *, actor: dict, ref: str | None,
                 message: str) -> dict:
    have = _existing_keys(store, namespace)
    new = []
    for lk in links:
        key = (lk["principle_id"], lk["run_id"], lk["trial_id"], lk["binding"])
        if key not in have:
            have.add(key)
            new.append(lk)
    if not new:
        return {"links": 0, "commit_seq": None}
    with store._write() as conn:
        seq = store._new_commit(conn, namespace=namespace, kind="evidence", actor=actor,
                                ref=ref, message=message)
        for i in range(0, len(new), 500):
            store._insert(conn, S.evidence_links, clean_json(new[i:i + 500]),
                          namespace=namespace, seq=seq)
    return {"links": len(new), "commit_seq": seq}


def link_snapshot(store, namespace: str, snapshot_id: str, *, actor: dict) -> dict:
    """快照提交后:为其 trial 结果建立证据关联(对当前原则)。"""
    at = store.watermark()
    with store._read() as conn:
        trials = _rows(conn.execute(select(S.trial_outcomes).where(and_(
            S.trial_outcomes.c.namespace == namespace,
            S.trial_outcomes.c.snapshot_id == snapshot_id))))
    if not trials:
        return {"links": 0, "commit_seq": None}
    resolver, canon = _context(store, namespace, at)
    links = build_links(trials, principles_current(store, namespace, at=at), resolver, canon,
                        trial_times=_trial_times(store, namespace, at))
    for lk in links:
        lk["source_snapshot_id"] = snapshot_id
    return _write_links(store, namespace, links, actor=actor, ref=snapshot_id,
                        message=f"evidence for snapshot {snapshot_id[:8]}")


def relink(store, namespace: str, *, actor: dict) -> dict:
    """对全部 trial 结果 × 当前原则回补证据(只追加尚不存在的关联)。"""
    at = store.watermark()
    resolver, canon = _context(store, namespace, at)
    links = build_links(store.trial_outcomes(namespace, at=at),
                        principles_current(store, namespace, at=at), resolver, canon,
                        trial_times=_trial_times(store, namespace, at))
    return _write_links(store, namespace, links, actor=actor, ref=None,
                        message="relink: 全部 trial × 当前原则")


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------

def _age_days(when: str | None, now: _dt.datetime) -> float:
    if not when:
        return 0.0
    try:
        t = _dt.datetime.fromisoformat(str(when))
    except ValueError:
        return 0.0
    if t.tzinfo is None:
        t = t.replace(tzinfo=now.tzinfo)
    return max(0.0, (now - t).total_seconds() / 86400.0)


def confidence_from(w_support: float, w_refute: float, *,
                    n_explicit: int = 0) -> tuple[float, str]:
    """后验 = (支持 + 1) / (支持 + 反驳 + 2)。只有被动证据时反驳占优记 mixed:被动匹配
    不知道原则的适用条件(如"cap 被撞时"),不能据此判原则被推翻。"""
    mass = w_support + w_refute
    post = (w_support + 1.0) / (mass + 2.0)
    if mass >= 3 and post >= 0.75:
        level = "high"
    elif mass >= 1 and post >= 0.6:
        level = "medium"
    elif mass >= 1 and post < 0.4:
        level = "contradicted" if n_explicit else "mixed"
    else:
        level = "low"
    return post, level


#: v2.18.2 多因素降权的指数:被动关联权重 × n^-power,n = trial 推断出的主题数。
#: **默认关闭**:``library replay`` 在正式库(水位 179)上的留一 run 评测里,开启后证据后验
#: 区分 support / refute 的 AUC 从 0.631 降到 0.625(0.5)/ 更低(1.0)。dev 文档 §22
MULTI_FACTOR_POWER = 0.0


def _multi_factor_w(lk: dict, power: float) -> float:
    if not power or lk.get("binding") != "passive":
        return 1.0
    m = lk.get("match") if isinstance(lk.get("match"), dict) else {}
    n = len(m.get("topics") or []) or 1
    return n ** -power


def aggregate(links: list[dict], *, now: _dt.datetime | None = None,
              directions: dict[str, int] | None = None, run_dedup: bool = False,
              multi_factor: float = MULTI_FACTOR_POWER) -> dict[str, dict]:
    """按证据加权汇总。

    ``directions`` 给出每条原则当前的方向时,**用关联里存的增益 CI 重新判定**
    支持 / 反驳 / 中性,而不是沿用写入时算好的 ``relation``。这样 relation 语义
    修正(direction=0 的中性带)对已入库的关联立即生效 —— 关联行本身不动,保持只追加;
    CI 缺失时才回落到写入时的判定。

    v2.18.2 两个可选口径(只在读侧,默认都关闭,供 ``library replay`` 对比):

    - ``run_dedup``:一个 run 对一条原则的总权重不超过其中最强的一条关联,支持 / 反驳 /
      中性按权重占比分摊;跨底座迁移战绩按 run 计。动机是同一 run 的多个 trial 会重复
      投票(正式库 1291 条关联只来自 456 个 (原则, run) 对)。但留一 run 评测显示
      trial 级的重复恰好是"下一个同方向 trial 成不成"的有效信号,去重后后验区分
      support / refute 的 AUC 从 0.631 降到 0.599,所以不作默认。
    - ``multi_factor``:被动关联按 trial 主题数降权(见 :data:`MULTI_FACTOR_POWER`)。
    """
    now = now or _dt.datetime.now(_dt.timezone.utc).astimezone()
    acc: dict[str, dict] = defaultdict(lambda: {
        "w_support": 0.0, "w_refute": 0.0, "w_neutral": 0.0, "n_links": 0, "n_explicit": 0,
        "runs": set(), "cross_support": 0.0, "cross_refute": 0.0})
    groups: dict[tuple, list] = defaultdict(list)
    for i, lk in enumerate(links):
        rel = lk["relation"]
        d = (directions or {}).get(lk["principle_id"])
        if d is not None:
            rel = _relation_from_ci(d, lk.get("ci_low"), lk.get("ci_high")) or rel
        w = float(lk["weight"]) * 0.5 ** (_age_days(lk.get("trial_at"), now) / HALF_LIFE_DAYS)
        w *= _multi_factor_w(lk, multi_factor)
        key = (lk["principle_id"], lk["run_id"]) if run_dedup else (lk["principle_id"], i)
        groups[key].append((lk, rel, w))
    for (pid, _k), items in groups.items():
        a = acc[pid]
        total = sum(w for _, _, w in items)
        scale = (max(w for _, _, w in items) / total) if total > 0 else 0.0
        cross = {"support": 0.0, "refute": 0.0}
        for lk, rel, w in items:
            a[f"w_{rel}"] += w * scale
            a["n_links"] += 1
            a["n_explicit"] += lk["binding"] == "explicit"
            a["runs"].add(lk["run_id"])
            if lk["base_relation"] == "cross_base" and rel in cross:
                cross[rel] += w
        if cross["support"] > cross["refute"]:
            a["cross_support"] += 1
        elif cross["refute"] > cross["support"]:
            a["cross_refute"] += 1
    out = {}
    for pid, a in acc.items():
        post, level = confidence_from(a["w_support"], a["w_refute"], n_explicit=a["n_explicit"])
        out[pid] = {"w_support": round(a["w_support"], 4), "w_refute": round(a["w_refute"], 4),
                    "w_neutral": round(a["w_neutral"], 4), "n_links": a["n_links"],
                    "n_explicit": a["n_explicit"], "n_runs": len(a["runs"]),
                    "posterior": round(post, 4), "confidence": level,
                    "transfer": round((a["cross_support"] + 1) /
                                      (a["cross_support"] + a["cross_refute"] + 2), 4)}
    return out


def principle_directions(store, namespace: str, *, at: int | None = None) -> dict[str, int]:
    """{principle_id: direction},含已失效版本(历史证据仍要按当时的原则判读)。"""
    from .curation import principles_current

    at = store._at(at)
    return {p["principle_id"]: int(p["direction"])
            for p in principles_current(store, namespace, at=at, include_inactive=True)}


def evidence_links(store, namespace: str, *, at: int | None = None,
                   principle_id: str | None = None) -> list[dict]:
    at = store._at(at)
    with store._read() as conn:
        q = select(S.evidence_links).where(and_(S.evidence_links.c.namespace == namespace,
                                                S.evidence_links.c.commit_seq <= at))
        if principle_id:
            q = q.where(S.evidence_links.c.principle_id == principle_id)
        return _rows(conn.execute(q.order_by(S.evidence_links.c.id)))


def principle_stats(store, namespace: str, *, at: int | None = None) -> dict[str, dict]:
    at = store._at(at)
    return aggregate(evidence_links(store, namespace, at=at),
                     directions=principle_directions(store, namespace, at=at))
