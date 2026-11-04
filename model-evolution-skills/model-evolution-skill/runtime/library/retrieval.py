"""v2.18.0 M3 / v2.18.1:分级检索与方向表(dev 文档 §7.2 §7.5 §21,Q17 Q20)。

全部是纯函数,输入是 :func:`runtime.library.repository.LocalRepository.run_view`
返回的视图(run 启动时按固定水位取一次并缓存在本机),因此检索不依赖服务在线,
同一水位的检索结果可复现(Q4)。

**级联**(v2.18.1):先精确、再逐级放宽,前一档凑不满 k 才进入下一档;每张卡片标注
档位与匹配理由,让 Agent 知道这是不是放宽后的结论。

====  ==========================================  ==============================
档    条件                                         底座权重
====  ==========================================  ==============================
L0    同任务 + 口径一致                             1.0
L1    同数据底座跨目标,或画像高度相似的近似任务     0.85(近似任务 × 相似度)
      (同目标 + 列重合 ≥0.7)
L2    同业务相关任务(画像相似度 ≥0.4),或 L0/L1    0.6 × (0.5 + 0.5 × 相似度)
      只差"指标同族"
L3    通用:无指纹的历史原则 / 跨业务且有迁移战绩    0.4 / 0.3 × 迁移战绩
      或多底座支持;冷启动(L0–L2 为空)时放开
L4    口径无关:评估 / 工程类原则跨指标、跨任务类型   0.5
====  ==========================================  ==============================

打分:score = 主题相关度 × 档权重 × 证据因子(0.5 + 0.5 × 后验)+ 0.1 × 词面重合。
主题相关度:精确 1.0 / 父子 0.6;无主题命中时只靠假设文本重叠(≥0.3)给 0.3×重叠。

任务相似度来自多维画像(:mod:`runtime.library.task_profile`):视图里的
``task_profile``(本 run)与 ``fingerprint_profiles``(库内各任务)。存量任务没有画像时
由 atid(目标、量级)与库内原则(业务域、对象列前缀)推出退化画像。
"""
from __future__ import annotations

from collections import defaultdict

from . import task_profile as TP
from .curation import overlap, tokens
from .evidence import (_gain_ci, base_relation, infer_trial_topics, scope_level,
                       scope_matches, topic_relevance, trial_scope)  # noqa: F401  (planner_context 经此导入)
from .snapshot import parse_atid

LABEL = {"same_task": "同任务", "same_base": "同数据·跨任务", "unknown": "底座未知",
         "cross_base": "跨底座"}
TIERS = ("L0", "L1", "L2", "L3", "L4")
TIER_LABEL = {"L0": "同任务", "L1": "近似任务", "L2": "相关任务", "L3": "通用·需验证",
              "L4": "方法与工程"}
CLAIM_CHARS = 400
#: L1 近似任务门槛:同目标且列重合足够高(列集小改)
NEAR_SIM, NEAR_COLUMNS = 0.75, 0.7
#: L2 相关任务门槛
RELATED_SIM = 0.4
#: 放宽档(L2 及以下)的最低分,避免用噪声凑满 k
MIN_RELAXED_SCORE = 0.04


def canon_fn(view: dict):
    amap = view.get("aliases") or {}

    def canon(a):
        seen = set()
        while a in amap and amap[a] != a and a not in seen:
            seen.add(a)
            a = amap[a]
        return a
    return canon


def resolver_of(view: dict) -> dict[str, str]:
    from .curation import _topic_resolver

    return _topic_resolver({t["code"]: t for t in view.get("topics") or []})


def _principle_atids(p: dict, canon) -> set[str]:
    ev = p.get("evidence") if isinstance(p.get("evidence"), dict) else {}
    return {canon(a) for a in ev.get("abstract_task_ids") or [] if a}


def _principle_bases(p: dict, canon) -> set[str]:
    return {b for b in ((parse_atid(a) or {}).get("data_base")
                        for a in _principle_atids(p, canon)) if b}


def task_profiles(view: dict) -> dict[str, dict]:
    """{atid: 画像}:库内画像优先;缺失维度由 atid 与绑定该任务的原则补齐
    (业务域 ← 原则 condition.business_domains,列族 ← 原则 objects)。"""
    canon = canon_fn(view)
    stored = {canon(a): dict(p or {}) for a, p in (view.get("fingerprint_profiles") or {}).items()}
    doms: dict[str, set] = defaultdict(set)
    objs: dict[str, set] = defaultdict(set)
    for p in view.get("principles") or []:
        cond = p.get("condition") if isinstance(p.get("condition"), dict) else {}
        for a in _principle_atids(p, canon):
            doms[a].update(str(x).lower() for x in cond.get("business_domains") or [])
            objs[a].update(p.get("objects") or [])
    atids = set(stored) | set(doms) | {canon(f["atid"]) for f in view.get("fingerprints") or []}
    out = {}
    for a in atids:
        prof = {**TP.profile_from_atid(a), **stored.get(a, {})}
        if not prof.get("business_domains") and doms.get(a):
            prof["business_domains"] = sorted(doms[a])
        if objs.get(a):
            # 原则 objects 是列族的一小部分,与真实列族分开存(相似度里用重合系数)
            prof["object_families"] = sorted(objs[a])
        out[a] = prof
    return out


def my_profile(view: dict, profiles: dict | None = None) -> dict:
    canon = canon_fn(view)
    atid = canon(view.get("atid"))
    base = dict((profiles or task_profiles(view)).get(atid) or TP.profile_from_atid(atid))
    own = view.get("task_profile") or {}
    return {**base, **{k: v for k, v in own.items() if v not in (None, [], {})}}


def _best_similarity(mine: dict, p: dict, profiles: dict, canon) -> tuple[float, dict, str | None]:
    best = (0.0, {}, None)
    for a in _principle_atids(p, canon):
        sim, dims = TP.similarity(mine, profiles.get(a) or TP.profile_from_atid(a))
        if sim > best[0]:
            best = (sim, dims, a)
    return best


def _card(p: dict, st: dict, rel: str, score: float, *, show_objects: bool,
          tier: str = "L0", why: list | None = None, scope_match: str = "exact",
          similarity: float | None = None) -> dict:
    claim = str(p["claim"])
    # 置信度来源必须区分:评估 / 工程类原则按设计不建被动证据(它们的对错不能由某个
    # trial 的 Δ 判读),把"没有证据"显示成 low 会让人以为它被证据否定了 —— 而这些
    # 踩坑记录恰恰最该被遵守。无证据时标注 unverified,不冒充 low。
    if st:
        conf, conf_src = st.get("confidence") or "low", "evidence"
    else:
        conf = str(p.get("confidence") or "low")
        conf_src = "author"
    return {"principle_id": p["principle_id"], "version": p.get("version"),
            "direction": p["direction"], "topics": p["topics"],
            "objects": p.get("objects") if show_objects else [],
            "claim": claim if len(claim) <= CLAIM_CHARS else claim[:CLAIM_CHARS] + "…",
            "base_relation": rel, "label": LABEL[rel], "score": round(score, 4),
            "tier": tier, "tier_label": TIER_LABEL[tier], "why": why or [],
            "scope_match": scope_match,
            "task_similarity": None if similarity is None else round(similarity, 3),
            "confidence": conf, "confidence_source": conf_src,
            "evidence": {k: st.get(k, 0) for k in ("w_support", "w_refute", "n_runs",
                                                   "n_explicit")},
            "contested_with": p.get("contested_with") or [],
            "limitations": (p.get("limitations") or [])[:2]}


def _classify(p: dict, st: dict, rel: str, sl: str, mine: dict, profiles: dict,
              canon) -> tuple[str, float, list, float | None, bool]:
    """→ (档位, 档权重, 理由, 任务相似度, 是否有通用性证据)。"""
    why: list[str] = []
    if sl == "metric_family":
        why.append(f"口径放宽:{p['scope'].get('primary_metric')} 与本任务指标同族")
    if sl == "agnostic":
        why.append("口径无关(评估/工程)")
        return "L4", 0.5 if rel in ("same_task", "same_base") else 0.4, why, None, False
    if rel == "same_task":
        return ("L0", 1.0, why, None, False) if sl == "exact" else \
            ("L2", 0.6, why, None, False)
    if rel == "same_base":
        why.insert(0, "同数据底座·跨目标")
        return ("L1", 0.8, why, None, False) if sl == "exact" else \
            ("L2", 0.55, why, None, False)
    if rel == "unknown":
        why.insert(0, "历史原则(无任务指纹)")
        return "L3", 0.4, why, None, True
    sim, dims, _a = _best_similarity(mine, p, profiles, canon)
    if dims:
        why.insert(0, TP.describe(dims))
    # "近似任务"必须由真实列集的重合支撑:同目标 + 列重合 ≥0.7,任务类型也要一致。
    # objects 推出的列族只是弱证据,不足以上升到近似的档位。
    if sl == "exact" and sim >= NEAR_SIM and dims.get("target") == 1.0 and \
            dims.get("columns", 0.0) >= NEAR_COLUMNS:
        return "L1", 0.85 * sim, why, sim, False
    # 跨任务且口径只是同族:再打 0.8 折(两次放宽叠加)
    relax = 0.8 if sl == "metric_family" else 1.0
    if sim >= RELATED_SIM:
        return "L2", 0.6 * (0.5 + 0.5 * sim) * relax, why, sim, False
    # 低相似度的跨底座原则:要么有迁移战绩 / 多底座支持(通用性证据),要么冷启动兜底
    transfer = float(st.get("transfer", 0.5))
    general = transfer > 0.5 or len(_principle_bases(p, canon)) >= 2
    why.append("跨业务" + ("·有迁移证据" if general else "·冷启动兜底"))
    return "L3", 0.3 * transfer * (0.5 + sim) * relax, why, sim, general


def search(view: dict, *, topics=(), scope: dict | None = None, atid: str | None = None,
           hypothesis: str | None = None, k: int = 8, cross_max: int = 2,
           directions=(-1, 0, 1), tiers=TIERS) -> list[dict]:
    canon = canon_fn(view)
    resolver = resolver_of(view)
    q_topics = [resolver.get(str(t).lower(), str(t)) for t in topics or []]
    atid = canon(atid or view.get("atid"))
    view = {**view, "atid": atid}
    scope = scope or view.get("scope")
    stats = view.get("stats") or {}
    profiles = task_profiles(view)
    mine = my_profile(view, profiles)
    h_tok = tokens(hypothesis) if hypothesis else set()
    by_tier: dict[str, list] = defaultdict(list)
    for p in view.get("principles") or []:
        if p["direction"] not in directions:
            continue
        sl = scope_level(scope, p["scope"], p.get("category"))
        if sl is None:
            continue
        # 原则侧主题同样过 resolver:词表把旧词并入新词(topics_aliased)之后,历史原则
        # 仍写着旧 code,只归一查询侧会让它们从检索里整体消失。
        p_topics = [resolver.get(str(t).lower(), t) for t in p.get("topics") or []]
        r = topic_relevance(q_topics, p_topics)
        txt = overlap(h_tok, tokens(p["claim"])) if h_tok else 0.0
        if r <= 0:
            if txt < 0.3:
                continue
            r = 0.3 * txt
        rel = base_relation(atid, p, canon)
        st = stats.get(p["principle_id"]) or {}
        tier, w, why, sim, general = _classify(p, st, rel, sl, mine, profiles, canon)
        if tier not in tiers:
            continue
        why = why + ([f"主题{'精确' if r == 1.0 else '相关'}"] if r >= 0.6 else ["假设文本相近"])
        e_w = 0.5 + 0.5 * float(st.get("posterior", 0.5))
        score = r * w * e_w + 0.1 * txt
        by_tier[tier].append((score, rel, p, st, sl, why, sim, general))

    # L0 / L1 是"精确"档:合在一起按分数排(同底座精确主题可以排在同任务父主题前);
    # 放宽档按档位依次补位
    strict = sorted([(x, t) for t in ("L0", "L1") for x in by_tier[t]], key=lambda y: -y[0][0])
    picked = [(t, x) for x, t in strict[:k]]
    n_upper = len(picked)
    n_cross = 0
    for tier in ("L2", "L3", "L4"):
        if len(picked) >= k:
            break
        for x in sorted(by_tier[tier], key=lambda x: -x[0]):
            if len(picked) >= k:
                break
            score, rel, p, st, sl, why, sim, general = x
            if score < MIN_RELAXED_SCORE:
                continue
            if tier == "L3" and rel == "cross_base" and not general and n_upper:
                continue          # 非冷启动时,无通用性证据的跨业务原则不出
            if rel == "cross_base" and tier in ("L2", "L3") and n_upper:
                if n_cross >= cross_max:
                    continue
                n_cross += 1
            picked.append((tier, x))
        n_upper = len(picked)
    return [_card(p, st, rel, score, tier=tier, why=why, scope_match=sl, similarity=sim,
                  show_objects=tier in ("L0", "L1") and sl != "agnostic")
            for tier, (score, rel, p, st, sl, why, sim, _g) in picked]


def direction_table(view: dict, *, top: int = 5) -> dict:
    """同任务 trial 结果按方向(主题 + 首个对象)聚合:有效方向按增益降序,
    无效方向(rejected 或增益 CI 上界 < 0)按读数升序。替代旧 experiences.yaml。"""
    resolver = resolver_of(view)
    groups: dict[tuple, list] = defaultdict(list)
    for t in view.get("trial_outcomes") or []:
        tp, objs = infer_trial_topics(t, resolver)
        key = (tuple(tp) or (t.get("action") or "other",), objs[0] if objs else "")
        groups[key].append((t, tp, objs))
    eff, bad = [], []
    for (tp, obj), rows in groups.items():
        def row_view(t):
            d, lo, hi = _gain_ci(t)
            return {"topics": list(tp), "object": obj, "trial_id": t["trial_id"],
                    "run_id": t["run_id"], "verdict": t.get("verdict"), "gain": d,
                    "ci": [lo, hi], "hypothesis": str(t.get("hypothesis") or "")[:160],
                    "failure_reason": str(t.get("failure_reason") or "")[:160] or None,
                    "n_trials": len(rows)}
        good = [t for t, _, _ in rows if t.get("verdict") in ("accepted", "confirmed")
                and (_gain_ci(t)[0] or 0) > 0]
        if good:
            eff.append(row_view(max(good, key=lambda t: _gain_ci(t)[0] or 0)))
        failed = [t for t, _, _ in rows if t.get("verdict") == "rejected"
                  or ((_gain_ci(t)[2] is not None) and _gain_ci(t)[2] < 0)]
        if failed and not good:
            bad.append(row_view(min(failed, key=lambda t: _gain_ci(t)[0] or 0)))
    eff.sort(key=lambda r: -(r["gain"] or 0))
    bad.sort(key=lambda r: (r["gain"] or 0))
    return {"effective": eff[:top], "failed": bad[:top],
            "n_effective": len(eff), "n_failed": len(bad)}


def failed_for_topics(view: dict, topics) -> list[dict]:
    """与给定主题相关的已失败方向(检索时与原则一起返回,避免重复尝试)。"""
    resolver = resolver_of(view)
    q = [resolver.get(str(t).lower(), str(t)) for t in topics or []]
    table = direction_table(view, top=1000)
    return [r for r in table["failed"] if topic_relevance(q, r["topics"]) > 0][:5]


def render_card(c: dict) -> str:
    sign = {1: "建议", 0: "无增益", -1: "避免"}[c["direction"]]
    ev = c["evidence"]
    # 证据算出来的置信度与撰写者自报的置信度不能混为一谈(见 _card)
    if c.get("confidence_source") == "evidence":
        conf = f"置信 {c['confidence']}"
        ev_txt = (f"支持 {ev.get('w_support', 0):.1f} / 反驳 {ev.get('w_refute', 0):.1f} · "
                  f"{ev.get('n_runs', 0)} run")
    else:
        conf = f"置信 {c['confidence']}(撰写者自报,无证据)"
        ev_txt = "尚无证据关联"
    tier = f"{c['tier']} {c['tier_label']}" if c.get("tier") else c["label"]
    head = (f"- [{c['principle_id']}] ({sign} · {tier} · {conf} · "
            f"{ev_txt} · {','.join(c['topics'])})")
    lines = [head, f"  {c['claim']}"]
    if c.get("why") and c.get("tier") != "L0":
        lines.append(f"  匹配:{';'.join(c['why'])}"
                     + (" —— 放宽档结论,先小步验证再采用" if c["tier"] in ("L2", "L3") else ""))
    if c.get("objects"):
        lines.append(f"  对象:{', '.join(c['objects'][:8])}")
    if c.get("contested_with"):
        lines.append(f"  ⚠ 与 {', '.join(c['contested_with'])} 结论相反(并存)")
    return "\n".join(lines)


def trial_scope_of(atid: str | None, primary_metric: str | None) -> dict | None:
    return trial_scope({"atid": atid, "primary_metric": primary_metric})
