"""v2.18.2:检索回放评测(dev 文档 §22)。

检索打分里有十几个手定常数(档权重、门槛、维度权重、证据口径),改动它们需要一个
能度量"整体变好还是变坏"的基准。本模块把库里的历史 trial 当作查询:

- **查询**:trial 的主题(与证据关联同一套推断)、假设文本、口径、任务;
- **标签**:该 trial 对每条原则的判读 —— 显式引用或被动关联里 ``support`` 的原则是
  "这次给对了的经验"(good),``refute`` 的是"这次会误导的经验"(bad);
- **防泄漏**:打分用的证据统计**去掉被评 trial 所在 run 的全部关联**(留一 run),
  并排除由该 run 自己的快照写入的原则版本。

指标(对有 good 标签的查询取平均):``recall@k``、``ndcg@k``、``mrr``;
对所有带标签的查询:``bad_share@k`` = 前 k 张里有标签的卡片中 bad 的占比(越低越好)。
另有 ``evidence_auc``:对全部 (trial, 原则) 标签对,留一 run 的证据后验区分 good / bad
的 AUC —— 衡量证据汇总口径本身(与是否进入前 k 无关),0.5 = 没有区分力。

两种模式:

- ``same_task``:按 trial 的真实任务检索(衡量 L0/L1 排序);
- ``new_task``:把任务换成一个库里不存在的同业务新任务(只保留画像里的业务域、量级、
  切分),衡量冷启动时 L2/L3 的排序。

局限:原则与标签来自同一批历史 run(历史原则按任务分组导入,无法逐 run 排除),
绝对值偏乐观;用于**同一水位上不同打分口径的相对比较**。
"""
from __future__ import annotations

import math
from collections import defaultdict

from . import evidence as EV
from . import retrieval as RT

MODES = ("same_task", "new_task")
NEW_TASK_ATID_SUFFIX = "replaynew"


def _labels(trial: dict, links: list[dict], directions: dict[str, int]) -> dict[str, str]:
    out: dict[str, str] = {}
    for lk in links:
        d = directions.get(lk["principle_id"])
        rel = lk["relation"]
        if d is not None:
            rel = EV._relation_from_ci(d, lk.get("ci_low"), lk.get("ci_high")) or rel
        if rel == "support":
            out[lk["principle_id"]] = "good"
        elif rel == "refute":
            out.setdefault(lk["principle_id"], "bad")
    return out


def _new_task_atid(atid: str) -> str:
    """同模型类型、同量级的一个库里不存在的任务(目标与列哈希都换掉)。"""
    mt, rest = atid.split("-", 1)
    scale = rest.rsplit("-", 2)[1]
    return f"{mt}-{NEW_TASK_ATID_SUFFIX}-{scale}-00000000"


def _metrics(ranked: list[str], labels: dict[str, str], k: int) -> dict:
    top = ranked[:k]
    good = {p for p, v in labels.items() if v == "good"}
    labelled = [p for p in top if p in labels]
    out = {"bad_share": (sum(labels[p] == "bad" for p in labelled) / len(labelled))
           if labelled else None}
    if good:
        hits = [i for i, p in enumerate(top) if p in good]
        dcg = sum(1 / math.log2(i + 2) for i in hits)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(good), k)))
        out.update({"recall": len(hits) / len(good), "ndcg": dcg / ideal,
                    "mrr": 1 / (hits[0] + 1) if hits else 0.0})
    return out


def replay(view: dict, trials: list[dict], links: list[dict], *,
           run_sources: dict[str, set] | None = None, k: int = 8, mode: str = "same_task",
           aggregate_kw: dict | None = None, max_queries: int | None = None) -> dict:
    """``view``:``run_view(namespace, None)`` 的全库视图(原则、词表、别名、画像);
    ``links``:全部证据关联;``run_sources``:run_id → 该 run 的快照 id 集合(排除自产原则)。"""
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES} 之一")
    canon = RT.canon_fn(view)
    resolver = RT.resolver_of(view)
    directions = {p["principle_id"]: int(p["direction"]) for p in view.get("principles") or []}
    by_trial: dict[tuple, list] = defaultdict(list)
    for lk in links:
        by_trial[(lk["run_id"], lk["trial_id"])].append(lk)
    stats_cache: dict[str, dict] = {}

    def stats_without(run_id: str) -> dict:
        if run_id not in stats_cache:
            stats_cache[run_id] = EV.aggregate([lk for lk in links if lk["run_id"] != run_id],
                                               directions=directions, **(aggregate_kw or {}))
        return stats_cache[run_id]

    profiles = view.get("fingerprint_profiles") or {}
    per_q: list[dict] = []
    n_skipped = defaultdict(int)
    for t in trials:
        atid = canon(t.get("atid")) if t.get("atid") else None
        scope = EV.trial_scope({**t, "atid": atid}) if atid else None
        topics, _objs = EV.infer_trial_topics(t, resolver)
        if not atid or not scope or not topics:
            n_skipped["no_query"] += 1
            continue
        labels = _labels(t, by_trial.get((t["run_id"], t["trial_id"])) or [], directions)
        if not labels:
            n_skipped["no_label"] += 1
            continue
        own = (run_sources or {}).get(t["run_id"]) or set()
        principles = [p for p in view.get("principles") or []
                      if p.get("source_snapshot_id") not in own]
        q_view = {**view, "principles": principles, "stats": stats_without(t["run_id"])}
        if mode == "same_task":
            q_atid, q_view["task_profile"] = atid, {}
        else:
            q_atid = _new_task_atid(atid)
            prof = profiles.get(atid) or {}
            q_view["task_profile"] = {k2: prof[k2] for k2 in
                                      ("business_domains", "sample_scale", "split_type",
                                       "model_type") if prof.get(k2)}
        cards = RT.search(q_view, topics=topics, scope=scope, atid=q_atid,
                          hypothesis=t.get("hypothesis"), k=k)
        m = _metrics([c["principle_id"] for c in cards], labels, k)
        m["tiers"] = [c["tier"] for c in cards]
        m["run_id"], m["trial_id"] = t["run_id"], t["trial_id"]
        per_q.append(m)
        if max_queries and len(per_q) >= max_queries:
            break

    def mean(key):
        xs = [q[key] for q in per_q if q.get(key) is not None]
        return (round(sum(xs) / len(xs), 4), len(xs)) if xs else (None, 0)

    # 证据区分度:所有标签对(不论是否进入前 k)的留一 run 后验
    pairs = []
    for (run_id, _tid), lks in by_trial.items():
        st = stats_without(run_id)
        for pid, lab in _labels({}, lks, directions).items():
            pairs.append(((st.get(pid) or {}).get("posterior", 0.5), lab))
    pos = [s for s, lab in pairs if lab == "good"]
    neg = [s for s, lab in pairs if lab == "bad"]
    ev_auc = (sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))
              if pos and neg else None)

    tier_mix = defaultdict(int)
    for q in per_q:
        for tier in q["tiers"]:
            tier_mix[tier] += 1
    summary = {"mode": mode, "k": k, "n_queries": len(per_q), "skipped": dict(n_skipped),
               "tier_mix": dict(sorted(tier_mix.items())),
               "evidence_auc": None if ev_auc is None else round(ev_auc, 4),
               "n_evidence_pairs": [len(pos), len(neg)]}
    for key in ("recall", "ndcg", "mrr", "bad_share"):
        summary[key], summary[f"n_{key}"] = mean(key)
    by_run = defaultdict(list)
    for q in per_q:
        if q.get("ndcg") is not None:
            by_run[q["run_id"]].append(q["ndcg"])
    # 按 run 等权的 nDCG:trial 多的 run 不主导结果
    summary["ndcg_run_macro"] = round(sum(sum(v) / len(v) for v in by_run.values())
                                      / len(by_run), 4) if by_run else None
    return summary
