"""v2.18.2 第 2 期:与 run 解耦的经验整理(dev 文档 §23)。

run 收尾的合并只看得到"本次快照的候选 × 召回到的少数相近原则",做不了需要看全局的事:
早已入库的近似原则去重、跨任务共性归纳成上位原则、证据变化后的退役与重评。本模块把
这些工作从 run 里拿出来,按**主题组**(原则的主主题)异步进行:

1. :func:`plan` —— 纯程序,不用 LLM:对每个主题组算触发条件(证据冲突、跨任务可归纳、
   词面近似、争议待重评、展示多却从不被引用、组过大),只返回自上次整理以来有变化的组;
2. :func:`open_group` —— 把组内现行原则(含证据摘要)写成一个 ``kind=curation`` 快照。
   快照按内容寻址:组内原则与证据都没变时得到同一个快照 id,不会被重复整理;
3. :func:`curation_context` —— 申请命名空间合并租约(与 run 合并同一把锁,一次只锁一个组),
   返回给 LLM 的整理上下文;
4. :func:`curation_apply` —— 校验 LLM 的整理决议后在一个提交里落库(``kind=curate``),
   快照置为 ``merged``。撤销用 ``library revert --snapshot``,放弃用 ``merge-abort``。

决议的动作(每条组内原则恰好一个):

- ``keep``:不变(记一次复核);
- ``revise``:写新版本(改写结论、补条件 / 局限);口径不得变,方向由避免 / 无增益改为建议时
  走推翻保护(同口径、增益 CI 下界 > 0 的证据 trial);
- ``merge_into``:并入同组同口径同方向的另一条原则(目标必须 ``revise``,写出合并后的结论),
  本条被 supersede;
- ``retire``:退役。``retire_kind=obsolete``(实现 / 口径已变,不再适用)只需理由;
  ``refuted``(被证据推翻)须证据后验 < 0.4 且证据量 ≥ 1。

``keep`` / ``revise`` 可附 ``contest_with``(登记新的争议对)。另可提 ``generalized``:由组内
≥2 条、来自 ≥2 个任务、同方向的原则归纳出的上位原则,口径须覆盖每条子原则(可用 ``*``);
任务指纹与 ``derived_from`` 由服务端写入,子原则保留。
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict

from sqlalchemy import and_, delete, select

from . import schema as S
from . import curation as CU
from .store import LibraryValidationError, clean_json

DECISION_SCHEMA = "modelevo.curation-decision/1"
CONTEXT_SCHEMA = "modelevo.curation-context/1"
SNAPSHOT_KIND = "curation"
ACTIONS = ("keep", "revise", "merge_into", "retire")
RETIRE_KINDS = ("obsolete", "refuted")

#: 触发阈值(正式库水位 179 上标定,dev 文档 §23.2)
NEAR_DUP_LEXICAL = 0.3          # 同口径同方向两条结论的词面 Dice 相似度
UNUSED_MIN_SHOWN = 5            # Agent 主动检索中展示 ≥ 次数且从未被引用
LARGE_GROUP = 6                 # 组内现行原则数
CHANGE_MIN_LINKS = 3            # 上次整理后组内原则新增证据关联 ≥ 条才算"有变化"
REFUTED_POSTERIOR, REFUTED_MASS = 0.4, 1.0
#: 上下文里每条原则附带的判定性证据(支持 / 反驳)条数
CONTEXT_LINKS = 6


# --------------------------------------------------------------------------
# 分组与触发
# --------------------------------------------------------------------------

def _canon_fn(store, at: int):
    with store._read() as conn:
        amap = store._alias_map(conn, at)
    return lambda a: store._canonical(a, amap) if a else a


def _tasks(p: dict, canon) -> set[str]:
    ev = p.get("evidence") if isinstance(p.get("evidence"), dict) else {}
    return {canon(a) for a in ev.get("abstract_task_ids") or [] if a}


def _groups(store, namespace: str, at: int) -> tuple[dict[str, list], dict]:
    resolver = CU._topic_resolver(CU.topics_current(store, namespace, at=at))
    groups: dict[str, list] = defaultdict(list)
    for p in CU.principles_current(store, namespace, at=at):
        if p.get("topics"):
            t = p["topics"][0]
            groups[resolver.get(str(t).lower(), t)].append(p)
    return groups, resolver


def _last_curated(store, namespace: str) -> dict[str, int]:
    """主题组 → 最近一次未被撤销的整理提交序号。"""
    at = store.watermark()
    rev = CU.reverted_commits(store, namespace, at)
    with store._read() as conn:
        rows = conn.execute(select(S.commits.c.seq, S.snapshots.c.meta)
                            .join(S.snapshots, and_(S.snapshots.c.snapshot_id == S.commits.c.ref,
                                                    S.snapshots.c.namespace == namespace))
                            .where(and_(S.commits.c.namespace == namespace,
                                        S.commits.c.kind == "curate"))).all()
    out: dict[str, int] = {}
    for seq, meta in rows:
        if seq in rev:
            continue
        g = ((meta or {}).get("curation") or {}).get("group")
        if g:
            out[g] = max(out.get(g, 0), int(seq))
    return out


def _triggers(group: list[dict], stats: dict, fb: dict, canon) -> list[dict]:
    out: list[dict] = []
    conflict = [p["principle_id"] for p in group
                if (stats.get(p["principle_id"]) or {}).get("confidence") in ("mixed",
                                                                              "contradicted")]
    if conflict:
        out.append({"kind": "evidence_conflict", "principles": conflict})
    # 上位原则的口径须覆盖每条子原则,而 model_type 不能写 "*":只在同方向、同任务类型内
    # 找跨任务共性(正式库实测:不分任务类型时 29 组触发,首组整理模型正确地以"model_type
    # 不同无法归纳"全部保留,触发白白消耗一次 LLM 复核)
    by_dir: dict[tuple, list] = defaultdict(list)
    for p in group:
        by_dir[(p["direction"], (p.get("scope") or {}).get("model_type"))].append(p)
    for (d, mt), ps in by_dir.items():
        tasks = set().union(*(_tasks(p, canon) for p in ps)) if ps else set()
        if len(ps) >= 2 and len(tasks) >= 2:
            out.append({"kind": "generalizable", "direction": d, "model_type": mt,
                        "principles": [p["principle_id"] for p in ps], "n_tasks": len(tasks)})
    toks = {p["principle_id"]: CU.tokens(p["claim"]) for p in group}
    near = []
    for a, b in itertools.combinations(group, 2):
        if a["direction"] != b["direction"] or \
                CU._scope_key(a["scope"]) != CU._scope_key(b["scope"]):
            continue
        s = CU.lexical_score(toks[a["principle_id"]], toks[b["principle_id"]])
        if s >= NEAR_DUP_LEXICAL:
            near.append([a["principle_id"], b["principle_id"], round(s, 3)])
    if near:
        out.append({"kind": "near_duplicate", "pairs": near})
    ids = {p["principle_id"] for p in group}
    contests = sorted({tuple(sorted((p["principle_id"], o))) for p in group
                       for o in p.get("contested_with") or [] if o in ids})
    contests = [list(c) for c in contests
                if all((stats.get(x) or {}).get("w_support", 0)
                       + (stats.get(x) or {}).get("w_refute", 0) >= 1 for x in c)]
    if contests:
        out.append({"kind": "contest_review", "pairs": contests})
    unused = [p["principle_id"] for p in group
              if (fb.get(p["principle_id"]) or {}).get("shown", 0) >= UNUSED_MIN_SHOWN
              and not (fb.get(p["principle_id"]) or {}).get("cited")]
    if unused:
        out.append({"kind": "unused", "principles": unused})
    if len(group) >= LARGE_GROUP:
        out.append({"kind": "large_group", "n": len(group)})
    return out


def _changed_since(store, namespace: str, group: list[dict], since: int) -> dict:
    ids = [p["principle_id"] for p in group]
    with store._read() as conn:
        n_versions = conn.execute(select(S.principle_versions.c.id).where(and_(
            S.principle_versions.c.namespace == namespace,
            S.principle_versions.c.principle_id.in_(ids),
            S.principle_versions.c.commit_seq > since))).all()
        # 只数新快照带来的证据(evidence 提交且 ref=快照 id,即 link_snapshot)。合并 / 整理之后
        # 的 relink(ref 为空)只是把历史 trial 重新关联到新原则上,不是新证据 —— 数进来会让
        # 刚整理完的组立刻再次到期(10-29 首轮:feature.seq.periodicity 提了上位原则后被连续整理两次)
        n_links = conn.execute(select(S.evidence_links.c.id)
                               .join(S.commits, S.commits.c.seq == S.evidence_links.c.commit_seq)
                               .where(and_(
                                   S.evidence_links.c.namespace == namespace,
                                   S.evidence_links.c.principle_id.in_(ids),
                                   S.evidence_links.c.commit_seq > since,
                                   S.commits.c.kind == "evidence",
                                   S.commits.c.ref.isnot(None)))).all()
    return {"new_versions": len(n_versions), "new_links": len(n_links)}


def _feedback_by_principle(store, namespace: str, at: int) -> dict:
    from .feedback import feedback_stats

    return {r["principle_id"]: r for r in
            feedback_stats(store, namespace, at=at, top=100000).get("principles") or []}


def plan(store, namespace: str, *, include_unchanged: bool = False) -> dict:
    """需要整理的主题组(纯程序)。组从未整理过、或上次整理后有新原则版本 / 足够多的新证据,
    且至少有一个触发条件时列出;按触发条件数与组大小排序。"""
    from .evidence import principle_stats

    at = store.watermark()
    groups, _resolver = _groups(store, namespace, at)
    stats = principle_stats(store, namespace, at=at)
    fb = _feedback_by_principle(store, namespace, at)
    canon = _canon_fn(store, at)
    last = _last_curated(store, namespace)
    out = []
    for g, ps in sorted(groups.items()):
        trig = _triggers(ps, stats, fb, canon)
        if not trig:
            continue
        since = last.get(g)
        change = _changed_since(store, namespace, ps, since) if since else None
        due = since is None or change["new_versions"] > 0 or \
            change["new_links"] >= CHANGE_MIN_LINKS
        if not due and not include_unchanged:
            continue
        out.append({"group": g, "n_principles": len(ps), "triggers": trig,
                    "last_curated_seq": since, "changes": change, "due": due})
    out.sort(key=lambda r: (-len(r["triggers"]), -r["n_principles"], r["group"]))
    return {"namespace": namespace, "watermark": at, "n_groups": len(groups),
            "groups": out}


# --------------------------------------------------------------------------
# 打开一个组(快照)
# --------------------------------------------------------------------------

_KEEP = ("principle_id", "version", "claim", "category", "direction", "topics", "objects",
         "scope", "condition", "effect", "evidence", "limitations", "confidence",
         "contested_with")
_STAT_KEYS = ("w_support", "w_refute", "w_neutral", "n_runs", "n_explicit", "posterior",
              "confidence", "transfer")


def build_group_snapshot(store, namespace: str, group: str) -> dict:
    from .evidence import principle_stats

    at = store.watermark()
    groups, _ = _groups(store, namespace, at)
    ps = groups.get(group)
    if not ps:
        raise LibraryValidationError([f"主题组 {group!r} 没有现行原则"])
    stats = principle_stats(store, namespace, at=at)
    fb = _feedback_by_principle(store, namespace, at)
    canon = _canon_fn(store, at)
    disc = []
    for p in ps:
        st = stats.get(p["principle_id"]) or {}
        f = fb.get(p["principle_id"]) or {}
        disc.append({"local_id": p["principle_id"], "payload": {
            "principle": {k: p.get(k) for k in _KEEP},
            "tasks": sorted(_tasks(p, canon)),
            # 四舍五入后入快照:证据没有实质变化时快照 id 不变,不会重复整理
            "stats": {k: (round(st[k], 2) if isinstance(st.get(k), float) else st.get(k))
                      for k in _STAT_KEYS if k in st},
            "feedback": {k: f.get(k) for k in ("shown", "cited", "mean_gain_when_cited")
                         if f.get(k) is not None}}})
    return {"schema": "modelevo.snapshot/1", "kind": SNAPSHOT_KIND,
            "run": {"run_id": f"curation:{group}"},
            "curation": {"group": group, "triggers": _triggers(ps, stats, fb, canon)},
            "discoveries": disc}


def open_group(store, namespace: str, group: str, *, actor: dict) -> dict:
    payload = build_group_snapshot(store, namespace, group)
    rec = store.submit_snapshot(payload, namespace=namespace, actor=actor)
    snap = store.get_snapshot(namespace, rec["snapshot_id"])
    rec["state"] = snap.get("state")
    rec["group"] = group
    return rec


# --------------------------------------------------------------------------
# 上下文
# --------------------------------------------------------------------------

def _curation_snapshot(store, namespace: str, sid: str) -> dict:
    snap = store.get_snapshot(namespace, sid)
    if snap is None:
        raise LibraryValidationError([f"快照不存在: {sid}"])
    if snap["kind"] != SNAPSHOT_KIND:
        raise LibraryValidationError([f"{sid} 不是整理快照(kind={snap['kind']})"])
    if snap.get("state") not in CU.MERGEABLE_STATES:
        raise LibraryValidationError([f"快照状态为 {snap.get('state')},不能整理"
                                      "(内容未变的组不需要再整理)"])
    return snap


def curation_context(store, namespace: str, snapshot_id: str, *, holder: dict,
                     ttl: float = CU.MERGE_LEASE_TTL) -> dict:
    from .evidence import _relation_from_ci, evidence_links

    snap = _curation_snapshot(store, namespace, snapshot_id)
    lease = CU.acquire_lease(store, CU._lease_name(namespace), holder, ttl)
    w = store.watermark()
    ids = [d["local_id"] for d in snap["discoveries"]]
    dirs = {d["local_id"]: int(d["payload"]["principle"]["direction"])
            for d in snap["discoveries"]}
    trials = {(t["run_id"], t["trial_id"]): t for t in store.trial_outcomes(namespace, at=w)}
    ev: dict[str, list] = defaultdict(list)
    for lk in evidence_links(store, namespace, at=w):
        if lk["principle_id"] not in dirs:
            continue
        rel = _relation_from_ci(dirs[lk["principle_id"]], lk.get("ci_low"),
                                lk.get("ci_high")) or lk["relation"]
        if rel not in ("support", "refute"):
            continue
        t = trials.get((lk["run_id"], lk["trial_id"])) or {}
        ev[lk["principle_id"]].append({
            "relation": rel, "binding": lk["binding"], "base_relation": lk["base_relation"],
            "run_id": lk["run_id"], "trial_id": lk["trial_id"], "atid": t.get("atid"),
            "gain": lk.get("gain"), "ci": [lk.get("ci_low"), lk.get("ci_high")],
            "verdict": t.get("verdict"),
            "hypothesis": str(t.get("hypothesis") or "")[:200]})
    topics = CU.topics_current(store, namespace, at=w)
    return {
        "schema": CONTEXT_SCHEMA, "namespace": namespace, "watermark": w,
        "snapshot_id": snapshot_id, "lease": lease,
        "group": (snap.get("meta") or {}).get("curation", {}).get("group"),
        "triggers": (snap.get("meta") or {}).get("curation", {}).get("triggers") or [],
        "principles": [{**d["payload"],
                        "decisive_evidence": sorted(ev.get(d["local_id"], []),
                                                    key=lambda x: -abs(x.get("gain") or 0)
                                                    )[:CONTEXT_LINKS],
                        "n_decisive_evidence": len(ev.get(d["local_id"], []))}
                       for d in snap["discoveries"]],
        "principle_ids": ids,
        "topics": [{k: t.get(k) for k in ("code", "parent", "definition", "status",
                                          "alias_of")}
                   for t in sorted(topics.values(), key=lambda t: t["code"])],
        "rules": {"actions": ACTIONS, "retire_kinds": RETIRE_KINDS,
                  "refuted_requires": {"posterior_below": REFUTED_POSTERIOR,
                                       "evidence_mass_at_least": REFUTED_MASS},
                  "protocol": "references/experience-library-curation.md"},
    }


# --------------------------------------------------------------------------
# 决议校验与落库
# --------------------------------------------------------------------------

def validate_curation(store, namespace: str, decision: dict, snap: dict
                      ) -> tuple[list[str], dict]:
    from .evidence import principle_stats, scope_matches

    errors: list[str] = []
    if not isinstance(decision, dict) or decision.get("schema") != DECISION_SCHEMA:
        return [f"决议 schema 必须是 {DECISION_SCHEMA}"], {}
    sid = snap["snapshot_id"]
    if decision.get("snapshot_id") != sid:
        errors.append("snapshot_id 与整理上下文不一致")
    w = store.watermark()
    canon = _canon_fn(store, w)
    resolver = CU._topic_resolver(CU.topics_current(store, namespace, at=w))
    current = {p["principle_id"]: p for p in CU.principles_current(
        store, namespace, at=w, include_inactive=True)}
    stats = principle_stats(store, namespace, at=w)
    group = {d["local_id"]: d["payload"]["principle"] for d in snap["discoveries"]}
    for pid, old in group.items():
        cur = current.get(pid)
        if cur is None or cur["status"] != "active" or cur["version"] != old["version"]:
            errors.append(f"{pid}: 打开整理后已被修改或下线(现为 "
                          f"{(cur or {}).get('status')} v{(cur or {}).get('version')}),"
                          "请重新 curate-open")
    if errors:
        return errors, {}
    taken = set(current)
    with store._read() as conn:     # 被撤销提交里的原则 id 也不能复用
        taken.update(r[0] for r in conn.execute(select(S.principle_versions.c.principle_id)
                                                .where(S.principle_versions.c.namespace
                                                       == namespace)))
    plan: dict = {"principle_versions": [], "governance": [], "changelog": []}
    used: set[str] = set()

    acts: dict[str, dict] = {}
    for i, a in enumerate(decision.get("actions") or []):
        where = f"actions[{i}]"
        if not isinstance(a, dict):
            errors.append(f"{where}: 必须是对象")
            continue
        pid = a.get("principle_id")
        where = f"actions[{i}]({pid})"
        if pid not in group:
            errors.append(f"{where}: 不是本组原则")
        elif pid in acts:
            errors.append(f"{where}: 每条原则只能有一个动作")
        elif a.get("action") not in ACTIONS:
            errors.append(f"{where}: action 必须是 {ACTIONS} 之一")
        elif not str(a.get("reason") or "").strip():
            errors.append(f"{where}: reason 必填")
        else:
            acts[pid] = a
    missing = sorted(set(group) - set(acts))
    if missing:
        errors.append(f"以下原则没有对应动作: {missing}")

    revised: dict[str, dict] = {}
    for pid, a in acts.items():
        where = f"{pid}"
        old = current[pid]
        act = a["action"]
        if act == "revise":
            pr = CU._norm_principle(a.get("principle"), resolver, errors, where, used)
            if pr is None:
                continue
            if CU._scope_key(pr["scope"]) != CU._scope_key(old["scope"]):
                errors.append(f"{where}: revise 不得改口径({CU._scope_key(old['scope'])} → "
                              f"{CU._scope_key(pr['scope'])});不同口径请提 generalized")
            CU._check_overturn(store, namespace, old, pr, a.get("evidence_trials"), errors,
                               where)
            revised[pid] = pr
        elif act == "merge_into":
            t = a.get("target")
            ta = acts.get(t)
            if t not in group or t == pid:
                errors.append(f"{where}: merge_into 的目标必须是本组另一条原则")
            elif (ta or {}).get("action") != "revise":
                errors.append(f"{where}: 目标 {t} 必须 revise(写出合并后的结论)")
            elif CU._scope_key(current[t]["scope"]) != CU._scope_key(old["scope"]) or \
                    current[t]["direction"] != old["direction"]:
                errors.append(f"{where}: 只能并入同口径、同方向的原则;方向相反请用 contest_with")
        elif act == "retire":
            kind = a.get("retire_kind")
            if kind not in RETIRE_KINDS:
                errors.append(f"{where}: retire_kind 必须是 {RETIRE_KINDS} 之一")
            elif kind == "refuted":
                st = stats.get(pid) or {}
                mass = st.get("w_support", 0) + st.get("w_refute", 0)
                if not (mass >= REFUTED_MASS and st.get("posterior", 1) < REFUTED_POSTERIOR):
                    errors.append(f"{where}: 证据不足以判定被推翻(后验 {st.get('posterior')}、"
                                  f"证据量 {round(mass, 2)});可 keep + contest_with,"
                                  "或确属过时用 retire_kind=obsolete")
        for o in a.get("contest_with") or []:
            if act not in ("keep", "revise"):
                errors.append(f"{where}: 只有 keep / revise 可附 contest_with")
            elif o == pid or (current.get(o) or {}).get("status") != "active":
                errors.append(f"{where}: contest_with {o!r} 不是另一条现行原则")
            elif (acts.get(o) or {}).get("action") in ("retire", "merge_into"):
                errors.append(f"{where}: {o} 在本决议里被下线,不能再登记争议")

    gens = []
    for i, g in enumerate(decision.get("generalized") or []):
        where = f"generalized[{i}]"
        if not isinstance(g, dict):
            errors.append(f"{where}: 必须是对象")
            continue
        n_err = len(errors)
        kids = g.get("derived_from")
        pr = CU._norm_principle(g.get("principle"), resolver, errors, where, used)
        if not str(g.get("reason") or "").strip():
            errors.append(f"{where}: reason 必填")
        if not isinstance(kids, list) or len(set(kids)) < 2 or not set(kids) <= set(group):
            errors.append(f"{where}: derived_from 须为本组 ≥2 条原则")
            continue
        if pr is None:
            continue
        for k in kids:
            if (acts.get(k) or {}).get("action") in ("retire", "merge_into"):
                errors.append(f"{where}: 子原则 {k} 在本决议里被下线,不能作为归纳依据")
            if current[k]["direction"] != pr["direction"]:
                errors.append(f"{where}: 子原则 {k} 方向与上位原则不同")
            if not scope_matches(current[k]["scope"], pr["scope"]):
                errors.append(f"{where}: 上位原则口径 {pr['scope']} 不覆盖子原则 {k} 的口径 "
                              f"{current[k]['scope']}(可用 \"*\")")
        tasks = set().union(*(_tasks(current[k], canon) for k in kids))
        if len(tasks) < 2:
            errors.append(f"{where}: 子原则只来自 {len(tasks)} 个任务;同任务内的重复请用 merge_into")
        if len(errors) == n_err:
            gens.append((pr, sorted(set(kids)), tasks, str(g["reason"])))
    if errors:
        return errors, plan

    def write(pid: str, pr: dict, action: str, reason: str, ver: int, extra_atids=(),
              derived=None):
        ev = dict(pr.get("evidence") or {}) if isinstance(pr.get("evidence"), dict) else {}
        # 任务指纹由服务端维护:保留旧版本与被并入原则的绑定,不信任 LLM 增删
        atids = list(dict.fromkeys([*(ev.get("abstract_task_ids") or []), *extra_atids]))
        ev["abstract_task_ids"] = [a for a in atids if a]
        if derived:
            ev["derived_from"] = derived
        plan["principle_versions"].append({
            **{k: pr[k] for k in ("claim", "category", "direction", "topics", "objects",
                                  "scope", "condition", "effect", "limitations",
                                  "confidence")},
            "evidence": ev, "principle_id": pid, "version": ver, "source_snapshot_id": sid,
            "schema_version": CU.PRINCIPLE_SCHEMA_VERSION})
        plan["changelog"].append({
            "entry_type": "curate", "action": action, "subject_type": "principle",
            "subject_id": pid, "summary": f"[{','.join(pr['topics'])}] {pr['claim'][:200]}",
            "reason": reason, "evidence": {"version": ver,
                                           **({"derived_from": derived} if derived else {})}})

    merged_into: dict[str, list] = defaultdict(list)
    for pid, a in acts.items():
        if a["action"] == "merge_into":
            merged_into[a["target"]].append(pid)
    for pid, a in acts.items():
        act, reason = a["action"], str(a["reason"])
        old = current[pid]
        if act == "keep":
            plan["changelog"].append({"entry_type": "curate", "action": "keep",
                                      "subject_type": "principle", "subject_id": pid,
                                      "summary": "复核:保持不变", "reason": reason})
        elif act == "revise":
            extra = [*(old.get("evidence") or {}).get("abstract_task_ids", []),
                     *(a2 for m in merged_into.get(pid, [])
                       for a2 in (current[m].get("evidence") or {}).get("abstract_task_ids", []))]
            write(pid, revised[pid], "revise" + ("+absorb" if merged_into.get(pid) else ""),
                  reason, int(old["version"]) + 1, extra_atids=extra)
        elif act == "merge_into":
            plan["governance"].append({"subject_type": "principle", "subject_id": pid,
                                       "action": "supersede",
                                       "payload": {"superseded_by": [a["target"]]},
                                       "reason": reason})
            plan["changelog"].append({"entry_type": "curate", "action": "merge_into",
                                      "subject_type": "principle", "subject_id": pid,
                                      "summary": f"并入 {a['target']}", "reason": reason})
        elif act == "retire":
            plan["governance"].append({"subject_type": "principle", "subject_id": pid,
                                       "action": "retire",
                                       "payload": {"retire_kind": a["retire_kind"]},
                                       "reason": reason})
            plan["changelog"].append({"entry_type": "curate", "action": "retire",
                                      "subject_type": "principle", "subject_id": pid,
                                      "summary": f"退役({a['retire_kind']})", "reason": reason})
        for o in a.get("contest_with") or []:
            for x, y in ((pid, o), (o, pid)):
                plan["governance"].append({"subject_type": "principle", "subject_id": x,
                                           "action": "contest", "payload": {"with": [y]},
                                           "reason": reason})
    for pr, kids, tasks, reason in gens:
        pid = CU._new_principle_id(pr["claim"], pr["scope"], taken)
        taken.add(pid)
        write(pid, pr, "generalize", reason, 1, extra_atids=sorted(tasks), derived=kids)
    return errors, plan


def curation_apply(store, namespace: str, decision: dict, *, actor: dict) -> dict:
    sid = (decision or {}).get("snapshot_id")
    if not sid:
        raise LibraryValidationError(["决议缺少 snapshot_id"])
    snap = _curation_snapshot(store, namespace, sid)
    errors, plan = validate_curation(store, namespace, decision, snap)
    if errors:
        raise LibraryValidationError(errors)
    counts = {"principle_versions": len(plan["principle_versions"]),
              "governance": len(plan["governance"]),
              **{k: sum(1 for c in plan["changelog"] if c["action"].startswith(k))
                 for k in ("keep", "revise", "merge_into", "retire", "generalize")}}
    if decision.get("dry_run"):
        return {"snapshot_id": sid, "dry_run": True, "counts": counts}
    lease_name, token = CU._lease_name(namespace), decision.get("lease_token") or ""
    group = (snap.get("meta") or {}).get("curation", {}).get("group")
    with store._write() as conn:
        if not CU._lease_ok(conn, lease_name, token):
            raise LibraryValidationError(["合并租约无效或已过期:请重新执行 curate-context"])
        seq = store._new_commit(conn, namespace=namespace, kind="curate", actor=actor,
                                ref=sid, message=f"curate {group} " + json.dumps(
                                    counts, ensure_ascii=False))
        store._insert(conn, S.principle_versions, clean_json(plan["principle_versions"]),
                      namespace=namespace, seq=seq)
        store._insert(conn, S.governance_events, clean_json(plan["governance"]),
                      namespace=namespace, seq=seq)
        store._insert(conn, S.changelog_entries, clean_json(
            [{**c, "snapshot_id": sid} for c in plan["changelog"]]),
            namespace=namespace, seq=seq)
        store._insert(conn, S.snapshot_states, [{"snapshot_id": sid, "state": "merged",
                                                 "detail": counts}],
                      namespace=namespace, seq=seq)
        conn.execute(delete(S.leases).where(S.leases.c.name == lease_name))
    return {"snapshot_id": sid, "group": group, "commit_seq": seq, "counts": counts,
            "review_flags": CU._post_merge_review(store, namespace, plan)}
