"""v2.18.0 M2:快照合并 —— 合并上下文、决议校验、落库、撤销、更新日志(dev 文档 §5 §6)。

流程(Q15):
1. :func:`merge_context` 申请合并租约(同一命名空间同一时刻只一个合并),返回快照
   中的候选原则(discoveries)、库内相关原则、主题词表、卷宗与规则;
2. 合并模型(收尾 subagent,Q27)按 ``references/experience-library-merge.md``
   写合并决议;
3. :func:`merge_apply` 程序校验(§5.3 / §5.5)后在一个提交里写入原则新版本、治理
   事件、主题词、卷宗、更新日志,快照置 ``merged`` 并释放租约。

撤销(Q16)::func:`revert_merge` 追加一条 ``revert`` 治理事件指向该次合并的提交;
所有视图忽略被撤销提交写入的原则版本、主题词、卷宗与治理事件。数据不删除。

函数都以 :class:`LibraryStore` 为第一个参数;store 不反向依赖本模块。
"""
from __future__ import annotations

import difflib
import json
import re
import secrets
import time
from collections import defaultdict

from sqlalchemy import and_, delete, func, insert, select

from . import domains as DOM
from . import schema as S
from .artifacts import canonical_json, sha256_hex
from .snapshot import parse_atid
from .store import LibraryValidationError, _latest, _rows, clean_json

DECISION_SCHEMA = "modelevo.merge-decision/1"
CONTEXT_SCHEMA = "modelevo.merge-context/1"
PRINCIPLE_SCHEMA_VERSION = 2
#: 主题词顶层 → 原则 category(category 由第一个主题推出,LLM 不必另写)
TOPIC_ROOTS = {"feature": "feature_engineering", "model": "architecture",
               "hpo": "hpo", "eval": "evaluation", "engineering": "engineering"}
ACTIONS = ("add", "merge_into", "split_by_scope", "supersede", "contest",
           "drop_duplicate", "drop")
DOSSIER_SCOPES = ("task", "data_base")
DOSSIER_SECTIONS = ("ceiling", "effective", "falsified", "open_questions", "data_notes")
#: 卷宗总长上限(字符;约 1.5k token,Q30)
DOSSIER_MAX_CHARS = 3000
MERGE_LEASE_TTL = 1800
#: 需要合并的快照种类(证据 / 配方快照没有候选原则,不进合并队列)
MERGEABLE_KINDS = ("run", "legacy_principles")
MERGEABLE_STATES = ("submitted", "merge_failed", "reverted")
_CODE_RE = re.compile(r"^[a-z]+(\.[a-z0-9_]+){0,2}$")
_MODEL_TYPES = {"binary": "binary_classification", "binary_classification":
                "binary_classification", "classification": "binary_classification",
                "reg": "regression", "regression": "regression",
                "uplift": "uplift", "multiclass": "multiclass"}
_STOP = {"的", "了", "在", "是", "和", "对", "与", "the", "a", "an", "of", "for",
         "to", "and", "in", "on", "is"}


class LeaseBusy(LibraryValidationError):
    """合并租约被占用。"""


# --------------------------------------------------------------------------
# 文本相近度(相关原则召回、主题词去重)
# --------------------------------------------------------------------------

def tokens(text: str) -> set[str]:
    text = str(text or "").lower()
    out = {w for w in re.findall(r"[a-z][a-z0-9_]+", text) if w not in _STOP}
    for seg in re.findall(r"[一-鿿]+", text):
        out.update(seg[i:i + 2] for i in range(len(seg) - 1))
    return out - _STOP


def overlap(a: set, b: set) -> float:
    """词面重合度(以较短者归一)。短结论会因此被放大,合并召回另有 :func:`lexical_score`。"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def lexical_score(a: set, b: set) -> float:
    """对称的词面相似度(Jaccard ×2/(1+J) 形式的 Dice),用于合并召回。

    :func:`overlap` 用 min 长度归一,一句 5 个 token 的短结论与任意长文都能拿到
    0.6 —— 合并时会把不相关的原则召进来,而真正同主题的两条中文结论只有 0.23。
    """
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


#: 合并召回的词面兜底阈值
RECALL_LEXICAL_MIN = 0.2


def _discovery_topics(pl: dict, resolver: dict[str, str]) -> list[str]:
    """候选原则自带 / 从 target 推断的主题(已归一)。"""
    out = []
    for t in pl.get("topics") or []:
        code = resolver.get(str(t).lower())
        if code and code not in out:
            out.append(code)
    return out


def _discovery_scope(pl: dict) -> dict | None:
    """候选原则的口径:直接写的 scope,或旧字段 condition 里的同名字段。"""
    cond = pl.get("condition") if isinstance(pl.get("condition"), dict) else {}
    merged = {**cond, **(pl.get("scope") or {})}
    if not merged.get("model_type") and not merged.get("primary_metric"):
        return None
    return merged


def _scope_compatible(a: dict | None, b: dict | None) -> bool:
    """两边的口径在**共同已知的字段**上一致。

    不能直接用 :func:`evidence.scope_matches`(要求三个字段全等):合并时候选原则常只
    写了 model_type 与 primary_metric,严格比较会把所有原则排除,召回直接归零。
    这里只在双方都写了的字段上判断,``target_transform`` 缺省按 raw。
    """
    if not a or not b:
        return True
    for key in ("model_type", "primary_metric", "target_transform"):
        va, vb = str(a.get(key) or "").lower(), str(b.get(key) or "").lower()
        if va and vb and "*" not in (va, vb) and va != vb:
            return False
    return True


def _recall_related(pl: dict, current: list[dict], resolver: dict[str, str],
                    ptoks: dict[str, set], k: int) -> list[str]:
    """为一条候选原则召回库内相近原则,供合并模型判断 add / merge_into。

    召回顺序(Q5:合并模型只能看到这里给它的东西):
    1. **口径兼容**是前提 —— 口径不同的原则本来就不允许 merge_into / supersede,
       把它们塞进上下文只会诱使模型做无效合并;
    2. 共享主题的优先(精确 > 父子),主题是这套库的主索引;
    3. 同主题下再按词面相似度排序(对称归一,避免短句被放大);
    4. 没有任何主题命中时,才退回纯词面兜底。
    """
    topics = _discovery_topics(pl, resolver)
    scope = _discovery_scope(pl)
    dt = tokens(" ".join(str(pl.get(x) or "") for x in
                         ("claim", "statement", "target", "context_text")))
    scored: list[tuple[tuple, str]] = []
    for p in current:
        if not _scope_compatible(scope, p.get("scope")):
            continue
        p_topics = [resolver.get(str(t).lower(), t) for t in p.get("topics") or []]
        topic_hit = 0.0
        for q in topics:
            for pt in p_topics:
                if q == pt:
                    topic_hit = max(topic_hit, 2.0)
                elif "." in q and "." in pt and (pt.startswith(q + ".") or q.startswith(pt + ".")):
                    topic_hit = max(topic_hit, 1.0)
        lex = lexical_score(dt, ptoks.get(p["principle_id"], set()))
        if not topic_hit and lex < RECALL_LEXICAL_MIN:
            continue
        scored.append(((topic_hit, lex), p["principle_id"]))
    scored.sort(reverse=True)
    return [pid for _key, pid in scored[:k]]


# --------------------------------------------------------------------------
# 视图
# --------------------------------------------------------------------------

def reverted_commits(store, namespace: str, at: int) -> set[int]:
    with store._read() as conn:
        rows = _rows(conn.execute(select(S.governance_events).where(and_(
            S.governance_events.c.namespace == namespace,
            S.governance_events.c.subject_type == "snapshot",
            S.governance_events.c.action == "revert",
            S.governance_events.c.commit_seq <= at))))
    return {int((r["payload"] or {}).get("commit_seq") or 0) for r in rows}


def _visible(store, table, namespace: str, at: int, reverted: set[int]) -> list[dict]:
    with store._read() as conn:
        rows = _rows(conn.execute(select(table).where(and_(
            table.c.namespace == namespace, table.c.commit_seq <= at))))
    return [r for r in rows if r["commit_seq"] not in reverted]


def topics_current(store, namespace: str, *, at: int | None = None) -> dict[str, dict]:
    """每个 code 的最新版本。顶层词(feature/model/…)隐式存在。"""
    at = store._at(at)
    rev = reverted_commits(store, namespace, at)
    cur = _latest(_visible(store, S.topics, namespace, at, rev), lambda r: r["code"])
    for root in TOPIC_ROOTS:
        cur.setdefault(root, {"code": root, "parent": None, "definition": "顶层",
                              "aliases": [], "status": "active", "alias_of": None,
                              "implicit": True})
    return cur


def _topic_resolver(topics: dict[str, dict]) -> dict[str, str]:
    """任意写法(code / 别名 / 已并入的旧词)→ 规范 code。"""
    m: dict[str, str] = {}
    for code, t in topics.items():
        m[code.lower()] = code
        for a in t.get("aliases") or []:
            m.setdefault(str(a).lower(), code)

    def final(code: str) -> str:
        seen = set()
        while topics.get(code, {}).get("status") == "alias_of" and code not in seen:
            seen.add(code)
            code = topics[code].get("alias_of") or code
        return code

    return {k: final(v) for k, v in m.items()}


def principle_rows(store, namespace: str, *, at: int | None = None) -> list[dict]:
    at = store._at(at)
    return _visible(store, S.principle_versions, namespace, at,
                    reverted_commits(store, namespace, at))


def principles_current(store, namespace: str, *, at: int | None = None,
                       include_inactive: bool = False) -> list[dict]:
    """每条原则的最新可见版本 + 状态(active / superseded / retired)+ 争议对象。"""
    at = store._at(at)
    rev = reverted_commits(store, namespace, at)
    rows = _visible(store, S.principle_versions, namespace, at, rev)
    latest: dict[str, dict] = {}
    first: dict[str, int] = {}          # 原则 → 最早可见版本的提交序号
    for r in rows:
        cur = latest.get(r["principle_id"])
        if cur is None or r["version"] > cur["version"]:
            latest[r["principle_id"]] = r
        first[r["principle_id"]] = min(first.get(r["principle_id"], r["commit_seq"]),
                                       r["commit_seq"])
    gov = sorted((g for g in _visible(store, S.governance_events, namespace, at, rev)
                  if g["subject_type"] == "principle"),
                 key=lambda g: (g["commit_seq"], g["id"]))
    status = {pid: "active" for pid in latest}
    detail: dict[str, dict] = defaultdict(dict)
    contested: dict[str, set] = defaultdict(set)
    for g in gov:
        pid, act, pay = g["subject_id"], g["action"], g["payload"] or {}
        if act == "supersede":
            status[pid] = "superseded"
            detail[pid]["superseded_by"] = pay.get("superseded_by")
        elif act == "retire":
            status[pid] = "retired"
        elif act == "exclude":
            # 运维排除:与 retire 一样属于"不再参与检索 / 规划"的状态,区别是它表达
            # "这条结论有问题"而不是"这条结论被取代"。restore 同时解除两者(旧行为)。
            status[pid] = "excluded"
            detail[pid]["excluded_reason"] = g.get("reason")
        elif act == "restore":
            status[pid] = "active"
        elif act == "contest":
            contested[pid].update(pay.get("with") or [])
    seqs = set(first.values()) | {r["commit_seq"] for r in latest.values()}
    when: dict[int, str] = {}
    if seqs:
        with store._read() as conn:
            when = {r[0]: r[1] for r in conn.execute(
                select(S.commits.c.seq, S.commits.c.created_at)
                .where(S.commits.c.seq.in_(seqs)))}
    out = []
    for pid, r in sorted(latest.items()):
        st = status.get(pid, "active")
        if st != "active" and not include_inactive:
            continue
        # 首次提交 = 最早可见版本的提交时间;最近更新 = 最新版本的提交时间(只看内容版本,
        # 争议 / 退役等治理事件不算更新)
        out.append({**r, "status": st, "contested_with": sorted(contested.get(pid, [])),
                    "first_committed_at": when.get(first[pid]),
                    "updated_at": when.get(r["commit_seq"]),
                    **detail.get(pid, {})})
    return out


def dossier_current(store, namespace: str, scope_type: str, scope_id: str, *,
                    at: int | None = None) -> dict | None:
    at = store._at(at)
    rows = [r for r in _visible(store, S.dossiers, namespace, at,
                                reverted_commits(store, namespace, at))
            if r["scope_type"] == scope_type and r["scope_id"] == scope_id]
    return max(rows, key=lambda r: r["version"]) if rows else None


# --------------------------------------------------------------------------
# 租约
# --------------------------------------------------------------------------

def acquire_lease(store, name: str, holder: dict, ttl: float = MERGE_LEASE_TTL) -> dict:
    now = time.time()
    with store._write() as conn:
        row = conn.execute(select(S.leases).where(S.leases.c.name == name)).first()
        if row is not None and row._mapping["expires_at"] > now:
            m = row._mapping
            raise LeaseBusy([f"合并租约 {name} 被占用: {json.dumps(m['holder'], ensure_ascii=False)}"
                             f",{int(m['expires_at'] - now)}s 后过期"])
        conn.execute(delete(S.leases).where(S.leases.c.name == name))
        token = secrets.token_hex(16)
        conn.execute(insert(S.leases).values(
            name=name, holder=clean_json(holder or {}), token=token,
            acquired_at=now, expires_at=now + ttl))
    return {"name": name, "token": token, "expires_at": now + ttl}


def _lease_ok(conn, name: str, token: str) -> bool:
    row = conn.execute(select(S.leases).where(S.leases.c.name == name)).first()
    return row is not None and row._mapping["token"] == token and \
        row._mapping["expires_at"] > time.time()


def release_lease(store, name: str, token: str) -> bool:
    with store._write() as conn:
        res = conn.execute(delete(S.leases).where(and_(
            S.leases.c.name == name, S.leases.c.token == token)))
    return bool(res.rowcount)


def renew_lease(store, name: str, token: str,
                extra_ttl: float = MERGE_LEASE_TTL) -> dict:
    """把已持有的租约延期 ``extra_ttl`` 秒(P1-A:防止大快照合并超时)。

    只有持有正确 token 且租约尚未过期时才能续约,否则抛 ``LibraryValidationError``。
    返回新的 ``{name, token, expires_at}``。
    """
    now = time.time()
    with store._write() as conn:
        row = conn.execute(select(S.leases).where(S.leases.c.name == name)).first()
        if row is None or row._mapping["token"] != token:
            raise LibraryValidationError([f"合并租约 {name} 不存在或 token 不匹配"])
        if row._mapping["expires_at"] <= now:
            raise LibraryValidationError([f"合并租约 {name} 已过期,请重新执行 merge-context"])
        new_expires = row._mapping["expires_at"] + extra_ttl
        conn.execute(
            S.leases.update()
            .where(and_(S.leases.c.name == name, S.leases.c.token == token))
            .values(expires_at=new_expires)
        )
    return {"name": name, "token": token, "expires_at": new_expires}


def _lease_name(namespace: str) -> str:
    return f"merge:{namespace}"


# --------------------------------------------------------------------------
# 合并上下文
# --------------------------------------------------------------------------

def _snapshot_state(store, namespace: str, sid: str) -> tuple[dict | None, str | None]:
    snap = store.get_snapshot(namespace, sid)
    return snap, (snap or {}).get("state")


def merge_pending(store, namespace: str) -> list[dict]:
    out = []
    for s in store.list_snapshots(namespace):
        if s["kind"] in MERGEABLE_KINDS and s["state"] in ("submitted", "merge_failed"):
            out.append({k: s[k] for k in ("snapshot_id", "kind", "run_id", "atid",
                                          "state", "commit_seq")})
    return out


def merge_context(store, namespace: str, snapshot_id: str, *, holder: dict,
                  ttl: float = MERGE_LEASE_TTL, related_k: int = 8) -> dict:
    snap, state = _snapshot_state(store, namespace, snapshot_id)
    if snap is None:
        raise LibraryValidationError([f"快照不存在: {snapshot_id}"])
    if snap["kind"] not in MERGEABLE_KINDS:
        raise LibraryValidationError([f"{snap['kind']} 快照不需要合并"])
    if state not in MERGEABLE_STATES:
        raise LibraryValidationError([f"快照状态为 {state},不能合并"])
    lease = acquire_lease(store, _lease_name(namespace), holder, ttl)
    w = store.watermark()
    topics = topics_current(store, namespace, at=w)
    resolver = _topic_resolver(topics)
    current = principles_current(store, namespace, at=w)
    ptoks = {p["principle_id"]: tokens(p["claim"] + " " + " ".join(p["objects"] or []))
             for p in current}
    related: dict[str, list] = {}
    used: set[str] = set()
    for d in snap["discoveries"]:
        pl = d["payload"] or {}
        ids = _recall_related(pl, current, resolver, ptoks, related_k)
        related[d["local_id"]] = ids
        used.update(ids)
    atid = snap.get("atid")
    base = (parse_atid(atid) or {}).get("data_base") if atid else None
    dossiers = {"task": dossier_current(store, namespace, "task", atid, at=w) if atid else None,
                "data_base": dossier_current(store, namespace, "data_base", base, at=w)
                if base else None}
    keep = ("principle_id", "version", "claim", "category", "direction", "topics",
            "objects", "scope", "condition", "effect", "evidence", "limitations",
            "confidence", "status", "contested_with")
    return {
        "schema": CONTEXT_SCHEMA, "namespace": namespace, "watermark": w,
        "snapshot": {k: snap.get(k) for k in ("snapshot_id", "kind", "run_id", "atid",
                                              "state", "meta")},
        "lease": lease,
        "discoveries": [{"local_id": d["local_id"], "payload": d["payload"],
                         "related": related[d["local_id"]]} for d in snap["discoveries"]],
        "principles": [{k: p.get(k) for k in keep} for p in current
                       if p["principle_id"] in used],
        "n_principles_total": len(current),
        "topics": [{k: t.get(k) for k in ("code", "parent", "definition", "aliases",
                                          "status", "alias_of")}
                   for t in sorted(topics.values(), key=lambda t: t["code"])],
        "dossiers": {k: (v and {"scope_id": v["scope_id"], "version": v["version"],
                                "sections": v["sections"]}) for k, v in dossiers.items()},
        "dossier_scopes": {"task": atid, "data_base": base},
        "rules": {"actions": ACTIONS, "topic_roots": list(TOPIC_ROOTS),
                  "dossier_sections": DOSSIER_SECTIONS,
                  "dossier_max_chars": DOSSIER_MAX_CHARS,
                  "protocol": "references/experience-library-merge.md"},
    }


# --------------------------------------------------------------------------
# 决议校验
# --------------------------------------------------------------------------

def _norm_scope(scope, errors: list, where: str) -> dict:
    scope = dict(scope or {}) if isinstance(scope, dict) else {}
    mt = _MODEL_TYPES.get(str(scope.get("model_type") or "").lower())
    if not mt:
        errors.append(f"{where}: scope.model_type 缺失或未知({scope.get('model_type')!r})")
    pm = str(scope.get("primary_metric") or "").lower()
    if not pm:
        errors.append(f"{where}: scope.primary_metric 必填(与指标无关的评估 / 工程类原则写 \"*\")")
    tt = str(scope.get("target_transform") or "").lower()
    if not tt:
        if mt == "regression" and pm != "*":
            errors.append(f"{where}: 回归原则须写 scope.target_transform(raw/log1p/…)")
        tt = "raw"
    out = {**scope, "model_type": mt, "primary_metric": pm, "target_transform": tt}
    return {k: v for k, v in out.items() if v not in (None, "")}


def _scope_key(scope: dict) -> tuple:
    return (scope.get("model_type"), scope.get("target_transform", "raw"),
            scope.get("primary_metric"))


def _norm_principle(p, resolver: dict[str, str], errors: list, where: str,
                    used_topics: set) -> dict | None:
    if not isinstance(p, dict):
        errors.append(f"{where}: principle 必须是对象")
        return None
    claim = str(p.get("claim") or "").strip()
    if not claim:
        errors.append(f"{where}: claim 不能为空")
    try:
        direction = int(p.get("direction"))
    except (TypeError, ValueError):
        direction = None
    if direction not in (-1, 0, 1):
        errors.append(f"{where}: direction 必须是 -1(避免)/0(无增益)/1(建议)")
    raw_topics = p.get("topics")
    topics: list[str] = []
    if not isinstance(raw_topics, list) or not 1 <= len(raw_topics) <= 3:
        errors.append(f"{where}: topics 须为 1–3 个主题词")
    else:
        for t in raw_topics:
            code = resolver.get(str(t).lower())
            if code is None:
                errors.append(f"{where}: 主题 {t!r} 不在词表中(需要时在 topics_added 里新增)")
            elif code not in topics:
                topics.append(code)
                used_topics.add(code)
    objects = p.get("objects") or []
    if not isinstance(objects, list) or not all(isinstance(o, str) for o in objects):
        errors.append(f"{where}: objects 必须是字符串列表")
        objects = []
    lim = p.get("limitations") or []
    lim = lim if isinstance(lim, list) else [str(lim)]
    conf = p.get("confidence")
    if conf not in (None, "high", "medium", "low"):
        errors.append(f"{where}: confidence 只能是 high/medium/low")
    for k in ("condition", "effect", "evidence"):
        if p.get(k) is not None and not isinstance(p.get(k), dict):
            errors.append(f"{where}: {k} 必须是对象")
    scope = _norm_scope(p.get("scope"), errors, where)
    cond = dict(p.get("condition") or {}) if isinstance(p.get("condition"), dict) else {}
    if cond.get("business_domains"):
        # v2.18.2:业务域按受控词表归一(不认识的写法保留,由 library domains 列出)
        cond["business_domains"] = DOM.normalize(cond["business_domains"])
    root = topics[0].split(".")[0] if topics else None
    return {"principle_id": p.get("principle_id"), "claim": claim,
            "category": TOPIC_ROOTS.get(root, "feature_engineering"),
            "direction": direction, "topics": topics, "objects": objects,
            "scope": scope, "condition": cond,
            "effect": p.get("effect"), "evidence": p.get("evidence"),
            "limitations": [str(x) for x in lim], "confidence": conf}


def _new_principle_id(claim: str, scope: dict, taken: set[str]) -> str:
    base = f"p-{time.strftime('%Y%m')}-" + sha256_hex(
        canonical_json({"claim": claim, "scope": scope}))[:8]
    pid, i = base, 2
    while pid in taken:
        pid, i = f"{base}-{i}", i + 1
    taken.add(pid)
    return pid


def _near(a: str, b: str) -> bool:
    norm = lambda x: x.replace("_", "").rstrip("s")  # noqa: E731
    return norm(a) == norm(b) or difflib.SequenceMatcher(None, a, b).ratio() >= 0.85


def _check_topics_added(added, topics: dict[str, dict], errors: list) -> list[dict]:
    rows = []
    if not isinstance(added, list):
        errors.append("topics_added 必须是列表")
        return rows
    known_codes = set(topics)
    known_alias = {str(a).lower(): c for c, t in topics.items() for a in t.get("aliases") or []}
    new_codes = [t.get("code") for t in added if isinstance(t, dict)]
    for i, t in enumerate(added):
        where = f"topics_added[{i}]"
        if not isinstance(t, dict):
            errors.append(f"{where}: 必须是对象")
            continue
        code = str(t.get("code") or "")
        if not _CODE_RE.match(code):
            errors.append(f"{where}: code {code!r} 须为小写、点分、最多 3 层(如 feature.seq.window_count)")
            continue
        root = code.split(".")[0]
        if root not in TOPIC_ROOTS:
            errors.append(f"{where}: 顶层必须是 {list(TOPIC_ROOTS)} 之一")
            continue
        if code in known_codes:
            errors.append(f"{where}: {code} 已存在")
            continue
        if code.lower() in known_alias:
            errors.append(f"{where}: {code} 已是 {known_alias[code.lower()]} 的别名")
            continue
        parent = code.rsplit(".", 1)[0] if "." in code else None
        if parent and parent not in known_codes and parent not in new_codes:
            errors.append(f"{where}: 父节点 {parent} 不存在(先新增父节点)")
        siblings = [c for c in known_codes if (c.rsplit(".", 1)[0] if "." in c else None) == parent]
        for c in siblings:
            # 只比末段:同父节点的词共享前缀,整串比会把 lag / gap 这类不同概念判成相近
            if _near(c.rsplit(".", 1)[-1], code.rsplit(".", 1)[-1]):
                errors.append(f"{where}: {code} 与已有词 {c} 过于相近,请直接用 {c}")
                break
        if not str(t.get("definition") or "").strip():
            errors.append(f"{where}: definition 必填")
        aliases = t.get("aliases") or []
        if not isinstance(aliases, list):
            errors.append(f"{where}: aliases 必须是列表")
            aliases = []
        for a in aliases:
            if str(a).lower() in known_alias or str(a) in known_codes:
                errors.append(f"{where}: 别名 {a!r} 已被占用")
        rows.append({"code": code, "parent": parent,
                     "definition": str(t.get("definition") or "").strip(),
                     "aliases": [str(a) for a in aliases], "status": "active",
                     "alias_of": None})
    return rows


def _check_overturn(store, namespace: str, target: dict, new: dict, trials,
                    errors: list, where: str) -> None:
    """§5.5-1:推翻 avoid / 无增益结论(方向变为 +1)须有同口径、增益 CI 下界 > 0 的证据。"""
    if not (target["direction"] <= 0 and new.get("direction") == 1):
        return
    # 延迟导入:evidence 依赖本模块,顶部导入会成环
    from .evidence import scope_matches, trial_scope

    ok = False
    tscope = target["scope"]
    for ref in trials or []:
        if not isinstance(ref, dict):
            continue
        rows = [t for t in store.trial_outcomes(namespace, run_id=ref.get("run_id"))
                if t["trial_id"] == ref.get("trial_id")]
        if not rows:
            errors.append(f"{where}: 证据 trial 不存在: {ref}")
            continue
        t = rows[0]
        # 口径必须完整一致:只看 primary_metric 会让 log1p 上的 trial 给 raw 原则的
        # 推翻背书(§5.5 要求"同口径")
        t_scope = trial_scope(t)
        if not scope_matches(tscope, t_scope):
            errors.append(
                f"{where}: 证据 trial {t['trial_id']} 的口径与原则不同"
                f"({t_scope} vs {tscope})")
            continue
        lo, hi = t.get("delta_parent_ci_low"), t.get("delta_parent_ci_high")
        hib = t.get("higher_is_better")
        gain_lo = lo if hib is not False else (-hi if hi is not None else None)
        if gain_lo is not None and gain_lo > 0:
            ok = True
    if not ok:
        errors.append(f"{where}: 推翻 {target['principle_id']}(direction={target['direction']})"
                      "需要同口径且增益 CI 下界 > 0 的证据 trial(evidence_trials);"
                      "证据不足时请改用 contest")


def validate_decision(store, namespace: str, decision: dict, snap: dict) -> tuple[list[str], dict]:
    """返回 (错误, 落库计划)。错误为空时计划可直接写入。"""
    errors: list[str] = []
    if not isinstance(decision, dict) or decision.get("schema") != DECISION_SCHEMA:
        return [f"决议 schema 必须是 {DECISION_SCHEMA}"], {}
    sid = snap["snapshot_id"]
    if decision.get("snapshot_id") != sid:
        errors.append("snapshot_id 与合并上下文不一致")
    w = store.watermark()
    topics = topics_current(store, namespace, at=w)
    current = {p["principle_id"]: p for p in principles_current(
        store, namespace, at=w, include_inactive=True)}
    taken = set(current)
    max_ver: dict[str, int] = {}
    for r in principle_rows(store, namespace, at=w):
        max_ver[r["principle_id"]] = max(max_ver.get(r["principle_id"], 0), r["version"])
    with store._read() as conn:     # 被撤销提交里的版本号也不能复用
        for pid, mv in conn.execute(select(
                S.principle_versions.c.principle_id, func.max(S.principle_versions.c.version))
                .where(S.principle_versions.c.namespace == namespace)
                .group_by(S.principle_versions.c.principle_id)):
            max_ver[pid] = max(max_ver.get(pid, 0), int(mv))
            taken.add(pid)

    plan: dict = {"topics": [], "principle_versions": [], "governance": [],
                  "dossiers": [], "changelog": []}
    added = _check_topics_added(decision.get("topics_added") or [], topics, errors)
    vocab = {**topics, **{t["code"]: t for t in added}}
    for i, a in enumerate(decision.get("topics_aliased") or []):
        code, into = (a or {}).get("code"), (a or {}).get("into")
        if code not in topics or topics[code].get("implicit"):
            errors.append(f"topics_aliased[{i}]: {code!r} 不是可并入的已有词")
        elif into not in vocab or into == code:
            errors.append(f"topics_aliased[{i}]: 目标 {into!r} 不存在")
        else:
            old = topics[code]
            plan["topics"].append({"code": code, "parent": old.get("parent"),
                                   "definition": old.get("definition") or "",
                                   "aliases": old.get("aliases") or [],
                                   "status": "alias_of", "alias_of": into})
            vocab[code] = {**old, "status": "alias_of", "alias_of": into}
            plan["changelog"].append({"entry_type": "topic", "action": "alias",
                                      "subject_type": "topic", "subject_id": code,
                                      "summary": f"主题 {code} 并入 {into}",
                                      "reason": (a or {}).get("reason")})
    resolver = _topic_resolver(vocab)
    used_topics: set[str] = set()

    disc = {d["local_id"]: d for d in snap["discoveries"]}
    seen: set[str] = set()
    actions = decision.get("actions")
    if not isinstance(actions, list):
        errors.append("actions 必须是列表")
        actions = []

    # Q19:原则必须自带可核验的任务指纹,否则同任务权重(检索首档、planner 避免清单、
    # 方向表、profile 对账)对它永久失效 —— 而那条链路正是经验体系的主干。指纹由
    # 程序写,不信任 LLM:LLM 只负责结论,快照的 run.atid 是机器事实。
    snap_atid = snap.get("atid")

    def with_atids(pr: dict) -> dict:
        """把本次提交快照的任务指纹并进 ``evidence.abstract_task_ids``。"""
        if not snap_atid:
            return pr
        ev = dict(pr.get("evidence") or {}) if isinstance(pr.get("evidence"), dict) else {}
        have = [str(a) for a in (ev.get("abstract_task_ids") or []) if a]
        if snap_atid not in have:
            have.append(snap_atid)
        ev["abstract_task_ids"] = have
        return {**pr, "evidence": ev}

    def add_version(pr: dict, pid: str | None, local_id: str, action: str, reason: str):
        if pid is None:
            pid = pr.get("principle_id") if pr.get("principle_id") and \
                pr.get("principle_id") not in taken else None
            pid = pid or _new_principle_id(pr["claim"], pr["scope"], taken)
            taken.add(pid)
        ver = max_ver.get(pid, 0) + 1
        max_ver[pid] = ver
        pr = with_atids(pr)
        plan["principle_versions"].append({
            **{k: pr[k] for k in ("claim", "category", "direction", "topics", "objects",
                                  "scope", "condition", "effect", "evidence",
                                  "limitations", "confidence")},
            "principle_id": pid, "version": ver, "source_snapshot_id": sid,
            "schema_version": PRINCIPLE_SCHEMA_VERSION})
        plan["changelog"].append({
            "entry_type": "merge", "action": action, "subject_type": "principle",
            "subject_id": pid, "summary": f"[{','.join(pr['topics'])}] {pr['claim'][:200]}",
            "reason": reason, "evidence": {"local_id": local_id, "version": ver}})
        return pid

    def target_of(a: dict, where: str, *, need_active: bool = True) -> dict | None:
        t = current.get(a.get("target"))
        if t is None:
            errors.append(f"{where}: target {a.get('target')!r} 不存在")
        elif need_active and t["status"] != "active":
            errors.append(f"{where}: target {t['principle_id']} 状态为 {t['status']}")
            return None
        return t

    for i, a in enumerate(actions):
        where = f"actions[{i}]"
        if not isinstance(a, dict):
            errors.append(f"{where}: 必须是对象")
            continue
        lid, act, reason = a.get("local_id"), a.get("action"), str(a.get("reason") or "").strip()
        where = f"actions[{i}]({lid})"
        if lid not in disc:
            errors.append(f"{where}: local_id 不在快照中")
            continue
        if lid in seen:
            errors.append(f"{where}: 同一条 discovery 只能有一个动作")
            continue
        seen.add(lid)
        if act not in ACTIONS:
            errors.append(f"{where}: action 必须是 {ACTIONS} 之一")
            continue
        if not reason:
            errors.append(f"{where}: reason 必填")
        n_err = len(errors)
        if act == "drop":
            plan["changelog"].append({"entry_type": "merge", "action": "drop",
                                      "subject_type": "discovery", "subject_id": lid,
                                      "summary": str((disc[lid]["payload"] or {}).get("claim")
                                                     or "")[:200], "reason": reason})
            continue
        if act == "drop_duplicate":
            t = target_of(a, where, need_active=False)
            if t is not None:
                plan["changelog"].append({"entry_type": "merge", "action": "drop_duplicate",
                                          "subject_type": "principle",
                                          "subject_id": t["principle_id"],
                                          "summary": f"discovery {lid} 与已有原则重复",
                                          "reason": reason, "evidence": {"local_id": lid}})
            continue
        if act == "split_by_scope":
            prs = a.get("principles")
            if not isinstance(prs, list) or len(prs) < 2:
                errors.append(f"{where}: split_by_scope 需要至少 2 条 principles")
                continue
            normed = [_norm_principle(p, resolver, errors, f"{where}.principles[{j}]",
                                      used_topics) for j, p in enumerate(prs)]
            if len({_scope_key(p["scope"]) for p in normed if p}) < len(normed):
                errors.append(f"{where}: 拆分后的原则口径必须两两不同")
            t = target_of(a, where) if a.get("target") else None
            if len(errors) > n_err:
                continue
            new_ids = [add_version(p, None, lid, "split", reason) for p in normed]
            if t is not None:
                plan["governance"].append({"subject_type": "principle",
                                           "subject_id": t["principle_id"],
                                           "action": "supersede",
                                           "payload": {"superseded_by": new_ids},
                                           "reason": reason})
            continue
        pr = _norm_principle(a.get("principle"), resolver, errors, where, used_topics)
        if act in ("merge_into", "supersede", "contest"):
            t = target_of(a, where)
            if t is not None and pr is not None and act in ("merge_into", "supersede") \
                    and _scope_key(t["scope"]) != _scope_key(pr["scope"]):
                errors.append(f"{where}: 口径不同不得 {act}({_scope_key(t['scope'])} vs "
                              f"{_scope_key(pr['scope'])});请用 add 或 split_by_scope")
            if t is not None and pr is not None and act in ("merge_into", "supersede"):
                _check_overturn(store, namespace, t, pr, a.get("evidence_trials"),
                                errors, where)
        if len(errors) > n_err or pr is None:
            continue
        if act == "add":
            add_version(pr, None, lid, "add", reason)
        elif act == "merge_into":
            add_version(pr, t["principle_id"], lid, "merge_into", reason)
        elif act == "supersede":
            new_id = add_version(pr, None, lid, "supersede", reason)
            plan["governance"].append({"subject_type": "principle",
                                       "subject_id": t["principle_id"], "action": "supersede",
                                       "payload": {"superseded_by": [new_id]},
                                       "reason": reason})
        elif act == "contest":
            new_id = add_version(pr, None, lid, "contest", reason)
            for x, y in ((t["principle_id"], new_id), (new_id, t["principle_id"])):
                plan["governance"].append({"subject_type": "principle", "subject_id": x,
                                           "action": "contest", "payload": {"with": [y]},
                                           "reason": reason})

    missing = sorted(set(disc) - seen)
    if missing:
        errors.append(f"以下 discovery 没有对应动作: {missing[:10]}"
                      + (" …" if len(missing) > 10 else ""))
    for t in added:
        if t["code"] not in used_topics and not any(
                (x.get("parent") == t["code"]) for x in added):
            errors.append(f"新增主题 {t['code']} 没有被任何原则使用")
    plan["topics"] = added + plan["topics"]
    for t in added:
        plan["changelog"].append({"entry_type": "topic", "action": "add",
                                  "subject_type": "topic", "subject_id": t["code"],
                                  "summary": f"新增主题 {t['code']}:{t['definition'][:120]}"})

    dossier_scopes: set = set()
    for i, d in enumerate(decision.get("dossiers") or []):
        where = f"dossiers[{i}]"
        if not isinstance(d, dict) or d.get("scope_type") not in DOSSIER_SCOPES or \
                not d.get("scope_id"):
            errors.append(f"{where}: 需要 scope_type ∈ {DOSSIER_SCOPES} 与 scope_id")
            continue
        if (d["scope_type"], d["scope_id"]) in dossier_scopes:
            errors.append(f"{where}: 同一卷宗在一个决议里只能写一次")
            continue
        dossier_scopes.add((d["scope_type"], d["scope_id"]))
        sec = d.get("sections")
        if not isinstance(sec, dict) or set(sec) - set(DOSSIER_SECTIONS):
            errors.append(f"{where}: sections 只能包含 {DOSSIER_SECTIONS}")
            continue
        sec = {k: str(sec.get(k) or "").strip() for k in DOSSIER_SECTIONS}
        if sum(len(v) for v in sec.values()) > DOSSIER_MAX_CHARS:
            errors.append(f"{where}: 卷宗超过 {DOSSIER_MAX_CHARS} 字符,请压缩")
            continue
        prev = dossier_current(store, namespace, d["scope_type"], d["scope_id"], at=w)
        with store._read() as conn:
            mv = conn.execute(select(func.max(S.dossiers.c.version)).where(and_(
                S.dossiers.c.namespace == namespace,
                S.dossiers.c.scope_type == d["scope_type"],
                S.dossiers.c.scope_id == d["scope_id"]))).scalar() or 0
        plan["dossiers"].append({"scope_type": d["scope_type"], "scope_id": d["scope_id"],
                                 "version": int(mv) + 1, "sections": sec})
        plan["changelog"].append({"entry_type": "dossier", "action": "update" if prev else "add",
                                  "subject_type": d["scope_type"], "subject_id": d["scope_id"],
                                  "summary": f"卷宗 v{int(mv) + 1}"})
    return errors, plan


def init_topics(store, namespace: str, topics: list[dict], *, reason: str,
                actor: dict) -> int:
    """写入主题词表初版(Q28)。只在该命名空间还没有任何(非顶层)主题词时允许;
    之后的增长只能经合并决议的 topics_added(须被原则使用)。格式、父节点、相近度
    校验与 topics_added 相同。"""
    current = topics_current(store, namespace)
    if any(not t.get("implicit") for t in current.values()):
        raise LibraryValidationError(["主题词表已初始化;之后只能经合并决议增减"])
    # 父节点在前:按层级排序后逐个校验,前面的词对后面可见
    ordered = sorted(topics or [], key=lambda t: str((t or {}).get("code") or "").count("."))
    errors: list[str] = []
    rows = _check_topics_added(ordered, current, errors)
    if not rows:
        errors.append("初版词表为空")
    if errors:
        raise LibraryValidationError(errors)
    with store._write() as conn:
        seq = store._new_commit(conn, namespace=namespace, kind="topics", actor=actor,
                                message=f"主题词表初版 {len(rows)} 个词")
        store._insert(conn, S.topics, [{**t, "source_snapshot_id": None} for t in rows],
                      namespace=namespace, seq=seq)
        store._insert(conn, S.changelog_entries, [{
            "snapshot_id": None, "entry_type": "topic", "action": "add",
            "subject_type": "topic", "subject_id": t["code"],
            "summary": f"初版词表:{t['code']}:{t['definition'][:120]}",
            "reason": reason} for t in rows], namespace=namespace, seq=seq)
    return seq


# --------------------------------------------------------------------------
# 落库 / 放弃 / 撤销
# --------------------------------------------------------------------------

def merge_apply(store, namespace: str, decision: dict, *, actor: dict) -> dict:
    sid = (decision or {}).get("snapshot_id")
    snap, state = _snapshot_state(store, namespace, sid) if sid else (None, None)
    if snap is None:
        raise LibraryValidationError([f"快照不存在: {sid}"])
    if state not in MERGEABLE_STATES:
        raise LibraryValidationError([f"快照状态为 {state},不能合并"])
    errors, plan = validate_decision(store, namespace, decision, snap)
    if errors:
        raise LibraryValidationError(errors)
    lease_name, token = _lease_name(namespace), decision.get("lease_token") or ""
    counts = {"principle_versions": len(plan["principle_versions"]),
              "topics": len(plan["topics"]), "governance": len(plan["governance"]),
              "dossiers": len(plan["dossiers"]),
              "dropped": sum(1 for c in plan["changelog"]
                             if c["action"] in ("drop", "drop_duplicate"))}
    if decision.get("dry_run"):
        # 只校验:不写库、不动租约,合并模型可先确认决议无误再正式提交
        return {"snapshot_id": sid, "dry_run": True, "counts": counts}
    with store._write() as conn:
        if not _lease_ok(conn, lease_name, token):
            raise LibraryValidationError(["合并租约无效或已过期:请重新执行 merge-context"])
        seq = store._new_commit(conn, namespace=namespace, kind="merge", actor=actor,
                                ref=sid, message=f"merge {sid[:8]} " + json.dumps(
                                    counts, ensure_ascii=False))
        store._insert(conn, S.topics, [{**t, "source_snapshot_id": sid}
                                       for t in plan["topics"]], namespace=namespace, seq=seq)
        store._insert(conn, S.principle_versions, clean_json(plan["principle_versions"]),
                      namespace=namespace, seq=seq)
        store._insert(conn, S.governance_events, clean_json(plan["governance"]),
                      namespace=namespace, seq=seq)
        store._insert(conn, S.dossiers, [{**d, "source_snapshot_id": sid}
                                         for d in plan["dossiers"]], namespace=namespace, seq=seq)
        store._insert(conn, S.changelog_entries, clean_json(
            [{**c, "snapshot_id": sid} for c in plan["changelog"]]),
            namespace=namespace, seq=seq)
        store._insert(conn, S.snapshot_states, [{"snapshot_id": sid, "state": "merged",
                                                 "detail": counts}],
                      namespace=namespace, seq=seq)
        conn.execute(delete(S.leases).where(S.leases.c.name == lease_name))
    result = {"snapshot_id": sid, "commit_seq": seq, "counts": counts}
    # P0-A: 合并后轻量一致性复核 —— 不阻塞,只在返回结果里附 review_flags。
    # 检查同 atid + 同 scope 内是否出现方向矛盾的两条非 contest 原则;
    # claim 是否含弱表述("可能""建议尝试""或许""试试")。
    result["review_flags"] = _post_merge_review(store, namespace, plan)
    return result


# --------------------------------------------------------------------------
# P0-A: 合并后轻量一致性复核
# --------------------------------------------------------------------------

#: 弱表述正则 —— claim 出现这类词时打 weak_claim flag
_WEAK_CLAIM_RE = re.compile(
    r"(?:可能|建议尝试|或许|试试|也许|有可能|不确定|可以考虑|视情况|效果不明|初步)", re.UNICODE)


def _post_merge_review(store, namespace: str, plan: dict) -> list[dict]:
    """P0-A:合并后对本次写入的原则版本做轻量一致性检查,返回 flag 列表。

    不阻塞 merge_apply;flag 记入返回结果,调用方可写入日志/警报。

    检查项:
    1. claim 含弱表述 —— 原则应可证伪,不应是"建议尝试"式表述。
    2. 同 atid + 同 scope + 同主主题内,本次写入的原则与另一条原则方向相反
       (+1 对 −1/0),且两者未登记 contest(历史或本次决议)。
    """
    flags: list[dict] = []
    new_pvs = plan.get("principle_versions") or []
    if not new_pvs:
        return flags

    # 1. 弱表述检查
    for pv in new_pvs:
        claim = str(pv.get("claim") or "")
        m = _WEAK_CLAIM_RE.search(claim)
        if m:
            flags.append({
                "kind": "weak_claim",
                "principle_id": pv.get("principle_id"),
                "detail": f"claim 含弱表述 '{m.group()}'",
                "claim_preview": claim[:120],
            })

    # 2. 同 atid + 同 scope + 同主主题 方向矛盾检查。
    # 必须按主题分组:同一任务里"特征 A 有效"(+1)与"HPO 加深无效"(0)本来就该并存,
    # 只按 (atid, scope) 分组会让几乎每次合并都报警(正式库 5 组里 4 组),真冲突反被淹没。
    # 已登记为 contest 的对(历史 contested_with 或本次决议的 contest)是有意并存,豁免。
    new_pids = {pv["principle_id"] for pv in new_pvs if pv.get("principle_id")}
    if not new_pids:
        return flags
    try:
        current = principles_current(store, namespace)
    except Exception:  # noqa: BLE001 — 读失败不影响主流程
        return flags
    plan_contest: dict[str, set] = defaultdict(set)
    for g in plan.get("governance") or []:
        if g.get("action") == "contest":
            for other in (g.get("payload") or {}).get("with") or []:
                plan_contest[str(g.get("subject_id"))].add(str(other))
                plan_contest[str(other)].add(str(g.get("subject_id")))

    def _atids(p: dict) -> set[str]:
        ev = p.get("evidence") or {}
        return {str(a) for a in (ev.get("abstract_task_ids") or []) + (ev.get("task_ids") or [])}

    def _contested(x: dict, y: dict) -> bool:
        return (y["principle_id"] in (x.get("contested_with") or []) or
                x["principle_id"] in (y.get("contested_with") or []) or
                y["principle_id"] in plan_contest.get(x["principle_id"], set()))

    groups: dict[tuple, list] = defaultdict(list)
    affected: set[tuple] = set()
    for p in current:
        topic = (p.get("topics") or [""])[0]
        for atid in _atids(p):
            key = (_scope_key(p.get("scope") or {}), atid, topic)
            groups[key].append(p)
            if p["principle_id"] in new_pids:
                affected.add(key)
    for key in sorted(affected, key=str):
        sk, atid, topic = key
        group = groups[key]
        pairs = [(x, y) for x in group if x.get("direction") == 1
                 for y in group if y.get("direction") in (-1, 0)
                 and (x["principle_id"] in new_pids or y["principle_id"] in new_pids)
                 and not _contested(x, y)]
        if not pairs:
            continue
        flags.append({
            "kind": "direction_conflict", "atid": atid, "scope_key": str(sk), "topic": topic,
            "positive_principles": sorted({x["principle_id"] for x, _ in pairs})[:3],
            "negative_principles": sorted({y["principle_id"] for _, y in pairs})[:3],
            "detail": (f"同任务 '{atid}'、同口径、同主题 {topic} 下方向相反且未登记 contest;"
                       "确认是条件不同(应 split_by_scope / 补 condition)还是结论冲突(应 contest)"),
        })
    return flags


def merge_abort(store, namespace: str, snapshot_id: str, *, token: str, reason: str,
                actor: dict) -> int:
    release_lease(store, _lease_name(namespace), token)
    return store.set_snapshot_state(namespace, snapshot_id, "merge_failed", actor=actor,
                                    detail={"reason": reason})


def revert_merge(store, namespace: str, snapshot_id: str, *, reason: str,
                 actor: dict) -> int:
    """撤销该快照最近一次未撤销的合并或整理提交。"""
    with store._read() as conn:
        merges = [r[0] for r in conn.execute(select(S.commits.c.seq).where(and_(
            S.commits.c.namespace == namespace,
            S.commits.c.kind.in_(("merge", "curate")),       # curate:异步整理(consolidation.py)
            S.commits.c.ref == snapshot_id)).order_by(S.commits.c.seq))]
    done = reverted_commits(store, namespace, store.watermark())
    live = [m for m in merges if m not in done]
    if not live:
        raise LibraryValidationError([f"快照 {snapshot_id} 没有可撤销的合并"])
    target = live[-1]
    with store._write() as conn:
        seq = store._new_commit(conn, namespace=namespace, kind="revert", actor=actor,
                                ref=snapshot_id, message=f"revert merge seq={target}")
        store._insert(conn, S.governance_events, [{
            "subject_type": "snapshot", "subject_id": snapshot_id, "action": "revert",
            "payload": {"commit_seq": target}, "reason": reason or ""}],
            namespace=namespace, seq=seq)
        store._insert(conn, S.snapshot_states, [{"snapshot_id": snapshot_id,
                                                 "state": "reverted",
                                                 "detail": {"commit_seq": target,
                                                            "reason": reason}}],
                      namespace=namespace, seq=seq)
        store._insert(conn, S.changelog_entries, [{
            "snapshot_id": snapshot_id, "entry_type": "revert", "action": "revert",
            "subject_type": "snapshot", "subject_id": snapshot_id,
            "summary": f"撤销合并(提交 {target})", "reason": reason or ""}],
            namespace=namespace, seq=seq)
    return seq


# --------------------------------------------------------------------------
# 更新日志
# --------------------------------------------------------------------------

def changelog(store, namespace: str, *, since: int = 0, limit: int = 1000) -> list[dict]:
    with store._read() as conn:
        rows = _rows(conn.execute(
            select(S.changelog_entries, S.commits.c.actor, S.commits.c.created_at,
                   S.commits.c.kind.label("commit_kind"))
            .join(S.commits, S.commits.c.seq == S.changelog_entries.c.commit_seq)
            .where(and_(S.changelog_entries.c.namespace == namespace,
                        S.changelog_entries.c.commit_seq > since))
            .order_by(S.changelog_entries.c.commit_seq, S.changelog_entries.c.id)
            .limit(limit)))
    rev = reverted_commits(store, namespace, store.watermark())
    for r in rows:
        r["reverted"] = r["commit_seq"] in rev
    return rows


_ACTION_ZH = {"add": "新增", "merge_into": "并入", "split": "按口径拆分",
              "supersede": "取代", "contest": "标记争议", "drop": "丢弃",
              "drop_duplicate": "判重", "alias": "别名", "update": "更新",
              "revert": "撤销", "keep": "复核保留", "revise": "修订",
              "revise+absorb": "修订并吸收", "retire": "退役", "generalize": "归纳上位原则"}


def render_changelog(rows: list[dict], namespace: str) -> str:
    by_day: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_day[str(r["created_at"])[:10]][r["commit_seq"]].append(r)
    lines = [f"# 经验库更新日志 · {namespace}", ""]
    for day in sorted(by_day):
        lines += [f"## {day}", ""]
        for seq in sorted(by_day[day]):
            items = by_day[day][seq]
            a = items[0]["actor"] or {}
            head = (f"### 提交 {seq} · {items[0]['commit_kind']} · 快照 "
                    f"{(items[0]['snapshot_id'] or '—')[:8]} · 模型 {a.get('model') or '?'} · "
                    f"{a.get('user') or '?'}@{a.get('host') or '?'}")
            if any(i["reverted"] for i in items):
                head += "(已撤销)"
            lines += [head, ""]
            for i in items:
                act = _ACTION_ZH.get(i["action"], i["action"])
                line = f"- **{act}** `{i['subject_id'] or ''}` {i['summary']}"
                if i.get("reason"):
                    line += f" —— {i['reason']}"
                lines.append(line)
            lines.append("")
    return "\n".join(lines)
