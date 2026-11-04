"""v2.18.1:检索反馈落表与反馈指标。

v2.18.0 建了 ``retrieval_logs`` / ``applications`` 两张表,但没有任何写入方:检索遥测
只作为快照 meta 里的一个 JSON 字段存放,无法回答"展示了哪些卡片 → 哪些被引用 →
引用后效果如何"。本模块:

- :func:`retrieval_rows` / :func:`application_rows`:从快照载荷展开两类行(提交快照时
  同一事务写入;日志按内容哈希、依据按 (原则, run, trial) 去重,重新导出快照不会重复计);
- :func:`backfill`:对已入库的快照补写(只补尚不存在的行);
- :func:`feedback_stats`:曝光 → 引用率、引用后的增益、novel 比例、漏召回
  (trial 的依据引用了原则,但本 run 的检索从未展示过它)。
"""
from __future__ import annotations

from collections import defaultdict

STANCE = {"follows": "adopt", "extends": "adopt", "tests": "test", "contradicts": "challenge"}


def _gain(t: dict):
    d = t.get("delta_parent")
    if d is None:
        return None
    try:
        d = float(d)
    except (TypeError, ValueError):
        return None
    return -d if t.get("higher_is_better") is False else d


def _refs(t: dict) -> list[tuple[str, str, str]]:
    """trial → [(principle_id, declared relation, binding)]。"""
    extra = t.get("extra") if isinstance(t.get("extra"), dict) else {}
    basis = t.get("basis") or extra.get("basis") or []
    # v2.18.6:原样沿用 plan 预填的依据记 basis_prefilled,不算 Agent 主动引用
    binding = "basis_prefilled" if isinstance(basis, dict) and \
        basis.get("source") == "prefilled" else "basis"
    if isinstance(basis, dict):
        basis = basis.get("refs") or []
    out, seen = [], set()
    for item in basis:
        if isinstance(item, dict) and item.get("id") and str(item["id"]) not in seen:
            seen.add(str(item["id"]))
            out.append((str(item["id"]), str(item.get("relation") or "follows"), binding))
    for eid in t.get("experience_ids") or []:
        if str(eid) not in seen:
            seen.add(str(eid))
            out.append((str(eid), "follows", "experience_ids"))
    return out


def application_rows(trials: list[dict]) -> list[dict]:
    """trial → 依据行。同一 (run, trial) 在库里可能有多行(历史快照重复导入、
    atid 修复追加),只取最新一行(``id`` 最大);结果再按 (原则, run, trial) 去重
    —— 唯一键不允许同一依据记两次。"""
    latest: dict = {}
    for t in trials:
        key = (t.get("run_id"), t.get("trial_id"))
        cur = latest.get(key)
        if cur is None or (t.get("id") or 0) >= (cur.get("id") or 0):
            latest[key] = t
    rows, seen = [], set()
    for t in latest.values():
        for pid, rel, binding in _refs(t):
            key = (pid, t["run_id"], t["trial_id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "experience_id": pid, "run_id": t["run_id"], "trial_id": t["trial_id"],
                "binding": binding, "relation": rel,
                "result": str(t.get("verdict") or t.get("status") or "unknown"),
                "stance": STANCE.get(rel, "adopt"), "observed_delta": _gain(t),
                "p_positive": t.get("p_positive"), "generation": t.get("generation")})
    return rows


LOG_HASH_KEY = "_h"


def log_hash(row: dict) -> str:
    """一条检索日志的内容指纹(不含 _h 自身)。遥测文件是逐 run 追加的,同一 run
    重新导出快照时会把全部旧条目再带一遍 —— 提交时按它跳过已入库的行。"""
    from .artifacts import canonical_json, sha256_hex

    body = {k: v for k, v in row.items() if k != "extra"}
    body["extra"] = {k: v for k, v in (row.get("extra") or {}).items() if k != LOG_HASH_KEY}
    return sha256_hex(canonical_json(body))[:16]


def retrieval_rows(payload_or_meta: dict) -> list[dict]:
    run = payload_or_meta.get("run") or {}
    logs = ((payload_or_meta.get("retrieval") or {}).get("logs")) or []
    rid = run.get("run_id") or ""
    out = []
    for lg in logs:
        if not isinstance(lg, dict):
            continue
        shown = [str(x) for x in lg.get("shown_ids") or []]
        query = {k: lg.get(k) for k in ("kind", "topics", "hypothesis") if lg.get(k)}
        extra = {k: v for k, v in lg.items()
                 if k not in ("kind", "topics", "hypothesis", "shown_ids", "watermark",
                              "generation")}
        wm = lg.get("watermark")
        row = {"run_id": rid, "generation": lg.get("generation"),
               "watermark": int(wm) if isinstance(wm, (int, float)) else None,
               "query": query or {"kind": "unknown"}, "shown_ids": shown,
               "extra": extra or None}
        row["extra"] = {**(row["extra"] or {}), LOG_HASH_KEY: log_hash(row)}
        out.append(row)
    return out


def _existing_log_hashes(conn, S, namespace: str, run_ids) -> set:
    from sqlalchemy import select

    from .store import _rows

    if not run_ids:
        return set()
    rows = _rows(conn.execute(select(S.retrieval_logs.c.run_id,
                                     S.retrieval_logs.c.extra).where(
        S.retrieval_logs.c.namespace == namespace,
        S.retrieval_logs.c.run_id.in_(sorted(set(run_ids))))))
    return {(r["run_id"], (r["extra"] or {}).get(LOG_HASH_KEY)) for r in rows}


def new_log_rows(conn, S, namespace: str, rows: list[dict]) -> list[dict]:
    """去掉本命名空间同一 run 已入库的日志(按内容哈希)。"""
    have = _existing_log_hashes(conn, S, namespace, [r["run_id"] for r in rows])
    out = []
    for r in rows:
        key = (r["run_id"], (r.get("extra") or {}).get(LOG_HASH_KEY))
        if key in have:
            continue
        have.add(key)
        out.append(r)
    return out


def feedback_rows(payload: dict, trials: list[dict]) -> tuple[list[dict], list[dict]]:
    return retrieval_rows(payload), application_rows(trials)


# --------------------------------------------------------------------------
# 补写与指标(需要 store)
# --------------------------------------------------------------------------

def backfill(store, namespace: str, *, actor: dict) -> dict:
    """对已入库快照补写 retrieval_logs / applications(幂等:按快照、按唯一键去重)。"""
    from sqlalchemy import and_, select

    from . import schema as S
    from .store import _rows, clean_json

    with store._read() as conn:
        snaps = _rows(conn.execute(select(S.snapshots).where(
            S.snapshots.c.namespace == namespace)))
        trials = _rows(conn.execute(select(S.trial_outcomes).where(and_(
            S.trial_outcomes.c.namespace == namespace))))
    raw_logs = [r for s in snaps for r in retrieval_rows(s["meta"] or {})]
    raw_apps = application_rows(trials)
    # 存在性检查与写入放在同一事务里:与并发的 submit_snapshot 交错时不会撞唯一键
    with store._write() as conn:
        logs = new_log_rows(conn, S, namespace, raw_logs)
        have_apps = {tuple(r) for r in conn.execute(select(
            S.applications.c.experience_id, S.applications.c.run_id,
            S.applications.c.trial_id).where(S.applications.c.namespace == namespace))}
        apps, seen = [], set()
        for r in raw_apps:
            key = (r["experience_id"], r["run_id"], r["trial_id"])
            if key in have_apps or key in seen:
                continue
            seen.add(key)
            apps.append(r)
        if not logs and not apps:
            return {"retrieval_logs": 0, "applications": 0, "commit_seq": None}
        seq = store._new_commit(conn, namespace=namespace, kind="feedback", actor=actor,
                                message=f"backfill feedback: {len(logs)} logs, {len(apps)} apps")
        store._insert(conn, S.retrieval_logs, clean_json(logs), namespace=namespace, seq=seq)
        store._insert(conn, S.applications, clean_json(apps), namespace=namespace, seq=seq)
    return {"retrieval_logs": len(logs), "applications": len(apps), "commit_seq": seq}


def feedback_stats(store, namespace: str, *, at: int | None = None, top: int = 20) -> dict:
    from sqlalchemy import and_, select

    from . import schema as S
    from .store import _rows

    at = store._at(at)
    with store._read() as conn:
        logs = _rows(conn.execute(select(S.retrieval_logs).where(and_(
            S.retrieval_logs.c.namespace == namespace,
            S.retrieval_logs.c.commit_seq <= at))))
        apps = _rows(conn.execute(select(S.applications).where(and_(
            S.applications.c.namespace == namespace,
            S.applications.c.commit_seq <= at))))
        trials = _rows(conn.execute(select(
            S.trial_outcomes.c.run_id, S.trial_outcomes.c.trial_id,
            S.trial_outcomes.c.action, S.trial_outcomes.c.extra).where(and_(
                S.trial_outcomes.c.namespace == namespace,
                S.trial_outcomes.c.commit_seq <= at))))
    # 遥测逐 run 追加。planner-prefill 是程序自动预填的依据:算"展示过"(被引用时不是
    # 漏召回),但不算 Agent 主动看到的曝光(不进 n_shown / cite_rate)。
    surfaced_runs: dict[str, set] = defaultdict(set)
    shown = defaultdict(int)
    for lg in logs:
        for pid in lg["shown_ids"] or []:
            surfaced_runs[pid].add(lg["run_id"])
            if (lg.get("query") or {}).get("kind") != "planner-prefill":
                shown[pid] += 1
    cited = defaultdict(list)
    prefilled = defaultdict(int)
    missed = 0
    for a in apps:
        if a["binding"] == "basis_prefilled":
            prefilled[a["experience_id"]] += 1
            continue
        cited[a["experience_id"]].append(a)
        if a["run_id"] not in surfaced_runs.get(a["experience_id"], set()) and \
                a["binding"] == "basis":
            missed += 1
    n_basis = n_novel = n_missing = n_prefilled = 0
    seen_trials: set = set()
    for t in trials:
        if (t["run_id"], t["trial_id"]) in seen_trials:
            continue                      # 同一 trial 出现在多份快照里(重新导出)只算一次
        seen_trials.add((t["run_id"], t["trial_id"]))
        if t.get("action") in ("warm_start", "baseline"):
            continue
        b = ((t.get("extra") or {}).get("basis"))
        if isinstance(b, dict) and b.get("novel"):
            n_novel += 1
        elif isinstance(b, dict) and b.get("source") == "prefilled":
            n_prefilled += 1
        elif b:
            n_basis += 1
        else:
            n_missing += 1
    per = []
    for pid in set(shown) | set(cited) | set(prefilled):
        cs = cited.get(pid, [])
        gains = [a["observed_delta"] for a in cs if a["observed_delta"] is not None]
        per.append({"principle_id": pid, "shown": shown.get(pid, 0), "cited": len(cs),
                    "prefilled_kept": prefilled.get(pid, 0),
                    "cite_rate": round(len(cs) / shown[pid], 3) if shown.get(pid) else None,
                    "mean_gain_when_cited": round(sum(gains) / len(gains), 6) if gains else None,
                    "accepted_when_cited": sum(1 for a in cs
                                               if a["result"] in ("accepted", "confirmed"))})
    per.sort(key=lambda r: (-r["cited"], -r["shown"], r["principle_id"]))
    n_trials = n_basis + n_novel + n_missing + n_prefilled
    total_shown = sum(shown.values())
    n_cited = sum(len(v) for v in cited.values())
    return {"watermark": at, "n_retrievals": len(logs), "n_shown": total_shown,
            "n_trials": len(seen_trials),
            "n_applications": n_cited,
            "n_applications_prefilled": sum(prefilled.values()),
            "cite_rate": round(n_cited / total_shown, 4) if total_shown else None,
            "cited_not_shown": missed,
            "trials": {"with_basis": n_basis, "prefilled_kept": n_prefilled,
                       "novel": n_novel, "missing": n_missing,
                       "novel_rate": round(n_novel / n_trials, 4) if n_trials else None},
            "principles": per[:top]}
