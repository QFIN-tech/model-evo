"""v2.18.0 M3:经验使用的过程指标(Q31 验收口径)。

对一个或多个 run 计算:
- trial 依据覆盖:引用原则 / novel / 缺失(热启动与 baseline 不计);
- 每代证据数:经验库里与该 run trial 关联的证据条数(显式 + 被动),按代汇总;
- 重试已失败方向:trial 的方向(主题 + 首个对象)与**更早 run** 中同任务已失败方向
  相同的次数;
- 库级:无主题原则数、同主题同口径下高度相似的原则对数(合并后残留重复)。

读经验库(服务)与 run 目录;只读。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import yaml

from runtime.library.curation import overlap, tokens
from runtime.library.evidence import _gain_ci, infer_trial_topics, scope_matches
from runtime.library.legacy_import import trial_row
from runtime.library.retrieval import resolver_of
from runtime.library.snapshot import run_id_for

SKIP_ACTIONS = ("warm_start",)


def _run_trials(run_dir: Path) -> list[dict]:
    out = []
    for tdir in sorted((run_dir / "trials").glob("*")) if (run_dir / "trials").is_dir() else []:
        if not (tdir / "trial.yaml").is_file():
            continue
        try:
            row = trial_row(tdir)
        except Exception:  # noqa: BLE001
            continue
        if row:
            out.append(row)
    return out


def _direction_key(row: dict, resolver) -> tuple:
    tp, objs = infer_trial_topics(row, resolver)
    return (tuple(tp), objs[0] if objs else "")


def run_metrics(repo, namespace: str, run_dir, *, view: dict | None = None) -> dict:
    run_dir = Path(run_dir).resolve()
    rs = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8")) \
        if (run_dir / "run-state.json").exists() else {}
    atid = rs.get("abstract_task_id")
    rid = run_id_for(run_dir)
    view = view or repo.run_view(namespace, atid)
    resolver = resolver_of(view)
    trials = [t for t in _run_trials(run_dir)
              if t.get("action") not in SKIP_ACTIONS and t.get("generation") not in (None, 0)]
    cited = sum(1 for t in trials if (t.get("basis") or {}).get("refs"))
    prefilled = sum(1 for t in trials if (t.get("basis") or {}).get("source") == "prefilled")
    novel = sum(1 for t in trials if (t.get("basis") or {}).get("novel"))
    legacy_bound = sum(1 for t in trials if not t.get("basis") and t.get("experience_ids"))

    # 更早 run 的已失败方向(同任务)
    snaps = {s["run_id"]: s for s in repo.list_snapshots(namespace)}
    my_time = str(((snaps.get(rid) or {}).get("meta") or {}).get("run", {}).get("created_at")
                  or rs.get("created_at") or "")
    failed_before: set = set()
    for t in view.get("trial_outcomes") or []:
        if t["run_id"] == rid:
            continue
        t_time = str(((snaps.get(t["run_id"]) or {}).get("meta") or {}).get("run", {})
                     .get("created_at") or "")
        if my_time and t_time and t_time >= my_time:
            continue
        _d, _lo, hi = _gain_ci(t)
        if t.get("verdict") == "rejected" or (hi is not None and hi < 0):
            key = _direction_key(t, resolver)
            if key[0]:
                failed_before.add(key)
    retried = [t["trial_id"] for t in trials if _direction_key(t, resolver) in failed_before]

    # 证据:库里与本 run trial 关联的证据条数,按代
    gen_of = {t["trial_id"]: t.get("generation") for t in trials}
    per_gen: dict = defaultdict(lambda: {"explicit": 0, "prefilled": 0, "passive": 0})
    n_links = 0
    for p in view.get("principles") or []:
        try:
            links = repo.evidence(namespace, p["principle_id"])["links"]
        except Exception:  # noqa: BLE001
            continue
        for lk in links:
            if lk["run_id"] == rid and lk["trial_id"] in gen_of:
                per_gen[gen_of[lk["trial_id"]]][lk["binding"]] += 1
                n_links += 1
    return {"run_id": rid, "atid": atid, "trials": len(trials),
            "basis": {"cited": cited, "prefilled_kept": prefilled, "novel": novel, "legacy_experience_ids": legacy_bound,
                      "missing": len(trials) - cited - novel - legacy_bound,
                      "cited_share": round(cited / len(trials), 3) if trials else None},
            "evidence_links": n_links,
            "evidence_per_generation": {str(k): v for k, v in sorted(per_gen.items())},
            "retried_failed_directions": len(retried), "retried_trial_ids": retried[:20],
            "failed_directions_known_before": len(failed_before)}


def library_metrics(view: dict, *, dup_threshold: float = 0.8) -> dict:
    ps = view.get("principles") or []
    no_topic = [p["principle_id"] for p in ps if not p.get("topics")]
    dups = []
    for i, a in enumerate(ps):
        ta = tokens(a["claim"])
        for b in ps[i + 1:]:
            if not set(a["topics"]) & set(b["topics"]) or not scope_matches(a["scope"], b["scope"]):
                continue
            if a.get("direction") != b.get("direction"):
                continue
            if overlap(ta, tokens(b["claim"])) >= dup_threshold:
                dups.append([a["principle_id"], b["principle_id"]])
    return {"principles": len(ps), "without_topics": len(no_topic),
            "without_topics_ids": no_topic[:20], "near_duplicate_pairs": len(dups),
            "near_duplicates": dups[:20]}


def dump_yaml(obj) -> str:
    return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
