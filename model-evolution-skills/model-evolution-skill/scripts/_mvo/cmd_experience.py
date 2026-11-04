"""v2.18.0 `modelevo experience` 子命令:按假设主题检索经验库(Q17)。

都读本 run 固定水位的经验库视图(``<run>/experience/library-view.json``),不访问
服务(``evidence`` 在服务可达时补充证据明细)。检索会追加一行到
``experience-retrieval-telemetry.jsonl``,随收尾快照进库。
"""
from __future__ import annotations

import json

from .context import _load_config, _load_spec, _run_dir_from, logger
from runtime.logging_utils import now_iso


def _view(args):
    from runtime import library_view

    run_dir = _run_dir_from(args)
    view = library_view.load(run_dir)
    if not view.pinned:
        spec = _load_spec(args)
        atid = None
        try:
            atid = spec.resolve_abstract_task_id(run_dir, persist=False)
            atid = None if "pending" in str(atid) else atid
        except Exception:  # noqa: BLE001
            atid = None
        view = library_view.ensure(run_dir, atid=atid, spec=spec,
                                   config=_load_config(run_dir))
    return run_dir, view


def _telemetry(run_dir, entry: dict) -> None:
    try:
        with open(run_dir / "experience-retrieval-telemetry.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"at": now_iso(), **entry}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def cmd_experience_search(args) -> int:
    from runtime.library.retrieval import render_card

    run_dir, view = _view(args)
    topics = [t.strip() for t in (args.topics or "").split(",") if t.strip()]
    if not topics and not args.hypothesis:
        logger.error("至少给 --topics 或 --hypothesis")
        return 2
    cards = view.search(topics=topics, hypothesis=args.hypothesis, k=args.k)
    failed = view.failed_for(topics) if topics else []
    _telemetry(run_dir, {"kind": "search", "watermark": view.watermark, "topics": topics,
                         "hypothesis": (args.hypothesis or "")[:160],
                         "shown_ids": [c["principle_id"] for c in cards]})
    if args.json:
        print(json.dumps({"watermark": view.watermark, "stale": view.stale, "cards": cards,
                          "failed_directions": failed}, ensure_ascii=False, indent=2,
                         default=str))
        return 0
    print(f"# 经验检索 · 水位 {view.watermark}{'(stale)' if view.stale else ''} · "
          f"主题 {', '.join(topics) or '—'}\n")
    if not cards:
        print("(无匹配原则:可视为新方向,建 trial 时 --basis novel)\n")
    for c in cards:
        print(render_card(c))
    if failed:
        print("\n## 同任务已失败方向(勿重复)")
        for f in failed:
            print(f"- ✗ {'+'.join(f['topics'])} {f.get('object') or ''} 读数 {f.get('gain')} "
                  f"(`{f['trial_id']}`){(' · ' + f['failure_reason'][:80]) if f.get('failure_reason') else ''}")
    print("\n> 依据某条原则建 trial:`new-trial … --basis <id>:<follows|extends|tests|"
          "contradicts>`;不依据任何原则:`--basis novel`")
    return 0


def cmd_experience_get(args) -> int:
    _run_dir, view = _view(args)
    p = view.principle(args.principle_id)
    if p is None:
        logger.error("原则 %s 不在本 run 固定水位(%s)的视图中", args.principle_id, view.watermark)
        return 1
    print(json.dumps({**p, "stats": view.stats.get(args.principle_id) or {}},
                     ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_experience_evidence(args) -> int:
    from runtime import library_view

    run_dir, view = _view(args)
    out = {"principle_id": args.principle_id,
           "stats": view.stats.get(args.principle_id) or {}}
    try:
        settings = library_view.client_settings()
        rep = library_view.client(settings).evidence(
            view.namespace or settings["namespace"], args.principle_id, at=view.watermark)
        out["links"] = [{k: lk.get(k) for k in ("run_id", "trial_id", "binding", "relation",
                                                 "declared", "base_relation", "gain", "ci_low",
                                                 "ci_high", "weight")}
                        for lk in rep.get("links") or []][-args.limit:]
    except Exception as exc:  # noqa: BLE001  服务不可达时只给汇总
        out["links_error"] = str(exc)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_experience_topics(args) -> int:
    _run_dir, view = _view(args)
    for t in sorted(view.data.get("topics") or [], key=lambda t: t["code"]):
        if t.get("status") == "alias_of":
            continue
        al = f"  [{', '.join(t.get('aliases') or [])}]" if t.get("aliases") else ""
        print(f"{t['code']}  {t.get('definition') or ''}{al}")
    return 0


def add_experience_parser(sub) -> None:
    ep = sub.add_parser("experience", help="v2.18 按假设主题检索经验库(读本 run 固定水位视图)")
    esub = ep.add_subparsers(dest="experience_command", required=True)

    def _common(p):
        p.add_argument("--run")
        p.add_argument("--spec")

    p = esub.add_parser("search", help="按主题(+ 假设文本)检索原则与已失败方向")
    _common(p)
    p.add_argument("--topics", help="主题词或别名,逗号分隔(见 experience topics)")
    p.add_argument("--hypothesis", help="候选假设原文(文本重叠作小加成)")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_experience_search)

    p = esub.add_parser("get", help="一条原则的完整内容与证据汇总")
    _common(p)
    p.add_argument("principle_id")
    p.set_defaults(func=cmd_experience_get)

    p = esub.add_parser("evidence", help="一条原则的证据明细(服务可达时)")
    _common(p)
    p.add_argument("principle_id")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_experience_evidence)

    p = esub.add_parser("topics", help="主题词表")
    _common(p)
    p.set_defaults(func=cmd_experience_topics)
