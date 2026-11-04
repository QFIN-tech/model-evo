"""v2.18.0 `modelevo library` 子命令:经验库服务运维、种子导入、整库迁移。

dev 文档 §8.2 / §10 / Q36 / Q43(手动启停,不自动拉起)。
配置取 skill 级 Config 的 ``library.*``;客户端键可被 MODELEVO_LIBRARY_URI /
MODELEVO_LIBRARY_NS / MODELEVO_LIBRARY_TEAM 覆盖。
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import SKILL_ROOT
from .context import logger
from runtime.config import Config

_SERVER = SKILL_ROOT / "service" / "server.py"


def library_settings(args=None) -> dict:
    """合并 Config 默认、环境变量与命令行参数。"""
    cfg = Config.load()
    svc = dict(cfg.get("library.service", {}) or {})
    out = {
        "uri": os.environ.get("MODELEVO_LIBRARY_URI") or cfg.get("library.uri"),
        "namespace": os.environ.get("MODELEVO_LIBRARY_NS") or cfg.get("library.namespace"),
        "team": os.environ.get("MODELEVO_LIBRARY_TEAM", cfg.get("library.team") or ""),
        "timeout": float(cfg.get("library.timeout_sec", 15) or 15),
        "root": svc.get("root"), "db_url": svc.get("db_url") or "",
        "artifacts_url": svc.get("artifacts_url") or "",
        "host": svc.get("host") or "127.0.0.1", "port": int(svc.get("port") or 18765),
        "backup_keep_days": int(svc.get("backup_keep_days") or 30),
        "python": svc.get("python") or sys.executable,
        "browse": svc.get("browse", True) is not False,
        # v2.18.1:token 认证(服务 --tokens 启用后,客户端每个请求带 Bearer token)
        "token": os.environ.get("MODELEVO_LIBRARY_TOKEN") or cfg.get("library.token") or None,
        "tokens_file": svc.get("tokens") or "",
    }
    for key in ("uri", "namespace", "root", "db_url", "artifacts_url", "host",
                "port", "python", "tokens_file"):
        val = getattr(args, key, None) if args is not None else None
        if val not in (None, ""):
            out[key] = val
    if args is not None and getattr(args, "port", None) and \
            not getattr(args, "uri", None) and \
            not os.environ.get("MODELEVO_LIBRARY_URI"):
        out["uri"] = f"http://{_client_host(out['host'])}:{out['port']}"
    return out


def _client_host(bind: str) -> str:
    return "127.0.0.1" if bind in ("", "0.0.0.0", "::") else bind


def _pidfile(s: dict) -> Path:
    return Path(s["root"]) / "run" / "library.pid"


def _read_pid(s: dict) -> dict | None:
    p = _pidfile(s)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _client(s: dict, *, model: str = "", run: str = ""):
    from runtime.library import HttpRepository
    from runtime.library.identity import current_actor, resolve_model

    if resolve_model(model)[1]:
        logger.warning("--model %r 不是模型名(像 trial / run id),已忽略;请传执行本操作的"
                       "LLM 模型名,如 claude-opus-5-5", model)
    return HttpRepository(s["uri"], timeout=s["timeout"], token=s.get("token"),
                          actor=current_actor(team=s["team"], model=model or None,
                                              run=run or None))


def _server_cmd(s: dict) -> list[str]:
    cmd = [s["python"], str(_SERVER), "--root", str(s["root"]), "--host", s["host"],
           "--port", str(s["port"]), "--backup-keep-days", str(s["backup_keep_days"])]
    if s["db_url"]:
        cmd += ["--db-url", s["db_url"]]
    if s["artifacts_url"]:
        cmd += ["--artifacts-url", s["artifacts_url"]]
    if not s.get("browse", True):
        cmd += ["--no-browse"]
    if s.get("tokens_file"):
        cmd += ["--tokens", str(s["tokens_file"])]
    return cmd


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


# --------------------------------------------------------------------------
# serve / status / stop
# --------------------------------------------------------------------------

def cmd_library_serve(args) -> int:
    s = library_settings(args)
    info = _read_pid(s)
    if info and info.get("host") == socket.gethostname() and _alive(int(info["pid"])):
        _print({"status": "already_running", **info, "uri": s["uri"]})
        return 0
    probe = subprocess.run([s["python"], "-c", "import fastapi, uvicorn"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        logger.error("解释器 %s 缺少 fastapi/uvicorn。安装:%s -m pip install -r %s"
                     ",或在 library.service.python 指定已安装的解释器",
                     s["python"], s["python"], SKILL_ROOT / "service" / "requirements.txt")
        return 2
    cmd = _server_cmd(s)
    if not args.daemon:
        return subprocess.call(cmd, cwd=str(SKILL_ROOT))
    run_dir = Path(s["root"]) / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "library.log"
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, cwd=str(SKILL_ROOT), stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    client = _client(s)
    deadline = time.time() + float(args.wait)
    health, err = None, None
    while time.time() < deadline:
        if proc.poll() is not None:
            err = f"服务进程已退出(returncode={proc.returncode})"
            break
        try:
            health = client.health()
            break
        except Exception as exc:  # noqa: BLE001  启动中
            err = str(exc)
            time.sleep(0.5)
    if health is None:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        _print({"status": "failed", "error": err, "log": str(log_path), "log_tail": tail})
        return 1
    _print({"status": "started", "pid": proc.pid, "uri": s["uri"],
            "log": str(log_path), "health": health})
    return 0


def cmd_library_status(args) -> int:
    s = library_settings(args)
    info = _read_pid(s)
    out = {"uri": s["uri"], "namespace": s["namespace"], "pidfile": str(_pidfile(s)),
           "process": None, "health": None}
    if info:
        same_host = info.get("host") == socket.gethostname()
        out["process"] = {**info, "alive": _alive(int(info["pid"])) if same_host else None,
                          "same_host": same_host}
    try:
        out["health"] = _client(s).health()
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        out["hint"] = f"python {SKILL_ROOT / 'scripts' / 'modelevo.py'} library serve --daemon"
    _print(out)
    return 0 if out["health"] else 1


def cmd_library_stop(args) -> int:
    s = library_settings(args)
    info = _read_pid(s)
    if not info:
        _print({"status": "not_running", "pidfile": str(_pidfile(s))})
        return 0
    if info.get("host") != socket.gethostname():
        _print({"status": "other_host", **info})
        return 1
    pid = int(info["pid"])
    if _alive(pid):
        os.kill(pid, signal.SIGTERM)
        for _ in range(60):
            if not _alive(pid):
                break
            time.sleep(0.25)
    if _alive(pid):
        _print({"status": "still_running", "pid": pid})
        return 1
    _pidfile(s).unlink(missing_ok=True)
    _print({"status": "stopped", "pid": pid})
    return 0


# --------------------------------------------------------------------------
# import-legacy / dump / load
# --------------------------------------------------------------------------

def cmd_library_import_legacy(args) -> int:
    from runtime.library.legacy_import import import_legacy

    s = library_settings(args)
    memory = Path(args.memory or SKILL_ROOT / "memory")
    if not (memory / "global" / "principles.jsonl").exists():
        _print({"error": f"旧库目录不存在或缺 global/principles.jsonl: {memory}",
                "fix_hint": "用 --memory 指向旧 memory/ 目录(如主工作树的 <skill>/memory)"})
        return 1
    sources = [("main", memory / "global" / "principles.jsonl")]
    for spec in args.extra_principles or []:
        label, _, path = spec.partition("=")
        sources.append((label, Path(path)) if path else (Path(label).parent.name, Path(label)))
    repo = None if args.dry_run else _client(s)
    report = import_legacy(
        repo, namespace=s["namespace"], principle_sources=sources,
        top_trials_dir=memory / "top-trials",
        run_roots=[Path(p) for p in (args.runs_root or [])],
        test_namespace=args.test_namespace, dry_run=args.dry_run,
        atid_resolver=_resolve_run_atid if args.resolve_atid else None)
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                default=str), encoding="utf-8")
    _print({k: report[k] for k in ("namespace", "dry_run", "summary",
                                   "principle_sources", "top_trials", "runs")})
    if report["errors"]:
        _print({"errors": report["errors"][:10]})
        return 1
    return 0


def _resolve_run_atid(run_dir: Path) -> str | None:
    """旧 run 的任务指纹:按 runtime 唯一入口解析(不持久化);只接受量级与列哈希都
    合法的结果(splits 兜底得到的漂移指纹靠库里的指纹别名归一)。"""
    import logging

    from runtime.library.snapshot import parse_atid
    from runtime.spec import TaskSpec

    spec_path = Path(run_dir) / "task-spec.yaml"
    if not spec_path.exists():
        return None
    logging.disable(logging.WARNING)
    try:
        atid = TaskSpec.load(spec_path).resolve_abstract_task_id(Path(run_dir), persist=False)
    finally:
        logging.disable(logging.NOTSET)
    return atid if (parse_atid(atid) or {}).get("data_base") else None


def _direct_repo(s: dict, *, force: bool):
    """dump / load 直接打开库文件。Q39:库只由服务进程打开,服务在跑时拒绝。"""
    from runtime.library import ArtifactStore, LibraryStore

    info = _read_pid(s)
    if info and info.get("host") == socket.gethostname() and \
            _alive(int(info["pid"])) and not force:
        raise SystemExit("经验库服务正在运行;先 `library stop`(或 --force)")
    root = Path(s["root"]).resolve()
    db_url = s["db_url"] or f"sqlite:///{root}/library.db"
    art_url = s["artifacts_url"] or f"file://{root}/cas"
    return LibraryStore(db_url), ArtifactStore(art_url)


def cmd_library_dump(args) -> int:
    from runtime.library.dump import dump

    s = library_settings(args)
    store, arts = _direct_repo(s, force=args.force)
    _print(dump(store, arts, args.out))
    return 0


def cmd_library_load(args) -> int:
    from runtime.library.dump import load

    s = library_settings(args)
    store, arts = _direct_repo(s, force=args.force)
    _print(load(store, arts, args.source))
    return 0


# --------------------------------------------------------------------------
# M2:run 快照与合并
# --------------------------------------------------------------------------

def _read_json_file(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_file(path: str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")


def cmd_library_snapshot(args) -> int:
    from .library_snapshot import build_run_snapshot, submit_or_queue

    s = library_settings(args)
    repo = _client(s, model=args.model, run=str(args.run))
    staged: dict = {}
    payload, report = build_run_snapshot(args.run, None if args.dry_run else repo,
                                         model=args.model, upload=not args.dry_run,
                                         stage=staged)
    if args.dry_run:
        _print(report)
        return 0
    result = submit_or_queue(args.run, repo, payload, namespace=s["namespace"],
                             staged=staged)
    _print({**report, **result})
    return 0 if result["status"] == "submitted" else 3


def cmd_library_submit_pending(args) -> int:
    from .library_snapshot import submit_pending

    s = library_settings(args)
    _print(submit_pending(args.run, _client(s)))
    return 0


def cmd_library_merge_pending(args) -> int:
    s = library_settings(args)
    _print(_client(s).merge_pending(s["namespace"]))
    return 0


def cmd_library_merge_context(args) -> int:
    s = library_settings(args)
    ctx = _client(s, model=args.model).merge_context(s["namespace"], args.snapshot)
    _write_json_file(args.out, ctx)
    _print({"context": args.out, "snapshot_id": args.snapshot,
            "discoveries": len(ctx["discoveries"]),
            "related_principles": len(ctx["principles"]),
            "principles_total": ctx["n_principles_total"], "topics": len(ctx["topics"]),
            "lease_expires_in_sec": int(ctx["lease"]["expires_at"] - time.time()),
            "next": "按 references/experience-library-merge.md 写决议,再执行 "
                    "library merge-apply --decision <file>"})
    return 0


def cmd_library_merge_apply(args) -> int:
    from runtime.library import LibraryError

    s = library_settings(args)
    decision = _read_json_file(args.decision)
    if args.dry_run:
        decision["dry_run"] = True
    try:
        receipt = _client(s, model=args.model).merge_apply(s["namespace"], decision)
    except LibraryError as exc:
        _print({"status": "rejected", "error": str(exc), "errors": exc.errors,
                "hint": "逐条修正决议后重新 merge-apply;无法完成时 library merge-abort"})
        return 1
    _print({"status": "valid" if receipt.get("dry_run") else "merged", **receipt})
    return 0


# ---------------------------------------------------------------------------
# 异步整理(v2.18.2 第 2 期,runtime/library/consolidation.py)
# ---------------------------------------------------------------------------

_CURATE_NEXT = ("按 references/experience-library-curation.md 写整理决议,先 "
                "library curate-apply --decision <file> --dry-run 校验,再去掉 --dry-run 提交;"
                "无法完成时 library merge-abort --snapshot <id> --lease-token <token>")


def cmd_library_curate_plan(args) -> int:
    """需要整理的主题组(纯程序判定,不用 LLM)。"""
    s = library_settings(args)
    _print(_client(s).curation_plan(s["namespace"], include_unchanged=args.all))
    return 0


def cmd_library_curate_open(args) -> int:
    s = library_settings(args)
    _print(_client(s, model=args.model).curation_open(s["namespace"], args.group))
    return 0


def _write_curation_context(repo, ns: str, sid: str, out: str) -> dict:
    ctx = repo.curation_context(ns, sid)
    _write_json_file(out, ctx)
    return {"context": out, "snapshot_id": sid, "group": ctx["group"],
            "principles": len(ctx["principles"]), "triggers": [t["kind"] for t in ctx["triggers"]],
            "lease_token": ctx["lease"]["token"],
            "lease_expires_in_sec": int(ctx["lease"]["expires_at"] - time.time()),
            "next": _CURATE_NEXT}


def cmd_library_curate_context(args) -> int:
    s = library_settings(args)
    _print(_write_curation_context(_client(s, model=args.model), s["namespace"],
                                   args.snapshot, args.out))
    return 0


def cmd_library_curate_next(args) -> int:
    """取下一个需要整理的组:打开快照并导出上下文。组内容自上次整理后没有实质变化
    (快照判重且已整理)时跳过;没有可整理的组时 status=idle。"""
    s = library_settings(args)
    repo = _client(s, model=args.model)
    skipped = []
    for g in repo.curation_plan(s["namespace"])["groups"]:
        rec = repo.curation_open(s["namespace"], g["group"])
        if rec.get("state") not in ("submitted", "merge_failed", "reverted"):
            skipped.append(g["group"])
            continue
        _print({"status": "opened", **_write_curation_context(
            repo, s["namespace"], rec["snapshot_id"], args.out), "skipped": skipped})
        return 0
    _print({"status": "idle", "skipped": skipped})
    return 0


def cmd_library_curate_apply(args) -> int:
    from runtime.library import LibraryError

    s = library_settings(args)
    decision = _read_json_file(args.decision)
    if args.dry_run:
        decision["dry_run"] = True
    try:
        receipt = _client(s, model=args.model).curation_apply(s["namespace"], decision)
    except LibraryError as exc:
        _print({"status": "rejected", "error": str(exc), "errors": exc.errors,
                "hint": "逐条修正决议后重新 curate-apply;无法完成时 library merge-abort"})
        return 1
    _print({"status": "valid" if receipt.get("dry_run") else "curated", **receipt})
    return 0


def cmd_library_merge_abort(args) -> int:
    s = library_settings(args)
    seq = _client(s, model=args.model).merge_abort(
        s["namespace"], args.snapshot, lease_token=args.lease_token, reason=args.reason)
    _print({"status": "merge_failed", "commit_seq": seq})
    return 0


def cmd_library_merge_renew(args) -> int:
    """合并耗时较长时续约(租约 30 分钟;续约只对持有者、且未过期时有效)。"""
    s = library_settings(args)
    lease = _client(s, model=args.model).renew_lease(s["namespace"], args.lease_token,
                                                     extra_ttl=args.extra_sec)
    _print({"status": "renewed", "expires_in_sec": int(lease["expires_at"] - time.time())})
    return 0


def cmd_library_feedback(args) -> int:
    """检索反馈:曝光 → 引用率、引用后增益、novel 比例、被引用却未被展示(漏召回)。"""
    s = library_settings(args)
    repo = _client(s)
    out = {}
    if args.backfill:
        out["backfill"] = repo.feedback_backfill(s["namespace"])
    out.update(repo.feedback_stats(s["namespace"], top=args.top))
    _print(out)
    return 0


def cmd_library_replay(args) -> int:
    """检索回放评测:历史 trial 作查询、其证据判读作标签,算 recall/nDCG/MRR/bad_share
    与证据区分度 evidence_auc。``--run-dedup`` / ``--multi-factor`` 换证据汇总口径做对比。"""
    s = library_settings(args)
    repo = _client(s)
    modes = ("same_task", "new_task") if args.mode == "all" else (args.mode,)
    _print({mode: repo.replay(s["namespace"], k=args.k, mode=mode, run_dedup=args.run_dedup,
                              multi_factor=args.multi_factor, max_queries=args.max_queries)
            for mode in modes})
    return 0


def cmd_library_domains(args) -> int:
    """业务域受控词表(runtime/library/business_domains.yaml)及库内使用情况:
    每个词被多少条原则 / 多少个任务画像用到,以及不在词表里的写法(需要加词或加别名)。"""
    from runtime.library import domains as DOM

    s = library_settings(args)
    view = _client(s).run_view(s["namespace"], None)
    use: dict[str, dict] = {}

    def count(raw, kind):
        for v in [raw] if isinstance(raw, str) else raw or []:
            code, known = DOM.resolve(v)
            if not code:
                continue
            u = use.setdefault(code, {"known": known, "principles": 0, "tasks": 0,
                                      "spellings": set()})
            u[kind] += 1
            u["spellings"].add(str(v))

    for p in view.get("principles") or []:
        cond = p.get("condition") if isinstance(p.get("condition"), dict) else {}
        count(cond.get("business_domains"), "principles")
    for prof in (view.get("fingerprint_profiles") or {}).values():
        count((prof or {}).get("business_domains"), "tasks")
    voc = DOM.vocabulary()
    _print({
        "vocabulary": [{"code": c, "label": d["label"], "aliases": d["aliases"],
                        **{k: (use.get(c) or {}).get(k, 0) for k in ("principles", "tasks")},
                        "spellings": sorted((use.get(c) or {}).get("spellings", ()))}
                       for c, d in voc.items()],
        "unknown": [{"value": c, "principles": u["principles"], "tasks": u["tasks"]}
                    for c, u in sorted(use.items()) if not u["known"]],
        "hint": "unknown 里是新业务就在 business_domains.yaml 加词;是已有业务的另一种写法就加进 aliases",
    })
    return 0


def cmd_library_suggest_aliases(args) -> int:
    """按任务画像建议指纹别名(同目标、列重合高);只输出建议,确认后用 library alias。"""
    s = library_settings(args)
    _print(_client(s).suggest_aliases(min_columns=args.min_columns))
    return 0


def cmd_library_backfill_profiles(args) -> int:
    """为没有画像的存量任务回填画像:从快照记录的 run 目录读 task-spec.yaml 与原生列。
    run 目录已不在本机时跳过(报告里列出)。"""
    import yaml

    from runtime.library_view import task_profile_for

    s = library_settings(args)
    repo = _client(s)
    have = set(repo.fingerprint_profiles() or {})
    done, skipped = [], []
    seen: set = set()
    for snap in repo.list_snapshots(s["namespace"]):
        atid, run_dir = snap.get("atid"), snap.get("run_dir")
        if not atid or atid in seen or (atid in have and not args.force):
            continue
        spec_p = Path(run_dir or "") / "task-spec.yaml"
        if not run_dir or not spec_p.is_file():
            skipped.append({"atid": atid, "run_dir": run_dir, "reason": "run 目录或 task-spec 不在"})
            continue
        prof = task_profile_for(run_dir, yaml.safe_load(spec_p.read_text(encoding="utf-8")) or {},
                                atid)
        if not prof:
            skipped.append({"atid": atid, "run_dir": run_dir, "reason": "画像为空"})
            continue
        seen.add(atid)
        seq = None if args.dry_run else repo.put_task_profile(atid, prof)
        done.append({"atid": atid, "run_dir": run_dir, "commit_seq": seq,
                     "n_columns": prof.get("n_columns"),
                     "business_domains": prof.get("business_domains")})
    skipped = [x for x in skipped if x["atid"] not in seen]
    _print({"dry_run": bool(args.dry_run), "profiled": done, "skipped": skipped})
    return 0


def cmd_library_fix_fingerprints(args) -> int:
    """存量指纹修复:按 run 目录重新推出任务指纹。默认只出计划(--apply 才写库)。"""
    s = library_settings(args)
    run_dirs = {}
    for item in args.run_dir or []:
        rid, _, rd = str(item).partition("=")
        if not rd:
            logger.error("--run-dir 须为 <run_id>=<run_dir>: %s", item)
            return 2
        run_dirs[rid] = rd
    out = _client(s, model=args.model).fix_fingerprints(
        s["namespace"], apply=bool(args.apply), run_dirs=run_dirs)
    _print(out)
    if not args.apply:
        logger.info("计划 %d 项;确认无误后加 --apply 执行(可与 --run-dir 一起用)",
                    len(out.get("actions") or []))
    return 0


def cmd_library_alias(args) -> int:
    s = library_settings(args)
    seq = _client(s).add_fingerprint_alias(args.alias, args.canonical, reason=args.reason)
    _print({"status": "aliased", "alias": args.alias, "canonical": args.canonical,
            "commit_seq": seq})
    return 0


def cmd_library_token_hash(args) -> int:
    """生成 token 及其 sha256(写进服务的 --tokens 文件;明文只交给使用者)。"""
    import secrets

    from runtime.library.identity import token_hash

    tok = args.token or secrets.token_urlsafe(24)
    _print({"token": tok, "entry": {"sha256": token_hash(tok), "user": args.user,
                                    "team": args.team or "", "write": not args.read_only}})
    return 0


def cmd_library_revert(args) -> int:
    s = library_settings(args)
    seq = _client(s).revert_merge(s["namespace"], args.snapshot, reason=args.reason)
    _print({"status": "reverted", "commit_seq": seq})
    return 0


def cmd_library_changelog(args) -> int:
    from runtime.library.curation import render_changelog

    s = library_settings(args)
    rows = _client(s).changelog(s["namespace"], since=args.since)
    text = render_changelog(rows, s["namespace"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        _print({"out": args.out, "entries": len(rows)})
    else:
        print(text)
    return 0


def cmd_library_principles(args) -> int:
    s = library_settings(args)
    rows = _client(s).principles(s["namespace"], include_inactive=args.include_inactive)
    if args.json:
        _print(rows)
        return 0
    for r in rows:
        print(f"{r['principle_id']}  v{r['version']}  {r['status']:<10} "
              f"{'+' if r['direction'] > 0 else ('0' if r['direction'] == 0 else '-')}  "
              f"{','.join(r['topics'])}  {r['claim'][:80]}")
    print(f"共 {len(rows)} 条")
    return 0


def cmd_library_topics(args) -> int:
    s = library_settings(args)
    for t in _client(s).topics(s["namespace"]):
        extra = f" → {t['alias_of']}" if t.get("status") == "alias_of" else ""
        al = f"  [{', '.join(t.get('aliases') or [])}]" if t.get("aliases") else ""
        print(f"{t['code']}{extra}  {t.get('definition') or ''}{al}")
    return 0


def cmd_library_topics_init(args) -> int:
    import yaml

    s = library_settings(args)
    data = yaml.safe_load(Path(args.file).read_text(encoding="utf-8")) or {}
    topics = data.get("topics") if isinstance(data, dict) else data
    seq = _client(s, model=args.model).init_topics(s["namespace"], topics or [],
                                                   reason=args.reason)
    _print({"status": "initialized", "commit_seq": seq, "topics": len(topics or [])})
    return 0


def cmd_library_metrics(args) -> int:
    """Q31 过程指标:各 run 的依据覆盖 / 每代证据 / 重试已失败方向 + 库级指标。"""
    from .library_metrics import library_metrics, run_metrics

    s = library_settings(args)
    repo = _client(s)
    out = {"runs": []}
    views: dict = {}
    for rd in args.run or []:
        st = Path(rd) / "run-state.json"
        atid = json.loads(st.read_text(encoding="utf-8")).get("abstract_task_id") \
            if st.exists() else None
        if atid not in views:
            views[atid] = repo.run_view(s["namespace"], atid)
        out["runs"].append(run_metrics(repo, s["namespace"], rd, view=views[atid]))
    any_view = next(iter(views.values()), None) or repo.run_view(s["namespace"], None)
    out["library"] = library_metrics(any_view)
    _print(out)
    return 0


# --------------------------------------------------------------------------
# 查看:browse / recipe export / leaderboard
# --------------------------------------------------------------------------

def cmd_library_browse(args) -> int:
    """把经验库渲染成明文目录(给不能访问库根共享盘的用户;服务端已自动生成 browse/)。"""
    from runtime.library import browse

    s = library_settings(args)
    try:
        info = browse.export_dir(_client(s), s["namespace"], Path(args.out))
    except RuntimeError as exc:
        _print({"error": str(exc)})
        return 1
    _print(info)
    return 0


def _resolve_recipe_id(repo, namespace: str, rid: str) -> str | None:
    if len(rid) == 64:
        return rid
    hits = {r["recipe_id"] for r in repo.recipes(namespace) if r["recipe_id"].startswith(rid)}
    return hits.pop() if len(hits) == 1 else None


def cmd_library_recipe_export(args) -> int:
    """导出一个配方为明文目录;给 --run 时把占位符填成该 run 与本 skill 的路径(可运行副本)。"""
    from runtime.library.recipes import check_python, render

    s = library_settings(args)
    repo = _client(s)
    rid = _resolve_recipe_id(repo, s["namespace"], args.recipe_id)
    if rid is None:
        _print({"error": f"配方 id 不存在或前缀不唯一: {args.recipe_id}"})
        return 1
    files = repo.get_bundle(rid)
    if args.run:
        files = render(files, run_dir=str(Path(args.run).resolve()), skill_root=str(SKILL_ROOT))
        errors = check_python(files)
        if errors:
            _print({"error": "渲染后代码校验失败", "errors": errors})
            return 1
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        _print({"error": f"{out} 非空,拒绝覆盖"})
        return 1
    for name, data in files.items():
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    _print({"recipe_id": rid, "out": str(out), "files": sorted(files),
            "rendered_for_run": args.run or None})
    return 0


def cmd_library_leaderboard(args) -> int:
    """某任务的榜单(同一配方只占一个名次)。"""
    from runtime.library.recipes import RUNTIME_API_VERSION

    s = library_settings(args)
    repo = _client(s)
    view = repo.run_view(s["namespace"], args.atid, runtime_api_version=RUNTIME_API_VERSION)
    rows = view.get("leaderboard") or []
    if args.json:
        _print(rows)
        return 0
    print(f"# {view.get('atid') or args.atid} · 水位 {view.get('watermark')}")
    for r in rows:
        m = r.get("measurement") or {}
        print(f"{r['rank']:>2}  {r['recipe_id'][:12]}  gain={m.get('gain')!s:<12} "
              f"value={m.get('value')!s:<12} 同组={r.get('family_size', 1)}  "
              f"{r.get('source_trial_id')}  ({r.get('source_run_id')})")
    for algo, b in sorted((view.get("baselines") or {}).items()):
        print(f"baseline {algo}: {b['recipe_id'][:12]}  {b.get('source_trial_id')}")
    return 0 if rows or view.get("baselines") else 1


def add_library_parser(sub) -> None:
    lp = sub.add_parser("library", help="v2.18 中心化经验库:服务启停 / 导入 / 迁移")
    lsub = lp.add_subparsers(dest="library_command", required=True)

    def _common(p, *, server: bool = False):
        p.add_argument("--uri", help="客户端访问地址(覆盖 library.uri)")
        p.add_argument("--namespace", help="命名空间(覆盖 library.namespace)")
        p.add_argument("--root", help="库根目录(覆盖 library.service.root)")
        if server:
            p.add_argument("--db-url", dest="db_url")
            p.add_argument("--artifacts-url", dest="artifacts_url")
            p.add_argument("--host", help="监听地址(Q44 默认 127.0.0.1)")
            p.add_argument("--port", type=int)
            p.add_argument("--python", help="装有 fastapi/uvicorn 的解释器")

    p = lsub.add_parser("serve", help="启动经验库服务(--daemon 后台)")
    _common(p, server=True)
    p.add_argument("--daemon", action="store_true")
    p.add_argument("--wait", type=float, default=60, help="--daemon 时等健康检查的秒数")
    p.set_defaults(func=cmd_library_serve)

    p = lsub.add_parser("status", help="服务健康、pid、水位")
    _common(p)
    p.set_defaults(func=cmd_library_status)

    p = lsub.add_parser("stop", help="停止本机经验库服务")
    _common(p)
    p.set_defaults(func=cmd_library_stop)

    p = lsub.add_parser("import-legacy",
                        help="导入旧 memory/ 与历史 run 的 trial 结果(可重复执行)")
    _common(p)
    p.add_argument("--memory", help="旧库目录(含 global/principles.jsonl 与 top-trials/;"
                   "v2.18.0 起 skill 不再自带,须显式给出)")
    p.add_argument("--extra-principles", action="append",
                   help="额外原则 JSONL,可重复;写法 label=path 或 path")
    p.add_argument("--runs-root", action="append",
                   help="历史 run 搜索根目录,可重复(含 run-state.json 的目录视为 run)")
    p.add_argument("--test-namespace", default="test",
                   help="名称含 smoke/e2e/test 的 run 导入到此命名空间")
    p.add_argument("--dry-run", action="store_true", help="只构造快照,不上传不提交")
    p.add_argument("--report", help="完整导入报告写到此 JSON 文件")
    p.add_argument("--resolve-atid", action="store_true",
                   help="run-state 没有任务指纹的旧 run 用 TaskSpec 解析补齐")
    p.set_defaults(func=cmd_library_import_legacy)

    p = lsub.add_parser("dump", help="整库导出(迁移用,Q36;服务须停止)")
    _common(p, server=True)
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_library_dump)

    p = lsub.add_parser("load", help="整库导入到空库(迁移用,Q36;服务须停止)")
    _common(p, server=True)
    p.add_argument("--from", dest="source", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_library_load)

    # ---- M2:run 快照与合并
    def _model(p):
        p.add_argument("--model", default=os.environ.get("MODELEVO_MODEL", ""),
                       help="执行本操作的模型名(记入 actor 与更新日志,Q27)")

    p = lsub.add_parser("snapshot", help="由 run 目录构造经验快照并提交(不可达时进 outbox)")
    _common(p)
    _model(p)
    p.add_argument("--run", required=True)
    p.add_argument("--dry-run", action="store_true", help="只构造并报告,不上传不提交")
    p.set_defaults(func=cmd_library_snapshot)

    p = lsub.add_parser("submit-pending", help="补交 run 的 outbox 快照")
    _common(p)
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_library_submit_pending)

    p = lsub.add_parser("merge-pending", help="列出待合并的快照")
    _common(p)
    p.set_defaults(func=cmd_library_merge_pending)

    p = lsub.add_parser("merge-context", help="申请合并租约并导出合并上下文")
    _common(p)
    _model(p)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", required=True, help="上下文 JSON 写到此文件")
    p.set_defaults(func=cmd_library_merge_context)

    p = lsub.add_parser("merge-apply", help="提交合并决议(程序校验后落库)")
    _common(p)
    _model(p)
    p.add_argument("--decision", required=True)
    p.add_argument("--dry-run", action="store_true", help="只校验决议,不落库、不释放租约")
    p.set_defaults(func=cmd_library_merge_apply)

    p = lsub.add_parser("curate-plan", help="需要整理的主题组(程序判定;--all 含无变化的组)")
    _common(p)
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_library_curate_plan)

    p = lsub.add_parser("curate-open", help="把一个主题组写成整理快照")
    _common(p)
    _model(p)
    p.add_argument("--group", required=True, help="主题组(原则的主主题 code)")
    p.set_defaults(func=cmd_library_curate_open)

    p = lsub.add_parser("curate-context", help="申请租约并导出整理上下文")
    _common(p)
    _model(p)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_library_curate_context)

    p = lsub.add_parser("curate-next", help="打开下一个需要整理的组并导出上下文(没有则 idle)")
    _common(p)
    _model(p)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_library_curate_next)

    p = lsub.add_parser("curate-apply", help="提交整理决议(程序校验后落库)")
    _common(p)
    _model(p)
    p.add_argument("--decision", required=True)
    p.add_argument("--dry-run", action="store_true", help="只校验,不落库、不释放租约")
    p.set_defaults(func=cmd_library_curate_apply)

    p = lsub.add_parser("merge-abort", help="放弃本次合并:快照记 merge_failed 并释放租约")
    _common(p)
    _model(p)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--lease-token", required=True)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_library_merge_abort)

    p = lsub.add_parser("revert", help="按快照撤销最近一次合并(Q16)")
    _common(p)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_library_revert)

    p = lsub.add_parser("changelog", help="输出更新日志(Markdown)")
    _common(p)
    p.add_argument("--since", type=int, default=0, help="只看提交序号大于此值的条目")
    p.add_argument("--out")
    p.set_defaults(func=cmd_library_changelog)

    p = lsub.add_parser("browse", help="把经验库渲染成明文目录(服务端已自动生成 {root}/browse/)")
    _common(p)
    p.add_argument("--out", required=True, help="输出目录(已存在的镜像目录会被整体替换)")
    p.set_defaults(func=cmd_library_browse)

    p = lsub.add_parser("recipe", help="配方:export 导出为明文目录")
    rsub = p.add_subparsers(dest="recipe_command", required=True)
    p = rsub.add_parser("export", help="导出配方;--run 时渲染成该 run 可运行的副本")
    _common(p)
    p.add_argument("recipe_id", help="配方 id(可用唯一前缀,如 12 位)")
    p.add_argument("--out", required=True)
    p.add_argument("--run", help="填入该 run 的 RUN_DIR 与本 skill 的 SKILL_ROOT")
    p.set_defaults(func=cmd_library_recipe_export)

    p = lsub.add_parser("leaderboard", help="某任务的榜单(同一配方只占一个名次)")
    _common(p)
    p.add_argument("--atid", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_library_leaderboard)

    p = lsub.add_parser("principles", help="列出当前原则")
    _common(p)
    p.add_argument("--include-inactive", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_library_principles)

    p = lsub.add_parser("topics", help="列出主题词表")
    _common(p)
    p.set_defaults(func=cmd_library_topics)

    p = lsub.add_parser("metrics", help="Q31 过程指标(依据覆盖 / 每代证据 / 重试已失败方向 / 库级)")
    _common(p)
    p.add_argument("--run", action="append", help="run 目录,可重复")
    p.set_defaults(func=cmd_library_metrics)

    p = lsub.add_parser("merge-renew", help="合并租约续约(大快照合并超过 30 分钟时)")
    _common(p)
    _model(p)
    p.add_argument("--lease-token", required=True)
    p.add_argument("--extra-sec", type=float, default=None, help="缺省再延 1800 秒")
    p.set_defaults(func=cmd_library_merge_renew)

    p = lsub.add_parser("feedback", help="检索反馈指标(--backfill 先为存量快照补写反馈表)")
    _common(p)
    p.add_argument("--backfill", action="store_true")
    p.add_argument("--top", type=int, default=20)
    p.set_defaults(func=cmd_library_feedback)

    p = lsub.add_parser("domains", help="业务域受控词表与库内使用情况(列出不在词表里的写法)")
    _common(p)
    p.set_defaults(func=cmd_library_domains)

    p = lsub.add_parser("replay", help="检索回放评测(历史 trial 作查询;可换证据口径对比)")
    _common(p)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--mode", choices=("same_task", "new_task", "all"), default="all")
    p.add_argument("--run-dedup", action="store_true", help="证据按 run 去重(对比用)")
    p.add_argument("--multi-factor", type=float, default=None,
                   help="被动证据按 trial 主题数降权的指数(对比用;缺省 = 现行口径)")
    p.add_argument("--max-queries", type=int, default=None)
    p.set_defaults(func=cmd_library_replay)

    p = lsub.add_parser("suggest-aliases", help="按任务画像建议指纹别名(只建议,不写入)")
    _common(p)
    p.add_argument("--min-columns", type=float, default=0.9, help="列 Jaccard 下限")
    p.set_defaults(func=cmd_library_suggest_aliases)

    p = lsub.add_parser("fix-fingerprints",
                        help="存量指纹修复:按 run 目录重推 atid(Null/漂移指纹);默认只出计划")
    _common(p)
    _model(p)
    p.add_argument("--apply", action="store_true", help="执行(追加行,可撤销)")
    p.add_argument("--run-dir", action="append", metavar="RUN_ID=DIR",
                   help="快照里没有 run_dir 的 run,补它的目录;可重复")
    p.set_defaults(func=cmd_library_fix_fingerprints)

    p = lsub.add_parser("backfill-profiles", help="为存量任务回填多维画像(读快照记录的 run 目录)")
    _common(p)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="已有画像的任务也重算(内容未变不写)")
    p.set_defaults(func=cmd_library_backfill_profiles)

    p = lsub.add_parser("alias", help="登记指纹别名:alias 的经验并入 canonical")
    _common(p)
    p.add_argument("--alias", required=True)
    p.add_argument("--canonical", required=True)
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_library_alias)

    p = lsub.add_parser("token-hash", help="生成访问 token 与 --tokens 文件条目")
    p.add_argument("--user", required=True)
    p.add_argument("--team", default="")
    p.add_argument("--token", default="", help="缺省随机生成")
    p.add_argument("--read-only", action="store_true")
    p.set_defaults(func=cmd_library_token_hash)

    p = lsub.add_parser("topics-init", help="写入主题词表初版(仅词表为空时,Q28)")
    _common(p)
    _model(p)
    p.add_argument("--file", required=True, help="YAML:topics: [{code, definition, aliases}]")
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_library_topics_init)
