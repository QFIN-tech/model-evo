"""v2.18.0 M2:从 run 目录构造经验快照(dev 文档 §5.1,Q14)并提交 / 落 outbox(Q26)。

位于 CLI 胶水层:可以使用 runtime 的其他模块(导出文件的逐条校验沿用
``runtime.memory.load_experience_export``),``runtime/library`` 本身不反向依赖。

快照内容:
- discoveries:收尾 subagent 写的 ``experience/exported-global-experiences.yaml``;
- trial_outcomes:全部 trial(含 rejected / invalid / FAILED);
- recipes + measurements:accepted / confirmed 且有代码的 trial,配方去本机路径;
- retrieval:``experience-retrieval-telemetry.jsonl``;
- dossier_draft:``experience/dossier-draft.md``(可选)。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import yaml

from runtime.library.artifacts import build_bundle_manifest, sha256_hex
from runtime.library.identity import resolve_model
from runtime.library.legacy_import import _n_trees, higher_is_better, trial_row
from runtime.library.recipes import RUNTIME_API_VERSION, canonicalize
from runtime.library.repository import LibraryUnavailable
from runtime.library.snapshot import SNAPSHOT_SCHEMA, run_id_for, snapshot_id

RECIPE_VERDICTS = ("accepted", "confirmed")
OUTBOX = Path("experience") / "outbox"


def bundle_id_of(files: dict[str, bytes]) -> str:
    """bundle id = 清单的 sha256,不需要连服务(Q26:构建阶段必须能离线完成)。"""
    return sha256_hex(build_bundle_manifest(
        {n: sha256_hex(b) for n, b in files.items()}))


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _discoveries(run_dir: Path) -> tuple[list[dict], list]:
    src = run_dir / "experience" / "exported-global-experiences.yaml"
    if not src.exists():
        return [], []
    from runtime.memory import load_experience_export

    res = load_experience_export(src)
    out = []
    for idx, _line, item in res.get("items") or []:
        lid = str(item.get("experience_id") or item.get("principle_id") or f"item-{idx}")
        out.append({"local_id": lid, "payload": item})
    return out, res.get("errors") or []


#: 单个文件超过该大小不进配方(代码与配置不会这么大;防止误把数据 / 模型打进 bundle)
MAX_RECIPE_FILE_BYTES = 20 * 1024 * 1024
#: env-manifest 里保留的键(包版本、解释器、GBM 后端)
_ENV_KEYS = ("interpreter", "trial_interpreter", "gbm_backends", "signature", "fingerprint")


def _environment(run_dir: Path) -> bytes | None:
    """配方的运行环境:run 的 env-manifest(preflight 写)压缩成包版本表 + skill 版本。"""
    man = _read_json(run_dir / "env-manifest.json") or {}
    rs = _read_json(run_dir / "run-state.json") or {}
    env = {k: man[k] for k in _ENV_KEYS if k in man}
    env["packages"] = {str(x.get("name")): x.get("version") for x in man.get("packages") or []
                       if isinstance(x, dict) and x.get("import_ok")}
    env.update({k: rs.get(k) for k in ("skill_version", "harness_hash", "harness_protocol_version",
                                       "harness_epoch")
                if rs.get(k) is not None})
    if not env.get("packages") and len(env) == 1:
        return None
    return json.dumps(env, ensure_ascii=False, indent=2, sort_keys=True, default=str).encode("utf-8")


def _dir_files(d: Path, prefix: str, skip: tuple = ()) -> dict[str, bytes]:
    out = {}
    if not d.is_dir():
        return out
    for f in sorted(d.rglob("*")):
        rel = f.relative_to(d)
        if not f.is_file() or "__pycache__" in rel.parts or f.suffix == ".pyc" \
                or str(rel) in skip or f.stat().st_size > MAX_RECIPE_FILE_BYTES:
            continue
        out[f"{prefix}{rel.as_posix()}"] = f.read_bytes()
    return out


def _recipe_files(run_dir: Path, tdir: Path) -> dict[str, bytes] | None:
    """一个配方的完整文件集(经验库存内容,不存路径):

    - ``train.py`` / ``trial.yaml``:入口与配置(放根目录,渲染与热启动按此取);
    - ``code/<其余>``、``config/<其余>``:trial 的其余代码模块(如 deep_seq.py)与配置
      (search-space.yaml、hpo-warm-start.json …);
    - ``task-spec.yaml``、``hypothesis.md``、``evaluation.json``:任务与结论上下文;
    - ``environment.json``:包版本、解释器、skill 版本(复现用);
    - ``data_view/``:trial 用非 v1 数据视图时,视图构建脚本与 manifest。
    """
    code = tdir / "code" / "train.py"
    if not code.is_file():
        return None
    files = {"train.py": code.read_bytes()}
    for name, p in (("trial.yaml", tdir / "config" / "trial.yaml"),
                    ("task-spec.yaml", run_dir / "task-spec.yaml"),
                    ("hypothesis.md", tdir / "hypothesis.md"),
                    ("evaluation.json", tdir / "evaluation.json")):
        if p.is_file():
            files[name] = p.read_bytes()
    files.update(_dir_files(tdir / "code", "code/", skip=("train.py",)))
    files.update(_dir_files(tdir / "config", "config/", skip=("trial.yaml",)))
    env = _environment(run_dir)
    if env:
        files["environment.json"] = env
    cfg = yaml.safe_load(files.get("trial.yaml", b"{}").decode("utf-8")) or {}
    view = cfg.get("data_view")
    if view and view != "v1":
        for name, p in ((f"data_view/{view}.py", run_dir / "views" / f"{view}.py"),
                        ("data_view/manifest.json",
                         run_dir / "sample-profile" / f"splits-{view}" / "manifest.json")):
            if p.is_file():
                files[name] = p.read_bytes()
    return files


def _task_profile(run_dir: Path, atid: str | None) -> dict:
    """v2.18.1:任务画像随快照提交(服务端按 atid 追加保存,供其它任务算相似度)。
    优先用固定视图时算好的;没有时由 task-spec.yaml 现算。"""
    view = _read_json(run_dir / "experience" / "library-view.json") or {}
    if view.get("task_profile"):
        return view["task_profile"]
    spec_p = run_dir / "task-spec.yaml"
    if not spec_p.is_file():
        return {}
    try:
        from runtime.library_view import task_profile_for

        return task_profile_for(run_dir, yaml.safe_load(spec_p.read_text(encoding="utf-8")) or {},
                                atid)
    except Exception:  # noqa: BLE001  画像缺失不影响快照
        return {}


def build_run_snapshot(run_dir, repo, *, model: str = "", upload: bool = True,
                       stage: dict | None = None) -> tuple[dict, dict]:
    """返回 (快照载荷, 报告)。

    构建阶段**完全不访问网络**:配方 bundle id 由本地计算(内容寻址,不需要服务确认),
    规范化后的文件按 recipe_id 收进 ``stage``(缺省时只算哈希不保留内容)。
    ``upload`` 仅用于 dry-run 语义(不暂存文件);真正上传统一在 :func:`submit_or_queue`。
    这是 Q26 的硬要求:服务不可达时收尾仍须把快照<em>连同配方</em>落进 outbox。
    """
    run_dir = Path(run_dir).resolve()
    rs = _read_json(run_dir / "run-state.json") or {}
    rid = run_id_for(run_dir)
    report: dict = {"run_id": rid, "warnings": []}
    discoveries, derr = _discoveries(run_dir)
    if derr:
        report["warnings"].append({"exported_global_experiences_errors": derr[:10]})

    trials, recipes, measurements, warm_measurements = [], [], [], []
    for tdir in sorted((run_dir / "trials").glob("*")) if (run_dir / "trials").is_dir() else []:
        if not (tdir / "trial.yaml").is_file():
            continue
        try:
            row = trial_row(tdir)
        except Exception as exc:  # noqa: BLE001  单个 trial 坏数据不阻断快照
            report["warnings"].append(f"{tdir.name}: {type(exc).__name__}: {exc}")
            continue
        if not row:
            continue
        row["run_id"] = rid
        trials.append(row)
        warm = ((yaml.safe_load((tdir / "trial.yaml").read_text(encoding="utf-8")) or {})
                .get("execution") or {}).get("warm_start") or {}
        if warm.get("recipe_id") and row.get("metric_value") is not None:
            # 热启动 trial 是库内配方在本 run 上的一次复测(替代旧的经验回填)
            warm_measurements.append({
                "recipe_id": warm["recipe_id"], "kind": "warm_remeasure", "run_id": rid,
                "trial_id": row["trial_id"], "primary_metric": row.get("primary_metric"),
                "higher_is_better": higher_is_better(row.get("primary_metric")),
                "value": row.get("metric_value"), "delta_baseline": row.get("delta_baseline"),
                "source_trial_id": warm.get("source_trial_id")})
            continue
        if row.get("verdict") not in RECIPE_VERDICTS or row.get("metric_value") is None:
            continue
        files = _recipe_files(run_dir, tdir)
        if not files:
            continue
        canon, cinfo = canonicalize(files, run_dir=str(run_dir))
        rec_id = bundle_id_of(canon)
        if stage is not None and upload:
            stage[rec_id] = canon
        cfg = yaml.safe_load(files.get("trial.yaml", b"{}").decode("utf-8")) or {}
        row["recipe_id"] = rec_id
        recipes.append({"recipe_id": rec_id, "atid": rs.get("abstract_task_id"),
                        "kind": "top_trial", "algorithm": row.get("algorithm"),
                        "source_run_id": rid, "source_trial_id": row["trial_id"],
                        "runtime_api_version": RUNTIME_API_VERSION,
                        "is_hpo": row.get("action") == "hpo",
                        "n_trees": _n_trees(cfg), "canonicalization": cinfo})
        ev = _read_json(tdir / "evaluation.json") or {}
        anchor = ((ev.get("delta") or {}).get("vs_anchor_paired_bootstrap") or {})
        ci = anchor.get("delta_ci") if isinstance(anchor.get("delta_ci"), list) else [None, None]
        measurements.append({
            "recipe_id": rec_id, "kind": "export", "run_id": rid,
            "trial_id": row["trial_id"], "primary_metric": row.get("primary_metric"),
            "higher_is_better": higher_is_better(row.get("primary_metric")),
            "value": row.get("metric_value"), "delta_baseline": row.get("delta_baseline"),
            "ci_low": ci[0] if ci else None, "ci_high": ci[1] if len(ci) > 1 else None,
            "p_positive": anchor.get("p_positive"),
            "confirmed": row.get("verdict") == "confirmed"})
    # 同一内容只登记一次(多个 trial 可能代码与配置完全相同)
    seen, uniq = set(), []
    for r in recipes:
        if r["recipe_id"] not in seen:
            seen.add(r["recipe_id"])
            uniq.append(r)

    tel = run_dir / "experience-retrieval-telemetry.jsonl"
    logs = []
    if tel.is_file():
        for line in tel.read_text(encoding="utf-8").splitlines():
            try:
                logs.append(json.loads(line))
            except ValueError:
                continue
    draft = run_dir / "experience" / "dossier-draft.md"
    model, bad_model = resolve_model(model)
    if bad_model:
        report["warnings"].append(
            f"--model {bad_model!r} 不是模型名(像 trial / run id),已忽略"
            + (f",改用环境变量里的 {model!r}" if model else ";快照 run.model 记为空")
            + "。请传执行收尾的 LLM 模型名,如 claude-opus-5-5")
    payload = {
        "schema": SNAPSHOT_SCHEMA, "kind": "run",
        "run": {"run_id": rid, "run_dir": str(run_dir),
                "atid": rs.get("abstract_task_id"), "skill_version": rs.get("skill_version"),
                "status": rs.get("status"), "model": model,
                "task_profile": _task_profile(run_dir, rs.get("abstract_task_id"))},
        "discoveries": discoveries, "trial_outcomes": trials, "recipes": uniq,
        "measurements": measurements + warm_measurements, "retrieval": {"logs": logs},
        "dossier_draft": draft.read_text(encoding="utf-8") if draft.is_file() else "",
    }
    report.update({"snapshot_id": snapshot_id(payload), "discoveries": len(discoveries),
                   "trial_outcomes": len(trials), "recipes": len(uniq),
                   "retrieval_logs": len(logs)})
    return payload, report


def _put_staged(repo, staged: dict) -> list[str]:
    """把暂存的配方 bundle 上传到 CAS。已存在的对象不重传(内容寻址)。"""
    from runtime.library.artifacts import ArtifactError

    sent = []
    for rid, files in (staged or {}).items():
        if repo.has_artifact(rid):
            continue
        got = repo.put_bundle(files)
        if got != rid:      # 内容寻址:哈希必须自洽,否则说明规范化不稳定
            raise ArtifactError(f"配方 bundle 哈希不一致: {got} != {rid}")
        sent.append(rid)
    return sent


def _queue(run_dir: Path, payload: dict, *, namespace: str, staged: dict) -> dict:
    """落 outbox:快照 + 暂存配方一起写(配方以文件形式保存在 outbox/cas/)。"""
    box = run_dir / OUTBOX
    box.mkdir(parents=True, exist_ok=True)
    sid = snapshot_id(payload)
    cas_dir = box / "cas" / sid[:12]
    for rid, files in (staged or {}).items():
        for name, data in files.items():
            out = cas_dir / rid / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
    path = box / f"snapshot-{sid[:12]}.json"
    path.write_text(json.dumps({"namespace": namespace, "queued_at": time.time(),
                                "payload": payload,
                                "staged_recipes": sorted(staged or {})},
                               ensure_ascii=False), encoding="utf-8")
    return {"status": "queued", "outbox": str(path),
            "queued_recipes": len(staged or {})}


def submit_or_queue(run_dir, repo, payload: dict, *, namespace: str,
                    staged: dict | None = None) -> dict:
    """上传暂存配方并提交快照;服务不可达时把快照与配方一起写进 outbox(Q26)。

    先传产物再提交快照:服务端提交时会校验 bundle 存在,顺序反了会 422。
    任何一步不可达都整体降级为 outbox —— 收尾不能因为服务抖动丢经验。
    """
    run_dir = Path(run_dir)
    sid = snapshot_id(payload)
    (run_dir / "experience").mkdir(parents=True, exist_ok=True)
    (run_dir / "experience" / "snapshot.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        _put_staged(repo, staged or {})
        receipt = repo.submit_snapshot(payload, namespace=namespace)
        return {"status": "submitted", "receipt": receipt,
                "uploaded_recipes": len(staged or {})}
    except LibraryUnavailable as exc:
        out = _queue(run_dir, payload, namespace=namespace, staged=staged or {})
        out["error"] = str(exc)
        return out


def submit_pending(run_dir, repo) -> list[dict]:
    """补交 outbox。先补齐配方产物,再提交快照;失败保留原文件等待下次重试。"""
    box = Path(run_dir) / OUTBOX
    out = []
    for path in sorted(box.glob("snapshot-*.json")) if box.is_dir() else []:
        item = json.loads(path.read_text(encoding="utf-8"))
        sid = snapshot_id(item["payload"])
        staged = {}
        for rid in item.get("staged_recipes") or []:
            d = box / "cas" / sid[:12] / rid
            if d.is_dir():
                staged[rid] = {str(p.relative_to(d)): p.read_bytes()
                               for p in sorted(d.rglob("*")) if p.is_file()}
        _put_staged(repo, staged)
        receipt = repo.submit_snapshot(item["payload"], namespace=item["namespace"])
        sent = box / "sent"
        sent.mkdir(exist_ok=True)
        path.rename(sent / path.name)
        out.append({"file": path.name, "receipt": receipt,
                    "uploaded_recipes": len(staged)})
    return out


def freeze_baseline(run_dir, trial, *, atid: str, settings: dict) -> dict:
    """首次运行该任务时,把现场渲染并已评估的 baseline 配方固化进经验库(Q21)。

    一个快照(kind=baseline)= 配方(kind=baseline)+ 测量 + pin_baseline 治理事件。
    提交前再向服务确认该 (atid, 算法) 仍没有固化配方(先到者为准,与旧实现
    "只写一次"一致);服务不可达时跳过并告警,下一次 baseline 会再尝试。
    """
    from runtime.library_view import client

    run_dir = Path(run_dir)
    tdir = Path(trial.trial_dir)
    algo = trial.data.get("algorithm")
    repo = client(settings, run=str(run_dir))
    if repo.frozen_baseline(settings["namespace"], atid, algo):
        return {"status": "exists"}
    files = _recipe_files(run_dir, tdir)
    if not files:
        return {"status": "no_code"}
    canon, cinfo = canonicalize(files, run_dir=str(run_dir))
    rid = repo.put_bundle(canon)
    row = trial_row(tdir) or {}
    rr = run_id_for(run_dir)
    payload = {
        "schema": SNAPSHOT_SCHEMA, "kind": "baseline",
        "run": {"run_id": rr, "run_dir": str(run_dir), "atid": atid},
        "recipes": [{"recipe_id": rid, "atid": atid, "kind": "baseline", "algorithm": algo,
                     "source_run_id": rr, "source_trial_id": trial.trial_id,
                     "runtime_api_version": RUNTIME_API_VERSION, "is_hpo": False,
                     "n_trees": _n_trees(yaml.safe_load(files.get("trial.yaml", b"{}")
                                                        .decode("utf-8")) or {}),
                     "canonicalization": cinfo}],
        "measurements": [{"recipe_id": rid, "kind": "baseline", "run_id": rr,
                          "trial_id": trial.trial_id,
                          "primary_metric": row.get("primary_metric"),
                          "higher_is_better": higher_is_better(row.get("primary_metric")),
                          "value": row.get("metric_value")}],
        "governance": [{"subject_type": "recipe", "subject_id": rid, "action": "pin_baseline",
                        "payload": {"atid": atid, "algorithm": algo,
                                    "source_trial": trial.trial_id},
                        "reason": "该任务首次运行的 baseline 配方;同 atid 的后续 run 复用"}],
    }
    rec = repo.submit_snapshot(payload, namespace=settings["namespace"])
    return {"status": "frozen", "recipe_id": rid, "receipt": rec}

