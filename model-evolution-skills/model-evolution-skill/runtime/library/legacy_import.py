"""v2.18.0 经验库:种子与历史导入(dev 文档 §10,Q29 Q34)。

把旧 ``memory/``(principles.jsonl、top-trials/)和各历史 run 的 trial 结果,
转成快照提交进经验库。可重复执行:快照按内容哈希幂等;已导入过的原则 id 不再
重复提交;历史 run 有新 trial 时提交新快照,trial 视图取最新一行。

只读源文件,不修改旧库。本模块不 import runtime 的其他模块(指标方向等小规则
在此内置)。
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .artifacts import build_bundle_manifest, sha256_hex
from .recipes import RUNTIME_API_VERSION, canonicalize
from .snapshot import SNAPSHOT_SCHEMA, run_id_for

#: 库内现有配方经 v2.16 单段硬化(docs/MODELEVO_V2_16_*)
LEGACY_RUNTIME_API = RUNTIME_API_VERSION
_LOWER_IS_BETTER = {"rmse", "mae", "mse", "logloss", "log_loss", "mape", "rmsle",
                    "brier", "brier_score"}
_TEST_RUN_RE = re.compile(r"(smoke|e2e|pytest|(^|[-_./])test([-_./]|$))", re.I)
_TREE_KEYS = ("n_estimators", "num_boost_round", "iterations", "num_iterations",
              "n_iter")
#: 不进配方 bundle 的文件(元数据另存 recipe.meta)
_BUNDLE_SKIP = {"frozen.yaml"}


def _read_yaml(path: Path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def higher_is_better(metric: str | None) -> bool | None:
    if not metric:
        return None
    return str(metric).lower() not in _LOWER_IS_BETTER


def _claim_key(entry: dict) -> str:
    text = str(entry.get("claim") or entry.get("statement") or "").strip()
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _eid(entry: dict) -> str | None:
    return entry.get("experience_id") or entry.get("principle_id")


def _is_top_trial_index(entry: dict) -> bool:
    return entry.get("category") == "top_trial_baseline" or \
        entry.get("experience_type") == "top_trial_baseline"


# --------------------------------------------------------------------------
# 原则
# --------------------------------------------------------------------------

def load_principles(sources: list[tuple[str, Path]]) -> tuple[list[dict], dict]:
    """按顺序读多个 JSONL;先到者优先,按 id 与 claim 去重。返回 (条目, 报告)。"""
    seen_ids, seen_claims = set(), set()
    out, report = [], {}
    for label, path in sources:
        path = Path(path)
        stat = {"path": str(path), "rows": 0, "kept": 0, "duplicates": 0,
                "top_trial_index": 0, "bad_lines": 0}
        if not path.exists():
            stat["missing"] = True
            report[label] = stat
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            stat["rows"] += 1
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                stat["bad_lines"] += 1
                continue
            if not isinstance(entry, dict) or not _eid(entry):
                stat["bad_lines"] += 1
                continue
            if _is_top_trial_index(entry):
                stat["top_trial_index"] += 1
                continue
            if _eid(entry) in seen_ids or _claim_key(entry) in seen_claims:
                stat["duplicates"] += 1
                continue
            seen_ids.add(_eid(entry))
            seen_claims.add(_claim_key(entry))
            out.append({"source": label, "entry": entry})
            stat["kept"] += 1
        report[label] = stat
    return out, report


def _group_key(entry: dict) -> str:
    ev = entry.get("evidence") if isinstance(entry.get("evidence"), dict) else {}
    tids = ev.get("task_ids") or []
    return str(tids[0]) if tids else (entry.get("task_id") or "unknown")


def _entry_time(entry: dict) -> str:
    t = entry.get("created_at") or entry.get("updated_at")
    if t:
        return str(t)
    m = re.match(r"exp-(\d{6})", str(_eid(entry)))
    return f"{m.group(1)[:4]}-{m.group(1)[4:]}" if m else "9999"


def principle_snapshots(items: list[dict], *, skip_ids: set[str]) -> list[dict]:
    """按来源 run(evidence.task_ids[0])分组成 legacy_principles 快照,按组内最早
    时间排序。``skip_ids``:库里已有的原则 id(再次导入只提交新增)。"""
    groups: dict[tuple, list] = defaultdict(list)
    for it in items:
        if _eid(it["entry"]) in skip_ids:
            continue
        groups[(it["source"], _group_key(it["entry"]))].append(it["entry"])
    payloads = []
    for (source, key), entries in groups.items():
        entries.sort(key=lambda e: (_entry_time(e), str(_eid(e))))
        atids = Counter()
        for e in entries:
            ev = e.get("evidence") if isinstance(e.get("evidence"), dict) else {}
            for a in ev.get("abstract_task_ids") or []:
                atids[a] += 1
        payloads.append({
            "schema": SNAPSHOT_SCHEMA, "kind": "legacy_principles",
            "run": {"run_id": f"legacy:{source}:{key}",
                    "atid": atids.most_common(1)[0][0] if atids else None,
                    "source": source, "group": key},
            "discoveries": [{"local_id": str(_eid(e)), "payload": e} for e in entries],
            "_order": _entry_time(entries[0]),
        })
    payloads.sort(key=lambda p: (p.pop("_order"), p["run"]["run_id"]))
    return payloads


# --------------------------------------------------------------------------
# top-trials 配方
# --------------------------------------------------------------------------

def _bundle_files(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())
            if p.is_file() and p.name not in _BUNDLE_SKIP}


def _n_trees(trial_cfg: dict) -> int | None:
    hp = trial_cfg.get("hyperparameters") if isinstance(trial_cfg, dict) else None
    if not isinstance(hp, dict):
        return None
    for k in _TREE_KEYS:
        if isinstance(hp.get(k), (int, float)):
            return int(hp[k])
    return None


def _put_bundle(repo, files: dict[str, bytes], dry_run: bool) -> str:
    if dry_run or repo is None:
        hashes = {n: sha256_hex(b) for n, b in files.items()}
        return sha256_hex(build_bundle_manifest(hashes))
    return repo.put_bundle(files)


def top_trial_snapshots(top_dir: Path, repo, *, dry_run: bool = False) -> tuple[list[dict], dict]:
    """每个 atid 目录一份 legacy_recipes 快照:rank 配方 + 测量,baseline 配方 +
    pin_baseline 治理事件。``*.bak*`` 目录跳过;leaderboard.yaml / experiences.yaml
    不导入(由视图重建)。"""
    payloads, report = [], {}
    if not top_dir or not Path(top_dir).is_dir():
        return payloads, {"missing": str(top_dir)}
    for adir in sorted(Path(top_dir).iterdir()):
        if not adir.is_dir() or ".bak" in adir.name:
            continue
        atid = adir.name
        lb_path = adir / "leaderboard.yaml"
        lb = (_read_yaml(lb_path) or {}) if lb_path.exists() else {}
        rows_by_trial = {r.get("trial_id"): r for r in lb.get("rows") or []
                         if isinstance(r, dict)}
        metric = lb.get("metric_name")
        hib = lb.get("higher_is_better")
        recipes, meas, gov, info = [], [], [], {"ranks": 0, "baselines": 0,
                                                 "residual_abs_paths": []}
        for rdir in sorted(adir.glob("rank*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
            if not rdir.is_dir():
                continue
            cfg = _read_yaml(rdir / "trial.yaml") if (rdir / "trial.yaml").exists() else {}
            trial_id = (cfg or {}).get("trial_id")
            row = rows_by_trial.get(trial_id) or {}
            files, cinfo = canonicalize(_bundle_files(rdir),
                                        run_dir=row.get("source_run"))
            rid = _put_bundle(repo, files, dry_run)
            src_run = row.get("source_run") or cinfo.get("source_run_dir")
            recipes.append({
                "recipe_id": rid, "atid": atid, "kind": "top_trial",
                "algorithm": (cfg or {}).get("algorithm") or row.get("algorithm"),
                "source_run_id": run_id_for(src_run) if src_run else None,
                "source_trial_id": trial_id, "runtime_api_version": LEGACY_RUNTIME_API,
                "is_hpo": False, "n_trees": _n_trees(cfg or {}),
                "legacy_rank": int(re.sub(r"\D", "", rdir.name) or 0),
                "source_run_dir": src_run, "canonicalization": cinfo})
            ev = _read_json(rdir / "evaluation.json") if (rdir / "evaluation.json").exists() else {}
            pm = (ev or {}).get("primary_metric") or {}
            delta = (ev or {}).get("delta") or {}
            m_name = metric or pm.get("name")
            meas.append({
                "recipe_id": rid, "kind": "legacy",
                "run_id": run_id_for(src_run) if src_run else None, "trial_id": trial_id,
                "primary_metric": m_name,
                "higher_is_better": hib if hib is not None else higher_is_better(m_name),
                "value": row.get("metric", pm.get("value")),
                "delta_baseline": row.get("delta_vs_baseline", delta.get("vs_baseline")),
                "gain": row.get("improvement_vs_baseline"),
                "confirmed": row.get("confirmed"),
                "legacy_rank": int(re.sub(r"\D", "", rdir.name) or 0),
                "remeasured": row.get("remeasured"), "updated": row.get("updated")})
            info["ranks"] += 1
            info["residual_abs_paths"] += cinfo["residual_abs_paths"]
        for bdir in sorted((adir / "baseline").glob("*")) if (adir / "baseline").is_dir() else []:
            if not bdir.is_dir():
                continue
            frozen = _read_yaml(bdir / "frozen.yaml") if (bdir / "frozen.yaml").exists() else {}
            frozen = frozen or {}
            files, cinfo = canonicalize(_bundle_files(bdir),
                                        run_dir=frozen.get("source_run"))
            rid = _put_bundle(repo, files, dry_run)
            src_run = frozen.get("source_run") or cinfo.get("source_run_dir")
            cfg = _read_yaml(bdir / "trial.yaml") if (bdir / "trial.yaml").exists() else {}
            recipes.append({
                "recipe_id": rid, "atid": atid, "kind": "baseline",
                "algorithm": frozen.get("algorithm") or bdir.name,
                "source_run_id": run_id_for(src_run) if src_run else None,
                "source_trial_id": frozen.get("source_trial"),
                "runtime_api_version": LEGACY_RUNTIME_API, "is_hpo": False,
                "n_trees": _n_trees(cfg or {}), "frozen": frozen,
                "canonicalization": cinfo})
            ev = _read_json(bdir / "evaluation.json") if (bdir / "evaluation.json").exists() else {}
            pm = (ev or {}).get("primary_metric") or {}
            if pm.get("value") is not None:
                meas.append({"recipe_id": rid, "kind": "baseline",
                             "run_id": run_id_for(src_run) if src_run else None,
                             "trial_id": frozen.get("source_trial"),
                             "primary_metric": pm.get("name"),
                             "higher_is_better": higher_is_better(pm.get("name")),
                             "value": pm.get("value")})
            gov.append({"subject_type": "recipe", "subject_id": rid,
                        "action": "pin_baseline",
                        "payload": {"atid": atid, "algorithm": frozen.get("algorithm") or bdir.name,
                                    **{k: frozen.get(k) for k in ("frozen_at", "source_run",
                                                                  "source_trial")}},
                        "reason": "旧库 baseline/<algo>/frozen.yaml 固化配方"})
            info["baselines"] += 1
            info["residual_abs_paths"] += cinfo["residual_abs_paths"]
        if not recipes:
            continue
        # 同一内容只登记一次。baseline 优先:旧榜单可能把 baseline trial 自己列为
        # 末位 rank(Δ=0),两者 bundle 相同;记成 top_trial 会让固化 baseline 查不到
        recipes.sort(key=lambda r: r["kind"] != "baseline")
        uniq, seen = [], set()
        for r in recipes:
            if r["recipe_id"] in seen:
                info.setdefault("same_content_as_baseline", []).append(
                    r.get("source_trial_id"))
                continue
            seen.add(r["recipe_id"])
            uniq.append(r)
        recipes = uniq
        info["residual_abs_paths"] = sorted(set(info["residual_abs_paths"]))[:20]
        report[atid] = info
        payloads.append({"schema": SNAPSHOT_SCHEMA, "kind": "legacy_recipes",
                         "run": {"run_id": f"legacy-top-trials:{atid}", "atid": atid},
                         "recipes": recipes, "measurements": meas, "governance": gov})
    return payloads, report


# --------------------------------------------------------------------------
# 历史 run 的 trial 结果
# --------------------------------------------------------------------------

def discover_runs(roots: list[Path], max_depth: int = 5) -> list[Path]:
    found = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop()
            if (d / "run-state.json").is_file():
                found.append(d)
                continue        # run 目录内部不再下钻
            if depth >= max_depth:
                continue
            try:
                kids = [p for p in d.iterdir() if p.is_dir() and not p.is_symlink()
                        and not p.name.startswith((".", "__"))]
            except OSError:
                continue
            stack.extend((k, depth + 1) for k in kids)
    return sorted(set(found))


def _text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, dict):
        return v.get("statement") or json.dumps(v, ensure_ascii=False)
    return str(v)


def trial_row(trial_dir: Path) -> dict | None:
    t = _read_yaml(trial_dir / "trial.yaml") or {}
    if not isinstance(t, dict) or not t.get("trial_id"):
        return None
    ev_path = trial_dir / "evaluation.json"
    e = _read_json(ev_path) if ev_path.exists() else {}
    e = e if isinstance(e, dict) else {}
    pm = e.get("primary_metric") or (t.get("evaluation") or {}).get("primary_metric") or {}
    delta = e.get("delta") or (t.get("evaluation") or {}).get("delta") or {}
    boot = delta.get("vs_parent_paired_bootstrap") or {}
    ci = boot.get("delta_ci")
    ci = list(ci) if isinstance(ci, (list, tuple)) else []
    ex = t.get("execution") or {}
    exp_ids = list(t.get("experience_ids") or [])
    exp_ids += [b.get("experience_id") for b in t.get("experience_bindings")
                or ex.get("experience_bindings") or []
                if isinstance(b, dict) and b.get("experience_id")]
    hyp = t.get("hypothesis")
    status = t.get("status")
    verdict = e.get("verdict")
    failure = None
    if verdict in ("rejected", "invalid") or status == "FAILED":
        failure = e.get("reason") or (
            f"status={status} returncode={(t.get('result') or {}).get('returncode')}")
    return {
        "trial_id": t["trial_id"], "generation": t.get("generation"),
        "action": t.get("search_action") or t.get("action"),
        "algorithm": t.get("algorithm"),
        "parent_trial_ids": t.get("parent_trial_ids") or t.get("parent_ids") or [],
        "changed_modules": t.get("changed_modules") or ex.get("changed_modules") or [],
        "family_hints": t.get("family_hints") or [], "data_view": t.get("data_view"),
        "hypothesis": _text(hyp), "status": status, "verdict": verdict,
        "primary_metric": pm.get("name"), "higher_is_better": higher_is_better(pm.get("name")),
        "metric_value": pm.get("value"), "delta_parent": delta.get("vs_parent"),
        "delta_parent_ci_low": ci[0] if len(ci) > 0 else None,
        "delta_parent_ci_high": ci[1] if len(ci) > 1 else None,
        "p_positive": boot.get("p_positive", boot.get("p_better")),
        "delta_baseline": delta.get("vs_baseline", delta.get("vs_anchor")),
        "failure_reason": failure, "experience_ids": exp_ids or None,
        # v2.18.0:LLM 在 new-trial 时声明的主题与依据(旧 run 没有)
        "topics": t.get("topics") or None, "basis": t.get("basis") or None,
        "hypothesis_detail": hyp if isinstance(hyp, dict) else None,
        "planner_proposal_id": t.get("planner_proposal_id"),
        "runtime_sec": (t.get("result") or {}).get("runtime_sec"),
        # v2.19.1:σ_seed/SE 实测比值(两侧都落盘时才算)。落在 extra
        # (不在 TRIAL_COLUMNS 里的键都进 extra),不需要迁移;
        # 由 store.run_view 按 data_base 聚合,作为新 run 的 σ_seed 先验。
        "sigma_seed_ratio": _sigma_seed_ratio(e),
    }


#: 只有这些来源的 σ_seed 是本 run 实测的(见 confirmation.resolve_sigma_seed)
_MEASURED_SIGMA_SOURCES = ("run_replicates", "multi_seed")


def _sigma_seed_ratio(ev: dict) -> float | None:
    """配对检验的 σ_seed / SE_bootstrap 实测比值(v2.19.1)。

    只取 floor 之前的原始估计 ``sigma_seed_estimate``,且来源必须是实测
    (复现组 / multi-seed)。落盘的 ``sigma_seed`` 可能是 floor(1×SE)或
    比值先验本身,拿它入库会让库内中位数退化成先验并自我循环;没有
    ``sigma_seed_estimate`` 的旧记录一律跳过(其 floor 后的值无法还原)。
    """
    if not isinstance(ev, dict):
        return None
    delta = ev.get("delta")
    if not isinstance(delta, dict):
        return None
    for c in delta.values():
        if not isinstance(c, dict):
            continue
        if c.get("sigma_seed_estimate_source") not in _MEASURED_SIGMA_SOURCES:
            continue
        try:
            sig = float(c.get("sigma_seed_estimate"))
            se = float(c.get("se_bootstrap"))
        except (TypeError, ValueError):
            continue
        if sig > 0 and se > 0:
            return round(sig / se, 6)
    return None


def run_evidence_snapshot(run_dir: Path, atid_resolver=None) -> tuple[dict | None, list[str]]:
    """``atid_resolver(run_dir) -> atid | None``:run-state 没有记录任务指纹的旧 run
    用它补(由 CLI 胶水层提供,本模块不依赖 runtime.spec)。"""
    run_dir = Path(run_dir)
    errors: list[str] = []
    try:
        rs = _read_json(run_dir / "run-state.json")
    except (OSError, ValueError) as exc:
        return None, [f"run-state.json 不可读: {exc}"]
    atid, atid_source = rs.get("abstract_task_id"), rs.get("abstract_task_id_source")
    if not atid and atid_resolver is not None:
        try:
            atid = atid_resolver(run_dir)
            atid_source = "resolved_at_import" if atid else None
        except Exception as exc:  # noqa: BLE001
            errors.append(f"atid 解析失败: {exc}")
    rid = run_id_for(run_dir)
    rows = []
    for tdir in sorted((run_dir / "trials").glob("*")) if (run_dir / "trials").is_dir() else []:
        if not (tdir / "trial.yaml").is_file():
            continue
        try:
            row = trial_row(tdir)
        except Exception as exc:  # noqa: BLE001  单个 trial 坏数据不阻断整 run
            errors.append(f"{tdir.name}: {type(exc).__name__}: {exc}")
            continue
        if row:
            row["run_id"] = rid
            rows.append(row)
    if not rows:
        return None, errors or ["无 trial"]
    return {"schema": SNAPSHOT_SCHEMA, "kind": "legacy_evidence",
            "run": {"run_id": rid, "run_dir": str(run_dir.resolve()),
                    "atid": atid, "atid_source": atid_source,
                    "skill_version": rs.get("skill_version"),
                    "status": rs.get("status"), "created_at": rs.get("created_at")},
            "trial_outcomes": rows}, errors


def namespace_for_run(run_dir: Path, default: str, test_namespace: str) -> str:
    # 只看 run 目录名:上级路径里的 pytest 临时目录等不代表 run 本身是测试
    return test_namespace if _TEST_RUN_RE.search(Path(run_dir).name) else default


# --------------------------------------------------------------------------
# 编排
# --------------------------------------------------------------------------

def _existing_principle_ids(repo, namespace: str) -> set[str]:
    ids = set()
    for s in repo.list_snapshots(namespace, kind="legacy_principles"):
        snap = repo.get_snapshot(namespace, s["snapshot_id"]) or {}
        ids.update(d["local_id"] for d in snap.get("discoveries") or [])
    return ids


def import_legacy(repo, *, namespace: str, principle_sources: list[tuple[str, Path]],
                  top_trials_dir: Path | None, run_roots: list[Path],
                  test_namespace: str = "test", dry_run: bool = False,
                  actor: dict | None = None, atid_resolver=None) -> dict:
    """导入种子与历史数据。``dry_run``:只构造快照、不上传不提交(repo 可为 None)。"""
    report: dict = {"namespace": namespace, "dry_run": dry_run, "snapshots": [],
                    "errors": []}

    def _submit(payload: dict, ns: str) -> None:
        item = {"kind": payload["kind"], "run_id": payload["run"].get("run_id"),
                "namespace": ns,
                "counts": {k: len(payload.get(k) or []) for k in
                           ("discoveries", "trial_outcomes", "recipes", "measurements",
                            "governance")}}
        if not dry_run:
            try:
                item["receipt"] = repo.submit_snapshot(payload, namespace=ns, actor=actor)
            except Exception as exc:  # noqa: BLE001  逐快照报告,不中断其余导入
                item["error"] = f"{type(exc).__name__}: {exc}"
                report["errors"].append(item)
        report["snapshots"].append(item)

    items, report["principle_sources"] = load_principles(principle_sources)
    skip = set() if dry_run or repo is None else _existing_principle_ids(repo, namespace)
    report["principles_already_imported"] = len(skip)
    for p in principle_snapshots(items, skip_ids=skip):
        _submit(p, namespace)

    payloads, report["top_trials"] = top_trial_snapshots(top_trials_dir, repo,
                                                         dry_run=dry_run)
    for p in payloads:
        _submit(p, namespace)

    runs = discover_runs(run_roots)
    report["runs"] = {"found": len(runs), "skipped": {}}
    dated = []
    for rd in runs:
        payload, errs = run_evidence_snapshot(rd, atid_resolver=atid_resolver)
        if errs:
            key = "skipped" if payload is None else "warnings"
            report["runs"].setdefault(key, {})[str(rd)] = errs[:5]
        if payload is not None:
            dated.append((str(payload["run"].get("created_at") or ""), str(rd), payload))
    for _t, rd, payload in sorted(dated):
        _submit(payload, namespace_for_run(Path(rd), namespace, test_namespace))

    dup = sum(1 for s in report["snapshots"] if (s.get("receipt") or {}).get("duplicate"))
    report["summary"] = {
        "snapshots": len(report["snapshots"]), "duplicates": dup,
        "errors": len(report["errors"]),
        "discoveries": sum(s["counts"]["discoveries"] for s in report["snapshots"]),
        "trial_outcomes": sum(s["counts"]["trial_outcomes"] for s in report["snapshots"]),
        "recipes": sum(s["counts"]["recipes"] for s in report["snapshots"]),
    }
    return report
