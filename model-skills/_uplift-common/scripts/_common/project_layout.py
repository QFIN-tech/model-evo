from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


EXPERIMENT_ID_PATTERN = re.compile(r"^exp-[0-9]{3}-[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
FLOW_MANIFEST_NAME = "_flow_manifest.json"
FLOW_LOG_NAME = "_flow_log.jsonl"
EXPERIMENTS_ROOT_NAME = "new-models"
EXPERIMENT_MANIFEST_NAME = "_experiment_manifest.json"
EXPERIMENT_LOG_NAME = "_experiment_log.jsonl"


class ProjectLayoutError(Exception):
    def __init__(self, issue_code: str, message: str) -> None:
        super().__init__(message)
        self.issue_code = issue_code


def make_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def safe_slug(name: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(name).strip())
    return slug.strip("_") or "flow"


def validate_experiment_id(experiment_id: str) -> str:
    value = str(experiment_id or "").strip()
    if not EXPERIMENT_ID_PATTERN.fullmatch(value):
        raise ProjectLayoutError(
            "INVALID_EXPERIMENT_ID",
            "experiment_id must match ^exp-[0-9]{3}-[A-Za-z0-9][A-Za-z0-9_-]{0,63}$.",
        )
    return value


def assert_existing_run_dir(run_dir: Path | str) -> Path:
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise ProjectLayoutError("RUN_DIR_NOT_FOUND", f"run_dir does not exist: {root}")
    return root


def ensure_skill_call_dir(run_dir: Path | str, skill_name: str, flow_name: str) -> Path:
    root = assert_existing_run_dir(run_dir)
    skill_root = root / safe_slug(skill_name)
    skill_root.mkdir(parents=False, exist_ok=True)
    base = f"{make_timestamp()}_{safe_slug(flow_name)}"
    scope_dir = skill_root / base
    suffix = 1
    while scope_dir.exists():
        suffix += 1
        scope_dir = skill_root / f"{base}_{suffix:02d}"
    scope_dir.mkdir(parents=False)
    _ensure_scope_subdirs(scope_dir)
    return scope_dir


def ensure_existing_skill_call_dir(run_dir: Path | str, flow_dir: Path | str) -> Path:
    root = assert_existing_run_dir(run_dir)
    if flow_dir in (None, ""):
        raise ProjectLayoutError("FLOW_DIR_REQUIRED", "flow_dir is required for this action.")
    scope_dir = _resolve_run_path(root, flow_dir)
    try:
        scope_dir.relative_to(root)
    except ValueError as exc:
        raise ProjectLayoutError("FLOW_DIR_OUTSIDE_RUN_DIR", f"flow_dir is outside run_dir: {scope_dir}") from exc
    if not scope_dir.is_dir():
        raise ProjectLayoutError("FLOW_DIR_INVALID", f"flow_dir does not exist: {scope_dir}")
    _ensure_scope_subdirs(scope_dir)
    return scope_dir


def infer_existing_flow_dir_from_artifact(run_dir: Path | str, path: Path | str) -> Path:
    root = assert_existing_run_dir(run_dir)
    artifact = _resolve_run_path(root, path)
    scope_dir = artifact.parent.parent if artifact.parent.name == "artifacts" else artifact.parent
    return ensure_existing_skill_call_dir(root, scope_dir)


def ensure_experiment_dir(run_dir: Path | str, experiment_id: str) -> Path:
    root = assert_existing_run_dir(run_dir)
    experiment = validate_experiment_id(experiment_id)
    experiments_root = root / EXPERIMENTS_ROOT_NAME
    experiments_root.mkdir(parents=False, exist_ok=True)
    scope_dir = experiments_root / experiment
    if scope_dir.exists():
        raise ProjectLayoutError(
            "EXPERIMENT_ALREADY_EXISTS",
            f"experiment already exists: {EXPERIMENTS_ROOT_NAME}/{experiment}",
        )
    scope_dir.mkdir(parents=False)
    _ensure_scope_subdirs(scope_dir)
    return scope_dir


def experiment_manifest_path(run_dir: Path | str, experiment_id: str) -> Path:
    root = assert_existing_run_dir(run_dir)
    experiment = validate_experiment_id(experiment_id)
    return root / EXPERIMENTS_ROOT_NAME / experiment / EXPERIMENT_MANIFEST_NAME


def to_run_relative_path(run_dir: Path | str, target_path: Path | str | None) -> str | None:
    if target_path in (None, ""):
        return None
    root = Path(run_dir).expanduser().resolve()
    target = Path(str(target_path)).expanduser()
    if not target.is_absolute():
        target = root / target
    try:
        return target.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(target.resolve())


def resolve_run_path(run_dir: Path | str, value: Path | str) -> Path:
    root = Path(run_dir).expanduser().resolve(strict=False)
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def relativize_paths(value: Any, run_dir: Path | str) -> Any:
    if isinstance(value, dict):
        return {key: relativize_paths(item, run_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [relativize_paths(item, run_dir) for item in value]
    if isinstance(value, str) and _looks_like_path_string(value):
        return to_run_relative_path(run_dir, value)
    return value


def write_scope_manifest(scope_dir: Path | str, manifest: dict[str, Any]) -> Path:
    root = Path(scope_dir).expanduser().resolve()
    scope = str(manifest.get("scope") or "")
    if scope == "flow":
        path = root / FLOW_MANIFEST_NAME
    elif scope == "experiment":
        path = root / EXPERIMENT_MANIFEST_NAME
    else:
        raise ProjectLayoutError("INVALID_SCOPE", "manifest.scope must be flow or experiment.")
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def append_scope_log(scope_dir: Path | str, event: dict[str, Any]) -> Path:
    root = Path(scope_dir).expanduser().resolve()
    scope = str(event.get("scope") or "")
    if scope == "flow":
        path = root / FLOW_LOG_NAME
    elif scope == "experiment":
        path = root / EXPERIMENT_LOG_NAME
    else:
        raise ProjectLayoutError("INVALID_SCOPE", "event.scope must be flow or experiment.")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def next_action_paths(scope_dir: Path | str, action: str) -> tuple[Path, Path]:
    root = Path(scope_dir).expanduser().resolve()
    _ensure_scope_subdirs(root)
    index = _next_action_index(root)
    action_slug = safe_slug(action or "unknown")
    return (
        root / "inputs" / f"{index:04d}_{action_slug}.request.json",
        root / "results" / f"{index:04d}_{action_slug}.result.json",
    )


def write_flow_action_records(
    *,
    run_dir: Path | str,
    flow_dir: Path | str,
    skill_name: str,
    action: str,
    request_path: Path | str,
    result_path: Path | str,
    result: dict[str, Any],
) -> None:
    root = assert_existing_run_dir(run_dir)
    scope_dir = ensure_existing_skill_call_dir(root, flow_dir)
    now = _now()
    artifacts = result.get("artifacts") or []
    event = {
        "schema_version": 1,
        "event_type": "action_completed",
        "scope": "flow",
        "scope_id": scope_dir.name,
        "skill_name": skill_name,
        "action": action,
        "status": str(result.get("status") or "failed"),
        "created_at": now,
        "request_path": to_run_relative_path(root, request_path),
        "result_path": to_run_relative_path(root, result_path),
        "artifact_paths": _artifact_paths(artifacts),
        "issue_codes": _issue_codes(result),
    }
    report_path = (result.get("outputs") or {}).get("report_path")
    if report_path:
        event["report_path"] = report_path
    append_scope_log(scope_dir, event)
    existing = _read_json_if_exists(scope_dir / FLOW_MANIFEST_NAME)
    write_scope_manifest(
        scope_dir,
        {
            "schema_version": 1,
            "scope": "flow",
            "scope_id": scope_dir.name,
            "skill_name": skill_name,
            "latest_action": action,
            "status": str(result.get("status") or "failed"),
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
            "inputs": result.get("input_paths") or result.get("input_refs") or {},
            "outputs": result.get("outputs") or {},
            "artifacts": artifacts,
            "log_path": to_run_relative_path(root, scope_dir / FLOW_LOG_NAME),
        },
    )


def write_experiment_action_records(
    *,
    run_dir: Path | str,
    experiment_dir: Path | str,
    experiment_id: str,
    skill_name: str,
    action: str,
    request_path: Path | str,
    result_path: Path | str,
    result: dict[str, Any],
    extra_manifest: dict[str, Any] | None = None,
) -> None:
    root = assert_existing_run_dir(run_dir)
    scope_dir = Path(experiment_dir).expanduser().resolve()
    try:
        scope_dir.relative_to(root)
    except ValueError as exc:
        raise ProjectLayoutError("FLOW_DIR_OUTSIDE_RUN_DIR", f"experiment_dir is outside run_dir: {scope_dir}") from exc
    _ensure_scope_subdirs(scope_dir)
    now = _now()
    artifacts = result.get("artifacts") or []
    event = {
        "schema_version": 1,
        "event_type": "action_completed",
        "scope": "experiment",
        "scope_id": experiment_id,
        "experiment_id": experiment_id,
        "skill_name": skill_name,
        "action": action,
        "status": str(result.get("status") or "failed"),
        "created_at": now,
        "request_path": to_run_relative_path(root, request_path),
        "result_path": to_run_relative_path(root, result_path),
        "artifact_paths": _artifact_paths(artifacts),
        "issue_codes": _issue_codes(result),
    }
    report_path = (result.get("outputs") or {}).get("report_path")
    if report_path:
        event["report_path"] = report_path
    append_scope_log(scope_dir, event)
    existing = _read_json_if_exists(scope_dir / EXPERIMENT_MANIFEST_NAME)
    manifest = {
        "schema_version": 1,
        "scope": "experiment",
        "scope_id": experiment_id,
        "experiment_id": experiment_id,
        "skill_name": skill_name,
        "latest_action": action,
        "status": str(result.get("status") or "failed"),
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "inputs": result.get("input_paths") or result.get("input_refs") or {},
        "outputs": result.get("outputs") or {},
        "artifacts": artifacts,
        "log_path": to_run_relative_path(root, scope_dir / EXPERIMENT_LOG_NAME),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    write_scope_manifest(scope_dir, manifest)


def _ensure_scope_subdirs(scope_dir: Path) -> None:
    for child in ("inputs", "results", "artifacts"):
        (scope_dir / child).mkdir(exist_ok=True)


def _resolve_run_path(run_dir: Path, value: Path | str) -> Path:
    return resolve_run_path(run_dir, value)


def _looks_like_path_string(value: str) -> bool:
    if not value:
        return False
    normalized = value.replace("\\", "/")
    if "/" in normalized:
        return True
    suffixes = (
        ".json",
        ".jsonl",
        ".csv",
        ".md",
        ".joblib",
        ".txt",
        ".svg",
        ".parquet",
    )
    return normalized.endswith(suffixes)


def _next_action_index(scope_dir: Path) -> int:
    max_index = 0
    for child in (scope_dir / "inputs").glob("*.request.json"):
        match = re.match(r"^([0-9]{4})_", child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    for child in (scope_dir / "results").glob("*.result.json"):
        match = re.match(r"^([0-9]{4})_", child.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _artifact_paths(artifacts: Any) -> list[str]:
    if not isinstance(artifacts, list):
        return []
    return [
        str(item.get("path"))
        for item in artifacts
        if isinstance(item, dict) and item.get("path")
    ]


def _issue_codes(result: dict[str, Any]) -> list[str]:
    return [
        str(item.get("code"))
        for item in result.get("issues") or []
        if isinstance(item, dict) and item.get("code")
    ]


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
