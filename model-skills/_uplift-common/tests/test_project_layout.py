from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


COMMON = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(COMMON))

from _common.project_layout import (  # noqa: E402
    ProjectLayoutError,
    append_scope_log,
    ensure_existing_skill_call_dir,
    ensure_experiment_dir,
    ensure_skill_call_dir,
    next_action_paths,
    resolve_run_path,
    to_run_relative_path,
    validate_experiment_id,
    write_flow_action_records,
    write_scope_manifest,
)


def test_run_dir_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ProjectLayoutError) as exc:
        ensure_skill_call_dir(tmp_path / "missing", "uplift-model-task-spec", "task_spec")
    assert exc.value.issue_code == "RUN_DIR_NOT_FOUND"
    assert not (tmp_path / "missing").exists()


def test_flow_and_experiment_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    flow_dir = ensure_skill_call_dir(run_dir, "uplift-model-task-spec", "task spec")
    assert flow_dir.relative_to(run_dir).parts[0] == "uplift-model-task-spec"
    assert (flow_dir / "inputs").is_dir()
    assert (flow_dir / "results").is_dir()
    assert (flow_dir / "artifacts").is_dir()

    experiment_dir = ensure_experiment_dir(run_dir, "exp-001-s_learner_baseline")
    assert experiment_dir == run_dir / "new-models" / "exp-001-s_learner_baseline"
    assert (experiment_dir / "artifacts").is_dir()


def test_experiment_id_validation_and_no_overwrite(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ProjectLayoutError) as exc:
        validate_experiment_id("001-bad")
    assert exc.value.issue_code == "INVALID_EXPERIMENT_ID"

    ensure_experiment_dir(run_dir, "exp-001-ok")
    with pytest.raises(ProjectLayoutError) as exc:
        ensure_experiment_dir(run_dir, "exp-001-ok")
    assert exc.value.issue_code == "EXPERIMENT_ALREADY_EXISTS"


def test_manifest_log_and_relative_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    flow_dir = ensure_skill_call_dir(run_dir, "uplift-model-task-spec", "task_spec")
    result_path = flow_dir / "results" / "draft_task_config.result.json"
    artifact_path = flow_dir / "artifacts" / "task_config.draft.v1.json"

    assert to_run_relative_path(run_dir, result_path) == (
        f"{flow_dir.relative_to(run_dir).as_posix()}/results/draft_task_config.result.json"
    )

    manifest_path = write_scope_manifest(
        flow_dir,
        {
            "schema_version": 1,
            "scope": "flow",
            "scope_id": flow_dir.name,
            "skill_name": "uplift-model-task-spec",
            "latest_action": "draft_task_config",
            "status": "needs_confirmation",
            "created_at": "2026-07-03T00:00:00Z",
            "updated_at": "2026-07-03T00:00:00Z",
            "inputs": {},
            "outputs": {"result_path": to_run_relative_path(run_dir, result_path)},
            "artifacts": [{"path": to_run_relative_path(run_dir, artifact_path)}],
            "log_path": to_run_relative_path(run_dir, flow_dir / "_flow_log.jsonl"),
        },
    )
    log_path = append_scope_log(
        flow_dir,
        {
            "schema_version": 1,
            "event_type": "action_completed",
            "scope": "flow",
            "scope_id": flow_dir.name,
            "skill_name": "uplift-model-task-spec",
            "action": "draft_task_config",
            "status": "needs_confirmation",
            "created_at": "2026-07-03T00:00:00Z",
            "request_path": to_run_relative_path(run_dir, flow_dir / "inputs" / "request.json"),
            "result_path": to_run_relative_path(run_dir, result_path),
            "artifact_paths": [to_run_relative_path(run_dir, artifact_path)],
            "issue_codes": [],
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert manifest["outputs"]["result_path"].startswith("uplift-model-task-spec/")
    assert event["result_path"].startswith("uplift-model-task-spec/")


def test_resolve_run_path_resolves_posix_relative_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    resolved = resolve_run_path(run_dir, "uplift-model-reporting/artifacts/report_facts.v1.json")

    assert resolved == (run_dir / "uplift-model-reporting" / "artifacts" / "report_facts.v1.json").resolve()


def test_resolve_run_path_keeps_current_os_absolute_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    absolute = (tmp_path / "outside" / "artifact.json").resolve(strict=False)

    assert resolve_run_path(run_dir, absolute) == absolute


def test_resolve_run_path_does_not_require_existing_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    missing = "upstream/results/missing.result.json"

    resolved = resolve_run_path(run_dir, missing)

    assert resolved == (run_dir / "upstream" / "results" / "missing.result.json").resolve(strict=False)
    assert not resolved.exists()


def test_existing_flow_dir_validation_codes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(ProjectLayoutError) as exc:
        ensure_existing_skill_call_dir(run_dir, "")
    assert exc.value.issue_code == "FLOW_DIR_REQUIRED"

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ProjectLayoutError) as exc:
        ensure_existing_skill_call_dir(run_dir, outside)
    assert exc.value.issue_code == "FLOW_DIR_OUTSIDE_RUN_DIR"

    with pytest.raises(ProjectLayoutError) as exc:
        ensure_existing_skill_call_dir(run_dir, "uplift-model-task-spec/missing")
    assert exc.value.issue_code == "FLOW_DIR_INVALID"


def test_numbered_action_records_do_not_overwrite(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    flow_dir = ensure_skill_call_dir(run_dir, "uplift-model-task-spec", "task_spec")

    request_1, result_1 = next_action_paths(flow_dir, "draft_task_config")
    request_1.write_text("{}", encoding="utf-8")
    result_payload_1 = {
        "status": "needs_confirmation",
        "input_paths": {},
        "outputs": {"flow_dir": to_run_relative_path(run_dir, flow_dir)},
        "artifacts": [],
        "issues": [],
    }
    result_1.write_text(json.dumps(result_payload_1), encoding="utf-8")
    write_flow_action_records(
        run_dir=run_dir,
        flow_dir=flow_dir,
        skill_name="uplift-model-task-spec",
        action="draft_task_config",
        request_path=request_1,
        result_path=result_1,
        result=result_payload_1,
    )

    request_2, result_2 = next_action_paths(flow_dir, "confirm_artifact")
    assert request_1.name.startswith("0001_")
    assert result_1.name.startswith("0001_")
    assert request_2.name.startswith("0002_")
    assert result_2.name.startswith("0002_")

    request_2.write_text("{}", encoding="utf-8")
    result_payload_2 = {
        "status": "success",
        "input_paths": {},
        "outputs": {
            "flow_dir": to_run_relative_path(run_dir, flow_dir),
            "result_path": to_run_relative_path(run_dir, result_2),
        },
        "artifacts": [],
        "issues": [],
    }
    result_2.write_text(json.dumps(result_payload_2), encoding="utf-8")
    write_flow_action_records(
        run_dir=run_dir,
        flow_dir=flow_dir,
        skill_name="uplift-model-task-spec",
        action="confirm_artifact",
        request_path=request_2,
        result_path=result_2,
        result=result_payload_2,
    )
    events = (flow_dir / "_flow_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    manifest = json.loads((flow_dir / "_flow_manifest.json").read_text(encoding="utf-8"))
    assert len(events) == 2
    assert json.loads(events[-1])["result_path"].endswith("0002_confirm_artifact.result.json")
    assert manifest["latest_action"] == "confirm_artifact"
