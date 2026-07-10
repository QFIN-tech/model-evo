from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run.py"
DEFAULT_THRESHOLDS = {
    "missing_rate_warning": 0.5,
    "missing_rate_exclude": 0.8,
    "psi_warning": 0.1,
    "psi_exclude": 0.25,
    "near_constant_mode_ratio": 0.99,
    "high_concentration_mode_ratio": 0.95,
}


def _run(payload: dict, run_dir: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--input", "-", "--output-dir", str(run_dir)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prepare_inputs(tmp_path: Path, *, date_values: list[object] | None = None, time_column: str | None = "month_key") -> dict:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifacts = run_dir / "artifacts"
    data_dir = run_dir / "data"
    month_column = time_column or "month_key"
    dates = date_values or ["2026-01", "2026-01", "2026-02", "2026-02"]
    train_rows = [
        {"id": 1, "treatment": 1, "outcome": 1, month_column: dates[0], "feature_good": 0.1, "feature_sparse": "", "segment": "A"},
        {"id": 2, "treatment": 0, "outcome": 0, month_column: dates[1], "feature_good": 0.2, "feature_sparse": 1, "segment": "B"},
        {"id": 3, "treatment": 1, "outcome": 0, month_column: dates[2], "feature_good": 0.3, "feature_sparse": 1, "segment": "A"},
        {"id": 4, "treatment": 0, "outcome": 1, month_column: dates[3], "feature_good": 0.4, "feature_sparse": "", "segment": "B"},
    ]
    test_rows = [
        {"id": 5, "treatment": 1, "outcome": 1, month_column: dates[0], "feature_good": 0.15, "feature_sparse": "", "segment": "A"},
        {"id": 6, "treatment": 0, "outcome": 0, month_column: dates[-1], "feature_good": 0.45, "feature_sparse": 1, "segment": "B"},
    ]
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    _write_csv(train_path, train_rows)
    _write_csv(test_path, test_rows)

    task_path = artifacts / "task_config.confirmed.v1.json"
    task_payload = {
        "exclude_columns": ["id", "treatment", "outcome", month_column],
        "report_preferences": {"language": "en"},
    }
    if time_column is not None:
        task_payload["time_column"] = time_column
    _write_json(
        task_path,
        {
            "artifact_kind": "task_config",
            "artifact_status": "confirmed",
            "payload": task_payload,
        },
    )
    spec_path = artifacts / "modeling_sample_spec.confirmed.v1.json"
    _write_json(
        spec_path,
        {
            "artifact_kind": "modeling_sample_spec",
            "artifact_status": "confirmed",
            "input_paths": {"task_config_path": str(task_path)},
            "payload": {
                "datasets": {"train": str(train_path), "test": str(test_path)},
                "columns": {"treatment": "treatment", "outcome": "outcome", "outcome_type": "binary"},
                "feature_columns": {"forced_exclude": []},
            },
        },
    )
    return {
        "run_dir": run_dir,
        "task_config_path": "artifacts/task_config.confirmed.v1.json",
        "modeling_sample_spec_path": "artifacts/modeling_sample_spec.confirmed.v1.json",
    }


def _diagnostics_payload(paths: dict, diagnostics: list[str] | None = None, **extra: object) -> dict:
    payload = {
        "task_config_path": paths["task_config_path"],
        "modeling_sample_spec_path": paths["modeling_sample_spec_path"],
        "analysis_scope": {
            "diagnostics": diagnostics or ["basic_quality", "split_psi"],
            "confirmed": True,
        },
    }
    payload.update(extra)
    return {"action": "diagnostics_only", "payload": payload}


def _successful_diagnostics(paths: dict) -> dict:
    return _run(_diagnostics_payload(paths), paths["run_dir"])


def _successful_selection(paths: dict, diagnostics_stdout: dict) -> dict:
    return _run(
        {
            "action": "selection_only",
            "payload": {
                "source_result_path": diagnostics_stdout["outputs"]["result_path"],
                "new_thresholds": DEFAULT_THRESHOLDS,
                "selection_scope": {"confirmed": True, "rule_source": "default"},
            },
        },
        paths["run_dir"],
    )


def test_diagnostics_only_requires_confirmed_analysis_scope(tmp_path: Path) -> None:
    paths = _prepare_inputs(tmp_path)

    stdout = _run(
        {
            "action": "diagnostics_only",
            "payload": {
                "task_config_path": paths["task_config_path"],
                "modeling_sample_spec_path": paths["modeling_sample_spec_path"],
                "analysis_scope": {"diagnostics": ["basic_quality"]},
            },
        },
        paths["run_dir"],
    )

    assert stdout["status"] == "needs_confirmation"
    assert "analysis_scope.confirmed=true" in stdout["summary"]


def test_diagnostics_only_runs_after_analysis_scope_confirmation(tmp_path: Path) -> None:
    paths = _prepare_inputs(tmp_path)

    stdout = _successful_diagnostics(paths)

    assert stdout["status"] == "success"
    outputs = stdout["outputs"]
    assert outputs["feature_basic_quality_path"]
    assert outputs["feature_psi_split_path"]
    assert outputs["feature_selection_recommendation_path"] is None


@pytest.mark.parametrize("action", ["diagnostics_and_selection", "monthly_psi", "uplift_bivar"])
def test_removed_actions_are_not_supported(tmp_path: Path, action: str) -> None:
    paths = _prepare_inputs(tmp_path)

    stdout = _run({"action": action, "payload": {}}, paths["run_dir"])

    assert stdout["status"] == "failed"
    assert stdout["outputs"]["unsupported_reason"] == f"Unsupported action: {action}"


def test_monthly_psi_requires_explicit_date_column_or_task_time_column(tmp_path: Path) -> None:
    paths = _prepare_inputs(tmp_path, time_column=None)

    stdout = _run(_diagnostics_payload(paths, diagnostics=["monthly_psi"]), paths["run_dir"])

    assert stdout["status"] == "needs_input"
    assert "monthly_psi requires date_column" in stdout["summary"]


@pytest.mark.parametrize(
    "date_values",
    [
        ["2026-01", "2026-01", "2026-02", "2026-02"],
        ["202601", "202601", "202602", "202602"],
        ["2026-01-01", "2026-01-01", "2026-02-01", "2026-02-01"],
        ["2026-01-01 00:00:00", "2026-01-01 12:00:00", "2026-02-01 00:00:00", "2026-02-01 08:30:00"],
    ],
)
def test_monthly_psi_accepts_month_level_date_column_from_payload(tmp_path: Path, date_values: list[object]) -> None:
    paths = _prepare_inputs(tmp_path, date_values=date_values, time_column=None)

    stdout = _run(
        _diagnostics_payload(paths, diagnostics=["monthly_psi"], date_column="month_key"),
        paths["run_dir"],
    )

    assert stdout["status"] == "success"
    assert stdout["outputs"]["feature_psi_monthly_path"]


def test_monthly_psi_accepts_task_config_time_column(tmp_path: Path) -> None:
    paths = _prepare_inputs(tmp_path, date_values=["202601", "202601", "202602", "202602"], time_column="month_key")

    stdout = _run(_diagnostics_payload(paths, diagnostics=["monthly_psi"]), paths["run_dir"])

    assert stdout["status"] == "success"
    assert stdout["outputs"]["feature_psi_monthly_path"]


@pytest.mark.parametrize(
    "date_values",
    [
        ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"],
        ["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-15"],
        ["2026-01", "", "2026-02", "2026-02"],
        ["2026-01", "not-a-date", "2026-02", "2026-02"],
    ],
)
def test_monthly_psi_rejects_daily_missing_or_unparseable_values(tmp_path: Path, date_values: list[object]) -> None:
    paths = _prepare_inputs(tmp_path, date_values=date_values, time_column=None)

    stdout = _run(
        _diagnostics_payload(paths, diagnostics=["monthly_psi"], date_column="month_key"),
        paths["run_dir"],
    )

    assert stdout["status"] == "needs_input"
    assert "date_column" in stdout["summary"]


def test_selection_only_requires_confirmed_selection_scope(tmp_path: Path) -> None:
    paths = _prepare_inputs(tmp_path)
    diagnostics = _successful_diagnostics(paths)

    stdout = _run(
        {
            "action": "selection_only",
            "payload": {
                "source_result_path": diagnostics["outputs"]["result_path"],
                "new_thresholds": DEFAULT_THRESHOLDS,
            },
        },
        paths["run_dir"],
    )

    assert stdout["status"] == "needs_confirmation"
    assert "selection_scope.confirmed=true" in stdout["summary"]


def test_selection_only_generates_recommendation_with_confirmed_default_rules(tmp_path: Path) -> None:
    paths = _prepare_inputs(tmp_path)
    diagnostics = _successful_diagnostics(paths)

    stdout = _successful_selection(paths, diagnostics)

    assert stdout["status"] == "success"
    assert stdout["outputs"]["feature_selection_recommendation_path"]
    assert stdout["outputs"]["feature_plan_path"] is None
    next_step_inputs = {step["action"]: step["inputs"] for step in stdout["next_steps"]}
    assert next_step_inputs["accept_recommendation"]["flow_dir"] == stdout["outputs"]["flow_dir"]
    assert next_step_inputs["modify_recommendation"]["flow_dir"] == stdout["outputs"]["flow_dir"]


def test_selection_only_rejects_unknown_rule_source(tmp_path: Path) -> None:
    paths = _prepare_inputs(tmp_path)

    stdout = _run(
        {
            "action": "selection_only",
            "payload": {
                "source_result_path": "missing.json",
                "new_thresholds": DEFAULT_THRESHOLDS,
                "selection_scope": {"confirmed": True, "rule_source": "legacy"},
            },
        },
        paths["run_dir"],
    )

    assert stdout["status"] == "needs_input"
    assert "selection_scope.rule_source" in stdout["summary"]


def test_homogeneity_fields_are_not_written_to_contract_outputs(tmp_path: Path) -> None:
    paths = _prepare_inputs(tmp_path)

    diagnostics = _run(
        _diagnostics_payload(
            paths,
            sample_homogeneity_result_path="artifacts/homogeneity.json",
            accepted_homogeneity_risk=True,
            skip_homogeneity_check=True,
            explicit_skip_homogeneity_check=True,
        ),
        paths["run_dir"],
    )

    assert diagnostics["status"] == "success"
    flow_dir = paths["run_dir"] / diagnostics["outputs"]["flow_dir"]
    result = json.loads((paths["run_dir"] / diagnostics["outputs"]["result_path"]).read_text(encoding="utf-8"))
    manifest = json.loads((flow_dir / "_flow_manifest.json").read_text(encoding="utf-8"))
    report = (paths["run_dir"] / diagnostics["outputs"]["report_path"]).read_text(encoding="utf-8")
    serialized = json.dumps({"stdout": diagnostics, "result": result, "manifest": manifest})

    assert "sample_homogeneity_result_path" not in serialized
    assert "accepted_homogeneity_risk" not in serialized
    assert "skip_homogeneity_check" not in serialized
    assert "explicit_skip_homogeneity_check" not in serialized
    assert "sample_homogeneity_result_path" not in report
    assert "accepted_homogeneity_risk" not in report
    assert "skip_homogeneity_check" not in report
    assert "explicit_skip_homogeneity_check" not in report


def test_feature_plan_actions_keep_existing_lifecycle(tmp_path: Path) -> None:
    paths = _prepare_inputs(tmp_path)
    diagnostics = _successful_diagnostics(paths)
    selection = _successful_selection(paths, diagnostics)
    recommendation_path = selection["outputs"]["feature_selection_recommendation_path"]

    accepted = _run(
        {
            "action": "accept_recommendation",
            "payload": {
                "flow_dir": selection["outputs"]["flow_dir"],
                "recommendation_path": recommendation_path,
            },
        },
        paths["run_dir"],
    )
    assert accepted["status"] == "success"
    accepted_plan = json.loads((paths["run_dir"] / accepted["outputs"]["feature_plan_path"]).read_text(encoding="utf-8"))
    assert accepted_plan["artifact_status"] == "confirmed"

    modified = _run(
        {
            "action": "modify_recommendation",
            "payload": {
                "flow_dir": selection["outputs"]["flow_dir"],
                "recommendation_path": recommendation_path,
                "include_features": ["feature_good"],
                "exclude_features": ["feature_sparse"],
            },
        },
        paths["run_dir"],
    )
    assert modified["status"] == "needs_confirmation"
    assert modified["next_steps"][0]["inputs"]["flow_dir"] == selection["outputs"]["flow_dir"]
    modified_plan = json.loads((paths["run_dir"] / modified["outputs"]["feature_plan_path"]).read_text(encoding="utf-8"))
    assert modified_plan["artifact_status"] == "draft"

    manual = _run(
        {
            "action": "manual_feature_plan",
            "payload": {
                "task_config_path": paths["task_config_path"],
                "modeling_sample_spec_path": paths["modeling_sample_spec_path"],
                "feature_list": ["feature_good", "segment"],
            },
        },
        paths["run_dir"],
    )
    assert manual["status"] == "needs_confirmation"
    assert manual["next_steps"][0]["inputs"]["flow_dir"] == manual["outputs"]["flow_dir"]
    manual_plan = json.loads((paths["run_dir"] / manual["outputs"]["feature_plan_path"]).read_text(encoding="utf-8"))
    assert manual_plan["artifact_status"] == "draft"

    confirmed = _run(
        {
            "action": "confirm_artifact",
            "payload": {
                "flow_dir": manual["outputs"]["flow_dir"],
                "source_draft_path": manual["outputs"]["feature_plan_path"],
                "artifact_kind": "feature_plan",
            },
        },
        paths["run_dir"],
    )
    assert confirmed["status"] == "success"
    confirmed_plan = json.loads((paths["run_dir"] / confirmed["outputs"]["feature_plan_path"]).read_text(encoding="utf-8"))
    assert confirmed_plan["artifact_status"] == "confirmed"


def test_report_keeps_diagnostics_and_refreshes_confirmed_feature_plan(tmp_path: Path) -> None:
    paths = _prepare_inputs(tmp_path)
    diagnostics = _run(
        _diagnostics_payload(paths, ["basic_quality", "split_psi", "uplift_bivar"]),
        paths["run_dir"],
    )
    selection = _successful_selection(paths, diagnostics)
    selection_report = (paths["run_dir"] / selection["outputs"]["report_path"]).read_text(
        encoding="utf-8"
    )

    assert "| 0 | 2 | 66.67% |" in selection_report
    assert "- Max missing rate: `50.00%`" in selection_report
    assert "- Max PSI: `N/A`" not in selection_report
    assert "- Uplift Bivar summary: `{}`" not in selection_report
    assert "'feature_count': 3" in selection_report

    modified = _run(
        {
            "action": "modify_recommendation",
            "payload": {
                "flow_dir": selection["outputs"]["flow_dir"],
                "recommendation_path": selection["outputs"]["feature_selection_recommendation_path"],
                "include_features": ["feature_good", "segment"],
                "exclude_features": ["feature_sparse"],
            },
        },
        paths["run_dir"],
    )
    confirmed = _run(
        {
            "action": "confirm_artifact",
            "payload": {
                "flow_dir": modified["outputs"]["flow_dir"],
                "source_draft_path": modified["outputs"]["feature_plan_path"],
                "artifact_kind": "feature_plan",
            },
        },
        paths["run_dir"],
    )
    confirmed_report = (paths["run_dir"] / confirmed["outputs"]["report_path"]).read_text(
        encoding="utf-8"
    )

    assert "- Feature plan status: `confirmed`" in confirmed_report
    assert "- Modeling feature count: `2`" in confirmed_report
    assert "- Feature preview: `segment, feature_good`" in confirmed_report
    assert "| 0 | 2 | 66.67% |" in confirmed_report
    assert "- Max PSI: `N/A`" not in confirmed_report
    assert "'feature_count': 3" in confirmed_report
