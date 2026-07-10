from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run.py"
AUUC_VERSION = {"package": "scikit-uplift", "version": "0.5.1"}


def _run(payload: dict, run_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), "--input", "-", "--output-dir", str(run_dir)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )


def _candidate(run_dir: Path, name: str, metric: float) -> str:
    result_path = run_dir / "new-models" / name / "results" / "train.result.json"
    result_path.parent.mkdir(parents=True)
    artifacts = run_dir / "new-models" / name / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    raw_curve_path = artifacts / "test_auuc_curve_raw.v1.csv"
    normalized_curve_path = artifacts / "test_auuc_curve_normalized.v1.csv"
    uplift_bins_path = artifacts / "uplift_bins.v1.csv"
    raw_curve_path.write_text("population_index,cumulative_gain\n0,0\n1,0.1\n", encoding="utf-8")
    normalized_curve_path.write_text("population_index,cumulative_gain\n0,0\n1,0.1\n", encoding="utf-8")
    uplift_bins_path.write_text("split,bin_id,row_count\n", encoding="utf-8")
    raw_curve_rel = raw_curve_path.relative_to(run_dir).as_posix()
    normalized_curve_rel = normalized_curve_path.relative_to(run_dir).as_posix()
    uplift_bins_rel = uplift_bins_path.relative_to(run_dir).as_posix()
    candidate_payload = {
        "artifact_kind": "model_candidate",
        "artifact_version": 1,
        "candidate_id": name,
        "producer_kind": "modeling",
        "model_spec": {"model_type": "s_learner"},
        "model_artifact_path": f"new-models/{name}/artifacts/model.joblib",
        "model_metadata_path": f"new-models/{name}/artifacts/model_metadata.v1.json",
    }
    candidate_path = artifacts / "model_candidate.v1.json"
    candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")
    payload = {
        "skill_name": "uplift-model-s-learner-modeling",
        "run_id": name,
        "phase": "train",
        "status": "success",
        "input_paths": {
            "task_config_path": "artifacts/task_config.confirmed.v1.json",
            "modeling_sample_spec_path": "artifacts/modeling_sample_spec.confirmed.v1.json",
            "feature_plan_path": "artifacts/feature_plan.confirmed.v1.json",
        },
        "outputs": {
            "model_candidate_path": candidate_path.relative_to(run_dir).as_posix(),
            "learner": "s_learner",
            "split_evaluations": {
                "test": {
                    "dataset_role": "test",
                    "population_fingerprint": "sha256:same-population",
                    "metric_name": "auuc_raw",
                    "metric_value": metric,
                    "metric_direction": "higher_is_better",
                    "metric_method": "sklift.metrics",
                    "metric_version": "0.5.1",
                    "auuc_version": AUUC_VERSION,
                    "support": {"row_count": 10, "treatment_count": 5, "control_count": 5},
                    "raw_curve_path": raw_curve_rel,
                    "normalized_curve_path": normalized_curve_rel,
                    "uplift_bins_path": uplift_bins_rel,
                }
            },
        },
        "issues": [],
        "artifacts": [],
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = result_path.parents[1] / "_experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scope": "experiment",
                "experiment_id": name,
                "outputs": {"modeling_result_path": result_path.relative_to(run_dir).as_posix()},
            }
        ),
        encoding="utf-8",
    )
    return result_path.relative_to(run_dir).as_posix()


def _tuning_candidate(run_dir: Path, name: str, base_result_path: str, metric: float) -> str:
    result_path = run_dir / "new-models" / name / "results" / "execute.result.json"
    result_path.parent.mkdir(parents=True)
    artifacts = run_dir / "new-models" / name / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    raw_curve_path = artifacts / "winner_test_auuc_curve_raw.v1.csv"
    normalized_curve_path = artifacts / "winner_test_auuc_curve_normalized.v1.csv"
    raw_curve_path.write_text("population_index,cumulative_gain\n0,0\n1,0.2\n", encoding="utf-8")
    normalized_curve_path.write_text("population_index,cumulative_gain\n0,0\n1,0.2\n", encoding="utf-8")
    candidate_payload = {
        "artifact_kind": "model_candidate",
        "artifact_version": 1,
        "candidate_id": f"{name}:trial_001",
        "producer_kind": "tuning",
        "model_spec": {"model_type": "s_learner"},
        "model_artifact_path": f"new-models/{name}/artifacts/tuned_model.joblib",
        "model_metadata_path": f"new-models/{name}/artifacts/model_metadata.v1.json",
        "base_candidate_result_path": base_result_path,
        "lineage": {
            "task_config_path": "artifacts/task_config.confirmed.v1.json",
            "modeling_sample_spec_path": "artifacts/modeling_sample_spec.confirmed.v1.json",
            "feature_plan_path": "artifacts/feature_plan.confirmed.v1.json",
            "modeling_result_path": base_result_path,
        },
    }
    candidate_path = artifacts / "model_candidate.v1.json"
    candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")
    payload = {
        "skill_name": "uplift-model-tuning",
        "run_id": name,
        "phase": "execute",
        "status": "success",
        "input_paths": {
            "task_config_path": "artifacts/task_config.confirmed.v1.json",
            "modeling_sample_spec_path": "artifacts/modeling_sample_spec.confirmed.v1.json",
            "feature_plan_path": "artifacts/feature_plan.confirmed.v1.json",
            "modeling_result_path": base_result_path,
        },
        "outputs": {
            "experiment_id": name,
            "source_experiment_id": "exp-001-s_learner_baseline",
            "model_candidate_path": candidate_path.relative_to(run_dir).as_posix(),
            "learner": "s_learner",
            "tuning_summary": {"winner_reason": "improved"},
            "split_evaluations": {
                "test": {
                    "dataset_role": "test",
                    "population_fingerprint": "sha256:same-population",
                    "metric_name": "auuc_raw",
                    "metric_value": metric,
                    "metric_direction": "higher_is_better",
                    "metric_method": "sklift.metrics",
                    "metric_version": "0.5.1",
                    "auuc_version": AUUC_VERSION,
                    "support": {"row_count": 10, "treatment_count": 5, "control_count": 5},
                    "raw_curve_path": raw_curve_path.relative_to(run_dir).as_posix(),
                    "normalized_curve_path": normalized_curve_path.relative_to(run_dir).as_posix(),
                }
            },
        },
        "issues": [],
        "artifacts": [],
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path = result_path.parents[1] / "_experiment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "scope": "experiment",
                "experiment_id": name,
                "outputs": {"winner_result_path": result_path.relative_to(run_dir).as_posix()},
            }
        ),
        encoding="utf-8",
    )
    return result_path.relative_to(run_dir).as_posix()


def test_draft_comparison_plan_from_candidate_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    (artifacts / "task_config.confirmed.v1.json").write_text(
        json.dumps({"artifact_kind": "task_config", "artifact_status": "confirmed"}),
        encoding="utf-8",
    )
    first = _candidate(run_dir, "exp-001-s_learner_baseline", 0.11)
    second = _candidate(run_dir, "exp-002-s_learner_tuned", 0.13)

    completed = _run(
        {
            "action": "draft_comparison_plan",
            "payload": {
                "task_config_path": "artifacts/task_config.confirmed.v1.json",
                "candidate_result_paths": [first, second],
            },
        },
        run_dir,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["status"] == "needs_confirmation"
    outputs = stdout["outputs"]
    assert outputs["comparison_plan_path"].startswith("uplift-model-result-comparison/")
    assert outputs["recommended_comparison_mode"] == "strict_comparison"

    flow_dir = run_dir / outputs["flow_dir"]
    plan = json.loads((run_dir / outputs["comparison_plan_path"]).read_text(encoding="utf-8"))
    assert plan["requires_user_confirmation"] is True
    assert plan["checks"]["task_config"]["is_consistent"] is True
    assert (flow_dir / "_flow_manifest.json").is_file()
    assert (flow_dir / "_flow_log.jsonl").is_file()


def test_draft_comparison_plan_from_experiment_ids_does_not_scan_latest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    (artifacts / "task_config.confirmed.v1.json").write_text(
        json.dumps({"artifact_kind": "task_config", "artifact_status": "confirmed"}),
        encoding="utf-8",
    )
    _candidate(run_dir, "exp-001-s_learner_baseline", 0.11)
    _candidate(run_dir, "exp-002-s_learner_tuned", 0.13)
    latest = run_dir / "new-models" / "latest"
    latest.mkdir()

    completed = _run(
        {
            "action": "draft_comparison_plan",
            "payload": {
                "task_config_path": "artifacts/task_config.confirmed.v1.json",
                "experiment_ids": ["exp-001-s_learner_baseline", "exp-002-s_learner_tuned"],
            },
        },
        run_dir,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["status"] == "needs_confirmation"
    plan = json.loads((run_dir / stdout["outputs"]["comparison_plan_path"]).read_text(encoding="utf-8"))
    assert [item["experiment_id"] for item in plan["candidate_sources"]] == [
        "exp-001-s_learner_baseline",
        "exp-002-s_learner_tuned",
    ]


def test_compare_uses_confirmed_comparison_plan_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    (artifacts / "task_config.confirmed.v1.json").write_text(
        json.dumps({"artifact_kind": "task_config", "artifact_status": "confirmed"}),
        encoding="utf-8",
    )
    _candidate(run_dir, "exp-001-s_learner_baseline", 0.11)
    _candidate(run_dir, "exp-002-s_learner_tuned", 0.13)

    draft = _run(
        {
            "action": "draft_comparison_plan",
            "payload": {
                "task_config_path": "artifacts/task_config.confirmed.v1.json",
                "experiment_ids": ["exp-001-s_learner_baseline", "exp-002-s_learner_tuned"],
            },
        },
        run_dir,
    )
    draft_stdout = json.loads(draft.stdout)

    completed = _run(
        {
            "action": "compare",
            "payload": {
                "flow_dir": draft_stdout["outputs"]["flow_dir"],
                "comparison_plan_path": draft_stdout["outputs"]["comparison_plan_path"],
            },
        },
        run_dir,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["status"] == "success"
    outputs = stdout["outputs"]
    assert outputs["plan_confirmed"] is True
    assert outputs["comparison_plan_path"] == draft_stdout["outputs"]["comparison_plan_path"]
    assert outputs["recommended_candidate_result_path"].endswith("exp-002-s_learner_tuned/results/train.result.json")
    flow_dir = run_dir / outputs["flow_dir"]
    result_files = sorted((flow_dir / "results").glob("*.result.json"))
    assert [path.name for path in result_files] == [
        "0001_draft_comparison_plan.result.json",
        "0002_compare.result.json",
    ]
    summary = json.loads((run_dir / outputs["model_comparison_summary_path"]).read_text(encoding="utf-8"))
    assert summary["plan_confirmed"] is True


def test_compare_resolves_tuning_baseline_path_against_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    (artifacts / "task_config.confirmed.v1.json").write_text(
        json.dumps({"artifact_kind": "task_config", "artifact_status": "confirmed"}),
        encoding="utf-8",
    )
    baseline = _candidate(run_dir, "exp-001-s_learner_baseline", 0.11)
    _candidate(run_dir, "exp-002-t_learner_baseline", 0.10)
    _tuning_candidate(run_dir, "exp-003-s_learner_tuned", baseline, 0.13)

    draft = _run(
        {
            "action": "draft_comparison_plan",
            "payload": {
                "task_config_path": "artifacts/task_config.confirmed.v1.json",
                "experiment_ids": [
                    "exp-001-s_learner_baseline",
                    "exp-002-t_learner_baseline",
                    "exp-003-s_learner_tuned",
                ],
            },
        },
        run_dir,
    )
    draft_stdout = json.loads(draft.stdout)

    completed = _run(
        {
            "action": "compare",
            "payload": {
                "flow_dir": draft_stdout["outputs"]["flow_dir"],
                "comparison_plan_path": draft_stdout["outputs"]["comparison_plan_path"],
            },
        },
        run_dir,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["status"] == "success"
    issue_codes = {issue["code"] for issue in stdout["issues"]}
    assert "BASELINE_CANDIDATE_NOT_INCLUDED" not in issue_codes
    assert stdout["outputs"]["recommended_candidate_result_path"].endswith("exp-003-s_learner_tuned/results/execute.result.json")
    assert stdout["next_steps"][0]["skill"] == "uplift-model-reporting"
    reporting_support = stdout["next_steps"][0]["inputs"]["supporting_paths"]
    assert reporting_support["sample_preparation_result_path"] is None
    assert reporting_support["sample_homogeneity_result_path"] is None
    assert reporting_support["feature_quality_result_path"] is None
    assert reporting_support["comparison_result_path"] is not None
    assert reporting_support["candidate_result_paths"]


def test_compare_candidates_legacy_action_still_accepted(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    (artifacts / "task_config.confirmed.v1.json").write_text(
        json.dumps({"artifact_kind": "task_config", "artifact_status": "confirmed"}),
        encoding="utf-8",
    )
    first = _candidate(run_dir, "exp-001-s_learner_baseline", 0.11)
    second = _candidate(run_dir, "exp-002-s_learner_tuned", 0.13)

    completed = _run(
        {
            "action": "compare_candidates",
            "payload": {
                "task_config_path": "artifacts/task_config.confirmed.v1.json",
                "candidate_result_paths": [first, second],
            },
        },
        run_dir,
    )

    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["status"] == "success"
    assert stdout["outputs"]["plan_confirmed"] is False
    assert stdout["outputs"]["model_comparison_summary_path"].startswith("uplift-model-result-comparison/")
