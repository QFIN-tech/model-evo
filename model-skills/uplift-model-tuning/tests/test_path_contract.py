from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

spec = importlib.util.spec_from_file_location("uplift_model_tuning_run", SCRIPTS_DIR / "run.py")
assert spec is not None and spec.loader is not None
tuning_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuning_run)


def test_candidate_payload_resolves_run_relative_model_candidate_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "run"
    candidate_path = (
        output_dir
        / "new-models"
        / "exp-001-t_learner_baseline"
        / "artifacts"
        / "model_candidate.v1.json"
    )
    candidate_path.parent.mkdir(parents=True)
    candidate = {
        "artifact_kind": "model_candidate",
        "artifact_version": 1,
        "candidate_id": "exp-001-t_learner_baseline",
        "model_spec": {
            "model_type": "t_learner",
            "base_estimators": {
                "treatment_outcome": "lightgbm",
                "control_outcome": "lightgbm",
            },
        },
    }
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    unrelated_cwd = tmp_path / "cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    outputs = {
        "model_candidate_path": candidate_path.relative_to(output_dir).as_posix(),
    }

    assert tuning_run._candidate_payload(output_dir, outputs) == candidate


def test_load_tuning_inputs_resolves_baseline_metadata_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_dir = tmp_path / "run"
    artifacts_dir = output_dir / "artifacts"
    experiment_dir = output_dir / "new-models" / "exp-001-t_learner_baseline"
    experiment_artifacts = experiment_dir / "artifacts"
    experiment_results = experiment_dir / "results"
    artifacts_dir.mkdir(parents=True)
    experiment_artifacts.mkdir(parents=True)
    experiment_results.mkdir(parents=True)

    task_path = artifacts_dir / "task_config.confirmed.v1.json"
    sample_path = artifacts_dir / "modeling_sample_spec.confirmed.v1.json"
    feature_path = artifacts_dir / "feature_plan.confirmed.v1.json"
    modeling_path = experiment_results / "train.result.json"
    candidate_path = experiment_artifacts / "model_candidate.v1.json"
    metadata_path = experiment_artifacts / "model_metadata.v1.json"

    task_rel = task_path.relative_to(output_dir).as_posix()
    sample_rel = sample_path.relative_to(output_dir).as_posix()
    feature_rel = feature_path.relative_to(output_dir).as_posix()
    candidate_rel = candidate_path.relative_to(output_dir).as_posix()
    metadata_rel = metadata_path.relative_to(output_dir).as_posix()

    task_path.write_text(
        json.dumps(
            {
                "artifact_kind": "task_config",
                "artifact_status": "confirmed",
                "payload": {"outcome_type": "binary"},
            }
        ),
        encoding="utf-8",
    )
    sample_path.write_text(
        json.dumps(
            {
                "artifact_kind": "modeling_sample_spec",
                "artifact_status": "confirmed",
                "input_paths": {"task_config_path": task_rel},
                "payload": {
                    "is_valid": True,
                    "columns": {"treatment": "treat", "outcome": "y"},
                    "datasets": {
                        "train": str((experiment_artifacts / "train.csv").resolve()),
                        "test": str((experiment_artifacts / "test.csv").resolve()),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    feature_path.write_text(
        json.dumps(
            {
                "artifact_kind": "feature_plan",
                "artifact_status": "confirmed",
                "input_paths": {
                    "task_config_path": task_rel,
                    "modeling_sample_spec_path": sample_rel,
                },
                "payload": {"selected_features": {"features": ["x1"]}},
            }
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            {
                "artifact_kind": "model_candidate",
                "artifact_version": 1,
                "candidate_id": "exp-001-t_learner_baseline",
                "model_spec": {
                    "model_type": "t_learner",
                    "base_estimators": {
                        "treatment_outcome": "lightgbm",
                        "control_outcome": "lightgbm",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "artifact_kind": "model_metadata",
                "effective_parameters": {"n_estimators": 10},
                "best_iteration": 7,
            }
        ),
        encoding="utf-8",
    )
    modeling_path.write_text(
        json.dumps(
            {
                "skill_name": "uplift-model-t-learner-modeling",
                "status": "success",
                "input_paths": {
                    "task_config_path": task_rel,
                    "modeling_sample_spec_path": sample_rel,
                    "feature_plan_path": feature_rel,
                },
                "outputs": {
                    "learner": "t_learner",
                    "model_candidate_path": candidate_rel,
                    "model_metadata_path": metadata_rel,
                    "model_artifact_path": (experiment_artifacts / "model.joblib").relative_to(output_dir).as_posix(),
                },
            }
        ),
        encoding="utf-8",
    )

    unrelated_cwd = tmp_path / "cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    inputs = tuning_run._load_tuning_inputs(
        output_dir,
        {
            "task_config_path": task_rel,
            "modeling_sample_spec_path": sample_rel,
            "feature_plan_path": feature_rel,
            "modeling_result_path": modeling_path.relative_to(output_dir).as_posix(),
        },
    )

    assert inputs["baseline_model_metadata_path"] == metadata_path.resolve()
    assert inputs["baseline_parameters"] == {"n_estimators": 10}
    assert inputs["baseline_best_iteration"] == 7
