from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import csv
import importlib.metadata
import json
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _common.data_loader import DataLoaderImportError, DataSourceError, load_table
from _common.io import read_cli_json_input, read_json, write_json, write_text
from _common.project_layout import (
    ProjectLayoutError,
    append_scope_log,
    assert_existing_run_dir,
    ensure_experiment_dir,
    next_action_paths,
    relativize_paths,
    resolve_run_path,
    to_run_relative_path,
    validate_experiment_id,
    write_experiment_action_records,
    write_scope_manifest,
)
from _common.report_language import is_zh

SKILL_NAME = "uplift-model-s-learner-modeling"
SKILL_CREATED_BY = "uplift-model-s-learner-modeling-skill"
SPLIT_NAMES = ("train", "test", "valid", "oot")
TREATMENT_FEATURE = "__uplift_modeling_treatment__"
ZERO_TOLERANCE = 1e-12
AUUC_VERSION = {"package": "scikit-uplift", "version": "0.5.1"}
AUUC_METRIC_METHOD = "sklift.metrics"
ALLOWED_REF_FIELDS = {"data_ref"}
ALLOWED_PARAMETERS = {
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "random_state",
    "n_jobs",
    "early_stopping_rounds",
    "deterministic",
    "force_col_wise",
}
DEFAULT_PARAMETERS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 20,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 50,
    "deterministic": True,
    "force_col_wise": True,
}
DEFAULT_LLM_USAGE = {
    "provider": "unknown",
    "model": "unknown",
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "usage_source": "not_used",
}


class MissingInputError(Exception):
    def __init__(self, message: str, *, missing_fields: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_fields = missing_fields or []


class NeedsConfirmationError(Exception):
    pass


class PackageImportError(Exception):
    def __init__(self, package: str, message: str) -> None:
        super().__init__(message)
        self.package = package


def run_train(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    phase = "train"
    experiment_id = validate_experiment_id(str(payload.get("experiment_id") or ""))
    try:
        packages = _required_packages()
        inputs = _load_modeling_inputs(output_dir, payload)
        requested, effective, parameter_source, parameter_warnings = _resolve_parameters(
            payload,
            has_valid=bool(inputs["dataset_paths"].get("valid")),
        )
        frames = _load_frames(inputs)
        encoder = _fit_encoder(frames["train"], inputs["selected_features"])
        estimator, eval_history = _fit_model(
            packages,
            inputs,
            frames,
            encoder,
            effective,
        )
        best_iteration = getattr(estimator, "best_iteration_", None) or None
        model_path = run_dir / "artifacts" / "s_learner_model.v1.joblib"
        text_path = run_dir / "artifacts" / "s_learner_model.v1.txt"
        feature_importance_path = run_dir / "artifacts" / "feature_importance.v1.csv"
        packages["joblib"].dump(
            {
                "model_type": "s_learner",
                "estimator": estimator,
                "encoder": encoder,
                "selected_features": inputs["selected_features"],
                "outcome_type": inputs["outcome_type"],
                "parameters": effective,
                "best_iteration": best_iteration,
            },
            model_path,
        )
        estimator.booster_.save_model(str(text_path))
        importance = _feature_importance_frame(packages, estimator, encoder["model_feature_order"])
        importance.to_csv(feature_importance_path, index=False)

        score_frames = []
        bin_tables = []
        metrics: dict[str, Any] = {}
        curve_paths: dict[str, dict[str, str]] = {}
        split_evaluations: dict[str, Any] = {}
        issues = [
            {
                "code": "TRAINING_PARAMETER_WARNING",
                "level": "warning",
                "blocking": False,
                "message": warning,
                "suggested_fix": "Add a validation split when early stopping is required.",
            }
            for warning in parameter_warnings
        ]
        for split in ("train", "valid", "test", "oot"):
            frame = frames.get(split)
            if frame is None:
                continue
            score = _score_frame(packages, estimator, inputs, frame, split, encoder, best_iteration)
            score_frames.append(score)
            bins, bin_warnings = _build_uplift_bins(packages, score, split=split)
            bin_tables.append(bins)
            auuc = _evaluate_auuc(packages, score)
            raw_curve_path = run_dir / "artifacts" / f"{split}_auuc_curve_raw.v1.csv"
            normalized_curve_path = run_dir / "artifacts" / f"{split}_auuc_curve_normalized.v1.csv"
            _build_auuc_curve(packages, score, normalized=False).to_csv(raw_curve_path, index=False)
            _build_auuc_curve(packages, score, normalized=True).to_csv(normalized_curve_path, index=False)
            curve_paths[split] = {
                "raw_curve_path": str(raw_curve_path.resolve()),
                "normalized_curve_path": str(normalized_curve_path.resolve()),
            }
            valid_bin_count = int((bins["support_status"] == "ok").sum()) if not bins.empty else 0
            metrics[split] = {
                "row_count": int(len(score)),
                "treatment_count": int((score["actual_treatment"] == 1).sum()),
                "control_count": int((score["actual_treatment"] == 0).sum()),
                "auuc_raw": auuc["raw"],
                "auuc_normalized": auuc["normalized"],
                "auuc_method": AUUC_METRIC_METHOD,
                "auuc_version": dict(AUUC_VERSION),
                "valid_bin_count": valid_bin_count,
                "overall_absolute_uplift": _overall_absolute_uplift(score),
                "overall_relative_uplift": _overall_relative_uplift(score),
                **curve_paths[split],
            }
            split_evaluations[split] = _split_evaluation_from_metric(
                metrics[split],
                split,
                str(inputs["dataset_paths"][split]),
                metric_method=AUUC_METRIC_METHOD,
            )
            for warning in [*auuc["warnings"], *bin_warnings]:
                issues.append(
                    {
                        "code": "EVALUATION_WARNING",
                        "level": "warning",
                        "blocking": False,
                        "message": f"{split}: {warning}",
                        "suggested_fix": "Review split support and detailed evaluation artifacts.",
                    }
                )
        if "test" not in metrics:
            raise MissingInputError("test dataset is required; training without test is forbidden.")
        quality = _evaluation_quality(metrics["test"])
        if quality == "insufficient":
            issues.append(
                {
                    "code": "INSUFFICIENT_EVALUATION_EVIDENCE",
                    "level": "warning",
                    "blocking": False,
                    "message": "Test evaluation evidence is insufficient for downstream selection.",
                    "suggested_fix": "Improve test treatment/control support and valid uplift bins.",
                }
            )

        pd = packages["pandas"]
        score_frame = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
        uplift_bins = pd.concat(bin_tables, ignore_index=True) if bin_tables else pd.DataFrame()
        score_frame_path = run_dir / "artifacts" / "score_frame.v1.csv"
        uplift_bins_path = run_dir / "artifacts" / "uplift_bins.v1.csv"
        evaluation_metrics_path = run_dir / "artifacts" / "evaluation_metrics.v1.json"
        model_metadata_path = run_dir / "artifacts" / "model_metadata.v1.json"
        model_candidate_path = run_dir / "artifacts" / "model_candidate.v1.json"
        score_frame.to_csv(score_frame_path, index=False)
        uplift_bins.to_csv(uplift_bins_path, index=False)
        evaluation_metrics = {
            "artifact_kind": "evaluation_metrics",
            "artifact_version": 1,
            "method": AUUC_METRIC_METHOD,
            "auuc_version": dict(AUUC_VERSION),
            "splits": metrics,
            "evaluation_quality": quality,
            "score_frame_path": str(score_frame_path.resolve()),
            "uplift_bins_path": str(uplift_bins_path.resolve()),
            "curve_paths": curve_paths,
        }
        write_json(evaluation_metrics_path, relativize_paths(evaluation_metrics, output_dir))
        metadata = {
            "artifact_kind": "model_metadata",
            "artifact_version": 1,
            "model_type": "s_learner",
            "base_estimators": {"outcome": "lightgbm"},
            "input_paths": inputs["input_paths"],
            "selected_features": inputs["selected_features"],
            "feature_count": len(inputs["selected_features"]),
            "parameter_source": parameter_source,
            "requested_parameters": requested,
            "effective_parameters": effective,
            "best_iteration": best_iteration,
            "validation_metric_summary": eval_history,
            "runtime_versions": _runtime_versions(),
            "model_artifact_path": str(model_path.resolve()),
            "lightgbm_text_path": str(text_path.resolve()),
            "feature_importance_path": str(feature_importance_path.resolve()),
            "created_by": SKILL_CREATED_BY,
            "created_at": _now(),
        }
        write_json(model_metadata_path, relativize_paths(metadata, output_dir))
        primary = metrics["test"]
        candidate = {
            "artifact_kind": "model_candidate",
            "artifact_version": 1,
            "candidate_id": experiment_id,
            "producer_kind": "modeling",
            "model_spec": {"model_type": "s_learner", "base_estimators": {"outcome": "lightgbm"}},
            "model_metadata_path": str(model_metadata_path.resolve()),
            "model_artifact_path": str(model_path.resolve()),
            "primary_evaluation": {
                "dataset_role": "test",
                "metric_name": "auuc_raw",
                "metric_value": primary.get("auuc_raw"),
                "metric_direction": "higher_is_better",
                "metric_method": AUUC_METRIC_METHOD,
                "metric_version": AUUC_VERSION["version"],
                "auuc_version": dict(AUUC_VERSION),
                "metrics_path": str(evaluation_metrics_path.resolve()),
            },
        }
        write_json(model_candidate_path, relativize_paths(candidate, output_dir))
        report_path = _write_report(
            run_dir,
            inputs=inputs,
            requested=requested,
            effective=effective,
            parameter_source=parameter_source,
            best_iteration=best_iteration,
            metrics=metrics,
            quality=quality,
            issues=issues,
            artifact_paths=[
                str(model_path.resolve()),
                str(text_path.resolve()),
                str(model_metadata_path.resolve()),
                str(evaluation_metrics_path.resolve()),
                str(score_frame_path.resolve()),
                str(uplift_bins_path.resolve()),
                *[path for split_paths in curve_paths.values() for path in split_paths.values()],
                str(feature_importance_path.resolve()),
                str(model_candidate_path.resolve()),
            ],
            language=_language_from_task_config(inputs["task_config"]),
        )
        outputs = {
            "experiment_id": experiment_id,
            "experiment_dir": str(run_dir.resolve()),
            "flow_dir": str(run_dir.resolve()),
            "modeling_result_path": None,
            "model_candidate_path": str(model_candidate_path.resolve()),
            "model_metadata_path": str(model_metadata_path.resolve()),
            "evaluation_metrics_path": str(evaluation_metrics_path.resolve()),
            "score_frame_path": str(score_frame_path.resolve()),
            "uplift_bins_path": str(uplift_bins_path.resolve()),
            "curve_paths": curve_paths,
            "feature_importance_path": str(feature_importance_path.resolve()),
            "model_artifact_path": str(model_path.resolve()),
            "lightgbm_text_path": str(text_path.resolve()),
            "report_path": str(report_path.resolve()),
            "learner": "s_learner",
            "evaluation_quality": quality,
            "training_summary": {
                "outcome_type": inputs["outcome_type"],
                "parameter_source": parameter_source,
                "best_iteration": best_iteration,
                "feature_count": len(inputs["selected_features"]),
                "validation_metric_summary": eval_history,
            },
            "modeling_metrics": evaluation_metrics,
            "split_evaluations": split_evaluations,
        }
        return _result(
            run_dir=run_dir,
            phase=phase,
            status="success",
            summary="S-Learner training and uplift evaluation completed.",
            input_paths=inputs["input_paths"],
            outputs=outputs,
            issues=issues,
            artifacts=[
                {"kind": "candidate_model", "path": str(model_path.resolve())},
                {"kind": "model_metadata", "path": str(model_metadata_path.resolve())},
                {"kind": "evaluation_metrics", "path": str(evaluation_metrics_path.resolve())},
                {"kind": "score_frame", "path": str(score_frame_path.resolve())},
                {"kind": "uplift_bins", "path": str(uplift_bins_path.resolve())},
                *[
                    {"kind": kind, "path": path}
                    for split_paths in curve_paths.values()
                    for kind, path in (
                        ("auuc_curve_raw", split_paths["raw_curve_path"]),
                        ("auuc_curve_normalized", split_paths["normalized_curve_path"]),
                    )
                ],
                {"kind": "feature_importance", "path": str(feature_importance_path.resolve())},
                {"kind": "model_candidate", "path": str(model_candidate_path.resolve())},
                {"kind": "report", "path": str(report_path.resolve())},
            ],
            progress=[
                {"step": "input_validation", "status": "success"},
                {"step": "model_training", "status": "success"},
                {"step": "uplift_evaluation", "status": "success"},
                {"step": "artifact_generation", "status": "success"},
            ],
            next_steps=[
                {
                    "skill": "uplift-model-tuning",
                    "action": "draft_plan",
                    "reason": "Optionally improve this learner with a bounded search.",
                    "inputs": {**inputs["input_paths"], "modeling_result_path": None},
                    "requires_user_confirmation": True,
                },
                {
                    "skill": "uplift-model-result-comparison",
                    "action": "draft_comparison_plan",
                    "reason": "Draft a comparison plan after at least two experiment candidates are available.",
                    "inputs": {"candidate_result_paths": [None]},
                    "requires_user_confirmation": True,
                },
                {
                    "skill": "uplift-model-reporting",
                    "action": "generate_report",
                    "reason": "Assemble a report from structured upstream results.",
                    "inputs": {"modeling_result_path": None},
                    "requires_user_confirmation": True,
                },
            ],
        )
    except (DataLoaderImportError, PackageImportError) as exc:
        return _dependency_failure_result(run_dir, phase, exc)
    except NeedsConfirmationError as exc:
        return _needs_confirmation_result(run_dir, phase, str(exc))
    except (MissingInputError, DataSourceError) as exc:
        return _needs_input_result(run_dir, phase, str(exc), getattr(exc, "missing_fields", []))
    except Exception as exc:  # noqa: BLE001
        return _unexpected_error_result(run_dir, phase, exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        assert_existing_run_dir(output_dir)
        request = read_cli_json_input(args.input)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, ProjectLayoutError):
            print(json.dumps(_layout_error_stdout(exc), sort_keys=True))
            return 0
        print(str(exc), file=sys.stderr)
        return 1
    action = str(request.get("action") or "")
    body = request.get("payload")
    if not isinstance(body, dict):
        body = {}
        request_error = "request JSON must contain an object payload."
    else:
        request_error = None
    try:
        experiment_id = validate_experiment_id(str(body.get("experiment_id") or ""))
        run_dir = ensure_experiment_dir(output_dir, experiment_id)
        request_path, result_path = next_action_paths(run_dir, action or "unknown")
    except ProjectLayoutError as exc:
        print(json.dumps(_layout_error_stdout(exc), sort_keys=True))
        return 0
    write_json(request_path, request)
    if request_error:
        result = _needs_input_result(run_dir, action or "unknown", request_error, ["payload"])
    elif action != "train":
        result = _unsupported_result(run_dir, action or "unknown", "action must be train.")
    else:
        legacy_fields = _legacy_ref_rejections(body)
        if legacy_fields:
            result = _legacy_ref_result(run_dir, action, legacy_fields)
        else:
            result = run_train(run_dir, output_dir, body)
    _attach_transport_paths(result, output_dir, run_dir, result_path, experiment_id)
    result = relativize_paths(result, output_dir)
    write_json(result_path, result)
    write_experiment_action_records(
        run_dir=output_dir,
        experiment_dir=run_dir,
        experiment_id=experiment_id,
        skill_name=SKILL_NAME,
        action=action or "unknown",
        request_path=request_path,
        result_path=result_path,
        result=result,
        extra_manifest={
            "experiment_type": "model_candidate",
            "learner": "s_learner",
            "evaluation_summary": _evaluation_summary(result.get("outputs") or {}),
        },
    )
    print(json.dumps(_stdout_payload(result), sort_keys=True))
    return 0


def _load_modeling_inputs(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    task_path = _required_input_path(output_dir, payload.get("task_config_path"), "task_config_path")
    spec_path = _required_input_path(output_dir, payload.get("modeling_sample_spec_path"), "modeling_sample_spec_path")
    feature_plan_path = _required_input_path(output_dir, payload.get("feature_plan_path"), "feature_plan_path")
    task = _read_artifact(task_path, artifact_kind="task_config", require_confirmed=True)
    sample = _read_artifact(spec_path, artifact_kind="modeling_sample_spec", require_confirmed=True)
    feature = _read_artifact(feature_plan_path, artifact_kind="feature_plan", require_confirmed=True)
    if not _same_path(sample.get("input_paths", {}).get("task_config_path"), task_path, output_dir):
        raise NeedsConfirmationError("modeling_sample_spec lineage is stale.")
    feature_inputs = feature.get("input_paths") or {}
    if not _same_path(feature_inputs.get("task_config_path"), task_path, output_dir):
        raise NeedsConfirmationError("feature_plan lineage is stale; return to feature-quality analysis.")
    if not _same_path(feature_inputs.get("modeling_sample_spec_path"), spec_path, output_dir):
        raise NeedsConfirmationError("feature_plan lineage is stale; return to feature-quality analysis.")
    task_payload = task.get("payload") or {}
    sample_payload = sample.get("payload") or {}
    if sample_payload.get("is_valid") is not True:
        raise MissingInputError("modeling_sample_spec.payload.is_valid must be true.")
    outcome_type = task_payload.get("outcome_type")
    if outcome_type not in {"binary", "continuous"}:
        raise MissingInputError("outcome_type must be binary or continuous.")
    datasets = sample_payload.get("datasets") or {}
    if not datasets.get("train"):
        raise MissingInputError("train dataset is required.")
    if not datasets.get("test"):
        raise MissingInputError("test dataset is required; training without test is forbidden.")
    selected = _selected_features_from_plan(feature, output_dir)
    if not selected:
        raise MissingInputError("selected features must not be empty.")
    columns = sample_payload.get("columns") or {}
    return {
        "input_paths": {
            "task_config_path": str(task_path),
            "modeling_sample_spec_path": str(spec_path),
            "feature_plan_path": str(feature_plan_path),
        },
        "task_config": task,
        "sample_spec": sample,
        "feature_plan": feature,
        "selected_features": selected,
        "dataset_paths": {name: datasets.get(name) for name in SPLIT_NAMES},
        "treatment_column": str(columns.get("treatment") or "__uplift_modeling_treatment__"),
        "outcome_column": str(columns.get("outcome") or "__uplift_modeling_outcome__"),
        "outcome_type": str(outcome_type),
        "unit_id_column": task_payload.get("unit_id_column"),
    }


def _load_frames(inputs: dict[str, Any]) -> dict[str, Any]:
    frames = {}
    required = set(inputs["selected_features"]) | {inputs["treatment_column"], inputs["outcome_column"]}
    for split, path_value in inputs["dataset_paths"].items():
        if not path_value:
            continue
        frame = load_table({"kind": "local_csv", "path": str(path_value)})
        missing = sorted(required - set(frame.columns))
        if missing:
            raise MissingInputError(f"{split} dataset is missing required columns: {missing}")
        treatment_values = set(frame[inputs["treatment_column"]].dropna().unique())
        if not treatment_values or not treatment_values.issubset({0, 1}):
            raise MissingInputError(f"{split} internal treatment must contain only mapped 0/1 values.")
        frames[split] = frame
    return frames


def _selected_features_from_plan(feature_plan: dict[str, Any], output_dir: Path) -> list[str]:
    selected = feature_plan.get("payload", {}).get("selected_features", {})
    if selected.get("features"):
        return [str(item) for item in selected.get("features") if str(item)]
    if selected.get("features_path"):
        payload = _read_json_object(resolve_run_path(output_dir, selected["features_path"]))
        return [str(item) for item in payload.get("features", []) if str(item)]
    raise MissingInputError("feature_plan is missing selected features.")


def _resolve_parameters(payload: dict[str, Any], *, has_valid: bool) -> tuple[dict[str, Any], dict[str, Any], str, list[str]]:
    mode = payload.get("parameter_mode", "recommended_defaults")
    if mode not in {"recommended_defaults", "user_overrides"}:
        raise MissingInputError("parameter_mode must be recommended_defaults or user_overrides.")
    overrides = payload.get("parameter_overrides") or {}
    if not isinstance(overrides, dict):
        raise MissingInputError("parameter_overrides must be an object.")
    unknown = sorted(set(overrides) - ALLOWED_PARAMETERS)
    if unknown:
        raise MissingInputError(f"Unsupported parameter overrides: {unknown}")
    requested = {**DEFAULT_PARAMETERS, **overrides}
    _validate_parameters(requested)
    effective = dict(requested)
    warnings = []
    if not has_valid:
        effective["early_stopping_rounds"] = None
        if mode == "recommended_defaults" and "n_estimators" not in overrides:
            effective["n_estimators"] = 200
        elif "n_estimators" in overrides:
            warnings.append("No validation split; the requested iteration count cannot early stop.")
    return requested, effective, "user_overrides" if overrides else "recommended_defaults", warnings


def _validate_parameters(parameters: dict[str, Any]) -> None:
    for name in ("n_estimators", "num_leaves", "min_child_samples"):
        value = parameters[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise MissingInputError(f"{name} must be a positive integer.")
    if not isinstance(parameters["learning_rate"], (int, float)) or isinstance(parameters["learning_rate"], bool) or parameters["learning_rate"] <= 0:
        raise MissingInputError("learning_rate must be positive.")
    if not isinstance(parameters["n_jobs"], int) or isinstance(parameters["n_jobs"], bool) or parameters["n_jobs"] == 0:
        raise MissingInputError("n_jobs must be a non-zero integer.")
    max_depth = parameters["max_depth"]
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth == 0 or max_depth < -1:
        raise MissingInputError("max_depth must be -1 or a positive integer.")
    for name in ("subsample", "colsample_bytree"):
        value = parameters[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= 1:
            raise MissingInputError(f"{name} must be in (0, 1].")
    for name in ("reg_alpha", "reg_lambda"):
        value = parameters[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise MissingInputError(f"{name} must be non-negative.")
    rounds = parameters["early_stopping_rounds"]
    if rounds is not None and (not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0):
        raise MissingInputError("early_stopping_rounds must be a positive integer or null.")
    if not isinstance(parameters["random_state"], int) or isinstance(parameters["random_state"], bool):
        raise MissingInputError("random_state must be an integer.")


def _fit_encoder(train_frame: Any, selected_features: list[str]) -> dict[str, Any]:
    pd = _pd_module()
    categorical: dict[str, list[str]] = {}
    numeric_medians: dict[str, float] = {}
    model_feature_order = [TREATMENT_FEATURE]
    for feature in selected_features:
        series = train_frame[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_like = pd.api.types.is_numeric_dtype(series) or numeric.notna().sum() >= max(1, int(series.notna().sum() * 0.8))
        if numeric_like:
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            numeric_medians[feature] = median
            model_feature_order.append(feature)
        else:
            categories = [str(item) for item in series.dropna().astype(str).unique()]
            categorical[feature] = categories
            encoded = [f"{feature}={category}" for category in categories]
            encoded.append(f"{feature}=__MISSING__")
            encoded.append(f"{feature}=__OTHER__")
            model_feature_order.extend(encoded)
    return {
        "selected_features": selected_features,
        "categorical": categorical,
        "numeric_medians": numeric_medians,
        "model_feature_order": model_feature_order,
    }


def _encode_frame(packages: dict[str, Any], frame: Any, selected_features: list[str], encoder: dict[str, Any], treatment: Any) -> Any:
    pd = packages["pandas"]
    np = packages["numpy"]
    encoded = pd.DataFrame(index=frame.index)
    for feature in selected_features:
        if feature in encoder["categorical"]:
            categories = encoder["categorical"][feature]
            values = frame[feature].astype("object").where(frame[feature].notna(), "__MISSING__").astype(str)
            known = set(categories) | {"__MISSING__"}
            for category in categories:
                encoded[f"{feature}={category}"] = (values == category).astype(int)
            encoded[f"{feature}=__MISSING__"] = (values == "__MISSING__").astype(int)
            encoded[f"{feature}=__OTHER__"] = (~values.isin(known)).astype(int)
        else:
            numeric = pd.to_numeric(frame[feature], errors="coerce").fillna(encoder["numeric_medians"][feature])
            encoded[feature] = numeric.astype(float)
    values = np.asarray(treatment)
    if values.ndim == 0:
        values = np.full(len(frame), values)
    if len(values) != len(frame):
        raise MissingInputError("Treatment length must match frame length.")
    encoded.insert(0, TREATMENT_FEATURE, values.astype(int))
    return encoded[encoder["model_feature_order"]]


def _fit_model(packages: dict[str, Any], inputs: dict[str, Any], frames: dict[str, Any], encoder: dict[str, Any], parameters: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    lgb = packages["lightgbm"]
    train = frames["train"]
    x_train = _encode_frame(packages, train, inputs["selected_features"], encoder, train[inputs["treatment_column"]])
    y_train = train[inputs["outcome_column"]]
    estimator_parameters = {
        key: value
        for key, value in parameters.items()
        if key != "early_stopping_rounds" and value is not None
    }
    estimator_parameters["verbosity"] = -1
    if inputs["outcome_type"] == "binary":
        estimator = lgb.LGBMClassifier(objective="binary", metric="binary_logloss", **estimator_parameters)
        metric = "binary_logloss"
    elif inputs["outcome_type"] == "continuous":
        estimator = lgb.LGBMRegressor(objective="regression", metric="l2", **estimator_parameters)
        metric = "l2"
    else:
        raise MissingInputError("outcome_type must be binary or continuous.")
    fit_kwargs: dict[str, Any] = {}
    valid = frames.get("valid")
    history: dict[str, Any] = {}
    if valid is not None and len(valid) > 0:
        x_valid = _encode_frame(packages, valid, inputs["selected_features"], encoder, valid[inputs["treatment_column"]])
        fit_kwargs["eval_set"] = [(x_valid, valid[inputs["outcome_column"]])]
        callbacks = [lgb.record_evaluation(history)]
        rounds = parameters.get("early_stopping_rounds")
        if rounds:
            callbacks.insert(0, lgb.early_stopping(int(rounds), verbose=False))
        fit_kwargs["callbacks"] = callbacks
    estimator.fit(x_train, y_train, **fit_kwargs)
    values = history.get("valid_0", {}).get(metric, [])
    summary = {
        "metric": metric if values else None,
        "iteration_count": len(values),
        "last_value": float(values[-1]) if values else None,
        "best_value": float(min(values)) if values else None,
    }
    return estimator, summary


def _predict_potential_outcomes(packages: dict[str, Any], estimator: Any, inputs: dict[str, Any], frame: Any, encoder: dict[str, Any], best_iteration: int | None) -> tuple[Any, Any]:
    np = packages["numpy"]
    x1 = _encode_frame(packages, frame, inputs["selected_features"], encoder, 1)
    x0 = _encode_frame(packages, frame, inputs["selected_features"], encoder, 0)
    kwargs = {"num_iteration": best_iteration} if best_iteration else {}
    if inputs["outcome_type"] == "binary":
        p1 = estimator.predict_proba(x1, **kwargs)[:, 1]
        p0 = estimator.predict_proba(x0, **kwargs)[:, 1]
    else:
        p1 = estimator.predict(x1, **kwargs)
        p0 = estimator.predict(x0, **kwargs)
    return np.asarray(p1), np.asarray(p0)


def _score_frame(packages: dict[str, Any], estimator: Any, inputs: dict[str, Any], frame: Any, split: str, encoder: dict[str, Any], best_iteration: int | None) -> Any:
    pd = packages["pandas"]
    prediction_t1, prediction_t0 = _predict_potential_outcomes(packages, estimator, inputs, frame, encoder, best_iteration)
    unit_id_column = inputs.get("unit_id_column")
    row_id = frame[unit_id_column].to_numpy() if unit_id_column and unit_id_column in frame.columns else list(range(len(frame)))
    return pd.DataFrame(
        {
            "row_id": row_id,
            "split": split,
            "actual_treatment": frame[inputs["treatment_column"]].to_numpy(),
            "actual_outcome": frame[inputs["outcome_column"]].to_numpy(),
            "prediction_t1": prediction_t1,
            "prediction_t0": prediction_t0,
            "uplift_score": prediction_t1 - prediction_t0,
        }
    )


def _build_uplift_bins(packages: dict[str, Any], score_frame: Any, *, split: str, max_bins: int = 10) -> tuple[Any, list[str]]:
    pd = packages["pandas"]
    np = packages["numpy"]
    if score_frame.empty:
        return pd.DataFrame(), ["Score frame is empty."]
    sorted_frame = score_frame.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    bin_count = min(max_bins, len(sorted_frame))
    sorted_frame["bin_id"] = (np.floor(np.arange(len(sorted_frame)) * bin_count / len(sorted_frame)) + 1).astype(int)
    rows = []
    warnings = []
    overall_abs = _overall_absolute_uplift(score_frame)
    for bin_id, group in sorted_frame.groupby("bin_id", sort=True):
        treatment = group[group["actual_treatment"] == 1]
        control = group[group["actual_treatment"] == 0]
        treat_mean = _mean(treatment["actual_outcome"])
        control_mean = _mean(control["actual_outcome"])
        absolute = None if treat_mean is None or control_mean is None else treat_mean - control_mean
        relative = None if absolute is None or control_mean in {None, 0} else absolute / control_mean
        support = "ok" if len(treatment) and len(control) else "insufficient_group_support"
        if support != "ok":
            warnings.append(f"bin {bin_id} lacks treatment or control support.")
        rows.append(
            {
                "split": split,
                "bin_id": int(bin_id),
                "row_count": int(len(group)),
                "score_min": float(group["uplift_score"].min()),
                "score_max": float(group["uplift_score"].max()),
                "treatment_count": int(len(treatment)),
                "control_count": int(len(control)),
                "treatment_outcome_mean": treat_mean,
                "control_outcome_mean": control_mean,
                "absolute_uplift": absolute,
                "relative_uplift": relative,
                "overall_absolute_uplift": overall_abs,
                "support_status": support,
            }
        )
    return pd.DataFrame(rows), warnings


def _evaluate_auuc(packages: dict[str, Any], score_frame: Any) -> dict[str, Any]:
    np = packages["numpy"]
    sklift_metrics = packages["sklift_metrics"]
    if score_frame.empty:
        return {"raw": None, "normalized": None, "warnings": ["Score frame is empty."]}
    treatment_count = int((score_frame["actual_treatment"] == 1).sum())
    control_count = int((score_frame["actual_treatment"] == 0).sum())
    if treatment_count == 0 or control_count == 0:
        return {"raw": None, "normalized": None, "warnings": ["Treatment or control group is missing."]}
    y_true, uplift, treatment = _sklift_auuc_inputs(score_frame)
    x_actual, y_actual = sklift_metrics.uplift_curve(y_true, uplift, treatment)
    x = np.asarray(x_actual, dtype=float)
    y = np.asarray(y_actual, dtype=float)
    if len(x) < 2 or abs(float(x[-1])) <= ZERO_TOLERANCE:
        return {"raw": None, "normalized": None, "warnings": ["AUUC curve is not available."]}
    random_baseline = x * (float(y[-1]) / float(x[-1]))
    raw = float(np.trapezoid(y, x) - np.trapezoid(random_baseline, x))
    normalized = float(sklift_metrics.uplift_auc_score(y_true, uplift, treatment))
    return {"raw": raw, "normalized": normalized, "warnings": []}


def _build_auuc_curve(packages: dict[str, Any], score_frame: Any, *, normalized: bool) -> Any:
    pd = packages["pandas"]
    np = packages["numpy"]
    sklift_metrics = packages["sklift_metrics"]
    if score_frame.empty:
        return pd.DataFrame(columns=["population_index", "cumulative_gain", "random_baseline", "adjusted_gain"])
    treatment_count = int((score_frame["actual_treatment"] == 1).sum())
    control_count = int((score_frame["actual_treatment"] == 0).sum())
    if treatment_count == 0 or control_count == 0:
        return pd.DataFrame(columns=["population_index", "cumulative_gain", "random_baseline", "adjusted_gain"])
    y_true, uplift, treatment = _sklift_auuc_inputs(score_frame)
    x_actual, y_actual = sklift_metrics.uplift_curve(y_true, uplift, treatment)
    x = np.asarray(x_actual, dtype=float)
    gains = np.asarray(y_actual, dtype=float)
    if len(x) == 0:
        return pd.DataFrame(columns=["population_index", "cumulative_gain", "random_baseline", "adjusted_gain"])
    baseline = x * (float(gains[-1]) / float(x[-1])) if abs(float(x[-1])) > ZERO_TOLERANCE else np.zeros_like(gains)
    adjusted = gains - baseline
    if normalized:
        denom = max(float(np.max(np.abs(adjusted))), ZERO_TOLERANCE)
        gains = adjusted / denom
        baseline = np.zeros_like(gains)
        adjusted = gains
    return pd.DataFrame(
        {
            "population_index": x,
            "cumulative_gain": gains,
            "random_baseline": baseline,
            "adjusted_gain": adjusted,
        }
    )


def _split_evaluation_from_metric(metric: dict[str, Any], split: str, dataset_path: str, *, metric_method: str) -> dict[str, Any]:
    return {
        "dataset_role": split,
        "population_fingerprint": _file_fingerprint(Path(dataset_path)),
        "metric_name": "auuc_raw",
        "metric_value": metric.get("auuc_raw"),
        "metric_direction": "higher_is_better",
        "metric_method": metric_method,
        "metric_version": AUUC_VERSION["version"],
        "auuc_version": dict(AUUC_VERSION),
        "auuc_raw": metric.get("auuc_raw"),
        "auuc_normalized": metric.get("auuc_normalized"),
        "raw_curve_path": metric.get("raw_curve_path"),
        "normalized_curve_path": metric.get("normalized_curve_path"),
        "support": {
            "row_count": metric.get("row_count"),
            "treatment_count": metric.get("treatment_count"),
            "control_count": metric.get("control_count"),
            "valid_bin_count": metric.get("valid_bin_count"),
        },
    }


def _sklift_auuc_inputs(score_frame: Any) -> tuple[Any, Any, Any]:
    return (
        score_frame["actual_outcome"].astype(float).to_numpy(),
        score_frame["uplift_score"].astype(float).to_numpy(),
        score_frame["actual_treatment"].astype(int).to_numpy(),
    )


def _feature_importance_frame(packages: dict[str, Any], estimator: Any, model_feature_order: list[str]) -> Any:
    pd = packages["pandas"]
    booster = estimator.booster_
    gain = booster.feature_importance(importance_type="gain").astype(float)
    split = booster.feature_importance(importance_type="split").astype(float)
    result = pd.DataFrame(
        {
            "feature_name": model_feature_order,
            "feature_role": ["treatment", *(["business_feature"] * (len(model_feature_order) - 1))],
            "gain": gain,
            "split": split,
        }
    )
    result["gain_share"] = gain / gain.sum() if gain.sum() else 0.0
    result["split_share"] = split / split.sum() if split.sum() else 0.0
    result["gain_rank"] = result["gain"].rank(method="min", ascending=False).astype(int)
    result["split_rank"] = result["split"].rank(method="min", ascending=False).astype(int)
    return result.sort_values(["gain", "feature_name"], ascending=[False, True]).reset_index(drop=True)[
        ["feature_name", "feature_role", "gain", "gain_share", "gain_rank", "split", "split_share", "split_rank"]
    ]


def _overall_absolute_uplift(score_frame: Any) -> float | None:
    treatment = score_frame[score_frame["actual_treatment"] == 1]
    control = score_frame[score_frame["actual_treatment"] == 0]
    treat_mean = _mean(treatment["actual_outcome"])
    control_mean = _mean(control["actual_outcome"])
    return None if treat_mean is None or control_mean is None else treat_mean - control_mean


def _overall_relative_uplift(score_frame: Any) -> float | None:
    absolute = _overall_absolute_uplift(score_frame)
    control = score_frame[score_frame["actual_treatment"] == 0]
    control_mean = _mean(control["actual_outcome"])
    if absolute is None or control_mean in {None, 0}:
        return None
    return absolute / control_mean


def _mean(series: Any) -> float | None:
    values = series.dropna()
    return None if values.empty else float(values.mean())


def _evaluation_quality(test_metrics: dict[str, Any]) -> str:
    no_auuc = test_metrics.get("auuc_normalized") is None and test_metrics.get("auuc_raw") is None
    lacks_groups = not test_metrics.get("treatment_count") or not test_metrics.get("control_count")
    few_bins = int(test_metrics.get("valid_bin_count") or 0) < 8
    return "insufficient" if no_auuc or lacks_groups or few_bins else "sufficient"


def _write_report(
    run_dir: Path,
    *,
    inputs: dict[str, Any],
    requested: dict[str, Any],
    effective: dict[str, Any],
    parameter_source: str,
    best_iteration: int | None,
    metrics: dict[str, Any],
    quality: str,
    issues: list[dict[str, Any]],
    artifact_paths: list[str],
    language: str,
) -> Path:
    path = run_dir / "report.md"
    zh = is_zh(language)
    artifact_lookup = _artifact_path_lookup(artifact_paths)
    uplift_bins_path = artifact_lookup.get("uplift_bins.v1.csv")
    feature_importance_path = artifact_lookup.get("feature_importance.v1.csv")
    model_metadata_path = artifact_lookup.get("model_metadata.v1.json")
    lines = (
        [
            "# S-Learner 建模",
            "",
            "## 1. 输入与血缘",
            "",
            "- 输入 artifact：已确认",
            f"- 报告语言：`{language}`",
            "",
            "## 2. 模型与目标变量",
            "",
            "- Learner：LightGBM S-Learner",
            f"- Outcome 类型：`{inputs['outcome_type']}`",
            f"- 特征数：{len(inputs['selected_features'])}",
            "",
            "## 3. 参数",
            "",
            *_parameter_report_lines(
                requested=requested,
                effective=effective,
                parameter_source=parameter_source,
                best_iteration=best_iteration,
                has_valid=bool(inputs["dataset_paths"].get("valid")),
                model_metadata_path=model_metadata_path,
                zh=True,
            ),
            "",
            "## 4. AUUC",
            "",
        ]
        if zh
        else [
            "# S-Learner Modeling",
            "",
            "## 1. Inputs And Lineage",
            "",
            "- Input artifacts: confirmed",
            f"- Report language: `{language}`",
            "",
            "## 2. Model And Outcome",
            "",
            "- Learner: LightGBM S-Learner",
            f"- Outcome type: `{inputs['outcome_type']}`",
            f"- Feature count: {len(inputs['selected_features'])}",
            "",
            "## 3. Parameters",
            "",
            *_parameter_report_lines(
                requested=requested,
                effective=effective,
                parameter_source=parameter_source,
                best_iteration=best_iteration,
                has_valid=bool(inputs["dataset_paths"].get("valid")),
                model_metadata_path=model_metadata_path,
                zh=False,
            ),
            "",
            "## 4. AUUC",
            "",
        ]
    )
    header = "| 切分 | Normalized AUUC | Raw AUUC | 行数 | Treatment | Control | 有效分箱 |" if zh else "| Split | Normalized AUUC | Raw AUUC | Rows | Treatment | Control | Valid bins |"
    lines.extend([header, "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for split, metric in sorted(metrics.items(), key=lambda item: _split_sort_key(item[0])):
        lines.append(
            f"| {split} | {_format_decimal(metric.get('auuc_normalized'))} | {_format_decimal(metric.get('auuc_raw'))} | {metric.get('row_count')} | {metric.get('treatment_count')} | {metric.get('control_count')} | {metric.get('valid_bin_count')} |"
        )
    lines.extend(
        [
            "",
            "## 5. Uplift 分箱" if zh else "## 5. Uplift Bins",
            "",
            *_uplift_bin_report_lines(uplift_bins_path, zh),
            "",
            "## 6. 特征重要性" if zh else "## 6. Feature Importance",
            "",
            *_feature_importance_report_lines(feature_importance_path, zh),
            "",
            "## 7. 证据与警告" if zh else "## 7. Evidence And Warnings",
            "",
            f"- {'评估质量' if zh else 'Evaluation quality'}：`{quality}`",
            "",
        ]
    )
    if issues:
        lines.extend(f"- {item.get('code')}: {item.get('message')}" for item in issues)
    else:
        lines.append("- 无。" if zh else "- None.")
    lines.extend(
        [
            "",
            "## 8. 下一步" if zh else "## 8. Next Steps",
            "",
            "- 可选：对这个 S-Learner 候选模型执行调参。" if zh else "- Optional: tune this S-Learner candidate.",
            "- 在血缘和 AUUC 方法/版本一致的前提下比较成功候选结果。" if zh else "- Compare successful candidates only when lineage and AUUC method/version match.",
            "- 最终报告只从结构化上游事实生成。" if zh else "- Generate final reports only from structured upstream facts.",
            "",
            "## 附录：系统引用" if zh else "## Appendix: System References",
            "",
        ]
    )
    for key, value in inputs["input_paths"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append(f"- result_path: `{(run_dir / 'results' / 'train.result.json').resolve()}`")
    lines.extend(f"- `{value}`" for value in artifact_paths)
    lines.extend(
        [
            "",
            "> 只从可信来源加载 joblib 模型文件。" if zh else "> Load joblib model files only from trusted sources.",
            "",
        ]
    )
    write_text(path, "\n".join(lines))
    return path


def _parameter_report_lines(
    *,
    requested: dict[str, Any],
    effective: dict[str, Any],
    parameter_source: str,
    best_iteration: int | None,
    has_valid: bool,
    model_metadata_path: str | None,
    zh: bool,
) -> list[str]:
    lines = [
        f"- {'来源' if zh else 'Source'}：`{parameter_source}`" if zh else f"- Source: `{parameter_source}`",
        f"- {'训练轮数' if zh else 'Training iterations'}：`{effective.get('n_estimators')}`" if zh else f"- Training iterations: `{effective.get('n_estimators')}`",
        _early_stopping_report_line(effective, has_valid, zh),
        _best_iteration_report_line(best_iteration, effective, has_valid, zh),
        "",
        "### 关键参数" if zh else "### Key Parameters",
        "",
        "| 参数 | 请求值 | 生效值 |" if zh else "| Parameter | Requested | Effective |",
        "| --- | ---: | ---: |",
    ]
    for name in ("n_estimators", "learning_rate", "num_leaves", "max_depth", "min_child_samples", "subsample", "colsample_bytree"):
        lines.append(f"| `{name}` | `{_format_parameter_value(requested.get(name))}` | `{_format_parameter_value(effective.get(name))}` |")
    if model_metadata_path:
        lines.extend(
            [
                "",
                f"- {'完整参数' if zh else 'Full parameters'}：`{model_metadata_path}`" if zh else f"- Full parameters: `{model_metadata_path}`",
            ]
        )
    return lines


def _early_stopping_report_line(effective: dict[str, Any], has_valid: bool, zh: bool) -> str:
    rounds = effective.get("early_stopping_rounds")
    if has_valid and rounds:
        return f"- 早停：启用，`early_stopping_rounds={rounds}`" if zh else f"- Early stopping: enabled, `early_stopping_rounds={rounds}`"
    if has_valid:
        return "- 早停：未启用（`early_stopping_rounds` 为空，按固定轮数训练）" if zh else "- Early stopping: disabled (`early_stopping_rounds` is null; trained for fixed iterations)"
    return "- 早停：未启用（未提供 valid split，按固定轮数训练）" if zh else "- Early stopping: disabled (no valid split; trained for fixed iterations)"


def _best_iteration_report_line(best_iteration: int | None, effective: dict[str, Any], has_valid: bool, zh: bool) -> str:
    rounds = effective.get("early_stopping_rounds")
    if not has_valid:
        return "- 最佳迭代轮次：不适用（无 valid split）" if zh else "- Best iteration: not applicable (no valid split)"
    if not rounds:
        return "- 最佳迭代轮次：不适用（早停未启用）" if zh else "- Best iteration: not applicable (early stopping disabled)"
    if best_iteration is None:
        return "- 最佳迭代轮次：未返回（已启用早停，但 LightGBM 未提供 best_iteration_）" if zh else "- Best iteration: unavailable (early stopping enabled, but LightGBM did not return best_iteration_)"
    return f"- 最佳迭代轮次：`{best_iteration}`" if zh else f"- Best iteration: `{best_iteration}`"


def _format_parameter_value(value: Any) -> str:
    return "null" if value is None else str(value)


def _artifact_path_lookup(paths: list[str]) -> dict[str, str]:
    return {Path(value).name: value for value in paths}


def _read_csv_rows(path_value: str | None) -> list[dict[str, str]]:
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _uplift_bin_report_lines(path_value: str | None, zh: bool) -> list[str]:
    rows = _read_csv_rows(path_value)
    if not rows:
        return ["未生成。" if zh else "Not generated."]
    output: list[str] = []
    for split in sorted({row["split"] for row in rows}, key=_split_sort_key):
        output.extend(
            [
                f"### {split}",
                "",
                (
                    "| segment | total_count | treatment_count | control_count | treatment_mean_y | control_mean_y | absolute_uplift | relative_uplift | score_range |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        split_rows = [row for row in rows if row["split"] == split]
        split_rows.sort(key=lambda row: int(float(row.get("bin_id") or 0)))
        for row in split_rows:
            output.append(
                "| "
                + " | ".join(
                    [
                        f"D{int(float(row.get('bin_id') or 0))}",
                        str(row.get("row_count") or ""),
                        str(row.get("treatment_count") or ""),
                        str(row.get("control_count") or ""),
                        _format_percent(row.get("treatment_outcome_mean")),
                        _format_percent(row.get("control_outcome_mean")),
                        _format_percent(row.get("absolute_uplift")),
                        _format_percent(row.get("relative_uplift")),
                        f"{_format_scientific(row.get('score_min'))} ~ {_format_scientific(row.get('score_max'))}",
                    ]
                )
                + " |"
            )
        output.append("")
    return output


def _feature_importance_report_lines(path_value: str | None, zh: bool) -> list[str]:
    rows = _read_csv_rows(path_value)
    if not rows:
        return ["未生成。" if zh else "Not generated."]
    output = [
        "| feature_name | feature_role | gain | gain_share | gain_rank | split | split_share | split_rank |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows[:20]:
        output.append(
            f"| {row.get('feature_name')} | {row.get('feature_role')} | {_format_decimal(row.get('gain'))} | {_format_decimal(row.get('gain_share'), 6)} | {row.get('gain_rank')} | {_format_decimal(row.get('split'), 0)} | {_format_decimal(row.get('split_share'), 6)} | {row.get('split_rank')} |"
        )
    return output


def _format_percent(value: Any) -> str:
    if value in {None, ""}:
        return "N/A"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(parsed):
        return "N/A"
    return f"{parsed * 100:.2f}%"


def _format_scientific(value: Any) -> str:
    if value in {None, ""}:
        return "N/A"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(parsed):
        return "N/A"
    return f"{parsed:.2e}"


def _read_artifact(path: Path, *, artifact_kind: str, require_confirmed: bool = False) -> dict[str, Any]:
    artifact = _read_json_object(path)
    if artifact.get("artifact_kind") != artifact_kind:
        raise MissingInputError(f"{path} is not a {artifact_kind} artifact.")
    if require_confirmed and artifact.get("artifact_status") != "confirmed":
        raise MissingInputError(f"{artifact_kind} must be confirmed.")
    return artifact


def _read_json_object(path: Path | str) -> dict[str, Any]:
    return read_json(Path(path))


def _required_input_path(output_dir: Path, value: Any, field_name: str) -> Path:
    if not value:
        raise MissingInputError(f"{field_name} is required.", missing_fields=[field_name])
    path = _resolve_input_path(output_dir, str(value))
    if not path.exists():
        raise MissingInputError(f"{field_name} does not exist: {path}", missing_fields=[field_name])
    return path


def _resolve_input_path(output_dir: Path, value: str) -> Path:
    return resolve_run_path(output_dir, value)


def _same_path(left: Any, right: Any, run_dir: Path | None = None) -> bool:
    if not left or not right:
        return False
    try:
        left_path = resolve_run_path(run_dir, left) if run_dir else Path(str(left)).expanduser().resolve()
        right_path = resolve_run_path(run_dir, right) if run_dir else Path(str(right)).expanduser().resolve()
        return left_path == right_path
    except OSError:
        return str(left) == str(right)


def _language_from_task_config(task: dict[str, Any]) -> str:
    return str(task.get("payload", {}).get("report_preferences", {}).get("language") or "zh-CN")


def _split_sort_key(value: str) -> tuple[int, str]:
    order = {"train": 0, "valid": 1, "test": 2, "oot": 3}
    return (order.get(value, 99), value)


def _file_fingerprint(path: Path) -> str:
    return f"sha256:{_sha256(path)}"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_decimal(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(parsed):
        return "N/A"
    return f"{parsed:.{digits}f}"


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("lightgbm", "pandas", "numpy", "scikit-learn", "joblib", "scikit-uplift"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def _required_packages() -> dict[str, Any]:
    return {
        "pandas": _pd_module(),
        "numpy": _np_module(),
        "joblib": _joblib_module(),
        "lightgbm": _lightgbm_module(),
        "sklift_metrics": _sklift_metrics_module(),
    }


def _pd_module() -> Any:
    try:
        import pandas as pd  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise PackageImportError("pandas", "Missing package: pandas") from exc
    return pd


def _np_module() -> Any:
    try:
        import numpy as np  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise PackageImportError("numpy", "Missing package: numpy") from exc
    return np


def _joblib_module() -> Any:
    try:
        import joblib  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise PackageImportError("joblib", "Missing package: joblib") from exc
    return joblib


def _lightgbm_module() -> Any:
    try:
        import lightgbm as lgb  # type: ignore
    except (ImportError, ModuleNotFoundError, OSError, PermissionError) as exc:
        raise PackageImportError("lightgbm", "Missing or unloadable package: lightgbm") from exc
    return lgb


def _sklift_metrics_module() -> Any:
    try:
        version = importlib.metadata.version("scikit-uplift")
    except importlib.metadata.PackageNotFoundError as exc:
        raise PackageImportError("scikit-uplift", "Missing package: scikit-uplift==0.5.1") from exc
    if version != AUUC_VERSION["version"]:
        raise PackageImportError("scikit-uplift", f"scikit-uplift=={AUUC_VERSION['version']} is required for AUUC; installed {version}.")
    try:
        from sklift import metrics as sklift_metrics  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise PackageImportError("scikit-uplift", "Missing package: scikit-uplift==0.5.1") from exc
    return sklift_metrics


def _legacy_ref_rejections(payload: dict[str, Any]) -> list[dict[str, str]]:
    rejected_fields = []
    for field in sorted(payload):
        if field in ALLOWED_REF_FIELDS:
            continue
        if field.endswith("_refs"):
            rejected_fields.append({"field": field, "suggested_field": f"{field[:-5]}_paths"})
        elif field.endswith("_ref"):
            rejected_fields.append({"field": field, "suggested_field": f"{field[:-4]}_path"})
    return rejected_fields


def _legacy_ref_result(run_dir: Path, phase: str, rejected_fields: list[dict[str, str]]) -> dict[str, Any]:
    field_names = ", ".join(item["field"] for item in rejected_fields)
    issues = [
        {
            "code": "LEGACY_REF_FIELD_NOT_ACCEPTED",
            "level": "critical",
            "blocking": True,
            "field": item["field"],
            "suggested_field": item["suggested_field"],
            "message": f"{item['field']} is not accepted by self-contained runners; use {item['suggested_field']}.",
        }
        for item in rejected_fields
    ]
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="needs_input",
        summary=f"Legacy _ref input fields are not accepted: {field_names}.",
        input_paths={},
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "rejected_fields": rejected_fields,
            "missing_fields": [item["suggested_field"] for item in rejected_fields],
            "modeling_result_path": None,
            "model_candidate_path": None,
            "report_path": None,
        },
        issues=issues,
        progress=[{"step": "reject_legacy_ref_fields", "status": "needs_input", "message": "Replace legacy _ref inputs with explicit _path fields."}],
        next_steps=[
            {
                "action": "replace_legacy_ref_field",
                "field": item["field"],
                "suggested_field": item["suggested_field"],
                "reason": "Self-contained skill handoff uses explicit filesystem paths.",
            }
            for item in rejected_fields
        ],
    )


def _result(
    *,
    run_dir: Path,
    phase: str,
    status: str,
    summary: str,
    input_paths: dict[str, Any],
    outputs: dict[str, Any],
    issues: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    error: dict[str, Any] | None = None,
    progress: list[dict[str, Any]] | None = None,
    next_steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = {
        "skill_name": SKILL_NAME,
        "run_id": run_dir.name,
        "phase": phase,
        "status": status,
        "summary": summary,
        "input_paths": input_paths,
        "outputs": outputs,
        "issues": issues or [],
        "artifacts": artifacts or [],
        "error": error,
        "metadata": {"llm_usage": dict(DEFAULT_LLM_USAGE)},
        "progress": progress or [],
        "next_steps": next_steps or [],
        "created_at": _now(),
    }
    result["user_interaction"] = {
        "type": "completion" if status == "success" else "recovery",
        "subject": "s_learner_modeling",
        "facts": {"issue_count": len(result["issues"])},
    }
    return result


def _needs_input_result(run_dir: Path, phase: str, message: str, missing_fields: list[str] | None = None) -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="needs_input",
        summary=message,
        input_paths={},
        outputs={"flow_dir": str(run_dir.resolve()), "missing_fields": missing_fields or [], "report_path": None},
        issues=[{"code": "MISSING_OR_INVALID_INPUT", "level": "critical", "blocking": True, "message": message}],
        progress=[{"step": phase, "status": "needs_input", "message": message}],
    )


def _needs_confirmation_result(run_dir: Path, phase: str, message: str) -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="needs_confirmation",
        summary=message,
        input_paths={},
        outputs={"flow_dir": str(run_dir.resolve()), "confirmation_required_reason": message, "report_path": None},
        issues=[{"code": "USER_CONFIRMATION_REQUIRED", "level": "warning", "blocking": False, "message": message}],
        progress=[{"step": phase, "status": "needs_confirmation", "message": message}],
    )


def _unsupported_result(run_dir: Path, phase: str, message: str) -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="failed",
        summary=message,
        input_paths={},
        outputs={"flow_dir": str(run_dir.resolve()), "unsupported_reason": message, "report_path": None},
        error={"code": "UNSUPPORTED_INPUT", "message": message, "recoverable": True, "retryable": False, "raw_error": None},
        progress=[{"step": phase, "status": "failed", "message": message}],
    )


def _dependency_failure_result(run_dir: Path, phase: str, exc: Any) -> dict[str, Any]:
    package = getattr(exc, "package", "unknown")
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="failed",
        summary="Missing Python dependency for this action.",
        input_paths={},
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "runtime_requirements_path": None,
            "runtime_requirements_reference": "SKILL.md#缺依赖处理",
            "report_path": None,
        },
        issues=[
            {
                "code": "PYTHON_IMPORT_ERROR",
                "level": "critical",
                "blocking": True,
                "message": str(exc),
                "package": package,
                "requirements_reference": "SKILL.md#缺依赖处理",
            }
        ],
        next_steps=[
            {
                "action": "read_skill_runtime_requirements",
                "reference": "SKILL.md#缺依赖处理",
                "reason": "Read this skill's dependency section in the installed skill context.",
            }
        ],
        progress=[{"step": phase, "status": "failed", "message": str(exc)}],
    )


def _unexpected_error_result(run_dir: Path, phase: str, exc: Exception) -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="failed",
        summary=f"{phase} failed unexpectedly.",
        input_paths={},
        outputs={"flow_dir": str(run_dir.resolve()), "report_path": None},
        error={"code": "S_LEARNER_MODELING_FAILED", "message": f"{phase} failed unexpectedly.", "recoverable": False, "retryable": False, "raw_error": str(exc)},
        progress=[{"step": phase, "status": "failed", "message": str(exc)}],
    )


def _attach_transport_paths(
    result: dict[str, Any],
    output_dir: Path,
    run_dir: Path,
    result_path: Path,
    experiment_id: str,
) -> None:
    outputs = result.setdefault("outputs", {})
    outputs["experiment_id"] = experiment_id
    outputs["experiment_dir"] = to_run_relative_path(output_dir, run_dir)
    outputs["flow_dir"] = to_run_relative_path(output_dir, run_dir)
    outputs["result_path"] = to_run_relative_path(output_dir, result_path)
    if "modeling_result_path" in outputs and outputs["modeling_result_path"] is None:
        outputs["modeling_result_path"] = to_run_relative_path(output_dir, result_path)
    for step in result.get("next_steps") or []:
        inputs = step.get("inputs")
        if isinstance(inputs, dict):
            if inputs.get("modeling_result_path") is None:
                inputs["modeling_result_path"] = to_run_relative_path(output_dir, result_path)
            if inputs.get("candidate_result_paths") == [None]:
                inputs.pop("candidate_result_paths", None)
                inputs["experiment_ids"] = [experiment_id]


def _write_experiment_records(
    output_dir: Path,
    run_dir: Path,
    action: str,
    result: dict[str, Any],
    result_path: Path,
    experiment_id: str,
) -> None:
    now = _now()
    request_path = run_dir / "inputs" / f"{action}.request.json"
    artifacts = result.get("artifacts") or []
    artifact_paths = [
        str(item.get("path"))
        for item in artifacts
        if isinstance(item, dict) and item.get("path")
    ]
    status = str(result.get("status") or "failed")
    issue_codes = [
        str(item.get("code"))
        for item in result.get("issues") or []
        if isinstance(item, dict) and item.get("code")
    ]
    event = {
        "schema_version": 1,
        "event_type": "action_completed",
        "scope": "experiment",
        "scope_id": experiment_id,
        "experiment_id": experiment_id,
        "skill_name": SKILL_NAME,
        "action": action,
        "status": status,
        "created_at": now,
        "request_path": to_run_relative_path(output_dir, request_path),
        "result_path": to_run_relative_path(output_dir, result_path),
        "artifact_paths": artifact_paths,
        "issue_codes": issue_codes,
    }
    if result.get("outputs", {}).get("report_path"):
        event["report_path"] = result["outputs"]["report_path"]
    append_scope_log(run_dir, event)
    outputs = result.get("outputs") or {}
    inputs = result.get("input_paths") or {}
    manifest = {
        "schema_version": 1,
        "scope": "experiment",
        "scope_id": experiment_id,
        "experiment_id": experiment_id,
        "experiment_type": "model_candidate",
        "learner": outputs.get("learner") or "s_learner",
        "skill_name": SKILL_NAME,
        "latest_action": action,
        "status": status,
        "created_at": now,
        "updated_at": now,
        "inputs": {
            "task_config_path": inputs.get("task_config_path"),
            "modeling_sample_spec_path": inputs.get("modeling_sample_spec_path"),
            "feature_plan_path": inputs.get("feature_plan_path"),
        },
        "outputs": {
            "modeling_result_path": outputs.get("modeling_result_path"),
            "model_candidate_path": outputs.get("model_candidate_path"),
            "evaluation_metrics_path": outputs.get("evaluation_metrics_path"),
            "report_path": outputs.get("report_path"),
        },
        "evaluation_summary": _evaluation_summary(outputs),
        "artifacts": artifacts,
        "log_path": to_run_relative_path(output_dir, run_dir / "_experiment_log.jsonl"),
    }
    write_scope_manifest(run_dir, manifest)


def _evaluation_summary(outputs: dict[str, Any]) -> dict[str, Any]:
    metrics = outputs.get("modeling_metrics") if isinstance(outputs.get("modeling_metrics"), dict) else {}
    splits = metrics.get("splits") if isinstance(metrics.get("splits"), dict) else {}
    test = splits.get("test") if isinstance(splits.get("test"), dict) else {}
    return {
        "primary_dataset_role": "test",
        "primary_metric_name": "auuc_raw",
        "primary_metric_value": test.get("auuc_raw"),
        "evaluation_quality": outputs.get("evaluation_quality"),
    }


def _layout_error_stdout(exc: ProjectLayoutError) -> dict[str, Any]:
    missing = ["experiment_id"] if exc.issue_code == "INVALID_EXPERIMENT_ID" else ["output_dir"]
    return {
        "status": "needs_input",
        "summary": str(exc),
        "outputs": {"missing_fields": missing, "report_path": None},
        "issues": [
            {
                "code": exc.issue_code,
                "level": "critical",
                "blocking": True,
                "message": str(exc),
            }
        ],
        "next_steps": [],
    }


def _stdout_payload(result: dict[str, Any]) -> dict[str, Any]:
    outputs = result.get("outputs") or {}
    return {
        "status": result["status"],
        "summary": result["summary"],
        "outputs": outputs,
        "issues": result.get("issues") or [],
        "next_steps": result.get("next_steps") or [],
    }


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
