from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import csv
import importlib.metadata
import json
import math
import platform
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _common.data_loader import DataLoaderImportError, DataSourceError, load_table
from _common.io import read_cli_json_input, read_json, write_json, write_text
from _common.project_layout import (
    ProjectLayoutError,
    assert_existing_run_dir,
    ensure_existing_skill_call_dir,
    ensure_experiment_dir,
    ensure_skill_call_dir,
    next_action_paths,
    relativize_paths,
    resolve_run_path,
    to_run_relative_path,
    validate_experiment_id,
    write_experiment_action_records,
    write_flow_action_records,
)
from _common.report_language import is_zh
from learner_adapters import AdapterInputError, get_learner_adapter

SKILL_NAME = "uplift-model-tuning"
SKILL_CREATED_BY = "uplift-model-tuning-skill"
TREATMENT_FEATURE = "__uplift_modeling_treatment__"
ZERO_TOLERANCE = 1e-12
AUUC_VERSION = {"package": "scikit-uplift", "version": "0.5.1"}
AUUC_METRIC_METHOD = "sklift.metrics"
ALLOWED_REF_FIELDS = {"data_ref"}
SEARCHED_PARAM_ORDER = ("learning_rate", "num_leaves", "min_child_samples", "reg_lambda")
FIXED_PARAM_PREVIEW_ORDER = (
    "n_estimators",
    "max_depth",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "early_stopping_rounds",
    "n_jobs",
)
DEFAULT_LLM_USAGE = {
    "provider": "unknown",
    "model": "unknown",
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "usage_source": "not_used",
}
MODEL_SPECS = {
    "s_learner": {"model_type": "s_learner", "base_estimators": {"outcome": "lightgbm"}},
    "t_learner": {"model_type": "t_learner", "base_estimators": {"treatment_outcome": "lightgbm", "control_outcome": "lightgbm"}},
}
DEFAULT_SEARCH_SPACE = {
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.15, "scale": "log"},
    "num_leaves": {"type": "categorical", "choices": [7, 15, 31, 63]},
    "min_child_samples": {"type": "int", "low": 10, "high": 100},
    "reg_lambda": {"type": "float", "low": 0.0, "high": 5.0, "scale": "linear"},
}
PROFILES = {
    "quick": {
        "max_duration_minutes": 15,
        "max_trials": 10,
        "per_trial_timeout_minutes": 5,
        "min_completed_trials": 3,
    },
    "standard": {
        "max_duration_minutes": 60,
        "max_trials": 50,
        "per_trial_timeout_minutes": 15,
        "min_completed_trials": 10,
    },
    "thorough": {
        "max_duration_minutes": 240,
        "max_trials": 200,
        "per_trial_timeout_minutes": 30,
        "min_completed_trials": 30,
    },
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
    "n_jobs": 1,
    "early_stopping_rounds": 50,
    "deterministic": True,
    "force_col_wise": True,
}
ALLOWED_PARAMETERS = set(DEFAULT_PARAMETERS)


class MissingInputError(Exception):
    def __init__(self, message: str, *, missing_fields: list[str] | None = None, code: str = "MISSING_OR_INVALID_INPUT") -> None:
        super().__init__(message)
        self.missing_fields = missing_fields or []
        self.code = code


class NeedsConfirmationError(Exception):
    pass


class PackageImportError(Exception):
    def __init__(self, package: str, message: str) -> None:
        super().__init__(message)
        self.package = package


def run_draft_plan(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    _validate_parameter_strategy_payload(payload)
    experiment_id = validate_experiment_id(str(payload.get("experiment_id") or ""))
    inputs = _load_tuning_inputs(output_dir, payload)
    preferences = payload.get("preferences") or {}
    if not isinstance(preferences, dict):
        raise MissingInputError("preferences must be an object.")
    profile = str(preferences.get("profile") or "standard").lower()
    if profile not in {*PROFILES, "custom"}:
        raise MissingInputError("preferences.profile must be quick, standard, thorough, or custom.")
    dataset_paths = inputs["dataset_paths"]
    selection_role = str(
        preferences.get("selection_dataset_role")
        or payload.get("selection_dataset_role")
        or ("validation" if dataset_paths.get("valid") else "test")
    ).lower()
    selection_key = _split_key(selection_role)
    if selection_key not in dataset_paths or not dataset_paths.get(selection_key):
        raise MissingInputError(
            f"selection_dataset_role is not available: {selection_role}",
            missing_fields=["preferences.selection_dataset_role"],
        )
    _require_baseline_auuc_version(inputs, selection_key)
    budget = dict(PROFILES.get(profile, PROFILES["standard"]))
    budget.update(preferences.get("budget") or {})
    _validate_budget(budget)
    budget["profile"] = profile
    budget["max_parallel_trials"] = 1

    baseline_parameters = dict(inputs["baseline_parameters"])
    fixed = {**DEFAULT_PARAMETERS, **baseline_parameters}
    fixed.update(preferences.get("fixed_parameters") or preferences.get("fixed_params") or {})
    _validate_parameters(fixed)
    search_space = _merge_search_space(preferences.get("search_space_overrides") or {})
    for name in list(search_space):
        if name in (preferences.get("fixed_parameters") or preferences.get("fixed_params") or {}):
            search_space.pop(name)
    candidate_grid = preferences.get("candidate_grid") or []
    if candidate_grid and not isinstance(candidate_grid, list):
        raise MissingInputError("preferences.candidate_grid must be a list when provided.")
    for item in candidate_grid:
        if not isinstance(item, dict):
            raise MissingInputError("Each candidate_grid item must be an object.")
        _validate_candidate_override(item)
    warnings = []
    if selection_role in {"test", "oot"}:
        warnings.append(
            {
                "code": "HOLDOUT_USED_FOR_SELECTION",
                "acknowledged": False,
                "message": "Test/OOT holdout is used for tuning selection.",
            }
        )
        if bool(fixed.get("early_stopping_rounds")):
            warnings.append(
                {
                    "code": "HOLDOUT_USED_FOR_EARLY_STOPPING",
                    "acknowledged": False,
                    "message": "The same holdout may be used for early stopping.",
                }
            )
    if profile == "thorough":
        warnings.append(
            {
                "code": "HIGH_COST_RANDOM_SEARCH",
                "acknowledged": False,
                "message": "Thorough tuning can be costly.",
            }
        )
    plan = {
        "artifact_kind": "model_tuning_plan",
        "artifact_status": "draft",
        "artifact_version": 1,
        "status": "draft",
        "created_at": _now(),
        "created_by": SKILL_CREATED_BY,
        "experiment_id": experiment_id,
        "source_experiment_id": (inputs["modeling_result"].get("outputs") or {}).get("experiment_id"),
        "source_modeling_result_path": inputs["input_paths"]["modeling_result_path"],
        "input_paths": inputs["input_paths"],
        "model_spec": inputs["model_spec"],
        "parameter_strategy": "shared",
        "selection": {
            "dataset_role": selection_role,
            "dataset_key": selection_key,
            "dataset_path": dataset_paths[selection_key],
            "population_fingerprint": _file_fingerprint(Path(dataset_paths[selection_key])),
            "metric_name": "auuc_raw",
            "metric_direction": "higher_is_better",
            "metric_method": AUUC_METRIC_METHOD,
            "metric_version": AUUC_VERSION["version"],
            "auuc_version": dict(AUUC_VERSION),
            "early_stopping_dataset_role": "validation" if dataset_paths.get("valid") else selection_role,
            "holdout_warnings": warnings,
        },
        "search": {
            "strategy": "random",
            "search_space": search_space,
            "fixed_parameters": fixed,
            "candidate_grid": candidate_grid,
        },
        "budget": budget,
        "reproducibility": {
            "sampler_seed": int(preferences.get("sampler_seed", preferences.get("random_seed", 42))),
            "model_seed": int(preferences.get("model_seed", fixed.get("random_state", 42))),
        },
        "requested_configuration": {
            "profile": profile,
            "budget": preferences.get("budget") or {},
            "fixed_parameters": preferences.get("fixed_parameters")
            or preferences.get("fixed_params")
            or {},
            "search_space_overrides": preferences.get("search_space_overrides") or {},
            "candidate_grid_count": len(candidate_grid),
        },
        "confirmation": {"required": True, "confirmed": False},
    }
    plan_path = run_dir / "artifacts" / "tuning_plan.draft.v1.json"
    write_json(plan_path, relativize_paths(plan, output_dir))
    return _result(
        run_dir=run_dir,
        phase="draft_plan",
        status="needs_confirmation",
        summary="A bounded random-search tuning plan is ready for confirmation.",
        input_paths=inputs["input_paths"],
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "experiment_id": experiment_id,
            "source_experiment_id": plan.get("source_experiment_id"),
            "source_modeling_result_path": inputs["input_paths"]["modeling_result_path"],
            "tuning_plan_path": str(plan_path.resolve()),
            "warning_acknowledgements_required": [item["code"] for item in warnings],
            "selection_dataset_role": selection_role,
            "max_trials": budget["max_trials"],
            "candidate_count": len(candidate_grid) if candidate_grid else int(budget["max_trials"]),
            "learner": inputs["model_spec"].get("model_type"),
        },
        issues=[
            {
                "code": item["code"],
                "level": "warning",
                "blocking": True,
                "message": item["message"],
                "suggested_fix": "Review and confirm the tuning plan before execute.",
            }
            for item in warnings
        ],
        artifacts=[{"kind": "tuning_plan_draft", "path": str(plan_path.resolve())}],
        progress=[
            {"step": "input_validation", "status": "success"},
            {"step": "plan_generation", "status": "success"},
        ],
        next_steps=[
            {
                "skill": SKILL_NAME,
                "action": "confirm_plan",
                "reason": "Confirm the bounded tuning plan before training candidate trials.",
                "inputs": {"flow_dir": str(run_dir.resolve()), "tuning_plan_path": str(plan_path.resolve())},
                "requires_user_confirmation": True,
            }
        ],
        interaction_type="confirmation",
    )


def run_confirm_plan(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    source_path = _required_input_path(
        output_dir,
        payload.get("source_plan_path") or payload.get("tuning_plan_path"),
        "source_plan_path",
    )
    plan = read_json(source_path)
    if plan.get("artifact_kind") != "model_tuning_plan" or plan.get("status") != "draft":
        raise MissingInputError("source_plan_path must point to a draft model tuning plan.")
    experiment_id = validate_experiment_id(str(plan.get("experiment_id") or ""))
    requested_experiment_id = payload.get("experiment_id")
    if requested_experiment_id and validate_experiment_id(str(requested_experiment_id)) != experiment_id:
        raise MissingInputError(
            "payload.experiment_id must match tuning plan experiment_id.",
            missing_fields=["experiment_id"],
            code="INVALID_EXPERIMENT_ID",
        )
    _validate_parameter_strategy_payload(plan)
    acknowledgements = set(str(item) for item in payload.get("warning_acknowledgements") or [])
    required = {
        str(item.get("code"))
        for item in plan.get("selection", {}).get("holdout_warnings", [])
        if item.get("code")
    }
    missing = sorted(required - acknowledgements)
    if missing:
        raise NeedsConfirmationError(f"Required warning acknowledgements are missing: {missing}")
    confirmed = json.loads(json.dumps(plan))
    confirmed["artifact_status"] = "confirmed"
    confirmed["status"] = "confirmed"
    confirmed["confirmed_by"] = str(payload.get("confirmed_by") or "user")
    confirmed["confirmed_at"] = _now()
    confirmed["source_plan_path"] = str(source_path)
    for item in confirmed.get("selection", {}).get("holdout_warnings", []):
        item["acknowledged"] = str(item.get("code")) in acknowledgements
    confirmed["confirmation"] = {
        "required": True,
        "confirmed": True,
        "confirmed_by": confirmed["confirmed_by"],
        "confirmed_at": confirmed["confirmed_at"],
        "source_plan_path": str(source_path),
        "warning_acknowledgements": sorted(acknowledgements),
    }
    plan_path = run_dir / "artifacts" / "tuning_plan.confirmed.v1.json"
    write_json(plan_path, relativize_paths(confirmed, output_dir))
    return _result(
        run_dir=run_dir,
        phase="confirm_plan",
        status="success",
        summary="The tuning plan is confirmed and ready to execute.",
        input_paths={"source_plan_path": str(source_path)},
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "experiment_id": experiment_id,
            "source_experiment_id": confirmed.get("source_experiment_id"),
            "source_modeling_result_path": confirmed.get("source_modeling_result_path"),
            "tuning_plan_path": str(plan_path.resolve()),
        },
        artifacts=[{"kind": "tuning_plan_confirmed", "path": str(plan_path.resolve())}],
        progress=[
            {"step": "warning_acknowledgement", "status": "success"},
            {"step": "plan_confirmation", "status": "success"},
        ],
        next_steps=[
            {
                "skill": SKILL_NAME,
                "action": "execute",
                "reason": "Run the confirmed bounded tuning study.",
                "inputs": {"tuning_plan_path": str(plan_path.resolve())},
                "requires_user_confirmation": False,
            }
        ],
    )


def run_execute(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    packages = _required_packages()
    plan_path = _required_input_path(output_dir, payload.get("tuning_plan_path"), "tuning_plan_path")
    plan = read_json(plan_path)
    if (
        plan.get("artifact_kind") != "model_tuning_plan"
        or plan.get("status") != "confirmed"
        or not plan.get("confirmation", {}).get("confirmed")
    ):
        raise MissingInputError("execute requires a confirmed tuning_plan_path.")
    experiment_id = validate_experiment_id(str(plan.get("experiment_id") or ""))
    _validate_parameter_strategy_payload(plan)
    inputs = _load_tuning_inputs(output_dir, plan.get("input_paths") or {})
    if _canonical_model_spec(plan.get("model_spec")) != _canonical_model_spec(inputs["model_spec"]):
        raise MissingInputError(
            "tuning plan model_spec does not match the baseline modeling result.",
            code="MODEL_SPEC_MISMATCH",
        )
    adapter = get_learner_adapter(str(inputs["model_spec"].get("model_type")))
    frames = _load_frames(inputs)
    encoder = adapter.fit_encoder(packages, frames["train"], inputs["selected_features"])
    trial_metrics, best_trial, summary = _run_trials(
        adapter,
        packages,
        run_dir,
        plan,
        inputs,
        frames,
        encoder,
    )
    winner = _materialize_winner(adapter, packages, run_dir, output_dir, plan, inputs, frames, encoder, best_trial, summary)
    trial_metrics_path = run_dir / "artifacts" / "trial_metrics.v1.json"
    leaderboard_path = run_dir / "artifacts" / "leaderboard.v1.csv"
    write_json(trial_metrics_path, {"artifact_kind": "trial_metrics", "artifact_version": 1, "trials": trial_metrics})
    _write_leaderboard(leaderboard_path, trial_metrics)
    report_path = _write_tuning_report(
        run_dir,
        plan=plan,
        trial_metrics=trial_metrics,
        winner=winner,
        summary=summary,
        issues=winner["issues"],
        language=_language_from_task_config(inputs["task_config"]),
    )
    tuning_report_path = run_dir / "artifacts" / "tuning_report.md"
    write_text(tuning_report_path, report_path.read_text(encoding="utf-8"))
    outputs = {
        "experiment_id": experiment_id,
        "experiment_dir": str(run_dir.resolve()),
        "flow_dir": str(run_dir.resolve()),
        "source_experiment_id": plan.get("source_experiment_id"),
        "source_modeling_result_path": inputs["input_paths"]["modeling_result_path"],
        "tuning_flow_dir": _flow_dir_from_plan_path(output_dir, plan_path),
        "tuning_plan_path": str(plan_path.resolve()),
        "trial_metrics_path": str(trial_metrics_path.resolve()),
        "leaderboard_path": str(leaderboard_path.resolve()),
        "tuning_result_path": None,
        "winner_model_result_path": winner["winner_model_result_path"],
        "winner_result_path": winner["winner_model_result_path"],
        "winner_metrics": winner["winner_metrics"],
        "winner_metrics_path": winner["winner_metrics_path"],
        "model_candidate_path": winner["model_candidate_path"],
        "model_artifact_path": winner["model_artifact_path"],
        "evaluation_metrics_path": winner["winner_metrics_path"],
        "split_evaluations": winner["split_evaluations"],
        "learner": inputs["model_spec"].get("model_type"),
        "tuning_summary": summary,
        "holdout_warnings": plan.get("selection", {}).get("holdout_warnings", []),
        "report_path": str(report_path.resolve()),
        "tuning_report_path": str(tuning_report_path.resolve()),
    }
    status = "success" if summary["completed_random_trials"] >= int(plan["budget"]["min_completed_trials"]) else "partial_success"
    issues = list(winner["issues"])
    if status == "partial_success":
        issues.append(
            {
                "code": "INSUFFICIENT_SEARCH",
                "level": "warning",
                "blocking": False,
                "message": "The study returned a usable winner without meeting min_completed_trials.",
                "suggested_fix": "Increase max_trials or review failed candidate settings.",
            }
        )
    return _result(
        run_dir=run_dir,
        phase="execute",
        status=status,
        summary="The bounded tuning study completed with a usable winner.",
        input_paths={"tuning_plan_path": str(plan_path.resolve()), **inputs["input_paths"]},
        outputs=outputs,
        issues=issues,
        artifacts=[
            {"kind": "trial_metrics", "path": str(trial_metrics_path.resolve())},
            {"kind": "leaderboard", "path": str(leaderboard_path.resolve())},
            {"kind": "winner_metrics", "path": winner["winner_metrics_path"]},
            {"kind": "winner_model_result", "path": winner["winner_model_result_path"]},
            {"kind": "winner_score_frame", "path": winner["score_frame_path"]},
            {"kind": "winner_uplift_bins", "path": winner["uplift_bins_path"]},
            *[
                {"kind": kind, "path": path}
                for split_paths in (winner.get("curve_paths") or {}).values()
                for kind, path in (
                    ("winner_auuc_curve_raw", split_paths["raw_curve_path"]),
                    ("winner_auuc_curve_normalized", split_paths["normalized_curve_path"]),
                )
            ],
            {"kind": "model_candidate", "path": winner["model_candidate_path"]},
            {"kind": "report", "path": str(report_path.resolve())},
            {"kind": "tuning_report", "path": str(tuning_report_path.resolve())},
        ],
        progress=[
            {"step": "input_validation", "status": "success"},
            {"step": "random_search", "status": status},
            {"step": "winner_materialization", "status": "success"},
            {"step": "artifact_generation", "status": "success"},
        ],
        next_steps=[
            {
                "skill": "uplift-model-result-comparison",
                "action": "draft_comparison_plan",
                "reason": "Draft a comparison plan including this tuning winner and other experiment candidates.",
                "inputs": {
                    "experiment_ids": [
                        item for item in [plan.get("source_experiment_id"), experiment_id] if item
                    ]
                },
                "requires_user_confirmation": True,
            }
        ],
    )


def _run_trials(
    adapter: Any,
    packages: dict[str, Any],
    run_dir: Path,
    plan: dict[str, Any],
    inputs: dict[str, Any],
    frames: dict[str, Any],
    encoder: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    deadline = started + float(plan["budget"]["max_duration_minutes"]) * 60
    selection_key = str(plan["selection"]["dataset_key"])
    baseline = _baseline_trial(packages, plan, inputs, frames, selection_key)
    trials = [baseline]
    completed_random = 0
    failed_random = 0
    attempted_random = 0
    stop_reason = "max_trials"
    for index, parameters in enumerate(_candidate_parameters(plan), start=1):
        if time.monotonic() >= deadline:
            stop_reason = "study_budget_exhausted"
            break
        trial_id = f"trial_{index:03d}"
        attempted_random += 1
        trial_started = time.monotonic()
        try:
            fitted = adapter.fit_trial(packages, inputs, frames, encoder, parameters)
            best_iteration = fitted.get("best_iteration")
            score = adapter.score_frame(
                packages,
                fitted,
                inputs,
                frames[selection_key],
                selection_key,
                encoder,
                best_iteration,
            )
            metrics = _metric_from_score(packages, score, selection_key, inputs["dataset_paths"][selection_key])
            trial = {
                "trial_id": trial_id,
                "status": "completed",
                "parameters": parameters,
                "duration_seconds": time.monotonic() - trial_started,
                "best_iteration": best_iteration,
                "validation_metric_summary": fitted.get("validation_metric_summary"),
                "dataset_role": plan["selection"]["dataset_role"],
                **_trial_metric_fields(metrics),
                "_model": fitted,
            }
            completed_random += 1
        except Exception as exc:  # noqa: BLE001
            failed_random += 1
            trial = {
                "trial_id": trial_id,
                "status": "failed_training",
                "parameters": parameters,
                "duration_seconds": time.monotonic() - trial_started,
                "dataset_role": plan["selection"]["dataset_role"],
                "auuc_raw": None,
                "auuc_normalized": None,
                "objective_eligible": False,
                "eligibility_reason": str(exc),
            }
        trials.append(trial)
    comparable = [trial for trial in trials if trial.get("objective_eligible")]
    if not comparable:
        raise MissingInputError("No objective-eligible trial was produced.")
    best = max(comparable, key=lambda item: float(item["auuc_raw"]))
    if best["trial_id"] == "trial_000":
        random_eligible = [trial for trial in trials[1:] if trial.get("objective_eligible")]
        winner_reason = "no_valid_tuning_trial" if not random_eligible else "no_improvement"
    else:
        winner_reason = "improved"
    clean_trials = [{key: value for key, value in trial.items() if key not in {"_estimator", "_model"}} for trial in trials]
    summary = {
        "baseline_trial_id": "trial_000",
        "winner_trial_id": best["trial_id"],
        "winner_reason": winner_reason,
        "stop_reason": stop_reason,
        "attempted_random_trials": attempted_random,
        "completed_random_trials": completed_random,
        "failed_random_trials": failed_random,
        "max_trials": int(plan["budget"]["max_trials"]),
        "candidate_count": attempted_random,
        "sampler_seed": int(plan["reproducibility"]["sampler_seed"]),
        "model_seed": int(plan["reproducibility"]["model_seed"]),
        "max_duration_minutes": float(plan["budget"]["max_duration_minutes"]),
        "per_trial_timeout_minutes": float(plan["budget"]["per_trial_timeout_minutes"]),
    }
    return clean_trials, best, summary


def _baseline_trial(
    packages: dict[str, Any],
    plan: dict[str, Any],
    inputs: dict[str, Any],
    frames: dict[str, Any],
    selection_key: str,
) -> dict[str, Any]:
    outputs = inputs["modeling_result"].get("outputs", {})
    direct = (outputs.get("split_evaluations") or {}).get(selection_key)
    if not isinstance(direct, dict):
        raise MissingInputError(f"Baseline result must expose outputs.split_evaluations.{selection_key}. Rerun baseline modeling with scikit-uplift==0.5.1.")
    _require_auuc_version(direct, f"baseline outputs.split_evaluations.{selection_key}")
    primary = direct
    return {
        "trial_id": "trial_000",
        "status": "reused_baseline",
        "parameters": inputs["baseline_parameters"],
        "duration_seconds": 0.0,
        "best_iteration": inputs["baseline_best_iteration"],
        "dataset_role": plan["selection"]["dataset_role"],
        **_trial_metric_fields(primary),
        "eligibility_reason": None,
    }


def _require_baseline_auuc_version(inputs: dict[str, Any], selection_key: str) -> None:
    outputs = inputs["modeling_result"].get("outputs", {})
    split_eval = (outputs.get("split_evaluations") or {}).get(selection_key)
    if not isinstance(split_eval, dict):
        raise MissingInputError(f"Baseline result must expose outputs.split_evaluations.{selection_key}. Rerun baseline modeling with scikit-uplift==0.5.1.")
    _require_auuc_version(split_eval, f"baseline outputs.split_evaluations.{selection_key}")


def _require_auuc_version(split_eval: dict[str, Any], context: str) -> None:
    actual = split_eval.get("auuc_version")
    if actual != AUUC_VERSION:
        raise MissingInputError(
            f"{context}.auuc_version must be {AUUC_VERSION}; got {actual}. Rerun baseline modeling with scikit-uplift==0.5.1.",
            code="AUUC_VERSION_MISMATCH",
        )


def _materialize_winner(
    adapter: Any,
    packages: dict[str, Any],
    run_dir: Path,
    output_dir: Path,
    plan: dict[str, Any],
    inputs: dict[str, Any],
    frames: dict[str, Any],
    encoder: dict[str, Any],
    best_trial: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if best_trial["trial_id"] == "trial_000":
        model_path = inputs["baseline_model_path"]
        model_metadata_path = inputs["baseline_model_metadata_path"]
        model = adapter.load_model_object(packages, model_path)
        winner_encoder = model["encoder"]
        best_iteration = model.get("best_iteration") or inputs["baseline_best_iteration"]
        effective_parameters = inputs["baseline_parameters"]
    else:
        model_path = run_dir / "artifacts" / adapter.artifact_names["joblib"]
        model_metadata_path = run_dir / "artifacts" / "tuned_model_metadata.v1.json"
        model = best_trial["_model"]
        winner_encoder = model.get("encoder") or encoder
        best_iteration = best_trial.get("best_iteration")
        effective_parameters = best_trial["parameters"]
        adapter.dump_model(packages, model_path, model, inputs, effective_parameters, best_iteration)
        adapter.write_model_metadata(model_metadata_path, model, {**inputs, "created_at": _now()}, effective_parameters, best_iteration)
    score_frames = []
    bin_tables = []
    curve_paths: dict[str, dict[str, str]] = {}
    split_evaluations = {}
    for split in ("train", "valid", "test", "oot"):
        frame = frames.get(split)
        dataset_path = inputs["dataset_paths"].get(split)
        if frame is None or not dataset_path:
            continue
        score = adapter.score_frame(packages, model, inputs, frame, split, winner_encoder, best_iteration)
        score_frames.append(score)
        bins, bin_warnings = _build_uplift_bins(packages, score, split=split)
        bin_tables.append(bins)
        metric = _metric_from_score(packages, score, split, dataset_path)
        raw_curve_path = run_dir / "artifacts" / f"winner_{split}_auuc_curve_raw.v1.csv"
        normalized_curve_path = run_dir / "artifacts" / f"winner_{split}_auuc_curve_normalized.v1.csv"
        _build_auuc_curve(packages, score, normalized=False).to_csv(raw_curve_path, index=False)
        _build_auuc_curve(packages, score, normalized=True).to_csv(normalized_curve_path, index=False)
        curve_paths[split] = {
            "raw_curve_path": str(raw_curve_path.resolve()),
            "normalized_curve_path": str(normalized_curve_path.resolve()),
        }
        metric.update(curve_paths[split])
        split_evaluations[split] = metric
        support = metric.get("support") or {}
        if not support.get("treatment_count") or not support.get("control_count"):
            issues.append(
                {
                    "code": "INSUFFICIENT_EVALUATION_ARM_SUPPORT",
                    "level": "warning",
                    "blocking": False,
                    "message": f"{split}: treatment/control arm support is required for AUUC.",
                    "suggested_fix": "Prepare evaluation splits with both treatment and control arms.",
                }
            )
        for warning in bin_warnings:
            issues.append(
                {
                    "code": "EVALUATION_WARNING",
                    "level": "warning",
                    "blocking": False,
                    "message": f"{split}: {warning}",
                    "suggested_fix": "Review split support and detailed evaluation artifacts.",
                }
            )
    pd = packages["pandas"]
    score_frame = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    uplift_bins = pd.concat(bin_tables, ignore_index=True) if bin_tables else pd.DataFrame()
    score_frame_path = run_dir / "artifacts" / "winner_score_frame.v1.csv"
    uplift_bins_path = run_dir / "artifacts" / "winner_uplift_bins.v1.csv"
    winner_metrics_path = run_dir / "artifacts" / "winner_metrics.v1.json"
    model_candidate_path = run_dir / "artifacts" / "model_candidate.v1.json"
    winner_result_path = run_dir / "artifacts" / "winner_model_result.v1.json"
    score_frame.to_csv(score_frame_path, index=False)
    uplift_bins.to_csv(uplift_bins_path, index=False)
    selection_key = str(plan["selection"]["dataset_key"])
    primary = split_evaluations.get(selection_key) or split_evaluations.get("test") or {}
    winner_metrics = {
        "artifact_kind": "winner_metrics",
        "artifact_version": 1,
        "trial_id": best_trial["trial_id"],
        "dataset_role": plan["selection"]["dataset_role"],
        "auuc_raw": primary.get("auuc_raw") or primary.get("metric_value"),
        "auuc_normalized": primary.get("auuc_normalized"),
        "auuc_version": primary.get("auuc_version"),
        "row_count": (primary.get("support") or {}).get("row_count"),
        "treatment_count": (primary.get("support") or {}).get("treatment_count"),
        "control_count": (primary.get("support") or {}).get("control_count"),
        "valid_bin_count": (primary.get("support") or {}).get("valid_bin_count"),
        "raw_curve_path": primary.get("raw_curve_path"),
        "normalized_curve_path": primary.get("normalized_curve_path"),
        "best_iteration": best_iteration,
        "evaluation_quality": _evaluation_quality(primary.get("support") or {}),
    }
    model_candidate = {
        "artifact_kind": "model_candidate",
        "artifact_version": 1,
        "candidate_id": f"{run_dir.name}:{best_trial['trial_id']}",
        "producer_kind": "tuning",
        "model_spec": plan.get("model_spec") or inputs["model_spec"],
        "parameter_strategy": str(plan.get("parameter_strategy") or "shared"),
        "model_artifact_path": str(Path(model_path).resolve()),
        "model_metadata_path": str(Path(model_metadata_path).resolve()) if model_metadata_path else None,
        "primary_evaluation": {
            "dataset_role": primary.get("dataset_role"),
            "metric_name": primary.get("metric_name"),
            "metric_value": primary.get("metric_value"),
            "metric_direction": primary.get("metric_direction"),
            "metric_method": primary.get("metric_method"),
            "metric_version": primary.get("metric_version"),
            "auuc_version": primary.get("auuc_version"),
            "metrics_path": str(winner_metrics_path.resolve()),
        },
        "lineage": dict(inputs["input_paths"]),
        "base_candidate_result_path": inputs["input_paths"]["modeling_result_path"],
        "risk_flags": [
            str(item.get("code"))
            for item in plan.get("selection", {}).get("holdout_warnings", [])
            if item.get("acknowledged")
        ],
        "winner_reason": summary["winner_reason"],
    }
    write_json(winner_metrics_path, relativize_paths(winner_metrics, output_dir))
    write_json(model_candidate_path, relativize_paths(model_candidate, output_dir))
    write_json(
        winner_result_path,
        relativize_paths({
            "artifact_kind": "winner_model_result",
            "artifact_version": 1,
            "model_candidate": model_candidate,
            "winner_metrics": winner_metrics,
            "split_evaluations": split_evaluations,
            "tuning_summary": summary,
            "score_frame_path": str(score_frame_path.resolve()),
            "uplift_bins_path": str(uplift_bins_path.resolve()),
            "curve_paths": curve_paths,
            "created_at": _now(),
        }, output_dir),
    )
    return {
        "issues": issues,
        "model_artifact_path": str(Path(model_path).resolve()),
        "model_candidate": model_candidate,
        "model_candidate_path": str(model_candidate_path.resolve()),
        "winner_metrics": winner_metrics,
        "winner_metrics_path": str(winner_metrics_path.resolve()),
        "winner_model_result_path": str(winner_result_path.resolve()),
        "split_evaluations": split_evaluations,
        "score_frame_path": str(score_frame_path.resolve()),
        "uplift_bins_path": str(uplift_bins_path.resolve()),
        "curve_paths": curve_paths,
    }


def _candidate_parameters(plan: dict[str, Any]) -> list[dict[str, Any]]:
    fixed = dict(plan["search"]["fixed_parameters"])
    grid = plan["search"].get("candidate_grid") or []
    max_trials = int(plan["budget"]["max_trials"])
    seed = int(plan["reproducibility"]["sampler_seed"])
    model_seed = int(plan["reproducibility"]["model_seed"])
    candidates: list[dict[str, Any]] = []
    for item in grid[:max_trials]:
        params = {**fixed, **item}
        params["random_state"] = model_seed
        _validate_parameters(params)
        candidates.append(params)
    rng = random.Random(seed)
    while len(candidates) < max_trials:
        suggested = {
            name: _suggest_value(spec, rng)
            for name, spec in (plan["search"].get("search_space") or {}).items()
        }
        params = {**fixed, **suggested}
        params["random_state"] = model_seed + len(candidates) + 1
        _validate_parameters(params)
        candidates.append(params)
    return candidates


def _candidate_payload(output_dir: Path, outputs: dict[str, Any]) -> dict[str, Any]:
    path = outputs.get("model_candidate_path")
    if path:
        return read_json(resolve_run_path(output_dir, str(path)))
    candidate = outputs.get("model_candidate")
    if isinstance(candidate, dict):
        return candidate
    raise MissingInputError("modeling_result_path does not expose model_candidate or model_candidate_path.")


def _canonical_model_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        value = {}
    model_type = str(value.get("model_type") or "")
    if model_type in MODEL_SPECS:
        return json.loads(json.dumps(MODEL_SPECS[model_type], sort_keys=True))
    return {"model_type": model_type, "base_estimators": value.get("base_estimators") or {}}


def _load_tuning_inputs(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    task_path = _required_input_path(output_dir, payload.get("task_config_path"), "task_config_path")
    sample_path = _required_input_path(output_dir, payload.get("modeling_sample_spec_path"), "modeling_sample_spec_path")
    feature_path = _required_input_path(output_dir, payload.get("feature_plan_path"), "feature_plan_path")
    modeling_path = _required_input_path(output_dir, payload.get("modeling_result_path"), "modeling_result_path")
    task = _read_artifact(task_path, artifact_kind="task_config", require_confirmed=True)
    sample = _read_artifact(sample_path, artifact_kind="modeling_sample_spec", require_confirmed=True)
    feature = _read_artifact(feature_path, artifact_kind="feature_plan", require_confirmed=True)
    modeling_result = read_json(modeling_path)
    if modeling_result.get("skill_name") not in {"uplift-model-s-learner-modeling", "uplift-model-t-learner-modeling"} or modeling_result.get("status") != "success":
        raise MissingInputError("modeling_result_path must point to a successful S-Learner or T-Learner result.")
    input_paths = {
        "task_config_path": str(task_path),
        "modeling_sample_spec_path": str(sample_path),
        "feature_plan_path": str(feature_path),
        "modeling_result_path": str(modeling_path),
    }
    modeling_inputs = modeling_result.get("input_paths") or {}
    for key in ("task_config_path", "modeling_sample_spec_path", "feature_plan_path"):
        if not _same_path(_resolve_input_path(output_dir, str(modeling_inputs.get(key))), input_paths[key]):
            raise MissingInputError(f"modeling_result lineage mismatch for {key}.")
    sample_payload = sample.get("payload") or {}
    if sample_payload.get("is_valid") is not True:
        raise MissingInputError("modeling_sample_spec.payload.is_valid must be true.")
    selected = _selected_features_from_plan(feature, output_dir)
    outputs = modeling_result.get("outputs") or {}
    model_candidate = _candidate_payload(output_dir, outputs)
    model_spec = _canonical_model_spec(model_candidate.get("model_spec") or {"model_type": outputs.get("learner")})
    if model_spec.get("model_type") not in MODEL_SPECS:
        raise MissingInputError("model_candidate.model_spec.model_type must be s_learner or t_learner.")
    metadata_path = outputs.get("model_metadata_path")
    model_metadata = read_json(_resolve_input_path(output_dir, str(metadata_path))) if metadata_path else {}
    baseline_model_path = outputs.get("model_artifact_path") or model_metadata.get("model_artifact_path")
    if not baseline_model_path:
        raise MissingInputError("modeling_result_path does not expose model_artifact_path.")
    columns = sample_payload.get("columns") or {}
    datasets = sample_payload.get("datasets") or {}
    return {
        "input_paths": input_paths,
        "task_config": task,
        "sample_spec": sample,
        "feature_plan": feature,
        "modeling_result": modeling_result,
        "model_spec": model_spec,
        "selected_features": selected,
        "dataset_paths": {name: datasets.get(name) for name in ("train", "valid", "test", "oot")},
        "treatment_column": str(columns.get("treatment") or "__uplift_modeling_treatment__"),
        "outcome_column": str(columns.get("outcome") or "__uplift_modeling_outcome__"),
        "outcome_type": str(task.get("payload", {}).get("outcome_type") or columns.get("outcome_type")),
        "unit_id_column": task.get("payload", {}).get("unit_id_column"),
        "baseline_model_path": _resolve_input_path(output_dir, str(baseline_model_path)),
        "baseline_model_metadata_path": _resolve_input_path(output_dir, str(metadata_path)) if metadata_path else None,
        "baseline_parameters": model_metadata.get("effective_parameters") or DEFAULT_PARAMETERS,
        "baseline_best_iteration": model_metadata.get("best_iteration"),
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
    if "train" not in frames or "test" not in frames:
        raise MissingInputError("train and test datasets are required for tuning.")
    return frames


def _selected_features_from_plan(feature_plan: dict[str, Any], output_dir: Path) -> list[str]:
    selected = feature_plan.get("payload", {}).get("selected_features", {})
    if selected.get("features"):
        return [str(item) for item in selected.get("features") if str(item)]
    if selected.get("features_path"):
        payload = read_json(resolve_run_path(output_dir, selected["features_path"]))
        return [str(item) for item in payload.get("features", []) if str(item)]
    raise MissingInputError("feature_plan is missing selected features.")


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
            model_feature_order.extend([f"{feature}={category}" for category in categories])
            model_feature_order.append(f"{feature}=__MISSING__")
            model_feature_order.append(f"{feature}=__OTHER__")
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


def _metric_from_score(packages: dict[str, Any], score_frame: Any, split: str, dataset_path: str) -> dict[str, Any]:
    auuc = _evaluate_auuc(packages, score_frame)
    bins, _ = _build_uplift_bins(packages, score_frame, split=split)
    support = {
        "row_count": int(len(score_frame)),
        "treatment_count": int((score_frame["actual_treatment"] == 1).sum()) if not score_frame.empty else 0,
        "control_count": int((score_frame["actual_treatment"] == 0).sum()) if not score_frame.empty else 0,
        "valid_bin_count": int((bins["support_status"] == "ok").sum()) if not bins.empty else 0,
    }
    return {
        "dataset_role": split,
        "population_fingerprint": _file_fingerprint(Path(dataset_path)),
        "metric_name": "auuc_raw",
        "metric_value": auuc["raw"],
        "metric_direction": "higher_is_better",
        "metric_method": AUUC_METRIC_METHOD,
        "metric_version": AUUC_VERSION["version"],
        "auuc_version": dict(AUUC_VERSION),
        "auuc_raw": auuc["raw"],
        "auuc_normalized": auuc["normalized"],
        "support": support,
    }


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
        "support": {
            "row_count": metric.get("row_count"),
            "treatment_count": metric.get("treatment_count"),
            "control_count": metric.get("control_count"),
            "valid_bin_count": metric.get("valid_bin_count"),
        },
    }


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


def _sklift_auuc_inputs(score_frame: Any) -> tuple[Any, Any, Any]:
    return (
        score_frame["actual_outcome"].astype(float).to_numpy(),
        score_frame["uplift_score"].astype(float).to_numpy(),
        score_frame["actual_treatment"].astype(int).to_numpy(),
    )


def _write_tuning_report(
    run_dir: Path,
    *,
    plan: dict[str, Any],
    trial_metrics: list[dict[str, Any]],
    winner: dict[str, Any],
    summary: dict[str, Any],
    issues: list[dict[str, Any]],
    language: str,
) -> Path:
    path = run_dir / "report.md"
    zh = is_zh(language)
    baseline = next((trial for trial in trial_metrics if trial.get("trial_id") == "trial_000"), None)
    winner_trial = next((trial for trial in trial_metrics if trial.get("trial_id") == summary["winner_trial_id"]), None)
    delta = _number(winner_trial.get("auuc_raw") if winner_trial else None) - _number(baseline.get("auuc_raw") if baseline else None)
    if not math.isfinite(delta):
        delta = math.nan
    relative = delta / abs(_number(baseline.get("auuc_raw"))) if baseline and _number(baseline.get("auuc_raw")) not in (0.0, math.nan) and math.isfinite(delta) else math.nan
    lines = ["# 模型调参 Study" if zh else "# Model Tuning Study", ""]
    lines.extend(
        [
            "## 1. 调参结论" if zh else "## 1. Tuning Conclusion",
            "",
            f"- 获胜 trial：`{summary.get('winner_trial_id')}`" if zh else f"- Winner trial: `{summary.get('winner_trial_id')}`",
            f"- 相对 baseline 改善：{'是' if summary.get('winner_reason') == 'improved' else '否'}（delta={_format_delta(delta)}, relative={_format_percent(relative)}）" if zh else f"- Improved over baseline: `{summary.get('winner_reason') == 'improved'}`",
            f"- Selection split：`{plan.get('selection', {}).get('dataset_role')}`",
            f"- Learner：`{(plan.get('model_spec') or {}).get('model_type')}`",
            f"- Parameter strategy：`{plan.get('parameter_strategy') or 'shared'}`",
            "- Primary metric：`auuc_raw`",
            f"- Stop reason：`{summary.get('stop_reason')}`",
            f"- Trial 完成情况：attempted=`{summary.get('attempted_random_trials')}`，completed=`{summary.get('completed_random_trials')}`，failed=`{summary.get('failed_random_trials')}`" if zh else f"- Trial completion: attempted=`{summary.get('attempted_random_trials')}`, completed=`{summary.get('completed_random_trials')}`, failed=`{summary.get('failed_random_trials')}`",
            f"- 报告语言：`{language}`" if zh else f"- Report language: `{language}`",
            "",
            "## 2. Winner vs Baseline",
            "",
            "| Trial | Role | AUUC Raw | AUUC Normalized | Delta | Relative Improvement |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for trial, role, row_delta, row_relative in (
        (baseline, "baseline", 0.0, 0.0),
        (winner_trial, "winner", delta, relative),
    ):
        if not trial:
            continue
        lines.append(
            f"| `{trial.get('trial_id')}` | {role} | {_format_metric(trial.get('auuc_raw'))} | {_format_metric(trial.get('auuc_normalized'))} | {_format_delta(row_delta)} | {_format_percent(row_relative)} |"
        )
    lines.extend(
        [
            "",
            "## 3. Trial Leaderboard（Top 5）" if zh else "## 3. Trial Leaderboard (Top 5)",
            "",
            "| Rank | Trial | Status | AUUC Raw | AUUC Normalized | Objective Eligible | Duration Seconds | Best Iteration |",
            "| ---: | --- | --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    ranked = sorted(
        [trial for trial in trial_metrics if _finite(trial.get("auuc_raw"))],
        key=lambda item: float(item.get("auuc_raw")),
        reverse=True,
    )[:5]
    if not ranked:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    for rank, trial in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | `{trial.get('trial_id')}` | `{trial.get('status')}` | {_format_metric(trial.get('auuc_raw'))} | {_format_metric(trial.get('auuc_normalized'))} | `{trial.get('objective_eligible')}` | {_format_seconds(trial.get('duration_seconds'))} | {_format_count(trial.get('best_iteration'))} |"
        )
    lines.extend(
        [
            "",
            f"- 完整 trial 明细见 `{winner['winner_model_result_path']}`；排序表见 `{run_dir / 'artifacts' / 'leaderboard.v1.csv'}`，不在报告内全部展开。" if zh else f"- Full trial details are available under `{run_dir / 'artifacts'}`.",
            "",
            "## 4. 调参空间、预算与 Winner 参数" if zh else "## 4. Search Space, Budget, and Winner Parameters",
            "",
            f"- Search strategy：`{plan.get('search', {}).get('strategy')}`",
            f"- Profile：`{plan.get('budget', {}).get('profile')}`",
            f"- Budget：max_trials=`{plan.get('budget', {}).get('max_trials')}`，min_completed_trials=`{plan.get('budget', {}).get('min_completed_trials')}`，max_duration_minutes=`{plan.get('budget', {}).get('max_duration_minutes')}`，per_trial_timeout_minutes=`{plan.get('budget', {}).get('per_trial_timeout_minutes')}`",
            f"- Seeds：sampler=`{plan.get('reproducibility', {}).get('sampler_seed')}`，model=`{plan.get('reproducibility', {}).get('model_seed')}`",
            "",
            "### 搜索参数" if zh else "### Searched Parameters",
            "",
            "| Parameter | Space |",
            "| --- | --- |",
        ]
    )
    searched = plan.get("search", {}).get("search_space") or {}
    for name in SEARCHED_PARAM_ORDER:
        if name in searched:
            lines.append(f"| `{name}` | `{_compact_value(searched[name])}` |")
    for name in sorted(set(searched) - set(SEARCHED_PARAM_ORDER)):
        lines.append(f"| `{name}` | `{_compact_value(searched[name])}` |")
    lines.extend(
        [
            "",
            "### 固定参数预览" if zh else "### Fixed Parameter Preview",
            "",
            _inline_params(plan.get("search", {}).get("fixed_parameters") or {}, FIXED_PARAM_PREVIEW_ORDER),
            "",
            "### Winner 关键生效参数" if zh else "### Winner Key Effective Parameters",
            "",
            _inline_params(winner["model_candidate"].get("primary_evaluation", {}) | {"trial_id": summary.get("winner_trial_id")}, ("trial_id", "metric_value")),
            "",
            "## 5. Split 复核" if zh else "## 5. Split Review",
            "",
            f"- 本次模型选择使用 `{plan.get('selection', {}).get('dataset_role')}`；train/test/OOT 仅作复核或后续 comparison，本报告不声明最终采用。" if zh else "- Split review is for downstream comparison and reporting.",
            "",
            "| Split | AUUC Raw | AUUC Normalized | Rows | Treatment | Control | Valid Bins |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, payload in sorted(winner["split_evaluations"].items(), key=lambda item: _split_sort_key(item[0])):
        support = payload.get("support") or {}
        lines.append(
            f"| `{split}` | {_format_metric(payload.get('metric_value'))} | {_format_metric(payload.get('auuc_normalized'))} | {_format_count(support.get('row_count'))} | {_format_count(support.get('treatment_count'))} | {_format_count(support.get('control_count'))} | {_format_count(support.get('valid_bin_count'))} |"
        )
    lines.extend(
        [
            "",
            "## 6. 风险、限制与下一步" if zh else "## 6. Risks, Limitations, and Next Steps",
            "",
            "- Random Search 不保证找到全局最优，只表示当前预算和搜索空间下的最优观察结果。" if zh else "- Random Search does not guarantee a global optimum.",
            "- `trial_000` 复用 baseline model 和 evidence，不是重新训练得到的 trial。" if zh else "- `trial_000` reuses the baseline model and evidence.",
            "- 本次 winner 是 study winner，不是 primary model、final model 或 deployment model。" if zh else "- The winner is not a deployment decision.",
        ]
    )
    warnings = plan.get("selection", {}).get("holdout_warnings") or []
    if warnings:
        lines.append(
            "- Holdout 风险确认：" + ", ".join(f"`{item.get('code')}` acknowledged=`{item.get('acknowledged')}`" for item in warnings)
        )
    else:
        lines.append("- 本次未记录 test/OOT selection holdout 风险确认项。" if zh else "- No holdout acknowledgement was recorded.")
    if issues:
        lines.append("- Issues：" + ", ".join(f"`{item.get('code')}`" for item in issues))
    lines.extend(
        [
            "",
            "下一步：" if zh else "Next steps:",
            "- 与 baseline 或其他候选一起进入 `uplift-model-result-comparison`。",
            "- 如需要最终业务报告，使用 `uplift-model-reporting` 从结构化上游事实生成。",
            "",
            "Artifact paths：" if zh else "Artifact paths:",
            f"- winner_metrics: `{winner['winner_metrics_path']}`",
            f"- trial_metrics: `{run_dir / 'artifacts' / 'trial_metrics.v1.json'}`",
            f"- leaderboard: `{run_dir / 'artifacts' / 'leaderboard.v1.csv'}`",
            f"- model_candidate: `{winner['model_candidate_path']}`",
            "",
        ]
    )
    write_text(path, "\n".join(lines))
    return path


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
        run_dir, scope = _scope_dir_for_action(output_dir, action, body)
        request_path, result_path = next_action_paths(run_dir, action or "unknown")
    except ProjectLayoutError as exc:
        print(json.dumps(_layout_error_stdout(exc), sort_keys=True))
        return 0
    write_json(request_path, request)
    if request_error:
        result = _needs_input_result(run_dir, action or "unknown", request_error, ["payload"])
    elif action not in {"draft_plan", "confirm_plan", "execute"}:
        result = _unsupported_result(run_dir, action or "unknown", "action must be draft_plan, confirm_plan, or execute.")
    else:
        legacy_fields = _legacy_ref_rejections(body)
        if legacy_fields:
            result = _legacy_ref_result(run_dir, action, legacy_fields)
        else:
            try:
                if action == "draft_plan":
                    result = run_draft_plan(run_dir, output_dir, body)
                elif action == "confirm_plan":
                    result = run_confirm_plan(run_dir, output_dir, body)
                else:
                    result = run_execute(run_dir, output_dir, body)
            except (DataLoaderImportError, PackageImportError) as exc:
                result = _dependency_failure_result(run_dir, action, exc)
            except NeedsConfirmationError as exc:
                result = _needs_confirmation_result(run_dir, action, str(exc))
            except (MissingInputError, DataSourceError, AdapterInputError) as exc:
                result = _needs_input_result(
                    run_dir,
                    action,
                    str(exc),
                    getattr(exc, "missing_fields", []),
                    code=getattr(exc, "code", "MISSING_OR_INVALID_INPUT"),
                )
            except Exception as exc:  # noqa: BLE001
                result = _unexpected_error_result(run_dir, action, exc)
    _attach_transport_paths(result, output_dir, run_dir, result_path, scope)
    result = relativize_paths(result, output_dir)
    write_json(result_path, result)
    if scope == "experiment":
        experiment_id = str(result.get("outputs", {}).get("experiment_id") or run_dir.name)
        write_experiment_action_records(
            run_dir=output_dir,
            experiment_dir=run_dir,
            experiment_id=experiment_id,
            skill_name=SKILL_NAME,
            action=action or "unknown",
            request_path=request_path,
            result_path=result_path,
            result=result,
            extra_manifest=_tuning_experiment_manifest_fields(result),
        )
    else:
        write_flow_action_records(
            run_dir=output_dir,
            flow_dir=run_dir,
            skill_name=SKILL_NAME,
            action=action or "unknown",
            request_path=request_path,
            result_path=result_path,
            result=result,
        )
    print(json.dumps(_stdout_payload(result), sort_keys=True))
    return 0


def _scope_dir_for_action(output_dir: Path, action: str, body: dict[str, Any]) -> tuple[Path, str]:
    if action == "execute":
        plan_path = _required_layout_input_path(output_dir, body.get("tuning_plan_path"), "tuning_plan_path")
        plan = read_json(plan_path)
        experiment_id = validate_experiment_id(str(plan.get("experiment_id") or ""))
        return ensure_experiment_dir(output_dir, experiment_id), "experiment"
    if body.get("flow_dir"):
        return ensure_existing_skill_call_dir(output_dir, str(body["flow_dir"])), "flow"
    if action and action != "draft_plan":
        raise ProjectLayoutError("FLOW_DIR_REQUIRED", "flow_dir is required for tuning plan confirmation.")
    return ensure_skill_call_dir(output_dir, SKILL_NAME, "model_tuning"), "flow"


def _required_layout_input_path(output_dir: Path, value: Any, field_name: str) -> Path:
    if not value:
        raise ProjectLayoutError("MISSING_INPUT", f"{field_name} is required.")
    path = _resolve_input_path(output_dir, str(value))
    if not path.exists():
        raise ProjectLayoutError("INPUT_PATH_NOT_FOUND", f"{field_name} does not exist: {path}")
    return path


def _read_artifact(path: Path, *, artifact_kind: str, require_confirmed: bool = False) -> dict[str, Any]:
    artifact = read_json(path)
    if artifact.get("artifact_kind") != artifact_kind:
        raise MissingInputError(f"{path} is not a {artifact_kind} artifact.")
    if require_confirmed and artifact.get("artifact_status") != "confirmed":
        raise MissingInputError(f"{artifact_kind} must be confirmed.")
    return artifact


def _required_input_path(output_dir: Path, value: Any, field_name: str) -> Path:
    if not value:
        raise MissingInputError(f"{field_name} is required.", missing_fields=[field_name])
    path = _resolve_input_path(output_dir, str(value))
    if not path.exists():
        raise MissingInputError(f"{field_name} does not exist: {path}", missing_fields=[field_name])
    return path


def _resolve_input_path(output_dir: Path, value: str) -> Path:
    return resolve_run_path(output_dir, value)


def _flow_dir_from_plan_path(output_dir: Path, plan_path: Path) -> str | None:
    parent = plan_path.parent.parent if plan_path.parent.name == "artifacts" else plan_path.parent
    return to_run_relative_path(output_dir, parent)


def _same_path(left: Any, right: Any, run_dir: Path | None = None) -> bool:
    if not left or not right:
        return False
    try:
        left_path = resolve_run_path(run_dir, left) if run_dir else Path(str(left)).expanduser().resolve()
        right_path = resolve_run_path(run_dir, right) if run_dir else Path(str(right)).expanduser().resolve()
        return left_path == right_path
    except OSError:
        return str(left) == str(right)


def _validate_parameter_strategy_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        return
    preferences = payload.get("preferences") if isinstance(payload.get("preferences"), dict) else {}
    for container in (payload, preferences, payload.get("search") if isinstance(payload.get("search"), dict) else {}):
        strategy = str(container.get("parameter_strategy") or "")
        if strategy and strategy != "shared":
            raise MissingInputError(
                "v1 tuning only supports parameter_strategy = shared.",
                code="UNSUPPORTED_PARAMETER_STRATEGY",
            )
        for field in ("per_arm_parameters", "component_parameters", "treatment_parameters", "control_parameters"):
            if field in container:
                raise MissingInputError(
                    f"v1 tuning does not support {field}.",
                    code="UNSUPPORTED_PARAMETER_STRATEGY",
                )
    if "per_arm_parameters" in payload or "component_parameters" in payload:
        raise MissingInputError(
            "v1 tuning only supports shared parameters.",
            code="UNSUPPORTED_PARAMETER_STRATEGY",
        )
    for key, item in _recursive_strategy_items(payload):
        if key == "parameter_strategy" and str(item or "") not in {"", "shared"}:
            raise MissingInputError(
                "v1 tuning only supports parameter_strategy = shared.",
                code="UNSUPPORTED_PARAMETER_STRATEGY",
            )
        if key in {"per_arm_parameters", "component_parameters", "treatment_parameters", "control_parameters"}:
            raise MissingInputError(
                f"v1 tuning does not support {key}.",
                code="UNSUPPORTED_PARAMETER_STRATEGY",
            )


def _recursive_strategy_items(value: Any) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            items.append((str(key), item))
            items.extend(_recursive_strategy_items(item))
    elif isinstance(value, list):
        for item in value:
            items.extend(_recursive_strategy_items(item))
    return items


def _merge_search_space(overrides: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(overrides, dict):
        raise MissingInputError("search_space_overrides must be an object.")
    unknown = sorted(set(overrides) - set(DEFAULT_SEARCH_SPACE))
    if unknown:
        raise MissingInputError(f"Unsupported search-space overrides: {unknown}")
    space = json.loads(json.dumps(DEFAULT_SEARCH_SPACE))
    for name, override in overrides.items():
        if not isinstance(override, dict):
            raise MissingInputError(f"search_space_overrides.{name} must be an object.")
        space[name].update(override)
    return space


def _validate_budget(budget: dict[str, Any]) -> None:
    for name in ("max_trials", "min_completed_trials"):
        if not isinstance(budget.get(name), int) or isinstance(budget.get(name), bool) or budget[name] <= 0:
            raise MissingInputError(f"budget.{name} must be a positive integer.")
    for name in ("max_duration_minutes", "per_trial_timeout_minutes"):
        if not isinstance(budget.get(name), (int, float)) or isinstance(budget.get(name), bool) or budget[name] <= 0:
            raise MissingInputError(f"budget.{name} must be positive.")
    if budget["min_completed_trials"] > budget["max_trials"]:
        raise MissingInputError("budget.min_completed_trials cannot exceed budget.max_trials.")


def _validate_candidate_override(values: dict[str, Any]) -> None:
    unknown = sorted(set(values) - ALLOWED_PARAMETERS)
    if unknown:
        raise MissingInputError(f"Unsupported candidate parameters: {unknown}")


def _validate_parameters(parameters: dict[str, Any]) -> None:
    unknown = sorted(set(parameters) - ALLOWED_PARAMETERS)
    if unknown:
        raise MissingInputError(f"Unsupported parameters: {unknown}")
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


def _suggest_value(spec: dict[str, Any], rng: random.Random) -> Any:
    kind = str(spec.get("type") or "")
    if kind == "categorical":
        choices = spec.get("choices") or []
        if not choices:
            raise MissingInputError("categorical search space must provide choices.")
        return rng.choice(choices)
    if kind == "int":
        return rng.randint(int(spec["low"]), int(spec["high"]))
    if kind == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        if str(spec.get("scale")) == "log":
            return math.exp(rng.uniform(math.log(low), math.log(high)))
        return rng.uniform(low, high)
    raise MissingInputError(f"Unsupported search space type: {kind}")


def _load_model_object(packages: dict[str, Any], path: Path | str) -> dict[str, Any]:
    model = packages["joblib"].load(path)
    if not isinstance(model, dict) or "estimator" not in model or "encoder" not in model:
        raise MissingInputError("model artifact is not a supported S-Learner model object.")
    return model


def _trial_metric_fields(metric: dict[str, Any]) -> dict[str, Any]:
    support = metric.get("support") or {}
    return {
        "auuc_raw": metric.get("metric_value"),
        "auuc_normalized": metric.get("auuc_normalized"),
        "row_count": support.get("row_count"),
        "treatment_count": support.get("treatment_count"),
        "control_count": support.get("control_count"),
        "valid_bin_count": support.get("valid_bin_count"),
        "objective_eligible": _finite(metric.get("metric_value"))
        and bool(support.get("treatment_count"))
        and bool(support.get("control_count")),
        "eligibility_reason": None,
    }


def _write_leaderboard(path: Path, trials: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "trial_id",
                "status",
                "auuc_raw",
                "auuc_normalized",
                "objective_eligible",
                "duration_seconds",
                "best_iteration",
            ],
        )
        writer.writeheader()
        writer.writerows([{key: trial.get(key) for key in writer.fieldnames} for trial in trials])


def _overall_absolute_uplift(score_frame: Any) -> float | None:
    treatment = score_frame[score_frame["actual_treatment"] == 1]
    control = score_frame[score_frame["actual_treatment"] == 0]
    treat_mean = _mean(treatment["actual_outcome"])
    control_mean = _mean(control["actual_outcome"])
    return None if treat_mean is None or control_mean is None else treat_mean - control_mean


def _mean(series: Any) -> float | None:
    values = series.dropna()
    return None if values.empty else float(values.mean())


def _evaluation_quality(support: dict[str, Any]) -> str:
    lacks_groups = not support.get("treatment_count") or not support.get("control_count")
    few_bins = int(support.get("valid_bin_count") or 0) < 8
    return "insufficient" if lacks_groups or few_bins else "sufficient"


def _language_from_task_config(task: dict[str, Any]) -> str:
    return str(task.get("payload", {}).get("report_preferences", {}).get("language") or "zh-CN")


def _split_key(role: str) -> str:
    return "valid" if role == "validation" else role


def _split_sort_key(value: str) -> tuple[int, str]:
    order = {"train": 0, "valid": 1, "validation": 1, "test": 2, "oot": 3}
    return (order.get(value, 99), value)


def _file_fingerprint(path: Path) -> str:
    digest = _sha256(path)
    return f"sha256:{digest}"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("lightgbm", "pandas", "numpy", "scikit-learn", "joblib", "scikit-uplift"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


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
    interaction_type: str | None = None,
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
        "type": interaction_type or ("completion" if status in {"success", "partial_success"} else "recovery"),
        "subject": "model_tuning",
        "facts": {"issue_count": len(result["issues"])},
    }
    return result


def _needs_input_result(run_dir: Path, phase: str, message: str, missing_fields: list[str] | None = None, *, code: str = "MISSING_OR_INVALID_INPUT") -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="needs_input",
        summary=message,
        input_paths={},
        outputs={"flow_dir": str(run_dir.resolve()), "missing_fields": missing_fields or [], "report_path": None},
        issues=[{"code": code, "level": "critical", "blocking": True, "message": message}],
        progress=[{"step": phase, "status": "needs_input", "message": message}],
    )


def _needs_confirmation_result(run_dir: Path, phase: str, message: str) -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="needs_input",
        summary=message,
        input_paths={},
        outputs={"flow_dir": str(run_dir.resolve()), "confirmation_required_reason": message, "report_path": None},
        issues=[{"code": "USER_CONFIRMATION_REQUIRED", "level": "warning", "blocking": True, "message": message}],
        progress=[{"step": phase, "status": "needs_input", "message": message}],
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


def _legacy_ref_result(run_dir: Path, phase: str, rejected_fields: list[dict[str, str]]) -> dict[str, Any]:
    field_names = ", ".join(item["field"] for item in rejected_fields)
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
            "report_path": None,
        },
        issues=[
            {
                "code": "LEGACY_REF_FIELD_NOT_ACCEPTED",
                "level": "critical",
                "blocking": True,
                "field": item["field"],
                "suggested_field": item["suggested_field"],
                "message": f"{item['field']} is not accepted by self-contained runners; use {item['suggested_field']}.",
            }
            for item in rejected_fields
        ],
        progress=[{"step": "reject_legacy_ref_fields", "status": "needs_input"}],
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
        phase=phase or "unknown",
        status="failed",
        summary=f"{phase or 'action'} failed unexpectedly.",
        input_paths={},
        outputs={"flow_dir": str(run_dir.resolve()), "report_path": None},
        error={"code": "MODEL_TUNING_FAILED", "message": f"{phase or 'action'} failed unexpectedly.", "recoverable": False, "retryable": False, "raw_error": str(exc)},
        progress=[{"step": phase or "unknown", "status": "failed", "message": str(exc)}],
    )


def _attach_transport_paths(
    result: dict[str, Any],
    output_dir: Path,
    run_dir: Path,
    result_path: Path,
    scope: str,
) -> None:
    outputs = result.setdefault("outputs", {})
    result_path_value = to_run_relative_path(output_dir, result_path)
    if scope == "experiment":
        outputs["experiment_id"] = outputs.get("experiment_id") or run_dir.name
        outputs["experiment_dir"] = to_run_relative_path(output_dir, run_dir)
        outputs["flow_dir"] = outputs.get("tuning_flow_dir")
        if outputs.get("tuning_result_path") is None:
            outputs["tuning_result_path"] = result_path_value
    else:
        outputs["flow_dir"] = to_run_relative_path(output_dir, run_dir)
    outputs["result_path"] = result_path_value
    for step in result.get("next_steps") or []:
        inputs = step.get("inputs")
        if isinstance(inputs, dict) and inputs.get("candidate_result_paths"):
            inputs.pop("candidate_result_paths", None)
            inputs["experiment_ids"] = [
                item for item in [outputs.get("source_experiment_id"), outputs.get("experiment_id")] if item
            ]


def _tuning_experiment_manifest_fields(result: dict[str, Any]) -> dict[str, Any]:
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    keys = [
        "source_experiment_id",
        "source_modeling_result_path",
        "tuning_flow_dir",
        "tuning_plan_path",
        "tuning_result_path",
        "winner_result_path",
        "model_candidate_path",
        "evaluation_metrics_path",
        "report_path",
        "tuning_report_path",
    ]
    return {
        "experiment_type": "model_candidate",
        "learner": outputs.get("learner"),
        "tuning": {key: outputs.get(key) for key in keys if outputs.get(key) is not None},
    }


def _layout_error_stdout(exc: ProjectLayoutError) -> dict[str, Any]:
    field = "output_dir"
    if exc.issue_code == "INVALID_EXPERIMENT_ID":
        field = "experiment_id"
    elif exc.issue_code.startswith("FLOW_DIR"):
        field = "flow_dir"
    elif exc.issue_code in {"MISSING_INPUT", "INPUT_PATH_NOT_FOUND"}:
        field = "tuning_plan_path"
    return {
        "status": "needs_input",
        "summary": str(exc),
        "outputs": {"missing_fields": [field], "report_path": None},
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
    return {
        "status": result["status"],
        "summary": result["summary"],
        "outputs": result.get("outputs") or {},
        "issues": result.get("issues") or [],
        "next_steps": result.get("next_steps") or [],
    }


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _number(value: Any) -> float:
    if value in (None, ""):
        return math.nan
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _finite(value: Any) -> bool:
    return math.isfinite(_number(value))


def _format_metric(value: Any) -> str:
    parsed = _number(value)
    return "N/A" if not math.isfinite(parsed) else f"{parsed:.6g}"


def _format_delta(value: Any) -> str:
    parsed = _number(value)
    return "N/A" if not math.isfinite(parsed) else f"{parsed:+.6g}"


def _format_percent(value: Any) -> str:
    parsed = _number(value)
    return "N/A" if not math.isfinite(parsed) else f"{parsed:.2%}"


def _format_seconds(value: Any) -> str:
    parsed = _number(value)
    return "N/A" if not math.isfinite(parsed) else f"{parsed:.2f}"


def _format_count(value: Any) -> str:
    parsed = _number(value)
    return "N/A" if not math.isfinite(parsed) else f"{int(round(parsed)):,}"


def _compact_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _inline_params(params: dict[str, Any], ordered_names: tuple[str, ...]) -> str:
    parts = [f"`{name}`=`{_compact_value(params[name])}`" for name in ordered_names if name in params]
    return "- " + ("，".join(parts) if parts else "N/A")


if __name__ == "__main__":
    raise SystemExit(main())
