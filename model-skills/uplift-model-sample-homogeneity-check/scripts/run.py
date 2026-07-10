from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import csv
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _common.data_loader import DataLoaderImportError, DataSourceError, load_table
from _common.io import read_cli_json_input, read_json, write_json, write_text
from _common.project_layout import (
    ProjectLayoutError,
    assert_existing_run_dir,
    ensure_existing_skill_call_dir,
    ensure_skill_call_dir,
    next_action_paths,
    relativize_paths,
    resolve_run_path,
    to_run_relative_path,
    write_flow_action_records,
)
from _common.report_language import is_zh
from homogeneity_interaction import build_user_interaction, covariate_changes, covariate_summary

SKILL_NAME = "uplift-model-sample-homogeneity-check"
SKILL_CREATED_BY = "uplift-model-sample-homogeneity-check-skill"
SPLIT_NAMES = ("train", "test", "valid", "oot")
DISPLAY_SPLIT_ORDER = ("train", "valid", "test", "oot")
RISK_ORDER = {"pass": 0, "warning": 1, "severe": 2, "critical": 3}
ALLOWED_REF_FIELDS = {"data_ref"}
DEFAULT_DIAGNOSTIC_CONFIG = {
    "auc_warning_threshold": 0.60,
    "auc_severe_threshold": 0.70,
    "auc_critical_threshold": 0.80,
    "smd_warning_threshold": 0.10,
    "smd_severe_threshold": 0.20,
    "smd_key_critical_threshold": 0.50,
    "smd_severe_count_critical_threshold": 5,
    "max_categorical_levels": 10,
    "top_n_imbalanced": 20,
    "auc_holdout_ratio": 0.30,
    "random_seed": 42,
    "min_group_size_for_auc": 30,
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


class DiagnosticError(Exception):
    pass


class PackageImportError(Exception):
    def __init__(self, package: str, message: str) -> None:
        super().__init__(message)
        self.package = package


def plan_covariates(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    phase = "plan_covariates"
    try:
        spec_path = _required_input_path(
            output_dir, payload.get("modeling_sample_spec_path"), "modeling_sample_spec_path"
        )
        spec = _read_artifact(spec_path, artifact_kind="modeling_sample_spec", require_confirmed=True)
        frames = _load_enabled_split_frames(spec)
        if not frames:
            raise MissingInputError("modeling_sample_spec does not reference enabled split datasets.")
        plan_payload = _build_covariate_plan(
            spec,
            frames,
            user_covariates=payload.get("user_covariates"),
            max_categorical_levels=int(
                payload.get("max_categorical_levels", DEFAULT_DIAGNOSTIC_CONFIG["max_categorical_levels"])
            ),
        )
        included = [item for item in plan_payload["covariates"] if item["included"]]
        if not included:
            raise MissingInputError(
                "No supported homogeneity covariates are available; provide pre-treatment covariates."
            )
        artifact_path = _write_covariate_plan_draft(
            run_dir,
            input_paths={"modeling_sample_spec_path": str(spec_path)},
            payload=plan_payload,
            prior_plan_path=_optional_input_path(output_dir, payload.get("prior_plan_path")),
        )
        summary = _covariate_summary(plan_payload)
        return _result(
            run_dir=run_dir,
            phase=phase,
            status="needs_confirmation",
            summary="Homogeneity covariate plan is ready for user confirmation.",
            input_paths={"modeling_sample_spec_path": str(spec_path)},
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "homogeneity_covariate_plan_path": str(artifact_path.resolve()),
                "confirmation_summary": summary,
                "report_path": None,
            },
            artifacts=[{"kind": "homogeneity_covariate_plan", "path": str(artifact_path.resolve())}],
            progress=[
                {
                    "step": "read_modeling_sample_spec",
                    "status": "success",
                    "message": "Modeling sample spec loaded.",
                },
                {
                    "step": "plan_covariates",
                    "status": "needs_confirmation",
                    "message": "Draft covariate plan written.",
                },
            ],
            next_steps=[
                {
                    "skill": SKILL_NAME,
                    "action": "confirm_artifact",
                    "reason": "homogeneity_covariate_plan must be confirmed before diagnostics can run.",
                    "inputs": {
                        "flow_dir": str(run_dir.resolve()),
                        "source_draft_path": str(artifact_path.resolve()),
                        "artifact_kind": "homogeneity_covariate_plan",
                    },
                    "requires_user_confirmation": True,
                }
            ],
        )
    except (DataLoaderImportError, PackageImportError) as exc:
        return _dependency_failure_result(run_dir, phase, exc)
    except (MissingInputError, DataSourceError) as exc:
        return _needs_input_result(run_dir, phase, str(exc), getattr(exc, "missing_fields", []))
    except Exception as exc:  # noqa: BLE001
        return _unexpected_error_result(run_dir, phase, exc)


def confirm_artifact(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    phase = "confirm_artifact"
    try:
        source_draft_path = _required_input_path(
            output_dir, payload.get("source_draft_path"), "source_draft_path"
        )
        artifact_kind = str(payload.get("artifact_kind") or "homogeneity_covariate_plan")
        confirmed_by = str(payload.get("confirmed_by") or "user")
        if artifact_kind != "homogeneity_covariate_plan":
            raise MissingInputError(
                "uplift-model-sample-homogeneity-check can only confirm homogeneity_covariate_plan."
            )
        if payload.get("confirmed_payload") is not None:
            raise MissingInputError(
                "confirm_artifact does not accept payload changes; rerun plan_covariates."
            )
        draft = _read_artifact(
            source_draft_path, artifact_kind="homogeneity_covariate_plan", require_draft=True
        )
        spec_path = resolve_run_path(output_dir, str(draft.get("input_paths", {}).get("modeling_sample_spec_path") or ""))
        spec = _read_artifact(spec_path, artifact_kind="modeling_sample_spec", require_confirmed=True)
        frames = _load_enabled_split_frames(spec)
        _validate_plan_payload(draft.get("payload") or {}, frames)
        version = int(draft.get("artifact_version") or 1)
        artifact_path = run_dir / "artifacts" / f"homogeneity_covariate_plan.confirmed.v{version}.json"
        confirmed = dict(draft)
        confirmed["artifact_status"] = "confirmed"
        confirmed["confirmed_by"] = confirmed_by
        confirmed["confirmed_at"] = _now()
        confirmed["source_draft_path"] = str(source_draft_path)
        write_json(artifact_path, confirmed)
        return _result(
            run_dir=run_dir,
            phase=phase,
            status="success",
            summary="homogeneity_covariate_plan confirmed.",
            input_paths={
                "source_draft_path": str(source_draft_path),
                "modeling_sample_spec_path": str(spec_path),
            },
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "homogeneity_covariate_plan_path": str(artifact_path.resolve()),
                "homogeneity_covariate_plan_status": "confirmed",
                "report_path": None,
            },
            artifacts=[{"kind": artifact_kind, "path": str(artifact_path.resolve())}],
            progress=[
                {
                    "step": "validate_draft",
                    "status": "success",
                    "message": "Draft covariate plan checked.",
                },
                {
                    "step": "confirm_artifact",
                    "status": "success",
                    "message": "Confirmed artifact written.",
                },
            ],
            next_steps=[
                {
                    "skill": SKILL_NAME,
                    "action": "run_diagnostics",
                    "reason": "The covariate plan is confirmed; balance diagnostics can run.",
                    "inputs": {
                        "flow_dir": str(run_dir.resolve()),
                        "modeling_sample_spec_path": str(spec_path),
                        "homogeneity_covariate_plan_path": str(artifact_path.resolve()),
                    },
                    "requires_user_confirmation": False,
                }
            ],
            user_interaction_context={
                "active_result": covariate_summary(confirmed.get("payload") or {}),
                "changes": covariate_changes(None, confirmed.get("payload") or {}),
            },
        )
    except (DataLoaderImportError, PackageImportError) as exc:
        return _dependency_failure_result(run_dir, phase, exc)
    except (MissingInputError, DataSourceError) as exc:
        return _needs_input_result(run_dir, phase, str(exc), getattr(exc, "missing_fields", []))
    except Exception as exc:  # noqa: BLE001
        return _unexpected_error_result(run_dir, phase, exc)


def run_diagnostics(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    phase = "run_diagnostics"
    try:
        spec_path = _required_input_path(
            output_dir, payload.get("modeling_sample_spec_path"), "modeling_sample_spec_path"
        )
        plan_path = _required_input_path(
            output_dir,
            payload.get("homogeneity_covariate_plan_path"),
            "homogeneity_covariate_plan_path",
        )
        spec = _read_artifact(spec_path, artifact_kind="modeling_sample_spec", require_confirmed=True)
        plan = _read_artifact(
            plan_path, artifact_kind="homogeneity_covariate_plan", require_confirmed=True
        )
        if not _same_path(plan.get("input_paths", {}).get("modeling_sample_spec_path"), spec_path, output_dir):
            raise NeedsConfirmationError(
                "homogeneity_covariate_plan lineage does not match modeling_sample_spec_path."
            )
        config_snapshot = _effective_diagnostic_config(payload.get("diagnostic_config_overrides") or {})
        config_path = _write_diagnostic_config_snapshot(run_dir, config_snapshot)
        frames = _load_enabled_split_frames(spec)
        covariates = [item for item in plan["payload"]["covariates"] if item.get("included")]
        if not covariates:
            raise MissingInputError("Confirmed homogeneity_covariate_plan has no included covariates.")
        diagnostics = _run_split_diagnostics(
            frames, spec, covariates, config_snapshot["effective_values"]
        )
        if not diagnostics["any_smd"]:
            raise DiagnosticError("No usable split or covariate could be diagnosed.")
        covariate_path = run_dir / "artifacts" / "covariates.v1.csv"
        smd_path = run_dir / "artifacts" / "smd_detail.v1.csv"
        importance_path = run_dir / "artifacts" / "auc_feature_importance.v1.csv"
        _write_csv(covariate_path, diagnostics["covariate_rows"])
        _write_csv(smd_path, diagnostics["smd_rows"])
        _write_csv(importance_path, diagnostics["importance_rows"])
        task_config_path = spec.get("input_paths", {}).get("task_config_path")
        task_config = read_json(task_config_path) if task_config_path else None
        report_data = {
            "version": 1,
            "diagnostic_completeness": "Complete" if diagnostics["complete"] else "Partial",
            "overall_status": diagnostics["overall_status"],
            "report_path": str((run_dir / "report.md").resolve()),
            "covariates_path": str(covariate_path.resolve()),
            "smd_detail_path": str(smd_path.resolve()),
            "auc_feature_importance_path": str(importance_path.resolve()),
            "diagnostic_config_path": str(config_path.resolve()),
            "config": config_snapshot,
            "language": _language_from_task_config(task_config),
            "modeling_sample_spec_path": str(spec_path),
            "homogeneity_covariate_plan_path": str(plan_path),
            "task_config_path": task_config_path,
            **diagnostics,
        }
        report_path = _write_report(run_dir, report_data)
        status = "success" if diagnostics["complete"] else "partial_success"
        next_step_requires_confirmation = diagnostics["overall_status"] in {"severe", "critical"}
        return _result(
            run_dir=run_dir,
            phase=phase,
            status=status,
            summary="Sample homogeneity diagnostics completed.",
            input_paths={
                "modeling_sample_spec_path": str(spec_path),
                "homogeneity_covariate_plan_path": str(plan_path),
            },
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "diagnostic_completeness": report_data["diagnostic_completeness"],
                "overall_status": diagnostics["overall_status"],
                "report_path": str(report_path.resolve()),
                "sample_homogeneity_result_path": None,
                "covariates_path": str(covariate_path.resolve()),
                "smd_detail_path": str(smd_path.resolve()),
                "auc_feature_importance_path": str(importance_path.resolve()),
                "diagnostic_config_path": str(config_path.resolve()),
                "auc_by_split": diagnostics["auc_by_split"],
                "smd_summary_by_split": diagnostics["smd_summary_by_split"],
                "balance_summary": diagnostics["balance_summary"],
                "top_imbalanced_variables": diagnostics["top_imbalanced_variables"],
                "skipped_variables": diagnostics["skipped_variables"],
                "recommended_action": _recommended_action(diagnostics["overall_status"]),
            },
            issues=diagnostics["issues"],
            artifacts=[
                {"kind": "report", "path": str(report_path.resolve())},
                {"kind": "diagnostic_table", "path": str(covariate_path.resolve())},
                {"kind": "diagnostic_table", "path": str(smd_path.resolve())},
                {"kind": "diagnostic_table", "path": str(importance_path.resolve())},
                {"kind": "diagnostic_config", "path": str(config_path.resolve())},
            ],
            progress=[
                {
                    "step": "load_inputs",
                    "status": "success",
                    "message": "Confirmed sample and covariate plan loaded.",
                },
                {
                    "step": "run_diagnostics",
                    "status": status,
                    "message": "AUC and SMD diagnostics completed.",
                },
                {
                    "step": "generate_report",
                    "status": "success",
                    "message": "Markdown report and CSV appendices written.",
                },
            ],
            next_steps=[
                {
                    "skill": "uplift-model-feature-quality-analysis",
                    "action": "analyze_feature_quality",
                    "reason": "Use homogeneity diagnostics as sample risk context before feature analysis.",
                    "inputs": {
                        "modeling_sample_spec_path": str(spec_path),
                        "sample_homogeneity_result_path": None,
                    },
                    "requires_user_confirmation": next_step_requires_confirmation,
                }
            ],
        )
    except NeedsConfirmationError as exc:
        return _result(
            run_dir=run_dir,
            phase=phase,
            status="needs_confirmation",
            summary=str(exc),
            input_paths={},
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "confirmation_required_reason": str(exc),
                "report_path": None,
            },
            issues=[
                _issue(
                    code="STALE_HOMOGENEITY_COVARIATE_PLAN",
                    level="warning",
                    blocking=False,
                    message=str(exc),
                    suggested_fix="Confirm a plan for this modeling sample or rerun plan_covariates.",
                )
            ],
            progress=[{"step": "lineage_check", "status": "needs_confirmation", "message": str(exc)}],
        )
    except (DataLoaderImportError, PackageImportError) as exc:
        return _dependency_failure_result(run_dir, phase, exc)
    except (MissingInputError, DataSourceError) as exc:
        return _needs_input_result(run_dir, phase, str(exc), getattr(exc, "missing_fields", []))
    except DiagnosticError as exc:
        return _diagnostic_failure_result(run_dir, phase, str(exc))
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
        run_dir = _flow_folder_for_action(output_dir, action, body)
        request_path, result_path = next_action_paths(run_dir, action or "unknown")
    except ProjectLayoutError as exc:
        print(json.dumps(_layout_error_stdout(exc), sort_keys=True))
        return 0
    write_json(request_path, request)
    if request_error:
        result = _needs_input_result(run_dir, action or "unknown", request_error, ["payload"])
    elif action not in {"plan_covariates", "confirm_artifact", "run_diagnostics"}:
        result = _unsupported_result(run_dir, action or "unknown", f"Unsupported action: {action}")
    else:
        legacy_fields = _legacy_ref_rejections(body)
        if legacy_fields:
            result = _legacy_ref_result(run_dir, action, legacy_fields)
        elif action == "plan_covariates":
            result = plan_covariates(run_dir, output_dir, body)
        elif action == "confirm_artifact":
            result = confirm_artifact(run_dir, output_dir, body)
        else:
            result = run_diagnostics(run_dir, output_dir, body)
    _attach_transport_paths(result, output_dir, run_dir, result_path)
    result = relativize_paths(result, output_dir)
    write_json(result_path, result)
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


def _flow_folder_for_action(output_dir: Path, action: str, body: dict[str, Any]) -> Path:
    if body.get("flow_dir"):
        return ensure_existing_skill_call_dir(output_dir, str(body["flow_dir"]))
    if action and action != "plan_covariates":
        raise ProjectLayoutError("FLOW_DIR_REQUIRED", "flow_dir is required after plan_covariates.")
    return ensure_skill_call_dir(output_dir, SKILL_NAME, "homogeneity_check")


def _read_artifact(
    path: Path,
    *,
    artifact_kind: str,
    require_confirmed: bool = False,
    require_draft: bool = False,
) -> dict[str, Any]:
    if not path.exists():
        raise MissingInputError(f"Artifact file does not exist: {path}")
    artifact = read_json(path)
    if artifact.get("artifact_kind") != artifact_kind:
        raise MissingInputError(f"{path} is not a {artifact_kind} artifact.")
    status = artifact.get("artifact_status")
    if require_confirmed and status != "confirmed":
        raise MissingInputError(f"{path} is not confirmed.")
    if require_draft and status != "draft":
        raise MissingInputError(f"{path} is not a draft artifact.")
    return artifact


def _required_input_path(output_dir: Path, value: Any, field: str) -> Path:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise MissingInputError(f"{field} is required.", missing_fields=[field])
    return _resolve_input_path(output_dir, str(value))


def _optional_input_path(output_dir: Path, value: Any) -> Path | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _resolve_input_path(output_dir, str(value))


def _resolve_input_path(output_dir: Path, value: str) -> Path:
    return resolve_run_path(output_dir, value)


def _same_path(value: Any, path: Path, run_dir: Path | None = None) -> bool:
    if not value:
        return False
    try:
        left_path = resolve_run_path(run_dir, value) if run_dir else Path(str(value)).expanduser().resolve()
        right_path = resolve_run_path(run_dir, path) if run_dir else path.resolve()
        return left_path == right_path
    except Exception:  # noqa: BLE001
        return False


def _artifact_envelope(
    *,
    artifact_status: str,
    artifact_version: int,
    input_paths: dict[str, Any],
    payload: dict[str, Any],
    validation: dict[str, Any],
    source_draft_path: str | None = None,
) -> dict[str, Any]:
    return {
        "artifact_kind": "homogeneity_covariate_plan",
        "artifact_status": artifact_status,
        "artifact_version": artifact_version,
        "source_type": "generated_by_skill",
        "input_paths": input_paths,
        "created_by": SKILL_CREATED_BY,
        "created_at": _now(),
        "confirmed_by": None,
        "confirmed_at": None,
        "source_draft_path": source_draft_path,
        "payload": payload,
        "validation": validation,
    }


def _build_covariate_plan(
    spec: dict[str, Any],
    frames: dict[str, Any],
    *,
    user_covariates: list[str] | None,
    max_categorical_levels: int,
) -> dict[str, Any]:
    train = frames["train"] if "train" in frames else next(iter(frames.values()))
    forced_exclude = set(spec["payload"].get("feature_columns", {}).get("forced_exclude", []))
    columns_payload = spec["payload"].get("columns", {})
    forced_exclude.update(
        value
        for value in [
            columns_payload.get("treatment"),
            columns_payload.get("outcome"),
            columns_payload.get("source_treatment"),
            columns_payload.get("source_outcome"),
            columns_payload.get("split"),
        ]
        if value
    )
    requested = list(user_covariates or [])
    candidates = requested if requested else [column for column in train.columns if column not in forced_exclude]
    covariates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for column in candidates:
        if column not in train.columns:
            excluded.append(
                {
                    "name": column,
                    "reason": "missing_column",
                    "suggested_fix": "Choose a column present in enabled split datasets.",
                }
            )
            continue
        if column in forced_exclude or _is_disallowed_name(column):
            excluded.append(
                {
                    "name": column,
                    "reason": "forced_exclude_or_leakage_risk",
                    "suggested_fix": "Use pre-treatment covariates only.",
                }
            )
            continue
        variable_type, level_count = _infer_variable_type(train[column])
        if variable_type == "unsupported":
            excluded.append(
                {
                    "name": column,
                    "reason": "unsupported_type",
                    "suggested_fix": "Convert the variable to numeric, binary, or categorical.",
                }
            )
            continue
        if variable_type == "categorical" and level_count > max_categorical_levels:
            covariates.append(
                {
                    "name": column,
                    "variable_type": variable_type,
                    "semantic_role": "user_specified" if user_covariates else "candidate_pre_treatment",
                    "is_key": bool(user_covariates),
                    "included": False,
                    "level_count": level_count,
                    "reason": "skipped_high_cardinality",
                }
            )
            excluded.append(
                {
                    "name": column,
                    "reason": "skipped_high_cardinality",
                    "suggested_fix": "Group levels, use topK+other, or select a lower-cardinality representation.",
                }
            )
            continue
        covariates.append(
            {
                "name": column,
                "variable_type": variable_type,
                "semantic_role": "user_specified" if user_covariates else "candidate_pre_treatment",
                "is_key": bool(user_covariates),
                "included": True,
                "level_count": level_count,
                "reason": "user_specified" if user_covariates else "recommended_candidate",
            }
        )
    return {
        "covariates": covariates,
        "excluded_variables": excluded,
        "max_categorical_levels": max_categorical_levels,
    }


def _classify_auc_status(auc: float | None, config: dict[str, Any]) -> str:
    if auc is None or math.isnan(float(auc)):
        return "warning"
    if auc >= config["auc_critical_threshold"]:
        return "critical"
    if auc >= config["auc_severe_threshold"]:
        return "severe"
    if auc >= config["auc_warning_threshold"]:
        return "warning"
    return "pass"


def _classify_smd_status(abs_smd: float | None, config: dict[str, Any]) -> str:
    if abs_smd is None or math.isnan(float(abs_smd)):
        return "warning"
    if abs_smd >= config["smd_severe_threshold"]:
        return "severe"
    if abs_smd >= config["smd_warning_threshold"]:
        return "warning"
    return "pass"


def _covariate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    included = [item for item in payload["covariates"] if item["included"]]
    return {
        "included_variables_count": len(included),
        "numeric_count": sum(1 for item in included if item["variable_type"] == "numeric"),
        "binary_count": sum(1 for item in included if item["variable_type"] == "binary"),
        "categorical_count": sum(1 for item in included if item["variable_type"] == "categorical"),
        "skipped_variables_count": len(payload["excluded_variables"]),
        "questions": [],
    }


def _effective_diagnostic_config(overrides: dict[str, Any]) -> dict[str, Any]:
    effective = dict(DEFAULT_DIAGNOSTIC_CONFIG)
    allowed = set(effective)
    for key, value in overrides.items():
        if key not in allowed:
            raise MissingInputError(f"Unsupported diagnostic_config_overrides key: {key}")
        effective[key] = value
    ratio = float(effective["auc_holdout_ratio"])
    if ratio <= 0 or ratio >= 1:
        raise MissingInputError("auc_holdout_ratio must be between 0 and 1.")
    if int(effective["max_categorical_levels"]) < 1:
        raise MissingInputError("max_categorical_levels must be positive.")
    source = "user_payload" if overrides else "default"
    return {"source": source, "overrides": overrides, "effective_values": effective}


def _infer_variable_type(series: Any) -> tuple[str, int]:
    pd = _pd()
    clean = series.dropna()
    level_count = int(clean.nunique(dropna=True))
    if level_count <= 2:
        return "binary", level_count
    if pd.api.types.is_numeric_dtype(series):
        return "numeric", level_count
    if (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
    ):
        return "categorical", level_count
    return "unsupported", level_count


def _is_disallowed_name(column: str) -> bool:
    lowered = column.lower()
    if lowered.startswith("__uplift_"):
        return True
    leakage_tokens = ("outcome", "label", "target", "visit", "conversion", "converted")
    return any(token in lowered for token in leakage_tokens) or lowered in {
        "id",
        "user_id",
        "row_id",
    }


def _is_number(value: Any) -> bool:
    if value in {None, ""}:
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _issue(
    *, code: str, level: str, blocking: bool, message: str, suggested_fix: str
) -> dict[str, Any]:
    return {
        "code": code,
        "level": level,
        "blocking": blocking,
        "message": message,
        "suggested_fix": suggested_fix,
    }


def _load_enabled_split_frames(spec: dict[str, Any]) -> dict[str, Any]:
    frames: dict[str, Any] = {}
    datasets = spec.get("payload", {}).get("datasets", {})
    for split in SPLIT_NAMES:
        dataset_path = datasets.get(split)
        if not dataset_path:
            continue
        path = Path(str(dataset_path)).expanduser().resolve()
        if not path.exists():
            raise MissingInputError(f"Dataset file does not exist: {path}")
        frame = load_table({"kind": "local_csv", "path": str(path)})
        if len(frame) > 0:
            frames[split] = frame
    return frames


def _max_risk(statuses: list[str]) -> str:
    if not statuses:
        return "warning"
    return max(statuses, key=lambda item: RISK_ORDER[item])


def _numeric_or_negative(value: Any) -> float:
    return -1.0 if not _is_number(value) else float(value)


def _prepare_auc_frame(frame: Any, covariates: list[dict[str, Any]]) -> Any:
    pd = _pd()
    np = _np()
    columns = [item["name"] for item in covariates]
    prepared = pd.get_dummies(frame[columns], dummy_na=True)
    return prepared.replace([np.inf, -np.inf], np.nan).fillna(-999999)


def _recommended_action(overall_status: str) -> str:
    if overall_status == "pass":
        return "continue_to_feature_quality_analysis"
    if overall_status == "warning":
        return "continue_to_feature_quality_analysis"
    if overall_status == "severe":
        return "continue_with_accepted_risk"
    return "review_sample_split"


def _risk_title(status: str) -> str:
    return {"pass": "Pass", "warning": "Warning", "severe": "Severe", "critical": "Critical"}[
        status
    ]


def _run_auc(
    split: str,
    frame: Any,
    treatment_column: str,
    covariates: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    y = frame[treatment_column].astype(int)
    treatment_rows = int((y == 1).sum())
    control_rows = int((y == 0).sum())
    if min(treatment_rows, control_rows) < int(config["min_group_size_for_auc"]):
        return (
            {
                "rows": int(len(frame)),
                "treatment_rows": treatment_rows,
                "control_rows": control_rows,
                "auc": None,
                "status": "warning",
                "top_predictive_variables": [],
                "reason": "insufficient_group_size",
            },
            [],
            [
                _issue(
                    code="AUC_INSUFFICIENT_GROUP_SIZE",
                    level="warning",
                    blocking=False,
                    message=f"{split} split has insufficient treatment/control rows for AUC.",
                    suggested_fix="Increase sample size or rely on SMD-only diagnostics for this split.",
                )
            ],
        )
    x = _prepare_auc_frame(frame, covariates)
    try:
        train_test_split = _train_test_split()
        roc_auc_score = _roc_auc_score()
        x_train, x_holdout, y_train, y_holdout = train_test_split(
            x,
            y,
            test_size=float(config["auc_holdout_ratio"]),
            random_state=int(config["random_seed"]),
            stratify=y,
        )
        model, model_issues = _treatment_predictability_model(config)
        model.fit(x_train, y_train)
        score = model.predict_proba(x_holdout)[:, 1]
        auc = float(roc_auc_score(y_holdout, score))
        pd = _pd()
        importances = (
            pd.Series(model.feature_importances_, index=x.columns)
            .groupby(lambda item: item.split("_", 1)[0])
            .sum()
        )
        importance_rows = [
            {"split": split, "variable": variable, "importance": float(value), "rank": rank}
            for rank, (variable, value) in enumerate(
                importances.sort_values(ascending=False).items(), start=1
            )
        ]
        top = [row["variable"] for row in importance_rows[:5]]
        return (
            {
                "rows": int(len(frame)),
                "treatment_rows": treatment_rows,
                "control_rows": control_rows,
                "auc": auc,
                "status": _classify_auc_status(auc, config),
                "top_predictive_variables": top,
            },
            importance_rows,
            model_issues,
        )
    except PackageImportError:
        raise
    except Exception as exc:  # noqa: BLE001
        return (
            {
                "rows": int(len(frame)),
                "treatment_rows": treatment_rows,
                "control_rows": control_rows,
                "auc": None,
                "status": "warning",
                "top_predictive_variables": [],
                "reason": "auc_training_failed",
            },
            [],
            [
                _issue(
                    code="AUC_TRAINING_FAILED",
                    level="warning",
                    blocking=False,
                    message=f"{split} split AUC diagnostic failed: {exc}",
                    suggested_fix="Review covariate encoding or rely on SMD diagnostics for this split.",
                )
            ],
        )


def _run_split_diagnostics(
    frames: dict[str, Any],
    spec: dict[str, Any],
    covariates: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    treatment_column = spec["payload"]["columns"]["treatment"]
    covariate_rows = [
        {
            "variable": item["name"],
            "variable_type": item["variable_type"],
            "included": item.get("included", True),
            "role_or_source": item.get("semantic_role", ""),
            "level_count": item.get("level_count", ""),
            "reason": item.get("reason", ""),
        }
        for item in covariates
    ]
    smd_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    auc_by_split: dict[str, Any] = {}
    smd_summary_by_split: dict[str, Any] = {}
    skipped_variables: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    split_statuses: list[str] = []
    complete = True
    any_smd = False
    for split, frame in frames.items():
        split_covariates: list[dict[str, Any]] = []
        for item in covariates:
            variable = item["name"]
            if variable not in frame.columns:
                skipped_variables.append(
                    {
                        "split": split,
                        "variable": variable,
                        "reason": "missing_column",
                        "suggested_fix": "Use columns present in every enabled split.",
                    }
                )
                complete = False
                continue
            variable_type, level_count = _infer_variable_type(frame[variable])
            if variable_type == "categorical" and level_count > int(config["max_categorical_levels"]):
                skipped_variables.append(
                    {
                        "split": split,
                        "variable": variable,
                        "reason": "skipped_high_cardinality",
                        "suggested_fix": "Group levels or use topK+other before rerunning diagnostics.",
                    }
                )
                complete = False
                continue
            split_covariates.append({**item, "variable_type": variable_type, "level_count": level_count})
        if split_covariates:
            auc, importances, auc_issues = _run_auc(
                split, frame, treatment_column, split_covariates, config
            )
            auc_by_split[split] = auc
            importance_rows.extend(importances)
            issues.extend(auc_issues)
        else:
            auc_by_split[split] = {
                "rows": int(len(frame)),
                "auc": None,
                "status": "warning",
                "top_predictive_variables": [],
                "reason": "no_covariates",
            }
            complete = False
        split_smd_rows = _smd_rows_for_split(split, frame, treatment_column, split_covariates, config)
        smd_rows.extend(split_smd_rows)
        any_smd = any_smd or bool(split_smd_rows)
        summary = _smd_summary(split_smd_rows, config)
        smd_summary_by_split[split] = summary
        split_statuses.extend([auc_by_split[split]["status"], summary["status"]])
    severe_smd_count = sum(1 for row in smd_rows if row["status"] == "severe")
    key_critical = any(
        row.get("is_key")
        and _is_number(row.get("abs_smd"))
        and float(row["abs_smd"]) >= float(config["smd_key_critical_threshold"])
        for row in smd_rows
    )
    overall_status = _max_risk(split_statuses + (["warning"] if skipped_variables else []))
    if key_critical or severe_smd_count >= int(config["smd_severe_count_critical_threshold"]):
        overall_status = "critical"
    top_rows = sorted(smd_rows, key=lambda row: _numeric_or_negative(row.get("abs_smd")), reverse=True)
    top_n = int(config["top_n_imbalanced"])
    top_imbalanced = top_rows[:top_n]
    for row in top_rows[top_n:]:
        if row["status"] == "severe":
            top_imbalanced.append(row)
    if overall_status in {"severe", "critical"}:
        issues.append(
            _issue(
                code="HOMOGENEITY_RISK_REQUIRES_CONFIRMATION",
                level=overall_status,
                blocking=False,
                message=(
                    f"Overall homogeneity risk is {overall_status}; downstream "
                    "continuation requires user confirmation."
                ),
                suggested_fix=(
                    "Review sample split, stratification, matching, weighting, "
                    "or explicitly accept the risk before continuing."
                ),
            )
        )
    return {
        "complete": complete and not skipped_variables and all(split in auc_by_split for split in frames),
        "any_smd": any_smd,
        "overall_status": overall_status,
        "covariate_rows": covariate_rows,
        "smd_rows": smd_rows,
        "importance_rows": importance_rows,
        "auc_by_split": auc_by_split,
        "smd_summary_by_split": smd_summary_by_split,
        "balance_summary": {
            "enabled_splits": sorted(frames, key=_split_display_key),
            "checked_covariates": len(covariates),
            "severe_smd_variable_count": severe_smd_count,
            "skipped_variables_count": len(skipped_variables),
        },
        "top_imbalanced_variables": top_imbalanced,
        "skipped_variables": skipped_variables,
        "issues": issues,
    }


def _split_display_key(split: str) -> tuple[int, str]:
    aliases = {"validation": "valid"}
    normalized = aliases.get(str(split), str(split))
    order = {name: index for index, name in enumerate(DISPLAY_SPLIT_ORDER)}
    return order.get(normalized, 50), str(split)


def _treatment_predictability_model(config: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    try:
        from lightgbm import LGBMClassifier  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise PackageImportError("lightgbm", "Missing package: lightgbm") from exc
    return (
        LGBMClassifier(
            n_estimators=80,
            learning_rate=0.05,
            num_leaves=15,
            random_state=int(config["random_seed"]),
            verbosity=-1,
        ),
        [],
    )


def _smd_binary(treatment: Any, control: Any) -> tuple[float | None, float, float]:
    pd = _pd()
    all_values = sorted(
        pd.concat([treatment, control]).dropna().unique(), key=lambda item: str(item)
    )
    positive = all_values[-1] if all_values else 1
    p_t = float((treatment == positive).mean()) if len(treatment) else 0.0
    p_c = float((control == positive).mean()) if len(control) else 0.0
    denom = math.sqrt(((p_t * (1 - p_t)) + (p_c * (1 - p_c))) / 2)
    return (None if denom == 0 else (p_t - p_c) / denom, p_t, p_c)


def _smd_numeric(treatment: Any, control: Any) -> tuple[float | None, float, float]:
    pd = _pd()
    t = pd.to_numeric(treatment, errors="coerce").dropna()
    c = pd.to_numeric(control, errors="coerce").dropna()
    if t.empty or c.empty:
        return None, float("nan"), float("nan")
    mean_t = float(t.mean())
    mean_c = float(c.mean())
    denom = math.sqrt((float(t.var(ddof=0)) + float(c.var(ddof=0))) / 2)
    return (None if denom == 0 else (mean_t - mean_c) / denom, mean_t, mean_c)


def _smd_rows_for_split(
    split: str,
    frame: Any,
    treatment_column: str,
    covariates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    treated = frame[frame[treatment_column] == 1]
    control = frame[frame[treatment_column] == 0]
    for item in covariates:
        variable = item["name"]
        variable_type = item["variable_type"]
        if variable_type == "numeric":
            smd, treated_value, control_value = _smd_numeric(treated[variable], control[variable])
            abs_smd = None if smd is None else abs(float(smd))
            rows.append(_smd_row(split, item, None, treated_value, control_value, smd, abs_smd, config))
        elif variable_type == "binary":
            smd, treated_value, control_value = _smd_binary(treated[variable], control[variable])
            abs_smd = None if smd is None else abs(float(smd))
            rows.append(_smd_row(split, item, None, treated_value, control_value, smd, abs_smd, config))
        else:
            levels = sorted(frame[variable].dropna().unique(), key=lambda value: str(value))
            for level in levels:
                t_rate = float((treated[variable] == level).mean()) if len(treated) else 0.0
                c_rate = float((control[variable] == level).mean()) if len(control) else 0.0
                denom = math.sqrt(((t_rate * (1 - t_rate)) + (c_rate * (1 - c_rate))) / 2)
                smd = None if denom == 0 else (t_rate - c_rate) / denom
                abs_smd = None if smd is None else abs(float(smd))
                rows.append(_smd_row(split, item, str(level), t_rate, c_rate, smd, abs_smd, config))
    return rows


def _smd_row(
    split: str,
    item: dict[str, Any],
    level: str | None,
    treatment_value: float,
    control_value: float,
    smd: float | None,
    abs_smd: float | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "split": split,
        "variable": item["name"],
        "variable_type": item["variable_type"],
        "level": level,
        "treatment_value_or_rate": treatment_value,
        "control_value_or_rate": control_value,
        "smd": "" if smd is None else float(smd),
        "abs_smd": "" if abs_smd is None else float(abs_smd),
        "status": _classify_smd_status(abs_smd, config),
        "is_key": bool(item.get("is_key")),
    }


def _smd_summary(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    values = [float(row["abs_smd"]) for row in rows if row["abs_smd"] != ""]
    statuses = [row["status"] for row in rows] or ["warning"]
    return {
        "checked_variables": len({row["variable"] for row in rows}),
        "variables_over_0_1": len(
            {
                row["variable"]
                for row in rows
                if row["abs_smd"] != ""
                and float(row["abs_smd"]) >= float(config["smd_warning_threshold"])
            }
        ),
        "variables_over_0_2": len(
            {
                row["variable"]
                for row in rows
                if row["abs_smd"] != ""
                and float(row["abs_smd"]) >= float(config["smd_severe_threshold"])
            }
        ),
        "max_abs_smd": max(values) if values else None,
        "status": _max_risk(statuses),
    }


def _validate_plan_payload(payload: dict[str, Any], frames: dict[str, Any]) -> None:
    if not isinstance(payload.get("covariates"), list):
        raise MissingInputError("homogeneity_covariate_plan.payload.covariates must be a list.")
    columns_by_split = {split: set(frame.columns) for split, frame in frames.items()}
    for item in payload["covariates"]:
        if not item.get("included"):
            continue
        name = item.get("name")
        if not name:
            raise MissingInputError("Included covariates require a name.")
        missing = [split for split, columns in columns_by_split.items() if name not in columns]
        if missing:
            raise MissingInputError(f"Covariate {name} is missing from splits: {', '.join(missing)}.")


def _write_covariate_plan_draft(
    run_dir: Path,
    *,
    input_paths: dict[str, Any],
    payload: dict[str, Any],
    prior_plan_path: Path | None,
) -> Path:
    version = _artifact_revision(prior_plan_path)
    artifact_path = run_dir / "artifacts" / f"homogeneity_covariate_plan.draft.v{version}.json"
    artifact = _artifact_envelope(
        artifact_status="draft",
        artifact_version=version,
        input_paths=input_paths,
        payload=payload,
        validation={
            "filled_fields": ["covariates", "excluded_variables", "max_categorical_levels"],
            "missing_fields": [],
            "assumptions": ["Pre-treatment covariate status requires user confirmation."],
            "warnings": [],
        },
    )
    write_json(artifact_path, artifact)
    return artifact_path


def _artifact_revision(prior_plan_path: Path | None) -> int:
    if not prior_plan_path:
        return 1
    try:
        prior = read_json(prior_plan_path)
        return int(prior.get("artifact_version") or 0) + 1
    except Exception:  # noqa: BLE001
        return 1


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_diagnostic_config_snapshot(run_dir: Path, snapshot: dict[str, Any]) -> Path:
    path = run_dir / "artifacts" / "diagnostic_config.json"
    write_json(path, snapshot)
    return path


def _write_report(run_dir: Path, data: dict[str, Any]) -> Path:
    report_path = run_dir / "report.md"
    language = str(data.get("language") or "zh-CN")
    zh = is_zh(language)
    summary = data["balance_summary"]
    recommended = _recommended_action(data["overall_status"])
    main_drivers = [
        f"{row['split']}:{row['variable']}" for row in data["top_imbalanced_variables"][:5]
    ]
    confirmation_required = str(data["overall_status"] in {"severe", "critical"}).lower()
    max_categorical_levels = data["config"]["effective_values"]["max_categorical_levels"]
    auc_header = (
        "| Split | Rows | Treatment Rows | Control Rows | AUC | Status | Top Predictive Variables |"
    )
    lines = (
        [
            "# 样本同质性检查",
            "",
            "## 摘要",
            "",
            f"- 诊断完整性：{data['diagnostic_completeness']}",
            f"- 平衡风险等级：{_risk_title(data['overall_status'])}",
            f"- 总体结论：treatment/control 平衡风险为 `{data['overall_status']}`。",
            f"- 主要驱动因素：{', '.join(main_drivers) if main_drivers else '无'}",
            f"- 建议动作：`{recommended}`",
            f"- 报告语言：`{language}`",
            "",
            "## 上游上下文",
            "",
            "- 建模样本规格：已确认",
            "- 协变量方案：已确认",
            "",
            "## 协变量集合",
            "",
            f"- 纳入变量数量：{summary['checked_covariates']}",
            f"- 类别变量最大水平数：{max_categorical_levels}",
            "- 完整协变量列表：见附录 `covariates_path`。",
            "",
            "## Treatment 可预测性 AUC",
            "",
            "| 切分 | 行数 | Treatment 行数 | Control 行数 | AUC | 状态 | 主要预测变量 |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        if zh
        else [
            "# Sample Homogeneity Check",
            "",
            "## Summary",
            "",
            f"- Diagnostic completeness: {data['diagnostic_completeness']}",
            f"- Balance risk level: {_risk_title(data['overall_status'])}",
            f"- Overall conclusion: treatment/control balance risk is `{data['overall_status']}`.",
            f"- Main drivers: {', '.join(main_drivers) if main_drivers else 'none'}",
            f"- Recommended action: `{recommended}`",
            f"- Report language: `{language}`",
            "",
            "## Upstream Context",
            "",
            "- Modeling sample spec: confirmed",
            "- Covariate plan: confirmed",
            "",
            "## Covariate Set",
            "",
            f"- Included variables count: {summary['checked_covariates']}",
            f"- Categorical max levels: {max_categorical_levels}",
            "- Full covariate list: see appendix `covariates_path`.",
            "",
            "## Treatment Predictability AUC",
            "",
            auc_header,
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for split, item in sorted(data["auc_by_split"].items(), key=lambda entry: _split_display_key(entry[0])):
        top = ", ".join(item.get("top_predictive_variables") or [])
        rows = item.get("rows", "")
        treatment_rows = item.get("treatment_rows", "")
        control_rows = item.get("control_rows", "")
        auc = _format_float(item.get("auc"))
        status = item.get("status")
        lines.append(f"| {split} | {rows} | {treatment_rows} | {control_rows} | {auc} | {status} | {top} |")
    lines.extend(
        [
            "",
            (
                "完整特征重要性：见附录 `auc_feature_importance_path`。"
                if zh
                else "Full feature importance: see appendix `auc_feature_importance_path`."
            ),
            "",
            "## SMD 平衡摘要" if zh else "## SMD Balance Summary",
            "",
            (
                "| 切分 | 已检查变量 | 变量数 >= 0.1 | 变量数 >= 0.2 | 最大绝对 SMD | 状态 |"
                if zh
                else "| Split | Checked Variables | Variables >= 0.1 | Variables >= 0.2 | Max Abs SMD | Status |"
            ),
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for split, item in sorted(
        data["smd_summary_by_split"].items(), key=lambda entry: _split_display_key(entry[0])
    ):
        max_abs_smd = _format_float(item["max_abs_smd"])
        lines.append(
            f"| {split} | {item['checked_variables']} | {item['variables_over_0_1']} | "
            f"{item['variables_over_0_2']} | {max_abs_smd} | {item['status']} |"
        )
    lines.extend(
        [
            "",
            "## 主要不平衡变量" if zh else "## Top Imbalanced Variables",
            "",
            (
                "| 切分 | 变量 | 类型 | 水平 | Treatment 值/比例 | Control 值/比例 | SMD | 状态 |"
                if zh
                else "| Split | Variable | Type | Level | Treatment Value/Rate | Control Value/Rate | SMD | Status |"
            ),
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in data["top_imbalanced_variables"]:
        treatment_value = _format_float(row["treatment_value_or_rate"])
        control_value = _format_float(row["control_value_or_rate"])
        smd = _format_float(row["smd"])
        lines.append(
            f"| {row['split']} | {row['variable']} | {row['variable_type']} | "
            f"{row.get('level') or ''} | {treatment_value} | {control_value} | {smd} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            (
                "完整 SMD 明细：见附录 `smd_detail_path`。"
                if zh
                else "Full SMD detail: see appendix `smd_detail_path`."
            ),
            "",
            "## 跳过的变量" if zh else "## Skipped Variables",
            "",
        ]
    )
    if data["skipped_variables"]:
        lines.extend(
            f"- `{item['variable']}` in `{item.get('split', 'plan')}`: {item['reason']}. {item['suggested_fix']}"
            for item in data["skipped_variables"]
        )
    else:
        lines.append("无。" if zh else "None.")
    lines.extend(
        [
            "",
            "## 建议" if zh else "## Recommendations",
            "",
            f"- {'建议下一步' if zh else 'Recommended next step'}：`{recommended}`",
            f"- {'是否需要用户确认' if zh else 'User confirmation required'}：{confirmation_required}",
            (
                "- severe/critical 风险继续下游前，需要用户明确确认接受风险。"
                if zh
                else "- Severe/critical risk requires explicit user acceptance before downstream continuation."
            ),
            "",
            "## 方法说明" if zh else "## Method Notes",
            "",
            (
                "- AUC 使用 split 内 LightGBM 诊断分类器，根据已确认协变量预测 treatment 分配。"
                if zh
                else "- AUC uses a split-local LightGBM diagnostic classifier predicting treatment assignment from confirmed covariates."
            ),
            (
                "- SMD 对数值变量使用合并标准差，对二元变量使用合并 Bernoulli 方差，对类别变量使用 one-vs-rest 水平。"
                if zh
                else "- SMD uses pooled standard deviation for numeric variables, pooled Bernoulli variance for binary variables, and one-vs-rest levels for categorical variables."
            ),
            "",
            "## 附录：系统引用" if zh else "## Appendix: System References",
            "",
            f"- modeling_sample_spec_path: `{data['modeling_sample_spec_path']}`",
            f"- homogeneity_covariate_plan_path: `{data['homogeneity_covariate_plan_path']}`",
            f"- result_path: `{(run_dir / 'results' / 'run_diagnostics.result.json').resolve()}`",
            f"- diagnostic_config_path: `{data['diagnostic_config_path']}`",
            f"- covariates_path: `{data['covariates_path']}`",
            f"- auc_feature_importance_path: `{data['auc_feature_importance_path']}`",
            f"- smd_detail_path: `{data['smd_detail_path']}`",
            "",
        ]
    )
    write_text(report_path, "\n".join(lines))
    return report_path


def _format_float(value: Any) -> str:
    if value in {None, ""} or (isinstance(value, float) and math.isnan(value)):
        return "not available"
    return f"{float(value):.4f}"


def _language_from_task_config(task_config: dict[str, Any] | None) -> str:
    if not task_config:
        return "zh-CN"
    return str((task_config.get("payload") or {}).get("report_preferences", {}).get("language") or "zh-CN")


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


def _legacy_ref_result(
    run_dir: Path, phase: str, rejected_fields: list[dict[str, str]]
) -> dict[str, Any]:
    field_names = ", ".join(item["field"] for item in rejected_fields)
    issues = [
        {
            "code": "LEGACY_REF_FIELD_NOT_ACCEPTED",
            "level": "critical",
            "blocking": True,
            "field": item["field"],
            "suggested_field": item["suggested_field"],
            "message": (
                f"{item['field']} is not accepted by self-contained runners; "
                f"use {item['suggested_field']}."
            ),
        }
        for item in rejected_fields
    ]
    next_steps = [
        {
            "action": "replace_legacy_ref_field",
            "field": item["field"],
            "suggested_field": item["suggested_field"],
            "reason": "Self-contained skill handoff uses explicit filesystem paths.",
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
            "homogeneity_covariate_plan_path": None,
            "sample_homogeneity_result_path": None,
            "report_path": None,
        },
        issues=issues,
        progress=[
            {
                "step": "reject_legacy_ref_fields",
                "status": "needs_input",
                "message": "Replace legacy _ref inputs with explicit _path fields.",
            }
        ],
        next_steps=next_steps,
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
    user_interaction_context: dict[str, Any] | None = None,
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
    result["user_interaction"] = build_user_interaction(result, **(user_interaction_context or {}))
    return result


def _needs_input_result(
    run_dir: Path, phase: str, message: str, missing_fields: list[str] | None = None
) -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="needs_input",
        summary=message,
        input_paths={},
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "missing_fields": missing_fields or [],
            "report_path": None,
        },
        issues=[
            {
                "code": "MISSING_OR_INVALID_INPUT",
                "level": "critical",
                "blocking": True,
                "message": message,
            }
        ],
        progress=[{"step": phase, "status": "needs_input", "message": message}],
    )


def _unsupported_result(run_dir: Path, phase: str, message: str) -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="failed",
        summary=message,
        input_paths={},
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "unsupported_reason": message,
            "report_path": None,
        },
        error={
            "code": "UNSUPPORTED_INPUT",
            "message": message,
            "recoverable": True,
            "retryable": False,
            "raw_error": None,
        },
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


def _diagnostic_failure_result(run_dir: Path, phase: str, message: str) -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="failed",
        summary=message,
        input_paths={},
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "report_path": None,
        },
        issues=[
            {
                "code": "SAMPLE_HOMOGENEITY_DIAGNOSTIC_FAILED",
                "level": "critical",
                "blocking": True,
                "message": message,
            }
        ],
        progress=[{"step": phase, "status": "failed", "message": message}],
    )


def _unexpected_error_result(run_dir: Path, phase: str, exc: Exception) -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="failed",
        summary=f"{phase} failed unexpectedly.",
        input_paths={},
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "report_path": None,
        },
        error={
            "code": "SAMPLE_HOMOGENEITY_CHECK_FAILED",
            "message": f"{phase} failed unexpectedly.",
            "recoverable": False,
            "retryable": False,
            "raw_error": str(exc),
        },
        progress=[{"step": phase, "status": "failed", "message": str(exc)}],
    )


def _attach_transport_paths(
    result: dict[str, Any],
    output_dir: Path,
    run_dir: Path,
    result_path: Path,
) -> None:
    outputs = result.setdefault("outputs", {})
    result_path_value = to_run_relative_path(output_dir, result_path)
    outputs["flow_dir"] = to_run_relative_path(output_dir, run_dir)
    outputs["result_path"] = result_path_value
    if "sample_homogeneity_result_path" in outputs and outputs["sample_homogeneity_result_path"] is None:
        outputs["sample_homogeneity_result_path"] = result_path_value
    for step in result.get("next_steps") or []:
        inputs = step.get("inputs")
        if (
            isinstance(inputs, dict)
            and "sample_homogeneity_result_path" in inputs
            and inputs["sample_homogeneity_result_path"] is None
        ):
            inputs["sample_homogeneity_result_path"] = result_path_value


def _layout_error_stdout(exc: ProjectLayoutError) -> dict[str, Any]:
    return {
        "status": "needs_input",
        "summary": str(exc),
        "outputs": {"missing_fields": ["flow_dir" if exc.issue_code.startswith("FLOW_DIR") else "output_dir"], "report_path": None},
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


def _pd() -> Any:
    try:
        import pandas as pd  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise DataLoaderImportError("pandas", "Missing package: pandas") from exc
    return pd


def _np() -> Any:
    try:
        import numpy as np  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise PackageImportError("numpy", "Missing package: numpy") from exc
    return np


def _roc_auc_score() -> Any:
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise PackageImportError("scikit-learn", "Missing package: scikit-learn") from exc
    return roc_auc_score


def _train_test_split() -> Any:
    try:
        from sklearn.model_selection import train_test_split  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise PackageImportError("scikit-learn", "Missing package: scikit-learn") from exc
    return train_test_split


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
