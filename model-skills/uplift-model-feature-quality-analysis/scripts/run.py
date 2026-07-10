from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
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

SKILL_NAME = "uplift-model-feature-quality-analysis"
SKILL_CREATED_BY = "uplift-model-feature-quality-analysis-skill"
SPLIT_NAMES = ("train", "test", "valid", "oot")
DEFAULT_PSI_N_BINS = 10
EPSILON = 1e-6
ALLOWED_REF_FIELDS = {"data_ref"}
DEFAULT_THRESHOLDS: dict[str, float] = {
    "missing_rate_warning": 0.5,
    "missing_rate_exclude": 0.8,
    "psi_warning": 0.1,
    "psi_exclude": 0.25,
    "near_constant_mode_ratio": 0.99,
    "high_concentration_mode_ratio": 0.95,
}
DIAGNOSTICS_NOT_RUN = {
    "basic_quality": "not_run",
    "split_psi": "not_run",
    "monthly_psi": "not_run",
    "uplift_bivar": "not_run",
    "iv": "deferred_not_implemented",
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


def run_diagnostics(run_dir: Path, output_dir: Path, payload: dict[str, Any], phase: str) -> dict[str, Any]:
    try:
        scope = _parse_analysis_scope(payload, phase)
        context = _build_context(output_dir, payload)
        frames = context["frames"]
        candidates = context["candidates"]
        statuses = dict(DIAGNOSTICS_NOT_RUN)
        statuses.update({name: "not_requested" for name in DIAGNOSTICS_NOT_RUN if name != "iv"})
        artifact_paths: dict[str, str | None] = {
            "feature_basic_quality_path": None,
            "feature_psi_split_path": None,
            "feature_psi_monthly_path": None,
            "feature_uplift_bivar_path": None,
        }
        basic_rows: list[dict[str, Any]] = []
        psi_rows: list[dict[str, Any]] = []
        psi_status = "not_requested"
        monthly_summary: dict[str, Any] = {}
        bivar_summary: dict[str, Any] = {}
        artifacts: list[dict[str, Any]] = []

        selected = scope["diagnostics"]
        if "basic_quality" in selected:
            basic_rows = _compute_basic_quality(frames, candidates)
            path = run_dir / "artifacts" / "feature_basic_quality.v1.csv"
            _write_csv(path, basic_rows)
            artifact_paths["feature_basic_quality_path"] = str(path.resolve())
            statuses["basic_quality"] = "success"
            artifacts.append({"kind": "diagnostic_table", "path": str(path.resolve())})
        if "split_psi" in selected:
            psi_rows, psi_status = _compute_split_psi(
                frames,
                candidates,
                n_bins=int(scope["psi_n_bins"]),
            )
            path = run_dir / "artifacts" / "feature_psi_split.v1.csv"
            _write_csv(path, psi_rows)
            artifact_paths["feature_psi_split_path"] = str(path.resolve())
            statuses["split_psi"] = psi_status
            artifacts.append({"kind": "diagnostic_table", "path": str(path.resolve())})
        if "monthly_psi" in selected:
            date_column = payload.get("date_column") or context["task"]["payload"].get("time_column")
            if not date_column:
                raise MissingInputError("monthly_psi requires date_column or task_config.payload.time_column.")
            scope_name = str(payload.get("dataset_scope") or "train")
            if scope_name not in frames:
                raise MissingInputError(f"dataset_scope is not available: {scope_name}")
            rows, monthly_summary = _compute_monthly_psi(
                frames[scope_name],
                candidate_features=candidates,
                date_column=str(date_column),
                dataset_scope=scope_name,
                baseline_month=payload.get("baseline_month"),
                comparison_month_start=payload.get("comparison_month_start"),
                comparison_month_end=payload.get("comparison_month_end"),
                n_bins=int(scope["psi_n_bins"]),
            )
            path = run_dir / "artifacts" / "feature_psi_monthly.v1.csv"
            _write_csv(path, rows)
            artifact_paths["feature_psi_monthly_path"] = str(path.resolve())
            statuses["monthly_psi"] = "success"
            artifacts.append({"kind": "diagnostic_table", "path": str(path.resolve())})
        if "uplift_bivar" in selected:
            scope_name = str(payload.get("dataset_scope") or "train")
            if scope_name not in frames:
                raise MissingInputError(f"dataset_scope is not available: {scope_name}")
            columns = context["spec"]["payload"]["columns"]
            rows, bivar_summary, bivar_issues = _compute_uplift_bivar(
                frames[scope_name],
                candidate_features=candidates,
                treatment_column=columns["treatment"],
                outcome_column=columns["outcome"],
                outcome_type=columns["outcome_type"],
                max_numeric_bins=int(payload.get("max_numeric_bins", DEFAULT_PSI_N_BINS)),
                max_categorical_levels=int(payload.get("max_categorical_levels", 20)),
            )
            context["issues"].extend(bivar_issues)
            path = run_dir / "artifacts" / "feature_uplift_bivar.v1.csv"
            _write_csv(path, rows)
            artifact_paths["feature_uplift_bivar_path"] = str(path.resolve())
            statuses["uplift_bivar"] = "success"
            artifacts.append({"kind": "diagnostic_table", "path": str(path.resolve())})

        recommendation_path = None
        recommendation = None
        source_result_path = str((run_dir / "results" / f"{phase}.result.json").resolve())
        if scope["generate_recommendation"]:
            if not basic_rows and not psi_rows:
                raise MissingInputError(
                    "A recommendation requires basic_quality or split_psi diagnostics."
                )
            recommendation_path, recommendation = _build_recommendation(
                run_dir,
                candidate_features=candidates,
                basic_quality_rows=basic_rows,
                psi_rows=psi_rows,
                source_result_path=source_result_path,
                diagnostic_paths=artifact_paths,
                thresholds=DEFAULT_THRESHOLDS,
            )
            artifacts.append(
                {"kind": "feature_selection_recommendation", "path": recommendation_path}
            )

        candidate_features_path = run_dir / "artifacts" / "candidate_features.v1.json"
        write_json(candidate_features_path, {"features": candidates})
        outputs = _diagnostic_outputs(
            candidates=candidates,
            statuses=statuses,
            basic_rows=basic_rows,
            psi_rows=psi_rows,
            psi_status=psi_status,
            artifact_paths=artifact_paths,
            recommendation_path=recommendation_path,
            recommendation=recommendation,
            monthly_summary=monthly_summary,
            bivar_summary=bivar_summary,
            psi_n_bins=int(scope["psi_n_bins"]),
        )
        outputs["flow_dir"] = str(run_dir.resolve())
        outputs["candidate_features_path"] = str(candidate_features_path.resolve())
        outputs["feature_plan_path"] = None
        report_path = _write_report(
            run_dir,
            output_dir=output_dir,
            data={
                **outputs,
                "input_paths": context["input_paths"],
                "result_path": str((run_dir / "results" / f"{phase}.result.json").resolve()),
                "language": _language_from_task_config(context["task"]),
            },
        )
        outputs["report_path"] = str(report_path.resolve())
        artifacts.append({"kind": "report", "path": str(report_path.resolve())})
        return _result(
            run_dir=run_dir,
            phase=phase,
            status="success",
            summary="Requested feature diagnostics completed.",
            input_paths=context["input_paths"],
            outputs=outputs,
            issues=context["issues"],
            artifacts=artifacts,
            progress=[
                {"step": name, "status": statuses[name], "message": f"{name}: {statuses[name]}"}
                for name in selected
            ]
            + [{"step": "report", "status": "success", "message": "Feature report written."}],
            next_steps=_recommendation_steps(recommendation_path, source_result_path, run_dir),
        )
    except (DataLoaderImportError, PackageImportError) as exc:
        return _dependency_failure_result(run_dir, phase, exc)
    except NeedsConfirmationError as exc:
        return _needs_confirmation_result(run_dir, phase, str(exc))
    except (MissingInputError, DataSourceError) as exc:
        return _needs_input_result(run_dir, phase, str(exc), getattr(exc, "missing_fields", []))
    except Exception as exc:  # noqa: BLE001
        return _unexpected_error_result(run_dir, phase, exc)


def run_selection_only(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    phase = "selection_only"
    try:
        _require_selection_scope(payload)
        source_path = _required_input_path(output_dir, payload.get("source_result_path"), "source_result_path")
        source = _read_successful_feature_result(source_path)
        source_inputs = source.get("input_paths") or {}
        context = _build_context(output_dir, {**source_inputs, **payload})
        _require_same_paths(output_dir, source_inputs, context["input_paths"], ("task_config_path", "modeling_sample_spec_path"))
        thresholds = payload.get("new_thresholds")
        if not isinstance(thresholds, dict) or not thresholds:
            raise MissingInputError("selection_only requires new_thresholds.")
        outputs = source.get("outputs") or {}
        candidates = _load_source_candidates(outputs, output_dir)
        if set(candidates) != set(context["candidates"]):
            raise NeedsConfirmationError(
                "Candidate features changed since diagnostics; rerun diagnostics_only before selection_only."
            )
        basic_rows = _read_diagnostic_csv(outputs.get("feature_basic_quality_path"), "basic", output_dir)
        psi_rows = _read_diagnostic_csv(outputs.get("feature_psi_split_path"), "psi", output_dir)
        if not basic_rows and not psi_rows:
            raise MissingInputError("Source diagnostics do not contain reusable selection inputs.")
        result_path = str((run_dir / "results" / "selection_only.result.json").resolve())
        recommendation_path, recommendation = _build_recommendation(
            run_dir,
            candidate_features=candidates,
            basic_quality_rows=basic_rows,
            psi_rows=psi_rows,
            source_result_path=result_path,
            diagnostic_paths={
                "feature_basic_quality_path": outputs.get("feature_basic_quality_path"),
                "feature_psi_split_path": outputs.get("feature_psi_split_path"),
                "feature_psi_monthly_path": outputs.get("feature_psi_monthly_path"),
                "feature_uplift_bivar_path": outputs.get("feature_uplift_bivar_path"),
            },
            thresholds=thresholds,
        )
        report_path = _write_report(
            run_dir,
            output_dir=output_dir,
            data={
                **_diagnostic_outputs(
                    candidates=candidates,
                    statuses=outputs.get("diagnostics_status") or dict(DIAGNOSTICS_NOT_RUN),
                    basic_rows=basic_rows,
                    psi_rows=psi_rows,
                    psi_status=str((outputs.get("diagnostics_status") or {}).get("split_psi", "success")),
                    artifact_paths={
                        "feature_basic_quality_path": outputs.get("feature_basic_quality_path"),
                        "feature_psi_split_path": outputs.get("feature_psi_split_path"),
                        "feature_psi_monthly_path": outputs.get("feature_psi_monthly_path"),
                        "feature_uplift_bivar_path": outputs.get("feature_uplift_bivar_path"),
                    },
                    recommendation_path=recommendation_path,
                    recommendation=recommendation,
                    monthly_summary=outputs.get("monthly_psi_summary") or {},
                    bivar_summary=outputs.get("uplift_bivar_summary") or {},
                    psi_n_bins=(outputs.get("diagnostics_summary") or {}).get("psi_n_bins"),
                ),
                "input_paths": context["input_paths"],
                "result_path": str((run_dir / "results" / "selection_only.result.json").resolve()),
                "language": _language_from_task_config(context["task"]),
            },
        )
        outputs_payload = {
            "flow_dir": str(run_dir.resolve()),
            "source_result_path": str(source_path),
            "feature_selection_recommendation_path": recommendation_path,
            "feature_plan_path": None,
            "report_path": str(report_path.resolve()),
            "selection_summary": _selection_summary(recommendation, recommendation_path),
        }
        return _result(
            run_dir=run_dir,
            phase=phase,
            status="success",
            summary="Selection thresholds were applied to existing diagnostics.",
            input_paths={**context["input_paths"], "source_result_path": str(source_path)},
            outputs=outputs_payload,
            artifacts=[
                {"kind": "feature_selection_recommendation", "path": recommendation_path},
                {"kind": "report", "path": str(report_path.resolve())},
            ],
            progress=[{"step": "selection", "status": "success", "message": "Diagnostics reused."}],
            next_steps=_recommendation_steps(recommendation_path, result_path, run_dir),
        )
    except (DataLoaderImportError, PackageImportError) as exc:
        return _dependency_failure_result(run_dir, phase, exc)
    except NeedsConfirmationError as exc:
        return _needs_confirmation_result(run_dir, phase, str(exc))
    except (MissingInputError, DataSourceError) as exc:
        return _needs_input_result(run_dir, phase, str(exc), getattr(exc, "missing_fields", []))
    except Exception as exc:  # noqa: BLE001
        return _unexpected_error_result(run_dir, phase, exc)


def run_accept_recommendation(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    phase = "accept_recommendation"
    try:
        recommendation_path, recommendation, source_path, source = _recommendation_context(output_dir, payload)
        input_paths = _validated_recommendation_inputs(output_dir, payload, source)
        existing_path = _optional_input_path(output_dir, payload.get("existing_feature_plan_path"))
        existing = _read_feature_plan(existing_path, require_confirmed=True) if existing_path else None
        if (
            existing
            and existing.get("source_type") == "accepted_recommendation"
            and _same_path(existing.get("input_paths", {}).get("recommendation_path"), recommendation_path, output_dir)
        ):
            selected = _selected_features_from_plan(existing, output_dir)
            report_path = _write_feature_plan_report(run_dir, output_dir, existing_path)
            result = _result(
                run_dir=run_dir,
                phase=phase,
                status="success",
                summary="The confirmed feature plan already uses this recommendation.",
                input_paths={
                    **input_paths,
                    "recommendation_path": str(recommendation_path),
                    "source_result_path": str(source_path),
                },
                outputs={
                    "flow_dir": str(run_dir.resolve()),
                    "reused_existing": True,
                    "feature_plan_path": str(existing_path.resolve()),
                    "selected_features": {"count": len(selected), "preview": selected[:5]},
                    "report_path": str(report_path.resolve()),
                },
                artifacts=[{"kind": "report", "path": str(report_path.resolve())}],
                progress=[
                    {"step": "duplicate_guard", "status": "success", "message": "Existing plan reused."}
                ],
            )
            result["user_interaction"] = _feature_interaction(result, selected)
            return result
        selected = _load_feature_list_path(recommendation["selected_features"]["features_path"], output_dir)
        previous = _selected_features_from_plan(existing, output_dir) if existing else None
        plan_path, _plan = _write_feature_plan(
            run_dir,
            status="confirmed",
            input_paths={
                **input_paths,
                "recommendation_path": str(recommendation_path),
                "source_result_path": str(source_path),
            },
            selected_features=selected,
            source_type="accepted_recommendation",
            diagnostics_status=_diagnostics_completeness(source),
            recommended_selected_count=recommendation["summary"]["recommended_selected_count"],
            confirmed_by=str(payload.get("confirmed_by") or "user"),
        )
        report_path = _write_feature_plan_report(run_dir, output_dir, plan_path)
        result = _result(
            run_dir=run_dir,
            phase=phase,
            status="success",
            summary="Recommendation accepted as a confirmed feature plan.",
            input_paths={
                **input_paths,
                "recommendation_path": str(recommendation_path),
                "source_result_path": str(source_path),
            },
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "reused_existing": False,
                "feature_plan_path": str(plan_path.resolve()),
                "selected_features": {"count": len(selected), "preview": selected[:5]},
                "report_path": str(report_path.resolve()),
            },
            artifacts=[
                {"kind": "feature_plan", "path": str(plan_path.resolve())},
                {"kind": "report", "path": str(report_path.resolve())},
            ],
            progress=[
                {"step": "accept_recommendation", "status": "success", "message": "Confirmed plan written."}
            ],
        )
        result["user_interaction"] = _feature_interaction(result, selected, previous)
        return result
    except (DataLoaderImportError, PackageImportError) as exc:
        return _dependency_failure_result(run_dir, phase, exc)
    except NeedsConfirmationError as exc:
        return _needs_confirmation_result(run_dir, phase, str(exc))
    except (MissingInputError, DataSourceError) as exc:
        return _needs_input_result(run_dir, phase, str(exc), getattr(exc, "missing_fields", []))
    except Exception as exc:  # noqa: BLE001
        return _unexpected_error_result(run_dir, phase, exc)


def run_modify_recommendation(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    phase = "modify_recommendation"
    try:
        recommendation_path, recommendation, source_path, source = _recommendation_context(output_dir, payload)
        input_paths = _validated_recommendation_inputs(output_dir, payload, source)
        context = _build_context(output_dir, {**input_paths, **payload})
        original = _load_feature_list_path(recommendation["selected_features"]["features_path"], output_dir)
        includes = _dedupe(payload.get("include_features") or [])
        excludes = _dedupe(payload.get("exclude_features") or [])
        overlap = sorted(set(includes) & set(excludes))
        if overlap:
            raise MissingInputError("Features cannot be both included and excluded: " + ", ".join(overlap))
        _validate_features(includes, context["train_columns"], context["forced_exclude"], allow_empty=True)
        missing_excludes = [item for item in excludes if item not in context["train_columns"]]
        if missing_excludes:
            raise MissingInputError("Excluded features are not present in train: " + ", ".join(missing_excludes))
        final = [item for item in original if item not in set(excludes)]
        final.extend(item for item in includes if item not in final)
        final = _validate_features(final, context["train_columns"], context["forced_exclude"])
        added = [item for item in final if item not in original]
        removed = [item for item in original if item not in final]
        plan_path, _plan = _write_feature_plan(
            run_dir,
            status="draft",
            input_paths={
                **input_paths,
                "recommendation_path": str(recommendation_path),
                "source_result_path": str(source_path),
            },
            selected_features=final,
            source_type="modified_recommendation",
            diagnostics_status=_diagnostics_completeness(source),
            recommended_selected_count=recommendation["selected_features"]["count"],
            manual_include_count=len(added),
            manual_exclude_count=len(removed),
        )
        report_path = _write_feature_plan_report(run_dir, output_dir, plan_path)
        result = _result(
            run_dir=run_dir,
            phase=phase,
            status="needs_confirmation",
            summary="Modified feature plan draft created; confirm it before modeling.",
            input_paths={
                **input_paths,
                "recommendation_path": str(recommendation_path),
                "source_result_path": str(source_path),
            },
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "feature_plan_path": str(plan_path.resolve()),
                "feature_plan_status": "draft",
                "changes": {
                    "added_count": len(added),
                    "added_examples": added[:5],
                    "removed_count": len(removed),
                    "removed_examples": removed[:5],
                    "total_count": len(final),
                },
                "report_path": str(report_path.resolve()),
            },
            artifacts=[
                {"kind": "feature_plan", "path": str(plan_path.resolve())},
                {"kind": "report", "path": str(report_path.resolve())},
            ],
            progress=[{"step": phase, "status": "success", "message": "Draft written."}],
            next_steps=[_confirm_step(plan_path, run_dir)],
        )
        result["user_interaction"] = _feature_interaction(result, final, original)
        return result
    except (DataLoaderImportError, PackageImportError) as exc:
        return _dependency_failure_result(run_dir, phase, exc)
    except NeedsConfirmationError as exc:
        return _needs_confirmation_result(run_dir, phase, str(exc))
    except (MissingInputError, DataSourceError) as exc:
        return _needs_input_result(run_dir, phase, str(exc), getattr(exc, "missing_fields", []))
    except Exception as exc:  # noqa: BLE001
        return _unexpected_error_result(run_dir, phase, exc)


def run_manual_feature_plan(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    phase = "manual_feature_plan"
    try:
        context = _build_context(output_dir, payload)
        selected = _resolve_feature_list(output_dir, payload)
        if selected is None:
            raise MissingInputError("manual_feature_plan requires feature_list or feature_list_path.")
        selected = _validate_features(selected, context["train_columns"], context["forced_exclude"])
        plan_path, _plan = _write_feature_plan(
            run_dir,
            status="draft",
            input_paths=context["input_paths"],
            selected_features=selected,
            source_type="manual_user_input",
            diagnostics_status="not_run",
            recommended_selected_count=None,
            manual_include_count=len(selected),
        )
        report_path = _write_feature_plan_report(run_dir, output_dir, plan_path)
        return _result(
            run_dir=run_dir,
            phase=phase,
            status="needs_confirmation",
            summary="Manual feature plan draft created.",
            input_paths=context["input_paths"],
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "feature_plan_path": str(plan_path.resolve()),
                "feature_plan_status": "draft",
                "selected_features": {"count": len(selected), "preview": selected[:5]},
                "report_path": str(report_path.resolve()),
            },
            issues=context["issues"],
            artifacts=[
                {"kind": "feature_plan", "path": str(plan_path.resolve())},
                {"kind": "report", "path": str(report_path.resolve())},
            ],
            next_steps=[_confirm_step(plan_path, run_dir)],
        )
    except (DataLoaderImportError, PackageImportError) as exc:
        return _dependency_failure_result(run_dir, phase, exc)
    except NeedsConfirmationError as exc:
        return _needs_confirmation_result(run_dir, phase, str(exc))
    except (MissingInputError, DataSourceError) as exc:
        return _needs_input_result(run_dir, phase, str(exc), getattr(exc, "missing_fields", []))
    except Exception as exc:  # noqa: BLE001
        return _unexpected_error_result(run_dir, phase, exc)


def run_confirm_artifact(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    phase = "confirm_artifact"
    try:
        source_draft_path = _required_input_path(output_dir, payload.get("source_draft_path"), "source_draft_path")
        artifact_kind = str(payload.get("artifact_kind") or "feature_plan")
        if artifact_kind != "feature_plan":
            raise MissingInputError("uplift-model-feature-quality-analysis can only confirm feature_plan.")
        if payload.get("confirmed_payload") is not None:
            raise MissingInputError("confirm_artifact does not accept payload changes; rerun the planning action.")
        draft = _read_feature_plan(source_draft_path, require_draft=True)
        draft_inputs = draft.get("input_paths") or {}
        task_path = _required_input_path(
            output_dir,
            payload.get("task_config_path") or draft_inputs.get("task_config_path"),
            "task_config_path",
        )
        spec_path = _required_input_path(
            output_dir,
            payload.get("modeling_sample_spec_path") or draft_inputs.get("modeling_sample_spec_path"),
            "modeling_sample_spec_path",
        )
        _validate_feature_plan_lineage(draft, task_path, spec_path, output_dir)
        previous_path = _optional_input_path(output_dir, payload.get("previous_feature_plan_path"))
        previous = _selected_features_from_plan(_read_feature_plan(previous_path, require_confirmed=True), output_dir) if previous_path else None
        selected = _selected_features_from_plan(draft, output_dir)
        plan_path, _plan = _write_feature_plan(
            run_dir,
            status="confirmed",
            input_paths={**draft_inputs, "task_config_path": str(task_path), "modeling_sample_spec_path": str(spec_path)},
            selected_features=selected,
            source_type=str(draft.get("source_type") or "confirmed_draft"),
            diagnostics_status=str(draft.get("payload", {}).get("diagnostics_status") or "partial"),
            recommended_selected_count=draft.get("payload", {}).get("selection_summary", {}).get("recommended_selected_count"),
            manual_include_count=int(draft.get("payload", {}).get("selection_summary", {}).get("manual_include_count") or 0),
            manual_exclude_count=int(draft.get("payload", {}).get("selection_summary", {}).get("manual_exclude_count") or 0),
            confirmed_by=str(payload.get("confirmed_by") or "user"),
            source_draft_path=str(source_draft_path),
        )
        report_path = _write_feature_plan_report(run_dir, output_dir, plan_path)
        result = _result(
            run_dir=run_dir,
            phase=phase,
            status="success",
            summary="Feature plan confirmed.",
            input_paths={"source_draft_path": str(source_draft_path), "task_config_path": str(task_path), "modeling_sample_spec_path": str(spec_path)},
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "feature_plan_path": str(plan_path.resolve()),
                "feature_plan_status": "confirmed",
                "selected_features": {"count": len(selected), "preview": selected[:5]},
                "report_path": str(report_path.resolve()),
            },
            artifacts=[
                {"kind": "feature_plan", "path": str(plan_path.resolve())},
                {"kind": "report", "path": str(report_path.resolve())},
            ],
        )
        result["user_interaction"] = _feature_interaction(result, selected, previous)
        return result
    except (DataLoaderImportError, PackageImportError) as exc:
        return _dependency_failure_result(run_dir, phase, exc)
    except NeedsConfirmationError as exc:
        return _needs_confirmation_result(run_dir, phase, str(exc))
    except (MissingInputError, DataSourceError) as exc:
        return _needs_input_result(run_dir, phase, str(exc), getattr(exc, "missing_fields", []))
    except Exception as exc:  # noqa: BLE001
        return _unexpected_error_result(run_dir, phase, exc)


def run_reuse_feature_set(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    phase = "reuse_feature_set"
    try:
        previous_path = _required_input_path(
            output_dir, payload.get("previous_feature_plan_path"), "previous_feature_plan_path"
        )
        previous_plan = _read_feature_plan(previous_path, require_confirmed=True)
        context = _build_context(output_dir, payload)
        selected = _validate_features(
            _selected_features_from_plan(previous_plan, output_dir),
            context["train_columns"],
            context["forced_exclude"],
        )
        plan_path, _plan = _write_feature_plan(
            run_dir,
            status="confirmed",
            input_paths={**context["input_paths"], "previous_feature_plan_path": str(previous_path)},
            selected_features=selected,
            source_type="reused_previous_feature_set",
            diagnostics_status="not_run_for_selected_inputs",
            recommended_selected_count=None,
            confirmed_by=str(payload.get("confirmed_by") or "user"),
            selection_summary={"source": "reused_previous_feature_set"},
        )
        report_path = _write_feature_plan_report(run_dir, output_dir, plan_path)
        return _result(
            run_dir=run_dir,
            phase=phase,
            status="success",
            summary="Previous feature set reused for selected inputs without rerunning diagnostics.",
            input_paths={**context["input_paths"], "previous_feature_plan_path": str(previous_path)},
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "feature_plan_path": str(plan_path.resolve()),
                "diagnostics_status": "not_run_for_selected_inputs",
                "selected_features": {"count": len(selected), "preview": selected[:5]},
                "report_path": str(report_path.resolve()),
            },
            issues=[
                _issue(
                    code="FEATURE_DIAGNOSTICS_NOT_RUN_FOR_SELECTED_INPUTS",
                    level="warning",
                    blocking=False,
                    message="The reused feature set has not been quality-checked against the selected inputs.",
                    suggested_fix="Rerun feature diagnostics when quality evidence is required.",
                )
            ],
            artifacts=[
                {"kind": "feature_plan", "path": str(plan_path.resolve())},
                {"kind": "report", "path": str(report_path.resolve())},
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


def run_validate_feature_plan(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    phase = "validate_feature_plan"
    try:
        feature_plan_path = _required_input_path(output_dir, payload.get("feature_plan_path"), "feature_plan_path")
        task_path = _required_input_path(output_dir, payload.get("task_config_path"), "task_config_path")
        spec_path = _required_input_path(output_dir, payload.get("modeling_sample_spec_path"), "modeling_sample_spec_path")
        plan = _read_feature_plan(feature_plan_path, require_confirmed=True)
        _validate_feature_plan_lineage(plan, task_path, spec_path, output_dir)
        return _result(
            run_dir=run_dir,
            phase=phase,
            status="success",
            summary="Feature plan lineage matches the selected inputs.",
            input_paths={
                "task_config_path": str(task_path),
                "modeling_sample_spec_path": str(spec_path),
                "feature_plan_path": str(feature_plan_path),
            },
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "feature_plan_path": str(feature_plan_path),
                "report_path": None,
            },
        )
    except (MissingInputError, DataSourceError) as exc:
        return _needs_input_result(run_dir, phase, str(exc), getattr(exc, "missing_fields", []))
    except NeedsConfirmationError as exc:
        return _needs_confirmation_result(run_dir, phase, str(exc))
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
    phase = _normalize_action(action)
    try:
        run_dir = _flow_folder_for_action(output_dir, phase, body)
        request_path, result_path = next_action_paths(run_dir, phase or "unknown")
    except ProjectLayoutError as exc:
        print(json.dumps(_layout_error_stdout(exc), sort_keys=True))
        return 0
    write_json(request_path, request)
    supported = {
        "diagnostics_only",
        "selection_only",
        "accept_recommendation",
        "modify_recommendation",
        "manual_feature_plan",
        "confirm_artifact",
        "reuse_feature_set",
        "validate_feature_plan",
    }
    if request_error:
        result = _needs_input_result(run_dir, phase or "unknown", request_error, ["payload"])
    elif phase not in supported:
        result = _unsupported_result(run_dir, phase or "unknown", f"Unsupported action: {action}")
    else:
        legacy_fields = _legacy_ref_rejections(body)
        if legacy_fields:
            result = _legacy_ref_result(run_dir, phase, legacy_fields)
        elif phase == "diagnostics_only":
            result = run_diagnostics(run_dir, output_dir, body, phase)
        elif phase == "selection_only":
            result = run_selection_only(run_dir, output_dir, body)
        elif phase == "accept_recommendation":
            result = run_accept_recommendation(run_dir, output_dir, body)
        elif phase == "modify_recommendation":
            result = run_modify_recommendation(run_dir, output_dir, body)
        elif phase == "manual_feature_plan":
            result = run_manual_feature_plan(run_dir, output_dir, body)
        elif phase == "confirm_artifact":
            result = run_confirm_artifact(run_dir, output_dir, body)
        elif phase == "reuse_feature_set":
            result = run_reuse_feature_set(run_dir, output_dir, body)
        else:
            result = run_validate_feature_plan(run_dir, output_dir, body)
    _refresh_feature_recommendation_source(result, output_dir, result_path)
    _attach_transport_paths(result, output_dir, run_dir, result_path)
    result = relativize_paths(result, output_dir)
    write_json(result_path, result)
    write_flow_action_records(
        run_dir=output_dir,
        flow_dir=run_dir,
        skill_name=SKILL_NAME,
        action=phase or "unknown",
        request_path=request_path,
        result_path=result_path,
        result=result,
    )
    print(json.dumps(_stdout_payload(result), sort_keys=True))
    return 0


def _normalize_action(action: str) -> str:
    return action


def _parse_analysis_scope(payload: dict[str, Any], phase: str) -> dict[str, Any]:
    scope = payload.get("analysis_scope")
    if not isinstance(scope, dict):
        raise NeedsConfirmationError(
            "Show the candidate diagnostics and recommended analysis plan, then rerun diagnostics_only with analysis_scope.confirmed=true."
        )
    if scope.get("confirmed") is not True:
        raise NeedsConfirmationError(
            "analysis_scope.confirmed=true is required before diagnostics_only can run."
        )
    diagnostics = scope.get("diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise MissingInputError("analysis_scope.diagnostics must contain at least one analysis.")
    normalized = _dedupe(diagnostics)
    supported = {"basic_quality", "split_psi", "monthly_psi", "uplift_bivar"}
    unknown = [item for item in normalized if item not in supported]
    if unknown:
        raise MissingInputError("Unsupported diagnostics: " + ", ".join(unknown))
    return {
        "diagnostics": normalized,
        "generate_recommendation": False,
        "psi_n_bins": _positive_int(scope.get("psi_n_bins") or scope.get("n_bins") or payload.get("psi_n_bins") or payload.get("n_bins") or DEFAULT_PSI_N_BINS, "psi_n_bins"),
    }


def _require_selection_scope(payload: dict[str, Any]) -> None:
    scope = payload.get("selection_scope")
    if not isinstance(scope, dict) or scope.get("confirmed") is not True:
        raise NeedsConfirmationError(
            "Show the diagnostics summary and selection rules, then rerun selection_only with selection_scope.confirmed=true."
        )
    rule_source = scope.get("rule_source")
    if rule_source not in {"default", "custom"}:
        raise MissingInputError("selection_scope.rule_source must be either default or custom.")


def _build_context(
    output_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    task_path = _required_input_path(output_dir, payload.get("task_config_path"), "task_config_path")
    spec_path = _required_input_path(output_dir, payload.get("modeling_sample_spec_path"), "modeling_sample_spec_path")
    task = _read_artifact(task_path, artifact_kind="task_config", require_confirmed=True)
    spec = _read_artifact(spec_path, artifact_kind="modeling_sample_spec", require_confirmed=True)
    if not _same_path(spec.get("input_paths", {}).get("task_config_path"), task_path, output_dir):
        raise NeedsConfirmationError(
            "modeling_sample_spec_path lineage does not match task_config_path."
        )
    input_paths: dict[str, Any] = {
        "task_config_path": str(task_path),
        "modeling_sample_spec_path": str(spec_path),
    }
    issues: list[dict[str, Any]] = []
    frames = _load_enabled_split_frames(spec)
    train_columns = list(frames["train"].columns)
    forced_exclude = list(spec.get("payload", {}).get("feature_columns", {}).get("forced_exclude") or [])
    forced_exclude.extend(task.get("payload", {}).get("exclude_columns") or [])
    candidates, candidate_issues = _derive_candidate_features(
        train_columns=train_columns,
        forced_exclude=forced_exclude,
        feature_list=_resolve_feature_list(output_dir, payload),
        exclude_list=_normalize_list(payload.get("exclude_list") or payload.get("non_feature_list") or []),
    )
    if any(item["blocking"] for item in candidate_issues):
        raise MissingInputError(candidate_issues[0]["message"])
    return {
        "task_path": task_path,
        "task": task,
        "spec_path": spec_path,
        "spec": spec,
        "frames": frames,
        "train_columns": train_columns,
        "forced_exclude": forced_exclude,
        "candidates": candidates,
        "input_paths": input_paths,
        "issues": issues + candidate_issues,
    }


def _load_enabled_split_frames(spec: dict[str, Any]) -> dict[str, Any]:
    datasets = spec.get("payload", {}).get("datasets") or {}
    frames: dict[str, Any] = {}
    for split in SPLIT_NAMES:
        dataset_path = datasets.get(split)
        if not dataset_path:
            continue
        frames[split] = load_table({"kind": "local_csv", "path": str(dataset_path)})
    if "train" not in frames:
        raise MissingInputError("modeling_sample_spec must reference a train dataset.")
    return frames


def _resolve_feature_list(output_dir: Path, payload: dict[str, Any]) -> list[str] | None:
    if payload.get("feature_list") is not None:
        return _normalize_list(payload.get("feature_list"))
    if payload.get("feature_list_path"):
        path = _required_input_path(output_dir, payload.get("feature_list_path"), "feature_list_path")
        return _read_feature_list_file(path)
    return None


def _derive_candidate_features(
    *,
    train_columns: list[str],
    forced_exclude: list[str],
    feature_list: list[str] | None,
    exclude_list: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    forced = set(forced_exclude)
    excludes = set(exclude_list)
    if feature_list is not None:
        candidates = _dedupe(feature_list)
        conflict = sorted(set(candidates) & excludes)
        if conflict:
            raise NeedsConfirmationError("feature_list and exclude_list conflict for: " + ", ".join(conflict[:20]))
    else:
        candidates = [column for column in train_columns if column not in forced and column not in excludes]
    missing = [feature for feature in candidates if feature not in train_columns]
    if missing:
        raise MissingInputError("Feature list contains columns not present in train: " + ", ".join(missing[:20]))
    forced_requested = [feature for feature in candidates if feature in forced]
    if forced_requested:
        raise NeedsConfirmationError("Feature list includes forced exclude columns: " + ", ".join(forced_requested[:20]))
    final = [feature for feature in candidates if feature not in excludes]
    if not final:
        issues.append(
            _issue(
                code="NO_CANDIDATE_FEATURES",
                level="critical",
                blocking=True,
                message="No candidate features remain after exclusions.",
                suggested_fix="Provide at least one valid modeling feature.",
            )
        )
    return final, issues


def _compute_basic_quality(frames: dict[str, Any], candidate_features: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, frame in frames.items():
        for feature in candidate_features:
            if feature not in frame.columns:
                rows.append(
                    {
                        "feature": feature,
                        "split": split,
                        "row_count": len(frame),
                        "missing_count": None,
                        "missing_rate": None,
                        "non_null_rate": None,
                        "nunique_non_missing": None,
                        "mode_ratio": None,
                        "severity": "severe",
                        "default_recommendation": "recommended_exclude",
                        "status": "failed_missing_column",
                    }
                )
                continue
            series = frame[feature]
            row_count = int(len(series))
            missing_count = int(series.isna().sum())
            non_missing = series.dropna()
            missing_rate = float(missing_count / row_count) if row_count else 1.0
            nunique = int(non_missing.nunique(dropna=True))
            mode_ratio = None if non_missing.empty else float(non_missing.value_counts(dropna=True).iloc[0] / len(non_missing))
            severity, recommendation = _classify_quality(missing_rate, nunique, mode_ratio)
            rows.append(
                {
                    "feature": feature,
                    "split": split,
                    "row_count": row_count,
                    "missing_count": missing_count,
                    "missing_rate": missing_rate,
                    "non_null_rate": 1.0 - missing_rate,
                    "nunique_non_missing": nunique,
                    "mode_ratio": mode_ratio,
                    "severity": severity,
                    "default_recommendation": recommendation,
                    "status": "success",
                }
            )
    return rows


def _classify_quality(missing_rate: float, nunique: int, mode_ratio: float | None) -> tuple[str, str]:
    if missing_rate > DEFAULT_THRESHOLDS["missing_rate_exclude"]:
        return "severe", "recommended_exclude"
    if nunique <= 1:
        return "severe", "recommended_exclude"
    if mode_ratio is not None and mode_ratio >= DEFAULT_THRESHOLDS["near_constant_mode_ratio"]:
        return "severe", "recommended_exclude"
    if missing_rate > DEFAULT_THRESHOLDS["missing_rate_warning"]:
        return "warning", "recommended_keep"
    if mode_ratio is not None and mode_ratio >= DEFAULT_THRESHOLDS["high_concentration_mode_ratio"]:
        return "warning", "recommended_keep"
    return "pass", "recommended_keep"


def _compute_split_psi(frames: dict[str, Any], candidate_features: list[str], *, n_bins: int) -> tuple[list[dict[str, Any]], str]:
    comparisons = [split for split in ("test", "valid", "oot") if split in frames]
    if not comparisons:
        return [], "skipped_no_comparison_split"
    train = frames["train"]
    rows: list[dict[str, Any]] = []
    for feature in candidate_features:
        if feature not in train.columns:
            continue
        train_series = train[feature]
        binning = _fit_binning(train_series, n_bins=n_bins)
        baseline_distribution = _distribution(_apply_binning(train_series, binning))
        for split in comparisons:
            frame = frames[split]
            if feature not in frame.columns:
                rows.append(
                    {
                        "feature": feature,
                        "comparison_split": split,
                        "baseline_split": "train",
                        "psi": None,
                        "severity": "severe",
                        "default_recommendation": "recommended_exclude",
                        "status": "failed_missing_column",
                    }
                )
                continue
            comparison_distribution = _distribution(_apply_binning(frame[feature], binning))
            psi_value = _psi_value(baseline_distribution, comparison_distribution)
            severity = _classify_psi(psi_value)
            rows.append(
                {
                    "feature": feature,
                    "comparison_split": split,
                    "baseline_split": "train",
                    "psi": psi_value,
                    "severity": severity,
                    "default_recommendation": "recommended_exclude" if severity == "severe" else "recommended_keep",
                    "status": "success",
                }
            )
    return rows, "success"


def _compute_monthly_psi(
    frame: Any,
    *,
    candidate_features: list[str],
    date_column: str,
    dataset_scope: str,
    baseline_month: str | None,
    comparison_month_start: str | None,
    comparison_month_end: str | None,
    n_bins: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pd = _pd_module()
    if date_column not in frame.columns:
        raise MissingInputError(f"date_column does not exist in dataset_scope={dataset_scope}: {date_column}")
    working = frame.copy()
    working["__feature_quality_month__"] = _monthly_dimension_labels(frame[date_column], date_column)
    months = sorted(working["__feature_quality_month__"].unique())
    if not months:
        raise MissingInputError("No valid months are available for monthly PSI.")
    baseline = _normalize_month(baseline_month) if baseline_month else months[0]
    if baseline not in months:
        raise MissingInputError(f"baseline_month is not present in dataset_scope={dataset_scope}: {baseline}")
    comparison_months = _comparison_months(months, baseline, comparison_month_start, comparison_month_end)
    baseline_frame = working[working["__feature_quality_month__"] == baseline]
    rows: list[dict[str, Any]] = []
    for feature in candidate_features:
        if feature not in working.columns:
            rows.append({"feature": feature, "dataset_scope": dataset_scope, "baseline_month": baseline, "comparison_month": None, "psi": None, "severity": "failed", "status": "failed_missing_column"})
            continue
        binning = _fit_binning(baseline_frame[feature], n_bins=n_bins)
        baseline_dist = _distribution(_apply_binning(baseline_frame[feature], binning))
        for month in comparison_months:
            comparison = working[working["__feature_quality_month__"] == month]
            psi_value = _psi_value(baseline_dist, _distribution(_apply_binning(comparison[feature], binning)))
            rows.append({"feature": feature, "dataset_scope": dataset_scope, "baseline_month": baseline, "comparison_month": month, "psi": psi_value, "severity": _classify_psi(psi_value), "status": "success"})
    return rows, _summarize_monthly_psi(rows, baseline, comparison_months)


def _compute_uplift_bivar(
    frame: Any,
    *,
    candidate_features: list[str],
    treatment_column: str,
    outcome_column: str,
    outcome_type: str,
    max_numeric_bins: int,
    max_categorical_levels: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    missing = [column for column in [treatment_column, outcome_column] if column not in frame.columns]
    if missing:
        raise MissingInputError("Missing required modeling columns: " + ", ".join(missing))
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for feature in candidate_features:
        if feature not in frame.columns:
            issues.append(_issue(code="UPLIFT_BIVAR_FEATURE_MISSING", level="warning", blocking=False, message=f"Feature is missing and was skipped: {feature}", suggested_fix="Remove the feature or fix the split schema."))
            continue
        labels = _apply_binning(
            frame[feature],
            _fit_binning(frame[feature], n_bins=max_numeric_bins, categorical_max_levels=max_categorical_levels),
        )
        binned = frame.assign(__feature_quality_bin__=labels)
        for bin_id, bin_label in enumerate(sorted(binned["__feature_quality_bin__"].unique(), key=_bin_sort_key), start=1):
            subset = binned[binned["__feature_quality_bin__"] == bin_label]
            rows.append(_bivar_row(feature, bin_id, str(bin_label), subset, treatment_column, outcome_column, outcome_type))
    summary = _summarize_uplift_bivar(rows)
    return rows, summary, issues


def _fit_binning(series: Any, *, n_bins: int = DEFAULT_PSI_N_BINS, categorical_max_levels: int = 50) -> dict[str, Any]:
    pd = _pd_module()
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = int(series.notna().sum())
    numeric_like = pd.api.types.is_numeric_dtype(series) or numeric.notna().sum() >= max(1, int(non_missing * 0.8))
    if numeric_like:
        positive = numeric[numeric > 0].dropna()
        bins: list[float] = []
        if not positive.empty:
            try:
                _, raw_bins = pd.qcut(positive, q=max(2, int(n_bins)), retbins=True, duplicates="drop")
                raw_bins[0] = -math.inf
                raw_bins[-1] = math.inf
                bins = [float(item) for item in raw_bins]
            except ValueError:
                bins = []
        return {"kind": "numeric", "bins": bins, "n_bins": int(n_bins)}
    top_values = series.dropna().astype(str).value_counts().head(categorical_max_levels).index.astype(str).tolist()
    return {"kind": "categorical", "top_values": top_values}


def _apply_binning(series: Any, binning: dict[str, Any]) -> Any:
    pd = _pd_module()
    if binning["kind"] == "categorical":
        top_values = set(binning["top_values"])
        return series.map(lambda value: "Missing" if pd.isna(value) else str(value) if str(value) in top_values else "__OTHER__")
    numeric = pd.to_numeric(series, errors="coerce")
    labels: list[str] = []
    bins = binning.get("bins", [])
    for value in numeric:
        if pd.isna(value):
            labels.append("Missing")
        elif value < 0:
            labels.append("Negative")
        elif value == 0:
            labels.append("Zero")
        elif len(bins) < 2:
            labels.append("Positive")
        else:
            labels.append(_positive_bin(float(value), bins))
    return pd.Series(labels, index=series.index)


def _positive_bin(value: float, bins: list[float]) -> str:
    for idx in range(1, len(bins)):
        if value <= bins[idx]:
            return f"Positive_Q{idx}"
    return f"Positive_Q{len(bins) - 1}"


def _distribution(labels: Any) -> dict[str, float]:
    counts = labels.value_counts(dropna=False)
    total = float(counts.sum())
    if total == 0:
        return {}
    return {str(label): float(count / total) for label, count in counts.items()}


def _psi_value(expected: dict[str, float], actual: dict[str, float]) -> float:
    value = 0.0
    for label in sorted(set(expected) | set(actual)):
        e = expected.get(label, 0.0) + EPSILON
        a = actual.get(label, 0.0) + EPSILON
        value += (a - e) * math.log(a / e)
    return float(value)


def _classify_psi(value: float) -> str:
    if value > DEFAULT_THRESHOLDS["psi_exclude"]:
        return "severe"
    if value > DEFAULT_THRESHOLDS["psi_warning"]:
        return "warning"
    return "stable"


def _comparison_months(months: list[str], baseline_month: str, start_value: str | None, end_value: str | None) -> list[str]:
    start = _normalize_month(start_value) if start_value else None
    end = _normalize_month(end_value) if end_value else None
    output = []
    for month in months:
        if month == baseline_month:
            continue
        if start and month < start:
            continue
        if end and month > end:
            continue
        output.append(month)
    return output


def _monthly_dimension_labels(series: Any, date_column: str) -> Any:
    pd = _pd_module()
    if series.isna().any():
        raise MissingInputError(f"date_column contains missing values and must be month-level: {date_column}")
    as_text = series.astype(str).str.strip()
    if (as_text == "").any():
        raise MissingInputError(f"date_column contains missing values and must be month-level: {date_column}")

    yyyy_mm = as_text.str.fullmatch(r"\d{4}-\d{2}")
    if bool(yyyy_mm.all()):
        try:
            parsed_months = pd.PeriodIndex(as_text, freq="M")
        except Exception as exc:  # noqa: BLE001
            raise MissingInputError(f"date_column contains unparseable month values: {date_column}") from exc
        return parsed_months.astype(str)

    yyyymm = as_text.str.fullmatch(r"\d{6}")
    if bool(yyyymm.all()):
        parsed_dates = pd.to_datetime(as_text, format="%Y%m", errors="coerce")
        if parsed_dates.isna().any():
            raise MissingInputError(f"date_column contains unparseable month values: {date_column}")
        return parsed_dates.dt.to_period("M").astype(str)

    parsed_dates = pd.to_datetime(series, errors="coerce")
    if parsed_dates.isna().any():
        raise MissingInputError(
            f"date_column must be month-level, such as YYYY-MM, YYYYMM, or month-start dates: {date_column}"
        )
    if not bool((parsed_dates.dt.day == 1).all()):
        raise MissingInputError(
            f"date_column must be month-level; daily dates or multiple dates inside one month are not accepted: {date_column}"
        )
    return parsed_dates.dt.to_period("M").astype(str)


def _normalize_month(value: str | None) -> str:
    if not value:
        raise MissingInputError("Month value is empty.")
    pd = _pd_module()
    try:
        return str(pd.Period(str(value), freq="M"))
    except Exception as exc:  # noqa: BLE001
        raise MissingInputError(f"Invalid month value: {value}") from exc


def _summarize_monthly_psi(rows: list[dict[str, Any]], baseline_month: str, comparison_months: list[str]) -> dict[str, Any]:
    distribution: dict[str, int] = {}
    max_psi: float | None = None
    failed = 0
    for row in rows:
        if row.get("status") != "success":
            failed += 1
            continue
        severity = str(row["severity"])
        distribution[severity] = distribution.get(severity, 0) + 1
        if row["psi"] is not None:
            max_psi = float(row["psi"]) if max_psi is None else max(max_psi, float(row["psi"]))
    return {"baseline_month": baseline_month, "comparison_month_count": len(comparison_months), "severity_distribution": distribution, "max_psi": max_psi, "failed_feature_count": failed}


def _bivar_row(feature: str, bin_id: int, bin_label: str, subset: Any, treatment_column: str, outcome_column: str, outcome_type: str) -> dict[str, Any]:
    del outcome_type
    treatment = subset[subset[treatment_column] == 1]
    control = subset[subset[treatment_column] == 0]
    treatment_value = _outcome_value(treatment[outcome_column])
    control_value = _outcome_value(control[outcome_column])
    observed_difference = None if treatment_value is None or control_value is None else treatment_value - control_value
    return {
        "feature": feature,
        "bin_id": bin_id,
        "bin_label": bin_label,
        "row_count": int(len(subset)),
        "treatment_count": int(len(treatment)),
        "control_count": int(len(control)),
        "treatment_outcome_value": treatment_value,
        "control_outcome_value": control_value,
        "observed_difference": observed_difference,
        "support_status": _support_status(treatment, control, treatment_value, control_value),
    }


def _outcome_value(series: Any) -> float | None:
    pd = _pd_module()
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.mean())


def _support_status(treatment: Any, control: Any, treatment_value: float | None, control_value: float | None) -> str:
    if treatment.empty and control.empty:
        return "insufficient_both_groups"
    if treatment.empty:
        return "insufficient_treatment"
    if control.empty:
        return "insufficient_control"
    if treatment_value is None or control_value is None:
        return "insufficient_outcome"
    return "ok"


def _bin_sort_key(label: Any) -> tuple[int, str]:
    text = str(label)
    if text == "Missing":
        return (0, text)
    if text == "Negative":
        return (1, text)
    if text == "Zero":
        return (2, text)
    if text == "Positive":
        return (3, text)
    if text.startswith("Positive_Q"):
        try:
            return (3, f"{int(text.rsplit('Q', 1)[1]):04d}")
        except ValueError:
            return (4, text)
    if text == "__OTHER__":
        return (98, text)
    return (10, text)


def _build_recommendation(
    run_dir: Path,
    *,
    candidate_features: list[str],
    basic_quality_rows: list[dict[str, Any]],
    psi_rows: list[dict[str, Any]],
    source_result_path: str,
    diagnostic_paths: dict[str, str | None],
    thresholds: dict[str, float],
) -> tuple[str, dict[str, Any]]:
    thresholds = {**DEFAULT_THRESHOLDS, **thresholds}
    exclusion_reasons: dict[str, list[str]] = defaultdict(list)
    warning_features: set[str] = set()
    for row in basic_quality_rows:
        if row["split"] != "train":
            continue
        feature = row["feature"]
        if row["status"] == "failed_missing_column":
            exclusion_reasons[feature].append("missing_from_train")
        elif row["missing_rate"] is not None and row["missing_rate"] > thresholds["missing_rate_exclude"]:
            exclusion_reasons[feature].append("missing_rate_exclude")
        elif row["nunique_non_missing"] is not None and row["nunique_non_missing"] <= 1:
            exclusion_reasons[feature].append("constant_or_all_missing")
        elif row["mode_ratio"] is not None and row["mode_ratio"] >= thresholds["near_constant_mode_ratio"]:
            exclusion_reasons[feature].append("near_constant_mode_ratio")
        elif (row["missing_rate"] is not None and row["missing_rate"] > thresholds["missing_rate_warning"]) or (row["mode_ratio"] is not None and row["mode_ratio"] >= thresholds["high_concentration_mode_ratio"]):
            warning_features.add(feature)
    for row in psi_rows:
        feature = row["feature"]
        if row["status"] == "failed_missing_column":
            exclusion_reasons[feature].append(f"missing_from_{row['comparison_split']}")
        elif row["psi"] is not None and row["psi"] > thresholds["psi_exclude"]:
            exclusion_reasons[feature].append(f"split_psi_exclude_{row['comparison_split']}")
        elif row["psi"] is not None and row["psi"] > thresholds["psi_warning"]:
            warning_features.add(feature)
    excluded = sorted(exclusion_reasons)
    selected = [feature for feature in candidate_features if feature not in set(excluded)]
    selected_path = run_dir / "artifacts" / "selected_features.v1.json"
    excluded_path = run_dir / "artifacts" / "excluded_features.v1.json"
    reasons_path = run_dir / "artifacts" / "exclusion_reasons.v1.json"
    write_json(selected_path, {"features": selected})
    write_json(excluded_path, {"features": excluded})
    write_json(reasons_path, {"exclusion_reasons": dict(exclusion_reasons)})
    reason_distribution = Counter(reason for reasons in exclusion_reasons.values() for reason in reasons)
    recommendation = {
        "kind": "feature_selection_recommendation",
        "artifact_kind": "feature_selection_recommendation",
        "artifact_version": 1,
        "source_result_path": source_result_path,
        "diagnostic_paths": diagnostic_paths,
        "status": "has_recommended_features" if selected else "no_recommended_features",
        "thresholds": thresholds,
        "summary": {
            "candidate_count": len(candidate_features),
            "recommended_selected_count": len(selected),
            "recommended_excluded_count": len(excluded),
            "warning_feature_count": len(warning_features),
        },
        "selected_features": {
            "count": len(selected),
            "preview": selected[:20],
            "features_path": str(selected_path.resolve()),
        },
        "excluded_features": {
            "count": len(excluded),
            "preview": excluded[:20],
            "features_path": str(excluded_path.resolve()),
        },
        "exclusion_reasons": {
            "count": len(exclusion_reasons),
            "preview": {feature: exclusion_reasons[feature] for feature in excluded[:10]},
            "reasons_path": str(reasons_path.resolve()),
        },
        "reason_distribution": dict(reason_distribution),
        "created_by": SKILL_CREATED_BY,
        "created_at": _now(),
    }
    recommendation_path = run_dir / "artifacts" / "feature_selection_recommendation.v1.json"
    write_json(recommendation_path, recommendation)
    return str(recommendation_path.resolve()), recommendation


def _diagnostic_outputs(
    *,
    candidates: list[str],
    statuses: dict[str, str],
    basic_rows: list[dict[str, Any]],
    psi_rows: list[dict[str, Any]],
    psi_status: str,
    artifact_paths: dict[str, str | None],
    recommendation_path: str | None,
    recommendation: dict[str, Any] | None,
    monthly_summary: dict[str, Any],
    bivar_summary: dict[str, Any],
    psi_n_bins: int | None,
) -> dict[str, Any]:
    basic_summary = _summarize_basic_quality(basic_rows) if basic_rows else {"severity_distribution": {}}
    psi_summary = _summarize_split_psi(psi_rows, psi_status) if psi_rows else {"severity_distribution": {}}
    return {
        "mode": "diagnostics",
        "candidate_count": len(candidates),
        "diagnostics_status": statuses,
        **artifact_paths,
        "diagnostics_summary": {
            "missing_rate_distribution": basic_summary.get("severity_distribution", {}),
            "split_psi_distribution": psi_summary.get("severity_distribution", {}),
            "monthly_psi_distribution": monthly_summary.get("severity_distribution", {}),
            "uplift_bivar_summary": bivar_summary,
            "psi_n_bins": psi_n_bins,
        },
        "selection_basis": {
            "participating": [name for name in ("basic_quality", "split_psi") if statuses.get(name) not in {"not_requested", "not_run"}],
            "reference_only": [name for name in ("monthly_psi", "uplift_bivar") if statuses.get(name) == "success"],
        },
        "selection_summary": _selection_summary(recommendation, recommendation_path),
        "feature_selection_recommendation_path": recommendation_path,
        "monthly_psi_summary": monthly_summary,
        "uplift_bivar_summary": bivar_summary,
    }


def _selection_summary(recommendation: dict[str, Any] | None, recommendation_path: str | None) -> dict[str, Any]:
    if not recommendation:
        return {
            "status": "not_requested",
            "recommended_selected_count": 0,
            "recommended_excluded_count": 0,
            "recommendation_path": None,
        }
    return {
        "status": recommendation["status"],
        "recommended_selected_count": recommendation["summary"]["recommended_selected_count"],
        "recommended_excluded_count": recommendation["summary"]["recommended_excluded_count"],
        "recommendation_path": recommendation_path,
    }


def _summarize_basic_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_rows = [row for row in rows if row["split"] == "train"]
    distribution: dict[str, int] = {}
    for row in train_rows:
        distribution[row["severity"]] = distribution.get(row["severity"], 0) + 1
    return {
        "train_feature_count": len(train_rows),
        "severity_distribution": distribution,
        "train_recommended_exclude_count": len([row for row in train_rows if row["default_recommendation"] == "recommended_exclude"]),
    }


def _summarize_split_psi(rows: list[dict[str, Any]], status: str) -> dict[str, Any]:
    if status != "success":
        return {"status": status, "severity_distribution": {}, "max_psi": None}
    distribution: dict[str, int] = {}
    max_psi: float | None = None
    for row in rows:
        severity = str(row["severity"])
        distribution[severity] = distribution.get(severity, 0) + 1
        if row["psi"] is not None:
            max_psi = float(row["psi"]) if max_psi is None else max(max_psi, float(row["psi"]))
    return {"status": status, "severity_distribution": distribution, "max_psi": max_psi}


def _summarize_uplift_bivar(rows: list[dict[str, Any]]) -> dict[str, Any]:
    distribution: dict[str, int] = {}
    for row in rows:
        support = str(row["support_status"])
        distribution[support] = distribution.get(support, 0) + 1
    return {"feature_count": len({row["feature"] for row in rows}), "bin_count": len(rows), "support_status_distribution": distribution}


def _write_feature_plan(
    run_dir: Path,
    *,
    status: str,
    input_paths: dict[str, Any],
    selected_features: list[str],
    source_type: str,
    diagnostics_status: str,
    recommended_selected_count: int | None,
    manual_include_count: int = 0,
    manual_exclude_count: int = 0,
    confirmed_by: str | None = None,
    source_draft_path: str | None = None,
    selection_summary: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    if not selected_features:
        raise MissingInputError("feature_plan cannot have an empty selected feature list.")
    version = 1
    selected_path = run_dir / "artifacts" / f"feature_plan_features.{status}.v{version}.json"
    write_json(selected_path, {"features": selected_features})
    summary = {
        "source": source_type,
        "recommended_selected_count": recommended_selected_count,
        "manual_include_count": manual_include_count,
        "manual_exclude_count": manual_exclude_count,
    }
    summary.update(selection_summary or {})
    artifact = _artifact_envelope(
        artifact_kind="feature_plan",
        artifact_status=status,
        artifact_version=version,
        source_type=source_type,
        input_paths=input_paths,
        payload={
            "selected_features": {
                "count": len(selected_features),
                "preview": selected_features[:20],
                "features": selected_features,
                "features_path": str(selected_path.resolve()),
            },
            "diagnostics_status": diagnostics_status,
            "selection_summary": summary,
        },
        validation={
            "filled_fields": ["selected_features", "diagnostics_status", "selection_summary"],
            "missing_fields": [],
            "assumptions": [],
            "warnings": [],
        },
        confirmed_by=confirmed_by if status == "confirmed" else None,
        confirmed_at=_now() if status == "confirmed" else None,
        source_draft_path=source_draft_path,
    )
    plan_path = run_dir / "artifacts" / f"feature_plan.{status}.v{version}.json"
    write_json(plan_path, artifact)
    return plan_path, artifact


def _recommendation_context(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    recommendation_path = _required_input_path(output_dir, payload.get("recommendation_path"), "recommendation_path")
    recommendation = _read_json_object(recommendation_path)
    if recommendation.get("kind") != "feature_selection_recommendation" and recommendation.get("artifact_kind") != "feature_selection_recommendation":
        raise MissingInputError("recommendation_path is not a feature selection recommendation.")
    source_value = recommendation.get("source_result_path") or payload.get("source_result_path")
    if not source_value:
        raise MissingInputError("Recommendation is missing source_result_path; rerun feature analysis.")
    source_path = _required_input_path(output_dir, source_value, "source_result_path")
    source = _read_successful_feature_result(source_path)
    return recommendation_path, recommendation, source_path, source


def _validated_recommendation_inputs(output_dir: Path, payload: dict[str, Any], source: dict[str, Any]) -> dict[str, str]:
    source_inputs = source.get("input_paths") or {}
    task_path = _required_input_path(output_dir, payload.get("task_config_path") or source_inputs.get("task_config_path"), "task_config_path")
    spec_path = _required_input_path(output_dir, payload.get("modeling_sample_spec_path") or source_inputs.get("modeling_sample_spec_path"), "modeling_sample_spec_path")
    _require_same_paths(output_dir, source_inputs, {"task_config_path": str(task_path), "modeling_sample_spec_path": str(spec_path)}, ("task_config_path", "modeling_sample_spec_path"))
    return {"task_config_path": str(task_path), "modeling_sample_spec_path": str(spec_path)}


def _read_successful_feature_result(path: Path) -> dict[str, Any]:
    result = _read_json_object(path)
    if result.get("skill_name") != SKILL_NAME or result.get("status") != "success":
        raise MissingInputError("Referenced feature-quality result is not successful.")
    return result


def _load_source_candidates(outputs: dict[str, Any], output_dir: Path) -> list[str]:
    if outputs.get("candidate_features_path"):
        return _load_feature_list_path(outputs["candidate_features_path"], output_dir)
    recommendation_path = outputs.get("feature_selection_recommendation_path")
    if recommendation_path:
        recommendation = _read_json_object(resolve_run_path(output_dir, recommendation_path))
        selected = _load_feature_list_path(recommendation["selected_features"]["features_path"], output_dir)
        excluded = _load_feature_list_path(recommendation["excluded_features"]["features_path"], output_dir)
        return selected + [item for item in excluded if item not in selected]
    raise MissingInputError("Source result has no candidate feature artifact; rerun full feature analysis.")


def _read_diagnostic_csv(path_value: Any, kind: str, output_dir: Path | None = None) -> list[dict[str, Any]]:
    if not path_value:
        return []
    path = resolve_run_path(output_dir, path_value) if output_dir else Path(str(path_value)).expanduser().resolve()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    numeric = {"basic": {"missing_rate", "nunique_non_missing", "mode_ratio"}, "psi": {"psi"}}[kind]
    integer = {"nunique_non_missing"}
    for row in rows:
        for key in numeric:
            value = row.get(key)
            if value in {None, ""}:
                row[key] = None
            else:
                row[key] = int(float(value)) if key in integer else float(value)
    return rows


def _read_feature_plan(path: Path, *, require_draft: bool = False, require_confirmed: bool = False) -> dict[str, Any]:
    plan = _read_json_object(path)
    if plan.get("artifact_kind") != "feature_plan":
        raise MissingInputError("feature_plan_path must point to a feature_plan artifact.")
    if require_draft and plan.get("artifact_status") != "draft":
        raise MissingInputError("source_draft_path must point to a draft feature_plan.")
    if require_confirmed and plan.get("artifact_status") != "confirmed":
        raise MissingInputError("feature_plan_path must point to a confirmed feature_plan.")
    return plan


def _selected_features_from_plan(plan: dict[str, Any], output_dir: Path | None = None) -> list[str]:
    selected = plan.get("payload", {}).get("selected_features", {})
    if selected.get("features"):
        return [str(item) for item in selected.get("features") if str(item)]
    if selected.get("features_path"):
        return _load_feature_list_path(selected["features_path"], output_dir)
    raise MissingInputError("feature_plan is missing selected features.")


def _load_feature_list_path(path_value: Any, output_dir: Path | None = None) -> list[str]:
    path = resolve_run_path(output_dir, path_value) if output_dir else Path(str(path_value)).expanduser().resolve()
    return [str(item) for item in _read_json_object(path).get("features", []) if str(item)]


def _validate_feature_plan_lineage(plan: dict[str, Any], task_path: Path, spec_path: Path, output_dir: Path) -> None:
    inputs = plan.get("input_paths") or {}
    stale = []
    if not _same_path(inputs.get("task_config_path"), task_path, output_dir):
        stale.append("task_config_path")
    if not _same_path(inputs.get("modeling_sample_spec_path"), spec_path, output_dir):
        stale.append("modeling_sample_spec_path")
    if stale:
        raise NeedsConfirmationError(
            "feature_plan lineage is stale for selected inputs (" + ", ".join(stale) + "); rerun feature analysis or use reuse_feature_set explicitly."
        )


def _require_same_paths(output_dir: Path, source: dict[str, Any], incoming: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if not _same_path(_resolve_input_path(output_dir, str(source.get(key))), _resolve_input_path(output_dir, str(incoming.get(key)))):
            raise NeedsConfirmationError(f"{key} changed since diagnostics; rerun full feature analysis.")


def _validate_features(features: Any, train_columns: list[str], forced_exclude: list[str], *, allow_empty: bool = False) -> list[str]:
    values = _dedupe(features)
    if not values and not allow_empty:
        raise MissingInputError("Selected feature list is empty.")
    missing = [item for item in values if item not in train_columns]
    if missing:
        raise MissingInputError("Features are not present in train: " + ", ".join(missing[:20]))
    forced = [item for item in values if item in set(forced_exclude)]
    if forced:
        raise MissingInputError("Features are forced-excluded: " + ", ".join(forced[:20]))
    return values


def _diagnostics_completeness(result: dict[str, Any]) -> str:
    statuses = result.get("outputs", {}).get("diagnostics_status", {})
    requested = [value for key, value in statuses.items() if key != "iv" and value != "not_requested"]
    return "complete" if requested and all(value == "success" for value in requested) else "partial"


def _confirm_step(draft_path: Path, flow_dir: Path) -> dict[str, Any]:
    return {
        "skill": SKILL_NAME,
        "action": "confirm_artifact",
        "reason": "The modified feature plan must be confirmed before modeling.",
        "inputs": {
            "flow_dir": str(flow_dir.resolve()),
            "source_draft_path": str(draft_path.resolve()),
            "artifact_kind": "feature_plan",
        },
        "requires_user_confirmation": True,
    }


def _recommendation_steps(recommendation_path: str | None, source_result_path: str, flow_dir: Path) -> list[dict[str, Any]]:
    if not recommendation_path:
        return []
    flow_dir_value = str(flow_dir.resolve())
    return [
        {
            "skill": SKILL_NAME,
            "action": action,
            "reason": reason,
            "inputs": {
                **({"flow_dir": flow_dir_value} if action in {"accept_recommendation", "modify_recommendation"} else {}),
                "recommendation_path": recommendation_path,
                "source_result_path": source_result_path,
            },
            "requires_user_confirmation": action != "modify_recommendation",
        }
        for action, reason in (
            ("accept_recommendation", "Adopt the recommendation directly."),
            ("modify_recommendation", "Adjust the recommended feature set."),
            ("selection_only", "Apply different thresholds to existing diagnostics."),
        )
    ]


def _report_recommendation(
    selection: dict[str, Any], output_dir: Path | None = None
) -> dict[str, Any] | None:
    path = selection.get("recommendation_path")
    if not path:
        return None
    try:
        resolved = resolve_run_path(output_dir, path) if output_dir else Path(str(path)).expanduser().resolve()
        return _read_json_object(resolved)
    except Exception:  # noqa: BLE001
        return None


def _report_feature_plan_summary(
    data: dict[str, Any], recommendation: dict[str, Any] | None, output_dir: Path | None = None
) -> dict[str, Any]:
    feature_plan_path = data.get("feature_plan_path")
    if feature_plan_path:
        try:
            resolved = (
                resolve_run_path(output_dir, feature_plan_path)
                if output_dir
                else Path(str(feature_plan_path)).expanduser().resolve()
            )
            plan = _read_feature_plan(resolved)
            selected = plan.get("payload", {}).get("selected_features", {})
            selection = plan.get("payload", {}).get("selection_summary", {})
            return {
                "status": plan.get("artifact_status") or data.get("feature_plan_status") or "unknown",
                "source": plan.get("source_type") or selection.get("source") or "unknown",
                "count": selected.get("count") or len(selected.get("features") or []),
                "preview": selected.get("preview") or (selected.get("features") or [])[:20],
                "basis": data.get("selection_basis", {}).get("participating") or [],
                "path": feature_plan_path,
            }
        except Exception:  # noqa: BLE001
            pass
    if recommendation:
        selected = recommendation.get("selected_features") or {}
        return {
            "status": "recommended_not_confirmed",
            "source": "feature_selection_recommendation",
            "count": selected.get("count"),
            "preview": selected.get("preview") or [],
            "basis": data.get("selection_basis", {}).get("participating") or [],
            "path": None,
        }
    return {
        "status": data.get("feature_plan_status") or "not_requested",
        "source": "not_available",
        "count": None,
        "preview": [],
        "basis": data.get("selection_basis", {}).get("participating") or [],
        "path": feature_plan_path,
    }


def _report_basic_quality_summary(
    path_value: Any, candidate_count: int, output_dir: Path | None = None
) -> dict[str, Any]:
    rows = _read_diagnostic_csv(path_value, "basic", output_dir)
    train_rows = [row for row in rows if row.get("split") == "train"]
    buckets = [
        ("0", lambda value: value == 0),
        ("0 - 10%", lambda value: value is not None and 0 < value <= 0.1),
        ("10% - 50%", lambda value: value is not None and 0.1 < value <= 0.5),
        ("50% - 80%", lambda value: value is not None and 0.5 < value <= 0.8),
        ("> 80%", lambda value: value is not None and value > 0.8),
        ("unknown", lambda value: value is None),
    ]
    distribution = {label: 0 for label, _predicate in buckets}
    max_missing: dict[str, Any] | None = None
    for row in train_rows:
        value = row.get("missing_rate")
        for label, predicate in buckets:
            if predicate(value):
                distribution[label] += 1
                break
        if value is not None and (not max_missing or value > max_missing.get("missing_rate", -1)):
            max_missing = row
    constant_count = len(
        [
            row
            for row in train_rows
            if (row.get("nunique_non_missing") is not None and row.get("nunique_non_missing") <= 1)
            or row.get("missing_rate") == 1
        ]
    )
    high_concentration_count = len(
        [
            row
            for row in train_rows
            if row.get("mode_ratio") is not None and row.get("mode_ratio") >= DEFAULT_THRESHOLDS["high_concentration_mode_ratio"]
        ]
    )
    total = len(train_rows) or candidate_count
    return {
        "total": total,
        "distribution": distribution,
        "constant_count": constant_count,
        "high_concentration_count": high_concentration_count,
        "max_missing_feature": max_missing.get("feature") if max_missing and max_missing.get("missing_rate", 0) > 0 else None,
        "max_missing_rate": max_missing.get("missing_rate") if max_missing else None,
    }


def _report_psi_summary(
    path_value: Any, candidate_count: int, output_dir: Path | None = None
) -> dict[str, Any]:
    rows = _read_diagnostic_csv(path_value, "psi", output_dir)
    worst_by_feature: dict[str, dict[str, Any]] = {}
    max_row: dict[str, Any] | None = None
    for row in rows:
        psi = row.get("psi")
        feature = str(row.get("feature") or "")
        if not feature or psi is None:
            continue
        current = worst_by_feature.get(feature)
        if current is None or psi > current.get("psi", -1):
            worst_by_feature[feature] = row
        if max_row is None or psi > max_row.get("psi", -1):
            max_row = row
    distribution = {"<= 0.10": 0, "> 0.10 and <= 0.25": 0, "> 0.25": 0, "unknown": 0}
    for row in worst_by_feature.values():
        psi = row.get("psi")
        if psi is None:
            distribution["unknown"] += 1
        elif psi <= DEFAULT_THRESHOLDS["psi_warning"]:
            distribution["<= 0.10"] += 1
        elif psi <= DEFAULT_THRESHOLDS["psi_exclude"]:
            distribution["> 0.10 and <= 0.25"] += 1
        else:
            distribution["> 0.25"] += 1
    total = len(worst_by_feature) or candidate_count
    return {
        "total": total,
        "distribution": distribution,
        "max_psi": max_row.get("psi") if max_row else None,
        "max_psi_feature": max_row.get("feature") if max_row else None,
        "max_psi_split": max_row.get("comparison_split") if max_row else None,
        "warning_or_worse_count": distribution["> 0.10 and <= 0.25"] + distribution["> 0.25"],
    }


def _format_report_count(value: Any) -> str:
    return "N/A" if value is None else str(value)


def _format_report_metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_report_percent(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return str(value)


def _append_distribution_table(lines: list[str], distribution: dict[str, int], total: int, *, first_column: str, zh: bool) -> None:
    lines.extend(
        [
            f"| {first_column} | 数量 | 占比 |" if zh else f"| {first_column} | Count | Percentage |",
            "| --- | ---: | ---: |",
        ]
    )
    denominator = total if total > 0 else sum(distribution.values())
    for label, count in distribution.items():
        percentage = "N/A" if denominator <= 0 else f"{count / denominator * 100:.2f}%"
        lines.append(f"| {label} | {count} | {percentage} |")


def _find_diagnostics_result(output_dir: Path, source_result_path: Any) -> dict[str, Any]:
    current = source_result_path
    visited: set[Path] = set()
    while current:
        path = resolve_run_path(output_dir, current)
        if path in visited or not path.exists():
            return {}
        visited.add(path)
        result = _read_successful_feature_result(path)
        outputs = result.get("outputs") or {}
        if outputs.get("diagnostics_status") or outputs.get("feature_basic_quality_path") or outputs.get("feature_psi_split_path"):
            return result
        current = (result.get("input_paths") or {}).get("source_result_path")
    return {}


def _write_feature_plan_report(
    run_dir: Path, output_dir: Path, feature_plan_path: Path
) -> Path:
    plan = _read_feature_plan(feature_plan_path)
    plan_inputs = plan.get("input_paths") or {}
    diagnostics_result = _find_diagnostics_result(output_dir, plan_inputs.get("source_result_path"))
    diagnostics_outputs = diagnostics_result.get("outputs") or {}

    recommendation_path = plan_inputs.get("recommendation_path")
    recommendation = None
    if recommendation_path:
        resolved_recommendation = resolve_run_path(output_dir, recommendation_path)
        if resolved_recommendation.exists():
            recommendation = _read_json_object(resolved_recommendation)

    diagnostic_paths = (recommendation or {}).get("diagnostic_paths") or {}
    selected = _selected_features_from_plan(plan, output_dir)
    task_path_value = plan_inputs.get("task_config_path")
    task = {}
    if task_path_value:
        task_path = resolve_run_path(output_dir, task_path_value)
        if task_path.exists():
            task = _read_json_object(task_path)

    statuses = diagnostics_outputs.get("diagnostics_status") or dict(DIAGNOSTICS_NOT_RUN)
    data = {
        **diagnostics_outputs,
        "candidate_count": diagnostics_outputs.get("candidate_count") or len(selected),
        "diagnostics_status": statuses,
        "feature_basic_quality_path": diagnostics_outputs.get("feature_basic_quality_path")
        or diagnostic_paths.get("feature_basic_quality_path"),
        "feature_psi_split_path": diagnostics_outputs.get("feature_psi_split_path")
        or diagnostic_paths.get("feature_psi_split_path"),
        "feature_psi_monthly_path": diagnostics_outputs.get("feature_psi_monthly_path")
        or diagnostic_paths.get("feature_psi_monthly_path"),
        "feature_uplift_bivar_path": diagnostics_outputs.get("feature_uplift_bivar_path")
        or diagnostic_paths.get("feature_uplift_bivar_path"),
        "feature_selection_recommendation_path": recommendation_path,
        "selection_summary": _selection_summary(recommendation, recommendation_path),
        "feature_plan_path": str(feature_plan_path.resolve()),
        "feature_plan_status": plan.get("artifact_status"),
        "input_paths": {
            key: plan_inputs.get(key)
            for key in ("task_config_path", "modeling_sample_spec_path")
            if plan_inputs.get(key)
        },
        "language": _language_from_task_config(task),
    }
    return _write_report(run_dir, output_dir=output_dir, data=data)


def _write_report(run_dir: Path, *, output_dir: Path, data: dict[str, Any]) -> Path:
    language = str(data.get("language") or "zh-CN")
    zh = is_zh(language)
    path = run_dir / "report.md"
    selection = data.get("selection_summary") or {}
    diagnostics_summary = data.get("diagnostics_summary") or {}
    diagnostics_status = data.get("diagnostics_status") or {}
    candidate_count = int(data.get("candidate_count") or 0)
    recommendation = _report_recommendation(selection, output_dir)
    plan_summary = _report_feature_plan_summary(data, recommendation, output_dir)
    basic_summary = _report_basic_quality_summary(
        data.get("feature_basic_quality_path"), candidate_count, output_dir
    )
    psi_summary = _report_psi_summary(data.get("feature_psi_split_path"), candidate_count, output_dir)
    selection_basis = data.get("selection_basis") or {}
    participating = selection_basis.get("participating") or []
    reference_only = selection_basis.get("reference_only") or []

    lines = [
        "# 特征质量分析" if zh else "# Feature Quality Analysis",
        "",
        "## 摘要" if zh else "## Summary",
        "",
        f"- 候选特征数：{candidate_count}" if zh else f"- Candidate features: {candidate_count}",
        f"- Basic Quality：`{diagnostics_status.get('basic_quality')}`" if zh else f"- Basic quality: `{diagnostics_status.get('basic_quality')}`",
        f"- Split PSI：`{diagnostics_status.get('split_psi')}`" if zh else f"- Split PSI: `{diagnostics_status.get('split_psi')}`",
        f"- Monthly PSI：`{diagnostics_status.get('monthly_psi', 'not_run')}`" if zh else f"- Monthly PSI: `{diagnostics_status.get('monthly_psi', 'not_run')}`",
        f"- Uplift Bivar：`{diagnostics_status.get('uplift_bivar', 'not_run')}`" if zh else f"- Uplift Bivar: `{diagnostics_status.get('uplift_bivar', 'not_run')}`",
        f"- IV：`{diagnostics_status.get('iv')}`" if zh else f"- IV: `{diagnostics_status.get('iv')}`",
        f"- 参与筛选的诊断：`{participating}`" if zh else f"- Diagnostics used for selection: `{participating}`",
        f"- 仅供参考的诊断：`{reference_only}`" if zh else f"- Reference-only diagnostics: `{reference_only}`",
        f"- 报告语言：`{language}`" if zh else f"- Report language: `{language}`",
        "",
        "## 当前特征方案" if zh else "## Current Feature Plan",
        "",
        f"- 特征方案状态：`{plan_summary.get('status')}`" if zh else f"- Feature plan status: `{plan_summary.get('status')}`",
        f"- 方案来源：`{plan_summary.get('source')}`" if zh else f"- Source: `{plan_summary.get('source')}`",
        f"- 入模特征数：`{_format_report_count(plan_summary.get('count'))}`" if zh else f"- Modeling feature count: `{_format_report_count(plan_summary.get('count'))}`",
        f"- 特征预览：`{', '.join(str(item) for item in plan_summary.get('preview') or []) or 'N/A'}`" if zh else f"- Feature preview: `{', '.join(str(item) for item in plan_summary.get('preview') or []) or 'N/A'}`",
        f"- 筛选依据：`{', '.join(str(item) for item in plan_summary.get('basis') or []) or 'N/A'}`" if zh else f"- Selection basis: `{', '.join(str(item) for item in plan_summary.get('basis') or []) or 'N/A'}`",
        "",
        "## 缺失率分布" if zh else "## Missing Rate Distribution",
        "",
    ]
    if data.get("feature_basic_quality_path"):
        _append_distribution_table(lines, basic_summary["distribution"], int(basic_summary["total"] or 0), first_column=("缺失率分桶" if zh else "Missing Rate Bucket"), zh=zh)
        lines.extend(
            [
                "",
                f"- 常量 / 全缺失特征数：`{basic_summary['constant_count']}`" if zh else f"- Constant / all-missing features: `{basic_summary['constant_count']}`",
                f"- 高集中度特征数：`{basic_summary['high_concentration_count']}`" if zh else f"- High-concentration features: `{basic_summary['high_concentration_count']}`",
                f"- 最大缺失率：`{_format_report_percent(basic_summary['max_missing_rate'])}`" if zh else f"- Max missing rate: `{_format_report_percent(basic_summary['max_missing_rate'])}`",
                f"- 最大缺失率特征：`{basic_summary['max_missing_feature'] or 'N/A'}`" if zh else f"- Max missing feature: `{basic_summary['max_missing_feature'] or 'N/A'}`",
            ]
        )
    else:
        lines.append("- Basic Quality 未提供。" if zh else "- Basic quality was not provided.")

    lines.extend(["", "## PSI 分布" if zh else "## PSI Distribution", ""])
    if data.get("feature_psi_split_path"):
        _append_distribution_table(lines, psi_summary["distribution"], int(psi_summary["total"] or 0), first_column="PSI Bucket", zh=zh)
        lines.extend(
            [
                "",
                f"- 最大 PSI：`{_format_report_metric(psi_summary['max_psi'])}`" if zh else f"- Max PSI: `{_format_report_metric(psi_summary['max_psi'])}`",
                f"- 最大 PSI 特征：`{psi_summary['max_psi_feature'] or 'N/A'}`" if zh else f"- Max PSI feature: `{psi_summary['max_psi_feature'] or 'N/A'}`",
                f"- 最大 PSI split：`{psi_summary['max_psi_split'] or 'N/A'}`" if zh else f"- Max PSI split: `{psi_summary['max_psi_split'] or 'N/A'}`",
                f"- PSI 预警及以上特征数：`{psi_summary['warning_or_worse_count']}`" if zh else f"- PSI warning-or-worse features: `{psi_summary['warning_or_worse_count']}`",
            ]
        )
    else:
        lines.append("- Split PSI 未提供。" if zh else "- Split PSI was not provided.")

    lines.extend(["", "## Monthly PSI" if zh else "## Monthly PSI", ""])
    if data.get("feature_psi_monthly_path"):
        lines.append(f"- Monthly PSI 明细：{_generated_note(data.get('feature_psi_monthly_path'), language)}" if zh else f"- Monthly PSI table: {_generated_note(data.get('feature_psi_monthly_path'), language)}")
        lines.append(f"- Monthly PSI 摘要：`{diagnostics_summary.get('monthly_psi_distribution') or data.get('monthly_psi_summary') or {}}`" if zh else f"- Monthly PSI summary: `{diagnostics_summary.get('monthly_psi_distribution') or data.get('monthly_psi_summary') or {}}`")
    else:
        lines.append("- Monthly PSI：未提供。" if zh else "- Monthly PSI: not provided.")

    lines.extend(["", "## Uplift Bivar" if zh else "## Uplift Bivar", ""])
    if data.get("feature_uplift_bivar_path"):
        lines.append(f"- Uplift Bivar 明细：{_generated_note(data.get('feature_uplift_bivar_path'), language)}" if zh else f"- Uplift Bivar table: {_generated_note(data.get('feature_uplift_bivar_path'), language)}")
        lines.append(f"- Uplift Bivar 摘要：`{diagnostics_summary.get('uplift_bivar_summary') or data.get('uplift_bivar_summary') or {}}`" if zh else f"- Uplift Bivar summary: `{diagnostics_summary.get('uplift_bivar_summary') or data.get('uplift_bivar_summary') or {}}`")
    else:
        lines.append("- Uplift Bivar：未提供。" if zh else "- Uplift Bivar: not provided.")

    lines.extend(["", "## 筛选建议" if zh else "## Selection Recommendation", ""])
    if selection.get("status") == "not_requested":
        lines.append("- 本次未请求筛选建议。" if zh else "- Not requested in this action.")
    else:
        reason_distribution = recommendation.get("reason_distribution") if recommendation else {}
        excluded_preview = (recommendation.get("excluded_features", {}) if recommendation else {}).get("preview") or []
        lines.extend(
            [
                f"- 状态：`{selection.get('status')}`",
                f"- 建议保留特征数：{selection.get('recommended_selected_count')}",
                f"- 建议排除特征数：{selection.get('recommended_excluded_count')}",
                f"- 剔除原因分布：`{reason_distribution or {}}`",
                f"- 剔除特征预览：`{', '.join(str(item) for item in excluded_preview) or 'N/A'}`",
                f"- 建议引用：{_generated_note(selection.get('recommendation_path'), language)}",
            ]
            if zh
            else [
                f"- Status: `{selection.get('status')}`",
                f"- Recommended selected count: {selection.get('recommended_selected_count')}",
                f"- Recommended excluded count: {selection.get('recommended_excluded_count')}",
                f"- Exclusion reason distribution: `{reason_distribution or {}}`",
                f"- Excluded feature preview: `{', '.join(str(item) for item in excluded_preview) or 'N/A'}`",
                f"- Recommendation artifact: {_generated_note(selection.get('recommendation_path'), language)}",
            ]
        )
    lines.extend(
        [
            "",
            "## 完整计算结果" if zh else "## Complete Calculation Results",
            "",
            "- 特征质量报告：`report.md`" if zh else "- Feature quality report: `report.md`",
            f"- Basic Quality 明细：{_generated_note(data.get('feature_basic_quality_path'), language)}" if zh else f"- Basic quality details: {_generated_note(data.get('feature_basic_quality_path'), language)}",
            f"- Split PSI 明细：{_generated_note(data.get('feature_psi_split_path'), language)}" if zh else f"- Split PSI details: {_generated_note(data.get('feature_psi_split_path'), language)}",
            f"- Monthly PSI 明细：{_generated_note(data.get('feature_psi_monthly_path'), language)}" if zh else f"- Monthly PSI details: {_generated_note(data.get('feature_psi_monthly_path'), language)}",
            f"- Uplift Bivar 明细：{_generated_note(data.get('feature_uplift_bivar_path'), language)}" if zh else f"- Uplift Bivar details: {_generated_note(data.get('feature_uplift_bivar_path'), language)}",
            f"- 特征方案引用：{_generated_note(data.get('feature_plan_path'), language)}" if zh else f"- Feature plan artifact: {_generated_note(data.get('feature_plan_path'), language)}",
            "",
            "## 限制" if zh else "## Limitations",
            "",
            "- 本报告只汇总已执行的诊断，不补算未请求的指标。" if zh else "- This report only summarizes requested diagnostics and does not recompute missing metrics.",
            "- Monthly PSI 和 Uplift Bivar 仅作参考，不自动改变筛选建议。" if zh else "- Monthly PSI and Uplift Bivar are reference-only and do not automatically change selection.",
            "- IV 和相关性分析在本轮未实现。" if zh else "- IV and correlation analysis are not implemented in this gate.",
            "- 筛选建议在确认 feature_plan 前不是建模输入。" if zh else "- Recommendations are not modeling inputs until a feature_plan is confirmed.",
            "",
            "## 附录：系统引用" if zh else "## Appendix: System References",
            "",
        ]
    )
    for key, value in (data.get("input_paths") or {}).items():
        if value:
            lines.append(f"- {key}: `{value}`")
    for key in (
        "result_path",
        "candidate_features_path",
        "feature_basic_quality_path",
        "feature_psi_split_path",
        "feature_psi_monthly_path",
        "feature_uplift_bivar_path",
        "feature_selection_recommendation_path",
    ):
        value = data.get(key)
        if value:
            lines.append(f"- {key}: `{value}`")
    recommendation_path = selection.get("recommendation_path")
    if recommendation_path and recommendation_path != data.get("feature_selection_recommendation_path"):
        lines.append(f"- recommendation_path: `{recommendation_path}`")
    lines.append("")
    write_text(path, "\n".join(lines))
    return path


def _generated_note(value: Any, language: str) -> str:
    if not value:
        return "未生成" if is_zh(language) else "not generated"
    return "已生成，见附录。" if is_zh(language) else "generated; see appendix."


def _flow_folder_for_action(output_dir: Path, phase: str, body: dict[str, Any]) -> Path:
    if body.get("flow_dir"):
        return ensure_existing_skill_call_dir(output_dir, str(body["flow_dir"]))
    initial_actions = {
        "diagnostics_only",
        "selection_only",
        "manual_feature_plan",
        "reuse_feature_set",
    }
    flow_required_actions = {
        "accept_recommendation",
        "modify_recommendation",
        "confirm_artifact",
        "validate_feature_plan",
    }
    if phase in flow_required_actions:
        raise ProjectLayoutError("FLOW_DIR_REQUIRED", "flow_dir is required for this feature-quality action.")
    return ensure_skill_call_dir(output_dir, SKILL_NAME, "feature_quality")


def _refresh_feature_recommendation_source(
    result: dict[str, Any],
    output_dir: Path,
    result_path: Path,
) -> None:
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    recommendation_value = outputs.get("feature_selection_recommendation_path")
    if not recommendation_value:
        return
    recommendation_path = _resolve_input_path(output_dir, str(recommendation_value))
    if not recommendation_path.exists():
        return
    source_result_path = to_run_relative_path(output_dir, result_path)
    recommendation = read_json(recommendation_path)
    if isinstance(recommendation, dict):
        recommendation["source_result_path"] = source_result_path
        write_json(recommendation_path, recommendation)
    for step in result.get("next_steps") or []:
        inputs = step.get("inputs")
        if isinstance(inputs, dict) and "source_result_path" in inputs:
            inputs["source_result_path"] = source_result_path


def _read_artifact(path: Path, *, artifact_kind: str, require_confirmed: bool = False, require_draft: bool = False) -> dict[str, Any]:
    artifact = _read_json_object(path)
    if artifact.get("artifact_kind") != artifact_kind:
        raise MissingInputError(f"{path} is not a {artifact_kind} artifact.")
    if require_confirmed and artifact.get("artifact_status") != "confirmed":
        raise MissingInputError(f"{artifact_kind} must be confirmed.")
    if require_draft and artifact.get("artifact_status") != "draft":
        raise MissingInputError(f"{artifact_kind} must be draft.")
    return artifact


def _artifact_envelope(
    *,
    artifact_kind: str,
    artifact_status: str,
    artifact_version: int,
    source_type: str,
    input_paths: dict[str, Any],
    payload: dict[str, Any],
    validation: dict[str, Any],
    confirmed_by: str | None,
    confirmed_at: str | None,
    source_draft_path: str | None,
) -> dict[str, Any]:
    return {
        "artifact_kind": artifact_kind,
        "artifact_status": artifact_status,
        "artifact_version": int(artifact_version),
        "source_type": source_type,
        "created_by": SKILL_CREATED_BY,
        "created_at": _now(),
        "input_paths": input_paths,
        "payload": payload,
        "validation": validation,
        "confirmed_by": confirmed_by,
        "confirmed_at": confirmed_at,
        "source_draft_path": source_draft_path,
    }


def _read_json_object(path: Path | str) -> dict[str, Any]:
    return read_json(Path(path))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_feature_list_file(path: Path) -> list[str]:
    if not path.exists():
        raise MissingInputError(f"feature_list_path does not exist: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            payload = payload.get("features") or payload.get("feature_list")
        return _normalize_list(payload)
    if path.suffix.lower() == ".csv":
        frame = load_table({"kind": "local_csv", "path": str(path)})
        if frame.empty or len(frame.columns) == 0:
            return []
        return _normalize_list(frame.iloc[:, 0].dropna().tolist())
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _required_input_path(output_dir: Path, value: Any, field_name: str) -> Path:
    if not value:
        raise MissingInputError(f"{field_name} is required.", missing_fields=[field_name])
    path = _resolve_input_path(output_dir, str(value))
    if not path.exists():
        raise MissingInputError(f"{field_name} does not exist: {path}", missing_fields=[field_name])
    return path


def _optional_input_path(output_dir: Path, value: Any) -> Path | None:
    if not value:
        return None
    return _required_input_path(output_dir, value, "path")


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


def _normalize_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(item) for item in values if str(item)]


def _dedupe(values: Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in values:
        value = str(item)
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MissingInputError(f"{field_name} must be a positive integer.") from exc
    if parsed < 2:
        raise MissingInputError(f"{field_name} must be at least 2.")
    return parsed


def _language_from_task_config(task: dict[str, Any]) -> str:
    return str(task.get("payload", {}).get("report_preferences", {}).get("language") or "zh-CN")


def _feature_interaction(result: dict[str, Any], selected: list[str], previous: list[str] | None = None) -> dict[str, Any]:
    interaction: dict[str, Any] = {
        "type": "confirmation" if result["status"] == "needs_confirmation" else "completion",
        "subject": "feature_plan",
        "facts": {
            "issue_count": len(result.get("issues") or []),
            "selected_feature_count": len(selected),
            "feature_examples": selected[:5],
        },
    }
    changes = _feature_changes(previous, selected)
    if changes:
        interaction["changes"] = changes
    if result["status"] == "needs_confirmation":
        interaction["decision"] = {"required": True, "kind": "adopt_or_revise"}
    return interaction


def _feature_changes(previous: list[str] | None, adopted: list[str]) -> dict[str, Any] | None:
    if previous is None:
        return None
    added = [item for item in adopted if item not in previous]
    removed = [item for item in previous if item not in adopted]
    if not added and not removed:
        return None
    return {
        "added_count": len(added),
        "added_examples": added[:5],
        "removed_count": len(removed),
        "removed_examples": removed[:5],
        "total_count": len(adopted),
    }


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
            "feature_basic_quality_path": None,
            "feature_psi_split_path": None,
            "feature_selection_recommendation_path": None,
            "feature_plan_path": None,
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
    result.setdefault("user_interaction", {"type": "completion" if status == "success" else "recovery", "subject": "feature_plan", "facts": {"issue_count": len(result["issues"])}})
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
        error={"code": "FEATURE_QUALITY_ANALYSIS_FAILED", "message": f"{phase} failed unexpectedly.", "recoverable": False, "retryable": False, "raw_error": str(exc)},
        progress=[{"step": phase, "status": "failed", "message": str(exc)}],
    )


def _attach_transport_paths(
    result: dict[str, Any],
    output_dir: Path,
    run_dir: Path,
    result_path: Path,
) -> None:
    outputs = result.setdefault("outputs", {})
    outputs["flow_dir"] = to_run_relative_path(output_dir, run_dir)
    outputs["result_path"] = to_run_relative_path(output_dir, result_path)


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


def _pd_module() -> Any:
    try:
        import pandas as pd  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise PackageImportError("pandas", "Missing package: pandas") from exc
    return pd


def _issue(*, code: str, level: str, blocking: bool, message: str, suggested_fix: str) -> dict[str, Any]:
    return {
        "code": code,
        "level": level,
        "blocking": blocking,
        "message": message,
        "suggested_fix": suggested_fix,
    }


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
