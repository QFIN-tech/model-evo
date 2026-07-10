from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import csv
import html
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _common.io import read_cli_json_input, read_json, write_json, write_text
from _common.project_layout import (
    ProjectLayoutError,
    append_scope_log,
    assert_existing_run_dir,
    ensure_existing_skill_call_dir,
    ensure_skill_call_dir,
    experiment_manifest_path,
    next_action_paths,
    relativize_paths,
    resolve_run_path,
    to_run_relative_path,
    validate_experiment_id,
    write_flow_action_records,
    write_scope_manifest,
)
from _common.report_language import is_zh

SKILL_NAME = "uplift-model-result-comparison"
PRIMARY_METRIC = "auuc_raw"
METRIC_DIRECTION = "higher_is_better"
AUUC_VERSION = {"package": "scikit-uplift", "version": "0.5.1"}
DEFAULT_LLM_USAGE = {
    "provider": "unknown",
    "model": "unknown",
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "usage_source": "not_used",
}
ALLOWED_REF_FIELDS = {"data_ref"}
COMPARISON_DATASET_ROLES = {"test", "oot", "train", "valid", "validation"}
ELIGIBLE_TUNING_REASONS = {"improved", "no_improvement", "no_valid_tuning_trial"}


class MissingInputError(Exception):
    def __init__(self, message: str, *, missing_fields: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_fields = missing_fields or []


@dataclass
class Candidate:
    candidate_id: str
    producer_kind: str
    source_result_path: str
    learner: str
    model_spec: dict[str, Any]
    model_path: str
    model_metadata_path: str
    split_evaluation: dict[str, Any]
    lineage: dict[str, str]
    risk_flags: list[str] = field(default_factory=list)
    raw_curve_path: str = ""
    normalized_curve_path: str = ""
    uplift_bins_path: str = ""
    base_candidate_result_path: str = ""
    winner_reason: str = ""
    duplicate_model_path: bool = False
    duplicate_group_id: str = ""
    input_order: int = 0

    def metric_value(self) -> float:
        return float(self.split_evaluation["metric_value"])

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "candidate_id": self.candidate_id,
            "producer_kind": self.producer_kind,
            "producer_result_path": self.source_result_path,
            "learner": self.learner,
            "model_spec": self.model_spec,
            "model_path": self.model_path,
            "model_metadata_path": self.model_metadata_path,
            "split_evaluation": self.split_evaluation,
            "lineage": self.lineage,
            "risk_flags": self.risk_flags,
            "raw_curve_path": self.raw_curve_path,
            "normalized_curve_path": self.normalized_curve_path,
            "uplift_bins_path": self.uplift_bins_path,
            "winner_reason": self.winner_reason,
            "duplicate_model_path": self.duplicate_model_path,
            "duplicate_group_id": self.duplicate_group_id,
        }
        if self.base_candidate_result_path:
            payload["base_candidate_result_path"] = self.base_candidate_result_path
        return payload


@dataclass
class ExcludedCandidate:
    source_result_path: str
    code: str
    message: str
    skill_name: str | None = None
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_result_path": self.source_result_path,
            "skill_name": self.skill_name,
            "status": self.status,
            "code": self.code,
            "message": self.message,
        }


def run_compare(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    payload, plan_metadata = _payload_from_comparison_plan(output_dir, payload)
    task_path = _required_input_path(output_dir, payload.get("task_config_path"), "task_config_path")
    task_config = _read_artifact(task_path, artifact_kind="task_config", require_confirmed=True)
    candidate_paths = payload.get("candidate_result_paths")
    if not isinstance(candidate_paths, list) or len(candidate_paths) < 2:
        raise MissingInputError(
            "candidate_result_paths must contain two or more result JSON paths.",
            missing_fields=["candidate_result_paths"],
        )
    role = _split_key(str(payload.get("comparison_dataset_role") or "test").lower())
    if role not in COMPARISON_DATASET_ROLES:
        raise MissingInputError("comparison_dataset_role must be test, oot, train, valid, or validation.")
    role = _split_key(role)
    normalized: list[Candidate] = []
    excluded: list[ExcludedCandidate] = []
    for index, value in enumerate(candidate_paths):
        try:
            candidate_path = _required_input_path(output_dir, value, "candidate_result_paths")
            candidate, exclusion = _normalize_candidate(
                candidate_path,
                role,
                task_path,
                output_dir,
                input_order=index,
            )
            if candidate:
                normalized.append(candidate)
            if exclusion:
                excluded.append(exclusion)
        except Exception as exc:  # noqa: BLE001
            excluded.append(
                ExcludedCandidate(
                    source_result_path=str(value),
                    code="CANDIDATE_RESULT_PATH_INVALID",
                    message=str(exc),
                )
            )
    _mark_duplicate_models(normalized)
    comparability, blocking, warnings = _validate_comparability(normalized, str(task_path), role)
    recommended = (
        _recommend_candidate(normalized, output_dir)
        if role != "train" and comparability["is_fairly_comparable"] and len(normalized) >= 2
        else None
    )
    rows = _build_rows(normalized, excluded, recommended)
    table_path = run_dir / "artifacts" / "model_comparison.v1.csv"
    summary_path = run_dir / "artifacts" / "model_comparison_summary.v1.json"
    curve_svg_path = run_dir / "artifacts" / f"{role}_auuc_curve_comparison.v1.svg"
    report_path = run_dir / "report.md"
    _write_table(table_path, rows)
    curve_candidates = _curve_display_candidates(normalized, recommended, output_dir)
    curve_payload = [
        (_candidate_label(candidate), _read_curve(candidate.raw_curve_path), candidate.metric_value())
        for candidate in curve_candidates
    ]
    curve_svg_output = str(curve_svg_path.resolve()) if write_curve_svg(curve_svg_path, curve_payload, role_label=_role_label(role)) else None
    curve_warnings = _curve_artifact_issues(curve_candidates, curve_payload, curve_svg_output)
    risk_summary = sorted({risk for candidate in normalized for risk in candidate.risk_flags if risk})
    comparison_summary = {
        "artifact_kind": "model_comparison_summary",
        "artifact_version": 1,
        "comparison_plan_path": plan_metadata.get("comparison_plan_path"),
        "plan_confirmed": bool(plan_metadata.get("plan_confirmed")),
        "comparison_dataset_role": role,
        "primary_metric": PRIMARY_METRIC,
        "metric_direction": METRIC_DIRECTION,
        "auuc_version": dict(AUUC_VERSION),
        "candidate_count": len(normalized),
        "rows": rows,
        "comparability": comparability,
        "risk_summary": risk_summary,
        "recommended_primary_model_path": recommended.model_path if recommended else None,
        "recommended_candidate_result_path": recommended.source_result_path if recommended else None,
        "curve_svg_path": curve_svg_output,
        "recommendation_rationale": _recommendation_rationale(recommended, comparability),
        "created_at": _now(),
    }
    write_json(summary_path, relativize_paths(comparison_summary, output_dir))
    _write_report(
        report_path,
        task_config=task_config,
        candidates=normalized,
        recommended=recommended,
        rows=rows,
        comparability=comparability,
        risk_summary=risk_summary,
        comparison_dataset_role=role,
        curve_svg_path=curve_svg_output,
        run_dir=output_dir,
    )
    baseline_warnings = _missing_baseline_issues(normalized, output_dir)
    issues = [
        *[_issue(item.code, item.message, blocking=False) for item in excluded],
        *blocking,
        *warnings,
        *baseline_warnings,
        *curve_warnings,
        *[_issue(code, f"Candidate carries risk flag {code}.", blocking=False) for code in risk_summary],
    ]
    status = "success"
    if excluded or warnings or baseline_warnings or curve_warnings or not recommended:
        status = "partial_success"
    outputs: dict[str, Any] = {
        "flow_dir": str(run_dir.resolve()),
        "comparison_plan_path": plan_metadata.get("comparison_plan_path"),
        "plan_confirmed": bool(plan_metadata.get("plan_confirmed")),
        "comparison_table_path": str(table_path.resolve()),
        "model_comparison_summary_path": str(summary_path.resolve()),
        "report_path": str(report_path.resolve()),
        "comparison_dataset_role": role,
        "candidate_count": len(normalized),
        "normalized_candidates": [candidate.to_dict() for candidate in normalized],
        "comparability": comparability,
        "risk_summary": risk_summary,
        "recommendation_rationale": comparison_summary["recommendation_rationale"],
        "curve_svg_path": curve_svg_output,
    }
    if recommended:
        outputs["recommended_primary_model_path"] = recommended.model_path
        outputs["recommended_candidate_result_path"] = recommended.source_result_path
    else:
        outputs["recommended_primary_model_path"] = None
        outputs["recommended_candidate_result_path"] = None
    return _result(
        run_dir=run_dir,
        phase="compare",
        status=status,
        summary="Candidate comparison completed.",
        input_paths={"task_config_path": str(task_path), "candidate_result_paths": [str(_resolve_input_path(output_dir, str(item))) for item in candidate_paths]},
        outputs=outputs,
        issues=issues,
        artifacts=[
            {"kind": "comparison_table", "path": str(table_path.resolve())},
            {"kind": "model_comparison_summary", "path": str(summary_path.resolve())},
            *([{"kind": "auuc_curve_comparison", "path": curve_svg_output}] if curve_svg_output else []),
            {"kind": "report", "path": str(report_path.resolve())},
        ],
        progress=[
            {"step": "input_validation", "status": "success"},
            {"step": "candidate_normalization", "status": "success"},
            {"step": "comparability_validation", "status": "success" if comparability["is_fairly_comparable"] else "partial_success"},
            {"step": "artifact_generation", "status": "success"},
        ],
        next_steps=_next_steps(
            [str(_resolve_input_path(output_dir, str(item))) for item in candidate_paths],
            normalized,
            recommended,
            output_dir,
        ),
    )


def run_draft_comparison_plan(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    role = _split_key(str(payload.get("comparison_dataset_role") or "test").lower())
    if role not in COMPARISON_DATASET_ROLES:
        raise MissingInputError("comparison_dataset_role must be test, oot, train, valid, or validation.")
    candidate_paths, candidate_sources = _candidate_result_paths_for_plan(output_dir, payload)
    if len(candidate_paths) < 2:
        raise MissingInputError(
            "At least two candidates are required to draft a comparison plan.",
            missing_fields=["experiment_ids", "candidate_result_paths"],
        )
    task_path = _task_config_path_for_plan(output_dir, payload, candidate_paths)
    normalized: list[Candidate] = []
    excluded: list[ExcludedCandidate] = []
    for index, candidate_path in enumerate(candidate_paths):
        try:
            candidate, exclusion = _normalize_candidate(
                candidate_path,
                role,
                task_path,
                output_dir,
                input_order=index,
            )
            if candidate:
                normalized.append(candidate)
            if exclusion:
                excluded.append(exclusion)
        except Exception as exc:  # noqa: BLE001
            excluded.append(
                ExcludedCandidate(
                    source_result_path=str(candidate_path),
                    code="CANDIDATE_RESULT_PATH_INVALID",
                    message=str(exc),
                )
            )
    _mark_duplicate_models(normalized)
    comparability, blocking, warnings = _validate_comparability(normalized, str(task_path), role)
    if len(normalized) < 2:
        raise MissingInputError("Fewer than two eligible candidates are available.")
    plan = _build_comparison_plan(
        candidate_sources=candidate_sources,
        candidates=normalized,
        excluded=excluded,
        task_path=task_path,
        role=role,
        comparability=comparability,
        blocking=blocking,
        warnings=warnings,
    )
    plan = relativize_paths(plan, output_dir)
    plan_path = run_dir / "artifacts" / "comparison_plan.v1.json"
    write_json(plan_path, plan)
    report_path = _write_comparison_plan_report(run_dir / "report.md", plan)
    issues = [
        *[_issue(item.code, item.message, blocking=False) for item in excluded],
        *blocking,
        *warnings,
    ]
    return _result(
        run_dir=run_dir,
        phase="draft_comparison_plan",
        status="needs_confirmation",
        summary="Comparison plan is ready for user confirmation.",
        input_paths={
            "task_config_path": str(task_path),
            "candidate_result_paths": [str(path) for path in candidate_paths],
        },
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "comparison_plan_path": str(plan_path.resolve()),
            "report_path": str(report_path.resolve()),
            "comparison_dataset_role": role,
            "candidate_count": len(normalized),
            "recommended_comparison_mode": plan["recommended_comparison_mode"],
            "requires_user_confirmation": True,
        },
        issues=issues,
        artifacts=[
            {"kind": "comparison_plan", "path": str(plan_path.resolve())},
            {"kind": "report", "path": str(report_path.resolve())},
        ],
        progress=[
            {"step": "input_validation", "status": "success"},
            {"step": "candidate_normalization", "status": "success"},
            {"step": "comparison_plan", "status": "needs_confirmation"},
        ],
        next_steps=[
            {
                "skill": SKILL_NAME,
                "action": "compare",
                "reason": "Run formal comparison after the comparison plan is confirmed.",
                "inputs": {
                    "flow_dir": str(run_dir.resolve()),
                    "comparison_plan_path": str(plan_path.resolve()),
                },
                "requires_user_confirmation": True,
            }
        ],
    )


def _normalize_candidate(
    result_path: Path,
    role: str,
    task_path: Path,
    output_dir: Path,
    *,
    input_order: int,
) -> tuple[Candidate | None, ExcludedCandidate | None]:
    result = read_json(result_path)
    skill_name = str(result.get("skill_name") or "")
    status = str(result.get("status") or "")
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    if skill_name in {"uplift-model-s-learner-modeling", "uplift-model-t-learner-modeling"}:
        if status != "success":
            return None, ExcludedCandidate(str(result_path), "UNSUPPORTED_CANDIDATE_RESULT", "Modeling candidate must be successful.", skill_name, status)
        candidate_payload = _candidate_payload(output_dir, outputs)
        split_eval = _split_evaluation_for_modeling(result, role)
        lineage = {
            key: str(_resolve_input_path(output_dir, str(value)))
            for key, value in (result.get("input_paths") or {}).items()
            if key in {"task_config_path", "modeling_sample_spec_path", "feature_plan_path"}
        }
        producer_kind = "modeling"
        learner = _learner_from_candidate(candidate_payload, outputs)
        base_path = ""
        winner_reason = ""
        risk_flags: list[str] = []
    elif skill_name == "uplift-model-tuning":
        summary = outputs.get("tuning_summary") or {}
        winner_reason = str(summary.get("winner_reason") or "")
        eligible = status == "success" or (status == "partial_success" and winner_reason in ELIGIBLE_TUNING_REASONS)
        if not eligible:
            return None, ExcludedCandidate(str(result_path), "TUNING_RESULT_HAS_NO_ELIGIBLE_WINNER", "Model tuning result has no eligible winner.", skill_name, status)
        candidate_payload = _candidate_payload(output_dir, outputs)
        split_eval = (outputs.get("split_evaluations") or {}).get(role)
        if not isinstance(split_eval, dict):
            return None, ExcludedCandidate(str(result_path), "MISSING_SPLIT_EVALUATION", f"split_evaluations.{role} is required.", skill_name, status)
        lineage = {
            key: str(_resolve_input_path(output_dir, str(value)))
            for key, value in (candidate_payload.get("lineage") or {}).items()
            if key in {"task_config_path", "modeling_sample_spec_path", "feature_plan_path", "modeling_result_path"}
        }
        producer_kind = "tuning"
        learner = _learner_from_candidate(candidate_payload, outputs)
        base_path = str(candidate_payload.get("base_candidate_result_path") or "")
        risk_flags = [str(item) for item in candidate_payload.get("risk_flags") or [] if str(item)]
    else:
        return None, ExcludedCandidate(str(result_path), "UNSUPPORTED_CANDIDATE_RESULT", f"Unsupported candidate producer: {skill_name}", skill_name, status)
    exclusion_code, problem = _validate_split_evaluation(split_eval, role)
    if problem:
        return None, ExcludedCandidate(str(result_path), exclusion_code, problem, skill_name, status)
    if not _same_path(lineage.get("task_config_path"), task_path):
        return None, ExcludedCandidate(str(result_path), "TASK_CONFIG_MISMATCH", "Candidate task_config_path does not match.", skill_name, status)
    raw_curve_path = _candidate_artifact_path(output_dir, outputs, split_eval, role, "raw_curve_path")
    normalized_curve_path = _candidate_artifact_path(output_dir, outputs, split_eval, role, "normalized_curve_path")
    uplift_bins_path = _candidate_artifact_path(output_dir, outputs, split_eval, role, "uplift_bins_path")
    return (
        Candidate(
            candidate_id=str(candidate_payload.get("candidate_id") or result.get("run_id") or result_path.parent.parent.name),
            producer_kind=producer_kind,
            source_result_path=str(result_path),
            learner=learner,
            model_spec=candidate_payload.get("model_spec") or {},
            model_path=str(candidate_payload.get("model_artifact_path") or outputs.get("model_artifact_path") or ""),
            model_metadata_path=str(candidate_payload.get("model_metadata_path") or outputs.get("model_metadata_path") or ""),
            split_evaluation=split_eval,
            lineage=lineage,
            risk_flags=sorted(set(risk_flags)),
            raw_curve_path=raw_curve_path,
            normalized_curve_path=normalized_curve_path,
            uplift_bins_path=uplift_bins_path,
            base_candidate_result_path=base_path,
            winner_reason=winner_reason,
            input_order=input_order,
        ),
        None,
    )


def _candidate_artifact_path(output_dir: Path, outputs: dict[str, Any], split_eval: dict[str, Any], role: str, key: str) -> str:
    candidates: list[Any] = [split_eval.get(key)]
    metrics = outputs.get("modeling_metrics") if isinstance(outputs.get("modeling_metrics"), dict) else {}
    split_metrics = (metrics.get("splits") or {}).get(role) if isinstance(metrics.get("splits"), dict) else None
    if isinstance(split_metrics, dict):
        candidates.append(split_metrics.get(key))
    winner_metrics = outputs.get("winner_metrics") if isinstance(outputs.get("winner_metrics"), dict) else {}
    candidates.append(winner_metrics.get(key))
    candidates.append(outputs.get(key))
    for value in candidates:
        if isinstance(value, str) and value:
            return str(_resolve_input_path(output_dir, value))
    return ""


def _candidate_payload(output_dir: Path, outputs: dict[str, Any]) -> dict[str, Any]:
    path = outputs.get("model_candidate_path")
    if path:
        return read_json(_resolve_input_path(output_dir, str(path)))
    candidate = outputs.get("model_candidate")
    if isinstance(candidate, dict):
        return candidate
    raise MissingInputError("Candidate result does not expose model_candidate or model_candidate_path.")


def _learner_from_candidate(candidate: dict[str, Any], outputs: dict[str, Any]) -> str:
    model_spec = candidate.get("model_spec") if isinstance(candidate.get("model_spec"), dict) else {}
    return str(model_spec.get("model_type") or outputs.get("learner") or "s_learner")


def _split_evaluation_for_modeling(result: dict[str, Any], role: str) -> dict[str, Any]:
    outputs = result.get("outputs") or {}
    direct = outputs.get("split_evaluations")
    if isinstance(direct, dict) and isinstance(direct.get(role), dict):
        return direct[role]
    raise MissingInputError(f"modeling result must expose outputs.split_evaluations.{role}.")


def _dataset_path_from_lineage(result: dict[str, Any], role: str) -> str:
    sample_path = (result.get("input_paths") or {}).get("modeling_sample_spec_path")
    if not sample_path:
        raise MissingInputError("modeling result input_paths.modeling_sample_spec_path is required.")
    sample = read_json(Path(sample_path))
    datasets = sample.get("payload", {}).get("datasets") or {}
    path = datasets.get(role)
    if not path:
        raise MissingInputError(f"sample spec does not expose dataset path for {role}.")
    return str(path)


def _validate_split_evaluation(value: dict[str, Any], role: str) -> tuple[str, str | None]:
    required = (
        "dataset_role",
        "population_fingerprint",
        "metric_name",
        "metric_direction",
    )
    missing = [field for field in required if value.get(field) in (None, "")]
    auuc_version = value.get("auuc_version")
    if not isinstance(auuc_version, dict):
        missing.append("auuc_version")
    else:
        if auuc_version.get("package") in (None, ""):
            missing.append("auuc_version.package")
        if auuc_version.get("version") in (None, ""):
            missing.append("auuc_version.version")
    if missing:
        return "MISSING_SPLIT_EVALUATION", f"Missing split_evaluation fields: {missing}"
    if _split_key(str(value.get("dataset_role"))) != role:
        return "EVALUATION_POPULATION_MISMATCH", "split_evaluation.dataset_role must match comparison_dataset_role."
    if value.get("metric_name") != PRIMARY_METRIC:
        return "EVALUATION_METHOD_MISMATCH", "Only auuc_raw is supported as primary metric."
    if value.get("metric_direction") != METRIC_DIRECTION:
        return "EVALUATION_METHOD_MISMATCH", "Metric direction must be higher_is_better."
    if value.get("auuc_version") != AUUC_VERSION:
        return "EVALUATION_METHOD_MISMATCH", f"auuc_version must be {AUUC_VERSION}."
    if not _finite(value.get("metric_value")):
        return "NON_RECOMMENDABLE_CANDIDATE", "metric_value must be finite."
    return "", None


def _mark_duplicate_models(candidates: list[Candidate]) -> None:
    groups: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.model_path, []).append(candidate)
    for model_path, members in groups.items():
        if len(members) < 2:
            continue
        group_id = "model-path-" + _short_hash(model_path)
        for member in members:
            member.duplicate_model_path = True
            member.duplicate_group_id = group_id
            if "DUPLICATE_MODEL_PATH" not in member.risk_flags:
                member.risk_flags.append("DUPLICATE_MODEL_PATH")


def _validate_comparability(
    candidates: list[Candidate],
    task_config_path: str,
    role: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    blocking = []
    warnings = []
    if len(candidates) < 2:
        blocking.append(_issue("INSUFFICIENT_ELIGIBLE_CANDIDATES", "Fewer than two eligible candidates."))
    for candidate in candidates:
        if not _same_path(candidate.lineage.get("task_config_path"), task_config_path):
            blocking.append(_issue("TASK_CONFIG_MISMATCH", f"Candidate {candidate.candidate_id} has a different task_config_path."))
    if candidates:
        first = candidates[0].split_evaluation
        for candidate in candidates[1:]:
            candidate_eval = candidate.split_evaluation
            if _split_key(str(candidate_eval.get("dataset_role"))) != role:
                blocking.append(_issue("EVALUATION_POPULATION_MISMATCH", "dataset_role differs."))
            if candidate_eval.get("population_fingerprint") != first.get("population_fingerprint"):
                blocking.append(_issue("EVALUATION_POPULATION_MISMATCH", "population_fingerprint differs."))
            for field in ("metric_name", "metric_direction"):
                if candidate_eval.get(field) != first.get(field):
                    blocking.append(_issue("EVALUATION_METHOD_MISMATCH", f"{field} differs across candidates."))
            first_version = first.get("auuc_version") or {}
            candidate_version = candidate_eval.get("auuc_version") or {}
            for field in ("package", "version"):
                if candidate_version.get(field) != first_version.get(field):
                    blocking.append(_issue("EVALUATION_METHOD_MISMATCH", f"auuc_version.{field} differs across candidates."))
    feature_paths = {candidate.lineage.get("feature_plan_path") for candidate in candidates if candidate.lineage.get("feature_plan_path")}
    if len(feature_paths) > 1:
        warnings.append(_issue("FEATURE_PLAN_DIFFERS", "Feature plans differ; metric can be compared but explanations differ.", blocking=False))
    sample_paths = {candidate.lineage.get("modeling_sample_spec_path") for candidate in candidates if candidate.lineage.get("modeling_sample_spec_path")}
    if len(sample_paths) > 1:
        warnings.append(_issue("SAMPLE_SPEC_DIFFERS_BUT_EVALUATION_POPULATION_MATCHES", "Sample specs differ while evaluation population fingerprint matches.", blocking=False))
    comparability = {
        "is_fairly_comparable": not blocking,
        "blocking_reasons": [item["code"] for item in blocking],
        "warnings": [item["code"] for item in warnings],
    }
    return comparability, blocking, warnings


def _recommend_candidate(candidates: list[Candidate], run_dir: Path) -> Candidate | None:
    if len(candidates) < 2:
        return None
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.metric_value(),
            0 if candidate.producer_kind == "modeling" else 1,
            candidate.input_order,
        ),
    )
    best = ranked[0]
    if best.producer_kind == "tuning" and best.winner_reason == "no_improvement":
        baseline = next((item for item in candidates if _same_path(item.source_result_path, best.base_candidate_result_path, run_dir)), None)
        if baseline:
            return baseline
    return best


def _build_rows(candidates: list[Candidate], excluded: list[ExcludedCandidate], recommended: Candidate | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ranked = sorted(candidates, key=lambda item: item.metric_value(), reverse=True)
    for rank, candidate in enumerate(ranked, start=1):
        support = candidate.split_evaluation.get("support") or {}
        rows.append(
            {
                "rank": rank,
                "candidate_id": candidate.candidate_id,
                "producer_kind": candidate.producer_kind,
                "source_result_path": candidate.source_result_path,
                "base_candidate_result_path": candidate.base_candidate_result_path or "",
                "is_recommended": bool(recommended and candidate.source_result_path == recommended.source_result_path),
                "learner": candidate.learner,
                "dataset_role": candidate.split_evaluation.get("dataset_role"),
                "primary_metric_name": candidate.split_evaluation.get("metric_name"),
                "primary_metric_value": candidate.split_evaluation.get("metric_value"),
                "auuc_normalized": candidate.split_evaluation.get("auuc_normalized"),
                "row_count": support.get("row_count"),
                "treatment_count": support.get("treatment_count"),
                "control_count": support.get("control_count"),
                "valid_bin_count": support.get("valid_bin_count"),
                "model_path": candidate.model_path,
                "model_metadata_path": candidate.model_metadata_path,
                "raw_curve_path": candidate.raw_curve_path,
                "normalized_curve_path": candidate.normalized_curve_path,
                "uplift_bins_path": candidate.uplift_bins_path,
                "risk_flags": ";".join(candidate.risk_flags),
                "exclusion_code": "",
                "exclusion_message": "",
            }
        )
    for item in excluded:
        rows.append(
            {
                "rank": "",
                "candidate_id": "",
                "producer_kind": item.skill_name or "",
                "source_result_path": item.source_result_path,
                "base_candidate_result_path": "",
                "is_recommended": False,
                "learner": "",
                "dataset_role": "",
                "primary_metric_name": "",
                "primary_metric_value": "",
                "auuc_normalized": "",
                "row_count": "",
                "treatment_count": "",
                "control_count": "",
                "valid_bin_count": "",
                "model_path": "",
                "model_metadata_path": "",
                "raw_curve_path": "",
                "normalized_curve_path": "",
                "uplift_bins_path": "",
                "risk_flags": "",
                "exclusion_code": item.code,
                "exclusion_message": item.message,
            }
        )
    return rows


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["rank", "candidate_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _curve_display_candidates(candidates: list[Candidate], recommended: Candidate | None, run_dir: Path) -> list[Candidate]:
    if not recommended:
        return []
    baseline = _find_baseline_candidate(candidates, recommended, run_dir)
    selected = [recommended]
    if baseline and baseline is not recommended:
        selected.append(baseline)
    return selected


def _read_curve(path: str) -> list[tuple[float, float]]:
    if not path:
        return []
    curve_path = Path(path)
    if not curve_path.exists():
        return []
    points: list[tuple[float, float]] = []
    try:
        with curve_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                x_value = _number(row.get("population_index"))
                y_value = _number(row.get("cumulative_gain"))
                if x_value is not None and y_value is not None:
                    points.append((x_value, y_value))
    except Exception:  # noqa: BLE001
        return []
    return points


def _sample_curve(points: list[tuple[float, float]], max_points: int = 900) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    step = (len(points) - 1) / (max_points - 1)
    sampled = [points[round(index * step)] for index in range(max_points)]
    sampled[0] = points[0]
    sampled[-1] = points[-1]
    return sampled


def write_curve_svg(path: Path, curves: list[tuple[str, list[tuple[float, float]], float]], *, role_label: str) -> bool:
    available = [(label, points, auuc) for label, points, auuc in curves if points]
    if not available:
        return False
    width, height = 820, 460
    left, right, top, bottom = 76, 28, 58, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_values = [point[1] for _, points, _ in available for point in points]
    y_min = min(min(y_values), 0.0)
    y_max = max(max(y_values), 0.0)
    if abs(y_max - y_min) <= 1e-12:
        y_max += 1.0
        y_min -= 1.0

    def sx(x_value: float, max_x: float) -> float:
        return left + (x_value / max_x if max_x else 0.0) * plot_w

    def sy(y_value: float) -> float:
        return top + (y_max - y_value) / (y_max - y_min) * plot_h

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    title = f"{role_label} AUUC Curve Comparison"
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{html.escape(title, quote=True)}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            "<style>"
            "text{font-family:Segoe UI,Arial,sans-serif;fill:#24303f}"
            ".title{font-size:22px;font-weight:700}.sub{font-size:13px;fill:#5d6b7a}"
            ".axis{stroke:#6b7280;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}"
            ".zero{stroke:#9ca3af;stroke-dasharray:4 4}.legend{font-size:13px}"
            "</style>"
        ),
        f'<text class="title" x="76" y="30">{html.escape(title)}</text>',
        '<text class="sub" x="76" y="50">Raw cumulative gain by ranked population</text>',
    ]
    for index in range(5):
        y_value = y_min + (y_max - y_min) * index / 4
        y_screen = sy(y_value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y_screen:.2f}" x2="{width - right}" y2="{y_screen:.2f}"/>')
        parts.append(f'<text x="{left - 10}" y="{y_screen + 4:.2f}" text-anchor="end" font-size="12">{y_value:.0f}</text>')
    for percent in (0, 25, 50, 75, 100):
        x_screen = left + plot_w * percent / 100
        parts.append(f'<line class="grid" x1="{x_screen:.2f}" y1="{top}" x2="{x_screen:.2f}" y2="{height - bottom}"/>')
        parts.append(f'<text x="{x_screen:.2f}" y="{height - bottom + 22}" text-anchor="middle" font-size="12">{percent}%</text>')
    zero_y = sy(0.0)
    parts.extend(
        [
            f'<line class="zero" x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" y2="{zero_y:.2f}"/>',
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
        ]
    )
    for index, (label, points, auuc) in enumerate(available):
        color = colors[index % len(colors)]
        sampled = _sample_curve(points)
        max_x = max(point[0] for point in points) or 1.0
        polyline = " ".join(f"{sx(x, max_x):.2f},{sy(y):.2f}" for x, y in sampled)
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" '
            'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        legend_y = top + 20 + index * 22
        parts.append(
            f'<text class="legend" x="{width - right - 230}" y="{legend_y}">'
            f'<tspan fill="{color}">&#9679;</tspan> {html.escape(label)} '
            f'(AUUC={auuc:.4g})</text>'
        )
    parts.extend(
        [
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 24}" text-anchor="middle" font-size="13">Population ranked by predicted uplift</text>',
            f'<text transform="translate(22 {top + plot_h / 2:.2f}) rotate(-90)" text-anchor="middle" font-size="13">Raw cumulative gain</text>',
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return True


def _curve_artifact_issues(
    candidates: list[Candidate],
    curve_payload: list[tuple[str, list[tuple[float, float]], float]],
    curve_svg_path: str | None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    issues = []
    for candidate, (_, points, _) in zip(candidates, curve_payload, strict=False):
        if points:
            continue
        issues.append(
            _issue(
                "MISSING_CURVE_ARTIFACT",
                f"Candidate {candidate.candidate_id} does not expose a readable raw AUUC curve artifact.",
                blocking=False,
            )
        )
    if not curve_svg_path and not issues:
        issues.append(
            _issue(
                "MISSING_CURVE_ARTIFACT",
                "No readable raw AUUC curve artifacts were available for comparison chart generation.",
                blocking=False,
            )
        )
    return issues


def _relative_report_path(report_path: Path, target_path: Path) -> str:
    try:
        return target_path.resolve().relative_to(report_path.parent.resolve()).as_posix()
    except ValueError:
        return target_path.resolve().as_posix()


def _missing_baseline_issues(candidates: list[Candidate], run_dir: Path) -> list[dict[str, Any]]:
    issues = []
    for candidate in candidates:
        if candidate.producer_kind != "tuning" or not candidate.base_candidate_result_path:
            continue
        if any(_same_path(item.source_result_path, candidate.base_candidate_result_path, run_dir) for item in candidates):
            continue
        issues.append(
            _issue(
                "BASELINE_CANDIDATE_NOT_INCLUDED",
                f"Tuning candidate {candidate.candidate_id} references a baseline result path that is not included in candidate_result_paths.",
                blocking=False,
            )
        )
    return issues


def _write_report(
    path: Path,
    *,
    task_config: dict[str, Any],
    candidates: list[Candidate],
    recommended: Candidate | None,
    rows: list[dict[str, Any]],
    comparability: dict[str, Any],
    risk_summary: list[str],
    comparison_dataset_role: str,
    curve_svg_path: str | None,
    run_dir: Path,
) -> None:
    language = str(task_config.get("payload", {}).get("report_preferences", {}).get("language") or "zh-CN")
    zh = is_zh(language)
    role_label = _role_label(comparison_dataset_role)
    baseline = _find_baseline_candidate(candidates, recommended, run_dir)
    lines = ["# 多模型结果比较" if zh else "# Multi-Model Result Comparison", ""]
    lines.append(f"本次比较使用的 split：**{role_label}**。" if zh else f"Comparison split: **{role_label}**.")
    lines.extend(["", "## 1. 最终推荐模型" if zh else "## 1. Final Recommended Model"])
    if recommended:
        baseline_value = baseline.metric_value() if baseline else None
        delta = recommended.metric_value() - baseline_value if baseline_value is not None else math.nan
        lines.append(f"推荐模型：**{_candidate_label(recommended)}**。" if zh else f"Recommended model: **{_candidate_label(recommended)}**.")
        lines.extend(
            [
                "",
                f"| 推荐模型 | 推荐 {role_label} AUUC | modeling baseline {role_label} AUUC | 提升 |" if zh else f"| Recommended model | Recommended {role_label} AUUC | modeling baseline {role_label} AUUC | Improvement |",
                "| --- | ---: | ---: | ---: |",
                f"| {_candidate_label(recommended)} | {_value(recommended.metric_value())} | {_value(baseline_value)} | {_value(delta)} ({_relative_improvement(delta, baseline_value)}) |",
                "",
                "推荐理由：" + " " + _recommendation_rationale(recommended, comparability) if zh else "Recommendation rationale: " + _recommendation_rationale(recommended, comparability),
            ]
        )
        if recommended.producer_kind == "tuning" and recommended.base_candidate_result_path and not baseline:
            lines.append("")
            lines.append(
                "注意：推荐调参候选的 baseline result path 未包含在本次 candidate_result_paths 中，baseline 提升无法展示。"
                if zh
                else "Note: the recommended tuned candidate's baseline result path was not included in candidate_result_paths, so baseline improvement cannot be shown."
            )
    else:
        lines.append("本次比较没有可用的主模型推荐。" if zh else "No primary model recommendation is available.")
    lines.extend(["", "## 2. 完整候选排名" if zh else "## 2. Full Candidate Ranking"])
    if rows:
        excluded_count = max(len(rows) - len(candidates), 0)
        lines.append(
            f"本次共有 **{len(candidates)}** 个可比较候选。"
            if zh
            else f"This comparison includes **{len(candidates)}** comparable candidates."
        )
        if excluded_count:
            lines.append(
                f"另有 **{excluded_count}** 个候选被排除，原因见下表。"
                if zh
                else f"Another **{excluded_count}** candidates were excluded; see the table below for details."
            )
        lines.extend(
            [
                "",
                f"| 排名 | 模型 | Candidate ID | {role_label} AUUC | AUUC Normalized | 是否推荐 | 风险/说明 |"
                if zh
                else f"| Rank | Model | Candidate ID | {role_label} AUUC | AUUC Normalized | Recommended | Risks / Notes |",
                "| ---: | --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        candidate_lookup = {candidate.candidate_id: candidate for candidate in candidates}
        for row in rows:
            candidate = candidate_lookup.get(str(row.get("candidate_id") or ""))
            model_label = _candidate_label(candidate) if candidate else ("已排除候选" if zh else "Excluded Candidate")
            notes = [str(value) for value in (row.get("risk_flags"), row.get("exclusion_code")) if value]
            if candidate and candidate.winner_reason == "no_improvement":
                notes.append("调参未提升" if zh else "no improvement from tuning")
            lines.append(
                f"| {row.get('rank') or 'N/A'} | {model_label} | `{row.get('candidate_id') or 'N/A'}` | "
                f"{_value(row.get('primary_metric_value'))} | {_value(row.get('auuc_normalized'))} | "
                f"{('是' if zh else 'Yes') if row.get('is_recommended') else ('否' if zh else 'No')} | "
                f"{'; '.join(notes) if notes else '—'} |"
            )
    else:
        lines.append("没有候选结果可展示。" if zh else "No candidate results are available.")
    lines.extend(["", "## 3. 模型产物索引" if zh else "## 3. Model Artifact Index"])
    index_candidates = []
    if recommended:
        index_candidates.append((recommended, "推荐模型" if zh else "recommended model"))
    if baseline and baseline is not recommended:
        index_candidates.append((baseline, "baseline 对照" if zh else "baseline comparison"))
    if index_candidates:
        lines.extend(["", "| 模型 | 角色 | 模型文件 | 参数/元数据 | 指标 |" if zh else "| Model | Role | Model file | Metadata | Metrics |", "| --- | --- | --- | --- | --- |"])
        for candidate, role in index_candidates:
            lines.append(
                f"| {_candidate_label(candidate)} | {role} | `{Path(candidate.model_path).name}` | `{Path(candidate.model_metadata_path).name}` | `{candidate.split_evaluation.get('metric_name')}`=`{_value(candidate.metric_value())}` |"
            )
    else:
        lines.append("N/A")
    lines.extend(
        [
            "",
            f"## 4. {role_label} AUUC Curve 对比" if zh else f"## 4. {role_label} AUUC Curve Comparison",
            "",
        ]
    )
    if curve_svg_path:
        lines.extend(["", f"![{role_label} AUUC Curve Comparison]({_relative_report_path(path, Path(curve_svg_path))})", ""])
    else:
        lines.extend(
            [
                "当前候选缺少可引用的 AUUC curve artifact；comparison 不会重算曲线。"
                if zh
                else "Current candidates do not expose readable AUUC curve artifacts; comparison does not recompute curves.",
                "",
            ]
        )
    lines.extend(["## 5. 分箱性能对比" if zh else "## 5. Uplift Chart Comparison", ""])
    chart_candidates = []
    if recommended:
        chart_candidates.append(recommended)
    if baseline and baseline is not recommended:
        chart_candidates.append(baseline)
    if not chart_candidates:
        lines.append("没有可展示的推荐模型或 baseline 分箱表。" if zh else "No recommended model or baseline uplift chart is available.")
    for candidate in chart_candidates:
        support = candidate.split_evaluation.get("support") or {}
        lines.extend(
            [
                f"### {_candidate_label(candidate)}",
                "",
                "| Rows | Treatment | Control | Valid Bins | AUUC Raw | AUUC Normalized |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| {_value(support.get('row_count'))} | {_value(support.get('treatment_count'))} | {_value(support.get('control_count'))} | {_value(support.get('valid_bin_count'))} | {_value(candidate.split_evaluation.get('metric_value'))} | {_value(candidate.split_evaluation.get('auuc_normalized'))} |",
                "",
            ]
        )
    lines.extend(["## 6. 可比性与限制" if zh else "## 6. Comparability and Limitations"])
    lines.append(f"- fairly_comparable: {comparability['is_fairly_comparable']}")
    for code in comparability["blocking_reasons"]:
        lines.append(f"- blocking: {code}")
    for code in comparability["warnings"]:
        lines.append(f"- warning: {code}")
    for code in risk_summary:
        lines.append(f"- risk: {code}")
    lines.append("- 推荐只表示本次比较上下文下的 primary metric 结果，不是最终上线或采用决定。" if zh else "- The recommendation is not a deployment or adoption decision.")
    lines.extend(["", "这是轻量级比较摘要，不是最终建模报告。" if zh else "This is a lightweight comparison summary, not the final modeling report.", ""])
    write_text(path, "\n".join(lines))


def _candidate_result_paths_for_plan(
    output_dir: Path,
    payload: dict[str, Any],
) -> tuple[list[Path], list[dict[str, Any]]]:
    experiment_ids = payload.get("experiment_ids")
    candidate_paths = payload.get("candidate_result_paths")
    has_experiments = isinstance(experiment_ids, list) and bool(experiment_ids)
    has_paths = isinstance(candidate_paths, list) and bool(candidate_paths)
    if has_experiments and has_paths:
        raise MissingInputError("Use either experiment_ids or candidate_result_paths, not both.")
    if not has_experiments and not has_paths:
        raise MissingInputError(
            "experiment_ids or candidate_result_paths is required.",
            missing_fields=["experiment_ids", "candidate_result_paths"],
        )
    if has_experiments:
        paths: list[Path] = []
        sources: list[dict[str, Any]] = []
        for raw_id in experiment_ids:
            experiment_id = validate_experiment_id(str(raw_id))
            manifest_path = experiment_manifest_path(output_dir, experiment_id)
            if not manifest_path.exists():
                raise MissingInputError(f"experiment manifest does not exist: {manifest_path}")
            manifest = read_json(manifest_path)
            manifest_outputs = manifest.get("outputs") or {}
            result_value = (
                manifest_outputs.get("modeling_result_path")
                or manifest_outputs.get("tuning_result_path")
                or manifest_outputs.get("winner_result_path")
            )
            if not result_value:
                raise MissingInputError(f"experiment {experiment_id} does not expose a candidate result path.")
            result_path = _resolve_input_path(output_dir, str(result_value))
            if not result_path.exists():
                raise MissingInputError(f"modeling_result_path does not exist: {result_path}")
            paths.append(result_path)
            sources.append(
                {
                    "source_type": "experiment",
                    "experiment_id": experiment_id,
                    "experiment_manifest_path": str(manifest_path),
                    "candidate_result_path": str(result_path),
                }
            )
        return paths, sources
    paths = [_required_input_path(output_dir, value, "candidate_result_paths") for value in candidate_paths]
    return paths, [
        {
            "source_type": "candidate_result_path",
            "candidate_result_path": str(path),
        }
        for path in paths
    ]


def _payload_from_comparison_plan(
    output_dir: Path,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_value = payload.get("comparison_plan_path")
    if not plan_value:
        direct = dict(payload)
        direct.setdefault("comparison_plan_path", None)
        return direct, {"comparison_plan_path": None, "plan_confirmed": False}
    plan_path = _required_input_path(output_dir, plan_value, "comparison_plan_path")
    plan = read_json(plan_path)
    if plan.get("artifact_kind") != "comparison_plan":
        raise MissingInputError("comparison_plan_path must point to a comparison_plan artifact.")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise MissingInputError("comparison_plan_path must contain at least two candidates.")
    candidate_paths = [
        str(candidate.get("producer_result_path"))
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("producer_result_path")
    ]
    if len(candidate_paths) < 2:
        raise MissingInputError("comparison_plan candidates must expose producer_result_path.")
    merged = dict(payload)
    merged["candidate_result_paths"] = candidate_paths
    merged["task_config_path"] = plan.get("task_config_path") or payload.get("task_config_path")
    merged["comparison_dataset_role"] = (
        plan.get("comparison_dataset_role")
        or ((plan.get("checks") or {}).get("evaluation_dataset_role") or {}).get("role")
        or payload.get("comparison_dataset_role")
        or "test"
    )
    return merged, {
        "comparison_plan_path": str(plan_path),
        "plan_confirmed": bool(payload.get("plan_confirmed", True)),
    }


def _task_config_path_for_plan(
    output_dir: Path,
    payload: dict[str, Any],
    candidate_paths: list[Path],
) -> Path:
    if payload.get("task_config_path"):
        return _required_input_path(output_dir, payload.get("task_config_path"), "task_config_path")
    first = read_json(candidate_paths[0])
    task_value = (first.get("input_paths") or {}).get("task_config_path")
    if not task_value:
        raise MissingInputError("task_config_path is required when it cannot be inferred from candidates.")
    return _required_input_path(output_dir, task_value, "task_config_path")


def _build_comparison_plan(
    *,
    candidate_sources: list[dict[str, Any]],
    candidates: list[Candidate],
    excluded: list[ExcludedCandidate],
    task_path: Path,
    role: str,
    comparability: dict[str, Any],
    blocking: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    task_paths = sorted({candidate.lineage.get("task_config_path") for candidate in candidates if candidate.lineage.get("task_config_path")})
    sample_paths = sorted({candidate.lineage.get("modeling_sample_spec_path") for candidate in candidates if candidate.lineage.get("modeling_sample_spec_path")})
    feature_paths = sorted({candidate.lineage.get("feature_plan_path") for candidate in candidates if candidate.lineage.get("feature_plan_path")})
    populations = sorted({str(candidate.split_evaluation.get("population_fingerprint")) for candidate in candidates})
    metric_names = sorted({str(candidate.split_evaluation.get("metric_name")) for candidate in candidates})
    metric_directions = sorted({str(candidate.split_evaluation.get("metric_direction")) for candidate in candidates})
    metric_methods = sorted({str(candidate.split_evaluation.get("metric_method")) for candidate in candidates})
    auuc_versions = sorted(
        {
            json.dumps(candidate.split_evaluation.get("auuc_version") or {}, sort_keys=True)
            for candidate in candidates
        }
    )
    blocking_codes = {item["code"] for item in blocking}
    warning_codes = {item["code"] for item in warnings}
    risks = [
        *[f"{item['code']}: {item['message']}" for item in blocking],
        *[f"{item['code']}: {item['message']}" for item in warnings],
        *[f"{item.code}: {item.message}" for item in excluded],
    ]
    if blocking_codes & {"TASK_CONFIG_MISMATCH", "EVALUATION_POPULATION_MISMATCH", "EVALUATION_METHOD_MISMATCH"}:
        mode = "not_recommended"
    elif warning_codes & {"FEATURE_PLAN_DIFFERS"}:
        mode = "reference_comparison"
    elif comparability.get("is_fairly_comparable"):
        mode = "strict_comparison"
    else:
        mode = "not_recommended"
    return {
        "artifact_kind": "comparison_plan",
        "artifact_version": 1,
        "task_config_path": str(task_path),
        "comparison_dataset_role": role,
        "candidate_sources": candidate_sources,
        "candidates": [candidate.to_dict() for candidate in candidates],
        "excluded_candidates": [item.to_dict() for item in excluded],
        "checks": {
            "task_config": {"is_consistent": len(task_paths) == 1 and _same_path(task_paths[0], task_path) if task_paths else False, "paths": task_paths},
            "sample_spec": {"is_consistent": len(sample_paths) <= 1, "paths": sample_paths},
            "feature_plan": {"is_consistent": len(feature_paths) <= 1, "paths": feature_paths},
            "evaluation_dataset_role": {"is_consistent": True, "role": role},
            "evaluation_population_fingerprint": {"is_consistent": len(populations) <= 1, "values": populations},
            "metric": {
                "is_consistent": len(metric_names) <= 1 and len(metric_directions) <= 1 and len(metric_methods) <= 1,
                "metric_names": metric_names,
                "metric_directions": metric_directions,
                "metric_methods": metric_methods,
            },
            "auuc_version": {"is_consistent": len(auuc_versions) <= 1, "values": auuc_versions},
        },
        "recommended_comparison_mode": mode,
        "risks": risks,
        "requires_user_confirmation": True,
        "comparability": comparability,
        "created_at": _now(),
    }


def _write_comparison_plan_report(path: Path, plan: dict[str, Any]) -> Path:
    lines = [
        "# Comparison Plan",
        "",
        f"- status: `needs_confirmation`",
        f"- recommended_comparison_mode: `{plan['recommended_comparison_mode']}`",
        f"- candidate_count: `{len(plan.get('candidates') or [])}`",
        "",
        "## Checks",
        "",
    ]
    checks = plan.get("checks") or {}
    for name, value in checks.items():
        is_consistent = value.get("is_consistent") if isinstance(value, dict) else None
        lines.append(f"- {name}: `{is_consistent}`")
    risks = plan.get("risks") or []
    lines.extend(["", "## Risks", ""])
    if risks:
        lines.extend(f"- {risk}" for risk in risks)
    else:
        lines.append("- None.")
    lines.extend(["", "User confirmation is required before running final comparison.", ""])
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
        run_dir = _flow_folder_for_action(output_dir, body)
        request_path, result_path = next_action_paths(run_dir, action or "unknown")
    except ProjectLayoutError as exc:
        print(json.dumps(_layout_error_stdout(exc), sort_keys=True))
        return 0
    write_json(request_path, request)
    if request_error:
        result = _needs_input_result(run_dir, action or "unknown", request_error, ["payload"])
    elif action not in {"compare_candidates", "compare", "draft_comparison_plan"}:
        result = _unsupported_result(run_dir, action or "unknown", "action must be draft_comparison_plan, compare_candidates, or compare.")
    else:
        legacy_fields = _legacy_ref_rejections(body)
        if legacy_fields:
            result = _legacy_ref_result(run_dir, action, legacy_fields)
        else:
            try:
                if action == "draft_comparison_plan":
                    result = run_draft_comparison_plan(run_dir, output_dir, body)
                else:
                    result = run_compare(run_dir, output_dir, body)
            except MissingInputError as exc:
                result = _needs_input_result(run_dir, action, str(exc), exc.missing_fields)
            except ProjectLayoutError as exc:
                result = _layout_error_result(run_dir, action, exc)
            except Exception as exc:  # noqa: BLE001
                result = _unexpected_error_result(run_dir, action, exc)
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


def _flow_folder_for_action(output_dir: Path, body: dict[str, Any]) -> Path:
    if body.get("flow_dir"):
        return ensure_existing_skill_call_dir(output_dir, str(body["flow_dir"]))
    return ensure_skill_call_dir(output_dir, SKILL_NAME, "model_comparison")


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


def _split_key(role: str) -> str:
    return "valid" if role == "validation" else role


def _issue(code: str, message: str, *, blocking: bool = True, level: str = "warning") -> dict[str, Any]:
    return {
        "code": code,
        "level": level,
        "blocking": blocking,
        "message": message,
        "suggested_fix": "Select compatible explicit result paths or rerun upstream skills.",
    }


def _next_steps(candidate_paths: list[str], candidates: list[Candidate], recommended: Candidate | None, run_dir: Path) -> list[dict[str, Any]]:
    missing_baseline_paths = _missing_baseline_paths(candidates, run_dir)
    if recommended and not missing_baseline_paths:
        return [
            {
                "skill": "uplift-model-reporting",
                "action": "generate_report",
                "reason": "Use the comparison result and upstream candidate facts in reporting.",
                "inputs": {
                    "subject_result_paths": candidate_paths,
                    "supporting_paths": {
                        "sample_preparation_result_path": None,
                        "sample_homogeneity_result_path": None,
                        "feature_quality_result_path": None,
                        "comparison_result_path": None,
                        "candidate_result_paths": candidate_paths,
                    },
                    "recommended_candidate_result_path": recommended.source_result_path,
                },
                "requires_user_confirmation": True,
            }
        ]
    if missing_baseline_paths:
        return [
            {
                "skill": SKILL_NAME,
                "action": "draft_comparison_plan",
                "reason": "Draft a new comparison plan including referenced baseline result paths.",
                "inputs": {"candidate_result_paths": _dedupe_paths([*candidate_paths, *missing_baseline_paths], run_dir)},
                "requires_user_confirmation": True,
            }
        ]
    return [
        {
            "skill": SKILL_NAME,
            "action": "draft_comparison_plan",
            "reason": "Select compatible explicit candidate result paths and draft a comparison plan.",
            "inputs": {"candidate_result_paths": candidate_paths},
            "requires_user_confirmation": True,
        }
    ]


def _missing_baseline_paths(candidates: list[Candidate], run_dir: Path) -> list[str]:
    paths = []
    for candidate in candidates:
        if candidate.producer_kind != "tuning" or not candidate.base_candidate_result_path:
            continue
        if any(_same_path(item.source_result_path, candidate.base_candidate_result_path, run_dir) for item in candidates):
            continue
        paths.append(candidate.base_candidate_result_path)
    return _dedupe_paths(paths, run_dir)


def _dedupe_paths(paths: list[str], run_dir: Path | None = None) -> list[str]:
    deduped: list[str] = []
    for path in paths:
        if any(_same_path(path, existing, run_dir) for existing in deduped):
            continue
        deduped.append(path)
    return deduped


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
        "type": "summary" if status in {"success", "partial_success"} else "recovery",
        "subject": "model_comparison",
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


def _unexpected_error_result(run_dir: Path, phase: str, exc: Exception) -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase or "unknown",
        status="failed",
        summary=f"{phase or 'action'} failed unexpectedly.",
        input_paths={},
        outputs={"flow_dir": str(run_dir.resolve()), "report_path": None},
        error={"code": "MODEL_COMPARISON_FAILED", "message": f"{phase or 'action'} failed unexpectedly.", "recoverable": False, "retryable": False, "raw_error": str(exc)},
        progress=[{"step": phase or "unknown", "status": "failed", "message": str(exc)}],
    )


def _layout_error_result(run_dir: Path, phase: str, exc: ProjectLayoutError) -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase or "unknown",
        status="needs_input",
        summary=str(exc),
        input_paths={},
        outputs={"flow_dir": str(run_dir.resolve()), "missing_fields": ["experiment_ids"], "report_path": None},
        issues=[{"code": exc.issue_code, "level": "critical", "blocking": True, "message": str(exc)}],
        progress=[{"step": phase or "unknown", "status": "needs_input", "message": str(exc)}],
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
    for step in result.get("next_steps") or []:
        inputs = step.get("inputs")
        if isinstance(inputs, dict):
            supporting = inputs.get("supporting_paths")
            if isinstance(supporting, dict) and supporting.get("comparison_result_path") is None:
                supporting["comparison_result_path"] = to_run_relative_path(output_dir, result_path)


def _write_flow_records(
    output_dir: Path,
    run_dir: Path,
    action: str,
    result: dict[str, Any],
    result_path: Path,
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
        "scope": "flow",
        "scope_id": run_dir.name,
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
    existing = _read_existing_manifest(run_dir / "_flow_manifest.json")
    manifest = {
        "schema_version": 1,
        "scope": "flow",
        "scope_id": run_dir.name,
        "skill_name": SKILL_NAME,
        "latest_action": action,
        "status": status,
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "inputs": result.get("input_paths") or {},
        "outputs": result.get("outputs") or {},
        "artifacts": artifacts,
        "log_path": to_run_relative_path(output_dir, run_dir / "_flow_log.jsonl"),
    }
    write_scope_manifest(run_dir, manifest)


def _read_existing_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = read_json(path)
    except Exception:  # noqa: BLE001
        return {}
    return value if isinstance(value, dict) else {}


def _layout_error_stdout(exc: ProjectLayoutError) -> dict[str, Any]:
    missing = ["experiment_ids"] if exc.issue_code == "INVALID_EXPERIMENT_ID" else ["output_dir"]
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
    return {
        "status": result["status"],
        "summary": result["summary"],
        "outputs": result.get("outputs") or {},
        "issues": result.get("issues") or [],
        "next_steps": result.get("next_steps") or [],
    }


def _read_artifact_json(path: str | None, run_dir: Path | None = None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        resolved = resolve_run_path(run_dir, path) if run_dir else Path(path)
        return read_json(resolved)
    except Exception:  # noqa: BLE001
        return {}


def _same_path(left: Any, right: Any, run_dir: Path | None = None) -> bool:
    if not left or not right:
        return False
    try:
        left_path = resolve_run_path(run_dir, left) if run_dir else Path(str(left)).expanduser().resolve()
        right_path = resolve_run_path(run_dir, right) if run_dir else Path(str(right)).expanduser().resolve()
        return left_path == right_path
    except OSError:
        return str(left) == str(right)


def _file_fingerprint(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _short_hash(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _find_baseline_candidate(candidates: list[Candidate], recommended: Candidate | None, run_dir: Path) -> Candidate | None:
    if recommended and recommended.base_candidate_result_path:
        match = next((item for item in candidates if _same_path(item.source_result_path, recommended.base_candidate_result_path, run_dir)), None)
        if match:
            return match
    return next((item for item in candidates if item.producer_kind == "modeling"), None)


def _candidate_label(candidate: Candidate) -> str:
    if candidate.learner == "t_learner":
        return "T-Learner Tuned" if candidate.producer_kind == "tuning" else "T-Learner Baseline"
    return "S-Learner Tuned" if candidate.producer_kind == "tuning" else "S-Learner Baseline"


def _recommendation_rationale(recommended: Candidate | None, comparability: dict[str, Any]) -> str:
    if not recommended:
        if comparability["blocking_reasons"]:
            return "No recommendation because the candidate set is not fairly comparable."
        return "No recommendation because fewer than two eligible candidates are available."
    if recommended.producer_kind == "modeling":
        return "Baseline candidate is recommended by tie-break, duplicate handling, or no-improvement tuning outcome."
    return "Candidate has the highest comparable auuc_raw under the fixed rule."


def _role_label(role: str) -> str:
    return {"valid": "Validation", "test": "Test", "train": "Train", "oot": "OOT"}.get(role, role)


def _relative_improvement(delta: Any, baseline: Any) -> str:
    delta_number = _number(delta)
    base_number = _number(baseline)
    if not math.isfinite(delta_number) or not math.isfinite(base_number) or base_number == 0:
        return "N/A"
    return f"{delta_number / abs(base_number):.2%}"


def _value(value: Any) -> str:
    number = _number(value)
    if math.isfinite(number):
        return f"{number:.6g}"
    return "N/A"


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


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
