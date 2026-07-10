from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import csv
import hashlib
import json
import sys
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
    next_action_paths,
    relativize_paths,
    resolve_run_path,
    to_run_relative_path,
    write_flow_action_records,
    write_scope_manifest,
)
from _common.report_language import is_zh, normalize_report_preferences
from task_interaction import build_user_interaction, task_changes, task_summary

SKILL_NAME = "uplift-model-task-spec"
SKILL_CREATED_BY = "uplift-model-task-spec-skill"
REQUIRED_FIELDS = [
    "data_ref",
    "output_dir",
    "outcome_column",
    "outcome_type",
    "treatment_column",
    "treatment_value",
    "control_value",
]
OPTIONAL_COLUMNS = ["unit_id_column", "time_column"]
ALLOWED_OUTCOME_TYPES = {"binary", "continuous"}
DEFAULT_LLM_USAGE = {
    "provider": "unknown",
    "model": "unknown",
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "usage_source": "not_used",
}
ALLOWED_REF_FIELDS = {"data_ref"}


class NeedsClarificationError(Exception):
    def __init__(
        self,
        message: str,
        *,
        missing_fields: list[str] | None = None,
        missing_columns: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.missing_fields = missing_fields or []
        self.missing_columns = missing_columns or []


class UnsupportedInputError(Exception):
    pass


def draft_task_config(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    legacy_fields = _legacy_ref_rejections(payload)
    if legacy_fields:
        return _legacy_ref_result(run_dir, "draft_task_config", legacy_fields)
    prior_path = _optional_input_path(output_dir, payload.get("prior_task_config_path"))
    input_refs = {"prior_task_config_path": str(prior_path)} if prior_path else {}
    existing = read_json(prior_path) if prior_path else None
    try:
        routing = _routing_from_payload(payload)
        normalized = _normalize_payload(payload, output_dir=output_dir)
        _validate_payload(output_dir, normalized)
        data_input_refs, external_inputs = _data_lineage(output_dir, normalized["data_ref"])
        input_refs.update(data_input_refs)
        version = 1
        artifact_path = run_dir / "artifacts" / f"task_config.draft.v{version}.json"
        changed_fields = _changed_fields(
            existing.get("payload") if isinstance(existing, dict) else None,
            normalized,
        )
        artifact = _artifact_envelope(
            artifact_status="draft",
            artifact_version=version,
            source_type="generated_by_skill",
            input_refs=input_refs,
            payload=normalized,
            validation=_validation_for_payload(normalized),
            confirmed_by=None,
            confirmed_at=None,
            source_draft_path=None,
            external_inputs=external_inputs,
            routing=routing,
        )
        write_json(artifact_path, artifact)
        result = _result(
            run_dir=run_dir,
            phase="draft_task_config",
            status="needs_confirmation",
            summary="Draft task_config is ready for user confirmation.",
            input_refs=input_refs,
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "task_config_path": str(artifact_path.resolve()),
                "task_config_status": "draft",
                "draft_update_mode": "updated" if prior_path else "created",
                "changed_fields": changed_fields,
                "missing_fields": [],
                "missing_columns": [],
                "questions": [],
                "confirmation_summary": _confirmation_summary(normalized, external_inputs),
                "report_path": None,
            },
            artifacts=[{"kind": "task_config", "path": str(artifact_path.resolve())}],
            progress=[
                {
                    "step": "validate_mapping",
                    "status": "success",
                    "message": "Required fields and CSV header checked.",
                },
                {
                    "step": "write_draft",
                    "status": "needs_confirmation",
                    "message": "Draft artifact written.",
                },
            ],
            next_steps=[
                {
                    "skill": SKILL_NAME,
                    "action": "confirm_artifact",
                    "reason": (
                        "task_config draft must be confirmed before "
                        "uplift-model-sample-preparation can run."
                    ),
                    "inputs": {
                        "flow_dir": str(run_dir.resolve()),
                        "source_draft_path": str(artifact_path.resolve()),
                        "artifact_kind": "task_config",
                    },
                    "requires_user_confirmation": True,
                }
            ],
            external_inputs=external_inputs,
        )
    except NeedsClarificationError as exc:
        result = _needs_clarification_result(
            run_dir,
            "draft_task_config",
            str(exc),
            missing_fields=exc.missing_fields,
            missing_columns=exc.missing_columns,
        )
    except UnsupportedInputError as exc:
        result = _unsupported_result(run_dir, "draft_task_config", str(exc))
    except Exception as exc:  # noqa: BLE001
        result = _unexpected_error_result(run_dir, "draft_task_config", exc)
    return result


def confirm_artifact(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    legacy_fields = _legacy_ref_rejections(payload)
    if legacy_fields:
        return _legacy_ref_result(run_dir, "confirm_artifact", legacy_fields)
    artifact_kind = str(payload.get("artifact_kind") or "task_config")
    confirmed_by = str(payload.get("confirmed_by") or "user")
    confirmed_payload = payload.get("confirmed_payload")
    try:
        source_draft_path = _required_input_path(
            output_dir, payload.get("source_draft_path"), "source_draft_path"
        )
        input_refs = {"source_draft_path": str(source_draft_path)}
        if artifact_kind != "task_config":
            raise UnsupportedInputError("uplift-model-task-spec can only confirm task_config artifacts.")
        if confirmed_payload is not None:
            raise UnsupportedInputError(
                "confirm_artifact does not accept payload changes; update the draft first."
            )
        draft = read_json(source_draft_path)
        if draft.get("artifact_kind") != "task_config":
            raise NeedsClarificationError("source_draft_path is not a task_config artifact.")
        if draft.get("artifact_status") != "draft":
            raise NeedsClarificationError("source_draft_path must point to a draft artifact.")
        normalized = _normalize_payload(draft.get("payload") or {}, output_dir=output_dir)
        _validate_payload(output_dir, normalized)
        data_input_refs, external_inputs = _data_lineage(output_dir, normalized["data_ref"])
        sample_inputs: dict[str, Any] = {}
        sample_inputs.update(data_input_refs)
        version = int(draft.get("artifact_version") or 1)
        artifact_path = run_dir / "artifacts" / f"task_config.confirmed.v{version}.json"
        confirmed = dict(draft)
        confirmed["artifact_status"] = "confirmed"
        confirmed["payload"] = normalized
        confirmed["validation"] = _validation_for_payload(normalized)
        confirmed["confirmed_by"] = confirmed_by
        confirmed["confirmed_at"] = _now()
        confirmed["source_draft_path"] = str(source_draft_path)
        if external_inputs:
            confirmed["external_inputs"] = external_inputs
        else:
            confirmed.pop("external_inputs", None)
        write_json(artifact_path, confirmed)
        if external_inputs:
            sample_inputs["data_source"] = external_inputs["data_source"]
        report_path = _write_task_spec_markdown(
            run_dir,
            confirmed=confirmed,
            confirmed_path=artifact_path,
        )
        next_steps = [
            {
                "skill": "uplift-model-sample-preparation",
                "action": "raw_sample_check",
                "reason": "task_config is confirmed; uplift-model-sample-preparation can check the sample.",
                "inputs": {"task_config_path": str(artifact_path.resolve()), **sample_inputs},
                "requires_user_confirmation": False,
            }
        ]
        result = _result(
            run_dir=run_dir,
            phase="confirm_artifact",
            status="success",
            summary="task_config confirmed.",
            input_refs=input_refs,
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "task_config_path": str(artifact_path.resolve()),
                "task_config_status": "confirmed",
                "confirmation_summary": _confirmation_summary(normalized, external_inputs),
                "report_path": str(report_path.resolve()),
            },
            artifacts=[{"kind": "task_config", "path": str(artifact_path.resolve())}],
            progress=[
                {
                    "step": "validate_draft",
                    "status": "success",
                    "message": "Draft payload checked.",
                },
                {
                    "step": "confirm_artifact",
                    "status": "success",
                    "message": "Confirmed artifact written.",
                },
            ],
            next_steps=next_steps,
            external_inputs=external_inputs,
            user_interaction_context={
                "active_result": task_summary(_public_payload(normalized, external_inputs)),
                "changes": task_changes(None, normalized),
            },
        )
    except NeedsClarificationError as exc:
        result = _needs_clarification_result(run_dir, "confirm_artifact", str(exc))
    except UnsupportedInputError as exc:
        result = _unsupported_result(run_dir, "confirm_artifact", str(exc))
    except Exception as exc:  # noqa: BLE001
        result = _unexpected_error_result(run_dir, "confirm_artifact", exc)
    return result


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
    body = request.get("payload") or request
    if not isinstance(body, dict):
        print("payload must be an object.", file=sys.stderr)
        return 1
    if "routing_input" not in body and isinstance(request.get("routing_input"), dict):
        body = dict(body)
        body["routing_input"] = request["routing_input"]
    if not action:
        action = str(body.get("action") or "")
    if action not in {"draft_task_config", "confirm_artifact"}:
        print(f"Unsupported action: {action}", file=sys.stderr)
        return 1

    try:
        if action == "draft_task_config":
            run_dir = ensure_skill_call_dir(output_dir, SKILL_NAME, "task_spec")
            request_path, result_path = next_action_paths(run_dir, action)
            write_json(request_path, request)
            result = draft_task_config(run_dir, output_dir, body)
        else:
            explicit_flow_dir = body.get("flow_dir")
            if explicit_flow_dir:
                run_dir = ensure_existing_skill_call_dir(output_dir, str(explicit_flow_dir))
            else:
                raise ProjectLayoutError("FLOW_DIR_REQUIRED", "flow_dir is required for confirm_artifact.")
            request_path, result_path = next_action_paths(run_dir, action)
            write_json(request_path, request)
            result = confirm_artifact(run_dir, output_dir, body)
    except ProjectLayoutError as exc:
        print(json.dumps(_layout_error_stdout(exc), sort_keys=True))
        return 0
    _attach_transport_paths(result, output_dir, run_dir, result_path)
    result = relativize_paths(result, output_dir)
    write_json(result_path, result)
    write_flow_action_records(
        run_dir=output_dir,
        flow_dir=run_dir,
        skill_name=SKILL_NAME,
        action=action,
        request_path=request_path,
        result_path=result_path,
        result=result,
    )
    print(json.dumps(_stdout_payload(result), sort_keys=True))
    return 0


def _artifact_envelope(
    *,
    artifact_status: str,
    artifact_version: int,
    source_type: str,
    input_refs: dict[str, Any],
    payload: dict[str, Any],
    validation: dict[str, Any],
    confirmed_by: str | None,
    confirmed_at: str | None,
    source_draft_path: str | None,
    external_inputs: dict[str, Any] | None = None,
    routing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = {
        "artifact_kind": "task_config",
        "artifact_status": artifact_status,
        "artifact_version": artifact_version,
        "source_type": source_type,
        "source_refs": [],
        "input_refs": input_refs,
        "created_by": SKILL_CREATED_BY,
        "created_at": _now(),
        "confirmed_by": confirmed_by,
        "confirmed_at": confirmed_at,
        "source_draft_path": source_draft_path,
        "payload": payload,
        "validation": validation,
    }
    if routing:
        artifact["routing"] = routing
    if external_inputs:
        artifact["external_inputs"] = external_inputs
    return artifact


def _routing_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    routing_input = payload.get("routing_input")
    if routing_input is None:
        return None
    if not isinstance(routing_input, dict):
        raise NeedsClarificationError("routing_input must be an object when provided.")
    task_type = str(routing_input.get("task_type") or "").strip().lower()
    if task_type != "uplift":
        raise UnsupportedInputError("routing_input.task_type must be uplift.")
    routing_basis = routing_input.get("routing_basis")
    if routing_basis is not None and not isinstance(routing_basis, dict):
        raise NeedsClarificationError("routing_input.routing_basis must be an object when provided.")
    return {
        "source_skill": "model-task-routing",
        "routing_input": routing_input,
        "consumed_by": SKILL_NAME,
        "consumed_at": _now(),
    }


def _changed_fields(old_payload: dict[str, Any] | None, new_payload: dict[str, Any]) -> list[str]:
    if old_payload is None:
        return sorted(new_payload)
    fields = sorted(set(old_payload) | set(new_payload))
    return [field for field in fields if old_payload.get(field) != new_payload.get(field)]


def _confirmation_summary(
    payload: dict[str, Any], external_inputs: dict[str, Any] | None = None
) -> dict[str, Any]:
    summary = {
        "data_ref": payload["data_ref"],
        "output_dir": payload["output_dir"],
        "business_context": payload["business_context"],
        "outcome": {
            "column": payload["outcome_column"],
            "type": payload["outcome_type"],
        },
        "treatment": {
            "column": payload["treatment_column"],
            "treatment_value": payload["treatment_value"],
            "control_value": payload["control_value"],
        },
        "unit_id_column": payload["unit_id_column"],
        "time_column": payload["time_column"],
        "exclude_columns": payload["exclude_columns"],
        "report_preferences": payload["report_preferences"],
    }
    if external_inputs:
        summary.pop("data_ref", None)
        summary["data_source"] = external_inputs["data_source"]
    return summary


def _public_payload(
    payload: dict[str, Any], external_inputs: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not external_inputs:
        return payload
    public = dict(payload)
    public.pop("data_ref", None)
    public["data_source"] = external_inputs["data_source"]
    return public


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
        input_refs={},
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "rejected_fields": rejected_fields,
            "missing_fields": [item["suggested_field"] for item in rejected_fields],
            "task_config_path": None,
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


def _attach_transport_paths(
    result: dict[str, Any],
    output_dir: Path,
    run_dir: Path,
    result_path: Path,
) -> None:
    outputs = result.setdefault("outputs", {})
    outputs["flow_dir"] = to_run_relative_path(output_dir, run_dir)
    outputs["result_path"] = to_run_relative_path(output_dir, result_path)


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
        "inputs": result.get("input_refs") or {},
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
    return {
        "status": "needs_input",
        "summary": str(exc),
        "outputs": {"missing_fields": ["output_dir"], "report_path": None},
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


def _needs_clarification_result(
    run_dir: Path,
    phase: str,
    message: str,
    *,
    missing_fields: list[str] | None = None,
    missing_columns: list[str] | None = None,
) -> dict[str, Any]:
    questions = []
    if missing_fields:
        questions.append(f"Please provide required task fields: {', '.join(missing_fields)}.")
    if missing_columns:
        questions.append(
            f"Please provide CSV column names that exist: {', '.join(missing_columns)}."
        )
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="needs_clarification",
        summary=message,
        input_refs={},
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "missing_fields": missing_fields or [],
            "missing_columns": missing_columns or [],
            "questions": questions,
            "task_config_path": None,
            "report_path": None,
        },
        progress=[{"step": phase, "status": "needs_clarification", "message": message}],
    )


def _normalize_nullable_column(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"none", "null", "无", "沒有", "没有", "no"}:
            return None
        return stripped
    raise NeedsClarificationError("Optional column fields must be strings or null.")


def _normalize_business_context(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        user_scenario = value.strip()
        if user_scenario == "" or user_scenario.lower() in {
            "none",
            "null",
            "no",
            "n/a",
            "na",
            "无",
            "没有",
            "暂无",
        }:
            return None
        _raise_if_encoding_damaged_business_context(user_scenario)
        return {"source": "user_natural_language", "user_scenario": user_scenario}
    if not isinstance(value, dict):
        raise NeedsClarificationError("business_context must be an object, string, or null.")
    source = str(value.get("source") or "user_natural_language").strip()
    user_scenario = str(value.get("user_scenario") or "").strip()
    if source != "user_natural_language" or not user_scenario:
        raise NeedsClarificationError(
            "business_context requires source=user_natural_language and a non-empty user_scenario."
        )
    _raise_if_encoding_damaged_business_context(user_scenario)
    return {"source": source, "user_scenario": user_scenario}


def _raise_if_encoding_damaged_business_context(text: str) -> None:
    if _looks_encoding_damaged(text):
        raise NeedsClarificationError(
            "business_context.user_scenario appears encoding-damaged. "
            "Please resend the business scenario as UTF-8 or ASCII-escaped JSON."
        )


def _looks_encoding_damaged(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 6:
        return False
    damaged_count = stripped.count("?") + stripped.count("\ufffd")
    return damaged_count / len(stripped) >= 0.6


def _normalize_payload(payload: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    missing_fields = [field for field in REQUIRED_FIELDS if _is_missing(payload.get(field))]
    if missing_fields:
        raise NeedsClarificationError(
            "Required task_config fields are missing.", missing_fields=missing_fields
        )
    normalized_output_dir = str(_resolve_output_dir(payload["output_dir"]))
    if output_dir.resolve() != Path(normalized_output_dir):
        raise NeedsClarificationError(
            "payload.output_dir must match --output-dir.", missing_fields=["output_dir"]
        )
    outcome_type = str(payload["outcome_type"]).strip().lower()
    if outcome_type not in ALLOWED_OUTCOME_TYPES:
        raise UnsupportedInputError("outcome_type must be binary or continuous.")
    exclude_columns = payload.get("exclude_columns", [])
    if exclude_columns is None:
        exclude_columns = []
    if not isinstance(exclude_columns, list) or not all(
        isinstance(item, str) for item in exclude_columns
    ):
        raise NeedsClarificationError("exclude_columns must be a list of strings.")
    business_context = _normalize_business_context(payload.get("business_context"))
    report_preferences_input = payload.get("report_preferences")
    if not isinstance(report_preferences_input, dict) and payload.get("user_language"):
        report_preferences_input = {"language": payload.get("user_language")}
    report_preferences, _language_warning = normalize_report_preferences(report_preferences_input)
    return {
        "data_ref": str(payload["data_ref"]).strip(),
        "output_dir": normalized_output_dir,
        "outcome_column": str(payload["outcome_column"]).strip(),
        "outcome_type": outcome_type,
        "treatment_column": str(payload["treatment_column"]).strip(),
        "treatment_value": payload["treatment_value"],
        "control_value": payload["control_value"],
        "unit_id_column": _normalize_nullable_column(payload.get("unit_id_column")),
        "time_column": _normalize_nullable_column(payload.get("time_column")),
        "exclude_columns": [item.strip() for item in exclude_columns if item.strip()],
        "modeling_goal": str(payload.get("modeling_goal") or "uplift_modeling").strip(),
        "business_context": business_context,
        "report_preferences": report_preferences,
    }


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _resolve_output_dir(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def _read_csv_columns(data_path: Path) -> set[str]:
    try:
        with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read CSV header: {data_path}") from exc
    return {str(column) for column in header}


def _required_input_path(output_dir: Path, value: Any, field: str) -> Path:
    if _is_missing(value):
        raise NeedsClarificationError(f"{field} is required.", missing_fields=[field])
    return _resolve_input_path(output_dir, str(value))


def _optional_input_path(output_dir: Path, value: Any) -> Path | None:
    if _is_missing(value):
        return None
    return _resolve_input_path(output_dir, str(value))


def _resolve_input_path(output_dir: Path, value: str) -> Path:
    return resolve_run_path(output_dir, value)


def _resolve_data_path(output_dir: Path, data_ref: str) -> Path:
    return resolve_run_path(output_dir, data_ref)


def _data_lineage(output_dir: Path, data_ref: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _resolve_data_path(output_dir, data_ref)
    if path.suffix.lower() != ".csv":
        raise UnsupportedInputError(
            "uplift-model-task-spec first version only supports local CSV data_ref."
        )
    if not path.exists():
        raise NeedsClarificationError(f"CSV file does not exist: {path}")
    return {}, {"data_source": _local_csv_source(path)}


def _local_csv_source(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "kind": "local_csv",
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result(
    *,
    run_dir: Path,
    phase: str,
    status: str,
    summary: str,
    input_refs: dict[str, Any],
    outputs: dict[str, Any],
    issues: list[dict[str, Any]] | None = None,
    artifacts: list[dict[str, Any]] | None = None,
    error: dict[str, Any] | None = None,
    progress: list[dict[str, Any]] | None = None,
    next_steps: list[dict[str, Any]] | None = None,
    external_inputs: dict[str, Any] | None = None,
    user_interaction_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "skill_name": SKILL_NAME,
        "run_id": run_dir.name,
        "phase": phase,
        "status": status,
        "summary": summary,
        "input_refs": input_refs,
        "outputs": outputs,
        "issues": issues or [],
        "artifacts": artifacts or [],
        "error": error,
        "metadata": {"llm_usage": dict(DEFAULT_LLM_USAGE)},
        "progress": progress or [],
        "next_steps": next_steps or [],
        "created_at": _now(),
    }
    if external_inputs:
        result["external_inputs"] = external_inputs
    result["user_interaction"] = build_user_interaction(result, **(user_interaction_context or {}))
    return result


def _unsupported_result(run_dir: Path, phase: str, message: str) -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="unsupported",
        summary=message,
        input_refs={},
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "unsupported_reason": message,
            "task_config_path": None,
            "report_path": None,
        },
        error={
            "code": "UNSUPPORTED_INPUT",
            "message": message,
            "recoverable": True,
            "retryable": False,
            "raw_error": None,
        },
        progress=[{"step": phase, "status": "unsupported", "message": message}],
    )


def _unexpected_error_result(run_dir: Path, phase: str, exc: Exception) -> dict[str, Any]:
    return _result(
        run_dir=run_dir,
        phase=phase,
        status="error",
        summary=f"{phase} failed unexpectedly.",
        input_refs={},
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "task_config_path": None,
            "report_path": None,
        },
        error={
            "code": "TASK_DEFINITION_FAILED",
            "message": f"{phase} failed unexpectedly.",
            "recoverable": False,
            "retryable": False,
            "raw_error": str(exc),
        },
        progress=[{"step": phase, "status": "error", "message": str(exc)}],
    )


def _validate_payload(output_dir: Path, payload: dict[str, Any]) -> None:
    if payload["modeling_goal"] != "uplift_modeling":
        raise UnsupportedInputError("uplift-model-task-spec only supports modeling_goal=uplift_modeling.")
    data_path = _resolve_data_path(output_dir, payload["data_ref"])
    if data_path.suffix.lower() != ".csv":
        raise UnsupportedInputError(
            "uplift-model-task-spec first version only supports local CSV data_ref."
        )
    if not data_path.exists():
        raise NeedsClarificationError(f"CSV file does not exist: {data_path}")
    columns = _read_csv_columns(data_path)
    requested = [payload["outcome_column"], payload["treatment_column"]]
    requested.extend(
        column for column in [payload["unit_id_column"], payload["time_column"]] if column
    )
    requested.extend(payload["exclude_columns"])
    missing_columns = [column for column in requested if column not in columns]
    if missing_columns:
        raise NeedsClarificationError(
            "Some specified columns do not exist in the CSV header.",
            missing_columns=missing_columns,
        )


def _validation_for_payload(payload: dict[str, Any]) -> dict[str, Any]:
    assumptions = []
    if payload["unit_id_column"] is None:
        assumptions.append("unit_id_column is not provided.")
    if payload["time_column"] is None:
        assumptions.append("time_column is not provided.")
    return {
        "filled_fields": list(REQUIRED_FIELDS),
        "missing_fields": [],
        "assumptions": assumptions,
        "warnings": [],
    }


def _write_task_spec_markdown(
    run_dir: Path,
    *,
    confirmed: dict[str, Any],
    confirmed_path: Path,
) -> Path:
    version = int(confirmed["artifact_version"])
    payload = confirmed["payload"]
    routing = confirmed.get("routing") if isinstance(confirmed.get("routing"), dict) else None
    report_path = run_dir / "report.md"
    business_context = payload.get("business_context")
    language = str(payload.get("report_preferences", {}).get("language") or "zh-CN")
    if is_zh(language):
        lines = [
            "# 任务定义",
            "",
            f"状态：{confirmed['artifact_status']}",
            f"版本：{version}",
            "",
        ]
    else:
        lines = [
            "# Task Definition",
            "",
            f"Status: {confirmed['artifact_status']}",
            f"Version: {version}",
            "",
        ]
    if business_context:
        lines.extend(
            [
                "## 业务背景" if is_zh(language) else "## Business Context",
                "",
                str(business_context["user_scenario"]),
                "",
            ]
        )
    if is_zh(language):
        lines.extend(
            [
                "## 已确认输入",
                "",
                f"- 数据：`{payload['data_ref']}`",
                f"- 输出目录：`{payload['output_dir']}`",
                f"- 建模目标：`{payload['modeling_goal']}`",
                f"- 报告语言：`{language}`",
                "",
                "## 目标变量",
                "",
                f"- 字段：`{payload['outcome_column']}`",
                f"- 类型：`{payload['outcome_type']}`",
                "",
                "## 处理变量",
                "",
                f"- 字段：`{payload['treatment_column']}`",
                f"- 处理组取值：`{payload['treatment_value']}`",
                f"- 对照组取值：`{payload['control_value']}`",
                "",
                "## 可选上下文",
                "",
                f"- 单元 ID 字段：{_format_optional(payload['unit_id_column'], language)}",
                f"- 时间字段：{_format_optional(payload['time_column'], language)}",
                f"- 排除字段：{_format_list(payload['exclude_columns'], language)}",
                "",
                "## 下一步",
                "",
                "继续执行 `uplift-model-sample-preparation`。",
                "",
                "## 附录：系统引用",
                "",
                f"- task_config_path: `{confirmed_path.resolve()}`",
                f"- flow_dir: `{run_dir.resolve()}`",
                *_routing_report_lines(routing, language),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Confirmed Inputs",
                "",
                f"- Data: `{payload['data_ref']}`",
                f"- Output directory: `{payload['output_dir']}`",
                f"- Goal: `{payload['modeling_goal']}`",
                f"- Report language: `{language}`",
                "",
                "## Outcome",
                "",
                f"- Column: `{payload['outcome_column']}`",
                f"- Type: `{payload['outcome_type']}`",
                "",
                "## Treatment",
                "",
                f"- Column: `{payload['treatment_column']}`",
                f"- Treatment value: `{payload['treatment_value']}`",
                f"- Control value: `{payload['control_value']}`",
                "",
                "## Optional Context",
                "",
                f"- Unit ID column: {_format_optional(payload['unit_id_column'], language)}",
                f"- Time column: {_format_optional(payload['time_column'], language)}",
                f"- Excluded columns: {_format_list(payload['exclude_columns'], language)}",
                "",
                "## Next Step",
                "",
                "Continue to `uplift-model-sample-preparation`.",
                "",
                "## Appendix: System References",
                "",
                f"- task_config_path: `{confirmed_path.resolve()}`",
                f"- flow_dir: `{run_dir.resolve()}`",
                *_routing_report_lines(routing, language),
                "",
            ]
        )
    write_text(report_path, "\n".join(lines))
    return report_path


def _format_optional(value: Any, language: str = "en-US") -> str:
    if value is None:
        return "未提供" if is_zh(language) else "not provided"
    return f"`{value}`"


def _format_list(values: list[Any], language: str = "en-US") -> str:
    if not values:
        return "无" if is_zh(language) else "none"
    return ", ".join(f"`{value}`" for value in values)


def _routing_report_lines(routing: dict[str, Any] | None, language: str) -> list[str]:
    if not routing:
        return []
    routing_input = routing.get("routing_input") or {}
    if not isinstance(routing_input, dict):
        return []
    routed_at = routing_input.get("routed_at")
    q1_target = (routing_input.get("routing_basis") or {}).get("q1_target")
    if is_zh(language):
        lines = ["- routing_source: `model-task-routing`"]
        if routed_at:
            lines.append(f"- routed_at: `{routed_at}`")
        if q1_target:
            lines.append(f"- routing_target: {q1_target}")
        return lines
    lines = ["- routing_source: `model-task-routing`"]
    if routed_at:
        lines.append(f"- routed_at: `{routed_at}`")
    if q1_target:
        lines.append(f"- routing_target: {q1_target}")
    return lines


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
