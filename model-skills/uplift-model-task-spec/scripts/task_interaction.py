from __future__ import annotations

from typing import Any


def build_user_interaction(
    result: dict[str, Any],
    *,
    active_result: dict[str, Any] | None = None,
    changes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = result["status"]
    interaction: dict[str, Any] = {
        "type": _type_for(status),
        "subject": "uplift_modeling_task",
        "facts": _facts(result),
    }
    if status == "needs_confirmation":
        interaction["decision"] = {"required": True, "kind": "adopt_or_revise"}
    if active_result:
        interaction["active_result"] = active_result
    if changes:
        interaction["changes"] = changes
    return interaction


def task_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return _without_none(
        {
            "outcome": {
                "column": payload.get("outcome_column"),
                "type": payload.get("outcome_type"),
            },
            "treatment": {
                "column": payload.get("treatment_column"),
                "treatment_value": payload.get("treatment_value"),
                "control_value": payload.get("control_value"),
            },
            "unit_id_column": payload.get("unit_id_column"),
            "time_column": payload.get("time_column"),
            "excluded_field_count": len(payload.get("exclude_columns") or []),
        }
    )


def task_changes(previous: dict[str, Any] | None, adopted: dict[str, Any]) -> dict[str, Any] | None:
    if not previous:
        return None
    hidden = {"data_ref", "output_dir"}
    changed: dict[str, Any] = {}
    for key in sorted(set(previous) | set(adopted)):
        if key in hidden or previous.get(key) == adopted.get(key):
            continue
        if key == "exclude_columns":
            before = list(previous.get(key) or [])
            after = list(adopted.get(key) or [])
            changed["excluded_fields"] = _list_change(before, after)
        else:
            changed[key] = _scalar_change(previous.get(key), adopted.get(key))
    return {"changed_fields": changed} if changed else None


def _facts(result: dict[str, Any]) -> dict[str, Any]:
    outputs = result.get("outputs") or {}
    facts: dict[str, Any] = {"issue_count": len(result.get("issues") or [])}
    summary = outputs.get("confirmation_summary")
    if summary:
        facts.update(_task_summary(summary))
    missing_fields = outputs.get("missing_fields") or []
    missing_columns = outputs.get("missing_columns") or []
    if missing_fields:
        facts["missing_information"] = missing_fields[:5]
    if missing_columns:
        facts["unknown_fields"] = missing_columns[:5]
    return facts


def _task_summary(summary: dict[str, Any]) -> dict[str, Any]:
    treatment = summary.get("treatment") or {}
    return _without_none(
        {
            "outcome": summary.get("outcome") or {},
            "treatment": treatment,
            "unit_id_column": summary.get("unit_id_column"),
            "time_column": summary.get("time_column"),
            "excluded_field_count": len(summary.get("exclude_columns") or []),
        }
    )


def _without_none(value: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, dict):
            item = _without_none(item)
            if not item:
                continue
        compact[key] = item
    return compact


def _list_change(before: list[Any], after: list[Any]) -> dict[str, Any]:
    added = [item for item in after if item not in before]
    removed = [item for item in before if item not in after]
    return {
        "added_count": len(added),
        "added_examples": added[:5],
        "removed_count": len(removed),
        "removed_examples": removed[:5],
        "total_count": len(after),
    }


def _scalar_change(before: Any, after: Any) -> dict[str, Any]:
    change: dict[str, Any] = {
        "before_present": before is not None,
        "after_present": after is not None,
    }
    if before is not None:
        change["before"] = before
    if after is not None:
        change["after"] = after
    return change


def _type_for(status: str) -> str:
    return {
        "success": "completion",
        "partial_success": "summary",
        "needs_input": "clarification",
        "needs_clarification": "clarification",
        "needs_confirmation": "confirmation",
        "unsupported": "recovery",
        "error": "recovery",
        "validation_failed_need_split_revision": "recovery",
    }[status]
