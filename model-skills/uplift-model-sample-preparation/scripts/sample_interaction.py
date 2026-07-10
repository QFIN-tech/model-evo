from __future__ import annotations

from typing import Any


def build_user_interaction(
    result: dict[str, Any],
    *,
    active_result: dict[str, Any] | None = None,
    changes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = result["status"]
    outputs = result.get("outputs") or {}
    facts: dict[str, Any] = {"issue_count": len(result.get("issues") or [])}
    if "split_row_counts" in outputs:
        facts["split_row_counts"] = {
            key: value for key, value in outputs["split_row_counts"].items() if value is not None
        }
    if "blocking_reasons" in outputs:
        facts["blocking_reason_count"] = len(outputs["blocking_reasons"])
    interaction: dict[str, Any] = {
        "type": _type_for(status),
        "subject": _subject(result.get("phase")),
        "facts": facts,
    }
    if status == "needs_confirmation":
        interaction["decision"] = {"required": True, "kind": "adopt_or_revise"}
    if active_result:
        interaction["active_result"] = active_result
    if changes:
        interaction["changes"] = changes
    return interaction


def plan_summary(artifact_kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    if artifact_kind == "split_plan":
        plan = payload.get("plan") or {}
        result = {"method": plan.get("split_method")}
        if plan.get("ratios"):
            result["ratios"] = plan["ratios"]
        if plan.get("oot_window"):
            result["out_of_time_window"] = plan["oot_window"]
        return {key: value for key, value in result.items() if value is not None}
    plan = payload.get("plan") or {}
    treatment = plan.get("treatment") or {}
    outcome = plan.get("outcome") or {}
    return {
        key: value
        for key, value in {
            "treatment_field": treatment.get("source_column"),
            "treatment_value": treatment.get("treatment_value"),
            "control_value": treatment.get("control_value"),
            "outcome_field": outcome.get("source_column"),
            "outcome_type": outcome.get("type"),
        }.items()
        if value is not None
    }


def plan_changes(previous: dict[str, Any] | None, adopted: dict[str, Any]) -> dict[str, Any] | None:
    if not previous:
        return None
    before = plan_summary(
        "split_plan" if "split_method" in (previous.get("plan") or {}) else "data_semantics_plan",
        previous,
    )
    after = plan_summary(
        "split_plan" if "split_method" in (adopted.get("plan") or {}) else "data_semantics_plan",
        adopted,
    )
    fields = {
        key: _scalar_change(before.get(key), after.get(key))
        for key in sorted(set(before) | set(after))
        if before.get(key) != after.get(key)
    }
    return {"changed_fields": fields} if fields else None


def _subject(phase: Any) -> str:
    return {
        "raw_sample_check": "sample_semantics",
        "split_planning": "sample_split",
        "post_split_validation": "modeling_sample",
        "confirm_artifact": "sample_plan",
    }.get(str(phase), "sample_preparation")


def _type_for(status: str) -> str:
    return {
        "success": "completion",
        "partial_success": "summary",
        "needs_input": "clarification",
        "needs_clarification": "clarification",
        "needs_confirmation": "confirmation",
        "unsupported": "recovery",
        "failed": "recovery",
        "validation_failed_need_split_revision": "recovery",
    }[status]


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
