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
    summary = outputs.get("confirmation_summary") or {}
    for key in ("included_variables_count", "skipped_variables_count"):
        if key in summary:
            facts[key] = summary[key]
    if "overall_status" in outputs:
        facts["balance_risk"] = outputs["overall_status"]
        facts["imbalanced_variable_count"] = len(outputs.get("top_imbalanced_variables") or [])
        facts["skipped_variable_count"] = len(outputs.get("skipped_variables") or [])
    interaction: dict[str, Any] = {
        "type": _type_for(status),
        "subject": "treatment_control_comparability",
        "facts": facts,
    }
    if status == "needs_confirmation":
        interaction["decision"] = {"required": True, "kind": "adopt_or_revise"}
    if active_result:
        interaction["active_result"] = active_result
    if changes:
        interaction["changes"] = changes
    return interaction


def covariate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    included = [item["name"] for item in payload.get("covariates", []) if item.get("included")]
    return {"variable_count": len(included), "variable_examples": included[:5]}


def covariate_changes(
    previous: dict[str, Any] | None, adopted: dict[str, Any]
) -> dict[str, Any] | None:
    if not previous:
        return None
    before = [item["name"] for item in previous.get("covariates", []) if item.get("included")]
    after = [item["name"] for item in adopted.get("covariates", []) if item.get("included")]
    added = [item for item in after if item not in before]
    removed = [item for item in before if item not in after]
    if not added and not removed:
        return None
    return {
        "added_count": len(added),
        "added_examples": added[:5],
        "removed_count": len(removed),
        "removed_examples": removed[:5],
        "total_count": len(after),
    }


def _type_for(status: str) -> str:
    return {
        "success": "completion",
        "partial_success": "summary",
        "needs_input": "clarification",
        "needs_clarification": "clarification",
        "needs_confirmation": "confirmation",
        "failed": "recovery",
        "validation_failed_need_split_revision": "recovery",
    }[status]
