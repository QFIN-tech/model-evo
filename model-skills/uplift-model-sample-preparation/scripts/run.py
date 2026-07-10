from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _common.data_loader import (
    DataLoaderImportError,
    DataSourceError,
    UnsupportedDataSourceError,
    inspect_data_source,
    load_table,
)
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
from sample_interaction import build_user_interaction, plan_changes, plan_summary

SKILL_NAME = "uplift-model-sample-preparation"
SKILL_CREATED_BY = "uplift-model-sample-preparation-skill"
MODELING_TREATMENT_COLUMN = "__uplift_modeling_treatment__"
MODELING_OUTCOME_COLUMN = "__uplift_modeling_outcome__"
MODELING_SPLIT_COLUMN = "__uplift_split__"
SPLIT_NAMES = ("train", "test", "valid", "oot")
DISPLAY_SPLIT_ORDER = ("train", "valid", "test", "oot")
ALLOWED_REF_FIELDS = {"data_ref"}
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


class UnsupportedInputError(Exception):
    pass


def run_raw_sample_check(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    phase = "raw_sample_check"
    try:
        task_config_path = _required_input_path(output_dir, payload.get("task_config_path"), "task_config_path")
        task_config = _read_artifact(task_config_path, artifact_kind="task_config", require_confirmed=True)
        task_payload = task_config["payload"]
        dataframe, external_inputs = _load_source_table(task_config, payload, output_dir=output_dir)
        plan_payload, issues = _build_data_semantics_payload(dataframe, task_payload)
        input_paths = {"task_config_path": str(task_config_path)}
        blocking = any(issue["blocking"] for issue in issues)
        if blocking:
            return _result(
                run_dir=run_dir,
                phase=phase,
                status="failed",
                summary="Raw sample check found blocking issues.",
                input_paths=input_paths,
                outputs={
                    "flow_dir": str(run_dir.resolve()),
                    "data_semantics_preview": plan_payload,
                    "data_semantics_plan_path": None,
                    "report_path": None,
                },
                issues=issues,
                progress=[
                    {
                        "step": "raw_sample_check",
                        "status": "failed",
                        "message": "Blocking sample issues found.",
                    }
                ],
                external_inputs=external_inputs,
            )
        artifact_path = _write_data_semantics_plan_draft(
            run_dir,
            input_paths=input_paths,
            external_inputs=external_inputs,
            payload=plan_payload,
        )
        return _result(
            run_dir=run_dir,
            phase=phase,
            status="needs_confirmation",
            summary="Draft data_semantics_plan created and needs user confirmation.",
            input_paths=input_paths,
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "data_semantics_plan_path": str(artifact_path.resolve()),
                "data_source": external_inputs.get("data_source"),
                "report_path": None,
            },
            issues=issues,
            artifacts=[{"kind": "data_semantics_plan", "path": str(artifact_path.resolve())}],
            progress=[
                {
                    "step": "raw_sample_check",
                    "status": "success",
                    "message": "Raw sample support checked.",
                },
                {
                    "step": "data_semantics_plan",
                    "status": "needs_confirmation",
                    "message": "Draft artifact written.",
                },
            ],
            next_steps=[
                {
                    "skill": SKILL_NAME,
                    "action": "confirm_artifact",
                    "reason": "data_semantics_plan must be confirmed before split_planning.",
                    "inputs": {
                        "flow_dir": str(run_dir.resolve()),
                        "source_draft_path": str(artifact_path.resolve()),
                        "artifact_kind": "data_semantics_plan",
                    },
                    "requires_user_confirmation": True,
                }
            ],
            external_inputs=external_inputs,
        )
    except DataLoaderImportError as exc:
        return _dependency_failure_result(run_dir, phase, exc)
    except UnsupportedDataSourceError as exc:
        return _unsupported_result(run_dir, phase, str(exc))
    except UnsupportedInputError as exc:
        return _unsupported_result(run_dir, phase, str(exc))
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
        artifact_kind = str(payload.get("artifact_kind") or "")
        confirmed_by = str(payload.get("confirmed_by") or "user")
        if artifact_kind not in {"data_semantics_plan", "split_plan"}:
            raise UnsupportedInputError(
                "uplift-model-sample-preparation can confirm data_semantics_plan or split_plan only."
            )
        if payload.get("confirmed_payload") is not None:
            raise UnsupportedInputError(
                "confirm_artifact does not accept payload changes; rerun the planning action."
            )
        draft = _read_artifact(source_draft_path, artifact_kind=artifact_kind, require_draft=True)
        version = int(draft.get("artifact_version") or 1)
        artifact_path = run_dir / "artifacts" / f"{artifact_kind}.confirmed.v{version}.json"
        confirmed = dict(draft)
        confirmed["artifact_status"] = "confirmed"
        confirmed["confirmed_by"] = confirmed_by
        confirmed["confirmed_at"] = _now()
        confirmed["source_draft_path"] = str(source_draft_path)
        write_json(artifact_path, confirmed)
        output_field = f"{artifact_kind}_path"
        next_steps = _confirmation_next_steps(run_dir, artifact_kind, artifact_path, confirmed)
        return _result(
            run_dir=run_dir,
            phase=phase,
            status="success",
            summary=f"{artifact_kind} confirmed.",
            input_paths={"source_draft_path": str(source_draft_path)},
            outputs={
                "flow_dir": str(run_dir.resolve()),
                output_field: str(artifact_path.resolve()),
                f"{artifact_kind}_status": "confirmed",
                "report_path": None,
            },
            artifacts=[{"kind": artifact_kind, "path": str(artifact_path.resolve())}],
            progress=[
                {
                    "step": "validate_draft",
                    "status": "success",
                    "message": "Draft artifact checked.",
                },
                {
                    "step": "confirm_artifact",
                    "status": "success",
                    "message": "Confirmed artifact written.",
                },
            ],
            next_steps=next_steps,
            user_interaction_context={
                "active_result": plan_summary(artifact_kind, confirmed.get("payload") or {}),
                "changes": plan_changes(None, confirmed.get("payload") or {}),
            },
        )
    except UnsupportedInputError as exc:
        return _unsupported_result(run_dir, phase, str(exc))
    except MissingInputError as exc:
        return _needs_input_result(run_dir, phase, str(exc), exc.missing_fields)
    except Exception as exc:  # noqa: BLE001
        return _unexpected_error_result(run_dir, phase, exc)


def run_split_planning(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    phase = "split_planning"
    try:
        if payload.get("role_column"):
            raise UnsupportedInputError(
                "role_column is not supported in the first uplift-model-sample-preparation version."
            )
        task_config_path = _required_input_path(output_dir, payload.get("task_config_path"), "task_config_path")
        semantics_path = _required_input_path(
            output_dir,
            payload.get("data_semantics_plan_path"),
            "data_semantics_plan_path",
        )
        task_config = _read_artifact(task_config_path, artifact_kind="task_config", require_confirmed=True)
        semantics = _read_artifact(
            semantics_path, artifact_kind="data_semantics_plan", require_confirmed=True
        )
        dataframe, external_inputs = _load_source_table(task_config, payload, output_dir=output_dir)
        input_paths = {
            "task_config_path": str(task_config_path),
            "data_semantics_plan_path": str(semantics_path),
        }
        plan_payload, issues = _build_split_plan_payload(
            dataframe,
            task_config["payload"],
            split_method=str(payload.get("split_method") or "random"),
            ratios=payload.get("ratios"),
            time_column=payload.get("time_column"),
            oot_window=payload.get("oot_window"),
            non_oot_random_ratios=payload.get("non_oot_random_ratios"),
            random_seed=int(payload.get("random_seed", 42)),
        )
        blocking = any(issue["blocking"] for issue in issues)
        if blocking:
            return _result(
                run_dir=run_dir,
                phase=phase,
                status="failed",
                summary="Split planning found blocking issues.",
                input_paths=input_paths,
                outputs={
                    "flow_dir": str(run_dir.resolve()),
                    "split_plan_preview": plan_payload,
                    "split_plan_path": None,
                    "report_path": None,
                },
                issues=issues,
                progress=[
                    {
                        "step": "split_planning",
                        "status": "failed",
                        "message": "Blocking split issues found.",
                    }
                ],
                external_inputs=external_inputs,
            )
        artifact_path = _write_split_plan_draft(
            run_dir,
            input_paths=input_paths,
            external_inputs=external_inputs,
            payload=plan_payload,
        )
        return _result(
            run_dir=run_dir,
            phase=phase,
            status="needs_confirmation",
            summary="Draft split_plan created and needs user confirmation.",
            input_paths=input_paths,
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "split_plan_path": str(artifact_path.resolve()),
                "report_path": None,
            },
            issues=issues,
            artifacts=[{"kind": "split_plan", "path": str(artifact_path.resolve())}],
            progress=[
                {
                    "step": "split_planning",
                    "status": "success",
                    "message": "Split plan checked.",
                },
                {
                    "step": "split_plan",
                    "status": "needs_confirmation",
                    "message": "Draft artifact written.",
                },
            ],
            next_steps=[
                {
                    "skill": SKILL_NAME,
                    "action": "confirm_artifact",
                    "reason": "split_plan must be confirmed before post_split_validation.",
                    "inputs": {
                        "flow_dir": str(run_dir.resolve()),
                        "source_draft_path": str(artifact_path.resolve()),
                        "artifact_kind": "split_plan",
                    },
                    "requires_user_confirmation": True,
                }
            ],
            external_inputs=external_inputs,
        )
    except DataLoaderImportError as exc:
        return _dependency_failure_result(run_dir, phase, exc)
    except UnsupportedDataSourceError as exc:
        return _unsupported_result(run_dir, phase, str(exc))
    except UnsupportedInputError as exc:
        return _unsupported_result(run_dir, phase, str(exc))
    except (MissingInputError, DataSourceError) as exc:
        return _needs_input_result(run_dir, phase, str(exc), getattr(exc, "missing_fields", []))
    except Exception as exc:  # noqa: BLE001
        return _unexpected_error_result(run_dir, phase, exc)


def run_post_split_validation(
    run_dir: Path, output_dir: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    phase = "post_split_validation"
    try:
        task_config_path = _required_input_path(output_dir, payload.get("task_config_path"), "task_config_path")
        semantics_path = _required_input_path(
            output_dir,
            payload.get("data_semantics_plan_path"),
            "data_semantics_plan_path",
        )
        split_plan_path = _required_input_path(output_dir, payload.get("split_plan_path"), "split_plan_path")
        task_config = _read_artifact(task_config_path, artifact_kind="task_config", require_confirmed=True)
        semantics = _read_artifact(
            semantics_path, artifact_kind="data_semantics_plan", require_confirmed=True
        )
        split_plan = _read_artifact(split_plan_path, artifact_kind="split_plan", require_confirmed=True)
        stale_issues = _lineage_issues(output_dir, task_config_path, semantics_path, split_plan)
        dataframe, external_inputs = _load_source_table(task_config, payload, output_dir=output_dir)
        prepared = _apply_modeling_columns(dataframe, semantics["payload"])
        split_frames = _split_data(prepared, split_plan["payload"])
        support, support_issues, warnings = _validate_split_frames(
            split_frames, semantics["payload"], task_config["payload"]
        )
        issues = stale_issues + support_issues
        row_counts = {
            name: None if frame is None else int(len(frame)) for name, frame in split_frames.items()
        }
        input_paths = {
            "task_config_path": str(task_config_path),
            "data_semantics_plan_path": str(semantics_path),
            "split_plan_path": str(split_plan_path),
        }
        blocking_reasons = [issue["message"] for issue in issues if issue["blocking"]]
        if blocking_reasons:
            return _result(
                run_dir=run_dir,
                phase=phase,
                status="validation_failed_need_split_revision",
                summary="Post-split validation failed; revise split_plan before writing datasets.",
                input_paths=input_paths,
                outputs={
                    "flow_dir": str(run_dir.resolve()),
                    "failed_split_plan_path": str(split_plan_path),
                    "split_row_counts": row_counts,
                    "blocking_reasons": blocking_reasons,
                    "support": support,
                    "modeling_sample_spec_path": None,
                    "modeling_sample_path": None,
                    "report_path": None,
                },
                issues=issues,
                progress=[
                    {
                        "step": "dry_run_split",
                        "status": "success",
                        "message": "Split executed in memory.",
                    },
                    {
                        "step": "post_split_validation",
                        "status": "validation_failed_need_split_revision",
                        "message": "Blocking issues found before writing datasets.",
                    },
                ],
                next_steps=[
                    {
                        "skill": SKILL_NAME,
                        "action": "split_planning",
                        "reason": "Revise split_plan using the validation failure context.",
                        "inputs": {
                            "task_config_path": str(task_config_path),
                            "data_semantics_plan_path": str(semantics_path),
                        },
                        "requires_user_confirmation": False,
                    }
                ],
                external_inputs=external_inputs,
            )
        spec_path, dataset_paths = _write_modeling_sample_spec(
            run_dir,
            input_paths=input_paths,
            external_inputs=external_inputs,
            split_frames=split_frames,
            semantics_payload=semantics["payload"],
            task_payload=task_config["payload"],
            support=support,
            warnings=warnings,
        )
        report_path = _write_sample_preparation_markdown(
            run_dir,
            modeling_sample_spec_path=spec_path,
            split_plan_path=split_plan_path,
            semantics_path=semantics_path,
            task_config_path=task_config_path,
        )
        return _result(
            run_dir=run_dir,
            phase=phase,
            status="success",
            summary="Post-split validation succeeded; modeling_sample_spec is ready.",
            input_paths=input_paths,
            outputs={
                "flow_dir": str(run_dir.resolve()),
                "modeling_sample_spec_path": str(spec_path.resolve()),
                "modeling_sample_path": dataset_paths["combined"],
                "modeling_sample_paths": dataset_paths,
                "split_row_counts": row_counts,
                "report_path": str(report_path.resolve()),
            },
            issues=issues,
            artifacts=[
                {"kind": "modeling_sample_spec", "path": str(spec_path.resolve())},
                {"kind": "modeling_sample", "path": dataset_paths["combined"]},
            ],
            progress=[
                {
                    "step": "dry_run_split",
                    "status": "success",
                    "message": "Split executed in memory.",
                },
                {
                    "step": "write_datasets",
                    "status": "success",
                    "message": "Split CSV files written.",
                },
                {
                    "step": "modeling_sample_spec",
                    "status": "success",
                    "message": "Confirmed artifact written.",
                },
            ],
            next_steps=[
                {
                    "skill": "uplift-model-sample-homogeneity-check",
                    "action": "plan_covariates",
                    "reason": (
                        "Sample preparation succeeded; treatment/control homogeneity "
                        "should be checked before feature quality analysis."
                    ),
                    "inputs": {
                        "task_config_path": str(task_config_path),
                        "modeling_sample_spec_path": str(spec_path.resolve()),
                    },
                    "requires_user_confirmation": False,
                }
            ],
            external_inputs=external_inputs,
        )
    except DataLoaderImportError as exc:
        return _dependency_failure_result(run_dir, phase, exc)
    except UnsupportedDataSourceError as exc:
        return _unsupported_result(run_dir, phase, str(exc))
    except UnsupportedInputError as exc:
        return _unsupported_result(run_dir, phase, str(exc))
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
        run_dir = _flow_folder_for_action(output_dir, action, body)
        request_path, result_path = next_action_paths(run_dir, action or "unknown")
    except ProjectLayoutError as exc:
        print(json.dumps(_layout_error_stdout(exc), sort_keys=True))
        return 0
    write_json(request_path, request)
    if request_error:
        result = _needs_input_result(run_dir, action or "unknown", request_error, ["payload"])
    elif action not in {
        "raw_sample_check",
        "confirm_artifact",
        "split_planning",
        "post_split_validation",
    }:
        result = _unsupported_result(run_dir, action or "unknown", f"Unsupported action: {action}")
    else:
        legacy_fields = _legacy_ref_rejections(body)
        if legacy_fields:
            result = _legacy_ref_result(run_dir, action, legacy_fields)
        elif action == "raw_sample_check":
            result = run_raw_sample_check(run_dir, output_dir, body)
        elif action == "confirm_artifact":
            result = confirm_artifact(run_dir, output_dir, body)
        elif action == "split_planning":
            result = run_split_planning(run_dir, output_dir, body)
        else:
            result = run_post_split_validation(run_dir, output_dir, body)
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
    if action and action != "raw_sample_check":
        raise ProjectLayoutError("FLOW_DIR_REQUIRED", "flow_dir is required after raw_sample_check.")
    return ensure_skill_call_dir(output_dir, SKILL_NAME, "sample_preparation")


def _load_source_table(
    task_config: dict[str, Any], payload: dict[str, Any], *, output_dir: Path
) -> tuple[Any, dict[str, Any]]:
    source = _data_source_from_task(task_config, payload)
    metadata = inspect_data_source(source, base_dir=output_dir)
    if not metadata.get("supported"):
        raise UnsupportedInputError(str(metadata.get("message") or "Unsupported data source."))
    dataframe = load_table(source, base_dir=output_dir)
    return dataframe, {"data_source": metadata}


def _data_source_from_task(task_config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any] | str:
    if payload.get("data_source"):
        return payload["data_source"]
    external = task_config.get("external_inputs") or {}
    if external.get("data_source"):
        return external["data_source"]
    task_payload = task_config.get("payload") or {}
    if task_payload.get("data_ref"):
        return str(task_payload["data_ref"])
    raise MissingInputError("task_config does not contain data_ref or data_source.")


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


def _resolve_input_path(output_dir: Path, value: str) -> Path:
    return resolve_run_path(output_dir, value)


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
    external_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = {
        "artifact_kind": artifact_kind,
        "artifact_status": artifact_status,
        "artifact_version": artifact_version,
        "source_type": source_type,
        "input_paths": input_paths,
        "created_by": SKILL_CREATED_BY,
        "created_at": _now(),
        "confirmed_by": confirmed_by,
        "confirmed_at": confirmed_at,
        "source_draft_path": source_draft_path,
        "payload": payload,
        "validation": validation,
    }
    if external_inputs:
        artifact["external_inputs"] = external_inputs
    return artifact


def _all_not_null(series: Any) -> bool:
    return int(series.isna().sum()) == 0


def _apply_modeling_columns(dataframe: Any, semantics_payload: dict[str, Any]) -> Any:
    pd = _pd()
    plan = semantics_payload["plan"]
    treatment_plan = plan["treatment"]
    outcome_plan = plan["outcome"]
    prepared = dataframe.copy()
    treatment = prepared[treatment_plan["source_column"]]
    prepared[MODELING_TREATMENT_COLUMN] = (treatment == treatment_plan["treatment_value"]).astype(
        int
    )
    if outcome_plan["type"] == "binary":
        outcome = prepared[outcome_plan["source_column"]]
        prepared[MODELING_OUTCOME_COLUMN] = (outcome == outcome_plan["positive_value"]).astype(int)
    else:
        prepared[MODELING_OUTCOME_COLUMN] = pd.to_numeric(
            prepared[outcome_plan["source_column"]], errors="raise"
        )
    return prepared


def _binary_semantics(
    dataframe: Any,
    task_config: dict[str, Any],
    issues: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    outcome_col = task_config["outcome_column"]
    treatment_col = task_config["treatment_column"]
    treatment_value = task_config["treatment_value"]
    control_value = task_config["control_value"]
    outcome = dataframe[outcome_col]
    treatment = dataframe[treatment_col]
    observed_values = sorted(
        [_json_value(value) for value in outcome.dropna().unique()], key=lambda value: str(value)
    )
    positive_value = task_config.get("positive_value")
    negative_value = task_config.get("negative_value")
    assumptions: list[str] = []
    if positive_value is None and negative_value is None and set(observed_values) == {0, 1}:
        positive_value = 1
        negative_value = 0
        assumptions.append(
            "binary 0/1 outcome mapping was suggested from observed values and needs user confirmation"
        )
    if positive_value is None or negative_value is None:
        issues.append(
            _issue(
                code="BINARY_OUTCOME_MAPPING_MISSING",
                level="critical",
                blocking=True,
                message=(
                    "Binary outcome requires explicit positive_value and negative_value, "
                    "or observed {0, 1} values for a draft suggestion."
                ),
                suggested_fix="Confirm binary outcome encoding or provide positive_value and negative_value.",
            )
        )
        positive_value = 1
        negative_value = 0
    invalid_outcomes = outcome.dropna()[~outcome.dropna().isin([positive_value, negative_value])]
    if len(invalid_outcomes) > 0:
        issues.append(
            _issue(
                code="BINARY_OUTCOME_VALUES_OUT_OF_MAPPING",
                level="critical",
                blocking=True,
                message="Observed non-null outcome values are not fully covered by mapping.",
                suggested_fix="Fix outcome encoding or confirm the correct positive/negative mapping.",
            )
        )
    treated_mask = treatment == treatment_value
    control_mask = treatment == control_value
    positive_mask = outcome == positive_value
    negative_mask = outcome == negative_value
    support = {
        "treated_rows": int(treated_mask.sum()),
        "control_rows": int(control_mask.sum()),
        "treated_positive_rows": int((treated_mask & positive_mask).sum()),
        "treated_negative_rows": int((treated_mask & negative_mask).sum()),
        "control_positive_rows": int((control_mask & positive_mask).sum()),
        "control_negative_rows": int((control_mask & negative_mask).sum()),
    }
    if support["treated_rows"] <= 0 or support["control_rows"] <= 0:
        issues.append(
            _issue(
                code="INSUFFICIENT_TREATMENT_CONTROL_SUPPORT",
                level="critical",
                blocking=True,
                message="Treatment and control groups must both have more than 0 rows.",
                suggested_fix="Check treatment/control values or expand the sample window.",
            )
        )
    missing_cells = [
        key
        for key, value in support.items()
        if key.endswith("_rows") and key not in {"treated_rows", "control_rows"} and value <= 0
    ]
    if missing_cells:
        issues.append(
            _issue(
                code="INSUFFICIENT_GROUP_OUTCOME_SUPPORT",
                level="critical",
                blocking=True,
                message=(
                    "Treatment/control groups must each contain positive and negative outcomes. "
                    f"Missing support: {missing_cells}"
                ),
                suggested_fix="Check outcome encoding or expand the sample window.",
            )
        )
    plan = {
        "outcome": {
            "source_column": outcome_col,
            "type": "binary",
            "positive_value": _json_value(positive_value),
            "negative_value": _json_value(negative_value),
            "modeling_column": MODELING_OUTCOME_COLUMN,
        },
        "treatment": {
            "source_column": treatment_col,
            "treatment_value": _json_value(treatment_value),
            "control_value": _json_value(control_value),
            "modeling_column": MODELING_TREATMENT_COLUMN,
        },
        "policies": {"missing_or_invalid_values": "error"},
        "forced_exclude_columns": _forced_exclude_columns(task_config),
    }
    evidence = {
        "outcome_values": _value_counts(outcome),
        "treatment_values": _value_counts(treatment),
        "support": support,
    }
    return plan, evidence, assumptions


def _build_data_semantics_payload(
    dataframe: Any, task_config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outcome_type = task_config.get("outcome_type")
    if outcome_type not in {"binary", "continuous"}:
        raise UnsupportedInputError(
            "uplift-model-sample-preparation only supports binary and continuous outcome_type."
        )
    outcome_col = task_config.get("outcome_column")
    treatment_col = task_config.get("treatment_column")
    required_columns = [column for column in [outcome_col, treatment_col] if column]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    issues: list[dict[str, Any]] = []
    warnings: list[str] = []
    if missing_columns:
        issues.append(
            _issue(
                code="SAMPLE_REQUIRED_COLUMNS_MISSING",
                level="critical",
                blocking=True,
                message=f"Required columns missing from dataset: {missing_columns}",
                suggested_fix="Confirm task_config field names or provide a dataset with required columns.",
            )
        )
        return {
            "plan": {},
            "evidence": {
                "row_count": int(len(dataframe)),
                "missing_required_columns": missing_columns,
                "null_counts": {},
            },
            "is_valid": False,
            "warnings": warnings,
        }, issues
    assert outcome_col is not None
    assert treatment_col is not None
    null_counts = {
        outcome_col: int(dataframe[outcome_col].isna().sum()),
        treatment_col: int(dataframe[treatment_col].isna().sum()),
    }
    for column, count in null_counts.items():
        if count > 0:
            issues.append(
                _issue(
                    code="KEY_FIELD_NULL_VALUES",
                    level="critical",
                    blocking=True,
                    message=f"Key field {column} contains {count} null values.",
                    suggested_fix="Provide a clean dataset or confirm a missing-value policy.",
                )
            )
    treatment = dataframe[treatment_col]
    valid_treatment_mask = treatment.isin([task_config["treatment_value"], task_config["control_value"]])
    if not bool(valid_treatment_mask.all()):
        issues.append(
            _issue(
                code="TREATMENT_VALUES_OUT_OF_MAPPING",
                level="critical",
                blocking=True,
                message="Treatment column contains values outside treatment_value/control_value.",
                suggested_fix="Check treatment/control mapping or clean unsupported treatment values.",
            )
        )
    if outcome_type == "binary":
        plan, evidence_extra, assumptions = _binary_semantics(dataframe, task_config, issues)
    else:
        plan, evidence_extra = _continuous_semantics(dataframe, task_config, issues, warnings)
        assumptions = []
    evidence = {
        "row_count": int(len(dataframe)),
        "missing_required_columns": missing_columns,
        "null_counts": null_counts,
        **evidence_extra,
    }
    return {
        "plan": plan,
        "evidence": evidence,
        "is_valid": not any(issue["blocking"] for issue in issues),
        "warnings": warnings,
        "_validation_assumptions": assumptions,
    }, issues


def _build_split_plan_payload(
    dataframe: Any,
    task_config: dict[str, Any],
    *,
    split_method: str,
    ratios: dict[str, float] | None,
    time_column: str | None,
    oot_window: dict[str, Any] | None,
    non_oot_random_ratios: dict[str, float] | None,
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pd = _pd()
    issues: list[dict[str, Any]] = []
    warnings: list[str] = []
    if split_method not in {"random", "time"}:
        raise UnsupportedInputError("split_method only supports random and time.")
    if split_method == "random":
        checked_ratios = _checked_ratios(
            ratios or {"train": 0.8, "test": 0.2, "valid": 0.0}, "ratios"
        )
        counts = _allocated_counts(len(dataframe), checked_ratios)
        plan = {
            "split_method": "random",
            "random_seed": random_seed,
            "ratios": checked_ratios,
            "stratification": {"enabled": False, "columns": []},
        }
        evidence = {
            "row_count": int(len(dataframe)),
            "estimated_rows": {**counts, "oot": 0},
            "recommended_reason": (
                "Random split is used; stratification is disabled in the first version."
            ),
        }
    else:
        selected_time_column = time_column or task_config.get("time_column")
        if not selected_time_column:
            issues.append(
                _issue(
                    code="TIME_COLUMN_REQUIRED",
                    level="critical",
                    blocking=True,
                    message="time split requires time_column.",
                    suggested_fix="Provide time_column or use split_method=random.",
                )
            )
            selected_time_column = ""
        elif selected_time_column not in dataframe.columns:
            issues.append(
                _issue(
                    code="TIME_COLUMN_MISSING",
                    level="critical",
                    blocking=True,
                    message=f"time_column does not exist in dataset: {selected_time_column}",
                    suggested_fix="Confirm time_column or provide a dataset with the column.",
                )
            )
        window = oot_window or {}
        start = window.get("start")
        end_exclusive = window.get("end_exclusive")
        if not start:
            issues.append(
                _issue(
                    code="OOT_WINDOW_START_REQUIRED",
                    level="critical",
                    blocking=True,
                    message="time split requires oot_window.start.",
                    suggested_fix="Provide oot_window.start.",
                )
            )
        checked_ratios = _checked_ratios(
            non_oot_random_ratios or {"train": 0.8, "test": 0.2, "valid": 0.0},
            "non_oot_random_ratios",
        )
        estimated_rows = {"train": 0, "test": 0, "valid": 0, "oot": 0}
        semantics = "time_column >= start"
        if selected_time_column and selected_time_column in dataframe.columns and start:
            try:
                parsed = pd.to_datetime(dataframe[selected_time_column], errors="raise")
                start_ts = pd.to_datetime(start, errors="raise")
                if end_exclusive is not None:
                    end_ts = pd.to_datetime(end_exclusive, errors="raise")
                    oot_mask = (parsed >= start_ts) & (parsed < end_ts)
                    semantics = "start <= time_column < end_exclusive"
                else:
                    oot_mask = parsed >= start_ts
                oot_rows = int(oot_mask.sum())
                if oot_rows == 0:
                    issues.append(
                        _issue(
                            code="OOT_SPLIT_EMPTY",
                            level="critical",
                            blocking=True,
                            message="OOT split is empty for the requested window.",
                            suggested_fix="Choose an OOT window that contains rows.",
                        )
                    )
                non_oot_counts = _allocated_counts(int((~oot_mask).sum()), checked_ratios)
                estimated_rows = {**non_oot_counts, "oot": oot_rows}
            except Exception as exc:  # noqa: BLE001
                issues.append(
                    _issue(
                        code="TIME_COLUMN_PARSE_FAILED",
                        level="critical",
                        blocking=True,
                        message=f"time_column strict datetime parsing failed: {exc}",
                        suggested_fix="Clean time_column values or choose another time column.",
                    )
                )
        plan = {
            "split_method": "time",
            "time_column": selected_time_column,
            "oot_window": {"start": start, "end_exclusive": end_exclusive},
            "random_seed": random_seed,
            "non_oot_random_ratios": checked_ratios,
            "stratification": {"enabled": False, "columns": []},
        }
        evidence = {
            "row_count": int(len(dataframe)),
            "time_column": selected_time_column,
            "oot_window_semantics": semantics,
            "estimated_rows": estimated_rows,
            "recommended_reason": (
                "Rows in the OOT window are reserved first; non-OOT rows are randomly split. "
                "Stratification is disabled in the first version."
            ),
        }
    return {
        "plan": plan,
        "evidence": evidence,
        "is_valid": not any(issue["blocking"] for issue in issues),
        "warnings": warnings,
    }, issues


def _checked_ratios(ratios: dict[str, float], field_name: str) -> dict[str, float]:
    checked = {name: float(ratios.get(name, 0.0)) for name in ("train", "test", "valid")}
    if any(value < 0 for value in checked.values()):
        raise UnsupportedInputError(f"{field_name} cannot contain negative values.")
    if not math.isclose(sum(checked.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise UnsupportedInputError(f"{field_name}.train + .test + .valid must equal 1.")
    return checked


def _continuous_semantics(
    dataframe: Any,
    task_config: dict[str, Any],
    issues: list[dict[str, Any]],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pd = _pd()
    outcome_col = task_config["outcome_column"]
    treatment_col = task_config["treatment_column"]
    outcome = dataframe[outcome_col]
    treatment = dataframe[treatment_col]
    try:
        numeric = pd.to_numeric(outcome, errors="raise")
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                code="CONTINUOUS_OUTCOME_NON_NUMERIC",
                level="critical",
                blocking=True,
                message=f"Continuous outcome must be strictly numeric: {exc}",
                suggested_fix="Clean non-numeric outcome values or choose the correct outcome column.",
            )
        )
        numeric = pd.to_numeric(outcome, errors="coerce")
    if not _all_not_null(outcome):
        issues.append(
            _issue(
                code="CONTINUOUS_OUTCOME_MISSING",
                level="critical",
                blocking=True,
                message="Continuous outcome contains missing values.",
                suggested_fix="Provide complete outcome values or define a missing-value policy.",
            )
        )
    treatment_value = task_config["treatment_value"]
    control_value = task_config["control_value"]
    treated_mask = treatment == treatment_value
    control_mask = treatment == control_value
    valid_mask = numeric.notna()
    support = {
        "treated_rows": int(treated_mask.sum()),
        "control_rows": int(control_mask.sum()),
        "treated_valid_outcome_rows": int((treated_mask & valid_mask).sum()),
        "control_valid_outcome_rows": int((control_mask & valid_mask).sum()),
    }
    if support["treated_valid_outcome_rows"] <= 0 or support["control_valid_outcome_rows"] <= 0:
        issues.append(
            _issue(
                code="INSUFFICIENT_CONTINUOUS_GROUP_SUPPORT",
                level="critical",
                blocking=True,
                message="Treatment and control groups must both have valid continuous outcome rows.",
                suggested_fix="Check treatment/control mapping or provide a larger clean sample.",
            )
        )
    clean = numeric.dropna()
    overall_std = _std(clean)
    if overall_std <= 0:
        issues.append(
            _issue(
                code="CONTINUOUS_OUTCOME_ZERO_VARIANCE",
                level="critical",
                blocking=True,
                message="Overall continuous outcome standard deviation must be greater than 0.",
                suggested_fix="Use a non-constant continuous outcome or expand the sample window.",
            )
        )
    for group_name, mask in [("treated", treated_mask), ("control", control_mask)]:
        group_std = _std(numeric.loc[mask & valid_mask])
        if group_std == 0:
            warnings.append(f"{group_name} outcome std is 0")
    plan = {
        "outcome": {
            "source_column": outcome_col,
            "type": "continuous",
            "modeling_column": MODELING_OUTCOME_COLUMN,
        },
        "treatment": {
            "source_column": treatment_col,
            "treatment_value": _json_value(treatment_value),
            "control_value": _json_value(control_value),
            "modeling_column": MODELING_TREATMENT_COLUMN,
        },
        "policies": {"missing_or_invalid_values": "error"},
        "forced_exclude_columns": _forced_exclude_columns(task_config),
    }
    evidence = {
        "outcome_numeric_summary": _numeric_summary(numeric),
        "treatment_values": _value_counts(treatment),
        "support": support,
    }
    return plan, evidence


def _allocated_counts(row_count: int, ratios: dict[str, float]) -> dict[str, int]:
    valid = int(math.floor(row_count * ratios["valid"]))
    test = int(math.floor(row_count * ratios["test"]))
    train = row_count - test - valid
    return {"train": train, "test": test, "valid": valid}


def _forced_exclude_columns(task_config: dict[str, Any]) -> list[str]:
    ordered: list[str] = [
        task_config["treatment_column"],
        task_config["outcome_column"],
        MODELING_TREATMENT_COLUMN,
        MODELING_OUTCOME_COLUMN,
        MODELING_SPLIT_COLUMN,
    ]
    for optional in ("unit_id_column", "time_column"):
        if task_config.get(optional):
            ordered.append(task_config[optional])
    ordered.extend(task_config.get("exclude_columns") or [])
    seen: set[str] = set()
    result: list[str] = []
    for column in ordered:
        if column and column not in seen:
            result.append(column)
            seen.add(column)
    return result


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


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _lineage_issues(
    output_dir: Path,
    task_config_path: Path,
    semantics_path: Path,
    split_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    input_paths = split_plan.get("input_paths") or {}
    if not _same_path(input_paths.get("task_config_path"), task_config_path, output_dir):
        issues.append(
            _issue(
                code="STALE_SPLIT_PLAN_TASK_CONFIG_PATH",
                level="critical",
                blocking=True,
                message="split_plan was generated from a different task_config_path.",
                suggested_fix="Regenerate split_plan from the selected task_config_path.",
            )
        )
    if not _same_path(input_paths.get("data_semantics_plan_path"), semantics_path, output_dir):
        issues.append(
            _issue(
                code="STALE_SPLIT_PLAN_DATA_SEMANTICS_PATH",
                level="critical",
                blocking=True,
                message="split_plan was generated from a different data_semantics_plan_path.",
                suggested_fix="Regenerate split_plan from the selected data_semantics_plan_path.",
            )
        )
    return issues


def _same_path(value: Any, path: Path, run_dir: Path | None = None) -> bool:
    if not value:
        return False
    try:
        left_path = resolve_run_path(run_dir, value) if run_dir else Path(str(value)).expanduser().resolve()
        right_path = resolve_run_path(run_dir, path) if run_dir else path.resolve()
        return left_path == right_path
    except Exception:  # noqa: BLE001
        return False


def _numeric_summary(series: Any) -> dict[str, Any]:
    pd = _pd()
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "valid_count": int(len(clean)),
        "mean": None if clean.empty else float(clean.mean()),
        "std": None if clean.empty else _std(clean),
        "min": None if clean.empty else float(clean.min()),
        "max": None if clean.empty else float(clean.max()),
    }


def _split_data(dataframe: Any, split_plan_payload: dict[str, Any]) -> dict[str, Any | None]:
    pd = _pd()
    plan = split_plan_payload["plan"]
    method = plan["split_method"]
    if method == "random":
        frames = _split_random(dataframe, plan["ratios"], int(plan["random_seed"]))
        frames["oot"] = None
        return frames
    parsed = pd.to_datetime(dataframe[plan["time_column"]], errors="raise")
    start = pd.to_datetime(plan["oot_window"]["start"], errors="raise")
    end_exclusive = plan["oot_window"].get("end_exclusive")
    if end_exclusive is None:
        oot_mask = parsed >= start
    else:
        end = pd.to_datetime(end_exclusive, errors="raise")
        oot_mask = (parsed >= start) & (parsed < end)
    oot = dataframe.loc[oot_mask].copy()
    non_oot = dataframe.loc[~oot_mask].copy()
    frames = _split_random(non_oot, plan["non_oot_random_ratios"], int(plan["random_seed"]))
    frames["oot"] = oot
    return frames


def _split_random(dataframe: Any, ratios: dict[str, float], random_seed: int) -> dict[str, Any | None]:
    shuffled = dataframe.sample(frac=1.0, random_state=random_seed)
    counts = _allocated_counts(len(shuffled), ratios)
    train_end = counts["train"]
    test_end = train_end + counts["test"]
    train = shuffled.iloc[:train_end].sort_index().copy()
    test = shuffled.iloc[train_end:test_end].sort_index().copy()
    valid = shuffled.iloc[test_end:].sort_index().copy() if ratios["valid"] > 0 else None
    return {"train": train, "test": test, "valid": valid}


def _std(series: Any) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.astype(float).std(ddof=0))


def _safe_mean_from_counts(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _observed_outcome_summary(
    treated_mean: float | None,
    control_mean: float | None,
    warnings: list[str],
) -> dict[str, float | None]:
    difference = None
    relative_difference = None
    if treated_mean is not None and control_mean is not None:
        difference = float(treated_mean - control_mean)
        if control_mean == 0:
            warnings.append(
                "observed outcome relative difference is unavailable because control mean outcome is 0"
            )
        else:
            relative_difference = float(treated_mean / control_mean - 1)
    return {
        "treated_mean_outcome": treated_mean,
        "control_mean_outcome": control_mean,
        "observed_outcome_difference": difference,
        "observed_outcome_relative_difference": relative_difference,
    }


def _support_for_frame(frame: Any, semantics_payload: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    outcome_type = semantics_payload["plan"]["outcome"]["type"]
    treatment = frame[MODELING_TREATMENT_COLUMN]
    outcome = frame[MODELING_OUTCOME_COLUMN]
    treated_mask = treatment == 1
    control_mask = treatment == 0
    blocking_reasons: list[str] = []
    warnings: list[str] = []
    if outcome_type == "binary":
        positive_mask = outcome == 1
        negative_mask = outcome == 0
        treated_rows = int(treated_mask.sum())
        control_rows = int(control_mask.sum())
        treated_positive_rows = int((treated_mask & positive_mask).sum())
        control_positive_rows = int((control_mask & positive_mask).sum())
        treated_mean = _safe_mean_from_counts(treated_positive_rows, treated_rows)
        control_mean = _safe_mean_from_counts(control_positive_rows, control_rows)
        support = {
            "row_count": int(len(frame)),
            "treated_rows": treated_rows,
            "control_rows": control_rows,
            "treated_positive_rows": treated_positive_rows,
            "treated_negative_rows": int((treated_mask & negative_mask).sum()),
            "control_positive_rows": control_positive_rows,
            "control_negative_rows": int((control_mask & negative_mask).sum()),
            **_observed_outcome_summary(treated_mean, control_mean, warnings),
        }
        for key in [
            "treated_rows",
            "control_rows",
            "treated_positive_rows",
            "treated_negative_rows",
            "control_positive_rows",
            "control_negative_rows",
        ]:
            if support[key] <= 0:
                blocking_reasons.append(f"{key} is 0")
        return support, blocking_reasons, warnings
    valid_outcome = outcome.notna()
    treated_values = outcome.loc[treated_mask & valid_outcome]
    control_values = outcome.loc[control_mask & valid_outcome]
    overall_values = outcome.loc[valid_outcome]
    support = {
        "row_count": int(len(frame)),
        "treated_rows": int(treated_mask.sum()),
        "control_rows": int(control_mask.sum()),
        "valid_outcome_rows": int(valid_outcome.sum()),
        "overall_mean": None if overall_values.empty else float(overall_values.mean()),
        "overall_std": _std(overall_values),
        "overall_min": None if overall_values.empty else float(overall_values.min()),
        "overall_max": None if overall_values.empty else float(overall_values.max()),
        "treated_valid_outcome_rows": int(len(treated_values)),
        "control_valid_outcome_rows": int(len(control_values)),
        "treated_mean_outcome": None if treated_values.empty else float(treated_values.mean()),
        "control_mean_outcome": None if control_values.empty else float(control_values.mean()),
        "treated_std": _std(treated_values),
        "control_std": _std(control_values),
    }
    support.update(
        _observed_outcome_summary(
            support["treated_mean_outcome"],
            support["control_mean_outcome"],
            warnings,
        )
    )
    if support["treated_valid_outcome_rows"] <= 0:
        blocking_reasons.append("treated_valid_outcome_rows is 0")
    if support["control_valid_outcome_rows"] <= 0:
        blocking_reasons.append("control_valid_outcome_rows is 0")
    if support["overall_std"] <= 0:
        blocking_reasons.append("overall outcome std is 0")
    if support["treated_std"] == 0:
        warnings.append("treated outcome std is 0")
    if support["control_std"] == 0:
        warnings.append("control outcome std is 0")
    return support, blocking_reasons, warnings


def _validate_split_frames(
    split_frames: dict[str, Any | None],
    semantics_payload: dict[str, Any],
    task_payload: dict[str, Any],
) -> tuple[dict[str, Any | None], list[dict[str, Any]], list[str]]:
    support: dict[str, Any | None] = {}
    issues: list[dict[str, Any]] = []
    warnings: list[str] = []
    for name in SPLIT_NAMES:
        frame = split_frames.get(name)
        if frame is None:
            support[name] = None
            continue
        if len(frame) == 0:
            support[name] = {"row_count": 0}
            issues.append(
                _issue(
                    code="SPLIT_EMPTY",
                    level="critical",
                    blocking=True,
                    message=f"{name} split is enabled but has 0 rows.",
                    suggested_fix="Revise split ratios or OOT window.",
                )
            )
            continue
        split_support, blocking_reasons, split_warnings = _support_for_frame(frame, semantics_payload)
        support[name] = split_support
        warnings.extend([f"{name}: {warning}" for warning in split_warnings])
        for reason in blocking_reasons:
            issues.append(
                _issue(
                    code="SPLIT_SUPPORT_BLOCKING",
                    level="critical",
                    blocking=True,
                    message=f"{name} split support failed: {reason}.",
                    suggested_fix=(
                        "Revise split_plan so every enabled split has enough "
                        "treatment/control and outcome support."
                    ),
                )
            )
    train_frame = split_frames.get("train")
    if train_frame is not None and len(train_frame) > 0:
        forced_exclude = _forced_exclude_columns(task_payload)
        candidate_columns = [column for column in train_frame.columns if column not in set(forced_exclude)]
        if len(candidate_columns) == 0:
            issues.append(
                _issue(
                    code="NO_CANDIDATE_FEATURE_COLUMNS",
                    level="critical",
                    blocking=True,
                    message="No candidate feature columns remain after forced exclusions.",
                    suggested_fix="Provide feature columns or reduce exclude_columns.",
                )
            )
        elif len(candidate_columns) < 2:
            warnings.append("candidate feature count is less than 2")
    return support, issues, warnings


def _value_counts(series: Any) -> dict[str, int]:
    pd = _pd()
    counts = series.value_counts(dropna=False)
    result: dict[str, int] = {}
    for value, count in counts.items():
        key = "null" if pd.isna(value) else str(_json_value(value))
        result[key] = int(count)
    return result


def _write_data_semantics_plan_draft(
    run_dir: Path,
    *,
    input_paths: dict[str, Any],
    external_inputs: dict[str, Any] | None,
    payload: dict[str, Any],
) -> Path:
    version = 1
    artifact_path = run_dir / "artifacts" / "data_semantics_plan.draft.v1.json"
    validation_assumptions = payload.pop("_validation_assumptions", [])
    artifact = _artifact_envelope(
        artifact_kind="data_semantics_plan",
        artifact_status="draft",
        artifact_version=version,
        source_type="generated_by_skill",
        input_paths=input_paths,
        payload=payload,
        validation={
            "filled_fields": ["plan", "evidence", "is_valid", "warnings"],
            "missing_fields": [],
            "assumptions": validation_assumptions,
            "warnings": payload["warnings"],
        },
        confirmed_by=None,
        confirmed_at=None,
        source_draft_path=None,
        external_inputs=external_inputs,
    )
    write_json(artifact_path, artifact)
    return artifact_path


def _write_split_plan_draft(
    run_dir: Path,
    *,
    input_paths: dict[str, Any],
    external_inputs: dict[str, Any] | None,
    payload: dict[str, Any],
) -> Path:
    version = 1
    artifact_path = run_dir / "artifacts" / "split_plan.draft.v1.json"
    artifact = _artifact_envelope(
        artifact_kind="split_plan",
        artifact_status="draft",
        artifact_version=version,
        source_type="generated_by_skill",
        input_paths=input_paths,
        payload=payload,
        validation={
            "filled_fields": ["plan", "evidence", "is_valid", "warnings"],
            "missing_fields": [],
            "assumptions": ["stratification is disabled in the first version"],
            "warnings": payload["warnings"],
        },
        confirmed_by=None,
        confirmed_at=None,
        source_draft_path=None,
        external_inputs=external_inputs,
    )
    write_json(artifact_path, artifact)
    return artifact_path


def _write_modeling_sample_spec(
    run_dir: Path,
    *,
    input_paths: dict[str, Any],
    external_inputs: dict[str, Any] | None,
    split_frames: dict[str, Any | None],
    semantics_payload: dict[str, Any],
    task_payload: dict[str, Any],
    support: dict[str, Any | None],
    warnings: list[str],
) -> tuple[Path, dict[str, Any]]:
    pd = _pd()
    version = 1
    dataset_paths: dict[str, Any] = {}
    combined_frames = []
    for name in SPLIT_NAMES:
        frame = split_frames.get(name)
        if frame is None:
            dataset_paths[name] = None
            continue
        dataset_path = (run_dir / "artifacts" / f"modeling_sample.{name}.v{version}.csv").resolve()
        frame.to_csv(dataset_path, index=False)
        dataset_paths[name] = str(dataset_path)
        tagged = frame.copy()
        tagged[MODELING_SPLIT_COLUMN] = name
        combined_frames.append(tagged)
    combined_path = (run_dir / "artifacts" / f"modeling_sample.v{version}.csv").resolve()
    pd.concat(combined_frames, ignore_index=True).to_csv(combined_path, index=False)
    dataset_paths["combined"] = str(combined_path)
    train_frame = split_frames["train"]
    assert train_frame is not None
    forced_exclude = _forced_exclude_columns(task_payload)
    candidate_preview = [column for column in train_frame.columns if column not in set(forced_exclude)]
    payload = {
        "datasets": dataset_paths,
        "columns": {
            "treatment": MODELING_TREATMENT_COLUMN,
            "outcome": MODELING_OUTCOME_COLUMN,
            "split": MODELING_SPLIT_COLUMN,
            "source_treatment": semantics_payload["plan"]["treatment"]["source_column"],
            "source_outcome": semantics_payload["plan"]["outcome"]["source_column"],
            "outcome_type": semantics_payload["plan"]["outcome"]["type"],
        },
        "feature_columns": {
            "forced_exclude": forced_exclude,
            "candidate_count": len(candidate_preview),
            "candidate_preview": candidate_preview[:20],
        },
        "support": support,
        "is_valid": True,
        "warnings": warnings,
    }
    artifact_path = run_dir / "artifacts" / "modeling_sample_spec.confirmed.v1.json"
    artifact = _artifact_envelope(
        artifact_kind="modeling_sample_spec",
        artifact_status="confirmed",
        artifact_version=version,
        source_type="generated_by_skill",
        input_paths=input_paths,
        payload=payload,
        validation={
            "filled_fields": [
                "datasets",
                "columns",
                "feature_columns",
                "support",
                "is_valid",
                "warnings",
            ],
            "missing_fields": [],
            "assumptions": [],
            "warnings": warnings,
        },
        confirmed_by="system",
        confirmed_at=_now(),
        source_draft_path=None,
        external_inputs=external_inputs,
    )
    write_json(artifact_path, artifact)
    return artifact_path, dataset_paths


def _confirmation_next_steps(
    run_dir: Path, artifact_kind: str, artifact_path: Path, confirmed: dict[str, Any]
) -> list[dict[str, Any]]:
    if artifact_kind == "data_semantics_plan":
        task_config_path = confirmed.get("input_paths", {}).get("task_config_path")
        return [
            {
                "skill": SKILL_NAME,
                "action": "split_planning",
                "reason": "data_semantics_plan is confirmed; split_planning can run.",
                "inputs": {
                    "flow_dir": str(run_dir.resolve()),
                    "task_config_path": task_config_path,
                    "data_semantics_plan_path": str(artifact_path.resolve()),
                },
                "requires_user_confirmation": False,
            }
        ]
    input_paths = confirmed.get("input_paths", {})
    return [
        {
            "skill": SKILL_NAME,
            "action": "post_split_validation",
            "reason": "split_plan is confirmed; post-split validation can run.",
            "inputs": {
                "flow_dir": str(run_dir.resolve()),
                "task_config_path": input_paths.get("task_config_path"),
                "data_semantics_plan_path": input_paths.get("data_semantics_plan_path"),
                "split_plan_path": str(artifact_path.resolve()),
            },
            "requires_user_confirmation": False,
        }
    ]


def _write_sample_preparation_markdown(
    run_dir: Path,
    *,
    modeling_sample_spec_path: Path,
    split_plan_path: Path,
    semantics_path: Path,
    task_config_path: Path,
) -> Path:
    spec = read_json(modeling_sample_spec_path)
    split_plan = read_json(split_plan_path)
    semantics = read_json(semantics_path)
    task_config = read_json(task_config_path)
    payload = spec["payload"]
    task_payload = task_config["payload"]
    language = str(task_payload.get("report_preferences", {}).get("language") or "zh-CN")
    version = int(spec["artifact_version"])
    report_path = run_dir / "report.md"
    if is_zh(language):
        lines = [
            "# 样本准备",
            "",
            f"状态：{spec['artifact_status']}",
            f"版本：{version}",
            f"报告语言：`{language}`",
            "",
            "## 数据语义",
            "",
            *_data_semantics_lines(semantics["payload"], language),
            "",
            "## 切分方案",
            "",
            *_split_plan_lines(split_plan["payload"], language),
            "",
            "## 已准备数据集",
            "",
            *_dataset_lines(payload, language),
            "",
            "## 样本支撑",
            "",
            "以下数值是观测到的 outcome 描述性汇总，表示 treatment/control 组之间的总体 outcome 绝对差异和相对差异；它们不是 uplift 模型 LIFT，也不是因果效应估计。",
            "",
            *_support_lines(payload, language),
            "",
            *_observed_difference_lines(payload, language),
            "",
            "## 候选特征空间",
            "",
            "这里仅表示样本准备后的候选特征空间。最终特征选择由 `uplift-model-feature-quality-analysis` 负责。",
            "",
            *_feature_candidate_lines(payload, language),
            "",
            "## 警告与假设",
            "",
            *_warning_lines(language, spec, split_plan, semantics),
            "",
            "## 下一步",
            "",
            "继续执行 `uplift-model-sample-homogeneity-check`。",
            "",
            "## 附录：系统引用",
            "",
            "上游配置：",
            f"- task_config_path: `{task_config_path.resolve()}`",
            f"- data_semantics_plan_path: `{semantics_path.resolve()}`",
            f"- split_plan_path: `{split_plan_path.resolve()}`",
            "",
            "主源与结果：",
            f"- modeling_sample_spec_path: `{modeling_sample_spec_path.resolve()}`",
            f"- result_path: `{(run_dir / 'results' / 'post_split_validation.result.json').resolve()}`",
            "",
            "数据集路径：",
            *_dataset_path_lines(payload, language),
            "",
        ]
    else:
        lines = [
            "# Sample Preparation",
            "",
            f"Status: {spec['artifact_status']}",
            f"Version: {version}",
            f"Report language: `{language}`",
            "",
            "## Data Semantics",
            "",
            *_data_semantics_lines(semantics["payload"], language),
            "",
            "## Split Plan",
            "",
            *_split_plan_lines(split_plan["payload"], language),
            "",
            "## Prepared Datasets",
            "",
            *_dataset_lines(payload, language),
            "",
            "## Sample Support",
            "",
            "The following values are descriptive observed outcome summaries of treatment/control group outcome differences; they are not model LIFT or causal estimates.",
            "",
            *_support_lines(payload, language),
            "",
            *_observed_difference_lines(payload, language),
            "",
            "## Feature Candidate Space",
            "",
            "This is only the candidate feature space after sample preparation. Final feature selection is owned by `uplift-model-feature-quality-analysis`.",
            "",
            *_feature_candidate_lines(payload, language),
            "",
            "## Warnings and Assumptions",
            "",
            *_warning_lines(language, spec, split_plan, semantics),
            "",
            "## Next Step",
            "",
            "Continue to `uplift-model-sample-homogeneity-check`.",
            "",
            "## Appendix: System References",
            "",
            "Upstream config:",
            f"- task_config_path: `{task_config_path.resolve()}`",
            f"- data_semantics_plan_path: `{semantics_path.resolve()}`",
            f"- split_plan_path: `{split_plan_path.resolve()}`",
            "",
            "Primary source and result:",
            f"- modeling_sample_spec_path: `{modeling_sample_spec_path.resolve()}`",
            f"- result_path: `{(run_dir / 'results' / 'post_split_validation.result.json').resolve()}`",
            "",
            "Dataset paths:",
            *_dataset_path_lines(payload, language),
            "",
        ]
    write_text(report_path, "\n".join(lines))
    return report_path


def _data_semantics_lines(payload: dict[str, Any], language: str) -> list[str]:
    plan = payload["plan"]
    outcome = plan["outcome"]
    treatment = plan["treatment"]
    lines = (
        [
            f"- Outcome 源字段：`{outcome['source_column']}`",
            f"- Outcome 类型：`{outcome['type']}`",
        ]
        if is_zh(language)
        else [
            f"- Outcome source column: `{outcome['source_column']}`",
            f"- Outcome type: `{outcome['type']}`",
        ]
    )
    if outcome["type"] == "binary":
        lines.extend(
            [
                f"- {'正例取值' if is_zh(language) else 'Positive value'}: `{outcome['positive_value']}`",
                f"- {'负例取值' if is_zh(language) else 'Negative value'}: `{outcome['negative_value']}`",
            ]
        )
    lines.extend(
        [
            f"- {'Treatment 源字段' if is_zh(language) else 'Treatment source column'}: `{treatment['source_column']}`",
            f"- {'Treatment 取值' if is_zh(language) else 'Treatment value'}: `{treatment['treatment_value']}`",
            f"- {'Control 取值' if is_zh(language) else 'Control value'}: `{treatment['control_value']}`",
            f"- {'建模 treatment 字段' if is_zh(language) else 'Modeling treatment column'}: `{treatment['modeling_column']}`",
            f"- {'建模 outcome 字段' if is_zh(language) else 'Modeling outcome column'}: `{outcome['modeling_column']}`",
        ]
    )
    return lines


def _split_plan_lines(payload: dict[str, Any], language: str) -> list[str]:
    plan = payload["plan"]
    lines = [f"- {'方法' if is_zh(language) else 'Method'}: `{plan['split_method']}`"]
    if plan["split_method"] == "random":
        ratios = plan["ratios"]
        lines.extend(
            [
                f"- {'训练集比例' if is_zh(language) else 'Train ratio'}: {_format_number(ratios['train'], language)}",
                f"- {'测试集比例' if is_zh(language) else 'Test ratio'}: {_format_number(ratios['test'], language)}",
                f"- {'验证集比例' if is_zh(language) else 'Valid ratio'}: {_format_number(ratios['valid'], language)}",
            ]
        )
    else:
        window = plan["oot_window"]
        ratios = plan["non_oot_random_ratios"]
        lines.extend(
            [
                f"- {'时间字段' if is_zh(language) else 'Time column'}: `{plan['time_column']}`",
                f"- OOT start: `{window['start']}`",
                f"- OOT {'结束时间（不含）' if is_zh(language) else 'end exclusive'}: {_format_optional_value(window.get('end_exclusive'), language)}",
                f"- Non-OOT {'训练集比例' if is_zh(language) else 'train ratio'}: {_format_number(ratios['train'], language)}",
                f"- Non-OOT {'测试集比例' if is_zh(language) else 'test ratio'}: {_format_number(ratios['test'], language)}",
                f"- Non-OOT {'验证集比例' if is_zh(language) else 'valid ratio'}: {_format_number(ratios['valid'], language)}",
            ]
        )
    lines.append(f"- {'随机种子' if is_zh(language) else 'Random seed'}: `{plan['random_seed']}`")
    lines.append("- 分层：未启用" if is_zh(language) else "- Stratification: disabled")
    return lines


def _dataset_lines(payload: dict[str, Any], language: str) -> list[str]:
    lines = (
        ["| 切分 | 数据集 | 行数 |", "| --- | --- | ---: |"]
        if is_zh(language)
        else ["| Split | Dataset | Rows |", "| --- | --- | ---: |"]
    )
    datasets = payload["datasets"]
    support = payload["support"]
    for split in DISPLAY_SPLIT_ORDER:
        dataset = datasets.get(split)
        split_support = support.get(split)
        rows = "" if split_support is None else str(split_support.get("row_count", ""))
        lines.append(f"| {split} | {_format_dataset_name(dataset, language)} | {rows} |")
    return lines


def _support_lines(payload: dict[str, Any], language: str) -> list[str]:
    if payload["columns"]["outcome_type"] == "binary":
        return _binary_support_lines(payload["support"], language)
    return _continuous_support_lines(payload["support"], language)


def _binary_support_lines(support: dict[str, Any], language: str) -> list[str]:
    rows = (
        [
            "| 切分 | 组别 | 行数 | 正例 | 负例 | 正例率 | 负例率 | 平均 Y |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        if is_zh(language)
        else [
            "| Split | Group | Rows | Positive | Negative | Positive Rate | Negative Rate | Mean Y |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in DISPLAY_SPLIT_ORDER:
        item = support.get(split)
        if item is None:
            continue
        treated_negative_rate = _rate(item["treated_negative_rows"], item["treated_rows"])
        control_negative_rate = _rate(item["control_negative_rows"], item["control_rows"])
        rows.append(
            f"| {split} | Treatment | {item['treated_rows']} | {item['treated_positive_rows']} | {item['treated_negative_rows']} | {_format_percent(_rate(item['treated_positive_rows'], item['treated_rows']), language)} | {_format_percent(treated_negative_rate, language)} | {_format_percent(item['treated_mean_outcome'], language)} |"
        )
        rows.append(
            f"| {split} | Control | {item['control_rows']} | {item['control_positive_rows']} | {item['control_negative_rows']} | {_format_percent(_rate(item['control_positive_rows'], item['control_rows']), language)} | {_format_percent(control_negative_rate, language)} | {_format_percent(item['control_mean_outcome'], language)} |"
        )
    return rows


def _observed_difference_lines(payload: dict[str, Any], language: str) -> list[str]:
    support = payload["support"]
    binary = payload["columns"]["outcome_type"] == "binary"
    rows = (
        [
            "| 切分 | Treatment 平均 Y | Control 平均 Y | 观测绝对差异 | 观测相对差异 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        if is_zh(language)
        else [
            "| Split | Treatment Mean Y | Control Mean Y | Observed Absolute Difference | Observed Relative Difference |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in DISPLAY_SPLIT_ORDER:
        item = support.get(split)
        if item is None:
            continue
        treatment_mean = item.get("treated_mean_outcome")
        control_mean = item.get("control_mean_outcome")
        absolute = None if treatment_mean is None or control_mean is None else float(treatment_mean) - float(control_mean)
        relative = None if absolute is None or control_mean in {None, 0} else absolute / float(control_mean)
        if binary:
            treatment_text = _format_percent(treatment_mean, language)
            control_text = _format_percent(control_mean, language)
            absolute_text = _format_percentage_points(absolute, language)
            relative_text = _format_percent(relative, language)
        else:
            treatment_text = _format_number(treatment_mean, language)
            control_text = _format_number(control_mean, language)
            absolute_text = _format_number(absolute, language)
            relative_text = _format_percent(relative, language)
        rows.append(f"| {split} | {treatment_text} | {control_text} | {absolute_text} | {relative_text} |")
    return rows


def _continuous_support_lines(support: dict[str, Any], language: str) -> list[str]:
    rows = (
        ["| 切分 | 组别 | 行数 | 有效 outcome 行数 | 平均 Y |", "| --- | --- | ---: | ---: | ---: |"]
        if is_zh(language)
        else ["| Split | Group | Rows | Valid Outcome Rows | Mean Y |", "| --- | --- | ---: | ---: | ---: |"]
    )
    for split in DISPLAY_SPLIT_ORDER:
        item = support.get(split)
        if item is None:
            continue
        rows.append(
            f"| {split} | Treatment | {item['treated_rows']} | {item['treated_valid_outcome_rows']} | {_format_number(item['treated_mean_outcome'], language)} |"
        )
        rows.append(
            f"| {split} | Control | {item['control_rows']} | {item['control_valid_outcome_rows']} | {_format_number(item['control_mean_outcome'], language)} |"
        )
    return rows


def _feature_candidate_lines(payload: dict[str, Any], language: str) -> list[str]:
    feature_columns = payload["feature_columns"]
    lines = ["强制排除字段：" if is_zh(language) else "Forced excluded columns:"]
    lines.extend(f"- `{column}`" for column in feature_columns["forced_exclude"])
    lines.extend(
        [
            "",
            f"{'候选特征数量' if is_zh(language) else 'Candidate feature count'}: {feature_columns['candidate_count']}",
            "",
            "候选特征预览：" if is_zh(language) else "Candidate preview:",
        ]
    )
    lines.extend(f"- `{column}`" for column in feature_columns["candidate_preview"])
    return lines


def _warning_lines(language: str, *artifacts: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for artifact in artifacts:
        warnings = artifact.get("validation", {}).get("warnings", []) + artifact.get(
            "payload", {}
        ).get("warnings", [])
        assumptions = artifact.get("validation", {}).get("assumptions", [])
        lines.extend(f"- {'警告' if is_zh(language) else 'Warning'}: {warning}" for warning in warnings)
        lines.extend(
            f"- {'假设' if is_zh(language) else 'Assumption'}: {assumption}"
            for assumption in assumptions
        )
    return lines or (["无。"] if is_zh(language) else ["None."])


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _dataset_path_lines(payload: dict[str, Any], language: str) -> list[str]:
    datasets = payload["datasets"]
    ordered = ["combined", *DISPLAY_SPLIT_ORDER]
    lines = []
    for name in ordered:
        value = datasets.get(name)
        if value:
            lines.append(f"- {name}: `{value}`")
    return lines or (["- 无。"] if is_zh(language) else ["- None."])


def _format_dataset_name(value: Any, language: str) -> str:
    if value is None:
        return "未使用" if is_zh(language) else "not used"
    return f"`{Path(str(value)).name}`"


def _format_optional_value(value: Any, language: str) -> str:
    if value is None:
        return "未提供" if is_zh(language) else "not provided"
    return f"`{value}`"


def _format_number(value: Any, language: str = "en-US") -> str:
    if value is None:
        return "不可用" if is_zh(language) else "not available"
    return f"{float(value):.4f}"


def _format_percent(value: Any, language: str = "en-US") -> str:
    if value is None:
        return "不可用" if is_zh(language) else "not available"
    return f"{float(value) * 100:.2f}%"


def _format_percentage_points(value: Any, language: str = "en-US") -> str:
    if value is None:
        return "不可用" if is_zh(language) else "not available"
    return f"{float(value) * 100:.2f} pp"


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
            "data_semantics_plan_path": None,
            "split_plan_path": None,
            "modeling_sample_spec_path": None,
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
    external_inputs: dict[str, Any] | None = None,
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
    if external_inputs:
        result["external_inputs"] = external_inputs
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


def _dependency_failure_result(
    run_dir: Path, phase: str, exc: DataLoaderImportError
) -> dict[str, Any]:
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
                "package": exc.package,
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
        outputs={
            "flow_dir": str(run_dir.resolve()),
            "report_path": None,
        },
        error={
            "code": "SAMPLE_PREPARATION_FAILED",
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


def _pd() -> Any:
    try:
        import pandas as pd  # type: ignore
    except (ImportError, ModuleNotFoundError) as exc:
        raise DataLoaderImportError("pandas", "Missing package: pandas") from exc
    return pd


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
