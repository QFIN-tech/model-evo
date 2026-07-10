from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run import run_generate_report  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_relative(run_dir: Path, path: Path) -> str:
    return path.relative_to(run_dir).as_posix()


def test_generate_report_reads_run_relative_feature_quality_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    report_dir = output_dir / "uplift-model-reporting" / "20260704T000000Z_uplift_reporting"
    (report_dir / "artifacts").mkdir(parents=True)

    task_path = output_dir / "uplift-model-task-spec" / "flow" / "artifacts" / "task_config.confirmed.v1.json"
    _write_json(
        task_path,
        {
            "artifact_kind": "task_config",
            "artifact_status": "confirmed",
            "payload": {
                "outcome_column": "converted",
                "outcome_type": "binary",
                "treatment_column": "is_treated",
                "treatment_value": 1,
                "control_value": 0,
                "unit_id_column": "user_id",
                "time_column": "pday",
                "report_preferences": {"language": "en"},
            },
        },
    )

    sample_result_path = output_dir / "uplift-model-sample-preparation" / "flow" / "results" / "sample.result.json"
    _write_json(
        sample_result_path,
        {
            "skill_name": "uplift-model-sample-preparation",
            "status": "success",
            "outputs": {"report_path": "uplift-model-sample-preparation/flow/report.md"},
            "issues": [],
        },
    )

    recommendation_path = output_dir / "uplift-model-feature-quality-analysis" / "flow" / "artifacts" / "feature_selection.json"
    basic_quality_path = output_dir / "uplift-model-feature-quality-analysis" / "flow" / "artifacts" / "basic_quality.csv"
    psi_split_path = output_dir / "uplift-model-feature-quality-analysis" / "flow" / "artifacts" / "psi_split.csv"
    _write_json(
        recommendation_path,
        {
            "status": "success",
            "selected_features": {"count": 2, "preview": ["age_days", "exposure"]},
            "diagnostic_paths": {
                "feature_basic_quality_path": _run_relative(output_dir, basic_quality_path),
                "feature_psi_split_path": _run_relative(output_dir, psi_split_path),
            },
        },
    )
    basic_quality_path.parent.mkdir(parents=True, exist_ok=True)
    basic_quality_path.write_text("feature,missing_rate,status,severity\nage_days,0.05,ok,ok\n", encoding="utf-8")
    psi_split_path.write_text("feature,comparison_split,psi\nage_days,oot,0.08\n", encoding="utf-8")

    diagnostics_result_path = output_dir / "uplift-model-feature-quality-analysis" / "flow" / "results" / "0001_diagnostics_and_selection.result.json"
    _write_json(
        diagnostics_result_path,
        {
            "skill_name": "uplift-model-feature-quality-analysis",
            "status": "success",
            "outputs": {
                "feature_selection_recommendation_path": _run_relative(output_dir, recommendation_path),
                "feature_basic_quality_path": _run_relative(output_dir, basic_quality_path),
                "feature_psi_split_path": _run_relative(output_dir, psi_split_path),
                "selection_summary": {"status": "has_recommended_features", "recommended_selected_count": 2},
            },
            "issues": [],
        },
    )
    feature_plan_path = output_dir / "uplift-model-feature-quality-analysis" / "flow" / "artifacts" / "feature_plan.confirmed.v1.json"
    _write_json(
        feature_plan_path,
        {
            "artifact_kind": "feature_plan",
            "artifact_status": "confirmed",
            "input_paths": {
                "recommendation_path": _run_relative(output_dir, recommendation_path),
                "source_result_path": _run_relative(output_dir, diagnostics_result_path),
            },
            "payload": {
                "selected_features": {"count": 1, "preview": ["age_days"], "features": ["age_days"]},
                "selection_summary": {"recommended_selected_count": 1, "source": "accepted_recommendation"},
            },
        },
    )
    feature_result_path = output_dir / "uplift-model-feature-quality-analysis" / "flow" / "results" / "0002_accept_recommendation.result.json"
    _write_json(
        feature_result_path,
        {
            "skill_name": "uplift-model-feature-quality-analysis",
            "status": "success",
            "phase": "accept_recommendation",
            "input_paths": {
                "recommendation_path": _run_relative(output_dir, recommendation_path),
                "source_result_path": _run_relative(output_dir, diagnostics_result_path),
            },
            "outputs": {
                "feature_plan_path": _run_relative(output_dir, feature_plan_path),
                "selected_features": {"count": 1, "preview": ["age_days"]},
            },
            "issues": [],
        },
    )

    result = run_generate_report(
        report_dir,
        output_dir,
        {
            "task_config_path": _run_relative(output_dir, task_path),
            "subject_result_paths": [_run_relative(output_dir, sample_result_path)],
            "supporting_paths": {
                "feature_quality_result_path": _run_relative(output_dir, feature_result_path),
                "feature_plan_path": _run_relative(output_dir, feature_plan_path),
            },
            "report_decision": {"adoption": "no_adoption_statement"},
        },
    )

    facts = json.loads(Path(result["outputs"]["report_facts_path"]).read_text(encoding="utf-8"))
    feature_quality = next(section for section in facts["sections"] if section["id"] == "feature_quality")
    feature_plan = feature_quality["facts"]["feature_plan"]
    markdown = Path(result["outputs"]["markdown_report_path"]).read_text(encoding="utf-8")

    assert feature_quality["status"] == "complete"
    assert feature_plan["status"] == "confirmed"
    assert feature_plan["selected_feature_count"] == 1
    assert feature_plan["selected_feature_preview"] == ["age_days"]
    assert "exposure" not in markdown
    assert not any(
        issue.get("code") == "MISSING_REPORT_EVIDENCE"
        and issue.get("message") == "feature_quality_result_path not provided"
        for issue in result["issues"]
    )
