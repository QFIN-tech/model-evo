from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import csv
import html
import json
import math
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

SKILL_NAME = "uplift-model-reporting"
REPORT_KIND = "uplift_modeling_report"
SECTION_IDS = [
    "executive_summary",
    "task_spec",
    "sample_quality",
    "homogeneity",
    "feature_quality",
    "candidate_models",
    "model_comparison",
    "adoption_decision",
    "risks_and_limitations",
    "next_actions",
    "trace_summary",
]
ALLOWED_ADOPTIONS = {"adopt_recommended", "adopt_specific_candidate", "no_adoption_statement", "not_applicable"}
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


def run_generate_report(run_dir: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    task_path = _required_input_path(output_dir, payload.get("task_config_path"), "task_config_path")
    task_config = _read_artifact(task_path, artifact_kind="task_config", require_confirmed=True)
    raw_subjects = payload.get("subject_result_paths") or []
    if isinstance(raw_subjects, str):
        raw_subjects = [raw_subjects]
    if not isinstance(raw_subjects, list) or not raw_subjects:
        raise MissingInputError("subject_result_paths must contain at least one result JSON path.", missing_fields=["subject_result_paths"])
    subject_paths = [_required_input_path(output_dir, item, "subject_result_paths") for item in raw_subjects]
    supporting_paths = _resolve_supporting_paths(output_dir, payload.get("supporting_paths") or {})
    subject_results = {str(path): read_json(path) for path in subject_paths}
    result_sources = dict(subject_results)
    for key in (
        "sample_preparation_result_path",
        "sample_homogeneity_result_path",
        "feature_quality_result_path",
        "comparison_result_path",
    ):
        path = supporting_paths.get(key)
        if isinstance(path, str):
            result_sources.setdefault(path, read_json(resolve_run_path(output_dir, path)))
    for path in supporting_paths.get("candidate_result_paths", []) or []:
        result_sources.setdefault(path, read_json(resolve_run_path(output_dir, path)))
    comparison_path, comparison_result = _find_result(result_sources, "uplift-model-result-comparison")
    comparison_outputs = comparison_result.get("outputs") if isinstance(comparison_result, dict) else {}
    recommended_candidate_path = str(comparison_outputs.get("recommended_candidate_result_path") or payload.get("recommended_candidate_result_path") or "")
    report_decision = _validate_report_decision(payload.get("report_decision"), recommended_candidate_path)
    facts = _build_report_facts(
        task_path=task_path,
        task_config=task_config,
        subject_paths=[str(path) for path in subject_paths],
        supporting_paths=supporting_paths,
        output_dir=output_dir,
        result_sources=result_sources,
        comparison_path=comparison_path,
        comparison_result=comparison_result,
        report_decision=report_decision,
    )
    _localize_report_assets(run_dir, output_dir, facts)
    markdown = _render_markdown(facts)
    html_report = _render_html(markdown, title="Uplift 建模报告")
    report_path = run_dir / "report.md"
    markdown_path = run_dir / "artifacts" / "uplift-report.v1.md"
    html_path = run_dir / "artifacts" / "uplift-report.v1.html"
    facts_path = run_dir / "artifacts" / "report_facts.v1.json"
    manifest_path = run_dir / "artifacts" / "report_manifest.v1.json"
    write_text(report_path, markdown)
    write_text(markdown_path, markdown)
    write_text(html_path, html_report)
    write_json(facts_path, facts)
    manifest = {
        "artifact_kind": "report_manifest",
        "artifact_version": 1,
        "report_kind": REPORT_KIND,
        "created_at": facts["generated_at"],
        "input_paths": {
            "task_config_path": str(task_path),
            "subject_result_paths": [str(path) for path in subject_paths],
            "supporting_paths": supporting_paths,
        },
        "output_paths": {
            "final_report_path": str(report_path.resolve()),
            "report_facts_path": str(facts_path.resolve()),
            "markdown_report_path": str(markdown_path.resolve()),
            "html_report_path": str(html_path.resolve()),
        },
        "section_statuses": {section["id"]: section["status"] for section in facts["sections"]},
        "missing_evidence": facts["missing_evidence"],
        "warnings": facts["warnings"],
    }
    write_json(manifest_path, manifest)
    missing_count = len(facts["missing_evidence"])
    status = "success" if missing_count == 0 else "partial_success"
    outputs = {
        "flow_dir": str(run_dir.resolve()),
        "final_report_path": str(report_path.resolve()),
        "report_facts_path": str(facts_path.resolve()),
        "report_manifest_path": str(manifest_path.resolve()),
        "markdown_report_path": str(markdown_path.resolve()),
        "html_report_path": str(html_path.resolve()),
        "report_path": str(report_path.resolve()),
        "report_summary": {
            "report_kind": REPORT_KIND,
            "section_count": len(SECTION_IDS),
            "missing_evidence_count": missing_count,
            "adoption": report_decision["adoption"],
        },
        "missing_evidence": facts["missing_evidence"],
        "risk_summary": facts["risk_summary"],
    }
    comparison_curve_path = _model_comparison_curve_report_path(facts)
    if comparison_curve_path:
        outputs["model_comparison_curve_svg_path"] = comparison_curve_path
    artifacts = [
        {"kind": "final_report", "path": str(report_path.resolve())},
        {"kind": "report_facts", "path": str(facts_path.resolve())},
        {"kind": "report_manifest", "path": str(manifest_path.resolve())},
        {"kind": "markdown_report", "path": str(markdown_path.resolve())},
        {"kind": "html_report", "path": str(html_path.resolve())},
    ]
    if comparison_curve_path:
        artifacts.append({"kind": "model_comparison_curve_svg", "path": comparison_curve_path})
    return _result(
        run_dir=run_dir,
        phase="generate_report",
        status=status,
        summary="Uplift modeling report generated from explicit upstream paths.",
        input_paths={
            "task_config_path": str(task_path),
            "subject_result_paths": [str(path) for path in subject_paths],
            "supporting_paths": supporting_paths,
        },
        outputs=outputs,
        issues=facts["issues"],
        artifacts=artifacts,
        progress=[
            {"step": "validate_paths", "status": "success"},
            {"step": "extract_facts", "status": "success"},
            {"step": "validate_adoption_decision", "status": "success"},
            {"step": "render_markdown_html", "status": "success"},
            {"step": "write_manifest", "status": "success"},
        ],
        next_steps=facts["next_actions"],
    )


def _localize_report_assets(run_dir: Path, output_dir: Path, facts: dict[str, Any]) -> None:
    comparison = _section_facts(facts, "model_comparison")
    curve_path = comparison.get("curve_svg_path")
    if not curve_path:
        return
    source = resolve_run_path(output_dir, curve_path)
    if not source.exists():
        return
    file_name = "model_comparison_auuc_curve.v1.svg"
    artifact_target = run_dir / "artifacts" / file_name
    root_target = run_dir / file_name
    for target in (artifact_target, root_target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    comparison["curve_svg_report_path"] = str(artifact_target.resolve())
    comparison["curve_svg_markdown_path"] = file_name


def _section_facts(facts: dict[str, Any], section_id: str) -> dict[str, Any]:
    for section in facts.get("sections") or []:
        if section.get("id") == section_id:
            section_facts = section.get("facts")
            if isinstance(section_facts, dict):
                return section_facts
    return {}


def _model_comparison_curve_report_path(facts: dict[str, Any]) -> str | None:
    comparison = _section_facts(facts, "model_comparison")
    value = comparison.get("curve_svg_report_path")
    return str(value) if value else None


def _build_report_facts(
    *,
    task_path: Path,
    task_config: dict[str, Any],
    subject_paths: list[str],
    supporting_paths: dict[str, Any],
    output_dir: Path,
    result_sources: dict[str, dict[str, Any]],
    comparison_path: str | None,
    comparison_result: dict[str, Any] | None,
    report_decision: dict[str, Any],
) -> dict[str, Any]:
    language = str(task_config.get("payload", {}).get("report_preferences", {}).get("language") or "zh-CN")
    task_payload = task_config.get("payload") or {}
    sample = _source_by_skill(result_sources, "uplift-model-sample-preparation")
    homogeneity = _source_by_skill(result_sources, "uplift-model-sample-homogeneity-check")
    feature = _source_by_skill(result_sources, "uplift-model-feature-quality-analysis")
    modeling_sources = [
        *_sources_by_skill(result_sources, "uplift-model-s-learner-modeling"),
        *_sources_by_skill(result_sources, "uplift-model-t-learner-modeling"),
    ]
    tuning_sources = _sources_by_skill(result_sources, "uplift-model-tuning")
    comparison_outputs = comparison_result.get("outputs") if comparison_result else {}
    candidate_rows = _candidate_rows(modeling_sources, tuning_sources, comparison_outputs, output_dir)
    sample_facts = _sample_quality_facts(sample, supporting_paths, output_dir)
    homogeneity_facts = _homogeneity_facts(homogeneity)
    feature_facts = _feature_quality_facts(feature, supporting_paths, output_dir)
    comparison_facts = _model_comparison_facts(comparison_path, comparison_outputs, candidate_rows, output_dir)
    missing = []
    if not sample_facts.get("has_evidence"):
        missing.append({"section": "sample_quality", "missing_reason": "sample_preparation_result_path not provided"})
    if not homogeneity_facts.get("has_evidence"):
        missing.append({"section": "homogeneity", "missing_reason": "sample_homogeneity_result_path not provided"})
    if not feature_facts.get("has_evidence"):
        missing.append({"section": "feature_quality", "missing_reason": "feature_quality_result_path not provided"})
    if not candidate_rows:
        missing.append({"section": "candidate_models", "missing_reason": "candidate model result paths not provided"})
    if not comparison_result:
        missing.append({"section": "model_comparison", "missing_reason": "comparison_result_path not provided"})
    warnings = []
    risk_summary = sorted(
        {
            str(risk)
            for row in candidate_rows
            for risk in row.get("risk_flags", [])
            if str(risk)
        }
    )
    if comparison_outputs.get("risk_summary"):
        risk_summary = sorted(set(risk_summary) | {str(item) for item in comparison_outputs.get("risk_summary") or []})
    issues = [
        {
            "code": "MISSING_REPORT_EVIDENCE",
            "level": "warning",
            "blocking": False,
            "message": item["missing_reason"],
            "suggested_fix": "Provide the corresponding upstream result path and regenerate the report.",
        }
        for item in missing
    ]
    sections = [
        _section("executive_summary", "complete", {"candidate_count": len(candidate_rows), "adoption": report_decision}),
        _section("task_spec", "complete", _task_facts(task_payload)),
        _section("sample_quality", "complete" if sample_facts.get("has_evidence") else "missing", sample_facts),
        _section("homogeneity", "complete" if homogeneity_facts.get("has_evidence") else "missing", homogeneity_facts),
        _section("feature_quality", "complete" if feature_facts.get("has_evidence") else "missing", feature_facts),
        _section("candidate_models", "complete" if candidate_rows else "missing", {"rows": candidate_rows}),
        _section("model_comparison", "complete" if comparison_result else "missing", comparison_facts),
        _section("adoption_decision", "complete", {"adoption": report_decision}),
        _section("risks_and_limitations", "complete", {"risk_summary": risk_summary, "missing_evidence": missing}),
        _section("next_actions", "complete", {"actions": _next_actions(report_decision, missing)}),
        _section(
            "trace_summary",
            "complete",
            {
                "task_config_path": str(task_path),
                "subject_result_paths": subject_paths,
                "supporting_paths": supporting_paths,
            },
        ),
    ]
    return {
        "artifact_kind": "report_facts",
        "artifact_version": 1,
        "report_kind": REPORT_KIND,
        "generated_at": _now(),
        "language": language,
        "sections": sections,
        "task": _task_facts(task_payload),
        "candidate_rows": candidate_rows,
        "comparison": comparison_facts,
        "model_adoption": report_decision,
        "missing_evidence": missing,
        "warnings": warnings,
        "issues": issues,
        "risk_summary": risk_summary,
        "next_actions": _next_actions(report_decision, missing),
        "section_source_coverage": _section_source_coverage(sections),
        "trace_summary": {
            "task_config_path": str(task_path),
            "subject_result_paths": subject_paths,
            "supporting_paths": supporting_paths,
            "state_pointer_used": False,
        },
    }


def _render_markdown(facts: dict[str, Any]) -> str:
    language = str(facts.get("language") or "zh-CN")
    zh = is_zh(language)
    sections = {section["id"]: section for section in facts["sections"]}
    lines = ["# Uplift 建模报告" if zh else "# Uplift Modeling Report", ""]
    lines.extend(_render_executive_summary(sections["executive_summary"], facts, language=language))
    lines.extend(_render_task_spec(sections["task_spec"], language=language))
    lines.extend(_render_sample_quality(sections["sample_quality"], language=language))
    lines.extend(_render_homogeneity(sections["homogeneity"], language=language))
    lines.extend(_render_feature_quality(sections["feature_quality"], language=language))
    lines.extend(_render_model_comparison(sections["model_comparison"], language=language))
    lines.extend(_render_adoption_and_risks(facts, language=language))
    lines.extend(_render_trace_summary(sections["trace_summary"], language=language))
    return "\n".join(lines).rstrip() + "\n"


def _render_executive_summary(sec: dict[str, Any], facts: dict[str, Any], *, language: str) -> list[str]:
    del sec
    zh = is_zh(language)
    rows = facts.get("candidate_rows") or []
    comparison = facts.get("comparison") or {}
    recommended = comparison.get("recommended")
    lines = ["## 执行摘要" if zh else "## Executive Summary", ""]
    lines.append("状态: `完整`" if zh else "Status: `complete`")
    lines.append("")
    if rows:
        lines.append(f"- 本报告汇总 `{len(rows)}` 个候选模型，并保留上游样本、特征、模型和比较证据。" if zh else f"- This report summarizes `{len(rows)}` candidate models.")
    else:
        lines.append("- 候选模型证据不完整，模型结论需要保守解读。" if zh else "- Candidate model evidence is incomplete.")
    if recommended:
        conclusion = comparison.get("conclusion")
        lines.append(conclusion if conclusion else ("模型比较给出了推荐候选；是否采用仍取决于人工复核。" if zh else "Model comparison produced a recommendation; adoption still requires review."))
    else:
        lines.append("- 模型比较未给出可采用的推荐候选。" if zh else "- Model comparison did not produce an adoptable recommendation.")
    if facts.get("missing_evidence"):
        lines.append(f"- 有 `{len(facts['missing_evidence'])}` 类证据缺失，详见风险与追踪摘要。" if zh else f"- `{len(facts['missing_evidence'])}` evidence groups are missing.")
    lines.append("")
    return lines


def _render_task_spec(sec: dict[str, Any], *, language: str) -> list[str]:
    zh = is_zh(language)
    facts = sec.get("facts") or {}
    return [
        "## 任务定义" if zh else "## Task Definition",
        "",
        "状态: `完整`" if zh else "Status: `complete`",
        "",
        "| 字段 | 值 |" if zh else "| Field | Value |",
        "| --- | --- |",
        f"| Y | {_code_or_na(facts.get('outcome_column'))} ({_value_or_na(facts.get('outcome_type'))}) |",
        f"| Treatment | {_code_or_na(facts.get('treatment_column'))} (treatment={_value_or_na(facts.get('treatment_value'))}, control={_value_or_na(facts.get('control_value'))}) |",
        f"| Unit ID | {_value_or_na(facts.get('unit_id_column'))} |",
        f"| Time | {_value_or_na(facts.get('time_column'))} |",
        "",
    ]


def _render_sample_quality(sec: dict[str, Any], *, language: str) -> list[str]:
    zh = is_zh(language)
    facts = sec.get("facts") or {}
    lines = ["## 样本质量" if zh else "## Sample Quality", "", f"状态: `{_zh_status(sec.get('status'))}`" if zh else f"Status: `{sec.get('status')}`", ""]
    if sec.get("status") == "missing":
        lines.append("- 证据未提供；本报告不会补算该环节指标。" if zh else "- Evidence was not provided; reporting does not recompute this section.")
        lines.append("")
        return lines
    lines.append("说明：下表是 treatment/control 的描述性 outcome 摘要，不是 uplift model lift 或因果效应估计。" if zh else "The table below is descriptive and is not a causal estimate.")
    support = facts.get("group_support") or []
    if support:
        lines.extend(["", "| 切分 | Treatment 行数 | Control 行数 | Treatment 平均 Y | Control 平均 Y |" if zh else "| Split | Treatment Rows | Control Rows | Treatment Mean Y | Control Mean Y |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in support:
            lines.append(f"| {row.get('split')} | {_format_count(row.get('treated_rows'))} | {_format_count(row.get('control_rows'))} | {_format_percent(row.get('treated_mean_outcome'))} | {_format_percent(row.get('control_mean_outcome'))} |")
    diffs = facts.get("observed_differences") or []
    if diffs:
        lines.extend(["", "| 切分 | Treatment 平均 Y | Control 平均 Y | 观测绝对差异 | 观测相对差异 |" if zh else "| Split | Treatment Mean Y | Control Mean Y | Observed Absolute Difference | Observed Relative Difference |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in diffs:
            lines.append(f"| {row.get('split')} | {_format_percent(row.get('treatment_mean_y'))} | {_format_percent(row.get('control_mean_y'))} | {_format_pp(row.get('observed_absolute_difference'))} | {_format_percent(row.get('observed_relative_difference'))} |")
    if facts.get("report_path"):
        lines.extend(["", f"- 上游报告: `{_compact_path(facts.get('report_path'))}`"])
    lines.append("")
    return lines


def _render_homogeneity(sec: dict[str, Any], *, language: str) -> list[str]:
    zh = is_zh(language)
    facts = sec.get("facts") or {}
    lines = ["## 同质性" if zh else "## Homogeneity", "", f"状态: `{_zh_status(sec.get('status'))}`" if zh else f"Status: `{sec.get('status')}`", ""]
    if sec.get("status") == "missing":
        lines.append("- 未提供同质性诊断结果。" if zh else "- No homogeneity diagnostics were provided.")
        lines.append("")
        return lines
    lines.append(_homogeneity_conclusion(facts, language=language))
    auc_rows = facts.get("auc_rows") or []
    if auc_rows:
        lines.extend(["", "### AUC 测试结果" if zh else "### AUC Diagnostics", "", "| Split | Rows | Treatment | Control | AUC | Status | Top Variables |", "| --- | ---: | ---: | ---: | ---: | --- | --- |"])
        for row in auc_rows:
            lines.append(f"| {_display_dataset_role(row.get('split'))} | {_format_count(row.get('rows'))} | {_format_count(row.get('treatment_rows'))} | {_format_count(row.get('control_rows'))} | {_format_decimal(row.get('auc'), 4)} | {_value_or_na(row.get('status'))} | {_join_preview(row.get('top_predictive_variables'), limit=3)} |")
    smd_rows = facts.get("smd_rows") or []
    if smd_rows:
        lines.extend(["", "### SMD 测试结果" if zh else "### SMD Diagnostics", "", "| Split | Checked Variables | SMD >= 0.1 | SMD >= 0.2 | Max Abs SMD | Status |", "| --- | ---: | ---: | ---: | ---: | --- |"])
        for row in smd_rows:
            lines.append(f"| {_display_dataset_role(row.get('split'))} | {_format_count(row.get('checked_variables'))} | {_format_count(row.get('variables_ge_0_1'))} | {_format_count(row.get('variables_ge_0_2'))} | {_format_decimal(row.get('max_abs_smd'), 4)} | {_value_or_na(row.get('status'))} |")
    top_rows = facts.get("top_imbalanced_variables") or []
    if top_rows:
        lines.extend(["", "Top 不平衡变量：" if zh else "Top Imbalanced Variables:", "", "| Split | Variable | SMD | Status |", "| --- | --- | ---: | --- |"])
        for row in top_rows[:5]:
            lines.append(f"| {_display_dataset_role(row.get('split'))} | {row.get('variable')} | {_format_decimal(row.get('smd'), 4)} | {_value_or_na(row.get('status'))} |")
    artifact_rows = _artifact_rows(
        [
            ("Markdown 报告" if zh else "Markdown report", facts.get("report_path")),
            ("SMD 明细" if zh else "SMD details", facts.get("smd_detail_path")),
            ("AUC 特征重要性" if zh else "AUC feature importance", facts.get("auc_feature_importance_path")),
            ("协变量列表" if zh else "Covariate list", facts.get("covariates_path")),
        ]
    )
    if artifact_rows:
        lines.extend(["", "### 完整评估结果" if zh else "### Detailed Artifacts", ""])
        lines.extend(artifact_rows)
    lines.append("")
    return lines


def _render_feature_quality(sec: dict[str, Any], *, language: str) -> list[str]:
    zh = is_zh(language)
    facts = sec.get("facts") or {}
    lines = ["## 特征质量" if zh else "## Feature Quality", "", f"状态: `{_zh_status(sec.get('status'))}`" if zh else f"Status: `{sec.get('status')}`", ""]
    if sec.get("status") == "missing":
        lines.append("- 未提供特征质量诊断结果。" if zh else "- No feature quality diagnostics were provided.")
        lines.append("")
        return lines
    plan = facts.get("feature_plan") or {}
    lines.extend(
        [
            "### 当前特征方案" if zh else "### Current Feature Plan",
            "",
            f"- 特征方案状态: `{_display_feature_plan_status(plan.get('status'))}`" if zh else f"- Feature plan status: `{_value_or_na(plan.get('status'))}`",
            f"- 入模特征数: `{_format_count(plan.get('selected_feature_count'))}`" if zh else f"- Selected feature count: `{_format_count(plan.get('selected_feature_count'))}`",
            f"- 特征预览: `{_join_preview(plan.get('selected_feature_preview'), limit=12)}`" if zh else f"- Feature preview: `{_join_preview(plan.get('selected_feature_preview'), limit=12)}`",
            f"- 筛选依据: `{_join_preview(facts.get('selection_basis'), limit=6)}`" if zh else f"- Selection basis: `{_join_preview(facts.get('selection_basis'), limit=6)}`",
        ]
    )
    missing_stats = facts.get("missing_rate_distribution") or []
    if missing_stats:
        lines.extend(["", "### 缺失率分布" if zh else "### Missing Rate Distribution", "", "| Missing Rate Bucket | Count | Percentage |", "| --- | ---: | ---: |"])
        for row in missing_stats:
            lines.append(f"| {row.get('bucket')} | {_format_count(row.get('count'))} | {_format_percent(row.get('percentage'))} |")
        lines.append(f"\n常量 / 全缺失特征数: `{_format_count(facts.get('constant_or_all_missing_count'))}`" if zh else f"\nConstant/all-missing features: `{_format_count(facts.get('constant_or_all_missing_count'))}`")
    psi_stats = facts.get("psi_distribution") or []
    if psi_stats:
        lines.extend(["", "### PSI 分布" if zh else "### PSI Distribution", "", "| PSI Bucket | Count | Percentage |", "| --- | ---: | ---: |"])
        for row in psi_stats:
            lines.append(f"| {row.get('bucket')} | {_format_count(row.get('count'))} | {_format_percent(row.get('percentage'))} |")
        lines.append(f"\n- 最大 PSI: `{_format_decimal(facts.get('max_psi'), 4)}`" if zh else f"\n- Max PSI: `{_format_decimal(facts.get('max_psi'), 4)}`")
        lines.append(f"- 最大 PSI 特征: `{_value_or_na(facts.get('max_psi_feature'))}`" if zh else f"- Max PSI feature: `{_value_or_na(facts.get('max_psi_feature'))}`")
        lines.append(f"- 最大 PSI split: `{_value_or_na(facts.get('max_psi_split'))}`" if zh else f"- Max PSI split: `{_value_or_na(facts.get('max_psi_split'))}`")
    artifact_rows = _artifact_rows(
        [
            ("特征质量报告" if zh else "Feature quality report", facts.get("report_path")),
            ("Basic Quality 明细" if zh else "Basic quality details", facts.get("feature_basic_quality_path")),
            ("Split PSI 明细" if zh else "Split PSI details", facts.get("feature_psi_split_path")),
            ("Monthly PSI 明细" if zh else "Monthly PSI details", facts.get("feature_psi_monthly_path")),
            ("入模特征列表" if zh else "Selected features", facts.get("selected_features_path")),
        ]
    )
    if artifact_rows:
        lines.extend(["", "### 完整计算结果" if zh else "### Detailed Artifacts", ""])
        lines.extend(artifact_rows)
    lines.append("")
    return lines


def _render_model_comparison(sec: dict[str, Any], *, language: str) -> list[str]:
    zh = is_zh(language)
    facts = sec.get("facts") or {}
    lines = ["## 模型比较" if zh else "## Model Comparison", ""]
    if sec.get("status") == "missing":
        lines.append("- 未提供模型比较结果；本报告不推荐采用某个候选模型。" if zh else "- No model comparison result was provided.")
    else:
        lines.append("状态: `完整`" if zh else "Status: `complete`")
        lines.append("")
        if facts.get("conclusion"):
            lines.append(facts["conclusion"])
            lines.append("")
        lines.append(f"- 比较样本：`{_display_dataset_role(facts.get('comparison_dataset_role'))}`")
        lines.append(f"- 可公平比较：`{_yes_no((facts.get('comparability') or {}).get('is_fairly_comparable'), language=language)}`")
        if facts.get("recommendation_rationale"):
            lines.append(f"- 推荐说明：{_localized_rationale(facts.get('recommendation_rationale'), language=language)}")
        if facts.get("curve_svg_markdown_path"):
            lines.extend(
                [
                    "",
                    "### AUUC 曲线对比" if zh else "### AUUC Curve Comparison",
                    "",
                    f"![AUUC Curve Comparison]({facts.get('curve_svg_markdown_path')})",
                ]
            )
        lines.extend(["", "### AUUC 对比" if zh else "### AUUC Comparison", ""])
        rows = facts.get("rows") or []
        if rows:
            lines.extend(["| 模型 | 评估集 | AUUC | 排名 | 是否推荐 |" if zh else "| Model | Split | AUUC | Rank | Recommended |", "| --- | --- | ---: | ---: | --- |"])
            for row in rows:
                lines.append(f"| {row.get('display_name')} | {_display_dataset_role(row.get('dataset_role'))} | {_format_decimal(row.get('metric_value'), 4)} | {_format_count(row.get('rank'))} | {_yes_no(row.get('is_recommended'), language=language)} |")
        artifact_rows = _artifact_rows(
            [
                ("对比表" if zh else "Comparison table", facts.get("comparison_table_path")),
                ("模型对比摘要" if zh else "Model comparison summary", facts.get("model_comparison_summary_path")),
                ("上游比较报告" if zh else "Upstream comparison report", facts.get("report_path")),
            ]
        )
        if artifact_rows:
            lines.extend(["", "### 完整模型结果入口" if zh else "### Detailed Model Artifacts", ""])
            lines.extend(artifact_rows)
    lines.append("")
    return lines


def _render_adoption_and_risks(facts: dict[str, Any], *, language: str) -> list[str]:
    zh = is_zh(language)
    adoption = facts.get("model_adoption") or {}
    lines = ["## 采用决策" if zh else "## Adoption Decision", ""]
    selected_name = _selected_candidate_name(adoption, facts.get("candidate_rows") or [])
    lines.append(f"当前报告采用模型对比推荐作为报告表达：{selected_name}。" if zh else f"The report expression follows the comparison recommendation: {selected_name}.")
    lines.append(
        "这不是最终上线决策，也不会创建最终模型指针。" if zh else "This is not a deployment decision and does not create a production model pointer."
    )
    lines.extend(["", "## 风险与限制" if zh else "## Risks and Limitations", ""])
    if facts.get("risk_summary"):
        lines.extend(f"- {_localized_risk(code, language=language)}" for code in facts["risk_summary"])
    else:
        lines.append("- 未从上游结果中发现已标记风险。" if zh else "- No upstream risk flag was found.")
    for item in facts.get("missing_evidence") or []:
        lines.append(f"- {item.get('section')} 证据不完整：{item.get('missing_reason')}")
    lines.extend(
        [
            "- 样本中的 treatment/control outcome 差异只是描述性统计，不是模型 uplift 或因果效应估计。" if zh else "- Observed group differences are descriptive, not causal estimates.",
            "- 模型对比推荐不是最终部署或上线决策。" if zh else "- Comparison recommendations are not deployment decisions.",
            "- 当前报告生成环节未执行 final refit、生产验证或最终模型指针创建。" if zh else "- Reporting did not perform final refit, production validation, or model pointer creation.",
            "",
            "## 建议下一步" if zh else "## Recommended Next Steps",
            "",
        ]
    )
    for item in facts.get("next_actions") or []:
        lines.append(f"- {item.get('reason')}")
    lines.append("")
    return lines


def _render_trace_summary(sec: dict[str, Any], *, language: str) -> list[str]:
    zh = is_zh(language)
    facts = sec.get("facts") or {}
    lines = ["## 追踪摘要" if zh else "## Trace Summary", ""]
    lines.append(f"- task_config_path: `{facts.get('task_config_path')}`")
    lines.append(f"- subject_result_paths: `{len(facts.get('subject_result_paths') or [])}`")
    supporting = facts.get("supporting_paths") or {}
    for key in sorted(supporting):
        value = supporting[key]
        if isinstance(value, list):
            lines.append(f"- {key}: `{len(value)}` paths")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.append("- 可复现说明: `本报告只读取本次显式传入的上游路径，未重算样本、特征、模型或对比指标。`" if zh else "- reproducibility: `All inputs came from explicit paths; reporting did not recompute upstream metrics.`")
    lines.append("")
    return lines


def _render_html(markdown: str, *, title: str) -> str:
    parts = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\">",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;line-height:1.55;max-width:1040px;margin:32px auto;padding:0 20px;color:#202124}table{border-collapse:collapse;width:100%;margin:12px 0}th,td{border:1px solid #d0d7de;padding:6px 8px;text-align:left}th{background:#f6f8fa}code{background:#f6f8fa;padding:1px 4px;border-radius:3px}</style>",
        "</head><body>",
    ]
    for line in markdown.splitlines():
        if line.startswith("# "):
            parts.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            parts.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            parts.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt, src = line[2:-1].split("](", 1)
            parts.append(
                f'<p><img src="{html.escape(src, quote=True)}" '
                f'alt="{html.escape(alt, quote=True)}" style="max-width:100%;height:auto"></p>'
            )
        elif line.startswith("- "):
            parts.append(f"<p>{html.escape(line)}</p>")
        elif line.startswith("|"):
            parts.append(f"<pre>{html.escape(line)}</pre>")
        elif not line.strip():
            parts.append("")
        else:
            parts.append(f"<p>{html.escape(line)}</p>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


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
    elif action != "generate_report":
        result = _unsupported_result(run_dir, action or "unknown", "action must be generate_report.")
    else:
        legacy_fields = _legacy_ref_rejections(body)
        if legacy_fields:
            result = _legacy_ref_result(run_dir, action, legacy_fields)
        else:
            try:
                result = run_generate_report(run_dir, output_dir, body)
            except MissingInputError as exc:
                result = _needs_input_result(run_dir, action, str(exc), exc.missing_fields)
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
    return ensure_skill_call_dir(output_dir, SKILL_NAME, "uplift_reporting")


def _read_artifact(path: Path, *, artifact_kind: str, require_confirmed: bool = False) -> dict[str, Any]:
    artifact = read_json(path)
    if artifact.get("artifact_kind") != artifact_kind:
        raise MissingInputError(f"{path} is not a {artifact_kind} artifact.")
    if require_confirmed and artifact.get("artifact_status") != "confirmed":
        raise MissingInputError(f"{artifact_kind} must be confirmed.")
    return artifact


def _resolve_supporting_paths(output_dir: Path, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MissingInputError("supporting_paths must be an object.", missing_fields=["supporting_paths"])
    resolved: dict[str, Any] = {}
    for key, item in value.items():
        if key.endswith("_paths"):
            if not isinstance(item, list):
                raise MissingInputError(f"{key} must be a list.")
            resolved[key] = [str(_required_input_path(output_dir, path, key)) for path in item]
        elif key.endswith("_path"):
            if item in (None, ""):
                continue
            resolved[key] = str(_required_input_path(output_dir, item, key))
        else:
            resolved[key] = item
    return resolved


def _required_input_path(output_dir: Path, value: Any, field_name: str) -> Path:
    if not value:
        raise MissingInputError(f"{field_name} is required.", missing_fields=[field_name])
    path = _resolve_input_path(output_dir, str(value))
    if not path.exists():
        raise MissingInputError(f"{field_name} does not exist: {path}", missing_fields=[field_name])
    return path


def _resolve_input_path(output_dir: Path, value: str) -> Path:
    return resolve_run_path(output_dir, value)


def _validate_report_decision(raw: Any, recommended_candidate_path: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        if recommended_candidate_path:
            raise MissingInputError("report_decision is required when comparison recommends a candidate.", missing_fields=["report_decision"])
        raw = {"adoption": "not_applicable"}
    adoption = str(raw.get("adoption") or "")
    if adoption not in ALLOWED_ADOPTIONS:
        raise MissingInputError("report_decision.adoption is invalid.", missing_fields=["report_decision.adoption"])
    selected = str(raw.get("selected_candidate_result_path") or "")
    if adoption == "adopt_recommended":
        selected = recommended_candidate_path
    if adoption == "adopt_specific_candidate" and not selected:
        raise MissingInputError("selected_candidate_result_path is required for adopt_specific_candidate.", missing_fields=["report_decision.selected_candidate_result_path"])
    return {
        "adoption": adoption,
        "selected_candidate_result_path": selected,
        "recommended_candidate_result_path": recommended_candidate_path,
        "decision_maker": str(raw.get("decision_maker") or "user"),
        "decision_at": _now(),
        "notes": str(raw.get("notes") or ""),
    }


def _source_by_skill(sources: dict[str, dict[str, Any]], skill_name: str) -> dict[str, Any] | None:
    for source in sources.values():
        if source.get("skill_name") == skill_name:
            return source
    return None


def _sources_by_skill(sources: dict[str, dict[str, Any]], skill_name: str) -> list[tuple[str, dict[str, Any]]]:
    return [(path, source) for path, source in sources.items() if source.get("skill_name") == skill_name]


def _find_result(sources: dict[str, dict[str, Any]], skill_name: str) -> tuple[str | None, dict[str, Any] | None]:
    for path, source in sources.items():
        if source.get("skill_name") == skill_name:
            return path, source
    return None, None


def _sample_quality_facts(result: dict[str, Any] | None, supporting_paths: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    facts = _result_brief(result)
    spec_path = supporting_paths.get("modeling_sample_spec_path")
    if not spec_path and result:
        outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
        spec_path = outputs.get("modeling_sample_spec_path")
    spec = _safe_read_json(spec_path, output_dir) if spec_path else {}
    payload = spec.get("payload") if isinstance(spec.get("payload"), dict) else {}
    support = payload.get("support") if isinstance(payload.get("support"), dict) else {}
    group_support = []
    observed_differences = []
    for split, values in sorted(support.items(), key=lambda item: _split_sort_key(item[0])):
        if not isinstance(values, dict):
            continue
        group_support.append(
            {
                "split": split,
                "treated_rows": values.get("treated_rows"),
                "control_rows": values.get("control_rows"),
                "treated_mean_outcome": values.get("treated_mean_outcome"),
                "control_mean_outcome": values.get("control_mean_outcome"),
            }
        )
        observed_differences.append(
            {
                "split": split,
                "treatment_mean_y": values.get("treated_mean_outcome"),
                "control_mean_y": values.get("control_mean_outcome"),
                "observed_absolute_difference": values.get("observed_outcome_difference"),
                "observed_relative_difference": values.get("observed_outcome_relative_difference"),
            }
        )
    facts.update(
        {
            "has_evidence": bool(result or group_support),
            "modeling_sample_spec_path": str(spec_path) if spec_path else None,
            "group_support": group_support,
            "observed_differences": observed_differences,
        }
    )
    return facts


def _homogeneity_facts(result: dict[str, Any] | None) -> dict[str, Any]:
    facts = _result_brief(result)
    outputs = result.get("outputs") if result and isinstance(result.get("outputs"), dict) else {}
    auc_rows = []
    for split, row in sorted((outputs.get("auc_by_split") or {}).items(), key=lambda item: _split_sort_key(item[0])):
        if isinstance(row, dict):
            auc_rows.append({"split": split, **row})
    smd_rows = []
    for split, row in sorted((outputs.get("smd_summary_by_split") or {}).items(), key=lambda item: _split_sort_key(item[0])):
        if not isinstance(row, dict):
            continue
        smd_rows.append(
            {
                "split": split,
                "checked_variables": row.get("checked_variables"),
                "variables_ge_0_1": row.get("variables_ge_0_1", row.get("variables_over_0_1")),
                "variables_ge_0_2": row.get("variables_ge_0_2", row.get("variables_over_0_2")),
                "max_abs_smd": row.get("max_abs_smd"),
                "status": row.get("status"),
            }
        )
    facts.update(
        {
            "has_evidence": bool(result and (auc_rows or smd_rows)),
            "overall_status": outputs.get("overall_status"),
            "recommended_action": outputs.get("recommended_action"),
            "auc_rows": auc_rows,
            "smd_rows": smd_rows,
            "top_imbalanced_variables": outputs.get("top_imbalanced_variables") or [],
            "smd_detail_path": outputs.get("smd_detail_path"),
            "auc_feature_importance_path": outputs.get("auc_feature_importance_path"),
            "covariates_path": outputs.get("covariates_path"),
        }
    )
    return facts


def _feature_quality_facts(result: dict[str, Any] | None, supporting_paths: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    facts = _result_brief(result)
    outputs = result.get("outputs") if result and isinstance(result.get("outputs"), dict) else {}
    input_paths = result.get("input_paths") if result and isinstance(result.get("input_paths"), dict) else {}
    plan = _safe_read_json(supporting_paths.get("feature_plan_path"), output_dir) if supporting_paths.get("feature_plan_path") else {}
    plan_input_paths = plan.get("input_paths") if isinstance(plan.get("input_paths"), dict) else {}
    source_result = _safe_read_json(input_paths.get("source_result_path"), output_dir) if input_paths.get("source_result_path") else {}
    source_outputs = source_result.get("outputs") if isinstance(source_result.get("outputs"), dict) else {}
    recommendation_path = (
        outputs.get("feature_selection_recommendation_path")
        or input_paths.get("recommendation_path")
        or plan_input_paths.get("recommendation_path")
        or source_outputs.get("feature_selection_recommendation_path")
    )
    recommendation = _safe_read_json(recommendation_path, output_dir)
    diagnostic_paths = recommendation.get("diagnostic_paths") if isinstance(recommendation.get("diagnostic_paths"), dict) else {}
    selected = recommendation.get("selected_features") if isinstance(recommendation.get("selected_features"), dict) else {}
    plan_payload = plan.get("payload") if isinstance(plan.get("payload"), dict) else {}
    plan_selected = plan_payload.get("selected_features") if isinstance(plan_payload.get("selected_features"), dict) else {}
    plan_selected_preview = (
        plan_selected.get("preview")
        or plan_selected.get("features")
        or plan.get("selected_features")
        or plan.get("features")
        or []
    )
    selected_preview = plan_selected_preview or selected.get("preview") or []
    basic_quality_path = (
        outputs.get("feature_basic_quality_path")
        or source_outputs.get("feature_basic_quality_path")
        or diagnostic_paths.get("feature_basic_quality_path")
    )
    psi_split_path = (
        outputs.get("feature_psi_split_path")
        or source_outputs.get("feature_psi_split_path")
        or diagnostic_paths.get("feature_psi_split_path")
    )
    basic_rows = _safe_read_csv(basic_quality_path, output_dir)
    psi_rows = _safe_read_csv(psi_split_path, output_dir)
    missing_distribution = _missing_rate_distribution(basic_rows)
    psi_distribution, max_psi = _psi_distribution(psi_rows)
    selection_basis = outputs.get("selection_basis") or source_outputs.get("selection_basis") or {}
    selection_summary = outputs.get("selection_summary") or source_outputs.get("selection_summary") or plan_payload.get("selection_summary") or {}
    facts.update(
        {
            "has_evidence": bool(result and (recommendation or basic_rows or psi_rows or selected_preview)),
            "feature_plan": {
                "status": plan.get("artifact_status") or recommendation.get("status") or selection_summary.get("status"),
                "selected_feature_count": _first_present(
                    plan_selected.get("count"),
                    len(plan_selected_preview) if plan_selected_preview else None,
                    selected.get("count"),
                    selection_summary.get("recommended_selected_count"),
                ),
                "selected_feature_preview": selected_preview,
            },
            "selection_basis": selection_basis.get("participating") if isinstance(selection_basis, dict) else [],
            "missing_rate_distribution": missing_distribution,
            "constant_or_all_missing_count": _constant_or_all_missing_count(basic_rows),
            "psi_distribution": psi_distribution,
            "max_psi": max_psi.get("psi"),
            "max_psi_feature": max_psi.get("feature"),
            "max_psi_split": max_psi.get("comparison_split"),
            "feature_basic_quality_path": basic_quality_path,
            "feature_psi_split_path": psi_split_path,
            "feature_psi_monthly_path": outputs.get("feature_psi_monthly_path") or diagnostic_paths.get("feature_psi_monthly_path"),
            "selected_features_path": selected.get("features_path") or plan_selected.get("features_path"),
            "feature_selection_recommendation_path": recommendation_path,
        }
    )
    return facts


def _model_comparison_facts(
    comparison_path: str | None,
    outputs: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    summary = _safe_read_json(outputs.get("model_comparison_summary_path"), output_dir)
    summary_rows = summary.get("rows") if isinstance(summary.get("rows"), list) else []
    normalized = outputs.get("normalized_candidates") or []
    recommended_path = str(outputs.get("recommended_candidate_result_path") or summary.get("recommended_candidate_result_path") or "")
    rows = []
    for row in summary_rows:
        if not isinstance(row, dict) or row.get("exclusion_code"):
            continue
        source_path = str(row.get("source_result_path") or row.get("producer_result_path") or "")
        if not source_path:
            continue
        rows.append(_comparison_display_row(row, source_path, recommended_path, output_dir))
    for index, row in enumerate(normalized, start=1):
        if rows:
            break
        if not isinstance(row, dict):
            continue
        metric = row.get("split_evaluation") or {}
        producer_path = str(row.get("producer_result_path") or "")
        rows.append(
            {
                "display_name": _readable_model_name(row),
                "dataset_role": metric.get("dataset_role"),
                "metric_value": metric.get("metric_value", metric.get("auuc_raw")),
                "auuc_normalized": metric.get("auuc_normalized"),
                "rank": index,
                "is_recommended": bool(recommended_path and producer_path == recommended_path),
                "producer_result_path": producer_path,
                "base_candidate_result_path": row.get("base_candidate_result_path") or "",
                "producer_kind": row.get("producer_kind") or "",
                "learner": row.get("learner") or "",
                "risk_flags": row.get("risk_flags") or [],
            }
        )
    if not rows and candidate_rows:
        for index, row in enumerate(candidate_rows, start=1):
            rows.append(
                {
                    "display_name": row.get("display_name"),
                    "dataset_role": row.get("dataset_role"),
                    "metric_value": row.get("metric_value"),
                    "auuc_normalized": row.get("auuc_normalized"),
                    "rank": index,
                    "is_recommended": bool(
                        recommended_path and _same_path(row.get("producer_result_path"), recommended_path, output_dir)
                    ),
                    "producer_result_path": row.get("producer_result_path"),
                    "base_candidate_result_path": row.get("base_candidate_result_path") or "",
                    "producer_kind": row.get("producer_kind") or "",
                    "learner": row.get("learner") or "",
                    "risk_flags": row.get("risk_flags") or [],
                }
            )
    recommended = next((row for row in rows if row.get("is_recommended")), None)
    baseline = _paired_baseline_row(rows, recommended, output_dir)
    return {
        "path": comparison_path,
        "has_evidence": bool(outputs),
        "candidate_count": outputs.get("candidate_count") or len(rows),
        "comparison_dataset_role": outputs.get("comparison_dataset_role"),
        "comparability": outputs.get("comparability") or {},
        "recommendation_rationale": outputs.get("recommendation_rationale"),
        "recommended_candidate_result_path": recommended_path or None,
        "recommended": recommended,
        "baseline": baseline,
        "rows": rows,
        "conclusion": _model_comparison_conclusion(rows, recommended, baseline),
        "comparison_table_path": outputs.get("comparison_table_path"),
        "model_comparison_summary_path": outputs.get("model_comparison_summary_path"),
        "report_path": outputs.get("report_path"),
        "curve_svg_path": outputs.get("curve_svg_path") or summary.get("curve_svg_path"),
        "risk_summary": outputs.get("risk_summary") or [],
    }


def _comparison_display_row(row: dict[str, Any], source_path: str, recommended_path: str, output_dir: Path) -> dict[str, Any]:
    producer_kind = str(row.get("producer_kind") or "")
    learner = str(row.get("learner") or "")
    name_payload = {
        "producer_kind": producer_kind,
        "learner": learner,
        "model_spec": {"model_type": learner} if learner else {},
    }
    return {
        "display_name": _readable_model_name(name_payload),
        "dataset_role": row.get("dataset_role"),
        "metric_value": _first_present(row.get("primary_metric_value"), row.get("metric_value"), row.get("auuc_raw")),
        "auuc_normalized": row.get("auuc_normalized"),
        "rank": row.get("rank"),
        "is_recommended": _truthy(row.get("is_recommended")) or bool(recommended_path and _same_path(source_path, recommended_path, output_dir)),
        "producer_result_path": source_path,
        "base_candidate_result_path": str(row.get("base_candidate_result_path") or ""),
        "producer_kind": producer_kind,
        "learner": learner,
        "risk_flags": _risk_flags(row.get("risk_flags")),
    }


def _paired_baseline_row(rows: list[dict[str, Any]], recommended: dict[str, Any] | None, output_dir: Path) -> dict[str, Any] | None:
    if not recommended:
        return None
    base_path = str(recommended.get("base_candidate_result_path") or "")
    if base_path:
        match = next((row for row in rows if _same_path(row.get("producer_result_path"), base_path, output_dir)), None)
        if match:
            return match
    baselines = [row for row in rows if row.get("producer_kind") == "modeling"]
    rec_learner = str(recommended.get("learner") or "")
    if rec_learner:
        same_learner = [row for row in baselines if row.get("learner") == rec_learner]
        if len(same_learner) == 1:
            return same_learner[0]
    if len(baselines) == 1:
        return baselines[0]
    return None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _truthy(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _risk_flags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [item for item in value.split(";") if item]
    return []


def _result_brief(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    return {
        "status": result.get("status"),
        "summary": result.get("summary"),
        "report_path": outputs.get("report_path"),
        "issue_count": len(result.get("issues") or []),
    }


def _candidate_rows(
    modeling_sources: list[tuple[str, dict[str, Any]]],
    tuning_sources: list[tuple[str, dict[str, Any]]],
    comparison_outputs: dict[str, Any],
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    comparison_by_path = {
        str(row.get("producer_result_path")): row
        for row in comparison_outputs.get("normalized_candidates") or []
        if isinstance(row, dict)
    }
    for path, result in [*modeling_sources, *tuning_sources]:
        outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
        candidate = _safe_read_json(outputs.get("model_candidate_path"), output_dir) if outputs.get("model_candidate_path") else {}
        if not isinstance(candidate, dict) or not candidate:
            candidate = outputs.get("model_candidate")
        if not isinstance(candidate, dict):
            candidate = {}
        comparison_row = comparison_by_path.get(path, {})
        split_eval = comparison_row.get("split_evaluation") or _primary_split_eval(outputs)
        support = split_eval.get("support") if isinstance(split_eval, dict) else {}
        rows.append(
            {
                "producer_result_path": path,
                "candidate_id": candidate.get("candidate_id") or result.get("run_id"),
                "producer_kind": candidate.get("producer_kind") or ("tuning" if result.get("skill_name") == "uplift-model-tuning" else "modeling"),
                "display_name": _readable_model_name(candidate),
                "dataset_role": split_eval.get("dataset_role") if isinstance(split_eval, dict) else None,
                "metric_value": split_eval.get("metric_value") if isinstance(split_eval, dict) else None,
                "auuc_normalized": split_eval.get("auuc_normalized") if isinstance(split_eval, dict) else None,
                "row_count": support.get("row_count") if isinstance(support, dict) else None,
                "risk_flags": candidate.get("risk_flags") or comparison_row.get("risk_flags") or [],
                "model_path": candidate.get("model_artifact_path") or outputs.get("model_artifact_path"),
                "feature_importance_path": outputs.get("feature_importance_path"),
                "feature_importance_components": _feature_importance_components(outputs.get("feature_importance_path"), output_dir),
                "report_path": outputs.get("report_path"),
            }
        )
    return rows


def _primary_split_eval(outputs: dict[str, Any]) -> dict[str, Any]:
    split_evaluations = outputs.get("split_evaluations")
    if isinstance(split_evaluations, dict):
        for key in ("test", "valid", "train", "oot"):
            if isinstance(split_evaluations.get(key), dict):
                return split_evaluations[key]
    metrics = outputs.get("modeling_metrics") or {}
    splits = metrics.get("splits") or {}
    for key in ("test", "valid", "train", "oot"):
        split = splits.get(key)
        if isinstance(split, dict):
            return {
                "dataset_role": key,
                "metric_value": split.get("auuc_raw"),
                "auuc_normalized": split.get("auuc_normalized"),
                "support": {
                    "row_count": split.get("row_count"),
                    "treatment_count": split.get("treatment_count"),
                    "control_count": split.get("control_count"),
                    "valid_bin_count": split.get("valid_bin_count"),
                },
            }
    return {}


def _task_facts(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome_column": payload.get("outcome_column"),
        "outcome_type": payload.get("outcome_type"),
        "treatment_column": payload.get("treatment_column"),
        "treatment_value": payload.get("treatment_value"),
        "control_value": payload.get("control_value"),
        "unit_id_column": payload.get("unit_id_column"),
        "time_column": payload.get("time_column"),
    }


def _section(section_id: str, status: str, facts: dict[str, Any]) -> dict[str, Any]:
    return {"id": section_id, "status": status, "facts": facts}


def _next_actions(report_decision: dict[str, Any], missing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if missing:
        return [{"action": "complete_missing_evidence", "reason": "补齐缺失的上游结果后重新生成报告。"}]
    if report_decision.get("adoption") in {"adopt_recommended", "adopt_specific_candidate"}:
        return [{"action": "proceed_to_human_review", "reason": "人工复核推荐候选模型、风险说明和下一步处理。"}]
    return [{"action": "accept_report", "reason": "人工复核并接受当前报告表达。"}]


def _safe_read_json(path: Any, run_dir: Path | None = None) -> dict[str, Any]:
    try:
        resolved = resolve_run_path(run_dir, path) if run_dir else Path(str(path))
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


def _feature_importance_components(path_value: Any, run_dir: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    if not path_value:
        return {}
    path = resolve_run_path(run_dir, path_value) if run_dir else Path(str(path_value))
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows or "component" not in rows[0]:
        return {}
    output: dict[str, list[dict[str, Any]]] = {}
    for component in sorted({row.get("component") or "unknown" for row in rows}):
        component_rows = [row for row in rows if (row.get("component") or "unknown") == component]
        component_rows.sort(key=lambda row: int(float(row.get("gain_rank") or 999999)))
        output[component] = [
            {
                "feature_name": row.get("feature_name"),
                "gain_share": row.get("gain_share"),
                "split_share": row.get("split_share"),
            }
            for row in component_rows[:5]
        ]
    return output


def _readable_model_name(row: dict[str, Any]) -> str:
    producer = str(row.get("producer_kind") or "")
    model_spec = row.get("model_spec") if isinstance(row.get("model_spec"), dict) else {}
    learner = str(model_spec.get("model_type") or row.get("learner") or "")
    if learner == "t_learner":
        if producer == "tuning":
            return "T-Learner Tuned"
        if producer == "modeling":
            return "T-Learner Baseline"
        return "T-Learner"
    if producer == "tuning":
        return "S-Learner Tuned"
    if producer == "modeling":
        return "S-Learner Baseline"
    return "S-Learner"


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
        "subject": "uplift_reporting",
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
        error={"code": "UPLIFT_REPORTING_FAILED", "message": f"{phase or 'action'} failed unexpectedly.", "recoverable": False, "retryable": False, "raw_error": str(exc)},
        progress=[{"step": phase or "unknown", "status": "failed", "message": str(exc)}],
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
    return {
        "status": result["status"],
        "summary": result["summary"],
        "outputs": result.get("outputs") or {},
        "issues": result.get("issues") or [],
        "next_steps": result.get("next_steps") or [],
    }


def _format_metric(value: Any) -> str:
    parsed = _number(value)
    return "N/A" if not math.isfinite(parsed) else f"{parsed:.6g}"


def _format_decimal(value: Any, digits: int) -> str:
    parsed = _number(value)
    return "N/A" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def _format_percent(value: Any) -> str:
    parsed = _number(value)
    return "N/A" if not math.isfinite(parsed) else f"{parsed:.2%}"


def _format_pp(value: Any) -> str:
    parsed = _number(value)
    return "N/A" if not math.isfinite(parsed) else f"{parsed * 100:.2f} pp"


def _format_count(value: Any) -> str:
    parsed = _number(value)
    return "N/A" if not math.isfinite(parsed) else f"{int(round(parsed)):,}"


def _number(value: Any) -> float:
    if value in (None, ""):
        return math.nan
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def _safe_read_csv(path_value: Any, run_dir: Path | None = None) -> list[dict[str, Any]]:
    if not path_value:
        return []
    path = resolve_run_path(run_dir, path_value) if run_dir else Path(str(path_value))
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except Exception:  # noqa: BLE001
        return []


def _missing_rate_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    by_feature: dict[str, float] = {}
    for row in rows:
        feature = str(row.get("feature") or "")
        if not feature:
            continue
        rate = _number(row.get("missing_rate"))
        if not math.isfinite(rate):
            continue
        by_feature[feature] = max(by_feature.get(feature, 0.0), rate)
    total = len(by_feature)
    if total == 0:
        return []
    buckets = [
        ("<= 50%", lambda value: value <= 0.5),
        ("> 50% and <= 80%", lambda value: 0.5 < value <= 0.8),
        ("> 80%", lambda value: value > 0.8),
    ]
    return [
        {"bucket": label, "count": count, "percentage": count / total}
        for label, predicate in buckets
        for count in [sum(1 for value in by_feature.values() if predicate(value))]
    ]


def _constant_or_all_missing_count(rows: list[dict[str, Any]]) -> int:
    features = set()
    for row in rows:
        status = str(row.get("status") or "").lower()
        severity = str(row.get("severity") or "").lower()
        if status in {"constant", "all_missing"} or severity in {"constant", "all_missing"}:
            features.add(str(row.get("feature") or ""))
    features.discard("")
    return len(features)


def _psi_distribution(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        return [], {}
    values = []
    max_row: dict[str, Any] = {}
    max_value = -math.inf
    for row in rows:
        psi = _number(row.get("psi"))
        if not math.isfinite(psi):
            continue
        values.append(psi)
        if psi > max_value:
            max_value = psi
            max_row = {**row, "psi": psi}
    total = len(values)
    if total == 0:
        return [], {}
    buckets = [
        ("<= 0.10", lambda value: value <= 0.10),
        ("> 0.10 and <= 0.25", lambda value: 0.10 < value <= 0.25),
        ("> 0.25", lambda value: value > 0.25),
    ]
    distribution = [
        {"bucket": label, "count": count, "percentage": count / total}
        for label, predicate in buckets
        for count in [sum(1 for value in values if predicate(value))]
    ]
    return distribution, max_row


def _split_sort_key(value: Any) -> tuple[int, str]:
    order = {"train": 0, "valid": 1, "validation": 1, "test": 2, "oot": 3}
    key = str(value or "").lower()
    return order.get(key, 99), key


def _value_or_na(value: Any) -> str:
    return "未提供" if value in (None, "") else str(value)


def _display_dataset_role(value: Any) -> str:
    mapping = {
        "train": "Train",
        "valid": "Validation",
        "validation": "Validation",
        "test": "Test",
        "oot": "OOT",
    }
    return mapping.get(str(value or "").lower(), _value_or_na(value))


def _display_feature_plan_status(value: Any) -> str:
    mapping = {
        "has_recommended_features": "recommended",
        "confirmed": "confirmed",
        "success": "available",
    }
    return mapping.get(str(value or ""), _value_or_na(value))


def _code_or_na(value: Any) -> str:
    return "`未提供`" if value in (None, "") else f"`{value}`"


def _join_preview(values: Any, *, limit: int) -> str:
    if not isinstance(values, list):
        return "未提供"
    items = [str(item) for item in values if item not in (None, "")]
    if not items:
        return "未提供"
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join([*items[:limit], "..."])


def _artifact_rows(items: list[tuple[str, Any]]) -> list[str]:
    rows = []
    for label, path in items:
        rows.append(f"- {label}: `{_compact_path(path)}`")
    return rows


def _compact_path(path_value: Any) -> str:
    if not path_value:
        return "未提供"
    path = Path(str(path_value).replace("\\", "/"))
    parts = path.parts
    if len(parts) <= 4:
        return str(path_value)
    return "/".join(parts[-4:])


def _zh_status(value: Any) -> str:
    mapping = {
        "complete": "完整",
        "missing": "缺失",
        "partial": "部分完整",
        "success": "完整",
        "partial_success": "部分完整",
    }
    return mapping.get(str(value), _value_or_na(value))


def _yes_no(value: Any, *, language: str) -> str:
    truthy = value is True or str(value).lower() == "true"
    if is_zh(language):
        return "是" if truthy else "否"
    return "Yes" if truthy else "No"


def _homogeneity_conclusion(facts: dict[str, Any], *, language: str) -> str:
    auc_values = [_number(row.get("auc")) for row in facts.get("auc_rows") or []]
    smd_values = [_number(row.get("max_abs_smd")) for row in facts.get("smd_rows") or []]
    max_auc = max([value for value in auc_values if math.isfinite(value)], default=math.nan)
    max_smd = max([value for value in smd_values if math.isfinite(value)], default=math.nan)
    status = facts.get("overall_status") or "unknown"
    if is_zh(language):
        parts = [f"同质性检验结论：当前 treatment/control 同质性诊断结果为 `{status}`。"]
        if math.isfinite(max_auc):
            parts.append(f"AUC 最大值为 `{max_auc:.4f}`，treatment/control 较难区分。")
        if math.isfinite(max_smd):
            ge_01 = sum(_int_count(row.get("variables_ge_0_1")) for row in facts.get("smd_rows") or [])
            ge_02 = sum(_int_count(row.get("variables_ge_0_2")) for row in facts.get("smd_rows") or [])
            parts.append(f"SMD 最大绝对值为 `{max_smd:.4f}`，其中 SMD >= 0.1 的变量数为 `{ge_01}`，SMD >= 0.2 的变量数为 `{ge_02}`。")
        if facts.get("recommended_action"):
            parts.append(f"建议动作：{_localized_action(facts.get('recommended_action'))}。")
        return "".join(parts)
    return f"Homogeneity status: `{status}`."


def _model_comparison_conclusion(
    rows: list[dict[str, Any]],
    recommended: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> str:
    if not rows:
        return ""
    names = "、".join(str(row.get("display_name")) for row in rows if row.get("display_name"))
    if not recommended:
        return f"本次对比包含 {len(rows)} 个候选模型：{names}。当前结果未给出推荐候选模型。"
    rec_value = _number(recommended.get("metric_value"))
    base_value = _number(baseline.get("metric_value")) if baseline else math.nan
    if baseline and math.isfinite(rec_value) and math.isfinite(base_value):
        delta = rec_value - base_value
        relative = delta / abs(base_value) if base_value else math.nan
        return (
            f"本次对比包含 {len(rows)} 个候选模型：{names}。按当前模型对比结果，推荐候选模型为 "
            f"{recommended.get('display_name')}。它在 {_display_dataset_role(recommended.get('dataset_role'))} 集上的 auuc_raw 为 "
            f"`{_format_decimal(rec_value, 4)}`，baseline 为 `{_format_decimal(base_value, 4)}`；"
            f"绝对提升 `{_format_decimal(delta, 4)}`，相对提升 `{_format_percent(relative)}`。"
        )
    return (
        f"本次对比包含 {len(rows)} 个候选模型：{names}。按当前模型对比结果，推荐候选模型为 "
        f"{recommended.get('display_name')}。"
    )


def _localized_rationale(value: Any, *, language: str) -> str:
    text = str(value or "")
    if not is_zh(language):
        return text
    if "no-improvement" in text or "tie-break" in text or "duplicate" in text:
        return "调参候选未带来可确认提升，当前推荐保持 baseline 候选。"
    return text or "见模型对比表。"


def _localized_action(value: Any) -> str:
    mapping = {
        "continue_to_feature_quality_analysis": "继续进入特征质量分析",
        "analyze_feature_quality": "继续进入特征质量分析",
        "proceed_to_human_review": "进入人工复核",
    }
    return mapping.get(str(value or ""), _value_or_na(value))


def _int_count(value: Any) -> int:
    parsed = _number(value)
    return int(round(parsed)) if math.isfinite(parsed) else 0


def _localized_risk(code: Any, *, language: str) -> str:
    value = str(code or "")
    if not is_zh(language):
        return value
    mapping = {
        "DUPLICATE_MODEL_PATH": "多个候选指向同一个模型文件，不能当作独立模型重复解读。",
        "DUPLICATE_MODEL_REF": "多个候选指向同一个模型文件，不能当作独立模型重复解读。",
        "HOLDOUT_USED_FOR_EARLY_STOPPING": "使用 holdout 数据参与 early stopping，可能影响泛化评估解释。",
        "HOLDOUT_USED_FOR_SELECTION": "使用 holdout 数据参与模型选择，可能带来选择偏差。",
        "FEATURE_PLAN_DIFFERS": "候选模型使用的特征方案不一致，特征解释不能直接等价比较。",
        "TASK_CONFIG_MISMATCH": "候选模型任务定义不一致，不能公平比较。",
        "EVALUATION_POPULATION_MISMATCH": "候选模型评估人群不一致，不能公平比较。",
    }
    return mapping.get(value, value or "存在上游风险，需要人工复核。")


def _selected_candidate_name(adoption: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    selected = str(adoption.get("selected_candidate_result_path") or "")
    for row in rows:
        if selected and str(row.get("producer_result_path") or "") == selected:
            return str(row.get("display_name") or "推荐候选模型")
    return "推荐候选模型"


def _section_source_coverage(sections: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    coverage = {}
    for section in sections:
        facts = section.get("facts") or {}
        source_count = 0
        for key, value in facts.items():
            if key.endswith("_path") and value:
                source_count += 1
        coverage[section["id"]] = {
            "status": section.get("status"),
            "source_count": source_count,
            "missing_evidence_count": 1 if section.get("status") == "missing" else 0,
        }
    return coverage


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
