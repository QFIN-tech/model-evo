# -*- coding: utf-8 -*-
"""run 顶层整合报告: <run_dir>/report.md。

汇总本 run 的概览 / 三档指标 / 入模特征 Top20 / 特征重要性 Top20 / (可选) vs baseline 对比,
一页看懂。纯读已有产物(config.json + features/used-feature-list.csv +
explainability/feature-importance.csv + comparison/comparison_oot.md), 不引入新计算。

调用时机: 各入口(run_build / select_features / run_tuning)在 finalize_logs_stage 之后调,
确保所有 stage 产物齐备。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from stages.layout import RunLayout

_TOP_N = 20


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(x: Any, ndigits: int = 4) -> str:
    try:
        return f"{float(x):.{ndigits}f}"
    except (TypeError, ValueError):
        return "—"


def _format_train_detail(algo: str, runtime: dict) -> str:
    """按 algo 把训练细节格式化成一行展示。

    - xgb: best_iteration (early stopping 最优轮)
    - dnn: best_epoch / total_epochs / early_stopped
    - lr:  n_iter / converged
    """
    if algo == "dnn":
        best_epoch = runtime.get("best_epoch", "—")
        total = runtime.get("total_epochs", "—")
        stopped = runtime.get("early_stopped", "—")
        return f"best_epoch: **{best_epoch}** / total_epochs: **{total}** / early_stopped: **{stopped}**"
    if algo == "lr":
        n_iter = runtime.get("n_iter", "—")
        converged = runtime.get("converged", "—")
        return f"n_iter: **{n_iter}** / converged: **{converged}**"
    # xgb 或未知 algo 默认走 best_iteration
    best_iter = runtime.get("best_iteration", "—")
    return f"best_iteration: **{best_iter}**"


def _format_split_detail(runtime: dict) -> str:
    """把 split_report 摘要成一行展示。

    split_mode 取 runtime.split_mode (默认 internal);
    split_report 字段支持两套 schema:
    - pre-split schema (从 feature-analysis manifest 提取):
      strategy / counts / pos_rates / oot_boundary
    - internal schema (SplitReport.to_dict()):
      split_strategy / sample_counts / oot_boundary
    无 split_report 字段时兜底显示 "—"。
    """
    split_mode = runtime.get("split_mode", "internal")
    report = runtime.get("split_report") or {}
    strategy = report.get("strategy") or report.get("split_strategy", "—")
    counts = report.get("counts") or report.get("sample_counts") or {}
    boundary = report.get("oot_boundary", "—")
    n_train = counts.get("train", "—")
    n_val = counts.get("val", "—")
    n_oot = counts.get("oot", "—")
    return (
        f"split_mode: **{split_mode}** ({strategy}) | "
        f"train={n_train} / test={n_val} / oot={n_oot} | "
        f"oot_boundary: `{boundary}`"
    )


def _render_overview(cfg: dict) -> List[str]:
    run_name = cfg.get("run_name", "—")
    algo = cfg.get("algo", "—")
    label = cfg.get("label", "—")
    timestamp = cfg.get("timestamp", "—")
    produced_by = cfg.get("produced_by", "—")
    runtime = cfg.get("runtime") or {}
    n_feat = runtime.get("n_features", "—")
    train_detail = _format_train_detail(algo, runtime)
    split_detail = _format_split_detail(runtime)

    return [
        "## 模型概览",
        "",
        f"- run_name: `{run_name}`",
        f"- algo: **{algo}** / label: `{label}` / timestamp: `{timestamp}`",
        f"- produced_by: `{produced_by}`",
        f"- 入模特征数: **{n_feat}** / {train_detail}",
        f"- {split_detail}",
        "",
    ]


def _render_metrics_table(metrics: dict) -> List[str]:
    if not metrics:
        return ["## 三档评估指标", "", "（runtime.metrics 缺失）", ""]
    rows = [
        "## 三档评估指标",
        "",
        "| 集合 | AUC | KS | Gini |",
        "|------|-----|----|------|",
    ]
    for stage in ("train", "val", "oot"):
        m = metrics.get(stage) or {}
        stage_label = "test" if stage == "val" else stage
        rows.append(
            f"| {stage_label} | {_fmt(m.get('auc'))} | "
            f"{_fmt(m.get('ks'))} | {_fmt(m.get('gini'))} |"
        )
    rows.append("")
    return rows


def _render_top_features(features_csv: Path) -> List[str]:
    rows = _read_csv_rows(features_csv)
    names = [
        r.get("feature_name", "").strip()
        for r in rows
        if r.get("feature_name") and (r.get("status") or "kept") == "kept"
    ]
    total = len(names)
    out = ["## 入模特征 Top 20", ""]
    if not names:
        out += ["（used-feature-list.csv 缺失或为空）", ""]
        return out
    for i, name in enumerate(names[:_TOP_N], 1):
        out.append(f"{i}. `{name}`")
    if total > _TOP_N:
        out.append(f"\n> 共 {total} 个, 完整清单见 `features/used-feature-list.csv`")
    out.append("")
    return out


def _render_importance(imp_csv: Path) -> List[str]:
    rows = _read_csv_rows(imp_csv)
    if not rows:
        return [
            "## 特征重要性 Top 20",
            "",
            "（feature-importance.csv 缺失: 该算法未提供 importance）",
            "",
        ]
    out = [
        "## 特征重要性 Top 20",
        "",
        "| # | feature | importance |",
        "|---|---------|-----------|",
    ]
    for i, r in enumerate(rows[:_TOP_N], 1):
        feat = r.get("feature", "—")
        imp = _fmt(r.get("importance"), ndigits=6)
        out.append(f"| {i} | `{feat}` | {imp} |")
    out.append("")
    return out


def _render_baseline_compare(comparison_md: Path, runtime: dict) -> List[str]:
    baseline_run = runtime.get("baseline_run")
    if not baseline_run:
        return []  # baseline run 自身, 整段省略

    out = ["## vs baseline 对比", "", f"- baseline: `{baseline_run}`", ""]

    baseline_metrics = runtime.get("baseline_metrics") or {}
    cur_metrics = runtime.get("metrics") or {}

    if baseline_metrics and cur_metrics:
        out += [
            "| 集合 | baseline AUC | 本 run AUC | Δ AUC | baseline KS | 本 run KS | Δ KS |",
            "|------|-------------|-----------|-------|-------------|----------|------|",
        ]
        for stage in ("train", "val", "oot"):
            b = baseline_metrics.get(stage, {})
            c = cur_metrics.get(stage, {})
            b_auc = b.get("auc"); c_auc = c.get("auc")
            b_ks = b.get("ks"); c_ks = c.get("ks")
            try:
                d_auc = f"{float(c_auc) - float(b_auc):+.4f}"
            except (TypeError, ValueError):
                d_auc = "—"
            try:
                d_ks = f"{float(c_ks) - float(b_ks):+.4f}"
            except (TypeError, ValueError):
                d_ks = "—"
            stage_label = "test" if stage == "val" else stage
            out.append(
                f"| {stage_label} | {_fmt(b_auc)} | {_fmt(c_auc)} | {d_auc} | "
                f"{_fmt(b_ks)} | {_fmt(c_ks)} | {d_ks} |"
            )
        out.append("")

    if comparison_md.exists():
        out.append("> 详细分档对比见 `comparison/comparison_oot.md`")
        out.append("")

    return out


def write_run_summary_stage(
    layout: RunLayout,
    produced_by: Optional[str] = None,
) -> Path:
    """落 <run_dir>/report.md: 单 run 顶层整合报告。

    Args:
        layout: RunLayout(用 run_dir / features_dir / explainability_dir / comparison_dir)
        produced_by: 来源标识(用于日志; 报告内容本身从 config.json 读 produced_by)

    Returns:
        report.md 绝对路径
    """
    cfg = _read_json(layout.config_json) or {}
    runtime = cfg.get("runtime") or {}
    metrics = runtime.get("metrics") or {}

    features_csv = layout.features_dir / "used-feature-list.csv"
    imp_csv = layout.explainability_dir / "feature-importance.csv"
    comparison_md = layout.comparison_dir / "comparison_oot.md"

    L: List[str] = [f"# 单 run 报告 - {layout.run_name}", ""]
    L += _render_overview(cfg)
    L += _render_metrics_table(metrics)
    L += _render_top_features(features_csv)
    L += _render_importance(imp_csv)
    L += _render_baseline_compare(comparison_md, runtime)

    L += [
        "## 产物索引",
        "",
        f"- 配置快照: `config.json`",
        f"- 特征清单: `features/used-feature-list.csv` + `features/report.md`",
        f"- 模型文件: `model/`",
        f"- 评估报告: `evaluation/{layout.run_name}_{{train,test,oot}}_eval.md`",
        f"- 预测明细: `predictions/report.md`",
        f"- 可解释性: `explainability/feature-importance.csv`" +
        (" + `shap-summary.csv`" if (layout.explainability_dir / "shap-summary.csv").exists() else ""),
        f"- 日志: `logs/run.log`",
        "",
    ]

    report_path = layout.run_dir / "report.md"
    report_path.write_text("\n".join(L), encoding="utf-8")
    print(f"[run_summary] report.md 已落盘: {report_path} (produced_by={produced_by or 'default'})")
    return report_path
