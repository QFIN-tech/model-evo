# -*- coding: utf-8 -*-
"""features/ 子目录产出: 实际入模特征清单 + 报告。

入参是经过清洗(object dtype 转 float / 来源去重)后的最终 features 列表,
独立写盘以便下游(model-recommend / 特征贡献分析)直接消费。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from stages.layout import RunLayout, write_manifest


def _write_csv(
    features: Sequence[str],
    out_path: Path,
    dropped_by_rule: Optional[Dict[str, Iterable[str]]] = None,
) -> None:
    """三列 csv: feature_name / status / dropped_by_rule。

    - feature_name: 入模特征
    - status: kept(入模) / dropped_<rule>(被某规则剔除)
    - dropped_by_rule: 触发的规则名(仅 dropped 行非空)
    """
    rule_lookup: Dict[str, str] = {}
    if dropped_by_rule:
        for rule, feats in dropped_by_rule.items():
            for f in feats:
                rule_lookup.setdefault(f, rule)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name", "status", "dropped_by_rule"])
        for name in features:
            writer.writerow([name, "kept", ""])
        for name, rule in rule_lookup.items():
            if name in features:
                continue
            writer.writerow([name, f"dropped_{rule}", rule])


def _render_report(
    features: Sequence[str],
    upstream_source: Optional[str],
    dropped: Iterable[str],
    run_name: str,
    dropped_by_rule: Optional[Dict[str, Iterable[str]]] = None,
) -> str:
    """渲染 features/report.md。

    Args:
        features: 实际入模的最终特征列表
        upstream_source: 上游来源文件(feature-matching 的 feature-list.csv 路径或用户指定)
        dropped: 清洗阶段被剔除的特征(空集为常态)
        run_name: 本次 run 目录名
        dropped_by_rule: 边界过滤按规则剔除的明细 {rule: [features]}(可为 None)
    """
    dropped_list = list(dropped)
    dropped_by_rule = dropped_by_rule or {}
    n_dropped_by_rule = sum(len(v) for v in dropped_by_rule.values())
    L = [f"# 特征清单 - {run_name}", ""]
    L += ["## 概述", "",
          f"- 入模特征数: **{len(features)}**",
          f"- 来源: `{upstream_source or '(yaml model.features 直接指定)'}`",
          f"- 清洗剔除数: {len(dropped_list)}",
          f"- 边界过滤剔除数: {n_dropped_by_rule}", ""]
    if dropped_list:
        L += ["## 剔除清单(NaN/常数/dtype 不可转)", ""]
        for f in dropped_list:
            L.append(f"- `{f}`")
        L.append("")
    if dropped_by_rule:
        L += ["## 边界特征过滤", "",
              "剔除会让训练失败或泄漏的特征(常量/泄漏/ID-like/全缺失)。", "",
              "| 规则 | 剔除数 |", "|------|--------|"]
        for rule, feats in dropped_by_rule.items():
            if feats:
                L.append(f"| {rule} | {len(feats)} |")
        L.append("")
        L += ["### 被剔除特征(按规则分组)", ""]
        for rule, feats in dropped_by_rule.items():
            if not feats:
                continue
            L += [f"#### {rule}", ""]
            for f in feats:
                L.append(f"- `{f}`")
            L.append("")
    L += ["## 入模特征(前 20)", ""]
    for f in features[:20]:
        L.append(f"- `{f}`")
    if len(features) > 20:
        L.append(f"- ... (共 {len(features)} 个,完整清单见 `used-feature-list.csv`)")
    L.append("")
    return "\n".join(L)


def write_features_stage(
    layout: RunLayout,
    features: Sequence[str],
    upstream_source: Optional[str] = None,
    dropped: Optional[Iterable[str]] = None,
    dropped_by_rule: Optional[Dict[str, Iterable[str]]] = None,
    produced_by: Optional[str] = None,
) -> None:
    """落 features/ 阶段产物: used-feature-list.csv + report.md + _manifest.json。

    Args:
        layout: RunLayout 实例
        features: 实际入模特征(顺序与训练一致)
        upstream_source: 上游来源描述(如 feature-matching 的 feature-list.csv 路径);可为 None
        dropped: 训练前清洗阶段被剔除的特征(可为 None)
        dropped_by_rule: select_features 按规则剔除的明细 {rule: [features]}(可为 None);
            传入时 csv 增加 dropped 行, status=dropped_<rule>
        produced_by: manifest 来源标识(下游 skill 复用本函数时传值)
    """
    dropped = list(dropped or [])
    csv_path = layout.features_dir / "used-feature-list.csv"
    report_path = layout.features_dir / "report.md"
    _write_csv(features, csv_path, dropped_by_rule=dropped_by_rule)
    report_path.write_text(
        _render_report(
            features, upstream_source, dropped, layout.run_name,
            dropped_by_rule=dropped_by_rule,
        ),
        encoding="utf-8",
    )
    n_dropped_by_rule = sum(len(v) for v in (dropped_by_rule or {}).values())
    write_manifest(
        layout.features_dir,
        stage="features",
        files=[csv_path, report_path],
        extra={
            "n_features": len(features),
            "upstream_source": upstream_source,
            "n_dropped": len(dropped),
            "n_dropped_by_rule": n_dropped_by_rule,
        },
        produced_by=produced_by,
    )
