# -*- coding: utf-8 -*-
"""predictions/ 子目录产出: train/test/oot 三档 predictions parquet + report.md。

数据安全:
- predictions parquet **含原始 user_no/id_cols + 标签 + 分数 + 分档**
- 只允许落到 session 输出目录(由 CLAUDE.md/`runs/*` 已 gitignore 兜底);严禁外发或入库
- report.md 仅展示聚合分布(直方图区间 + 分档),不展示明细
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from stages.eval_data import EvalData
from stages.layout import RunLayout, write_manifest


def _bucket_assign(scores: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """按分数从高到低等频分档(1=高风险, n=低风险),返回 int 档号。"""
    if len(scores) == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(-scores, kind="mergesort")  # 风险高 -> 低
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return (ranks * n_bins // len(scores)).astype(int) + 1


def _score_dist_table(scores: np.ndarray) -> pd.DataFrame:
    """得分按 [0, 0.1, 0.2, ..., 1.0] 落直方图,返回 count / pct。"""
    bins = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(scores, bins=bins)
    total = counts.sum()
    rows = []
    for i, c in enumerate(counts):
        rows.append({
            "range": f"[{bins[i]:.1f}, {bins[i+1]:.1f}{')' if i < 9 else ']'}",
            "count": int(c),
            "pct": float(c) / total if total else 0.0,
        })
    return pd.DataFrame(rows)


def _build_pred_df(
    df: pd.DataFrame, score: np.ndarray, target: str, id_cols: List[str],
) -> pd.DataFrame:
    """从源 df + score 组装 [id_cols..., label, score, bucket] 明细表。

    标签列固定写为 `label`, 无论源表是 `label` / `TARGET` / 其他; 下游
    invoke/evaluation.py 以 `label` 为硬约定读 parquet。
    """
    cols_keep = [c for c in id_cols if c in df.columns]
    out_df = df[cols_keep].copy() if cols_keep else pd.DataFrame(index=df.index)
    out_df["label"] = df[target].to_numpy()
    out_df["score"] = score
    out_df["bucket"] = _bucket_assign(score, n_bins=10)
    return out_df


def _render_md(
    run_name: str,
    id_cols: List[str],
    files: dict,
    sample_counts: dict,
    score_dists: dict,
) -> str:
    L: List[str] = [f"# 预测明细 - {run_name}", ""]
    L += ["## 安全提示", "",
          "- predictions parquet **含原始 id 列 + 标签 + 分数 + 分档**",
          f"- id 列: `{', '.join(id_cols)}`" if id_cols else "- id 列: (无)",
          "- 只允许在本 session 内消费; 严禁外发 / 提交 git / 落到共享目录", ""]
    L += ["## 概述", "",
          f"- train 样本数: **{sample_counts['train']}**  文件: `{files['train']}`",
          f"- test  样本数: **{sample_counts['test']}**  文件: `{files['test']}`",
          f"- oot   样本数: **{sample_counts['oot']}**  文件: `{files['oot']}`",
          "- 分档: 按分数从高到低等频 10 档 (1=高风险, 10=低风险)", ""]
    for stage in ("train", "test", "oot"):
        L += [f"## {stage.upper()} 分数分布 (直方图)", "",
              "| 区间 | 数量 | 占比 |", "|------|------|------|"]
        for _, r in score_dists[stage].iterrows():
            L.append(f"| {r['range']} | {r['count']} | {r['pct']:.2%} |")
        L.append("")
    return "\n".join(L)


def write_predictions_stage(
    layout: RunLayout,
    data: EvalData,
    id_cols: Optional[List[str]] = None,
    produced_by: Optional[str] = None,
) -> Path:
    """落 predictions/ 阶段产物: train/test/oot 三档 parquet + report.md + _manifest.json。

    Args:
        layout: RunLayout
        data: EvalData(用其 train/val/oot 的 df + score + target)
        id_cols: 落盘的 id 列(从配置 model.id_cols 传入); 缺失列自动跳过

    Returns:
        oot_predictions.parquet 路径(主明细)
    """
    id_cols = list(id_cols or [])

    stages = [
        ("train", data.train_df, data.train_score, "train_predictions.parquet"),
        ("test", data.val_df, data.val_score, "test_predictions.parquet"),
        ("oot", data.oot_df, data.oot_score, "oot_predictions.parquet"),
    ]

    files: dict = {}
    sample_counts: dict = {}
    score_dists: dict = {}
    all_paths: List[Path] = []

    for stage, df, score, fname in stages:
        out_df = _build_pred_df(df, score, data.target, id_cols)
        path = layout.predictions_dir / fname
        out_df.to_parquet(path, index=False)
        files[stage] = fname
        sample_counts[stage] = int(len(out_df))
        score_dists[stage] = _score_dist_table(score)
        all_paths.append(path)

    oot_parquet_path = layout.predictions_dir / "oot_predictions.parquet"

    md_path = layout.predictions_dir / "report.md"
    md_path.write_text(
        _render_md(layout.run_name, id_cols, files, sample_counts, score_dists),
        encoding="utf-8",
    )
    all_paths.append(md_path)
    write_manifest(
        layout.predictions_dir,
        stage="predictions",
        files=all_paths,
        extra={
            "n_samples": sample_counts,
            "id_cols": id_cols,
            "contains_raw_id": bool(id_cols),
            "schema": ["id_cols...", "label", "score", "bucket"],
            "files_by_stage": files,
        },
        produced_by=produced_by,
    )
    return oot_parquet_path
