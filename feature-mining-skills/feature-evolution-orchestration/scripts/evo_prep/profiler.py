# -*- coding: utf-8 -*-
"""特征画像: feature-profile.csv + data-desc.md。

data-desc.md 为特征画像
(每特征一行: null_rate/min/中位/max 的 markdown 表), 后续补上 dtype/分位数与特征含义,
供轮次简报(round-brief)与候选生成参考。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_feature_profile(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """全样本特征画像。

    Returns:
        DataFrame, 列: feature / dtype / null_rate / n_unique /
        min / p25 / median / p75 / max / mean / std (数值列才有统计量)
    """
    rows = []
    n = len(df)
    for c in feature_cols:
        s = df[c]
        row = {
            "feature": c,
            "dtype": str(s.dtype),
            "null_rate": round(float(s.isna().mean()), 6) if n else 0.0,
            "n_unique": int(s.nunique(dropna=True)),
        }
        sn = pd.to_numeric(s, errors="coerce")
        if sn.notna().any():
            row.update(
                {
                    "min": float(sn.min()),
                    "p25": float(sn.quantile(0.25)),
                    "median": float(sn.quantile(0.50)),
                    "p75": float(sn.quantile(0.75)),
                    "max": float(sn.max()),
                    "mean": float(sn.mean()),
                    "std": float(sn.std()) if sn.notna().sum() > 1 else 0.0,
                }
            )
        else:
            row.update({k: np.nan for k in ("min", "p25", "median", "p75", "max", "mean", "std")})
        rows.append(row)
    return pd.DataFrame(rows)


def load_feature_metadata(path: str | None) -> dict:
    """读特征元数据 csv → {feature: description}。

    metadata 文件格式(2 或 3 列): 前两列按 (feature, description) 解析,
    有表头自动跳过(feature/feature_name/名称 等常见表头)。
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("feature_metadata 文件不存在: %s" % path)
    meta = pd.read_csv(p, encoding="utf-8-sig", header=None)
    if meta.shape[1] < 2:
        raise ValueError("feature_metadata 须至少两列(feature, description): %s" % path)
    header_tokens = {"feature", "feature_name", "name", "特征", "字段", "名称"}
    first = str(meta.iloc[0, 0]).strip().lower()
    if first in header_tokens:
        meta = meta.iloc[1:]
    out = {}
    for _, r in meta.iterrows():
        key = str(r.iloc[0]).strip()
        if key and key.lower() != "nan":
            out[key] = str(r.iloc[1]).strip()
    return out


def build_data_desc_md(
    profile_df: pd.DataFrame,
    metadata: dict | None = None,
    text_cols: list | None = None,
    text_note: str = "文本列(经 semantic-feature 语义特征化后使用)",
) -> str:
    """data-desc.md: 特征画像 markdown 表。

    含含义列(有元数据时)与文本列段; 供 round-brief 引用, 也供 Agent 生成候选时查字段。
    """
    metadata = metadata or {}
    lines = [
        "# 特征画像(data-desc)",
        "",
        "| 特征名称 | 类型 | null_rate(%) | 样本最小值 | 样本中位数 | 样本最大值 | 含义 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in profile_df.iterrows():

        def _fmt(v):
            return "nan" if pd.isna(v) else ("%.4g" % float(v))

        null_pct = "%.2f" % (float(r["null_rate"]) * 100)
        desc = metadata.get(r["feature"], "")
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (r["feature"], r["dtype"], null_pct, _fmt(r["min"]), _fmt(r["median"]), _fmt(r["max"]), desc)
        )
    for c in text_cols or []:
        lines.append("| %s | text | - | - | - | - | %s |" % (c, metadata.get(c, text_note)))
    lines.append("")
    return "\n".join(lines)
