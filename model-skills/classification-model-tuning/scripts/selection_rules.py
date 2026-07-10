# -*- coding: utf-8 -*-
"""特征筛选规则: 高 PSI / 低 IV / 高缺失率 三条独立规则, 可分别启停 + 配阈值。

每条规则只负责"返回该规则要剔除的 feature 集合", 编排在 select() 里做并集 + 保序。
读 feature-analysis 落的 csv (stats/iv/psi), 不依赖 markdown 报告。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# 阈值默认值, 跟 CLAUDE.md / feature-analysis 报告口径对齐。
DEFAULT_PSI_THRESHOLD = 0.10        # 与 CLAUDE.md 红线一致, > 0.1 视为不稳定
DEFAULT_IV_THRESHOLD = 0.02         # IV < 0.02 几乎无区分度, 业内通用截点
DEFAULT_MISSING_THRESHOLD = 0.95    # 缺失率 > 95% 的列, 即便有信号也很难用


@dataclass(frozen=True)
class SelectionResult:
    """筛选结果, 用于落 config.json runtime + 给下游训练。"""
    kept_features: List[str]
    dropped_features: List[str]
    dropped_by_rule: Dict[str, List[str]]   # rule_name -> dropped features
    thresholds: Dict[str, float]
    rules_enabled: Dict[str, bool]

    def as_dict(self) -> dict:
        return {
            "kept_features": list(self.kept_features),
            "dropped_features": list(self.dropped_features),
            "dropped_by_rule": {k: list(v) for k, v in self.dropped_by_rule.items()},
            "thresholds": dict(self.thresholds),
            "rules_enabled": dict(self.rules_enabled),
        }


def apply_high_psi_rule(
    features: List[str], psi_df: pd.DataFrame, threshold: float = DEFAULT_PSI_THRESHOLD
) -> Set[str]:
    """剔除 train→oot PSI > threshold 的特征。psi_df 可能为空表(无 oot)→ 不剔除。"""
    if psi_df is None or psi_df.empty or "psi" not in psi_df.columns:
        return set()
    feat_set = set(features)
    bad = psi_df[(psi_df["psi"].notna()) & (psi_df["psi"] > threshold)]["feature"].tolist()
    return {f for f in bad if f in feat_set}


def apply_low_iv_rule(
    features: List[str], iv_df: pd.DataFrame, threshold: float = DEFAULT_IV_THRESHOLD
) -> Set[str]:
    """剔除 IV < threshold 的特征。IV 缺失(NaN) 时按"低 IV"处理, 一并剔除。"""
    if iv_df is None or iv_df.empty or "iv" not in iv_df.columns:
        return set()
    feat_set = set(features)
    low_iv = iv_df[(iv_df["iv"].isna()) | (iv_df["iv"] < threshold)]["feature"].tolist()
    return {f for f in low_iv if f in feat_set}


def apply_high_missing_rule(
    features: List[str],
    stats_df: pd.DataFrame,
    threshold: float = DEFAULT_MISSING_THRESHOLD,
) -> Set[str]:
    """剔除 missing_rate > threshold 的特征。"""
    if stats_df is None or stats_df.empty or "missing_rate" not in stats_df.columns:
        return set()
    feat_set = set(features)
    bad = stats_df[
        (stats_df["missing_rate"].notna()) & (stats_df["missing_rate"] > threshold)
    ]["feature"].tolist()
    return {f for f in bad if f in feat_set}


def _read_csv(path: Path) -> pd.DataFrame:
    """容错读 csv: 文件不存在返回空 DataFrame, 让规则自动跳过。"""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def select(
    baseline_features: List[str],
    analysis_dir: str,
    *,
    enable_psi: bool = True,
    enable_iv: bool = True,
    enable_missing: bool = True,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    iv_threshold: float = DEFAULT_IV_THRESHOLD,
    missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
) -> SelectionResult:
    """按启用规则做特征筛选, 返回保留/剔除/分规则细节。

    Args:
        baseline_features: baseline run 用过的特征列表(保序)
        analysis_dir: feature-analysis 报告目录, 内含 stats.csv / iv_table.csv / psi_table.csv
        enable_*: 各规则开关
        *_threshold: 对应阈值

    Returns:
        SelectionResult: kept 保序; dropped 也保序(按 baseline_features 顺序)。
    """
    ana = Path(analysis_dir)
    stats_df = _read_csv(ana / "stats.csv")
    iv_df = _read_csv(ana / "iv_table.csv")
    psi_df = _read_csv(ana / "psi_table.csv")

    dropped_by_rule: Dict[str, List[str]] = {}
    drop_union: Set[str] = set()

    if enable_psi:
        s = apply_high_psi_rule(baseline_features, psi_df, psi_threshold)
        dropped_by_rule["high_psi"] = [f for f in baseline_features if f in s]
        drop_union |= s
    if enable_iv:
        s = apply_low_iv_rule(baseline_features, iv_df, iv_threshold)
        dropped_by_rule["low_iv"] = [f for f in baseline_features if f in s]
        drop_union |= s
    if enable_missing:
        s = apply_high_missing_rule(baseline_features, stats_df, missing_threshold)
        dropped_by_rule["high_missing"] = [f for f in baseline_features if f in s]
        drop_union |= s

    kept = [f for f in baseline_features if f not in drop_union]
    dropped = [f for f in baseline_features if f in drop_union]

    return SelectionResult(
        kept_features=kept,
        dropped_features=dropped,
        dropped_by_rule=dropped_by_rule,
        thresholds={
            "psi": psi_threshold,
            "iv": iv_threshold,
            "missing": missing_threshold,
        },
        rules_enabled={
            "high_psi": enable_psi,
            "low_iv": enable_iv,
            "high_missing": enable_missing,
        },
    )
