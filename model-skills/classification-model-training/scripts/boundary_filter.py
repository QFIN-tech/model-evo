# -*- coding: utf-8 -*-
"""边界特征过滤: 常量 / 泄漏 / ID-like / 全缺失 四条独立规则, 可分别启停 + 配阈值。

本模块做"安全过滤": 剔除会让训练失败或泄漏的特征 (数据安全/正确性问题);
基于 IV/PSI 的优化筛选由 classification-model-tuning/scripts/selection_rules.py 负责。

每条规则只负责"返回该规则要剔除的 feature 集合", 编排在 filter_boundary_features() 里做并集 + 保序。
读 feature-analysis 落的 csv (stats/feature-quality/iv_table), 不依赖 markdown 报告。

上游 csv 缺失时跳过该规则, 不阻断训练。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


# 阈值默认值
DEFAULT_IV_MAX = 1.0              # IV > 1.0 视为泄漏 (特征穿越), 业内常见截点
DEFAULT_CONST_UNIQUE_MAX = 1      # unique <= 1 视为常量 (只有 1 个值或 0 个)
DEFAULT_ID_LIKE_RATIO = 0.9       # unique / sample_total > 0.9 视为 ID 类 (几乎一行一值)
DEFAULT_MISSING_MAX = 1.0         # missing_rate >= 1.0 视为全缺失


@dataclass(frozen=True)
class BoundaryFilterResult:
    """边界过滤结果, 用于落 config.json runtime + 给 features/report.md 渲染。"""
    kept_features: List[str]
    dropped_features: List[str]
    dropped_by_rule: Dict[str, List[str]]   # rule_name -> dropped features (保序)
    thresholds: Dict[str, float]
    rules_enabled: Dict[str, bool]
    sample_total: int                       # 用于 ID-like 比率分母, 落 manifest 供溯源
    n_before: int                           # 过滤前的原始 features 数

    def as_dict(self) -> dict:
        return {
            "kept_features": list(self.kept_features),
            "dropped_features": list(self.dropped_features),
            "dropped_by_rule": {k: list(v) for k, v in self.dropped_by_rule.items()},
            "thresholds": dict(self.thresholds),
            "rules_enabled": dict(self.rules_enabled),
            "sample_total": self.sample_total,
            "n_before": self.n_before,
        }


def apply_constant_rule(
    features: List[str],
    stats_df: pd.DataFrame,
    unique_max: int = DEFAULT_CONST_UNIQUE_MAX,
) -> Set[str]:
    """剔除常量特征: unique <= unique_max 或 std == 0。任一命中即删。

    stats_df 为空表或缺少 unique/std 列时不剔除 (warn-and-skip)。
    """
    if stats_df is None or stats_df.empty:
        return set()
    if "unique" not in stats_df.columns and "std" not in stats_df.columns:
        return set()
    feat_set = set(features)
    bad: Set[str] = set()
    if "unique" in stats_df.columns:
        bad |= set(
            stats_df[(stats_df["unique"].notna()) & (stats_df["unique"] <= unique_max)]["feature"].tolist()
        )
    if "std" in stats_df.columns:
        bad |= set(
            stats_df[(stats_df["std"].notna()) & (stats_df["std"] == 0)]["feature"].tolist()
        )
    return {f for f in bad if f in feat_set}


def apply_leakage_rule(
    features: List[str],
    iv_df: pd.DataFrame,
    iv_max: float = DEFAULT_IV_MAX,
) -> Set[str]:
    """剔除泄漏特征: IV > iv_max (特征穿越)。

    IV NaN 不删。iv_df 为空表或缺少 iv 列时不剔除 (warn-and-skip)。
    """
    if iv_df is None or iv_df.empty or "iv" not in iv_df.columns:
        return set()
    feat_set = set(features)
    bad = iv_df[(iv_df["iv"].notna()) & (iv_df["iv"] > iv_max)]["feature"].tolist()
    return {f for f in bad if f in feat_set}


def apply_id_like_rule(
    features: List[str],
    stats_df: pd.DataFrame,
    sample_total: int,
    ratio: float = DEFAULT_ID_LIKE_RATIO,
) -> Set[str]:
    """剔除 ID 类特征: unique / sample_total > ratio。

    sample_total <= 0 时跳过 (无法计算比率)。
    stats_df 为空表或缺少 unique 列时不剔除 (warn-and-skip)。
    """
    if sample_total <= 0:
        return set()
    if stats_df is None or stats_df.empty or "unique" not in stats_df.columns:
        return set()
    feat_set = set(features)
    bad = stats_df[
        (stats_df["unique"].notna()) & (stats_df["unique"] / sample_total > ratio)
    ]["feature"].tolist()
    return {f for f in bad if f in feat_set}


def apply_all_missing_rule(
    features: List[str],
    stats_df: pd.DataFrame,
    missing_max: float = DEFAULT_MISSING_MAX,
) -> Set[str]:
    """剔除全缺失特征: missing_rate >= missing_max。

    stats_df 为空表或缺少 missing_rate 列时不剔除 (warn-and-skip)。
    """
    if stats_df is None or stats_df.empty or "missing_rate" not in stats_df.columns:
        return set()
    feat_set = set(features)
    bad = stats_df[
        (stats_df["missing_rate"].notna()) & (stats_df["missing_rate"] >= missing_max)
    ]["feature"].tolist()
    return {f for f in bad if f in feat_set}


def _read_csv(path: Path) -> pd.DataFrame:
    """容错读 csv: 文件不存在返回空 DataFrame, 让规则自动跳过。"""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def filter_boundary_features(
    baseline_features: List[str],
    analysis_dir: str,
    *,
    sample_total: int,
    enable_constant: bool = True,
    enable_leakage: bool = True,
    enable_id_like: bool = True,
    enable_all_missing: bool = True,
    iv_max: float = DEFAULT_IV_MAX,
    const_unique_max: int = DEFAULT_CONST_UNIQUE_MAX,
    id_like_ratio: float = DEFAULT_ID_LIKE_RATIO,
    missing_max: float = DEFAULT_MISSING_MAX,
) -> BoundaryFilterResult:
    """按启用规则做边界过滤, 返回保留/剔除/分规则细节。

    Args:
        baseline_features: 原始特征列表 (保序)
        analysis_dir: feature-analysis 报告目录, 内含 stats.csv / feature-quality.csv / iv_table.csv
        sample_total: 样本总量 (取自 feature-analysis manifest 的 overview.n_total), 用于 ID-like 比率分母
        enable_*: 各规则开关
        *_threshold: 对应阈值

    Returns:
        BoundaryFilterResult: kept 保序; dropped 也保序 (按 baseline_features 顺序)。
    """
    ana = Path(analysis_dir)
    stats_df = _read_csv(ana / "stats.csv")
    # leakage 优先读 feature-quality.csv (含 iv 列), 空时回退 iv_table.csv
    iv_df = _read_csv(ana / "feature-quality.csv")
    if iv_df.empty or "iv" not in iv_df.columns:
        iv_df = _read_csv(ana / "iv_table.csv")

    # warn-and-skip: 关键 csv 缺失时打印警告
    if stats_df.empty:
        warnings.warn(
            f"[boundary_filter] stats.csv 缺失或为空 ({ana / 'stats.csv'}), "
            f"constant/id_like/all_missing 规则将跳过",
            stacklevel=2,
        )
    if iv_df.empty:
        warnings.warn(
            f"[boundary_filter] feature-quality.csv 与 iv_table.csv 均缺失或无 iv 列 "
            f"({ana}), leakage 规则将跳过",
            stacklevel=2,
        )
    if sample_total <= 0 and enable_id_like:
        warnings.warn(
            f"[boundary_filter] sample_total={sample_total} 非正数, id_like 规则将跳过",
            stacklevel=2,
        )

    dropped_by_rule: Dict[str, List[str]] = {}
    drop_union: Set[str] = set()

    if enable_constant:
        s = apply_constant_rule(baseline_features, stats_df, const_unique_max)
        dropped_by_rule["constant"] = [f for f in baseline_features if f in s]
        drop_union |= s
    if enable_leakage:
        s = apply_leakage_rule(baseline_features, iv_df, iv_max)
        dropped_by_rule["leakage"] = [f for f in baseline_features if f in s]
        drop_union |= s
    if enable_id_like:
        s = apply_id_like_rule(baseline_features, stats_df, sample_total, id_like_ratio)
        dropped_by_rule["id_like"] = [f for f in baseline_features if f in s]
        drop_union |= s
    if enable_all_missing:
        s = apply_all_missing_rule(baseline_features, stats_df, missing_max)
        dropped_by_rule["all_missing"] = [f for f in baseline_features if f in s]
        drop_union |= s

    kept = [f for f in baseline_features if f not in drop_union]
    dropped = [f for f in baseline_features if f in drop_union]

    return BoundaryFilterResult(
        kept_features=kept,
        dropped_features=dropped,
        dropped_by_rule=dropped_by_rule,
        thresholds={
            "iv_max": iv_max,
            "const_unique_max": const_unique_max,
            "id_like_ratio": id_like_ratio,
            "missing_max": missing_max,
        },
        rules_enabled={
            "constant": enable_constant,
            "leakage": enable_leakage,
            "id_like": enable_id_like,
            "all_missing": enable_all_missing,
        },
        sample_total=sample_total,
        n_before=len(baseline_features),
    )
