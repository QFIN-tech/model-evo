# -*- coding: utf-8 -*-
"""_dataset._impute — 缺失值处理工具。

设计要点:
- NullAudit frozen dataclass + to_dict(): 单特征缺失处理摘要,
  下游可 JSON 序列化(落 config/report), 无需 ad-hoc 转换
- ImputeReport frozen dataclass: fit 阶段结构化报告
  (fitted_features/skipped_features/audits/n_indicators/max_miss_rate),
  下游直接消费结构化字段, 不必解析 imputer 内部状态
- 状态收口: audits/output_features/report/fitted 公开, _medians/_indicator_features 私有
- 单职责: 删 3 个未消费方法 summary_table / overview / fit_transform
  (全仓库零消费, 摘要落盘由 stages/ 阶段统一处理, 不在 imputer 里堆)
- transform 向量化: df.fillna(dict) 一次填完, 不走 per-column loop
- _should_add_indicator 抽出, 阈值判断单点化
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ── 默认阈值 ─────────────────────────────────────────────────────────
DEFAULT_INDICATOR_LOW = 0.05   # 缺失率下界：低于不加指示列
DEFAULT_INDICATOR_HIGH = 0.80  # 缺失率上界：高于不加指示列


@dataclass(frozen=True)
class NullAudit:
    """单特征缺失处理摘要 (immutable)。

    - frozen 防误改; to_dict() 兼容 JSON 序列化 (落 config/report)
    - has_indicator 显式标示是否生成指示列
    """
    feature: str
    miss_rate: float
    fill_value: float
    has_indicator: bool

    def to_dict(self) -> dict:
        """转 dict 用于 report.md / config.json 序列化。"""
        return {
            "feature": self.feature,
            "miss_rate": float(self.miss_rate),
            "fill_value": float(self.fill_value),
            "has_indicator": bool(self.has_indicator),
        }


@dataclass(frozen=True)
class ImputeReport:
    """fit 阶段的结构化报告 (immutable)。

    下游 (report/特征日志) 直接消费结构化字段, 不必解析 imputer 内部状态。
    """
    fitted_features: List[str]
    skipped_features: List[str] = field(default_factory=list)
    audits: List[NullAudit] = field(default_factory=list)
    n_indicators: int = 0
    max_miss_rate: float = 0.0

    def to_dict(self) -> dict:
        """转 dict 用于序列化。"""
        return {
            "fitted_features": list(self.fitted_features),
            "skipped_features": list(self.skipped_features),
            "audits": [a.to_dict() for a in self.audits],
            "n_indicators": int(self.n_indicators),
            "max_miss_rate": float(self.max_miss_rate),
        }


class DNNImputer:
    """缺失值处理器: fit 学习中位数与指示列集合, transform 输出扩展后的 DataFrame。

    - 状态收口: audits/output_features/report/fitted 公开 (结构化), _medians/_indicator_features 私有
    - fit 内联构建 ImputeReport, 不依赖外部读内部字段拼摘要
    - transform 向量化: df.fillna(dict) 一次填完, 不走 per-column loop
    - 单职责: 删 summary_table / overview / fit_transform (摘要落盘由 stages/ 处理)
    - _should_add_indicator 抽出, 阈值判断单点化
    """

    def __init__(
        self,
        features: List[str],
        indicator_low: float = DEFAULT_INDICATOR_LOW,
        indicator_high: float = DEFAULT_INDICATOR_HIGH,
    ) -> None:
        """构造缺失值处理器。

        Args:
            features: 待处理的原始特征列表
            indicator_low: 加指示列的缺失率下界
            indicator_high: 加指示列的缺失率上界
        """
        self.features: List[str] = list(features)
        self.indicator_low: float = float(indicator_low)
        self.indicator_high: float = float(indicator_high)

        # 私有 fit 状态
        self._medians: Dict[str, float] = {}
        self._indicator_features: List[str] = []

        # 公开 fit 状态
        self.audits: Dict[str, NullAudit] = {}
        self.output_features: List[str] = []
        self.report: "ImputeReport | None" = None
        self.fitted: bool = False

    # ── 私有 ─────────────────────────────────────────────────────────
    def _should_add_indicator(self, miss_rate: float) -> bool:
        """缺失率是否落入加指示列区间 [low, high]。"""
        return self.indicator_low <= miss_rate <= self.indicator_high

    @staticmethod
    def _safe_median(series: pd.Series) -> float:
        """计算中位数, 全空列/非有限值兜底为 0.0。"""
        if not series.notna().any():
            return 0.0
        median = float(series.median())
        if not np.isfinite(median):
            return 0.0
        return median

    # ── 公共接口 ─────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "DNNImputer":
        """基于训练集计算中位数并决定指示列集合, 内联构建 audits 与 report。

        Args:
            df: 训练集 DataFrame (须含 self.features 列)

        Returns:
            self (支持链式调用; 报告读 self.report)
        """
        self._medians.clear()
        self._indicator_features.clear()
        self.audits.clear()

        fitted: List[str] = []
        skipped: List[str] = []
        audit_list: List[NullAudit] = []
        max_miss = 0.0

        for col in self.features:
            if col not in df.columns:
                skipped.append(col)
                continue
            series = df[col]
            miss_rate = float(series.isna().mean())
            median = self._safe_median(series)
            has_indicator = self._should_add_indicator(miss_rate)

            self._medians[col] = median
            if has_indicator:
                self._indicator_features.append(col)

            audit = NullAudit(
                feature=col,
                miss_rate=miss_rate,
                fill_value=median,
                has_indicator=has_indicator,
            )
            self.audits[col] = audit
            audit_list.append(audit)
            fitted.append(col)
            max_miss = max(max_miss, miss_rate)

        # 输出特征: 原始特征 + 指示列 (顺序固定, 保证三段数据列对齐)
        self.output_features = fitted + [
            f"{c}_missing" for c in self._indicator_features
        ]
        self.report = ImputeReport(
            fitted_features=fitted,
            skipped_features=skipped,
            audits=audit_list,
            n_indicators=len(self._indicator_features),
            max_miss_rate=max_miss,
        )
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """对单段数据做填充 + 指示列生成。

        Args:
            df: 待处理 DataFrame (须含 fit 时 fitted_features 列)

        Returns:
            (扩展后的 DataFrame[output_features], output_features)
        """
        if not self.fitted:
            raise RuntimeError("DNNImputer 未 fit，请先调用 fit()")

        # 缺列检查: 防止 transform 时静默用 0 填充掩盖上游问题
        missing_cols = [c for c in self._medians if c not in df.columns]
        if missing_cols:
            raise KeyError(
                f"transform 缺列: {missing_cols} 不在 DataFrame 中"
            )

        # 先构造指示列 (在填充之前, 捕获原始 NaN 位置)
        indicator_df = pd.DataFrame(index=df.index)
        for col in self._indicator_features:
            indicator_df[f"{col}_missing"] = df[col].isna().astype(np.int8)

        # 向量化填充: 一次 dict.fillna 替代 per-column loop
        filled = df[list(self._medians.keys())].fillna(self._medians)

        if not indicator_df.empty:
            filled = pd.concat([filled, indicator_df], axis=1)

        return filled[self.output_features], list(self.output_features)


__all__ = [
    "DNNImputer",
    "NullAudit",
    "ImputeReport",
    "DEFAULT_INDICATOR_LOW",
    "DEFAULT_INDICATOR_HIGH",
]
