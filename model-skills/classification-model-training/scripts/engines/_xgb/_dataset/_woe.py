# -*- coding: utf-8 -*-
"""WoE (Weight of Evidence) 编码器。

设计要点:
- WoEFeatureMap dataclass: 单特征完整映射 (bin_edges/woe/counts/event_rates/iv),
  fit 一次缓存好, 不依赖 _binners[feat].binning_table.build() 反复重建
- WoEReport dataclass: fit 返回结构化报告 (fitted/skipped/iv_ranking),
  下游直接消费结构化字段, 不靠 log print 传信息
- 状态收口: feature_maps 公开结构化缓存, _binners 私有不暴露
- 单职责: 只做 WoE 编码, 删 iv_filter 筛选分支 (IV 筛选属于特征工程, 不在编码器里做)
- 删 3 个未消费方法: get_iv_summary / get_scorecard_table / fit_transform
  (本仓库零消费, 需要时单独建评分卡模块, 不在编码器里堆积)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from optbinning import OptimalBinning
    HAS_OPTBINNING = True
except ImportError:
    HAS_OPTBINNING = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WoEFeatureMap:
    """单特征的 WoE 完整映射 (immutable)。

    - 字段完整: 含 counts/event_rates, 不只 woe_values/bin_edges
    - frozen 防误改; to_dict() 兼容 JSON 序列化 (落 config/report)
    - fit 一次缓存好, 不依赖 binning_table.build() 反复重建
    """

    feature: str
    iv: float
    n_bins: int
    bin_edges: List[str]
    woe_values: List[float]
    counts: List[int]
    event_rates: List[float]
    is_fitted: bool = True

    def to_dict(self) -> dict:
        """转 dict 用于 report.md / config.json 序列化。"""
        return {
            "feature": self.feature,
            "iv": float(self.iv),
            "n_bins": int(self.n_bins),
            "bin_edges": list(self.bin_edges),
            "woe_values": [float(v) for v in self.woe_values],
            "counts": [int(c) for c in self.counts],
            "event_rates": [float(r) for r in self.event_rates],
        }


@dataclass(frozen=True)
class WoEReport:
    """fit 阶段的结构化报告 (immutable)。

    下游 (train_lr / report) 直接消费结构化字段, 不必解析 log。
    """

    fitted_features: List[str]
    skipped_features: List[str]
    iv_ranking: List[tuple] = field(default_factory=list)  # [(feat, iv), ...] 按 IV 降序

    def to_dict(self) -> dict:
        """转 dict 用于序列化。"""
        return {
            "fitted_features": list(self.fitted_features),
            "skipped_features": list(self.skipped_features),
            "iv_ranking": [(f, float(v)) for f, v in self.iv_ranking],
        }


class WoeBinner:
    """WoE 编码器: fit 学习分箱映射, transform 输出 WoE 编码后的 DataFrame。

    - 状态收口: feature_maps 公开结构化缓存 (WoEFeatureMap), _binners 私有
    - fit 内联构建 WoEReport, 不依赖 log print 传递信息
    - 单职责: 删 iv_filter 筛选 (特征工程不在编码器做), 删 3 个未消费方法
    - transform metric 校验: 只接受 "woe" / "event_rate"
    """

    def __init__(
        self,
        max_n_bins: int = 8,
        min_bin_size: float = 0.05,
    ) -> None:
        """构造 WoE 编码器。

        Args:
            max_n_bins: 最大分箱数
            min_bin_size: 最小分箱比例
        """
        if not HAS_OPTBINNING:
            raise ImportError(
                "WoeBinner 依赖 optbinning 库，请安装: pip install optbinning"
            )
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size

        self._binners: Dict[str, OptimalBinning] = {}
        self.feature_maps: Dict[str, WoEFeatureMap] = {}
        self.fitted_features: List[str] = []
        self.report: Optional[WoEReport] = None
        self._is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y,
        features: Optional[List[str]] = None,
    ) -> "WoeBinner":
        """学习各特征的最优分箱与 WoE 映射, 内联构建 feature_maps 与 report。

        Args:
            X: 训练集特征 DataFrame
            y: 训练集标签 (0/1)
            features: 指定要编码的特征列表; None 则使用 X 的全部列

        Returns:
            self (支持链式调用; 报告读 self.report)
        """
        y_arr = np.asarray(y)
        features = features or list(X.columns)

        fitted: List[str] = []
        skipped: List[str] = []
        iv_pairs: List[tuple] = []

        for feat in features:
            if feat not in X.columns:
                logger.warning(f"[WoE] 特征 '{feat}' 不在 DataFrame 中，跳过")
                skipped.append(feat)
                continue

            x_col = X[feat].values

            try:
                optb = OptimalBinning(
                    name=feat,
                    dtype="numerical",
                    max_n_bins=self.max_n_bins,
                    min_bin_size=self.min_bin_size,
                )
                optb.fit(x_col, y_arr)
                table_df = optb.binning_table.build()

                # Totals 行的 Bin 列为空字符串, 用 index 过滤;
                # Special/Missing 行的 Bin 是字符串, 用列值过滤
                real_bins = table_df.drop(index="Totals", errors="ignore")
                real_bins = real_bins[
                    ~real_bins["Bin"].astype(str).isin(["Special", "Missing"])
                ]

                iv_total = float(real_bins["IV"].sum())
                bin_edges = real_bins["Bin"].astype(str).tolist()
                woe_values = [float(v) if pd.notna(v) else 0.0 for v in real_bins["WoE"]]
                counts = [int(v) if pd.notna(v) else 0 for v in real_bins["Count"]]
                event_rates = [float(v) if pd.notna(v) else 0.0 for v in real_bins["Event rate"]]

                self._binners[feat] = optb
                self.feature_maps[feat] = WoEFeatureMap(
                    feature=feat,
                    iv=iv_total,
                    n_bins=len(real_bins),
                    bin_edges=bin_edges,
                    woe_values=woe_values,
                    counts=counts,
                    event_rates=event_rates,
                )
                fitted.append(feat)
                iv_pairs.append((feat, round(iv_total, 6)))

            except Exception as e:
                logger.warning(f"[WoE] 特征 '{feat}' 分箱失败: {e}，跳过")
                skipped.append(feat)

        self.fitted_features = fitted
        iv_pairs.sort(key=lambda x: x[1], reverse=True)
        self.report = WoEReport(
            fitted_features=fitted,
            skipped_features=skipped,
            iv_ranking=iv_pairs,
        )
        self._is_fitted = True

        logger.info(
            f"[WoE] fit 完成: {len(fitted)} 个特征成功, {len(skipped)} 个跳过"
        )
        if iv_pairs:
            logger.info(f"[WoE] Top5 IV: {iv_pairs[:5]}")

        return self

    def transform(
        self,
        X: pd.DataFrame,
        metric: str = "woe",
    ) -> pd.DataFrame:
        """对数据进行 WoE 编码。

        Args:
            X: 待编码的 DataFrame
            metric: 编码指标, 仅 "woe" 或 "event_rate"

        Returns:
            WoE 编码后的 DataFrame (列名不变)
        """
        if not self._is_fitted:
            raise RuntimeError("WoeBinner 未 fit，请先调用 fit()")
        if metric not in ("woe", "event_rate"):
            raise ValueError(f"metric 仅支持 'woe' / 'event_rate', 收到 '{metric}'")

        result = pd.DataFrame(index=X.index)

        for feat in self.fitted_features:
            if feat not in X.columns:
                logger.warning(f"[WoE] transform 时特征 '{feat}' 不存在，填充 0")
                result[feat] = 0.0
                continue
            optb = self._binners[feat]
            result[feat] = optb.transform(X[feat].values, metric=metric)

        return result


__all__ = ["WoeBinner", "WoEFeatureMap", "WoEReport"]
