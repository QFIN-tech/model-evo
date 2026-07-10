# -*- coding: utf-8 -*-
"""feature-analysis 配置校验: 通用校验(_modelevo-shared) + 分析专有校验。

对外暴露 load_config 与 validate_config,供 run_analysis / fetch 入口 import。
"""
from __future__ import annotations

import csv
import os
from typing import List, Optional, Tuple

import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts 到 sys.path

from config_io import load_config, validate_common, validate_split_ranges  # noqa: F401  re-export


def validate_config(cfg: dict) -> None:
    """完整校验: 通用必填 + split 必填 + 分析参数合理性。

    Args:
        cfg: load_config 返回的字典

    Raises:
        ValueError: 任何校验未通过
    """
    validate_common(cfg)
    model = cfg.get("model") or {}
    if not model.get("split"):
        raise ValueError(
            "配置 model.split 缺失: feature-analysis 现在内部切分, "
            "必须配 train_range / test_range / oot_range 三档 pday 区间"
        )
    validate_split_ranges(model)
    analysis = cfg.get("analysis") or {}

    if analysis:
        psi_cfg = analysis.get("psi") or {}
        warn = psi_cfg.get("warn_threshold", 0.10)
        if warn < 0 or warn > 1:
            raise ValueError(f"analysis.psi.warn_threshold 应在 [0,1] 内, 实际={warn}")


def cross_validate_features(
    user_features: List[str],
    data_feature_list_path: str,
) -> Tuple[List[str], List[str]]:
    """交叉校验: 用户指定的特征是否在数据特征清单中存在。

    Args:
        user_features: 用户指定的特征名列表
        data_feature_list_path: feature-matching 产出的 feature-list.csv 路径

    Returns:
        (valid_features, missing_features):
          - valid_features: 在数据中存在的特征(保序)
          - missing_features: 不在数据中的特征名列表
    """
    if not os.path.exists(data_feature_list_path):
        raise FileNotFoundError(f"数据特征清单不存在: {data_feature_list_path}")

    data_features: set = set()
    with open(data_feature_list_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "feature_name" not in reader.fieldnames:
            raise ValueError(
                f"feature-list.csv 必须包含 feature_name 列, 实际列: {reader.fieldnames}"
            )
        for row in reader:
            name = (row.get("feature_name") or "").strip()
            if name:
                data_features.add(name)

    valid: List[str] = []
    missing: List[str] = []
    for f in user_features:
        if f in data_features:
            valid.append(f)
        else:
            missing.append(f)
    return valid, missing
