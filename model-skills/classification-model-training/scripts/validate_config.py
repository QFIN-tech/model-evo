# -*- coding: utf-8 -*-
"""classification-model-training 配置校验: 通用校验(来自 _modelevo-shared) + 训练专有校验。

对外暴露 load_config 与 validate_config,供 run_build / _real_run / tune_train 等
脚本统一 import,接口与拆分前一致。
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts 到 sys.path

from config_io import load_config, validate_common  # noqa: F401  re-export


def validate_config(cfg: dict) -> None:
    """完整校验: 通用必填 + 训练专有(必填 version + 非空 features + boundary_filter 阈值)。

    本 skill 不校验 model.split: 切分由 feature-analysis 完成, 本 skill
    直接读其产出的 splits/{train,test,oot}.parquet。yaml 中 model.split
    字段对 feature-analysis 仍必填, 但本 skill 不消费, 故不校验。

    Args:
        cfg: load_config 返回的字典

    Raises:
        ValueError: 任何校验未通过
    """
    validate_common(cfg)
    model = cfg.get("model") or {}

    if not model.get("version"):
        raise ValueError("配置 model.version 缺失或为空(模型命名需要 {name}_{version}_{ts})")

    _validate_run_label(model)
    _validate_features_non_empty(model)
    _validate_boundary_filter(model.get("boundary_filter") or {})


def _validate_run_label(model: dict) -> None:
    """校验 yaml.model.run_label 不含 algo/suffix 保留字前缀。

    背景: run_label 会被 run_build.py:_resolve_version 直接用作目录后缀 version,
    与 RunLayout.create 的命名规则 `{algo}{suffix}-{version}` 叠加。若用户把
    "模型简称"塞给 run_label(如 `lgb-v1` / `tuned-v1` / `feat`), 会产出
    `lgb-lgb-v1` / `xgb-tuned-tuned-v1` / `xgb-feat` 等重复前缀或缺版本号目录。
    本函数在 config 加载期就拦截, 避免直到 run_dir 创建才发现命名错乱。

    复用 stages.layout.validate_version_label 的判定, 保证 yaml 侧与 CLI 侧
    (run_build/run_tuning/select_features 的 --version/--label) 规则一致。
    """
    run_label = model.get("run_label")
    if not run_label:
        return
    from stages.layout import validate_version_label
    try:
        validate_version_label(str(run_label))
    except ValueError as e:
        # 加一句上下文, 让用户一眼看出问题来源在 yaml 而非 CLI
        raise ValueError(f"model.run_label 非法: {e}") from e


def _validate_features_non_empty(model: dict) -> None:
    """训练专有: 拦截空 features, 避免 0-feature 一路透传到 xgboost 深处才报错。

    validate_common 在 local_file 模式下放行空 features(原意是"用本地 parquet 全列"),
    但 run_build.py 不实现该 fallback, 空列表直接喂给 trainer 会触发
    `Check failed: mparam_.num_feature != 0 (0 vs. 0) : 0 feature is supplied`。
    与 SKILL.md §7 契约一致: 特征列表为空且未配 feature_list_source 时停止执行。
    """
    if model.get("features") or model.get("feature_table"):
        return
    raise ValueError(
        "训练前必须有非空特征清单(validate_common 在 local_file 模式放行空 features, "
        "但训练实际跑不动)。请二选一:\n"
        "  1. 在 yaml model.features 显式列出特征名; 或\n"
        "  2. 在 yaml model.feature_list_source 指向 feature-matching 产出的 feature-list.csv\n"
        "     (通常位于 <session_dir>/sample-features/feature-matching/feature-list.csv)"
    )


def _validate_boundary_filter(bf: dict) -> None:
    """校验 model.boundary_filter: 阈值范围 + 开关类型。

    未配字段走默认值, 不报错; 仅校验显式配置的阈值/开关取值合法性。
    """
    if not bf:
        return
    if "iv_max" in bf and bf["iv_max"] <= 0:
        raise ValueError("model.boundary_filter.iv_max 必须 > 0")
    if "const_unique_max" in bf and bf["const_unique_max"] < 0:
        raise ValueError("model.boundary_filter.const_unique_max 必须 >= 0")
    if "id_like_ratio" in bf and not (0 < bf["id_like_ratio"] <= 1):
        raise ValueError("model.boundary_filter.id_like_ratio 须在 (0, 1]")
    if "missing_max" in bf and not (0 < bf["missing_max"] <= 1):
        raise ValueError("model.boundary_filter.missing_max 须在 (0, 1]")
    for k in ("enable_constant", "enable_leakage", "enable_id_like", "enable_all_missing"):
        if k in bf and not isinstance(bf[k], bool):
            raise ValueError(f"model.boundary_filter.{k} 必须是 bool")
