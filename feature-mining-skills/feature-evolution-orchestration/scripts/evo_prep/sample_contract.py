# -*- coding: utf-8 -*-
"""样本输入契约校验: 本地 parquet/csv + label/dt/id 列 + 可选特征元数据/文本列。

输入契约(session_config.yaml 的 sample 段):
  path:             本地样本路径(.parquet 或 .csv, csv 自动 utf-8-sig 去 BOM)
  label_col:        0/1 标签列(仅支持二分类)
  dt_col:           时间列(8 位 YYYYMMDD 或可解析日期), 用于时序切分
  id_cols:          主键列列表(不入模, 用于 join/落盘)
  feature_metadata: 可选, csv 路径, 前两列按 (feature, description) 解析
  text_cols:        可选, 文本列(semantic-feature 语义特征用; 不进 baseline)
  base_model_score_col: 可选, 已有训练好模型的打分列(评估模式二选一):
      - 配置后 baseline = 该打分列, 融合判定 = 固定 XGB 在 [既有分 + 候选特征] 上
        的 OOT 是否超既有分单独的 OOT(即"与训练好的模型融合后必须更高");
      - 不配置则用原始数值特征训固定超参 XGB 作 baseline;
      - 该列不作为候选原料(候选输入帧不可见), 也不计入特征画像的候选列。
  exclude_cols:     可选, 显式排除列(如已知泄漏列)

校验失败抛 ContractError(prepare_session 映射为 exit 3)。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config_io import check_sensitive


class ContractError(ValueError):
    """数据契约校验失败。"""


def load_sample(path: str) -> pd.DataFrame:
    """读本地样本: .parquet 走 read_parquet, 其余按 csv(utf-8-sig 去 BOM)。"""
    p = Path(path)
    if not p.exists():
        raise ContractError("样本文件不存在: %s" % path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, encoding="utf-8-sig")


def parse_sample_contract(cfg: dict) -> dict:
    """从 yaml 配置解析并校验 sample 段, 返回规整后的 contract dict。

    Raises:
        ContractError: 缺必填字段 / 命中安全红线
    """
    sample = cfg.get("sample") or {}
    for key in ("path", "label_col", "dt_col"):
        if not sample.get(key):
            raise ContractError("配置 sample.%s 缺失或为空" % key)
    id_cols = sample.get("id_cols") or []
    if not isinstance(id_cols, list):
        raise ContractError("sample.id_cols 须为列表")

    contract = {
        "path": str(sample["path"]),
        "label_col": str(sample["label_col"]),
        "dt_col": str(sample["dt_col"]),
        "id_cols": [str(c) for c in id_cols],
        "feature_metadata": sample.get("feature_metadata"),
        "text_cols": [str(c) for c in (sample.get("text_cols") or [])],
        "base_model_score_col": str(sample.get("base_model_score_col") or ""),
        "exclude_cols": [str(c) for c in (sample.get("exclude_cols") or [])],
    }
    # 安全红线: 配置文本字段不得硬编码身份证/手机号
    for key in ("path", "label_col", "dt_col"):
        check_sensitive(contract[key])
    return contract


def validate_sample_df(df: pd.DataFrame, contract: dict) -> list:
    """校验样本 DataFrame 是否满足契约, 返回候选特征列(原始顺序)。

    候选特征列 = 全部列 - id_cols - label_col - dt_col - text_cols - exclude_cols

    Raises:
        ContractError: 缺列 / label 非 0/1 / dt 不可解析 / 样本为空
    """
    label_col = contract["label_col"]
    dt_col = contract["dt_col"]
    keep = set(contract["id_cols"]) | {label_col, dt_col}

    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ContractError(
            "样本缺少契约列: %s; 实际列: %s" % (missing, list(df.columns))
        )
    if len(df) == 0:
        raise ContractError("样本为空: %s" % contract["path"])

    labels = df[label_col].dropna().unique().tolist()
    if not set(labels).issubset({0, 1}):
        raise ContractError(
            "label_col=%s 仅支持二分类 0/1, 实际取值含: %s" % (label_col, labels[:10])
        )

    dt = normalize_dt(df[dt_col])
    if dt.isna().all():
        raise ContractError("dt_col=%s 无法解析为日期或可排序值" % dt_col)

    for c in contract["text_cols"] + contract["exclude_cols"]:
        if c not in df.columns:
            raise ContractError("配置列 %s 不在样本中" % c)

    score_col = contract["base_model_score_col"]
    if score_col:
        if score_col not in df.columns:
            raise ContractError("配置列 %s 不在样本中" % score_col)
        cover = pd.to_numeric(df[score_col], errors="coerce").notna().mean()
        if cover < 0.95:
            raise ContractError(
                "base_model_score_col=%s 可数值化覆盖率 %.2f%% 低于 95%%" % (score_col, cover * 100)
            )

    feature_cols = [
        c
        for c in df.columns
        if c not in keep
        and c not in contract["text_cols"]
        and c not in contract["exclude_cols"]
        and c != score_col
    ]
    if not feature_cols:
        raise ContractError("候选特征列为空(全部被 id/label/dt/text/exclude 排除)")
    return feature_cols


def normalize_dt(s: pd.Series) -> pd.Series:
    """时间列规整: 8 位数字按 YYYYMMDD 解析, 否则按通用日期解析; 失败置 NaT。"""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    as_str = s.astype(str).str.strip()
    digit = as_str.str.fullmatch(r"\d{8}")
    out = pd.to_datetime(as_str, errors="coerce", format="%Y%m%d")
    rest = pd.to_datetime(as_str.where(~digit), errors="coerce", format="mixed")
    return out.fillna(rest)
