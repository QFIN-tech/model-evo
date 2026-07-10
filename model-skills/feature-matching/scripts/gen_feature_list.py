# -*- coding: utf-8 -*-
"""特征清单加载/落盘工具。

来源优先级(由 fetch_sample 调用):
  1. yaml 内 model.features 非空: 直接用
  2. yaml 内 model.feature_list_source 非空: 读该路径(.txt 按行 / .csv 取 feature_name 列)
  3. 都为空: 按 feature_table / business_domain 从 feature-knowledge.md 索引自动识别
     (feature_knowledge.resolve_feature_list_csv)

落盘: 同时把最终 features 列表写到 <out_dir>/feature-list.csv(仅 feature_name 一列),
供 feature-analysis 等下游引用。
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

import _bootstrap  # noqa: F401  注入 _modelevo-shared/scripts

# 自包含: 让 gen_feature_list 被任意路径 import 时仍能找到自己
# (config_io 已会注入本目录, 但若调用方绕过 config_io 直接 import gen_feature_list,
# 这里保证 feature_knowledge 等兄弟模块仍可解析)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# repo 根 = 本文件 ../../..  (feature-matching/scripts/gen_feature_list.py -> model-skills)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_txt(path: Path) -> List[str]:
    """按行读特征名,跳过空行与 # 注释行,顺序保留。"""
    features: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            name = raw.strip()
            if not name or name.startswith("#"):
                continue
            if name in seen:
                continue
            seen.add(name)
            features.append(name)
    return features


def _read_csv(path: Path) -> List[str]:
    """从 csv 读 feature_name 列,顺序保留,去重。"""
    features: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "feature_name" not in reader.fieldnames:
            raise ValueError(
                f"CSV 必须包含 feature_name 列,实际列: {reader.fieldnames}"
            )
        for row in reader:
            name = (row.get("feature_name") or "").strip()
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            features.append(name)
    return features


def load_feature_list(
    source: Optional[str],
    feature_table: Optional[str] = None,
    business_domain: Optional[str] = None,
) -> List[str]:
    """加载特征清单。

    Args:
        source: 文件路径; None 时按 feature_table / business_domain 从 feature-knowledge.md
                索引自动识别。相对路径按 repo 根解析。.csv 走 csv 解析(取 feature_name 列),
                其他后缀按 txt 按行解析。
        feature_table: source 为空时用于索引匹配的特征表(库.表)。
        business_domain: source 为空时用于索引匹配的业务域(具体取值由 model-knowledge 知识库登记)。

    Returns:
        去重保序的 feature 名列表。

    Raises:
        FileNotFoundError: 来源文件不存在。
        ValueError: 列表为空、csv 缺 feature_name 列, 或自动识别未命中。
    """
    if source:
        p = Path(source)
        if not p.is_absolute():
            # 相对路径解析顺序(与 config_io.validate_common 对齐):
            #   ① yaml 所在目录 / 调用方 _config_dir — 用户直觉(若调用方注入 _config_dir 到环境)
            #   ② 当前工作目录(cwd)
            #   ③ repo 根(model-skills/...) — 向后兼容 model-knowledge/... 风格
            cfg_dir = os.environ.get("_CONFIG_DIR")
            resolved = None
            if cfg_dir:
                cand = (Path(cfg_dir) / p)
                if cand.exists():
                    resolved = cand.resolve()
            if resolved is None:
                cwd_candidate = (Path.cwd() / p)
                if cwd_candidate.exists():
                    resolved = cwd_candidate.resolve()
            if resolved is None:
                resolved = (_REPO_ROOT / p).resolve()
            p = resolved
    else:
        from feature_knowledge import resolve_feature_list_csv

        p = resolve_feature_list_csv(feature_table, business_domain)
        if p is None:
            raise ValueError(
                "未指定特征清单, 且按 feature_table=%s / business_domain=%s 未在"
                " feature-knowledge.md 索引中识别到清单; 请显式传 features /"
                " feature_list_source, 或在特征知识库登记该特征表"
                % (feature_table, business_domain)
            )

    if not p.exists():
        raise FileNotFoundError(f"特征清单文件不存在: {p}")

    if p.suffix.lower() == ".csv":
        features = _read_csv(p)
    else:
        features = _read_txt(p)

    if not features:
        raise ValueError(f"特征清单为空: {p}")
    return features


def write_feature_list_csv(features: List[str], out_path: str) -> None:
    """把特征清单写成单列 csv(列名 feature_name)。"""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name"])
        for name in features:
            writer.writerow([name])
