# -*- coding: utf-8 -*-
"""语义特征注册表读写(semantic-feature 第三段: 注册进 feature-evolution)。

registry.json 结构:
  {
    "schema_version": 1,
    "produced_by": "feature-mining-skills/semantic-feature",
    "id_cols": [...],                       # join 键(与样本契约一致, 已去重唯一)
    "encoder": {...},                       # encoder 配置快照(复现用)
    "items": [
      {"name": "sem_txt_lr_score", "source_text_cols": ["txt"], "head": "lr_score",
       "mode": "full", "output": {"type": "score", "columns": ["sem_txt_score", "sem_txt_present"]},
       "metrics": {"sem_txt_score": {"train_auc": ..., "train_pr_auc": ..., "test_auc": ..., ...}},
       "diagnostics": {"conditional_auc_by_baseline_decile": [...], "spearman_vs_base_score": ...},
       "cost": {"encode_seconds": ..., "coverage": ...},
       "artifact": "sem_txt_lr_score/head.json"}
    ]
  }
feature_runtime.evaluation 侧按 registry + features.parquet 把语义列 join 进评估帧。
"""
from __future__ import annotations

import re

from evo_core import paths, state_io

REGISTRY_SCHEMA_VERSION = 1
PRODUCED_BY = "feature-mining-skills/semantic-feature"


def sanitize_name(s: str) -> str:
    """来源名规整为合法列名片段(小写+下划线)。"""
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(s).strip().lower()).strip("_") or "src"


def write_registry(session_dir, registry: dict) -> None:
    registry = dict(registry)
    registry.setdefault("schema_version", REGISTRY_SCHEMA_VERSION)
    registry.setdefault("produced_by", PRODUCED_BY)
    p = paths.semantic_registry_json(session_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    state_io.write_json(p, registry)


def read_registry(session_dir) -> dict | None:
    p = paths.semantic_registry_json(session_dir)
    if not p.exists():
        return None
    return state_io.read_json(p)


def flatten_columns(registry: dict) -> list:
    """全部语义特征输出列(有序去重)。"""
    cols, seen = [], set()
    for item in registry.get("items", []):
        for c in item.get("output", {}).get("columns", []):
            if c not in seen:
                seen.add(c)
                cols.append(c)
    return cols


def update_state_columns(session_dir, columns: list) -> None:
    """回写 state.semantic_columns(供编排层与反馈引用)。"""
    state_io.update_state(session_dir, semantic_columns=list(columns))
