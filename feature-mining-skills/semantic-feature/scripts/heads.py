# -*- coding: utf-8 -*-
"""Supervised Head(semantic-feature 第二段: Embedding -> 监督 Head -> Semantic Feature)。

两种 head(变体 = 文本源 × head):
  lr_score     Embedding -> LogisticRegression -> 1 列语义打分 sem_{src}_score
               (监督信号: 业务 label; 只用 train 段拟合, test/oot 仅推理)
  pca_lowdim   Embedding -> PCA(k) -> k 列低维语义表征 sem_{src}_p1..pk
               (无监督压缩; 与 lr_score 互补, 供后续候选做组合/交互的原料)

工件落盘用 npz+json(系数/投影矩阵), 不用 pickle(避免反序列化安全面)。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

DEFAULT_PCA_DIM = 8
DEFAULT_LR_PARAMS = {"C": 1.0, "max_iter": 1000}


def _norm(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ---- lr_score ----
def fit_lr_head(X_train: np.ndarray, y_train, params: dict | None = None) -> dict:
    """在 train 段拟合 LR, 返回可落盘工件 {type, classes, coef, intercept, n_features}。"""
    p = dict(DEFAULT_LR_PARAMS)
    if params:
        p.update(params)
    clf = LogisticRegression(**p)
    clf.fit(_norm(X_train), np.asarray(y_train))
    return {
        "type": "lr_score",
        "classes": [int(c) for c in clf.classes_],
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "n_features": int(clf.coef_.shape[1]),
    }


def apply_lr_head(X: np.ndarray, artifact: dict) -> np.ndarray:
    """应用 LR 工件 -> 正类概率(n,)。正类 = classes 中的 1(二分类约定)。"""
    coef = np.asarray(artifact["coef"], dtype=float)
    intercept = np.asarray(artifact["intercept"], dtype=float)
    Z = _norm(X) @ coef.T + intercept  # (n, 1) for binary
    pos_idx = artifact["classes"].index(1) if 1 in artifact["classes"] else 0
    if Z.shape[1] == 1:
        # sklearn 二分类: 输出的是 classes_[1] 的 logit
        prob = 1.0 / (1.0 + np.exp(-Z[:, 0]))
        return prob if artifact["classes"][-1] == 1 else 1.0 - prob
    Z = Z - Z.max(axis=1, keepdims=True)
    ez = np.exp(Z)
    return ez[:, pos_idx] / ez.sum(axis=1)


# ---- pca_lowdim ----
def fit_pca_head(X_train: np.ndarray, k: int) -> dict:
    """在 train 段拟合 PCA, 返回 {type, components, mean, k}。"""
    k = max(1, min(int(k), _norm(X_train).shape[1]))
    pca = PCA(n_components=k)
    pca.fit(_norm(X_train))
    return {
        "type": "pca_lowdim",
        "k": int(k),
        "components": pca.components_.tolist(),
        "mean": pca.mean_.tolist(),
    }


def apply_pca_head(X: np.ndarray, artifact: dict) -> np.ndarray:
    """应用 PCA 工件 -> (n, k) 低维表征。"""
    comps = np.asarray(artifact["components"], dtype=float)
    mean = np.asarray(artifact["mean"], dtype=float)
    return (_norm(X) - mean) @ comps.T


# ---- 工件读写 ----
def save_artifact(dir_path, name: str, artifact: dict) -> Path:
    """工件落盘: {name}/head.json(系数在 json 内, 维度可控)。"""
    d = Path(dir_path) / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "head.json"
    p.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    return p


def load_artifact(dir_path, name: str) -> dict:
    return json.loads((Path(dir_path) / name / "head.json").read_text(encoding="utf-8"))
