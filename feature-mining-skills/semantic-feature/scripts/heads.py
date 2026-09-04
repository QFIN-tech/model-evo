# -*- coding: utf-8 -*-
"""Supervised Head(semantic-feature 第二段: Embedding -> 监督 Head -> Semantic Feature)。

Head 变体(变体 = 文本源 × head):
  lr_score          Embedding -> LogisticRegression -> 1 列语义打分 sem_{src}_score
  pca_lowdim        Embedding -> PCA(k) -> k 列低维表征 sem_{src}_p1..pk(无监督)
  gbdt_head         Embedding -> 浅层 XGB -> 1 列非线性打分 sem_{src}_gscore
  quantile_bins     监督分的分箱重表达 sem_{src}_q1..qk(序数, 树模型友好)
  cluster_centroid  KMeans 簇 ID sem_{src}_cluster(人群画像式)

防穿越与防 in-sample 虚高:
  监督 head(lr_score/gbdt_head)在 train 段产出 **out-of-fold(OOF) 分数**——K 折交叉拟合,
  每折样本的分数来自未见过它的模型; test/oot 用全量拟合的 head 推理。
  直接把 train 上的 in-sample 分数当特征会让 head 过拟合传导进特征、
  并让下游单变量门(G2)在 train 上虚高。

不平衡: lr_params 支持 class_weight(极低正样本率建议 balanced);
PCA/KMeans 无监督, 不受不平衡影响(这是它们常比监督 head 稳的原因)。

工件落盘用 json(系数/投影/簇心), 不用 pickle(避免反序列化安全面)。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

DEFAULT_PCA_DIM = 8
DEFAULT_LR_PARAMS = {"C": 1.0, "max_iter": 1000}
DEFAULT_OOF_FOLDS = 5
DEFAULT_GBDT_PARAMS = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.1, "random_state": 42}
DEFAULT_QUANTILE_BINS = 8
DEFAULT_N_CLUSTERS = 12

HEAD_TYPES = ("lr_score", "pca_lowdim", "gbdt_head", "quantile_bins", "cluster_centroid")


def _norm(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _folds(n: int, k: int, seed: int = 42) -> list:
    """确定性 K 折索引(乱序用固定种子)。返回 [(valid_idx), ...]。"""
    k = max(2, min(int(k), n))
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    return [order[i::k] for i in range(k)]


# ---- lr_score(OOF) ----
def fit_lr_head(X_train: np.ndarray, y_train, params: dict | None = None, oof_folds: int = DEFAULT_OOF_FOLDS) -> dict:
    """在 train 段拟合 LR。

    返回工件 {type, classes, coef, intercept, n_features, oof_folds, oof_score}。
    oof_score 是 train 段的 out-of-fold 正类概率(防 in-sample 虚高), 可直接作为 train 段特征值。
    """
    p = dict(DEFAULT_LR_PARAMS)
    if params:
        p.update(params)
    X = _norm(X_train)
    y = np.asarray(y_train)
    clf = LogisticRegression(**p)
    clf.fit(X, y)

    # OOF: 每折样本的分数来自"未见它"的模型
    k = int(oof_folds) if oof_folds else 0
    oof = np.full(len(y), np.nan)
    if k >= 2:
        for vi in _folds(len(y), k):
            fold_mask = np.zeros(len(y), dtype=bool)
            fold_mask[vi] = True
            if len(np.unique(y[~fold_mask])) < 2:
                continue
            clf_f = LogisticRegression(**p)
            clf_f.fit(X[~fold_mask], y[~fold_mask])
            oof[fold_mask] = _lr_prob(X[fold_mask], clf_f)
        # 极小折异常兜底: 仍 NaN 的行用全量 head 补
        if np.isnan(oof).any():
            oof[np.isnan(oof)] = _lr_prob(X[np.isnan(oof)], clf)

    return {
        "type": "lr_score",
        "classes": [int(c) for c in clf.classes_],
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "n_features": int(clf.coef_.shape[1]),
        "oof_folds": k,
        "oof_score": None if k < 2 else oof.tolist(),
    }


def _lr_prob(X: np.ndarray, clf) -> np.ndarray:
    """sklearn LR 正类概率(二分类约定 classes_[-1] 为正类)。"""
    return clf.predict_proba(X)[:, -1]


def apply_lr_head(X: np.ndarray, artifact: dict) -> np.ndarray:
    """应用 LR 工件 -> 正类概率(n,)。系数 json 落盘, 手算 sigmoid。"""
    coef = np.asarray(artifact["coef"], dtype=float)
    intercept = np.asarray(artifact["intercept"], dtype=float)
    Z = _norm(X) @ coef.T + intercept
    pos_idx = artifact["classes"].index(1) if 1 in artifact["classes"] else 0
    if Z.shape[1] == 1:
        prob = 1.0 / (1.0 + np.exp(-Z[:, 0]))
        return prob if artifact["classes"][-1] == 1 else 1.0 - prob
    Z = Z - Z.max(axis=1, keepdims=True)
    ez = np.exp(Z)
    return ez[:, pos_idx] / ez.sum(axis=1)


# ---- gbdt_head(OOF) ----
def fit_gbdt_head(X_train: np.ndarray, y_train, params: dict | None = None,
                  oof_folds: int = DEFAULT_OOF_FOLDS) -> dict:
    """浅层 XGB head(非线性)。工件内嵌 booster base64(json 落盘, 不用 pickle)。"""
    import base64
    import os
    import tempfile

    import xgboost as xgb

    p = dict(DEFAULT_GBDT_PARAMS)
    if params:
        p.update(params)
    X = _norm(X_train)
    y = np.asarray(y_train)
    clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", **p)
    clf.fit(X, y)
    # mkstemp 后先关句柄再交给 xgboost 的 C++ 层(Windows 下 Python 持有句柄时
    # save_model 二次打开同一路径会因文件锁 Permission denied; Linux 无此问题)
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        clf.get_booster().save_model(tmp)
        with open(tmp, "rb") as f:
            booster_b64 = base64.b64encode(f.read()).decode("ascii")
    finally:
        os.unlink(tmp)

    k = int(oof_folds) if oof_folds else 0
    oof = np.full(len(y), np.nan)
    if k >= 2:
        for vi in _folds(len(y), k):
            fold_mask = np.zeros(len(y), dtype=bool)
            fold_mask[vi] = True
            if len(np.unique(y[~fold_mask])) < 2:
                continue
            clf_f = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", **p)
            clf_f.fit(X[~fold_mask], y[~fold_mask])
            oof[fold_mask] = clf_f.predict_proba(X[fold_mask])[:, -1]
        if np.isnan(oof).any():
            oof[np.isnan(oof)] = clf.predict_proba(X[np.isnan(oof)])[:, -1]

    return {
        "type": "gbdt_head",
        "booster_b64": booster_b64,
        "n_features": int(X.shape[1]),
        "params": p,
        "oof_folds": k,
        "oof_score": None if k < 2 else oof.tolist(),
    }


def apply_gbdt_head(X: np.ndarray, artifact: dict) -> np.ndarray:
    import base64
    import tempfile

    import xgboost as xgb

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tf.write(base64.b64decode(artifact["booster_b64"]))
        tmp = tf.name
    booster = xgb.Booster(model_file=tmp)
    import os

    os.unlink(tmp)
    return booster.inplace_predict(np.asarray(_norm(X), dtype=np.float32))


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


# ---- quantile_bins(监督分的非线性重表达) ----
def fit_quantile_head(X_train: np.ndarray, y_train, base_artifact: dict | None = None,
                      n_bins: int = DEFAULT_QUANTILE_BINS) -> dict:
    """对 lr_score 的分数做分位数分箱: train 段学箱边界, 输出序数 bin 编号。

    base_artifact: 复用已拟合的 lr head 工件(缺省自己再拟一个)。
    """
    if base_artifact is None:
        base_artifact = fit_lr_head(X_train, y_train, oof_folds=0)
    s = apply_lr_head(X_train, base_artifact)
    qs = np.quantile(s, np.linspace(0, 1, int(n_bins) + 1)[1:-1])
    edges = sorted(set(qs.tolist()))
    return {"type": "quantile_bins", "edges": edges, "n_bins": len(edges) + 1, "base": base_artifact}


def apply_quantile_head(X: np.ndarray, artifact: dict) -> np.ndarray:
    """分数 -> 分箱序数(1..n_bins) 列向量 (n,1)。"""
    s = apply_lr_head(X, artifact["base"])
    edges = artifact["edges"]
    return np.asarray(
        [[float(sum(1 for e in edges if e <= v) + 1)] for v in s], dtype=float
    )


# ---- cluster_centroid ----
def fit_cluster_head(X_train: np.ndarray, n_clusters: int = DEFAULT_N_CLUSTERS, seed: int = 42) -> dict:
    """KMeans 簇 ID(人群画像式特征)。工件含簇心, json 落盘。"""
    X = _norm(X_train)
    n_clusters = max(2, min(int(n_clusters), max(2, len(X) - 1)))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(X)
    return {"type": "cluster_centroid", "n_clusters": int(n_clusters),
            "centroids": km.cluster_centers_.tolist()}


def apply_cluster_head(X: np.ndarray, artifact: dict) -> np.ndarray:
    """最近簇心 -> 簇 ID (n,)。"""
    C = np.asarray(artifact["centroids"], dtype=float)
    Xn = _norm(X)
    d = ((Xn[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(axis=1).astype(float)


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
