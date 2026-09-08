# -*- coding: utf-8 -*-
"""heads 单测: OOF 方向正确性 / PCA / gbdt / quantile / cluster / 工件序列化。"""
import numpy as np
import pytest

import heads


def _separable(n=200, seed=7):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = (X[:, 0] * 2 + X[:, 1] > 0).astype(int)
    return X, y


def test_lr_head_orders_positive_class():
    X, y = _separable()
    art = heads.fit_lr_head(X, y, oof_folds=0)
    s = heads.apply_lr_head(X, art)
    assert s.shape == (len(X),)
    assert s.min() >= 0 and s.max() <= 1
    # 强特征样本的打分应显著高于弱特征样本(方向正确)
    strong = heads.apply_lr_head(np.array([[3.0, 0, 0, 0, 0]]), art)[0]
    weak = heads.apply_lr_head(np.array([[-3.0, 0, 0, 0, 0]]), art)[0]
    assert strong > 0.9 and weak < 0.1


def test_lr_head_oof_not_insample():
    """OOF 分数不应系统性优于全量 head 对 train 的 in-sample 打分(信息泄漏检测)。"""
    rng = np.random.RandomState(3)
    n = 400
    X = rng.randn(n, 20)
    y = (X[:, 0] + 0.5 * rng.randn(n) > 0).astype(int)   # 弱信号(易过拟合口径)
    art = heads.fit_lr_head(X, y, oof_folds=5)
    assert art["oof_score"] is not None
    oof = np.asarray(art["oof_score"])
    assert oof.shape == (n,) and np.isfinite(oof).all()
    insample = heads.apply_lr_head(X, art)
    # OOF 分数与 in-sample 分数应同分布量级但不相同
    assert not np.allclose(oof, insample)


def test_lr_head_oof_disabled():
    X, y = _separable()
    art = heads.fit_lr_head(X, y, oof_folds=0)
    assert art["oof_score"] is None


def test_lr_head_class_weight_balanced_accepted():
    X, y = _separable()
    # 类不平衡 1:19
    y_imb = np.zeros(len(y), dtype=int)
    y_imb[: max(1, len(y) // 20)] = 1
    art = heads.fit_lr_head(X, y_imb, params={"class_weight": "balanced"}, oof_folds=0)
    s = heads.apply_lr_head(X, art)
    assert np.isfinite(s).all() and s.min() >= 0 and s.max() <= 1


def test_pca_head_dim_and_variance_order():
    X, _ = _separable()
    art = heads.fit_pca_head(X, 3)
    Z = heads.apply_pca_head(X, art)
    assert Z.shape == (len(X), 3)
    var = Z.var(axis=0)
    assert var[0] >= var[1] >= var[2]        # 主成分按方差降序


def test_gbdt_head_roundtrip_and_oof():
    X, y = _separable(n=300)
    art = heads.fit_gbdt_head(X, y, oof_folds=5)
    assert art["oof_score"] is not None
    s = heads.apply_gbdt_head(X, art)
    assert s.shape == (len(X),)
    assert np.isfinite(s).all()
    # 方向正确: 工件序列化(b64 json)后可复原
    heads.save_artifact("/tmp/_sem_test_heads", "gbdt", art)
    loaded = heads.load_artifact("/tmp/_sem_test_heads", "gbdt")
    assert np.allclose(heads.apply_gbdt_head(X, loaded), s)


def test_quantile_head_bins():
    X, y = _separable(n=300)
    art = heads.fit_quantile_head(X, y, n_bins=5)
    assert art["n_bins"] >= 2
    b = heads.apply_quantile_head(X, art)
    assert b.shape == (len(X), 1)
    assert b.min() >= 1 and b.max() <= art["n_bins"]


def test_cluster_head_ids():
    # 三个明显分离的团
    rng = np.random.RandomState(0)
    centers = np.array([[0, 0], [10, 10], [-10, 10]], dtype=float)
    X = np.vstack([c + rng.randn(50, 2) * 0.3 for c in centers])
    art = heads.fit_cluster_head(X, n_clusters=3)
    ids = heads.apply_cluster_head(X, art)
    assert ids.shape == (len(X),)
    assert set(np.unique(ids)) <= set(range(3))
    # 同团样本应同簇
    assert len(set(ids[:50].astype(int))) == 1
    assert len(set(ids[50:100].astype(int))) == 1


def test_artifact_roundtrip(tmp_path):
    X, y = _separable()
    art = heads.fit_lr_head(X, y, oof_folds=0)
    heads.save_artifact(tmp_path, "item", art)
    loaded = heads.load_artifact(tmp_path, "item")
    assert np.allclose(heads.apply_lr_head(X, loaded), heads.apply_lr_head(X, art))


def test_nan_inputs_survive():
    X, y = _separable()
    art = heads.fit_lr_head(X, y, oof_folds=0)
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    X_bad[1, 1] = np.inf
    s = heads.apply_lr_head(X_bad, art)
    assert np.isfinite(s).all()
