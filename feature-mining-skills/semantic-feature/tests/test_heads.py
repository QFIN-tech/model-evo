# -*- coding: utf-8 -*-
"""heads 单测: LR head 方向正确性 / PCA 维度 / 工件序列化。"""
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
    art = heads.fit_lr_head(X, y)
    s = heads.apply_lr_head(X, art)
    assert s.shape == (len(X),)
    assert s.min() >= 0 and s.max() <= 1
    # 强特征样本的打分应显著高于弱特征样本(方向正确)
    strong = heads.apply_lr_head(np.array([[3.0, 0, 0, 0, 0]]), art)[0]
    weak = heads.apply_lr_head(np.array([[-3.0, 0, 0, 0, 0]]), art)[0]
    assert strong > 0.9 and weak < 0.1


def test_pca_head_dim_and_reconstruction_order():
    X, _ = _separable()
    art = heads.fit_pca_head(X, 3)
    Z = heads.apply_pca_head(X, art)
    assert Z.shape == (len(X), 3)
    # 前两维方差应不小于第三维(主成分按方差降序)
    var = Z.var(axis=0)
    assert var[0] >= var[1] >= var[2]


def test_artifact_roundtrip(tmp_path):
    X, y = _separable()
    art = heads.fit_lr_head(X, y)
    heads.save_artifact(tmp_path, "item", art)
    loaded = heads.load_artifact(tmp_path, "item")
    assert np.allclose(heads.apply_lr_head(X, loaded), heads.apply_lr_head(X, art))


def test_nan_inputs_survive():
    X, y = _separable()
    art = heads.fit_lr_head(X, y)
    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    X_bad[1, 1] = np.inf
    s = heads.apply_lr_head(X_bad, art)
    assert np.isfinite(s).all()
