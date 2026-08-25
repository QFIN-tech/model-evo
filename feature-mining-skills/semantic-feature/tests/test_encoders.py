# -*- coding: utf-8 -*-
"""encoders 单测: hash 后端确定性 / 后端分发 / 缓存命中与失效。"""
import numpy as np
import pytest

import encoders


def test_hash_encoder_deterministic_and_shape():
    encode = encoders.build_encoder({"backend": "hash", "dim": 64})
    texts = ["客户逾期三天", "正常还款客户", "客户逾期三天"]
    m1 = encode(texts)
    m2 = encode(texts)
    assert m1.shape == (3, 64)
    assert np.allclose(m1, m2)          # 确定性
    assert np.allclose(m1[0], m1[2])    # 相同文本相同向量
    assert not np.allclose(m1[0], m1[1])


def test_hash_encoder_empty():
    encode = encoders.build_encoder({"backend": "hash", "dim": 32})
    assert encode([]).shape == (0, 32)


def test_unknown_backend_rejected():
    with pytest.raises(encoders.EncoderError):
        encoders.build_encoder({"backend": "not_exist"})


def test_api_backend_requires_config():
    with pytest.raises(encoders.EncoderError):
        encoders.build_encoder({"backend": "api", "api": {}})


def test_st_backend_missing_dependency_message():
    # 本机未装 sentence-transformers 时应给出安装指引; 装了则跳过该断言
    try:
        enc = encoders.build_encoder({"backend": "sentence_transformers", "model": "x"})
    except encoders.EncoderError as e:
        assert "sentence-transformers" in str(e)
    else:
        assert callable(enc)


def test_encode_cached_hit_and_invalidate(tmp_path):
    calls = {"n": 0}

    def encode(texts):
        calls["n"] += 1
        return np.ones((len(texts), 4))

    cache = tmp_path / "c.npz"
    texts = ["a", "b"]
    m1 = encoders.encode_cached(encode, texts, cache, "sig-1")
    m2 = encoders.encode_cached(encode, texts, cache, "sig-1")
    assert calls["n"] == 1                       # 命中缓存不重算
    assert np.allclose(m1, m2)

    encoders.encode_cached(encode, texts, cache, "sig-2")   # 换后端签名 -> 失效
    assert calls["n"] == 2
    encoders.encode_cached(encode, ["a", "c"], cache, "sig-2")  # 换文本 -> 失效
    assert calls["n"] == 3
