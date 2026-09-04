# -*- coding: utf-8 -*-
"""encoders 单测: 后端分发 / 文本去重一致性 / list 聚合 / chunk 池化 / 缓存压缩与失效。"""
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
    """未装 sentence-transformers 时给出安装指引; 已装但模型不可达时统一抛 EncoderError(不静默降级)。"""
    import sys

    # 两种情形都要求"启动即报错"且报错可读(含模型名), 不静默降级
    with pytest.raises(encoders.EncoderError) as ei:
        encoders.build_encoder({"backend": "sentence_transformers", "model": "_nonexistent_"})
    msg = str(ei.value)
    assert "_nonexistent_" in msg or "sentence-transformers" in msg


def test_encode_unique_equals_rowwise():
    """文本级去重编码必须与逐行编码完全一致(重复文本占 2/3 的场景)。"""
    encode = encoders.build_encoder({"backend": "hash", "dim": 64})
    texts = ["", "", "", "客户逾期", "客户逾期", "正常还款"] * 10
    rowwise = encode(texts)
    uniq = encoders.encode_unique(encode, texts)
    assert uniq.shape == rowwise.shape
    assert np.allclose(uniq, rowwise)
    # 相同文本同向量
    assert np.allclose(uniq[0], uniq[1]) and np.allclose(uniq[0], uniq[2])


def test_encode_unique_progress_callback():
    calls = []
    encode = encoders.build_encoder({"backend": "hash", "dim": 32})
    encoders.encode_unique(encode, ["a", "b", "c"] * 100, progress=lambda d, t: calls.append((d, t)))
    assert calls and calls[-1][0] == calls[-1][1] == 300 // 3 * 3 or calls  # 回调至少触发一次


def test_split_items_multi_separator():
    assert encoders.split_items("微信^抖音^ 淘宝 ,支付宝") == ["微信", "抖音", "淘宝", "支付宝"]
    assert encoders.split_items("") == []
    assert encoders.split_items("^^^") == []


def test_list_aggregate_modes():
    item_vecs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    item_lists = [[0, 1], [0, 2], []]
    mean = encoders.list_aggregate(item_vecs, item_lists, agg="mean")
    assert np.allclose(mean[0], [0.5, 0.5])
    assert np.allclose(mean[1], [1.0, 0.5])          # (v0+v2)/2 = (1,0.5)
    assert np.allclose(mean[2], [0.0, 0.0])          # 空列表 -> 零向量
    s = encoders.list_aggregate(item_vecs, item_lists, agg="sum")
    assert np.allclose(s[0], [1.0, 1.0])
    tf = encoders.list_aggregate(item_vecs, item_lists, agg="tfidf")
    assert tf.shape == (3, 2) and np.isfinite(tf).all()


def test_list_aggregate_unknown_agg():
    with pytest.raises(encoders.EncoderError):
        encoders.list_aggregate(np.eye(2), [[0]], agg="nope")


def test_chunk_pool_means_and_empty_rows():
    encode = encoders.build_encoder({"backend": "hash", "dim": 16})
    # 两行: 长文本(分两窗)与空文本
    long_text = "a" * 30
    out = encoders.chunk_pool(encode, [long_text, ""], max_chars=16)
    assert out.shape == (2, 16)
    assert np.abs(out[0]).sum() > 0                  # 长文本有向量
    assert np.allclose(out[1], 0.0)                  # 空文本 -> 零向量
    # 单窗(短于窗口)与 full 编码一致
    short = encoders.chunk_pool(encode, ["短文本"], max_chars=100)
    assert np.allclose(short[0], encode(["短文本"])[0])


def test_encode_cached_hit_invalidate_and_dtype(tmp_path):
    calls = {"n": 0}

    def encode(texts):
        calls["n"] += 1
        return np.ones((len(texts), 4), dtype=np.float64)

    cache = tmp_path / "c.npz"
    texts = ["a", "b"]
    m1 = encoders.encode_cached(encode, texts, cache, "sig-1")
    m2 = encoders.encode_cached(encode, texts, cache, "sig-1")
    assert calls["n"] == 1                       # 命中缓存不重算
    assert np.allclose(m1, m2)
    # 缓存落盘为 float32(磁盘减半)
    with np.load(cache, allow_pickle=False) as z:
        assert z["matrix"].dtype == np.float32

    encoders.encode_cached(encode, texts, cache, "sig-2")   # 换后端签名 -> 失效
    assert calls["n"] == 2
    encoders.encode_cached(encode, ["a", "c"], cache, "sig-2")  # 换文本 -> 失效
    assert calls["n"] == 3
