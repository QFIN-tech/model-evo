# -*- coding: utf-8 -*-
"""Embedding Encoder 后端(semantic-feature 第一段: Text -> Embedding)。

三个可替换后端(semantic_config.yaml 的 encoder 段选择):
  hash                 字符 n-gram 哈希向量(sklearn HashingVectorizer)。确定性、零重依赖,
                       冒烟/单测/无 GPU 环境用; 语义能力弱于真模型。
  sentence_transformers BGE / Qwen Embedding 等本地模型(需 pip install sentence-transformers,
                       首次运行会下载模型权重)。懒加载, 缺依赖时报错并给出安装指引。
  api                  OpenAI 兼容 /embeddings HTTP 端点(仅用标准库 urllib, key 从环境变量读,
                       不落盘)。

统一接口: encoder(texts: list[str]) -> np.ndarray (n, dim)
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request

import numpy as np

DEFAULT_HASH_DIM = 256
DEFAULT_API_BATCH = 32


class EncoderError(ValueError):
    """encoder 配置/依赖/调用失败。"""


def encoder_signature(encoder_cfg: dict) -> str:
    """encoder 配置签名(嵌入缓存 key 的一部分, 换后端/模型即失效)。"""
    return json.dumps(encoder_cfg, sort_keys=True, ensure_ascii=False)


def build_encoder(encoder_cfg: dict):
    """按配置构造 encoder 闭包。"""
    backend = str(encoder_cfg.get("backend") or "hash").lower()
    if backend == "hash":
        dim = int(encoder_cfg.get("dim") or DEFAULT_HASH_DIM)
        return _build_hash_encoder(dim)
    if backend == "sentence_transformers":
        model_name = encoder_cfg.get("model") or "BAAI/bge-small-zh-v1.5"
        return _build_st_encoder(model_name, int(encoder_cfg.get("batch_size") or 32))
    if backend == "api":
        api_cfg = encoder_cfg.get("api") or {}
        return _build_api_encoder(api_cfg)
    raise EncoderError("未知 encoder backend: %s(hash/sentence_transformers/api)" % backend)


# ---- hash 后端 ----
def _build_hash_encoder(dim: int):
    """字符 1~2-gram 哈希: 中文友好(不依赖分词), 确定性, 无重依赖。"""
    from sklearn.feature_extraction.text import HashingVectorizer

    vec = HashingVectorizer(analyzer="char", ngram_range=(1, 2), n_features=dim, norm="l2", dtype=np.float64)

    def encode(texts: list) -> np.ndarray:
        if not texts:
            return np.zeros((0, dim), dtype=float)
        return np.asarray(vec.transform(texts).todense())

    return encode


# ---- sentence_transformers 后端 ----
def _build_st_encoder(model_name: str, batch_size: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise EncoderError(
            "backend=sentence_transformers 需要 pip install sentence-transformers(当前缺失): %s" % e
        ) from e

    model = SentenceTransformer(model_name)

    def encode(texts: list) -> np.ndarray:
        if not texts:
            return np.zeros((0, model.get_sentence_embedding_dimension()), dtype=float)
        return np.asarray(model.encode(texts, batch_size=batch_size, show_progress_bar=False))

    return encode


# ---- api 后端(OpenAI 兼容 /embeddings) ----
def _build_api_encoder(api_cfg: dict):
    base_url = api_cfg.get("base_url")
    if not base_url:
        raise EncoderError("backend=api 需要配置 encoder.api.base_url")
    model_name = api_cfg.get("model")
    if not model_name:
        raise EncoderError("backend=api 需要配置 encoder.api.model(如 bge-m3 / text-embedding-*)")
    key_env = str(api_cfg.get("api_key_env") or "SEMANTIC_API_KEY")
    batch_size = int(api_cfg.get("batch_size") or DEFAULT_API_BATCH)
    timeout_s = float(api_cfg.get("timeout_s") or 60.0)
    url = base_url.rstrip("/") + "/embeddings"

    def _encode_batch(batch: list) -> np.ndarray:
        api_key = os.environ.get(key_env, "")
        req = urllib.request.Request(
            url,
            data=json.dumps({"model": model_name, "input": batch}).encode("utf-8"),
            headers={"Content-Type": "application/json", **({"Authorization": "Bearer %s" % api_key} if api_key else {})},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = sorted(payload["data"], key=lambda d: int(d.get("index", 0)))
        return np.asarray([d["embedding"] for d in data], dtype=float)

    def encode(texts: list) -> np.ndarray:
        if not texts:
            raise EncoderError("api 后端不支持空文本列表")
        rows = [_encode_batch(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        return np.vstack(rows)

    return encode


# ---- 嵌入缓存(同文本+同后端不重算; 换 encoder 自动失效) ----
def texts_hash(texts: list) -> str:
    """文本列表指纹(md5)。"""
    h = hashlib.md5()
    for t in texts:
        h.update(str(t).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def encode_cached(encode, texts: list, cache_path, signature: str) -> np.ndarray:
    """带磁盘缓存的编码: cache npz 存 {signature, texts_md5, matrix}。

    cache_path 父目录须存在; 任何缓存读写失败都回退为直接编码(缓存是纯优化)。
    """
    cache_path = _p(cache_path)
    try:
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as z:
                if str(z["signature"]) == signature and str(z["texts_md5"]) == texts_hash(texts):
                    return z["matrix"]
    except Exception:
        pass  # 缓存损坏/版本不符 -> 重算
    matrix = encode(texts)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            signature=np.array(signature),
            texts_md5=np.array(texts_hash(texts)),
            matrix=np.asarray(matrix, dtype=float),
        )
    except Exception:
        pass
    return np.asarray(matrix, dtype=float)


def _p(path):
    from pathlib import Path

    return Path(path)
