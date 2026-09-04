# -*- coding: utf-8 -*-
"""Embedding Encoder 后端(semantic-feature 第一段: Text -> Embedding)。

可替换后端(semantic_config.yaml 的 encoder 段选择):
  hash                 字符 n-gram 哈希向量(sklearn HashingVectorizer)。确定性、零重依赖,
                       冒烟/单测/无 GPU 环境用。
  sentence_transformers BGE / Qwen Embedding 等本地模型(需 pip install sentence-transformers)。
                       支持 fp16 推理开关与 query_prefix。
  api                  OpenAI 兼容 /embeddings HTTP 端点(标准库 urllib, key 从环境变量读)。

文本模式(source.mode, 由 run_semantic 按源选择, 编码器只收"文档列表"):
  full   整行文本
  list   条目级编码 + 聚合(list_aggregate)
  chunk  分窗编码 + mean 池化(chunk_pool)

统一接口: encoder(texts: list[str]) -> np.ndarray (n, dim)

成本控制:
  encode_unique   文本级去重编码(唯一文本集合一次, 按行映射回), 结果与逐行编码一致
  进度回调        progress(done, total) 每 N 批回报, 编码层不直接 print
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request

import numpy as np

DEFAULT_HASH_DIM = 256
DEFAULT_API_BATCH = 32
PROGRESS_EVERY = 50  # 每 50 批回报一次进度(批大小由后端决定)


class EncoderError(ValueError):
    """encoder 配置/依赖/调用失败。"""


def encoder_signature(encoder_cfg: dict) -> str:
    """encoder 配置签名(嵌入缓存 key 的一部分, 换后端/模型/fp16/prefix 即失效)。"""
    return json.dumps(encoder_cfg, sort_keys=True, ensure_ascii=False)


def build_encoder(encoder_cfg: dict):
    """按配置构造 encoder 闭包。"""
    backend = str(encoder_cfg.get("backend") or "hash").lower()
    if backend == "hash":
        dim = int(encoder_cfg.get("dim") or DEFAULT_HASH_DIM)
        return _build_hash_encoder(dim)
    if backend == "sentence_transformers":
        model_name = encoder_cfg.get("model") or "BAAI/bge-small-zh-v1.5"
        return _build_st_encoder(
            model_name,
            int(encoder_cfg.get("batch_size") or 32),
            fp16=bool(encoder_cfg.get("fp16")),
            query_prefix=str(encoder_cfg.get("query_prefix") or ""),
        )
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
def _build_st_encoder(model_name: str, batch_size: int, fp16: bool = False, query_prefix: str = ""):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise EncoderError(
            "backend=sentence_transformers 需要 pip install sentence-transformers(当前缺失): %s" % e
        ) from e

    try:
        model = SentenceTransformer(model_name)
    except EncoderError:
        raise
    except Exception as e:  # 模型不存在/无网络/权重损坏等 -> 统一可读报错(不静默降级)
        raise EncoderError(
            "sentence_transformers 模型加载失败(%s): %s。"
            "请检查 model 路径是否可达(本地路径或已缓存的模型名)" % (model_name, e)
        ) from e
    if fp16:
        model = model.half()

    def encode(texts: list) -> np.ndarray:
        if not texts:
            dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") \
                else model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=float)
        if query_prefix:
            texts = [query_prefix + t for t in texts]
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
    retries = int(api_cfg.get("retries") or 2)
    url = base_url.rstrip("/") + "/embeddings"

    def _encode_batch(batch: list) -> np.ndarray:
        api_key = os.environ.get(key_env, "")
        payload = json.dumps({"model": model_name, "input": batch}).encode("utf-8")
        last_err = None
        for _attempt in range(retries + 1):
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    **({"Authorization": "Bearer %s" % api_key} if api_key else {}),
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    out = json.loads(resp.read().decode("utf-8"))
                data = sorted(out["data"], key=lambda d: int(d.get("index", 0)))
                return np.asarray([d["embedding"] for d in data], dtype=float)
            except Exception as e:  # 网络抖动/限速 -> 重试
                last_err = e
        raise EncoderError("api 后端调用失败(重试 %d 次): %s" % (retries, last_err))

    def encode(texts: list) -> np.ndarray:
        if not texts:
            raise EncoderError("api 后端不支持空文本列表")
        rows = [_encode_batch(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        return np.vstack(rows)

    return encode


# ---- 文本级去重编码(成本控制核心) ----
def encode_unique(encode, texts: list, progress=None) -> np.ndarray:
    """对唯一文本集合编码, 按行映射回向量。

    业务文本大量重复(空串/模板串)时编码调用量显著下降; 相同文本必然得到相同向量,
    结果与逐行编码完全一致。progress(done, total) 唯一文本编码进度回调。
    """
    texts = [str(t) for t in texts]
    uniq = list(dict.fromkeys(texts))  # 保序去重
    idx = {t: i for i, t in enumerate(uniq)}
    if not uniq:
        return encode([])
    done = 0
    out = []
    step = max(1, (len(uniq) + PROGRESS_EVERY - 1) // PROGRESS_EVERY * 8)  # 粗分块回调
    for i in range(0, len(uniq), step):
        out.append(encode(uniq[i:i + step]))
        done = min(i + step, len(uniq))
        if progress:
            progress(done, len(uniq))
    U = np.vstack(out)
    return U[[idx[t] for t in texts]]


# ---- list 模式: 条目级嵌入 + 聚合 ----
def split_items(text: str, seps: str = "^,;|、 ") -> list:
    """把分隔符拼接的条目列表拆为条目(去空白)。"""
    for s in seps:
        text = text.replace(s, "\x00")
    return [it.strip() for it in text.split("\x00") if it.strip()]


def list_aggregate(item_vecs: np.ndarray, item_lists: list, agg: str = "tfidf") -> np.ndarray:
    """条目向量聚合成行向量。

    item_vecs: (n_unique_items, dim) 唯一条目向量矩阵(行序 = 条目索引)
    item_lists: 每行的条目索引列表(指向 item_vecs 行)
    agg: mean | sum | tfidf(条目在语料中的 IDF 权重加权平均)
    """
    n_items, dim = item_vecs.shape
    if agg == "tfidf":
        df = np.zeros(n_items)
        for lst in item_lists:
            for i in set(lst):
                df[i] += 1
        n_rows = max(1, len(item_lists))
        idf = np.log((1.0 + n_rows) / (1.0 + df)) + 1.0

    out = np.zeros((len(item_lists), dim), dtype=float)
    for r, lst in enumerate(item_lists):
        if not lst:
            continue
        V = item_vecs[lst]
        if agg == "sum":
            out[r] = V.sum(axis=0)
        elif agg == "mean":
            out[r] = V.mean(axis=0)
        elif agg == "tfidf":
            out[r] = (V * idf[lst][:, None]).sum(axis=0) / idf[lst].sum()
        else:
            raise EncoderError("未知 list 聚合方式: %s(mean/sum/tfidf)" % agg)
    return out


# ---- chunk 模式: 分窗编码 + 池化 ----
def chunk_pool(encode, texts: list, max_chars: int, progress=None) -> np.ndarray:
    """按 max_chars 分窗编码, 行向量 = 各窗 mean 池化。空文本行为零向量。"""
    windows = []  # (row, text)
    row_spans = []  # 每行的窗区间 [start, end)
    for r, t in enumerate(texts):
        t = str(t)
        chunks = [t[i:i + max_chars] for i in range(0, len(t), max_chars)] if t else []
        start = len(windows)
        windows.extend((r, c) for c in chunks)
        row_spans.append((start, len(windows)))
    if not windows:
        dim = encode(["_probe_"]).shape[1]
        return np.zeros((len(texts), dim), dtype=float)
    uniq_texts = list(dict.fromkeys(c for _, c in windows))
    idx = {c: i for i, c in enumerate(uniq_texts)}
    U = encode(uniq_texts) if len(uniq_texts) <= len(windows) else None
    if U is None:  # 窗几乎无重复, 直接逐窗
        U = encode([c for _, c in windows])
        win_vecs = U
        win_rows = [r for r, _ in windows]
    else:
        win_vecs = U[[idx[c] for _, c in windows]]
        win_rows = [r for r, _ in windows]
    dim = win_vecs.shape[1]
    out = np.zeros((len(texts), dim), dtype=float)
    counts = np.zeros(len(texts))
    for v, r in zip(win_vecs, win_rows):
        out[r] += v
        counts[r] += 1
    nonzero = counts > 0
    out[nonzero] /= counts[nonzero, None]
    return out


# ---- 嵌入缓存(同文本+同后端不重算; float32+压缩控制磁盘) ----
def texts_hash(texts: list) -> str:
    """文本列表指纹(md5)。"""
    h = hashlib.md5()
    for t in texts:
        h.update(str(t).encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def encode_cached(encode, texts: list, cache_path, signature: str) -> np.ndarray:
    """带磁盘缓存的编码: cache npz(compressed) 存 {signature, texts_md5, matrix(float32)}。

    cache_path 父目录须存在; 任何缓存读写失败都回退为直接编码(缓存是纯优化)。
    """
    from pathlib import Path

    cache_path = Path(cache_path)
    try:
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as z:
                if str(z["signature"]) == signature and str(z["texts_md5"]) == texts_hash(texts):
                    return np.asarray(z["matrix"], dtype=float)
    except Exception:
        pass  # 缓存损坏/版本不符 -> 重算
    matrix = encode(texts)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            signature=np.array(signature),
            texts_md5=np.array(texts_hash(texts)),
            matrix=np.asarray(matrix, dtype=np.float32),
        )
    except Exception:
        pass
    return np.asarray(matrix, dtype=float)


def _p(path):
    from pathlib import Path

    return Path(path)
