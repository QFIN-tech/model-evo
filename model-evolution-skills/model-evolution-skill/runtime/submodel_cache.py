"""子模型级产物缓存(v2.17.0,run4 regression.2 #10)。

事故:hurdle 的 019/021/022/023 只改了回归子模型,分类子模型与父代 booster
sha256 逐位相同,却各自重训 40~77 min(约占 25% trial-hours)。本模块给出
内容寻址的"训练或加载":

    fitted, info = fit_or_load("clf", estimator, X, y, eval_set=[(Xe, ye)],
                               fit_kwargs={"verbose": False}, run_dir=RUN_DIR)

缓存键(blake2b):
  - X / y / eval_set 数组内容(列名 + dtype + 值;大数组按块流式哈希);
  - 估计器类 + `get_params(deep=False)`,排除不影响结果的键(n_jobs、
    nthread、verbosity、verbose、silent);设备只记类型(cuda/cpu,不含卡号);
  - 库版本(估计器所属库、numpy、python 主次版本)、seed、fit_kwargs
    (去掉 verbose 类)、extra_key、缓存格式版本。

存储:``RUN_DIR/.submodel-cache/<key>/``(xgboost = 原生 .ubj;其它 = joblib)
+ ``meta.json``(键组成、evals_result、模型文件 sha256、拟合耗时)。命中时
加载并校验 sha256,返回 info = {status: hit|miss|disabled|uncacheable, ...}。

**只在结果确定性可保证时命中**:随机采样(subsample/colsample/bagging…<1)
而 seed 缺失、LightGBM 未开 deterministic、未知估计器带随机性但无
random_state —— 一律只训练不缓存(status=uncacheable,reason 留痕)。
xgboost CPU hist 同输入同参数的 booster 逐字节一致(与线程数无关,单测
覆盖)。``MODELEVO_SUBMODEL_CACHE=0`` 整体关闭。
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIRNAME = ".submodel-cache"
#: 落盘布局 / 键构成变化时递增(旧缓存自然失配)
CACHE_FORMAT = 1
#: 不影响拟合结果的参数键(不进缓存键)
_IGNORED_PARAMS = frozenset({"n_jobs", "nthread", "verbosity", "verbose",
                             "silent", "num_threads", "thread_count"})
_IGNORED_FIT_KW = frozenset({"verbose", "verbose_eval"})
#: 随机采样类参数:取值 < 1(或 bagging_freq>0)即需要显式 seed
_STOCHASTIC_PARAMS = ("subsample", "colsample_bytree", "colsample_bylevel",
                      "colsample_bynode", "bagging_fraction",
                      "feature_fraction", "feature_fraction_bynode",
                      "max_features")
_SEED_PARAMS = ("random_state", "seed", "random_seed")
_HASH_CHUNK_BYTES = 64 << 20


def cache_enabled() -> bool:
    return os.environ.get("MODELEVO_SUBMODEL_CACHE", "1").strip() != "0"


# ---------------------------------------------------------------------------
# 内容哈希
# ---------------------------------------------------------------------------
def _update_array(h, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    h.update(f"nd{arr.shape}|{arr.dtype.str}|".encode())
    if arr.dtype.hasobject:
        import pandas as pd

        flat = pd.Series(arr.ravel(order="C"))
        h.update(pd.util.hash_pandas_object(flat, index=False)
                 .to_numpy().tobytes())
        return
    if arr.ndim == 0 or arr.size == 0:
        h.update(np.ascontiguousarray(arr).tobytes())
        return
    rows = max(1, _HASH_CHUNK_BYTES // max(1, arr.itemsize *
                                           int(np.prod(arr.shape[1:]) or 1)))
    for i in range(0, arr.shape[0], rows):
        h.update(np.ascontiguousarray(arr[i:i + rows]).tobytes())


def _update_frame(h, X) -> None:
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        h.update(f"df{X.shape}|".encode())
        for c in X.columns:
            col = X[c]
            h.update(f"c:{c!s}|{col.dtype!s}|".encode())
            if isinstance(col.dtype, pd.CategoricalDtype):
                h.update(repr(list(col.cat.categories)).encode())
                _update_array(h, col.cat.codes.to_numpy())
            else:
                _update_array(h, col.to_numpy())
        return
    if isinstance(X, pd.Series):
        h.update(f"sr{len(X)}|{X.dtype!s}|{X.name!s}|".encode())
        if isinstance(X.dtype, pd.CategoricalDtype):
            h.update(repr(list(X.cat.categories)).encode())
            _update_array(h, X.cat.codes.to_numpy())
        else:
            _update_array(h, X.to_numpy())
        return
    if X is None:
        h.update(b"none|")
        return
    if hasattr(X, "toarray") and hasattr(X, "nnz"):   # scipy sparse
        h.update(f"sp{X.shape}|".encode())
        Xc = X.tocsr()
        for part in (Xc.data, Xc.indices, Xc.indptr):
            _update_array(h, part)
        return
    _update_array(h, np.asarray(X))


def content_hash(*objs) -> str:
    """数组/帧内容的 blake2b(32 字节十六进制)。"""
    h = hashlib.blake2b(digest_size=32)
    for o in objs:
        h.update(b"<obj>")
        _update_frame(h, o)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 参数 / 环境
# ---------------------------------------------------------------------------
def _lib_of(est) -> str:
    return (type(est).__module__ or "").split(".")[0]


def _lib_version(lib: str) -> str:
    mod = sys.modules.get(lib)
    if mod is None:
        try:
            mod = __import__(lib)
        except Exception:  # noqa: BLE001
            return ""
    return str(getattr(mod, "__version__", ""))


def _device_type(params: dict) -> str:
    for k in ("device", "device_type", "task_type", "tree_method"):
        v = str(params.get(k) or "").lower()
        if "cuda" in v or "gpu" in v:
            return "cuda"
    return "cpu"


def _jsonable(v: Any):
    if v is None or isinstance(v, (bool, int, float, str)):
        if isinstance(v, float) and not np.isfinite(v):
            return repr(v)
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in sorted(v.items(),
                                                       key=lambda kv: str(kv[0]))}
    if hasattr(v, "get_params"):
        return {"__estimator__": f"{type(v).__module__}.{type(v).__qualname__}",
                "params": _jsonable(_effective_params(v))}
    if callable(v):
        return f"<callable:{getattr(v, '__module__', '')}."\
               f"{getattr(v, '__qualname__', type(v).__name__)}>"
    return f"<{type(v).__module__}.{type(v).__qualname__}:{v!r}>"[:400]


def _effective_params(est) -> dict:
    try:
        p = dict(est.get_params(deep=False))
    except Exception:  # noqa: BLE001
        p = {}
    return {k: v for k, v in p.items() if k not in _IGNORED_PARAMS}


def determinism_check(est) -> str | None:
    """结果确定性无法保证时返回原因;可缓存返回 None。"""
    try:
        p = dict(est.get_params(deep=False))
    except Exception:  # noqa: BLE001
        return "估计器无 get_params(无法确定参数口径)"
    lib = _lib_of(est)
    seed = next((p[k] for k in _SEED_PARAMS if k in p), "__absent__")
    stochastic = []
    for k in _STOCHASTIC_PARAMS:
        v = p.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and \
                0 < float(v) < 1:
            stochastic.append(k)
    if int(p.get("bagging_freq") or 0) > 0:
        stochastic.append("bagging_freq")
    if stochastic and (seed is None or seed == "__absent__"):
        return f"随机采样参数 {stochastic} 未给 seed(random_state)"
    if lib == "lightgbm" and not p.get("deterministic"):
        return "lightgbm 未开 deterministic=True(多线程直方图累加不保证逐位复现)"
    if lib == "lightgbm" and _device_type(p) == "cuda":
        return "lightgbm GPU 训练不保证逐位复现"
    if lib not in ("xgboost", "lightgbm") and seed is None and \
            any(k in p for k in _SEED_PARAMS):
        return "估计器 random_state=None(随机性不受控)"
    if not callable(getattr(est, "fit", None)):
        return "估计器无 fit"
    return None


def _is_arraylike(v) -> bool:
    return isinstance(v, np.ndarray) or type(v).__module__.startswith(
        "pandas") or (hasattr(v, "toarray") and hasattr(v, "nnz"))


def _fit_kwargs_key(fit_kwargs: dict | None) -> dict:
    out = {}
    for k, v in sorted((fit_kwargs or {}).items()):
        if k in _IGNORED_FIT_KW or k == "eval_set":
            continue
        if _is_arraylike(v):
            out[k] = "blake2b:" + content_hash(v)        # sample_weight 等按内容
        elif isinstance(v, (list, tuple)) and v and all(
                _is_arraylike(x) or x is None for x in v):
            out[k] = "blake2b:" + content_hash(*v)
        else:
            out[k] = _jsonable(v)
    return out


def cache_key(name: str, est, data_hash: str, *, fit_kwargs=None,
              extra_key=None) -> tuple[str, dict]:
    params = _effective_params(est)
    lib = _lib_of(est)
    seed = next((params[k] for k in _SEED_PARAMS if k in params), None)
    parts = {
        "format": CACHE_FORMAT,
        "name": str(name),
        "estimator": f"{type(est).__module__}.{type(est).__qualname__}",
        "params": _jsonable({k: v for k, v in params.items()
                             if k not in ("device",)}),
        "device": _device_type(params),
        "seed": _jsonable(seed),
        "lib": lib, "lib_version": _lib_version(lib),
        "numpy": np.__version__,
        "python": f"{sys.version_info[0]}.{sys.version_info[1]}",
        "data": data_hash,
        "fit_kwargs": _fit_kwargs_key(fit_kwargs),
        "extra": _jsonable(extra_key),
    }
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.blake2b(blob, digest_size=16).hexdigest(), parts


# ---------------------------------------------------------------------------
# 落盘 / 加载
# ---------------------------------------------------------------------------
class _Lock:
    """跨进程排他锁(flock;进程死亡即释放)。同键并发:后到者等待后命中。"""

    def __init__(self, path: Path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        finally:
            self.fh.close()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _evals_result_of(est):
    ev = getattr(est, "evals_result_", None)
    if ev is None and hasattr(est, "evals_result"):
        try:
            ev = est.evals_result()
        except Exception:  # noqa: BLE001
            ev = None
    return ev


def _save(est, d: Path) -> tuple[str, str]:
    lib = _lib_of(est)
    if lib == "xgboost" and hasattr(est, "save_model"):
        p = d / "model.ubj"
        est.save_model(str(p))
        return p.name, "xgboost_native"
    import joblib

    p = d / "model.joblib"
    joblib.dump(est, p)
    return p.name, "joblib"


def _load(proto, d: Path, meta: dict):
    from sklearn.base import clone

    p = d / meta["model_file"]
    if _sha256(p) != meta.get("model_sha256"):
        raise ValueError("模型文件 sha256 与 meta 不一致(缓存损坏)")
    if meta.get("format_kind") == "xgboost_native":
        est = clone(proto)
        est.load_model(str(p))
        ev = meta.get("evals_result")
        if ev:
            est.evals_result_ = ev
        return est
    import joblib

    return joblib.load(p)


def booster_bytes(est) -> bytes | None:
    """xgboost 估计器的 booster 原始字节(ubj;测试与审计用)。"""
    try:
        return bytes(est.get_booster().save_raw("ubj"))
    except Exception:  # noqa: BLE001
        return None


def _resolve_run_dir(run_dir) -> Path | None:
    if run_dir:
        return Path(run_dir)
    env = os.environ.get("MODELEVO_RUN_DIR")
    return Path(env) if env else None


def fit_or_load(name: str, estimator, X, y, *, eval_set=None,
                fit_kwargs: dict | None = None, run_dir=None,
                extra_key=None) -> tuple[Any, dict]:
    """训练 ``clone(estimator)`` 或从缓存加载;返回 (已拟合估计器, info)。

    estimator 本身不被修改(总是在 clone 上拟合)。eval_set 会以
    ``eval_set=`` 关键字交给 fit(也进缓存键);fit_kwargs 其余项原样转交。
    info:{name, status(hit|miss|disabled|uncacheable|error), key, hash_sec,
    fit_sec / load_sec, reason, path}。
    """
    from sklearn.base import clone

    fit_kwargs = dict(fit_kwargs or {})
    kw = dict(fit_kwargs)
    if eval_set is not None:
        kw["eval_set"] = eval_set
    info: dict = {"name": str(name), "status": None, "key": None,
                  "hash_sec": 0.0}
    est = clone(estimator)

    def _fit_plain(status: str, reason: str | None = None):
        t0 = time.time()
        est.fit(X, y, **kw)
        info.update(status=status, fit_sec=round(time.time() - t0, 3))
        if reason:
            info["reason"] = reason
        return est, info

    rd = _resolve_run_dir(run_dir)
    if not cache_enabled():
        return _fit_plain("disabled", "MODELEVO_SUBMODEL_CACHE=0")
    if rd is None:
        return _fit_plain("disabled", "未给 run_dir(无缓存落点)")
    why = determinism_check(est)
    if not why and fit_kwargs.get("callbacks"):
        why = "fit callbacks 无法可靠纳入缓存键(参数不可序列化)"
    if not why and any(callable(v) for v in _effective_params(est).values()):
        why = "估计器参数含可调用对象(自定义目标/指标,无法可靠纳入缓存键)"
    if why:
        return _fit_plain("uncacheable", why)
    t0 = time.time()
    ev_arrays = []
    for pair in (eval_set or []):
        ev_arrays.extend(list(pair))
    data_hash = content_hash(X, y, *ev_arrays) + f"|n_eval={len(eval_set or [])}"
    info["hash_sec"] = round(time.time() - t0, 3)
    key, parts = cache_key(name, est, data_hash, fit_kwargs=fit_kwargs,
                           extra_key=extra_key)
    info["key"] = key
    root = rd / CACHE_DIRNAME
    d = root / key
    info["path"] = str(d)
    with _Lock(root / f"{key}.lock"):
        meta_p = d / "meta.json"
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                t1 = time.time()
                loaded = _load(est, d, meta)
                info.update(status="hit", load_sec=round(time.time() - t1, 3),
                            saved_fit_sec=meta.get("fit_sec"),
                            source=meta.get("writer"))
                return loaded, info
            except Exception as exc:  # noqa: BLE001  损坏 → 重训覆盖
                info["reason"] = f"缓存读取失败,重训: {type(exc).__name__}: {exc}"
                shutil.rmtree(d, ignore_errors=True)
        t1 = time.time()
        est.fit(X, y, **kw)
        fit_sec = round(time.time() - t1, 3)
        info.update(status="miss", fit_sec=fit_sec)
        try:
            tmp = root / f".{key}.tmp{os.getpid()}"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            fname, kind = _save(est, tmp)
            meta = {
                "key": key, "key_parts": parts, "model_file": fname,
                "format_kind": kind, "model_sha256": _sha256(tmp / fname),
                "evals_result": _jsonable(_evals_result_of(est)),
                "best_iteration": _jsonable(getattr(est, "best_iteration",
                                                    None)
                                            if kind == "xgboost_native"
                                            else None),
                "fit_sec": fit_sec, "hash_sec": info["hash_sec"],
                "writer": os.environ.get("MODELEVO_TRIAL_ID")
                or os.environ.get("MODELEVO_TRIAL_EXEC") or "",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False,
                                                      indent=1),
                                           encoding="utf-8")
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            os.replace(tmp, d)
        except Exception as exc:  # noqa: BLE001  写盘失败不影响本轮
            info["write_error"] = f"{type(exc).__name__}: {exc}"
        return est, info


def summarize(infos: dict) -> dict:
    """{段名: info} → metrics.submodel_cache 摘要。"""
    segs = {k: {kk: v.get(kk) for kk in ("status", "key", "hash_sec",
                                         "fit_sec", "load_sec",
                                         "saved_fit_sec", "reason")
                if v.get(kk) is not None}
            for k, v in (infos or {}).items()}
    sts = [v.get("status") for v in (infos or {}).values()]
    return {"segments": segs,
            "hits": sum(s == "hit" for s in sts),
            "misses": sum(s == "miss" for s in sts),
            "hash_sec": round(sum(float(v.get("hash_sec") or 0)
                                  for v in (infos or {}).values()), 3),
            "saved_fit_sec": round(sum(float(v.get("saved_fit_sec") or 0)
                                       for v in (infos or {}).values()
                                       if v.get("status") == "hit"), 3),
            "enabled": cache_enabled()}


def cache_inventory(run_dir) -> list[dict]:
    root = Path(run_dir) / CACHE_DIRNAME
    out = []
    if not root.exists():
        return out
    for mp in sorted(root.glob("*/meta.json")):
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        f = mp.parent / str(m.get("model_file") or "")
        out.append({"key": m.get("key"), "name": (m.get("key_parts") or {})
                    .get("name"), "bytes": f.stat().st_size if f.exists() else 0,
                    "fit_sec": m.get("fit_sec"), "writer": m.get("writer"),
                    "created_at": m.get("created_at")})
    return out
