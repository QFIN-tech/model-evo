"""v2.19.0:强参照与天花板估计(``modelevo.py ceiling``)。

用途(路线图 §6.3 #2 / §6.6 #11):在 baseline 之后,用固定预算跑"这份数据
在当前特征面上大致能到哪里"的强参照,结果经 ``runtime.ceiling.write_ceiling``
落盘,供 stop report(gap_to_ref)、planner-context 与 benchmark 报告消费。

口径(与 runtime/ceiling.py 一致):

- 只用 train 训练;validation 是裁决面,只用于早停与最终打分(与搜索 trial 的
  v2.16 ``es_source=validation`` 同口径)。不读 shadow_test / final_test 标签;
- engine ``strong``:``strong_default.strong_params`` 按行列规模缩放的强参数,
  validation 早停,K seed bagging 取平均;同时记录单 seed 值与 bag 值。预算
  允许时再加 lightgbm 强参数一组,以及两族 bag 的等权混合;
- engine ``autogluon``:独立 venv 解释器子进程(best_quality,time_limit=预算),
  子进程只拿到 train(含标签)与 validation **特征**,返回 validation 预测,
  打分在本进程完成——validation 标签不离开 runtime;
- 参照允许 seed bagging / 集成:它是测量,不是可交付 trial,不受单模型拓扑门
  约束;
- ``feature_surface``:``base`` = profile 材料化的特征面;``trial:<id>`` = 该
  trial 的特征矩阵(目前支持第五族标准契约 ``trial_feature_blocks``:基础面 +
  特征块缓存复建)。stop 判停只采信 ``trial:<id>`` 面(主会话口径)。

只读模式(``out_dir``):结果、GPU 租约、特征块缓存的写入全部落到 out_dir;
对 run 目录零写入(trial 模块加载时关闭字节码缓存,特征块缓存走符号链接镜像)。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .ceiling import write_ceiling
from .logging_utils import get_logger, load_json, load_yaml, now_iso

logger = get_logger(__name__)

#: 支持的主指标 → (任务族, AutoGluon eval_metric)
CLASSIFICATION_METRICS = {"auc": "roc_auc", "logloss": "log_loss",
                          "auprc": "average_precision", "ks": "roc_auc",
                          "gini": "roc_auc"}
REGRESSION_METRICS = {"rmse": "root_mean_squared_error",
                      "mae": "mean_absolute_error", "r2": "r2",
                      "mape": "mean_absolute_percentage_error"}
#: 低基数非数值列走原生类别(xgboost/lightgbm enable_categorical);更高基数
#: 走频次编码(strong_default.frequency_encoding,只用 train 分布)
MAX_NATIVE_CATEGORIES = 255
#: AutoGluon 独立 venv 解释器(可经 --autogluon-python / 环境变量覆盖)
AUTOGLUON_PYTHON_ENV = "MODELEVO_AUTOGLUON_PYTHON"
DEFAULT_AUTOGLUON_PYTHON = ("/data/oceanus_share/j-wangyaoxuan-jk/"
                            "modelevo-202611/venvs/autogluon/bin/python")
#: GPU 训练的最小格数(行 × 列);小数据 CPU 更快
GPU_MIN_CELLS = 20_000_000


class ReferenceError_(RuntimeError):
    """参照无法执行(指标/任务类型不支持、特征面不可复建等)。"""


# ---------------------------------------------------------------------------
# 指标与任务
# ---------------------------------------------------------------------------

def resolve_task(spec) -> dict:
    """从 TaskSpec 解析 (model_type, primary_metric, higher_is_better)。

    uplift / multiclass 明确报错(v2.19.0 暂不支持)。"""
    from .metrics import metric_direction

    mtype = str(spec.model_type or "")
    metric = str(spec.primary_metric or "").lower()
    if mtype == "uplift":
        raise ReferenceError_("ceiling 暂不支持 uplift 任务(v2.19.0)")
    if mtype == "binary_classification":
        if metric not in CLASSIFICATION_METRICS:
            raise ReferenceError_(
                f"二分类主指标 {metric} 暂不支持;支持 {sorted(CLASSIFICATION_METRICS)}")
    elif mtype == "regression":
        if metric not in REGRESSION_METRICS:
            raise ReferenceError_(
                f"回归主指标 {metric} 暂不支持;支持 {sorted(REGRESSION_METRICS)}")
    else:
        raise ReferenceError_(f"ceiling 暂不支持任务类型 {mtype}")
    return {"model_type": mtype, "primary_metric": metric,
            "higher_is_better": bool(metric_direction(metric))}


def score_predictions(task: dict, y: np.ndarray, pred: np.ndarray) -> float:
    from .metrics import compute_metric

    return float(compute_metric(task["primary_metric"], np.asarray(y),
                                np.asarray(pred), score=np.asarray(pred)))


def _better(task: dict, a: float, b: float) -> bool:
    return a > b if task["higher_is_better"] else a < b


# ---------------------------------------------------------------------------
# 特征面
# ---------------------------------------------------------------------------

@dataclass
class FeatureSurface:
    X_tr: pd.DataFrame
    y_tr: np.ndarray
    X_va: pd.DataFrame
    y_va: np.ndarray
    surface: str = "base"
    info: dict = field(default_factory=dict)
    # v2.19.0:对齐测试面(缺省 shadow_test)的特征与样本键(identity 面,与
    # 特征面同一 frame 写出、行序一致)。参照在此面上只**打分落盘**,不读标签;
    # 评分与样本键核验在 benchmark aligned 中统一做
    X_ho: "pd.DataFrame | None" = None
    keys_ho: "np.ndarray | None" = None
    holdout_split: str | None = None


def govern_frames(X_tr: pd.DataFrame, X_va: pd.DataFrame,
                  X_ho: "pd.DataFrame | None" = None):
    """参照口径的 dtype 治理:数值 → float32;可数值化 object → float32;
    低基数 → 原生 category(train 类别表);高基数 → 频次编码。

    与 baseline 模板(one-hot ≤100、>100 直接剔除)相比信息只多不少——参照
    要的是"能到哪里",不复刻 trial 的治理细节。

    v2.19.0:``X_ho``(对齐测试面)给定时按与 X_va 完全相同的变换处理(类别表、
    频次表都只来自 train),返回 (X_tr, X_va, gov, X_ho);否则返回三元组。"""
    from .safe_dtype import safe_to_numeric
    from .strong_default import frequency_encoding

    X_tr = X_tr.copy()
    X_va = X_va.copy()
    others = [X_va] + ([X_ho.copy()] if X_ho is not None else [])
    n_cat = n_freq = n_num = 0
    for c in list(X_tr.columns):
        s = X_tr[c]
        if pd.api.types.is_bool_dtype(s):
            X_tr[c] = s.astype("float32")
            for o in others:
                o[c] = o[c].astype("float32")
            continue
        if pd.api.types.is_numeric_dtype(s):
            if s.dtype != np.float32:
                X_tr[c] = s.astype("float32")
                for o in others:
                    o[c] = pd.to_numeric(o[c], errors="coerce").astype(
                        "float32")
            continue
        conv = None if isinstance(s.dtype, pd.CategoricalDtype) \
            else safe_to_numeric(s)
        if conv is not None:
            X_tr[c] = conv.astype("float32")
            for o in others:
                v = safe_to_numeric(o[c])
                o[c] = v.astype("float32") if v is not None else np.nan
            n_num += 1
            continue
        nun = s.nunique(dropna=True)
        if nun <= MAX_NATIVE_CATEGORIES:
            cats = pd.Index(s.dropna().astype(str).unique())
            X_tr[c] = pd.Categorical(s.astype(str).where(s.notna()),
                                     categories=cats)
            for o in others:
                v = o[c]
                o[c] = pd.Categorical(v.astype(str).where(v.notna()),
                                      categories=cats)
            n_cat += 1
        else:
            enc_tr = None
            for o in others:
                enc_tr, o[c] = frequency_encoding(s, o[c])
            X_tr[c] = enc_tr
            n_freq += 1
    X_va = others[0]
    gov = {"native_categorical": n_cat, "frequency_encoded": n_freq,
           "parsed_numeric": n_num}
    if X_ho is not None:
        return X_tr, X_va, gov, others[1]
    return X_tr, X_va, gov


def _holdout_keys(run_dir: Path, split: str) -> np.ndarray:
    """对齐测试面的样本键(identity 面;与特征面行序一致)。"""
    from .data import SAMPLE_KEY_COLUMN, load_identities

    ids = load_identities(Path(run_dir), splits=(split,))
    if split not in ids:
        raise ReferenceError_(f"identity 面缺 {split}(先跑 profile)")
    keys = ids[split][SAMPLE_KEY_COLUMN].astype(str).to_numpy()
    if len(set(keys)) != len(keys):
        raise ReferenceError_(f"{split} 的样本键有重复,无法对齐")
    return keys


def _subsample_index(y: np.ndarray, max_rows: int, task: dict,
                     seed: int = 20261101) -> np.ndarray:
    """train 行抽样(二分类按标签分层;回归简单随机),返回有序行位置。"""
    n = len(y)
    if not max_rows or n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    if task["model_type"] == "binary_classification":
        out = []
        for v in np.unique(y):
            pos = np.flatnonzero(y == v)
            k = max(1, int(round(max_rows * len(pos) / n)))
            out.append(rng.choice(pos, size=min(k, len(pos)), replace=False))
        idx = np.concatenate(out)
    else:
        idx = rng.choice(n, size=max_rows, replace=False)
    return np.sort(idx)


def _apply_subsample(X: pd.DataFrame, y: np.ndarray, idx: np.ndarray):
    if len(idx) == len(y):
        return X, y
    return X.iloc[idx].reset_index(drop=True), y[idx]


def _splits_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "sample-profile" / "splits"


def _load_labels(run_dir: Path, spec) -> tuple[np.ndarray, np.ndarray]:
    from .data import load_labels

    labs = load_labels(run_dir, splits=("train", "validation"))
    if "train" not in labs or "validation" not in labs:
        raise ReferenceError_("标签面缺失(sample-profile/splits/labels);先跑 profile")
    tgt = spec.target_name
    return (labs["train"][tgt].to_numpy(), labs["validation"][tgt].to_numpy())


def load_base_surface(run_dir: Path, spec, task: dict, *,
                      max_rows: int | None = None,
                      holdout: str | None = None) -> FeatureSurface:
    """profile 材料化的基础特征面(splits/{train,validation}.parquet)。"""
    import pyarrow.parquet as pq

    from .data import resolve_feature_columns

    sd = _splits_dir(run_dir)
    if not (sd / "train.parquet").exists() \
            or not (sd / "validation.parquet").exists():
        raise ReferenceError_(f"特征面缺失: {sd}(先跑 profile)")
    header = pq.read_schema(sd / "train.parquet").names
    try:
        feats = resolve_feature_columns(pd.DataFrame(columns=header), spec)
    except ValueError:
        # 白名单列不在特征面(例如被 banned 物理隔离)→ 退回全列口径
        feats = [c for c in header if c not in {
            spec.target_name, spec.time_column, spec.id_column,
            "__mvo_sample_key"}]
    y_tr, y_va = _load_labels(run_dir, spec)
    X_tr = pd.read_parquet(sd / "train.parquet", columns=feats)
    X_va = pd.read_parquet(sd / "validation.parquet", columns=feats)
    n_full = len(y_tr)
    idx = _subsample_index(y_tr, max_rows or 0, task)
    X_tr, y_tr = _apply_subsample(X_tr, y_tr, idx)
    X_ho = keys_ho = None
    if holdout:
        X_ho = pd.read_parquet(sd / f"{holdout}.parquet", columns=feats)
        keys_ho = _holdout_keys(run_dir, holdout)
        if len(keys_ho) != len(X_ho):
            raise ReferenceError_(f"{holdout} 特征面 {len(X_ho)} 行 != identity "
                                  f"{len(keys_ho)} 行")
        X_tr, X_va, gov, X_ho = govern_frames(X_tr, X_va, X_ho)
    else:
        X_tr, X_va, gov = govern_frames(X_tr, X_va)
    return FeatureSurface(X_tr, y_tr, X_va, y_va, "base", {
        "n_features": int(X_tr.shape[1]), "n_train": int(len(X_tr)),
        "n_validation": int(len(X_va)), "n_train_full": int(n_full),
        "n_holdout": None if X_ho is None else int(len(X_ho)),
        "subsampled": bool(len(idx) < n_full), "governance": gov},
        X_ho=X_ho, keys_ho=keys_ho, holdout_split=holdout)


# ---------------------------------------------------------------------------
# --from-trial:trial 特征面复建(只读)
# ---------------------------------------------------------------------------

def _mirror_feature_cache(src: Path, dst: Path) -> int:
    """把 run 的特征块缓存以符号链接镜像到 dst(只读复用:命中读源文件,
    新写入 / 锁 / 内容哈希 memo 全部落在 dst)。返回链接文件数。"""
    n = 0
    if not src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        return 0
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.endswith(".lock"):
                continue
            t = dst / rel / f
            if not t.exists() and not t.is_symlink():
                os.symlink(Path(root) / f, t)
                n += 1
    return n


def _load_trial_module(code_path: Path, name: str):
    """加载 trial code/train.py(不写 __pycache__,不永久改 sys.path)。"""
    import importlib.util

    saved_path = list(sys.path)
    saved_flag = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, code_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.dont_write_bytecode = saved_flag
        sys.path[:] = saved_path


def load_trial_surface(run_dir: Path, spec, task: dict, trial_id: str, *,
                       max_rows: int | None = None,
                       cache_dir: Path | None = None,
                       holdout: str | None = None) -> FeatureSurface:
    """trial 特征面 = trial 配置的基础列(exclude_columns)+ trial_feature_blocks。

    只支持第五族标准契约(``trial_feature_blocks(split)``,无训练期拟合状态,
    内容寻址缓存命中即读)。第一~四族加宽(build_wide / _build_widen_columns /
    _widen / widen_face)的复建逻辑内联在 finalize-retrain 中且带训练期状态,
    未抽成可复用函数,这里明确报错而不是静默退化成基础面。

    ``cache_dir``:特征块缓存根(只读模式下为 out_dir 内的镜像)。
    """
    import pyarrow.parquet as pq

    tdir = Path(run_dir) / "trials" / trial_id
    code = tdir / "code" / "train.py"
    if not code.exists():
        raise ReferenceError_(f"trial 代码不存在: {code}")
    old_env = os.environ.get("MODELEVO_FEATURE_CACHE_DIR")
    if cache_dir is not None:
        os.environ["MODELEVO_FEATURE_CACHE_DIR"] = str(cache_dir)
    try:
        champ = _load_trial_module(
            code, f"mvo_ref_{trial_id.replace('-', '_')}")
        fb = getattr(champ, "trial_feature_blocks", None)
        if not callable(fb):
            fams = [s for s in ("build_wide", "_build_widen_columns",
                                "_widen", "_widen_pruned", "widen_face")
                    if hasattr(champ, s)]
            raise ReferenceError_(
                f"trial {trial_id} 不是第五族标准契约(无 trial_feature_blocks;"
                f"检测到加宽符号 {fams or '无'}),--from-trial 暂不支持;"
                "请用 base 面(去掉 --from-trial)")
        sd = Path(getattr(champ, "SPLITS", _splits_dir(run_dir)))
        if Path(sd).resolve() != _splits_dir(run_dir).resolve():
            raise ReferenceError_(
                f"trial 的 SPLITS({sd})不是本 run 的基础面,暂不支持视图面 trial")
        cfg = champ.load_config() if hasattr(champ, "load_config") else {}
        header = pq.read_schema(sd / "train.parquet").names
        feats = champ.exclude_columns(pd.DataFrame(columns=header), cfg)
        feats = [c for c in feats if c in set(header)]
        y_tr, y_va = _load_labels(run_dir, spec)
        X_tr = pd.read_parquet(sd / "train.parquet", columns=feats)
        X_va = pd.read_parquet(sd / "validation.parquet", columns=feats)
        X_ho = keys_ho = None
        if holdout:
            X_ho = pd.read_parquet(sd / f"{holdout}.parquet", columns=feats)
            keys_ho = _holdout_keys(run_dir, holdout)
            if len(keys_ho) != len(X_ho):
                raise ReferenceError_(f"{holdout} 特征面 {len(X_ho)} 行 != "
                                      f"identity {len(keys_ho)} 行")
            X_tr, X_va, gov, X_ho = govern_frames(X_tr, X_va, X_ho)
        else:
            X_tr, X_va, gov = govern_frames(X_tr, X_va)
        n_base = X_tr.shape[1]
        from .feature_cache import align_rows

        t0 = time.time()
        W_tr = align_rows(fb("train"), X_tr).astype("float32")
        W_va = align_rows(fb("validation"), X_va).astype("float32")
        dup = [c for c in W_tr.columns if c in X_tr.columns]
        if dup:
            raise ReferenceError_(f"特征块列与基础面重名: {dup[:5]}")
        X_tr = pd.concat([X_tr, W_tr], axis=1)
        X_va = pd.concat([X_va, W_va], axis=1)
        if X_ho is not None:
            W_ho = align_rows(fb(holdout), X_ho).astype("float32")
            X_ho = pd.concat([X_ho, W_ho], axis=1)
            del W_ho
        block_sec = time.time() - t0
        del W_tr, W_va
    finally:
        if cache_dir is not None:
            if old_env is None:
                os.environ.pop("MODELEVO_FEATURE_CACHE_DIR", None)
            else:
                os.environ["MODELEVO_FEATURE_CACHE_DIR"] = old_env
    n_full = len(y_tr)
    idx = _subsample_index(y_tr, max_rows or 0, task)
    X_tr, y_tr = _apply_subsample(X_tr, y_tr, idx)
    # 与 trial 自报的特征数对账(只作提示:治理口径不同时 one-hot 列数会不同)
    trial_n = None
    try:
        trial_n = int(load_json(tdir / "metrics.json").get("n_features"))
    except Exception:  # noqa: BLE001
        pass
    return FeatureSurface(X_tr, y_tr, X_va, y_va, f"trial:{trial_id}", {
        "n_features": int(X_tr.shape[1]), "n_base_columns": int(n_base),
        "n_block_columns": int(X_tr.shape[1] - n_base),
        "trial_reported_n_features": trial_n,
        "n_train": int(len(X_tr)), "n_train_full": int(n_full),
        "n_validation": int(len(X_va)), "subsampled": bool(len(idx) < n_full),
        "block_seconds": round(block_sec, 1), "governance": gov,
        "family": "trial_feature_blocks",
        "n_holdout": None if X_ho is None else int(len(X_ho)),
        "note": "基础列按 trial 配置(exclude_columns)、特征块经缓存复建;"
                "dtype 治理用参照口径(原生类别 + 频次编码),与 trial 的 "
                "one-hot 治理信息等价或更多"},
        X_ho=X_ho, keys_ho=keys_ho, holdout_split=holdout)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def _acquire_gpu(leases_dir: Path, n_cells: float):
    """规模够大且 xgboost 带 CUDA 时经 GpuLease 租一张卡;否则 (None, None)。"""
    if n_cells < GPU_MIN_CELLS or os.environ.get("MODELEVO_REF_DEVICE") == "cpu":
        return None, None
    try:
        import xgboost

        if not xgboost.build_info().get("USE_CUDA"):
            return None, None
        from .scheduler import GpuLease, gpu_candidates, resource_snapshot

        for g in gpu_candidates(resource_snapshot()):
            if g.get("memory_free_gb", 0) < 10:
                continue
            lease = GpuLease.acquire(Path(leases_dir), int(g["index"]))
            if lease is not None:
                return lease, int(g["index"])
    except Exception as exc:  # noqa: BLE001 —— 租卡失败退回 CPU
        logger.warning("GPU 租约失败,参照退回 CPU: %s", exc)
    return None, None


# ---------------------------------------------------------------------------
# engine: strong
# ---------------------------------------------------------------------------

def _fit_predict(algo: str, task: dict, s: FeatureSurface, params: dict,
                 seed: int, n_jobs: int, device: str | None) -> tuple:
    from .strong_default import (STRONG_EARLY_STOP_ROUNDS,
                                 STRONG_N_ESTIMATORS_CAP)

    clf = task["model_type"] == "binary_classification"
    if algo == "xgboost":
        import xgboost as xgb

        kw = dict(params, n_estimators=STRONG_N_ESTIMATORS_CAP,
                  early_stopping_rounds=STRONG_EARLY_STOP_ROUNDS,
                  random_state=seed, n_jobs=n_jobs, enable_categorical=True,
                  max_cat_to_onehot=1)
        if device:
            kw["device"] = device
        if clf:
            kw["eval_metric"] = "auc" if task["primary_metric"] == "auc" \
                else "logloss"
            m = xgb.XGBClassifier(**kw)
        else:
            kw["eval_metric"] = "mae" if task["primary_metric"] == "mae" \
                else "rmse"
            m = xgb.XGBRegressor(**kw)
        m.fit(s.X_tr, s.y_tr, eval_set=[(s.X_va, s.y_va)], verbose=False)
        if device:
            m.set_params(device="cpu")
        best = int(getattr(m, "best_iteration", -1) or -1)
    elif algo == "lightgbm":
        import lightgbm as lgb

        kw = dict(params, n_estimators=STRONG_N_ESTIMATORS_CAP,
                  random_state=seed, n_jobs=n_jobs, verbose=-1)
        cb = [lgb.early_stopping(STRONG_EARLY_STOP_ROUNDS, verbose=False)]
        if clf:
            m = lgb.LGBMClassifier(**kw)
            metric = "auc" if task["primary_metric"] == "auc" else "binary_logloss"
        else:
            m = lgb.LGBMRegressor(**kw)
            metric = "l1" if task["primary_metric"] == "mae" else "l2"
        m.fit(s.X_tr, s.y_tr, eval_set=[(s.X_va, s.y_va)],
              eval_metric=metric, callbacks=cb)
        best = int(getattr(m, "best_iteration_", -1) or -1)
    else:
        raise ReferenceError_(f"未知参照算法 {algo}")
    pred = m.predict_proba(s.X_va)[:, 1] if clf else m.predict(s.X_va)
    pred_ho = None
    if s.X_ho is not None:  # v2.19.0:对齐测试面只打分,不读标签
        pred_ho = np.asarray(m.predict_proba(s.X_ho)[:, 1] if clf
                             else m.predict(s.X_ho), dtype="float64")
    return np.asarray(pred, dtype="float64"), best, pred_ho


def run_strong_family(algo: str, task: dict, s: FeatureSurface, *,
                      n_seeds: int, deadline: float, n_jobs: int,
                      device: str | None = None,
                      base_seed: int = 20261101,
                      log: Callable[[str], None] = print) -> dict:
    """一族强参数的 K seed bagging(validation 早停);按截止时间缩减 seed 数。"""
    from .strong_default import strong_params

    from .strong_default import STRONG_N_ESTIMATORS_CAP

    params = strong_params(algo, len(s.X_tr), s.X_tr.shape[1])
    preds, seed_vals, iters, secs = [], [], [], []
    preds_ho: list = []
    t_start = time.time()
    cap_adjust = None
    for k in range(int(n_seeds)):
        if k > 0:
            est = float(np.mean(secs))
            if time.time() + est > deadline:
                log(f"[ceiling] {algo}: 预算不足,bag 停在 {k} 个 seed")
                break
        t0 = time.time()
        p, best, p_ho = _fit_predict(algo, task, s, params, base_seed + k,
                                     n_jobs,
                                     device if algo == "xgboost" else None)
        # v2.19.0:首个 seed 撞轮数上限(req-t7 254 万行实测 4999/5000,参照被
        # 截断)→ 学习率 ×2.5 重训该 seed,其余 seed 沿用新学习率;留痕
        if k == 0 and best >= 0.98 * STRONG_N_ESTIMATORS_CAP \
                and cap_adjust is None and time.time() < deadline:
            lr0 = float(params.get("learning_rate", 0.02))
            params = {**params, "learning_rate": round(lr0 * 2.5, 4)}
            cap_adjust = {"from_lr": lr0, "to_lr": params["learning_rate"],
                          "best_iter_at_cap": int(best),
                          "cap": STRONG_N_ESTIMATORS_CAP}
            log(f"[ceiling] {algo}: 撞轮数上限 {best}/"
                f"{STRONG_N_ESTIMATORS_CAP},lr {lr0} → "
                f"{params['learning_rate']} 重训")
            p, best, p_ho = _fit_predict(
                algo, task, s, params, base_seed + k, n_jobs,
                device if algo == "xgboost" else None)
        secs.append(time.time() - t0)
        preds.append(p)
        preds_ho.append(p_ho)
        iters.append(best)
        seed_vals.append(score_predictions(task, s.y_va, p))
        log(f"[ceiling] {algo} seed {k}: {task['primary_metric']}="
            f"{seed_vals[-1]:.6f} best_iter={best} ({secs[-1]:.0f}s)")
    if not preds:
        return {"status": "skipped", "reason": "预算不足以训练一个成员"}
    bag = np.mean(preds, axis=0)
    return {
        "status": "ok",
        "value": score_predictions(task, s.y_va, bag),
        "single_seed_value": seed_vals[0],
        "seed_values": seed_vals,
        "seed_std": float(np.std(seed_vals)) if len(seed_vals) > 1 else None,
        "n_seeds": len(preds), "n_seeds_requested": int(n_seeds),
        "best_iterations": iters,
        "cap_bound": bool(iters and max(iters) >= 0.98
                          * STRONG_N_ESTIMATORS_CAP),
        "cap_adjust": cap_adjust,
        "seconds": round(time.time() - t_start, 1),
        "params": {**params, "early_stopping": "validation",
                   "device": device if algo == "xgboost" and device else "cpu"},
        "_pred": bag,
        # v2.19.0:对齐测试面预测(bag 与首个 seed 单模型),由 compute_ceiling 落盘
        "_pred_ho": (np.mean(preds_ho, axis=0)
                     if preds_ho and preds_ho[0] is not None else None),
        "_pred_ho_single": preds_ho[0] if preds_ho else None,
    }


def run_strong(task: dict, s: FeatureSurface, *, budget_sec: float,
               n_seeds: int = 5, n_jobs: int = 16,
               leases_dir: Path | None = None,
               algos: tuple[str, ...] = ("xgboost", "lightgbm"),
               log: Callable[[str], None] = print) -> dict[str, dict]:
    """strong engine:xgboost(必跑)+ lightgbm(预算剩余时)+ 两族 bag 混合。"""
    t0 = time.time()
    deadline = t0 + budget_sec
    out: dict[str, dict] = {}
    lease, gpu = (None, None)
    if leases_dir is not None:
        lease, gpu = _acquire_gpu(leases_dir, float(len(s.X_tr))
                                  * s.X_tr.shape[1])
    device = f"cuda:{gpu}" if gpu is not None else None
    try:
        for i, algo in enumerate(algos):
            if i > 0 and time.time() > t0 + 0.6 * budget_sec:
                out[f"strong_{algo}"] = {
                    "status": "skipped", "reason": "预算已用超 60%,跳过"}
                continue
            try:
                out[f"strong_{algo}"] = run_strong_family(
                    algo, task, s, n_seeds=n_seeds, deadline=deadline,
                    n_jobs=n_jobs, device=device, log=log)
            except ImportError as exc:
                out[f"strong_{algo}"] = {"status": "unavailable",
                                         "reason": str(exc)}
    finally:
        if lease is not None:
            lease.release()
    ok = [k for k, v in out.items() if v.get("status") == "ok"]
    if len(ok) >= 2:
        blend = np.mean([out[k]["_pred"] for k in ok], axis=0)
        blend_ho = None
        if all(out[k].get("_pred_ho") is not None for k in ok):
            blend_ho = np.mean([out[k]["_pred_ho"] for k in ok], axis=0)
        out["strong_blend"] = {
            "_pred_ho": blend_ho,
            "status": "ok", "value": score_predictions(task, s.y_va, blend),
            "members": ok, "seconds": round(sum(out[k]["seconds"] for k in ok), 1),
            "n_seeds": int(sum(out[k]["n_seeds"] for k in ok)),
            "params": {"blend": "等权平均各族 bag 预测"}}
    for v in out.values():
        v.pop("_pred", None)  # _pred_ho* 留给 compute_ceiling 落盘后再删
    return out


# ---------------------------------------------------------------------------
# engine: autogluon(独立 venv 子进程)
# ---------------------------------------------------------------------------

AG_SCRIPT = r'''
import json, sys, time
import pandas as pd
from autogluon.tabular import TabularPredictor
import autogluon.tabular as agt
a = json.loads(sys.argv[1])
tr = pd.read_parquet(a["train"])
va = pd.read_parquet(a["valid_features"])
t0 = time.time()
p = TabularPredictor(label=a["label"], problem_type=a["problem_type"],
                     eval_metric=a["eval_metric"], path=a["model_dir"],
                     verbosity=1)
p.fit(tr, presets=a["presets"], time_limit=a["time_limit"],
      num_cpus=a.get("num_cpus", "auto"), num_gpus=0)
def _pr(frame, model=None):
    if a["problem_type"] == "binary":
        return p.predict_proba(frame, model=model)[p.positive_class].to_numpy()
    return p.predict(frame, model=model).to_numpy()
pred = _pr(va)
pd.DataFrame({"pred": pred}).to_parquet(a["pred_out"], index=False)
ho = pd.read_parquet(a["holdout_features"]) if a.get("holdout_features") else None
if ho is not None:
    pd.DataFrame({"pred": _pr(ho)}).to_parquet(a["holdout_pred_out"], index=False)
lb = p.leaderboard(silent=True)
info = {"version": agt.__version__, "fit_seconds": time.time() - t0,
        "best_model": str(p.model_best), "n_models": int(len(lb)),
        "positive_class": str(getattr(p, "positive_class", None))}
# v2.19.0:与框架同口径的单模型参照 —— 按 AutoGluon 自己的内部验证分
# (score_val,来自 k 折 OOF,不用我们的 validation 标签)选 L1 最优基模型,
# refit_full 成"全量 train 上训一次、无折 bagging"的单模型再打分。
try:
    base = lb[(lb["stack_level"] == 1)
              & (~lb["model"].str.contains("Ensemble"))
              & (~lb["model"].str.endswith("_FULL"))]
    if len(base):
        name = str(base.sort_values("score_val", ascending=False)
                   .iloc[0]["model"])
        t1 = time.time()
        full = p.refit_full(model=name).get(name, name)
        sp = _pr(va, full)
        pd.DataFrame({"pred": sp}).to_parquet(a["single_pred_out"],
                                              index=False)
        if ho is not None:
            pd.DataFrame({"pred": _pr(ho, full)}).to_parquet(
                a["single_holdout_pred_out"], index=False)
        info.update(single_model=name, single_model_refit=str(full),
                    single_model_score_val=float(
                        base.sort_values("score_val", ascending=False)
                        .iloc[0]["score_val"]),
                    single_refit_seconds=time.time() - t1)
except Exception as exc:  # noqa: BLE001 —— 单模型参照失败不影响集成参照
    info["single_model_error"] = repr(exc)[:300]
json.dump(info, open(a["info_out"], "w"))
'''


def resolve_autogluon_python(explicit: str | None = None) -> str | None:
    for cand in (explicit, os.environ.get(AUTOGLUON_PYTHON_ENV),
                 DEFAULT_AUTOGLUON_PYTHON):
        if cand and Path(cand).exists():
            return str(cand)
    return None


def run_autogluon(task: dict, s: FeatureSurface, *, budget_sec: float,
                  work_dir: Path, python: str | None = None,
                  n_jobs: int = 16, keep_models: bool = False,
                  log: Callable[[str], None] = print) -> dict:
    """AutoGluon best_quality;只用 train 训练,validation 特征打分(标签不外传)。"""
    py = resolve_autogluon_python(python)
    if py is None:
        return {"status": "unavailable",
                "reason": f"AutoGluon 解释器不存在(--autogluon-python / "
                          f"{AUTOGLUON_PYTHON_ENV} / {DEFAULT_AUTOGLUON_PYTHON})"}
    probe = subprocess.run([py, "-c", "import autogluon.tabular as t;"
                            "print(t.__version__)"],
                           capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        return {"status": "unavailable",
                "reason": "autogluon.tabular 不可导入: "
                          + (probe.stderr.strip().splitlines() or ["?"])[-1]}
    work = Path(work_dir)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    label = "__mvo_ref_label"
    tr = s.X_tr.copy()
    tr[label] = s.y_tr
    tr.to_parquet(work / "train.parquet", index=False)
    s.X_va.to_parquet(work / "valid_features.parquet", index=False)
    if s.X_ho is not None:
        s.X_ho.to_parquet(work / "holdout_features.parquet", index=False)
    del tr
    args = {"train": str(work / "train.parquet"),
            "valid_features": str(work / "valid_features.parquet"),
            "label": label,
            "problem_type": "binary" if task["model_type"]
            == "binary_classification" else "regression",
            "eval_metric": (CLASSIFICATION_METRICS | REGRESSION_METRICS)[
                task["primary_metric"]],
            "model_dir": str(work / "models"), "presets": "best_quality",
            "time_limit": int(budget_sec), "num_cpus": int(n_jobs),
            "pred_out": str(work / "pred.parquet"),
            "single_pred_out": str(work / "single_pred.parquet"),
            "holdout_features": (str(work / "holdout_features.parquet")
                                 if s.X_ho is not None else None),
            "holdout_pred_out": str(work / "holdout_pred.parquet"),
            "single_holdout_pred_out": str(work / "single_holdout_pred.parquet"),
            "info_out": str(work / "info.json")}
    (work / "ag_fit.py").write_text(AG_SCRIPT, encoding="utf-8")
    t0 = time.time()
    log(f"[ceiling] autogluon: best_quality time_limit={int(budget_sec)}s "
        f"train={len(s.X_tr)} features={s.X_tr.shape[1]}")
    try:
        with open(work / "ag.log", "w", encoding="utf-8") as fh:
            proc = subprocess.run([py, "-I", str(work / "ag_fit.py"),
                                   json.dumps(args)], cwd=str(work),
                                  stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=budget_sec * 2 + 900)
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": "AutoGluon 子进程超时",
                "seconds": round(time.time() - t0, 1)}
    finally:
        for f in ("train.parquet", "valid_features.parquet",
                  "holdout_features.parquet"):
            (work / f).unlink(missing_ok=True)
        if not keep_models:
            shutil.rmtree(work / "models", ignore_errors=True)
    secs = round(time.time() - t0, 1)
    if proc.returncode != 0 or not (work / "pred.parquet").exists():
        tail = (work / "ag.log").read_text(encoding="utf-8",
                                           errors="replace")[-1500:]
        return {"status": "failed", "seconds": secs,
                "reason": f"AutoGluon 退出码 {proc.returncode}", "log_tail": tail}
    pred = pd.read_parquet(work / "pred.parquet")["pred"].to_numpy()
    info = json.loads((work / "info.json").read_text(encoding="utf-8"))
    single = None
    if (work / "single_pred.parquet").exists():
        sp = pd.read_parquet(work / "single_pred.parquet")["pred"].to_numpy()
        single = score_predictions(task, s.y_va, sp)
    def _ho(name):
        f = work / name
        return pd.read_parquet(f)["pred"].to_numpy() if f.exists() else None
    return {"status": "ok", "value": score_predictions(task, s.y_va, pred),
            # v2.19.0:同一次训练里的最佳单模型(refit_full,无 bagging)
            "single_model_value": single,
            "_pred_ho": _ho("holdout_pred.parquet"),
            "_pred_ho_single": _ho("single_holdout_pred.parquet"),
            "seconds": secs, "n_seeds": 1,
            "engine_info": {**info, "presets": "best_quality",
                            "time_limit": int(budget_sec),
                            "tuning_data": None,
                            "python": py, "log": str(work / "ag.log")}}


# ---------------------------------------------------------------------------
# baseline 读取与主入口
# ---------------------------------------------------------------------------

def _single_model_reference(task: dict, refs: dict) -> tuple[str | None,
                                                             float | None]:
    """单模型口径的最优参照:strong 各族的单 seed 值 + AutoGluon refit 单模型。"""
    cands: list[tuple[str, float]] = []
    for name, r in refs.items():
        if r.get("status") != "ok":
            continue
        for key, label in (("single_seed_value", f"{name}:single_seed"),
                           ("single_model_value", f"{name}:single_model")):
            v = r.get(key)
            if v is not None and math.isfinite(float(v)):
                cands.append((label, float(v)))
    best = None
    for label, v in cands:
        if best is None or _better(task, v, best[1]):
            best = (label, v)
    return best if best else (None, None)


def _persist_holdout(root: Path, s: FeatureSurface, refs: dict) -> dict | None:
    """对齐测试面预测落盘(并从 refs 中移除内部数组)。

    每个参照至多两份:``<名>.parquet``(该参照的主口径:strong 为 K-seed bag、
    AutoGluon 为集成、FLAML/TabPFN 为单模型)与 ``<名>__single.parquet``(单模型
    口径:strong 首个 seed、AutoGluon refit_full 最佳基模型)。
    """
    import hashlib

    out_files: dict[str, dict] = {}
    d = root / "ceiling" / "holdout"
    if s.X_ho is not None:
        import shutil

        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    for name, r in refs.items():
        ho = r.pop("_pred_ho", None)
        ho_single = r.pop("_pred_ho_single", None)
        if s.X_ho is None:
            continue
        single_kind = name in ("flaml", "tabpfn_v2")
        for suffix, arr, kind in (
                ("", ho, "single_model" if single_kind else "ensemble"),
                ("__single", None if single_kind else ho_single,
                 "single_model")):
            if arr is None:
                continue
            arr = np.asarray(arr, dtype="float64")
            if len(arr) != len(s.keys_ho):
                raise ReferenceError_(f"{name}{suffix} 测试面预测 {len(arr)} 行 "
                                      f"!= 样本键 {len(s.keys_ho)} 行")
            f = d / f"{name}{suffix}.parquet"
            pd.DataFrame({"score": arr, "__mvo_sample_key": s.keys_ho}
                         ).to_parquet(f, index=False)
            out_files[f"{name}{suffix}"] = {"file": str(f), "kind": kind,
                                            "trained_on": "train"}
    if s.X_ho is None:
        return None
    sha = hashlib.sha256("\n".join(sorted(map(str, s.keys_ho))).encode()
                         ).hexdigest()[:16]
    return {"split": s.holdout_split, "n_rows": int(len(s.keys_ho)),
            "keys_sha256_16": sha, "files": out_files}


def baseline_reference(run_dir: Path, task: dict) -> tuple[str | None,
                                                            float | None]:
    """run-state baseline_id 对应 trial 的 validation 主指标(evaluation.json)。"""
    st_path = Path(run_dir) / "run-state.json"
    bid = (load_json(st_path) if st_path.exists() else {}).get("baseline_id")
    if not bid:
        return None, None
    tdir = Path(run_dir) / "trials" / bid
    val = None
    ev = tdir / "evaluation.json"
    if ev.exists():
        pm = (load_json(ev).get("primary_metric") or {})
        if str(pm.get("name", "")).lower() == task["primary_metric"]:
            val = pm.get("value")
    if val is None and (tdir / "metrics.json").exists():
        val = load_json(tdir / "metrics.json").get(
            f"valid_{task['primary_metric']}")
    try:
        val = float(val) if val is not None else None
    except (TypeError, ValueError):
        val = None
    return bid, val


def compute_ceiling(run_dir: "Path | str", spec, *,
                    engines: tuple[str, ...] = ("strong",),
                    budget_minutes: float = 45.0,
                    out_dir: "Path | str | None" = None,
                    max_rows: int | None = None,
                    from_trial: str | None = None,
                    n_seeds: int = 5, n_jobs: int | None = None,
                    autogluon_python: str | None = None,
                    lgbm: bool = True,
                    autogluon_budget_minutes: float | None = None,
                    refs_python: str | None = None,
                    holdout: str | None = None,
                    log: Callable[[str], None] = print) -> dict:
    """跑参照并写 ceiling-estimate.json;返回 payload(含 ``path``)。

    ``autogluon_budget_minutes``(v2.19.0):AutoGluon 单独的预算;用于与框架
    run 的实际 compute 时长对齐(``ceiling --match-run-compute``),否则沿用
    ``budget_minutes``。

    ``holdout``(v2.19.0,如 ``shadow_test``):每个参照训练后在该面上**只打分**,
    预测按样本键落盘到 ``<ceiling>/holdout/<参照>.parquet``(列 score +
    __mvo_sample_key),供 ``benchmark aligned`` 在同一批行上统一评分;这里不
    读该面的标签。

    out_dir=None → 写 ``<run>/ceiling/``;否则一切写入落在 out_dir
    (``<out_dir>/ceiling/ceiling-estimate.json``,load_ceiling(out_dir) 可读)。
    预算按 engine 计:每个 engine 各得 ``budget_minutes``。
    """
    run_dir = Path(run_dir).resolve()
    root = Path(out_dir).resolve() if out_dir else run_dir
    if out_dir and (root == run_dir or run_dir in root.parents):
        raise ReferenceError_("--out-dir 不能位于 run 目录内(只读模式要求零写入)")
    task = resolve_task(spec)
    unknown = [e for e in engines
               if e not in ("strong", "autogluon", "flaml", "tabpfn")]
    if unknown:
        raise ReferenceError_(
            f"未知 engine: {unknown}(支持 strong, autogluon, flaml, tabpfn)")
    n_jobs = int(n_jobs or max(8, min(32, (os.cpu_count() or 16) - 8)))
    work_root = root / "ceiling" / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    t_load = time.time()
    if from_trial:
        cache = None
        if out_dir:
            from .feature_cache import cache_root

            cache = work_root / "feature-cache-mirror"
            n_link = _mirror_feature_cache(cache_root(run_dir), cache)
            log(f"[ceiling] 只读模式:特征块缓存镜像 {n_link} 个文件 → {cache}")
        surface = load_trial_surface(run_dir, spec, task, from_trial,
                                     max_rows=max_rows, cache_dir=cache,
                                     holdout=holdout)
    else:
        surface = load_base_surface(run_dir, spec, task, max_rows=max_rows,
                                    holdout=holdout)
    load_sec = round(time.time() - t_load, 1)
    log(f"[ceiling] 特征面 {surface.surface}: train={len(surface.X_tr)} "
        f"valid={len(surface.X_va)} features={surface.X_tr.shape[1]} "
        f"({load_sec}s)")
    budget_sec = float(budget_minutes) * 60.0
    refs: dict[str, dict] = {}
    if "strong" in engines:
        refs.update(run_strong(
            task, surface, budget_sec=budget_sec, n_seeds=n_seeds,
            n_jobs=n_jobs, leases_dir=root / ".gpu-leases",
            algos=("xgboost", "lightgbm") if lgbm else ("xgboost",), log=log))
    if "autogluon" in engines:
        refs["autogluon_best_quality"] = run_autogluon(
            task, surface,
            budget_sec=(float(autogluon_budget_minutes) * 60.0
                        if autogluon_budget_minutes else budget_sec),
            work_dir=work_root / "autogluon", python=autogluon_python,
            n_jobs=n_jobs, log=log)
    # v2.19.0:FLAML(单模型 AutoML)/ TabPFN v2(≤1 万行),见 runtime/refengines
    if "flaml" in engines or "tabpfn" in engines:
        from .refengines import TABPFN_MAX_ROWS, run_flaml, run_tabpfn

        if "flaml" in engines:
            refs["flaml"] = run_flaml(
                task, surface, budget_sec=budget_sec,
                work_dir=work_root / "flaml", python=refs_python,
                n_jobs=n_jobs, log=log)
        if "tabpfn" in engines:
            if len(surface.X_tr) > TABPFN_MAX_ROWS * 5:
                refs["tabpfn_v2"] = {
                    "status": "skipped",
                    "reason": (f"train {len(surface.X_tr)} 行远超 TabPFN 适用"
                               f"规模(≤{TABPFN_MAX_ROWS}),不跑")}
            else:
                refs["tabpfn_v2"] = run_tabpfn(
                    task, surface, budget_sec=budget_sec,
                    work_dir=work_root / "tabpfn", python=refs_python,
                    leases_dir=root / ".gpu-leases", log=log)
    holdout_meta = _persist_holdout(root, surface, refs)
    best_name, best_val = None, None
    for name, r in refs.items():
        v = r.get("value")
        if r.get("status") != "ok" or v is None or not math.isfinite(float(v)):
            continue
        if best_val is None or _better(task, float(v), best_val):
            best_name, best_val = name, float(v)
    if best_val is None:
        raise ReferenceError_(
            "所有参照都未成功: " + json.dumps(
                {k: v.get("reason") for k, v in refs.items()},
                ensure_ascii=False))
    single_name, single_val = _single_model_reference(task, refs)
    bid, bval = baseline_reference(run_dir, task)
    payload = {
        "primary_metric": task["primary_metric"],
        "higher_is_better": task["higher_is_better"],
        "model_type": task["model_type"],
        "ceiling_estimate": best_val,
        "best_reference": best_name,
        # v2.19.0:与框架交付同口径(单模型,无 seed/fold bagging、无集成)的
        # 参照;ceiling_estimate − single_model_estimate = 集成/bagging 溢价,
        # 即"单模型约束的代价"的估计
        "single_model_estimate": single_val,
        "single_model_reference": single_name,
        "ensemble_premium": (None if single_val is None else (
            best_val - single_val if task["higher_is_better"]
            else single_val - best_val)),
        "references": refs,
        "baseline_trial": bid,
        "baseline_value": bval,
        "feature_surface": surface.surface,
        "surface_info": surface.info,
        "max_rows": max_rows,
        "engines": list(engines),
        "budget_minutes": float(budget_minutes),
        "autogluon_budget_minutes": (float(autogluon_budget_minutes)
                                     if autogluon_budget_minutes else None),
        "load_seconds": load_sec,
        "computed_at": now_iso(),
        "run_dir": str(run_dir),
        "read_only_out_dir": str(root) if out_dir else None,
        "holdout": holdout_meta,
    }
    if bval is not None:
        from .ceiling import gap_to_ref

        payload["baseline_gap_to_ref"] = gap_to_ref(payload, bval)
    path = write_ceiling(root, payload)
    payload["path"] = str(path)
    # 工作区为空则删掉(只读模式保留特征块镜像以便复用)
    try:
        if work_root.exists() and not any(work_root.iterdir()):
            work_root.rmdir()
    except OSError:
        pass
    return payload


__all__ = ["ReferenceError_", "FeatureSurface", "resolve_task",
           "govern_frames", "load_base_surface", "load_trial_surface",
           "run_strong", "run_strong_family", "run_autogluon",
           "baseline_reference", "compute_ceiling", "score_predictions"]
