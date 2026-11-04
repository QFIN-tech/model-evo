"""v2.19.0 追加参照引擎:FLAML(单模型 AutoML)与 TabPFN v2(小样本)。

放在子包里而不是 runtime 顶层:harness hash 只覆盖 ``runtime/*.py``,参照引擎
不参与任何裁决,新增/修订不应让在途 run 的协议 hash 漂移。

两者都在独立 venv 子进程里运行(``--refs-python`` / 环境变量
``MODELEVO_REFS_PYTHON`` / 缺省路径),与 runtime 解释器的依赖隔离;只把
train(特征 + 标签)与 validation **特征**交给子进程,validation 标签不出主进程,
打分在主进程完成(与 AutoGluon engine 同口径)。

- FLAML:``AutoML.fit`` 只用 train(内部 holdout/CV 选模型与超参),
  ``retrain_full`` 后输出**一个**模型 —— 与 ModelEvo 交付同口径的单模型参照。
- TabPFN v2:预训练表格基础模型,上下文学习;只对 ``n_train ≤ max_rows``
  (缺省 10000)的数据集运行,超出时分层抽样到上限并留痕。推理期的排列集成
  是模型自身的一部分,按单模型参照计。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

REFS_PYTHON_ENV = "MODELEVO_REFS_PYTHON"
DEFAULT_REFS_PYTHON = ("/data/oceanus_share/j-wangyaoxuan-jk/modelevo-202611/"
                       "venvs/refs/bin/python")
TABPFN_MAX_ROWS = 10000
#: TabPFN v2 公开权重(HuggingFace Prior-Labs/TabPFN-v2-clf / -reg,非 gated)。
#: 新版 tabpfn 包缺省下载 gated 的 v3 权重(需 HF 账号接受条款),这里显式
#: 指定 v2 权重路径;目录可用环境变量 MODELEVO_TABPFN_MODEL_DIR 覆盖
TABPFN_MODEL_DIR_ENV = "MODELEVO_TABPFN_MODEL_DIR"
DEFAULT_TABPFN_MODEL_DIR = ("/data/oceanus_share/j-wangyaoxuan-jk/"
                            "modelevo-202611/venvs/refs/tabpfn-models")

FLAML_SCRIPT = r'''
import json, sys, time
import pandas as pd
from flaml import AutoML
import flaml
a = json.loads(sys.argv[1])
tr = pd.read_parquet(a["train"])
va = pd.read_parquet(a["valid_features"])
y = tr.pop(a["label"])
t0 = time.time()
m = AutoML()
m.fit(X_train=tr, y_train=y, task=a["task"], metric=a["metric"],
      time_budget=a["time_budget"], n_jobs=a["n_jobs"], seed=a["seed"],
      retrain_full=True, verbose=0)
def _pr(frame):
    return m.predict_proba(frame)[:, 1] if a["task"] == "classification" \
        else m.predict(frame)
pd.DataFrame({"pred": _pr(va)}).to_parquet(a["pred_out"], index=False)
if a.get("holdout_features"):
    ho = pd.read_parquet(a["holdout_features"])
    pd.DataFrame({"pred": _pr(ho)}).to_parquet(a["holdout_pred_out"],
                                                index=False)
json.dump({"version": flaml.__version__, "fit_seconds": time.time() - t0,
           "best_estimator": str(m.best_estimator),
           "best_config": {k: (v if isinstance(v, (int, float, str, bool))
                               else str(v))
                           for k, v in (m.best_config or {}).items()},
           "best_loss": float(m.best_loss)}, open(a["info_out"], "w"))
'''

TABPFN_SCRIPT = r'''
import json, sys, time
import numpy as np
import pandas as pd
a = json.loads(sys.argv[1])
from tabpfn import TabPFNClassifier, TabPFNRegressor
tr = pd.read_parquet(a["train"])
va = pd.read_parquet(a["valid_features"])
y = tr.pop(a["label"]).to_numpy()
ho = pd.read_parquet(a["holdout_features"]) if a.get("holdout_features") else None
for c in tr.columns:  # 类别列转整数码;类别表只来自 train(未见值 = -1)
    if str(tr[c].dtype) in ("category", "object"):
        cats = pd.Categorical(tr[c].astype(str)).categories
        tr[c] = pd.Categorical(tr[c].astype(str), categories=cats).codes
        va[c] = pd.Categorical(va[c].astype(str), categories=cats).codes
        if ho is not None:
            ho[c] = pd.Categorical(ho[c].astype(str), categories=cats).codes
t0 = time.time()
kw = dict(device=a["device"], ignore_pretraining_limits=True,
          random_state=a["seed"])
if a.get("model_path"):
    kw["model_path"] = a["model_path"]
m = TabPFNClassifier(**kw) if a["task"] == "classification" \
    else TabPFNRegressor(**kw)
m.fit(tr.to_numpy(dtype="float32"), y)
def _pr(frame):
    X = frame.to_numpy(dtype="float32")
    preds = []
    for i in range(0, len(X), 5000):  # 分块推理,控显存
        xb = X[i:i + 5000]
        preds.append(m.predict_proba(xb)[:, 1] if a["task"] == "classification"
                     else m.predict(xb))
    return np.concatenate(preds)
pd.DataFrame({"pred": _pr(va)}).to_parquet(a["pred_out"], index=False)
if ho is not None:
    pd.DataFrame({"pred": _pr(ho)}).to_parquet(a["holdout_pred_out"],
                                                index=False)
import tabpfn
json.dump({"version": getattr(tabpfn, "__version__", "?"),
           "fit_predict_seconds": time.time() - t0, "device": a["device"]},
          open(a["info_out"], "w"))
'''


def _tabpfn_weights(clf: bool) -> str | None:
    d = Path(os.environ.get(TABPFN_MODEL_DIR_ENV) or DEFAULT_TABPFN_MODEL_DIR)
    f = d / ("tabpfn-v2-classifier.ckpt" if clf else "tabpfn-v2-regressor.ckpt")
    return str(f) if f.exists() else None


def resolve_refs_python(explicit: str | None = None) -> str | None:
    for cand in (explicit, os.environ.get(REFS_PYTHON_ENV),
                 DEFAULT_REFS_PYTHON):
        if cand and Path(cand).exists():
            return str(cand)
    return None


def _probe(py: str, module: str) -> str | None:
    """子进程能否 import module;不能时返回原因。"""
    pr = subprocess.run([py, "-c", f"import {module}"], capture_output=True,
                        text=True, timeout=300)
    if pr.returncode == 0:
        return None
    return (pr.stderr.strip().splitlines() or ["?"])[-1]


def _run_script(*, name: str, script: str, args: dict, py: str, work: Path,
                timeout: float, env: dict | None = None) -> tuple[int, float]:
    (work / f"{name}_fit.py").write_text(script, encoding="utf-8")
    t0 = time.time()
    try:
        with open(work / f"{name}.log", "w", encoding="utf-8") as fh:
            proc = subprocess.run(
                [py, "-I", str(work / f"{name}_fit.py"), json.dumps(args)],
                cwd=str(work), stdout=fh, stderr=subprocess.STDOUT,
                timeout=timeout, env={**os.environ, **(env or {})})
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = 124
    return rc, round(time.time() - t0, 1)


def _write_inputs(s, work: Path, label: str, idx: np.ndarray | None = None
                  ) -> None:
    tr = s.X_tr if idx is None else s.X_tr.iloc[idx].reset_index(drop=True)
    y = s.y_tr if idx is None else np.asarray(s.y_tr)[idx]
    tr = tr.copy()
    tr[label] = y
    tr.to_parquet(work / "train.parquet", index=False)
    s.X_va.to_parquet(work / "valid_features.parquet", index=False)
    if s.X_ho is not None:
        s.X_ho.to_parquet(work / "holdout_features.parquet", index=False)


def _holdout_args(s, work: Path) -> dict:
    if s.X_ho is None:
        return {"holdout_features": None}
    return {"holdout_features": str(work / "holdout_features.parquet"),
            "holdout_pred_out": str(work / "holdout_pred.parquet")}


def _finish(task: dict, s, work: Path, name: str, rc: int, secs: float,
            extra: dict) -> dict:
    from ..reference import score_predictions

    for f in ("train.parquet", "valid_features.parquet",
              "holdout_features.parquet"):
        (work / f).unlink(missing_ok=True)
    if rc != 0 or not (work / "pred.parquet").exists():
        tail = ""
        if (work / f"{name}.log").exists():
            tail = (work / f"{name}.log").read_text(
                encoding="utf-8", errors="replace")[-1500:]
        return {"status": "failed", "seconds": secs,
                "reason": f"{name} 退出码 {rc}", "log_tail": tail}
    pred = pd.read_parquet(work / "pred.parquet")["pred"].to_numpy()
    info = json.loads((work / "info.json").read_text(encoding="utf-8")) \
        if (work / "info.json").exists() else {}
    v = score_predictions(task, s.y_va, pred)
    # 单模型口径:与 AutoGluon 的 single_model_value 同键,供
    # reference._single_model_reference 统一挑选
    ho = work / "holdout_pred.parquet"
    p_ho = pd.read_parquet(ho)["pred"].to_numpy() if ho.exists() else None
    return {"status": "ok", "value": v, "single_model_value": v,
            "_pred_ho": p_ho, "_pred_ho_single": p_ho,
            "seconds": secs, "n_seeds": 1,
            "engine_info": {**info, **extra, "log": str(work / f"{name}.log")}}


def run_flaml(task: dict, s, *, budget_sec: float, work_dir: Path,
              python: str | None = None, n_jobs: int = 16, seed: int = 0,
              log: Callable[[str], None] = print) -> dict:
    py = resolve_refs_python(python)
    if py is None:
        return {"status": "unavailable",
                "reason": f"参照 venv 解释器不存在({REFS_PYTHON_ENV} / "
                          f"{DEFAULT_REFS_PYTHON})"}
    why = _probe(py, "flaml.automl")
    if why:
        return {"status": "unavailable", "reason": f"flaml 不可导入: {why}"}
    work = Path(work_dir)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    label = "__mvo_ref_label"
    _write_inputs(s, work, label)
    clf = task["model_type"] == "binary_classification"
    metric = {"auc": "roc_auc", "logloss": "log_loss", "rmse": "rmse",
              "mae": "mae"}.get(task["primary_metric"],
                                "roc_auc" if clf else "rmse")
    args = {"train": str(work / "train.parquet"),
            "valid_features": str(work / "valid_features.parquet"),
            "label": label, "task": "classification" if clf else "regression",
            "metric": metric, "time_budget": int(budget_sec),
            "n_jobs": int(n_jobs), "seed": int(seed),
            "pred_out": str(work / "pred.parquet"),
            **_holdout_args(s, work),
            "info_out": str(work / "info.json")}
    log(f"[ceiling] flaml: time_budget={int(budget_sec)}s "
        f"train={len(s.X_tr)} features={s.X_tr.shape[1]}")
    rc, secs = _run_script(name="flaml", script=FLAML_SCRIPT, args=args,
                           py=py, work=work, timeout=budget_sec * 2 + 900)
    return _finish(task, s, work, "flaml", rc, secs,
                   {"time_budget": int(budget_sec), "metric": metric,
                    "retrain_full": True, "python": py})


def run_tabpfn(task: dict, s, *, budget_sec: float, work_dir: Path,
               python: str | None = None, max_rows: int = TABPFN_MAX_ROWS,
               leases_dir: Path | None = None, seed: int = 0,
               log: Callable[[str], None] = print) -> dict:
    py = resolve_refs_python(python)
    if py is None:
        return {"status": "unavailable",
                "reason": f"参照 venv 解释器不存在({REFS_PYTHON_ENV} / "
                          f"{DEFAULT_REFS_PYTHON})"}
    why = _probe(py, "tabpfn")
    if why:
        return {"status": "unavailable", "reason": f"tabpfn 不可导入: {why}"}
    from ..reference import _acquire_gpu, _subsample_index

    work = Path(work_dir)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    n_full = len(s.y_tr)
    idx = None
    if n_full > max_rows:
        idx = _subsample_index(np.asarray(s.y_tr), max_rows, task)
    label = "__mvo_ref_label"
    _write_inputs(s, work, label, idx)
    lease, gpu = (None, None)
    if leases_dir is not None:
        lease, gpu = _acquire_gpu(Path(leases_dir), 1e9)  # 强制尝试租卡
    env = {"HF_ENDPOINT": os.environ.get("HF_ENDPOINT",
                                         "https://hf-mirror.com")}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    clf = task["model_type"] == "binary_classification"
    args = {"train": str(work / "train.parquet"),
            "valid_features": str(work / "valid_features.parquet"),
            "label": label, "task": "classification" if clf else "regression",
            "device": "cuda" if gpu is not None else "cpu",
            "model_path": _tabpfn_weights(clf),
            "seed": int(seed), "pred_out": str(work / "pred.parquet"),
            **_holdout_args(s, work),
            "info_out": str(work / "info.json")}
    n_used = len(idx) if idx is not None else n_full
    log(f"[ceiling] tabpfn: train={n_used}/{n_full} "
        f"features={s.X_tr.shape[1]} device={args['device']}")
    try:
        rc, secs = _run_script(name="tabpfn", script=TABPFN_SCRIPT, args=args,
                               py=py, work=work,
                               timeout=max(budget_sec, 600) * 2, env=env)
    finally:
        if lease is not None:
            lease.release()
    return _finish(task, s, work, "tabpfn", rc, secs,
                   {"n_train_used": int(n_used), "n_train_full": int(n_full),
                    "weights": args["model_path"] or "package_default",
                    "subsampled": bool(idx is not None),
                    "device": args["device"], "python": py})


__all__ = ["run_flaml", "run_tabpfn", "resolve_refs_python",
           "TABPFN_MAX_ROWS"]
