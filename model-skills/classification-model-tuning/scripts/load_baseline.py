# -*- coding: utf-8 -*-
"""从 model-training 的 baseline run_dir 读取必要信息,封装为 BaselineSnapshot。

config.json 由 model-training 的 run_layout.write_config_snapshot 落盘,
包含完整 cfg / runtime metrics / train_info / n_features。
model/_manifest.json 含 used_params 与 algo。

train_info 字段随 algo 不同(xgb: best_iteration; dnn: best_epoch/total_epochs/
early_stopped/best_val_auc; lr: n_iter/converged),整体透传给下游 diagnose / 诊断展示。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BaselineSnapshot:
    """对 model-training run 的只读视图,供调优诊断/重训复用。"""

    run_dir: Path
    run_name: str
    algo: str
    label: str
    cfg: Dict[str, Any]                # baseline 落盘的完整 yaml cfg(已剔除 _config_dir)
    data_dir: str                      # baseline 训练时的数据目录(绝对, 含 train/test/oot.parquet)
    train_path: str                    # train.parquet 绝对路径
    test_path: str                     # test.parquet 绝对路径(当 val 段)
    oot_path: str                      # oot.parquet 绝对路径
    features: List[str]                # 入模特征(从 cfg.model.features 取)
    used_params: Dict[str, Any]        # baseline 训练超参(从 model/_manifest.json 取)
    metrics: Dict[str, Dict[str, float]]  # {train/val/oot: {auc, ks}}
    train_info: Dict[str, Any] = field(default_factory=dict)
    best_iteration: Optional[int] = None  # algo-aware: xgb=best_iteration / dnn=best_epoch / lr=n_iter, 供 diagnose() 用
    n_features: Optional[int] = None
    new_psi: Optional[float] = None    # 训练→OOT psi (从 evaluation/_manifest.json 取)
    base_psi: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def _resolve_best_iteration(algo: str, train_info: Dict[str, Any]) -> Optional[int]:
    """从 train_info 提取 diagnose() 需要的"最佳轮次"标量。

    algo-aware:
    - xgb: best_iteration
    - dnn: best_epoch
    - lr : n_iter
    缺失返回 None。
    """
    if not train_info:
        return None
    if algo == "dnn":
        val = train_info.get("best_epoch")
    elif algo == "lr":
        val = train_info.get("n_iter")
    else:
        val = train_info.get("best_iteration")
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    """读 json,文件不存在抛 FileNotFoundError。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load(run_dir: str) -> BaselineSnapshot:
    """从 baseline run 目录读取快照。

    Args:
        run_dir: model-training 产出的 run 目录(含 config.json / model/ / evaluation/)

    Returns:
        BaselineSnapshot

    Raises:
        FileNotFoundError: 必要文件缺失
        ValueError: 关键字段缺失(如 model.features / runtime.metrics)
    """
    rd = Path(run_dir).resolve()
    if not rd.is_dir():
        raise FileNotFoundError(f"baseline run_dir 不是目录: {rd}")

    cfg_snap = _read_json(rd / "config.json")
    model_manifest = _read_json(rd / "model" / "_manifest.json")

    cfg = cfg_snap.get("config") or {}
    model_cfg = cfg.get("model") or {}
    runtime = cfg_snap.get("runtime") or {}

    features = list(model_cfg.get("features") or [])
    if not features:
        raise ValueError(f"baseline config.json 缺少 config.model.features: {rd}")

    metrics = runtime.get("metrics") or {}
    if not metrics or "train" not in metrics or "oot" not in metrics:
        raise ValueError(
            f"baseline config.json 缺少 runtime.metrics(train/val/oot): {rd}"
        )

    used_params = model_manifest.get("used_params") or {}
    algo = (cfg_snap.get("algo") or model_cfg.get("algo") or "xgb").lower()

    # train_info: 透传 baseline 的训练细节字典 (xgb: best_iteration; dnn: best_epoch/
    # total_epochs/early_stopped/best_val_auc; lr: n_iter/converged)。
    # runtime 里的 best_iteration/best_epoch/n_iter 也 merge 进来 (model-training 落盘时
    # train_info 可能从 model_manifest 或 runtime 读取, 两处都支持)。
    train_info = dict(model_manifest.get("train_info") or {})
    if not train_info:
        # model_manifest 无 train_info 时, 从 runtime 提取已知字段构建
        for k in ("best_iteration", "best_epoch", "total_epochs",
                  "early_stopped", "best_val_auc", "n_iter", "converged"):
            if k in runtime:
                train_info[k] = runtime[k]

    best_iteration = _resolve_best_iteration(algo, train_info)

    # evaluation manifest 是可选(若 baseline 跑完整流程则一定有)
    new_psi = base_psi = None
    eval_manifest_path = rd / "evaluation" / "_manifest.json"
    if eval_manifest_path.exists():
        em = _read_json(eval_manifest_path)
        new_psi = em.get("new_psi")
        base_psi = em.get("base_psi")

    data_path = (cfg_snap.get("input") or {}).get("data_path") or ""
    # baseline 可能用单 data_path 或三档分离路径, 三档优先
    input_meta = cfg_snap.get("input") or {}
    data_dir = input_meta.get("data_dir") or ""
    train_path = input_meta.get("train_path") or ""
    test_path = input_meta.get("test_path") or ""
    oot_path = input_meta.get("oot_path") or ""
    if not train_path and data_path:
        # 三档路径缺失时, 三档均回退到 data_path
        train_path = data_path
        test_path = data_path
        oot_path = data_path

    return BaselineSnapshot(
        run_dir=rd,
        run_name=cfg_snap.get("run_name") or rd.name,
        algo=algo,
        label=cfg_snap.get("label") or cfg_snap.get("version") or "",
        cfg=cfg,
        data_dir=data_dir,
        train_path=train_path,
        test_path=test_path,
        oot_path=oot_path,
        features=features,
        used_params=used_params,
        metrics=metrics,
        train_info=train_info,
        best_iteration=best_iteration,
        n_features=runtime.get("n_features"),
        new_psi=new_psi,
        base_psi=base_psi,
        extras={"timestamp": cfg_snap.get("timestamp"), "suffix": cfg_snap.get("suffix", "")},
    )
