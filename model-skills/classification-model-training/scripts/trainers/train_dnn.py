# -*- coding: utf-8 -*-
"""model-training DNN 训练: 用 engines._dnn 引擎的 MLP + 三段式数据。

与 xgb 路径(tune_train)对齐, 提供算法无关的 predictor 协议产物
(predict_proba(df[features]) -> np.ndarray), 供 eval_data 复用。
本封装把 缺失填充(DNNImputer) + 标准化(StandardScaler) + nn.Module 打包进
DnnPredictor, 使下游对比与 xgb 完全同构。

数据切分由 run_build 按 model.split 内部完成, 本脚本直接读三档 parquet,
test 当 val 做 early stopping。

模块隔离: dnn 引擎的代码是自洽超集(_core 含 MetricsReport/DriftProbe,
_dataset 含 DNNImputer, entry 内联 BinaryMLP/DNNTrainer),
import 时用 engines._dnn._core / engines._dnn._dataset / engines._dnn._model 即可。
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from engines._dnn._model import BinaryMLP, DNNTrainer  # noqa: E402
from engines._dnn._dataset import DNNImputer  # noqa: E402
from engines._dnn._core import MetricsReport  # noqa: E402


# DNN 默认超参: 对齐 dnn 引擎 argparse 默认值(hidden 128->64->32, dropout 0.3,
# lr 0.001, batch 512, epochs 100 由 early stopping 兜底, pos_weight 自动按负正比)。
# device: "auto" = cuda 可用则 cuda, 否则 cpu; 也可显式 "cuda" / "cpu"。
# random_state: 与 xgb/lr 对齐(42), 保证 MLP 初始化/Dropout/DataLoader shuffle 可复现。
DNN_PARAMS = {
    "hidden_dims": [128, 64, 32],
    "dropout": 0.3,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "batch_size": 512,
    "epochs": 100,
    "patience": 10,
    "pos_weight": "auto",
    "device": "auto",
    "random_state": 42,
}


def _set_seed(seed: int) -> None:
    """统一设 Python/numpy/torch 随机种子, 使 MLP 初始化/Dropout/DataLoader shuffle 可复现。

    cudnn.deterministic=True 关闭非确定性算法; benchmark=False 关闭自动 kernel 选择,
    对 MLP 几乎无性能损失。完全跨硬件可复现不保证(cuda 版本/GPU 型号差异), 同机复现足够。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(spec: str) -> torch.device:
    """把 device 字符串解析成 torch.device, auto / cuda / cpu 三选一。

    - "auto": cuda 可用则 cuda, 否则 cpu
    - "cuda": 强制 cuda, 不可用时回退 cpu 并警告
    - "cpu": 强制 cpu
    """
    spec = (spec or "auto").lower()
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            print("[dnn] 警告: device='cuda' 但 CUDA 不可用, 回退 cpu")
            return torch.device("cpu")
        return torch.device("cuda")
    # auto 或其他值
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class DnnPredictor:
    """算法无关预测器: 把 缺失填充 + 标准化 + MLP 打包, 暴露 predict_proba(df)。

    与 XgbFitter 同构: 下游 eval_data 只依赖 predict_proba(df[features]) 与
    可选 get_feature_importance()。本对象可直接 pickle(nn.Module/scaler/imputer
    均可序列化), 故模型落盘为 .pkl。

    Attributes:
        imputer: 已 fit 的 DNNImputer(持有中位数与指示列集合)
        scaler: 已 fit 的 StandardScaler(基于填充后的扩展特征)
        model: 已训练的 BinaryMLP(nn.Module), 推理时置 eval 模式
        features: 入模原始特征(列名+顺序, 与训练一致)
        device: 训练时所用 device, 推理时 model 与 input tensor 都 to(device)
    """

    imputer: DNNImputer
    scaler: StandardScaler
    model: object
    features: List[str] = field(default_factory=list)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """对原始特征 DataFrame 做 填充->标准化->前向, 返回正类概率。

        Args:
            df: 含 self.features 列的 DataFrame(通常为 oot_df[features])

        Returns:
            正类概率 np.ndarray, shape=(n,)
        """
        X_filled, _ = self.imputer.transform(df[self.features])
        X_scaled = self.scaler.transform(X_filled.values)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(
                torch.FloatTensor(X_scaled).to(self.device)
            )
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def get_feature_importance(self) -> dict:
        """DNN 无原生 gain 重要性, 返回空 dict(对比报告该维度自动留空)。"""
        return {}


def train_dnn_model(
    train_path: str, test_path: str, oot_path: str,
    target: str, features: List[str], params: dict,
) -> tuple:
    """用 MLP 训练 + 评估 Train/Val/OOT(三段式, test 当 val 做 early stopping)。

    Args:
        train_path: 训练段 parquet(已清洗为数值列)
        test_path: 测试段 parquet(已清洗),作为 early-stopping eval_set
        oot_path: OOT 段 parquet(已清洗)
        target: 标签列
        features: 入模特征
        params: DNN 超参字典(见 DNN_PARAMS)

    Returns:
        (predictor, metrics, info)
        - predictor: DnnPredictor
        - metrics: {train,val,oot: {auc,ks}}
        - info: DNNTrainer.train 返回的训练信息(best_epoch 等)
    """
    # 设随机种子: 与 xgb/lr 对齐(默认 42), 保证 MLP 初始化/Dropout/DataLoader shuffle 可复现。
    # 必须在模型构造 + DataLoader 构造之前调用, 否则不生效。
    seed = int(params.get("random_state", 42))
    _set_seed(seed)
    print(f"[dnn] random_state={seed}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(test_path)
    oot_df = pd.read_parquet(oot_path)
    print(f"[dnn] 切分: train={len(train_df)} val={len(val_df)} oot={len(oot_df)}")

    # 缺失填充: 中位数 + 中等缺失率指示列(仅用训练段 fit, 防泄漏)
    imputer = DNNImputer(features=features)
    imputer.fit(train_df[features])
    X_train_f, model_features = imputer.transform(train_df[features])
    X_val_f, _ = imputer.transform(val_df[features])

    # 标准化: 基于填充后的扩展特征, 仅用训练段 fit
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_f.values)
    X_val_s = scaler.transform(X_val_f.values)

    y_train = train_df[target].to_numpy().astype(np.float32)
    y_val = val_df[target].to_numpy().astype(np.float32)

    # pos_weight: 自动按 负/正 比(类不平衡), 或用户显式指定
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    if params.get("pos_weight", "auto") == "auto":
        pos_weight = (neg / pos) if pos > 0 else 1.0
    else:
        pos_weight = float(params["pos_weight"])
    print(f"[dnn] pos_weight={pos_weight:.4f} (neg={neg:.0f} pos={pos:.0f})")

    bs = int(params["batch_size"])
    device = _resolve_device(str(params.get("device", "auto")))
    print(f"[dnn] device={device} (cuda available={torch.cuda.is_available()})")
    # drop_last=True 防 BatchNorm 遇 size-1 末批报错; val 在 eval 模式无此问题
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(y_train)),
        batch_size=bs, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(y_val)),
        batch_size=bs, shuffle=False,
    )

    model = BinaryMLP(
        input_dim=X_train_s.shape[1],
        hidden_dims=list(params["hidden_dims"]),
        dropout=float(params["dropout"]),
    )
    trainer = DNNTrainer(
        model, learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        pos_weight=pos_weight, device=str(device),
    )
    info = trainer.train(
        train_loader, val_loader,
        epochs=int(params["epochs"]), patience=int(params["patience"]),
    )
    print(f"[dnn] best_epoch={info.get('best_epoch')} "
          f"best_val_auc={info.get('best_val_auc')} early_stopped={info.get('early_stopped')}")

    predictor = DnnPredictor(
        imputer=imputer, scaler=scaler, model=model,
        features=list(features), device=device,
    )

    reporter = MetricsReport()
    splits = {
        "train": (train_df[target].to_numpy(), predictor.predict_proba(train_df)),
        "val": (val_df[target].to_numpy(), predictor.predict_proba(val_df)),
        "oot": (oot_df[target].to_numpy(), predictor.predict_proba(oot_df)),
    }
    m = reporter.evaluate_splits(splits)
    for k in ("train", "val", "oot"):
        print(f"[dnn] {k}: AUC={m[k].auc:.4f} KS={m[k].ks:.4f}")
    return predictor, m, info
