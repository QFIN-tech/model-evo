# -*- coding: utf-8 -*-
"""DNN 模型定义与训练器。

从 `engines/_dnn/entry.py` 提取, 供 `trainers/train_dnn.py` 复用。
评估/报告逻辑已委托 `classification-model-evaluation`, 此处只保留训练核心。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from engines._dnn._core import _ks_via_roc  # noqa: E402

logger = logging.getLogger(__name__)


class BinaryMLP(nn.Module):
    """多层全连接网络（二分类）。

    结构：Input → [Linear → BatchNorm → ReLU → Dropout] × N → Linear → Sigmoid
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """输出正类概率。"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class DNNTrainer:
    """DNN 训练器：封装训练循环、早停、学习率调度。"""

    def __init__(
        self,
        model: BinaryMLP,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        pos_weight: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5, min_lr=1e-6
        )
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        self.history: List[Dict[str, float]] = []

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 10,
    ) -> Dict[str, Any]:
        """训练模型，返回训练信息。"""
        best_val_auc = 0.0
        best_epoch = 0
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # ── Training ──
            self.model.train()
            train_loss = 0.0
            train_preds, train_labels = [], []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * len(y_batch)
                train_preds.append(torch.sigmoid(logits).detach().cpu().numpy())
                train_labels.append(y_batch.cpu().numpy())

            train_loss /= len(train_loader.dataset)
            train_preds = np.concatenate(train_preds)
            train_labels = np.concatenate(train_labels)
            train_auc = roc_auc_score(train_labels, train_preds)

            # ── Validation ──
            self.model.eval()
            val_preds, val_labels = [], []
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    logits = self.model(X_batch)
                    loss = self.criterion(logits, y_batch)
                    val_loss += loss.item() * len(y_batch)
                    val_preds.append(torch.sigmoid(logits).cpu().numpy())
                    val_labels.append(y_batch.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_preds = np.concatenate(val_preds)
            val_labels = np.concatenate(val_labels)
            val_auc = roc_auc_score(val_labels, val_preds)
            val_ks = float(_ks_via_roc(val_labels, val_preds))

            # 学习率调度
            self.scheduler.step(val_auc)

            # 训练集 KS
            train_ks = float(_ks_via_roc(train_labels, train_preds))

            self.history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "train_auc": round(train_auc, 6),
                "val_auc": round(val_auc, 6),
                "train_ks": round(train_ks, 6),
                "val_ks": round(val_ks, 6),
            })

            # 早停
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                lr_current = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss={train_loss:.4f} AUC={train_auc:.4f} KS={train_ks:.4f} | "
                    f"Val Loss={val_loss:.4f} AUC={val_auc:.4f} KS={val_ks:.4f} | "
                    f"LR={lr_current:.6f}"
                )

            if patience_counter >= patience:
                logger.info(f"早停触发 @ Epoch {epoch}（最佳 Epoch {best_epoch}, Val AUC={best_val_auc:.4f}）")
                break

        # 恢复最佳权重
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "best_epoch": best_epoch,
            "best_val_auc": round(best_val_auc, 6),
            "total_epochs": epoch,
            "early_stopped": patience_counter >= patience,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测正类概率。"""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(X_tensor)).cpu().numpy()
        return probs
