# -*- coding: utf-8 -*-
"""XGBoost 训练侧封装。

把 sklearn 风格的 XGBClassifier 包成统一 fitter 接口，供 entry 调度。
承担: 入参合并 / 早停与 eval_set 配对 / 自动正负样本权重 / 欠采样 / 模型落盘。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class XgbFitter:
    """XGBoost fitter。

    职责：把训练-评估-落盘三段流程收口到一处，避免 entry 内散落 xgb 调用。
    支持自动 early stopping、自动 scale_pos_weight、K 折交叉验证辅助。
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 6,
        "learning_rate": 0.02,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """构造 fitter。

        Args:
            params: XGBoost 入参，与 DEFAULT_PARAMS 合并（用户传入优先）。
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] | None = None
        self.feature_importance_: dict[str, float] | None = None
        self.best_iteration_: int | None = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        X_eval: pd.DataFrame | None = None,
        y_eval: np.ndarray | pd.Series | None = None,
        early_stopping_rounds: int = 30,
        auto_scale_pos_weight: bool = True,
        sample_strategy: str = "auto_weight",
        sample_weight: np.ndarray | None = None,
    ) -> xgb.XGBClassifier:
        """执行训练。

        Args:
            X_train / y_train: 训练样本。
            X_eval / y_eval: 验证集，**应始终是 val**（早停用）。
                约定 OOT 只参与最终评估，禁止作 X_eval 传入。传 None 则关闭早停
                （适用于 meta-learner 等无验证集场景）。
            early_stopping_rounds: 早停耐心轮数。
            auto_scale_pos_weight: 是否按正负样本比自动设置 scale_pos_weight。
            sample_strategy:
                "auto_weight"  — 自动计算 scale_pos_weight（默认）。
                "undersample"  — 对负样本欠采样到 10:1。
                "none"         — 不做不均衡处理。
            sample_weight: 外部样本权重（如时序衰减），与不均衡策略相乘后生效。

        Returns:
            训练好的 XGBClassifier。
        """
        if X_eval is not None and y_eval is not None:
            if len(X_eval) == 0 or len(y_eval) == 0:
                raise ValueError(
                    "X_eval / y_eval 已传入但为空；如无验证集请显式传 None 关闭早停。"
                )
        params = self.params.copy()

        y_train_arr = np.array(y_train)
        X_train_fit = X_train
        y_train_fit = y_train_arr
        fit_sample_weight = sample_weight.copy() if sample_weight is not None else None

        if sample_strategy == "auto_weight":
            if auto_scale_pos_weight and "scale_pos_weight" not in params:
                neg_count = int((y_train_arr == 0).sum())
                pos_count = int((y_train_arr == 1).sum())
                if pos_count > 0:
                    params["scale_pos_weight"] = neg_count / pos_count

        elif sample_strategy == "undersample":
            max_ratio = 10
            pos_idx = np.where(y_train_arr == 1)[0]
            neg_idx = np.where(y_train_arr == 0)[0]
            pos_count = len(pos_idx)
            max_neg = pos_count * max_ratio
            if len(neg_idx) > max_neg:
                rng = np.random.RandomState(42)
                neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
            keep_idx = np.concatenate([pos_idx, neg_idx])
            keep_idx = np.sort(keep_idx)
            X_train_fit = X_train.iloc[keep_idx] if hasattr(X_train, "iloc") else X_train[keep_idx]
            y_train_fit = y_train_arr[keep_idx]
            if fit_sample_weight is not None:
                fit_sample_weight = fit_sample_weight[keep_idx]
            logger.info(f"欠采样后训练集: {len(keep_idx)} 条（正:{pos_count} 负:{len(keep_idx)-pos_count}）")

        self.feature_names = list(X_train.columns) if hasattr(X_train, "columns") else None

        n_estimators = params.pop("n_estimators", 300)
        es = early_stopping_rounds if (X_eval is not None and y_eval is not None) else None
        params.pop("early_stopping_rounds", None)

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=es,
            **params,
        )

        fit_params: dict[str, Any] = {"verbose": False}
        if X_eval is not None and y_eval is not None:
            fit_params["eval_set"] = [(X_eval, y_eval)]
        if fit_sample_weight is not None:
            fit_params["sample_weight"] = fit_sample_weight

        self.model.fit(X_train_fit, y_train_fit, **fit_params)

        self.best_iteration_ = getattr(self.model, 'best_iteration', n_estimators)

        importance = self.model.feature_importances_
        if self.feature_names:
            self.feature_importance_ = dict(zip(self.feature_names, importance.tolist()))
        else:
            self.feature_importance_ = {f"f{i}": float(v) for i, v in enumerate(importance)}

        return self.model

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """返回正类概率。"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train()")
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self, sort: bool = True) -> dict[str, float]:
        """返回特征重要性映射。

        Args:
            sort: 是否按重要性降序排列。

        Returns:
            feature -> importance。
        """
        if self.feature_importance_ is None:
            raise ValueError("模型未训练，请先调用 train()")
        if sort:
            return dict(sorted(self.feature_importance_.items(), key=lambda x: x[1], reverse=True))
        return self.feature_importance_

    def save_model(self, model_path: str) -> str:
        """落盘模型 + 旁路 meta。

        Args:
            model_path: 模型文件路径（支持 .json 或 .ubj）。

        Returns:
            实际落盘路径。
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train()")

        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_path)

        meta_path = str(model_path).replace('.json', '_meta.json').replace('.ubj', '_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance_,
                'best_iteration': self.best_iteration_,
                'params': self.params,
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"模型已保存: {model_path}")
        return model_path

    @classmethod
    def load_model(cls, model_path: str) -> 'XgbFitter':
        """从文件恢复 fitter。

        Args:
            model_path: 模型文件路径。

        Returns:
            重建后的 XgbFitter 实例。
        """
        trainer = cls()
        trainer.model = xgb.XGBClassifier()
        trainer.model.load_model(model_path)

        meta_path = str(model_path).replace('.json', '_meta.json').replace('.ubj', '_meta.json')
        if Path(meta_path).exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                trainer.feature_names = meta.get('feature_names')
                trainer.feature_importance_ = meta.get('feature_importance')
                trainer.best_iteration_ = meta.get('best_iteration')
                trainer.params = meta.get('params', cls.DEFAULT_PARAMS)

        logger.info(f"模型已加载: {model_path}")
        return trainer

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        n_folds: int = 5,
        metrics_func: Callable[[np.ndarray, np.ndarray], dict[str, float]] | None = None,
    ) -> list[dict[str, Any]]:
        """K 折交叉验证辅助。

        Args:
            X: 特征矩阵。
            y: 标签。
            n_folds: 折数。
            metrics_func: 自定义评估器 func(y_true, y_prob) -> dict，缺省只算 AUC。

        Returns:
            每折评估结果列表。
        """
        from sklearn.metrics import roc_auc_score

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results: list[dict[str, Any]] = []

        y_arr = np.array(y)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_arr)):
            X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            X_val = X.iloc[val_idx] if hasattr(X, "iloc") else X[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            fitter = XgbFitter(self.params.copy())
            fitter.train(X_tr, y_tr, X_val, y_val)
            y_prob = fitter.predict_proba(X_val)

            if metrics_func:
                result = metrics_func(y_val, y_prob)
            else:
                result = {"auc": roc_auc_score(y_val, y_prob)}

            result["fold"] = fold_idx + 1
            fold_results.append(result)

        return fold_results

    def tune_params_optuna(self, *args, **kwargs):
        """不可用:调参走 _modelevo-shared/tuning_engine.TuningEngine。

        用法:
            from tuning_engine import TuningEngine
            engine = TuningEngine(X_train, y_train, X_eval, y_eval, base_params=..., metric=...)
            engine.run_batch(n_trials=30)
        """
        raise NotImplementedError(
            "tune_params_optuna 不可用, 请使用 _modelevo-shared/tuning_engine.TuningEngine.run_batch。"
        )

    def tune_params(self, *args, **kwargs):
        """不可用:随机搜索并入 TuningEngine 的 Optuna 主路径。"""
        raise NotImplementedError(
            "tune_params 不可用, 请使用 _modelevo-shared/tuning_engine.TuningEngine。"
        )

    def save(self, path: str | Path) -> None:
        """落盘模型 + 旁路 meta（与 save_model 等价的别名）。

        Args:
            path: 落盘路径（推荐 .json 或 .ubj）。
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train()")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(path))

        meta = {
            "params": self.params,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance_,
            "best_iteration": self.best_iteration_,
        }
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"模型已保存到 {path}")

    @classmethod
    def load(cls, path: str | Path) -> "XgbFitter":
        """从文件恢复 fitter（load_model 别名，使用 Booster 兜底加载）。

        Args:
            path: 模型文件路径。

        Returns:
            重建后的 XgbFitter 实例。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {path}")

        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}

        fitter = cls(params=meta.get("params"))

        booster = xgb.Booster()
        booster.load_model(str(path))

        fitter.model = xgb.XGBClassifier()
        fitter.model._Booster = booster
        fitter.model._le = None

        fitter.feature_names = meta.get("feature_names")
        fitter.feature_importance_ = meta.get("feature_importance")
        fitter.best_iteration_ = meta.get("best_iteration")

        logger.info(f"模型已从 {path} 加载")
        return fitter
