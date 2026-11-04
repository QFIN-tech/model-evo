"""官方 hurdle 容器(v2.17.0,run4 regression.2 #15)。

零膨胀非负目标(动支金额 87% 为 0)的两段式回归:

    E[y | X] = P(y > threshold | X) · E[y | y > threshold, X]
    predict(X) = p · max(μ, 0)

背景:run4 每个 hurdle trial 手写 `HurdleXGB`(不是 sklearn 估计器 → 交付
frozen_rounds 的 sklearn.clone 失败、回落 rerun_recipe 白跑 10.4 min);
`--action hpo --from-parent <hurdle>` 曾静默把 hurdle 换成单模型。本模块给出
满足 sklearn 协议的官方容器:

- ``HurdleRegressor(clf=..., reg=...)``:两段各自的估计器原型(xgboost 优先,
  任何 sklearn 协议估计器均可);缺省时按 ``clf_params`` / ``reg_params``
  构造 XGBClassifier / XGBRegressor;
- ``get_params`` / ``set_params`` 支持 ``clf__<键>`` / ``reg__<键>`` 分段键;
  无前缀的非容器键 = 两段共用(与 v2.16.2 hurdle HPO 超参段同一语义),
  ``hurdle_lr`` / ``hurdle_cap`` 是 learning_rate / n_estimators 的别名;
- ``fit(X, y, eval_set=[(X_es, y_es)])``:两段各自早停 —— 分类段用
  1[y_es > threshold],回归段用 eval 面的正例行;无 eval_set 时自动关闭
  该段的 early_stopping_rounds(冻结轮数重训即此路径);
- **不暴露 predict_proba**(回归容器;交付/打分按任务类型一律 predict),
  诊断用 ``positive_prob`` / ``conditional_mean``;
- 可 ``sklearn.clone``(构造参数原样保存,拟合产物挂在 ``clf_`` / ``reg_``);
  可 joblib 持久化并在新进程加载(类定义在 runtime 包内,不依赖 __main__);
- 两段拟合经 ``runtime.submodel_cache.fit_or_load``:只改一个子模型的
  trial,另一个子模型在同输入、同参数、同 seed 下直接命中缓存
  (``cache_dir`` = RUN_DIR;``MODELEVO_SUBMODEL_CACHE=0`` 关闭);
  命中/未命中摘要见 ``submodel_cache_summary()``。
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

SEGMENTS = ("clf", "reg")
#: 进度日志回调(模块级,不进模型 pickle;trial 模板设为 step,使
#: "[step] hurdle clf fit start/done … (Ts)" 行与 v2.16.2 成本外推同口径)
_PROGRESS = None


def set_progress_logger(fn) -> None:
    global _PROGRESS
    _PROGRESS = fn


def _log(msg: str) -> None:
    if callable(_PROGRESS):
        try:
            _PROGRESS(msg)
        except Exception:  # noqa: BLE001
            pass


#: 与 v2.16.2 HURDLE_HP_SEGMENT 同口径的别名
PARAM_ALIASES = {"hurdle_lr": "learning_rate", "hurdle_cap": "n_estimators"}


def _norm_key(key: str) -> str:
    k = str(key)
    for p in ("clf__", "reg__"):
        if k.startswith(p):
            return p + PARAM_ALIASES.get(k[len(p):], k[len(p):])
    return PARAM_ALIASES.get(k, k)


def _take(X, mask: np.ndarray):
    idx = np.flatnonzero(mask)
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return np.asarray(X)[idx]


def _align_cpu(est) -> None:
    """xgboost GPU 训练的 booster 打分前切回 CPU(与模板 align_predict_device 同口径)。"""
    try:
        if not type(est).__module__.startswith("xgboost"):
            return
        import json as _json

        booster = est.get_booster()
        cfg = _json.loads(booster.save_config())
        dev = str(((cfg.get("learner") or {}).get("generic_param") or {})
                  .get("device") or "")
        if "cuda" in dev.lower() or "gpu" in dev.lower():
            booster.set_param({"device": "cpu"})
    except Exception:  # noqa: BLE001 优化项
        pass


def _fit_accepts(est, name: str) -> bool:
    try:
        sig = inspect.signature(est.fit)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values())


class HurdleRegressor(BaseEstimator, RegressorMixin):
    """两段式 hurdle 回归容器:predict = P(y>threshold)·max(E[y|y>threshold], 0)。

    参数
    ----
    clf, reg : 估计器原型或 None。None 时按 clf_params / reg_params 构造
        xgboost.XGBClassifier / XGBRegressor(tree_method="hist")。
    clf_params, reg_params : dict 或 None,仅在对应原型为 None 时生效。
    threshold : 正例阈值(y > threshold 为正例),默认 0。
    submodel_cache : 是否经子模型缓存拟合(True 时需 cache_dir 或环境变量
        MODELEVO_RUN_DIR 给出 RUN_DIR;MODELEVO_SUBMODEL_CACHE=0 全局关闭)。
    cache_dir : RUN_DIR(缓存落 RUN_DIR/.submodel-cache/)。
    """

    def __init__(self, clf=None, reg=None, *, clf_params=None, reg_params=None,
                 threshold=0.0, submodel_cache=True, cache_dir=None):
        self.clf = clf
        self.reg = reg
        self.clf_params = clf_params
        self.reg_params = reg_params
        self.threshold = threshold
        self.submodel_cache = submodel_cache
        self.cache_dir = cache_dir

    # ------------------------------------------------------------ sklearn 协议
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        try:
            from sklearn.utils import RegressorTags

            tags.estimator_type = "regressor"
            if getattr(tags, "regressor_tags", None) is None:
                tags.regressor_tags = RegressorTags()
        except Exception:  # noqa: BLE001  旧版 sklearn
            pass
        return tags

    _estimator_type = "regressor"

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "clf_") and hasattr(self, "reg_")

    def _own_params(self) -> list[str]:
        return list(super().get_params(deep=False).keys())

    def _default_segment(self, seg: str):
        params = dict((self.clf_params if seg == "clf" else self.reg_params)
                      or {})
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover
            raise ImportError("HurdleRegressor 未给 clf/reg 原型时需要 xgboost"
                              ) from exc
        params.setdefault("tree_method", "hist")
        if seg == "clf":
            params.setdefault("eval_metric", "logloss")
            return xgb.XGBClassifier(**params)
        return xgb.XGBRegressor(**params)

    def segment_prototype(self, seg: str):
        """段原型(未拟合)的副本:显式原型 clone;否则按 *_params 构造。"""
        proto = self.clf if seg == "clf" else self.reg
        return clone(proto) if proto is not None else self._default_segment(seg)

    def get_params(self, deep: bool = True) -> dict:
        out = super().get_params(deep=False)
        if not deep:
            return out
        for seg in SEGMENTS:
            proto = getattr(self, seg)
            if proto is not None and hasattr(proto, "get_params"):
                for k, v in proto.get_params(deep=True).items():
                    out[f"{seg}__{k}"] = v
            else:
                for k, v in (getattr(self, f"{seg}_params") or {}).items():
                    out[f"{seg}__{k}"] = v
        return out

    def _segment_keys(self, seg: str) -> set:
        proto = getattr(self, seg)
        if proto is None:
            proto = self._default_segment(seg)
        try:
            return set(proto.get_params(deep=True))
        except Exception:  # noqa: BLE001
            return set()

    def _set_segment(self, seg: str, updates: dict) -> None:
        if not updates:
            return
        proto = getattr(self, seg)
        if proto is not None and hasattr(proto, "set_params"):
            proto.set_params(**updates)
            return
        known = self._segment_keys(seg)
        unknown = sorted(k for k in updates if known and k not in known)
        if unknown:
            raise ValueError(f"hurdle {seg} 段未知超参键 {unknown}")
        attr = f"{seg}_params"
        setattr(self, attr, {**(getattr(self, attr) or {}), **updates})

    def set_params(self, **params):
        """容器键直接设置;``clf__k`` / ``reg__k`` 作用于对应段;其余无前缀键
        两段共用(两段都不认识的键报错,不静默空转)。分段键优先于共用键。"""
        if not params:
            return self
        own = set(self._own_params())
        own_upd, shared = {}, {}
        seg_upd: dict[str, dict] = {s: {} for s in SEGMENTS}
        for k, v in params.items():
            if k in own:
                own_upd[k] = v
                continue
            nk = _norm_key(k)
            for s in SEGMENTS:
                if nk.startswith(s + "__"):
                    seg_upd[s][nk[len(s) + 2:]] = v
                    break
            else:
                shared[nk] = v
        if own_upd:
            super().set_params(**own_upd)
        if shared:
            keys = {s: self._segment_keys(s) for s in SEGMENTS}
            for k, v in shared.items():
                hit = [s for s in SEGMENTS if k in keys[s]]
                if not hit:
                    raise ValueError(
                        f"HurdleRegressor 未知超参键 {k!r}(容器参数 "
                        f"{sorted(own)};分段键用 clf__/reg__ 前缀)")
                for s in hit:
                    seg_upd[s].setdefault(k, v)   # 分段键优先
        for s in SEGMENTS:
            self._set_segment(s, seg_upd[s])
        return self

    # ------------------------------------------------------------ 拟合
    def _fit_segment(self, seg: str, proto, X, y, eval_set, fit_extra: dict):
        from . import submodel_cache as _sc

        kw: dict[str, Any] = {}
        if eval_set and _fit_accepts(proto, "eval_set"):
            es = eval_set
        else:
            es = None
            try:
                if proto.get_params().get("early_stopping_rounds") is not None:
                    proto.set_params(early_stopping_rounds=None)
            except Exception:  # noqa: BLE001
                pass
        if type(proto).__module__.startswith("xgboost") and \
                _fit_accepts(proto, "verbose"):
            kw["verbose"] = False
        for k, v in (fit_extra or {}).items():
            if v is not None and _fit_accepts(proto, k):
                kw[k] = v
        _log(f"hurdle {seg} fit start: rows={len(y)} "
             f"es_rows={len(es[0][1]) if es else 0}")
        if self.submodel_cache:
            est, info = _sc.fit_or_load(
                seg, proto, X, y, eval_set=es, fit_kwargs=kw,
                run_dir=self.cache_dir,
                extra_key={"container": "HurdleRegressor",
                           "threshold": float(self.threshold)})
        else:
            est = clone(proto)
            if es is not None:
                kw["eval_set"] = es
            import time as _t

            t0 = _t.time()
            est.fit(X, y, **kw)
            info = {"name": seg, "status": "disabled",
                    "reason": "submodel_cache=False",
                    "fit_sec": round(_t.time() - t0, 3)}
        info["early_stopping"] = es is not None
        try:
            _bi = int(est.best_iteration)
        except Exception:  # noqa: BLE001
            _bi = None
        _log(f"hurdle {seg} fit done: best_iter={_bi} "
             f"cache={info.get('status')}")
        return est, info

    def fit(self, X, y, eval_set=None, *, sample_weight=None,
            clf_eval_set=None, reg_eval_set=None):
        thr = float(self.threshold)
        y_arr = np.asarray(y, dtype="float64")
        pos = y_arr > thr
        n_pos = int(pos.sum())
        if n_pos == 0:
            raise ValueError("HurdleRegressor: 训练集无正例(y > threshold),"
                             "条件回归段无法拟合")
        if n_pos == len(y_arr):
            raise ValueError("HurdleRegressor: 训练集全为正例,分类段只有一类")
        if clf_eval_set is None and eval_set:
            clf_eval_set = [(Xe, (np.asarray(ye, dtype="float64") > thr)
                             .astype("int8")) for Xe, ye in eval_set]
        if reg_eval_set is None and eval_set:
            reg_eval_set = []
            for Xe, ye in eval_set:
                ye = np.asarray(ye, dtype="float64")
                pe = ye > thr
                if pe.any():
                    reg_eval_set.append((_take(Xe, pe), ye[pe]))
            reg_eval_set = reg_eval_set or None
        sw = None if sample_weight is None else np.asarray(sample_weight)
        clf_proto = self.segment_prototype("clf")
        reg_proto = self.segment_prototype("reg")
        self.clf_, info_c = self._fit_segment(
            "clf", clf_proto, X, pos.astype("int8"), clf_eval_set,
            {"sample_weight": sw})
        self.reg_, info_r = self._fit_segment(
            "reg", reg_proto, _take(X, pos), y_arr[pos], reg_eval_set,
            {"sample_weight": None if sw is None else sw[pos]})
        _align_cpu(self.clf_)
        _align_cpu(self.reg_)
        try:
            classes = list(getattr(self.clf_, "classes_", [0, 1]))
            self.pos_index_ = classes.index(1)
        except (ValueError, TypeError):
            self.pos_index_ = 1
        self.n_features_in_ = int(np.shape(X)[1]) if np.ndim(X) == 2 else None
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns],
                                                dtype=object)
        self.train_pos_rate_ = float(pos.mean())
        self.n_reg_fit_rows_ = n_pos
        self.submodel_cache_info_ = {"clf": info_c, "reg": info_r}
        return self

    # ------------------------------------------------------------ 打分
    def positive_prob(self, X) -> np.ndarray:
        """P(y > threshold | X)(诊断用;容器本身不暴露 predict_proba)。"""
        proba = self.clf_.predict_proba(X)
        return np.asarray(proba)[:, getattr(self, "pos_index_", 1)]

    def conditional_mean(self, X) -> np.ndarray:
        """E[y | y > threshold, X](条件回归段原始输出)。"""
        return np.asarray(self.reg_.predict(X), dtype="float64")

    def predict(self, X) -> np.ndarray:
        p = self.positive_prob(X).astype("float64")
        mu = np.maximum(self.conditional_mean(X), 0.0)
        return p * mu

    # ------------------------------------------------------------ 遥测
    @property
    def feature_importances_(self) -> np.ndarray:
        a = np.asarray(self.clf_.feature_importances_, dtype="float64")
        b = np.asarray(self.reg_.feature_importances_, dtype="float64")
        a = a / a.sum() if a.sum() > 0 else a
        b = b / b.sum() if b.sum() > 0 else b
        return 0.5 * (a + b)

    @property
    def best_iteration(self):
        """两段最优轮次(0-based)取较大者;任一段不可得返回 None。"""
        vals = []
        for est in (getattr(self, "clf_", None), getattr(self, "reg_", None)):
            try:
                vals.append(int(est.best_iteration))
            except Exception:  # noqa: BLE001
                return None
        return max(vals) if vals else None

    def segment_telemetry(self) -> dict:
        """两段遥测:最优轮次(1-based)、实训轮数、cap、是否撞 cap。"""
        out = {}
        for seg in SEGMENTS:
            est = getattr(self, f"{seg}_", None)
            if est is None:
                continue
            bi = rounds = cap = None
            try:
                bi = int(est.best_iteration) + 1
            except Exception:  # noqa: BLE001
                pass
            try:
                ev = est.evals_result()
                rounds = int(len(next(iter(next(iter(ev.values())).values()))))
            except Exception:  # noqa: BLE001
                pass
            try:
                cap = int(est.get_params().get("n_estimators") or 0) or None
            except Exception:  # noqa: BLE001
                pass
            out[f"{seg}_best_iteration"] = bi
            out[f"{seg}_rounds"] = rounds
            out[f"{seg}_cap"] = cap
            out[f"{seg}_cap_bound"] = bool(rounds and cap and rounds >= cap)
        return out

    def submodel_cache_summary(self) -> dict:
        from .submodel_cache import summarize

        return summarize(getattr(self, "submodel_cache_info_", {}) or {})


def is_hurdle_container(model) -> bool:
    return isinstance(model, HurdleRegressor)
