# -*- coding: utf-8 -*-
"""train-only 常数精调。

设计要点(与常见的 test 档搜常数做法相反, 本模块刻意修正两点):
1. 搜索目标只看 **train 档** 单特征方向修正 AUC(train 本来就允许拟合),
   test/OOT 全程不参与搜索, 保持干净(G4/G5 的判定不受污染);
2. 回填不再经过 LLM, 而是机械追加 `DEFAULT_PARAMS.update(...)` 到代码尾部,
   固化后的代码重过完整 G1~G6 门禁。

候选契约的可选扩展(不声明则完全不触发精调):
    PARAM_BOUNDS = {"center": (0.0, 4.0)}   # 待精调常数及上下界(宜宽)
    DEFAULT_PARAMS = {"center": 0.0}        # 初始值(= 手写值), 键与 PARAM_BOUNDS 一致
    def transform(df, params=None):
        p = dict(DEFAULT_PARAMS) if params is None else dict(params)
        ...
"""
from __future__ import annotations

import random
import zlib

import numpy as np
import pandas as pd

from . import candidate_exec, metrics

N_RANDOM = 48          # 随机探针数(种子由 cid 决定, 结果可复现)
REFINE_PER_PARAM = 7   # 坐标细化: 每参数在最优邻域取点数
MIN_GAIN = 1e-6        # 精调有效门槛(train AUC 增益)


def _norm_spec(spec) -> dict | None:
    """规整子进程返回的 param_spec; 无效(缺声明/键不一致/界非法)返回 None。"""
    if not isinstance(spec, dict):
        return None
    bounds, defaults = spec.get("bounds"), spec.get("defaults")
    if not isinstance(bounds, dict) or not bounds:
        return None
    if not isinstance(defaults, dict) or set(defaults) != set(bounds):
        return None
    out_b, out_d = {}, {}
    for k, v in bounds.items():
        try:
            lo, hi = float(v[0]), float(v[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not (lo < hi) or not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        out_b[str(k)] = (lo, hi)
    for k, v in defaults.items():
        try:
            d = float(v)
        except (TypeError, ValueError):
            return None
        out_d[str(k)] = min(max(d, out_b[k][0]), out_b[k][1])  # 越界裁剪
    return {"bounds": out_b, "defaults": out_d}


def _objective(y: pd.Series, series) -> float | None:
    """train 档单特征方向修正 AUC; 不可算返回 None。"""
    s = pd.Series(series, dtype=float)
    if len(s) != len(y):
        return None
    auc = metrics.compute_auc_direction_fixed(pd.Series(y).reset_index(drop=True), s.reset_index(drop=True))
    if auc is None or np.isnan(auc):
        return None
    return float(auc)


def _bake(code: str, params: dict) -> str:
    """把最优常数固化回代码尾部(transform 须按约定读取 DEFAULT_PARAMS)。"""
    items = ", ".join("%r: %r" % (k, params[k]) for k in sorted(params))
    return (
        code.rstrip("\n")
        + "\n\n# ==== tuned by evolution(train-only 常数精调, 勿手改) ====\nDEFAULT_PARAMS.update({%s})\n" % items
    )


def _search_once(code, df, y, param_list, best, best_auc, timeout_s):
    """批量试跑一组参数, 保留 train AUC 最优者(失败整批放弃, 不影响已有最优)。"""
    if not param_list:
        return best, best_auc
    run = candidate_exec.run_transform_batch(code, df, param_list, timeout_s=timeout_s)
    if not run["ok"]:
        return best, best_auc
    for p, s in zip(param_list, run["series_list"]):
        auc = _objective(y, s)
        if auc is not None and auc > best_auc:
            best, best_auc = dict(p), auc
    return best, best_auc


def tune_candidate_params(code: str, df_train: pd.DataFrame, y_train, cid: str,
                          timeout_s: float = 60.0, enabled: bool = True) -> dict:
    """尝试常数精调(仅 train 档)。

    Returns:
        {tuned, code, params?, default_auc?, best_auc?, n_evals?, skip_reason?}
        tuned=False 时 code 原样返回(无参数约定 / 无改进 / 被配置关闭 / 执行失败)。
    """
    if not enabled:
        return {"tuned": False, "code": code, "skip_reason": "disabled"}
    if "PARAM_BOUNDS" not in code or "DEFAULT_PARAMS" not in code:
        return {"tuned": False, "code": code, "skip_reason": "no param spec"}

    first = candidate_exec.run_transform_batch(code, df_train, [None], timeout_s=timeout_s)
    if not first["ok"]:
        return {"tuned": False, "code": code, "skip_reason": "批量执行失败: %s" % first.get("error")}
    spec = _norm_spec(first.get("param_spec"))
    if spec is None:
        return {"tuned": False, "code": code, "skip_reason": "参数约定无效(键不一致/界非法)"}
    if not first.get("accepts_params"):
        return {"tuned": False, "code": code, "skip_reason": "transform 不接受 params 入参"}
    default_auc = _objective(y_train, first["series_list"][0])
    if default_auc is None:
        return {"tuned": False, "code": code, "skip_reason": "默认参数下 train AUC 不可计算"}

    bounds, best, best_auc = spec["bounds"], dict(spec["defaults"]), default_auc
    rng = random.Random(zlib.crc32(str(cid).encode("utf-8")))
    samples = [{k: rng.uniform(lo, hi) for k, (lo, hi) in bounds.items()} for _ in range(N_RANDOM)]
    best, best_auc = _search_once(code, df_train, y_train, samples, best, best_auc, timeout_s)

    # 坐标细化: 每参数在当前最优邻域取点, 其余固定
    refine = []
    for k, (lo, hi) in sorted(bounds.items()):
        w = (hi - lo) / 4.0
        lo2, hi2 = max(lo, best[k] - w), min(hi, best[k] + w)
        for v in np.linspace(lo2, hi2, REFINE_PER_PARAM):
            p = dict(best)
            p[k] = float(v)
            refine.append(p)
    best, best_auc = _search_once(code, df_train, y_train, refine, best, best_auc, timeout_s)
    n_evals = 1 + N_RANDOM + len(refine)

    if best_auc - default_auc < MIN_GAIN:
        return {"tuned": False, "code": code, "default_auc": default_auc,
                "best_auc": best_auc, "n_evals": n_evals, "skip_reason": "搜索无改进"}
    return {"tuned": True, "code": _bake(code, best), "params": best,
            "default_auc": default_auc, "best_auc": best_auc, "n_evals": n_evals}
