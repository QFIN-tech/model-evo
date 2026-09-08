# -*- coding: utf-8 -*-
"""gates.py - G1~G6 关卡实现(含多seed降噪).

要点:
1. G3 场景化: saturated 场景对照=champion全列(严苛); cold_start 对照=本轮已接受(宽松)
2. G4/G5 多seed: 在 K 个不同种子子样本上各训一次, 增益取均值+bootstrap CI
   - mean_gain>=门槛 且 CI下界>=0 才过
   - 救回"单seed略降但多seed均值正"的候选; 拒"单seed侥幸正但均值负"的候选
3. 指标可选 PR-AUC(valid 全量+正样本率低时比 AUC 更敏感)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import candidate_exec, feature_runtime, metrics, paths, state_io, xgb_utils

DEFAULT_GATES = {
    "g1_timeout_s": 60.0,
    "g2_min_auc": 0.5,               # 放行"单变量弱、组合强"的特征; 单看无信号者仍会被 G4/G5 挡
    "g2_min_coverage": 0.3,
    "g2_max_psi": 0.3,
    "g3_max_abs_corr": 0.9,           # 饱和场景缺省(更严)
    "g3_max_abs_corr_cold": 0.95,     # 冷启动场景
    "g4_min_test_gain": 0.0,
    "g5_min_oot_gain": 0.0005,
    "g6_min_oot_gain": 0.001,
    # 多seed
    "n_seeds": 3,                     # 多seed 子样本数(1=关闭, 回退单点)
    "g4_g5_ci_lower_bound": 0.0,      # bootstrap CI 下界门槛(增益 CI 下界须>=此值)
    "g4_g5_metric": "auc",            # "auc" 或 "pr_auc"(饱和场景+正样本率低时建议 pr_auc)
}


def load_gates(session_dir) -> dict:
    """缺省阈值 <- session_config.yaml gates 段覆盖."""
    gates = dict(DEFAULT_GATES)
    state = state_io.load_state(session_dir)
    overrides = (state.get("config_snapshot") or {}).get("gates") or {}
    if isinstance(overrides, dict):
        for k in DEFAULT_GATES:
            if k in overrides:
                gates[k] = overrides[k]
    return gates


def _gain(new, old):
    if new is None or old is None:
        return None
    return round(float(new) - float(old), 6)


def _metric_fn(name: str):
    """返回 (metric_fn, direction). metric_fn(y, score)->float."""
    if name == "pr_auc":
        return metrics.compute_pr_auc, "max"
    return metrics.compute_auc, "max"


def train_eval_model(frames, contract_meta, extra_col=None, xgb_params=None, metric="auc"):
    """固定超参训练 XGB 并三档打分. metric 指定增益口径(auc/pr_auc)."""
    label_col = contract_meta["label_col"]
    cols = feature_runtime.champion_feature_cols(contract_meta)
    if extra_col is not None:
        cols = cols + [extra_col]
    X_train = xgb_utils.as_numeric_frame(frames["train"], cols)
    model = xgb_utils.train_xgb(X_train, frames["train"][label_col], params=xgb_params)
    out = {"model": model, "scores": {}}
    mfn, _ = _metric_fn(metric)
    for split, df in frames.items():
        scores = xgb_utils.predict_scores(model, xgb_utils.as_numeric_frame(df, cols))
        out["scores"][split] = scores
        out.setdefault("metric", {})[split] = mfn(df[label_col], scores)
        out.setdefault("auc", {})[split] = metrics.compute_auc(df[label_col], scores)
        out.setdefault("ks", {})[split] = metrics.compute_ks(df[label_col], scores)
    return out


def champion_baseline_metrics(frames, contract_meta, xgb_params=None, metric="auc"):
    """当前 champion 重训重评, 作 G4/G5 对照基准."""
    return train_eval_model(frames, contract_meta, extra_col=None, xgb_params=xgb_params, metric=metric)


# ---- G1 ----
def gate_g1(code, sample_df, gates):
    return candidate_exec.execute_candidate(code, sample_df, timeout_s=float(gates["g1_timeout_s"]))


# ---- G2 ----
def gate_g2(series_train, y_train, series_oot, gates):
    auc = metrics.compute_auc_direction_fixed(y_train, series_train)
    coverage = float(pd.Series(series_train, dtype=float).notna().mean())
    psi = metrics.compute_psi(pd.Series(series_train, dtype=float), pd.Series(series_oot, dtype=float))
    details = {"train_auc": auc, "coverage": round(coverage, 6), "psi_train_vs_oot": psi}
    if auc is None or np.isnan(auc) or auc < float(gates["g2_min_auc"]):
        return {"passed": False, "reason": "train 方向修正 AUC %.4f < %.4f" % (auc if auc is not None else float("nan"), gates["g2_min_auc"]), "details": details}
    if coverage < float(gates["g2_min_coverage"]):
        return {"passed": False, "reason": "覆盖率 %.4f < %.4f" % (coverage, gates["g2_min_coverage"]), "details": details}
    if psi is not None and not np.isnan(psi) and psi > float(gates["g2_max_psi"]):
        return {"passed": False, "reason": "train vs OOT PSI %.4f > %.4f" % (psi, gates["g2_max_psi"]), "details": details}
    return {"passed": True, "details": details}


# ---- G3 (场景化) ----
def gate_g3(series_train, accepted_train, gates, scenario="cold_start", champ_cols_train=None):
    """G3 冗余. saturated 场景对照=champion全列; cold_start 对照=本轮已接受.

    Args:
        accepted_train: {fid: Series}(本轮已接受)
        champ_cols_train: {col_name: Series}(champion 全列, saturated 场景用)
        scenario: "saturated" 或 "cold_start"
    """
    threshold = float(gates["g3_max_abs_corr"] if scenario == "saturated" else gates["g3_max_abs_corr_cold"])
    # 选择对照集
    if scenario == "saturated" and champ_cols_train is not None:
        ref = champ_cols_train  # 严苛: 对 champion 全列
    else:
        ref = accepted_train or {}  # 宽松: 对本轮已接受
    max_corr, max_with = 0.0, None
    s = pd.Series(series_train, dtype=float)
    for name, other in ref.items():
        o = pd.Series(other, dtype=float)
        pair = pd.concat([s, o], axis=1).dropna()
        if len(pair) < 10 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
            corr = 0.0
        else:
            corr = abs(float(pair.corr(method="spearman").iloc[0, 1]))
        if corr != corr:
            corr = 0.0
        if corr > max_corr:
            max_corr, max_with = corr, name
    details = {"max_abs_spearman": round(max_corr, 6), "max_with": max_with, "ref": "champion_all" if scenario == "saturated" else "accepted"}
    if max_with is not None and max_corr > threshold:
        return {"passed": False, "reason": "与 %s |Spearman|=%.4f > %.4f(冗余)" % (max_with, max_corr, threshold), "details": details}
    return {"passed": True, "details": details}


# ---- G4/G5 多seed (核心) ----
def _bootstrap_ci(gains, n_boot=500, seed=42):
    """bootstrap 增益序列的均值与下界 95% CI."""
    arr = np.array(gains, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, mean
    rng = np.random.RandomState(seed)
    boots = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return mean, float(np.percentile(boots, 2.5))


def gate_g4_g5_multiseed(frames, contract_meta, series_by_split, cid, champion_ref, gates,
                         xgb_params=None, train_idx_seeds=None):
    """G4/G5 多seed: K 个种子子样本各训一次, 增益取均值+CI.

    Args:
        train_idx_seeds: list[np.ndarray], 每个种子抽的 train 行号(同帧保证公平); None=单seed(全量)
    Returns:
        {passed, reason?, details} details 含多seed增益明细
    """
    metric = gates.get("g4_g5_metric", "auc")
    metric_key = "metric"
    mfn, _ = _metric_fn(metric)

    if train_idx_seeds is None or len(train_idx_seeds) <= 1:
        # 回退单seed
        for split, df in frames.items():
            df[cid] = pd.Series(series_by_split[split], dtype=float).to_numpy()
        try:
            result = train_eval_model(frames, contract_meta, extra_col=cid, xgb_params=xgb_params, metric=metric)
        finally:
            for split, df in frames.items():
                df.drop(columns=[cid], inplace=True, errors="ignore")
        gain_test = _gain(result[metric_key]["test"], champion_ref[metric_key]["test"])
        gain_oot = _gain(result[metric_key]["oot"], champion_ref[metric_key]["oot"])
        details = {"n_seeds": 1, "gains_test": [gain_test], "gains_oot": [gain_oot],
                   "mean_gain_test": gain_test, "mean_gain_oot": gain_oot,
                   "ci_lower_oot": gain_oot, "auc": result["auc"], "ks": result["ks"]}
        return _g45_verdict(details, gates, metric)

    # 多seed: 每个种子各训一次
    gains_test, gains_oot = [], []
    for si, idx in enumerate(train_idx_seeds):
        eval_frames = dict(frames)
        eval_frames["train"] = frames["train"].iloc[idx].reset_index(drop=True)
        series_eval = dict(series_by_split)
        series_eval["train"] = series_by_split["train"].iloc[idx].reset_index(drop=True)
        for split, df in eval_frames.items():
            df[cid] = pd.Series(series_eval[split], dtype=float).to_numpy()
        try:
            result = train_eval_model(eval_frames, contract_meta, extra_col=cid, xgb_params=xgb_params, metric=metric)
        finally:
            for split, df in eval_frames.items():
                df.drop(columns=[cid], inplace=True, errors="ignore")
        gains_test.append(_gain(result[metric_key]["test"], champion_ref[metric_key]["test"]))
        gains_oot.append(_gain(result[metric_key]["oot"], champion_ref[metric_key]["oot"]))

    mean_test, ci_lo_test = _bootstrap_ci(gains_test)
    mean_oot, ci_lo_oot = _bootstrap_ci(gains_oot)
    details = {"n_seeds": len(train_idx_seeds), "gains_test": gains_test, "gains_oot": gains_oot,
               "mean_gain_test": round(mean_test, 6), "mean_gain_oot": round(mean_oot, 6),
               "ci_lower_oot": round(ci_lo_oot, 6), "ci_lower_test": round(ci_lo_test, 6)}
    return _g45_verdict(details, gates, metric)


def _g45_verdict(details, gates, metric):
    """据多seed均值+CI 下判定."""
    mt, mo = details["mean_gain_test"], details["mean_gain_oot"]
    clo = details["ci_lower_oot"]
    if mt is None or mo is None:
        return {"passed": False, "reason": "%s 不可计算(某档单类)" % metric, "details": details}
    if mt < float(gates["g4_min_test_gain"]):
        return {"passed": False, "reason": "test %s 增益均值 %.6f < %.6f" % (metric, mt, gates["g4_min_test_gain"]), "details": details}
    if mo < float(gates["g5_min_oot_gain"]):
        return {"passed": False, "reason": "OOT %s 增益均值 %.6f < %.6f" % (metric, mo, gates["g5_min_oot_gain"]), "details": details}
    if not np.isnan(clo) and clo < float(gates["g4_g5_ci_lower_bound"]):
        return {"passed": False, "reason": "OOT 增益 CI 下界 %.6f < %.4f(多seed不稳, 单seed侥幸)" % (clo, gates["g4_g5_ci_lower_bound"]), "details": details}
    return {"passed": True, "details": details}


# ---- G6 (轮级融合, 永远全量) ----
def gate_g6(frames_factory, contract_meta_factory, provisional, state, gates, xgb_params=None, metric="auc"):
    """G6 融合: champion+本轮全部 provisional 重训, OOT 须超上代 champion."""
    metric = gates.get("g4_g5_metric", "auc")
    frames = frames_factory()
    contract_meta = contract_meta_factory()
    result = train_eval_model(frames, contract_meta, extra_col=None, xgb_params=xgb_params, metric=metric)
    prev_oot = state.get("champion_oot_auc" if metric == "auc" else "champion_oot_pr_auc")
    gain_oot = _gain(result["metric"]["oot"], prev_oot)
    details = {"metric": result["metric"], "auc": result["auc"], "ks": result["ks"],
               "prev_champion_oot": prev_oot, "gain_oot": gain_oot,
               "provisional_cids": [p.get("cid") for p in provisional]}
    if gain_oot is None:
        return {"passed": False, "reason": "融合模型 OOT 不可计算", "details": details}
    if gain_oot < float(gates["g6_min_oot_gain"]):
        return {"passed": False, "reason": "融合 OOT 增益 %.6f < %.6f(整轮回滚)" % (gain_oot, gates["g6_min_oot_gain"]), "details": details}
    return {"passed": True, "details": details, "result": result}
