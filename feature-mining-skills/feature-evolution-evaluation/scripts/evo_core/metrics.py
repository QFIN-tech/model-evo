# -*- coding: utf-8 -*-
"""metrics.py - 指标口径模块(AUC/KS/IV/PSI/十分桶 + PR-AUC).

口径(AUC/KS/IV/PSI/十分桶)与 model-skills 评估逐行对齐, 另新增:
- compute_pr_auc: average_precision_score (valid 全量+低正样本率场景下
  PR-AUC 比 AUC 更敏感, 作饱和场景 g4_g5_metric 默认口径)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

_PSI_EPS = 1e-6


# ---- AUC / PR-AUC / KS ----
def compute_auc(y, score):
    """y 单类返回 None ."""
    y = pd.Series(y)
    if y.nunique() < 2:
        return None
    return float(roc_auc_score(y, score))


def compute_pr_auc(y, score):
    """PR-AUC = average_precision_score. y 单类返回 None."""
    y = pd.Series(y)
    if y.nunique() < 2:
        return None
    return float(average_precision_score(y, score))


def compute_auc_direction_fixed(y, score) -> float:
    """单变量特征口径: 方向修正 AUC = max(a, 1-a); 单类/全NaN 返回 nan."""
    df = pd.DataFrame({"y": pd.Series(y).values, "s": pd.Series(score, dtype=float).values})
    df = df.dropna()
    if df["y"].nunique() < 2 or df["s"].nunique() < 2:
        return float("nan")
    a = float(roc_auc_score(df["y"], df["s"]))
    return max(a, 1.0 - a)


def compute_ks(y, score):
    df = pd.DataFrame({"label": pd.Series(y).values, "s": pd.Series(score, dtype=float).values})
    df = df.dropna()
    if df["label"].nunique() < 2:
        return None
    s = df.sort_values("s", ascending=False)
    cp = (s["label"] == 1).cumsum() / max((s["label"] == 1).sum(), 1)
    cn = (s["label"] == 0).cumsum() / max((s["label"] == 0).sum(), 1)
    return float(abs(cp - cn).max())


# ---- IV  ----
def _bin_edges(series, n_bins):
    arr = series.dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.array([])
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(arr, qs))
    if edges.size < 2:
        return np.array([])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def compute_iv(feature, label, n_bins=10):
    df = pd.DataFrame({"x": feature.values, "y": label.values})
    df = df[df["y"].isin([0, 1])]
    y = df["y"].astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return {"iv": float("nan"), "n_bins_effective": 0, "auc": float("nan")}
    edges = _bin_edges(df["x"], n_bins) if pd.api.types.is_numeric_dtype(df["x"]) else np.array([])
    if edges.size >= 2:
        df["bin"] = pd.cut(df["x"], bins=edges, include_lowest=True).astype(object)
    else:
        df["bin"] = np.nan
    df.loc[df["x"].isna(), "bin"] = "MISSING"
    total_pos = int(y.sum()); total_neg = int((1 - y).sum())
    iv_total = 0.0; woe_map = {}
    for b, sub in df.groupby("bin", dropna=False):
        pos = int(sub["y"].sum()); neg = int(len(sub) - pos)
        pos_share = (pos + 0.5) / (total_pos + 0.5)
        neg_share = (neg + 0.5) / (total_neg + 0.5)
        woe = float(math.log(pos_share / neg_share))
        iv_total += (pos_share - neg_share) * woe
        woe_map[b] = woe
    df["woe"] = df["bin"].map(woe_map)
    try:
        auc = float(roc_auc_score(y, df["woe"]))
        if auc < 0.5:
            auc = 1 - auc
    except (ValueError, FloatingPointError):
        auc = float("nan")
    return {"iv": round(float(iv_total), 6), "n_bins_effective": int(df["bin"].nunique(dropna=False)),
            "auc": round(auc, 6) if not np.isnan(auc) else float("nan")}


# ---- PSI  ----
def compute_psi(train_series, oot_series, n_bins=10):
    edges = _bin_edges(train_series, n_bins)
    n_train = len(train_series); n_oot = len(oot_series)
    if n_train == 0 or n_oot == 0 or edges.size < 2:
        return float("nan")

    def _dist(series):
        n = len(series); non_null = series.dropna()
        if non_null.size == 0:
            cuts = np.zeros(edges.size - 1)
        else:
            cats = pd.cut(non_null, bins=edges, include_lowest=True)
            cuts = cats.value_counts(sort=False).to_numpy(dtype=float)
        miss = n - int(cuts.sum())
        return np.concatenate([cuts, [miss]]) / n
    pt = _dist(train_series); po = _dist(oot_series)
    pt = np.where(pt == 0, _PSI_EPS, pt); po = np.where(po == 0, _PSI_EPS, po)
    return round(float(np.sum((po - pt) * np.log(po / pt))), 6)


# ---- 十分桶  ----
def decile_buckets(y, score, n_bins=10):
    s = pd.DataFrame({"label": pd.Series(y).values, "score": pd.Series(score, dtype=float).values})
    s = s.dropna().sort_values("score", ascending=False).reset_index(drop=True)
    if len(s) == 0:
        return []
    s["decile"] = pd.cut(pd.Series(range(len(s))), bins=n_bins, labels=False) + 1
    s["decile"] = n_bins + 1 - s["decile"]
    overall_lr = float(s["label"].mean()); total_pos = int(s["label"].sum()); cum_pos = 0; result = []
    for d in range(n_bins, 0, -1):
        b = s[s["decile"] == d]
        bucket_pos = int(b["label"].sum()); cum_pos += bucket_pos
        bucket_lr = float(b["label"].mean())
        result.append({"decile": d, "count": int(len(b)),
                       "score_min": round(float(b["score"].min()), 4), "score_max": round(float(b["score"].max()), 4),
                       "label_rate": round(bucket_lr, 6),
                       "lift": round(bucket_lr / overall_lr, 4) if overall_lr > 0 else None,
                       "recall": round(bucket_pos / total_pos, 4) if total_pos > 0 else None,
                       "cum_recall": round(cum_pos / total_pos, 4) if total_pos > 0 else None})
    return result


def eval_bundle(y, score, n_bins=10):
    y = pd.Series(y); n = int(len(y))
    bundle = {"count": n, "label_rate": round(float(y.mean()), 6) if n else None,
              "auc": None, "pr_auc": None, "ks": None, "buckets": []}
    if n == 0:
        return bundle
    bundle["auc"] = round(compute_auc(y, score), 6) if compute_auc(y, score) is not None else None
    bundle["pr_auc"] = round(compute_pr_auc(y, score), 6) if compute_pr_auc(y, score) is not None else None
    ks = compute_ks(y, score)
    bundle["ks"] = round(ks, 6) if (ks is not None and n > 50) else None
    bundle["buckets"] = decile_buckets(y, score, n_bins=n_bins)
    return bundle


# ---- 单档评估打包(baseline / champion 落盘用) ----
def build_eval_json(produced_by, model_meta, split, bundle):
    """评估 JSON(对齐 model-skills eval_single 核心字段, 含 pr_auc)."""
    return {
        "schema_version": 1, "produced_by": produced_by, "model_meta": model_meta,
        "data_info": {"split": split, "count": bundle["count"], "label_rate": bundle["label_rate"]},
        "metrics": {"auc": bundle.get("auc"), "pr_auc": bundle.get("pr_auc"), "ks": bundle.get("ks")},
        "score_buckets": bundle["buckets"],
    }


def build_eval_md(model_name, split, bundle):
    """人类可读评估报告(对齐 model-skills eval_single md 摘要行 + 分桶表风格)."""
    auc_s = "%.4f" % bundle["auc"] if bundle.get("auc") is not None else "N/A"
    ks_s = "%.4f" % bundle["ks"] if bundle.get("ks") is not None else "N/A"
    lr_s = "%.4f%%" % (bundle["label_rate"] * 100) if bundle.get("label_rate") is not None else "N/A"
    md = "# %s — %s 评估\n\n> AUC=%s, KS=%s, 样本量=%s, label率=%s\n\n" % (
        model_name, split, auc_s, ks_s, bundle.get("count", 0), lr_s)
    md += "| 分桶 | 样本数 | 分数区间 | label率 | Lift | 召回率 | 累计召回 |\n|---|---|---|---|---|---|---|\n"
    for b in bundle["buckets"]:
        lift = "%.2f" % b["lift"] if b.get("lift") is not None else "-"
        recall = "%.2f%%" % (b["recall"] * 100) if b.get("recall") is not None else "-"
        cum = "%.2f%%" % (b["cum_recall"] * 100) if b.get("cum_recall") is not None else "-"
        md += "| %s | %s | [%s, %s] | %.4f%% | %s | %s | %s |\n" % (
            b["decile"], b["count"], b["score_min"], b["score_max"],
            b["label_rate"] * 100, lift, recall, cum)
    return md
