# -*- coding: utf-8 -*-
"""单变量预测力: 等频分箱 + WOE/IV + 单变量 AUC。

缺失单独分桶, 不参与分位数计算; 0 频数桶按 0.5 拉普拉斯平滑避免 log(0)。
不做特征剔除, 仅输出指标供人工判断。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _bin_edges(series: pd.Series, n_bins: int) -> np.ndarray:
    """对非空数值列计算等频分箱边界(去重后可能少于 n_bins+1)。"""
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


def _bin_labels(edges: np.ndarray) -> List[str]:
    """生成形如 (a, b] 的桶标签。"""
    out = []
    for i in range(edges.size - 1):
        lo = edges[i]
        hi = edges[i + 1]
        lo_s = "-inf" if np.isneginf(lo) else f"{lo:.4g}"
        hi_s = "+inf" if np.isposinf(hi) else f"{hi:.4g}"
        out.append(f"({lo_s}, {hi_s}]")
    return out


def compute_iv_for_feature(
    feature: pd.Series, label: pd.Series, n_bins: int = 10
) -> dict:
    """计算单个特征的 IV / 有效分箱数 / 单变量 AUC。

    Args:
        feature: 待评估的特征列
        label: 0/1 标签列(与 feature 同 index)
        n_bins: 等频分箱数

    Returns:
        dict 含 iv / n_bins_effective / auc / bins_detail(每桶 cnt/pos/neg/woe)
    """
    df = pd.DataFrame({"x": feature.values, "y": label.values})
    df = df[df["y"].isin([0, 1])]
    y = df["y"].astype(int)
    if y.sum() == 0 or (1 - y).sum() == 0:
        return {"iv": np.nan, "n_bins_effective": 0, "auc": np.nan, "bins_detail": []}

    edges = _bin_edges(df["x"], n_bins) if pd.api.types.is_numeric_dtype(df["x"]) else np.array([])
    bins_detail = []
    if edges.size >= 2:
        df["bin"] = pd.cut(df["x"], bins=edges, include_lowest=True)
        df["bin"] = df["bin"].astype(object)
    else:
        df["bin"] = np.nan
    df.loc[df["x"].isna(), "bin"] = "MISSING"

    total_pos = int(y.sum())
    total_neg = int((1 - y).sum())
    iv_total = 0.0
    woe_map: dict = {}
    for b, sub in df.groupby("bin", dropna=False):
        pos = int(sub["y"].sum())
        neg = int(len(sub) - pos)
        # pos_share / neg_share: WoE 公式里的 distribution share, 即该 bin 占全部正/负样本的比例
        # (0 频数桶用 0.5 拉普拉斯平滑避免 log(0)); 与下方落盘的 pos_rate(bin 内正样本率)是不同量,
        # 不要混淆: pos_share = pos / total_pos; pos_rate = pos / len(sub)
        pos_share = (pos + 0.5) / (total_pos + 0.5)
        neg_share = (neg + 0.5) / (total_neg + 0.5)
        woe = float(np.log(pos_share / neg_share))
        iv_bin = (pos_share - neg_share) * woe
        iv_total += iv_bin
        woe_map[b] = woe
        bins_detail.append(
            {
                "bin": str(b),
                "cnt": int(len(sub)),
                "pos": pos,
                "neg": neg,
                "pos_rate": round(pos / len(sub), 6) if len(sub) else 0.0,
                "woe": round(woe, 6),
                "iv": round(iv_bin, 6),
            }
        )

    df["woe"] = df["bin"].map(woe_map)
    try:
        auc = float(roc_auc_score(y, df["woe"]))
        if auc < 0.5:
            auc = 1 - auc
    except (ValueError, FloatingPointError) as e:
        # 已在 L58 兜底单类标签;此处主要捕获 woe 全 NaN / 全相等等边界
        print(f"[feature_iv] AUC 计算失败 (n={len(y)}, pos={int(y.sum())}): {e}")
        auc = np.nan

    return {
        "iv": round(float(iv_total), 6),
        "n_bins_effective": int(df["bin"].nunique(dropna=False)),
        "auc": round(auc, 6) if not np.isnan(auc) else np.nan,
        "bins_detail": bins_detail,
    }


def compute_iv_table(
    df: pd.DataFrame, features: List[str], label_col: str, n_bins: int = 10
) -> pd.DataFrame:
    """批量计算特征 IV/AUC 表, 按 IV 降序。

    Args:
        df: 含 features + label_col 的全样本
        features: 待评估特征清单
        label_col: 标签列名(0/1)
        n_bins: 等频分箱数

    Returns:
        DataFrame, 列: feature/iv/auc/n_bins_effective, 按 iv 降序
    """
    rows = []
    for f in features:
        if f not in df.columns:
            rows.append(
                {"feature": f, "iv": np.nan, "auc": np.nan, "n_bins_effective": 0}
            )
            continue
        res = compute_iv_for_feature(df[f], df[label_col], n_bins=n_bins)
        rows.append(
            {
                "feature": f,
                "iv": res["iv"],
                "auc": res["auc"],
                "n_bins_effective": res["n_bins_effective"],
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("iv", ascending=False, na_position="last").reset_index(drop=True)


def build_woe_table(
    df: pd.DataFrame, features: List[str], label_col: str, n_bins: int = 10
) -> pd.DataFrame:
    """批量产出 WOE 分桶明细 long-format 表。

    复用 compute_iv_for_feature 已算的 bins_detail, 不重算。
    每特征每桶一行(含 MISSING 桶), 列:
        feature / bin / cnt / pos / neg / pos_rate / woe / iv_bin

    Args:
        df: 含 features + label_col 的全样本
        features: 待评估特征清单
        label_col: 标签列名(0/1)
        n_bins: 等频分箱数

    Returns:
        DataFrame; 特征不在 df 中时该特征不出行(不占位)
    """
    rows = []
    for f in features:
        if f not in df.columns:
            continue
        res = compute_iv_for_feature(df[f], df[label_col], n_bins=n_bins)
        for b in res["bins_detail"]:
            rows.append({
                "feature": f,
                "bin": b["bin"],
                "cnt": b["cnt"],
                "pos": b["pos"],
                "neg": b["neg"],
                "pos_rate": b["pos_rate"],
                "woe": b["woe"],
                "iv_bin": b["iv"],
            })
    return pd.DataFrame(
        rows,
        columns=["feature", "bin", "cnt", "pos", "neg", "pos_rate", "woe", "iv_bin"],
    )
