"""误差归因分析(v11):把「当前模型错在哪」翻译成「该造什么特征」。

planner-context 的八节只说"什么方向成功了",没有任何产物说缺口在哪。
本模块在每次 trial 评估后自动产出 error-analysis.json:
- 分箱损失超额:score 分箱内 logloss 份额 vs 样本份额,top 段即"模型
  最无力的人群";
- 高损失群体画像:损失 top 5% 样本 vs 整体的特征均值差(标准化),
  直接指出哪些特征维度上该人群异常;
- 校准方向:各分箱预测均值 vs 实际正样本率(高估/低估)。

分类任务用 logloss 口径;回归用平方误差;uplift 用 score 分箱内的
处置−对照增量 vs 预测 CATE 的偏离。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .logging_utils import dump_json, get_logger
from .safe_dtype import is_feature_column, safe_to_numeric

logger = get_logger(__name__)

#: 高损失群体画像取损失最高的样本比例
TOP_LOSS_FRACTION = 0.05
#: 画像输出最多特征数
PROFILE_TOP_K = 20


def _bin_index(score: np.ndarray, n_bins: int) -> np.ndarray:
    """score 等频分箱 → 每样本 bin 序号(0-based,降序=高分箱在前)。"""
    order = np.argsort(-score, kind="stable")
    bins = np.empty(len(score), dtype=int)
    edges = np.linspace(0, len(score), n_bins + 1).astype(int)
    for b in range(n_bins):
        bins[order[edges[b]:edges[b + 1]]] = b
    return bins


def _per_sample_loss(y: np.ndarray, score: np.ndarray,
                     model_type: str) -> np.ndarray:
    if model_type == "regression":
        return (np.asarray(y, dtype=float) - np.asarray(score,
                                                       dtype=float)) ** 2
    # 分类(含 uplift 的概率侧不可用,由调用方切专属口径)
    eps = 1e-9
    p = np.clip(np.asarray(score, dtype=float), eps, 1 - eps)
    yy = np.asarray(y, dtype=float)
    return -(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def _uplift_bin_deviation(y: np.ndarray, w: np.ndarray,
                          score: np.ndarray, n_bins: int) -> list[dict]:
    """uplift:分箱内实际增量(处置−对照)vs 预测 CATE 均值的偏离。"""
    bins = _bin_index(score, n_bins)
    out = []
    for b in range(n_bins):
        m = bins == b
        if m.sum() < 30:
            continue
        t, c = m & (w == 1), m & (w == 0)
        if t.sum() < 10 or c.sum() < 10:
            continue
        actual = float(y[t].mean() - y[c].mean())
        predicted = float(np.mean(score[m]))
        out.append({"bin": b, "n": int(m.sum()),
                    "actual_uplift": actual, "predicted_uplift": predicted,
                    "deviation": predicted - actual})
    return out


def non_feature_columns(spec=None, label_columns=()) -> set[str]:
    """v2.17.0(D1):误差归因画像必须排除的列集合。

    评估帧是"特征面 join 标签面"(runner._evaluate_trial),标签面的列
    (目标/处置/时间/ID/样本键,以 labels parquet 的实际列集为准)以及
    join 冲突产生的 ``<col>_lab`` 后缀列都不是"该造什么特征"的候选。
    旧实现只排除 ``__mvo_sample_key``,于是目标列本身(z≈+14)登顶
    high_loss_profile,planner-context §11 写成"高损失信号特征: <target>"。
    """
    out: set[str] = set()
    for c in label_columns or ():
        out.add(str(c))
        out.add(f"{c}_lab")
    if spec is not None:
        for attr in ("target_name", "treatment_column", "time_column",
                     "id_column"):
            try:
                v = getattr(spec, attr)
            except Exception:  # noqa: BLE001
                v = None
            if v:
                out.add(str(v))
                out.add(f"{v}_lab")
    return out


def build_error_analysis(
    y: np.ndarray,
    score: np.ndarray,
    parent_score: np.ndarray | None,
    X_valid: pd.DataFrame | None,
    *,
    model_type: str = "binary_classification",
    treatment: np.ndarray | None = None,
    n_bins: int = 10,
    label: str = "",
    exclude_columns=(),
) -> dict:
    """核心分析;产物结构见模块 docstring。X_valid 缺失时跳过画像节。

    ``exclude_columns``:标签面/身份列(见 :func:`non_feature_columns`),
    不进入高损失画像。
    """
    _excl = frozenset(str(c) for c in (exclude_columns or ()))
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    n = len(score)
    if n < 200:
        return {"skipped": True, "reason": f"样本不足({n}<200)"}

    is_uplift = treatment is not None
    out: dict = {"label": label, "n_samples": int(n), "model_type":
                 model_type, "n_bins": n_bins}

    if is_uplift:
        out["uplift_bins"] = _uplift_bin_deviation(
            y, np.asarray(treatment), score, n_bins)
        loss = None  # uplift 的个体"损失"无概率口径,不做超额/画像
    else:
        loss = _per_sample_loss(y, score, model_type)
        bins = _bin_index(score, n_bins)
        total = float(loss.sum()) or 1.0
        rows = []
        for b in range(n_bins):
            m = bins == b
            share_loss = float(loss[m].sum() / total)
            share_n = float(m.mean())
            rows.append({
                "bin": b,
                "n": int(m.sum()),
                "score_mean": float(score[m].mean()),
                "label_rate": float(np.asarray(y, dtype=float)[m].mean()),
                "loss_share": share_loss,
                "n_share": share_n,
                # 超额 = 损失份额 / 样本份额;>2 即"损失浓度"异常段
                "loss_over_n": share_loss / max(share_n, 1e-9),
                "mean_loss": float(loss[m].mean()),
            })
        out["loss_bins"] = rows

        # 高损失群体画像:top 5% 损失样本 vs 整体的特征均值差
        if X_valid is not None and len(X_valid) == n:
            k = max(30, int(n * TOP_LOSS_FRACTION))
            idx = np.argsort(-loss)[:k]
            profile = []
            base_rows = np.ones(n, dtype=bool)
            base_rows[idx] = False
            # 只对有真实判别力的维度画像:等频损失分箱至少要有 2 个非空箱,
            # 否则 top-5% 与其余样本的"均值差"只是分箱伪影。
            n_nonempty = sum(1 for r in rows if r["n"] > 0)
            for c in X_valid.columns:
                # 非特征列(样本键/时间/实体/目标)不参与画像:它们不是
                # "该造什么特征"的候选,混进来只会污染产物。
                if not is_feature_column(str(c), extra_exclude=_excl):
                    continue
                col = safe_to_numeric(X_valid[c])
                if col is None:
                    continue
                if col.notna().sum() < n // 2:
                    continue
                hi, lo = col.iloc[idx], col.iloc[base_rows]
                if lo.std() and np.isfinite(lo.std()) and lo.std() > 0:
                    z = (hi.mean() - lo.mean()) / lo.std()
                    if np.isfinite(z) and abs(z) > 0.1:
                        profile.append({"feature": str(c),
                                        "z_highloss_vs_rest": float(z)})
            profile.sort(key=lambda r: -abs(r["z_highloss_vs_rest"]))
            out["high_loss_profile"] = profile[:PROFILE_TOP_K]
            out["high_loss_top_fraction"] = TOP_LOSS_FRACTION
            if n_nonempty < 2:
                # 无判别力时仍输出画像字段,但显式标注不可用,避免下游
                # 把"分箱伪影"当成真实缺口方向。
                out["high_loss_profile_unreliable"] = (
                    "loss 分箱非空箱数 < 2,画像仅作诊断参考")

        # 校准方向(分类):分箱预测均值 vs 实际正样本率
        if model_type in ("binary_classification", "multiclass"):
            cal = []
            for r in out["loss_bins"]:
                cal.append({
                    "bin": r["bin"],
                    "pred_mean": r["score_mean"],
                    "actual_rate": r["label_rate"],
                    "bias": r["score_mean"] - r["label_rate"],
                })
            out["calibration"] = cal

    # 父代对照:错误集中的转移(哪些段比父代更差/更好)
    if parent_score is not None and loss is not None:
        ploss = _per_sample_loss(y, parent_score, model_type)
        delta = ploss - loss  # >0 = 该样本上子代比父代损失更低(改善)
        bins = _bin_index(score, n_bins)
        out["delta_vs_parent_bins"] = [
            {"bin": b,
             "mean_improvement": float(delta[bins == b].mean())
             if (bins == b).any() else None}
            for b in range(n_bins)]
    return out


def write_error_analysis(out_dir: Path, payload: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "error-analysis.json"
    dump_json(path, payload)
    return path


def summarize_for_evaluation(payload: dict) -> dict:
    """evaluation.json 内的摘要(全文在 error-analysis.json)。"""
    if payload.get("skipped"):
        return {"skipped": True, "reason": payload.get("reason")}
    s: dict = {}
    lb = payload.get("loss_bins") or []
    if lb:
        worst = max(lb, key=lambda r: r.get("loss_over_n", 0))
        s["worst_bin"] = {"bin": worst["bin"],
                          "loss_over_n": round(worst["loss_over_n"], 3),
                          "n": worst["n"]}
    hp = payload.get("high_loss_profile") or []
    if hp:
        s["high_loss_signals"] = [
            f"{r['feature']}({r['z_highloss_vs_rest']:+.2f})"
            for r in hp[:5]]
    ub = payload.get("uplift_bins") or []
    if ub:
        worst = max(ub, key=lambda r: abs(r.get("deviation", 0)))
        s["worst_uplift_bin"] = {"bin": worst["bin"],
                                 "deviation": round(worst["deviation"], 5)}
    return s
