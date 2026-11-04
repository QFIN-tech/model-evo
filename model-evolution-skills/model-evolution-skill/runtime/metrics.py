"""指标计算(§12.):统一口径,LLM 不得自行计算 acceptance 指标。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

# 回归指标:compute_metric 的 y_pred 缺省时回退用 score 列
# (回归任务的 validation_score.parquet 的 score 即预测值,不是概率)
REGRESSION_METRICS = {"rmse", "mae", "mape", "r2"}

# uplift 指标:score 列是 CATE 估计(个体处置效应,可正可负),
# 评测必须知道每个样本的 treatment 归属(t: 1=处置 0=对照)
UPLIFT_METRICS = {"uplift_auc", "qini", "auuc"}

# 指标方向:true 表示越大越好
METRIC_DIRECTIONS = {
    "auc": True, "ks": True, "gini": True, "logloss": False,
    "rmse": False, "mae": False, "mape": False, "r2": True,
    "nll": False, "accuracy": True, "f1": True, "precision": True,
    "recall": True, "auprc": True, "normalized_gini": True,
    "uplift_auc": True, "qini": True, "auuc": True,
    "lift_top10": True, "lift_top30": True,
}

# 头部圈人 lift 指标:top X% 样本的正样本率 / 全体正样本率
# (分类专属,只做交付视图,不作 primary)
TOP_LIFT_METRICS = {"lift_top10": 10, "lift_top30": 30}


def metric_direction(name: str) -> bool:
    if name in METRIC_DIRECTIONS:
        return METRIC_DIRECTIONS[name]
    # 未知指标默认越大越好,并显式标注
    return True


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


class _BootstrapAUC:
    """bootstrap 专用 AUC:score 只排一次序,每个重采样用整数权重 O(n) 求值。

    v2.13.5(req-t7-rand400w 实测回写):评估段在 44.7 万行 validation 上
    做 2000 次配对 bootstrap ×(vs_parent + vs_baseline)+ 2000 次 CI,
    每次重采样都调 sklearn roc_auc_score(重新 argsort),单 trial 固定
    ~7.3 min(本机实测 98 ms/次),比 GPU 训练本身还长。

    重采样 idx 上的 AUC 等于原数组上权重 w = bincount(idx) 的加权
    Mann–Whitney 统计量(同分按 1/2,即 sklearn 的梯形口径);重采样本身
    (rng 抽取序)与旧实现逐位相同。整数权重下分子分母都是精确整数,
    结果是真值的一次舍入,与 sklearn(tpr/fpr 各自舍入后梯形累加)只差
    浮点舍入量级(实测 |Δ| < 1e-12)。与 v2.14.3.1 的同名实现同口径,
    便于日后合并。
    """

    def __init__(self, y_true: np.ndarray, score: np.ndarray):
        y = np.asarray(y_true)
        labels = np.unique(y)
        self.ok = len(labels) == 2
        if not self.ok:
            return
        s_all = np.asarray(score, dtype=float)
        if not np.isfinite(s_all).all():
            # NaN/inf 分数:sklearn 会抛错 → 旧路径记 NaN;快路径不接管
            self.ok = False
            return
        order = np.argsort(-s_all, kind="mergesort")
        s = s_all[order]
        self.order = order
        self.pos = (y[order] == labels[1])
        self.starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]

    def at(self, counts: np.ndarray) -> float:
        w = counts[self.order].astype(np.int64)
        wp = np.where(self.pos, w, 0)
        wn = w - wp
        p, n = int(wp.sum()), int(wn.sum())
        if p == 0 or n == 0:
            return float("nan")
        tp = np.add.reduceat(wp, self.starts)
        fp = np.add.reduceat(wn, self.starts)
        tp_before = np.cumsum(tp) - tp
        num = int(np.sum(fp * (2 * tp_before + tp)))
        return num / (2 * p * n)


def _fast_bootstrap_auc(metric: str, treatment) -> bool:
    """bootstrap 走 _BootstrapAUC 的条件;MODELEVO_BOOTSTRAP_FAST_AUC=0 退回 sklearn。"""
    import os as _os

    return (str(metric).lower() == "auc" and treatment is None
            and _os.environ.get("MODELEVO_BOOTSTRAP_FAST_AUC", "1") != "0")


def ks_stat(y_true: np.ndarray, score: np.ndarray) -> float:
    """KS = max |CDF_pos - CDF_neg|。"""
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    pos, neg = score[y_true == 1], score[y_true != 1]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    all_v = np.sort(np.concatenate([pos, neg]))
    cdf_pos = np.searchsorted(np.sort(pos), all_v, side="right") / len(pos)
    cdf_neg = np.searchsorted(np.sort(neg), all_v, side="right") / len(neg)
    return float(np.max(np.abs(cdf_pos - cdf_neg)))


def logloss(y_true: np.ndarray, score: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(score, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def gini(y_true: np.ndarray, score: np.ndarray) -> float:
    return 2.0 * roc_auc(y_true, score) - 1.0


def uplift_curve(y: np.ndarray, w: np.ndarray, score: np.ndarray,
                 *, normalize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Uplift 曲线(Radcliffe 2007):按 score 降序累积。

    返回 (x, y_cum):
    - x:已选样本占比 [0,1](normalize=False 时为样本数);
    - y_cum:累积增益。top-k 的增益定义
      ``cum_y_t(k)/n_t * n - cum_y_c(k)/n_c * n``
      (处置组累计响应 × 总量占比 − 对照组累计响应 × 总量占比;
      normalize=True 时把总量占比换成各自组占比,即
      ``cum_y_t(k) − cum_y_c(k) * n_t/n_c``,两口径单调性一致)。

    score 是 CATE 估计(可正可负);排序方向固定降序,负效应样本沉底。
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    score = np.asarray(score, dtype=float)
    n = len(y)
    n_t, n_c = float((w == 1).sum()), float((w == 0).sum())
    if n == 0 or n_t == 0 or n_c == 0:
        raise ValueError("uplift 指标需要 treatment 同时含处置与对照样本")
    order = np.argsort(-score, kind="stable")
    y_s, w_s = y[order], w[order]
    cum_y_t = np.cumsum(y_s * (w_s == 1))
    cum_y_c = np.cumsum(y_s * (w_s == 0))
    k = np.arange(1, n + 1, dtype=float)
    if normalize:
        gain = cum_y_t - cum_y_c * (n_t / n_c)
    else:
        scale_t, scale_c = n / n_t, n / n_c
        gain = cum_y_t * scale_t - cum_y_c * scale_c
    x = k / n if normalize else k
    x = np.concatenate([[0.0], x])
    gain = np.concatenate([[0.0], gain])
    return x, gain


def _auuc(y: np.ndarray, w: np.ndarray, score: np.ndarray) -> float:
    """AUUC:uplift 曲线下面积(梯形积分,x=样本占比)。"""
    x, gain = uplift_curve(y, w, score, normalize=True)
    return float(np.trapezoid(gain, x)) if hasattr(np, "trapezoid") \
        else float(np.trapz(gain, x))


def _qini_coefficient(y: np.ndarray, w: np.ndarray, score: np.ndarray) -> float:
    """Qini 系数:实际 uplift 曲线与随机基线围成的面积。

    随机基线是从 (0,0) 到 (1, ate*n_t) 的直线——排序无信息时
    按任意顺序累积,增益近似线性增长到 ATE×n_t。曲线下总面积
    (梯形)减基线下面积即 Qini 系数;score 排序有效时显著为正。
    """
    x, gain = uplift_curve(y, w, score, normalize=True)
    n_t = float((w == 1).sum())
    ate = float(y[w == 1].mean() - y[w == 0].mean())
    baseline = x * (ate * n_t)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    area_curve = _trapz(gain, x)
    area_base = _trapz(baseline, x)
    return float(area_curve - area_base)


def top_lift(y_true: np.ndarray, score: np.ndarray, top_percent: int) -> float:
    """头部 lift:top X% 样本(按 score 降序)的正样本率 / 全体正样本率。

    圈人交付口径:X=10 即 lift_top10。全样本无正样本时返回 nan。
    """
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    n = len(y_true)
    base = float(y_true.mean()) if n else float("nan")
    if n == 0 or base == 0:
        return float("nan")
    k = max(1, int(round(n * top_percent / 100.0)))
    k = min(k, n)
    order = np.argsort(-score, kind="stable")
    return float(y_true[order[:k]].mean() / base)


def segment_ranking(y_true: np.ndarray, score: np.ndarray,
                    n_bins: int = 10,
                    treatment: np.ndarray | None = None) -> dict:
    """分段排序性:按 score 分位分 N 段(默认 10 段),校验段内目标
    均值是否随段位单调(风险模型排序性/单调性检验的通用口径)。

    - 段 0 是 score 最低段,段 n_bins-1 是最高段;
    - bin_means:普通任务为段内 y_true 均值(分类即正样本率);
      uplift 任务为段内 处置组均值 − 对照组均值(段内增量);
    - monotonicity ∈ [0,1]:相邻段变化方向一致的比例(max(up,down)
      口径,方向由主指标裁决,这里只度量有序性);
    - spearman:score 与目标的 |Spearman| 秩相关(uplift 时不计算,
      因个体观测混含处置/对照,秩相关无意义);
    - lift_top:top 段均值 / 全体均值(uplift 时为 top 段增量/ATE)。

    样本不足(段均 <10)或 score 退化(分不出 n_bins 段)时返回
    passed=None——排序性是附加诊断,不算失败也不阻塞 verdict。
    """
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    is_uplift = treatment is not None
    if is_uplift:
        treatment = np.asarray(treatment)
    n = len(y_true)
    out: dict[str, Any] = {
        "n_bins": int(n_bins), "monotonicity": None, "spearman": None,
        "bin_means": None, "lift_top": None, "passed": None,
    }
    if n_bins < 2 or n < n_bins * 10:
        return out
    # 分位分箱:rank 化后均分 N 段,并列值不跨段(比 qcut+digitize
    # 对重复边界更稳);score 退化(唯一值 < n_bins)分不出段。
    if len(np.unique(score)) < n_bins:
        return out
    order = np.argsort(score, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    bins = (ranks * n_bins / n).astype(int).clip(0, n_bins - 1)
    if len(np.unique(bins)) < n_bins:
        return out

    means, overall = [], None
    try:
        for b in range(n_bins):
            m = bins == b
            if is_uplift:
                mt, mc = m & (treatment == 1), m & (treatment == 0)
                if mt.sum() == 0 or mc.sum() == 0:
                    return out
                means.append(y_true[mt].mean() - y_true[mc].mean())
            else:
                means.append(y_true[m].mean())
        if is_uplift:
            t, c = treatment == 1, treatment == 0
            overall = y_true[t].mean() - y_true[c].mean()
        else:
            overall = y_true.mean()
    except Exception:  # noqa: BLE001
        return out
    means = np.asarray(means, dtype=float)
    if not np.all(np.isfinite(means)):
        return out
    up = int((np.diff(means) > 0).sum())
    down = int((np.diff(means) < 0).sum())
    out["monotonicity"] = float(max(up, down) / (n_bins - 1))
    out["bin_means"] = [float(m) for m in means]
    if overall not in (0, None) and np.isfinite(overall):
        out["lift_top"] = float(means[-1] / overall)
    if not is_uplift:
        import pandas as pd

        if pd.Series(score).nunique() > 1:
            corr = pd.Series(score).corr(pd.Series(y_true),
                                         method="spearman")
            if corr is not None and np.isfinite(corr):
                out["spearman"] = float(abs(corr))
    return out


def segment_passed(seg: dict, threshold: float = 0.8) -> bool | None:
    """分段排序性判定(v2.15.9):单调性达标且段均值随 score 递增。

    score 在三类任务里都与目标同向(分类=正类概率、回归=预测值、
    uplift=CATE),段均值应随段位递增,与"指标越大/越小越好"无关。
    旧实现把递增方向与 metric_direction 绑定:rmse 等越小越好的回归任务
    即便段均值严格单调递增也恒判 False(实测 req-t7 run4 regression,
    monotonicity=1.0、passed=False)。bin_means 缺失时返回 None。
    """
    means = seg.get("bin_means")
    mono = seg.get("monotonicity")
    if not means or mono is None:
        return None
    increasing = means[-1] >= means[0]
    return bool(float(mono) >= float(threshold) and increasing)


def compute_metric(name: str, y_true: np.ndarray, y_pred: np.ndarray,
                   *, score: np.ndarray | None = None,
                   treatment: np.ndarray | None = None) -> float:
    """统一指标入口。分类用 score(概率),回归用 y_pred(预测值),
    uplift 用 score(CATE 估计)+ treatment(处置归属)。

    回归指标(rmse/mae/mape/r2)在 y_pred=None 时回退用 score:
    Trial 的统一打分产物是 validation_score.parquet 的 score 列,
    回归任务中该列即预测值。uplift 指标(uplift_auc/qini/auuc)
    必须传 treatment,缺失时抛 ValueError(静默当作普通分类会让
    指标完全失真)。"""
    name = name.lower()
    y_true = np.asarray(y_true)
    if name in REGRESSION_METRICS and y_pred is None and score is not None:
        y_pred = score
    if name == "auc":
        return roc_auc(y_true, score)
    if name == "ks":
        return ks_stat(y_true, score)
    if name in ("logloss", "nll"):
        return logloss(y_true, score)
    if name == "gini":
        return gini(y_true, score)
    if name in UPLIFT_METRICS:
        if treatment is None:
            raise ValueError(f"uplift 指标 {name} 需要 treatment 数组")
        if score is None:
            raise ValueError(f"uplift 指标 {name} 需要 score(CATE 估计)")
        w = np.asarray(treatment)
        if name == "auuc":
            return _auuc(y_true, w, score)
        if name == "qini":
            return _qini_coefficient(y_true, w, score)
        return _auuc(y_true, w, score)  # uplift_auc:AUUC 同口径别名
    if name == "rmse":
        d = y_true - np.asarray(y_pred, dtype=float)
        return float(np.sqrt(np.mean(d ** 2)))
    if name == "mae":
        return float(np.mean(np.abs(y_true - np.asarray(y_pred, dtype=float))))
    if name == "mape":
        y_pred = np.asarray(y_pred, dtype=float)
        nz = y_true != 0
        return float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])))
    if name == "r2":
        ss_res = np.sum((y_true - np.asarray(y_pred, dtype=float)) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    if name == "auprc":
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true, score))
    if name in TOP_LIFT_METRICS:
        if treatment is not None:
            raise ValueError(f"{name} 是分类头部圈人口径,不支持 uplift")
        return top_lift(y_true, score, TOP_LIFT_METRICS[name])
    raise ValueError(f"不支持的指标: {name}")


def bootstrap_ci(
    y_true: np.ndarray, score: np.ndarray, metric: str, *,
    treatment: np.ndarray | None = None,
    n_boot: int = 2000, seed: int = 20260814, alpha: float = 0.05,
) -> tuple[float, float]:
    """指标 bootstrap 置信区间;大样本自动子采样以保证速度。

    回归指标(score 即预测值)与分类指标(score 即概率)同口径重采样;
    uplift 指标连 treatment 一起重采样(样本配对关系不能拆)。
    v12(P0-2):默认 n_boot 200→2000(200 次的 2.5% 分位数由 ≤5 个
    极端样本决定,分辨率不足)。
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    treatment = None if treatment is None else np.asarray(treatment)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    max_n = 200_000
    if n > max_n:
        idx = rng.choice(n, max_n, replace=False)
        y_true, score = y_true[idx], score[idx]
        if treatment is not None:
            treatment = treatment[idx]
        n = max_n
    values = np.empty(n_boot)
    fast = _BootstrapAUC(y_true, score) \
        if _fast_bootstrap_auc(metric, treatment) else None
    if fast is not None and not fast.ok:
        fast = None
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if fast is not None:
            values[b] = fast.at(np.bincount(idx, minlength=n))
            continue
        yt, sc = y_true[idx], score[idx]
        w = treatment[idx] if treatment is not None else None
        try:
            v = compute_metric(metric, yt, sc, score=sc, treatment=w)
        except Exception:  # noqa: BLE001
            v = np.nan
        values[b] = v
    values = values[np.isfinite(values)]
    if len(values) < n_boot * 0.5:
        return float("nan"), float("nan")
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def align_score_by_keys(score_old: np.ndarray, keys: np.ndarray,
                        old_keys: np.ndarray) -> np.ndarray:
    """把旧预测按当前样本键重排；键不唯一/集合不同即拒绝比较。"""
    score_old = np.asarray(score_old, dtype=float)
    keys = np.asarray(keys, dtype=object)
    old_keys = np.asarray(old_keys, dtype=object)
    if keys.ndim == 1:
        keys = keys.reshape(-1, 1)
    if old_keys.ndim == 1:
        old_keys = old_keys.reshape(-1, 1)
    if len(old_keys) != len(score_old):
        raise ValueError(
            "paired comparison:old_keys 行数与 score_old 不一致")
    if keys.shape[1] != old_keys.shape[1]:
        raise ValueError("paired comparison:新旧样本键维数不一致")
    new_index = pd.MultiIndex.from_arrays(
        [keys[:, i] for i in range(keys.shape[1])])
    old_index = pd.MultiIndex.from_arrays(
        [old_keys[:, i] for i in range(old_keys.shape[1])])
    if not new_index.is_unique or not old_index.is_unique:
        raise ValueError(
            "paired comparison:对齐键存在重复(键不唯一不可比)")
    old_pos = pd.Index(old_index).get_indexer(new_index)
    if (old_pos < 0).any() or len(new_index) != len(old_index):
        missing = int((old_pos < 0).sum())
        raise ValueError(
            f"paired comparison:新旧样本集合不同(missing={missing},"
            f"new={len(new_index)},old={len(old_index)})")
    return score_old[old_pos]


def paired_bootstrap_delta(
    y_true: np.ndarray, score_new: np.ndarray, score_old: np.ndarray,
    metric: str, *, treatment: np.ndarray | None = None,
    keys: np.ndarray | None = None, old_keys: np.ndarray | None = None,
    n_boot: int = 2000, seed: int = 20260814,
) -> dict[str, Any]:
    """配对 bootstrap:估计 Δ(new-old) 的分布与 noise level。

    v12(P0-2 修复):
    - noise 改为 Δ 分布的 **标准差**(bootstrap SE),旧口径 MAD≈0.674σ
      使 "Δ ≥ 1×noise" 的 accepted 门槛实际对应单侧 p≈0.25,是假阳性
      工厂——交付模型 trial-013 曾以 delta_ci=[-0.0024,+0.0057] 跨 0、
      p_positive=0.765 的结果拿到 accepted;
    - n_boot 默认 200→2000;
    - 附带 z_score(delta_mean/se)供收缩分选型与 BH 校正复用。
    回归指标下 score 列即预测值,delta 方向感知由 Evaluator 处理;
    uplift 指标连 treatment 一起重采样(同一批样本的两个模型必须
    面对同样的处置分组)。

    v19(req-t7-drawdown 实测事故回写):本函数此前假设 y/score_new/
    score_old 三者**行位置对齐**(同一 idx 同时切三个数组)——跨数据
    视图比较时行序不同(视图落盘各自天内洗牌,行序是输入帧序的
    函数),不同用户的分数被配对,产生 delta=+0.2844 的物理不可能
    结果(trial-009 实测)。防御:调用方传 keys(对齐键矩阵,各列
    逐一相等才可比)时先做 key-join 对齐;键不可对齐抛 ValueError——
    宁可不比,不可错比。同一视图内(行序天然一致)不传 keys 行为
    不变,历史调用零改动。
    """
    y_true = np.asarray(y_true)
    score_new = np.asarray(score_new, dtype=float)
    score_old = np.asarray(score_old, dtype=float)
    if keys is not None:
        keys = np.asarray(keys, dtype=object)
        if keys.ndim == 1:
            keys = keys.reshape(-1, 1)
        if old_keys is None:
            raise ValueError(
                "paired_bootstrap_delta:传入 keys 时必须同时传 old_keys")
        if len(keys) != len(y_true) or len(keys) != len(score_new):
            raise ValueError(
                "paired_bootstrap_delta: keys 行数与 y/score 不一致")
        # 只重排旧模型分数。y_true/score_new/treatment 均属于当前面，
        # 绝不能用旧侧位置重排 treatment（uplift 历史错位根因）。
        score_old = align_score_by_keys(score_old, keys, old_keys)
    treatment = None if treatment is None else np.asarray(treatment)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    fast_new = fast_old = None
    if _fast_bootstrap_auc(metric, treatment):
        fast_new = _BootstrapAUC(y_true, score_new)
        fast_old = _BootstrapAUC(y_true, score_old)
        if not (fast_new.ok and fast_old.ok):
            fast_new = fast_old = None
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if fast_new is not None:
            counts = np.bincount(idx, minlength=n)
            deltas[b] = fast_new.at(counts) - fast_old.at(counts)
            continue
        yt = y_true[idx]
        w = treatment[idx] if treatment is not None else None
        try:
            v_new = compute_metric(metric, yt, score_new[idx],
                                   score=score_new[idx], treatment=w)
            v_old = compute_metric(metric, yt, score_old[idx],
                                   score=score_old[idx], treatment=w)
            deltas[b] = v_new - v_old
        except Exception:  # noqa: BLE001
            deltas[b] = np.nan
    valid = deltas[np.isfinite(deltas)]
    if len(valid) < n_boot * 0.5:
        return {"noise": None, "delta_mean": float("nan"),
                "delta_ci": (float("nan"), float("nan")), "p_positive": None,
                "se": None, "z_score": None}
    se = float(np.std(valid))
    mean = float(np.mean(valid))
    return {
        # noise = bootstrap SE(σ);1×noise 门槛 ≈ 单侧 p≈0.16,
        # p_positive 是原始 Δ>0 概率；Evaluator 按指标方向转成 p_better，
        # 再由 Generation 做 FDR 校正。
        "noise": se if se > 0 else None,
        "delta_mean": mean,
        "delta_ci": (float(np.percentile(valid, 2.5)),
                     float(np.percentile(valid, 97.5))),
        "p_positive": float((valid > 0).mean()),
        "se": se,
        "z_score": (mean / se) if se > 0 else None,
    }


def benjamini_hochberg(p_values: list[float],
                       q: float = 0.10) -> list[bool]:
    """BH 多重比较校正(v12,P0-2):同代 k 个 trial 各自做"Δ>0"检验,
    取 k 个最差情形的显著性阈值族。返回与入参等长的布尔序列,
    True = 经 BH 校正后仍显著。

    p_positive 是"Δ>0 的后验概率",1-p_positive 即单侧 p 值。
    输入含 None(bootstrap 失败)按不显著处理。
    """
    k = len(p_values)
    if k == 0:
        return []
    idx = sorted(range(k), key=lambda i: (
        1.0 if p_values[i] is None else float(p_values[i])))
    ranked = [(i, p_values[i]) for i in idx]
    flags = [False] * k
    running = False
    # BH:第 i 小 p 值(1-based rank r)与 i·q/k 比较,从大到小扫,
    # 一旦命中其后全部判显著
    for r in range(k, 0, -1):
        i, p = ranked[r - 1]
        if p is None:
            continue
        if (1.0 - float(p)) >= 1.0 - (r / k) * q:  # p <= (r/k)·q
            running = True
        if running:
            flags[i] = True
    return flags
