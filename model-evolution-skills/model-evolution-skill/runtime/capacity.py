"""天花板诊断(v11):给 plan 的「加行 / 加列 / 调参」排硬序。

当前 stop 判据全是滞后指标(最近几代没涨);本模块提供三个前瞻仪表,
全部在 ≤20 万抽样面上分钟级完成:

1. 样本量学习曲线(sample_size_curve):25k/50k/100k/200k 各训一次。
   仍陡 → 加行(data_view / 样本定义)价值高;已平 → 只能加列。
2. 列数消融曲线(feature_count_curve):按 importance 累计 top-k 训练,
   找饱和点。已饱和 → 转模型/超参探索有硬依据。
3. 局部歧义探针(local_ambiguity):投影分桶内标签方差。它只描述当前
   表示与当前桶化下的局部不可分性，不是不依赖模型的“不可约误差下界”。

产物 capacity-report.json(sample 面证据,结论标注 scope)。

v12(P1-7)修复:
- 前缀采样改读 splits manifest 的 ordering(P0-1 配套):splits 已按
  time_column 排序 → 时间序前缀;未排序 → 分层随机抽样。旧行为固定
  取 ``iloc[:k]``,而落盘行序曾是 loader 原始序(train 前 63,854 行
  标签有序),12.5%/25% 两个点训练集全是同一标签,产出 AUC=0.5 的
  垃圾点与"已饱和,转 HPO"的伪诊断,误导 45% 算力;
- 每格点 3 seed 重复,差值给 95% CI(单次训练的 saturated 判定与掷
  硬币无异);
- 不可约噪声探针重做:旧实现随机投影 + round(·,1) + 精确字节哈希,
  连续特征上冲突率恒 0.0(探针形同虚设);改为分位桶化后按桶内
  标签方差估计噪声下界。

v2.17.2(C2):新增 ``target_reachability``(spec.performance_target vs 当前
最优实测 + 样本曲线外推上界 + 局部歧义指数 → reachable / unlikely /
not_reachable),capacity-report 落盘、planner-context 第 10 节与报告 §15b
呈现;读侧用最新 trial 重算(:func:`assess_target_reachability`)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .logging_utils import dump_json, get_logger
from .metrics import compute_metric, metric_direction
from .safe_dtype import safe_to_numeric

logger = get_logger(__name__)

SAMPLE_FRACTIONS = (0.125, 0.25, 0.5, 1.0)
FEATURE_GRID = (0.25, 0.5, 0.75, 1.0)
#: 每格点重复 seed 数(P1-7:单次训练的曲线点没有置信带可言)
N_SEEDS = 3
#: 近似重复探针的随机投影维度(降维后哈希找冲突对)
DUP_PROJ_DIM = 24


def _fit_score(X_tr: pd.DataFrame, y_tr: np.ndarray,
               X_va: pd.DataFrame, y_va: np.ndarray, metric: str,
               *, treatment_tr=None, treatment_va=None, seed: int = 20260814,
               n_jobs: int = 0) -> float:
    """诊断专用最小 LGBM(与 baseline 同默认参;失败抛异常)。"""
    import lightgbm as lgb

    nj = n_jobs or max(1, int(__import__("os").environ.get(
        "OMP_NUM_THREADS", "4") or 4))
    direction = metric_direction(metric)
    is_uplift = treatment_tr is not None
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
                  subsample=0.9, colsample_bytree=0.9, random_state=seed,
                  n_jobs=nj, verbosity=-1)
    from .metrics import UPLIFT_METRICS

    if metric in UPLIFT_METRICS or is_uplift:
        w_tr = np.asarray(treatment_tr)
        feats = [c for c in X_tr.columns]
        X_tr2 = X_tr.assign(__treatment__=w_tr)
        m_t = lgb.LGBMClassifier(**params).fit(
            X_tr2.loc[w_tr == 1, feats], y_tr[w_tr == 1])
        m_c = lgb.LGBMClassifier(**params).fit(
            X_tr2.loc[w_tr == 0, feats], y_tr[w_tr == 0])
        X_va_t = X_va.assign(__treatment__=np.ones(len(X_va), dtype="int8"))
        X_va_c = X_va.assign(__treatment__=np.zeros(len(X_va),
                                                    dtype="int8"))
        s = (m_t.predict_proba(X_va_t[feats])[:, 1]
             - m_c.predict_proba(X_va_c[feats])[:, 1])
        return compute_metric(metric, y_va, None, score=s,
                              treatment=np.asarray(treatment_va))
    if metric in {"rmse", "mae", "mape", "r2"}:
        m = lgb.LGBMRegressor(**params).fit(X_tr, y_tr)
        return compute_metric(metric, y_va, m.predict(X_va))
    m = lgb.LGBMClassifier(**params).fit(X_tr, y_tr)
    s = m.predict_proba(X_va)[:, 1]
    return compute_metric(metric, y_va, None, score=s)


def _prep(X: pd.DataFrame) -> pd.DataFrame:
    """诊断侧 dtype 治理:数值化 + 低基数 one-hot(与模板同口径)。

    v2.12.2.8:数值化改走 `safe_to_numeric`。旧实现用
    `pd.to_numeric(errors="coerce")` 再以 `notna().sum() >= 0.95*n`
    判"可数值化",在含非数字值的长 object 列上既会段错误(见
    safe_dtype 模块 docstring),又会把十六进制键静默解析成数字后
    **误判为可数值化**,把 ID 列放进诊断特征面。
    """
    X = X.copy()
    cats = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            continue
        conv = safe_to_numeric(X[c])
        if conv is not None:
            X[c] = conv.astype("float32")
        elif X[c].nunique() <= 100:
            cats.append(c)
    if cats:
        X = pd.get_dummies(X, columns=cats)
    return X.fillna(0).astype("float32")


def _prep_pair(X_tr: pd.DataFrame,
               X_va: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """两侧联合 dtype 治理 + **列对齐**(v2.11.6)。

    旧实现 `_prep(X_tr), _prep(X_va)` 各自独立 one-hot:train 出现的
    低基数类别档若在 validation 缺席(反之亦然),两侧列数不等,后续
    `LGBMClassifier.fit(X_tr)` → `predict_proba(X_va)` 直接抛
    `LightGBMError: The number of features in data (1499) is not the
    same as it was in training data (1500)` —— **容量诊断整条命令崩死**,
    且无 CLI 参数可绕(实测 drawdown-req-t7-v211:`prime_cust_mob6`
    的"当月新增"档 train 211 例、validation 面不存在,三仪表全废)。

    对齐口径与模板 `prepare_features` 一致:train 侧定列,validation
    按 train 列集 reindex 补 0(缺席档 = 该档计数 0,语义正确)。
    """
    tr = _prep(X_tr)
    va = _prep(X_va)
    missing = [c for c in tr.columns if c not in va.columns]
    extra = [c for c in va.columns if c not in tr.columns]
    if missing or extra:
        # 双向都补:validation 独有的档也保留(否则丢信息),但顺序
        # 以 train 为准,多余列追加在后 —— 两侧列集与列序必须一致。
        for c in missing:
            va[c] = np.float32(0)
        for c in extra:
            tr[c] = np.float32(0)
        va = va.reindex(columns=list(tr.columns))
    return tr, va


def sample_size_curve(X_tr: pd.DataFrame, y_tr: np.ndarray,
                      X_va: pd.DataFrame, y_va: np.ndarray, metric: str, *,
                      treatment_tr=None, treatment_va=None,
                      ordering: str | None = None,
                      prefix_rate_rel_dev: float | None = None,
                      n_seeds: int | None = None) -> dict:
    """样本量学习曲线:各比例子样本训练 → 外推增量趋势。

    v12(P1-7):
    - 子样本索引经 prefix_sample_safe(ordering):排序已确认 → 时间序
      前缀;否则分层随机(旧行为盲取 iloc[:k],前 63,854 行标签有序的
      train 上两个小格点全是同一标签,曲线是 AUC=0.5 的垃圾);
    - 每格点 N_SEEDS 个 seed,曲线点报 mean±std;末段增益是否为 0 的
      判定改为 95% CI 是否覆盖 0(单点无置信带,saturated 与掷硬币同)。

    v17.1:prefix_rate_rel_dev(manifest 的前缀标签偏差)透传给
    prefix_sample_safe——时间排序但天内标签聚集的数据(drawdown-t7
    实测 prefix dev 0.13~2.07)不再盲用时间前缀。
    """
    from .hpo import prefix_sample_safe

    n_seeds = int(N_SEEDS if n_seeds is None else n_seeds)
    n_seeds = max(1, n_seeds)
    n = len(X_tr)
    direction = metric_direction(metric)
    pts = []
    used_prefix = True
    for frac in SAMPLE_FRACTIONS:
        k = max(1000, int(n * frac))
        idx = prefix_sample_safe(n, k / n, ordering=ordering,
                                 seed=20260814 + int(frac * 100),
                                 prefix_rate_rel_dev=prefix_rate_rel_dev)
        if not (idx == np.arange(len(idx))).all():
            used_prefix = False
        vals = []
        for s in range(n_seeds):
            v = _fit_score(
                X_tr.iloc[idx], np.asarray(y_tr)[idx], X_va, y_va, metric,
                treatment_tr=None if treatment_tr is None
                else np.asarray(treatment_tr)[idx],
                treatment_va=treatment_va, seed=20260814 + s)
            vals.append(float(v))
        mu = float(np.mean(vals))
        pts.append({"rows": int(k), "fraction": frac, "metric": mu,
                    "metric_std": float(np.std(vals)),
                    "metric_values": vals})
    improved = (pts[-1]["metric"] - pts[-2]["metric"]) if direction \
        else (pts[-2]["metric"] - pts[-1]["metric"])
    first = pts[0]["metric"]
    total = (pts[-1]["metric"] - first) if direction else (first -
                                                           pts[-1]["metric"])
    last_step_gain_rel = improved / max(abs(first), 1e-9)
    # 末段增益的 95% CI:sqrt(2/n_seeds)·std(两独立格点均值差)。
    # n_seeds == 1 时 std 不可估,退化为"单点无置信带":ci_covers_zero
    # 置 True(不敢声称仍陡),与 v12 引入置信带之前的行为一致。
    if n_seeds < 2:
        se_diff = float("inf")
        ci_covers_zero = True
    else:
        se_diff = float(np.sqrt(2.0 / n_seeds) * max(
            pts[-1]["metric_std"], pts[-2]["metric_std"], 1e-12))
        ci_covers_zero = bool(improved - 1.96 * se_diff <= 0
                              <= improved + 1.96 * se_diff)
    return {"points": pts, "last_step_gain": float(improved),
            "last_step_gain_rel": float(last_step_gain_rel),
            "last_step_gain_se": se_diff,
            "last_step_ci_covers_zero": ci_covers_zero,
            "n_seeds": n_seeds,
            "total_gain_rel": float(total / max(abs(first), 1e-9)),
            "sampling": ("time_prefix" if used_prefix
                         else "stratified_random"),
            "prefix_rate_rel_dev": prefix_rate_rel_dev,
            # 斜率判读:末段增益显著为正(95% CI 不含 0)且 >10% 总增益
            # → 仍陡,加行优先
            "still_steep": bool(
                total > 0 and improved > 0.1 * total
                and not ci_covers_zero)}


def feature_count_curve(X_tr: pd.DataFrame, y_tr: np.ndarray,
                        X_va: pd.DataFrame, y_va: np.ndarray, metric: str,
                        importance: pd.DataFrame | None, *,
                        treatment_tr=None, treatment_va=None) -> dict:
    """列数消融曲线:importance 累计 top-k% 特征训练 → 饱和点。"""
    if importance is None or not len(importance):
        return {"skipped": True, "reason": "无 feature_importance.csv"}
    cols = list(importance["feature"])
    cols = [c for c in cols if c in X_tr.columns]
    if len(cols) < 20:
        return {"skipped": True, "reason": f"有效列过少({len(cols)})"}
    pts = []
    for frac in FEATURE_GRID:
        k = max(10, int(len(cols) * frac))
        use = cols[:k]
        v = _fit_score(X_tr[use], y_tr, X_va[use], y_va, metric,
                       treatment_tr=treatment_tr,
                       treatment_va=treatment_va)
        pts.append({"feature_frac": frac, "n_features": int(k),
                    "metric": float(v)})
    direction = metric_direction(metric)
    tail = (pts[-1]["metric"] - pts[-2]["metric"]) if direction \
        else (pts[-2]["metric"] - pts[-1]["metric"])
    # v2.11.6:饱和判据改为**单侧**(tail < 阈值),不再用 abs(tail)。
    # 旧口径下"尾部列增益为 **负**"(加列反而掉点,特征面已过峰)
    # 会被判成 saturated=False → verdict 打出"列数曲线未饱和,特征
    # 空间仍有增量"的**反向建议**(实测 drawdown-req-t7-v211:
    # 996→1328 列 tail_gain = −0.0027 < 0,工具仍在推荐加列)。
    # 尾部非显著正增益(含负值)一律视为饱和:继续加列无收益,
    # 负值额外提示按 gain 裁剪。
    _th = 0.002 * max(abs(pts[-1]["metric"]), 1e-9)
    return {"points": pts, "tail_gain": float(tail),
            # 尾部列(75%→100%)增益非显著为正 → 列空间已饱和
            "saturated": bool(tail < _th),
            "tail_gain_negative": bool(tail < 0)}


def irreducible_noise_probe(X: pd.DataFrame, y: np.ndarray) -> dict:
    """当前表示下的局部标签歧义启发式(v12/P1-7 重做)。

    旧实现:随机投影 + round(·,1) + 精确字节哈希找"特征相同、标签
    相反"的行——连续特征上哈希几乎恒不碰撞,实测冲突率恒 0.0,
    探针形同虚设但结论被当"硬依据"进 planner-context。

    新口径:随机投影降到 DUP_PROJ 维后按**行数自适应**分位桶化(每桶
    平均 ~20 行,而不是固定 4^24 个稀疏桶——每桶 1 行时桶内方差恒 0,
    探针再次失效)。桶内标签方差占全局方差的比例 = 近邻标签不可预测性
    的启发式比例,local_ambiguity_index ∈ [0,1]。保留 ``noise_bound``
    仅兼容旧报告，禁止把它解释成信息论或 Bayes error 下界。
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 200 or len(np.unique(y)) < 2:
        return {"local_ambiguity_index": None, "noise_bound": None,
                "n_buckets": 0,
                "note": "样本不足或标签退化"}
    Xn = _prep(X)
    rng = np.random.default_rng(20260814)
    proj = rng.standard_normal((Xn.shape[1], DUP_PROJ_DIM)) \
        / np.sqrt(max(Xn.shape[1], 1))
    Z = (Xn.to_numpy(dtype="float32") @ proj.astype("float32"))
    n = len(y)
    total_var = float(np.var(y))
    if total_var <= 0:
        return {"local_ambiguity_index": None, "noise_bound": None,
                "n_buckets": 0,
                "note": "标签方差为 0"}
    # 行数自适应桶数:目标总桶数 ≈ n/20(每桶 ~20 行才有组内统计)。
    # 组合桶数 = bins^dim,取 dim 使 bins^dim 落在目标附近:
    # bins=3 时 dim = round(log(n/20)/log(3));dim 至少 2。
    target_buckets = max(4, n // 20)
    n_bins = 3
    dim = max(2, int(round(np.log(target_buckets) / np.log(n_bins))))
    dim = min(dim, Z.shape[1])
    edges = np.quantile(
        Z[:, :dim], np.linspace(0, 1, n_bins + 1)[1:-1], axis=0)
    codes = np.zeros(n, dtype=np.int64)
    for j in range(dim):
        codes = codes * n_bins + np.digitize(Z[:, j], edges[:, j])
    df = pd.DataFrame({"key": codes, "y": y})
    grp = df.groupby("key")["y"]
    sizes = grp.size()
    means = grp.mean()
    within = df.merge(sizes.rename("n"), on="key")
    within_var = float(np.mean(
        (within["y"] - within["key"].map(means)) ** 2))
    noise_bound = float(np.clip(within_var / total_var, 0.0, 1.0))
    return {"local_ambiguity_index": noise_bound,
            "noise_bound": noise_bound,  # deprecated compatibility alias
            "n_buckets": int(len(sizes)),
            "bins_per_dim": int(n_bins),
            "dims_used": int(dim),
            "median_bucket_size": float(np.median(sizes)),
            "is_theoretical_lower_bound": False,
            "note": "分位桶化近邻标签方差占比；依赖投影/桶化/当前表示，"
                    "仅作方向启发，不是不可约误差下界"}


def verdict(sample_curve: dict, feature_curve: dict,
            irreducible: dict | None) -> dict:
    """三仪表 → 行动排序(加行 / 加列 / 模型超参)。

    v21:使用 local_ambiguity_index(桶内方差占比启发式);
    conflict_rate 键保留兼容旧报告读取。
    """
    advice: list[dict] = []
    sc = sample_curve or {}
    if sc.get("still_steep"):
        advice.append({"action": "add_rows",
                       "priority": 1,
                       "why": f"样本曲线仍陡(末段相对增益 "
                              f"{sc.get('last_step_gain_rel', 0):.4f},"
                              f"95% CI 不含 0);data_view/样本定义方向优先"})
    fc = feature_curve or {}
    if not fc.get("skipped") and not fc.get("saturated"):
        advice.append({"action": "add_columns", "priority": 1,
                       "why": "列数曲线未饱和,特征空间仍有增量"})
    if fc.get("tail_gain_negative"):
        # v2.11.6:尾部增益为负 = 加列有害,不是"还有空间"
        advice.append({"action": "prune_columns", "priority": 1,
                       "why": f"列数曲线尾部增益为**负**"
                              f"({fc.get('tail_gain', 0):+.4f}):"
                              "75%→100% 的尾部列加进去反而掉点,"
                              "特征面已过峰,应按 gain 裁剪而非继续加列"})
    ir = irreducible or {}
    nb = ir.get("local_ambiguity_index", ir.get("noise_bound"))
    if nb is not None and nb > 0.10:
        advice.append({"action": "new_information", "priority": 1,
                       "why": f"当前表示的局部歧义指数 {nb:.3f} 偏高;"
                              "优先试新信息/data_view，但该启发式不是"
                              "不可约误差证明"})
    if sc and not sc.get("still_steep"):
        advice.append({"action": "model_hpo", "priority": 2,
                       "why": "样本曲线已平(末段增益 CI 覆盖 0),"
                              "加行边际低;模型/超参方向(含 hpo trial)"
                              "性价比上升"})
    if fc and not fc.get("skipped") and fc.get("saturated"):
        advice.append({"action": "model_hpo", "priority": 2,
                       "why": "列数曲线已饱和,加列边际低;"
                              "转模型/超参探索有硬依据"})
    advice.sort(key=lambda a: a["priority"])
    if not advice:
        advice.append({"action": "model_hpo", "priority": 2,
                       "why": "各仪表均无强信号,按默认配比探索"})
    return {"action_ranking": advice,
            "scope": "sample 面(≤20 万抽样),结论进入全量决策前"
                     "需标注 scope"}


# ---------------------------------------------------------------------------
# v2.17.2(C2):目标可达性
# ---------------------------------------------------------------------------
#: 外推上界允许的"训练数据再翻倍"次数(固定样本源下最多还能拿到的数据量
#: 级;1 = 再多一倍数据)。曲线外推到无穷数据在 per-doubling 增益衰减慢
#: 时没有意义(上界发散),而固定样本源的任务实际可得数据是有限的。
TARGET_EXTRA_DOUBLINGS = 1
#: 局部歧义指数"偏高"阈值(与 verdict() 的 new_information 建议同口径)
_AMBIGUITY_HIGH = 0.10


def _performance_target(spec_data: dict | None, metric: str) -> float | None:
    """spec.performance_target → 主指标目标值({auc: 0.85} 或标量)。"""
    pt = (spec_data or {}).get("performance_target")
    val = pt.get(metric) if isinstance(pt, dict) else pt
    if isinstance(val, bool):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _best_observed(run_dir: Path, metric: str,
                   reference_trial: str | None = None
                   ) -> tuple[float | None, str | None, str | None]:
    """当前最优实测(validation 主指标):全部已评估 trial 中最优者,
    无则回落 reference trial 的 evaluation.json;返回 (值, 来源, trial_id)。

    口径:integrity 有效、verdict 非 invalid、非 screening-only、非抽样面
    (data_view=sample 的分数不与全量可比)。
    """
    from .logging_utils import load_json as _lj
    from .logging_utils import load_yaml as _ly

    hib = metric_direction(metric)
    best, best_tid = None, None
    tdir = Path(run_dir) / "trials"
    if tdir.is_dir():
        for d in sorted(tdir.iterdir()):
            p = d / "trial.yaml"
            if not p.exists():
                continue
            try:
                data = _ly(p) or {}
            except Exception:  # noqa: BLE001
                continue
            ev = data.get("evaluation") if isinstance(data, dict) else None
            if not isinstance(ev, dict) or not ev:
                continue
            if ev.get("verdict") == "invalid" or \
                    (ev.get("integrity") or {}).get("valid", True) is False:
                continue
            if data.get("screen_only") or \
                    str(data.get("data_view") or "v1") == "sample":
                continue
            pm = ev.get("primary_metric") or {}
            if pm.get("name") not in (None, metric):
                continue
            try:
                v = float(pm.get("value"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            if best is None or (v > best if hib else v < best):
                best, best_tid = v, str(data.get("trial_id") or d.name)
    if best is not None:
        return best, "best_trial", best_tid
    if reference_trial:
        p = Path(run_dir) / "trials" / str(reference_trial) / \
            "evaluation.json"
        if p.exists():
            try:
                v = float(((_lj(p) or {}).get("primary_metric") or {})
                          .get("value"))
                if np.isfinite(v):
                    return v, "reference_trial", str(reference_trial)
            except (TypeError, ValueError):
                pass
    return None, None, None


def target_reachability(sample_curve: dict | None,
                        feature_curve: dict | None,
                        irreducible: dict | None, *,
                        target: float | None, metric: str,
                        best_observed: float | None,
                        best_source: str | None = None,
                        best_trial: str | None = None,
                        extra_doublings: int = TARGET_EXTRA_DOUBLINGS
                        ) -> dict:
    """v2.17.2(C2):业务目标在当前数据/表示下是否可达(诊断,不参与放行)。

    事故(run4 req-t7-400w):spec 写了 auc 0.85,capacity-check 只给行动
    排序,整个 run 没有任何产物回答"0.85 是否可达";冠军 0.812 时缺口
    0.038(≈ 40×MDE),仍按常规节奏续投特征/HPO,直到人工复盘才写出
    "0.85 不可达"的经验卡。

    外推口径(学习曲线凹性:每翻倍一次数据的增益不增):
    - 点估计:再翻倍 D 次的增益 ≤ max(0, 末段增益)·D;
    - 上界:末段增益取 95% 上界(+1.96·SE;SE 不可估时取点估计 ×2);
    - ``extrapolated_ceiling`` = best_observed ± 上界 headroom(方向感知)。
    末段增益来自抽样面诊断模型,全量冠军已在更大数据上,其下一次翻倍的
    增益只会更小 —— 因此是**偏宽**的上界。
    判定:缺口 ≤ 0 或 ≤ 点估计 headroom → reachable;≤ 上界 → unlikely;
    超上界 → not_reachable(列数曲线未饱和时降为 unlikely:行外推不覆盖
    新列/新信息的增量)。``local_ambiguity_index`` 偏高写进理由(启发式,
    不是误差下界)。
    """
    hib = metric_direction(metric)
    ir = irreducible or {}
    lai = ir.get("local_ambiguity_index", ir.get("noise_bound"))
    out: dict[str, Any] = {
        "target": target, "metric": metric,
        "best_observed": best_observed,
        "best_observed_source": best_source,
        "best_observed_trial": best_trial,
        "extrapolated_ceiling": None, "extrapolated_ceiling_point": None,
        "extra_doublings": int(extra_doublings),
        "local_ambiguity_index": lai,
        "verdict": None, "rationale": "",
        "scope": "抽样面学习曲线外推 + 当前最优实测;诊断参考,不参与放行"}
    if target is None:
        out["skipped"] = True
        out["rationale"] = f"spec 未声明 performance_target.{metric}"
        return out
    sc = sample_curve or {}
    pts = sc.get("points") or []
    if best_observed is None and pts:
        best_observed = float(pts[-1].get("metric"))
        out.update(best_observed=best_observed,
                   best_observed_source="sample_size_curve")
    if best_observed is None:
        out["skipped"] = True
        out["rationale"] = "无可比实测(无已评估 trial、reference trial 与样本曲线)"
        return out
    sign = 1.0 if hib else -1.0
    shortfall = (float(target) - float(best_observed)) * sign
    g_last = sc.get("last_step_gain")
    se = sc.get("last_step_gain_se")
    g_point = max(0.0, float(g_last)) if isinstance(
        g_last, (int, float)) and np.isfinite(g_last) else 0.0
    if isinstance(se, (int, float)) and np.isfinite(se):
        g_up = g_point + 1.96 * float(se)
    else:
        g_up = 2.0 * g_point
    D = max(0, int(extra_doublings))
    head_point, head_up = g_point * D, g_up * D
    out.update(shortfall=round(shortfall, 8),
               headroom_point=round(head_point, 8),
               headroom_upper=round(head_up, 8),
               extrapolated_ceiling_point=round(
                   float(best_observed) + sign * head_point, 8),
               extrapolated_ceiling=round(
                   float(best_observed) + sign * head_up, 8))
    fc = feature_curve or {}
    cols_open = bool(fc) and not fc.get("skipped") and not fc.get("saturated")
    _src = {"best_trial": f"最优 trial {best_trial}",
            "reference_trial": f"reference trial {best_trial}",
            "sample_size_curve": "抽样面曲线末点"}.get(
        out["best_observed_source"] or "", "实测")
    head = (f"目标 {metric}={target:g},当前最优 {best_observed:.6g}"
            f"({_src}),缺口 {shortfall:+.6g};样本曲线末段增益 "
            f"{g_point:.4g}/翻倍(上界 {g_up:.4g}),再翻倍 {D} 次的外推上界 "
            f"{out['extrapolated_ceiling']:.6g}")
    if shortfall <= 0:
        verdict_, why = "reachable", "当前最优已达标"
    elif shortfall <= head_point:
        verdict_, why = "reachable", "缺口在样本曲线点估计 headroom 内(加行可兑现)"
    elif shortfall <= head_up:
        verdict_, why = "unlikely", "缺口超出点估计、仅落在外推上界内"
    elif cols_open:
        verdict_, why = ("unlikely", "缺口超出行外推上界;列数曲线未饱和,"
                         "新列/新信息仍可能贡献,行外推不覆盖这部分")
    else:
        verdict_, why = ("not_reachable", "缺口超出行外推上界,且列数曲线已"
                         "饱和:当前数据与表示下不可达,需新信息源/新样本定义"
                         "或调整目标")
    if isinstance(lai, (int, float)) and lai > _AMBIGUITY_HIGH \
            and shortfall > 0:
        why += (f";局部歧义指数 {lai:.3f} 偏高(当前表示下近邻标签分歧大,"
                "启发式,非误差下界)")
    out["verdict"] = verdict_
    out["rationale"] = f"{head} → {verdict_}:{why}"
    return out


def assess_target_reachability(run_dir: Path, report: dict | None,
                               spec=None) -> dict:
    """capacity-report + 当前 trial 实测 → target_reachability(读侧可重算)。

    best_observed 随搜索推进而变,capacity-report 可能是早期缓存;
    planner-context / report 调用本函数用最新 trial 重算,失败时调用方
    回落报告里落盘的旧值。
    """
    report = report or {}
    if spec is None:
        from .spec import TaskSpec

        spec = TaskSpec.load(Path(run_dir) / "task-spec.yaml")
    metric = str(report.get("metric") or spec.primary_metric)
    target = _performance_target(getattr(spec, "data", None) or {}, metric)
    best, src, tid = _best_observed(run_dir, metric,
                                    report.get("reference_trial"))
    return target_reachability(
        report.get("sample_size_curve"), report.get("feature_count_curve"),
        report.get("irreducible_noise"), target=target, metric=metric,
        best_observed=best, best_source=src, best_trial=tid)


def _splits_fingerprint(splits: Path) -> str:
    """抽样面内容指纹(v2.12.2.9):manifest + 四平面文件大小/mtime。

    用于 capacity-report 的缓存命中判定。口径取"manifest 内容 + 各
    parquet 的 (size, mtime)"——比逐文件 sha256 便宜,又足以识别
    "抽样面被重新材料化"。manifest 缺失时返回空串(调用方不缓存)。
    """
    import hashlib
    mf = splits / "manifest.json"
    if not mf.exists():
        return ""
    h = hashlib.sha256()
    try:
        h.update(mf.read_bytes())
        for p in sorted(splits.rglob("*.parquet")):
            st = p.stat()
            h.update(f"{p.relative_to(splits)}|{st.st_size}|{int(st.st_mtime)}"
                     .encode("utf-8"))
    except Exception:  # noqa: BLE001 — 指纹失败即视为不可缓存
        return ""
    return h.hexdigest()[:16]


def run_capacity_check(run_dir: Path, *, splits_subdir: str =
                       "sample-profile/splits-sample",
                       reference_trial: str | None = None,
                       n_seeds: int | None = None,
                       use_cache: bool = True) -> dict:
    """CLI 入口:读抽样面 splits(+可选 trial 的 importance)→ 全套诊断。

    v2.12.2.9:两处降本 + 一处缓存。
    - ``n_seeds``:样本量曲线每格点的重复数(None = 模块默认 N_SEEDS)。
      实测该曲线是整条命令的主要成本(4 格点 × 3 seed = 12 次拟合,
      16 万行抽样面约 9 分钟),而 std 只服务于"末段增益是否覆盖 0"
      这一个判定,降到 2 已足够。
    - ``use_cache``:抽样面 manifest + 文件指纹未变且
      ``capacity-report.json`` 已存在时直接复用,跳过全部拟合。**判定
      逻辑与产物结构不变**;缓存键不匹配(重新材料化了抽样面)则重算。

    缓存与 n_seeds 都只影响"算得多细",不影响 verdict 的语义口径。
    """
    from .spec import TaskSpec
    from .logging_utils import load_json as _load_json

    spec = TaskSpec.load(Path(run_dir) / "task-spec.yaml")
    splits = Path(run_dir) / splits_subdir
    if not (splits / "train.parquet").exists():
        # [FIX-2026-09-18] baseline 不会自动材料化抽样面,而
        # capacity-check 硬依赖它 —— 旧报错只写"先执行 write-splits-sample",
        # 操作员还得自己拼参数。这里直接给出可复制执行的完整命令。
        _cli = Path(__file__).resolve().parents[1] / "scripts" / "modelevo.py"
        raise SystemExit(
            f"缺少抽样面 splits: {splits}\n"
            f"capacity-check 需要 ≤max_sample_rows 的抽样面(baseline 不"
            f"会自动生成)。请先执行:\n"
            f"  python {_cli} write-splits-sample --run {run_dir}\n"
            f"说明:抽样面按比例等距抽样、固定 seed,final_test 不抽样"
            f"不落盘;产物落在 {splits_subdir}/。")

    report_path = Path(run_dir) / "sample-profile" / "capacity-report.json"
    fingerprint = _splits_fingerprint(splits)
    if use_cache and fingerprint and report_path.exists():
        try:
            cached = _load_json(report_path) or {}
            if cached.get("splits_fingerprint") == fingerprint:
                logger.info(
                    "capacity-check 复用缓存(抽样面指纹 %s 未变);"
                    "需强制重算时删 sample-profile/capacity-report.json",
                    fingerprint)
                # v2.17.2(C2):目标可达性随最新 trial 实测重算(不重拟合)
                try:
                    _tr = assess_target_reachability(run_dir, cached, spec)
                    if _tr != cached.get("target_reachability"):
                        cached["target_reachability"] = _tr
                        dump_json(report_path, cached)
                except Exception as _exc:  # noqa: BLE001
                    logger.warning("目标可达性重算失败(沿用缓存): %s", _exc)
                return cached
        except Exception:  # noqa: BLE001 — 缓存不可读则正常重算
            pass

    tr = pd.read_parquet(splits / "train.parquet")
    va = pd.read_parquet(splits / "validation.parquet")
    # v21 四平面:标签在 splits/labels/(load_labels),不在特征面
    labels_dir = splits / "labels"
    if (labels_dir / "train.parquet").exists():
        tr_lab = pd.read_parquet(labels_dir / "train.parquet")
        va_lab = pd.read_parquet(labels_dir / "validation.parquet")
        tr = tr.join(tr_lab.set_index(tr.index), how="left",
                     rsuffix="_lab")
        va = va.join(va_lab.set_index(va.index), how="left",
                     rsuffix="_lab")
    from .data import resolve_feature_columns

    feats = resolve_feature_columns(tr, spec)
    X_tr, X_va = _prep_pair(tr[feats], va[feats])
    y_tr = tr[spec.target_name].values
    y_va = va[spec.target_name].values
    w_tr = w_va = None
    if spec.treatment_column:
        w_tr, w_va = tr[spec.treatment_column].values, \
            va[spec.treatment_column].values

    importance = None
    if reference_trial:
        ipath = Path(run_dir) / "trials" / reference_trial / "feature" \
            / "feature_importance.csv"
        if ipath.exists():
            importance = pd.read_csv(ipath)
    else:
        # v16/P0-2c:自动发现 reference trial——列数消融曲线依赖
        # feature_importance.csv,不传 --reference-trial 时它静默
        # skipped(实测 120w run:三仪表废一个)。按 gain 降序取最近
        # 一个含 importance 的已评估 trial 兜底,找不到才放弃。
        try:
            from .logging_utils import load_json as _lj

            _trials_dir = Path(run_dir) / "trials"
            _cands: list[tuple[str, float]] = []
            if _trials_dir.is_dir():
                for td in _trials_dir.iterdir():
                    ip = td / "feature" / "feature_importance.csv"
                    tp = td / "trial.yaml"
                    if not (ip.exists() and tp.exists()):
                        continue
                    try:
                        _meta = _lj(td / "evaluation.json") \
                            if (td / "evaluation.json").exists() else {}
                        if not _meta:
                            continue
                        _mtime = ip.stat().st_mtime
                        _cands.append((td.name, _mtime))
                    except Exception:  # noqa: BLE001
                        continue
            if _cands:
                _cands.sort(key=lambda x: -x[1])
                reference_trial = _cands[0][0]
                ipath = _trials_dir / reference_trial / "feature" \
                    / "feature_importance.csv"
                importance = pd.read_csv(ipath)
                logger.info("capacity-check 自动 reference trial: %s "
                            "(最新 feature_importance.csv)",
                            reference_trial)
        except Exception as _exc:  # noqa: BLE001
            logger.warning("reference trial 自动发现失败: %s", _exc)

    # P1-7/P0-1:前缀采样安全性由 manifest 的 ordering 决定;
    # v17.1:同时读 prefix_rate_rel_dev——时间排序但前缀标签偏差
    # 超阈(天内标签聚集,drawdown-t7 实测 dev 0.13~2.07)时
    # prefix_sample_safe 降级分层随机,样本量曲线不再虚高。
    ordering = None
    prefix_dev = None
    mf_path = splits / "manifest.json"
    if mf_path.exists():
        try:
            from .logging_utils import load_json

            _tr_entry = ((load_json(mf_path).get("splits") or {})
                         .get("train") or {})
            ordering = _tr_entry.get("ordering")
            _dev = _tr_entry.get("prefix_rate_rel_dev")
            prefix_dev = float(_dev) if _dev is not None else None
        except Exception:  # noqa: BLE001
            ordering = None

    metric = spec.primary_metric
    sc = sample_size_curve(X_tr, y_tr, X_va, y_va, metric,
                           treatment_tr=w_tr, treatment_va=w_va,
                           ordering=ordering,
                           prefix_rate_rel_dev=prefix_dev,
                           n_seeds=n_seeds)
    fc = feature_count_curve(X_tr, y_tr, X_va, y_va, metric, importance,
                             treatment_tr=w_tr, treatment_va=w_va)
    ir = irreducible_noise_probe(tr[feats], y_tr)
    out = {"sample_size_curve": sc, "feature_count_curve": fc,
           "irreducible_noise": ir,
           "verdict": verdict(sc, fc, ir),
           "metric": metric, "splits": splits_subdir,
           "reference_trial": reference_trial,
           # v2.12.2.9:缓存键(抽样面指纹)。指纹变了说明抽样面被重新
           # 材料化,旧结论作废——上层据此判定缓存是否可复用。
           "splits_fingerprint": fingerprint}
    # v2.17.2(C2):目标可达性(spec.performance_target vs 最优实测 + 样本
    # 曲线外推上界);失败不阻断容量诊断
    try:
        out["target_reachability"] = assess_target_reachability(
            run_dir, out, spec)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("目标可达性判定失败(跳过): %s", _exc)
    dump_json(Path(run_dir) / "sample-profile" / "capacity-report.json",
              out)
    return out
