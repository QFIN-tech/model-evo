"""Fresh confirmation 判据与交付位绑定(v2.14.0)。

事故来源:req-t7-rand400w run2(2026-09-30)。数值最优的 trial-022(冠军
trial-019 配方的 confirm,val 0.807798)被判 rejected,交付却落在三代前的
trial-011(trial-006 配方,val 0.806054),差 1.7×MDE,且唯一一次
final_test gate 被花在它身上。四层缺陷叠加:

1. **确认门问错了问题**。旧判据 `|Δ vs parent| ≤ 2·SE` 是双侧等价,SE
   只含评测抽样噪声、不含 seed 方差。该 run 实测 seed sd≈1.3e-4(val)/
   1.6e-4(shadow),两 seed 间 Δ 的 sd 与门宽同阶 ⇒ 一个完全可复现的
   配方单次过门率只有 35~45%;trial-022 因 val **比父代好** +1.78e-4 >
   1.68e-4 被判失败,trial-011 在 shadow 上显著**更差**却通过。
2. **确认失败 = rejected**:合法、完整、最优的模型从所有候选池消失。
3. **交付池不绑定现任冠军**:`best_trial(delivery=True)` 取「任意曾过门的
   trial」,冠军换了两次,gen-03 对 trial-006 的确认仍然有效。
4. **不可逆 gate 前没有分叉检查**。

本模块集中新判据(纯函数,runner/evaluator/leaderboard/CLI 共用):

- 非劣(下侧,交付安全):Δ_dir ≥ −margin_lower
- 复现保真(上侧,防「借 confirm 之名换配方」):Δ_dir ≤ margin_upper
- margin = z · sqrt(SE_eval² + σ_seed²·(1 + 1/n_child))
  —— 父代单 seed、子代 n_child 个 seed 取均值,两侧 seed 效应独立。
- σ_seed 来源优先级:本 trial multi-seed(≥3 个 seed)→ 本 run 同配方
  复现组的合并方差(df≥2,**排除当前 trial**,否则 σ̂=|Δ|/√2 自证通过)
  → 配置先验 → 下限 floor_ratio×SE。

交付位绑定:配方 = 沿 confirm 边回溯到的非 confirm 根(confirm 的语义是
"同配方换 seed 复现")。交付只接受**绑定搜索冠军配方**的确认;陈旧确认
仅当与搜索冠军差距 ≤ 容差(默认 MDE,且不低于 2×noise)时可用,否则
判 diverged ⇒ 交付位空缺,走未确认路径并要求人工裁决。
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .trial import trial_currently_failed

CONFIRM_CRITERION = "noninferiority_seed_aware_v1"


def _finite_pos(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x)) and float(x) > 0
    except (TypeError, ValueError):
        return False


def _ev(trial) -> dict:
    return trial.data.get("evaluation") or {}


def _metric_value(trial) -> float | None:
    v = (_ev(trial).get("primary_metric") or {}).get("value")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def _mde(config, *, run_dir=None, baseline_metric=None) -> float:
    """v2.17.0:MDE 统一走 runtime.mde.effective_mde(量纲感知)。

    旧 run 配置无 ``minimum_meaningful_improvement_relative`` 键 ⇒ 结果
    与直接读绝对值完全一致;拿不到 run_dir/baseline 时同样退回绝对值。
    """
    if config is None:
        return 0.001
    from .mde import effective_mde

    try:
        return float(effective_mde(config, run_dir=run_dir,
                                   baseline_metric=baseline_metric))
    except Exception:  # noqa: BLE001 —— 解析失败退回绝对值(旧口径)
        return float(config.get(
            "evaluator.minimum_meaningful_improvement", 0.001) or 0.001)


def confirm_config(config, *, run_dir=None,
                   baseline_metric: float | None = None) -> dict:
    """确认门参数(缺省值与 config.DEFAULTS 同步)。

    v2.17.0:``mde`` 改为 :func:`runtime.mde.effective_mde`(传 run_dir 或
    baseline_metric 时按量纲感知口径;都不传时等于绝对值,行为不变)。
    """
    def _get(key, default):
        if config is None:
            return default
        v = config.get(key, default)
        return default if v is None else v

    prior = None if config is None else config.get(
        "evaluator.confirm_seed_sd_prior", None)
    prior_ratio = None if config is None else config.get(
        "evaluator.confirm_seed_sd_prior_ratio", None)
    # "scaled" = 按 n_val 复算比值(runtime.confirmation.scaled_sigma_prior);
    # 数值 = 固定比值;其余 = 关闭
    ratio_mode = None
    if isinstance(prior_ratio, str):
        if prior_ratio.strip().lower() == "scaled":
            ratio_mode = "scaled"
    elif _finite_pos(prior_ratio):
        ratio_mode = "fixed"
    return {
        "z_lower": float(_get("evaluator.confirm_margin_z", 2.0)),
        "z_upper": float(_get("evaluator.confirm_fidelity_z", 4.0)),
        "floor_ratio": float(_get("evaluator.confirm_seed_sd_floor_ratio",
                                  1.0)),
        "prior": float(prior) if _finite_pos(prior) else None,
        # v2.19.1:比值先验(σ_seed = ratio×SE);新 run 由
        # V2_19_1_RUN_CONFIG 写入 scaled 模式,见 scaled_sigma_prior
        "prior_ratio": (float(prior_ratio)
                        if ratio_mode == "fixed" else None),
        "prior_ratio_mode": ratio_mode,
        "cfg_margin": float(_get("evaluator.confirmation_margin", 0.0) or 0.0),
        "reference_check": bool(_get("evaluator.confirm_reference_check",
                                     True)),
        "stability_mde_ratio": float(_get(
            "evaluator.confirm_stability_mde_ratio", 0.5)),
        "mde": _mde(config, run_dir=run_dir, baseline_metric=baseline_metric),
    }


# ---------------------------------------------------------------------------
# σ_seed 与 margin
# ---------------------------------------------------------------------------

#: v2.19.1:σ_seed 的规模先验 ``σ_seed = r(n_val)·SE``,
#: ``r = clip(base·(n_val/ref_n)^exponent, lo, hi)``。
#:
#: 经验依据(170 个真实 trial 的落盘指标 + 5 seed 参照):σ_seed/SE 的比值随
#: validation 行数单调上升 —— n_val≈5 万时中位 0.42、P90 0.69;req-t7
#: (n_val 44.7 万)实测 1.19;百万级以上继续放大。旧口径的
#: ``floor_ratio×SE = 1×SE`` 把小于 1 的真实比值一律抬到 1,在小数据上过保守
#: (零效应误拒率 2.9%),又完全没有刻画大数据上比值 >1 的部分(400 万级
#: 误放率 11.9%)。这组常数拟合上述观测(5 万→0.50、44.7 万→1.20)。
SIGMA_PRIOR_BASE = 0.5
SIGMA_PRIOR_REF_N = 50000.0
SIGMA_PRIOR_EXPONENT = 0.4
SIGMA_PRIOR_LO = 0.5
SIGMA_PRIOR_HI = 1.5


def scaled_sigma_prior(se: float | None, n_val: Any, *,
                       base: float = SIGMA_PRIOR_BASE,
                       ref_n: float = SIGMA_PRIOR_REF_N,
                       exponent: float = SIGMA_PRIOR_EXPONENT,
                       lo: float = SIGMA_PRIOR_LO,
                       hi: float = SIGMA_PRIOR_HI) -> float | None:
    """按 validation 行数给出 **σ_seed/SE 比值** 先验 ``r(n_val)``。

    返回比值而不是绝对值 ``r·SE``:调用方(resolve_sigma_seed 的
    ``prior_ratio``)会再乘该次检验自己的 SE —— 与经验库历史比值
    (library 的 seed_variance_ratio)同一口径,两处可互相替代。
    条件不满足(无 SE / 无 n_val)时返回 None。
    """
    if not _finite_pos(se):
        return None
    try:
        n = float(n_val or 0)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(n) or n <= 0:
        return None
    r = float(base) * (n / float(ref_n)) ** float(exponent)
    return min(max(r, float(lo)), float(hi))


def resolve_sigma_seed(*, seed_metrics: Iterable[float] | None = None,
                       replicate_sd: float | None = None,
                       replicate_df: int = 0,
                       prior: float | None = None,
                       se: float | None = None,
                       floor_ratio: float = 1.0,
                       prior_ratio: float | None = None,
                       floor_measured: bool = True) -> dict:
    """按优先级解析 seed 标准差。

    优先级:本 trial multi-seed(≥3)→ 同配方复现组(df≥2)→ 先验 → floor。

    先验有两种形态:
    - ``prior_ratio``(v2.19.1,新 run):**比值**先验,``prior_ratio×SE``,
      由 :func:`scaled_sigma_prior` 按 n_val 算出,或取自经验库同数据底座的
      历史比值。它排在复现组之后、绝对先验之前。
    - ``prior``(旧):绝对值常数(``evaluator.confirm_seed_sd_prior``)。

    ``floor_measured=False``(新 run)时,floor 只用于"完全没有估计"的情形,
    不再把**实测**的小比值抬到 1×SE —— 小数据上实测中位 0.42,抬到 1 会让
    无效应差异被系统性拦下。默认 True 保持旧口径(confirm 判据用它)。
    """
    est, src, df = None, None, 0
    measured = False
    vals = [float(v) for v in (seed_metrics or [])
            if v is not None and math.isfinite(float(v))]
    if len(vals) >= 3:
        est, src, df = float(np.std(vals, ddof=1)), "multi_seed", len(vals) - 1
        measured = True
    elif _finite_pos(replicate_sd) and int(replicate_df or 0) >= 2:
        est, src, df = float(replicate_sd), "run_replicates", int(replicate_df)
        measured = True
    elif _finite_pos(prior_ratio) and _finite_pos(se):
        est, src = float(prior_ratio) * float(se), "scaled_prior"
    elif _finite_pos(prior):
        est, src = float(prior), "config_prior"
    floor = float(floor_ratio) * float(se) \
        if _finite_pos(se) and float(floor_ratio) > 0 else 0.0
    if not floor_measured and measured:
        floor = 0.0
    out = {"estimate": est, "estimate_source": src, "df": df,
           "floor": floor, "measured": measured}
    if est is None or not math.isfinite(est):
        out.update(sigma_seed=floor if floor > 0 else None,
                   source="floor_se_ratio" if floor > 0 else "unavailable")
    elif est < floor:
        out.update(sigma_seed=floor, source=f"{src}+floor")
    else:
        out.update(sigma_seed=est, source=src)
    return out


def sigma_for_split(sigma_info: dict | None, se: float | None,
                    floor_ratio: float) -> dict:
    """同一 σ̂ 估计换到另一个 split(如 shadow)时按该 split 的 SE 重算 floor。"""
    info = sigma_info or {}
    est = info.get("estimate")
    src = info.get("estimate_source")
    floor = float(floor_ratio) * float(se) \
        if _finite_pos(se) and float(floor_ratio) > 0 else 0.0
    out = {"estimate": est, "estimate_source": src,
           "df": info.get("df", 0), "floor": floor}
    if not _finite_pos(est):
        out.update(sigma_seed=floor if floor > 0 else None,
                   source="floor_se_ratio" if floor > 0 else "unavailable")
    elif float(est) < floor:
        out.update(sigma_seed=floor, source=f"{src}+floor")
    else:
        out.update(sigma_seed=float(est), source=src)
    return out


def confirmation_margins(*, se: float | None, sigma_seed: float | None,
                         n_child: int = 1, z_lower: float = 2.0,
                         z_upper: float = 4.0,
                         cfg_margin: float = 0.0) -> dict | None:
    """Δ(child − parent) 的判定带;SE/σ/cfg 全缺时返回 None(无从判定)。"""
    se_v = float(se) if _finite_pos(se) else 0.0
    sig = float(sigma_seed) if _finite_pos(sigma_seed) else 0.0
    cfg_margin = float(cfg_margin or 0.0)
    if se_v <= 0 and sig <= 0 and cfg_margin <= 0:
        return None
    n = max(1, int(n_child or 1))
    sd = math.sqrt(se_v ** 2 + sig ** 2 * (1.0 + 1.0 / n))
    return {"sd_delta": sd, "n_child": n,
            "z_lower": float(z_lower), "z_upper": float(z_upper),
            "lower": max(cfg_margin, float(z_lower) * sd),
            "upper": max(cfg_margin, float(z_upper) * sd)}


def judge_delta(delta: float, margins: dict, higher_better: bool) -> dict:
    """方向感知的非劣 + 保真判定。"""
    d = float(delta) if higher_better else -float(delta)
    non_inferior = d >= -float(margins["lower"])
    fidelity_ok = d <= float(margins["upper"])
    return {"delta": float(delta), "delta_directional": d,
            "non_inferior": bool(non_inferior),
            "fidelity_ok": bool(fidelity_ok),
            "passed": bool(non_inferior and fidelity_ok)}


def legacy_two_sided(delta: float | None, se: float | None,
                     cfg_margin: float = 0.0) -> bool | None:
    """旧判据 |Δ| ≤ max(cfg, 2·SE),仅作诊断留痕(不参与裁决)。"""
    if delta is None:
        return None
    m = max(float(cfg_margin or 0.0), 2.0 * float(se) if _finite_pos(se)
            else 0.0)
    if m <= 0:
        return abs(float(delta)) < 1e-12
    return abs(float(delta)) <= m


# ---------------------------------------------------------------------------
# 配方身份与复现组
# ---------------------------------------------------------------------------

def recipe_root(trial, by_id: dict) -> str:
    """配方根:沿 confirm 边回溯到第一个非 confirm trial。

    confirm = "同配方换 seed 复现"(new-trial --action confirm 复制父代
    代码),因此 confirm 子代与父代属同一配方;交付位与搜索冠军按配方
    而不是按 trial 对齐。
    """
    seen: set[str] = set()
    cur = trial
    while str(cur.data.get("search_action") or "") == "confirm" \
            and len(cur.parent_ids) == 1:
        pid = cur.parent_ids[0]
        if pid in seen:
            break
        seen.add(pid)
        nxt = by_id.get(pid)
        if nxt is None:
            return pid
        cur = nxt
    return cur.trial_id


def pooled_replicate_sd(trials: list, *, exclude_ids: Iterable[str] = (),
                        data_view: str | None = None) -> dict:
    """同配方复现组(配方根 + 其 confirm 子代)的合并 seed 方差。

    每个 confirm 子代的 seed 由 new-trial 强制不同于父代,组内差异即 seed
    效应(同一 validation 行集,不含评测抽样噪声)。**调用方必须排除当前
    待判 trial**:它与父代的差正是被检验的量,纳入即自证通过。
    """
    exclude = set(exclude_ids or ())
    by_id = {t.trial_id: t for t in trials}
    groups: dict[str, list[float]] = {}
    for t in trials:
        if t.trial_id in exclude:
            continue
        ev = _ev(t)
        v = _metric_value(t)
        if v is None or ev.get("verdict") == "invalid" \
                or (ev.get("integrity") or {}).get("valid", True) is False \
                or trial_currently_failed(t):
            continue
        if data_view is not None and \
                str(t.data.get("data_view") or "v1") != str(data_view):
            continue
        groups.setdefault(recipe_root(t, by_id), []).append(v)
    ss, df, used = 0.0, 0, {}
    for root, vals in groups.items():
        if len(vals) < 2:
            continue
        arr = np.asarray(vals, dtype=float)
        ss += float(((arr - arr.mean()) ** 2).sum())
        df += len(arr) - 1
        used[root] = len(arr)
    return {"sd": math.sqrt(ss / df) if df > 0 else None, "df": df,
            "groups": used}


def confirm_attempts(trials: list, parent_id: str) -> list:
    """对某父代(配方)已**实际完成评估**的 confirm 尝试。

    invalid/运行失败不计(基础设施问题不应消耗确认预算)。
    """
    out = []
    for t in trials:
        if str(t.data.get("search_action") or "") != "confirm":
            continue
        if parent_id not in t.parent_ids:
            continue
        v = _ev(t).get("verdict")
        if v in (None, "invalid") or trial_currently_failed(t):
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# 交付位
# ---------------------------------------------------------------------------

def delivery_tolerance(config, noise: float | None, *, run_dir=None,
                       baseline_metric: float | None = None) -> float:
    """陈旧确认可替代搜索冠军的最大差距:默认 MDE,且不低于 2×noise。

    v2.17.0:MDE 走 effective_mde(量纲感知;旧 run 无 relative 键时不变)。
    """
    tol = None if config is None else config.get(
        "evaluator.delivery_divergence_tolerance", None)
    if tol is None:
        tol = _mde(config, run_dir=run_dir, baseline_metric=baseline_metric)
    tol = float(tol or 0.0)
    if _finite_pos(noise):
        tol = max(tol, 2.0 * float(noise))
    return tol


def fresh_confirmed(trial) -> bool:
    ev = _ev(trial)
    return ev.get("verdict") == "confirmed" and \
        (ev.get("fresh_confirmation") or {}).get("passed") is True


# ---------------------------------------------------------------------------
# v2.20.0:先验豁免(confirmation_waived_by_prior)
# ---------------------------------------------------------------------------

#: 先验豁免的判据版本(落进 evidence 与 model_meta,便于复盘)
PRIOR_WAIVER_CRITERION = "seed_robust_gain_v1"


def measured_sigma_seed(trial, *, sigma_seed: float | None = None,
                        prior_ratio: float | None = None) -> dict:
    """实测 σ_seed 的单点解析(v2.20.0):只认实测来源,不落公式先验。

    优先级:
    1. 调用方显式给的 ``sigma_seed``(本 run 复现组的合并估计);
    2. 本 trial evaluation 里落的 ``sigma_seed_estimate``,且来源必须是
       ``multi_seed`` / ``run_replicates``(floor 之前的原始估计);
    3. 经验库同 ``data_base`` 的实测比值 ``prior_ratio`` × 本次 SE。

    任何一步都拿不到实测时返回 ``{"available": False, ...}``。
    """
    if _finite_pos(sigma_seed):
        return {"available": True, "sigma_seed": float(sigma_seed),
                "source": "caller"}

    def _from_eval(container) -> float | None:
        for key in ("vs_baseline_paired_bootstrap", "vs_anchor_paired_bootstrap"):
            c = (container or {}).get(key) or {}
            if not isinstance(c, dict):
                continue
            if c.get("sigma_seed_estimate_source") not in (
                    "multi_seed", "run_replicates"):
                continue
            if _finite_pos(c.get("sigma_seed_estimate")):
                return float(c["sigma_seed_estimate"])
        return None

    delta = _ev(trial).get("delta") or {}
    est = _from_eval(delta)
    if est is None:
        # confirm 侧落在 confirmation_validation 上
        cv = _ev(trial).get("confirmation_validation") or {}
        if cv.get("sigma_seed_estimate_source") in ("multi_seed",
                                                    "run_replicates") \
                and _finite_pos(cv.get("sigma_seed_estimate")):
            est = float(cv["sigma_seed_estimate"])
    if est is not None:
        return {"available": True, "sigma_seed": est, "source": "trial_eval"}
    se = None
    for key in ("vs_baseline_paired_bootstrap", "vs_anchor_paired_bootstrap"):
        c = (delta or {}).get(key) or {}
        if isinstance(c, dict):
            se = c.get("se_bootstrap", c.get("se"))
            if _finite_pos(se):
                break
    if _finite_pos(prior_ratio) and _finite_pos(se):
        return {"available": True,
                "sigma_seed": float(prior_ratio) * float(se),
                "source": "library_ratio", "ratio": float(prior_ratio)}
    return {"available": False, "reason": "无实测 σ_seed 来源"}


def seed_robust_gain(trial, *, sigma_seed: float | None = None,
                     prior_ratio: float | None = None,
                     z_required: float = 2.0,
                     higher_better: bool = True) -> dict:
    """v2.20.0:该 trial 相对 baseline 的增益是否**换 seed 后仍在**。

    这不是"confirm 会不会过"的预测 —— confirm 门是等价性检验(非劣 +
    保真),margin 恒为 ±z·sd,通过率与数据无关 ≈ 0.977,拿它做判据等于
    恒真(v2.20 复核结论)。

    本函数问的是另一个、并且在数据上有判别力的问题:把这个配方的分数
    整体挪动 ``z_required`` 个 seed 标准差(最坏方向),是否仍显著优于
    baseline。即

        z = Δ_dir / sqrt(SE² + σ_seed²) ≥ z_required

    ``σ_seed`` 必须来自**实测**(见 :func:`measured_sigma_seed`):本 run
    复现组 / 本 trial multi-seed / 经验库同 data_base 的历史比值。拿不到
    实测时返回 ``available=False`` —— 用公式先验去豁免等于用自己假设的
    噪声证明自己稳。

    返回 ``{"available", "z", "delta", "se", "sigma_seed", "sigma_source",
    "passed", "reason"}``。
    """
    ev = _ev(trial)
    delta = ev.get("delta") or {}
    d = delta.get("vs_baseline")
    if d is None:
        # 锚定模式下 vs_anchor 是当代的比较基准,但豁免要的是"相对起跑线"
        d = delta.get("vs_anchor")
    try:
        d = float(d)
    except (TypeError, ValueError):
        return {"available": False, "reason": "无 vs_baseline 点值增益",
                "passed": False}
    pb = (delta.get("vs_baseline_paired_bootstrap")
          or delta.get("vs_anchor_paired_bootstrap") or {})
    se = pb.get("se_bootstrap", pb.get("se"))
    if not _finite_pos(se):
        return {"available": False, "reason": "无配对 bootstrap SE",
                "delta": d, "passed": False}
    sig = measured_sigma_seed(trial, sigma_seed=sigma_seed,
                              prior_ratio=prior_ratio)
    if not sig.get("available"):
        return {"available": False,
                "reason": sig.get("reason") or "无实测 σ_seed(不用公式先验)",
                "delta": d, "se": float(se), "passed": False}
    # 方向归一:指标越小越好时 Δ 取反,使 z>0 恒表示"更好"
    ddir = d if higher_better else -d
    sd = math.sqrt(float(se) ** 2 + float(sig["sigma_seed"]) ** 2)
    z = ddir / sd if sd > 0 else 0.0
    return {"available": True, "criterion": PRIOR_WAIVER_CRITERION,
            "delta": d, "delta_directional": ddir, "se": float(se),
            "sigma_seed": float(sig["sigma_seed"]),
            "sigma_source": sig.get("source"),
            "z": float(z), "z_required": float(z_required),
            "passed": bool(z >= float(z_required))}


def divergence_summary(res: dict) -> str:
    """人读的一行分叉说明(日志/报错/报告共用)。"""
    if not res:
        return "交付位解析不可用"
    st = res.get("status")
    sc = res.get("search_champion")
    if st == "bound":
        return (f"交付位 {res.get('delivery_trial')} 绑定搜索冠军 {sc} 的配方 "
                f"{res.get('search_recipe')}")
    if st == "stale_within_tolerance":
        return (f"搜索冠军 {sc} 未确认;交付位落在陈旧确认 "
                f"{res.get('delivery_trial')}(差距 {res.get('gap'):+.6f} ≤ "
                f"容差 {res.get('tolerance'):.6f})")
    if st == "waived_by_prior":
        w = res.get("prior_waiver") or {}
        return (f"交付位 {res.get('delivery_trial')} 绑定搜索冠军配方,"
                f"**未经 confirm**,按先验豁免放行"
                f"(z={w.get('z')}≥{w.get('z_required')},"
                f"σ_seed={w.get('sigma_seed')}[{w.get('sigma_source')}])"
                " —— 换 seed 后增益仍在,但缺独立复现证明")
    if st == "diverged":
        stale = res.get("stale_candidates") or []
        best = stale[0] if stale else {}
        return (f"交付位分叉:搜索冠军 {sc}(配方 {res.get('search_recipe')},"
                f"{res.get('search_metric')})未通过独立确认;"
                f"最近的已确认 trial {best.get('trial_id')}(配方 "
                f"{best.get('recipe')},{best.get('metric')})落后 "
                f"{best.get('gap'):+.6f} > 容差 {res.get('tolerance'):.6f}"
                " —— 不得静默回落到它")
    if st == "unconfirmed":
        return f"搜索冠军 {sc} 未通过独立确认,且本 run 无任何已确认 trial"
    return "无合法搜索冠军"
