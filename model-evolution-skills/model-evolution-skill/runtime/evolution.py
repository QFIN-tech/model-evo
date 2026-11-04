"""Evolution:Explore / Exploit / Crossover 三类搜索动作(§9)与演进编排。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .generation import GenerationRecord, StopCondition, empty_reflection
from .leaderboard import CandidateArchive, Leaderboard
from .logging_utils import (get_logger, now_iso, dump_json, load_json,
                            load_yaml, locked_state_write)
from .spec import TaskSpec
from .trial import TrialFactory, TrialRecord, trial_is_screening

logger = get_logger(__name__)

DEFAULT_ACTION_MIX = {"explore": 0.5, "exploit": 0.3, "crossover": 0.2}


# ---------------------------------------------------------------------------
# v2.17.0(run4 regression.2 #1/#2/#17):晋级口径与 alpha 排程
# ---------------------------------------------------------------------------
# "尺子不可演进":以下行为全部由 run 的配置快照决定。代码 DEFAULTS 为旧
# 口径(parent / first_parent / uniform),在途 run 的快照里没有这些键 ⇒
# 行为不变;新 init 的 run 由 cmd_init 写入 anchor / iut(见
# runtime.protocol_gates.V2_17_0_RUN_CONFIG)。

#: anchor 模式的 vs_parent 护栏:vs_parent 的 p_better 必须 > 该值
#: (子代相对直接父代不得"显著更差")
ANCHOR_PARENT_GUARD_P = 0.05


def promotion_reference(config) -> str:
    """晋级对照:``anchor``(对交付锚点)| ``parent``(逐步对父代,旧)。"""
    v = str((config.get("evaluator.promotion_reference", "parent")
             if config is not None else "parent") or "parent").strip().lower()
    return "anchor" if v == "anchor" else "parent"


def crossover_parent_test(config) -> str:
    """多父代 crossover 的 vs_parent 口径:``iut`` | ``first_parent``(旧)。"""
    v = str((config.get("evaluator.crossover_parent_test", "first_parent")
             if config is not None else "first_parent")
            or "first_parent").strip().lower()
    return "iut" if v == "iut" else "first_parent"


def alpha_schedule(config) -> str:
    """代级 alpha 排程:``lord``(batch-LORD++)| ``uniform``(旧)。"""
    v = str((config.get("evaluator.alpha_schedule", "uniform")
             if config is not None else "uniform") or "uniform").strip().lower()
    return "lord" if v in ("lord", "lord++", "lord_plus_plus") else "uniform"


def _lord_gamma_raw(j: int) -> float:
    """LORD++ 标准 γ 形状:log(max(j,2)) / (j·e^{√log j}),j≥1。"""
    import math

    if j < 1:
        return 0.0
    return math.log(max(j, 2)) / (j * math.exp(math.sqrt(math.log(j))))


def lord_gamma(j: int, horizon: int) -> float:
    """有限视界归一的 γ_j(Σ_{j≤H} γ_j = 1;j>H 或 j<1 时为 0)。

    无限视界的标准归一常数(≈0.0772)在 8~20 代的 run 上过于吝啬
    (γ_1≈0.054 ⇒ α_1≈0.0027 < 现行均摊 0.0071);视界取 max_generations,
    预算保证在视界内成立。
    """
    h = max(1, int(horizon))
    if j < 1 or j > h:
        return 0.0
    norm = sum(_lord_gamma_raw(i) for i in range(1, h + 1))
    return _lord_gamma_raw(j) / norm if norm > 0 else 0.0


def _max_generations_of(spec, config) -> int:
    try:
        return int(((getattr(spec, "resource_budget", None) or {}).get(
            "max_generations")) or config.get("evolution.max_generations", 8))
    except Exception:  # noqa: BLE001
        return 8


def lord_level(ledger: list[dict], *, alpha_total: float, q_cap: float,
               horizon: int, w0_ratio: float = 0.5,
               upto_generation: int | None = None) -> dict:
    """近似 batch-LORD++:每代(有提升主张的代)一个 batch。

    α_t = γ_t·W0 + (α−W0)·γ_{t−τ1} + α·Σ_{j≥2} γ_{t−τj}
    t = 本 batch 序号(只数 family_size>0 的代),τj = 第 j 个发现所在
    batch(一代多个发现按重数计);W0 = w0_ratio·α(默认 α/2)。
    同代内用 BH 在 α_t 水平上裁决("近似":batch 内 BH 不是严格的
    online batch-BH)。wealth = W0 − Σα_spent + 回补,恒 ≥ 0。
    """
    alpha_total = float(alpha_total)
    w0 = max(0.0, min(alpha_total, float(w0_ratio) * alpha_total))
    batches = [e for e in (ledger or [])
               if int(e.get("family_size") or 0) > 0
               and (upto_generation is None
                    or int(e.get("generation", -1)) < int(upto_generation))]
    batches = sorted(batches, key=lambda e: int(e.get("generation", 0)))
    t = len(batches) + 1
    taus: list[int] = []
    for i, e in enumerate(batches, start=1):
        taus.extend([i] * int(e.get("discoveries") or 0))
    a_t = lord_gamma(t, horizon) * w0
    if taus:
        a_t += (alpha_total - w0) * lord_gamma(t - taus[0], horizon)
        a_t += alpha_total * sum(lord_gamma(t - tj, horizon)
                                 for tj in taus[1:])
    spent = sum(float(e.get("q_used") or 0.0) for e in batches)
    earned = ((alpha_total - w0) + alpha_total * (len(taus) - 1)) \
        if taus else 0.0
    wealth = max(0.0, w0 - spent + earned)
    q = max(0.0, min(float(q_cap), a_t, wealth))
    return {"schedule": "lord", "q": q, "alpha_t_raw": a_t,
            "batch_index": t, "gamma_t": lord_gamma(t, horizon),
            "w0": w0, "horizon": int(horizon), "discoveries_so_far": len(taus),
            "wealth_before": wealth}


def _alpha_level_for(state: dict, spec, config, gid: int) -> dict:
    """本代(gid)的检验水平 q 与排程明细(uniform / lord)。纯函数。"""
    alpha_total = float(config.get("evaluator.online_alpha_budget", 0.10))
    q_cap = float(config.get("evaluator.generation_fdr_q", 0.10))
    schedule = alpha_schedule(config)
    max_g = _max_generations_of(spec, config)
    # 已 finalize 的探索代数(gen-00 是 baseline,不计入分摊);口径与
    # v2.13 原实现逐字一致(重复 finalize 时含本代,只影响留痕字段)
    n_finalized = len([g for g in (state.get("finalized_generations") or [])
                       if g >= 1])
    gens_remaining = max(1, max_g - n_finalized)
    alpha_spent = float(state.get("alpha_spent") or 0.0)
    out = {"schedule": schedule, "alpha_total": alpha_total, "q_cap": q_cap,
           "alpha_spent": alpha_spent,
           "generations_remaining": gens_remaining}
    if schedule == "lord":
        horizon = config.get("evaluator.lord_horizon", None)
        try:
            horizon = int(horizon) if horizon else max_g
        except (TypeError, ValueError):
            horizon = max_g
        lv = lord_level(
            state.get("alpha_ledger") or [], alpha_total=alpha_total,
            q_cap=q_cap, horizon=horizon,
            w0_ratio=float(config.get("evaluator.lord_w0_ratio", 0.5) or 0.5),
            upto_generation=gid)
        out.update(lv)
        out["alpha_remaining"] = lv["wealth_before"]
    else:
        alpha_remaining = max(0.0, alpha_total - alpha_spent)
        out.update(q=min(q_cap, alpha_remaining / gens_remaining),
                   alpha_remaining=alpha_remaining)
    return out


def resolve_promotion_anchor(run_dir: Path, generation: int | None = None,
                             *, config=None) -> dict:
    """v2.17.0:本代的交付锚点(同一代内固定)。

    优先级:
    1. evolution-state.promotion_anchors[gen](上一代 finalize 时登记的、
       供本代使用的锚点 —— 重评旧代 trial 时仍用当代锚点);
    2. evolution-state.delivery_resolution.delivery_trial(已确认冠军);
    3. delivery_resolution 分叉/未确认时取最优的已确认 trial
       (stale_candidates[0]):锚点"只在确认通过后更新",冠军暂未确认
       不能让锚点退回 baseline(那会把门槛放松到起点);
    4. run-state.baseline_id。

    v2.19.0(``evaluator.anchor_mode=incumbent``,新 run):2~4 换成"现任
    冠军"——已登记代之前同视图最优的 accepted/confirmed trial(含 gen-00
    warm-start),见 :func:`_incumbent_anchor`。
    """
    rd = Path(run_dir)
    st: dict = {}
    try:
        p = rd / "evolution-state.json"
        if p.exists():
            st = load_json(p) or {}
    except Exception:  # noqa: BLE001
        st = {}
    if generation is not None:
        rec = (st.get("promotion_anchors") or {}).get(str(int(generation)))
        if isinstance(rec, dict) and rec.get("anchor_id"):
            return {**rec, "registered": True}
    out = _anchor_from_state(rd, st, config=config,
                             before_generation=generation)
    out["registered"] = False
    return out


def anchor_mode(config) -> str:
    """v2.19.0:锚点取值口径 ``incumbent`` | ``delivery``(旧)。"""
    v = str((config.get("evaluator.anchor_mode", "delivery")
             if config is not None else "delivery") or "delivery"
            ).strip().lower()
    return "incumbent" if v == "incumbent" else "delivery"


def _incumbent_anchor(run_dir: Path, *, trials: list | None = None,
                      before_generation: int | None = None) -> dict | None:
    """v2.19.0:现任冠军锚点 —— 同视图最优的 accepted/confirmed trial。

    动机(ds-flash 实测):旧锚点在首次 confirm 前一直是弱 baseline,
    warm-start 一步 +0.011 之后,每个 trial 的 vs_anchor≈+0.011、p=1,
    同代 BH 不再筛选任何东西(trial-009 Δparent=0.0000、p=0.577 仍 accepted)。
    现任冠军随每代的 accepted 前移;链式小步仍可相对它累积显著。

    - 视图:取 baseline 的 data_view(缺省 v1),跨视图 trial 不参选;
    - 资格:verdict ∈ {accepted, confirmed}、完整性有效、非 screening、当前
      非失败;``before_generation`` 给定时只看更早的代(同代固定);
    - 选择:主指标方向感知最优,平手取 trial_id 字典序最小者(确定性)。
    无合格 trial 返回 None(调用方回落旧口径)。
    """
    from .metrics import metric_direction
    from .trial import trial_currently_failed

    rd = Path(run_dir)
    if trials is None:
        trials = []
        troot = rd / "trials"
        if troot.is_dir():  # 纯读取:不经 TrialFactory(其构造会 mkdir)
            for d in sorted(troot.iterdir()):
                if (d / "trial.yaml").exists():
                    try:
                        trials.append(TrialRecord.load(d))
                    except Exception:  # noqa: BLE001
                        continue
    bid, base_view = None, "v1"
    try:
        rs = rd / "run-state.json"
        if rs.exists():
            bid = (load_json(rs) or {}).get("baseline_id")
    except Exception:  # noqa: BLE001
        bid = None
    if bid:
        for t in trials:
            if t.trial_id == bid:
                base_view = str(t.data.get("data_view") or "v1")
                break
    best = None
    for t in trials:
        if before_generation is not None and t.generation >= int(
                before_generation):
            continue
        ev = t.data.get("evaluation") or {}
        if ev.get("verdict") not in ("accepted", "confirmed"):
            continue
        if (ev.get("integrity") or {}).get("valid", True) is False \
                or trial_is_screening(t) or trial_currently_failed(t):
            continue
        if str(t.data.get("data_view") or "v1") != base_view:
            continue
        pm = ev.get("primary_metric") or {}
        try:
            val = float(pm.get("value"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(val):
            continue
        up = metric_direction(pm.get("name") or "")
        key = (val if up else -val)
        if best is None or key > best[0] or (
                key == best[0] and t.trial_id < best[1].trial_id):
            best = (key, t, val)
    if best is None:
        return None
    return {"anchor_id": best[1].trial_id, "source": "incumbent",
            "anchor_value": best[2], "data_view": base_view}


def _anchor_from_state(run_dir: Path, st: dict,
                       delivery_res: dict | None = None, *, config=None,
                       trials: list | None = None,
                       before_generation: int | None = None) -> dict:
    if anchor_mode(config) == "incumbent":
        try:
            inc = _incumbent_anchor(run_dir, trials=trials,
                                    before_generation=before_generation)
        except Exception as exc:  # noqa: BLE001 —— 回落旧口径
            logger.warning("现任冠军锚点解析失败(回落交付位口径): %s", exc)
            inc = None
        if inc is not None:
            return inc
    dr = delivery_res if delivery_res is not None else \
        (st.get("delivery_resolution") or {})
    if dr.get("delivery_trial"):
        return {"anchor_id": dr["delivery_trial"], "source": "delivery_trial"}
    stale = dr.get("stale_candidates") or []
    if stale and isinstance(stale[0], dict) and stale[0].get("trial_id"):
        return {"anchor_id": stale[0]["trial_id"],
                "source": "best_confirmed"}
    bid = None
    try:
        rs = Path(run_dir) / "run-state.json"
        if rs.exists():
            bid = (load_json(rs) or {}).get("baseline_id")
    except Exception:  # noqa: BLE001
        bid = None
    return {"anchor_id": bid, "source": "baseline" if bid else "none"}


def _is_promotion_claim(trial) -> bool:
    """代级校正的候选形态(与 _apply_generation_fdr 的排除条件同口径)。"""
    ev = trial.data.get("evaluation") or {}
    return not (str(trial.data.get("search_action") or "") == "confirm"
                or ev.get("root_baseline_only")
                or trial_is_screening(trial))


def _reference_se(trial, reference: str) -> tuple[float | None, str]:
    d = (trial.data.get("evaluation") or {}).get("delta") or {}
    if reference == "anchor":
        se = (d.get("vs_anchor_paired_bootstrap") or {}).get("se")
        if se is not None:
            return se, "vs_anchor"
    se = (d.get("vs_parent_paired_bootstrap") or {}).get("se")
    return se, "vs_parent"


def promotion_threshold_preview(run_dir, spec, config) -> dict:
    """v2.17.0:下一代过代级校正所需的门槛预览(纯读取,不写任何文件)。

    返回 q_next、family=1/2/3 时 lone-discovery 的单侧 z 门槛
    (BH 下第 1 小 p 需 ≤ q/m)、alpha_remaining、schedule、reference,
    以及 delta_hint = z × 最近同类 trial 配对 SE 的中位数(主指标单位,
    方向无关的正数;无样本时为 None)。WP2 planner-context 消费。
    """
    import math
    from statistics import NormalDist

    rd = Path(run_dir)
    st: dict = {}
    try:
        p = rd / "evolution-state.json"
        if p.exists():
            st = load_json(p) or {}
    except Exception:  # noqa: BLE001
        st = {}
    fin = [int(g) for g in (st.get("finalized_generations") or [])]
    next_gid = (max(fin) + 1) if fin else 1
    lv = _alpha_level_for(st, spec, config, next_gid)
    q = float(lv.get("q") or 0.0)
    reference = promotion_reference(config)
    n_boot = int(config.get("evaluator.bootstrap_samples", 2000) or 2000)

    # 最近同类 trial 的配对 SE(参照口径一致:anchor 模式优先 vs_anchor)
    se_rows: list[tuple[int, str, float, str]] = []
    trials: list = []
    try:
        troot = rd / "trials"
        if troot.is_dir():  # 纯读取:不经 TrialFactory(其构造会 mkdir)
            for d in sorted(troot.iterdir()):
                if (d / "trial.yaml").exists():
                    try:
                        trials.append(TrialRecord.load(d))
                    except Exception:  # noqa: BLE001
                        continue
    except Exception:  # noqa: BLE001
        trials = []
    for t in trials:
        if not _is_promotion_claim(t) or not t.parent_ids:
            continue
        se, src = _reference_se(t, reference)
        try:
            se = float(se)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(se) or se <= 0:
            continue
        se_rows.append((t.generation, str(t.data.get("search_action")
                                          or "explore"), se, src))
    se_rows.sort(key=lambda r: r[0])
    recent = se_rows[-6:]
    se_med = float(np.median([r[2] for r in recent])) if recent else None
    by_action: dict[str, float] = {}
    for action in sorted({r[1] for r in se_rows}):
        rows = [r[2] for r in se_rows if r[1] == action][-6:]
        by_action[action] = float(np.median(rows))

    def _z(p_thr: float) -> float | None:
        if p_thr <= 0 or p_thr >= 1:
            return None
        return float(NormalDist().inv_cdf(1.0 - p_thr))

    thresholds = []
    z_by_family: dict[str, float | None] = {}
    delta_hint: dict[str, float | None] | None = {} if se_med else None
    for m in (1, 2, 3):
        p_thr = q / m
        z = _z(p_thr)
        z_by_family[str(m)] = z
        dh = (z * se_med) if (z is not None and se_med) else None
        if delta_hint is not None:
            delta_hint[str(m)] = dh
        thresholds.append({
            "family": m, "p_one_sided": p_thr, "z": z, "delta_hint": dh,
            "below_bootstrap_resolution": bool(p_thr < 1.0 / n_boot)})
    z1 = z_by_family.get("1")
    anchor = resolve_promotion_anchor(rd, next_gid, config=config) \
        if reference == "anchor" else None
    mde_val = None
    try:
        from .mde import effective_mde

        mde_val = float(effective_mde(config, run_dir=rd))
    except Exception:  # noqa: BLE001
        mde_val = None
    return {
        "next_generation": next_gid,
        "schedule": lv.get("schedule"),
        "reference": reference,
        "anchor_id": (anchor or {}).get("anchor_id"),
        "anchor_source": (anchor or {}).get("source"),
        "q_next": q,
        "alpha_total": lv.get("alpha_total"),
        "alpha_spent": lv.get("alpha_spent"),
        "alpha_remaining": lv.get("alpha_remaining"),
        "generations_remaining": lv.get("generations_remaining"),
        "lord": ({k: lv.get(k) for k in ("batch_index", "gamma_t", "w0",
                                         "horizon", "discoveries_so_far")}
                 if lv.get("schedule") == "lord" else None),
        "z_by_family": z_by_family,
        "thresholds": thresholds,
        "se_reference": se_med,
        "se_source": (recent[-1][3] if recent else None),
        "se_n": len(recent),
        "delta_hint": delta_hint,
        "delta_hint_by_action": ({a: (z1 * s if z1 is not None else None)
                                  for a, s in by_action.items()}
                                 if by_action else {}),
        "mde": mde_val,
        "p_resolution": 1.0 / n_boot,
        "note": ("family=m 时本代第 1 小的单侧 p 需 ≤ q_next/m(BH 的单发现"
                 "情形);delta_hint = z×SE 中位数,即本代候选相对"
                 + ("交付锚点" if reference == "anchor" else "直接父代")
                 + "大约需要改善多少才能晋级"),
    }


def compute_undelivered_gain(trials: list, *, anchor_id: str | None,
                             anchor_source: str | None, spec, config,
                             run_dir=None, reference: str = "parent") -> dict:
    """v2.17.0(#17):同视图、非 screening、非 invalid 的最优 trial 相对交付
    锚点的 Δ(优先用 vs_anchor 及其 SE),附前 3 名候选。"""
    import math

    from .metrics import metric_direction

    higher = metric_direction(spec.primary_metric)
    by_id = {t.trial_id: t for t in trials}
    anchor = by_id.get(anchor_id) if anchor_id else None
    mde_val = None
    try:
        from .mde import effective_mde

        mde_val = float(effective_mde(config, run_dir=run_dir))
    except Exception:  # noqa: BLE001
        mde_val = None
    out: dict = {"anchor_id": anchor_id, "anchor_source": anchor_source,
                 "reference": reference, "mde": mde_val,
                 "best_trial": None, "delta": None, "gain": None,
                 "se": None, "z": None, "p_better": None, "source": None,
                 "gain_in_mde": None, "meets_mde": False,
                 "top_candidates": []}
    if anchor is None:
        out["note"] = "锚点缺失或不在 trials/ 中,无法计算未交付增益"
        return out

    def _val(t):
        v = ((t.data.get("evaluation") or {}).get("primary_metric")
             or {}).get("value")
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    a_val = _val(anchor)
    a_view = str(anchor.data.get("data_view") or "v1")
    if a_val is None:
        out["note"] = "锚点无主指标值"
        return out
    # v2.17.0(D4):与交付位**同配方**的 trial 不是"未交付的更优配方"——
    # confirm 链上的父代/子代与同配方复现只差 seed,它们与锚点的点值差是
    # seed 噪声(实测 trial-003 vs 其 confirm trial-005:+12.35 被报成
    # 7.3×MDE 的未交付增益)。配方归属沿用 confirmation.recipe_root(与
    # leaderboard.delivery_resolution 同一解析)。
    from .confirmation import recipe_root

    a_recipe = recipe_root(anchor, by_id)
    out["anchor_recipe"] = a_recipe
    same_recipe: list[str] = []
    rows = []
    for t in trials:
        if t.trial_id == anchor_id or trial_is_screening(t):
            continue
        if recipe_root(t, by_id) == a_recipe:
            same_recipe.append(t.trial_id)
            continue
        ev = t.data.get("evaluation") or {}
        if ev.get("verdict") in (None, "invalid") \
                or (ev.get("integrity") or {}).get("valid", True) is False:
            continue
        if str(t.data.get("search_action") or "") == "automl":
            continue
        if str(t.data.get("data_view") or "v1") != a_view:
            continue
        v = _val(t)
        if v is None:
            continue
        d = ev.get("delta") or {}
        if d.get("anchor_id") == anchor_id and d.get("vs_anchor") is not None:
            delta = float(d["vs_anchor"])
            pb = d.get("vs_anchor_paired_bootstrap") or {}
            se, p_better, src = pb.get("se"), pb.get("p_better"), "vs_anchor"
        else:
            delta, se, p_better, src = v - a_val, None, None, \
                "metric_difference"
        gain = delta if higher else -delta
        z = (gain / float(se)) if (se and float(se) > 0) else None
        rows.append({"trial_id": t.trial_id, "delta": delta, "gain": gain,
                     "se": se, "z": z, "p_better": p_better, "source": src,
                     "verdict": ev.get("verdict"),
                     "generation": t.generation})
    rows.sort(key=lambda r: (-r["gain"], r["trial_id"]))
    out["top_candidates"] = rows[:3]
    out["excluded_same_recipe"] = sorted(same_recipe)
    if rows:
        b = rows[0]
        out.update(best_trial=b["trial_id"], delta=b["delta"], gain=b["gain"],
                   se=b["se"], z=b["z"], p_better=b["p_better"],
                   source=b["source"])
        if mde_val and mde_val > 0:
            out["gain_in_mde"] = b["gain"] / mde_val
            out["meets_mde"] = bool(b["gain"] >= mde_val)
    out["note"] = ("gain>0 表示该 trial 优于交付锚点(方向感知);source="
                   "metric_difference 时无配对 SE,只是点值差;与交付位同配方"
                   "(confirm 链/复现,见 excluded_same_recipe)不计入")
    return out


def exclude_same_recipe_from_stop(stop: dict | None, trials: list) -> dict | None:
    """[v2.17.0 D4] 读取侧修正已落盘的 stop_report(旧 finalize 写入的
    undelivered_gain 可能把交付位的同配方 trial 当成"未交付更优候选")。

    与 :func:`compute_undelivered_gain` 同一配方解析(confirmation.recipe_root):
    剔除 top_candidates / best_trial 中与锚点同配方者并重选 best,同步剔除
    remaining_high_potential_directions 里 ``confirm_or_consolidate:<同配方>``。
    返回新 dict(不改入参);无 undelivered_gain 时原样返回。
    """
    if not isinstance(stop, dict) or not stop:
        return stop
    ug = stop.get("undelivered_gain")
    if not isinstance(ug, dict) or not ug.get("anchor_id"):
        return stop
    from .confirmation import recipe_root

    by_id = {t.trial_id: t for t in trials or []}
    anchor = by_id.get(ug.get("anchor_id"))
    if anchor is None:
        return stop
    a_recipe = recipe_root(anchor, by_id)

    def _same(tid) -> bool:
        t = by_id.get(str(tid))
        return t is not None and recipe_root(t, by_id) == a_recipe

    cands = [c for c in (ug.get("top_candidates") or [])
             if isinstance(c, dict)]
    drop = sorted({str(c.get("trial_id")) for c in cands
                   if _same(c.get("trial_id"))}
                  | ({str(ug["best_trial"])} if ug.get("best_trial")
                     and _same(ug.get("best_trial")) else set()))
    if not drop:
        return stop
    new_ug = dict(ug)
    keep = [c for c in cands if str(c.get("trial_id")) not in drop]
    new_ug["top_candidates"] = keep
    new_ug["excluded_same_recipe"] = sorted(
        set(ug.get("excluded_same_recipe") or []) | set(drop))
    new_ug["anchor_recipe"] = a_recipe
    for k in ("best_trial", "delta", "gain", "se", "z", "p_better", "source",
              "gain_in_mde"):
        new_ug[k] = None
    new_ug["meets_mde"] = False
    if keep:
        b = keep[0]
        new_ug.update(best_trial=b.get("trial_id"), delta=b.get("delta"),
                      gain=b.get("gain"), se=b.get("se"), z=b.get("z"),
                      p_better=b.get("p_better"), source=b.get("source"))
        mde = ug.get("mde")
        try:
            if mde and float(mde) > 0 and b.get("gain") is not None:
                new_ug["gain_in_mde"] = float(b["gain"]) / float(mde)
                new_ug["meets_mde"] = bool(float(b["gain"]) >= float(mde))
        except (TypeError, ValueError):
            pass
    new_ug["read_side_corrected"] = (
        "v2.17.0(D4):读取时剔除与交付位同配方的候选 " + ", ".join(drop))
    out = dict(stop)
    out["undelivered_gain"] = new_ug
    rem = out.get("remaining_high_potential_directions")
    if isinstance(rem, list):
        out["remaining_high_potential_directions"] = [
            r for r in rem
            if not (isinstance(r, str) and r.startswith(
                "confirm_or_consolidate:")
                and _same(r.split(":", 1)[1]))]
    return out


class CrossoverCompatibilityError(ValueError):
    pass


def check_crossover_compatibility(parents: list[dict],
                                  existing_combos: list[tuple]) -> list[str]:
    """Crossover 前兼容性检查(§13.3);返回冲突列表,空表示兼容。"""
    conflicts: list[str] = []
    if len(parents) < 2:
        conflicts.append("crossover 需要至少两个父代")
        return conflicts
    modules = [set(p.get("changed_modules") or []) for p in parents]
    # 同一模块被多个父代修改且未声明兼容
    all_mods: dict[str, int] = {}
    for mods in modules:
        for m in mods:
            all_mods[m] = all_mods.get(m, 0) + 1
    for mod, cnt in all_mods.items():
        if cnt > 1:
            conflicts.append(f"多个父代修改同一模块: {mod}")
    # 样本定义冲突
    sample_defs = {p.get("sample_definition") for p in parents
                   if p.get("sample_definition")}
    if len(sample_defs) > 1:
        conflicts.append(f"样本定义冲突: {sorted(sample_defs)}")
    # 数据视图不一致(协议 v7):跨视图组合不硬拒——per-view splits 保证
    # 同一实体切分口径,提示 Planner 在 Trial 内显式声明基准视图并 reindex
    views = {p.get("data_view") or "v1" for p in parents}
    if len(views) > 1:
        conflicts.append(
            f"父代数据视图不一致({sorted(views)}):组合 Trial 必须显式声明"
            "以某一视图为基准,另一父代的特征工程在 Trial 内重做对齐")
    # 已存在类似组合(state 中存 list[list[str]],这里统一 tuple 比较)
    combo = tuple(sorted(p["trial_id"] for p in parents))
    if combo in [tuple(c) for c in existing_combos]:
        conflicts.append(f"组合已实验过: {combo}")
    return conflicts


class EvolutionEngine:
    """Generation 级编排:建议动作配比、父代选择、停止判定。

    AutonomousPlanner 生成并筛选结构化候选；本引擎负责代级统计、
    Archive、停止条件与校正，Agent 只补充自定义候选实现。
    """

    def __init__(self, run_dir: Path, spec: TaskSpec, config: Config):
        self.run_dir = Path(run_dir)
        self.spec = spec
        self.config = config
        self.trial_factory = TrialFactory(run_dir, spec=spec, config=config)
        self.leaderboard = Leaderboard(self.run_dir, spec, config)
        self.archive = CandidateArchive(self.run_dir, spec, config)
        self.stop = StopCondition(spec, config, run_dir=self.run_dir)
        self.state_path = self.run_dir / "evolution-state.json"
        self.state: dict = self._load_state()

    def _load_state(self) -> dict:
        import json

        if self.state_path.exists():
            st = json.loads(self.state_path.read_text(encoding="utf-8"))
            # 旧状态文件兼容:补 finalized_generations 键(幂等守卫用)
            st.setdefault("finalized_generations", [])
            # v2.13:动态 alpha spending 台账(旧状态无键 → 0.0/[])
            st.setdefault("alpha_spent", 0.0)
            if not isinstance(st.get("alpha_ledger"), list):
                st["alpha_ledger"] = []
            return st
        return {
            "started_at": now_iso(),
            "n_generations": 0,
            "generation_best_deltas": [],
            "finalized_generations": [],
            "alpha_spent": 0.0,
            "alpha_ledger": [],
            "trials_accepted": 0,
            "crossover_combos": [],
            "started_trial_count_at_gen": {},
            "stopped": None,
        }

    def save_state(self) -> None:
        dump_json(self.state_path, self.state)

    # ---- 动作配比(§9.3,v12/P1-9 接线)----
    def suggest_action_mix(self, *, top_lineage_share: float | None = None,
                           last_deltas: list[float] | None = None,
                           noise: float | None = None) -> dict:
        """动作配比建议(v12 修复死代码:旧触发条件 |d|<1e-9 在浮点指标
        上永不发生,且全仓无调用者)。

        触发条件改为 noise 口径:最近两代 best delta 均低于 noise(方向
        感知)→ 上调 explore;top lineage 占比 >0.8 → 上调 explore。
        由 finalize-generation 后的 planner-context 消费(第 5 节
        budget 携带)。
        """
        mix = {k: float(self.config.get(f"evolution.{k}_ratio",
                                        DEFAULT_ACTION_MIX[k]))
               for k in DEFAULT_ACTION_MIX}
        need_more_explore = False
        if last_deltas and len(last_deltas) >= 2:
            thr = noise if (noise is not None and np.isfinite(noise)
                            and noise > 0) else 1e-9
            recent = last_deltas[-2:]
            if all((abs(d) < thr) if self._direction_up()
                   else (abs(d) < thr) for d in recent):
                need_more_explore = True  # 连续两轮 best 提升低于噪声带
        if top_lineage_share is not None and top_lineage_share > 0.8:
            need_more_explore = True  # top candidate 同一 lineage
        if need_more_explore:
            mix["explore"] = min(0.7, mix["explore"] + 0.2)
            total = sum(mix.values())
            mix = {k: v / total for k, v in mix.items()}
        return mix

    def _direction_up(self) -> bool:
        from .metrics import metric_direction

        return metric_direction(self.spec.primary_metric)

    # ---- Crossover 父代选择与登记(§13)----
    def crossover_parents(self, k: int = 2) -> list[dict]:
        parents = self.archive.diverse_parents(k)
        if len(parents) < 2:
            raise CrossoverCompatibilityError(
                "Archive 中互补父代不足 2 个,无法 crossover")
        combo = tuple(sorted(p["trial_id"] for p in parents))
        if combo in [tuple(c) for c in self.state.get("crossover_combos", [])]:
            raise CrossoverCompatibilityError(f"组合已实验过: {combo}")
        return parents

    def register_crossover(self, parent_ids: list[str]) -> None:
        """登记已执行的 crossover 组合,防重复实验(§13.3)。

        同一组合再次登记直接拒绝(CrossoverCompatibilityError)。
        """
        combos = self.state.setdefault("crossover_combos", [])
        combo = sorted(str(p) for p in parent_ids)
        if not combo:
            return
        if combo in [list(c) for c in combos]:
            raise CrossoverCompatibilityError(f"组合已实验过: {combo}")
        combos.append(combo)
        self.save_state()

    @staticmethod
    def _model_axis_tried(trials: list) -> bool:
        """v2.10.1/F5:model/HPO 轴是否真的开过(实测,非声明)。

        判据二选一:changed_modules 含 model/loss,或 search_action ∈
        {hpo, automl}。只看启动过的 trial(RUNNING 及之后),PROPOSED/
        FAILED 不计(没跑出来的 axis 不算试过)。
        """
        for t in trials:
            st = str(t.data.get("status") or "")
            if st in ("PROPOSED", "CODE_WRITTEN", "FAILED"):
                continue
            mods = t.data.get("changed_modules") or []
            if isinstance(mods, str):
                mods = [mods]
            if any(str(m).strip().lower() in ("model", "loss", "hpo", "automl")
                   for m in mods):
                return True
            if str(t.data.get("search_action") or "").strip().lower() \
                    in ("hpo", "automl"):
                return True
        return False

    def _inventory_unconsumed(self) -> int | None:
        """信息源清单未消费项数(清单未建 → None,不触发出口)。"""
        path = self.run_dir / "sample-profile" / "information-inventory.yaml"
        if not path.exists():
            return None
        try:
            data = load_yaml(path) or {}
            sources = data.get("sources") or []
            if not sources:
                return None
            return sum(
                1 for s in sources
                if isinstance(s, dict)
                and (s.get("consumption_status") == "unconsumed"))
        except Exception:  # noqa: BLE001
            return None

    def _planner_candidate_upper_bound(self) -> float:
        path = self.run_dir / "planner" / "latest.json"
        if not path.exists():
            return float("inf")
        try:
            value = load_json(path).get("candidate_value_upper_bound")
            return float(value) if value is not None else float("inf")
        except Exception:  # noqa: BLE001
            return float("inf")

    def _max_generations(self) -> int:
        # v2.13:与 StopCondition.check 同口径(spec 优先,config 兜底)
        try:
            return int((self.spec.resource_budget or {}).get(
                "max_generations") or self.config.get(
                "evolution.max_generations", 8))
        except Exception:  # noqa: BLE001
            return 8

    def _champion_generation_delta(
            self, gid: int, trials: list, best, best_metric,
            baseline_metric, *, empty_generation: bool) -> tuple:
        """[v2.17.0 D3] 新 run 的代际 delta:本代 finalize 后冠军 − 本代开始前冠军。

        "本代开始前的冠军"按优先级取:
          1. evolution-state.champion_metric_history 中本代已有记录(重复
             finalize:沿用首次记录的起点,尺子不随重算漂移);
          2. 历史中最近一个更早代的 champion_metric;
          3. 首代(此前没有任何 finalize)→ baseline 主指标;
          4. 在途 run 首次接入(已 finalize 过但无历史)→ 以更早代 trial 现算
             leaderboard 冠军(与 best_trial 同口径),仍无则 baseline。
        方向感知:正 = 冠军变好;冠军没换 = 0。空代 → None(不计入收敛/ROI)。
        返回 (delta, history_entry)。
        """
        from .metrics import metric_direction

        hi = metric_direction(self.spec.primary_metric)
        hist = [h for h in (self.state.get("champion_metric_history") or [])
                if isinstance(h, dict)]
        own = next((h for h in hist if int(h.get("generation", -1)) == gid),
                   None)
        before, src = None, None
        if own is not None:
            before = own.get("champion_before")
            src = own.get("champion_before_source")
        else:
            earlier = [h for h in hist if int(h.get("generation", -1)) < gid
                       and h.get("champion_metric") is not None]
            if earlier:
                h0 = max(earlier, key=lambda h: int(h.get("generation", -1)))
                before = h0.get("champion_metric")
                src = f"history:gen-{int(h0.get('generation')):02d}"
            else:
                prev_fin = [g for g in (self.state.get("finalized_generations")
                                        or []) if int(g) < gid]
                if prev_fin:
                    _prior = [t for t in trials
                              if t.generation is not None
                              and int(t.generation) < gid]
                    _bt = self.leaderboard.best_trial(_prior) if _prior \
                        else None
                    if _bt is not None:
                        before = ((_bt.data.get("evaluation") or {})
                                  .get("primary_metric") or {}).get("value")
                        src = f"recomputed:{_bt.trial_id}"
                if before is None and baseline_metric is not None:
                    before, src = baseline_metric, "baseline"
        delta = None
        if not empty_generation and best_metric is not None \
                and before is not None:
            delta = (float(best_metric) - float(before)) if hi \
                else (float(before) - float(best_metric))
        entry = {
            "generation": int(gid),
            "champion_trial": best.trial_id if best is not None else None,
            "champion_metric": best_metric,
            "champion_before": before,
            "champion_before_source": src,
            "delta": delta,
            "empty_generation": bool(empty_generation),
            "at": now_iso(),
        }
        return delta, entry

    def _ceiling_summary(self, champion_value) -> dict | None:
        """v2.19.0:天花板估计摘要(stop report / 判停用);失败只告警。"""
        try:
            from .ceiling import ceiling_summary
            from .mde import effective_mde

            try:
                mde = float(effective_mde(self.config, run_dir=self.run_dir))
            except Exception:  # noqa: BLE001
                mde = None
            return ceiling_summary(self.run_dir, champion_value, mde=mde)
        except Exception as exc:  # noqa: BLE001
            logger.warning("天花板估计读取失败(忽略): %s", exc)
            return None

    def _apply_generation_fdr(self, gid: int,
                              trials: list[TrialRecord],
                              *, charge_alpha: bool = True) -> dict:
        """对同代全部提升主张做 BH + 跨代 alpha 排程校正。

        Trial 级 p>=0.95 只能控制单次误报，不能控制数十次自适应搜索。
        v2.13 [FIX-2026-09-19]:旧口径 ``alpha_total/2^(gid+1)`` 按代几何
        收紧——gen-06 起 q<0.0005,低于 2000 次 bootstrap 的单侧 p 分辨率,
        真实 +2.5σ 提升结构性不可能通过;且当代无候选时 alpha 照样被消耗。
        新口径:有限 ``max_generations`` 代数内动态均摊,
        ``q_used = min(q_cap, alpha_remaining / gens_remaining)``,
        总花费仍 ≤ ``alpha_total``;有提升主张(family_size>0)才扣费。
        幂等:``charge_alpha=False``(重复 finalize)不重复扣费,复用
        ledger 中该代的 q_used。未经校正的 accepted 会降为 promising。

        v2.17.0(run4 regression.2 #1/#2.3):
        - ``evaluator.alpha_schedule=lord``:q_used 由 batch-LORD++ 给出
          (有发现时回补,见 :func:`lord_level`);ledger 留痕 schedule /
          batch_index / discoveries。uniform 保持上述口径。
        - ``evaluator.promotion_reference=anchor``:候选的 p 值取
          ``delta.vs_anchor_paired_bootstrap`` 的单侧 p(对交付锚点),
          附加护栏 vs_parent 的 p_better > 0.05(不得显著差于直接父代)
          与质量门(integrity/stability/slice/temporal);通过者
          verdict=accepted(具备冠军/confirm 资格),未通过者回到 vs_parent
          单试验 verdict(accepted 降 promising)。每个 trial 仍只算一次
          检验,family 与扣费口径不变。vs_anchor 缺失(锚点不可比/不可用)
          的 trial 按 parent 口径参与(reference=parent 并留痕原因)。
        - 两种模式都写 ``evaluation.parent_verdict``(= 校正前 verdict)。
        """
        from .metrics import benjamini_hochberg

        reference_mode = promotion_reference(self.config)
        # 幂等重放:该代已记账 → 复用原 q_used,不再扣费
        ledger = self.state.setdefault("alpha_ledger", [])
        replay = next((e for e in ledger if e.get("generation") == gid), None)
        if charge_alpha and replay is not None:
            charge_alpha = False  # ledger 命中视为 already
        level = _alpha_level_for(self.state, self.spec, self.config, gid)
        alpha_total = float(level["alpha_total"])
        gens_remaining = int(level["generations_remaining"])
        alpha_spent = float(self.state.get("alpha_spent") or 0.0)
        alpha_remaining = float(level.get("alpha_remaining") or 0.0)
        schedule = level["schedule"]
        if replay is not None:
            q_used = float(replay.get("q_used") or 0.0)
            schedule = str(replay.get("schedule") or schedule)
        else:
            q_used = float(level["q"])
        method = ("benjamini_hochberg_with_batch_lord_plus_plus"
                  if schedule == "lord"
                  else "benjamini_hochberg_with_dynamic_alpha_spending")
        try:
            guard_p = float(self.config.get(
                "evaluator.anchor_parent_guard_p", ANCHOR_PARENT_GUARD_P))
        except (TypeError, ValueError):
            guard_p = ANCHOR_PARENT_GUARD_P

        def _one_sided(pb: dict) -> float | None:
            p_better = (pb or {}).get("p_better", (pb or {}).get(
                "p_positive"))
            if p_better is None:
                return None
            return max(0.0, min(1.0, 1.0 - float(p_better)))

        candidates: list[tuple[TrialRecord, float, dict]] = []
        for trial in trials:
            if trial.generation != gid:
                continue
            ev = trial.data.get("evaluation") or {}
            # v2.13.2:screening-only trial 不提提升主张,不进 BH 候选集
            # (也不因此消耗 alpha 配额)。与 confirm/root_baseline_only
            # 并列 —— 三者都是"有 verdict 但不是提升主张"的形态。
            if not _is_promotion_claim(trial):
                continue
            delta = ev.get("delta") or {}
            p_parent = _one_sided(delta.get("vs_parent_paired_bootstrap"))
            meta = {"reference": "parent", "anchor_id": None,
                    "fallback": None}
            p_value = p_parent
            if reference_mode == "anchor":
                p_anchor = _one_sided(delta.get(
                    "vs_anchor_paired_bootstrap"))
                if p_anchor is not None:
                    p_value = p_anchor
                    meta.update(reference="anchor",
                                anchor_id=delta.get("anchor_id"))
                else:
                    note = delta.get("anchor_note") or {}
                    meta["fallback"] = (
                        note.get("reason") or note.get("status")
                        or ("anchor_incomparable"
                            if delta.get("anchor_incomparable")
                            else "vs_anchor 缺失"))
                    meta["anchor_id"] = delta.get("anchor_id")
            if p_value is None:
                continue
            candidates.append((trial, p_value, meta))
        # [FIX-2026-09-19] family_size==0:无提升主张,不扣费(旧版
        # 空转也消耗;confirm-only 代把预算白烧掉)
        ledger_entry = replay
        if candidates and charge_alpha:
            self.state["alpha_spent"] = round(alpha_spent + q_used, 12)
            ledger_entry = {"generation": gid, "q_used": q_used,
                            "family_size": len(candidates),
                            "schedule": schedule,
                            "reference": reference_mode}
            if schedule == "lord":
                ledger_entry.update(
                    batch_index=level.get("batch_index"),
                    gamma_t=level.get("gamma_t"),
                    alpha_t_raw=level.get("alpha_t_raw"),
                    wealth_before=level.get("wealth_before"))
            ledger.append(ledger_entry)
        flags = benjamini_hochberg([p for _, p, _ in candidates], q=q_used)
        rejected_ids: list[str] = []
        passed_ids: list[str] = []
        blocked: dict[str, str] = {}
        for (trial, p_value, meta), bh_passed in zip(candidates, flags):
            ev = trial.data.get("evaluation") or {}
            prev = ev.get("selection_correction") or {}
            pre_verdict = prev.get("pre_correction_verdict", ev.get("verdict"))
            pre_conf = prev.get("pre_correction_confidence",
                                ev.get("confidence"))
            pre_reason = prev.get("pre_correction_reason", ev.get("reason"))
            correction = {
                "method": method,
                "schedule": schedule,
                "reference": meta["reference"],
                "anchor_id": meta["anchor_id"],
                "generation": gid,
                "family_size": len(candidates),
                "alpha_total": alpha_total,
                "alpha_spent": float(self.state.get("alpha_spent") or 0.0),
                "q_used": q_used,
                "alpha_remaining_before": alpha_remaining,
                "generations_remaining": gens_remaining,
                "p_value_one_sided": p_value,
                "bh_passed": bool(bh_passed),
                "passed": bool(bh_passed),
                "pre_correction_verdict": pre_verdict,
                "pre_correction_confidence": pre_conf,
                "pre_correction_reason": pre_reason,
            }
            if meta.get("fallback"):
                correction["reference_fallback_reason"] = meta["fallback"]
            # 从校正前状态重算(重复 finalize 幂等:不叠加上次的改写)。
            # 校正之外的改写(如 leak_scan/人工裁 invalid)优先,绝不回滚。
            cur_verdict = ev.get("verdict")
            external_override = cur_verdict not in (
                pre_verdict, "accepted", "promising")
            verdict, confidence, reason = pre_verdict, pre_conf, pre_reason
            if external_override:
                verdict, confidence, reason = (
                    cur_verdict, ev.get("confidence"), ev.get("reason"))
                correction["external_verdict_override"] = cur_verdict
                if meta["reference"] == "anchor":
                    correction["passed"] = False
                    correction["promotion_blocked_by"] = "external_verdict"
                elif bh_passed:
                    passed_ids.append(trial.trial_id)  # 旧口径:仅报告列表
            elif meta["reference"] == "anchor":
                pb_parent = ((ev.get("delta") or {}).get(
                    "vs_parent_paired_bootstrap") or {})
                pp = pb_parent.get("p_better", pb_parent.get("p_positive"))
                guard_ok = pp is None or float(pp) > guard_p
                correction["parent_guard"] = {
                    "vs_parent_p_better": pp, "threshold": guard_p,
                    "passed": bool(guard_ok),
                    "note": (None if pp is not None else
                             "无 vs_parent 配对证据(跨视图/缺失),护栏不可评估,"
                             "按通过处理")}
                quality_ok = (
                    (ev.get("integrity") or {}).get("valid", True) is not False
                    and (ev.get("stability") or {}).get("passed", True)
                    is not False
                    and (ev.get("slice_evaluation") or {}).get("passed", True)
                    is not False
                    and (ev.get("temporal_evaluation") or {}).get(
                        "passed", True) is not False
                    and pre_verdict not in ("invalid",))
                block = None
                if bh_passed and not quality_ok:
                    block = "quality_gate"
                elif bh_passed and not guard_ok:
                    block = "parent_guard"
                promoted = bool(bh_passed and block is None)
                correction["passed"] = promoted
                correction["promotion_blocked_by"] = block
                if promoted:
                    verdict, confidence = "accepted", "high"
                    reason = (
                        f"vs 交付锚点 {meta['anchor_id']} 单侧 p="
                        f"{p_value:.4g} 通过同代 BH(q={q_used:.4g},"
                        f"family={len(candidates)},{schedule});vs_parent "
                        f"未显著更差(p_better={pp});具备冠军/confirm 资格")
                    passed_ids.append(trial.trial_id)
                else:
                    if block:
                        blocked[trial.trial_id] = block
                    if pre_verdict == "accepted":
                        verdict, confidence = "promising", "medium"
                        reason = (
                            "单试验 vs_parent 显著，但"
                            + ({"parent_guard": "被 vs_parent 护栏拦下",
                                "quality_gate": "未过质量门"}.get(block)
                               or "vs 交付锚点未通过同代 BH 与 alpha 排程")
                            + ";不可晋升为冠军")
                        rejected_ids.append(trial.trial_id)
                    elif block == "parent_guard":
                        reason = (f"{pre_reason};vs 锚点过代级校正但 vs_parent "
                                  f"显著更差(p_better={pp}≤{guard_p}),不晋级")
            else:
                if pre_verdict == "accepted" and not bh_passed:
                    verdict, confidence = "promising", "medium"
                    reason = ("单试验显著，但未通过同代 BH 与跨代 alpha "
                              "spending；不可晋升为冠军")
                    rejected_ids.append(trial.trial_id)
                elif bh_passed:
                    passed_ids.append(trial.trial_id)
            ev["selection_correction"] = correction
            ev["verdict"] = verdict
            if confidence is not None:
                ev["confidence"] = confidence
            if reason is not None:
                ev["reason"] = reason
            # vs_parent 口径的单试验 verdict(方向台账消费);外部改写
            # (如 invalid)优先
            ev["parent_verdict"] = cur_verdict if external_override \
                else pre_verdict
            trial.data["evaluation"] = ev
            dump_json(Path(trial.trial_dir) / "evaluation.json", ev)
            trial.flush()
        # 本代发现数(LORD++ 回补依据;按"BH 通过且最终 accepted"计,保守)
        n_disc = sum(
            1 for (t, _, _), f in zip(candidates, flags)
            if f and ((t.data.get("evaluation") or {}).get("verdict")
                      == "accepted"))
        if ledger_entry is not None and candidates:
            ledger_entry["discoveries"] = int(n_disc)
        # v2.17.2(C3):gen-00 warm-start 预注册家族(vs_baseline BH,不扣 alpha)
        warm_family = None
        try:
            warm_family = self._apply_warm_start_fdr(
                gid, trials,
                regular_ids={t.trial_id for t, _, _ in candidates},
                alpha_total=alpha_total)
        except Exception as exc:  # noqa: BLE001  校正失败不阻塞 finalize
            logger.warning("warm-start 家族校正失败(忽略): %s", exc)
        out = {"generation": gid, "family_size": len(candidates),
               "alpha_total": alpha_total, "q_used": q_used,
               "schedule": schedule, "reference": reference_mode,
               "alpha_remaining_before": alpha_remaining,
               "generations_remaining": gens_remaining,
               "passed": passed_ids, "downgraded": rejected_ids,
               "blocked": blocked, "discoveries": int(n_disc)}
        if warm_family is not None:
            out["warm_start_family"] = warm_family
        return out

    def _apply_warm_start_fdr(self, gid: int, trials: list[TrialRecord], *,
                              regular_ids: set[str],
                              alpha_total: float) -> dict | None:
        """v2.17.2(C3):gen-00 warm-start trial 的预注册家族校正。

        warm trial 复用全局库历史配方,没有父代 → 没有 vs_parent p 值,
        普通代级 FDR 永远跳过它们 → 无 selection_correction → 永远不可
        confirm(run4 opus5.5 trial-002/003:vs_baseline p_better=1.0 仍不可
        确认)。配方在本 run 看到数据之前就已登记,构成**预注册家族**:
        family = gen-00 全部 warm trial(按登记计数,不按结果筛),p = vs_baseline
        配对 bootstrap 的单侧 p(缺失按不显著),在 q = min(generation_fdr_q,
        online_alpha_budget) 上做 BH;不消耗跨代 alpha 预算(gen-00 本不参与
        分摊)。只决定 confirm 资格,不改 verdict。

        触发:finalize gen-00 时;或 gen-00 已 finalize 过、warm trial 仍缺
        校正记录(旧 run 回填)。已走普通家族(锚定模式下有 vs_anchor p)
        的 warm trial 不重复校正。
        """
        from .metrics import benjamini_hochberg
        from .trial import WARM_START_NOT_CONFIRMABLE

        warm = [t for t in trials
                if int(t.generation or 0) == 0
                and str(t.data.get("search_action") or "") == "warm_start"
                and t.trial_id not in regular_ids]
        if not warm:
            return None
        needs = gid == 0 or any(
            (t.data.get("evaluation") or {})
            and not (t.data.get("evaluation") or {}).get(
                "selection_correction")
            for t in warm)
        if not needs:
            return None
        try:
            q_cap = float(self.config.get("evaluator.generation_fdr_q", 0.10))
        except (TypeError, ValueError):
            q_cap = 0.10
        q = min(q_cap, float(alpha_total))

        def _p(t) -> float | None:
            pb = ((t.data.get("evaluation") or {}).get("delta") or {}).get(
                "vs_baseline_paired_bootstrap") or {}
            pbt = pb.get("p_better", pb.get("p_positive"))
            if pbt is None:
                return None
            return max(0.0, min(1.0, 1.0 - float(pbt)))

        pvals = [_p(t) for t in warm]
        flags = benjamini_hochberg(pvals, q=q)
        passed, not_passed = [], []
        for t, p, bh in zip(warm, pvals, flags):
            ev = t.data.get("evaluation") or {}
            if not ev:
                not_passed.append(t.trial_id)
                continue  # 未评估:无从写校正(资格门按缺记录给原因)
            ok = bool(bh) and ev.get("verdict") not in ("invalid",)
            correction = {
                "method": "benjamini_hochberg_preregistered_family",
                "family": "warm_start",
                "reference": "baseline",
                "generation": 0,
                "family_size": len(warm),
                "family_members": [w.trial_id for w in warm],
                "q_used": q,
                "alpha_charged": False,
                "p_value_one_sided": p,
                "bh_passed": bool(bh),
                "passed": ok,
                "pre_correction_verdict": ev.get("verdict"),
                "note": ("gen-00 warm-start 预注册家族:vs_baseline 单侧 p 的 "
                         "BH(不消耗跨代 alpha);只决定 confirm 资格,不改 "
                         "verdict"),
            }
            if not ok:
                correction["not_confirmable_reason"] = \
                    WARM_START_NOT_CONFIRMABLE
                not_passed.append(t.trial_id)
            else:
                passed.append(t.trial_id)
            if ev.get("selection_correction") != correction:
                ev["selection_correction"] = correction
                t.data["evaluation"] = ev
                if t.trial_dir is not None:
                    dump_json(Path(t.trial_dir) / "evaluation.json", ev)
                t.flush()
        return {"family_size": len(warm), "q_used": q,
                "alpha_charged": False, "passed": passed,
                "not_confirmable": not_passed}

    def _closing_consolidation(self, gid: int, trials: list) -> dict | None:
        """v2.19.1:停止时的收尾整合通道(evaluator.closing_consolidation)。

        背景:代级 BH 按同一代的 family 校正,停止某代时家族已被扣过 alpha;
        被拦下的正增益候选**(相对现任冠军 Δ ≥ MDE、完整性有效、非 screening
        非 confirm)** 就这么随停止一起作废。benchmark 复盘:72 个被拦候选里
        6 个在测试面上确实更好(全部是 amazon 的 catboost 血缘),它们与冠军
        的差已经过 MDE 门槛,只是没能在当代 family 里排到 BH 门槛。

        做法:取 best(champion 视图内最优),单独做**一次**检验
        family=1、q = min(generation_fdr_q, 剩余 alpha)(剩余为 0 不检验),
        并把 q 扣进 alpha_spent。通过**只授予 confirm 资格**:写
        ``evaluation.closing_consolidation``(passed=True),verdict 与代级
        selection_correction 都不动 —— 候选不当冠军、交付位状态不变
        (停止报告里的 delivery_resolution 不会过期),由 confirm(新 seed
        重训 + shadow 对照,独立于本次挑选)裁决它能否成为冠军;confirm 不过
        则维持原交付。通过时停止报告的 remaining 追加
        ``confirm_or_consolidate:<候选>``。资格判定见
        :func:`runtime.trial.confirmation_parent_eligibility`。护栏:
        - 只对 ``evaluator.closing_consolidation`` 打开的新 run 生效;
        - 每个 run 只开一次(``state.closing_consolidation`` 已存在即跳过);
        - 只考虑被同代 BH 拦下的 promising(有代级校正记录且 passed=False,
          不是被质量门 / vs_parent 护栏拦下);rejected 一律不进 —— 它可能
          是稳定性 / 切片 / 时序门判的;
        - 必须与冠军同视图、质量门全部有效、非 screening / confirm / warm_start;
        - 点值 Δ 必须 ≥ MDE(方向感知),避免把噪声送进检验;
        - vs_parent 护栏与代级校正同口径(不得显著差于父代)。
        返回留痕字典(无论通过与否),run 结束时应写进 stop report。
        """
        if not self.config.get("evaluator.closing_consolidation", False):
            return None
        prev = self.state.get("closing_consolidation")
        if isinstance(prev, dict) and prev:
            return {**prev, "skipped": "already_ran"}
        from .metrics import metric_direction
        from .mde import effective_mde
        from .trial import trial_currently_failed

        try:
            mde = float(effective_mde(self.config, run_dir=self.run_dir))
        except Exception:  # noqa: BLE001
            mde = float(self.config.get(
                "evaluator.minimum_meaningful_improvement", 0.001) or 0.001)
        better = (lambda a, b: a > b) if metric_direction(
            self.spec.primary_metric) else (lambda a, b: a < b)

        def _val(t):
            v = ((t.data.get("evaluation") or {}).get("primary_metric")
                 or {}).get("value")
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        # 冠军:排除 confirm / screening / 当前失败,与 baseline 同视图
        base_view = "v1"
        try:
            rs = load_json(self.run_dir / "run-state.json") or {}
            bid = rs.get("baseline_id")
            for t in trials:
                if t.trial_id == bid:
                    base_view = str(t.data.get("data_view") or "v1")
                    break
        except Exception:  # noqa: BLE001
            pass
        pool = [t for t in trials
                if _val(t) is not None
                and (t.data.get("evaluation") or {}).get("verdict")
                in ("accepted", "confirmed")
                and str(t.data.get("data_view") or "v1") == base_view
                and not trial_is_screening(t)
                and not trial_currently_failed(t)]
        champion = None
        for t in pool:
            if champion is None or better(_val(t), _val(champion)):
                champion = t
        if champion is None:
            return {"status": "no_champion"}
        ch_val = _val(champion)
        # 候选:被同代 BH 拦下的 promising(有代级校正记录、未通过、且不是被
        # 质量门 / vs_parent 护栏拦下)、非 confirm/warm_start/screening、
        # 质量门(完整性/稳定性/切片/时序)全部有效、相对冠军 Δ ≥ MDE。
        # rejected 不进:它可能是稳定性 / 切片 / 时序门判的,不能经此复活。
        def _quality_ok(ev: dict) -> bool:
            return all((ev.get(k) or {}).get(f, True) is not False
                       for k, f in (("integrity", "valid"),
                                    ("stability", "passed"),
                                    ("slice_evaluation", "passed"),
                                    ("temporal_evaluation", "passed")))

        cand, cand_delta = None, None
        n_eligible = 0
        n_reach_mde = 0
        for t in trials:
            if t.trial_id == champion.trial_id:
                continue
            ev = t.data.get("evaluation") or {}
            action = str(t.data.get("search_action") or "")
            if action in ("confirm", "warm_start") or trial_is_screening(t) \
                    or trial_currently_failed(t):
                continue
            if ev.get("verdict") != "promising":
                continue
            sc = ev.get("selection_correction") or {}
            if not sc or sc.get("passed") is not False \
                    or sc.get("promotion_blocked_by") \
                    or sc.get("method") == "closing_consolidation":
                continue
            if not _quality_ok(ev):
                continue
            if str(t.data.get("data_view") or "v1") != base_view:
                continue
            v = _val(t)
            if v is None:
                continue
            n_eligible += 1
            delta = (v - ch_val) if metric_direction(
                self.spec.primary_metric) else (ch_val - v)
            if delta < mde:
                continue
            n_reach_mde += 1
            if cand is None or better(v, _val(cand)):
                cand, cand_delta = t, delta
        out: dict = {"status": "no_candidate", "mde": mde,
                     "champion": champion.trial_id,
                     "champion_value": ch_val,
                     "n_candidates_eligible": n_eligible,
                     "n_candidates_reaching_mde": n_reach_mde}
        if cand is None:
            self.state["closing_consolidation"] = dict(out, at=now_iso())
            return out
        ev = cand.data.get("evaluation") or {}
        delta = ev.get("delta") or {}
        pb = delta.get("vs_anchor_paired_bootstrap") \
            or delta.get("vs_parent_paired_bootstrap") or {}
        p_better = pb.get("p_better", pb.get("p_positive"))
        if p_better is None:
            out.update(status="no_p_value", candidate=cand.trial_id,
                       delta_vs_champion=cand_delta,
                       note="候选缺少配对 bootstrap,无法单独检验")
            self.state["closing_consolidation"] = dict(out, at=now_iso())
            return out
        p_one = max(0.0, min(1.0, 1.0 - float(p_better)))
        # 剩余 alpha 与代级校正同口径(uniform = 总额 − 已花;lord = 当前
        # wealth);严格不超支:q ≤ 剩余,剩余为 0 时不检验
        level = _alpha_level_for(self.state, self.spec, self.config, gid + 1)
        remaining = max(0.0, float(level.get("alpha_remaining") or 0.0))
        q = min(float(self.config.get("evaluator.generation_fdr_q", 0.10)),
                remaining)
        if q <= 0:
            out.update(status="no_alpha_remaining", candidate=cand.trial_id,
                       delta_vs_champion=cand_delta, alpha_remaining=remaining)
            self.state["closing_consolidation"] = dict(out, at=now_iso())
            return out
        # vs_parent 护栏(与代级校正同口径)
        pb_parent = delta.get("vs_parent_paired_bootstrap") or {}
        pp = pb_parent.get("p_better", pb_parent.get("p_positive"))
        guard_p = float(self.config.get(
            "evaluator.anchor_parent_guard_p", ANCHOR_PARENT_GUARD_P))
        guard_ok = pp is None or float(pp) > guard_p
        passed = bool(p_one <= q and guard_ok)
        # 检验本身花掉 q(无论通过与否);lord 的 wealth 由 ledger 推导、
        # ledger 按代索引,这里不写 ledger,只记 alpha_spent 与留痕
        self.state["alpha_spent"] = round(float(
            self.state.get("alpha_spent") or 0.0) + q, 12)
        record = {
            "method": "closing_consolidation",
            "reference": "champion",
            "champion_id": champion.trial_id,
            "generation": int(gid),
            "family_size": 1,
            "q_used": q,
            "alpha_charged": q,
            "alpha_schedule": level.get("schedule"),
            "alpha_remaining_before": remaining,
            "p_value_one_sided": p_one,
            "p_source": ("vs_anchor" if delta.get("vs_anchor_paired_bootstrap")
                         else "vs_parent"),
            "passed": passed,
            "grants": "confirm_eligibility" if passed else None,
            "delta_vs_champion": cand_delta,
            "mde": mde,
            "parent_guard": {"vs_parent_p_better": pp,
                             "threshold": guard_p, "passed": bool(guard_ok)},
            "at": now_iso(),
            "note": ("停止时的收尾整合:被同代 BH 拦下、但相对现任冠军 "
                     "Δ≥MDE 的最佳候选单独检验一次(family=1);通过只授予 "
                     "confirm 资格,verdict 不变,能否成为冠军由 confirm 裁决"),
        }
        ev["closing_consolidation"] = record
        cand.data["evaluation"] = ev
        try:
            dump_json(Path(cand.trial_dir) / "evaluation.json", ev)
            cand.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning("收尾整合写盘失败(%s): %s", cand.trial_id, exc)
        out.update(status=("confirm_eligible" if passed else "not_eligible"),
                   candidate=cand.trial_id, p_value_one_sided=p_one,
                   q_used=q, alpha_charged=q, alpha_remaining=remaining,
                   delta_vs_champion=cand_delta,
                   parent_guard=record["parent_guard"])
        if passed:
            out["next_action"] = (
                f"收尾前先对 {cand.trial_id} 做 confirm(new-trial --action "
                f"confirm --from-parent {cand.trial_id});confirm 通过且优于"
                f"现任冠军即换冠军并绑定交付位,不通过则维持原交付")
        self.state["closing_consolidation"] = dict(out, at=now_iso())
        logger.info("收尾整合(%s): %s p=%.4g q=%.4g → %s", champion.trial_id,
                    cand.trial_id, p_one, q, out["status"])
        return out

    def _archive_superseded_stop(self, gid: int, *,
                                 empty_generation: bool) -> dict | None:
        """v2.17.2(B7):把被本代取代的停止报告归档进 ``stop_history``。

        条件:evolution-state 有 ``stopped``、其所属代(``generation``;旧报告
        按 ``n_generations − 1`` 推断)早于本代、且本代非空代(有新证据)。
        归档后清空 ``stopped``,run-state 的 ``stop_reason`` 移到
        ``superseded_stop_reason``、phase STOPPED → EVOLVING。本代若再次命中
        停止判定,随后写入新的 ``stopped``。返回归档摘要或 None。
        """
        prev = self.state.get("stopped")
        if not isinstance(prev, dict) or not prev or empty_generation:
            return None
        sg = prev.get("generation")
        if sg is None:
            try:
                sg = int(prev.get("n_generations")) - 1
            except (TypeError, ValueError):
                return None
        if int(sg) >= int(gid):
            return None
        entry = dict(prev, generation=int(sg), superseded_at=now_iso(),
                     superseded_by_generation=int(gid))
        self.state.setdefault("stop_history", []).append(entry)
        self.state.pop("stopped", None)
        try:
            st_path = self.run_dir / "run-state.json"
            with locked_state_write(st_path):
                st = load_json(st_path) if st_path.exists() else {}
                if st.get("stop_reason") is not None:
                    st["superseded_stop_reason"] = st.pop("stop_reason")
                if st.get("phase") == "STOPPED":
                    st["phase"] = "EVOLVING"
                st["status_updated_at"] = now_iso()
                dump_json(st_path, st)
        except Exception as exc:  # noqa: BLE001  run-state 同步失败不阻塞
            logger.warning("停止状态归档同步 run-state 失败: %s", exc)
        logger.info("gen-%02d 有新证据:gen-%02d 的停止报告(%s)归档进 "
                    "stop_history", gid, int(sg), prev.get("stop_reason"))
        return {"generation": int(sg), "stop_reason": prev.get("stop_reason"),
                "superseded_by_generation": int(gid)}

    # ---- Generation 收尾(§11.3)----
    def finalize_generation(
        self, gid: int,
        planned_trials: list[str],
        reflection: dict | None = None,
        unexplored_high_potential: list[str] | None = None,
        wall_time_hours: float = 0.0,
        baseline_metric: float | None = None,
        compute_hours: float | None = None,
    ) -> dict:
        # 幂等守卫:同一代号重复 finalize 不再重复推进 n_generations /
        # append generation_best_deltas(历史事故:gen-06 被重复 finalize,
        # deltas 数组混入重复项,污染 stop 判据输入与报告的代际序列)。
        # 重复调用时重算只读部分(leaderboard/archive/stop_report 仍
        # 返回最新视图),但状态推进只做一次。
        already = gid in (self.state.get("finalized_generations") or [])
        trials = self.trial_factory.all_trials()
        # v2.13:重复 finalize 不重复扣 alpha(幂等,复用 ledger 记录)
        fdr_report = self._apply_generation_fdr(
            gid, trials, charge_alpha=not already)
        # flush 后重新加载，确保 leaderboard/archive 消费校正后的裁决。
        trials = self.trial_factory.all_trials()
        lb = self.leaderboard.build(trials)
        archive_update = self.archive.update(trials)
        best = self.leaderboard.best_trial(trials)
        best_metric = None
        if best is not None:
            best_metric = best.data["evaluation"]["primary_metric"]["value"]
        # v2.14.0:交付位解析随每代落 evolution-state(停止判定与收尾
        # 共用同一口径;status=diverged 时编排方必须在收尾前处理)
        try:
            _delivery_res = self.leaderboard.delivery_resolution(trials)
        except Exception as exc:  # noqa: BLE001
            logger.warning("交付位解析失败: %s", exc)
            _delivery_res = {}
        self.state["delivery_resolution"] = _delivery_res
        # v2.17.0:登记下一代的晋级锚点(同一代内固定;已登记的不覆盖 ——
        # 重复 finalize 时下一代可能已按旧锚点评估过)。锚点只在确认通过
        # 后更新:冠军未确认(diverged)时取最优已确认 trial,不退回 baseline。
        try:
            _next_anchor = _anchor_from_state(
                self.run_dir, self.state, _delivery_res, config=self.config,
                trials=trials, before_generation=gid + 1)
            _next_anchor.update(registered_at=now_iso(),
                                registered_by_generation=gid)
            self.state.setdefault("promotion_anchors", {}).setdefault(
                str(gid + 1), _next_anchor)
        except Exception as exc:  # noqa: BLE001
            logger.warning("晋级锚点登记失败(下一代按 delivery_resolution "
                           "现算): %s", exc)

        # 上一代 best(方向感知)以计算代际 delta
        from .metrics import metric_direction
        better = (lambda a, b: a > b) if metric_direction(
            self.spec.primary_metric) else (lambda a, b: a < b)
        # v2.17.0(D3):空代 = 本代没有任何已评估 trial(有主指标)。空代没有
        # 新证据:delta 记 None、不计入收敛/ROI、不得触发停止;finalize 仍完成
        # (幂等),并给出明确告警。
        _gen_evaluated = [
            t for t in trials if t.generation == gid
            and ((t.data.get("evaluation") or {}).get("primary_metric")
                 or {}).get("value") is not None]
        empty_generation = not _gen_evaluated
        _champion_protocol = False
        try:
            from .protocol_gates import run_at_least

            _champion_protocol = run_at_least(self.run_dir)
        except Exception:  # noqa: BLE001 行为门读失败按旧 run 处理
            _champion_protocol = False
        champion_entry: dict | None = None
        if _champion_protocol:
            # 新 run:代际 delta = 本代 finalize 后冠军 − 本代开始前冠军
            delta_gen, champion_entry = self._champion_generation_delta(
                gid, trials, best, best_metric, baseline_metric,
                empty_generation=empty_generation)
        else:
            # 旧 run 保持原口径(尺子不可演进):本代冠军 − 上一代最优
            prev_best = None
            for t in trials:
                # v2.13.2:screening trial 的分数不作为上一代 best —— 它是
                # "全加"的宽面,分数系统性偏高,混进来会把 delta_gen 压成
                # 假的负值,进而影响"连续多代无正增益"的收敛判据。
                if t.generation == gid - 1 and not trial_is_screening(t):
                    v = (t.data.get("evaluation") or {}).get(
                        "primary_metric", {}).get("value")
                    if v is not None and (prev_best is None
                                          or better(v, prev_best)):
                        prev_best = v
            delta_gen = None
            if best_metric is not None and prev_best is not None \
                    and not empty_generation:
                delta_gen = (best_metric - prev_best) if metric_direction(
                    self.spec.primary_metric) else (prev_best - best_metric)
        empty_generation_warning = None
        if empty_generation:
            empty_generation_warning = (
                f"gen-{gid:02d} 是空代:本代没有任何已评估 trial(planned "
                f"{len([t for t in trials if t.generation == gid])} 个,均未产出"
                "主指标)。代际 delta 记 None,不计入收敛/ROI 判据,本次 "
                "finalize 不会因空代触发停止判定(只保留硬预算原因);请先确认"
                "本代 trial 是否漏跑(run-generation / diagnose)")
            logger.warning(empty_generation_warning)

        # 本代结束后,自动把本代 crossover Trial 的父代组合登记防重(§13.3)
        for t in trials:
            if t.generation == gid and \
                    t.data.get("search_action") == "crossover" and t.parent_ids:
                try:
                    self.register_crossover(t.parent_ids)
                except CrossoverCompatibilityError:
                    logger.warning("crossover 组合重复登记被跳过: %s",
                                   t.parent_ids)

        if not already:
            self.state["n_generations"] = max(
                int(self.state.get("n_generations", 0) or 0), gid + 1)
            self.state["generation_best_deltas"].append(delta_gen)
            self.state.setdefault("finalized_generations", []).append(gid)
            if champion_entry is not None:
                self.state.setdefault("champion_metric_history", []).append(
                    champion_entry)
            if empty_generation:
                _eg = self.state.setdefault("empty_generations", [])
                if gid not in _eg:
                    _eg.append(gid)
        else:
            logger.warning(
                "gen-%02d 已 finalize 过:跳过 n_generations 推进与 "
                "generation_best_deltas 追加(幂等守卫,只重算只读视图)",
                gid)
        # 最近一代新增 accepted(§17.3 口径:不是全史累计)
        # v2.11.7:排除"仅建立跑线"的根 Trial。它们 verdict 为 accepted
        # (无父代可比),但 reason 自述不参与提升裁决;旧计数只按 verdict
        # 统计,把它们当成提升证据 ⇒ 收敛出口与穷尽出口的
        # `trials_accepted == 0` 同时不成立,演进永远无法自动收口
        # (gen-02 实测:trial-007/008 漏传 --parents,两个出口全被堵死)。
        def _is_improvement(t) -> bool:
            ev = t.data.get("evaluation") or {}
            if ev.get("verdict") not in ("accepted", "confirmed"):
                return False
            # v2.13.2:screening trial 的 accepted 不是提升证据(它被显式
            # 排除在代级校正之外)。与 root_baseline_only 同理:旧计数只按
            # verdict 统计,会让收敛出口的 trials_accepted==0 失锁。
            # v2.14.0:confirm 同理 —— 它复现父代配方,不提出新增益
            # (代级 FDR 早已把它排除在候选集外),计入会让"只做确认"的
            # 一代堵死收敛/穷尽出口。
            # v2.15.6:warm_start 是"验证历史配方有效性",不是"发现新方向"。
            # 排除它与排除 confirm 同理:它不提出新增益,计入会误报 gen-00
            # "有新发现"并污染后续收敛/穷尽判据。
            # v2.17.0:anchor 模式下 accepted = "显著优于交付锚点";子代相对
            # 直接父代退步(parent_verdict=rejected)时它仍可当选,但不是
            # "本代新增的提升证据"(否则同一条未交付链每代都会把收敛/穷尽
            # 出口堵死)。parent 模式 parent_verdict 恒为校正前 verdict,
            # rejected 不可能变成 accepted,此条不生效。
            if _anchor_mode and ev.get("parent_verdict") in (
                    "rejected", "invalid"):
                return False
            return not ev.get("root_baseline_only") \
                and not trial_is_screening(t) \
                and str(t.data.get("search_action") or "") != "confirm" \
                and str(t.data.get("search_action") or "") != "warm_start"

        _anchor_mode = promotion_reference(self.config) == "anchor"

        accepted_now = sum(
            1 for t in trials
            if t.generation == gid and _is_improvement(t))
        self.state["trials_accepted"] = sum(
            1 for t in trials if _is_improvement(t))
        self.state["accepted_in_last_generation"] = accepted_now

        # 停止判定
        noise = None
        for t in trials:
            n = (t.data.get("evaluation") or {}).get("noise")
            if n is not None:
                noise = n
                break
        # P1-8:noise 取**当代** trial 的噪声估计(旧循环取的是"第一个
        # 有 noise 的 trial"——gen-00 baseline 的口径会贯穿全 run)
        for t in trials:
            if t.generation == gid:
                n = (t.data.get("evaluation") or {}).get("noise")
                if n is not None:
                    noise = n
                    break
        # 数据视图汇总(协议 v7):trial 声明的 data_view 全集,供单口径
        # 停止护栏判定;未声明的旧 trial 归 v1
        data_views = sorted({t.data.get("data_view") or "v1"
                             for t in trials})
        heavy_now = sum(
            1 for t in trials if t.generation == gid
            and (t.data.get("expected_resources") or {}).get("heavy"))
        # [v2.13.2] 特征 bundle 槽位对账。只记不拒 —— 候选供给薄
        # (gen-01 时 directions.json 尚不存在,信息源清单每源 ≤2 个粒度)
        # 是合法状态,硬拒会把 run 卡死;但"配了宽度却没用满"必须留痕,
        # 否则 bundle_beam_width 又会变成无从审计的静默配置。
        feature_beam = None
        _beam_width = int(self.config.get(
            "feature_exploration.bundle_beam_width", 4))
        if _beam_width > 0:
            _feature_trials = [
                t for t in trials if t.generation == gid
                and (trial_is_screening(t) or (
                    str(t.data.get("search_action") or "")
                    in ("explore", "data_view")
                    and {str(m).strip().lower()
                         for m in (t.data.get("changed_modules") or [])}
                    & {"feature", "sample", "data"}))]
            _screening = sum(1 for t in _feature_trials
                             if trial_is_screening(t))
            feature_beam = {
                "bundle_beam_width": _beam_width,
                "feature_trials": len(_feature_trials),
                "screening_trials": _screening,
                "underfilled": max(0, _beam_width - len(_feature_trials)),
                "note": ("underfilled>0 = 本代特征槽位未填满。可能是候选"
                         "供给薄(合法),也可能是编排漏建 —— 按 note 核查,"
                         "不要静默当成'本代就是窄'"),
            }
        # v16/P0-2:compute 口径聚合——全部已启动 trial 的 runtime_sec
        # 之和(未显式传入时兜底自算)。wall 预算 pause-blind 的修复:
        # 双轨呈现进 stop_report 与 generation.yaml.budget_metrics
        # v2.17.2(B5):单源 compute_hours_summary(trial Σruntime + 收尾
        # finalize-all/evaluate-final 用时);finalize_hours 分列进 budget_metrics
        from .trial import compute_hours_summary

        _compute_acct = compute_hours_summary(self.run_dir, trials)
        if compute_hours is None:
            compute_hours = _compute_acct["compute_hours"]
        # shadow sanity(v8):本代 best trial 的 validation↔shadow gap
        # v2.17.2(C4):诊断位改名 gap_sanity_passed(兼容旧键 passed)
        shadow_check = None
        if best is not None:
            sh = (best.data.get("evaluation") or {}).get("shadow_test") or {}
            _gap_ok = sh.get("gap_sanity_passed", sh.get("passed"))
            if _gap_ok is not None:
                shadow_check = {
                    "trial_id": best.trial_id,
                    "criterion": "gap_sanity",
                    "gap_sanity_passed": _gap_ok,
                    "passed": _gap_ok,  # 兼容键(generation.yaml 旧读者)
                    "gap_relative": sh.get("gap_relative"),
                }
        # 分方向聚合(v8):各 search_action 的逐代 best delta 序列与本代
        # 新增 accepted(报告维度,不改停止触发语义)
        # v2.13.2:screening trial 全程排除 —— 它按 action 分组会与普通
        # explore 同桶,而它的分数(宽面全加)不构成该方向的可归因证据,
        # 混进去会污染逐代 best delta 序列,进而影响停止判据的输入。
        _scored_trials = [t for t in trials if not trial_is_screening(t)]
        direction_deltas: dict[str, list[float]] = {}
        best_by_gen_dir: dict[tuple[int, str], float] = {}
        for t in _scored_trials:
            v = (t.data.get("evaluation") or {}).get("primary_metric", {})\
                .get("value")
            if v is None:
                continue
            key = (t.generation, str(t.data.get("search_action") or "explore"))
            if key not in best_by_gen_dir or better(v, best_by_gen_dir[key]):
                best_by_gen_dir[key] = v
        gens = sorted({g for g, _ in best_by_gen_dir})
        for action in sorted({a for _, a in best_by_gen_dir}):
            series = [best_by_gen_dir[(g, action)]
                      for g in gens if (g, action) in best_by_gen_dir]
            deltas: list[float] = []
            for i in range(1, len(series)):
                d = (series[i] - series[i - 1]) if metric_direction(
                    self.spec.primary_metric) else (series[i - 1] - series[i])
                deltas.append(float(d))
            direction_deltas[action] = deltas
        direction_accepted = {
            action: sum(
                1 for t in _scored_trials
                if t.generation == gid
                and str(t.data.get("search_action") or "explore") == action
                and (t.data.get("evaluation") or {}).get("verdict")
                in ("accepted", "confirmed"))
            for action in direction_deltas}
        # P1-8:逐代 best delta 的 bootstrap CI(收敛判据统计化)。
        # 从各代 best trial 的 vs_parent_paired_bootstrap 取 delta_ci;
        # 缺 CI 的代记 (None, None) 退化为点值口径。
        recent_gen_cis: list[tuple[float, float]] = []
        # v2.17.0(D3):空代(无已评估 trial)不计入收敛判据 —— 取最近 3 个
        # 非空代(无空代时与旧口径 gid-2..gid 完全一致)
        _ci_gens: list[int] = []
        for g in range(gid, -1, -1):
            if any(t.generation == g and ((t.data.get("evaluation") or {})
                                          .get("primary_metric") or {}
                                          ).get("value") is not None
                   for t in _scored_trials):
                _ci_gens.append(g)
            if len(_ci_gens) >= 3:
                break
        for g in sorted(_ci_gens):
            best_t = None
            for t in _scored_trials:
                if t.generation != g:
                    continue
                v = (t.data.get("evaluation") or {}).get(
                    "primary_metric", {}).get("value")
                if v is None:
                    continue
                if best_t is None or better(v, (best_t.data.get(
                        "evaluation") or {}).get(
                    "primary_metric", {}).get("value") or v):
                    best_t = t
            ci = None
            if best_t is not None:
                ci = ((best_t.data.get("evaluation") or {}).get("delta")
                      or {}).get("vs_parent_paired_bootstrap", {}
                                 ).get("delta_ci")
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                recent_gen_cis.append((float(ci[0]), float(ci[1])))
            else:
                recent_gen_cis.append((float("nan"), float("nan")))
        # v2.15.8:代级 ROI 停止接线(旧实现从不传 generation_compute_hours,
        # StopCondition 的 ROI 分支是死代码)。每次 finalize 按
        # finalized_generations 现算、与 generation_best_deltas 逐代对齐
        # (增量追加会在已有 N 代 delta 的在跑 run 上错位配对);warm trial
        # 只是重放历史配方,不计入。只对 init 时记了 skill_version≥2.15.8
        # 的 run 启用:在跑的旧 run 中途接入会让历史代的 ROI 突然生效。
        generation_compute_hours: list[float] | None = None
        try:
            from . import version_at_least

            _rs_path = self.run_dir / "run-state.json"
            _rs = load_json(_rs_path) if _rs_path.exists() else {}
            if version_at_least(_rs.get("skill_version"), "2.15.8"):
                _sec_by_gen: dict[int, float] = {}
                for t in trials:
                    if str(t.data.get("search_action") or "") == "warm_start":
                        continue
                    rt = (t.data.get("result") or {}).get("runtime_sec")
                    if isinstance(rt, (int, float)):
                        g = int(t.generation or 0)
                        _sec_by_gen[g] = _sec_by_gen.get(g, 0.0) + float(rt)
                _fin = list(self.state.get("finalized_generations") or [])
                generation_compute_hours = [
                    _sec_by_gen.get(int(g), 0.0) / 3600.0 for g in _fin]
                if len(generation_compute_hours) != len(
                        self.state["generation_best_deltas"]):
                    generation_compute_hours = None
        except Exception as exc:  # noqa: BLE001
            logger.warning("代级 compute 汇总失败(ROI 停止本代不参与): %s", exc)
            generation_compute_hours = None
        # v2.17.0 挂钩(a):信息源清单按 trial 回填(WP2 实现;函数不存在
        # 时静默跳过)。放在停止判定之前:穷尽出口读 unconsumed 计数。
        try:
            from . import inventory as _inventory

            if hasattr(_inventory, "backfill_from_trials"):
                _inventory.backfill_from_trials(self.run_dir, trials,
                                                generation=gid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("information-inventory 回填失败(忽略): %s", exc)
        # v2.17.0(#17):未交付累积增益(只进 stop_report,不参与触发)
        undelivered = None
        try:
            _ua = _anchor_from_state(self.run_dir, self.state, _delivery_res)
            undelivered = compute_undelivered_gain(
                trials, anchor_id=_ua.get("anchor_id"),
                anchor_source=_ua.get("source"), spec=self.spec,
                config=self.config, run_dir=self.run_dir,
                reference=promotion_reference(self.config))
        except Exception as exc:  # noqa: BLE001
            logger.warning("未交付增益计算失败(忽略): %s", exc)
        # v2.17.2(B7):上一代的停止报告在本代有新证据(非空代)时归档进
        # stop_history 并清空 stopped/stop_reason —— 旧实现 stopped 永不清除,
        # 停止后继续演进的 run 一直显示旧停止原因(run4 opus5.5)
        stop_superseded = self._archive_superseded_stop(
            gid, empty_generation=empty_generation)
        stop_report = self.stop.check(
            n_trials_started=self.trial_factory.count_started(),
            n_generations=self.state["n_generations"],
            wall_time_hours=wall_time_hours,
            compute_hours=compute_hours,
            trials_accepted=accepted_now,
            best_metric=best_metric,
            baseline_metric=baseline_metric,
            generation_best_deltas=self.state["generation_best_deltas"],
            noise=noise,
            unexplored_high_potential=unexplored_high_potential or [],
            data_views_tried=data_views,
            direction_deltas=direction_deltas,
            direction_accepted=direction_accepted,
            recent_gen_cis=recent_gen_cis,
            # v18/P0-3:信息通道穷尽出口的外部输入——信息源清单(建过才有
            # unconsumed);缺失 → 对应条件 False,不触发(诚实出口,不是绕过)。
            inventory_unconsumed=self._inventory_unconsumed(),
            # v2.10.1/F5:model/HPO 轴实测判据(旧版恒 True,实测 0 个
            # model 轴 trial 的 run 也报 true 并据此提前停止)
            model_axis_tried=self._model_axis_tried(trials),
            candidate_value_upper_bound=self._planner_candidate_upper_bound(),
            # v2.14.0:必须是**搜索冠军配方**的确认(或差距在容差内的
            # 陈旧确认)。旧口径 any(...) 被 gen-03 对 trial-006 的确认
            # 永久满足,冠军换两次都不影响停止与交付(run2 事故)。
            fresh_confirmation_available=(
                _delivery_res.get("delivery_trial") is not None),
            generation_compute_hours=generation_compute_hours,
            undelivered_gain=undelivered,
            empty_generation=empty_generation,
            ceiling=self._ceiling_summary(best_metric),
        )
        if stop_report is not None:
            # v2.14.0:停止时交付位状态随 stop_report 呈现 —— diverged/
            # unconfirmed 意味着收尾不会产出已确认交付,编排方必须在
            # finalize 之前向用户呈报,而不是走默认分支继续
            stop_report["delivery_resolution"] = {
                k: _delivery_res.get(k) for k in (
                    "status", "search_champion", "delivery_trial", "gap",
                    "tolerance")}
            stop_report["generation"] = int(gid)  # v2.17.2(B7):按代归档用
            # v2.19.1:收尾整合通道 —— 停止建议触发后,把"同视图、未晋级、
            # 比现任冠军好 ≥ MDE"的最佳候选单独检验一次(family=1,剩余
            # alpha),通过只授予 confirm 资格(verdict/交付位不变,上面的
            # delivery_resolution 仍然有效),remaining 追加
            # confirm_or_consolidate:<候选>。动机(benchmark 8 数据集
            # 复盘):被同代 BH 拦下的 72 个正增益候选里 6 个在测试面上更好,
            # 全部落在 amazon;它们与冠军的差已被证明 > MDE 但没进 family,
            # 停止时把预算耗掉就等于放弃。每个 run 只开一次。
            _closing = self._closing_consolidation(gid, trials)
            if _closing is not None:
                stop_report["closing_consolidation"] = _closing
                if _closing.get("status") == "confirm_eligible":
                    _rem = stop_report.setdefault(
                        "remaining_high_potential_directions", [])
                    _item = f"confirm_or_consolidate:{_closing['candidate']}"
                    if isinstance(_rem, list) and _item not in _rem:
                        _rem.append(_item)
            self.state["stopped"] = stop_report
            # 停止判定命中时同步 run-state(phase 单调推进到 STOPPED;
            # 整体进展明细仍在 evolution-state.json)
            # v2.17.2(B7):重复 finalize 已收尾(FINAL_EVALUATED)的代时不把
            # phase 倒回 STOPPED(没有新 trial,交付结论仍有效)
            st_path = self.run_dir / "run-state.json"
            with locked_state_write(st_path):
                st = load_json(st_path) if st_path.exists() else {}
                _keep_final = already and st.get("phase") == "FINAL_EVALUATED"
                st.update({"phase": ("FINAL_EVALUATED" if _keep_final
                                     else "STOPPED"),
                           "stop_reason": stop_report.get("stop_reason"),
                           "status_updated_at": now_iso()})
                dump_json(st_path, st)

        # v2.15.6:warm_start_validation — 验证 gen-00 历史 top-trial 配方在本
        # run 的有效性。判据是 delta_vs_baseline 的统计显著性(CI 下界 > 0
        # 则 gain_confirmed);reference_auc 仅供展示,不参与任何裁决。
        # AUC 绝对值会随样本日期变化,但 delta_vs_baseline 对此天然免疫
        # (baseline 与 warm_start trial 使用同一切分)。
        warm_start_validation: list[dict] | None = None
        _ws_trials = [t for t in trials
                      if str(t.data.get("search_action") or "") == "warm_start"]
        if _ws_trials:
            warm_start_validation = []
            _baseline_auc = baseline_metric
            for _wt in _ws_trials:
                _ev = _wt.data.get("evaluation") or {}
                _pm = (_ev.get("primary_metric") or {}).get("value")
                _delta_bl = (_ev.get("delta") or {}).get("vs_baseline")
                _pb = (_ev.get("delta") or {}).get(
                    "vs_baseline_paired_bootstrap") or {}
                _ci = _pb.get("delta_ci")
                _p_pos = _pb.get("p_positive")
                # status 判定:仅看 delta_vs_baseline 的 CI 下界是否 > 0
                if _ci and len(_ci) == 2 and _ci[0] is not None:
                    if float(_ci[0]) > 0:
                        _status = "gain_confirmed"
                    elif _p_pos is not None and float(_p_pos) >= 0.5:
                        _status = "inconclusive"
                    else:
                        _status = "no_gain"
                elif _delta_bl is not None:
                    # 无 bootstrap CI 时退化为点值口径
                    _status = "gain_confirmed" if float(_delta_bl) > 0 \
                        else "no_gain"
                else:
                    _status = "inconclusive"
                # reference_auc:来源配方在库内的最新测量。warm trial 创建时已记进
                # execution.warm_start(v2.18.0 取自经验库榜单);缺失时按 recipe_id /
                # source_trial_id 在本 run 固定水位的视图里查(不再按 rankN 读目录)
                _ws = (_wt.data.get("execution") or {}).get("warm_start") or {}
                _ref_auc: float | None = _ws.get("reference_auc")
                if _ref_auc is None:
                    try:
                        from . import library_view
                        for _row in library_view.load(self.run_dir).leaderboard:
                            if _row.get("recipe_id") == _ws.get("recipe_id") or (
                                    _ws.get("source_trial_id") and
                                    _row.get("source_trial_id") == _ws.get("source_trial_id")):
                                _ref_auc = (_row.get("measurement") or {}).get("value")
                                break
                    except Exception:  # noqa: BLE001
                        pass
                _action = ("promoted_to_strong_anchor"
                           if _status == "gain_confirmed" else "excluded")
                warm_start_validation.append({
                    "trial_id": _wt.trial_id,
                    "status": _status,
                    "delta_vs_baseline": _delta_bl,
                    "delta_vs_baseline_ci": list(_ci) if _ci else None,
                    "p_positive": _p_pos,
                    "baseline_auc": _baseline_auc,
                    "actual_auc": _pm,
                    "reference_auc": _ref_auc,   # 仅供展示,不参与裁决
                    "verdict": (_wt.data.get("evaluation") or {}).get("verdict"),
                    "action": _action,
                    "note": (
                        "reference_auc 仅用于显示,AUC 绝对值受样本日期影响,"
                        "不参与 status 判定"),
                })
                logger.info(
                    "warm_start_validation rank trial %s: status=%s "
                    "delta_vs_baseline=%.5f",
                    _wt.trial_id, _status, _delta_bl or 0.0)

        gen_rec = GenerationRecord(self.run_dir, gid)
        gen_rec.write_summary(
            planned_trials=planned_trials,
            best_trial=best.trial_id if best else None,
            best_metric=best_metric,
            best_delta_vs_previous_generation=delta_gen,
            reflection=reflection or empty_reflection(),
            archive_update=archive_update,
            stop_report=stop_report,
            shadow_check=shadow_check,
            budget_metrics={
                "wall_time_hours": round(wall_time_hours, 3),
                "compute_hours": (round(float(compute_hours), 3)
                                  if compute_hours is not None else None),
                # v2.17.2(B5):收尾用时分列(compute_hours 已含)
                "trial_compute_hours": round(
                    float(_compute_acct["trial_compute_hours"]), 3),
                "finalize_hours": round(
                    float(_compute_acct["finalize_hours"]), 3),
                "selection_correction": fdr_report,
            },
            feature_beam=feature_beam,
            warm_start_validation=warm_start_validation,
        )
        # 方向台账(v11):finalize 后刷新,供下一代 planner-context
        try:
            from .directions import DirectionLedger

            ledger = DirectionLedger(self.run_dir)
            ledger.refresh(trials, self.spec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("方向台账刷新失败: %s", exc)
        # v18/P1-3:heavy/data_view 配额对账(确定性,替代纯文档纪律)。
        # 连续未投代数累计(本代 heavy_now==0 则 +1,投了清零),≥2 时
        # 下一代创建口有硬约束依据;核对行进 stop_report 与本代摘要。
        try:
            _min_heavy = int(self.config.get(
                "evolution.heavy_trials.min_per_generation", 1))
        except Exception:  # noqa: BLE001
            _min_heavy = 1
        if heavy_now >= _min_heavy:
            self.state["heavy_quota_miss_streak"] = 0
        else:
            self.state["heavy_quota_miss_streak"] = int(
                self.state.get("heavy_quota_miss_streak") or 0) + 1
        heavy_quota = {
            "heavy_in_generation": heavy_now,
            "min_required": _min_heavy,
            "miss_streak": self.state.get("heavy_quota_miss_streak"),
            "next_gen_force": (
                (self.state.get("heavy_quota_miss_streak") or 0) >= 2),
        }
        # ---- v19.4:代内并行度统计(P0-1 守护) ----
        # 口径:本代已启动 trial 的 wall 区间(首 trial 启动 → 末 trial
        # 评估完成)与 Σruntime。avg_parallelism = Σruntime / wall——
        # 纯串行 ≈ 1,完全并行 ≈ trial 数。**不用**"有 trial 在跑的
        # 时长/代跨度":纯串行时该值 ≈ 1.0 反而"通过",检不出串行
        # (实测 gen-02 四 trial 全串行,该口径 0.98)。
        parallelism: dict | None = None
        try:
            import datetime as _dt

            spans = []
            for t in trials:
                if t.generation != gid:
                    continue
                hist = t.data.get("status_history") or []
                # RUNNING 可多次出现(重试/evaluate 恢复);占用区间 =
                # 首个 RUNNING → 最后一个终态,与"机器被本 trial 占了
                # 多久"语义一致(含评估段;runtime_sec 只算训练,用它
                # 会把并行度系统性低估——实测 gen-02 真值 3.3 被算成
                # 0.22,因为 wall≈200min 而训练 wall 各仅 ~11min)
                t0 = next((h.get("at") for h in hist
                           if h.get("state") == "RUNNING"), None)
                # 首个终态(原始终态):backfill/evaluate 恢复会在尾部
                # 追加第二次 EVALUATED,取 reversed 最后一个会把 span
                # 撑到重评时刻(实测 gen-02 avg 从真值 ~1.0 假高到 3.6)
                t1 = next((h.get("at") for h in hist
                           if h.get("state") in ("EVALUATED", "FAILED",
                                                 "TIMEOUT")), None)
                if t0 and t1:
                    d0 = _dt.datetime.fromisoformat(t0)
                    d1 = _dt.datetime.fromisoformat(t1)
                    w = (d1 - d0).total_seconds()
                    if w > 0:
                        spans.append((d0, d1, w))
            if len(spans) >= 2:
                wall = (max(d1 for _, d1, _ in spans)
                        - min(d0 for d0, _, _ in spans)).total_seconds()
                total = sum(w for _, _, w in spans)
                if wall > 0:
                    parallelism = {
                        "n_trials_started": len(spans),
                        "wall_sec": round(wall, 1),
                        "sum_trial_wall_sec": round(total, 1),
                        "avg_parallelism": round(total / wall, 2),
                    }
                    if parallelism["avg_parallelism"] < 1.3:
                        # v2.17.0 挂钩(b):资源准入摘要(WP5a 实现;函数
                        # 不存在时静默跳过)。throttled=True ⇒ 串行是资源
                        # 受限(内存/共享卡准入),不是编排问题。
                        _adm = None
                        try:
                            from . import scheduler as _scheduler

                            if hasattr(_scheduler, "admission_summary"):
                                _adm = _scheduler.admission_summary(
                                    self.run_dir, generation=gid)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("admission_summary 失败(忽略):"
                                           " %s", exc)
                            _adm = None
                        if isinstance(_adm, dict):
                            parallelism["admission_summary"] = _adm
                        if isinstance(_adm, dict) and _adm.get("throttled"):
                            parallelism["throttled"] = True
                            parallelism["warning"] = (
                                "avg_parallelism<1.5:资源受限串行(非编排"
                                "问题),反思无需答辩"
                                f"(准入等待 {_adm.get('waits_sec')}s;"
                                f"{_adm.get('reason')})")
                            logger.info("gen-%s %s", gid,
                                        parallelism["warning"])
                        else:
                            parallelism["warning"] = (
                                "avg_parallelism<1.5:本代 trial 近乎全串行——"
                                "代内 ≥2 个就绪 trial 应走 run-generation 批量"
                                "提交(线程池+多卡),逐 trial 后台 run + 前台"
                                "wait 的编排是串行化的根源;反思必答本条")
                            logger.warning("gen-%s %s", gid,
                                           parallelism["warning"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("代内并行度统计失败(忽略): %s", exc)
        # v2.17.0:并行度(含准入摘要)写进 generation.yaml(此前只在返回值里)
        if parallelism is not None:
            try:
                gen_rec.add_fields(parallelism=parallelism)
            except Exception as exc:  # noqa: BLE001
                logger.warning("parallelism 写入 generation.yaml 失败: %s",
                               exc)
        # v2.17.0(D3):代际 delta 口径与空代标记写进 generation.yaml
        try:
            gen_rec.add_fields(
                empty_generation=bool(empty_generation),
                empty_generation_warning=empty_generation_warning,
                generation_delta_protocol=(
                    "champion_vs_champion" if _champion_protocol
                    else "legacy_best_vs_prev_generation_best"),
                champion_delta=champion_entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning("generation.yaml 代际 delta 口径写入失败: %s", exc)
        self.save_state()
        return {"archive_update": archive_update, "stop_report": stop_report,
                "best_metric": best_metric, "generation_delta": delta_gen,
                "empty_generation": bool(empty_generation),
                "empty_generation_warning": empty_generation_warning,
                "generation_delta_protocol": (
                    "champion_vs_champion" if _champion_protocol
                    else "legacy_best_vs_prev_generation_best"),
                "champion_delta": champion_entry,
                "data_views_tried": data_views,
                "heavy_trials_in_generation": heavy_now,
                "heavy_quota_check": heavy_quota,
                "parallelism": parallelism,
                "shadow_check": shadow_check,
                "warm_start_validation": warm_start_validation,
                "undelivered_gain": undelivered,
                "stop_superseded": stop_superseded,
                "selection_correction": fdr_report}
