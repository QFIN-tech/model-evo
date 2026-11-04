"""ModelEvo 端到端质量基准与发布门。

基准不以“跑通”作为成功，而比较固定预算下基线、框架产物与预先登记
的参考上界。输入必须来自未参与搜索的评测集，且记录泄漏/复现/资源证据。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .logging_utils import dump_json, load_json, load_yaml, now_iso


def _load(path: Path) -> dict:
    return load_json(path) if path.suffix.lower() == ".json" \
        else (load_yaml(path) or {})


def _improvement(score: float, baseline: float, direction: str) -> float:
    return score - baseline if direction == "maximize" else baseline - score


def evaluate_benchmark(payload: dict, *, thresholds: dict | None = None) -> dict:
    thresholds = {
        "median_lift_capture": 0.90,
        "p10_lift_capture": 0.75,
        "reproducibility_rate": 0.95,
        "within_mde_rate": 0.90,
        "delivery_rate": 0.95,
        "max_final_test_touches": 1,
        "min_tasks": 6,
        "min_model_types": 3,
        "min_search_seeds": 3,
        **(thresholds or {}),
    }
    rows = []
    for task in payload.get("tasks") or []:
        required = ("task_id", "model_type", "baseline_score",
                    "achieved_score", "reference_score", "compute_hours",
                    "reference_compute_hours", "compute_budget_hours", "mde",
                    "search_seeds", "delivered")
        missing = [k for k in required if task.get(k) is None]
        if missing:
            rows.append({"task_id": task.get("task_id"), "valid": False,
                         "errors": [f"missing:{k}" for k in missing]})
            continue
        direction = str(task.get("direction") or "maximize")
        if direction not in ("maximize", "minimize"):
            rows.append({"task_id": task.get("task_id"), "valid": False,
                         "errors": ["invalid_direction"]})
            continue
        baseline = float(task["baseline_score"])
        achieved = float(task["achieved_score"])
        reference = float(task["reference_score"])
        achieved_gain = _improvement(achieved, baseline, direction)
        reference_gain = _improvement(reference, baseline, direction)
        if reference_gain <= 0:
            capture = None
            errors = ["reference_not_better_than_baseline"]
        else:
            capture = achieved_gain / reference_gain
            errors = []
        regret = _improvement(reference, achieved, direction)
        leakage = bool(task.get("leakage_detected", False))
        invalid_champion = bool(task.get("invalid_champion", False))
        touches = int(task.get("final_test_touches", 0) or 0)
        reproducible = bool(task.get("reproducible", False))
        compute_hours = float(task["compute_hours"])
        reference_compute_hours = float(task["reference_compute_hours"])
        compute_budget_hours = float(task["compute_budget_hours"])
        mde = float(task["mde"])
        search_seeds = int(task["search_seeds"])
        delivered = bool(task["delivered"])
        if leakage:
            errors.append("leakage_detected")
        if invalid_champion:
            errors.append("invalid_champion")
        if task.get("reference_preregistered") is not True:
            errors.append("reference_not_preregistered")
        if task.get("independent_test") is not True:
            errors.append("test_not_independent")
        if touches > int(thresholds["max_final_test_touches"]):
            errors.append("final_test_reused")
        if touches != 1:
            errors.append("final_test_touch_count_must_equal_one")
        if compute_budget_hours <= 0 or compute_hours < 0 or \
                reference_compute_hours < 0:
            errors.append("invalid_compute_budget")
        if compute_hours > compute_budget_hours + 1e-12:
            errors.append("modelevo_compute_budget_exceeded")
        if reference_compute_hours > compute_budget_hours + 1e-12:
            errors.append("reference_compute_budget_exceeded")
        if search_seeds < int(thresholds["min_search_seeds"]):
            errors.append("insufficient_search_seeds")
        if mde <= 0:
            errors.append("invalid_mde")
        if not delivered:
            errors.append("delivery_failed")
        rows.append({
            "task_id": task["task_id"], "valid": not errors,
            "model_type": task.get("model_type"),
            "direction": direction, "baseline_score": baseline,
            "achieved_score": achieved, "reference_score": reference,
            "achieved_gain": achieved_gain,
            "reference_gain": reference_gain,
            "lift_capture": capture, "regret": regret,
            "compute_hours": compute_hours,
            "reference_compute_hours": reference_compute_hours,
            "compute_budget_hours": compute_budget_hours,
            "mde": mde,
            "within_mde": bool(regret <= mde) if mde > 0 else False,
            "search_seeds": search_seeds,
            "delivered": delivered,
            "reproducible": reproducible,
            "final_test_touches": touches,
            "errors": errors,
        })
    captures = np.asarray([r["lift_capture"] for r in rows
                           if r.get("lift_capture") is not None], dtype=float)
    reproducibility = (sum(bool(r.get("reproducible")) for r in rows)
                       / len(rows)) if rows else 0.0
    within_mde = (sum(bool(r.get("within_mde")) for r in rows)
                  / len(rows)) if rows else 0.0
    delivery_rate = (sum(bool(r.get("delivered")) for r in rows)
                     / len(rows)) if rows else 0.0
    model_types = {r.get("model_type") for r in rows if r.get("model_type")}
    task_ids = [r.get("task_id") for r in rows]
    protocol_cfg = payload.get("protocol") or {}
    valid_protocol = (all(r.get("valid") for r in rows) and bool(rows)
                      and len(set(task_ids)) == len(task_ids)
                      and protocol_cfg.get("fixed_compute_budget") is True)
    aggregate = {
        "n_tasks": len(rows),
        "median_lift_capture": float(np.median(captures))
        if len(captures) else None,
        "p25_lift_capture": float(np.percentile(captures, 25))
        if len(captures) else None,
        "p10_lift_capture": float(np.percentile(captures, 10))
        if len(captures) else None,
        "mean_regret": float(np.mean([
            r["regret"] for r in rows if r.get("regret") is not None]))
        if any(r.get("regret") is not None for r in rows) else None,
        "reproducibility_rate": reproducibility,
        "within_mde_rate": within_mde,
        "delivery_rate": delivery_rate,
        "n_model_types": len(model_types),
        "protocol_valid": valid_protocol,
        "min_tasks": len(rows) >= int(thresholds["min_tasks"]),
        "min_model_types": len(model_types) >= int(
            thresholds["min_model_types"]),
    }
    checks = {
        "protocol_valid": valid_protocol,
        "min_tasks": aggregate["min_tasks"],
        "min_model_types": aggregate["min_model_types"],
        "median_lift_capture": aggregate["median_lift_capture"] is not None
        and aggregate["median_lift_capture"] >=
        float(thresholds["median_lift_capture"]),
        "p10_lift_capture": aggregate["p10_lift_capture"] is not None
        and aggregate["p10_lift_capture"] >=
        float(thresholds["p10_lift_capture"]),
        "reproducibility_rate": reproducibility >=
        float(thresholds["reproducibility_rate"]),
        "within_mde_rate": within_mde >=
        float(thresholds["within_mde_rate"]),
        "delivery_rate": delivery_rate >=
        float(thresholds["delivery_rate"]),
    }
    return {
        "schema_version": 1, "evaluated_at": now_iso(),
        "suite_id": payload.get("suite_id"),
        "thresholds": thresholds, "tasks": rows,
        "aggregate": aggregate, "quality_gate": {
            "passed": all(checks.values()), "checks": checks},
    }


def run_benchmark(results_path: Path, output_path: Path | None = None,
                  thresholds: dict | None = None) -> dict:
    payload = _load(Path(results_path))
    report = evaluate_benchmark(payload, thresholds=thresholds)
    output = Path(output_path) if output_path else \
        Path(results_path).with_name("benchmark-report.json")
    dump_json(output, report)
    report["output_path"] = str(output)
    return report


# ---------------------------------------------------------------------------
# v2.19.0:套件报告(benchmark report --suite --runs-root)
# ---------------------------------------------------------------------------
# 每个数据集一个 run 目录(<runs-root>/<name>,由 scripts/benchmark/prepare.py
# 生成)。三把分数都在 validation 面上取,口径一致:
#   S_base = run-state baseline_id 对应 trial 的 validation 主指标;
#   S_mvo  = 交付 final-model/evaluation.json 的 validation(搜索冠军);未交付
#            时退回 verdict ∈ {accepted, confirmed} 的最优 trial(delivered=False);
#            shadow_test 若存在一并记录(交付模型的第二把尺子);
#   S_ref  = ceiling/ceiling-estimate.json(runtime.ceiling 读侧)。
# lift_capture / gap_to_ref 走 runtime.ceiling;发布质量门复用 evaluate_benchmark。

def _trial_primary(trial_dir: Path, metric: str) -> float | None:
    ev = trial_dir / "evaluation.json"
    if not ev.exists():
        return None
    try:
        pm = (load_json(ev).get("primary_metric") or {})
    except Exception:  # noqa: BLE001
        return None
    if str(pm.get("name", "")).lower() != metric:
        return None
    try:
        return float(pm.get("value"))
    except (TypeError, ValueError):
        return None


def collect_run_scores(run_dir: Path, metric: str,
                       higher_is_better: bool) -> dict:
    """单个 run 的 S_base / S_mvo / shadow / 计算量 / 交付状态(只读)。"""
    run_dir = Path(run_dir)
    out: dict[str, Any] = {"run_dir": str(run_dir), "exists": run_dir.is_dir()}
    if not run_dir.is_dir():
        return out
    st = load_json(run_dir / "run-state.json") if (
        run_dir / "run-state.json").exists() else {}
    bid = st.get("baseline_id")
    out["phase"] = st.get("phase") or st.get("status")
    out["baseline_trial"] = bid
    out["S_base"] = _trial_primary(run_dir / "trials" / bid, metric) \
        if bid else None
    fm = run_dir / "final-model" / "evaluation.json"
    s_mvo, src, shadow, champ = None, None, None, None
    if fm.exists():
        try:
            ev = load_json(fm)
            v = (ev.get("validation") or {}).get(metric)
            s_mvo = float(v) if v is not None else None
            # v2.19.0:交付模型 = 交付配方(delivery_recipe)的重训;
            # final-model 的 validation 记的是确认子代(换 seed)的分数,
            # bike_sharing 实测两者差 0.79 rmse。S_mvo 取配方本身的分数
            _rec = (ev.get("delivery_resolution") or {}).get(
                "delivery_recipe")
            if _rec:
                _rv = _trial_primary(run_dir / "trials" / _rec, metric)
                if _rv is not None:
                    s_mvo = _rv
            shadow = (ev.get("shadow_test") or {}).get(metric)
            champ = ev.get("best_trial") or ev.get("search_champion")
            src = "final-model"
        except Exception:  # noqa: BLE001
            pass
    delivered = s_mvo is not None
    rt_sec, n_trials, reproducible = 0.0, 0, False
    best_tid, best_val = None, None
    tdir = run_dir / "trials"
    for t in sorted(tdir.iterdir()) if tdir.is_dir() else []:
        evp = t / "evaluation.json"
        if not evp.exists():
            continue
        n_trials += 1
        try:
            ev = load_json(evp)
        except Exception:  # noqa: BLE001
            continue
        rt = (ev.get("resource") or {}).get("runtime_sec")
        if isinstance(rt, (int, float)):
            rt_sec += float(rt)
        verdict = str(ev.get("verdict") or "")
        if verdict == "confirmed" and (champ is None or t.name == champ):
            reproducible = True
        v = _trial_primary(t, metric)
        # baseline 自身不是搜索结果:只有 baseline 时 S_mvo 记缺失(no_search),
        # 否则 lift_capture 会被记成 0 而不是"尚未搜索"
        if v is None or verdict not in ("accepted", "confirmed") \
                or t.name == bid:
            continue
        if best_val is None or (v > best_val if higher_is_better
                                else v < best_val):
            best_tid, best_val = t.name, v
    if s_mvo is None and best_val is not None:
        s_mvo, src, champ = best_val, "best_accepted_trial", best_tid
    gate = load_json(run_dir / "final-test-gate.json") if (
        run_dir / "final-test-gate.json").exists() else {}
    out.update({
        "S_mvo": s_mvo, "S_mvo_source": src, "champion": champ,
        "shadow_test": float(shadow) if shadow is not None else None,
        "delivered": delivered, "reproducible": reproducible,
        "n_trials": n_trials, "compute_hours": rt_sec / 3600.0,
        "final_test_touches": int(gate.get("count", 0) or 0),
    })
    return out


def _mde_for(metric: str, base: float | None, mde_cfg: dict) -> float:
    if metric in ("auc", "ks", "gini", "auprc"):
        return float(mde_cfg.get(metric, mde_cfg.get("auc", 0.002)))
    rel = float(mde_cfg.get(f"{metric}_rel", mde_cfg.get("rmse_rel", 0.005)))
    return abs(float(base)) * rel if base else rel


def _pct(xs: list[float], q: float) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


def build_suite_report(suite: dict, runs_root: Path, *,
                       tier: str | None = None,
                       out_dir: Path | None = None,
                       thresholds: dict | None = None) -> dict:
    """汇总套件下每个 run 的 S_base/S_mvo/S_ref,写 benchmark-report.json/.md。"""
    from .ceiling import gap_to_ref, lift_capture, load_ceiling
    from .metrics import metric_direction

    runs_root = Path(runs_root)
    out_dir = Path(out_dir) if out_dir else runs_root
    mde_cfg = suite.get("mde_default") or {}
    budget = (suite.get("budget") or {}).get(tier or "fast") or {}
    rows, gate_tasks = [], []
    for ds in suite.get("datasets") or []:
        if tier and tier not in (ds.get("tiers") or []):
            continue
        metric = str(ds.get("primary_metric", "auc")).lower()
        hib = bool(metric_direction(metric))
        run = runs_root / ds["name"]
        sc = collect_run_scores(run, metric, hib)
        ceil = load_ceiling(run) if run.is_dir() else None
        # v2.19.0:主参照取单模型口径(与交付同口径:单模型、无 bagging/集成);
        # 集成参照(ceiling_estimate)作为上限单列,两者之差 = 单模型约束代价
        s_ref_ens = ceil["ceiling_estimate"] if ceil else None
        s_ref_single = (ceil or {}).get("single_model_estimate")
        s_ref = s_ref_single if s_ref_single is not None else s_ref_ens
        ceil_single = ({**ceil, "ceiling_estimate": s_ref}
                       if ceil and s_ref is not None else None)
        row = {"task_id": ds["name"], "model_type": ds.get("model_type"),
               "primary_metric": metric, "higher_is_better": hib,
               "shape_tags": ds.get("shape_tags") or [], **sc,
               "S_ref": s_ref,
               "S_ref_basis": ("single_model" if s_ref_single is not None
                               else "ensemble" if s_ref_ens is not None
                               else None),
               "S_ref_ensemble": s_ref_ens,
               "single_model_reference": (ceil or {}).get(
                   "single_model_reference"),
               "best_reference": (ceil or {}).get("best_reference"),
               "feature_surface": (ceil or {}).get("feature_surface"),
               "reference_seconds": None}
        if ceil:
            row["reference_seconds"] = float(sum(
                float(r.get("seconds") or 0) for k, r in
                (ceil.get("references") or {}).items()
                if k != "strong_blend"))
        row["gap_to_ref"] = gap_to_ref(ceil_single, sc.get("S_mvo"), hib) \
            if ceil_single and sc.get("S_mvo") is not None else None
        row["gap_to_ensemble_ref"] = gap_to_ref(ceil, sc.get("S_mvo"), hib) \
            if ceil and sc.get("S_mvo") is not None else None
        row["baseline_gap_to_ref"] = gap_to_ref(
            ceil_single, sc.get("S_base"), hib) \
            if ceil_single and sc.get("S_base") is not None else None
        # v2.19.0:base 面参照(未加宽特征面)被 S_mvo 超过时,说明框架靠特征
        # 工程越过了"模型侧上限",此时 lift_capture/gap 没有意义(home_credit
        # 实测 lift_capture=21)。标记不可比,需在冠军面(--from-trial)重测
        _surface = (ceil or {}).get("feature_surface") or "base"
        row["ref_exceeded_on_base_surface"] = bool(
            _surface == "base" and sc.get("S_mvo") is not None
            and s_ref is not None and (
                sc["S_mvo"] > s_ref if hib else sc["S_mvo"] < s_ref))
        row["lift_capture_ensemble"] = lift_capture(
            base=sc.get("S_base"), achieved=sc.get("S_mvo"),
            reference=s_ref_ens, higher_is_better=hib) \
            if None not in (sc.get("S_base"), sc.get("S_mvo"),
                            s_ref_ens) else None
        row["lift_capture"] = lift_capture(
            base=sc.get("S_base"), achieved=sc.get("S_mvo"),
            reference=s_ref, higher_is_better=hib) \
            if None not in (sc.get("S_base"), sc.get("S_mvo"), s_ref) else None
        if row["ref_exceeded_on_base_surface"]:
            row["lift_capture_base_surface"] = row["lift_capture"]
            row["lift_capture"] = None
            row["lift_capture_ensemble"] = None
        row["mde"] = _mde_for(metric, sc.get("S_base"), mde_cfg)
        row["status"] = ("missing_run" if not sc.get("exists") else
                         "no_baseline" if sc.get("S_base") is None else
                         "no_reference" if s_ref is None else
                         "no_search" if sc.get("S_mvo") is None else
                         "ref_below_mvo_base_surface"
                         if row["ref_exceeded_on_base_surface"] else "ok")
        rows.append(row)
        if row["status"] == "ok":
            gate_tasks.append({
                "task_id": ds["name"], "model_type": ds.get("model_type"),
                "direction": "maximize" if hib else "minimize",
                "baseline_score": sc["S_base"], "achieved_score": sc["S_mvo"],
                "reference_score": s_ref,
                "compute_hours": sc.get("compute_hours", 0.0),
                "reference_compute_hours": (row["reference_seconds"] or 0) / 3600,
                "compute_budget_hours": float(budget.get("wall_hours", 1.0)),
                "mde": row["mde"], "search_seeds": 1,
                "delivered": bool(sc.get("delivered")),
                "reproducible": bool(sc.get("reproducible")),
                # 参照在搜索前/独立于搜索计算才算预登记;validation 面参与了
                # 搜索,不是独立测试面 —— 如实标 False,质量门会据此不通过
                "reference_preregistered": True,
                "independent_test": False,
                "final_test_touches": sc.get("final_test_touches", 0)})
    caps = [r["lift_capture"] for r in rows if r["lift_capture"] is not None]
    gaps = [r["gap_to_ref"] for r in rows if r["gap_to_ref"] is not None]
    gate = evaluate_benchmark({"suite_id": suite.get("suite_id"),
                               "protocol": {"fixed_compute_budget": True},
                               "tasks": gate_tasks}, thresholds=thresholds)
    report = {
        "schema_version": 1, "kind": "suite_report",
        "generated_at": now_iso(), "suite_id": suite.get("suite_id"),
        "tier": tier, "runs_root": str(runs_root),
        "surface": "validation",
        "tasks": rows,
        "summary": {
            "n_tasks": len(rows),
            "n_complete": sum(r["status"] == "ok" for r in rows),
            "lift_capture_median": _pct(caps, 50),
            "lift_capture_p10": _pct(caps, 10),
            "gap_to_ref_median": _pct(gaps, 50),
            "gap_to_ref_p90": _pct(gaps, 90),
            "delivery_rate": (sum(bool(r.get("delivered")) for r in rows)
                              / len(rows)) if rows else 0.0,
        },
        "quality_gate": gate["quality_gate"],
        "quality_gate_detail": {"aggregate": gate["aggregate"],
                                "tasks": gate["tasks"]},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "benchmark-report.json", report)
    (out_dir / "benchmark-report.md").write_text(
        render_suite_markdown(report), encoding="utf-8")
    report["output_path"] = str(out_dir / "benchmark-report.json")
    return report


def _f(v, nd: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def render_suite_markdown(report: dict) -> str:
    s = report.get("summary") or {}
    lines = [
        f"# Benchmark 报告 {report.get('suite_id')}({report.get('tier') or 'all'})",
        "",
        f"生成于 {report.get('generated_at')};三把分数都在 validation 面上取。"
        "S_ref = 单模型口径参照(与交付同口径);S_ref_ens = 集成/bagging 参照"
        "(上限)。lift_capture = (S_mvo − S_base)/(S_ref − S_base);gap 为正 = "
        "离参照还差多少(主指标单位,方向已对齐)。",
        "",
        "| 数据集 | 指标 | S_base | S_mvo | S_ref(单模型) | S_ref_ens(集成) "
        "| 面 | gap_to_ref | lift_capture | lift_capture_ens | 状态 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report.get("tasks") or []:
        lines.append(
            f"| {r['task_id']} | {r['primary_metric']} | {_f(r.get('S_base'))} "
            f"| {_f(r.get('S_mvo'))} | {_f(r.get('S_ref'))} "
            f"| {_f(r.get('S_ref_ensemble'))} "
            f"| {r.get('feature_surface') or '—'} | {_f(r.get('gap_to_ref'))} "
            f"| {_f(r.get('lift_capture'), 3)} "
            f"| {_f(r.get('lift_capture_ensemble'), 3)} | {r['status']} |")
    g = report.get("quality_gate") or {}
    lines += [
        "",
        f"- 完成 {s.get('n_complete')}/{s.get('n_tasks')};lift_capture 中位 "
        f"{_f(s.get('lift_capture_median'), 3)}、P10 {_f(s.get('lift_capture_p10'), 3)};"
        f"gap_to_ref 中位 {_f(s.get('gap_to_ref_median'))}、P90 "
        f"{_f(s.get('gap_to_ref_p90'))};交付率 {_f(s.get('delivery_rate'), 2)}",
        f"- 发布质量门:passed={g.get('passed')}(未通过项:"
        f"{', '.join(k for k, v in (g.get('checks') or {}).items() if not v) or '无'})",
        "",
        "状态:missing_run = run 目录不存在;no_baseline = baseline 未出分;"
        "no_reference = 未跑 ceiling;no_search = 尚无 accepted/交付冠军;"
        "ref_below_mvo_base_surface = base 面参照被框架超过(靠特征工程),"
        "lift_capture 不可比,需 --from-trial 在冠军面重测。",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# v2.19.0:对齐测试面评分(benchmark aligned)
# ---------------------------------------------------------------------------
#
# 所有参与者在**同一测试面、同一批样本**上评分:标签只在这里读一次(只读,
# 不回写任何 run 产物),每个参与者的预测按 __mvo_sample_key 与标签面核验
# ——行数相等、无重复键、键集合完全相同,任一不满足即判该数据集
# misaligned、不出分。框架侧取 trial 自己落盘的 train-only 测试面分数;
# 参照取 ``ceiling --holdout`` 落盘的预测(同样只用 train 训练)。
# 交付模型(train+validation 重训)训练数据不同,单列展示、不参与排名。

ALIGNED_FRAMEWORK = "ModelEvo 交付配方"
ALIGNED_BASELINE = "ModelEvo baseline"
ALIGNED_DELIVERED = "ModelEvo 交付模型(train+val 重训)"

#: 参照落盘名 → 报告名
REF_DISPLAY = {
    "strong_xgboost": "strong XGB(5-seed bag)",
    "strong_xgboost__single": "strong XGB(单模型)",
    "strong_lightgbm": "strong LGBM(5-seed bag)",
    "strong_lightgbm__single": "strong LGBM(单模型)",
    "strong_blend": "strong XGB+LGBM 混合",
    "autogluon_best_quality": "AutoGluon 集成",
    "autogluon_best_quality__single": "AutoGluon 最佳单模型",
    "flaml": "FLAML(单模型)",
    "tabpfn_v2": "TabPFN v2(单模型)",
}


def _load_scored(path: Path) -> "tuple[np.ndarray, np.ndarray]":
    import pandas as pd

    df = pd.read_parquet(path, columns=["score", "__mvo_sample_key"])
    return (df["__mvo_sample_key"].astype(str).to_numpy(),
            df["score"].to_numpy(dtype="float64"))


def _check_alignment(name: str, keys: np.ndarray, truth_keys: np.ndarray,
                     truth_index: dict) -> "tuple[np.ndarray | None, str | None]":
    """核验并返回按 truth 行序重排的位置;不对齐时返回 (None, 原因)。"""
    if len(keys) != len(truth_keys):
        return None, f"{name}: 行数 {len(keys)} != 测试面 {len(truth_keys)}"
    if len(set(keys)) != len(keys):
        return None, f"{name}: 样本键有重复"
    missing = [k for k in keys if k not in truth_index]
    if missing:
        return None, f"{name}: {len(missing)} 个样本键不在测试面(例 {missing[0]})"
    order = np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        order[truth_index[k]] = i
    return order, None


def aligned_holdout_report(suite: dict, runs_root: Path, *,
                           split: str = "shadow_test",
                           tier: str | None = None,
                           out_dir: Path | None = None,
                           n_boot: int = 2000,
                           extra_ceilings: dict | None = None,
                           extra_participants: dict | None = None) -> dict:
    import hashlib

    from .data import SAMPLE_KEY_COLUMN, load_identities, load_labels
    from .metrics import compute_metric, metric_direction, paired_bootstrap_delta
    from .spec import TaskSpec

    runs_root = Path(runs_root)
    out_dir = Path(out_dir) if out_dir else runs_root / "aligned-report"
    datasets = []
    for ds in suite.get("datasets") or []:
        if tier and tier not in (ds.get("tiers") or []):
            continue
        run = runs_root / ds["name"]
        row: dict[str, Any] = {"task_id": ds["name"], "split": split,
                               "shape_tags": ds.get("shape_tags") or []}
        datasets.append(row)
        try:
            # 只取目标列与主指标:评分不需要完整的 spec 校验
            spec = TaskSpec(data=load_yaml(run / "task-spec.yaml") or {})
            metric = spec.primary_metric
            hib = bool(metric_direction(metric))
            row.update(metric=metric, higher_is_better=hib)
            labs = load_labels(run, splits=(split,)).get(split)
            ids = load_identities(run, splits=(split,)).get(split)
            if labs is None or ids is None:
                row["status"] = "no_holdout"
                continue
            y = labs[spec.target_name].to_numpy()
            truth_keys = ids[SAMPLE_KEY_COLUMN].astype(str).to_numpy()
            if len(truth_keys) != len(y) or len(set(truth_keys)) != len(y):
                row["status"] = "misaligned"
                row["alignment_errors"] = ["标签面与 identity 面行数/键不一致"]
                continue
            truth_index = {k: i for i, k in enumerate(truth_keys)}
            row["n_rows"] = int(len(y))
            row["keys_sha256_16"] = hashlib.sha256(
                "\n".join(sorted(truth_keys)).encode()).hexdigest()[:16]
            st = load_json(run / "run-state.json")
            fm = load_json(run / "final-model" / "evaluation.json") if (
                run / "final-model" / "evaluation.json").exists() else {}
            bid = st.get("baseline_id")
            rec = (fm.get("delivery_resolution") or {}).get("delivery_recipe")
            cands: list[tuple[str, Path, dict]] = []
            if bid:
                cands.append((ALIGNED_BASELINE,
                              run / "trials" / bid / f"{split}_score.parquet",
                              {"trained_on": "train", "kind": "single_model",
                               "trial": bid, "family": "modelevo"}))
            if rec:
                cands.append((ALIGNED_FRAMEWORK,
                              run / "trials" / rec / f"{split}_score.parquet",
                              {"trained_on": "train", "kind": "single_model",
                               "trial": rec, "family": "modelevo",
                               "same_as_baseline": rec == bid}))
            cands.append((ALIGNED_DELIVERED,
                          run / "final-model" / f"{split}_score.parquet",
                          {"trained_on": "train+validation",
                           "kind": "single_model", "family": "modelevo",
                           "ranked": False}))
            ceil = load_json(run / "ceiling" / "ceiling-estimate.json") if (
                run / "ceiling" / "ceiling-estimate.json").exists() else {}
            # v2.19.0:另一特征面的参照(例如冠军面之外再测原始特征面,
            # extra_ceilings = {标签: 根目录},读 <根>/<数据集>/ceiling/)
            ceil_sets = [("", ceil)]
            for label, root in (extra_ceilings or {}).items():
                f = Path(root) / ds["name"] / "ceiling" / "ceiling-estimate.json"
                if f.exists():
                    ceil_sets.append((label, load_json(f)))
            for label, cobj in ceil_sets:
                ho = (cobj.get("holdout") or {})
                if ho.get("split") != split:
                    continue
                tag = f"[{label}:{cobj.get('feature_surface')}]" if label else ""
                for name, meta in (ho.get("files") or {}).items():
                    cands.append((REF_DISPLAY.get(name, name) + tag,
                                  Path(meta["file"]),
                                  {"trained_on": meta.get("trained_on", "train"),
                                   "kind": meta.get("kind"), "ref": name,
                                   "family": "reference",
                                   "feature_surface": cobj.get(
                                       "feature_surface")}))
            # v2.19.1:外部对照基线(AIDE / 无框架 Agent 等)。目录布局
            # <根>/<数据集>/shadow_score.parquet(列 score + __mvo_sample_key);
            # 与其它参与者同一批行、同一口径参与排名。
            for label, root in (extra_participants or {}).items():
                f = Path(root) / ds["name"] / f"{split}_score.parquet"
                if not f.exists():
                    f = Path(root) / ds["name"] / "shadow_score.parquet"
                if f.exists():
                    # 训练口径以对照自己的 run.json 为准(AIDE 新协议是
                    # train+validation);非 train-only 的参与者不进名次
                    rj = Path(root) / ds["name"] / "run.json"
                    tr_on = (load_json(rj).get("trained_on") if rj.exists()
                             else None) or "train"
                    cands.append((f"{label}", f,
                                  {"trained_on": tr_on, "kind": "external",
                                   "family": label, "participant": label,
                                   "ranked": tr_on == "train"}))
            row["feature_surface"] = ceil.get("feature_surface")
            scores: dict[str, np.ndarray] = {}
            entrants: list[dict] = []
            errors: list[str] = []
            for name, path, meta in cands:
                if not path.exists():
                    errors.append(f"{name}: 缺预测文件 {path.name}")
                    continue
                keys, sc = _load_scored(path)
                order, err = _check_alignment(name, keys, truth_keys,
                                              truth_index)
                if err:
                    errors.append(err)
                    continue
                s_al = sc[order]
                scores[name] = s_al
                val = compute_metric(metric, y, s_al, score=s_al)
                entrants.append({"name": name, "value": float(val), **meta,
                                 "aligned": True})
            row["alignment_errors"] = errors
            if ALIGNED_FRAMEWORK not in scores:
                row["status"] = "misaligned" if errors else "no_framework"
                row["entrants"] = entrants
                continue
            sign = 1.0 if hib else -1.0
            for e in entrants:
                # 同训练口径才配对:train-only → 交付配方;
                # train+validation → 交付模型(train+val 重训)
                if e["name"] in (ALIGNED_FRAMEWORK, ALIGNED_DELIVERED):
                    continue        # 两组各自的 ModelEvo 基准
                ref = (ALIGNED_DELIVERED
                       if e.get("trained_on") == "train+validation"
                       else ALIGNED_FRAMEWORK)
                if ref not in scores:
                    e["vs_reference"] = ref
                    e["vs_framework"] = None
                    continue
                pb = paired_bootstrap_delta(y, scores[e["name"]], scores[ref],
                                            metric, n_boot=n_boot)
                lo, hi = pb.get("delta_ci") or (None, None)
                # 方向对齐:正数 = 该参与者比参照更好
                e["vs_reference"] = ref
                e["delta_vs_framework"] = sign * float(e["value"]
                                                       - _value_of(entrants, ref))
                ci = sorted([sign * lo, sign * hi]) if lo is not None else None
                e["delta_ci95"] = ci
                e["vs_framework"] = ("better" if ci and ci[0] > 0 else
                                     "worse" if ci and ci[1] < 0 else "tie")
            ranked = [e for e in entrants if e.get("ranked", True)]
            ranked.sort(key=lambda e: -e["value"] if hib else e["value"])
            for i, e in enumerate(ranked, 1):
                e["rank"] = i
            row["n_ranked"] = len(ranked)
            row["entrants"] = sorted(
                entrants, key=lambda e: (e.get("rank") is None,
                                         e.get("rank") or 0))
            row["status"] = "ok" if not errors else "ok_with_missing"
        except Exception as exc:  # noqa: BLE001 —— 单个数据集失败不阻塞汇总
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
    report = {"schema_version": 1, "kind": "aligned_holdout_report",
              "generated_at": now_iso(), "suite_id": suite.get("suite_id"),
              "split": split, "runs_root": str(runs_root),
              "datasets": datasets, "summary": _aligned_summary(datasets)}
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json(out_dir / "aligned-report.json", report)
    (out_dir / "aligned-report.md").write_text(
        render_aligned_markdown(report), encoding="utf-8")
    report["output_path"] = str(out_dir / "aligned-report.json")
    return report


#: 测试面行数 → 置信度档位(只是读数提示,不参与名次与胜负判定)
ALIGNED_CONFIDENCE_BANDS = ((10000, "高"), (1000, "中"), (0, "低"))


def _row_confidence(n_rows) -> str:
    if n_rows is None:
        return "—"
    return next(lab for lo, lab in ALIGNED_CONFIDENCE_BANDS if n_rows >= lo)


def _value_of(entrants: list[dict], name: str) -> float:
    return next(e["value"] for e in entrants if e["name"] == name)


def _aligned_summary(datasets: list[dict]) -> dict:
    """跨数据集:平均名次、框架对各参照的胜/平/负(95% CI 判定)。"""
    ok = [d for d in datasets if str(d.get("status", "")).startswith("ok")]
    ranks: dict[str, list[int]] = {}
    wtl: dict[str, dict[str, int]] = {}
    for d in ok:
        for e in d.get("entrants") or []:
            if e.get("rank") is not None:
                ranks.setdefault(e["name"], []).append(e["rank"])
            if e.get("vs_framework"):
                # 从框架视角:参照 better = 框架 loss
                k = {"better": "framework_loss", "worse": "framework_win",
                     "tie": "tie"}[e["vs_framework"]]
                wtl.setdefault(e["name"], {"framework_win": 0, "tie": 0,
                                           "framework_loss": 0,
                                           "reference": e.get("vs_reference")})[k] += 1
    avg_rank = sorted(({"name": n, "avg_rank": float(np.mean(v)),
                        "n_datasets": len(v)} for n, v in ranks.items()),
                      key=lambda r: r["avg_rank"])
    return {"n_datasets": len(datasets), "n_aligned": len(ok),
            "n_misaligned": sum(d.get("status") == "misaligned"
                                for d in datasets),
            "avg_rank": avg_rank, "framework_vs": wtl}


def render_aligned_markdown(report: dict) -> str:
    s = report.get("summary") or {}
    lines = [f"# 对齐测试面对比({report.get('split')})", "",
             f"生成于 {report.get('generated_at')}。所有参与者在同一测试面、同一批样本"
             "(按 __mvo_sample_key 逐键核验)上评分;除标注外均只用 train 训练。"
             "Δ 按训练口径配对:train-only 参与者对 ModelEvo 交付配方,train+validation 参与者对"
             "ModelEvo 交付模型(train+val 重训);方向已对齐(正 = 更好),CI 为配对 bootstrap 95%。",
             "", f"- 对齐核验通过 {s.get('n_aligned')}/{s.get('n_datasets')} 个数据集"
             f"(不对齐 {s.get('n_misaligned')})", "",
             "## 各数据集测试面规模", "",
             "置信度按测试面行数:≥10000 高,≥1000 中,其余低。只是读数提示,"
             "不参与名次与胜负判定。", "",
             "| 数据集 | 指标 | 测试面行数 | 置信度 | 状态 |", "|---|---|---|---|---|"]
    for d in report.get("datasets") or []:
        lines.append(f"| {d['task_id']} | {d.get('metric')} | {d.get('n_rows', '—')} "
                     f"| {_row_confidence(d.get('n_rows'))} | {d.get('status')} |")
    lines.append("")
    for d in report.get("datasets") or []:
        lines += [f"## {d['task_id']}({d.get('metric')},n={d.get('n_rows')},"
                  f"键指纹 {d.get('keys_sha256_16')},特征面 "
                  f"{d.get('feature_surface')},状态 {d.get('status')})", ""]
        if d.get("alignment_errors"):
            lines += [f"- 对齐/缺失:{'; '.join(d['alignment_errors'])}", ""]
        if not d.get("entrants"):
            continue
        lines += ["| 名次 | 参与者 | 训练数据 | 口径 | 分数 | 对照 | Δ | 95% CI | 判定 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for e in d["entrants"]:
            ci = e.get("delta_ci95")
            lines.append(
                f"| {e.get('rank', '—')} | {e['name']} | {e.get('trained_on')} "
                f"| {e.get('kind')} | {_f(e['value'])} "
                f"| {e.get('vs_reference') or '—'} "
                f"| {_f(e.get('delta_vs_framework'))} "
                f"| {('[' + _f(ci[0]) + ', ' + _f(ci[1]) + ']') if ci else '—'} "
                f"| {e.get('vs_framework', '基准')} |")
        lines.append("")
    lines += ["## 跨数据集平均名次(只计 train-only 参与者)", "",
              "| 参与者 | 平均名次 | 数据集数 |", "|---|---|---|"]
    for r in s.get("avg_rank") or []:
        lines.append(f"| {r['name']} | {r['avg_rank']:.2f} | {r['n_datasets']} |")
    lines += ["", "## ModelEvo vs 各参与者(胜/平/负,按 95% CI,同训练口径配对)", "",
              "| 参与者 | ModelEvo 侧 | 框架胜 | 平 | 框架负 |", "|---|---|---|---|---|"]
    for n, c in (s.get("framework_vs") or {}).items():
        lines.append(f"| {n} | {c.get('reference') or '—'} | {c['framework_win']} "
                     f"| {c['tie']} | {c['framework_loss']} |")
    return "\n".join(lines) + "\n"
