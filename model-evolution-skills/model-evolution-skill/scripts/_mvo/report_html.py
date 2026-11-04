"""训练历程 HTML 报告(v2.16.1):交付时由 `report` 默认生成 report/report.html。

只读已落盘产物,确定性地组装成一份数据(`report/journey-data.json`),再注入
静态模板 `assets/report-journey-template.html`(内联 SVG/原生 JS,无外部依赖,
左侧目录)。重点呈现**冠军路径**——从基线到交付配方的每一步改动、为什么
有效、证据与增益——以及进化地图、全部尝试的 Δ+CI、证伪方向、三面最终效果、
过程算力、用户裁决与经验要点。

说明文字来源:可选的 `report/journey-narrative.yaml`(收尾 subagent 撰写)优先;
缺省回退到 trial 的 hypothesis/rationale/falsification 与导出经验卡。

数据边界:shadow 逐桶统计只用盘上的 shadow 标签(与 evaluate-final 同源);
**从不读取 final_test 标签**——final_test 只用 evaluation.json 里 gate 内算好的
指标与分段均值。本模块位于 scripts/ 侧,不进 harness_hash。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from runtime.logging_utils import get_logger, load_json, load_yaml, now_iso

logger = get_logger(__name__)

TEMPLATE = Path(__file__).resolve().parents[2] / "assets" / "report-journey-template.html"
DATA_PLACEHOLDER = "__JOURNEY_DATA__"
NARRATIVE_FILE = "journey-narrative.yaml"

_PHASE_NAMES = {
    "train:fit": "拟合", "train:feature": "特征解析", "eval": "评估",
    "train:hpo": "HPO 搜索", "train:load": "数据加载", "train:other": "其它",
    "train:score": "打分与收尾", "train:tail": "打分与收尾",
}


# ------------------------------------------------------------------ 小工具
def short_id(trial_id: str | None) -> str:
    m = re.match(r"trial-(\d+)", str(trial_id or ""))
    return m.group(1) if m else str(trial_id or "")


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _ci(v) -> list[float] | None:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        lo, hi = _num(v[0]), _num(v[1])
        return [lo, hi] if lo is not None and hi is not None else None
    if isinstance(v, str):
        parts = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v)
        if len(parts) == 2:
            return [float(parts[0]), float(parts[1])]
    return None


def _gain(delta, higher_is_better: bool) -> float | None:
    d = _num(delta)
    if d is None:
        return None
    return d if higher_is_better else -d


def short_label(statement: str, limit: int = 34) -> str:
    """hypothesis.statement → 图上短标签:去掉"包N/组合A"等前缀,取首个分句。"""
    s = re.sub(r"\s+", " ", str(statement or "")).strip()
    s = re.sub(r"^(包\s*\d+|组合\s*[A-Z]|[A-Z]\s*合并|条件消融)\s*[:：]?\s*", "", s)
    m = re.search(r"[:：;；。(（]", s[2:])          # 至少保留 2 个字符的前缀
    if m:
        s = s[:2 + m.start()].rstrip()
    return s if len(s) <= limit else s[:limit] + "…"


def step_category(row: dict) -> str:
    mods = str(row.get("changed_modules") or "").lower()
    act = str(row.get("search_action") or "").lower()
    if "feature" in mods:
        return "feature"
    if act == "hpo" or any(k in mods for k in ("model", "loss", "hpo")):
        return "model"
    if act == "data_view" or "data" in mods:
        return "data"
    return "other"


def _read_json(path: Path, sources: list[str], run_dir: Path):
    if not path.exists():
        return None
    try:
        data = load_json(path)
        sources.append(str(path.relative_to(run_dir)))
        return data
    except Exception:  # noqa: BLE001
        return None


def _read_yaml(path: Path, sources: list[str], run_dir: Path):
    if not path.exists():
        return None
    try:
        data = load_yaml(path)
        sources.append(str(path.relative_to(run_dir)))
        return data
    except Exception:  # noqa: BLE001
        return None


def _effective_mde(config, run_dir: Path) -> float:
    """v2.17.0:量纲感知 MDE(runtime.mde.effective_mde);模块缺失时退回旧绝对值键。"""
    try:
        from runtime.mde import effective_mde

        return float(effective_mde(config, run_dir=run_dir))
    except Exception:  # noqa: BLE001
        return float(config.get("evaluator.minimum_meaningful_improvement",
                                0.001) or 0.001)


# ------------------------------------------------------------------ 冠军路径
def champion_path(rows_by_id: dict[str, dict], champion: str | None) -> list[str]:
    """沿 leaderboard `parent` 的**第一个父代**从冠军回溯到根,返回根→冠军。"""
    path: list[str] = []
    cur = champion
    seen: set[str] = set()
    while cur and cur in rows_by_id and cur not in seen:
        seen.add(cur)
        path.append(cur)
        parents = [p for p in str(rows_by_id[cur].get("parent") or "").split(";")
                   if p and p != "None"]
        cur = parents[0] if parents else None
    return list(reversed(path))


def _pick_champion(lb: dict, rows: list[dict], metric: str, hib: bool) -> str | None:
    champ = ((lb.get("delivery_resolution") or {}).get("search_champion"))
    if champ:
        return champ
    ok = [r for r in rows if r.get("verdict") in ("accepted", "confirmed")
          and _num(r.get(metric)) is not None]
    if not ok:
        return None
    best = (max if hib else min)(ok, key=lambda r: float(r[metric]))
    return best["trial_id"]


# ------------------------------------------------------------------ 叙述文件
def load_narrative(run_dir: Path) -> dict:
    p = Path(run_dir) / "report" / NARRATIVE_FILE
    if not p.exists():
        return {}
    try:
        return load_yaml(p) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("journey-narrative.yaml 解析失败(忽略): %s", exc)
        return {}


def _narr_lookup(table: dict | None, trial_id: str):
    if not isinstance(table, dict):
        return None
    return table.get(trial_id) or table.get(short_id(trial_id)) \
        or table.get(f"trial-{short_id(trial_id)}")


# ------------------------------------------------------------------ shadow 分桶
def shadow_segments(run_dir: Path, spec, baseline_id: str | None,
                    inverse: str | None, n_bins: int = 10) -> dict | None:
    """交付模型 vs 基线在 shadow 上的十分位真实均值与累计捕获(按预测降序)。

    只读 final-model/shadow_test_score.parquet、基线 trial 的 shadow 分数与盘上
    shadow 标签;inverse=="expm1" 时按原值统计。任一缺失返回 None。
    """
    import numpy as np
    import pandas as pd

    from runtime.data import SAMPLE_KEY_COLUMN
    from runtime.metrics import align_score_by_keys
    from runtime.segment_report import _bin_by_score

    lab_p = run_dir / "sample-profile" / "splits" / "labels" / "shadow_test.parquet"
    deliv_p = run_dir / "final-model" / "shadow_test_score.parquet"
    base_p = run_dir / "trials" / str(baseline_id or "") / "shadow_test_score.parquet"
    if not (lab_p.exists() and deliv_p.exists() and base_p.is_file()):
        return None
    lab = pd.read_parquet(lab_p)
    keys = lab[SAMPLE_KEY_COLUMN].astype(str).values
    y = lab[spec.target_name].to_numpy(dtype=float)
    y_val = np.expm1(y) if inverse == "expm1" else y
    total = float(y_val.sum())
    if total <= 0:
        return None
    out: dict[str, Any] = {"n": int(len(y)), "scale": "raw" if inverse == "expm1" else "label",
                           "inverse": inverse}
    for tag, path in (("deliv", deliv_p), ("base", base_p)):
        df = pd.read_parquet(path)
        sc = align_score_by_keys(df["score"].to_numpy(dtype=float), keys,
                                 df[SAMPLE_KEY_COLUMN].astype(str).values)
        bins = _bin_by_score(sc, n_bins)
        if bins is None:
            return None
        means, pos, cum = [], [], []
        acc = 0.0
        for b in range(n_bins - 1, -1, -1):          # 桶 1 = 预测最高
            m = bins == b
            yb = y_val[m]
            acc += float(yb.sum())
            means.append(round(float(yb.mean()), 4))
            pos.append(round(float((yb > 0).mean()), 4))
            cum.append(round(acc / total, 4))
        order = np.argsort(-sc, kind="stable")
        cy = np.cumsum(y_val[order]) / total
        out[tag] = {"mean": means, "pos_rate": pos, "cum": cum,
                    "top": {f"{k}%": round(float(cy[max(int(len(cy) * k / 100) - 1, 0)]), 4)
                            for k in (1, 5, 10, 20, 30)}}
    return out


# ------------------------------------------------------------------ 主收集
def collect_journey_data(run_dir: Path, spec, config) -> dict:
    """组装报告数据(只读)。每个分项独立容错,失败写入 warnings。"""
    from runtime import __version__
    from runtime.metrics import metric_direction

    run_dir = Path(run_dir)
    sources: list[str] = []
    warnings: list[str] = []
    narr = load_narrative(run_dir)
    if narr:
        sources.append(f"report/{NARRATIVE_FILE}")
    metric = str(spec.primary_metric)
    hib = bool(metric_direction(metric))
    sd = spec.data or {}
    secondary = [m for m in (sd.get("secondary_metrics") or []) if m != metric]
    split = sd.get("split_policy") or {}
    target = sd.get("target") or {}
    prof = _read_json(run_dir / "sample-profile" / "sample-profile.json", sources, run_dir) or {}

    data: dict[str, Any] = {
        "meta": {
            "task_id": sd.get("task_id"), "task_name": sd.get("task_name"),
            "objective": str(sd.get("objective_description") or "")[:600],
            "model_type": sd.get("model_type"), "metric": metric,
            "higher_is_better": hib, "secondary": secondary,
            "target": target.get("name"),
            "value_definition": target.get("value_definition") or target.get("positive_definition"),
            "split": {k: split.get(k) for k in ("type", "train", "validation", "shadow_test", "final_test")},
            "n_rows": prof.get("n_rows"), "n_features": prof.get("n_features"),
            "skill_version": __version__, "generated_at": now_iso(),
        },
        "headline": narr.get("headline"),
        "notes": list(narr.get("notes") or []),
        "warnings": warnings, "sources": sources,
    }

    # ---- leaderboard / trials
    lb = _read_json(run_dir / "leaderboard" / "leaderboard.json", sources, run_dir) or {}
    rows = [r for r in (lb.get("rows") or []) if _num(r.get(metric)) is not None]
    rows_by_id = {r["trial_id"]: r for r in rows}
    baseline_id = lb.get("baseline_id")
    champion = _pick_champion(lb, rows, metric, hib)
    path = champion_path(rows_by_id, champion)
    on_path = set(path)
    tol = _num((lb.get("delivery_resolution") or {}).get("tolerance"))
    data["mde"] = tol if tol else _effective_mde(config, run_dir)
    data["baseline_id"] = baseline_id
    data["champion_id"] = champion
    data["delivery"] = lb.get("delivery_resolution") or {}
    data["n_trials_total"] = len(lb.get("rows") or [])
    data["n_trials_failed"] = data["n_trials_total"] - len(rows)

    labels = narr.get("labels") or {}
    trials = []
    for r in sorted(rows, key=lambda x: x["trial_id"]):
        tid = r["trial_id"]
        ty = _read_yaml(run_dir / "trials" / tid / "trial.yaml", sources, run_dir) or {}
        hyp = ty.get("hypothesis") or {}
        stmt = str(hyp.get("statement") or "")
        lab = _narr_lookup(labels, tid) or ("基线 · " + str(r.get("algorithm") or "")
                                            if tid == baseline_id else short_label(stmt))
        parents = [p for p in str(r.get("parent") or "").split(";") if p and p != "None"]
        ci = _ci(r.get("delta_ci"))
        conf = None
        if str(r.get("search_action")) == "confirm":
            ev = _read_json(run_dir / "trials" / tid / "evaluation.json", sources, run_dir) or {}
            fc = ev.get("fresh_confirmation") or {}
            ref = fc.get("reference") or {}
            conf = {"passed": fc.get("passed"), "ref_trial": ref.get("reference_trial"),
                    "ref_delta": _num(ref.get("delta"))}
        trials.append({
            "id": tid, "sid": short_id(tid), "g": int(r.get("generation") or 0),
            "action": r.get("search_action"), "category": step_category(r),
            "metric": float(r[metric]),
            "secondary": {m: _num(r.get(m)) for m in secondary},
            "dp": _num(r.get("delta_vs_parent")), "ci": ci,
            "gain": _gain(r.get("delta_vs_parent"), hib),
            "gain_ci": (sorted([_gain(ci[0], hib), _gain(ci[1], hib)]) if ci else None),
            "p_better": _num(r.get("p_better")), "verdict": r.get("verdict") or (
                "baseline" if tid == baseline_id else None),
            "parents": parents, "label": lab, "on_path": tid in on_path,
            "hypothesis": stmt[:400], "rationale": str(hyp.get("rationale") or "")[:300],
            "falsification": str(hyp.get("falsification") or "")[:300],
            "runtime_sec": _num(r.get("runtime_sec")), "confirm": conf,
        })
    data["trials"] = trials
    data["path"] = path
    tmap = {t["id"]: t for t in trials}

    # ---- 冠军路径步骤
    steps, shares = [], {}
    if path:
        prev = tmap[path[0]]["metric"]
        data["baseline_metric"] = prev
        for tid in path[1:]:
            t = tmap[tid]
            if t["action"] == "confirm":
                continue
            disp = t["metric"] - prev
            gain_disp = disp if hib else -disp
            n = _narr_lookup(narr.get("steps"), tid) or {}
            evid = list(n.get("evidence") or [])
            if not evid:
                if t["ci"]:
                    evid.append(f"Δ vs 父代 {t['dp']:+.5f},95% CI [{t['ci'][0]:+.5f}, {t['ci'][1]:+.5f}]"
                                + (f",p_better {t['p_better']:.3f}" if t["p_better"] is not None else ""))
                if t["falsification"]:
                    evid.append("预注册证伪条件:" + t["falsification"][:160])
            steps.append({
                "id": tid, "sid": t["sid"], "g": t["g"], "category": t["category"],
                "title": n.get("title") or t["label"],
                "why": n.get("why") or t["rationale"] or t["hypothesis"][:240],
                "evidence": evid, "dp": t["dp"], "ci": t["ci"], "gain": t["gain"],
                "display_gain": gain_disp, "after": t["metric"],
            })
            shares[t["category"]] = shares.get(t["category"], 0.0) + gain_disp
            prev = t["metric"]
        data["champion_metric"] = tmap[path[-1]]["metric"]
    data["steps"] = steps
    data["shares"] = shares

    # ---- 证伪 / 递减
    fals = []
    for t in trials:
        if t["on_path"] or t["action"] == "confirm" or t["verdict"] not in ("rejected", "promising"):
            continue
        n = _narr_lookup(narr.get("falsified"), t["id"]) or {}
        fals.append({"id": t["id"], "sid": t["sid"], "title": n.get("title") or t["label"],
                     "why": n.get("why") or (t["falsification"] or t["hypothesis"])[:200],
                     "dp": t["dp"], "gain": t["gain"], "verdict": t["verdict"]})
    data["falsified"] = fals
    confs = [t["confirm"] for t in trials if t["confirm"]]
    data["confirmations"] = {"n": len(confs), "passed": sum(1 for c in confs if c.get("passed"))}

    # ---- 最终效果(只读 gate 内产物)
    try:
        ev = _read_json(run_dir / "final-model" / "evaluation.json", sources, run_dir) or {}
        meta = _read_json(run_dir / "final-model" / "model_meta.json", sources, run_dir) or {}
        b = rows_by_id.get(baseline_id) or {}
        c = rows_by_id.get(champion) or {}
        pick = lambda d: ({m: _num((d or {}).get(m)) for m in [metric] + secondary}
                          if d else None)
        faces = [
            {"key": "validation", "label": "validation(选择口径)", "split": split.get("validation"),
             "baseline": pick(b), "search": pick(c),
             "delivery": None, "delivery_note": "已并入训练" if meta.get("n_validation_rows") else None},
            {"key": "shadow", "label": "shadow(确认口径)", "split": split.get("shadow_test"),
             "baseline": pick(ev.get("baseline_shadow_test")),
             "search": pick(ev.get("shadow_test_search_model")),
             "delivery": pick(ev.get("shadow_test")) if ev.get("shadow_test_source") != "search_model_fallback" else None},
            {"key": "final", "label": "final_test(一次性盲测)", "split": split.get("final_test"),
             "baseline": pick(ev.get("baseline_final_test")), "search": None,
             "delivery": pick(ev.get("final_test"))},
        ]
        data["final"] = {
            "evaluated": bool(ev.get("final_test")), "faces": faces,
            "refit_mode": meta.get("refit_mode"), "es_holdout": meta.get("es_holdout"),
            # v2.17.0(#5):refit 口径留痕
            "refit_mode_requested": meta.get("refit_mode_requested"),
            "refit_mode_fallback_reason": meta.get("refit_mode_fallback_reason"),
            "clone_probe": (meta.get("clone_probe") or {}).get("status"),
            "shadow_precheck": meta.get("shadow_precheck"),
            "delivery_status": meta.get("delivery_status"),
            "final_bin_means": (ev.get("segment_evaluation") or {}).get("bin_means"),
            "raw_scale": {k: v for k, v in (ev.get("raw_scale") or {}).items()
                          if k in ("inverse_transform", "note", "final_test", "shadow_test")} or None,
            # v2.17.0(#20):gate 内算好的回归水位诊断(只含统计量)
            "regression_diagnostics": ev.get("regression_diagnostics") or None,
        }
        rs_note = ((ev.get("raw_scale") or {}).get("note"))
        if rs_note and not narr.get("notes"):
            data["notes"].append(rs_note)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"final: {exc}")
        data["final"] = None

    # ---- v2.17.0(#21):模型局限(与 report.md §14 同源;模板经 notes 呈现)
    try:
        from runtime.report import collect_limitations

        lims = collect_limitations(run_dir, spec)
        data["limitations"] = lims
        data["notes"].extend(f"模型局限 · {x}" for x in lims)
        if lims:
            sources.append("模型局限:final-model/{evaluation,model_meta}.json + "
                           "leaderboard + evolution-state + run-state + trials")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"limitations: {exc}")
        data["limitations"] = []

    # ---- shadow 分桶(交付 vs 基线)
    try:
        inv = ((data.get("final") or {}).get("raw_scale") or {}).get("inverse_transform")
        data["shadow_seg"] = shadow_segments(run_dir, spec, baseline_id, inv)
        if data["shadow_seg"]:
            sources.append("final-model/shadow_test_score.parquet + shadow 标签(盘上)")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"shadow_seg: {exc}")
        data["shadow_seg"] = None

    # ---- 代际 / 算力
    timing = _read_json(run_dir / "logs" / "timing-report.json", sources, run_dir) or {}
    gens = []
    gen_ids = sorted({t["g"] for t in trials})
    tg = {int(g.get("generation")): g for g in (timing.get("generations") or [])
          if g.get("generation") is not None}
    best = None
    for g in gen_ids:
        cand = [t["metric"] for t in trials if t["g"] <= g
                and t["verdict"] in ("accepted", "confirmed", "baseline")]
        if cand:
            best = (max if hib else min)(cand)
        gens.append({"g": g, "wall_min": _num((tg.get(g) or {}).get("wall_min")),
                     "n_trials": sum(1 for t in trials if t["g"] == g), "best": best})
    data["generations"] = gens
    agg: dict[str, float] = {}
    for k, v in (timing.get("phase_share") or {}).items():
        name = _PHASE_NAMES.get(k, "其它")
        agg[name] = agg.get(name, 0.0) + float(v or 0)
    data["phase_share"] = sorted(([k, round(v, 4)] for k, v in agg.items()), key=lambda x: -x[1])
    data["trial_hours"] = _num(timing.get("trial_hours"))

    # ---- 事件:路径里程碑 + 停止报告 + 用户裁决
    events = []
    for s in steps:
        events.append({"g": s["g"], "kind": "step",
                       "title": f"gen-{s['g']:02d} · {s['title']}",
                       "text": f"增益 {s['display_gain']:+.4f}"})
    for gdir in sorted((run_dir / "generations").glob("gen-*")):
        gy = _read_yaml(gdir / "generation.yaml", sources, run_dir) or {}
        st = gy.get("stop_report") or {}
        if st.get("stop_reason"):
            m = re.search(r"(\d+)$", gdir.name)
            events.append({"g": int(m.group(1)) if m else None, "kind": "stop",
                           "title": f"{gdir.name} · Runtime 停止判定",
                           "text": str(st.get("stop_reason"))})
    events.sort(key=lambda e: (e["g"] if e["g"] is not None else 99, e["kind"] != "stop"))
    data["events"] = events
    decisions = []
    dp = run_dir / "logs" / "user-decisions.md"
    if dp.exists():
        sources.append("logs/user-decisions.md")
        for line in dp.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^##\s+(D-\d+)\s+(.+?)\s+[—-]{1,2}\s+(.+)$", line.strip())
            if m:
                decisions.append({"id": m.group(1), "title": m.group(2), "status": m.group(3)})
    data["decisions"] = decisions

    # ---- 经验要点
    ins = [{"title": i.get("title"), "text": i.get("text"), "kind": i.get("kind", "insight")}
           for i in (narr.get("insights") or []) if isinstance(i, dict)]
    if not ins:
        exp = _read_yaml(run_dir / "experience" / "exported-global-experiences.yaml", sources, run_dir) or []
        kind = {"positive": "有效", "negative": "避免", "neutral": "方法论"}
        for e in exp if isinstance(exp, list) else []:
            claim = str(e.get("claim") or e.get("statement") or "")
            if claim:
                ins.append({"title": kind.get(str(e.get("outcome")), "经验"),
                            "text": claim[:200], "kind": str(e.get("outcome") or "")})
        ins = ins[:12]
    data["insights"] = ins
    if not data["headline"] and steps:
        data["headline"] = (f"{metric} {data['baseline_metric']:.4f} → {data['champion_metric']:.4f}"
                            f"(冠军路径 {len(steps)} 步)")
    return data


def render_html(data: dict) -> str:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    if DATA_PLACEHOLDER not in tpl:
        raise RuntimeError(f"模板缺少占位符 {DATA_PLACEHOLDER}: {TEMPLATE}")
    payload = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    title = str((data.get("meta") or {}).get("task_name") or (data.get("meta") or {}).get("task_id") or "ModelEvo")
    title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return tpl.replace("__JOURNEY_TITLE__", title).replace(DATA_PLACEHOLDER, payload)


def build_journey_html(run_dir: Path, spec, config) -> Path:
    """生成 report/report.html 与 report/journey-data.json,返回 html 路径。"""
    run_dir = Path(run_dir)
    data = collect_journey_data(run_dir, spec, config)
    out = run_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "journey-data.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    path = out / "report.html"
    path.write_text(render_html(data), encoding="utf-8")
    for w in data.get("warnings") or []:
        logger.warning("report.html 分项降级: %s", w)
    return path
