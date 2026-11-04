"""Report:最终交付(§18)。固定模板 16 节,人类交付产物,不承担 Memory 职责。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .logging_utils import get_logger, load_json, load_yaml, now_iso, dump_json
from .spec import TaskSpec
from .trial import TrialRecord, lineage_of

logger = get_logger(__name__)


import json as _json  # v2.10.1/F8b:读 capacity-report.json
import re  # v2.10.1/F8b:内联片段标题降级


def _demote_headings(text: str, drop_leading_title: bool = False) -> str:
    """v2.10.1/F8b:内联外部 md 片段时把其标题整体降两级。

    被内联的文件(如 `final-model/delivery-caveats.md`)自带独立编号
    (`# 标题` / `## 1.` / `## 2.`),直接内联会与报告自身的节号冲突
    (读者会把 15b 之后的 `## 1.` 误认成报告第 1 节)。

    只动 **ATX 标题行**(`#` 开头且后跟空格);引用行(`> `)、列表、
    表格、正文一律原样 —— 早期实现漏掉源文件自身的 `# ` 首行,
    又把 `> ` 误映射成 `# `,两次都由验证环节抓出(见 CHANGELOG)。
    """
    lines_in = text.strip().splitlines()
    # drop_leading_title:源文件的首行 H1 与调用方的包装标题语义重复,
    # 内联时直接去掉(否则出现两条相邻的同级标题)
    if drop_leading_title and lines_in \
            and re.match(r"^# [^#]", lines_in[0]):
        lines_in = lines_in[1:]
    out = []
    for ln in lines_in:
        m = re.match(r"^(#{1,6}) (.*)$", ln)
        out.append(f"{'#' * min(m.end(1) + 2, 6)} {m.group(2)}" if m else ln)
    return "\n".join(out).lstrip("\n")


def _metric_higher_better(metric: str) -> bool:
    from .metrics import metric_direction

    return metric_direction(metric)


# ---------------------------------------------------------------------------
# v2.17.0(#21):「模型局限」自动汇总
# ---------------------------------------------------------------------------
# 事故(run4 regression.2):report.md §14 显示"未记录",而交付期水位偏差
# (+20%)、冠军分类子模型撞 cap、−58 的未交付链、单数据视图、3 次 harness
# 变更、交付 refit 回落全都在盘上产物里。下列函数只读磁盘产物,逐来源
# 独立容错(单个来源坏掉不影响其余),缺失项跳过;全部缺失时调用方显示
# "未记录"。report.md §14 与 report.html(journey-data.limitations)共用。

def _fnum(v) -> "float | None":
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _g(v, fmt: str = ".6g") -> str:
    f = _fnum(v)
    return format(f, fmt) if f is not None else "—"


def _short(v, limit: int = 120) -> str:
    if isinstance(v, float):
        return _g(v)
    if isinstance(v, (list, tuple)):
        s = ", ".join(_short(x, 40) for x in list(v)[:6])
        s += " …" if len(v) > 6 else ""
    elif isinstance(v, dict):
        s = _json.dumps(v, ensure_ascii=False, default=str)
    else:
        s = str(v)
    return s if len(s) <= limit else s[:limit] + "…"


def _latest_stop_report(run_dir: Path, ev_state: dict | None,
                        trials=None) -> dict:
    """evolution-state.stopped 优先;否则取最新一代 generation.yaml 的 stop_report。

    v2.17.0(D4):返回前经 :func:`_correct_stop_same_recipe` 剔除与交付位同配方的
    "未交付更优候选"(旧 finalize 落盘的 stop_report 未做该排除)。
    """
    stop = (ev_state or {}).get("stopped")
    if isinstance(stop, dict) and stop:
        return _correct_stop_same_recipe(run_dir, stop, trials)
    gens = sorted((Path(run_dir) / "generations").glob("gen-*"))
    for gdir in reversed(gens):
        p = gdir / "generation.yaml"
        if not p.exists():
            continue
        try:
            st = (load_yaml(p) or {}).get("stop_report") or {}
        except Exception:  # noqa: BLE001
            continue
        if isinstance(st, dict) and st:
            return _correct_stop_same_recipe(run_dir, st, trials)
    return {}


def _correct_stop_same_recipe(run_dir: Path, stop, trials=None):
    """stop_report.undelivered_gain 的同配方读取侧修正(失败原样返回)。"""
    if not isinstance(stop, dict) or not isinstance(
            stop.get("undelivered_gain"), dict):
        return stop
    try:
        from .evolution import exclude_same_recipe_from_stop

        if not trials:
            from .trial import TrialRecord

            tdir = Path(run_dir) / "trials"
            trials = [TrialRecord.load(d) for d in sorted(tdir.iterdir())
                      if (d / "trial.yaml").exists()] if tdir.exists() else []
        return exclude_same_recipe_from_stop(stop, trials) or stop
    except Exception as exc:  # noqa: BLE001
        logger.warning("stop_report 同配方修正失败(原样呈现): %s", exc)
        return stop


def _trial_evaluations(run_dir: Path, trials=None) -> list[tuple[str, dict]]:
    """(trial_id, evaluation) 列表:优先用已加载的 TrialRecord,否则读 trial.yaml。"""
    out: list[tuple[str, dict]] = []
    if trials:
        for t in trials:
            ev = (getattr(t, "data", None) or {}).get("evaluation") or {}
            out.append((str(getattr(t, "trial_id", "")), ev))
        return out
    tdir = Path(run_dir) / "trials"
    if not tdir.exists():
        return out
    for d in sorted(tdir.iterdir()):
        p = d / "trial.yaml"
        if not p.exists():
            continue
        try:
            ev = (load_yaml(p) or {}).get("evaluation") or {}
        except Exception:  # noqa: BLE001
            ev = {}
        out.append((d.name, ev if isinstance(ev, dict) else {}))
    return out


def _lim_level_shift(final_eval: dict, meta: dict) -> list[str]:
    """(a) 交付期水位偏差:evaluation.regression_diagnostics;缺失回退 shadow 预检。"""
    out: list[str] = []
    rd = final_eval.get("regression_diagnostics") or {}
    for split in ("final_test", "shadow_test"):
        face = rd.get(split) or {}
        dl = face.get("delivery") or {}
        lv = dl.get("level") or {}
        bad = [r for r in dl.get("level_by_period") or []
               if isinstance(r, dict) and r.get("level_shift_warning")]
        if not (lv.get("level_shift_warning") or bad):
            continue
        ratio = _fnum(lv.get("level_ratio"))
        b_ratio = _fnum(((face.get("baseline") or {}).get("level") or {})
                        .get("level_ratio"))
        # v2.17.0(D8):整体与分期分开表述(判据 max(固定阈, z·SE))
        if lv.get("level_shift_warning"):
            txt = (f"交付期水位偏差({split}):交付模型预测均值 "
                   f"{_g(lv.get('pred_mean'))} vs 实际 {_g(lv.get('y_mean'))}")
            if ratio is not None:
                txt += f"(水位比 {ratio:.3f},{ratio - 1:+.1%})"
            if b_ratio is not None:
                txt += f";基线水位比 {b_ratio:.3f}"
            out.append(txt + " —— 排序可用、绝对金额需校准(交付后按最新期水位"
                             "做缩放/校准再用于金额口径)")
        if bad:
            out.append(
                f"交付期分期水位偏离({split}):" + ", ".join(
                    f"{r.get('period')}={_g(r.get('level_ratio'), '.3f')}"
                    for r in bad[:6])
                + " 超出逐期噪声带" + ("" if lv.get("level_shift_warning")
                                  else "(整体水位在阈内,属局部偏离)")
                + " —— 偏离期的绝对金额需按期校准")
    if out:
        return out
    sp = (meta or {}).get("shadow_precheck") or {}
    lv = ((sp.get("level_stats") or {}).get("delivery_model") or {})
    if sp.get("level_shift_warning") or lv.get("level_shift_warning"):
        ratio = _fnum(lv.get("level_ratio"))
        out.append(
            f"交付期水位偏差(shadow 预检):交付模型预测均值 {_g(lv.get('pred_mean'))}"
            f" vs 实际 {_g(lv.get('y_mean'))}"
            + (f"(水位比 {ratio:.3f},{ratio - 1:+.1%})" if ratio is not None
               else "")
            + " —— 排序可用、绝对金额需校准")
    return out


def _lim_cap_bound(run_dir: Path, champ_id: str | None) -> list[str]:
    """(b) 冠军(或其子模型)撞树数上限:metrics.es_info.cap_bound、<sub>_cap_bound、
    子模型最优轮数 ≥ 95%×cap。

    v2.16.3 批次:遥测足以判定时走单一语义(``runtime.hpo.unified_cap_bound``),
    只剩旗标/最优轮数时沿用 v2.17.0 的旧判据。"""
    if not champ_id:
        return []
    p = Path(run_dir) / "trials" / str(champ_id) / "metrics.json"
    if not p.exists():
        return []
    m = load_json(p) or {}
    es = m.get("es_info") or (m.get("refit") or {}).get("es_info") or {}
    cap = _fnum(es.get("cap"))
    # v2.16.3 批次(D-011):先按单一语义(实训轮数 ≥ cap,两段任一段撞即
    # True)判定;遥测足以判定(非 legacy 旗标)时以它为准 —— 旧 baseline 血缘
    # 的 es_info 旗标与 agent 写的 <sub>_cap_bound 用 best ≥ 0.95×cap,与 hurdle
    # 实训轮数对同一 trial 相反(run4 trial-022:7923/8000 判撞 cap,实际 clf
    # 实训 7973 轮、早停已触发)。
    try:
        from .hpo import unified_cap_bound

        _u = unified_cap_bound(m)
    except Exception:  # noqa: BLE001
        _u = {}
    if _u.get("cap_bound") is not None and not str(
            _u.get("source") or "").startswith("legacy_flag"):
        _segs = _u.get("segments") or {}
        if _u["cap_bound"]:
            hits = [f"{_u.get('source')}.{k} 实训 {_g(s.get('rounds'))}"
                    f"/cap {_g(s.get('cap'))}" for k, s in _segs.items()
                    if s.get("cap_bound")]
            return [f"冠军 {champ_id} 撞树数上限:" + ";".join(hits)
                    + " —— 轮数被上限截断,学习率/上限组合可能仍有增益;"
                    "交付重训沿用同一上限"]
        if _u.get("near_cap"):
            near = [f"{k} 最优轮数 {_g(s.get('best_iteration'))}"
                    f"/cap {_g(s.get('cap'))}" for k, s in _segs.items()
                    if s.get("near_cap")]
            return [f"冠军 {champ_id} 最优轮数贴近树数上限(≥95%×cap,早停已"
                    "触发,非撞 cap):" + ";".join(near)
                    + " —— 仅作提示"]
        return []
    parts: list[str] = []
    if es.get("cap_bound") is True or es.get("stopped_by") == "cap":
        parts.append(f"整体 es_info.cap_bound=true(cap={_g(cap)},"
                     f"stopped_by={es.get('stopped_by')})")
    for sect_name, sect in m.items():
        if not isinstance(sect, dict) or sect_name in ("es_info",):
            continue
        subs = {k[:-len("_cap_bound")] for k in sect
                if isinstance(k, str) and k.endswith("_cap_bound")}
        subs |= {k[:-len("_best_iteration")] for k in sect
                 if isinstance(k, str) and k.endswith("_best_iteration")}
        for sub in sorted(subs):
            it = _fnum(sect.get(f"{sub}_best_iteration"))
            flag = sect.get(f"{sub}_cap_bound")
            hit = flag is True or (
                flag is None and cap is not None and it is not None
                and it >= 0.95 * cap)
            if hit:
                parts.append(f"{sect_name}.{sub} 子模型最优轮数 {_g(it)}"
                             + (f"/cap {_g(cap)}" if cap is not None else ""))
    if not parts:
        return []
    return [f"冠军 {champ_id} 撞树数上限:" + ";".join(parts)
            + " —— 轮数被上限截断,学习率/上限组合可能仍有增益;交付重训沿用同一上限"]


def _undelivered_positive(ug) -> "bool | None":
    """undelivered_gain 是否指向一个优于交付锚点的 trial。

    WP1 口径(runtime.evolution.compute_undelivered_gain):恒为 dict,
    best_trial/gain 为空表示无更优候选;anchor_id 为空表示无法计算(返回
    None,调用方回退 leaderboard)。其它形态按非空即真。
    """
    if isinstance(ug, dict):
        if "best_trial" in ug or "gain" in ug or "anchor_id" in ug:
            if not ug.get("best_trial"):
                return None if not ug.get("anchor_id") else False
            g = _fnum(ug.get("gain"))
            return g is not None and g > 0
        return bool(ug)
    if isinstance(ug, (list, tuple)):
        return any(_undelivered_positive(x) for x in ug)
    if isinstance(ug, bool):
        return ug
    if isinstance(ug, (int, float)):
        return ug != 0
    return bool(ug) if ug is not None else None


def _fmt_undelivered(ug) -> str:
    if isinstance(ug, (int, float)) and not isinstance(ug, bool):
        return f"累计增益 {_g(ug)}"
    if isinstance(ug, dict) and ug.get("best_trial"):
        # WP1 schema:best_trial 相对交付锚点的 Δ / gain / SE / z / MDE 倍数
        txt = (f"{ug.get('best_trial')} 相对交付锚点 {ug.get('anchor_id')}:"
               f"Δ={_g(ug.get('delta'))}(增益 {_g(ug.get('gain'))}")
        if _fnum(ug.get("gain_in_mde")) is not None:
            txt += f",{_fnum(ug.get('gain_in_mde')):.1f}×MDE"
        if _fnum(ug.get("z")) is not None:
            txt += f",z={_fnum(ug.get('z')):.2f}"
        if _fnum(ug.get("p_better")) is not None:
            txt += f",p_better={_fnum(ug.get('p_better')):.3f}"
        txt += f";口径 {ug.get('source') or '—'})"
        others = [c for c in ug.get("top_candidates") or []
                  if isinstance(c, dict)
                  and c.get("trial_id") != ug.get("best_trial")
                  and (_fnum(c.get("gain")) or 0) > 0]
        if others:
            txt += ";其余候选 " + ", ".join(
                f"{c.get('trial_id')}(增益 {_g(c.get('gain'))})"
                for c in others[:3])
        if ug.get("action_hint"):
            txt += f";处置:{ug.get('action_hint')}"
        return txt
    if isinstance(ug, dict):
        keys = ("trial_id", "candidate", "best_trial", "chain", "trials",
                "anchor", "anchor_id", "vs", "metric", "delta", "gain",
                "delta_vs_anchor", "z", "p_value", "se", "note")
        parts = [f"{k}={_short(ug[k], 80)}" for k in keys
                 if ug.get(k) is not None]
        return ", ".join(parts) if parts else _short(ug, 300)
    if isinstance(ug, (list, tuple)):
        return ";".join(_fmt_undelivered(x) for x in list(ug)[:5]) \
            + (f";… 共 {len(ug)} 项" if len(ug) > 5 else "")
    return _short(ug, 300)


def _lim_undelivered(stop: dict, lb: dict, spec, delivered: str | None
                     ) -> list[str]:
    """(c) 未交付的更优候选:stop_report.undelivered_gain;没有(或无法计算)时用
    leaderboard 中比交付位更优的 promising trial。"""
    tag = "仅 validation 证据、含选择偏差,未经独立确认"
    ug = (stop or {}).get("undelivered_gain")
    pos = _undelivered_positive(ug) if ug not in (None, {}, [], "") else None
    if pos:
        return [f"未交付的更优候选(stop_report.undelivered_gain;{tag}):"
                f"{_fmt_undelivered(ug)}"]
    if pos is False:
        return []           # runtime 已算出:锚点之上没有更优 trial
    rows = (lb or {}).get("rows") or []
    metric = spec.primary_metric
    d_row = next((r for r in rows if r.get("trial_id") == delivered), None)
    d_val = _fnum((d_row or {}).get(metric))
    if d_val is None:
        return []
    hib = _metric_higher_better(metric)
    better = []
    for r in rows:
        v = _fnum(r.get(metric))
        if r.get("verdict") != "promising" or v is None \
                or r.get("failure_reason"):
            continue
        if (v > d_val) if hib else (v < d_val):
            better.append((v, r))
    if not better:
        return []
    better.sort(key=lambda x: x[0], reverse=hib)
    items = ", ".join(f"{r.get('trial_id')}({metric}={_g(v)},较交付位 "
                      f"{v - d_val:+.6g})" for v, r in better[:5])
    more = f" 等 {len(better)} 个" if len(better) > 5 else ""
    return [f"未交付的更优候选({tag}):leaderboard 中 {len(better)} 个 promising "
            f"trial 的 validation {metric} 优于交付位 {delivered}:{items}{more}"
            " —— 未进入确认链,不代表样本外一定更优"]


def _lim_single_view(stop: dict) -> list[str]:
    """(d) 只试过单一数据视图(stop_report.non_sample_data_views / data_views_tried)。"""
    views = (stop or {}).get("non_sample_data_views")
    if not isinstance(views, list):
        tried = (stop or {}).get("data_views_tried")
        views = [v for v in tried if v != "sample"] \
            if isinstance(tried, list) else None
    if not isinstance(views, list) or len(set(views)) != 1:
        return []
    return [f"只试过单一数据视图({views[0]}):全部结论限于该特征/样本口径,"
            "未验证其它数据视图(新数据源、观察窗、样本面)能否带来增益"]


def _lim_sample_face(lb: dict, delivered: str | None) -> list[str]:
    """(e) 抽样面(data_view=sample)上裁决的 trial 的结论范围。"""
    rows = [r for r in (lb or {}).get("rows") or []
            if str(r.get("data_view") or "") == "sample"]
    if not rows:
        return []
    neg = [r for r in rows if r.get("verdict") in ("rejected", "promising")]
    txt = (f"{len(rows)} 个 trial 在抽样面(data_view=sample)上裁决"
           f"(其中 rejected/promising {len(neg)} 个):其 Δ 与 verdict 只代表"
           "抽样面口径,配对 SE 大于全量面,抽样面上'无显著增益'不等于全量无效")
    if delivered and any(r.get("trial_id") == delivered for r in rows):
        txt += ";交付位本身在抽样面上选出"
    return [txt]


def _lim_divergence(run_dir: Path, final_eval: dict, trials=None) -> list[str]:
    """(f) 主/次指标背离:trial evaluation.regression_diagnostics.vs_parent_divergence
    与交付模型 vs 基线(final_test/shadow)。"""
    out: list[str] = []
    ids: list[str] = []
    prim = sec = None
    for tid, ev in _trial_evaluations(run_dir, trials):
        dv = ((ev.get("regression_diagnostics") or {})
              .get("vs_parent_divergence") or {})
        if dv.get("diverged") is True:
            ids.append(tid)
            prim, sec = dv.get("primary") or prim, dv.get("secondary") or sec
    if ids:
        out.append(
            f"主/次指标背离:{len(ids)} 个 trial 的 {prim or 'rmse'} 与 "
            f"{sec or 'mae'} 相对父代方向相反({', '.join(ids[:6])}"
            f"{' 等' if len(ids) > 6 else ''})—— 主指标的改善/退化可能掩盖了"
            "另一部分样本(如中低额段)的反向变化,裁决只看主指标")
    rd = final_eval.get("regression_diagnostics") or {}
    for split in ("final_test", "shadow_test"):
        dv = (((rd.get(split) or {}).get("delivery") or {})
              .get("vs_baseline_divergence") or {})
        if dv.get("diverged") is True:
            out.append(
                f"交付模型 vs 基线({split}):{dv.get('primary')} Δ="
                f"{_g(dv.get('delta_primary'))} 与 {dv.get('secondary')} Δ="
                f"{_g(dv.get('delta_secondary'))} 方向相反 —— 对基线的提升"
                "并非在所有误差口径上成立")
    return out


def _lim_harness_changes(run_state: dict) -> list[str]:
    """(g) 运行中登记的 harness 变更(run-state.harness_change_log)。"""
    log = [e for e in (run_state or {}).get("harness_change_log") or []
           if isinstance(e, dict)]
    if not log:
        return []
    items = []
    for e in log[:4]:
        changed = e.get("changed") or []
        items.append(
            f"{str(e.get('at') or '')[:16]} "
            f"{'/'.join(map(str, changed[:4]))}"
            f"{'…' if len(changed) > 4 else ''}"
            f"(measurement_impact={e.get('measurement_impact')})")
    return [f"运行中登记了 {len(log)} 次 harness 变更:{';'.join(items)}"
            f"{' …' if len(log) > 4 else ''} —— 跨变更的 trial 比较依赖登记时"
            "'分数口径不变'的声明,详见 run-state.harness_change_log"]


def _lim_refit_fallback(meta: dict) -> list[str]:
    """(h) 交付 refit 口径回落(model_meta;旧版产物按 es_source 推断)。"""
    meta = meta or {}
    reason = meta.get("refit_mode_fallback_reason")
    if reason:
        return [f"交付重训口径回落:请求 {meta.get('refit_mode_requested')},"
                f"实际 {meta.get('refit_mode_effective') or meta.get('refit_mode')}"
                f";原因:{reason} —— 交付模型的轮数由 train+validation 最新尾部"
                "早停决定,与搜索期 validation 早停口径不同"]
    if "refit_mode_requested" not in meta \
            and meta.get("es_source") == "validation" \
            and meta.get("refit_mode") == "rerun_recipe":
        return ["交付重训口径回落(由旧版 model_meta 推断,未记录回落原因):冠军"
                "早停面=validation 本应 frozen_rounds,实际 rerun_recipe"
                "(train+validation 最新尾部早停后全量重拟合)—— 交付模型的轮数"
                "由更晚的一段数据决定,与搜索期 validation 早停口径不同"]
    return []


def _cap_confounded_rows(run_dir: Path, spec, trials=None) -> list[dict]:
    """v2.17.2(A1):撞 cap 混杂的 trial 明细 [{trial_id, detail}]。

    evaluation.cap_confounded(_detail) 优先;旧产物(v2.17.2 前评估)按
    :func:`runtime.evaluator.cap_confounded_detail` 从 evaluation.complexity
    + metrics.es_info 重算(与方向台账同口径)。
    """
    from .evaluator import cap_confounded_detail

    hib = _metric_higher_better(spec.primary_metric)
    rows: list[dict] = []
    recs = list(trials or [])
    if not recs:
        tdir = Path(run_dir) / "trials"
        if tdir.exists():
            for d in sorted(tdir.iterdir()):
                if (d / "trial.yaml").exists():
                    try:
                        recs.append(TrialRecord.load(d))
                    except Exception:  # noqa: BLE001
                        continue
    for t in recs:
        data = getattr(t, "data", None) or {}
        ev = data.get("evaluation") or {}
        if not isinstance(ev, dict) or not ev:
            continue
        det = ev.get("cap_confounded_detail") if ev.get(
            "cap_confounded") else None
        if det is None:
            es = None
            mp = Path(run_dir) / "trials" / str(t.trial_id) / "metrics.json"
            if mp.exists():
                try:
                    es = (load_json(mp) or {}).get("es_info")
                except Exception:  # noqa: BLE001
                    es = None
            det = cap_confounded_detail(
                ev, es_info=es, higher_is_better=hib,
                is_root_trial=not (getattr(t, "parent_ids", None) or []),
                confirm_trial=str(data.get("search_action") or "")
                == "confirm")
        if det:
            rows.append({"trial_id": str(t.trial_id),
                         "detail": det if isinstance(det, dict) else {}})
    return rows


def _lim_cap_confounded(run_dir: Path, spec, trials=None) -> list[str]:
    """(i) v2.17.2(A1):负结论被撞 cap 混杂的 trial(不作为方向负证据)。"""
    rows = _cap_confounded_rows(run_dir, spec, trials)
    if not rows:
        return []
    items = []
    for r in rows[:6]:
        d = r["detail"]
        items.append(f"{r['trial_id']}(Δparent={_g(d.get('delta_vs_parent'))},"
                     f"best_iter {_g(d.get('best_iteration'))}/cap "
                     f"{_g(d.get('cap'))},{d.get('verdict') or '—'})")
    return [f"{len(rows)} 个 trial 的负结论被树数上限截断混杂(cap_confounded):"
            + ";".join(items) + (" …" if len(rows) > 6 else "")
            + " —— 训练撞 cap 且无余量外扩,负 Δ 混入欠训练效应;verdict 保留但"
            "不作为该方向的负证据(方向台账不记负奖励),如需裁决应提 lr/放宽 cap 重跑"]


def shadow_gap_sanity(ev: dict | None) -> dict | None:
    """v2.17.2(C4):trial evaluation 的 shadow gap 诊断位(读侧兼容)。

    v2.17.2 起 runner 写 ``shadow_test.gap_sanity_passed`` +
    ``criterion="gap_sanity"``;旧产物只有 ``shadow_test.passed`` —— 与
    ``fresh_confirmation.passed``(确认/交付门)同名,报告与人工复盘曾把
    "shadow gap 诊断未过"误读成"确认失败"(run4 trial-020:fresh_confirmation
    通过而 shadow_test.passed=false)。它只是 validation→shadow 的落差提示,
    **非阻断**。无 shadow_test 段返回 None。
    """
    sh = (ev or {}).get("shadow_test")
    if not isinstance(sh, dict) or not sh:
        return None
    if "gap_sanity_passed" in sh:
        passed, key = sh.get("gap_sanity_passed"), "gap_sanity_passed"
    elif "passed" in sh:
        passed, key = sh.get("passed"), "passed(legacy)"
    else:
        passed, key = None, None
    return {"passed": passed, "key": key,
            "criterion": sh.get("criterion") or "gap_sanity",
            "gap_vs_validation": sh.get("gap_vs_validation"),
            "gap_relative": sh.get("gap_relative"),
            "blocking": False}


def _confirmation_gate_lines(ev: dict | None, trial_id: str | None) -> list[str]:
    """§12:交付 trial 的确认门(阻断)与 shadow gap 诊断(非阻断)分行呈现。"""
    ev = ev or {}
    out: list[str] = []
    fc = ev.get("fresh_confirmation") or {}
    if isinstance(fc, dict) and fc:
        out.append(f"- 独立确认门 fresh_confirmation.passed={fc.get('passed')}"
                   f"(`{trial_id}`,criterion="
                   f"{fc.get('criterion') or 'legacy_two_sided'};**交付门**)")
    gs = shadow_gap_sanity(ev)
    if gs is not None:
        _rel = _fnum(gs.get("gap_relative"))
        out.append(
            f"- shadow gap 诊断 {gs.get('key') or 'gap_sanity_passed'}="
            f"{gs.get('passed')}(criterion={gs.get('criterion')},"
            f"gap_vs_validation={_g(gs.get('gap_vs_validation'))}"
            + (f",相对 {_rel:.2%}" if _rel is not None else "")
            + ")—— **非阻断诊断**:只提示 validation→shadow 落差,不参与"
            "确认/交付裁决,与上面的确认门不是同一判据")
    return out


def collect_limitations(run_dir: Path, spec, *, final_eval: dict | None = None,
                        meta: dict | None = None, lb: dict | None = None,
                        ev_state: dict | None = None,
                        run_state: dict | None = None,
                        trials=None) -> list[str]:
    """从磁盘产物自动汇总「模型局限」(v2.17.0,#21)。参数缺省时自行读盘。

    来源(缺失项跳过):(a) 交付期水位偏差;(b) 冠军/子模型撞 cap;
    (c) 未交付的更优候选;(d) 单一数据视图;(e) 抽样面结论范围;
    (f) 主/次指标背离;(g) harness 变更;(h) 交付 refit 回落;
    (i) v2.17.2:撞 cap 混杂的负结论(cap_confounded)。
    """
    run_dir = Path(run_dir)

    def _rj(rel: str) -> dict:
        p = run_dir / rel
        try:
            return (load_json(p) or {}) if p.exists() else {}
        except Exception:  # noqa: BLE001
            return {}

    final_eval = final_eval if final_eval is not None \
        else _rj("final-model/evaluation.json")
    meta = meta if meta is not None else _rj("final-model/model_meta.json")
    lb = lb if lb is not None else _rj("leaderboard/leaderboard.json")
    ev_state = ev_state if ev_state is not None \
        else _rj("evolution-state.json")
    run_state = run_state if run_state is not None else _rj("run-state.json")
    res = (lb or {}).get("delivery_resolution") or {}
    delivered = (final_eval.get("best_trial") or meta.get("best_trial")
                 or res.get("delivery_trial") or res.get("search_champion"))
    stop: dict = {}
    try:
        stop = _latest_stop_report(run_dir, ev_state, trials)
    except Exception:  # noqa: BLE001
        stop = {}
    sources = (
        lambda: _lim_level_shift(final_eval, meta),
        lambda: _lim_cap_bound(run_dir, delivered),
        lambda: _lim_undelivered(stop, lb, spec, delivered),
        lambda: _lim_single_view(stop),
        lambda: _lim_sample_face(lb, delivered),
        lambda: _lim_divergence(run_dir, final_eval, trials),
        lambda: _lim_harness_changes(run_state),
        lambda: _lim_refit_fallback(meta),
        # v2.17.2(A1):撞 cap 混杂的负结论
        lambda: _lim_cap_confounded(run_dir, spec, trials),
    )
    out: list[str] = []
    for fn in sources:
        try:
            out.extend(fn() or [])
        except Exception as exc:  # noqa: BLE001  单来源失败不影响其余
            logger.warning("模型局限自动汇总分项失败(跳过): %s", exc)
    return out


REPORT_SECTIONS = [
    "任务背景与业务目标", "样本定义", "数据质量与 leakage audit",
    "split 与 evaluation protocol", "baseline", "Evolution 概览",
    "Leaderboard", "关键有效方向", "关键失败方向", "diminishing-return 方向",
    "最终模型 recipe", "validation / shadow-test / final-test 结果",
    "稳定性与切片表现", "模型局限", "stop reason", "上线/部署建议",
]


class ReportBuilder:
    def __init__(self, run_dir: Path, spec: TaskSpec, config: Config):
        self.run_dir = Path(run_dir)
        self.spec = spec
        self.config = config

    # ---- 方向自动汇总(v10.3)----
    def _auto_directions(self, key: str, trials: list) -> list[str]:
        """Task Experience 缺失时按 verdict/delta 确定性汇总方向。

        - successful_directions:verdict=accepted 的 trial;
        - failed_directions:FAILED 或 verdict=invalid;
        - diminishing_return_directions:分数高于 baseline 但 verdict=
          rejected(noise gate 内,即"有增量但不可分")。
        """
        out = []
        for t in trials:
            v = t.verdict
            st = (t.data.get("result") or {}).get("status")
            hyp = ((t.data.get("hypothesis") or {}).get("statement")
                   or "")[:60]
            pm = spec_pm = None
            ev = t.data.get("evaluation") or {}
            pm = (ev.get("primary_metric") or {}).get("value")
            delta = (ev.get("delta") or {}).get("vs_baseline")
            desc = (f"- {t.trial_id}: {hyp}"
                    + (f"({self.spec.primary_metric}="
                       f"{pm:.4f}, Δbaseline={delta:+.4f})"
                       if isinstance(pm, (int, float))
                       and isinstance(delta, (int, float)) else ""))
            reason = ev.get("reason") or t.data.get("failure_reason") or v
            if key == "successful_directions" and v == "accepted":
                out.append(desc)
            elif key == "failed_directions" and (
                    st == "failed" or v == "invalid"
                    # v2.18.4:被拒的方向也要进"失败方向"并带原因。旧口径
                    # 只认 FAILED/invalid,于是 verdict=rejected 的 7 个
                    # trial(slice 门 / 跨视图不可比)在 §9 完全消失,
                    # 读者看不到"哪些方向试过、为什么没成"。
                    or (v == "rejected" and not (
                        isinstance(delta, (int, float)) and delta > 0))):
                out.append(f"- {t.trial_id}: {hyp}"
                           f"(原因:{reason})")
            elif key == "diminishing_return_directions" and v == "rejected" \
                    and isinstance(delta, (int, float)) and delta > 0:
                out.append(desc)
        return out

    # ---- 数据收集 ----
    def _read(self, rel: str, loader=None):
        p = self.run_dir / rel
        if not p.exists():
            return None
        return (loader or load_json)(p)

    def _best_nonchampion(self, lb: dict) -> dict | None:
        """v2.12.2.5:无冠军时的交付回落候选。

        背景(drawdown-t7-sv24 实测):§11/§12/§13 与 final-model 全部读
        `final-model/*`,而该目录只有 `finalize-retrain` 成功才落盘;确认
        门未过时这三节整片留空(`recipe: {}`、`validation: —`、
        `multi_seed_std: —`),即使 leaderboard 里有 33 条带完整 CI 的
        Trial 也一条都读不出来 —— 交付报告看起来像"什么都没做"。

        回落口径:leaderboard 里 verdict ∈ {confirmed, accepted} 且
        integrity 有效、`failure_reason` 为空的最优行(按 primary_metric
        降序)。只做**只读呈现**,不改变任何交付判定,也不写 final-model。
        """
        rows = [r for r in (lb.get("rows") or [])
                if r.get("verdict") in ("confirmed", "accepted")
                and not r.get("failure_reason")
                and r.get(self.spec.primary_metric) is not None]
        if not rows:
            return None
        # v2.14.0:优先回落到搜索冠军(与交付位解析同源:收缩分、同视图、
        # 非 screening),而不是原始指标最高的行
        _sc = (lb.get("delivery_resolution") or {}).get("search_champion")
        for r in rows:
            if _sc and r.get("trial_id") == _sc:
                return r
        rev = _metric_higher_better(self.spec.primary_metric)
        rows.sort(key=lambda r: r[self.spec.primary_metric], reverse=rev)
        return rows[0]

    def _delivery_section(self, lb: dict, final_eval: dict) -> list[str]:
        """v2.14.0:交付位 vs 搜索冠军(§11 首段)。"""
        res = lb.get("delivery_resolution") or {}
        if not res.get("search_champion"):
            return []
        from .confirmation import divergence_summary

        out = ["### 交付位与搜索冠军", ""]
        out.append(f"- 搜索冠军: `{res.get('search_champion')}`(配方 "
                   f"`{res.get('search_recipe')}`,"
                   f"{self.spec.primary_metric}={res.get('search_metric')})")
        out.append(f"- 交付位解析: `{res.get('status')}` —— "
                   f"{divergence_summary(res)}")
        delivered = final_eval.get("best_trial")
        if delivered:
            dr = final_eval.get("delivery_resolution") or {}
            out.append(f"- 实际盲测/交付的 trial: `{delivered}`"
                       + (f"(落后搜索冠军 {dr.get('gap_vs_search_champion'):+.6f})"
                          if isinstance(dr.get("gap_vs_search_champion"),
                                        (int, float))
                          and dr.get("gap_vs_search_champion") > 0 else ""))
            if delivered != res.get("delivery_trial"):
                out.append(
                    "- ⚠ **实际交付与当前解析的交付位不一致**:盲测可能发生在"
                    "旧口径/冠军更替之前;冠军配方的样本外效果**未被估计**。")
            if dr.get("divergence_acknowledged"):
                out.append("- 交付位分叉经 `--ack-divergence` 用户裁决放行"
                           "(见 run-state.delivery_divergence_acks)")
        if res.get("status") in ("diverged", "unconfirmed"):
            out.append("- **搜索冠军配方没有通过独立确认的交付**:可用模型见 "
                       "`final-model/unconfirmed/`(未确认、无样本外估计)。")
        if res.get("status") == "waived_by_prior":
            # v2.20.0:豁免交付必须与"已确认交付"在报告里可区分 —— 两者的
            # 证据强度不同(前者少一份换 seed 的独立复现证明)。
            w = res.get("prior_waiver") or {}
            out.append(
                "- ⚠ **该交付未经 confirm**,按**先验豁免**放行:"
                f"z={w.get('z')}≥{w.get('z_required')}、"
                f"σ_seed={w.get('sigma_seed')}"
                f"[{w.get('sigma_source')}]、判据 "
                f"`{w.get('criterion')}`。含义:把分数整体挪动 "
                f"{w.get('z_required')} 个 seed 标准差后仍优于 baseline;"
                "但**没有**换 seed 的独立复现证据。")
        out.append("")
        return out

    def _trial_artifact(self, trial_id: str, rel: str, loader=None):
        """读某个 trial 目录下的产物(回落路径用)。"""
        p = self.run_dir / "trials" / str(trial_id) / rel
        if not p.exists():
            return None
        return (loader or load_json)(p)

    def build(self) -> Path:
        profile = self._read("sample-profile/sample-profile.json") or {}
        splits = self._read("sample-profile/splits-summary.json") or {}
        ev_state = self._read("evolution-state.json") or {}
        task_exp = (load_yaml(self.run_dir / "experience/task-experience.yaml")
                    if (self.run_dir / "experience/task-experience.yaml").exists()
                    else {}) or {}
        lb = self._read("leaderboard/leaderboard.json") or {}
        final_eval = self._read("final-model/evaluation.json") or {}
        recipe = (load_yaml(self.run_dir / "final-model/recipe.yaml")
                  if (self.run_dir / "final-model/recipe.yaml").exists() else {}) or {}
        archive = (load_yaml(self.run_dir / "leaderboard/archive.yaml")
                   if (self.run_dir / "leaderboard/archive.yaml").exists() else {}) or {}
        # v2.12.2.5:无冠军时的回落候选(只读呈现,不改交付判定)
        fallback = None
        fallback_ev = None
        if not final_eval:
            fallback = self._best_nonchampion(lb)
            if fallback:
                fallback_ev = (self._trial_artifact(
                    fallback["trial_id"], "evaluation.json") or {})
                if not recipe:
                    recipe = {"_fallback_trial": fallback["trial_id"],
                              "_note": "finalize-retrain 未产出冠军 recipe;"
                                       "以下为该 Trial 的实测配置(非交付配方)"}
                    _cfg = self._trial_artifact(fallback["trial_id"],
                                                "config/trial.yaml",
                                                load_yaml) or {}
                    for _k in ("algorithm", "hyperparameters", "seed",
                               "features", "hpo"):
                        if _cfg.get(_k) is not None:
                            recipe[_k] = _cfg[_k]

        trials_dir = self.run_dir / "trials"
        trials = []
        if trials_dir.exists():
            for d in sorted(trials_dir.iterdir()):
                if (d / "trial.yaml").exists():
                    from .trial import TrialRecord
                    trials.append(TrialRecord.load(d))

        lines: list[str] = [f"# ModelEvo 建模报告 — {self.spec.task_id}", "",
                            f"- 生成时间: {now_iso()}",
                            f"- Harness: {self.config.get('version')}"]

        # 1 任务背景与业务目标
        lines += ["", "## 1. 任务背景与业务目标", ""]
        lines.append(f"- 任务名: {self.spec.data.get('task_name')}")
        lines.append(f"- 目标: {self.spec.data.get('objective_description')}")
        lines.append(f"- 模型类型: {self.spec.model_type}")
        lines.append(f"- 业务域: {', '.join(self.spec.data.get('business_domains') or [])}")

        # 2 样本定义
        # v2.18.4:总样本量取全量加载行数(profile.n_rows),不再取
        # sample_summary.total_rows —— 后者是 profile 分析面(下采样后
        # 的 train split),实测 400w 样本被写成"总样本量: 739444"。
        # 分析面口径的数字一律显式标注来源,避免与全量混淆。
        lines += ["", "## 2. 样本定义", ""]
        s = profile.get("sample_summary") or {}
        n_full = profile.get("n_rows")
        if n_full is None:
            n_full = s.get("total_rows")
        lines.append(f"- 总样本量: {n_full if n_full is not None else 'n/a'}")
        profiled = profile.get("n_rows_profiled")
        if profile.get("sampled_from") or (
                isinstance(n_full, int) and isinstance(profiled, int)
                and profiled != n_full):
            lines.append(
                f"- 分析面(profile 统计口径): {s.get('total_rows', 'n/a')} 行"
                f"(全量 {n_full} 行;train 分析面,正样本率/实体数均按此面)"
                if isinstance(profiled, int) else
                f"- 分析面(profile 统计口径): {s.get('total_rows', 'n/a')} 行")
        if s.get("label_rate") is not None:
            lines.append(f"- 正样本率(train 分析面): {s['label_rate']:.4f}")
        if s.get("target_stats") is not None:
            ts = s["target_stats"]
            lines.append(f"- 目标分布: mean={ts.get('mean', float('nan')):.4f} "
                         f"std={ts.get('std', float('nan')):.4f} "
                         f"min={ts.get('min', float('nan')):.4g} "
                         f"max={ts.get('max', float('nan')):.4g}")
        if s.get("unique_entities") is not None:
            lines.append(f"- 唯一实体数(train 分析面): {s.get('unique_entities')}")
        t = self.spec.data.get("target") or {}
        if self.spec.model_type == "regression":
            tgt_desc = t.get("value_definition") or t.get("positive_definition") or "—"
            lines.append(f"- 标签: {t.get('name')}(回归目标:{tgt_desc};"
                         f"单位: {t.get('unit', '—')})")
        else:
            lines.append(f"- 标签: {t.get('name')} = {t.get('positive_definition', '—')}")
        lines.append(f"- 观察窗口: {t.get('observation_window', '—')};"
                     f"表现窗口: {t.get('performance_window', '—')}")

        # 3 数据质量与 leakage audit
        lines += ["", "## 3. 数据质量与 leakage audit", ""]
        leak = profile.get("leakage") or {}
        cnt = leak.get("severity_counts") or {}
        lines.append(f"- leakage 发现: critical={cnt.get('critical', 0)}, "
                     f"warning={cnt.get('warning', 0)}, info={cnt.get('info', 0)}")
        fs = profile.get("feature_summary") or {}
        lines.append(f"- 特征数: {fs.get('n_features', 'n/a')};"
                     f"常量: {len(fs.get('constant_features', []))};"
                     f"高缺失: {len(fs.get('high_missing_features', []))};"
                     f"高基数: {len(fs.get('high_cardinality_features', []))}")

        # 4 split 与 evaluation protocol
        lines += ["", "## 4. split 与 evaluation protocol", ""]
        lines.append("```yaml")
        import yaml as _yaml
        lines.append(_yaml.safe_dump(self.spec.data.get("split_policy") or {},
                                     allow_unicode=True, sort_keys=False).strip())
        lines.append("```")
        for r in (splits.get("splits") or []):
            rate = r.get("positive_rate")
            if rate is not None:
                desc = f" / 正样本率 {format(rate, '.4f')}"
            elif r.get("target_mean") is not None:
                desc = (f" / 目标均值 {format(r['target_mean'], '.4f')}"
                        f"(std {format(r.get('target_std', float('nan')), '.4f')})")
            else:
                desc = ""
            lines.append(f"- {r.get('split')}: {r.get('rows')} 行{desc}")

        # 5 baseline
        lines += ["", "## 5. baseline", ""]
        baseline_id = lb.get("baseline_id")
        baseline_row = next((r for r in lb.get("rows", [])
                             if r.get("trial_id") == baseline_id), None)

        def _metric(row):
            # 行内指标列名为实际指标名(auc 等);兼容旧 primary_metric 列
            return row.get(self.spec.primary_metric,
                           row.get("primary_metric"))

        if baseline_row:
            lines.append(f"- {baseline_id} ({baseline_row.get('algorithm')}): "
                         f"{self.spec.primary_metric}="
                         f"{_metric(baseline_row)}")
        elif baseline_id:
            # v12(P2,O6):§5 与 §7 的一致性——baseline_id 已解析但行内
            # 无主指标时如实写"无主指标"(旧文案"baseline 未完成"与
            # §7 出现 Δbaseline 值自相矛盾)
            lines.append(f"- {baseline_id}: 无主指标值(评估未完成或失败;"
                         "delta 口径见 leaderboard)")
        else:
            lines.append("- baseline 未完成")

        # 6 Evolution 概览
        lines += ["", "## 6. Evolution 概览", ""]
        # v12(P2,O6):generations 计数与 leaderboard 实际最大代对齐
        # (gen-04 空转一代被 finalize 后旧口径仍按 n_generations 报 5,
        # 报告内部自相矛盾)
        lb_max_gen = max((int(r.get("generation") or 0) for r in
                          (lb.get("rows") or [])), default=None)
        n_gens_report = ev_state.get("n_generations", 0)
        if lb_max_gen is not None and lb_max_gen + 1 < n_gens_report:
            lines.append(f"- generations: {n_gens_report}"
                         f"(含空代;实际有 trial 的最大代 gen-"
                         f"{lb_max_gen:02d})")
        else:
            lines.append(f"- generations: {n_gens_report}")
        lines.append(f"- trials(进入过 RUNNING): {len(trials)}")
        acc = sum(1 for t in trials
                  if t.verdict in ("accepted", "confirmed"))
        lines.append(f"- accepted/confirmed: {acc};"
                     f" rejected/invalid: "
                     f"{sum(1 for t in trials if t.verdict in ('rejected', 'invalid'))}"
                     f"; unconfirmed(confirm 未通过独立确认): "
                     f"{sum(1 for t in trials if t.verdict == 'unconfirmed')}")

        # 7 Leaderboard
        lines += ["", "## 7. Leaderboard", ""]
        rows = lb.get("rows") or []

        def _fm(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

        if rows:
            lines.append("| rank | trial | 算法 | 模型数 | 代 | 动作 | "
                         f"{self.spec.primary_metric} | Δparent | Δbaseline "
                         "| p_positive | verdict |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for r in rows[:20]:
                ppos = r.get("p_positive")
                ppos_s = f"{ppos:.3f}" if isinstance(
                    ppos, (int, float)) else "—"
                lines.append(
                    f"| {r.get('rank')} | {r.get('trial_id')} | {r.get('algorithm')} "
                    f"| {r.get('model_num', 1)} "
                    f"| {r.get('generation')} | {r.get('search_action')} "
                    f"| {_fm(_metric(r))} | {_fm(r.get('delta_vs_parent'))} "
                    f"| {_fm(r.get('delta_vs_baseline'))} | {ppos_s} "
                    f"| {r.get('verdict') or '—'} |")
            lines.append("")
            lines.append(f"- 累计交付模型数(total_model_num): "
                         f"{lb.get('total_model_num', '—')}"
                         "(单模型 trial 计 1,集成计其声明模型数)")
            # v2.18.4:未通过的 trial 必须带原因(交付校验「leaderboard
            # 含 failed/invalid 与原因」)。原因在 evaluation.reason,
            # 旧表不带该列,读者只能看到空白的 failure_reason。
            _rej = [(r.get("trial_id"), r.get("verdict"),
                     r.get("verdict_reason") or r.get("failure_reason"))
                    for r in rows
                    if (r.get("verdict") or "") in
                    ("rejected", "invalid", "unconfirmed")
                    or r.get("failure_reason")]
            if _rej:
                lines.append("")
                lines.append("未通过 trial 与原因:")
                for _tid, _v, _r in _rej:
                    lines.append(f"- {_tid}: verdict={_v};"
                                 f"原因={_r or '未记录'}")
        else:
            lines.append("Leaderboard 为空。")

        # 8/9/10 关键方向
        def _directions(key: str, empty: str) -> list[str]:
            items = task_exp.get(key) or []
            out = []
            for it in items:
                if isinstance(it, dict):
                    _why = it.get("reason") or it.get("failure_reason")
                    if not _why:
                        # v2.18.4:task-experience 由旧版落盘时只记
                        # failure_reason(运行时失败,方向性拒绝恒为 null)。
                        # 用 trial 的 evaluation.reason 补齐,否则 §9 只有
                        # trial 名 + 假设,读者看不出为什么被拒。
                        _t = next((x for x in trials
                                   if x.trial_id == it.get("trial_id")), None)
                        if _t is not None:
                            _why = ((_t.data.get("evaluation") or {})
                                    .get("reason"))
                    out.append(f"- {it.get('trial_id', '')}: "
                               f"{it.get('hypothesis', it)}"
                               + (f"(原因:{_why})" if _why else ""))
                else:
                    out.append(f"- {it}")
            if out:
                return out
            # v10.3:Task Experience 未写时从 leaderboard 自动汇总,
            # 报告不再依赖收尾 Agent 手工誊写(方向分类按 verdict/
            # delta 确定性规则,LLM 只负责收尾时补充叙事性经验)
            return self._auto_directions(key, trials) or [f"- {empty}"]

        lines += ["", "## 8. 关键有效方向", "", *_directions(
            "successful_directions", "无 accepted 方向")]
        lines += ["", "## 9. 关键失败方向", "", *_directions(
            "failed_directions", "无记录")]
        # v2.17.2(A1):撞 cap 混杂的负结论紧随失败方向提示(不作负证据)
        try:
            _ccl = _lim_cap_confounded(self.run_dir, self.spec, trials)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cap_confounded 汇总失败(跳过): %s", exc)
            _ccl = []
        if _ccl:
            lines += ["", f"> ⚠ {_ccl[0]}"]
        lines += ["", "## 10. diminishing-return 方向", "", *_directions(
            "diminishing_return_directions", "无记录")]

        # 11 最终模型 recipe
        lines += ["", "## 11. 最终模型 recipe", ""]
        # v2.14.0:交付位与搜索冠军的关系放在 recipe 之前 —— run2 的报告
        # 写 best_trial=trial-011,而全 run 最优配方落后/领先多少、为什么
        # 没交付它,读者从报告里看不出来。
        lines += self._delivery_section(lb, final_eval)
        if fallback and not final_eval:
            lines.append(
                f"> **无已确认冠军**(`final-model/recipe.yaml` 缺失)。以下为 "
                f"回落展示的最优非冠军 Trial `{fallback['trial_id']}` 的实测"
                f"配置,仅供参考,**不是交付配方**;未通过交付校验项见 §15。")
            lines.append("")
        lines.append("```yaml")
        lines.append(_yaml.safe_dump(recipe, allow_unicode=True,
                                     sort_keys=False).strip() or "{}")
        lines.append("```")

        # 12 validation / shadow / final test
        lines += ["", "## 12. validation / shadow-test / final-test 结果", ""]
        better = "越小越好" if not _metric_higher_better(
            self.spec.primary_metric) else "越大越好"
        lines.append(f"- 指标方向: {self.spec.primary_metric} {better}")
        for k in ("validation", "shadow_test", "final_test"):
            v = final_eval.get(k)
            if isinstance(v, dict):
                pm = v.get(self.spec.primary_metric)
                _extra = ", ".join(
                    f"{m}={x:.6g}" for m, x in v.items()
                    if m != self.spec.primary_metric
                    and isinstance(x, (int, float)))
                lines.append(f"- {k}: {self.spec.primary_metric}="
                             f"{pm if isinstance(pm, float) else pm}"
                             + (f"({_extra})" if _extra else ""))
            else:
                lines.append(f"- {k}: {'—' if k != 'final_test' else '未评测(盲测保留)'}")
        # v2.17.2(C4):交付 trial 的确认门(阻断)与 shadow gap 诊断(非阻断)
        # 分行呈现;shadow 位读 gap_sanity_passed,旧产物回落 passed
        _gate_tid = (final_eval.get("best_trial")
                     or (lb.get("delivery_resolution") or {}).get(
                         "delivery_trial")
                     or (fallback or {}).get("trial_id"))
        if _gate_tid:
            try:
                _gate_ev = next(((t.data.get("evaluation") or {})
                                 for t in trials if t.trial_id == _gate_tid),
                                None) or (self._trial_artifact(
                                    _gate_tid, "evaluation.json") or {})
                lines += _confirmation_gate_lines(_gate_ev, _gate_tid)
            except Exception as exc:  # noqa: BLE001  诊断呈现不阻塞报告
                logger.warning("确认门/shadow 诊断呈现失败: %s", exc)
        # v2.15.9:交付重训口径与 shadow 三方对照(交付模型 / 冠军 train-only
        # 搜索模型 / 基线 train-only)。交付模型含 validation 期数据,基线只用
        # train —— 样本外提升里包含"多用了一段更新数据"的贡献,须如实标注。
        _meta = self._read("final-model/model_meta.json") or {}
        if _meta.get("refit_mode"):
            lines.append(f"- 交付重训口径: refit_mode={_meta.get('refit_mode')}"
                         f";{_meta.get('retrain_scope')}")
            _eh = _meta.get("es_holdout") or {}
            if _eh.get("first_pday"):
                lines.append(f"  - 早停面: train+validation 尾部 "
                             f"{_eh.get('n_rows')} 行"
                             f"({_eh.get('first_pday')} ~ {_eh.get('last_pday')})")
            elif _meta.get("es_source") == "validation":
                # v2.16.0:搜索期在 outer validation 上早停,交付冻结其最优轮数
                lines.append("  - 早停面: outer validation(搜索期单段拟合);"
                             "交付按 train-only 复现的最优轮数冻结重训。"
                             "validation 参与了早停,其分数略偏乐观,样本外口径以 "
                             "shadow_test / final_test 为准")
            # v2.17.0(#5):refit 口径回落留痕
            if _meta.get("refit_mode_fallback_reason"):
                lines.append(
                    f"  - ⚠ 口径回落: {_meta.get('refit_mode_requested')} → "
                    f"{_meta.get('refit_mode_effective')};"
                    f"{_meta.get('refit_mode_fallback_reason')}")
            _cp = _meta.get("clone_probe") or {}
            if _cp.get("status") not in (None, "not_required"):
                lines.append(f"  - clone 探测: {_cp.get('status')}"
                             + (f"({_cp.get('error')})"
                                if _cp.get("error") else ""))
        # v2.17.0(#20):回归水位诊断(交付模型 / 基线;gate 内算好的统计量)
        # v2.17.0(D8):整体与分期告警分开表述;判据 |ratio−1| >
        # max(固定阈, z·SE)(旧产物无 SE 字段时沿用落盘告警位)
        _rd = final_eval.get("regression_diagnostics") or {}
        for _split in ("final_test", "shadow_test"):
            _face = _rd.get(_split) or {}
            _dl = _face.get("delivery") or {}
            _lv = _dl.get("level") or {}
            if _fnum(_lv.get("level_ratio")) is None:
                continue
            _blv = ((_face.get("baseline") or {}).get("level") or {})
            _se = _fnum(_lv.get("level_ratio_se"))
            _thr = _fnum(_lv.get("level_ratio_threshold"))
            _ov_warn = bool(_lv.get("level_shift_warning"))
            lines.append(
                f"- 回归水位({_split})· 整体: 交付模型 pred_mean="
                f"{_g(_lv.get('pred_mean'))} / y_mean={_g(_lv.get('y_mean'))}"
                f" = {_g(_lv.get('level_ratio'), '.3f')}"
                + (f"(SE {_se:.3f},告警阈 ±{_thr:.3f})"
                   if _se is not None and _thr is not None else "")
                + (f";基线 {_g(_blv.get('level_ratio'), '.3f')}"
                   if _fnum(_blv.get("level_ratio")) is not None else "")
                + (" ⚠ 整体水位偏离超阈值:排序可用、绝对金额需校准" if _ov_warn
                   else ("(超过固定阈但在 z·SE 噪声带内,不告警)"
                         if _lv.get("level_within_noise") else "")))
            _per = [r for r in (_dl.get("level_by_period") or [])
                    if isinstance(r, dict)
                    and _fnum(r.get("level_ratio")) is not None]
            if _per:
                _bad = [r for r in _per if r.get("level_shift_warning")]
                _noise = [r for r in _per if r.get("level_within_noise")]
                if _bad:
                    _txt = (f"{len(_bad)}/{len(_per)} 期超出噪声带:" + ", ".join(
                        f"{r.get('period')}={_g(r.get('level_ratio'), '.3f')}"
                        + (f"±{_fnum(r.get('level_ratio_se')):.3f}"
                           if _fnum(r.get("level_ratio_se")) is not None
                           else "") for r in _bad[:6])
                        + " ⚠ 分期水位偏离"
                        + ("(整体水位在阈内,属局部偏离)" if not _ov_warn
                           else ""))
                else:
                    _mx = max(abs(_fnum(r.get("level_ratio")) - 1.0)
                              for r in _per)
                    _txt = (f"{len(_per)} 期均在噪声带内(最大 |ratio−1|="
                            f"{_mx:.3f}"
                            + (f",其中 {len(_noise)} 期超固定阈但 < z·SE"
                               if _noise else "") + "),不告警")
                lines.append(f"- 回归水位({_split})· 分期"
                             f"({_rd.get('period_column') or 'period'}): {_txt}")
        if final_eval.get("shadow_test_source"):
            lines.append(f"- shadow_test 分数来源: "
                         f"{final_eval.get('shadow_test_source')}")
        for k, title in (("shadow_test_search_model",
                          "shadow_test(冠军 train-only 搜索模型)"),
                         ("baseline_shadow_test",
                          "baseline shadow_test(train-only)")):
            v = final_eval.get(k)
            if isinstance(v, dict):
                lines.append(f"- {title}: {self.spec.primary_metric}="
                             f"{v.get(self.spec.primary_metric)}")
        _imp_sh = final_eval.get("improvement_vs_baseline_shadow_test")
        if _imp_sh is not None:
            lines.append(f"- 样本外提升(vs baseline,shadow_test 口径): "
                         f"{_imp_sh:+.4f}")
        if _meta.get("refit_mode") in ("rerun_recipe", "frozen_rounds") and (
                final_eval.get("baseline_final_test")
                or final_eval.get("baseline_shadow_test")):
            lines.append("- 口径注:交付模型在 train+validation 上重训,基线只用 "
                         "train;对基线的样本外提升包含'多用一段更新数据'的贡献")
        _raw = final_eval.get("raw_scale") or {}
        if _raw:
            lines.append(f"- 原值尺度({_raw.get('inverse_transform')}):"
                         f"{_raw.get('note')}")
            for _sp in ("final_test", "shadow_test"):
                r = _raw.get(_sp)
                if isinstance(r, dict):
                    lines.append(
                        f"  - {_sp}: RMSE={r.get('rmse'):.2f} "
                        f"MAE={r.get('mae'):.2f} "
                        f"均值偏置={r.get('mean_bias'):.2f} "
                        f"总额比={r.get('sum_ratio')} "
                        f"Spearman={r.get('spearman'):.4f}"
                        f";分桶表 final-model/{r.get('segment_report')}/")
        # v2.12.2.5:无冠军时回落展示最优非冠军 Trial 的两面实测分数。
        # 旧实现只读 final-model/evaluation.json,确认门未过时整节留空,
        # 33 条带 CI 的 Trial 一条都读不出来。
        if fallback and not final_eval:
            _fv = fallback.get(self.spec.primary_metric)
            # v2.12.2.6:`shadow_test_score.parquet` 是**二进制 parquet**,
            # 不能过默认的 load_json(实测 UnicodeDecodeError,整份 report
            # 在写盘前崩溃)。这里只需要"是否存在",直接查路径。
            _sd = ((self.run_dir / "trials" / str(fallback["trial_id"])
                    / "shadow_test_score.parquet").exists())
            lines.append("")
            lines.append(
                f"**回落口径(非交付结论)**:最优非冠军 "
                f"`{fallback['trial_id']}`({fallback.get('verdict')}) — "
                f"validation {self.spec.primary_metric}={_fv}")
            if fallback.get("delta_vs_baseline") is not None:
                lines.append(f"- Δvs baseline: "
                             f"{fallback['delta_vs_baseline']:+.6f}")
            if fallback.get("delta_ci"):
                lines.append(f"- 配对 CI: {fallback['delta_ci']}")
            if _sd:
                lines.append("- shadow_test 分数已落盘"
                             "(`trials/<id>/shadow_test_score.parquet`),"
                             "其 AUC 见该 Trial 的 evaluation.json")
            lines.append("- **该 Trial 无 `fresh_confirmation.passed=true`"
                         ",不得作为已确认交付冠军。**")
        # v12(P0-8):baseline 的 final_test 口径——"提升了多少"首次可测量
        bl_final = final_eval.get("baseline_final_test") or {}
        imp_final = final_eval.get("improvement_vs_baseline_final_test")
        if bl_final.get(self.spec.primary_metric) is not None:
            lines.append(f"- baseline final_test: {self.spec.primary_metric}="
                         f"{bl_final.get(self.spec.primary_metric)}")
        if imp_final is not None:
            lines.append(f"- **样本外提升(vs baseline,final_test 口径): "
                         f"{imp_final:+.4f}**"
                         "(validation 口径的 improvement_vs_baseline 见 "
                         "final-model/evaluation.json;样本外口径才是"
                         "交付结论)")
        elif final_eval.get("final_test"):
            lines.append("- baseline 未上 final_test:样本外提升不可验证"
                         "(交付前用 baseline 模型对 final_test 打分,"
                         "产物 final-model/baseline_final_test_score.parquet)")
        # v2.10.1/F8a:交付层打分口径修正提示。评估口径与被交付模型口径
        # 不一致时(典型:交付打分对 NaN 做 fillna,而模型是 NaN 原生缺失
        # 路由),上面的样本外提升会失真甚至反号。有修正记录就必须紧邻
        # 原始数字呈现,不能让读者只看到一边。
        _sc = final_eval.get("scoring_contract_note") or {}
        if _sc:
            lines += ["", "### 交付打分口径修正(必须与上表同时阅读)", ""]
            lines.append(f"- 缺陷:{_sc.get('issue')}")
            if _sc.get("scope"):
                lines.append(f"- 影响范围:{_sc.get('scope')}")
            lines.append(
                f"- final_test: {_sc.get('final_test_auc_as_recorded_fillna0')}"
                f"(记录版) / **{_sc.get('final_test_auc_corrected_nan_native')}**"
                f"(修正版)")
            if _sc.get("shadow_auc_corrected_nan_native") is not None:
                lines.append(
                    f"- shadow_test: {_sc.get('shadow_auc_as_recorded_fillna0')}"
                    f"(记录版) / **{_sc.get('shadow_auc_corrected_nan_native')}**"
                    f"(修正版)")
            lines.append(
                f"- 样本外提升: {_sc.get('improvement_vs_baseline_final_test_as_recorded')}"
                f"(记录版) → **{_sc.get('improvement_vs_baseline_final_test_corrected')}**"
                f"(修正版)")
            _pb = _sc.get("corrected_paired_bootstrap") or {}
            if _pb:
                lines.append(
                    f"- 修正版配对 bootstrap: CI={_pb.get('ci')}, "
                    f"p_positive={_pb.get('p_positive')}, "
                    f"noise={_pb.get('noise_MAD')}")
            if _sc.get("caveat"):
                lines.append(f"- 留档口径:{_sc.get('caveat')}")
            lines.append("- **这不是模型问题,是交付层打分口径与训练/"
                         "选型口径不一致。**")

        # 13 稳定性与切片
        lines += ["", "## 13. 稳定性与切片表现", ""]
        stab = final_eval.get("stability") or {}
        # v2.12.2.5:无冠军时回落读最优非冠军 Trial 的 evaluation.json。
        # 旧实现只读 final-model/evaluation.json ⇒ 确认门未过时本节的
        # slice/segment/temporal 证据(在 trial 层是齐的)全部显示为 "—"。
        if fallback and not final_eval and fallback_ev:
            stab = fallback_ev.get("stability") or {}
        # v2.18.4:evaluate-final 落盘的 final-model/evaluation.json 只含
        # 盲测/影子指标,不含 stability —— 本行因此恒为 "—",即使冠军
        # trial 的 evaluation.json 里有实测的 multi_seed_std(confirm 三
        # seed 才有)。回落读冠军 trial 的 stability,并注明来源。
        _stab_src = "final-model/evaluation.json"
        if stab.get("multi_seed_std") is None:
            _cid = (final_eval.get("best_trial")
                    or final_eval.get("search_champion"))
            if _cid:
                _rev = self._trial_artifact(_cid, "evaluation.json") or {}
                _rs = _rev.get("stability") or {}
                if _rs.get("multi_seed_std") is not None:
                    stab = _rs
                    _stab_src = f"trials/{_cid}/evaluation.json"
        lines.append(f"- multi_seed_std: {stab.get('multi_seed_std', '—')}"
                     + (f"(来源 {_stab_src})"
                        if stab.get("multi_seed_std") is not None else ""))
        if fallback and not final_eval:
            lines.append(
                f"- 以下切片/分段证据来自回落 Trial `{fallback['trial_id']}`"
                "(非交付冠军)")
        ci = stab.get("bootstrap_ci")
        if ci:
            lines.append(f"- bootstrap_ci: [{ci[0]:.4f}, {ci[1]:.4f}]")
        _ev13 = fallback_ev if (fallback and not final_eval) else final_eval
        # v2.18.4:final-model/evaluation.json 同样不含 slice/temporal ——
        # 有冠军时这两行恒缺失。缺键时回落冠军 trial 的 evaluation.json。
        if final_eval and not fallback_ev and not (
                _ev13.get("slice_evaluation")
                or _ev13.get("temporal_evaluation")):
            _cid13 = (final_eval.get("best_trial")
                      or final_eval.get("search_champion"))
            if _cid13:
                _rev13 = self._trial_artifact(_cid13, "evaluation.json") or {}
                if _rev13.get("slice_evaluation") \
                        or _rev13.get("temporal_evaluation"):
                    _ev13 = _rev13
                    lines.append(
                        f"- 切片/时序证据来自冠军 trial "
                        f"`{_cid13}` 的 evaluation.json")
        sl = _ev13.get("slice_evaluation") or {}
        if sl:
            lines.append(f"- 最差切片: {sl.get('worst_slice')} "
                         f"(Δ={sl.get('worst_slice_delta')})")
            if sl.get("passed") is not None:
                lines.append(f"- 切片门 passed: {sl.get('passed')}")
        _tp = _ev13.get("temporal_evaluation") or {}
        if _tp:
            lines.append(f"- 时序门: worst_delta={_tp.get('worst_delta')} "
                         f"passed={_tp.get('passed')}")
        seg = _ev13.get("segment_evaluation") or {}
        if seg and seg.get("bin_means"):
            means = " / ".join(f"{m:.4f}" for m in seg["bin_means"])
            lines.append(f"- 分段排序性({seg.get('n_bins')} 段): "
                         f"monotonicity={seg.get('monotonicity')} "
                         f"spearman={seg.get('spearman')} "
                         f"lift_top={seg.get('lift_top')} "
                         f"passed={seg.get('passed')}")
            lines.append(f"- 段均值(低→高): {means}")
        # final_test 分桶报表(若 evaluate-final 已产出,指引读者)
        seg_rep = final_eval.get("segment_report") or {}
        if seg_rep:
            lines.append(f"- 全量分桶表(桶内正样本率/误差/增量与累计口径):"
                         f" final-model/{seg_rep.get('xlsx')}"
                         f"(可读版 final-model/{seg_rep.get('md')})")

        # 14 模型局限
        lines += ["", "## 14. 模型局限", ""]
        _lims = (final_eval.get("limitations") or
                 task_exp.get("limitations") or [])
        # v2.12.2.5:无冠军/无记录时,回落汇总"为什么没有交付冠军"的
        # 结构化说明(确认门失败原因写在 trial 的 evaluation.json 里,
        # 旧实现一律显示"未记录")。
        if not _lims and fallback and not final_eval:
            _fc = fallback_ev.get("fresh_confirmation") or {}
            _lims = [
                f"无 fresh_confirmation.passed=true 的交付冠军;"
                f"回落 Trial {fallback['trial_id']} 的 verdict="
                f"{fallback.get('verdict')}",
            ]
            if _fc:
                _lims.append(
                    f"确认门(回落 Trial): criterion="
                    f"{_fc.get('criterion') or 'legacy_two_sided'}, "
                    f"validation_passed="
                    f"{_fc.get('validation_passed', _fc.get('validation_equivalence_passed'))}, "
                    f"delta_vs_parent={_fc.get('delta_vs_parent')}, "
                    f"margin_lower={_fc.get('margin_lower', _fc.get('equivalence_margin'))}, "
                    f"sigma_seed={_fc.get('sigma_seed')}, "
                    f"passed={_fc.get('passed')}")
            _sl = fallback_ev.get("slice_evaluation") or {}
            if _sl:
                _lims.append(
                    f"最弱切片 {_sl.get('worst_slice')} "
                    f"Δ={_sl.get('worst_slice_delta')} "
                    f"passed={_sl.get('passed')}")
            _lims.append(
                "确认门失败不等于候选能力不足;量化归因见该 Trial 的 "
                "evaluation.json(fresh_confirmation / slice_evaluation / "
                "temporal_evaluation 三块)与 run 的 logs/ 反思")
        # v2.17.0(#21):从磁盘产物自动汇总的局限(水位偏差/撞 cap/未交付
        # 候选/单视图/抽样面/指标背离/harness 变更/refit 回落),缺失项跳过
        _lims = list(_lims or [])
        try:
            _auto = collect_limitations(
                self.run_dir, self.spec, final_eval=final_eval,
                meta=self._read("final-model/model_meta.json") or {},
                lb=lb, ev_state=ev_state,
                run_state=self._read("run-state.json") or {}, trials=trials)
        except Exception as exc:  # noqa: BLE001  汇总失败不阻塞报告
            logger.warning("模型局限自动汇总失败: %s", exc)
            _auto = []
        _lims += [x for x in _auto if x not in _lims]
        for lim in (_lims or ["未记录"]):
            lines.append(f"- {lim}")

        # 15 stop reason
        lines += ["", "## 15. stop reason", ""]
        # v2.17.0(D4):同配方(confirm 链/复现)不是"未交付更优候选"
        stop = _correct_stop_same_recipe(self.run_dir, ev_state.get("stopped"),
                                         trials)
        if stop:
            lines.append(f"- 停止原因: **{stop.get('stop_reason')}**")
            # v2.17.0:WP1 stop_report 新字段(存在才呈现)
            if stop.get("advisory") is not None:
                lines.append("- 停止判定性质: " + (
                    "advisory(建议性,可继续搜索)" if stop.get("advisory")
                    else "硬停止") + (
                    f";硬拦截项 {stop.get('blocks')}" if stop.get("blocks")
                    else ""))
            _rdt = stop.get("reason_detail")
            if isinstance(_rdt, dict) and _rdt:
                lines.append("- 停止原因明细:")
                lines += [f"  - `{k}`: {v}" for k, v in _rdt.items()]
            elif _rdt:
                lines.append(f"- 停止原因明细: {_short(_rdt, 400)}")
            if _undelivered_positive(stop.get("undelivered_gain")):
                lines.append(f"- 未交付增益(仅 validation 证据、含选择偏差): "
                             f"{_fmt_undelivered(stop.get('undelivered_gain'))}")
            lines.append(f"- 累计提升 vs baseline: "
                         f"{stop.get('cumulative_improvement_vs_baseline')}")
            lines.append(f"- 近几代 best delta: "
                         f"{stop.get('recent_generations', {}).get('best_delta')}")
            lines.append(f"- 未执行的高潜力方向: "
                         f"{stop.get('remaining_high_potential_directions') or '无'}")
            # [FIX-2026-09-17] 被特征期门(v2.16.0 前为 capacity 闸门)
            # 拒绝、因而从未执行的方向必须显式列出。旧实现只报 stop
            # report 里的 remaining_high_potential_directions(它来自
            # --unexplored 人工入参),于是"被闸门拒掉的 model/HPO"
            # 既不在未执行清单里、也不在任何报告位置 —— 实测
            # drawdown/t7 报告写"未执行的高潜力方向: 无",而 HPO
            # 恰恰是唯一被官方证实有效的剩余杠杆、且被闸门拒了 3 代。
            from .feature_phase import BATCH_KEY as _FP_KEY

            _cg = self._read("planner/latest.json",
                             lambda p: _json.loads(
                                 p.read_text(encoding="utf-8")))
            _cg = (_cg or {}).get(_FP_KEY) or {}
            _blocked = _cg.get("blocked_proposal_ids") or []
            _released = _cg.get("released_proposal_id")
            if _blocked or _released:
                lines.append(
                    f"- **被特征期门拒绝的方向**: "
                    f"{len(_blocked)} 个 proposal"
                    f"(最新批次:已完成 {_cg.get('feature_generations_done')}"
                    f"/{_cg.get('min_generations')} 代)"
                    f"{';已按收敛判定释放 ' + str(_released) if _released else ''}")
                if _cg.get("release_reason"):
                    lines.append(f"  - 释放理由: {_cg['release_reason']}")
                lines.append(
                    "  - 说明:被拒方向不等于已探索过,报告中不作为"
                    "「已穷尽」证据")
            lines.append(f"- 接近上限置信度: "
                         f"{stop.get('confidence_of_near_ceiling')}")
        else:
            lines.append("- 搜索未触发停止条件(可能仍在进行或被人工停止)")

        # 15b 验收目标与缺口(v2.10.1/F8b)
        _pt = (self.spec.data.get("performance_target") or {})
        _tgt = _pt.get(self.spec.primary_metric)
        _deliv = None
        # v2.11.7:_deliv_src 是**条件绑定**——只在循环里命中任一指标时
        # 赋值,却在循环外被无条件引用(条件表达式的判断位也要先求值)。
        # final_eval 无任何指标时(交付链中断/未完成)旧实现必抛
        #   UnboundLocalError: local variable '_deliv_src' referenced
        #   before assignment (runtime/report.py:428)
        # ——"交付失败时报告也生不出来",掩蔽第一现场。预置 None 即可。
        _deliv_src = None
        for _k in ("final_test", "shadow_test", "validation"):
            _v = final_eval.get(_k) or {}
            if _v.get(self.spec.primary_metric) is not None:
                _deliv = _v.get(self.spec.primary_metric)
                _deliv_src = _k
                break
        # v2.10.1/F8b 补:交付打分口径有修正记录时,缺口必须按**修正值**算
        # (记录值是被交付层 fillna 缺陷压低的那个数;用它算缺口会虚报,
        #  实测差 0.0136 —— 与 §12 的修正节自相矛盾)。
        _scn = final_eval.get("scoring_contract_note") or {}
        _corr = _scn.get(f"{_deliv_src}_auc_corrected_nan_native") \
            if _deliv_src else None
        if isinstance(_corr, (int, float)) and _corr != _deliv:
            lines += ["", f"> 注:下表缺口按**修正口径**计算"
                          f"({_deliv_src} {_deliv} → {_corr});"
                          f"原始记录值见 §12 与 final-model/"
                          f"scoring-correction.json。"]
            _deliv = _corr
            _deliv_src = f"{_deliv_src}(修正口径)"
        if _tgt is not None or _deliv is not None:
            lines += ["", "## 15b. 验收目标与信息量天花板", ""]
            if _tgt is not None:
                lines.append(f"- 业务目标: {self.spec.primary_metric} = {_tgt}")
            if _deliv is not None:
                lines.append(f"- 交付实测({_deliv_src}): "
                             f"{self.spec.primary_metric} = {_deliv}")
            if _tgt is not None and _deliv is not None:
                try:
                    _gap = float(_tgt) - float(_deliv)
                except (TypeError, ValueError):
                    _gap = None
                if _gap is not None:
                    lines.append(f"- **缺口: {_gap:+.4f}**"
                                 f"({'达标' if _gap <= 0 else '未达标'})")
                    _noise = (final_eval.get("stability") or {}).get("noise")
                    if isinstance(_noise, (int, float)) and _noise > 0:
                        lines.append(
                            f"- 口径:缺口 ≈ **{abs(_gap)/_noise:.0f} × 噪声**"
                            f"(noise={_noise:.5f});缺口远大于噪声时,"
                            f"属信息量问题而非调参问题")
            _cap = self._read("sample-profile/capacity-report.json",
                              lambda p: _json.loads(
                                  p.read_text(encoding="utf-8")))
            if _cap:
                _ss = _cap.get("sample_size_curve") or {}
                _fc = _cap.get("feature_count_curve") or {}
                _in = _cap.get("irreducible_noise") or {}
                lines.append(
                    f"- 天花板诊断:样本量曲线 still_steep="
                    f"{_ss.get('still_steep')}(末段相对增益 "
                    f"{_ss.get('last_step_gain_rel')});列数曲线 saturated="
                    f"{_fc.get('saturated')}(末段增益 {_fc.get('tail_gain')});"
                    f"局部歧义指数 "
                    f"{_in.get('local_ambiguity_index', _in.get('noise_bound'))}"
                    "（启发式，非不可约误差下界）")
                lines.append("- 判读(抽样面参考,不参与放行):列数饱和 = "
                             "加特征边际小;样本量仍陡 = 抽样面偏小或数据侧"
                             "仍有余量(前者由全量重训兑现);局部歧义指数"
                             "偏高 = 当前表示下近邻标签分歧大,不是误差下界")
                # v2.17.2(C2):目标可达性(用最新 trial 实测重算,失败回落
                # capacity-report 落盘值)
                try:
                    from .capacity import assess_target_reachability

                    _tr = assess_target_reachability(self.run_dir, _cap,
                                                     self.spec)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("目标可达性重算失败(回落落盘值): %s", exc)
                    _tr = _cap.get("target_reachability") or {}
                if (_tr or {}).get("verdict"):
                    lines.append(
                        f"- **目标可达性: `{_tr['verdict']}`**"
                        f"(外推上界 {_g(_tr.get('extrapolated_ceiling'))},"
                        f"当前最优 {_g(_tr.get('best_observed'))}):"
                        f"{_tr.get('rationale')}")
            _caveat = self._read("final-model/delivery-caveats.md",
                                 lambda p: p.read_text(encoding="utf-8"))
            if _caveat:
                lines += ["", "### 交付口径依赖", "",
                          _demote_headings(_caveat,
                                           drop_leading_title=True)]

        # 16 上线建议
        lines += ["", "## 16. 上线/部署建议", ""]
        lines.append(f"- 建议监控特征 PSI(训练 vs 线上),"
                     "重点特征见 sample-profile/drift-profile.parquet")
        lines.append(f"- 上线前建议用最新时间片做一次 shadow 验证")
        if recipe:
            lines.append(f"- 模型入口: final-model/(recipe + 模型文件)")

        out = self.run_dir / "report" / "report.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out
