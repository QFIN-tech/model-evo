"""Planner Context:一键组装 Planner 决策上下文(§21.2,v8)。

八节(全部只读落盘产物,缺失容忍):
1 task_spec        TaskSpec.summary()
2 profile          sample-profile.json 摘要 + compression_audit(data_view 候选)
3 leaderboard_top  leaderboard.json rows 前 top_k(精简列)
4 archive_top      CandidateArchive.top(top_k)
5 budget           run-state + evolution-state:n_trials/n_generations/wall_time
6 data_views_tried trials 的 data_view 全集(未声明归 v1)
7 experience       经验库视图(固定水位):卷宗 / 方向表 / 避免清单 / 热启动榜单(v2.18.0)
8 last_reflection  上一代 reflection 的 key_findings 等(截断)

v2.17.0 新增:mde(runtime.mde.mde_info,含 scale_warning,渲染在最显著
位置)、promotion_preview(evolution.promotion_threshold_preview:本代过
校正所需 z 与 Δ 估计,接口缺失时降级)、sample_noise_precheck(抽样面
manifest.noise_precheck)、target_scale(目标口径);7c warm baseline 只
展示主指标名 + 目标口径一致的条目(v2.18.0 起第 7 节改读经验库视图)。

主会话/Generation subagent 用本命令替代"自行读八个文件",
消除漏读导致的计划质量退化。

v2.17.2(C2):第 10 节呈现目标可达性(capacity.assess_target_reachability,
用最新 trial 实测重算)。
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .leaderboard import CandidateArchive
from .logging_utils import get_logger, load_json, load_yaml
from .spec import TaskSpec
from .trial import TrialFactory

logger = get_logger(__name__)

# leaderboard 行保留的精简列(完整列见 leaderboard.csv)
_LB_COLUMNS = ("trial_id", "algorithm", "generation", "search_action",
               "data_view", "primary_metric", "delta_vs_baseline",
               "p_positive", "verdict")


def build_planner_context(run_dir: Path, spec: TaskSpec, config: Config, *,
                          top_k: int = 5) -> dict:
    run_dir = Path(run_dir)

    # ---- 1) TaskSpec summary ----
    task_spec = spec.summary()

    # ---- 2) profile 摘要(含 v8 compression_audit;v13 leak 分类)----
    profile: dict = {"ready": False}
    sp_path = run_dir / "sample-profile" / "sample-profile.json"
    if sp_path.exists():
        sp = load_json(sp_path)
        profile = {
            "ready": True,
            "n_rows": sp.get("n_rows"),
            "n_features": sp.get("n_features"),
            "label_rate": (sp.get("sample_summary") or {}).get("label_rate"),
            "leakage": ((sp.get("leakage") or {}).get("severity_counts")
                        or {}),
            "compression_audit": sp.get("compression_audit"),
            "leak_classification": sp.get("leak_classification"),
        }
    # ---- 2b) v13 reparse backlog:信息合法但未被消费的冻结源 ----
    # 与 compression_audit 候选并列,planner 的一等输入——探索承诺
    # (每条 plan)必须兑现或显式证伪(feature_exploration_complete
    # 门槛第 2 条),不允许静默遗漏。
    lc = profile.get("leak_classification") or {}
    confirmed_reparse = lc.get("confirmed", {}).get("reparse") \
        or spec.reparse_columns
    reparse_backlog = []
    for col in confirmed_reparse:
        reparse_backlog.append({
            "column": col,
            "plan": (lc.get("confirmed", {}).get("reparse_plans") or
                     spec.reparse_plans).get(col, ""),
        })
    profile["reparse_backlog"] = reparse_backlog

    # ---- 3) leaderboard top ----
    leaderboard_top: list[dict] = []
    metric_name = spec.primary_metric
    lb_path = run_dir / "leaderboard" / "leaderboard.json"
    if lb_path.exists():
        lb = load_json(lb_path)
        for row in (lb.get("rows") or [])[:top_k]:
            slim = {k: row.get(k) for k in _LB_COLUMNS
                    if k != "primary_metric"}
            slim[metric_name] = row.get(metric_name)
            leaderboard_top.append(slim)
    # ---- 4) archive top ----
    archive_top: list[dict] = []
    try:
        arch = CandidateArchive(run_dir, spec, config)
        archive_top = [{"trial_id": c.get("trial_id"),
                        "algorithm": c.get("algorithm"),
                        "search_action": c.get("search_action"),
                        "data_view": c.get("data_view"),
                        "changed_modules": c.get("changed_modules"),
                        "candidate_score": c.get("candidate_score")}
                       for c in arch.top(top_k)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取 archive 失败: %s", exc)

    # ---- 5) budget ----
    budget: dict = {"ready": False}
    st_path = run_dir / "run-state.json"
    ev_path = run_dir / "evolution-state.json"
    rb = spec.resource_budget or {}
    if st_path.exists():
        st = load_json(st_path)
        budget = {"ready": True,
                  "phase": st.get("phase"),
                  "n_trials_started": None,
                  "n_generations": None,
                  "max_trials": rb.get("max_trials"),
                  "max_generations": rb.get("max_generations"),
                  "stopped": None}
        if ev_path.exists():
            ev = load_json(ev_path)
            budget["n_generations"] = ev.get("n_generations")
            budget["stopped"] = ev.get("stopped")
        try:
            budget["n_trials_started"] = TrialFactory(run_dir).count_started()
        except Exception:  # noqa: BLE001
            pass
        # v16/P0-2:compute 口径(全部已启动 trial 的 runtime 之和)
        # 与 wall 双轨呈现——plan 阶段能看到真实搜索消耗 vs 墙钟
        try:
            _sec = 0.0
            for t in TrialFactory(run_dir).all_trials():
                rt = (t.data.get("result") or {}).get("runtime_sec")
                if isinstance(rt, (int, float)):
                    _sec += float(rt)
            budget["compute_hours"] = round(_sec / 3600.0, 3)
        except Exception:  # noqa: BLE001
            pass
        # v12(P1-9):动作配比建议接线(旧 suggest_action_mix 死代码)
        try:
            from .evolution import EvolutionEngine

            engine = EvolutionEngine(run_dir, spec, config)
            deltas = [d for d in (ev.get("generation_best_deltas") or [])
                      if d is not None] if ev_path.exists() else []
            noise = None
            for t in TrialFactory(run_dir).all_trials():
                n = (t.data.get("evaluation") or {}).get("noise")
                if n is not None:
                    noise = n
                    break
            budget["suggested_action_mix"] = engine.suggest_action_mix(
                last_deltas=deltas, noise=noise)
        except Exception:  # noqa: BLE001
            pass

    # ---- 6) data_views_tried ----
    data_views_tried: list[str] = []
    try:
        data_views_tried = sorted({t.data.get("data_view") or "v1"
                                   for t in TrialFactory(run_dir).all_trials()}
                                  ) if TrialFactory(run_dir).all_trials() else []
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取 trials 失败: %s", exc)

    # ---- 6b) 抽样面功效声明(G4,v12.1)----
    # plan 时必须知道"抽样面只能测 ≥X 的效应",防止小增量被抽样噪声
    # 吞掉后被误判"方向无效"(v3-4 实测:Δ+0.0005 的真增量 < 面的
    # 可检下限 0.006,阴性不可采信)。splits-sample/manifest.json 缺失
    # 或未生成声明时静默跳过。
    sample_power_statement: dict | None = None
    # v2.17.0(#19):抽样面噪声预检(WP5b 写入 manifest.noise_precheck:
    # recommend_v1 / se_ratio / reason),存在即呈现
    sample_noise_precheck: dict | None = None
    try:
        _ps_path = run_dir / "sample-profile" / "splits-sample" \
            / "manifest.json"
        if _ps_path.exists():
            import json as _json
            _manifest = _json.loads(
                _ps_path.read_text(encoding="utf-8"))
            sample_power_statement = _manifest.get("power_statement")
            _np = _manifest.get("noise_precheck")
            if isinstance(_np, dict) and _np:
                sample_noise_precheck = _np
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取抽样面功效声明失败: %s", exc)

    # ---- 0) v2.17.0 统计尺子:MDE + 晋级门预览 ----
    # MDE 走 runtime.mde.mde_info(量纲感知;scale_warning 显著呈现)。
    # 晋级门预览由 WP1 的 evolution.promotion_threshold_preview 提供,
    # 接口缺失/报错时降级为 ready=False(不阻塞 planner-context)。
    mde: dict = {"ready": False}
    try:
        from .mde import mde_info

        mde = dict(mde_info(config, run_dir=run_dir), ready=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("MDE 解析失败(不阻塞 planner-context): %s", exc)
    promotion_preview: dict = {"ready": False}
    try:
        from . import evolution as _evo

        _fn = getattr(_evo, "promotion_threshold_preview", None)
        if callable(_fn):
            _pv = _fn(run_dir, spec, config)
            if isinstance(_pv, dict):
                promotion_preview = dict(_pv, ready=True)
        else:
            promotion_preview = {"ready": False,
                                 "reason": "promotion_threshold_preview 不可用"}
    except Exception as exc:  # noqa: BLE001
        promotion_preview = {"ready": False, "reason": f"预览失败: {exc}"}
        logger.warning("晋级门预览失败(不阻塞 planner-context): %s", exc)
    # 本任务目标口径(#18:经验/warm baseline 可比性判据)
    target_scale: dict = {}
    try:
        from .memory import infer_target_transform

        target_scale = {
            "primary_metric": spec.data.get("primary_metric"),
            "target_name": (spec.data.get("target") or {}).get("name"),
            "target_transform": infer_target_transform(spec.data)}
    except Exception:  # noqa: BLE001
        target_scale = {}


    # ---- 7) 经验(v2.18.0,Q17):卷宗 + 同任务方向表 + 避免清单 + 热启动榜单 ----
    # 来自本 run 固定水位的经验库视图。不再每代推送 8–10 张完整卡片:具体原则由
    # Agent 起草假设后按主题调用 `experience search` 取回,建 trial 时 --basis 引用。
    experience: dict = {}
    try:
        from . import library_view
        from .library import retrieval as _rt

        _atid = None
        try:
            _atid = spec.resolve_abstract_task_id(run_dir)
            if not _atid or "pending" in _atid:
                _atid = None
        except Exception:  # noqa: BLE001
            _atid = None
        _view = library_view.ensure(run_dir, atid=_atid, spec=spec,
                                    config=config)
        _canon = _rt.canon_fn(_view.data)
        _avoid = []
        for _p in _view.principles:
            if _p.get("direction") != -1 or _p.get("category") == "engineering":
                continue
            if _rt.base_relation(_view.atid, _p, _canon) != "same_task":
                continue
            if not _rt.scope_matches(_view.data.get("scope"), _p.get("scope")):
                continue
            _st = _view.stats.get(_p["principle_id"]) or {}
            _avoid.append((-(_st.get("posterior") or 0.5), _p, _st))
        _avoid.sort(key=lambda x: x[0])
        experience = {
            "watermark": _view.watermark, "stale": _view.stale,
            "reason": _view.data.get("reason"), "atid": _view.atid,
            "n_principles": len(_view.principles),
            "dossiers": {k: (v or {}).get("sections") for k, v in _view.dossiers.items()},
            "direction_table": _view.direction_table(top=5),
            "avoid": [{"principle_id": p["principle_id"], "claim": str(p["claim"])[:160],
                       "topics": p["topics"],
                       # 无证据关联时标 unverified:避免类原则里有相当比例属于评估 /
                       # 工程类(按设计不建被动证据),把"没有证据"显示成 low 会误导
                       "confidence": (st.get("confidence") or "low") if st
                       else str(p.get("confidence") or "unverified"),
                       "confidence_source": "evidence" if st else "author",
                       "contested_with": p.get("contested_with") or []}
                      for _, p, st in _avoid[:6]],
            "warm_baselines": [{"rank": r.get("rank"), "recipe_id": r["recipe_id"],
                                "source_trial_id": r.get("source_trial_id"),
                                "algorithm": r.get("algorithm"),
                                "value": (r.get("measurement") or {}).get("value"),
                                "gain": (r.get("measurement") or {}).get("gain"),
                                "measurement_kind": (r.get("measurement") or {}).get("kind"),
                                "confirmed": (r.get("measurement") or {}).get("confirmed")}
                               for r in _view.leaderboard[:5]],
        }
        try:
            import json as _json
            from .logging_utils import now_iso as _now
            with open(run_dir / "experience-retrieval-telemetry.jsonl", "a",
                      encoding="utf-8") as _tf:
                _tf.write(_json.dumps({
                    "at": _now(), "kind": "planner-context",
                    "watermark": _view.watermark, "stale": _view.stale,
                    "shown_ids": [x["principle_id"] for x in experience["avoid"]],
                    "dossiers": [k for k, v in experience["dossiers"].items() if v],
                }, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("经验库视图读取失败(不阻塞 planner-context): %s", exc)

    # ---- 8) last reflection ----
    last_reflection: dict = {"ready": False}
    gens_dir = run_dir / "generations"
    if gens_dir.is_dir():
        gids = sorted(p.name for p in gens_dir.iterdir() if p.is_dir()
                      and p.name.startswith("gen-"))
        for name in reversed(gids):
            rpath = gens_dir / name / "reflection.md"
            if rpath.exists():
                text = rpath.read_text(encoding="utf-8", errors="replace")
                last_reflection = {"ready": True,
                                   "generation": name,
                                   # 截断:只给结论段提示,全文由 subagent 按需读
                                   "excerpt": text[:1200]}
                break

    # ---- 9) 方向台账 + 天花板诊断(v11)----
    direction_ledger: dict = {"ready": False}
    try:
        from .directions import DirectionLedger

        ledger = DirectionLedger(run_dir)
        if ledger.path.exists():
            ledger.state = load_json(ledger.path)
            direction_ledger = {
                "ready": True,
                "summary_md": ledger.render_summary(),
                "ucb_next_slots": ledger.ucb_allocation(
                    int(config.get("evolution.parallel_trials", 4)
                        if str(config.get("evolution.parallel_trials")) !=
                        "auto" else 4)),
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("方向台账读取失败(不阻塞 planner-context): %s",
                       exc)
    capacity: dict = {"ready": False}
    cap_path = run_dir / "sample-profile" / "capacity-report.json"
    if cap_path.exists():
        cap = load_json(cap_path)
        capacity = {"ready": True,
                    "action_ranking": cap.get("verdict", {}).get(
                        "action_ranking"),
                    "conflict_rate": (cap.get("irreducible_noise")
                                      or {}).get("conflict_rate"),
                    "local_ambiguity_index": (cap.get("irreducible_noise")
                                               or {}).get(
                        "local_ambiguity_index",
                        (cap.get("irreducible_noise") or {}).get(
                            "noise_bound")),
                    "noise_bound": (cap.get("irreducible_noise")
                                    or {}).get("noise_bound")}
        # v2.17.2(C2):目标可达性 —— 用最新 trial 实测重算(capacity-report
        # 可能是早期缓存),失败回落报告落盘值
        try:
            from .capacity import assess_target_reachability

            capacity["target_reachability"] = assess_target_reachability(
                run_dir, cap, spec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("目标可达性重算失败(回落落盘值): %s", exc)
            capacity["target_reachability"] = cap.get("target_reachability")
    # v2.16.0:特征期门状态(与 Planner/new-trial 同一判定)。capacity
    # 诊断只作参考,model/loss/hpo 何时放行由这里决定。
    feature_phase: dict = {"ready": False}
    try:
        from .feature_phase import feature_phase_status

        feature_phase = dict(feature_phase_status(
            TrialFactory(run_dir).all_trials(), config), ready=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("特征期门状态读取失败(不阻塞 planner-context): %s",
                       exc)

    # ---- 11) error analysis(v12,P1-2:接入 planner-context)----
    # 该模块从 v11 起产出 error-analysis.json 但从未被消费(planner
    # 只见"什么方向成功了",不见"缺口在哪")。取当代 best trial 的
    # worst_bin/high_loss_signals,回答"该造什么特征"。
    error_analysis: dict = {"ready": False}
    try:
        lb_rows = (load_json(run_dir / "leaderboard" / "leaderboard.json")
                   .get("rows") or []) \
            if (run_dir / "leaderboard" / "leaderboard.json").exists() else []
        best_tid = None
        for row in lb_rows:
            if (row.get("verdict") or "") not in ("rejected", "invalid",
                                                  "unconfirmed"):
                best_tid = row.get("trial_id")
                break
        if best_tid:
            ea_path = run_dir / "trials" / best_tid / "error-analysis.json"
            if ea_path.exists():
                ea = load_json(ea_path)
                if not ea.get("skipped"):
                    # v2.17.0(D1):旧产物里的画像可能含目标/时间/ID 列,
                    # 读取侧同口径过滤(新产物在 runner 写入前已排除)
                    from .error_analysis import non_feature_columns

                    _ea_excl = non_feature_columns(spec)
                    error_analysis = {
                        "ready": True,
                        "trial_id": best_tid,
                        "worst_bin": (ea.get("loss_bins") and {
                            "bin": (max(ea["loss_bins"],
                                        key=lambda r: r.get("loss_over_n", 0))
                                    .get("bin")),
                        }) or None,
                        "high_loss_signals": [
                            r.get("feature")
                            for r in [
                                r for r in (ea.get("high_loss_profile")
                                            or [])
                                if str(r.get("feature")) not in _ea_excl
                                and not str(r.get("feature")).startswith(
                                    "__mvo_")][:5]],
                        "worst_uplift_bin": (ea.get("uplift_bins") and {
                            "bin": (max(ea["uplift_bins"],
                                        key=lambda r: abs(
                                            r.get("deviation", 0)))
                                .get("bin")),
                        }) or None,
                    }
    except Exception as exc:  # noqa: BLE001
        logger.warning("error-analysis 读取失败(不阻塞): %s", exc)

    # ---- 12) 信息源清单(v18/P1-1)----
    # "信息已榨干"的反向对账面:reparse 源与强 IV 族的消费状态 +
    # 粒度空间未探项。未消费项是 plan 的一等输入(先消费再谈收敛),
    # 反思必答"本代消费/证伪了哪个未消费项"。
    information_inventory: dict = {"ready": False}
    try:
        inv_path = run_dir / "sample-profile" / "information-inventory.yaml"
        if inv_path.exists():
            from .inventory import render_inventory_section

            inv = load_yaml(inv_path) or {}
            information_inventory = {
                "ready": True,
                "summary": inv.get("summary"),
                "rendered": render_inventory_section(inv),
            }
    except Exception as exc:  # noqa: BLE001
        logger.warning("information-inventory 读取失败(不阻塞): %s", exc)

    # ---- 13) v2.19.0:画像规则命中 + 天花板与剩余差距 ----
    profile_rules: dict = {"ready": False}
    try:
        from .profile_rules import evaluate as _pr_eval
        from .profile_rules import summary_lines as _pr_lines

        _pr = _pr_eval(run_dir, spec)
        profile_rules = {"ready": bool((_pr.get("facts") or {}).get(
            "available") or _pr.get("hits")),
            "n_hits": len(_pr.get("hits") or []),
            "rules": [h.get("rule") for h in _pr.get("hits") or []],
            "lines": _pr_lines(_pr)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("画像规则评估失败(不阻塞): %s", exc)
    ceiling: dict = {"ready": False}
    try:
        from .ceiling import ceiling_summary

        _champ_val = None
        if leaderboard_top:
            _champ_val = leaderboard_top[0].get(metric_name)
        _mde_val = (mde or {}).get("value") if isinstance(mde, dict) \
            else None
        _cs = ceiling_summary(run_dir, _champ_val, mde=_mde_val)
        ceiling = dict(_cs, ready=True) if _cs else {"ready": False}
    except Exception as exc:  # noqa: BLE001
        logger.warning("天花板摘要读取失败(不阻塞): %s", exc)

    return {
        "task_spec": task_spec,
        "mde": mde,
        "profile_rules": profile_rules,
        "ceiling": ceiling,
        "promotion_preview": promotion_preview,
        "target_scale": target_scale,
        "profile": profile,
        "leaderboard_top": leaderboard_top,
        "archive_top": archive_top,
        "budget": budget,
        "data_views_tried": data_views_tried,
        "sample_power_statement": sample_power_statement,
        "sample_noise_precheck": sample_noise_precheck,
        "experience": experience,
        "last_reflection": last_reflection,
        "direction_ledger": direction_ledger,
        "capacity": capacity,
        "feature_phase": feature_phase,
        "error_analysis": error_analysis,
        "information_inventory": information_inventory,
    }


def _fmt_num(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.6g}"
    return str(v) if v is not None else "—"


def _render_promotion_preview(pv: dict) -> list[str]:
    """晋级门预览渲染(WP1 evolution.promotion_threshold_preview;结构宽容)。

    主结构:q_next / schedule / reference / anchor_id 标量 + ``thresholds``
    列表({family, p_one_sided, z, delta_hint, below_bootstrap_resolution});
    兼容 ``families`` / ``z_by_family`` 字典写法;``delta_hint`` 可为标量
    或 {family: Δ} 字典。
    """
    L: list[str] = []
    head = []
    for k, label in (("next_generation", "下一代"), ("q_next", "本代 q"),
                     ("schedule", "排程"), ("reference", "对照"),
                     ("anchor_id", "锚点"), ("delta_hint", "Δ 估计")):
        v = pv.get(k)
        if v is not None and not isinstance(v, (dict, list)):
            head.append(f"{label}={_fmt_num(v)}")
    if head:
        L.append("- " + ", ".join(head))
    dh_map = pv.get("delta_hint") if isinstance(pv.get("delta_hint"),
                                                dict) else {}
    fams = pv.get("thresholds")
    if fams is None:
        fams = pv.get("families")
    if fams is None:
        fams = pv.get("z_by_family") or pv.get("z_thresholds") \
            or pv.get("family_thresholds")
    rows = []
    if isinstance(fams, dict):
        rows = [(k, v) for k, v in fams.items()]
    elif isinstance(fams, list):
        for i, v in enumerate(fams):
            if isinstance(v, dict):
                k = v.get("family") or v.get("family_size") or v.get("k") \
                    or v.get("m") or i + 1
            else:
                k = i + 1
            rows.append((k, v))
    for k, v in rows[:6]:
        if isinstance(v, dict):
            z = next((v.get(x) for x in ("z", "z_threshold", "z_required",
                                         "z_min") if v.get(x) is not None),
                     None)
            d = next((v.get(x) for x in ("delta_hint", "delta", "delta_min",
                                         "delta_required")
                      if v.get(x) is not None), None)
            p1 = v.get("p_one_sided")
            low_res = v.get("below_bootstrap_resolution")
        else:
            z, d, p1, low_res = v, None, None, None
        if d is None:
            d = dh_map.get(str(k)) if dh_map else None
        L.append(f"  - 同代 family={k}: "
                 + (f"单侧 p ≤ {_fmt_num(p1)}, " if p1 is not None else "")
                 + f"需 z ≥ {_fmt_num(z)}"
                 + (f",Δ 估计 ≈ {_fmt_num(d)}" if d is not None else "")
                 + ("(低于 bootstrap 分辨率)" if low_res else ""))
    if pv.get("se_reference") is not None:
        L.append(f"- Δ 估计口径:z × 近期同类 trial 配对 SE 中位数 "
                 f"{_fmt_num(pv.get('se_reference'))}"
                 f"(n={pv.get('se_n')}, {pv.get('se_source') or '—'})")
    by_act = pv.get("delta_hint_by_action")
    if isinstance(by_act, dict) and by_act:
        L.append("- 按动作的 Δ 估计(family=1):" + ", ".join(
            f"{a}≈{_fmt_num(v)}" for a, v in list(by_act.items())[:6]))
    if pv.get("note"):
        L.append(f"- 说明:{str(pv.get('note'))[:160]}")
    if not L:
        extra = {k: v for k, v in pv.items()
                 if k != "ready" and isinstance(v, (int, float, str, bool))}
        if extra:
            L.append("- " + ", ".join(f"{k}={_fmt_num(v)}"
                                      for k, v in list(extra.items())[:8]))
    return L


def render_planner_context(ctx: dict) -> str:
    """markdown 摘要(目标 <4000 字符);缺失节渲染"(未就绪)"。"""
    L: list[str] = ["# Planner Context", ""]

    # v2.17.0:MDE 量纲告警放在最显著位置(原值回归 MDE=0.001 曾让所有 Δ
    # 都"有意义",穷尽判定/HPO 递减门/确认门/交付容差全部失效)
    mde = ctx.get("mde") or {}
    if mde.get("scale_warning"):
        L += [f"> ⚠ **MDE 量纲告警**:{mde.get('scale_warning')}", ""]

    ts = ctx.get("task_spec") or {}
    L += ["## 1. TaskSpec", "",
          f"- task_id: {ts.get('task_id')}",
          f"- model_type: {ts.get('model_type')}",
          f"- primary_metric: {ts.get('primary_metric')}",
          f"- objective: {str(ts.get('objective_description'))[:120]}"]
    tsc = ctx.get("target_scale") or {}
    if tsc:
        L.append(f"- 目标口径: target={tsc.get('target_name')}, "
                 f"target_transform={tsc.get('target_transform')}"
                 "(经验/warm baseline 只与同口径比较)")
    if mde.get("ready"):
        L.append(f"- **MDE = {_fmt_num(mde.get('value'))}**(来源 "
                 f"{mde.get('source')}:absolute={_fmt_num(mde.get('absolute'))}"
                 f", relative={_fmt_num(mde.get('relative'))}, baseline 主指标="
                 f"{_fmt_num(mde.get('baseline_metric'))})")
    pv = ctx.get("promotion_preview") or {}
    if pv.get("ready") or mde.get("ready"):
        L += ["", "### 1b. 晋级门预览(本代过校正所需 z 与 Δ 估计)", ""]
        if pv.get("ready"):
            L += _render_promotion_preview(pv) or ["- (预览为空)"]
        else:
            L.append(f"- (不可用:{pv.get('reason') or '接口未就绪'})")

    p = ctx.get("profile") or {}
    L += ["", "## 2. SampleProfile", ""]
    if p.get("ready"):
        L += [f"- rows={p.get('n_rows')} features={p.get('n_features')} "
              f"label_rate={p.get('label_rate')}",
              f"- leakage: {p.get('leakage')}"]
        ca = p.get("compression_audit") or {}
        if ca:
            L += [f"- 压缩检测: risk={ca.get('risk_level')}, "
                  f"data_view 候选 {len(ca.get('data_view_candidates') or [])} 条"]
            for c in (ca.get("data_view_candidates") or [])[:3]:
                L += [f"  - `{c.get('view_id_hint')}`: "
                      f"{str(c.get('suggested_action'))[:100]}"]
        # v13:reparse backlog(信息合法但未被消费的冻结源;探索承诺
        # 必须兑现或显式证伪,见 feature-exploration.md 门槛第 2 条)
        rb = p.get("reparse_backlog") or []
        if rb:
            L += [f"- reparse backlog: {len(rb)} 条(授权解析源,"
                  "兑现或证伪进超级 trial)"]
            for r in rb[:5]:
                L += [f"  - `{r.get('column')}`: {r.get('plan') or '(待补 plan)'}"]
    else:
        L.append("- (未就绪: 先执行 profile)")

    L += ["", "## 3. Leaderboard Top", ""]
    lb = ctx.get("leaderboard_top") or []
    if lb:
        L.append("| trial | algo | gen | action | view | metric | Δbaseline "
                 "| p_pos | verdict |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in lb:
            metric = next((v for k, v in r.items()
                           if k not in _LB_COLUMNS and v is not None), None)
            ppos = r.get("p_positive")
            ppos_s = f"{ppos:.3f}" if isinstance(ppos, (int, float)) else "—"
            L.append(
                f"| {r.get('trial_id')} | {r.get('algorithm')} "
                f"| {r.get('generation')} | {r.get('search_action')} "
                f"| {r.get('data_view')} | {metric} "
                f"| {r.get('delta_vs_baseline')} | {ppos_s} "
                f"| {r.get('verdict')} |")
    else:
        L.append("- (未就绪: 无 leaderboard)")

    L += ["", "## 4. Archive Top", ""]
    arch = ctx.get("archive_top") or []
    if arch:
        for c in arch:
            L.append(f"- {c.get('trial_id')} ({c.get('algorithm')}, "
                     f"{c.get('search_action')}, view={c.get('data_view')}, "
                     f"modules={c.get('changed_modules')}, "
                     f"score={c.get('candidate_score')})")
    else:
        L.append("- (未就绪: archive 为空)")

    b = ctx.get("budget") or {}
    L += ["", "## 5. Budget", ""]
    if b.get("ready"):
        L.append(f"- phase={b.get('phase')} "
                 f"trials={b.get('n_trials_started')}/{b.get('max_trials')} "
                 f"generations={b.get('n_generations')}/{b.get('max_generations')}"
                 + (f" stopped={b['stopped'].get('stop_reason')}"
                    if b.get("stopped") else ""))
        mix = b.get("suggested_action_mix")
        if mix:
            L.append(f"- 建议动作配比(v12 接线): "
                     + ", ".join(f"{k}={v:.2f}" for k, v in mix.items()))
    else:
        L.append("- (未就绪: 无 run-state)")

    L += ["", "## 6. Data Views Tried", "",
          f"- {', '.join(ctx.get('data_views_tried') or ['(无)'])}"]

    # G4 功效声明(v12.1):抽样面可检效应下限进 plan 视野
    ps = ctx.get("sample_power_statement")
    if ps:
        L += ["", "### 6b. 抽样面功效声明(G4)", "",
              f"- validation 行数: {ps.get('validation_rows')}",
              f"- 可靠可检最小 Δ: **{ps.get('min_reliably_detectable_delta')}**"
              f"(95%CI 全宽 ≈{ps.get('delta_ci_full_width_approx')})",
              f"- 判读: 联合增量低于该值时,抽样面阴性**不可采信为方向无效**,"
              f"必须全量 confirmation 复核"]
    npc = ctx.get("sample_noise_precheck")
    if npc:
        _rec = npc.get("recommend_v1")
        # v2.18.4:se_ratio 是**单指标 bootstrap SE 比**,而判据用
        # paired_estimate(basis 字段)。只显示 ratio 会让读者得出与
        # recommend_v1 相反的结论(实测 se_ratio=4.07 却 recommend_v1=false)。
        # 两个口径并列显示,并标出使用的是哪个。
        _basis = str(npc.get("basis") or "metric_bootstrap_se")
        _used = npc.get("se_sample_paired_est") if _basis == "paired_estimate" \
            else npc.get("se_sample")
        _mde = npc.get("mde")
        _mde2 = _fmt_num(_mde * 2) if isinstance(_mde, (int, float)) else "—"
        L += ["", "### 6c. 抽样面噪声预检", "",
              f"- recommend_v1: **{_rec}**, MDE={_fmt_num(_mde)}",
              f"- 判据用量: SE_sample={_fmt_num(_used)}(basis={_basis})"
              f" vs 2×MDE={_mde2}"]
        L.append(f"- 参考: 单指标 bootstrap SE 比 se_ratio="
                 f"{_fmt_num(npc.get('se_ratio'))}"
                 f"(抽样 {_fmt_num(npc.get('se_sample'))} / 全量 "
                 f"{_fmt_num(npc.get('se_full'))}"
                 + (";该比值**不是**判据,判据按配对口径估计)"
                    if _basis == "paired_estimate" else ")"))
        L.append(f"- 理由: {str(npc.get('reason') or '—')[:200]}")
        # recommend_v1=false 有两种成因(data.sample_noise_precheck):
        # 噪声在 2×MDE 内(抽样面可筛),或噪声过大但全量面太贵(仍用抽样
        # 面,但阴性不可信)。两者判读相反,必须分开写。
        _noisy = (isinstance(_used, (int, float))
                  and isinstance(_mde, (int, float)) and _used > 2 * _mde)
        if _rec:
            L.append("- 判读: 抽样面配对 SE 相对 MDE 过大,本任务建议直接在 "
                     "v1 全量面跑 trial(抽样面阴性无分辨力)")
        elif _noisy:
            L.append("- 判读: 抽样面对 MDE 量级**无**分辨力,只因全量面耗时"
                     "过长才仍用抽样面 —— 抽样面阴性结论不可信,方向须回 v1 "
                     "全量面复核")
        elif isinstance(_mde, (int, float)):
            L.append("- 判读: 抽样面对 MDE 量级仍有分辨力,**但**低于其最小"
                     "可检 Δ 的阴性结论仍不可采信(见 6b),晋级判据一律以 "
                     "v1 全量面为准")

    L += ["", "## 7. 经验(经验库视图)", ""]
    ex = ctx.get("experience") or {}
    if ex.get("watermark") is None:
        L += ["- 经验库不可用(服务不可达且无本机缓存),本 run 不使用经验;"
              "diagnose 会给出提示。", ""]
    else:
        L.append(f"库水位 {ex.get('watermark')}"
                 f"{'(本机缓存,stale)' if ex.get('stale') else ''} · 任务 "
                 f"`{ex.get('atid') or '?'}` · 库内原则 {ex.get('n_principles', 0)} 条")
        L.append("")
        _sec_names = (("ceiling", "天花板与冠军配方"), ("effective", "已证实有效"),
                      ("falsified", "已证伪 / 无增益"), ("open_questions", "未决问题"),
                      ("data_notes", "数据注意事项"))
        for _key, _title in (("task", "7.0 任务卷宗"), ("data_base", "7.1 数据底座卷宗")):
            _d = (ex.get("dossiers") or {}).get(_key)
            if not _d:
                continue
            L += [f"### {_title}", ""]
            for _k, _n in _sec_names:
                if _d.get(_k):
                    L.append(f"- **{_n}**:{_d[_k]}")
            L.append("")
        _dt = ex.get("direction_table") or {}
        if _dt.get("effective") or _dt.get("failed"):
            L += ["### 7.2 同任务方向表(历史 trial 聚合)", ""]
            for r in _dt.get("effective") or []:
                L.append(f"- ✓ {'+'.join(r['topics'])} {r.get('object') or ''} "
                         f"增益 {_fmt_num(r.get('gain'))}(`{r['trial_id']}`)")
            for r in _dt.get("failed") or []:
                L.append(f"- ✗ {'+'.join(r['topics'])} {r.get('object') or ''} "
                         f"读数 {_fmt_num(r.get('gain'))}"
                         f"{(' · ' + r['failure_reason'][:60]) if r.get('failure_reason') else ''}"
                         f"(`{r['trial_id']}`)")
            L.append("")
        if ex.get("avoid"):
            L += ["### 7.3 同任务避免清单", ""]
            for c in ex["avoid"]:
                _ct = f" ⚠ 与 {', '.join(c['contested_with'])} 相反" if c.get("contested_with") else ""
                L.append(f"- `{c['principle_id']}` [{c['confidence']}][{','.join(c['topics'])}] "
                         f"{c['claim']}{_ct}")
            L.append("")
        if ex.get("warm_baselines"):
            L += ["### 7.4 Warm Baselines(同任务榜单,gen-00 已热启动前 warm_rank_limit 名)", "",
                  "| rank | trial | algo | metric | gain | 测量 | confirmed |",
                  "|---|---|---|---|---|---|---|"]
            for r in ex["warm_baselines"]:
                L.append(f"| {r.get('rank')} | {r.get('source_trial_id') or '—'} "
                         f"| {r.get('algorithm') or '—'} | {_fmt_num(r.get('value'))} "
                         f"| {_fmt_num(r.get('gain'))} | {r.get('measurement_kind') or '—'} "
                         f"| {'✓' if r.get('confirmed') else '—'} |")
            L.append("")
        L += ["> 起草本代候选假设后,按主题检索相关原则与已失败方向:",
              "> `python <skill-root>/scripts/modelevo.py experience search --run <run> "
              "--topics <主题,…> [--hypothesis \"…\"]`;",
              "> 建 trial 必须写依据:`new-trial … --basis <原则id>:<follows|extends|tests|"
              "contradicts>` 或 `--basis novel`,并用 `--topics` 标主题。", ""]

    r = ctx.get("last_reflection") or {}
    L += ["", "## 8. Last Reflection", ""]
    if r.get("ready"):
        L.append(f"- {r.get('generation')} 摘要(截断):")
        L.append("```")
        L.append(str(r.get("excerpt"))[:1000])
        L.append("```")
    else:
        L.append("- (未就绪: 无 reflection)")

    d = ctx.get("direction_ledger") or {}
    L += ["", "## 9. 方向台账(v11)", ""]
    if d.get("ready"):
        L.append(d.get("summary_md") or "(空)")
        if d.get("ucb_next_slots"):
            L.append("")
            L.append(f"- UCB 建议下一代槽位: {d['ucb_next_slots']}")
    else:
        L.append("- (未就绪: 首代后生成)")

    fp = ctx.get("feature_phase") or {}
    cap = ctx.get("capacity") or {}
    L += ["", "## 10. 特征期门与天花板诊断", ""]
    if fp.get("ready"):
        _done = f"{fp.get('generations_done')}/{fp.get('min_generations')}"
        if fp.get("active"):
            L.append(f"- 特征期门: **特征期内**(已完成探索代数 {_done});"
                     "model/loss/hpo 默认拒绝,特征侧收敛时 Planner 每代释放"
                     " 1 个名额,人工放行用 --override-feature-phase \"理由\"")
        else:
            L.append(f"- 特征期门: 特征期已满(已完成探索代数 {_done});"
                     "model/loss/hpo 不拦")
    if cap.get("ready"):
        L.append("- 天花板诊断(capacity-check,抽样面参考,不参与放行):")
        for a in (cap.get("action_ranking") or []):
            L.append(f"  - [{a.get('priority')}] {a.get('action')}: "
                     f"{a.get('why')}")
        _tr = cap.get("target_reachability") or {}
        if _tr.get("verdict"):
            # v2.17.2(C2):业务目标可达性(诊断,不参与放行)
            _mark = {"reachable": "可达", "unlikely": "**不太可能**",
                     "not_reachable": "**不可达**"}.get(_tr["verdict"],
                                                      _tr["verdict"])
            L.append(f"- 目标可达性: {_mark}(`{_tr['verdict']}`)—— "
                     f"{_tr.get('rationale')}")
            if _tr["verdict"] != "reachable":
                L.append("  - 判读:缺口超出行外推 headroom 时,继续同口径"
                         "特征/HPO 只能逼近天花板;优先新信息源/样本定义,"
                         "或与用户确认目标")
        elif _tr.get("skipped"):
            L.append(f"- 目标可达性: 未判定({_tr.get('rationale')})")
    else:
        L.append("- 天花板诊断: 未运行(可选诊断,不影响放行)")

    ea = ctx.get("error_analysis") or {}
    L += ["", "## 11. 误差归因(当代 best 的缺口)", ""]
    if ea.get("ready"):
        L.append(f"- 基准 trial: {ea.get('trial_id')}")
        wb = ea.get("worst_bin") or {}
        if wb:
            L.append(f"- 最弱分箱: bin {wb.get('bin')}"
                     "(loss 超额最高的人群——特征缺口所在)")
        signals = ea.get("high_loss_signals") or []
        if signals:
            L.append(f"- 高损失信号特征: {', '.join(signals)}"
                     "(高损失人群在这些维度异常;画像只覆盖已入模特征,"
                     "真正的缺口在未入模的新信息方向)")
        ub = ea.get("worst_uplift_bin") or {}
        if ub:
            L.append(f"- uplift 偏差最大分箱: bin {ub.get('bin')}")
        L.append("- 判读:优先参考已知分组维度上的对照差(高损失 vs 低损失)"
                 "与残差可预测性——可预测=表达能力问题(加变换/换模型),"
                 "不可预测=只能靠新信息(加列/data_view)")
    else:
        L.append("- (未就绪: 首个 trial 评估后生成)")

    inv = ctx.get("information_inventory") or {}
    L += ["", "## 12. 信息源清单(v18:榨干前的反向对账)", ""]
    if inv.get("ready"):
        L.append(inv.get("rendered") or "")
        L.append("- 判读:unconsumed>0 时优先消费(先清账再谈收敛);"
                 "反思必答'本代消费/证伪了哪个未消费项';"
                 "information_channels_exhausted 停止出口要求 unconsumed=0")
    else:
        L.append("- (未就绪: profile 产出 information-inventory.yaml)")

    # v2.19.0:画像规则表命中(Planner 据此生成 profile_rule 候选)
    prs = ctx.get("profile_rules") or {}
    L += ["", "## 13. 画像规则命中(v2.19.0,profile_rule 候选来源)", ""]
    if prs.get("ready"):
        L += list(prs.get("lines") or [])
        L.append("- 判读:命中给出 family 与先验(×MDE,待 benchmark 校准);"
                 "同 family 有实测后由后验替代。large_sample/class_imbalance "
                 "的提示是口径/效率建议,不一定物化为候选")
    else:
        L.append("- (未就绪: profile 产物缺失)")

    # v2.19.0(#11):天花板与剩余差距
    cl = ctx.get("ceiling") or {}
    L += ["", "## 14. 天花板与剩余差距(v2.19.0)", ""]
    if cl.get("ready"):
        surf = str(cl.get("feature_surface") or "base")
        L.append(f"- 参照天花板: {_fmt_num(cl.get('ceiling_estimate'))}"
                 f"(best_reference={cl.get('best_reference')},"
                 f"feature_surface=`{surf}`,computed_at={cl.get('computed_at')})")
        L.append(f"- 当前冠军: {_fmt_num(cl.get('champion_value'))};"
                 f"gap_to_ref = {_fmt_num(cl.get('gap_to_ref'))}"
                 + (f"(= {cl['gap_in_mde']:.2f}×MDE"
                    + (",已在 MDE 内" if cl.get("within_mde") else "") + ")"
                    if isinstance(cl.get("gap_in_mde"), (int, float)) else ""))
        if surf == "base":
            L.append("- 判读:`base` 面参照只代表**未加宽特征面上的模型侧上限**;"
                     "冠军加宽后超过它是常态,不代表到顶。base 面 gap 只用于判断"
                     "模型轴(HPO/模型族)是否还有空间,不用于压低特征轴候选")
        else:
            L.append(f"- 判读:`{surf}` 是与冠军同特征面的参照,gap 即同面剩余"
                     "差距;gap ≤ MDE 时继续同面搜索的期望收益很低")
    else:
        L.append("- (无天花板估计)`modelevo.py ceiling --run <run>` 是用户按需"
                 "调用的功能,不属于日常迭代;不要把它列入提案或待办")

    return "\n".join(L) + "\n"
