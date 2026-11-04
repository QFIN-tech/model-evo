"""只读与诊断命令:status / diagnose / wait / delete-trial / snapshot / planner-context / capacity-check / preflight / acknowledge / check-products / export-experience

v2.11.9 从单文件 `scripts/modelevo.py` 拆出(拆分动机见
`_mvo/__init__.py`)。**本文件是机械搬运,未改语义** —— 便于对照原
实现审阅。新增命令请按职责放进对应模块,不要回填到 `modelevo.py`
(那里只留 argparse 与分发)。
"""
from __future__ import annotations

from runtime.config import Config  # noqa: E402
from runtime.memory import (EvidenceStore,  # noqa: E402
                            TaskExperience)
from pathlib import Path
from runtime.scheduler import Scheduler, resource_snapshot  # noqa: E402
from runtime.trial import TrialFactory  # noqa: E402
from runtime.logging_utils import (dump_json, dump_yaml, get_logger,  # noqa: E402
                                   load_json, load_yaml, locked_state_write,
                                   now_iso)
import json
from runtime.harness import protocol_stamp  # noqa: E402
import time
from .context import _SKILL_ROOT, _load_config, _load_spec, _run_dir_from, _trial_reached_terminal, _trial_summary_line, _update_run_state, logger


def cmd_status(args) -> int:
    run_dir = _run_dir_from(args)
    st = load_json(run_dir / "run-state.json") if (
        run_dir / "run-state.json").exists() else {}
    ev_state = load_json(run_dir / "evolution-state.json") if (
        run_dir / "evolution-state.json").exists() else {}
    factory = TrialFactory(run_dir)
    trials = factory.all_trials()
    gate = load_json(run_dir / "final-test-gate.json") if (
        run_dir / "final-test-gate.json").exists() else {"count": 0}
    # v16/P0-2:compute 口径(全部已启动 trial 的 runtime 之和)——
    # 与 wall 双轨呈现,预算决策不再 pause-blind
    # v2.17.2(B5):单源 compute_hours_summary,计入收尾用时并分列
    from runtime.trial import compute_hours_summary

    _acct = compute_hours_summary(run_dir, trials)
    # v2.17.2(B7):停止报告只在"其后没有新一代 trial"时有效;已被后续
    # 代取代(演进继续)的停止只作历史呈现,不再显示为当前停止
    _stop = ev_state.get("stopped") or {}
    _stop_gen = _stop.get("generation")
    if _stop and _stop_gen is None and _stop.get("n_generations"):
        _stop_gen = int(_stop["n_generations"]) - 1
    _superseded = bool(_stop) and _stop_gen is not None and any(
        t.generation is not None and int(t.generation) > int(_stop_gen)
        for t in trials)
    out = {
        "run_dir": str(run_dir),
        "phase": st.get("phase") or st.get("status"),
        "current_trial": st.get("current_trial"),
        "current_trial_status": st.get("current_trial_status"),
        "harness": {k: st.get(k) for k in
                    ("harness_protocol_version", "harness_hash")},
        "n_trials": len(trials),
        "n_trials_started": factory.count_started(),
        "n_generations": ev_state.get("n_generations", 0),
        "stopped": None if _superseded else _stop.get("stop_reason"),
        "final_test_touches": gate.get("count", 0),
        "compute_hours": round(_acct["compute_hours"], 3),
        "trial_compute_hours": round(_acct["trial_compute_hours"], 3),
        "finalize_hours": round(_acct["finalize_hours"], 3),
    }
    if _stop:
        out["stopped_generation"] = _stop_gen
    if _superseded:
        out["stop_superseded"] = {
            "stop_reason": _stop.get("stop_reason"),
            "generation": _stop_gen,
            "note": "停止报告之后已有新一代 trial(演进继续);下次 "
                    "finalize-generation 归档进 stop_history"}
    _hist = ev_state.get("stop_history") or []
    if _hist:
        out["stop_history"] = [
            {"generation": h.get("generation"),
             "stop_reason": h.get("stop_reason")} for h in _hist]
    # v2.14.0:搜索冠军 vs 交付位(收尾前的必看项;diverged/unconfirmed
    # 意味着默认收尾不会交付冠军配方,必须先向用户呈报)
    try:
        from runtime.confirmation import divergence_summary
        from runtime.leaderboard import Leaderboard

        spec = _load_spec(args)
        res = Leaderboard(run_dir, spec, _load_config(run_dir)) \
            .delivery_resolution(trials)
        out["delivery_resolution"] = {
            k: res.get(k) for k in ("status", "search_champion",
                                    "delivery_trial", "gap", "tolerance")}
        out["delivery_resolution"]["summary"] = divergence_summary(res)
        # v2.20.0:先验豁免的证据(判据版本 / z / σ_seed 来源)随 status 呈现,
        # 收尾前据此区分"已确认交付"与"按先验豁免交付"
        if res.get("prior_waiver"):
            out["delivery_resolution"]["prior_waiver"] = res["prior_waiver"]
    except Exception as exc:  # noqa: BLE001  status 是只读汇总,不因此失败
        out["delivery_resolution"] = {"error": str(exc)[:200]}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def cmd_plan(args) -> int:
    """生成下一代结构化 proposal batch（不执行、不触碰终测）。"""
    from runtime.planner import AutonomousPlanner

    run_dir = _run_dir_from(args)
    spec = _load_spec(args)
    config = _load_config(run_dir)
    batch = AutonomousPlanner(run_dir, spec, config).plan(
        slots=int(getattr(args, "slots", 4)))
    print(json.dumps(batch, ensure_ascii=False, indent=2))
    return 0


def cmd_benchmark(args) -> int:
    """汇总固定预算 benchmark 并执行框架发布质量门。

    v2.19.0:实现移到 _mvo/cmd_benchmark.py(新增 prepare / report 动作);
    这里保留同名入口做委托,旧导入路径与 ``benchmark --results`` 用法不变。"""
    from .cmd_benchmark import cmd_benchmark as _impl

    return _impl(args)


def _diag_product_state(trial_dir: Path) -> dict:
    """Trial 产物齐备性静态盘点(不执行任何代码)。"""
    return {
        "has_trial_yaml": (trial_dir / "trial.yaml").exists(),
        "has_code": any((trial_dir / "code").iterdir())
        if (trial_dir / "code").is_dir() else False,
        "has_validation_score": (trial_dir
                                 / "validation_score.parquet").exists(),
        "has_metrics_json": (trial_dir / "metrics.json").exists(),
        "has_evaluation_json": (trial_dir / "evaluation.json").exists(),
        "has_model_artifact": any(
            (trial_dir / "model").glob("model.*")) if (
            trial_dir / "model").is_dir() else False,
        "has_feature_importance": (trial_dir / "feature"
                                   / "feature_importance.csv").exists()
        or (trial_dir / "feature_importance.csv").exists(),
        "logs": {name: (trial_dir / name).exists()
                 for name in ("stdout.log", "stderr.log", "trial.log")},
    }


def _diag_hp_unconsumed(tdir: Path) -> str | None:
    """lint 规则 11 同口径(规则函数单源 assets/lint_trial.py):返回问题描述或 None。"""
    tp = Path(tdir) / "code" / "train.py"
    if not tp.exists():
        return None
    try:
        from .render import _lint_module

        errs = _lint_module().check_hyperparameters_consumption(
            Path(tdir).parent.parent, Path(tdir),
            tp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 诊断只读,失败不阻断其它检查
        return None
    return errs[0] if errs else None


def _diag_trial_anomalies(trial, spec, run_state: dict,
                          lb_row: dict | None) -> list[dict]:
    """对单个 trial 逐条体检,返回 [{code, severity, detail, remedy}]。

    覆盖实测 triage 场景(verdict 不被 leaderboard 收录/被强制 invalid/
    evaluation 不被识别/delta_vs_baseline 恒空/epoch 排除),每条 anomaly
    都给出确定性 remedy——主会话按 remedy 执行,不再回读 runtime 源码。
    """
    anomalies: list[dict] = []
    d = trial.data
    ev = d.get("evaluation") or {}
    tdir = Path(trial.trial_dir)
    verdict = ev.get("verdict")
    status = trial.status

    # 1. 状态机口径:FAILED 却带 evaluation(verdict 不再采信)
    if status == "FAILED" and verdict:
        anomalies.append({
            "code": "failed_with_evaluation",
            "severity": "high",
            "detail": f"status=FAILED 且 evaluation.verdict={verdict}"
                      "(FAILED 的 verdict 不进 leaderboard 有效行)",
            "remedy": "run --trial %s 修复重跑,或确认失败原因后由"
                      " finalize-generation 归档" % trial.trial_id,
        })
    # 2. code/ 为空(创建半途终止)
    prod = _diag_product_state(tdir)
    # 2b. v2.16.0 拓扑门:级联/融合未获人工批准
    _topo = d.get("model_topology") or {}
    if _topo.get("decision") == "blocked" and status in ("PROPOSED",
                                                         "CODE_WRITTEN"):
        anomalies.append({
            "code": "topology_approval_required",
            "severity": "high",
            "detail": "检测拓扑 %s(需要批准:%s);信号:%s" % (
                _topo.get("kind"), ", ".join(_topo.get("needs") or []),
                "; ".join("%s@L%s" % (s.get("rule"), s.get("line"))
                          for s in (_topo.get("signals") or [])[:4])),
            "remedy": "改写为单模型后重跑(源码未冻结);或升级到用户,由用户"
                      "在自己的终端执行:%s。Agent 不得自行批准"
                      % _topo.get("approve_command"),
        })
    if _topo.get("ledger_valid") is False:
        anomalies.append({
            "code": "topology_ledger_invalid",
            "severity": "critical",
            "detail": "approvals/model-topology.jsonl 哈希链断裂:%s"
                      % _topo.get("ledger_error"),
            "remedy": "升级到用户核查账本;链断裂期间全部批准作废",
        })
    if not prod["has_code"] and status in ("PROPOSED", "CODE_WRITTEN"):
        anomalies.append({
            "code": "empty_scaffold",
            "severity": "info",
            "detail": "code/ 为空且未启动:创建半途终止的脚手架"
                      "(finalize-generation 会自动标记 abandoned)",
            "remedy": "忽略,或删除 trials/%s(留痕即可)" % trial.trial_id,
        })
    # 2c. v2.17.0(D2):--set 超参无消费点(run 前 lint 规则 11 必拒)。
    # 旧版创建期只告警,trial 留在 PROPOSED,diagnose 不报 → run 时才炸。
    if prod["has_code"] and status in ("PROPOSED", "CODE_WRITTEN"):
        _hp_issue = _diag_hp_unconsumed(tdir)
        if _hp_issue:
            anomalies.append({
                "code": "hyperparameters_unconsumed",
                "severity": "high",
                "detail": _hp_issue,
                "remedy": "未启动 trial:delete-trial --trial %s 后用已读 "
                          "config.hyperparameters 的父代重建(baseline 血缘"
                          "父代 new-trial --from-parent --set 会自动注入消费段);"
                          "或在 trials/%s/code/train.py 估计器构造处读取 "
                          "cfg[\"hyperparameters\"] 后再 run"
                          % (trial.trial_id, trial.trial_id),
            })
    # 3. 跑完但缺产物(评估入口会拒)
    if status in ("SUCCEEDED", "RUNNING") and not prod[
            "has_validation_score"]:
        anomalies.append({
            "code": "missing_validation_score",
            "severity": "high",
            "detail": "状态 %s 但 validation_score.parquet 缺失"
                      "(evaluator 入口直接 FileNotFoundError)"
                      % status,
            "remedy": "读 stderr.log 定位训练尾部失败;修复后"
                      " run --trial %s 重跑" % trial.trial_id,
        })
    # 4. 泄漏哨兵强制 invalid
    if ev.get("leakage_suspects"):
        anomalies.append({
            "code": "leakage_suspect",
            "severity": "critical",
            "detail": "leak_scan 命中,verdict 已强制 invalid:"
                      "%s" % ev.get("leakage_suspects"),
            "remedy": "升级到人:改特征面重跑,或证明误报后重评;"
                      "反思必须覆盖归因(v13 泄漏核查必答)",
        })
    # 5. 手改痕迹:evaluation.json 与 trial.yaml 内联 evaluation 不一致
    if prod["has_evaluation_json"]:
        try:
            ev_file = load_json(tdir / "evaluation.json") or {}
            v_file = ev_file.get("verdict")
            if v_file != verdict:
                anomalies.append({
                    "code": "evaluation_verdict_mismatch",
                    "severity": "high",
                    "detail": "evaluation.json.verdict=%s 与 "
                              "trial.yaml 内联 verdict=%s 不一致"
                              "(存在手改/半程写入)"
                              % (v_file, verdict),
                    "remedy": "run --trial %s 重评(evaluate_only 只重跑"
                              "评估,不重训)使两处口径归一"
                              % trial.trial_id,
                })
        except Exception as exc:  # noqa: BLE001
            anomalies.append({
                "code": "evaluation_json_unreadable",
                "severity": "high",
                "detail": "evaluation.json 解析失败: %s" % exc,
                "remedy": "run --trial %s 重评(以重评产物覆盖损坏文件)"
                          % trial.trial_id,
            })
    # 6. 跨 epoch 排除(铁律 10)
    trial_epoch = d.get("harness_epoch")
    run_epoch = run_state.get("harness_epoch", 0)
    if trial_epoch is not None and int(trial_epoch) != int(run_epoch):
        anomalies.append({
            "code": "cross_epoch_excluded",
            "severity": "high",
            "detail": "trial harness_epoch=%s 与 run-state=%s 不同:"
                      "leaderboard 拒绝混排,该行已被排除(铁律 10)"
                      % (trial_epoch, run_epoch),
            "remedy": "acknowledge-harness-change 显式登记口径决策;"
                      "确属不可比则接受排除(不手改 run-state)",
        })
    # 7. delta_vs_baseline 锚定缺失
    if lb_row is not None and verdict in ("accepted", "promising",
                                          "confirmed") \
            and lb_row.get("delta_vs_baseline") is None:
        bid = run_state.get("baseline_id")
        anomalies.append({
            "code": "baseline_anchor_missing",
            "severity": "medium",
            "detail": "verdict=%s 但 delta_vs_baseline 为空"
                      "(baseline_id=%s 未落盘或 baseline 分数不可比)"
                      % (verdict, bid),
            "remedy": "python tools/backfill_baseline.py 补齐锚定;"
                      "跨数据视图导致的不可比属正常(vs_baseline 记空)",
        })
    # 8. 无父代可比导致的 rejected(根 trial 之外的意外形态)
    # v2.18.4:跨数据视图的 trial 本来就不可配对(evaluator 记
    # baseline_incomparable,reason="无父代可比分数"),这是协议内预期,
    # 不是缺陷。旧口径对 4 个 data_view=sample 的探索臂全部报 medium
    # 异常并让 Agent "重评",而重评只会再得到同一个不可比结论。
    if verdict == "rejected" and str(ev.get("reason") or "").find(
            "无父代可比") >= 0 and trial.parent_ids:
        if (ev.get("delta") or {}).get("baseline_incomparable") or \
                str(d.get("data_view") or "v1") != "v1":
            anomalies.append({
                "code": "cross_view_incomparable",
                "severity": "info",
                "detail": "跨数据视图 trial(data_view=%s)与父代不可配对,"
                          "verdict=rejected 属协议内预期,不是模型变差"
                          % (d.get("data_view") or "v1"),
                "remedy": "无需处置;跨视图方向判断以同视图对照为准"
                          "(contrast --trials)",
            })
        else:
            anomalies.append({
                "code": "parent_scores_unavailable",
                "severity": "medium",
                "detail": "有父代但评估时未取到父代分数(父代"
                          " validation_score.parquet 缺失或跨视图行数不齐)",
                "remedy": "run --trial %s 重评(evaluate_only 会重新加载"
                          " 父代分数);确认父代产物在档" % trial.trial_id,
            })
    # 9. v2.11 孤儿 RUNNING(stale-running):状态 RUNNING 但既无活
    # runner 锁、心跳也已停摆。实测(run drawdown-req-t7-6w)trial-007
    # 心跳 19:58 消失后 30 分钟无人发现,进程死了状态机还停在 RUNNING,
    # 直到下一次 run 重入才被锁检测兜住——盲区 = 白烧的等待 + 迟到的
    # 重启。诊断命令是主会话唯一合法的巡检口,把孤儿状态在这里显式报出。
    if status == "RUNNING":
        _lock = tdir / ".runner.lock"
        _live = None
        try:
            if _lock.exists():
                _pid = int((_lock.read_text(encoding="utf-8")
                            or "0").strip() or 0)
                if _pid:
                    import os as _os

                    try:
                        _os.kill(_pid, 0)
                        _live = _pid
                    except ProcessLookupError:
                        _live = None
                    except PermissionError:
                        _live = _pid
        except (OSError, ValueError):
            _live = None
        if _live is None:
            # 心跳 = trial.log 末条 heartbeat 的时刻(分钟级)
            _hb_iso = None
            try:
                _log = tdir / "trial.log"
                if _log.exists():
                    _lines = _log.read_text(encoding="utf-8",
                                            errors="ignore").splitlines()
                    for _ln in reversed(_lines[-60:]):
                        if "heartbeat" in _ln or "process exited" in _ln:
                            _hb_iso = _ln.strip("[]")[:19]
                            break
            except OSError:
                pass
            anomalies.append({
                "code": "stale_running",
                "severity": "high",
                "detail": "状态 RUNNING 但无活 runner 锁(孤儿进程)"
                          + (f";末条心跳 {_hb_iso}" if _hb_iso else ""),
                "remedy": "run --trial %s 重入:有分数走 evaluate_only,"
                          "无分数自动回退 CODE_WRITTEN 重训(v2.8.1 "
                          "自愈闭环;无需手工清理)" % trial.trial_id,
            })
    # 10. v2.15.2:warm-start 配方不兼容(专用失败原因)。
    # 背景:gen-00 从经验库渲染历史配方作为 warm trial(v2.18.0 前为 memory/top-trials)。
    # 配方里的 RUN_DIR 若无法被安全改写为本 run 路径(行形态不是
    # `pathlib.Path(...)`),修复后的实现**拒绝带错路径进训练**,直接落
    # FAILED + failure_reason=warm_start_recipe_incompatible。
    # 这条既不是模型退化,也不是环境故障(v2.15.0 曾把它混进 infra_error,
    # 让"配方格式问题"看起来像基础设施抖动)。单列出来供一眼区分。
    if str(trial.data.get("failure_reason") or "") \
            == "warm_start_recipe_incompatible":
        anomalies.append({
            "code": "warm_start_recipe_incompatible",
            "severity": "info",
            "detail": "warm trial 的配方 RUN_DIR 无法安全改写为本 run 路径"
                      "(配方行不是 `RUN_DIR = pathlib.Path(...)` 形态),"
                      "已在训练前主动放弃——不是模型/数据问题,"
                      "也不是环境故障",
            "remedy": "无需处置;该配方不再参与本 run 的热启动",
        })
    return anomalies


def cmd_wait(args) -> int:
    """确定性等待(v17.2)——实现见 `_mvo/cmd_wait.py`(v2.17.0 起轻量单源)。

    STALE_RUNNING(exit 4)/ WAIT_TIMEOUT(exit 3)/ ADMISSION_QUEUED 等契约
    不变;`modelevo.py wait` 以脚本运行时走快路径,不 import 本模块。wait
    只读、幂等:被杀(137)时原样重调即可,训练不受影响。
    """
    from .cmd_wait import cmd_wait as _cmd_wait

    return _cmd_wait(args)


def cmd_delete_trial(args) -> int:
    """删除未启动的 trial(带引用与产物检查,留痕日志)。

    [FIX-2026-09-03] 背景:gen-03 实测 Agent 用契约外 `rm -rf` 清理
    占位/废弃 trial,把 trial-008 的 parent trial-007 一起删掉,
    leaderboard lineage 留下幽灵链(trial-007 > trial-008,前者已
    不存在)。harness 化删除堵住三个口:
    - 仅限 status==PROPOSED(未启动、无预算占用,count_started 不计);
    - 无产物(smoke/feature/model 等子目录必须为空);
    - 无引用(任何现存 trial 的 parent_trial_ids 含它即拒绝);
    删除动作落 run 目录 delete-log.jsonl 留痕,复盘可见谁删了什么。
    """
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    config = _load_config(run_dir)
    factory = TrialFactory(run_dir, spec=spec, config=config)
    tdir = Path(run_dir) / "trials" / args.trial
    if not tdir.exists():
        logger.error("Trial 不存在: %s", tdir)
        return 1
    from runtime.trial import TrialRecord

    trial = TrialRecord.load(tdir)
    if trial.status != "PROPOSED":
        logger.error(
            "拒绝删除:%s 状态为 %s(仅 PROPOSED 可删;已启动 trial 是"
            "预算与 leaderboard 口径的一部分,删除破坏审计,请走反思"
            "记录 verdict=invalid)", trial.trial_id, trial.status)
        return 1
    # 产物检查:code/config/trial.yaml/hypothesis.md 是脚手架本身,
    # 其余子目录/文件视为产物(存在即拒绝)
    scaffold_ok = {"code", "config", "trial.yaml", "hypothesis.md"}
    products = [p.name for p in tdir.iterdir() if p.name not in scaffold_ok]
    if products and not getattr(args, "force", False):
        logger.error(
            "拒绝删除:%s 存在产物 %s(已有训练/评估产出,不可能是纯占位;"
            "确认误建可用 --force 豁免并留痕)", trial.trial_id, products)
        return 1
    # 引用检查:任何现存 trial 以它为 parent 即拒绝(防幽灵 lineage)
    referring = []
    for t in factory.all_trials():
        if t.trial_id == trial.trial_id:
            continue
        if trial.trial_id in (t.parent_ids or []):
            referring.append(t.trial_id)
    if referring:
        logger.error(
            "拒绝删除:%s 被以下现存 trial 声明为父代: %s(删除会产生"
            "幽灵 lineage;先删/改子代引用)", trial.trial_id, referring)
        return 1
    # 留痕 + 删除
    import shutil as _shutil_d

    log_path = Path(run_dir) / "delete-log.jsonl"
    import json as _json_d

    with log_path.open("a", encoding="utf-8") as _f:
        _f.write(_json_d.dumps({
            "at": __import__("datetime").datetime.now().isoformat(),
            "trial_id": trial.trial_id,
            "reason": getattr(args, "reason", "") or "",
            "force_products": products if getattr(args, "force", False)
            else [],
        }, ensure_ascii=False) + "\n")
    _shutil_d.rmtree(tdir)
    logger.info("已删除 trial %s(留痕 %s;原因: %s)", trial.trial_id,
                log_path, getattr(args, "reason", "") or "(未填写)")
    print(f"DELETED\t{trial.trial_id}")
    return 0


def cmd_diagnose(args) -> int:
    """故障 triage 一键诊断(v17):回答"trial 为何不在 leaderboard /
    verdict 为何异常 / 产物为何不被识别",替代回读 runtime 源码刑侦。

    主会话铁律 13 的配套出口:遇到异常状态时唯一合法动作是跑本命令,
    输出 = 结构化 findings(每条含 code/severity/detail/remedy)+
    环境一致性检查结果。--trial 聚焦单 trial;缺省全量扫描。
    """
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    factory = TrialFactory(run_dir)
    st = load_json(run_dir / "run-state.json") if (
        run_dir / "run-state.json").exists() else {}

    # ---- 环境一致性(harness/泄漏分类/环境指纹)----
    env_findings: list[dict] = []
    stamp = protocol_stamp(_SKILL_ROOT / "runtime")
    if st.get("harness_hash") and \
            st["harness_hash"] != stamp["harness_hash"]:
        from runtime.harness import auto_acknowledgeable

        _plumb = auto_acknowledgeable(st.get("harness_files") or {},
                                      stamp["harness_files"])
        if _plumb is not None:
            # v2.17.2(B8):只动 plumbing 模块 → 下次 run/run-generation 自动
            # 登记,不影响分数可比性;不再报 critical
            env_findings.append({
                "code": "harness_hash_drift_plumbing_only",
                "severity": "info",
                "detail": "harness 变更仅涉及 plumbing 模块 %s(历史分数仍可比)"
                          % _plumb["changed"],
                "remedy": "无需手工处理:下一次 run / run-generation 自动登记"
                          "(或 acknowledge-harness-change --run %s --reason ...,"
                          "无需任何 flag)" % run_dir,
            })
        else:
            env_findings.append({
                "code": "harness_hash_drift",
                "severity": "critical",
                "detail": "run-state harness_hash=%s ≠ 当前 %s"
                          "(runtime 代码已变更,历史分数不可比)"
                          % (st["harness_hash"], stamp["harness_hash"]),
                "remedy": "acknowledge-harness-change --run %s --reason ..."
                          % run_dir,
            })
    if st.get("leak_manifest_hash") and \
            st["leak_manifest_hash"] != spec.leak_manifest_hash:
        env_findings.append({
            "code": "leak_manifest_drift",
            "severity": "critical",
            "detail": "spec 泄漏分类 hash=%s ≠ run-state 记录 %s"
                      "(数据面定义已变更,必须重新 baseline)"
                      % (spec.leak_manifest_hash,
                         st["leak_manifest_hash"]),
            "remedy": "重新 baseline;hard-ban→reparse 解封需用户显式确认",
        })
    if st.get("phase") in ("BASELINED", "EVOLVING") \
            and st.get("env_fingerprint"):
        from runtime.preflight import DRIFT_KINDS, check_env_drift

        drift = check_env_drift(st, _SKILL_ROOT)
        if drift:
            # v20:环境变化不是 critical —— 它由 env-ledger.jsonl 记账,
            # 不阻塞运行,故也不参与本命令的 exit 1 判定(那条判定留给
            # 真正需要升级到人的形态:harness 漂移/泄漏分类漂移/
            # final_test gate 违规)。severity=info 保持它仍然可见。
            env_findings.append({
                "code": "env_fingerprint_drift",
                "severity": "info",
                "detail": {
                    "kinds": drift["kinds"],
                    "kinds_meaning": {k: DRIFT_KINDS.get(k, k)
                                      for k in drift["kinds"]},
                    "fingerprint": {"recorded": drift.get("recorded"),
                                    "current": drift.get("current")},
                    "events": drift["events"],
                },
                "remedy": "无需处置(环境变化只记录不阻塞,已记入 "
                          "env-ledger.jsonl);trial.yaml 的 "
                          "env_fingerprint 字段标出每个 trial 的实测环境",
            })
        else:
            # v20:主动记一条账,否则"环境一直没变"与"从没检查过"在
            # 台账上不可区分。fingerprint 是确定性对比,不写文件。
            env_findings.append({
                "code": "env_fingerprint_ok",
                "severity": "info",
                "detail": "环境指纹与 run-state 记录一致(%s)"
                          % st.get("env_fingerprint"),
                "remedy": "无需处置",
            })
    gate = load_json(run_dir / "final-test-gate.json") if (
        run_dir / "final-test-gate.json").exists() else {"count": 0}
    # v2.17.0(#9):经验导出源 schema 诊断(export-experience 失败场景)
    try:
        env_findings.extend(_diag_experience_export(run_dir))
        env_findings.extend(_diag_library(run_dir))
    except Exception:  # noqa: BLE001 — 诊断本身不得失败
        pass
    if gate.get("count", 0) > 1:
        env_findings.append({
            "code": "final_test_gate_violation",
            "severity": "critical",
            "detail": "final-test-gate count=%s(铁律 3:≤1)"
                      % gate.get("count"),
            "remedy": "停止全部评估动作,升级到人审查反馈环污染",
        })

    # ---- Trial 级扫描 ----
    trials = factory.all_trials()
    want = getattr(args, "trial", None)
    if want:
        trials = [t for t in trials if t.trial_id == want]
        if not trials:
            print(json.dumps({"error": "trial 不存在: %s" % want,
                              "existing": sorted(d.name for d in
                                                 (run_dir / "trials")
                                                 .iterdir())
                              if (run_dir / "trials").exists() else []},
                             ensure_ascii=False))
            return 1
    # leaderboard 行verdict 对照(leaderboard.json 是 build 的落盘投影,
    # 不在这里重建——诊断只读,避免与在途状态互踩)
    lb_rows: dict[str, dict] = {}
    lb_path = run_dir / "leaderboard" / "leaderboard.json"
    if lb_path.exists():
        try:
            lb_rows = {r.get("trial_id"): r for r in
                       (load_json(lb_path) or {}).get("rows") or []}
        except Exception:  # noqa: BLE001
            lb_rows = {}
    trial_findings: dict[str, list[dict]] = {}
    for t in trials:
        found = _diag_trial_anomalies(t, spec, st, lb_rows.get(t.trial_id))
        if found:
            trial_findings[t.trial_id] = found

    out = {
        "run_dir": str(run_dir),
        "phase": st.get("phase") or st.get("status"),
        "baseline_id": st.get("baseline_id"),
        "harness_epoch": st.get("harness_epoch", 0),
        "env_findings": env_findings,
        "n_trials": len(trials),
        "trial_findings": trial_findings,
        "lb_rows_absent": sorted(
            t.trial_id for t in trials
            if t.trial_id not in lb_rows),
        "hint": "remedy 即处置命令;禁止为 triage 回读 runtime 源码"
                "(铁律 13);本命令只读,不改任何状态",
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    # exit code:任一 critical(env 或 trial 级)即 1(可被脚本判定);
    # trial 级 leakage_suspect 属于"必须升级到人"的 critical 形态,
    # 与 env 漂移同级——都应在 shell 层可检出
    all_findings = list(env_findings)
    for found in trial_findings.values():
        all_findings.extend(found)
    return 1 if any(f.get("severity") == "critical"
                    for f in all_findings) else 0


def _diag_experience_export(run_dir: Path) -> list[dict]:
    """v2.17.0(#9):经验导出 schema 诊断(只读,不写全局库)。

    旧形态:顶层写成 ``experiences:`` 包装时 export-experience 抛
    ``'str' object has no attribute 'get'``,diagnose 没有任何 finding。
    这里直接对导出源做与导出器同口径的逐条校验,并读上一次导出报告。
    """
    out: list[dict] = []
    src = Path(run_dir) / "experience" / "exported-global-experiences.yaml"
    remedy_cmd = ("python %s export-experience --run %s"
                  % (_SKILL_ROOT / "scripts" / "modelevo.py", run_dir))
    if src.exists():
        try:
            from runtime.memory import load_experience_export

            chk = load_experience_export(src)
        except Exception as exc:  # noqa: BLE001
            chk = {"errors": [{"index": None, "field": None,
                               "error": f"校验器异常: {exc}"}],
                   "structure": "unknown", "fix_hint": ""}
        if chk.get("errors"):
            out.append({
                "code": "experience_export_schema_error",
                "severity": "warning",
                "detail": {"source": str(src),
                           "structure": chk.get("structure"),
                           "errors": [
                               {k: e.get(k) for k in
                                ("index", "line", "field", "error", "fix")}
                               for e in (chk.get("errors") or [])[:10]]},
                "remedy": (chk.get("fix_hint") or "") + ";然后执行 "
                          + remedy_cmd,
            })
    return out


def cmd_export_experience(args) -> int:
    """v2.18.0:收尾导出 = 提交本 run 的经验快照(Q14)。

    快照含收尾 subagent 写的 ``experience/exported-global-experiences.yaml``(逐条
    schema 校验沿用 v2.17.0)、全部 trial 结果、入榜配方与测量、检索日志;服务不可
    达时写 outbox(Q26)。合并进全局原则由收尾 subagent 按
    ``references/experience-library-merge.md`` 执行(merge-context → 决议 →
    merge-apply)。旧实现写 skill 内置 memory/,已停用。
    """
    from runtime import library_view
    from runtime.memory import load_experience_export
    from .library_snapshot import build_run_snapshot, submit_or_queue

    run_dir = _run_dir_from(args)
    src = run_dir / "experience" / "exported-global-experiences.yaml"
    chk: dict = {}
    if src.exists():
        chk = load_experience_export(src)
        for err in (chk.get("errors") or [])[:20]:
            logger.error("经验导出 schema 错误: 第 %s 条(行 %s)字段 %s: %s —— %s",
                         err.get("index"), err.get("line"), err.get("field") or "(条目)",
                         err.get("error"), err.get("fix"))
        if chk.get("errors"):
            logger.error("修复提示: %s(合法条目仍会进快照;修好后重跑本命令会提交新快照)",
                         chk.get("fix_hint"))
    else:
        logger.warning("未找到 %s:快照只含 trial 结果与配方,没有候选原则", src)
    settings = library_view.client_settings(_load_config(run_dir))
    repo = library_view.client(settings, model=getattr(args, "model", None) or None,
                               run=str(run_dir))
    staged: dict = {}
    payload, report = build_run_snapshot(run_dir, repo,
                                         model=getattr(args, "model", "") or "",
                                         stage=staged)
    result = submit_or_queue(run_dir, repo, payload, namespace=settings["namespace"],
                             staged=staged)
    out = {**report, **result}
    if result["status"] == "submitted":
        out["next"] = ("收尾 subagent 合并:python %s library merge-context --snapshot %s "
                       "--out <run>/experience/merge-context.json,再按 "
                       "references/experience-library-merge.md 写决议并 merge-apply"
                       % (_SKILL_ROOT / "scripts" / "modelevo.py", report["snapshot_id"]))
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    # v2.17.0(#9)契约保留:导出源有 schema 错误时 exit 1(合法条目已进快照,修好重跑)
    if chk.get("errors"):
        return 1
    return 0 if result["status"] == "submitted" else 3


def _diag_library(run_dir: Path) -> list[dict]:
    """v2.18.0:经验库视图 / outbox 诊断(只读本地文件,不访问服务)。"""
    out: list[dict] = []
    st = load_json(run_dir / "run-state.json") if (run_dir / "run-state.json").exists() else {}
    lib = st.get("library") or {}
    serve = "python %s library serve --daemon" % (_SKILL_ROOT / "scripts" / "modelevo.py")
    # 切换(§11.4):v2.18.0 前 init 的 run 依赖已删除的 skill 内置 memory/,不支持在
    # 本版本上续跑;已完成最终评估的 run 只读不受影响。未记 skill_version 的是 v2.15.8
    # 之前的 run,早已结束,不再判定
    from runtime import version_at_least
    if st.get("skill_version") and not version_at_least(st["skill_version"], "2.18.0") \
            and st.get("phase") != "FINAL_EVALUATED":
        out.append({"code": "run_predates_experience_library", "severity": "critical",
                    "detail": {"skill_version": st.get("skill_version"),
                               "phase": st.get("phase")},
                    "remedy": "本 run 由 v2.18.0 之前的版本创建(旧 memory/ 经验库),不支持在本版本上"
                              "续跑:用切换前的安装完成它,或用本版本新开 run"})
    if lib and lib.get("watermark") is None:
        out.append({"code": "library_unreachable", "severity": "warning",
                    "detail": {"uri": lib.get("uri"), "source": lib.get("source")},
                    "remedy": f"本 run 未取到经验库视图(服务不可达且无本机缓存),不使用经验;"
                              f"请用户启动服务:{serve}"})
    elif lib.get("stale"):
        out.append({"code": "library_stale", "severity": "info",
                    "detail": {"watermark": lib.get("watermark"), "uri": lib.get("uri")},
                    "remedy": "经验库视图来自本机缓存(固定时服务不可达);可继续,结果可能缺最新经验"})
    box = run_dir / "experience" / "outbox"
    pending = sorted(p.name for p in box.glob("snapshot-*.json")) if box.is_dir() else []
    if pending:
        out.append({"code": "library_outbox_pending", "severity": "warning",
                    "detail": {"files": pending[:5]},
                    "remedy": "快照提交时服务不可达;服务恢复后执行 python %s library "
                              "submit-pending --run %s" % (_SKILL_ROOT / "scripts" / "modelevo.py",
                                                         run_dir)})
    return out


def cmd_assets_config(args) -> int:
    """打印 skill 级配置模板(直接读 assets/global-config.yaml 原文)。"""
    path = _SKILL_ROOT / "assets" / "global-config.yaml"
    print(path.read_text(encoding="utf-8"), end="")
    return 0


def cmd_snapshot(args) -> int:
    """打印当前资源快照与并发计划(Planner 上下文)。

    v2.12.2.9:输出**两种口径**,消除历史上反复出现的误读。
    旧实现只给 ``plan(planned_trials=1)`` 的结果,于是 ``parallel=1``
    被读成"整个 run 会串行跑"——实测一次排障据此错误地判断 48h 预算
    不可达(实际代际并发是 4,见下)。两种口径语义:
      - ``single_trial``:单 trial 独占机器时的分配(threads 吃满上限),
        **baseline 是有意按此口径串行的**(见 scheduler.plan 注释);
      - ``generation_batch``:本代 N 个 trial 的实际分配,**这才是
        run-generation 的并发计划**;N 由 config 的
        evolution.max_trials 与快照容量共同决定(容量 = 线程压到下限
        4 时最多几路,cap 8)。

    v2.17.0(run4 regression.2 #11):generation_batch 只按 CPU/GPU 算并发,
    不看内存(同机其它 run 常驻 300~740GB 时 8 并发必 OOM)。新增:
      - ``system_memory``:MemAvailable、cgroup 限额/用量/可用、有效可用;
      - ``admission``:按内存准入口径的建议并发 suggested_parallel(单
        trial 估计 = 该 run 最近 3 个有峰值的 trial 的最大峰值 × 1.15),
        memory_bound=True 表示内存比 CPU 更紧;
      - ``gpu_occupancy``:各卡显存/利用率、外来进程数(非本 run)、是否干净。
    给 ``--run``(或在 run 目录内执行)时按该 run 估计与归属;否则只给机器视角。
    """
    from runtime.scheduler import (gpu_occupancy_view,
                                   resource_admission_view, system_memory)

    _rd = _run_dir_from(args)
    run_dir = _rd if ((_rd / "trials").is_dir()
                      or (_rd / "run-state.json").exists()) else None
    config = _load_config(run_dir) if run_dir is not None else Config.load()
    snap = resource_snapshot()
    sched = Scheduler(config, snap)
    single = sched.plan(planned_trials=1).to_dict()

    # 代际批口径:以容量上限作为批宽(快照是纯资源视角,不读 run-state,
    # 实际并发以 run-generation 的入参 trial 数为准)。槽位算法见
    # scheduler.plan:capacity = min(8, usable // 线程下限 4)。
    cap = max(1, min(8, int(
        config.get("execution.parallel_trials_auto_cap", 16))))
    batch_n = cap
    batch = sched.plan(planned_trials=batch_n).to_dict()
    mem = system_memory()
    try:
        admission = resource_admission_view(
            run_dir, config, max_parallel=batch.get("n_parallel_trials"),
            memory=mem)
    except Exception as exc:  # noqa: BLE001  snapshot 只读汇总,不因此失败
        admission = {"error": str(exc)[:200]}
    try:
        gpu_occ = gpu_occupancy_view(run_dir)
    except Exception as exc:  # noqa: BLE001
        gpu_occ = {"available": False, "error": str(exc)[:200]}
    print(json.dumps({
        "snapshot": snap,
        "plan": single,
        "plans_by_scope": {
            "single_trial": single,
            "generation_batch": batch,
            "note": ("single_trial = 单 trial 独占口径(baseline 有意串行,"
                     "parallel 恒为 1 属预期);generation_batch = "
                     "run-generation 的并发计划,以本代实际提交数为准"
                     "(batch_n 为容量上限 %d)。generation_batch 只按 "
                     "CPU/GPU 算;内存口径见 admission.suggested_parallel"
                     "(训练启动前逐个准入,不满足就排队)" % batch_n),
        },
        "run_dir": str(run_dir) if run_dir is not None else None,
        "system_memory": mem,
        "admission": admission,
        "gpu_occupancy": gpu_occ,
    }, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_planner_context(args) -> int:
    """一键组装 Planner 八类上下文(v8;替代 Agent 自行读八个文件)。"""
    from runtime.planner_context import (build_planner_context,
                                         render_planner_context)

    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    config = _load_config(run_dir)
    ctx = build_planner_context(run_dir, spec, config, top_k=int(args.top_k))
    if args.json:
        print(json.dumps(ctx, ensure_ascii=False, indent=2,
                         default=str))
    else:
        print(render_planner_context(ctx))
    return 0


def cmd_capacity_check(args) -> int:
    """容量诊断:样本量曲线/列数曲线/局部标签歧义 → 行动排序。

    在抽样面(splits-sample)上跑;--reference-trial 提供已落盘的
    feature_importance.csv 时列数曲线生效。v12(P1-7):每格点重复
    seed + 末段增益 95% CI;探针改桶内方差口径。

    v2.12.2.9 降本三项(判定语义不变):
    - ``--seeds``:样本量曲线每格点重复数(默认 2;旧行为固定 3)。
      该曲线是整条命令的主要成本,而 std 只服务"末段增益是否覆盖 0"
      一个判定;
    - ``--no-cache``:默认按抽样面指纹复用已有 capacity-report.json,
      指纹未变则秒回;
    - ``--force``:等价于 --no-cache 并忽略现有报告。
    v2.16.0:产物只作参考诊断(planner-context §10、report),不再参与
    model/loss/hpo 放行——「先特征、后模型」由特征期门执行(见
    runtime/feature_phase.py)。本命令可选,任何时候跑都不会挡候选。
    """
    from runtime.capacity import run_capacity_check

    run_dir = _run_dir_from(args)
    out = run_capacity_check(run_dir,
                             splits_subdir=args.splits_dir,
                             reference_trial=args.reference_trial,
                             n_seeds=getattr(args, "seeds", None),
                             use_cache=not getattr(args, "no_cache", False))
    v = out.get("verdict") or {}
    print(json.dumps({"verdict": v,
                      "sample_curve_last_gain": (out.get(
                          "sample_size_curve") or {}).get("last_step_gain"),
                      "sample_curve_ci_covers_zero": (out.get(
                          "sample_size_curve") or {}).get(
                          "last_step_ci_covers_zero"),
                      "sample_curve_n_seeds": (out.get(
                          "sample_size_curve") or {}).get("n_seeds"),
                      "feature_curve_saturated": (out.get(
                          "feature_count_curve") or {}).get("saturated"),
                      "local_ambiguity_index": (out.get(
                          "irreducible_noise") or {}).get(
                              "local_ambiguity_index")},
                     ensure_ascii=False, indent=2))
    print(f"产物: {run_dir}/sample-profile/capacity-report.json")
    return 0


def cmd_preflight(args) -> int:
    """依赖预检(v11):三层依赖策略的确定性执行。

    required tier(requirements.txt 声明)缺失 → 明确报出并要求安装;
    optional tier(optuna/catboost/torch…)try-import 记录 import_ok;
    产物 env-manifest.json 与 harness_hash 并列写 run-state。

    v20:本命令是「当前环境」基准的刷新点,漂移**不再判死** —— 覆盖
    指纹前先 diff 并记入 env-ledger.jsonl(旧实现是无条件静默覆盖,
    等于一条零留痕的官方手改通道;用户决策 2026-09:不拦 pip,只记账)。
    """
    from runtime.preflight import (DRIFT_KINDS, classify_env_drift,
                                   record_env_events, run_preflight)

    run_dir = _run_dir_from(args)
    manifest = run_preflight(
        _SKILL_ROOT, output_path=run_dir / "env-manifest.json")
    with locked_state_write(run_dir / "run-state.json"):
        st = load_json(run_dir / "run-state.json") if (
            run_dir / "run-state.json").exists() else {}
        # v20:覆盖前记账 —— 这条通道此前无 reason、无留痕,与 harness
        # 侧被点名批评的"手改 run-state"同类,只是它是官方的。
        diff = classify_env_drift(st.get("env_manifest") or {}, manifest)
        if diff["changed"] and not diff["unknown"]:
            record_env_events(run_dir, diff["events"],
                              phase=st.get("phase"),
                              generation=st.get("current_generation"))
        st["env_manifest"] = manifest
        st["env_fingerprint"] = manifest.get("fingerprint")
        dump_json(run_dir / "run-state.json", st)
    missing = [m["name"] for m in manifest["packages"]
               if m["tier"] == "required" and not m["import_ok"]]
    if missing:
        # v20:缺依赖是 setup 阶段该阻塞的形态(环境不完整,跑出来的
        # baseline 参照点本身不可信),与"装了什么包"是两件事。装上
        # 之后重跑 preflight 会把基线刷新到新环境并记一条
        # required_appeared(修复),不再把整轮运行判死。
        logger.error("required 依赖缺失: %s\n请用户安装后重跑 preflight"
                     "(安装动作本身不需要许可;环境变化会记入 "
                     "env-ledger.jsonl,不阻塞运行)", missing)
        return 1
    # v14:trial 子进程解释器的 required 缺失同样阻塞(实测事故:
    # CLI 解释器全绿、PATH python 无 pandas → baseline 10 连败)
    ti = manifest.get("trial_interpreter") or {}
    ti_missing = ti.get("missing") or []
    if ti.get("probe_ok") and ti_missing:
        logger.error("trial 子进程解释器(%s)缺依赖: %s\n"
                     "preflight 探的是 CLI 解释器,trial 经 scheduler "
                     "另起进程(命令首位 python → sys.executable);"
                     "两者不一致时请用完整环境解释器启动 modelevo.py",
                     ti.get("interpreter"), ti_missing)
        return 1
    print(json.dumps({"fingerprint": manifest.get("fingerprint"),
                      "interpreter": manifest.get("interpreter"),
                      "required_ok": not missing,
                      "trial_interpreter": ti,
                      "optional_available": {m["name"]: m["import_ok"]
                                             for m in manifest["packages"]
                                             if m["tier"] == "optional"}},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_acknowledge_harness_change(args) -> int:
    """铁律 10 的受支持出口(v14):登记 harness 变更,替代手改 run-state。

    - 只动了 plumbing 模块(不影响任何落盘数字)→ 直接重新盖章,
      历史 trial 仍可比;
    - 动了 measurement 模块 → 默认拒绝;显式 --allow-measurement-impact
      时盖章并把 harness_epoch +1,新旧 epoch 的 trial 由 leaderboard
      拒绝混排(等价"重新 baseline",但证据留痕而不是抹掉);
    - 动了 measurement 模块但**实测分数口径未变**(evaluator/metrics/
      split 落盘口径逐文件核对后确认不动分数量纲与可比性)→ 显式
      --measurement-no-score-impact 降级登记:epoch 不 +1,历史 trial
      继续同表可比(v14.2 修复:旧实现只要 measurement 非空就 +1,
      实测两次不动口径的变更把 leaderboard 旧 epoch 全部排除,
      leaderboard 系统性不全)。
    实测背景:一次运行手改过三次 run-state.json 的 harness_hash,改了
    什么、为什么改,一条记录都没有 —— 铁律 10 名存实亡。

    v2.17.0(WP5b #22/#13):
    - `--verify-against <trial>[,...]`:用当前 evaluator 对这些 trial 的已
      落盘预测重新评估(不重训),与 evaluation.json 的主指标/Δ/CI/SE 逐项
      比对;全部一致才放行 --measurement-no-score-impact,不一致拒绝并列出
      差异(run4 regression.2 两次手工重跑 contrast 才敢登记)。不带该参数
      时行为不变;
    - 输出特征块缓存失效预判(runtime.feature_cache.predict_invalidation),
      并记入 harness_change_log。
    """
    run_dir = _run_dir_from(args)
    st = load_json(run_dir / "run-state.json") if (
        run_dir / "run-state.json").exists() else {}
    stamp = protocol_stamp(_SKILL_ROOT / "runtime")
    from runtime.harness import classify_harness_change

    diff = classify_harness_change(st.get("harness_files") or {},
                                   stamp["harness_files"])
    _cache_pred = None
    try:
        from runtime.feature_cache import predict_invalidation

        _cache_pred = predict_invalidation(
            run_dir, diff["changed"] if not diff.get("unknown") else None)
        print(f"特征块缓存预判: {_cache_pred['summary']}"
              f"(schema={_cache_pred['cache_schema_version']},"
              f"条目 {_cache_pred['n_valid']}/{_cache_pred['n_entries']} 有效)")
    except Exception as _exc:  # noqa: BLE001  预判失败不阻塞登记
        logger.warning("特征块缓存失效预判失败(不阻塞): %s", _exc)
    _hash_unchanged = st.get("harness_hash") == stamp["harness_hash"]
    _verify = None
    _verify_ids = [t.strip() for t in str(
        getattr(args, "verify_against", None) or "").split(",") if t.strip()]
    if _hash_unchanged and not _verify_ids:
        print("harness_hash 未变化,无需登记")
        return 0
    if _verify_ids:
        from runtime.harness import verify_trial_evaluation

        _spec = _load_spec(args)
        _cfg = _load_config(run_dir)
        _verify = [verify_trial_evaluation(run_dir, _spec, _cfg, tid)
                   for tid in _verify_ids]
        print("verify-against(当前 evaluator 重评已落盘预测,不重训):")
        for _r in _verify:
            _flag = "一致" if _r.get("consistent") else "不一致"
            print(f"  {_r['trial']}: {_flag}  比对字段 "
                  f"{len(_r.get('compared_fields') or [])} 个"
                  + (f"  父代={_r.get('parent_used')}"
                     if _r.get("parent_used") else ""))
            if _r.get("error"):
                print(f"    错误: {_r['error']}")
            for _d in (_r.get("diffs") or [])[:12]:
                print(f"    {_d['field']}: 存档={_d['stored']} "
                      f"重算={_d['recomputed']}")
        _all_ok = all(r.get("consistent") for r in _verify)
        if _hash_unchanged:
            # v2.17.0(D7):hash 未变时 --verify-against 照常复评比对并打印,
            # 但不登记变更(没有可登记的 harness 变化);不一致 exit 1
            # —— 旧实现先判 hash 未变直接 return 0,复评被静默跳过。
            print("harness_hash 未变化,不登记变更(仅执行 verify-against 复评"
                  "比对)" + ("" if _all_ok else ":存在不一致,见上"))
            if not _all_ok:
                logger.error(
                    "verify-against 存在不一致(harness_hash 未变):已落盘的"
                    "评估数字与当前 evaluator 复评不符,请先 diagnose 定位"
                    "(evaluation.json 被手改 / 预测文件被覆写 / 依赖库版本漂移)")
                return 1
            return 0
        if args.measurement_no_score_impact and not _all_ok:
            logger.error(
                "verify-against 存在不一致(见上),拒绝 "
                "--measurement-no-score-impact 登记:本次 harness 变更改变了"
                "已落盘预测的评估数字。确认影响口径时改用 "
                "--allow-measurement-impact(harness_epoch +1)")
            return 1
    if diff["unknown"]:
        logger.warning("run-state 无 harness_files(v14 之前的 run):"
                       "全部变更按 measurement 处理")
    if diff["measurement"] and not (args.allow_measurement_impact
                                    or args.measurement_no_score_impact):
        logger.error(
            "measurement 模块已变更 %s:按铁律 10 必须重新 baseline。"
            "确认影响分数口径时显式加 --allow-measurement-impact"
            "(harness_epoch +1,新旧 epoch 不混排);"
            "确认实测口径未变(不动 evaluator/metrics 的落盘数字)时加"
            " --measurement-no-score-impact(epoch 不变,历史 trial 仍可比)",
            diff["measurement"])
        return 1
    if (args.measurement_no_score_impact
            and not args.allow_measurement_impact and diff["measurement"]):
        # v14.2 降级标记:measurement 文件变了,但逐文件核对确认
        # 分数量纲/可比性不变 → 不 bump epoch,留痕 measurement_impact
        # = "no-score-impact" 供审计区分
        epoch = int(st.get("harness_epoch", 0))
        impact = "no-score-impact"
    else:
        epoch = int(st.get("harness_epoch", 0)) + (
            1 if diff["measurement"] else 0)
        impact = bool(diff["measurement"])
    log = list(st.get("harness_change_log") or [])
    _entry = {
        "at": now_iso(),
        "from_hash": st.get("harness_hash"),
        "to_hash": stamp["harness_hash"],
        "changed": diff["changed"],
        "measurement_impact": impact,
        "harness_epoch": epoch,
        "reason": args.reason,
    }
    if _verify is not None:
        _entry["verified_against"] = [
            {"trial": r["trial"], "consistent": bool(r.get("consistent")),
             "n_fields": len(r.get("compared_fields") or []),
             "n_diffs": len(r.get("diffs") or [])} for r in _verify]
    if _cache_pred is not None:
        _entry["feature_cache"] = {
            k: _cache_pred.get(k) for k in (
                "will_invalidate", "n_entries", "n_valid",
                "cache_schema_version", "summary")}
    log.append(_entry)
    _update_run_state(run_dir, harness_epoch=epoch,
                      harness_change_log=log, **stamp)
    print(f"已登记 harness 变更: {st.get('harness_hash')} → "
          f"{stamp['harness_hash']}(epoch={epoch},"
          f"measurement={impact})")
    return 0


def cmd_check_products(args) -> int:
    """四产物自消费冒烟(v14/v21):check_data_products + check_consumers。

    profile 尾部自动执行;write-splits-view / write-splits-sample
    不经过 cmd_profile,落盘后需手动跑本命令。
    """
    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    from runtime.preflight import check_consumers, check_data_products

    from .context import _resolve_splits_arg

    # v2.17.2(B6):run_dir 已 resolve、残留 MODELEVO_PRESCREEN_ROOT 已在
    # _run_dir_from 中失效;--splits-dir 同样落成绝对路径
    splits_dir = _resolve_splits_arg(run_dir, args.splits_dir)
    issues = (check_data_products(run_dir, spec,
                                  splits_dir=splits_dir)
              + check_consumers(run_dir, spec,
                                splits_dir=splits_dir))
    if issues:
        for msg in issues:
            logger.error("产物自消费冒烟失败: %s", msg)
        return 1
    print(f"四产物自消费冒烟通过: {splits_dir}")
    return 0
