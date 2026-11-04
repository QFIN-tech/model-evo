"""Generation:一轮演进单元(§11)与停止条件(§17)。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .logging_utils import dump_yaml, load_yaml, now_iso
from .spec import TaskSpec


class GenerationRecord:
    def __init__(self, run_dir: Path, generation_id: int):
        self.gid = int(generation_id)
        self.dir = Path(run_dir) / "generations" / f"gen-{self.gid:02d}"
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---- 输入侧:计划 ----
    def write_plan(self, plan: dict) -> Path:
        path = self.dir / "plan.yaml"
        dump_yaml(path, {"generation_id": f"gen-{self.gid:02d}",
                         "created_at": now_iso(), **plan})
        return path

    def load_plan(self) -> dict | None:
        path = self.dir / "plan.yaml"
        return load_yaml(path) if path.exists() else None

    # ---- 输出侧:generation.yaml(§11.2)----
    def write_summary(
        self,
        *,
        planned_trials: list[str],
        best_trial: str | None,
        best_metric: float | None,
        best_delta_vs_previous_generation: float | None,
        reflection: dict,
        archive_update: dict,
        next_plan: dict | None = None,
        stop_report: dict | None = None,
        shadow_check: dict | None = None,
        budget_metrics: dict | None = None,
        feature_beam: dict | None = None,
        warm_start_validation: list[dict] | None = None,
    ) -> Path:
        payload = {
            "generation_id": f"gen-{self.gid:02d}",
            "planned_trials": planned_trials,
            "summary": {
                "best_trial": best_trial,
                "best_metric": best_metric,
                "best_delta_vs_previous_generation":
                    best_delta_vs_previous_generation,
            },
            "reflection": reflection,
            "archive_update": archive_update,
            "next_plan": next_plan,
            "stop_report": stop_report,
            "shadow_check": shadow_check,  # v8:shadow sanity(仅非 None 写入)
            "budget_metrics": budget_metrics,  # v16:wall/compute 双口径
            # v2.13.2:特征 bundle 槽位对账(仅非 None 写入)。
            # 只记不拒:候选供给薄是合法状态,硬拒会把 run 卡死;
            # 但"想填满却没填满"必须留痕,否则宽度配置的失真无法审计。
            "feature_beam": feature_beam,
            # v2.15.6:历史 top-trial 配方在本 run 的有效性验证(仅 gen-00 非 None)。
            # 判据是 delta_vs_baseline 的统计显著性;reference_auc 仅供展示,
            # 不参与任何裁决(AUC 绝对值会随样本日期变化)。
            "warm_start_validation": warm_start_validation,
            "written_at": now_iso(),
        }
        path = self.dir / "generation.yaml"
        dump_yaml(path, payload)
        return path

    def add_fields(self, **fields: Any) -> Path:
        """v2.17.0:向已写出的 generation.yaml 追加字段(如 parallelism)。"""
        path = self.dir / "generation.yaml"
        data = (load_yaml(path) or {}) if path.exists() else {
            "generation_id": f"gen-{self.gid:02d}"}
        data.update(fields)
        dump_yaml(path, data)
        return path


def empty_reflection() -> dict:
    return {
        "key_findings": [],
        "successful_directions": [],
        "failed_directions": [],
        "diminishing_return_directions": [],
        "unexplored_high_potential_directions": [],
        "timeout_unverified_directions": [],
    }


# ---------------------------------------------------------------------------
# Stop Condition(§17)
# ---------------------------------------------------------------------------

#: v2.17.0(#17):STOPPED 的命令语义(据实核对代码,2026-10):
#: - run-state.phase=STOPPED 只是状态标记。new-trial / run-trial /
#:   run-generation / evaluate / finalize-generation / planner-context 均
#:   **不检查 phase**,停止后继续演进不会被拒(phase 单调,不会回退到
#:   EVOLVING;继续跑的代照常 finalize、可再次命中停止)。
#: - 真正的硬拦截只来自预算护栏本身,见 :func:`stop_blocks`。
STOP_ADVISORY_NOTE = (
    "STOPPED 是建议性状态:new-trial/run-trial/run-generation/"
    "finalize-generation 均不检查 phase,继续演进不会被拒;只有 blocks "
    "中列出的预算护栏会硬拦截。停止后应先处理 delivery_resolution 与 "
    "undelivered_gain,再决定收尾(finalize-retrain/evaluate-final)或继续")

#: reason code → 中文解释(stop_report.reason_detail)
REASON_DETAIL: dict[str, str] = {
    "budget_max_trials": (
        "已启动 trial 数 ≥ max_trials。**硬拦截**:new-trial 创建口"
        "(TrialFactory.create)抛 BudgetExhausted;要继续须提高 "
        "resource_budget.max_trials。"),
    "budget_max_generations": (
        "已 finalize 代数 ≥ max_generations。仅建议停止:没有任何命令检查"
        "代数上限,继续 new-trial/run 不会被拒;uniform alpha 排程在超出后"
        "按 gens_remaining=1 计(每代 q=min(q_cap, 剩余 alpha)),LORD 排程"
        "超出视界后 γ_t=0、只靠发现回补。"),
    "budget_wall_time": (
        "墙钟 ≥ max_wall_time_hours。**硬拦截 run-generation**:wall 预算已"
        "耗尽时整代拒绝启动并把本代 trial 记 timeout(gen_executor,所有 run);"
        "skill_version≥2.15.8 的 run 另有余量预检(余量 < 最短单 trial 超时即"
        "拒绝)。new-trial 与单 trial run-trial 不拦。"),
    "budget_compute": (
        "累计 trial 计算时长 ≥ max_compute_hours。仅建议停止,不拦截任何命令。"),
    "performance_target_reached": (
        "冠军主指标达到 task-spec performance_target。仅建议停止。"),
    "performance_improvement_target_reached": (
        "冠军相对 baseline 的提升达到 performance_target."
        "improvement_vs_baseline。仅建议停止。"),
    "roi_diminishing_return": (
        "代级 ROI(代冠军增益 / 该代 compute 小时)连续 lookback 代低于历史"
        "最佳的 roi_diminishing_ratio 倍,且最近一代增益的保守下界"
        "(Δ−1.28·noise)< MDE。全局信号,不受单口径护栏限制。注意:"
        "generation_best_deltas 的口径——v2.17.0 起新 run = 本代 finalize "
        "后的冠军主指标 − 本代开始前的冠军主指标(方向感知;首代起点为 "
        "baseline 主指标;逐代冠军记录在 evolution-state.champion_metric_history)"
        ",冠军没换时恰为 0,换成更差的可比冠军时为负;旧 run(skill_version "
        "< 2.17.0)沿用旧口径 = 本代 leaderboard 冠军 − 上一代最优非 screening "
        "trial,冠军没换时也可能为正(上一代最优不是冠军时)。空代(本代无任何"
        "已评估 trial)的 delta 记 None,不计入 ROI 与收敛判据,也不会因空代"
        "触发本条。0/负值会被当成'无增益'——此时请结合 undelivered_gain 判断"
        "是'真没增益'还是'增益没晋级'。仅建议停止。"),
    "ceiling_gap_within_mde": (
        "v2.19.0:冠军与天花板估计(<run>/ceiling/ceiling-estimate.json,等预算"
        "强参照在 validation 上的最优值)的差距 gap_to_ref ≤ MDE。不受单口径护栏"
        "限制。只在参照建在 trial 特征面(feature_surface=trial:<id>)上时触发;"
        "base 面参照只在 stop_report.ceiling 中呈现。仅建议停止:把剩余预算转给"
        "确认与交付。"),
    "plateau_below_half_mde": (
        "v2.19.0:最近 evaluator.plateau_stop_generations 个非空代的冠军增益"
        "(方向感知)都 < evaluator.plateau_stop_ratio × MDE。冷启动 run 进入"
        "'在已饱和的特征面上做通用变换'的低收益区时的停止信号(无经验对照 run"
        "实测:gen-04 起 15 个 trial |Δ|≤5e-4,10 h 只换 +0.0003)。不受单口径"
        "护栏限制。仅建议停止。"),
    "convergence_no_accepted": (
        "统计收敛:最近 3 个非空代的 best delta 的 95% CI 均覆盖 0 且最近一代"
        "无新增 accepted(空代 delta=None 不计入)。受单口径护栏约束:只在已"
        "尝试 ≥2 个非 sample 数据视图时才可触发。仅建议停止。"),
    "information_channels_exhausted": (
        "信息通道穷尽出口:最近一代无新增 accepted、best delta 不为正(或在"
        "噪声带内)、model/hpo 轴已开过、信息源清单无未消费项、Planner 候选"
        "价值上界 < MDE、且有新鲜确认的交付位——六条件全满足,视图数不限"
        "(单视图时 single_view_caveat=true)。仅建议停止。"),
    "single_data_view_insufficient": (
        "**这是单口径护栏的状态码,不是'可以停'的统计结论**。条件:≥3 个非空代"
        "(空代 delta=None 不计)、"
        "最近一代无新增 accepted、只尝试过 1 个非 sample 数据视图,且穷尽出口"
        "不满足。文档所说'单口径护栏阻断收敛停止'指的是:此时不得用 "
        "convergence_no_accepted 宣布'统计收敛/接近天花板'(结论只在一个数据"
        "口径上成立);但该码本身仍会使 stop_report 非空、phase 推进到 STOPPED"
        "(建议性,不拦截任何命令)——含义是'搜索已停滞且证据面单一'。出口:"
        "补一个独立数据视图(data_view trial)解除护栏,或满足 "
        "information_channels_exhausted 的六条件。与 roi_diminishing_return "
        "同时出现时,ROI 才是实际的停止依据,本码只是限定结论适用范围。"),
}


def stop_blocks(reasons: list[str]) -> list[str]:
    """v2.17.0(#17):停止原因中**会硬拦截**后续命令的项(据实核对代码)。"""
    out: list[str] = []
    if "budget_max_trials" in reasons:
        out.append("new-trial(TrialFactory.create → BudgetExhausted:"
                   "max_trials 已耗尽)")
    if "budget_wall_time" in reasons:
        out.append("run-generation(wall 预算已耗尽 → 整代拒绝启动、本代 "
                   "trial 记 timeout;skill_version≥2.15.8 另有余量预检)")
    return out


class StopCondition:
    """四类停止:budget / performance / convergence / diminishing_return。"""

    def __init__(self, spec: TaskSpec, config: Config, run_dir=None):
        self.spec = spec
        self.config = config
        # v2.17.0:MDE 走 runtime.mde.effective_mde(量纲感知),需要 run_dir
        # 读 baseline 主指标;None 时由 check() 的 baseline_metric 兜底
        self.run_dir = run_dir
        self._baseline_metric_hint: float | None = None
        from .metrics import metric_direction
        self.direction_better = metric_direction(spec.primary_metric)
        self.better = (lambda a, b: a > b) if self.direction_better \
            else (lambda a, b: a < b)

    def _mde(self) -> float:
        """有效 MDE(旧 run 无 relative 键 ⇒ 等于绝对值,行为不变)。"""
        try:
            from .mde import effective_mde

            return float(effective_mde(
                self.config, run_dir=self.run_dir,
                baseline_metric=self._baseline_metric_hint))
        except Exception:  # noqa: BLE001
            return float(self.config.get(
                "evaluator.minimum_meaningful_improvement", 0.001) or 0.001)

    def check(
        self,
        *,
        n_trials_started: int,
        n_generations: int,
        wall_time_hours: float,
        trials_accepted: int,
        best_metric: float | None,
        baseline_metric: float | None,
        generation_best_deltas: list[float],
        noise: float | None,
        unexplored_high_potential: list[str],
        compute_hours: float | None = None,
        max_compute_hours: float | None = None,
        data_views_tried: list[str] | None = None,
        direction_deltas: dict[str, list[float]] | None = None,
        direction_accepted: dict[str, int] | None = None,
        recent_gen_cis: list[tuple[float, float]] | None = None,
        # v18/P0-3:信息通道穷尽出口的外部输入(由 finalize 计算)
        inventory_unconsumed: int | None = None,
        # v2.10.1/F5:model/HPO 轴是否真的开过(由 finalize 从 trial 的
        # changed_modules/search_action 实测);None = 未知,按未开过处理。
        model_axis_tried: bool | None = None,
        candidate_value_upper_bound: float | None = None,
        fresh_confirmation_available: bool | None = None,
        # v2.15.x ROI:每代的全链路 compute 成本(含 confirm),用于代级
        # 边际 ROI 计算。长度与 generation_best_deltas 一一对应。
        # None = 未提供,代级 ROI 停止条件不触发。
        generation_compute_hours: list[float] | None = None,
        # v2.17.0(#17):未交付累积增益(由 finalize 计算,见
        # evolution.compute_undelivered_gain);只进 stop_report,不参与触发
        undelivered_gain: dict | None = None,
        # v2.17.0(D3):本代无任何已评估 trial(空代)。空代没有新证据,
        # 停止判定只保留与空代无关的硬预算(trials/wall/compute)
        empty_generation: bool = False,
        # v2.19.0:天花板估计摘要(runtime.ceiling.ceiling_summary,含
        # gap_to_ref);None = 未运行 ceiling,对应停止出口不触发
        ceiling: dict | None = None,
    ) -> dict | None:
        """返回停止报告(§17.5);None 表示继续。

        trials_accepted 口径:最近一代新增 accepted(§17.3"没有新的
        accepted Trial"),不是全史累计——由调用方传入。

        单口径护栏(协议 v7→v12 修正):convergence / diminishing_return
        只在已尝试 **≥2 个非 sample 数据视图**时才可触发——特征探索阶段
        强制产生 sample 视图,旧口径把 sample 也算进 views,护栏形同
        虚设(实际只有 budget 会触发停止)。budget/performance 不受限。

        v12(P1-8)统计化:
        - convergence 判据改为"最近 3 代 best delta 的 bootstrap CI 均
          覆盖 0"(旧口径点值 < noise,训练随机性即可穿透);
        - unexplored_high_potential 从硬否决降为 confidence 降档信号
          (旧口径下 LLM 写一条就无限期阻止停止、写空就立刻停——
          停止判据可被自由操纵);
        - confidence_of_near_ceiling 按"收敛证据强度"实算(旧口径
          budget 停止一律贴 medium,无任何计算)。
        """
        reasons: list[str] = []
        self._baseline_metric_hint = baseline_metric
        # v2.17.0(D3):delta=None(空代 / 无冠军)不计入收敛与 ROI 判据 ——
        # 逐代序列按"非空代"压缩;compute 小时与之逐代对齐后同步压缩。
        _deltas_raw = list(generation_best_deltas or [])
        generation_best_deltas = [d for d in _deltas_raw if d is not None]
        if generation_compute_hours is not None:
            if len(generation_compute_hours) == len(_deltas_raw):
                generation_compute_hours = [
                    h for d, h in zip(_deltas_raw, generation_compute_hours)
                    if d is not None]
            else:
                generation_compute_hours = None
        views = sorted(set(data_views_tried or ["v1"]))
        # G7/P1-8:sample 视图是特征探索的工作面,不算独立"数据口径"
        non_sample_views = [v for v in views if v != "sample"]
        single_view = len(non_sample_views) <= 1

        budget = self.spec.resource_budget or {}
        max_trials = int(budget.get("max_trials")
                         or self.config.get("evolution.max_trials", 80))
        max_gens = int(budget.get("max_generations")
                       or self.config.get("evolution.max_generations", 8))
        max_wall = float(budget.get("max_wall_time_hours")
                         or self.config.get("execution.max_wall_time_hours", 24))
        if n_trials_started >= max_trials:
            reasons.append("budget_max_trials")
        if n_generations >= max_gens:
            reasons.append("budget_max_generations")
        if wall_time_hours >= max_wall:
            reasons.append("budget_wall_time")
        # v16/P0-2:compute 口径预算(默认判定)。墙钟含 LLM 编排延迟、
        # 会话空闲与外部中断——实测 120w run 32.1h 墙钟中 19.4h 为空闲
        # (gen-02→gen-03 之间含过夜),计算仅 3.7h 却被 wall 预算判停。
        # 新口径:budget_compute_hours 显式配置时以"实际计算时长"判定;
        # wall 保留为绝对上限护栏(防真失控)。stop_report 双轨呈现。
        max_compute = None
        if max_compute_hours is not None:
            max_compute = float(max_compute_hours)
        else:
            max_compute = budget.get("max_compute_hours")
            if max_compute is None:
                max_compute = self.config.get(
                    "evolution.max_compute_hours")
        if (max_compute is not None and compute_hours is not None
                and float(compute_hours) >= float(max_compute)):
            reasons.append("budget_compute")

        # performance stop(§17.2)
        perf_target = (self.spec.data.get("performance_target") or {})
        target = perf_target.get(self.spec.primary_metric)
        if target is not None and best_metric is not None and \
                self.better(best_metric, float(target)):
            reasons.append("performance_target_reached")
        imp_target = perf_target.get("improvement_vs_baseline")
        if (imp_target is not None and best_metric is not None
                and baseline_metric is not None
                and self.better(best_metric - baseline_metric, float(imp_target))):
            reasons.append("performance_improvement_target_reached")

        # ---- v2.15.x ROI:代级边际 ROI 停止 ----
        # 原理:单 trial 的 MDE 拦不住"每代微涨"的慢速漂移,5 代 +0.0002
        # 都"不显著",但累计消耗 20h 算力。代级 ROI = 该代冠军增益 /
        # 该代全链路 compute。连续 2 代 R_gen 低于历史最佳的 1/5 且
        # 冠军增益 LCB < MDE,触发停止(非 single_view 护栏不适用——
        # 代级 ROI 崩塌是全局信号,无论视图数都应停止)。
        # 需要调用方传入 generation_compute_hours(与 generation_best_deltas
        # 等长);缺失时静默跳过,不影响现有停止出口。
        # 代级 ROI 停止不受 single_view 护栏限制——ROI 崩塌是全局信号,
        # 无论视图数都应停止。convergence/diminishing_return 才需要多视图。
        if (
            generation_compute_hours is not None
            and len(generation_compute_hours) >= 2
            and len(generation_best_deltas) >= 2
        ):
            _mde_roi = self._mde()
            _roi_ratio_threshold = float(self.config.get(
                "evaluator.roi_diminishing_ratio", 0.2))
            _roi_lookback = int(self.config.get(
                "evaluator.roi_diminishing_lookback", 2))
            # 按代计算 R_gen = delta / compute_hours(防 0 除)
            _eps_h = 1e-3
            _rgen = [
                float(d) / max(float(h), _eps_h)
                for d, h in zip(generation_best_deltas,
                                generation_compute_hours)
                if d is not None and h is not None
            ]
            if len(_rgen) >= _roi_lookback + 1:
                # 历史最佳 R_gen(不含最近 lookback 代)
                _hist_best = max(_rgen[:-_roi_lookback]) if \
                    _rgen[:-_roi_lookback] else None
                _recent = _rgen[-_roi_lookback:]
                if _hist_best is not None and _hist_best > 0:
                    _ratio_ok = all(
                        r < _roi_ratio_threshold * _hist_best
                        for r in _recent)
                    # LCB 判据:最近一代冠军增益的保守下界 < MDE
                    # 近似用 noise 作 SE(无 se_eval 时的降级)
                    _last_delta = generation_best_deltas[-1]
                    _lcb_ok = (
                        _last_delta is not None
                        and noise is not None
                        and np.isfinite(noise)
                        and noise > 0
                        and float(_last_delta) - 1.28 * float(noise)
                        < _mde_roi
                    )
                    if _ratio_ok and _lcb_ok:
                        reasons.append("roi_diminishing_return")

        # ---- v2.19.0:天花板差距与平台期停止(均不受单口径护栏限制)----
        _mde_v219 = None
        # base 面参照是"未加宽特征面"上的强模型,冠军加宽后超过它不代表
        # 到顶 —— 只有参照建在 trial 特征面(trial:<id>)上时才据此判停,
        # base 面只呈现不触发
        _gap = (ceiling or {}).get("gap_to_ref")
        if _gap is not None and str((ceiling or {}).get(
                "feature_surface") or "base") != "base" \
                and self.config.get("evaluator.ceiling_stop_enabled", True):
            _mde_v219 = self._mde()
            try:
                if float(_gap) <= float(_mde_v219):
                    reasons.append("ceiling_gap_within_mde")
            except (TypeError, ValueError):
                pass
        _plateau_n = self.config.get("evaluator.plateau_stop_generations")
        try:
            _plateau_n = int(_plateau_n) if _plateau_n else 0
        except (TypeError, ValueError):
            _plateau_n = 0
        if _plateau_n > 0 and len(generation_best_deltas) >= _plateau_n:
            if _mde_v219 is None:
                _mde_v219 = self._mde()
            _ratio = float(self.config.get("evaluator.plateau_stop_ratio",
                                           0.5) or 0.5)
            # generation_best_deltas 已是方向感知的"冠军增益"(v2.17.0 起
            # 新 run 口径),正数 = 变好
            if all(float(d) < _ratio * float(_mde_v219)
                   for d in generation_best_deltas[-_plateau_n:]):
                reasons.append("plateau_below_half_mde")

        # ---- convergence stop(§17.3,v12 统计化)----
        # 最近 3 代 best delta 的 95% CI 全部覆盖 0 且最近一代无新增
        # accepted → 统计收敛(点值 < noise 的旧口径会被训练随机性穿透)
        if len(generation_best_deltas) >= 3 and not single_view:
            recent = [d for d in generation_best_deltas[-3:]
                      if d is not None]
            cis = (recent_gen_cis or [])[-3:]
            if len(recent) == 3:
                if cis and len(cis) == 3:
                    all_cover_zero = all(
                        lo <= 0 <= hi for lo, hi in cis)
                    if all_cover_zero and trials_accepted == 0:
                        reasons.append("convergence_no_accepted")
                elif noise is not None and np.isfinite(noise) \
                        and noise > 0 and trials_accepted == 0:
                    # 无逐代 CI 时退化为旧口径(方向感知)
                    if all((d < noise) if self.direction_better
                           else (d > -noise) for d in recent):
                        reasons.append("convergence_no_accepted")

        # ---- v18/P0-3:信息通道穷尽出口(information_channels_exhausted)
        # 单口径护栏的死锁根因(实测两 v17.4 run 均以
        # single_data_view_insufficient 挂尾):样本源固定时 add_rows
        # 物理不可兑现,而真实任务的口径多样性体现在 reparse 解析粒度/
        # 特征族加宽上,不是 splits 目录数——两 run 全部特征方向都走
        # "v1 面加列"(路径 B),独立视图 0 个,护栏永远触发,演进在
        # 统计已收敛后仍不能自动停,靠 LLM 自觉升级到人收口;400w run
        # 还出现"为触发收敛判定去 gen-05 补独立视图"的激励倒挂。
        # 出口 = 六条件**全**满足(确定性代码裁决,反思只举证):
        #   1. 最近一代无新增 accepted(不再有方向被验证为增量);
        #   2. 最近一代 best delta 在噪声带内(点值收敛);
        #   3. 最近一代只投了 exploit/hpo/confirm(细化与合并,不是
        #      新方向的探索——explore 停止 = 探索者自认无方向可投);
        #   4. model 轴与 hpo 轴各已有 ≥1 个评估 trial(模型/超参轴
        #      至少开过一次,不是没投过就宣布穷尽);
        #   5. (原为 ensemble-final 对照增益 < noise;v19 降级为 advisory,
        #      v2.16.0 随 ensemble-final 一并删除);
        #   6. 信息源清单无未消费项(information-inventory 的
        #      unconsumed=0;清单缺失视为不满足,不触发——清单先建)。
        # 满足时即便 data_views=1 也允许停止,confidence 降档并标注
        # single_view_caveat(天花板结论的适用范围仍如实呈现)。
        # v19(gen-05 实测回写):出口不再限定 single_view——穷尽证据
        # (delta 噪声带内 + 无 accepted + 清单清账)与视图数无关,
        # 多视图 run(gen-05 实测 4 个视图)同样卡死在该出口缺失上,
        # 停止依赖 LLM 自觉绕过字面条件。视图数只影响 confidence 档位。
        _exhausted_views_ok = True  # 任何视图数都允许走穷尽证据出口
        if _exhausted_views_ok and generation_best_deltas \
                and trials_accepted == 0:
            _ic_conditions = self._information_exhausted_conditions(
                generation_best_deltas, noise,
                inventory_unconsumed, model_axis_tried,
                candidate_value_upper_bound,
                fresh_confirmation_available)
            if all(_ic_conditions.values()):
                reasons.append("information_channels_exhausted")
        # legacy:不满足穷尽出口时的单口径护栏(维持 v12 行为)
        if len(generation_best_deltas) >= 3 and trials_accepted == 0 and \
                single_view and "information_channels_exhausted" \
                not in reasons:
            reasons.append("single_data_view_insufficient"
                           if "convergence_no_accepted" not in reasons else "")

        reasons = [r for r in reasons if r]
        # v2.17.0(D3):空代不得触发停止 —— 只保留与"本代有无证据"无关的
        # 硬预算(已启动 trial 数 / 墙钟 / compute);收敛、ROI、穷尽、单口径
        # 护栏、代数上限、性能目标都需要本代有新证据才能(重新)判定。
        _EMPTY_GEN_KEEP = ("budget_max_trials", "budget_wall_time",
                           "budget_compute")
        suppressed_by_empty: list[str] = []
        if empty_generation:
            suppressed_by_empty = [r for r in reasons
                                   if r not in _EMPTY_GEN_KEEP]
            reasons = [r for r in reasons if r in _EMPTY_GEN_KEEP]

        # ---- v18/P0-3:信息通道穷尽出口的分条件明细(审计链)----
        information_exhausted_detail: dict | None = None
        if "information_channels_exhausted" in reasons:
            information_exhausted_detail = \
                self._information_exhausted_conditions(
                    generation_best_deltas, noise,
                    inventory_unconsumed,
                    model_axis_tried, candidate_value_upper_bound,
                    fresh_confirmation_available)

        # ---- 分方向边际递减(v8,报告维度;不改触发语义)----
        per_direction_diminishing: dict[str, dict] = {}
        if direction_deltas and noise is not None and np.isfinite(noise) \
                and noise > 0:
            for action, deltas in direction_deltas.items():
                if len(deltas) < 3:
                    continue
                acc = (direction_accepted or {}).get(action, 0)
                recent = deltas[-3:]
                if acc == 0 and all(
                        (d < noise) if self.direction_better
                        else (d > -noise) for d in recent):
                    per_direction_diminishing[action] = {
                        "stale_generations": 3,
                        "last_delta": recent[-1],
                    }
        all_directions_diminishing = bool(
            per_direction_diminishing
            and direction_deltas
            and set(per_direction_diminishing) == set(direction_deltas))

        if not reasons:
            return None

        not_executed = []
        if any(r.startswith("budget") for r in reasons) and \
                unexplored_high_potential:
            not_executed = unexplored_high_potential
        # ---- confidence 实算(v12/P1-8:不再 budget 停止一律贴 medium)----
        # 高:性能目标达成;中高:统计收敛 + 全方向递减 + 无未探索方向;
        # 中:统计收敛但存在未探索方向;低:budget 触顶且仍有未探索方向
        # v18:information_channels_exhausted 出口 = medium 且 stop_report
        # 携带 single_view_caveat(单口径证据的适用范围如实呈现)
        if "performance_target_reached" in reasons:
            confidence = "high"
        elif "convergence_no_accepted" in reasons \
                and all_directions_diminishing \
                and not unexplored_high_potential:
            confidence = "medium_high"
        elif "convergence_no_accepted" in reasons:
            confidence = "medium"
        elif "information_channels_exhausted" in reasons:
            confidence = "medium"
        elif "ceiling_gap_within_mde" in reasons:
            confidence = "medium_high"
        elif any(r.startswith("budget") for r in reasons) \
                and unexplored_high_potential:
            confidence = "low"
        else:
            confidence = "medium"
        # 累计提升按指标方向呈现(§1.3)
        cumulative = None
        if best_metric is not None and baseline_metric is not None:
            cumulative = (best_metric - baseline_metric
                          if self.direction_better
                          else baseline_metric - best_metric)
        # ---- v2.17.0(#17):停止语义显式化 ----
        # remaining_high_potential_directions 在 confidence 计算之后才补
        # confirm_or_consolidate 条目:它是"收尾前该做的事",不改变停止
        # 触发与 confidence 档位(尺子不变)。
        remaining = list(unexplored_high_potential or [])
        ug = dict(undelivered_gain) if isinstance(undelivered_gain, dict) \
            else None
        if ug and ug.get("meets_mde") and ug.get("best_trial"):
            if ug.get("reference") == "anchor":
                _item = f"confirm_or_consolidate:{ug['best_trial']}"
                if _item not in remaining:
                    remaining.append(_item)
            else:
                ug.setdefault("action_hint", (
                    "本 run 为 parent 协议(promotion_reference=parent):逐步 "
                    "vs_parent 的小步链无法晋级进入确认链;要把这部分增益纳入"
                    "交付,需以 anchor 协议(v2.17.0 起新 init 的 run)重开,"
                    "或由用户裁决对该 trial 手工 confirm"))
        blocks = stop_blocks(reasons)
        return {
            "stop_reason": " + ".join(reasons),
            # v2.17.0(#17):STOPPED 是否拦截后续命令(据实核对代码)
            "advisory": not blocks,
            "blocks": blocks,
            "advisory_note": STOP_ADVISORY_NOTE,
            "reason_detail": {r: REASON_DETAIL.get(
                r, "(未登记的 reason code,见 runtime/generation.py)")
                for r in reasons},
            "undelivered_gain": ug,
            "checked_at": now_iso(),
            "n_trials_started": n_trials_started,
            "n_generations": n_generations,
            "wall_time_hours": round(wall_time_hours, 3),
            # v16/P0-2:双口径预算呈现(compute=trial 实际计算时长;
            # wall 含编排延迟/空闲/中断,不再是唯一判定)
            "compute_hours": (round(float(compute_hours), 3)
                              if compute_hours is not None else None),
            "compute_utilization": (
                round(float(compute_hours) / wall_time_hours, 4)
                if compute_hours is not None and wall_time_hours > 0
                else None),
            "best_metric": best_metric,
            "baseline_metric": baseline_metric,
            "cumulative_improvement_vs_baseline": cumulative,
            "recent_generations": {"best_delta": _deltas_raw[-5:]},
            # v2.17.0(D3):空代时被压制的停止原因(只读呈现,不触发)
            "empty_generation": bool(empty_generation),
            "suppressed_by_empty_generation": suppressed_by_empty,
            "remaining_high_potential_directions": remaining,
            "not_executed_reason": [r for r in reasons if r.startswith("budget")],
            "confidence_of_near_ceiling": confidence,
            "data_views_tried": views,
            "non_sample_data_views": non_sample_views,
            # v18/P0-3:信息通道穷尽出口的六条件明细与单口径 caveat
            "information_exhausted_detail": information_exhausted_detail,
            "single_view_caveat": (
                "information_channels_exhausted" in reasons
                and single_view),
            # v8 分方向维度(增强信号,不参与触发)
            "per_direction_diminishing": per_direction_diminishing,
            "all_directions_diminishing": all_directions_diminishing,
            # v2.19.0:天花板估计与剩余差距(无估计时为 None)
            "ceiling": ceiling,
        }

    def _information_exhausted_conditions(
        self,
        generation_best_deltas: list[float],
        noise: float | None,
        inventory_unconsumed: int | None,
        model_axis_tried: bool | None = None,
        candidate_value_upper_bound: float | None = None,
        fresh_confirmation_available: bool | None = None,
    ) -> dict[str, bool]:
        """v18/P0-3:信息通道穷尽出口的分条件明细(全部 True 才出口)。

        - no_accepted_last_gen:调用方保证(trials_accepted==0 才进入);
        - last_delta_within_noise:最近一代 best delta 在噪声带内;
        - model_axis_tried:model/hpo 轴开过(direction_deltas 的
          explore/exploit/hpo 键任一有逐代序列;由 check() 调用处传入
          的聚合判断,这里恒 True——入口已含该判断);
        - inventory_fully_consumed:信息源清单已建且 unconsumed=0
          (None = 清单缺失,不触发——清单先建再谈穷尽)。

        v19 把 ensemble-final 增益条件从 all() 硬门降级为 advisory
        (融合增益来自已有分数,不是新信息通道);v2.16.0 随
        ensemble-final 删除,advisory 字段一并移除。
        """
        last_delta = generation_best_deltas[-1] \
            if generation_best_deltas else None
        conditions: dict[str, bool] = {}
        conditions["no_accepted_last_gen"] = True
        # v19:改名 not_positive_gain——原口径 |delta|<noise 只覆盖
        # "点值收敛",漏掉"明确为负"(gen-05 实测对齐口径 -0.0014,
        # p≈0.000 = 有证据的退化,同样是"无增量"证据;但 2.5σ 在带外,
        # 旧判据 False → 出口永不触发)。方向感知:主指标越大越好时
        # delta<=0 即无增益(负向有证据比噪声内更强),|delta|<noise
        # 保留为噪声带情形。键名保留旧名以兼容既有消费方/测试。
        conditions["last_delta_within_noise"] = bool(
            last_delta is not None and noise is not None
            and np.isfinite(noise) and noise > 0
            and (abs(float(last_delta)) < float(noise)
                 or float(last_delta) <= 0))
        # v2.10.1/F5:原为硬编码 True,注释称"由 check() 调用处传入的
        # 聚合判断,这里恒 True——入口已含该判断",但调用方从未传入该
        # 判断(实测 0 个 model 轴 trial 的 run 仍报 true 并据此停止)。
        # 新口径:由 finalize 从 trial 的 changed_modules/search_action
        # 实测得出;取不到(None)时按 False 处理 —— 未知不等于已试过,
        # 宁可继续也不能以未验证的理由停工。
        conditions["model_axis_tried"] = bool(model_axis_tried)
        conditions["inventory_fully_consumed"] = bool(
            inventory_unconsumed is not None
            and int(inventory_unconsumed) == 0)
        mde = self._mde()
        # None 仅用于兼容直接调用旧 API；EvolutionEngine 总会传入规划器
        # 最新值。真实闭环里无规划结果不允许据此宣布接近天花板。
        conditions["candidate_value_upper_bound_below_mde"] = (
            True if candidate_value_upper_bound is None else
            bool(np.isfinite(candidate_value_upper_bound)
                 and float(candidate_value_upper_bound) < mde))
        conditions["fresh_confirmation_available"] = (
            True if fresh_confirmation_available is None else
            bool(fresh_confirmation_available))
        return conditions
