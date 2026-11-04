"""Generation 并行执行:有期限 barrier(§11.3 / §16.5.2)。

- 同一 Generation 的 N 个 Trial 并行拉起(线程池 + 独立进程组);
- v2.13:每个 Trial 的超时 = base × expected_resources.timeout_multiplier
  (封顶 evolution.hpo.max_timeout_minutes),代级 deadline 取
  max(per_trial_timeout),不再被 wall 余量截短;
- 到 deadline 仍在跑的 Trial 被 kill 并记 failed(timeout);
- fail-closed 预检:wall 余量不足(若调用方提供)时整代不启动;
- 反思不因单个 Trial 挂起而阻塞,且必须覆盖超时未验证方向;
- v2.17.0:线程池里的每个 Trial 在启动训练前过内存准入
  (runner._execute_once → scheduler.MemoryAdmission,跨线程/跨进程 flock
  串行判定),不满足就排队,逐个启动而非一次全起;每 trial 的排队时长与
  本代 admission_summary 写进代摘要(admission_wait_sec / admission)。
- v2.17.2(B2):已有活 runner(``.runner.lock`` 持有者存活,如 ``run
  --detach`` 正在跑)的 trial 一律跳过、**不改写**其 status/status_history,
  列入摘要 ``already_running``;barrier 兜底只修正本执行器实际启动的 trial。
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import time
from pathlib import Path
from typing import Any

from .daemon import runner_lock_alive
from .logging_utils import get_logger, now_iso
from .runner import ConcurrentRunError, TrialRunner, load_parent_scores
from .scheduler import ResourcePlan
from .trial import TrialRecord

logger = get_logger(__name__)


def _best_parent_scores(run_dir: Path, spec, parents: list[str]):
    """多父代时取指标最优父代的分数作对比基准;单父代直取。"""
    from .data import load_split_frames
    from .metrics import compute_metric, metric_direction

    if not parents:
        return None
    if len(parents) == 1:
        return load_parent_scores(run_dir, parents[0])
    import pandas as pd

    from .data import view_splits_dir

    frames = load_split_frames(run_dir)
    valid = frames.get("validation")
    # v21 四平面:标签在 splits/labels/(循环外只 join 一次——旧实现
    # 把 join 写在 for pid 循环体内,每父代重复 join 累积 y_lab/t_lab
    # 重名列;且 view_splits_dir 返回的已是绝对路径,run_dir / 前缀
    # 在相对 --run 时会拼出错误路径使 join 被静默跳过)
    labels_dir = view_splits_dir(run_dir, "v1") / "labels"
    if valid is not None:
        if not (labels_dir / "validation.parquet").exists():
            raise FileNotFoundError(
                f"标签面缺失: {labels_dir}/validation.parquet(先跑 profile)")
        _va_lab = pd.read_parquet(labels_dir / "validation.parquet")
        valid = valid.join(_va_lab.set_index(valid.index), how="left",
                           rsuffix="_lab")
    best_pid, best_val = None, None
    for pid in parents:
        pack = load_parent_scores(run_dir, pid)
        if not pack or valid is None:
            continue
        s = pack["validation"]["score"].values
        y = valid[spec.target_name].values
        # v12(P1-11,O4 修复):uplift 的 compute_metric 必须传
        # treatment,漏传时 ValueError 被吞 → best_pid 恒 None →
        # runner 视作根 trial → evaluator 直接 accepted(uplift
        # crossover 全部被误判)
        tcol = spec.treatment_column
        t = valid[tcol].values if (tcol and tcol in valid.columns) else None
        try:
            val = compute_metric(spec.primary_metric, y, s, score=s,
                                 treatment=t)
        except Exception:  # noqa: BLE001
            continue
        if best_val is None or (val > best_val if metric_direction(
                spec.primary_metric) else val < best_val):
            best_pid, best_val = pid, val
    if best_pid is None:
        return None
    return load_parent_scores(run_dir, best_pid)


class GenerationExecutor:
    """并行执行一批 Trial 并施加有期限 barrier。"""

    def __init__(self, runner: TrialRunner):
        self.runner = runner
        # v2.17.2(B2):本执行器实际启动(交给 runner.run)的 trial,与因已有
        # 活 runner 而跳过的 trial(只读、绝不 flush)
        self._started: set[str] = set()
        self._already_running: dict[str, int] = {}

    @staticmethod
    def _foreign_lock_pid(trial) -> int | None:
        """trial 目录 .runner.lock 的**别的进程**活持有者 pid;无则 None。

        本进程自己的锁(线程池内 runner.run 持有)不算——那是本执行器
        启动的 trial。"""
        tdir = getattr(trial, "trial_dir", None)
        if not tdir:
            return None
        try:
            pid = runner_lock_alive(Path(tdir))
        except Exception:  # noqa: BLE001  探针失败按无锁处理(旧行为)
            return None
        return pid if pid is not None and pid != os.getpid() else None

    def _skip_already_running(self, trial, pid: int, where: str) -> None:
        """记录跳过(只读):不改 status/status_history、不 flush。"""
        self._already_running[trial.trial_id] = int(pid)
        self._started.discard(trial.trial_id)
        logger.warning(
            "Trial %s 已有活 runner(pid=%s,%s):本批跳过,不改写其状态"
            "(wait --trial 等它自己的终态)", trial.trial_id, pid, where)

    def _exclude_already_running(self, trials: list[TrialRecord]
                                 ) -> list[TrialRecord]:
        keep = []
        for t in trials:
            pid = self._foreign_lock_pid(t)
            if pid is not None:
                self._skip_already_running(t, pid, "启动前探测")
            else:
                keep.append(t)
        return keep

    def execute_generation(
        self,
        trials: list[TrialRecord],
        *,
        plan: ResourcePlan,
        wall_time_budget_hours: float | None = None,
        run_dir: Path | None = None,
        baseline_id: str | None = None,
        wall_remaining_sec: float | None = None,
    ) -> dict:
        """并行执行 trials;返回执行摘要(deadline、结果清单)。

        - 并发上限取 plan.n_parallel_trials;
        - v2.13 [FIX-2026-09-19]:每 Trial 超时 =
          base × expected_resources.timeout_multiplier(缺省 1),
          封顶 evolution.hpo.max_timeout_minutes×60;代级 deadline =
          now + max(per_trial_timeout),不再被 wall 余量截短(旧版
          barrier 只剩 214s 而单 trial 需 250~400s,4/6 被杀);
        - wall_remaining_sec(可选,None=不检查):fail-closed 预检,
          小于最小 trial timeout 时整代不启动,返回结构化拒绝结果;
        - 到 barrier 仍在 RUNNING 的 Trial 由其自身超时机制 kill
          (每个子进程组独立计时,barrier 到点统一兜底终止)。
        """
        if not trials:
            return {"executed": [], "generation_deadline": None}
        # v2.17.2(B2):有别的活 runner 的 trial 不进批(只读跳过)
        self._started = set()
        self._already_running = {}
        trials = self._exclude_already_running(list(trials))
        if not trials:
            return {"executed": [], "generation_deadline": None,
                    "n_succeeded": 0, "n_failed": 0,
                    "already_running": sorted(self._already_running)}

        started = time.time()
        per_trial_timeout_sec = self._per_trial_timeout_sec(trials)
        min_trial_timeout_sec = min(per_trial_timeout_sec.values())
        deadline = started + max(per_trial_timeout_sec.values())
        # v2.13:wall 余量 fail-closed 预检(未提供则不检查,保持旧行为)
        if wall_remaining_sec is not None \
                and float(wall_remaining_sec) < min_trial_timeout_sec:
            logger.error(
                "wall 余量 %.0fs < 最小单 trial 超时 %.0fs:整代拒绝启动"
                "(比启动后再杀便宜得多;提高 resource_budget."
                "max_wall_time_hours 或减少本代 trial)",
                float(wall_remaining_sec), min_trial_timeout_sec)
            return {
                "generation_not_started": True,
                "reason": "insufficient_wall_budget",
                "wall_remaining_sec": float(wall_remaining_sec),
                "min_trial_timeout_sec": float(min_trial_timeout_sec),
                "remedy": ("提高 resource_budget.max_wall_time_hours "
                           "或减少本代 trial / 缩小 hpo budget_configs "
                           "后重试 run-generation"),
                "per_trial_timeout_sec": per_trial_timeout_sec,
                "generation_deadline": now_iso(),
            }
        # 旧 wall 预算检查(wall_time_budget_hours)保留:预算耗尽时
        # 逐 trial 记 timeout(不动 per-trial 口径,只作绝对护栏)
        if wall_time_budget_hours is not None:
            wall_deadline_sec = wall_time_budget_hours * 3600
            run_started = self._run_started_at(run_dir) if run_dir else started
            remaining = run_started + wall_deadline_sec - started
            if remaining <= 0:
                logger.error("wall-time 预算已耗尽,Generation 拒绝启动")
                for t in trials:
                    self._mark_timeout_without_run(t)
                return {"executed": [t.trial_id for t in trials],
                        "generation_deadline": now_iso(),
                        "wall_time_exhausted": True}
            # v2.13:deadline 不再取 wall 截短(那正是单 trial 250~400s
            # 被 214s barrier 杀掉的根因);wall 只走上面的预检口径
        logger.info("Generation barrier: %d 个 Trial,并发 %d,deadline %.0f 秒",
                    len(trials), plan.n_parallel_trials, deadline - started)

        run_dir = run_dir or self.runner.run_dir
        racing = self._plan_racing(trials)
        if racing is None:
            results = self._run_batch(trials, plan=plan, deadline=deadline,
                                      run_dir=run_dir,
                                      baseline_id=baseline_id,
                                      per_trial_timeout_sec=per_trial_timeout_sec)
            summary = self._summarize(results, racing=None)
            self._attach_admission(summary, run_dir, trials)
            self._write_deadline_artifact(
                run_dir, trials, per_trial_timeout_sec, deadline)
            self._generation_breaker(summary, run_dir)
            return summary

        # ---- successive halving(v8,默认关)----
        # 轮 1:全量 trial 以子样本/短预算跑 quick 评,排序淘汰下半区;
        # 轮 2:胜者回退 CODE_WRITTEN 后全量重跑(deadline 仍为原 barrier)。
        ratio, frac, survive = racing
        quick_deadline = started + frac * (deadline - started)
        logger.info("successive halving: %d trials quick_ratio=%.2f "
                    "budget_frac=%.2f survive=%.2f", len(trials), ratio,
                    frac, survive)
        quick_results = self._run_batch(
            trials, plan=plan, deadline=quick_deadline, run_dir=run_dir,
            baseline_id=baseline_id, quick_ratio=ratio,
            per_trial_timeout_sec=per_trial_timeout_sec)
        order = self._halving_rank(quick_results)
        n_keep = max(1, int(len(order) * survive + 0.999))
        winners, eliminated = set(order[:n_keep]), set(order[n_keep:])
        for tid, t in quick_results.items():
            t.data.setdefault("evaluation", {})
            if tid in eliminated:
                t.data["evaluation"]["racing"] = {
                    "quick_only": True, "quick_ratio": ratio}
                t.flush()
        promoted = [quick_results[tid] for tid in order[:n_keep]]
        for t in promoted:
            # 全量重跑:回退到 CODE_WRITTEN,runner.run(force_rerun=True)
            t.data["status"] = "CODE_WRITTEN"
            t.data["status_history"].append(
                {"state": "CODE_WRITTEN", "at": now_iso(),
                 "note": "halving promote: quick → full"})
            t.data["evaluation"]["racing"] = {
                "promoted": True, "quick_ratio": ratio}
            t.flush()
        full_results = self._run_batch(
            promoted, plan=plan, deadline=deadline, run_dir=run_dir,
            baseline_id=baseline_id,
            per_trial_timeout_sec=per_trial_timeout_sec)
        for tid in eliminated:
            full_results[tid] = quick_results[tid]
        # 全量轮的 evaluation 覆盖会冲掉 racing 标注;重新写回(败者保留
        # quick 轮结果,胜者标注 promoted)
        for tid in winners:
            t = full_results.get(tid)
            if t is not None:
                t.data.setdefault("evaluation", {})
                t.data["evaluation"]["racing"] = {
                    "promoted": True, "quick_ratio": ratio}
                t.flush()
        summary = self._summarize(
            full_results,
            racing={"enabled": True, "quick_ratio": ratio,
                    "eliminated": sorted(eliminated),
                    "promoted": sorted(winners)})
        self._attach_admission(summary, run_dir, trials)
        self._write_deadline_artifact(
            run_dir, trials, per_trial_timeout_sec, deadline)
        self._generation_breaker(summary, run_dir)
        return summary

    def _attach_admission(self, summary: dict, run_dir: Path | None,
                          trials: list[TrialRecord]) -> None:
        """v2.17.0:把内存准入排队写进代摘要。

        - admission_wait_sec:{trial_id: 本次排队秒数}(trial.yaml.admission);
        - n_admission_timeout:准入超时(训练未启动)的 trial 数;
        - admission:scheduler.admission_summary(run_dir, 本代) ——
          throttled=True 即"资源受限串行",finalize 的 avg_parallelism 告警
          据此改写归因。
        """
        try:
            from .scheduler import ADMISSION_TIMEOUT, admission_summary

            waits = {}
            for t in trials:
                adm = t.data.get("admission") or {}
                if "wait_sec" in adm:
                    waits[t.trial_id] = adm.get("wait_sec")
            summary["admission_wait_sec"] = waits
            summary["n_admission_timeout"] = sum(
                1 for t in trials
                if t.data.get("failure_reason") == ADMISSION_TIMEOUT)
            gids = sorted({t.generation for t in trials})
            summary["admission"] = admission_summary(
                Path(run_dir or self.runner.run_dir),
                gids[-1] if len(gids) == 1 else None)
        except Exception as exc:  # noqa: BLE001  摘要是审计用,不阻塞执行
            logger.warning("准入摘要生成失败(忽略): %s", exc)

    def _generation_breaker(self, summary: dict, run_dir: Path | None) -> None:
        """v2.15.8:代级失败率熔断(generation_failure_rate,默认 0.5)。

        本代失败数/总数超阈值 → 打开熔断,下一代 run 前须 root-cause +
        ``run --reset-breaker``。只对 init 时记了 skill_version≥2.15.8 的
        run 生效;旧 run 只告警(中途接入会让在跑 run 在下一代被拦)。
        判定结果写进 summary["breaker"]。

        v2.17.0:内存准入超时(训练从未启动)不计入失败率——资源争用期
        熔断 run 只会让排队更久,root-cause 也无从写起。
        """
        _n_adm = int(summary.get("n_admission_timeout") or 0)
        n_failed = max(0, int(summary.get("n_failed") or 0) - _n_adm)
        n_total = n_failed + int(summary.get("n_succeeded") or 0)
        if n_total < 2 or n_failed == 0:
            return
        try:
            from . import version_at_least
            from .logging_utils import load_json

            _rs_path = Path(run_dir or self.runner.run_dir) / "run-state.json"
            _rs = load_json(_rs_path) if _rs_path.exists() else {}
            enforce = version_at_least(_rs.get("skill_version"), "2.15.8")
        except Exception:  # noqa: BLE001
            enforce = False
        rate = n_failed / n_total
        if rate <= self.runner.breaker.rate_limit:
            return
        if enforce:
            trip = self.runner._check_breaker(n_failed, n_total)
            summary["breaker"] = {"tripped": bool(trip), **(trip or {})}
        else:
            logger.warning(
                "本代失败率 %.0f%%(%d/%d) 超过熔断阈值 %.0f%%:旧 run 只告警"
                "不熔断(v2.15.8 起新 run 会打开熔断),请先 diagnose",
                rate * 100, n_failed, n_total,
                self.runner.breaker.rate_limit * 100)
            summary["breaker"] = {"tripped": False, "warn_only": True,
                                  "rate": rate}

    # ------------------------------------------------------------------
    def _per_trial_timeout_sec(self, trials: list[TrialRecord]) -> dict:
        """v2.13:每 trial 超时秒数 = base × timeout_multiplier,封顶。

        expected_resources.timeout_multiplier 来自 trial.yaml(heavy/hpo
        由创建口/渲染口打标,默认 1);封顶 evolution.hpo.
        max_timeout_minutes(默认 480)×60,与 cmd_trial 单 trial 路径
        (heavy/hpo 4× 放大)同口径——旧批路径只取 base 超时,放大
        从不生效,heavy trial 全部按 base 撞墙。
        """
        base_min = self.runner.resolve_timeout_minutes()
        try:
            cap_sec = float(self.runner.config.get(
                "evolution.hpo.max_timeout_minutes", 480)) * 60.0
        except Exception:  # noqa: BLE001
            cap_sec = 480 * 60.0
        out: dict[str, float] = {}
        for t in trials:
            er = t.data.get("expected_resources") or {}
            try:
                mult = float(er.get("timeout_multiplier") or 1.0)
            except (TypeError, ValueError):
                mult = 1.0
            if not mult or mult < 0 or mult != mult:  # 0/NaN 兜底
                mult = 1.0
            out[t.trial_id] = min(base_min * 60.0 * mult, cap_sec)
        return out

    def _write_deadline_artifact(self, run_dir: Path,
                                 trials: list[TrialRecord],
                                 per_trial_timeout_sec: dict,
                                 deadline: float) -> None:
        """把 per-trial 超时与代级 deadline 落进本代 generations/ 产物。

        写入 gen-{gid}/executor-deadline.json(generation 目录由 trial
        代号推导;推导失败时静默跳过——产物是审计用,不阻塞执行)。
        """
        try:
            import datetime as _dt

            gids = sorted({t.generation for t in trials})
            gid = gids[-1] if gids else 0
            from .logging_utils import dump_json

            out_dir = Path(run_dir) / "generations" / f"gen-{gid:02d}"
            dump_json(out_dir / "executor-deadline.json", {
                "written_at": now_iso(),
                "generation": gid,
                "deadline_epoch": round(float(deadline), 1),
                "deadline_iso": _dt.datetime.fromtimestamp(
                    deadline).astimezone().isoformat(),
                "base_timeout_sec": round(
                    self.runner.resolve_timeout_minutes() * 60.0, 1),
                "per_trial_timeout_sec": {
                    k: round(float(v), 1)
                    for k, v in per_trial_timeout_sec.items()},
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning("executor-deadline 产物写入失败(忽略): %s", exc)

    # ------------------------------------------------------------------
    def _plan_racing(self, trials: list[TrialRecord]) -> tuple | None:
        """读取 halving 配置;不满足启用条件返回 None(走单轮路径)。"""
        cfg = self.runner.config
        if not bool(cfg.get("gen_executor.halving_enabled", False)):
            return None
        min_trials = int(cfg.get("gen_executor.halving_min_trials", 4))
        if len(trials) < min_trials:
            return None
        ratio = float(cfg.get("gen_executor.halving_ratio", 0.5))
        frac = float(cfg.get("gen_executor.halving_budget_fraction", 0.3))
        survive = float(cfg.get("gen_executor.halving_survive_fraction", 0.5))
        if not (0 < ratio < 1 and 0 < frac < 1 and 0 < survive < 1):
            logger.warning("halving 配置非法,忽略: ratio=%s frac=%s "
                           "survive=%s", ratio, frac, survive)
            return None
        return (ratio, frac, survive)

    def _apply_quick_env(self, plan: ResourcePlan,
                         ratio: float) -> ResourcePlan:
        """向 plan.env 注入 MODELEVO_QUICK_RATIO(trial 模板读取子采样)。

        旧 trial 代码不识别该变量 → quick 轮等价全量(降级语义,排序仍按
        validation primary_metric,正确性不受影响)。
        """
        import copy

        plan = copy.copy(plan)
        plan.env = dict(plan.env or {})
        plan.env["MODELEVO_QUICK_RATIO"] = str(ratio)
        return plan

    def _halving_rank(self, results: dict[str, TrialRecord]) -> list[str]:
        """按 validation primary_metric 方向排序(优者在前)。"""
        from .metrics import metric_direction

        better_dir = metric_direction(self.runner.spec.primary_metric)

        def _val(t: TrialRecord):
            return ((t.data.get("evaluation") or {})
                    .get("primary_metric") or {}).get("value")

        # None 排最后;越大越好 → key 取负(升序 = 值降序),反之直接升序
        scored = [(_val(t), tid) for tid, t in results.items()]
        scored.sort(key=lambda p: (
            p[0] is None,
            (-(p[0] if p[0] is not None else 0)) if better_dir
            else (p[0] if p[0] is not None else 0),
            p[1]))
        return [tid for _, tid in scored]

    def _run_batch(self, trials: list[TrialRecord], *, plan: ResourcePlan,
                   deadline: float, run_dir: Path,
                   baseline_id: str | None,
                   quick_ratio: float | None = None,
                   per_trial_timeout_sec: dict | None = None,
                   ) -> dict[str, TrialRecord]:
        results: dict[str, TrialRecord] = {}
        if quick_ratio is not None:
            plan = self._apply_quick_env(plan, quick_ratio)
        if per_trial_timeout_sec is None:
            per_trial_timeout_sec = self._per_trial_timeout_sec(trials)

        def _run_one(trial: TrialRecord) -> TrialRecord:
            parents = trial.parent_ids
            parent_scores = None
            if parents:
                # 多父代(crossover)取指标最优父代作对比基准(§12.2)
                parent_scores = _best_parent_scores(
                    run_dir, self.runner.spec, parents)
            baseline_score = None
            baseline_view = None
            if baseline_id:
                bl = load_parent_scores(run_dir, baseline_id)
                baseline_score = bl.get("validation") if bl else None
                # v2.13.2:视图随分数一起传(load_parent_scores 已返回
                # data_view)。跨视图 baseline 与 sample 面 trial 行集
                # 不同,runner 据此跳过对齐并记 baseline_incomparable,
                # 而不是裁 invalid(实测事故:gen-01 整批 sample trial
                # 报废)。视图不能靠调用方另行传参——漏传即静默退回
                # 旧行为,正是本事故的成因。
                baseline_view = bl.get("data_view") if bl else None
            spec = self.runner.spec
            slice_cols = [c for c in (spec.time_column, spec.id_column) if c]
            # v2.13:每 Trial 用自己的超时(now + per_trial_timeout),
            # 不再统一吃 batch barrier 的剩余量(旧版同 barrier 截断使
            # heavy trial 的 4× 放大完全失效);仍封顶到当前批次
            # barrier(quick 轮等场景 barrier 本身更短)
            _own = per_trial_timeout_sec.get(
                trial.trial_id, self.runner.resolve_timeout_minutes() * 60)
            remain_min = min(_own, deadline - time.time()) / 60.0
            remain_min = max(remain_min, 0.05)
            # v18/P1-3:批内槽位序号——多卡分配按它把 trial 绑到
            # plan.gpu_indices 的不同卡上(空闲显存降序)
            # v2.17.2(B2):启动前复查活锁(批前探测与此刻之间可能有
            # `run --detach` 抢先拉起);命中即跳过,不碰 trial 记录
            _pid = self._foreign_lock_pid(trial)
            if _pid is not None:
                raise ConcurrentRunError(
                    trial.trial_id, _pid,
                    Path(trial.trial_dir) / ".runner.lock")
            self._started.add(trial.trial_id)
            return self.runner.run(
                trial, plan=plan, parent_scores=parent_scores,
                baseline_score=baseline_score, baseline_view=baseline_view,
                slice_columns=slice_cols,
                timeout_minutes=remain_min,
                force_rerun=bool(
                    quick_ratio is None
                    and (trial.data.get("evaluation") or {})
                    .get("racing", {}).get("promoted")),
                slot_index=trials.index(trial))

        with cf.ThreadPoolExecutor(
                max_workers=max(plan.n_parallel_trials, 1)) as pool:
            futs = {pool.submit(_run_one, t): t for t in trials}
            for fut in cf.as_completed(futs):
                trial = futs[fut]
                try:
                    results[trial.trial_id] = fut.result()
                except ConcurrentRunError as exc:
                    # v2.17.2(B2):并发 run 拒绝 = 跳过,不改状态/历史
                    self._skip_already_running(trial, exc.pid, "runner 拒绝")
                except Exception as exc:  # noqa: BLE001
                    _pid = self._foreign_lock_pid(trial)
                    if _pid is not None:
                        # 异常期间别的 runner 已接管该 trial:手里的记录是
                        # 旧快照,flush 会覆写它 → 同样只读跳过
                        logger.error("Trial %s 执行异常: %s", trial.trial_id,
                                     exc)
                        self._skip_already_running(trial, _pid, "异常后探测")
                        continue
                    logger.error("Trial %s 执行异常: %s", trial.trial_id, exc)
                    trial.data["failure_reason"] = "infra_error"
                    trial.data["status"] = "FAILED"
                    trial.data["status_history"].append(
                        {"state": "FAILED", "at": now_iso()})
                    trial.flush()
                    results[trial.trial_id] = trial
        return results

    def _summarize(self, results: dict[str, TrialRecord],
                   racing: dict | None) -> dict:
        # barrier 兜底:仍在 RUNNING 的 Trial 记 timeout(子进程已由其
        # 自身 timeout kill;这里只修正状态)
        # v2.17.2(B2):只修正本执行器实际启动、且没有别的活 runner 的
        # trial——别人的在跑 trial 不是本 barrier 的超时
        for t in results.values():
            if t.data.get("status") == "RUNNING":
                if t.trial_id not in self._started:
                    continue
                _pid = self._foreign_lock_pid(t)
                if _pid is not None:
                    self._skip_already_running(t, _pid, "barrier 兜底")
                    continue
                t.data["status"] = "FAILED"
                t.data["failure_reason"] = "timeout"
                t.data["status_history"].append(
                    {"state": "FAILED", "at": now_iso()})
                t.flush()
        # v2.16.0:被拓扑门拦下的 trial 未执行(状态不变),不计成功也不计失败
        # v2.17.2(B2):已有活 runner 而跳过的 trial 同样不计
        _skipped = set(self._already_running)
        _blocked = [t.trial_id for t in results.values()
                    if (t.data.get("model_topology") or {}).get("decision")
                    == "blocked" and t.status in ("PROPOSED", "CODE_WRITTEN")]
        _counted = [t for t in results.values()
                    if t.trial_id not in _skipped]
        out = {
            "executed": [tid for tid in results
                         if tid not in _blocked and tid not in _skipped],
            "generation_deadline": now_iso(),
            "n_succeeded": sum(1 for t in _counted
                               if t.status not in ("FAILED",)
                               and t.trial_id not in _blocked),
            "n_failed": sum(1 for t in _counted
                            if t.status == "FAILED"),
            "timeout_trials": [t.trial_id for t in _counted
                               if t.data.get("failure_reason") == "timeout"],
        }
        if _blocked:
            out["topology_blocked"] = _blocked
        if _skipped:
            out["already_running"] = sorted(_skipped)
            out["already_running_pids"] = dict(
                sorted(self._already_running.items()))
        if racing is not None:
            out["racing"] = racing
        return out

    def _mark_timeout_without_run(self, trial: TrialRecord) -> None:
        trial.data["status"] = "FAILED"
        trial.data["failure_reason"] = "timeout"
        trial.data["status_history"].append({"state": "FAILED",
                                             "at": now_iso()})
        trial.flush()

    def _run_started_at(self, run_dir: Path | None) -> float:
        if run_dir is None:
            return time.time()
        st = run_dir / "run-state.json"
        if st.exists():
            import json

            try:
                return float(json.loads(st.read_text(encoding="utf-8"))
                             .get("started_at_epoch", time.time()))
            except Exception:  # noqa: BLE001
                return time.time()
        return time.time()
