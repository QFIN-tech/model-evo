"""Trial 执行编排:把 Scheduler/Evaluator/TrialRecorder 串成标准流程。

单入口 run_trial():
  PROPOSED → (代码落盘校验) → CODE_WRITTEN → RUNNING(进程组+deadline)
  → SUCCEEDED/FAILED → 读取 metrics/score → EVALUATED → REFLECTED
代码先落盘再执行是强约束(§10.3)。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .confirmation import (CONFIRM_CRITERION, confirm_config,
                           confirmation_margins, judge_delta,
                           legacy_two_sided, sigma_for_split)
from .data import (SAMPLE_KEY_COLUMN, load_split_frames, sample_keys,
                   view_splits_dir)
from .evaluator import Evaluator, slice_by_columns
from .gpu_capability import trial_can_use_gpu
from .integrity import (bind_prediction_artifact, freeze_trial_source,
                        sha256_file, verify_prediction_artifact,
                        verify_confirmation_seed, verify_trial_source)
from .logging_utils import (_CNFormatter, get_logger, load_json, now_iso,
                            dump_json, locked_state_write)
from .metrics import metric_direction
from .scheduler import (ADMISSION_TIMEOUT, AttemptPolicy, ExecutionResult,
                        FailureCircuitBreaker, GpuLease, MemoryAdmission,
                        ResourcePlan, RETRYABLE_FAILURES, Scheduler,
                        annotate_gpu_occupancy, append_admission_log,
                        classify_failure, estimate_trial_peak_gb,
                        gpu_candidates, gpu_candidates_by_occupancy,
                        query_gpu_occupancy, record_attempt,
                        run_trial_process)
from .spec import TaskSpec
from .trial import (TrialRecord, TrialFactory,
                    confirmation_parent_eligibility)

logger = get_logger(__name__)


class ConcurrentRunError(RuntimeError):
    """v2.17.2(B2):该 trial 已有别的活 runner(``.runner.lock`` 持有者存活)。

    调用方(run-generation 批路径)据此**跳过**该 trial 且不改写其
    status/status_history —— 旧实现把它当普通异常吞掉并标 FAILED(infra_error),
    覆写了 ``run --detach`` 正在跑的 trial(run4 opus5.5 实测)。
    继承 RuntimeError:旧调用方按 RuntimeError 捕获的语义不变。
    """

    def __init__(self, trial_id: str, pid: int, lock_path: Path):
        self.trial_id = trial_id
        self.pid = int(pid)
        self.lock_path = Path(lock_path)
        super().__init__(
            f"并发 run:该 trial 已有活 runner(pid={pid}, {lock_path});"
            "确认其退出后再重入")


class TrialRunner:
    """确定性 Trial 执行器;LLM Agent 只负责写 code/ 与 hypothesis。"""

    def __init__(self, run_dir: Path, spec: TaskSpec, config: Config):
        self.run_dir = Path(run_dir)
        self.spec = spec
        self.config = config
        self.evaluator = Evaluator(spec, config)
        self.scheduler = Scheduler(config)
        self.attempt_policy = AttemptPolicy.from_config(config)
        self.breaker = FailureCircuitBreaker(config)
        # 熔断状态(§16.5.6):打开后拒绝执行新 Trial,直到 root-cause
        # reflection 完成并显式 reset
        self.breaker_open = False
        self.breaker_reason: str | None = None
        self._load_breaker_state()

    def _load_breaker_state(self) -> None:
        state_path = self.run_dir / "breaker-state.json"
        if state_path.exists():
            state = load_json(state_path)
            self.breaker_open = bool(state.get("open"))
            self.breaker_reason = state.get("reason")

    def _persist_breaker_state(self) -> None:
        # 纯覆盖写,无读-改-写;_atomic_write 的唯一 tmp 名已保证并发安全
        dump_json(self.run_dir / "breaker-state.json", {
            "open": self.breaker_open,
            "reason": self.breaker_reason,
            "updated_at": now_iso(),
        })

    # ------------------------------------------------------------------
    def resolve_timeout_minutes(self, override: float | None = None) -> float:
        """超时优先级:显式覆写 > TaskSpec.resource_budget > 配置(§6.1)。"""
        if override is not None:
            return float(override)
        spec_timeout = self.spec.resource_budget.get("trial_timeout_minutes")
        if spec_timeout:
            return float(spec_timeout)
        return float(self.config.get("execution.trial_timeout_minutes", 60))

    def _count_running_trials(self) -> tuple[int, int]:
        """统计 run 目录下 status==RUNNING 的 Trial 数(v14.3;
        v19.4 返回 (数量, 本进程槽位号))。

        供 `run()` 在 plan 未显式传入时收敛并发:手动并行多个
        `run --trial`(不经 run-generation)时,每个 runner 进程启动时
        都看到 RUNNING 兄弟 + 自身 → planned_trials 反映真实并发,
        避免旧 8 路保守默认闲置 CPU、也避免各按独占 16 线程超卖。
        批路径(run-generation/baseline)显式传 plan,不走这里。
        槽位号(v19.4)= 进入前已 RUNNING 的兄弟数,用于手动并行
        路径把并发 trial 错开到不同 GPU。
        """
        n = 0
        try:
            for tdir in (self.run_dir / "trials").glob("*"):
                if not (tdir / "trial.yaml").exists():
                    continue
                try:
                    t = TrialRecord.load(tdir)
                    if t.status == "RUNNING":
                        n += 1
                except Exception:  # noqa: BLE001  单个损坏记录不阻塞
                    continue
        except Exception:  # noqa: BLE001  目录不可读时保守返回 1
            return 1, 0
        return n, n

    # ------------------------------------------------------------------
    # v2.8.1 孤儿 RUNNING 自愈(skill-fix-20260910):runner 父进程被
    # 外部 kill 后 trial 卡 RUNNING,重入 run() 时 RUNNING→RUNNING
    # 非法迁移直接报错,唯一出路是手工回退全量重训(实测 3 次 trial
    # 浪费 19~35 分钟训练)。per-trial 锁 + 孤儿判定闭环:
    # 锁存在且 pid 活 = 并发 run,拒绝;锁不存在或 pid 已死 = 孤儿
    # (旧版本运行留下的 RUNNING 无锁文件,天然按孤儿处理)。
    _RUNNER_LOCK = ".runner.lock"
    #: v2.12.2.8:评估段 in-flight 计数(识别信号级崩溃导致的无限重入)
    _EVAL_INFLIGHT = ".eval-inflight"

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:  # pid 存在但不属本用户
            return True
        except OSError:
            return False

    def _check_runner_lock(self, tdir: Path) -> int | None:
        """返回活 runner 的 pid;无锁/锁进程已死(孤儿)返回 None,
        锁被别的活进程持有(并发 run)返回该 pid。"""
        lock = Path(tdir) / self._RUNNER_LOCK
        if not lock.exists():
            return None
        try:
            pid = int((lock.read_text(encoding="utf-8")
                       or "0").strip() or 0)
        except (OSError, ValueError):
            return None  # 锁损坏视同孤儿:保守放行自愈
        return pid if pid and self._pid_alive(pid) else None

    def _acquire_runner_lock(self, tdir: Path) -> Path:
        lock = Path(tdir) / self._RUNNER_LOCK
        lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
        return lock

    @staticmethod
    def _release_runner_lock(lock: Path) -> None:
        try:
            Path(lock).unlink(missing_ok=True)
        except OSError:
            pass

    def _reenter_failed_trial(self, trial: TrialRecord) -> str:
        """FAILED → CODE_WRITTEN 重入:按原失败分类记 attempt(§16.5.4)。

        v2.16.2(run4 regression D-006):上一轮的 result / failure_reason
        归档进本次 attempt 记录(previous_result)后从 trial 顶层清除。旧实现
        原样保留 → 重入后 `wait --trial` 读到上一轮 result(failed、rc=-9),
        SUMMARY 把 failure=code_error 当作当前 attempt 的结果。返回原失败原因。
        """
        prev_reason = trial.data.get("failure_reason") or "code_error"
        action = self.attempt_policy.next_action(trial, prev_reason)
        _prev_result = trial.data.pop("result", None)
        trial.data.pop("failure_reason", None)
        # v2.17.0:准入超时(训练未启动)重入记 requeue,不消耗 retry/fix 预算
        # v2.17.2(B1):infra 类失败(oom/infra_error/timeout)retry 预算用尽后
        # 的重入同样记 requeue——代码未变、失败不是代码的错,不得消耗
        # fix_code 次数(run4 trial-006:OOM 挂死被 SIGTERM → 记了 fix_code)
        from .scheduler import is_infra_failure

        if action in ("retry", "requeue"):
            _kind = action
        else:
            _kind = "requeue" if is_infra_failure(prev_reason) \
                else "fix_code"
        record_attempt(trial, _kind,
                       f"FAILED Trial 重跑(原失败: {prev_reason})",
                       {"failure_reason": prev_reason,
                        **({"previous_result": _prev_result}
                           if _prev_result else {})})
        trial.data["status"] = "CODE_WRITTEN"
        trial.data["status_history"].append(
            {"state": "CODE_WRITTEN", "at": now_iso()})
        return prev_reason

    def _eval_crash_count(self, tdir: Path) -> int:
        """已完成训练的 trial 连续"评估未产出结果"的次数。

        用于识别**信号级崩溃**(段错误/OOM-kill)——这类失败绕过
        `except Exception`,进程直接死,锁成孤儿而分数已落盘,是
        "evaluate_only 无限重入"的判定依据。正常完成一次评估会清零。
        """
        p = Path(tdir) / self._EVAL_INFLIGHT
        try:
            return int((p.read_text(encoding="utf-8") or "0").strip() or 0)
        except (OSError, ValueError):
            return 0

    def _mark_eval_inflight(self, tdir: Path) -> None:
        """评估开始:计数 +1(进程若被信号杀死,这次 +1 会留下来)。"""
        p = Path(tdir) / self._EVAL_INFLIGHT
        try:
            p.write_text(f"{self._eval_crash_count(tdir) + 1}\n",
                         encoding="utf-8")
        except OSError:
            pass

    def _clear_eval_inflight(self, tdir: Path) -> None:
        """评估产出结果:清零。"""
        p = Path(tdir) / self._EVAL_INFLIGHT
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass

    def evaluate_only(self, trial: TrialRecord, *, parent_scores=None,
                      baseline_score=None, baseline_view=None,
                      slice_columns=None) -> TrialRecord:
        """仅重跑评估阶段(§16.5.4 evaluator_error 路径/评估口径修正)。

        不重新训练:复用已落盘的 validation_score.parquet。
        """
        manifest = freeze_trial_source(Path(trial.trial_dir))
        trial.data.setdefault("integrity", {})["source_manifest_sha256"] = \
            manifest["combined_sha256"]
        # v2.12.2.8:同 `run` 路径的 in-flight 标记,保证 signal 级崩溃
        # 在 evaluate_only 重入时也能被计数(否则 _eval_crash_count 恒 0,
        # 崩溃熔断形同虚设)。
        _tdir = Path(trial.trial_dir)
        self._mark_eval_inflight(_tdir)
        evaluation = self._evaluate_trial(
            trial, parent_scores=parent_scores,
            baseline_score=baseline_score, baseline_view=baseline_view,
            slice_columns=slice_columns,
            include_shadow=True)
        self._clear_eval_inflight(_tdir)
        trial.data["evaluation"] = evaluation
        trial.data["evaluation_path"] = "evaluation.json"
        fi = self._load_feature_importance(Path(trial.trial_dir))
        if fi is not None:
            evaluation["top_features"] = fi
        dump_json(Path(trial.trial_dir) / "evaluation.json", evaluation)
        record_attempt(trial, "reevaluate", "仅重跑评估阶段")
        if trial.status == "RUNNING":
            trial.transition("SUCCEEDED")
        if trial.status == "SUCCEEDED":
            trial.transition("EVALUATED")
        elif trial.status == "EVALUATED":
            trial.transition("EVALUATED")  # 重评覆盖
        trial.flush()
        return trial

    # ---- 实时状态:run-state.json 跟随 Trial 生命周期更新 ----
    # phase 是任务整体阶段(单调推进,不随单 trial 回退);trial 细节
    # 落 current_trial_*(单值快照,并行时以 trials/*/trial.yaml 为准)
    # v2.17.2(B7):STOPPED 在 FINAL_EVALUATED 之前(停止是建议性的,收尾
    # 盲测必须能覆盖它;与 scripts/_mvo/context.PHASE_ORDER 同序)
    PHASE_ORDER = ["INITED", "PROFILED", "BASELINED", "EVOLVING",
                   "STOPPED", "FINAL_EVALUATED"]

    def _update_run_status(self, status: str, trial: TrialRecord,
                           **extra) -> None:
        st_path = self.run_dir / "run-state.json"
        # flock 临界区:并行 trial 同时收尾时序列化读-改-写,
        # 避免丢更新 / phase 回退(_atomic_write 只保单个写原子)
        with locked_state_write(st_path):
            state = load_json(st_path) if st_path.exists() else {}
            # trial 级状态映射到任务 phase:执行中 → EVOLVING;FAILED 不
            # 回退 phase(任务仍在演进,失败信息落 last_failure_reason)
            phase = state.get("phase") or state.get("status") or "INITED"
            if phase not in self.PHASE_ORDER:
                phase = "INITED"
            if status in ("RUNNING", "RUNNING_RETRY"):
                new_phase = "EVOLVING"
            elif status in self.PHASE_ORDER and \
                    self.PHASE_ORDER.index(status) > self.PHASE_ORDER.index(phase):
                new_phase = status  # 只前进不后退(如 STOPPED/FINAL_EVALUATED)
            elif status == "EVALUATED":
                # [FIX-2026-09-01] EVALUATED 只把 phase 推到 EVOLVING(演进
                # 中任一 trial 评估完成都属于演进),但**不覆盖已更高的
                # phase**(BASELINED 阶段做 shadow 补打分时,评估会把 phase
                # 从 BASELINED 顶成 EVOLVING,导致 baseline 命令被误判为
                # 未完成、反复重跑——实测一 run 重复 5 次产生 6 个 baseline)。
                # 已达 EVOLVING/FINAL_EVALUATED/STOPPED 的保持不动。
                new_phase = "EVOLVING" if (
                    self.PHASE_ORDER.index("EVOLVING")
                    > self.PHASE_ORDER.index(phase)) else phase
            else:
                new_phase = phase
            # v2.17.2(B7):STOPPED 后又有 trial 开跑 = 演进继续;run-state 的
            # stop_reason 移到 superseded_stop_reason(evolution-state 的
            # stopped 由下一次 finalize-generation 归档进 stop_history)
            if phase == "STOPPED" and new_phase == "EVOLVING" \
                    and state.get("stop_reason"):
                state["superseded_stop_reason"] = state.pop("stop_reason")
            state.update({
                "phase": new_phase,
                "current_trial": trial.trial_id,
                "current_trial_status": status,
                "current_trial_generation": trial.generation,
                "status_updated_at": now_iso(),
                **extra,
            })
            dump_json(st_path, state)

    def _trial_logger(self, trial: TrialRecord) -> logging.Logger:
        """每个 Trial 的实时日志:trial.log(与 stdout/stderr 互补)。"""
        name = f"modelevo.trial.{trial.trial_id}"
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = False
        if not lg.handlers:
            fh = logging.FileHandler(
                Path(trial.trial_dir) / "trial.log", encoding="utf-8")
            fh.setFormatter(_CNFormatter())
            lg.addHandler(fh)
        return lg

    def check_topology(self, trial: TrialRecord) -> dict:
        """v2.16.0 模型拓扑门:裁决结果写入 trial.yaml.model_topology。

        拦截(allowed=False)时留痕 approvals/requests/ 与
        logs/user-decisions.md(PENDING),trial 状态不变。门自身异常按放行
        处理、告警并留痕 decision=gate_error——静态检测器的缺陷不应让整代
        停摆。
        """
        from .topology import evaluate_gate, gate_message, record_request

        try:
            gate = evaluate_gate(self.run_dir, trial, self.spec.model_type)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s 拓扑门执行异常(按放行处理): %s",
                           trial.trial_id, exc)
            gate = {"allowed": True, "decision": "gate_error",
                    "error": str(exc)[:300]}
        trial.data["model_topology"] = gate
        trial.flush()
        if not gate.get("allowed", True):
            msg = gate_message(trial.trial_id, gate)
            self._trial_logger(trial).error("%s", msg)
            logger.error("%s", msg)
            try:
                record_request(self.run_dir, trial, gate)
            except Exception as exc:  # noqa: BLE001
                logger.warning("拓扑批准请求留痕失败: %s", exc)
        return gate

    def run(
        self,
        trial: TrialRecord,
        *,
        plan: ResourcePlan | None = None,
        parent_scores: dict[str, pd.DataFrame] | None = None,
        baseline_score: pd.DataFrame | None = None,
        baseline_view: str | None = None,
        slice_columns: list[str] | None = None,
        timeout_minutes: float | None = None,
        force_rerun: bool = False,
        slot_index: int = 0,
    ) -> TrialRecord:
        """执行一个 Trial 到终态(EVALUATED/FAILED),含自动重试(§16.5.4)。

        parent_scores/baseline_score: {split: DataFrame(score 列)} 或
        DataFrame,用于 delta 与 slice 评测;由调用方从父 trial 目录读取。
        EVALUATED Trial 重新 run 时只重评不重训(评估口径可能已修正);
        force_rerun=True 时绕过该短路(successive halving 胜者全量重跑)。

        v14.3:plan 未显式传入时,按**当前 RUNNING 的兄弟 trial 数**收敛
        并发(runner._count_running_trials)——单 trial 独占吃满预算,
        手动并行多个 `run --trial` 时按实际并发数分线程,不超卖。
        批路径(run-generation/baseline)已传 plan,不经过这里。
        slot_index(v18/P1-3):批内序号,决定多卡分配时本 trial 绑
        gpu_indices 中的哪一张(空闲显存降序指派)。

        v2.17.2(B2):入口先查 ``.runner.lock``——别的进程的活 runner 正持有
        该 trial 时,在**任何**状态改写(FAILED 重入记 attempt、EVALUATED 重评、
        孤儿回退)之前抛 :class:`ConcurrentRunError`。调用方手里的 trial 记录
        是进入前加载的快照,任何 flush 都会覆写在跑 runner 的 trial.yaml。
        """
        _foreign = self._check_runner_lock(Path(trial.trial_dir)) \
            if getattr(trial, "trial_dir", None) else None
        if _foreign is not None and _foreign != os.getpid():
            raise ConcurrentRunError(
                trial.trial_id, _foreign,
                Path(trial.trial_dir) / self._RUNNER_LOCK)
        if trial.status == "EVALUATED" and not force_rerun:
            return self.evaluate_only(
                trial, parent_scores=parent_scores,
                baseline_score=baseline_score, baseline_view=baseline_view,
                slice_columns=slice_columns)
        if plan is None:
            _running, _slot = self._count_running_trials()
            # 本进程在标 RUNNING 前统计,运行中的兄弟 + 自身都计入;
            # 手动并行 N 个 → 按 N 路分线程,并发×线程≈可用核不超卖
            plan = self.scheduler.plan(planned_trials=max(1, _running))
            # v19.4:手动并行路径的 GPU 撞卡修复。批路径靠 slot_index
            # 把并发 trial 绑到 gpu_indices[slot % len];手动路径
            # 恒 slot_index=0 → 并发 trial 全挤 gpu_indices[0]
            # (实测 20:10:44 双 trial 同落 gpu=1)。此处以"本进程
            # 进入前已 RUNNING 的兄弟数"作槽位号,拿到与本进程并发
            # 的兄弟错开的卡。
            # v2.13.3:slot 只是**首选**卡,最终归属由 _execute_once 的
            # 排他租约裁决——slot 计算的竞态窗口(两进程同 slot)由租约
            # 兜底:抢不到首选卡就换下一张空闲卡,不再"各自认定同一张
            # 最空卡、谁也不让"(那是 gen-04 三 trial 全落 GPU0 的直接
            # 成因)。批路径同样走租约,故此处只影响日志与首选序。
            if getattr(plan, "gpu_indices", None) and len(
                    plan.gpu_indices) > 1:
                slot_index = (_slot) % len(plan.gpu_indices)
                lg_extra = ("手动并行 slot=%d → 首选 gpu=%s"
                            "(最终以租约为准)"
                            % (slot_index,
                               plan.gpu_indices[slot_index]))
                self._trial_logger(trial).info(lg_extra)
        timeout_min = self.resolve_timeout_minutes(timeout_minutes)
        grace = int(self.config.get("execution.kill_grace_period_sec", 30))
        lg = self._trial_logger(trial)

        tdir = Path(trial.trial_dir)
        code_dir = tdir / "code"
        # ---- 强约束:代码必须已落盘(§10.3)----
        if not any(code_dir.iterdir()):
            raise FileNotFoundError(
                f"code/ 为空:代码必须先在 Trial 运行目录落盘再执行({code_dir})")
        # FAILED Trial 重跑:按原失败分类记 attempt(retry / fix_code),
        # 以 attempt 为单位记录在同一 trial 下(§16.5.4)
        if trial.status == "FAILED":
            prev_reason = self._reenter_failed_trial(trial)
            lg.info("重跑(原失败: %s, attempt=%s)", prev_reason,
                    len(trial.data.get("attempts", [])))
        # ---- v2.8.1 孤儿 RUNNING 自愈(skill-fix-20260910)----
        # runner 被外部 kill 后 trial 卡 RUNNING:锁活 → 并发保护;
        # 孤儿且有分数 → evaluate_only 复用已完成训练;孤儿无分数
        # → 回退 CODE_WRITTEN 重训。旧版本 RUNNING 无锁 → 天然孤儿。
        if trial.status == "RUNNING":
            live_pid = self._check_runner_lock(tdir)
            if live_pid is not None:
                raise ConcurrentRunError(trial.trial_id, live_pid,
                                         tdir / self._RUNNER_LOCK)
            if (tdir / "validation_score.parquet").exists():
                # v2.12.2.8:评估段崩溃熔断(skill-fix-20260918)。
                # 若上一轮评估是**信号级崩溃**(段错误/OOM-kill),Python
                # 的 except 接不住, runner 进程直接死 —— 锁成孤儿、分数
                # 已落盘, 下一次 `run --trial` 又走 evaluate_only 回到同一
                # 崩溃点。实测 req-cnt-t7 因此空转 5 轮, 每轮还写一个
                # ~16GB core dump。用落盘的 in-flight 计数识别该形态:
                # 同一 trial 连续多次"进了评估却没写出结果"即判失败,
                # 不再重入, 把崩溃暴露成 FAILED 而不是无限循环。
                n_prev = self._eval_crash_count(tdir)
                _EVAL_CRASH_LIMIT = 2
                if n_prev >= _EVAL_CRASH_LIMIT:
                    lg.error(
                        "评估段连续 %d 次未产出结果(疑似信号级崩溃:"
                        "段错误/OOM-kill)。停止重入 evaluate_only —— "
                        "继续重入只会重复崩溃并产生 core dump。请先"
                        "排查崩溃原因(检查 stderr.log / core 文件 / "
                        "内存上限),修复后新建 trial。", n_prev)
                    trial.data["failure_reason"] = "eval_crash_loop"
                    trial.data["status"] = "FAILED"
                    trial.data["status_history"].append({
                        "state": "FAILED", "at": now_iso(),
                        "note": (f"eval_crash_loop: 评估段连续 {n_prev} 次"
                                 "未产出结果,疑信号级崩溃")})
                    trial.flush()
                    self._update_run_status(
                        "FAILED", trial,
                        last_failure_reason="eval_crash_loop")
                    return trial
                lg.warning(
                    "孤儿 RUNNING(锁进程已死)且已有落盘分数:"
                    "跳过重训,直接 evaluate_only 补评估"
                    "(该 trial 评估段已连续 %d 次未产出结果)", n_prev)
                return self.evaluate_only(
                    trial, parent_scores=parent_scores,
                    baseline_score=baseline_score,
                    baseline_view=baseline_view,
                    slice_columns=slice_columns)
            lg.warning(
                "孤儿 RUNNING(锁进程已死)且无分数:回退 CODE_WRITTEN "
                "重训(runner 被外部 kill,无落盘产物可复用)")
            trial.data["status"] = "CODE_WRITTEN"
            trial.data["status_history"].append(
                {"state": "CODE_WRITTEN", "at": now_iso(),
                 "note": ("orphan RUNNING reverted: runner killed "
                          "externally, no score → retrain")})
            trial.flush()
        if trial.status == "PROPOSED":
            trial.transition("CODE_WRITTEN")
        # force_rerun(halving 胜者全量重跑):EVALUATED → CODE_WRITTEN
        # 已在 GenerationExecutor 侧回退状态;这里只需放行 RUNNING 迁移
        if force_rerun and trial.status == "EVALUATED":
            trial.data["status"] = "CODE_WRITTEN"
            trial.data["status_history"].append(
                {"state": "CODE_WRITTEN", "at": now_iso(),
                 "note": "force_rerun"})
        # ---- 熔断检查(§16.5.6):触发时拒绝执行 ----
        if self.breaker_open:
            raise RuntimeError(
                "失败熔断已触发:先执行 root-cause reflection 并写入 Task "
                f"Experience,再用 --reset-breaker 恢复({self.breaker_reason})")

        # ---- v2.16.0 模型拓扑门:默认单模型,级联/融合须人工批准 ----
        # 放在 runner(immutable harness)而不是 lint:lint 从工作区副本加载、
        # 且有 MODELEVO_SKIP_LINT 逃生口,都在 Agent 可控范围内。位置在冻结
        # 之前:拦截时不冻结源码、不迁移状态(trial 仍可改写成单模型后重跑),
        # 也不记失败(不进熔断)。run / run-generation / baseline 都经过这里。
        if not self.check_topology(trial).get("allowed", True):
            return trial

        # Trial 是不可变实验单元。首次执行前冻结 code/config；之后任何
        # 修码必须创建新 trial，避免同一 ID 指向多份源码与多组结果。
        try:
            _source_manifest = freeze_trial_source(tdir)
            trial.data.setdefault("integrity", {})[
                "source_manifest_sha256"] = \
                _source_manifest["combined_sha256"]
        except Exception as exc:  # noqa: BLE001
            trial.data["failure_reason"] = "integrity_error"
            trial.data["status"] = "FAILED"
            trial.data.setdefault("status_history", []).append({
                "state": "FAILED", "at": now_iso(), "note": str(exc)})
            trial.flush()
            logger.error("Trial 源码冻结失败 %s:%s", trial.trial_id, exc)
            return trial

        trial.transition("RUNNING")
        trial.flush()
        _runner_lock = self._acquire_runner_lock(tdir)
        try:  # v2.8.1:锁住 RUNNING 生命周期,所有 return 路径释放
            self._update_run_status("RUNNING", trial)
            lg.info("开始执行: cmd=%s timeout=%.1fmin threads=%d",
                    trial.data["execution"]["command"], timeout_min,
                    plan.threads_per_trial)
            # v2.13.3:此处**不再**打 plan.gpu_index。它是批次级快照
            # ("机器上最空的那张卡"),而本 trial 的真实卡号由
            # _execute_once 的排他租约裁决;两者在并发批里必然不同,
            # 打 plan 值会让每个 trial 的日志都印同一张卡。实测
            # gen-03 六路并发因此看起来"全部共用 GPU0"——
            # 真实分配是 [0,2,3,4,5,6]。真实卡号在下一条租约日志
            # (_execute_once 内)与 trial.data.environment.gpu_index。
            lg.info("plan 快照卡(非本 trial 归属)=%s;本 trial 卡号见下条"
                    " GPU 租约日志", plan.gpu_index)
            lg.info("父代: %s; search_action=%s; hypothesis=%s",
                    trial.parent_ids or "无(根 Trial)",
                    trial.data.get("search_action"),
                    (trial.data.get("hypothesis") or {}).get("statement", ""))

            # ---- 执行 + 自动重试循环(§16.5.4)----
            # timeout/oom/infra_error 允许自动重试(降配);code_error 交回
            # Agent 修复(返回 FAILED);evaluator_error 只重评不重训。
            result = None
            while True:
                result = self._execute_once(trial, code_dir, tdir, plan,
                                            timeout_min, grace,
                                            slot_index=slot_index)
                if result.ok:
                    self.breaker.record(None)
                    break
                reason = result.failure_reason or "code_error"
                trial.data["failure_reason"] = reason
                if self.attempt_policy.next_action(trial, reason) != "retry":
                    break
                plan = self._downgrade_plan(plan)
                record_attempt(trial, "retry", f"自动重试({reason}),降配执行",
                               {"threads": plan.threads_per_trial,
                                "timeout_minutes": timeout_min})
                lg.warning("失败(%s) runtime=%.1fs,自动降配重试(threads=%d)",
                           reason, result.runtime_sec, plan.threads_per_trial)
                self._update_run_status("RUNNING_RETRY", trial,
                                        last_failure_reason=reason)

            trial.data["result"] = {
                "status": "succeeded" if result.ok else "failed",
                "runtime_sec": round(result.runtime_sec, 2),
                "peak_memory_gb": round(result.peak_memory_gb, 3),
                "returncode": result.returncode,
            }
            trial.flush()
            # v18/P0-2:训练时长**立即**落盘。compute_hours 按 result.runtime_sec
            # 聚合;旧序在评估完成后才整体 flush——评估段(400w 尺 ~1h 的分组
            # 置换)被外部 kill 时 result 恒空,该 trial 的训练时长从预算计量
            # 中消失(实测 400w run compute_hours 少计 3 倍+)。评估段自己的
            # 时长在下方评估完成后增量累加。
            _eval_started_at = time.time()
            # v2.13.5:评估段(分组置换)线程预算 = 本 trial 的调度配额。
            # runner 实例在 gen-executor 线程池里被多个 trial 共享,按
            # trial_id 分槽,不能挂单值属性(并发 trial 会互相覆盖)
            if not hasattr(self, "_eval_thread_budget"):
                self._eval_thread_budget = {}
            self._eval_thread_budget[trial.trial_id] = int(
                getattr(plan, "threads_per_trial", 0) or 0) or None
            if not result.ok:
                reason = result.failure_reason or "code_error"
                trial.data["failure_reason"] = reason
                trial.data["status"] = "FAILED"
                trial.data["status_history"].append(
                    {"state": "FAILED", "at": now_iso()})
                trial.flush()
                self.breaker.record(reason)
                self._check_breaker()
                # 失败诊断:stderr 尾部摘要在日志可见(完整内容在 stderr.log)
                tail = (result.stderr_tail or "").strip()
                tail_lines = "\n".join(tail.splitlines()[-12:])
                lg.error("终态 FAILED(%s) runtime=%.1fs peak_mem=%.2fGB",
                         reason, result.runtime_sec, result.peak_memory_gb)
                if tail_lines:
                    lg.error("stderr 尾部(完整见 stderr.log):\n%s", tail_lines)
                self._update_run_status("FAILED", trial,
                                        last_failure_reason=reason,
                                        last_error_tail=tail[-800:] or None)
                return trial
            try:
                verify_trial_source(tdir)
            except Exception as exc:  # 子进程改写了 code/config
                trial.data["failure_reason"] = "integrity_error"
                trial.data["status"] = "FAILED"
                trial.data["status_history"].append(
                    {"state": "FAILED", "at": now_iso(),
                     "note": f"post-run source verification:{exc}"})
                trial.flush()
                self._update_run_status(
                    "FAILED", trial, last_failure_reason="integrity_error")
                return trial
            trial.transition("SUCCEEDED")
            lg.info("训练完成: runtime=%.1fs peak_mem=%.2fGB,进入评估",
                    result.runtime_sec, result.peak_memory_gb)

            # ---- 评估;evaluator_error 只重评(§16.5.4)----
            # v2.12.2.8:评估前打 in-flight 标记。信号级崩溃(段错误/
            # OOM-kill)绕过 except,标记会留下;正常出结果即清零。
            self._mark_eval_inflight(tdir)
            evaluation = None
            for eval_attempt in range(2):
                try:
                    evaluation = self._evaluate_trial(
                        trial, parent_scores=parent_scores,
                        baseline_score=baseline_score,
                        baseline_view=baseline_view,
                        slice_columns=slice_columns,
                        include_shadow=True)
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.error("评估失败 %s: %s", trial.trial_id, exc)
                    lg.error("评估失败(attempt %d): %s", eval_attempt + 1, exc)
                    if eval_attempt == 0:
                        record_attempt(trial, "retry",
                                       "evaluator_error:仅重跑评估阶段")
                        continue
                    trial.data["failure_reason"] = "evaluator_error"
                    trial.data["status"] = "FAILED"
                    trial.data["status_history"].append(
                        {"state": "FAILED", "at": now_iso()})
                    trial.flush()
                    self._update_run_status("FAILED", trial,
                                            last_failure_reason="evaluator_error")
                    return trial
            if evaluation is None:
                # 兜底:`_evaluate_trial` 返回 None(而非抛异常)时,旧实现
                # 会写出 `"evaluation": null` 并把 trial 标成 EVALUATED ——
                # 一个没有评估产物的"成功"trial,下游 leaderboard 因缺分数
                # 报 lb_rows_absent,而状态机显示已评估。显式收口为 FAILED。
                trial.data["failure_reason"] = "evaluator_error"
                trial.data["status"] = "FAILED"
                trial.data["status_history"].append(
                    {"state": "FAILED", "at": now_iso(),
                     "note": "evaluator_error:未产出 evaluation(None)"})
                trial.flush()
                self._update_run_status("FAILED", trial,
                                        last_failure_reason="evaluator_error")
                return trial
            self._clear_eval_inflight(tdir)
            trial.data["evaluation"] = evaluation
            trial.data["evaluation_path"] = "evaluation.json"
            dump_json(tdir / "evaluation.json", evaluation)
            # 摘要特征重要性(top20)进 evaluation,Planner/反思免开 parquet
            fi = self._load_feature_importance(tdir)
            if fi is not None:
                evaluation["top_features"] = fi
                dump_json(tdir / "evaluation.json", evaluation)
            # v18/P0-2:评估段时长并入 runtime 计量(分组置换/误差归因/
            # 分桶报表在 400w 尺 ~1h,此前一概不计)。从训练结束起算——
            # eval 段起点在 result.flush 之前的时间轴上,这里补差值。
            try:
                _r = trial.data.get("result") or {}
                _eval_sec = time.time() - _eval_started_at
                _r["evaluate_sec"] = round(_eval_sec, 2)
                _r["runtime_sec"] = round(
                    float(_r.get("runtime_sec") or 0) + _eval_sec, 2)
                trial.data["result"] = _r
            except Exception:  # noqa: BLE001
                pass
            trial.transition("EVALUATED")
            trial.flush()
            pm = (evaluation.get("primary_metric") or {})
            lg.info("评估完成: %s=%s verdict=%s confidence=%s",
                    pm.get("name"), pm.get("value"),
                    evaluation.get("verdict"),
                    evaluation.get("confidence"))
            # v2.13.5:评估分项计时进 trial.log(timing-report 消费)
            if evaluation.get("timings"):
                lg.info("评估分项计时: %s", json.dumps(
                    evaluation["timings"], ensure_ascii=False))
            self._update_run_status("EVALUATED", trial,
                                    last_verdict=evaluation.get("verdict"),
                                    last_primary_metric=pm.get("value"))
            return trial
        finally:
            self._release_runner_lock(_runner_lock)

    # ------------------------------------------------------------------
    # v2.11:孤儿 RUNNING 快速检测。旧闭环(run 重入时 .runner.lock
    # pid 判活)要等下一次 run 同一 trial 才触发,实测盲区 ~30min
    # (run drawdown-req-t7-6w trial-007:心跳 19:58 消失,20:28 才
    # 被重入检测兜住)。本探针按"心跳停摆"主动判死:trial.log 末条
    # heartbeat 距今超过阈值(默认 5min = 5× 心跳周期)且锁进程已死
    # 时,可安全回退 CODE_WRITTEN 重训;锁进程仍活(如在评估段,不写
    # 心跳)时不判死。
    HEARTBEAT_STALE_SEC = 300

    def is_stale_running(self, trial) -> tuple[bool, str]:
        """RUNNING trial 的孤儿探针。返回 (is_stale, reason)。

        is_stale=True 的充要条件:无活 runner 锁 且 (无心跳日志 或
        末条心跳/终态行距今 > HEARTBEAT_STALE_SEC)。评估段不写心跳,
        但评估段内锁进程活,被第一个条件挡住——不会误杀。
        """
        if trial.status != "RUNNING":
            return False, ""
        tdir = Path(trial.trial_dir)
        if self._check_runner_lock(tdir) is not None:
            return False, "lock alive"
        log = tdir / "trial.log"
        last_ts = None
        try:
            if log.exists():
                lines = log.read_text(encoding="utf-8",
                                      errors="ignore").splitlines()
                for ln in reversed(lines[-80:]):
                    if "[runner.py" in ln:
                        # harness 日志行(评估段也在写)→ 有进程在工作
                        last_ts = time.time()
                        break
                    if "heartbeat:" in ln or "process exited" in ln:
                        # 心跳行形如 [2026-09-14T19:58:04+08:00] heartbeat: ...
                        head = ln.split("]", 1)[0].lstrip("[").strip()
                        from datetime import datetime as _dtm
                        last_ts = _dtm.fromisoformat(head).timestamp()
                        break
        except (OSError, ValueError):
            last_ts = None
        if last_ts is None:
            # 无锁 + 无心跳 = 创建后从未真正起跑或日志已丢 → 视为陈旧
            return True, "no heartbeat and no live lock"
        age = time.time() - last_ts
        if age > self.HEARTBEAT_STALE_SEC:
            return True, f"heartbeat stale {age:.0f}s > {self.HEARTBEAT_STALE_SEC}s"
        return False, f"heartbeat fresh ({age:.0f}s)"

    def _load_feature_importance(self, tdir: Path) -> list | None:
        """读取 Trial 产出的特征重要性(feature/ 子目录),取 top20 摘要。

        优先 csv;兼容旧 parquet 与旧位置(根目录)。
        """
        candidates = [
            Path(tdir) / "feature" / "feature_importance.csv",
            Path(tdir) / "feature" / "feature_importance.parquet",
            Path(tdir) / "feature_importance.csv",
            Path(tdir) / "feature_importance.parquet",
        ]
        p = next((c for c in candidates if c.exists()), None)
        if p is None:
            return None
        try:
            fi = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
            fi = fi.sort_values("importance", ascending=False)
            total = float(fi["importance"].sum()) or 1.0
            return [
                {"feature": str(r["feature"]),
                 "importance": float(r["importance"]),
                 "share": float(r["importance"]) / total}
                for _, r in fi.head(20).iterrows()
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取特征重要性失败 %s: %s", tdir, exc)
            return None

    # ------------------------------------------------------------------
    #: v2.13.3:GPU 跨进程排他租约目录(每卡一文件,写持有者 pid:token)。
    #: 见 scheduler 模块级 GPU 租约说明——修并发 trial 撞同一张卡。
    _GPU_LEASES = ".gpu-leases"

    # ------------------------------------------------------------------
    def _acquire_gpu_lease(self, trial=None, plan=None, slot_index: int = 0,
                           capability=None):
        """v2.13.3:为本次执行抢一张卡的排他租约。

        返回 (GpuLease | None, gpu_index | None)。无卡(plan 未给卡且
        快照无卡)时返回 (None, None)——调用方按 CPU 执行。

        候选顺序:plan 已指派的卡优先(保住批路径 slot 铺开的语义;
        单路时即 plan.env 的 CUDA_VISIBLE_DEVICES),其后按
        gpu_candidates 的健康度序补位。**关键**是"抢不到就换下一张",
        而不是像旧逻辑那样无条件用 plan 指定的那张——旧逻辑下并发
        进程各自认定同一张最空卡,谁也不让,结果全挤一张。
        所有候选都被活进程持有(说明卡确实跑满了)→ 返回 (None, None)
        按 CPU 跑,而不是超发抢卡。

        v2.13.3 [能力门控]:取卡前先问"本 trial 能不能真用上这张卡"
        (见 runtime.gpu_capability)。机器有卡 ≠ 本 trial 能用卡——
        lightgbm 的 PyPI wheel 无 CUDA 编译,preflight 早就探到
        ``gbm_backends.lightgbm_cuda=false``,但该事实此前没有任何
        消费方,于是 6 个 lightgbm trial 各领一张 A800 的排他租约、
        全程 0 GPU 计算量(实测 gen-03,租约 445s / 有用 58s)。能力
        快照缺失时 fail-open 放行(见 gpu_capability 模块说明)。
        """
        _can, _why = trial_can_use_gpu(
            getattr(trial, "data", None), capability,
            run_dir=getattr(self, "run_dir", None))
        if not _can:
            logger.info(
                "GPU 能力门控: %s 不取卡(%s)——按 CPU 执行,把卡留给能用卡的 "
                "trial。环境事实来自 env-manifest.gbm_backends(与 trial 同"
                "解释器探测),不是故障",
                getattr(trial, "trial_id", "?"), _why)
            return None, None
        snap_gpus = list((self.scheduler.snapshot() or {}).get("gpus") or [])
        if not snap_gpus:
            return None, None
        preferred = []
        if getattr(plan, "gpu_indices", None):
            preferred.append(
                plan.gpu_indices[slot_index % len(plan.gpu_indices)])
        elif getattr(plan, "gpu_index", None) is not None:
            preferred.append(int(plan.gpu_index))
        order: list[int] = []
        for gi in preferred:
            if gi not in order:
                order.append(gi)
        for g in gpu_candidates(self.scheduler.snapshot()):
            gi = int(g.get("index"))
            if gi not in order:
                order.append(gi)
        leases_dir = Path(self.run_dir) / self._GPU_LEASES
        # v2.17.0:按实际占用选卡。租约只在本 run 内互斥,看不见同机其它
        # run 的进程(run4 regression.2:trial-016 被派到与其它 run 2 个进程
        # 共享的 GPU 1,分类子模型 42 min;空闲卡约 10 min)。先在"无外来
        # 进程、显存充足"的卡里抢;全部被占 → 退回旧排序并留痕;nvidia-smi
        # 查询失败 → 旧逻辑。plan 首选卡若是干净卡仍优先(保住批路径铺开)。
        occ = self._gpu_occupancy()
        mode = "legacy" if occ is None else "fallback_all_occupied"
        _require_clean = self.config.get(
            "execution.require_clean_gpu", False) is True
        # v2.17.3(验证回写):min_free_gpu_gb 只在 require_clean_gpu 下才是
        # 硬门槛。默认 false 时不得拒绝"低显存但可用"的卡 —— 旧行为会取其中
        # 空闲显存最大的一张,静默改成 CPU 属于非声明式的行为变化。
        _require_free_gb = (float(self.config.get(
            "execution.min_free_gpu_gb", 20.0) or 0.0)
            if _require_clean else 0.0)
        if occ is not None:
            clean, _busy = gpu_candidates_by_occupancy(occ)
            # v2.17.3:准入门槛——外来进程会把 GPU 变成 CPU 级速度
            # (实测:同卡 2 进程 1.77×、4 进程 3.5×,run4 相邻 trial 单树耗时
            # 0.11 s → 0.46 s 的 2.84× 未解释残差正落在该区间)。默认只算
            # "告警",显式开 require_clean_gpu 才拒绝被占卡;min_free_gpu_gb
            # 是显存硬门槛(卡被别人的大模型占着时应直接换卡或降级 CPU)。
            clean_order = ([gi for gi in preferred if gi in clean]
                           + [gi for gi in clean if gi not in preferred])
            for gi in clean_order:
                lease = GpuLease.acquire(leases_dir, gi)
                if lease is not None:
                    self._note_gpu_selection(trial, "occupancy", gi, occ)
                    return lease, gi
            if _require_clean:
                logger.warning(
                    "GPU 准入拒绝:无干净卡(干净卡 %s 均被本 run 租约占用或"
                    "不存在;被外来进程占用的卡 %s)——require_clean_gpu=true,"
                    "本 trial 按 CPU 执行而不是与其它 run 共享算力",
                    clean, _busy)
                return None, None
            logger.warning(
                "GPU 按占用选卡: 无可用的干净卡(干净卡 %s 均被本 run 租约占用"
                "或不存在;被外来进程占用的卡 %s)——退回旧排序 %s(可能与其它 "
                "run 共享算力,训练会变慢)", clean, _busy, order)
        elif getattr(self.scheduler, "live", False):
            mode = "query_failed"
        for gi in order:
            # 被外来进程占用且显存余量不足的卡不抢(实测共享训练慢 1.8~3.5×)
            if occ is not None and _require_free_gb > 0:
                _card = next((g for g in occ
                              if int(g.get("index", -1)) == gi), None)
                if _card is not None and not _card.get("clean") \
                        and float(_card.get("memory_free_gb") or 0.0) \
                        < _require_free_gb:
                    logger.warning(
                        "GPU %s 空闲显存 %.1fGB < 门槛 %.1fGB 且被外来进程占用"
                        "(foreign=%s, util=%s%%),跳过该卡",
                        gi, float(_card.get("memory_free_gb") or 0.0),
                        _require_free_gb, _card.get("foreign_procs"),
                        _card.get("util_pct"))
                    continue

            lease = GpuLease.acquire(leases_dir, gi)
            if lease is not None:
                self._note_gpu_selection(trial, mode, gi, occ)
                return lease, gi
        logger.warning(
            "GPU 全部被活进程持有(候选 %s),本 trial 按 CPU 执行——"
            "不超发抢卡(并发度高于卡数时属预期,非错误)", order)
        return None, None

    def _gpu_occupancy(self):
        """v2.17.0:实时 GPU 占用(标注外来进程);非 live 调度器或查询失败
        返回 None(调用方走旧逻辑)。注入快照的调度器(测试/离线回放)以
        快照为准,不碰真实设备。"""
        if not getattr(getattr(self, "scheduler", None), "live", False):
            return None
        try:
            occ = query_gpu_occupancy()
            if not occ:
                return None
            return annotate_gpu_occupancy(occ, getattr(self, "run_dir", None))
        except Exception as exc:  # noqa: BLE001  探测异常不阻塞选卡
            logger.warning("GPU 占用查询异常,回退旧选卡逻辑: %s", exc)
            return None

    def _note_gpu_selection(self, trial, mode: str, gi: int, occ) -> None:
        """选卡留痕:trial.yaml.environment.gpu_selection;非 legacy 模式另写
        logs/admission.jsonl(event=gpu_select)。"""
        card = next((g for g in (occ or []) if int(g.get("index", -1)) == gi),
                    {})
        rec = {"mode": mode, "gpu": gi,
               "foreign_procs": card.get("foreign_procs"),
               "util_pct": card.get("util_pct"),
               "memory_free_gb": card.get("memory_free_gb"),
               "occupied": [{"index": g.get("index"),
                             "foreign_procs": g.get("foreign_procs"),
                             "util_pct": g.get("util_pct")}
                            for g in (occ or []) if not g.get("clean")]}
        try:
            trial.data.setdefault("environment", {})["gpu_selection"] = rec
        except Exception:  # noqa: BLE001  鸭子类型 trial
            pass
        if mode == "legacy":
            return
        try:
            append_admission_log(self.run_dir, {
                "at": now_iso(), "epoch": round(time.time(), 3),
                "event": "gpu_select",
                "trial_id": getattr(trial, "trial_id", None),
                "generation": (getattr(trial, "data", None) or {})
                .get("generation"), "pid": os.getpid(), **rec})
        except Exception:  # noqa: BLE001
            pass

    def _execute_once(self, trial, code_dir, tdir, plan, timeout_min, grace,
                      slot_index: int = 0):
        """执行一次训练子进程(含 v2.17.0 内存准入)。

        先过内存准入(不满足就排队;等满 execution.admission_max_wait_min
        → 不启动训练,返回 failure_reason=admission_timeout),准入预留持有
        到训练子进程结束;再取 GPU 租约并拉起子进程。deadline
        (MODELEVO_DEADLINE_TS)在准入**之后**计算,排队不吃训练时长。
        """
        _ticket = self._admit_memory(trial, tdir)
        if _ticket is not None and not _ticket.admitted:
            return ExecutionResult(
                ok=False, returncode=None, runtime_sec=0.0,
                peak_memory_gb=0.0, failure_reason=ADMISSION_TIMEOUT,
                stdout_path=Path(tdir) / "stdout.log",
                stderr_path=Path(tdir) / "stderr.log",
                stderr_tail=(
                    f"admission_timeout: 内存准入排队 "
                    f"{_ticket.wait_sec / 60:.1f} min 仍不满足(预估峰值 "
                    f"{_ticket.estimate_gb:.1f}GB,来源 {_ticket.source};"
                    f"有效可用 {_ticket.available_gb}GB)。训练未启动——infra "
                    "类失败,不计方向负奖励、不进熔断;内存回落后原样重跑 "
                    "`run --trial` 即可(重入记 requeue)"))
        try:
            return self._execute_admitted(trial, code_dir, tdir, plan,
                                          timeout_min, grace,
                                          slot_index=slot_index,
                                          ticket=_ticket)
        finally:
            if _ticket is not None:
                _ticket.release()

    def _admit_memory(self, trial, tdir):
        """v2.17.0:启动训练前的内存准入(见 scheduler.MemoryAdmission)。

        返回 AdmissionTicket;准入关闭 / 非落盘 trial(鸭子类型替身无从估计)
        / 准入组件异常时返回 None 放行(fail-open:准入是防 OOM 的护栏,
        自身故障不应让训练停摆)。排队期间 trial.yaml.admission.state=queued
        并写 trial.log,`wait` 能看到"在排队"而不是"卡住"。
        """
        if getattr(trial, "trial_dir", None) is None:
            return None
        config = getattr(self, "config", None)
        data = getattr(trial, "data", None) or {}
        try:
            adm = MemoryAdmission(self.run_dir, config)
            if not adm.enabled:
                return None
            est = estimate_trial_peak_gb(self.run_dir, data, config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("内存准入组件异常,按放行处理: %s", exc)
            return None
        lg = self._trial_logger(trial)

        def _on_queue(ev: dict) -> None:
            lg.warning(
                "内存准入排队(训练尚未启动): 预估峰值 %.1fGB(来源 %s%s),"
                "有效可用 %sGB − 本 run 未兑现预留 %.1fGB − 安全余量 %.0fGB 不足;"
                "每 %.0fs 复查,最多等 %.0f min(日志 logs/admission.jsonl)",
                est["estimate_gb"], est.get("source"),
                f":{est['basis_trial']}" if est.get("basis_trial") else "",
                ev.get("available_gb"), ev.get("outstanding_gb") or 0.0,
                adm.margin_gb, adm.poll_sec, adm.max_wait_sec / 60.0)
            trial.data["admission"] = {
                "state": "queued", "since": now_iso(),
                "estimate_gb": est["estimate_gb"],
                "estimate_source": est.get("source"),
                "basis_trial": est.get("basis_trial"),
                "available_gb": ev.get("available_gb")}
            try:
                trial.flush()
            except Exception:  # noqa: BLE001
                pass

        def _on_wait(ev: dict, waited: float) -> None:
            lg.info("内存准入仍在排队: 已等 %.1f min,有效可用 %sGB,本 run "
                    "未兑现预留 %.1fGB", waited / 60.0, ev.get("available_gb"),
                    ev.get("outstanding_gb") or 0.0)

        try:
            ticket = adm.acquire(trial.trial_id, est,
                                 generation=data.get("generation"),
                                 on_queue=_on_queue, on_wait=_on_wait)
        except Exception as exc:  # noqa: BLE001
            logger.warning("内存准入异常,按放行处理: %s", exc)
            return None
        trial.data["admission"] = ticket.to_record()
        try:
            trial.flush()
        except Exception:  # noqa: BLE001
            pass
        if ticket.admitted and ticket.wait_sec >= 1.0:
            lg.info("内存准入通过: 排队 %.1f min(预估峰值 %.1fGB,有效可用 %sGB)",
                    ticket.wait_sec / 60.0, ticket.estimate_gb,
                    ticket.available_gb)
        return ticket

    def _execute_admitted(self, trial, code_dir, tdir, plan, timeout_min,
                          grace, slot_index: int = 0, ticket=None):
        command = trial.data["execution"]["command"].split()
        # 运行时解释器对齐:确保 `python` 落在环境解释器上。
        # 本机默认 PATH 的 python 缺依赖(如 pandas)会导致全部 trial
        # ModuleNotFoundError;subprocess 继承的 PATH 不保证为首位。
        # 仅在命令以 "python" 开头时替换,保留自定义解释器路径。
        if command and command[0] == "python":
            _py = sys.executable
            if _py:
                command[0] = _py
        # v14:把本次 trial 的绝对 deadline 交给子进程,长搜索(hpo)
        # 据此提前收敛并落盘(run_hpo_search 的 deadline_ts),而不是
        # 撞墙被 SIGTERM 后全部作废。已扣除 kill grace 余量。
        env_extra = dict(plan.env or {})
        env_extra["MODELEVO_DEADLINE_TS"] = str(
            time.time() + timeout_min * 60 - grace)
        # v2.13.4 [工件写保护]:只有经 runner 执行的 trial 进程才可写
        # TRIAL_DIR 下的模型/帧/shadow 工件。模板 dump_model/dump_eval_frame/
        # dump_shadow_score 校验本值 == TRIAL_DIR.name,不符即拒写。实测事故
        # (req-t7-drawdown):收尾期外部脚本加载冻结 train.py 调 fit_and_score,
        # 两次把冠军 model.joblib / validation_input.parquet 覆写成探针与
        # 手写重训的产物,完整性哈希失配。
        env_extra["MODELEVO_TRIAL_EXEC"] = Path(tdir).resolve().name
        # v2.15.0 [容量口径单源]:run 配置的 baseline.n_estimators_cap /
        # early_stopping_rounds 经环境变量下发,runtime.hpo.build_gbm 的
        # 默认值据此对齐。旧实现不下发,模板渲染的 HPO trial(无模块级
        # N_ESTIMATORS_CAP 常量)恒落回 runtime 默认 2000 —— 与嫁接路径
        # 丢 cap 是同一缺陷的另一入口(req-t7-rand400w run3 trial-011:
        # cap=8000 的 run 上 HPO 重训被截在 1999 轮,Δ 反号 −0.0024)。
        # 已显式设置的环境变量(诊断/人工覆写)优先,不覆盖。
        for _ek, _ck in (("MODELEVO_N_ESTIMATORS_CAP",
                          "baseline.n_estimators_cap"),
                         ("MODELEVO_EARLY_STOP_ROUNDS",
                          "baseline.early_stopping_rounds")):
            _cv = self.config.get(_ck)
            if _cv and _ek not in env_extra and not os.environ.get(_ek):
                env_extra[_ek] = str(int(_cv))
        # v2.16.0:早停面(validation | train_tail)单源下发;模板缺 env 时
        # 也默认 validation。已显式设置的环境变量优先。
        if ("MODELEVO_ES_SOURCE" not in env_extra
                and not os.environ.get("MODELEVO_ES_SOURCE")):
            env_extra["MODELEVO_ES_SOURCE"] = str(self.config.get(
                "training.early_stop_source", "validation") or "validation")
        # v2.17.3:早停集行数上限(training.early_stop_eval_rows,0=不降采样)。
        # 只影响"何时停"——早停位置在分层子样本上确定,落盘分数仍全量口径。
        if ("MODELEVO_ES_EVAL_ROWS" not in env_extra
                and not os.environ.get("MODELEVO_ES_EVAL_ROWS")):
            try:
                _es_rows = int(self.config.get(
                    "training.early_stop_eval_rows", 0) or 0)
            except (TypeError, ValueError):
                _es_rows = 0
            if _es_rows > 0:
                env_extra["MODELEVO_ES_EVAL_ROWS"] = str(_es_rows)
        # v2.17.3:搜索代不产 shadow 工件。shadow 分数只被 fresh
        # confirmation 消费(runner 的 shadow_policy=confirmation_only),其余
        # trial 的 shadow 段纯浪费 —— 44.7 万行上重放 transform + 特征块 +
        # 全量打分,req-t7 run4 实测冷缓存世代 557~686 s/trial、热缓存
        # ~55 s。confirm(以及既有的 shadow 开关前置条件)照常产出;
        # 人工诊断可用 MODELEVO_SHADOW_SCORE=1 显式覆盖。
        # ⚠ 默认**不启用**:fresh confirmation 硬依赖父代的
        # shadow_test_score(runner 的 confirm 分支要求父代 shadow 作独立复现
        # 对照),关掉搜索代的 shadow 会让之后的 confirm 直接失败。要启用
        # 就在 run 配置里显式列出动作:
        #   evaluator.shadow_skip_search_actions: [explore, crossover, data_view]
        # 此时被列出的动作不产 shadow 工件,其余(confirm/exploit/hpo/...)照常。
        _act = str((getattr(trial, "data", None) or {}).get(
            "search_action") or "")
        _skip_actions = self.config.get(
            "evaluator.shadow_skip_search_actions", []) or []
        if isinstance(_skip_actions, str):
            _skip_actions = [x.strip() for x in _skip_actions.split(",") if x.strip()]
        _skip_actions = {str(x) for x in _skip_actions}
        # confirm 硬依赖:父代的 shadow_test_score 是 fresh confirmation 的
        # 独立复现对照,**本 trial 自己**也必须有 shadow 工件才能被确认。
        # 误配进 skip 列表会直接让确认失败,这里强制剔除并告警。
        if "confirm" in _skip_actions:
            _skip_actions.discard("confirm")
            logger.warning(
                "evaluator.shadow_skip_search_actions 含 confirm,已忽略 —— "
                "confirm trial 必须产出 shadow_test_score,否则 fresh "
                "confirmation 无法通过")
        if ("MODELEVO_SHADOW_SCORE" not in env_extra
                and not os.environ.get("MODELEVO_SHADOW_SCORE")
                and _skip_actions):
            env_extra["MODELEVO_SHADOW_SCORE"] = (
                "skip" if _act in _skip_actions else "1")
        # v2.13.3:GPU 归属由**排他租约**裁决,取代旧的"plan 说哪张就是
        # 哪张"。租约持有到子进程结束(try/finally),期间别的 trial 抢
        # 不走这张卡。plan 指定的卡被占时自动换下一张空闲卡——这正是
        # 并发 trial 不再撞卡的关键。
        _lease, _gpu = self._acquire_gpu_lease(trial, plan, slot_index)
        if _lease is not None:
            env_extra["CUDA_VISIBLE_DEVICES"] = str(_gpu)
            try:
                trial.data.setdefault("environment", {})["gpu_index"] = _gpu
                trial.data["environment"]["gpu_lease"] = \
                    f"pid={os.getpid()} gpu={_gpu}"
            except Exception:  # noqa: BLE001
                pass
            logger.info("GPU 租约: %s → gpu=%s(候选序见 scheduler,"
                        "本 trial 实际归属)", trial.trial_id, _gpu)
        else:
            # 抢不到任何卡:显式清空,防 plan.env 残留的 CUDA_VISIBLE_DEVICES
            # 指向一张已被别的进程独占的卡(旧行为正是这样撞上去的)
            env_extra["CUDA_VISIBLE_DEVICES"] = ""
        # v2.13.1 [GPU 加固]:CUDA_VISIBLE_DEVICES 有实际卡 → MODELEVO_GPU
        # 强制同步为 "1" 并覆写 plan.env 的旧值。实测 gen-04 三个 HPO
        # trial(手动并行 run --trial 路径)子进程模板 _gpu_allocated 读到
        # MODELEVO_GPU 非 "1" → build_gbm 未设 device=cuda/task_type=GPU,
        # xgboost/lightgbm/catboost 全部回退 CPU 24 线程训练(心跳 2200%
        # 满载)。CUDA_VISIBLE_DEVICES 是"真实绑卡"的权威信号(v2.13.3 起
        # 由租约裁决写入,单路时取自 plan.env 的 gpu_index),有卡即置 GPU
        # 意图;无卡置 "0" 防模板残留旧值误判。改这里同时覆盖批路径与
        # 手动路径,不依赖 scheduler 探测是否注入。
        _cvd = env_extra.get("CUDA_VISIBLE_DEVICES")
        if _cvd not in (None, ""):
            env_extra["MODELEVO_GPU"] = "1"
        else:
            env_extra["MODELEVO_GPU"] = "0"
        # v14:解析后的解释器写回 trial 记录。命令串是复现配方,
        # 只在内存里替换会让 trial.yaml 留着 "python train.py" —— 按
        # 记录复现的人照样撞 ModuleNotFoundError(执行与溯源脱节)。
        try:
            trial.data.setdefault("environment", {})["interpreter"] = \
                command[0]
            trial.data["execution"]["resolved_command"] = " ".join(command)
            trial.flush()
        except Exception:  # noqa: BLE001  溯源落盘失败不阻塞执行
            pass
        try:
            # v2.17.0:有准入预留时登记训练进程 pid(未兑现预留按实占扣减)
            _spawn_kw = ({"on_spawn": ticket.attach_pid}
                         if ticket is not None
                         and getattr(ticket, "path", None) is not None
                         else {})
            return run_trial_process(
                command, cwd=code_dir,
                stdout_path=tdir / "stdout.log",
                stderr_path=tdir / "stderr.log",
                timeout_sec=timeout_min * 60, grace_sec=grace,
                env_extra=env_extra,
                heartbeat_log=tdir / "trial.log",
                **_spawn_kw,
            )
        finally:
            # v2.13.3:租约持有到子进程结束;异常/超时/重试路径同样释放
            if _lease is not None:
                _lease.release()

    def _downgrade_plan(self, plan: ResourcePlan) -> ResourcePlan:
        """重试降配(§16.5.4):减半线程;GPU Trial 回退 CPU。

        v2.11:线程下限护栏(scheduler 同口径 execution.threads_floor,
        默认 8)。旧实现减半可一路降到 1——实测(run
        drawdown-req-t7-6w)threads=4 的全量训练已在必然超时区,继续
        降配只是把"白跑一次"变"白跑两次"。
        """
        _floor = int(self.config.get("execution.threads_floor", 8))
        threads = max(min(_floor, max(1, plan.threads_per_trial)),
                      plan.threads_per_trial // 2)
        env = dict(plan.env)
        env.update({
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "NUMEXPR_NUM_THREADS": str(threads),
        })
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["CUDA_VISIBLE_DEVICES"] = ""
        # v17.4:GPU 意图信号同步降级——trial 模板只看 MODELEVO_GPU,
        # 防 pick_gpu() 在子进程内又把 CPU 重试顶回一张卡
        env["MODELEVO_GPU"] = "0"
        return ResourcePlan(n_parallel_trials=1, threads_per_trial=threads,
                            gpu_index=None, env=env)

    def _check_breaker(self, generation_failures: int = 0,
                       generation_total: int = 0) -> dict | None:
        """失败熔断判定(§16.5.6),委托 ``FailureCircuitBreaker.tripped``。

        v2.15.8:旧实现自己重写了"连续同类失败"判据,``tripped()`` 无人
        调用,其代级失败率分支(generation_failure_rate)从未生效。单
        trial 失败时以 (0, 0) 调用 —— 只可能触发连续同因分支,与旧行为
        一致;代级失败率由 GenerationExecutor 在一代收尾时传入计数。
        """
        trip = self.breaker.tripped(generation_failures, generation_total)
        if trip is None:
            return None
        self.breaker_open = True
        if trip["kind"] == "consecutive_same_reason":
            self.breaker_reason = trip["reason"]
            logger.error("失败熔断触发:连续 %d 次 %s;暂停新 Trial 调度",
                         trip["count"], trip["reason"])
        else:
            self.breaker_reason = (
                f"generation_failure_rate:{trip['rate']:.0%}"
                f"({generation_failures}/{generation_total})")
            logger.error("失败熔断触发:本代失败率 %.0f%% > 阈值 %.0f%%;"
                         "暂停新 Trial 调度(root-cause 后 run --reset-breaker)",
                         trip["rate"] * 100, self.breaker.rate_limit * 100)
        self._persist_breaker_state()
        return trip

    def reset_breaker(self) -> None:
        self.breaker_open = False
        self.breaker_reason = None
        self.breaker._recent = []
        self._persist_breaker_state()

    # ------------------------------------------------------------------
    def _evaluate_trial(self, trial, *, parent_scores=None, baseline_score=None,
                        baseline_view=None,
                        slice_columns=None, include_shadow=False) -> dict:
        tdir = Path(trial.trial_dir)
        # v2.13.5:评估段分项计时(evaluation.timings)。req-t7-rand400w 的
        # 瓶颈定位只能靠文件 mtime 反推 —— bootstrap/分组置换/报表各占
        # 多少从产物里读不出。
        _eval_timings: dict = {}
        _t_eval0 = time.time()
        _is_confirm = str(trial.data.get("search_action") or "") == "confirm"
        if _is_confirm:
            if len(trial.parent_ids) != 1:
                raise PermissionError("confirm 必须且只能有一个父 Trial")
            _parent_dir = self.run_dir / "trials" / trial.parent_ids[0]
            _parent_trial = TrialRecord.load(_parent_dir)
            _eligible, _why = confirmation_parent_eligibility(_parent_trial)
            if not _eligible:
                raise PermissionError(
                    f"confirm 父代不具备晋升资格:{_why}")
            verify_trial_source(_parent_dir)
            verify_confirmation_seed(tdir, _parent_dir)
        # 数据视图(协议 v7):trial 声明 data_view 时评估读对应视图目录,
        # 标签与特征同视图,跨视图对比经 rank/metric 层仍然成立
        view = trial.data.get("data_view") or "v1"
        frames = load_split_frames(
            self.run_dir,
            str(view_splits_dir(self.run_dir, view).relative_to(self.run_dir)))
        valid = frames.get("validation")
        if valid is None:
            raise RuntimeError("缺少 validation split(sample-profile/splits)")
        # v21 四平面:标签在 splits/labels/,特征面不含目标/时间/ID 列。
        # view_splits_dir 返回的已是绝对路径,再拼 Path(self.run_dir) 前缀
        # 在相对 --run 下会得到错误路径 → join 被静默跳过 → 后续
        # valid[target] KeyError,且 slice 门因时间/ID 列缺席被静默禁用
        labels_dir = view_splits_dir(self.run_dir, view) / "labels"
        if not (labels_dir / "validation.parquet").exists():
            raise FileNotFoundError(
                f"标签面缺失: {labels_dir}/validation.parquet(先跑 profile)")
        _va_lab = pd.read_parquet(labels_dir / "validation.parquet")
        if SAMPLE_KEY_COLUMN not in _va_lab.columns:
            # 旧 run 的一次性兼容迁移：从权威标签键列生成哈希身份，
            # 不允许使用行号伪装成样本身份。
            _va_lab[SAMPLE_KEY_COLUMN] = sample_keys(
                _va_lab, self.spec).values
            _va_lab.to_parquet(labels_dir / "validation.parquet", index=False)
        valid = valid.join(_va_lab.set_index(valid.index), how="left",
                           rsuffix="_lab")
        score_path = tdir / "validation_score.parquet"
        if not score_path.exists():
            raise FileNotFoundError(
                f"Trial 未产出 validation_score.parquet: {score_path}")
        source_manifest = verify_trial_source(tdir)
        model_files = [p for p in (tdir / "model").rglob("*") if p.is_file()] \
            if (tdir / "model").is_dir() else []
        prediction_meta = bind_prediction_artifact(
            score_path, _va_lab, source_manifest=source_manifest,
            split_manifest_path=view_splits_dir(
                self.run_dir, view) / "manifest.json",
            data_view=str(view), model_paths=model_files)
        score_df = pd.read_parquet(score_path)
        score = score_df["score"].values
        y = valid[self.spec.target_name].values

        # uplift:treatment 归属来自 validation split(与 y 同行序)
        treatment = None
        tcol = self.spec.treatment_column
        if tcol is not None:
            if tcol not in valid.columns:
                raise RuntimeError(
                    f"validation split 缺少 treatment 列 {tcol}"
                    "(检查 data-loader 是否输出了该列)")
            treatment = valid[tcol].values

        def _as_df(x):
            return x if isinstance(x, pd.DataFrame) else None

        # ---- v2.17.0:多父代 crossover 的 vs_parent 口径 ----
        # crossover_parent_test=iut:对每个父代都做配对 bootstrap,取
        # p_better 最低(最保守)者作正式 vs_parent(slice/temporal 同用);
        # first_parent(旧)保持调用方传入的父代分数不变。
        _vs_parent_id = None
        _iut_rows = None
        if not _is_confirm and len(trial.parent_ids) >= 2 and str(
                trial.data.get("search_action") or "") == "crossover":
            from .evolution import crossover_parent_test

            if crossover_parent_test(self.config) == "iut":
                _iut = self._crossover_iut(trial, view, score_df, y,
                                           treatment)
                _iut_rows = _iut.get("rows")
                if _iut.get("pick_pack") is not None:
                    parent_scores = _iut["pick_pack"]
                    _vs_parent_id = _iut["pick"]
        elif len(trial.parent_ids) == 1:
            _vs_parent_id = trial.parent_ids[0]

        parent_score = None
        parent_df = None
        parent_alignment_ok = True
        if parent_scores:
            p = parent_scores.get("validation") if isinstance(
                parent_scores, dict) else parent_scores
            # v2.13.2:跨视图父代结构性不可比(见 baseline_view_compatible)。
            # 此处只关对齐门,不动 integrity.valid —— 保持"无 vs_parent
            # 即无提升主张"的既有语义(evaluator delta is None → rejected),
            # 而不是把它升级成 invalid。
            _parent_view = (parent_scores.get("data_view")
                            if isinstance(parent_scores, dict) else None)
            if not baseline_view_compatible(view, _parent_view):
                parent_alignment_ok = False
                parent_df = None
                logger.info(
                    "%s:父代属 %s 视图,trial 属 %s —— 跨视图不可配对,"
                    "跳过 vs_parent(不再裁 invalid)",
                    trial.trial_id, _parent_view or "v1", view)
            else:
                parent_df = _as_df(p)
            if parent_df is not None:
                parent_score = parent_df["score"].values
                try:
                    from .metrics import align_score_by_keys

                    parent_score = align_score_by_keys(
                        parent_score, score_df[SAMPLE_KEY_COLUMN].values,
                        parent_df[SAMPLE_KEY_COLUMN].values)
                    parent_df = pd.DataFrame({
                        "score": parent_score,
                        SAMPLE_KEY_COLUMN:
                            score_df[SAMPLE_KEY_COLUMN].values})
                except Exception as exc:  # evaluator 会裁为 invalid
                    parent_alignment_ok = False
                    logger.warning("%s:父代预测不可配对:%s",
                                   trial.trial_id, exc)
        # ---- v2.17.0:锚定晋级的锚点分数(promotion_reference=anchor)----
        # 锚点 = 本代开始时的交付锚点(上一代 finalize 登记,同代固定);
        # 不同视图记 anchor_incomparable 并跳过,不影响单试验 verdict。
        _anchor = self._promotion_anchor_scores(
            trial, view, score_df, _vs_parent_id,
            parent_ok=(parent_score is not None and parent_alignment_ok))
        base_score = None
        base_df = None
        baseline_comparable = True
        if baseline_score is not None:
            base_df = _as_df(baseline_score)
            # v2.13.2:跨视图 baseline 结构性不可比 —— 不再喂给
            # align_score_by_keys(必抛),而是置 baseline_comparable=False
            # 并**保留未对齐的原始分数**继续往下传:evaluator 的
            # baseline 分支以 `baseline_score is not None` 为入口,
            # 入口处再按 baseline_comparable 分流(comparable=False 即
            # 跳过 align、走 `delta.vs_baseline=None` +
            # `baseline_incomparable=True`,evaluator.py:272-274)。
            # 若在这里把 base_score 置 None,那个既有分支根本进不去,
            # 事故会从"invalid"变成"静默无 vs_baseline 字段"—— 同样
            # 是失真,只是更难发现。
            if base_df is not None:
                base_score = base_df["score"].values
                if not baseline_view_compatible(view, baseline_view):
                    baseline_comparable = False
                    # base_df 置 None:不让 _pairing_keys 把旧键传下去
                    # (evaluator 在 comparable=False 时不消费它们)
                    base_df = None
                    logger.info(
                        "%s:baseline 属 %s 视图,trial 属 %s —— 跨视图"
                        "不可配对,记 baseline_incomparable(不再裁 invalid)",
                        trial.trial_id, baseline_view or "v1", view)
                else:
                    try:
                        from .metrics import align_score_by_keys

                        base_score = align_score_by_keys(
                            base_score, score_df[SAMPLE_KEY_COLUMN].values,
                            base_df[SAMPLE_KEY_COLUMN].values)
                        base_df = pd.DataFrame({
                            "score": base_score,
                            SAMPLE_KEY_COLUMN:
                                score_df[SAMPLE_KEY_COLUMN].values})
                    except Exception as exc:
                        logger.warning("%s:baseline 预测不可配对:%s",
                                       trial.trial_id, exc)
                        # evaluator 保持 comparable=True 并将身份不一致
                        # 裁为 invalid,禁止静默跳过后按 root trial 接受。

        slices = None
        slice_parent = None
        if slice_columns:
            raw = slice_by_columns(valid, self.spec, slice_columns)
            slices = {k: {"y_true": y[v["mask"]], "score": score[v["mask"]],
                          **({"treatment": treatment[v["mask"]]}
                             if treatment is not None else {})}
                      for k, v in raw.items()}
            if parent_score is not None and parent_alignment_ok:
                if len(parent_score) != len(y):
                    # v16/P2-4:跨数据视图的父代分数与当前 validation
                    # 行数不一致(实测 400w gen-01:sample 面 trial 配
                    # v1 父代分数,slice_parent 掩码处 IndexError,
                    # evaluator_error×2)——与 baseline 分支同口径,
                    # 跳过切片对照而非崩溃。
                    logger.warning(
                        "%s: 父代分数 %d 行 != 当前视图 validation %d 行"
                        "(跨数据视图),slice 对照跳过",
                        trial.trial_id, len(parent_score), len(y))
                    slice_parent = None
                else:
                    slice_parent = {k: parent_score[v["mask"]]
                                    for k, v in raw.items()}

        metrics_path = tdir / "metrics.json"
        metrics = load_json(metrics_path) if metrics_path.exists() else {}
        # v18/P2-5:temporal_evaluation 接线(死代码激活)。按时间列的
        # 月份切片构造逐期分数包(parent_score 同切),评估器输出
        # worst_delta/passed——时序回归是 time_based split 任务的主风险,
        # 此前 runner 从不传 temporal_scores,evaluation.json 恒 {}。
        temporal_packs = None
        tcol = self.spec.time_column
        if tcol and tcol in valid.columns and parent_score is not None \
                and parent_alignment_ok \
                and len(parent_score) == len(y):
            try:
                import pandas as _pd

                _tkey = _pd.to_datetime(valid[tcol]).dt.strftime("%Y-%m")
                temporal_packs = []
                for _m in sorted(_tkey.dropna().unique()):
                    _mask = (_tkey == _m).values
                    if _mask.sum() < 50:
                        continue  # 期太小,单期指标噪声大
                    temporal_packs.append({
                        "y_true": y[_mask],
                        "score": score[_mask],
                        "parent_score": parent_score[_mask],
                        **({"treatment": treatment[_mask]}
                           if treatment is not None else {}),
                    })
            except Exception as exc:  # noqa: BLE001
                logger.warning("temporal 分月切片构造失败(不阻塞): %s",
                               exc)
        complexity = {
            "feature_count": metrics.get("n_features"),
            # model_num:Trial 交付的模型数(单模型=1,集成>1),
            # 由 Trial 代码在 metrics.json 声明;缺省按单模型口径记 1
            "model_num": int(metrics["model_num"]) if metrics.get(
                "model_num") else 1,
            # learning curve 口径(v8):早停轮次与实际训练轮数
            # (旧 trial metrics.json 无键 → None,容忍)
            "best_iteration": metrics.get("best_iteration"),
            "n_estimators_used": metrics.get("n_estimators_used"),
        }
        # v2.15.0:两段式早停的第一段遥测(_fit_with_es)。cap_bound=True
        # 表示第一段实训轮数达到 cap、早停 patience 未到期 —— 训练被轮数
        # 上限截断(run3 的搜索冠军 021、交付 022 与交付模型都是这种
        # 形态,却被反思写成"早停真实触发")。只做留痕,不改裁决。
        # v2.16.3(D-011):cap_bound 改由 runtime.hpo.unified_cap_bound 按
        # **单一语义**(实训轮数 ≥ cap;两段式任一子模型撞即 True)从全部遥测
        # 重算,不再照抄 es_info 的旗标 —— 旧 baseline 血缘的 es_info 用
        # best ≥ 0.95×cap,hurdle 遥测用 rounds ≥ cap,同一 trial 两处相反。
        _es_meta = metrics.get("es_info") if isinstance(
            metrics.get("es_info"), dict) else {}
        try:
            from .hpo import unified_cap_bound

            _ucb = unified_cap_bound(metrics)
        except Exception as _exc:  # noqa: BLE001 遥测不阻塞评估
            logger.warning("cap_bound 统一判定失败(不阻塞): %s", _exc)
            _ucb = {}
        if _es_meta or _ucb.get("cap_bound") is not None:
            _seg_caps = [s.get("cap") for s in (_ucb.get("segments") or {})
                         .values() if s.get("cap")]
            complexity.update({
                "cap": _es_meta.get("cap") or (_seg_caps[0] if _seg_caps
                                               else None),
                "cap_bound": bool(_ucb.get("cap_bound")) if _ucb else
                bool(_es_meta.get("cap_bound")),
                "es_stopped_by": (_ucb.get("stopped_by")
                                  or _es_meta.get("stopped_by")),
                "cap_bound_source": _ucb.get("source"),
                "near_cap": _ucb.get("near_cap"),
                "cap_extended_to": _es_meta.get("cap_extended_to"),
                "stage1_rounds": _es_meta.get("stage1_rounds"),
                # v2.17.2(A1):余量内外扩后仍撞 cap(评估侧 cap_confounded 判据)
                "cap_bound_unresolved": _es_meta.get("cap_bound_unresolved"),
            })
        # v2.16.3:HPO 的生效 lr 模式写进 trial 记录(trials/<id>/trial.yaml 的
        # hpo 节;config/ 已冻结不可改)。创建期预判在 config 的
        # hpo.lr_mode_effective,这里是运行期实测值,二者不同即说明预判失准
        if metrics.get("hpo_lr_mode") or metrics.get("hpo_lr_mode_effective"):
            trial.data["hpo"] = {
                **(trial.data.get("hpo") if isinstance(
                    trial.data.get("hpo"), dict) else {}),
                "lr_mode_effective": (metrics.get("hpo_lr_mode_effective")
                                      or metrics.get("hpo_lr_mode")),
                "lr_mode_requested": metrics.get("hpo_lr_mode_requested"),
                "lr_mode_note": metrics.get("hpo_lr_mode_note"),
            }
        # 根语义由 lineage 决定，不能因父代分数错位/缺失就降级成 root
        # 并绕过提升证据门。
        is_root = not bool(trial.parent_ids)
        _eval_timings["load_and_align"] = round(time.time() - _t_eval0, 2)
        _t_ev = time.time()
        # v2.14.0:confirm 判据的 run 级输入(同配方复现组 σ_seed、
        # multi-seed 证据),见 runtime/confirmation.py
        _confirm_ctx = self._confirm_context(trial, view) \
            if _is_confirm else None
        _anchor = _anchor or {}
        evaluation = self.evaluator.evaluate(
            y, score,
            treatment=treatment,
            parent_score=parent_score,
            baseline_score=base_score,
            baseline_comparable=baseline_comparable,
            is_root_trial=is_root,
            confirm_trial=_is_confirm,
            multi_seed_scores=self._collect_multi_seed_scores(trial, y),
            slices=slices,
            slice_parent_scores=slice_parent,
            temporal_scores=temporal_packs,
            resource={"runtime_sec": (trial.data.get("result") or {})
                      .get("runtime_sec"),
                "peak_memory_gb": (trial.data.get("result") or {})
                .get("peak_memory_gb")},
            complexity=complexity,
            confirm_context=_confirm_ctx,
            # v2.17.0:锚定晋级 / 回归诊断分期 / 量纲感知 MDE
            anchor_score=_anchor.get("score"),
            anchor_id=_anchor.get("anchor_id"),
            anchor_same_as_parent=bool(_anchor.get("same_as_parent")),
            anchor_note=_anchor.get("note"),
            periods=self._regression_periods(valid),
            run_dir=self.run_dir,
            # v2.19.0:配对 SE 计入 seed 方差(confirm 不计入,其 margin 已含)
            # v2.19.1:n_val 进 σ_seed 规模先验
            seed_sd_context=(None if _is_confirm
                             else self._seed_sd_context(trial, view,
                                                        n_val=len(y))),
            **_pairing_keys(score_df, parent_df, base_df,
                            baseline_comparable, parent_alignment_ok),
        )
        if _confirm_ctx is not None:
            evaluation["multi_seed_evidence"] = _confirm_ctx.get("multi_seed")
        _eval_timings["evaluator_bootstrap"] = round(time.time() - _t_ev, 2)
        evaluation["integrity"]["prediction"] = prediction_meta
        evaluation["trial_id"] = trial.trial_id
        # v2.17.0:正式 vs_parent 的父代身份与 crossover IUT 明细
        _dl = evaluation.setdefault("delta", {})
        if _anchor.get("source"):
            _dl["anchor_source"] = _anchor["source"]
        if _iut_rows is not None:
            _dl["crossover_parent_test"] = "iut"
            _dl["vs_parents_iut"] = _iut_rows
        if _vs_parent_id and "vs_parent" in _dl:
            _dl["vs_parent_id"] = _vs_parent_id
        # ---- 新列质量审计(G2,v12.1)----
        # 依据:v3-4 实测——自写解析器产出退化列(近端窗口计数恒为常数),
        # 联合 AUC 阴性被采信为"方向无效"。trial 在 metrics.json 声明
        # new_column_quality(模板注入);harness 消费并置
        # implementation_quality 标签,供反思把"实现缺陷"与"方向无效"
        # 区分开。缺省字段 → unverified。
        ncq = metrics.get("new_column_quality")
        if isinstance(ncq, dict) and ncq:
            total = int(ncq.get("total") or 0)
            degen = int(ncq.get("degenerate") or 0)
            share = (degen / total) if total else None
            if share is not None and total > 0 and \
                    share > float(self.config.get(
                        "evaluator.degenerate_max_share", 0.40)):
                evaluation["implementation_quality"] = {
                    "status": "suspect",
                    "new_columns_total": total,
                    "degenerate_columns": degen,
                    "degenerate_share": round(share, 4),
                    "note": "退化新列占比超阈:阴性结论不可直接采信为"
                            "'方向无效',必须先归因实现质量(G1 契约)",
                }
                logger.warning(
                    "%s: 新增列退化占比 %.0f%%(%d/%d)> 阈值,"
                    "implementation_quality=suspect",
                    trial.trial_id, share * 100, degen, total)
            else:
                evaluation["implementation_quality"] = {
                    "status": "ok",
                    "new_columns_total": total,
                    "degenerate_columns": degen,
                    "degenerate_share": round(share, 4) if share is not None
                    else None,
                }
        # ---- v13 泄漏哨兵消费(leak_scan)----
        # trial 在 metrics.json 声明 leak_scan(模板内置:top-K gain 特征
        # 的 train 单变量 AUC)。suspects 非空 = 模型可能从特征里重建了
        # 标签/极端未来信息 → verdict 强制 invalid(leakage_suspect),
        # 升级到人裁决。这不是唯一防线(物理隔离防弱代理),哨兵只兜
        # 动态重建类;见 SKILL.md v13 泄漏防御。
        ls = metrics.get("leak_scan")
        if isinstance(ls, dict) and ls.get("suspects") and self.config.get(
                "evaluator.leak_scan_respect_adjudication", False):
            ls = self._waive_adjudicated_suspects(ls, evaluation)
        if isinstance(ls, dict) and ls.get("suspects"):
            evaluation["leakage_suspects"] = ls["suspects"]
            evaluation["verdict"] = "invalid"
            evaluation["confidence"] = "low"
            evaluation["reason"] = (
                "leak_scan 哨兵命中:top gain 特征的 train 单变量 AUC>0.9"
                "(分类)/ |Spearman ρ|>0.95(回归,v2.16.2)"
                f"({ls['suspects'][:3]});疑似标签/未来信息重建,"
                "人工裁决前不得进 leaderboard 有效行")
            logger.warning(
                "%s: leak_scan suspects=%s → invalid(leakage_suspect)",
                trial.trial_id, ls["suspects"][:3])
        # ---- v13 gain 集中度标记(代理特征吞噬模型的提示,不否决)----
        # top-1 gain share>5% 且 >5×次名 → 标记进 evaluation,反思必答
        # "该特征信息流是否可追溯到合法源"。req-t7 实测:泄漏代理独占
        # 5.85% gain(次名 11 倍差距)——形态本身就是信号。
        try:
            fi_path = tdir / "feature" / "feature_importance.csv"
            if fi_path.exists():
                fi = pd.read_csv(fi_path)
                if len(fi) >= 2 and fi["importance"].iloc[0] > 0:
                    top_share = float(fi["importance"].iloc[0]
                                      / fi["importance"].sum())
                    ratio = float(fi["importance"].iloc[0]
                                  / max(fi["importance"].iloc[1], 1e-12))
                    if top_share > 0.05 and ratio > 5.0:
                        evaluation["gain_concentration"] = {
                            "feature": str(fi["feature"].iloc[0]),
                            "top_share": round(top_share, 4),
                            "ratio_vs_second": round(ratio, 2),
                            "note": "top-1 gain 独占>5% 且 >5×次名:"
                                    "反思必须核查该特征的信息流是否可追溯"
                                    "到合法源(泄漏代理的典型形态,"
                                    "req-t7 实测回写)",
                        }
        except Exception as exc:  # noqa: BLE001
            logger.debug("gain 集中度检查跳过 %s: %s", trial.trial_id, exc)
        # ---- 分组置换重要性(v11):超级 trial 方向生死的无偏裁决 ----
        # gain 是共线性下的模型内分配量;置换量"打乱该族,validation
        # 掉多少"。模型工件与特征列齐备时自动产出
        # feature/group_importance.csv(摘要进 evaluation)。
        _gi_policy = str(self.config.get("evaluator.group_importance_policy",
                                         "all") or "all").lower()
        _t_gi = time.time()
        _gi_inherited = None
        # v2.15:gen-00(baseline + warm trial)跳过 GI。
        # baseline 只建锚点不提出方向;warm trial 的方向先验已由历史 recipe
        # 编码(rank1/2 能进 top-3 就是族价值的实证)。400w 上 GI 单 trial
        # ~40min,三个 gen-00 trial 全算 ≈2 GPU 小时纯浪费。
        if bool(self.config.get(
                "evaluator.skip_group_importance_on_generation_0", True)) \
                and int(trial.generation or 0) == 0:
            evaluation["group_importance"] = {
                "skipped": True,
                "reason": "gen-00 baseline/warm trial,方向先验来自历史 recipe",
            }
        elif _gi_policy in ("auto", "skip_model_only"):
            # v2.15.0:只改模型的 trial(hpo/confirm/changed_modules=[model])
            # 特征面与父代相同,族级置换的生死结论直接继承父代。run3 的
            # 4 个 HPO + 3 个 confirm 各花 200~800s 重算同一特征面的
            # group_importance(合计约 1 GPU 小时),结论与父代一致。
            _gi_inherited = self._gi_inherit_from_parent(trial)
        if _gi_inherited is not None:
            evaluation["group_importance"] = _gi_inherited
        elif _gi_policy == "skip_rejected" and \
                evaluation.get("verdict") in ("rejected", "invalid",
                                              "unconfirmed"):
            # v2.13.5:被拒 trial 按配置跳过置换(显式留痕,不是信号崩了)
            evaluation["group_importance"] = {
                "skipped": True,
                "reason": f"group_importance_policy=skip_rejected,"
                          f"verdict={evaluation.get('verdict')}"}
        else:
            try:
                gi_summary = self._run_group_importance(trial, valid, y,
                                                        treatment)
                if gi_summary is not None:
                    evaluation["group_importance"] = gi_summary
            except Exception as exc:  # noqa: BLE001
                logger.warning("分组置换重要性失败 %s: %s",
                               trial.trial_id, exc)
                # v14:置换 delta 是族生死的第一裁决口径(优先于 gain),
                # 缺失必须在产物里可见,不能只留一行 WARNING —— 否则反思
                # 环节看到的是"没有这个信号",而不是"这个信号崩了"。
                evaluation["group_importance"] = {
                    "skipped": True, "error": str(exc)[:200]}
        _eval_timings["group_importance"] = round(time.time() - _t_gi, 2)
        _t_ea = time.time()
        # ---- 误差归因(v11):把「错在哪」翻译成「该造什么特征」 ----
        try:
            from .error_analysis import (build_error_analysis,
                                         non_feature_columns,
                                         summarize_for_evaluation,
                                         write_error_analysis)

            # v2.17.0(D1):画像只用特征面列 —— 标签面(以 labels parquet
            # 列集为准)+ 目标/处置/时间/ID + join 后缀列一律排除
            ea = build_error_analysis(
                y, score, parent_score, valid,
                model_type=self.spec.model_type, treatment=treatment,
                n_bins=int(self.config.get("evaluator.segment_bins", 10)),
                label=trial.trial_id,
                exclude_columns=non_feature_columns(
                    self.spec, list(_va_lab.columns)))
            write_error_analysis(tdir, ea)
            evaluation["error_analysis"] = summarize_for_evaluation(ea)
        except Exception as exc:  # noqa: BLE001
            logger.warning("误差归因失败 %s: %s", trial.trial_id, exc)
        _eval_timings["error_analysis"] = round(time.time() - _t_ea, 2)
        _t_seg = time.time()
        # ---- 分桶报表(xlsx + md):validation 全量分桶交付视图 ----
        try:
            from .segment_report import build_segment_report

            is_reg = self.spec.model_type == "regression"
            seg_rep = build_segment_report(
                y, score, self.spec,
                treatment=treatment,
                n_bins=int(self.config.get("evaluator.segment_bins", 10)),
                y_pred=score if is_reg else None,
                out_dir=tdir,
                extra_summary={
                    "trial_id": trial.trial_id,
                    "algorithm": trial.data.get("algorithm"),
                    "split": "validation",
                },
                label=trial.trial_id)
            if seg_rep is not None:
                evaluation["segment_report"] = {
                    "xlsx": "segment-report.xlsx",
                    "md": "segment-report.md",
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("分桶报表产出失败 %s: %s", trial.trial_id, exc)
        _eval_timings["segment_report"] = round(time.time() - _t_seg, 2)
        _t_sh = time.time()
        # 所有 shadow 预测都先做**无标签身份绑定**，以便未来 confirm
        # 能把它作为父代对照。这里只投影读取 sample key 列，不读取目标、
        # 不算指标、不写 evaluation，因此普通搜索仍得不到 shadow 反馈。
        shadow_path = tdir / "shadow_test_score.parquet"
        _shadow_label_path = labels_dir / "shadow_test.parquet"
        _shadow_identity_meta = None
        if shadow_path.exists() and _shadow_label_path.exists():
            try:
                _shadow_keys_only = pd.read_parquet(
                    _shadow_label_path, columns=[SAMPLE_KEY_COLUMN])
                _shadow_identity_meta = bind_prediction_artifact(
                    shadow_path, _shadow_keys_only,
                    source_manifest=source_manifest,
                    split_manifest_path=view_splits_dir(
                        self.run_dir, view) / "manifest.json",
                    data_view=str(view), model_paths=model_files)
            except Exception as exc:  # 身份不可信时不能静默留给 confirm
                evaluation.setdefault("integrity", {}).setdefault(
                    "errors", []).append(
                        f"shadow_prediction_identity:{exc}")
                logger.warning("%s:shadow 预测身份绑定失败:%s",
                               trial.trial_id, exc)
        # shadow 是独立确认面，只允许 confirm trial 消费。普通候选即使
        # 偷偷产出了 shadow 分数也不进入评估/选型反馈。
        _shadow_policy = str(self.config.get(
            "evaluator.shadow_policy", "confirmation_only"))
        _allow_shadow = (include_shadow and _is_confirm and
                         self.config.get("evaluator.shadow_test_enabled", True)
                         and _shadow_policy == "confirmation_only")
        if _allow_shadow:
            shadow = frames.get("shadow_test")
            if shadow_path.exists() and shadow is not None:
                try:
                    from .metrics import (align_score_by_keys, compute_metric,
                                          paired_bootstrap_delta)

                    _sh_lab_path = labels_dir / "shadow_test.parquet"
                    if not _sh_lab_path.exists():
                        raise FileNotFoundError("shadow_test 标签面缺失")
                    _sh_lab = pd.read_parquet(_sh_lab_path)
                    if SAMPLE_KEY_COLUMN not in _sh_lab.columns:
                        _sh_lab[SAMPLE_KEY_COLUMN] = sample_keys(
                            _sh_lab, self.spec).values
                        _sh_lab.to_parquet(_sh_lab_path, index=False)
                    if _shadow_identity_meta is None:
                        raise PermissionError(
                            "shadow 预测未通过无标签样本身份绑定")
                    _shadow_meta = _shadow_identity_meta
                    _sh_df = pd.read_parquet(shadow_path)
                    sscore = _sh_df["score"].values
                    shadow = shadow.join(
                        _sh_lab.set_index(shadow.index), how="left",
                        rsuffix="_lab")

                    # uplift:shadow 同样需要 treatment 归属
                    st_treat = None
                    scol = self.spec.treatment_column
                    if scol is not None and scol in shadow.columns:
                        st_treat = shadow[scol].values
                    evaluation["shadow_test"] = {
                        self.spec.primary_metric: compute_metric(
                            self.spec.primary_metric,
                            shadow[self.spec.target_name].values, sscore,
                            score=sscore, treatment=st_treat),
                    }
                    # ---- shadow gap sanity(v8):validation ↔ shadow 距离 ----
                    # passed=False 提示过拟合/分布漂移风险(附加诊断,
                    # 不阻塞 verdict;反思必须覆盖,见 SKILL.md shadow 节)
                    valid_val = (evaluation.get("primary_metric") or {}
                                 ).get("value")
                    shadow_val = evaluation["shadow_test"].get(
                        self.spec.primary_metric)
                    if valid_val is not None and shadow_val is not None:
                        gap = abs(float(shadow_val) - float(valid_val))
                        gap_rel = gap / max(abs(float(valid_val)), 1e-9)
                        noise = evaluation.get("noise")
                        mult = float(self.config.get(
                            "evaluator.shadow_gap_multiplier", 2.0))
                        rel_max = float(self.config.get(
                            "evaluator.shadow_gap_relative_max", 0.15))
                        if noise is not None and np.isfinite(noise) \
                                and noise > 0:
                            passed = gap <= mult * noise
                        else:
                            passed = gap_rel <= rel_max
                        # v2.17.2(C4):诊断位改名 gap_sanity_passed 并标
                        # criterion=gap_sanity——旧键 passed 与确认门
                        # fresh_confirmation.passed 同名,被误读成"shadow
                        # 确认通过"。读取侧兼容旧键(见 evolution/report)
                        evaluation["shadow_test"].update({
                            "gap_vs_validation": gap,
                            "gap_relative": gap_rel,
                            "criterion": "gap_sanity",
                            "gap_sanity_passed": bool(passed),
                        })
                    _parent_shadow = parent_scores.get("shadow_test") \
                        if isinstance(parent_scores, dict) else None
                    if not isinstance(_parent_shadow, pd.DataFrame):
                        raise ValueError(
                            "confirmation 需要父代 shadow_test_score 作为独立复现对照")
                    _old = align_score_by_keys(
                        _parent_shadow["score"].values,
                        _sh_df[SAMPLE_KEY_COLUMN].values,
                        _parent_shadow[SAMPLE_KEY_COLUMN].values)
                    _pb = paired_bootstrap_delta(
                        shadow[self.spec.target_name].values, sscore, _old,
                        self.spec.primary_metric, treatment=st_treat,
                        n_boot=int(self.config.get(
                            "evaluator.bootstrap_samples", 2000)))
                    _shadow_delta = evaluation["shadow_test"][
                        self.spec.primary_metric] - compute_metric(
                            self.spec.primary_metric,
                            shadow[self.spec.target_name].values,
                            _old if self.spec.model_type == "regression" else None,
                            score=_old, treatment=st_treat)
                    # v2.14.0:shadow 侧与 validation 同口径(非劣 + 保真,
                    # margin 含 seed 方差);σ̂ 沿用 validation 侧估计,
                    # floor 按 shadow 自己的 SE 重算。旧口径 |Δ|≤2·SE
                    # 只作诊断字段留痕。
                    _cc = confirm_config(self.config, run_dir=self.run_dir)
                    _cv = evaluation.get("confirmation_validation") or {}
                    _se = _pb.get("se")
                    _sig_sh = sigma_for_split(
                        {"estimate": _cv.get("sigma_seed_estimate"),
                         "estimate_source": _cv.get(
                             "sigma_seed_estimate_source"),
                         "df": _cv.get("sigma_seed_df")},
                        _se, _cc["floor_ratio"])
                    _m_sh = confirmation_margins(
                        se=_se, sigma_seed=_sig_sh.get("sigma_seed"),
                        n_child=1, z_lower=_cc["z_lower"],
                        z_upper=_cc["z_upper"], cfg_margin=_cc["cfg_margin"])
                    _hb = metric_direction(self.spec.primary_metric)
                    if _m_sh is not None:
                        _j_sh = judge_delta(float(_shadow_delta), _m_sh, _hb)
                    else:
                        _pt = abs(float(_shadow_delta)) < 1e-12
                        _j_sh = {"delta": float(_shadow_delta),
                                 "non_inferior": _pt, "fidelity_ok": _pt,
                                 "passed": _pt}
                    _prior_verdict = evaluation.get("verdict")
                    _validation_ok = _prior_verdict == "confirmed"
                    _ref = self._confirm_reference_check(
                        trial, shadow, sscore, _sh_df, st_treat,
                        sigma_info=_cv, cc=_cc, higher_better=_hb) \
                        if _cc["reference_check"] else {
                            "available": False, "reason": "disabled"}
                    _ref_ok = (not _ref.get("available")) or bool(
                        _ref.get("non_inferior"))
                    _passed = bool(_validation_ok and _j_sh["passed"]
                                   and _ref_ok)
                    _fail: list[str] = []
                    if not _validation_ok:
                        _fail.append(
                            f"validation 侧未通过({_prior_verdict}:"
                            f"{evaluation.get('reason')})")
                    if not _j_sh["non_inferior"]:
                        _fail.append(
                            f"shadow 非劣失败:Δ={float(_shadow_delta):+.6f}"
                            f" < −{(_m_sh or {}).get('lower')}")
                    if not _j_sh["fidelity_ok"]:
                        _fail.append(
                            f"shadow 保真失败:Δ={float(_shadow_delta):+.6f}"
                            f" > +{(_m_sh or {}).get('upper')}(子代远优于"
                            "父代,不是同配方复现)")
                    if not _ref_ok:
                        _fail.append(
                            f"参照 {_ref.get('reference_trial')} 对照失败:"
                            f"shadow Δ={_ref.get('delta'):+.6f} < "
                            f"−{_ref.get('margin_lower')}(配方相对其参照在"
                            "独立数据上显著更差,增益不可复现)")
                    evaluation["fresh_confirmation"] = {
                        "criterion": CONFIRM_CRITERION,
                        "split": "shadow_test",
                        "independent_from_selection": True,
                        # 兼容键:v2.14.0 起语义 = validation 侧非劣+保真通过
                        "validation_equivalence_passed": _validation_ok,
                        "validation_passed": _validation_ok,
                        "validation": {k: _cv.get(k) for k in (
                            "delta_primary", "delta_used", "delta_basis",
                            "n_seeds", "se_eval", "sigma_seed",
                            "sigma_seed_source", "margin_lower",
                            "margin_upper", "non_inferior", "fidelity_ok",
                            "legacy_two_sided_equivalence")},
                        "delta_vs_parent": _shadow_delta,
                        "paired_bootstrap": _pb,
                        # 兼容键:= 非劣下界(旧语义为双侧等价半宽)
                        "equivalence_margin": (_m_sh or {}).get("lower"),
                        "margin_lower": (_m_sh or {}).get("lower"),
                        "margin_upper": (_m_sh or {}).get("upper"),
                        "se_eval": _se,
                        "sigma_seed": _sig_sh.get("sigma_seed"),
                        "sigma_seed_source": _sig_sh.get("source"),
                        "non_inferior": _j_sh["non_inferior"],
                        "fidelity_ok": _j_sh["fidelity_ok"],
                        "shadow_passed": _j_sh["passed"],
                        "legacy_two_sided_equivalence": legacy_two_sided(
                            float(_shadow_delta), _se, _cc["cfg_margin"]),
                        "reference": _ref,
                        "multi_seed": evaluation.get("multi_seed_evidence"),
                        "failure_reasons": _fail,
                        "passed": _passed,
                        "prediction_integrity": _shadow_meta,
                    }
                    if _prior_verdict == "invalid":
                        # leak_scan 等已裁 invalid:确认结论不得覆盖它
                        pass
                    elif _passed:
                        evaluation["verdict"] = "confirmed"
                        evaluation["confidence"] = "high"
                        evaluation["reason"] = (
                            "validation 与独立 shadow_test 非劣+保真确认通过"
                            "(margin 含 seed 方差)")
                    else:
                        evaluation["verdict"] = "unconfirmed"
                        evaluation["confidence"] = "high"
                        evaluation["reason"] = (
                            "独立确认未通过,禁止晋升:" + ";".join(_fail))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("shadow_test 评测失败 %s: %s",
                                   trial.trial_id, exc)
                    evaluation["fresh_confirmation"] = {
                        "split": "shadow_test", "passed": False,
                        "error": str(exc)}
                    if evaluation.get("verdict") != "invalid":
                        evaluation["verdict"] = "unconfirmed"
                        evaluation["confidence"] = "high"
                        evaluation["reason"] = "独立 shadow_test 确认不可用"
            else:
                evaluation["fresh_confirmation"] = {
                    "split": "shadow_test", "passed": False,
                    "error": "confirm trial 未产出 shadow_test_score"}
                if evaluation.get("verdict") != "invalid":
                    evaluation["verdict"] = "unconfirmed"
                    evaluation["confidence"] = "high"
                    evaluation["reason"] = "缺少独立 shadow_test 确认证据"
        _eval_timings["shadow"] = round(time.time() - _t_sh, 2)
        _eval_timings["total"] = round(time.time() - _t_eval0, 2)
        evaluation["timings"] = _eval_timings
        # v2.17.0:parent_verdict = vs_parent 口径的单试验 verdict(含
        # leak_scan 等评估后改写;代级校正/锚定晋级不改它)。confirm 不写。
        if not _is_confirm:
            evaluation["parent_verdict"] = evaluation.get("verdict")
        return evaluation

    def _waive_adjudicated_suspects(self, ls: dict, evaluation: dict) -> dict:
        """v2.19.0:leak_scan 哨兵尊重 spec 的泄漏裁决(leak_scan_respect_adjudication)。

        benchmark creditcard 实测:PCA 分量 V14/V10/V12 单变量 AUC 0.93~0.95
        是数据本身的强可分性;Agent 按复核清单裁为 kept_ambiguous(合法保留),
        但哨兵不读该裁决,22 个 trial 中 21 个判 invalid,run 无法交付。

        - 命中列在 ``features.leak_confirmation.kept_ambiguous`` 里 → 豁免;
        - 其余命中列的单变量 AUC 若不超过本次扫描中已豁免列的最高值 + 0.02
          (例如由合法列派生的交互列)→ 豁免;
        - 豁免明细写 ``evaluation.leakage_suspects_waived``,不静默放行;明显强于
          合法列的命中(标签重建的典型形态)照旧判 invalid。
        """
        # TaskSpec.kept_ambiguous_columns / leak_confirmation_note 是 property
        try:
            kept = set(self.spec.kept_ambiguous_columns or [])
        except Exception:  # noqa: BLE001 —— 裁决读不到就不豁免
            kept = set()
        if not kept:
            return ls
        sus = [x for x in (ls.get("suspects") or []) if isinstance(x, dict)]

        def _score(x):
            for k in ("train_auc", "train_abs_spearman"):
                if x.get(k) is not None:
                    try:
                        return float(x[k])
                    except (TypeError, ValueError):
                        return None
            return None

        waived = [x for x in sus if str(x.get("feature")) in kept]
        ref = max([v for v in (_score(x) for x in waived) if v is not None],
                  default=None)
        rest = []
        for x in sus:
            if x in waived:
                continue
            v = _score(x)
            if ref is not None and v is not None and v <= ref + 0.02:
                waived.append({**x, "waived_as": "derived_not_stronger_than_"
                               "adjudicated", "reference_score": ref})
            else:
                rest.append(x)
        if waived:
            evaluation["leakage_suspects_waived"] = {
                "suspects": waived,
                "basis": "features.leak_confirmation.kept_ambiguous",
                "note": getattr(self.spec, "leak_confirmation_note", None),
            }
        return {**ls, "suspects": rest}

    def _seed_sd_context(self, trial: TrialRecord, view: str,
                         n_val: int | None = None) -> dict | None:
        """v2.19.x:搜索 trial 配对检验的 σ_seed 输入(seed_variance_in_se)。

        与 confirm 同源:同视图同配方复现组的合并 seed 方差,排除当前
        trial。配置关闭时返回 None(evaluator 不改动 bootstrap 结果)。

        v2.19.1 增补:
        - ``n_val``:validation 行数,供 :func:`scaled_sigma_prior` 按规模
          给先验(σ_seed = r(n_val)·SE);
        - ``prior_ratio``:经验库同数据底座的历史比值(见
          :func:`runtime.library_view.seed_variance_ratio`),优先于公式。
          库未启用 / 无历史时不设,evaluator 退回公式。
        """
        if not self.config.get("evaluator.seed_variance_in_se", False):
            return None
        from .confirmation import pooled_replicate_sd

        ctx: dict = {}
        if n_val is not None:
            try:
                ctx["n_val"] = int(n_val)
            except (TypeError, ValueError):
                pass
        try:
            others = []
            for d in sorted((self.run_dir / "trials").iterdir()):
                if d.name == trial.trial_id or not (d / "trial.yaml").exists():
                    continue
                try:
                    others.append(TrialRecord.load(d))
                except Exception:  # noqa: BLE001  坏记录不阻塞评估
                    continue
            rep = pooled_replicate_sd(others, data_view=view)
            ctx.update(replicate_sd=rep["sd"], replicate_df=rep["df"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s:σ_seed 复现组统计失败(回退先验/floor):%s",
                           trial.trial_id, exc)
        # v2.19.1:经验库同数据底座的历史 σ_seed/SE 比值(库未启用时为空)
        try:
            from .library_view import seed_variance_ratio

            r = seed_variance_ratio(self.run_dir)
            if r is not None:
                ctx["prior_ratio"] = float(r["ratio"])
                ctx["prior_ratio_source"] = r.get("source")
        except Exception:  # noqa: BLE001  库视图缺失不影响裁决
            pass
        return ctx

    def _confirm_context(self, trial: TrialRecord, view: str) -> dict:
        """confirm 判据的 run 级输入(v2.14.0)。

        - 同配方复现组的合并 seed 方差(**排除当前 trial**,否则它与父代
          的差正是被检验量,纳入即自证通过);
        - multi-seed 证据状态:声明了却没产出必须可见,不能静默退化
          (run2 实测:confirm 声明 multi_seed=3,复制来的 baseline 血缘
          train.py 没有多 seed 段,seed_scores/ 从未产出,无人知晓)。
        """
        from .confirmation import pooled_replicate_sd

        ctx: dict = {}
        try:
            others = []
            for d in sorted((self.run_dir / "trials").iterdir()):
                if d.name == trial.trial_id or not (d / "trial.yaml").exists():
                    continue
                try:
                    others.append(TrialRecord.load(d))
                except Exception:  # noqa: BLE001  坏记录不阻塞评估
                    continue
            rep = pooled_replicate_sd(others, data_view=view)
            ctx.update(replicate_sd=rep["sd"], replicate_df=rep["df"],
                       replicate_groups=rep["groups"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s:同配方复现组统计失败(σ_seed 回退先验/"
                           "floor):%s", trial.trial_id, exc)
        ctx["multi_seed"] = self._multi_seed_evidence(trial)
        ms = ctx["multi_seed"]
        if ms.get("status") in ("missing", "partial"):
            logger.warning(
                "%s:confirm 声明 multi_seed=%s 但只找到 %s 个 seed 变体"
                "(%s)——σ_seed 将回退到同配方复现组/先验/floor。trial 代码"
                "需包含模板的 multi-seed 段(写 seed_scores/seed-k.parquet)",
                trial.trial_id, ms.get("declared"), ms.get("found"),
                ms.get("status"))
        return ctx

    def _multi_seed_evidence(self, trial: TrialRecord) -> dict:
        """声明的 multi_seed 与实际产出的 seed 变体数对账。

        v2.16.2(run4 regression D-006):confirm 脚手架曾把 config.multi_seed
        写成 1,而 trial 代码对 confirm 默认实跑 3 个 seed(metrics.multi_seed_n=3,
        seed_scores/ 落了 2 个变体)→ status=not_declared,证据被误报为"未声明"。
        现以"config 声明"与"代码实跑(metrics.multi_seed_n)"取大者为声明数;
        config 原值另记 declared_config。σ_seed 一直按落盘 seed_scores 计算,
        本函数只影响证据状态字段。
        """
        tdir = Path(trial.trial_dir)
        declared_cfg = 1
        try:
            from .logging_utils import load_yaml

            cfg = load_yaml(tdir / "config" / "trial.yaml") or {}
            declared_cfg = int(cfg.get("multi_seed")
                               or (cfg.get("hyperparameters") or {}).get(
                                   "multi_seed") or 1)
        except Exception:  # noqa: BLE001
            pass
        executed = 0
        try:
            _m = json.loads((tdir / "metrics.json").read_text(
                encoding="utf-8")) if (tdir / "metrics.json").exists() else {}
            executed = int((_m or {}).get("multi_seed_n") or 0)
        except Exception:  # noqa: BLE001
            executed = 0
        declared = max(declared_cfg, executed)
        seed_dir = tdir / "seed_scores"
        found = len(list(seed_dir.glob("*.parquet"))) \
            if seed_dir.is_dir() else 0
        expected = max(0, declared - 1)
        if declared <= 1:
            status = "not_declared"
        elif found >= expected:
            status = "complete"
        elif found > 0:
            status = "partial"
        else:
            status = "missing"
        return {"declared": declared, "declared_config": declared_cfg,
                "executed_by_code": executed or None,
                "expected_variants": expected,
                "found": found, "status": status}

    def _confirm_reference_check(self, trial: TrialRecord,
                                 shadow: pd.DataFrame, sscore: np.ndarray,
                                 sh_df: pd.DataFrame, st_treat,
                                 *, sigma_info: dict, cc: dict,
                                 higher_better: bool) -> dict:
        """子代 vs 父代的参照(父代当初的比较对象)在 shadow 上的对照。

        confirm 回答两个问题:① 换 seed 后能否达到父代水平(非劣);
        ② 父代宣称的增益在独立数据上是否还在。②只能对照父代的参照
        (父代的父代)来看 —— run2 的 D-07 恰恰想测它,旧 harness 没有
        入口。参照在 shadow 上**显著更好**于子代(超出同口径 margin)
        即判失败;p_better≥0.95 记 gain_reproduced(诊断,不作门)。

        v2.17.0:父代经**锚定晋级**(selection_correction.reference=anchor
        且 passed)时,父代当初的比较对象是交付锚点而不是它的父代 ⇒ 参照
        = ``selection_correction.anchor_id``(shadow 上子代不得显著差于锚点);
        锚点缺 shadow 分数/不同视图时与既有缺失处理同口径:跳过并留痕。
        """
        from .metrics import (align_score_by_keys, compute_metric,
                              paired_bootstrap_delta)

        parent_dir = self.run_dir / "trials" / trial.parent_ids[0]
        try:
            parent = TrialRecord.load(parent_dir)
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "reason": f"父代记录不可读:{exc}"}
        _sc = ((parent.data.get("evaluation") or {}).get(
            "selection_correction") or {})
        if _sc.get("reference") == "anchor" and _sc.get("passed") is True \
                and _sc.get("anchor_id"):
            ref_id, ref_kind = str(_sc["anchor_id"]), "anchor"
        else:
            if len(parent.parent_ids) != 1:
                return {"available": False, "reference_kind": "grandparent",
                        "reason": "父代为根 trial 或多父代(crossover),无单一参照"}
            ref_id, ref_kind = parent.parent_ids[0], "grandparent"
        try:
            ref_scores = load_parent_scores(self.run_dir, ref_id)
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "reference_trial": ref_id,
                    "reference_kind": ref_kind,
                    "reason": f"参照分数完整性校验失败:{exc}"}
        ref_sh = (ref_scores or {}).get("shadow_test")
        if not isinstance(ref_sh, pd.DataFrame):
            return {"available": False, "reference_trial": ref_id,
                    "reference_kind": ref_kind,
                    "reason": "参照无 shadow_test_score"}
        if str((ref_scores or {}).get("data_view") or "v1") != str(
                trial.data.get("data_view") or "v1"):
            return {"available": False, "reference_trial": ref_id,
                    "reference_kind": ref_kind,
                    "reason": "参照与子代不同数据视图,不可配对"}
        y_sh = shadow[self.spec.target_name].values
        ref = align_score_by_keys(ref_sh["score"].values,
                                  sh_df[SAMPLE_KEY_COLUMN].values,
                                  ref_sh[SAMPLE_KEY_COLUMN].values)
        pb = paired_bootstrap_delta(
            y_sh, sscore, ref, self.spec.primary_metric, treatment=st_treat,
            n_boot=int(self.config.get("evaluator.bootstrap_samples", 2000)))
        m = self.spec.primary_metric
        is_reg = self.spec.model_type == "regression"
        delta = compute_metric(m, y_sh, sscore if is_reg else None,
                               score=sscore, treatment=st_treat) \
            - compute_metric(m, y_sh, ref if is_reg else None, score=ref,
                             treatment=st_treat)
        sig = sigma_for_split(
            {"estimate": sigma_info.get("sigma_seed_estimate"),
             "estimate_source": sigma_info.get("sigma_seed_estimate_source"),
             "df": sigma_info.get("sigma_seed_df")},
            pb.get("se"), cc["floor_ratio"])
        margins = confirmation_margins(
            se=pb.get("se"), sigma_seed=sig.get("sigma_seed"), n_child=1,
            z_lower=cc["z_lower"], z_upper=cc["z_upper"],
            cfg_margin=cc["cfg_margin"])
        p_pos = pb.get("p_positive")
        p_better = (p_pos if higher_better else
                    (1.0 - p_pos if p_pos is not None else None))
        out = {"available": margins is not None, "reference_trial": ref_id,
               "reference_kind": ref_kind,
               "delta": float(delta), "paired_bootstrap": pb,
               "p_better": p_better,
               "gain_reproduced": bool(p_better is not None
                                       and p_better >= 0.95),
               "sigma_seed": sig.get("sigma_seed"),
               "sigma_seed_source": sig.get("source")}
        if margins is not None:
            j = judge_delta(float(delta), margins, higher_better)
            out.update(margin_lower=margins["lower"],
                       non_inferior=j["non_inferior"])
        return out

    # ---- v2.17.0:锚定晋级 / crossover IUT / 回归诊断分期(评估区)----
    def _crossover_iut(self, trial: TrialRecord, view: str,
                       score_df: pd.DataFrame, y: np.ndarray,
                       treatment) -> dict:
        """多父代 crossover 的交并检验(IUT):逐父代配对 bootstrap。

        返回 rows=[{parent_id, delta, se, p_better, z, available, ...}]、
        pick=p_better 最低(最保守;并列取方向感知 z 更小者)的父代及其
        分数包。不可用的父代(完整性失败/跨视图/不可配对)留痕不参与挑选。
        """
        from .metrics import (REGRESSION_METRICS, align_score_by_keys,
                              compute_metric, paired_bootstrap_delta)

        metric = self.spec.primary_metric
        is_reg = metric in REGRESSION_METRICS
        hb = metric_direction(metric)
        score = score_df["score"].values
        cur = compute_metric(metric, y, score if is_reg else None,
                             score=score, treatment=treatment)
        rows: list[dict] = []
        packs: dict = {}
        for pid in trial.parent_ids:
            row: dict = {"parent_id": pid}
            try:
                pack = load_parent_scores(self.run_dir, pid)
            except Exception as exc:  # noqa: BLE001
                row.update(available=False,
                           reason=f"父代分数完整性校验失败:{exc}"[:200])
                rows.append(row)
                continue
            pdf = (pack or {}).get("validation")
            if not isinstance(pdf, pd.DataFrame):
                row.update(available=False, reason="父代无 validation_score")
                rows.append(row)
                continue
            if not baseline_view_compatible(view, (pack or {}).get(
                    "data_view")):
                row.update(available=False, reason="父代与 trial 不同数据视图")
                rows.append(row)
                continue
            try:
                ps = align_score_by_keys(pdf["score"].values,
                                         score_df[SAMPLE_KEY_COLUMN].values,
                                         pdf[SAMPLE_KEY_COLUMN].values)
                pb = paired_bootstrap_delta(
                    y, score, ps, metric, treatment=treatment,
                    n_boot=int(self.config.get("evaluator.bootstrap_samples",
                                               2000)))
                d = float(cur - compute_metric(
                    metric, y, ps if is_reg else None, score=ps,
                    treatment=treatment))
            except Exception as exc:  # noqa: BLE001
                row.update(available=False,
                           reason=f"父代预测不可配对:{exc}"[:200])
                rows.append(row)
                continue
            p_pos = pb.get("p_positive")
            p_better = (p_pos if hb else
                        (1.0 - p_pos if p_pos is not None else None))
            se = pb.get("se")
            z = ((d if hb else -d) / float(se)) if se else None
            row.update(available=True, delta=d, se=se, p_better=p_better,
                       z=z, delta_ci=pb.get("delta_ci"))
            rows.append(row)
            packs[pid] = pack
        avail = [r for r in rows if r.get("available")
                 and r.get("p_better") is not None]
        if not avail:
            return {"rows": rows, "pick": None, "pick_pack": None}
        pick = min(avail, key=lambda r: (
            float(r["p_better"]),
            float(r["z"]) if r.get("z") is not None else float("inf")))
        for r in rows:
            r["selected"] = r is pick
        return {"rows": rows, "pick": pick["parent_id"],
                "pick_pack": packs[pick["parent_id"]]}

    def _promotion_anchor_scores(self, trial: TrialRecord, view: str,
                                 score_df: pd.DataFrame,
                                 vs_parent_id: str | None,
                                 *, parent_ok: bool) -> dict | None:
        """anchor 模式:解析本代锚点并按样本键对齐其 validation 分数。

        非 anchor 模式 / confirm / 根 trial / screening → None(不计算)。
        """
        from .evolution import promotion_reference, resolve_promotion_anchor
        from .metrics import align_score_by_keys
        from .trial import trial_is_screening

        if promotion_reference(self.config) != "anchor":
            return None
        if str(trial.data.get("search_action") or "") == "confirm" \
                or not trial.parent_ids or trial_is_screening(trial):
            return None
        try:
            info = resolve_promotion_anchor(self.run_dir, trial.generation,
                                            config=self.config)
        except Exception as exc:  # noqa: BLE001
            return {"anchor_id": None, "note": {
                "status": "unavailable", "reason": f"锚点解析失败:{exc}"[:200]}}
        aid = info.get("anchor_id")
        out: dict = {"anchor_id": aid, "score": None, "same_as_parent": False,
                     "note": None, "source": info.get("source")}
        if not aid:
            out["note"] = {"status": "unavailable",
                           "reason": "无交付锚点且 run-state 无 baseline_id"}
            return out
        if aid == trial.trial_id:
            out["note"] = {"status": "self", "reason": "锚点即本 trial"}
            return out
        if vs_parent_id and aid == vs_parent_id and parent_ok:
            out["same_as_parent"] = True
            return out
        try:
            pack = load_parent_scores(self.run_dir, aid)
        except Exception as exc:  # noqa: BLE001
            out["note"] = {"status": "unavailable",
                           "reason": f"锚点分数完整性校验失败:{exc}"[:200]}
            return out
        adf = (pack or {}).get("validation")
        if not isinstance(adf, pd.DataFrame):
            out["note"] = {"status": "unavailable",
                           "reason": "锚点无 validation_score"}
            return out
        if not baseline_view_compatible(view, (pack or {}).get("data_view")):
            out["note"] = {"status": "incomparable",
                           "reason": (f"锚点属 {(pack or {}).get('data_view')}"
                                      f" 视图,trial 属 {view}:不可配对")}
            return out
        try:
            out["score"] = align_score_by_keys(
                adf["score"].values, score_df[SAMPLE_KEY_COLUMN].values,
                adf[SAMPLE_KEY_COLUMN].values)
        except Exception as exc:  # noqa: BLE001
            out["note"] = {"status": "incomparable",
                           "reason": f"锚点预测不可配对:{exc}"[:200]}
        return out

    def _regression_periods(self, valid: pd.DataFrame):
        """回归诊断的分期列:validation 时间列(日;>400 期时降到月)。"""
        try:
            from .metrics import REGRESSION_METRICS

            tc = self.spec.time_column
            if self.spec.primary_metric not in REGRESSION_METRICS \
                    or not tc or tc not in valid.columns:
                return None
            col = valid[tc]
            if pd.api.types.is_datetime64_any_dtype(col):
                per = col.dt.strftime("%Y-%m-%d")
                if per.nunique() > 400:
                    per = col.dt.strftime("%Y-%m")
            else:
                per = col.astype(str)
                if per.nunique() > 400:
                    return None
            return per.to_numpy(dtype=str)
        except Exception:  # noqa: BLE001 —— 诊断输入,失败即不分期
            return None

    def _gi_inherit_from_parent(self, trial: TrialRecord) -> dict | None:
        """[v2.15.0] 只改模型的 trial 继承父代的 group_importance。

        适用:search_action ∈ {hpo, confirm},或 changed_modules ⊆ {model};
        且父代有可用的置换结论、两者 n_features 相同(特征面未变)。父代
        的 feature/group_importance.* 一并复制到本 trial(@id:group_importance
        引用展开读这里)。任一条件不满足返回 None → 照常重算。
        """
        import shutil

        sa = str(trial.data.get("search_action") or "")
        mods = {str(m) for m in (trial.data.get("changed_modules") or [])}
        if not (sa in ("hpo", "confirm") or (mods and mods <= {"model"})):
            return None
        pids = list(trial.parent_ids or [])
        if len(pids) != 1:
            return None
        pdir = self.run_dir / "trials" / pids[0]
        try:
            parent = TrialRecord.load(pdir)
        except Exception:  # noqa: BLE001
            return None
        gi = (parent.data.get("evaluation") or {}).get("group_importance")
        if not isinstance(gi, dict) or (gi.get("skipped")
                                        and not gi.get("inherited")):
            return None
        try:
            n_me = int((load_json(Path(trial.trial_dir) / "metrics.json")
                        or {}).get("n_features") or 0)
            n_pa = int((load_json(pdir / "metrics.json") or {}).get(
                "n_features") or 0)
        except Exception:  # noqa: BLE001
            return None
        if not n_me or n_me != n_pa:
            return None
        out = dict(gi)
        out["inherited"] = True
        out["inherited_from"] = gi.get("inherited_from") or parent.trial_id
        out["gi_elapsed_sec"] = 0.0
        out["note_inherited"] = (
            "v2.15.0 group_importance_policy=auto(需显式开启;默认 skip_rejected):只改模型的 trial 特征面与"
            "父代相同,族级置换结论继承自 " + str(out["inherited_from"])
            + "(模型不同只影响 delta 幅度,不改族生死判定;需要重算时把"
            "evaluator.group_importance_policy 设为 auto 恢复继承,设为 all 强制重算)")
        src_f = pdir / "feature"
        dst_f = Path(trial.trial_dir) / "feature"
        try:
            dst_f.mkdir(parents=True, exist_ok=True)
            for name in ("group_importance.json", "group_importance.csv"):
                if (src_f / name).exists() and not (dst_f / name).exists():
                    shutil.copy2(src_f / name, dst_f / name)
        except OSError:
            pass
        return out

    def _run_group_importance(self, trial: TrialRecord,
                              valid: pd.DataFrame, y: np.ndarray,
                              treatment: np.ndarray | None) -> dict | None:
        """加载 trial 模型工件跑分组置换;工件/列缺失时静默跳过。

        v12(P1-3):
        - n_repeat 默认 1→5(单次置换 delta_std 恒 0,裁决随机性不可见);
        - 列缺失不再整段放弃(旧:feature_columns.json 记 one-hot 后列,
          与原始 validation 帧不齐 → return None,28 trial 仅 2 产出);
          按可用列交集裁决并记录覆盖率。
        uplift 的 score_fn 需要 p1−p0 双次打分,S-learner 模型携带
        treatment 特征列,predict_proba 口径即正确;T-learner 等双模型
        工件不支持单 predict 口径时自动跳过(不阻塞评估)。

        v19.2 子样本口径:置换在 `group_importance_sample_rows` 行为上限
        的分层子样本上做(默认 0 = 全量,保持历史行为)。动机是 400w
        尺度的实测成本——598,955 行 × 55 族 × 5 repeat 的评估段需要
        ~40 分钟,是 trial_timeout/heavy 预算的大头,且外部 kill 后
        checkpoint 续算也要重跑同一量级。子样本在所有 trial 上取同一
        分层口径,跨 trial 可比性反而更好;它只影响"族生死/排序"这一
        离散裁决,不影响落盘的 validation AUC(那是全量口径)。
        """
        from .group_importance import run_group_importance

        n_rows = int(self.config.get("evaluator."
                                     "group_importance_sample_rows", 0) or 0)
        # v2.16.2(run4 regression D-003):把抽样的行位置交给 group_importance,
        # 让它对 trial 自存的模型输入帧(model/validation_input.parquet,行序 =
        # validation split)按**同一行位置**取子集。旧实现只抽基础帧,自存帧
        # 446,924 行 ≠ 子样本 200,309 行 → 弃用自存帧、回退基础帧(加宽列 0 填充)
        # → 加宽族恒 unverifiable,每 trial 空耗 35~44 min。
        row_positions = None
        n_full_rows = int(len(valid))
        if n_rows > 0 and len(valid) > n_rows:
            rng = np.random.default_rng(20260911)
            y_arr = np.asarray(y)
            # v2.17.0:连续目标(回归)按唯一值分层时每个唯一值至少保 1 行,
            # 子样本会远超上限(原值金额上万个唯一值)。改为"零值 + 正值
            # 10 分位"分层;离散目标(≤50 个取值)仍按唯一值分层。
            strata = y_arr
            try:
                if np.unique(y_arr).size > 50:
                    yf = y_arr.astype(float)
                    strata = np.zeros(yf.size, dtype=int)
                    pos = np.isfinite(yf) & (yf != 0)
                    if pos.any():
                        edges = np.unique(np.quantile(
                            yf[pos], np.linspace(0, 1, 11)[1:-1]))
                        strata[pos] = 1 + np.searchsorted(
                            edges, yf[pos], side="right")
            except (TypeError, ValueError):
                strata = y_arr
            keep: list[int] = []
            for cls in np.unique(strata):
                idx = np.flatnonzero(strata == cls)
                take = int(round(n_rows * len(idx) / max(len(y_arr), 1)))
                take = max(1, min(take, len(idx)))
                keep.extend(rng.choice(idx, size=take, replace=False)
                            .tolist())
            keep = sorted(keep)
            valid = valid.iloc[keep]
            y = y_arr[keep]
            if treatment is not None:
                treatment = np.asarray(treatment)[keep]
            row_positions = np.asarray(keep, dtype="int64")
            logger.info("分组置换使用分层子样本 %d/%d 行"
                        "(evaluator.group_importance_sample_rows)",
                        len(keep), len(y_arr))

        import os as _os

        # v19.4:并发 worker 数(config 键 → 环境变量;环境变量优先,
        # 与 group_importance 的读取序一致)。config 缺省 4。
        _workers = int(self.config.get(
            "evaluator.group_importance_workers", 4))
        _os.environ.setdefault("MODELEVO_GI_WORKERS", str(_workers))
        return run_group_importance(
            Path(trial.trial_dir), valid, y, self.spec.primary_metric,
            treatment=treatment,
            row_positions=row_positions, n_full_rows=n_full_rows,
            n_repeat=int(self.config.get("evaluator.group_importance_repeat",
                                         5)),
            # v2.13.5:置换线程预算 = 本 trial 的调度线程配额(与训练同一份
            # "并发×线程≈可用核"分配);缺省时 group_importance 自行兜底
            threads=(getattr(self, "_eval_thread_budget", None) or {}).get(
                trial.trial_id))

    def _collect_multi_seed_scores(self, trial, y_true) -> list | None:
        """收集 Trial 落盘的多 seed 分数(可选 seed_scores/ 目录)。

        Trial 代码可为每个 seed 输出 validation_score.parquet 的变体
        (seed-1.parquet ...)。未输出时返回 None,稳定性退化为 bootstrap CI。
        """
        seed_dir = Path(trial.trial_dir) / "seed_scores"
        if not seed_dir.is_dir():
            return None
        import pandas as pd

        scores = []
        for p in sorted(seed_dir.glob("*.parquet")):
            try:
                scores.append(pd.read_parquet(p)["score"].values)
            except Exception as exc:  # noqa: BLE001
                logger.warning("读取 seed score 失败 %s: %s", p, exc)
        return scores or None


def baseline_view_compatible(trial_view: str | None,
                             baseline_view: str | None) -> bool:
    """trial 与 baseline 是否属同一数据视图(可做配对比较)。

    v2.13.2:跨视图行集不同(sample 面 30110 行 vs v1 面 43563 行),
    `align_score_by_keys` 的键集合必然不等 → 抛错 → evaluator 把整条
    trial 裁为 invalid("预测身份/源码完整性检查失败")。实测事故
    example-seq-v3-v2.12.2.4:gen-01 的 10 个 sample 面 trial 全部
    因此报废,Agent 只能在 v1 面重跑 5 个。

    视图不同是**结构性不可比**(预期内,正常裁决),与"对齐失败 =
    身份/完整性事故"(必须 invalid,fail-closed)是两回事。本函数只
    判前者;调用方据此跳过对齐并走 evaluator 既有的
    `baseline_incomparable` 分支。

    未知视图(None/空)按 v1 处理:旧 trial 无 data_view 字段,
    语义即 v1(`trial.data.get("data_view") or "v1"` 是既有口径)。
    """
    return str(trial_view or "v1") == str(baseline_view or "v1")


def _pairing_keys(current_score: pd.DataFrame,
                  parent_score: pd.DataFrame | None,
                  base_score: pd.DataFrame | None,
                  baseline_comparable: bool,
                  parent_comparable: bool = True) -> dict:
    """v19:为跨视图配对比较构造对齐键(req-t7-drawdown 伪影事故回写)。

    数据视图 trial 与 v1 父代/baseline 比较时,两视图落盘行序不同
    (历史:视图落盘各自天内洗牌,trial-009 实测同位置用户重合率
    0.0001)——行位置配对把不同用户的分数配对,产生 delta=+0.2844
    的伪 accepted。防御:对 (时间列, ID列) 构造键矩阵传给
    paired_bootstrap_delta 做 key-join 对齐;键不匹配时 bootstrap
    显式报错(宁可不比,不可错比)。同视图(无跨视图分数)时返回
    空 dict,行为不变。

    baseline 键只在 baseline_comparable 时传——不可比(baseline_incomparable)
    分支 evaluation 不消费 baseline 键。

    v2.13.2:父代键同口径加 parent_comparable 门。旧实现只看
    ``parent_score is not None`` —— 跨视图父代(如 v1 baseline →
    sample 子代 → sample 孙代,实测 example-seq-v3-v2.12.2.4
    trial-003→004 已出现该血缘)仍会把旧键传下去,evaluator 二次
    align 抛错后又把 trial 裁为 invalid,修复等于没做。
    """
    if SAMPLE_KEY_COLUMN not in current_score.columns:
        raise ValueError("当前预测缺稳定样本键")
    keys = current_score[SAMPLE_KEY_COLUMN].astype(str).values
    out: dict = {}
    if parent_score is not None and parent_comparable:
        if SAMPLE_KEY_COLUMN not in parent_score.columns:
            raise ValueError("父代预测缺稳定样本键")
        out["parent_keys"] = keys
        out["parent_old_keys"] = parent_score[
            SAMPLE_KEY_COLUMN].astype(str).values
    if base_score is not None and baseline_comparable:
        if SAMPLE_KEY_COLUMN not in base_score.columns:
            raise ValueError("baseline 预测缺稳定样本键")
        out["baseline_keys"] = keys
        out["baseline_old_keys"] = base_score[
            SAMPLE_KEY_COLUMN].astype(str).values
    return out


def load_parent_scores(run_dir: Path, parent_id: str) -> dict | None:
    from .logging_utils import load_yaml

    trial_dir = Path(run_dir) / "trials" / parent_id
    p = trial_dir / "validation_score.parquet"
    if not p.exists():
        return None
    record = load_yaml(trial_dir / "trial.yaml") \
        if (trial_dir / "trial.yaml").exists() else {}
    view = str((record or {}).get("data_view") or "v1")
    labels_dir = view_splits_dir(Path(run_dir), view) / "labels"
    source_manifest = verify_trial_source(trial_dir)
    split_manifest = view_splits_dir(Path(run_dir), view) / "manifest.json"
    out: dict = {"data_view": view}
    for split in ("validation", "shadow_test"):
        score_path = trial_dir / f"{split}_score.parquet"
        label_path = labels_dir / f"{split}.parquet"
        if not score_path.exists() or not label_path.exists():
            continue
        meta = verify_prediction_artifact(score_path)
        if meta.get("source_manifest_sha256") != source_manifest.get(
                "combined_sha256"):
            raise PermissionError(
                f"父代 {parent_id}/{split} 预测与当前冻结源码不匹配")
        if meta.get("data_view") != view:
            raise PermissionError(
                f"父代 {parent_id}/{split} 预测 data_view 不匹配")
        if split_manifest.exists() and meta.get(
                "split_manifest_sha256") != sha256_file(split_manifest):
            raise PermissionError(
                f"父代 {parent_id}/{split} 预测所用 split manifest 已变化")
        score_df = pd.read_parquet(score_path)
        labels = pd.read_parquet(label_path)
        if SAMPLE_KEY_COLUMN not in labels.columns:
            raise PermissionError(
                f"父代 {parent_id} 的 {split} 标签缺样本键；请重跑 "
                "profile/trial 建立完整性工件")
        if len(score_df) != len(labels):
            raise ValueError(
                f"父代 {parent_id}/{split} 分数与标签行数不一致")
        if SAMPLE_KEY_COLUMN not in score_df.columns:
            raise PermissionError(
                f"父代 {parent_id}/{split} 预测缺稳定样本键；不能按行号猜测")
        out[split] = score_df
    return out if "validation" in out else None
