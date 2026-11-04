"""`wait` 命令实现(v2.17.0 起单独成模块,轻量)。

**为什么单独成模块。** `wait` 是长时只读轮询(agent 以 `--timeout 570`
循环调用)。旧实现挂在完整 CLI 上:进程启动即 import pandas / numpy /
pyarrow / runtime.runner(~85MB 常驻),stale 探针还要 runtime.runner。
run4 regression.2 同机内存争用时(可用内存 23~50GB)cgroup OOM killer
两次杀掉了 wait(exit 137),训练进程幸存——轮询进程不该为重模块付内存。

本模块只依赖标准库 + 轻模块(runtime.config/spec/trial/daemon/topology、
_mvo.context);`scripts/modelevo.py` 以脚本运行且子命令为 `wait` 时走
快路径直接调本模块 `main()`,不加载其余 CLI。测试见
tests/test_v2_17_0_scheduler.py(import 集合与峰值 RSS)。

**幂等。** wait 只读 trial.yaml / pid 文件 / trial.log,不写任何状态、
不 kill 任何进程:被杀(137/143)后原样重调同一命令即可,训练不受影响。
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from .context import (_load_spec, _run_dir_from, _trial_reached_terminal,
                      _trial_summary_line, logger)

WAIT_KILLED_HINT = ("wait 只读、不改状态、不 kill 训练:若 wait 进程被杀"
                    "(exit 137/143,如 cgroup OOM)原样重调同一命令即可,"
                    "训练不受影响")

WAIT_HELP = (
    "v17.2:确定性等待 trial 终态(替代 sleep+tail 轮询)——终态输出与 run "
    "同格式的 SUMMARY 行;v2.11 检测孤儿 RUNNING(锁死+心跳停摆)打印 "
    "STALE_RUNNING 并 exit 4;v2.17.0 轻量(不 import pandas/numpy/runner)。"
    + WAIT_KILLED_HINT)


def add_wait_arguments(sp) -> None:
    """`wait` 子命令参数(modelevo.build_parser 与快路径共用,单源)。"""
    sp.add_argument("--spec")
    sp.add_argument("--run")
    _wg = sp.add_mutually_exclusive_group(required=True)
    _wg.add_argument("--trial", help="等单个 trial 终态")
    _wg.add_argument("--generation", type=int, default=None,
                     help="v2.13.5:等一代(detached run-generation worker "
                          "退出且该代已启动 trial 全部终态;终态输出 GEN_DONE "
                          "与每个 trial 的 SUMMARY 行)")
    _wg.add_argument("--job", default=None, metavar="TAG",
                     help="v2.13.5:等任意 --detach 任务(logs/detached-<TAG>"
                          ".pid 的进程退出),如 finalize-all / "
                          "finalize-retrain / run-trial-XXX")
    sp.add_argument("--timeout", type=float, default=None, metavar="SEC",
                    help="最长等待秒数(超时输出 WAIT_TIMEOUT 行并 "
                         "exit 3;不 kill 进程;缺省无限等)。agent 的 Bash "
                         "工具单次上限 10 min:用 --timeout 570 循环调用,"
                         "完成即返回,不再 sleep 固定时长")
    sp.add_argument("--poll-interval", type=float, default=15.0,
                    metavar="SEC",
                    help="trial.yaml 轮询间隔(5~60s,默认 15s;纯文件"
                         "mtime/stat 级轮询,无 API 调用成本)")


def _sleep_until(deadline: float | None, interval: float) -> None:
    # 首轮全量 interval,其后 min(interval, 到 deadline 的余量):
    # timeout=2 + interval=15 的组合不该等满 15s 才发现超时
    time.sleep(interval if deadline is None else max(
        0.5, min(interval, deadline - time.monotonic())))


def wait_job(run_dir: Path, tag: str, timeout: float | None,
             interval: float) -> int:
    """等 logs/detached-<tag>.pid 的进程退出。exit 0=已结束,3=超时,1=无 pid。"""
    from runtime.daemon import pid_alive, read_pid_file

    pid_file = run_dir / "logs" / f"detached-{tag}.pid"
    pid = read_pid_file(pid_file)
    if not pid:
        print(f"WAIT_NO_JOB\t{tag}\t{pid_file}")
        return 1
    deadline = time.monotonic() + timeout if timeout and timeout > 0 else None
    while pid_alive(pid):
        if deadline is not None and time.monotonic() >= deadline:
            print(f"WAIT_TIMEOUT\t{tag}\tpid={pid}")
            return 3
        _sleep_until(deadline, interval)
    log = run_dir / "logs" / f"detached-{tag}.log"
    tail = ""
    if log.exists():
        tail = "\n".join(log.read_text(encoding="utf-8",
                                       errors="replace").splitlines()[-5:])
    print(f"JOB_DONE\t{tag}\tpid={pid}")
    if tail:
        print(tail)
    return 0


def wait_generation(run_dir: Path, spec, generation: int,
                    timeout: float | None, interval: float) -> int:
    """等一代:detached worker 退出(若有)且该代全部已启动 trial 到终态。"""
    from runtime.daemon import pid_alive, read_pid_file
    from runtime.trial import TrialFactory

    tag = f"gen-{int(generation):02d}"
    pid = read_pid_file(run_dir / "logs" / f"detached-{tag}.pid")
    deadline = time.monotonic() + timeout if timeout and timeout > 0 else None
    pat = re.compile(rf"^trial-\d+-g{int(generation):02d}-")
    while True:
        trials = [t for t in TrialFactory(run_dir).all_trials()
                  if pat.match(t.trial_id)]
        started = [t for t in trials if t.status not in ("PROPOSED",
                                                         "CODE_WRITTEN")
                   or _trial_reached_terminal(t)]
        pending = [t for t in started if not _trial_reached_terminal(t)]
        worker_alive = bool(pid) and pid_alive(pid)
        if not worker_alive and not pending:
            print(f"GEN_DONE\t{tag}\ttrials={len(started)}")
            for t in started:
                print(_trial_summary_line(t, spec))
            return 0
        if deadline is not None and time.monotonic() >= deadline:
            # v2.17.0:内存准入排队中的 trial 标 queued(在排队,不是卡住)
            print(f"WAIT_TIMEOUT\t{tag}\trunning={len(pending)}\t"
                  f"worker_alive={int(worker_alive)}\t"
                  + ",".join(
                      f"{t.trial_id}:{t.status}"
                      + (":queued" if ((t.data.get('admission') or {})
                                       .get('state') == "queued") else "")
                      for t in pending))
            return 3
        _sleep_until(deadline, interval)


def cmd_wait(args) -> int:
    """确定性等待(v17.2):阻塞到 trial 终态,结尾输出 SUMMARY 一行。

    替代 subagent 的 sleep+tail 轮询(实测 gen-01/gen-02 各 13~15 次
    `sleep 115; tail …` 循环,每次轮询都是一次全上下文 API 调用,
    ~3.5M tokens 信息增量≈0)。调用契约:

    - 通常配 harness 可追踪的后台训练使用:Bash `run_in_background: true`
      跑 `run --trial <id>`,同 subagent 内跟一条前台
      `wait --trial <id> [--timeout S]`——wait 阻塞到终态并回传
      SUMMARY 行,subagent 完成通知闭环,无需轮询日志;
    - 也可直接阻塞跑 wait 前先后台启动 run;trial 已终态时立即返回;
    - timeout 到点仍未终态:输出 `WAIT_TIMEOUT` 行并 exit 3(调用方
      决定续等或升级;进程仍在跑,wait 从不 kill);
    - trial 不存在:exit 1(与 run --trial 的行为一致)。

    v2.11 孤儿检测:轮询中发现 RUNNING trial 满足"无活 runner 锁 +
    心跳停摆"时,wait 不再无限等——打印 STALE_RUNNING 行并 exit 4,
    让调用方立即重入 run(自愈闭环),而不是陪一个已死的进程耗到
    timeout(实测 run drawdown-req-t7-6w:心跳 19:58 消失后,调用方
    按 timeout 节奏空等 30min 才发现进程已死)。

    v2.17.0:轻量(见模块说明);内存准入排队中的 trial 打印一次
    ADMISSION_QUEUED 行(在排队、训练尚未启动,不是卡住);wait 被杀
    (137)时原样重调即可——本命令只读,重入完全幂等。
    """
    run_dir = _run_dir_from(args)
    print(f"WAIT_HINT\t{WAIT_KILLED_HINT}", flush=True)
    # v2.13.5:--generation / --job 两种等待对象(detached run-generation /
    # finalize 等)。req-t7-rand400w 的 subagent 全靠 `sleep 600; tail` 轮询
    # (Bash 单次上限 10 min,完成后最多晚 10 min 才被发现,每次轮询还是一次
    # 全上下文调用)。
    _interval = max(5.0, min(float(getattr(args, "poll_interval", 15.0)
                                   or 15.0), 60.0))
    if getattr(args, "trial", None) is None:
        if getattr(args, "generation", None) is not None:
            return wait_generation(run_dir, _load_spec(args),
                                   int(args.generation), args.timeout,
                                   _interval)
        return wait_job(run_dir, str(args.job), args.timeout, _interval)
    trial_dir = run_dir / "trials" / args.trial
    if not trial_dir.exists():
        logger.error("Trial 不存在: %s", trial_dir)
        return 1
    spec = _load_spec(args)
    from runtime.daemon import stale_running_probe
    from runtime.trial import TrialRecord

    deadline = time.monotonic() + float(args.timeout) \
        if args.timeout and args.timeout > 0 else None
    interval = max(5.0, min(float(args.poll_interval), 60.0))
    _queued_shown = False
    while True:
        trial = TrialRecord.load(trial_dir)
        # v2.16.2:传 run_dir —— 只认当前 attempt(重入中的 trial 不因上一轮
        # 终态 / result 立即返回,见 context._reentry_pending)
        if _trial_reached_terminal(trial, run_dir):
            break
        # v2.16.0:被拓扑门拦下的 trial 不会自己起跑,立即返回而不是等到超时
        _topo = trial.data.get("model_topology") or {}
        if trial.status in ("PROPOSED", "CODE_WRITTEN") \
                and _topo.get("decision") == "blocked":
            from runtime.topology import EXIT_APPROVAL_REQUIRED

            print(f"TOPOLOGY_BLOCKED\t{trial.trial_id}\t{_topo.get('kind')}"
                  f"\tneeds={','.join(_topo.get('needs') or [])}")
            return EXIT_APPROVAL_REQUIRED
        if trial.status == "RUNNING":
            _adm = trial.data.get("admission") or {}
            if _adm.get("state") == "queued" and not _queued_shown:
                _queued_shown = True
                print(f"ADMISSION_QUEUED\t{trial.trial_id}\tsince="
                      f"{_adm.get('since')}\testimate_gb="
                      f"{_adm.get('estimate_gb')}\tavailable_gb="
                      f"{_adm.get('available_gb')}", flush=True)
            _stale, _why = stale_running_probe(trial_dir, trial.status)
            # v2.11:stale 探针只对"曾真正起跑"的 trial 生效——判定
            # 依据是 trial.log 的心跳/终态行;无该文件(如单测夹具、
            # 从未 run 过的 RUNNING 占位)时探针恒判 stale,会把
            # "尚未起跑"误报成孤儿。规避:trial.log 不存在时跳过
            # 探针,交给 timeout 语义(与 v17.2 行为一致)。
            if _stale and (trial_dir / "trial.log").exists():
                print(f"STALE_RUNNING\t{trial.trial_id}\t{_why}")
                logger.error(
                    "孤儿 RUNNING(锁进程已死+心跳停摆): %s —— 立即"
                    "重入 `run --trial %s` 自愈(有分数走 evaluate_only,"
                    "无分数回退 CODE_WRITTEN 重训)", trial.trial_id,
                    trial.trial_id)
                return 4
        if deadline is not None and time.monotonic() >= deadline:
            print("WAIT_TIMEOUT\t{}\tstatus={}".format(
                trial.trial_id, trial.status))
            return 3
        _sleep_until(deadline, interval)
    # 终态:SUMMARY 行与 run 同格式(cmd_run_trial 终点同款),Agent
    # 一套解析器通吃两个命令。无 evaluation(FAILED 且未评估)时字段
    # 落 "-"。附带一行 verdict,保持与 run 的输出习惯一致。
    ev = trial.data.get("evaluation") or {}
    print(f"{trial.trial_id}: status={trial.status} "
          f"verdict={ev.get('verdict') or '-'}")
    print(_trial_summary_line(trial, spec))
    return 0


def build_wait_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="modelevo.py wait", description=WAIT_HELP)
    add_wait_arguments(p)
    p.set_defaults(func=cmd_wait)
    return p


def main(argv: list[str]) -> int:
    """快路径入口(`modelevo.py wait ...` 以脚本运行时由入口直接调用)。

    异常口径与完整 CLI 的 main 一致:SystemExit 透传,其余记日志返回 2。
    """
    args = build_wait_parser().parse_args(list(argv))
    args.command = "wait"
    args._raw_argv = ["wait", *argv]
    try:
        return cmd_wait(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("wait 失败: %s", exc, exc_info=True)
        return 2
