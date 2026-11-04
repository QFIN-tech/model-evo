"""Trial 执行的会话解耦(v2.13.3)。

**背景(实测事故,run example-seq-v3-v2.13.0.1 gen-04)。** 三个 HPO
trial 在 16:53 同时启动,之后**三次同时死亡**:

    16:53:25 启动 → 17:36:52/55/53 三条独立的心跳全部停止(2607/2609/2608s)
    21:27 被发现 → 重启 → 22:27:04/10/19 再次同时死亡
    22:51 重启 → 23:51:23/43/42 第三次同时死亡(3582/3590/3585s)

三次死亡时刻对齐到秒,且都发生在 agent 会话侧后台任务的生命周期边界上:
`runner`(``modelevo.py run``)是 ``claude`` 进程的子进程,而 ``train.py``
虽已 ``start_new_session=True``(见 ``scheduler._execute_once`` 的注释),
仍与 runner 一起消失。主机 960GB 内存排除 OOM;``wait`` 从不 kill
(``cmd_inspect.cmd_wait`` 超时只打印 WAIT_TIMEOUT)。合计空烧 7.9 小时,
有效产出 0(第四轮 00:10 重启后才跑完)。

**根因。** trial 的整个生命周期(runner + train.py + 心跳)挂在 agent
会话的进程树上。训练是"小时级、与 Agent 往返无关"的工作,把它托管给
一个会被上下文/会话边界回收的父进程,是把机器时间的控制权交给最不可靠
的一环。

**处置。** 提供 ``--detach``:把 ``run`` / ``run-generation`` 的实际执行
体用**双重 fork + setsid + 重定向 stdio 到日志文件**拉起到一个独立会话,
父进程(agent 的子进程)立即退出。这样:

- 训练进程的父进程是 init,不再随 agent 会话被回收;
- 控制终端/进程组与会话无关,``kill`` agent 的进程组不再连带;
- stdout/stderr 落 ``logs/detached-<tag>.log``(不再依赖管道,父进程
  退出不会触发 SIGPIPE / 写端关闭)。

``wait`` 只读 ``trial.yaml``,与谁拉起无关,天然兼容本模式。

**不做的事。** 不在这里实现"重启即自动续跑"——续跑语义属于 runner
(``TrialRunner.run`` 的孤儿自愈 + HPO checkpoint 回放 + 阶段标记),
daemon 只负责**让进程活到它自己能完成**。
"""

from __future__ import annotations

import os
import shlex
import signal
import sys
import time
from pathlib import Path

#: 被 detach 拉起的子进程带这个环境变量,用于识别"已是分离进程",
#: 避免二次 detach 无限套娃
DETACH_ENV = "MODELEVO_DETACHED"


def is_detached_child() -> bool:
    return os.environ.get(DETACH_ENV) == "1"


def spawn_detached(
    argv: list[str],
    *,
    log_path: Path,
    cwd: Path,
    env_extra: dict[str, str] | None = None,
) -> int:
    """在独立会话拉起 ``argv``,立即返回子进程 pid。

    实现要点(每一项都对应一种"进程被连带杀"的形态):

    - ``os.setsid()``:新会话 + 新进程组。父进程所在进程组被 ``kill``
      时(os.kill(-pgid) / shell 的 job 回收)不再波及本进程;
    - **双重 fork**:中间进程立刻退出,真正的 worker 被 init(pid 1)
      收养 —— 父进程只有一个直接子进程,而它已经退出,所以父进程退出
      时不会再"顺手"回收 worker;
    - stdio 全部重定向到 ``log_path``:不继承 agent 的管道 fd。继承
      管道时父进程一退出写端关闭,worker 第一次 print 就吃 EPIPE/SIGPIPE
      (静默死亡,与本次事故的形态一致);
    - 关闭继承的其他 fd(closerange)避免持有 agent 的会话资源。

    返回值是 worker 的 pid(经管道从中间进程回传),调用方可用于
    写 pid 文件 / 报告给用户。
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env[DETACH_ENV] = "1"
    env.update(env_extra or {})

    # 用管道把 worker pid 从中间进程带回父进程
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid > 0:
        # ---- 父(agent 侧):读 pid 后立即返回 ----
        os.close(w_fd)
        try:
            with os.fdopen(r_fd, "rb") as rf:
                raw = rf.read(64)
        except OSError:
            raw = b""
        try:
            return int(raw.decode().strip() or "0")
        except ValueError:
            return 0

    # ---- 中间进程:setsid 后再 fork,自己立刻退出 ----
    try:
        os.close(r_fd)
        os.setsid()
        pid2 = os.fork()
        if pid2 > 0:
            # 中间进程写 pid 给父进程后退出,worker 被 init 收养
            try:
                os.write(w_fd, str(pid2).encode())
            finally:
                os.close(w_fd)
            os._exit(0)
        # ---- worker ----
        os.chdir(str(cwd))
        # 重定向 stdio 到日志文件(追加:同一 trial 的多次 detach 不互相截断)
        lf = os.open(str(log_path),
                     os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(lf, 1)
        os.dup2(lf, 2)
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
        if lf > 2:
            os.close(lf)
        if devnull > 2:
            os.close(devnull)
        os.write(1, (f"[detach] worker pid={os.getpid()} "
                     f"cwd={cwd} argv={shlex.join(argv)}\n").encode())
        sys.stdout.flush()
        try:
            os.execvpe(argv[0], argv, env)
        except Exception as exc:  # noqa: BLE001
            os.write(1, f"[detach] exec 失败: {exc}\n".encode())
            os._exit(127)
    except Exception:  # noqa: BLE001
        os._exit(1)


def build_worker_argv(argv: list[str]) -> list[str]:
    """把 ``--detach`` 换成 ``--no-detach``(worker 侧不再 detach)。"""
    out: list[str] = []
    for a in argv:
        if a == "--detach":
            out.append("--no-detach")
        else:
            out.append(a)
    if "--no-detach" not in out:
        out.append("--no-detach")
    return out


def pid_alive(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def read_pid_file(path: Path) -> int | None:
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        pid = int(raw or "0")
    except ValueError:
        return None
    return pid if pid > 0 else None


def write_pid_file(path: Path, pid: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(f"{pid}\n", encoding="utf-8")


def wait_for_exit(pid: int, *, timeout_sec: float | None = None,
                  poll_sec: float = 5.0) -> bool:
    """轮询前 pid 是否已退出;返回 True=已退出(或 timeout 内退出)。"""
    t0 = time.time()
    while True:
        if not pid_alive(pid):
            return True
        if timeout_sec is not None and time.time() - t0 > timeout_sec:
            return False
        time.sleep(poll_sec)


def terminate(pid: int, *, grace_sec: float = 10.0) -> None:
    """SIGTERM → 宽限 → SIGKILL(setsid 后是进程组组长,按组杀)。"""
    if not pid_alive(pid):
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return
    t0 = time.time()
    while time.time() - t0 < grace_sec:
        if not pid_alive(pid):
            return
        time.sleep(0.5)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# v2.17.0(WP5a):轻量孤儿探针(供 `wait` 快路径使用)
#
# ``wait`` 是长时只读轮询。旧实现为了 stale 探针 import runtime.runner
# (numpy/pandas/evaluator,~80MB 常驻);run4 regression.2 同机内存争用时
# wait 进程两次被 cgroup OOM 杀掉(exit 137,训练幸存)。本探针只用标准库,
# 判定口径与 ``TrialRunner.is_stale_running`` 一致(测试对两者做对拍):
#   无活 runner 锁 且 (无心跳/终态行 或 末条距今 > stale_sec) → stale;
#   trial.log 末尾出现 harness 日志行("[runner.py")→ 有进程在工作。
# ---------------------------------------------------------------------------

#: 与 TrialRunner.HEARTBEAT_STALE_SEC 同值(5 × 心跳周期)
HEARTBEAT_STALE_SEC = 300
RUNNER_LOCK = ".runner.lock"


def runner_lock_alive(trial_dir: Path) -> int | None:
    """trial 目录 .runner.lock 的活持有者 pid;无锁/锁损坏/进程已死返回 None。"""
    pid = read_pid_file(Path(trial_dir) / RUNNER_LOCK)
    return pid if pid and pid_alive(pid) else None


def stale_running_probe(trial_dir: Path, status: str,
                        stale_sec: float = HEARTBEAT_STALE_SEC
                        ) -> tuple[bool, str]:
    """RUNNING trial 的孤儿探针(纯标准库)。返回 (is_stale, reason)。"""
    if status != "RUNNING":
        return False, ""
    tdir = Path(trial_dir)
    if runner_lock_alive(tdir) is not None:
        return False, "lock alive"
    log = tdir / "trial.log"
    last_ts = None
    try:
        if log.exists():
            lines = log.read_text(encoding="utf-8",
                                  errors="ignore").splitlines()
            for ln in reversed(lines[-80:]):
                if "[runner.py" in ln:
                    last_ts = time.time()
                    break
                if "heartbeat:" in ln or "process exited" in ln:
                    from datetime import datetime as _dtm

                    head = ln.split("]", 1)[0].lstrip("[").strip()
                    last_ts = _dtm.fromisoformat(head).timestamp()
                    break
    except (OSError, ValueError):
        last_ts = None
    if last_ts is None:
        return True, "no heartbeat and no live lock"
    age = time.time() - last_ts
    if age > stale_sec:
        return True, f"heartbeat stale {age:.0f}s > {stale_sec:.0f}s"
    return False, f"heartbeat fresh ({age:.0f}s)"
