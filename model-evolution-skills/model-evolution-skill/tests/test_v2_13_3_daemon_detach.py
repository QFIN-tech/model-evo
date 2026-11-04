"""v2.13.3 回归测试:trial 执行与 agent 会话解耦(daemon)。

已核实事故(run example-seq-v3-v2.13.0.1 gen-04):三个 HPO trial
16:53 同时启动,之后**三次同时死亡**,死亡时刻对齐到秒且都落在
agent 会话侧后台任务的生命周期边界上:

    16:53:25 启动 → 17:36:52/55/53 心跳全停(2607/2609/2608s)
    21:27 重启  → 22:27:04/10/19 再次同时死亡(3574/3575/3579s)
    22:51 重启  → 23:51:23/43/42 第三次死亡(3582/3590/3585s)

`runner`(``modelevo.py run``)是 ``claude`` 进程的子进程;``train.py``
虽已 ``start_new_session=True``,仍随 runner 一起消失。主机 960GB 内存
排除 OOM;``wait`` 从不 kill。合计空烧 7.9 小时,有效产出 0。

本文件锁定 ``runtime/daemon.py`` 的语义:
  A. ``spawn_detached`` 返回的 worker 在**拉起者退出后继续运行**;
  B. worker 与拉起者**不在同一会话/进程组**,对拉起者所在进程组发
     SIGTERM 不波及 worker(事故的直接复现与修复验证);
  C. worker 的 stdout/stderr 落日志文件,不依赖继承的管道
     (继承管道时父进程退出 → 写端关闭 → 首次 print 即 SIGPIPE 静默死亡);
  D. ``build_worker_argv`` 把 ``--detach`` 换成 ``--no-detach``,
     且幂等(不会重复追加);
  E. CLI ``run`` / ``run-generation`` 接受 ``--detach``,缺省前台
     (既有调用方行为不变);worker 侧带 ``MODELEVO_DETACHED=1``,
     ``is_detached_child()`` 据此阻止套娃。
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402  hermetic bootstrap

sys.path.insert(0, str(SKILL))

from runtime.daemon import (  # noqa: E402
    DETACH_ENV,
    build_worker_argv,
    is_detached_child,
    pid_alive,
    read_pid_file,
    spawn_detached,
    write_pid_file,
)

#: 一个短命 worker:写若干 tick 后自行退出(便于验证"活着"与"跑完")
_WORKER = """\
import os, sys, time
print(f"worker start pid={os.getpid()} ppid={os.getppid()}", flush=True)
for i in range(12):
    print(f"tick {i}", flush=True)
    time.sleep(0.4)
print("worker done", flush=True)
"""


def _spawn_worker(tmp: Path, *, log_name: str = "w.log"):
    wf = tmp / "worker.py"
    wf.write_text(_WORKER, encoding="utf-8")
    log = tmp / log_name
    pid = spawn_detached([sys.executable, str(wf)], log_path=log,
                         cwd=tmp)
    return pid, log


def test_spawn_returns_live_worker_pid(tmp_path):
    pid, log = _spawn_worker(tmp_path)
    assert pid > 0
    assert pid_alive(pid), "worker 应立即存活"
    # 等它跑完,避免污染后续用例
    for _ in range(60):
        if not pid_alive(pid):
            break
        time.sleep(0.2)
    assert "worker done" in log.read_text()


def test_worker_survives_launcher_exit(tmp_path):
    """拉起者(模拟 agent 的子进程)退出后 worker 仍在跑。"""
    inner = (
        "import sys; sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "from runtime.daemon import spawn_detached\n"
        "pid = spawn_detached([%r, %r], log_path=Path(%r), cwd=Path(%r))\n"
        "print(pid, flush=True)\n"
    ) % (str(SKILL), sys.executable, str(tmp_path / "worker.py"),
         str(tmp_path / "w.log"), str(tmp_path))
    (tmp_path / "worker.py").write_text(_WORKER, encoding="utf-8")
    r = subprocess.run([sys.executable, "-c", inner],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    pid = int(r.stdout.strip().split()[0])
    # 拉起者已退出(returncode 拿到了);worker 必须还在
    assert pid_alive(pid), "拉起者退出后 worker 被连带杀死"
    for _ in range(60):
        if not pid_alive(pid):
            break
        time.sleep(0.2)
    assert "worker done" in (tmp_path / "w.log").read_text()


def test_worker_survives_launcher_process_group_kill(tmp_path):
    """事故直接复现:对拉起者所在进程组发 SIGTERM,worker 不受影响。

    gen-04 的形态就是"整棵 agent 进程树被回收"。setsid + 双重 fork
    之后 worker 属于新会话,按拉起者的 pgid 杀不到它。
    """
    (tmp_path / "worker.py").write_text(_WORKER, encoding="utf-8")
    inner = (
        "import sys, time; sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "from runtime.daemon import spawn_detached\n"
        "pid = spawn_detached([%r, %r], log_path=Path(%r), cwd=Path(%r))\n"
        "print(pid, flush=True)\n"
        "time.sleep(30)\n"
    ) % (str(SKILL), sys.executable, str(tmp_path / "worker.py"),
         str(tmp_path / "w.log"), str(tmp_path))
    # 拉起者自成会话(= agent 会话的替身),这样对它整组发信号
    # 不会波及本测试进程
    p = subprocess.Popen([sys.executable, "-c", inner],
                         stdout=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        pid = int(p.stdout.readline().strip())
        time.sleep(1.0)
        assert pid_alive(pid)
        assert os.getpgid(pid) != os.getpgid(p.pid), \
            "worker 必须在自己的进程组里(setsid 未生效)"
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        time.sleep(2.0)
        assert p.poll() is not None, "拉起者应已被杀"
        assert pid_alive(pid), \
            "拉起者进程组被回收后 worker 被连带杀死 —— 事故未修复"
    finally:
        if p.poll() is None:
            p.kill()
        try:
            os.kill(pid, signal.SIGKILL)
        except (OSError, UnboundLocalError):
            pass


def test_worker_has_own_session(tmp_path):
    """worker 在新会话里(与拉起者不同会话),且被 init(ppid=1)收养。

    注意:新会话的**组长**是双重 fork 的中间进程(它调用 setsid),
    worker 是该会话成员而非组长——这正是双重 fork 的本意:非组长
    进程不会因为打开 tty 而获得控制终端,比"让 worker 当组长"更稳。
    断言的是"与拉起者不同会话、不同进程组、已被 init 收养"这三件
    真正决定生存性的事。
    """
    (tmp_path / "worker.py").write_text(
        "import os, sys, time\n"
        "print(f'{os.getpid()} {os.getsid(0)} {os.getppid()} "
        "{os.getpgid(0)}', flush=True)\n"
        "time.sleep(1.0)\n", encoding="utf-8")
    my_sid = os.getsid(0)
    my_pgid = os.getpgid(0)
    pid = spawn_detached([sys.executable, str(tmp_path / "worker.py")],
                         log_path=tmp_path / "w.log", cwd=tmp_path)
    time.sleep(0.8)
    lines = (tmp_path / "w.log").read_text().splitlines()
    # 第一行是 [detach] 元信息,第二行才是 worker 输出
    body = [ln for ln in lines if not ln.startswith("[detach]")][0]
    wpid, sid, ppid, pgid = (int(x) for x in body.split())
    assert wpid == pid
    assert sid != my_sid, "worker 必须在新会话里(setsid 未生效)"
    assert pgid != my_pgid, "worker 必须在新进程组里"
    assert ppid == 1, "worker 应被 init 收养(双重 fork 的中间进程已退出)"


def test_stdio_goes_to_log_not_pipe(tmp_path):
    """stdio 落文件:拉起者退出不会触发写端关闭(SIGPIPE 静默死亡)。"""
    (tmp_path / "worker.py").write_text(
        "import sys, time\n"
        "for i in range(6):\n"
        "    print('line', i, flush=True)\n"
        "    time.sleep(0.3)\n", encoding="utf-8")
    inner = (
        "import sys; sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "from runtime.daemon import spawn_detached\n"
        "print(spawn_detached([%r, %r], log_path=Path(%r), cwd=Path(%r)),"
        " flush=True)\n"
    ) % (str(SKILL), sys.executable, str(tmp_path / "worker.py"),
         str(tmp_path / "w.log"), str(tmp_path))
    r = subprocess.run([sys.executable, "-c", inner],
                       capture_output=True, text=True)
    pid = int(r.stdout.strip())
    for _ in range(60):
        if not pid_alive(pid):
            break
        time.sleep(0.2)
    txt = (tmp_path / "w.log").read_text()
    assert "line 5" in txt, "worker 未写完全部输出(可能被 SIGPIPE 杀死)"
    assert "line 0" in txt


def test_log_is_appended_not_truncated(tmp_path):
    """同一 tag 多次 detach 时日志追加,不互相截断(多次重启可追溯)。"""
    (tmp_path / "worker.py").write_text("print('hello', flush=True)\n",
                                        encoding="utf-8")
    for _ in range(2):
        pid = spawn_detached([sys.executable, str(tmp_path / "worker.py")],
                             log_path=tmp_path / "w.log", cwd=tmp_path)
        for _ in range(40):
            if not pid_alive(pid):
                break
            time.sleep(0.1)
    assert (tmp_path / "w.log").read_text().count("hello") == 2


# --------------------------------------------------------------------------
# argv 重建
# --------------------------------------------------------------------------

def test_build_worker_argv_swaps_detach_flag():
    out = build_worker_argv(["run", "--trial", "t1", "--detach", "--run", "r"])
    assert "--detach" not in out
    assert "--no-detach" in out
    assert out.count("--no-detach") == 1
    assert out[0] == "run" and "t1" in out


def test_build_worker_argv_idempotent():
    """已有 --no-detach 时不重复追加(worker 内不再 detach 的保证)。"""
    out = build_worker_argv(["run", "--trial", "t1", "--no-detach"])
    assert out.count("--no-detach") == 1


def test_build_worker_argv_without_detach_adds_no_detach():
    out = build_worker_argv(["run-generation", "--generation", "3"])
    assert out.count("--no-detach") == 1


# --------------------------------------------------------------------------
# 环境标记与 pid 文件
# --------------------------------------------------------------------------

def test_worker_env_marked_detached(tmp_path):
    (tmp_path / "worker.py").write_text(
        "import os\n"
        "print(os.environ.get('MODELEVO_DETACHED', 'missing'), flush=True)\n",
        encoding="utf-8")
    pid = spawn_detached([sys.executable, str(tmp_path / "worker.py")],
                         log_path=tmp_path / "w.log", cwd=tmp_path)
    for _ in range(40):
        if not pid_alive(pid):
            break
        time.sleep(0.1)
    txt = (tmp_path / "w.log").read_text()
    assert "1" in txt, f"worker 未带 {DETACH_ENV} 标记: {txt!r}"


def test_is_detached_child_reads_env(monkeypatch):
    monkeypatch.delenv(DETACH_ENV, raising=False)
    assert is_detached_child() is False
    monkeypatch.setenv(DETACH_ENV, "1")
    assert is_detached_child() is True


def test_pid_file_roundtrip(tmp_path):
    p = tmp_path / "logs" / "x.pid"
    assert read_pid_file(p) is None       # 不存在
    write_pid_file(p, 4242)
    assert read_pid_file(p) == 4242
    p.write_text("not-a-pid", encoding="utf-8")
    assert read_pid_file(p) is None       # 损坏 → None(不抛)


def test_pid_alive_edge_cases():
    assert pid_alive(0) is False
    assert pid_alive(-1) is False
    assert pid_alive(os.getpid()) is True


# --------------------------------------------------------------------------
# CLI 接线
# --------------------------------------------------------------------------

def test_cli_run_accepts_detach_flag():
    sys.path.insert(0, str(SKILL / "scripts"))
    import modelevo as cli

    p = cli.build_parser()
    a = p.parse_args(["run", "--trial", "t1", "--detach"])
    assert a.detach is True and a.no_detach is False
    a2 = p.parse_args(["run", "--trial", "t1"])
    assert a2.detach is False and a2.no_detach is False, \
        "缺省必须是前台(既有调用方行为不变)"
    a3 = p.parse_args(["run-generation", "--generation", "3", "--detach"])
    assert a3.detach is True


def test_cli_main_records_raw_argv(monkeypatch, tmp_path):
    """main 必须把解析用的 argv 记进 args._raw_argv,detach 才能重建命令。

    worker 命令 = 同一个 CLI 命令去掉 --detach;若用 sys.argv 重建,
    测试/嵌入式调用(直接传 argv=)会拿到 pytest 的参数。
    """
    sys.path.insert(0, str(SKILL / "scripts"))
    import modelevo as cli

    captured = {}

    def _fake(args):
        captured["raw"] = list(getattr(args, "_raw_argv", []))
        return 0

    p = cli.build_parser()
    orig = p.parse_args

    def _spy(argv=None, **kw):
        a = orig(argv, **kw)
        a.func = _fake
        return a

    monkeypatch.setattr(p, "parse_args", _spy)
    monkeypatch.setattr(cli, "build_parser", lambda: p)
    cli.main(["run", "--trial", "t1", "--detach"])
    assert captured["raw"] == ["run", "--trial", "t1", "--detach"]
