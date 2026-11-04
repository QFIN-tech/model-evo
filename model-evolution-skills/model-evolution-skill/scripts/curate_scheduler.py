#!/usr/bin/env python
"""经验库整理的周期调度(v2.18.2,dev 文档 §23.8)。

本机容器没有 cron 守护进程(PID 1 是 s6-svscan,用户无 sudo),所以与经验库服务一样以
用户态后台进程运行、手动启停:到点调用 ``scripts/curate_worker.py``,跑完再睡到下一次。
容器重启后需重新 ``start``(与 ``library serve --daemon`` 相同)。

    python scripts/curate_scheduler.py start --model claude-opus-5-5 --weekday sat --at 02:00
    python scripts/curate_scheduler.py status
    python scripts/curate_scheduler.py stop
    python scripts/curate_scheduler.py run-now --model claude-opus-5-5   # 立即跑一轮(前台)

状态目录缺省 ``<库根>/curation/``:``scheduler.pid``、``scheduler.json``(配置与下次运行时间)、
``scheduler.log``、每轮的 ``runs/<时间>/``(worker 的上下文、决议与日志)。
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
WORKER = SKILL / "scripts" / "curate_worker.py"
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _default_state_dir() -> Path:
    sys.path.insert(0, str(SKILL / "scripts"))
    sys.path.insert(0, str(SKILL))
    from _mvo.cmd_library import library_settings

    return Path(library_settings()["root"]) / "curation"


def next_run(now: _dt.datetime, weekday: int, hh: int, mm: int) -> _dt.datetime:
    """``now`` 之后(不含)最近的一个 weekday hh:mm。"""
    days = (weekday - now.weekday()) % 7
    cand = now.replace(hour=hh, minute=mm, second=0, microsecond=0) + _dt.timedelta(days=days)
    return cand if cand > now else cand + _dt.timedelta(days=7)


def _log(state: Path, msg: str) -> None:
    with (state / "scheduler.log").open("a", encoding="utf-8") as f:
        f.write(f"{_dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pid(state: Path) -> int | None:
    try:
        pid = int((state / "scheduler.pid").read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if _alive(pid) else None


def run_once(state: Path, model: str, max_groups: int) -> int:
    work = state / "runs" / _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    work.mkdir(parents=True, exist_ok=True)
    _log(state, f"run start model={model} max_groups={max_groups} work={work}")
    with (work / "worker.out").open("w", encoding="utf-8") as out:
        rc = subprocess.call([sys.executable, str(WORKER), "--model", model,
                              "--max-groups", str(max_groups), "--work-dir", str(work)],
                             cwd=SKILL, stdout=out, stderr=subprocess.STDOUT)
    _log(state, f"run end rc={rc} work={work}")
    return rc


def loop(state: Path, cfg: dict) -> None:
    # time.sleep 被信号打断后会自动续睡(PEP 475),stop 要等到这一段睡完;
    # Event.wait 能被处理函数里的 set() 立即唤醒
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    _log(state, f"scheduler up pid={os.getpid()} cfg={json.dumps(cfg, ensure_ascii=False)}")
    while not stop.is_set():
        nxt = next_run(_dt.datetime.now(), cfg["weekday"], cfg["hh"], cfg["mm"])
        (state / "scheduler.json").write_text(json.dumps(
            {**cfg, "pid": os.getpid(), "next_run": nxt.isoformat(timespec="minutes")},
            ensure_ascii=False, indent=1), encoding="utf-8")
        # 分段等待:机器休眠 / 改时间后按墙钟重新判断
        while not stop.is_set() and _dt.datetime.now() < nxt:
            stop.wait(min(60.0, max(1.0, (nxt - _dt.datetime.now()).total_seconds())))
        if stop.is_set():
            break
        try:
            run_once(state, cfg["model"], cfg["max_groups"])
        except Exception as exc:  # noqa: BLE001 — 一轮失败不影响下一轮
            _log(state, f"run error {type(exc).__name__}: {exc}")
    _log(state, "scheduler down")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("start", "run-now", "_loop"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True, help="原样传给 claude --model")
        p.add_argument("--max-groups", type=int, default=40)
        p.add_argument("--state-dir", default=None)
        if name != "run-now":
            p.add_argument("--weekday", choices=WEEKDAYS, default="sat")
            p.add_argument("--at", default="02:00", help="本机时间 HH:MM")
    for name in ("status", "stop"):
        sub.add_parser(name).add_argument("--state-dir", default=None)
    args = ap.parse_args(argv)
    state = Path(args.state_dir) if args.state_dir else _default_state_dir()
    state.mkdir(parents=True, exist_ok=True)

    if args.cmd == "status":
        cfg = {}
        try:
            cfg = json.loads((state / "scheduler.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        print(json.dumps({"running": _pid(state) is not None, "pid": _pid(state),
                          "state_dir": str(state), **{k: cfg.get(k) for k in
                                                      ("model", "weekday_name", "at",
                                                       "max_groups", "next_run")}},
                         ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "stop":
        pid = _pid(state)
        if pid:
            os.kill(pid, signal.SIGTERM)
        print(json.dumps({"stopped": bool(pid), "pid": pid}))
        return 0
    if args.cmd == "run-now":
        return run_once(state, args.model, args.max_groups)

    hh, mm = (int(x) for x in args.at.split(":"))
    cfg = {"model": args.model, "max_groups": args.max_groups,
           "weekday": WEEKDAYS.index(args.weekday), "weekday_name": args.weekday,
           "at": args.at, "hh": hh, "mm": mm}
    if args.cmd == "_loop":
        (state / "scheduler.pid").write_text(str(os.getpid()))
        loop(state, cfg)
        return 0
    if _pid(state):
        print(json.dumps({"error": "already running", "pid": _pid(state)}))
        return 1
    # start:脱离终端在后台运行
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_loop",
                             "--model", args.model, "--max-groups", str(args.max_groups),
                             "--weekday", args.weekday, "--at", args.at,
                             "--state-dir", str(state)],
                            cwd=SKILL, stdin=subprocess.DEVNULL,
                            stdout=(state / "scheduler.out").open("a"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(50):
        if _pid(state) == proc.pid:
            break
        time.sleep(0.1)
    print(json.dumps({"started": True, "pid": proc.pid, "state_dir": str(state),
                      "next_run": next_run(_dt.datetime.now(), cfg["weekday"], hh, mm)
                      .isoformat(timespec="minutes")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
