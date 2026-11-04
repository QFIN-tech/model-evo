"""统一日志格式(§20)与安全 JSON 落盘工具。

所有日志与落盘时间统一使用北京时间(UTC+8,固定偏移,不随系统时区漂移)。
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# 北京时间:UTC+8 固定偏移(无夏令时)
CN_TZ = timezone(timedelta(hours=8), name="UTC+8")

LOG_FORMAT = (
    "%(asctime)s [%(filename)s:%(lineno)d][%(funcName)s] %(levelname)s: %(message)s"
)
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _CNFormatter(logging.Formatter):
    """converter 使用北京时间的 Formatter。"""

    def __init__(self, fmt: str = LOG_FORMAT, datefmt: str = LOG_DATEFMT):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def formatTime(self, record, datefmt=None):  # noqa: N802(logging API)
        return datetime.fromtimestamp(
            record.created, tz=CN_TZ).strftime(datefmt or self.datefmt)


def get_logger(name: str = "modelevo", level: int = logging.INFO) -> logging.Logger:
    """返回带统一格式的 logger(幂等,重复调用不叠加 handler)。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_CNFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def now_iso() -> str:
    """北京时间戳,所有落盘时间统一使用该格式(UTC+8)。"""
    return datetime.now(CN_TZ).strftime("%Y-%m-%dT%H:%M:%S+08:00")


_SAFE_TYPES = (int, float, str, bool, type(None))


def to_jsonable(obj: Any) -> Any:
    """递归转换为可 JSON 序列化结构(处理 numpy 标量/数组/日期/Path)。"""
    import numpy as np

    if obj is None or isinstance(obj, _SAFE_TYPES):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    return str(obj)


def dump_json(path: Path, payload: Any, *, sort_keys: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2,
                      sort_keys=sort_keys, default=str)
    _atomic_write(path, text + "\n")


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(record), ensure_ascii=False,
                           default=str) + "\n")


def dump_yaml(path: Path, payload: Any) -> None:
    """YAML 落盘;对中文不转义,键序保持插入序。"""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(to_jsonable(payload), allow_unicode=True,
                          sort_keys=False, default_flow_style=False, width=120)
    _atomic_write(path, text)


def _atomic_write(path: Path, text: str) -> None:
    """原子落盘;tmp 名含 pid+tid,并行 trial 同时写同一共享文件
    (run-state.json / breaker-state.json)不会互相覆盖/消费对方的 tmp。
    """
    tmp = path.parent / (
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        # replace 成功后 tmp 已不存在;失败残留也不留死文件
        tmp.unlink(missing_ok=True)


@contextmanager
def locked_state_write(path: Path):
    """跨进程共享状态文件(run-state.json 等)的读-改-写临界区。

    flock 文件锁:同进程多线程(GenerationExecutor 线程池)与多个
    run_launch.py 进程并发收尾时,序列化 load→update→dump 全过程,
    避免丢更新;锁本身不阻塞读,只约束写。yield 前锁已持有。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "a+b") as lk:
        fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lk.fileno(), fcntl.LOCK_UN)


def load_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(record), ensure_ascii=False,
                           default=str) + "\n")


def read_jsonl(path: Path) -> list:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
