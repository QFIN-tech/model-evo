# -*- coding: utf-8 -*-
"""logs/ 子目录产出: run.log + {entry}.log + _manifest.json。

`tee_stdout(layout)` 是一个 contextmanager,把进入块期间的 stdout/stderr 同时
复制到屏幕和 logs/run.log,便于事后回溯训练日志。

`process_tee(log_path)` 是进程级 contextmanager,把整个 main 生命周期的 stdout/stderr
额外镜像到 logs/{entry}.log(如 run_build.log / run_tuning.log / select_features.log),
覆盖 RunLayout.create 之前/之后的输出(import 警告、run_dir print、完成回执等)。
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, TextIO

from stages.layout import RunLayout, write_manifest


class _Tee:
    """同时写多个底层流;flush 各自传播。"""

    def __init__(self, streams: List[TextIO]) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:  # 部分库会调用,兼容一下
        return False


@contextmanager
def tee_stdout(layout: RunLayout) -> Iterator[Path]:
    """把 stdout / stderr 同时镜像到 logs/run.log。

    用法:
        with tee_stdout(layout) as log_path:
            ...  # 训练 / 评估 / 落盘
    """
    log_path = layout.logs_dir / "run.log"
    fh = log_path.open("w", encoding="utf-8", buffering=1)
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = _Tee([real_out, fh])
    sys.stderr = _Tee([real_err, fh])
    try:
        yield log_path
    finally:
        sys.stdout = real_out
        sys.stderr = real_err
        fh.close()


@contextmanager
def process_tee(log_path: Path) -> Iterator[Path]:
    """进程级 stdout/stderr 镜像到 log_path(如 logs/run_build.log)。

    与 tee_stdout 的差异:
    - tee_stdout 只覆盖 `with tee_stdout(layout):` 块内(训练核心阶段),产 run.log
    - process_tee 覆盖 RunLayout.create 之后到 main() 末尾的全部输出,产 {entry}.log,
      含 import 警告、run_dir print、data_dir 推断、完成回执等 tee 之外的内容

    两者可嵌套(process_tee 在外, tee_stdout 在内),日志会同时落到两个文件;
    退出时无条件恢复 stdout/stderr 并关闭文件句柄。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("w", encoding="utf-8", buffering=1)
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = _Tee([real_out, fh])
    sys.stderr = _Tee([real_err, fh])
    try:
        yield log_path
    finally:
        sys.stdout = real_out
        sys.stderr = real_err
        fh.close()


def finalize_logs_stage(layout: RunLayout, produced_by: Optional[str] = None,
                        extra_logs: Optional[List[Path]] = None) -> None:
    """收尾: 写 logs/_manifest.json。run.log 由 tee_stdout 已落盘。

    Args:
        layout: RunLayout 实例
        produced_by: 产出方标识(如 "skills/model-tuning")
        extra_logs: 额外日志文件路径(如 run_build.log / run_tuning.log),
                    一并写入 manifest files 列表
    """
    log_files = [layout.logs_dir / "run.log"]
    if extra_logs:
        log_files.extend(extra_logs)
    write_manifest(
        layout.logs_dir,
        stage="logs",
        files=log_files,
        extra={"captured_streams": ["stdout", "stderr"]},
        produced_by=produced_by,
    )
