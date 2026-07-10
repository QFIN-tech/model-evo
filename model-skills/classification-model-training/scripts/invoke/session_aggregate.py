# -*- coding: utf-8 -*-
"""会话级聚合薄包装: 调 classification-model-comparison 的 aggregate_session_comparison.py。

每个 run 跑完自身 comparison/ 后, 调用本模块刷新 session 级 model-comparison/,
对 session 内所有 run 的 eval JSON 做 N-way 横向对比, 产三件套 + _manifest.json。

容错策略: 聚合脚本缺失 / subprocess 失败 / session_dir 不存在时,
仍 mkdir model-comparison/ 并落 fallback _manifest.json (status=skipped/failed),
保证 SKILL.md 声明的 model-comparison/ 产物始终存在, 不影响主流程。
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 本文件路径: classification-model-training/scripts/invoke/session_aggregate.py
# 仓库根 = parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGGREGATE = (
    _REPO_ROOT / "classification-model-comparison"
    / "scripts" / "aggregate_session_comparison.py"
)


def _write_fallback_manifest(
    out_dir: Path,
    status: str,
    reason: str,
    produced_by: str,
) -> None:
    """脚本缺失 / 失败时, 仍 mkdir + 落 fallback _manifest.json。

    Args:
        out_dir: <session_dir>/model-comparison 目录
        status: "skipped" (脚本缺失/无 eval) 或 "failed" (subprocess 非 0)
        reason: 失败/跳过原因
        produced_by: manifest 来源标识
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "session-aggregate",
        "produced_by": produced_by,
        "status": status,
        "reason": reason,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "files": [],
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[invoke_session_aggregate] 落 fallback manifest (status={status}): "
        f"{out_dir / '_manifest.json'}",
        file=sys.stderr,
    )


def invoke_session_aggregate(
    session_dir: str,
    produced_by: str = "skills/model-training",
) -> Optional[bool]:
    """调 aggregate_session_comparison.py 刷新 session 级对比。

    Args:
        session_dir: session_dir 绝对路径
        produced_by: manifest 来源标识

    Returns:
        True 成功; False 失败 (脚本缺失/失败但已落 fallback manifest); None 无 session_dir
    """
    out_dir = Path(session_dir) / "model-comparison"

    if not _AGGREGATE.exists():
        print(
            f"[invoke_session_aggregate] 找不到 aggregate_session_comparison.py: {_AGGREGATE}",
            file=sys.stderr,
        )
        _write_fallback_manifest(
            out_dir, status="skipped",
            reason=f"aggregate script missing: {_AGGREGATE}",
            produced_by=produced_by,
        )
        return False

    cmd = [
        sys.executable, str(_AGGREGATE),
        "--session-dir", str(session_dir),
        "--produced-by", produced_by,
    ]
    print(f"[invoke_session_aggregate] 刷新 session 级对比: {session_dir}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stdout)
        print(
            f"[invoke_session_aggregate] 聚合失败 (code={result.returncode}), "
            f"session 级 comparison 未刷新, 但不影响主流程:\n{result.stderr}",
            file=sys.stderr,
        )
        _write_fallback_manifest(
            out_dir, status="failed",
            reason=f"subprocess returncode={result.returncode}; stderr={result.stderr[:500]}",
            produced_by=produced_by,
        )
        return False
    print(result.stdout)
    return True
