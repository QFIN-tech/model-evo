"""v17.3 修复批次:guard subagent 身份判定失效(回归测试)。

事故(2026-09 实测,run example-seq-v3-400w-v19.1):
Claude Code 的 PreToolUse payload 中,主会话与 subagent 的
`transcript_path` 指向同一个主会话转录(该文件内 `isSidechain` 恒为
false)。旧 `_in_subagent()` 只认 transcript tail-scan → 恒判 False
→ subagent 被按主会话口径拦截,演进循环整体无法启动(实测该 run 的
subagent 被拒 27 次,phase 卡在 INITED)。

可靠信号是 payload 自带的 `agent_id` / `agent_type`:实测仅 subagent
的 payload 携带。本测试锁定该口径,并断言主会话拦截强度不变。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]


def _load_guard():
    spec = importlib.util.spec_from_file_location(
        "modelevo_guard_v17dot3", SKILL / "assets" / "guard_main_session.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return guard


def test_in_subagent_prefers_payload_agent_id() -> None:
    """payload 携带 agent_id/agent_type → 判为 subagent(主路径)。"""
    print("== guard:payload agent_id 优先判定(v17.3) ==")
    guard = _load_guard()

    # transcript_path 指向主会话转录(事故形态):两个键任一存在即 subagent
    main_only = {"transcript_path": "/nonexistent/main.jsonl"}
    assert guard._in_subagent(main_only) is False, "裸主会话形态应判 False"

    assert guard._in_subagent(
        {"agent_id": "a123", "transcript_path": "/nonexistent/main.jsonl"}
    ) is True, "有 agent_id 应判 subagent"

    assert guard._in_subagent(
        {"agent_type": "general-purpose",
         "transcript_path": "/nonexistent/main.jsonl"}
    ) is True, "有 agent_type 应判 subagent"

    # 无 transcript_path 也无 agent_id → 保守按主会话
    assert guard._in_subagent({}) is False, "字段全缺应判 False"

    print("   payload agent_id/agent_type 判定正确")


def test_subagent_evolve_command_allowed_main_denied() -> None:
    """演进命令:subagent 放行,主会话拦截(修复前后差异点)。"""
    print("== guard:subagent 演进命令放行 / 主会话拦截 ==")
    import io
    import json
    import os

    guard = _load_guard()
    # profile 已改为主会话直接执行；用仍属于演进循环的命令验证边界。
    evolve = "python <root>/scripts/modelevo.py " + \
        "run-generation --generation 1"

    def run(payload: dict) -> bool:
        saved = {k: os.environ.get(k) for k in
                 ("CLAUDE_PROJECT_DIR", "MODELEVO_ALLOW_MAIN_EXEC",
                  "MODELEVO_ALLOW_SOURCE_READ")}
        os.environ["CLAUDE_PROJECT_DIR"] = "/proj"
        for k in ("MODELEVO_ALLOW_MAIN_EXEC", "MODELEVO_ALLOW_SOURCE_READ"):
            os.environ.pop(k, None)
        buf = io.StringIO()
        old_out, old_in = sys.stdout, sys.stdin
        allowed = False
        try:
            sys.stdout = buf
            sys.stdin = io.StringIO(json.dumps(payload))
            try:
                guard.main()
                allowed = True
            except SystemExit as e:
                allowed = e.code == 0
        finally:
            sys.stdout, sys.stdin = old_out, old_in
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return allowed

    base = {"tool_name": "Bash", "tool_input": {"command": evolve}}

    # 事故形态:transcript_path 指向主会话,但 payload 带 agent_id
    assert run({**base, "agent_id": "a123",
                "transcript_path": "/nonexistent/main.jsonl"}) is True, \
        "subagent 的演进命令必须放行(v17.3 修复点)"

    # 主会话拦截强度不变
    assert run({**base, "transcript_path": "/nonexistent/main.jsonl"}) is False, \
        "主会话执行演进命令必须仍被拦截"

    # 只读命令对主会话仍放行
    assert run({"tool_name": "Bash",
                "tool_input": {"command":
                               "python <root>/scripts/modelevo.py status"},
                "transcript_path": "/nonexistent/main.jsonl"}) is True, \
        "主会话只读命令应放行"

    print("   subagent 放行 / 主会话拦截 边界正确")
