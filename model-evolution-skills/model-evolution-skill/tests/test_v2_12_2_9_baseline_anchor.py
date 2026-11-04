"""回归测试:v2.12.2.9 baseline 跨算法锚点。

事故来源(2026-09-18,req-cnt-t7 run):
    ``cmd_baseline`` 里 ``baseline_id`` 在 ``runner.run()`` **返回之后**
    才落盘,而 ``runner.run()`` 内部已经跑完了评估 —— 于是每个 baseline
    trial 评估时 ``baseline_score`` 都是 ``None``,
    ``evaluation.delta.vs_baseline`` 恒空。

    实测后果:
      - trial-001 的 ``evaluation.delta`` 是空 dict;
      - 每次 run 都要靠 ``tools/backfill_baseline.py`` 事后补算一遍,
        而该工具的 docstring 只承认"baseline 被 kill"一种场景 ——
        **正常退出同样触发**(前序 run 的 leaderboard `vs_base` 列全空);
      - backfill 会重跑整条评估链(实测多花 61s + 一次人工判断)。

    本文件锁定修复:锚点在创建/评估前解析并传入 ``runner.run``,
    且复用路径同样补传(否则重入时已评估的 trial 永远补不上 delta)。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT))

_CMD_STARTUP = SKILL_ROOT / "scripts" / "_mvo" / "cmd_startup.py"


def _cmd_baseline_ast() -> ast.FunctionDef:
    tree = ast.parse(_CMD_STARTUP.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "cmd_baseline":
            return node
    raise AssertionError("cmd_startup.py 里找不到 cmd_baseline")


def _runner_run_calls(fn: ast.FunctionDef) -> list[ast.Call]:
    """cmd_baseline 内形如 `runner.run(...)` 的全部调用。"""
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "run" \
                and isinstance(f.value, ast.Name) and f.value.id == "runner":
            out.append(node)
    return out


def test_every_runner_run_in_baseline_passes_baseline_score():
    """**核心回归**:cmd_baseline 里每一处 runner.run 都必须带锚点。

    旧实现只有 ``runner.run(trial, plan=plan, slot_index=algo_i)``,
    锚点参数缺失 → delta_vs_baseline 恒空。
    """
    calls = _runner_run_calls(_cmd_baseline_ast())
    assert calls, "cmd_baseline 里没有 runner.run 调用?实现已变,请复核本测试"
    for c in calls:
        kw = {k.arg for k in c.keywords if k.arg}
        assert "baseline_score" in kw, (
            f"cmd_baseline 第 {c.lineno} 行的 runner.run 未传 baseline_score:"
            f" keywords={sorted(kw)}")


def _assign_lines(fn: ast.FunctionDef, name: str) -> list[int]:
    """cmd_baseline 内对 ``name`` 的全部赋值行号(含元组解包)。"""
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                elts = tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]
                if any(isinstance(e, ast.Name) and e.id == name for e in elts):
                    out.append(node.lineno)
    return sorted(out)


def test_anchor_is_resolved_before_each_run_call():
    """锚点必须**先于**每一处 ``runner.run`` 确定。

    顺序是 v2.12.2.9 修复的全部要点:旧实现把 ``baseline_id`` 写在
    ``runner.run()`` **之后**,而 run 内部已经评估完毕。

    v2.15.7(两阶段执行)起口径收紧为:``baseline_id`` 在任何训练之前
    就锁定为 trial-001(run-state 已有则沿用),每一轮的锚点分数由
    ``_anchor_scores()`` 现读。因此每处 runner.run 之前必须既有
    ``baseline_id`` 的锁定赋值,又有 ``_bs`` 的取值。
    """
    fn = _cmd_baseline_ast()
    lock_lines = [ln for ln in _assign_lines(fn, "baseline_id")]
    bs_lines = _assign_lines(fn, "_bs")
    assert lock_lines and bs_lines, "cmd_baseline 缺锚点锁定或锚点分数取值"
    first_run = min(c.lineno for c in _runner_run_calls(fn))
    assert any(ln < first_run for ln in lock_lines), (
        f"baseline_id 必须在第一处 runner.run(第 {first_run} 行)之前锁定")
    for call in _runner_run_calls(fn):
        assert any(ln < call.lineno for ln in bs_lines), (
            f"第 {call.lineno} 行的 runner.run 之前没有锚点分数取值")
    src = ast.get_source_segment(_CMD_STARTUP.read_text(encoding="utf-8"), fn)
    assert "_anchor_scores()" in src and "load_parent_scores(run_dir, baseline_id)" in src


def test_root_trial_gets_no_anchor_but_later_ones_do():
    """锚点 trial 自身不传锚点,其余 trial 传。

    自比会构成退化锚点(delta 恒 0),必须显式排除自己。v2.15.7 起由
    ``trial.trial_id == baseline_id`` 分支给 ``_bs, _bv = None, None``。
    """
    src = _CMD_STARTUP.read_text(encoding="utf-8")
    fn_src = ast.get_source_segment(src, _cmd_baseline_ast())
    assert "if trial.trial_id == baseline_id:" in fn_src, \
        "缺少『不拿自己当锚点』的守卫"
    guard = fn_src.split("if trial.trial_id == baseline_id:")[1].split("else:")[0]
    assert "_bs, _bv = None, None" in guard


def test_reuse_path_also_backfills_delta():
    """复用路径必须能补 delta:否则重入 run 时 EVALUATED trial 永远空着。"""
    src = _CMD_STARTUP.read_text(encoding="utf-8")
    fn_src = ast.get_source_segment(src, _cmd_baseline_ast())
    assert "done[algo]" in fn_src
    # 复用分支里出现"delta 为空则重评估"的判断
    assert "evaluation" in fn_src and "delta" in fn_src, \
        "复用分支缺少 delta 补算逻辑"


def test_load_parent_scores_is_imported():
    """锚点预测要经 load_parent_scores 读取(与 cmd_run_trial 同口径)。

    cmd_run_trial 用的是 _resolve_baseline_id + load_parent_scores 组合;
    baseline 必须复用同一对 helper,不能自造读取路径。
    """
    src = _CMD_STARTUP.read_text(encoding="utf-8")
    assert "load_parent_scores" in src
    assert "_resolve_baseline_id" in src
    # 且确实是从 runner 模块导入,不是本地糊的
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported |= {a.name for a in node.names}
    assert "load_parent_scores" in imported
    assert "_resolve_baseline_id" in imported
