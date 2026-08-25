# -*- coding: utf-8 -*-
"""build_feedback.py: 汇总当前进化状态 -> feedback/latest-feedback.md(确定性, 无 LLM)。

反馈是闭环的「学习信号」: 下一轮候选生成前, 编排层(LLM)必须先读本文件,
内容包括:
  1. champion 状态(baseline vs champion, 累计增益, 接受数, 连续无接受轮数, 终止原因)
  2. 最近一轮候选判定表(哪个关卡挂的、差多少, 含增益数字)
  3. 已接受特征清单(fid + 假设 + 单变量 AUC)
  4. 探索引导: 未被任何候选用过的原始特征 / 已注册语义特征
  5. 下一轮配额与黑名单(近几轮指纹, 防重复提案)

退出码: 0 成功 / 2 session 状态不可用
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (注入 sys.path, 勿删)

from evo_core import paths, state_io

PRODUCED_BY = "feature-mining-skills/feature-evolution-evaluation"
_RECENT_LEDGER_ROWS = 30
_UNUSED_FEATURE_HINT_N = 20


def _fmt(v, nd: int = 6) -> str:
    if v is None:
        return "N/A"
    try:
        if v != v:  # NaN
            return "N/A"
    except TypeError:
        pass
    return ("%." + str(nd) + "f") % v


def _collect_round_results(session_dir, round_no: int) -> list:
    rdir = paths.results_dir(session_dir, round_no)
    if not rdir.is_dir():
        return []
    return [state_io.read_json(p) for p in sorted(rdir.glob("*.result.json"))]


def _unused_features(session_dir, state: dict, ledger: list) -> list:
    """候选从未用过的原始特征(画像顺序), 供探索引导。"""
    used = set()
    for r in ledger:
        for c in r.get("base_features") or []:
            used.add(str(c))
    try:
        import pandas as pd

        profile = pd.read_csv(paths.feature_profile_csv(session_dir))
        all_features = [str(x) for x in profile["feature"].tolist()]
    except Exception:
        snap = state.get("config_snapshot") or {}
        all_features = []
    return [f for f in all_features if f not in used]


def _semantic_items(session_dir) -> list:
    reg_path = paths.semantic_registry_json(session_dir)
    if not reg_path.exists():
        return []
    return state_io.read_json(reg_path).get("items", [])


def build_feedback_md(session_dir) -> str:
    """生成 latest-feedback.md 全文。"""
    state = state_io.load_state(session_dir)
    ledger = state_io.read_ledger(session_dir)
    round_no = int(state.get("current_round") or 0)

    base_auc, cham_auc = state.get("baseline_oot_auc"), state.get("champion_oot_auc")
    cum_gain = (cham_auc - base_auc) if (cham_auc is not None and base_auc is not None) else None
    evo = (state.get("config_snapshot") or {}).get("evolution") or {}
    k = int(evo.get("candidates_per_round") or 4)

    lines = ["# 进化反馈(第 %d 轮后)" % round_no, ""]

    # 1. champion 状态
    lines += [
        "## 1. 当前状态",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| baseline OOT AUC | %s |" % _fmt(base_auc),
        "| champion OOT AUC | %s |" % _fmt(cham_auc),
        "| 累计 OOT 增益 | %s |" % _fmt(cum_gain),
        "| 已接受特征数 | %d |" % int(state.get("accepted_count") or 0),
        "| 连续无接受轮数 | %d / %s |" % (int(state.get("no_accept_streak") or 0), evo.get("no_accept_rounds")),
        "| 终止原因 | %s |" % (state.get("stop_reason") or "未命中(可继续)"),
        "",
    ]

    # 2. 最近一轮判定
    results = _collect_round_results(session_dir, round_no)
    lines += ["## 2. 第 %d 轮候选判定" % round_no, ""]
    if not results:
        lines += ["(本轮无评估记录)", ""]
    else:
        lines += ["| cid | 类别 | 判定 | 失败关卡 | 原因/增益 |", "|---|---|---|---|---|"]
        for r in results:
            if r.get("gate_failed") in (None, "-") and r.get("verdict") == "candidate_pass":
                note = "gain_oot=%s" % _fmt((r.get("gates", {}).get("G4_G5") or {}).get("gain_oot"))
            else:
                note = str(r.get("reason") or "-")[:70]
            lines.append("| %s | %s | %s | %s | %s |" % (
                r.get("cid"), r.get("category") or "-", r.get("verdict"),
                r.get("gate_failed") or "-", note))
        lines.append("")

    # 3. 已接受特征
    lines += ["## 3. 已接受特征(champion)", ""]
    fids = list(state.get("champion_fids") or [])
    if not fids:
        lines += ["(暂无, champion = baseline)", ""]
    else:
        lines += ["| fid | 假设 | 单变量AUC |", "|---|---|---|"]
        for fid in fids:
            meta_path = paths.accepted_meta_json(session_dir, fid)
            if meta_path.exists():
                meta = state_io.read_json(meta_path)
                single_auc = (meta.get("single_feature") or {}).get("train_auc")
                lines.append("| %s | %s | %s |" % (fid, str(meta.get("hypothesis") or "")[:60], _fmt(single_auc, 4)))
            else:
                lines.append("| %s | (元数据缺失) | - |" % fid)
        lines.append("")

    # 4. 探索引导
    unused = _unused_features(session_dir, state, ledger)
    lines += ["## 4. 探索引导", ""]
    if unused:
        lines += ["尚未被任何候选使用过的原始特征(前 %d 个): %s" % (_UNUSED_FEATURE_HINT_N, ", ".join(unused[:_UNUSED_FEATURE_HINT_N])), ""]
    else:
        lines += ["所有原始特征均已被候选使用过, 建议转向组合/交互/非线性加工。", ""]
    sem_items = _semantic_items(session_dir)
    if sem_items:
        lines += ["已注册语义特征(semantic-feature, 可作为候选原料):", ""]
        lines += ["| 语义特征 | 来源文本列 | head | OOT AUC |", "|---|---|---|---|"]
        for it in sem_items:
            m = (it.get("metrics") or {})
            lines.append("| %s | %s | %s | %s |" % (
                ",".join(it.get("output", {}).get("columns", []) or [it.get("name")]),
                ",".join(it.get("source_text_cols") or []), it.get("head"), _fmt(m.get("oot_auc"), 4)))
        lines.append("")

    # 5. 下一轮约束
    recent_fps = state_io.recent_fingerprints(session_dir)
    lines += [
        "## 5. 下一轮要求",
        "",
        "- 候选数 K=%d, 其中 **>=1 个 explore**(新原料/新算子/新思路) + **>=1 个 exploit**(在已接受特征或成功方向上加工)" % k,
        "- 每个候选必须写清 hypothesis(一句话假设)与 base_features(用到的列)",
        "- 近几轮已尝试的假设指纹共 %d 个, **不要重复相同原料+相同假设**的提案" % len(recent_fps),
        "- 只认 OOT 增益: train 上再好, OOT 不涨一律会被拒",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成 latest-feedback.md")
    parser.add_argument("--session-dir", required=True)
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    try:
        md = build_feedback_md(session_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    out = paths.latest_feedback_md(session_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    state_io.write_manifest(
        out.parent, PRODUCED_BY, ["latest-feedback.md"],
        overview={"current_round": state_io.load_state(session_dir).get("current_round")},
    )
    print("feedback 已更新: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
