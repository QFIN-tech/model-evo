# -*- coding: utf-8 -*-
"""finalize_session.py: 进化收口 -- session report.md + state 置 finalized。

用法:
    python finalize_session.py --session-dir <session_dir>

报告内容:
  1. 总览(baseline vs champion 三档 AUC/KS, 累计 OOT 增益, 终止原因)
  2. 已接受特征表(fid / 假设 / 单变量 AUC / 提出轮次)
  3. 进化史(逐轮 commit/rollback/no_accept 台账汇总)
  4. 产物索引(哪里拿导出包/反馈/画像)

退出码: 0 成功 / 2 session 状态不可用
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (注入 sys.path, 勿删)

from evo_core import paths, state_io

PRODUCED_BY = "feature-mining-skills/feature-evolution-evaluation"


def _exit(code: int, msg: str) -> int:
    print(msg, file=sys.stderr)
    return code


def _fmt(v, nd: int = 6) -> str:
    if v is None:
        return "N/A"
    try:
        if v != v:  # NaN
            return "N/A"
    except TypeError:
        pass
    return ("%." + str(nd) + "f") % v


def _round_history(ledger: list) -> list:
    """逐轮结算记录(commit/rollback/no_accept)。"""
    history = {}
    for r in ledger:
        rtype = r.get("type")
        if rtype in ("round_commit", "round_rollback", "round_no_accept"):
            history.setdefault(int(r.get("round") or 0), []).append(r)
    return [history[k] for k in sorted(history)]


def build_report_md(session_dir) -> str:
    state = state_io.load_state(session_dir)
    ledger = state_io.read_ledger(session_dir)

    baseline = {
        s: state_io.read_json(paths.baseline_eval_json(session_dir, s)).get("metrics") or {}
        for s in paths.SPLITS
    }
    champion_path = paths.champion_eval_json(session_dir)
    champion = state_io.read_json(champion_path) if champion_path.exists() else {}
    cham_metrics = champion.get("metrics") or {}
    cham_ks = champion.get("ks") or {}

    name = state.get("mining_name")
    fids = list(state.get("champion_fids") or [])
    n_cand = sum(1 for r in ledger if r.get("type") == "candidate")
    n_pass = sum(1 for r in ledger if r.get("type") == "candidate" and r.get("verdict") == "candidate_pass")

    lines = [
        "# 特征进化报告 - %s" % name,
        "",
        "> 闭环: Feature 生成/组合 -> 固定超参训练 -> G1~G6 关卡 -> Feedback -> 下一轮演进; 只认 Validation/OOT 稳定增益。",
        "",
        "## 1. 总览",
        "",
        "| 档 | baseline AUC | champion AUC | ΔAUC | baseline KS | champion KS |",
        "|---|---|---|---|---|---|",
    ]
    for s in paths.SPLITS:
        b_auc, c_auc = baseline[s].get("auc"), cham_metrics.get(s)
        delta = ("+" if (c_auc is not None and b_auc is not None and c_auc >= b_auc) else "") + _fmt(
            (c_auc - b_auc) if (c_auc is not None and b_auc is not None) else None
        )
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            s, _fmt(b_auc), _fmt(c_auc), delta, _fmt(baseline[s].get("ks")), _fmt(cham_ks.get(s))))
    lines += [
        "",
        "- 累计 OOT 增益: **%s**" % _fmt(
            (state.get("champion_oot_auc") - state.get("baseline_oot_auc"))
            if (state.get("champion_oot_auc") is not None and state.get("baseline_oot_auc") is not None) else None),
        "- 进化轮次: %d | 候选总数: %d | 单候选过关卡: %d | 最终接受: %d" % (
            state.get("current_round"), n_cand, n_pass, state.get("accepted_count")),
        "- 终止原因: %s" % (state.get("stop_reason") or "手动停止"),
        "",
        "## 2. 已接受特征",
        "",
    ]
    if not fids:
        lines += ["本轮进化未产生稳定 OOT 增益特征, champion = baseline。", ""]
    else:
        lines += ["| fid | 提出轮次 | 类别 | 假设 | 单变量AUC |", "|---|---|---|---|---|"]
        for fid in fids:
            meta_path = paths.accepted_meta_json(session_dir, fid)
            meta = state_io.read_json(meta_path) if meta_path.exists() else {}
            single_auc = (meta.get("single_feature") or {}).get("train_auc")
            lines.append("| %s | %s | %s | %s | %s |" % (
                fid, meta.get("round", "-"), meta.get("category") or "-",
                str(meta.get("hypothesis") or "")[:80], _fmt(single_auc, 4)))
        lines.append("")

    lines += ["## 3. 进化史(轮级结算)", ""]
    history = _round_history(ledger)
    if not history:
        lines += ["(无轮级记录)", ""]
    else:
        lines += ["| 轮 | 结算 | 说明 |", "|---|---|---|"]
        for round_records in history:
            for r in round_records:
                if r["type"] == "round_commit":
                    note = "接受 %s, champion OOT=%s (+%s)" % (
                        ",".join(r.get("accepted_fids") or []), _fmt(r.get("champion_oot_auc")), _fmt(r.get("gain_oot")))
                elif r["type"] == "round_rollback":
                    note = "回滚 %s: %s" % (",".join(r.get("cids") or []), r.get("reason"))
                else:
                    note = r.get("note") or ""
                lines.append("| %s | %s | %s |" % (r.get("round"), r["type"], note))
        lines.append("")

    lines += [
        "## 4. 产物索引",
        "",
        "| 产物 | 位置 |",
        "|---|---|",
        "| 特征画像 | `profile/feature-profile.csv`, `profile/data-desc.md` |",
        "| baseline 评估 | `baseline/evaluation/` |",
        "| champion 评估 | `evolution/accepted/_champion_eval.json` |",
        "| 候选与判定 | `evolution/rounds/rXXX/` |",
        "| 候选台账 | `evolution/ledger.jsonl` |",
        "| 最新反馈 | `evolution/feedback/latest-feedback.md` |",
        "| 导出交付包 | `evolution/export/accepted-features/` |",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="进化收口: report.md + state=finalized")
    parser.add_argument("--session-dir", required=True)
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    try:
        md = build_report_md(session_dir)
    except FileNotFoundError as e:
        return _exit(2, str(e))

    paths.report_md(session_dir).write_text(md, encoding="utf-8")
    state_io.update_state(session_dir, status="finalized")
    state_io.write_manifest(
        session_dir, PRODUCED_BY,
        ["report.md", "evolution/state.json", "evolution/ledger.jsonl"],
        overview=state_io.load_state(session_dir),
    )
    print("report 已生成: %s" % paths.report_md(session_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
