# -*- coding: utf-8 -*-
"""commit_round.py: 轮末 G6 融合模型验证 + champion 更新或整轮回滚。

用法:
    python commit_round.py --session-dir <session_dir> --round 3

流程:
  1. 读 results/*.result.json, 取 verdict=candidate_pass 的 provisional 候选
  2. 无 provisional -> no_accept_streak+1, 直接进终止判定
  3. 有 -> 给候选编 fid, 构造融合帧(champion + provisional 全部特征), 训 G6 融合模型
  4. G6 过(OOT 增益 >= g6_min_oot_gain): 候选代码/元数据落 accepted/, champion 指标
     与 champion_fids 更新, 三档评估落 _champion_eval.json
  5. G6 不过: 整轮回滚(候选留在 rounds/ 与 ledger 里可追溯, 但不入 champion),
     no_accept_streak+1 -- 防贪心顺序接受造成的过拟合
  6. 终止判定: max_rounds / no_accept_rounds / target_oot_gain / time_budget_hours,
     命中则写 state.stop_reason(编排层读到后停止开新轮)

退出码: 0 成功(含回滚) / 2 session 状态不可用 / 4 运行时失败
"""
from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401  (注入 sys.path, 勿删)

from evo_core import candidate_exec, feature_runtime, gates, paths, state_io

PRODUCED_BY = "feature-mining-skills/feature-evolution-evaluation"


def _exit(code: int, msg: str) -> int:
    print(msg, file=sys.stderr)
    return code


def _now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def load_provisional(session_dir, round_no: int) -> list:
    """读本轮 result.json, 返回 provisional 接受的候选(按 cid 排序)。"""
    rdir = paths.results_dir(session_dir, round_no)
    out = []
    if not rdir.is_dir():
        return out
    for rj in sorted(rdir.glob("*.result.json")):
        r = state_io.read_json(rj)
        if r.get("verdict") == "candidate_pass":
            out.append(r)
    return out


def build_fused_frames(session_dir, provisional: list, timeout_s: float) -> tuple:
    """构造 G6 融合帧: champion 帧 + provisional 候选特征(按 cid 顺序编 fid 依次追加)。

    build_frames 返回的帧归本调用链所有, 原地追加 fid 列(大数据量下避免整帧拷贝);
    transform 输入帧每档只构造一次, 后面的候选能看到前面的 fid(与 _apply_accepted 同语义)。

    Returns:
        (frames, contract_meta, fid_assign): fid_assign = [{cid, fid}]
    """
    frames, contract_meta = feature_runtime.build_frames(session_dir)
    state = state_io.load_state(session_dir)
    next_fid = int(state.get("next_fid") or 1)
    fid_assign = []
    input_frames = {s: feature_runtime.candidate_input_frame(df) for s, df in frames.items()}
    for r in provisional:
        fid = "f%03d" % next_fid
        next_fid += 1
        code = paths.candidate_py(session_dir, r["round"], r["cid"]).read_text(encoding="utf-8")
        for split, df in frames.items():
            run = candidate_exec.run_transform(code, input_frames[split], timeout_s=timeout_s)
            if not run["ok"]:
                raise RuntimeError("候选 %s 在 %s 档执行失败: %s" % (r["cid"], split, run["error"]))
            vals = run["series"].to_numpy()
            df[fid] = vals
            input_frames[split][fid] = vals
        fid_assign.append({"cid": r["cid"], "fid": fid, "result": r})
    contract_meta = dict(contract_meta)
    contract_meta["accepted_fids"] = list(contract_meta["accepted_fids"]) + [a["fid"] for a in fid_assign]
    return frames, contract_meta, fid_assign


def write_champion_eval(session_dir, state: dict, g6_details: dict) -> None:
    """champion 三档评估落 accepted/_champion_eval.json(指标级; 分桶明细在 finalize 报告重算)。"""
    champion_eval = {
        "schema_version": 1,
        "produced_by": PRODUCED_BY,
        "champion_fids": list(state.get("champion_fids") or []),
        "baseline_oot_auc": state.get("baseline_oot_auc"),
        "metrics": g6_details["auc"],
        "ks": g6_details["ks"],
        "gain_oot_vs_prev": g6_details["gain_oot"],
    }
    state_io.write_json(paths.champion_eval_json(session_dir), champion_eval)


def check_stop(session_dir, state: dict) -> dict:
    """终止判定, 命中则返回 {stop_reason}, 否则 {}。"""
    evo = (state.get("config_snapshot") or {}).get("evolution") or {}
    round_no = int(state.get("current_round") or 0)
    max_rounds = int(evo.get("max_rounds") or 0)
    if max_rounds and round_no >= max_rounds:
        return {"stop_reason": "max_rounds(轮次上限 %d 已到)" % max_rounds}
    no_accept = int(evo.get("no_accept_rounds") or 0)
    if no_accept and int(state.get("no_accept_streak") or 0) >= no_accept:
        return {"stop_reason": "no_accept_rounds(连续 %d 轮无接受, 探索枯竭)" % no_accept}
    target = evo.get("target_oot_gain")
    base_auc, cham_auc = state.get("baseline_oot_auc"), state.get("champion_oot_auc")
    if target and base_auc is not None and cham_auc is not None and (cham_auc - base_auc) >= float(target):
        return {"stop_reason": "target_oot_gain(累计 OOT 增益 %.6f 已达目标 %.6f)" % (cham_auc - base_auc, float(target))}
    budget = evo.get("time_budget_hours")
    if budget:
        try:
            created = datetime.fromisoformat(str(state.get("created_at")))
            elapsed_h = (_now() - created).total_seconds() / 3600.0
            if elapsed_h >= float(budget):
                return {"stop_reason": "time_budget_hours(已运行 %.1fh >= %.1fh)" % (elapsed_h, float(budget))}
        except ValueError:
            pass
    return {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="轮末 G6 融合模型验证与 champion 提交/回滚")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--round", required=True, type=int)
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    try:
        state = state_io.load_state(session_dir)
    except FileNotFoundError as e:
        return _exit(2, str(e))

    gates_cfg = gates.load_gates(session_dir)
    provisional = load_provisional(session_dir, args.round)
    timeout_s = float(gates_cfg["g1_timeout_s"])

    if not provisional:
        state = state_io.update_state(session_dir, no_accept_streak=int(state.get("no_accept_streak") or 0) + 1)
        state_io.append_ledger(session_dir, {
            "type": "round_no_accept", "round": args.round,
            "note": "本轮无 provisional 接受候选",
        })
        print("round %d: 无 provisional 候选, no_accept_streak=%d" % (args.round, state["no_accept_streak"]))
    else:
        try:
            aug_frames, contract_meta, fid_assign = build_fused_frames(session_dir, provisional, timeout_s)
            state = state_io.load_state(session_dir)  # build_fused_frames 读过 state, 取最新
            g6 = gates.gate_g6(
                lambda: aug_frames, lambda: contract_meta, provisional, state, gates_cfg
            )
        except Exception as e:
            traceback.print_exc()
            return _exit(4, "G6 融合模型构造/训练失败: %s" % e)

        if g6["passed"]:
            # 提交: 候选代码与元数据入 accepted/, champion 状态更新
            paths.accepted_dir(session_dir).mkdir(parents=True, exist_ok=True)
            for a in fid_assign:
                r = a["result"]
                src = paths.candidate_py(session_dir, r["round"], r["cid"])
                shutil.copy2(src, paths.accepted_py(session_dir, a["fid"]))
                state_io.write_json(paths.accepted_meta_json(session_dir, a["fid"]), {
                    "schema_version": 1,
                    "produced_by": PRODUCED_BY,
                    "fid": a["fid"], "cid": r["cid"], "round": r["round"],
                    "hypothesis": r.get("hypothesis"), "category": r.get("category"),
                    "base_features": r.get("base_features"),
                    "single_feature": (r.get("gates", {}).get("G2") or {}),
                    "with_candidate_auc": ((r.get("gates", {}).get("G4_G5") or {}).get("auc") or {}),
                })
            new_fids = list(state.get("champion_fids") or []) + [a["fid"] for a in fid_assign]
            state = state_io.update_state(
                session_dir,
                champion_fids=new_fids,
                next_fid=int(state.get("next_fid") or 1) + len(fid_assign),
                champion_oot_auc=g6["details"]["auc"]["oot"],
                champion_oot_ks=g6["details"]["ks"]["oot"],
                accepted_count=int(state.get("accepted_count") or 0) + len(fid_assign),
                no_accept_streak=0,
            )
            write_champion_eval(session_dir, state, g6["details"])
            state_io.append_ledger(session_dir, {
                "type": "round_commit", "round": args.round,
                "accepted_fids": [a["fid"] for a in fid_assign],
                "champion_oot_auc": g6["details"]["auc"]["oot"],
                "gain_oot": g6["details"]["gain_oot"],
            })
            print("round %d: G6 通过, 接受 %s, champion oot_auc=%.6f (+%.6f)"
                  % (args.round, [a["fid"] for a in fid_assign],
                     g6["details"]["auc"]["oot"], g6["details"]["gain_oot"]))
        else:
            state = state_io.update_state(
                session_dir, no_accept_streak=int(state.get("no_accept_streak") or 0) + 1
            )
            state_io.append_ledger(session_dir, {
                "type": "round_rollback", "round": args.round,
                "cids": [r["cid"] for r in provisional],
                "reason": g6["reason"],
            })
            print("round %d: G6 未过, 整轮回滚(%s)" % (args.round, g6["reason"]))

    # 追加 round-summary 结算段
    summary = paths.round_summary_md(session_dir, args.round)
    if summary.exists():
        streak = state.get("no_accept_streak")
        with open(summary, "a", encoding="utf-8") as f:
            f.write("\n> 结算: champion_fids=%s, no_accept_streak=%s, stop_reason=%s\n"
                    % (state.get("champion_fids"), streak, state.get("stop_reason") or "-"))

    stop = check_stop(session_dir, state)
    if stop:
        state = state_io.update_state(session_dir, **stop)
        print("终止条件命中: %s" % stop["stop_reason"])
    print("commit_round 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
