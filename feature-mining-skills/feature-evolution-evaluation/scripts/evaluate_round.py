# -*- coding: utf-8 -*-
"""evaluate_round.py - 一轮候选逐个过 G1~G5(含多seed降噪, 无 LLM).

要点:
1. 多seed: evolution.n_seeds 配置后, G4/G5 在 K 个不同种子子样本上各训一次,
   增益取均值+bootstrap CI. 救回单seed噪声误杀; 拒单seed侥幸.
2. 场景化G3: saturated 对照 champion 全列; cold_start 对照本轮已接受.
3. 指标可选 PR-AUC(gates.g4_g5_metric, 饱和+正样本率低时默认).
4. 原料指纹黑名单: 候选 base_features 排序 sha1, 防同族反复试.

用法:
    python evaluate_round.py --session-dir <session> --round <N> [--cids c001,c002]

退出码: 0 成功 / 2 状态不可用 / 4 运行时失败
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401

from evo_core import candidate_exec, feature_runtime, gates, param_tuning, paths, state_io

PRODUCED_BY = "feature-mining-skills/feature-evolution-evaluation"
G1_SAMPLE_ROWS = 2000
TRAIN_CAP_SEED = 20260619


def apply_train_cap_multiseed(frames: dict, cap, round_no: int, n_seeds: int) -> tuple:
    """多seed train 采样: 生成 n_seeds 个不同种子的子样本行号.

    每个种子: 全正 + 1/k 负(对齐本仓库负采样口径) 或 随机抽样(n_seeds>1 时).
    候选与 champion 基准用同一组种子子样本(同帧公平).
    Returns: (eval_frames_list, train_idx_seeds) 或 (None, None) 表示未采样.
    """
    if not cap or n_seeds <= 1:
        return None, None
    n_train = len(frames["train"])
    cap = int(cap)
    if n_train <= cap:
        return None, None
    train_idx_seeds = []
    for si in range(n_seeds):
        rng = random.Random(TRAIN_CAP_SEED + int(round_no) * 100 + si)
        train_idx_seeds.append(np_sorted_sample(rng, n_train, cap))  # noqa: F821
    return None, train_idx_seeds


def _exit(code, msg):
    print(msg, file=sys.stderr)
    return code


def candidate_fingerprint(meta: dict) -> str:
    """假设指纹 = sha1(排序 base_features + 归一化 hypothesis). """
    base = ",".join(sorted(str(c) for c in (meta.get("base_features") or [])))
    hyp = " ".join(str(meta.get("hypothesis") or "").split()).lower()
    return hashlib.sha1(("%s|%s" % (base, hyp)).encode("utf-8")).hexdigest()


def material_fingerprint(meta: dict) -> str:
    """原料指纹 = sha1(排序 base_features). 防同族反复试(不管假设怎么改)."""
    base = ",".join(sorted(str(c) for c in (meta.get("base_features") or [])))
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def evaluate_one(session_dir, round_no, cand, frames, input_frames, eval_frames_single, train_idx_seeds,
                 contract_meta, champion_ref, gates_cfg, tune_enabled, scenario, champ_cols_train):
    """单候选过 G1~G5(含多seed). 返回 result dict."""
    cid = cand["cid"]
    metric = gates_cfg.get("g4_g5_metric", "auc")
    result = {"schema_version": 2, "cid": cid, "round": round_no,
              "hypothesis": cand["meta"].get("hypothesis"),
              "category": cand["meta"].get("category"),
              "base_features": cand["meta"].get("base_features") or [],
              "verdict": "candidate_reject", "gate_failed": None, "reason": None,
              "gates": {}, "tuning": {"tuned": False}, "scenario": scenario,
              "metric": metric}
    label_col = contract_meta["label_col"]
    code = cand["code"]

    # G1 静态+试跑
    sample_df = input_frames["train"].head(G1_SAMPLE_ROWS)
    g1 = gates.gate_g1(code, sample_df, gates_cfg)
    result["gates"]["G1"] = {k: v for k, v in g1.items() if k != "series"}
    if not g1["passed"]:
        result["gate_failed"], result["reason"] = "G1", g1["reason"]
        return result

    # 常数精调(可选, train 档)
    tuning = param_tuning.tune_candidate_params(
        code, sample_df, frames["train"][label_col].head(len(sample_df)), cid,
        timeout_s=float(gates_cfg["g1_timeout_s"]), enabled=tune_enabled)
    result["tuning"] = {k: v for k, v in tuning.items() if k != "code"}
    if tuning["tuned"]:
        code = tuning["code"]
        g1 = gates.gate_g1(code, sample_df, gates_cfg)
        result["gates"]["G1"] = {k: v for k, v in g1.items() if k != "series"}
        if not g1["passed"]:
            result["gate_failed"], result["reason"] = "G1", "精调后未过 G1: %s" % g1["reason"]
            return result

    # 全量 transform 三档
    series_by_split = {}
    for split, df in frames.items():
        run = candidate_exec.run_transform(code, input_frames[split], timeout_s=float(gates_cfg["g1_timeout_s"]))
        if not run["ok"]:
            result["gate_failed"], result["reason"] = "G1", "全量执行失败(%s档): %s" % (split, run["error"])
            return result
        series_by_split[split] = run["series"]

    # G2 单特征质量
    g2 = gates.gate_g2(series_by_split["train"], frames["train"][label_col], series_by_split["oot"], gates_cfg)
    result["gates"]["G2"] = g2["details"]
    if not g2["passed"]:
        result["gate_failed"], result["reason"] = "G2", g2["reason"]
        return result

    # G3 场景化
    accepted_train = {fid: frames["train"][fid] for fid in contract_meta["accepted_fids"]}
    g3 = gates.gate_g3(series_by_split["train"], accepted_train, gates_cfg, scenario, champ_cols_train)
    result["gates"]["G3"] = g3["details"]
    if not g3["passed"]:
        result["gate_failed"], result["reason"] = "G3", g3["reason"]
        return result

    # G4/G5 多seed
    g45 = gates.gate_g4_g5_multiseed(
        eval_frames_single, contract_meta, series_by_split, cid, champion_ref, gates_cfg,
        xgb_params=None, train_idx_seeds=train_idx_seeds)
    result["gates"]["G4_G5"] = g45["details"]
    if not g45["passed"]:
        result["gate_failed"], result["reason"] = "G4_G5", g45["reason"]
        return result

    result["verdict"] = "candidate_pass"
    return result


def write_round_summary(session_dir, round_no, results):
    lines = ["# Round %d 评估摘要" % round_no, "",
             "| cid | 类别 | 判定 | 失败关卡 | 原因 | 多seed增益(OOT均值[CI下界]) | 假设 |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        g45 = (r.get("gates", {}).get("G4_G5")) or {}
        gain_str = "-"
        if g45:
            mo = g45.get("mean_gain_oot"); clo = g45.get("ci_lower_oot"); ns = g45.get("n_seeds", 1)
            if mo is not None:
                gain_str = "%.6f [%.6f] (n=%d)" % (mo, clo if clo is not None else mo, ns)
        hyp = (r.get("hypothesis") or "")[:50].replace("|", "/")
        reason = (r.get("reason") or "-")[:70].replace("|", "/")
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            r["cid"], r.get("category") or "-", r["verdict"], r.get("gate_failed") or "-", reason, gain_str, hyp))
    n_pass = sum(1 for r in results if r["verdict"] == "candidate_pass")
    lines += ["", "> provisional 接受 %d/%d(多seed均值+CI 过 G4/G5; 最终入 champion 待 G6)" % (n_pass, len(results)), ""]
    p = paths.round_summary_md(session_dir, round_no)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="一轮候选逐个过 G1~G5(含多seed)")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("--cids", default=None)
    args = parser.parse_args(argv)
    session_dir = Path(args.session_dir)

    try:
        state = state_io.load_state(session_dir)
    except FileNotFoundError as e:
        return _exit(2, str(e))

    gates_cfg = gates.load_gates(session_dir)
    cfg = state.get("config_snapshot") or {}
    evo = cfg.get("evolution") or {}
    scenario = evo.get("scenario") or "cold_start"
    n_seeds = int(gates_cfg.get("n_seeds", 1))

    cids = [c.strip() for c in args.cids.split(",")] if args.cids else None
    cdir = paths.candidates_dir(session_dir, args.round)
    if not cdir.is_dir():
        return _exit(2, "round %d 无候选目录 %s" % (args.round, cdir))
    candidates = []
    for py in sorted(cdir.glob("*.py")):
        cid = py.stem
        if cids and cid not in cids:
            continue
        meta_path = paths.candidate_meta_json(session_dir, args.round, cid)
        meta = state_io.read_json(meta_path) if meta_path.exists() else {}
        candidates.append({"cid": cid, "code": py.read_text(encoding="utf-8"), "meta": meta})
    if not candidates:
        return _exit(2, "round %d 无待评估候选" % args.round)

    try:
        frames, contract_meta = feature_runtime.build_frames(session_dir)
        input_frames = {s: feature_runtime.candidate_input_frame(df) for s, df in frames.items()}
        # 多seed 子样本行号(每轮固定种子, 候选与 champion 同帧)
        train_idx_seeds = None
        if n_seeds > 1:
            cap = evo.get("train_sample_cap")
            if cap:
                _, train_idx_seeds = apply_train_cap_multiseed(frames, cap, args.round, n_seeds)
        # champion 基准: 同一组子样本上重训(保证对比公平)
        eval_frames_single = frames  # 单seed 时全量; 多seed 时 gate_g4_g5_multiseed 内部按子样本切片
        champion_ref = gates.champion_baseline_metrics(
            _subsample_frames(frames, train_idx_seeds[0]) if train_idx_seeds else frames,
            contract_meta, metric=gates_cfg.get("g4_g5_metric", "auc"))
        # champion 全列(train, 供 G3 saturated 对照)
        # base_feature_cols = baseline 特征列(冻结 champion 子集); 缺省回退到 base_cols
        base_feat_cols = contract_meta.get("base_feature_cols")
        if base_feat_cols is None:
            base_feat_cols = contract_meta.get("base_cols", [])
        champ_cols_train = {c: frames["train"][c] for c in base_feat_cols if c in frames["train"].columns}
    except Exception as e:
        traceback.print_exc()
        return _exit(4, "评估帧构造失败: %s" % e)

    blacklist = state_io.recent_fingerprints(session_dir)
    # 原料指纹黑名单(整族勿试)
    material_blacklist = set(state.get("material_blacklist_fingerprints") or [])
    tune_enabled = evo.get("param_tuning") is not False

    results = []
    for cand in candidates:
        cid = cand["cid"]
        if state_io.ledger_has_cid(session_dir, cid):
            print("[skip] %s 已在 ledger" % cid)
            continue
        fp = candidate_fingerprint(cand["meta"])
        mfp = material_fingerprint(cand["meta"])
        if fp in blacklist:
            result = {"schema_version": 2, "cid": cid, "round": args.round,
                      "verdict": "candidate_reject", "gate_failed": "duplicate",
                      "reason": "假设指纹与近几轮重复", "gates": {}}
        elif mfp in material_blacklist:
            result = {"schema_version": 2, "cid": cid, "round": args.round,
                      "verdict": "candidate_reject", "gate_failed": "material_duplicate",
                      "reason": "原料指纹黑名单(同族原料近几轮已试, 换原料)", "gates": {}}
        else:
            print("[eval] %s ..." % cid)
            result = evaluate_one(session_dir, args.round, cand, frames, input_frames,
                                  eval_frames_single, train_idx_seeds, contract_meta, champion_ref,
                                  gates_cfg, tune_enabled, scenario, champ_cols_train)
        result["fingerprint"] = fp
        result["material_fingerprint"] = mfp
        results.append(result)
        state_io.write_json(paths.candidate_result_json(session_dir, args.round, cid), result)
        state_io.append_ledger(session_dir, {
            "type": "candidate", "cid": cid, "round": args.round,
            "verdict": result["verdict"], "gate_failed": result.get("gate_failed"),
            "fingerprint": fp, "material_fingerprint": mfp,
            "mean_gain_oot": ((result.get("gates", {}).get("G4_G5") or {}).get("mean_gain_oot")),
            "ci_lower_oot": ((result.get("gates", {}).get("G4_G5") or {}).get("ci_lower_oot")),
        })
        print("       -> %s %s" % (result["verdict"], result.get("reason") or ""))

    if results:
        write_round_summary(session_dir, args.round, results)
        state_io.update_state(session_dir, current_round=args.round, status="running")
    print("evaluate_round 完成: %d 候选, provisional %d" % (
        len(results), sum(1 for r in results if r["verdict"] == "candidate_pass")))
    return 0


def _subsample_frames(frames, idx):
    """返回 train 取 idx 子集的新帧 dict(不污染原帧)."""
    out = dict(frames)
    out["train"] = frames["train"].iloc[idx].reset_index(drop=True)
    return out


def np_sorted_sample(rng, n, k):
    """固定种子的随机抽样并排序(对齐用). 延迟 import numpy."""
    import numpy as np
    return np.sort(rng.sample(range(n), k) if hasattr(rng, 'sample') else np.array([]))


if __name__ == "__main__":
    sys.exit(main())
