# -*- coding: utf-8 -*-
"""probe_roi.py - Stage 0 可挖性诊断.

开跑前回答"池里还有多少信号可挖", 避免在已榨干的池里空跑 30 轮.

诊断三件套:
  a. 池内冗余扫描: 未用特征逐个算与 champion 全列的 |Spearman| max
     - 高冗余(>0.9): 被 champion 隐式覆盖, 换族不换信息 -> 原料黑名单
     - 低冗余(<0.5): 真·新信息维度 -> 原料白名单
  b. 单变量质量预筛: 低冗余子集算方向修正 AUC, AUC>阈 的进"有信号"白名单
  c. ROI 判定: 低冗余+有信号原料数 vs 每轮候选数 K, 判"该跑/该劝退/中间态"

输出 (落 profile/):
  - roi_report.md             诊断报告(冗余分布直方图/ROI判定/建议)
  - material_whitelist.txt    低冗余+有信号原料(候选生成必读, 一行一个特征名)
  - material_blacklist.txt    高冗余原料(勿用, 一行一个)

残差反推(champion_residual.md)由同目录 champion_residual.py 单独产出(可选, 需 baseline 预测).

成本: 全量帧上算 Spearman 是 O(n_unused * n_champion * n_rows). 大数据量时
      在 train 子集(默认 50000 行)上算, 够估冗余趋势.

退出码: 0 成功 / 2 session 不可用 / 4 运行时失败
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401  注入 sys.path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evo_core import paths, state_io
from evo_prep import sample_contract

# 诊断子样本行数(算 Spearman 的帧大小; 大数据量降本)
PROBE_SAMPLE_ROWS = 50000
# 冗余阈值
REDUNDANCY_HIGH = 0.9   # >此值进黑名单(被 champion 隐式覆盖)
REDUNDANCY_LOW = 0.5    # <此值进白名单(真新信息)
# 单变量 AUC 阈值(对齐 G2 g2_min_auc)
SIGNAL_AUC_MIN = 0.5


def _exit(code: int, msg: str) -> int:
    print(msg, file=sys.stderr)
    return code


def _spearman_max_vs_champ(col: np.ndarray, champ_frame: pd.DataFrame, sample_idx: np.ndarray) -> float:
    """单个未用特征 vs champion 全列的 |Spearman| 最大值(抽样加速).

    col: 未用特征值(全量, 已数值化)
    champ_frame: champion 列DataFrame(抽样后)
    sample_idx: 抽样行号(对 col 和 champ 同步切片)
    返回 nan 表示无法计算(常数列等).
    """
    c = col[sample_idx]
    c = np.nan_to_num(c, nan=0.0)
    if np.std(c) < 1e-12:
        return float("nan")
    mx = 0.0
    for colname in champ_frame.columns:
        o = champ_frame[colname].to_numpy()
        if np.std(o) < 1e-12:
            continue
        r = spearmanr(c, o).correlation
        if not np.isnan(r):
            mx = max(mx, abs(r))
    return mx


def _rankdata_average(x):
    """scipy.stats.rankdata(method='average') 的 numpy 版(1D 向量), 处理 ties 取平均秩.

    等价 rankdata(x, method='average'). 用于把 Spearman 退化为 Pearson of ranks,
    从而可用矩阵乘一次性算"未用特征 vs 全部 champion 列"的相关, 避免逐列 Python 循环.
    """
    arr = np.asarray(x, dtype=float)
    n = arr.size
    if n == 0:
        return arr
    sorter = np.argsort(arr, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    arr_sorted = arr[sorter]
    obs = np.r_[True, arr_sorted[1:] != arr_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], n]
    ranks = 0.5 * (count[dense] + count[dense - 1] + 1).astype(float)
    return ranks


def _precompute_champ_ranks(champ_frame: pd.DataFrame) -> np.ndarray:
    """预排名 champion 全列 -> (n, n_champ) 排名矩阵.

    原 _spearman_max_vs_champ 每个未用特征都对 champion 全列逐列 spearmanr(Python 循环),
    未用特征数 × champion 列数大时极慢(如 2000×400 在 2 万行上约百分钟).
    预排名后每个未用特征只算一次 Pearson(向量化矩阵乘), 整体降到分钟级.
    """
    champ = champ_frame.to_numpy().astype(float)
    if champ.shape[1] == 0:
        return champ
    ranked = np.empty_like(champ)
    for j in range(champ.shape[1]):
        ranked[:, j] = _rankdata_average(champ[:, j])
    return ranked


def _spearman_max_vs_champ_vec(c_rank: np.ndarray, champ_ranked: np.ndarray) -> float:
    """向量化版: 单个未用特征(已排名+中心化+标准化) vs 全部 champion
    排名列(已排名)的 |Spearman| 最大值.

    Spearman(x, y) = Pearson(rank(x), rank(y)); 向量化算 c_rank 与 champ_ranked 各列的
    Pearson 相关, 取 |max|. c_rank: (n,) 已中心化标准化; champ_ranked: (n, n_champ)
    未中心化(本函数内中心化). 返回 0.0 表示全常数或空.
    """
    if c_rank is None or champ_ranked.size == 0:
        return 0.0
    if np.std(c_rank) < 1e-12:
        return float("nan")
    cr = c_rank - c_rank.mean()
    cr = cr / (np.std(cr) + 1e-12)
    cr2 = champ_ranked - champ_ranked.mean(axis=0)
    std2 = champ_ranked.std(axis=0)
    std2 = np.where(std2 < 1e-12, 1.0, std2)
    cr2 = cr2 / std2
    corrs = cr2.T.dot(cr) / len(cr)
    return float(np.nanmax(np.abs(corrs))) if corrs.size else 0.0


def _direction_fixed_auc(y: np.ndarray, s: np.ndarray) -> float:
    """方向修正 AUC = max(a, 1-a); 单类/全NaN 返回 nan (对齐 metrics.compute_auc_direction_fixed)."""
    m = np.isfinite(s) & np.isfinite(y)
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2 or len(np.unique(s)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    a = float(roc_auc_score(y, s))
    return max(a, 1.0 - a)


def probe_redundancy(session_dir: Path, contract: dict, champ_cols: list,
                     all_feature_cols: list, sample_rows: int = PROBE_SAMPLE_ROWS) -> dict:
    """池内冗余扫描: 未用特征 vs champion 全列 |Spearman| max 分布.

    Returns: {unused_features, redundancy: {feat: corr}, n_high, n_mid, n_low, sample_n}
    """
    splits = {s: pd.read_parquet(paths.split_parquet(session_dir, s)) for s in paths.SPLITS}
    train = splits["train"]
    y = train[contract["label_col"]].to_numpy().astype(float)

    # 抽样行号(固定种子, 可复现)
    n = len(train)
    rng = np.random.RandomState(42)
    if n > sample_rows:
        sample_idx = np.sort(rng.choice(n, size=sample_rows, replace=False))
    else:
        sample_idx = np.arange(n)

    # champion 帧(抽样, 数值化)
    champ_frame = train[champ_cols].iloc[sample_idx].apply(pd.to_numeric, errors="coerce").astype(float)
    champ_frame = champ_frame.fillna(0.0)

    unused = [c for c in all_feature_cols if c not in set(champ_cols)]
    print("[probe] 未用特征 %d 个, champion %d 列, 诊断样本 %d 行" % (len(unused), len(champ_cols), len(sample_idx)))

    # 预排名 champion 全列(避免逐列 spearmanr Python 循环)
    champ_ranked = _precompute_champ_ranks(champ_frame)
    print("[probe] champion 排名矩阵预计算完成, shape=%s" % (champ_ranked.shape,))

    redundancy = {}
    for i, feat in enumerate(unused):
        try:
            col = pd.to_numeric(train[feat], errors="coerce").to_numpy().astype(float)
        except Exception:
            redundancy[feat] = float("nan")
            continue
        # 向量化版 Spearman(逐列循环在大宽表上慢约百倍)
        c = col[sample_idx]
        c = np.nan_to_num(c, nan=0.0)
        if np.std(c) < 1e-12:
            redundancy[feat] = float("nan")
        else:
            c_rank = _rankdata_average(c)
            redundancy[feat] = _spearman_max_vs_champ_vec(c_rank, champ_ranked)
        if (i + 1) % 200 == 0:
            print("  冗余扫描 %d/%d" % (i + 1, len(unused)), flush=True)

    # 分布统计
    vals = [v for v in redundancy.values() if not np.isnan(v)]
    n_high = sum(1 for v in vals if v > REDUNDANCY_HIGH)
    n_mid = sum(1 for v in vals if REDUNDANCY_LOW <= v <= REDUNDANCY_HIGH)
    n_low = sum(1 for v in vals if v < REDUNDANCY_LOW)
    return {
        "unused_features": unused,
        "redundancy": redundancy,
        "n_high_redundant": n_high, "n_mid": n_mid, "n_low_redundant": n_low,
        "sample_n": len(sample_idx),
    }


def probe_signal_auc(session_dir: Path, contract: dict, low_redundant_feats: list,
                      sample_rows: int = PROBE_SAMPLE_ROWS) -> dict:
    """低冗余子集的单变量方向修正 AUC(对齐 G2).

    Returns: {auc: {feat: auc}, n_with_signal}
    """
    train = pd.read_parquet(paths.split_parquet(session_dir, "train"))
    y = train[contract["label_col"]].to_numpy().astype(float)
    n = len(train)
    rng = np.random.RandomState(42)
    idx = np.sort(rng.choice(n, size=min(n, sample_rows), replace=False)) if n > sample_rows else np.arange(n)
    y_s = y[idx]

    auc_map = {}
    for feat in low_redundant_feats:
        try:
            s = pd.to_numeric(train[feat], errors="coerce").to_numpy().astype(float)[idx]
        except Exception:
            auc_map[feat] = float("nan")
            continue
        auc_map[feat] = _direction_fixed_auc(y_s, s)

    vals = [v for v in auc_map.values() if not np.isnan(v)]
    n_sig = sum(1 for v in vals if v >= SIGNAL_AUC_MIN)
    return {"auc": auc_map, "n_with_signal": n_sig}


def write_reports(session_dir: Path, redundancy_res: dict, signal_res: dict,
                  champ_cols: list, n_candidates_per_round: int, scenario: str) -> dict:
    """落 roi_report.md / material_whitelist.txt / material_blacklist.txt. 返回 ROI 判定."""
    red = redundancy_res["redundancy"]
    auc = signal_res["auc"]

    # 白名单: 低冗余(<REDUNDANCY_LOW) 且 有信号(AUC>=SIGNAL_AUC_MIN)
    whitelist = []
    for feat, r in red.items():
        if np.isnan(r) or r >= REDUNDANCY_LOW:
            continue
        a = auc.get(feat, float("nan"))
        if not np.isnan(a) and a >= SIGNAL_AUC_MIN:
            whitelist.append((feat, float(r), float(a)))
    whitelist.sort(key=lambda x: -x[2])  # 按 AUC 降序

    # 黑名单: 高冗余(>REDUNDANCY_HIGH)
    blacklist = [(feat, float(r)) for feat, r in red.items() if not np.isnan(r) and r > REDUNDANCY_HIGH]
    blacklist.sort(key=lambda x: -x[1])

    # ROI 判定
    K = n_candidates_per_round
    n_wl = len(whitelist)
    if scenario == "saturated":
        # 饱和模型: 要求更宽裕, <K*2 即判枯竭
        roi_status = "exhausted" if n_wl < K * 2 else ("marginal" if n_wl < K * 5 else "promising")
    else:
        roi_status = "exhausted" if n_wl < K else ("marginal" if n_wl < K * 3 else "promising")

    # roi_report.md
    lines = [
        "# Stage 0 可挖性诊断报告\n",
        "> 开跑前判定池内是否还有信号可挖, 避免在榨干的池里空跑.\n",
        "## 冗余分布",
        "- 未用特征总数: %d" % len(redundancy_res["unused_features"]),
        "- 高冗余(\\|corr\\|>%.1f, 进黑名单): %d" % (REDUNDANCY_HIGH, redundancy_res["n_high_redundant"]),
        "- 中冗余(%.1f~%.1f): %d" % (REDUNDANCY_LOW, REDUNDANCY_HIGH, redundancy_res["n_mid"]),
        "- 低冗余(<%.1f, 候选原料池): %d" % (REDUNDANCY_LOW, redundancy_res["n_low_redundant"]),
        "- 诊断样本行数: %d" % redundancy_res["sample_n"],
        "",
        "## 单变量信号(低冗余子集)",
        "- 有信号(AUC>=%.2f): %d" % (SIGNAL_AUC_MIN, signal_res["n_with_signal"]),
        "",
        "## ROI 判定",
        "- 场景: %s" % scenario,
        "- 每轮候选数 K = %d" % K,
        "- 低冗余+有信号原料(白名单)数: %d" % n_wl,
        "- **判定: %s**" % roi_status,
        "",
    ]
    if roi_status == "exhausted":
        lines += [
            "> ⚠️ 池内已榨干. 低冗余+有信号原料不足支撑进化. 建议不要空跑轮次, 换范式:",
            "> 1. 引入池外新原料(外部数据/图特征)",
            "> 2. 跑 semantic-feature skill 把文本列转语义特征",
            "> 3. 切 classification-model-tuning 调参",
            "> 4. 接受当前 champion 为生产模型",
        ]
    elif roi_status == "marginal":
        lines += ["> ⚠️ ROI 边际. 可跑但命中率会低, 建议: 提高轮次预算(20-30轮)+ 严控原料只从白名单选."]
    else:
        lines += ["> ✅ ROI 可观. 低冗余+有信号原料充足, 正常进入进化循环."]

    # 白名单 Top
    lines += ["", "## 白名单 Top30(低冗余+有信号, 候选原料优先从这里选)", "| feature | \\|corr\\| | AUC |", "|---|---|---|"]
    for feat, r, a in whitelist[:30]:
        lines.append("| %s | %.4f | %.4f |" % (feat, r, a))

    lines += ["", "## 黑名单 Top30(高冗余, 勿作原料)", "| feature | \\|corr\\| |", "|---|---|"]
    for feat, r in blacklist[:30]:
        lines.append("| %s | %.4f |" % (feat, r))

    pdir = paths.profile_dir(session_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "roi_report.md").write_text("\n".join(lines), encoding="utf-8")
    (pdir / "material_whitelist.txt").write_text(
        "\n".join(f for f, _, _ in whitelist), encoding="utf-8")
    (pdir / "material_blacklist.txt").write_text(
        "\n".join(f for f, _ in blacklist), encoding="utf-8")

    return {"roi_status": roi_status, "n_whitelist": n_wl, "n_blacklist": len(blacklist)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stage 0 可挖性诊断")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--sample-rows", type=int, default=PROBE_SAMPLE_ROWS)
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    try:
        state = state_io.load_state(session_dir)
    except FileNotFoundError as e:
        return _exit(2, str(e))

    try:
        contract = sample_contract.parse_sample_contract(state.get("config_snapshot") or {})
        cfg = state.get("config_snapshot") or {}
        all_feature_cols = sample_contract.validate_sample_df(
            pd.read_parquet(paths.sample_parquet(session_dir)), contract)
        # champion 列 = baseline 特征 + 已接受 fid
        champ_cols = list(state_io.read_json(paths.baseline_model_dir(session_dir) / "model_meta.json")["feature_cols"])
        champ_cols = champ_cols + list(state.get("champion_fids") or [])
        scenario = (cfg.get("evolution") or {}).get("scenario") or "cold_start"
        K = (cfg.get("evolution") or {}).get("candidates_per_round") or 4

        print("[probe] Stage 0 可挖性诊断开始 (scenario=%s, K=%d)" % (scenario, K))
        red_res = probe_redundancy(session_dir, contract, champ_cols, all_feature_cols, args.sample_rows)
        low_red = [f for f, r in red_res["redundancy"].items() if not np.isnan(r) and r < REDUNDANCY_LOW]
        sig_res = probe_signal_auc(session_dir, contract, low_red, args.sample_rows)
        roi = write_reports(session_dir, red_res, sig_res, champ_cols, K, scenario)

        # 写入 state
        state["stage0_roi"] = {"status": roi["roi_status"], "n_whitelist": roi["n_whitelist"],
                               "n_blacklist": roi["n_blacklist"], "scenario": scenario}
        state_io.save_state(session_dir, state)
        print("[probe] ROI 判定: %s (白名单 %d, 黑名单 %d)" % (roi["roi_status"], roi["n_whitelist"], roi["n_blacklist"]))
        if roi["roi_status"] == "exhausted":
            print("[probe] ⚠️ 池内已榨干, 建议不进进化循环, 见 profile/roi_report.md")
            return 0  # 仍返回0(诊断成功), 是否进循环由 orchestration 按 roi_status 决定
        return 0
    except Exception as e:
        traceback.print_exc()
        return _exit(4, "Stage 0 诊断失败: %s" % e)


if __name__ == "__main__":
    sys.exit(main())
