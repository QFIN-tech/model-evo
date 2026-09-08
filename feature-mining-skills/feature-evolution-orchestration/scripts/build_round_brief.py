# -*- coding: utf-8 -*-
"""build_round_brief.py: 进化轮简报 + case batch(候选生成前必读的 LLM 上下文)。

用法:
    python build_round_brief.py --session-dir <session_dir> --round 3

产物(rounds/rXXX/):
  round-brief.md   轮简报: 任务/champion 状态/可用特征(含语义)/上一轮结果/配额与契约
  case-batch.json  均衡采样的样本批次(正负各半, 含 baseline 打分), 供观察数据模式

case batch 思路: LLM 看得到具体样本(特征值 + label + 当前
模型打分), 才能提出有依据的假设; 字符上限超长时按减半策略收缩。

退出码: 0 成功 / 2 session 状态不可用 / 4 运行时失败
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401  (注入 sys.path, 勿删)

from evo_core import paths, state_io

PRODUCED_BY = "feature-mining-skills/feature-evolution-orchestration"
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_CASE_CHARS = 80000
_CASE_VALUE_CHARS = 120  # 单个字符串值截断(文本列截断后进 case, 供观察片段)

# 算子精要(完整目录见 skill references/operator-catalog.md), 每轮注入简报,
# 规则: 先多字段交互算出数值, 再代入非线性复合算子。
_OPERATOR_GUIDE = """## 6. 可用算子参考(完整目录: skill `references/operator-catalog.md`)

构造套路: **先用多字段交互算出一个数值(差分/比率/排名/分箱/语义x数值), 再代入非线性复合算子**; 分段函数须结合复合算子, 不写裸分段。

- 非线性复合: log1p / sqrt / 多项式 / tanh / 分式 a/(b+eps) / 高斯核 exp(-((x-mu)/sigma)^2) / erf / np.where 分段复合
- 交互输入: a-b, a/(a+b+eps), rank(pct=True), qcut 等频分箱, sem_* 语义列 x 数值列, 已接受 fid 列再加工(exploit)
- 文本/类别列可直接加工(输入帧原样可见): str.len() / str.count(关键词) / re 正则 / value_counts 频次编码; 重语义先跑 semantic-feature 注册
- 可选常数精调: 声明 PARAM_BOUNDS/DEFAULT_PARAMS + transform(df, params=None), 系统只在 train 上搜常数并固化回代码(阈值/带宽/幂次类非单调常数推荐声明, 详见算子目录 2c)
- 进化原则: 只增不改(修正已接受特征 -> 新候选叠加); 一个候选一个可证伪假设; 从 case 正负差异出发; 回避上轮失败路径; NaN 显式兜底(覆盖率<0.3 被 G2 拒); 泛化优先, 只认 OOT 稳定增益
"""


def _exit(code: int, msg: str) -> int:
    print(msg, file=sys.stderr)
    return code


def _round_value(v):
    """case 值规整: 数值 4 位小数, 字符串截断, NaN -> null。"""
    if v is None or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    s = str(v)
    return s[:_CASE_VALUE_CHARS]


def build_case_batch(train: pd.DataFrame, label_col: str, drop_cols: list, xgb_score, batch_size: int, round_no: int) -> list:
    """均衡采样(正负各半, 随机种子=轮次保证可复现), 返回 case dict 列表。"""
    rng = random.Random(20260619 + round_no)
    pos_idx = train.index[train[label_col] == 1].tolist()
    neg_idx = train.index[train[label_col] == 0].tolist()
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    half = max(1, batch_size // 2)
    chosen = pos_idx[:half] + neg_idx[:half]
    if not chosen:  # 单类兜底
        chosen = train.index.tolist()[:batch_size]

    keep_cols = [c for c in train.columns if c not in drop_cols and c != label_col]
    cases = []
    for idx in chosen:
        row = {c: _round_value(train.at[idx, c]) for c in keep_cols}
        row[label_col] = int(train.at[idx, label_col]) if pd.notna(train.at[idx, label_col]) else None
        if xgb_score is not None:
            row["xgb_score"] = round(float(xgb_score[idx]), 4)
        cases.append(row)
    return cases


def shrink_cases(cases: list, label_col: str, max_chars: int) -> list:
    """字符上限收缩: 序列化超长则数量减半, 保持正负均衡。"""
    while cases and len(json.dumps(cases, ensure_ascii=False)) > max_chars:
        if len(cases) <= 2:
            # 极端兜底: 只留 1 正 1 负
            pos = [c for c in cases if c.get(label_col) == 1][:1]
            neg = [c for c in cases if c.get(label_col) == 0][:1]
            cases = (pos + neg) or cases[:1]
            break
        half = len(cases) // 2
        pos = [c for c in cases if c.get(label_col) == 1]
        neg = [c for c in cases if c.get(label_col) == 0]
        cases = pos[: max(1, half // 2)] + neg[: max(1, half // 2)]
    return cases


def _semantic_lines(session_dir) -> list:
    reg_path = paths.semantic_registry_json(session_dir)
    if not reg_path.exists():
        return ["(未注册语义特征; 文本列可先跑 semantic-feature skill)"]
    items = state_io.read_json(reg_path).get("items", [])
    if not items:
        return ["(语义注册表为空)"]
    lines = ["| 语义特征 | 来源 | head | OOT AUC |", "|---|---|---|---|"]
    for it in items:
        cols = it.get("output", {}).get("columns", [])
        best = max(
            ((m.get("oot_auc") or 0) for m in it.get("metrics", {}).values()),
            default=0,
        )
        lines.append("| %s | %s | %s | %s |" % (
            ", ".join(cols) or it.get("name"), ",".join(it.get("source_text_cols") or []),
            it.get("head"), ("%.4f" % best) if best else "-"))
    return lines


def _accepted_lines(session_dir, state: dict) -> list:
    fids = list(state.get("champion_fids") or [])
    if not fids:
        return ["(暂无, champion = baseline)"]
    lines = ["| fid | 假设 |", "|---|---|"]
    for fid in fids:
        meta_path = paths.accepted_meta_json(session_dir, fid)
        meta = state_io.read_json(meta_path) if meta_path.exists() else {}
        lines.append("| %s | %s |" % (fid, str(meta.get("hypothesis") or "")[:100]))
    return lines


def build_round_brief_md(session_dir, round_no: int, state: dict, n_cases: int) -> str:
    evo = (state.get("config_snapshot") or {}).get("evolution") or {}
    k = int(evo.get("candidates_per_round") or 4)
    prev = round_no - 1
    lines = [
        "# Round %d 轮简报" % round_no,
        "",
        "> 任务: 围绕当前建模任务提出新的特征假设, 只认 Validation/OOT 稳定增益。",
        "",
        "## 1. 当前状态",
        "",
        "- baseline OOT AUC: %s | champion OOT AUC: %s" % (state.get("baseline_oot_auc"), state.get("champion_oot_auc")),
        "- 已接受特征: %d 个 | 连续无接受轮数: %d" % (state.get("accepted_count") or 0, state.get("no_accept_streak") or 0),
        "- 本轮 case batch: %d 条(见同目录 case-batch.json, 含特征值/label/baseline 打分)" % n_cases,
        "",
        "## 2. 已接受特征(champion, exploit 可在其上加工)",
        "",
    ] + _accepted_lines(session_dir, state) + [
        "",
        "## 3. 可用原料",
        "",
        "- 原始特征画像(全量): `profile/feature-profile.csv` / `profile/data-desc.md`",
        "- 语义特征(semantic-feature 注册):",
        "",
    ] + _semantic_lines(session_dir) + _material_lines(session_dir) + [
        "",
        "## 4. 上一轮结果",
        "",
    ] + (
        ["见 `evolution/rounds/r%03d/round-summary.md`; 详细反馈见 `evolution/feedback/latest-feedback.md`" % prev]
        if prev >= 1 else ["(首轮, 无历史)"]
    ) + [
        "",
        "## 5. 本轮要求",
        "",
        "- 提出 **K=%d 个候选**, 配额: >=1 个 explore(新原料/新算子) + >=1 个 exploit(成功方向加工)" % k,
        "- 候选落盘: `evolution/rounds/r%03d/candidates/cNNN.py` + `cNNN.meta.json`" % round_no,
        "- 代码契约: 模块级 `def transform(df: pd.DataFrame) -> pd.Series`; 输入帧含原始特征列+语义特征列"
        "+已接受特征列(fid 命名), **不含 label/id/dt/既有模型分(如配置 base_model_score_col)**; import 仅限 math/numpy/pandas/statistics/re; 必须确定性",
        "- meta.json 必填: `cid` / `hypothesis`(一句话假设) / `category`(explore|exploit) / `base_features`(用到的列)",
        "- 写完候选后调 `evaluate_round.py` 评估, 再调 `commit_round.py` 结算",
        "",
    ] + _OPERATOR_GUIDE.strip().splitlines() + [
        "",
    ]
    return "\n".join(lines)


def _material_lines(session_dir: Path) -> list:
    """把 Stage0 诊断产物(material whitelist/blacklist/residual)写进简报,
    候选生成时 LLM 必读(原料只从白名单选, 避开黑名单)."""
    lines = []
    pdir = paths.profile_dir(session_dir)
    wl = pdir / "material_whitelist.txt"
    bl = pdir / "material_blacklist.txt"
    res = pdir / "champion_residual.md"
    roi = pdir / "roi_report.md"
    lines.append("")
    lines.append("### 诊断原料(候选必读)")
    if roi.exists():
        lines.append("- 可挖性诊断: `profile/roi_report.md`(池内冗余分布 + ROI 判定)")
    if wl.exists():
        n = sum(1 for _ in open(wl, encoding="utf-8") if _.strip())
        lines.append("- ✅ **白名单(低冗余+有信号, 候选 base_features 只能从这里选)**: `profile/material_whitelist.txt`(%d 个)" % n)
    if bl.exists():
        n = sum(1 for _ in open(bl, encoding="utf-8") if _.strip())
        lines.append("- ❌ **黑名单(高冗余, 勿作原料)**: `profile/material_blacklist.txt`(%d 个)" % n)
    if res.exists():
        lines.append("- 残差反推: `profile/champion_residual.md`(champion 预测错的样本在哪些未用特征异常 = 模型缺的信号, 优先作交互原料)")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成轮简报 round-brief.md + case-batch.json")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--round", required=True, type=int)
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    try:
        state = state_io.load_state(session_dir)
    except FileNotFoundError as e:
        return _exit(2, str(e))

    try:
        cfg = state.get("config_snapshot") or {}
        sample_cfg = cfg.get("sample") or {}
        evo_cfg = cfg.get("evolution") or {}
        label_col = str(sample_cfg.get("label_col") or "label")
        drop_cols = [str(sample_cfg.get("dt_col") or "dt")] + [str(c) for c in (sample_cfg.get("id_cols") or [])]
        score_col = str(sample_cfg.get("base_model_score_col") or "")
        if score_col:
            drop_cols.append(score_col)  # 既有模型分不进 case(xgb_score 单独提供)

        train = pd.read_parquet(paths.split_parquet(session_dir, "train")).reset_index(drop=True)
        xgb_score = None
        pred_path = paths.baseline_predictions_parquet(session_dir, "train")
        if pred_path.exists():
            pred = pd.read_parquet(pred_path)
            if len(pred) == len(train):
                xgb_score = pred["xgb_score"].tolist()

        batch_size = int(evo_cfg.get("batch_size") or DEFAULT_BATCH_SIZE)
        max_chars = int(evo_cfg.get("max_case_chars") or DEFAULT_MAX_CASE_CHARS)
        cases = build_case_batch(train, label_col, drop_cols, xgb_score, batch_size, args.round)
        cases = shrink_cases(cases, label_col, max_chars)

        rdir = paths.round_dir(session_dir, args.round)
        rdir.mkdir(parents=True, exist_ok=True)
        state_io.write_json(paths.case_batch_json(session_dir, args.round), {
            "schema_version": 1,
            "produced_by": PRODUCED_BY,
            "round": args.round,
            "n_cases": len(cases),
            "label_col": label_col,
            "cases": cases,
        })
        paths.round_brief_md(session_dir, args.round).write_text(
            build_round_brief_md(session_dir, args.round, state, len(cases)), encoding="utf-8"
        )
        state_io.write_manifest(
            rdir, PRODUCED_BY, ["round-brief.md", "case-batch.json"],
            overview={"round": args.round, "n_cases": len(cases)},
        )
    except Exception as e:
        traceback.print_exc()
        return _exit(4, "轮简报生成失败: %s" % e)

    print("round-brief 完成: %d 条 case" % len(cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
