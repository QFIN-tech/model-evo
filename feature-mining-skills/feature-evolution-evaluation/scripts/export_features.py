# -*- coding: utf-8 -*-
"""export_features.py: 导出 accepted-features 交付包(给下游训练/生产取用)。

用法:
    python export_features.py --session-dir <session_dir>

产物(evolution/export/accepted-features/):
  {fid}.py / {fid}.meta.json   已接受特征代码与元数据(按 fid 顺序应用, 后者可见前者)
  features.parquet             id_cols + split + label + 全部 fid 列(champion 特征值)
  model/model.json(+meta)      融合模型(champion)与特征列清单
  README.md                    应用方式说明(输入列契约 / 应用顺序 / 注意事项)
  _manifest.json

退出码: 0 成功 / 2 无已接受特征 / 4 运行时失败
"""
from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401  (注入 sys.path, 勿删)

from evo_core import feature_runtime, gates, paths, state_io

PRODUCED_BY = "feature-mining-skills/feature-evolution-evaluation"


def _exit(code: int, msg: str) -> int:
    print(msg, file=sys.stderr)
    return code


def build_export_frames(session_dir) -> tuple:
    """构造导出帧(含 champion 模型三档打分)。返回 (frames, contract_meta, eval_result)。"""
    frames, contract_meta = feature_runtime.build_frames(session_dir)
    eval_result = gates.champion_baseline_metrics(frames, contract_meta)  # champion 重训(与 G6 同口径)
    return frames, contract_meta, eval_result


def build_readme(state: dict, contract_meta: dict, fids: list) -> str:
    base = contract_meta["base_cols"]
    return "\n".join([
        "# Accepted Features 导出包",
        "",
        "> session: %s  |  baseline OOT AUC=%s -> champion OOT AUC=%s" % (
            state.get("mining_name"), state.get("baseline_oot_auc"), state.get("champion_oot_auc")),
        "",
        "## 应用方式",
        "",
        "1. 准备输入帧: 含基础特征列(见下)与本包 `{fid}.py` 依赖的输入列; 不需要 label/id/dt。",
        "2. 按 fid 升序逐个执行 `transform(df)`(后一个特征可以看到前一个特征列), 列名为 fid。",
        "3. `model/model.json` 为融合模型(XGB, 固定超参), 特征列清单见 `model/model_meta.json`。",
        "",
        "## 基础特征列(%d 个)" % len(base),
        "",
        "`%s`" % ", ".join(base),
        "",
        "## 已接受特征(%d 个)" % len(fids),
        "",
        "| fid | 假设 | 提出轮次 |",
        "|---|---|---|",
    ] + [
        "| %s | %s | %s |" % (fid, meta.get("hypothesis", ""), meta.get("round", "-"))
        for fid, meta in fids
    ] + [
        "",
        "## 注意事项",
        "",
        "- 特征代码按 G1 白名单约束(仅 math/numpy/pandas/statistics/re), 应用环境需满足同样依赖。",
        "- 超出 train 分布的取值会自然落 NaN/极端值, 上线前建议按本包 features.parquet 的分布做截断。",
        "",
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="导出 accepted-features 交付包")
    parser.add_argument("--session-dir", required=True)
    args = parser.parse_args(argv)

    session_dir = Path(args.session_dir)
    try:
        state = state_io.load_state(session_dir)
    except FileNotFoundError as e:
        return _exit(2, str(e))

    fids = list(state.get("champion_fids") or [])
    if not fids:
        return _exit(2, "无已接受特征(champion=baseline), 无可导出内容")

    try:
        frames, contract_meta, eval_result = build_export_frames(session_dir)
    except Exception as e:
        traceback.print_exc()
        return _exit(4, "导出帧构造失败: %s" % e)

    out_dir = paths.export_dir(session_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "model"
    model_dir.mkdir(exist_ok=True)

    # 1. 特征代码与元数据
    fid_metas = []
    for fid in fids:
        shutil.copy2(paths.accepted_py(session_dir, fid), out_dir / ("%s.py" % fid))
        meta = state_io.read_json(paths.accepted_meta_json(session_dir, fid))
        fid_metas.append((fid, meta))
        state_io.write_json(out_dir / ("%s.meta.json" % fid), meta)

    # 2. 特征值 parquet(id + split + label + fids + champion score)
    label_col = contract_meta["label_col"]
    id_cols = contract_meta["id_cols"]
    parts = []
    for split, df in frames.items():
        part = df[id_cols + [label_col] + fids].copy()
        part.insert(len(id_cols), "split", split)
        part["champion_score"] = np.asarray(eval_result["scores"][split], dtype=float)
        parts.append(part)
    pd.concat(parts, ignore_index=True).to_parquet(out_dir / "features.parquet", index=False)

    # 3. 融合模型
    eval_result["model"].save_model(str(model_dir / "model.json"))
    state_io.write_json(model_dir / "model_meta.json", {
        "schema_version": 1,
        "produced_by": PRODUCED_BY,
        "algo": "xgb",
        "feature_cols": feature_runtime.champion_feature_cols(contract_meta),
        "champion_fids": fids,
        "metrics": {s: {"auc": eval_result["auc"][s], "ks": eval_result["ks"][s]} for s in paths.SPLITS},
    })

    # 4. README + manifest
    (out_dir / "README.md").write_text(build_readme(state, contract_meta, fid_metas), encoding="utf-8")
    state_io.write_manifest(
        out_dir, PRODUCED_BY,
        ["%s.py" % f for f in fids] + ["features.parquet", "model/model.json", "README.md"],
        overview={
            "n_accepted": len(fids),
            "baseline_oot_auc": state.get("baseline_oot_auc"),
            "champion_oot_auc": state.get("champion_oot_auc"),
        },
    )
    print("导出完成: %s(%d 个特征, champion oot_auc=%s)"
          % (out_dir, len(fids), state.get("champion_oot_auc")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
