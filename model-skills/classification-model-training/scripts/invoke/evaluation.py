# -*- coding: utf-8 -*-
"""evaluation/ 子目录产出: 调 classification-model-evaluation/scripts/eval_single.py
对 train/test/oot 三档 predictions 一次性传入临时目录, 产 4 份三件套:
  - train / test / oot 各 1 份 (version = 文件名 stem)
  - all 1 份 (version = "all", 三档样本纵向拼接 = 全量样本整体一份评估)

依赖约束: 本 skill 不自带评估报告逻辑, 统一委托 classification-model-evaluation。
输入: predictions/{train,test,oot}_predictions.parquet (schema: id_cols + label + score + bucket)
输出: evaluation/{run_name}_{train,test,oot,all}_eval.{json,md,xlsx} × 4 档 + _manifest.json
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from stages.layout import RunLayout, write_manifest

# 定位 classification-model-evaluation/scripts/eval_single.py
# 本文件路径: classification-model-training/scripts/invoke/evaluation.py
# 仓库根 = parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_EVAL_SINGLE = _REPO_ROOT / "classification-model-evaluation" / "scripts" / "eval_single.py"

# predictions parquet → eval_single CSV 的列保留
_KEEP_COLS = ["score", "label"]


def _ensure_eval_single() -> Path:
    """确认 eval_single.py 存在, 否则报错。"""
    if not _EVAL_SINGLE.exists():
        raise FileNotFoundError(
            f"[invoke_evaluation] 找不到 classification-model-evaluation 脚本: {_EVAL_SINGLE}\n"
            "请确认 classification-model-evaluation skill 已部署到本仓库"
        )
    return _EVAL_SINGLE


def _parquet_to_eval_csv(parquet_path: Path, csv_path: Path) -> int:
    """predictions parquet → eval_single 输入 CSV, 仅保留 score + label。

    Args:
        parquet_path: predictions/{split}_predictions.parquet
        csv_path: 临时 CSV 路径

    Returns:
        行数
    """
    df = pd.read_parquet(parquet_path)
    missing = [c for c in _KEEP_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"[invoke_evaluation] {parquet_path.name} 缺列 {missing}, "
            f"predictions parquet schema 应含 id_cols + label + score + bucket"
        )
    df[_KEEP_COLS].to_csv(csv_path, index=False)
    return len(df)


def _run_eval_single(
    eval_single: Path,
    input_dir: Path,
    out_dir: Path,
    name: str,
    model_type: str,
    hyperparams: Optional[Dict[str, Any]],
) -> List[Path]:
    """调一次 eval_single.py (目录模式), 返回产出的全部三件套路径。

    eval_single.py 会为目录下每个 CSV/parquet 各产一份三件套 + 一份 all 合并三件套。
    本函数返回所有产物的路径列表 (具体份数取决于目录内文件数)。
    """
    cmd = [
        sys.executable, str(eval_single),
        "--input-dir", str(input_dir),
        "--score-col", "score",
        "--name", name,
        "--model-type", model_type,
        "-o", str(out_dir),
        "--label-col", "label",
    ]
    if hyperparams:
        cmd.extend(["--hyperparams", json.dumps(hyperparams, ensure_ascii=False)])

    print(f"[invoke_evaluation] eval_single: name={name} input_dir={input_dir.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"[invoke_evaluation] eval_single.py 失败 (code={result.returncode}): "
            f"name={name} input_dir={input_dir}"
        )

    # 收集产物: out_dir 下所有 {name}_*_eval.{json,md,xlsx}
    produced: List[Path] = []
    for ext in ("json", "md", "xlsx"):
        produced.extend(sorted(out_dir.glob(f"{name}_*_eval.{ext}")))
    if not produced:
        raise FileNotFoundError(
            f"[invoke_evaluation] eval_single.py 未产出任何文件: {out_dir}/{name}_*_eval.*\n"
            f"eval_single stdout: {result.stdout}"
        )
    return produced


def invoke_evaluation_stage(
    layout: RunLayout,
    model_cfg: dict,
    used_params: Dict[str, Any],
    produced_by: Optional[str] = None,
) -> Dict[str, List[Path]]:
    """对 train/test/oot 三档 predictions 一次性调 eval_single.py 产 4 份三件套。

    流程:
      1. 把三档 predictions parquet 转 CSV (仅 score + label), 落到临时目录
      2. 一次性传临时目录给 eval_single.py (目录模式)
      3. eval_single.py 产 train/test/oot 各一份 + all 合并一份
      4. 清理临时目录

    Args:
        layout: RunLayout (用 predictions_dir + evaluation_dir + run_name)
        model_cfg: 配置 model 段 (透传 algo 等)
        used_params: 训练实际超参 (传给 --hyperparams)
        produced_by: manifest 来源标识

    Returns:
        {split: [json_path, md_path, xlsx_path]} 四档产物路径
        key 顺序: train / test / oot / all
    """
    eval_single = _ensure_eval_single()

    # 透传字段: 缺失走 eval_single 默认
    model_type = str(model_cfg.get("algo") or "xgb")

    run_name = layout.run_name
    hyperparams = {k: v for k, v in (used_params or {}).items()}

    # 三档 predictions 文件名 (与 write_predictions_stage 一致)
    splits = [
        ("train", layout.predictions_dir / "train_predictions.parquet"),
        ("test", layout.predictions_dir / "test_predictions.parquet"),
        ("oot", layout.predictions_dir / "oot_predictions.parquet"),
    ]

    # 临时目录: 落三档 CSV 供 eval_single.py 目录模式读取
    # 跑完整个阶段后清理; 目录名加 _tmp_ 前缀避免与 evaluation_dir 内其他文件混淆
    tmp_input_dir = layout.evaluation_dir / "_tmp_eval_input"
    if tmp_input_dir.exists():
        shutil.rmtree(tmp_input_dir)
    tmp_input_dir.mkdir(parents=True)

    available_splits: List[str] = []
    try:
        for split, parquet_path in splits:
            if not parquet_path.exists():
                print(f"[invoke_evaluation] 跳过 {split}: predictions 不存在 {parquet_path}")
                continue
            # CSV 文件名直接用 split 名, eval_single.py 按 stem 派生 version
            csv_path = tmp_input_dir / f"{split}.csv"
            n = _parquet_to_eval_csv(parquet_path, csv_path)
            print(f"[invoke_evaluation] {split}: {n} 行 → {csv_path.name}")
            available_splits.append(split)

        if not available_splits:
            raise FileNotFoundError(
                f"[invoke_evaluation] 三档 predictions 均不存在: {[p for _, p in splits]}"
            )

        # 一次性调 eval_single.py, 产 (N+1) 份三件套 (N=available_splits 数, +1=all)
        all_produced = _run_eval_single(
            eval_single=eval_single,
            input_dir=tmp_input_dir,
            out_dir=layout.evaluation_dir,
            name=run_name,
            model_type=model_type,
            hyperparams=hyperparams or None,
        )
    finally:
        if tmp_input_dir.exists():
            shutil.rmtree(tmp_input_dir)

    # 按版本分组产物
    result: Dict[str, List[Path]] = {}
    for ext in ("json", "md", "xlsx"):
        for f in sorted(layout.evaluation_dir.glob(f"{run_name}_*_eval.{ext}")):
            # {run_name}_{version}_eval.{ext} → 取 version 段
            stem = f.stem  # {run_name}_{version}_eval
            prefix = f"{run_name}_"
            if stem.startswith(prefix) and stem.endswith("_eval"):
                version = stem[len(prefix):-len("_eval")]
                result.setdefault(version, []).append(f)

    # 按 train/test/oot/all 顺序排, 其他 version (理论不应出现) 排在后面
    order = {"train": 0, "test": 1, "oot": 2, "all": 3}
    ordered = sorted(result.items(), key=lambda kv: order.get(kv[0], 99))
    result = dict(ordered)

    all_files: List[Path] = []
    for files in result.values():
        all_files.extend(files)

    write_manifest(
        layout.evaluation_dir,
        stage="evaluation",
        files=all_files,
        extra={
            "eval_engine": "classification-model-evaluation/scripts/eval_single.py",
            "splits": list(result.keys()),
            "model_type": model_type,
            "hyperparams": hyperparams,
        },
        produced_by=produced_by,
    )
    return result
