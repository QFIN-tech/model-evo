# -*- coding: utf-8 -*-
"""recommend 评估委托: 调 classification-model-evaluation/scripts/eval_single.py
对 train/test/oot 三档 predictions parquet 一次性传入临时目录, 产 4 份三件套:
  - train / test / oot 各 1 份 (version = 文件名 stem)
  - all 1 份 (version = "all", 三档样本纵向拼接 = 全量样本整体一份评估)

依赖约束: 评估逻辑委托 classification-model-evaluation。
输入: predictions/{train,test,oot}.parquet (schema: id_cols + label + score)
输出: {out_dir}/{model_id}_{train,test,oot,all}_eval.{json,md,xlsx} × 4 档 + _manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# 定位 classification-model-evaluation/scripts/eval_single.py
# 本文件路径: classification-model-recommend/scripts/invoke_evaluation.py
# 仓库根 = parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_SINGLE = _REPO_ROOT / "classification-model-evaluation" / "scripts" / "eval_single.py"

# parquet → eval_single CSV 的列保留
_KEEP_COLS = ["score", "label"]


def _ensure_eval_single() -> Path:
    """确认 eval_single.py 存在, 否则报错。"""
    if not _EVAL_SINGLE.exists():
        raise FileNotFoundError(
            f"[invoke_evaluation] 找不到 classification-model-evaluation 脚本: {_EVAL_SINGLE}\n"
            "请确认 classification-model-evaluation skill 已部署到本仓库"
        )
    return _EVAL_SINGLE


def _parquet_to_eval_csv(parquet_path: Path, csv_path: Path, score_col: str, label_col: str) -> int:
    """predictions parquet → eval_single 输入 CSV, 仅保留 score + label(列名统一为 score/label)。

    Args:
        parquet_path: predictions/{split}.parquet
        csv_path: 临时 CSV 路径
        score_col: 原打分列名
        label_col: 原标签列名

    Returns:
        行数
    """
    df = pd.read_parquet(parquet_path)
    if score_col not in df.columns:
        raise KeyError(f"[invoke_evaluation] {parquet_path.name} 缺打分列 {score_col}")
    if label_col not in df.columns:
        raise KeyError(f"[invoke_evaluation] {parquet_path.name} 缺标签列 {label_col}")
    out = pd.DataFrame({
        "score": df[score_col].astype(float),
        "label": df[label_col].astype(int),
    })
    out = out[out["score"].notna() & out["label"].notna()]
    out.to_csv(csv_path, index=False)
    return len(out)


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


def _write_manifest(
    out_dir: Path,
    model_id: str,
    files: List[Path],
    splits: List[str],
    model_type: str,
) -> None:
    """写 evaluation/_manifest.json。"""
    manifest = {
        "stage": "evaluation",
        "eval_engine": "classification-model-evaluation/scripts/eval_single.py",
        "model_id": model_id,
        "splits": splits,
        "model_type": model_type,
        "files": [str(f.relative_to(out_dir)) for f in files],
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    p = argparse.ArgumentParser(description="recommend 评估委托 (调 eval_single.py 目录模式)")
    p.add_argument("--train-parquet", required=True, help="Train parquet 路径")
    p.add_argument("--test-parquet", required=True, help="Test parquet 路径")
    p.add_argument("--oot-parquet", required=True, help="OOT parquet 路径")
    p.add_argument("--score-col", default="score", help="模型分列名, 默认 score")
    p.add_argument("--label-col", default="label", help="标签字段名, 默认 label")
    p.add_argument("--out-dir", required=True, help="报告输出目录")
    p.add_argument("--model-id", default="model", help="模型ID, 用于输出文件名前缀")
    p.add_argument("--model-type", default="xgboost", help="模型类型(透传 eval_single.py)")
    return p


def main() -> None:
    """入口: 三档 parquet → 临时目录 CSV → 调 eval_single.py 一次 → 写 manifest。"""
    args = _build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    eval_single = _ensure_eval_single()
    out_dir = Path(args.out_dir)

    splits = [
        ("train", Path(args.train_parquet)),
        ("test", Path(args.test_parquet)),
        ("oot", Path(args.oot_parquet)),
    ]

    # 临时目录: 落三档 CSV 供 eval_single.py 目录模式读取
    tmp_input_dir = out_dir / "_tmp_eval_input"
    if tmp_input_dir.exists():
        shutil.rmtree(tmp_input_dir)
    tmp_input_dir.mkdir(parents=True)

    available_splits: List[str] = []
    all_files: List[Path] = []
    try:
        for split, parquet_path in splits:
            if not parquet_path.exists():
                print(f"[invoke_evaluation] 跳过 {split}: parquet 不存在 {parquet_path}")
                continue
            # CSV 文件名直接用 split 名, eval_single.py 按 stem 派生 version
            csv_path = tmp_input_dir / f"{split}.csv"
            n = _parquet_to_eval_csv(parquet_path, csv_path, args.score_col, args.label_col)
            print(f"[invoke_evaluation] {split}: {n} 行 → {csv_path.name}")
            available_splits.append(split)

        if not available_splits:
            raise FileNotFoundError(
                f"[invoke_evaluation] 三档 parquet 均不存在: {[p for _, p in splits]}"
            )

        # 一次性调 eval_single.py, 产 (N+1) 份三件套 (N=available_splits 数, +1=all)
        all_files = _run_eval_single(
            eval_single=eval_single,
            input_dir=tmp_input_dir,
            out_dir=out_dir,
            name=args.model_id,
            model_type=args.model_type,
            hyperparams=None,
        )
    finally:
        if tmp_input_dir.exists():
            shutil.rmtree(tmp_input_dir)

    # 收集实际产出的 version 列表 (train/test/oot/all)
    produced_versions: List[str] = []
    for f in all_files:
        if f.suffix == ".json":
            stem = f.stem  # {model_id}_{version}_eval
            prefix = f"{args.model_id}_"
            if stem.startswith(prefix) and stem.endswith("_eval"):
                version = stem[len(prefix):-len("_eval")]
                if version not in produced_versions:
                    produced_versions.append(version)
    # 按 train/test/oot/all 排, 其他 version 排在后面
    order = {"train": 0, "test": 1, "oot": 2, "all": 3}
    produced_versions.sort(key=lambda v: order.get(v, 99))

    _write_manifest(
        out_dir=out_dir,
        model_id=args.model_id,
        files=all_files,
        splits=produced_versions,
        model_type=args.model_type,
    )

    print(f"\n[invoke_evaluation] 完成: {len(produced_versions)} 档 → {out_dir}")
    for v in produced_versions:
        print(f"  - {v}")
    for f in all_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
