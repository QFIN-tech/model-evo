# -*- coding: utf-8 -*-
"""comparison/ 子目录产出: 调 classification-model-comparison/scripts/compare_models.py
对 train/test/oot 三档分别做 N-way 对比 (新模型 eval JSON vs 一个或多个基线 eval JSON)。

依赖约束: 本 skill 不自带对比逻辑, 统一委托 classification-model-comparison。
输入:
  - 新模型 eval JSON: {layout.evaluation_dir}/{run_name}_{split}_eval.json
  - 基线 eval JSON: 传入的 baseline_eval_dirs 下每个目录 glob *_{split}_eval.json,
    多目录命中时全部并入 N-way 对比, 同名文件按 model_id 去重保留首个
输出: {layout.comparison_dir}/comparison_{split}.{json,md,xlsx} × 3 档 + _manifest.json
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from stages.layout import RunLayout, write_manifest

# 定位 classification-model-comparison/scripts/compare_models.py
# 本文件路径: classification-model-training/scripts/invoke/comparison.py
# 仓库根 = parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPARE = _REPO_ROOT / "classification-model-comparison" / "scripts" / "compare_models.py"

_SPLITS = ("train", "test", "oot")


def _ensure_compare() -> Path:
    """确认 compare_models.py 存在, 否则报错。"""
    if not _COMPARE.exists():
        raise FileNotFoundError(
            f"[invoke_comparison] 找不到 classification-model-comparison 脚本: {_COMPARE}\n"
            "请确认 classification-model-comparison skill 已部署到本仓库"
        )
    return _COMPARE


def _normalize_dirs(
    baseline_eval_dir: Union[str, Sequence[str], None],
) -> List[Path]:
    """把入参规范化为 Path 列表。

    支持三种入参:
      - None → 空列表
      - str  → 单目录(若含通配符, 走 glob 展开为多目录)
      - 序列(str/Path) → 逐项加入, str 含通配符时 glob 展开
    """
    if baseline_eval_dir is None:
        return []
    if isinstance(baseline_eval_dir, (str, Path)):
        items: Iterable = [baseline_eval_dir]
    else:
        items = baseline_eval_dir
    dirs: List[Path] = []
    for item in items:
        s = str(item)
        if any(ch in s for ch in "*?["):
            for hit in sorted(Path().glob(s) if not Path(s).is_absolute() else Path("/").glob(s.lstrip("/"))):
                if hit.is_dir():
                    dirs.append(hit.resolve())
        else:
            p = Path(s).resolve()
            if p.is_dir():
                dirs.append(p)
    # 去重保序
    seen = set(); out: List[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d); out.append(d)
    return out


def _find_baseline_jsons(
    baseline_dirs: Sequence[Path], split: str,
) -> List[Path]:
    """在多个 baseline_eval_dir 下 glob *_{split}_eval.json, 按 model_id 去重。

    去重规则: 文件名形如 `<model_id>_<split>_eval.json`, 同 model_id 跨目录命中
    只保留首个(避免 model-recommend/yx_001/evaluation 与其软链/复制目录重复计入)。
    """
    hits: List[Path] = []
    seen_ids: set = set()
    for d in baseline_dirs:
        if not d.exists():
            continue
        for m in sorted(d.glob(f"*_{split}_eval.json")):
            model_id = m.name.rsplit(f"_{split}_eval.json", 1)[0]
            if model_id in seen_ids:
                continue
            seen_ids.add(model_id)
            hits.append(m)
    return hits


def _run_compare(
    compare_script: Path,
    new_json: Path,
    baseline_jsons: List[Path],
    out_dir: Path,
    split: str,
) -> List[Path]:
    """调一次 compare_models.py, 返回产出的三件套路径。

    Args:
        compare_script: compare_models.py 绝对路径
        new_json: 新模型 eval JSON
        baseline_jsons: 基线 eval JSON 列表(0 个直接报错, 上层应在调用前判定)
        out_dir: comparison/ 目录
        split: train/test/oot (用于输出文件名)

    Returns:
        [json_path, md_path, xlsx_path]
    """
    out_prefix = out_dir / f"comparison_{split}"
    cmd = [
        sys.executable, str(compare_script),
        "--jsons", str(new_json), *[str(b) for b in baseline_jsons],
        "-o", str(out_prefix),
        "--fmt", "all",
    ]
    new_name = new_json.name
    base_names = ", ".join(b.name for b in baseline_jsons)
    print(f"[invoke_comparison] {split}: {new_name} vs [{base_names}]")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"[invoke_comparison] compare_models.py 失败 (code={result.returncode}): "
            f"split={split}"
        )

    files = [out_dir / f"comparison_{split}.{ext}" for ext in ("json", "md", "xlsx")]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(
                f"[invoke_comparison] 预期产物未生成: {f}\n"
                f"compare_models stdout: {result.stdout}"
            )
    return files


def invoke_comparison_stage(
    layout: RunLayout,
    baseline_eval_dir: Union[str, Sequence[str], None] = None,
    produced_by: Optional[str] = None,
) -> Dict[str, List[Path]]:
    """对 train/test/oot 三档分别调 compare_models.py 产 N-way 对比三件套。

    baseline_eval_dir 入参支持单路径、含通配符的路径、或路径序列; None 表示无基线,
    直接返回空结果(不写 _manifest, 交由调用方判断)。

    Args:
        layout: RunLayout (用 evaluation_dir + comparison_dir + run_name)
        baseline_eval_dir: 基线评估目录(如 model-recommend/yx_001/evaluation),
                          或多个目录的序列, 或含通配符(如 "model-recommend/*/evaluation");
                          目录不存在或某 split 缺 JSON 时跳过该 split
        produced_by: manifest 来源标识

    Returns:
        {split: [json_path, md_path, xlsx_path]} 各 split 产物路径
    """
    compare_script = _ensure_compare()
    baseline_dirs = _normalize_dirs(baseline_eval_dir)

    all_files: List[Path] = []
    result: Dict[str, List[Path]] = {}
    skipped: List[Dict[str, str]] = []

    if not baseline_dirs:
        print(f"[invoke_comparison] 未传入任何 baseline_eval_dir, 跳过 comparison 阶段")
        return result

    for split in _SPLITS:
        new_json = layout.evaluation_dir / f"{layout.run_name}_{split}_eval.json"
        if not new_json.exists():
            print(f"[invoke_comparison] 跳过 {split}: 新模型 eval JSON 不存在 {new_json}")
            skipped.append({"split": split, "reason": f"new_eval_missing: {new_json.name}"})
            continue

        baseline_jsons = _find_baseline_jsons(baseline_dirs, split)
        if not baseline_jsons:
            print(f"[invoke_comparison] 跳过 {split}: 基线 eval JSON 未找到 in {baseline_dirs}")
            skipped.append({"split": split, "reason": f"baseline_eval_missing in {baseline_dirs}"})
            continue

        if not layout.comparison_dir.exists():
            layout.comparison_dir.mkdir(parents=True, exist_ok=True)

        files = _run_compare(
            compare_script=compare_script,
            new_json=new_json,
            baseline_jsons=baseline_jsons,
            out_dir=layout.comparison_dir,
            split=split,
        )
        all_files.extend(files)
        result[split] = files

    if not result:
        print(f"[invoke_comparison] 无 split 产 comparison 产物, 不写 _manifest, 不留 comparison/ 目录")
        return result

    write_manifest(
        layout.comparison_dir,
        stage="comparison",
        files=all_files,
        extra={
            "compare_engine": "classification-model-comparison/scripts/compare_models.py",
            "baseline_eval_dirs": [str(d) for d in baseline_dirs],
            "splits": list(result.keys()),
            "skipped": skipped,
        },
        produced_by=produced_by,
    )
    return result
