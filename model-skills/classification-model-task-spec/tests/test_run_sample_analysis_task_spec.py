# -*- coding: utf-8 -*-
"""run_sample_analysis_task_spec.py 测试: 标准流程 / 丢弃 / 参数校验。"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import run_sample_analysis_task_spec as rsa


def _make_sample(tmp_path, n_per_pday=10, pday_list=None, extra_rows=None):
    """造一个小 parquet 样本。"""
    if pday_list is None:
        pday_list = ["20260413", "20260430", "20260508", "20260516", "20260524"]
    rows = []
    for pday in pday_list:
        for i in range(n_per_pday):
            rows.append({
                "user_no": "u_%s_%d" % (pday, i),
                "label": 1 if i < 2 else 0,   # 2/10 正样本
                "pday": pday,
            })
    if extra_rows:
        rows.extend(extra_rows)
    df = pd.DataFrame(rows)
    p = tmp_path / "sample.parquet"
    df.to_parquet(p, index=False)
    return p, df


def _run_script(sample_path, tmp_path, train_range, test_range, oot_range,
                model_name="test_model", timestamp="20260629-161231"):
    out_dir = tmp_path / "data-profile"
    out_dir.mkdir(exist_ok=True)
    cmd = [
        sys.executable, str(_SCRIPTS / "run_sample_analysis_task_spec.py"),
        "--sample", str(sample_path),
        "--train-range", train_range,
        "--test-range", test_range,
        "--oot-range", oot_range,
        "--model-name", model_name,
        "--timestamp", timestamp,
        "--output-dir", str(out_dir),
    ]
    import subprocess
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc, out_dir


def test_standard_flow(tmp_path):
    """用例 1: 标准流程 5 个 pday, 60:20:20。"""
    p, df = _make_sample(tmp_path)
    proc, out_dir = _run_script(
        p, tmp_path,
        "20260401,20260510",   # train 覆盖 4/13 4/30 5/8
        "20260511,20260520",   # eval 覆盖 5/16
        "20260521,20260530",   # oot 覆盖 5/24
    )
    assert proc.returncode == 0, "脚本失败: %s\n%s" % (proc.stdout, proc.stderr)

    for fname in ["report.md", "report.xlsx", "_manifest.json", "_split_manifest.json",
                  "train.parquet", "test.parquet", "oot.parquet"]:
        assert (out_dir / fname).exists(), "缺文件: %s" % fname

    manifest = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_summary"]["total_samples"] == 50
    assert manifest["model_name"] == "test_model"
    assert manifest["timestamp"] == "20260629-161231"
    assert manifest["user_confirmed"] is False

    sp = manifest["split"]["splits"]
    total_split = sp["train"]["rows"] + sp["test"]["rows"] + sp["oot"]["rows"]
    assert total_split == 50

    train_df = pd.read_parquet(out_dir / "train.parquet")
    test_df = pd.read_parquet(out_dir / "test.parquet")
    oot_df = pd.read_parquet(out_dir / "oot.parquet")
    assert len(train_df) + len(test_df) + len(oot_df) == 50

    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "样本分析报告" in report_md
    assert "test_model" in report_md

    assert "样本分析 + 切分完成" in proc.stdout


def test_dropped_rows(tmp_path):
    """用例 2: 区间外有行被丢弃。"""
    extra = [{"user_no": "u_extra_%d" % i, "label": 0, "pday": "20260601"} for i in range(10)]
    p, df = _make_sample(tmp_path, extra_rows=extra)
    proc, out_dir = _run_script(
        p, tmp_path,
        "20260401,20260510",
        "20260511,20260520",
        "20260521,20260530",
    )
    assert proc.returncode == 0, "脚本失败: %s\n%s" % (proc.stdout, proc.stderr)
    assert "丢弃" in proc.stderr or "丢弃" in proc.stdout

    manifest = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"]["dropped_rows"] == 10
    sp = manifest["split"]["splits"]
    assert sp["train"]["rows"] + sp["test"]["rows"] + sp["oot"]["rows"] == 50


def test_missing_columns(tmp_path):
    """用例 3a: 缺 user_no 列。"""
    df = pd.DataFrame([{"label": 0, "pday": "20260413"} for _ in range(10)])
    p = tmp_path / "sample.parquet"
    df.to_parquet(p, index=False)
    proc, _ = _run_script(p, tmp_path, "20260401,20260510", "20260511,20260520", "20260521,20260530")
    assert proc.returncode != 0
    assert "user_no" in proc.stdout or "user_no" in proc.stderr


def test_bad_label(tmp_path):
    """用例 3b: label 含 2。"""
    rows = [{"user_no": "u_%d" % i, "label": 2, "pday": "20260413"} for i in range(10)]
    df = pd.DataFrame(rows)
    p = tmp_path / "sample.parquet"
    df.to_parquet(p, index=False)
    proc, _ = _run_script(p, tmp_path, "20260401,20260510", "20260511,20260520", "20260521,20260530")
    assert proc.returncode != 0
    assert "非法" in proc.stdout or "非法" in proc.stderr


def test_bad_pday(tmp_path):
    """用例 3c: pday 含非 8 位。"""
    rows = [{"user_no": "u_%d" % i, "label": 0, "pday": "2026"} for i in range(10)]
    df = pd.DataFrame(rows)
    p = tmp_path / "sample.parquet"
    df.to_parquet(p, index=False)
    proc, _ = _run_script(p, tmp_path, "20260401,20260510", "20260511,20260520", "20260521,20260530")
    assert proc.returncode != 0


def test_overlap_ranges(tmp_path):
    """用例 3d: 三档区间重叠。"""
    p, _ = _make_sample(tmp_path)
    proc, _ = _run_script(
        p, tmp_path,
        "20260401,20260515",   # 与 eval 重叠
        "20260511,20260520",
        "20260521,20260530",
    )
    assert proc.returncode != 0
    assert "重叠" in proc.stdout or "重叠" in proc.stderr


def test_sample_not_exist(tmp_path):
    """用例 3e: --sample 文件不存在。"""
    proc, _ = _run_script(
        tmp_path / "nonexistent.parquet", tmp_path,
        "20260401,20260510", "20260511,20260520", "20260521,20260530",
    )
    assert proc.returncode != 0
    assert "不存在" in proc.stdout or "不存在" in proc.stderr


def test_parse_range_ok():
    """parse_range 合法输入。"""
    assert rsa.parse_range("20260401,20260510") == ("20260401", "20260510")


def test_parse_range_bad():
    """parse_range 非法输入。"""
    with pytest.raises(ValueError):
        rsa.parse_range("20260401")
    with pytest.raises(ValueError):
        rsa.parse_range("2026,20260510")
    with pytest.raises(ValueError):
        rsa.parse_range("20260510,20260401")   # 起 > 止


def test_validate_ranges_adjacent():
    """相邻区间(允许)。"""
    rsa.validate_ranges(("20260401", "20260510"), ("20260511", "20260520"), ("20260521", "20260530"))


def test_validate_ranges_overlap():
    """重叠区间(报错)。"""
    with pytest.raises(ValueError):
        rsa.validate_ranges(("20260401", "20260515"), ("20260511", "20260520"), ("20260521", "20260530"))


def test_judge_stability():
    """稳定性判定阈值。"""
    segs = [{"pday": "20260413", "positive_rate": 0.11},
            {"pday": "20260430", "positive_rate": 0.13},
            {"pday": "20260508", "positive_rate": 0.15}]
    s = rsa.judge_stability(segs)
    assert s["volatility_pp"] == 4.0
    assert s["judgment"] == "显著波动"

    segs2 = [{"pday": "20260413", "positive_rate": 0.13},
             {"pday": "20260430", "positive_rate": 0.135}]
    s2 = rsa.judge_stability(segs2)
    assert s2["judgment"] == "稳定"


def test_judge_sufficiency():
    """充足度判定。"""
    assert rsa.judge_sufficiency({"positive_samples": 15000, "total_samples": 150000})["judgment"] == "充足"
    assert rsa.judge_sufficiency({"positive_samples": 5000, "total_samples": 50000})["judgment"] == "基本可用"
    assert rsa.judge_sufficiency({"positive_samples": 100, "total_samples": 1000})["judgment"] == "不足，建议补充样本"
