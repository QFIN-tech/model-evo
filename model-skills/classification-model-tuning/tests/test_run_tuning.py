# -*- coding: utf-8 -*-
"""run_tuning 端到端: version 决策 / 非 xgb 守门 (fast) + E2E (slow)。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

# 注: import run_tuning 会触发 _bootstrap,把 model-training/scripts 注入 sys.path
import run_tuning


# -------------------- fast: version resolution --------------------

def test_resolve_version_none_when_no_cli():
    """无 CLI 时返回 None,由 RunLayout.create 自动自增。"""
    assert run_tuning._resolve_version(None) is None


def test_resolve_version_cli_wins():
    assert run_tuning._resolve_version("custom_v2") == "custom_v2"


@pytest.mark.parametrize("bad", ["xgb-v1", "lgb-v1", "tuned-v1", "feat", "feat-v2"])
def test_resolve_version_rejects_reserved_tokens(bad):
    """CLI --version/--label 含 algo/suffix 保留字前缀时立即拒绝。

    真实 bug 场景: 用户传 --version tuned-v1 会产出 xgb-tuned-tuned-v1 重复前缀目录。
    本 skill 的 CLI 与 model-training 的 yaml 侧走同一套 validate_version_label 判定。
    """
    with pytest.raises(ValueError, match="保留字"):
        run_tuning._resolve_version(bad)


# -------------------- fast: output_dir inference --------------------

def test_resolve_output_dir_cli_wins(tmp_path):
    baseline = tmp_path / "new-models" / "xgb-v1"
    assert run_tuning._resolve_output_dir(baseline, "/tmp/explicit") == "/tmp/explicit"


def test_resolve_output_dir_inferred_from_baseline(tmp_path):
    """baseline_run.parent.parent == session_dir(其下挂 new-models/)。"""
    baseline = tmp_path / "new-models" / "xgb-v1"
    assert run_tuning._resolve_output_dir(baseline, None) == str(tmp_path)


# -------------------- fast: 非 TTY confirm 守门 --------------------

def test_confirm_non_tty_uses_default(monkeypatch):
    """非 TTY 时 _confirm 按 default_yes 走,不会阻塞 input。"""
    monkeypatch.setattr("sys.stdin", type("F", (), {"isatty": lambda self: False})())
    assert run_tuning._confirm("ok?", default_yes=True) is True
    assert run_tuning._confirm("ok?", default_yes=False) is False


# -------------------- slow: E2E --------------------

def _make_baseline_data(tmp_path: Path, n: int = 3000) -> Path:
    """造三档 train/test/oot parquet 到 tmp_path/"splits/", 返回该目录。

    与 model-training 数据契约一致(读 <data_dir>/splits/{train,test,oot}.parquet)。
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 6))
    lin = X @ np.array([1.2, -0.8, 0.6, 0.0, 0.4, -0.5])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-lin))).astype(int)
    days = pd.date_range("2025-01-01", "2025-07-28", periods=n).strftime("%Y%m%d")
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["label"] = y
    df["pday"] = days
    df["base_score"] = 1 / (1 + np.exp(-(X @ np.array([1, -0.7, 0.5, 0, 0.3, -0.4]))))
    df["user_id"] = [f"u{i}" for i in range(n)]
    n_train = int(n * 0.6); n_test = int(n * 0.2)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    df.iloc[:n_train].reset_index(drop=True).to_parquet(splits_dir / "train.parquet", index=False)
    df.iloc[n_train:n_train + n_test].reset_index(drop=True).to_parquet(splits_dir / "test.parquet", index=False)
    df.iloc[n_train + n_test:].reset_index(drop=True).to_parquet(splits_dir / "oot.parquet", index=False)
    return tmp_path


def _baseline_cfg(features):
    return {
        "spark": {"app_name": "t", "master": "local[*]"},
        "model": {
            "name": "bm_tune", "version": "v1", "sample_table": "db.t",
            "dt_col": "pday", "label_col": "label", "id_cols": ["user_id"],
            "fetch_dt": ["20250101", "20250728"], "where": None,
            "features": features,
        },
    }


@pytest.mark.slow
def test_end_to_end_rule_tuning_full_layout(tmp_path):
    """跑 baseline (model-training) → rule 调优 (model-tuning) → 验证产物。"""
    from run_build import run as run_training

    data_dir = _make_baseline_data(tmp_path)
    features = [f"f{i}" for i in range(6)]
    session_dir = tmp_path / "session"
    res_base = run_training(_baseline_cfg(features), str(data_dir),
                            str(session_dir), version="v1")
    baseline_dir = Path(res_base["run_dir"])

    # 调用 run_tuning.run 模拟 CLI args
    args = argparse.Namespace(
        baseline_run=str(baseline_dir),
        method="rule",
        n_trials=3,
        label=None,
        version=None,
        output_dir=None,
        auto_apply=True,
    )
    res_tune = run_tuning.run(args)

    tuned_dir = Path(res_tune["run_dir"])
    assert tuned_dir.parent == baseline_dir.parent           # 同级 new-models/
    assert tuned_dir.name == "xgb-tuned-v1"
    assert res_tune["baseline_run"] == baseline_dir.name

    # 八段产物齐全
    assert (tuned_dir / "config.json").exists()
    for sub in ("features", "model", "evaluation",
                "predictions", "explainability", "comparison", "logs"):
        assert (tuned_dir / sub / "_manifest.json").exists()

    # config.json runtime 含调优关键字段
    snap = json.loads((tuned_dir / "config.json").read_text())
    assert snap["produced_by"] == "skills/model-tuning"
    runtime = snap["runtime"]
    assert runtime["baseline_run"] == baseline_dir.name
    assert "diagnosis" in runtime
    assert runtime["method"] == "rule"
    assert "recommended_params" in runtime
    assert "final_params" in runtime
    assert "baseline_metrics" in runtime

    # 每个 stage manifest 的 produced_by 应为 model-tuning
    for sub in ("features", "model", "evaluation",
                "predictions", "explainability", "comparison", "logs"):
        m = json.loads((tuned_dir / sub / "_manifest.json").read_text())
        assert m["produced_by"] == "skills/model-tuning"

    # 关键文件可用
    assert (tuned_dir / "model" / "model.json").exists()
    assert (tuned_dir / "evaluation" / f"{tuned_dir.name}_oot_eval.md").exists()
    assert (tuned_dir / "comparison" / "comparison_oot.md").exists()
    assert (tuned_dir / "predictions" / "oot_predictions.parquet").exists()
    assert (tuned_dir / "logs" / "run.log").stat().st_size > 0
    assert (tuned_dir / "logs" / "run_tuning.log").stat().st_size > 0
    # _manifest.json files 列表应含进程级日志
    m = json.loads((tuned_dir / "logs" / "_manifest.json").read_text())
    names = [f["name"] for f in m["files"]]
    assert "run_tuning.log" in names


# -------------------- slow: dnn E2E --------------------

def _baseline_cfg_dnn(features):
    """dnn baseline cfg: 用 dnn trainer, 调小 epochs 加速测试。"""
    cfg = _baseline_cfg(features)
    cfg["model"]["algo"] = "dnn"
    cfg["model"]["hyper_params"] = {
        "hidden_dims": [32, 16], "dropout": 0.2, "learning_rate": 0.005,
        "weight_decay": 1e-4, "batch_size": 256, "epochs": 30, "patience": 5,
        "pos_weight": "auto",
    }
    return cfg


@pytest.mark.slow
def test_end_to_end_rule_tuning_dnn(tmp_path):
    """dnn baseline → rule 调优 → 验证 algo 直通 (不切到 xgb)。"""
    from run_build import run as run_training

    data_dir = _make_baseline_data(tmp_path)
    features = [f"f{i}" for i in range(6)]
    session_dir = tmp_path / "session"
    res_base = run_training(_baseline_cfg_dnn(features), str(data_dir),
                            str(session_dir), version="v1")
    baseline_dir = Path(res_base["run_dir"])

    # 验证 baseline 落的是 dnn
    assert baseline_dir.name == "dnn-v1"

    args = argparse.Namespace(
        baseline_run=str(baseline_dir),
        method="rule",
        n_trials=3,
        label=None,
        version=None,
        output_dir=None,
        auto_apply=True,
    )
    res_tune = run_tuning.run(args)

    tuned_dir = Path(res_tune["run_dir"])
    # 关键: tuned run 也是 dnn (不切到 xgb)
    assert tuned_dir.name == "dnn-tuned-v1"

    snap = json.loads((tuned_dir / "config.json").read_text())
    runtime = snap["runtime"]
    assert "diagnosis" in runtime
    assert "recommended_params" in runtime
    # recommended_params 应包含 dnn 键 (dropout/learning_rate 等), 不含 xgb 键
    rec = runtime["recommended_params"]
    assert "dropout" in rec or "learning_rate" in rec
    assert "max_depth" not in rec and "n_estimators" not in rec


# -------------------- slow: lr E2E --------------------

def _baseline_cfg_lr(features):
    """lr baseline cfg: 用 lr trainer。"""
    cfg = _baseline_cfg(features)
    cfg["model"]["algo"] = "lr"
    cfg["model"]["hyper_params"] = {
        "max_n_bins": 8, "min_bin_size": 0.05,
        "regularization": "l2", "C": 1.0, "max_iter": 500,
    }
    return cfg


@pytest.mark.slow
def test_end_to_end_rule_tuning_lr(tmp_path):
    """lr baseline → rule 调优 → 验证 algo 直通。"""
    from run_build import run as run_training

    data_dir = _make_baseline_data(tmp_path)
    features = [f"f{i}" for i in range(6)]
    session_dir = tmp_path / "session"
    res_base = run_training(_baseline_cfg_lr(features), str(data_dir),
                            str(session_dir), version="v1")
    baseline_dir = Path(res_base["run_dir"])

    assert baseline_dir.name == "lr-v1"

    args = argparse.Namespace(
        baseline_run=str(baseline_dir),
        method="rule",
        n_trials=3,
        label=None,
        version=None,
        output_dir=None,
        auto_apply=True,
    )
    res_tune = run_tuning.run(args)

    tuned_dir = Path(res_tune["run_dir"])
    assert tuned_dir.name == "lr-tuned-v1"

    snap = json.loads((tuned_dir / "config.json").read_text())
    runtime = snap["runtime"]
    rec = runtime["recommended_params"]
    # recommended_params 应含 lr 键 (C/max_n_bins 等), 不含 xgb/dnn 键
    assert "C" in rec or "max_n_bins" in rec
    assert "max_depth" not in rec and "dropout" not in rec
