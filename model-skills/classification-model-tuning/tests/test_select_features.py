# -*- coding: utf-8 -*-
"""select_features 端到端: label/output_dir 决策 (fast) + E2E (slow)。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import select_features


# -------------------- fast: version resolution --------------------

def test_resolve_version_none_when_no_cli():
    """无 CLI 时返回 None,由 RunLayout.create 自动自增。"""
    assert select_features._resolve_version(None) is None


def test_resolve_version_cli_wins():
    assert select_features._resolve_version("custom_v2") == "custom_v2"


@pytest.mark.parametrize("bad", ["xgb-v1", "lgb-v1", "feat-v1", "tuned", "feat"])
def test_resolve_version_rejects_reserved_tokens(bad):
    """CLI --version/--label 含 algo/suffix 保留字前缀时立即拒绝。

    真实 bug 场景: 用户传 --version feat-v1 会产出 xgb-feat-feat-v1 重复前缀目录。
    本 skill 的 CLI 与 model-training 的 yaml 侧走同一套 validate_version_label 判定。
    """
    with pytest.raises(ValueError, match="保留字"):
        select_features._resolve_version(bad)


# -------------------- fast: output_dir inference --------------------

def test_resolve_output_dir_cli_wins(tmp_path):
    baseline = tmp_path / "new-models" / "xgb-v1"
    assert select_features._resolve_output_dir(baseline, "/tmp/explicit") == "/tmp/explicit"


def test_resolve_output_dir_inferred_from_baseline(tmp_path):
    baseline = tmp_path / "new-models" / "xgb-v1"
    assert select_features._resolve_output_dir(baseline, None) == str(tmp_path)


# -------------------- fast: 非 TTY confirm 守门 --------------------

def test_confirm_non_tty_uses_default(monkeypatch):
    monkeypatch.setattr("sys.stdin", type("F", (), {"isatty": lambda self: False})())
    assert select_features._confirm("ok?", default_yes=True) is True
    assert select_features._confirm("ok?", default_yes=False) is False


# -------------------- slow: E2E --------------------

def _make_baseline_data(tmp_path: Path, n: int = 3000) -> Path:
    """造三档 train/test/oot parquet 到 tmp_path/"splits/", 返回该目录。

    与 model-training 数据契约一致(读 <data_dir>/splits/{train,test,oot}.parquet)。
    样本: 6 个特征, 其中 f3 故意做成"高 PSI + 全缺失"双双触发剔除(由 _fake_analysis_dir 决定)。
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
            "name": "bm_feat", "version": "v1", "sample_table": "db.t",
            "dt_col": "pday", "label_col": "label", "id_cols": ["user_id"],
            "fetch_dt": ["20250101", "20250728"], "where": None,
            "features": features,
        },
    }


def _fake_analysis_dir(tmp_path: Path, features, force_drop) -> Path:
    """造 feature-analysis csv: force_drop 中的特征故意触发剔除规则。"""
    ana = tmp_path / "ana"
    ana.mkdir()
    # missing: force_drop[0] = 0.99 (>0.95)
    miss = [0.99 if f == force_drop[0] else 0.05 for f in features]
    pd.DataFrame({"feature": features, "missing_rate": miss}).to_csv(
        ana / "stats.csv", index=False
    )
    # iv: force_drop[1] = 0.005 (<0.02)
    ivs = [0.005 if f == force_drop[1] else 0.30 for f in features]
    pd.DataFrame({"feature": features, "iv": ivs}).to_csv(
        ana / "iv_table.csv", index=False
    )
    # psi: force_drop[2] = 0.5 (>0.10)
    psis = [0.5 if f == force_drop[2] else 0.03 for f in features]
    pd.DataFrame({"feature": features, "psi": psis}).to_csv(
        ana / "psi_table.csv", index=False
    )
    return ana


@pytest.mark.slow
def test_end_to_end_select_features_full_layout(tmp_path):
    """跑 baseline → 造 analysis csv → select_features → 验证产物 + 配置。"""
    from run_build import run as run_training

    data_dir = _make_baseline_data(tmp_path)
    features = [f"f{i}" for i in range(6)]
    session_dir = tmp_path / "session"
    res_base = run_training(_baseline_cfg(features), str(data_dir),
                            str(session_dir), version="v1")
    baseline_dir = Path(res_base["run_dir"])

    # 让 f0(高缺失) / f1(低 IV) / f2(高 PSI) 各因不同规则被剔除
    ana_dir = _fake_analysis_dir(tmp_path, features, force_drop=("f0", "f1", "f2"))

    args = argparse.Namespace(
        baseline_run=str(baseline_dir),
        analysis_dir=str(ana_dir),
        label=None,
        version=None,
        output_dir=None,
        auto_apply=True,
        no_psi=False, no_iv=False, no_missing=False,
        psi_threshold=0.10, iv_threshold=0.02, missing_threshold=0.95,
    )
    res = select_features.run(args)

    new_dir = Path(res["run_dir"])
    assert new_dir.parent == baseline_dir.parent           # 同级 new-models/
    assert new_dir.name == "xgb-feat-v1"
    assert res["baseline_run"] == baseline_dir.name
    assert res["n_kept"] == 3
    assert res["n_dropped"] == 3

    # 八段产物齐全 + produced_by 标记
    assert (new_dir / "config.json").exists()
    for sub in ("features", "model", "evaluation",
                "predictions", "explainability", "comparison", "logs"):
        m_path = new_dir / sub / "_manifest.json"
        assert m_path.exists()
        m = json.loads(m_path.read_text())
        assert m["produced_by"] == "skills/model-tuning"

    # config.json runtime 含 selection 字段
    snap = json.loads((new_dir / "config.json").read_text())
    assert snap["produced_by"] == "skills/model-tuning"
    runtime = snap["runtime"]
    assert runtime["baseline_run"] == baseline_dir.name
    assert "selection" in runtime
    sel = runtime["selection"]
    assert set(sel["kept_features"]) == {"f3", "f4", "f5"}
    assert set(sel["dropped_features"]) == {"f0", "f1", "f2"}
    assert sel["dropped_by_rule"]["high_missing"] == ["f0"]
    assert sel["dropped_by_rule"]["low_iv"] == ["f1"]
    assert sel["dropped_by_rule"]["high_psi"] == ["f2"]
    assert sel["rules_enabled"] == {"high_psi": True, "low_iv": True, "high_missing": True}
    assert runtime["analysis_dir"] == str(ana_dir.resolve())

    # 关键文件可用
    assert (new_dir / "model" / "model.json").exists()
    assert (new_dir / "evaluation" / f"{new_dir.name}_oot_eval.md").exists()
    assert (new_dir / "comparison" / "comparison_oot.md").exists()
    assert (new_dir / "predictions" / "oot_predictions.parquet").exists()
    assert (new_dir / "logs" / "run.log").stat().st_size > 0
    assert (new_dir / "logs" / "select_features.log").stat().st_size > 0
    m = json.loads((new_dir / "logs" / "_manifest.json").read_text())
    names = [f["name"] for f in m["files"]]
    assert "select_features.log" in names

    # features stage 的 used-feature-list.csv: 三列 feature_name/status/dropped_by_rule
    # kept 行只有 f3/f4/f5; dropped 行覆盖 f0/f1/f2, 标对应规则
    used_list = pd.read_csv(new_dir / "features" / "used-feature-list.csv")
    kept = used_list[used_list["status"] == "kept"]
    dropped = used_list[used_list["status"].str.startswith("dropped_")]
    assert set(kept["feature_name"]) == {"f3", "f4", "f5"}
    assert set(dropped["feature_name"]) == {"f0", "f1", "f2"}
    assert set(dropped["dropped_by_rule"]) == {"high_missing", "low_iv", "high_psi"}
