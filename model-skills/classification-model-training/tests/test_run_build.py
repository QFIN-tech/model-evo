# -*- coding: utf-8 -*-
"""run_build 端到端测试: feature-analysis 产出的 splits/ 三档 → 7 子目录产物 + label 决议 + 多算法子包隔离。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from run_build import _resolve_version, _resolve_data_dir


def _make_split_parquets(tmp_path: Path, n: int = 3000) -> Path:
    """造 feature-analysis 风格的 splits/{train,test,oot}.parquet 到 tmp_path/splits/, 返回 tmp_path。

    pday 范围横跨 2025-01-01 ~ 2025-07-28, 按 train/test/oot 三段切。
    同时落一个 feature-analysis/analysis/_manifest.json (含 overview) 以验证 split_report 提取。
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 6))
    lin = X @ np.array([1.2, -0.8, 0.6, 0.0, 0.4, -0.5])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-lin))).astype(int)
    days = pd.date_range("2025-01-01", "2025-07-28", periods=n).strftime("%Y%m%d").astype(int)

    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["label"] = y
    df["pday"] = days
    df["user_id"] = [f"u{i}" for i in range(n)]

    # 三档切分 (与 _SPLIT_RANGES 区间对齐)
    train_df = df.query("pday >= 20250101 and pday <= 20250415").reset_index(drop=True)
    test_df = df.query("pday >= 20250416 and pday <= 20250531").reset_index(drop=True)
    oot_df = df.query("pday >= 20250601 and pday <= 20250728").reset_index(drop=True)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir(parents=True)
    train_df.to_parquet(splits_dir / "train.parquet", index=False)
    test_df.to_parquet(splits_dir / "test.parquet", index=False)
    oot_df.to_parquet(splits_dir / "oot.parquet", index=False)

    # 落 feature-analysis manifest (含 overview) 供 _load_pre_split_data 提取 split_report
    # _load_pre_split_data 查找 splits_dir.parent / "feature-analysis" / "analysis" / "_manifest.json"
    # splits_dir = tmp_path / "splits", 故 splits_dir.parent = tmp_path, manifest 应落在 tmp_path 下
    fa_dir = tmp_path / "feature-analysis" / "analysis"
    fa_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "produced_by": "skills/feature-analysis",
        "files": [],
        "overview": {
            "n_total": n,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_oot": len(oot_df),
            "split_strategy": "explicit",
            "oot_boundary": "pday >= 20250601",
            "sample_counts": {"train": len(train_df), "val": len(test_df), "oot": len(oot_df)},
            "pos_rates": {
                "train": float(train_df["label"].mean()),
                "val": float(test_df["label"].mean()),
                "oot": float(oot_df["label"].mean()),
            },
            "time_col_used": "pday",
        },
    }
    (fa_dir / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def _base_cfg(features):
    return {
        "spark": {"app_name": "t", "master": "local[*]"},
        "model": {
            "name": "bm_e2e", "version": "v1", "sample_table": "db.t",
            "dt_col": "pday", "label_col": "label", "id_cols": ["user_id"],
            "fetch_dt": ["20250101", "20250728"], "where": None,
            "features": features,
        },
    }


def test_resolve_version_priority_cli_wins():
    """CLI --version 优先于 yaml.model.run_label。"""
    assert _resolve_version("v2", {"run_label": "from_yaml"}) == "v2"


def test_resolve_version_yaml_fallback():
    """无 CLI 时回退到 yaml.run_label。"""
    assert _resolve_version(None, {"run_label": "from_yaml"}) == "from_yaml"


def test_resolve_version_none_when_missing():
    """CLI/yaml 都没有 → 返回 None(交给 RunLayout.create 自动自增)。"""
    assert _resolve_version(None, {}) is None


def test_validate_config_rejects_empty_features_local_file():
    """local_file 模式 features=[] + feature_list_source=null 必须在 validate_config 拦截,
    不能透传到 xgboost 深处报 `0 feature is supplied`。"""
    from validate_config import validate_config
    from config_io import validate_common

    cfg = {
        "model": {
            "name": "m", "version": "v1",
            "mode": "local_file",
            "dt_col": "pday", "label_col": "label", "id_cols": ["user_id"],
            "features": [],
            "feature_list_source": None,
        }
    }
    # validate_common 在 local_file 模式放行空 features(原意 fallback 全列, 但 run_build 不实现)
    validate_common(cfg)
    # validate_config 必须补上训练专有校验, 在入训练前就报错
    with pytest.raises(ValueError, match="非空特征清单"):
        validate_config(cfg)


def test_validate_config_accepts_features_from_list_source(tmp_path):
    """local_file 模式下 features=[] + feature_list_source 指向 csv 时应通过(validate_common
    会把 features 加载为非空列表, validate_config 不再报错)。"""
    from validate_config import validate_config

    feats_csv = tmp_path / "feature-list.csv"
    feats_csv.write_text("feature_name\nf0\nf1\n", encoding="utf-8")

    cfg = {
        "model": {
            "name": "m", "version": "v1",
            "mode": "local_file",
            "dt_col": "pday", "label_col": "label", "id_cols": ["user_id"],
            "features": [],
            "feature_list_source": str(feats_csv),
        }
    }
    validate_config(cfg)
    assert cfg["model"]["features"] == ["f0", "f1"]


@pytest.mark.parametrize("bad_label", ["xgb-v1", "lgb-v1", "tuned-v1", "feat", "feat-v2"])
def test_validate_config_rejects_bad_run_label(tmp_path, bad_label):
    """yaml.model.run_label 含 algo/suffix 保留字前缀时, validate_config 直接拒绝。

    真实 bug 场景: 用户/agent 把"模型简称"塞给 run_label(lgb-v1 / tuned-v1 / feat),
    与目录规则 `{algo}{suffix}-{version}` 叠加产生 lgb-lgb-v1 / xgb-tuned-tuned-v1 /
    xgb-feat 等重复前缀目录。本测试锁死: 这类值在 config 加载期就拦截, 不进 run_dir 创建。
    """
    from validate_config import validate_config

    feats_csv = tmp_path / "feature-list.csv"
    feats_csv.write_text("feature_name\nf0\nf1\n", encoding="utf-8")
    cfg = {
        "model": {
            "name": "m", "version": "v1",
            "mode": "local_file",
            "dt_col": "pday", "label_col": "label", "id_cols": ["user_id"],
            "run_label": bad_label,
            "features": [],
            "feature_list_source": str(feats_csv),
        }
    }
    with pytest.raises(ValueError, match="run_label 非法"):
        validate_config(cfg)


def test_validate_config_accepts_pure_run_label(tmp_path):
    """model.run_label 为纯版本号(如 v2)时 validate_config 通过。"""
    from validate_config import validate_config

    feats_csv = tmp_path / "feature-list.csv"
    feats_csv.write_text("feature_name\nf0\nf1\n", encoding="utf-8")
    cfg = {
        "model": {
            "name": "m", "version": "v1",
            "mode": "local_file",
            "dt_col": "pday", "label_col": "label", "id_cols": ["user_id"],
            "run_label": "v2",
            "features": [],
            "feature_list_source": str(feats_csv),
        }
    }
    validate_config(cfg)  # 不抛即通过


def test_resolve_data_dir_explicit_wins(tmp_path):
    """CLI --data_dir 优先于默认推断。"""
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (explicit / "splits" / "train.parquet").parent.mkdir(parents=True)
    (explicit / "splits" / "train.parquet").write_bytes(b"")
    default_dir = tmp_path / "sample-features"
    default_dir.mkdir(parents=True)
    (default_dir / "splits").mkdir()
    (default_dir / "splits" / "train.parquet").write_bytes(b"")
    assert _resolve_data_dir(str(explicit), str(tmp_path)) == str(explicit)


def test_resolve_data_dir_infers_from_session(tmp_path, capsys):
    """未传 --data_dir 时从 <session_dir>/sample-features/splits/train.parquet 推断。"""
    default_dir = tmp_path / "sample-features"
    default_dir.mkdir(parents=True)
    (default_dir / "splits").mkdir()
    (default_dir / "splits" / "train.parquet").write_bytes(b"")
    resolved = _resolve_data_dir(None, str(tmp_path))
    assert resolved == str(default_dir)
    assert "自动推断" in capsys.readouterr().out


def test_resolve_data_dir_missing_raises(tmp_path):
    """未传 --data_dir 且默认路径不存在 → SystemExit 报错。"""
    with pytest.raises(SystemExit):
        _resolve_data_dir(None, str(tmp_path))


@pytest.mark.slow
def test_end_to_end_xgb_full_layout(tmp_path):
    """xgb 端到端: splits/ 三档 → 7 个子目录所有 manifest/产物齐全 + 关键文件可用。"""
    from run_build import run

    # _make_split_parquets 会同时落 splits/ 和 feature-analysis manifest
    # 布局: tmp_path/splits/ + tmp_path.parent/feature-analysis/analysis/_manifest.json
    # run_build._load_pre_split_data 从 splits_dir.parent/feature-analysis/analysis/ 找 manifest
    splits_root = tmp_path / "sample-features"
    splits_root.mkdir(parents=True)
    data_dir = _make_split_parquets(splits_root)
    features = [f"f{i}" for i in range(6)]
    out_dir = tmp_path / "model-training"
    src_yaml = tmp_path / "train_config.yaml"
    src_yaml.write_text("model:\n  algo: xgb\n  features: [f0, f1]\n", encoding="utf-8")
    res = run(_base_cfg(features), str(data_dir), str(out_dir), version="v1",
              source_yaml_path=str(src_yaml))

    run_dir = Path(res["run_dir"])
    assert run_dir.parent.name == "new-models"
    assert run_dir.name == "xgb-v1"

    assert (run_dir / "config.json").exists()
    for sub in ("features", "model", "evaluation",
                "predictions", "explainability", "logs", "config"):
        assert (run_dir / sub / "_manifest.json").exists(), f"缺 {sub}/_manifest.json"

    assert (run_dir / "features" / "used-feature-list.csv").exists()
    assert (run_dir / "model" / "model.json").exists()
    run_name = run_dir.name
    assert (run_dir / "evaluation" / f"{run_name}_train_eval.md").exists()
    assert (run_dir / "evaluation" / f"{run_name}_test_eval.md").exists()
    assert (run_dir / "evaluation" / f"{run_name}_oot_eval.md").exists()
    assert (run_dir / "evaluation" / f"{run_name}_oot_eval.xlsx").exists()
    assert (run_dir / "predictions" / "oot_predictions.parquet").exists()
    assert (run_dir / "predictions" / "report.md").exists()
    assert (run_dir / "explainability" / "feature-importance.csv").exists()
    assert (run_dir / "explainability" / "shap-summary.csv").exists()
    assert (run_dir / "logs" / "run.log").exists()

    preds = pd.read_parquet(run_dir / "predictions" / "oot_predictions.parquet")
    assert set(["user_id", "label", "score", "bucket"]).issubset(preds.columns)
    assert preds["bucket"].between(1, 10).all()

    md = (run_dir / "evaluation" / f"{run_name}_oot_eval.md").read_text(encoding="utf-8")
    assert "AUC" in md and "KS" in md

    cfg_snap = json.loads((run_dir / "config.json").read_text())
    assert "metrics" in cfg_snap["runtime"]
    assert cfg_snap["version"] == "v1"
    assert cfg_snap["suffix"] == ""
    assert cfg_snap["label"] == "v1"
    assert "data_dir" in cfg_snap["input"]
    assert "train_path" in cfg_snap["input"]
    assert "test_path" in cfg_snap["input"]
    assert "oot_path" in cfg_snap["input"]
    # pre-split 模式: split_mode = "pre-split", 三档路径指向 feature-analysis splits/
    assert cfg_snap["runtime"]["split_mode"] == "pre-split"
    assert "splits/train.parquet" in cfg_snap["input"]["train_path"]
    assert "splits/test.parquet" in cfg_snap["input"]["test_path"]
    assert "splits/oot.parquet" in cfg_snap["input"]["oot_path"]
    # split_report 从 feature-analysis manifest 提取
    sr = cfg_snap["runtime"]["split_report"]
    assert sr["strategy"] == "explicit"
    assert sr["counts"]["train"] > 0
    assert sr["counts"]["val"] > 0
    assert sr["counts"]["oot"] > 0

    assert (run_dir / "logs" / "run.log").stat().st_size > 0
    assert (run_dir / "logs" / "run_build.log").stat().st_size > 0
    log_m = json.loads((run_dir / "logs" / "_manifest.json").read_text())
    log_names = [f["name"] for f in log_m["files"]]
    assert "run_build.log" in log_names

    assert (run_dir / "config" / "train_config.yaml").exists()
    cfg_m = json.loads((run_dir / "config" / "_manifest.json").read_text())
    assert cfg_m["stage"] == "config"
    assert cfg_m["source_yaml"] == str(src_yaml.resolve())


def _make_split_parquets_shared(tmp_path: Path, algo: str) -> tuple:
    """造 splits/ 三档 + 配置(不含 split, 切分由 feature-analysis 负责), 返回 (data_dir, cfg)。"""
    splits_root = tmp_path / "sample-features"
    splits_root.mkdir(parents=True)
    data_dir = _make_split_parquets(splits_root)
    cfg = {
        "model": {
            "name": f"bm_{algo}", "version": "v1", "dt_col": "pday",
            "label_col": "label", "algo": algo,
            "features": [f"f{i}" for i in range(6)],
            "id_cols": ["user_id"],
            "sample_table": "db.t", "fetch_dt": ["20250101", "20250728"],
        },
    }
    return data_dir, cfg


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["dnn", "lr"])
def test_non_xgb_end_to_end_inproc(tmp_path, algo):
    """dnn / lr 路径端到端(独立子包, in-process)。"""
    from run_build import run

    data_dir, cfg = _make_split_parquets_shared(tmp_path, algo)
    out_dir = tmp_path / "out"
    res = run(cfg, str(data_dir), str(out_dir), version="v1")

    run_dir = Path(res["run_dir"])
    ext = "json" if algo == "xgb" else "pkl"
    assert (run_dir / "model" / f"model.{ext}").exists()
    run_name = run_dir.name
    assert (run_dir / "evaluation" / f"{run_name}_oot_eval.md").exists()
    assert (run_dir / "predictions" / "oot_predictions.parquet").exists()

    scorecard = run_dir / "model" / "scorecard.csv"
    if algo == "lr":
        assert scorecard.exists(), "lr 路径应落 model/scorecard.csv"
        m = json.loads((run_dir / "model" / "_manifest.json").read_text())
        assert m.get("has_scorecard") is True
        cols = pd.read_csv(scorecard).columns.tolist()
        assert cols == ["feature", "bin", "woe", "coef", "score"]
    else:
        assert not scorecard.exists(), "非 lr 路径不应产 scorecard.csv"
        m = json.loads((run_dir / "model" / "_manifest.json").read_text())
        assert m.get("has_scorecard") is False


@pytest.mark.slow
def test_end_to_end_pre_split_reads_feature_analysis_splits(tmp_path):
    """splits/ 三档 (feature-analysis 产) → 端到端训练 (pre-split 模式)。

    验证:
    - run_build 直接读 splits/{train,test,oot}.parquet, 不产 .cache/splits/
    - config.json.runtime.split_mode == "pre-split"
    - config.json.runtime.split_report 从 feature-analysis _manifest.json 提取
    - 后续产物 (model/evaluation/predictions) 齐备
    """
    from run_build import run

    splits_root = tmp_path / "sample-features"
    splits_root.mkdir(parents=True)
    data_dir = _make_split_parquets(splits_root)
    features = [f"f{i}" for i in range(6)]
    cfg = {
        "model": {
            "name": "bm_presplit", "version": "v1", "algo": "xgb",
            "dt_col": "pday", "label_col": "label", "id_cols": ["user_id"],
            "features": features,
            "sample_table": "db.t", "fetch_dt": ["20250101", "20250728"],
        },
    }
    out_dir = tmp_path / "out"
    res = run(cfg, str(data_dir), str(out_dir), version="v1")

    run_dir = Path(res["run_dir"])
    assert run_dir.name == "xgb-v1"

    # pre-split 模式: 不产 .cache/splits/
    assert not (run_dir / ".cache" / "splits").exists()

    cfg_snap = json.loads((run_dir / "config.json").read_text())
    rt = cfg_snap["runtime"]
    assert rt["split_mode"] == "pre-split"
    sr = rt["split_report"]
    assert sr["strategy"] == "explicit"
    assert "train" in sr["counts"]
    assert "val" in sr["counts"]
    assert "oot" in sr["counts"]
    assert sr["oot_boundary"].startswith("pday >=")

    assert sr["counts"]["train"] > 0
    assert sr["counts"]["val"] > 0
    assert sr["counts"]["oot"] > 0

    run_name = run_dir.name
    assert (run_dir / "model" / "model.json").exists()
    assert (run_dir / "evaluation" / f"{run_name}_oot_eval.md").exists()
    assert (run_dir / "predictions" / "oot_predictions.parquet").exists()

    report_md = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "split_mode" in report_md
    assert "pre-split" in report_md


def test_resolve_data_dir_infers_splits_only(tmp_path, capsys):
    """默认路径仅有 splits/train.parquet → 自动推断成功。"""
    default_dir = tmp_path / "sample-features"
    default_dir.mkdir(parents=True)
    (default_dir / "splits").mkdir()
    (default_dir / "splits" / "train.parquet").write_bytes(b"")
    resolved = _resolve_data_dir(None, str(tmp_path))
    assert resolved == str(default_dir)
    assert "自动推断" in capsys.readouterr().out


@pytest.mark.slow
def test_boundary_filter_drops_leakage_and_constant(tmp_path):
    """边界过滤端到端: 造 stats.csv/feature-quality.csv 让 f0/f1/f5 命中规则,
    验证被剔除 + dropped_by_rule 落 csv + config.json.runtime.boundary_filter 摘要。"""
    from run_build import run

    # 1. 造 splits/ 三档 + feature-analysis manifest (复用 _make_split_parquets)
    splits_root = tmp_path / "sample-features"
    splits_root.mkdir(parents=True)
    data_dir = _make_split_parquets(splits_root)

    # 2. 在 feature-analysis/analysis/ 下补写 stats.csv + feature-quality.csv
    # _make_split_parquets 用 f0..f5 六个特征, 我们让:
    #   f0: unique=1 / std=0 命中 constant
    #   f1: iv=1.5 命中 leakage
    #   f5: missing_rate=1.0 命中 all_missing (unique/std 正常, 避免与 constant 撞规则)
    #   f2/f3/f4: 全合格保留
    analysis_dir = data_dir / "feature-analysis" / "analysis"
    stats = pd.DataFrame({
        "feature": ["f0", "f1", "f2", "f3", "f4", "f5"],
        "unique": [1, 100, 50, 80, 60, 50],
        "std": [0.0, 1.0, 2.0, 1.5, 1.2, 0.5],
        "missing_rate": [0.1, 0.2, 0.3, 0.05, 0.1, 1.0],
    })
    fq = pd.DataFrame({
        "feature": ["f0", "f1", "f2", "f3", "f4", "f5"],
        "iv": [0.3, 1.5, 0.5, 0.6, 0.4, 0.2],
        "auc": [0.6, 0.99, 0.65, 0.7, 0.62, 0.55],
    })
    stats.to_csv(analysis_dir / "stats.csv", index=False)
    fq.to_csv(analysis_dir / "feature-quality.csv", index=False)

    # 3. 跑 run (algo=xgb, 6 个特征全配, 让边界过滤剔除 3 个)
    features = ["f0", "f1", "f2", "f3", "f4", "f5"]
    cfg = {
        "model": {
            "name": "bm_bf", "version": "v1", "algo": "xgb",
            "dt_col": "pday", "label_col": "label", "id_cols": ["user_id"],
            "features": features,
            "sample_table": "db.t", "fetch_dt": ["20250101", "20250728"],
        },
    }
    out_dir = tmp_path / "out"
    res = run(cfg, str(data_dir), str(out_dir), version="v1")
    run_dir = Path(res["run_dir"])

    # 4. 断言 used-feature-list.csv 含 dropped_<rule> 行
    used_csv = run_dir / "features" / "used-feature-list.csv"
    assert used_csv.exists()
    df = pd.read_csv(used_csv)
    # 三列结构
    assert list(df.columns) == ["feature_name", "status", "dropped_by_rule"]
    # f0 dropped_constant
    f0_row = df[df["feature_name"] == "f0"].iloc[0]
    assert f0_row["status"] == "dropped_constant"
    assert f0_row["dropped_by_rule"] == "constant"
    # f1 dropped_leakage
    f1_row = df[df["feature_name"] == "f1"].iloc[0]
    assert f1_row["status"] == "dropped_leakage"
    assert f1_row["dropped_by_rule"] == "leakage"
    # f5 dropped_all_missing
    f5_row = df[df["feature_name"] == "f5"].iloc[0]
    assert f5_row["status"] == "dropped_all_missing"
    assert f5_row["dropped_by_rule"] == "all_missing"
    # f2/f3/f4 kept
    kept = set(df[df["status"] == "kept"]["feature_name"].tolist())
    assert kept == {"f2", "f3", "f4"}

    # 5. 断言 config.json.runtime.boundary_filter 摘要
    cfg_snap = json.loads((run_dir / "config.json").read_text())
    bf = cfg_snap["runtime"]["boundary_filter"]
    assert bf["n_before"] == 6
    assert bf["n_after"] == 3
    assert bf["n_dropped"] == 3
    assert bf["dropped_by_rule"]["constant"] == 1
    assert bf["dropped_by_rule"]["leakage"] == 1
    assert bf["dropped_by_rule"]["all_missing"] == 1
    assert bf["sample_total"] > 0

    # 6. 断言训练实际入模特征 = [f2, f3, f4] (从 feature-importance.csv 验证)
    fi = pd.read_csv(run_dir / "explainability" / "feature-importance.csv")
    trained_features = set(fi["feature"].tolist())
    assert trained_features == {"f2", "f3", "f4"}, f"训练特征应为 f2/f3/f4, 实际: {trained_features}"

    # 7. 断言 features/report.md 含"边界特征过滤"段
    report = (run_dir / "features" / "report.md").read_text(encoding="utf-8")
    assert "边界特征过滤" in report
    assert "constant" in report and "leakage" in report and "all_missing" in report
