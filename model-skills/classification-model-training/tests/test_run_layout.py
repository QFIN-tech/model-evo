# -*- coding: utf-8 -*-
"""run_layout 单测: 目录骨架 + version 归一化 + manifest schema + config 快照。"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from stages.layout import (
    RunLayout,
    normalize_version,
    next_version,
    validate_version_label,
    write_config_snapshot,
    write_train_config_yaml,
    write_manifest,
)


@pytest.mark.parametrize("raw,expected", [
    ("v1", "v1"),
    ("v2", "v2"),
    ("  v3  ", "v3"),
    ("中文 任务/测试", "________"),     # 8 个非 ASCII 字符全归一为 _
    ("", "v1"),
    ("a.b/c", "a.b_c"),
])
def test_normalize_version(raw, expected):
    assert normalize_version(raw) == expected


@pytest.mark.parametrize("raw", [
    None, "", "v1", "v2", "v10", "custom-tag", "custom_tag",
    "20260710", "exp.v1", "exp01", "final",
])
def test_validate_version_label_accepts_pure_version(raw):
    """None / 空 / 纯版本号 / 自定义 tag 全部放行(交给 normalize_version 归一)。"""
    validate_version_label(raw)  # 不抛即通过


@pytest.mark.parametrize("bad", [
    "xgb-v1",       # algo 重复(实际线上出过 lgb-lgb-v1 的 bug)
    "lgb-v1",
    "dnn-v2",
    "lr-v3",
    "tuned-v1",     # suffix 重复(实际线上出过 xgb-tuned-tuned-v1)
    "feat",         # suffix 但缺 v 号(会产出 xgb-feat 目录, 缺版本号)
    "feat-v2",
    "FEAT",         # 大小写不敏感
    "XGB_v1",       # 混合分隔符
    "v1.feat",      # 点分隔也命中
])
def test_validate_version_label_rejects_reserved_tokens(bad):
    """含 algo(xgb/dnn/lr/lgb) 或 suffix(tuned/feat) 保留字 token 一律拒绝, 大小写不敏感。"""
    with pytest.raises(ValueError, match="保留字"):
        validate_version_label(bad)


def test_next_version_empty(tmp_path):
    """无任何已有 run 时返回 v1。"""
    assert next_version(str(tmp_path), "xgb", "") == "v1"
    assert next_version(str(tmp_path), "xgb", "-feat") == "v1"


def test_next_version_increments(tmp_path):
    """已有 xgb-v1 时返回 v2;已有 xgb-feat-v2 时对 -feat 返回 v3,baseline 仍 v2。"""
    (tmp_path / "new-models" / "xgb-v1").mkdir(parents=True)
    assert next_version(str(tmp_path), "xgb", "") == "v2"
    (tmp_path / "new-models" / "xgb-feat-v1").mkdir(parents=True)
    (tmp_path / "new-models" / "xgb-feat-v2").mkdir(parents=True)
    assert next_version(str(tmp_path), "xgb", "-feat") == "v3"
    assert next_version(str(tmp_path), "xgb", "") == "v2"


def test_next_version_ignores_other_algos(tmp_path):
    """不同 algo 的目录互不影响。"""
    (tmp_path / "new-models" / "dnn-v1").mkdir(parents=True)
    (tmp_path / "new-models" / "dnn-v2").mkdir(parents=True)
    assert next_version(str(tmp_path), "xgb", "") == "v1"
    assert next_version(str(tmp_path), "dnn", "") == "v3"


def test_create_run_layout_makes_full_skeleton(tmp_path):
    """RunLayout.create 应一次性 mkdir 所有子目录(含 config/),目录名格式可控。"""
    when = datetime(2026, 6, 15, 17, 0, 0)
    layout = RunLayout.create(str(tmp_path), algo="xgb", suffix="", version="v1", when=when)
    assert layout.run_dir.name == "xgb-v1"
    assert layout.run_dir.parent.name == "new-models"
    # 所有子目录均存在(含 config_dir)
    for d in [layout.features_dir, layout.model_dir, layout.evaluation_dir,
              layout.predictions_dir, layout.explainability_dir, layout.logs_dir,
              layout.config_dir]:
        assert d.is_dir(), f"缺子目录 {d}"
    # config.json 是路径而非目录,初始不存在
    assert not layout.config_json.exists()


def test_write_train_config_yaml_copies(tmp_path):
    """write_train_config_yaml 应把入参 yaml 复制到 config/train_config.yaml + 写 manifest。"""
    layout = RunLayout.create(str(tmp_path), "xgb", suffix="", version="v1",
                               when=datetime(2026, 6, 15))
    src = tmp_path / "src_config.yaml"
    src.write_text("model:\n  algo: xgb\n  features: [a, b]\n", encoding="utf-8")
    dst = write_train_config_yaml(layout, str(src))
    assert dst is not None
    assert dst == layout.config_dir / "train_config.yaml"
    assert dst.read_text(encoding="utf-8").startswith("model:")
    # manifest 落盘 + source_yaml 指向 src
    m = json.loads((layout.config_dir / "_manifest.json").read_text())
    assert m["stage"] == "config"
    assert m["files"][0]["name"] == "train_config.yaml"
    assert str(src.resolve()) == m["source_yaml"]


def test_write_train_config_yaml_skips_when_none(tmp_path):
    """source_yaml_path=None 时跳过,不落 yaml 也不抛错。"""
    layout = RunLayout.create(str(tmp_path), "xgb", suffix="", version="v1",
                               when=datetime(2026, 6, 15))
    assert write_train_config_yaml(layout, None) is None
    assert not (layout.config_dir / "train_config.yaml").exists()


def test_write_train_config_yaml_skips_when_missing(tmp_path):
    """source_yaml_path 指向不存在文件时跳过。"""
    layout = RunLayout.create(str(tmp_path), "xgb", suffix="", version="v1",
                               when=datetime(2026, 6, 15))
    assert write_train_config_yaml(layout, "/nonexistent/path.yaml") is None
    assert not (layout.config_dir / "train_config.yaml").exists()


def test_create_with_suffix(tmp_path):
    """suffix='-feat' 时目录名应为 xgb-feat-v1。"""
    layout = RunLayout.create(str(tmp_path), algo="xgb", suffix="-feat", version="v1")
    assert layout.run_dir.name == "xgb-feat-v1"
    assert layout.suffix == "-feat"
    assert layout.version == "v1"


def test_create_autoincrements_version_when_none(tmp_path):
    """version=None 时自动调 next_version 自增。"""
    (tmp_path / "new-models" / "xgb-v1").mkdir(parents=True)
    layout = RunLayout.create(str(tmp_path), algo="xgb", suffix="", version=None)
    assert layout.run_dir.name == "xgb-v2"


def test_write_manifest_schema(tmp_path):
    """manifest 必含 stage / schema_version / produced_by / created_at / files。"""
    layout = RunLayout.create(str(tmp_path), "xgb", suffix="", version="v1",
                               when=datetime(2026, 6, 15))
    f = layout.model_dir / "model.json"
    f.write_text('{"x": 1}')
    write_manifest(layout.model_dir, stage="model", files=[f],
                   extra={"algo": "xgb", "train_info": {"best_iteration": 42}})
    data = json.loads((layout.model_dir / "_manifest.json").read_text())
    assert data["stage"] == "model"
    assert data["schema_version"] == "1"
    assert data["produced_by"].startswith("skills/")
    assert "created_at" in data
    assert data["files"][0]["name"] == "model.json"
    assert data["files"][0]["size"] > 0
    assert data["algo"] == "xgb" and data["train_info"]["best_iteration"] == 42


def test_write_manifest_dnn_train_info(tmp_path):
    """dnn manifest 的 train_info 应含 best_epoch/total_epochs/early_stopped。"""
    layout = RunLayout.create(str(tmp_path), "dnn", suffix="", version="v1",
                               when=datetime(2026, 6, 15))
    f = layout.model_dir / "model.pkl"
    f.write_bytes(b"\x80")
    write_manifest(layout.model_dir, stage="model", files=[f],
                   extra={"algo": "dnn", "train_info": {
                       "best_epoch": 8, "total_epochs": 12,
                       "early_stopped": True, "best_val_auc": 0.8289,
                   }})
    data = json.loads((layout.model_dir / "_manifest.json").read_text())
    assert data["algo"] == "dnn"
    assert data["train_info"]["best_epoch"] == 8
    assert data["train_info"]["total_epochs"] == 12
    assert data["train_info"]["early_stopped"] is True
    assert data["train_info"]["best_val_auc"] == 0.8289


def test_write_manifest_marks_missing_file(tmp_path):
    """不存在的文件应在 manifest 标 status=missing,不抛错。"""
    layout = RunLayout.create(str(tmp_path), "xgb", suffix="", version="v1",
                               when=datetime(2026, 6, 15))
    ghost = layout.model_dir / "ghost.pkl"  # 不创建
    write_manifest(layout.model_dir, stage="model", files=[ghost])
    data = json.loads((layout.model_dir / "_manifest.json").read_text())
    assert data["files"][0] == {"name": "ghost.pkl", "status": "missing"}


def test_config_snapshot_strips_private_keys(tmp_path):
    """config.json 应剔除 _config_dir 等私有字段,含 input/output/runtime/config。"""
    layout = RunLayout.create(str(tmp_path), "xgb", suffix="", version="v1",
                               when=datetime(2026, 6, 15))
    cfg = {
        "_config_dir": "/tmp/private",
        "model": {"name": "m", "version": "v1"},
    }
    # train_info 整体 merge 进 runtime (xgb 路径: best_iteration)
    write_config_snapshot(layout, cfg, "/tmp/data_dir",
                          extra={"n_features": 10,
                                 "best_iteration": 137})
    data = json.loads(layout.config_json.read_text())
    assert "_config_dir" not in data["config"]
    assert data["config"]["model"]["name"] == "m"
    assert data["input"]["data_dir"].endswith("data_dir")
    assert data["input"]["train_path"].endswith("train.parquet")
    assert data["input"]["test_path"].endswith("test.parquet")
    assert data["input"]["oot_path"].endswith("oot.parquet")
    assert data["runtime"]["best_iteration"] == 137
    assert data["runtime"]["n_features"] == 10
    assert data["run_name"] == layout.run_name
    assert data["version"] == "v1"
    assert data["suffix"] == ""
    assert data["label"] == "v1"  # label 字段保留旧语义
