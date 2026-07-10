# -*- coding: utf-8 -*-
"""run_analysis 端到端 smoke test: 造数 -> 跑 -> 看 markdown 产物。"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from run_analysis import run_analysis  # noqa: E402


# 与 _make_tmp_data 对齐的 pday 三档区间 (供 model.split 配置)
_SPLIT_RANGES = {
    "train_range": ["20250101", "20250415"],
    "test_range":  ["20250416", "20250531"],
    "oot_range":   ["20250601", "20250728"],
}


def _make_tmp_data(tmp_path, n=3000):
    """准备测试用 sample.parquet, pday 横跨 2025-01-01 ~ 2025-07-28 供 model.split 三档切分。"""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=n)
    pdays = pd.date_range("20250101", "20250728", periods=n).strftime("%Y%m%d").astype(int)
    df = pd.DataFrame(
        {
            "pday": pdays,
            "user_id": [f"u{i}" for i in range(n)],
            "label": y,
            "fea_signal": y + rng.normal(0, 0.5, size=n),
            "fea_noise": rng.normal(0, 1, size=n),
        }
    )
    data_path = tmp_path / "sample.parquet"
    df.to_parquet(data_path)
    return data_path


def _split_yaml_block():
    return textwrap.dedent(
        """\
          split:
            train_range: ["20250101", "20250415"]
            test_range:  ["20250416", "20250531"]
            oot_range:   ["20250601", "20250728"]
        """
    )


def test_run_analysis_smoke(tmp_path):
    data_path = _make_tmp_data(tmp_path)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """\
            spark:
              app_name: t
              master: local[*]
            model:
              name: smoke
              sample_table: db.t
              dt_col: pday
              label_col: label
              id_cols: [user_id]
              fetch_dt: ["20250101", "20250728"]
              features: [fea_signal, fea_noise]
              split:
                train_range: ["20250101", "20250415"]
                test_range:  ["20250416", "20250531"]
                oot_range:   ["20250601", "20250728"]
            analysis:
              iv:
                n_bins: 5
              psi:
                n_bins: 5
                warn_threshold: 0.1
            """
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    report_path = run_analysis(str(cfg_path), str(data_path), str(out_dir))
    md = Path(report_path).read_text(encoding="utf-8")
    # 关键章节都在
    assert "特征分析报告" in md
    assert "基础统计" in md
    assert "IV" in md
    assert "PSI" in md
    assert "fea_signal" in md

    # 主交付改名为 report.md
    assert Path(report_path).name == "report.md"

    # 语义化合并 csv
    profile = pd.read_csv(out_dir / "feature-profile.csv")
    assert "feature" in profile.columns and "missing_rate" in profile.columns
    quality = pd.read_csv(out_dir / "feature-quality.csv")
    assert "feature" in quality.columns and "iv" in quality.columns
    assert "psi" in quality.columns

    # 保留的细分 csv (model-tuning 契约)
    assert (out_dir / "stats.csv").exists()
    assert (out_dir / "iv_table.csv").exists()
    assert (out_dir / "psi_table.csv").exists()

    # 多 sheet xlsx
    assert (out_dir / "report.xlsx").exists()

    # 内部切分产物落 <session_dir>/sample-features/splits/ (output_dir.parent.parent / "splits")
    splits_dir = out_dir.parent.parent / "splits"
    assert (splits_dir / "train.parquet").exists()
    assert (splits_dir / "test.parquet").exists()
    assert (splits_dir / "oot.parquet").exists()

    # _manifest.json schema + overview 含 split 元信息
    import json
    manifest = json.loads((out_dir / "_manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["produced_by"] == "skills/feature-analysis"
    assert "report.md" in manifest["files"]
    assert "feature-profile.csv" in manifest["files"]
    assert "feature-quality.csv" in manifest["files"]
    assert manifest["overview"]["n_features"] == 2
    assert manifest["overview"]["n_total"] == 3000
    ov = manifest["overview"]
    assert ov["split_strategy"] == "explicit"
    assert ov["sample_counts"]["train"] > 0
    assert ov["sample_counts"]["val"] > 0
    assert ov["sample_counts"]["oot"] > 0


def _base_cfg_text():
    return textwrap.dedent(
        """\
        spark:
          app_name: t
          master: local[*]
        model:
          name: smoke
          sample_table: db.t
          dt_col: pday
          label_col: label
          id_cols: [user_id]
          fetch_dt: ["20250101", "20250728"]
          features: [fea_signal, fea_noise]
          split:
            train_range: ["20250101", "20250415"]
            test_range:  ["20250416", "20250531"]
            oot_range:   ["20250601", "20250728"]
        analysis:
          iv:
            n_bins: 5
          psi:
            n_bins: 5
            warn_threshold: 0.1
        """
    )


def test_run_analysis_no_features_errors(tmp_path):
    """model.features / feature_list_source 都空且未传 --feature_list_source 时报错。"""
    data_path = _make_tmp_data(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(_base_cfg_text().replace(
        "features: [fea_signal, fea_noise]", "features: []"
    ), encoding="utf-8")
    with pytest.raises(ValueError, match="未指定特征清单"):
        run_analysis(str(cfg_path), str(data_path), str(tmp_path / "out"))


def test_run_analysis_with_feature_list_txt(tmp_path):
    """--feature_list_source 指向 .txt 文件, 加载特征并分析。"""
    data_path = _make_tmp_data(tmp_path)
    feat_txt = tmp_path / "my_feats.txt"
    feat_txt.write_text("fea_signal\nfea_noise\n", encoding="utf-8")

    cfg_path = tmp_path / "cfg.yaml"
    cfg = _base_cfg_text().replace(
        "features: [fea_signal, fea_noise]", "features: []"
    )
    cfg_path.write_text(cfg, encoding="utf-8")

    out_dir = tmp_path / "out"
    report_path = run_analysis(
        str(cfg_path), str(data_path), str(out_dir), feature_list_source=str(feat_txt)
    )
    md = Path(report_path).read_text(encoding="utf-8")
    assert "fea_signal" in md
    assert "fea_noise" in md


def test_run_analysis_with_feature_list_source_csv(tmp_path):
    """--feature_list_source 指向 .csv 文件(含 feature_name 列)。"""
    data_path = _make_tmp_data(tmp_path)
    feat_csv = tmp_path / "my_feats.csv"
    feat_csv.write_text("feature_name\nfea_signal\n", encoding="utf-8")

    cfg_path = tmp_path / "cfg.yaml"
    cfg = _base_cfg_text().replace(
        "features: [fea_signal, fea_noise]", "features: []"
    )
    cfg_path.write_text(cfg, encoding="utf-8")

    out_dir = tmp_path / "out"
    report_path = run_analysis(
        str(cfg_path), str(data_path), str(out_dir), feature_list_source=str(feat_csv)
    )
    md = Path(report_path).read_text(encoding="utf-8")
    assert "fea_signal" in md


def test_cross_validate_excludes_missing_features(tmp_path, capsys):
    """交叉校验: 不在数据中的特征自动排除。"""
    data_path = _make_tmp_data(tmp_path)

    fl_csv = tmp_path / "feature-list.csv"
    fl_csv.write_text("feature_name\nfea_signal\nfea_noise\n", encoding="utf-8")

    feat_txt = tmp_path / "my_feats.txt"
    feat_txt.write_text("fea_signal\nunknown_col\nfea_noise\n", encoding="utf-8")

    cfg_path = tmp_path / "cfg.yaml"
    cfg = _base_cfg_text().replace(
        "features: [fea_signal, fea_noise]", "features: []"
    )
    cfg_path.write_text(cfg, encoding="utf-8")

    out_dir = tmp_path / "out"
    report_path = run_analysis(
        str(cfg_path), str(data_path), str(out_dir),
        feature_list_source=str(feat_txt),
        cross_validate_csv=str(fl_csv),
    )
    md = Path(report_path).read_text(encoding="utf-8")
    assert "fea_signal" in md
    assert "fea_noise" in md
    assert "unknown_col" not in md
    captured = capsys.readouterr().out + capsys.readouterr().err
    assert "unknown_col" in captured


def test_cross_validate_all_missing_errors(tmp_path):
    """交叉校验后全部特征都不存在, 报错。"""
    data_path = _make_tmp_data(tmp_path)
    fl_csv = tmp_path / "feature-list.csv"
    fl_csv.write_text("feature_name\na\nb\n", encoding="utf-8")

    feat_txt = tmp_path / "my_feats.txt"
    feat_txt.write_text("unknown_1\nunknown_2\n", encoding="utf-8")

    cfg_path = tmp_path / "cfg.yaml"
    cfg = _base_cfg_text().replace(
        "features: [fea_signal, fea_noise]", "features: []"
    )
    cfg_path.write_text(cfg, encoding="utf-8")

    with pytest.raises(ValueError, match="无有效特征"):
        run_analysis(
            str(cfg_path), str(data_path), str(tmp_path / "out"),
            feature_list_source=str(feat_txt),
            cross_validate_csv=str(fl_csv),
        )


def test_run_analysis_missing_split_errors(tmp_path):
    """model.split 未配置时报错(内部切分必填)。"""
    data_path = _make_tmp_data(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    # 去掉 split 块 (与 _base_cfg_text 缩进对齐)
    cfg = _base_cfg_text().replace(
        '  split:\n'
        '    train_range: ["20250101", "20250415"]\n'
        '    test_range:  ["20250416", "20250531"]\n'
        '    oot_range:   ["20250601", "20250728"]\n',
        ''
    )
    cfg_path.write_text(cfg, encoding="utf-8")

    with pytest.raises(ValueError, match="model.split 缺失"):
        run_analysis(str(cfg_path), str(data_path), str(tmp_path / "out"))
