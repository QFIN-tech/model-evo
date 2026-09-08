# -*- coding: utf-8 -*-
"""session 目录布局的唯一真相。所有脚本经本模块定位产物, 不各自拼路径。

session 根 = runs/{YYYYMMDD-HHMMSS}-{mining_name}/, 布局见 feature-mining-skills/README.md。
"""
from __future__ import annotations

from pathlib import Path

SPLITS = ("train", "test", "oot")


# ---- data ----
def data_dir(session_dir) -> Path:
    return Path(session_dir) / "data"


def sample_parquet(session_dir) -> Path:
    return data_dir(session_dir) / "sample.parquet"


def feature_metadata_csv(session_dir) -> Path:
    return data_dir(session_dir) / "feature-metadata.csv"


def splits_dir(session_dir) -> Path:
    return data_dir(session_dir) / "splits"


def split_parquet(session_dir, split: str) -> Path:
    assert split in SPLITS, "split 必须是 %s 之一" % (SPLITS,)
    return splits_dir(session_dir) / ("%s.parquet" % split)


def split_manifest_json(session_dir) -> Path:
    return splits_dir(session_dir) / "_split_manifest.json"


# ---- champion 物化缓存(大数据量: 已接受特征值只算一次, 见 feature_runtime) ----
def champion_cache_dir(session_dir) -> Path:
    return data_dir(session_dir) / "champion-cache"


def champion_cache_parquet(session_dir, split: str) -> Path:
    assert split in SPLITS, "split 必须是 %s 之一" % (SPLITS,)
    return champion_cache_dir(session_dir) / ("%s.parquet" % split)


def champion_cache_manifest(session_dir) -> Path:
    return champion_cache_dir(session_dir) / "manifest.json"
    return data_dir(session_dir) / "_split_manifest.json"


# ---- profile ----
def profile_dir(session_dir) -> Path:
    return Path(session_dir) / "profile"


def feature_profile_csv(session_dir) -> Path:
    return profile_dir(session_dir) / "feature-profile.csv"


def data_desc_md(session_dir) -> Path:
    return profile_dir(session_dir) / "data-desc.md"


# ---- baseline ----
def baseline_dir(session_dir) -> Path:
    return Path(session_dir) / "baseline"


def baseline_config_yaml(session_dir) -> Path:
    return baseline_dir(session_dir) / "baseline_config.yaml"


def baseline_model_dir(session_dir) -> Path:
    return baseline_dir(session_dir) / "model"


def baseline_predictions_parquet(session_dir, split: str) -> Path:
    assert split in SPLITS
    return baseline_dir(session_dir) / "predictions" / ("%s_predictions.parquet" % split)


def baseline_eval_json(session_dir, split: str) -> Path:
    assert split in SPLITS
    return baseline_dir(session_dir) / "evaluation" / ("baseline_%s_eval.json" % split)


def baseline_eval_md(session_dir, split: str) -> Path:
    assert split in SPLITS
    return baseline_dir(session_dir) / "evaluation" / ("baseline_%s_eval.md" % split)


# ---- semantic (semantic-feature) ----
def semantic_dir(session_dir) -> Path:
    return Path(session_dir) / "semantic"


def semantic_registry_json(session_dir) -> Path:
    return semantic_dir(session_dir) / "registry.json"


def semantic_item_dir(session_dir, name: str) -> Path:
    return semantic_dir(session_dir) / name


# ---- evolution ----
def evolution_dir(session_dir) -> Path:
    return Path(session_dir) / "evolution"


def state_json(session_dir) -> Path:
    return evolution_dir(session_dir) / "state.json"


def ledger_jsonl(session_dir) -> Path:
    return evolution_dir(session_dir) / "ledger.jsonl"


def accepted_dir(session_dir) -> Path:
    return evolution_dir(session_dir) / "accepted"


def accepted_py(session_dir, fid: str) -> Path:
    return accepted_dir(session_dir) / ("%s.py" % fid)


def accepted_meta_json(session_dir, fid: str) -> Path:
    return accepted_dir(session_dir) / ("%s.meta.json" % fid)


def champion_eval_json(session_dir) -> Path:
    return accepted_dir(session_dir) / "_champion_eval.json"


def feedback_dir(session_dir) -> Path:
    return evolution_dir(session_dir) / "feedback"


def latest_feedback_md(session_dir) -> Path:
    return feedback_dir(session_dir) / "latest-feedback.md"


def rounds_dir(session_dir) -> Path:
    return evolution_dir(session_dir) / "rounds"


def round_dir(session_dir, round_no: int) -> Path:
    return rounds_dir(session_dir) / ("r%03d" % int(round_no))


def case_batch_json(session_dir, round_no: int) -> Path:
    return round_dir(session_dir, round_no) / "case-batch.json"


def round_brief_md(session_dir, round_no: int) -> Path:
    return round_dir(session_dir, round_no) / "round-brief.md"


def round_summary_md(session_dir, round_no: int) -> Path:
    return round_dir(session_dir, round_no) / "round-summary.md"


def candidates_dir(session_dir, round_no: int) -> Path:
    return round_dir(session_dir, round_no) / "candidates"


def candidate_py(session_dir, round_no: int, cid: str) -> Path:
    return candidates_dir(session_dir, round_no) / ("%s.py" % cid)


def candidate_meta_json(session_dir, round_no: int, cid: str) -> Path:
    return candidates_dir(session_dir, round_no) / ("%s.meta.json" % cid)


def results_dir(session_dir, round_no: int) -> Path:
    return round_dir(session_dir, round_no) / "results"


def candidate_result_json(session_dir, round_no: int, cid: str) -> Path:
    return results_dir(session_dir, round_no) / ("%s.result.json" % cid)


def export_dir(session_dir) -> Path:
    return evolution_dir(session_dir) / "export" / "accepted-features"


# ---- session 级 ----
def session_manifest_json(session_dir) -> Path:
    return Path(session_dir) / "_manifest.json"


def mining_spec_md(session_dir) -> Path:
    return Path(session_dir) / "mining-spec.md"


def report_md(session_dir) -> Path:
    return Path(session_dir) / "report.md"
