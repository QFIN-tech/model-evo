"""回归测试:v2.15.8 经验闭环(指纹 / 检索 / 回填 / 冲突 / 导出 / 渲染 / 迁移)。

事故来源(2026-10-09 ~ 10-13,run4-v2.15.0.memory-refactor* 系列):
    - 同一份数据算出 bc5329dc / 7580b665 两个任务指纹(run_dir='.' 不
      resolve,退回 splits 列 hash),热启动 / Layer 0 / 检索 +10 静默失效;
    - 检索 do_not_retry 恒前置:45 次检索里 41 次是同样 5 张"避免"卡,
      建议方向排第 46、top-trial 条目排 108 之后;
    - 避免类卡片被 accepted 计成"成功"(seq-channel-diminishing 后验 0.875);
    - 冲突检测按空格分词(中文无效)、superseded 不落盘;
    - 原则源缺失时导出整体断路(top-trial 榜单也不更新);
    - Warm Baselines 表读错字段,恒显示 —。
见 docs/MODELEVO_V2_15_8_MEMORY_LOOP_FIX_20261013.md。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[1]
for _p in (str(SKILL), str(SKILL / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from runtime.spec import TaskSpec  # noqa: E402

ATID = "binary-y-200w_1000w-abcd1234"


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------

QUERY = {"model_type": "binary_classification",
         "business_domains": ["user_operation"]}


def _spec(**over) -> TaskSpec:
    data = {"task_id": "demo", "task_name": "demo", "model_type":
            "binary_classification", "target": {"name": "y"},
            "primary_metric": "auc"}
    data.update(over)
    return TaskSpec(data=data)


def _run_dir(tmp_path: Path, *, raw_hash="abcd1234", rows=4_000_000,
             run_state: dict | None = None) -> Path:
    run = tmp_path / "run"
    (run / "sample-profile").mkdir(parents=True)
    prof = {"n_rows": rows}
    if raw_hash:
        prof["raw_columns_hash"] = raw_hash
    (run / "sample-profile" / "sample-profile.json").write_text(
        json.dumps(prof), encoding="utf-8")
    (run / "run-state.json").write_text(json.dumps(run_state or {}),
                                        encoding="utf-8")
    return run


# --------------------------------------------------------------------------
# 1. 任务指纹单源化
# --------------------------------------------------------------------------

def test_atid_dot_run_dir_equals_resolved(tmp_path, monkeypatch):
    """在 run 目录内以 Path('.') 调用,与绝对路径结果一致(7580b665 事故)。"""
    run = _run_dir(tmp_path)
    spec = _spec()
    monkeypatch.chdir(run)
    assert spec.resolve_abstract_task_id(Path("."), persist=False) == ATID
    assert spec.resolve_abstract_task_id(run, persist=False) == ATID


def test_atid_prefers_raw_columns_hash_over_splits(tmp_path):
    pd = pytest.importorskip("pandas")
    run = _run_dir(tmp_path)
    (run / "sample-profile" / "splits").mkdir()
    pd.DataFrame({"a": [1], "b": [2]}).to_parquet(
        run / "sample-profile" / "splits" / "train.parquet")
    assert _spec().resolve_abstract_task_id(run).endswith("-abcd1234")


def test_atid_splits_fallback_not_persisted(tmp_path):
    pd = pytest.importorskip("pandas")
    run = _run_dir(tmp_path, raw_hash=None)
    (run / "sample-profile" / "splits").mkdir()
    pd.DataFrame({"a": [1], "b": [2]}).to_parquet(
        run / "sample-profile" / "splits" / "train.parquet")
    atid = _spec().resolve_abstract_task_id(run)
    assert "pending" not in atid and not atid.endswith("abcd1234")
    st = json.loads((run / "run-state.json").read_text())
    assert "abstract_task_id" not in st, "splits 兜底不得持久化"


def test_atid_trusted_value_persisted_and_wins(tmp_path):
    run = _run_dir(tmp_path)
    spec = _spec()
    assert spec.resolve_abstract_task_id(run) == ATID
    st = json.loads((run / "run-state.json").read_text())
    assert st["abstract_task_id"] == ATID
    assert st["abstract_task_id_source"] == "raw_columns_hash"
    # run-state 的可信值优先于现场推导(即便 profile 之后被改)
    (run / "sample-profile" / "sample-profile.json").write_text(
        json.dumps({"n_rows": 10, "raw_columns_hash": "zzzzzzzz"}))
    assert spec.resolve_abstract_task_id(run) == ATID


def test_derive_is_read_only(tmp_path):
    run = _run_dir(tmp_path)
    _spec().derive_abstract_task_id(run)
    assert "abstract_task_id" not in json.loads(
        (run / "run-state.json").read_text())


# --------------------------------------------------------------------------
# 2. 分桶配额检索
# --------------------------------------------------------------------------

def test_memory_metadata_emits_modalities():
    spec = _spec(features={"reparse": ["seq_d365"]},
                 split_policy={"type": "time_based"})
    meta = spec.memory_metadata({"feature_count": 1500, "total_rows": 1})
    assert meta["data_modalities"] == [
        "event_sequence", "wide_table_static", "temporal_features",
        "time_based_split"]
    assert _spec(data_modalities=["x"]).memory_metadata()["data_modalities"] \
        == ["x"]


# --------------------------------------------------------------------------
# 3. 回填语义
# --------------------------------------------------------------------------

# v2.18.0:test_warm_feedback_uses_vs_baseline_ci 已移除 —— 测的是旧 memory/ 全局库的接线(由经验库视图取代),新行为见 tests/test_v2_18_integration.py

# v2.18.0:test_feedback_reports_shown_never_bound 已移除 —— 测的是旧 memory/ 全局库的接线(由经验库视图取代),新行为见 tests/test_v2_18_integration.py

# --------------------------------------------------------------------------
# 5. 渲染 / Profile 对账
# --------------------------------------------------------------------------

# v2.18.0:test_render_buckets_and_warm_baseline_fields 已移除 —— 测的是旧 memory/ 全局库的接线(由经验库视图取代),新行为见 tests/test_v2_18_integration.py

# v2.18.0:test_cross_check_uses_claim_and_ignores_avoid_cards 已移除 —— 测的是旧 memory/ 全局库的接线(由经验库视图取代),新行为见 tests/test_v2_18_integration.py

