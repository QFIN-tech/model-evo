"""benchmark 对照基线的执行环境工具(退出分类、内存门、训练来源审计)。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "benchmark"))

import _bench_env as B  # noqa: E402


def test_classify_exit():
    assert B.classify_exit(0) == "ok"
    assert B.classify_exit(-9) == "killed"
    assert B.classify_exit(137) == "killed"
    assert B.classify_exit(1) == "code_error"
    assert B.classify_exit(None) == "timeout"


def test_error_tail_drops_warning_noise():
    err = "x.py:1: UserWarning: noise\n  warnings.warn(\nValueError: real\n"
    assert B.error_tail(err) == "ValueError: real"


def test_wait_for_memory(monkeypatch):
    monkeypatch.setattr(B, "free_memory_gb", lambda: 500.0)
    assert B.wait_for_memory(100)["admitted"] is True
    monkeypatch.setattr(B, "free_memory_gb", lambda: 5.0)
    assert B.wait_for_memory(100, max_wait_sec=0)["admitted"] is False


def test_audit_training_source():
    from baseline_aide import LEAK_PATTERNS

    assert B.audit_training_source("y = test[target_col]", LEAK_PATTERNS)["status"] == "review"
    assert B.audit_training_source("a = pd.concat([train, test])", LEAK_PATTERNS)["status"] == "review"
    assert B.audit_training_source("p = m.predict(test)\nm.fit(X, y)", LEAK_PATTERNS)["status"] == "clean"


def test_aide_task_dir_hides_test_labels(tmp_path):
    import pandas as pd

    from baseline_aide import _build_task_dir

    run = tmp_path / "run"
    lab = run / "sample-profile" / "splits" / "labels"
    lab.mkdir(parents=True)
    sp = lab.parent
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(sp / "train.parquet")
    pd.DataFrame({"y": [0, 1, 0]}).to_parquet(lab / "train.parquet")
    pd.DataFrame({"a": [4, 5]}).to_parquet(sp / "validation.parquet")
    pd.DataFrame({"y": [1, 0]}).to_parquet(lab / "validation.parquet")
    work = tmp_path / "task"
    _build_task_dir(run, work, {"target": {"name": "y"}, "primary_metric": "auc"})
    tr = pd.read_csv(work / "train.csv")
    te = pd.read_csv(work / "test.csv")
    assert len(tr) == 5 and list(tr["target"]) == [0, 1, 0, 1, 0]
    assert "target" not in te.columns and len(te) == 2


def test_aide_time_hint_sorts_and_describes(tmp_path):
    import pandas as pd

    from baseline_aide import _build_task_dir

    run = tmp_path / "run"
    lab = run / "sample-profile" / "splits" / "labels"
    lab.mkdir(parents=True)
    sp = lab.parent
    pd.DataFrame({"a": [3, 1, 2]}).to_parquet(sp / "train.parquet")
    pd.DataFrame({"y": [30, 10, 20], "m": ["2011-03", "2011-01", "2011-02"]}).to_parquet(lab / "train.parquet")
    pd.DataFrame({"a": [5, 4]}).to_parquet(sp / "validation.parquet")
    pd.DataFrame({"y": [50, 40], "m": ["2011-05", "2011-04"]}).to_parquet(lab / "validation.parquet")
    work = tmp_path / "task"
    _build_task_dir(run, work, {"target": {"name": "y"}, "primary_metric": "rmse"}, time_column="m")
    tr = pd.read_csv(work / "train.csv")
    assert list(tr["a"]) == [1, 2, 3, 4, 5] and "m" not in tr.columns
    desc = (work / "description.md").read_text(encoding="utf-8")
    assert "自 2011-04 起" in desc and "不要用随机打乱" in desc
