# -*- coding: utf-8 -*-
"""run_semantic 端到端: 合成含信号文本 -> hash 嵌入 -> 双 head -> 注册 + state 回写。

hash 后端零重依赖, 本用例保持快跑(不标 slow); 真 embedding 后端见 SKILL.md 冒烟说明。
"""
import numpy as np
import pandas as pd
import pytest

import run_semantic
from evo_core import paths, state_io


def _make_session(tmp_path):
    rng = np.random.RandomState(11)
    n = 600
    y = (rng.randn(n) > 0).astype(int)
    # 文本带信号: 正样本含"逾期 催收", 负样本含"正常 还款"
    remark = np.where(y == 1, "客户逾期 多次催收 拒接电话", "客户正常 按时还款 信用良好")
    df = pd.DataFrame({
        "uid": range(n),
        "dt": np.sort(rng.randint(20260601, 20260630, n)),
        "y": y,
        "f1": rng.randn(n),
        "remark": remark,
    })
    session_dir = tmp_path / "runs" / "20260619-120000-demo_sem"
    sdir = paths.splits_dir(session_dir)
    sdir.mkdir(parents=True)
    n_train, n_test = 420, 120
    for name, sub in {
        "train": df.iloc[:n_train],
        "test": df.iloc[n_train:n_train + n_test],
        "oot": df.iloc[n_train + n_test:],
    }.items():
        sub.to_parquet(paths.split_parquet(session_dir, name), index=False)
    state = state_io.init_state(session_dir, "demo_sem", {
        "sample": {"label_col": "y", "dt_col": "dt", "id_cols": ["uid"], "text_cols": ["remark"]},
    })
    return session_dir, state


def test_run_semantic_full_pipeline(tmp_path):
    session_dir, _ = _make_session(tmp_path)

    rc = run_semantic.main(["--session-dir", str(session_dir)])
    assert rc == 0

    # registry + features + 工件
    reg = state_io.read_json(paths.semantic_registry_json(session_dir))
    names = [it["name"] for it in reg["items"]]
    assert names == ["sem_remark_lr_score", "sem_remark_pca_lowdim"]
    assert reg["id_cols"] == ["uid"]
    item = reg["items"][0]
    assert item["output"]["columns"] == ["sem_remark_score"]
    assert item["source_text_cols"] == ["remark"]

    sem = pd.read_parquet(paths.semantic_dir(session_dir) / "features.parquet")
    assert "sem_remark_score" in sem.columns
    assert "sem_remark_p1" in sem.columns and "sem_remark_p8" in sem.columns
    assert len(sem) == 600

    # 文本信号可分: 监督打分 OOT AUC 应显著高于随机
    oot_auc = item["metrics"]["sem_remark_score"]["oot_auc"]
    assert oot_auc > 0.8

    # state 回写
    state = state_io.load_state(session_dir)
    assert "sem_remark_score" in state["semantic_columns"]
    assert "sem_remark_p3" in state["semantic_columns"]

    # 幂等: 重跑 rc=0(嵌入缓存生效), registry 不变
    rc = run_semantic.main(["--session-dir", str(session_dir)])
    assert rc == 0
    assert [it["name"] for it in state_io.read_json(paths.semantic_registry_json(session_dir))["items"]] == names


def test_run_semantic_no_text_cols_exit2(tmp_path):
    session_dir, _ = _make_session(tmp_path)
    state = state_io.load_state(session_dir)
    state["config_snapshot"]["sample"]["text_cols"] = []
    state_io.save_state(session_dir, state)
    rc = run_semantic.main(["--session-dir", str(session_dir)])
    assert rc == 2
