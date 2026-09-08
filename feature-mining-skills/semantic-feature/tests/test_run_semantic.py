# -*- coding: utf-8 -*-
"""run_semantic 端到端: 合成含信号文本 -> hash 嵌入 -> head -> 注册 + state 回写。

hash 后端零重依赖, 本用例保持快跑(不标 slow); 真 embedding 后端见 SKILL.md 冒烟说明。
覆盖: 缺省双 head / present 列与 NaN / PR-AUC 指标 / id 重复防御 / list 模式 / 诊断字段。
"""
import numpy as np
import pandas as pd
import pytest

import run_semantic
from evo_core import paths, state_io


def _make_session(tmp_path, text_mode=None, with_base_score=False, dup_ids=False):
    rng = np.random.RandomState(11)
    n = 600
    y = (rng.randn(n) > 0).astype(int)
    if text_mode == "list":
        # 条目列表文本: 正样本装"催收/借条"类 App, 负样本装"生活/视频"类
        pos_apps = ["借条", "催收助手", "贷超", "分期乐"]
        neg_apps = ["抖音", "微信", "淘宝", "视频"]
        remark = ["^".join(rng.choice(pos_apps, 3)) if yi == 1
                  else "^".join(rng.choice(neg_apps, 3)) for yi in y]
    else:
        remark = np.where(y == 1, "客户逾期 多次催收 拒接电话", "客户正常 按时还款 信用良好")
    df = pd.DataFrame({
        "uid": range(n),
        "dt": np.sort(rng.randint(20260601, 20260630, n)),
        "y": y,
        "f1": rng.randn(n),
        "remark": remark,
    })
    if with_base_score:
        df["base_score"] = 0.5 + 0.3 * y + 0.05 * rng.randn(n)
    if dup_ids:
        # 制造 train 段内部 id 重复(uid=5 改成 uid=0, 两行都落 train)
        df.loc[5, "uid"] = df.loc[0, "uid"]
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
    cfg = {"sample": {"label_col": "y", "dt_col": "dt", "id_cols": ["uid"], "text_cols": ["remark"]}}
    if with_base_score:
        cfg["sample"]["base_model_score_col"] = "base_score"
    state = state_io.init_state(session_dir, "demo_sem", cfg)
    return session_dir, state


def test_run_semantic_full_pipeline(tmp_path):
    session_dir, _ = _make_session(tmp_path)
    rc = run_semantic.main(["--session-dir", str(session_dir)])
    assert rc == 0

    reg = state_io.read_json(paths.semantic_registry_json(session_dir))
    names = [it["name"] for it in reg["items"]]
    assert "sem_remark_lr_score" in names
    assert "sem_remark_pca_lowdim" in names
    assert "sem_remark_present" in names            # 覆盖指示列
    assert reg["id_cols"] == ["uid"]
    item = next(it for it in reg["items"] if it["name"] == "sem_remark_lr_score")
    assert item["output"]["columns"] == ["sem_remark_score"]
    assert item["source_text_cols"] == ["remark"]
    # 指标含 AUC 与 PR-AUC
    m = item["metrics"]["sem_remark_score"]
    assert "oot_auc" in m and "oot_pr_auc" in m
    assert m["oot_auc"] > 0.8                        # 文本信号可分
    # 成本元信息
    assert "encode_seconds" in item["cost"] and "coverage" in item["cost"]

    sem = pd.read_parquet(paths.semantic_dir(session_dir) / "features.parquet")
    assert "sem_remark_score" in sem.columns
    assert "sem_remark_present" in sem.columns
    assert "sem_remark_p1" in sem.columns and "sem_remark_p8" in sem.columns
    assert len(sem) == 600

    # state 回写
    state = state_io.load_state(session_dir)
    assert "sem_remark_score" in state["semantic_columns"]
    assert "sem_remark_present" in state["semantic_columns"]

    # 幂等: 重跑 rc=0(嵌入缓存生效)
    rc = run_semantic.main(["--session-dir", str(session_dir)])
    assert rc == 0


def test_run_semantic_empty_text_nan_and_present(tmp_path):
    """空文本行: present=0 且监督分 NaN(pca 保留几何零向量)。"""
    session_dir, _ = _make_session(tmp_path)
    # 把 oot 一半文本置空
    for split in ("oot",):
        p = paths.split_parquet(session_dir, split)
        df = pd.read_parquet(p)
        df.loc[df.index[: len(df) // 2], "remark"] = ""
        df.to_parquet(p, index=False)
    rc = run_semantic.main(["--session-dir", str(session_dir)])
    assert rc == 0

    sem = pd.read_parquet(paths.semantic_dir(session_dir) / "features.parquet")
    oot_rows = sem[sem["uid"] >= 540]           # oot 段 uid 从 540 起
    empty_mask = oot_rows["sem_remark_present"] == 0
    assert empty_mask.sum() > 0
    assert oot_rows.loc[empty_mask, "sem_remark_score"].isna().all()   # 监督分 NaN
    assert oot_rows.loc[~empty_mask, "sem_remark_score"].notna().all()


def test_run_semantic_pr_auc_metric_present(tmp_path):
    session_dir, _ = _make_session(tmp_path)
    rc = run_semantic.main(["--session-dir", str(session_dir)])
    assert rc == 0
    reg = state_io.read_json(paths.semantic_registry_json(session_dir))
    for it in reg["items"]:
        if it["head"] == "present":
            continue
        for col, m in it["metrics"].items():
            assert "train_pr_auc" in m and "test_pr_auc" in m and "oot_pr_auc" in m


def test_run_semantic_diagnostics_with_base_score(tmp_path):
    """配了 base_model_score_col 时产出条件 AUC 诊断。"""
    session_dir, _ = _make_session(tmp_path, with_base_score=True)
    rc = run_semantic.main(["--session-dir", str(session_dir)])
    assert rc == 0
    reg = state_io.read_json(paths.semantic_registry_json(session_dir))
    lr_item = next(it for it in reg["items"] if it["name"] == "sem_remark_lr_score")
    assert "diagnostics" in lr_item
    d = lr_item["diagnostics"]
    assert "conditional_auc_by_baseline_decile" in d
    assert len(d["conditional_auc_by_baseline_decile"]) > 0
    assert "spearman_vs_base_score" in d


def test_run_semantic_dup_ids_exit2(tmp_path):
    """id_cols 重复 -> 退出码 2 且给出可读报错(缺省 error 模式)。"""
    session_dir, _ = _make_session(tmp_path, dup_ids=True)
    rc = run_semantic.main(["--session-dir", str(session_dir)])
    assert rc == 2


def test_run_semantic_list_mode(tmp_path):
    """list 模式: 条目级编码 + 聚合, 信号可分。"""
    session_dir, _ = _make_session(tmp_path, text_mode="list")
    # 通过 --config 注入 mode=list
    cfg_path = session_dir / "semantic_config.yaml"
    cfg_path.write_text(
        "semantic:\n"
        "  encoder: {backend: hash, dim: 128}\n"
        "  sources:\n"
        "    - name: remark\n      text_cols: [remark]\n      mode: list\n      agg: tfidf\n",
        encoding="utf-8",
    )
    rc = run_semantic.main(["--session-dir", str(session_dir), "--config", str(cfg_path)])
    assert rc == 0
    reg = state_io.read_json(paths.semantic_registry_json(session_dir))
    lr_item = next(it for it in reg["items"] if it["name"] == "sem_remark_lr_score")
    assert lr_item["mode"] == "list"
    assert lr_item["metrics"]["sem_remark_score"]["oot_auc"] > 0.7


def test_run_semantic_no_text_cols_exit2(tmp_path):
    session_dir, _ = _make_session(tmp_path)
    state = state_io.load_state(session_dir)
    state["config_snapshot"]["sample"]["text_cols"] = []
    state_io.save_state(session_dir, state)
    rc = run_semantic.main(["--session-dir", str(session_dir)])
    assert rc == 2
