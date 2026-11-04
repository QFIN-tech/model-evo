"""v2.13.3 回归测试:HPO 断点续跑 + smoke 预算闸。

两个已核实缺陷(证据:run example-seq-v3-v2.13.0.1 的 gen-04):

  1. **smoke 预检把整个 HPO 搜索跑了一遍。** ``run_smoke`` 走完整
     ``fit_and_score`` 路径,而模板的 hpo 分支不看 ``MODELEVO_SMOKE_RUN``
     ——实测三个 HPO trial 的 ``[step] smoke ok`` 耗时 xgb 1030~1108s、
     catboost 2100~2124s、lgbm 300s,而真正的 40 config 搜索只要 986s
     (``hpo-study.json.search_seconds``)。smoke 占单次 attempt 的 30~50%,
     且它跑在 1 万行小样本上,对超参优劣本无判别力。

  2. **``hpo-checkpoint.json`` 只写不读。** ``run_hpo_search`` 每评完一个
     config 立即落盘(``_record``),但启动时从不读取——runner 被外部 kill
     后(实测 gen-04 三个 trial 各被杀 3 次,16:53→00:10 共 7.9 小时)
     40 个 config 全部重算,``hpo.py`` docstring 自述用途只是"attempt 账本
     可对账"。

本文件锁定:
  A. ``MODELEVO_SMOKE_RUN=1`` 时 budget 降为 1,且**不写** trial 根目录的
     ``hpo-checkpoint.json`` / ``hpo-study.json``(改写到 smoke 专属文件),
     避免 1 万行小样本的 inner 分数污染全量 study 先验;
  B. checkpoint 存在且空间/预算一致时回放已评 config,只补差额;
     回放不重算(``fit_score_inner`` 调用次数 = budget − replayed);
  C. checkpoint 陈旧(空间变更 / 记载预算大于当前预算)时整份忽略,
     从零搜索;
  D. ``_out_of_budget`` 的均速外推用**本轮**新拟合数当分母,回放不稀释
     单 config 成本估计(否则回放后外推不出"跑不完"而撞墙);
  E. ``n_configs_replayed`` / ``n_configs_fit_this_attempt`` /
     ``smoke_budget_reduced`` 三个字段进返回值,供模板写入 metrics。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402  hermetic bootstrap

sys.path.insert(0, str(SKILL))

from runtime.hpo import (  # noqa: E402
    _space_distribution,
    run_hpo_search,
    write_hpo_artifacts,
)

optuna = pytest.importorskip("optuna")

#: 一个两维小空间(确定性、无 log),够验证采样/回放/等价键
SPACE = {
    "a": {"type": "float", "low": 0.0, "high": 1.0},
    "b": {"type": "int", "low": 1, "high": 5},
}


def _linear_scorer(calls: list):
    """确定性 inner 目标:越接近 (0.3, 3) 越好;记录每次调用的参数。"""
    def _f(params: dict) -> float:
        calls.append(dict(params))
        return -(abs(params["a"] - 0.3) + abs(params["b"] - 3) * 0.1)
    return _f


def _write_ckpt(path: Path, trials: list, *, budget: int,
                space: dict | None = None) -> None:
    path.write_text(json.dumps({
        "n_configs_evaluated": len(trials),
        "elapsed_sec": 12.0,
        "budget": budget,
        "space": SPACE if space is None else space,
        "trials": trials,
    }), encoding="utf-8")


def _distinct(n: int, *, metric=None) -> list:
    """n 条**参数互不相同**的已评轨迹。

    真实 checkpoint 的每条 params 都是采样器独立抽的,不会重复;
    回放按参数逐键去重(防同一 config 被计两次),用重复 fixture
    会测不出真实条数。
    """
    out = []
    for i in range(n):
        v = (-1.0 + i * 0.01) if metric is None else metric
        out.append({"params": {"a": round(0.01 + i * 0.001, 6),
                               "b": 1 + (i % 5)},
                    "inner_metric": v})
    return out


# --------------------------------------------------------------------------
# A. smoke 预算闸
# --------------------------------------------------------------------------

def test_smoke_reduces_budget_to_one(tmp_path, monkeypatch):
    """smoke 轮只评 1 个 config;真实轮评满 budget。"""
    monkeypatch.setenv("MODELEVO_SMOKE_RUN", "1")
    calls: list = []
    res = run_hpo_search(
        SPACE, 40, _linear_scorer(calls),
        metric_direction_up=True, seed=7,
        checkpoint_path=tmp_path / "hpo-checkpoint.json")
    assert res["n_configs_evaluated"] == 1, res
    assert len(calls) == 1, "smoke 轮必须只拟合 1 个 config"
    assert res["smoke_budget_reduced"] is True
    assert res["budget_requested"] == 1


def test_smoke_does_not_pollute_full_checkpoint(tmp_path, monkeypatch):
    """smoke 的 checkpoint 写到独立文件,不碰全量那份。"""
    full = tmp_path / "hpo-checkpoint.json"
    monkeypatch.setenv("MODELEVO_SMOKE_RUN", "1")
    run_hpo_search(SPACE, 40, _linear_scorer([]),
                   metric_direction_up=True, seed=7,
                   checkpoint_path=full)
    assert not full.exists(), "smoke 不得写全量 checkpoint"
    smoke = tmp_path / "hpo-checkpoint-smoke.json"
    assert smoke.exists(), "smoke 的轨迹应落在 smoke 专属文件"


def test_smoke_does_not_write_study_artifact(tmp_path, monkeypatch):
    """smoke 轮不落 hpo-study.json(否则磁盘上留一份 n=1 的伪 study)。"""
    monkeypatch.setenv("MODELEVO_SMOKE_RUN", "1")
    res = run_hpo_search(SPACE, 40, _linear_scorer([]),
                         metric_direction_up=True, seed=7)
    write_hpo_artifacts(tmp_path, res)
    assert not (tmp_path / "hpo-study.json").exists()


def test_full_run_writes_study_artifact(tmp_path, monkeypatch):
    """对照:非 smoke 轮正常落盘。"""
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    res = run_hpo_search(SPACE, 3, _linear_scorer([]),
                         metric_direction_up=True, seed=7)
    write_hpo_artifacts(tmp_path, res)
    study = json.loads((tmp_path / "hpo-study.json").read_text())
    assert study["n_configs_evaluated"] == 3
    assert study["smoke_budget_reduced"] is False


def test_smoke_disables_resume(tmp_path, monkeypatch):
    """smoke 轮即便磁盘上有全量 checkpoint 也不回放(小样本分数不可比)。"""
    full = tmp_path / "hpo-checkpoint.json"
    _write_ckpt(full, _distinct(20, metric=0.99), budget=40)
    monkeypatch.setenv("MODELEVO_SMOKE_RUN", "1")
    calls: list = []
    res = run_hpo_search(SPACE, 40, _linear_scorer(calls),
                         metric_direction_up=True, seed=7,
                         checkpoint_path=full)
    assert res["n_configs_replayed"] == 0
    assert len(calls) == 1


# --------------------------------------------------------------------------
# B. 断点续跑
# --------------------------------------------------------------------------

def test_resume_replays_and_fits_only_remainder(tmp_path, monkeypatch):
    """已评 25/40 → 本轮只拟合 15 个,且 best 取自全部 40 个。"""
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    ck = tmp_path / "hpo-checkpoint.json"
    _write_ckpt(ck, _distinct(25), budget=40)

    calls: list = []
    res = run_hpo_search(SPACE, 40, _linear_scorer(calls),
                         metric_direction_up=True, seed=7,
                         checkpoint_path=ck)
    assert len(calls) == 15, f"应只补差额 15 个,实际 {len(calls)}"
    assert res["n_configs_replayed"] == 25
    assert res["n_configs_fit_this_attempt"] == 15
    assert res["n_configs_evaluated"] == 40
    assert res["budget_truncated_by_deadline"] is False


def test_resume_keeps_best_from_replayed_trials(tmp_path, monkeypatch):
    """回放轨迹里的最优点必须参与 best 选择(否则续跑会丢已搜到的解)。"""
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    ck = tmp_path / "hpo-checkpoint.json"
    _write_ckpt(ck, [{"params": {"a": 0.3, "b": 3},
                      "inner_metric": 0.0}], budget=2)

    def _worse(_p):
        return -100.0

    res = run_hpo_search(SPACE, 2, _worse,
                         metric_direction_up=True, seed=7,
                         checkpoint_path=ck)
    assert res["inner_score"] == 0.0, res
    assert res["best_params"] == {"a": 0.3, "b": 3}


def test_resume_disabled_by_flag(tmp_path, monkeypatch):
    """resume=False 时不回放(显式退出通道)。"""
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    ck = tmp_path / "hpo-checkpoint.json"
    _write_ckpt(ck, _distinct(5, metric=0.0), budget=10)
    calls: list = []
    res = run_hpo_search(SPACE, 10, _linear_scorer(calls),
                         metric_direction_up=True, seed=7,
                         checkpoint_path=ck, resume=False)
    assert res["n_configs_replayed"] == 0
    assert len(calls) == 10


def test_no_checkpoint_runs_full_budget(tmp_path, monkeypatch):
    """checkpoint 不存在 = 首次运行,行为与旧版一致。"""
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    calls: list = []
    res = run_hpo_search(SPACE, 6, _linear_scorer(calls),
                         metric_direction_up=True, seed=7,
                         checkpoint_path=tmp_path / "nope.json")
    assert len(calls) == 6
    assert res["n_configs_replayed"] == 0
    assert res["n_configs_fit_this_attempt"] == 6


# --------------------------------------------------------------------------
# C. 陈旧 checkpoint 判定
# --------------------------------------------------------------------------

def test_stale_space_is_ignored(tmp_path, monkeypatch):
    """空间已变更 → 整份轨迹忽略(旧空间的最优点不是新空间的结果)。"""
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    ck = tmp_path / "hpo-checkpoint.json"
    _write_ckpt(ck, _distinct(5, metric=0.0), budget=10,
                space={"a": {"type": "float", "low": 0.0, "high": 2.0}})
    calls: list = []
    res = run_hpo_search(SPACE, 10, _linear_scorer(calls),
                         metric_direction_up=True, seed=7,
                         checkpoint_path=ck)
    assert res["n_configs_replayed"] == 0
    assert len(calls) == 10


def test_stale_budget_is_ignored(tmp_path, monkeypatch):
    """记载预算大于当前预算 → 忽略(旧 study 的 40 个点填不满新 study)。"""
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    ck = tmp_path / "hpo-checkpoint.json"
    _write_ckpt(ck, _distinct(30, metric=0.0), budget=40)
    calls: list = []
    res = run_hpo_search(SPACE, 10, _linear_scorer(calls),
                         metric_direction_up=True, seed=7,
                         checkpoint_path=ck)
    assert res["n_configs_replayed"] == 0
    assert len(calls) == 10


def test_corrupt_checkpoint_falls_back(tmp_path, monkeypatch):
    """checkpoint 损坏 → 从零搜索,不抛异常。"""
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    ck = tmp_path / "hpo-checkpoint.json"
    ck.write_text("{ not json", encoding="utf-8")
    calls: list = []
    res = run_hpo_search(SPACE, 4, _linear_scorer(calls),
                         metric_direction_up=True, seed=7,
                         checkpoint_path=ck)
    assert len(calls) == 4
    assert res["n_configs_replayed"] == 0


def test_checkpoint_budget_ge_current_is_not_stale(tmp_path, monkeypatch):
    """记载预算 == 当前预算 → 正常回放(等价 study)。"""
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    ck = tmp_path / "hpo-checkpoint.json"
    _write_ckpt(ck, _distinct(4, metric=0.0), budget=6)
    calls: list = []
    res = run_hpo_search(SPACE, 6, _linear_scorer(calls),
                         metric_direction_up=True, seed=7,
                         checkpoint_path=ck)
    assert res["n_configs_replayed"] == 4
    assert len(calls) == 2


# --------------------------------------------------------------------------
# D. deadline 外推不被打折
# --------------------------------------------------------------------------

def test_out_of_budget_uses_new_fits_as_denominator(tmp_path, monkeypatch):
    """回放 30 个后只剩 10 个名额,deadline 已过 → 提前收敛而非硬跑。

    旧实现用 ``len(trials_log)`` 当均速分母:回放 30 个使分母虚高,
    单 config 成本被低估 30 倍,外推不出"跑不完",于是撞 wall 被杀。
    """
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    ck = tmp_path / "hpo-checkpoint.json"
    _write_ckpt(ck, _distinct(30, metric=-1.0), budget=40)

    calls: list = []

    def _slow(p):
        import time as _t
        calls.append(dict(p))
        _t.sleep(0.05)
        return -1.0

    import time as _t
    res = run_hpo_search(SPACE, 40, _slow,
                         metric_direction_up=True, seed=7,
                         deadline_ts=_t.time() + 0.08,
                         checkpoint_path=ck)
    assert res["n_configs_replayed"] == 30
    # deadline 极紧:应远早于补满 10 个就收敛
    assert len(calls) < 10, f"deadline 外推失效,补跑了 {len(calls)} 个"
    assert res["budget_truncated_by_deadline"] is True


# --------------------------------------------------------------------------
# E. 空间 → optuna distribution 同口径(log 必须保留)
# --------------------------------------------------------------------------

def test_space_distribution_matches_suggest_semantics():
    """``log=True`` 的 float 必须是 FloatDistribution(log=True)。

    suggest_float(log=True) 建的就是 FloatDistribution(log=True);
    回放若给 UniformDistribution,optuna 会拒绝 add_trial(值不在
    支撑集),或更糟——静默接受但 TPE 先验被污染。
    """
    d = _space_distribution({"type": "float", "low": 1e-4, "high": 1.0,
                             "log": True})
    assert type(d).__name__ == "FloatDistribution"
    assert d.log is True
    d2 = _space_distribution({"type": "float", "low": 0.0, "high": 1.0})
    assert d2.log is False
    d3 = _space_distribution({"type": "int", "low": 1, "high": 5})
    assert type(d3).__name__ == "IntDistribution"
    d4 = _space_distribution({"type": "categorical",
                              "choices": ["tpe", "random"]})
    assert type(d4).__name__ == "CategoricalDistribution"


def test_replayed_trial_accepted_by_optuna(tmp_path, monkeypatch):
    """回放的参数必须能通过 optuna.add_trial 的支撑集校验。"""
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    space = {"lr": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
             "depth": {"type": "int", "low": 2, "high": 8},
             "algo": {"type": "categorical", "choices": ["a", "b"]}}
    ck = tmp_path / "hpo-checkpoint.json"
    ck.write_text(json.dumps({
        "budget": 4, "space": space,
        "trials": [{"params": {"lr": 0.01, "depth": 4, "algo": "a"},
                    "inner_metric": 0.5}],
    }), encoding="utf-8")
    res = run_hpo_search(space, 4, lambda p: 0.0,
                         metric_direction_up=True, seed=7,
                         checkpoint_path=ck)
    assert res["n_configs_replayed"] == 1, res
    assert res["n_configs_evaluated"] == 4


def test_replayed_best_beats_random(tmp_path, monkeypatch):
    """端到端:回放一个最优点后,补跑不会把它挤掉。"""
    monkeypatch.delenv("MODELEVO_SMOKE_RUN", raising=False)
    ck = tmp_path / "hpo-checkpoint.json"
    _write_ckpt(ck, _distinct(3, metric=0.0), budget=8)
    res = run_hpo_search(SPACE, 8, _linear_scorer([]),
                         metric_direction_up=True, seed=11,
                         checkpoint_path=ck)
    assert res["n_configs_replayed"] == 3
    assert res["n_configs_evaluated"] == 8
    assert res["inner_score"] >= 0.0
