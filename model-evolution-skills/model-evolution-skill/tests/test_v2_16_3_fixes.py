"""v2.16.3 回归测试:run4 regression(loan_amt_t7)第三批修复。

口径见 docs/MODELEVO_V2_16_3_FIXES_20261023.md。全部离线、不写共享经验库:

  A. D-014 经验库坏条目(context 字符串 / limitations 字符串 / evidence 字符串):
     读侧防御(profile 对账)、存储层归一(all())、写入口(_adapt_shorthand /
     _fill_run_context)、全量重写不丢原文、迁移脚本留痕;
  B. model_meta.retrain_scope 如实描述(单段配方无全量 refit);
  C. --hpo-lr-mode:auto/search/shrink × 空间含/不含 lr(含 clf__ 前缀)矩阵、
     生效值三处留痕(config 创建期 / hpo-study / metrics)、config/ 不被运行期改写;
  D. 回归分桶 Bias = 预测 − 真实;
  E. task-experience.final_recipe 交付期幂等写入;
  F. cap_bound 单一语义(两段式任一段撞即 True)与旧代码副本升级;
  G. timing-report 的 HPO inner 搜索单列 hpo_search(守恒)。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402  hermetic bootstrap

sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL / "assets"))

from runtime.config import Config  # noqa: E402
from runtime.logging_utils import dump_yaml, load_yaml  # noqa: E402

# 坏条目样本:与共享库 principles.jsonl 里 27 条字符串 context 同形态
_BAD_CARD = {
    "experience_id": "exp-202610-bad00001",
    "claim": "seq_token 解析族是本回归任务最强的特征信息通道之一,加宽后 rmse 显著下降",
    "context": "gen-01,父代 trial-002;seq_token_id 原列被机械规则判为实体标识",
    "limitations": "单数据源、单时间窗",
    "evidence": "trial-003 Δ −0.011",
    "category": "feature_engineering",
    "outcome": "positive",
    "action_type": "add",
    "confidence": "high",
    "condition": {"model_type": "binary_classification"},
}


# ===========================================================================
# A. D-014 经验库坏条目
# ===========================================================================

def test_supports_column_tolerates_string_context():
    """读侧防御:字符串 context 不再 AttributeError,且原文参与匹配。"""
    from runtime.profile import _experience_supports_column

    card = {"claim": "加宽有效", "context": "seq_token_id 被误封"}
    assert _experience_supports_column(card, "seq_token_id")      # 原文命中
    assert not _experience_supports_column(card, "user_no")
    # 其它字段的非预期类型同样降级(不崩)
    assert not _experience_supports_column(
        {"claim": "x", "context": 3, "condition": "y", "limitations": "z"},
        "abc_col")
    assert _experience_supports_column(
        {"claim": "x", "limitations": "仅 loan_amt_t30 有效"}, "loan_amt_t30")
    # dict context 的 objective_description 仍参与(旧行为不变)
    assert _experience_supports_column(
        {"claim": "x", "context": {"objective_description": "seq 通道"}},
        "seq_d365")
    assert not _experience_supports_column("not-a-dict", "seq_d365")
    # 归一后的卡({"note": 原文} + context_text)同样命中
    assert _experience_supports_column(
        {"claim": "x", "context": {"note": "seq_token_id 被误封"}},
        "seq_token_id")
    assert _experience_supports_column(
        {"claim": "x", "context": {}, "context_text": "seq_token_id 被误封"},
        "seq_token_id")


# v2.18.0:test_profile_cross_check_with_bad_library_downgrades 已移除 —— 测的是旧 memory/ 全局库的接线(由经验库视图取代),新行为见 tests/test_v2_18_integration.py

# ===========================================================================
# B. retrain_scope 如实描述
# ===========================================================================

def test_retrain_scope_text_single_stage_vs_refit():
    from _mvo.cmd_deliver import _rerun_full_refit, _retrain_scope_text

    single = {"stage1": {"fit_mode": "single", "n_fit_rows": 900}}
    s = _retrain_scope_text(True, "train_tail", 0.1, single)
    assert "fit on earliest 0.9" in s and "latest tail 0.1" in s
    assert s.endswith("no full refit") and "; full refit" not in s
    two = {"es_info": {"refit_trees": 812}}
    assert _rerun_full_refit(two) is True
    assert "then full refit" in _retrain_scope_text(True, "train_tail", .1, two)
    s_none = _retrain_scope_text(True, "train_tail", 0.1, None)
    assert "no full refit" in s_none and "no refit telemetry" in s_none
    assert "full refit on" in _retrain_scope_text(
        True, "train_tail", 0.1, {"full_refit_known": True})
    assert "frozen rounds" in _retrain_scope_text(False, "validation", .1)


def test_rerun_report_carries_fit_mode(tmp_path):
    from _mvo.cmd_deliver import _rerun_champion_recipe

    class _M:
        stage1_ = {"clf_best_iteration": 5, "reg_best_iteration": 4,
                   "n_fit_rows": 90, "n_es_rows": 10, "fit_mode": "single",
                   "es_source": "train_tail"}

    ns = types.SimpleNamespace(TRIAL_DIR=tmp_path, _last_model=None)

    def _fs(X, y, Xv, yv, seed, cfg):
        assert os.environ.get("MODELEVO_ES_SOURCE") == "train_tail"
        ns._last_model = _M()
        return np.zeros(len(Xv)), {"model_num": 2}

    ns.fit_and_score = _fs
    X = pd.DataFrame({"a": range(100)})
    model, rep = _rerun_champion_recipe(ns, X, X.a, X.iloc[:10], X.a[:10], 0,
                                        {}, tmp_path)
    assert rep["stage1"]["fit_mode"] == "single"
    assert rep["full_refit"] is False


def test_finalize_retrain_uses_scope_helper():
    src = (SKILL / "scripts" / "_mvo" / "cmd_deliver.py").read_text(
        encoding="utf-8")
    body = src.split("def cmd_finalize_retrain(")[1].split("\ndef ")[0]
    assert "_retrain_scope_text(" in body
    assert 'f"train+validation; full refit"' not in body


# ===========================================================================
# C. lr_mode 矩阵 + 留痕
# ===========================================================================

_SP_LR = {"max_depth": {"type": "int", "low": 3, "high": 8},
          "learning_rate": {"type": "float", "low": .01, "high": .2}}
_SP_NOLR = {"max_depth": {"type": "int", "low": 3, "high": 8}}
_SP_SEG_LR = {"max_depth": {"type": "int", "low": 3, "high": 8},
              "clf__learning_rate": {"type": "float", "low": .01, "high": .2}}


@pytest.mark.parametrize("mode,space,rows,eff,lr_in_space,warn", [
    ("search", _SP_LR, 10 ** 6, "search", True, False),
    ("search", _SP_NOLR, 10 ** 6, "search", False, False),
    ("shrink", _SP_LR, 10 ** 6, "shrink", False, True),
    ("shrink", _SP_NOLR, 10 ** 6, "shrink", False, True),     # gen-05 形态
    ("shrink", _SP_SEG_LR, 10 ** 6, "shrink", False, True),   # 分段前缀
    ("shrink", _SP_LR, 1000, "shrink", False, True),          # 显式不看行数
    ("auto", _SP_LR, 10 ** 6, "shrink", False, False),
    ("auto", _SP_LR, 1000, "search", True, True),
    ("auto", _SP_NOLR, 10 ** 6, "search", False, True),
    ("auto", _SP_SEG_LR, 10 ** 6, "shrink", False, False),
])
def test_plan_lr_mode_matrix(mode, space, rows, eff, lr_in_space, warn):
    from runtime.hpo import param_base_name, plan_lr_mode

    p = plan_lr_mode(space, {"lr_mode": mode, "search_lr": 0.05,
                             "lr_shrink_factor": 0.5}, rows)
    assert p["mode"] == eff and p["requested_mode"] == mode
    assert any(param_base_name(k) == "learning_rate"
               for k in p["space"]) is lr_in_space
    assert bool(p.get("warn")) is warn
    if mode != "search":
        assert p["note"], p          # 请求 auto/shrink 一律说明裁定理由
    if eff == "shrink":
        assert p["lr_key"] == "learning_rate"
        assert p["refit_lr"] == pytest.approx(0.025)
        assert p["refit_cap_scale"] == pytest.approx(2.0)


def test_record_lr_mode_touches_study_and_metrics_not_config(tmp_path):
    from runtime.hpo import plan_lr_mode, record_lr_mode

    (tmp_path / "config").mkdir()
    dump_yaml(tmp_path / "config" / "trial.yaml",
              {"hpo": {"lr_mode": "shrink"}})
    cfg_sha = hashlib.sha256(
        (tmp_path / "config" / "trial.yaml").read_bytes()).hexdigest()
    (tmp_path / "hpo-study.json").write_text(json.dumps({"lr_mode": "x"}))
    (tmp_path / "metrics.json").write_text(json.dumps({"n": 1}))
    p = plan_lr_mode(_SP_NOLR, {"lr_mode": "shrink"}, 10 ** 6)
    record_lr_mode(tmp_path, p)
    record_lr_mode(tmp_path, p)                                    # 幂等
    st = json.loads((tmp_path / "hpo-study.json").read_text())
    assert st["lr_mode_effective"] == "shrink"
    assert st["lr_mode_requested"] == "shrink" and st["lr_mode_note"]
    m = json.loads((tmp_path / "metrics.json").read_text())
    assert m["hpo_lr_mode_effective"] == "shrink" and m["n"] == 1
    # config/ 在 run 开始时被 source-manifest 冻结:运行期绝不改写
    assert hashlib.sha256((tmp_path / "config" / "trial.yaml").read_bytes()
                          ).hexdigest() == cfg_sha


def test_lr_mode_preview_at_creation(tmp_path):
    from _mvo.render import _hpo_lr_mode_preview

    run = tmp_path / "run"
    (run / "trials" / "p1").mkdir(parents=True)
    (run / "trials" / "p1" / "metrics.json").write_text(
        json.dumps({"n_train": 2_538_086}))
    rc = Config.load(skill_config=False)
    # gen-05 trial-022 形态:显式 shrink、空间无 lr、sample_rows 400k
    out = _hpo_lr_mode_preview(run, ["p1"], {
        "search_space": _SP_NOLR, "lr_mode": "shrink", "sample_rows": 400000,
        "search_lr": 0.05, "lr_shrink_factor": 0.5}, rc)
    assert out["fields"]["lr_mode_effective"] == "shrink"
    assert out["fields"]["lr_mode_requested"] == "shrink"
    assert out["warn"] and "0.025" in out["message"]
    # hurdle_lr 别名按 lr 维处理
    out2 = _hpo_lr_mode_preview(run, ["p1"], {
        "search_space": {"hurdle_lr": {"type": "float", "low": .01,
                                       "high": .1}},
        "lr_mode": "auto"}, rc)
    assert out2["fields"]["lr_mode_effective"] == "shrink"
    # auto + 无父代行数:创建期无法裁定,如实记 None
    out3 = _hpo_lr_mode_preview(run, [], {"search_space": _SP_LR,
                                          "lr_mode": "auto"}, rc)
    assert out3["fields"]["lr_mode_effective"] is None and not out3["warn"]


def test_inject_hpo_config_records_lr_mode(tmp_path):
    """创建期:trial.yaml(config)写生效 lr 模式,且 WARNING 说明原因。"""
    pytest.importorskip("xgboost")
    from test_v2_16_2_fixes import (_FakeTrial, _mk_run, _mk_trial_dir,
                                    _parent_code)
    from _mvo import render as R

    run = _mk_run(tmp_path)
    pdir = run / "trials" / "trial-008-g02-confirm-xgboost-1"
    pdir.mkdir(parents=True)
    (pdir / "metrics.json").write_text(json.dumps({"n_train": 2_538_086}))
    d = _mk_trial_dir(run, "trial-022-g05-hpo-xgboost-1", _parent_code())
    sp = tmp_path / "space.yaml"
    dump_yaml(sp, _SP_NOLR)
    t = _FakeTrial(d)
    import logging

    seen: list = []

    class _H(logging.Handler):
        def emit(self, rec):
            seen.append((rec.levelname, rec.getMessage()))

    lg = logging.getLogger("modelevo.cli")      # harness logger 不向 root 传播
    h = _H()
    lg.addHandler(h)
    try:
        R._inject_hpo_config(run, t, str(sp), 40, sample_rows=400000,
                             hpo_opts={"lr_mode": "shrink",
                                       "lr_shrink_factor": 0.5})
    finally:
        lg.removeHandler(h)
    cfg = load_yaml(d / "config" / "trial.yaml")
    assert cfg["hpo"]["lr_mode"] == "shrink"                  # 原始请求保留
    assert cfg["hpo"]["lr_mode_effective"] == "shrink"
    assert cfg["hpo"]["lr_mode_note"]
    assert t.data["execution"]["hpo_lr_mode_note"]
    assert any(lv == "WARNING" and "hpo lr_mode" in msg
               and "不再沿用父代 lr" in msg for lv, msg in seen), seen


def test_hurdle_hpo_shrink_without_lr_dim_runs_shrink(tmp_path):
    """运行期:gen-05 形态(显式 shrink、空间无 lr)真正按 shrink 执行并留痕。"""
    pytest.importorskip("xgboost")
    from test_v2_16_2_fixes import (_data, _load, _mk_run, _mk_trial_dir,
                                    _parent_code)
    from _mvo import render as R

    run = _mk_run(tmp_path)
    d = _mk_trial_dir(run, "trial-022-g05-hpo-xgboost-1", _parent_code())
    R._inject_hurdle_hp_segment(d / "code" / "train.py")
    m = _load(d / "code" / "train.py")
    X_tr, y_tr, X_va, y_va = _data()
    space = {"max_depth": {"type": "int", "low": 2, "high": 4}}
    cfg = {"hyperparameters": {}, "seed": 7, "search_action": "hpo",
           "hpo": {"search_space": space, "budget_configs": 3,
                   "inner_fraction": 0.2, "lr_mode": "shrink",
                   "search_lr": 0.2, "lr_shrink_factor": 0.5,
                   "parallel_configs": 1, "topology": "structural:hurdle",
                   "enqueue": []}}
    _, mx = m.fit_and_score(X_tr, y_tr, X_va, y_va, 7, cfg)
    assert mx["hpo_lr_mode"] == mx["hpo_lr_mode_effective"] == "shrink"
    assert mx["hpo_lr_mode_requested"] == "shrink" and mx["hpo_lr_mode_note"]
    assert mx["best_params"]["learning_rate"] == pytest.approx(0.1)
    st = json.loads((d / "hpo-study.json").read_text(encoding="utf-8"))
    assert st["lr_mode"] == "shrink" and st["lr_mode_requested"] == "shrink"
    assert all("learning_rate" not in t["params"] for t in st["trials"])


def test_runner_records_effective_lr_mode_in_trial_record():
    src = (SKILL / "runtime" / "runner.py").read_text(encoding="utf-8")
    assert 'trial.data["hpo"]' in src and '"lr_mode_effective"' in src


# ===========================================================================
# D. 分桶 Bias = 预测 − 真实
# ===========================================================================

def test_regression_segment_bias_is_pred_minus_true(tmp_path):
    from runtime.segment_report import (build_segment_report,
                                        regression_segment_table)
    from runtime.spec import TaskSpec

    rng = np.random.default_rng(0)
    y = rng.gamma(2.0, 2.0, 2000)
    p = y + 1.0                                    # 统一高估 1
    df = regression_segment_table(y, p, 10)
    assert np.allclose(df["Bias"], 1.0)            # 正 = 高估
    assert np.allclose(df["MAE"], 1.0) and np.allclose(df["RMSE"], 1.0)
    assert np.allclose(df["预测平均值"] - df["真实均值"], df["Bias"])
    # 与 evaluation.raw_scale.mean_bias(预测 − 真实)同号
    assert np.sign(df["Bias"].mean()) == np.sign(float((p - y).mean()))
    spec = TaskSpec(data={"task_id": "r", "model_type": "regression",
                          "target": {"name": "y"}, "primary_metric": "rmse"})
    out = build_segment_report(y, p, spec, out_dir=tmp_path, y_pred=p)
    assert out is not None and "Bias" in out[0].columns


# ===========================================================================
# E. final_recipe
# ===========================================================================

def _meta():
    return {"best_trial": "trial-008-g02-confirm-xgboost-350",
            "algorithm": "xgboost", "refit_mode": "rerun_recipe",
            "retrain_scope": "train+validation; ...; no full refit",
            "n_features": 2180, "n_train_rows": 2538086,
            "n_validation_rows": 446924, "n_train_validation_rows": 2985010,
            "final_model_sha256": "ab" * 32, "delivery_status": "confirmed",
            "estimator_class": "champ_train.HurdleXGB",
            "hyperparameters": {"n_estimators": None}, "seed": 7,
            "data_view": "v1"}


def test_final_recipe_summary_and_idempotent_write(tmp_path):
    from _mvo.cmd_deliver import final_recipe_summary
    from runtime.memory import TaskExperience

    champ = tmp_path / "trials" / "trial-008-g02-confirm-xgboost-350"
    champ.mkdir(parents=True)
    (champ / "metrics.json").write_text(json.dumps({
        "best_params": {"max_depth": 5, "learning_rate": 0.025,
                        "clf__reg_lambda": 47.2},
        "hurdle_hp": {"effective": {"clf": {"max_depth": 5},
                                    "reg": {"max_depth": 5}}}}))
    r = final_recipe_summary(_meta(), champ)
    assert r["champion_trial_id"] == "trial-008-g02-confirm-xgboost-350"
    assert r["refit_mode"] == "rerun_recipe" and r["n_features"] == 2180
    assert r["hyperparameters"]["learning_rate"] == 0.025
    assert r["hyperparameters_source"] == "metrics.best_params"
    assert r["segments"]["clf"]["max_depth"] == 5
    assert (r["n_train_rows"], r["n_validation_rows"]) == (2538086, 446924)
    assert r["model_sha256"] == "ab" * 32

    run = tmp_path / "run"
    te = TaskExperience(run)
    te.save({"task_id": "x", "final_recipe": None, "key_findings": ["k"]})
    w1 = te.record_final_recipe(r)
    txt1 = te.path.read_text(encoding="utf-8")
    w2 = te.record_final_recipe(dict(r))           # 同哈希同口径:不动
    assert w1 == w2 and te.path.read_text(encoding="utf-8") == txt1
    data = te.load()
    assert data["final_recipe"]["model_sha256"] == "ab" * 32
    assert data["key_findings"] == ["k"]           # 其它字段不动
    # 换了交付模型 → 更新
    te.record_final_recipe(dict(r, model_sha256="cd" * 32))
    assert te.load()["final_recipe"]["model_sha256"] == "cd" * 32


def test_final_recipe_wired_into_delivery_commands():
    src = (SKILL / "scripts" / "_mvo" / "cmd_deliver.py").read_text(
        encoding="utf-8")
    fr = src.split("def cmd_finalize_retrain(")[1].split("\ndef ")[0]
    ev = src.split("def cmd_evaluate_final(")[1].split("\ndef ")[0]
    assert "_record_final_recipe(run_dir, meta, champ_dir)" in fr
    # 写在未确认重训分支(return 1)之后,只对成功交付写
    assert fr.index("_record_final_recipe(") > fr.index("if unconfirmed_mode:\n"
                                                        "        _write_unconfirmed")
    assert "_record_final_recipe(" in ev


# ===========================================================================
# F. cap_bound 单一语义
# ===========================================================================

def test_cap_bound_of_rule():
    from runtime.hpo import cap_bound_of

    assert cap_bound_of(rounds=8000, cap=8000) is True
    assert cap_bound_of(rounds=7973, cap=8000) is False
    assert cap_bound_of(best_iteration=7923, es_rounds=50, cap=8000) is False
    assert cap_bound_of(best_iteration=7960, es_rounds=50, cap=8000) is True
    assert cap_bound_of(rounds=10, cap=None) is None
    assert cap_bound_of(best_iteration=10, cap=100) is None    # 信息不足


def test_unified_cap_bound_trial022_shape():
    """trial-022:es_info(旧 0.95 口径)=True,而两段实训轮数都 < cap。"""
    from runtime.hpo import unified_cap_bound

    m = {"best_iteration": 7923,
         "es_info": {"cap": 8000, "cap_bound": True, "stopped_by": "cap",
                     "source": "baseline_best_iteration_ratio"},
         "hurdle": {"clf_best_iteration": 7923, "clf_cap_bound": True,
                    "cap": 8000, "es_rounds": 50,
                    "fit_stats": {"clf_best_iteration": 7923,
                                  "reg_best_iteration": 5387,
                                  "clf_rounds": 7973, "reg_rounds": 5437}},
         "hpo_refit_es_info": {"clf_best_iteration": 7923, "clf_rounds": 7973,
                               "clf_cap": 8000, "reg_best_iteration": 5387,
                               "reg_rounds": 5437, "reg_cap": 8000}}
    u = unified_cap_bound(m)
    assert u["cap_bound"] is False and u["stopped_by"] == "early_stop"
    assert u["near_cap"] is True                   # 7923 ≥ 0.95×8000:只是提示
    assert u["source"] == "hpo_refit_es_info"
    # 任一子模型撞即 True
    m["hpo_refit_es_info"]["reg_rounds"] = 8000
    assert unified_cap_bound(m)["cap_bound"] is True
    # 没有分段遥测时读 hurdle.fit_stats
    m.pop("hpo_refit_es_info")
    assert unified_cap_bound(m)["source"] == "hurdle.fit_stats"
    assert unified_cap_bound(m)["cap_bound"] is False


def test_unified_cap_bound_single_model_and_legacy():
    from runtime.hpo import assess_hpo_study, unified_cap_bound

    single = {"es_info": {"stage1_rounds": 2000, "stage1_best_iteration": 1951,
                          "cap": 2000, "cap_bound": True}}
    assert unified_cap_bound(single)["cap_bound"] is True
    legacy = {"es_info": {"cap_bound": True}}
    u = unified_cap_bound(legacy)
    assert u["cap_bound"] is True and u["source"].startswith("legacy_flag")
    assert unified_cap_bound({})["cap_bound"] is None
    # baseline 血缘:n_estimators_used(早停面曲线长度)即实训轮数
    base = {"best_iteration": 1976, "n_estimators_used": 2000,
            "es_info": {"cap": 2000}}
    assert unified_cap_bound(base)["cap_bound"] is True
    assert unified_cap_bound(dict(base, n_estimators_used=1990))[
        "cap_bound"] is False
    assert callable(assess_hpo_study)


def test_report_limitation_uses_unified_cap_bound(tmp_path):
    """v2.17.0 报告「模型局限」的撞 cap 一条与单一语义一致。"""
    from runtime.report import _lim_cap_bound

    td = tmp_path / "trials" / "t22"
    td.mkdir(parents=True)
    (td / "metrics.json").write_text(json.dumps({
        "es_info": {"cap": 8000, "cap_bound": True, "stopped_by": "cap",
                    "source": "baseline_best_iteration_ratio"},
        "hurdle": {"clf_best_iteration": 7923, "clf_cap_bound": True,
                   "cap": 8000,
                   "fit_stats": {"clf_best_iteration": 7923,
                                 "reg_best_iteration": 5387,
                                 "clf_rounds": 7973, "reg_rounds": 5437}}}))
    lines = _lim_cap_bound(tmp_path, "t22")
    assert lines and "撞树数上限" not in lines[0] and "贴近" in lines[0]
    m = json.loads((td / "metrics.json").read_text())
    m["hurdle"]["fit_stats"]["clf_rounds"] = 8000
    (td / "metrics.json").write_text(json.dumps(m))
    lines = _lim_cap_bound(tmp_path, "t22")
    assert lines and "撞树数上限" in lines[0] and "clf 实训 8000" in lines[0]


def test_assess_hpo_study_uses_unified_cap_bound(tmp_path):
    from runtime.hpo import assess_hpo_study

    (tmp_path / "hpo-study.json").write_text(json.dumps(
        {"n_configs_evaluated": 40, "diagnostics": {}}))
    (tmp_path / "metrics.json").write_text(json.dumps({
        "es_info": {"cap": 8000, "cap_bound": True,
                    "source": "baseline_best_iteration_ratio"},
        "hpo_refit_es_info": {"clf_rounds": 7973, "clf_cap": 8000,
                              "reg_rounds": 5437, "reg_cap": 8000}}))
    q = assess_hpo_study(tmp_path, default_budget=40)
    assert q["cap_bound"] is False and q["complete"] is True


def test_es_cap_telemetry_reads_composite_stage1():
    from runtime.hpo import es_cap_telemetry

    hurdle = types.SimpleNamespace(stage1_={
        "clf_rounds": 7973, "reg_rounds": 5437, "clf_best_iteration": 7923,
        "reg_best_iteration": 5387})
    t = es_cap_telemetry(hurdle, cap=8000, es_rounds=50, best_iteration=7923)
    assert t["cap_bound"] is False and t["stopped_by"] == "early_stop"
    single = types.SimpleNamespace()
    t2 = es_cap_telemetry(single, cap=2000, es_rounds=50, best_iteration=1976,
                          rounds=2000)
    assert t2["cap_bound"] is True


def test_baseline_template_and_hurdle_segment_use_single_semantics():
    from runtime.baseline import BASELINE_TEMPLATE
    from _mvo.templates import HURDLE_HP_SEGMENT

    assert "es_cap_telemetry" in BASELINE_TEMPLATE
    assert "0.95 * int(N_ESTIMATORS_CAP)" not in BASELINE_TEMPLATE
    assert "cap_bound_of" in HURDLE_HP_SEGMENT


def test_upgrade_cap_bound_code_on_child_copy(tmp_path):
    from _mvo.render import _apply_es_source_upgrade, _upgrade_cap_bound_code

    old = (
        "import json\n"
        "N_ESTIMATORS_CAP = 8000\n"
        "EARLY_STOP_ROUNDS = 50\n"
        "def main(model, metrics):\n"
        "    try:\n"
        '        _bi = metrics.get("best_iteration")\n'
        "        if _bi and N_ESTIMATORS_CAP:\n"
        "            _bound = int(_bi) >= 0.95 * int(N_ESTIMATORS_CAP)\n"
        '            metrics["es_info"] = {\n'
        '                "cap": int(N_ESTIMATORS_CAP), "cap_bound": bool(_bound),\n'
        '                "stopped_by": "cap" if _bound else "early_stop",\n'
        '                "source": "baseline_best_iteration_ratio"}\n'
        "    except Exception:  # noqa: BLE001\n"
        "        pass\n"
        "    return metrics\n")
    new, notes = _upgrade_cap_bound_code(old)
    assert notes and "0.95 * int" not in new
    assert '"source": "unified_cap_bound"' in new
    assert _upgrade_cap_bound_code(new) == (new, [])            # 幂等
    ns: dict = {}
    exec(compile(new, "child", "exec"), ns)
    hurdle = types.SimpleNamespace(stage1_={"clf_rounds": 7973,
                                            "reg_rounds": 5437})
    out = ns["main"](hurdle, {"best_iteration": 7923})
    assert out["es_info"]["cap_bound"] is False                 # 旧口径为 True
    # 副本升级入口串联
    p = tmp_path / "train.py"
    p.write_text(old, encoding="utf-8")
    assert any("cap_bound" in n for n in _apply_es_source_upgrade(p))


# ===========================================================================
# G. timing-report:hpo_search 单列
# ===========================================================================

_STDOUT = """[step] load_data start (10.0s)
[step] load_data done (20.0s)
[step] train start: algo=xgboost-hurdle (30.0s)
[step] hurdle hpo search start: configs=40 (31.0s)
[step] hurdle clf fit start: rows=360000 (31.1s)
[step] hurdle clf fit start: rows=360000 (31.1s)
[step] hurdle clf fit done: best_iter=1468 (300.0s)
[step] hurdle clf fit done: best_iter=1546 (299.5s)
[step] hurdle reg fit done: best_iter=860 (400.0s)
[step] hurdle hpo search done: n=40 (1000.0s)
[step] hurdle 超参生效(hpo_refit): clf={} (1000.0s)
[step] hurdle hpo refit done: clf_best=7923 (1500.0s)
[step] train done (1500.0s)
[step] score & dump done (1510.0s)
"""


def test_step_segments_split_hpo_search_and_conserve():
    from _mvo.cmd_speed import _step_segments

    cats, hpo = _step_segments(_STDOUT, run_sec=1520.0)
    assert hpo == pytest.approx(1000.0 - 31.0)        # 搜索窗口
    # fit = outer 重训 500 s + "train start" 段 10 s(旧分类不变);inner 搜索
    # 里的 "hurdle clf fit ..." 行不再计入 fit
    assert cats["fit"] == pytest.approx(510.0)
    assert sum(cats.values()) + hpo == pytest.approx(1520.0)   # 守恒
    # 交错的回退秒数(299.5 < 300.0)不重复计时
    assert cats["tail"] == pytest.approx(10.0)
    # 中断的搜索:窗口延续到末尾,tail 归搜索
    cut = _STDOUT.split("[step] hurdle hpo search done")[0]
    cats2, hpo2 = _step_segments(cut, run_sec=500.0)
    assert hpo2 == pytest.approx(500.0 - 31.0) and "tail" not in cats2


def test_timing_report_json_and_print(tmp_path, capsys):
    from _mvo.cmd_speed import cmd_timing_report, trial_timing

    run = tmp_path / "run"
    td = run / "trials" / "trial-022-g05-hpo-xgboost-1627"
    td.mkdir(parents=True)
    (td / "stdout.log").write_text(_STDOUT, encoding="utf-8")
    (td / "trial.log").write_text(
        "2026-10-23 10:00:00 [runner.py:1][run] INFO: 开始执行\n"
        "2026-10-23 10:25:20 [runner.py:2][run] INFO: 训练完成: "
        "runtime=1520.0s\n"
        "2026-10-23 10:30:20 [runner.py:3][run] INFO: 评估完成\n",
        encoding="utf-8")
    r = trial_timing(td)
    assert r["hpo_search_sec"] == pytest.approx(969.0)
    assert sum(r["train_phases"].values()) + r["hpo_search_sec"] == \
        pytest.approx(r["run_sec"], abs=0.5)
    assert cmd_timing_report(types.SimpleNamespace(run=str(run),
                                                   json=False)) == 0
    printed = capsys.readouterr().out
    assert "hpo_search" in printed and "hpo_search=   969s" in printed
    rep = json.loads((run / "logs" / "timing-report.json").read_text(
        encoding="utf-8"))
    assert rep["phase_hours"]["hpo_search"] == pytest.approx(969 / 3600,
                                                             abs=0.01)
    assert rep["phase_hours"]["train:fit"] == pytest.approx(510 / 3600,
                                                            abs=0.01)
    assert rep["phase_hours"]["eval"] == pytest.approx(300 / 3600, abs=0.01)
