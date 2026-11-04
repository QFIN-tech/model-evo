"""v2.14.0 回归测试:确认门判据 + 交付位绑定搜索冠军。

事故来源:run `req-t7-rand400w`(run2,2026-09-30)。数值最优的 trial-022
(冠军 trial-019 配方的 confirm,val 0.807798)被判 rejected,交付却落在
三代前的 trial-011(trial-006 配方,val 0.806054,落后 1.7×MDE),唯一一次
final_test gate 花在它身上。本文件用该 run 的**实测数字**锁定四层修复:

1. 确认门:双侧等价 |Δ|≤2·SE(不含 seed 方差)→ 方向感知非劣 + 保真,
   margin 含 seed 方差;
2. 确认失败记 unconfirmed(不是 rejected);
3. 交付位必须绑定搜索冠军配方,陈旧确认只在容差内可用;
4. evaluate-final 在 gate.acquire 之前做分叉检查。
另含:σ_seed 估计(排除当前 trial)、confirm 预算、multi-seed 证据对账、
baseline 模板 multi-seed 段、leaderboard role/recipe 列。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402  hermetic bootstrap

from runtime.config import Config  # noqa: E402
from runtime.confirmation import (confirm_attempts,  # noqa: E402
                                  confirmation_margins, judge_delta,
                                  legacy_two_sided, pooled_replicate_sd,
                                  recipe_root, resolve_sigma_seed)
from runtime.evaluator import Evaluator  # noqa: E402
from runtime.leaderboard import Leaderboard  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import TrialRecord  # noqa: E402

# ---- run2 实测(trials/*/evaluation.json)----
# (trial, 动作, 父代, val auc, verdict, val Δ, val SE, shadow Δ, shadow SE)
RUN2 = {
    "t006": dict(action="explore", parent=None, val=0.806111,
                 verdict="accepted"),
    "t011": dict(action="confirm", parent="t006", val=0.806054,
                 verdict="confirmed", fresh=True,
                 dval=-5.712587625239518e-05, se_val=1.0e-04,
                 dsh=-0.0002122594665995292, se_sh=0.00010909222207940427),
    "t015": dict(action="exploit", parent="t006", val=0.806262,
                 verdict="accepted"),
    "t016": dict(action="confirm", parent="t015", val=0.806132,
                 verdict="rejected", dval=-0.00013036781105757456,
                 se_val=1.0e-04, dsh=-0.00023505640123222538,
                 se_sh=0.00011219086636006094),
    "t019": dict(action="hpo", parent="t015", val=0.807621,
                 verdict="accepted"),
    "t020": dict(action="confirm", parent="t019", val=0.807473,
                 verdict="rejected", dval=-0.00014794062462120117,
                 se_val=8.2e-05, dsh=-0.00018726455806372666,
                 se_sh=8.246624968432436e-05),
    "t022": dict(action="confirm", parent="t019", val=0.807798,
                 verdict="rejected", dval=0.00017767590891870633,
                 se_val=8.403445919796693e-05,
                 dsh=0.00012134394369767154, se_sh=8.370719621094889e-05),
}


def _spec() -> TaskSpec:
    return TaskSpec(data={
        "task_id": "v2140", "task_name": "v2140",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "shadow_test": "2026-03",
                         "final_test": "2026-04"},
        "data": {},
    })


def _cfg() -> Config:
    return Config.load(skill_config=False)


def _trial(tmp: Path, tid: str, *, action: str, parent: str | None,
           val: float, verdict: str, fresh: bool = False,
           se: float = 1.0e-04) -> TrialRecord:
    d = tmp / "trials" / tid
    d.mkdir(parents=True, exist_ok=True)
    ev = {"primary_metric": {"name": "auc", "value": val},
          "verdict": verdict, "integrity": {"valid": True},
          "delta": {"vs_parent_paired_bootstrap": {"se": se}}}
    if action == "confirm":
        ev["fresh_confirmation"] = {"passed": fresh}
    return TrialRecord(data={
        "trial_id": tid, "algorithm": "xgboost", "generation": 1,
        "status": "EVALUATED", "search_action": action,
        "parent_trial_ids": [parent] if parent else [],
        "evaluation": ev}, trial_dir=d)


def _run2_trials(tmp: Path, overrides: dict | None = None) -> list:
    out = []
    for tid, row in RUN2.items():
        row = {**row, **((overrides or {}).get(tid) or {})}
        out.append(_trial(tmp, tid, action=row["action"],
                          parent=row["parent"], val=row["val"],
                          verdict=row["verdict"],
                          fresh=bool(row.get("fresh"))))
    return out


# ---------------------------------------------------------------------------
# 1. 确认门判据:run2 四次 confirm 回放
# ---------------------------------------------------------------------------

def _judge(delta, se, higher_better=True, sigma=None):
    sig = resolve_sigma_seed(se=se, floor_ratio=1.0) if sigma is None \
        else {"sigma_seed": sigma}
    m = confirmation_margins(se=se, sigma_seed=sig["sigma_seed"], n_child=1)
    return judge_delta(delta, m, higher_better), m


@pytest.mark.parametrize("tid", ["t011", "t016", "t020", "t022"])
def test_run2_confirms_all_pass_new_criterion(tid):
    """同配方换 seed 的四次 confirm:新判据下 validation/shadow 全部非劣。

    旧判据(|Δ|≤2·SE)只放过了 t011 —— 而 t011 是 shadow 上显著更差
    (CI 整段为负)的那一次。
    """
    row = RUN2[tid]
    for d, se in ((row["dval"], row["se_val"]), (row["dsh"], row["se_sh"])):
        j, m = _judge(d, se)
        assert j["passed"], (tid, d, m)


def test_run2_legacy_gate_was_a_lottery():
    """旧判据回放:4 次 confirm 只过 1 次,且 t022 因"比父代好"被拒。"""
    passed = []
    for tid in ("t011", "t016", "t020", "t022"):
        row = RUN2[tid]
        ok = legacy_two_sided(row["dval"], row["se_val"]) and \
            legacy_two_sided(row["dsh"], row["se_sh"])
        passed.append(tid) if ok else None
    assert passed == ["t011"]
    assert RUN2["t022"]["dval"] > 2 * RUN2["t022"]["se_val"]  # 好过头


def test_genuine_degradation_still_fails():
    j, m = _judge(-1.0e-03, 8.4e-05)
    assert not j["non_inferior"] and not j["passed"]


def test_fidelity_upper_bound_catches_non_replay():
    """借 confirm 之名换配方(远优于父代)→ 保真失败,不得绕过代级 FDR。"""
    j, m = _judge(+5.0e-03, 8.4e-05)
    assert j["non_inferior"] and not j["fidelity_ok"] and not j["passed"]


def test_direction_aware_for_lower_is_better_metric():
    j, _ = _judge(+1.0e-03, 8.4e-05, higher_better=False)   # logloss 变大=变差
    assert not j["non_inferior"]
    j, _ = _judge(-1.0e-04, 8.4e-05, higher_better=False)
    assert j["passed"]


def test_margin_includes_seed_variance_and_shrinks_with_seeds():
    m1 = confirmation_margins(se=1e-4, sigma_seed=1e-4, n_child=1)
    m3 = confirmation_margins(se=1e-4, sigma_seed=1e-4, n_child=3)
    assert math.isclose(m1["sd_delta"], math.sqrt(1e-8 + 2e-8))
    assert m3["sd_delta"] < m1["sd_delta"]
    assert m1["lower"] > 2 * 1e-4  # 旧门宽只有 2·SE
    assert confirmation_margins(se=None, sigma_seed=None) is None


# ---------------------------------------------------------------------------
# 2. σ_seed 解析
# ---------------------------------------------------------------------------

def test_sigma_seed_priority_and_floor():
    se = 1e-4
    s = resolve_sigma_seed(seed_metrics=[0.8076, 0.8075, 0.8078], se=se)
    assert s["source"] == "multi_seed" and s["df"] == 2
    s = resolve_sigma_seed(seed_metrics=[0.8, 0.8], replicate_sd=2e-4,
                           replicate_df=3, se=se)
    assert s["source"] == "run_replicates" and s["sigma_seed"] == 2e-4
    s = resolve_sigma_seed(replicate_sd=2e-4, replicate_df=1, prior=3e-4,
                           se=se)
    assert s["source"] == "config_prior"      # df<2 的复现组不用
    s = resolve_sigma_seed(se=se)
    assert s["source"] == "floor_se_ratio" and s["sigma_seed"] == se
    s = resolve_sigma_seed(seed_metrics=[0.8, 0.8, 0.8], se=se)
    assert s["sigma_seed"] == se and s["source"] == "multi_seed+floor"


def test_replicates_group_by_recipe_and_exclude_current(tmp_path):
    trials = _run2_trials(tmp_path)
    by_id = {t.trial_id: t for t in trials}
    assert recipe_root(by_id["t022"], by_id) == "t019"
    assert recipe_root(by_id["t011"], by_id) == "t006"
    assert recipe_root(by_id["t019"], by_id) == "t019"
    # 评估 t022 时必须排除 t022 本身(否则 σ̂=|Δ|/√2 自证通过)
    rep = pooled_replicate_sd([t for t in trials if t.trial_id != "t022"])
    assert rep["df"] == 3 and set(rep["groups"]) == {"t006", "t015", "t019"}
    assert 5e-5 < rep["sd"] < 1.2e-4
    rep_all = pooled_replicate_sd(trials, exclude_ids=["t022"])
    assert rep_all == rep


def test_evaluator_confirm_verdict_run2_t022(tmp_path):
    """trial-022 的 validation 形态:旧口径 rejected,新口径 confirmed。"""
    ev = Evaluator(_spec(), _cfg())
    row = RUN2["t022"]
    e = {"integrity": {"valid": True}, "noise": row["se_val"],
         "primary_metric": {"value": row["val"]},
         "delta": {"vs_parent": row["dval"],
                   "vs_parent_paired_bootstrap": {"se": row["se_val"]}},
         "stability": {"passed": True}}
    verdict, _, reason = ev._verdict(e, confirm_trial=True)
    assert verdict == "confirmed", reason
    cv = e["confirmation_validation"]
    assert cv["legacy_two_sided_equivalence"] is False
    assert cv["non_inferior"] and cv["fidelity_ok"]


def test_confirm_stability_limit_uses_mde(tmp_path):
    """confirm 的 multi-seed 门 = max(2·noise, ratio×MDE):seed 方差天然
    大于评测噪声的大样本上,不再结构性误杀。"""
    import numpy as np

    ev = Evaluator(_spec(), _cfg())
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) > 0.7).astype(int)
    parent = rng.normal(size=n) + y * 0.8
    child = parent + 0.02 * rng.normal(size=n)
    seeds = [parent + 0.05 * rng.normal(size=n) for _ in range(2)]
    out = ev.evaluate(y, child, parent_score=parent, confirm_trial=True,
                      multi_seed_scores=seeds)
    assert out["stability"]["confirm_std_limit"] >= 0.5 * 0.001
    cv = out["confirmation_validation"]
    assert cv["delta_basis"] == "seed_mean" and cv["n_seeds"] == 3
    assert cv["sigma_seed_source"].startswith("multi_seed")
    assert out["verdict"] in ("confirmed", "unconfirmed")
    assert out["verdict"] != "rejected"


# ---------------------------------------------------------------------------
# 3. 交付位绑定搜索冠军
# ---------------------------------------------------------------------------

def test_run2_delivery_diverges_instead_of_stale_fallback(tmp_path):
    """run2 原样(旧 verdict):不得再静默交付 t011。"""
    lb = Leaderboard(tmp_path, _spec(), _cfg())
    trials = _run2_trials(tmp_path)
    assert lb.best_trial(trials, delivery=True) is None
    res = lb.last_delivery_resolution
    assert res["status"] == "diverged"
    assert res["search_champion"] == "t019"
    assert res["search_recipe"] == "t019"
    stale = {s["trial_id"]: s for s in res["stale_candidates"]}
    assert set(stale) == {"t011"}
    assert math.isclose(stale["t011"]["gap"], 0.807621 - 0.806054,
                        abs_tol=1e-9)
    assert stale["t011"]["gap"] > res["tolerance"] >= 0.001


def test_run2_delivery_binds_champion_recipe_under_new_verdicts(tmp_path):
    """新判据下 t020/t022 均通过 → 交付位绑定冠军配方,取 t022。"""
    lb = Leaderboard(tmp_path, _spec(), _cfg())
    trials = _run2_trials(tmp_path, overrides={
        "t020": {"verdict": "confirmed", "fresh": True},
        "t022": {"verdict": "confirmed", "fresh": True},
        "t016": {"verdict": "confirmed", "fresh": True}})
    best = lb.best_trial(trials, delivery=True)
    res = lb.last_delivery_resolution
    assert res["status"] == "bound"
    assert res["search_recipe"] == "t019"
    assert best.trial_id == "t022"
    assert set(res["bound_candidates"]) == {"t020", "t022"}


def test_stale_confirmation_within_tolerance_is_usable(tmp_path):
    lb = Leaderboard(tmp_path, _spec(), _cfg())
    trials = _run2_trials(tmp_path, overrides={"t019": {"val": 0.8063}})
    # 冠军 t019(0.8063)领先 t011 0.000246 < MDE → 陈旧确认可用但留痕
    best = lb.best_trial(trials, delivery=True)
    assert best is not None and best.trial_id == "t011"
    assert lb.last_delivery_resolution["status"] == "stale_within_tolerance"


def test_divergence_check_helper(tmp_path):
    from scripts._mvo.cmd_deliver import _delivery_divergence_check

    lb = Leaderboard(tmp_path, _spec(), _cfg())
    trials = _run2_trials(tmp_path)
    ok, info = _delivery_divergence_check(lb, trials, "t011")
    assert ok is False and info["bound"] is False
    assert info["gap"] > info["tolerance"]
    ok, info = _delivery_divergence_check(lb, trials, "t022")
    assert ok is True and info["bound"] is True


def test_leaderboard_build_marks_roles_and_recipe(tmp_path):
    lb = Leaderboard(tmp_path, _spec(), _cfg())
    trials = _run2_trials(tmp_path)
    df = lb.build(trials)
    rows = {r["trial_id"]: r for r in df.to_dict("records")}
    assert rows["t022"]["recipe"] == "t019"
    assert rows["t019"]["role"] == "search_champion"
    assert not rows["t011"]["role"] or "delivery" not in rows["t011"]["role"]
    js = json.loads((tmp_path / "leaderboard" / "leaderboard.json").read_text())
    assert js["delivery_resolution"]["status"] == "diverged"
    md = (tmp_path / "leaderboard" / "leaderboard.md").read_text()
    assert "搜索冠军" in md and "交付位" in md and "分叉" in md


# ---------------------------------------------------------------------------
# 4. 不可逆 gate 前的分叉守卫 / CLI
# ---------------------------------------------------------------------------

def _src(rel: str) -> str:
    return (SKILL / rel).read_text(encoding="utf-8")


def test_evaluate_final_checks_divergence_before_gate():
    body = _src("scripts/_mvo/cmd_deliver.py").split(
        "def cmd_evaluate_final(")[1]
    i_div = body.index("_delivery_divergence_check(")
    i_gate = body.index('gate.acquire(purpose="final_model_evaluation")')
    assert i_div < i_gate, "分叉守卫必须在 acquire gate 之前"
    assert "ack_divergence" in body


def test_finalize_retrain_no_silent_stale_fallback():
    body = _src("scripts/_mvo/cmd_deliver.py").split(
        "def cmd_finalize_retrain(")[1].split(
            "\ndef cmd_finalize_baseline_score(")[0]
    assert '_res.get("status") == "diverged"' in body
    assert "_write_divergence_note(" in body
    assert "_delivery_divergence_check(lb, trials, champ_id)" in body
    assert '"search_champion": _res.get("search_champion")' in body


def test_stop_condition_and_improvement_count_bound_to_champion():
    src = _src("runtime/evolution.py")
    assert '_delivery_res.get("delivery_trial") is not None' in src
    assert "fresh_confirmation_available=any(" not in src
    assert 'str(t.data.get("search_action") or "") != "confirm"' in src


def test_cli_flags_parse():
    sys.path.insert(0, str(SKILL / "scripts"))
    import modelevo as cli

    p = cli.build_parser()
    assert p.parse_args(["finalize-retrain", "--run", "x",
                         "--ack-divergence", "r"]).ack_divergence == "r"
    assert p.parse_args(["finalize-all", "--run", "x",
                         "--ack-divergence", "r"]).ack_divergence == "r"
    assert p.parse_args(["evaluate-final", "--run", "x",
                         "--ack-divergence", "r"]).ack_divergence == "r"
    assert p.parse_args(["new-trial", "--run", "x", "--action", "confirm",
                         "--allow-extra-confirm", "r"]
                        ).allow_extra_confirm == "r"


# ---------------------------------------------------------------------------
# 5. confirm 预算 / multi-seed 证据 / verdict 语义
# ---------------------------------------------------------------------------

def test_confirm_attempts_count_evaluated_only(tmp_path):
    trials = _run2_trials(tmp_path)
    assert {t.trial_id for t in confirm_attempts(trials, "t019")} == \
        {"t020", "t022"}
    bad = _trial(tmp_path, "t021", action="confirm", parent="t019",
                 val=0.0, verdict="invalid")
    bad.data["status"] = "FAILED"
    assert {t.trial_id for t in confirm_attempts(trials + [bad], "t019")} \
        == {"t020", "t022"}


def test_planner_respects_confirm_budget():
    src = _src("runtime/planner.py")
    body = src.split("def _confirmation_candidates(")[1].split("\n    def ")[0]
    assert "max_confirm_attempts_per_recipe" in body
    assert "confirm_attempts(self.trials" in body
    assert _cfg().get("evolution.max_confirm_attempts_per_recipe") == 2


def test_new_trial_confirm_budget_guard():
    body = _src("scripts/_mvo/cmd_trial.py").split("def cmd_new_trial(")[1]
    assert "confirm_attempts(factory.all_trials(), from_parent)" in body
    assert "allow_extra_confirm" in body


def test_multi_seed_evidence_status(tmp_path):
    import pandas as pd
    import yaml

    from runtime.runner import TrialRunner

    d = tmp_path / "trials" / "t9"
    (d / "config").mkdir(parents=True)
    (d / "config" / "trial.yaml").write_text(
        yaml.safe_dump({"multi_seed": 3}))
    t = TrialRecord(data={"trial_id": "t9"}, trial_dir=d)
    assert TrialRunner._multi_seed_evidence(None, t)["status"] == "missing"
    (d / "seed_scores").mkdir()
    pd.DataFrame({"score": [0.1]}).to_parquet(d / "seed_scores" / "seed-1.parquet")
    assert TrialRunner._multi_seed_evidence(None, t)["status"] == "partial"
    pd.DataFrame({"score": [0.1]}).to_parquet(d / "seed_scores" / "seed-2.parquet")
    assert TrialRunner._multi_seed_evidence(None, t)["status"] == "complete"


def test_unconfirmed_is_a_verdict_and_not_a_candidate(tmp_path):
    from runtime.trial import TRIAL_VERDICTS

    assert "unconfirmed" in TRIAL_VERDICTS
    lb = Leaderboard(tmp_path, _spec(), _cfg())
    t = _trial(tmp_path, "u1", action="confirm", parent=None, val=0.99,
               verdict="unconfirmed")
    assert lb.best_trial([t]) is None


def _reference_runner(tmp_path, monkeypatch, ref_scores):
    import pandas as pd
    import yaml

    import runtime.runner as rr
    from runtime.data import SAMPLE_KEY_COLUMN

    for tid, parents in (("P", ["G"]), ("C", ["P"])):
        d = tmp_path / "trials" / tid
        d.mkdir(parents=True)
        (d / "trial.yaml").write_text(yaml.safe_dump({
            "trial_id": tid, "parent_trial_ids": parents,
            "data_view": "v1"}))
    keys = [f"k{i}" for i in range(len(ref_scores))]
    monkeypatch.setattr(rr, "load_parent_scores", lambda run_dir, pid: {
        "data_view": "v1",
        "shadow_test": pd.DataFrame({"score": ref_scores,
                                     SAMPLE_KEY_COLUMN: keys})})
    runner = rr.TrialRunner.__new__(rr.TrialRunner)
    runner.run_dir, runner.spec, runner.config = tmp_path, _spec(), _cfg()
    child = TrialRecord(data={"trial_id": "C", "parent_trial_ids": ["P"],
                              "data_view": "v1"},
                        trial_dir=tmp_path / "trials" / "C")
    return runner, child, keys


@pytest.mark.parametrize("child_quality,expect_ok", [(1.0, True),
                                                     (-1.0, False)])
def test_confirm_reference_check(tmp_path, monkeypatch, child_quality,
                                 expect_ok):
    """父代宣称的增益在独立 shadow 上还在吗:子代 vs 父代的参照。"""
    import numpy as np
    import pandas as pd

    from runtime.confirmation import confirm_config
    from runtime.data import SAMPLE_KEY_COLUMN

    rng = np.random.default_rng(1)
    n = 3000
    y = (rng.random(n) > 0.7).astype(int)
    ref = rng.normal(size=n) + 0.5 * y
    runner, child, keys = _reference_runner(tmp_path, monkeypatch, ref)
    sscore = ref + child_quality * 0.8 * y + 0.1 * rng.normal(size=n)
    shadow = pd.DataFrame({"label": y})
    sh_df = pd.DataFrame({"score": sscore, SAMPLE_KEY_COLUMN: keys})
    out = runner._confirm_reference_check(
        child, shadow, sscore, sh_df, None,
        sigma_info={}, cc=confirm_config(runner.config), higher_better=True)
    assert out["available"] is True and out["reference_trial"] == "G"
    assert out["non_inferior"] is expect_ok
    assert out["gain_reproduced"] is expect_ok


def test_runner_shadow_block_uses_new_criterion():
    src = _src("runtime/runner.py")
    assert '"criterion": CONFIRM_CRITERION' in src
    assert 'evaluation["verdict"] = "unconfirmed"' in src
    assert "_confirm_reference_check(" in src
    # 旧的双侧判定不再决定 passed(只作 legacy 诊断字段)
    assert "_equiv = abs(float(_shadow_delta)) <= _margin" not in src
    # leak_scan 等已裁 invalid 时不被确认结论覆盖
    assert 'if _prior_verdict == "invalid":' in src


def test_baseline_template_has_multi_seed_block():
    import ast

    from runtime.baseline import BASELINE_TEMPLATE, ESTIMATOR_CODE, FIT_CODE
    from runtime.code_audit import check_generated_code

    for algo in sorted(ESTIMATOR_CODE):
        code = BASELINE_TEMPLATE.format(
            algorithm=algo, target="y", model_type="binary_classification",
            treatment="", seed=0, n_estimators_cap=2000,
            early_stopping_rounds=50, es_fraction=0.1,
            estimator_code=ESTIMATOR_CODE[algo], fit_code=FIT_CODE[algo])
        ast.parse(code)
        assert check_generated_code(code) == [], algo
        assert "seed_scores" in code and "_fit_seed_variant(" in code
        assert 'os.environ["MODELEVO_SEED_VARIANT"] = "1"' in code
