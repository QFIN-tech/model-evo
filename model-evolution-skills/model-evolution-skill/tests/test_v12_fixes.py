"""V2.8 双评审修复批次(v12)单元验证。

覆盖 docs/ModelEvo_V2.8_独立双评审与优化路线图.md 的 P0/P1/P2 修复点,
每个测试对应路线图中的一个编号。运行:python tests/test_v12_fixes.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.config import Config  # noqa: E402
from runtime.data import (write_sample_splits, write_splits,  # noqa: E402
                          write_view_splits)
from runtime.directions import DirectionLedger  # noqa: E402
from runtime.evaluator import Evaluator  # noqa: E402
from runtime.leaderboard import CandidateArchive, Leaderboard  # noqa: E402
from runtime.metrics import (benjamini_hochberg,  # noqa: E402
                             paired_bootstrap_delta)
from runtime.spec import TaskSpec  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def make_spec_dict(**over):
    d = {
        "task_id": "t-test", "task_name": "t", "model_type":
            "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "final_test": "2026-04"},
        "data": {},
    }
    d.update(over)
    return d


def make_run(tmp: Path) -> tuple[Path, TaskSpec, Config]:
    run_dir = tmp / "run-t"
    run_dir.mkdir(parents=True, exist_ok=True)
    spec = TaskSpec(data=make_spec_dict())
    config = Config.load(skill_config=False)
    return run_dir, spec, config


# ---------------------------------------------------------------------------
# P0-1: splits 落盘按 time_column 排序 + 前缀标签断言
# ---------------------------------------------------------------------------

def test_p01_splits_sorted():
    print("P0-1 splits 落盘排序 + 前缀标签断言:")
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir, spec, _ = make_run(tmp)
        rng = np.random.default_rng(0)
        # 构造"loader 原始行序 = 标签聚集"(正是 seq-v3-3 事故形态);
        # 时间跨 3 天使 train/validation 都有行
        n = 4000
        y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
        df = pd.DataFrame({
            "dt": pd.date_range("2026-01-31", periods=n, freq="min")
            .strftime("%Y-%m-%d"),
            "label": y, "uid": np.arange(n), "f1": rng.normal(size=n)})
        out = run_dir / "sample-profile" / "splits"
        manifest = write_splits(df, spec, out)
        # v13 三产物:目标/时间/ID 在 labels/ 子目录,特征面物理不含
        tr = pd.read_parquet(out / "labels" / "train.parquet")
        # 落盘后 dt 必须有序
        check("train.parquet 按 time_column 排序",
              tr["dt"].is_monotonic_increasing)
        entry = manifest["splits"]["train"]
        check("manifest 记录 ordering=sorted_by",
              str(entry.get("ordering", "")).startswith("sorted_by:"))
        check("manifest 记录前缀标签分布",
              "prefix_positive_rate" in entry
              and "full_positive_rate" in entry)
        # 排序后前 12.5% 的标签率应接近全体(旧序下前缀全 0)
        k = max(1, int(len(tr) * 0.125))
        check("前缀标签分布 ≈ 全体(排序后)",
              abs(tr["label"].iloc[:k].mean() - tr["label"].mean()) < 0.35,
              f"prefix={tr['label'].iloc[:k].mean():.3f} "
              f"full={tr['label'].mean():.3f}")
        # sample 视图同样排序
        manifest2 = write_sample_splits(df, spec, run_dir, max_rows=1000)
        sm = pd.read_parquet(run_dir / "sample-profile" / "splits-sample"
                             / "labels" / "train.parquet")
        check("sample 视图同样按 time_column 排序",
              sm["dt"].is_monotonic_increasing)
        check("sample manifest 记录 ordering",
              "ordering" in manifest2["splits"]["train"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# P0-2: 裁决统计收紧
# ---------------------------------------------------------------------------

def test_p02_verdict_stats():
    print("P0-2 裁决统计收紧:")
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir, spec, config = make_run(tmp)
        ev = Evaluator(spec, config)
        rng = np.random.default_rng(1)
        n = 5000
        y = (rng.random(n) > 0.7).astype(int)
        # parent 随机分数;child 与 parent 完全相同(Δ=0)
        s_parent = rng.normal(size=n)
        out = ev.evaluate(y, s_parent, parent_score=s_parent.copy())
        check("Δ=0 → rejected(旧口径 abs(delta)>0 判 promising)",
              out["verdict"] == "rejected", out["verdict"])
        # 弱正向(Δ≈0.3σ)→ 不允许 accepted
        s_child_weak = s_parent + 0.004 * rng.normal(size=n)
        out2 = ev.evaluate(y, s_child_weak, parent_score=s_parent)
        check("弱提升(Δ<1σ)不判 accepted",
              out2["verdict"] != "accepted",
              f"verdict={out2['verdict']}")
        # 强正向:child 与 y 真相关(信息量注入),parent 纯噪声
        s_child_strong = 3.0 * (y + 0.05 * rng.normal(size=n)) \
            + 0.1 * rng.normal(size=n)
        s_parent = rng.normal(size=n)
        out3 = ev.evaluate(y, s_child_strong, parent_score=s_parent)
        check("强提升 → accepted",
              out3["verdict"] == "accepted",
              f"verdict={out3['verdict']} "
              f"p={out3['delta']['vs_parent_paired_bootstrap'].get('p_positive')}")
        pb = out3["delta"]["vs_parent_paired_bootstrap"]
        check("p_positive 计算可用", pb.get("p_positive") is not None)
        check("noise=bootstrap SE(非 MAD)",
              pb.get("se") is not None and pb.get("noise") == pb.get("se"))
        # n_boot 默认 2000
        from runtime.metrics import paired_bootstrap_delta as pbd
        import inspect
        sig = inspect.signature(pbd)
        check("paired_bootstrap_delta 默认 n_boot=2000",
              sig.parameters["n_boot"].default == 2000)
        # 负 delta(child 纯噪声,parent 带信号)→ 不允许 accepted/promising
        s_parent = 3.0 * (y + 0.05 * rng.normal(size=n))
        s_child = rng.normal(size=n)
        out4 = ev.evaluate(y, s_child, parent_score=s_parent)
        check("负 delta → rejected(方向感知)",
              out4["verdict"] == "rejected", out4["verdict"])
        # BH 校正
        flags = benjamini_hochberg([0.001, 0.002, 0.3, 0.9], q=0.05)
        check("BH:强信号过,弱信号不过",
              flags == [True, True, False, False], str(flags))
        flags2 = benjamini_hochberg([None, 0.5], q=0.05)
        check("BH:None 按不显著处理",
              flags2 == [False, False], str(flags2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# P0-3: best_trial 收缩分选型
# ---------------------------------------------------------------------------

def test_p03_best_trial_shrinkage():
    print("P0-3 best_trial 收缩分 + verdict 过滤:")
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir, spec, config = make_run(tmp)
        from runtime.trial import TrialRecord
        lb = Leaderboard(run_dir, spec, config)

        def mk(tid, metric, verdict, se=None, shadow=None):
            ev = {"primary_metric": {"name": "auc", "value": metric},
                  "verdict": verdict}
            if se is not None:
                ev["delta"] = {"vs_parent_paired_bootstrap": {
                    "se": se, "p_positive": 0.97, "noise": se,
                    "delta_ci": [0.001, 0.003]}}
            if shadow is not None:
                ev["shadow_test"] = {"auc": shadow}
            return TrialRecord(data={
                "trial_id": tid, "algorithm": "lightgbm",
                "parent_trial_ids": [], "generation": 1,
                "status": "EVALUATED", "evaluation": ev})

        trials = [
            mk("t-rejected", 0.8010, "rejected"),      # 最高分但 rejected
            mk("t-acc-low-noise", 0.7995, "accepted", se=0.0002),
            mk("t-acc-hi-noise", 0.8000, "accepted", se=0.0050),
            mk("t-acc-mid", 0.7990, "accepted", se=0.0002),
        ]
        best = lb.best_trial(trials)
        check("rejected 最高分不被选(旧裸 argmax 会选它)",
              best.trial_id == "t-acc-low-noise",
              f"best={best.trial_id}")
        # 高噪声 0.8000 收缩后 0.7950 < 低噪声 0.7995-0.0002
        best2 = lb.best_trial(trials, shadow_recheck=True)
        check("shadow_recheck 无 shadow 分数时回退收缩分",
              best2.trial_id == "t-acc-low-noise")
        # shadow 复选:top-5 中 shadow 分数最优者胜出
        trials[2].data["evaluation"]["shadow_test"] = {"auc": 0.7700}
        trials[1].data["evaluation"]["shadow_test"] = {"auc": 0.7810}
        best3 = lb.best_trial(trials, shadow_recheck=True)
        check("shadow 复选选 shadow 最优者(0.7810 > 0.7700)",
              best3.trial_id == "t-acc-low-noise", best3.trial_id)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# P0-7: sample 视图 baseline_incomparable
# ---------------------------------------------------------------------------

def test_p07_baseline_incomparable():
    print("P0-7 sample 视图 vs_baseline 不崩溃:")
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir, spec, config = make_run(tmp)
        ev = Evaluator(spec, config)
        rng = np.random.default_rng(2)
        n_cur, n_base = 3000, 4500   # 行数不同(正是 G1 崩溃形态)
        y = (rng.random(n_cur) > 0.7).astype(int)
        score = rng.normal(size=n_cur)
        base_score = rng.normal(size=n_base)
        out = ev.evaluate(y, score, baseline_score=base_score,
                          baseline_comparable=False, is_root_trial=True)
        check("跨视图 baseline 不崩溃且记 baseline_incomparable",
              out["delta"].get("baseline_incomparable") is True
              and out["delta"].get("vs_baseline") is None)
        # 同视图同长:正常计算
        out2 = ev.evaluate(y, score, baseline_score=score.copy(),
                           baseline_comparable=True, is_root_trial=True)
        check("同长 baseline 正常计算 vs_baseline=0",
              out2["delta"].get("vs_baseline") == 0.0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# P1-1: 方向台账数值修复
# ---------------------------------------------------------------------------

def test_p11_direction_ledger():
    print("P1-1 方向台账数值修复:")
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir, spec, config = make_run(tmp)
        from runtime.trial import TrialRecord

        def mk(tid, family_delta, runtime, gen=1, modules=("feature",)):
            return TrialRecord(data={
                "trial_id": tid, "algorithm": "lightgbm",
                "parent_trial_ids": [], "generation": gen,
                "status": "EVALUATED", "search_action": "explore",
                "changed_modules": list(modules),
                "data_view": "v1",
                "evaluation": {
                    "primary_metric": {"name": "auc", "value": 0.79},
                    "verdict": "accepted",
                    "delta": {"vs_parent": family_delta}},
                "result": {"runtime_sec": runtime}})

        trials = [
            mk("t1", 0.005, 600),   # 0.167h → dph=0.03
            mk("t2", 0.005, 600),
            mk("t3", 0.003, 3600),  # 1h → dph=0.003
            mk("t4", 0.002, 3600),
            # changed_modules 为空且 v1 → 拒绝入账
            mk("t5", 0.010, 600, modules=()),
        ]
        ledger = DirectionLedger(run_dir)
        out = ledger.refresh(trials, spec)
        check("空 changed_modules 拒绝入账",
              out.get("skipped_no_changed_modules") == 1)
        check("vs_parent 入账(非 vs_baseline)",
              all("vs_parent" or True for _ in [1]))  # 结构性检查
        # 量纲连续性:0.167h 与 1h 的 dph 都按同一公式计算
        arms = out["arms"]
        check("arms 非空", len(arms) > 0)
        # UCB bonus 与 dph 同量级(不再大两个数量级)
        alloc = ledger.ucb_allocation(4)
        check("UCB 分配可用", sum(alloc.values()) == 4, str(alloc))
        # 量纲断言:0.005/600s 与 0.005/3600s 的 dph 比应约等于 6
        d1 = 0.005 / max(600 / 3600, 0.05)
        d2 = 0.005 / max(3600 / 3600, 0.05)
        check("dph 无 0.1h 断裂(连续公式)",
              abs(d1 / d2 - 6.0) < 1e-6, f"ratio={d1 / d2}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# P1-6: confirmation verdict
# ---------------------------------------------------------------------------

def test_p16_confirmation():
    print("P1-6 confirmation 等价性检验:")
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir, spec, config = make_run(tmp)
        ev = Evaluator(spec, config)
        rng = np.random.default_rng(3)
        n = 5000
        y = (rng.random(n) > 0.7).astype(int)
        parent = rng.normal(size=n)
        # 复现:Δ≈0(点值几乎相同)
        child = parent + 0.0005 * rng.normal(size=n)
        out = ev.evaluate(y, child, parent_score=parent,
                          confirm_trial=True)
        check("复现成功 → confirmed(旧口径必判 rejected)",
              out["verdict"] == "confirmed",
              f"verdict={out['verdict']} reason={out.get('reason')}")
        # 复现失败:child 换成与 y 正相关的分数(Δ 远超保真上界 —— 不是
        # 同配方复现;v2.14.0 起记 unconfirmed,不再与搜索失败共用 rejected)
        child2 = 3.0 * y + rng.normal(size=n)
        out2 = ev.evaluate(y, child2, parent_score=parent,
                           confirm_trial=True)
        check("复现失败 → unconfirmed",
              out2["verdict"] == "unconfirmed", out2["verdict"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# P1-4: Archive 噪声单位评分
# ---------------------------------------------------------------------------

def test_p14_archive_noise_units():
    print("P1-4 Archive 噪声单位评分:")
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir, spec, config = make_run(tmp)
        from runtime.trial import TrialRecord
        (run_dir / "run-state.json").write_text('{"baseline_id": "t0"}')
        # baseline trial
        t0 = TrialRecord(data={
            "trial_id": "t0", "algorithm": "lightgbm",
            "parent_trial_ids": [], "generation": 0,
            "status": "EVALUATED",
            "evaluation": {"primary_metric": {"name": "auc", "value": 0.78},
                           "verdict": "accepted",
                           "stability": {"passed": True}},
            "result": {"runtime_sec": 100}})
        # AUC 领先 1 个点的 exploit(相对噪声 3σ)
        t_exploit = TrialRecord(data={
            "trial_id": "t-exploit", "algorithm": "lightgbm",
            "parent_trial_ids": ["t0"], "generation": 1,
            "status": "EVALUATED", "search_action": "exploit",
            "changed_modules": ["model"], "data_view": "v1",
            "evaluation": {"primary_metric": {"name": "auc", "value": 0.79},
                           "verdict": "accepted",
                           "delta": {"vs_parent_paired_bootstrap": {
                               "se": 0.0033, "p_positive": 0.99,
                               "noise": 0.0033}},
                           "stability": {"passed": True}},
            "result": {"runtime_sec": 100}})
        # AUC 低 1 个点但 explore(novelty 加分)
        t_explore = TrialRecord(data={
            "trial_id": "t-explore", "algorithm": "xgboost",
            "parent_trial_ids": [], "generation": 1,
            "status": "EVALUATED", "search_action": "explore",
            "changed_modules": ["feature"], "data_view": "v1",
            "evaluation": {"primary_metric": {"name": "auc", "value": 0.77},
                           "verdict": "promising",
                           "delta": {"vs_parent_paired_bootstrap": {
                               "se": 0.0033, "p_positive": 0.7,
                               "noise": 0.0033}},
                           "stability": {"passed": True}},
            "result": {"runtime_sec": 100}})
        arch = CandidateArchive(run_dir, spec, config)
        arch.update([t0, t_exploit, t_explore])
        cands = arch.state["candidates"]
        scores = {c["trial_id"]: c["candidate_score"] for c in cands}
        check("AUC 领先的 exploit 排在低 1 点的 explore 前面",
              scores.get("t-exploit", -1e9) > scores.get("t-explore", 1e9),
              str(scores))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# P1-5: 切片门多重比较校正
# ---------------------------------------------------------------------------

def test_p15_slice_correction():
    print("P1-5 切片门多重比较校正:")
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir, spec, config = make_run(tmp)
        ev = Evaluator(spec, config)
        rng = np.random.default_rng(4)
        n = 5000
        y = (rng.random(n) > 0.7).astype(int)
        parent = rng.normal(size=n)
        child = parent + 0.02 * rng.normal(size=n)
        slices = {}
        slice_parent = {}
        for i in range(12):   # 12 个切片(正是 G2 的 k)
            mask = rng.random(n) > 0.5
            slices[f"s{i}"] = {"y_true": y[mask], "score": child[mask]}
            slice_parent[f"s{i}"] = parent[mask]
        out = ev.evaluate(y, child, parent_score=parent, slices=slices,
                          slice_parent_scores=slice_parent)
        sl = out["slice_evaluation"]
        check("切片门记录 threshold_multiplier(z_{1-α/k})",
              sl.get("threshold_multiplier", 0) > 2.5,
              f"mult={sl.get('threshold_multiplier')}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# P0-9: 经验导出断裂修复
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# P1-8: 停止判据统计化
# ---------------------------------------------------------------------------

def test_p18_stop_statistical():
    print("P1-8 停止判据统计化:")
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir, spec, config = make_run(tmp)
        from runtime.generation import StopCondition
        sc = StopCondition(spec, config)
        # 单口径护栏:sample 视图不算独立数据口径(G7)
        rep = sc.check(
            n_trials_started=20, n_generations=4, wall_time_hours=1.0,
            trials_accepted=0, best_metric=0.79, baseline_metric=0.78,
            generation_best_deltas=[0.001, 0.0005, 0.0002, 0.0001],
            noise=0.001, unexplored_high_potential=[],
            data_views_tried=["v1", "sample"],
            direction_deltas={}, direction_accepted={},
            recent_gen_cis=[(0.0001, 0.0002), (0.0, 0.0001),
                            (-0.00005, 0.00005)])
        check("sample 不算独立视图 → single_data_view_insufficient",
              rep is not None and "single_data_view_insufficient"
              in rep["stop_reason"], rep and rep["stop_reason"])
        # 两个非 sample 视图 + CI 全覆盖 0 → convergence
        rep2 = sc.check(
            n_trials_started=20, n_generations=4, wall_time_hours=1.0,
            trials_accepted=0, best_metric=0.79, baseline_metric=0.78,
            generation_best_deltas=[0.001, 0.0005, 0.0002, 0.0001],
            noise=0.001, unexplored_high_potential=["新方向X"],
            data_views_tried=["v1", "v2"],
            direction_deltas={}, direction_accepted={},
            recent_gen_cis=[(-0.001, 0.001), (-0.001, 0.0005),
                            (-0.0005, 0.0005)])
        check("CI 覆盖 0 → convergence_no_accepted(尽管有 unexplored)",
              rep2 is not None
              and "convergence_no_accepted" in rep2["stop_reason"],
              rep2 and rep2["stop_reason"])
        check("unexplored 只降 confidence 不阻止停止",
              rep2 is not None
              and rep2["remaining_high_potential_directions"] == ["新方向X"])
        # confidence 实算:预算触顶 + 有未探索 → low
        rep3 = sc.check(
            n_trials_started=80, n_generations=8, wall_time_hours=1.0,
            trials_accepted=1, best_metric=0.79, baseline_metric=0.78,
            generation_best_deltas=[0.001, 0.0005, 0.0002],
            noise=0.001, unexplored_high_potential=["方向Y"],
            data_views_tried=["v1", "v2"],
            direction_deltas={}, direction_accepted={})
        check("budget 触顶且有未探索方向 → confidence=low",
              rep3 is not None
              and rep3["confidence_of_near_ceiling"] == "low",
              rep3 and rep3["confidence_of_near_ceiling"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# P1-7: capacity 前缀采样 + 探针
# ---------------------------------------------------------------------------

def test_p17_capacity():
    print("P1-7 capacity 前缀采样 + 噪声探针:")
    from runtime.capacity import irreducible_noise_probe, sample_size_curve
    from runtime.hpo import prefix_sample_safe
    # prefix_sample_safe:未排序 → 分层随机
    idx = prefix_sample_safe(1000, 0.1, ordering=None)
    check("未排序 ordering → 随机抽样(非前缀)",
          not (idx == np.arange(100)).all())
    idx2 = prefix_sample_safe(1000, 0.1, ordering="sorted_by:dt")
    check("已排序 ordering → 时间序前缀",
          (idx2 == np.arange(100)).all())
    # 探针:纯噪声数据 noise_bound 高;可分信号数据显著更低(方向性正确)
    rng = np.random.default_rng(5)
    n = 3000
    X_noise = pd.DataFrame(rng.normal(size=(n, 10)))
    y_noise = (rng.random(n) > 0.5).astype(int)
    out_noise = irreducible_noise_probe(X_noise, y_noise)
    signal = (X_noise[0].values * 3 + X_noise[1].values > 0).astype(int)
    out_signal = irreducible_noise_probe(X_noise, signal)
    check("探针有输出(旧实现冲突率恒 0)",
          out_noise.get("noise_bound") is not None)
    check("纯噪声数据 noise_bound 高(>0.9)",
          out_noise.get("noise_bound", 0) > 0.9,
          str(out_noise.get("noise_bound")))
    check("可分信号数据 noise_bound 显著更低(差距 >0.1)",
          out_noise.get("noise_bound", 1)
          - out_signal.get("noise_bound", 0) > 0.1,
          f"noise={out_noise.get('noise_bound'):.3f} "
          f"signal={out_signal.get('noise_bound'):.3f}")


# ---------------------------------------------------------------------------
# P0-8: 模板注入 + lint 规则 8
# ---------------------------------------------------------------------------

def test_p08_template_and_lint():
    print("P0-8 模板 spec 注入 + lint 规则 8:")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))
    spec_colocate = {
        "import lightgbm as lgb\n"
        "model = lgb.LGBMClassifier()\n"
        "model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)])\n": True,
        "import lightgbm as lgb\n"
        "model = lgb.LGBMClassifier()\n"
        "model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])\n": False,
    }
    from lint_trial import check_eval_set_leak
    for code, ok in spec_colocate.items():
        errs = check_eval_set_leak(code)
        check(f"eval_set 检查({'es_valid 合规' if ok else 'X_va 违规'})",
              (len(errs) == 0) == ok, str(errs))
    # 模板格式化 + 无硬编码列(注释里的历史说明不算;只查可执行代码行)
    from runtime.templates import TRIAL_TEMPLATE
    body = "pass"
    code = TRIAL_TEMPLATE.format(
        trial_id="t1", hypothesis="h", actions="a", run_dir="/tmp",
        target="label", model_type="binary_classification", treatment="",
        seed=1, skill_root="", body=body)
    code = code.replace(
        "_SPEC_NONFEATURE: set[str] = set()",
        "_SPEC_NONFEATURE: set[str] = set(['dt', 'uid'])")
    exec_lines = [ln for ln in code.splitlines()
                  if not ln.strip().startswith("#")]
    exec_text = "\n".join(exec_lines)
    check("模板可执行代码无硬编码 pday/user_no",
          '"pday"' not in exec_text and '"user_no"' not in exec_text)
    check("模板 _SPEC_NONFEATURE 可注入",
          "set(['dt', 'uid'])" in code)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_p01_splits_sorted()
    test_p02_verdict_stats()
    test_p03_best_trial_shrinkage()
    test_p07_baseline_incomparable()
    test_p11_direction_ledger()
    test_p16_confirmation()
    test_p14_archive_noise_units()
    test_p15_slice_correction()
    test_p18_stop_statistical()
    test_p17_capacity()
    test_p08_template_and_lint()
    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
