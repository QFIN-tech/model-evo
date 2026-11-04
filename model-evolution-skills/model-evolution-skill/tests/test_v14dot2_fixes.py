"""v14.2 修复批次单元验证(leaderboard 不全 / models_so_far / GPU 优先)。

覆盖三处修复:
1. acknowledge-harness-change 新增 --measurement-no-score-impact 降级
   登记(measurement 变更但实测分数口径未变 → epoch 不 +1);
2. leaderboard build 与 best_trial 的 epoch 门控同口径(冠军不再落在
   被排除的旧 epoch 行上);models_so_far 列移除;
3. scheduler/pick_gpu "有卡即用"(>20GB 首选,无满足卡时退选空闲显存
   最大的卡,不再把小卡判给 CPU)。
运行:python tests/test_v14dot2_fixes.py
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cli_source  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.config import Config  # noqa: E402
from runtime.harness import harness_file_hashes  # noqa: E402
from runtime.leaderboard import Leaderboard  # noqa: E402
from runtime.scheduler import Scheduler  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402
from runtime.trial import TrialRecord  # noqa: E402

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


def make_spec():
    return TaskSpec(data={
        "task_id": "t-test", "task_name": "t",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "final_test": "2026-04"},
        "data": {}})


def mk_trial(tid, metric, verdict, epoch=None):
    ev = {"primary_metric": {"name": "auc", "value": metric},
          "verdict": verdict}
    d = {"trial_id": tid, "algorithm": "lightgbm", "parent_trial_ids": [],
         "generation": 1, "status": "EVALUATED", "evaluation": ev}
    if epoch is not None:
        d["harness_epoch"] = epoch
    return TrialRecord(data=d)


# ---------------------------------------------------------------------------
# 1) acknowledge 降级登记
# ---------------------------------------------------------------------------

def _load_cli():
    p = Path(__file__).resolve().parents[1] / "scripts" / "modelevo.py"
    spec_m = importlib.util.spec_from_file_location("mcli", str(p))
    m = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(m)
    return m


def test_acknowledge_no_score_impact():
    print("v14.2 acknowledge --measurement-no-score-impact:")
    m = _load_cli()
    tmp = Path(tempfile.mkdtemp())
    try:
        run = tmp / "run-x"
        run.mkdir()
        st = {"harness_epoch": 1, "harness_hash": "deadbeef00000000",
              "harness_files": {k: "aaaa" for k in harness_file_hashes()}}
        (run / "run-state.json").write_text(json.dumps(st))

        class A:
            pass
        a = A()
        a.run, a.reason = str(run), "test"
        a.allow_measurement_impact = False
        a.measurement_no_score_impact = False
        check("measurement 变更无 flag 仍拒绝",
              m.cmd_acknowledge_harness_change(a) == 1)
        a.measurement_no_score_impact = True
        rc = m.cmd_acknowledge_harness_change(a)
        st2 = json.loads((run / "run-state.json").read_text())
        check("降级登记成功且 epoch 不 +1",
              rc == 0 and st2["harness_epoch"] == 1,
              f"rc={rc} epoch={st2['harness_epoch']}")
        check("change_log 留痕 no-score-impact",
              st2["harness_change_log"][-1]["measurement_impact"]
              == "no-score-impact")
        # 再变更 + allow → epoch +1(旧语义不回归)
        st2["harness_hash"] = "cafe000000000000"
        st2["harness_files"]["metrics.py"] = "bbbb"
        (run / "run-state.json").write_text(json.dumps(st2))
        a.allow_measurement_impact = True
        a.measurement_no_score_impact = False
        rc = m.cmd_acknowledge_harness_change(a)
        st3 = json.loads((run / "run-state.json").read_text())
        check("allow-measurement-impact 仍 epoch+1",
              rc == 0 and st3["harness_epoch"] == 2,
              f"rc={rc} epoch={st3['harness_epoch']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2) leaderboard epoch 门控同口径 + models_so_far 移除
# ---------------------------------------------------------------------------

def test_leaderboard_epoch_and_columns():
    print("v14.2 leaderboard epoch 门控 / models_so_far:")
    tmp = Path(tempfile.mkdtemp())
    try:
        spec = make_spec()
        config = Config.load(skill_config=False)
        run_dir = tmp / "run-t"
        run_dir.mkdir()
        lb = Leaderboard(run_dir, spec, config)
        trials = [
            mk_trial("t-old-best", 0.85, "accepted", epoch=0),
            mk_trial("t-new", 0.80, "accepted", epoch=1),
            mk_trial("t-new-rej", 0.83, "rejected", epoch=1),
        ]
        df = lb.build(trials)
        check("models_so_far 列已移除", "models_so_far" not in df.columns)
        check("model_num 列保留", "model_num" in df.columns)
        check("build 排除旧 epoch 行",
              "t-old-best" not in list(df["trial_id"]))
        best = lb.best_trial(trials)
        check("best_trial 与被排除行同口径(不选旧 epoch 最高分)",
              best.trial_id == "t-new", f"best={best.trial_id}")
        # verdict 过滤不回归:表内第一行 rejected 时冠军走 accepted
        trials2 = [mk_trial("t-rej-top", 0.84, "rejected", epoch=1),
                   mk_trial("t-acc", 0.80, "accepted", epoch=1)]
        b2 = Leaderboard(run_dir, spec, config).best_trial(trials2)
        check("verdict 过滤仍生效", b2.trial_id == "t-acc")
        # 单 epoch 时不过滤(全旧 epoch 也能选出冠军)
        trials3 = [mk_trial("t-a", 0.79, "accepted", epoch=0),
                   mk_trial("t-b", 0.81, "accepted", epoch=0)]
        b3 = Leaderboard(run_dir, spec, config).best_trial(trials3)
        check("单 epoch 池不受门控影响", b3.trial_id == "t-b")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 3) scheduler / pick_gpu 有卡即用
# ---------------------------------------------------------------------------

def test_gpu_preferred():
    print("v14.2 scheduler/pick_gpu 有卡即用:")
    tmp = Path(tempfile.mkdtemp())
    try:
        spec = make_spec()
        config = Config.load(skill_config=False)
        snap = {"cpu_count": 8, "cpu_physical": 8, "load1": 1.0,
                "ram_total_gb": 64, "ram_available_gb": 40,
                "gpus": [{"index": 0, "memory_total_gb": 24,
                          "memory_free_gb": 10.0, "util_pct": 30}]}
        plan = Scheduler(config, snap).plan()
        check("10GB 空闲卡降级使用(旧实现判给 CPU)",
              plan.gpu_index == 0
              and plan.env.get("CUDA_VISIBLE_DEVICES") == "0",
              f"gpu={plan.gpu_index}")
        snap_big = {"gpus": [{"index": 1, "memory_total_gb": 40,
                              "memory_free_gb": 30.0, "util_pct": 10}],
                    "cpu_count": 8, "cpu_physical": 8, "load1": 1.0,
                    "ram_total_gb": 64, "ram_available_gb": 40}
        plan_b = Scheduler(config, snap_big).plan()
        check(">20GB 首选路径不回归", plan_b.gpu_index == 1)
        snap_none = dict(snap, gpus=[])
        plan_n = Scheduler(config, snap_none).plan()
        check("无卡回退 CPU(CUDA_VISIBLE_DEVICES 置空)",
              plan_n.gpu_index is None
              and plan_n.env.get("CUDA_VISIBLE_DEVICES") == "")
        # 模板 pick_gpu 源码同步去掉 20GB 硬门槛
        cli = cli_source()
        seg = cli[cli.index("def pick_gpu()"):cli.index("def n_jobs()")]
        check("模板 pick_gpu 无 >20 硬门槛", "best_free > 20" not in seg)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_acknowledge_no_score_impact()
    test_leaderboard_epoch_and_columns()
    test_gpu_preferred()
    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)
