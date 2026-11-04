#!/usr/bin/env python3
"""工作区标准工具:修复/补齐 baseline_id 与 delta_vs_baseline。

触发场景(实测事故):
  baseline 命令中途被 kill(timeout/断电),run-state.json 的 baseline_id
  未及落盘 → 后续所有 trial 的 evaluator 拿不到 baseline_score,
  leaderboard 的 delta_vs_baseline 恒空(None)。

本工具:
  1) 检测 run-state.json 的 baseline_id:
      已设 → 校验其 validation_score 存在,直接用;
      未设 → 兜底取 gen-00 序号最小的非 warm-start trial(trial-001)并回写 run-state;
      显式 --baseline-id 可覆盖(须为真实 gen-00 baseline trial)。
  2) 对全部 terminal trial(SUCCEEDED/EVALUATED/REFLECTED)调
     runner.evaluate_only(baseline_score=...) 重算 evaluation.json,
     复用已落盘 validation_score.parquet,不重训,补齐 vs_baseline
     (vs_parent 由父代分数同时保留)。
  3) 重建 leaderboard。幂等,可重复执行。

用法(在 run 目录):
  python tools/backfill_baseline.py [--baseline-id <trial_id>]

依赖:skill 的 runtime(hash 不变,不触发铁律 10);不修改 immutable runtime。
"""
import argparse
import pathlib
import sys

RUN_DIR = pathlib.Path(__file__).resolve().parents[1]
SKILL_ROOT = RUN_DIR / ".claude" / "skills" / "modelevo"
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

from runtime.config import Config
from runtime.logging_utils import dump_json, load_json, locked_state_write
from runtime.runner import TrialRunner, load_parent_scores
from runtime.spec import TaskSpec
from runtime.trial import TrialRecord
from runtime.leaderboard import Leaderboard
from runtime.trial import TrialFactory


def _pick_default_baseline(factory, run_dir):
    """未显式指定时:run-state 已设优先;否则兜底 gen-00 序号最小的
    非 warm-start trial(v2.15.7:即 trial-001 baseline,不看是否有指标——
    旧口径"首个有指标 trial"会在 baseline 失败时误取 warm-start trial)。"""
    st_path = run_dir / "run-state.json"
    if st_path.exists():
        st = load_json(st_path)
        bid = st.get("baseline_id")
        if bid and (run_dir / "trials" / bid).is_dir():
            return bid
    gen0 = [t for t in factory.all_trials()
            if t.generation == 0
            and t.data.get("search_action") != "warm_start"
            and not str((t.data.get("hypothesis") or {}).get(
                "statement", "")).startswith("[warmstart")]
    return min(gen0, key=lambda t: t.trial_id).trial_id if gen0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-id", default=None,
                    help="显式指定基线 trial_id(须为 gen-00 baseline)")
    args = ap.parse_args()

    run_dir = RUN_DIR
    spec = TaskSpec.load(run_dir / "task-spec.yaml")
    config = Config(run_dir / "modelevo-config.yaml")
    factory = TrialFactory(run_dir, spec=spec, config=config)

    baseline_id = args.baseline_id or _pick_default_baseline(factory, run_dir)
    if not baseline_id:
        print("[ERR] 未找到可用的 baseline trial(需 gen-00 且已评估)")
        return 1
    bl = load_parent_scores(run_dir, baseline_id)
    if bl is None:
        print(f"[ERR] baseline trial {baseline_id} 无 validation_score.parquet")
        return 1

    # 回写 run-state.baseline_id(幂等)
    st_path = run_dir / "run-state.json"
    st = load_json(st_path) if st_path.exists() else {}
    if st.get("baseline_id") != baseline_id:
        st["baseline_id"] = baseline_id
        with locked_state_write(st_path):
            dump_json(st_path, st)
        print(f"[1] run-state.json baseline_id -> {baseline_id}")
    else:
        print(f"[1] run-state.json baseline_id 已为 {baseline_id}")

    baseline_score = bl["validation"]
    print(f"[2] baseline validation score 行数 = {len(baseline_score)}")

    from runtime.gen_executor import _best_parent_scores
    runner = TrialRunner(run_dir, spec, config)
    n = 0
    for tdir in sorted((run_dir / "trials").iterdir()):
        if not tdir.is_dir():
            continue
        trial = TrialRecord.load(tdir)
        if trial.trial_id == baseline_id:
            continue  # baseline 自身无 vs_baseline
        if trial.status not in ("SUCCEEDED", "EVALUATED", "REFLECTED"):
            print(f"[skip] {trial.trial_id} status={trial.status}")
            continue
        parent_scores = _best_parent_scores(run_dir, spec, trial.parent_ids)
        runner.evaluate_only(trial, parent_scores=parent_scores,
                             baseline_score=baseline_score)
        n += 1
        d = (trial.data.get("evaluation") or {}).get("delta") or {}
        print(f"[3] {trial.trial_id} vs_baseline={d.get('vs_baseline')}")

    lb = Leaderboard(run_dir, spec)
    lb.build(factory.all_trials(), baseline_id=baseline_id)
    print(f"[4] leaderboard rebuilt;{n} trials backfilled(baseline={baseline_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
