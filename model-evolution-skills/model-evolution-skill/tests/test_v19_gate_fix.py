"""回归测试:FinalTestGate 前置校验 + --best-trial 不可绕客观门。

v2.11.9 自包含化:原先本文件 `shutil.copytree` 8.3GB 的示例 run **三次**,
且把示例 run 的绝对路径写死在源码里。现在改为:

- 示例 run 相对 skill 根定位(`<skill>/../example-*`),不写死绝对路径;
- 用 `_fixture.slim_run` 生成 15MB 瘦身夹具(元数据原样 + parquet 截断),
  再按场景 `copytree` 15MB 小夹具;
- 示例 run 缺失时**显式跳过**(打印 SKIP 并 exit 0),不静默通过;
- 选型分裂由 `_fixture.declare_recipe_champion` **显式构造**,不再依赖
  示例 run 的 recipe 恰好与选型不一致(该前提在示例 run 重新交付后已消失,
  旧版因此稳定失败——实测 2026-09-16)。
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL, tmp_run_dir  # noqa: E402
from _fixture import find_run, reset_gate, slim_run, declare_recipe_champion  # noqa: E402

sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))

#: 示例 run(相对 <skill>/../);缺失则整个文件跳过
RUN_SRC = find_run(
    "example-seq-v3-4-2.8.6-v17.4-v4-flash/run-req-t7-drawdown",
    "example-seq-v3-4-2.8.6-v19.2",
)

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    print(("  [PASS] " if cond else "  [FAIL] ") + name
          + (f"  -- {detail}" if detail and not cond else ""))
    ok, fail = ok + (1 if cond else 0), fail + (0 if cond else 1)


def _fresh_run(base: pathlib.Path, name: str) -> pathlib.Path:
    """从瘦身夹具复制一份干净的 run 工作副本。"""
    dst = base / name
    shutil.copytree(base / "slim", dst)
    reset_gate(dst)
    return dst


def main() -> int:
    if RUN_SRC is None:
        print("  [SKIP] 未找到示例 run(<skill>/../example-*-.../),"
              "本文件依赖一个真实 run 作夹具。")
        print("\n=== gate-fix: SKIPPED(无示例 run)===")
        return 0

    tmp = tmp_run_dir("gate-fix-")
    try:
        print("== 0. 构建瘦身夹具 ==")
        slim = slim_run(RUN_SRC, tmp / "slim")
        print(f"  [INFO] 夹具来源 {RUN_SRC}")
        print(f"  [INFO] 瘦身夹具 {slim}")

        from scripts.modelevo import cmd_evaluate_final  # noqa: E402

        # 真实存在的、但**不是**确定性选型的 trial;用它制造"交付链口径分裂"
        other = "trial-003-g01-lightgbm-2234"

        # --- 场景 1:recipe 声明与确定性选型不一致 → acquire 前失败,gate 不消耗
        print("\n== 1. 选型分裂在 acquire 前失败 ==")
        r = _fresh_run(tmp, "run1")
        declare_recipe_champion(r, other)
        args = types.SimpleNamespace(spec=str(r / "task-spec.yaml"), run=str(r),
                                     allow_no_shadow=False, best_trial=None)
        rc = cmd_evaluate_final(args)
        gate_path = r / "final-test-gate.json"
        gate = (json.loads(gate_path.read_text())
                if gate_path.exists() else {"count": 0})
        check("选型分裂在 acquire 前失败: exit=1", rc == 1, f"rc={rc}")
        check("gate 未被消耗: count==0", gate.get("count", 0) == 0,
              str(gate))
        check("evaluation.json 未落盘",
              not (r / "final-model" / "evaluation.json").exists())

        # --- 场景 2:显式指定旧 trial 也不能绕 fresh confirmation
        print("\n== 2. --best-trial 不能绕独立确认门 ==")
        args2 = types.SimpleNamespace(spec=str(r / "task-spec.yaml"), run=str(r),
                                      allow_no_shadow=False, best_trial=other)
        rc2 = cmd_evaluate_final(args2)
        ev_path = r / "final-model" / "evaluation.json"
        check("旧 trial 无 fresh confirmation 被拒绝",
              rc2 == 1 and not ev_path.exists(), f"rc2={rc2}")
        gate2 = json.loads(gate_path.read_text())
        check("拒绝发生在 gate 前", gate2.get("count", 0) == 0,
              str(gate2))

        # --- 场景 3:acquire 后失败路径回滚(标签行数不匹配)
        print("\n== 3. 行数不匹配在前置校验失败,gate 不消耗 ==")
        import pandas as pd

        r3 = _fresh_run(tmp, "run3")
        sp = r3 / "final-model" / "final_test_score.parquet"
        df = pd.read_parquet(sp)
        df.iloc[: max(1, len(df) // 2)].to_parquet(sp, index=False)
        args3 = types.SimpleNamespace(spec=str(r3 / "task-spec.yaml"),
                                      run=str(r3), allow_no_shadow=False,
                                      best_trial=None)
        rc3 = cmd_evaluate_final(args3)
        g3_path = r3 / "final-test-gate.json"
        g3 = (json.loads(g3_path.read_text())
              if g3_path.exists() else {"count": 0})
        check("行数不匹配在前置校验失败", rc3 == 1, f"rc3={rc3}")
        check("gate 未消耗(场景3)", g3.get("count", 0) == 0, str(g3))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n=== gate-fix: {ok} passed, {fail} failed ===")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
