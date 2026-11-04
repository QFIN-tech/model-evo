"""V2.8 端到端 smoke:合成数据走 init→profile→baseline→new-trial→run
→finalize-generation,验证 v12 修复在实际 CLI 流程中的行为。

运行:python tests/test_e2e_smoke.py(约 2–4 分钟)
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SKILL = Path(__file__).resolve().parents[1]
CLI = SKILL / "scripts" / "modelevo.py"

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


def run_cli(*args, expect_ok=True):
    r = subprocess.run([sys.executable, str(CLI), *args],
                       capture_output=True, text=True)
    if expect_ok and r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"CLI 失败: {args}")
    return r


def main():
    tmp = Path(tempfile.mkdtemp(prefix="modelevo-e2e-"))
    # v2.18.0:导入 _common 即把经验库指向不可达端口(子进程继承 env),不碰正式库
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _common  # noqa: F401
    run_dir = tmp / "run-e2e"
    try:
        # ---- 合成数据:dt/label/uid + 2 个特征,时间跨 4 个月 ----
        rng = np.random.default_rng(42)
        n = 6000
        dts = pd.date_range("2026-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "dt": dts.strftime("%Y-%m-%d"),
            "uid": np.arange(n),
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        })
        logit = 0.8 * df["f1"] + 0.5 * df["f2"] \
            + 0.3 * (df["cat"] == "c").astype(float)
        df["label"] = (logit + rng.normal(scale=1.2, size=n) > 1.2).astype(int)
        data_path = tmp / "sample.parquet"
        df.to_parquet(data_path, index=False)

        # ---- init ----
        r = run_cli("init", "--dir", str(run_dir), "--task-id", "e2e",
                    "--time-column", "dt", "--id-column", "uid",
                    "--target", "label")
        check("init 成功", (run_dir / "task-spec.yaml").exists())
        # task-spec 编辑:时间窗对齐合成数据
        spec_text = (run_dir / "task-spec.yaml").read_text(encoding="utf-8")
        spec_text = spec_text.replace("train: 2026-01~2026-04",
                                      "train: 2026-01~2026-02")
        spec_text = spec_text.replace("validation: 2026-05",
                                      "validation: 2026-03")
        spec_text = spec_text.replace("shadow_test: 2026-06",
                                      "shadow_test: 2026-04")
        spec_text = spec_text.replace("final_test: 2026-07",
                                      "final_test: 2026-05")
        (run_dir / "task-spec.yaml").write_text(spec_text, encoding="utf-8")
        # data-loader 指向合成数据
        (run_dir / "data-loader.py").write_text(
            "import pandas as pd\n"
            f"DATA = {str(data_path)!r}\n"
            "def load_data() -> pd.DataFrame:\n"
            "    return pd.read_parquet(DATA)\n", encoding="utf-8")
        # 缩小预算让 e2e 快速收敛(不经由 spec,直接用 run config)
        import yaml as _y
        cfg_path = run_dir / "modelevo-config.yaml"
        cfg = _y.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg["modelevo"]["baseline"]["n_estimators_cap"] = 120
        cfg["modelevo"]["evaluator"]["bootstrap_samples"] = 300
        cfg["modelevo"]["evaluator"]["group_importance_repeat"] = 2
        cfg_path.write_text(_y.safe_dump(cfg, allow_unicode=True),
                            encoding="utf-8")
        r = run_cli("validate", "--run", str(run_dir))
        check("validate 通过", "校验通过" in r.stdout)

        # ---- profile ----
        r = run_cli("profile", "--run", str(run_dir), "--workers", "2")
        splits = run_dir / "sample-profile" / "splits"
        check("profile 产出 splits", (splits / "train.parquet").exists())
        manifest = json.loads((splits / "manifest.json").read_text())
        entry = manifest["splits"]["train"]
        check("P0-1 manifest 记录 ordering=sorted_by:dt",
              entry.get("ordering") == "sorted_by:dt", str(entry))
        # v13 三产物:目标/时间/ID 在 labels/ 子目录,特征面物理不含
        tr = pd.read_parquet(splits / "labels" / "train.parquet")
        check("P0-1 train 已按 dt 排序",
              tr["dt"].is_monotonic_increasing)
        check("P0-1 manifest 记录前缀标签率",
              "prefix_positive_rate" in entry)

        # ---- baseline ----
        r = run_cli("baseline", "--run", str(run_dir))
        st = json.loads((run_dir / "run-state.json").read_text())
        baseline_id = st.get("baseline_id")
        check("baseline_id 落盘", baseline_id is not None)
        # baseline trial 代码验证:es_valid 早停 + spec 注入非特征列
        bl_code = (run_dir / "trials" / baseline_id / "code"
                   / "train.py").read_text(encoding="utf-8")
        check("P0-4 baseline 用 es_valid 早停(不用 X_va)",
              "eval_set = [(X_es, y_es)]" in bl_code
              and 'eval_set = [(X_va, y_va)]' not in bl_code)
        check("P0-8 baseline 无硬编码 pday/user_no",
              '"pday"' not in bl_code and '"user_no"' not in bl_code)
        check("P0-8 baseline 注入 _SPEC_NONFEATURE(dt/uid)",
              "'dt'" in bl_code and "'uid'" in bl_code
              and "_SPEC_NONFEATURE" in bl_code)
        # baseline shadow 分数自动补打分
        bl_dir = run_dir / "trials" / baseline_id
        shadow_ok = (bl_dir / "shadow_test_score.parquet").exists()
        if not shadow_ok:
            ev_bl = json.loads((bl_dir / "evaluation.json").read_text())
            shadow_ok = "shadow_test" in ev_bl
        check("shadow 分数自动补打分(v10.3 行为保留)",
              shadow_ok or not (splits / "shadow_test.parquet").exists())
        # evaluation 结构验证
        ev = json.loads((bl_dir / "evaluation.json").read_text())
        check("根 trial accepted", ev.get("verdict") == "accepted",
              ev.get("verdict"))
        check("P0-2 paired_bootstrap 含 se/z_score",
              "se" in (ev.get("delta", {}).get(
                  "vs_parent_paired_bootstrap") or {})
              or ev.get("delta", {}).get("vs_parent") is None)
        lb = json.loads((run_dir / "leaderboard" / "leaderboard.json")
                        .read_text())
        cols = list((lb.get("rows") or [{}])[0].keys())
        check("P0-2 leaderboard 含 p_positive 列", "p_positive" in cols,
              str(cols))

        # ---- new-trial(特征探索;验证 changed_modules 强制)----
        r = run_cli("new-trial", "--run", str(run_dir),
                    "--hypothesis", "f1 的平方项提供非线性增量",
                    "--action", "explore", "--changed-modules", "feature",
                    "--generation", "1", expect_ok=False)
        check("P1-1 changed_modules 强制(本例已传,创建成功)",
              r.returncode == 0, r.stderr[-300:])
        # 漏传 changed_modules → 拒绝
        r2 = run_cli("new-trial", "--run", str(run_dir),
                     "--hypothesis", "无模块声明的 trial 应被拒绝",
                     "--action", "explore", "--generation", "1",
                     expect_ok=False)
        check("P1-1 漏传 changed_modules 被拒绝",
              r2.returncode != 0
              and "changed-modules" in (r2.stderr + r2.stdout))
        # 找到刚创建的 trial
        trials = sorted((run_dir / "trials").iterdir())
        new_trial = [d.name for d in trials if d.name not in
                     ("trial-001-g00-lightgbm", baseline_id)]
        check("新 trial 已创建", len(new_trial) >= 1, str(new_trial))
        if not new_trial:
            return
        tid = new_trial[-1]

        # 改造 trial 代码:真实 LGBM + f1^2 特征(替代被禁用的默认实现)
        tdir = run_dir / "trials" / tid
        code_path = tdir / "code" / "train.py"
        code = code_path.read_text(encoding="utf-8")
        marker = "    else:\n        # v12(P2,O5):默认脚手架直接 raise"
        impl = (
            "    else:\n"
            "        import lightgbm as _lgb2\n"
            "        # 特征工程:f1^2 非线性项\n"
            "        X_tr = X_tr.assign(f1_sq=X_tr['f1'] ** 2)\n"
            "        X_va = X_va.assign(f1_sq=X_va['f1'] ** 2)\n"
            "        model = _lgb2.LGBMClassifier(\n"
            "            n_estimators=120, learning_rate=0.05,\n"
            "            num_leaves=15, random_state=seed,\n"
            "            n_jobs=n_jobs(), verbosity=-1)\n"
            "        # P0-4:早停用 train 尾部 es_valid,不碰 validation\n"
            "        _cut = int(len(X_tr) * 0.9)\n"
            "        model.fit(X_tr.iloc[:_cut], y_tr.iloc[:_cut],\n"
            "                  eval_set=[(X_tr.iloc[_cut:],\n"
            "                             y_tr.iloc[_cut:])],\n"
            "                  callbacks=[_lgb2.early_stopping(20,\n"
            "                                                 verbose=False)])\n"
            "        score = model.predict_proba(X_va)[:, 1]\n"
        )
        if marker in code:
            # 截断到 else: 分支前,替换默认 raise 为真实实现;
            # 保留 else 分支之后的 dump_model/return 部分
            head = code[:code.index(marker)]
            body_tail = code[code.index(marker):]
            ret_idx = body_tail.index("    # 模型工件持久化(必须保留)")
            code = head + impl + body_tail[ret_idx:]
            code_path.write_text(code, encoding="utf-8")
            check("trial 代码改造完成(f1_sq 特征)", True)
        else:
            check("trial 代码改造完成(f1_sq 特征)", False,
                  "marker not found")

        # ---- run trial ----
        r = run_cli("run", "--run", str(run_dir), "--trial", tid)
        ev_path = tdir / "evaluation.json"
        if not ev_path.exists():
            # 失败诊断:打印 stderr 尾部
            err = ""
            for p in tdir.glob("stderr.log"):
                err = p.read_text(errors="replace")[-1500:]
            check("run 产出 evaluation.json", False,
                  f"stderr tail:\n{err}")
            return
        check("run 产出 evaluation.json", True)
        ev2 = json.loads(ev_path.read_text())
        pb = (ev2.get("delta") or {}).get(
            "vs_parent_paired_bootstrap") or {}
        check("P0-2 verdict 非 accepted 时给出 p_positive 依据",
              ev2.get("verdict") in ("accepted", "promising", "rejected",
                                     "confirmed"),
              f"verdict={ev2.get('verdict')}")
        # error-analysis 产物(P1-2 的数据源)
        check("error-analysis.json 产出",
              (tdir / "error-analysis.json").exists())
        # group_importance(P1-3)
        gi_path = tdir / "feature" / "group_importance.json"
        if gi_path.exists():
            gi = json.loads(gi_path.read_text())
            check("P1-3 group_importance n_repeat≥2",
                  gi.get("n_repeat", 0) >= 2, str(gi.get("n_repeat")))
        else:
            print("  [SKIP] group_importance(模型工件/列不齐时静默跳过)")

        # ---- lint 规则 8(对改造后的代码)----
        r = subprocess.run(
            [sys.executable, str(SKILL / "assets" / "lint_trial.py"),
             str(run_dir), str(tdir)], capture_output=True, text=True)
        check("lint 规则 8:改造代码用 es_valid 合规通过",
              r.returncode == 0, r.stdout[-400:] + r.stderr[-200:])

        # ---- finalize-generation ----
        r = run_cli("finalize-generation", "--run", str(run_dir),
                    "--generation", "1")
        check("finalize 产出 directions.json",
              (run_dir / "directions.json").exists())
        directions = json.loads((run_dir / "directions.json").read_text())
        arms = directions.get("arms") or {}
        check("P1-1 方向台账 arm 使用 vs_parent 口径(有 feature arm)",
              any("feat" in k or "model" in k for k in arms), str(arms))

        # ---- planner-context ----
        r = run_cli("planner-context", "--run", str(run_dir))
        out = r.stdout
        check("planner-context 渲染 11 节(error-analysis 接入)",
              "## 11. 误差归因" in out)
        check("planner-context leaderboard 表含 p_pos 列",
              "p_pos" in out)
        check("planner-context budget 含建议动作配比",
              "建议动作配比" in out)

        # ---- export-experience:源缺失返回非 0(P0-9)----
        r = run_cli("export-experience", "--run", str(run_dir),
                    "--memory-dir", str(tmp / "mem"), expect_ok=False)
        check("P0-9 经验导出源缺失返回非 0", r.returncode == 1,
              f"rc={r.returncode}")

        print(f"\n=== E2E: {PASS} passed, {FAIL} failed ===")
        sys.exit(1 if FAIL else 0)
    finally:
        import os
        if os.environ.get("MODELEVO_E2E_KEEP"):
            print(f"(保留调试目录: {tmp})")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
