"""v2.14.0 端到端:confirm 新判据 + 交付位绑定 + gate 前分叉守卫(CLI 实跑)。

合成数据走 init→profile→baseline→confirm(baseline 血缘,验证模板
multi-seed 段真正执行)→finalize-generation→confirm 预算→finalize-retrain
→(注入更优的未确认冠军)evaluate-final 在 gate 前拒绝→撤销后盲测→report。

运行:python tests/test_e2e_v2_14_confirm_delivery.py(约 2–5 分钟)
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
import yaml

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


def _new_trial_id(run_dir: Path, before: set) -> str:
    now = {d.name for d in (run_dir / "trials").iterdir()}
    new = sorted(now - before)
    assert new, "new-trial 未创建目录"
    return new[-1]


def main():
    tmp = Path(tempfile.mkdtemp(prefix="modelevo-e2e-v2140-"))
    # v2.18.0:导入 _common 即把经验库指向不可达端口(子进程继承 env),不碰正式库
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _common  # noqa: F401
    run_dir = tmp / "run-e2e"
    try:
        rng = np.random.default_rng(7)
        n = 6000
        dts = pd.date_range("2026-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "dt": dts.strftime("%Y-%m-%d"), "uid": np.arange(n),
            "f1": rng.normal(size=n), "f2": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n)})
        logit = 0.8 * df["f1"] + 0.5 * df["f2"] \
            + 0.3 * (df["cat"] == "c").astype(float)
        df["label"] = (logit + rng.normal(scale=1.2, size=n) > 1.2).astype(int)
        data_path = tmp / "sample.parquet"
        df.to_parquet(data_path, index=False)

        run_cli("init", "--dir", str(run_dir), "--task-id", "e2e2140",
                "--time-column", "dt", "--id-column", "uid",
                "--target", "label")
        spec_text = (run_dir / "task-spec.yaml").read_text(encoding="utf-8")
        for a, b in (("train: 2026-01~2026-04", "train: 2026-01~2026-02"),
                     ("validation: 2026-05", "validation: 2026-03"),
                     ("shadow_test: 2026-06", "shadow_test: 2026-04"),
                     ("final_test: 2026-07", "final_test: 2026-05")):
            spec_text = spec_text.replace(a, b)
        (run_dir / "task-spec.yaml").write_text(spec_text, encoding="utf-8")
        (run_dir / "data-loader.py").write_text(
            "import pandas as pd\n"
            f"DATA = {str(data_path)!r}\n"
            "def load_data() -> pd.DataFrame:\n"
            "    return pd.read_parquet(DATA)\n", encoding="utf-8")
        cfg_path = run_dir / "modelevo-config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg["modelevo"]["baseline"]["n_estimators_cap"] = 120
        cfg["modelevo"]["baseline"]["algorithms"] = ["xgboost"]
        cfg["modelevo"]["evaluator"]["bootstrap_samples"] = 300
        cfg["modelevo"]["evaluator"]["group_importance_repeat"] = 2
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True),
                            encoding="utf-8")
        run_cli("validate", "--run", str(run_dir))
        run_cli("profile", "--run", str(run_dir), "--workers", "2")
        run_cli("baseline", "--run", str(run_dir))
        st = json.loads((run_dir / "run-state.json").read_text())
        bid = st["baseline_id"]
        bl_code = (run_dir / "trials" / bid / "code" / "train.py").read_text(
            encoding="utf-8")
        check("baseline 代码含 multi-seed 段", "_fit_seed_variant(" in bl_code)

        # ---- confirm #1(baseline 血缘)----
        before = {d.name for d in (run_dir / "trials").iterdir()}
        run_cli("new-trial", "--run", str(run_dir), "--from-parent", bid,
                "--action", "confirm", "--generation", "1",
                "--hypothesis", "baseline 配方换 seed 可复现")
        c1 = _new_trial_id(run_dir, before)
        run_cli("run", "--run", str(run_dir), "--trial", c1)
        ev1 = json.loads((run_dir / "trials" / c1 / "evaluation.json")
                         .read_text())
        fc = ev1.get("fresh_confirmation") or {}
        ms = ev1.get("multi_seed_evidence") or {}
        cv = ev1.get("confirmation_validation") or {}
        check("confirm multi-seed 真正执行(seed_scores 2 个变体)",
              ms.get("status") == "complete" and ms.get("found") == 2, str(ms))
        check("validation 判据按 seed 均值(n_seeds=3)",
              cv.get("n_seeds") == 3 and cv.get("delta_basis") == "seed_mean",
              str({k: cv.get(k) for k in ("n_seeds", "delta_basis")}))
        check("σ_seed 来源 multi_seed", str(cv.get("sigma_seed_source") or "")
              .startswith("multi_seed"), str(cv.get("sigma_seed_source")))
        check("fresh_confirmation 为新判据且含分量",
              fc.get("criterion") == "noninferiority_seed_aware_v1"
              and "margin_lower" in fc and "legacy_two_sided_equivalence" in fc,
              str(list(fc)))
        check("根父代:参照对照标记为不可用(而非失败)",
              (fc.get("reference") or {}).get("available") is False,
              str(fc.get("reference")))
        check("confirm verdict ∈ {confirmed, unconfirmed}(不再是 rejected)",
              ev1.get("verdict") in ("confirmed", "unconfirmed"),
              ev1.get("verdict"))
        print(f"    c1 verdict={ev1.get('verdict')} reason={ev1.get('reason')}")

        run_cli("finalize-generation", "--run", str(run_dir),
                "--generation", "1")
        lb = json.loads((run_dir / "leaderboard" / "leaderboard.json")
                        .read_text())
        res = lb.get("delivery_resolution") or {}
        check("leaderboard.json 含 delivery_resolution", bool(res), str(res))
        rows = {r["trial_id"]: r for r in lb["rows"]}
        check("confirm 行 recipe=baseline", rows[c1].get("recipe") == bid,
              str(rows[c1].get("recipe")))
        if ev1.get("verdict") == "confirmed":
            check("交付位绑定搜索冠军配方", res.get("status") == "bound"
                  and res.get("delivery_trial") == c1, str(res))
        r = run_cli("status", "--run", str(run_dir))
        check("status 输出 delivery_resolution",
              "delivery_resolution" in r.stdout)

        # ---- confirm 预算:第 2 次允许,第 3 次需用户裁决 ----
        before = {d.name for d in (run_dir / "trials").iterdir()}
        run_cli("new-trial", "--run", str(run_dir), "--from-parent", bid,
                "--action", "confirm", "--generation", "2",
                "--hypothesis", "baseline 配方换第二个独立 seed 仍可复现父代水平")
        c2 = _new_trial_id(run_dir, before)
        run_cli("run", "--run", str(run_dir), "--trial", c2)
        r = run_cli("new-trial", "--run", str(run_dir), "--from-parent", bid,
                    "--action", "confirm", "--generation", "3",
                    "--hypothesis", "baseline 配方换第三个独立 seed 仍可复现父代水平", expect_ok=False)
        check("第 3 次 confirm 超预算被拒", r.returncode != 0
              and "max_confirm_attempts_per_recipe" in (r.stdout + r.stderr),
              (r.stdout + r.stderr)[-400:])
        before = {d.name for d in (run_dir / "trials").iterdir()}
        r = run_cli("new-trial", "--run", str(run_dir), "--from-parent", bid,
                    "--action", "confirm", "--generation", "3",
                    "--hypothesis", "baseline 配方换第三个独立 seed 仍可复现父代水平",
                    "--allow-extra-confirm", "e2e: 用户裁决追加",
                    expect_ok=False)
        check("--allow-extra-confirm 放行并留痕", r.returncode == 0,
              (r.stdout + r.stderr)[-400:])
        if r.returncode == 0:
            c3 = _new_trial_id(run_dir, before)
            rec = yaml.safe_load((run_dir / "trials" / c3 / "trial.yaml")
                                 .read_text())
            check("trial.yaml.extra_confirm 留痕",
                  (rec.get("extra_confirm") or {}).get("reason")
                  == "e2e: 用户裁决追加", str(rec.get("extra_confirm")))
            run_cli("delete-trial", "--run", str(run_dir), "--trial", c3,
                    expect_ok=False)

        run_cli("finalize-generation", "--run", str(run_dir),
                "--generation", "2")
        lb = json.loads((run_dir / "leaderboard" / "leaderboard.json")
                        .read_text())
        res = lb.get("delivery_resolution") or {}
        if res.get("status") != "bound":
            print(f"  [SKIP] 交付链段:两次 confirm 均未通过({res})")
            print(f"\n=== E2E v2.14.0: {PASS} passed, {FAIL} failed ===")
            sys.exit(1 if FAIL else 0)

        # ---- finalize-retrain:绑定交付 ----
        r = run_cli("finalize-retrain", "--run", str(run_dir),
                    expect_ok=False)
        check("finalize-retrain 绑定交付 rc=0", r.returncode == 0,
              (r.stdout + r.stderr)[-800:])
        meta = json.loads((run_dir / "final-model" / "model_meta.json")
                          .read_text())
        check("model_meta 记录搜索冠军与交付位状态",
              meta.get("search_champion") == res.get("search_champion")
              and (meta.get("delivery_resolution") or {}).get("status")
              == "bound", str({k: meta.get(k) for k in (
                  "best_trial", "search_champion", "delivery_resolution")}))
        # v2.16.0:validation 早停的冠军交付自动冻结轮数(不在 validation 上
        # 二次早停);train_tail 口径维持 rerun_recipe
        import os as _os_es

        _exp_mode = ("rerun_recipe"
                     if _os_es.environ.get("MODELEVO_ES_SOURCE") == "train_tail"
                     else "frozen_rounds")
        check("交付 refit_mode 跟随冠军早停面(v2.16.0)",
              meta.get("refit_mode") == _exp_mode
              and (meta.get("n_validation_rows") or 0) > 0,
              str({k: meta.get(k) for k in ("refit_mode", "es_source",
                                            "retrain_builder",
                                            "n_train_rows",
                                            "n_validation_rows")}))
        b = run_cli("finalize-baseline-score", "--run", str(run_dir),
                    expect_ok=False)
        check("finalize-baseline-score rc=0", b.returncode == 0,
              (b.stdout + b.stderr)[-800:])

        # ---- 注入一个更优、未确认的搜索冠军 → evaluate-final 必须在 gate 前拒绝
        fake = run_dir / "trials" / "trial-099-g02-xgboost-fake"
        fake.mkdir(parents=True)
        champ_val = float(res["search_metric"])
        (fake / "trial.yaml").write_text(yaml.safe_dump({
            "trial_id": fake.name, "algorithm": "xgboost", "generation": 2,
            "status": "EVALUATED", "search_action": "explore",
            "parent_trial_ids": [bid], "data_view": "v1",
            "evaluation": {
                "primary_metric": {"name": "auc", "value": champ_val + 0.02},
                "verdict": "accepted", "integrity": {"valid": True},
                "delta": {"vs_parent_paired_bootstrap": {"se": 1e-4}}}},
            allow_unicode=True), encoding="utf-8")
        r = run_cli("evaluate-final", "--run", str(run_dir), expect_ok=False)
        gate = json.loads((run_dir / "final-test-gate.json").read_text()) \
            if (run_dir / "final-test-gate.json").exists() else {"count": 0}
        check("分叉时 evaluate-final 在 gate 前拒绝",
              r.returncode != 0 and gate.get("count", 0) == 0,
              f"rc={r.returncode} gate={gate} "
              f"{(r.stdout + r.stderr)[-600:]}")
        check("写出 DELIVERY-DIVERGENCE.md",
              (run_dir / "final-model" / "DELIVERY-DIVERGENCE.md").exists())
        shutil.rmtree(fake)

        r = run_cli("evaluate-final", "--run", str(run_dir), expect_ok=False)
        check("无分叉时 evaluate-final 正常盲测", r.returncode == 0,
              (r.stdout + r.stderr)[-800:])
        if r.returncode == 0:
            ev_final = json.loads((run_dir / "final-model" / "evaluation.json")
                                  .read_text())
            check("evaluation.json 含 delivery_resolution",
                  (ev_final.get("delivery_resolution") or {}).get("status")
                  == "bound", str(ev_final.get("delivery_resolution")))
        run_cli("report", "--run", str(run_dir))
        rep = (run_dir / "report" / "report.md").read_text(encoding="utf-8")
        check("report 含「交付位与搜索冠军」节", "交付位与搜索冠军" in rep)
        _html = run_dir / "report" / "report.html"
        check("report 默认生成训练历程 report.html(v2.16.1)",
              _html.exists() and 'id="toc"' in _html.read_text(encoding="utf-8"))

        print(f"\n=== E2E v2.14.0: {PASS} passed, {FAIL} failed ===")
        sys.exit(1 if FAIL else 0)
    finally:
        import os
        if os.environ.get("MODELEVO_E2E_KEEP"):
            print(f"(保留调试目录: {tmp})")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
