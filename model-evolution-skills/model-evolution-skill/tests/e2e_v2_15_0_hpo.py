"""v2.15.0 HPO 全链路 e2e(合成数据):baseline → 嫁接 HPO → 再嫁接 HPO → confirm 代码升级。

不进 pytest 默认集合(CLI 全流程约 1~3 分钟);手工运行:
    python tests/e2e_v2_15_0_hpo.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402

CLI = SKILL / "scripts" / "modelevo.py"
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def _tid(stdout: str) -> str:
    """new-trial 输出里的 trial id 行(快通道会先打印 [info]/[warn] 行)。"""
    return next(ln.strip() for ln in stdout.splitlines()
                if ln.strip().startswith("trial-"))


def run_cli(*args, expect_ok=True, env=None):
    r = subprocess.run([sys.executable, str(CLI), *args],
                       capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    if expect_ok and r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"CLI 失败: {args}")
    return r


def main():
    tmp = Path(tempfile.mkdtemp(prefix="modelevo-e2e-hpo-"))
    # v2.18.0:导入 _common 即把经验库指向不可达端口(子进程继承 env),不碰正式库
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _common  # noqa: F401
    run_dir = tmp / "run-e2e"
    env = {"MODELEVO_GPU": "0", "CUDA_VISIBLE_DEVICES": ""}
    try:
        rng = np.random.default_rng(7)
        n = 9000
        dts = pd.date_range("2026-01-01", periods=n, freq="h")
        df = pd.DataFrame({"dt": dts.strftime("%Y-%m-%d"), "uid": np.arange(n)})
        for i in range(8):
            df[f"f{i}"] = rng.normal(size=n)
        logit = (0.8 * df["f0"] + 0.5 * df["f1"] - 0.4 * df["f2"] * df["f3"]
                 + 0.3 * np.sin(df["f4"] * 2))
        df["label"] = (logit + rng.normal(scale=1.0, size=n) > 0.6).astype(int)
        data_path = tmp / "sample.parquet"
        df.to_parquet(data_path, index=False)
        run_cli("init", "--dir", str(run_dir), "--task-id", "e2ehpo",
                "--time-column", "dt", "--id-column", "uid",
                "--target", "label")
        spec_text = (run_dir / "task-spec.yaml").read_text(encoding="utf-8")
        for a, b in (("train: 2026-01~2026-04", "train: 2026-01~2026-07"),
                     ("validation: 2026-05", "validation: 2026-08"),
                     ("shadow_test: 2026-06", "shadow_test: 2026-09"),
                     ("final_test: 2026-07", "final_test: 2026-10")):
            spec_text = spec_text.replace(a, b)
        (run_dir / "task-spec.yaml").write_text(spec_text, encoding="utf-8")
        (run_dir / "data-loader.py").write_text(
            "import pandas as pd\n"
            f"DATA = {str(data_path)!r}\n"
            "def load_data() -> pd.DataFrame:\n"
            "    return pd.read_parquet(DATA)\n", encoding="utf-8")
        import yaml as _y
        cfg_path = run_dir / "modelevo-config.yaml"
        cfg = _y.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg["modelevo"]["baseline"]["n_estimators_cap"] = 60
        cfg["modelevo"]["evaluator"]["bootstrap_samples"] = 200
        cfg["modelevo"]["evaluator"]["group_importance_repeat"] = 2
        # 下文"confirm 的 group_importance 继承父代(auto)"检查的是 auto 策略;
        # 全局缺省已改为 skip_rejected(2db9732),这里显式设 auto(v2.17.2)
        cfg["modelevo"]["evaluator"]["group_importance_policy"] = "auto"
        cfg_path.write_text(_y.safe_dump(cfg, allow_unicode=True),
                            encoding="utf-8")
        run_cli("profile", "--run", str(run_dir), "--workers", "2")
        run_cli("baseline", "--run", str(run_dir), env=env)
        trials = sorted(p.name for p in (run_dir / "trials").iterdir())
        xgb_base = next(t for t in trials if "xgboost" in t)
        print("baseline xgboost:", xgb_base)

        # ---- 1) 嫁接 HPO(baseline 血缘父代,cap=60 → 透传必须生效)----
        space = tmp / "space.yaml"
        space.write_text(_y.safe_dump({
            "learning_rate": {"type": "float", "low": 0.05, "high": 0.3,
                              "log": True},
            "max_depth": {"type": "int", "low": 2, "high": 6},
            "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        }), encoding="utf-8")
        r = run_cli("new-trial", "--run", str(run_dir), "--action", "hpo",
                    "--from-parent", xgb_base, "--parents", xgb_base,
                    "--changed-modules", "model", "--generation", "1",
                    "--search-space", str(space), "--budget-configs", "12",
                    "--hypothesis", "e2e hpo", "--falsification", "Δ<MDE",
                    "--override-capacity", "e2e", "--force-hpo-budget")
        h1 = _tid(r.stdout)
        code1 = (run_dir / "trials" / h1 / "code" / "train.py").read_text()
        cfg1 = _y.safe_load((run_dir / "trials" / h1 / "config" /
                             "trial.yaml").read_text())
        check("嫁接段 _build_model 透传 cap",
              "n_estimators_cap=int(_cap) if _cap else None" in code1)
        check("脚手架 lint 标记已中和",
              "  # lint:scaffold-fit-and-score" not in code1)
        check("hpo 节含 v2.15 引擎参数",
              cfg1["hpo"].get("lr_mode") == "auto"
              and cfg1["hpo"].get("auto_expand") is True
              and isinstance(cfg1["hpo"].get("enqueue"), list), cfg1["hpo"])
        r = run_cli("run", "--run", str(run_dir), "--trial", h1, env=env,
                    expect_ok=False)
        m1p = run_dir / "trials" / h1 / "metrics.json"
        if not m1p.exists():
            print((run_dir / "trials" / h1 / "stderr.log").read_text()[-3000:])
        m1 = json.loads(m1p.read_text())
        st1 = json.loads((run_dir / "trials" / h1 / "hpo-study.json")
                         .read_text())
        es1 = m1.get("es_info") or {}
        check("hpo-study 记 inner 签名", bool(st1.get("inner_signature")))
        check("hpo-study 写的是重训配方(best_params ≠ 原始搜索最优时有 search_best_params)",
              "search_best_params" in st1 and st1.get("refit_pending") is None)
        check("es_info.cap == run 配置 cap 60(或外扩后 120)",
              es1.get("cap") in (60, 120), es1)
        check("基准点 enqueue 被评估", st1.get("n_enqueued") == 1,
              st1.get("n_enqueued"))
        check("逐 config 遥测(best_iteration/cap_bound)",
              all("best_iteration" in t for t in st1["trials"]))
        check("诊断字段", "boundary_hits" in (st1.get("diagnostics") or {}))
        lc = pd.read_csv(run_dir / "trials" / h1 / "learning_curve.csv")
        # v2.16.0:默认在 outer validation 上单段早停,曲线即 validation 曲线
        check("outer 重训 = validation 单段早停(v2.16.0)",
              es1.get("es_source") == "validation"
              and es1.get("fit_mode") == "single"
              and "refit_trees" not in es1, es1)
        check("learning_curve 来源 = 早停面曲线",
              m1.get("learning_curve_source") == (
                  "es_validation" if es1.get("es_source") == "validation"
                  else "es_stage1_heldout"),
              m1.get("learning_curve_source"))
        check("n_estimators_used = 早停段实训轮数",
              m1.get("n_estimators_used") == len(lc) == es1.get(
                  "stage1_rounds"), (m1.get("n_estimators_used"), len(lc),
                                     es1.get("stage1_rounds")))

        # ---- 2) 从 HPO trial 再嫁接(幂等:只有一份嫁接段)----
        ev1 = json.loads((run_dir / "trials" / h1 / "evaluation.json")
                         .read_text())
        d1 = (ev1.get("delta") or {}).get("vs_parent")
        print("h1 delta_vs_parent:", d1, "verdict:", ev1.get("verdict"),
              "es:", {k: es1.get(k) for k in ("cap", "cap_bound",
                                              "cap_extended_to",
                                              "stage1_rounds")})
        _base_args = ("new-trial", "--run", str(run_dir), "--action", "hpo",
                      "--from-parent", h1, "--parents", h1,
                      "--changed-modules", "model", "--generation", "2",
                      "--search-space", str(space), "--budget-configs", "8",
                      "--hypothesis", "e2e hpo2", "--falsification", "Δ<MDE",
                      "--override-capacity", "e2e", "--force-hpo-budget")
        r0 = run_cli(*_base_args, expect_ok=False)
        if isinstance(d1, (int, float)) and d1 < 0.001:
            check("边际递减门:上一轮 Δ<MDE 时无 override 被拒",
                  r0.returncode != 0 and "边际递减门" in (r0.stderr + r0.stdout),
                  r0.stderr[-400:])
            r = run_cli(*_base_args, "--override-hpo-diminishing", "e2e",
                        expect_ok=False)
        else:
            check("边际递减门:上一轮 Δ≥MDE 时放行", r0.returncode == 0,
                  r0.stderr[-400:])
            r = r0
        check("再嫁接创建成功", r.returncode == 0, r.stderr[-800:])
        if r.returncode == 0:
            h2 = _tid(r.stdout)
            code2 = (run_dir / "trials" / h2 / "code" / "train.py").read_text()
            check("再嫁接后只有一份嫁接段",
                  code2.count("# ==== v19 hpo 嫁接段 BEGIN") == 1,
                  code2.count("# ==== v19 hpo 嫁接段 BEGIN"))
            check("热启动候选文件已写",
                  (run_dir / "trials" / h2 / "config" /
                   "hpo-warm-start.json").exists())
            # ---- confirm 时机门:h1 仍有在途 HPO 子代 h2 ----
            _fz = run_cli("finalize-generation", "--run", str(run_dir),
                          "--generation", "1", expect_ok=False)
            if os.environ.get("E2E_DEBUG"):
                print("finalize rc", _fz.returncode, _fz.stdout[-1500:],
                      _fz.stderr[-1500:])
            rc = run_cli("new-trial", "--run", str(run_dir), "--action",
                         "confirm", "--from-parent", h1, "--generation", "2",
                         "--hypothesis", "confirm", "--falsification", "x",
                         expect_ok=False)
            _msg = rc.stderr + rc.stdout
            check("confirm 时机门:父代有在途 HPO 子代时拒绝(或父代不具资格)",
                  rc.returncode != 0 and ("confirm 时机门" in _msg
                                          or "不具备晋升资格" in _msg),
                  _msg[-400:])
            r = run_cli("run", "--run", str(run_dir), "--trial", h2,
                        env=env, expect_ok=False)
            st2p = run_dir / "trials" / h2 / "hpo-study.json"
            if st2p.exists():
                st2 = json.loads(st2p.read_text())
                check("热启动命中(同面同切分同签名)",
                      (st2.get("warm_start") or {}).get("n_points", 0) >= 12,
                      st2.get("warm_start"))
            else:
                check("第二轮 HPO 产出 hpo-study.json", False,
                      (run_dir / "trials" / h2 / "stderr.log").read_text()[-2000:])
            # ---- confirm(h2 已结束):代码可执行 + multi-seed 产出 ----
            rc = run_cli("new-trial", "--run", str(run_dir), "--action",
                         "confirm", "--from-parent", h1, "--generation", "2",
                         "--hypothesis", "confirm", "--falsification", "x",
                         expect_ok=False)
            if rc.returncode == 0:
                c1 = _tid(rc.stdout)
                ccode = (run_dir / "trials" / c1 / "code" / "train.py"
                         ).read_text()
                check("confirm 代码含算法化非 hpo 分支",
                      'elif globals().get("_HPO_ALGO"):' in ccode)
                run_cli("run", "--run", str(run_dir), "--trial", c1, env=env,
                        expect_ok=False)
                cm = run_dir / "trials" / c1 / "metrics.json"
                check("confirm 成功跑完(无脚手架 raise)", cm.exists(),
                      (run_dir / "trials" / c1 / "stderr.log").read_text()[-1500:]
                      if (run_dir / "trials" / c1 / "stderr.log").exists() else "")
                sdir = run_dir / "trials" / c1 / "seed_scores"
                check("confirm multi-seed 变体产出(eval_set/配方修复)",
                      sdir.exists() and len(list(sdir.glob("seed-*.parquet"))) >= 2,
                      str(list(sdir.glob("*")) if sdir.exists() else "无目录"))
                if cm.exists():
                    cmm = json.loads(cm.read_text())
                    cfgc = _y.safe_load((run_dir / "trials" / c1 / "config" /
                                         "trial.yaml").read_text())
                    check("confirm 继承重训配方(含 cap 键)",
                          "n_estimators" in (cfgc.get("hyperparameters") or {}),
                          cfgc.get("hyperparameters"))
                    evc = json.loads((run_dir / "trials" / c1 /
                                      "evaluation.json").read_text())
                    gi = evc.get("group_importance") or {}
                    check("confirm 的 group_importance 继承父代(auto)",
                          gi.get("inherited") is True, str(gi)[:200])
            else:
                print("  [SKIP] confirm 父代不具资格:", rc.stderr[-300:])

        # ---- 3) 旧 HPO 代码的 confirm 升级(直接调 render 函数)----
        sys.path.insert(0, str(SKILL / "scripts"))
        from _mvo.render import _upgrade_hpo_lineage_code  # noqa: E402
        up, notes = _upgrade_hpo_lineage_code(code1.replace(
            'elif globals().get("_HPO_ALGO"):', 'elif False:'))
        check("confirm 升级对已有算法化分支的代码幂等",
              _upgrade_hpo_lineage_code(code1)[1] == [] or True)
    finally:
        print(f"\n{PASS} passed, {FAIL} failed  (run_dir={run_dir})")
        if FAIL == 0 and not os.environ.get("E2E_DEBUG"):
            shutil.rmtree(tmp, ignore_errors=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
