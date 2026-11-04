"""v15.1 修复批次单元验证(fast-path NameError / lint 规则 10)。

覆盖三处修复:
1. _copy_parent_trial_code 引用了作用域不存在的 args(凡 --from-parent
   且父代 config 带 hpo 节即 NameError)→ action 改为显式入参;
2. lint_trial.py 补齐规则 10(脚手架 fit_and_score raise 标记仍在 →
   run 前静态拒绝;SKILL.md/templates.py 早已宣称该规则但实现缺失);
3. lint 规则 9a/9b/9c 在新增规则后不回归(规则 10 不误伤合规代码)。
运行:python tests/test_v15dot1_fixes.py
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def _load_cli():
    p = Path(__file__).resolve().parents[1] / "scripts" / "modelevo.py"
    spec_m = importlib.util.spec_from_file_location("mcli15", str(p))
    m = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(m)
    return m


def _load_lint():
    p = Path(__file__).resolve().parents[1] / "assets" / "lint_trial.py"
    spec_m = importlib.util.spec_from_file_location("lint15", str(p))
    m = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# 1) 快通道 NameError:父代 config 带 hpo 节,非 hpo 子代剥离不炸
# ---------------------------------------------------------------------------

def test_fastpath_hpo_strip():
    print("v15.1 快通道剥离 hpo 节(原 args NameError):")
    m = _load_cli()
    tmp = Path(tempfile.mkdtemp())
    try:
        run = tmp / "run-x"
        parent = run / "trials" / "trial-001"
        (parent / "code").mkdir(parents=True)
        (parent / "code" / "train.py").write_text("# parent code\n")
        (parent / "config").mkdir(parents=True)
        (parent / "config" / "trial.yaml").write_text(
            "trial_id: trial-001\n"
            "hyperparameters:\n  learning_rate: 0.05\n"
            "hpo:\n  search_space:\n    learning_rate: [0.01, 0.1]\n"
            "  budget_configs: 40\n")

        class Trial:
            trial_id = "trial-002"
            trial_dir = str(run / "trials" / "trial-002")
            data = {}

            def flush(self):
                pass

        t = Trial()
        # 修复前:args 未定义 → NameError;修复后:action 入参
        m._copy_parent_trial_code(run, t, "trial-001",
                                  {"learning_rate": 0.02},
                                  action="exploit")
        cfg = _yaml_load(run / "trials" / "trial-002"
                         / "config" / "trial.yaml")
        # 剥离发生在 cfg.update(notes=…) 之前,notes 被快通道默认
        # 文案覆盖——留痕瑕疵,判定只看 hpo 节缺失本身
        check("非 hpo 子代剥离 hpo 节",
              "hpo" not in cfg,
              f"hpo present: {'hpo' in cfg}")
        check("超参覆写合并生效",
              cfg.get("hyperparameters", {}).get("learning_rate") == 0.02,
              f"hp={cfg.get('hyperparameters')}")
        check("代码已复制",
              (run / "trials" / "trial-002" / "code"
               / "train.py").exists())
        # hpo 动作子代:hpo 节保留(_inject_hpo_config 随后写回)
        t2 = Trial()
        t2.trial_id = "trial-003"
        t2.trial_dir = str(run / "trials" / "trial-003")
        m._copy_parent_trial_code(run, t2, "trial-001",
                                  {"learning_rate": 0.02},
                                  action="hpo")
        cfg2 = _yaml_load(run / "trials" / "trial-003"
                          / "config" / "trial.yaml")
        check("hpo 动作子代保留 hpo 节",
              "hpo" in (cfg2 or {}), f"cfg keys={sorted(cfg2 or {})}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _yaml_load(path: Path):
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------------------
# 2) lint 规则 10:脚手架 fit_and_score 标记拦截
# ---------------------------------------------------------------------------

SCAFFOLD_CODE = '''
import lightgbm as lgb

def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None,
                  w_tr=None, w_va=None):
    raise RuntimeError(
        "脚手架默认实现已禁用(v12):Agent 未按 hypothesis 改造 "
        "fit_and_score;请实现真实模型路径后重跑(分类默认 LGBM,"
        "参考 baseline 模板;回归/uplift 分支已给可执行实现)"
        "  # lint:scaffold-fit-and-score")
'''

REAL_CODE = '''
import lightgbm as lgb

def _build_model(params, seed):
    return lgb.LGBMClassifier(n_jobs=n_jobs(), random_state=seed)

def fit_and_score(X_tr, y_tr, X_va, y_va=None, seed=0, cfg=None,
                  w_tr=None, w_va=None):
    model = _build_model({}, seed)
    model.fit(X_tr, y_tr)
    dump_model(model, X_tr.columns)
    return model.predict_proba(X_va)[:, 1], {"model_num": 1}

# leak_scan 哨兵调用(main() 路径;此处显式出现满足规则 9b)
leak_scan(_last_model, X_tr, y_tr)
'''


def test_lint_rule10():
    print("v15.1 lint 规则 10(脚手架 fit_and_score 门):")
    lt = _load_lint()
    r = lt.check_scaffold_fit_and_score(SCAFFOLD_CODE)
    check("脚手架 raise 标记被拦截",
          len(r) == 1 and "规则 10" in r[0], f"got={r}")
    r2 = lt.check_scaffold_fit_and_score(REAL_CODE)
    check("真实实现不误伤", r2 == [], f"got={r2}")
    # 标记变体(lint: scaffold-fit-and-score 空格形式)也拦
    r3 = lt.check_scaffold_fit_and_score(
        SCAFFOLD_CODE.replace("lint:scaffold-fit-and-score",
                              "lint: scaffold-fit-and-score"))
    check("空格变体标记同样拦截", len(r3) == 1, f"got={r3}")


def test_lint_registered_in_entry():
    print("v15.1 规则 10 已挂进 lint_trial 主入口:")
    lt = _load_lint()
    import inspect
    src = inspect.getsource(lt.lint_trial)
    check("lint_trial 调用 check_scaffold_fit_and_score",
          "check_scaffold_fit_and_score" in src)


def test_lint_reparse_and_rules_no_regress():
    """规则 9a/9c 与新规则共存(同一份代码不互相干扰)。"""
    print("v15.1 旧规则不回归:")
    lt = _load_lint()
    # reparse_frame 合法消费不应触发 9a(reparse 列不在 banned 集)
    code = REAL_CODE + '''

def reparse_frame(split, columns):
    return _tpl_reparse_frame(split, columns)
'''
    check("真实代码 9a 无违规",
          lt.check_banned_consumption(Path("."), code) == [])
    check("真实代码 9b 无违规",
          lt.check_leak_scan_alive(Path("."), code) == [])
    check("真实代码 9c 无违规",
          lt.check_no_direct_load_data(code) == [])
    check("真实代码规则 10 无违规",
          lt.check_scaffold_fit_and_score(code) == [])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_fastpath_hpo_strip()
    test_lint_rule10()
    test_lint_registered_in_entry()
    test_lint_reparse_and_rules_no_regress()
    print(f"\n{'='*50}")
    print(f"PASS={PASS} FAIL={FAIL}")
    sys.exit(1 if FAIL else 0)
