"""v19.3 修复批次回归测试(harness bug 修复)。

事故来源:run example-seq-v3-4-2.8.6-v19.2(2026-09-11)实测。该 run 的
`data_view` 动作**完全不可用**(任何视图含 no-op 都落不了盘),且 hpo
嫁接路径首次 run 必炸。本测试锁定五处修复。

1. Bug#1  `cmd_write_splits_view` 的函数作用域陷阱
   —— `SPLIT_NAMES` 在函数前段引用、后段 import → UnboundLocalError
   被 `except Exception` 吞 → 骨架占位注入静默失效。
2. Bug#2  `view_templates` 用 `globals().get()` 读 `build_view` 的
   局部变量 → 永远取不到;保护列为空的 recency 视图会把标签列加权成
   `w30_<target>`(train 单变量 AUC=1.0,实测复现)。
3. Bug#3  CLI 拼回 labels+reparse 时对齐键(pday/user_no)重复 →
   `write_view_splits` 的 ID 守卫抛 broadcast 错
   `(300000,) (300000,2)`,即 gen-03 §10.1 上报的那条。
4. Bug#4  `_graft_hpo_section` 嫁接段缺 SKILL_ROOT/`_HPO_ALGO`/依赖
   helper,并保留 lightgbm import(lint 规则 2 拒 run)。
5. Bug#5  `classify_failure` 把项目内 `ModuleNotFoundError` 归为
   infra_error → 无效重试烧全量训练。

另:新增 lint 规则 14(函数内先用后导),覆盖 Bug#1 这类静态漏检。
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import pathlib
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import SKILL, tmp_run_dir, cli_source  # noqa: E402
from _fixture import find_run, slim_run  # noqa: E402

sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL / "assets"))

# ---------------------------------------------------------------------------
# v2.11.9 自包含化:此前 RUN 直接指向真实示例 run(5.5GB),本文件的
# `write_view_splits(..., RUN, "v193-t3")` **往用户数据里写盘并 rmtree
# 清理**,并发跑两份或在 run 被占用时 flaky(已实测复现)。
# 现在改为:示例 run 相对 skill 根定位 → 瘦身成 ~15MB 临时夹具 → 所有
# 读写都落在夹具上;示例 run 缺失时显式跳过。
# ---------------------------------------------------------------------------
_SRC = find_run("example-seq-v3-4-2.8.6-v19.2",
                "example-seq-v3-4-2.8.6-v17.4-v4-flash/run-req-t7-drawdown")
_TMP: pathlib.Path | None = None
RUN: pathlib.Path | None = None


def _ensure_run() -> pathlib.Path | None:
    """惰性构建瘦身夹具(只在真正需要 run 数据的用例里触发)。"""
    global _TMP, RUN
    if RUN is not None:
        return RUN
    if _SRC is None:
        return None
    _TMP = tmp_run_dir("v19dot3-")
    RUN = slim_run(_SRC, _TMP / "run")
    return RUN

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))


def _load_view(src: str):
    p = pathlib.Path(tempfile.mkdtemp()) / "v.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("vt", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Bug#1:函数作用域陷阱
# ---------------------------------------------------------------------------
def test_scope_trap() -> None:
    print("== 1. cmd_write_splits_view 作用域陷阱(Bug#1)==")
    src = cli_source()
    tree = ast.parse(src)
    fnode = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef)
                 and n.name == "cmd_write_splits_view")
    binds, uses = [], []
    for n in ast.walk(fnode):
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if (a.asname or a.name) == "SPLIT_NAMES":
                    binds.append(n.lineno)
        if isinstance(n, ast.Name) and n.id == "SPLIT_NAMES":
            (uses if isinstance(n.ctx, ast.Load) else binds).append(n.lineno)
    check("SPLIT_NAMES 绑定先于全部引用(UnboundLocalError 根因消除)",
          bool(binds) and all(min(binds) < u for u in uses),
          f"bind@{sorted(binds)} uses@{sorted(uses)}")
    _n_import = len(re.findall(
        r"^\s*from runtime\.data import \(SPLIT_NAMES", src, re.M))
    check("函数内不再重复 import SPLIT_NAMES(注释不算)",
          _n_import == 1, f"实际 import 语句 {_n_import} 条")
    # 占位注入失败必须 abort(不再"保持占位回退")
    check("骨架注入失败不再静默回落",
          "保持占位回退" not in src
          and "拒绝生成无防护的视图骨架" in src)


# ---------------------------------------------------------------------------
# 2. Bug#2:占位读取 + 泄漏硬防线
# ---------------------------------------------------------------------------
def test_view_templates() -> None:
    print("== 2. view_templates 占位读取与泄漏防线(Bug#2)==")
    from runtime.view_templates import render_view_template

    vsrc = (SKILL / "runtime/view_templates.py").read_text()
    check("不再用 globals().get 读局部占位",
          'globals().get("_SPLIT_ROWS")' not in vsrc
          and 'globals().get("_PROTECTED_COLS")' not in vsrc)

    def render(prot, rows, tpl="recency", **kw):
        s = render_view_template(tpl, view_id="t", half_life=30.0,
                                 k_periods=3,
                                 active_col=kw.get("active_col"),
                                 time_column="pday", target_name="y")
        s = s.replace("_SPLIT_ROWS = []  # @SPLIT_ROWS@",
                      f"_SPLIT_ROWS = {rows}  # @SPLIT_ROWS@")
        s = s.replace("_PROTECTED_COLS = []  # @PROTECTED_COLS@",
                      f"_PROTECTED_COLS = {prot}  # @PROTECTED_COLS@")
        return _load_view(s)

    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"pday": pd.date_range("2026-01-01", periods=100),
                       "x": np.arange(100.0),
                       "loan_days": np.arange(100) % 7,
                       "y": np.random.RandomState(0).randint(0, 2, 100)})

    for tpl, kw in (("recency", {}), ("active", {"active_col": "loan_days"})):
        try:
            render("[]", "[]", tpl, **kw).build_view(df)
            check(f"{tpl} 保护列为空时拒绝生成", False, "未 raise")
        except ValueError as e:
            check(f"{tpl} 保护列为空时拒绝生成", "保护列为空" in str(e))

    out = render("['y', 'pday']", "[60, 40]").build_view(df)
    added = [c for c in out.columns if c not in df.columns]
    check("recency 加权列不携带标签", not [c for c in added if c.endswith("_y")],
          str(added))
    check("_SPLIT_ROWS 被真实读取(逐段参考日生效)",
          abs(float(out["w30_x"].iloc[59] / df["x"].iloc[59]) - 1.0) < 1e-9)

    out2 = render("['y']", "[60, 40]", "active",
                  active_col="loan_days").build_view(df)
    check("active 加权列不携带标签",
          not [c for c in out2.columns if c == "act_y"])


# ---------------------------------------------------------------------------
# 3. Bug#3:重名列 / broadcast
# ---------------------------------------------------------------------------
def test_write_view_splits() -> None:
    print("== 3. write_view_splits 重名列与产物泄漏断言(Bug#3)==")
    import pandas as pd
    from runtime.data import (SPLIT_NAMES, load_labels, load_split_frames,
                              write_view_splits)
    from runtime.spec import TaskSpec

    dsrc = (SKILL / "runtime/data.py").read_text()
    check("ID 守卫含重名列拒绝", "视图输入必须列名单义" in dsrc)
    check("产物级派生列泄漏断言存在", "携带保护列" in dsrc)
    check("ID 守卫含形状防御(不再裸 broadcast)",
          "pv.ndim != 1" in dsrc)

    if _ensure_run() is None:
        check("write_view_splits 行为面(需示例 run)", True,
              "SKIP: 无示例 run")
        return
    spec = TaskSpec.load(RUN / "task-spec.yaml")
    frames = load_split_frames(RUN)
    labels = load_labels(RUN)
    rp = RUN / "sample-profile/splits/reparse"
    repar = {n: pd.read_parquet(rp / f"{n}.parquet")
             for n in SPLIT_NAMES if (rp / f"{n}.parquet").exists()}
    parts = []
    for n in SPLIT_NAMES:
        if n in frames and n in labels:
            cols = [frames[n].reset_index(drop=True),
                    labels[n].reset_index(drop=True)]
            if n in repar:
                cols.append(repar[n].reset_index(drop=True))
            parts.append(pd.concat(cols, axis=1))
    dirty = pd.concat(parts, ignore_index=True, copy=False)
    check("复现前提:拼回后存在重名列", dirty.columns.duplicated().any(),
          str(sorted(set(dirty.columns[dirty.columns.duplicated()]))))

    clean = dirty.loc[:, ~dirty.columns.duplicated(keep="first")].copy()
    try:
        m = write_view_splits(clean, spec, RUN, "v193-t3",
                              note="regression")
        check("去重后落盘成功(旧实现抛 broadcast)", True,
              f"n_features={m.get('n_features')}")
    except Exception as e:  # noqa: BLE001
        check("去重后落盘成功(旧实现抛 broadcast)", False,
              f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(RUN / "sample-profile/splits-v193-t3",
                      ignore_errors=True)

    try:
        write_view_splits(dirty, spec, RUN, "v193-t3b")
        check("脏帧被显式拒绝", False, "未 raise")
    except ValueError as e:
        check("脏帧被显式拒绝(不再是 broadcast 崩)", "重名列" in str(e))
    finally:
        shutil.rmtree(RUN / "sample-profile/splits-v193-t3b",
                      ignore_errors=True)

    leaky = clean.copy()
    leaky["w30_label_req_t7"] = leaky["label_req_t7"] * 0.5
    try:
        write_view_splits(leaky, spec, RUN, "v193-t3c")
        check("派生列携带 target 被拒绝", False, "未 raise")
    except ValueError as e:
        check("派生列携带 target 被拒绝", "携带保护列" in str(e))
    finally:
        shutil.rmtree(RUN / "sample-profile/splits-v193-t3c",
                      ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. Bug#4:hpo 嫁接
# ---------------------------------------------------------------------------
def test_graft() -> None:
    print("== 4. hpo 嫁接路径(Bug#4)==")
    import logging

    import modelevo as M
    import lint_trial as L

    if _ensure_run() is None:
        check("hpo 嫁接路径(需示例 run)", True, "SKIP: 无示例 run")
        return
    parent = (RUN / "trials/trial-004-g02-xgboost-1821/code/train.py"
              ).read_text()
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "code").mkdir()
    (tmp / "code" / "train.py").write_text(parent)

    class FakeTrial:
        trial_dir = str(tmp)
        trial_id = "trial-900-g09-xgboost-9999"
        data = {"algorithm": "xgboost"}

    logging.disable(logging.CRITICAL)
    M._graft_hpo_section(FakeTrial())
    g = (tmp / "code" / "train.py").read_text()

    # v2.10.1:注入用的是 `import sys as _sys_graft` 别名(v19.4 起),
    # 旧断言只认字面 `sys.path.insert`,在实现正确时永久假报警。
    check("嫁接段注入 SKILL_ROOT + sys.path",
          "SKILL_ROOT" in g
          and re.search(r"(?:sys|_sys_graft)\.path\.insert", g) is not None)
    check("嫁接段注入模块级 _HPO_ALGO",
          bool(re.search(r"^_HPO_ALGO = ", g, re.M)),
          re.findall(r"^_HPO_ALGO = .*", g, re.M) or "缺失")
    for dep in ("_lgbm_fit_safe", "_gpu_allocated", "n_jobs"):
        check(f"嫁接段搬运依赖 {dep}", f"def {dep}(" in g)
    check("搬运 _GPU_MIN_ROWS 常量", "_GPU_MIN_ROWS = " in g)
    check("lint 规则 2(双 GBM 库)不触发",
          len(L.GBM_LIBS & L.scan_imports(g)) < 2,
          str(sorted(L.GBM_LIBS & L.scan_imports(g))))
    check("lint 规则 1(算法一致)",
          not L.check_algorithm_consistency("trial-900-g09-xgboost-9999", g))
    try:
        ast.parse(g)
        check("嫁接产物语法正确", True)
    except SyntaxError as e:
        check("嫁接产物语法正确", False, str(e))
    # 幂等
    M._graft_hpo_section(FakeTrial())
    g2 = (tmp / "code" / "train.py").read_text()
    check("嫁接幂等", g2.count("SKILL_ROOT = ") == 1
          and len(re.findall(r"^_HPO_ALGO = ", g2, re.M)) == 1)
    shutil.rmtree(tmp, ignore_errors=True)

    check("统一注入入口 _inject_hpo_algo_at 存在",
          hasattr(M, "_inject_hpo_algo_at"))
    r = M._inject_hpo_algo_at("SEED = 20260814\n", "catboost")
    check("注入实现正确", r is not None and "_HPO_ALGO = 'catboost'" in r)
    check("AST 判定的幂等(注释里的字样不算)",
          M._inject_hpo_algo_at(
              'SEED = 20260814\n# _HPO_ALGO 说明\n', "xgboost") is not None)
    check("已注入时幂等返回 None",
          M._inject_hpo_algo_at(r, "catboost") is None)


# ---------------------------------------------------------------------------
# 5. Bug#5:失败分类
# ---------------------------------------------------------------------------
def test_classify_failure() -> None:
    print("== 5. classify_failure 项目内 ImportError 归类(Bug#5)==")
    from runtime.scheduler import classify_failure as cf
    cases = [
        ("ModuleNotFoundError: No module named 'runtime'", "code_error"),
        ("ModuleNotFoundError: No module named 'tools'", "code_error"),
        ("ModuleNotFoundError: No module named 'xgboost'", "infra_error"),
        ("ModuleNotFoundError: No module named 'optuna'", "infra_error"),
        ("ConnectionError: refused", "infra_error"),
        ("ValueError: x", "code_error"),
    ]
    for tail, want in cases:
        got = cf(1, tail)
        check(f"归类 {want} <- {tail[:44]}", got == want, f"got {got}")
    check("重试集合仍含 infra/oom/timeout",
          set(("timeout", "oom", "infra_error"))
          <= set(__import__("runtime.scheduler", fromlist=["x"]
                            ).RETRYABLE_FAILURES))


# ---------------------------------------------------------------------------
# 6. lint 规则 14
# ---------------------------------------------------------------------------
def test_lint_rule14() -> None:
    print("== 6. lint 规则 14(函数内先用后导)==")
    import lint_trial as L
    A = ("import json\n\ndef cmd(a):\n"
         "    x = [len({}) for n in SPLIT_NAMES]\n"
         "    return x\n"
         "    from runtime.data import SPLIT_NAMES  # noqa\n")
    check("抓到函数内先用后导", bool(L.check_function_scope_shadowing(A)))
    check("合法 module-level import 在后不误报",
          not L.check_function_scope_shadowing(
              "def cmd():\n    return SPLIT_NAMES\n"
              "from runtime.data import SPLIT_NAMES\n"))
    C = ("def f(a):\n    total = 0\n    for i in range(3):\n"
         "        total += i * a\n    try:\n        pass\n"
         "    except ValueError as e:\n        print(e)\n"
         "    with open('x') as fh:\n        print(fh)\n"
         "    g = lambda z: z + 1\n"
         "    d = {k: v for k, v in enumerate([1, 2])}\n"
         "    return total, g(1), d\n")
    check("合法写法(循环/with/except/lambda/comprehension)不误报",
          not L.check_function_scope_shadowing(C))
    # 真实 trial 全量回归:零误报(这些 trial 都真跑通过)
    if _ensure_run() is None:
        check("真实 trial 零误报(需示例 run)", True, "SKIP: 无示例 run")
        return
    fp = 0
    for t in sorted((RUN / "trials").iterdir()):
        cp = t / "code" / "train.py"
        if cp.exists():
            fp += len(L.check_function_scope_shadowing(cp.read_text()))
    check("14 个真实 trial 零误报(Bug 修复前的 lint 空白)", fp == 0,
          f"误报 {fp}")
    check("规则 14 接入 lint_trial 聚合",
          "check_function_scope_shadowing(code)" in
          (SKILL / "assets/lint_trial.py").read_text())


def main() -> int:
    try:
        for t in (test_scope_trap, test_view_templates, test_write_view_splits,
                  test_graft, test_classify_failure, test_lint_rule14):
            t()
            print()
    finally:
        if _TMP is not None:
            shutil.rmtree(_TMP, ignore_errors=True)
    print("=" * 60)
    print(f"v19.3 fixes: {len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
