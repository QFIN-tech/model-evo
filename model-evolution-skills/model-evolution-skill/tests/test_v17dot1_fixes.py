"""v17.1 修复批次单元验证(CLI 缺省解析 + 前缀采样标签偏差闸)。

背景(2026-09-02 drawdown-t7 run 实测回写):

1. **CLI 裸调崩溃**:status/leaderboard 在 run 目录内不带 --run/
   --spec 执行时,`_run_dir_from` 直接 `Path(args.spec).parent` →
   `Path(None)` → TypeError 崩溃。SKILL.md 的命令速查标榜"在 run
   目录执行时可省",实际省了就炸。
2. **前缀采样标签偏差闸(P0 级)**:drawdown-t7 的 labels 落盘序为
   "按天排序、天内按 label 聚集"(实测单日 runs=['1'*4151,
   '0'*8238];validation 前缀 12.5% 行 100% 正样本而全体 33%,
   manifest prefix_rate_rel_dev=2.07)。v12 的防线(prefix_sample_safe
   看 ordering)对"sorted_by:pday"直接放行时间前缀——恰好把天内
   标签聚集带进子样本。受影响机制:capacity 样本量曲线(格点
   正样本率虚高→still_steep 误判)、hpo inner-split 尾部(尾部
   偏 0/1 单类→选参失真)、smoke 预检 valid 前缀(断言语义失效)、
   halving quick 轮前缀(排序失真)。修复三层:
   (a) `_sort_split_frame` 排序 key 升级 (time, within-day-shuffle)
       ——新落盘的 splits 从源头无偏;
   (b) `prefix_sample_safe` 增参 prefix_rate_rel_dev:超阈(0.2)
       降级分层随机,manifest 偏差真正被消费;
   (c) 模板 smoke/quick 轮改标签分层头(_stratified_head_labels),
       已落盘的旧 splits(无法重写)上也安全。

运行:python tests/test_v17dot1_fixes.py
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL, trial_body  # noqa: E402

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
    spec = importlib.util.spec_from_file_location(
        "modelevo_cli_v17dot1", SKILL / "scripts" / "modelevo.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


# ---------------------------------------------------------------------------
# 1. CLI 缺省解析(_run_dir_from / _load_spec)
# ---------------------------------------------------------------------------

def test_cli_default_resolution() -> None:
    print("== CLI 缺省解析 ==")
    cli = _load_cli()
    a = type("A", (), {"run": None, "spec": None})()
    # 无参数 → cwd(v17.1 前直接 Path(None) 崩溃)
    import os
    check("无 --run/--spec → cwd(不崩溃)",
          cli._run_dir_from(a) == Path.cwd())
    b = type("A", (), {"run": None, "spec": "/x/y/task-spec.yaml"})()
    check("--spec → 其父目录", cli._run_dir_from(b) == Path("/x/y"))
    c = type("A", (), {"run": "/r", "spec": None})()
    check("--run 优先", cli._run_dir_from(c) == Path("/r"))
    d = type("A", (), {"run": None, "spec": None})()
    os.environ["MODELEVO_RUN_DIR"] = "/env/run"
    try:
        check("MODELEVO_RUN_DIR 兜底", cli._run_dir_from(d) == Path("/env/run"))
    finally:
        del os.environ["MODELEVO_RUN_DIR"]


def test_load_spec_actionable_error() -> None:
    print("== _load_spec 可行动报错 ==")
    cli = _load_cli()
    a = type("A", (), {"run": None, "spec": None})()
    with tempfile.TemporaryDirectory() as td:
        import os
        old = os.getcwd()
        os.chdir(td)  # 无 task-spec.yaml 的目录
        try:
            try:
                cli._load_spec(a)
                check("spec 缺失应抛 FileNotFoundError", False)
            except FileNotFoundError as e:
                check("spec 缺失抛 FileNotFoundError 且提示用法",
                      "--spec" in str(e) or "run 目录" in str(e),
                      str(e)[:80])
            except Exception as e:  # noqa: BLE001
                check("spec 缺失抛 FileNotFoundError 且提示用法",
                      False, f"{type(e).__name__}: {e}")
        finally:
            os.chdir(old)


def test_status_leaderboard_bare_call() -> None:
    print("== status/leaderboard 裸调用(端到端)==")
    with tempfile.TemporaryDirectory() as td:
        # 最小 run 目录:task-spec + run-state + 空 trials
        (Path(td) / "trials").mkdir(parents=True)
        (Path(td) / "leaderboard").mkdir()
        (Path(td) / "task-spec.yaml").write_text(
            "task_id: t\ntask_name: t\nmodel_type: binary_classification\n"
            "target:\n  name: y\n  positive_definition: y=1\n"
            "primary_metric: auc\n"
            "split_policy:\n  type: time_based\n  train: 2025-01\n"
            "  validation: 2025-02\n  final_test: 2025-03\n",
            encoding="utf-8")
        (Path(td) / "run-state.json").write_text(
            json.dumps({"phase": "BASELINED"}), encoding="utf-8")
        for cmd in ("status", "leaderboard"):
            r = subprocess.run(
                [sys.executable, str(SKILL / "scripts" / "modelevo.py"), cmd],
                capture_output=True, text=True, cwd=td)
            check(f"{cmd} 裸调用 rc=0(修复前 TypeError 崩溃)",
                  r.returncode == 0,
                  (r.stderr or "").strip().splitlines()[-1][:100]
                  if r.returncode else "")


# ---------------------------------------------------------------------------
# 2. 前缀采样标签偏差闸
# ---------------------------------------------------------------------------

def _fake_spec():
    class S:
        time_column = "pday"
        target_name = "y"
        model_type = "binary_classification"
    return S()


def _patho_frame(n_days=6, pos=400, neg=800) -> pd.DataFrame:
    """复现 drawdown-t7 病态序:天内全部正样本在前、负样本在后。"""
    rows = []
    for d in range(n_days):
        day = f"2026-01-{d + 1:02d}"
        rows += [(day, 1)] * pos + [(day, 0)] * neg
    return pd.DataFrame(rows, columns=["pday", "y"])


def test_sort_split_frame_within_day_shuffle() -> None:
    print("== _sort_split_frame 天内洗牌 ==")
    import runtime.data as D
    df = _patho_frame()
    out, ordering = D._sort_split_frame(df, _fake_spec())
    y = out["y"].values
    n = len(y)
    check("ordering 仍报 sorted_by(时间语义保留)",
          ordering == "sorted_by:pday")
    check("跨天时间序保持单调", out["pday"].is_monotonic_increasing)
    check("天内洗牌后前缀 12.5% 正样本率 ≈ 全体",
          abs(y[:int(n * 0.125)].mean() - y.mean()) < 0.05,
          f"prefix={y[:int(n * 0.125)].mean():.3f} full={y.mean():.3f}")
    check("尾部 12.5%(hpo inner-split 口径)也 ≈ 全体",
          abs(y[int(n * 0.875):].mean() - y.mean()) < 0.05,
          f"tail={y[int(n * 0.875):].mean():.3f}")
    out2, _ = D._sort_split_frame(df, _fake_spec())
    check("确定性(同数据同序)", out["y"].equals(out2["y"]))
    # _prefix_label_check 偏差降到阈值内
    chk = D._prefix_label_check(out, _fake_spec(), ordering)
    check("manifest prefix_rate_rel_dev < 0.2",
          chk.get("prefix_rate_rel_dev", 99) < 0.2,
          str(chk.get("prefix_rate_rel_dev")))
    # 无时间列 → 原样返回
    df2 = pd.DataFrame({"y": [1, 0, 1, 0]})
    out3, ord3 = D._sort_split_frame(df2, _fake_spec())
    check("无时间列保持 v12 降级语义",
          ord3 == "original_rows_no_time_column"
          and out3["y"].tolist() == [1, 0, 1, 0])


def test_prefix_sample_safe_dev_gate() -> None:
    print("== prefix_sample_safe 偏差闸 ==")
    from runtime.hpo import PREFIX_DEV_THRESHOLD, prefix_sample_safe
    check("阈值常量存在且为 0.2", PREFIX_DEV_THRESHOLD == 0.2)
    idx = prefix_sample_safe(1000, 0.1, ordering="sorted_by:pday",
                             prefix_rate_rel_dev=0.05)
    check("低偏差(<0.2)保持时间前缀(v12 行为)",
          (idx == np.arange(100)).all())
    idx2 = prefix_sample_safe(1000, 0.1, ordering="sorted_by:pday",
                              prefix_rate_rel_dev=2.07)
    check("高偏差(2.07)降级分层随机",
          not (idx2 == np.arange(100)).all())
    idx3 = prefix_sample_safe(1000, 0.1, ordering="sorted_by:pday")
    check("偏差缺失(旧调用方)保持 v12 行为",
          (idx3 == np.arange(100)).all())
    idx4 = prefix_sample_safe(1000, 0.1, ordering=None,
                              prefix_rate_rel_dev=0.0)
    check("未排序时无论偏差都随机",
          not (idx4 == np.arange(100)).all())


def test_sample_size_curve_passes_dev() -> None:
    print("== sample_size_curve 透传偏差 ==")
    import inspect
    from runtime import capacity as C
    sig = inspect.signature(C.sample_size_curve)
    check("签名含 prefix_rate_rel_dev",
          "prefix_rate_rel_dev" in sig.parameters)
    # run_capacity_check 读 manifest 的 ordering+dev(静态检查源码行为
    # 通过函数行为验证代价高,这里验证签名与最小调用路径)
    src = (SKILL / "runtime" / "capacity.py").read_text(encoding="utf-8")
    check("run_capacity_check 读 manifest prefix_rate_rel_dev",
          "prefix_rate_rel_dev" in src and "prefix_dev" in src)


def test_stratified_head_labels() -> None:
    print("== 模板 _stratified_head_labels(标签分层头)==")
    # v2.11.9:模板 body 已是 _mvo/templates.TRIAL_BODY 模块常量,直接取
    # (旧写法从 scripts/modelevo.py 源码做字面量切片 + 手工反转义)
    body = trial_body()
    ns = {"pd": pd, "np": np, "os": __import__("os")}
    exec(compile(body, "trial_body", "exec"), ns)  # noqa: S102
    fn = ns["_stratified_head_labels"]
    # 病态序:前 30% 全正
    y = pd.Series([1] * 30 + [0] * 70)
    sel = fn(y, 50)
    check("分层头正样本率 ≈ 全体(0.30)",
          abs(y.loc[sel].mean() - 0.30) < 0.06,
          f"got {y.loc[sel].mean():.3f}")
    check("行序保持接近原序(sorted)", list(sel) == sorted(sel))
    check("k>len 时全取", len(fn(pd.Series([1, 0]), 10)) == 2)
    check("单类标签回退普通头部",
          list(fn(pd.Series([1] * 100), 10)) == list(range(10)))
    # 模板正文引用了该函数(smoke 与 quick 轮两处)
    check("模板 run_smoke 使用分层头", "run_smoke" in body
          and body.count("_stratified_head_labels(") >= 3)


def test_baseline_template_quick_stratified() -> None:
    print("== baseline 模板 quick 轮分层 ==")
    import runtime.baseline as B
    src = (SKILL / "runtime" / "baseline.py").read_text(encoding="utf-8")
    check("baseline quick 轮改为分层头(不再裸 iloc 前缀)",
          "_quick" in src and "value_counts" in src
          and "labels_tr.iloc[: len(train)]" not in src)


def test_generated_trial_code_compiles() -> None:
    print("== 生成的 trial 代码可编译 ==")
    from runtime.templates import TRIAL_TEMPLATE
    body = trial_body()                      # v2.11.9:模块常量,直接取
    code = TRIAL_TEMPLATE.format(
        trial_id="t", hypothesis="h", actions="a", run_dir="/tmp/x",
        target="y", model_type="binary_classification", treatment="",
        seed=1, skill_root="/tmp/s", body=body)
    try:
        compile(code, "train.py", "exec")
        check("TRIAL_TEMPLATE.format(body) 产物可编译", True)
    except SyntaxError as e:
        check("TRIAL_TEMPLATE.format(body) 产物可编译", False,
              f"line {e.lineno}: {e.msg}")


# ---------------------------------------------------------------------------
# 3. 文档接线(MAINTENANCE §3:改行为必须同步 digest/SKILL)
# ---------------------------------------------------------------------------

def test_docs_wiring() -> None:
    print("== 文档接线 ==")
    digest = (SKILL / "references" / "protocol-digest.md").read_text(
        encoding="utf-8")
    check("protocol-digest 提及 prefix 偏差/分层防护",
          "prefix_rate_rel_dev" in digest and "分层" in digest)
    check("protocol-digest CLI 缺省解析语义更新",
          "MODELEVO_RUN_DIR" in digest)


def main() -> int:
    test_cli_default_resolution()
    test_load_spec_actionable_error()
    test_status_leaderboard_bare_call()
    test_sort_split_frame_within_day_shuffle()
    test_prefix_sample_safe_dev_gate()
    test_sample_size_curve_passes_dev()
    test_stratified_head_labels()
    test_baseline_template_quick_stratified()
    test_generated_trial_code_compiles()
    test_docs_wiring()
    print(f"\n=== v17.1: {PASS} passed, {FAIL} failed ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
