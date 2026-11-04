"""ModelEvo v2.11 回归测试:两 run(v2.10.1 run-drawdown-req-t7-v2101 完成 +
example/run-drawdown-req-t7-6w 在途)执行历史回写的 6 个 harness 缺陷/浪费。

证据基线(2026-09-14):
  W1 加宽面重复计算   → 一个 run 13~15 次全量 _widen(单次 830~1000s),
                         占 compute 55%+;trial-004 单 trial 87% 时间在 widen
  W2 模板自重复       → 渲染产物含 2 个 load_config / 2 个 dump_eval_frame
                         (trial-003/008 AST 实测);二分类 trial 携带重复的
                         DummyRegressor 分支
  G1 线程下限缺失     → 高负载期 scheduler 把 threads 压到 4,24.6 万行
                         smoke 跑 1401s,trial-007 attempt-1 白烧 1.5h
  G2 孤儿 RUNNING 盲区 → 心跳 19:58 消失,20:28 才被重入检测兜住(~30min)
  G3 冠军单源校验滞后 → recipe 声明 best_trial=trial-004、evaluate-final
                         选型 trial-002,首跑失败(口径分裂在 evaluate-final
                         才暴露,retrain 产生它时不校验)
  S1 代间空档         → gen-02 结束→gen-03 启动 41min、setup→首代 77min
                         无 trial 在跑(reflection/planner/写码全串行)
  S2 widen 缓存无门禁 → lint 对"裸 _widen"零约束,浪费只有事后归因才知道

跑法:python tests/test_v211_fixes.py [--skill-root <path>]
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
import time
from pathlib import Path

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from _common import SKILL, trial_body, cli_source  # noqa: E402

#: v2.11.9:模块级默认值,pytest 直收本文件时 main() 不会执行,
#: 旧版把 SKILL 留成 main() 里的 global 赋值 → NameError。
SKILL_PARENT = SKILL.parent

PASS = 0
FAIL = 0
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}" + (f"  -- {detail}" if detail else ""))
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"  [FAIL] {name}" + (f"  -- {detail}" if detail else ""))


def _load_body_template(cli_src: str) -> str:
    """按 harness 同款锚点抽取渲染 body 字面量(与 _graft_hpo_section 一致)。"""
    # v2.11.9:模板已是 _mvo/templates.TRIAL_BODY 模块常量,直接取。
    # 旧实现从 scripts/modelevo.py 源码找赋值锚点 + ast.literal_eval
    # 还原字面量;模板抽出后源码里已无该字面量。
    del cli_src                              # 保持旧签名(调用方仍传)
    return trial_body()


def _render(cli_src: str, model_type: str = "binary_classification") -> str:
    from runtime.templates import TRIAL_TEMPLATE
    body = _load_body_template(cli_src)
    return TRIAL_TEMPLATE.format(
        trial_id="trial-T", hypothesis="h", actions="a",
        run_dir="/tmp/run", target="y", model_type=model_type,
        treatment="", seed=1, skill_root="/tmp/s", body=body)


# ---------------------------------------------------------------------------
# W2 模板去重:渲染产物无重复顶层定义
# ---------------------------------------------------------------------------
def test_template_no_duplicate_defs() -> None:
    print("== W2 模板去重(渲染产物 AST 级无重复 def)==")
    root = SKILL
    cli_src = cli_source()
    for mt in ("binary_classification", "regression", "uplift"):
        code = _render(cli_src, mt)
        tree = ast.parse(code)
        fns = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
        dup = sorted({f for f in fns if fns.count(f) > 1})
        check(f"model_type={mt} 无重复顶层 def", not dup, str(dup))
        try:
            compile(code, "train.py", "exec")
            check(f"model_type={mt} 渲染产物可编译", True)
        except SyntaxError as e:
            check(f"model_type={mt} 渲染产物可编译", False,
                  f"line {e.lineno}: {e.msg}")
    # 回归 Dummy 分支归并后仅一处构造点(旧版同一函数内出现两次,
    # 二分类 trial 白携带一个永不触达的 elif 分支)
    body = _load_body_template(cli_src)
    check("回归 Dummy 分支仅一处构造点",
          body.count('model = DummyRegressor(strategy="median")') == 1)
    # 分类渲染产物中 DummyRegressor 只出现在未触达的 elif 分支
    # (可达性):binary 主分支构造点为 0
    code_bin = _render(cli_src, "binary_classification")
    _bin_calls = [
        n for n in ast.walk(ast.parse(code_bin))
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") ==
        "DummyRegressor"
    ]
    check("分类 trial 的 DummyRegressor 构造仅剩回归 elif 一处",
          len(_bin_calls) == 1, f"calls={len(_bin_calls)}")
    # hpo 嫁接锚点仍然完好(模板结构变更 = 嫁接 SystemExit)
    for anchor in ("\ndef dump_json(", "\ndef step(",
                   "\ndef fit_and_score(", "\n\n\ndef dump_eval_frame("):
        check(f"嫁接锚点存在 {anchor.strip()}", anchor in body)


# ---------------------------------------------------------------------------
# W1/S2 widen_cached:模板内置 + 缓存语义 + lint 规则 15
# ---------------------------------------------------------------------------
def test_widen_cached() -> None:
    print("== W1 widen_cached(模板内置缓存)==")
    import os as _os
    import pandas as pd
    cli_src = cli_source()
    body = _load_body_template(cli_src)
    for sym in ("def widen_cached(", "def _widen_face_key(",
                "_WIDEN_MEMO", "MODELEVO_WIDEN_CACHE"):
        check(f"body 含 {sym}", sym in body)
    # 缓存语义(独立环境执行:TRIAL_DIR/RUN_DIR/stub _widen)
    #
    # v2.11.9:落盘缓存目录改为**本进程独占的临时目录**。此前写死
    # `/tmp/_widen_cache_test_run` 并在 main() 开头 rmtree,并发跑两份
    # 本文件时,A 的 rmtree 会删掉 B 正在断言的缓存 → 随机失败
    # (已实测复现:54 passed/1 failed 与 55 passed/0 failed 并发出现)。
    import tempfile as _tempfile
    run_dir = Path(_tempfile.mkdtemp(prefix="widen-cache-"))
    ns: dict = {}
    prelude = """
import os, pathlib, json
import numpy as np, pandas as pd
RUN_DIR = pathlib.Path('__RUN_DIR__')
TRIAL_DIR = RUN_DIR / 'trials' / 'trial-T'
WIDEN_CALLS = {'n': 0}
_WIDEN_MEMO = {}
def _widen(raw_tr, re_tr, raw_va, re_va):
    WIDEN_CALLS['n'] += 1
    X = pd.DataFrame({'a': [1.0] * len(raw_tr), 'w': [2.0] * len(raw_tr)})
    V = pd.DataFrame({'a': [1.0] * len(raw_va), 'w': [2.0] * len(raw_va)})
    return X, V, {'k': 1}, {'widened_n_cols': 2}
def step(msg):
    pass
"""
    body_exec = body[body.find("def _widen_face_key"):
                     body.find("def _stratified_head_labels")]
    exec(prelude.replace("__RUN_DIR__", str(run_dir)) + body_exec, ns)
    # 同口径二次调用 → memo 命中,_widen 只执行一次
    df1 = pd.DataFrame({"c1": [1, 2, 3]})
    df2 = pd.DataFrame({"c1": [4, 5, 6]})
    r1 = ns["widen_cached"]("main", df1, df2, df2, df1)
    r2 = ns["widen_cached"]("main", df1, df2, df2, df1)
    check("同进程同口径 → memo 命中(单次计算)",
          ns["WIDEN_CALLS"]["n"] == 1, f"calls={ns['WIDEN_CALLS']['n']}")
    check("memo 返回同一对象", r1[0] is r2[0])
    # 口径变化(列集不同)→ 重算
    df3 = pd.DataFrame({"c1": [1], "c2": [9]})
    ns["widen_cached"]("main", df3, df2, df2, df1)
    check("口径变化 → 重算", ns["WIDEN_CALLS"]["n"] == 2,
          f"calls={ns['WIDEN_CALLS']['n']}")
    # 全量轮落盘缓存:清 memo 后仍命中 disk
    import pathlib as _pl
    RUN_DIR = run_dir
    r3 = ns["widen_cached"]("main", df1, df2, df2, df1)
    check("全量轮写落盘缓存",
          (RUN_DIR / ".widen-cache").exists()
          and any((RUN_DIR / ".widen-cache").glob("*.tr.parquet")),
          str(list((RUN_DIR / '.widen-cache').iterdir())
              if (RUN_DIR / '.widen-cache').exists() else 'no dir'))
    ns["_WIDEN_MEMO"].clear()
    ns["WIDEN_CALLS"]["n"] = 0
    r4 = ns["widen_cached"]("main", df1, df2, df2, df1)
    check("清 memo 后 → disk 命中(零计算)", ns["WIDEN_CALLS"]["n"] == 0,
          f"calls={ns['WIDEN_CALLS']['n']}")
    check("disk 命中带 widen_cache=disk 标记",
          r4[3].get("widen_cache") == "disk", str(r4[3]))
    # smoke/小样本不落盘(目录里已有全量缓存,断言文件集不增长)
    _before = {p.name for p in (RUN_DIR / ".widen-cache").iterdir()}
    ns2: dict = {}
    exec(prelude.replace("__RUN_DIR__", str(run_dir)) + body_exec, ns2)
    ns2["widen_cached"]("smoke", df1, df2, df2, df1, subsample_rows=100)
    _after = {p.name for p in (RUN_DIR / ".widen-cache").iterdir()}
    check("subsample 轮不写落盘缓存", _after == _before,
          f"新增={_after - _before}(smoke 帧不得覆盖全量缓存)")
    # kill switch
    _os.environ["MODELEVO_WIDEN_CACHE"] = "0"
    ns3: dict = {}
    exec(prelude.replace("__RUN_DIR__", str(run_dir)) + body_exec, ns3)
    ns3["widen_cached"]("main", df1, df2, df2, df1)
    ns3["widen_cached"]("main", df1, df2, df2, df1)
    _os.environ.pop("MODELEVO_WIDEN_CACHE")
    check("MODELEVO_WIDEN_CACHE=0 → 直算(memo 关闭)",
          ns3["WIDEN_CALLS"]["n"] == 2, f"calls={ns3['WIDEN_CALLS']['n']}")
    # lint 规则 15
    sys.path.insert(0, str(SKILL / "assets"))
    import lint_trial as m
    code_bad = ("import pandas as pd\n"
                "def _widen(a, b, c, d):\n    return a, b, {}, {}\n"
                "X = _widen(a, b, c, d)\n")
    check("裸 _widen → 规则 15 拒绝",
          any("规则 15" in e for e in m.check_widen_cache(code_bad)))
    check("widen_cached 调用 → 通过",
          not m.check_widen_cache(
              code_bad.replace("_widen(a, b, c, d)",
                               "widen_cached('main', a, b, c, d)")))
    check("豁免标记 lint:skip-widen-cache 生效",
          not m.check_widen_cache(code_bad + "# lint:skip-widen-cache\n"))
    check("无加宽代码不触发", not m.check_widen_cache("import pandas\nx=1\n"))
    # 模板/baseline 脚手架不误报
    code = _render(cli_src)
    check("分类脚手架不误报规则 15", not m.check_widen_cache(code))
    from runtime.baseline import BASELINE_TEMPLATE
    bc = BASELINE_TEMPLATE.format(
        algorithm="lightgbm", target="y", model_type="binary_classification",
        treatment="", seed=1, n_estimators_cap=100, early_stopping_rounds=50,
        es_fraction=0.1, estimator_code="", fit_code="")
    check("baseline 脚手架不误报规则 15", not m.check_widen_cache(bc))
    # lint_trial 总入口接线
    src_lint = (SKILL / "assets" / "lint_trial.py").read_text(encoding="utf-8")
    check("lint_trial 总入口已挂 check_widen_cache",
          "errors.extend(check_widen_cache(code))" in src_lint)


# ---------------------------------------------------------------------------
# G1 线程下限
# ---------------------------------------------------------------------------
def test_threads_floor() -> None:
    print("== G1 线程下限(高负载不再压到 4)==")
    from runtime.config import Config
    from runtime.scheduler import Scheduler, ResourcePlan
    from runtime.runner import TrialRunner
    cfg = Config.load(skill_config=False)
    check("DEFAULTS 有 threads_floor",
          int(cfg.get("execution.threads_floor", 0)) == 8)
    snap = {"cpu_physical": 64, "cpu_count": 128, "load1": 100.0,
            "ram_available_gb": 500, "gpus": []}
    plan = Scheduler(cfg, snapshot=snap).plan(planned_trials=2)
    check("高负载(usable=13)threads ≥ 8",
          plan.threads_per_trial >= 8, f"threads={plan.threads_per_trial}")
    snap_ok = {"cpu_physical": 64, "cpu_count": 128, "load1": 0.5,
               "ram_available_gb": 500, "gpus": []}
    plan_ok = Scheduler(cfg, snapshot=snap_ok).plan(planned_trials=4)
    check("正常负载行为不变(16×4)",
          plan_ok.threads_per_trial == 16 and plan_ok.n_parallel_trials == 4,
          f"{plan_ok.threads_per_trial}×{plan_ok.n_parallel_trials}")
    r = TrialRunner.__new__(TrialRunner)
    r.config = cfg
    d = r._downgrade_plan(ResourcePlan(
        n_parallel_trials=1, threads_per_trial=24, gpu_index=0, env={}))
    check("降配 24 → 12(减半,不低于 floor)", d.threads_per_trial == 12)
    d2 = r._downgrade_plan(ResourcePlan(
        n_parallel_trials=1, threads_per_trial=8, gpu_index=0, env={}))
    check("降配 8 → 8(floor 生效,不再对半到 4)",
          d2.threads_per_trial == 8)
    # global-config 对齐
    gc = (SKILL / "assets" / "global-config.yaml").read_text(encoding="utf-8")
    check("global-config.yaml 声明 threads_floor", "threads_floor" in gc)


# ---------------------------------------------------------------------------
# G2 孤儿 RUNNING 快速检测
# ---------------------------------------------------------------------------
def test_stale_running(tmp: "Path | None" = None) -> None:
    # v2.11.9:形参 `tmp` 会被 pytest 当成 fixture 而报错(setup error)。
    # 给默认值后 pytest 不再解析它,直跑路径(显式传参)语义不变。
    if tmp is None:
        import tempfile as _tf
        tmp = Path(_tf.mkdtemp(prefix="stale-running-"))
    print("== G2 孤儿 RUNNING 检测(diagnose finding + wait exit 4 + 探针)==")
    from runtime.runner import TrialRunner
    r = TrialRunner.__new__(TrialRunner)

    def make_trial(status: str, heartbeat_age: float | None,
                   lock_pid: str | None) -> object:
        tdir = tmp / f"t-{status}-{heartbeat_age}-{lock_pid}"
        (tdir / "code").mkdir(parents=True, exist_ok=True)
        if lock_pid:
            (tdir / ".runner.lock").write_text(lock_pid + "\n")
        if heartbeat_age is not None:
            ts = time.time() - heartbeat_age
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S+08:00",
                                  time.localtime(ts))
            (tdir / "trial.log").write_text(
                f"[{stamp}] heartbeat: runtime=10s cpu=100% mem=2.0GB\n")
        return type("T", (), {"status": status, "trial_dir": tdir})()

    # 活锁 → 不判死(评估段不写心跳但进程活)。锁 pid 用本进程:
    # os.kill(self,0) 必活,跨用户/复用 pid 的不确定性不影响断言。
    import os as _os2
    live = make_trial("RUNNING", 9999.0, str(_os2.getpid()))
    s, why = r.is_stale_running(live)
    check("锁进程活 → 不判死", not s, why)
    # 锁死 + 心跳陈旧 → 判死(4194303 = pid_max-1,必然无进程)
    dead = make_trial("RUNNING", 600.0, "4194303")
    s, why = r.is_stale_running(dead)
    check("锁死 + 心跳 600s → 判死", s, why)
    # 锁死 + 心跳新鲜 → 不判死
    fresh = make_trial("RUNNING", 30.0, "4194303")
    s, why = r.is_stale_running(fresh)
    check("锁死 + 心跳 30s(<5min)→ 不判死(防误杀)",
          not s, why)
    # 非 RUNNING → 无关
    s, _ = r.is_stale_running(make_trial("EVALUATED", None, None))
    check("非 RUNNING → False", not s)
    # 无锁无日志(从未起跑)→ 判死
    s, why = r.is_stale_running(make_trial("RUNNING", None, None))
    check("无锁无心跳 → 判死", s, why)
    # CLI: diagnose 的 stale_running finding
    cli_src = cli_source()
    check("diagnose 已挂 stale_running finding",
          '"code": "stale_running"' in cli_src)
    check("cmd_wait 已接 STALE_RUNNING(exit 4)",
          "STALE_RUNNING" in cli_src and "return 4" in cli_src)
    _m = re.search(r'"wait",.*?set_defaults\(func=cmd_wait\)', cli_src, re.S)
    check("wait parser help 提及 exit 4",
          _m is not None and "STALE_RUNNING" in _m.group(0))


# ---------------------------------------------------------------------------
# G3 冠军单源校验前置
# ---------------------------------------------------------------------------
def test_champion_check_position() -> None:
    print("== G3 冠军单源校验前置到 retrain 完成时刻 ==")
    cli_src = cli_source()
    i_retrain = cli_src.find("def cmd_finalize_retrain(")
    i_baseline = cli_src.find("def cmd_finalize_baseline_score(")
    body = cli_src[i_retrain:i_baseline]
    check("retrain 期末复核存在",
          "冠军单源校验失败(retrain 期末复核)" in body)
    # 复核在 RETRAIN_ALL_DONE 之前(fail-fast,不写完产物才校验)
    check("复核先于 RETRAIN_ALL_DONE 打印",
          body.find("冠军单源校验失败") < body.find("RETRAIN_ALL_DONE"))
    check("--champion 覆写路径跳过复核(人工入口已留痕)",
          "if not getattr(args, \"champion\", None):" in body)


# ---------------------------------------------------------------------------
# S1 代间流水线 steering(digest 接线)
# ---------------------------------------------------------------------------
def test_digest_steering() -> None:
    print("== S1 digest 代间流水线 steering ==")
    digest = (SKILL / "references" / "protocol-digest.md").read_text(
        encoding="utf-8")
    check("digest 有代间流水线节", "代间流水线" in digest)
    check("digest 有 STALE_RUNNING 出口说明", "STALE_RUNNING" in digest)
    check("预起草下代纪律(上代 barrier 期间)",
          "预起草" in digest and "预案分支" in digest)
    check("confirmation 硬依赖串行未被流水线化",
          "confirmation" in digest and "硬" in digest)


# ---------------------------------------------------------------------------
# 模板瘦身幅度(防回归:去重后不得回涨)
# ---------------------------------------------------------------------------
def test_template_slim() -> None:
    print("== 模板瘦身幅度 ==")
    cli_src = cli_source()
    code = _render(cli_src)
    n = len(code.splitlines())
    # 去重(−39 行)与 widen_cached(+93 行)合并后 1085 行;基线 1015。
    # v2.11.6:reparse_frame 零列短路(+7 行,无 reparse 任务的交付链
    # 修复)后 1092 行 → 上限 1130。
    # 断言去重未回涨:重复 def 存在时行数会更多(上限 = 基线 + 缓存块)。
    check("渲染产物 ≤ 1130 行(基线 1015 + 缓存 93 − 去重 39 + v2.11.6 短路 7)",
          n <= 1130, f"lines={n}")
    check("回归分支保留(2 元组契约)",
          code.count("DummyRegressor(strategy=\"median\")") == 1)
    uplift = _render(cli_src, "uplift")
    check("uplift trial 保留 T-learner 分支",
          "T-learner" in uplift)


def main() -> int:
    global SKILL
    ap = argparse.ArgumentParser()
    # v2.11.9:默认值由 _common 推导(本文件位置),不再写死本机绝对路径
    ap.add_argument("--skill-root", default=str(SKILL))
    a = ap.parse_args()
    SKILL = Path(a.skill_root).resolve()

    import shutil
    import tempfile
    # v2.11.9:widen-cache 落盘目录现在由 test_widen_cached 内部
    # 用 tempfile 独占创建(此前是固定 /tmp 路径 + 此处 rmtree 清场,
    # 并发跑两份会互删缓存)。此处不再需要全局清场。
    with tempfile.TemporaryDirectory() as td:
        test_template_no_duplicate_defs()
        test_widen_cached()
        test_threads_floor()
        test_stale_running(Path(td))
        test_champion_check_position()
        test_digest_steering()
        test_template_slim()

    print(f"\n{'='*60}\nv2.11 fixes: {PASS} passed, {FAIL} failed")
    if FAILED:
        print("FAILED: " + "; ".join(FAILED))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
