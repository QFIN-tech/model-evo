#!/usr/bin/env python
"""benchmark 对照基线:AIDE(AI-Driven Exploration,WeCoAI/aideml)。

    python scripts/benchmark/baseline_aide.py --runs-root <runs> \
        --out-dir <dir> --aide-root /tmp/aide_src/aideml \
        --python <aide venv python> --model deepseek-flash \
        [--only a,b] [--tier full] [--steps N]

**测什么**:把"LLM 写代码 + 执行反馈"换成 AIDE 的树搜索(多个 draft + 调试/改进
分支),但不给 ModelEvo 的裁决门、代际编排、血缘与完整性校验。与
`baseline_raw_agent.py`(单链)一起构成"结构化搜索"与"无纪律单链"两端。

口径(与 ModelEvo 侧对齐):
- AIDE 只用 train 建模(内部 k-fold 选模型),validation 作为它自己的反馈面
  (AIDE 用 test.csv 打分,**这里把 validation 喂成 test.csv**),shadow/final
  对 AIDE 不可见;
- 跑完后用 AIDE 产出的最终方案对 shadow_test 重新拟合 + 打分(与 raw agent
  同口径,保证"最终模型"不依赖当轮是否保存了预测);
- 分数按 `__mvo_sample_key` 落盘到 `<out>/<dataset>/shadow_score.parquet`。

模型:与其它对照同一 LLM(`--model`),三个 stage(code / feedback / report)
都用它;provider 固定 anthropic(经 ANTHROPIC_BASE_URL 指到同一端点)。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SKILL = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bench_env as BENV  # noqa: E402

#: 方案代码的线程上限(本机多 run 并发,全核并行反而更慢)
AIDE_THREADS = BENV.threads("aide")
SAMPLE_KEY = "__mvo_sample_key"
#: 任务协议版本(写进 run.json;报告据 trained_on 决定是否进 train-only 名次)
TASK_PROTOCOL = "kaggle_trainval_v2"
#: journal 无新节点超过该秒数即判卡死(单节点执行上限 3600 s + LLM 余量)
STALL_SEC = int(os.environ.get("AIDE_STALL_SEC", "5400"))
#: AIDE 的 test.csv 带 validation 标签:这些写法可能把它用于训练
LEAK_PATTERNS = (r"concat\(\s*\[[^\]]*\btest",
                 r"\btest\w*\s*\[\s*(target\w*|[\"']target[\"'])\s*\]",
                 r"\b(fit|train)\w*\(.*\btest\w*\b")
BUDGET = {
    "fast": {"steps": 6, "wall_hours": 1.0, "k_fold": 5},
    "full": {"steps": 16, "wall_hours": 4.0, "k_fold": 5},
    "large": {"steps": 22, "wall_hours": 10.0, "k_fold": 5},
}


def _read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _build_task_dir(run_dir: Path, work: Path, spec: dict,
                    gpu: int | None = None,
                    time_column: str | None = None,
                    single_model: bool = False) -> str:
    """AIDE 的任务目录(Kaggle 式,AIDE 原生用法)。

    train.csv = train + validation(带 `target`),test.csv = validation 特征
    (**不带标签**,只用于产出 submission)。AIDE 用自己的 k 折 CV 选模型。

    旧协议把带标签的 validation 放进 test.csv,方案会用 test 标签选超参
    (实测 credit_g),shadow 打分时 test.csv 换成无标签帧就走了另一条分支,
    复现不了 AIDE 当时选出的方案。标签不进 test.csv,换帧打分才是机械的。
    代价:最终模型用 train+validation 训练,报告中与同口径的 ModelEvo 交付
    模型对照,不进 train-only 名次。

    ``time_column``(套件 ``derive_time.column``)给定时:train.csv 按时间排序,
    说明里写明"按时间切分、测试期在训练期之后,用时间顺序留出集评估"。
    只补任务信息,不替 AIDE 设计方案,也不加特征。动机:bike_sharing 强漂移,
    同一 LightGBM 随机 5 折 RMSE 33.5、时间末 20% 65.7,随机 CV 会系统性选错;
    ModelEvo / raw-agent 的 validation 天然在训练期之后,AIDE 缺这条信息。
    """
    splits = run_dir / "sample-profile" / "splits"
    labels = splits / "labels"
    target = spec["target"]["name"]
    work.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(splits / "train.parquet")
    tr_y = pd.read_parquet(labels / "train.parquet")[target]
    va = pd.read_parquet(splits / "validation.parquet")
    va_y = pd.read_parquet(labels / "validation.parquet")[target]
    tr = tr.copy()
    tr["target"] = tr_y.to_numpy()
    va_l = va.copy()
    va_l["target"] = va_y.to_numpy()
    time_txt = ""
    if time_column:
        # 时间列只用于排序与说明,不进 csv(与其它参与者特征面一致)
        tr_t = pd.read_parquet(labels / "train.parquet")[time_column].astype(str)
        va_t = pd.read_parquet(labels / "validation.parquet")[time_column].astype(str)
        tr = tr.iloc[np.argsort(tr_t.to_numpy(), kind="stable")]
        va_l = va_l.iloc[np.argsort(va_t.to_numpy(), kind="stable")]
        start = min(va_t)
        time_txt = (f"\n        - 时间:数据按时间切分,测试期在训练数据之后(自 {start} 起);"
                    "`train.csv` 已按时间顺序排列。请用时间顺序的留出集评估方案"
                    "(例如最后 2 个月),不要用随机打乱的交叉验证。")
    pd.concat([tr, va_l], ignore_index=True).to_csv(work / "train.csv",
                                                    index=False)
    va.to_csv(work / "test.csv", index=False)
    metric = str(spec.get("primary_metric") or "auc")
    maximize = metric.lower() in ("auc", "roc_auc", "pr_auc", "gini", "ks")
    env_txt = BENV.gpu_hint(gpu, AIDE_THREADS)
    # 单模型对照:约束写进任务说明。评估方式那一条在单模型模式下换掉,
    # 否则与"不要做交叉验证"自相矛盾(run 前面已有一次实测踩到)。
    eval_txt = ("用 `train.csv` 上的交叉验证评估方案" if not single_model
                else "用 `train.csv` 上的**单次留出**(如最后 20%)评估方案,不要做交叉验证")
    single_txt = ("" if not single_model else textwrap.dedent("""\
        - **本次是单模型对照**:最终方案只能训练**一个**估计器实例(一种算法、
          一套超参),预测时直接用它输出。禁止多模型加权/平均/投票/堆叠,禁止
          同族多 seed、多配置、多折平均,也禁止"训练多个候选再挑一个"。
          早停允许(训练集内部留尾部验证集),但交付的仍是一个模型。
        """))
    desc = textwrap.dedent(f"""\
        # 任务

        {spec.get('task_name') or run_dir.name}:预测 `target` 列。

        - 数据:`train.csv`(含 `target` 列)。`test.csv` 只有特征、没有标签;
          {eval_txt}。提交 `submission.csv`,列 `target`(预测值,行序与
          `test.csv` 一致)。
        - 指标:`{metric}`({'越大越好' if maximize else '越小越好'})。
        - 约束:只能用本工作目录里的数据,不得访问网络;最终方案要写进
          `runfile.py`,并在 `./working/submission.csv` 输出 `target` 列。
{single_txt}
        - 执行环境:{env_txt}{time_txt}
        """).replace("{single_txt}", single_txt).replace("{eval_txt}", eval_txt)
    (work / "description.md").write_text(desc, encoding="utf-8")
    return str(work / "description.md")


def _run_aide(aide_root: Path, py: str, task_dir: Path, *, model: str,
              steps: int, k_fold: int, wall_hours: float,
              exp_name: str, log_dir: Path,
              max_tokens: int = 32000, gpu: int | None = None,
              no_cv: bool = False) -> dict:
    cfg = {
        "data_dir": str(task_dir),
        "desc_file": str(Path(task_dir) / "description.md"),
        "log_dir": str(log_dir),
        "workspace_dir": str(log_dir / "workspaces"),
        "exp_name": exp_name,
        "copy_data": True,
        "preprocess_data": False,
        "generate_report": False,
        "agent": {
            "steps": int(steps),
            # --no-cv:k_fold_validation=1 关掉 AIDE 通用指引里的"用 k 折评估"
            "k_fold_validation": 1 if no_cv else int(k_fold),
            "expose_prediction": True,
            "data_preview": True,
            "metric_maximize": None,
            # AIDE 缺省 max_tokens=4096;驱动模型会先输出思考块,4096 会被截断
            # (实测 stop_reason=max_tokens),统一给足。
            "code": {"model": model, "provider": "anthropic",
                     "max_tokens": int(max_tokens)},
            "feedback": {"model": model, "provider": "anthropic",
                         "max_tokens": int(max_tokens)},
            "search": {"max_debug_depth": 3, "debug_prob": 0.5,
                       "num_drafts": 3},
        },
        "report": {"model": model, "provider": "anthropic",
                   "max_tokens": int(max_tokens)},
    }
    cfg_p = Path(task_dir) / "aide-config.json"
    cfg_p.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    # 执行层线程上限:必须在 AIDE **父进程**启动前设好。AIDE 在进程内直接
    # exec 生成的代码(interpreter 用 multiprocessing fork),OpenMP / joblib
    # 的线程池在首次 import 时就按环境变量初始化;子进程再设已经晚了。
    # LOKY_MAX_CPU_COUNT 管 sklearn/joblib 的 n_jobs=-1(它不受 OMP 影响)。
    env = BENV.device_env(gpu, AIDE_THREADS)
    # AIDE 的 anthropic 后端读 SDK 默认凭据
    env.setdefault("ANTHROPIC_API_KEY", env.get("ANTHROPIC_AUTH_TOKEN", ""))
    # AIDE 用 OmegaConf.from_cli() 读**命令行 key=value** 覆盖默认 config.yaml,
    # 不认 --config 文件。逐项展开(嵌套用点号)。
    def _flat(d, prefix=""):
        out = []
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.extend(_flat(v, key + "."))
            elif isinstance(v, bool):
                out.append(f"{key}={'true' if v else 'false'}")
            elif v is None:
                out.append(f"{key}=null")
            else:
                out.append(f"{key}={v}")
        return out

    t0 = time.time()
    # 看门狗:journal 超过 STALL_SEC 没有新节点就判卡死并杀掉(实测 bike_sharing
    # 的 aide.run 在 waitpid 里挂了 3 h,0% CPU)。超墙钟同样处理。两种情况都
    # 不算失败:已落盘的 best_solution 仍是预算内的结果。
    deadline = t0 + int(wall_hours * 3600) + 1800
    logf = Path(log_dir) / f"aide-{int(t0)}.out"
    stalled = None
    with open(logf, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen([py, "-m", "aide.run", *_flat(cfg)],
                                cwd=str(aide_root), stdout=fh,
                                stderr=subprocess.STDOUT, text=True, env=env,
                                start_new_session=True)
        while proc.poll() is None:
            time.sleep(30)
            js = list(Path(log_dir).glob("*/journal.json"))
            last = max([t0] + [j.stat().st_mtime for j in js])
            now = time.time()
            if now > deadline:
                stalled = "wall_budget"
            elif now - last > STALL_SEC:
                stalled = f"no_progress_{int(now - last)}s"
            if stalled:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                break
    rc = None if stalled else proc.returncode
    out = logf.read_text(encoding="utf-8", errors="replace")
    err = out
    return {"rc": rc, "exit": BENV.classify_exit(rc), "stalled": stalled,
            "seconds": round(time.time() - t0, 1),
            "stdout_tail": (out or "")[-3000:],
            "stderr_tail": (err or "")[-3000:]}


#: 违反"单模型(无融合)"约束的写法(只用于标注,不据此改分数)
_ENSEMBLE_HINTS = (
    r"Voting(Classifier|Regressor)", r"Stacking(Classifier|Regressor)",
    r"\b(StackingCV|Bagging)(Classifier|Regressor)", r"CalibratedClassifierCV",
    r"KFold\(|StratifiedKFold\(|TimeSeriesSplit\(",
    r"\bestimators\s*=", r"(LGBM|XGB|CatBoost|RandomForest|ExtraTrees)\w*\(.*\n.*\1",
    r"for\s+\w+\s+in\s+SEEDS", r"blend|weights\s*=|np\.mean\(\w*preds",
)


def _audit_strategy(code: str) -> dict:
    """标注最终方案是否真的只用了一个模型(静态扫描,人工复核用)。"""
    import re as _re
    hits = []
    for pat in _ENSEMBLE_HINTS:
        m = _re.search(pat, code)
        if m:
            hits.append(f"{pat} @L{code[:m.start()].count(chr(10)) + 1}")
    return {"status": "review" if hits else "single", "hits": hits[:10]}


def _best_aide_solution(log_dir: Path) -> Path | None:
    """AIDE 的最终方案:save_run 写出 <exp>/best_solution.py(取 metric 最好的节点)。"""
    cands = sorted(log_dir.glob("*/best_solution.py"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if (log_dir / "best_solution.py").exists():
        cands.insert(0, log_dir / "best_solution.py")
    return cands[0] if cands else None


def _shadow_from_code(work: Path, run_dir: Path, runfile: Path,
                      target: str, gpu: int | None = None) -> dict:
    """用 AIDE 的最终方案(runfile.py)对 shadow_test 打分。

    AIDE 的 runfile 是**自包含脚本**:`main()` 读 ./input/{train,test}.csv、
    建最终 pipeline、写 ./working/submission.csv。这里不重写它的建模逻辑,
    只在同一份代码上把"最终 pipeline 对 test.csv 打分"换成"对 shadow 打分":
    在 shadow 帧前面临时替换 test.csv 的副本会让代码改口径,所以改为
    **导入 runfile 后调用它自己的构建路径**——把 shadow 帧写成它的
    input/test.csv(去掉 target 列),跑 main(),再从 working/submission.csv
    取预测。这与 AIDE 自己评估 test 的方式完全一致,只是换了输入帧。
    """
    splits = run_dir / "sample-profile" / "splits"
    run_dir_work = work / "aide-run"
    run_dir_work.mkdir(parents=True, exist_ok=True)
    inp = run_dir_work / "input"
    (run_dir_work / "working").mkdir(exist_ok=True)
    if inp.is_symlink() or inp.exists():
        shutil.rmtree(inp, ignore_errors=True)
    inp.mkdir(exist_ok=True)
    for f in ("train.csv", "test.csv"):
        src = work / "task" / f
        if src.exists():
            shutil.copy2(src, inp / f)
    # shadow 帧作为 test.csv(不含 target)
    sh_frame = pd.read_parquet(splits / "shadow_test.parquet")
    sh_frame.to_csv(inp / "test.csv", index=False)
    # 线程限制:方案代码普遍写 n_jobs=-1 / 全核并行。本机同时跑着多个
    # benchmark run,全局限流下 -1 会让单折迭代慢到分钟级(实测第 1 折后
    # 卡住)。这里把 n_jobs 机械夹到 AIDE_THREADS,只是执行资源口径,
    # 不改方案本身的结构与超参。
    src = Path(runfile).read_text(encoding="utf-8")
    for old_tok, new_tok in (("n_jobs=-1", f"n_jobs={AIDE_THREADS}"),
                             ("n_jobs = -1", f"n_jobs = {AIDE_THREADS}"),
                             ("nthread=-1", f"nthread={AIDE_THREADS}"),
                             ("n_jobs=-1,", f"n_jobs={AIDE_THREADS},")):
        src = src.replace(old_tok, new_tok)
    (run_dir_work / "runfile.py").write_text(src, encoding="utf-8")
    t0 = time.time()
    env = BENV.device_env(gpu, AIDE_THREADS)
    sub = run_dir_work / "working" / "submission.csv"
    attempts = []
    for _ in range(BENV.KILLED_RETRIES + 1):
        sub.unlink(missing_ok=True)     # 不能让上一次的残留冒充本次产出
        gate = BENV.wait_for_memory()
        try:
            proc = subprocess.run([sys.executable, "runfile.py"],
                                  cwd=str(run_dir_work), capture_output=True,
                                  text=True, timeout=7200, env=env)
            rc, stderr = proc.returncode, proc.stderr
        except subprocess.TimeoutExpired:
            rc, stderr = None, ""
        attempts.append({"rc": rc, "exit": BENV.classify_exit(rc),
                         "free_gb": gate.get("free_gb")})
        if attempts[-1]["exit"] != "killed":
            break
    if rc != 0 or not sub.exists():
        return {"ok": False, "attempts": attempts,
                "error": ("runfile 未产出有效 submission.csv: "
                          + BENV.error_tail(stderr))}
    df = pd.read_csv(sub)
    col = "target" if "target" in df.columns else df.columns[-1]
    pred = df[col].to_numpy()
    keys = pd.read_parquet(splits / "labels" / "shadow_test.parquet",
                           columns=[SAMPLE_KEY])[SAMPLE_KEY].astype(str)
    if len(pred) != len(keys):
        return {"ok": False, "error": f"行数不符 {len(pred)} vs {len(keys)}"}
    out = work / "shadow_score.parquet"
    pd.DataFrame({"score": pred, SAMPLE_KEY: keys.to_numpy()}).to_parquet(
        out, index=False)
    return {"ok": True, "n": int(len(pred)), "attempts": attempts,
            "seconds": round(time.time() - t0, 1)}


def run_one(ds: dict, runs_root: Path, out_dir: Path, *, aide_root: Path,
            py: str, model: str, tier: str, steps: int | None,
            wall_hours: float | None, gpu: int | None = None,
            time_hint: bool = False, single_model: bool = False,
            no_cv: bool = False) -> dict:
    name = ds["name"]
    run_dir = (runs_root / name).resolve()
    if not (run_dir / "sample-profile" / "splits" / "train.parquet").exists():
        return {"dataset": name, "status": "no_profile"}
    budget = BUDGET.get(tier) or BUDGET["full"]
    out_dir = Path(out_dir).resolve()
    work = out_dir / name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    task_dir = work / "task"
    spec = _read_yaml(run_dir / "task-spec.yaml")
    tcol = ((ds.get("derive_time") or {}).get("column")
            if time_hint and ds.get("split") == "time" else None)
    _build_task_dir(run_dir, task_dir, spec, gpu=gpu, time_column=tcol,
                    single_model=single_model)
    log_dir = work / "aide-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    for _ in range(BENV.KILLED_RETRIES + 1):
        gate = BENV.wait_for_memory()
        res = _run_aide(aide_root, py, task_dir, model=model,
                        steps=int(steps or budget["steps"]),
                        k_fold=int(budget["k_fold"]),
                        wall_hours=float(wall_hours or budget["wall_hours"]),
                        exp_name=f"bench-{name}", log_dir=log_dir, gpu=gpu,
                    no_cv=no_cv)
        attempts.append({"rc": res["rc"], "exit": res["exit"],
                         "seconds": res["seconds"],
                         "free_gb": gate.get("free_gb")})
        # 被杀但已有最优方案:用这份部分结果,不重花 LLM 预算
        if res["exit"] != "killed" or _best_aide_solution(log_dir):
            break
    rec = {"dataset": name, "model": model, "tier": tier,
           "protocol": TASK_PROTOCOL, "trained_on": "train+validation",
           "mode": "single_model" if single_model else "free",
           "time_hint": tcol,
           "gpu": gpu, "threads": int(AIDE_THREADS), "aide": res,
           "aide_attempts": attempts, "partial": res["exit"] != "ok"}
    # AIDE 的最终模型:save_run 的 best_solution.py
    sol = _best_aide_solution(log_dir)
    if sol:
        shutil.copy2(sol, work / "aide_solution.py")
        _sol_src = sol.read_text(encoding="utf-8")
        rec["training_source_audit"] = BENV.audit_training_source(
            _sol_src, LEAK_PATTERNS)
        rec["strategy_audit"] = _audit_strategy(_sol_src)
        rec["shadow"] = _shadow_from_code(work, run_dir,
                                          work / "aide_solution.py",
                                          spec["target"]["name"], gpu=gpu)
    else:
        rec["shadow"] = {"ok": False, "error": "AIDE 未产出 best_solution.py"}
    rec["status"] = ("ok" if rec["shadow"].get("ok")
                     else ("aide_failed" if res["exit"] != "ok" else "no_shadow"))
    (work / "run.json").write_text(json.dumps(rec, ensure_ascii=False,
                                              indent=1, default=str),
                                   encoding="utf-8")
    return rec


def main(argv=None) -> int:
    global AIDE_THREADS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--aide-root", default="/data/oceanus_share/j-wangyaoxuan-jk/modelevo-202611/contrast-src/aideml")
    ap.add_argument("--python", default=str(
        Path("/data/oceanus_share/j-wangyaoxuan-jk/modelevo-202611/venvs/aide"
             "/bin/python")))
    ap.add_argument("--suite", default=str(SKILL / "assets"
                                           / "benchmark-suite.yaml"))
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--tier", default="full",
                    choices=("fast", "full", "large"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--wall-hours", type=float, default=None)
    ap.add_argument("--threads", type=int, default=AIDE_THREADS,
                    help="执行层线程上限(缺省 24,也可用 AIDE_THREADS)")
    ap.add_argument("--gpu", default=os.environ.get("AIDE_GPU"),
                    help="手工指定物理卡号;缺省按实际占用自动挑")
    ap.add_argument("--no-gpu", action="store_true", help="强制 CPU")
    ap.add_argument("--single-model", action="store_true",
                    help="单模型公平对照:说明里禁止一切融合/多模型/多折平均")
    ap.add_argument("--no-cv", action="store_true",
                    help="把 AIDE 的 k_fold_validation 置 1(不再在通用指引里要求 k 折)")
    ap.add_argument("--time-hint", action="store_true",
                    help="split=time 的数据集:train.csv 按时间排序并在说明里写明时间切分")
    a = ap.parse_args(argv)
    # --threads 覆盖模块级口径(_run_aide / _shadow_from_code 都读它)
    AIDE_THREADS = max(1, int(a.threads))
    suite = _read_yaml(Path(a.suite))
    only = {x for x in (a.only or "").split(",") if x}
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in suite.get("datasets") or []:
        name = ds.get("name")
        if only and name not in only:
            continue
        if not only and a.tier not in (ds.get("tiers") or []):
            continue
        if a.no_gpu:
            gpu = None
        elif a.gpu is not None:
            gpu = int(str(a.gpu).split(",")[0])
        else:
            gpu = BENV.pick_gpu()
        print(f"[aide] {name} ... gpu={gpu} threads={AIDE_THREADS}", flush=True)
        rec = run_one(ds, Path(a.runs_root), out_dir, aide_root=Path(a.aide_root),
                      py=a.python, model=a.model, tier=a.tier, steps=a.steps,
                      wall_hours=a.wall_hours, gpu=gpu, time_hint=a.time_hint,
                      single_model=a.single_model, no_cv=a.no_cv)
        print(f"[aide] {name}: {rec['status']} rc="
              f"{rec.get('aide', {}).get('rc')}", flush=True)
        with (out_dir / "aide.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
