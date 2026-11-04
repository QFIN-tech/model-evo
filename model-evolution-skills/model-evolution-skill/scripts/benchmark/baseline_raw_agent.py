#!/usr/bin/env python
"""benchmark 对照基线:「无 ModelEvo 的同基础 Agent」(v2.19.1)。

    python scripts/benchmark/baseline_raw_agent.py --runs-root <runs> \
        --out-dir <dir> --model deepseek-flash [--only a,b] [--tier full]

**测什么**:同一 LLM、同一份数据、同一个测试面,只有"让 LLM 直接写代码做建模"
这一个能力,没有 ModelEvo 的 profile / 裁决门 / 代际编排 / 血缘 / 完整性校验。
它回答的是"框架比一个能干但没有纪律的 Agent 强多少",与 AIDE(树搜索 + 执行反馈)
一起构成两组对照:前者只给 LLM 与执行器,后者给结构化搜索。

协议(与 ModelEvo 侧对齐才能使分数可比):

1. 只用 `sample-profile/splits/train.parquet` + `labels/train.parquet` 训练;
   validation 只用于 Agent 自选模型(拿到指标反馈),不用于最终打分;
   **shadow_test 与 final_test 对 Agent 完全不可见**——工作目录里只放 train
   与 validation,标签列统一命名为 `target`。
2. 产出 `predict.py`(含 `predict(df)->np.ndarray`),由本脚本在**干净的子进程**
   里 import 后对 shadow_test 打分,分数带 `__mvo_sample_key` 落盘。
3. 评分只在本脚本内做一次(shadow_test 标签只在这里读),输出与
   `benchmark aligned` 的参与者格式一致:
   `<out>/<dataset>/shadow_score.parquet`(列:score, __mvo_sample_key)。
4. 预算与 ModelEvo 侧同一档位:`--tier full` → 6 代 / 24 trial / 4h 的等价
   口径换算成"最多 N 轮迭代 + 墙钟上限",见 `BUDGET`。

Agent 侧接口:每轮把"任务 + 上一轮代码 + 上一轮 validation 结果/报错"发给
LLM,要求返回完整 `predict.py`。这与 AIDE 的精简模式同构(单链、执行反馈),
差别在于这里刻意不给它搜索树、不给它自己的评估纪律。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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

SAMPLE_KEY = "__mvo_sample_key"
#: 执行层线程上限(raw-agent 侧;与 AIDE 同一口径,缺省 24)
RAW_AGENT_THREADS = BENV.threads("raw_agent")
#: 与 benchmark-suite.yaml 的 budget 同档:full = 6 代 / 24 trial / 4 h,
#: large = 8 代 / 30 trial / 10 h。折算成"LLM 迭代轮数 + 墙钟"。
BUDGET = {
    "fast": {"rounds": 6, "wall_hours": 1.0},
    "full": {"rounds": 12, "wall_hours": 4.0},
    "large": {"rounds": 16, "wall_hours": 10.0},
}

#: 单模型公平对照:强制单一估计器、单次拟合,不做任何融合
SINGLE_MODEL_RULES = """- **本次是单模型对照**:predict.py 的 fit() 里只能训练**一个**估计器实例
  (一种算法、一套超参),predict() 直接用它输出预测。禁止:
    * 多个模型的加权/平均/投票/堆叠(包括同族多 seed、多配置、多折平均);
    * Cross-Validation 平均、OOF 融合、bagging 包装器;
    * 先训练多个候选再挑一个(挑选本身也是多次拟合)。
  早停是允许的:可以在 train 内部留一个尾部验证集做早停,但最终交付仍是一个模型。
  评估你的方案时,用**单次留出**(例如从 train 里留最后 20% 或随机 20%)算一次分数,
  不要做 k 折。"""

SYSTEM = """你是一位资深机器学习工程师。你要为一个表格任务写出**一个可执行的
predict.py**。

约束:
- 只能读工作目录里的 train.parquet / train_labels.parquet / validation.parquet /
  validation_labels.parquet,以及你自己生成的文件。数据路径在 data_dir.txt 里。
- 目标列名固定为 `target`;行序必须与输入 parquet 一致。
- predict.py 必须定义:
    def fit(train: pd.DataFrame, y: np.ndarray) -> object   # 返回模型对象
    def predict(model, frame: pd.DataFrame) -> np.ndarray   # 返回与 frame 行数一致的预测
- 允许 import 本环境已装的库(numpy/pandas/scikit-learn/xgboost/lightgbm/catboost/torch)。
- 不要访问网络,不要尝试读取除上述之外的数据文件。
- {env}
{rules}
- 每次回答输出**完整的 predict.py 全文**,放在一个 ```python 代码块里,不要输出别的内容。"""

USER_TMPL = """任务:{desc}

数据:{shape}
列:{cols}

已发现的数据字典(若有):
{dict}

本轮是第 {rnd}/{total} 轮。{prev}
请输出完整的 predict.py。"""


#: 违反"单模型(无融合)"约束的写法(只用于标注,不据此改分数)
_ENSEMBLE_HINTS = (
    r"Voting(Classifier|Regressor)", r"Stacking(Classifier|Regressor)",
    r"\b(StackingCV|Bagging)(Classifier|Regressor)", r"CalibratedClassifierCV",
    r"KFold\(|StratifiedKFold\(|TimeSeriesSplit\(",
    r"\bestimators\s*=", r"(LGBM|XGB|CatBoost|RandomForest|ExtraTrees)\w*\(.*\n.*\1",
    r"for\s+\w+\s+in\s+SEEDS", r"blend|weights\s*=|np\.mean\(\w*preds",
)


def _audit_strategy(work: Path) -> dict:
    """标注最终方案是否真的只用了一个模型(静态扫描,人工复核用)。"""
    f = work / "predict.py"
    if not f.exists():
        return {"status": "missing"}
    code = f.read_text(encoding="utf-8", errors="replace")
    hits = []
    import re as _re
    for pat in _ENSEMBLE_HINTS:
        m = _re.search(pat, code)
        if m:
            line = code[:m.start()].count("\n") + 1
            hits.append(f"{pat} @L{line}")
    return {"status": "review" if hits else "single", "hits": hits[:10]}


def _read_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _llm_client():
    import anthropic

    base = os.environ.get("ANTHROPIC_BASE_URL")
    tok = os.environ.get("ANTHROPIC_AUTH_TOKEN") or \
        os.environ.get("ANTHROPIC_API_KEY")
    if not base or not tok:
        raise SystemExit("需要 ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN")
    return anthropic.Anthropic(base_url=base, auth_token=tok)


def _ask(client, model: str, system: str, user: str,
         max_tokens: int = 16000) -> str:
    r = client.messages.create(model=model, max_tokens=max_tokens,
                               system=system,
                               messages=[{"role": "user", "content": user}])
    return "".join(getattr(b, "text", "") for b in r.content)


def _extract_code(text: str) -> str:
    """取最后一个 ```python 代码块;没有代码块时返回原文。"""
    blocks = []
    cur = None
    for line in text.splitlines():
        if line.strip().startswith("```"):
            if cur is None:
                cur = []
            else:
                blocks.append("\n".join(cur))
                cur = None
            continue
        if cur is not None:
            cur.append(line)
    if cur is not None:
        blocks.append("\n".join(cur))
    for b in reversed(blocks):
        if "def predict" in b:
            return b
    return text if "def predict" in text else ""


def _prepare_workdir(run_dir: Path, work: Path) -> dict:
    """把 train / validation 复制成 Agent 可见的两个 parquet(不含任何 holdout)。"""
    splits = run_dir / "sample-profile" / "splits"
    labels = splits / "labels"
    work.mkdir(parents=True, exist_ok=True)
    info = {}
    for split, alias in (("train", "train"), ("validation", "validation")):
        X = pd.read_parquet(splits / f"{split}.parquet")
        y = pd.read_parquet(labels / f"{split}.parquet")
        target = [c for c in y.columns if c not in
                  ("bench_split_date", "bench_row_id", SAMPLE_KEY)][0]
        info[alias] = {"target": target, "n": len(X)}
        X.to_parquet(work / f"{alias}.parquet", index=False)
        pd.DataFrame({"target": y[target].to_numpy()}).to_parquet(
            work / f"{alias}_labels.parquet", index=False)
    (work / "data_dir.txt").write_text(str(work.resolve()) + "\n",
                                       encoding="utf-8")
    return info


def _run_candidate(work: Path, model_type: str, timeout: float = 5400,
                   env: dict | None = None) -> dict:
    """在干净子进程里跑 predict.py:fit → validation 打分 + shadow 打分。"""
    code = work / "predict.py"
    if not code.exists():
        return {"ok": False, "error": "predict.py 不存在"}
    runner = work / "_runner.py"
    runner.write_text(textwrap.dedent(f"""
        import json, sys, traceback
        import numpy as np, pandas as pd
        sys.path.insert(0, {str(work)!r})
        w = {str(work)!r}
        out = {{}}
        try:
            import predict as P
            tr = pd.read_parquet(w + "/train.parquet")
            ty = pd.read_parquet(w + "/train_labels.parquet")["target"].to_numpy()
            va = pd.read_parquet(w + "/validation.parquet")
            m = P.fit(tr, ty)
            p = np.asarray(P.predict(m, va)).ravel()
            if len(p) != len(va):
                raise ValueError(f"predict 返回 {{len(p)}} 行 != {{len(va)}} 行")
            pd.DataFrame({{"pred": p}}).to_parquet(w + "/validation_pred.parquet",
                                                  index=False)
            out = {{"ok": True, "n": int(len(p))}}
        except Exception:
            out = {{"ok": False, "error": traceback.format_exc()[-3000:]}}
        json.dump(out, open(w + "/_result.json", "w"))
    """), encoding="utf-8")
    BENV.wait_for_memory()
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, str(runner)], cwd=str(work),
                              capture_output=True, text=True, timeout=timeout,
                              env=env)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "执行超时"}
    res = {"ok": False, "error": f"rc={rc}"}
    if (work / "_result.json").exists():
        try:
            res = json.loads((work / "_result.json").read_text())
        except ValueError:
            pass
    if not res.get("ok"):
        res["error"] = (res.get("error") or "") + "\n" + \
            (proc.stderr or "")[-1500:]
    res["seconds"] = round(time.time() - t0, 1)
    return res


def _score_validation(work: Path, run_dir: Path) -> float | None:
    """validation 指标(给 Agent 反馈用;标签来自 run 的 labels 目录)。"""
    run_dir = Path(run_dir)
    spec = _read_yaml(run_dir / "task-spec.yaml")
    target = spec["target"]["name"]
    metric = str(spec.get("primary_metric") or "auc")
    y = pd.read_parquet(run_dir / "sample-profile" / "splits" / "labels"
                        / "validation.parquet")[target].to_numpy()
    pred = pd.read_parquet(work / "validation_pred.parquet")["pred"].to_numpy()
    from sklearn.metrics import roc_auc_score
    try:
        if metric in ("auc", "roc_auc"):
            return float(roc_auc_score(y, pred))
        if metric in ("rmse",):
            return float(np.sqrt(np.mean((y - pred) ** 2)))
    except Exception:
        return None
    return None


def _shadow_predict(work: Path, run_dir: Path, out: Path,
                    env: dict | None = None) -> dict:
    """用最终 predict.py 对 shadow_test 打分(只在这里读 shadow 特征,不读标签)。"""
    splits = Path(run_dir) / "sample-profile" / "splits"
    X = pd.read_parquet(splits / "shadow_test.parquet")
    script = work / "_shadow.py"
    script.write_text(textwrap.dedent(f"""
        import json, sys, traceback
        import numpy as np, pandas as pd
        sys.path.insert(0, {str(work)!r})
        w = {str(work)!r}
        try:
            import predict as P
            tr = pd.read_parquet(w + "/train.parquet")
            ty = pd.read_parquet(w + "/train_labels.parquet")["target"].to_numpy()
            sh = pd.read_parquet({str(splits / 'shadow_test.parquet')!r})
            m = P.fit(tr, ty)
            p = np.asarray(P.predict(m, sh)).ravel()
            pd.DataFrame({{"pred": p}}).to_parquet(w + "/shadow_pred.parquet",
                                                  index=False)
            json.dump({{"ok": True}}, open(w + "/_sh.json", "w"))
        except Exception:
            json.dump({{"ok": False, "error": traceback.format_exc()[-3000:]}},
                      open(w + "/_sh.json", "w"))
    """), encoding="utf-8")
    BENV.wait_for_memory()
    proc = subprocess.run([sys.executable, str(script)], cwd=str(work),
                          capture_output=True, text=True, timeout=7200,
                          env=env)
    st = {"ok": False}
    if (work / "_sh.json").exists():
        st = json.loads((work / "_sh.json").read_text())
    if not st.get("ok"):
        return {"ok": False, "error": (st.get("error") or "")
                + (proc.stderr or "")[-1200:]}
    pred = pd.read_parquet(work / "shadow_pred.parquet")["pred"].to_numpy()
    keys = pd.read_parquet(splits / "labels" / "shadow_test.parquet",
                           columns=[SAMPLE_KEY])[SAMPLE_KEY].astype(str)
    if len(pred) != len(keys):
        return {"ok": False, "error": f"shadow 预测 {len(pred)} 行 != {len(keys)}"}
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"score": pred, SAMPLE_KEY: keys.to_numpy()}).to_parquet(
        out, index=False)
    return {"ok": True, "n": int(len(pred))}


def run_one(ds: dict, runs_root: Path, out_dir: Path, *, model: str,
            tier: str, max_rounds: int | None, wall_hours: float | None,
            keep: bool = False, gpu: int | None = None,
            n_threads: int = RAW_AGENT_THREADS,
            single_model: bool = False) -> dict:
    name = ds["name"]
    run_dir = (runs_root / name).resolve()
    if not (run_dir / "sample-profile" / "splits" / "train.parquet").exists():
        return {"dataset": name, "status": "no_profile"}
    budget = BUDGET.get(tier) or BUDGET["full"]
    rounds = int(max_rounds or budget["rounds"])
    wall = float(wall_hours or budget["wall_hours"])
    out_dir = Path(out_dir).resolve()   # subprocess 会切 cwd,路径必须绝对
    work = out_dir / name / "work"
    if work.exists():
        shutil.rmtree(work)
    # 执行层:线程夹到 n_threads,GPU 收敛成单卡;device 意图同时也是
    # MODELEVO_GPU 的口径(GBM 模板靠它决定 device=cuda)。
    env = BENV.device_env(gpu, n_threads)
    env_hint = BENV.gpu_hint(gpu, n_threads)
    info = _prepare_workdir(run_dir, work)
    spec = _read_yaml(run_dir / "task-spec.yaml")
    desc = f"{spec.get('task_name') or name};指标 {spec.get('primary_metric')};" \
           f"类型 {spec.get('model_type')}"
    X = pd.read_parquet(work / "train.parquet")
    cols = ", ".join(map(str, list(X.columns)[:60]))
    dd = run_dir / "DATA_DICTIONARY.md"
    dd_txt = dd.read_text(encoding="utf-8")[:6000] if dd.exists() else "(无)"
    client = _llm_client()
    system = SYSTEM.format(env=env_hint, rules=SINGLE_MODEL_RULES if single_model else "")
    t0 = time.time()
    best = None
    log = []
    code = ""
    prev = "这是第一轮:从零开始设计。"
    for rnd in range(1, rounds + 1):
        if time.time() - t0 > wall * 3600:
            log.append({"round": rnd, "skipped": "wall_budget"})
            break
        user = USER_TMPL.format(desc=desc, shape=str(X.shape), cols=cols,
                                dict=dd_txt, rnd=rnd, total=rounds, prev=prev)
        try:
            reply = _ask(client, model, system, user)
        except Exception as exc:  # noqa: BLE001
            log.append({"round": rnd, "error": f"llm: {exc}"[:400]})
            continue
        code = _extract_code(reply) or code
        if not code:
            prev = "上一轮没有给出可用的 predict.py。请严格按格式输出完整代码块。"
            log.append({"round": rnd, "error": "no_code"})
            continue
        (work / "predict.py").write_text(code, encoding="utf-8")
        res = _run_candidate(work, spec.get("model_type") or "", env=env)
        score = _score_validation(work, run_dir) if res.get("ok") else None
        log.append({"round": rnd, "ok": res.get("ok"),
                    "seconds": res.get("seconds"), "validation": score,
                    "error": (res.get("error") or "")[-600:]})
        if score is not None and (best is None or score > best):
            best = score
            (work / "predict_best.py").write_text(code, encoding="utf-8")
            shutil.copy2(work / "predict.py", work / "predict_best.py")
        if res.get("ok"):
            prev = (f"上一轮 validation {spec.get('primary_metric')}={score}。"
                    "在此基础上改进;若已到平台期可改换模型族或特征处理方式。")
        else:
            prev = ("上一轮执行失败,报错:\n```\n"
                    + (res.get("error") or "")[-1500:] +
                    "\n```\n请修复后输出完整 predict.py。")
    # 最终模型 = validation 最好的那轮(没有则最后一轮)
    best_src = work / "predict_best.py"
    if best_src.exists():
        shutil.copy2(best_src, work / "predict.py")
    sh = _shadow_predict(work, run_dir,
                         out_dir / name / "shadow_score.parquet", env=env)
    if not sh.get("ok") and code and not best_src.exists():
        sh = {"ok": False, "error": "无可执行候选"}
    rec = {"dataset": name, "status": "ok" if sh.get("ok") else "failed",
           "model": model, "tier": tier, "rounds": len(log),
           "gpu": gpu, "threads": int(n_threads),
           "mode": "single_model" if single_model else "free",
           "strategy_audit": _audit_strategy(work),
           "training_source_audit": BENV.audit_training_source(
               (work / "predict.py").read_text(encoding="utf-8")
               if (work / "predict.py").exists() else "",
               (r"validation_labels", r"validation\.parquet")),
           "best_validation": best, "wall_hours": round(
               (time.time() - t0) / 3600, 2),
           "shadow": sh, "log": log if keep else log[-6:]}
    (out_dir / name).mkdir(parents=True, exist_ok=True)
    (out_dir / name / "run.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--suite", default=str(SKILL / "assets"
                                           / "benchmark-suite.yaml"))
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--tier", default="full",
                    choices=("fast", "full", "large"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--wall-hours", type=float, default=None)
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--threads", type=int, default=RAW_AGENT_THREADS,
                    help="执行层线程上限(缺省 24,也可用 RAW_AGENT_THREADS)")
    ap.add_argument("--gpu", default=os.environ.get("RAW_AGENT_GPU"),
                    help="手工指定物理卡号;缺省按实际占用自动挑")
    ap.add_argument("--no-gpu", action="store_true", help="强制 CPU")
    ap.add_argument("--single-model", action="store_true",
                    help="单模型公平对照:提示词禁止一切融合/多模型/多折平均")
    a = ap.parse_args(argv)
    suite = _read_yaml(Path(a.suite))
    only = {x for x in (a.only or "").split(",") if x}
    out_dir = Path(a.out_dir)
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
        print(f"[raw-agent] {name} ... gpu={gpu} threads={a.threads}",
              flush=True)
        rec = run_one(ds, Path(a.runs_root), out_dir, model=a.model,
                      tier=a.tier, max_rounds=a.max_rounds,
                      wall_hours=a.wall_hours, keep=a.keep_work,
                      gpu=gpu, n_threads=a.threads,
                      single_model=a.single_model)
        print(f"[raw-agent] {name}: {rec['status']} "
              f"val={rec.get('best_validation')} "
              f"{rec.get('shadow', {}).get('error', '')[:200]}", flush=True)
        with (out_dir / "raw-agent.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
