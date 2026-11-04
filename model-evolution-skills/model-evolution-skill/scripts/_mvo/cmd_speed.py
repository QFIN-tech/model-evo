"""v2.13.5 提速与可观测命令:contrast / timing-report / feature-cache /
finalize-all / wait --generation|--job。

来源:req-t7-rand400w(23.6h wall)的耗时拆解 —— 评估 39%、seq 解析 34%、
GPU 训练 7%;另有 agent 手写 bootstrap 对照 60 min、finalize 串行 2.2h、
sleep 轮询最多晚 10 min 发现完成。这些都是框架缺口,不是模型问题。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from runtime.logging_utils import dump_json, load_json, now_iso  # noqa: E402

from .context import (_SKILL_ROOT, _load_spec, _run_dir_from,
                      _trial_reached_terminal, logger)


# ---------------------------------------------------------------------------
# contrast:兄弟 trial 之间的配对 bootstrap 对照(validation 面)
# ---------------------------------------------------------------------------
def cmd_contrast(args) -> int:
    """任意 trial 两两(或对 --ref)的配对 bootstrap Δ(validation 面)。

    req-t7-rand400w gen-01:agent 为回答"当日口径 vs 保守口径"手写
    bootstrap 脚本,sklearn 逐次重排,四臂对照跑了 ~60 min。本命令复用
    harness 的 paired_bootstrap_delta(快路径 O(n)/次,2000 次约 1 min),
    按样本键对齐(跨视图不可比时拒绝),结果落 logs/contrast-*.json。
    只读 validation(shadow 是确认面、final_test 是盲测面,均不开放)。
    """
    import numpy as np
    import pandas as pd

    from runtime.data import SAMPLE_KEY_COLUMN, view_splits_dir
    from runtime.metrics import (align_score_by_keys, compute_metric,
                                 metric_direction, paired_bootstrap_delta)

    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    ids = [t.strip() for t in str(args.trials).split(",") if t.strip()]
    if len(ids) < 2:
        logger.error("contrast 需要至少两个 trial(--trials A,B[,C...])")
        return 2
    metric = (args.metric or spec.primary_metric).lower()
    direction = metric_direction(metric)
    scores: dict[str, pd.DataFrame] = {}
    views: dict[str, str] = {}
    from runtime.logging_utils import load_yaml

    for tid in ids:
        tdir = run_dir / "trials" / tid
        sp = tdir / "validation_score.parquet"
        if not sp.exists():
            logger.error("%s 无 validation_score.parquet", tid)
            return 1
        rec = load_yaml(tdir / "trial.yaml") if (tdir / "trial.yaml").exists() \
            else {}
        views[tid] = str((rec or {}).get("data_view") or "v1")
        scores[tid] = pd.read_parquet(sp)
    if len(set(views.values())) > 1:
        logger.error("跨数据视图不可配对:%s(行集不同,配对 bootstrap 无意义)",
                     views)
        return 1
    view = next(iter(views.values()))
    lab = pd.read_parquet(view_splits_dir(run_dir, view) / "labels"
                          / "validation.parquet")
    y = lab[spec.target_name].to_numpy()
    tcol = spec.treatment_column
    treatment = lab[tcol].to_numpy() if tcol and tcol in lab.columns else None
    aligned: dict[str, np.ndarray] = {}
    for tid in ids:
        df = scores[tid]
        if len(df) != len(y):
            logger.error("%s 分数 %d 行 != validation %d 行", tid, len(df),
                         len(y))
            return 1
        if SAMPLE_KEY_COLUMN in df.columns and SAMPLE_KEY_COLUMN in lab.columns:
            aligned[tid] = align_score_by_keys(
                df["score"].to_numpy(), lab[SAMPLE_KEY_COLUMN].to_numpy(),
                df[SAMPLE_KEY_COLUMN].to_numpy())
        else:
            aligned[tid] = df["score"].to_numpy(dtype=float)
    point = {tid: compute_metric(metric, y, s, score=s, treatment=treatment)
             for tid, s in aligned.items()}
    pairs = []
    if args.ref:
        if args.ref not in aligned:
            logger.error("--ref %s 不在 --trials 里", args.ref)
            return 2
        pairs = [(t, args.ref) for t in ids if t != args.ref]
    else:
        pairs = [(ids[i], ids[j]) for i in range(len(ids))
                 for j in range(i + 1, len(ids))]
    rows = []
    t0 = time.time()
    for a, b in pairs:
        r = paired_bootstrap_delta(y, aligned[a], aligned[b], metric,
                                   treatment=treatment,
                                   n_boot=int(args.n_boot),
                                   seed=int(args.seed))
        p_pos = r.get("p_positive")
        rows.append({
            "a": a, "b": b,
            "metric_a": point[a], "metric_b": point[b],
            "delta_mean": r.get("delta_mean"),
            "delta_ci": list(r.get("delta_ci") or ()),
            "se": r.get("se"), "z": r.get("z_score"),
            "p_better": (p_pos if direction else
                         (1.0 - p_pos if p_pos is not None else None)),
        })
    elapsed = time.time() - t0
    print(f"contrast  metric={metric}  view={view}  n={len(y)}  "
          f"n_boot={args.n_boot}  ({elapsed:.1f}s)")
    print("a\tb\tdelta(a-b)\tci95\tz\tp_better(a)")
    for r in rows:
        ci = r["delta_ci"]
        print(f"{r['a']}\t{r['b']}\t{r['delta_mean']:+.6f}\t"
              f"[{ci[0]:+.6f},{ci[1]:+.6f}]\t"
              f"{(r['z'] if r['z'] is not None else float('nan')):.2f}\t"
              f"{(r['p_better'] if r['p_better'] is not None else float('nan')):.3f}")
    out = {"at": now_iso(), "metric": metric, "view": view, "n": int(len(y)),
           "n_boot": int(args.n_boot), "seed": int(args.seed),
           "point": point, "pairs": rows,
           "note": "validation 面配对 bootstrap(harness 口径);只作分析证据,"
                   "不改任何 trial 的 verdict,不计入代级 alpha"}
    out_path = Path(args.out) if args.out else (
        run_dir / "logs" / f"contrast-{datetime.now():%Y%m%d-%H%M%S}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out_path, out)
    print(f"-> {out_path}")
    return 0


# ---------------------------------------------------------------------------
# timing-report:run 的耗时拆解(trial 分项 + 代际窗口)
# ---------------------------------------------------------------------------
_STEP_RE = re.compile(r"\[step\] (.*?) \(([\d.]+)s\)")
_TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"


def _ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


def _step_category(label: str) -> str:
    low = label.lower()
    if any(k in low for k in ("widen", "feature block", "parse", "reparse",
                              "vocab", "build_wide")):
        return "feature"
    # v2.16.3:HPO 之后的 outer 重训(`hurdle hpo refit done` /
    # `超参生效(hpo_refit)`)是拟合,不是搜索;先于 "hpo" 判定
    if "refit" in low:
        return "fit"
    if "hpo" in low or "fit_and_score" in low:
        return "hpo"
    if low.startswith("train") or "retrain" in low or "fit" in low:
        return "fit"
    if any(k in low for k in ("score", "shadow", "dump", "predict")):
        return "score"
    if any(k in low for k in ("load", "dtype", "es_split", "governance")):
        return "load"
    return "other"


#: v2.16.3:HPO inner 搜索窗口的起止标记(模板 `hpo search start/done`,
#: hurdle 超参段 `hurdle hpo search start/done`)
_HPO_SEARCH_START = "hpo search start"
_HPO_SEARCH_DONE = "hpo search done"


def _step_segments(text: str, run_sec: float | None = None
                   ) -> tuple[dict, float]:
    """stdout.log 的 [step] 行 → (训练段分项秒数, HPO inner 搜索秒数)。

    每个 [step] 行带"自进程启动以来的累计秒数";相邻两行之差记入**后一行**
    的类别。v2.16.3:
    - ``hpo search start`` 之后到 ``hpo search done``(含)为止的分段全部记入
      HPO 搜索(返回值第二项)。旧实现按行标签分类,inner 每个 config 的
      ``hurdle clf fit start/done`` 等行都落进 ``fit``(run4 regression
      gen-05 trial-022:约 2900 s 的 inner 搜索被报成 train:fit);搜索未见
      done(中断/超时)时窗口延续到末尾,``tail`` 也归搜索;
    - 累计秒数以**单调最大值**为基准:并行 config 的日志可能交错(后打印的行
      秒数略小),旧实现以回退值为新基准,下一段重复计时。
    守恒:Σ分项 + 搜索 == max(最后一行秒数, run_sec)。
    """
    cats: dict[str, float] = {}
    hpo = 0.0
    prev = 0.0
    in_search = False
    for label, sec in _STEP_RE.findall(text):
        seg = max(0.0, float(sec) - prev)
        low = label.lower()
        if in_search:
            hpo += seg
            if _HPO_SEARCH_DONE in low:
                in_search = False
        else:
            cat = _step_category(label)
            cats[cat] = cats.get(cat, 0.0) + seg
            if _HPO_SEARCH_START in low:
                in_search = True
        prev = max(prev, float(sec))
    if run_sec is not None and prev and float(run_sec) > prev:
        tail = float(run_sec) - prev
        if in_search:
            hpo += tail
        else:
            cats["tail"] = cats.get("tail", 0.0) + tail
    return cats, hpo


def trial_timing(tdir: Path) -> dict:
    """单个 trial 的分项耗时(秒)。数据源:trial.log(起止与评估段)、
    stdout.log 的 [step] 行(训练段分项)、evaluation.json 的 timings
    (v2.13.5 起;更早的 trial 只有 group_importance.gi_elapsed_sec)。

    v2.16.3:HPO trial 的 inner 搜索单列 ``hpo_search_sec``,不再混进
    ``train_phases``(其中旧 ``fit`` 吃掉了 inner 每个 config 的拟合);
    ``train_phases`` 总和 + ``hpo_search_sec`` == 训练进程耗时(守恒)。
    """
    out: dict = {"trial_id": tdir.name}
    tl = tdir / "trial.log"
    text = tl.read_text(encoding="utf-8", errors="replace") if tl.exists() \
        else ""
    st = re.findall(_TS + r" \[runner\.py:\d+\]\[run\] INFO: 开始执行", text)
    tr = re.findall(_TS + r" \[runner\.py:\d+\]\[run\] INFO: 训练完成: "
                          r"runtime=([\d.]+)s", text)
    ev = re.findall(_TS + r" \[runner\.py:\d+\]\[run\] INFO: 评估完成", text)
    if st:
        out["start"] = st[-1]
    if tr:
        out["run_sec"] = float(tr[-1][1])
    if tr and ev:
        out["eval_sec"] = _ts(ev[-1]) - _ts(tr[-1][0])
    so = tdir / "stdout.log"
    cats: dict[str, float] = {}
    hpo_sec = 0.0
    if so.exists():
        cats, hpo_sec = _step_segments(
            so.read_text(encoding="utf-8", errors="replace"),
            out.get("run_sec"))
    out["train_phases"] = {k: round(v, 1) for k, v in cats.items()}
    if hpo_sec > 0:
        out["hpo_search_sec"] = round(hpo_sec, 1)
    ej = tdir / "evaluation.json"
    if ej.exists():
        try:
            ev_d = load_json(ej)
            if isinstance(ev_d.get("timings"), dict):
                out["eval_phases"] = ev_d["timings"]
            out["verdict"] = ev_d.get("verdict")
            gi = ev_d.get("group_importance") or {}
            if "gi_elapsed_sec" in gi and "eval_phases" not in out:
                out["eval_phases"] = {"group_importance": gi["gi_elapsed_sec"]}
        except Exception:  # noqa: BLE001
            pass
    return out


def cmd_timing_report(args) -> int:
    run_dir = _run_dir_from(args)
    tdirs = sorted(p for p in (run_dir / "trials").glob("trial-*")
                   if p.is_dir())
    rows = [trial_timing(p) for p in tdirs]
    agg: dict[str, float] = {}
    for r in rows:
        for k, v in (r.get("train_phases") or {}).items():
            agg["train:" + k] = agg.get("train:" + k, 0.0) + float(v)
        # v2.16.3:HPO inner 搜索单列(与 train:fit / eval 并列),不再混进 fit
        if r.get("hpo_search_sec"):
            agg["hpo_search"] = agg.get("hpo_search", 0.0) + float(
                r["hpo_search_sec"])
        if r.get("eval_sec"):
            agg["eval"] = agg.get("eval", 0.0) + float(r["eval_sec"])
    total = sum(agg.values()) or 1.0
    # 代际窗口:同代 trial 的最早开始 → 最晚评估完成
    gens: dict[str, list] = {}
    for r in rows:
        m = re.match(r"trial-\d+-g(\d+)-", r["trial_id"])
        if not m or "start" not in r:
            continue
        end = _ts(r["start"]) + float(r.get("run_sec") or 0) + \
            float(r.get("eval_sec") or 0)
        gens.setdefault(m.group(1), []).append((_ts(r["start"]), end))
    gen_rows = []
    for g in sorted(gens):
        s = min(a for a, _ in gens[g])
        e = max(b for _, b in gens[g])
        gen_rows.append({"generation": int(g), "start":
                         datetime.fromtimestamp(s).strftime("%m-%d %H:%M"),
                         "end": datetime.fromtimestamp(e).strftime(
                             "%m-%d %H:%M"),
                         "wall_min": round((e - s) / 60, 1),
                         "n_trials": len(gens[g])})
    for i in range(1, len(gen_rows)):
        prev_end = datetime.strptime(gen_rows[i - 1]["end"], "%m-%d %H:%M")
        cur_start = datetime.strptime(gen_rows[i]["start"], "%m-%d %H:%M")
        gen_rows[i]["gap_before_min"] = round(
            (cur_start - prev_end).total_seconds() / 60, 1)
    report = {"run": str(run_dir), "at": now_iso(),
              "trial_hours": round(total / 3600, 2),
              "phase_share": {k: round(v / total, 3)
                              for k, v in sorted(agg.items(),
                                                 key=lambda kv: -kv[1])},
              "phase_hours": {k: round(v / 3600, 2) for k, v in agg.items()},
              "generations": gen_rows, "trials": rows}
    if getattr(args, "json", False):
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"trial-hours={report['trial_hours']}  (阶段占比,越靠前越该优化)")
        for k, v in report["phase_share"].items():
            print(f"  {k:18s} {report['phase_hours'][k]:6.2f} h  {v:6.1%}")
        print("代际窗口(同代首个 trial 开始 → 最后评估完成;gap = 两代之间"
              "的规划/反思/等待):")
        for g in gen_rows:
            print(f"  gen-{g['generation']:02d} {g['start']} → {g['end']}  "
                  f"{g['wall_min']:6.1f} min  trials={g['n_trials']}"
                  + (f"  gap_before={g['gap_before_min']} min"
                     if "gap_before_min" in g else ""))
        print("trial 明细(fit = 搜索窗口外的拟合/重训;hpo_search = HPO inner "
              "搜索,v2.16.3 起单列):")
        for r in rows:
            tp = r.get("train_phases") or {}
            ep = r.get("eval_phases") or {}
            print(f"  {r['trial_id']:32s} run={r.get('run_sec', 0):7.0f}s "
                  f"eval={r.get('eval_sec', 0) or 0:6.0f}s "
                  f"feature={tp.get('feature', 0):6.0f}s "
                  f"fit={tp.get('fit', 0):6.0f}s "
                  f"hpo_search={r.get('hpo_search_sec', 0) or 0:6.0f}s "
                  f"gi={ep.get('group_importance', '-')}")
    out_path = run_dir / "logs" / "timing-report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out_path, report)
    return 0


# ---------------------------------------------------------------------------
# feature-cache:列出 / 清理特征块缓存
# ---------------------------------------------------------------------------
def cmd_feature_cache(args) -> int:
    """列出/清空特征块缓存;v2.17.0 增加子模型缓存与失效预判。

    - 缺省:列出 .feature-cache 条目 + 失效预判(``predict_invalidation``)
      + .submodel-cache 汇总;
    - ``--clear``:清空 .feature-cache(行为不变);
    - ``--clear-submodel``:清空 .submodel-cache(HPO inner 未命中的
      子模型也会落盘,没有自动回收)。两类缓存都可重算,不碰任何裁决产物。
    """
    import shutil

    from runtime.feature_cache import cache_inventory, cache_root

    run_dir = _run_dir_from(args)
    root = cache_root(run_dir)  # v2.17.3:MODELEVO_FEATURE_CACHE_DIR 可指向共享目录
    try:
        from runtime import submodel_cache as _smc
    except Exception:  # noqa: BLE001
        _smc = None
    sm_root = run_dir / (_smc.CACHE_DIRNAME if _smc else ".submodel-cache")
    if getattr(args, "clear", False) or getattr(args, "clear_submodel", False):
        if getattr(args, "clear", False) and root.exists():
            shutil.rmtree(root)
            logger.info("已清空特征块缓存 %s", root)
        if getattr(args, "clear_submodel", False) and sm_root.exists():
            shutil.rmtree(sm_root)
            logger.info("已清空子模型缓存 %s", sm_root)
        return 0
    inv = cache_inventory(run_dir)
    if not inv:
        print(f"(无特征块缓存:{root})")
    else:
        total = 0
        print("block\tsplit\trows\tcols\tMB\tcompute_s\tcode\twriter")
        for m in inv:
            total += int(m.get("bytes") or 0)
            print(f"{m.get('name')}\t{m.get('split')}\t{m.get('n_rows')}\t"
                  f"{len(m.get('columns') or [])}\t"
                  f"{(m.get('bytes') or 0) / 2**20:.0f}\t"
                  f"{m.get('compute_sec')}\t{m.get('code_fingerprint')}\t"
                  f"{Path(str(m.get('writer') or '')).parent.parent.name or m.get('writer')}")
        print(f"total {total / 2**30:.2f} GB in {root}")
        # v2.17.0:失效预判(缓存键只含内核/数据内容/CACHE_SCHEMA_VERSION)
        try:
            from runtime.feature_cache import predict_invalidation

            pred = predict_invalidation(run_dir)
            print("CACHE_INVALIDATION\t" + json.dumps(
                {k: pred.get(k) for k in pred if k != "stale_dirs"},
                ensure_ascii=False, default=str))
        except Exception as exc:  # noqa: BLE001
            print(f"(失效预判不可用:{exc})")
    if _smc is not None:
        sm = _smc.cache_inventory(run_dir)
        if sm:
            sm_bytes = sum(int(m.get("bytes") or 0) for m in sm)
            sm_fit = sum(float(m.get("fit_sec") or 0) for m in sm)
            print(f"submodel-cache {len(sm)} 条,{sm_bytes / 2**30:.2f} GB,"
                  f"累计拟合 {sm_fit / 3600:.2f} h 可复用({sm_root});"
                  "清理用 --clear-submodel")
    return 0


# ---------------------------------------------------------------------------
# finalize-all:冠军重训 ∥ baseline 盲测分母(两者独立,不再串行)
# ---------------------------------------------------------------------------
def _cli_argv(*rest: str) -> list[str]:
    return [sys.executable, str(_SKILL_ROOT / "scripts" / "modelevo.py"),
            *rest]


def cmd_finalize_all(args) -> int:
    """finalize-retrain 与 finalize-baseline-score 并行执行。

    req-t7-rand400w:两者串行 65 + 44 min,而它们读写的是 final-model/ 下
    互不相交的产物(冠军重训 vs baseline_final_test_score),完全独立。
    本命令各起一个子进程(日志 logs/finalize-retrain.log /
    logs/finalize-baseline-score.log),CPU 线程按可用核对半分,两者都
    成功才返回 0。evaluate-final(一次性盲测)**不**包含在内,仍须显式执行。
    """
    run_dir = _run_dir_from(args)
    from .cmd_trial import _maybe_detach

    _rc = _maybe_detach(args, run_dir, "finalize-all")
    if _rc is not None:
        return _rc
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cores = os.cpu_count() or 16
    half = str(max(8, cores // 2 - 2))
    common = ["--run", str(run_dir)]
    if getattr(args, "spec", None):
        common += ["--spec", str(args.spec)]
    jobs = {
        "finalize-retrain": _cli_argv(
            "finalize-retrain", *common,
            *(["--champion", args.champion] if args.champion else []),
            *(["--snapshot-only"] if args.snapshot_only else []),
            *(["--ack-divergence", args.ack_divergence]
              if getattr(args, "ack_divergence", None) else []),
            *(["--refit-mode", args.refit_mode]
              if getattr(args, "refit_mode", None) else []),
            *(["--full-reproduction"]
              if getattr(args, "full_reproduction", False) else [])),
        "finalize-baseline-score": _cli_argv(
            "finalize-baseline-score", *common,
            *(["--baseline", args.baseline] if args.baseline else [])),
    }
    procs = {}
    t0 = time.time()
    for tag, argv in jobs.items():
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", half)
        fh = open(logs / f"{tag}.log", "a", encoding="utf-8")
        fh.write(f"\n# {now_iso()} finalize-all: {' '.join(argv)}\n")
        fh.flush()
        procs[tag] = (subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT,
                                       env=env, cwd=str(Path.cwd())), fh)
        logger.info("finalize-all: 启动 %s pid=%s(日志 %s)", tag,
                    procs[tag][0].pid, logs / f"{tag}.log")
    rcs = {}
    # v2.17.2(B5):轮询两个子进程,逐个记各自用时(compute_sec = 两者之和,
    # 与 trial Σruntime 同口径);旧实现顺序 wait,第二个的用时被第一个掩盖
    job_sec: dict[str, float] = {}
    while len(rcs) < len(procs):
        for tag, (p, fh) in procs.items():
            if tag in rcs:
                continue
            rc = p.poll()
            if rc is None:
                continue
            rcs[tag] = rc
            job_sec[tag] = round(time.time() - t0, 1)
            fh.close()
            logger.info("finalize-all: %s 结束 rc=%s(%.1f min)", tag,
                        rcs[tag], (time.time() - t0) / 60)
        if len(rcs) < len(procs):
            time.sleep(1.0)
    from runtime.trial import record_timing_log

    record_timing_log(logs / "finalize-all.json", {
        "at": now_iso(), "returncodes": rcs,
        "wall_min": round((time.time() - t0) / 60, 1),
        "wall_sec": round(time.time() - t0, 1),
        "job_sec": job_sec,
        "compute_sec": round(sum(job_sec.values()), 1)})
    print("FINALIZE_ALL\t" + "\t".join(f"{k}={v}" for k, v in rcs.items()))
    return 0 if all(v == 0 for v in rcs.values()) else 1


# ---------------------------------------------------------------------------
# wait --generation / --job:分离进程(run-generation/finalize)的确定性等待
# v2.17.0:实现迁到 `_mvo/cmd_wait.py`(轻量单源,wait 快路径不加载本模块);
# 此处再导出保持旧引用可用。
# ---------------------------------------------------------------------------
from .cmd_wait import wait_generation, wait_job  # noqa: E402,F401
