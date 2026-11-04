"""prescreen 子命令(v2.17.0 WP5b #16):train 内时间切片的多 seed 预筛。

背景(run4 regression.2):agent 每代手写 `adhoc/screen_g0x_*.py` ——
按 pday 排序 train,前 80% 拟合 / 5% 早停 / 末 15% 留出,多 seed 对照,
"只读 train 特征面/标签,不读 validation/shadow 标签"全靠自觉。两次分类侧
预筛都正确预言了方向;回归侧 SE≈30,只能排除坏区。本命令把它收编为官方
能力:

    modelevo.py prescreen --run R --trial <候选> [--ref <对照,缺省父代>]
        --seeds K --holdout-frac 0.15 --es-frac 0.05 [--max-rows N]

- 只用 train 面:按 identity 面时间列排序,前段拟合、中段早停、尾段留出
  (runtime.data.build_prescreen_splits,流式、内存有界、同参数复用);
- 在**沙箱副本**里原样运行候选与对照的 trial 代码(K 个 seed,配对):
  沙箱镜像 run 布局,trial 的 SPLITS(相对 __file__)天然指向沙箱数据面,
  train/validation 分别是拟合(+早停尾段)/留出段;MODELEVO_PRESCREEN_ROOT
  让 runtime.data 访问函数与 feature_block 把沙箱外路径重映射进沙箱;
  早停面经 MODELEVO_ES_SOURCE=train_tail + 沙箱副本 ES_FRACTION 改写落在
  中段;特征块缓存键含数据内容指纹 → 行集不同自动换键(沙箱自有缓存);
- 不写 trial 目录、不改任何状态;validation/shadow/final_test 的标签与特征
  一律不读(访问日志记入报告);
- 输出 Δ(候选 − 对照,指标原始方向)、配对 bootstrap SE、seed 间 SE、
  逐 seed 结果,落 prescreen/<trial>/report.json,标注
  screening_evidence=true,不进 FDR / leaderboard / 方向台账。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from runtime.logging_utils import dump_json, dump_yaml, load_yaml, now_iso

from .context import (_SKILL_ROOT, _load_config, _load_spec, _run_dir_from,
                      logger)

#: 允许预筛的 trial 状态(CODE_WRITTEN 或更后)
_OK_STATES = ("CODE_WRITTEN", "RUNNING", "SUCCEEDED", "FAILED", "EVALUATED",
              "REFLECTED")
_SEED_RE = re.compile(r"(?m)^SEED\s*=\s*(\d+)\s*$")
_ESF_RE = re.compile(r"(?m)^ES_FRACTION\s*=\s*[0-9.eE+-]+(.*)$")
_SKILL_RE = re.compile(r"(?m)^SKILL_ROOT\s*=\s*.*$")


def _trial_record(run_dir: Path, tid: str) -> dict:
    p = run_dir / "trials" / tid / "trial.yaml"
    if not p.exists():
        raise FileNotFoundError(f"trial 不存在: {tid}")
    return load_yaml(p) or {}


def _prepare_code(src_tdir: Path, dst_tdir: Path, *, seed: int | None,
                  es_frac_train: float) -> dict:
    """复制 code/ + config/ 到沙箱 trial 目录,注入 seed / 早停比例 / skill 根。"""
    if dst_tdir.exists():
        shutil.rmtree(dst_tdir)
    shutil.copytree(src_tdir / "code", dst_tdir / "code",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if (src_tdir / "config").is_dir():
        shutil.copytree(src_tdir / "config", dst_tdir / "config")
    if (src_tdir / "trial.yaml").exists():
        shutil.copy2(src_tdir / "trial.yaml", dst_tdir / "trial.yaml")
    rec = load_yaml(src_tdir / "trial.yaml") or {}
    entry = str((rec.get("execution") or {}).get("code_entry") or "train.py")
    code_p = dst_tdir / "code" / entry
    code = code_p.read_text(encoding="utf-8")
    notes: dict = {"code_entry": entry}
    if seed is not None:
        if _SEED_RE.search(code):
            code = _SEED_RE.sub(f"SEED = {int(seed)}", code, count=1)
            notes["seed_injected"] = True
        else:
            notes["seed_injected"] = False
    if _ESF_RE.search(code):
        code = _ESF_RE.sub(lambda m: f"ES_FRACTION = {es_frac_train!r}"
                           + m.group(1), code, count=1)
        notes["es_fraction_injected"] = True
    else:
        notes["es_fraction_injected"] = False
    notes["es_source_env_supported"] = "MODELEVO_ES_SOURCE" in code
    if _SKILL_RE.search(code):
        code = _SKILL_RE.sub(f"SKILL_ROOT = {str(_SKILL_ROOT)!r}", code,
                             count=1)
    code_p.write_text(code, encoding="utf-8")
    cfg_p = dst_tdir / "config" / "trial.yaml"
    if cfg_p.exists():
        cfg = load_yaml(cfg_p) or {}
        cfg["multi_seed"] = 1          # 预筛的 seed 由本命令外层控制
        if str(cfg.get("search_action") or "") == "confirm":
            cfg["search_action"] = "prescreen"
        if seed is not None:
            cfg["seed"] = int(seed)
        dump_yaml(cfg_p, cfg)
    return notes


def _run_one(sandbox: Path, src_tdir: Path, name: str, *, seed: int | None,
             es_frac_train: float, threads: int, gpu: str | None,
             timeout_min: float) -> dict:
    from runtime.scheduler import run_trial_process

    dst = sandbox / "trials" / name
    notes = _prepare_code(src_tdir, dst, seed=seed,
                          es_frac_train=es_frac_train)
    rec = load_yaml(src_tdir / "trial.yaml") or {}
    command = str((rec.get("execution") or {}).get("command")
                  or f"python {notes['code_entry']} --config "
                     "../config/trial.yaml").split()
    if command and command[0] == "python":
        command[0] = sys.executable
    env = {
        "MODELEVO_TRIAL_EXEC": name,
        "MODELEVO_TRIAL_ID": name,
        "MODELEVO_RUN_DIR": str(sandbox),
        "MODELEVO_PRESCREEN": "1",
        "MODELEVO_PRESCREEN_ROOT": str(sandbox),
        "MODELEVO_ES_SOURCE": "train_tail",
        "MODELEVO_QUICK_RATIO": "0",
        "MODELEVO_SMOKE_RUN": "0",
        "OMP_NUM_THREADS": str(int(threads)),
        "MODELEVO_DEADLINE_TS": str(time.time() + timeout_min * 60 - 30),
        "PYTHONPATH": os.pathsep.join(
            [str(_SKILL_ROOT)] + ([os.environ["PYTHONPATH"]]
                                  if os.environ.get("PYTHONPATH") else [])),
    }
    if gpu not in (None, ""):
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "MODELEVO_GPU": "1"})
    else:
        env.update({"CUDA_VISIBLE_DEVICES": "", "MODELEVO_GPU": "0"})
    res = run_trial_process(
        command, cwd=dst / "code", stdout_path=dst / "stdout.log",
        stderr_path=dst / "stderr.log", timeout_sec=timeout_min * 60,
        grace_sec=10, env_extra=env, heartbeat_log=dst / "trial.log")
    out = {"name": name, "seed": seed, "ok": bool(res.ok),
           "returncode": res.returncode,
           "runtime_sec": round(float(res.runtime_sec or 0), 2),
           "peak_memory_gb": res.peak_memory_gb,
           "failure_reason": res.failure_reason, **notes}
    sp = dst / "validation_score.parquet"
    if not res.ok or not sp.exists():
        out["ok"] = False
        out["stderr_tail"] = (res.stderr_tail or "")[-800:]
        return out
    out["score_path"] = str(sp)
    return out


def _metric(metric: str, y, s, treatment=None) -> float:
    import numpy as np

    from runtime.metrics import REGRESSION_METRICS, compute_metric

    s = np.asarray(s, dtype=float)
    return float(compute_metric(
        metric, y, s if metric in REGRESSION_METRICS else None, score=s,
        treatment=treatment))


def cmd_prescreen(args) -> int:
    import numpy as np
    import pandas as pd

    from runtime.data import PrescreenAccessLog, build_prescreen_splits
    from runtime.metrics import metric_direction, paired_bootstrap_delta

    spec = _load_spec(args)
    run_dir = _run_dir_from(args)
    cfg = _load_config(run_dir)
    cand_id = str(args.trial)
    try:
        cand = _trial_record(run_dir, cand_id)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    if str(cand.get("status")) not in _OK_STATES:
        logger.error("prescreen 需要候选处于 CODE_WRITTEN 或更后状态(当前 %s)",
                     cand.get("status"))
        return 1
    ref_id = args.ref or next(iter(cand.get("parent_trial_ids") or []), None)
    if not ref_id:
        logger.error("候选 %s 无父代,请用 --ref 指定对照 trial", cand_id)
        return 2
    try:
        ref = _trial_record(run_dir, str(ref_id))
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    if str(ref.get("status")) not in _OK_STATES:
        logger.error("对照 %s 状态 %s 不可运行", ref_id, ref.get("status"))
        return 1
    view = str(cand.get("data_view") or "v1")
    if str(ref.get("data_view") or "v1") != view:
        logger.error("候选(%s)与对照(%s)不在同一数据视图,train 行集不同,"
                     "预筛不可配对", view, ref.get("data_view") or "v1")
        return 1
    if view == "sample":
        logger.warning("候选在抽样面:预筛使用抽样面 train(行数更少,SE 更大)")
    K = max(1, int(args.seeds))
    hf, ef = float(args.holdout_frac), float(args.es_frac)
    if not (0 < hf < 0.5) or not (0 < ef < 0.5) or hf + ef >= 0.8:
        logger.error("holdout-frac/es-frac 需在 (0, 0.5) 且二者之和 < 0.8")
        return 2
    max_rows = int(args.max_rows) if args.max_rows else None
    tag = hashlib.sha256(json.dumps(
        [view, hf, ef, max_rows or 0], sort_keys=True).encode()
    ).hexdigest()[:10]
    sandbox = run_dir / "prescreen" / "_sandbox" / f"{view}-{tag}"
    sandbox.mkdir(parents=True, exist_ok=True)
    access = PrescreenAccessLog()
    t0 = time.time()
    try:
        info = build_prescreen_splits(
            run_dir, spec, sandbox, view_id=view, holdout_frac=hf,
            es_frac=ef, max_rows=max_rows, access_log=access)
    except Exception as exc:  # noqa: BLE001
        logger.error("prescreen 数据面构建失败: %s", exc)
        return 1
    for name in ("task-spec.yaml", "modelevo-config.yaml"):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, sandbox / name)
    if (run_dir / "views").is_dir() and not (sandbox / "views").exists():
        shutil.copytree(run_dir / "views", sandbox / "views",
                        ignore=shutil.ignore_patterns("__pycache__"))
    print(f"prescreen 数据面: fit={info['n_fit']} es={info['n_es']} "
          f"holdout={info['n_holdout']}(train 源 {info['n_train_source']} 行"
          f"{',复用' if info.get('reused') else ''}) → {info['splits_dir']}")
    ho_dir = Path(info["splits_dir"])
    ho_lab = pd.read_parquet(ho_dir / "labels" / "validation.parquet")
    y = ho_lab[spec.target_name].to_numpy()
    tcol = spec.treatment_column
    treat = ho_lab[tcol].to_numpy() if tcol and tcol in ho_lab.columns \
        else None
    metric = (getattr(args, "metric", None) or spec.primary_metric).lower()
    higher = metric_direction(metric)
    threads = int(args.threads or max(1, min(16, (os.cpu_count() or 4) // 2)))
    es_frac_train = float(info["es_fraction_of_sandbox_train"]) or 0.05
    base_seed = None
    _src_code = (run_dir / "trials" / cand_id / "code" / str(
        (cand.get("execution") or {}).get("code_entry") or "train.py"))
    if _src_code.exists():
        _m = _SEED_RE.search(_src_code.read_text(encoding="utf-8"))
        base_seed = int(_m.group(1)) if _m else None
    token = f"{os.getpid()}{int(time.time()) % 100000}"
    per_seed: list[dict] = []
    cand_scores, ref_scores = [], []
    for k in range(K):
        seed = (base_seed + 1000 * k) if base_seed is not None else None
        row: dict = {"k": k, "seed": seed}
        for role, tid in (("cand", cand_id), ("ref", str(ref_id))):
            r = _run_one(sandbox, run_dir / "trials" / tid,
                         f"{tid}__ps{token}-{role}-s{k}", seed=seed,
                         es_frac_train=es_frac_train, threads=threads,
                         gpu=args.gpu, timeout_min=float(args.timeout_min))
            row[role] = {kk: r.get(kk) for kk in (
                "ok", "runtime_sec", "peak_memory_gb", "failure_reason",
                "seed_injected", "es_fraction_injected",
                "es_source_env_supported", "stderr_tail") if kk in r}
            if not r.get("ok"):
                logger.error("prescreen %s(%s, seed=%s)运行失败: %s",
                             role, tid, seed, r.get("failure_reason")
                             or r.get("stderr_tail"))
                per_seed.append(row)
                _dump_report(run_dir, cand_id, _report_head(
                    cand_id, ref_id, view, metric, higher, K, hf, ef,
                    max_rows, info, access, sandbox) | {
                    "status": "failed", "per_seed": per_seed})
                return 1
            s = pd.read_parquet(r["score_path"])["score"].to_numpy(float)
            if len(s) != len(y):
                logger.error("%s 留出段打分 %d 行 != 留出段 %d 行", role,
                             len(s), len(y))
                return 1
            (cand_scores if role == "cand" else ref_scores).append(s)
            row[role]["metric"] = _metric(metric, y, s, treat)
        row["delta"] = row["cand"]["metric"] - row["ref"]["metric"]
        per_seed.append(row)
        print(f"  seed {seed}: cand={row['cand']['metric']:.6g} "
              f"ref={row['ref']['metric']:.6g} Δ={row['delta']:+.6g}")
    deltas = np.array([r["delta"] for r in per_seed], dtype=float)
    d_mean = float(deltas.mean())
    se_seed = float(deltas.std(ddof=1) / np.sqrt(K)) if K > 1 else None
    s_c = np.mean(cand_scores, axis=0)
    s_r = np.mean(ref_scores, axis=0)
    pb = paired_bootstrap_delta(y, s_c, s_r, metric, treatment=treat,
                                n_boot=int(args.n_boot))
    se_boot = pb.get("se")
    parts = [v for v in (se_boot, se_seed) if v is not None]
    se = float(np.sqrt(sum(v * v for v in parts))) if parts else None
    z = (d_mean / se) if se else None
    from statistics import NormalDist

    p_better = None
    if z is not None:
        p_up = NormalDist().cdf(z)
        p_better = p_up if higher else 1.0 - p_up
    try:
        from runtime.mde import effective_mde

        mde = float(effective_mde(cfg, run_dir=run_dir))
    except Exception:  # noqa: BLE001
        mde = None
    gain = d_mean if higher else -d_mean
    if p_better is None:
        verdict = "inconclusive"
    elif p_better >= 0.9 and (mde is None or gain >= 0.5 * mde):
        verdict = "likely_better"
    elif p_better <= 0.1:
        verdict = "likely_worse"
    else:
        verdict = "inconclusive"
    report = _report_head(cand_id, ref_id, view, metric, higher, K, hf, ef,
                          max_rows, info, access, sandbox) | {
        "status": "ok",
        "per_seed": per_seed,
        "delta_mean": d_mean,
        "delta_definition": "metric(cand) − metric(ref),指标原始方向",
        "se_seed": se_seed,
        "paired_bootstrap": {k: pb.get(k) for k in (
            "delta_mean", "delta_ci", "se", "z_score", "p_positive")}
        | {"on": "seed 平均分数", "n_boot": int(args.n_boot)},
        "se": se,
        "se_note": "se = sqrt(se_boot² + se_seed²)(行抽样 + 训练随机性,保守)",
        "z": z,
        "p_better": p_better,
        "mde": mde,
        "resolution": (None if (se is None or mde is None) else
                       ("可分辨 MDE 量级" if se <= mde else
                        "SE > MDE:只能排除坏区,不能确认小增益")),
        "screen_verdict": verdict,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_p = _dump_report(run_dir, cand_id, report)
    if not args.keep_sandbox:
        for p in (sandbox / "trials").glob(f"*__ps{token}-*"):
            shutil.rmtree(p, ignore_errors=True)
    print(f"prescreen {cand_id} vs {ref_id}  metric={metric}  K={K}  "
          f"holdout={info['n_holdout']}")
    print(f"  Δ={d_mean:+.6g}  SE={se if se is None else round(se, 6)}  "
          f"(boot={se_boot if se_boot is None else round(se_boot, 6)}, "
          f"seed={se_seed if se_seed is None else round(se_seed, 6)})  "
          f"z={z if z is None else round(z, 2)}  p_better="
          f"{p_better if p_better is None else round(p_better, 3)}  "
          f"→ {verdict}")
    print("  screening_evidence=true:不进 FDR / leaderboard / 方向台账;"
          "晋级仍须正式 trial 在 validation 上裁决")
    print(f"  -> {out_p}")
    return 0


def _report_head(cand_id, ref_id, view, metric, higher, K, hf, ef, max_rows,
                 info, access, sandbox) -> dict:
    return {
        "trial": cand_id, "ref": str(ref_id), "data_view": view,
        "created_at": now_iso(),
        "screening_evidence": True,
        "enters_fdr": False, "enters_leaderboard": False,
        "enters_direction_ledger": False,
        "metric": metric,
        "direction": "higher_better" if higher else "lower_better",
        "seeds": int(K),
        "params": {"holdout_frac": hf, "es_frac": ef,
                   "max_rows": max_rows},
        "split": {k: info.get(k) for k in (
            "ordering", "n_train_source", "n_used", "n_fit", "n_es",
            "n_holdout", "fit_time_range", "es_time_range",
            "holdout_time_range", "es_fraction_of_sandbox_train")},
        "data_access": list(access),
        "data_access_note": "harness 侧只读真实 run 的 train 四产物;trial "
                            "代码在沙箱内运行,沙箱不含 validation/shadow/"
                            "final_test 的任何数据",
        "sandbox": str(sandbox),
    }


def _dump_report(run_dir: Path, cand_id: str, report: dict) -> Path:
    out_dir = run_dir / "prescreen" / cand_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / "report.json"
    dump_json(out_p, report)
    return out_p
