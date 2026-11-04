"""v2.19.0:confirm seed 变体分卡并行(baseline 血缘 trial 代码通用)。

为什么串行(8 个真实 run 核实):``confirm.parallel_seeds: auto`` 的并行实现
只在 trial 模板 main()(scripts/_mvo/templates.py 的
``_run_seed_variants_parallel``)里;而全部 confirm trial 的 code/train.py 都是
**baseline 血缘**(runtime/baseline.py 渲染、快通道 confirm 复制父代代码),其
multi-seed 段是一个串行 for 循环,从不经过并行路径(metrics 里也没有
multi_seed_parallel 留痕)。3 个 seed 单卡串行 115~230 min。

怎么修:baseline 血缘的 multi-seed 段在主轮拟合后调用 :func:`start_seed_workers`:
- 并行条件与模板同口径(confirm.parallel_seeds != off、本 trial 在 GPU 上、经
  GPU 租约再租到无外来进程的干净卡、内存准入放行),缺一即串行并留痕原因;
- 主进程把本轮已就绪的拟合面(X_fit / y_fit / eval_set / X_va)pickle 到
  /dev/shm(无则 trial 目录),每张额外卡拉起一个 worker 子进程
  (``python -m runtime.seed_parallel``):把 trial 脚本当模块导入(不执行
  main),调用脚本自身的 ``_fit_seed_variant``(与主轮同估计器、同早停口径,
  seed = seed0 + 1000×k),按与串行段相同的打分口径写
  seed_scores/seed-k.parquet —— 文件名即合并键,与串行完全同口径;
- 主进程同时串行训未分给 worker 的变体;:func:`finish_seed_workers` 等 worker
  结束、释放租约与准入预留、清理数据文件,失败变体由调用方串行补齐。
"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .logging_utils import get_logger

logger = get_logger(__name__)

WORKER_FLAG = "--seed-worker"


def _setting(run_dir: Path) -> str:
    v = os.environ.get("MODELEVO_CONFIRM_PARALLEL_SEEDS")
    if not v:
        try:
            from .config import Config

            rc = Path(run_dir) / "modelevo-config.yaml"
            v = Config.load(run_config=rc if rc.exists() else None).get(
                "confirm.parallel_seeds", "auto")
        except Exception:  # noqa: BLE001
            v = "auto"
    v = str("auto" if v is None else v).strip().lower()
    return "off" if v in ("off", "false", "0", "no", "serial") else "auto"


def _rss_tree_gb() -> float | None:
    try:
        import psutil

        p = psutil.Process(os.getpid())
        return sum(q.memory_info().rss for q in
                   [p] + p.children(recursive=True)) / float(2 ** 30)
    except Exception:  # noqa: BLE001
        return None


def plan_seed_parallel(run_dir: Path, trial_dir: Path, n_todo: int, *,
                       occupancy: list | None = None,
                       data_gb: float = 0.0) -> dict:
    """并行计划:mode=parallel|serial、reason、gpus(每个额外 worker 一张卡);
    parallel 时附 _leases / _ticket(由 finish_seed_workers 释放)。"""
    run_dir, trial_dir = Path(run_dir), Path(trial_dir)
    rec: dict = {"mode": "serial", "reason": None, "gpus": [],
                 "n_parallel": 1}
    if n_todo < 1:
        rec["reason"] = "no_variant_todo"
        return rec
    if _setting(run_dir) == "off":
        rec["reason"] = "config_off"
        return rec
    if os.environ.get("MODELEVO_GPU") != "1":
        rec["reason"] = "no_gpu_allocated"
        return rec
    from .config import Config
    from .scheduler import (PEAK_SAFETY_FACTOR, GpuLease, MemoryAdmission,
                            annotate_gpu_occupancy, estimate_trial_peak_gb,
                            gpu_candidates_by_occupancy, query_gpu_occupancy)

    own = {s.strip() for s in os.environ.get(
        "CUDA_VISIBLE_DEVICES", "").split(",") if s.strip()}
    occ = occupancy if occupancy is not None else query_gpu_occupancy()
    if not occ:
        rec["reason"] = "gpu_query_failed"
        return rec
    clean, _busy = gpu_candidates_by_occupancy(
        annotate_gpu_occupancy(occ, run_dir))
    cands = [g for g in clean if str(g) not in own]
    if not cands:
        rec["reason"] = "no_extra_gpu"
        return rec
    rc = run_dir / "modelevo-config.yaml"
    cfg = Config.load(run_config=rc if rc.exists() else None)
    try:
        from .logging_utils import load_yaml

        td = load_yaml(trial_dir / "trial.yaml") or {}
    except Exception:  # noqa: BLE001
        td = {}
    est = estimate_trial_peak_gb(run_dir, td, cfg)
    per = float(est.get("estimate_gb") or 0.0)
    rss = _rss_tree_gb()
    if rss:
        # worker 载入同一拟合面 + 训练:不低于本进程当前占用 ×1.15
        per = max(per, rss * float(PEAK_SAFETY_FACTOR))
    adm = MemoryAdmission(run_dir, cfg)
    ev = adm.evaluate(per + float(data_gb or 0.0))
    head = ev.get("headroom_gb")
    rec.update(per_worker_gb=round(per, 2), headroom_gb=head,
               estimate_source=est.get("source"), gpu_candidates=cands)
    if head is None:
        rec["reason"] = "no_memory_probe"
        return rec
    n_mem = int(max(0.0, float(head) - float(data_gb or 0.0)) // per) \
        if per > 0 else 0
    n_extra = min(int(n_todo), len(cands), n_mem)
    if n_extra < 1:
        rec["reason"] = "insufficient_memory"
        return rec
    leases = []
    for g in cands:
        if len(leases) >= n_extra:
            break
        lease = GpuLease.acquire(run_dir / ".gpu-leases", int(g))
        if lease is not None:
            leases.append(lease)
    if not leases:
        rec["reason"] = "gpu_lease_unavailable"
        return rec
    ticket = None
    if adm.enabled:
        try:
            ticket = adm.acquire(
                f"{trial_dir.name}#seed-par",
                {"estimate_gb": per * len(leases) + float(data_gb or 0.0),
                 "source": "confirm_parallel_seeds",
                 "basis_trial": est.get("basis_trial")}, max_wait_sec=0)
        except Exception:  # noqa: BLE001
            ticket = None
        if ticket is None or not ticket.admitted:
            for lease in leases:
                lease.release()
            rec["reason"] = "admission_rejected"
            return rec
    rec.update(mode="parallel", reason="ok",
               gpus=[lease.gpu_index for lease in leases],
               n_parallel=1 + len(leases), _leases=leases, _ticket=ticket)
    return rec


def _release(plan: dict) -> None:
    for lease in plan.pop("_leases", None) or []:
        try:
            lease.release()
        except Exception:  # noqa: BLE001
            pass
    t = plan.pop("_ticket", None)
    if t is not None:
        try:
            t.release()
        except Exception:  # noqa: BLE001
            pass


def _frame_gb(*objs) -> float:
    tot = 0
    for o in objs:
        try:
            if hasattr(o, "memory_usage"):
                m = o.memory_usage(deep=False)
                tot += int(m.sum() if hasattr(m, "sum") else m)
            elif hasattr(o, "nbytes"):
                tot += int(o.nbytes)
            elif isinstance(o, (list, tuple)):
                tot += int(_frame_gb(*o) * 2 ** 30)
        except Exception:  # noqa: BLE001
            continue
    return tot / float(2 ** 30)


def _data_dir(trial_dir: Path) -> Path:
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return shm
    return Path(trial_dir) / "seed_scores"


def start_seed_workers(run_dir, trial_dir, script_path, todo: list,
                       seed0: int, n_jobs: int, X_fit, y_fit, eval_set,
                       X_va, model_type: str, *,
                       primary_params: dict | None = None,
                       primary_xy: Any = None,
                       occupancy: list | None = None,
                       spawn: Any = None) -> dict:
    """主进程调用:按计划把 todo 中除第一个外的变体分给额外卡上的 worker。

    返回 handle:{"assigned": [k…](worker 负责,调用方串行段跳过), "record":
    留痕, ...};串行时 assigned=[]。主进程保留 todo[0](及未分配的)自己训。
    """
    run_dir, trial_dir = Path(run_dir), Path(trial_dir)
    todo = [int(k) for k in todo]
    data_gb = _frame_gb(X_fit, y_fit, eval_set, X_va)
    plan = plan_seed_parallel(run_dir, trial_dir, max(0, len(todo) - 1),
                              occupancy=occupancy, data_gb=data_gb)
    record = {k: v for k, v in plan.items() if not k.startswith("_")}
    handle: dict = {"assigned": [], "record": record, "_plan": plan,
                    "procs": [], "data_path": None}
    if plan["mode"] != "parallel":
        if record.get("reason") not in ("no_variant_todo",):
            print(f"[step] multi-seed 串行(并行条件不满足: "
                  f"{record.get('reason')})", flush=True)
        return handle
    sd = trial_dir / "seed_scores"
    sd.mkdir(parents=True, exist_ok=True)
    data_path = _data_dir(trial_dir) / (
        f".mevo-seedpar-{trial_dir.name}-{os.getpid()}.pkl")
    handle["data_path"] = str(data_path)
    t0 = time.time()
    try:
        with open(data_path, "wb") as f:
            pickle.dump({"X_fit": X_fit, "y_fit": y_fit,
                         "eval_set": eval_set, "X_va": X_va,
                         "model_type": model_type, "n_jobs": int(n_jobs),
                         # v2.20.1:worker 是独立进程,模块级 _PRIMARY_FIT_*
                         # 不会随进程传递。不显式下发会让 _fit_seed_variant
                         # 落进兜底分支(把 eval_set 拼进训练集 + 丢配方),
                         # 产出与主轮不同配方的伪 seed 分数。
                         "primary_params": primary_params,
                         "primary_xy": primary_xy},
                        f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:  # noqa: BLE001
        record.update(mode="serial", reason=f"dump_failed: {e}")
        _cleanup(handle)
        print(f"[warn] multi-seed 并行数据落盘失败,串行: {e}", flush=True)
        return handle
    record["dump_sec"] = round(time.time() - t0, 1)
    record["data_gb"] = round(data_gb, 2)
    skill_root = str(Path(__file__).resolve().parents[1])
    _spawn = spawn or subprocess.Popen
    for k, gpu in zip(todo[1:], plan["gpus"]):
        out = sd / f"seed-{k}.parquet"
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "MODELEVO_GPU": "1",
                    "MODELEVO_SEED_VARIANT": "1",
                    "PYTHONPATH": skill_root + (
                        os.pathsep + env["PYTHONPATH"]
                        if env.get("PYTHONPATH") else "")})
        log = open(sd / f".seed-worker-{k}.log", "wb")
        cmd = [sys.executable, "-m", "runtime.seed_parallel", WORKER_FLAG,
               "--script", str(script_path), "--data", str(data_path),
               "--seed", str(int(seed0) + 1000 * int(k)), "--out", str(out)]
        try:
            p = _spawn(cmd, cwd=skill_root, env=env, stdout=log,
                       stderr=subprocess.STDOUT)
        except Exception as e:  # noqa: BLE001
            log.close()
            print(f"[warn] seed worker {k} 拉起失败(主进程串行补齐): {e}",
                  flush=True)
            continue
        handle["procs"].append((k, gpu, p, log))
        handle["assigned"].append(k)
    if not handle["assigned"]:
        record.update(mode="serial", reason="spawn_failed")
        _cleanup(handle)
        return handle
    record["n_parallel"] = 1 + len(handle["assigned"])
    print(f"[step] multi-seed 并行(v2.19.0): 主进程训 "
          f"{[k for k in todo if k not in handle['assigned']]},worker "
          f"{[(k, g) for k, g, _, _ in handle['procs']]}", flush=True)
    return handle


def _cleanup(handle: dict) -> None:
    dp = handle.get("data_path")
    if dp:
        try:
            Path(dp).unlink()
        except OSError:
            pass
        handle["data_path"] = None
    plan = handle.get("_plan")
    if plan:
        _release(plan)


def finish_seed_workers(handle: dict | None, *, kill: bool = False) -> dict:
    """等 worker 结束(kill=True 时直接终止),释放租约 / 准入预留 / 数据文件。
    返回留痕(写 metrics.multi_seed_parallel);失败变体由调用方串行补齐。"""
    if not handle:
        return {}
    rec = handle.get("record") or {}
    workers = rec.setdefault("workers", {})
    try:
        for k, gpu, p, log in handle.get("procs") or []:
            if kill and p.poll() is None:
                p.kill()
            rc = p.wait()
            out = Path(log.name).parent / f"seed-{k}.parquet"
            ok = bool(rc == 0 and out.exists())
            workers[str(k)] = {"gpu": gpu, "rc": rc, "ok": ok}
            log.close()
            if ok:
                try:
                    Path(log.name).unlink()
                except OSError:
                    pass
    finally:
        handle["procs"] = []
        _cleanup(handle)
    return {k: v for k, v in rec.items() if not str(k).startswith("_")}


def score_variant(model, X_va, model_type: str):
    """与 baseline 血缘串行段同口径的打分。"""
    if model_type == "binary_classification":
        return model.predict_proba(X_va)[:, 1]
    if model_type == "multiclass":
        return model.predict_proba(X_va).max(axis=1)
    return model.predict(X_va)


def _worker_main(argv: list) -> int:
    import argparse
    import importlib.util

    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument(WORKER_FLAG, action="store_true")
    ap.add_argument("--script", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location("_mevo_seed_trial",
                                                  a.script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # 只执行模块级定义;__name__ 非 __main__
    with open(a.data, "rb") as f:
        d = pickle.load(f)
    mod.SEED = int(a.seed)
    # v2.20.1:把主轮的配方 / 训练面 / 早停面装回模块全局。worker 只导入了
    # 模块级定义(__name__ != "__main__",main() 未执行),这些全局全是 None。
    # 判据:只有"嫁接版" _fit_seed_variant 才依赖这三个全局(它由
    # templates.TRIAL_BODY 定义,模块级声明 _PRIMARY_FIT_PARAMS/_PRIMARY_FIT_XY,
    # 缺省 None)。baseline 血缘自带的版本只依赖 SEED,模块里根本没有这两个名字,
    # 对它们要求 primary_* 会把正常的并行路径整条打成串行。
    _needs_primary = hasattr(mod, "_PRIMARY_FIT_XY")
    if _needs_primary:
        _pp = d.get("primary_params")
        _pxy = d.get("primary_xy")
        if _pp is None or _pxy is None:
            # 宁可退串行(调用方按 rc!=0 在主进程补齐),不可静默产出异配方
            # 分数:兜底分支会把 eval_set(outer validation)拼进训练集。
            print("[fatal] 嫁接版 seed 变体未收到主轮配方/训练面"
                  "(primary_params/primary_xy 为 None),拒绝训练;"
                  "该变体将由主进程串行补齐", flush=True)
            return 2
        mod._PRIMARY_FIT_PARAMS = dict(_pp)
        mod._PRIMARY_FIT_XY = (_pxy[0], _pxy[1])
        if d.get("eval_set"):
            mod._LAST_EVAL_XY = (d["eval_set"][0][0], d["eval_set"][0][1])
    os.environ["MODELEVO_SEED_VARIANT"] = "1"
    t0 = time.time()
    m = mod._fit_seed_variant(d["n_jobs"], d["X_fit"], d["y_fit"],
                              d["eval_set"])
    if hasattr(mod, "align_predict_device"):
        mod.align_predict_device(m)
    sc = score_variant(m, d["X_va"], d["model_type"])
    out = Path(a.out)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    pd.DataFrame({"score": sc}).to_parquet(tmp, index=False)
    os.replace(tmp, out)
    print(f"[step] seed worker done seed={a.seed} → {out.name} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
