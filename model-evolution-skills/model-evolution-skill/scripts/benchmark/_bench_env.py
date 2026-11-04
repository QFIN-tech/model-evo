"""对照基线的执行环境口径:线程上限 + GPU 选卡(供 raw-agent / AIDE 共用)。

线程:本机 128 核上同时跑 5 组 ModelEvo 驱动 + 4 个对照,全核并行反而更慢
(实测 AIDE 的 n_jobs=-1 会把机器压到 load 240+,单折都要数小时)。所有
对照的子进程统一夹到 ``<KIND>_THREADS``(缺省 24)。

GPU:ModelEvo 侧由 ``runtime.scheduler`` 租卡并注入 ``MODELEVO_GPU=1``;
对照脚本是独立进程,这里按 nvidia-smi 的**实际占用**挑卡,把
``CUDA_VISIBLE_DEVICES`` 收敛成单卡(子进程里就是 device 0),并把
``MODELEVO_GPU`` 置 1 —— 本仓库的 GBM 模板只认这个信号来决定
``device=cuda``。选不到卡时两者都显式清空/置 0,不会让子进程摸到无关的卡。
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

#: 显存占用超过这个值(未跑任何任务)的卡视为被别人的进程占用
_GPU_BUSY_MIB = 4000
#: 利用率超过这个百分比同样视为忙
_GPU_BUSY_UTIL = 20


def threads(kind: str, default: int = 24) -> int:
    """``RAW_AGENT_THREADS`` / ``AIDE_THREADS``,缺省 24。"""
    try:
        return max(1, int(os.environ.get(f"{kind.upper()}_THREADS", default)))
    except ValueError:
        return default


def thread_env(n: int, base: dict | None = None) -> dict:
    """OpenMP / BLAS / joblib 的线程口径。

    joblib 的 ``n_jobs=-1`` 不受 OMP 影响,必须单独用 ``LOKY_MAX_CPU_COUNT``
    夹住(否则 sklearn 侧仍会吃满 128 核)。
    """
    env = dict(base if base is not None else os.environ)
    s = str(int(n))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "LOKY_MAX_CPU_COUNT", "MKL_DYNAMIC"):
        env[k] = s
    return env


def pick_gpu(exclude: tuple[int, ...] = (), prefer: str | None = None) -> int | None:
    """按实际占用挑一张最空的卡;没有可用卡返回 None。

    ``exclude`` 用于同一进程内的多路并发避免撞卡,``prefer`` 是
    ``CUDA_VISIBLE_DEVICES`` 式的手工指定(逗号分隔,取第一个可用)。
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001  nvidia-smi 缺失/驱动异常 → 按无卡处理
        return None
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx, used, util = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if idx in exclude:
            continue
        rows.append((used, util, idx))
    if not rows:
        return None
    if prefer:
        want = {int(x) for x in str(prefer).replace(" ", "").split(",")
                if x.lstrip("-").isdigit()}
        hit = [r for r in rows if r[2] in want]
        rows = hit or rows
    # 只有**确实空闲**的卡才给:显存占用与利用率都在阈值内。
    # 旧实现在全部忙时仍返回最空的一张,实测 2026-11-03 raw-agent/house_sales
    # 被派到一张 80 GB、100% 利用率的卡上,CatBoost GPU 单次 fit 300 轮要
    # 85~107 s(同数据 CPU 1500 轮只要 3.7 s),在别的卡上甚至直接
    # `CUDA error 2: out of memory`。忙就回落 CPU,由调用方按 gpu=None 处理。
    for used, util, idx in rows:
        if used <= _GPU_BUSY_MIB and util <= _GPU_BUSY_UTIL:
            return idx
    return None


def device_env(gpu: int | None, n_threads: int,
               base: dict | None = None) -> dict:
    """子进程环境:线程上限 + 单卡可见 + ``MODELEVO_GPU`` 意图信号。"""
    env = thread_env(n_threads, base)
    if gpu is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["MODELEVO_GPU"] = "0"
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
        env["MODELEVO_GPU"] = "1"
    return env


def gpu_hint(gpu: int | None, n: int) -> str:
    """给 LLM 看的可用性说明(拼进 system prompt)。

    只写"有没有卡"和"线程上限",不写设备号 —— 子进程里 ``CUDA_VISIBLE_DEVICES``
    已收敛为单卡,device 0 恒可用,模型不该去猜物理编号。
    """
    if gpu is None:
        return (f"本机当前没有空闲的 GPU,全部用 CPU(不要写 task_type='GPU' / "
                f"device='cuda',也不要尝试回退到 GPU)。训练线程数上限 {n},"
                "不要把 n_jobs / nthread 设为 -1。")
    return ("本机有 GPU 可用(CUDA_VISIBLE_DEVICES 已设置好,直接用 "
            "`device='cuda'` / xgboost 的 `device='cuda'` / lightgbm 的 "
            "`device_type='gpu'` 即可,不要自己指定物理卡号)。"
            f"训练线程数上限 {n},不要把 n_jobs / nthread 设为 -1。")


#: 启动重负载前要求的可用内存;pod 内存是多租户共享的,只能降低 OOM 风险
MIN_FREE_GB = float(os.environ.get("BENCH_MIN_FREE_GB", "100"))
#: 被杀(OOM 等)时的重试次数;代码错误不重试
KILLED_RETRIES = 2


def free_memory_gb() -> float | None:
    """MemAvailable 与 cgroup 余量(上限 − rss,页缓存可回收)取小。"""
    vals = []
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                vals.append(int(line.split()[1]) / 1024 ** 2)
                break
        cg = Path("/sys/fs/cgroup/memory")
        limit = int((cg / "memory.limit_in_bytes").read_text())
        for line in (cg / "memory.stat").read_text().splitlines():
            if line.startswith("total_rss "):
                vals.append((limit - int(line.split()[1])) / 1024 ** 3)
                break
    except (OSError, ValueError):
        pass
    return min(vals) if vals else None


def wait_for_memory(min_gb: float | None = None, max_wait_sec: float = 3600,
                    poll_sec: float = 30) -> dict:
    """可用内存不足时排队;等满仍不足就放行并留痕(不让对照永久卡住)。"""
    need = MIN_FREE_GB if min_gb is None else float(min_gb)
    t0 = time.time()
    while True:
        free = free_memory_gb()
        waited = round(time.time() - t0, 1)
        if free is None or free >= need:
            return {"admitted": True, "free_gb": free, "waited_sec": waited}
        if waited >= max_wait_sec:
            return {"admitted": False, "free_gb": free, "waited_sec": waited}
        time.sleep(poll_sec)


def classify_exit(rc) -> str:
    """ok / killed(信号或 137,基础设施失败,可重试)/ timeout / code_error。"""
    if rc == 0:
        return "ok"
    if rc is None:
        return "timeout"
    return "killed" if rc < 0 or rc == 137 else "code_error"


def error_tail(stderr: str | None, limit: int = 1500) -> str:
    """去掉 warnings 噪声后的 stderr 尾部(否则真正的报错被挤出截断窗口)。"""
    lines = [ln for ln in (stderr or "").splitlines()
             if "Warning:" not in ln and "warnings.warn" not in ln]
    return "\n".join(lines)[-limit:]


def audit_training_source(code: str, patterns: tuple[str, ...]) -> dict:
    """静态查方案是否可能用了 validation 训练。命中只代表要人工复核,不是定论。"""
    hits = [ln.strip()[:160] for ln in (code or "").splitlines()
            if any(re.search(p, ln) for p in patterns)]
    return {"status": "review" if hits else "clean", "hits": hits[:20]}
