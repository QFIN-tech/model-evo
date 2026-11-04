"""资源调度与并发执行(§16)。

Planner 决定"做哪些实验",Scheduler 决定"同时跑多少、分配多少资源"(§16.4)。
失败隔离:单个 Trial 挂起/失败不阻塞 Generation,也不静默吞噬预算(§16.5)。
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil

from .config import Config
from .logging_utils import get_logger, now_iso
from .trial import TrialRecord

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 资源快照与并发决策
# ---------------------------------------------------------------------------

def resource_snapshot() -> dict:
    vm = psutil.virtual_memory()
    load1, _, _ = psutil.getloadavg()
    gpus: list[dict] = []
    try:
        # v19(req-t7-drawdown 核查回写):环境同时装了弃用的 pynvml
        # 与继任的 nvidia-ml-py(同 import 名 pynvml),弃用包装在前面
        # 时 FutureWarning 刷屏。优先 pynvml,失败回退 nvidia-smi
        # 文本解析(nvidia-ml-py 装而未用的机器/无 python 绑定的机器
        # 都能探到卡)。
        try:
            import warnings as _w

            with _w.catch_warnings():
                _w.simplefilter("ignore", FutureWarning)
                import pynvml

            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                gpus.append({
                    "index": i,
                    "memory_total_gb": mem.total / 2**30,
                    "memory_free_gb": mem.free / 2**30,
                    "util_pct": util.gpu,
                })
            pynvml.nvmlShutdown()
        except Exception as _nv:  # noqa: BLE001
            gpus = _gpu_snapshot_via_nvidia_smi()
            if gpus:
                logger.info("pynvml 探测失败(%s),nvidia-smi 回退探到 "
                            "%d 张卡", _nv, len(gpus))
    except Exception as exc:  # noqa: BLE001  无 GPU / 驱动异常
        # 降级本身是设计(CPU 机器无卡可探),但必须可见:有卡机器
        # 恰好没装绑定库时曾一路静默走 CPU 分支,唯一线索是 gpu=None
        logger.warning("GPU 探测失败,按无 GPU 调度(无卡 / 绑定库缺失 / "
                       "驱动异常): %s", exc)
    return {
        "at": now_iso(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False) or psutil.cpu_count(),
        "load1": load1,
        "ram_total_gb": vm.total / 2**30,
        "ram_available_gb": vm.available / 2**30,
        "gpus": gpus,
    }


def _gpu_snapshot_via_nvidia_smi() -> list[dict]:
    """v19:nvidia-smi 文本解析回退(pynvml 绑定不可用时)。

    输出格式:`0, NVIDIA A800, 81920, 0` (index, name, total MiB,
    used MiB);解析失败返回 [](调用方按无卡处理并已有 WARNING)。
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_gb": float(parts[2]) / 1024,
                "memory_free_gb": (float(parts[2]) - float(parts[3])) / 1024,
                "util_pct": 0,  # 文本回退不查利用率,保守按空闲
            })
        return gpus
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# v2.13.3:GPU 跨进程排他租约
#
# 修什么:并发 trial 撞同一张卡。scheduler.plan() 是纯函数——每个
# runner 进程各自快照、各自取"空闲显存最大"的卡,进程之间没有任何
# 互斥,于是 N 个并发 trial 同时启动时**全部选中同一张** GPU(实测
# gen-04 三个 HPO trial 同落 GPU0,78407MiB/81920MiB 占满)。既有的
# gpu_indices[slot % len] 只在"批内序号"这一层错开,挡不住跨进程竞态;
# 手动并行 `run --trial` 路径更是恒 slot_index=0。
#
# 怎么修:每张卡一个租约文件(<run_dir>/.gpu-leases/gpu-<i>.lock),写
# 持有者 "pid:token"。取卡前逐个抢,抢到的才写进 CUDA_VISIBLE_DEVICES。
# 发布用"写好内容 + fsync + 硬链接"而不是 O_EXCL 建空文件——后者在
# "已创建、内容未写"的窗口里会被别的进程误判成陈旧租约而回收(实测
# 3 进程抢卡时 2 个落到同一张)。租约语义：
#   - 进程活着 = 持有(不判 util/显存,与"谁在用"解耦,避免误判抢占);
#   - 进程已死 = 陈旧租约,原子摘除(rename)并复核后回收;
#   - 无法查活(EPERM 跨用户)= 保守视为活着,不抢(宁少用不错用);
#   - 同进程内并发线程**不可**依赖 pid 区分(pid 相同会误判成"自己持有"
#     而互相覆盖),故租约值含每线程 token,并以模块锁串行化抢卡。
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """pid 是否存活(EPERM 视为存活:卡被别的用户占着,不该抢)。"""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


#: 同进程内的租约互斥(v2.13.3)。GenerationExecutor 用一个线程池并发跑
#: 批内 trial,同 pid 的多线程共享进程身份——只靠 pid 判活时两个线程
#: 都会"看到自己的租约"从而写进同一张卡。故持有者身份取
#: **每线程** token(见 _proc_token),并以本锁串行化"查租约+抢租约"
#: 临界区。锁不覆盖训练——跨进程互斥由文件租约承担。
_LEASE_MTX = threading.Lock()

_TLS = threading.local()


def _proc_token() -> str:
    """本线程的租约身份 token(线程内稳定,便于同线程幂等重入)。

    含 pid 与 thread ident 供人读,随机后缀保证同一线程复用 ident 时
    不会与上一轮的残留租约混淆。
    """
    tok = getattr(_TLS, "token", None)
    if tok is None:
        tok = (f"{os.getpid()}-{threading.get_ident()}-"
               f"{os.urandom(4).hex()}")
        _TLS.token = tok
    return tok


def _try_lease(lock: Path) -> bool:
    """原子抢一张卡的租约。True=本线程成为持有者。

    已有租约时:持有者活着(pid 与 token 都匹配)→ 不可抢;进程已死
    → 陈旧,原子重抢回收;租约损坏(空/非法)→ 视同陈旧,同一原子路径
    回收(先 O_EXCL 尝试,失败再判活,避免无谓 unlink 造成 TOCTOU 窗口)。
    同进程并发线程由 _LEASE_MTX 串行化(见其定义处说明)。
    """
    lock = Path(lock)
    with _LEASE_MTX:
        return _try_lease_locked(lock)


def _try_lease_locked(lock: Path) -> bool:
    """抢租约(调用方持 _LEASE_MTX)。

    发布用 **硬链接** 而非 O_CREAT|O_EXCL:后者"先建空文件、后写内容"
    之间有一个真实窗口——抢输的进程此刻读到空租约,会把"内容还没写
    完"误判成"持有者已死"而回收,两个进程同时拿到同一张卡(实测 3
    进程抢卡时 2 个拿到 gpu1)。硬链接的语义是"内容写好并 fsync 之后
    才对外可见",从根本上消掉这个窗口;O_EXCL→link 的原子性在
    overlayfs 与 ceph 上均已实测(8 进程同抢恰好 1 个成功)。
    """
    token = _proc_token()
    lease_dir = lock.parent
    tmp = lease_dir / f".lease-tmp-{os.getpid()}-{threading.get_ident()}"
    try:
        with open(tmp, "wb") as fh:
            fh.write(f"{os.getpid()}:{token}\n".encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        return False

    try:
        for _ in range(4):  # 陈旧回收后可能被别人抢先,重试有限次
            try:
                os.link(str(tmp), str(lock))
                return True
            except FileExistsError:
                pass
            except OSError:
                return False
            try:
                observed = lock.read_text(encoding="utf-8").strip()
            except OSError:
                return False   # 持有者正在释放 → 让调用方换下一张卡
            pid_s, _, tok = observed.partition(":")
            try:
                pid = int(pid_s)
            except ValueError:
                pid = 0        # 记录损坏:与"持有者已死"同路径回收
            # 持有者活着就绝不抢——**包括本进程自己的另一个线程**。刻意
            # 不做"同线程幂等"捷径:每次 _execute_once 都在 finally 里
            # 释放,不存在嵌套抢卡,而返回成功会静默掩盖重复占用。
            if _pid_alive(pid):
                return False
            # 陈旧租约:rename 原子摘除 → 复核内容未被换过 → 丢弃重抢
            stale = lease_dir / f".lease-stale-{os.getpid()}-{token}"
            try:
                os.rename(str(lock), str(stale))
            except OSError:
                continue       # 已被别的回收者摘走 → 重试图 link
            try:
                current = stale.read_text(encoding="utf-8").strip()
            except OSError:
                current = None
            if current != observed:
                # 摘下的是别人新持有的租约 → 放回(link 目标被占则说明
                # 又有新持有者,更该放回去让它自然存在)
                try:
                    os.link(str(stale), str(lock))
                except OSError:
                    pass
                try:
                    os.unlink(str(stale))
                except OSError:
                    pass
                return False
            try:
                os.unlink(str(stale))
            except OSError:
                pass
            # 继续循环:重新尝试发布
        return False
    finally:
        try:
            os.unlink(str(tmp))
        except OSError:
            pass


def _lease_held_by_us(lock: Path) -> bool:
    """租约是否确由本线程持有——释放前必须确认,防止误删别的进程
    刚回收走的租约。"""
    try:
        return Path(lock).read_text(
            encoding="utf-8").strip() == f"{os.getpid()}:{_proc_token()}"
    except OSError:
        return False


@dataclass
class GpuLease:
    """一张 GPU 的跨进程排他租约(见模块级 GPU 租约说明)。

    租约文件在 <run_dir> 下,故只在同一 run 内互斥;锁的只是"谁用哪张
    卡",不分配显存(卡内多进程共享仍由 CUDA 自己处理)。
    """

    lock_path: Path
    gpu_index: int
    leases_dir: Path

    @staticmethod
    def acquire(leases_dir: Path, gpu_index: int) -> GpuLease | None:
        """抢占 gpu_index 的租约;已被活进程持有则返回 None。"""
        leases_dir = Path(leases_dir)
        lock = leases_dir / f"gpu-{int(gpu_index)}.lock"
        try:
            leases_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # noqa: BLE001
            logger.warning("GPU 租约目录不可建(%s),跳过租约(%s)",
                           leases_dir, exc)
            return None
        if not _try_lease(lock):
            return None
        return GpuLease(lock_path=lock, gpu_index=int(gpu_index),
                        leases_dir=leases_dir)

    def release(self) -> None:
        """释放租约(仅当仍由本进程持有;否则说明已被回收,不删)。"""
        if _lease_held_by_us(self.lock_path):
            try:
                self.lock_path.unlink()
            except OSError:
                pass

    def __enter__(self) -> GpuLease:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def gpu_candidates(snapshot: dict) -> list[dict]:
    """按抢租约优先级排序的 GPU 列表(不改动既有选卡语义)。

    排序键与 plan() 的两级偏好一致:先"空闲显存 >20GB 且 util<80%"的
    健康卡(同级内空闲显存降序),再退化的卡(空闲显存降序),最后才是
    无信息的卡。runner 按此序逐个试抢,抢到的即本 trial 的卡。
    """
    gpus = list((snapshot or {}).get("gpus") or [])

    def _rank(g: dict) -> tuple:
        healthy = (g.get("memory_free_gb", 0) > 20
                   and g.get("util_pct", 100) < 80)
        return (
            0 if healthy else 1,                 # 健康卡优先
            -float(g.get("memory_free_gb", 0)),  # 同级内空闲显存降序
            int(g.get("index", 0)),
        )

    return sorted(gpus, key=_rank)


# ---------------------------------------------------------------------------
# v2.17.0(WP5a):GPU 按实际占用选卡
#
# 修什么:租约只在本 run 内互斥,scheduler 看不见同机其它 run 的进程。
# 实测(run4 regression.2 gen-04):trial-016 被派到与其它 run 2 个进程
# 共享的 GPU 1(nvidia-smi 100%),分类子模型 5107 轮跑了 42 min;同参数
# 在空闲卡上约 10 min。
#
# 怎么修:nvidia-smi 查每卡显存/利用率与计算进程,把进程归属到"本 run /
# 外来"。优先选无外来进程、显存充足的卡;全部被占 → 退回旧排序并留痕;
# 查询失败 → 原样旧逻辑。结果缓存 GPU_OCCUPANCY_TTL_SEC 秒。
# 容器内 nvidia-smi 报的是宿主 pid(本命名空间不可见):可见 pid 按 cwd
# 归属;不可见 pid 只有在本 run 持有该卡活租约时按 1 个记为本 run。
# ---------------------------------------------------------------------------

GPU_OCCUPANCY_TTL_SEC = 5.0
#: 选卡时"显存充足"的下限(与 gpu_candidates 的健康卡口径一致)
GPU_MIN_FREE_GB = 20.0
_GPU_OCC_MTX = threading.Lock()
_GPU_OCC_CACHE: dict[str, Any] = {"at": 0.0, "data": None}


def _nvidia_smi(args: list[str]) -> str | None:
    """运行 nvidia-smi;失败返回 None(测试替换本函数离线 mock)。"""
    try:
        out = subprocess.run(["nvidia-smi", *args], capture_output=True,
                             text=True, timeout=10)
    except Exception:  # noqa: BLE001  无驱动 / 无命令 / 超时
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _csv_rows(text: str | None) -> list[list[str]]:
    rows = []
    for line in (text or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and any(parts):
            rows.append(parts)
    return rows


def _num(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def query_gpu_occupancy(*, max_age_sec: float | None = None) -> list | None:
    """每卡显存/利用率与计算进程列表(nvidia-smi;缓存几秒)。

    返回 ``[{index, uuid, memory_total_gb, memory_used_gb, memory_free_gb,
    util_pct, procs: [{pid, used_gb}], procs_known}]``;卡查询失败返回
    None(调用方回退旧逻辑)。进程查询单独失败时 ``procs_known=False``。
    """
    ttl = GPU_OCCUPANCY_TTL_SEC if max_age_sec is None else max_age_sec
    with _GPU_OCC_MTX:
        now = time.time()
        if _GPU_OCC_CACHE["data"] is not None \
                and now - _GPU_OCC_CACHE["at"] < ttl:
            return [dict(g, procs=list(g["procs"]))
                    for g in _GPU_OCC_CACHE["data"]]
        cards_txt = _nvidia_smi([
            "--query-gpu=index,uuid,memory.total,memory.used,"
            "utilization.gpu", "--format=csv,noheader,nounits"])
        if cards_txt is None:
            return None
        cards: list[dict] = []
        by_uuid: dict[str, dict] = {}
        for parts in _csv_rows(cards_txt):
            if len(parts) < 5:
                continue
            idx, total, used = _num(parts[0]), _num(parts[2]), _num(parts[3])
            if idx is None or total is None or used is None:
                continue
            util = _num(parts[4])
            g = {"index": int(idx), "uuid": parts[1],
                 "memory_total_gb": round(total / 1024, 2),
                 "memory_used_gb": round(used / 1024, 2),
                 "memory_free_gb": round((total - used) / 1024, 2),
                 # [N/A] 利用率按满载保守处理(与旧口径 util<80 的健康判定一致)
                 "util_pct": int(util) if util is not None else 100,
                 "procs": [], "procs_known": False}
            cards.append(g)
            by_uuid[parts[1]] = g
        if not cards:
            return None
        apps_txt = _nvidia_smi([
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits"])
        if apps_txt is not None:
            for g in cards:
                g["procs_known"] = True
            for parts in _csv_rows(apps_txt):
                if len(parts) < 3 or parts[0] not in by_uuid:
                    continue
                pid = _num(parts[1])
                if pid is None:
                    continue
                mem = _num(parts[2])
                by_uuid[parts[0]]["procs"].append(
                    {"pid": int(pid),
                     "used_gb": round(mem / 1024, 2) if mem else None})
        _GPU_OCC_CACHE.update(at=now, data=cards)
        return [dict(g, procs=list(g["procs"])) for g in cards]


def _pid_belongs_to_run(pid: int, run_dir: Path) -> bool | None:
    """pid 是否属于本 run:True/False;本命名空间不可见(容器)返回 None。"""
    proc = Path(f"/proc/{int(pid)}")
    if not proc.exists():
        return None
    try:
        cwd = Path(os.readlink(str(proc / "cwd")))
    except OSError:
        return False  # 可见但不可读 = 别的用户的进程,不是本 run
    try:
        root = Path(run_dir).resolve()
        return cwd == root or root in cwd.parents
    except OSError:
        return False


def _run_leased_gpus(run_dir: Path) -> set[int]:
    """本 run 持有活租约的卡号(持有者进程存活)。"""
    out: set[int] = set()
    try:
        for f in (Path(run_dir) / ".gpu-leases").glob("gpu-*.lock"):
            try:
                pid = int(f.read_text(encoding="utf-8").split(":", 1)[0])
                if _pid_alive(pid):
                    out.add(int(f.stem.split("-", 1)[1]))
            except (OSError, ValueError, IndexError):
                continue
    except OSError:
        pass
    return out


def annotate_gpu_occupancy(cards: list | None, run_dir: Path | None) -> list:
    """给占用快照标注 own_procs / foreign_procs / leased_by_run / clean。

    foreign_procs=None 表示进程归属未知(进程查询失败);此时 clean 按
    "显存几乎为空且利用率低"判断(看起来空闲才算干净)。
    """
    leased = _run_leased_gpus(run_dir) if run_dir is not None else set()
    out = []
    for g in cards or []:
        g = dict(g)
        own = foreign = 0
        unattributed_budget = 1 if g.get("index") in leased else 0
        for p in g.get("procs") or []:
            owner = (_pid_belongs_to_run(p["pid"], run_dir)
                     if run_dir is not None else False)
            if owner is True:
                own += 1
            elif owner is None and unattributed_budget > 0:
                own += 1
                unattributed_budget -= 1
            else:
                foreign += 1
        known = bool(g.get("procs_known"))
        g["own_procs"] = own if known else None
        g["foreign_procs"] = foreign if known else None
        g["leased_by_run"] = g.get("index") in leased
        free = float(g.get("memory_free_gb") or 0.0)
        util = float(g.get("util_pct") if g.get("util_pct") is not None
                     else 100)
        if known:
            g["clean"] = (foreign == 0 and free >= GPU_MIN_FREE_GB
                          and (util < 80 or g["leased_by_run"]))
        else:
            g["clean"] = (float(g.get("memory_used_gb") or 0.0) < 2.0
                          and util < 10 and free >= GPU_MIN_FREE_GB)
        out.append(g)
    return out


def gpu_candidates_by_occupancy(cards: list) -> tuple[list[int], list[int]]:
    """(干净卡序, 被外来进程占用的卡序)。

    干净卡:空闲显存降序、利用率升序;被占卡:外来进程数升序、利用率升序、
    空闲显存降序。runner 先在干净卡里抢租约,抢不到再退回旧排序。
    """
    clean = sorted((g for g in cards if g.get("clean")),
                   key=lambda g: (-float(g.get("memory_free_gb") or 0.0),
                                  float(g.get("util_pct") or 0),
                                  int(g.get("index", 0))))
    busy = sorted((g for g in cards if not g.get("clean")),
                  key=lambda g: (g.get("foreign_procs")
                                 if g.get("foreign_procs") is not None
                                 else 99,
                                 float(g.get("util_pct") or 0),
                                 -float(g.get("memory_free_gb") or 0.0),
                                 int(g.get("index", 0))))
    return ([int(g["index"]) for g in clean],
            [int(g["index"]) for g in busy])


@dataclass
class ResourcePlan:
    """一次 Trial 执行的资源分配。"""

    n_parallel_trials: int = 1
    threads_per_trial: int = 4
    gpu_index: int | None = None
    # v18/P1-3:多卡分配——并发 N>1 时按空闲显存降序逐 trial 指派;
    # None = 单路(全兼容旧消费方)。runner 按 trial 序号取
    # gpu_indices[i % len] 注入 CUDA_VISIBLE_DEVICES。
    gpu_indices: list[int] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_parallel_trials": self.n_parallel_trials,
            "threads_per_trial": self.threads_per_trial,
            "gpu_index": self.gpu_index,
            "gpu_indices": list(self.gpu_indices),
        }


class Scheduler:
    """根据机器空闲状态计算并发与每 Trial 资源(§16.3)。"""

    def __init__(self, config: Config, snapshot: dict | None = None):
        self.config = config
        # v2.17.0:live=True(未注入快照)时 runner 选卡会实时查 nvidia-smi
        # 占用;注入快照(测试/离线回放)以快照为准,不碰真实设备
        self.live = snapshot is None
        self._snap = snapshot or resource_snapshot()

    def snapshot(self) -> dict:
        return self._snap

    def _usable_gpu_count(self, snap: dict) -> int:
        """v2.19.0:可承接一个 trial 的卡数(空闲显存 ≥ min_free_gpu_gb 且
        util < 80%,与 gpu_candidates 的健康卡口径一致)。"""
        try:
            _min_free = float(self.config.get(
                "execution.min_free_gpu_gb", GPU_MIN_FREE_GB) or 0.0)
        except (TypeError, ValueError):
            _min_free = GPU_MIN_FREE_GB
        n = 0
        for g in snap.get("gpus") or []:
            try:
                if float(g.get("memory_free_gb", 0)) >= _min_free \
                        and float(g.get("util_pct", 100)) < 80:
                    n += 1
            except (TypeError, ValueError):
                continue
        return n

    def plan(self, *, running_trials: int = 0,
             planned_trials: int | None = None) -> ResourcePlan:
        """计算并发与线程分配。

        planned_trials:本批实际要跑的 Trial 数(run-generation/baseline
        成批提交时传入)。auto 模式按它收窄并发槽位并把线程反比放大
        (并发×线程≈usable 的不变式在槽位不满时仍成立):
        4 个 trial 各拿 usable//4 线程,而不是按 8 路 pre-split 只分到
        usable//8。不传时表示"未声明并发批次"——v14.3 起默认按
        **1 路独占**分配(单 trial 逐个 `run --trial` 的场景吃满可用核,
        而非旧 8 路保守默认把 ~7/8 CPU 闲置);线程数上限取自
        execution.parallel_trials_auto_cap(默认 16)。手动并行多个
        `run --trial` 时由 runner 按 RUNNING trial 数传入
        planned_trials 收敛(见 runner._count_running_trials)。
        """
        snap = self._snap
        cpu_phys = max(snap["cpu_physical"] or 4, 4)
        cpu_logical = max(snap.get("cpu_count") or cpu_phys, cpu_phys)
        # 资源治理(按逻辑核计):目标利用率 = 逻辑核×90%,预留 2 核
        # 给协调进程;load1 反映整机当前占用,从预算中直接扣除
        busy = snap.get("load1", 0.0) or 0.0
        usable = max(int(cpu_logical * 0.9) - 2 - int(busy), 2)
        ram_budget_gb = max(snap["ram_available_gb"] * 0.7, 4)

        cfg_parallel = self.config.get("evolution.parallel_trials", "auto")
        if cfg_parallel == "auto":
            # 目标:并发×线程 ≈ 可用核(吃满预算而非固定小份额)。
            # planned_trials 传入 → 按本批 trial 数收窄槽位、线程反比
            # 放大;不传 → 视为"单 trial 独占"(v14.3,不再 8 路保守)。
            # 线程上限 = execution.parallel_trials_auto_cap(默认 16),
            # 防单 trial 独占时 OMP 线程数压过可用核/内存。
            # v19(req-t7-drawdown 实测回写):自我锁死修复。旧公式
            # slots=min(8, planned_trials) 把"本批提交数"当"并发容量"
            # ——单 trial 提交路径恒传 planned_trials=1 → 槽位钉死 1 →
            # planner 读到 parallel=1 后逐个串行提交,106 可用核只吃
            # 24。修正:槽位按**容量口径**算上限(usable // threads_cap
            # 的最小可用线程档),planned_trials 只做"显式声明的批大小"
            # 收窄,不再放大(>容量时)或锁死(<容量但声明=1 时)。
            # 线程随槽位反比:slots 越多线程越少,至下限 4。
            auto_cap = int(self.config.get(
                "execution.parallel_trials_auto_cap", 16))
            # v2.11:线程下限护栏。旧公式在整机高负载(load1 高 → usable
            # 被压到个位数)时会把 threads 压到 4 甚至 2——实测
            # (run drawdown-req-t7-6w)24.6 万行的 smoke 在 threads=4 下
            # 跑了 1401s,trial 直接进入必然超时区,attempt 白烧 1.5h。
            # threads < 8 的 GBM 训练已无实用价值:与其提交一个必然
            # 超时/降配的 trial,不如把线程抬回下限并显式告警(宁慢,
            # 不交一个注定失败的配置;下限可经 execution.threads_floor
            # 配置,默认 8)。
            _threads_floor = int(self.config.get(
                "execution.threads_floor", 8))
            # v2.19.0(#13):GPU trial 按卡分线程。旧口径线程上限
            # parallel_trials_auto_cap=24 → 并发被"可用核 / 24"压到 4~5 路,
            # 8 卡机器实测平均并发 1.0~2.5、GPU 利用 12~30%。GBDT 上卡后
            # CPU 只做数据准备与早停打分,线程数不再是瓶颈:有可用卡时
            # 槽位 = min(可用卡数, 本批 trial 数),线程 = 可用核 / 槽位,
            # 钳到 [gpu_trial_threads_min, gpu_trial_threads_max]。
            _n_gpu_ok = self._usable_gpu_count(snap)
            if (planned_trials is not None and _n_gpu_ok > 0
                    and bool(self.config.get(
                        "execution.gpu_aware_parallel", True))):
                _g_min = max(2, int(self.config.get(
                    "execution.gpu_trial_threads_min", 8) or 8))
                _g_max = max(_g_min, int(self.config.get(
                    "execution.gpu_trial_threads_max", 24) or 24))
                slots = max(1, min(_n_gpu_ok, int(planned_trials)))
                threads = min(_g_max, max(_g_min, usable // slots))
                # 可用核连下限线程都喂不起 slots 路时收窄槽位(不超卖)
                n_par = max(1, min(slots, max(1, usable // _g_min)))
                logger.info(
                    "scheduler(v2.19.0 GPU 口径):可用卡 %d、本批 %d → "
                    "%d 路 × %d 线程(线程钳制 [%d, %d])",
                    _n_gpu_ok, int(planned_trials), n_par, threads,
                    _g_min, _g_max)
            elif planned_trials is not None:
                # 容量 = 把线程压到下限 4 时最多几路(capped 8);
                # 同时保证 planned=1 的单 trial 仍能拿到 threads_cap
                # 独占(v14.3 语义不变)
                capacity = max(1, min(8, usable // 4))
                slots = max(1, min(capacity, max(1, int(planned_trials)))
                            if int(planned_trials) > 1 else
                            max(1, min(capacity, 1)))
                threads = min(auto_cap, max(4, usable // slots))
                if threads < _threads_floor:
                    threads = min(auto_cap, _threads_floor)
                    logger.warning(
                        "scheduler:可用核不足(usable=%d, slots=%d)把 "
                        "threads 压到下限以下,已抬回 %d——高负载期提交的 "
                        "trial 训练时长会显著放大,建议等负载回落或减并发",
                        usable, slots, threads)
                n_par = max(1, min(slots, usable // threads))
            else:
                # 单 trial 独占:全部预算给一个 trial(上限 auto_cap)
                threads = min(auto_cap, max(4, usable))
                if threads < _threads_floor:
                    threads = min(auto_cap, _threads_floor)
                n_par = 1
        else:
            n_par = max(1, int(cfg_parallel))
            threads = max(2, usable // n_par)
            _threads_floor = int(self.config.get(
                "execution.threads_floor", 8))
            if threads < _threads_floor:
                threads = _threads_floor
        n_par = max(1, n_par - running_trials)

        # GPU 优先(v14.2):有卡即用——首选空闲显存最大且 >20GB、
        # util<80% 的卡(不与重负载进程撞卡);无满足条件的卡但机器
        # 有 GPU 时,退而选空闲显存最大的卡(小卡/半占用卡也远快于
        # CPU 训练,OOM 由 runner 降配重试兜底)。旧实现 >20GB 硬门槛
        # 把 8~16GB 卡全部判给 CPU,实测有卡不用。
        gpu_index = None
        gpus = snap.get("gpus") or []
        free = [g for g in gpus
                if g["memory_free_gb"] > 20 and g["util_pct"] < 80]
        if free:
            best = max(free, key=lambda g: g["memory_free_gb"])
            gpu_index = best["index"]
        elif gpus:
            best = max(gpus, key=lambda g: g["memory_free_gb"])
            gpu_index = best["index"]
            logger.info("无 >20GB 空闲卡,降级用空闲显存最大的 GPU=%s"
                        "(free=%.1fGB, util=%d%%)——有卡优先于 CPU",
                        best["index"], best["memory_free_gb"],
                        best["util_pct"])

        logger.info(
            "Scheduler plan: cpu_phys=%d cpu_logical=%d load1=%.1f → "
            "usable=%d → threads/trial=%d × parallel=%d (≈%d 核), "
            "gpu_index=%s(gpus 可用 %d 张)",
            cpu_phys, cpu_logical, busy, usable, threads, n_par,
            threads * n_par,
            gpu_index if gpu_index is not None else "无", len(gpus))

        env = {
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "NUMEXPR_NUM_THREADS": str(threads),
            # v17.4:GPU 意图信号(权威,不依赖 CUDA_VISIBLE_DEVICES 空串
            # 语义;trial 模板据此决定是否给模型加 device=cuda。降配重试
            # 时 runner 会把本值置 "0",防 pick_gpu() 又把 CPU 重试顶回卡)
            "MODELEVO_GPU": "1" if gpu_index is not None else "0",
        }
        if gpu_index is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        # v18/P1-3:多卡分配。并发 N>1 且机器有多张卡时,按空闲显存
        # 降序逐 trial 指派(第 i 个 trial → gpu_indices[i % len])。
        # 旧行为整批 trial 共享同一 CUDA_VISIBLE_DEVICES,4×A800 上
        # 并发 trial 全挤 GPU0(v17.4 GBDT 上卡后瓶颈显性化)。
        gpu_indices: list[int] = []
        if gpu_index is not None and n_par > 1:
            ranked = sorted(gpus, key=lambda g: -g["memory_free_gb"])
            gpu_indices = [g["index"] for g in ranked[:n_par]]
            if len(gpu_indices) > 1:
                logger.info("多卡分配: %d 路并发 → %d 张卡 %s",
                            n_par, len(gpu_indices), gpu_indices)
        return ResourcePlan(
            n_parallel_trials=n_par, threads_per_trial=threads,
            gpu_index=gpu_index, gpu_indices=gpu_indices, env=env)


# ---------------------------------------------------------------------------
# 进程组执行:per-trial timeout 与 kill(§16.5.1)
# ---------------------------------------------------------------------------

def _kill_process_group(pid: int, grace_sec: int = 30) -> None:
    """先 SIGTERM 给宽限期,后 SIGKILL;作用于整个进程组(含子进程)。"""
    try:
        pg = psutil.Process(pid)
        children = pg.children(recursive=True)
        targets = [pg] + children
    except psutil.NoSuchProcess:
        return
    for p in targets:
        try:
            p.send_signal(signal.SIGTERM)
        except psutil.NoSuchProcess:
            pass
    deadline = time.time() + grace_sec
    while time.time() < deadline:
        if not any(p.is_running() for p in targets):
            return
        time.sleep(1.0)
    for p in targets:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


#: v2.17.2(B1):信号退出码(Popen 负值形态与 shell 128+N 形态的 SIGKILL/SIGTERM)
SIGNAL_EXIT_CODES = frozenset({-9, -15, 137, 143})
#: 特征块进程池 worker 丢失(BrokenProcessPool / runtime.feature_cache 的
#: FeatureBlockWorkerLost 标记)——多为 OOM-kill → 判 oom(infra 类)
FEATURE_POOL_LOST_MARKS = ("brokenprocesspool", "feature_block_worker_lost",
                           "featureblockworkerlost")
#: 特征块 drain 超时(worker 卡死/丢失)→ infra_error
FEATURE_POOL_HANG_MARKS = ("feature_block_drain_timeout",
                           "featureblockdraintimeout")


def classify_failure(returncode: int | None, stderr_tail: str,
                     *, timed_out: bool = False, oom: bool = False) -> str:
    """失败分类(§16.5.3)。

    v2.17.2(B1):``oom=True`` 由调用方按 OOM 证据给出(run_trial_process:
    信号退出且本 cgroup 的 oom_kill 计数在训练期间增加);stderr 里有特征块
    进程池 worker 丢失标记(BrokenProcessPool / FeatureBlockWorkerLost)时
    同样判 oom——旧实现把 run4 trial-006 的 rc=-15 判成 code_error,白耗一次
    fix_code。oom 属 infra 类(INFRA_FAILURES),重入不计 fix_code。
    """
    if timed_out:
        return "timeout"
    if oom:
        return "oom"
    text = (stderr_tail or "").lower()
    if "memory" in text and ("killed" in text or "cannot allocate" in text
                             or "out of memory" in text):
        return "oom"
    if any(m in text for m in FEATURE_POOL_LOST_MARKS):
        return "oom"
    if any(m in text for m in FEATURE_POOL_HANG_MARKS):
        return "infra_error"
    # v19.3 修复(失败分类错判 → 无效重试烧全量训练):
    # ModuleNotFoundError/ImportError 指**项目内模块**(runtime / 本 run
    # 的 trial 包 / 相对路径模块)时是**代码错误**,不是环境缺失 ——
    # 重试必然复现。实测 gen-03 trial-008:hpo 嫁接段缺 sys.path 注入,
    # 首次 run 抛 `ModuleNotFoundError: No module named 'runtime'`,
    # 被归成 infra_error 进 RETRYABLE_FAILURES,同一份代码被重试一遍,
    # 白烧一轮全量训练(正确处置是修代码 + fix_code 轮)。
    # 判据:缺失模块名不带点(顶层第三方包名)且不是 runtime/项目包 →
    # 才算 infra(装包/环境问题可重试)。
    if "modulenotfounderror" in text or "importerror" in text:
        _proj_mods = ("runtime", "scripts", "tools", "views")
        _third_party = any(
            m in text for m in ("no module named 'numpy'",
                                "no module named 'pandas'",
                                "no module named 'sklearn'",
                                "no module named 'lightgbm'",
                                "no module named 'xgboost'",
                                "no module named 'catboost'",
                                "no module named 'optuna'",
                                "no module named 'torch'",
                                "no module named 'pyarrow'",
                                "no module named 'scipy'",
                                "no module named 'joblib'"))
        _is_proj = any(f"no module named '{m}'" in text
                       or f'no module named "{m}"' in text
                       for m in _proj_mods)
        if _is_proj and not _third_party:
            return "code_error"
    infra_marks = ("modulenotfounderror", "importerror", "filenotfounderror",
                   "connectionerror", "permissionerror", "oserror")
    if any(m in text for m in infra_marks):
        return "infra_error"
    return "code_error"


RETRYABLE_FAILURES = ("timeout", "oom", "infra_error")

#: v2.17.0:内存准入超时(训练从未启动)。属于 infra 类,但与 timeout/oom
#: 不同,它**没有触及 trial 代码**:不进失败熔断、不消耗 retry/fix 预算、
#: runner 内不自动重试(已等满 admission_max_wait_min),重入记 requeue。
ADMISSION_TIMEOUT = "admission_timeout"
#: 未触及 trial 代码的失败(资源未准入)
NON_TRIAL_FAILURES = frozenset({ADMISSION_TIMEOUT})
#: infra 类失败:环境/资源问题,不是方向本身的证据(不给方向记负奖励)。
#: 消费方(经验回填 / 方向台账 / 失败方向汇总)应以 is_infra_failure 判定
#: v2.17.2(B1):oom 归入 infra 类(资源不足,不是 trial 代码的错;重入记
#: requeue 不消耗 fix_code 次数)
INFRA_FAILURES = frozenset({"infra_error", "timeout", "oom",
                            ADMISSION_TIMEOUT})


def is_infra_failure(reason: str | None) -> bool:
    """failure_reason 是否属于 infra 类(不应给方向记负奖励)。"""
    return str(reason or "") in INFRA_FAILURES


@dataclass
class ExecutionResult:
    ok: bool
    returncode: int | None
    runtime_sec: float
    peak_memory_gb: float
    failure_reason: str | None
    stdout_path: Path
    stderr_path: Path
    stderr_tail: str = ""  # 失败诊断:stderr 尾部摘要(成功时为空)


def run_trial_process(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    timeout_sec: float,
    grace_sec: int = 30,
    env_extra: dict[str, str] | None = None,
    heartbeat_log: Path | None = None,
    heartbeat_interval_sec: float = 60.0,
    on_spawn=None,
) -> ExecutionResult:
    """在独立进程组运行 Trial 命令,强制 deadline 与资源测量。

    heartbeat_log 给定时,每隔 heartbeat_interval_sec 把 runtime/
    CPU%/内存快照追加到该日志(默认 1 分钟一行),供训练过程追踪;
    进程结束后补一行终态摘要。

    on_spawn(pid)(v2.17.0,可选):子进程拉起后回调一次——内存准入据此
    把训练进程 pid 记进预留,按"预估峰值 − 当前实占"计算未兑现的预留。"""
    cwd = Path(cwd)
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(env_extra or {})

    def _hb(msg: str) -> None:
        if heartbeat_log is None:
            return
        try:
            with heartbeat_log.open("a", encoding="utf-8") as f:
                f.write(f"[{now_iso()}] {msg}\n")
        except OSError:
            pass

    def _proc_cpu_pct(ps_proc) -> float:
        """进程组 CPU%(相对单核 100);采集失败返回 -1。"""
        try:
            procs = [ps_proc] + ps_proc.children(recursive=True)
            total = 0.0
            for p in procs:
                try:
                    total += p.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total
        except psutil.NoSuchProcess:
            return -1.0

    def _pss_bytes(pid: int) -> int | None:
        """读 /proc/<pid>/smaps_rollup 的 Pss(kB);读不到返 None。"""
        try:
            with open(f"/proc/{pid}/smaps_rollup") as f:
                for line in f:
                    if line.startswith("Pss:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return None

    def _proc_tree_mem(procs) -> tuple[float, bool]:
        """进程树内存(GB, 是否 Pss 口径)。

        skill-fix-20260910:旧实现对树内各进程 RSS 求和,多进程
        共享页(fork 的加载器/共享 mmap parquet)被重复累计
        (实测 960GB 物理机心跳报 3648.3GB)。改为逐进程优先
        smaps_rollup 的 Pss(比例分摊,共享页只计一次);
        rollup 不可用时回退 RSS 并在返回值标注口径。"""
        total = 0
        pss_ok = True
        for p in procs:
            try:
                v = _pss_bytes(p.pid)
                if v is None:
                    pss_ok = False
                    v = p.memory_info().rss
                total += v
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / 2**30, pss_ok

    started = time.time()
    peak_mem = 0.0
    peak_mem_pss = True  # skill-fix-20260910:全程 Pss 口径;
    # 任一采样点回退 RSS 时置 False,心跳 mem_pss=False 标注
    next_hb = started + heartbeat_interval_sec
    # v2.17.2(B1):训练前 OOM-kill 计数快照(信号退出时比对,作 OOM 证据)
    try:
        _oom_before = read_cgroup_oom_kills()
    except Exception:  # noqa: BLE001  探测失败 = 无证据(旧行为)
        _oom_before = {}
    # v14:attempt 滚动日志。旧实现 "wb" 截断写,成功重跑会把此前
    # 每次失败的 traceback 全部覆盖(trial-020/021 的 4 次超时原始
    # 报错因此永久丢失,复盘只能靠代码差异反推)。归档旧日志后新开。
    import shutil as _shutil

    for _p in (stdout_path, stderr_path):
        if _p.exists():
            _n = 1
            while (_p.parent / f"{_p.stem}.attempt-{_n}.log").exists():
                _n += 1
            _shutil.move(str(_p),
                         _p.parent / f"{_p.stem}.attempt-{_n}.log")
    with stdout_path.open("wb") as so, stderr_path.open("wb") as se:
        # skill-fix-20260910(修复 2 连带杀事故回写):
        # start_new_session=True(POSIX setsid)有双层语义——
        # (1) 超时终止:_kill_process_group 以本 pid 为根按进程
        #     树枚举,setsid 后 child 是新会话组长,枚举语义不变;
        # (2) 存活隔离:runner/agent 被外部 kill(kill 进程组)
        #     时不再连带杀 trial——训练跑完落盘分数,runner 侧
        #     由孤儿 RUNNING 自愈(runner.run 的 .runner.lock
        #     判定)接管,闭环不丢已完成训练。禁止改回
        #     process_group=0 或去掉本参数。
        proc = subprocess.Popen(
            command, cwd=str(cwd), stdout=so, stderr=se,
            env=env, start_new_session=True,  # 独立进程组(setsid)
        )
        if on_spawn is not None:
            try:
                on_spawn(proc.pid)
            except Exception as _cb_exc:  # noqa: BLE001  回调失败不影响训练
                logger.debug("on_spawn 回调失败(忽略): %s", _cb_exc)
        timed_out = False
        oom = False
        try:
            ps = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            ps = None
        if ps is not None:
            _proc_cpu_pct(ps)  # 预热 cpu_percent 基线(首次调用返回 0)
        deadline = started + timeout_sec
        while True:
            try:
                proc.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                pass
            if time.time() > deadline:
                timed_out = True
                logger.warning("trial 超时(%.0fs),终止进程组 pid=%s",
                               timeout_sec, proc.pid)
                _kill_process_group(proc.pid, grace_sec)
                try:
                    proc.wait(timeout=grace_sec + 10)
                except subprocess.TimeoutExpired:
                    pass
                break
            if ps is not None:
                try:
                    _m_gb, _m_pss = _proc_tree_mem(
                        [ps] + ps.children(recursive=True))
                    peak_mem = max(peak_mem, _m_gb)
                    peak_mem_pss = peak_mem_pss and _m_pss
                except psutil.NoSuchProcess:
                    pass
            if time.time() >= next_hb and ps is not None:
                _hb("heartbeat: runtime=%.0fs cpu=%.0f%% mem=%.1fGB"
                    " mem_pss=%s"
                    % (time.time() - started, _proc_cpu_pct(ps),
                       peak_mem, peak_mem_pss))
                next_hb = time.time() + heartbeat_interval_sec
        runtime = time.time() - started
        if timed_out and proc.returncode is None:
            proc.returncode = -9
        # v2.17.2(B1):信号退出 + 训练期间 oom_kill 计数增加 → OOM 证据
        if not timed_out and proc.returncode in SIGNAL_EXIT_CODES:
            try:
                oom = oom_kill_increased(_oom_before,
                                         read_cgroup_oom_kills())
            except Exception:  # noqa: BLE001
                oom = False
            if oom:
                _hb("oom evidence: cgroup/vmstat oom_kill 计数在训练期间增加"
                    "(rc=%s 判 oom)" % proc.returncode)
        _hb("process exited: rc=%s runtime=%.1fs peak_mem=%.2fGB"
            " mem_pss=%s"
            % (proc.returncode, runtime, peak_mem, peak_mem_pss))

    stderr_tail = ""
    try:
        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
    except OSError:
        pass
    reason = None if proc.returncode == 0 and not timed_out else classify_failure(
        proc.returncode, stderr_tail, timed_out=timed_out, oom=oom)
    return ExecutionResult(
        ok=proc.returncode == 0 and not timed_out,
        returncode=proc.returncode,
        runtime_sec=runtime,
        peak_memory_gb=peak_mem,
        failure_reason=reason,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stderr_tail=stderr_tail if reason else "",
    )


# ---------------------------------------------------------------------------
# 失败熔断(§16.5.6)
# ---------------------------------------------------------------------------

class FailureCircuitBreaker:
    def __init__(self, config: Config):
        cb = config.get("execution.failure_circuit_breaker") or {}
        self.consecutive_limit = int(cb.get("consecutive_failures", 3))
        self.rate_limit = float(cb.get("generation_failure_rate", 0.5))
        self._recent: list[str] = []  # 最近同类 failure_reason 序列

    def record(self, failure_reason: str | None) -> None:
        # v2.17.0:准入超时不是 trial 的失败(训练从未启动),既不累计也
        # 不清零连续计数——资源争用期不应把 run 熔断
        if failure_reason in NON_TRIAL_FAILURES:
            return
        if failure_reason:
            self._recent.append(failure_reason)
        else:
            self._recent = []

    def tripped(self, generation_failures: int, generation_total: int) -> \
            dict | None:
        """返回触发描述;None 表示未触发。"""
        if self._recent and len(self._recent) >= self.consecutive_limit:
            last = self._recent[-self.consecutive_limit:]
            if len(set(last)) == 1:
                return {"kind": "consecutive_same_reason",
                        "reason": last[0], "count": len(last)}
        if generation_total >= 2 and \
                generation_failures / generation_total > self.rate_limit:
            return {"kind": "generation_failure_rate",
                    "rate": generation_failures / generation_total}
        return None


# ---------------------------------------------------------------------------
# attempt 管理(§16.5.4)
# ---------------------------------------------------------------------------

@dataclass
class AttemptPolicy:
    max_retries: int = 1
    max_fix_rounds: int = 2

    @classmethod
    def from_config(cls, config: Config) -> "AttemptPolicy":
        return cls(
            max_retries=int(config.get("execution.max_retries_per_trial", 1)),
            max_fix_rounds=int(config.get("execution.max_fix_rounds_per_trial", 2)),
        )

    def next_action(self, trial: TrialRecord, failure_reason: str) -> str:
        """返回 retry / fix_code / give_up。

        - timeout/oom/infra_error:自动重试(可降配),上限 max_retries;
        - code_error:交回 Trial Agent 修复,上限 max_fix_rounds;
        - evaluator_error:重跑评估(由 runner 层处理,此处同 retry);
        - admission_timeout(v2.17.0):返回 requeue——runner 不自动重试,
          重入时记 requeue 而不消耗 retry/fix 预算;
        - 超限:failed 终态,verdict=invalid。
        """
        if failure_reason in NON_TRIAL_FAILURES:
            return "requeue"
        attempts = trial.data.get("attempts") or []
        n_retries = sum(1 for a in attempts if a.get("kind") == "retry")
        n_fixes = sum(1 for a in attempts if a.get("kind") == "fix_code")
        if failure_reason == "code_error":
            if n_fixes < self.max_fix_rounds:
                return "fix_code"
            return "give_up"
        if failure_reason in RETRYABLE_FAILURES:
            if n_retries < self.max_retries:
                return "retry"
            return "give_up"
        return "give_up"


def record_attempt(trial: TrialRecord, kind: str, note: str,
                   extra: dict | None = None) -> None:
    trial.data.setdefault("attempts", []).append({
        "kind": kind, "note": note, "at": now_iso(), **(extra or {}),
    })
    trial.flush()


# ---------------------------------------------------------------------------
# v2.17.0(WP5a):内存准入
#
# 修什么(run4 regression.2,8×A800 / 960GB 与其它 run 共享):同机其它 run
# 常驻 300~740 GB,可用内存一度只剩 23~50 GB;本 run 单 trial 峰值
# 79~179 GB。generation_batch 计划给出 8 并发却不看内存,4 个 trial 一起
# 起必然 OOM;agent 只能每 ~2 min 手工错峰 `run --detach`,finalize 还把
# 这种资源受限串行记成 avg_parallelism<1.5 的编排问题。
#
# 怎么修:每次启动训练前过准入——
#   有效可用 = min(MemAvailable, cgroup 限额 − 不可回收用量)
#   未兑现预留 = Σ 本 run 活预留 max(0, 预估峰值 − 训练进程树当前实占)
#   准入条件:有效可用 − 未兑现预留 − 安全余量 ≥ 本 trial 预估峰值
# 不满足就排队(每 admission_poll_sec 复查),等满 admission_max_wait_min
# 记 FAILED(admission_timeout)。预留文件在 <run_dir>/.admission/,持有
# 进程死亡即陈旧(自动回收);"判定 + 落预留"在 flock 临界区内,所以
# run-generation 的线程池与手工并行的 `run --detach` worker 都逐个准入。
# 日志:<run_dir>/logs/admission.jsonl(queued / admitted / timeout /
# gpu_select)。
# ---------------------------------------------------------------------------

#: 探测入口(模块级常量,测试替换为 mock 文件)
MEMINFO_PATH = "/proc/meminfo"
CGROUP_ROOT = "/sys/fs/cgroup"
PROC_SELF_CGROUP = "/proc/self/cgroup"
#: v2.17.2(B1):全局 OOM-kill 计数(cgroup 计数不可得时的回退证据)
PROC_VMSTAT = "/proc/vmstat"
_GIB = float(2 ** 30)
_CGROUP_UNLIMITED = 2 ** 60
#: 由实测峰值外推本 trial 峰值的放大系数(父代 / lineage)
PEAK_SAFETY_FACTOR = 1.15
#: 预估峰值不超过该值的 trial 不排队(单独无法触发 OOM,等它不划算)
SMALL_TRIAL_GB = 2.0
#: 无历史时按数据面体积估峰值的倍数(run4:数据面 20GB → baseline 77GB)
DATA_SIZE_MEM_FACTOR = 4.0
ADMISSION_DIR = ".admission"
ADMISSION_LOG = "admission.jsonl"
#: 准入逃生口(诊断用):MODELEVO_MEMORY_ADMISSION=0 关闭准入
ADMISSION_ENV = "MODELEVO_MEMORY_ADMISSION"
_ADMISSION_MTX = threading.Lock()


def _gb(v) -> float | None:
    return None if v is None else round(float(v) / _GIB, 2)


def read_meminfo(path: str | None = None) -> dict:
    """``/proc/meminfo`` → {键: 字节};读不到返回 {}。"""
    out: dict = {}
    try:
        with open(path or MEMINFO_PATH, encoding="ascii",
                  errors="ignore") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if not parts:
                    continue
                try:
                    val = int(parts[0])
                except ValueError:
                    continue
                unit = parts[1].lower() if len(parts) > 1 else ""
                out[key.strip()] = val * 1024 if unit == "kb" else val
    except OSError:
        return {}
    return out


def _read_cg_int(p: Path) -> int | None:
    """cgroup 数值文件;"max" 返回 -1(无限额);读不到 None。"""
    try:
        raw = Path(p).read_text(encoding="ascii").strip()
    except OSError:
        return None
    if raw == "max":
        return -1
    try:
        return int(raw)
    except ValueError:
        return None


def _read_cg_stat(p: Path) -> dict:
    out: dict = {}
    try:
        for line in Path(p).read_text(encoding="ascii",
                                      errors="ignore").splitlines():
            k, _, v = line.partition(" ")
            try:
                out[k] = int(v.strip())
            except ValueError:
                pass
    except OSError:
        pass
    return out


def _proc_cgroup_paths(proc_cgroup: str) -> tuple[str | None, str | None]:
    """(v2 统一层级路径, v1 memory 控制器路径),取自 /proc/self/cgroup。"""
    v2 = v1 = None
    try:
        text = Path(proc_cgroup).read_text(encoding="ascii", errors="ignore")
    except OSError:
        return None, None
    for line in text.splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        hid, ctrls, path = parts
        if hid == "0" and ctrls == "":
            v2 = path
        elif "memory" in ctrls.split(","):
            v1 = path
    return v2, v1


def _cg_dirs(base: Path, rel: str | None) -> list[Path]:
    """rel 从叶到根的祖先目录(存在的)+ base 本身(容器内挂载点即本 cgroup)。"""
    out: list[Path] = []
    parts = [p for p in (rel or "").strip("/").split("/") if p]
    for i in range(len(parts), 0, -1):
        d = base.joinpath(*parts[:i])
        if d.is_dir() and d not in out:
            out.append(d)
    if base not in out:
        out.append(base)
    return out


def read_cgroup_oom_kills(root: str | None = None,
                          proc_cgroup: str | None = None,
                          vmstat_path: str | None = None) -> dict[str, int]:
    """v2.17.2(B1):OOM-kill 计数快照 ``{来源: 计数}``(训练前后对比用)。

    cgroup v2 读本 cgroup 层级各级 ``memory.events`` 的 ``oom_kill``;v1 读
    ``memory.oom_control`` 的 ``oom_kill``(内核 ≥4.13)。都读不到时回退
    ``/proc/vmstat`` 的全局 ``oom_kill``。任何一项在训练期间增加 = OOM 证据。
    """
    root_p = Path(root or CGROUP_ROOT)
    v2rel, v1rel = _proc_cgroup_paths(proc_cgroup or PROC_SELF_CGROUP)
    out: dict[str, int] = {}
    for d in _cg_dirs(root_p, v2rel):
        ev = _read_cg_stat(d / "memory.events")
        if "oom_kill" in ev:
            out[f"v2:{d}"] = int(ev["oom_kill"])
    for d in _cg_dirs(root_p / "memory", v1rel):
        oc = _read_cg_stat(d / "memory.oom_control")
        if "oom_kill" in oc:
            out[f"v1:{d}"] = int(oc["oom_kill"])
    if not out:
        vm = _read_cg_stat(Path(vmstat_path or PROC_VMSTAT))
        if "oom_kill" in vm:
            out["vmstat"] = int(vm["oom_kill"])
    return out


def oom_kill_increased(before: dict | None, after: dict | None) -> bool:
    """任一来源的 oom_kill 计数增加。"""
    if not before or not after:
        return False
    return any(int(after.get(k, 0)) > int(v) for k, v in before.items())


def read_cgroup_memory(root: str | None = None,
                       proc_cgroup: str | None = None) -> dict | None:
    """本进程 cgroup 的内存限额与用量;无有限额时返回 None。

    v2 读 memory.max / memory.current / memory.stat(inactive_file);
    v1 读 memory.limit_in_bytes / memory.usage_in_bytes / memory.stat
    (total_inactive_file)。沿层级取**最紧**的一层;cgroup 可用 =
    限额 − (用量 − 可回收的 inactive file 页)。
    """
    root_p = Path(root or CGROUP_ROOT)
    v2rel, v1rel = _proc_cgroup_paths(proc_cgroup or PROC_SELF_CGROUP)
    best: dict | None = None

    def _consider(cand: dict) -> None:
        nonlocal best
        used = max(0, cand["usage_bytes"] - cand["reclaimable_bytes"])
        cand["available_bytes"] = max(0, cand["limit_bytes"] - used)
        if best is None or cand["available_bytes"] < best["available_bytes"]:
            best = cand

    for d in _cg_dirs(root_p, v2rel):
        lim = _read_cg_int(d / "memory.max")
        if lim is None or lim < 0 or lim >= _CGROUP_UNLIMITED:
            continue
        usage = _read_cg_int(d / "memory.current")
        if usage is None or usage < 0:
            continue
        stat = _read_cg_stat(d / "memory.stat")
        _consider({"version": 2, "path": str(d), "limit_bytes": lim,
                   "usage_bytes": usage,
                   "reclaimable_bytes": int(stat.get("inactive_file", 0))})
    if best is None:
        for d in _cg_dirs(root_p / "memory", v1rel):
            lim = _read_cg_int(d / "memory.limit_in_bytes")
            if lim is None or lim < 0 or lim >= _CGROUP_UNLIMITED:
                continue
            usage = _read_cg_int(d / "memory.usage_in_bytes")
            if usage is None or usage < 0:
                continue
            stat = _read_cg_stat(d / "memory.stat")
            recl = stat.get("total_inactive_file",
                            stat.get("inactive_file", 0))
            _consider({"version": 1, "path": str(d), "limit_bytes": lim,
                       "usage_bytes": usage, "reclaimable_bytes": int(recl)})
    return best


def system_memory(*, meminfo_path: str | None = None,
                  cgroup_root: str | None = None,
                  proc_cgroup: str | None = None) -> dict:
    """系统内存视图(GB):MemAvailable 与 cgroup 限额取紧者为有效可用。

    返回 ``{mem_total_gb, mem_available_gb, cgroup: {...}|None,
    available_gb, capacity_gb, source}``;全部探测失败时 available_gb=None
    (准入 fail-open)。
    """
    mi = read_meminfo(meminfo_path)
    total, avail = mi.get("MemTotal"), mi.get("MemAvailable")
    source = "meminfo"
    if avail is None and mi.get("MemFree") is not None:
        # 老内核无 MemAvailable:MemFree + Buffers + Cached 近似
        avail = mi["MemFree"] + mi.get("Buffers", 0) + mi.get("Cached", 0)
        source = "meminfo_approx"
    if total is None or avail is None:
        try:
            vm = psutil.virtual_memory()
            total, avail, source = vm.total, vm.available, "psutil"
        except Exception:  # noqa: BLE001
            total = avail = None
            source = "unavailable"
    cg = read_cgroup_memory(cgroup_root, proc_cgroup)
    eff, cap, cg_out = avail, total, None
    if cg is not None:
        cg_out = {"version": cg["version"], "path": cg["path"],
                  "limit_gb": _gb(cg["limit_bytes"]),
                  "usage_gb": _gb(cg["usage_bytes"]),
                  "reclaimable_gb": _gb(cg["reclaimable_bytes"]),
                  "available_gb": _gb(cg["available_bytes"])}
        eff = cg["available_bytes"] if eff is None \
            else min(eff, cg["available_bytes"])
        cap = cg["limit_bytes"] if cap is None \
            else min(cap, cg["limit_bytes"])
        source += "+cgroup"
    return {"mem_total_gb": _gb(total), "mem_available_gb": _gb(avail),
            "cgroup": cg_out, "available_gb": _gb(eff),
            "capacity_gb": _gb(cap), "source": source}


# ---- 单 trial 峰值估计 ------------------------------------------------------

def _pos_float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 and math.isfinite(f) else None


def _peak_of(data: dict | None) -> float | None:
    return _pos_float(((data or {}).get("result") or {})
                      .get("peak_memory_gb"))


def _status_epoch(data: dict) -> float:
    """最后一条 status_history 的时间(排序"最近"用);解析失败 0。"""
    from datetime import datetime

    for ent in reversed((data or {}).get("status_history") or []):
        try:
            return datetime.fromisoformat(str(ent.get("at"))).timestamp()
        except (TypeError, ValueError):
            continue
    return 0.0


def _parents_of(data: dict) -> list[str]:
    raw = (data or {}).get("parent_trial_ids")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw]


def _load_trial_yaml(run_dir: Path, tid: str) -> dict | None:
    from .logging_utils import load_yaml

    p = Path(run_dir) / "trials" / str(tid) / "trial.yaml"
    try:
        d = load_yaml(p) if p.exists() else None
    except Exception:  # noqa: BLE001  损坏的 trial.yaml 不阻塞估计
        return None
    return d if isinstance(d, dict) else None


def _all_trial_data(run_dir: Path) -> dict:
    out: dict = {}
    root = Path(run_dir) / "trials"
    try:
        dirs = sorted(root.iterdir()) if root.is_dir() else []
    except OSError:
        return out
    for d in dirs:
        data = _load_trial_yaml(run_dir, d.name)
        if data is not None:
            out[str(data.get("trial_id") or d.name)] = data
    return out


def data_plane_gb(run_dir: Path, data_view: str | None = None) -> float | None:
    """数据面(splits 目录)磁盘体积 GB;不存在返回 None。"""
    sp = Path(run_dir) / "sample-profile"
    view = str(data_view or "v1")
    d = sp / ("splits" if view in ("v1", "base") else f"splits-{view}")
    if not d.is_dir():
        d = sp / "splits"
    if not d.is_dir():
        return None
    total = 0
    for base, _dirs, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(base, f))
            except OSError:
                pass
    return total / _GIB


def _view_key(v) -> str:
    s = str(v or "v1").strip()
    return "v1" if s in ("", "v1", "base", "None") else s


def _peer_peak_estimate(run_dir: Path, data: dict, parents: list,
                        config: Config | None = None) -> dict | None:
    """v2.19.0:同类 trial 实测峰值校准(见 estimate_trial_peak_gb 步骤 0)。

    同类 = 同数据视图,且其父代链(父代及全部祖先)包含本 trial 的任一
    父代(兄弟 / 父代的其它后代);父代本身不算(配方不同,峰值常偏离)。
    取最近 ``execution.memory_peer_window``(默认 3)个的最大实测峰值 ×
    ``execution.memory_peer_safety_factor``(默认 1.25)。无同类返回 None。
    """
    window, factor = 3, 1.25
    if config is not None:
        try:
            window = max(1, int(config.get(
                "execution.memory_peer_window", 3) or 3))
            factor = float(config.get(
                "execution.memory_peer_safety_factor", 1.25) or 1.25)
        except (TypeError, ValueError):
            window, factor = 3, 1.25
    tid = str((data or {}).get("trial_id") or "")
    view = _view_key((data or {}).get("data_view"))
    pset = {str(p) for p in parents}
    all_ = _all_trial_data(run_dir)
    memo: dict = {}

    def _chain(t: str, depth: int = 0) -> set:
        if t in memo:
            return memo[t]
        memo[t] = set()  # 防环
        acc: set = set()
        if depth < 64:
            for p in _parents_of(all_.get(t) or {}):
                acc.add(p)
                acc |= _chain(p, depth + 1)
        memo[t] = acc
        return acc

    cands = []
    for otid, od in all_.items():
        if otid == tid or otid in pset:
            continue
        pk = _peak_of(od)
        if not pk or _view_key(od.get("data_view")) != view:
            continue
        if _chain(otid) & pset:
            cands.append((_status_epoch(od), otid, pk))
    if not cands:
        return None
    recent = sorted(cands, reverse=True)[:window]
    _, btid, bpk = max(recent, key=lambda c: c[2])
    return {"estimate_gb": bpk * factor, "basis_trial": btid,
            "basis_gb": bpk, "basis_trials": [c[1] for c in recent],
            "safety_factor": factor}


def estimate_trial_peak_gb(run_dir: Path, trial_data: dict | None,
                           config: Config | None = None) -> dict:
    """单 trial 峰值内存估计,来源优先级:

    0. v2.19.0 同类实测(``source=peer``):本 trial 无自身 attempt 实测
       时,同视图、共享父代链的已完成 trial 最近 k 个峰值的最大值 ×
       memory_peer_safety_factor(默认 1.25),见 _peer_peak_estimate;
    1. 父代 ``result.peak_memory_gb`` × 1.15(多父代取最大);
    2. 同 lineage 最近 trial × 1.15:先看本 trial 历次 attempt 的
       previous_result,再看祖先 / 共享祖先的亲缘 trial 中最近完成者;
    3. trial.yaml 的 ``expected_resources.peak_memory_gb``(原值);
    4. 缺省:``execution.memory_estimate_default_gb``;数据面存在时取
       min(缺省, max(1, 数据面体积 × 4))(小数据不按大缺省排队)。

    返回 ``{estimate_gb, source, basis_trial, basis_gb}``。
    """
    data = trial_data or {}
    run_dir = Path(run_dir)
    default_gb = 64.0
    if config is not None:
        default_gb = float(config.get("execution.memory_estimate_default_gb",
                                      64) or 64)
    tid = str(data.get("trial_id") or "")

    def _out(est, source, basis_trial=None, basis_gb=None, **kw):
        return {"estimate_gb": round(float(est), 2), "source": source,
                "basis_trial": basis_trial,
                "basis_gb": round(basis_gb, 2) if basis_gb else None, **kw}

    # 1. 父代
    parents = _parents_of(data)
    own_attempts = [a for a in (data.get("attempts") or [])
                    if isinstance(a, dict)
                    and _peak_of({"result": a.get("previous_result")})]
    # v2.19.0:同类 trial 实测校准(本 trial 无自身 attempt 实测时优先)。
    # 旧口径取父代峰值 ×1.15:父代若是 gen-00 冷缓存 / warm-start 的高峰
    # (run4-v2.18.0-4 trial-002 141 GB),其全部后代都按 163 GB 排队,
    # 而这些后代实测 53~114 GB(中位 ~62 GB)——高估约 2 倍,内存口径
    # 并发被压到 4 路。改为:同数据视图、且与本 trial 共享父代(兄弟 /
    # 父代的其它后代)的已完成 trial,取最近 k 个实测峰值的最大值 ×
    # memory_peer_safety_factor(默认 1.25)。无同类实测时维持旧逻辑。
    if parents and not own_attempts:
        peer = _peer_peak_estimate(run_dir, data, parents, config)
        if peer is not None:
            return _out(peer["estimate_gb"], "peer", peer["basis_trial"],
                        peer["basis_gb"], basis_trials=peer["basis_trials"],
                        safety_factor=peer["safety_factor"])
    best = None
    for pid in parents:
        pk = _peak_of(_load_trial_yaml(run_dir, pid))
        if pk and (best is None or pk > best[1]):
            best = (pid, pk)
    if best:
        return _out(best[1] * PEAK_SAFETY_FACTOR, "parent", best[0], best[1])

    # 2. 同 lineage:自身历次 attempt → 祖先/亲缘中最近完成者
    own = [_peak_of({"result": a.get("previous_result")})
           for a in (data.get("attempts") or []) if isinstance(a, dict)]
    own = [p for p in own if p]
    if not own and _peak_of(data):
        own = [_peak_of(data)]
    if own:
        return _out(own[-1] * PEAK_SAFETY_FACTOR, "lineage", tid or None,
                    own[-1])
    all_ = _all_trial_data(run_dir)
    memo: dict = {}

    def _anc(t: str, depth: int = 0) -> set:
        if t in memo:
            return memo[t]
        memo[t] = set()  # 防环
        acc: set = set()
        if depth < 64:
            for p in _parents_of(all_.get(t) or {}):
                acc.add(p)
                acc |= _anc(p, depth + 1)
        memo[t] = acc
        return acc

    mine: set = set()
    for p in parents:
        mine.add(p)
        mine |= _anc(p)
    family = mine | ({tid} if tid else set())
    cands = []
    for otid, od in all_.items():
        if otid == tid:
            continue
        pk = _peak_of(od)
        if not pk:
            continue
        if otid in mine or (family and _anc(otid) & family):
            cands.append((_status_epoch(od), otid, pk))
    if cands:
        _, otid, pk = max(cands)
        return _out(pk * PEAK_SAFETY_FACTOR, "lineage", otid, pk)

    # 3. 声明值
    declared = _pos_float((data.get("expected_resources") or {})
                          .get("peak_memory_gb"))
    if declared:
        return _out(declared, "declared")

    # 4. 缺省
    dp = data_plane_gb(run_dir, data.get("data_view"))
    if dp is not None:
        return _out(min(default_gb, max(1.0, dp * DATA_SIZE_MEM_FACTOR)),
                    "default", data_plane_gb=round(dp, 3))
    return _out(default_gb, "default")


def _pending_trials_peak_gb(run_dir: Path, config: Config | None) -> dict | None:
    """v2.19.0:待跑 trial(PROPOSED/CODE_WRITTEN/FAILED)各自峰值估计的最大值;
    无待跑 trial 返回 None。"""
    best = None
    for tid, d in _all_trial_data(run_dir).items():
        if str(d.get("status") or "") not in ("PROPOSED", "CODE_WRITTEN",
                                              "FAILED"):
            continue
        e = estimate_trial_peak_gb(run_dir, d, config)
        if best is None or float(e["estimate_gb"]) > float(best["estimate_gb"]):
            best = dict(e, basis_trials=[tid])
    if best is not None:
        best["source"] = f"pending_max:{best.get('source')}"
    return best


def recent_trials_peak_gb(run_dir: Path, k: int = 3) -> dict | None:
    """最近 k 个有实测峰值的 trial:取其最大峰值 × 1.15(snapshot 建议并发用)。"""
    rows = []
    for tid, d in _all_trial_data(run_dir).items():
        pk = _peak_of(d)
        if pk:
            rows.append((_status_epoch(d), tid, pk))
    if not rows:
        return None
    rows.sort(reverse=True)
    top = rows[:max(1, int(k))]
    peak = max(r[2] for r in top)
    return {"estimate_gb": round(peak * PEAK_SAFETY_FACTOR, 2),
            "source": "recent_trials", "basis_trials": [r[1] for r in top],
            "basis_gb": round(peak, 2)}


# ---- 预留与准入 ---------------------------------------------------------------

def _proc_tree_mem_gb(pid: int) -> float | None:
    """进程树内存(GB,优先 Pss,回退 RSS);进程不存在返回 None。"""
    try:
        root = psutil.Process(int(pid))
        procs = [root] + root.children(recursive=True)
    except (psutil.Error, ValueError):
        return None
    total = 0
    for q in procs:
        v = None
        try:
            with open(f"/proc/{q.pid}/smaps_rollup") as fh:
                for line in fh:
                    if line.startswith("Pss:"):
                        v = int(line.split()[1]) * 1024
                        break
        except (OSError, ValueError, IndexError):
            v = None
        try:
            total += v if v is not None else q.memory_info().rss
        except psutil.Error:
            pass
    return total / _GIB


def _atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-"
                         f"{threading.get_ident()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, default=str),
                   encoding="utf-8")
    os.replace(str(tmp), str(path))


@dataclass
class AdmissionTicket:
    """一次准入结果。decision ∈ admitted / small / no_probe / disabled /
    timeout;只有 timeout 不允许启动训练。"""

    trial_id: str
    decision: str
    estimate_gb: float
    source: str | None = None
    wait_sec: float = 0.0
    path: Path | None = None
    token: str = ""
    available_gb: float | None = None
    basis_trial: str | None = None

    @property
    def admitted(self) -> bool:
        return self.decision != "timeout"

    def attach_pid(self, pid: int) -> None:
        """记下训练进程 pid:未兑现预留按 预估 − 实占 计。"""
        if self.path is None:
            return
        with _ADMISSION_MTX:
            try:
                d = json.loads(Path(self.path).read_text(encoding="utf-8"))
                if d.get("token") != self.token:
                    return
                d["child_pid"] = int(pid)
                _atomic_write_json(Path(self.path), d)
            except (OSError, ValueError):
                pass

    def release(self) -> None:
        """释放预留(只删自己的;幂等)。"""
        if self.path is None:
            return
        with _ADMISSION_MTX:
            try:
                d = json.loads(Path(self.path).read_text(encoding="utf-8"))
                if d.get("token") == self.token:
                    Path(self.path).unlink()
            except (OSError, ValueError):
                pass
        self.path = None

    def to_record(self) -> dict:
        """写进 trial.yaml.admission 的摘要。"""
        return {"state": "timeout" if not self.admitted else "admitted",
                "decision": self.decision,
                "estimate_gb": round(float(self.estimate_gb), 2),
                "estimate_source": self.source,
                "basis_trial": self.basis_trial,
                "wait_sec": round(float(self.wait_sec), 1),
                "available_gb": self.available_gb, "at": now_iso()}


class MemoryAdmission:
    """跨进程内存准入(见本节模块说明)。"""

    def __init__(self, run_dir: Path, config: Config | None = None, *,
                 memory_probe=None, sleep=None, clock=None):
        self.run_dir = Path(run_dir)

        def _g(key, default):
            if config is None:
                return default
            v = config.get(key, default)
            return default if v is None else v

        self.enabled = bool(_g("execution.memory_admission", True))
        if os.environ.get(ADMISSION_ENV) == "0":
            self.enabled = False
        self.margin_gb = float(_g("execution.memory_safety_margin_gb", 60))
        self.default_gb = float(_g("execution.memory_estimate_default_gb",
                                   64))
        self.poll_sec = max(0.01, float(_g("execution.admission_poll_sec",
                                           30)))
        self.max_wait_sec = max(0.0, float(_g(
            "execution.admission_max_wait_min", 120)) * 60.0)
        self._probe = memory_probe or system_memory
        self._sleep = sleep or time.sleep
        self._clock = clock or time.time

    # ---- 预留 ----------------------------------------------------------
    @property
    def res_dir(self) -> Path:
        return self.run_dir / ADMISSION_DIR

    @contextmanager
    def _locked(self):
        with _ADMISSION_MTX:
            self.res_dir.mkdir(parents=True, exist_ok=True)
            fh = open(self.res_dir / ".lock", "a+")
            locked = False
            try:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                    locked = True
                except OSError as exc:
                    # 文件系统不支持 flock(少见的网络 FS):退化为进程内互斥,
                    # 不让准入整体失效(跨进程仍能经预留文件看到彼此)
                    logger.warning("准入 flock 不可用(%s),退化为进程内互斥",
                                   exc)
                yield
            finally:
                try:
                    if locked:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                finally:
                    fh.close()

    def reservations(self, *, reclaim: bool = True) -> list[dict]:
        """本 run 的活预留,附 used/outstanding。reclaim=True 时顺手删除
        持有进程已死的陈旧预留(只读视图如 snapshot 传 False)。"""
        out: list[dict] = []
        if not self.res_dir.is_dir():
            return out
        for f in sorted(self.res_dir.glob("*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            try:
                pid = int(d.get("pid") or 0)
            except (TypeError, ValueError):
                pid = 0
            if not _pid_alive(pid):
                if reclaim:
                    try:
                        f.unlink()
                    except OSError:
                        pass
                continue
            est = float(d.get("estimate_gb") or 0.0)
            used = None
            cp = d.get("child_pid")
            if cp and _pid_alive(int(cp)):
                used = _proc_tree_mem_gb(int(cp))
            d["used_gb"] = round(used, 2) if used is not None else None
            d["outstanding_gb"] = round(max(0.0, est - (used or 0.0)), 2)
            out.append(d)
        return out

    def _write_reservation(self, trial_id: str, est: float, source,
                           generation) -> tuple[Path, str]:
        token = os.urandom(4).hex()
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_"
                       for c in str(trial_id))
        path = self.res_dir / f"{safe}.{os.getpid()}.{token}.json"
        _atomic_write_json(path, {
            "trial_id": trial_id, "pid": os.getpid(), "token": token,
            "estimate_gb": round(float(est), 2), "estimate_source": source,
            "generation": generation, "admitted_at": now_iso(),
            "admitted_epoch": round(time.time(), 3), "child_pid": None})
        return path, token

    # ---- 判定 ----------------------------------------------------------
    def evaluate(self, estimate_gb: float) -> dict:
        """一次准入判定(不加锁、不落预留;snapshot/诊断也用)。"""
        mem = self._probe() or {}
        avail = mem.get("available_gb")
        res = self.reservations()
        outstanding = round(sum(r["outstanding_gb"] for r in res), 2)
        need = float(estimate_gb)
        cap = mem.get("capacity_gb")
        if cap:
            # 估计超出整机能力:按"整机几乎全空"准入,而不是永远排队
            need = min(need, max(float(cap) - self.margin_gb, 0.0))
        headroom = None if avail is None else round(
            float(avail) - outstanding - self.margin_gb, 2)
        if avail is None:
            ok, why = True, "no_probe"
        elif headroom >= need:
            ok, why = True, "admitted"
        elif float(estimate_gb) <= SMALL_TRIAL_GB:
            ok, why = True, "small"
        else:
            ok, why = False, "insufficient_memory"
        return {"ok": ok, "decision": why, "available_gb": avail,
                "mem_available_gb": mem.get("mem_available_gb"),
                "cgroup_available_gb": (mem.get("cgroup") or {})
                .get("available_gb"),
                "outstanding_gb": outstanding, "n_reserved": len(res),
                "headroom_gb": headroom, "need_gb": round(need, 2),
                "margin_gb": self.margin_gb}

    def _log(self, event: str, trial_id: str, generation, est: dict,
             ev: dict | None, **extra) -> None:
        rec = {"at": now_iso(), "epoch": round(time.time(), 3),
               "event": event, "trial_id": trial_id,
               "generation": generation, "pid": os.getpid(),
               "estimate_gb": est.get("estimate_gb"),
               "estimate_source": est.get("source"),
               "basis_trial": est.get("basis_trial")}
        for k in ("available_gb", "mem_available_gb", "cgroup_available_gb",
                  "outstanding_gb", "n_reserved", "need_gb", "margin_gb",
                  "decision"):
            if ev and k in ev:
                rec[k] = ev[k]
        rec.update(extra)
        append_admission_log(self.run_dir, rec)

    def acquire(self, trial_id: str, estimate: dict, *, generation=None,
                max_wait_sec: float | None = None, on_queue=None,
                on_wait=None) -> AdmissionTicket:
        """阻塞直到准入或超时。返回 AdmissionTicket(timeout 时不落预留)。

        on_queue(ev):首次被拒时回调一次;on_wait(ev, waited):排队期间
        至多每 5 分钟回调一次(runner 用来写 trial.log / trial.yaml)。
        """
        est_gb = float(estimate.get("estimate_gb") or self.default_gb)
        src = estimate.get("source")
        basis = estimate.get("basis_trial")
        if not self.enabled:
            return AdmissionTicket(trial_id, "disabled", est_gb, src,
                                   basis_trial=basis)
        max_wait = self.max_wait_sec if max_wait_sec is None \
            else max(0.0, float(max_wait_sec))
        t0 = self._clock()
        queued_at = None
        last_note = t0
        min_avail = None
        while True:
            with self._locked():
                ev = self.evaluate(est_gb)
                avail = ev.get("available_gb")
                if avail is not None:
                    min_avail = avail if min_avail is None \
                        else min(min_avail, avail)
                if ev["ok"]:
                    waited = max(0.0, self._clock() - t0)
                    path, token = self._write_reservation(
                        trial_id, est_gb, src, generation)
                    self._log("admitted", trial_id, generation, estimate, ev,
                              wait_sec=round(waited, 1),
                              queued=queued_at is not None,
                              min_available_gb=min_avail)
                    return AdmissionTicket(
                        trial_id, ev["decision"], est_gb, src,
                        wait_sec=waited, path=path, token=token,
                        available_gb=avail, basis_trial=basis)
            waited = max(0.0, self._clock() - t0)
            if queued_at is None:
                queued_at = waited
                self._log("queued", trial_id, generation, estimate, ev)
                logger.warning(
                    "内存准入排队: %s 预估峰值 %.1fGB(%s),有效可用 %sGB − 本 run "
                    "未兑现预留 %.1fGB − 安全余量 %.0fGB 不足;每 %.0fs 复查,最多等"
                    " %.0f min", trial_id, est_gb, src, avail,
                    ev.get("outstanding_gb") or 0.0, self.margin_gb,
                    self.poll_sec, max_wait / 60.0)
                if on_queue is not None:
                    try:
                        on_queue(ev)
                    except Exception:  # noqa: BLE001
                        pass
            elif on_wait is not None and self._clock() - last_note >= 300:
                last_note = self._clock()
                try:
                    on_wait(ev, waited)
                except Exception:  # noqa: BLE001
                    pass
            if waited >= max_wait:
                self._log("timeout", trial_id, generation, estimate, ev,
                          wait_sec=round(waited, 1),
                          min_available_gb=min_avail)
                logger.error(
                    "内存准入超时: %s 排队 %.1f min 仍不满足(可用 %sGB,需 %.1fGB"
                    " + 余量 %.0fGB)→ FAILED(admission_timeout,infra 类)",
                    trial_id, waited / 60.0, avail, est_gb, self.margin_gb)
                return AdmissionTicket(trial_id, "timeout", est_gb, src,
                                       wait_sec=waited, available_gb=avail,
                                       basis_trial=basis)
            self._sleep(max(0.0, min(self.poll_sec, max_wait - waited)))


def append_admission_log(run_dir: Path, record: dict) -> None:
    """追加一行到 <run_dir>/logs/admission.jsonl(失败只告警,不阻塞)。"""
    try:
        p = Path(run_dir) / "logs" / ADMISSION_LOG
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError as exc:
        logger.warning("admission.jsonl 写入失败(忽略): %s", exc)


def read_admission_log(run_dir: Path) -> list[dict]:
    p = Path(run_dir) / "logs" / ADMISSION_LOG
    out: list[dict] = []
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return out
    for ln in lines:
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if isinstance(d, dict):
            out.append(d)
    return out


def admission_summary(run_dir, generation=None) -> dict:
    """准入排队摘要(finalize-generation 的 avg_parallelism 告警归因用)。

    返回 ``{throttled, waits_sec, n_waited, reason}``(另附 n_timeout /
    max_wait_sec / min_available_gb / trials 明细)。throttled=True 表示
    本代(generation=None 时全 run)至少一个 trial 因内存准入排过队或准入
    超时——串行是资源受限,不是编排问题。日志缺失时 throttled=False。
    """
    per: dict[str, dict] = {}
    min_avail = None
    for e in read_admission_log(run_dir):
        if generation is not None:
            try:
                if e.get("generation") is None \
                        or int(e.get("generation")) != int(generation):
                    continue
            except (TypeError, ValueError):
                continue
        ev = e.get("event")
        if ev not in ("queued", "admitted", "timeout"):
            continue
        tid = str(e.get("trial_id") or "?")
        st = per.setdefault(tid, {"wait_sec": 0.0, "queued": False,
                                  "timeout": False, "pending_since": None,
                                  "pid": None})
        if ev == "queued":
            st["queued"] = True
            st["pending_since"] = e.get("epoch")
            st["pid"] = e.get("pid")
            a = e.get("available_gb")
            if isinstance(a, (int, float)):
                min_avail = a if min_avail is None else min(min_avail, a)
        else:
            try:
                st["wait_sec"] += float(e.get("wait_sec") or 0.0)
            except (TypeError, ValueError):
                pass
            st["pending_since"] = None
            if ev == "timeout":
                st["timeout"] = True
            m = e.get("min_available_gb")
            if isinstance(m, (int, float)) and (e.get("queued")
                                                or ev == "timeout"):
                min_avail = m if min_avail is None else min(min_avail, m)
    now = time.time()
    for st in per.values():
        # 仍在排队(持有进程活着):计到当前
        if st["pending_since"] and st["pid"] and _pid_alive(int(st["pid"])):
            st["wait_sec"] += max(0.0, now - float(st["pending_since"]))
    waited = {t: s for t, s in per.items() if s["queued"] or s["timeout"]}
    total = round(sum(s["wait_sec"] for s in waited.values()), 1)
    n_timeout = sum(1 for s in waited.values() if s["timeout"])
    max_wait = round(max((s["wait_sec"] for s in waited.values()),
                         default=0.0), 1)
    throttled = bool(waited)
    reason = ""
    if throttled:
        reason = (f"资源受限串行:{len(waited)} 个 trial 因内存准入排队,累计等待 "
                  f"{total / 60:.1f} min(最长 {max_wait / 60:.1f} min)")
        if n_timeout:
            reason += f";{n_timeout} 个准入超时(admission_timeout,infra 类)"
        if min_avail is not None:
            reason += f";排队期间有效可用内存最低 {min_avail:.0f} GB"
        reason += "。并行度低由同机内存争用造成,不是编排问题"
    return {"throttled": throttled, "waits_sec": total,
            "n_waited": len(waited), "reason": reason,
            "n_timeout": n_timeout, "max_wait_sec": max_wait,
            "min_available_gb": min_avail,
            "trials": {t: round(s["wait_sec"], 1)
                       for t, s in sorted(waited.items())}}


def resource_admission_view(run_dir: Path | None, config: Config | None,
                            *, max_parallel: int | None = None,
                            memory: dict | None = None) -> dict:
    """snapshot 用:系统内存 + 按准入算出的建议并发数。

    单 trial 估计取 run 内最近 3 个有实测峰值的 trial 的最大峰值 × 1.15;
    无历史时按 estimate_trial_peak_gb 的缺省口径。建议并发 =
    floor((有效可用 − 本 run 未兑现预留 − 安全余量) / 单 trial 估计),
    封顶 max_parallel(CPU 口径的 generation_batch 并发)。
    """
    adm = MemoryAdmission(Path(run_dir) if run_dir else Path("."), config)
    mem = memory if memory is not None else system_memory()
    est = None
    if run_dir is not None and (Path(run_dir) / "trials").is_dir():
        # v2.19.0:有待跑 trial 时按它们各自的校准估计(同类实测 ×1.25)取
        # 最大值,而不是"最近 3 个实测的最大峰值 ×1.15"(混入 gen-00 冷缓存
        # 高峰时把整批并发压到 4~5 路)
        est = _pending_trials_peak_gb(Path(run_dir), config)
        if est is None:
            est = recent_trials_peak_gb(Path(run_dir))
    if est is None:
        est = estimate_trial_peak_gb(Path(run_dir) if run_dir else Path("."),
                                     {}, config)
    res = adm.reservations(reclaim=False) if run_dir is not None else []
    outstanding = round(sum(r["outstanding_gb"] for r in res), 2)
    avail = mem.get("available_gb")
    per = float(est["estimate_gb"])
    if avail is None:
        suggested = None
    elif per <= SMALL_TRIAL_GB:
        suggested = max_parallel
    else:
        suggested = max(0, int((float(avail) - outstanding - adm.margin_gb)
                               // per))
        if max_parallel is not None:
            suggested = min(suggested, int(max_parallel))
    return {
        "enabled": adm.enabled, "margin_gb": adm.margin_gb,
        "per_trial_estimate_gb": est.get("estimate_gb"),
        "estimate_source": est.get("source"),
        "basis_trials": est.get("basis_trials")
        or ([est["basis_trial"]] if est.get("basis_trial") else []),
        "available_gb": avail, "outstanding_gb": outstanding,
        "n_reserved": len(res), "suggested_parallel": suggested,
        "cpu_parallel": max_parallel,
        "memory_bound": (suggested is not None and max_parallel is not None
                         and suggested < int(max_parallel)),
        "note": ("suggested_parallel = floor((available − outstanding − "
                 "margin) / per_trial_estimate),封顶 CPU 口径并发;"
                 "run-generation / run --detach 启动训练前按同一口径逐个准入"
                 "(排队写 logs/admission.jsonl)"),
    }


def gpu_occupancy_view(run_dir: Path | None) -> dict:
    """snapshot 用:各卡显存/利用率与外来进程(查询失败给出原因)。"""
    occ = query_gpu_occupancy()
    if occ is None:
        return {"available": False,
                "note": "nvidia-smi 查询失败或无 GPU:选卡回退旧逻辑"}
    cards = annotate_gpu_occupancy(occ, Path(run_dir) if run_dir else None)
    clean, busy = gpu_candidates_by_occupancy(cards)
    return {"available": True,
            "cards": [{k: g.get(k) for k in (
                "index", "memory_used_gb", "memory_free_gb", "util_pct",
                "own_procs", "foreign_procs", "leased_by_run", "clean")}
                for g in cards],
            "clean_order": clean, "occupied_order": busy}
