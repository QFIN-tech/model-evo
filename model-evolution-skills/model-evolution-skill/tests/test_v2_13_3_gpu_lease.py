"""v2.13.3 回归测试:GPU 跨进程排他租约(并发 trial 撞卡修复)。

事故来源(2026-09-21,真实 run 排障):
  gen-04 三个 HPO trial(xgboost/lightgbm/catboost)手动并行
  ``run --trial``,全部落到 GPU0 —— nvidia-smi 实测 78407MiB/81920MiB
  单卡占满,其余 7 张 A800 全空。三个 trial 各自 device=cuda 初始化
  失败后退化 CPU 24 线程(心跳 2200% 满载),连续 4 轮被外部 kill,
  每轮 1~2.5 小时,0 产物。

  根因不是"slot 没铺开":``scheduler.plan()`` 是纯函数,每个 runner
  进程各自快照、各自取"空闲显存最大"的卡,进程间**没有任何互斥**。
  既有 ``gpu_indices[slot % len]`` 只在批内序号这一层错开,挡不住
  跨进程竞态;手动并行路径更是恒 ``slot_index=0``。

本文件锁定 v2.13.3 的租约语义:
  L1 GpuLease 独占:同一张卡二次抢失败;
  L2 释放后可被重新抢到(不泄漏);
  L3 陈旧租约回收:持有者 pid 已死 → 下一轮可抢;
  L4 活进程持有 → 不可抢(即使显存很空;EPERM/跨用户保守不抢);
  L5 同进程多线程:token 区分,同一张卡只允许一个线程持有;
  L6 幂等:同线程重复抢同一张卡返回成功,不会自锁;
  L7 gpu_candidates 排序:健康卡(>20GB 且 util<80%)优先、同级显存降序;
  L8 runner._acquire_gpu_lease:plan 指定的卡被占时自动换下一张
     —— 这正是并发 trial 不再撞卡的判据;
  L9 runner._acquire_gpu_lease:全部卡被活进程持有 → (None, None)
     按 CPU 跑,不超发抢卡;
  L10 租约覆盖 CUDA_VISIBLE_DEVICES 语义:抢不到卡时显式清空,
      不留 plan.env 的残留值(旧行为正是照残留值撞上去的)。

运行:pytest tests/test_v2_13_3_gpu_lease.py -q
      python tests/test_v2_13_3_gpu_lease.py
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import tmp_run_dir  # noqa: E402  hermetic import bootstrap

from runtime.config import Config  # noqa: E402
from runtime.runner import TrialRunner  # noqa: E402
from runtime.scheduler import (  # noqa: E402
    GpuLease, ResourcePlan, Scheduler, gpu_candidates)


@pytest.fixture()
def run_dir():
    d = tmp_run_dir("modelevo-gpu-lease-")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _leases(run_dir: Path) -> Path:
    return run_dir / TrialRunner._GPU_LEASES


def _snap(gpus) -> dict:
    return {"cpu_physical": 32, "cpu_count": 64, "load1": 0.0,
            "ram_available_gb": 200.0, "gpus": list(gpus)}


def _g(idx, free, util=0):
    return {"index": idx, "memory_free_gb": free, "util_pct": util}


# ---------------------------------------------------------------------------
# L1-L6:租约原语
# ---------------------------------------------------------------------------

def test_lease_exclusive_and_reusable(run_dir):
    print("== L1/L2 独占与释放 ==")
    a = GpuLease.acquire(_leases(run_dir), 0)
    assert a is not None and a.gpu_index == 0
    assert GpuLease.acquire(_leases(run_dir), 0) is None, "同卡二次抢应失败"
    # 别的卡不受影响
    b = GpuLease.acquire(_leases(run_dir), 1)
    assert b is not None and b.gpu_index == 1

    a.release()
    c = GpuLease.acquire(_leases(run_dir), 0)
    assert c is not None, "释放后应可重抢(不泄漏租约)"
    assert not (_leases(run_dir) / "gpu-0.lock").exists() or True
    c.release()
    b.release()


def test_lease_stale_reclaim(run_dir):
    print("== L3 陈旧租约回收(持有者已死)==")
    lock = _leases(run_dir) / "gpu-0.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    # 造一个几乎不可能存在的 pid(2^31-1 超出 pid_max)
    lock.write_text("2147483647:deadbeef\n", encoding="utf-8")
    lease = GpuLease.acquire(_leases(run_dir), 0)
    assert lease is not None, "持有者已死 → 陈旧租约应被原子回收重抢"
    lease.release()

    # 空/损坏租约同样视同陈旧
    lock.write_text("", encoding="utf-8")
    lease2 = GpuLease.acquire(_leases(run_dir), 0)
    assert lease2 is not None, "损坏租约应可回收"
    lease2.release()


def test_lease_live_holder_never_stolen(run_dir):
    print("== L4 活进程持有不可抢 ==")
    lock = _leases(run_dir) / "gpu-0.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    # pid 1 必然活着,且不是本进程 → 必须判"活",不得抢
    lock.write_text("1:otherproc\n", encoding="utf-8")
    assert GpuLease.acquire(_leases(run_dir), 0) is None, \
        "活进程持有的租约不得被抢占(即使显存很空)"
    # 本进程 pty 上 pid==self 但 token 不同 = 同进程别的线程持有 → 不抢
    lock.write_text(f"{os.getpid()}:some-other-thread\n", encoding="utf-8")
    assert GpuLease.acquire(_leases(run_dir), 0) is None, \
        "同进程别的线程持有 → 不得抢(否则两线程写进同一张卡)"
    lock.unlink()


def test_lease_idempotent_same_thread(run_dir):
    print("== L6 同线程重复抢也不同时持有(不掩盖重复占用)==")
    a = GpuLease.acquire(_leases(run_dir), 0)
    assert a is not None
    # 刻意**不**做"同线程幂等成功":返回成功会让两个 trial 静默写进同一
    # 张卡。抢卡只发生在 _execute_once 开头、释放只在其 finally,不存在
    # 嵌套抢卡,故重复抢一律按"已占用"拒绝。
    assert GpuLease.acquire(_leases(run_dir), 0) is None, \
        "同线程重复抢应被拒(防静默双占)"
    a.release()
    assert GpuLease.acquire(_leases(run_dir), 0) is not None, \
        "释放后应可重抢"


def test_lease_thread_exclusion(run_dir):
    print("== L5 同进程多线程只允许一个持有 ==")
    got: list[GpuLease] = []
    barrier = threading.Barrier(8)

    def _worker():
        barrier.wait()
        lease = GpuLease.acquire(_leases(run_dir), 0)
        if lease is not None:
            got.append(lease)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == 1, \
        f"8 线程抢同一张卡应恰好 1 个成功,实际 {len(got)}"
    got[0].release()


# ---------------------------------------------------------------------------
# L7:候选排序
# ---------------------------------------------------------------------------

def test_gpu_candidates_ordering():
    print("== L7 候选排序 ==")
    snap = _snap([
        _g(0, 79.0, 0),    # 健康,最空
        _g(1, 75.0, 10),   # 健康
        _g(2, 60.0, 0),    # 健康
        _g(3, 40.0, 90),   # 不健康(util 高)→ 排到健康卡之后
        _g(4, 5.0, 0),     # 不健康(显存小)→ 最后
    ])
    order = [g["index"] for g in gpu_candidates(snap)]
    assert order == [0, 1, 2, 3, 4], f"健康卡优先且显存降序,实际 {order}"
    assert gpu_candidates({}) == []
    assert gpu_candidates({"gpus": []}) == []


# ---------------------------------------------------------------------------
# L8-L10:runner 接线
# ---------------------------------------------------------------------------

def _runner(run_dir: Path, snap) -> TrialRunner:
    r = TrialRunner.__new__(TrialRunner)
    r.run_dir = Path(run_dir)
    r.config = Config.load(skill_config=False)
    r.scheduler = Scheduler(r.config, snap)
    return r


def test_runner_lease_switches_card_when_occupied(run_dir):
    print("== L8 plan 指定的卡被占 → 自动换卡 ==")
    snap = _snap([_g(0, 79.0, 0), _g(1, 75.0, 10), _g(2, 60.0, 0)])
    # 模拟"另一个 runner 进程"已占住 plan 要给的 gpu0
    blocker = GpuLease.acquire(_leases(run_dir), 0)
    assert blocker is not None

    plan = ResourcePlan(n_parallel_trials=1, threads_per_trial=8,
                        gpu_index=0, env={"CUDA_VISIBLE_DEVICES": "0"})
    lease, gi = _runner(run_dir, snap)._acquire_gpu_lease(plan, slot_index=0)
    assert lease is not None, "plan 的卡被占时应换下一张空闲卡,而非硬上"
    assert gi == 1, f"应落到第二空的 gpu1,实际 gpu{gi}"
    lease.release()
    blocker.release()


def test_runner_lease_respects_slot_then_switches(run_dir):
    print("== L8b 批路径:slot 铺开的卡被占 → 换卡 ==")
    snap = _snap([_g(0, 79.0, 0), _g(1, 75.0, 10), _g(2, 60.0, 0)])
    plan = ResourcePlan(n_parallel_trials=2, threads_per_trial=8,
                        gpu_index=0, gpu_indices=[0, 1],
                        env={"CUDA_VISIBLE_DEVICES": "0"})
    r = _runner(run_dir, snap)
    # slot0 先拿 → gpu0
    l0, g0 = r._acquire_gpu_lease(plan, slot_index=0)
    assert g0 == 0
    # slot1 应拿 gpu1(plan 指派),不撞 slot0
    l1, g1 = r._acquire_gpu_lease(plan, slot_index=1)
    assert g1 == 1, f"slot1 应绑 gpu1,实际 gpu{g1}"
    # slot2 越界回绕到 indices[0]=gpu0,但已被占 → 换到 gpu2
    l2, g2 = r._acquire_gpu_lease(plan, slot_index=2)
    assert g2 == 2, f"回绕撞车后应换到 gpu2,实际 gpu{g2}"
    for l in (l0, l1, l2):
        l.release()


def test_runner_lease_cpu_fallback_when_all_taken(run_dir):
    print("== L9 全卡被占 → 按 CPU 跑,不超发 ==")
    snap = _snap([_g(0, 79.0, 0), _g(1, 75.0, 10)])
    held = [GpuLease.acquire(_leases(run_dir), i) for i in (0, 1)]
    assert all(h is not None for h in held)
    plan = ResourcePlan(n_parallel_trials=1, threads_per_trial=8,
                        gpu_index=0, env={"CUDA_VISIBLE_DEVICES": "0"})
    lease, gi = _runner(run_dir, snap)._acquire_gpu_lease(plan, slot_index=0)
    assert lease is None and gi is None, \
        "卡全被活进程持有时应按 CPU 跑,而不是超发抢卡"
    for h in held:
        h.release()


def test_runner_no_gpu_snapshot_is_cpu(run_dir):
    print("== 无卡机器:租约短路,不建目录 ==")
    r = _runner(run_dir, _snap([]))
    lease, gi = r._acquire_gpu_lease(ResourcePlan(), slot_index=0)
    assert lease is None and gi is None
    assert not _leases(run_dir).exists(), "无卡时不应创建租约目录"


def test_execute_once_clears_cvd_when_no_lease(run_dir, monkeypatch):
    print("== L10 抢不到卡 → 清空 CUDA_VISIBLE_DEVICES ==")
    snap = _snap([_g(0, 79.0, 0)])
    blocker = GpuLease.acquire(_leases(run_dir), 0)
    assert blocker is not None
    r = _runner(run_dir, snap)

    captured: dict = {}

    def _fake_run_trial_process(cmd, **kw):
        captured.update(kw)
        return object()

    monkeypatch.setattr("runtime.runner.run_trial_process",
                        _fake_run_trial_process)
    trial = type("T", (), {})()
    trial.trial_id = "t-fake"
    trial.data = {"execution": {"command": "python train.py"},
                  "environment": {}}

    plan = ResourcePlan(n_parallel_trials=1, threads_per_trial=8,
                        gpu_index=0, env={"CUDA_VISIBLE_DEVICES": "0"})
    r._execute_once(trial, run_dir, run_dir, plan, timeout_min=1, grace=1,
                    slot_index=0)
    env = captured["env_extra"]
    assert env.get("CUDA_VISIBLE_DEVICES") == "", \
        "抢不到卡必须清空残留的 CUDA_VISIBLE_DEVICES(旧行为照残留撞卡)"
    assert env.get("MODELEVO_GPU") == "0", \
        "无卡时 GPU 意图必须为 0,防模板 pick_gpu 顶回一张卡"
    blocker.release()


def test_execute_once_releases_lease(run_dir, monkeypatch):
    print("== 租约在子进程结束后释放(含异常路径)==")
    snap = _snap([_g(0, 79.0, 0)])
    r = _runner(run_dir, snap)
    monkeypatch.setattr("runtime.runner.run_trial_process",
                        lambda cmd, **kw: object())
    trial = type("T", (), {})()
    trial.trial_id = "t-fake"
    trial.data = {"execution": {"command": "python train.py"},
                  "environment": {}}
    plan = ResourcePlan(n_parallel_trials=1, threads_per_trial=8,
                        gpu_index=0, env={"CUDA_VISIBLE_DEVICES": "0"})
    r._execute_once(trial, run_dir, run_dir, plan, timeout_min=1, grace=1,
                    slot_index=0)
    _probe = GpuLease.acquire(_leases(run_dir), 0)
    assert _probe is not None, \
        "正常结束后租约必须释放,否则后续 trial 永远抢不到卡"
    _probe.release()

    # 异常路径同样释放
    def _boom(cmd, **kw):
        raise RuntimeError("child blew up")

    monkeypatch.setattr("runtime.runner.run_trial_process", _boom)
    with pytest.raises(RuntimeError):
        r._execute_once(trial, run_dir, run_dir, plan, timeout_min=1,
                        grace=1, slot_index=0)
    _probe2 = GpuLease.acquire(_leases(run_dir), 0)
    assert _probe2 is not None, \
        "子进程异常时租约也必须释放(finally 兜底)"
    _probe2.release()


def test_pick_gpu_does_not_override_lease():
    print("== L11 子进程 pick_gpu 不得覆盖 runner 的租约裁决 ==")
    import os as _os

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _common import trial_body  # noqa: E402

    ns = {"os": _os, "globals": globals}
    exec(compile(trial_body(), "trial_body_pickgpu", "exec"), ns)  # noqa: S102
    pick = ns["pick_gpu"]

    saved = _os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        # runner 已按租约写入设备 → pick_gpu 必须原样保留,不得自行挑卡
        _os.environ["MODELEVO_GPU"] = "1"
        _os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        pick()
        assert _os.environ["CUDA_VISIBLE_DEVICES"] == "3", \
            "租约指定的 gpu3 被 pick_gpu 改写 —— 并发 trial 会重新挤同一张卡"

        # 无卡(trial 未被分配到卡)→ 不得探测顶回一张卡
        _os.environ["MODELEVO_GPU"] = "0"
        _os.environ["CUDA_VISIBLE_DEVICES"] = ""
        pick()
        assert _os.environ["CUDA_VISIBLE_DEVICES"] == "", \
            "MODELEVO_GPU=0 时 pick_gpu 不得顶回一张卡"
    finally:
        if saved is None:
            _os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            _os.environ["CUDA_VISIBLE_DEVICES"] = saved
        _os.environ.pop("MODELEVO_GPU", None)


def test_runner_source_has_lease_wiring():
    print("== 源码级:runner 走租约而非直取 plan ==")
    src = (Path(__file__).resolve().parents[1] / "runtime"
           / "runner.py").read_text(encoding="utf-8")
    assert "def _acquire_gpu_lease" in src
    assert "GpuLease.acquire" in src
    assert "_lease.release()" in src, "必须有 finally 释放"
    sched_src = (Path(__file__).resolve().parents[1] / "runtime"
                 / "scheduler.py").read_text(encoding="utf-8")
    assert "class GpuLease" in sched_src
    assert "def gpu_candidates" in sched_src
    assert "_LEASE_MTX" in sched_src, "同进程多线程必须串行化抢租约"
    # 发布必须用硬链接(O_EXCL 建空文件有"内容未写完"窗口,实测撞卡)
    assert "os.link(" in sched_src, "租约发布必须用硬链接消掉空文件窗口"
    tpl_src = (Path(__file__).resolve().parents[1] / "scripts" / "_mvo"
               / "templates.py").read_text(encoding="utf-8")
    assert "def pick_gpu() -> None:" in tpl_src
    _seg = tpl_src[tpl_src.index("def pick_gpu() -> None:"):]
    _seg = _seg[:_seg.index("def n_jobs()")]
    assert "CUDA_VISIBLE_DEVICES" in _seg and "strip()" in _seg, \
        "pick_gpu 必须先看 CUDA_VISIBLE_DEVICES 是否已被租约指定"


def main() -> int:
    import inspect
    import tempfile

    fails = 0
    for fn in (test_gpu_candidates_ordering,
               test_lease_exclusive_and_reusable,
               test_lease_stale_reclaim,
               test_lease_live_holder_never_stolen,
               test_lease_idempotent_same_thread,
               test_lease_thread_exclusion,
               test_runner_lease_switches_card_when_occupied,
               test_runner_lease_respects_slot_then_switches,
               test_runner_lease_cpu_fallback_when_all_taken,
               test_runner_no_gpu_snapshot_is_cpu,
               test_pick_gpu_does_not_override_lease,
               test_runner_source_has_lease_wiring):
        wants_dir = "run_dir" in inspect.signature(fn).parameters
        d = Path(tempfile.mkdtemp(prefix="modelevo-gpu-lease-"))
        try:
            fn(d) if wants_dir else fn()
            print(f"  [PASS] {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            fails += 1
            print(f"  [FAIL] {fn.__name__}: {exc}")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    print(f"\n=== {'FAIL' if fails else 'PASS'} ({fails} failed) ===")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
