"""v2.13.3 回归测试:GPU 能力门控(机器有卡 ≠ 本 trial 能用卡)。

事故来源(2026-09-22,run-req-t7-trigger gen-03,真实 run 排障):
  ``env-manifest.json`` 在 08:51 就探明 ``gbm_backends.lightgbm_cuda=
  false``(lightgbm PyPI wheel 无 CUDA 编译),但该事实**只写进清单、
  没有任何消费方**(grep 全仓仅命中 preflight 自己)。于是 gen-03 六路
  并发的 lightgbm trial 各领一张 A800 的排他租约,而每个 ``train.py``
  的 stdout 都写着"lightgbm wheel 非 CUDA 编译,回退 CPU"。

  账:trial-013 租约 12:07:40→12:15:05(445s),其中 ``train start→
  done`` 仅 296.7s→355.2s(58.5s),其余 ~387s 是纯 CPU 的 g01 parse +
  widen —— 租约期内 GPU 计算量恒为 0,7 张卡被空锁 ~7 分钟。

本文件锁定能力门控语义:
  C1 manifest 可判 lightgbm 无 CUDA 后端 → 不取卡(reason 非空);
  C2 xgboost(同 manifest 里 cuda=true)→ 照常取卡;
  C3 键缺失 / manifest 缺失 / 损坏 → fail-open 取卡(过取卡只是浪费,
     误判"不能用卡"会让能 GPU 加速的 trial 静默退回 CPU,更难发现);
  C4 无编译探测的算法族(catboost)→ 放行,不由本模块裁决;
  C5 runner._execute_once:门控不通过时**不建租约文件**、子进程
     CUDA_VISIBLE_DEVICES 清空、MODELEVO_GPU="0";
  C6 runner._execute_once:门控通过且卡空闲 → 照常拿卡(不误伤)。
  C7 trial.log 的 "开始执行" 不再把 plan.gpu_index 冒充本 trial 卡号
     (gen-03 六条全是 gpu=0 的误导日志)。

运行:pytest tests/test_v2_13_3_gpu_capability.py -q
      python tests/test_v2_13_3_gpu_capability.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import tmp_run_dir  # noqa: E402  hermetic import bootstrap

from runtime.config import Config  # noqa: E402
from runtime.gpu_capability import (  # noqa: E402
    capability_from_manifest, trial_can_use_gpu)
from runtime.runner import TrialRunner  # noqa: E402
from runtime.scheduler import GpuLease, ResourcePlan, Scheduler  # noqa: E402

#: 本 run 的真实快照(2026-09-22 08:51,run-req-t7-trigger/env-manifest.json)
_LGBM_NO_CUDA = {
    "lightgbm_version": "4.6.0",
    "lightgbm_cuda": False,
    "xgboost_version": "2.1.4",
    "xgboost_cuda": True,
}


@pytest.fixture()
def run_dir():
    d = tmp_run_dir("modelevo-gpu-cap-")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _write_manifest(run_dir: Path, gbm) -> None:
    (Path(run_dir) / "env-manifest.json").write_text(
        json.dumps({"packages": [], "gbm_backends": gbm}),
        encoding="utf-8")


def _snap(gpus) -> dict:
    return {"cpu_physical": 32, "cpu_count": 64, "load1": 0.0,
            "ram_available_gb": 200.0, "gpus": list(gpus)}


def _g(idx, free, util=0):
    return {"index": idx, "memory_free_gb": free, "util_pct": util}


def _runner(run_dir: Path, snap, capability=None) -> TrialRunner:
    r = TrialRunner.__new__(TrialRunner)
    r.run_dir = Path(run_dir)
    r.config = Config.load(skill_config=False)
    r.scheduler = Scheduler(r.config, snap)
    r._capability = capability
    return r


# ---------------------------------------------------------------------------
# C1-C4:纯函数裁决
# ---------------------------------------------------------------------------

def test_gate_blocks_lightgbm_without_cuda():
    print("== C1 lightgbm 无 CUDA 后端 → 不取卡 ==")
    can, why = trial_can_use_gpu({"algorithm": "lightgbm"}, _LGBM_NO_CUDA)
    assert can is False, "lightgbm_cuda=false 时不该取卡(租约期内 0 GPU 计算)"
    assert why, "拒绝必须带可归因的 reason(旧行为缺的正是这条线索)"
    print("   reason:", why)


def test_gate_allows_xgboost_with_cuda():
    print("== C2 xgboost 有 CUDA 后端 → 照常取卡 ==")
    can, why = trial_can_use_gpu({"algorithm": "xgboost"}, _LGBM_NO_CUDA)
    assert can is True and why == ""


def test_gate_fails_open_on_missing_evidence():
    print("== C3 证据缺失 → fail-open 放行 ==")
    for algo in ("lightgbm", "xgboost"):
        assert trial_can_use_gpu({"algorithm": algo}, None)[0] is True, \
            "manifest 缺失不得阻止取卡"
        assert trial_can_use_gpu({"algorithm": algo}, {})[0] is True, \
            "gbm_backends 键缺失(旧 manifest)不得阻止取卡"
        assert trial_can_use_gpu({"algorithm": algo},
                                 {"lightgbm_cuda": None})[0] is True
    # 算法缺失/为空同样放行(判据不足时不裁决)
    assert trial_can_use_gpu({}, _LGBM_NO_CUDA)[0] is True
    assert trial_can_use_gpu({"algorithm": ""}, _LGBM_NO_CUDA)[0] is True
    assert trial_can_use_gpu(None, _LGBM_NO_CUDA)[0] is True


def test_gate_abstains_on_probe_less_algorithms():
    print("== C4 无编译探测的族(catboost)→ 放行,不由本模块裁决 ==")
    assert trial_can_use_gpu({"algorithm": "catboost"},
                             _LGBM_NO_CUDA)[0] is True


def test_capability_from_manifest_variants():
    print("== capability_from_manifest 容错 ==")
    assert capability_from_manifest({"gbm_backends": _LGBM_NO_CUDA}) \
        == _LGBM_NO_CUDA
    assert capability_from_manifest({}) is None
    assert capability_from_manifest(None) is None
    assert capability_from_manifest({"gbm_backends": "oops"}) is None


def test_gate_loads_manifest_from_disk(run_dir):
    print("== C3b run_dir 给定时自行读盘(含 run-state 内嵌副本)==")
    _write_manifest(run_dir, _LGBM_NO_CUDA)
    can, why = trial_can_use_gpu({"algorithm": "lightgbm"}, None,
                                 run_dir=run_dir)
    assert can is False, "env-manifest.json 存在时应据其裁决"
    assert why

    # 只留 run-state 内嵌副本(planner 读的也是这份)
    (Path(run_dir) / "env-manifest.json").unlink()
    (Path(run_dir) / "run-state.json").write_text(
        json.dumps({"env_manifest": {"gbm_backends": _LGBM_NO_CUDA}}),
        encoding="utf-8")
    can2, _ = trial_can_use_gpu({"algorithm": "lightgbm"}, None,
                                run_dir=run_dir)
    assert can2 is False

    # 损坏的 manifest:不得抛异常,且 fail-open
    (Path(run_dir) / "run-state.json").write_text("{not json",
                                                  encoding="utf-8")
    assert trial_can_use_gpu({"algorithm": "lightgbm"}, None,
                             run_dir=run_dir)[0] is True


# ---------------------------------------------------------------------------
# C5-C6:runner 接线
# ---------------------------------------------------------------------------

def _trial(algo: str):
    t = type("T", (), {})()
    t.trial_id = f"t-{algo}"
    t.data = {"algorithm": algo,
              "execution": {"command": "python train.py"},
              "environment": {}}
    return t


def _exec(r, run_dir, trial, monkeypatch, captured=None):
    def _fake(cmd, **kw):
        if captured is not None:
            captured.update(kw)
        return object()

    monkeypatch.setattr("runtime.runner.run_trial_process", _fake)
    plan = ResourcePlan(n_parallel_trials=1, threads_per_trial=8,
                        gpu_index=0, env={"CUDA_VISIBLE_DEVICES": "0"})
    return r._execute_once(trial, run_dir, run_dir, plan, timeout_min=1,
                           grace=1, slot_index=0)


def test_execute_once_skips_lease_for_cpu_only_algorithm(run_dir, monkeypatch):
    print("== C5 门控不通过:不建租约 + 清空 CVD + MODELEVO_GPU=0 ==")
    _write_manifest(run_dir, _LGBM_NO_CUDA)
    snap = _snap([_g(0, 79.0, 0), _g(1, 75.0, 0)])
    r = _runner(run_dir, snap)

    captured: dict = {}
    _exec(r, run_dir, _trial("lightgbm"), monkeypatch, captured)

    env = captured["env_extra"]
    assert env.get("CUDA_VISIBLE_DEVICES") == "", \
        "能力门控拒绝后必须清空 CUDA_VISIBLE_DEVICES(否则 plan 残留值照撞)"
    assert env.get("MODELEVO_GPU") == "0", \
        "CPU-only 算法置 GPU 意图=1 会让模板走 device=cuda 再回退,白留 stderr 噪音"
    assert not (Path(run_dir) / TrialRunner._GPU_LEASES).exists(), \
        "不取卡就不该建租约目录/文件(gen-03 正是空锁了 7 张卡)"


def test_execute_once_still_leases_when_capable(run_dir, monkeypatch):
    print("== C6 门控通过且卡空闲 → 照常拿卡(不误伤)==")
    _write_manifest(run_dir, _LGBM_NO_CUDA)  # xgboost_cuda=true
    snap = _snap([_g(0, 79.0, 0)])
    r = _runner(run_dir, snap)

    captured: dict = {}
    _exec(r, run_dir, _trial("xgboost"), monkeypatch, captured)

    assert captured["env_extra"].get("CUDA_VISIBLE_DEVICES") == "0", \
        "有 CUDA 后端的算法族必须照常绑卡"
    assert captured["env_extra"].get("MODELEVO_GPU") == "1"
    probe = GpuLease.acquire(Path(run_dir) / TrialRunner._GPU_LEASES, 0)
    assert probe is not None, "xgboost 的租约应在子进程结束后释放"
    probe.release()


def test_execute_once_leases_when_manifest_absent(run_dir, monkeypatch):
    print("== C6b 无 manifest(旧 run / 探测失败)→ fail-open 仍取卡 ==")
    snap = _snap([_g(0, 79.0, 0)])
    r = _runner(run_dir, snap)
    captured: dict = {}
    _exec(r, run_dir, _trial("lightgbm"), monkeypatch, captured)
    assert captured["env_extra"].get("CUDA_VISIBLE_DEVICES") == "0", \
        "证据缺失时不得阻止取卡(fail-open)"


def test_start_log_does_not_claim_plan_gpu(run_dir, monkeypatch):
    print("== C7 开始执行日志不再把 plan.gpu_index 冒充本 trial 卡号 ==")
    src = (Path(__file__).resolve().parents[1] / "runtime" / "runner.py")
    body = src.read_text(encoding="utf-8")

    assert '"开始执行: cmd=%s timeout=%.1fmin threads=%d"' in body, \
        "开始执行日志应只报 command/timeout/threads(批次级 plan 值不该混进来)"
    assert '开始执行: cmd=%s timeout=%.1fmin threads=%d gpu=%s' not in body, \
        ("旧格式把 plan.gpu_index 当本 trial 卡号打印——plan 是批次级快照,"
         "并发批里每个 trial 都印同一张卡(gen-03 六条 gpu=0 的误导日志成因)")

    # 行为面:真实归属必须落在租约日志 + trial.yaml 的 environment.gpu_index
    _write_manifest(run_dir, _LGBM_NO_CUDA)
    snap = _snap([_g(0, 79.0, 0), _g(1, 75.0, 0)])
    r = _runner(run_dir, snap)
    blocker = GpuLease.acquire(Path(run_dir) / TrialRunner._GPU_LEASES, 0)
    assert blocker is not None  # 占住 plan 指定的 gpu0

    trial = _trial("xgboost")
    captured: dict = {}
    _exec(r, run_dir, trial, monkeypatch, captured)
    blocker.release()

    assert trial.data["environment"]["gpu_index"] == 1, \
        "plan 指定的 gpu0 被占 → 真实归属应为 gpu1,并写回 trial 记录"
    assert captured["env_extra"]["CUDA_VISIBLE_DEVICES"] == "1"
