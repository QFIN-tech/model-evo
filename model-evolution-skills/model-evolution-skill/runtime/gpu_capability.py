"""GPU 能力门控(v2.13.3):卡要给能用上卡的 trial 发。

修什么:scheduler/runner 只问"机器有没有卡",不问"本 trial 能不能用卡"。
实测事故(2026-09-22,run-req-t7-trigger gen-03):``env-manifest.json``
在 08:51 就探明 ``gbm_backends.lightgbm_cuda=false``(PyPI wheel 无 CUDA),
但该事实**只写进清单、没有任何消费方**——grep 全仓仅命中 preflight 自己。
于是 6 个 lightgbm trial 各自拿到一张 A800 的排他租约(租约覆盖整个子
进程 445s,其中 GPU 计算量恒为 0;``train.py`` 每条 stdout 都写着
"lightgbm wheel 非 CUDA 编译,回退 CPU"),7 张卡被空锁 ~7 分钟。

能力事实的来源是**已有**的 ``env-manifest.json.gbm_backends``
(与 trial 同解释器探测,见 preflight.probe_gbm_backends),本模块只负责
把它翻译成"这个算法族在本机能不能上卡"。

失效方向(fail-open):快照缺失 / 键缺失 / 读不出来 → 一律**不阻止**取卡
(退回 v2.13.3 之前的行为)。过取卡只是浪费,误判成"不能用卡"会让本来
能 GPU 加速的 trial 静默退回 CPU——后者更难发现。
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["capability_from_manifest", "trial_can_use_gpu"]

#: 算法族 → gbm_backends 里表示"本机该族能上卡"的键。
#: 只登记**有可靠编译探测**的族;探测不到的族(探测一旦误报为
#: False 就会让能上卡的 trial 静默退回 CPU)按 fail-open 放行。
_CAPABILITY_KEYS = {
    "lightgbm": "lightgbm_cuda",
    "xgboost": "xgboost_cuda",
}


def capability_from_manifest(manifest) -> dict | None:
    """从 env-manifest(或其 run-state 内嵌副本)取 GBM 后端 CUDA 编译位。

    返回 ``gbm_backends`` 子字典;没有可用证据时返回 None(调用方
    fail-open 放行取卡)。
    """
    if not isinstance(manifest, dict):
        return None
    gbm = manifest.get("gbm_backends")
    return gbm if isinstance(gbm, dict) else None


def _load_capability(run_dir: Path):
    """读 run 的 GPU 能力快照。读不到返回 None(调用方 fail-open)。

    优先 run 根的 ``env-manifest.json``;退回 ``run-state.json`` 的
    内嵌 ``env_manifest``(两者内容一致,planner 也读的是后者)。
    """
    try:
        run_dir = Path(run_dir)
    except Exception:  # noqa: BLE001
        return None
    for path, extract in (
        (run_dir / "env-manifest.json", lambda d: d),
        (run_dir / "run-state.json",
         lambda d: (d or {}).get("env_manifest") or {}),
    ):
        try:
            if not path.exists():
                continue
            cap = capability_from_manifest(
                extract(json.loads(path.read_text(encoding="utf-8"))))
            if cap is not None:
                return cap
        except Exception:  # noqa: BLE001  快照损坏不阻塞调度
            continue
    return None


def trial_can_use_gpu(trial_data, capability, run_dir: Path | None = None):
    """本 trial 的算法族在本机能否真正上卡。

    返回 ``(can_use: bool, reason: str)``。``reason`` 只在 False 时非空,
    用于日志归因——正是旧行为缺失的那条线索:"为什么这张卡发了又没用"。

    ``capability`` 为 None 且 ``run_dir`` 给定时自行读盘;仍取不到则
    fail-open 返回 True。
    """
    if capability is None and run_dir is not None:
        capability = _load_capability(run_dir)
    if not isinstance(capability, dict):
        return True, ""

    algo = str((trial_data or {}).get("algorithm") or "").strip().lower()
    if not algo:
        return True, ""

    key = _CAPABILITY_KEYS.get(algo)
    if key is None:
        # 无编译探测的族(如 catboost):探测缺失本身不能作为"不能用卡"
        # 的证据,取卡失败由训练期的既有回退兜底(模板 _lgbm_fit_safe)。
        return True, ""

    if capability.get(key) is False:
        return False, f"{algo}({key}=false,本机无 CUDA 后端)"
    # True / 键缺失(旧 manifest)→ fail-open
    return True, ""
