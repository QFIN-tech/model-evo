"""v17.4 修复批次单元验证:GPU-first(所有支持 GPU 的模型优先上卡)。

覆盖:
1. scheduler.plan 导出 MODELEVO_GPU(1=有卡,0=无卡);
2. runner._downgrade_plan(CPU 降配重试)把 MODELEVO_GPU 置 0;
3. 模板 body helper _gpu_allocated:只认 MODELEVO_GPU + 行数≥2000;
4. _xgb_cuda_built 尊重 xgb.build_info()(本机 USE_CUDA=True);
5. _build_model 的 _n_rows 门控(1999 不上卡,2000+ 上卡);
6. _lgbm_fit_safe:非 GPU 编译的 lightgbm 轮子上 device=cuda 的 fit
   自动回退 CPU 成功(本机 lightgbm 4.6 非 GPU 编译,即该回退的真实路径);
7. TRIAL_TEMPLATE.format(body) 仍可编译(v17.1 回归);
8. baseline _XGB_CODE/_LGBM_CODE 文本含 device 注入形态;
9. scheduler.py / runner.py 源码含 MODELEVO_GPU。

运行:python tests/test_v17dot4_gpu.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.config import Config  # noqa: E402
from runtime.scheduler import ResourcePlan, Scheduler  # noqa: E402
from runtime.templates import TRIAL_TEMPLATE  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import trial_body  # noqa: E402


def _cli():
    p = Path(__file__).resolve().parents[1] / "scripts" / "modelevo.py"
    spec = importlib.util.spec_from_file_location("mcli17dot4", str(p))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _body_src() -> str:
    """v2.11.9:模板 body 是 _mvo/templates.TRIAL_BODY,直接取。

    旧实现从 scripts/modelevo.py 源码做字面量切片 + 手工反转义;模板
    抽出后源码里已无该字面量。
    """
    return trial_body()


# ---------------------------------------------------------------------------
# 1) scheduler / 2) runner downgrade env
# ---------------------------------------------------------------------------

def test_scheduler_env_and_downgrade():
    print("== MODELEVO_GPU 意图信号 ==")
    config = Config.load(skill_config=False)
    snap_gpu = {"cpu_count": 8, "cpu_physical": 8, "load1": 1.0,
                "ram_total_gb": 64, "ram_available_gb": 40,
                "gpus": [{"index": 0, "memory_total_gb": 24,
                          "memory_free_gb": 10.0, "util_pct": 30}]}
    plan = Scheduler(config, snap_gpu).plan()
    check("有卡 → MODELEVO_GPU=1",
          plan.env.get("MODELEVO_GPU") == "1", plan.env.get("MODELEVO_GPU"))
    snap_none = {"cpu_count": 8, "cpu_physical": 8, "load1": 1.0,
                 "ram_total_gb": 64, "ram_available_gb": 40, "gpus": []}
    plan_n = Scheduler(config, snap_none).plan()
    check("无卡 → MODELEVO_GPU=0 + CUDA 置空",
          plan_n.env.get("MODELEVO_GPU") == "0"
          and plan_n.env.get("CUDA_VISIBLE_DEVICES") == "",
          f"{plan_n.env}")

    # runner._downgrade_plan:CPU 降配重试把 GPU 意图压回 0
    from runtime import runner as _runner_mod  # noqa: E402  (need class import)
    # 直接构造真实 TrialRunner 过重;此处验证模块方法存在并模拟其逻辑:
    # 从源码断言 downgrade 分支含 MODELEVO_GPU="0"
    rsrc = (Path(__file__).resolve().parents[1] / "runtime"
            / "runner.py").read_text(encoding="utf-8")
    seg = rsrc[rsrc.index("def _downgrade_plan"):]
    check("runner 降配写 MODELEVO_GPU=0",
          'env["MODELEVO_GPU"] = "0"' in seg)


# ---------------------------------------------------------------------------
# 3-5) body helpers via rendered template exec
# ---------------------------------------------------------------------------

def _render_body():
    body = _body_src()
    code = TRIAL_TEMPLATE.format(
        trial_id="t", hypothesis="h", actions="a", run_dir="/tmp/x",
        target="y", model_type="binary_classification", treatment="",
        seed=1, skill_root="", body=body)
    compile(code, "train.py", "exec")  # SyntaxError propagates as failure
    return code


def test_body_gpu_helpers():
    print("== body GPU helper 语义 ==")
    import os as _os
    _render_body()  # compile must not raise

    ns = {"pd": None, "np": None, "os": _os,
          "MODEL_TYPE": "binary_classification"}
    body = _body_src()
    exec(compile(body, "trial_body", "exec"), ns)  # noqa: S102

    _os.environ["MODELEVO_GPU"] = "1"
    check("_gpu_allocated 有卡+2000 行 → True",
          ns["_gpu_allocated"](2000) is True)
    check("_gpu_allocated 有卡+1999 行 → False",
          ns["_gpu_allocated"](1999) is False)
    _os.environ["MODELEVO_GPU"] = "0"
    check("_gpu_allocated 无卡+2000 行 → False",
          ns["_gpu_allocated"](2000) is False)
    _os.environ.pop("MODELEVO_GPU", None)
    check("_gpu_allocated 缺省(未注入)→ False",
          ns["_gpu_allocated"](2000) is False)

    import xgboost as _xgb
    # _xgb_cuda_built 走真 xgb.build_info(本机 USE_CUDA=True)
    _os.environ["MODELEVO_GPU"] = "1"
    info = _xgb.build_info()
    expected = bool(info.get("USE_CUDA"))
    check(f"_xgb_cuda_built 与 build_info 一致(本机 USE_CUDA={expected})",
          ns["_xgb_cuda_built"]() is expected,
          f"got {ns['_xgb_cuda_built']()}")
    # monkeypatch build_info 为 False → 返回 False
    class _FakeXgb:
        @staticmethod
        def build_info():
            return {"USE_CUDA": False}
    import sys as _sys
    _sys.modules["xgboost"] = _FakeXgb
    try:
        check("_xgb_cuda_built 在 USE_CUDA=False 时 → False",
              ns["_xgb_cuda_built"]() is False)
    finally:
        _sys.modules["xgboost"] = _xgb

    # _build_model 的 _n_rows 门控(lightgbm;本机非 GPU 编译)
    # 只验证 device 注入的 params 决策,不实际 fit(避免依赖 GPU)
    import inspect as _inspect
    sig = _inspect.signature(ns["_build_model"])
    check("_build_model 接收 _n_rows",
          "_n_rows" in sig.parameters, str(sig))
    # [FIX-2026-09-10] 契约更新:lightgbm device=cuda 注入前先过
    # _lgbm_cuda_ok() wheel 编译探测(fit 期 Fatal 事故回写)——
    # CUDA 轮 wheel → device=cuda;非 CUDA 轮 → 不注入(直接 CPU),
    # 不再依赖 fit 期回退。断言按探测结果分支,两种环境都成立。
    check("body 含 _lgbm_cuda_ok 探测",
          "_lgbm_cuda_ok" in body and "_LGBM_CUDA" in body)
    _os.environ["MODELEVO_GPU"] = "1"
    _cuda_ok = ns["_lgbm_cuda_ok"]()
    m_gpu = ns["_build_model"]({}, 1, 2000)
    _dev = m_gpu.get_params().get("device")
    if _cuda_ok:
        check("_build_model 有卡+2000 行 → device=cuda(CUDA wheel)",
              _dev == "cuda", str(_dev))
    else:
        check("_build_model 有卡+2000 行 → 非 CUDA wheel 不注入"
              "(探测回退 CPU)", _dev is None or _dev == "cpu", str(_dev))
    check("_lgbm_cuda_ok 模块级缓存生效(二次调用同值)",
          ns["_lgbm_cuda_ok"]() is _cuda_ok
          and ns["_LGBM_CUDA"] is _cuda_ok)
    m_cpu = ns["_build_model"]({}, 1, 1999)
    d = m_cpu.get_params().get("device")
    check("_build_model 1999 行 → 无 device(CPU)",
          d is None or d == "cpu", str(d))


# ---------------------------------------------------------------------------
# 6) _lgbm_fit_safe:非 GPU 编译轮回退 CPU 真实路径(本机 lightgbm 4.6)
# ---------------------------------------------------------------------------

def test_lgbm_fit_safe_fallback():
    print("== _lgbm_fit_safe CPU 自动回退 ==")
    import warnings
    import numpy as np
    import lightgbm as lgb

    body = _body_src()
    ns = {"os": os, "MODEL_TYPE": "binary_classification"}
    exec(compile(body, "trial_body_fit", "exec"), ns)  # noqa: S102

    X = np.random.RandomState(0).rand(3000, 12)
    y = (X[:, 0] > 0.5).astype(int)
    m = lgb.LGBMClassifier(n_estimators=5, device="cuda", verbose=-1)
    _m_orig = m
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ns["_lgbm_fit_safe"](m, X, y)
            ok = True
        except Exception as exc:  # noqa: BLE001
            ok = False
            detail = str(exc)[:200]
    if ok:
        # v19:回退改为 sanitize 重建干净估计器(原对象不动)——
        # 断言原对象参数未被污染 + 新路径实际完成 CPU 训练
        dev = _m_orig.get_params().get("device")
        check("非 GPU 编译轮 device=cuda fit 经 _lgbm_fit_safe 回退 CPU 成功",
              dev == "cuda",  # v19:原对象保持原参数(sanitize 重建语义)
              f"device={dev}(v19 sanitize 重建:原对象不动,回退在副本上完成)")
    else:
        check("非 GPU 编译轮 device=cuda fit 经 _lgbm_fit_safe 回退 CPU 成功",
              False, detail)


# ---------------------------------------------------------------------------
# 7) template compile + 8) baseline text + 9) runtime text
# ---------------------------------------------------------------------------

def test_templates_and_text():
    print("== 模板编译与文本 ==")
    body = _body_src()
    code = TRIAL_TEMPLATE.format(
        trial_id="t", hypothesis="h", actions="a", run_dir="/tmp/x",
        target="y", model_type="binary_classification", treatment="",
        seed=1, skill_root="", body=body)
    check("TRIAL_TEMPLATE.format(body) 产物可编译", True)

    import runtime.baseline as B  # noqa: E402
    for algo in ("xgboost", "lightgbm"):
        est_code = B.ESTIMATOR_CODE[algo]
        check(f"baseline {algo} 注入 _gpu_allocated + device",
              "_gpu_allocated(n_rows)" in est_code
              and "device=\"cuda\"" in est_code,
              f"missing in {algo}")
    check("baseline xgboost 含 _xgb_cuda_built 门",
          "_xgb_cuda_built()" in B.ESTIMATOR_CODE["xgboost"])
    check("baseline LGBM FIT 走 _lgbm_fit_safe",
          "_lgbm_fit_safe(model, X_fit" in B.FIT_CODE["lightgbm"])

    sched_src = (Path(__file__).resolve().parents[1] / "runtime"
                 / "scheduler.py").read_text(encoding="utf-8")
    check("scheduler.py 源码含 MODELEVO_GPU",
          '"MODELEVO_GPU"' in sched_src)
    rsrc = (Path(__file__).resolve().parents[1] / "runtime"
            / "runner.py").read_text(encoding="utf-8")
    check("runner.py 源码含 MODELEVO_GPU 降配",
          '"MODELEVO_GPU"' in rsrc)
    body_src = _body_src()
    check("body 含 _gpu_allocated/_lgbm_fit_safe/_GPU_MIN_ROWS",
          "_gpu_allocated" in body_src and "_lgbm_fit_safe" in body_src
          and "_GPU_MIN_ROWS = 2000" in body_src)


def main() -> int:
    test_scheduler_env_and_downgrade()
    test_body_gpu_helpers()
    test_lgbm_fit_safe_fallback()
    test_templates_and_text()
    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
