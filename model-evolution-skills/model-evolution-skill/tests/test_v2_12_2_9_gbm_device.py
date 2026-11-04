"""回归测试:v2.12.2.9 GBM 设备口径(xgboost 打分对齐 + lightgbm 探针静音)。

事故来源(2026-09-18,req-cnt-t7 run 排障):
    1. **xgboost 每次打分都触发设备回退**。模型在 GPU 上训练
       (``device=cuda``),但预测走普通 ``model.predict_proba(X)``,X 在
       CPU 内存上 —— 于是每个 trial 的 ``stderr.log`` 都有
       ``Falling back to prediction using DMatrix due to mismatched
       devices``。validation + shadow 两次打分各付一次代价。
    2. **lightgbm 探针噪声伪装成崩溃**。``_lgbm_cuda_ok()`` 每 trial 试
       fit 一次 ``device=cuda``;PyPI 的 lightgbm wheel 不带 CUDA,抛出的
       ``[LightGBM] [Fatal] CUDA Tree Learner was not enabled in this
       build.`` 由 C++ 层**直写 fd 2** —— Python 的
       ``contextlib.redirect_stderr`` 拦不住,落进 trial 的 stderr.log,
       看起来像 trial 崩溃(实测一次排障即被误导)。

    本文件锁定两处修复的**可验证性质**:
      (a) 打分前后预测数值等价(切设备不改分数口径);
      (b) 探针再怎么失败也不往 fd 2 写字节。
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = SKILL_ROOT / "runtime"

xgb = pytest.importorskip("xgboost")


def _u8(result: "subprocess.CompletedProcess") -> dict:
    return {"rc": result.returncode, "out": result.stdout, "err": result.stderr}


# --------------------------------------------------------------------------
# 1. xgboost:切回 CPU 打分不改分数口径
# --------------------------------------------------------------------------

def test_set_param_device_cpu_preserves_predictions():
    """``booster.set_param({'device': 'cpu'})`` 前后预测数值等价。

    实测 xgboost 2.1.4 下最大绝对差 ~6e-08(CPU/GPU 浮点累加顺序),
    ``np.array_equal`` 为 False 但量级远低于任何一对样本的相对次序
    翻转所需 —— AUC 由秩统计得到,不受影响。这里用 1e-6 作硬阈值,
    比实测值宽两个数量级仍能拦住真正的口径变化。
    """
    rs = np.random.RandomState(0)
    X = rs.rand(800, 6).astype("float32")
    y = (rs.rand(800) > 0.5).astype(int)
    m = xgb.XGBClassifier(n_estimators=8, max_depth=3, tree_method="hist",
                          random_state=0, n_jobs=2)
    m.fit(X, y)
    booster = m.get_booster()

    before = m.predict_proba(X)[:, 1]
    booster.set_param({"device": "cpu"})
    after = m.predict_proba(X)[:, 1]

    assert before.shape == after.shape == (800,)
    assert np.abs(before - after).max() < 1e-6


def _load_body_helper(name: str, next_name: str):
    """从 TRIAL_BODY 里取出某个函数定义并 exec 出来。

    ``align_predict_device`` 定义在 ``_mvo.templates.TRIAL_BODY`` 字符串
    里(渲染进 trial 的 ``code/train.py``),模块级不可见 —— 必须 exec
    源码段才能测到真正跑在 trial 里的那份实现。
    """
    sys.path.insert(0, str(SKILL_ROOT))
    sys.path.insert(0, str(SKILL_ROOT / "scripts"))
    from _mvo.templates import TRIAL_BODY

    src = TRIAL_BODY
    i = src.index(f"def {name}")
    j = src.index(f"def {next_name}", i)
    ns: dict = {}
    exec(src[i:j], ns)
    return ns[name]


def _booster_device(m) -> str:
    """读 booster 的真实设备(xgboost 2.1.4 口径)。

    ``booster.attributes()`` 恒为空 dict —— 真实值在 ``save_config()``
    的 ``learner.generic_param.device``。修复实现最初就栽在这里,本
    测试与实现共用同一读法,并在下面单列一条断言锁住该事实。
    """
    import json
    cfg = json.loads(m.get_booster().save_config())
    return str(((cfg.get("learner") or {}).get("generic_param") or {})
               .get("device") or "")


def test_booster_attributes_is_empty_in_xgboost_2():
    """固定事实:attributes() 不是读设备的接口(防止实现回退到错读法)。"""
    rs = np.random.RandomState(3)
    X = rs.rand(120, 4).astype("float32")
    y = (rs.rand(120) > 0.5).astype(int)
    m = xgb.XGBClassifier(n_estimators=2, max_depth=2, tree_method="hist",
                          random_state=0, n_jobs=2)
    m.fit(X, y)
    assert m.get_booster().attributes() == {}
    assert "cpu" in _booster_device(m)


def test_align_helper_resolves_booster_device():
    """``align_predict_device`` 能在打分前把 cuda booster 切回 cpu。"""
    align_predict_device = _load_body_helper(
        "align_predict_device", "_model_score")

    rs = np.random.RandomState(1)
    X = rs.rand(300, 4).astype("float32")
    y = (rs.rand(300) > 0.5).astype(int)
    m = xgb.XGBClassifier(n_estimators=4, max_depth=2, tree_method="hist",
                          random_state=0, n_jobs=2)
    m.fit(X, y)
    m.get_booster().set_param({"device": "cuda"})   # 人为置于 cuda
    assert "cuda" in _booster_device(m).lower()

    align_predict_device(m)
    assert "cuda" not in _booster_device(m).lower(), _booster_device(m)


def test_align_helper_is_noop_for_non_xgboost():
    """非 xgboost 模型必须安全跳过(不能因类型不匹配抛错)。"""
    align_predict_device = _load_body_helper(
        "align_predict_device", "_model_score")

    class _Dummy:
        pass

    align_predict_device(_Dummy())      # 不抛异常即通过
    align_predict_device(None)


# --------------------------------------------------------------------------
# 2. lightgbm 探针:失败噪声不得落到 fd 2
# --------------------------------------------------------------------------

def test_python_level_redirect_does_not_catch_cpp_fatal():
    """先固定事实:C++ 层 fatal 绕过 contextlib.redirect_stderr。

    这条断言解释了为什么修复必须用 dup2 而不是 redirect_stderr ——
    若将来 lightgbm 改成走 Python 层 stderr,本测试会失败并提醒
    可以简化实现(而不是让它成为无人知晓的过期知识)。
    """
    lgb = pytest.importorskip("lightgbm")
    rs = np.random.RandomState(0)
    X = rs.rand(64, 4).astype("float32")
    y = (rs.rand(64) > 0.5).astype(int)

    script = textwrap.dedent(f"""
        import contextlib, io, sys
        sys.path.insert(0, {str(RUNTIME.parent)!r})
        import numpy as np, lightgbm as lgb
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                lgb.LGBMClassifier(device='cuda', n_estimators=1,
                                   verbosity=-1).fit(
                    np.random.RandomState(0).rand(64, 4).astype('float32'),
                    (np.random.RandomState(1).rand(64) > 0.5).astype(int))
        except Exception:
            pass
        sys.stdout.write('CAPTURED=' + str(len(buf.getvalue())))
    """)
    r = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, timeout=300)
    # 探针的 fatal 仍出现在子进程 stderr(说明 Python 层拦不住)
    assert "CUDA Tree Learner was not enabled" in r.stderr, _u8(r)
    # 且 Python 缓冲区是空的
    assert "CAPTURED=0" in r.stdout, _u8(r)


@pytest.mark.parametrize("where", ["baseline_template", "hpo_module"])
def test_cuda_probe_writes_nothing_to_fd2(where):
    """**核心回归**:探针在 fd 2 上必须零字节输出。

    ``runtime.baseline`` 的 ``_lgbm_cuda_ok`` 定义在 BASELINE_TEMPLATE
    字符串里(模块级不可见),所以要 exec 该段源码 —— 这样测的正是
    实际渲染进 trial 的那份实现。``runtime.hpo`` 的是模块级函数。
    """
    if where == "baseline_template":
        src = (RUNTIME / "baseline.py").read_text(encoding="utf-8")
        i = src.index("def _lgbm_cuda_ok")
        j = src.index("def _lgbm_fit_safe")
        probe_src = src[i:j]
        script = (
            "import os, sys\n"
            f"sys.path.insert(0, {str(RUNTIME.parent)!r})\n"
            f"_ns = {{'_LGBM_CUDA': None, 'os': os}}\n"
            f"exec({probe_src!r}, _ns)\n"
            "print('RESULT=' + str(_ns['_lgbm_cuda_ok']()))\n"
        )
    else:
        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(RUNTIME.parent)!r})\n"
            "from runtime.hpo import _lgbm_cuda_ok\n"
            "print('RESULT=' + str(_lgbm_cuda_ok()))\n"
        )

    r = subprocess.run([sys.executable, "-c", script],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, _u8(r)
    assert "RESULT=" in r.stdout, _u8(r)
    assert r.stderr == "", (
        "探针往 stderr 写了内容 —— lightgbm 的 C++ fatal 又漏出来了:\n"
        + r.stderr[:500])


def test_preflight_probes_gbm_backend_capability():
    """preflight 必须把"本机 GBM 能否上卡"变成一条可查的环境事实。"""
    sys.path.insert(0, str(SKILL_ROOT))
    from runtime.preflight import probe_gbm_backends

    probe = probe_gbm_backends()
    assert "lightgbm_cuda" in probe, probe
    assert "xgboost_cuda" in probe, probe
    assert "lightgbm_version" in probe, probe
    assert "xgboost_version" in probe, probe
    # 非 CUDA 编译的 lightgbm 必须带可读说明(否则又是"看起来像故障")
    if probe.get("lightgbm_cuda") is False:
        assert "非 CUDA 编译" in probe.get("note", ""), probe
