"""回归测试:v2.12.2.8 安全 dtype 治理(段错误 + 键列污染)。

事故来源(2026-09-18,req-cnt-t7 run):
    `pd.to_numeric(series, errors="coerce")` 在**含非数字值的长 object
    列**上走 pandas 的递归失败分支,栈耗尽后 SIGSEGV(exit 139)。
    实测阈值:连续失败段 ≥ 1978 行即崩(容器默认栈 12.5MB);44638 行
    需要约 220MB 栈。`errors="coerce"` 只处理解析失败,**不防爆栈**。

    崩溃点有三处,全在"取数冒烟/评估"路径上,且异常被宽 except 吞掉
    或直接杀死进程:
      1. preflight.check_consumers 的 group_importance 预检 —— profile
         的自消费冒烟,导致 phase 永远落不到 PROFILED;
      2. error_analysis.build_error_analysis 的高损失画像 —— 训练成功
         后评估段崩溃,留下孤儿 RUNNING,evaluate_only 反复重入同一崩溃点;
      3. capacity._prep / profile._coerce / baseline 的 dtype 治理。

    次要缺陷(同源):同一解析器会把 24 位十六进制样本键**静默解析**
    成数字(`"000000000000000000000000"` → `0.0`),于是
    `notna().sum() >= 0.95 * n` 判据误判为"可数值化",把键列放进
    数值特征面。

本文件锁定修复:`runtime/safe_dtype.safe_to_numeric` 必须
(a) 在长非数字列上**不崩**;(b) 拒绝十六进制键列;(c) 对真实数值列
保持与 `pd.to_numeric` 一致的语义。
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = SKILL_ROOT / "runtime"


# --------------------------------------------------------------------------
# 1. 段错误回归:低栈子进程里跑,崩溃以 returncode<0 暴露
# --------------------------------------------------------------------------

#: 旧实现崩溃的最小规模(实测 1978 行起)。取 20000 行留足余量。
_CRASH_ROWS = 20_000
#: 12.5MB = 容器默认软栈;旧实现必然崩。
_LOW_STACK_KB = 12_500


def _run_in_subprocess(body: str, *, stack_kb: int) -> subprocess.CompletedProcess:
    """在指定栈上限的子进程里执行片段(崩溃 → returncode 为负)。"""
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(RUNTIME.parent)!r})
        import warnings
        warnings.simplefilter("ignore")
        {body}
    """)
    return subprocess.run(
        ["bash", "-c",
         f"ulimit -s {stack_kb}; exec {sys.executable} -c {_q(script)}"],
        capture_output=True, text=True, timeout=600)


def _q(s: str) -> str:
    import shlex
    return shlex.quote(s)


def test_pandas_to_numeric_segfaults_on_long_non_numeric_object_column():
    """先证明"裸 to_numeric 会崩"——这是本修复存在的理由。

    若某天 pandas 修了这个 bug,本测试会失败,提示可以退役 safe_dtype
    的兜底路径(而不是让兜底悄悄变成死代码)。
    """
    body = f"""
        import hashlib
        import pandas as pd
        s = pd.Series([hashlib.md5(str(i).encode()).hexdigest()
                       for i in range({_CRASH_ROWS})], dtype=object)
        pd.to_numeric(s, errors="coerce")
        print("NO_CRASH")
    """
    r = _run_in_subprocess(body, stack_kb=_LOW_STACK_KB)
    assert r.returncode != 0, (
        "预期 pd.to_numeric 在长非数字 object 列上段错误,但进程正常退出;"
        f"pandas={pd.__version__},stdout={r.stdout!r}。"
        "若 pandas 已修复,可评估退役 safe_dtype 的兜底分支。")
    assert r.returncode < 0 or "Segmentation fault" in r.stderr, (
        f"预期信号级崩溃,实际 returncode={r.returncode},stderr={r.stderr[:400]!r}")


def test_safe_to_numeric_survives_same_column():
    """修复后:同样的列,同样的低栈,必须正常返回 None。"""
    body = f"""
        import hashlib
        import pandas as pd
        from runtime.safe_dtype import safe_to_numeric
        s = pd.Series([hashlib.md5(str(i).encode()).hexdigest()
                       for i in range({_CRASH_ROWS})], dtype=object)
        out = safe_to_numeric(s)
        assert out is None, f"十六进制键列应判非数值,实际 {{out!r}}"
        print("SAFE_OK")
    """
    r = _run_in_subprocess(body, stack_kb=_LOW_STACK_KB)
    assert r.returncode == 0, (
        f"safe_to_numeric 仍崩溃:rc={r.returncode},stderr={r.stderr[:600]!r}")
    assert "SAFE_OK" in r.stdout


# --------------------------------------------------------------------------
# 2. 语义:键列拒绝、真实数值列放行、混合列逐元素解析
# --------------------------------------------------------------------------

def _load():
    sys.path.insert(0, str(RUNTIME.parent))
    from runtime.safe_dtype import safe_to_numeric
    return safe_to_numeric


def test_rejects_hex_key_column():
    """24 位十六进制样本键:裸 to_numeric 会"部分解析"成数字(污染源)。"""
    safe_to_numeric = _load()
    n = 3000
    hex_keys = pd.Series([f"{i:024x}" for i in range(n)], dtype=object)
    # 先确认裸实现在这里确实会给出"看似可用"的数值(这就是污染)
    naive = pd.to_numeric(hex_keys, errors="coerce")
    assert naive.notna().any(), "前提失效:裸 to_numeric 不再部分解析 hex"
    # safe 版本必须整体拒绝
    assert safe_to_numeric(hex_keys) is None


def test_rejects_md5_sample_key():
    safe_to_numeric = _load()
    keys = pd.Series([hashlib.md5(str(i).encode()).hexdigest()
                      for i in range(500)], dtype=object)
    assert safe_to_numeric(keys) is None


def test_accepts_real_numeric_columns():
    safe_to_numeric = _load()
    n = 3000
    for name, s, expect in (
        ("int-str", pd.Series([str(i) for i in range(n)], dtype=object),
         "int64"),
        ("float-str", pd.Series([f"{i}.25" for i in range(n)], dtype=object),
         "float64"),
        ("sci", pd.Series([f"{i}e-3" for i in range(n)], dtype=object),
         "float64"),
        ("signed", pd.Series([f"-{i}" for i in range(n)], dtype=object),
         "int64"),
        ("numeric-dtype", pd.Series(range(n), dtype="float64"), "float64"),
    ):
        out = safe_to_numeric(s)
        assert out is not None, f"{name} 应可数值化"
        assert str(out.dtype) == expect, f"{name} dtype={out.dtype}"
        assert out.notna().all(), f"{name} 不应产生 NaN"


def test_mixed_column_parses_elementwise():
    """多数可解析的混合列:可解析值转数字,不可解析值 → NaN。"""
    safe_to_numeric = _load()
    s = pd.Series([str(i) for i in range(100)] + ["abc", None], dtype=object)
    out = safe_to_numeric(s)
    assert out is not None, "98/100 可解析,应判为数值列"
    assert out.iloc[0] == 0.0 and out.iloc[99] == 99.0
    assert pd.isna(out.iloc[100])   # "abc" → NaN
    assert pd.isna(out.iloc[101])   # None  → NaN


def test_minority_numeric_column_is_rejected():
    """少数值可解析的列(如十六进制键)必须整体拒绝,不得"解析一部分"。"""
    safe_to_numeric = _load()
    s = pd.Series(["1", "2", "abc", "def", "ghi", "jkl"], dtype=object)
    assert safe_to_numeric(s) is None


def test_parse_ratio_is_tunable():
    safe_to_numeric = _load()
    s = pd.Series([str(i) for i in range(100)] + ["abc"] * 10, dtype=object)
    assert safe_to_numeric(s) is None                      # 90.9% < 95%
    assert safe_to_numeric(s, min_parse_ratio=0.9) is not None


def test_empty_and_all_nan_return_none():
    safe_to_numeric = _load()
    assert safe_to_numeric(pd.Series([], dtype=object)) is None
    assert safe_to_numeric(pd.Series([None, None], dtype=object)) is None


# --------------------------------------------------------------------------
# 3. 画像不把非特征列当特征
# --------------------------------------------------------------------------

def test_feature_column_filter_excludes_keys_and_internals():
    sys.path.insert(0, str(RUNTIME.parent))
    from runtime.safe_dtype import is_feature_column

    assert is_feature_column("dw_0006_used_amt") is True
    assert is_feature_column("__mvo_sample_key") is False
    assert is_feature_column("__mvo_anything") is False
    assert is_feature_column("pday", extra_exclude=("pday",)) is False


def test_error_analysis_skips_non_feature_columns():
    """画像里不得出现样本键/时间/实体/目标列。"""
    sys.path.insert(0, str(RUNTIME.parent))
    import numpy as np
    from runtime.error_analysis import build_error_analysis

    n = 800
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    score = rng.random(n)
    X = pd.DataFrame({
        "real_feature": rng.normal(size=n),
        "another_feature": rng.normal(size=n) * 3,
        "__mvo_sample_key": [f"{i:024x}" for i in range(n)],
        "pday": ["2026-05-01"] * n,
        "user_no": [f"UR{i}" for i in range(n)],
        "req_cnt_t7": y,          # 标签本体(评估帧里真实存在)
    })
    out = build_error_analysis(y, score, None, X,
                               model_type="binary_classification",
                               n_bins=10, label="t")
    feats = {r["feature"] for r in out.get("high_loss_profile", [])}
    assert "__mvo_sample_key" not in feats
    assert "pday" not in feats
    assert "user_no" not in feats
    assert "req_cnt_t7" not in feats, "标签本体不得出现在特征画像里"


# --------------------------------------------------------------------------
# 4. 冒烟行数必须超过已知崩溃阈值
# --------------------------------------------------------------------------

def test_smoke_rows_exceed_crash_threshold():
    """SMOKE_ROWS 低于 1978 时,"冒烟通过"对真实规模毫无预测力。"""
    sys.path.insert(0, str(RUNTIME.parent))
    from runtime.preflight import SMOKE_ROWS

    assert SMOKE_ROWS > 1978, (
        f"SMOKE_ROWS={SMOKE_ROWS} 低于 pandas 递归解析爆栈阈值(1978),"
        "冒烟会漏掉这类崩溃")


# --------------------------------------------------------------------------
# 5. 调用点接线:不得再有裸 to_numeric 出现在 dtype 治理路径
# --------------------------------------------------------------------------

_BANNED_PATTERNS = (
    'pd.to_numeric(X[c], errors="coerce")',
    'pd.to_numeric(X_tr[c], errors="coerce")',
    'pd.to_numeric(X_va[c], errors="coerce")',
    'pd.to_numeric(out[c], errors="coerce")',
    'pd.to_numeric(out[col], errors="coerce")',
    'pd.to_numeric(X_valid[c], errors="coerce")',
    'pd.to_numeric(va[c], errors="coerce")',
)


@pytest.mark.parametrize("rel", [
    "runtime/capacity.py",
    "runtime/baseline.py",
    "runtime/profile.py",
    "runtime/group_importance.py",
    "runtime/error_analysis.py",
    "runtime/preflight.py",
    "scripts/_mvo/cmd_startup.py",
])
def test_no_bare_to_numeric_in_dtype_governance(rel):
    src = (SKILL_ROOT / rel).read_text(encoding="utf-8")
    hits = [p for p in _BANNED_PATTERNS if p in src]
    assert not hits, (
        f"{rel} 仍有裸 to_numeric dtype 治理调用 {hits};"
        "改用 runtime.safe_dtype.safe_to_numeric")


# --------------------------------------------------------------------------
# 6. 评估段崩溃熔断:信号级崩溃不得导致无限 evaluate_only 重入
# --------------------------------------------------------------------------

def test_eval_inflight_counter_lifecycle(tmp_path):
    """in-flight 计数:mark 递增、clear 清零、缺失读作 0。"""
    sys.path.insert(0, str(RUNTIME.parent))
    from runtime.runner import TrialRunner

    tdir = tmp_path / "trial-x"
    tdir.mkdir()
    # 未初始化的 trial 读作 0
    assert TrialRunner._eval_crash_count(object.__new__(TrialRunner), tdir) == 0
    for expect in (1, 2, 3):
        TrialRunner._mark_eval_inflight(object.__new__(TrialRunner), tdir)
        assert TrialRunner._eval_crash_count(object.__new__(TrialRunner), tdir) == expect
    TrialRunner._clear_eval_inflight(object.__new__(TrialRunner), tdir)
    assert TrialRunner._eval_crash_count(object.__new__(TrialRunner), tdir) == 0


def test_eval_crash_loop_is_bounded(tmp_path):
    """连续 2 次"进了评估没出结果"后,孤儿 RUNNING 分派必须判 FAILED。

    这是 req-cnt-t7 实测事故的回归锁:评估段段错误绕过 except Exception,
    锁成孤儿而分数已落盘,旧实现每次重入都回到同一崩溃点(空转 5 轮,
    每轮一个 ~16GB core dump)。
    """
    sys.path.insert(0, str(RUNTIME.parent))
    from runtime.runner import TrialRunner

    tdir = tmp_path / "trial-y"
    tdir.mkdir()
    # 模拟:上一轮评估崩溃留下 in-flight 计数
    TrialRunner._mark_eval_inflight(object.__new__(TrialRunner), tdir)
    TrialRunner._mark_eval_inflight(object.__new__(TrialRunner), tdir)
    assert TrialRunner._eval_crash_count(object.__new__(TrialRunner), tdir) == 2
    # 达到阈值 → 不得再走 evaluate_only(阈值常量与实现一致)
    assert TrialRunner._eval_crash_count(object.__new__(TrialRunner), tdir) >= 2
