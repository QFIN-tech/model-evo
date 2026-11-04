"""v2.13.3 回归测试:confirm 判定不被搜索护栏短路。

已核实缺陷(证据:run example-seq-v3-v2.13.0.1 的 gen-03 trial-013):

  ``Evaluator._verdict`` 把 slice/temporal 门放在 confirm 等价性分支
  **之前**无条件执行(``if not stability_ok or not slice_ok or not
  temporal_ok: return rejected``)。但这两个门是**搜索 trial 的方向质量
  护栏**("新方向别在某个切片/时段上崩掉"),不是等价性复现的判据——
  protocol 对 confirm 明示的成功要素是 validation 等价 + shadow 等价 +
  multi-seed 稳定性(references/feature-exploration.md「抽样与全量复核」)。

  在 3 期切片(每期 ~1.1 万行、单期 AUC 的 SE≈0.0026)的口径下,这个
  前置**结构性不可通过**:单期噪声远大于整体配对噪声,任何 seed 翻动
  都会在某期掉 1~2 SE。实测 trial-013:validation Δ=−8.6e-05(p=0.428,
  等价区间 ±0.000908 内)、shadow Δ=−0.000855(< margin 0.001687)、
  multi_seed_std=9.2e-05 —— 三要素全部达成,却因 07-15 期 −0.001254
  (单期 1 SE 内)被短路成 rejected,fresh_confirmation.passed=false,
  champion 无法交付(``finalize-retrain`` 无冠军可交付)。

本文件直接驱动 ``_verdict``(证据 dict 显式构造),锁定:
  A. confirm:slice/temporal 不过**不再**否决等价性判定;
  B. confirm:stability 不过仍然硬否决(对应 multi-seed 要素);
  C. confirm:豁免事实落进 ``slice_temporal_waived``,含未过门的原始数值;
  D. 非 confirm 搜索 trial:slice/temporal 门行为**完全不变**(回归保护);
  E. integrity 失败仍判 invalid(豁免不得绕过完整性检查);
  F. 正面用例:slice/temporal 全过时无豁免标记,行为与旧版一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import SKILL  # noqa: E402  hermetic bootstrap

sys.path.insert(0, str(SKILL))

from runtime.config import Config  # noqa: E402
from runtime.evaluator import Evaluator  # noqa: E402
from runtime.spec import TaskSpec  # noqa: E402

#: trial-013 的实测数值(gen-03,run example-seq-v3-v2.13.0.1)
T013_DELTA = -0.000086
T013_NOISE = 0.000454       # 配对 bootstrap SE
T013_SLICE_WORST = -0.001254  # 2026-07-15 期(trigger 了旧口径的误杀)
T013_SLICE_MULT = 2.128
T013_SLICE_PASSED = False   # −0.001254 < −2.128×0.000454 = −0.000966


def _spec() -> TaskSpec:
    return TaskSpec(data={
        "task_id": "t-confirm-waiver", "task_name": "t",
        "model_type": "binary_classification",
        "target": {"name": "label", "positive_definition": "1"},
        "time": {"column": "dt"}, "entity": {"id_column": "uid"},
        "primary_metric": "auc",
        "split_policy": {"type": "time_based", "train": "2026-01",
                         "validation": "2026-02", "final_test": "2026-04"},
        "data": {},
    })


def _ev(*, delta=T013_DELTA, noise=T013_NOISE,
        slice_passed=T013_SLICE_PASSED,
        worst_slice=T013_SLICE_WORST,
        temporal_passed=True, worst_temporal=-0.000416,
        stability_passed=True, integrity_valid=True) -> dict:
    """构造 trial-013 形态的证据 dict(字段口径与 _evaluate 产出一致)。"""
    return {
        "integrity": {"valid": integrity_valid},
        "noise": noise,
        "delta": {
            "vs_parent": delta,
            "vs_parent_paired_bootstrap": {
                "se": noise, "delta_mean": delta,
                "p_better": 0.428, "p_positive": 0.428,
            },
        },
        "stability": {"passed": stability_passed, "multi_seed_std": 9.21e-05},
        "slice_evaluation": {
            "n_slices": 3, "passed": slice_passed,
            "worst_slice": "2026-07-15",
            "worst_slice_delta": worst_slice,
            "threshold_multiplier": T013_SLICE_MULT,
        },
        "temporal_evaluation": {
            "n_periods": 2, "passed": temporal_passed,
            "worst_delta": worst_temporal,
        },
        "noise_gate": "likely_noise",
    }


def _verdict(ev: dict, *, confirm: bool):
    e = Evaluator(_spec(), Config.load(skill_config=False))
    return e._verdict(ev, is_root_trial=False, confirm_trial=confirm)


# --------------------------------------------------------------------------
# A/C/E. confirm:slice 门不过 → 仍 confirmed,豁免留痕
# --------------------------------------------------------------------------

def test_confirm_slice_failure_no_longer_short_circuits():
    """核心断言:trial-013 的数值形态在 confirm 下判 confirmed。"""
    ev = _ev()
    verdict, conf, reason = _verdict(ev, confirm=True)
    assert verdict == "confirmed", f"{verdict}: {reason}"
    # 确认判据确实被执行(而不是被稳定性/切片短路);v2.14.0 起为非劣检验
    assert "非劣检验通过" in reason


def test_confirm_slice_failure_records_waiver():
    """豁免事实与原始数值必须落盘,不能静默放行。"""
    ev = _ev()
    _verdict(ev, confirm=True)
    w = ev.get("slice_temporal_waived")
    assert w is not None, "豁免必须留痕"
    assert w["waived"] is True
    assert w["slice_passed"] is False
    assert w["temporal_passed"] is True
    assert w["worst_slice_delta"] == T013_SLICE_WORST
    assert w["worst_temporal_delta"] == -0.000416
    assert "搜索期方向护栏" in w["reason"]


def test_confirm_temporal_failure_also_waived():
    """temporal 门同理:它是搜索护栏,不是复现判据。"""
    ev = _ev(temporal_passed=False, worst_temporal=-0.0012)
    verdict, _, reason = _verdict(ev, confirm=True)
    assert verdict == "confirmed", f"{verdict}: {reason}"
    assert ev["slice_temporal_waived"]["temporal_passed"] is False


def test_confirm_both_gates_failing_waived():
    ev = _ev(slice_passed=False, temporal_passed=False)
    verdict, _, _ = _verdict(ev, confirm=True)
    assert verdict == "confirmed"


def test_confirm_integrity_failure_still_invalid():
    """豁免不得绕过完整性检查。"""
    ev = _ev(integrity_valid=False)
    verdict, _, _ = _verdict(ev, confirm=True)
    assert verdict == "invalid"
    assert "slice_temporal_waived" not in ev


def test_confirm_delta_beyond_margin_still_fails():
    """豁免只放行 slice/temporal,确认判据本身不放松。

    v2.14.0:判据改为非劣 + 保真。子代远优于父代(+0.01 ≫ 保真上界)
    不是同配方复现 → unconfirmed(保真);远差于父代 → unconfirmed(非劣)。
    """
    ev = _ev(delta=0.01)
    verdict, _, reason = _verdict(ev, confirm=True)
    assert verdict == "unconfirmed"
    assert "保真" in reason
    ev = _ev(delta=-0.01)
    verdict, _, reason = _verdict(ev, confirm=True)
    assert verdict == "unconfirmed"
    assert "非劣" in reason


# --------------------------------------------------------------------------
# B. stability 仍是硬门
# --------------------------------------------------------------------------

def test_confirm_stability_still_hard_gate():
    """stability 不过仍然硬否决 —— 它直接对应 multi-seed 要素。"""
    ev = _ev(stability_passed=False, slice_passed=False)
    verdict, _, reason = _verdict(ev, confirm=True)
    assert verdict == "unconfirmed"
    assert "稳定性" in reason


# --------------------------------------------------------------------------
# D. 搜索 trial 的门槛完全不变(回归保护)
# --------------------------------------------------------------------------

def test_search_trial_slice_failure_still_rejects():
    ev = _ev()
    verdict, _, reason = _verdict(ev, confirm=False)
    assert verdict == "rejected", f"{verdict}: {reason}"
    assert "稳定性或切片/时序检查未通过" in reason
    assert "slice_temporal_waived" not in ev


def test_search_trial_temporal_failure_still_rejects():
    ev = _ev(temporal_passed=False)
    verdict, _, _ = _verdict(ev, confirm=False)
    assert verdict == "rejected"
    assert "slice_temporal_waived" not in ev


def test_search_trial_stability_failure_still_rejects():
    ev = _ev(stability_passed=False)
    verdict, _, _ = _verdict(ev, confirm=False)
    assert verdict == "rejected"


# --------------------------------------------------------------------------
# F. 无门失败时行为与旧版一致
# --------------------------------------------------------------------------

def test_confirm_all_gates_passing_unaffected():
    """slice/temporal 全过 → 无豁免标记,走正常等价性判定。"""
    ev = _ev(slice_passed=True, worst_slice=0.000587)
    verdict, _, reason = _verdict(ev, confirm=True)
    assert verdict == "confirmed"
    assert "slice_temporal_waived" not in ev
    assert "非劣检验通过" in reason


def test_search_trial_all_gates_passing_unaffected():
    """对照:搜索 trial 全过时照常走提升裁决(Δ<noise → rejected)。"""
    ev = _ev(slice_passed=True, worst_slice=0.000587)
    verdict, _, _ = _verdict(ev, confirm=False)
    assert verdict == "rejected"   # Δ 在噪声内,无提升
    assert "slice_temporal_waived" not in ev
