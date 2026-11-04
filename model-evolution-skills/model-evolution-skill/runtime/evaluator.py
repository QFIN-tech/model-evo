"""Evaluator:客观评测与验收机制(§12)。

关键原则(Immutable):
- final_test 不进入搜索反馈环(§12.1);
- acceptance 由代码计算,LLM 不决定自己是否成功(§2.2);
- noise-aware improvement gate,不用全局固定 MDE(§12.4)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from .config import Config
from .confirmation import (CONFIRM_CRITERION, confirm_config,
                           confirmation_margins, judge_delta,
                           legacy_two_sided, resolve_sigma_seed,
                           scaled_sigma_prior, SIGMA_PRIOR_BASE,
                           SIGMA_PRIOR_EXPONENT, SIGMA_PRIOR_HI,
                           SIGMA_PRIOR_LO, SIGMA_PRIOR_REF_N)
from .logging_utils import get_logger, now_iso, dump_json
from .metrics import (bootstrap_ci, compute_metric, metric_direction,
                      align_score_by_keys, paired_bootstrap_delta,
                      segment_passed, segment_ranking)
from .spec import TaskSpec

logger = get_logger(__name__)


def _scaled_prior_kwargs(config) -> dict:
    """σ_seed 规模先验的形状参数(默认值见 runtime.confirmation)。

    只在 evaluator 层读取,便于按 run 调参而不改代码常量;缺省键时不传,
    走 confirmation 模块的常量。
    """
    if config is None:
        return {}
    out = {}
    for key, name in (("evaluator.seed_variance_prior_base", "base"),
                      ("evaluator.seed_variance_prior_ref_n", "ref_n"),
                      ("evaluator.seed_variance_prior_exponent", "exponent"),
                      ("evaluator.seed_variance_prior_lo", "lo"),
                      ("evaluator.seed_variance_prior_hi", "hi")):
        v = config.get(key, None)
        if v is not None:
            out[name] = float(v)
    return out


#: 供测试与文档引用的默认形状(与 runtime.confirmation 常量同源)
SCALED_PRIOR_DEFAULTS = {"base": SIGMA_PRIOR_BASE,
                         "ref_n": SIGMA_PRIOR_REF_N,
                         "exponent": SIGMA_PRIOR_EXPONENT,
                         "lo": SIGMA_PRIOR_LO, "hi": SIGMA_PRIOR_HI}


class FinalTestGate:
    """final_test 一次性评测的访问控制(§12.1)。

    记录触碰次数与目的;超过 FINAL_TEST_MAX_TOUCHES 次拒绝访问。
    """

    MAX_TOUCHES = 1

    def __init__(self, run_dir: Path):
        self.path = Path(run_dir) / "final-test-gate.json"
        self.state = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            import json

            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"touches": [], "count": 0}

    def acquire(self, purpose: str = "final_evaluation") -> None:
        if self.state["count"] >= self.MAX_TOUCHES:
            raise PermissionError(
                f"final_test 已被评测 {self.state['count']} 次,"
                "不允许再次进入反馈环(§12.1)"
            )
        self.state["touches"].append({"purpose": purpose, "at": now_iso()})
        self.state["count"] += 1
        dump_json(self.path, self.state)

    def release_last(self, reason: str) -> None:
        """将最近一次终测标为中止，但**绝不恢复盲测额度**。

        一旦 ``acquire`` 成功，final_test 标签就可能已经进入进程内存；
        是否成功落盘指标并不能证明信息从未泄露。因此失败只能留痕并
        使本终测面失效，不能把 ``count`` 减回零。方法名为兼容旧调用
        保留，语义已从“回滚”改为 fail-closed 的“中止并失效”。
        """
        if not self.state["touches"]:
            return
        _last = self.state["touches"][-1]
        self.state["touches"][-1] = {**_last,
                                     "aborted_at": now_iso(),
                                     "abort_reason": reason}
        self.state["invalidated"] = True
        self.state["invalidation_reason"] = reason
        dump_json(self.path, self.state)


class Evaluator:
    """确定性评测:primary metric、delta、稳定性、切片、verdict。"""

    def __init__(self, spec: TaskSpec, config: Config):
        self.spec = spec
        self.config = config
        self.direction = metric_direction(spec.primary_metric)

    def _better(self, a: float, b: float) -> bool:
        if not np.isfinite(a) or not np.isfinite(b):
            return False
        return a > b if self.direction else a < b

    # ---- noise-aware gate(§12.4,v12 P0-2 收紧)----
    def noise_gate(self, delta: float, noise: float | None) -> str:
        """返回 likely_noise / promising / potentially_validated。

        v12(P0-2):noise 的口径已改为配对 bootstrap 的 **SE**(σ,
        见 metrics.paired_bootstrap_delta),本函数只做量级分档;
        accepted 的初步裁决在 _verdict 里以方向感知的
        p_better≥0.95 为门槛，代级 FDR 校正后才可保留。
        """
        if noise is None or not np.isfinite(noise) or noise <= 0:
            # 无噪声估计时退化为 bootstrap CI 是否含 0
            return "promising" if abs(delta) > 0 else "likely_noise"
        th = self.config.get("evaluator.noise_thresholds") or {}
        prom_r = float(th.get("promising_ratio", 0.5))
        val_r = float(th.get("validated_ratio", 1.0))
        if self.direction:
            if delta >= noise * val_r:
                return "potentially_validated"
            if delta >= noise * prom_r:
                return "promising"
            return "likely_noise"
        if delta <= -noise * val_r:
            return "potentially_validated"
        if delta <= -noise * prom_r:
            return "promising"
        return "likely_noise"

    # ---- 主入口 ----
    def evaluate(
        self,
        y_true: np.ndarray,
        score: np.ndarray,
        y_pred: np.ndarray | None = None,
        *,
        treatment: np.ndarray | None = None,
        parent_score: np.ndarray | None = None,
        baseline_score: np.ndarray | None = None,
        baseline_comparable: bool = True,
        is_root_trial: bool = False,
        confirm_trial: bool = False,
        multi_seed_scores: list[np.ndarray] | None = None,
        slices: dict[str, dict[str, np.ndarray]] | None = None,
        slice_parent_scores: dict[str, np.ndarray] | None = None,
        temporal_scores: list[np.ndarray] | None = None,
        resource: dict | None = None,
        complexity: dict | None = None,
        parent_keys: np.ndarray | None = None,
        parent_old_keys: np.ndarray | None = None,
        baseline_keys: np.ndarray | None = None,
        baseline_old_keys: np.ndarray | None = None,
        confirm_context: dict | None = None,
        # v2.17.0:锚定晋级(promotion_reference=anchor)——锚点分数由
        # runner 解析、按样本键对齐到当前行序后传入;anchor_note 记录
        # 锚点不可用/不可比的原因(只留痕,不影响 verdict)
        anchor_score: np.ndarray | None = None,
        anchor_id: str | None = None,
        anchor_same_as_parent: bool = False,
        anchor_note: dict | None = None,
        # v2.17.0:回归诊断的分期列(validation 时间列,与 y 同长)
        periods: np.ndarray | None = None,
        # v2.17.0:MDE 的量纲感知解析(runtime.mde.effective_mde)
        run_dir: Any = None,
        # v2.19.0:配对检验 SE 计入 seed 方差时的 run 级输入(同配方复现组
        # 的合并 σ_seed;runner._seed_sd_context 解析,排除当前 trial)
        seed_sd_context: dict | None = None,
    ) -> dict:
        metric = self.spec.primary_metric
        confirm_trial = bool(confirm_trial or
                             self.spec.data.get("trial_role") == "confirm")
        from .metrics import REGRESSION_METRICS, UPLIFT_METRICS
        if metric in REGRESSION_METRICS:
            # 回归:score 列即预测值,统一走 y_pred 口径
            if y_pred is None:
                y_pred = score

            def _reg_pred(s: np.ndarray) -> np.ndarray:
                return np.asarray(s, dtype=float)
        else:
            def _reg_pred(s: np.ndarray) -> np.ndarray:
                return s  # 分类:保持原样(用 score 概率口径)
        # uplift:指标必须有 treatment;缺失即评估失败(不静默降级)
        if metric in UPLIFT_METRICS and treatment is None:
            raise ValueError(
                f"uplift 指标 {metric} 需要 treatment 数组"
                "(validation split 的 treatment 列)")
        primary = compute_metric(metric, y_true, y_pred, score=score,
                                 treatment=treatment)
        out: dict[str, Any] = {
            "primary_metric": {"name": metric, "value": primary},
            "delta": {},
            "stability": {},
            "slice_evaluation": {},
            "segment_evaluation": {},
            "temporal_evaluation": {},
            "resource": resource or {},
            "complexity": complexity or {},
            "integrity": {"valid": True, "errors": []},
        }

        # ---- segment ranking(分段排序性,默认 10 段)----
        # 按 score 分位分箱,校验段内目标均值单调;分类段均值=正样本率,
        # 回归=段内目标均值,uplift=处置−对照段内增量。附加诊断,
        # 不阻塞 verdict(样本不足/score 退化时 passed=None)。
        try:
            seg = segment_ranking(
                y_true, score,
                n_bins=int(self.config.get("evaluator.segment_bins", 10)),
                treatment=treatment)
            # passed:单调性达标且段均值随 score 递增。v2.15.9:score 在三类
            # 任务里都与目标同向(分类=正类概率、回归=预测值、uplift=CATE),
            # 与"指标越大/越小越好"无关;旧实现把两者绑定,rmse 等越小越好
            # 的回归任务即便段均值严格单调递增也恒判 False(诊断字段,不进
            # verdict)
            if seg.get("bin_means") is not None:
                seg["passed"] = segment_passed(
                    seg, self.config.get(
                        "evaluator.segment_monotonic_threshold", 0.8))
            out["segment_evaluation"] = seg
        except Exception as exc:  # noqa: BLE001
            logger.warning("segment_ranking 失败: %s", exc)

        # ---- delta + noise ----
        noise = None
        if parent_score is not None:
            # P0-7:v1 baseline 的 45087 行分数对 sample 面 33598 行标签
            # 曾直接崩溃——跨视图比较在 runner 侧已拦截(v2.13.2 补齐:
            # runner 判视图不同即不传 parent_df,此处根本不会进入;
            # 下面这层是长度兜底,防旧调用方漏传视图)。
            try:
                if parent_keys is not None:
                    parent_score = align_score_by_keys(
                        parent_score, parent_keys, parent_old_keys)
                elif len(parent_score) != len(y_true):
                    raise ValueError(
                        f"parent_score 行数 {len(parent_score)} != "
                        f"validation {len(y_true)}")
            except Exception as exc:  # 键集合不一致不是 root trial
                out["integrity"]["valid"] = False
                out["integrity"]["errors"].append(
                    f"parent_prediction_incomparable:{exc}")
                parent_score = None
        if parent_score is not None:
            delta_parent = primary - compute_metric(
                metric, y_true, _reg_pred(parent_score) if metric in REGRESSION_METRICS else None,
                score=parent_score, treatment=treatment)
            out["delta"]["vs_parent"] = delta_parent
            if self.config.get("evaluator.bootstrap", True):
                # v19:配对键防御——keys 传入时 key-join 对齐(跨视图
                # 行序不同,行位置配对是伪影工厂,trial-009 实测
                # +0.2844);同视图行序天然一致,不传 keys 行为不变
                pb = paired_bootstrap_delta(
                    y_true, score, parent_score, metric,
                    treatment=treatment,
                    # 上方已重排成当前顺序，bootstrap 直接按位置配对，
                    # 避免二次用旧键重排。完整性检查已经完成。
                    n_boot=int(self.config.get("evaluator.bootstrap_samples",
                                               2000)))
                pb["p_better"] = pb.get("p_positive") if self.direction \
                    else (1.0 - pb["p_positive"]
                          if pb.get("p_positive") is not None else None)
                self._inflate_seed_variance(pb, seed_sd_context)
                noise = pb["noise"]
                out["delta"]["vs_parent_paired_bootstrap"] = pb
        if baseline_score is not None:
            # P0-7(G1):sample 视图 trial 用 v1 baseline 分数算 vs_baseline
            # 曾长度不匹配直接崩溃(旗舰路径从未跑通)。同视图同长才比,
            # 否则记 baseline_incomparable 跳过,不报错也不算错值。
            if baseline_comparable:
                try:
                    if baseline_keys is not None:
                        baseline_score = align_score_by_keys(
                            baseline_score, baseline_keys, baseline_old_keys)
                    elif len(baseline_score) != len(y_true):
                        raise ValueError(
                            f"baseline_score 行数 {len(baseline_score)} != "
                            f"validation {len(y_true)}")
                except Exception as exc:
                    out["integrity"]["valid"] = False
                    out["integrity"]["errors"].append(
                        f"baseline_prediction_incomparable:{exc}")
                    baseline_comparable = False
            if baseline_comparable and len(baseline_score) == len(y_true):
                out["delta"]["vs_baseline"] = primary - compute_metric(
                    metric, y_true, _reg_pred(baseline_score) if metric in REGRESSION_METRICS else None,
                    score=baseline_score, treatment=treatment)
                # v18/P2-6:root trial(confirm/无父代 explore)与
                # baseline 做配对 bootstrap——baseline 是天然父代对照。
                # 此前 root accepted 不经 noise gate(30w run trial-004
                # 实测口径缝:子代有 p_positive,root 只有 Δ>0);
                # 统一后 root 的提升判据与子代同尺,noise 也进
                # out["noise"] 供停止判定/shadow gap 消费。
                if is_root_trial and self.config.get(
                        "evaluator.bootstrap", True):
                    pb_root = paired_bootstrap_delta(
                        y_true, score, baseline_score, metric,
                        treatment=treatment,
                        n_boot=int(self.config.get(
                            "evaluator.bootstrap_samples", 2000)))
                    pb_root["p_better"] = pb_root.get("p_positive") \
                        if self.direction else (
                            1.0 - pb_root["p_positive"]
                            if pb_root.get("p_positive") is not None else None)
                    self._inflate_seed_variance(pb_root, seed_sd_context)
                    out["delta"]["vs_baseline_paired_bootstrap"] = pb_root
                    if noise is None:
                        noise = pb_root.get("noise")
            else:
                out["delta"]["vs_baseline"] = None
                out["delta"]["baseline_incomparable"] = True
        # ---- v2.17.0:vs_anchor(锚定晋级的对照)----
        # 口径同 vs_parent(点值 Δ + 配对 bootstrap + 方向感知 p_better);
        # 只供代级校正(evolution._apply_generation_fdr)消费,不进单试验
        # verdict,也不改写 noise(noise 仍是 vs_parent 的 SE)。
        self._attach_anchor_delta(
            out, y_true=y_true, score=score, primary=primary,
            treatment=treatment, anchor_score=anchor_score,
            anchor_id=anchor_id, anchor_same_as_parent=anchor_same_as_parent,
            anchor_note=anchor_note, seed_sd_context=seed_sd_context)
        out["noise"] = noise
        if "vs_parent" in out["delta"]:
            out["noise_gate"] = self.noise_gate(out["delta"]["vs_parent"], noise)

        # ---- secondary metrics ----
        # 分类任务自动附带头部圈人 lift(top10/top30,交付口径),
        # 与 TaskSpec 显式声明的 secondary 合并(spec 已拦截二者重复声明)
        sec = {}
        names = list(self.spec.data.get("secondary_metrics") or [])
        if metric not in REGRESSION_METRICS and metric not in UPLIFT_METRICS:
            for auto in ("lift_top10", "lift_top30"):
                if auto not in names:
                    names.append(auto)
        for name in names:
            try:
                sec[name] = compute_metric(name, y_true, y_pred, score=score,
                                           treatment=treatment)
            except Exception as exc:  # noqa: BLE001
                sec[name] = None
                logger.warning("secondary metric %s 失败: %s", name, exc)
        if sec:
            out["secondary_metrics"] = sec

        # ---- stability: multi-seed & bootstrap CI ----
        # v2.16.2(run4 regression D-006):回归指标走 y_pred 口径,而旧实现把
        # **主 seed** 的 y_pred 传给每个变体 → compute_metric 忽略 score=s,
        # 变体指标恒等于主 seed(trial-008:seed_metrics 三个同值、σ_seed 估计
        # 5e-16、multi_seed_std=0.0,确认门 σ_seed 退化为 floor)。变体必须用
        # 自己的预测:回归时 y_pred = 变体分数(_reg_pred 与主路径同口径);
        # 分类/uplift 仍按 score 概率口径,行为不变。
        _is_reg_metric = metric in REGRESSION_METRICS

        def _seed_metric(s: np.ndarray) -> float:
            return float(compute_metric(
                metric, y_true, _reg_pred(s) if _is_reg_metric else y_pred,
                score=s, treatment=treatment))

        seed_variant_vals: list[float] = []
        if multi_seed_scores and len(multi_seed_scores) > 1:
            vals = [_seed_metric(s) for s in multi_seed_scores]
            out["stability"]["multi_seed_std"] = float(np.std(vals))
            seed_variant_vals = [float(v) for v in vals]
        elif multi_seed_scores:
            seed_variant_vals = [_seed_metric(s) for s in multi_seed_scores]
        if self.config.get("evaluator.bootstrap", True):
            lo, hi = bootstrap_ci(
                y_true, score, metric, treatment=treatment,
                n_boot=int(self.config.get("evaluator.bootstrap_samples",
                                           2000)))
            out["stability"]["bootstrap_ci"] = [lo, hi]
        stability_passed = True
        ms_std = out["stability"].get("multi_seed_std")
        _stab_limit = 2 * noise if noise is not None and np.isfinite(noise) \
            and noise > 0 else None
        if confirm_trial and _stab_limit is not None:
            # v2.14.0:confirm 的 multi-seed 门同时看 MDE。大样本上 seed
            # 方差天然大于评测噪声(run2 实测 seed sd≈1.5×SE),只比
            # 2·noise 会让 multi-seed 一旦真正执行就结构性误杀可复现配方。
            _cc = confirm_config(self.config, run_dir=run_dir)
            _stab_limit = max(_stab_limit,
                              _cc["stability_mde_ratio"] * _cc["mde"])
            out["stability"]["confirm_std_limit"] = _stab_limit
        if ms_std is not None and _stab_limit is not None \
                and ms_std > _stab_limit:
            stability_passed = False
        out["stability"]["passed"] = stability_passed

        # ---- confirmation(v2.14.0):validation 侧非劣 + 保真 ----
        if confirm_trial and out["delta"].get("vs_parent") is not None:
            out["confirmation_validation"] = self._confirmation_validation(
                delta=float(out["delta"]["vs_parent"]), primary=primary,
                noise=noise, seed_variant_vals=seed_variant_vals,
                context=confirm_context)

        # ---- slice evaluation ----
        if slices:
            is_reg = metric in REGRESSION_METRICS
            worst = None
            worst_name = None
            for sname, pack in slices.items():
                try:
                    sy_pred = pack.get("y_pred")
                    if is_reg and sy_pred is None:
                        sy_pred = pack["score"]
                    sv = compute_metric(
                        metric, pack["y_true"], sy_pred, score=pack["score"],
                        treatment=pack.get("treatment"))
                except Exception:  # noqa: BLE001
                    continue
                delta_slice = None
                ps = (slice_parent_scores or {}).get(sname)
                if ps is not None:
                    delta_slice = sv - compute_metric(
                        metric, pack["y_true"], ps if is_reg else None,
                        score=ps, treatment=pack.get("treatment"))
                if delta_slice is not None and (worst is None or delta_slice < worst):
                    worst, worst_name = delta_slice, sname
            # P1-5(G2)切片门多重比较校正:旧口径取全部 k 个切片的最差
            # delta 与固定 2×noise 比较——k 个切片取 min 的期望偏差随 k
            # 增长,8–16 个切片时系统性误杀好 trial(seq-v3-3 gen-01
            # confirmation 与 gen-02 catboost 均被 0.0002 之差误杀)。
            # 改为 z_{1-α/k} 分位乘子:α=0.05、k=12 时约 2.93≈3×noise,
            # k 越少越宽松,与切片数解耦。
            k_slices = len(slices)
            mult = float(self.config.get(
                "evaluator.slice_threshold_multiplier", 0.0)) or None
            if mult is None:
                from scipy.stats import norm as _norm  # noqa: F401

                try:
                    from statistics import NormalDist

                    alpha_slice = float(self.config.get(
                        "evaluator.slice_alpha", 0.05))
                    mult = NormalDist().inv_cdf(
                        1 - alpha_slice / max(k_slices, 1))
                except Exception:  # noqa: BLE001  scipy/统计库缺失兜底
                    mult = 2.0 + 0.5 * np.log(max(k_slices, 1))
            # v2.19.0(slice_se_scaling=sqrt_k):单切片约 1/k 的样本,其 Δ 的
            # SE≈√k×整体 SE;旧口径乘整体 SE,零效应下 k=3/4 时误拒率
            # 29~43%(135 个真实 trial 中近零 trial 41% 被拒)。新 run 把
            # √k 并入乘子;在途 run 保持旧口径。
            _se_scale = 1.0
            if str(self.config.get("evaluator.slice_se_scaling", "none")
                   or "none").strip().lower() == "sqrt_k":
                _se_scale = float(np.sqrt(max(k_slices, 1)))
                mult = float(mult) * _se_scale
            out["slice_evaluation"] = {
                "n_slices": k_slices,
                "worst_slice": worst_name,
                "worst_slice_delta": worst,
                "threshold_multiplier": round(float(mult), 3),
                "se_scaling": round(_se_scale, 3),
                "passed": worst is None or (self.direction and worst > -mult * (noise or 0))
                or (not self.direction and worst < mult * (noise or 0)),
            }
            if noise is None:
                out["slice_evaluation"]["passed"] = True

        # ---- temporal regression ----
        if temporal_scores:
            is_reg = metric in REGRESSION_METRICS
            deltas = []
            for pack in temporal_scores:
                try:
                    ty_pred = pack.get("y_pred")
                    if is_reg and ty_pred is None:
                        ty_pred = pack["score"]
                    tp_pred = pack["parent_score"] if is_reg else None
                    deltas.append(compute_metric(
                        metric, pack["y_true"], ty_pred,
                        score=pack["score"], treatment=pack.get("treatment"))
                        - compute_metric(metric, pack["y_true"], tp_pred,
                                         score=pack["parent_score"],
                                         treatment=pack.get("treatment")))
                except Exception:  # noqa: BLE001
                    continue
            if deltas:
                worst_t = min(deltas) if self.direction else max(deltas)
                # v2.11.1:noise 可能为 None——confirm trial 与父代特征面
                # /seed 逐字相同时分数逐位相同,配对 bootstrap 的 delta
                # 恒为 0 ⇒ SE=0 ⇒ paired_bootstrap_delta 按设计返回
                # noise=None(见 metrics.py)。此前此处无条件 `2 * noise`
                # 抛 TypeError,评估段整体失败(failure_reason=
                # evaluator_error),即"复现得越完美越必然崩溃"。
                # 处置与 slice_evaluation 同族:噪声不可得时退化为不依赖
                # 噪声的判据——margin 退化为 0 且取含端形式(最差期不劣于
                # 父代即通过),复现面 worst_t≡0 恰好通过。
                if noise is not None and np.isfinite(noise) and noise > 0:
                    _temp_passed = (worst_t >= -(2 * noise)) if self.direction \
                        else (worst_t <= 2 * noise)
                else:
                    _temp_passed = (worst_t >= 0) if self.direction \
                        else (worst_t <= 0)
                out["temporal_evaluation"] = {
                    "n_periods": len(deltas),
                    "worst_delta": worst_t,
                    "passed": bool(_temp_passed),
                    "noise_available": noise is not None,
                }

        # ---- verdict(§12.3)----
        out["verdict"], out["confidence"], out["reason"] = self._verdict(
            out, is_root_trial=is_root_trial, confirm_trial=confirm_trial)
        # v2.17.0:vs_parent 口径的单试验 verdict(方向台账消费;代级校正
        # 与锚定晋级只改 verdict,不改它)。confirm 的裁决是复现判据,不写。
        if not confirm_trial:
            out["parent_verdict"] = out["verdict"]
            # v2.17.2(A1):撞 cap 且无余量的负 Δ 是"被截断",不是"方向更差"。
            # verdict 照旧(不改裁决),只附 cap_confounded 标记供方向台账/报告
            # 消费(台账不记负奖励,见 runtime/directions.py)
            _cc = cap_confounded_detail(
                out, higher_is_better=self.direction,
                is_root_trial=is_root_trial)
            if _cc is not None:
                out["cap_confounded"] = True
                out["cap_confounded_detail"] = _cc
        # ---- v2.17.0:回归诊断(水位/分期水位/主次指标背离)----
        # 纯诊断:在 verdict 之后计算,任何失败只留 error,不影响裁决。
        if metric in REGRESSION_METRICS:
            out["regression_diagnostics"] = self._regression_diagnostics(
                y_true, y_pred, periods=periods, parent_score=parent_score,
                metric=metric)
        out["evaluated_at"] = now_iso()
        return out

    def _regression_secondary(self, metric: str) -> str:
        from .metrics import REGRESSION_METRICS

        for name in (self.spec.data.get("secondary_metrics") or []):
            if name in REGRESSION_METRICS and name != metric:
                return str(name)
        return "mae"

    def _regression_diagnostics(self, y_true, y_pred, *, periods=None,
                                parent_score=None, metric: str) -> dict:
        try:
            from .regression_diagnostics import regression_diagnostics

            per = None
            if periods is not None and len(periods) == len(y_true):
                per = np.asarray(periods)
            par = None
            if parent_score is not None and len(parent_score) == len(y_true):
                par = np.asarray(parent_score, dtype=float)
            return regression_diagnostics(
                y_true, y_pred, periods=per, parent_pred=par,
                primary=metric, secondary=self._regression_secondary(metric))
        except Exception as exc:  # noqa: BLE001 —— 诊断失败不阻塞评估
            logger.warning("regression_diagnostics 失败(忽略): %s", exc)
            return {"error": str(exc)[:200]}

    def _inflate_seed_variance(self, pb: dict,
                               seed_sd_context: dict | None) -> None:
        """v2.19.0:配对检验的 SE 计入 seed 方差(evaluator.seed_variance_in_se)。

        bootstrap 只刻画 validation 抽样噪声;两个模型的差还含训练随机性
        (σ_seed 与 SE 同量级,中位约 1.2×SE)。开启时:
        ``se_eff = sqrt(SE² + factor·σ_seed²)``,p 值按正态近似
        ``Φ(Δ/se_eff)`` 重算,原值保留在 ``*_bootstrap`` 字段。σ_seed 解析与
        confirm 同口径(runtime.confirmation.resolve_sigma_seed):本 run 同配方
        复现组 → confirm_seed_sd_prior → floor_ratio×SE。关闭时不改动 pb。

        ``seed_sd_context is None`` 表示调用方不要求计入(confirm trial:其
        判据的 margin 已单独含 σ_seed,不重复计入);复评路径
        (harness.reverify)用 ``sigma_seed_override`` 复用落盘的 σ_seed,
        保证重算逐位一致。
        """
        if seed_sd_context is None or not self.config.get(
                "evaluator.seed_variance_in_se", False):
            return
        se = pb.get("se") if pb.get("se") is not None else pb.get("noise")
        try:
            se = float(se)
        except (TypeError, ValueError):
            return
        if not np.isfinite(se) or se <= 0:
            return
        delta = pb.get("delta_mean")
        if delta is None or not np.isfinite(float(delta)):
            return
        ctx = seed_sd_context
        if ctx.get("sigma_seed_override") is not None:
            sigma = {"sigma_seed": ctx["sigma_seed_override"],
                     "source": ctx.get("sigma_seed_source") or "override",
                     # 复核时原样带回落盘的原始估计(经验库比值的来源)
                     "estimate": ctx.get("sigma_seed_estimate"),
                     "estimate_source": ctx.get("sigma_seed_estimate_source")}
        else:
            cc = confirm_config(self.config)
            # v2.19.1:先验按 validation 行数缩放(σ_seed = r(n_val)·SE);
            # 经验库同数据底座的历史比值(prior_ratio)优先于公式,见
            # runner._seed_sd_context。实测值(复现组)不再被 1×SE 下限抬高。
            # 只有显式声明 prior_ratio_mode 的新 run 才启用比值先验(经验库
            # 比值与公式都算)—— v2.19.0 的 run(有 seed_variance_in_se 但无
            # 该键)即便 runner 给了库内比值也保持旧口径。
            prior_ratio = (ctx.get("prior_ratio")
                           if cc.get("prior_ratio_mode") else None)
            if prior_ratio is None and cc.get("prior_ratio_mode") == "scaled":
                prior_ratio = scaled_sigma_prior(
                    se, ctx.get("n_val"),
                    **_scaled_prior_kwargs(self.config))
            if prior_ratio is None:
                prior_ratio = cc.get("prior_ratio")
            # 搜索口径的 floor 可单独给(新 run 为 0:r<1 的下半段不被
            # 抬回 1×SE);未配置时沿用 confirm 的 floor_ratio(旧口径)。
            fr = self.config.get("evaluator.seed_variance_floor_ratio", None)
            sigma = resolve_sigma_seed(
                replicate_sd=ctx.get("replicate_sd"),
                replicate_df=int(ctx.get("replicate_df") or 0),
                prior=cc["prior"], prior_ratio=prior_ratio, se=se,
                floor_ratio=(cc["floor_ratio"] if fr is None else float(fr)),
                floor_measured=bool(self.config.get(
                    "evaluator.seed_variance_floor_measured", True)))
        sig = sigma.get("sigma_seed")
        if sig is None or not np.isfinite(float(sig)):
            return
        try:
            factor = float(self.config.get(
                "evaluator.seed_variance_factor", 1.0) or 0.0)
        except (TypeError, ValueError):
            factor = 1.0
        se_eff = float(np.sqrt(se ** 2 + max(factor, 0.0) * float(sig) ** 2))
        from statistics import NormalDist

        p_pos = float(NormalDist().cdf(float(delta) / se_eff))
        pb.update(
            se_bootstrap=se,
            p_positive_bootstrap=pb.get("p_positive"),
            p_better_bootstrap=pb.get("p_better"),
            sigma_seed=float(sig),
            sigma_seed_source=sigma.get("source"),
            # v2.19.1:floor 之前的原始估计及其来源。经验库只从实测来源
            # (run_replicates / multi_seed)取 σ_seed/SE 比值 —— 落盘的
            # sigma_seed 可能是 floor 或先验本身,拿它入库会自我循环。
            sigma_seed_estimate=sigma.get("estimate"),
            sigma_seed_estimate_source=sigma.get("estimate_source"),
            seed_variance_factor=factor,
            se=se_eff, noise=se_eff,
            z_score=float(delta) / se_eff,
            p_positive=p_pos,
            p_better=p_pos if self.direction else 1.0 - p_pos,
            p_basis="normal_approx_se_eff",
        )

    def _attach_anchor_delta(self, out: dict, *, y_true, score, primary,
                             treatment, anchor_score, anchor_id,
                             anchor_same_as_parent: bool,
                             anchor_note: dict | None,
                             seed_sd_context: dict | None = None) -> None:
        """v2.17.0:delta.vs_anchor / vs_anchor_paired_bootstrap / anchor_id。"""
        from .metrics import REGRESSION_METRICS

        delta = out["delta"]
        if anchor_id:
            delta["anchor_id"] = anchor_id
        if anchor_note:
            # 不可比/不可用:留痕跳过(anchor_incomparable 与 baseline 同口径)
            if anchor_note.get("status") == "incomparable":
                delta["anchor_incomparable"] = True
            delta["anchor_note"] = dict(anchor_note)
        if anchor_same_as_parent and "vs_parent" in delta:
            # 锚点即正式父代:Δ 与 bootstrap 逐字复用(同一对照,不重算)
            delta["vs_anchor"] = delta["vs_parent"]
            if "vs_parent_paired_bootstrap" in delta:
                delta["vs_anchor_paired_bootstrap"] = dict(
                    delta["vs_parent_paired_bootstrap"])
            delta["anchor_is_parent"] = True
            return
        if anchor_score is None:
            return
        metric = self.spec.primary_metric
        is_reg = metric in REGRESSION_METRICS
        anchor_score = np.asarray(anchor_score, dtype=float) if is_reg \
            else np.asarray(anchor_score)
        if len(anchor_score) != len(y_true):
            delta["anchor_incomparable"] = True
            delta["anchor_note"] = {
                "status": "incomparable",
                "reason": (f"anchor_score 行数 {len(anchor_score)} != "
                           f"validation {len(y_true)}")}
            return
        try:
            delta["vs_anchor"] = primary - compute_metric(
                metric, y_true, anchor_score if is_reg else None,
                score=anchor_score, treatment=treatment)
            if self.config.get("evaluator.bootstrap", True):
                pb = paired_bootstrap_delta(
                    y_true, score, anchor_score, metric, treatment=treatment,
                    n_boot=int(self.config.get("evaluator.bootstrap_samples",
                                               2000)))
                pb["p_better"] = pb.get("p_positive") if self.direction \
                    else (1.0 - pb["p_positive"]
                          if pb.get("p_positive") is not None else None)
                self._inflate_seed_variance(pb, seed_sd_context)
                delta["vs_anchor_paired_bootstrap"] = pb
        except Exception as exc:  # noqa: BLE001 —— 锚点对照失败只留痕
            delta.pop("vs_anchor", None)
            delta["anchor_note"] = {"status": "error",
                                    "reason": str(exc)[:200]}

    def _confirmation_validation(self, *, delta: float, primary: float,
                                 noise: float | None,
                                 seed_variant_vals: list[float] | None = None,
                                 context: dict | None = None) -> dict:
        """validation 侧确认判据(v2.14.0,见 runtime/confirmation.py)。

        multi-seed 可用时(主 seed + ≥1 变体)以 seed 均值 Δ 判定并把子代
        seed 方差按 1/n 缩小;σ_seed 解析顺序见 resolve_sigma_seed。
        """
        cc = confirm_config(self.config)
        ctx = context or {}
        seed_metrics = None
        if seed_variant_vals:
            seed_metrics = [float(primary), *[float(v)
                                              for v in seed_variant_vals]]
        sigma = resolve_sigma_seed(
            seed_metrics=seed_metrics,
            replicate_sd=ctx.get("replicate_sd"),
            replicate_df=int(ctx.get("replicate_df") or 0),
            prior=cc["prior"], se=noise, floor_ratio=cc["floor_ratio"])
        n_child = len(seed_metrics) if seed_metrics else 1
        parent_metric = float(primary) - float(delta)
        delta_used = (float(np.mean(seed_metrics)) - parent_metric) \
            if seed_metrics else float(delta)
        margins = confirmation_margins(
            se=noise, sigma_seed=sigma.get("sigma_seed"), n_child=n_child,
            z_lower=cc["z_lower"], z_upper=cc["z_upper"],
            cfg_margin=cc["cfg_margin"])
        out = {
            "criterion": CONFIRM_CRITERION,
            "split": "validation",
            "available": margins is not None,
            "delta_primary": float(delta),
            "delta_used": delta_used,
            "delta_basis": "seed_mean" if seed_metrics else "primary_seed",
            "n_seeds": n_child,
            "seed_metrics": seed_metrics,
            "se_eval": noise,
            "sigma_seed": sigma.get("sigma_seed"),
            "sigma_seed_source": sigma.get("source"),
            "sigma_seed_estimate": sigma.get("estimate"),
            "sigma_seed_estimate_source": sigma.get("estimate_source"),
            "sigma_seed_df": sigma.get("df"),
            "replicate_groups": ctx.get("replicate_groups"),
            "legacy_two_sided_equivalence": legacy_two_sided(
                delta, noise, cc["cfg_margin"]),
        }
        if margins is not None:
            out.update(margin_lower=margins["lower"],
                       margin_upper=margins["upper"],
                       sd_delta=margins["sd_delta"],
                       **judge_delta(delta_used, margins, self.direction))
        return out

    def _verdict(self, ev: dict, *, is_root_trial: bool = False,
                 confirm_trial: bool = False) -> tuple[str, str, str]:
        """§12.3 verdict 计算。

        v12 改动(P0-2 / P1-6):
        - accepted 的 Trial 级门槛是配对 bootstrap 的方向感知
          p_better ≥ 0.95；同代 BH/跨代 alpha spending 在 Generation
          收尾继续校正，未通过者降为 promising;
        - 新增 confirmation 语义:复现类 trial 的成功定义是 Δ≈0,
          判据改为等价性检验(Δ 的 95% CI 落 ±noise 内即 confirmed),
          允许进 Archive——旧口径把最可信的证据必然误判 rejected;
        - noise=None 时不再用 abs(delta) 判 promising(G4:负 delta
          也曾误入 promising)。

        v2.13.3(confirm 不被搜索护栏短路):
        - **slice/temporal 门不再作为 confirm 的准入前置**。二者是
          **搜索 trial 的方向质量护栏**("新方向别在某个切片/时段上
          崩掉"),不是等价性复现的判据;protocol 对 confirm 明示的
          成功要素是 validation 等价 + shadow 等价 + multi-seed 稳定性
          (references/feature-exploration.md「抽样与全量复核」)。
          旧口径把 slice 门放在 confirm 等价性分支之前无条件执行,在
          3 期切片(每期 ~1.1 万行、单期 AUC 的 SE≈0.0026)的口径下
          **结构性不可通过**:单期噪声远大于整体配对噪声,任何 seed
          翻动都会在某期掉 1~2 SE。实测 seq-v3-v2.13.0.1 gen-03
          trial-013:validation Δ=−8.6e-05(p=0.428,等价区间内)、
          shadow Δ=−0.000855(< margin 0.001687)、multi_seed_std=9.2e-05
          —— 三要素全部达成,却因 07-15 期 −0.001254(单期 1 SE 内)
          被 slice 门短路成 rejected,fresh_confirmation.passed=false,
          冠军无法交付。同款误杀在 seq-v3-3 gen-01 confirmation 与
          gen-02 catboost 已有记录(见下方 slice 门多重比较校正注释)。
        - 处置:confirm trial 的 stability 仍是硬门(它直接对应
          multi-seed 要素);slice/temporal **结果照常计算并写入
          evaluation.json**,只降级为诊断——额外产出
          `slice_temporal_waived` 段(含原始 through/未过事实与
          worst delta),使"被豁免"这件事在产物层面可见、可审计,
          而不是静默放行。豁免范围仅限 confirm(search_action/
          trial_role=confirm),普通搜索 trial 的门槛完全不变。
        """
        gate = ev.get("noise_gate")
        if (ev.get("integrity") or {}).get("valid") is False:
            return ("invalid", "high",
                    "预测身份/源码完整性检查失败，禁止进入选择与交付")
        stability_ok = ev.get("stability", {}).get("passed", True)
        slice_ok = ev.get("slice_evaluation", {}).get("passed", True)
        temporal_ok = ev.get("temporal_evaluation", {}).get("passed", True)
        delta = (ev.get("delta") or {}).get("vs_parent")
        dp = (ev.get("delta") or {}).get("vs_parent_paired_bootstrap") or {}
        p_pos = dp.get("p_better", dp.get("p_positive"))

        # confirm 的 slice/temporal 降级为诊断(见 docstring v2.13.3);
        # 豁免事实写进 ev,由 _evaluate 落进 evaluation.json。
        _slice_temporal_waived = False
        if confirm_trial and not (slice_ok and temporal_ok):
            _slice_temporal_waived = True
            ev["slice_temporal_waived"] = {
                "waived": True,
                "reason": ("confirm 的 slice/temporal 是搜索期方向护栏,"
                           "非等价性复现判据;结果保留为诊断"),
                "slice_passed": bool(slice_ok),
                "temporal_passed": bool(temporal_ok),
                "worst_slice_delta": (
                    ev.get("slice_evaluation") or {}).get("worst_slice_delta"),
                "worst_temporal_delta": (
                    ev.get("temporal_evaluation") or {}).get("worst_delta"),
            }

        # v2.14.0:confirm 的裁决整体交给 _confirm_verdict(稳定性门也在
        # 其中;失败记 unconfirmed 而非 rejected —— confirm 子代是同配方
        # 换 seed 的合法模型,"没确认"不等于"模型差")。
        if confirm_trial:
            return self._confirm_verdict(ev, stability_ok)

        if not stability_ok or (not _slice_temporal_waived
                                and (not slice_ok or not temporal_ok)):
            return ("rejected", "medium",
                    "稳定性或切片/时序检查未通过")
        # 根 Trial(baseline / 无父 explore):无父代可比,它建立起跑线。
        # v18/P2-6:root 与 baseline 的配对 bootstrap 可用时,提升判据
        # 与子代同尺(p_better≥0.95 → accepted;<0.80 → 只要 Δ>0 且
        # 稳定性过,保持 accepted——根 trial 的职责是建立起跑线,
        # 不做候选提升裁决);bootstrap 缺失(无 baseline 分数/配置关)
        # 时回退旧口径"通过稳定性即 accepted"。
        if is_root_trial and delta is None:
            rb = (ev.get("delta") or {}).get("vs_baseline_paired_bootstrap") \
                or {}
            _root_p = rb.get("p_better", rb.get("p_positive"))
            if _root_p is not None:
                d_bl = (ev.get("delta") or {}).get("vs_baseline")
                if d_bl is not None and self._better(d_bl, 0) \
                        and _root_p >= 0.95:
                    return ("accepted", "high",
                            "根 Trial:vs_baseline 配对 bootstrap "
                            "p_better≥0.95,起跑线即显著提升")
                # v2.11.7:标记"仅建立跑线、未主张提升"。该裁决的
                # reason 自述"不参与提升裁决",但调用方
                # evolution.finalize() 的 accepted_now 只按 verdict 计数,
                # 会把它当成提升证据 ⇒ 收敛出口与穷尽出口的
                # `trials_accepted == 0` 同时失锁,演进无法自动停止
                # (gen-02 实测:trial-007/008 漏传 --parents ⇒ 判为根
                # Trial ⇒ accepted ⇒ 两个出口全被堵死)。此处留下机器
                # 可读标记,由计数侧排除。
                ev["root_baseline_only"] = True
                return ("accepted", "medium",
                        "根 Trial:建立起跑线(vs_baseline "
                        f"p_better={_root_p:.3f},"
                        "不参与提升裁决)")
            # v2.11.7:同上——无父代可比即无提升主张,不得计入 accepted。
            ev["root_baseline_only"] = True
            return ("accepted", "medium", "根 Trial:建立起跑线,无父代提升可判")
        if delta is None:
            return ("rejected", "medium", "无父代可比分数")
        # 方向感知的显著性:负 delta(方向感知后)不可能 accepted
        if not self._better(delta, 0):
            return ("rejected", "medium", "相对父代无提升")
        if p_pos is not None and p_pos >= 0.95:
            return ("accepted", "high", "配对 bootstrap p_better≥0.95，等待代级 FDR 校正")
        if p_pos is not None and p_pos >= 0.80:
            return ("promising", "medium",
                    f"p_better={p_pos:.3f}∈[0.80,0.95),方向积极但未达显著")
        if p_pos is not None:
            return ("promising", "low",
                    f"p_better={p_pos:.3f}<0.80,信号弱")
        # 无 bootstrap 证据时按量级分档(不可能 accepted)
        if gate == "potentially_validated":
            return ("promising", "medium", "提升超过 noise 阈值但无显著性证据")
        if gate == "promising":
            return ("promising", "low", "正向信号但未超过 noise 阈值")
        return ("rejected", "medium", "提升幅度在噪声范围内")

    def _confirm_verdict(self, ev: dict,
                         stability_ok: bool) -> tuple[str, str, str]:
        """confirm trial 的 validation 侧裁决(v2.14.0)。

        旧口径(P1-6 起):|Δ| ≤ 2·SE 双侧等价。它有两个结构缺陷
        (req-t7-rand400w run2 实测):
        - SE 只含评测抽样噪声,不含 seed 方差 —— 大样本上两者同阶,
          可复现配方单次过门率 35~45%(4 次 confirm 过 1 次,且过的那次
          shadow 上显著更差);
        - 双侧:子代**比父代好**也判失败(trial-022 val +1.78e-4 >
          1.68e-4 被拒,而它是全 run 最优模型)。
        新口径:非劣(下侧,交付安全)+ 保真(上侧,z 更宽,防借 confirm
        之名换配方绕过代级 FDR),margin 含 seed 方差。失败记
        ``unconfirmed``:同配方换 seed 的合法模型,不是"模型差"。
        最终 passed 还需 runner 的 shadow 侧同口径判定。
        """
        delta = (ev.get("delta") or {}).get("vs_parent")
        if delta is None:
            return ("unconfirmed", "high",
                    "confirm 无父代可比分数:无法判定复现(检查父代 "
                    "validation_score 是否在档、是否同数据视图)")
        cv = ev.get("confirmation_validation")
        if cv is None:
            # 未经 evaluate() 的调用方(重判/测试):按 ev 的 noise 现算,
            # 无 multi-seed / 复现组信息 ⇒ σ_seed 取 floor
            cv = self._confirmation_validation(
                delta=float(delta),
                primary=float((ev.get("primary_metric") or {}).get("value")
                              or 0.0),
                noise=ev.get("noise"))
            ev["confirmation_validation"] = cv
        if not stability_ok:
            stab = ev.get("stability") or {}
            return ("unconfirmed", "medium",
                    "confirm multi-seed 稳定性未通过(seed std="
                    f"{stab.get('multi_seed_std')} > 上限 "
                    f"{stab.get('confirm_std_limit')}):配方对 seed 过于"
                    "敏感,不能认证为可复现")
        if not cv.get("available"):
            if abs(float(delta)) < 1e-4:
                return ("confirmed", "low",
                        "点值等价(无 bootstrap 噪声与 seed 方差估计)")
            return ("unconfirmed", "medium",
                    "confirmation 未通过:无噪声估计且 Δ 偏离父代")
        lo, up = cv["margin_lower"], cv["margin_upper"]
        d = cv["delta_used"]
        tail = (f"(sd={cv['sd_delta']:.6f}:SE={cv.get('se_eval')}, "
                f"σ_seed={cv.get('sigma_seed')}[{cv.get('sigma_seed_source')}]"
                f", n_seeds={cv.get('n_seeds')})")
        if not cv["non_inferior"]:
            return ("unconfirmed", "medium",
                    f"confirmation 未通过(非劣):Δ={d:+.6f} 低于下界 "
                    f"−{lo:.6f}{tail}——新 seed 在 validation 上显著不及"
                    "父代,父代成绩不可复现")
        if not cv["fidelity_ok"]:
            return ("unconfirmed", "medium",
                    f"confirmation 未通过(保真):Δ={d:+.6f} 超出上界 "
                    f"+{up:.6f}{tail}——子代远优于父代,不是同配方复现;"
                    "配方有改动应作为搜索 trial 提交并过代级 FDR")
        return ("confirmed", "high",
                f"validation 非劣检验通过:Δ={d:+.6f} ∈ [−{lo:.6f}, "
                f"+{up:.6f}]{tail};待独立 shadow 确认")


def cap_confounded_detail(ev: dict | None, *, es_info: dict | None = None,
                          higher_is_better: bool = True,
                          is_root_trial: bool = False,
                          confirm_trial: bool = False) -> dict | None:
    """v2.17.2(A1):撞 cap 混杂的负结论判定;命中返回明细,否则 None。

    事故(run4 req-t7-400w trial-017):gen-04 冠军面 HPO 的 lr-shrink 重训
    被 8000 棵树上限截断(best_iter 7999/8000,cap_extended_to=null),得到
    伪负 Δparent −0.00095、rejected;方向台账据此给 model:hpo 记负奖励,
    而 lr 修正子代(trial-018)在同一结构上 +0.000304 且 confirm 复现。

    判据(只对非根、非 confirm trial):
    1. 结论为负:方向感知增益 < 0;Δ 不可得时退回 verdict/parent_verdict =
       rejected。Δ>0 却 rejected(增益未过门)**不算** —— 它不是负奖励来源,
       且在整条血缘都撞 cap 的 run 里(run4 trial-004~016 全部 8000/8000),
       把它们一并剔除会让"两次未过门"的方向永远凑不够证伪试次(D1);
    2. 撞 cap 且无余量:``es_info.cap_bound_unresolved`` 为真(v2.17.2 模板
       在剩余余量内外扩后仍撞 cap 时写);旧产物兜底 —— ``cap_bound`` 为真
       且 ``stopped_by == "cap"``、``cap_extended_to`` 为空、``cap ≥
       MAX_TREES``(已在全局硬顶,无法再外扩)。es_info 与 evaluation.
       complexity(runner 从 es_info 抄的子集)任一满足即可。
    verdict 不改;调用方据明细加 ``cap_confounded`` 标记。
    """
    ev = ev or {}
    if is_root_trial or confirm_trial:
        return None
    delta = (ev.get("delta") or {}).get("vs_parent")
    gain = None
    if isinstance(delta, (int, float)) and not isinstance(delta, bool) \
            and np.isfinite(float(delta)):
        gain = float(delta) if higher_is_better else -float(delta)
    pv = ev.get("parent_verdict")
    if isinstance(pv, dict):
        pv = pv.get("verdict")
    negative = (gain < 0) if gain is not None else \
        ("rejected" in (ev.get("verdict"), pv))
    if not negative:
        return None
    from .config import MAX_TREES

    cx = ev.get("complexity") if isinstance(ev.get("complexity"),
                                            dict) else {}
    es = es_info if isinstance(es_info, dict) else {}

    def _int(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    rule, src = None, {}
    for name, d, stop_key in (("es_info", es, "stopped_by"),
                              ("complexity", cx, "es_stopped_by")):
        if d.get("cap_bound_unresolved") is True:
            rule, src = f"{name}.cap_bound_unresolved", d
            break
    if rule is None:
        for name, d, stop_key in (("es_info", es, "stopped_by"),
                                  ("complexity", cx, "es_stopped_by")):
            cap = _int(d.get("cap") or d.get("cap_initial"))
            if d.get("cap_bound") is True and d.get(stop_key) == "cap" \
                    and d.get("cap_extended_to") in (None, "", 0) \
                    and cap is not None and cap >= MAX_TREES:
                rule, src = f"legacy:{name}", d
                break
    if rule is None:
        return None
    return {
        "rule": rule,
        "delta_vs_parent": delta,
        "gain": gain,
        "verdict": ev.get("verdict"),
        "cap": _int(src.get("cap") or src.get("cap_initial")),
        "best_iteration": src.get("stage1_best_iteration",
                                  src.get("best_iteration",
                                          cx.get("best_iteration"))),
        "stopped_by": src.get("stopped_by", src.get("es_stopped_by")),
        "cap_extended_to": src.get("cap_extended_to"),
        "max_trees": MAX_TREES,
        "note": ("训练撞树数上限且无余量外扩(轮数被截断),负 Δ 混入了"
                 "欠训练效应:verdict 保留,但不作为该方向的负证据"
                 "(方向台账不记负奖励);如需裁决,提高 lr 或放宽 cap 后重跑"),
    }


def slice_by_columns(
    df, spec: TaskSpec, columns: list[str], max_groups: int = 8
) -> dict[str, dict]:
    """构造切片评测包:数值列按分位分箱,类别列取 top 组。"""
    packs: dict[str, dict] = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            try:
                bins = pd.qcut(s, 4, duplicates="drop")
            except ValueError:
                continue
            for level in pd.Categorical(bins).categories[:max_groups]:
                mask = (bins == level).values
                if mask.sum() < 50:
                    continue
                packs[f"{col}={level}"] = {"mask": mask}
        else:
            for val in s.value_counts().head(max_groups).index:
                mask = (s == val).values
                if mask.sum() < 50:
                    continue
                packs[f"{col}={val}"] = {"mask": mask}
    return packs
