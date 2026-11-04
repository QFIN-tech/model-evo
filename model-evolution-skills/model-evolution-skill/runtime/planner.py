"""确定性自主搜索规划器。

把“读上下文后自由发挥”替换为可审计流水线：候选生成 → 结构校验 →
价值/成本打分 → 多样化批选择。输出是 typed proposal batch，不接触
final_test，也不直接宣称效果；执行器仍以 trial/evaluator 的客观证据
裁决结果。
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from .config import Config
from .directions import DirectionLedger, _strip_feat_prefix
from .feature_phase import BATCH_KEY as FEATURE_PHASE_KEY
from .feature_phase import ERROR_CODE as FEATURE_PHASE_ERROR
from .feature_phase import (feature_generations_done, feature_phase_status,
                            is_model_direction)
from .logging_utils import dump_json, dump_yaml, load_json, load_yaml, now_iso
from .metrics import metric_direction
from .spec import TaskSpec
from .trial import TrialFactory, confirmation_parent_eligibility

ALLOWED_ACTIONS = {"explore", "exploit", "crossover", "data_view",
                   "hpo", "confirm"}
ALLOWED_MODULES = {"feature", "sample", "model", "loss", "data"}
MATERIALIZED_MODEL_ALGOS = ("lightgbm", "xgboost", "catboost")
FORBIDDEN_TOKENS = ("final_test", "final-test", "final test")

#: v2.19.0(#10):删除 v2.13.4 的 `_REPARSE_PREFIX_FAMILIES = {seq_d365: sq_,
#: seq_gap_d365: sqgap_}` 与 `seq*` 前缀推导 —— 它们是某一份数据集的列名约定,
#: 换数据即失效(新数据的序列列不叫 seq*、派生列也不带 sq_)。"某 reparse 源是否
#: 已被冠军消费"改为由 trial 产物判定,见 `_source_already_consumed_by_champion`。

#: v2.19.0:profile_rule 候选的手设常数(与 inventory partially 档同量级,
#: 先验增益另走 expected_gain = prior_gain×MDE,见 runtime/profile_rules.py)
PROFILE_RULE_TIER = {"information_gain": 0.6, "novelty": 0.5,
                     "uncertainty_prior": 0.3}
#: v2.19.0(#12):经验候选。检索分阈值(retrieval.search 的 score:主题相关度 ×
#: 档权重 × 证据因子;L0 精确主题 + 中性后验 = 0.75)、可物化的原则类别、
#: 每代最多提名数、先验刻度(prior_gain_mde = 刻度 × 证据后验 × 档折扣,
#: **待 benchmark 校准**:10-30 报告里原则层效果未被证明,初值保守)。
EXPERIENCE_MIN_SCORE = 0.5
EXPERIENCE_CATEGORIES = {"feature_engineering": ("explore", ["feature"]),
                         "architecture": ("explore", ["model"]),
                         "hpo": ("explore", ["model"])}
EXPERIENCE_MAX_CANDIDATES = 3
EXPERIENCE_PRIOR_SCALE_MDE = 1.0
EXPERIENCE_TIER_DISCOUNT = {"L0": 1.0, "L1": 0.85, "L2": 0.6}
#: v2.19.0(#14):同面 HPO 期望增益的后验。旧口径 0.75×MDE 降为**先验**
#: (exp-202609-1f43bd13 的同面实测中值),与本 run + 经验库同任务 HPO 实测 Δ
#: 做伪计数 HPO_PRIOR_K0 的收缩平均;后验 < HPO_MIN_EXPECTED_MDE×MDE 不提名。
HPO_PRIOR_GAIN_MDE = 0.75
HPO_PRIOR_K0 = 1.0
HPO_MIN_EXPECTED_MDE = 0.1
#: "正向历史" = 某次 HPO 的增益 ≥ 0.5×MDE(+1e-5 级的正值不算正证据)
HPO_POSITIVE_MDE = 0.5
#: 大样本阈值(与 profile_rules.LARGE_N 同值):无正向历史时默认不提名同面 HPO
HPO_LARGE_N = 1_000_000
#: v2.19.0(#14):confirm final_only 下"仍有高分候选"的判据:存在非 confirm
#: 候选的 expected_gain ≥ 0.5×MDE(有实测/先验支撑的增益,不看手设常数)
CONFIRM_HIGH_VALUE_MDE = 0.5


def _source_literal_re(column: str):
    """源列名作为**字符串字面量**出现(`"col"` / `'col'`)的正则。

    只认字面量而不是任意子串:避免 `dt` 这类短列名命中变量名,也避免
    `seq_d365` 命中 `seq_d365_tail` 这类派生列名。
    """
    import re

    return re.compile(r"""(['"])""" + re.escape(str(column)) + r"\1")

#: [FIX-2026-09-19 D2] 方向台账证伪判定的最小试次数:同一 family
#: 累计 n_tried 达到该值、零 accepted 且 mean_delta<=0 才判 falsified。
#: 单次负结果不下死刑(配对门/噪声的偶然性),两次及以上零收益才
#: 构成"该方向已证伪"的证据;做成模块常量便于测试与调参。
FALSIFIED_MIN_TRIED = 2
#: [v2.17.2 D1] 证伪判定的单尾 z(90%,与方向台账 LCB 同口径):
#: UCB = mean_gain + z·std/√n_eff 仍 < MDE ⇒ "乐观估计也够不到最小有意义
#: 提升",与 lcb_gain<0 并列构成证伪判据。
FALSIFIED_UCB_Z = 1.28
#: [v2.17.2 D1] avoid 经验卡按 family 分量匹配时忽略的通用段
#: (`info:<col>:<dim>` 的 info、`model:<x>` 的 model 等不构成方向)
_GENERIC_FAMILY_TOKENS = {"info", "feat", "model", "view", "action",
                          "screen", "confirmation", "loss", "sample", "any"}


def _norm_family(value: Any) -> str:
    """family / 经验卡 target 的比较键(v2.17.2 D1)。

    台账 arm 带 `feat:` 前缀(`feat:info:seq_gap_d365:recency`,见
    directions.family_of_trial),信息源候选是 `info:<col>:<dim>`,旧实现
    直接比字符串 → 永不命中;这里统一去 `feat:` 前缀、去首尾空白与尾部
    下划线(`seqt_` 与 `feat:seqt` 同键)、小写。
    """
    s = _strip_feat_prefix(str(value or "").strip().lower())
    return s.rstrip("_")

SEARCH_SPACES = {
    "lightgbm": {
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.15,
                          "log": True},
        "num_leaves": {"type": "int", "low": 15, "high": 127},
        "min_child_samples": {"type": "int", "low": 10, "high": 100},
        "subsample": {"type": "float", "low": 0.65, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.55, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1e-3, "high": 20.0,
                       "log": True},
    },
    "xgboost": {
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.15,
                          "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "min_child_weight": {"type": "float", "low": 0.5, "high": 20.0,
                             "log": True},
        "subsample": {"type": "float", "low": 0.65, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.55, "high": 1.0},
        "reg_lambda": {"type": "float", "low": 1e-3, "high": 20.0,
                       "log": True},
    },
    "catboost": {
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.15,
                          "log": True},
        "depth": {"type": "int", "low": 4, "high": 10},
        "l2_leaf_reg": {"type": "float", "low": 1e-3, "high": 20.0,
                        "log": True},
        "random_strength": {"type": "float", "low": 0.0, "high": 2.0},
    },
}


def _semantic_hash(value: dict) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _contains_forbidden(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False).lower()
    return any(token in text for token in FORBIDDEN_TOKENS)


def _proposal_identity(value: dict) -> dict:
    """跨代稳定的候选身份；不含 generation/分数/自然语言说明。"""
    return {k: value.get(k) for k in
            ("action", "family", "parent_ids", "algorithm", "data_view",
             "changed_modules", "recipe")}


def _legacy_identity(value: dict) -> dict:
    """旧 Trial 无 planner hash 时使用的保守去重身份。"""
    family = value.get("family")
    if isinstance(family, list):
        family = ",".join(sorted(str(x) for x in family))
    return {"action": value.get("action"), "family": family,
            "parent_ids": value.get("parent_ids") or [],
            "algorithm": str(value.get("algorithm") or "").lower()}


def _cli_script_path() -> Path:
    """物化命令里的 CLI 路径(v2.17.0)。

    旧实现 ``Path(__file__).resolve()`` 会把软链解析成实路径:Agent 用
    ``.claude/skills/modelevo/scripts/modelevo.py``(软链)调用 plan,
    批次里却写 ``/data/.../model-evo-git/.../modelevo.py``(实路径),
    同一 run 的日志/命令出现两种写法,复制命令时还会绕过软链指向的版本
    切换。优先取**当前 CLI 的调用路径**(``sys.argv[0]`` 为 modelevo.py 时,
    ``os.path.abspath`` 只补全不解析软链);非 CLI 调用(测试、库内调用)
    回落旧口径。
    """
    try:
        argv0 = sys.argv[0] if sys.argv else ""
        if argv0 and Path(argv0).name == "modelevo.py":
            cand = Path(os.path.abspath(argv0))
            if cand.exists():
                return cand
    except Exception:  # noqa: BLE001 — 路径探测失败回落旧口径
        pass
    return Path(__file__).resolve().parents[1] / "scripts" / "modelevo.py"


class AutonomousPlanner:
    """从已有证据生成下一代有限、可执行、可证伪的 proposal batch。"""

    def __init__(self, run_dir: Path, spec: TaskSpec, config: Config):
        self.run_dir = Path(run_dir)
        self.spec = spec
        self.config = config
        self.factory = TrialFactory(self.run_dir, spec=spec, config=config)
        self.trials = self.factory.all_trials()
        self.by_id = {t.trial_id: t for t in self.trials}
        self.direction_up = metric_direction(spec.primary_metric)
        # [FIX-2026-09-17] _validate 可能被 plan() 之外的路径调用
        # (测试、未来的 dry-run);默认空串,别让属性只在 plan 里存在。
        self._feature_phase_release_reason = ""
        # v2.17.0:confirm 提名的审计留痕(哪些 eligible 父代被跳过、为什么)
        self._confirm_skipped: list[dict] = []

    # ------------------------------------------------------------------
    # v2.17.0:量纲感知 MDE / 方向感知增益
    # ------------------------------------------------------------------
    def _mde(self) -> float:
        """当前 run 的有效 MDE(runtime.mde.effective_mde;主指标单位,正数)。

        旧实现各处直接读 ``evaluator.minimum_meaningful_improvement``(绝对
        值,默认 0.001):原值 rmse(≈12932)任务上 0.75×MDE 的 HPO 先验、
        信息源候选上界、边际递减门全部失真。解析失败回落旧键。
        """
        cached = getattr(self, "_mde_cache", None)
        if cached is not None:
            return cached
        val = None
        try:
            from .mde import effective_mde

            val = float(effective_mde(self.config,
                                      run_dir=getattr(self, "run_dir", None)))
        except Exception:  # noqa: BLE001 — 共享模块缺失/坏 run-state 回落旧键
            val = None
        if not val or val <= 0:
            try:
                val = float(self.config.get(
                    "evaluator.minimum_meaningful_improvement", 0.001)) \
                    or 0.001
            except Exception:  # noqa: BLE001
                val = 0.001
        self._mde_cache = val
        return val

    def _gain(self, delta) -> float | None:
        """原始 Δ → 方向感知增益(越大越好);非数值返回 None。"""
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            return None
        d = float(delta)
        if d != d:  # NaN
            return None
        up = getattr(self, "direction_up", True)
        return d if up else -d

    # ------------------------------------------------------------------
    # 特征期门(v2.16.0;口径见 runtime/feature_phase.py)
    # ------------------------------------------------------------------
    # 特征期内 `_validate` 拒绝 model/loss/hpo 候选
    # (feature_phase_model_locked)。拒绝时把候选**收集**进
    # batch.feature_phase_gate,并在"特征侧已充分收敛"的确定性信号出现
    # 时,允许**每代一个**模型方向名额带机器可读理由放行([FIX-2026-09-17]
    # 起的释放通道,原挂在 capacity 闸门上;实测 drawdown/t7 v2.12.2.2
    # gen-01/02/03 三代提名同一对 HPO 候选、三代被同一原因拒)。
    # v2.16.0 起本门不再读 capacity-report.json:那份报告的两个触发信号
    # 在抽样面上近乎常量,门的实际输出不携带信息(见 feature_phase.py)。
    # 判定信号取三者之并(任一成立即视为收敛):
    #   (a) information-inventory 的 unconsumed == 0;
    #   (b) 无 reparse 源仍有 unconsumed/partially_consumed 粒度;
    #   (c) 最近两代均有已评估 trial 且都未产生新的 accepted/confirmed。
    def _exploration_converged(self) -> tuple[bool, str]:
        """特征侧是否已收敛到值得释放一个模型方向名额的程度。"""
        reasons = []
        # (a)(b) 信息源清单清账
        # [FIX-2026-09-18] 旧实现读 inv.get("unconsumed") 与
        # inv["sources_total"]["unconsumed"],而 build_inventory() 的返回
        # 结构是 {sources, summary:{unconsumed,...}} —— 两个键都不存在。
        # 于是 `inv.get("available")` 为 None 直接落到 elif,该 elif 的
        # 条件同样恒假,整个 (a)(b) 分支静默跳过;只要 "unconsumed" 在
        # 清单里,函数就**永远不会**返回 True,每代一个的模型方向名额
        # (v2.12.2.2 专门为"闸门永久锁死 model/HPO"做的修复)从未生效。
        # 实测:example-seq-v3-v2.12.2.9 的 run 里,即使把两个 seq 源
        # 手动置为 fully_consumed,该函数仍返回 (False, "")。
        # 兼容口径:summary.unconsumed 优先(现结构),其次顶层
        # unconsumed/sources_total(历史结构与测试桩),都没有则视为
        # 未能清账(不阻断,只是不给收敛结论)。
        try:
            from .inventory import build_inventory

            inv = build_inventory(self.run_dir, self.spec) or {}
            _summary = inv.get("summary") or {}
            _unconsumed = inv.get("unconsumed")
            if _unconsumed is None:
                _unconsumed = _summary.get("unconsumed")
            if _unconsumed is None:
                _unconsumed = (inv.get("sources_total") or {}).get(
                    "unconsumed")
            if _unconsumed is not None:
                if int(_unconsumed) == 0:
                    reasons.append("information_inventory_unconsumed_0")
                else:
                    return False, ""
        except Exception:  # noqa: BLE001 — 清单缺失不是阻断条件
            pass
        # (c) 最近两代均无新 accepted/confirmed
        gens = sorted({t.generation for t in self.trials
                       if t.data.get("evaluation")})
        if len(gens) < 2:
            return bool(reasons), ",".join(reasons)
        _new_accepted = set()
        for g in gens[-2:]:
            for t in self.trials:
                if t.generation != g:
                    continue
                ev = t.data.get("evaluation") or {}
                # v2.14.0:confirm 复现父代配方,不是"新 accepted"
                if ev.get("verdict") in ("accepted", "confirmed") and \
                        ev.get("trial_id") and \
                        str(t.data.get("search_action") or "") != "confirm":
                    _new_accepted.add(t.trial_id)
        if not _new_accepted:
            reasons.append("two_generations_without_new_accepted")
        return bool(reasons), ",".join(reasons)

    def _next_generation(self) -> int:
        return max((t.generation for t in self.trials), default=0) + 1

    def _ranked_trials(self, verdicts=("accepted", "confirmed")) -> list:
        rows = []
        for trial in self.trials:
            ev = trial.data.get("evaluation") or {}
            value = (ev.get("primary_metric") or {}).get("value")
            if value is None or ev.get("verdict") not in verdicts:
                continue
            if (ev.get("integrity") or {}).get("valid", True) is False:
                continue
            rows.append(trial)
        return sorted(rows, key=lambda t: (
            -(t.data["evaluation"]["primary_metric"]["value"]
              if self.direction_up else
              -t.data["evaluation"]["primary_metric"]["value"]),
            t.trial_id))

    def _parent(self):
        ranked = self._ranked_trials()
        return ranked[0] if ranked else None

    def _base(self, *, action: str, family: str, hypothesis: str,
              rationale: str, falsification: str, parent=None,
              algorithm: str | None = None,
              changed_modules: list[str] | None = None,
              information_gain: float = 0.0,
              novelty: float = 0.0,
              expected_gain: float = 0.0,
              uncertainty: float = 0.0,
              uncertainty_prior: float = 0.0,
              potential_gain_upper_bound: float = 0.0,
              cost_hours: float = 0.5,
              recipe: dict | None = None,
              prior_note: str = "") -> dict:
        generation = self._next_generation()
        proposal = {
            "schema_version": 1,
            "generation": generation,
            "action": action,
            "family": family,
            "parent_ids": [parent.trial_id] if parent is not None else [],
            "algorithm": algorithm or (
                parent.data.get("algorithm") if parent is not None else
                (self.config.get("baseline.algorithms") or ["lightgbm"])[0]),
            "data_view": (parent.data.get("data_view") or "v1")
            if parent is not None else "v1",
            "changed_modules": changed_modules or [],
            "hypothesis": hypothesis,
            "rationale": rationale,
            "falsification": falsification,
            "recipe": recipe or {},
            "expected_resources": {"cost_hours": float(cost_hours),
                                   "heavy": cost_hours >= 2.0},
            "acquisition": {
                "expected_gain": float(expected_gain),
                "uncertainty": float(uncertainty),
                "information_gain": float(information_gain),
                "novelty": float(novelty),
                "uncertainty_prior": float(uncertainty_prior),
                "cost_penalty": float(cost_hours) * 0.15,
                "potential_gain_upper_bound": float(
                    potential_gain_upper_bound),
            },
        }
        a = proposal["acquisition"]
        if prior_note:
            a["prior_note"] = prior_note
        # [FIX-2026-09-19 D3 + v2.13.4 量纲订正] acquisition 量纲归一。
        #
        # D3 的原始意图:让**方向台账的实测后验**能压过**手设常数**。
        # 台账 mean_delta/std_delta 单位是主指标(AUC,量级 0.001),
        # information_gain/novelty 是手设常数(0.3~2.0);直接相加时
        # 常数主导排序,方向后验没有影响。修法是"把量纲对齐到常数那一
        # 侧"——即用**常数刻度**(O(1))表达实测值,而不是把常数除到
        # 主指标刻度去。
        #
        # v2.13.4 订正(req-t7-drawdown-v21331 gen-04 实测):D3 把
        # expected_gain/uncertainty 除以 MDE,而
        # information_gain/novelty **未除** —— 归一化只做了一半,造成
        # 常数项被放大 1000 倍的**反向**量纲错配:
        #   universe 候选(inventory): 1.2 + 0.8 + 0.5*500.0 = 252.0
        #   HPO 候选(model:hpo):     0.3 + 0.0 + 0.5*300.0 = 150.1
        # `uncertainty=0.5`(手设常数)被当作 500×MDE 的"实测不确定性"
        # 计入 0.5×权重,于是**任何**信息源候选恒压过**任何**模型候选;
        # 模型/HPO 轴在整个搜索期实际不可达,gen-04 因此只投了
        # `models:catboost`(低分候选)而从未提名同算法的 `model:hpo`。
        #
        # 正确口径:每个字段都用**与自己来源同源**的刻度。
        #   - 实测后验(台账、paired-bootstrap):除以 MDE → 单位是
        #     "多少个最小有意义提升",O(1),与常数项可比;
        #   - 手设常数(information_gain/novelty/uncertainty):本就是
        #     O(1) 刻度,原样进入 score。
        # 量化:实测 +0.002(=2×MDE)得 2.0 分,与 information_gain=2.0
        # 的"未消费源"同量级 —— 正是 D3 想要的结论。
        #
        # 字段语义随之明确:调用方给的 `uncertainty` / `expected_gain`
        # 一律按**主指标量纲**传(台账实测值,或常数=0),归一化中间量
        # 仍落盘在 `*_mde` 供审计;`potential_gain_upper_bound` 不归一化
        # (被 evolution 停止判定按主指标量纲直接与 MDE 比较)。
        #
        # 兜底(仅文档口径到位还不够 —— 契约必须可执行):调用方若把
        # 手设常数错传到 `uncertainty`(量级 O(0.1~1),远超任何真实
        # 噪声带),这里按 MDE 尺度判定并**不**让它以 MDE 单位进入
        # score——否则一个 0.5 就等于 500×MDE,任何模型/HPO 候选都会被
        # 信息源候选永久压制(这正是本任务 gen-04 的实际形态)。
        # 判定阈值 = `uncertainty_gate_mde`(默认 10,即 10×MDE),
        # 远超定向搜索在 validation 上的真实噪声上界(~0.5~2×MDE)。
        # v2.17.0:MDE 走 effective_mde(量纲感知),不再直接读绝对值旧键
        _mde = self._mde()
        _gate = float(self.config.get(
            "evaluator.uncertainty_gate_mde", 10.0) or 10.0)
        a["expected_gain_mde"] = round(a["expected_gain"] / _mde, 8)
        a["uncertainty_mde"] = round(a["uncertainty"] / _mde, 8)
        # prior 只有一个来源:显式 kwargs,或量纲守卫从 uncertainty
        # 误传值里救回来的常数(两者取大,不重复计入)。
        # 关键:prior 档下 `uncertainty_mde` **不得**再进 score —— 否则
        # 0.5 既按常数计一次、又按 500×MDE 计一次,正是本 bug 的形态
        # 在守卫自身复现。
        _prior = float(uncertainty_prior)
        if a["uncertainty"] > _gate * _mde:
            a["uncertainty_scale"] = "prior"
            _prior = max(_prior, a["uncertainty"])
            _u_measured = 0.0
        else:
            a["uncertainty_scale"] = "measured"
            _u_measured = a["uncertainty_mde"]
        a["uncertainty_prior"] = round(_prior, 8)

        # v2.15.x ROI 乘性重构:
        #
        # 旧公式(加性):
        #   score = gain/MDE + 0.5·u/MDE + info + novelty + prior - cost
        # 问题:info=1.2, novelty=0.8 的手设常数 2.0 与 ROI 相加,
        # ROI=0 的候选可以被常数"加回来"。
        #
        # 新公式:
        #   roi_mult = sigmoid(gain_LCB / MDE)  ∈ (0, 1)
        #   score = gain/MDE + (info + novelty + prior + 0.5·u) × roi_mult
        #           - cost_penalty
        #
        # 语义:
        # - gain/MDE 永远按原值加性进入——已在 MDE 单位里归一,量级 O(1),
        #   代表实测的边际贡献,不应被乘以一个衰减因子;
        # - 手设常数(info/novelty/prior/uncertainty)被 roi_mult 折扣——
        #   ROI 充裕时常数不折扣(×1),ROI 归零时常数减半(×0.5),
        #   ROI 明确为负时常数趋零(×→0);
        # - 这样 gain=2×MDE 的候选比 gain=0 时额外拿到 2.0 分(加性),
        #   再把常数放大(乘性);而 gain=0 的候选常数减半但不被清零。
        #
        # gain_LCB = gain - 1.28·std(单尾 90% 置信下界):
        # 低 n 或高 std 的候选 roi_mult 主动降权。
        # 截断:gain_LCB ≥ -2·MDE 防 sigmoid 完全崩塌。
        import math as _math
        _std = float(a["uncertainty"]) if a["uncertainty_scale"] == "measured" \
            else 0.0
        _gain_lcb = float(a["expected_gain"]) - 1.28 * _std
        _gain_lcb = max(_gain_lcb, -2.0 * _mde)
        _roi_arg = _gain_lcb / _mde
        try:
            _roi_mult = 1.0 / (1.0 + _math.exp(-_roi_arg))
        except OverflowError:
            _roi_mult = 0.0 if _roi_arg < 0 else 1.0
        a["roi_multiplier"] = round(_roi_mult, 6)
        a["gain_lcb"] = round(_gain_lcb, 8)

        _const_score = (a["information_gain"] + a["novelty"]
                        + a["uncertainty_prior"] + 0.5 * _u_measured)
        a["score"] = round(
            a["expected_gain_mde"] + _const_score * _roi_mult
            - a["cost_penalty"], 8)
        # 旧字段兼容保留(消费方可能读取)
        a["score_legacy"] = round(a["expected_gain_mde"]
                                  + 0.5 * _u_measured
                                  + a["information_gain"] + a["novelty"]
                                  + a["uncertainty_prior"]
                                  - a["cost_penalty"], 8)
        proposal["semantic_hash"] = _semantic_hash(
            _proposal_identity(proposal))
        proposal["proposal_id"] = "prop-" + proposal["semantic_hash"]
        proposal["materialize_command"] = self._materialize_command(proposal)
        return proposal

    def _materialize_command(self, p: dict) -> str:
        # v2.17.0:CLI 路径取当前调用路径(保留软链写法,见 _cli_script_path)
        cli = _cli_script_path()
        parts = ["python", str(cli), "new-trial", "--run",
                 str(self.run_dir), "--generation", str(p["generation"]),
                 "--action", p["action"], "--algorithm", p["algorithm"],
                 "--hypothesis", p["hypothesis"], "--rationale",
                 p["rationale"], "--falsification", p["falsification"],
                 "--family-hints", p["family"],
                 "--planner-proposal-id", p["proposal_id"],
                 "--planner-semantic-hash", p["semantic_hash"]]
        if p["changed_modules"]:
            parts += ["--changed-modules", ",".join(p["changed_modules"])]
        # v2.18.0(Q18):主题与预填依据(命中同主题原则 → <id>:tests,否则 novel)
        # confirm 子代不预填:new-trial 让 --from-parent 子代继承父代依据
        if not (p["action"] == "confirm" and p["parent_ids"]):
            _eb = p.get("experience_basis") or {}
            if _eb.get("basis"):
                # v2.19.0(#12):经验候选直接预填所依原则(follows)与其主题
                _topics, _basis = list(_eb.get("topics") or []), _eb["basis"]
            else:
                _topics, _basis = self._proposal_topics_basis(p)
            if _topics:
                parts += ["--topics", ",".join(_topics)]
            parts += ["--basis", _basis]
            # v2.18.6:同时带上预填值。new-trial 比对 --basis 是否原样沿用,
            # 沿用的依据记 source=prefilled(证据按被动权重计,反馈不算主动引用)
            if _basis != "novel":
                parts += ["--basis-prefill", _basis]
        if p["parent_ids"]:
            parts += ["--parents", ",".join(p["parent_ids"])]
            if p["action"] == "confirm":
                parts += ["--from-parent", p["parent_ids"][0]]
            elif p["action"] == "hpo":
                # [v2.15.0] 同算法(冠军面)HPO 走 --from-parent:逐字复制
                # 冠军代码保留加宽 transform(v2.13.4 起 --hpo-parent 对加宽
                # 面直接拒绝,planner 物化命令因此每次都要人工改写);异算法
                # 候选沿用 --hpo-parent(模板渲染、继承声明面)
                _par = self.by_id.get(p["parent_ids"][0])
                _palgo = str((_par.data.get("algorithm") if _par is not None
                              else "") or "").lower()
                if _palgo and _palgo == str(p["algorithm"] or "").lower():
                    parts += ["--from-parent", p["parent_ids"][0]]
                else:
                    parts += ["--hpo-parent", p["parent_ids"][0]]
            elif (p.get("recipe") or {}).get("set_overrides"):
                # v2.17.2(A2):单变量超参修正子代(如冠军撞 cap 时的 lr×2)——
                # 复制父代代码,只经 --set 覆写超参(零写码,同特征面)
                parts += ["--from-parent", p["parent_ids"][0]]
                for _k, _v in sorted(
                        p["recipe"]["set_overrides"].items()):
                    parts += ["--set", f"{_k}={_v}"]
            elif "model" not in p["changed_modules"]:
                parts += ["--from-parent", p["parent_ids"][0],
                          "--allow-code-edit"]
        if p["action"] == "hpo":
            parts += ["--search-space", str(
                self.run_dir / "planner" /
                f"default-search-space-{p['algorithm'].lower()}.yaml"),
                "--budget-configs", str(int(
                    p.get("recipe", {}).get("budget_configs") or
                    self.config.get("evolution.hpo.default_budget", 40)))]
        if p["data_view"] != "v1":
            parts += ["--data-view", p["data_view"]]
        # [v2.13.2] screening-only 候选(宽面初筛)必须把标记带进
        # new-trial —— 否则按本命令物化出来的 trial 只是一条普通的
        # explore,会正常参与 FDR/冠军/Archive,与 recipe 声明的
        # "只产证据不参与裁决"相矛盾。
        if (p.get("recipe") or {}).get("screen_only"):
            parts += ["--screen-only"]
        return " ".join(shlex.quote(str(x)) for x in parts)

    def _search_champion_info(self) -> dict:
        """v2.17.0:搜索冠军及其配方根(confirm 提名的锚)。

        优先与交付同源:``Leaderboard.delivery_resolution``(同视图/同 epoch
        可比池 + 收缩分排序,冠军配方、交付状态、交付容差一次给齐);不可用
        时回落 ``self._parent()``(accepted/confirmed 按主指标排序)+
        :func:`confirmation.recipe_root`,容差按 ``delivery_tolerance``。
        """
        from .confirmation import delivery_tolerance, recipe_root

        info: dict = {"trial": None, "recipe": None, "metric": None,
                      "status": None, "tolerance": None, "source": None,
                      "gap": None, "delivery_trial": None}
        try:
            from .leaderboard import Leaderboard

            lb = Leaderboard(self.run_dir, self.spec, self.config)
            res = lb.delivery_resolution(self.trials) or {}
            champ = self.by_id.get(str(res.get("search_champion") or ""))
            if champ is not None:
                info.update(trial=champ,
                            recipe=res.get("search_recipe")
                            or recipe_root(champ, self.by_id),
                            metric=res.get("search_metric"),
                            status=res.get("status"),
                            tolerance=res.get("tolerance"),
                            # v2.19.0:confirm final_only 的"未交付增益"读它
                            gap=res.get("gap"),
                            delivery_trial=res.get("delivery_trial"),
                            source="leaderboard.delivery_resolution")
                return info
        except Exception:  # noqa: BLE001 — 回落 planner 自身排序
            pass
        champ = self._parent()
        if champ is None:
            return info
        ev = champ.data.get("evaluation") or {}
        pb = ((ev.get("delta") or {}).get("vs_parent_paired_bootstrap")
              or {})
        noise = pb.get("se") or ev.get("noise")
        try:
            tol = delivery_tolerance(self.config, float(noise)
                                     if noise is not None else None)
        except Exception:  # noqa: BLE001
            tol = self._mde()
        info.update(trial=champ, recipe=recipe_root(champ, self.by_id),
                    metric=(ev.get("primary_metric") or {}).get("value"),
                    tolerance=tol, source="planner._parent")
        return info

    def _confirmation_candidates(self) -> list[dict]:
        """confirm 提名(v2.17.0 只提名冠军配方或容差内的陈旧确认)。

        事故(run4 regression.2 #4):旧实现遍历全部 eligible accepted 父代,
        冠军配方已被确认后,下一个"尚未确认"的 eligible 就被提名——连续
        3 代提名 confirm trial-003(落后冠军 002/013 两百多元),它即使确认
        通过也不能交付(交付位绑定冠军配方,stale 只在容差内可用)。

        现在只提名:
        1. **搜索冠军配方**(``delivery_resolution.search_champion`` 的配方
           根)上 eligible、尚未 fresh-confirmed 的父代;
        2. 冠军配方**未**被确认时,``selection_correction.passed`` 的其它
           eligible 候选,且落后冠军 ≤ 交付容差(否则确认了也不能作为
           stale 交付),按差距升序。
        每条提名附 ``rationale`` 与 ``confirm_nomination``(角色/差距/容差);
        跳过的 eligible 父代及原因留痕在 ``self._confirm_skipped``(写进
        batch.confirm_nomination_skipped)。
        """
        from .confirmation import confirm_attempts, fresh_confirmed, recipe_root

        self._confirm_skipped = []
        # v2.14.0:每配方 confirm 完成次数上限(见 config
        # evolution.max_confirm_attempts_per_recipe)。旧版只看"是否已有
        # 通过的 confirm",失败就会被反复提名 —— 抽签式重试。
        cap = int(self.config.get(
            "evolution.max_confirm_attempts_per_recipe", 2) or 0)
        champ_info = self._search_champion_info()
        champ = champ_info.get("trial")
        if champ is None:
            return []
        champ_recipe = champ_info.get("recipe") or champ.trial_id
        champ_val = champ_info.get("metric")
        if champ_val is None:
            champ_val = ((champ.data.get("evaluation") or {})
                         .get("primary_metric") or {}).get("value")
        try:
            tol = float(champ_info.get("tolerance"))
        except (TypeError, ValueError):
            tol = self._mde()
        # 冠军配方是否已有 fresh confirm(交付位已绑定)
        # v2.20.0:先验豁免(waived_by_prior)同样视为"交付位已满足" —— 那个
        # 配方已经过"换 seed 后增益仍在"的检验,再 confirm 只是付费重跑。
        champ_bound = champ_info.get("status") in ("bound", "waived_by_prior") \
            or any(
                fresh_confirmed(t) and recipe_root(t, self.by_id) == champ_recipe
                for t in self.trials
                if str(t.data.get("search_action") or "") == "confirm")

        def _gap(t) -> float | None:
            """冠军比 t 好多少(方向感知,≥0 表示 t 落后)。"""
            v = ((t.data.get("evaluation") or {}).get("primary_metric")
                 or {}).get("value")
            try:
                g = float(champ_val) - float(v)
            except (TypeError, ValueError):
                return None
            return g if self.direction_up else -g

        def _skip(t, reason: str, **kw) -> None:
            self._confirm_skipped.append(
                {"trial_id": t.trial_id, "reason": reason, **kw})

        champion_side, stale_side = [], []
        for parent in self._ranked_trials(("accepted",)):
            eligible, _ = confirmation_parent_eligibility(parent)
            if not eligible:
                continue
            if any(t.data.get("search_action") == "confirm"
                   and parent.trial_id in t.parent_ids
                   and (t.data.get("evaluation") or {}).get("verdict") ==
                   "confirmed"
                   and (((t.data.get("evaluation") or {}).get(
                       "fresh_confirmation") or {}).get("passed") is True)
                   for t in self.trials):
                continue
            if cap > 0 and len(confirm_attempts(self.trials,
                                                parent.trial_id)) >= cap:
                _skip(parent, "confirm_attempt_cap_reached", cap=cap)
                continue
            # [v2.15.0] 父代仍有在途 HPO 子代:确认完大概率当代即被超越
            # (run3 trial-019/020),等 HPO 出结果再对届时冠军提名
            if any(str(t.data.get("search_action") or "") == "hpo"
                   and parent.trial_id in t.parent_ids
                   and t.status in ("PROPOSED", "CODE_WRITTEN", "RUNNING")
                   for t in self.trials):
                _skip(parent, "inflight_hpo_child")
                continue
            gap = _gap(parent)
            if recipe_root(parent, self.by_id) == champ_recipe:
                champion_side.append((parent, gap))
                continue
            if champ_bound:
                _skip(parent, "champion_recipe_already_bound", gap=gap,
                      champion=champ.trial_id)
                continue
            if gap is None or gap > tol:
                _skip(parent, "gap_exceeds_delivery_tolerance", gap=gap,
                      tolerance=tol, champion=champ.trial_id)
                continue
            stale_side.append((parent, gap))
        stale_side.sort(key=lambda x: (x[1], x[0].trial_id))

        out = []
        for role, (parent, gap) in (
                [("search_champion_recipe", x) for x in champion_side]
                + [("stale_within_tolerance", x) for x in stale_side]):
            _gap_s = f"{gap:+.6g}" if gap is not None else "n/a"
            if role == "search_champion_recipe":
                why = (f"{parent.trial_id} 属搜索冠军 {champ.trial_id} 的配方"
                       f"(配方根 {champ_recipe}),已过选择校正、尚未通过"
                       "独立确认;交付位必须绑定冠军配方,确认它是交付前提")
            else:
                why = (f"搜索冠军 {champ.trial_id} 的配方尚未确认;"
                       f"{parent.trial_id} 已过选择校正,落后冠军 {_gap_s}"
                       f" ≤ 交付容差 {tol:.6g},确认后可作 stale_within_"
                       "tolerance 交付位(落后超过容差的候选不提名)")
            p = self._base(
                action="confirm", family="confirmation",
                parent=parent, changed_modules=[],
                hypothesis=f"{parent.trial_id} 的增益可在独立 shadow 面复现",
                rationale=why,
                falsification=("新 seed 在 validation 或独立 shadow 上显著"
                               "不及父代(非劣下界含 seed 方差),或显著不及"
                               "父代的参照,即确认失败"),
                information_gain=1.5, novelty=0.2,
                uncertainty=0.0, uncertainty_prior=0.5,
                potential_gain_upper_bound=0.0,
                cost_hours=0.75,
                recipe={"type": "exact_replay", "seed_policy":
                        "independent_seed", "confirmation_split": "shadow_test"})
            p["confirm_nomination"] = {
                "role": role, "search_champion": champ.trial_id,
                "champion_recipe": champ_recipe,
                "gap_to_champion": gap, "delivery_tolerance": tol,
                "champion_source": champ_info.get("source")}
            out.append(p)
        return out[:2]

    #: [FIX-2026-09-18] 信息源候选的优先级/成本分档。
    # 实测事故(example-seq-v3-v2.12.2.9 / run-jt-draw-t7 gen-01):
    # 清单里两个 unconsumed 的 reparse 源(seq_d365/seq_gap_d365)与 20 个
    # partially_consumed 的 base_face_family 源被一视同仁地生成候选,
    # information_gain 一律 1.2、novelty 一律 0.8,只有 cost_hours 不同
    # (reparse 2.5h → cost_penalty 0.375)。排序键是 acquisition.score,
    # 于是 reparse 源恒比 base_face 源低 0.225 分,在 valid 列表里排到
    # 第 21/22 位(共 23),被 slots 截断全部丢弃——**唯一被标记为
    # "未消费"的信息通道,在信息优先的协议下反而永远进不了批次**。
    # 分档口径:
    #   - unconsumed reparse 源:最高信息增益(信息优先原则),且其
    #     cost_penalty 减半——重活成本是真实约束,但不该成为"零消费"
    #     状态的惩罚;重活靠 expected_resources.heavy 与超时放大处理。
    #   - partially_consumed 源:维持原有中位值(1.2/0.8/0.5)。
    #   - 同一源最多产出 2 个粒度维度的候选(见 _inventory_candidates),
    #     避免一个源吃满整代名额。
    #: [v2.13.4] 分档里的 `uncertainty` 是**常数先验系数**,通过
    #: `uncertainty_prior` 传入 `_base`(原样入 score,O(1) 刻度),
    #: 不再塞进 `uncertainty` —— 后者是**实测**不确定性的通道
    #: (主指标量纲,由方向台账 std_delta 提供)。两者混用会让常数
    #: 被当作 MDE 单位的实测值放大 1000 倍(gen-04 实测事故)。
    _INVENTORY_TIER = {
        "unconsumed": {"information_gain": 2.0, "novelty": 1.0,
                       "uncertainty_prior": 0.6, "cost_scale": 0.5},
        "partially_consumed": {"information_gain": 1.2, "novelty": 0.8,
                               "uncertainty_prior": 0.5, "cost_scale": 1.0},
    }

    def _inventory_candidates(self, parent) -> list[dict]:
        path = self.run_dir / "sample-profile" / "information-inventory.yaml"
        if not path.exists():
            return []
        inv = load_yaml(path) or {}
        sources = [s for s in (inv.get("sources") or [])
                   if s.get("consumption_status") != "fully_consumed"]
        # [FIX-2026-09-18] 未消费源优先:同分时按状态排序会让截断稳定
        # 丢掉 reparse 源;这里先按状态分档,再让每个源最多贡献 2 个
        # 粒度维度(旧实现只取 spaces[0],同一源的其余 5 个维度永远
        # 不会被提名)。
        _rank = {"unconsumed": 0, "partially_consumed": 1}
        sources.sort(key=lambda s: (_rank.get(
            str(s.get("consumption_status")), 2), str(s.get("column"))))
        out = []
        for source in sources:
            column = str(source.get("column") or source.get("family") or
                         "unknown")
            status = str(source.get("consumption_status") or
                         "partially_consumed")
            tier = self._INVENTORY_TIER.get(
                status, self._INVENTORY_TIER["partially_consumed"])
            spaces = [g for g in source.get("granularity_space") or []
                      if g.get("state") in (None, "unexplored", "partial")]
            if not spaces:
                spaces = [{"dimension": "derived_interaction",
                           "hint": source.get("note", "")}]
            heavy = source.get("source_type") == "reparse"
            for slot, space in enumerate(spaces[:2]):
                dimension = space.get("dimension") or "derived_interaction"
                hint = space.get("hint") or source.get("note", "")
                _src_consumed = self._source_already_consumed_by_champion(
                    column, dimension)
                out.append(self._base(
                    action="explore",
                    family=f"info:{column}:{dimension}",
                    parent=parent, changed_modules=["feature"],
                    hypothesis=(f"消费信息源 {column} 的 {dimension} 粒度"
                                f"可提供新增信号:{hint}"),
                    rationale=(f"清单状态={status};粒度 {dimension};{hint}"),
                    falsification="新增列质量合格但配对增益未通过统计门，"
                                  "则证伪该粒度",
                    # [v2.13.4] 兑现优先权:重跑某信息源前,应先判定它是否
                    # 已被**现有冠军**消费(v2.19.0 起按 trial 产物判定:
                    # 代码字面量引用 + 新增列仍在冠军输入,不再用列名前缀
                    # 表)。见 `_source_already_consumed_by_champion`。
                    # 实现方式:information_gain 归零(而不是加负分)——
                    # "已消费"是**证据缺失**(该粒度确实在面内、无新增量),
                    # 不是"证据为负";归零后它仍能与其它弱候选竞争,只是
                    # 不再凭信息常数压过有实测先验的模型轴。
                    information_gain=(0.0 if
                                      _src_consumed else
                                      tier["information_gain"]),
                    novelty=tier["novelty"],
                    uncertainty=0.0,
                    uncertainty_prior=tier["uncertainty_prior"],
                    expected_gain=0.0,
                    potential_gain_upper_bound=2.0 * self._mde(),
                    cost_hours=(2.5 if heavy else 1.0) * tier["cost_scale"],
                    prior_note=(
                        f"信息源 {column} 已被现有冠军消费(v2.19.0 产物"
                        "判定:血缘中 trial "
                        f"{(self._consumption_evidence(column) or {}).get('introduced_by')}"
                        " 首次在代码里引用该源,其新增列仍在冠军模型输入中);"
                        "重跑属同口径复现,information_gain 归零。若要确认该"
                        "粒度,应在冠军面上做消融/confirm,而非新建 explore。"
                        if _src_consumed else ""),
                    recipe={"type": "feature_operator", "source": column,
                            "operator": dimension,
                            "granularity_slot": slot,
                            "leak_safe_fit": "train_only"}))
        return out

    # ------------------------------------------------------------------
    # v2.19.0(#10):画像规则表 → profile_rule 候选
    # ------------------------------------------------------------------
    def _profile_rules(self) -> dict:
        """本 run 的画像规则命中(runtime.profile_rules.evaluate,进程内缓存)。"""
        cached = getattr(self, "_profile_rules_cache", None)
        if cached is None:
            try:
                from .profile_rules import evaluate

                cached = evaluate(self.run_dir, self.spec)
            except Exception:  # noqa: BLE001 — 画像规则失败不阻断规划
                cached = {"facts": {}, "shapes": {}, "hits": []}
            self._profile_rules_cache = cached
        return cached

    def _available_algorithms(self) -> set:
        """run-state.env_manifest 中可导入的包名(无环境证据返回空集)。"""
        state_path = self.run_dir / "run-state.json"
        env = ((load_json(state_path) or {}).get("env_manifest") or {}) \
            if state_path.exists() else {}
        return {str(p.get("name") or "").lower()
                for p in env.get("packages") or []
                if p.get("import_ok") is True}

    def _family_posterior(self, family: str) -> dict | None:
        """同 family 在本 run 的实测后验(v2.19.0):有则替代画像/经验先验。

        证据来源(先直接、后间接):
          1. 本 run 已评估、非 confirm/screening 的 trial,其 family_hints 首项
             (归一后)等于该 family —— 由本 planner 物化的候选都会带上它;
          2. 方向台账中归一键相等的 arm(mean_gain/std_gain/n_tried)。
        返回 {n, mean_gain, std_gain, source};无证据返回 None。
        """
        key = _norm_family(family)
        gains = []
        for t in self.trials:
            if str(t.data.get("search_action") or "") == "confirm" or \
                    t.data.get("screen_only"):
                continue
            hints = t.data.get("family_hints")
            if isinstance(hints, str):
                hints = [hints]
            if not hints or _norm_family(hints[0]) != key:
                continue
            ev = t.data.get("evaluation") or {}
            if (ev.get("integrity") or {}).get("valid", True) is False:
                continue
            g = self._gain((ev.get("delta") or {}).get("vs_parent"))
            if g is not None:
                gains.append(g)
        if gains:
            n = len(gains)
            mean = sum(gains) / n
            std = (sum((g - mean) ** 2 for g in gains) / (n - 1)) ** 0.5 \
                if n > 1 else 0.0
            return {"n": n, "mean_gain": mean, "std_gain": std,
                    "source": "trials.family_hints"}
        ledger = self._direction_ledger()
        if ledger is not None:
            for fam, arm in ((ledger.state or {}).get("arms") or {}).items():
                if not isinstance(arm, dict) or _norm_family(fam) != key:
                    continue
                n = int(arm.get("n_tried") or 0)
                if n <= 0:
                    continue
                return {"n": n,
                        "mean_gain": float(arm.get("mean_gain",
                                                   arm.get("mean_delta"))
                                           or 0.0),
                        "std_gain": float(arm.get("std_gain",
                                                  arm.get("std_delta"))
                                          or 0.0),
                        "source": f"directions.json:{fam}"}
        return None

    def _profile_rule_candidates(self, parent) -> list[dict]:
        """画像规则命中物化为候选(来源 ``profile_rule``)。

        expected_gain = prior_gain_mde × MDE(规则表常量,待 benchmark 校准);
        同 family 已有实测(`_family_posterior`)时用后验均值/标准差替代先验。
        序列源已被冠军消费(`_source_already_consumed_by_champion`)时先验减半、
        information_gain 归零 —— 宽面算子仍可能有增量,但不再凭常数抢槽位。
        """
        if not self.config.get("planner.profile_rules", True):
            return []
        res = self._profile_rules()
        avail = self._available_algorithms()
        mde = self._mde()
        out = []
        for hit in res.get("hits") or []:
            for sg in hit.get("suggestions") or []:
                if not sg.get("materialize", True):
                    continue
                algo = sg.get("algorithm")
                if algo and avail and algo not in avail:
                    continue
                prior_mde = float(sg.get("prior_gain_mde") or 0.0)
                info = PROFILE_RULE_TIER["information_gain"]
                novelty = PROFILE_RULE_TIER["novelty"]
                notes = [f"画像规则 {hit.get('rule')} 命中;先验 "
                         f"{prior_mde:g}×MDE(待 benchmark 校准)"]
                col = sg.get("source_column")
                if col and self._source_already_consumed_by_champion(
                        col, "sequence_operator_family"):
                    prior_mde *= 0.5
                    info = 0.0
                    notes.append(f"{col} 已被冠军消费(引入者 "
                                 f"{(self._consumption_evidence(col) or {}).get('introduced_by')}"
                                 "),先验减半、information_gain 归零")
                expected, unc = prior_mde * mde, 0.0
                post = self._family_posterior(sg["family"])
                if post is not None:
                    expected = float(post["mean_gain"])
                    unc = float(post["std_gain"])
                    novelty = 0.0
                    notes.append(f"同 family 已有实测 n={post['n']}"
                                 f"({post['source']}),后验 mean="
                                 f"{expected:+.6g} 替代先验")
                ev = hit.get("evidence") or {}
                p = self._base(
                    action=sg.get("action") or "explore",
                    family=sg["family"], parent=parent,
                    algorithm=algo,
                    changed_modules=list(sg.get("changed_modules")
                                         or ["feature"]),
                    hypothesis=(f"画像显示 {hit.get('rule')}:{sg.get('hint')};"
                                f"预期增益约 {prior_mde:g}×MDE"),
                    rationale=("画像规则表命中,证据="
                               + json.dumps({k: v for k, v in ev.items()
                                             if k not in ("thresholds",
                                                          "top")},
                                            ensure_ascii=False,
                                            default=str)[:300]),
                    falsification=("相对父代配对增益未通过选择校正(或 < MDE)"
                                   "则该画像规则对本数据不成立"),
                    information_gain=info, novelty=novelty,
                    expected_gain=expected, uncertainty=unc,
                    uncertainty_prior=PROFILE_RULE_TIER["uncertainty_prior"],
                    potential_gain_upper_bound=max(expected, 0.0)
                    + 2.0 * unc,
                    cost_hours=float(sg.get("cost_hours") or 1.0),
                    prior_note=";".join(notes),
                    recipe={"type": "profile_rule", "rule": hit.get("rule"),
                            "operator": sg.get("operator") or sg.get("key"),
                            "source": col,
                            "columns": list(sg.get("columns") or [])[:20],
                            "leak_safe_fit": "train_only"})
                p["profile_rule"] = {"rule": hit.get("rule"),
                                     "id": hit.get("id"), "evidence": ev,
                                     "prior_gain_mde": float(
                                         sg.get("prior_gain_mde") or 0.0),
                                     "posterior": post,
                                     "advisory": hit.get("advisory")}
                out.append(p)
        return out

    # ------------------------------------------------------------------
    # v2.19.0(#12):经验库 direction=+1 原则 → experience 候选
    # ------------------------------------------------------------------
    def _experience_candidates(self, parent) -> list[dict]:
        """从本 run 固定的经验库视图物化"该先试什么"的候选(来源 ``experience``)。

        入选条件(全部满足):direction=+1;类别可物化(特征工程/架构/HPO);
        检索档 L0~L2 且口径一致(scope_match=exact);检索分 ≥ 阈值
        (planner.experience_min_score,默认 EXPERIENCE_MIN_SCORE);证据后验
        不是 mixed/contradicted;本 run 未有同主题被拒的 trial;本 run 尚无
        trial 已引用该原则。预填 ``--basis <id>:follows`` 与 ``--topics``。
        prior_gain_mde = EXPERIENCE_PRIOR_SCALE_MDE × 证据后验 × 档折扣。
        library.enabled=false / 视图为空 → 返回 []。
        跳过原因写 ``self._experience_skipped``(落 batch 供审计)。
        """
        self._experience_skipped = []
        if self.config.get("library.enabled", True) is False:
            return []
        view = self._library_view()
        if view is None or not view.principles or \
                view.data.get("reason") == "library_disabled":
            return []
        try:
            from .library import retrieval as _rt
        except Exception:  # noqa: BLE001
            return []
        pos = {p["principle_id"]: p for p in view.principles
               if p.get("direction") == 1
               and p.get("category") in EXPERIENCE_CATEGORIES}
        if not pos:
            return []
        q_topics = sorted({str(t) for p in pos.values()
                           for t in p.get("topics") or []})
        try:
            cards = _rt.search(view.data, topics=q_topics,
                               scope=view.data.get("scope"), atid=view.atid,
                               k=max(len(pos), 8), directions=(1,),
                               tiers=("L0", "L1", "L2"))
        except Exception:  # noqa: BLE001
            return []
        try:
            min_score = float(self.config.get("planner.experience_min_score",
                                              EXPERIENCE_MIN_SCORE))
        except (TypeError, ValueError):
            min_score = EXPERIENCE_MIN_SCORE
        # 本 run 已失败的主题 / 已引用的原则
        failed_topics, used_ids = set(), set()
        for t in self.trials:
            for eid in t.data.get("experience_ids") or []:
                used_ids.add(str(eid))
            for ref in ((t.data.get("basis") or {}).get("refs") or []):
                if isinstance(ref, dict) and ref.get("id"):
                    used_ids.add(str(ref["id"]))
            ev = t.data.get("evaluation") or {}
            if ev.get("verdict") == "rejected" and \
                    str(t.data.get("search_action") or "") != "confirm":
                failed_topics.update(str(x) for x in t.data.get("topics")
                                     or [])
        resolver = view.resolver()
        failed_topics = {resolver.get(x.lower(), x) for x in failed_topics}
        mde = self._mde()
        out = []
        for c in cards:
            pid = c["principle_id"]
            pr = pos.get(pid)
            if pr is None:
                continue
            why = None
            st = view.stats.get(pid) or {}
            if c.get("score", 0) < min_score:
                why = f"score {c.get('score')} < {min_score}"
            elif c.get("scope_match") != "exact":
                why = f"口径不一致({c.get('scope_match')})"
            elif st.get("confidence") in ("mixed", "contradicted"):
                why = f"证据后验 {st.get('confidence')}"
            elif pid in used_ids:
                why = "本 run 已有 trial 引用该原则"
            else:
                c_topics = {resolver.get(str(x).lower(), str(x))
                            for x in c.get("topics") or []}
                if c_topics & failed_topics:
                    why = ("同主题已在本 run 失败:"
                           + ",".join(sorted(c_topics & failed_topics)))
            if why:
                self._experience_skipped.append(
                    {"principle_id": pid, "tier": c.get("tier"),
                     "score": c.get("score"), "reason": why})
                continue
            if len(out) >= EXPERIENCE_MAX_CANDIDATES:
                self._experience_skipped.append(
                    {"principle_id": pid, "reason": "per_generation_cap"})
                continue
            action, modules = EXPERIENCE_CATEGORIES[pr["category"]]
            posterior = float(st.get("posterior", 0.5) or 0.0)
            prior_mde = (EXPERIENCE_PRIOR_SCALE_MDE * posterior
                         * EXPERIENCE_TIER_DISCOUNT.get(c.get("tier"), 0.5))
            family = f"exp:{pid}"
            expected, unc, novelty = prior_mde * mde, 0.0, 0.5
            post = self._family_posterior(family)
            if post is not None:
                expected, unc, novelty = (float(post["mean_gain"]),
                                          float(post["std_gain"]), 0.0)
            objs = [str(o) for o in (pr.get("objects") or [])][:8]
            claim = str(pr.get("claim") or "")
            p = self._base(
                action=action, family=family, parent=parent,
                changed_modules=list(modules),
                hypothesis=(f"沿经验原则 {pid}({c.get('tier')} "
                            f"{c.get('tier_label', '')})在本 run 复现增益:"
                            f"{claim[:160]}"),
                rationale=(f"经验库 direction=+1,检索分 {c.get('score')},"
                           f"证据后验 {posterior:.3f}({st.get('confidence') or c.get('confidence')})"
                           + (f";作用对象 {', '.join(objs)}" if objs else "")),
                falsification=("相对父代配对增益未通过选择校正则该原则在本任务"
                               "不成立(反馈为 refute)"),
                information_gain=0.5, novelty=novelty,
                expected_gain=expected, uncertainty=unc,
                uncertainty_prior=0.3,
                potential_gain_upper_bound=max(expected, 0.0) + 2.0 * unc,
                cost_hours=1.0,
                prior_note=(f"经验候选:prior={prior_mde:.3g}×MDE"
                            f"(刻度 {EXPERIENCE_PRIOR_SCALE_MDE}×后验×档折扣,"
                            "待 benchmark 校准)"
                            + (f";本 run 已有实测 n={post['n']},后验替代先验"
                               if post else "")),
                recipe={"type": "experience_principle", "principle_id": pid,
                        "objects": objs, "leak_safe_fit": "train_only"})
            p["experience_basis"] = {
                "principle_id": pid, "basis": f"{pid}:follows",
                "topics": list(c.get("topics") or []), "tier": c.get("tier"),
                "score": c.get("score"), "posterior": posterior,
                "prior_gain_mde": round(prior_mde, 6)}
            # 重新生成物化命令(带 --basis <id>:follows / --topics)
            p["materialize_command"] = self._materialize_command(p)
            out.append(p)
        return out

    def _direction_ledger(self):
        """方向台账(v2.17.0:统一入口,带旧台账口径兜底)。

        v2.17.0 起 directions.json 的 mean_delta/lcb_delta/dph 是方向感知
        增益(顶层 ``gain_convention``)。v2.17.0 之前 refresh 的台账是原始
        Δ:越大越好的任务两者相同;越小越好的任务(rmse/mae…)若读旧台账,
        "变差"会被当增益(run4 regression.2 的 feat:amt2)。此时从全量 trial
        在内存里按新口径重建(``persist=False``,不改盘上文件,下一次
        finalize-generation 会正式重写)。
        """
        if getattr(self, "_ledger_cache", None) is not None:
            return self._ledger_cache or None
        path = self.run_dir / "directions.json"
        if not path.exists():
            self._ledger_cache = False
            return None
        ledger = DirectionLedger(self.run_dir)
        state = load_json(path) or {}
        if not state.get("gain_convention") and not self.direction_up:
            try:
                state = ledger.refresh(self.trials, self.spec, persist=False)
                state["rebuilt_in_memory_for_gain_convention"] = True
            except Exception:  # noqa: BLE001 — 重建失败沿用旧台账
                pass
        ledger.state = state
        self._ledger_cache = ledger
        return ledger

    def _direction_candidates(self, parent, slots: int) -> list[dict]:
        ledger = self._direction_ledger()
        if ledger is None:
            return []
        alloc = ledger.ucb_allocation(max(1, slots))
        state = ledger.state or {}
        out = []
        for family, n in alloc.items():
            # v2.18.4:`replicate:seed` 是"同一配方的换 seed 复测"桶,其 Δ
            # 就是 seed 噪声的实测值(台账用途:给其它 arm 当噪声标尺);
            # `ablate:<族>` 是摘块臂,Δ 符号与"加该块"相反。两者都不是
            # 可强化的特征方向,不能生成 exploit 提案。
            if str(family).startswith(("replicate:", "ablate:")):
                continue
            arm = (state.get("arms") or {}).get(family) or {}
            # [FIX-2026-09-19 D1] 旧实现 `for _ in range(min(n, 2))` 对
            # 同一 arm 生成两个字段完全相同的 proposal → 语义 hash 与
            # proposal_id 相同 → 第二个被 plan() 判 duplicate_semantics,
            # 同一 proposal_id 同时出现在 proposals[] 与 rejected[] 的
            # 同名候选里(实测 proposal-batch-gen-02)。UCB 槽位数 n 只
            # 决定"该方向是否入选",不再复制 proposal;分档权重交给
            # acquisition score(台账 mean_delta/std 已随 D3 归一化进
            # 排序),不在这里硬造重复候选。每个 arm 只产 1 个候选。
            # v2.17.0:mean_delta 为方向感知增益(越大越好)
            module = "model" if family.startswith("model:") else "feature"
            _mean_gain = float(arm.get("mean_gain", arm.get("mean_delta"))
                               or 0.0)
            _std_gain = float(arm.get("std_gain", arm.get("std_delta"))
                              or 0.0)
            out.append(self._base(
                action="exploit", family=family, parent=parent,
                changed_modules=[module],
                hypothesis=f"沿 {family} 方向做单变量强化可继续获得边际增益",
                rationale=(f"方向台账 增益/hour={arm.get('delta_per_hour', 0)},"
                           f" mean_gain={_mean_gain:+.6g},"
                           f" n={arm.get('n_tried', 0)}, UCB 槽位={n}"),
                falsification="相对父代配对增益未通过选择校正即停止加码",
                expected_gain=_mean_gain,
                uncertainty=_std_gain,
                potential_gain_upper_bound=max(
                    0.0, _mean_gain + 2.0 * _std_gain),
                novelty=0.1, cost_hours=max(
                    0.5, float(arm.get("cost_hours") or 0.5)
                    / max(int(arm.get("n_tried") or 1), 1)),
                recipe={"type": "direction_refinement", "arm": family,
                        "one_variable_change": True}))
        return out

    def _falsified_families(self) -> set:
        """[FIX-2026-09-19 D2] 方向台账已证伪的 family 集合。

        跨代重复提名已证伪方向的实测事故:候选生成不查台账负面结果,
        且去重身份含 parent_ids——冠军换代后同一 family 身份变化即
        "新候选"(gen-03 重提 3 个 gen-01 已判负的 family,每条都是
        纯浪费槽位)。规则:n_tried >= FALSIFIED_MIN_TRIED 且
        n_accepted == 0 且 mean_delta <= 0 → falsified。只拦
        explore/exploit 谱系;confirm/hpo/model 族候选不受影响
        (confirm 是交付门不是方向加码;hpo/model 族由特征期门
        与算法换族逻辑另行管理)。

        v2.17.0:mean_delta 为方向感知增益(rmse 任务里 Δ=−50 的 seq 方向
        不再被误判证伪,+10.7 的变差方向才是负增益);n_accepted 由台账按
        parent_verdict 优先口径计数。

        v2.17.2(D1,run4 req-t7-400w 回写):`feat:info:seq_gap_d365:recency`
        两次未过门(n_accepted 0,mean +3.9e-5 ≈ 0.04×MDE)仍被 gen-04/06
        续投 —— mean>0 让旧判据放行。在 ``n_tried_eff >= FALSIFIED_MIN_TRIED
        且 n_accepted == 0`` 的前提下,以下任一成立即证伪:
          - ``mean_gain <= 0``(旧判据);
          - ``lcb_gain < 0``(台账保守下界为负;旧 stub 无 lcb 键时跳过);
          - ``UCB = mean + z·std/√n_eff < MDE``(乐观估计也够不到最小有意义
            提升;n_eff 缺省按台账同口径 ρ=0.7 折减)。
        ``n_tried_eff = n_tried − n_cap_confounded``(撞 cap 混杂的试次不是
        方向证据,见 directions.py)。返回台账原始键(调用方比较前经
        :func:`_norm_family` 归一);判据明细落 ``self._falsified_detail``。
        """
        self._falsified_detail = {}
        ledger = self._direction_ledger()
        if ledger is None:
            return set()
        state = ledger.state or {}
        out = set()
        _mde = self._mde()
        for family, arm in (state.get("arms") or {}).items():
            if not isinstance(arm, dict):
                continue
            n_tried = int(arm.get("n_tried") or 0) - int(
                arm.get("n_cap_confounded") or 0)
            n_accepted = int(arm.get("n_accepted") or 0)
            mean_gain = float(arm.get("mean_gain", arm.get("mean_delta"))
                              or 0.0)
            if n_tried < FALSIFIED_MIN_TRIED or n_accepted != 0:
                continue
            rule = None
            if mean_gain <= 0:
                rule = "mean_gain<=0"
            else:
                _lcb = arm.get("lcb_gain", arm.get("lcb_delta"))
                if isinstance(_lcb, (int, float)) and float(_lcb) < 0:
                    rule = "lcb_gain<0"
                else:
                    std = float(arm.get("std_gain", arm.get("std_delta"))
                                or 0.0)
                    n_eff = arm.get("n_eff")
                    if not isinstance(n_eff, (int, float)) or n_eff <= 0:
                        n_eff = n_tried / (1.0 + 0.7 * max(0, n_tried - 1))
                    ucb = mean_gain + FALSIFIED_UCB_Z * std / max(
                        float(n_eff), 1.0) ** 0.5
                    if ucb < _mde:
                        rule = "ucb_gain<mde"
            if rule:
                out.add(str(family))
                self._falsified_detail[str(family)] = {
                    "rule": rule, "n_tried_eff": n_tried,
                    "n_accepted": n_accepted,
                    "mean_gain": round(mean_gain, 8),
                    "lcb_gain": arm.get("lcb_gain", arm.get("lcb_delta")),
                    "n_cap_confounded": int(arm.get("n_cap_confounded")
                                            or 0)}
        return out

    # ------------------------------------------------------------------
    # v2.17.2(D1):avoid / do_not_retry 经验卡参与候选过滤
    # ------------------------------------------------------------------
    def _library_view(self):
        """本 run 固定水位的经验库视图(只读本地缓存,不访问服务)。"""
        cached = getattr(self, "_view_cache", None)
        if cached is None:
            try:
                from . import library_view

                cached = library_view.load(self.run_dir)
            except Exception:  # noqa: BLE001
                cached = False
            self._view_cache = cached
        return cached or None

    def _experience_avoid_targets(self) -> dict:
        """同任务"避免"原则的作用对象:{归一对象: [(原则 id, 类别)]}。

        事故(run4 req-t7-400w):`derived_interaction` 在 gen-01/02/04/06 被反复提名,
        而经验库里同任务的避免结论早已写明"加列方向已饱和"—— 经验只渲染给人读、
        不参与 plan() 过滤。v2.18.0:读经验库视图中 direction=-1、与本 run 同任务
        (原则证据含本任务指纹)、口径一致的原则,以其 ``objects``(列名 / 列前缀)
        作匹配键;无增益(direction=0)与被动证据有分歧(mixed / contradicted)的
        原则不拦。视图为空(服务不可达且无缓存)→ 空,不阻断规划。
        """
        cached = getattr(self, "_avoid_cache", None)
        if cached is not None:
            return cached
        out: dict[str, list] = {}
        try:
            view = self._library_view()
            if view is not None and view.atid:
                from .library import retrieval as _rt

                canon = _rt.canon_fn(view.data)
                for pr in view.principles:
                    if pr.get("direction") != -1 or \
                            pr.get("category") not in self._AVOID_CATEGORY_MODULES:
                        continue
                    if _rt.base_relation(view.atid, pr, canon) != "same_task" or \
                            not _rt.scope_matches(view.data.get("scope"), pr.get("scope")):
                        continue
                    if (view.stats.get(pr["principle_id"]) or {}).get("confidence") in \
                            ("mixed", "contradicted"):
                        continue
                    item = (pr["principle_id"], pr.get("category") or "")
                    keys = [_norm_family(raw) for raw in pr.get("objects") or []]
                    keys = [k for k in keys if k and k not in _GENERIC_FAMILY_TOKENS]
                    if not keys and pr.get("topics"):
                        # 没有具体对象的避免原则按主主题拦(如"派生交互已饱和"拦全部
                        # info:<列>:derived_interaction 候选;run4 req-t7-400w 事故)
                        keys = [f"topic:{pr['topics'][0]}"]
                    for key in keys:
                        out.setdefault(key, [])
                        if item not in out[key]:
                            out[key].append(item)
        except Exception:  # noqa: BLE001 — 经验库不可读不阻断规划
            out = {}
        self._avoid_cache = out
        return out

    def _log_prefill(self, topics: list[str], cards: list[dict]) -> None:
        """预填依据写一条遥测(kind=planner-prefill):程序自动给的依据不算"Agent 看过",
        反馈指标里不能把它当成漏召回,但要看得到它展示过什么。"""
        try:
            import json as _json
            from .logging_utils import now_iso as _now

            run_dir = getattr(self, "run_dir", None)
            if run_dir is None:
                return
            with open(Path(run_dir) / "experience-retrieval-telemetry.jsonl", "a",
                      encoding="utf-8") as tf:
                tf.write(_json.dumps({
                    "at": _now(), "kind": "planner-prefill", "topics": list(topics),
                    "shown_ids": [c["principle_id"] for c in cards],
                    "tiers": [c.get("tier") for c in cards]}, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001  遥测失败不影响 plan
            pass

    def _proposal_topics_basis(self, p: dict) -> tuple[list[str], str]:
        """候选的主题与预填依据(Q18):命中同主题原则时预填 ``<id>:tests``,否则
        ``novel``。Agent 可改写;依据由 new-trial 校验。"""
        view = self._library_view()
        if view is None or not view.principles:
            return [], "novel"
        try:
            from .library.evidence import infer_trial_topics

            topics, _objs = infer_trial_topics(
                {"family_hints": [p.get("family")] if p.get("family") else [],
                 "changed_modules": p.get("changed_modules") or [],
                 "action": p.get("action")}, view.resolver())
            if not topics:
                return [], "novel"
            cards = view.search(topics=topics, k=1)
            self._log_prefill(topics, cards)
            # 只有精确档(同任务 / 近似任务)才预填依据;放宽档的结论交给 Agent 判断
            if cards and cards[0]["score"] >= 0.5 and cards[0].get("tier", "L0") in ("L0", "L1"):
                return topics, f"{cards[0]['principle_id']}:tests"
            return topics, "novel"
        except Exception:  # noqa: BLE001
            return [], "novel"

    #: avoid 卡类别 → 可拦截的候选模块(架构/评估类卡的 target 常是源列名,
    #: 如 `seq_d365` 的"行尾为最近端"架构提示,不是"别再探索该源")
    _AVOID_CATEGORY_MODULES = {
        "feature_engineering": {"feature", "sample", "data"},
        "hpo": {"model", "loss"},
        "architecture": {"model", "loss"},
    }

    def _experience_avoid_hit(self, proposal: dict) -> list[str]:
        """候选 family 命中的 avoid 卡 id(全键或 `:` 分段精确相等)。

        `info:seq_gap_d365:recency` 的匹配键 = 全键 + {seq_gap_d365,
        recency};`derived_interaction` 卡因此命中所有
        `info:<col>:derived_interaction` 候选。前缀族 target(如
        `offline_inner_`)不做前缀展开 —— 宁可漏拦不可误拦。卡的类别须与
        候选改动模块相符(feature_engineering ↔ feature/sample/data,
        hpo/architecture ↔ model/loss;无类别的卡不限);evaluation 等其它
        类别的卡不拦。
        """
        targets = self._experience_avoid_targets()
        if not targets:
            return []
        mods = {str(m) for m in (proposal.get("changed_modules") or [])}
        fam = _norm_family(proposal.get("family"))
        keys = {fam} | {seg.rstrip("_") for seg in fam.split(":")
                        if seg and seg not in _GENERIC_FAMILY_TOKENS}
        view = self._library_view()
        if view is not None and proposal.get("family"):
            from .library.evidence import infer_trial_topics

            _tp, _ = infer_trial_topics({"family_hints": [proposal.get("family")]},
                                        view.resolver())
            keys |= {f"topic:{t}" for t in _tp}
        hits: list[str] = []
        for k in sorted(keys):
            for eid, cat in targets.get(k) or []:
                if cat and not (self._AVOID_CATEGORY_MODULES.get(cat, set())
                                & mods):
                    continue
                if eid not in hits:
                    hits.append(eid)
        return hits

    def _super_trial_candidate(self, parent) -> list[dict]:
        """[v2.13.2] screening-only 宽面初筛候选(仅 super_trial=true 时产出)。

        背景:`feature_exploration.super_trial` 自 v10 起就是配置键,
        但 runtime **零引用**(纯摆设)。它的语义是审计文档里的三级漏斗
        第一级 —— 把所有候选特征族一次性加宽,用 group_importance 的
        族级置换 delta 给后续 bundle 规划提供**先验**,从而回答
        "gen-02 该拆哪几族、哪些族可以判负",而不是靠 Agent 猜。

        它必须与普通 bundle 严格区分:宽面混合了全部族,其 vs_parent
        不可归因到任何单一方向,因此不进 FDR/alpha 台账、不计 accepted、
        不作冠军/Archive/父代、不入方向台账 —— 这些由 trial.yaml 的
        screen_only 标记驱动(见 runtime/trial.py:trial_is_screening)。

        默认 super_trial=false → 本函数返回 [],默认路径行为不变。
        """
        if not self.config.get("feature_exploration.super_trial", False):
            return []
        if self._feature_generations_done() >= int(
                self.config.get("feature_exploration.min_generations", 2)):
            # 只在特征期提名:特征期满后宽面初筛的先验已无消费方
            # (bundle 规划发生在特征期)。
            return []
        return [self._base(
            action="explore", family="screen:wide",
            parent=parent, changed_modules=["feature"],
            hypothesis=("一次性加宽全部候选特征族,用族级置换 delta 建立"
                        "方向先验(哪些族承载增量、哪些可判负),"
                        "供后续 bundle 拆分与消融选题"),
            rationale=("feature_exploration.super_trial=true;"
                       "宽面混合全部族,分数不可归因到单一方向,"
                       "故标记 screening-only:只产证据不参与裁决"),
            falsification="族级置换 delta 全部落在噪声内且无族可区分时,"
                          "初筛不提供方向信息(该结论本身仍是证据)",
            information_gain=2.0, novelty=1.0,
            uncertainty=0.0, uncertainty_prior=0.8,
            expected_gain=0.0,
            potential_gain_upper_bound=2.0 * self._mde(),
            cost_hours=3.0,
            recipe={"type": "feature_operator", "operator": "widen_all",
                    "screen_only": True, "leak_safe_fit": "train_only"})]

    def _champion_face(self):
        """[v2.13.4] 现有 champion 的(代码文本, 加宽统计, panel)。

        用于两件事:判定"某信息源是否已被冠军消费"、以及让模型轴候选
        的排序反映**实现价值**(冠军面 HPO ≫ 异算法 HPO,见
        `_model_candidates`)。所有读取都走 try/except —— 规划器在
        创建期/测试夹具下可能拿不到这些文件,拿不到就退化为旧行为。
        """
        cache = getattr(self, "_champion_face_cache", None)
        if cache is not None:
            return cache
        champ = self._parent()
        code, stats, panel = "", {}, None
        if champ is not None and champ.trial_dir is not None:
            try:
                code = (Path(champ.trial_dir) / "code" / "train.py"
                        ).read_text(encoding="utf-8")
            except Exception:  # noqa: BLE001
                code = ""
            try:
                meta = load_json(Path(champ.trial_dir) / "metrics.json") or {}
                stats = meta.get("widen_stats") or {}
            except Exception:  # noqa: BLE001
                stats = {}
            panel = self._champion_feature_panel(champ, code)
        self._champion_face_cache = (code, stats, panel)
        return self._champion_face_cache

    def _champion_feature_panel(self, champ, code: str):
        """冠军模型实际消费的特征列名(从其 model/feature_columns.json)。

        判定"特征面是谁"的唯一可靠依据:config 声明面只含基础列,
        加宽列是 transform 的运行时产物 —— 只有模型产物记着真实输入。
        """
        if not code:
            return None
        cols = self._model_feature_columns(champ)
        return set(cols) if cols else None

    @staticmethod
    def _model_feature_columns(t) -> list[str] | None:
        """trial 模型实际消费的列(model/feature_columns.json);缺失返回 None。"""
        if t is None or getattr(t, "trial_dir", None) is None:
            return None
        try:
            cols = load_json(Path(t.trial_dir) / "model" /
                             "feature_columns.json")
        except Exception:  # noqa: BLE001
            return None
        if isinstance(cols, list) and cols:
            return [str(c) for c in cols]
        return None

    def _model_n_features(self, t) -> int:
        """[v2.17.2 A5] "同面"判定用的真实模型列数。

        事故(run4 req-t7-400w):`--from-parent` 复制来的旧代码把
        ``metrics.n_features`` 记成基础列数(700),而模型实际消费 1647 列
        (700 裁剪基础面 + seqlag_/seqt_ 块);按 n_features 判"同面"会把
        跑在不同特征面上的 HPO 当成等价、或反之。优先取
        ``model/feature_columns.json`` 的长度(与 `_champion_feature_panel`
        同源),文件缺失才回落 ``metrics.n_features``;都拿不到返回 0。
        """
        cols = self._model_feature_columns(t)
        if cols:
            return len(cols)
        try:
            meta = load_json(Path(t.trial_dir) / "metrics.json") or {}
            return int(meta.get("n_features") or 0)
        except Exception:  # noqa: BLE001
            return 0

    def _source_already_consumed_by_champion(self, column: str,
                                             dimension: str) -> bool:
        """该信息源是否已被**现有冠军**消费(v2.19.0 起由 trial 产物判定)。

        事故形态(v2.13.4,req-t7-drawdown-v21331 gen-04):冠军已是某序列源的
        全宽解析,planner 仍提名该源的 explore,代码复制父代后列集完全相同 ->
        Δparent 恒 0。v2.13.4 用 `{seq_d365: sq_}` 前缀表 + `seq*` 推导判定,
        换一份数据(列不叫 seq*、派生列不带 sq_)即失效;v2.19.0 删除。

        现判据(纯数据,不看任何列名约定):沿冠军血缘(冠军及其祖先)找
        **引入**该源的 trial T:
          1. T 的代码(code/*.py)以字符串字面量引用了源列名,且 T 的父代
             代码没有引用(= T 首次读入该源;--from-parent 复制来的代码不算
             引入);
          2. T 的 model/feature_columns.json 相对父代有新增列(父代缺列清单时
             相对基础面 splits/train.parquet 的列);
          3. 这批新增列至少有一列仍在冠军的模型输入里。
        三条全中 -> 该源已被消费。选这个方案而不是"new-trial 登记 consumes":
        后者依赖 Agent 自报且历史 trial 没有该字段;代码字面量 + 列集差集两项
        都是 trial 落盘产物,确定性可复算。只判源级(不区分粒度维度,与旧口径
        一致);拿不到证据一律返回 False(宁可多跑一次,不误抑制)。
        """
        cache = getattr(self, "_consumed_cache", None)
        if cache is None:
            cache = self._consumed_cache = {}
        if column in cache:
            return bool(cache[column])
        cache[column] = None
        champ = self._parent()
        panel = self._champion_face()[2] if champ is not None else None
        if champ is None or not panel:
            cache[column] = False
            return False
        base_face = self._base_face_columns()
        seen, stack = set(), [champ]
        while stack:
            t = stack.pop()
            if t is None or t.trial_id in seen:
                continue
            seen.add(t.trial_id)
            parents = [self.by_id.get(pid) for pid in t.parent_ids]
            parents = [p for p in parents if p is not None]
            stack.extend(parents)
            if not self._trial_references(t, column):
                continue
            par = parents[0] if parents else None
            if par is not None and self._trial_references(par, column):
                continue        # 复制来的引用,引入者在更早的祖先
            cols = self._model_feature_columns(t)
            if not cols:
                continue
            ref = self._model_feature_columns(par) if par is not None \
                else None
            ref = set(ref) if ref else base_face
            if not ref:
                continue
            new_in_champ = [c for c in cols if c not in ref and c in panel]
            if new_in_champ:
                cache[column] = {"introduced_by": t.trial_id,
                                 "n_new_cols_in_champion": len(new_in_champ),
                                 "sample": new_in_champ[:5]}
                return True
        cache[column] = False
        return False

    def _consumption_evidence(self, column: str) -> dict | None:
        """`_source_already_consumed_by_champion` 命中时的证据(审计用)。"""
        v = (getattr(self, "_consumed_cache", None) or {}).get(column)
        return v if isinstance(v, dict) else None

    @staticmethod
    def _trial_references(t, column: str) -> bool:
        """trial 代码(code/*.py)是否以字符串字面量引用了某列。

        不扫 config/*.yaml:trial 配置的 features.excluded / reparse 清单在
        **每个** trial(含 baseline)里都列着全部 reparse 源(run4 实测),
        按配置判"引用"会让引入者恒为 baseline、判定恒假。
        """
        tdir = getattr(t, "trial_dir", None)
        if tdir is None:
            return False
        lit = _source_literal_re(column)
        for f in (Path(tdir) / "code").glob("*.py"):
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if lit.search(txt):
                return True
        return False

    def _base_face_columns(self) -> set:
        """基础特征面列名(splits/train.parquet schema);拿不到返回空集。"""
        cached = getattr(self, "_base_face_cache", None)
        if cached is not None:
            return cached
        out: set = set()
        try:
            import pyarrow.parquet as pq

            p = self.run_dir / "sample-profile" / "splits" / "train.parquet"
            if p.exists():
                out = {str(n) for n in pq.read_schema(str(p)).names}
        except Exception:  # noqa: BLE001
            out = set()
        self._base_face_cache = out
        return out

    def _champion_face_hpo_state(self, champ) -> dict:
        """[v2.15.0] 冠军特征面上的 HPO 状态(质量感知)。

        v2.13.4 的 `_champion_face_hpo_done` 只回答"有没有",两处缺口
        (req-t7-rand400w run3 实测):
          1. 冠军**本身**就是 HPO trial 时(R1 的 016 成为冠军后),它没有
             hpo 子代 → 判"未调参" → 每代仍按 0.75×MDE 先验提名同面 HPO;
          2. 跑过但**不完整**的 HPO(R1 只有 16 个 config 且 lr 顶在人为
             抬高的下界上)与完整的 HPO 同样记"已调参"。
        返回 {done, complete, diminishing, last_hpo, last_delta, reasons}。

        v2.17.2:同面判定改用真实模型列数(`_model_n_features`,A5);新增
        ``cap_bound`` / ``cap_bound_info``(冠军训练撞树数上限,A2)。
        """
        out = {"done": False, "complete": False, "diminishing": False,
               "last_hpo": None, "last_delta": None, "last_gain": None,
               "reasons": [], "cap_bound": False, "cap_bound_info": None,
               "last_cap_confounded": False}
        if champ is None:
            return out
        _cap = self._cap_bound_info(champ)
        out["cap_bound"] = bool(_cap)
        out["cap_bound_info"] = _cap
        n_champ = self._model_n_features(champ)
        if n_champ <= 0:
            return out
        cands = []
        if str(champ.data.get("search_action") or "") == "hpo":
            cands.append(champ)
        if str(champ.data.get("search_action") or "") == "confirm":
            for pid in champ.parent_ids:
                p = self.by_id.get(pid)
                if p is not None and str(
                        p.data.get("search_action") or "") == "hpo":
                    cands.append(p)
        for t in self.trials:
            if str(t.data.get("search_action") or "") != "hpo":
                continue
            if champ.trial_id not in t.parent_ids:
                continue
            if self._model_n_features(t) == n_champ and \
                    (t.data.get("evaluation") or {}):
                cands.append(t)
        if not cands:
            return out
        last = max(cands, key=lambda t: (t.generation, t.trial_id))
        out["done"] = True
        out["last_hpo"] = last.trial_id
        delta = ((last.data.get("evaluation") or {}).get("delta") or {}
                 ).get("vs_parent")
        # v2.17.0:边际递减按方向感知增益判(rmse 任务 Δ=−51.8 是改善,
        # 旧口径 `Δ < MDE` 恒真 → 有效 HPO 被误判递减);MDE 走 effective_mde
        _mde = self._mde()
        _g = self._gain(delta)
        if _g is not None:
            out["last_delta"] = float(delta)
            out["last_gain"] = float(_g)
            out["diminishing"] = float(_g) < _mde
        # v2.17.2(A1):上一轮 HPO 撞 cap 混杂(lr 收缩重训被截断)时,其负 Δ
        # 不作"边际递减"证据(run4 trial-017)
        try:
            from .directions import _is_cap_confounded

            if _is_cap_confounded(last, self.direction_up):
                out["last_cap_confounded"] = True
                out["diminishing"] = False
        except Exception:  # noqa: BLE001
            pass
        try:
            from .hpo import assess_hpo_study

            q = assess_hpo_study(Path(last.trial_dir), default_budget=int(
                self.config.get("evolution.hpo.default_budget", 40) or 40))
            out["complete"] = bool(q.get("complete"))
            out["reasons"] = list(q.get("reasons") or [])
        except Exception:  # noqa: BLE001
            out["complete"] = True
        if out.get("last_cap_confounded"):
            out["reasons"] = list(out.get("reasons") or []) + [
                "last_hpo_cap_confounded"]
        return out

    def _champion_face_hpo_done(self, champ) -> bool:
        """[v2.13.4] 冠军的**真实特征面**上是否已跑过 HPO。

        只认"跑在同一个特征面上"的 HPO:同父代、且 metrics 的
        `n_features` 与冠军一致。理由(实测事故):gen-04 的
        trial-018 确实是一条 hpo trial、父代也指向冠军,但它跑在
        1500 列基础面(冠军是 1887 列)—— 只看"有没有 hpo 子代"会
        把这条**口径不等价**的试验当成"冠军面已调过参"。

        v2.17.2(A5):列数取 `_model_n_features`(model/feature_columns.json
        优先),不再信复制旧代码写的 metrics.n_features。
        """
        if champ is None:
            return False
        n_champ = self._model_n_features(champ)
        if n_champ <= 0:
            return False
        for t in self.trials:
            if str(t.data.get("search_action") or "") != "hpo":
                continue
            if champ.trial_id not in t.parent_ids:
                continue
            if self._model_n_features(t) == n_champ:
                return True
        return False

    # ------------------------------------------------------------------
    # v2.17.2(A2):冠军撞树数上限 → lr 修正子代
    # ------------------------------------------------------------------
    #: 超参键的 lr 别名(分段键 `clf__learning_rate` 取 `__` 后段比较)
    _LR_KEYS = ("learning_rate", "eta", "lr")

    @classmethod
    def _lr_items(cls, hp: dict | None) -> dict:
        out = {}
        for k, v in (hp or {}).items():
            base = str(k).split("__")[-1].lower()
            if base in cls._LR_KEYS and isinstance(v, (int, float)) \
                    and not isinstance(v, bool) and float(v) > 0:
                out[str(k)] = float(v)
        return out

    def _trial_hparams(self, t) -> tuple[dict, str | None]:
        """trial 生效超参:config/trial.yaml.hyperparameters 优先;其中无 lr
        (HPO trial 的 config 为空)时回落 metrics.best_params。"""
        hp: dict = {}
        src = None
        try:
            cfg = load_yaml(Path(t.trial_dir) / "config" / "trial.yaml") \
                or {}
            hp = dict(cfg.get("hyperparameters") or {})
            src = "config.hyperparameters"
        except Exception:  # noqa: BLE001
            hp = {}
        if not self._lr_items(hp):
            try:
                bp = (load_json(Path(t.trial_dir) / "metrics.json")
                      or {}).get("best_params") or {}
                if self._lr_items(bp):
                    hp, src = dict(bp), "metrics.best_params"
            except Exception:  # noqa: BLE001
                pass
        return hp, src

    def _cap_bound_info(self, t) -> dict | None:
        """trial 训练是否撞树数上限:metrics.es_info.cap_bound 优先,其次
        evaluation.complexity.cap_bound;撞则返回明细,否则 None。"""
        if t is None:
            return None
        es = {}
        try:
            es = (load_json(Path(t.trial_dir) / "metrics.json")
                  or {}).get("es_info") or {}
        except Exception:  # noqa: BLE001
            es = {}
        cx = ((t.data.get("evaluation") or {}).get("complexity") or {})
        for src, d, stop_key, bi_key in (
                ("metrics.es_info", es, "stopped_by", "stage1_best_iteration"),
                ("evaluation.complexity", cx, "es_stopped_by",
                 "best_iteration")):
            if isinstance(d, dict) and d.get("cap_bound") is True:
                return {"trial_id": t.trial_id, "source": src,
                        "cap": d.get("cap"),
                        "best_iteration": d.get(bi_key,
                                                d.get("best_iteration")),
                        "stopped_by": d.get(stop_key),
                        "cap_extended_to": d.get("cap_extended_to"),
                        "cap_bound_unresolved": d.get(
                            "cap_bound_unresolved")}
        return None

    def _cap_bound_lr_candidate(self, champ, cap_info: dict) -> list[dict]:
        """冠军撞 cap 时的单变量 lr 修正子代(lr×2,其余超参与特征面不变)。

        事故(run4 req-t7-400w):冠军 trial-011(lr 0.0125,cap 8000)
        best_iter 7999 撞 cap,planner 只会提名同面 HPO(其 lr-shrink 重训
        同样被 cap 截断,trial-017 伪负 Δ,4.9h 白跑);人工补的 lr
        0.0125→0.025 子代(trial-018)+0.000304 且 confirm 复现(trial-020)。

        锚点:冠军本身;冠军是 confirm 时取其配方源父代。冠军/锚点的任一
        非 confirm/非 hpo 子代已改过 lr(单变量修正已做过)→ 不再提名;
        HPO 子代同时改结构/正则,不能回答"冠军是否欠训练",不计。
        评分:expected_gain=0.75×MDE(与同面 HPO 先验同档)+ information_gain
        1.0(消除 cap 混杂)、cost 1.0h —— 恒排在同面 HPO 之前。
        """
        self._cap_bound_note = None
        anchor = champ
        if str(champ.data.get("search_action") or "") == "confirm":
            _p = next((self.by_id.get(pid) for pid in champ.parent_ids
                       if pid in self.by_id), None)
            if _p is not None:
                anchor = _p
        hp, src = self._trial_hparams(anchor)
        lrs = self._lr_items(hp)
        if not lrs:
            self._cap_bound_note = {
                "trial_id": champ.trial_id, "skipped": "no_lr_hyperparameter",
                "reason": "冠军撞 cap,但 config.hyperparameters / "
                          "metrics.best_params 里找不到学习率键,无法物化 lr 修正"}
            return []
        roots = {champ.trial_id, anchor.trial_id}
        for t in self.trials:
            if not (set(t.parent_ids) & roots):
                continue
            if str(t.data.get("search_action") or "") in ("confirm", "hpo"):
                continue
            child = self._lr_items(self._trial_hparams(t)[0])
            for k, v in lrs.items():
                cv = child.get(k)
                if cv is not None and abs(cv - v) > 1e-12 * max(1.0, abs(v)):
                    self._cap_bound_note = {
                        "trial_id": champ.trial_id,
                        "skipped": "lr_already_changed_by_child",
                        "child": t.trial_id,
                        "reason": f"子代 {t.trial_id} 已把 {k} {v:g}→{cv:g}"}
                    return []
        new = {k: float(f"{2.0 * v:.6g}") for k, v in lrs.items()}
        _desc = ", ".join(f"{k} {lrs[k]:g}→{new[k]:g}" for k in sorted(new))
        _mde = self._mde()
        p = self._base(
            action="exploit", family="model:cap_bound_lr", parent=anchor,
            changed_modules=["model"],
            hypothesis=(f"冠军 {champ.trial_id} 训练撞树数上限(best_iter "
                        f"{cap_info.get('best_iteration')}/cap "
                        f"{cap_info.get('cap')}),疑似欠训练;只把 {_desc}"
                        "(其余超参与特征面不变),在 cap 内早停收敛"),
            rationale=("冠军 es_info.cap_bound=true:其分数被轮数上限截断而"
                       "低估,同面 HPO 的 lr 收缩重训会被同一上限截断(run4 "
                       "trial-017 伪负 Δ);先做单变量 lr 修正去掉 cap 混杂"
                       "(run4 实证 trial-018 +0.000304 且 confirm 复现)"),
            falsification=("Δparent≤0 或配对 CI 含 0 则冠军在 cap 内已近收敛;"
                           "另核子代 es_info.cap_bound=false,若仍撞 cap 则"
                           "结论无效(应继续提 lr)"),
            expected_gain=0.75 * _mde, information_gain=1.0, novelty=0.2,
            uncertainty=0.0, uncertainty_prior=0.5,
            potential_gain_upper_bound=_mde, cost_hours=1.0,
            prior_note=(f"冠军撞 cap({cap_info.get('source')});lr 来源 "
                        f"{src};排在同面 HPO 之前"),
            recipe={"type": "cap_bound_lr_correction",
                    "set_overrides": new, "from_lr": lrs,
                    "lr_source": src, "one_variable_change": True})
        p["cap_bound_correction"] = {"champion": champ.trial_id,
                                     "anchor": anchor.trial_id,
                                     **{k: cap_info.get(k) for k in (
                                         "source", "cap", "best_iteration",
                                         "stopped_by", "cap_extended_to")}}
        return [p]

    # ------------------------------------------------------------------
    # v2.19.0(#14):同面 HPO 的期望增益 = 历史 HPO 实测 Δ 的后验
    # ------------------------------------------------------------------
    def _n_train(self) -> int | None:
        try:
            from .profile_rules import profile_facts

            return (profile_facts(self.run_dir, self.spec) or {}).get(
                "n_train")
        except Exception:  # noqa: BLE001
            return None

    def _hpo_history(self) -> dict:
        """本 run + 经验库视图同任务的 HPO 实测增益(方向感知,主指标量纲)。

        本 run:search_action=hpo、已评估、非 invalid、非撞 cap 混杂的 trial,
        取 evaluation.delta.vs_parent。经验库:视图 trial_outcomes 中 action=hpo、
        atid(经别名归一)等于本 run 的同任务记录,取 delta_parent(按记录的
        higher_is_better 折成增益);与本 run trial_id 重复的去重。
        后验 = (Σgain + K0·先验) / (n + K0),先验 = HPO_PRIOR_GAIN_MDE×MDE。
        """
        cached = getattr(self, "_hpo_hist_cache", None)
        if cached is not None:
            return cached
        mde = self._mde()
        rows = []
        own_ids = set()
        try:
            from .directions import _is_cap_confounded
        except Exception:  # noqa: BLE001
            _is_cap_confounded = None
        for t in self.trials:
            if str(t.data.get("search_action") or "") != "hpo":
                continue
            ev = t.data.get("evaluation") or {}
            if not ev or (ev.get("integrity") or {}).get("valid", True) \
                    is False:
                continue
            try:
                if _is_cap_confounded is not None and \
                        _is_cap_confounded(t, self.direction_up):
                    continue
            except Exception:  # noqa: BLE001
                pass
            g = self._gain((ev.get("delta") or {}).get("vs_parent"))
            if g is None:
                continue
            own_ids.add(t.trial_id)
            rows.append({"source": "run", "trial_id": t.trial_id,
                         "gain": g})
        view = self._library_view()
        if view is not None and view.atid and \
                self.config.get("library.enabled", True) is not False:
            try:
                from .library import retrieval as _rt

                canon = _rt.canon_fn(view.data)
                me = canon(view.atid)
                for o in view.data.get("trial_outcomes") or []:
                    if str(o.get("action") or "") != "hpo" or \
                            canon(o.get("atid")) != me or \
                            o.get("trial_id") in own_ids:
                        continue
                    d = o.get("delta_parent")
                    if isinstance(d, bool) or not isinstance(d, (int, float)):
                        continue
                    hib = o.get("higher_is_better")
                    g = float(d) if hib in (True, None) else -float(d)
                    rows.append({"source": "library",
                                 "trial_id": o.get("trial_id"),
                                 "run_id": o.get("run_id"), "gain": g})
            except Exception:  # noqa: BLE001 — 经验库不可读只用本 run
                pass
        gains = [r["gain"] for r in rows]
        n = len(gains)
        prior = HPO_PRIOR_GAIN_MDE * mde
        post = (sum(gains) + HPO_PRIOR_K0 * prior) / (n + HPO_PRIOR_K0)
        out = {"n": n, "n_run": sum(r["source"] == "run" for r in rows),
               "n_library": sum(r["source"] == "library" for r in rows),
               "mean_gain": (sum(gains) / n) if n else None,
               "n_positive": sum(g >= HPO_POSITIVE_MDE * mde for g in gains),
               "prior_gain": prior, "posterior_gain": post,
               "rows": rows[:12]}
        self._hpo_hist_cache = out
        return out

    def _hpo_nomination(self) -> tuple[float | None, str]:
        """同面 HPO(尚未在冠军面跑过)的期望增益;返回 (None, 理由) = 不提名。"""
        mde = self._mde()
        h = self._hpo_history()
        n_train = self._n_train()
        if n_train and n_train > HPO_LARGE_N and h["n_positive"] == 0:
            return None, (f"大样本(n_train={n_train} > {HPO_LARGE_N})且同任务"
                          f"无正向 HPO 历史(n={h['n']},增益≥{HPO_POSITIVE_MDE}"
                          "×MDE 的 0 次):默认不提名同面 HPO")
        post = float(h["posterior_gain"])
        if h["n"] and post < HPO_MIN_EXPECTED_MDE * mde:
            return None, (f"同任务历史 HPO n={h['n']}(本 run {h['n_run']}/"
                          f"经验库 {h['n_library']}),实测均值 "
                          f"{float(h['mean_gain'] or 0.0):+.6g},后验 "
                          f"{post:+.6g} < {HPO_MIN_EXPECTED_MDE}×MDE:不提名")
        if not h["n"]:
            return post, (f"无同任务 HPO 历史,按先验 {HPO_PRIOR_GAIN_MDE}"
                          "×MDE 优先补上")
        return post, (f"同任务历史 HPO n={h['n']}(本 run {h['n_run']}/经验库 "
                      f"{h['n_library']}),实测均值 "
                      f"{float(h['mean_gain'] or 0.0):+.6g},收缩后验 "
                      f"{post:+.6g}(先验 {HPO_PRIOR_GAIN_MDE}×MDE,伪计数 "
                      f"{HPO_PRIOR_K0:g})")

    def _model_candidates(self, parent) -> list[dict]:
        configured = {str(x).lower() for x in
                      (self.config.get("baseline.algorithms") or [])}
        tried = {str(t.data.get("algorithm") or "").lower()
                 for t in self.trials}
        # preflight 的任务级环境快照位于 run-state.env_manifest，不是
        # run/env-manifest.json。仅对确认可导入的可选算法扩展候选集；
        # 没有环境证据时只使用配置声明的算法，避免规划不可执行 Trial。
        state_path = self.run_dir / "run-state.json"
        env = ((load_json(state_path) or {}).get("env_manifest") or {}) \
            if state_path.exists() else {}
        available = {str(p.get("name") or "").lower()
                     for p in env.get("packages") or []
                     if p.get("import_ok") is True}
        algos = [a for a in MATERIALIZED_MODEL_ALGOS
                 if a in configured or a in available]
        if available:
            algos = [a for a in algos if a in available]
        out = []
        for algo in algos:
            if algo.lower() in tried:
                continue
            out.append(self._base(
                action="hpo", family=f"model:{algo}", parent=parent,
                algorithm=algo, changed_modules=["model"],
                hypothesis=f"{algo} 的归纳偏置与当前冠军互补",
                rationale="算法族尚未在同一数据面接受客观评测；用受控 HPO 模板物化",
                falsification="同预算下配对增益不显著且成本更高则淘汰",
                novelty=1.0, information_gain=0.4,
                uncertainty=0.0, uncertainty_prior=0.4,
                potential_gain_upper_bound=2.0 * self._mde(),
                cost_hours=1.0,
                recipe={"type": "model_family", "algorithm": algo,
                        "tuning": "controlled_hpo", "budget_configs": 12}))
        if parent is not None:
            # [v2.13.4] 同算法 HPO 的**期望增益**必须有实测先验,而不是
            # 与异算法候选在同一把尺子上裸拼 novelty/information_gain。
            #
            # 实测事故(req-t7-drawdown-v21331 gen-04):批次里同时有
            # `model:catboost`(异算法,12 config,1.0h)与 `model:hpo`
            # (冠军面同算法,40 config,2.0h),两者 acquisition 同分档,
            # 而 `model:hpo` 因 cost_penalty 更高(-0.30 vs -0.15)被
            # 挤掉 —— 先投的是**价值更低**的那条。根因不是成本,是评分
            # 函数里没有"这个算法是不是已经在该面上胜出"这一项。
            #
            # 先验取值有实测来源,不是拍脑袋:同一数据面的闭环证据
            # (exp-202609-1f43bd13)给同算法 HPO 的 ΔAUC = +0.0007~0.0008
            # ≈ 0.75×MDE;异算法在这类高基宽表上实测为**负**
            # (catboost -0.0002 / lightgbm 0.0007 但未过门)。因此:
            #   - 已在该面胜出的算法:expected_gain = 0.75×MDE(实测中值),
            #     这是"有正期望的唯一模型杠杆";
            #   - 未在该面试过的算法:expected_gain = 0(无正证据,不虚报);
            #   - 等价面已跑过同算法 HPO:expected_gain = 0(不重复网络)。
            # "等价面" = 与冠军 metrics.n_features 一致(见
            # `_champion_face_hpo_done`),因为特征面不同则超参不可迁移。
            _champ_algo = str(
                (parent.data.get("algorithm") or "")).lower()
            # [v2.15.0] 质量感知:冠军本身是 HPO / 上一轮不完整 / 边际递减
            _st = self._champion_face_hpo_state(parent)
            _analyzed = bool(_st.get("done") or
                             self._champion_face_hpo_done(parent))
            _mde = self._mde()
            _skip_hpo = None
            if not _analyzed:
                # v2.19.0(#14):固定 0.75×MDE 改为同任务历史 HPO 实测后验;
                # 负收益/大样本无正向历史 → 不提名(理由落 batch.hpo_not_nominated)
                _same_algo_prior, _why = self._hpo_nomination()
                if _same_algo_prior is None:
                    _skip_hpo = _why
                    _same_algo_prior = 0.0
                _state_note = "该特征面上尚无等价 HPO trial;" + _why
            elif _st.get("last_cap_confounded"):
                # v2.17.2(A1):上一轮同面 HPO 撞 cap 混杂 —— 负 Δ 不作递减
                # 证据,但在 cap 问题解决前续投同面 HPO 仍会被同一上限截断
                _same_algo_prior = 0.0
                _state_note = (f"同面上一轮 HPO {_st.get('last_hpo')} 撞树数"
                               "上限且无余量(cap_confounded),其负 Δ 不作边际"
                               "递减证据;" + (
                                   "冠军本身撞 cap,先跑 lr 修正子代"
                                   if _st.get("cap_bound") else
                                   "确认重训 cap 有余量后再续搜")
                               + ",期望增益暂记 0")
            elif _st.get("diminishing"):
                _same_algo_prior = 0.0
                _state_note = (f"同面上一轮 HPO {_st.get('last_hpo')} 的增益="
                               f"{float(_st.get('last_gain') or 0.0):+.6g} "
                               f"< MDE={_mde:.6g},边际递减"
                               "门生效(new-trial 需 --override-hpo-"
                               "diminishing),期望增益归零")
            elif not _st.get("complete") and _st.get("last_gain"):
                # 上一轮不完整且仍有 ≥MDE 的增益:续搜(热启动复用全部观测)
                # v2.17.0:用方向感知增益(rmse 任务 last_delta<0 才是增益)
                _same_algo_prior = min(0.75 * _mde,
                                       0.5 * float(_st["last_gain"]))
                _state_note = (f"同面上一轮 HPO {_st.get('last_hpo')} 不完整"
                               f"({'; '.join(_st.get('reasons') or [])}),"
                               "按上一轮增益×0.5 估计续搜价值;v2.15 热启动"
                               "复用全部历史观测、边界自适应外扩")
            else:
                _same_algo_prior = 0.0
                _state_note = ("该特征面上已存在等价且完整的 HPO trial,"
                               "期望增益归零避免重复同一搜索网络")
            _u_prior = 0.55 if _analyzed else 0.30
            self._hpo_skipped = None
            if _skip_hpo:
                self._hpo_skipped = {
                    "parent": parent.trial_id, "family": "model:hpo",
                    "reason": _skip_hpo,
                    "history": {k: v for k, v in self._hpo_history().items()
                                if k != "rows"}}
            if not _skip_hpo:
                out.append(self._base(
                    action="hpo", family="model:hpo", parent=parent,
                    algorithm=_champ_algo or None,
                    changed_modules=["model"],
                    hypothesis=("在冻结特征面上做资源感知 HPO 可榨取模型余量"
                                f"(同算法 {_champ_algo or 'unknown'};"
                                f"期望增益 {_same_algo_prior / _mde:.2f}×MDE)"),
                    rationale=("仅在当前最佳血缘上搜索,避免全局盲扫;"
                               "期望增益取同任务历史 HPO 实测后验(v2.19.0),"
                               "不虚报"),
                    falsification="inner 改善不能在 validation 配对门复现则拒绝",
                    expected_gain=_same_algo_prior, information_gain=0.3,
                    uncertainty=0.0, uncertainty_prior=_u_prior,
                    potential_gain_upper_bound=self._mde(),
                    cost_hours=2.0,
                    prior_note=(
                        f"冠军算法={_champ_algo or 'unknown'};" + _state_note),
                    recipe={"type": "hpo", "budget_configs": int(
                        self.config.get("evolution.hpo.default_budget", 40)),
                        "search_scope": "model_only"}))
            # v2.17.2(A2):冠军撞树数上限 → 先提名单变量 lr 修正子代(评分
            # 恒高于同面 HPO,见 _cap_bound_lr_candidate);该父代上的 HPO
            # 候选附 cap_bound_parent 提示(lr 收缩重训会被同一上限截断)
            _cap = _st.get("cap_bound_info")
            if _cap:
                out += self._cap_bound_lr_candidate(parent, _cap)
                _hint = {
                    "trial_id": parent.trial_id,
                    **{k: _cap.get(k) for k in (
                        "source", "cap", "best_iteration", "stopped_by",
                        "cap_extended_to")},
                    "hint": ("父代训练撞树数上限(轮数被截断);HPO 的 lr 收缩"
                             "重训可能被同一上限截断而得伪负 Δ(run4 "
                             "trial-017)。优先跑 lr 修正子代,或确认重训 cap "
                             "有余量后再投 HPO")}
                for _p in out:
                    if _p.get("action") == "hpo" and \
                            parent.trial_id in (_p.get("parent_ids") or []):
                        _p["cap_bound_parent"] = dict(_hint)
        return out

    def _validate(self, proposal: dict,
                  release_id: str | None = None) -> list[str]:
        errors = []
        if proposal.get("action") not in ALLOWED_ACTIONS:
            errors.append("unsupported_action")
        if not set(proposal.get("changed_modules") or []) <= ALLOWED_MODULES:
            errors.append("unsupported_changed_module")
        if any(pid not in self.by_id for pid in proposal.get("parent_ids") or []):
            errors.append("unknown_parent")
        if proposal.get("action") == "confirm" and len(
                proposal.get("parent_ids") or []) != 1:
            errors.append("confirm_requires_one_parent")
        if _contains_forbidden(proposal):
            errors.append("final_test_reference_forbidden")
        if not proposal.get("hypothesis") or not proposal.get("falsification"):
            errors.append("hypothesis_or_falsification_missing")
        if is_model_direction(proposal.get("action"),
                              proposal.get("changed_modules")) and \
                self._feature_phase()["active"]:
            # [FIX-2026-09-17] 特征侧已收敛时,每代放行**一个**模型方向
            # 名额(release_id 由 plan() 确定性选出,取有效候选里
            # proposal_id 最小者)。理由随 batch 落盘,不再依赖人工
            # --override-feature-phase 的记忆。
            if release_id and proposal.get("proposal_id") == release_id:
                # 理由优先取 plan() 里算好的(含收敛判据明细);直接调用
                # _validate 的场景(测试/dry-run)没有它,就地补一条同口径
                # 的可读理由,避免留痕为空。
                _why = self._feature_phase_release_reason
                if not _why:
                    _conv, _w = self._exploration_converged()
                    _why = ("特征侧已收敛(" + (_w or "converged") + ");"
                            "特征期内每代释放一个模型方向名额。")
                    self._feature_phase_release_reason = _why
                proposal["feature_phase_release"] = {"granted": True,
                                                     "reason": _why}
            else:
                errors.append(FEATURE_PHASE_ERROR)
        return errors

    def _feature_generations_done(self) -> int:
        """[FIX-2026-09-19 D4] 已完成(有 evaluation)的探索代数(gen-00 不计)。"""
        return feature_generations_done(self.trials)

    def _feature_phase(self) -> dict:
        return feature_phase_status(self.trials, self.config)

    @staticmethod
    def _source_of(p: dict) -> str:
        """候选来源标签(batch.candidate_sources 统计用)。"""
        rt = str((p.get("recipe") or {}).get("type") or "")
        fam = str(p.get("family") or "")
        if p.get("action") == "confirm":
            return "confirm"
        if rt == "profile_rule":
            return "profile_rule"
        if rt == "experience_principle":
            return "experience"
        if fam.startswith("screen:"):
            return "super_trial"
        if fam.startswith("info:"):
            return "inventory"
        if rt in ("direction_refinement",):
            return "direction"
        return "model"

    def _source_counts(self, generated: list[dict]) -> dict:
        out: dict = {}
        for p in generated:
            k = self._source_of(p)
            out[k] = out.get(k, 0) + 1
        return out

    def _confirm_policy_decision(self, others: list[dict], *,
                                 requested_slots: int,
                                 remaining: int) -> dict:
        """v2.19.0(#14):本代是否提名 confirm。

        ``confirm.policy``:
          - ``per_generation``:旧行为,每代都按 `_confirmation_candidates` 提名;
          - ``final_only``(默认):只在以下任一成立时提名对**搜索冠军**的 confirm:
            (a) 停止建议已触发(evolution-state.json 的 ``stopped`` 非空);
            (b) 剩余预算 ≤ 1 代(剩余 trial 数 ≤ 本代请求槽位,或下一代已是
                max_generations 的最后一代);
            (c) 未交付增益 ≥ MDE(搜索冠军相对现交付位的差距;尚无交付位时取
                冠军相对 baseline 的增益)且没有更多高分候选(非 confirm 候选
                的 expected_gain 均 < CONFIRM_HIGH_VALUE_MDE×MDE)。
        背景:run4 HPO+confirm 占墙钟约 44%,每代 confirm 的冠军大多在下一代
        即被超越,复现结果用不上;交付只需要最终冠军确认一次。
        """
        policy = str(self.config.get("confirm.policy", "final_only")
                     or "final_only").strip().lower()
        out: dict = {"policy": policy, "nominate": True, "triggers": [],
                     "detail": {}}
        if policy != "final_only":
            out["policy"] = "per_generation"
            out["triggers"] = ["per_generation"]
            return out
        mde = self._mde()
        # (a) 停止建议
        try:
            es = load_json(self.run_dir / "evolution-state.json") \
                if (self.run_dir / "evolution-state.json").exists() else {}
        except Exception:  # noqa: BLE001
            es = {}
        stopped = (es or {}).get("stopped")
        if stopped:
            out["triggers"].append("stop_suggested")
            out["detail"]["stop_reason"] = (stopped.get("stop_reason")
                                            if isinstance(stopped, dict)
                                            else str(stopped))
        # (b) 剩余预算 ≤ 1 代
        budget = self.spec.resource_budget or {}
        try:
            max_gen = int(budget.get("max_generations") or
                          self.config.get("evolution.max_generations", 0)
                          or 0)
        except (TypeError, ValueError):
            max_gen = 0
        next_gen = self._next_generation()
        out["detail"].update(remaining_trials=remaining,
                             requested_slots=requested_slots,
                             next_generation=next_gen,
                             max_generations=max_gen or None)
        if requested_slots > 0 and remaining <= requested_slots:
            out["triggers"].append("remaining_trials_le_one_generation")
        if max_gen and next_gen >= max_gen:
            out["triggers"].append("last_generation")
        # (c) 未交付增益 ≥ MDE 且无更多高分候选
        info = self._search_champion_info()
        undelivered = None
        champ = info.get("trial")
        if champ is not None and info.get("status") != "bound":
            if info.get("delivery_trial") and info.get("gap") is not None:
                undelivered = float(info["gap"])
            elif not info.get("delivery_trial"):
                d = ((champ.data.get("evaluation") or {}).get("delta")
                     or {})
                undelivered = self._gain(d.get("vs_baseline",
                                               d.get("vs_anchor")))
        existing = {str(t.data.get("planner_semantic_hash"))
                    for t in self.trials
                    if t.data.get("planner_semantic_hash")}
        high = [p["proposal_id"] for p in others
                if p["semantic_hash"] not in existing
                and float((p.get("acquisition") or {}).get(
                    "expected_gain_mde") or 0.0) >= CONFIRM_HIGH_VALUE_MDE]
        out["detail"].update(undelivered_gain=undelivered,
                             high_value_candidates=high[:8],
                             champion_status=info.get("status"))
        if undelivered is not None and undelivered >= mde and not high:
            out["triggers"].append("undelivered_gain_ge_mde_no_high_value")
        out["nominate"] = bool(out["triggers"])
        if not out["nominate"]:
            out["reason"] = ("confirm.policy=final_only:停止建议未触发、剩余预算"
                             "多于 1 代、且仍有高分候选或未交付增益 < MDE —— "
                             "本代不提名 confirm(交付前对最终冠军确认一次)")
        return out

    def plan(self, slots: int = 4) -> dict:
        budget = self.spec.resource_budget or {}
        max_trials = int(budget.get("max_trials") or
                         self.config.get("evolution.max_trials", 80))
        pending = sum(t.status in ("PROPOSED", "CODE_WRITTEN")
                      for t in self.trials)
        remaining = max(
            0, max_trials - self.factory.count_started() - pending)
        slots = max(0, min(int(slots), remaining))
        parent = self._parent()
        # v2.19.0:候选来源 7 种 —— confirm / super_trial / inventory /
        # profile_rule(#10 画像规则表)/ experience(#12 经验生成)/ 方向台账 /
        # 模型族与同面 HPO(#14 历史后验)
        others = (self._super_trial_candidate(parent)
                  + self._inventory_candidates(parent)
                  + self._profile_rule_candidates(parent)
                  + self._experience_candidates(parent)
                  + self._direction_candidates(parent, slots)
                  + self._model_candidates(parent))
        # v2.19.0(#14):confirm 只做一次(confirm.policy=final_only 默认)
        confirm_policy = self._confirm_policy_decision(
            others, requested_slots=int(slots), remaining=remaining)
        confirms = self._confirmation_candidates() \
            if confirm_policy["nominate"] else []
        if confirm_policy["policy"] == "final_only":
            # final_only 只对搜索冠军配方提名(stale 容差内的旁支不确认)
            _dropped = [c for c in confirms
                        if (c.get("confirm_nomination") or {}).get("role")
                        != "search_champion_recipe"]
            confirms = [c for c in confirms if c not in _dropped]
            for c in _dropped:
                self._confirm_skipped.append({
                    "trial_id": (c.get("parent_ids") or [None])[0],
                    "reason": "confirm_policy_final_only_champion_only"})
            # 触发后的这一次 confirm 是交付前提:保底入批(不与探索候选比分)
            for c in confirms:
                c["confirm_nomination"]["policy_trigger"] = list(
                    confirm_policy["triggers"])
                c["confirm_nomination"]["reserved_slot"] = True
        generated = confirms + others

        existing_hashes = {str(t.data.get("planner_semantic_hash"))
                           for t in self.trials
                           if t.data.get("planner_semantic_hash")}
        existing_legacy = {_semantic_hash(_legacy_identity({
            "action": t.data.get("search_action"),
            "family": t.data.get("family_hints"),
            "parent_ids": t.parent_ids,
            "algorithm": t.data.get("algorithm"),
        })) for t in self.trials}
        valid, rejected = [], []
        # [FIX-2026-09-17] 特征期释放名额:仅当特征侧已收敛时启用,
        # 在**模型方向候选**里确定性挑一个(proposal_id 最小,与排序无关,
        # 保证同输入同输出)带机器可读理由放行。每代最多一个。
        self._feature_phase_release_reason = ""
        _release_id = None
        _phase = self._feature_phase()
        if _phase["active"]:
            _converged, _why = self._exploration_converged()
            if _converged:
                _modelish = [
                    p["proposal_id"] for p in generated
                    if is_model_direction(p.get("action"),
                                          p.get("changed_modules"))]
                # v2.17.2(A2):冠军撞 cap 的 lr 修正子代优先拿释放名额
                # (它消除的是测量混杂,不是普通模型方向)
                _capfix = [
                    p["proposal_id"] for p in generated
                    if (p.get("recipe") or {}).get("type")
                    == "cap_bound_lr_correction"]
                if _capfix:
                    _modelish = _capfix
                if _modelish:
                    _release_id = min(_modelish)
                    self._feature_phase_release_reason = (
                        "特征侧已收敛(" + _why + ");特征期内每代释放一个"
                        "模型方向探索名额,避免 model/HPO 在特征期被锁死"
                        "(实测 drawdown/t7 三代提名同一 HPO 候选、三代被"
                        "同一原因拒绝)。")
        seen = set()
        # [FIX-2026-09-19 D2] 方向台账已证伪的 family 集合:explore/exploit
        # 候选命中即拒(错误码 family_falsified);confirm/hpo/model 族
        # 不受此规则影响。列表也写进 batch.falsified_families 供 report
        # 呈现"哪些方向已判负、不再提名"。
        _falsified = self._falsified_families()
        # v2.17.2(D1):比较前两侧统一经 _norm_family 归一(台账键带 feat:
        # 前缀、信息源候选是 info:<col>:<dim>,旧实现直比永不命中);avoid
        # 经验卡(同 atid、do_not_retry/avoid)命中同样拒绝
        _falsified_norm = {_norm_family(f) for f in _falsified}
        _avoid_rejected: list[dict] = []
        for p in generated:
            errors = self._validate(p, release_id=_release_id)
            key = p["semantic_hash"]
            legacy_key = _semantic_hash(_legacy_identity(p))
            if key in seen or key in existing_hashes or \
                    legacy_key in existing_legacy:
                errors.append("duplicate_semantics")
            _avoid_ids: list[str] = []
            if p.get("action") in ("explore", "exploit"):
                if _norm_family(p.get("family")) in _falsified_norm:
                    errors.append("family_falsified")
                _avoid_ids = self._experience_avoid_hit(p)
                if _avoid_ids:
                    errors.append("experience_avoid")
            if errors:
                _rej = {"proposal_id": p.get("proposal_id"),
                        "errors": errors}
                if _avoid_ids:
                    _rej["family"] = p.get("family")
                    _rej["experience_ids"] = _avoid_ids
                    _avoid_rejected.append(dict(_rej))
                rejected.append(_rej)
                continue
            seen.add(key)
            valid.append(p)

        # 高价值优先，但每个 action/family 最多 2 个，避免一批全押同轴。
        # v2.19.0:final_only 触发的冠军 confirm 保底排在最前(见上)
        valid.sort(key=lambda p: (
            not (p.get("confirm_nomination") or {}).get("reserved_slot"),
            -p["acquisition"]["score"], p["proposal_id"]))
        selected, action_count, family_count = [], {}, {}
        for p in valid:
            if len(selected) >= slots:
                break
            action, family = p["action"], p["family"]
            if action_count.get(action, 0) >= 2 or \
                    family_count.get(family, 0) >= 1:
                continue
            selected.append(p)
            action_count[action] = action_count.get(action, 0) + 1
            family_count[family] = family_count.get(family, 0) + 1
        # 多样性约束后仍有空槽，再按价值补齐（family 仍最多 2）。
        for p in valid:
            if len(selected) >= slots:
                break
            if p in selected or family_count.get(p["family"], 0) >= 2:
                continue
            selected.append(p)
            family_count[p["family"]] = family_count.get(p["family"], 0) + 1

        # [FIX-2026-09-19 D1] 断言性去重:proposals[] 与 rejected[]
        # 的 proposal_id 集合必须不相交。防御性兜底——任何上游路径
        # (重复候选、释放名额重入)让同一 id 同时落两侧时,从 rejected
        # 移除已入选的那条,保证批次语义唯一(同一 proposal 不能既
        # "入选"又"被拒")。
        _selected_ids = {p.get("proposal_id") for p in selected}
        if _selected_ids:
            rejected = [r for r in rejected
                        if r.get("proposal_id") not in _selected_ids]

        # [v2.13.4] 模型/HPO 轴的可见性告警(req-t7-drawdown-v21331 实测)。
        #
        # 事故形态:该 run 的 `_model_candidates` 稳定产出 `model:hpo`
        # (冠军面 HPO,正是本任务唯一有实测正期望的剩余杠杆),但它在
        # 每代的 acquisition 排序里都排在信息源候选之后,被 slots 截断
        # —— 且**没有任何信号**说明"模型轴被挤掉了"。Agent 只看到
        # 一批 explore/confirm,据此判定"模型方向无潜力"并进入收尾。
        # 被截断 ≠ 已探索过,但批次里读不出这个区别。
        #
        # 修复分两层:根因是 acquisition 量纲错配(见 `_base`,已在
        # v2.13.4 订正);本段是**审计兜底** —— 只要本代生成了模型轴
        # 候选而一个都没入选,就把它们连同被挤掉的原因写进 batch,
        # 让 `planner-context` 与 report 能显式呈现"模型轴候选存在但
        # 未进批次",而不是让操作者从空集反推"没有这个方向"。
        _modelish_gen = [p for p in generated
                         if p.get("action") == "hpo"
                         or set(p.get("changed_modules") or [])
                         & {"model", "loss"}]
        _modelish_sel = [p for p in selected
                         if p.get("action") == "hpo"
                         or set(p.get("changed_modules") or [])
                         & {"model", "loss"}]
        model_axis_note = None
        if _modelish_gen and not _modelish_sel:
            _top = max(_modelish_gen,
                       key=lambda x: x["acquisition"]["score"])
            model_axis_note = {
                "generated": [p["proposal_id"] for p in _modelish_gen],
                "families": sorted({str(p.get("family"))
                                    for p in _modelish_gen}),
                "top_score": _top["acquisition"]["score"],
                "cutoff_score": (min((p["acquisition"]["score"]
                                      for p in selected), default=None)),
                "reason": ("本代生成了模型/HPO 轴候选但无一入选"
                           "(被 acquisition 排序 + slots 截断)。"
                           "截断 ≠ 已探索:该方向在后续代仍可能被提名,"
                           "不得据此判定模型轴无潜力。"),
            }

        upper_bound = max((p["acquisition"][
            "potential_gain_upper_bound"] for p in valid), default=0.0)
        batch = {
            "schema_version": 1,
            "created_at": now_iso(),
            "generation": self._next_generation(),
            "requested_slots": int(slots),
            "remaining_trial_budget": remaining,
            "candidate_count": len(generated),
            "valid_candidate_count": len(valid),
            "candidate_value_upper_bound": float(upper_bound),
            "proposals": selected,
            "rejected_candidates": rejected,
            # [FIX-2026-09-19 D2] 已证伪 family 显式落盘,供 report 呈现
            # "哪些方向已判负、为什么不再提名"(跨代重复提名已证伪方向
            # 是实测事故,不落盘就无法审计)。
            "falsified_families": sorted(_falsified),
            # v2.17.2(D1):证伪判据明细(mean_gain<=0 / lcb_gain<0 /
            # ucb_gain<mde)与被 avoid 经验卡拒绝的候选
            "falsified_detail": dict(
                getattr(self, "_falsified_detail", {}) or {}),
            "experience_avoid_rejected": _avoid_rejected,
            # v2.17.0:eligible 但未被提名 confirm 的父代及原因(非冠军配方
            # 且冠军已绑定 / 落后冠军超过交付容差 / 次数上限 / 在途 HPO)
            "confirm_nomination_skipped": list(
                getattr(self, "_confirm_skipped", []) or []),
            # v2.19.0(#14):confirm 提名策略与本代判定(触发条件明细)
            "confirm_policy": confirm_policy,
            # v2.19.0(#10/#12/#14):画像规则命中、经验候选跳过原因、同面 HPO
            # 不提名原因(审计"为什么这份数据得到这份计划")
            "profile_rules": [
                {"rule": h.get("rule"), "id": h.get("id"),
                 "evidence": {k: v for k, v in (h.get("evidence") or {}).items()
                              if k != "top"},
                 "families": [sg.get("family") for sg in
                              h.get("suggestions") or []
                              if sg.get("materialize", True)],
                 "advisory": h.get("advisory")}
                for h in (self._profile_rules().get("hits") or [])],
            "experience_candidates_skipped": list(
                getattr(self, "_experience_skipped", []) or [])[:20],
            "hpo_not_nominated": getattr(self, "_hpo_skipped", None),
            "candidate_sources": self._source_counts(generated),
            "mde": {"value": self._mde(),
                    "source": "runtime.mde.effective_mde"},
            "selection_policy": "acquisition_then_action_family_diversity",
            # [FIX-2026-09-17] 被特征期门拒绝的候选与放行理由显式落盘:
            # report/反思据此说明"哪些方向从未被执行、为什么",不再把
            # "被门拒"误报成"无未执行高潜力方向"。active = 特征期内
            # (硬拒 + 收敛释放唯一通道);期满后不拦。new-trial 读
            # released_proposal_id 认 Planner 释放的名额。
            FEATURE_PHASE_KEY: {
                "active": _phase["active"],
                "feature_generations_done": _phase["generations_done"],
                "min_generations": _phase["min_generations"],
                "released_proposal_id": _release_id,
                "release_reason": self._feature_phase_release_reason,
                "blocked_proposal_ids": sorted(
                    r["proposal_id"] for r in rejected
                    if FEATURE_PHASE_ERROR in (r.get("errors") or [])),
            },
            # [v2.13.2] 特征 bundle 宽度记账。`bundle_beam_width` 此前是
            # 零引用的配置键;这里把它落进批次(供审计"本代计划多宽"),
            # 实际咬人的门在 cmd_new_trial(实测 0/15 个 gen-01 trial 带
            # planner_proposal_id —— Agent 全部手建,Planner 侧约束咬不到)。
            "feature_beam": {
                "bundle_beam_width": int(self.config.get(
                    "feature_exploration.bundle_beam_width", 4)),
                "super_trial_enabled": bool(self.config.get(
                    "feature_exploration.super_trial", False)),
                "feature_slots_planned": sum(
                    1 for p in selected
                    if p.get("action") in ("explore", "data_view")
                    and (set(p.get("changed_modules") or [])
                         & {"feature", "sample", "data"})),
            },
        }
        # [v2.13.4] 模型/HPO 轴被截断时显式留痕(见上方 model_axis_note)。
        # 只有"确实生成了却没入选"才写,避免给批次塞恒空字段。
        if model_axis_note is not None:
            batch["model_axis_not_selected"] = model_axis_note
        # v2.17.2(A2):冠军撞 cap 但未提名 lr 修正(无 lr 键/子代已改过)的原因
        if getattr(self, "_cap_bound_note", None):
            batch["cap_bound_correction_skipped"] = self._cap_bound_note
        batch["batch_id"] = "batch-" + _semantic_hash(batch)
        out_dir = self.run_dir / "planner"
        out_dir.mkdir(parents=True, exist_ok=True)
        for algorithm, space in SEARCH_SPACES.items():
            search_space = out_dir / \
                f"default-search-space-{algorithm}.yaml"
            if not search_space.exists():
                dump_yaml(search_space, space)
        dump_yaml(out_dir / f"proposal-batch-gen-{batch['generation']:02d}.yaml",
                  batch)
        dump_json(out_dir / "latest.json", batch)
        return batch
