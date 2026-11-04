"""Trial:标准实验单元(§10)。

状态机:PROPOSED → CODE_WRITTEN → RUNNING → SUCCEEDED/FAILED
        → EVALUATED → REFLECTED → ACCEPTED/REJECTED/PROMISING(+INVALID)
强约束:代码先在 Trial 运行目录落盘,再执行(§10.3)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logging_utils import (dump_json, dump_yaml, get_logger, load_yaml,
                            now_iso, to_jsonable)

logger = get_logger(__name__)

TRIAL_STATES = (
    "PROPOSED", "CODE_WRITTEN", "RUNNING",
    "SUCCEEDED", "FAILED", "EVALUATED", "REFLECTED",
)
TRIAL_VERDICTS = ("accepted", "confirmed", "promising", "rejected", "invalid",
                  "unconfirmed")
# unconfirmed(v2.14.0):confirm trial 已完整评估但未通过独立确认。它是
# 父代配方换 seed 的合法模型,不是"没有提升"的搜索失败 —— 旧口径记
# rejected,run2 的全 run 最优模型 trial-022 因此从所有候选池消失。
# 不进冠军/Archive 池(白名单只收 accepted/confirmed),不计方向失败。
# data_view(v7):数据面探索——重建聚合口径/特征面,配套 per-view splits;
# 属 explore 谱系但单列动作,供停止判定的单口径护栏与反思 scope 核对
# hpo(v11):trial 内完整超参搜索(inner-validation 上搜,outer 只评
# 一次)。修 HPO 跨 trial 粒度错配:一个网格点不再吃掉一条 leaderboard
# 行 + 一次完整评估;搜索语义不进 noise gate(outer 干净比较)。
# confirm(v12,P1-6):复现类 trial——成功定义是 Δ≈0(等价性检验),
# verdict=confirmed 允许进 Archive;旧口径必然误判 rejected。
# automl(v12,P1-10):外部 AutoML 参照系(AutoGluon/FLAML 在当前特征面
# 跑满预算),既释放零 LLM 算力,又校准"天花板已到"的结论。
SEARCH_ACTIONS = ("explore", "exploit", "crossover", "data_view", "hpo",
                  "confirm", "automl", "warm_start")
# warm_start(v2.15.6):gen-00 直接复用全局经验库 top-trial 配方的 trial。
# 与 explore 的语义差异:warm_start 是"验证历史配方在本 run 的有效性",
# 而不是"从零搜索新方向"。两者都以 baseline 为对比基准(delta_vs_baseline
# 天然免疫样本日期漂移,因为 baseline 与 warm_start 使用同一切分),
# reference_auc(来源 run 的绝对 AUC)只作参考展示,不参与任何裁决。
# warm_start trial 在 _is_improvement 中排除,不计入"本代新发现的提升证据"。


def trial_currently_failed(trial: "TrialRecord") -> bool:
    """v2.12.2.5:该 Trial 当前是否**确实处于失败态**。

    顶层 `failure_reason` 是历史残留字段(runner 失败分支写,重跑成功
    路径在旧数据里不清空)。凡是要把"失败"当门槛的地方都应走本函数,
    而不是直接读 `data["failure_reason"]` —— 否则「先失败、同一 trial
    重跑成功」的 Trial 会被永久排除出 confirm 父代池与交付选型
    (drawdown-t7-sv24 实测:trial-028/029 均 succeeded + accepted,却因
    残留 code_error 被硬拒)。

    判据:当前 status == FAILED,或 result.status == failed。两者都缺失
    的旧记录才回退看 failure_reason(保守)。
    """
    _st = str(trial.data.get("status") or "")
    _res = str((trial.data.get("result") or {}).get("status") or "")
    if _st == "FAILED" or _res == "failed":
        return True
    if not _st and not _res:
        return bool(trial.data.get("failure_reason"))
    return False


def trial_is_screening(trial: "TrialRecord") -> bool:
    """[v2.13.2] 该 Trial 是否为 screening-only(宽面初筛)。

    screening trial 的语义:**贡献证据,不参与裁决**。它把大量候选特征族
    一次性加宽,产出 `feature/group_importance.json` 的
    families_alive/families_dead 作后续 bundle 规划的**先验**,但:

      * 不进代级 FDR/alpha spending(它不提"提升主张");
      * 不计入 accepted_now / trials_accepted(不驱动收敛/穷尽出口);
      * 不作冠军(best_trial)、不进 Archive、不作 crossover/confirm 父代;
      * 不入方向台账(否则 UCB arm 的 mean_delta/delta_per_hour 被污染)。

    它仍走完整 evaluator,verdict 与指标如实落盘并出现在 leaderboard
    (带 screen_only 列)——"它到底多强"必须可审计,只是不能当选。

    标记是 trial.yaml 上的布尔字段(经 `new-trial --screen-only` 写入),
    **不是**新的 search_action 取值:SEARCH_ACTIONS 被 7 处按动作分组的
    统计消费,新动作值会污染全部分组。
    """
    return bool(trial.data.get("screen_only"))


def confirmation_parent_eligibility(
        trial: "TrialRecord") -> tuple[bool, str]:
    """确认试验父代硬门：合法 accepted + 已完成选择校正。

    根 Trial 只建立基线而不提出增益主张，因此没有同代 FDR p 值；它可
    直接进入独立确认。其他候选必须显式通过代级选择校正，不能把“尚未
    finalize”解释成“默认通过”。
    """
    ev = trial.data.get("evaluation") or {}
    # [v2.13.2] screening trial 不参与裁决,自然也不能当晋升确认的父代
    # ——它没有经过代级选择校正(被显式排除在 FDR 之外),拿它去 confirm
    # 等于把"未经校正的先验"洗成"已确认的冠军"。
    if trial_is_screening(trial):
        return False, "parent_is_screening_only"
    # v2.19.1:停止时收尾整合通过的 promising 只获得 confirm 资格(verdict
    # 不变、不当冠军);由 confirm 独立裁决它能否成为冠军
    closing_ok = (ev.get("verdict") == "promising"
                  and (ev.get("closing_consolidation") or {}).get("passed")
                  is True)
    if ev.get("verdict") != "accepted" and not closing_ok:
        return False, "parent_verdict_not_accepted"
    if (ev.get("integrity") or {}).get("valid", True) is False:
        return False, "parent_integrity_invalid"
    # v2.12.2.5:看**当前**失败态而非历史残留字段(见 trial_currently_failed)
    if trial_currently_failed(trial):
        return False, "parent_has_failure"
    # v2.20.0:root_baseline_only 的豁免语义是"根 Trial 只建立跑线、不提增益
    # 主张,因此没有可校正的 p 值"。warm-start **没有父代**(cmd_startup 建
    # warm trial 时不传 parent_ids),所以它天然带 root_baseline_only,但它
    # 恰恰提出了 vs_baseline 增益主张 —— 豁免不适用于它。
    # 旧口径把这条放在 warm 家族检查之前,v2.17.2(C3)的预注册 BH 因此形同
    # 虚设:任何 p_better < 0.95 的 warm trial 都在这里直接放行,只有
    # p_better ∈ [0.95, 阈值) 的窄带才会真的被判 warm_start_not_confirmable。
    _is_warm = str(trial.data.get("search_action") or "") == "warm_start"
    if ev.get("root_baseline_only") is True \
            and not (_is_warm and _warm_root_fix_active(trial)):
        return True, "root_baseline"
    if closing_ok:
        return True, "closing_consolidation_confirm_only"
    correction = ev.get("selection_correction") or {}
    if correction.get("passed") is not True:
        # v2.17.2(C3):warm-start 走 gen-00 预注册家族的 vs_baseline BH
        # (evolution._apply_warm_start_fdr);未过 / 尚无校正记录都给出明确
        # 原因,不再与普通候选的 selection_correction_not_passed 混为一谈
        if str(trial.data.get("search_action") or "") == "warm_start":
            return False, WARM_START_NOT_CONFIRMABLE
        return False, "selection_correction_not_passed"
    # v2.17.0:锚定晋级(promotion_reference=anchor)下 passed 已是"通过
    # vs 锚点校正 + vs_parent 护栏 + 质量门"的合取;被护栏拦下的不晋级
    # (verdict 也不会是 accepted,这里是第二道显式门)。
    if correction.get("promotion_blocked_by"):
        return False, ("selection_correction_blocked:"
                       f"{correction.get('promotion_blocked_by')}")
    if correction.get("reference") == "anchor":
        return True, "selection_correction_passed_vs_anchor"
    if correction.get("family") == "warm_start":
        return True, "selection_correction_passed_warm_start"
    return True, "selection_correction_passed"


#: v2.17.2(C3):warm-start trial 未通过 gen-00 预注册家族校正(或尚未校正)
WARM_START_NOT_CONFIRMABLE = "warm_start_not_confirmable"


def _warm_root_fix_active(trial: "TrialRecord") -> bool:
    """v2.20.0 资格修正是否对该 trial 所在 run 生效。

    "尺子不可演进":在途 run(run-state.skill_version < 2.20.0 或无该键)
    保持旧口径,与 runtime.protocol_gates 其它行为门同口径。只有无
    trial_dir 的内存记录(读不到 run)按新口径处理。
    """
    tdir = getattr(trial, "trial_dir", None)
    if tdir is None:
        return True
    try:
        from .protocol_gates import V2_20_0, run_at_least

        return run_at_least(Path(tdir).parent.parent, V2_20_0)
    except Exception:  # noqa: BLE001  读失败按旧 run 处理
        return False

# changed_modules 枚举(v8,结构化声明;crossover 兼容性检查输入)。
# 历史值不拒绝,仅 warning(向后兼容)
VALID_CHANGED_MODULES = ("feature", "sample", "model", "loss", "data")

VALID_TRANSITIONS = {
    "PROPOSED": {"CODE_WRITTEN", "RUNNING", "FAILED"},
    "CODE_WRITTEN": {"RUNNING", "FAILED"},
    "RUNNING": {"SUCCEEDED", "FAILED"},
    "SUCCEEDED": {"EVALUATED"},
    "FAILED": {"EVALUATED"},
    "EVALUATED": {"REFLECTED"},
    "REFLECTED": set(),  # 终态由 verdict 表达
}
# EVALUATED 重新评估(评估口径修正/补 shadow)不算状态回退:
# 允许 EVALUATED → EVALUATED(重评覆盖)与 REFLECTED 前的重评
VALID_TRANSITIONS["EVALUATED"].add("EVALUATED")

FAILURE_REASONS = ("timeout", "oom", "code_error", "infra_error", "evaluator_error")


def slugify(text: str, maxlen: int = 24) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", str(text)).strip("-").lower()
    return s[:maxlen].strip("-") or "trial"


@dataclass
class TrialRecord:
    """Trial 强制字段(§10.2)+ 落盘/加载。"""

    data: dict[str, Any]
    trial_dir: Path | None = None

    # ---- 便捷访问 ----
    @property
    def trial_id(self) -> str:
        return self.data["trial_id"]

    @property
    def status(self) -> str:
        return self.data.get("status", "PROPOSED")

    @property
    def verdict(self) -> str | None:
        return self.data.get("evaluation", {}).get("verdict")

    @property
    def parent_ids(self) -> list[str]:
        raw = self.data.get("parent_trial_ids")
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw]
        return [str(x) for x in raw]

    @property
    def generation(self) -> int:
        return int(self.data.get("generation", 0))

    def transition(self, new_state: str) -> None:
        cur = self.status
        if new_state not in VALID_TRANSITIONS.get(cur, set()):
            raise ValueError(f"非法状态迁移: {cur} → {new_state}")
        self.data["status"] = new_state
        self.data.setdefault("status_history", []).append(
            {"state": new_state, "at": now_iso()})
        # v2.12.2.5(drawdown-t7-sv24 实测事故回写):进入 SUCCEEDED 时清除
        # 顶层 failure_reason。runner 在失败分支写该字段、重跑成功路径
        # **不清空**,而 confirmation_parent_eligibility() 只看它 ⇒
        # 「先失败、同一 trial 重跑成功」的 trial 会被永久排除出 confirm
        # 父代池。实测:gen-04 的 trial-028(lightgbm HPO)/trial-029
        # (catboost) 均 succeeded + accepted + integrity.valid,却因首轮
        # 解释器错误残留的 failure_reason=code_error 被硬拒
        # (parent_has_failure),使「换算法族再确认」这条路径整体失效。
        # 失败事实仍完整保留在 attempts/status_history 与 run-state 的
        # last_failure_reason 里,审计不受影响。
        if new_state == "SUCCEEDED":
            self.data.pop("failure_reason", None)

    # ---- 落盘 ----
    def flush(self) -> Path:
        assert self.trial_dir is not None, "trial_dir 未设置"
        d = Path(self.trial_dir)
        d.mkdir(parents=True, exist_ok=True)
        dump_yaml(d / "trial.yaml", self.data)
        return d

    def write_artifact(self, name: str, payload: Any, *, fmt: str = "json") -> Path:
        assert self.trial_dir is not None
        path = Path(self.trial_dir) / name
        if fmt == "json":
            dump_json(path, payload)
        elif fmt == "yaml":
            dump_yaml(path, payload)
        else:
            path.write_text(str(payload), encoding="utf-8")
        return path

    @classmethod
    def load(cls, trial_dir: Path) -> "TrialRecord":
        data = load_yaml(Path(trial_dir) / "trial.yaml")
        return cls(data=data, trial_dir=Path(trial_dir))


class BudgetExhausted(RuntimeError):
    """max_trials 预算已耗尽(§16.5.5):进入过 RUNNING 的 Trial 数达到上限。"""


class TrialFactory:
    """创建并登记 Trial;run_dir 为 run-{task_id}/ 根目录。

    创建口执行预算护栏:max_trials 统计所有进入过 RUNNING 的 Trial,
    达到上限后拒绝创建(§16.5.5),避免静默吞噬预算。
    """

    def __init__(self, run_dir: Path, spec: "TaskSpec | None" = None,
                 config=None):
        self.run_dir = Path(run_dir)
        self.trials_root = self.run_dir / "trials"
        self.trials_root.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.config = config

    def _max_trials(self) -> int | None:
        if self.spec is not None:
            v = self.spec.resource_budget.get("max_trials")
            if v:
                return int(v)
        if self.config is not None:
            v = self.config.get("evolution.max_trials")
            if v:
                return int(v)
        return None

    def _next_seq(self) -> int:
        seqs = []
        for d in self.trials_root.iterdir():
            m = re.match(r"trial-(\d+)", d.name)
            if m:
                seqs.append(int(m.group(1)))
        return (max(seqs) + 1) if seqs else 1

    def next_generation(self) -> int:
        """已落盘 trial 的最大代号 + 1(≥1)。

        new-trial 漏传 --generation 时的自动递增兜底(防代号错标:
        实测快通道 trial 漏传时落到 argparse 默认值 1,把 gen-N 的
        trial 标回 g01,污染按代号聚合的统计与 stop 判据输入)。
        """
        gens = [t.generation for t in self.all_trials()]
        return (max(gens) + 1) if gens else 1

    def create(
        self,
        *,
        algorithm: str,
        hypothesis: dict,
        execution: dict,
        search_action: str = "explore",
        parent_trial_ids: list[str] | None = None,
        generation: int = 0,
        expected_resources: dict | None = None,
    ) -> TrialRecord:
        if search_action not in SEARCH_ACTIONS:
            raise ValueError(f"search_action 必须是 {SEARCH_ACTIONS}")
        # changed_modules 归一化(v8):小写去空白;未知枚举值 warning 不拒绝
        changed_modules = [str(m).strip().lower()
                           for m in (execution.get("changed_modules") or [])
                           if str(m).strip()]
        unknown = [m for m in changed_modules
                   if m not in VALID_CHANGED_MODULES]
        if unknown:
            logger.warning("changed_modules 含未知枚举值 %s"
                           "(建议使用 %s;历史值保留不拒绝)",
                           unknown, VALID_CHANGED_MODULES)
        statement = str(hypothesis.get("statement", "")).strip()
        # 占位检查:排除显式占位词;长度按字符数(中文友好)。
        # [FIX-2026-09-03] 扩展黑名单:gen-03 实测 "RESERVED trial-010"
        # / "RESERVED-CLEANUP: …" 不含旧词表却同样是占位 trial(创建后
        # rm -rf 清理),把 reserved/cleanup/scaffold/占位 拦进创建口。
        placeholder_marks = ("待填写", "todo", "tbd", "xxx", "placeholder",
                             "reserved", "cleanup-check", "scaffold",
                             "占位", "cleanup ")
        if (not statement or len(statement) < 6
                or any(m in statement.lower() for m in placeholder_marks)):
            raise ValueError("hypothesis.statement 不能为空/占位"
                             "(占位 trial 无存在必要:trial_id 由时间戳生成,"
                             "确需放弃已建 trial 用 delete-trial 子命令)")
        # 预算护栏(§16.5.5):创建口拦截,而不是事后停止
        limit = self._max_trials()
        if limit is not None and self.count_started() >= limit:
            raise BudgetExhausted(
                f"max_trials={limit} 已耗尽(count_started="
                f"{self.count_started()});拒绝创建新 Trial")
        seq = self._next_seq()
        hh = int(now_iso()[11:13] + now_iso()[14:16]) % 10000  # HHMM 时间戳
        # 命名含代次和动作:trial-{seq}-g{gen:02d}-{search_action}-{algorithm}-{hh}
        trial_id = (f"trial-{seq:03d}-g{generation:02d}-"
                    f"{slugify(search_action)}-{slugify(algorithm)}-{hh}")
        trial_dir = self.trials_root / trial_id
        data = {
            "trial_id": trial_id,
            "algorithm": algorithm,
            "parent_trial_ids": parent_trial_ids or [],
            "generation": generation,
            "search_action": search_action,
            "changed_modules": changed_modules,
            "hypothesis": {
                "statement": hypothesis["statement"],
                "rationale": hypothesis.get("rationale", ""),
                "falsification": hypothesis.get("falsification", ""),
            },
            "execution": {
                "action_description": execution.get("action_description", []),
                "code_entry": execution.get("code_entry", "train.py"),
                "command": execution.get("command",
                                         "python train.py --config "
                                         "../config/trial.yaml"),
            },
            "expected_resources": expected_resources or {},
            "status": "PROPOSED",
            "status_history": [{"state": "PROPOSED", "at": now_iso()}],
            "attempts": [],
            "created_at": now_iso(),
            # run_date: 北京时间日期字符串，便于跨 run 检索和 top-trial 归因
            "run_date": now_iso()[:10],
        }
        rec = TrialRecord(data=data, trial_dir=trial_dir)
        rec.flush()
        (trial_dir / "code").mkdir(exist_ok=True)
        (trial_dir / "config").mkdir(exist_ok=True)
        return rec

    def all_trials(self) -> list[TrialRecord]:
        out = []
        if not self.trials_root.exists():
            return out
        for d in sorted(self.trials_root.iterdir()):
            if (d / "trial.yaml").exists():
                out.append(TrialRecord.load(d))
        return out

    def count_started(self) -> int:
        """max_trials 预算口径(§16.5.5):进入过 RUNNING 的 Trial 数。

        兼容状态字段被直接覆写(未经 transition)的历史数据:
        status 已越过 RUNNING 的也算已启动。
        """
        started_states = {"RUNNING", "SUCCEEDED", "FAILED",
                          "EVALUATED", "REFLECTED"}
        n = 0
        for t in self.all_trials():
            if t.data.get("status") in started_states or any(
                    s.get("state") in started_states
                    for s in t.data.get("status_history", [])):
                n += 1
        return n


# ---------------------------------------------------------------------------
# v2.17.2(B5):算力计账单源。旧口径只累加 trial 的 result.runtime_sec,
# 收尾(finalize-all 的冠军重训 + baseline 终测打分、evaluate-final 盲测)
# 不计(run4 opus5.5:finalize-all 95.8 min 在 compute_hours 里不可见)。
# finalize-all 写 logs/finalize-all.json(cmd_speed),evaluate-final 写
# logs/evaluate-final.json(cmd_deliver);本函数是两者唯一的读取方。
# ---------------------------------------------------------------------------
FINALIZE_TIMING_LOGS = ("finalize-all.json", "evaluate-final.json")


def record_timing_log(path: Path, entry: dict) -> dict:
    """追加一次收尾计时:顶层保留最近一次的字段(旧读者兼容),``history``
    累计每次执行(重跑也计入算力)。旧格式(无 history)的单条记录并入
    history 首条。返回写入的 dict。"""
    import json as _json

    path = Path(path)
    prev: dict = {}
    try:
        prev = _json.loads(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        prev = {}
    if not isinstance(prev, dict):
        prev = {}
    hist = prev.get("history")
    if not isinstance(hist, list):
        hist = [{k: v for k, v in prev.items() if k != "history"}] \
            if prev else []
    hist.append(dict(entry))
    out = dict(entry)
    out["history"] = hist
    dump_json(path, out)
    return out


def _timing_log_sec(path: Path) -> float:
    """一份收尾计时日志的累计秒数(history 逐次求和;旧格式单条)。

    单次口径优先 compute_sec(并行子任务时长之和,与 trial 的
    Σruntime 同口径),其次 wall_sec,再次 wall_min×60。
    """
    try:
        import json as _json

        rec = _json.loads(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return 0.0
    if not isinstance(rec, dict):
        return 0.0
    hist = rec.get("history")
    entries = hist if isinstance(hist, list) and hist else [rec]
    total = 0.0
    for e in entries:
        if not isinstance(e, dict):
            continue
        for key, scale in (("compute_sec", 1.0), ("wall_sec", 1.0),
                           ("wall_min", 60.0)):
            v = e.get(key)
            if isinstance(v, (int, float)) and v >= 0:
                total += float(v) * scale
                break
    return total


def compute_hours_summary(run_dir: Path | None,
                          trials: list["TrialRecord"]) -> dict:
    """算力计账(小时):

    - ``trial_compute_hours`` = Σ trial ``result.runtime_sec``(训练+评估段);
    - ``finalize_hours`` = finalize-all + evaluate-final 的收尾用时
      (``finalize_breakdown`` 分列);
    - ``compute_hours`` = 两者之和(stop_report / status / generation.yaml
      budget_metrics 共用此口径)。
    """
    sec = 0.0
    for t in trials or []:
        rt = (t.data.get("result") or {}).get("runtime_sec")
        if isinstance(rt, (int, float)):
            sec += float(rt)
    breakdown: dict[str, float] = {}
    if run_dir is not None:
        logs = Path(run_dir) / "logs"
        for name in FINALIZE_TIMING_LOGS:
            s = _timing_log_sec(logs / name)
            if s > 0:
                breakdown[name.rsplit(".", 1)[0]] = round(s / 3600.0, 4)
    trial_h = sec / 3600.0
    fin_h = sum(breakdown.values())
    return {"compute_hours": trial_h + fin_h,
            "trial_compute_hours": trial_h,
            "finalize_hours": fin_h,
            "finalize_breakdown": breakdown}


def lineage_of(trial: TrialRecord, by_id: dict[str, TrialRecord]) -> list[str]:
    """演进链路(§15.1 lineage):从根到该 Trial 的 trial_id 序列。"""
    chain: list[str] = []

    def walk(tid: str, seen: set):
        if tid in seen:
            return
        seen.add(tid)
        t = by_id.get(tid)
        if t is None:
            chain.append(tid)
            return
        for p in t.parent_ids:
            walk(p, seen)
        chain.append(tid)

    walk(trial.trial_id, set())
    return chain


def parse_parent_field(text: Any) -> list[str]:
    """从 leaderboard/Archive 的 parent 字段还原 trial_id 列表。

    兼容历史行以分号分隔、以及被迭代字符串化为逐字符列表的脏数据。
    """
    if text is None:
        return []
    if isinstance(text, str):
        return [x for x in text.split(";") if x]
    if isinstance(text, (list, tuple)):
        out: list[str] = []
        for x in text:
            if isinstance(x, str) and len(x) == 1 and out:
                out[-1] += x  # 逐字符列表:拼接回字符串
            elif x:
                out.append(str(x))
        joined = ";".join(out)
        return [x for x in joined.split(";") if x]
    return []
