"""方向台账(v11):按假设族聚合 trial 证据,UCB 分配下一代预算。

结构问题:direction_deltas 按 4 个 search_action 聚合(explore/exploit/
crossover/data_view),「explore 方向递减」对 planner 毫无可操作性;
动作配比是静态 50/30/20,只在两个触发条件下上调 explore。

本模块把方向单元换成**假设族**(特征前缀族/data_view/模型族),每个
arm 记 n_tried/n_accepted/mean_delta/std/cost_hours,按 delta-per-hour
的 UCB 分配预算。分工:LLM 提出 arm(它擅长),确定性代码分配预算
(它不擅长)。与 CandidateArchive 互补——archive 评候选个体,
台账评方向族。

载体 run-{task_id}/directions.json;planner-context 第 9 节呈现;
finalize-generation 时刷新。

v12(P1-1)数值修复(三处全部坐实的缺陷):
- 归因口径 vs_baseline → vs_parent:累计量被祖先增益支配,实测六个
  arm 的 mean_delta 全落在 0.001 区间内,归因完全失效;边际贡献
  才是方向裁决的正确口径;
- 0.1h 量纲断裂:hours>0.1 用 delta/小时、否则用原始 delta,两个
  mean_delta 差 2% 的 arm 其 dph 差 9.2 倍;统一 delta/max(hours,eps);
- UCB 探索项比收益项大两个数量级(实测 bonus≈0.48 vs dph≈0.005–0.02),
  排序完全由试次数决定,数值上退化为轮询:bonus 改按 delta_per_hour
  的 std 自适应标定。

v2.15.x ROI 三处修复(req-t7-rand400w run3 复盘回写):
1. **全链路成本归因**:confirm trial 的运行时间按 parent_ids 回溯,
   计入父代配方所在 arm 的 cost_sec。旧实现 changed_modules 为空的
   confirm 不入账,其 3~4h 成本对台账完全不可见,导致 model:hpo /
   model:xgboost 的 dph 被系统性高估(实测 model:xgboost 真实成本
   5.84h 不可见,dph 高估约 40%)。
2. **n_eff 相关性折减**:同一 arm 内多次 trial 来自同一配方的子代,
   不是独立同分布观测。按 n_eff = n / (1 + (n-1) × ρ̂) 折减有效
   样本量(ρ̂ 由 arm 内相邻 delta 的相关系数估算,无方差时取 0.7)。
   n_eff 同时用于 LCB 置信区间的分母。
3. **LCB 替代 mean_delta 驱动 UCB**:UCB 收益项改用
   LCB = mean_delta − z_lcb × SE_eff,z_lcb=1.28(单尾 90%)。
   旧口径以 mean_delta(选择偏差的乐观估计)作为 UCB 主项,负效应
   臂被截 0 但正效应臂的乐观偏差无人纠正。LCB 让"观测次数少、
   std 大"的 arm 主动降权,不再靠 UCB 探索项来"补偿"低 n 臂。

v2.17.0 方向感知增益(run4 regression.2 #3 回写):
- 旧实现把 ``delta.vs_parent`` 原值直接当"增益"。rmse 等越小越好的
  指标上 Δ>0 是**变差**:trial-021(feat:amt2)的 +10.7 被记成增益、
  dph=11.5 成为唯一正 arm,已证伪方向连续 4 代被 UCB 首选;反过来真
  正有效的 seq 方向(Δ≈−50)mean_delta<0,被 planner 判"已证伪"。
- 现在逐 trial 先换算 ``gain = +Δ``(越大越好)或 ``gain = −Δ``(越小
  越好,口径同 :func:`runtime.metrics.metric_direction`),mean/std/
  LCB/UCB/dph 全部基于 gain。落盘字段 ``mean_delta``/``std_delta``/
  ``lcb_delta`` **语义改为方向感知增益**(与 ``mean_gain``/``std_gain``/
  ``lcb_gain`` 同值,越大越好;AUC 任务数值逐位不变),原始 Δ 均值另记
  ``mean_raw_delta``;顶层 ``gain_convention`` 声明口径。
- n_accepted:evaluation 带 ``parent_verdict``(v2.17.0 锚定晋级后,
  vs_parent 口径的单试验裁决)时用它,否则沿用 ``verdict``——台账衡量
  的是"该方向相对父代的边际贡献",不应被锚定口径的晋级裁决改写。

v2.17.2 撞 cap 混杂(run4 req-t7-400w trial-017 回写):
- evaluation.cap_confounded(或按 :func:`runtime.evaluator.cap_confounded_detail`
  从旧产物重算)的 trial 与 infra 失败同处理:n_tried/cost 照常入账,
  **不记增益**(不给负奖励)、不计 accepted;arm 新增 ``n_cap_confounded``,
  planner 的证伪判定扣除这部分试次。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .logging_utils import dump_json, get_logger, load_json, now_iso
from .metrics import metric_direction
from .trial import trial_is_screening

logger = get_logger(__name__)

#: 小时数下限(防 0 除;0.1h 断裂修复后统一 dph=delta/max(hours,eps))
_HOURS_EPS = 0.05

#: UCB 探索项自适应基准:bonus = c·std(delta_per_hour)·sqrt(ln N/n)。
#: std 缺失(arm 内无方差)时按 dph 全局量级兜底,保证探索项与
#: 收益项同量纲(旧常数 0.5 比典型 dph 大两个数量级,排序退化为轮询)
UCB_C = 0.5

# ROI 修复常量 ---------------------------------------------------------------
#: LCB 单尾 z(90% 置信):LCB = mean_delta - z·SE_eff
_LCB_Z = 1.28
#: arm 内 trial 相关系数的默认先验(同一配方子代高度相关)
_RHO_PRIOR = 0.7
#: n_eff 最小值(至少视为 1 个有效观测)
_N_EFF_MIN = 1.0

def _strip_feat_prefix(hint: str) -> str:
    """去掉 hint 里已带的 `feat:` 前缀(可重复,如 `feat:feat:seqw`)。

    [v2.15.0] Planner 的 exploit 候选把**台账键本身**(如 `feat:seqw`)
    当 `--family-hints` 传下来,旧实现再拼一次前缀得到 `feat:feat:seqw`
    —— exploit 的结果入了一个新臂,原臂永远停在 n=1 的首次观测上
    (req-t7-rand400w run3:`feat:seqw` 的 +0.0033 在 gen-02 exploit
    得 −0.00004 后仍被当成上界,planner 的 candidate_value_upper_bound
    恒 > MDE,diminishing_return 停止出口被堵死)。台账每次 refresh
    从全量 trial 重建,在这里归一即可同时修正历史键。
    """
    while hint.startswith("feat:"):
        hint = hint[len("feat:"):]
    return hint


def direction_aware_gain(delta: Any, higher_is_better: bool) -> float | None:
    """v2.17.0:原始 Δ → 方向感知增益(越大越好);非有限值返回 None。"""
    if isinstance(delta, bool) or not isinstance(delta, (int, float)):
        return None
    d = float(delta)
    if not np.isfinite(d):
        return None
    return d if higher_is_better else -d


def accepted_for_ledger(ev: dict) -> bool:
    """v2.17.0:台账 n_accepted 口径——优先 parent_verdict(vs_parent 单试验
    裁决),缺失时回落 verdict。"""
    ev = ev or {}
    pv = ev.get("parent_verdict")
    if isinstance(pv, dict):  # 容忍 {verdict: ...} 包装
        pv = pv.get("verdict")
    v = pv if pv else ev.get("verdict")
    return v in ("accepted", "confirmed")


def _is_infra_failed(trial) -> bool:
    """v2.17.0:trial 因内存准入超时失败(admission_timeout,训练从未启动)
    且无评估 —— 零成本、零证据,不入方向台账。timeout/infra_error 维持
    旧口径(有实际耗时,成本照常入账)。"""
    data = getattr(trial, "data", None) or {}
    if data.get("evaluation"):
        return False
    reason = (data.get("result") or {}).get("failure_reason") \
        or data.get("failure_reason")
    return str(reason or "") == "admission_timeout"


def _is_cap_confounded(trial, higher_is_better: bool) -> bool:
    """v2.17.2(A1):trial 的负结论是否被"撞 cap 且无余量"混杂。

    evaluation.cap_confounded(v2.17.2 evaluator 写)优先;旧产物按同一判据
    (:func:`runtime.evaluator.cap_confounded_detail`)从 evaluation.complexity
    + metrics.json 的 es_info 重算 —— 台账每次 refresh 从全量 trial 重建,
    历史 run(如 run4 trial-017)在这里一并订正。
    """
    data = getattr(trial, "data", None) or {}
    ev = data.get("evaluation") or {}
    if not isinstance(ev, dict) or not ev:
        return False
    if ev.get("cap_confounded") is True:
        return True
    es_info = None
    tdir = getattr(trial, "trial_dir", None)
    if tdir is not None:
        mp = Path(tdir) / "metrics.json"
        if mp.exists():
            try:
                es_info = (load_json(mp) or {}).get("es_info")
            except Exception:  # noqa: BLE001 — metrics 坏了只用 complexity
                es_info = None
    try:
        from .evaluator import cap_confounded_detail

        return cap_confounded_detail(
            ev, es_info=es_info, higher_is_better=higher_is_better,
            is_root_trial=not (getattr(trial, "parent_ids", None)
                               or data.get("parent_trial_ids")),
            confirm_trial=str(data.get("search_action") or "") == "confirm"
        ) is not None
    except Exception:  # noqa: BLE001 — 判定失败按旧口径入账
        return False


#: 台账键:trial 的方向归属推断优先级
def family_of_trial(trial_data: dict, run_dir: Path | None = None) -> str:
    """推断 trial 的假设族键。

    优先级:data_view(v1 之外)> changed_modules 中首个非 feature 值
    (model/loss/sample)> 特征 trial 的 replicate:seed(列集与配方
    均同父代)/ 列集差集(加块 feat:<族>、摘块 ablate:<族>,v2.18.4)
    > 显式 family_hints(v16/P1-3)> 特征前缀族(v18/P1-2:
    group_importance / gain top)> feat:misc 兜底(仅当确无加宽列)。
    """
    dv = trial_data.get("data_view") or "v1"
    action = str(trial_data.get("search_action") or "explore")
    if action == "hpo":
        return "model:hpo"
    if dv != "v1":
        return f"view:{dv}"
    mods = [str(m) for m in (trial_data.get("changed_modules") or [])]
    non_feature = [m for m in mods if m not in ("feature", "")]
    if non_feature:
        return f"{non_feature[0]}:{trial_data.get('algorithm') or 'any'}"
    if "feature" in mods:
        # v2.18.4:列集差集 **先于** family_hints —— hint 是 Agent 写下的
        # 意图,列集是模型实际吃进去的东西。摘块/消融臂常把被摘的块名写进
        # hint(实测 trial-012 摘 seqh_ 却写 hints=['seqh_']),按 hint 归因
        # 会把"摘块"记成"加块",与 trial-008 的真实增益互相抵消。
        #
        # 只有差集**本身**先于 hint;差集给不出结论(父代缺列清单、新增
        # 列不足 2 列…)时必须先回 hint,不能落到 group_importance 的
        # top 族 —— 那正是冠军面支配族(seqc2)抢归因的旧根因。
        #
        # 列集与父代完全相同**且**训练配方语义相同时不是方向 trial,而是
        # 换 seed 复测(实测 trial-011/012 列集与配方指纹分别等于父代
        # trial-008/trial-002)。Δ 即 seed 噪声实测值,单独记 replicate:seed。
        # 列名不变但配方变了(改分桶边界、修特征计算)仍是真实特征方向。
        if _is_seed_replicate(run_dir, trial_data):
            return "replicate:seed"
        diff = _column_diff_family(run_dir, trial_data)
        if diff:
            fam0, kind = diff
            # 摘块臂的 vs_parent Δ 与"加该块"的增益符号相反,单独记
            # ablate:<族>,不与 feat:<族> 的加块证据混算。
            return f"ablate:{fam0}" if kind == "removed" else f"feat:{fam0}"
    # v16/P1-3:显式 family_hints(new-trial --family-hints 落
    # trial.yaml)——实测 120w run 全部特征工作挤在 feat:misc
    # 单臂(action_description 无人写前缀提示),UCB 与"哪个
    # 特征族该续投"脱节。多 hint 时取首个(主方向)。
    hints = trial_data.get("family_hints")
    if isinstance(hints, (list, tuple)) and hints:
        h = _strip_feat_prefix(str(hints[0]).strip().rstrip("_"))
        if h:
            return f"feat:{h}"
    if "feature" in mods:
        # v18/P1-2:从特征面差集推断族(消灭 feat:misc 塌缩)。
        # 400w run trial-007(wkmo_ 123 列)未传 family_hints →
        # 落 feat:misc;而 group_importance 已按真实前缀算出该族。
        # 推断链:本 trial metrics.json 的 n_features > 父代 → 取
        # feature_importance.csv 中本代新增前缀(gain top 的前缀族);
        # 无 importance 时退回 action_description 硬编码提示。
        fam = _feature_family_from_trial(run_dir, trial_data)
        if fam:
            return f"feat:{fam}"
        desc = " ".join(str(a) for a in
                        (trial_data.get("execution") or {}).get(
                            "action_description") or [])
        for hint in ("seq_", "ratio_", "miss_", "te_", "agg_", "window_"):
            if hint in desc:
                return f"feat:{hint.rstrip('_')}"
        return "feat:misc"
    return f"action:{action}"


def _columns_of(tdir: Path) -> list[str] | None:
    """trial 的实际模型列(按 model/feature_columns.json;缺失返回 None)。"""
    p = tdir / "model" / "feature_columns.json"
    if not p.exists():
        return None
    try:
        import json

        cols = json.loads(p.read_text(encoding="utf-8"))
        return [str(c) for c in cols] if isinstance(cols, list) else None
    except Exception:  # noqa: BLE001
        return None


def _family_of_new_columns(trial_cols: list[str],
                           parent_cols: list[str] | None
                           ) -> tuple[str, str] | None:
    """本 trial 相对父代列集差集的主前缀族与形态(v2.18.4)。

    返回 ``(族, "added" | "removed")``:加块 → 取新增列;新增列为空时
    取**父代独有列**(摘除的就是该块)。两种形态都不该拿整面
    group_importance 的 top 族替代 —— 冠军面里 seqc2 永远是支配族,
    与本次改了什么无关。
    """
    if not trial_cols:
        return None
    pset = set(parent_cols or [])
    kind = "added"
    new = [c for c in trial_cols if parent_cols is None or c not in pset]
    if not new and parent_cols:
        tset = set(trial_cols)
        new = [c for c in parent_cols if c not in tset]
        kind = "removed"
    if not new:
        return None
    counts: dict[str, int] = {}
    for c in new:
        head = c.split("_", 1)[0] if "_" in c else "base"
        counts[head] = counts.get(head, 0) + 1
    # 只认"差集里占多数"的前缀,避免单个跨族列把族判错
    fam, n = max(counts.items(), key=lambda kv: kv[1])
    if fam == "base" or n < 2:
        return None
    return fam, kind


def _parent_dirs(run_dir: Path | None, trial_data: dict
                 ) -> tuple[Path, Path] | None:
    """(本 trial 目录, 首个父代目录);缺 run_dir / trial_id / 父代时 None。"""
    tid = trial_data.get("trial_id")
    parents = trial_data.get("parent_trial_ids") or []
    if not tid or not parents or run_dir is None:
        return None
    root = Path(run_dir) / "trials"
    return root / str(tid), root / str(parents[0])


def _column_diff_family(run_dir: Path | None, trial_data: dict
                        ) -> tuple[str, str] | None:
    """按本 trial 与父代 ``model/feature_columns.json`` 的差集归族。"""
    dirs = _parent_dirs(run_dir, trial_data)
    if dirs is None:
        return None
    cols, pcols = _columns_of(dirs[0]), _columns_of(dirs[1])
    if not cols or not pcols:
        return None
    return _family_of_new_columns(cols, pcols)


def _recipe_fp_of(tdir: Path) -> str | None:
    """trial 训练配方的语义指纹(code/train.py;不可得返回 None)。"""
    src = tdir / "code" / "train.py"
    if not src.exists():
        return None
    try:
        from .recipe_fp import recipe_semantic_fingerprint

        fp = recipe_semantic_fingerprint(src.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return None if fp == "unparsable" else fp


def _is_seed_replicate(run_dir: Path | None, trial_data: dict) -> bool:
    """本 trial 是父代的换 seed 复测:列集相同**且**配方语义相同。

    列名相同不足以判定 —— 改分桶边界 / 修特征计算 bug / 换窗口但沿用
    列名都是真实的特征方向。配方指纹任一侧不可得时退一步:只有未声明
    family_hints(没有声称改了哪个族)才按复测处理。
    """
    dirs = _parent_dirs(run_dir, trial_data)
    if dirs is None:
        return False
    cols, pcols = _columns_of(dirs[0]), _columns_of(dirs[1])
    if not cols or not pcols or set(cols) != set(pcols):
        return False
    fp, pfp = _recipe_fp_of(dirs[0]), _recipe_fp_of(dirs[1])
    if fp and pfp:
        return fp == pfp
    return not trial_data.get("family_hints")


def _feature_family_from_trial(run_dir: Path, trial_data: dict) -> str | None:
    """特征 trial 的主前缀族推断(v18/P1-2)。

    v2.18.4:列集差集(最硬证据)已在 :func:`family_of_trial` 里先于
    family_hints 判定(:func:`_column_diff_family`);走到这里说明差集
    与 hint 都给不出结论,只剩下面两条软证据。

    数据源优先级:
    1. 本 trial 的 feature/group_importance.csv(评估侧已按前缀族
       裁决)——取 delta_mean 最大的族(排除 base);
    2. feature/feature_importance.csv 的 gain top 列前缀;
    3. 无加宽产物(trial 无自存新列族)→ None(调用方走 misc 兜底)。
    """
    tid = trial_data.get("trial_id")
    if not tid or run_dir is None:
        return None
    tdir = Path(run_dir) / "trials" / tid
    gi = tdir / "feature" / "group_importance.csv"
    if gi.exists():
        try:
            import pandas as pd

            df = pd.read_csv(gi)
            if len(df) and "family" in df.columns and \
                    "delta_mean" in df.columns:
                df = df[df["family"] != "base"]
                if len(df):
                    top = df.sort_values("delta_mean",
                                         ascending=False).iloc[0]
                    fam = str(top["family"]).strip()
                    if fam and fam != "base":
                        return fam
        except Exception:  # noqa: BLE001
            pass
    fi = tdir / "feature" / "feature_importance.csv"
    if fi.exists():
        try:
            import pandas as pd

            df = pd.read_csv(fi)
            if len(df) and "feature" in df.columns:
                cols = (trial_data.get("features") or {}).get("columns")
                # 无白名单时无法分辨"新增列",取 gain top 的前缀;
                # 白名单在时排除父代继承列更准,但 lineage 对比成本高,
                # gain top 已是本代模型里最重的信息
                top = str(df.iloc[0]["feature"])
                if "_" in top:
                    return top.split("_")[0]
        except Exception:  # noqa: BLE001
            pass
    return None


class DirectionLedger:
    """方向台账:刷新(读 trials)+ UCB 建议预算分配。"""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "directions.json"

    def refresh(self, trials: list, spec, *, persist: bool = True) -> dict:
        """从全量 trial 重建台账(幂等;每 finalize 后调用)。

        v12(P1-1):收益归因改 vs_parent(边际贡献),changed_modules
        为空的 trial 拒绝入账(退回 action 桶会让台账退化成"按动作
        分四组"——正是旧口径;实测 14 个 trial 中 8–9 个为空)。

        v2.15.x ROI 修复:
        1. confirm trial 的 runtime_sec 归入其父代 arm 的 cost_sec
           (全链路成本归因)。旧口径 confirm 不入账,其 3~4h 成本
           对台账完全不可见。
        2. 每个 arm 落盘 n_eff(相关性折减)和 lcb_delta(保守下界)。

        v2.17.0:delta 先换算为方向感知 gain 再聚合(见模块 docstring);
        n_accepted 优先读 evaluation.parent_verdict。
        """
        direction_up = metric_direction(spec.primary_metric)
        # trial_id → family 映射,用于 confirm 成本归因
        tid_to_family: dict[str, str] = {}
        arms: dict[str, dict] = {}
        skipped_no_modules = 0

        # 第一遍:构建所有非-confirm trial 的 family 映射
        for t in trials:
            if trial_is_screening(t):
                continue
            action = str(t.data.get("search_action") or "explore")
            if action == "confirm":
                continue
            mods = [str(m) for m in (t.data.get("changed_modules") or [])]
            dv = t.data.get("data_view") or "v1"
            key = family_of_trial(t.data, getattr(self, "run_dir", None))
            if not mods and dv == "v1" and key.startswith("action:"):
                continue
            tid_to_family[t.trial_id] = key

        # 第二遍:主循环——非 confirm trial 正常入账
        for t in trials:
            if trial_is_screening(t):
                continue
            action = str(t.data.get("search_action") or "explore")
            # confirm trial 不独立入账,其成本在第三遍归入父代
            if action == "confirm":
                continue
            # v2.17.0:内存准入超时(训练从未启动)不是方向证据——不计
            # n_tried,也不计成本(铁律 8)
            if _is_infra_failed(t):
                continue
            ev = (t.data.get("evaluation") or {})
            # P1-1(a):vs_parent(边际贡献)而非 vs_baseline(被祖先
            # 增益支配的累计量)。根 trial 无 vs_parent,其成本仍入账
            # 但收益记 0(建立起跑线的成本,不归任何方向)。
            delta = (ev.get("delta") or {}).get("vs_parent")
            key = family_of_trial(t.data, getattr(self, "run_dir", None))
            # P1-1(d):changed_modules 为空 → 方向归属无法判定,
            # 拒绝入账(记数供 planner-context 呈现),不再退回
            # action: 兜底桶
            mods = [str(m) for m in (t.data.get("changed_modules") or [])]
            dv = t.data.get("data_view") or "v1"
            if not mods and dv == "v1" and key.startswith("action:"):
                skipped_no_modules += 1
                continue
            arm = arms.setdefault(key, {
                "n_tried": 0, "n_accepted": 0, "deltas": [],
                "raw_deltas": [], "n_cap_confounded": 0,
                "cost_sec": 0.0, "last_generation": 0})
            arm["n_tried"] += 1
            arm["last_generation"] = max(arm["last_generation"],
                                         t.generation)
            runtime = (t.data.get("result") or {}).get("runtime_sec")
            if runtime:
                arm["cost_sec"] += float(runtime)
            # v2.17.2(A1):撞 cap 且无余量的负结论不是方向证据 —— 与 infra
            # 失败同处理:试次与成本照常入账,不记增益(不给负奖励)、不计
            # accepted;n_cap_confounded 计数(证伪判定扣除这部分试次)
            if _is_cap_confounded(t, direction_up):
                arm["n_cap_confounded"] += 1
                continue
            # v2.17.0:parent_verdict 优先(见模块 docstring)
            if accepted_for_ledger(ev):
                arm["n_accepted"] += 1
            # v2.17.0:方向感知增益(越小越好的指标取 −Δ)
            gain = direction_aware_gain(delta, direction_up)
            if gain is not None:
                arm["deltas"].append(gain)
                arm["raw_deltas"].append(float(delta))

        # 第三遍:confirm trial 的 runtime 归入其父代所在 arm
        # (全链路成本归因:一次 confirm 通常消耗父代训练时长的 50~200%,
        # 不入账导致 model:hpo/model:xgboost 的 dph 被系统性高估)
        for t in trials:
            if trial_is_screening(t):
                continue
            action = str(t.data.get("search_action") or "explore")
            if action != "confirm":
                continue
            runtime = (t.data.get("result") or {}).get("runtime_sec")
            if not runtime:
                continue
            # 按 parent_ids 找到父代 arm
            for pid in (t.parent_ids or []):
                parent_key = tid_to_family.get(pid)
                if parent_key and parent_key in arms:
                    arms[parent_key]["cost_sec"] += float(runtime)
                    break

        out: dict[str, Any] = {
            "updated_at": now_iso(),
            # v2.17.0:台账数值口径声明(消费方据此判读 mean_delta 等字段)
            "gain_convention": "direction_aware_gain_higher_is_better",
            "primary_metric": str(getattr(spec, "primary_metric", "") or ""),
            "metric_higher_is_better": bool(direction_up),
            "arms": {}}
        for key, arm in arms.items():
            ds = arm.pop("deltas")
            raw = arm.pop("raw_deltas")
            n = len(ds)
            mean_d = float(np.mean(ds)) if ds else 0.0
            std_d = float(np.std(ds, ddof=1)) if n > 1 else 0.0

            # n_eff:相关性折减(同一 arm 内多次观测来自同一配方子代,
            # 不是独立同分布;ρ̂ 由相邻 delta 相关系数估算,无法估算时
            # 使用先验 _RHO_PRIOR=0.7)
            if n >= 3:
                arr = np.array(ds)
                try:
                    rho = float(np.corrcoef(arr[:-1], arr[1:])[0, 1])
                    rho = max(0.0, min(rho, 0.99))
                except Exception:  # noqa: BLE001
                    rho = _RHO_PRIOR
            elif n == 2:
                rho = _RHO_PRIOR
            else:
                rho = 0.0
            n_eff = max(_N_EFF_MIN,
                        n / (1.0 + max(0.0, rho) * max(0, n - 1))) \
                if n > 0 else _N_EFF_MIN

            # LCB = mean_delta - z·(std/sqrt(n_eff))
            # 保守下界:低 n 或高 std 的 arm 主动降权;负均值截不截
            # 都留到 dph 截 0 统一处理
            se_eff = std_d / np.sqrt(n_eff) if n_eff > 0 and std_d > 0 else 0.0
            lcb_delta = mean_d - _LCB_Z * se_eff

            hours = arm["cost_sec"] / 3600.0
            # P1-1(b):量纲统一——去 0.1h 断裂,dph 恒为
            # lcb_delta / max(hours, eps);负 LCB 截 0(不给负 arm 预算)
            eff_hours = max(hours, _HOURS_EPS)
            dph = lcb_delta / eff_hours
            dph = dph if dph > 0 else 0.0
            out["arms"][key] = {
                **arm,
                # v2.17.0:mean/std/lcb_delta = 方向感知增益(越大越好)
                "mean_delta": round(mean_d, 8),
                "std_delta": round(std_d, 8),
                "lcb_delta": round(lcb_delta, 8),
                "mean_gain": round(mean_d, 8),
                "std_gain": round(std_d, 8),
                "lcb_gain": round(lcb_delta, 8),
                "mean_raw_delta": round(float(np.mean(raw)), 8)
                if raw else 0.0,
                "n_eff": round(n_eff, 3),
                "rho_est": round(rho, 4),
                "cost_hours": round(hours, 3),
                "delta_per_hour": round(dph, 8),
                "accept_rate": round(arm["n_accepted"]
                                     / max(arm["n_tried"], 1), 3),
            }
        out["skipped_no_changed_modules"] = skipped_no_modules
        self.state = out
        # v2.17.0:persist=False 供 planner 在内存里按新口径重建旧台账
        if persist:
            dump_json(self.path, out)
        return out

    def ucb_scores(self, *, c: float = UCB_C) -> dict:
        """各 arm 的 UCB 分(v2.17.0 从 :meth:`ucb_allocation` 拆出,可审计)。

        score = dph + c·scale·sqrt(ln total / n_eff);dph 已是方向感知增益
        的 LCB/hour(负值截 0),因此越小越好任务里"变差"的 trial 不会抬高
        该方向的 UCB。未试过的 arm 记 +inf。
        """
        st = getattr(self, "state", None)
        if st is None and self.path.exists():
            st = self.state = load_json(self.path)
        arms = (st or {}).get("arms") or {}
        if not arms:
            return {}
        total = sum(a["n_tried"] for a in arms.values()) or 1
        # 探索项量纲基准:arm 间 dph 的 std 自适应(退化时用典型量级)
        dph_vals = [float(a.get("delta_per_hour") or 0.0)
                    for a in arms.values()
                    if a["n_tried"] > 0]
        if len(dph_vals) >= 2:
            scale = float(np.std(dph_vals))
        else:
            scale = 0.0
        if scale <= 0:
            scale = float(np.mean(dph_vals)) if dph_vals else 0.01
        if scale <= 0:
            scale = 0.01
        scores = {}
        for key, a in arms.items():
            if a["n_tried"] == 0:
                scores[key] = float("inf")
            else:
                # n_eff 折减:相关观测不等于独立观测,探索项按 n_eff 计算
                n_eff = max(_N_EFF_MIN, float(a.get("n_eff") or a["n_tried"]))
                bonus = c * scale * np.sqrt(np.log(max(total, 2)) / n_eff)
                # 收益项用 delta_per_hour(已由 lcb_delta 驱动,见 refresh)
                scores[key] = float(a["delta_per_hour"]) + float(bonus)
        return scores

    def ucb_allocation(self, n_slots: int, *, c: float = UCB_C) -> dict:
        """UCB 分配下一代 n_slots 个 trial 预算(整数槽位,余数给 top)。

        v12(P1-1c):score = dph + c·scale·sqrt(ln total / n_eff),scale 按
        各 arm delta_per_hour 的 std 自适应(无方差时用全局典型量级
        0.01 兜底)——探索项与收益项同量纲,不再由试次数独占排序。
        未试过的 arm 视为无穷潜力(n_tried=0 → 先给 1 槽)。

        v2.15.x ROI 修复:UCB 收益项改用 lcb_delta / cost(全链路成本),
        探索项用 n_eff(相关性折减后的有效样本量)替代 n_tried。
        这使"观测次数多但高度相关"的 arm 不再获得虚假精度,也让
        LCB 归零的臂(低 n 或高 std)主动让出槽位。
        """
        if n_slots <= 0:
            return {}
        scores = self.ucb_scores(c=c)
        if not scores:
            return {}
        ranked = sorted(scores, key=lambda k: -scores[k])
        alloc = {k: 0 for k in ranked}
        # 有限预算下的整数分配:按 score 加权(探索项已同量纲,inf 按
        # top 有限值处理)
        finite = [s for s in scores.values() if np.isfinite(s)]
        base = max(finite) if finite else 1.0
        weights = {k: (base if not np.isfinite(scores[k]) else
                       max(scores[k], 0.0)) for k in ranked}
        wsum = sum(weights.values()) or 1.0
        for k in ranked:
            alloc[k] = int(n_slots * weights[k] / wsum)
        # 余数槽位按 score 从高到低补 1
        left = n_slots - sum(alloc.values())
        for k in ranked:
            if left <= 0:
                break
            alloc[k] += 1
            left -= 1
        return {k: v for k, v in sorted(alloc.items(),
                                        key=lambda kv: -kv[1]) if v > 0}

    def render_summary(self, top_k: int = 8) -> str:
        """planner-context 第 9 节的 markdown。

        v2.15.x:新增 LCB 和 n_eff 列,让 planner 看到保守下界和
        有效样本量,而不仅仅是 mean_delta 和 n_tried。
        """
        st = getattr(self, "state", None)
        if st is None and self.path.exists():
            st = load_json(self.path)
        arms = (st or {}).get("arms") or {}
        if not arms:
            return ("(方向台账为空:首个 finalize-generation 后生成;"
                    "changed_modules 为空的 trial 不入账——"
                    "new-trial 必须声明改动模块)")
        rows = sorted(arms.items(),
                      key=lambda kv: -kv[1].get("delta_per_hour", 0))
        _conv = (st or {}).get("gain_convention")
        L = []
        if _conv:
            _hb = (st or {}).get("metric_higher_is_better", True)
            L.append(f"(增益口径:方向感知,越大越好;主指标 "
                     f"{(st or {}).get('primary_metric') or '?'} "
                     f"{'越大越好 → gain=+Δ' if _hb else '越小越好 → gain=−Δ'})")
            L.append("")
        L += ["| 方向族 | 试次 | n_eff | accepted | mean_gain | LCB | "
              "全链路Δ/hour | 上次代 |",
              "|---|---|---|---|---|---|---|---|"]
        for key, a in rows[:top_k]:
            lcb = a.get("lcb_delta", a.get("mean_delta", 0.0))
            n_eff = a.get("n_eff", a["n_tried"])
            L.append(f"| {key} | {a['n_tried']} | {n_eff:.1f} "
                     f"| {a['n_accepted']} "
                     f"| {a['mean_delta']:+.5f} "
                     f"| {lcb:+.5f} "
                     f"| {a['delta_per_hour']:.2e} "
                     f"| g{a['last_generation']:02d} |")
        skipped = (st or {}).get("skipped_no_changed_modules") or 0
        if skipped:
            L.append(f"\n(另有 {skipped} 个 trial 因 changed_modules 为空"
                     "被拒绝入账;new-trial 必须声明 --changed-modules)")
        # v2.17.2(A1):撞 cap 混杂的 trial 只计试次/成本,不计增益
        _cc = {k: int(a.get("n_cap_confounded") or 0) for k, a in arms.items()
               if int(a.get("n_cap_confounded") or 0) > 0}
        if _cc:
            L.append("\n(撞 cap 且无余量的负结论不计增益/负奖励 —— "
                     + ", ".join(f"{k}: {v} 个" for k, v in sorted(_cc.items()))
                     + ";见 evaluation.cap_confounded)")
        return "\n".join(L)
