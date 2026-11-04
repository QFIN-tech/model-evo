"""Leaderboard(§15.1)与 Candidate Archive(§9.4/§15.2)。

Leaderboard 记录所有 Trial 的成绩;Archive 只保留值得继续演进的候选。
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .confirmation import (delivery_tolerance, divergence_summary,
                           fresh_confirmed, recipe_root)
from .logging_utils import (dump_json, dump_yaml, get_logger, load_yaml,
                            now_iso)
from .metrics import metric_direction

logger = get_logger(__name__)
from .spec import TaskSpec
from .trial import (TrialRecord, lineage_of, trial_currently_failed,
                    trial_is_screening)

LEADERBOARD_COLUMNS = [
    "trial_id", "algorithm", "model_num", "generation",
    "search_action", "changed_modules", "data_view", "parent",
    # v2.14.0:recipe = 配方根(confirm 子代归父代配方);role 标出搜索
    # 冠军/交付位。run2 的表按 auc 把 rejected 的 trial-022 排第 1、
    # 交付的 trial-011 排第 12,两者关系在表里完全不可见。
    "recipe",
    "primary_metric", "delta_vs_parent",
    "delta_vs_baseline", "p_better", "p_positive", "delta_ci", "stability",
    "runtime_sec", "verdict", "role", "recipe_fingerprint",
    # v2.13.2:screening-only 标记单列。与 reference_only(automl 外部锚点)
    # 分开表达 —— 两者都是"在表里可见但不参与选择",但 screening 是
    # **内部先验**(贡献 group_importance),automl 是**外部证词**,
    # 共用一个列会把 report/交付校验的口径搅在一起。
    "screen_only", "reference_only",
    "failure_reason", "verdict_reason", "harness_epoch", "lineage",
]
# build() 输出时在首列插入 rank
RANKED_COLUMNS = ["rank", *LEADERBOARD_COLUMNS]
# secondary 指标列(ks/logloss/lift_top10 等)按 spec 与各 trial
# evaluation.secondary_metrics 的并集展开,统一插在主指标列之后


def _recipe_fp(tdir) -> str | None:
    """配方的语义指纹前 12 位(实现见 :mod:`runtime.recipe_fp`)。

    读失败或源码不存在时返回 None(呈现层,不阻塞落表)。
    """
    if not tdir:
        return None
    try:
        from .recipe_fp import recipe_semantic_fingerprint

        src = Path(tdir) / "code" / "train.py"
        if not src.exists():
            return None
        fp = recipe_semantic_fingerprint(src.read_text(encoding="utf-8"))
        return None if fp == "unparsable" else fp[:12]
    except Exception:  # noqa: BLE001
        return None


class Leaderboard:
    """总 Leaderboard:全量 Trial(含 failed/invalid),CSV + JSON + MD 三格式。

    MD 为人类可读交付视图,指标列保留 METRIC_DECIMALS 位小数。
    """

    METRIC_DECIMALS = 4

    def __init__(self, run_dir: Path, spec: TaskSpec, config: Config = None):
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / "leaderboard"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.config = config
        self.direction = metric_direction(spec.primary_metric)
        self._baseline_view: str | None = None
        self._resolved_baseline_id: str | None = None

    def build(self, trials: list[TrialRecord], baseline_id: str | None = None) -> \
            pd.DataFrame:
        by_id = {t.trial_id: t for t in trials}
        metric_name = self.spec.primary_metric  # 列名用实际指标,如 auc
        # v2.10.1/F7:记录交付可比视图(best_trial 用;跨视图 AUC 不可比)
        # v2.11.6:同一函数返回**解析后**的 baseline_id,回填到落盘
        # JSON。旧实现把调用方原始参数直接写 JSON,而
        # evolution.finalize() 只传 trials 不传 baseline_id →
        # leaderboard.json 的 baseline_id 恒为 null:report §5 /
        # planner-context / 一切读 JSON 而非 CSV 的消费方都拿不到基线
        # 锚,report 直接打出与事实相反的"baseline 未完成"(实测
        # drawdown-req-t7-v211)。CSV/MD 投影一直正确(它们另走解析),
        # 只有 JSON 这条漏了。
        resolved_bid = self._remember_baseline_view(trials, baseline_id)
        self._resolved_baseline_id = resolved_bid
        # v2.11.6:局部变量同步为**解析后**的值,供下方 JSON 落盘处的
        # 兜底使用。解析职责已由 _remember_baseline_view 完整承担
        # (显式参数 → run-state.json → 启发式视图),此处只做赋值。
        #
        # 修复记录:旧实现在此处残留了一段重复的解析分支,调用
        #   self._resolve_baseline_id(trials)
        # ——该方法**从未定义在 Leaderboard 类上**(同名函数只存在于
        # CLI 模块层,签名为 (run_dir, spec, factory),语义是"读了
        # 当前试次集合")。由于该行只在 `if not baseline_id:` 分支里
        # 执行,而调用方 evolution.finalize() 恰好不传 baseline_id,
        # 于是 finalize-generation **每次都**抛
        #   AttributeError: 'Leaderboard' object has no attribute
        #   '_resolve_baseline_id'
        # (runtime/evolution.py:246 -> runtime/leaderboard.py:80),
        # 整个演进循环的收尾步骤被卡死,stop_report / archive_update /
        # generation_delta 全部产不出来。
        baseline_id = resolved_bid
        # secondary 指标列(如 ks/logloss/lift_top10),插在主指标之后;
        # 沿全量 trial 出现过的键取并集,保持列稳定
        secondary_names: list[str] = list(
            self.spec.data.get("secondary_metrics") or [])
        for t in trials:
            for k in ((t.data.get("evaluation") or {})
                      .get("secondary_metrics") or {}):
                if k not in secondary_names:
                    secondary_names.append(k)
        rows = []
        for t in trials:
            d = t.data
            result = d.get("result") or {}
            evaluation = d.get("evaluation") or {}
            primary = (evaluation.get("primary_metric") or {}).get("value")
            if primary is None:
                primary = result.get(metric_name)
            stability = evaluation.get("stability") or {}
            parents = t.parent_ids
            # model_num:该 Trial 交付的模型数(单模型=1,集成/多模型>1);
            # 来自 Trial metrics.json 的声明,未声明按单模型口径补 1
            complexity = evaluation.get("complexity") or {}
            model_num = complexity.get("model_num")
            if model_num is None:
                model_num = int((d.get("result") or {})
                                .get("model_num") or 1)
            sec_values = evaluation.get("secondary_metrics") or {}
            # v12(P0-2):p_positive / delta_ci 进 leaderboard 与
            # planner-context——排名与"提升"的显著性必须同表可见
            pb = ((evaluation.get("delta") or {})
                  .get("vs_parent_paired_bootstrap") or {})
            dp_ci = pb.get("delta_ci")
            # v18/P2-3:automl 参照行标注 reference_only——不计入交付
            # 候选(best_trial 过滤),external anchor 证词仍在表内可见
            is_automl_ref = bool(
                str(d.get("search_action") or "") == "automl"
                or ((d.get("metrics") or {}).get("automl_reference")))
            rows.append({
                "trial_id": t.trial_id,
                "algorithm": d.get("algorithm"),
                "model_num": int(model_num) if model_num else 1,
                "generation": d.get("generation", 0),
                "search_action": d.get("search_action"),
                # changed_modules(v8,结构化声明):分号拼接便于 CSV 呈现
                "changed_modules": ";".join(d.get("changed_modules") or [])
                or None,
                "reference_only": is_automl_ref or None,
                # v2.13.2:screening-only(宽面初筛)在表里可审计,但不是
                # 候选 —— 见 trial_is_screening 的调用点清单
                "screen_only": True if d.get("screen_only") else None,
                "data_view": d.get("data_view") or "v1",
                "parent": ";".join(parents) if parents else None,
                "recipe": recipe_root(t, by_id),
                metric_name: primary,
                **{name: sec_values.get(name) for name in secondary_names},
                "delta_vs_parent": (evaluation.get("delta") or {}).get("vs_parent"),
                "delta_vs_baseline": (evaluation.get("delta") or {}).get("vs_baseline"),
                # v2.17.0:锚定晋级(promotion_reference=anchor)的对照列;
                # 只有 run 内出现过 vs_anchor 时才输出(旧 run 表结构不变)
                "delta_vs_anchor": (evaluation.get("delta") or {}).get(
                    "vs_anchor"),
                "p_better_vs_anchor": ((evaluation.get("delta") or {}).get(
                    "vs_anchor_paired_bootstrap") or {}).get("p_better"),
                "anchor_id": (evaluation.get("delta") or {}).get("anchor_id"),
                "p_better": pb.get("p_better"),
                "p_positive": pb.get("p_positive"),
                "delta_ci": (None if dp_ci in (None, "")
                             else f"[{dp_ci[0]:+.5f},{dp_ci[1]:+.5f}]"
                             if isinstance(dp_ci, (list, tuple))
                             and len(dp_ci) == 2
                             and all(isinstance(v, (int, float))
                                     for v in dp_ci) else str(dp_ci)),
                "stability": stability.get("multi_seed_std"),
                # v2.18.4:同配方的新 seed 复测(trial-011 vs trial-008 是
                # 同一 1620 列面换 seed)在表里不可分辨;落 recipe_fingerprint
                # 让读者/planner 一眼看出哪些行是同一训练配方。
                "recipe_fingerprint": _recipe_fp(getattr(t, "trial_dir", None)),
                "runtime_sec": result.get("runtime_sec"),
                "verdict": evaluation.get("verdict") or (
                    "invalid" if d.get("status") == "FAILED" else None),
                "failure_reason": d.get("failure_reason"),
                # v2.18.4:被拒/未确认的方向原因在 evaluation.reason 里
                # (切片门 / 无父代可比 / 相对父代无提升 …)。旧口径只落
                # failure_reason(运行时失败),于是 7 个 rejected 行在表里
                # 全空白 —— 而交付校验要求"leaderboard 含原因"。
                "verdict_reason": evaluation.get("reason"),
                "harness_epoch": d.get("harness_epoch"),
                "lineage": " > ".join(lineage_of(t, by_id)),
            })
        # v14(铁律 10):跨 harness epoch 的 trial 分数不可比——
        # acknowledge-harness-change --allow-measurement-impact 会把
        # epoch +1,同表混排新旧 epoch 等于抹掉"重新 baseline"语义。
        # 拒绝混排:多 epoch 并存时只保留最新 epoch(旧行打标记),
        # 并把被排除的 trial 写进 manifest 供审计。
        dropped_epochs: list[dict] = []
        if rows:
            epochs = {r.get("harness_epoch") for r in rows
                      if r.get("harness_epoch") is not None}
            if len(epochs) > 1:
                latest = max(epochs)
                keep = [r for r in rows
                        if r.get("harness_epoch") in (None, latest)]
                dropped_epochs = [
                    {"trial_id": r["trial_id"],
                     "harness_epoch": r.get("harness_epoch")}
                    for r in rows if r not in keep]
                logger.warning(
                    "leaderboard 拒绝跨 harness_epoch 混排:"
                    "保留 epoch=%s,排除 %d 个旧 epoch trial(见 manifest)"
                    "——铁律 10:不同协议下的分数不可比",
                    latest, len(dropped_epochs))
                rows = keep
        # v2.14.0:搜索冠军 / 交付位角色标注(与 best_trial 同源解析)
        try:
            resolution = self.delivery_resolution(trials)
        except Exception as exc:  # noqa: BLE001  呈现层失败不阻塞落表
            logger.warning("交付位解析失败(role 列留空): %s", exc)
            resolution = {}
        for r in rows:
            roles = []
            if r["trial_id"] == resolution.get("search_champion"):
                roles.append("search_champion")
            if r["trial_id"] == resolution.get("delivery_trial"):
                roles.append("delivery")
            r["role"] = "+".join(roles) or None
        df = pd.DataFrame(rows)
        if len(df):
            # direction=True 表示越大越好 → 降序;logloss 等越小越好 → 升序
            df = df.sort_values(
                [metric_name, "trial_id"],
                ascending=[not self.direction, True]).reset_index(drop=True)
        else:
            # v17.1:零 trial 时给出主指标名占位列(旧空帧列名是
            # 'primary_metric',而下方 head 用 metric_name='auc' 选列
            # → KeyError "['auc'] not in index",leaderboard 裸调用崩溃)
            df = pd.DataFrame(columns=[metric_name, *LEADERBOARD_COLUMNS])
        df.insert(0, "rank", range(1, len(df) + 1))
        # 列序:rank + 固定列(secondary 指标列插在主指标之后)
        # v14.2:models_so_far 列移除(按 rank 累积的模型族数对选型/
        # crossover 无实际消费方,crossover 互补父代走 archive.
        # diverse_parents;模型族覆盖汇总仍在 manifest/MD 底部)
        head = ["rank", "trial_id", "algorithm", "model_num",
                "generation", "search_action",
                "changed_modules", "data_view",
                "parent", "recipe", metric_name]
        tail = ["delta_vs_parent", "delta_vs_baseline", "p_better", "p_positive",
                "delta_ci", "stability",
                "runtime_sec", "verdict", "role", "recipe_fingerprint",
                # v2.13.2:两类"在表里可见但不参与选择"的行标记。
                # reference_only 此前只在行 dict 里、没进 head/tail,于是
                # CSV 里被静默丢掉 —— 与它的设计意图("参照行标注")不符。
                "screen_only", "reference_only",
                "failure_reason", "verdict_reason",
                "harness_epoch", "lineage"]
        sec_present = [c for c in secondary_names if c in df.columns]
        # v2.17.0:锚定晋级列仅在 run 内存在 vs_anchor 时输出
        anchor_cols = []
        if "delta_vs_anchor" in df.columns and \
                df["delta_vs_anchor"].notna().any():
            anchor_cols = ["delta_vs_anchor", "p_better_vs_anchor",
                           "anchor_id"]
            _i = tail.index("delta_vs_baseline") + 1
            tail = tail[:_i] + anchor_cols + tail[_i:]
        cols = head + sec_present + tail
        df_out = df[cols]
        df_out.to_csv(self.dir / "leaderboard.csv", index=False)
        # 模型族覆盖汇总(非 invalid trial 的 algorithm 分布)
        valid = df_out[df_out["verdict"].fillna("") != "invalid"] \
            if "verdict" in df_out.columns else df_out
        fam_counts = valid["algorithm"].value_counts().to_dict() \
            if "algorithm" in valid.columns and len(valid) else {}
        dump_json(self.dir / "leaderboard.json",
                  {"updated_at": now_iso(),
                   "primary_metric": self.spec.primary_metric,
                   "baseline_id": getattr(self, "_resolved_baseline_id",
                                          None) or baseline_id,
                   "n_models": int(len(fam_counts)),
                   "total_model_num": int(sum(
                       r.get("model_num") or 1
                       for r in df_out.to_dict("records"))),
                   "model_family_counts": {str(k): int(v)
                                            for k, v in fam_counts.items()},
                   # v14:跨 epoch 被排除的 trial(铁律 10:不混排)
                   "excluded_cross_epoch_trials": dropped_epochs,
                   # v2.14.0:搜索冠军 vs 交付位(report/交付校验消费)
                   "delivery_resolution": resolution,
                   "rows": df_out.to_dict("records")})
        self._write_markdown(df_out, resolution)
        return df_out

    def _write_markdown(self, df: pd.DataFrame,
                        resolution: dict | None = None) -> None:
        """leaderboard.md:指标列保留 METRIC_DECIMALS 位小数,交付视图。"""
        import math

        metric_name = self.spec.primary_metric
        # 指标列 = 主指标 + secondary 指标列 + delta/stability;
        # LEADERBOARD_COLUMNS 用占位名 primary_metric,主指标真实列名
        # 就是 metric_name,须显式排除,否则会二次 map(先格式化成
        # 字符串再落入 "—" 分支,MD 里主指标全变 —)
        _anchor_cols = ("delta_vs_anchor", "p_better_vs_anchor", "anchor_id")
        sec_cols = [c for c in df.columns
                    if c not in LEADERBOARD_COLUMNS
                    and c not in _anchor_cols
                    and c not in (metric_name, "rank")]
        metric_cols = [metric_name, *sec_cols, "delta_vs_parent",
                       "delta_vs_baseline", "delta_vs_anchor",
                       "p_better_vs_anchor", "p_better", "p_positive",
                       "stability"]
        md = df.copy()
        for c in metric_cols:
            if c in md.columns:
                md[c] = md[c].map(
                    lambda v: (f"{v:.{self.METRIC_DECIMALS}f}"
                               if isinstance(v, (int, float))
                               and math.isfinite(v) else "—"))
        task_id = getattr(self.spec, "task_id", "")
        lines = [
            f"# Leaderboard — {task_id}".rstrip(),
            "",
            f"- primary_metric: **{metric_name}**"
            f"({'越大越好' if self.direction else '越小越好'})",
            f"- secondary_metrics: "
            f"{sec_cols if sec_cols else '(无)'}",
            f"- updated_at: {now_iso()}",
        ]
        if resolution and resolution.get("search_champion"):
            lines.append(f"- 搜索冠军: **{resolution['search_champion']}**"
                         f"(配方 {resolution.get('search_recipe')})")
            lines.append(f"- 交付位: **{resolution.get('delivery_trial') or '空缺'}**"
                         f"(status={resolution.get('status')})")
            if resolution.get("status") != "bound":
                lines.append(f"- ⚠ {divergence_summary(resolution)}")
        lines += [
            "",
            "| " + " | ".join(md.columns) + " |",
            "|" + "---|" * len(md.columns),
        ]
        for _, row in md.iterrows():
            cells = []
            for c, v in zip(md.columns, row):
                if isinstance(v, float) and pd.isna(v):
                    cells.append("—")
                elif v is None or (isinstance(v, float) and pd.isna(v)):
                    cells.append("")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        # 模型族覆盖(§9.2 crossover 互补父代选择的参考)
        valid = df[df["verdict"].fillna("") != "invalid"] \
            if "verdict" in df.columns else df
        fam_counts = valid["algorithm"].value_counts().to_dict() \
            if "algorithm" in valid.columns and len(valid) else {}
        total_mn = int(sum(int(r.get("model_num") or 1)
                           for _, r in df.iterrows()))
        lines += ["", "## 模型族覆盖", "",
                  f"- n_models: **{len(fam_counts)}**",
                  f"- 累计交付模型数(total_model_num): **{total_mn}**"
                  "(单模型 trial 计 1,集成计其声明模型数)"]
        for k, v in fam_counts.items():
            lines.append(f"- {k}: {int(v)} 个有效 trial")
        (self.dir / "leaderboard.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    def _remember_baseline_view(self, trials: list[TrialRecord],
                                baseline_id: str | None) -> str | None:
        """v2.10.1/F7:记录基线所属 data_view(交付可比口径)。

        v2.11.6:返回**解析后**的 baseline_id(供 build() 回填落盘 JSON)。
        解析顺序:显式参数 → run-state.json.baseline_id → None。
        两个消费方共用同一次解析,不再各解析一遍(旧实现只在内部
        用局部变量,调用方拿不到解析结果,JSON 于是漏写)。
        """
        self._baseline_view = None
        if not baseline_id:
            # v2.10.1/F7:调用方(evolution.finalize)只传 trials,不传
            # baseline_id;从 run-state 兜底读,避免退回"最多数视图"启发式。
            # v2.11.6:run-state.json 在 run 根,不在 leaderboard/ 子目录
            # ——旧实现只查 `self.dir / "run-state.json"`,该路径**从不
            # 存在**,于是这条兜底解析自 v2.10.1 起全程空转(实测
            # drawdown-req-t7-v211:baseline_id 静默退化为
            # "最多数非抽样视图"启发式,只是恰好等于 v1 而未被发现)。
            # 两个位置都查,兼容 __new__ 构造(无 run_dir)的测试路径。
            for _cand in (getattr(self, "run_dir", None), self.dir):
                if _cand is None:
                    continue
                _st = Path(_cand) / "run-state.json"
                if not _st.exists():
                    continue
                try:
                    baseline_id = (json.loads(_st.read_text(encoding="utf-8"))
                                   or {}).get("baseline_id")
                except (OSError, ValueError):
                    baseline_id = None
                if baseline_id:
                    break
        if not baseline_id:
            # 未显式给基线时退回"非抽样视图"里 trial 数最多者
            views = [str(t.data.get("data_view") or "v1") for t in trials]
            non_sample = [v for v in views if v != "sample"]
            if non_sample:
                from collections import Counter
                self._baseline_view = Counter(non_sample).most_common(1)[0][0]
            return None
        for t in trials:
            if t.trial_id == baseline_id:
                self._baseline_view = str(t.data.get("data_view") or "v1")
                return baseline_id
        # v2.11.6:baseline_id 已记录但该 trial 不在本批 trials 里
        # (调用方只传了子集)——仍原样返回,避免把有效的记录抹成 None。
        return baseline_id

    def best_trial(self, trials: list[TrialRecord],
                   shadow_recheck: bool = False,
                   delivery: bool = False) -> TrialRecord | None:
        """冠军选择(v12,P0-3 修复):不再裸 argmax。

        裸 argmax 的期望上偏 = N 次含噪估计的最大值偏差,量级与全部
        代际提升相同(实测前 11 名 AUC 全距 0.00194 < 自报噪声带,
        冠军是抽签抽出来的)。新口径:
        1. 先按 verdict∈{accepted, confirmed} 过滤(裁决面硬约束);
        2. 无候选时返回 None，绝不让 rejected/invalid 试验成为冠军;
        3. 按收缩分 metric − k·noise 排序(k=config
           evaluator.selection_shrinkage,默认 1.0):对噪声大的分数
           收缩,抑制 winner's curse;
        4. ``shadow_recheck`` 仅为旧调用兼容，已不再改变排名；shadow
           只允许 confirm trial 作独立等价性确认，不能成为第二选择集。

        v2.14.0:``delivery=True`` 改走 :meth:`delivery_resolution` ——
        交付位必须**绑定搜索冠军的配方**。旧口径取"任意曾过门的
        confirmed trial",run2 冠军换了两次,gen-03 对 trial-006 的确认
        仍然有效,交付静默落到落后 1.7×MDE 的 trial-011。解析明细留在
        ``self.last_delivery_resolution`` 供调用方呈现。
        """
        if shadow_recheck:
            logger.warning(
                "shadow_recheck 已弃用且被忽略；shadow 仅用于独立 confirm")
        if delivery:
            res = self.delivery_resolution(trials)
            if res.get("delivery_trial") is None:
                logger.error("无可用交付冠军:%s", divergence_summary(res))
                return None
            return next((t for t in trials
                         if t.trial_id == res["delivery_trial"]), None)
        ranked = self._rank(self._eligible(self._selection_pool(trials)))
        if not ranked:
            logger.error("无可用冠军:%s",
                         "没有 verdict=accepted/confirmed 且完整性有效的 Trial")
            return None
        return ranked[0]

    def _selection_pool(self, trials: list[TrialRecord]) -> list:
        """可比池:有主指标、非 automl、与基线同视图、最新 harness epoch。"""
        scored = [t for t in trials
                  if ((t.data.get("evaluation") or {})
                      .get("primary_metric") or {}).get("value") is not None
                  # v18/P2-3:automl 参照行不计入交付候选(external
                  # anchor 证词,不是搜索产物)
                  and str(t.data.get("search_action") or "") != "automl"]
        if not scored:
            return []
        # v2.10.1/F7:交付冠军必须来自**与基线同数据视图**的可比池。
        # 抽样视图(sample ≤20 万行)与全量视图的 validation 行集不同,
        # 两个 AUC 不可比;裸比会系统性偏向行数少的视图(实测 sample 面
        # 行连续三代压过 v1 真冠军,并被写进 generation summary 的
        # best_trial 字段,直接误导交付选型)。找不到基线视图时退回全量。
        # v2.11.7:best_trial 可能被**未经 build()** 的路径调用
        # (实测 cmd_evaluate_final:`Leaderboard(...)` 后直接
        # best_trial,不调 build)⇒ `_baseline_view` 仍是 __init__ 里的
        # None ⇒ F7 过滤被静默跳过 ⇒ 交付选型落到 sample 面 trial-014
        # (0.7799447),而 finalize-retrain 走了 build ⇒ 选 v1 面
        # trial-011(0.779918)。二者差 0.000027,冠军口径分裂 ⇒
        # evaluate-final 的冠军单源校验拒绝执行,交付链卡死(盲测机会
        # 未消耗)。此处自行解析一次(幂等:重复调用只重设同一视图),
        # 使选型结果与"是否调用过 build()"无关。
        _delivery_view = getattr(self, "_baseline_view", None)
        if _delivery_view is None:
            try:
                self._remember_baseline_view(trials, None)
            except Exception:  # noqa: BLE001  解析失败退回全量池
                pass
            _delivery_view = getattr(self, "_baseline_view", None)
        if _delivery_view:
            _same = [t for t in scored
                     if str(t.data.get("data_view") or "v1") == _delivery_view]
            if _same:
                scored = _same
        # v14.2:与 build() 的 epoch 门控同口径——冠军只能从"与 leaderboard
        # 同表可比"的池子里选(最新 epoch + 无 epoch 标记的旧 trial)。
        # 旧实现不看 epoch:leaderboard 已排除旧 epoch 行,best_trial 仍可能
        # 把冠军落在被排除的行上,evaluate-final/停止判定与表面对不上。
        epochs = {t.data.get("harness_epoch") for t in scored
                  if t.data.get("harness_epoch") is not None}
        if len(epochs) > 1:
            latest = max(epochs)
            kept = [t for t in scored
                    if t.data.get("harness_epoch") in (None, latest)]
            if kept:
                scored = kept
        return scored

    @staticmethod
    def _eligible(pool: list, *, fresh_only: bool = False) -> list:
        return [t for t in pool
                if (t.data.get("evaluation") or {}).get("verdict")
                in (("confirmed",) if fresh_only
                    else ("accepted", "confirmed"))
                # v2.13.2:screening-only 不作冠军 —— 它被显式排除在
                # 代级 FDR 之外,拿它交付等于绕过整条选择校正链。
                and not trial_is_screening(t)
                and (not fresh_only or fresh_confirmed(t))
                and ((t.data.get("evaluation") or {}).get("integrity")
                     or {}).get("valid", True) is not False
                and not trial_currently_failed(t)]

    @staticmethod
    def _noise(t: TrialRecord) -> float:
        pb = ((t.data["evaluation"].get("delta") or {})
              .get("vs_parent_paired_bootstrap") or {})
        n = pb.get("se") or t.data["evaluation"].get("noise")
        try:
            n = float(n)
        except (TypeError, ValueError):
            return 0.0
        return n if np.isfinite(n) and n > 0 else 0.0

    def _rank(self, pool: list) -> list:
        """收缩分 metric − k·noise 降序:高噪声候选的分位被拉低,排名对
        bootstrap 抖动稳健。"""
        k = float(self.config.get("evaluator.selection_shrinkage", 1.0)) \
            if getattr(self, "config", None) else 1.0

        def _shrink(t: TrialRecord) -> float:
            v = t.data["evaluation"]["primary_metric"]["value"]
            return (v if self.direction else -v) - k * self._noise(t)
        return sorted(pool, key=lambda t: (-_shrink(t), t.trial_id))

    def delivery_resolution(self, trials: list[TrialRecord]) -> dict:
        """交付位解析(v2.14.0):交付必须绑定搜索冠军的配方。

        - ``bound``:存在通过独立确认、且与搜索冠军同配方(配方根相同)
          的 trial ⇒ 取其中收缩分最高者;
        - ``stale_within_tolerance``:冠军配方未确认,但有其它配方的
          已确认 trial 与冠军差距 ≤ 容差(默认 MDE,且 ≥2×noise)⇒ 可用,
          明细留痕;
        - ``diverged``:冠军未确认,且所有已确认 trial 都落后超过容差
          ⇒ **交付位空缺**(run2 事故形态:trial-011 落后 0.00157);
        - ``unconfirmed``:冠军未确认且 run 内无任何已确认 trial;
        - ``no_search_champion``:无合法搜索冠军。
        """
        pool = self._selection_pool(trials)
        by_id = {t.trial_id: t for t in trials}
        search_ranked = self._rank(self._eligible(pool))
        confirmed = self._rank(self._eligible(pool, fresh_only=True))

        def _val(t) -> float:
            return float(t.data["evaluation"]["primary_metric"]["value"])

        res: dict[str, Any] = {
            "search_champion": None, "search_recipe": None,
            "search_metric": None, "delivery_trial": None,
            "delivery_recipe": None, "status": "no_search_champion",
            "gap": None, "tolerance": None,
            "bound_candidates": [], "stale_candidates": [],
        }
        if not search_ranked:
            self.last_delivery_resolution = res
            return res
        champ = search_ranked[0]
        champ_recipe = recipe_root(champ, by_id)
        tol = delivery_tolerance(getattr(self, "config", None),
                                 self._noise(champ),
                                 run_dir=getattr(self, "run_dir", None))
        res.update(search_champion=champ.trial_id, search_recipe=champ_recipe,
                   search_metric=_val(champ), tolerance=tol)
        bound = [t for t in confirmed if recipe_root(t, by_id) == champ_recipe]
        res["bound_candidates"] = [t.trial_id for t in bound]
        if bound:
            pick = bound[0]
            res.update(status="bound", delivery_trial=pick.trial_id,
                       delivery_recipe=champ_recipe,
                       gap=self._gap(_val(champ), _val(pick)))
            self.last_delivery_resolution = res
            return res
        # v2.20.0:先验豁免 —— 冠军配方未确认,但其增益"换 seed 后仍在"
        # (z ≥ z_required,σ_seed 取实测)且动作在豁免白名单内 ⇒ 交付位
        # 直接绑定冠军,status=waived_by_prior。只放开确认门,不放开裁决:
        # 完整性 / 当前失败 / screening 已由 _eligible 拦过。
        _waive = self._prior_waiver(champ, trials, by_id)
        if _waive.get("passed"):
            res.update(status="waived_by_prior",
                       delivery_trial=champ.trial_id,
                       delivery_recipe=champ_recipe,
                       gap=0.0, prior_waiver=_waive)
            self.last_delivery_resolution = res
            return res
        # 判定结论(通过与否、为什么没通过)一律留痕供审计:报告需要解释
        # "为什么这个冠军仍需 confirm"
        res["prior_waiver"] = _waive
        stale = []
        for t in confirmed:
            stale.append({"trial_id": t.trial_id,
                          "recipe": recipe_root(t, by_id),
                          "metric": _val(t),
                          "gap": self._gap(_val(champ), _val(t))})
        res["stale_candidates"] = stale
        within = [s for s in stale if s["gap"] <= tol]
        if within:
            pick = within[0]
            res.update(status="stale_within_tolerance",
                       delivery_trial=pick["trial_id"],
                       delivery_recipe=pick["recipe"], gap=pick["gap"])
        elif stale:
            res.update(status="diverged", gap=stale[0]["gap"])
        else:
            res["status"] = "unconfirmed"
        self.last_delivery_resolution = res
        return res

    def _gap(self, champ_value: float, other_value: float) -> float:
        """方向感知差距:冠军比 other 好多少(≥0 表示 other 落后)。"""
        return float(champ_value - other_value) if self.direction \
            else float(other_value - champ_value)

    def _prior_waiver(self, trial, trials: list, by_id: dict) -> dict:
        """v2.20.0:先验豁免判定(见 runtime.confirmation.seed_robust_gain)。

        只有 ``evaluator.prior_waiver.enabled`` 打开、且该 trial 的动作在
        ``evaluator.prior_waiver_actions`` 白名单里时才评估。σ_seed 取实测:
        本 run 同配方复现组 → 经验库同 data_base 的历史比值 × 本次 SE;
        拿不到实测就返回 ``available=False``(不豁免)。

        config 为 None 时从 run 目录惰性加载 —— finalize-retrain 的
        ``Leaderboard(run_dir, spec)`` 不带 config,而豁免必须读到 run 配置。
        惰性加载只服务于本判定,不改动 ``delivery_tolerance`` 等其他口径
        (它们仍按 config=None 的旧行为走)。
        """
        from .confirmation import (pooled_replicate_sd, recipe_root,
                                   seed_robust_gain)

        config = getattr(self, "config", None)
        if config is None:
            try:
                from .config import Config

                rc = Path(self.run_dir) / "modelevo-config.yaml"
                config = Config.load(run_config=rc if rc.exists() else None)
            except Exception:  # noqa: BLE001  配置读不到 = 不豁免
                return {"available": False, "reason": "配置不可读"}
        cfg = config.get("evaluator.prior_waiver", None)
        if not isinstance(cfg, dict) or not cfg.get("enabled"):
            return {"available": False, "reason": "prior_waiver 未启用"}
        actions = list(config.get("evaluator.prior_waiver_actions",
                                  ["warm_start"]) or ["warm_start"])
        action = str(trial.data.get("search_action") or "")
        if action not in actions:
            return {"available": False,
                    "reason": f"动作 {action!r} 不在豁免白名单 {actions}"}
        # σ_seed:本 run 同配方复现组(必须排除冠军自身,否则自证)→ 经验库比值
        sigma, sigma_src = None, None
        try:
            others = [t for t in trials if t.trial_id != trial.trial_id]
            rep = pooled_replicate_sd(
                others, data_view=str(trial.data.get("data_view") or "v1"))
            if rep.get("sd"):
                sigma, sigma_src = float(rep["sd"]), "run_replicates"
        except Exception:  # noqa: BLE001  复现组统计失败退回经验库比值
            pass
        ratio = None
        try:
            from .library_view import seed_variance_ratio

            r = seed_variance_ratio(getattr(self, "run_dir", None))
            if r:
                ratio = float(r["ratio"])
        except Exception:  # noqa: BLE001  库视图缺失不影响其他来源
            pass
        try:
            z_req = float(cfg.get("z_required", 2.0))
        except (TypeError, ValueError):
            z_req = 2.0
        out = seed_robust_gain(trial, sigma_seed=sigma, prior_ratio=ratio,
                               z_required=z_req, higher_better=self.direction)
        if sigma is not None:
            out["sigma_source"] = sigma_src
        out["action"] = action
        out["recipe"] = recipe_root(trial, by_id)
        return out

    def baseline_metric(self, trials: list[TrialRecord],
                        baseline_id: str | None) -> float | None:
        for t in trials:
            if t.trial_id == baseline_id:
                return (t.data.get("evaluation") or {}).get(
                    "primary_metric", {}).get("value")
        return None


# ---------------------------------------------------------------------------
# Candidate Archive(§9.4)
# ---------------------------------------------------------------------------

class CandidateArchive:
    """值得继续演进的候选集合;多目标透明评分,不做复杂 Pareto/GA。

    candidate_score = performance + stability + diversity + novelty
                     - complexity_penalty - resource_penalty
    """

    def __init__(self, run_dir: Path, spec: TaskSpec, config: Config):
        self.path = Path(run_dir) / "leaderboard" / "archive.yaml"
        self.spec = spec
        self.config = config
        self.direction = metric_direction(spec.primary_metric)
        self.state: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return load_yaml(self.path) or {"candidates": []}
        return {"updated_at": None, "candidates": []}

    def save(self) -> None:
        self.state["updated_at"] = now_iso()
        dump_yaml(self.path, self.state)

    # ---- 评分 ----
    def _score(self, cand: dict, best_metric: float | None,
               all_lineages: list[set], noise: float | None = None,
               baseline_metric: float | None = None) -> float:
        """candidate_score = performance + stability + diversity + novelty
        - complexity_penalty - resource_penalty。

        v12(P1-4):perf_score 改为以噪声为单位的相对基线分
        (metric − baseline)/noise——旧口径 (1−gap/|best|)×100 在 AUC
        场景 1 个百分点差距只扣 1.27 分,而 novelty=5/resource=10,
        AUC 低 1 个点的 explore trial 反而排在领先 exploit 前面,
        archive 系统性把注意力引向次优候选。新口径下 1×noise 的差距
        = 10 分,与 diversity(8)/novelty(5)/resource(10) 同量级。
        """
        perf = cand.get("primary_metric")
        if perf is None:
            return float("-inf")
        # performance:方向感知的噪声单位相对基线分(v12)
        perf_score = 0.0
        if noise is not None and np.isfinite(noise) and noise > 0 \
                and baseline_metric is not None \
                and np.isfinite(baseline_metric):
            gap = (perf - baseline_metric) if self.direction \
                else (baseline_metric - perf)
            # 1 noise = 10 分;下限 −20(2 noise 落后),上限 +20
            perf_score = float(np.clip(gap / noise, -2.0, 2.0) * 10.0)
        elif best_metric is not None and np.isfinite(best_metric):
            # 无噪声估计兜底:相对 best 归一(旧口径保留为退化路径)
            gap = (best_metric - perf) if self.direction else (perf - best_metric)
            span = abs(best_metric) if best_metric else 1.0
            perf_score = max(0.0, 1.0 - gap / max(span, 1e-9)) * 100
        stability = cand.get("stability") or {}
        stab_score = 10.0 if stability.get("passed", True) else -10.0
        ms_std = stability.get("multi_seed_std")
        if ms_std is not None and np.isfinite(ms_std):
            stab_score += max(0.0, 5.0 - ms_std * 10000)  # 微小 std 加分
        # diversity:同 lineage 数量越多惩罚越大
        lin = set(cand.get("parent_trial_ids") or [])
        same = sum(1 for L in all_lineages if lin and lin & L)
        div_score = 8.0 if same == 0 else max(0.0, 8.0 - 2.0 * same)
        nov_score = 5.0 if cand.get("search_action") == "explore" else \
            2.0 if cand.get("search_action") == "crossover" else 0.0
        complexity = cand.get("complexity") or {}
        cx_penalty = min(10.0, max(0.0, (complexity.get("feature_count", 0) - 200)
                                   / 100.0))
        res = cand.get("resource") or {}
        res_penalty = min(10.0, (res.get("runtime_sec", 0) or 0) / 1800.0 * 5)
        return perf_score + stab_score + div_score + nov_score - cx_penalty \
            - res_penalty

    def update(self, trials: list[TrialRecord]) -> dict:
        """Generation 结束后调用;返回 {added, removed}(§11.2)。

        收录条件:accepted/promising/confirmed 且有主指标值;无指标的
        rejected 不进 Archive(§15.2)。gen-00 baseline 经根 Trial 语义
        accepted 后自然入选。v12:confirmed(复现证据)入选(P1-6);
        Δ<noise 的 trial 不进 archive 排序(P0-2:噪声级结果不应占据
        crossover 父代与 planner 注意力)。
        """
        cap = int(self.config.get("evolution.archive_size", 12))
        by_id = {t.trial_id: t for t in trials}
        eligible: list[TrialRecord] = []
        for t in trials:
            # v2.13.2:screening-only trial 不进 Archive —— Archive 是
            # crossover 父代池与 Planner 注意力池,收录它等于让"全加宽面"
            # 这个未经校正的先验去参与组合。
            if trial_is_screening(t):
                continue
            ev = t.data.get("evaluation") or {}
            verdict = ev.get("verdict")
            if verdict in ("accepted", "promising", "confirmed") and \
                    (ev.get("primary_metric") or {}).get("value") is not None:
                eligible.append(t)
        if not eligible:
            return {"added": [], "removed": []}

        scored_pool = [t for t in eligible]
        lb = Leaderboard(self.path.parent.parent, self.spec, self.config)
        best = lb.best_trial(eligible)
        best_metric = (best.data.get("evaluation") or {}).get(
            "primary_metric", {}).get("value") if best else None
        # 噪声基准:eligible 池内配对 bootstrap se 的中位数(同代口径)
        ses = [((t.data.get("evaluation") or {}).get("delta") or {})
               .get("vs_parent_paired_bootstrap", {}).get("se")
               for t in eligible]
        ses = [s for s in ses if s is not None and np.isfinite(s) and s > 0]
        noise_ref = float(np.median(ses)) if ses else None
        baseline_metric = None
        st_path = self.path.parent.parent / "run-state.json"
        try:
            if st_path.exists():
                from .logging_utils import load_json
                bid = (load_json(st_path) or {}).get("baseline_id")
                for t in trials:
                    if t.trial_id == bid:
                        baseline_metric = (t.data.get("evaluation") or {})\
                            .get("primary_metric", {}).get("value")
        except Exception:  # noqa: BLE001
            baseline_metric = None
        existing_ids = {c["trial_id"] for c in self.state["candidates"]}
        all_lineages = [set(t.parent_ids) for t in eligible]

        scored = []
        for t in scored_pool:
            ev = t.data["evaluation"]
            cand = {
                "trial_id": t.trial_id,
                "algorithm": t.data.get("algorithm"),
                "parent_trial_ids": t.parent_ids,
                "search_action": t.data.get("search_action"),
                # v8:结构化改动模块(crossover 兼容性检查输入)
                "changed_modules": t.data.get("changed_modules") or [],
                "data_view": t.data.get("data_view") or "v1",
                "generation": t.generation,
                "primary_metric": ev["primary_metric"]["value"],
                "verdict": ev.get("verdict"),
                "stability": ev.get("stability", {}),
                "complexity": ev.get("complexity", {}),
                "resource": {"runtime_sec":
                             (t.data.get("result") or {}).get("runtime_sec")},
                "lineage": " > ".join(lineage_of(t, by_id)),
            }
            scored.append((self._score(cand, best_metric, all_lineages,
                                       noise=noise_ref,
                                       baseline_metric=baseline_metric),
                           cand))
        scored.sort(key=lambda p: p[0], reverse=True)

        # Global Best 永远保留
        keep: list[tuple[float, dict]] = []
        if best is not None:
            for s, c in scored:
                if c["trial_id"] == best.trial_id:
                    keep.append((s, c))
                    break
        for s, c in scored:
            if len(keep) >= cap:
                break
            if all(c["trial_id"] != k[1]["trial_id"] for k in keep):
                keep.append((s, c))

        new_ids = {c["trial_id"] for _, c in keep}
        added = sorted(new_ids - existing_ids)
        removed = sorted(existing_ids - new_ids)
        self.state["candidates"] = [
            {"candidate_score": round(float(s), 4), **c} for s, c in keep
        ]
        self.save()
        return {"added": added, "removed": removed}

    def top(self, k: int = 5) -> list[dict]:
        return self.state.get("candidates", [])[:k]

    def diverse_parents(self, k: int = 2) -> list[dict]:
        """为 crossover 挑互补父代:优先不同 lineage 的 top 候选。

        lineage 判定:共享任一父代 → 同源;无父 Trial(独立根,如 gen-00
        的 baseline 与独立 explore)彼此视为不同 lineage。
        """
        cands = self.state.get("candidates", [])
        if not cands:
            return []
        picked = [cands[0]]
        picked_parents = set(cands[0].get("parent_trial_ids") or [])
        picked_roots = set()  # 已选中的无父根 trial_id
        if not picked_parents:
            picked_roots.add(cands[0]["trial_id"])
        for c in cands[1:]:
            parents = set(c.get("parent_trial_ids") or [])
            if parents:
                is_same = bool(parents & picked_parents) or not parents
            else:
                # 独立根:只要不是已选中的同一个根即互补
                is_same = c["trial_id"] in picked_roots
            if not is_same:
                picked.append(c)
                picked_parents |= parents
                if not parents:
                    picked_roots.add(c["trial_id"])
            if len(picked) >= k:
                break
        return picked[:k]
