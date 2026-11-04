"""信息源清单(information-inventory,v18/P1-1):"信息已榨干"的反向对账面。

结构性缺口(两 v17.4 run 实测回写):全部特征增益来自同一个 reparse 源
(seq_d365)的不同解析粒度,而 compression_audit 只看冻结宽表组内方差
(rows_per_entity≈1 → risk_level=none、候选 0),对 reparse 源零感知;
"新信息已榨干"的反思结论是正向判定(投过的方向 >noise 即不算榨干),
**未消费的解析空间没有清单承载**——天花板结论的证据链在这里断一环。

本模块产出 `sample-profile/information-inventory.yaml`:按信息源(原始
列/解析源)逐项列出消费状态与粒度空间 checklist,让"榨干"变成可对账:

- source:原始列(spec 声明的 reparse 列)或普通特征面列族;
- consumption_status:unconsumed / partially_consumed / fully_consumed;
  - reparse 源:解析产物(带源前缀的列)是否进过任何 trial 特征面
    (按 run 目录 trials/*/config/trial.yaml 的 features 与
    feature_importance 前缀族交叉判定);
  - 普通列:是否在基础面 / 是否有二次加工族;
- granularity_space:reparse 源的粒度空间 checklist(固定维度:
  窗口×聚合×分桶×组合×序列结构),每项标 implemented/partial/unexplored
  ——planner 一眼看到"还有什么没做",反思对照着写"消费了哪项"。

消费方:planner-context 新节(reflection 与 plan 的一等输入)、
finalize 的 information_channels_exhausted 出口(unconsumed==0)。
refresh 入口按 run 目录最新 trial 状态重算消费状态(profile 建、
finalize 刷新)。
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .logging_utils import dump_yaml, get_logger, load_json, load_yaml

logger = get_logger(__name__)

#: reparse 源粒度空间的固定维度(序列类源;非序列源退化为
#: 窗口×聚合×组合 三维)。维度本身是"该源还能怎么被消费"的提示面,
#: 由 planner/Agent 对照实现情况打 implemented/partial/unexplored。
SEQUENCE_GRANULARITY_DIMENSIONS = [
    {"dimension": "window",
     "hint": "多窗计数(7/14/30/60/90/180/365)、近端 vs 远端分段"},
    {"dimension": "recency",
     "hint": "距最近/最近 K 次事件的衰减权重、连续时间衰减、指数加权"},
    {"dimension": "bucketing",
     "hint": "周桶/月桶/等频分桶的分段重聚合(周期性 vs 单窗累计)"},
    {"dimension": "aggregation",
     "hint": "计数/求和/均值/极值/方差/趋势斜率/漏斗完成率"},
    {"dimension": "sequence_structure",
     "hint": "事件类型转移对、共现、bigram/熵/相对顺序、间隔分布"},
    {"dimension": "cross_combination",
     "hint": "跨源列比值/差/交互、与基础面列的 group-by 统计编码"},
]

NON_SEQUENCE_GRANULARITY_DIMENSIONS = [
    {"dimension": "aggregation", "hint": "分组统计/极值/趋势"},
    {"dimension": "bucketing", "hint": "分箱/WOE/分位数口径"},
    {"dimension": "cross_combination", "hint": "列间比值/差/交互"},
]

# trial 特征面里"消费了 reparse 源"的判定:解析产物列前缀 = reparse
# 列名的规范化形式(如 seq_d365 → seq_*);add-hoc 解析常带源语义前缀,
# 兜底按列名包含源 stem 判定。
_PREFIX_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


def _source_prefixes(column: str) -> list[str]:
    """reparse 源列 → 可能的解析产物前缀候选。

    seq_d365 → ["seq", "seq_d365", "seqgap"] 类比的宽松集合;
    判定用"列名以下划线分段后首个 token == 规范化源 stem 或以源 stem
    开头",宽松但可解释(误判方向是漏报消费,对账面偏保守,可接受)。
    """
    stem = _PREFIX_SANITIZE_RE.sub("_", column.lower()).strip("_")
    stem0 = stem.split("_")[0]
    return [stem0, stem]


def _trial_feature_history(run_dir: Path) -> list[dict]:
    """读全部 trial 的特征面声明(快通道继承父代时取并集)。"""
    trials_dir = Path(run_dir) / "trials"
    history: list[dict] = []
    if not trials_dir.is_dir():
        return history
    for tdir in sorted(trials_dir.iterdir()):
        cfg_path = tdir / "config" / "trial.yaml"
        fi_csv = tdir / "feature" / "feature_importance.csv"
        entry: dict = {"trial_id": tdir.name, "columns": None,
                       "importance_prefixes": []}
        if cfg_path.exists():
            try:
                cfg = load_yaml(cfg_path) or {}
                feats = cfg.get("features") or {}
                cols = feats.get("columns")
                if isinstance(cols, list) and cols:
                    entry["columns"] = [str(c) for c in cols]
            except Exception:  # noqa: BLE001
                pass
        if fi_csv.exists():
            try:
                import pandas as pd

                fi = pd.read_csv(fi_csv)
                if "feature" in fi.columns and len(fi):
                    prefixes = sorted({
                        str(f).split("_")[0] for f in fi["feature"]
                        if "_" in str(f)})
                    entry["importance_prefixes"] = prefixes
            except Exception:  # noqa: BLE001
                pass
        history.append(entry)
    return history


def _split_prefixes(run_dir: Path) -> set[str]:
    """基础特征面的列前缀全集(消费判定用)。"""
    splits_dir = Path(run_dir) / "sample-profile" / "splits"
    train_p = splits_dir / "train.parquet"
    prefixes: set[str] = set()
    if train_p.exists():
        try:
            import pyarrow.parquet as pq

            schema = pq.read_schema(train_p)
            prefixes = {str(n).split("_")[0] for n in schema.names
                        if "_" in str(n)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取基础特征面 schema 失败: %s", exc)
    return prefixes


def build_inventory(run_dir: Path, spec) -> dict:
    """构建信息源清单(profile phase-2 调用;finalize refresh 复用)。"""
    run_dir = Path(run_dir)
    sources: list[dict] = []
    base_prefixes = _split_prefixes(run_dir)
    trial_history = _trial_feature_history(run_dir)

    # v2.19.0(#10):序列粒度集按**形态**选(分隔符密度/token 复用/长度 CV,
    # 见 runtime.profile_rules.sequence_kind),不再按列名关键词。形态统计
    # 拿不到(profile 产物缺失/列不在 splits)时才回落关键词兜底。
    try:
        from .profile_rules import column_shapes, is_sequence_shape

        _shapes = column_shapes(run_dir, spec)
    except Exception:  # noqa: BLE001 —— 形态统计失败走关键词兜底
        _shapes, is_sequence_shape = {}, (lambda _s: None)

    # ---- reparse 源(一等:信息合法但形态不可用,授权解析)----
    for col in sorted(set(spec.reparse_columns)):
        prefixes = _source_prefixes(col)
        consumed_prefixes = sorted(
            p for p in base_prefixes if p in prefixes)
        # importance 里出现源前缀族 → 已有解析产物进过特征面并承重
        importance_hits = sorted({
            p for entry in trial_history
            for p in (entry.get("importance_prefixes") or [])
            if p in prefixes})
        if importance_hits or consumed_prefixes:
            status = "partially_consumed"
        else:
            status = "unconsumed"
        is_sequence = is_sequence_shape(_shapes.get(col))
        sequence_basis = "shape"
        if is_sequence is None:
            # 兜底(仅形态统计缺失时):列名关键词。保留它是为了没有
            # splits/ 的旧 run 与测试夹具行为不变;有形态证据时绝不看列名。
            is_sequence = any(k in col.lower()
                              for k in ("seq", "event", "log", "stream",
                                        "序列"))
            sequence_basis = "name_keyword_fallback"
        sources.append({
            "source_type": "reparse",
            "column": col,
            "consumption_status": status,
            "evidence": {
                "base_face_prefixes": consumed_prefixes,
                "importance_prefix_families": importance_hits,
                # v2.19.0:序列判定依据(shape = 形态统计;否则关键词兜底)
                "sequence_basis": sequence_basis,
                "is_sequence": bool(is_sequence),
            },
            "granularity_space": [
                {"dimension": d["dimension"], "hint": d["hint"],
                 "state": "unexplored"}
                for d in (SEQUENCE_GRANULARITY_DIMENSIONS
                          if is_sequence
                          else NON_SEQUENCE_GRANULARITY_DIMENSIONS)
            ],
            "note": "粒度空间的 implemented/partial/unexplored 由各代"
                    "subagent 按当代新增列族回填(refresh);"
                    "全部项 implemented 或显式证伪 → fully_consumed",
        })

    # ---- 普通高潜力列族(IV strong/suspicious 但基础面直出、无二次加工)----
    try:
        fq = run_dir / "sample-profile" / "feature-quality.csv"
        if fq.exists():
            import pandas as pd

            quality = pd.read_csv(fq)
            iv_col = "iv" if "iv" in quality.columns else None
            if iv_col and "feature" in quality.columns:
                strong = quality[quality[iv_col] >= 0.3]
                for _, row in strong.head(20).iterrows():
                    fam = str(row["feature"]).split("_")[0]
                    # v2.15.8(09-19 路线图 #13 / F-J4):二次加工必须是
                    # **同源**派生族(前缀以本族开头且不等于本族,如
                    # seq → seqd/seqf)。旧判据"任一 trial 出现过任一非
                    # base 前缀"几乎恒真,强 IV 族被虚清零为
                    # fully_consumed,信息穷尽出口据此提前停止。判不出时
                    # 留在 partially_consumed —— 宁可多查一族,不早停。
                    has_secondary = any(
                        p != fam and p.startswith(fam)
                        for entry in trial_history
                        for p in (entry.get("importance_prefixes") or []))
                    sources.append({
                        "source_type": "base_face_family",
                        "column": str(row["feature"]),
                        "family": fam,
                        "iv": float(row[iv_col]),
                        "consumption_status":
                            "fully_consumed" if has_secondary
                            else "partially_consumed",
                        "note": "强 IV 族;partially = 基础面直出但无"
                                "二次加工族(比值/分桶/交互),值得对照",
                    })
    except Exception as exc:  # noqa: BLE001
        logger.warning("feature-quality 清单读取失败(跳过基础族): %s", exc)

    inventory = {
        "created_at_version": "v18",
        "run_dir": str(run_dir),
        "sources": sources,
        "summary": {
            "n_sources": len(sources),
            "unconsumed": sum(
                1 for s in sources
                if s.get("consumption_status") == "unconsumed"),
            "partially_consumed": sum(
                1 for s in sources
                if s.get("consumption_status") == "partially_consumed"),
            "fully_consumed": sum(
                1 for s in sources
                if s.get("consumption_status") == "fully_consumed"),
        },
        "note": "本清单是'信息已榨干'的反向对账面:反思必答'本代消费/"
                "证伪了哪个未消费项';information_channels_exhausted "
                "停止出口要求 unconsumed==0(清单缺失或未清零不触发)。",
    }
    return inventory


def _planner_granularity_hits(run_dir: Path) -> dict[str, set[str]]:
    """扫 planner proposal batch,把"已被规划为某个粒度候选"的
    (源 → 粒度维度集合) 收集出来。

    [FIX-2026-09-18] refresh_inventory 的注释说粒度状态"由各代 subagent
    按当代新增列族回填(refresh)",但**代码里没有任何一条路径会写
    implemented** —— 只有读取端(planner)和升格端(refresh),写入端缺失。
    实测 example-seq-v3-v2.12.2.9:凡是被规划、被执行的粒度,重跑
    refresh 后 state 仍为 unexplored、consumption_status 仍为
    unconsumed,`information_channels_exhausted` 停止出口因此永不触发。

    v2.12.2.9 协议下,planner 落盘的 proposal 里 recipe 已带
    `source` + `operator`(= granularity dimension) + `granularity_slot`
    (本修复新增,见 planner._inventory_candidates)。这里按
    family == "info:<column>:<dimension>" 解析,只认"该候选确实存在"
    这一事实本身 —— 状态语义是 **partial**(已规划/已投,不等于已榨干),
    由 Agent 在 reflection 里读完代际证据后手动升 `implemented`;
    refresh 不覆盖 Agent 写下的更高状态(见下方合并逻辑)。

    v2.17.2(D1):refresh 不再调用本函数(规划即标 partial 是自反馈,见
    :func:`_trial_granularity_hits`);保留供审计"哪些粒度被规划过"。
    """
    planner_dir = Path(run_dir) / "planner"
    hits: dict[str, set[str]] = {}
    if not planner_dir.is_dir():
        return hits
    for batch_path in sorted(planner_dir.glob("proposal-batch-*.yaml")):
        try:
            batch = load_yaml(batch_path) or {}
        except Exception:  # noqa: BLE001
            continue
        for prop in batch.get("proposals") or []:
            family = str(prop.get("family") or "")
            if not family.startswith("info:"):
                continue
            parts = family.split(":")
            if len(parts) < 3:
                continue
            column, dimension = parts[1], ":".join(parts[2:])
            hits.setdefault(column, set()).add(dimension)
    return hits


def _trial_granularity_hits(run_dir: Path) -> dict[str, set[str]]:
    """[v2.17.2 D1] 有 **trial 证据**的(源 → 粒度维度集合)。

    事故(run4 req-t7-400w):v2.12.2.10 起 refresh 把"planner 规划过"的
    粒度直接标 partial(``_planner_granularity_hits``),形成自反馈 ——
    提名 → partial → 源升 partially_consumed → 继续提名;gen-01/02/04/06
    的 ``info:<col>:derived_interaction`` 候选从未被执行,清单却显示
    已触达。现在只认 trial 证据:``trials/*/trial.yaml`` 的 family_hints
    含 ``info:<col>:<dim>``(容忍 feat: 前缀)且已有非 invalid 的评估
    (口径同 :func:`_backfill_state_for`)。规划本身不改变粒度状态。
    """
    trials_dir = Path(run_dir) / "trials"
    hits: dict[str, set[str]] = {}
    if not trials_dir.is_dir():
        return hits
    for tdir in sorted(trials_dir.iterdir()):
        p = tdir / "trial.yaml"
        if not p.exists():
            continue
        try:
            data = load_yaml(p) or {}
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(data, dict):
            continue
        state, _ = _backfill_state_for(data)
        if state is None:
            continue
        for col, dim in _info_hints(data):
            hits.setdefault(col, set()).add(dim)
    return hits


def refresh_inventory(run_dir: Path, spec) -> dict:
    """finalize 时的刷新入口:重建消费状态 + 保留 Agent 回填的
    granularity_space state(按 column+dimension 键合并)。

    状态合并优先级(高 → 低):Agent 已写入的 implemented /
    explicitly_falsified > Agent 已写入的 partial > planner 规划命中
    (partial)> unexplored。

    粘性规则:只有 **Agent 写下的** 状态是粘性的(刷新只升不降);
    planner 派生出来的 partial 每次刷新重新推导,否则一次失败的规划
    会把源永久标成"已消费",把 unconsumed 假清零。用 granularity 项上
    的 `state_source` 区分:缺省(None)= 历史文件,旧代码从不写 planner
    状态,故一律按 Agent 裁决处理(向后兼容且偏保守)。

    v2.17.2(D1):派生 partial 只认 **trial 证据**(`_trial_granularity_hits`,
    state_source=trial_evidence),不再认"planner 规划过";历史文件里
    state_source=planner 的派生态同样不粘,刷新时按 trial 证据重推(没有
    证据即回落 unexplored)。
    """
    run_dir = Path(run_dir)
    path = run_dir / "sample-profile" / "information-inventory.yaml"
    old_states: dict[tuple[str, str], tuple[str, bool]] = {}
    old_meta: dict[tuple[str, str], dict] = {}
    old_notes: dict[str, str] = {}
    _RANK = {"unexplored": 0, "partial": 1, "implemented": 2,
             "explicitly_falsified": 2}
    if path.exists():
        try:
            old = load_yaml(path) or {}
            for s in old.get("sources") or []:
                if not isinstance(s, dict):
                    continue
                for g in s.get("granularity_space") or []:
                    _st = g.get("state") or "unexplored"
                    _sticky = g.get("state_source") not in \
                        _DERIVED_STATE_SOURCES
                    old_states[(s.get("column"), g.get("dimension"))] = \
                        (_st, _sticky)
                    # v2.17.0:trial 自动回填的来源与证据随刷新保留
                    old_meta[(s.get("column"), g.get("dimension"))] = {
                        k: g.get(k) for k in
                        ("state_source", "state_evidence",
                         "state_generation", "state_verdict")
                        if g.get(k) is not None}
                old_notes[s.get("column")] = s.get("note") or ""
        except Exception:  # noqa: BLE001
            pass
    # v2.17.2(D1):只有 trial 证据才把粒度标 partial(规划不算)
    planned = _trial_granularity_hits(run_dir)
    fresh = build_inventory(run_dir, spec)
    for s in fresh.get("sources") or []:
        col = s.get("column")
        _hit = planned.get(str(col), set())
        for g in s.get("granularity_space") or []:
            prev, sticky = old_states.get(
                (col, g.get("dimension")), ("unexplored", False))
            dim_hit = g.get("dimension") in _hit
            if sticky:
                # Agent 裁决:只升不降,且不再被 planner 事实覆盖
                if _RANK.get(prev, 0) >= _RANK["partial"]:
                    g["state"] = prev
                    _meta = old_meta.get((col, g.get("dimension"))) or {}
                    if _meta.get("state_source") == BACKFILL_STATE_SOURCE:
                        g.update(_meta)
                    else:
                        g["state_source"] = "agent"
                elif dim_hit:
                    g["state"] = "partial"
                    g["state_source"] = TRIAL_EVIDENCE_STATE_SOURCE
                else:
                    g["state"] = prev or "unexplored"
            elif dim_hit:
                g["state"] = "partial"
                g["state_source"] = TRIAL_EVIDENCE_STATE_SOURCE
            else:
                g["state"] = "unexplored"
        if col in old_notes:
            s["note"] = old_notes[col]
        _promote_consumption_status(s)
    fresh["summary"] = _inventory_summary(fresh.get("sources") or [])
    fresh["refreshed_at_finalized"] = True
    dump_yaml(path, fresh)
    return fresh


def _inventory_summary(sources: list) -> dict:
    return {
        "n_sources": len(sources),
        "unconsumed": sum(
            1 for s in sources
            if s.get("consumption_status") == "unconsumed"),
        "partially_consumed": sum(
            1 for s in sources
            if s.get("consumption_status") == "partially_consumed"),
        "fully_consumed": sum(
            1 for s in sources
            if s.get("consumption_status") == "fully_consumed"),
    }


def _recompute_consumption_status(s: dict) -> None:
    """按当前粒度状态双向重算 consumption_status(v2.18.4)。

    与 :func:`_promote_consumption_status` 的区别:只升不降的那条规则在
    "假证伪被撤销"的场景下会把源永久钉在 partially_consumed。
    """
    gs = s.get("granularity_space") or []
    if not gs:
        return
    states = [g.get("state") or "unexplored" for g in gs]
    if all(st in ("implemented", "explicitly_falsified") for st in states):
        s["consumption_status"] = "fully_consumed"
    elif any(st in ("implemented", "partial", "explicitly_falsified")
             for st in states):
        s["consumption_status"] = "partially_consumed"
    else:
        s["consumption_status"] = "unconsumed"


def _promote_consumption_status(s: dict) -> None:
    """按粒度项状态升级源的 consumption_status(只升不降)。

    全部粒度项 implemented/证伪 → fully_consumed;任一项被触达(partial /
    implemented / explicitly_falsified)且仍为 unconsumed → partially_consumed。
    v2.17.0:显式证伪也算"已消费"(旧口径只认 implemented/partial,只做过
    证伪的源会一直挂 unconsumed,information_channels_exhausted 永不触发)。
    """
    gs = s.get("granularity_space") or []
    if gs and all(
            g.get("state") in ("implemented", "explicitly_falsified")
            for g in gs):
        s["consumption_status"] = "fully_consumed"
    elif gs and any(
            g.get("state") in ("implemented", "partial",
                               "explicitly_falsified")
            for g in gs) and \
            s.get("consumption_status") == "unconsumed":
        s["consumption_status"] = "partially_consumed"


# ---------------------------------------------------------------------------
# v2.17.0(#22):按 trial 的 family_hints + 裁决自动回填粒度状态
# ---------------------------------------------------------------------------
#: 自动回填写入的 state_source(refresh 视为粘性,且保留该来源标记)
BACKFILL_STATE_SOURCE = "trial_backfill"
#: v2.17.2(D1):refresh 按 trial 证据推导的 partial(非粘性,每次重推)
TRIAL_EVIDENCE_STATE_SOURCE = "trial_evidence"
#: 非粘性的派生来源:planner(v2.17.2 前的规划命中,历史文件)+ trial_evidence
_DERIVED_STATE_SOURCES = frozenset({"planner", TRIAL_EVIDENCE_STATE_SOURCE})

#: 回填内部的状态强弱(同一粒度多条证据取最强;implemented > 证伪 > partial)
_BACKFILL_RANK = {"unexplored": 0, "partial": 1,
                  "explicitly_falsified": 2, "implemented": 3}
#: Agent(人工)写入状态的强弱:implemented 与 explicitly_falsified 同为终态
_AGENT_RANK = {"unexplored": 0, "partial": 1, "implemented": 2,
               "explicitly_falsified": 2}


def _trial_view(t) -> tuple[dict, str | None, int | None]:
    data = getattr(t, "data", None)
    if data is None and isinstance(t, dict):
        data = t
    data = data or {}
    tid = data.get("trial_id") or getattr(t, "trial_id", None)
    try:
        gen = int(data.get("generation")) \
            if data.get("generation") is not None else None
    except (TypeError, ValueError):
        gen = None
    return data, tid, gen


def _info_hints(data: dict) -> list[tuple[str, str]]:
    """family_hints 里的 ``info:<column>:<dimension>``(容忍 feat: 前缀)。"""
    out = []
    hints = data.get("family_hints")
    if isinstance(hints, str):
        hints = [hints]
    for h in hints or []:
        h = str(h).strip()
        while h.startswith("feat:"):
            h = h[len("feat:"):]
        if not h.startswith("info:"):
            continue
        parts = h.split(":")
        if len(parts) < 3 or not parts[1] or not parts[2]:
            continue
        out.append((parts[1], ":".join(parts[2:])))
    return out


#: v2.18.4:裁决只是"质量门"或"没有对照"时的 reason 特征 —— 这些裁决
#: 不含方向信息,不得回填成 explicitly_falsified。
_NON_DIRECTIONAL_REASONS = (
    "稳定性或切片/时序检查未通过",
    "无父代可比",
    "无父代可比分数",
)


def _backfill_state_for(data: dict) -> tuple[str | None, str | None]:
    """trial 裁决 → (粒度目标状态, 采用的裁决);无证据返回 (None, None)。

    parent_verdict(vs_parent 单试验裁决,WP1)优先,缺失回落 verdict:
    accepted/confirmed → implemented;rejected → explicitly_falsified
    (信息源候选的证伪条件即"配对增益未过统计门");其余已评估裁决
    (promising/unconfirmed…)→ partial;invalid/未评估/失败 → 不计。
    screening-only trial 只记 partial(宽面混合,不可归因到单一粒度)。

    v2.18.4(**只认方向证据**,实测 req-t7 400w run4-v2.18.0-3):
    - 跨数据视图(data_view≠v1)的 trial 没有配对对照,其 verdict 是
      "与基线不可比"的口径,不是方向结论 —— 直接返回 None;
    - `reason` 落在质量门(稳定性/切片/时序)或"无父代可比"时同理:
      切片门拦住的是分布敏感度,不是"该粒度的信息不存在"。旧实现把
      trial-005/006/007(全部 data_view=sample、reason=无父代可比分数)
      回填成 seq_gap_d365:bucketing 与 seq_d365:aggregation 的
      **explicitly_falsified**,使两个真实未探的粒度被永久标成"已证伪",
      并污染 p-seq-channel-diminishing 一族的下游判断。
    """
    ev = data.get("evaluation") or {}
    if not isinstance(ev, dict) or not ev:
        return None, None
    pv = ev.get("parent_verdict")
    if isinstance(pv, dict):
        pv = pv.get("verdict")
    verdict = pv or ev.get("verdict")
    if not verdict or verdict == "invalid":
        return None, verdict
    if (ev.get("integrity") or {}).get("valid", True) is False:
        return None, verdict
    if str(data.get("data_view") or "v1") != "v1":
        return None, verdict
    delta = ev.get("delta") or {}
    if delta.get("baseline_incomparable"):
        return None, verdict
    reason = str(ev.get("reason") or "")
    if any(tok in reason for tok in _NON_DIRECTIONAL_REASONS):
        return None, verdict
    if data.get("screen_only"):
        return "partial", verdict
    if verdict in ("accepted", "confirmed"):
        return "implemented", verdict
    if verdict == "rejected":
        return "explicitly_falsified", verdict
    return "partial", verdict


def _collect_backfill_evidence(trials, by_col: dict,
                               generation: int | None,
                               unmatched: list | None = None) -> dict:
    """trial 裁决 → ``{(column, dimension): {state, verdict, trials,
    generation}}``(同一粒度取最强状态,证据只列该状态的 trial)。

    ``generation`` 非 None 时只看该代;``unmatched`` 非 None 时收集
    清单里找不到的 info hint。
    """
    evidence: dict[tuple[str, str], dict] = {}
    for t in trials or []:
        data, tid, gen = _trial_view(t)
        if generation is not None and gen != int(generation):
            continue
        hints = _info_hints(data)
        if not hints:
            continue
        state, verdict = _backfill_state_for(data)
        if state is None:
            continue
        for col, dim in hints:
            src = by_col.get(col)
            dims = {str(g.get("dimension")) for g in
                    (src.get("granularity_space") or [])} if src else set()
            if src is None or dim not in dims:
                if unmatched is not None:
                    unmatched.append(
                        {"trial_id": tid, "hint": f"info:{col}:{dim}",
                         "reason": "source_not_in_inventory" if src is None
                         else "dimension_not_in_granularity_space"})
                continue
            cur = evidence.get((col, dim))
            if cur is None or _BACKFILL_RANK[state] > \
                    _BACKFILL_RANK[cur["state"]]:
                evidence[(col, dim)] = {"state": state, "verdict": verdict,
                                        "trials": [tid], "generation": gen}
            elif state == cur["state"] and tid not in cur["trials"]:
                cur["trials"].append(tid)
    return evidence


def backfill_from_trials(run_dir: Path, trials, generation: int | None = None
                         ) -> dict:
    """按 trial 的 family_hints 与裁决回填信息源粒度状态(v2.17.0,#22)。

    现状:每代由 agent 人工回填 information-inventory(实测 run4
    regression.2 留下 ``logs/information-inventory.pre-g*-backfill.yaml``),
    漏填即让 unconsumed 假阳性、information_channels_exhausted 永不触发。

    规则:
    - 只认 ``info:<column>:<dimension>`` 形式的 family_hints(planner 信息源
      候选的 family;一个 trial 可带多个);``generation`` 给定时只处理该代
      trial;
    - 裁决映射见 :func:`_backfill_state_for`;同一粒度多条证据取最强;
    - **不覆盖人工写入的更高状态**:Agent 写下的 implemented/
      explicitly_falsified 为终态不动,Agent 的 partial 可被升级;planner
      推导的 partial、unexplored 可被任意升级;本函数之前回填的状态只升
      不降(证伪 → implemented 允许,反向不允许);
    - 写入项带 ``state_source=trial_backfill``、``state_evidence``(trial
      id)、``state_generation``、``state_verdict``;源的 consumption_status
      按 :func:`_promote_consumption_status` 升级;
    - 幂等:无变更不写盘;有变更时先备份到
      ``sample-profile/inventory-backups/information-inventory.pre-g<NN>-
      backfill[-<ts>].yaml`` 再写。

    返回变更摘要 ``{status, generation, changes, source_status_changes,
    skipped_agent_higher, unmatched_hints, backup, summary}``;
    status ∈ updated / unchanged / no_inventory / error。
    """
    run_dir = Path(run_dir)
    path = run_dir / "sample-profile" / "information-inventory.yaml"
    report: dict = {"status": "unchanged", "generation": generation,
                    "changes": [], "source_status_changes": [],
                    "skipped_agent_higher": [], "unmatched_hints": [],
                    "backup": None, "summary": None}
    if not path.exists():
        report["status"] = "no_inventory"
        return report
    try:
        inv = load_yaml(path) or {}
    except Exception as exc:  # noqa: BLE001
        report["status"] = "error"
        report["error"] = f"读取 information-inventory.yaml 失败: {exc}"
        return report
    sources = [s for s in (inv.get("sources") or []) if isinstance(s, dict)]
    by_col = {str(s.get("column")): s for s in sources}

    # 1) 汇总证据:(column, dimension) → 最强状态 + 证据 trial
    evidence = _collect_backfill_evidence(
        trials, by_col, generation, report["unmatched_hints"])

    all_trials_by_id: dict[str, dict] = {}
    for t in trials or []:
        data, tid, _gen = _trial_view(t)
        if tid:
            all_trials_by_id[str(tid)] = data

    # 1b) v2.18.4:复核既有回填 —— 旧版按 verdict 直接判证伪,把跨视图臂与
    # 被切片门拦下的臂都写成 explicitly_falsified。回填"只升不降"意味着
    # 判据修好后这些假证伪会永久留存,所以这里按**当前**判据重算每条
    # 回填证据;重算后不再成立的条目降级到剩余证据的最强状态(无证据则
    # 回 unexplored)。只动 state_source=trial_backfill 的条目,人工裁决
    # 与 planner 派生状态不受影响。
    demoted: list[dict] = []
    full_evidence: dict | None = None   # 惰性:只有撤证据时才全量重算
    if all_trials_by_id:
        for src in sources:
            col = str(src.get("column"))
            for g in src.get("granularity_space") or []:
                if g.get("state_source") != BACKFILL_STATE_SOURCE:
                    continue
                ev_tids = [str(x) for x in (g.get("state_evidence") or [])]
                kept: list[tuple[str, str]] = []   # (tid, state)
                dropped = False
                for tid in ev_tids:
                    tv = all_trials_by_id.get(tid)
                    if tv is None:
                        kept.append((tid, g.get("state")))  # 证据不在本次列表里:保留
                        continue
                    st, _v = _backfill_state_for(tv)
                    if st is None:
                        dropped = True
                    else:
                        kept.append((tid, st))
                if not dropped and all(st == g.get("state")
                                       for _t, st in kept):
                    continue
                # state_evidence 只记录"最强状态"的 trial:撤掉假证伪后,
                # 被它覆盖过的较弱证据(往代的 partial)不在列表里。生产按
                # 代调用(generation=gid),第 2 步也只扫本代 —— 所以这里
                # 必须对**全量** trial 重算该粒度,否则往代证据永久丢失。
                if full_evidence is None:
                    full_evidence = _collect_backfill_evidence(
                        trials, by_col, None)
                fe = full_evidence.get((col, str(g.get("dimension"))))
                if fe:
                    kept += [(str(t), fe["state"]) for t in fe["trials"]]
                dim = g.get("dimension")
                if not kept:
                    demoted.append({"column": col, "dimension": dim,
                                    "from": g.get("state"), "to": "unexplored",
                                    "trials": ev_tids})
                    g["state"] = "unexplored"
                    for k in ("state_source", "state_evidence",
                              "state_generation", "state_verdict"):
                        g.pop(k, None)
                    continue
                best = max((st for _t, st in kept),
                           key=lambda x: _BACKFILL_RANK.get(x, 0))
                best_tids = sorted({t for t, st in kept if st == best})
                if best == g.get("state") and best_tids == sorted(ev_tids):
                    continue
                demoted.append({"column": col, "dimension": dim,
                                "from": g.get("state"), "to": best,
                                "trials": ev_tids, "kept": best_tids})
                g["state"] = best
                g["state_evidence"] = best_tids
                if fe and fe["state"] == best:
                    if fe.get("generation") is not None:
                        g["state_generation"] = fe["generation"]
                    if fe.get("verdict"):
                        g["state_verdict"] = fe["verdict"]

    # 2) 合并进清单(不覆盖人工更高状态)
    for (col, dim), evd in sorted(evidence.items()):
        src = by_col[col]
        g = next(g for g in src.get("granularity_space") or []
                 if str(g.get("dimension")) == dim)
        prev = g.get("state") or "unexplored"
        prev_src = g.get("state_source")
        new = evd["state"]
        if prev_src == BACKFILL_STATE_SOURCE:
            # 本函数此前的回填:只升不降;同状态只补证据
            if _BACKFILL_RANK.get(new, 0) < _BACKFILL_RANK.get(prev, 0):
                continue
            if new == prev:
                old_ev = set(g.get("state_evidence") or [])
                if set(evd["trials"]) <= old_ev:
                    continue  # 幂等:证据已在
                g["state_evidence"] = sorted(old_ev | set(evd["trials"]))
                report["changes"].append(
                    {"column": col, "dimension": dim, "from": prev,
                     "to": new, "trials": evd["trials"],
                     "verdict": evd["verdict"], "evidence_only": True})
                continue
        elif prev_src in _DERIVED_STATE_SOURCES or prev == "unexplored":
            pass  # 派生态/未探:任意回填均可(trial 证据更硬)
        else:
            # Agent/人工写入(含历史文件缺省来源):只在严格更高时升级
            if _AGENT_RANK.get(new, 0) <= _AGENT_RANK.get(prev, 0):
                if new != prev:
                    report["skipped_agent_higher"].append(
                        {"column": col, "dimension": dim, "state": prev,
                         "backfill_state": new, "trials": evd["trials"]})
                continue
        g["state"] = new
        g["state_source"] = BACKFILL_STATE_SOURCE
        g["state_evidence"] = sorted(set(evd["trials"]))
        if evd.get("generation") is not None:
            g["state_generation"] = evd["generation"]
        if evd.get("verdict"):
            g["state_verdict"] = evd["verdict"]
        report["changes"].append(
            {"column": col, "dimension": dim, "from": prev, "to": new,
             "trials": evd["trials"], "verdict": evd["verdict"]})

    if demoted:
        report["demotions"] = demoted
        report["status"] = "updated"

    # 3) 源级 consumption_status
    if report["changes"] or demoted:
        touched = {c["column"] for c in report["changes"]} | \
            {c["column"] for c in demoted}
        for col in sorted(touched):
            src = by_col[col]
            before = src.get("consumption_status")
            if any(c["column"] == col for c in demoted):
                # v2.18.4:降级必须双向重算 —— _promote_consumption_status
                # 只升不降,假证伪被撤掉后源会永久停在 partially_consumed。
                _recompute_consumption_status(src)
            else:
                _promote_consumption_status(src)
            if src.get("consumption_status") != before:
                report["source_status_changes"].append(
                    {"column": col, "from": before,
                     "to": src.get("consumption_status")})
    if not report["changes"] and not demoted:
        report["summary"] = inv.get("summary")
        return report

    # 4) 备份 + 写回
    try:
        import shutil
        import time as _time

        bdir = run_dir / "sample-profile" / "inventory-backups"
        bdir.mkdir(parents=True, exist_ok=True)
        tag = f"g{int(generation):02d}" if generation is not None else "all"
        bpath = bdir / f"information-inventory.pre-{tag}-backfill.yaml"
        if bpath.exists():
            bpath = bdir / (f"information-inventory.pre-{tag}-backfill-"
                            f"{_time.strftime('%Y%m%d%H%M%S')}.yaml")
        shutil.copy2(path, bpath)
        report["backup"] = str(bpath)
    except Exception as exc:  # noqa: BLE001 — 备份失败则不写(宁可不回填)
        report["status"] = "error"
        report["error"] = f"备份失败,未写回: {exc}"
        return report
    inv["sources"] = sources
    inv["summary"] = _inventory_summary(sources)
    inv.setdefault("backfill_log", []).append({
        "generation": generation,
        "n_changes": len(report["changes"]),
        "n_demotions": len(demoted),
        "backup": report["backup"]})
    dump_yaml(path, inv)
    report["status"] = "updated"
    report["summary"] = inv["summary"]
    logger.info("information-inventory 自动回填:%d 项粒度状态变更"
                "%s(备份 %s)",
                len(report["changes"]),
                f"、{len(demoted)} 项假证伪复核撤销" if demoted else "",
                report["backup"])
    return report


def render_inventory_section(inventory: dict | None) -> str:
    """planner-context 第 12 节渲染(未消费项 = planner 一等输入)。"""
    if not inventory or not (inventory.get("sources") or []):
        return "(未就绪:information-inventory.yaml 缺失,profile 后产出)"
    lines = ["| 源 | 类型 | 状态 | 未探粒度 |", "|---|---|---|---|"]
    for s in inventory.get("sources") or []:
        gs = s.get("granularity_space") or []
        n_unexplored = sum(1 for g in gs if g.get("state") == "unexplored")
        lines.append(
            f"| {s.get('column')} | {s.get('source_type')} | "
            f"{s.get('consumption_status')} | "
            f"{n_unexplored}/{len(gs) if gs else '—'} |")
    lines.append("")
    lines.append(f"汇总:{inventory.get('summary')}")
    return "\n".join(lines)
