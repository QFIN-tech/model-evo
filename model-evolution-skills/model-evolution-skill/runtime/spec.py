"""TaskSpec contract(§6)。

TaskSpec 阶段只做简单探查,要求快速形成必备字段并与用户确认;
更复杂的分析在 SampleProfile 实现。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logging_utils import dump_yaml, load_yaml
from .topology import policy_summary as _topology_policy

MODEL_TYPES = ("binary_classification", "regression", "multiclass", "ordinal", "uplift")
SPLIT_TYPES = ("time_based", "random", "group_based")

# split_policy 各 split 名(用于窗口语法校验;random/group_based 用 ratios)
SPLIT_NAMES = ("train", "validation", "shadow_test", "final_test")


def window_token_error(token: str) -> str | None:
    """校验单个时间窗端点的语法;合法返回 None,否则返回人话原因。

    v19.2:被注解污染的窗口(如 `2026-01~2026-04(20260111~20260426,
    16 期,2,461,339 行)`)此前会静默降级为前缀匹配,只匹配到某几个
    日期前缀,其余行落进 train 兜底——split 构成被改写而无人知晓
    (实测 train 少 4 个快照日、validation 少 4 个、shadow/final_test
    整体平移一格,且 audit 报的实体重叠率把口径错误伪装成泄漏)。
    这份校验把该错误提前到 `validate`/`profile` 之前,并且把"哪个
    split、哪个端点、怎么改"直接说清楚。
    """
    text = token.strip()
    if not text:
        return "端点为空"
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y-%m", "%Y%m"):
        try:
            import datetime as _dt
            _dt.datetime.strptime(text, fmt)
            return None
        except (ValueError, TypeError):
            continue
    # 给最常见的"顺手加注解"形态一句明确的诊断(其余按语法错报)
    hint = ""
    if "(" in text or ")" in text or "行" in text or "期" in text:
        hint = ("——端点里出现了注解/统计文字。split 字段是**纯区间**,"
                "注解请放到该行的 YAML 注释里(# 之后)")
    return (f"{text!r} 不是合法的时间窗端点(支持 YYYY-MM、YYYY-MM-DD、"
            f"YYYYMM、YYYYMMDD){hint}")


def validate_split_policy_windows(split: dict) -> list[str]:
    """time_based split_policy 的窗口语法校验(返回错误清单)。"""
    errors: list[str] = []
    if (split or {}).get("type") != "time_based":
        return errors
    for name in SPLIT_NAMES:
        window = split.get(name)
        if not window or not isinstance(window, str):
            continue
        parts = window.split("~")
        if len(parts) > 2:
            errors.append(
                f"split_policy.{name} 含多个 '~':{window!r}(区间只允许 A~B)")
            continue
        for tok in parts:
            err = window_token_error(tok)
            if err:
                errors.append(f"split_policy.{name} 端点非法:{err}")
    return errors

# ---------------------------------------------------------------------------
# 泄漏分类(v13「毒性轴」):exclude 拆为 banned / reparse 两类
# ---------------------------------------------------------------------------
# banned = 信息有毒:窗口越过决策时点 / 与标签等价 / 同族代理 / 实体标识。
#   排除语义是"信息不得以任何形态进入任何数据产物"(物理隔离,materialize
#   时直接不落盘),不是"列名不进特征面"。req-t7-drawdown 实测:flat exclude
#   只挡列名,前缀聚合(agg_loan_nonzero_cnt 吞 loan_cnt_t7/t15/t30/t0)
#   把信息偷渡回来,以 5.85% gain 成为头号特征。
BANNED_REASONS = (
    "future_window",       # 窗口越过决策时点(机械可判:同 stem、窗口>performance_window)
    "label_equivalent",    # 与标签同源/等价编码(单变量 AUC>0.85 哨兵)
    "label_proxy_family",  # 与标签同族业务口径(如 loan_cnt vs req_cnt),需人裁
    "entity_identifier",   # 实体身份(唯一率≈1 且与实体数对齐)
    "user_confirmed",      # 用户显式判定(兜底)
)
# reparse = 形态不可用但信息合法(高基数文本/复合编码/object):
#   原始列不进通用特征面,但授权解析类 transform 消费,产物以前缀族入面。
REPARSE_REASONS = (
    "form_unusable",       # 形态问题(必须附 plan,声明解析去向)
)

# 主指标 × model_type 合法组合(方向错误时 validate 直接拦截)
CLASSIFICATION_METRICS = ("auc", "ks", "gini", "logloss", "nll", "auprc",
                          "normalized_gini", "accuracy", "f1", "precision",
                          "recall", "uplift_auc", "qini", "auuc",
                          "lift_top10", "lift_top30")
# 头部圈人 lift 只能做 secondary(单点口径,信息量不足以做主指标)
TOP_LIFT_METRIC_NAMES = ("lift_top10", "lift_top30")
REGRESSION_METRIC_NAMES = ("rmse", "mae", "mape", "r2")
# uplift 专属指标(score=CATE 估计 + treatment 归属才能计算);
# uplift 任务的 primary/secondary 指标只能从这里选
UPLIFT_METRIC_NAMES = ("uplift_auc", "qini", "auuc")
#: runtime.metrics.compute_metric 实现的全部指标名(v12/P2:spec.validate
#: 拒绝未实现的 primary_metric——旧行为允许声明任意指标,run 阶段才
#: ValueError,浪费整轮规划)
IMPLEMENTED_METRICS = CLASSIFICATION_METRICS + REGRESSION_METRIC_NAMES

# 这些字段必须进入 Memory metadata(§6.2)
MEMORY_META_FIELDS = (
    "model_type",
    "business_domains",
    "objective_description",
    "primary_metric",
)

SPEC_REQUIRED_TOP = ("task_id", "task_name", "model_type", "target", "primary_metric")


@dataclass
class TaskSpec:
    """一个建模任务的标准规格(§6.1)。"""

    data: dict[str, Any] = field(default_factory=dict)

    # ---- 便捷访问 ----
    @property
    def task_id(self) -> str:
        return self.data["task_id"]

    def derive_abstract_task_id(self, run_dir: "Path | None" = None) -> str:
        """推导语义级抽象任务 ID（只读，不持久化）。

        v2.15.8 起委托给 :meth:`resolve_abstract_task_id`;旧调用方保留
        本名字。需要"首次可信即落 run-state"的入口(profile 写回、
        baseline、planner-context、导出、回填)直接调 resolve。
        """
        return self.resolve_abstract_task_id(run_dir, persist=False)

    #: 可信的 atid 来源(可持久化进 run-state);splits_fallback 不在其中
    ATID_TRUSTED_SOURCES = ("raw_columns_hash", "raw_shard")

    def resolve_abstract_task_id(self, run_dir: "Path | None" = None, *,
                                 persist: bool = True) -> str:
        """单源解析抽象任务 ID，格式：{model_type}-{target_name}-{sample_scale}-{col_hash8}

        这个 ID 跨 run 稳定：同一份数据底座+同一个标签列的所有 run 得到
        相同的 abstract_task_id，不论建模路线（宽表/深度序列/集成）。
        col_hash8 是**原生全量面**列名排序 hash;sample_scale 是行数量级桶。

        v2.15.8(refactor 系列 run 实测事故):同一份数据曾算出 bc5329dc
        (原生 1513 列)与 7580b665(splits 1493 列)两个指纹。根因是本
        函数不对 run_dir 做 resolve —— 在 run 目录内以 ``--spec
        task-spec.yaml`` 调用时 run_dir=Path("."),``.parent`` 仍是
        ``.``,向上找不到 ``../data`` 原始分片,静默退回 splits hash;
        而 top-trials 只按 bc5329dc 建目录,于是热启动 / Layer 0 /
        检索 +10 全部静默失效。解析顺序:

        1. run-state 已持久化且来源可信的 ``abstract_task_id``;
        2. ``sample-profile.json["raw_columns_hash"]``(profile 落盘的
           原生全量面 hash,与原始分片同口径);
        3. 对 **resolve 后**的 run_dir 向上两级找原始分片(旧
           ``train_400w`` glob,刻意不泛化:宽松 glob 会误哈希无关数据);
        4. ``splits/train.parquet``:打 warning、来源记 splits_fallback、
           **不持久化**(建模面会随 banned 演进漂移,后续调用仍可升级)。

        persist=True 且来源可信、run-state 存在时，首次解析结果写回
        run-state(``abstract_task_id`` / ``abstract_task_id_source``)。
        run_dir 为空或拿不到任何列名时 col_hash8 用 "pending"。
        """
        import hashlib
        from pathlib import Path as _Path

        mt_map = {"binary_classification": "binary",
                  "regression": "reg",
                  "uplift": "uplift"}
        mt = mt_map.get(str(self.data.get("model_type") or ""), "cls")
        tn = str((self.data.get("target") or {}).get("name") or "unknown")
        # sanitize: only alnum/hyphen/underscore
        tn_safe = "".join(c if (c.isalnum() or c in "-_") else "-" for c in tn)

        col_hash = "pending"
        source = "pending"
        sample_scale = "?w"
        _run = _Path(run_dir).resolve() if run_dir is not None else None
        if _run is not None:
            # 1) run-state 已持久化的可信值
            try:
                _st_path = _run / "run-state.json"
                if _st_path.exists():
                    _st = json.loads(_st_path.read_text(encoding="utf-8")) or {}
                    if _st.get("abstract_task_id") and _st.get(
                            "abstract_task_id_source") in self.ATID_TRUSTED_SOURCES:
                        return str(_st["abstract_task_id"])
            except Exception:  # noqa: BLE001 — 坏 run-state 退回现场推导
                pass

            def _col_hash_from_parquet(pq_path: "_Path") -> "str | None":
                try:
                    import pyarrow.parquet as _pq
                    schema = _pq.read_schema(str(pq_path))
                    cols = sorted(str(c) for c in schema.names)
                    if cols:
                        return hashlib.sha256(
                            "|".join(cols).encode("utf-8")
                        ).hexdigest()[:8]
                except Exception:
                    return None

            prof: dict = {}
            for _pname in ("sample-profile.json", "profile.json"):
                ppath = _run / "sample-profile" / _pname
                if ppath.exists():
                    try:
                        prof = json.loads(ppath.read_text(encoding="utf-8")) or {}
                    except Exception:  # noqa: BLE001
                        prof = {}
                    break
            # 2) profile 落盘的原生全量面 hash
            if prof.get("raw_columns_hash"):
                col_hash, source = str(prof["raw_columns_hash"]), "raw_columns_hash"
            # 3) 原始分片(resolve 后的 run_dir 向上两级)
            if col_hash == "pending":
                for _ancestor in [_run, _run.parent, _run.parent.parent]:
                    _data = _ancestor / "data"
                    if _data.is_dir():
                        _shards = sorted(_data.glob("train_400w-part-*.parquet"))
                        if _shards:
                            _h = _col_hash_from_parquet(_shards[0])
                            if _h:
                                col_hash, source = _h, "raw_shard"
                                break
            # 4) splits 兜底(不持久化)
            if col_hash == "pending":
                _splits_pq = _run / "sample-profile" / "splits" / "train.parquet"
                if _splits_pq.exists():
                    _h = _col_hash_from_parquet(_splits_pq)
                    if _h:
                        col_hash, source = _h, "splits_fallback"
                        import logging as _logging
                        _logging.getLogger(__name__).warning(
                            "abstract_task_id 退回 splits 列 hash(%s):原生面 hash "
                            "不可得,指纹可能与同数据其它 run 不一致;不持久化",
                            _run)

            # sample_scale from sample-profile.json (n_rows)
            try:
                rows = int(prof.get("n_rows") or prof.get("total_rows") or 0)
                if rows > 0:
                    if rows < 100_000:
                        sample_scale = "lt10w"
                    elif rows < 500_000:
                        sample_scale = "10w_50w"
                    elif rows < 2_000_000:
                        sample_scale = "50w_200w"
                    elif rows < 10_000_000:
                        sample_scale = "200w_1000w"
                    else:
                        sample_scale = "ge1000w"
            except Exception:
                pass
            # best-effort from task_name when profile rows unavailable
            if sample_scale == "?w":
                name = str(self.data.get("task_name") or "")
                if "400w" in name or "400万" in name:
                    sample_scale = "200w_1000w"

        atid = f"{mt}-{tn_safe}-{sample_scale}-{col_hash}"
        if persist and _run is not None and source in self.ATID_TRUSTED_SOURCES \
                and sample_scale != "?w":
            try:
                from .logging_utils import dump_json, locked_state_write
                _st_path = _run / "run-state.json"
                if _st_path.exists():
                    with locked_state_write(_st_path):
                        _st = json.loads(_st_path.read_text(encoding="utf-8")) or {}
                        if _st.get("abstract_task_id") != atid:
                            _st["abstract_task_id"] = atid
                            _st["abstract_task_id_source"] = source
                            dump_json(_st_path, _st)
            except Exception:  # noqa: BLE001 — 持久化失败不影响返回值
                pass
        return atid

    @property
    def model_type(self) -> str:
        return self.data["model_type"]

    @property
    def primary_metric(self) -> str:
        return self.data["primary_metric"]

    @property
    def target_name(self) -> str:
        return self.data["target"]["name"]

    @property
    def time_column(self) -> str | None:
        return self.data.get("time", {}).get("column")

    @property
    def id_column(self) -> str | None:
        return self.data.get("entity", {}).get("id_column")

    @property
    def treatment_column(self) -> str | None:
        """uplift 任务的处置列(1=处置 0=对照);非 uplift 任务为 None。"""
        return (self.data.get("treatment") or {}).get("column")

    @property
    def resource_budget(self) -> dict:
        return self.data.get("resource_budget", {})

    @property
    def feature_columns(self) -> list[str] | None:
        """显式指定的特征列白名单;None 表示除排除列外全部。"""
        return self.data.get("features", {}).get("columns")

    @property
    def excluded_columns(self) -> list[str]:
        """全部排除列(banned + reparse;向后兼容的扁平视图)。

        v13 起 specs 应使用 banned_columns / reparse_columns;
        本 property 保留给旧 flat ``features.exclude`` 形态
        (无 reason 的裸列名列表,默认全部按 banned 处理——
        fail-safe 方向:误把泄漏列当 reparse 是灾难,误把
        待解析列当 banned 只损失探索面)。
        """
        feats = self.data.get("features", {}) or {}
        if feats.get("banned") is not None or feats.get("reparse") is not None:
            return list(self.banned_columns) + list(self.reparse_columns)
        return feats.get("exclude", [])

    @property
    def banned_columns(self) -> list[str]:
        """信息有毒列:任何形态不得进入任何数据产物(物理隔离)。

        v22.1:已复核撤销的机械封禁(``reinstated_banned``)仍然**保留在
        本列表中** —— 撤销只是让该列回到待裁决状态、重新出现在复核清单里,
        物理隔离本身要等用户写下真正的裁决(banned 改判为 reparse,或
        删除 banned 条目)才解除。这样"撤销"这个动作永远不会悄悄放行
        一个未重新裁决的列,数据面仍由显式 banned/reparse 决定。
        """
        feats = self.data.get("features", {}) or {}
        banned = feats.get("banned")
        if banned is None:
            # 旧 flat 形态:无 reason 的裸列表 → 全部视为 banned
            return list(feats.get("exclude") or [])
        out: list[str] = []
        for item in banned:
            out.append(item["column"] if isinstance(item, dict) else str(item))
        return out

    @property
    def reparse_columns(self) -> list[str]:
        """形态不可用但信息合法的列:授权解析,原始列不进通用面。"""
        feats = self.data.get("features", {}) or {}
        out: list[str] = []
        for item in feats.get("reparse") or []:
            out.append(item["column"] if isinstance(item, dict) else str(item))
        return out

    @property
    def reparse_plans(self) -> dict[str, str]:
        """{reparse 列: plan 文本};plan 是解析去向的显式承诺。"""
        feats = self.data.get("features", {}) or {}
        out: dict[str, str] = {}
        for item in feats.get("reparse") or []:
            if isinstance(item, dict):
                out[item["column"]] = str(item.get("plan") or "")
        return out

    @property
    def kept_ambiguous_columns(self) -> list[str]:
        """用户显式确认**保留为普通特征**的 ambiguous 列(v2.12.2.9)。

        背景:profile 的 phase-1 确认门只要存在 ambiguous 提案就置
        ``needs_user_confirmation=true``,而 spec 里原本**没有任何地方**
        能声明"这列我确认保留"——正确裁决(保留)反而关不上门,逼迫用户
        重跑 profile 或错误地把合法列 ban 掉。

        fail-safe 方向:本字段只用来**关闭确认门**,不改变任何数据面。
        列名写错(不在 audit 面内)由 validate 报错,防止手滑写成无效声明。
        """
        feats = self.data.get("features", {}) or {}
        conf = feats.get("leak_confirmation") or {}
        out: list[str] = []
        for item in conf.get("kept_ambiguous") or []:
            out.append(item["column"] if isinstance(item, dict)
                       else str(item))
        return out

    @property
    def leak_confirmation_note(self) -> str:
        """确认门的裁决说明(留痕用,进 leak-classification.md)。"""
        feats = self.data.get("features", {}) or {}
        conf = feats.get("leak_confirmation") or {}
        return str(conf.get("note") or "")

    @property
    def reinstated_banned_columns(self) -> list[str]:
        """复核后**撤销机械封禁**的列(v22.1)。

        背景(实测事故,2026-09-22,req-t7-propensity run):
            ``_leak_classify`` 规则 3 会把「非数值 + 唯一率≈1」直接判
            ``entity_identifier`` 并封禁。但逐人唯一的**复合文本字段**
            (行为事件序列/JSON/编码串)唯一率同样≈1:本 run 的
            ``seq_d365``/``seq_gap_d365`` 唯一率 0.9976 被误封,直接丢弃
            该数据面最强信息通道(全局经验 exp-202609-8a4fe068 标注
            seq reparse 加宽 ~+0.0035 AUC)。同数据源的三个历史 run 均
            正确判为 reparse,只有本 run 误判。

            当时唯一的挽回办法是**改 spec 重跑 profile**;spec 里没有
            「撤销封禁」这一档,一旦机械规则写窄,正确裁决就无处安放——
            与 kept_ambiguous 之于 ambiguous 完全同构的缺口。

        语义:仅从 banned/banned-certain 提案面**剔除该列**,让它回到
        待裁决状态(仍需写 banned / reparse / kept_ambiguous 之一才关门)。
        **不改变数据面**——数据面永远由 banned/reparse 决定,本字段只
        表达"这个封禁是机械误判,撤回"。

        与 kept_ambiguous 的区别:后者说"这列合法,当普通特征用";
        本字段说"这个封禁判错了,重新裁决"。
        """
        feats = self.data.get("features", {}) or {}
        conf = feats.get("leak_confirmation") or {}
        out: list[str] = []
        for item in conf.get("reinstated_banned") or []:
            out.append(item["column"] if isinstance(item, dict)
                       else str(item))
        return out

    @property
    def reinstated_banned_reasons(self) -> dict[str, str]:
        """{撤销列: 复核理由};写进 leak-classification.md 留痕。"""
        feats = self.data.get("features", {}) or {}
        conf = feats.get("leak_confirmation") or {}
        out: dict[str, str] = {}
        for item in conf.get("reinstated_banned") or []:
            if isinstance(item, dict):
                out[str(item["column"])] = str(item.get("reason") or "")
        return out

    @property
    def leak_manifest_hash(self) -> str:
        """泄漏分类指纹(v13):banned(+reason)与 reparse(+plan)的
        排序序列化 hash。BASELINED 时与 harness_hash 并列落 run-state;
        run/材料化/finalize 全部校验,不一致 = 分类已变更 = 新协议,
        重新 baseline(收紧也一样——数据面变了,历史不可比)。

        v2.12.2.9:kept_ambiguous 也进指纹——它是确认门的裁决产物,
        改它等于改"哪些列被视为已裁决",必须同样触发协议校验。
        v22.1:reinstated_banned 同理进指纹(撤销封禁 = 分类面变更)。
        """
        feats = self.data.get("features", {}) or {}
        banned = sorted(
            json.dumps({"c": (i["column"] if isinstance(i, dict) else i),
                        "r": (i.get("reason") if isinstance(i, dict)
                              else "user_confirmed")},
                       ensure_ascii=False, sort_keys=True)
            for i in (feats.get("banned") or feats.get("exclude") or []))
        reparse = sorted(
            json.dumps({"c": (i["column"] if isinstance(i, dict) else i),
                        "p": (i.get("plan") if isinstance(i, dict) else "")},
                       ensure_ascii=False, sort_keys=True)
            for i in (feats.get("reparse") or []))
        kept = sorted(self.kept_ambiguous_columns)
        reinstated = sorted(self.reinstated_banned_columns)
        return hashlib.sha256(
            ("\n".join(banned) + "\n--\n" + "\n".join(reparse)
             + "\n--\n" + "\n".join(kept)
             + "\n--\n" + "\n".join(reinstated))
            .encode("utf-8")).hexdigest()[:16]

    @property
    def data_face_hash(self) -> str:
        """只覆盖**数据面**的泄漏指纹:banned(+reason)与 reparse(+plan)。

        [FIX-2026-10-29] 与 :attr:`leak_manifest_hash` 的区别只在
        ``leak_confirmation``(kept_ambiguous / reinstated_banned)——按协议
        那两档"只关闭确认门,不改数据面"。实测(run4 req-t7 run1):用户裁决
        10 个 ambiguous 列保留后重跑 profile,确认门该关的都关了,但 20 GB
        四平面被原样重写一遍(train/validation/shadow/final + labels +
        reparse,约 7 min)。profile 用本指纹判断能否跳过 ``write_splits``:
        相等即数据面未变,产物可直接复用。``leak_manifest_hash`` 本身**不改**
        语义(它仍含 kept/reinstated,用于协议校验与审计)。
        """
        feats = self.data.get("features", {}) or {}
        banned = sorted(
            json.dumps({"c": (i["column"] if isinstance(i, dict) else i),
                        "r": (i.get("reason") if isinstance(i, dict)
                              else "user_confirmed")},
                       ensure_ascii=False, sort_keys=True)
            for i in (feats.get("banned") or feats.get("exclude") or []))
        reparse = sorted(
            json.dumps({"c": (i["column"] if isinstance(i, dict) else i),
                        "p": (i.get("plan") if isinstance(i, dict) else "")},
                       ensure_ascii=False, sort_keys=True)
            for i in (feats.get("reparse") or []))
        return hashlib.sha256(
            ("\n".join(banned) + "\n--\n" + "\n".join(reparse))
            .encode("utf-8")).hexdigest()[:16]

    # ---- 构造与校验 ----
    @classmethod
    def load(cls, path: Path) -> "TaskSpec":
        raw = load_yaml(Path(path))
        if not isinstance(raw, dict):
            raise ValueError(f"TaskSpec 必须是映射: {path}")
        spec = cls(data=raw)
        errors = spec.validate()
        if errors:
            raise ValueError(
                "TaskSpec 校验失败:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        return spec

    def validate(self) -> list[str]:
        errors: list[str] = []
        d = self.data

        for key in SPEC_REQUIRED_TOP:
            if not d.get(key):
                errors.append(f"缺少必备字段: {key}")

        if d.get("model_type") and d["model_type"] not in MODEL_TYPES:
            errors.append(f"model_type 必须是 {MODEL_TYPES} 之一,当前: {d['model_type']}")

        # v2.18.2:business_domains 参与经验库的任务相似度,按受控词表归一
        # (runtime/library/business_domains.yaml);这里只校验类型,不认识的写法不拦截
        _bd = d.get("business_domains")
        if _bd is not None and not (
                isinstance(_bd, list) and all(isinstance(x, str) for x in _bd)):
            errors.append("business_domains 必须是字符串列表(取值见 "
                          "runtime/library/business_domains.yaml,代码或别名均可)")

        # v2.15.10:data.derived_columns = loader 在原始源上派生的列(如标签
        # 变换),不进数据底座指纹;必须是字符串列表
        _dc = (d.get("data") or {}).get("derived_columns")
        if _dc is not None and not (
                isinstance(_dc, list) and all(isinstance(c, str) for c in _dc)):
            errors.append("data.derived_columns 必须是列名字符串列表")

        # primary_metric 与 model_type 方向一致性(回归/分类指标错配拦截)
        pm = str(d.get("primary_metric") or "").lower()
        mt = d.get("model_type")
        if pm and pm not in IMPLEMENTED_METRICS:
            errors.append(
                f"primary_metric={pm} 未实现(runtime.metrics."
                "compute_metric 支持的指标见 IMPLEMENTED_METRICS;"
                "声明未实现指标会在 run 阶段才 ValueError,此处提前拦截)")
        if pm and mt:
            if mt == "regression" and pm in CLASSIFICATION_METRICS:
                errors.append(
                    f"regression 任务不支持分类指标 primary_metric={pm};"
                    f"可选: {REGRESSION_METRIC_NAMES}")
            elif mt in ("binary_classification", "uplift") \
                    and pm in REGRESSION_METRIC_NAMES:
                errors.append(
                    f"{mt} 任务不支持回归指标 primary_metric={pm};"
                    "分类任务用 auc/ks/logloss 等")
            elif mt == "uplift" and pm not in UPLIFT_METRIC_NAMES:
                errors.append(
                    f"uplift 任务 primary_metric 必须是 {UPLIFT_METRIC_NAMES}"
                    f"之一(score 列为 CATE 估计,普通分类指标无意义);"
                    f"当前: {pm}")
            elif pm in TOP_LIFT_METRIC_NAMES:
                errors.append(
                    f"primary_metric 不能用 {pm}(头部圈人单点口径,"
                    "信息量不足以做主指标;可放 secondary_metrics)")
        # uplift 的 secondary_metrics 同样只能用 uplift 指标
        # (普通 auc/ks 评的是转化率而非 uplift,会误导验收)
        if mt == "uplift":
            for sm in d.get("secondary_metrics") or []:
                if str(sm).lower() not in UPLIFT_METRIC_NAMES:
                    errors.append(
                        f"uplift 任务 secondary_metrics 只能用 "
                        f"{UPLIFT_METRIC_NAMES}(当前含 {sm})")

        target = d.get("target") or {}
        if not target.get("name"):
            errors.append("target.name 缺失")
        if d.get("model_type") == "binary_classification" and not target.get(
            "positive_definition"
        ):
            errors.append("binary_classification 需要 target.positive_definition")

        split = d.get("split_policy") or {}
        if not split.get("type"):
            errors.append("split_policy.type 缺失")
        elif split["type"] not in SPLIT_TYPES:
            errors.append(f"split_policy.type 必须是 {SPLIT_TYPES} 之一")
        if split.get("type") == "time_based":
            if not split.get("train") or not split.get("validation"):
                errors.append("time_based split 需要 train 与 validation 区间")
            if not split.get("final_test"):
                errors.append("time_based split 需要 final_test 区间(一次性盲测)")
            errors.extend(validate_split_policy_windows(split))

        if d.get("model_type") == "uplift" and not (d.get("treatment") or {}).get(
            "column"
        ):
            errors.append("uplift 任务需要 treatment.column")

        task_id = d.get("task_id", "")
        if task_id and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", task_id):
            errors.append(
                "task_id 只允许小写字母/数字/连字符/下划线"
                f"(当前: {task_id})"
            )

        budget = d.get("resource_budget") or {}
        for key in ("max_trials", "max_generations"):
            v = budget.get(key)
            if v is not None and (not isinstance(v, int) or v < 1):
                errors.append(f"resource_budget.{key} 必须是正整数")

        errors.extend(self._validate_leak_classification())
        return errors

    def _validate_leak_classification(self) -> list[str]:
        """v13 泄漏分类校验:reason 受控枚举 + 通道互斥 + 必填 plan。

        枚举设计本身杜绝"把泄漏列标成 form_unusable 偷渡回来":
        banned 的 reason 全是信息问题,reparse 的 reason 全是形态问题。
        """
        errors: list[str] = []
        feats = d = self.data.get("features") or {}
        # 通道与旧 flat 字段互斥:混用会出现两份"排除"语义
        has_struct = feats.get("banned") is not None \
            or feats.get("reparse") is not None
        if has_struct and feats.get("exclude") is not None:
            errors.append(
                "features.banned/reparse 与旧 features.exclude 不得混用"
                "(v13 起请把 exclude 迁移为 banned/reparse)")

        banned_names: set[str] = set()
        for item in feats.get("banned") or []:
            col = item["column"] if isinstance(item, dict) else str(item)
            if col in banned_names:
                errors.append(f"features.banned 重复列: {col}")
            banned_names.add(col)
            if isinstance(item, dict):
                reason = item.get("reason")
                if reason not in BANNED_REASONS:
                    errors.append(
                        f"banned[{col}].reason 必须是 {BANNED_REASONS} "
                        f"之一(受控枚举,防误标偷渡);当前: {reason!r}")
        reparse_names: list[str] = []
        for item in feats.get("reparse") or []:
            col = item["column"] if isinstance(item, dict) else str(item)
            if isinstance(item, dict):
                reason = item.get("reason")
                if reason is not None and reason not in REPARSE_REASONS:
                    errors.append(
                        f"reparse[{col}].reason 必须是 {REPARSE_REASONS} "
                        f"之一;当前: {reason!r}")
                if not str(item.get("plan") or "").strip():
                    errors.append(
                        f"reparse[{col}].plan 必填(解析去向的显式承诺,"
                        "planner 据此排探索 backlog)")
            reparse_names.append(col)

        overlap = banned_names & set(reparse_names)
        if overlap:
            errors.append(
                f"banned 与 reparse 不得交叠(信息有毒 vs 形态不可用是"
                f"互斥语义): {sorted(overlap)}")

        # 目标/时间/ID/treatment 本来就不进特征面,进分类表是语义噪音
        reserved = {self.target_name, self.time_column, self.id_column,
                    self.treatment_column}
        reserved.discard(None)

        # v2.12.2.9:确认门裁决面。kept_ambiguous 只关闭 needs_user_confirmation,
        # 不改变数据面——因此与 banned/reparse 互斥是硬约束(同一列不可能
        # 既"确认保留"又"排除"),否则两份语义打架且指纹无法区分。
        conf = feats.get("leak_confirmation") or {}
        if not isinstance(conf, dict):
            errors.append(
                "features.leak_confirmation 必须是映射(支持键: "
                "kept_ambiguous / reinstated_banned / note)")
        else:
            kept_names: list[str] = []
            for item in conf.get("kept_ambiguous") or []:
                col = item["column"] if isinstance(item, dict) else str(item)
                if col in kept_names:
                    errors.append(
                        "features.leak_confirmation.kept_ambiguous "
                        f"重复列: {col}")
                kept_names.append(col)
            kept_set = set(kept_names)
            clash = kept_set & (banned_names | set(reparse_names))
            if clash:
                errors.append(
                    "leak_confirmation.kept_ambiguous 与 banned/reparse "
                    f"不得交叠(确认保留 vs 排除是互斥语义): "
                    f"{sorted(clash)}")
            stray_kept = kept_set & reserved
            if stray_kept:
                errors.append(
                    "目标/时间/ID/treatment 列不属于泄漏分类表: "
                    f"{sorted(stray_kept)}")

            # v22.1:撤销机械封禁面。撤销的是"这个封禁判错了",因此该列
            # 必须**仍在 banned 里**(否则无从撤销),且不得同时出现在
            # reparse/kept_ambiguous(那属于改判,该走各自通道)。
            rein_names: list[str] = []
            for item in conf.get("reinstated_banned") or []:
                col = item["column"] if isinstance(item, dict) else str(item)
                if col in rein_names:
                    errors.append(
                        "features.leak_confirmation.reinstated_banned "
                        f"重复列: {col}")
                rein_names.append(col)
            rein_set = set(rein_names)
            not_banned = rein_set - banned_names
            if not_banned:
                errors.append(
                    "reinstated_banned 只能撤销**已在 features.banned 中**"
                    f"的封禁;以下列不在 banned 里: {sorted(not_banned)}")
            rein_conflict = rein_set & (kept_set | set(reparse_names))
            if rein_conflict:
                errors.append(
                    "reinstated_banned 表示重新裁决,不得与 "
                    "kept_ambiguous/reparse 同时声明(请直接选一种裁决): "
                    f"{sorted(rein_conflict)}")
            stray_rein = rein_set & reserved
            if stray_rein:
                errors.append(
                    "目标/时间/ID/treatment 列不属于泄漏分类表: "
                    f"{sorted(stray_rein)}")

        stray = (banned_names | set(reparse_names)) & reserved
        if stray:
            errors.append(
                f"目标/时间/ID/treatment 列不属于泄漏分类表: {sorted(stray)}")
        return errors

    def memory_metadata(self, profile_summary: dict | None = None) -> dict:
        """构造经验检索 metadata(§6.2)。label rate / sample size 等由
        SampleProfile 结果补充。

        v2.15(需求扩展):增加语义级任务描述字段，使检索能按
        label 定义、样本量级、特征版本等维度找到真正同类任务：
        - label_definition:目标列名 + 正样本定义 + 观测/性能窗口
        - sample_scale:量级桶（<10w / 10w~100w / 100w~1000w / >1000w）
        - feature_version:banned 族指纹 + reparse 源列表
        - split_type:时间切分 / 随机等
        - target_name:目标列名，用于精确匹配（同标签任务优先）
        """
        tgt = self.data.get("target") or {}
        split = self.data.get("split_policy") or {}
        feat = self.data.get("features") or {}

        # label_definition: 捕捉业务语义，与 objective_description 互补
        label_def = {
            "target_name": tgt.get("name"),
            "positive_definition": str(
                tgt.get("positive_definition") or "")[:200],
            "performance_window": tgt.get("performance_window"),
            "observation_window": tgt.get("observation_window"),
        }

        # feature_version: 用 banned 列族 + reparse 源的有序列表作为指纹
        # 同一批特征版本 → 可以复用 recipe；不同则需要重新验证
        banned_cols = sorted([
            (b.get("column") or b) if isinstance(b, dict) else b
            for b in (feat.get("banned") or [])
        ])
        reparse_sources = sorted([
            (r.get("column") or r) if isinstance(r, dict) else r
            for r in (feat.get("reparse") or [])
        ])
        feature_version = {
            "banned_columns": banned_cols,
            "reparse_sources": reparse_sources,
            "banned_count": len(banned_cols),
            "reparse_count": len(reparse_sources),
        }

        meta: dict[str, Any] = {
            "model_type": self.data.get("model_type"),
            "business_domains": self.data.get("business_domains", []),
            "objective_description": self.data.get("objective_description", ""),
            "primary_metric": self.data.get("primary_metric"),
            "temporal": bool(self.time_column),
            "split_type": split.get("type"),
            "target_name": tgt.get("name"),
            "label_definition": label_def,
            "feature_version": feature_version,
            "data_modalities": self._data_modalities(
                reparse_sources, split, profile_summary),
        }
        if profile_summary:
            rows = profile_summary.get("total_rows") or 0
            # 量级桶：便于模糊匹配（精确行数跨任务差异大，桶匹配更稳）
            if rows < 100_000:
                scale = "<10w"
            elif rows < 1_000_000:
                scale = "10w~100w"
            elif rows < 10_000_000:
                scale = "100w~1000w"
            else:
                scale = ">1000w"
            meta.update({
                "label_rate": profile_summary.get("label_rate"),
                "sample_size": rows,
                "sample_scale": scale,
                "feature_scale": profile_summary.get("feature_count"),
                # label_rate_range 用于 retrieve 的区间匹配
                "label_rate_range": [
                    max(0.0, (profile_summary.get("label_rate") or 0) - 0.05),
                    min(1.0, (profile_summary.get("label_rate") or 0) + 0.05),
                ] if profile_summary.get("label_rate") is not None else None,
            })
        return meta

    def _data_modalities(self, reparse_sources: list, split: dict,
                         profile_summary: dict | None) -> list[str]:
        """检索 Layer 4 用的数据模态(v2.15.8)。

        旧 memory_metadata 从不产出 data_modalities,检索 +2/模态 一次
        也没生效。spec 显式声明优先;否则按全局库现有词表推断(不产出
        过泛的 ``tabular``):reparse 源或 ``seq*`` 列 → event_sequence;
        特征数 >200 → wide_table_static;时间切分 → temporal_features
        + time_based_split。
        """
        declared = self.data.get("data_modalities")
        if isinstance(declared, list) and declared:
            return [str(m) for m in declared]
        mods: list[str] = []
        feats = self.data.get("features") or {}
        cols = [str(c) for c in (reparse_sources or [])]
        cols += [str(c) for c in (feats.get("select") or [])
                 if isinstance(c, str)]
        if reparse_sources or any(c.lower().startswith("seq") for c in cols):
            mods.append("event_sequence")
        n_feat = (profile_summary or {}).get("feature_count")
        try:
            if n_feat is not None and int(n_feat) > 200:
                mods.append("wide_table_static")
        except (TypeError, ValueError):
            pass
        if (split or {}).get("type") == "time_based":
            mods += ["temporal_features", "time_based_split"]
        return mods

    def save(self, path: Path) -> None:
        dump_yaml(Path(path), self.data)

    def summary(self) -> dict:
        """Planner 用的精简摘要(§21.2),避免全量塞入上下文。"""
        return {
            "task_id": self.data.get("task_id"),
            "task_name": self.data.get("task_name"),
            "objective_description": self.data.get("objective_description"),
            "model_type": self.data.get("model_type"),
            "business_domains": self.data.get("business_domains", []),
            "target": self.data.get("target"),
            "treatment": self.data.get("treatment"),
            "primary_metric": self.data.get("primary_metric"),
            "secondary_metrics": self.data.get("secondary_metrics", []),
            "split_policy": self.data.get("split_policy"),
            "resource_budget": self.resource_budget,
            # v2.16.0:spec.ensemble_enabled 不再授权;Planner 看到的是拓扑策略
            "model_policy": _topology_policy(self.data.get("model_type")),
            # v13:泄漏分类摘要(planner 的解析源 backlog 输入)
            "leak_classification": {
                "banned": self.banned_columns,
                "reparse": self.reparse_columns,
                "reparse_plans": self.reparse_plans,
                "leak_manifest_hash": self.leak_manifest_hash,
            },
        }
