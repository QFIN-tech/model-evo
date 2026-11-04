"""v2.18.0 中心化经验库:表定义(SQLAlchemy Core)。

规则(docs/MODELEVO_V2_18_0_EXPERIENCE_LIBRARY_DEV_20261027.md §3):
- 事实表只 INSERT。状态变化(快照状态、原则版本、排除 / 撤销)一律追加新行,
  读侧取最新一行;
- 每次写操作先插一行 ``commits``,同一事务内的事实行 ``commit_seq`` 指向它。
  ``commits.seq`` 全库单调递增,即**库水位**:读接口按 ``commit_seq <= at``
  过滤,同一水位的读结果可复现;
- 每行带 ``namespace``(组织/团队 + prod/dev/test,如 ``prod``、``team-a/dev``)。
  ``fingerprints`` / ``fingerprint_aliases`` 描述数据本身,不分命名空间。

本文件与迁移 ``migrations/versions/0001_initial.py`` 必须一致(测试比对)。
"""
from __future__ import annotations

from sqlalchemy import (JSON, BigInteger, Boolean, Column, Float, ForeignKey,
                        Index, Integer, MetaData, String, Table, Text,
                        UniqueConstraint)
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

#: JSON 列:PostgreSQL 用 JSONB,其余(SQLite)用 JSON(TEXT 存储)
JSONT = JSON().with_variant(JSONB(), "postgresql")
#: 自增大整数:SQLite 只有 INTEGER PRIMARY KEY 才是 rowid 别名(自增)
SEQ = BigInteger().with_variant(Integer(), "sqlite")

ID = String(256)       # run_id / trial_id / principle_id 等标识
HASH = String(64)      # sha256 hex
NS = String(64)
KIND = String(32)


def _fact(name: str, *cols, **kw) -> Table:
    """事实表:自增 id + namespace + commit_seq + 业务列。"""
    return Table(
        name, metadata,
        Column("id", SEQ, primary_key=True, autoincrement=True),
        Column("namespace", NS, nullable=False),
        Column("commit_seq", SEQ, ForeignKey("commits.seq"), nullable=False),
        *cols, **kw)


commits = Table(
    "commits", metadata,
    Column("seq", SEQ, primary_key=True, autoincrement=True),
    Column("namespace", NS, nullable=False),
    Column("kind", KIND, nullable=False),
    Column("actor", JSONT, nullable=False),
    Column("message", Text, nullable=False),
    Column("ref", String(128)),
    Column("created_at", String(40), nullable=False),
)

fingerprints = Table(
    "fingerprints", metadata,
    Column("atid", ID, primary_key=True),
    Column("model_type", String(32)),
    Column("target_name", ID),
    Column("sample_scale", String(32)),
    Column("col_hash8", String(32)),
    Column("data_base", String(64)),
    Column("commit_seq", SEQ, ForeignKey("commits.seq"), nullable=False),
)

fingerprint_aliases = Table(
    "fingerprint_aliases", metadata,
    Column("id", SEQ, primary_key=True, autoincrement=True),
    Column("alias_atid", ID, nullable=False),
    Column("canonical_atid", ID, nullable=False),
    Column("reason", Text, nullable=False),
    Column("commit_seq", SEQ, ForeignKey("commits.seq"), nullable=False),
    Index("ix_fingerprint_aliases_alias", "alias_atid"),
)

snapshots = _fact(
    "snapshots",
    Column("snapshot_id", HASH, nullable=False),
    Column("kind", KIND, nullable=False),
    Column("run_id", ID),
    Column("run_dir", Text),
    Column("atid", ID),
    Column("meta", JSONT, nullable=False),
    UniqueConstraint("namespace", "snapshot_id", name="uq_snapshots_ns_id"),
    Index("ix_snapshots_run", "namespace", "run_id"),
)

snapshot_states = _fact(
    "snapshot_states",
    Column("snapshot_id", HASH, nullable=False),
    Column("state", KIND, nullable=False),
    Column("detail", JSONT, nullable=False),
    Index("ix_snapshot_states_snap", "namespace", "snapshot_id"),
)

discoveries = _fact(
    "discoveries",
    Column("snapshot_id", HASH, nullable=False),
    Column("local_id", ID, nullable=False),
    Column("payload", JSONT, nullable=False),
    UniqueConstraint("namespace", "snapshot_id", "local_id",
                     name="uq_discoveries_ns_snap_local"),
)

trial_outcomes = _fact(
    "trial_outcomes",
    Column("snapshot_id", HASH, nullable=False),
    Column("run_id", ID, nullable=False),
    Column("trial_id", ID, nullable=False),
    Column("atid", ID),
    Column("generation", Integer),
    Column("action", KIND),
    Column("algorithm", String(64)),
    Column("parent_trial_ids", JSONT),
    Column("changed_modules", JSONT),
    Column("family_hints", JSONT),
    Column("data_view", String(64)),
    Column("topics", JSONT),
    Column("hypothesis", Text),
    Column("status", KIND),
    Column("verdict", KIND),
    Column("primary_metric", String(64)),
    Column("higher_is_better", Boolean),
    Column("metric_value", Float),
    Column("delta_parent", Float),
    Column("delta_parent_ci_low", Float),
    Column("delta_parent_ci_high", Float),
    Column("p_positive", Float),
    Column("delta_baseline", Float),
    Column("failure_reason", Text),
    Column("experience_ids", JSONT),
    Column("recipe_id", HASH),
    Column("extra", JSONT),
    UniqueConstraint("namespace", "snapshot_id", "trial_id",
                     name="uq_trial_outcomes_ns_snap_trial"),
    Index("ix_trial_outcomes_atid", "namespace", "atid"),
    Index("ix_trial_outcomes_run", "namespace", "run_id", "trial_id"),
)

recipes = _fact(
    "recipes",
    Column("recipe_id", HASH, nullable=False),
    Column("atid", ID),
    Column("algorithm", String(64)),
    Column("kind", KIND, nullable=False),
    Column("source_run_id", ID),
    Column("source_trial_id", ID),
    Column("runtime_api_version", String(32)),
    Column("derived_from", HASH),
    Column("derivation", String(64)),
    Column("is_hpo", Boolean, nullable=False),
    Column("n_trees", Integer),
    Column("meta", JSONT, nullable=False),
    UniqueConstraint("namespace", "recipe_id", name="uq_recipes_ns_id"),
    Index("ix_recipes_atid", "namespace", "atid"),
)

measurements = _fact(
    "measurements",
    Column("recipe_id", HASH, nullable=False),
    Column("run_id", ID),
    Column("trial_id", ID),
    Column("kind", KIND, nullable=False),
    Column("protocol", String(64)),
    Column("primary_metric", String(64)),
    Column("higher_is_better", Boolean),
    Column("value", Float),
    Column("baseline_value", Float),
    Column("delta_baseline", Float),
    Column("gain", Float),
    Column("ci_low", Float),
    Column("ci_high", Float),
    Column("p_positive", Float),
    Column("confirmed", Boolean),
    Column("extra", JSONT),
    Index("ix_measurements_recipe", "namespace", "recipe_id"),
)
#: 配方 ↔ 任务。``recipes`` 的唯一键是 ``(namespace, recipe_id)``,而 ``recipe_id``
#: 是配方内容的 sha256 —— 同一配方出现在第二个任务时配方行插不进去,只能插测量。
#: 归属因此拆到本表:同一 ``recipe_id`` 对多个 atid 各有一行(迁移 0004 回填)。
recipe_atids = _fact(
    "recipe_atids",
    Column("recipe_id", HASH, nullable=False),
    Column("atid", ID, nullable=False),
    Column("kind", KIND, nullable=False),
    UniqueConstraint("namespace", "recipe_id", "atid", "kind",
                     name="uq_recipe_atids_ns_recipe_atid_kind"),
    Index("ix_recipe_atids_atid", "namespace", "atid", "kind"),
)

principle_versions = _fact(
    "principle_versions",
    Column("principle_id", ID, nullable=False),
    Column("version", Integer, nullable=False),
    Column("claim", Text, nullable=False),
    Column("category", String(64), nullable=False),
    Column("direction", Integer, nullable=False),
    Column("topics", JSONT, nullable=False),
    Column("objects", JSONT, nullable=False),
    Column("scope", JSONT, nullable=False),
    Column("condition", JSONT, nullable=False),
    Column("effect", JSONT),
    Column("evidence", JSONT),
    Column("limitations", JSONT),
    Column("confidence", String(16)),
    Column("source_snapshot_id", HASH),
    Column("schema_version", Integer, nullable=False),
    UniqueConstraint("namespace", "principle_id", "version",
                     name="uq_principle_versions_ns_id_ver"),
)

governance_events = _fact(
    "governance_events",
    Column("subject_type", KIND, nullable=False),
    Column("subject_id", ID, nullable=False),
    Column("action", KIND, nullable=False),
    Column("payload", JSONT, nullable=False),
    Column("reason", Text, nullable=False),
    Index("ix_governance_subject", "namespace", "subject_type", "subject_id"),
)

applications = _fact(
    "applications",
    Column("experience_id", ID, nullable=False),
    Column("run_id", ID, nullable=False),
    Column("trial_id", ID, nullable=False),
    Column("binding", KIND, nullable=False),
    Column("relation", KIND),
    Column("result", KIND, nullable=False),
    Column("stance", KIND, nullable=False),
    Column("observed_delta", Float),
    Column("p_positive", Float),
    Column("generation", Integer),
    UniqueConstraint("namespace", "experience_id", "run_id", "trial_id",
                     name="uq_applications_ns_exp_run_trial"),
)

retrieval_logs = _fact(
    "retrieval_logs",
    Column("run_id", ID, nullable=False),
    Column("generation", Integer),
    Column("watermark", SEQ),
    Column("query", JSONT, nullable=False),
    Column("shown_ids", JSONT, nullable=False),
    Column("extra", JSONT),
)

#: 任务画像(v2.18.1,迁移 0005):按 atid 追加,读侧取最新一行。与 ``fingerprints``
#: 一样描述数据与任务本身,不分命名空间。内容见 :mod:`runtime.library.task_profile`。
fingerprint_profiles = Table(
    "fingerprint_profiles", metadata,
    Column("id", SEQ, primary_key=True, autoincrement=True),
    Column("atid", ID, nullable=False),
    Column("profile", JSONT, nullable=False),
    Column("profile_hash", HASH, nullable=False),
    Column("commit_seq", SEQ, ForeignKey("commits.seq"), nullable=False),
    Index("ix_fingerprint_profiles_atid", "atid"),
)

#: 当前 schema 对应的 Alembic revision(健康检查报告用)
SCHEMA_REVISION = "0005_task_profiles"

# --------------------------------------------------------------------------
# 0002_merge(M2:快照合并,dev 文档 §5 §6 §7.1)
# --------------------------------------------------------------------------

#: 主题词表。每行是某个词的一个版本;读侧取每个 code 的最新行。
#: status=alias_of 表示该词已并入 alias_of(旧词保留,不删除)。
topics = _fact(
    "topics",
    Column("code", ID, nullable=False),
    Column("parent", ID),
    Column("definition", Text, nullable=False),
    Column("aliases", JSONT, nullable=False),
    Column("status", KIND, nullable=False),
    Column("alias_of", ID),
    Column("source_snapshot_id", HASH),
    Index("ix_topics_code", "namespace", "code"),
)

#: 卷宗:任务(atid)/ 数据底座各一份,每次合并追加新版本
dossiers = _fact(
    "dossiers",
    Column("scope_type", KIND, nullable=False),
    Column("scope_id", ID, nullable=False),
    Column("version", Integer, nullable=False),
    Column("sections", JSONT, nullable=False),
    Column("source_snapshot_id", HASH),
    UniqueConstraint("namespace", "scope_type", "scope_id", "version",
                     name="uq_dossiers_ns_scope_ver"),
)

#: 更新日志(Q5):每次合并 / 撤销 / 治理逐项一行;actor(含模型)取 commits
changelog_entries = _fact(
    "changelog_entries",
    Column("snapshot_id", HASH),
    Column("entry_type", KIND, nullable=False),
    Column("action", KIND, nullable=False),
    Column("subject_type", KIND),
    Column("subject_id", ID),
    Column("summary", Text, nullable=False),
    Column("reason", Text),
    Column("evidence", JSONT),
    Index("ix_changelog_ns_seq", "namespace", "commit_seq"),
)

#: 合并租约(运维用的可变表,不参与水位)
leases = Table(
    "leases", metadata,
    Column("name", String(128), primary_key=True),
    Column("holder", JSONT, nullable=False),
    Column("token", HASH, nullable=False),
    Column("acquired_at", Float, nullable=False),
    Column("expires_at", Float, nullable=False),
)

# --------------------------------------------------------------------------
# 0003_evidence(M3:证据关联,dev 文档 §7.4)
# --------------------------------------------------------------------------

#: trial 结果与原则之间的证据关联。explicit = trial 声明了依据(--basis);
#: passive = 按主题 + 口径自动匹配(Q19)。同一 (原则, run, trial, binding) 只记一次。
evidence_links = _fact(
    "evidence_links",
    Column("principle_id", ID, nullable=False),
    Column("run_id", ID, nullable=False),
    Column("trial_id", ID, nullable=False),
    Column("binding", KIND, nullable=False),
    Column("relation", KIND, nullable=False),
    Column("declared", KIND),
    Column("base_relation", KIND, nullable=False),
    Column("match", JSONT, nullable=False),
    Column("gain", Float),
    Column("ci_low", Float),
    Column("ci_high", Float),
    Column("weight", Float, nullable=False),
    Column("trial_at", String(40)),
    Column("source_snapshot_id", HASH),
    UniqueConstraint("namespace", "principle_id", "run_id", "trial_id", "binding",
                     name="uq_evidence_links_ns_p_run_trial_binding"),
    Index("ix_evidence_links_principle", "namespace", "principle_id"),
)

