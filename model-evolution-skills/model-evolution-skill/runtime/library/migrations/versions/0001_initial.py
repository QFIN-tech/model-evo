"""v2.18.0 经验库初始 schema(dev 文档 §3.2)。

Revision ID: 0001_initial
Revises:
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

JSONT = sa.JSON().with_variant(JSONB(), "postgresql")
SEQ = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
ID = sa.String(256)
HASH = sa.String(64)
NS = sa.String(64)
KIND = sa.String(32)


def _fact_cols():
    return [
        sa.Column("id", SEQ, primary_key=True, autoincrement=True),
        sa.Column("namespace", NS, nullable=False),
        sa.Column("commit_seq", SEQ, sa.ForeignKey("commits.seq"), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "commits",
        sa.Column("seq", SEQ, primary_key=True, autoincrement=True),
        sa.Column("namespace", NS, nullable=False),
        sa.Column("kind", KIND, nullable=False),
        sa.Column("actor", JSONT, nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("ref", sa.String(128)),
        sa.Column("created_at", sa.String(40), nullable=False),
    )
    op.create_table(
        "fingerprints",
        sa.Column("atid", ID, primary_key=True),
        sa.Column("model_type", sa.String(32)),
        sa.Column("target_name", ID),
        sa.Column("sample_scale", sa.String(32)),
        sa.Column("col_hash8", sa.String(32)),
        sa.Column("data_base", sa.String(64)),
        sa.Column("commit_seq", SEQ, sa.ForeignKey("commits.seq"), nullable=False),
    )
    op.create_table(
        "fingerprint_aliases",
        sa.Column("id", SEQ, primary_key=True, autoincrement=True),
        sa.Column("alias_atid", ID, nullable=False),
        sa.Column("canonical_atid", ID, nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
        sa.Column("commit_seq", SEQ, sa.ForeignKey("commits.seq"), nullable=False),
    )
    op.create_index("ix_fingerprint_aliases_alias", "fingerprint_aliases",
                    ["alias_atid"])

    op.create_table(
        "snapshots", *_fact_cols(),
        sa.Column("snapshot_id", HASH, nullable=False),
        sa.Column("kind", KIND, nullable=False),
        sa.Column("run_id", ID),
        sa.Column("run_dir", sa.Text),
        sa.Column("atid", ID),
        sa.Column("meta", JSONT, nullable=False),
        sa.UniqueConstraint("namespace", "snapshot_id", name="uq_snapshots_ns_id"),
    )
    op.create_index("ix_snapshots_run", "snapshots", ["namespace", "run_id"])

    op.create_table(
        "snapshot_states", *_fact_cols(),
        sa.Column("snapshot_id", HASH, nullable=False),
        sa.Column("state", KIND, nullable=False),
        sa.Column("detail", JSONT, nullable=False),
    )
    op.create_index("ix_snapshot_states_snap", "snapshot_states",
                    ["namespace", "snapshot_id"])

    op.create_table(
        "discoveries", *_fact_cols(),
        sa.Column("snapshot_id", HASH, nullable=False),
        sa.Column("local_id", ID, nullable=False),
        sa.Column("payload", JSONT, nullable=False),
        sa.UniqueConstraint("namespace", "snapshot_id", "local_id",
                            name="uq_discoveries_ns_snap_local"),
    )

    op.create_table(
        "trial_outcomes", *_fact_cols(),
        sa.Column("snapshot_id", HASH, nullable=False),
        sa.Column("run_id", ID, nullable=False),
        sa.Column("trial_id", ID, nullable=False),
        sa.Column("atid", ID),
        sa.Column("generation", sa.Integer),
        sa.Column("action", KIND),
        sa.Column("algorithm", sa.String(64)),
        sa.Column("parent_trial_ids", JSONT),
        sa.Column("changed_modules", JSONT),
        sa.Column("family_hints", JSONT),
        sa.Column("data_view", sa.String(64)),
        sa.Column("topics", JSONT),
        sa.Column("hypothesis", sa.Text),
        sa.Column("status", KIND),
        sa.Column("verdict", KIND),
        sa.Column("primary_metric", sa.String(64)),
        sa.Column("higher_is_better", sa.Boolean),
        sa.Column("metric_value", sa.Float),
        sa.Column("delta_parent", sa.Float),
        sa.Column("delta_parent_ci_low", sa.Float),
        sa.Column("delta_parent_ci_high", sa.Float),
        sa.Column("p_positive", sa.Float),
        sa.Column("delta_baseline", sa.Float),
        sa.Column("failure_reason", sa.Text),
        sa.Column("experience_ids", JSONT),
        sa.Column("recipe_id", HASH),
        sa.Column("extra", JSONT),
        sa.UniqueConstraint("namespace", "snapshot_id", "trial_id",
                            name="uq_trial_outcomes_ns_snap_trial"),
    )
    op.create_index("ix_trial_outcomes_atid", "trial_outcomes", ["namespace", "atid"])
    op.create_index("ix_trial_outcomes_run", "trial_outcomes",
                    ["namespace", "run_id", "trial_id"])

    op.create_table(
        "recipes", *_fact_cols(),
        sa.Column("recipe_id", HASH, nullable=False),
        sa.Column("atid", ID),
        sa.Column("algorithm", sa.String(64)),
        sa.Column("kind", KIND, nullable=False),
        sa.Column("source_run_id", ID),
        sa.Column("source_trial_id", ID),
        sa.Column("runtime_api_version", sa.String(32)),
        sa.Column("derived_from", HASH),
        sa.Column("derivation", sa.String(64)),
        sa.Column("is_hpo", sa.Boolean, nullable=False),
        sa.Column("n_trees", sa.Integer),
        sa.Column("meta", JSONT, nullable=False),
        sa.UniqueConstraint("namespace", "recipe_id", name="uq_recipes_ns_id"),
    )
    op.create_index("ix_recipes_atid", "recipes", ["namespace", "atid"])

    op.create_table(
        "measurements", *_fact_cols(),
        sa.Column("recipe_id", HASH, nullable=False),
        sa.Column("run_id", ID),
        sa.Column("trial_id", ID),
        sa.Column("kind", KIND, nullable=False),
        sa.Column("protocol", sa.String(64)),
        sa.Column("primary_metric", sa.String(64)),
        sa.Column("higher_is_better", sa.Boolean),
        sa.Column("value", sa.Float),
        sa.Column("baseline_value", sa.Float),
        sa.Column("delta_baseline", sa.Float),
        sa.Column("gain", sa.Float),
        sa.Column("ci_low", sa.Float),
        sa.Column("ci_high", sa.Float),
        sa.Column("p_positive", sa.Float),
        sa.Column("confirmed", sa.Boolean),
        sa.Column("extra", JSONT),
    )
    op.create_index("ix_measurements_recipe", "measurements", ["namespace", "recipe_id"])

    op.create_table(
        "principle_versions", *_fact_cols(),
        sa.Column("principle_id", ID, nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("claim", sa.Text, nullable=False),
        sa.Column("category", sa.String(64), nullable=False),
        sa.Column("direction", sa.Integer, nullable=False),
        sa.Column("topics", JSONT, nullable=False),
        sa.Column("objects", JSONT, nullable=False),
        sa.Column("scope", JSONT, nullable=False),
        sa.Column("condition", JSONT, nullable=False),
        sa.Column("effect", JSONT),
        sa.Column("evidence", JSONT),
        sa.Column("limitations", JSONT),
        sa.Column("confidence", sa.String(16)),
        sa.Column("source_snapshot_id", HASH),
        sa.Column("schema_version", sa.Integer, nullable=False),
        sa.UniqueConstraint("namespace", "principle_id", "version",
                            name="uq_principle_versions_ns_id_ver"),
    )

    op.create_table(
        "governance_events", *_fact_cols(),
        sa.Column("subject_type", KIND, nullable=False),
        sa.Column("subject_id", ID, nullable=False),
        sa.Column("action", KIND, nullable=False),
        sa.Column("payload", JSONT, nullable=False),
        sa.Column("reason", sa.Text, nullable=False),
    )
    op.create_index("ix_governance_subject", "governance_events",
                    ["namespace", "subject_type", "subject_id"])

    op.create_table(
        "applications", *_fact_cols(),
        sa.Column("experience_id", ID, nullable=False),
        sa.Column("run_id", ID, nullable=False),
        sa.Column("trial_id", ID, nullable=False),
        sa.Column("binding", KIND, nullable=False),
        sa.Column("relation", KIND),
        sa.Column("result", KIND, nullable=False),
        sa.Column("stance", KIND, nullable=False),
        sa.Column("observed_delta", sa.Float),
        sa.Column("p_positive", sa.Float),
        sa.Column("generation", sa.Integer),
        sa.UniqueConstraint("namespace", "experience_id", "run_id", "trial_id",
                            name="uq_applications_ns_exp_run_trial"),
    )

    op.create_table(
        "retrieval_logs", *_fact_cols(),
        sa.Column("run_id", ID, nullable=False),
        sa.Column("generation", sa.Integer),
        sa.Column("watermark", SEQ),
        sa.Column("query", JSONT, nullable=False),
        sa.Column("shown_ids", JSONT, nullable=False),
        sa.Column("extra", JSONT),
    )


def downgrade() -> None:
    for name in ("retrieval_logs", "applications", "governance_events",
                 "principle_versions", "measurements", "recipes", "trial_outcomes",
                 "discoveries", "snapshot_states", "snapshots",
                 "fingerprint_aliases", "fingerprints", "commits"):
        op.drop_table(name)
