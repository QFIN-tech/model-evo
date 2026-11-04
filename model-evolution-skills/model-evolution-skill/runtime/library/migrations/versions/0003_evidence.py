"""v2.18.0 M3:证据关联表 evidence_links(dev 文档 §7.4)。

Revision ID: 0003_evidence
Revises: 0002_merge
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0003_evidence"
down_revision = "0002_merge"
branch_labels = None
depends_on = None

JSONT = sa.JSON().with_variant(JSONB(), "postgresql")
SEQ = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
ID = sa.String(256)
HASH = sa.String(64)
KIND = sa.String(32)


def upgrade() -> None:
    op.create_table(
        "evidence_links",
        sa.Column("id", SEQ, primary_key=True, autoincrement=True),
        sa.Column("namespace", sa.String(64), nullable=False),
        sa.Column("commit_seq", SEQ, sa.ForeignKey("commits.seq"), nullable=False),
        sa.Column("principle_id", ID, nullable=False),
        sa.Column("run_id", ID, nullable=False),
        sa.Column("trial_id", ID, nullable=False),
        sa.Column("binding", KIND, nullable=False),
        sa.Column("relation", KIND, nullable=False),
        sa.Column("declared", KIND),
        sa.Column("base_relation", KIND, nullable=False),
        sa.Column("match", JSONT, nullable=False),
        sa.Column("gain", sa.Float),
        sa.Column("ci_low", sa.Float),
        sa.Column("ci_high", sa.Float),
        sa.Column("weight", sa.Float, nullable=False),
        sa.Column("trial_at", sa.String(40)),
        sa.Column("source_snapshot_id", HASH),
        sa.UniqueConstraint("namespace", "principle_id", "run_id", "trial_id", "binding",
                            name="uq_evidence_links_ns_p_run_trial_binding"),
    )
    op.create_index("ix_evidence_links_principle", "evidence_links",
                    ["namespace", "principle_id"])


def downgrade() -> None:
    op.drop_table("evidence_links")
