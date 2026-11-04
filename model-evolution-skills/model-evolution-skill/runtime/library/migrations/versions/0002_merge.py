"""v2.18.0 M2:快照合并 —— 主题词表、卷宗、更新日志、合并租约(dev 文档 §5 §6 §7.1)。

Revision ID: 0002_merge
Revises: 0001_initial
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0002_merge"
down_revision = "0001_initial"
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
        "topics", *_fact_cols(),
        sa.Column("code", ID, nullable=False),
        sa.Column("parent", ID),
        sa.Column("definition", sa.Text, nullable=False),
        sa.Column("aliases", JSONT, nullable=False),
        sa.Column("status", KIND, nullable=False),
        sa.Column("alias_of", ID),
        sa.Column("source_snapshot_id", HASH),
    )
    op.create_index("ix_topics_code", "topics", ["namespace", "code"])

    op.create_table(
        "dossiers", *_fact_cols(),
        sa.Column("scope_type", KIND, nullable=False),
        sa.Column("scope_id", ID, nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("sections", JSONT, nullable=False),
        sa.Column("source_snapshot_id", HASH),
        sa.UniqueConstraint("namespace", "scope_type", "scope_id", "version",
                            name="uq_dossiers_ns_scope_ver"),
    )

    op.create_table(
        "changelog_entries", *_fact_cols(),
        sa.Column("snapshot_id", HASH),
        sa.Column("entry_type", KIND, nullable=False),
        sa.Column("action", KIND, nullable=False),
        sa.Column("subject_type", KIND),
        sa.Column("subject_id", ID),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("reason", sa.Text),
        sa.Column("evidence", JSONT),
    )
    op.create_index("ix_changelog_ns_seq", "changelog_entries",
                    ["namespace", "commit_seq"])

    op.create_table(
        "leases",
        sa.Column("name", sa.String(128), primary_key=True),
        sa.Column("holder", JSONT, nullable=False),
        sa.Column("token", HASH, nullable=False),
        sa.Column("acquired_at", sa.Float, nullable=False),
        sa.Column("expires_at", sa.Float, nullable=False),
    )


def downgrade() -> None:
    for name in ("leases", "changelog_entries", "dossiers", "topics"):
        op.drop_table(name)
