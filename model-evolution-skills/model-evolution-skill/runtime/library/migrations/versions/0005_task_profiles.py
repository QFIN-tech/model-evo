"""v2.18.1:任务画像表 fingerprint_profiles。

Revision ID: 0005_task_profiles
Revises: 0004_recipe_atids

atid 只能判断"是否同一任务";分级检索需要知道两个任务**有多像**(业务域、目标、
列集重合、量级、样本切分)。画像由 run 随快照提交,按 atid 追加(内容变化时才写),
读侧取最新一行。存量任务没有画像,检索退回 atid 推出的最小画像。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0005_task_profiles"
down_revision = "0004_recipe_atids"
branch_labels = None
depends_on = None

SEQ = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
JSONT = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "fingerprint_profiles",
        sa.Column("id", SEQ, primary_key=True, autoincrement=True),
        sa.Column("atid", sa.String(256), nullable=False),
        sa.Column("profile", JSONT, nullable=False),
        sa.Column("profile_hash", sa.String(64), nullable=False),
        sa.Column("commit_seq", SEQ, sa.ForeignKey("commits.seq"), nullable=False),
    )
    op.create_index("ix_fingerprint_profiles_atid", "fingerprint_profiles", ["atid"])


def downgrade() -> None:
    op.drop_table("fingerprint_profiles")
