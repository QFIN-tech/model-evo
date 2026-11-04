"""v2.18.0 补丁:配方与任务的映射表 recipe_atids。

Revision ID: 0004_recipe_atids
Revises: 0003_evidence

``recipes`` 的唯一键是 ``(namespace, recipe_id)``,而 ``recipe_id`` 是配方内容的
sha256 —— 同一份配方出现在第二个任务(同一 baseline 被两个任务复用、或热启动把
同一配方带进别的任务)时,第二行插不进去,只能插测量。结果是该任务永远固化不了
baseline、榜单缺行、热启动无参照,而且没有任何报错。

本迁移把「配方属于哪个任务」拆成独立的事实行:同一 ``recipe_id`` 可以对多个
``atid`` 各有一行,``recipes`` 因此只描述配方本身,不再承载归属。
``recipes.atid`` 列保留(首次登记的任务),读侧一律改走本表。

回填:把现有 ``recipes.atid`` 逐行写入。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_recipe_atids"
down_revision = "0003_evidence"
branch_labels = None
depends_on = None

SEQ = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
ID = sa.String(256)
HASH = sa.String(64)
KIND = sa.String(32)


def upgrade() -> None:
    op.create_table(
        "recipe_atids",
        sa.Column("id", SEQ, primary_key=True, autoincrement=True),
        sa.Column("namespace", sa.String(64), nullable=False),
        sa.Column("commit_seq", SEQ, sa.ForeignKey("commits.seq"), nullable=False),
        sa.Column("recipe_id", HASH, nullable=False),
        sa.Column("atid", ID, nullable=False),
        sa.Column("kind", KIND, nullable=False),
        sa.UniqueConstraint("namespace", "recipe_id", "atid", "kind",
                            name="uq_recipe_atids_ns_recipe_atid_kind"),
    )
    op.create_index("ix_recipe_atids_atid", "recipe_atids",
                    ["namespace", "atid", "kind"])

    # 回填:现有配方按首次登记的 atid 补一行,commit_seq 取该配方行自己的提交
    op.execute(
        "INSERT INTO recipe_atids (namespace, commit_seq, recipe_id, atid, kind) "
        "SELECT namespace, commit_seq, recipe_id, atid, kind FROM recipes "
        "WHERE atid IS NOT NULL")


def downgrade() -> None:
    op.drop_table("recipe_atids")
