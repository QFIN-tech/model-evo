"""v2.18.0 经验库:Alembic 迁移入口。

服务启动与嵌入式打开时都调用 :func:`upgrade`,把库升级到最新 schema。
迁移脚本在本包 ``migrations/`` 下,与服务代码解耦(嵌入式也能迁移)。
"""
from __future__ import annotations

from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def _config(url: str | None = None):
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    if url:
        # ConfigParser 插值:URL 里的 % 要转义
        cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def upgrade(engine, revision: str = "head") -> None:
    from alembic import command

    cfg = _config()
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, revision)


def head_revision() -> str:
    from alembic.script import ScriptDirectory

    return ScriptDirectory.from_config(_config()).get_current_head()
