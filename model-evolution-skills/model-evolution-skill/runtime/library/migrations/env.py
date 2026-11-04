"""Alembic 环境:只支持由 runtime.library.migrate 传入现成连接(online)。"""
from __future__ import annotations

from alembic import context

from runtime.library.schema import metadata

config = context.config
connection = config.attributes.get("connection")
if connection is None:
    raise RuntimeError("请通过 runtime.library.migrate.upgrade(engine) 执行迁移")

context.configure(connection=connection, target_metadata=metadata,
                  render_as_batch=connection.dialect.name == "sqlite",
                  compare_type=True)
with context.begin_transaction():
    context.run_migrations()
