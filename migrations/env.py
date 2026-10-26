"""Alembic environment: migrates db.DATABASE_URL through a sync driver.

render_as_batch is on because SQLite cannot ALTER most column properties in place; batch
mode rebuilds the table instead, and is a no-op on Postgres.
"""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import db

config = context.config
if config.config_file_name is not None and not config.attributes.get("skip_logging"):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

url = config.get_main_option("sqlalchemy.url") or db.sync_url()


def run_migrations_offline() -> None:
    context.configure(url=url, target_metadata=db.Base.metadata, render_as_batch=True,
                      literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    db.ensure_data_dir()
    engine = create_engine(url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=db.Base.metadata,
                          render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
