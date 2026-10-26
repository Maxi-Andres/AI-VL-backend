"""A robot's last known state: online since when, last seen, latency, link, last battery.

Revision ID: 0005_robot_status
Revises: 0004_sites_robots
Create Date: 2026-10-26
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_robot_status"
down_revision = "0004_sites_robots"
branch_labels = None
depends_on = None

TS = sa.DateTime(timezone=True)
COLUMNS = (
    ("online", sa.Boolean()),
    ("status_since", TS),
    ("last_seen_at", TS),
    ("latency_ms", sa.Float()),
    ("link", sa.String(16)),
    ("battery_pct", sa.Float()),
    ("battery_charging", sa.Boolean()),
    ("battery_at", TS),
)


def upgrade() -> None:
    with op.batch_alter_table("robots") as batch:
        for name, kind in COLUMNS:
            batch.add_column(sa.Column(name, kind, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("robots") as batch:
        for name, _ in reversed(COLUMNS):
            batch.drop_column(name)
