"""A colour for every person and robot, when each person last read their notifications, and
an index for paging one company's activity.

Existing rows get a random colour here, as new ones do on creation (db.random_color).

Revision ID: 0006_colors_notifications
Revises: 0005_robot_status
Create Date: 2026-10-27
"""
from __future__ import annotations

import random

import sqlalchemy as sa
from alembic import op

revision = "0006_colors_notifications"
down_revision = "0005_robot_status"
branch_labels = None
depends_on = None

COLORS = ("teal", "indigo", "amber", "rose", "green", "violet", "sky", "slate")


def _paint(table: str) -> None:
    conn = op.get_bind()
    for (row_id,) in conn.execute(sa.text(f"SELECT id FROM {table}")).all():  # noqa: S608  # fixed table names
        conn.execute(sa.text(f"UPDATE {table} SET color = :c WHERE id = :i"),  # noqa: S608
                     {"c": random.choice(COLORS), "i": row_id})  # noqa: S311  # cosmetic


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("color", sa.String(16), nullable=False, server_default="teal"))
        batch.add_column(sa.Column("notifications_seen_at", sa.DateTime(timezone=True), nullable=True))
    with op.batch_alter_table("robots") as batch:
        batch.add_column(sa.Column("color", sa.String(16), nullable=False, server_default="teal"))
    _paint("users")
    _paint("robots")
    op.create_index("ix_audit_events_org_at", "audit_events", ["org_id", "at"])


def downgrade() -> None:
    op.drop_index("ix_audit_events_org_at", "audit_events")
    with op.batch_alter_table("robots") as batch:
        batch.drop_column("color")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("notifications_seen_at")
        batch.drop_column("color")
