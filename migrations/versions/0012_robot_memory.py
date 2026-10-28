"""Each robot's memory: the pictures and notes it was asked to keep.

Revision ID: 0012_robot_memory
Revises: 0011_password_from_internet
Create Date: 2026-10-28
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0012_robot_memory"
down_revision = "0011_password_from_internet"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memories",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("org_id", sa.String(32), sa.ForeignKey("organizations.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("robot_id", sa.String(32), sa.ForeignKey("robots.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("kind", sa.String(8), nullable=False),
        sa.Column("text", sa.String(500), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(32), sa.ForeignKey("users.id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("created_by_name", sa.String(120), nullable=False, server_default=""),
    )
    op.create_index("ix_memories_robot_at", "memories", ["robot_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_memories_robot_at", table_name="memories")
    op.drop_table("memories")
