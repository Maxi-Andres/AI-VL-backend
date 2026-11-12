"""The agent's conversations: a chat's kind ("vlm" or "agent") and an agent exchange's trace.

Revision ID: 0013_agent_chats
Revises: 0012_robot_memory
Create Date: 2026-11-12
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0013_agent_chats"
down_revision = "0012_robot_memory"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("chats") as b:
        b.add_column(sa.Column("kind", sa.String(16), nullable=False, server_default="vlm"))
    with op.batch_alter_table("chat_messages") as b:
        b.add_column(sa.Column("trace", sa.Text(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("chat_messages") as b:
        b.drop_column("trace")
    with op.batch_alter_table("chats") as b:
        b.drop_column("kind")
