"""The vision assistant's conversations, kept per person.

Revision ID: 0007_chats
Revises: 0006_colors_notifications
Create Date: 2026-10-27
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_chats"
down_revision = "0006_colors_notifications"
branch_labels = None
depends_on = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "chats",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("org_id", sa.String(32),
                  sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(32),
                  sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(120), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
    )
    op.create_index("ix_chats_org_id", "chats", ["org_id"])
    op.create_index("ix_chats_user_id", "chats", ["user_id"])
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("chat_id", sa.String(32),
                  sa.ForeignKey("chats.id", ondelete="CASCADE"), nullable=False),
        sa.Column("at", TS, nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("question", sa.String(4000), nullable=False),
        sa.Column("answer", sa.String(20000), nullable=False),
        sa.Column("status", sa.String(300), nullable=False),
        sa.Column("robot", sa.String(16), nullable=False),
    )
    op.create_index("ix_chat_messages_chat_id", "chat_messages", ["chat_id"])


def downgrade() -> None:
    op.drop_table("chat_messages")
    op.drop_table("chats")
