"""Each person's preferences (theme, voice, volume…), kept with their account.

Revision ID: 0009_preferences
Revises: 0008_public_url
Create Date: 2026-10-27
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0009_preferences"
down_revision = "0008_public_url"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("preferences", sa.JSON(), nullable=False, server_default="{}"))


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_column("preferences")
