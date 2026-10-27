"""The address people use to reach this Pilot, for invitation and reset links.

Revision ID: 0008_public_url
Revises: 0007_chats
Create Date: 2026-10-27
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_public_url"
down_revision = "0007_chats"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("organizations") as batch:
        batch.add_column(sa.Column("public_url", sa.String(255), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("organizations") as batch:
        batch.drop_column("public_url")
