"""With single sign-on on, may a password be used from the internet? Off unless an owner says.

Revision ID: 0011_password_from_internet
Revises: 0010_sso
Create Date: 2026-10-27
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011_password_from_internet"
down_revision = "0010_sso"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("organizations") as batch:
        batch.add_column(sa.Column("sso_password_anywhere", sa.Boolean(), nullable=False,
                                   server_default=sa.false()))


def downgrade() -> None:
    with op.batch_alter_table("organizations") as batch:
        batch.drop_column("sso_password_anywhere")
