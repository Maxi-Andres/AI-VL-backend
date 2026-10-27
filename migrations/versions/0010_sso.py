"""Single sign-on (OIDC): the company's identity provider, and each person's link to it.

Revision ID: 0010_sso
Revises: 0009_preferences
Create Date: 2026-10-27
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0010_sso"
down_revision = "0009_preferences"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("organizations") as batch:
        batch.add_column(sa.Column("sso_enabled", sa.Boolean(), nullable=False,
                                   server_default=sa.false()))
        batch.add_column(sa.Column("sso_provider", sa.String(16), nullable=True))
        batch.add_column(sa.Column("sso_issuer", sa.String(255), nullable=True))
        batch.add_column(sa.Column("sso_client_id", sa.String(255), nullable=True))
        batch.add_column(sa.Column("sso_client_secret", sa.String(512), nullable=True))
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("sso_subject", sa.String(255), nullable=True))
        batch.create_index("ix_users_sso_subject", ["sso_subject"], unique=True)


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_index("ix_users_sso_subject")
        batch.drop_column("sso_subject")
    with op.batch_alter_table("organizations") as batch:
        for col in ("sso_client_secret", "sso_client_id", "sso_issuer", "sso_provider", "sso_enabled"):
            batch.drop_column(col)
