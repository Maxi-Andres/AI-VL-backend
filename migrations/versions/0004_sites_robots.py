"""Sites and robots: where the company works, and which robots it has there.

Revision ID: 0004_sites_robots
Revises: 0003_password_resets
Create Date: 2026-10-26
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_sites_robots"
down_revision = "0003_password_resets"
branch_labels = None
depends_on = None

TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "sites",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("org_id", sa.String(32),
                  sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("timezone", sa.String(64), nullable=False),
        sa.Column("address", sa.String(255), nullable=False),
        sa.Column("lat", sa.Float(), nullable=True),
        sa.Column("lon", sa.Float(), nullable=True),
        sa.Column("plan_type", sa.String(32), nullable=False),
        sa.Column("created_at", TS, nullable=False),
    )
    op.create_index("ix_sites_org_id", "sites", ["org_id"])
    op.create_table(
        "robots",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("org_id", sa.String(32),
                  sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("site_id", sa.String(32), sa.ForeignKey("sites.id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("slot", sa.String(16), nullable=False),
        sa.Column("serial", sa.String(80), nullable=False),
        sa.Column("created_at", TS, nullable=False),
        sa.UniqueConstraint("org_id", "slot", name="uq_robots_org_slot"),
    )
    op.create_index("ix_robots_org_id", "robots", ["org_id"])


def downgrade() -> None:
    op.drop_table("robots")
    op.drop_table("sites")
