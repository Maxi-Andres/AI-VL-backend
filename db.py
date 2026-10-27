"""db.py — the gateway's own small database: organizations, users, sessions, keys, audit.

Everything else this app does is stateless relaying; this is the one thing it must remember.
SQLite by default (one file, nothing to run), reached through SQLAlchemy so that moving to
Postgres is a DATABASE_URL change, not a rewrite. The schema is owned by the Alembic
migrations in `migrations/`, which run on startup — never `create_all` in production.

The file lives OUTSIDE the repo tree (engineering standard §2: secrets and state live outside
the repo), in PILOT_DATA_DIR (default ~/.pilot), created 0700.

Config via env:
    PILOT_DATA_DIR  where the database and the first-run setup code live (default ~/.pilot)
    DATABASE_URL    full SQLAlchemy async URL; overrides the SQLite file in PILOT_DATA_DIR
"""
from __future__ import annotations

import os
import random
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    TypeDecorator,
    UniqueConstraint,
    event,
)
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DATA_DIR = Path(os.environ.get("PILOT_DATA_DIR", "~/.pilot")).expanduser()
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip() or (
    f"sqlite+aiosqlite:///{DATA_DIR / 'pilot.db'}"
)


def sync_url(url: str = DATABASE_URL) -> str:
    """The same database through a sync driver, for Alembic (it migrates synchronously)."""
    return url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg")


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)


def new_id() -> str:
    return uuid.uuid4().hex


# The colours a person or a robot can wear (their avatar, their icon). Each gets one at random
# when created — so a team reads apart at a glance — and may change it. Names, not hex: the
# frontend owns the actual shades for its light and dark themes.
COLORS = ("teal", "indigo", "amber", "rose", "green", "violet", "sky", "slate")


def random_color() -> str:
    return random.choice(COLORS)  # noqa: S311  # a cosmetic pick, not a secret


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC in Python on every backend. SQLite has no timezone type, so it
    stores naive UTC and hands it back naive; this re-attaches UTC on the way out, so no
    comparison anywhere ever mixes naive and aware datetimes."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is None:
            raise ValueError("naive datetime stored; use db.utcnow()")
        return value.astimezone(UTC) if value is not None else None

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


class Base(DeclarativeBase):
    pass


class Organization(Base):
    """The customer company. One per install today; every row that belongs to a company
    carries org_id anyway, so a multi-company deploy needs no schema change."""

    __tablename__ = "organizations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    # Stored lowercased: the address is the login name, and case must not create two people.
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    color: Mapped[str] = mapped_column(String(16), default=random_color)
    # Notifications (the bell) newer than this are unread for this person.
    notifications_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class Membership(Base):
    """A user's role in an organization. The role is per company, not per user."""

    __tablename__ = "memberships"
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    org_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("organizations.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class AuthSession(Base):
    """A signed-in browser. The id is the SHA-256 of the cookie value, never the value
    itself, so a copy of this table cannot be replayed as anyone's session."""

    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    org_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    persistent: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    user_agent: Mapped[str] = mapped_column(String(255), default="")
    ip: Mapped[str] = mapped_column(String(64), default="")


class ApiKey(Base):
    """A credential for a machine, not a person — today the camera bridge that pushes the
    robot's video in. Only the SHA-256 is stored; the key is shown once, at creation."""

    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    org_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("organizations.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String(80))
    role: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # The first characters of the key, kept so a list of keys is recognizable.
    prefix: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class Invitation(Base):
    """An offer to join an organization with a role. The link carries a random token; only
    its SHA-256 is stored, and it works once, until it expires or is revoked."""

    __tablename__ = "invitations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    org_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    email: Mapped[str] = mapped_column(String(254))
    role: Mapped[str] = mapped_column(String(16))
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    invited_by: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
    accepted_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class PasswordReset(Base):
    """A one-time link an admin hands to someone who forgot their password (later: emailed).
    Only the token's SHA-256 is stored; it works once, until it expires or a newer one is made."""

    __tablename__ = "password_resets"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_by: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
    used_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class Site(Base):
    """A place where the company's robots work: a warehouse, a plant, a lab."""

    __tablename__ = "sites"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    org_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120))
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    address: Mapped[str] = mapped_column(String(255), default="")
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    # The floor plan is a file in PILOT_DATA_DIR/site-plans/, named by the site id; this is
    # its media type, or empty when the site has none.
    plan_type: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class Robot(Base):
    """A robot the company owns. `slot` is how the running stack addresses it ("go2", "g1"):
    the executor, the relay and the camera bridge know one robot per model today, so a slot
    holds one robot per organization (see the two-robots decision in the roadmap)."""

    __tablename__ = "robots"
    __table_args__ = (UniqueConstraint("org_id", "slot", name="uq_robots_org_slot"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    org_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    site_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("sites.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(80))
    slot: Mapped[str] = mapped_column(String(16))
    serial: Mapped[str] = mapped_column(String(80), default="")
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    color: Mapped[str] = mapped_column(String(16), default=random_color)
    # Last known state, written by robot_watch.py every few seconds whether or not anyone has
    # the app open — so a card can say "offline · 2 h" and still show the last battery.
    # None everywhere means never measured (no ping address configured, or not yet).
    online: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    status_since: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    link: Mapped[str | None] = mapped_column(String(16), nullable=True)
    battery_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    battery_charging: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    battery_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)


class AuditEvent(Base):
    """Who did what, when, from where. Append-only by convention: nothing updates or
    deletes these rows."""

    __tablename__ = "audit_events"
    # The activity log reads one company's events newest first, a page at a time.
    __table_args__ = (Index("ix_audit_events_org_at", "org_id", "at"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, index=True)
    org_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    actor_user_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    actor_key_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(255), default="")
    ip: Mapped[str] = mapped_column(String(64), default="")
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)


engine: AsyncEngine = create_async_engine(DATABASE_URL)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

if engine.dialect.name == "sqlite":

    @event.listens_for(engine.sync_engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _record):
        # WAL: readers never block the writer (a login while sessions are being touched).
        # foreign_keys: SQLite ignores ON DELETE CASCADE without it.
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


def migrate() -> None:
    """Bring the schema to the newest migration. Synchronous on purpose: it runs once, in a
    worker thread, before the app accepts a request."""
    from alembic import command
    from alembic.config import Config

    here = Path(__file__).resolve().parent
    cfg = Config(str(here / "alembic.ini"))
    cfg.set_main_option("script_location", str(here / "migrations"))
    cfg.set_main_option("sqlalchemy.url", sync_url())
    # Leave logging alone: uvicorn owns it, and alembic.ini's [loggers] would replace it.
    cfg.attributes["skip_logging"] = True
    command.upgrade(cfg, "head")
