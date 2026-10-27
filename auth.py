"""auth.py — who is calling, and are they allowed to.

Every request and every WebSocket the gateway serves goes through `authorize`, installed as
an app-wide dependency in app.py. It is DEFAULT-DENY: a route is reachable only if POLICY
names it, and `tests/test_auth.py` fails the build when a route exists without a policy entry
— so a new endpoint is closed until someone decides who may call it.

Two kinds of caller:

- A PERSON, signed in from the browser. The credential is an opaque random token in an
  httpOnly, SameSite=Strict cookie; the server keeps the session (table `sessions`, keyed by
  the token's SHA-256). Server-side sessions instead of JWT access/refresh tokens on purpose:
  the SPA and the API share one origin, so nothing needs a self-contained token, and a
  server-side session can be revoked on the spot — sign out, remove a user, change a role —
  including the WebSockets that are already open.
- A MACHINE, with an API key in `Authorization: Bearer pk_…` — today only the camera bridge,
  which pushes the robot's video in. Keys are created with `pilot_admin.py`.

Roles are per organization: viewer < operator < admin < owner, each a set of permissions
(PERMISSIONS). The UI hides what a role cannot do, but THIS is what enforces it.

The camera bridge's key is created by this module, not by a person: when the install is set
up (and on any startup where it is missing) a "camera bridge" key is written to
PILOT_DATA_DIR/device-key (0600), where linux/run-with-robot.sh picks it up for the container.

A brand-new install has no accounts. Until the first one exists, a one-time setup code is
written to PILOT_DATA_DIR/setup-code (0600) and POST /api/auth/setup needs it — so the first
person on the network to open the page cannot claim an install that is not theirs.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import re
import secrets
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import APIRouter, HTTPException, Response, WebSocket, WebSocketException
from fastapi.requests import HTTPConnection
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update

import db

logger = logging.getLogger("uvicorn.error")

# --------------------------------------------------------------------------- #
# Roles and permissions
# --------------------------------------------------------------------------- #
ROLES = ("viewer", "operator", "admin", "owner")
_VIEW = {"view"}
_OPERATE = _VIEW | {"analyze", "operate"}
_ADMIN = _OPERATE | {"configure", "manage_users"}
PERMISSIONS: dict[str, frozenset[str]] = {
    "viewer": frozenset(_VIEW),        # watch video and telemetry
    "operator": frozenset(_OPERATE),   # + ask the AI, drive, act, speak, move joints
    "admin": frozenset(_ADMIN),        # + robot/video settings, people
    "owner": frozenset(_ADMIN | {"manage_org"}),
    "device": frozenset({"ingest"}),   # a machine pushing data in (API keys only)
}

PUBLIC = "public"                # no credential at all
AUTHENTICATED = "authenticated"  # any signed-in person, whatever the role

# (method, route path) -> what the caller needs. Method "WS" for WebSockets. Grouped by
# permission so a review reads it as "who can do what", not as a route list.
POLICY: dict[tuple[str, str], str] = {
    # The sign-in flow and the SPA shell itself (it has to load to show the login page).
    ("GET", "/api/auth/state"): PUBLIC,
    ("POST", "/api/auth/setup"): PUBLIC,
    ("POST", "/api/auth/login"): PUBLIC,
    ("GET", "/{full_path}"): PUBLIC,  # the SPA catch-all (path_format drops the ":path")
    ("POST", "/api/auth/logout"): AUTHENTICATED,
    ("GET", "/api/auth/me"): AUTHENTICATED,
    ("PATCH", "/api/auth/me"): AUTHENTICATED,          # your own name (Settings → Profile)
    ("POST", "/api/auth/password"): AUTHENTICATED,     # your own password, knowing the current
    # Joining through an invitation: the token in the body is the credential (team.py).
    ("POST", "/api/invitations/lookup"): PUBLIC,
    ("POST", "/api/invitations/accept"): PUBLIC,
    # manage_users — the team: who is in it, with what role (team.py).
    ("GET", "/api/team"): "manage_users",
    ("GET", "/api/team/address"): "manage_users",   # the address invitation links point to
    ("PUT", "/api/team/address"): "manage_users",
    ("GET", "/api/team/online"): AUTHENTICATED,   # who has Pilot open: names and colours only
    ("POST", "/api/team/invitations"): "manage_users",
    ("DELETE", "/api/team/invitations/{invitation_id}"): "manage_users",
    ("PATCH", "/api/team/members/{user_id}"): "manage_users",
    ("POST", "/api/team/members/{user_id}/password-reset"): "manage_users",
    # Setting a new password through a reset link: the token is the credential (team.py).
    ("POST", "/api/password-resets/lookup"): PUBLIC,
    ("POST", "/api/password-resets/complete"): PUBLIC,
    # view — look, never touch.
    ("GET", "/api/health"): "view",
    ("GET", "/api/options"): "view",
    ("GET", "/api/classes"): "view",
    ("GET", "/api/skills"): "view",
    ("GET", "/api/presence"): "view",
    ("GET", "/api/robot-net"): "view",
    ("GET", "/api/robot-transport"): "view",
    ("GET", "/api/robot-video"): "view",
    ("GET", "/api/robot-frame"): "view",
    ("GET", "/api/robot-camera/status"): "view",
    ("GET", "/api/robot-audio/volume"): "view",
    ("GET", "/api/tts/voices"): "view",
    ("WS", "/ws/view"): "view",
    ("WS", "/ws/view-h264"): "view",
    ("WS", "/ws/joints"): "view",
    ("GET", "/api/fleet"): "view",
    ("GET", "/api/sites/{site_id}/plan"): "view",
    # analyze — spend GPU on the models; nothing reaches the robot.
    ("POST", "/api/vlm"): "analyze",
    # The assistant's conversations, each private to its person (chats.py checks the owner).
    ("GET", "/api/chats"): "analyze",
    ("GET", "/api/chats/{chat_id}"): "analyze",
    ("POST", "/api/chats/messages"): "analyze",
    ("POST", "/api/chats/{chat_id}/messages"): "analyze",
    ("PATCH", "/api/chats/{chat_id}"): "analyze",
    ("DELETE", "/api/chats/{chat_id}"): "analyze",
    ("POST", "/api/vlm/stream"): "analyze",
    ("POST", "/api/detect"): "analyze",
    ("POST", "/api/transcribe"): "analyze",
    ("POST", "/api/speak"): "analyze",
    ("POST", "/api/command"): "analyze",  # interprets only; executing is /api/execute
    ("WS", "/ws/detect"): "analyze",
    # operate — the robot moves, speaks, or its microphone opens.
    ("POST", "/api/execute"): "operate",
    ("POST", "/api/robot-camera/{action}"): "operate",
    ("POST", "/api/robot-audio/say"): "operate",
    ("POST", "/api/robot-audio/stop"): "operate",
    ("POST", "/api/robot-audio/volume"): "operate",
    ("WS", "/ws/joint-control"): "operate",
    ("WS", "/ws/robot-ear"): "operate",
    # configure — change how the stack reaches and streams from the robot.
    ("POST", "/api/robot-net"): "configure",
    ("POST", "/api/robot-transport"): "configure",
    ("POST", "/api/robot-video"): "configure",
    ("POST", "/api/robot-camera/config"): "configure",
    # configure, sites and robots (fleet.py).
    ("GET", "/api/activity"): "configure",
    # The bell: each person's own notifications (what they may see is decided in fleet.py).
    ("GET", "/api/notifications"): AUTHENTICATED,
    ("POST", "/api/notifications/seen"): AUTHENTICATED,
    ("POST", "/api/sites"): "configure",
    ("PATCH", "/api/sites/{site_id}"): "configure",
    ("DELETE", "/api/sites/{site_id}"): "configure",
    ("PUT", "/api/sites/{site_id}/plan"): "configure",
    ("DELETE", "/api/sites/{site_id}/plan"): "configure",
    ("POST", "/api/robots"): "configure",
    ("PATCH", "/api/robots/{robot_id}"): "configure",
    ("DELETE", "/api/robots/{robot_id}"): "configure",
    # ingest — machines pushing the robot's video in.
    ("WS", "/ws/robot-cam"): "ingest",
    ("WS", "/ws/robot-h264"): "ingest",
}

# Audited on every call: rare, and each one changes something for everybody.
AUDITED = frozenset({"configure"})
# Driving sends /api/execute several times a second, so "operate" is audited at most once per
# OPERATE_AUDIT_EVERY per session and route: the log answers "who was driving, and when"
# without a row per stick sample. Written in the background — the control path never waits.
OPERATE_AUDIT_EVERY = 60.0
_operate_audited: dict[tuple[str, str], float] = {}
_background: set[asyncio.Task] = set()

# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
COOKIE_NAME = "pilot_session"
SHORT_SESSION = timedelta(hours=12)   # "keep me signed in" off: idle timeout
LONG_SESSION = timedelta(days=30)     # "keep me signed in" on: idle timeout
ABSOLUTE_SESSION = timedelta(days=90)  # never longer than this, however active
TOUCH_EVERY = timedelta(minutes=1)    # write last_seen at most this often per session
# Validated principals are kept in memory this long, so the control path (/api/execute
# every SEND_MS while driving) does not pay a database read per request. Revocations made
# by this process evict immediately; a change made by another process (pilot_admin.py)
# takes up to this long to land.
CACHE_TTL = 15.0
# Cookie Secure flag: on for HTTPS (the deploy), off for plain-HTTP dev. COOKIE_SECURE=1
# forces it on behind a TLS-terminating proxy, where this app sees http.
FORCE_SECURE_COOKIE = os.environ.get("COOKIE_SECURE", "").strip() in ("1", "true", "yes")

_hasher = PasswordHasher()  # argon2id with the library's current defaults
# Verified against when the email is unknown, so a miss costs the same time as a wrong
# password and response timing does not reveal which addresses have accounts.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(16))

PASSWORD_MIN = 12
PASSWORD_MAX = 128  # argon2 hashes whatever it is given; bound it (standard §2)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass
class Principal:
    kind: str                 # "user" | "device"
    org_id: str
    role: str
    credential_id: str        # session id (token hash) or API key id
    user_id: str | None = None
    name: str = ""
    email: str = ""
    color: str = ""
    expires_at: datetime | None = None
    cached_at: float = field(default_factory=time.monotonic)

    @property
    def permissions(self) -> frozenset[str]:
        return PERMISSIONS.get(self.role, frozenset())


_principals: dict[str, Principal] = {}                      # token/key hash -> principal
_sockets: dict[str, weakref.WeakSet] = defaultdict(weakref.WeakSet)  # credential -> open WS


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def client_ip(conn: HTTPConnection) -> str:
    return conn.client.host if conn.client else ""


async def hash_password(password: str) -> str:
    # argon2 is deliberately slow (~50 ms of CPU); off the event loop (standard §1).
    return await asyncio.to_thread(_hasher.hash, password)


async def verify_password(stored: str, password: str) -> bool:
    try:
        return await asyncio.to_thread(_hasher.verify, stored, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def check_password_policy(password: str, email: str) -> None:
    if len(password) < PASSWORD_MIN:
        raise HTTPException(422, f"Use at least {PASSWORD_MIN} characters.")
    if len(password) > PASSWORD_MAX:
        raise HTTPException(422, f"Use at most {PASSWORD_MAX} characters.")
    if password.strip().lower() in (email.lower(), email.split("@")[0].lower()):
        raise HTTPException(422, "The password cannot be your email address.")


def normalize_email(email: str) -> str:
    email = email.strip().lower()
    if not _EMAIL_RE.match(email) or len(email) > 254:
        raise HTTPException(422, "Enter a valid email address.")
    return email


async def _load_session(th: str) -> Principal | None:
    now = db.utcnow()
    async with db.SessionLocal() as s:
        row = (
            await s.execute(
                select(db.AuthSession, db.User, db.Membership)
                .join(db.User, db.User.id == db.AuthSession.user_id)
                .join(
                    db.Membership,
                    (db.Membership.user_id == db.AuthSession.user_id)
                    & (db.Membership.org_id == db.AuthSession.org_id),
                )
                .where(db.AuthSession.id == th)
            )
        ).first()
        if row is None:
            return None
        sess, user, member = row
        if sess.revoked_at is not None or sess.expires_at <= now or not user.is_active:
            return None
        if now - sess.last_seen_at >= TOUCH_EVERY:
            idle = LONG_SESSION if sess.persistent else SHORT_SESSION
            sess.expires_at = min(now + idle, sess.created_at + ABSOLUTE_SESSION)
            sess.last_seen_at = now
            await s.commit()
        return Principal(
            kind="user", org_id=sess.org_id, role=member.role, credential_id=th,
            user_id=user.id, name=user.name, email=user.email, color=user.color,
            expires_at=sess.expires_at,
        )


async def _load_api_key(kh: str) -> Principal | None:
    async with db.SessionLocal() as s:
        key = (await s.execute(select(db.ApiKey).where(db.ApiKey.key_hash == kh))).scalar()
        if key is None or key.revoked_at is not None:
            return None
        key.last_used_at = db.utcnow()
        await s.commit()
        return Principal(kind="device", org_id=key.org_id, role=key.role,
                         credential_id=key.id, name=key.name)


async def resolve_principal(conn: HTTPConnection) -> Principal | None:
    """The caller behind this request, or None. A Bearer API key wins over a cookie."""
    bearer = conn.headers.get("authorization", "")
    if bearer.lower().startswith("bearer "):
        raw, loader = bearer[7:].strip(), _load_api_key
    else:
        raw, loader = conn.cookies.get(COOKIE_NAME, ""), _load_session
    if not raw or len(raw) > 200:
        return None
    h = token_hash(raw)
    cached = _principals.get(h)
    if (
        cached is not None
        and time.monotonic() - cached.cached_at < CACHE_TTL
        and (cached.expires_at is None or cached.expires_at > db.utcnow())
    ):
        return cached
    principal = await loader(h)
    if principal is None:
        _principals.pop(h, None)
    else:
        _principals[h] = principal
    return principal


def _origin_ok(conn: HTTPConnection) -> bool:
    """Cross-site request forgery guard for cookie-authenticated writes and WebSockets.
    SameSite=Strict already keeps the cookie off cross-site requests; this is the second
    lock. A request without an Origin header is not from a browser page, so not CSRF."""
    origin = conn.headers.get("origin")
    if not origin:
        return True
    own = f"{conn.url.scheme.replace('ws', 'http')}://{conn.headers.get('host', '')}"
    allowed = {own} | {o for o in os.environ.get("CORS_ORIGINS", "").split(",") if o and o != "*"}
    return origin in allowed


def _deny(conn: HTTPConnection, status: int, detail: str):
    if conn.scope["type"] == "websocket":
        # 4401/4403 mirror HTTP 401/403, so the page can tell "sign in again" from "not
        # allowed" in the close event, which has no status line.
        raise WebSocketException(code=4000 + status, reason=detail)
    raise HTTPException(status, detail)


async def authorize(conn: HTTPConnection) -> Principal | None:
    """App-wide dependency: resolve the caller and enforce POLICY. Default deny."""
    route = conn.scope.get("route")
    path = getattr(route, "path_format", None) or getattr(route, "path", None)
    method = "WS" if conn.scope["type"] == "websocket" else conn.scope["method"]
    need = POLICY.get(("GET" if method == "HEAD" else method, path))
    if need is None:
        logger.error("auth: no policy for %s %s; denied", method, path)
        _deny(conn, 403, "Not allowed")
    if need == PUBLIC:
        return None
    if (method == "WS" or method not in ("GET", "HEAD")) and not _origin_ok(conn):
        _deny(conn, 403, "Cross-origin request refused")
    principal = await resolve_principal(conn)
    if principal is None:
        _deny(conn, 401, "Sign in to continue")
    if need != AUTHENTICATED and need not in principal.permissions:
        _deny(conn, 403, "Your role does not allow this")
    if method == "WS" and isinstance(conn, WebSocket):
        _sockets[principal.credential_id].add(conn)
    conn.scope["principal"] = principal
    if need in AUDITED:
        await audit(conn, f"{need}:{method} {path}", principal)
    elif need == "operate":
        key, now = (principal.credential_id, f"{method} {path}"), time.monotonic()
        if now - _operate_audited.get(key, float("-inf")) >= OPERATE_AUDIT_EVERY:
            _operate_audited[key] = now
            task = asyncio.create_task(audit(conn, f"operate:{method} {path}", principal))
            _background.add(task)
            task.add_done_callback(_background.discard)
    return principal


async def revoke_credential(credential_id: str, *, close_code: int = 4401) -> None:
    """Forget a session or key in this process and close its open WebSockets now."""
    for h, p in list(_principals.items()):
        if p.credential_id == credential_id:
            _principals.pop(h, None)
    for ws in list(_sockets.pop(credential_id, ())):
        try:
            await ws.close(code=close_code, reason="Signed out")
        except Exception as e:
            logger.info("auth: closing a revoked socket failed: %s", e)


async def revoke_user_sessions(user_id: str) -> int:
    """Sign a person out everywhere, now: their sessions in the database, the validated copies
    in this process, and their open WebSockets. Used when their role or access changes."""
    now = db.utcnow()
    async with db.SessionLocal() as s:
        ids = (await s.execute(
            select(db.AuthSession.id).where(db.AuthSession.user_id == user_id,
                                            db.AuthSession.revoked_at.is_(None))
        )).scalars().all()
        await s.execute(update(db.AuthSession)
                        .where(db.AuthSession.id.in_(ids)).values(revoked_at=now))
        await s.commit()
    for sid in ids:
        await revoke_credential(sid)
    return len(ids)


async def audit(conn: HTTPConnection | None, action: str, principal: Principal | None = None,
                *, target: str = "", detail: dict | None = None,
                org_id: str | None = None) -> None:
    """Append one audit row. Never raises: losing an audit row must not fail the action."""
    try:
        async with db.SessionLocal() as s:
            s.add(db.AuditEvent(
                org_id=org_id or (principal.org_id if principal else None),
                actor_user_id=principal.user_id if principal else None,
                actor_key_id=principal.credential_id if principal and principal.kind == "device" else None,
                action=action[:64], target=target[:255],
                ip=client_ip(conn) if conn else "", detail=detail,
            ))
            await s.commit()
    except Exception as e:
        logger.warning("audit: could not record %s: %s", action, e)


# --------------------------------------------------------------------------- #
# Brute-force throttle (in memory: one gateway process per install)
# --------------------------------------------------------------------------- #
FAIL_WINDOW = 15 * 60.0
MAX_FAILS_PER_ACCOUNT = 5   # per (ip, email) in the window
MAX_FAILS_PER_IP = 20       # per ip, any email, in the window
_fails: dict[tuple[str, str], deque[float]] = defaultdict(deque)


def _recent(key: tuple[str, str]) -> deque[float]:
    q, cutoff = _fails[key], time.monotonic() - FAIL_WINDOW
    while q and q[0] < cutoff:
        q.popleft()
    return q


def _throttled(ip: str, email: str) -> float:
    """Seconds until another attempt is allowed, or 0."""
    for key, limit in (((ip, email), MAX_FAILS_PER_ACCOUNT), ((ip, ""), MAX_FAILS_PER_IP)):
        q = _recent(key)
        if len(q) >= limit:
            return max(1.0, q[0] + FAIL_WINDOW - time.monotonic())
    return 0.0


def _record_fail(ip: str, email: str) -> None:
    now = time.monotonic()
    _fails[(ip, email)].append(now)
    _fails[(ip, "")].append(now)


# --------------------------------------------------------------------------- #
# First-run setup code
# --------------------------------------------------------------------------- #
SETUP_CODE_PATH = db.DATA_DIR / "setup-code"
_SETUP_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I/L
_setup_code: str | None = None
_setup_lock = asyncio.Lock()


async def _account_count() -> int:
    async with db.SessionLocal() as s:
        return (await s.execute(select(func.count()).select_from(db.User))).scalar_one()


def _write_setup_code() -> str:
    code = "-".join("".join(secrets.choice(_SETUP_ALPHABET) for _ in range(4)) for _ in range(2))
    write_secret_file(SETUP_CODE_PATH, code)
    return code


DEVICE_KEY_PATH = db.DATA_DIR / "device-key"


def new_api_key() -> str:
    return "pk_" + secrets.token_urlsafe(32)


def write_secret_file(path, value: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(value + "\n")


async def ensure_device_key() -> None:
    """Give the camera bridge a key if it has none on disk. Needs an organization."""
    if DEVICE_KEY_PATH.exists() and DEVICE_KEY_PATH.stat().st_size > 0:
        return
    async with db.SessionLocal() as s:
        org = (await s.execute(select(db.Organization).limit(1))).scalar()
        if org is None:
            return
        key = new_api_key()
        s.add(db.ApiKey(org_id=org.id, name="camera bridge", role="device",
                        key_hash=token_hash(key), prefix=key[:9]))
        await s.commit()
    await asyncio.to_thread(write_secret_file, DEVICE_KEY_PATH, key)
    logger.info("auth: created the camera bridge's device key in %s", DEVICE_KEY_PATH)


async def init() -> None:
    """Startup: migrate the schema, then arm first-run setup if there are no accounts."""
    global _setup_code
    db.ensure_data_dir()
    await asyncio.to_thread(db.migrate)
    await ensure_device_key()
    if await _account_count() == 0:
        _setup_code = await asyncio.to_thread(_write_setup_code)
        # The path, never the code (standard §2: never log a secret).
        logger.warning("First-run setup: no accounts yet. The setup code is in %s",
                       SETUP_CODE_PATH)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
router = APIRouter(prefix="/api/auth")


class LoginRequest(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=PASSWORD_MAX)
    remember: bool = False


class ProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    color: str | None = Field(default=None, max_length=16)


class PasswordChange(BaseModel):
    current_password: str = Field(max_length=PASSWORD_MAX)
    new_password: str = Field(max_length=PASSWORD_MAX)


class SetupRequest(BaseModel):
    setup_code: str = Field(max_length=32)
    org_name: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(max_length=254)
    password: str = Field(max_length=PASSWORD_MAX)


def _me(principal: Principal, org_name: str) -> dict:
    return {
        "user": {"id": principal.user_id, "name": principal.name, "email": principal.email,
                 "color": principal.color},
        "org": {"id": principal.org_id, "name": org_name},
        "role": principal.role,
        "permissions": sorted(principal.permissions),
    }


async def _org_name(org_id: str) -> str:
    async with db.SessionLocal() as s:
        org = await s.get(db.Organization, org_id)
        return org.name if org else ""


async def _start_session(conn: HTTPConnection, response: Response, user: db.User,
                         org_id: str, role: str, remember: bool) -> Principal:
    token = secrets.token_urlsafe(32)
    now = db.utcnow()
    expires = now + (LONG_SESSION if remember else SHORT_SESSION)
    async with db.SessionLocal() as s:
        s.add(db.AuthSession(
            id=token_hash(token), user_id=user.id, org_id=org_id, persistent=remember,
            created_at=now, last_seen_at=now, expires_at=expires,
            user_agent=conn.headers.get("user-agent", "")[:255], ip=client_ip(conn),
        ))
        await s.execute(update(db.User).where(db.User.id == user.id).values(last_login_at=now))
        await s.commit()
    response.set_cookie(
        COOKIE_NAME, token, httponly=True, samesite="strict", path="/",
        secure=FORCE_SECURE_COOKIE or conn.url.scheme == "https",
        # No max_age without "keep me signed in": the cookie dies with the browser.
        max_age=int(LONG_SESSION.total_seconds()) if remember else None,
    )
    return Principal(kind="user", org_id=org_id, role=role, credential_id=token_hash(token),
                     user_id=user.id, name=user.name, email=user.email, color=user.color,
                     expires_at=expires)


@router.get("/state")
async def auth_state():
    """What the sign-in page needs before anyone is signed in."""
    if await _account_count() == 0:
        return {"setup_required": True, "org_name": None}
    async with db.SessionLocal() as s:
        org = (await s.execute(select(db.Organization).limit(1))).scalar()
    return {"setup_required": False, "org_name": org.name if org else None}


@router.post("/setup")
async def setup(req: SetupRequest, conn: HTTPConnection, response: Response):
    """First run: create the organization and its owner, and sign the owner in."""
    global _setup_code
    ip = client_ip(conn)
    if wait := _throttled(ip, "setup"):
        raise HTTPException(429, "Too many attempts. Try again later.",
                            headers={"Retry-After": str(int(wait))})
    async with _setup_lock:
        if await _account_count() > 0 or _setup_code is None:
            raise HTTPException(409, "This install is already set up. Sign in instead.")
        given = req.setup_code.strip().upper().replace(" ", "")
        if not hmac.compare_digest(given.encode(), _setup_code.encode()):
            _record_fail(ip, "setup")
            await audit(conn, "auth.setup_failed")
            raise HTTPException(403, "That setup code is not right.")
        email = normalize_email(req.email)
        check_password_policy(req.password, email)
        async with db.SessionLocal() as s:
            org = db.Organization(name=req.org_name.strip())
            user = db.User(email=email, name=req.name.strip(),
                           password_hash=await hash_password(req.password))
            s.add_all([org, user])
            await s.flush()
            s.add(db.Membership(user_id=user.id, org_id=org.id, role="owner"))
            await s.commit()
        _setup_code = None
        try:
            SETUP_CODE_PATH.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("setup: could not delete %s: %s", SETUP_CODE_PATH, e)
    await ensure_device_key()
    principal = await _start_session(conn, response, user, org.id, "owner", remember=False)
    await audit(conn, "auth.setup", principal, target=org.name)
    return _me(principal, org.name)


@router.post("/login")
async def login(req: LoginRequest, conn: HTTPConnection, response: Response):
    ip = client_ip(conn)
    email = req.email.strip().lower()
    if wait := _throttled(ip, email):
        raise HTTPException(429, "Too many attempts. Try again in a few minutes.",
                            headers={"Retry-After": str(int(wait))})
    async with db.SessionLocal() as s:
        row = (
            await s.execute(
                select(db.User, db.Membership)
                .join(db.Membership, db.Membership.user_id == db.User.id)
                .where(db.User.email == email)
                .limit(1)
            )
        ).first()
    user, member = row if row else (None, None)
    ok = await verify_password(user.password_hash if user else _DUMMY_HASH, req.password)
    if not (ok and user and user.is_active):
        _record_fail(ip, email)
        await audit(conn, "auth.login_failed", detail={"email": email[:254]})
        # One message for every failure: never say whether the address exists.
        raise HTTPException(401, "Wrong email or password.")
    _fails.pop((ip, email), None)
    principal = await _start_session(conn, response, user, member.org_id, member.role,
                                     req.remember)
    await audit(conn, "auth.login", principal)
    return _me(principal, await _org_name(member.org_id))


@router.post("/logout")
async def logout(conn: HTTPConnection, response: Response):
    principal: Principal = conn.scope["principal"]
    if principal.kind == "user":
        async with db.SessionLocal() as s:
            await s.execute(update(db.AuthSession)
                            .where(db.AuthSession.id == principal.credential_id)
                            .values(revoked_at=db.utcnow()))
            await s.commit()
        await revoke_credential(principal.credential_id)
        await audit(conn, "auth.logout", principal)
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/me")
async def me(conn: HTTPConnection):
    principal: Principal = conn.scope["principal"]
    if principal.kind != "user":
        raise HTTPException(403, "Not a person")
    return _me(principal, await _org_name(principal.org_id))


def _person(conn: HTTPConnection) -> Principal:
    principal: Principal = conn.scope["principal"]
    if principal.kind != "user":
        raise HTTPException(403, "Not a person")
    return principal


def _forget_cached(user_id: str) -> None:
    """Drop this person's validated sessions from the cache, so the next request reads the
    database again (a new name shows everywhere at once). Their sockets stay open."""
    for h, p in list(_principals.items()):
        if p.user_id == user_id:
            _principals.pop(h, None)


@router.patch("/me")
async def update_me(body: ProfileUpdate, conn: HTTPConnection):
    principal = _person(conn)
    values: dict[str, str] = {}
    if body.name is not None:
        if not body.name.strip():
            raise HTTPException(422, "Your name cannot be empty.")
        values["name"] = body.name.strip()
    if body.color is not None:
        if body.color not in db.COLORS:
            raise HTTPException(422, f"Unknown colour {body.color!r}.")
        values["color"] = body.color
    if values:
        async with db.SessionLocal() as s:
            await s.execute(update(db.User).where(db.User.id == principal.user_id).values(**values))
            await s.commit()
        _forget_cached(principal.user_id)
        if "name" in values:  # a colour is not worth a line in the company's activity
            await audit(conn, "auth.profile_changed", principal, target=values["name"])
        principal.name = values.get("name", principal.name)
        principal.color = values.get("color", principal.color)
    return _me(principal, await _org_name(principal.org_id))


@router.post("/password")
async def change_password(body: PasswordChange, conn: HTTPConnection):
    """Change your own password. Proving the current one is what stops a borrowed, unlocked
    browser from taking the account over; every OTHER session of yours is signed out."""
    principal = _person(conn)
    ip, key = client_ip(conn), f"password:{principal.user_id}"
    if wait := _throttled(ip, key):
        raise HTTPException(429, "Too many attempts. Try again in a few minutes.",
                            headers={"Retry-After": str(int(wait))})
    async with db.SessionLocal() as s:
        user = await s.get(db.User, principal.user_id)
    if user is None or not await verify_password(user.password_hash, body.current_password):
        _record_fail(ip, key)
        raise HTTPException(403, "Your current password is not right.")
    check_password_policy(body.new_password, user.email)
    new_hash = await hash_password(body.new_password)
    now = db.utcnow()
    async with db.SessionLocal() as s:
        await s.execute(update(db.User).where(db.User.id == user.id).values(password_hash=new_hash))
        others = (await s.execute(
            select(db.AuthSession.id).where(db.AuthSession.user_id == user.id,
                                            db.AuthSession.id != principal.credential_id,
                                            db.AuthSession.revoked_at.is_(None))
        )).scalars().all()
        await s.execute(update(db.AuthSession).where(db.AuthSession.id.in_(others))
                        .values(revoked_at=now))
        await s.commit()
    for sid in others:
        await revoke_credential(sid)
    _fails.pop((ip, key), None)
    await audit(conn, "auth.password_changed", principal, detail={"signed_out": len(others)})
    return {"ok": True, "signed_out": len(others)}
