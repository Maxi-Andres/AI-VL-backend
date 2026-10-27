"""team.py — the people in an organization: invite them, change their role, switch them off.

Who may call these routes is auth.POLICY ("manage_users": admins and owners). On top of that,
WHAT they may do to whom is decided here, by rank (viewer < operator < admin < owner):

- nobody gives a role above their own, and nobody touches someone ranked above them — so an
  admin manages viewers, operators and other admins, and only an owner manages owners;
- nobody changes their own role or switches themselves off (no locking yourself out);
- a change of role or access signs the person out everywhere at once, open WebSockets
  included: they sign in again and get the new role, never a stale one.

Invitations are links with a random token in the URL FRAGMENT (`/invite#token=…`): browsers
never send a fragment to a server, so the token stays out of every access log and proxy on
the way. The token is stored only as its SHA-256, works once, and expires after INVITE_TTL.
Email delivery comes later (needs SMTP); today the admin copies the link.

Password resets work the same way (`/reset-password#token=…`, RESET_TTL, one use): an admin
makes the link for someone who forgot their password. The old password keeps working until
the new one is set; setting it signs the person out everywhere else. If the only owner forgets
theirs, the way back is on the server: `pilot_admin.py set-password`.
"""
from __future__ import annotations

import asyncio
import secrets
import socket
from datetime import timedelta
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Response
from fastapi.requests import HTTPConnection
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update

import auth
import db

INVITE_TTL = timedelta(days=7)
RESET_TTL = timedelta(hours=24)
RANK = {role: i for i, role in enumerate(auth.ROLES)}  # viewer 0 … owner 3

router = APIRouter()


def _principal(conn: HTTPConnection) -> auth.Principal:
    return conn.scope["principal"]


def _check_rank(actor: auth.Principal, role: str, *, what: str) -> None:
    if role not in RANK:
        raise HTTPException(422, f"Unknown role {role!r}.")
    if RANK[role] > RANK.get(actor.role, -1):
        raise HTTPException(403, f"You cannot {what} a role above your own.")


def _iso(dt) -> str | None:
    return dt.isoformat() if dt else None


# --------------------------------------------------------------------------- #
# The team
# --------------------------------------------------------------------------- #
@router.get("/api/team")
async def team(conn: HTTPConnection):
    me = _principal(conn)
    now = db.utcnow()
    async with db.SessionLocal() as s:
        members = (await s.execute(
            select(db.User, db.Membership)
            .join(db.Membership, db.Membership.user_id == db.User.id)
            .where(db.Membership.org_id == me.org_id)
            .order_by(db.Membership.created_at)
        )).all()
        inviter = db.User.__table__.alias("inviter")
        pending = (await s.execute(
            select(db.Invitation, inviter.c.name)
            .outerjoin(inviter, inviter.c.id == db.Invitation.invited_by)
            .where(db.Invitation.org_id == me.org_id,
                   db.Invitation.accepted_at.is_(None),
                   db.Invitation.revoked_at.is_(None),
                   db.Invitation.expires_at > now)
            .order_by(db.Invitation.created_at.desc())
        )).all()
    return {
        "members": [
            {"id": u.id, "name": u.name, "email": u.email, "role": m.role, "color": u.color,
             "active": u.is_active, "last_login_at": _iso(u.last_login_at),
             "joined_at": _iso(m.created_at), "you": u.id == me.user_id}
            for u, m in members
        ],
        "invitations": [
            {"id": inv.id, "email": inv.email, "role": inv.role, "invited_by": name or "",
             "created_at": _iso(inv.created_at), "expires_at": _iso(inv.expires_at)}
            for inv, name in pending
        ],
    }


# Online means a live session that made a request lately. Sessions note that at most once a
# minute (auth.TOUCH_EVERY) and every open page polls more often, so three minutes is
# "has the app open" without flickering between polls.
ONLINE_WINDOW = timedelta(minutes=3)


@router.get("/api/team/online")
async def online(conn: HTTPConnection):
    """Who in the company has Pilot open right now, out of how many active members. Names and
    colours only, so everyone may see it (the Fleet greeting and its Team figure)."""
    me = _principal(conn)
    now = db.utcnow()
    async with db.SessionLocal() as s:
        people = (await s.execute(
            select(db.User.id, db.User.name, db.User.color)
            .join(db.AuthSession, db.AuthSession.user_id == db.User.id)
            .where(db.AuthSession.org_id == me.org_id,
                   db.AuthSession.revoked_at.is_(None),
                   db.AuthSession.expires_at > now,
                   db.AuthSession.last_seen_at > now - ONLINE_WINDOW,
                   db.User.is_active.is_(True))
            .distinct()
        )).all()
        members = (await s.execute(
            select(func.count()).select_from(db.Membership)
            .join(db.User, db.User.id == db.Membership.user_id)
            .where(db.Membership.org_id == me.org_id, db.User.is_active.is_(True))
        )).scalar_one()
    return {"online": [{"id": i, "name": n, "color": c} for i, n, c in people], "members": members}


# --------------------------------------------------------------------------- #
# The address links point to
# --------------------------------------------------------------------------- #
# Interfaces that are never how a person reaches this machine: containers' bridges and the like.
_NOT_REACHABLE = ("lo", "docker", "br-", "veth", "virbr", "cni", "flannel")


async def _public_url(org_id: str) -> str | None:
    async with db.SessionLocal() as s:
        org = await s.get(db.Organization, org_id)
    return org.public_url if org else None


async def _host_addresses() -> list[tuple[str, str]]:
    """(interface, IPv4) for each address of this machine a person could reach it on."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ip", "-4", "-o", "addr", "show",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=2)
    except (OSError, TimeoutError):
        out = b""
    found = []
    for line in out.decode(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[2] == "inet":
            iface, ip = parts[1], parts[3].split("/")[0]
            if not iface.startswith(_NOT_REACHABLE):
                found.append((iface, ip))
    if not found:  # no `ip` here: the address the default route leaves from (sends nothing)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            try:
                probe.connect(("192.0.2.1", 9))
                found.append(("default", probe.getsockname()[0]))
            except OSError:
                pass
    return found


def _label(iface: str, ip: str) -> str:
    if iface.startswith("tailscale") or ip.startswith("100."):
        return "Tailscale — from anywhere, for people on your tailnet"
    if iface.startswith(("wl", "wlan")):
        return f"Wi-Fi ({iface}) — same network only"
    return f"Local network ({iface}) — same network only"


@router.get("/api/team/address")
async def address(conn: HTTPConnection):
    """The address links point to, and the ones this server can be reached on, with the scheme
    and port of the page asking (what this server is actually serving)."""
    me = _principal(conn)
    scheme = conn.url.scheme.replace("ws", "http")
    port = conn.url.port
    default = (scheme == "https" and port == 443) or (scheme == "http" and port == 80)
    suffix = f":{port}" if port and not default else ""
    candidates = [{"url": f"{scheme}://{ip}{suffix}", "label": _label(iface, ip)}
                  for iface, ip in await _host_addresses()]
    return {"public_url": await _public_url(me.org_id), "candidates": candidates}


class AddressBody(BaseModel):
    public_url: str = Field(max_length=255)


@router.put("/api/team/address")
async def set_address(body: AddressBody, conn: HTTPConnection):
    me = _principal(conn)
    url = body.public_url.strip().rstrip("/")
    if url:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.path or parts.query or parts.fragment:
            raise HTTPException(422, "Give the address as https://host:port — nothing after it.")
    async with db.SessionLocal() as s:
        org = await s.get(db.Organization, me.org_id)
        org.public_url = url or None
        await s.commit()
    await auth.audit(conn, "team.address_changed", me, target=url or "(not set)")
    return {"public_url": url or None}


class InviteRequest(BaseModel):
    email: str = Field(max_length=254)
    role: str = Field(max_length=16)


@router.post("/api/team/invitations")
async def invite(req: InviteRequest, conn: HTTPConnection):
    me = _principal(conn)
    email = auth.normalize_email(req.email)
    _check_rank(me, req.role, what="invite someone with")
    token = secrets.token_urlsafe(32)
    now = db.utcnow()
    async with db.SessionLocal() as s:
        if (await s.execute(select(db.User.id).where(db.User.email == email))).first():
            raise HTTPException(409, f"{email} already has an account.")
        # A new invitation for the same address replaces the old one: one live link per person.
        await s.execute(update(db.Invitation)
                        .where(db.Invitation.org_id == me.org_id, db.Invitation.email == email,
                               db.Invitation.accepted_at.is_(None),
                               db.Invitation.revoked_at.is_(None))
                        .values(revoked_at=now))
        inv = db.Invitation(org_id=me.org_id, email=email, role=req.role,
                            token_hash=auth.token_hash(token), invited_by=me.user_id,
                            created_at=now, expires_at=now + INVITE_TTL)
        s.add(inv)
        await s.commit()
    await auth.audit(conn, "team.invite", me, target=email, detail={"role": req.role})
    return {
        "invitation": {"id": inv.id, "email": email, "role": req.role,
                       "invited_by": me.name, "created_at": _iso(now),
                       "expires_at": _iso(inv.expires_at)},
        # The only time the token leaves the server. The page turns it into a full URL, on
        # the company's address when one is set (see /api/team/address).
        "link_path": f"/invite#token={token}",
        "base_url": await _public_url(me.org_id),
    }


@router.delete("/api/team/invitations/{invitation_id}")
async def revoke_invitation(invitation_id: str, conn: HTTPConnection):
    me = _principal(conn)
    async with db.SessionLocal() as s:
        n = (await s.execute(update(db.Invitation)
                             .where(db.Invitation.id == invitation_id,
                                    db.Invitation.org_id == me.org_id,
                                    db.Invitation.accepted_at.is_(None),
                                    db.Invitation.revoked_at.is_(None))
                             .values(revoked_at=db.utcnow()))).rowcount
        await s.commit()
    if not n:
        raise HTTPException(404, "That invitation is no longer pending.")
    await auth.audit(conn, "team.invite_revoked", me, target=invitation_id)
    return {"ok": True}


class MemberUpdate(BaseModel):
    role: str | None = Field(default=None, max_length=16)
    active: bool | None = None


@router.patch("/api/team/members/{user_id}")
async def update_member(user_id: str, req: MemberUpdate, conn: HTTPConnection):
    me = _principal(conn)
    if user_id == me.user_id:
        raise HTTPException(403, "You cannot change your own role or access.")
    async with db.SessionLocal() as s:
        row = (await s.execute(
            select(db.User, db.Membership)
            .join(db.Membership, db.Membership.user_id == db.User.id)
            .where(db.User.id == user_id, db.Membership.org_id == me.org_id)
        )).first()
        if row is None:
            raise HTTPException(404, "No such member.")
        user, member = row
        _check_rank(me, member.role, what="change someone with")
        changes: dict = {}
        if req.role is not None and req.role != member.role:
            _check_rank(me, req.role, what="give")
            changes["role"] = {"from": member.role, "to": req.role}
            member.role = req.role
        if req.active is not None and req.active != user.is_active:
            changes["active"] = req.active
            user.is_active = req.active
        if not changes:
            return {"ok": True, "signed_out": 0}
        # Never leave the company without an active owner.
        owners = (await s.execute(
            select(func.count()).select_from(db.Membership)
            .join(db.User, db.User.id == db.Membership.user_id)
            .where(db.Membership.org_id == me.org_id, db.Membership.role == "owner",
                   db.User.is_active.is_(True))
        )).scalar_one()
        if owners == 0:
            await s.rollback()
            raise HTTPException(409, "The company needs at least one active owner.")
        await s.commit()
    signed_out = await auth.revoke_user_sessions(user_id)
    await auth.audit(conn, "team.member_changed", me, target=user.email, detail=changes)
    return {"ok": True, "signed_out": signed_out}


# --------------------------------------------------------------------------- #
# Joining through an invitation (public: the token is the credential)
# --------------------------------------------------------------------------- #
class TokenRequest(BaseModel):
    token: str = Field(max_length=100)


class AcceptRequest(TokenRequest):
    name: str = Field(min_length=1, max_length=120)
    password: str = Field(max_length=auth.PASSWORD_MAX)
    remember: bool = False


async def _pending(conn: HTTPConnection, token: str):
    """The live invitation behind a token, with its org and inviter; throttled like sign-in,
    so the token space cannot be walked."""
    ip = auth.client_ip(conn)
    if wait := auth._throttled(ip, "invitation"):
        raise HTTPException(429, "Too many attempts. Try again later.",
                            headers={"Retry-After": str(int(wait))})
    async with db.SessionLocal() as s:
        row = (await s.execute(
            select(db.Invitation, db.Organization, db.User.name)
            .join(db.Organization, db.Organization.id == db.Invitation.org_id)
            .outerjoin(db.User, db.User.id == db.Invitation.invited_by)
            .where(db.Invitation.token_hash == auth.token_hash(token))
        )).first()
    if row is None:
        auth._record_fail(ip, "invitation")
        raise HTTPException(404, "This invitation link is not valid.")
    inv, org, inviter = row
    if inv.accepted_at is not None:
        raise HTTPException(410, "This invitation was already used. Sign in instead.")
    if inv.revoked_at is not None or inv.expires_at <= db.utcnow():
        raise HTTPException(410, "This invitation has expired. Ask for a new one.")
    return inv, org, inviter or ""


@router.post("/api/invitations/lookup")
async def lookup(req: TokenRequest, conn: HTTPConnection):
    inv, org, inviter = await _pending(conn, req.token)
    return {"email": inv.email, "role": inv.role, "org_name": org.name, "invited_by": inviter}


@router.post("/api/invitations/accept")
async def accept(req: AcceptRequest, conn: HTTPConnection, response: Response):
    inv, org, _ = await _pending(conn, req.token)
    auth.check_password_policy(req.password, inv.email)
    password_hash = await auth.hash_password(req.password)
    async with db.SessionLocal() as s:
        if (await s.execute(select(db.User.id).where(db.User.email == inv.email))).first():
            raise HTTPException(409, f"{inv.email} already has an account. Sign in instead.")
        # Spend the invitation in the same transaction that creates the account: two tabs
        # racing on one link cannot both get in.
        spent = (await s.execute(update(db.Invitation)
                                 .where(db.Invitation.id == inv.id,
                                        db.Invitation.accepted_at.is_(None))
                                 .values(accepted_at=db.utcnow()))).rowcount
        if not spent:
            raise HTTPException(410, "This invitation was already used. Sign in instead.")
        user = db.User(email=inv.email, name=req.name.strip(), password_hash=password_hash)
        s.add(user)
        await s.flush()
        s.add(db.Membership(user_id=user.id, org_id=org.id, role=inv.role))
        await s.commit()
    principal = await auth._start_session(conn, response, user, org.id, inv.role, req.remember)
    await auth.audit(conn, "team.invitation_accepted", principal, target=inv.email,
                     detail={"role": inv.role})
    return auth._me(principal, org.name)


# --------------------------------------------------------------------------- #
# Password resets
# --------------------------------------------------------------------------- #
@router.post("/api/team/members/{user_id}/password-reset")
async def make_password_reset(user_id: str, conn: HTTPConnection):
    me = _principal(conn)
    if user_id == me.user_id:
        raise HTTPException(403, "Change your own password while signed in, not with a reset link.")
    token = secrets.token_urlsafe(32)
    now = db.utcnow()
    async with db.SessionLocal() as s:
        row = (await s.execute(
            select(db.User, db.Membership)
            .join(db.Membership, db.Membership.user_id == db.User.id)
            .where(db.User.id == user_id, db.Membership.org_id == me.org_id)
        )).first()
        if row is None:
            raise HTTPException(404, "No such member.")
        user, member = row
        _check_rank(me, member.role, what="reset the password of someone with")
        if not user.is_active:
            raise HTTPException(409, "Enable this member before resetting their password.")
        # One live link per person: a new one cancels the previous.
        await s.execute(update(db.PasswordReset)
                        .where(db.PasswordReset.user_id == user_id,
                               db.PasswordReset.used_at.is_(None),
                               db.PasswordReset.revoked_at.is_(None))
                        .values(revoked_at=now))
        reset = db.PasswordReset(user_id=user_id, token_hash=auth.token_hash(token),
                                 created_by=me.user_id, created_at=now,
                                 expires_at=now + RESET_TTL)
        s.add(reset)
        await s.commit()
    await auth.audit(conn, "team.password_reset_link", me, target=user.email)
    return {"link_path": f"/reset-password#token={token}", "expires_at": _iso(reset.expires_at),
            "base_url": await _public_url(me.org_id)}


async def _live_reset(conn: HTTPConnection, token: str):
    ip = auth.client_ip(conn)
    if wait := auth._throttled(ip, "reset"):
        raise HTTPException(429, "Too many attempts. Try again later.",
                            headers={"Retry-After": str(int(wait))})
    async with db.SessionLocal() as s:
        row = (await s.execute(
            select(db.PasswordReset, db.User, db.Membership, db.Organization)
            .join(db.User, db.User.id == db.PasswordReset.user_id)
            .join(db.Membership, db.Membership.user_id == db.User.id)
            .join(db.Organization, db.Organization.id == db.Membership.org_id)
            .where(db.PasswordReset.token_hash == auth.token_hash(token))
        )).first()
    if row is None:
        auth._record_fail(ip, "reset")
        raise HTTPException(404, "This reset link is not valid.")
    reset, user, member, org = row
    if reset.used_at is not None:
        raise HTTPException(410, "This reset link was already used. Sign in instead.")
    if reset.revoked_at is not None or reset.expires_at <= db.utcnow() or not user.is_active:
        raise HTTPException(410, "This reset link has expired. Ask an admin for a new one.")
    return reset, user, member, org


@router.post("/api/password-resets/lookup")
async def lookup_reset(req: TokenRequest, conn: HTTPConnection):
    _, user, _, org = await _live_reset(conn, req.token)
    return {"email": user.email, "name": user.name, "org_name": org.name}


class CompleteResetRequest(TokenRequest):
    password: str = Field(max_length=auth.PASSWORD_MAX)
    remember: bool = False


@router.post("/api/password-resets/complete")
async def complete_reset(req: CompleteResetRequest, conn: HTTPConnection, response: Response):
    reset, user, member, org = await _live_reset(conn, req.token)
    auth.check_password_policy(req.password, user.email)
    password_hash = await auth.hash_password(req.password)
    async with db.SessionLocal() as s:
        spent = (await s.execute(update(db.PasswordReset)
                                 .where(db.PasswordReset.id == reset.id,
                                        db.PasswordReset.used_at.is_(None))
                                 .values(used_at=db.utcnow()))).rowcount
        if not spent:
            raise HTTPException(410, "This reset link was already used. Sign in instead.")
        await s.execute(update(db.User).where(db.User.id == user.id)
                        .values(password_hash=password_hash))
        await s.commit()
    # Whoever knew the old password is out, everywhere; then this browser signs in.
    await auth.revoke_user_sessions(user.id)
    principal = await auth._start_session(conn, response, user, org.id, member.role, req.remember)
    await auth.audit(conn, "auth.password_reset", principal, target=user.email,
                     detail={"via": "reset link"})
    return auth._me(principal, org.name)
