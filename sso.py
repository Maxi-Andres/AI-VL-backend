"""sso.py — signing in through the company's identity provider (OpenID Connect).

An owner turns it on in Settings → Single sign-on with three values from the provider (for
Microsoft Entra: the tenant, the application's client ID and a client secret), and the sign-in
page grows a "Sign in with Microsoft" button. Signing in with a password keeps working: it is the
way in when the provider is down or misconfigured (decided 2026-10-27, like Splunk's
?loginType=splunk).

WHO GETS IN: only people Pilot already knows (decided 2026-10-27). The provider proves who the
person is; it never decides that they belong here. A sign-in is matched, in order, to
  1. the account already linked to this provider account (users.sso_subject, "<iss>|<sub>"),
  2. the account with that email — linked on the spot, so a later address change at the provider
     does not lock the person out,
  3. a pending invitation to that email — the account is created with the invitation's role,
     exactly as accepting the link would, but with no password.
Anyone else is sent back to the sign-in page with "ask an admin to invite you".

THE FLOW (authorization code + PKCE, the current OIDC best practice for a server app):
  GET /api/auth/sso/start      -> 302 to the provider, with state, nonce and a PKCE challenge
  provider                     -> 302 to GET /api/auth/sso/callback?code&state
  GET /api/auth/sso/callback   -> code exchanged server to server (with the client secret and
                                  the PKCE verifier); the id_token's signature checked against
                                  the provider's published keys, then issuer, audience, expiry
                                  and nonce; a normal Pilot session cookie; 302 to "/".
The state travels in a short-lived cookie as well as through the provider, and both must match:
a callback that this browser did not start (login CSRF) is refused. That cookie is SameSite=Lax
because the callback is a navigation coming from the provider's site; the session cookie stays
Strict. Failures land on "/?sso_error=<reason>", which the sign-in page turns into a sentence.

THE REDIRECT URI registered at the provider is <public address>/api/auth/sso/callback — the
address set in Settings → Team (organizations.public_url), or, without one, the address the owner
is using. Entra accepts https everywhere and plain http only for localhost, so a Pilot reached as
https://<ip>:8443 needs a real name (a DNS record and a certificate) for anyone but this machine.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re
import secrets
import time
from urllib.parse import urlencode, urlsplit

import httpx
import jwt
from fastapi import APIRouter, HTTPException, Request
from fastapi.requests import HTTPConnection
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, update

import auth
import db

logger = logging.getLogger("uvicorn.error")
router = APIRouter()

# What the sign-in button says, by provider.
PROVIDERS = {"microsoft": "Microsoft", "google": "Google", "oidc": "SSO"}
CALLBACK_PATH = "/api/auth/sso/callback"
STATE_COOKIE = "pilot_sso"
FLOW_TTL = 600            # seconds a started sign-in may take at the provider
MAX_FLOWS = 1000          # /start is public: bound what an anonymous caller can make us keep
CACHE_SECONDS = 3600      # discovery document and signing keys
# Signing algorithms an id_token may use: asymmetric only. Never "none", never an HMAC (with an
# HMAC the client secret would be the key, and a public key could be passed off as one).
ALGORITHMS = ("RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512")
_HTTP_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
# The password hash of an account made through single sign-on: not an argon2 hash, so password
# sign-in can never match it. An admin's reset link gives that person a password if needed.
NO_PASSWORD = "!"  # noqa: S105 - a marker, not a password

_flows: dict[str, dict] = {}                        # token_hash(state) -> the started sign-in
_discovery: dict[str, tuple[float, dict]] = {}      # issuer -> (fetched at, document)
_jwks: dict[str, tuple[float, dict]] = {}           # jwks_uri -> (fetched at, key set)


def http_client() -> httpx.AsyncClient:
    """The client every call to the provider goes through (tests swap in a fake provider)."""
    return httpx.AsyncClient(timeout=_HTTP_TIMEOUT)


class ProviderError(Exception):
    """The provider could not be reached, or answered something unusable."""


# --------------------------------------------------------------------------- #
# The provider
# --------------------------------------------------------------------------- #
_TENANT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,99}")


def issuer_for(provider: str, value: str) -> str:
    """The issuer URL for what an owner typed: a tenant (Microsoft), nothing (Google), or the
    issuer itself (any other OIDC provider)."""
    value = (value or "").strip().rstrip("/")
    if provider == "microsoft":
        if value.lower() in ("", "common", "organizations", "consumers"):
            # Those accept accounts from ANY Microsoft tenant; Pilot is for one company.
            raise HTTPException(422, "Give your Microsoft Entra tenant: its Directory (tenant) ID or its domain.")
        if not _TENANT_RE.fullmatch(value):
            raise HTTPException(422, "That does not look like a tenant ID or a domain.")
        return f"https://login.microsoftonline.com/{value}/v2.0"
    if provider == "google":
        return "https://accounts.google.com"
    if provider == "oidc":
        p = urlsplit(value)
        local = p.hostname in ("localhost", "127.0.0.1", "::1")
        if not p.hostname or p.query or p.fragment or not (
                p.scheme == "https" or (p.scheme == "http" and local)):
            raise HTTPException(422, "Give the issuer as https://… — the address before /.well-known/openid-configuration.")
        return value
    raise HTTPException(422, "Unknown identity provider.")


def tenant_of(issuer: str | None) -> str:
    """The tenant inside a Microsoft issuer, for showing it back in Settings."""
    m = re.fullmatch(r"https://login\.microsoftonline\.com/([^/]+)/v2\.0", issuer or "")
    return m.group(1) if m else ""


async def discovery(provider: str, issuer: str, *, fresh: bool = False) -> dict:
    """The provider's OpenID configuration (endpoints, keys, the exact issuer it signs with)."""
    hit = _discovery.get(issuer)
    if hit and not fresh and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    url = f"{issuer}/.well-known/openid-configuration"
    try:
        async with http_client() as c:
            r = await c.get(url)
        doc = r.json() if r.status_code == 200 else None
    except (httpx.HTTPError, ValueError) as e:
        raise ProviderError(f"could not read {url}: {type(e).__name__}") from e
    if not isinstance(doc, dict):
        raise ProviderError(f"{url} answered {r.status_code}")
    for field_ in ("issuer", "authorization_endpoint", "token_endpoint", "jwks_uri"):
        if not isinstance(doc.get(field_), str):
            raise ProviderError(f"{url} has no {field_}")
    # A Microsoft tenant given by its domain answers with the tenant's ID in `issuer`, and that
    # is what its tokens carry; any other provider must name itself exactly as configured.
    if provider != "microsoft" and doc["issuer"].rstrip("/") != issuer:
        raise ProviderError(f"{url} says its issuer is {doc['issuer']}, not {issuer}")
    _discovery[issuer] = (time.monotonic(), doc)
    return doc


async def _signing_keys(jwks_uri: str, *, fresh: bool = False) -> dict:
    hit = _jwks.get(jwks_uri)
    if hit and not fresh and time.monotonic() - hit[0] < CACHE_SECONDS:
        return hit[1]
    try:
        async with http_client() as c:
            r = await c.get(jwks_uri)
        keys = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise ProviderError(f"could not read the signing keys: {type(e).__name__}") from e
    if r.status_code != 200 or not isinstance(keys, dict) or not isinstance(keys.get("keys"), list):
        raise ProviderError(f"the signing keys answered {r.status_code}")
    _jwks[jwks_uri] = (time.monotonic(), keys)
    return keys


async def verify_id_token(token: str, doc: dict, client_id: str, nonce: str) -> dict:
    """The id_token's claims, after checking its signature, issuer, audience, times and nonce.
    Raises jwt.PyJWTError (or ProviderError) on anything wrong."""
    header = jwt.get_unverified_header(token)
    alg = header.get("alg")
    if alg not in ALGORITHMS:
        raise jwt.InvalidAlgorithmError(f"algorithm {alg!r} is not accepted")
    kid = header.get("kid")

    def pick(keys: dict) -> dict | None:
        found = [k for k in keys["keys"] if kid is None or k.get("kid") == kid]
        return found[0] if found else None

    jwk = pick(await _signing_keys(doc["jwks_uri"]))
    if jwk is None:     # the provider rotated its keys since we cached them
        jwk = pick(await _signing_keys(doc["jwks_uri"], fresh=True))
    if jwk is None:
        raise jwt.InvalidKeyError(f"no published key {kid!r}")
    claims = jwt.decode(
        token, jwt.PyJWK(jwk, algorithm=alg).key, algorithms=[alg], audience=client_id,
        issuer=doc["issuer"], leeway=60,
        options={"require": ["exp", "iat", "iss", "aud", "sub"]})
    if not hmac.compare_digest(str(claims.get("nonce", "")).encode(), nonce.encode()):
        raise jwt.InvalidTokenError("nonce does not match")
    return claims


# --------------------------------------------------------------------------- #
# The sign-in
# --------------------------------------------------------------------------- #
async def _org() -> db.Organization | None:
    async with db.SessionLocal() as s:
        return (await s.execute(select(db.Organization).limit(1))).scalar()


def _base(conn: HTTPConnection) -> str:
    return str(conn.base_url).rstrip("/")


def redirect_uri(conn: HTTPConnection, org: db.Organization) -> str:
    return f"{org.public_url or _base(conn)}{CALLBACK_PATH}"


def _back(reason: str, **extra: str) -> RedirectResponse:
    """To the sign-in page, which says what went wrong."""
    r = RedirectResponse("/?" + urlencode({"sso_error": reason, **extra}), status_code=302)
    r.delete_cookie(STATE_COOKIE, path="/api/auth/sso")
    return r


def _prune() -> None:
    cutoff = time.monotonic() - FLOW_TTL
    for k in [k for k, f in _flows.items() if f["at"] < cutoff]:
        del _flows[k]
    while len(_flows) >= MAX_FLOWS:
        del _flows[next(iter(_flows))]       # oldest first (dicts keep insertion order)


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


@router.get("/api/auth/sso/start")
async def start(request: Request, remember: bool = False):
    org = await _org()
    if org is None or not org.sso_enabled or not org.sso_issuer or not org.sso_client_id:
        return _back("off")
    # The state cookie must be set on the host the provider sends the person back to.
    if org.public_url and urlsplit(org.public_url).netloc != request.url.netloc:
        return RedirectResponse(f"{org.public_url}/api/auth/sso/start?remember={int(remember)}", 302)
    try:
        doc = await discovery(org.sso_provider or "oidc", org.sso_issuer)
    except ProviderError as e:
        logger.warning("sso: %s", e)
        return _back("provider")
    state, nonce, verifier = (secrets.token_urlsafe(32), secrets.token_urlsafe(32),
                              secrets.token_urlsafe(64))
    _prune()
    _flows[auth.token_hash(state)] = {"nonce": nonce, "verifier": verifier, "remember": remember,
                                      "redirect_uri": redirect_uri(request, org),
                                      "at": time.monotonic()}
    params = {
        "client_id": org.sso_client_id, "response_type": "code", "response_mode": "query",
        "redirect_uri": redirect_uri(request, org), "scope": "openid email profile",
        "state": state, "nonce": nonce,
        "code_challenge": _b64url(hashlib.sha256(verifier.encode()).digest()),
        "code_challenge_method": "S256",
    }
    if org.sso_provider == "microsoft":
        params["prompt"] = "select_account"     # a shared console: always ask which account
    r = RedirectResponse(f"{doc['authorization_endpoint']}?{urlencode(params)}", status_code=302)
    r.set_cookie(STATE_COOKIE, state, max_age=FLOW_TTL, path="/api/auth/sso", httponly=True,
                 samesite="lax", secure=auth.FORCE_SECURE_COOKIE or request.url.scheme == "https")
    return r


async def _exchange(doc: dict, org: db.Organization, code: str, flow: dict) -> dict:
    data = {"grant_type": "authorization_code", "code": code,
            "redirect_uri": flow["redirect_uri"], "client_id": org.sso_client_id,
            "client_secret": org.sso_client_secret or "", "code_verifier": flow["verifier"]}
    try:
        async with http_client() as c:
            r = await c.post(doc["token_endpoint"], data=data,
                             headers={"Accept": "application/json"})
        body = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise ProviderError(f"token request failed: {type(e).__name__}") from e
    if r.status_code != 200 or not isinstance(body, dict) or not isinstance(body.get("id_token"), str):
        # The error code only: the provider's description can echo what was sent.
        err = body.get("error") if isinstance(body, dict) else None
        raise ProviderError(f"token endpoint answered {r.status_code} {err or ''}".strip())
    return body


def _email(claims: dict) -> str | None:
    # Entra puts the sign-in name in preferred_username and fills `email` only when it is set up
    # as an optional claim; a provider that says the address is NOT verified is not believed.
    if claims.get("email_verified") is False:
        return None
    raw = claims.get("email") or claims.get("preferred_username") or claims.get("upn")
    if not isinstance(raw, str) or "@" not in raw:
        return None
    return raw.strip().lower()[:254]


@router.get("/api/auth/sso/callback")
async def callback(request: Request, code: str | None = None, state: str | None = None,
                   error: str | None = None):
    flow = _flows.pop(auth.token_hash(state), None) if state else None
    cookie = request.cookies.get(STATE_COOKIE, "")
    if (flow is None or time.monotonic() - flow["at"] > FLOW_TTL
            or not hmac.compare_digest(cookie.encode(), (state or "").encode())):
        await auth.audit(request, "auth.sso_failed", detail={"reason": "state"})
        return _back("expired")
    if error or not code:
        await auth.audit(request, "auth.sso_failed", detail={"reason": error or "no code"})
        return _back("cancelled" if error == "access_denied" else "provider")
    org = await _org()
    if org is None or not org.sso_enabled:
        return _back("off")
    try:
        doc = await discovery(org.sso_provider or "oidc", org.sso_issuer or "")
        tokens = await _exchange(doc, org, code, flow)
        claims = await verify_id_token(tokens["id_token"], doc, org.sso_client_id or "", flow["nonce"])
    except ProviderError as e:
        logger.warning("sso: %s", e)
        await auth.audit(request, "auth.sso_failed", detail={"reason": str(e)[:200]})
        return _back("provider")
    except jwt.PyJWTError as e:
        logger.warning("sso: id_token refused: %s", e)
        await auth.audit(request, "auth.sso_failed", detail={"reason": f"id_token: {e}"[:200]})
        return _back("token")

    email = _email(claims)
    if not email:
        await auth.audit(request, "auth.sso_failed", detail={"reason": "no verified email"})
        return _back("no_email")
    subject = f"{claims['iss']}|{claims['sub']}"[:255]
    name = (claims.get("name") or email.split("@")[0]).strip()[:120] or email
    result = await _match(org, subject, email, name)
    if isinstance(result, str):
        await auth.audit(request, "auth.sso_failed", detail={"reason": result, "email": email})
        return _back(result, email=email)
    user, role, joined = result

    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(STATE_COOKIE, path="/api/auth/sso")
    principal = await auth._start_session(request, response, user, org.id, role, flow["remember"])
    if joined:
        await auth.audit(request, "team.invitation_accepted", principal, target=email,
                         detail={"role": role, "via": "sso"})
    await auth.audit(request, "auth.sso_login", principal,
                     detail={"provider": org.sso_provider})
    return response


async def _match(org: db.Organization, subject: str, email: str, name: str):
    """(user, role, joined now) for this provider account, or the reason it cannot sign in."""
    async with db.SessionLocal() as s:
        user = (await s.execute(select(db.User).where(db.User.sso_subject == subject))).scalar()
        if user is None:
            user = (await s.execute(select(db.User).where(db.User.email == email))).scalar()
            if user is not None:
                if user.sso_subject and user.sso_subject != subject:
                    return "linked_elsewhere"
                user.sso_subject = subject
        if user is not None:
            member = (await s.execute(select(db.Membership).where(
                db.Membership.user_id == user.id, db.Membership.org_id == org.id))).scalar()
            if member is None:
                return "no_account"
            if not user.is_active:
                return "disabled"
            await s.commit()
            return user, member.role, False

        inv = (await s.execute(select(db.Invitation).where(
            db.Invitation.org_id == org.id, db.Invitation.email == email,
            db.Invitation.accepted_at.is_(None), db.Invitation.revoked_at.is_(None),
            db.Invitation.expires_at > db.utcnow(),
        ).order_by(db.Invitation.created_at.desc()).limit(1))).scalar()
        if inv is None:
            return "no_account"
        # Spent in the same transaction that creates the account, as team.accept does.
        spent = (await s.execute(update(db.Invitation).where(
            db.Invitation.id == inv.id, db.Invitation.accepted_at.is_(None),
        ).values(accepted_at=db.utcnow()))).rowcount
        if not spent:
            return "no_account"
        user = db.User(email=email, name=name, password_hash=NO_PASSWORD, sso_subject=subject)
        s.add(user)
        await s.flush()
        s.add(db.Membership(user_id=user.id, org_id=org.id, role=inv.role))
        await s.commit()
        return user, inv.role, True


async def sign_in_option() -> dict | None:
    """For the sign-in page (GET /api/auth/state): the button to show, if any."""
    org = await _org()
    if org is None or not org.sso_enabled:
        return None
    return {"provider": org.sso_provider, "label": PROVIDERS.get(org.sso_provider or "", "SSO")}


# --------------------------------------------------------------------------- #
# Settings → Single sign-on (owners)
# --------------------------------------------------------------------------- #
def _settings(conn: HTTPConnection, org: db.Organization) -> dict:
    return {
        "enabled": bool(org.sso_enabled), "provider": org.sso_provider or "microsoft",
        "tenant": tenant_of(org.sso_issuer) if org.sso_provider == "microsoft" else "",
        "issuer": org.sso_issuer if org.sso_provider == "oidc" else "",
        "client_id": org.sso_client_id or "", "has_secret": bool(org.sso_client_secret),
        "redirect_uri": redirect_uri(conn, org), "public_url": org.public_url,
    }


@router.get("/api/org/sso")
async def get_settings(conn: HTTPConnection):
    me: auth.Principal = conn.scope["principal"]
    async with db.SessionLocal() as s:
        org = await s.get(db.Organization, me.org_id)
    return _settings(conn, org)


class SsoSettings(BaseModel):
    enabled: bool
    provider: str = Field(max_length=16)
    tenant: str = Field(default="", max_length=100)       # Microsoft
    issuer: str = Field(default="", max_length=255)       # any other OIDC provider
    client_id: str = Field(default="", max_length=255)
    # Write-only: empty keeps the saved one.
    client_secret: str = Field(default="", max_length=512)


@router.put("/api/org/sso")
async def put_settings(body: SsoSettings, conn: HTTPConnection):
    me: auth.Principal = conn.scope["principal"]
    if body.provider not in PROVIDERS:
        raise HTTPException(422, "Unknown identity provider.")
    issuer = issuer_for(body.provider, body.tenant if body.provider == "microsoft" else body.issuer)
    client_id = body.client_id.strip()
    async with db.SessionLocal() as s:
        org = await s.get(db.Organization, me.org_id)
        secret = body.client_secret.strip() or org.sso_client_secret or ""
        if body.enabled:
            if not client_id or not secret:
                raise HTTPException(422, "Turning it on needs the client ID and a client secret.")
            try:
                # Check before saving that the provider answers for this tenant/issuer.
                await discovery(body.provider, issuer, fresh=True)
            except ProviderError as e:
                logger.warning("sso: settings check: %s", e)
                raise HTTPException(422, "The identity provider did not answer for that tenant or issuer. Check it and try again.") from e
        org.sso_enabled, org.sso_provider, org.sso_issuer = body.enabled, body.provider, issuer
        org.sso_client_id, org.sso_client_secret = client_id or None, secret or None
        await s.commit()
        await s.refresh(org)
    await auth.audit(conn, "org.sso_changed", me, target=body.provider,
                     detail={"enabled": body.enabled, "secret_changed": bool(body.client_secret.strip())})
    return _settings(conn, org)
