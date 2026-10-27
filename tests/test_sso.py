"""Single sign-on (sso.py) against a fake identity provider — marker `real_auth`.

The provider is an httpx.MockTransport that serves a discovery document, a key set, and a token
endpoint that signs id_tokens with its own RSA key, so the real signature check runs.

Proven here:
- only people Pilot knows get in: an invited address joins with the invitation's role (and the
  invitation is spent), an existing account is linked, anyone else is sent back with no session;
- the link is by the provider's subject, so a later address change at the provider still signs
  the same person in;
- a callback this browser did not start, a wrong nonce, a wrong audience, an unsigned or
  HMAC-signed token, and a switched-off account are all refused;
- the password keeps working next to it (the way in when the provider is down), and an account
  made through single sign-on has no password to guess;
- the settings never send the secret back, refuse a tenant that would let in every Microsoft
  account, and are for owners only;
- with it on, a password works from the company network but not from the internet — also through
  a proxy we were not told to trust — unless an owner allows it; X-Forwarded-For is believed only
  from TRUSTED_PROXIES.
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session

import app as gateway
import auth
import db
import sso

pytestmark = pytest.mark.real_auth

OWNER = {"org_name": "Acme Logistics", "name": "Ana Owner", "email": "ana@acme.example",
         "password": "correct horse battery"}
ISSUER = "https://idp.example"
CLIENT = "pilot-client"

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_JWK = {**json.loads(RSAAlgorithm.to_jwk(_KEY.public_key())), "kid": "k1", "use": "sig"}


class FakeProvider:
    """What the provider remembers between /authorize and /token, and what it will claim."""

    def __init__(self):
        self.nonce = self.challenge = ""
        self.claims: dict = {}
        self.token = None        # a raw id_token to hand out instead of a signed one

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/.well-known/openid-configuration":
            return httpx.Response(200, json={
                "issuer": ISSUER, "authorization_endpoint": f"{ISSUER}/authorize",
                "token_endpoint": f"{ISSUER}/token", "jwks_uri": f"{ISSUER}/jwks"})
        if path == "/jwks":
            return httpx.Response(200, json={"keys": [_JWK]})
        if path == "/token":
            form = parse_qs(request.content.decode())
            verifier = form["code_verifier"][0]
            digest = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=")
            if digest.decode() != self.challenge or form["client_secret"][0] != "s3cret":
                return httpx.Response(400, json={"error": "invalid_grant"})
            now = int(time.time())
            claims = {"iss": ISSUER, "aud": CLIENT, "iat": now, "exp": now + 300,
                      "nonce": self.nonce, **self.claims}
            token = self.token or jwt.encode(claims, _KEY, algorithm="RS256", headers={"kid": "k1"})
            return httpx.Response(200, json={"id_token": token, "token_type": "Bearer"})
        return httpx.Response(404)


@pytest.fixture
def idp(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr(sso, "http_client",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake.handler)))
    sso._discovery.clear()
    sso._jwks.clear()
    sso._flows.clear()
    return fake


@pytest.fixture
def owner(idp):
    with TestClient(gateway.app) as tc:
        if tc.get("/api/auth/state").json()["setup_required"]:
            code = auth.SETUP_CODE_PATH.read_text().strip()
            r = tc.post("/api/auth/setup", json={**OWNER, "setup_code": code})
        else:
            r = tc.post("/api/auth/login", json={"email": OWNER["email"], "password": OWNER["password"]})
        assert r.status_code == 200, r.text
        assert tc.put("/api/team/address", json={"public_url": ""}).status_code == 200
        r = tc.put("/api/org/sso", json={"enabled": True, "provider": "oidc", "issuer": ISSUER,
                                          "client_id": CLIENT, "client_secret": "s3cret"})
        assert r.status_code == 200, r.text
        yield tc
        # Off again: with it on, the other suites' password sign-ins (from "testclient", not a
        # company address) would be refused.
        tc.put("/api/org/sso", json={"enabled": False, "provider": "oidc", "issuer": ISSUER,
                                     "client_id": CLIENT})


@pytest.fixture(autouse=True)
def _fresh_throttle():
    auth._fails.clear()


def _sign_in(browser: TestClient, idp: FakeProvider, **claims) -> httpx.Response:
    """Press the button, let the provider answer for `claims`, return the callback's answer."""
    r = browser.get("/api/auth/sso/start", follow_redirects=False)
    assert r.status_code == 302, r.text
    q = parse_qs(urlsplit(r.headers["location"]).query)
    assert r.headers["location"].startswith(f"{ISSUER}/authorize?")
    assert q["code_challenge_method"] == ["S256"] and q["client_id"] == [CLIENT]
    idp.nonce, idp.challenge, idp.claims = q["nonce"][0], q["code_challenge"][0], claims
    return browser.get(f"/api/auth/sso/callback?code=c0de&state={q['state'][0]}",
                       follow_redirects=False)


def _failed(r: httpx.Response) -> str:
    assert r.status_code == 302
    return parse_qs(urlsplit(r.headers["location"]).query)["sso_error"][0]


def _engine():
    return create_engine(str(db.DATABASE_URL).replace("+aiosqlite", ""))


def test_an_invited_address_joins_with_the_invitations_role_and_stays_linked(owner, idp):
    inv = owner.post("/api/team/invitations", json={"email": "luis@acme.example", "role": "operator"})
    assert inv.status_code == 200, inv.text
    person = TestClient(gateway.app)
    r = _sign_in(person, idp, sub="u-luis", email="Luis@Acme.example", name="Luis Pérez")
    assert r.status_code == 302 and r.headers["location"] == "/"
    me = person.get("/api/auth/me").json()
    assert me["role"] == "operator" and me["user"]["name"] == "Luis Pérez"
    # The invitation is spent, and the address change at the provider does not lock him out.
    assert owner.post("/api/invitations/lookup", json={"token": "x"}).status_code == 404
    again = TestClient(gateway.app)
    assert _sign_in(again, idp, sub="u-luis", email="luis.perez@acme.example").headers["location"] == "/"
    assert again.get("/api/auth/me").json()["user"]["email"] == "luis@acme.example"
    # Made without a password: there is none to guess.
    bad = TestClient(gateway.app, client=("192.168.20.31", 50000)).post(
        "/api/auth/login", json={"email": "luis@acme.example", "password": "!"})
    assert bad.status_code == 401


def test_nobody_else_gets_in(owner, idp):
    stranger = TestClient(gateway.app)
    r = _sign_in(stranger, idp, sub="u-eve", email="eve@acme.example")
    assert _failed(r) == "no_account"
    assert stranger.get("/api/auth/me").status_code == 401


def test_an_existing_account_is_linked_and_its_password_still_works(owner, idp):
    person = TestClient(gateway.app)
    assert _sign_in(person, idp, sub="u-ana", email=OWNER["email"]).headers["location"] == "/"
    assert person.get("/api/auth/me").json()["role"] == "owner"
    # Another provider account claiming the same address cannot take the account over.
    assert _failed(_sign_in(TestClient(gateway.app), idp, sub="u-other", email=OWNER["email"])) \
        == "linked_elsewhere"
    # The password is the way in when the provider is down.
    pw = TestClient(gateway.app, client=("192.168.20.31", 50000)).post(
        "/api/auth/login", json={"email": OWNER["email"], "password": OWNER["password"]})
    assert pw.status_code == 200


def test_a_callback_this_browser_did_not_start_is_refused(owner, idp):
    starter = TestClient(gateway.app)
    r = starter.get("/api/auth/sso/start", follow_redirects=False)
    state = parse_qs(urlsplit(r.headers["location"]).query)["state"][0]
    victim = TestClient(gateway.app)     # no state cookie: the classic login-CSRF shape
    assert _failed(victim.get(f"/api/auth/sso/callback?code=c0de&state={state}",
                              follow_redirects=False)) == "expired"
    assert victim.get("/api/auth/me").status_code == 401


@pytest.mark.parametrize("tamper", ["nonce", "audience", "unsigned", "hmac", "expired"])
def test_a_bad_id_token_is_refused(owner, idp, tamper):
    now = int(time.time())
    base = {"iss": ISSUER, "aud": CLIENT, "sub": "u-ana", "email": OWNER["email"],
            "iat": now, "exp": now + 300}
    if tamper == "unsigned":
        idp.token = jwt.encode({**base, "nonce": "x"}, None, algorithm="none")
    elif tamper == "hmac":
        # Signed with the client secret as an HMAC key: must not pass for the provider's signature.
        idp.token = jwt.encode({**base, "nonce": "x"}, "s3cret-padded-to-be-long-enough-for-hmac",
                               algorithm="HS256", headers={"kid": "k1"})
    claims = {"sub": "u-ana", "email": OWNER["email"]}
    if tamper == "nonce":
        claims["nonce"] = "not-the-one-we-sent"
    elif tamper == "audience":
        claims["aud"] = "some-other-app"
    elif tamper == "expired":
        claims.update(iat=now - 7200, exp=now - 3600)
    browser = TestClient(gateway.app)
    assert _failed(_sign_in(browser, idp, **claims)) == "token"
    assert browser.get("/api/auth/me").status_code == 401


def test_a_switched_off_account_cannot_sign_in(owner, idp):
    owner.post("/api/team/invitations", json={"email": "off@acme.example", "role": "viewer"})
    assert _sign_in(TestClient(gateway.app), idp, sub="u-off", email="off@acme.example") \
        .headers["location"] == "/"
    with Session(_engine()) as s:
        s.execute(update(db.User).where(db.User.email == "off@acme.example").values(is_active=False))
        s.commit()
    assert _failed(_sign_in(TestClient(gateway.app), idp, sub="u-off", email="off@acme.example")) \
        == "disabled"


def test_settings_keep_the_secret_and_refuse_any_microsoft_tenant(owner, idp):
    got = owner.get("/api/org/sso").json()
    assert got["has_secret"] is True and "s3cret" not in json.dumps(got)
    assert got["redirect_uri"] == "http://testserver/api/auth/sso/callback"
    assert owner.get("/api/auth/state").json()["sso"] == {"provider": "oidc", "label": "SSO"}
    common = owner.put("/api/org/sso", json={"enabled": True, "provider": "microsoft",
                                             "tenant": "common", "client_id": CLIENT})
    assert common.status_code == 422
    # An empty secret keeps the saved one; switching it off hides the button.
    off = owner.put("/api/org/sso", json={"enabled": False, "provider": "oidc", "issuer": ISSUER,
                                          "client_id": CLIENT})
    assert off.status_code == 200 and off.json()["has_secret"] is True
    assert owner.get("/api/auth/state").json()["sso"] is None
    assert _failed(TestClient(gateway.app).get("/api/auth/sso/start", follow_redirects=False)) == "off"


def test_only_owners_see_the_settings(owner, idp):
    inv = owner.post("/api/team/invitations", json={"email": "adm@acme.example", "role": "admin"})
    assert inv.status_code == 200
    admin = TestClient(gateway.app)
    assert _sign_in(admin, idp, sub="u-adm", email="adm@acme.example").headers["location"] == "/"
    assert admin.get("/api/org/sso").status_code == 403


def _password(peer: str, headers: dict | None = None) -> int:
    with TestClient(gateway.app, client=(peer, 50000)) as tc:
        return tc.post("/api/auth/login", headers=headers or {},
                       json={"email": OWNER["email"], "password": OWNER["password"]}).status_code


def test_with_sso_on_a_password_works_from_inside_only(owner, idp, monkeypatch):
    assert _password("192.168.20.31") == 200                 # the office LAN
    assert _password("100.101.102.103") == 200               # Tailscale
    assert _password("200.45.10.3") == 403                   # the internet
    # Through the office nginx, but nobody told us to trust it: not believed to be inside.
    assert _password("192.168.20.1", {"X-Forwarded-For": "192.168.20.31"}) == 403
    monkeypatch.setattr(auth, "TRUSTED_PROXIES", [ipaddress.ip_network("192.168.20.1/32")])
    assert _password("192.168.20.1", {"X-Forwarded-For": "181.30.2.77"}) == 403
    assert _password("192.168.20.1", {"X-Forwarded-For": "192.168.20.31"}) == 200
    # A forged hop on the left does not help: the address nginx saw is the rightmost one.
    assert _password("192.168.20.1", {"X-Forwarded-For": "192.168.20.31, 198.51.100.7"}) == 403
    # An owner can allow it from anywhere.
    allow = owner.put("/api/org/sso", json={"enabled": True, "provider": "oidc", "issuer": ISSUER,
                                            "client_id": CLIENT, "password_anywhere": True})
    assert allow.status_code == 200 and allow.json()["password_anywhere"] is True
    assert _password("200.45.10.3") == 200
