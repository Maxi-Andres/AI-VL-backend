"""Accounts and access: first-run setup, sign-in, sessions, roles, device keys.

Proven here, against the real auth.authorize (marker `real_auth`):
- every route the app serves has a policy entry, and no entry points at a route that is gone;
- the first-run setup needs the code from the server's disk, and works exactly once;
- the session cookie is httpOnly and SameSite=Strict, and signing out kills it — including a
  WebSocket that is already open;
- a wrong password and an unknown email look the same, and repeated failures are throttled;
- a viewer can watch but not move the robot, over HTTP and over WebSockets;
- the camera bridge's key can push video in and do nothing else;
- a cookie-authenticated write from another origin is refused.

Accounts other than the owner are inserted straight into the database (sync engine): the
invitation flow that creates them through the API is a later phase.
"""
from __future__ import annotations

import hashlib
import secrets
import threading

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketDisconnect

import app as gateway
import auth
import db

pytestmark = pytest.mark.real_auth

OWNER = {"org_name": "Acme Logistics", "name": "Ana Owner", "email": "Ana@Acme.example",
         "password": "correct horse battery"}


@pytest.fixture(scope="module")
def tc():
    with TestClient(gateway.app) as client:
        yield client


@pytest.fixture(autouse=True)
def _fresh_throttle():
    auth._fails.clear()
    auth._principals.clear()


def _owner_signed_in(tc) -> None:
    """Set the install up on first use, then make sure `tc` carries the owner's cookie."""
    if tc.get("/api/auth/state").json()["setup_required"]:
        code = auth.SETUP_CODE_PATH.read_text().strip()
        r = tc.post("/api/auth/setup", json={**OWNER, "setup_code": code})
        assert r.status_code == 200, r.text
    else:
        r = tc.post("/api/auth/login", json={"email": OWNER["email"], "password": OWNER["password"]})
        assert r.status_code == 200, r.text


PASSWORD = "another long password"  # noqa: S105  # a test account's, on a throwaway database


def _add_user(role: str, email: str, password: str = PASSWORD) -> None:
    engine = create_engine(db.sync_url())
    with Session(engine) as s:
        org = s.query(db.Organization).first()
        user = db.User(email=email, name=role.title(), password_hash=auth._hasher.hash(password))
        s.add(user)
        s.flush()
        s.add(db.Membership(user_id=user.id, org_id=org.id, role=role))
        s.commit()
    engine.dispose()


def _add_device_key() -> str:
    key = "pk_" + secrets.token_urlsafe(32)
    engine = create_engine(db.sync_url())
    with Session(engine) as s:
        org = s.query(db.Organization).first()
        s.add(db.ApiKey(org_id=org.id, name="camera bridge", role="device",
                        key_hash=hashlib.sha256(key.encode()).hexdigest(), prefix=key[:10]))
        s.commit()
    engine.dispose()
    return key


def _close_code(tc, path: str, **kw) -> int | None:
    """Open a WebSocket and report the close code it was refused with (None if accepted)."""
    try:
        with tc.websocket_connect(path, **kw):
            return None
    except WebSocketDisconnect as e:
        return e.code


def _routes(routes, prefix: str = ""):
    for r in routes:
        inner = getattr(r, "original_router", None)  # include_router (FastAPI wraps it lazily)
        if inner is not None:
            yield from _routes(inner.routes, prefix + r.include_context.prefix)
        else:
            yield prefix, r


def test_every_route_has_a_policy_and_every_policy_a_route():
    served = set()
    for prefix, r in _routes(gateway.app.routes):
        if isinstance(r, APIWebSocketRoute):
            served.add(("WS", prefix + r.path_format))
        elif isinstance(r, APIRoute):
            served |= {(m, prefix + r.path_format) for m in r.methods if m != "HEAD"}
    missing = served - auth.POLICY.keys()
    assert not missing, f"routes with no auth.POLICY entry (they are refused): {sorted(missing)}"
    stale = auth.POLICY.keys() - served - {("GET", "/{full_path}")}  # SPA: only with FRONTEND_DIST
    assert not stale, f"auth.POLICY entries for routes that no longer exist: {sorted(stale)}"


def test_first_run_setup_needs_the_code_and_works_once(tc):
    if not tc.get("/api/auth/state").json()["setup_required"]:
        pytest.skip("install already set up by an earlier test in this session")
    assert tc.get("/api/robot-frame").status_code == 401        # nobody can do anything yet
    bad = tc.post("/api/auth/setup", json={**OWNER, "setup_code": "AAAA-AAAA"})
    assert bad.status_code == 403
    code = auth.SETUP_CODE_PATH.read_text().strip()
    r = tc.post("/api/auth/setup", json={**OWNER, "setup_code": code})
    assert r.status_code == 200, r.text
    assert r.json()["role"] == "owner" and r.json()["user"]["email"] == "ana@acme.example"
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert not auth.SETUP_CODE_PATH.exists()                     # spent
    assert auth.DEVICE_KEY_PATH.stat().st_mode & 0o777 == 0o600  # the bridge's key, private
    assert tc.get("/api/auth/state").json() == {"setup_required": False,
                                                "org_name": "Acme Logistics"}
    again = tc.post("/api/auth/setup", json={**OWNER, "setup_code": code})
    assert again.status_code == 409


def test_sign_in_and_out(tc):
    _owner_signed_in(tc)
    assert tc.get("/api/auth/me").json()["org"]["name"] == "Acme Logistics"
    assert tc.get("/api/robot-frame").status_code == 503         # through auth, no frame yet
    old = tc.cookies.get(auth.COOKIE_NAME)
    assert tc.post("/api/auth/logout").status_code == 200
    assert tc.get("/api/auth/me").status_code == 401
    tc.cookies.set(auth.COOKIE_NAME, old)                        # replaying the old cookie
    assert tc.get("/api/auth/me").status_code == 401
    tc.cookies.clear()


def test_failures_look_the_same_and_are_throttled(tc):
    _owner_signed_in(tc)
    tc.cookies.clear()
    wrong = tc.post("/api/auth/login", json={"email": OWNER["email"], "password": "nope" * 4})
    unknown = tc.post("/api/auth/login", json={"email": "who@acme.example", "password": "nope" * 4})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()
    for _ in range(auth.MAX_FAILS_PER_ACCOUNT):
        tc.post("/api/auth/login", json={"email": OWNER["email"], "password": "nope" * 4})
    locked = tc.post("/api/auth/login", json={"email": OWNER["email"], "password": OWNER["password"]})
    assert locked.status_code == 429 and int(locked.headers["retry-after"]) > 0


def test_a_viewer_watches_but_cannot_move_the_robot(tc):
    _owner_signed_in(tc)
    _add_user("viewer", "vic@acme.example")
    tc.cookies.clear()
    r = tc.post("/api/auth/login", json={"email": "vic@acme.example",
                                         "password": PASSWORD})
    assert r.status_code == 200 and r.json()["permissions"] == ["view"]
    assert tc.get("/api/robot-frame").status_code == 503         # watching: allowed
    assert tc.post("/api/execute", json={"skill": "stand_up"}).status_code == 403
    assert tc.post("/api/robot-net", json={}).status_code == 403
    assert _close_code(tc, "/ws/joint-control?robot=g1") == 4403
    tc.cookies.clear()
    assert _close_code(tc, "/ws/joint-control?robot=g1") == 4401   # and nobody at all: 4401


def test_the_device_key_pushes_video_and_nothing_else(tc):
    _owner_signed_in(tc)
    key = _add_device_key()
    tc.cookies.clear()
    bearer = {"Authorization": f"Bearer {key}"}
    with tc.websocket_connect("/ws/robot-cam", headers=bearer):
        pass                                                     # accepted
    assert tc.get("/api/robot-frame", headers=bearer).status_code == 403
    assert tc.get("/api/robot-frame", headers={"Authorization": "Bearer pk_forged"}).status_code == 401


def test_a_cross_origin_write_is_refused(tc):
    _owner_signed_in(tc)
    evil = tc.post("/api/robot-audio/stop", headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403
    assert _close_code(tc, "/ws/view", headers={"Origin": "https://evil.example"}) == 4403
    tc.cookies.clear()


def test_signing_out_closes_open_websockets(tc):
    _owner_signed_in(tc)
    with tc.websocket_connect("/ws/view") as ws:
        tc.post("/api/auth/logout")
        box: dict = {}

        def read():  # skip the config greeting; stop at the server's close frame
            while (msg := ws.receive())["type"] != "websocket.close":
                pass
            box["code"] = msg["code"]

        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(3.0)
        assert box.get("code") == 4401, "the open socket survived sign-out"
