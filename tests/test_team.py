"""The team: invitations, roles, access — against the real auth.authorize (marker `real_auth`).

Proven here:
- an invitation link works once, carries its token in the URL fragment, and creates the
  account with exactly the role it was issued for;
- revoked, expired and already-used links say so, and an existing address cannot be invited;
- rank rules: nobody gives a role above their own, touches someone above them, or changes
  themselves; a viewer cannot even see the team;
- a change of role or access signs the person out at once, and the next sign-in carries the
  new role (or none, if switched off).

Every person gets their own TestClient (own cookie jar), all on the one app.
"""
from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session

import app as gateway
import auth
import db

pytestmark = pytest.mark.real_auth

OWNER = {"org_name": "Acme Logistics", "name": "Ana Owner", "email": "ana@acme.example",
         "password": "correct horse battery"}
PASSWORD = "a long enough password"  # noqa: S105  # test accounts, throwaway database


@pytest.fixture(scope="module")
def owner():
    with TestClient(gateway.app) as tc:
        if tc.get("/api/auth/state").json()["setup_required"]:
            code = auth.SETUP_CODE_PATH.read_text().strip()
            r = tc.post("/api/auth/setup", json={**OWNER, "setup_code": code})
        else:
            r = tc.post("/api/auth/login", json={"email": OWNER["email"],
                                                 "password": OWNER["password"]})
        assert r.status_code == 200, r.text
        yield tc


@pytest.fixture(autouse=True)
def _fresh_throttle():
    auth._fails.clear()


def _token(link_path: str) -> str:
    assert link_path.startswith("/invite#token="), link_path   # fragment: never sent to a server
    return link_path.split("#token=", 1)[1]


def _join(owner_tc, email: str, role: str) -> TestClient:
    """Invite `email` as `role` and accept it in a fresh browser; return that browser."""
    r = owner_tc.post("/api/team/invitations", json={"email": email, "role": role})
    assert r.status_code == 200, r.text
    person = TestClient(gateway.app)
    a = person.post("/api/invitations/accept",
                    json={"token": _token(r.json()["link_path"]), "name": role.title(),
                          "password": PASSWORD})
    assert a.status_code == 200, a.text
    assert a.json()["role"] == role
    return person


def _member_id(tc, email: str) -> str:
    return next(m["id"] for m in tc.get("/api/team").json()["members"] if m["email"] == email)


def test_an_invitation_link_works_once(owner):
    r = owner.post("/api/team/invitations", json={"email": "Olga@Acme.example", "role": "operator"})
    assert r.status_code == 200, r.text
    token = _token(r.json()["link_path"])
    assert owner.get("/api/team").json()["invitations"][0]["email"] == "olga@acme.example"

    olga = TestClient(gateway.app)
    look = olga.post("/api/invitations/lookup", json={"token": token}).json()
    assert look == {"email": "olga@acme.example", "role": "operator",
                    "org_name": "Acme Logistics", "invited_by": "Ana Owner"}
    weak = olga.post("/api/invitations/accept", json={"token": token, "name": "Olga",
                                                      "password": "short"})
    assert weak.status_code == 422                               # the link is not spent by this
    ok = olga.post("/api/invitations/accept", json={"token": token, "name": "Olga",
                                                    "password": PASSWORD})
    assert ok.status_code == 200 and ok.json()["role"] == "operator"
    assert olga.get("/api/auth/me").json()["user"]["email"] == "olga@acme.example"

    again = TestClient(gateway.app).post("/api/invitations/accept",
                                         json={"token": token, "name": "X", "password": PASSWORD})
    assert again.status_code == 410
    assert owner.post("/api/team/invitations",
                      json={"email": "olga@acme.example", "role": "viewer"}).status_code == 409


def test_revoked_expired_and_forged_links(owner):
    r = owner.post("/api/team/invitations", json={"email": "rev@acme.example", "role": "viewer"})
    inv_id, token = r.json()["invitation"]["id"], _token(r.json()["link_path"])
    assert owner.delete(f"/api/team/invitations/{inv_id}").status_code == 200
    stranger = TestClient(gateway.app)
    assert stranger.post("/api/invitations/lookup", json={"token": token}).status_code == 410

    r = owner.post("/api/team/invitations", json={"email": "old@acme.example", "role": "viewer"})
    token = _token(r.json()["link_path"])
    engine = create_engine(db.sync_url())
    with Session(engine) as s:
        s.execute(update(db.Invitation).where(db.Invitation.email == "old@acme.example")
                  .values(expires_at=db.utcnow() - timedelta(minutes=1)))
        s.commit()
    engine.dispose()
    assert stranger.post("/api/invitations/lookup", json={"token": token}).status_code == 410
    assert stranger.post("/api/invitations/lookup", json={"token": "forged"}).status_code == 404


def test_rank_rules(owner):
    admin = _join(owner, "adam@acme.example", "admin")
    assert admin.post("/api/team/invitations",
                      json={"email": "o2@acme.example", "role": "owner"}).status_code == 403
    owner_id = _member_id(admin, OWNER["email"])
    assert admin.patch(f"/api/team/members/{owner_id}", json={"role": "viewer"}).status_code == 403
    me_id = _member_id(admin, "adam@acme.example")
    assert admin.patch(f"/api/team/members/{me_id}", json={"role": "owner"}).status_code == 403

    viewer = _join(owner, "vera@acme.example", "viewer")
    assert viewer.get("/api/team").status_code == 403


def test_a_viewer_cannot_retune_detection_for_everyone(owner):
    import json

    viewer = _join(owner, "val@acme.example", "viewer")
    before = dict(gateway.hub.config)
    with viewer.websocket_connect("/ws/view") as ws:
        ws.receive_text()                                           # the config greeting
        ws.send_text(json.dumps({"conf": 0.99, "boxes": True}))
    with owner.websocket_connect("/ws/view") as ws:
        state = json.loads(ws.receive_text())["state"]
    assert state["conf"] == before["conf"], "a viewer changed the shared detection config"


def test_a_role_change_signs_the_person_out(owner):
    op = _join(owner, "otto@acme.example", "operator")
    assert op.get("/api/auth/me").status_code == 200
    otto = _member_id(owner, "otto@acme.example")
    r = owner.patch(f"/api/team/members/{otto}", json={"role": "viewer"})
    assert r.status_code == 200 and r.json()["signed_out"] == 1
    assert op.get("/api/auth/me").status_code == 401                # out, at once
    back = op.post("/api/auth/login", json={"email": "otto@acme.example", "password": PASSWORD})
    assert back.json()["role"] == "viewer"
    assert op.post("/api/execute", json={"skill": "stand_up"}).status_code == 403

    assert owner.patch(f"/api/team/members/{otto}", json={"active": False}).status_code == 200
    assert op.get("/api/auth/me").status_code == 401
    shut = op.post("/api/auth/login", json={"email": "otto@acme.example", "password": PASSWORD})
    assert shut.status_code == 401                                  # switched off: no way in


def test_a_password_reset_link(owner):
    op = _join(owner, "pia@acme.example", "operator")
    pia = _member_id(owner, "pia@acme.example")
    r = owner.post(f"/api/team/members/{pia}/password-reset")
    assert r.status_code == 200, r.text
    link = r.json()["link_path"]
    assert link.startswith("/reset-password#token=")             # fragment: never sent to a server
    token = link.split("#token=", 1)[1]

    browser = TestClient(gateway.app)
    look = browser.post("/api/password-resets/lookup", json={"token": token}).json()
    assert look["email"] == "pia@acme.example" and look["org_name"] == "Acme Logistics"
    done = browser.post("/api/password-resets/complete",
                        json={"token": token, "password": "a brand new password"})
    assert done.status_code == 200 and done.json()["role"] == "operator"
    assert op.get("/api/auth/me").status_code == 401                # the old session is out
    old = TestClient(gateway.app).post("/api/auth/login",
                                       json={"email": "pia@acme.example", "password": PASSWORD})
    assert old.status_code == 401                                   # the old password too
    again = TestClient(gateway.app).post("/api/password-resets/complete",
                                         json={"token": token, "password": "yet another password"})
    assert again.status_code == 410                                 # one use


def test_who_may_make_a_reset_link(owner):
    admin = _join(owner, "abe@acme.example", "admin")
    owner_id = _member_id(admin, OWNER["email"])
    assert admin.post(f"/api/team/members/{owner_id}/password-reset").status_code == 403
    me_id = _member_id(admin, "abe@acme.example")
    assert admin.post(f"/api/team/members/{me_id}/password-reset").status_code == 403
    viewer = _join(owner, "vik@acme.example", "viewer")
    vik = _member_id(owner, "vik@acme.example")
    assert viewer.post(f"/api/team/members/{vik}/password-reset").status_code == 403


def test_who_is_online(owner):
    viewer = _join(owner, "omar@acme.example", "viewer")
    omar = viewer.get("/api/auth/me").json()["user"]["id"]
    seen = viewer.get("/api/team/online").json()                    # everyone may ask
    ids = {p["id"] for p in seen["online"]}
    assert omar in ids and "Ana Owner" in {p["name"] for p in seen["online"]}
    assert seen["members"] >= 2
    assert all(set(p) == {"id", "name", "color"} for p in seen["online"])   # no emails
    # Gone quiet for longer than the window: no longer online, still a member.
    engine = create_engine(db.sync_url())
    with Session(engine) as s:
        s.execute(update(db.AuthSession).where(db.AuthSession.user_id == omar)
                  .values(last_seen_at=db.utcnow() - timedelta(minutes=10)))
        s.commit()
    engine.dispose()
    later = owner.get("/api/team/online").json()
    assert omar not in {p["id"] for p in later["online"]} and later["members"] == seen["members"]
