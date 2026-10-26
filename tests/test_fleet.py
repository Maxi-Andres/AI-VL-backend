"""Sites and robots, against the real auth.authorize (marker `real_auth`).

Proven here: an admin builds the fleet and anyone signed in can see it; a viewer cannot
change it; one robot per stack slot; deleting a site keeps its robots, unassigned; the floor
plan is accepted by its bytes, not its claimed type, and is bounded; the satellite layer exists
only with a key.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app as gateway
import auth
import fleet

pytestmark = pytest.mark.real_auth

OWNER = {"org_name": "Acme Logistics", "name": "Ana Owner", "email": "ana@acme.example",
         "password": "correct horse battery"}
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


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


def test_build_a_site_with_robots(owner):
    site = owner.post("/api/sites", json={"name": "Warehouse B", "timezone": "America/Argentina/Buenos_Aires",
                                          "address": "Ruta 8 km 50", "lat": -34.45, "lon": -58.9})
    assert site.status_code == 200, site.text
    sid = site.json()["id"]
    assert owner.post("/api/sites", json={"name": "X", "timezone": "Mars/Base"}).status_code == 422
    alias = owner.post("/api/sites", json={"name": "Alias", "timezone": "America/Buenos_Aires"})
    assert alias.status_code == 200                                 # what Chrome reports
    owner.delete(f"/api/sites/{alias.json()['id']}")
    assert owner.post("/api/sites", json={"name": "X", "lat": 1.0}).status_code == 422

    go2 = owner.post("/api/robots", json={"name": "Go2 · Unit 01", "slot": "go2",
                                          "serial": "B42", "site_id": sid})
    assert go2.status_code == 200 and go2.json()["model"] == "Unitree Go2"
    twin = owner.post("/api/robots", json={"name": "Go2 · Unit 02", "slot": "go2"})
    assert twin.status_code == 409                                  # one per slot, for now

    f = owner.get("/api/fleet").json()
    assert [s["name"] for s in f["sites"]] == ["Warehouse B"]
    assert {s["slot"]: s["free"] for s in f["slots"]} == {"go2": False, "g1": True}

    assert owner.delete(f"/api/sites/{sid}").status_code == 200
    f = owner.get("/api/fleet").json()
    assert f["sites"] == [] and f["robots"][0]["site_id"] is None   # the robot stays
    assert owner.delete(f"/api/robots/{go2.json()['id']}").status_code == 200


def test_satellite_only_with_a_key(owner, monkeypatch, tmp_path):
    monkeypatch.delenv("ESRI_API_KEY", raising=False)
    monkeypatch.setattr(fleet, "ESRI_KEY_PATH", tmp_path / "esri-api-key")
    assert owner.get("/api/fleet").json()["satellite"] is None
    (tmp_path / "esri-api-key").write_text("from-file\n")         # no restart needed
    assert owner.get("/api/fleet").json()["satellite"].endswith("?token=from-file")
    monkeypatch.setenv("ESRI_API_KEY", "AAPK a/b")                  # the env wins, escaped
    assert owner.get("/api/fleet").json()["satellite"].endswith("?token=AAPK%20a%2Fb")


def test_the_floor_plan(owner, monkeypatch):
    sid = owner.post("/api/sites", json={"name": "Plant"}).json()["id"]
    lie = owner.put(f"/api/sites/{sid}/plan", content=b"<svg onload=alert(1)>",
                    headers={"Content-Type": "image/png"})
    assert lie.status_code == 415                                   # judged by its bytes
    ok = owner.put(f"/api/sites/{sid}/plan", content=PNG, headers={"Content-Type": "image/png"})
    assert ok.status_code == 200 and ok.json()["has_plan"]
    got = owner.get(f"/api/sites/{sid}/plan")
    assert got.status_code == 200 and got.content == PNG and got.headers["content-type"] == "image/png"
    monkeypatch.setattr(fleet, "PLAN_MAX", 32)
    assert owner.put(f"/api/sites/{sid}/plan", content=PNG).status_code == 413
    assert owner.delete(f"/api/sites/{sid}/plan").json()["has_plan"] is False
    assert owner.get(f"/api/sites/{sid}/plan").status_code == 404


def test_a_viewer_sees_but_cannot_change(owner):
    r = owner.post("/api/team/invitations", json={"email": "fleetview@acme.example", "role": "viewer"})
    token = r.json()["link_path"].split("#token=", 1)[1]
    viewer = TestClient(gateway.app)
    assert viewer.post("/api/invitations/accept", json={"token": token, "name": "V",
                                                        "password": "a long enough password"}).status_code == 200
    assert viewer.get("/api/fleet").status_code == 200
    assert viewer.post("/api/sites", json={"name": "Nope"}).status_code == 403
    assert viewer.post("/api/robots", json={"name": "Nope", "slot": "g1"}).status_code == 403


def test_the_activity_feed(owner):
    owner.post("/api/sites", json={"name": "Feed Site"})
    feed = owner.get("/api/activity").json()
    assert feed[0]["action"] == "fleet.site_created" and feed[0]["target"] == "Feed Site"
    assert feed[0]["actor"] == "Ana Owner"
    assert all(e["action"] != "auth.login" for e in feed)          # sign-ins stay out of it
    first, second = (owner.get("/api/activity", params={"limit": 1, "offset": o}).json() for o in (0, 1))
    assert len(first) == len(second) == 1 and first[0]["id"] == feed[0]["id"] != second[0]["id"]
    assert second[0]["id"] == feed[1]["id"]                          # pages follow on
