"""/api/whep: the H.264 drive view's WebRTC offer, handed to mediamtx through Pilot.

The browser used to post it to mediamtx's own origin, whose self-signed certificate it refused
until someone accepted it by hand. mediamtx is replaced by an httpx MockTransport that records
what reached it; the sign-in is a stand-in (test_auth.py checks the route is not public).
"""
from __future__ import annotations

import httpx
import pytest
from fastapi.requests import HTTPConnection
from fastapi.testclient import TestClient

import app as gateway
import auth

ANA = auth.Principal(kind="user", org_id="test", role="viewer", credential_id="s-ana",
                     user_id="u-ana", name="Ana Ruiz", email="ana@acme.example")
SESSION = "e1982e56-472b-4e7a-b06d-b5724489a592"


@pytest.fixture
def mediamtx(monkeypatch):
    seen: list[httpx.Request] = []
    answer = {"status": 201}

    def handle(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.method == "POST" and answer["status"] == 201:
            return httpx.Response(201, content=b"v=0 answer",
                                  headers={"Location": f"/{req.url.path.split('/')[1]}/whep/{SESSION}"})
        return httpx.Response(answer["status"] if req.method == "POST" else 200)

    monkeypatch.setattr(gateway, "_whep_client",
                        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handle)))

    async def as_ana(conn: HTTPConnection):
        conn.scope["principal"] = ANA
        return ANA

    gateway.app.dependency_overrides[auth.authorize] = as_ana
    with TestClient(gateway.app) as tc:
        yield tc, seen, answer
    gateway.app.dependency_overrides.clear()


def test_the_offer_reaches_the_robots_path_and_the_session_stays_on_pilot(mediamtx):
    tc, seen, _ = mediamtx
    r = tc.post("/api/whep/g1", content=b"v=0 offer", headers={"Content-Type": "application/sdp"})
    assert r.status_code == 201 and r.content == b"v=0 answer"
    assert seen[0].url.path == "/g1/whep" and seen[0].content == b"v=0 offer"
    assert r.headers["location"] == f"/api/whep/g1/{SESSION}"      # never mediamtx's own address
    assert tc.delete(r.headers["location"]).status_code == 200
    assert seen[1].method == "DELETE" and seen[1].url.path == f"/g1/whep/{SESSION}"


def test_the_go2_keeps_its_historical_path(mediamtx):
    tc, seen, _ = mediamtx
    tc.post("/api/whep/go2", content=b"v=0 offer")
    assert seen[0].url.path == "/robot/whep"


def test_only_known_robots_and_session_ids_are_forwarded(mediamtx):
    tc, seen, _ = mediamtx
    assert tc.post("/api/whep/b2", content=b"v=0").status_code == 404
    assert tc.post("/api/whep/go2", content=b"").status_code == 400
    assert tc.delete("/api/whep/go2/..%2F..%2Fsomething").status_code == 404
    assert seen == []


def test_a_robot_not_publishing_is_a_404_the_page_falls_back_on(mediamtx):
    tc, _, answer = mediamtx
    answer["status"] = 404
    assert tc.post("/api/whep/go2", content=b"v=0 offer").status_code == 404
