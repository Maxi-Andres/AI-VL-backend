"""/api/vlm/stream passes the conversation on to iacore, so the assistant follows it.

iacore is replaced by an httpx MockTransport that records the body it got; the sign-in is a
stand-in (test_auth.py checks the route is not public).
"""
from __future__ import annotations

import json

import httpx
import pytest
from fastapi.requests import HTTPConnection
from fastapi.testclient import TestClient

import app as gateway
import auth

ANA = auth.Principal(kind="user", org_id="test", role="operator", credential_id="s-ana",
                     user_id="u-ana", name="Ana Ruiz", email="ana@acme.example")


@pytest.fixture
def iacore(monkeypatch):
    got = []

    def handle(req: httpx.Request) -> httpx.Response:
        got.append(json.loads(req.content))
        return httpx.Response(200, content=b"Marron.")

    async def as_ana(conn: HTTPConnection):
        conn.scope["principal"] = ANA
        return ANA

    gateway.app.dependency_overrides[auth.authorize] = as_ana
    with TestClient(gateway.app) as tc:
        # After start-up: the lifespan opens the real iacore client.
        monkeypatch.setattr(gateway, "client", httpx.AsyncClient(
            base_url="http://iacore.test", transport=httpx.MockTransport(handle)))
        yield tc, got
    gateway.app.dependency_overrides.clear()


def test_the_conversation_reaches_iacore(iacore):
    """Catches: "¿y la silla?" asked with no earlier turns — every question alone."""
    tc, got = iacore
    turns = [{"question": "¿Qué ves?", "answer": "Una mesa."}]
    r = tc.post("/api/vlm/stream", json={"image": "aGk=", "prompt": "¿De qué color?", "history": turns})
    assert r.status_code == 200 and r.text == "Marron."
    assert got[0]["history"] == turns


def test_an_oversized_conversation_is_refused(iacore):
    tc, got = iacore
    turns = [{"question": "q", "answer": "a"}] * 51
    assert tc.post("/api/vlm/stream", json={"prompt": "x", "history": turns}).status_code == 422
    assert got == []
