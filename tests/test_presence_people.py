"""/api/presence `web.people`: the people behind the browsers on the live session.

The header lists everyone connected through the web and what they are doing, so the list must
count PEOPLE, not sockets — the same person with the Live page open on two tabs is one person —
and it must never list a machine: a device key (the camera bridge) is not someone online.

Each socket picks who it is with a test header, read by a stand-in for `auth.authorize`; the
real sign-in is tested in test_auth.py. The sidecar probes are stubbed, so nothing leaves the
process.
"""
from __future__ import annotations

from contextlib import ExitStack

import pytest
from fastapi.requests import HTTPConnection
from fastapi.testclient import TestClient

import app as gateway
import auth

ANA = auth.Principal(kind="user", org_id="test", role="owner", credential_id="s-ana-1",
                     user_id="u-ana", name="Ana Ruiz", email="ana@acme.example", color="teal")
BOB = auth.Principal(kind="user", org_id="test", role="owner", credential_id="s-bob",
                     user_id="u-bob", name="Bob Stone", email="bob@acme.example")
BRIDGE = auth.Principal(kind="device", org_id="test", role="owner", credential_id="k-bridge",
                        name="camera bridge")
WHO = {"ana": ANA, "bob": BOB, "bridge": BRIDGE}


@pytest.fixture
def tc(monkeypatch):
    async def as_header(conn: HTTPConnection):
        p = WHO[conn.headers.get("x-test-as", "ana")]
        conn.scope["principal"] = p
        return p

    async def no_sidecar(_url: str):
        return None

    async def no_iacore() -> bool:
        return False

    monkeypatch.setattr(gateway, "_probe_json", no_sidecar)
    monkeypatch.setattr(gateway, "_probe_iacore", no_iacore)
    monkeypatch.setattr(gateway, "_presence_cache", (0.0, {}))
    gateway.app.dependency_overrides[auth.authorize] = as_header
    with TestClient(gateway.app) as client:
        yield client


def join(stack: ExitStack, tc: TestClient, path: str, who: str):
    ws = stack.enter_context(tc.websocket_connect(path, headers={"x-test-as": who}))
    # The config seed is sent right after the hub registers the socket: once it is read, the
    # socket is on the hub and the next presence round sees it.
    assert ws.receive_json()["type"] == "config"
    return ws


def people(tc: TestClient) -> list[dict]:
    gateway._presence_cache = (0.0, {})          # skip the TTL: each read is a fresh round
    return tc.get("/api/presence").json()["web"]["people"]


def test_one_person_on_two_tabs_is_listed_once(tc):
    """Catches: the list growing a duplicate row for every tab someone has open."""
    with ExitStack() as stack:
        join(stack, tc, "/ws/view", "ana")
        join(stack, tc, "/ws/detect", "ana")
        join(stack, tc, "/ws/view", "bob")
        listed = people(tc)
        web = tc.get("/api/presence").json()["web"]

    assert listed == [
        {"user_id": "u-ana", "name": "Ana Ruiz", "color": "teal",
         "streaming": True, "watching": True, "sockets": 2},
        {"user_id": "u-bob", "name": "Bob Stone", "color": None,
         "streaming": False, "watching": True, "sockets": 1},
    ]
    # The browser counts other code reads are unchanged: still sockets, not people.
    assert (web["producers"], web["viewers"], web["total"]) == (1, 2, 3)


def test_a_device_key_socket_is_not_a_person(tc):
    """Catches: the camera bridge showing up in the header as someone online."""
    with ExitStack() as stack:
        join(stack, tc, "/ws/view", "bridge")
        join(stack, tc, "/ws/view", "bob")
        listed = people(tc)

    assert [p["user_id"] for p in listed] == ["u-bob"]


def test_nothing_secret_is_listed(tc):
    """Catches: an email or a session id leaking to every viewer through presence."""
    with ExitStack() as stack:
        join(stack, tc, "/ws/view", "ana")
        listed = people(tc)

    assert set(listed[0]) == {"user_id", "name", "color", "streaming", "watching", "sockets"}
    assert "ana@acme.example" not in str(listed) and "s-ana-1" not in str(listed)


def test_a_person_who_leaves_is_gone(tc):
    with ExitStack() as stack:
        join(stack, tc, "/ws/view", "ana")
        assert len(people(tc)) == 1
    assert people(tc) == []
