"""The assistant's conversations, against the real auth.authorize (marker `real_auth`).

Proven here: a chat is made by its first message and titled after it; messages come back in
order; another person — even an owner — cannot see, add to or delete it (404, as if it did
not exist); deleting a chat takes its messages with it; a viewer, who may not use the
assistant, gets 403.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

import app as gateway
import auth
import db

pytestmark = pytest.mark.real_auth

OWNER = {"org_name": "Acme Logistics", "name": "Ana Owner", "email": "ana@acme.example",
         "password": "correct horse battery"}
PASSWORD = "another long password"  # noqa: S105  # a test account's, on a throwaway database


@pytest.fixture(scope="module")
def owner():
    with TestClient(gateway.app) as tc:
        if tc.get("/api/auth/state").json()["setup_required"]:
            r = tc.post("/api/auth/setup", json={**OWNER, "setup_code": auth.SETUP_CODE_PATH.read_text().strip()})
        else:
            r = tc.post("/api/auth/login", json={"email": OWNER["email"], "password": OWNER["password"]})
        assert r.status_code == 200, r.text
        yield tc


def _person(owner_tc, email: str, role: str) -> TestClient:
    r = owner_tc.post("/api/team/invitations", json={"email": email, "role": role})
    assert r.status_code == 200, r.text
    token = r.json()["link_path"].split("token=")[1]
    tc = TestClient(gateway.app)
    a = tc.post("/api/invitations/accept", json={"token": token, "name": role.title(), "password": PASSWORD})
    assert a.status_code == 200, a.text
    return tc


def test_a_chat_is_private_and_deletes_whole(owner):
    op = _person(owner, "chat-op@acme.example", "operator")
    first = op.post("/api/chats/messages", json={"kind": "question", "question": "What do you see?",
                                                 "answer": "A green screen.", "robot": "go2"})
    assert first.status_code == 200, first.text
    cid = first.json()["id"]
    assert first.json()["title"] == "What do you see?"
    op.post(f"/api/chats/{cid}/messages", json={"kind": "analysis", "question": "all · people",
                                                "answer": '{"objects": []}', "status": "0.8 s"})
    chat = op.get(f"/api/chats/{cid}").json()
    assert [m["kind"] for m in chat["messages"]] == ["question", "analysis"]
    assert op.get("/api/chats").json()[0] == {**op.get("/api/chats").json()[0], "id": cid, "messages": 2}

    # Someone else — the owner, even — does not see it at all.
    assert owner.get(f"/api/chats/{cid}").status_code == 404
    assert owner.post(f"/api/chats/{cid}/messages", json={"kind": "question", "question": "x"}).status_code == 404
    assert owner.delete(f"/api/chats/{cid}").status_code == 404
    assert cid not in {c["id"] for c in owner.get("/api/chats").json()}

    assert op.post(f"/api/chats/{cid}/messages", json={"kind": "shout", "question": "x"}).status_code == 422
    assert op.patch(f"/api/chats/{cid}", json={"title": "Warehouse check"}).json()["title"] == "Warehouse check"
    assert op.delete(f"/api/chats/{cid}").status_code == 200
    assert op.get(f"/api/chats/{cid}").status_code == 404
    engine = create_engine(db.sync_url())
    with Session(engine) as s:
        left = s.execute(select(func.count()).select_from(db.ChatMessage)
                         .where(db.ChatMessage.chat_id == cid)).scalar_one()
    engine.dispose()
    assert left == 0                                                 # no message outlives it


def test_a_viewer_has_no_assistant(owner):
    viewer = _person(owner, "chat-viewer@acme.example", "viewer")
    assert viewer.get("/api/chats").status_code == 403


def test_agent_chats_are_their_own(owner):
    """The Agent page's conversations: listed apart from Live's, never mixed, trace kept."""
    op = _person(owner, "chat-agent@acme.example", "operator")
    trace = {"instruction": "pick up the bottle", "plan": ["look", "pick"], "items": []}
    first = op.post("/api/chats/messages", json={"kind": "agent", "question": "pick up the bottle",
                                                 "answer": "Done.", "robot": "g1", "trace": trace})
    assert first.status_code == 200, first.text
    cid = first.json()["id"]
    assert first.json()["kind"] == "agent"
    vlm = op.post("/api/chats/messages", json={"kind": "question", "question": "What do you see?"})
    assert vlm.json()["kind"] == "vlm"

    assert [c["id"] for c in op.get("/api/chats", params={"kind": "agent"}).json()] == [cid]
    assert cid not in {c["id"] for c in op.get("/api/chats").json()}             # Live's list
    assert op.get(f"/api/chats/{cid}").json()["messages"][0]["trace"] == trace

    # Never mixed: a question does not go in an agent chat, nor an instruction in a Live one.
    assert op.post(f"/api/chats/{cid}/messages", json={"kind": "question", "question": "x"}).status_code == 422
    assert op.post(f"/api/chats/{vlm.json()['id']}/messages",
                   json={"kind": "agent", "question": "x"}).status_code == 422

    # A trace too large to keep is dropped; the instruction and the reply are not.
    big = {"items": ["x" * 1000] * 300}
    r = op.post(f"/api/chats/{cid}/messages", json={"kind": "agent", "question": "wave",
                                                    "answer": "Waved.", "trace": big})
    assert r.status_code == 200
    last = op.get(f"/api/chats/{cid}").json()["messages"][-1]
    assert (last["question"], last["answer"], last["trace"]) == ("wave", "Waved.", None)
