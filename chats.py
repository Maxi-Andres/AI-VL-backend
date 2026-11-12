"""chats.py — the conversations with the vision assistant (Live) and with the agent (the Agent
page), kept per person.

A chat belongs to the person who had it, and only to them: every route here looks a chat up
by its id AND the caller's user id, so another person's chat — whatever their role — answers
404, the same as one that does not exist. Who may use the assistant at all is auth.POLICY
("analyze").

The page asks the model; this only keeps what was said. A chat is made when its first message
arrives, titled after that first question; deleting one deletes its messages (cascade).
Lengths are bounded (standard §2): an answer the model ran on is cut, never refused.

A chat has a kind, set by its first message: "vlm" (questions and scans for objects, Live) or
"agent" (instructions, the Agent page, since 2026-11-12). Each page lists only its own kind,
and a message of the other kind is refused (422). An agent's message carries the trace the
page drew (`trace`, opaque JSON); one too large to keep is dropped, the instruction and the
reply are still kept.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.requests import HTTPConnection
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select

import auth
import db

router = APIRouter()

MAX_CHATS = 200       # listed per person, newest first
KINDS = {"vlm": ("question", "analysis"), "agent": ("agent",)}    # chat kind -> message kinds
MAX_TRACE = 200_000   # characters of an agent's trace, as JSON


def _me(conn: HTTPConnection) -> auth.Principal:
    me: auth.Principal = conn.scope["principal"]
    if not me.user_id:
        raise HTTPException(403, "Not a person")
    return me


async def _own(s, me: auth.Principal, chat_id: str) -> db.Chat:
    chat = await s.get(db.Chat, chat_id)
    if chat is None or chat.user_id != me.user_id or chat.org_id != me.org_id:
        raise HTTPException(404, "No such chat.")
    return chat


def _chat_json(chat: db.Chat, count: int | None = None) -> dict:
    out = {"id": chat.id, "title": chat.title, "kind": chat.kind,
           "created_at": chat.created_at.isoformat(), "updated_at": chat.updated_at.isoformat()}
    if count is not None:
        out["messages"] = count
    return out


def _title(question: str, kind: str) -> str:
    text = " ".join(question.split())
    if kind == "analysis":
        return f"Find objects · {text}"[:120]
    return text[:80] + ("…" if len(text) > 80 else "")


def _chat_kind(message_kind: str) -> str:
    for chat_kind, kinds in KINDS.items():
        if message_kind in kinds:
            return chat_kind
    raise HTTPException(422, f"Unknown kind {message_kind!r}.")


def _trace(raw: str | None):
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


@router.get("/api/chats")
async def list_chats(conn: HTTPConnection, kind: str = "vlm"):
    """The caller's chats of one kind ("vlm": Live's, the default; "agent")."""
    me = _me(conn)
    async with db.SessionLocal() as s:
        rows = (await s.execute(
            select(db.Chat, func.count(db.ChatMessage.id))
            .outerjoin(db.ChatMessage, db.ChatMessage.chat_id == db.Chat.id)
            .where(db.Chat.user_id == me.user_id, db.Chat.org_id == me.org_id,
                   db.Chat.kind == kind)
            .group_by(db.Chat.id)
            .order_by(db.Chat.updated_at.desc())
            .limit(MAX_CHATS)
        )).all()
    return [_chat_json(c, n) for c, n in rows]


@router.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str, conn: HTTPConnection):
    me = _me(conn)
    async with db.SessionLocal() as s:
        chat = await _own(s, me, chat_id)
        msgs = (await s.execute(
            select(db.ChatMessage).where(db.ChatMessage.chat_id == chat.id)
            .order_by(db.ChatMessage.id)
        )).scalars().all()
    return {**_chat_json(chat), "messages": [
        {"id": m.id, "at": m.at.isoformat(), "kind": m.kind, "question": m.question,
         "answer": m.answer, "status": m.status, "robot": m.robot, "trace": _trace(m.trace)}
        for m in msgs]}


class MessageBody(BaseModel):
    kind: str = Field(max_length=16)
    question: str = Field(max_length=4000)
    answer: str = ""
    status: str = ""
    robot: str = Field(default="", max_length=16)
    trace: dict | None = None      # an agent's exchange: the trace as the page drew it


@router.post("/api/chats/messages")
async def add_to_new_chat(body: MessageBody, conn: HTTPConnection):
    """The first message of a new chat: makes the chat, titled after it."""
    return await _add(None, body, conn)


@router.post("/api/chats/{chat_id}/messages")
async def add_message(chat_id: str, body: MessageBody, conn: HTTPConnection):
    return await _add(chat_id, body, conn)


async def _add(chat_id: str | None, body: MessageBody, conn: HTTPConnection) -> dict:
    me = _me(conn)
    kind = _chat_kind(body.kind)
    trace = json.dumps(body.trace, ensure_ascii=False) if body.trace is not None else None
    if trace is not None and len(trace) > MAX_TRACE:
        trace = None                    # too large to keep: the instruction and reply still are
    now = db.utcnow()
    async with db.SessionLocal() as s:
        if chat_id is None:
            chat = db.Chat(org_id=me.org_id, user_id=me.user_id, kind=kind,
                           title=_title(body.question, body.kind))
            s.add(chat)
            await s.flush()
        else:
            chat = await _own(s, me, chat_id)
            if chat.kind != kind:
                raise HTTPException(422, f"A {body.kind!r} message does not go in a {chat.kind} chat.")
        s.add(db.ChatMessage(chat_id=chat.id, at=now, kind=body.kind, question=body.question,
                             answer=body.answer[:20000], status=body.status[:300], robot=body.robot,
                             trace=trace))
        chat.updated_at = now
        await s.commit()
        return _chat_json(chat)


class ChatUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=120)


@router.patch("/api/chats/{chat_id}")
async def rename_chat(chat_id: str, body: ChatUpdate, conn: HTTPConnection):
    me = _me(conn)
    async with db.SessionLocal() as s:
        chat = await _own(s, me, chat_id)
        chat.title = body.title.strip() or chat.title
        await s.commit()
        return _chat_json(chat)


@router.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str, conn: HTTPConnection):
    me = _me(conn)
    async with db.SessionLocal() as s:
        chat = await _own(s, me, chat_id)
        # Explicitly, not only by the foreign key: SQLite enforces ON DELETE only with its
        # foreign_keys pragma, and a stray message must never outlive its chat.
        await s.execute(delete(db.ChatMessage).where(db.ChatMessage.chat_id == chat.id))
        await s.delete(chat)
        await s.commit()
    return {"ok": True}
