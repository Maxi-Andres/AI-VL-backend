"""memory.py — each robot's memory: the pictures and notes it was asked to keep.

Said to the assistant, by voice or typed: "guardá esta imagen" / "save this picture" keeps the
frame the question was about; "acordate que la llave está en el cajón" / "remember that…"
keeps a note. The gateway does it itself — no model is needed to store a file — and answers
in the language it was asked. The notes go back to the assistant with every question
(robot_notes), so "¿dónde está la llave?" can be answered; iacore uses them only for talk,
never on a reading.

A memory belongs to the ROBOT, not to whoever asked: the whole company sees it on the Memory
page (auth.POLICY "view"), operators add and delete ("analyze"). Pictures are files in
PILOT_DATA_DIR/memory/<id>, checked to be a JPEG or PNG and bounded in size; deleting the robot
deletes its memories (db cascade), and the files with them (fleet.py's robot delete).
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import re
import unicodedata

from fastapi import APIRouter, HTTPException
from fastapi.requests import HTTPConnection
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

import auth
import db

router = APIRouter()

MEMORY_DIR = db.DATA_DIR / "memory"
MAX_ITEMS = 500                  # listed per robot, newest first
MAX_IMAGE_BYTES = 6_000_000      # a frame is ~100-400 KB; this is a bound, not a target
NOTE_CHARS = 500
# What goes to the assistant with each question: the newest notes that fit.
NOTES_TO_ASSISTANT_CHARS = 1500

# The two requests, matched on the plain text (lower case, no accents), at the start. By the
# verb's STEM, so any form Whisper writes counts: "Sacad una foto a esto" (it heard the
# vosotros form) was not taken for a request, and the model then claimed it had saved the
# picture (2026-10-28). A question is not a request ("¿guardaste la foto?", "¿qué imagen
# guardaste?") unless it is a polite one ("¿podés guardar esta foto?"). Command words, in both
# languages — no topic words.
_NAMES = r"(?:(?:bender|huesos|robot)\W+)?(?:(?:por favor|please)\W+)?"
_POLITE = r"(?:(?:me |nos )?(?:podes|podrias|puedes|podria|can you|could you|would you|will you)\W+)?"
_PICTURE = r"\b(?:imagen\w*|foto\w*|captura|picture|photo\w*|image|snapshot|screenshot|pic)\b"
# "Guardá" or "save" + what it points at ("esto", "este control") is a picture too: Whisper made
# "guardá una foto de este control" into "Lo guardamos por todo este control" (2026-10-28).
# "Sacá" / "tomá" need the picture word: "sacá la mano" keeps nothing.
_POINTED = r"\b(?:esto|este|esta|eso|ese|esa|aca|ahi|this|that|it)\b"
SAVE_IMAGE = re.compile(
    r"^\W*" + _NAMES + _POLITE + r"(?:(?:lo|la|los|las|me lo|me la)\W+)?"
    r"(?:(?:guard|sac|tom|save|take|store|keep|snap)\w*\b[^?]*?" + _PICTURE
    + r"|(?:guard|save|store|keep)\w*\b[^?]*?" + _POINTED + r"|(?:fotografi|captur)\w*\b)")
REMEMBER = re.compile(
    r"^\W*" + _NAMES + _POLITE + r"(?:(?:acord|acuerd|record|recuerd|anot)\w*|guarda\w* en (?:tu )?memoria|"
    r"remember|note|keep in mind)\W+(?:de )?(?:que|that)\W+")
_ASKS_POLITELY = re.compile(r"^\W*" + _NAMES + r"(?:me |nos )?(?:podes|podrias|puedes|podria|can you|"
                            r"could you|would you|will you)\b")


def _plain(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn")


def command(prompt: str) -> tuple[str, str] | None:
    """("image", "") or ("note", what to keep) when `prompt` asks to keep something."""
    text = unicodedata.normalize("NFC", (prompt or "").strip())
    plain = _plain(text)
    if "?" in plain and not _ASKS_POLITELY.match(plain):
        return None
    if m := REMEMBER.match(plain):
        # Same length as the original (one letter per accented letter), so the note keeps
        # the original's capitals and accents.
        note = (text[m.end():] if len(plain) == len(text) else plain[m.end():]).strip(" .?")
        return ("note", note[:NOTE_CHARS]) if note else None
    if SAVE_IMAGE.match(plain):
        return ("image", "")
    return None


def _spanish(text: str) -> bool:
    if re.search(r"[¿¡ñáéíóú]", text, re.IGNORECASE):
        return True
    words = re.findall(r"[a-z]+", text.lower())
    es = sum(w in {"que", "la", "el", "de", "esta", "este", "una", "un", "en", "mi", "tu"} for w in words)
    en = sum(w in {"the", "this", "that", "a", "my", "your", "to", "of", "in", "picture"} for w in words)
    return es >= en


def _reply(key: str, spanish: bool, note: str = "") -> str:
    es = {"image": "Listo, guardé la imagen en mi memoria.",
          "note": f"Listo, me lo acuerdo: {note}.",
          "no_robot": "No hay un robot elegido donde guardarlo.",
          "no_image": "No tengo una imagen para guardar.",
          "bad_image": "No pude guardar esa imagen."}
    en = {"image": "Done, I saved the picture to my memory.",
          "note": f"Got it, I'll remember: {note}.",
          "no_robot": "There is no robot selected to keep it.",
          "no_image": "I have no picture to save.",
          "bad_image": "I could not save that picture."}
    return (es if spanish else en)[key]


def _image_bytes(image_b64: str) -> bytes | None:
    """The decoded picture when it is a JPEG or PNG within bounds; None otherwise."""
    data = image_b64.split(",", 1)[1] if image_b64.startswith("data:") else image_b64
    if not data or len(data) > MAX_IMAGE_BYTES * 4 // 3 + 4:
        return None
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None
    if not (raw.startswith(b"\xff\xd8\xff") or raw.startswith(b"\x89PNG\r\n\x1a\n")):
        return None
    return raw if len(raw) <= MAX_IMAGE_BYTES else None


async def _robot(s, org_id: str, slot: str) -> db.Robot | None:
    return (await s.execute(select(db.Robot).where(db.Robot.org_id == org_id,
                                                   db.Robot.slot == slot))).scalars().first()


async def _keep(me: auth.Principal, robot: db.Robot, kind: str, text: str = "",
                image: bytes | None = None) -> db.Memory:
    item = db.Memory(org_id=me.org_id, robot_id=robot.id, kind=kind, text=text[:NOTE_CHARS],
                     created_by=me.user_id, created_by_name=(me.name or me.email or "")[:120])
    async with db.SessionLocal() as s:
        s.add(item)
        await s.flush()
        if image is not None:
            MEMORY_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
            await asyncio.to_thread((MEMORY_DIR / item.id).write_bytes, image)
        await s.commit()
        await s.refresh(item)
    return item


async def handle(conn: HTTPConnection, slot: str | None, cmd: tuple[str, str], image_b64: str,
                 prompt: str) -> str:
    """Do what `command()` found and say so, in the language asked."""
    me: auth.Principal = conn.scope["principal"]
    spanish = _spanish(prompt)
    kind, note = cmd
    async with db.SessionLocal() as s:
        robot = await _robot(s, me.org_id, slot) if slot else None
    if robot is None:
        return _reply("no_robot", spanish)
    if kind == "image":
        if not image_b64:
            return _reply("no_image", spanish)
        raw = _image_bytes(image_b64)
        if raw is None:
            return _reply("bad_image", spanish)
        await _keep(me, robot, "image", image=raw)
        await auth.audit(conn, "memory.save", target=robot.name, detail={"kind": "image"})
        return _reply("image", spanish)
    await _keep(me, robot, "note", text=note)
    await auth.audit(conn, "memory.save", target=robot.name, detail={"kind": "note"})
    return _reply("note", spanish, note)


async def notes(org_id: str, slot: str) -> list[str]:
    """The robot's newest notes that fit NOTES_TO_ASSISTANT_CHARS, oldest first."""
    async with db.SessionLocal() as s:
        robot = await _robot(s, org_id, slot)
        if robot is None:
            return []
        texts = (await s.execute(
            select(db.Memory.text).where(db.Memory.robot_id == robot.id, db.Memory.kind == "note")
            .order_by(db.Memory.created_at.desc()).limit(50))).scalars().all()
    out, used = [], 0
    for text in texts:
        if used + len(text) > NOTES_TO_ASSISTANT_CHARS:
            break
        out.append(text)
        used += len(text)
    return out[::-1]


def _json(m: db.Memory) -> dict:
    return {"id": m.id, "kind": m.kind, "text": m.text, "created_at": m.created_at.isoformat(),
            "created_by": m.created_by_name}


@router.get("/api/memory")
async def list_memory(conn: HTTPConnection, robot: str):
    me: auth.Principal = conn.scope["principal"]
    async with db.SessionLocal() as s:
        r = await _robot(s, me.org_id, robot)
        if r is None:
            return {"robot": None, "items": [], "total": 0}
        total = (await s.execute(select(func.count(db.Memory.id))
                                 .where(db.Memory.robot_id == r.id))).scalar_one()
        items = (await s.execute(select(db.Memory).where(db.Memory.robot_id == r.id)
                                 .order_by(db.Memory.created_at.desc())
                                 .limit(MAX_ITEMS))).scalars().all()
    return {"robot": {"id": r.id, "name": r.name, "slot": r.slot},
            "items": [_json(m) for m in items], "total": total}


async def _own(s, me: auth.Principal, memory_id: str) -> db.Memory:
    item = await s.get(db.Memory, memory_id)
    if item is None or item.org_id != me.org_id:
        raise HTTPException(404, "No such memory.")
    return item


@router.get("/api/memory/{memory_id}/image")
async def memory_image(memory_id: str, conn: HTTPConnection):
    me: auth.Principal = conn.scope["principal"]
    async with db.SessionLocal() as s:
        item = await _own(s, me, memory_id)
    path = MEMORY_DIR / item.id
    if item.kind != "image" or not path.is_file():
        raise HTTPException(404, "This memory has no picture.")
    head = await asyncio.to_thread(lambda: path.open("rb").read(4))
    return FileResponse(path, media_type="image/png" if head == b"\x89PNG" else "image/jpeg",
                        headers={"Cache-Control": "private, max-age=86400"})


class NewMemory(BaseModel):
    robot: str = Field(..., max_length=16)
    kind: str = Field(..., pattern="^(image|note)$")
    text: str = Field("", max_length=NOTE_CHARS)
    image: str = Field("", max_length=MAX_IMAGE_BYTES * 4 // 3 + 32)


@router.post("/api/memory")
async def add_memory(body: NewMemory, conn: HTTPConnection):
    """Keep a note typed on the Memory page, or a picture sent from the app."""
    me: auth.Principal = conn.scope["principal"]
    async with db.SessionLocal() as s:
        robot = await _robot(s, me.org_id, body.robot)
    if robot is None:
        raise HTTPException(404, "No such robot.")
    if body.kind == "note":
        if not body.text.strip():
            raise HTTPException(422, "Write what to remember.")
        item = await _keep(me, robot, "note", text=body.text.strip())
    else:
        raw = _image_bytes(body.image)
        if raw is None:
            raise HTTPException(422, "Not a JPEG or PNG picture, or too big.")
        item = await _keep(me, robot, "image", image=raw)
    await auth.audit(conn, "memory.save", target=robot.name, detail={"kind": body.kind})
    return _json(item)


@router.delete("/api/memory/{memory_id}")
async def delete_memory(memory_id: str, conn: HTTPConnection):
    me: auth.Principal = conn.scope["principal"]
    async with db.SessionLocal() as s:
        item = await _own(s, me, memory_id)
        robot = await s.get(db.Robot, item.robot_id)
        kind = item.kind
        await s.delete(item)
        await s.commit()
    await asyncio.to_thread((MEMORY_DIR / memory_id).unlink, missing_ok=True)
    await auth.audit(conn, "memory.delete", target=robot.name if robot else "", detail={"kind": kind})
    return {"ok": True}


async def forget_robot(robot_id: str) -> None:
    """Remove the picture files of a robot being deleted (its rows go by cascade)."""
    async with db.SessionLocal() as s:
        ids = (await s.execute(select(db.Memory.id).where(db.Memory.robot_id == robot_id,
                                                          db.Memory.kind == "image"))).scalars().all()
    for memory_id in ids:
        await asyncio.to_thread((MEMORY_DIR / memory_id).unlink, missing_ok=True)
