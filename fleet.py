"""fleet.py — sites and robots: where the company works, and with which robots.

Everyone signed in may look ("view"); adding or changing a site or a robot is a setting, so it
needs "configure" (admins, owners) — see auth.POLICY.

A robot's SLOT is how the running stack addresses it: the executor, the on-robot relay and the
camera bridge know one Go2 and one G1 (ROBOT_SLOTS). So a slot holds one robot per company
until two robots of one model can be told apart on the network — an open decision in the
roadmap ("cómo separar los dos robots"). The robot's NAME and SERIAL are the company's; the
slot is only the wire address.

A site may carry a floor plan, an image uploaded as the raw request body (no multipart
dependency), checked by its first bytes, at most PLAN_MAX bytes, stored in
PILOT_DATA_DIR/site-plans/<site id>. Satellite imagery is deliberately not here: free tile
servers are not licensed for a product, and indoor sites need the plan anyway.
"""
from __future__ import annotations

import asyncio
import zoneinfo
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Request
from fastapi.requests import HTTPConnection
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, update

import auth
import db

ROBOT_SLOTS = {"go2": "Unitree Go2", "g1": "Unitree G1"}
PLAN_DIR = db.DATA_DIR / "site-plans"
PLAN_MAX = 10 * 1024 * 1024  # a large PNG plan is a few MB; bound it (standard §2)
# First bytes -> media type. Anything else is refused, whatever its Content-Type says.
_PLAN_MAGIC = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"))

router = APIRouter()


@lru_cache(maxsize=1024)
def _known_zone(name: str) -> bool:
    """Any zone this system can load — including the old aliases browsers still report
    ("America/Buenos_Aires"), which available_timezones() leaves out."""
    try:
        zoneinfo.ZoneInfo(name)
        return True
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        return False


def _principal(conn: HTTPConnection) -> auth.Principal:
    return conn.scope["principal"]


def _site_json(site: db.Site) -> dict:
    return {"id": site.id, "name": site.name, "timezone": site.timezone,
            "address": site.address, "lat": site.lat, "lon": site.lon,
            "has_plan": bool(site.plan_type)}


def _robot_json(robot: db.Robot) -> dict:
    return {"id": robot.id, "name": robot.name, "slot": robot.slot,
            "model": ROBOT_SLOTS.get(robot.slot, robot.slot), "serial": robot.serial,
            "site_id": robot.site_id}


@router.get("/api/fleet")
async def fleet(conn: HTTPConnection):
    """Everything the Fleet page draws, in one call: sites, robots, and the free slots."""
    me = _principal(conn)
    async with db.SessionLocal() as s:
        sites = (await s.execute(select(db.Site).where(db.Site.org_id == me.org_id)
                                 .order_by(db.Site.created_at))).scalars().all()
        robots = (await s.execute(select(db.Robot).where(db.Robot.org_id == me.org_id)
                                  .order_by(db.Robot.created_at))).scalars().all()
    used = {r.slot for r in robots}
    return {
        "sites": [_site_json(x) for x in sites],
        "robots": [_robot_json(r) for r in robots],
        "slots": [{"slot": k, "model": v, "free": k not in used} for k, v in ROBOT_SLOTS.items()],
    }


# --------------------------------------------------------------------------- #
# Sites
# --------------------------------------------------------------------------- #
class SiteBody(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    timezone: str | None = Field(default=None, max_length=64)
    address: str | None = Field(default=None, max_length=255)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    clear_location: bool = False


def _apply_site(site: db.Site, body: SiteBody) -> None:
    if body.timezone is not None and not _known_zone(body.timezone):
        raise HTTPException(422, f"Unknown time zone {body.timezone!r}.")
    if (body.lat is None) != (body.lon is None):
        raise HTTPException(422, "Send both latitude and longitude, or neither.")
    for field in ("name", "timezone", "address"):
        value = getattr(body, field)
        if value is not None:
            setattr(site, field, value.strip())
    if body.lat is not None:
        site.lat, site.lon = body.lat, body.lon
    if body.clear_location:
        site.lat = site.lon = None


async def _own_site(s, me: auth.Principal, site_id: str) -> db.Site:
    site = await s.get(db.Site, site_id)
    if site is None or site.org_id != me.org_id:
        raise HTTPException(404, "No such site.")
    return site


@router.post("/api/sites")
async def create_site(body: SiteBody, conn: HTTPConnection):
    me = _principal(conn)
    if not body.name:
        raise HTTPException(422, "Give the site a name.")
    site = db.Site(org_id=me.org_id, timezone="UTC", address="", plan_type="")
    _apply_site(site, body)
    async with db.SessionLocal() as s:
        s.add(site)
        await s.commit()
    await auth.audit(conn, "fleet.site_created", me, target=site.name)
    return _site_json(site)


@router.patch("/api/sites/{site_id}")
async def update_site(site_id: str, body: SiteBody, conn: HTTPConnection):
    me = _principal(conn)
    async with db.SessionLocal() as s:
        site = await _own_site(s, me, site_id)
        _apply_site(site, body)
        await s.commit()
    await auth.audit(conn, "fleet.site_changed", me, target=site.name)
    return _site_json(site)


@router.delete("/api/sites/{site_id}")
async def delete_site(site_id: str, conn: HTTPConnection):
    """Delete a site; its robots stay, unassigned."""
    me = _principal(conn)
    async with db.SessionLocal() as s:
        site = await _own_site(s, me, site_id)
        await s.execute(update(db.Robot).where(db.Robot.site_id == site_id).values(site_id=None))
        await s.delete(site)
        await s.commit()
    await asyncio.to_thread((PLAN_DIR / site_id).unlink, missing_ok=True)
    await auth.audit(conn, "fleet.site_deleted", me, target=site.name)
    return {"ok": True}


async def _read_bounded(request: Request, limit: int) -> bytes:
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, f"The plan is larger than {limit // (1024 * 1024)} MB.")
        chunks.append(chunk)
    return b"".join(chunks)


def _write_plan(site_id: str, data: bytes) -> None:
    PLAN_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = PLAN_DIR / f".{site_id}.tmp"
    tmp.write_bytes(data)
    tmp.replace(PLAN_DIR / site_id)  # atomic: a reader never sees half a file


@router.put("/api/sites/{site_id}/plan")
async def upload_plan(site_id: str, request: Request):
    me = _principal(request)
    data = await _read_bounded(request, PLAN_MAX)
    kind = next((t for magic, t in _PLAN_MAGIC if data.startswith(magic)), None)
    if kind is None and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        kind = "image/webp"
    if kind is None:
        raise HTTPException(415, "Upload the plan as a PNG, JPEG or WebP image.")
    async with db.SessionLocal() as s:
        site = await _own_site(s, me, site_id)
        await asyncio.to_thread(_write_plan, site_id, data)
        site.plan_type = kind
        await s.commit()
    await auth.audit(request, "fleet.plan_uploaded", me, target=site.name,
                     detail={"bytes": len(data), "type": kind})
    return _site_json(site)


@router.get("/api/sites/{site_id}/plan")
async def get_plan(site_id: str, conn: HTTPConnection):
    me = _principal(conn)
    async with db.SessionLocal() as s:
        site = await _own_site(s, me, site_id)
    if not site.plan_type:
        raise HTTPException(404, "This site has no plan.")
    return FileResponse(PLAN_DIR / site_id, media_type=site.plan_type,
                        headers={"Cache-Control": "private, no-cache"})


@router.delete("/api/sites/{site_id}/plan")
async def delete_plan(site_id: str, conn: HTTPConnection):
    me = _principal(conn)
    async with db.SessionLocal() as s:
        site = await _own_site(s, me, site_id)
        site.plan_type = ""
        await s.commit()
    await asyncio.to_thread((PLAN_DIR / site_id).unlink, missing_ok=True)
    await auth.audit(conn, "fleet.plan_removed", me, target=site.name)
    return _site_json(site)


# --------------------------------------------------------------------------- #
# Robots
# --------------------------------------------------------------------------- #
class RobotBody(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    slot: str | None = Field(default=None, max_length=16)
    serial: str | None = Field(default=None, max_length=80)
    site_id: str | None = Field(default=None, max_length=32)
    unassign: bool = False


async def _check_site(s, me: auth.Principal, site_id: str | None) -> None:
    if site_id is not None:
        await _own_site(s, me, site_id)


async def _check_slot(s, me: auth.Principal, slot: str, robot_id: str | None = None) -> None:
    if slot not in ROBOT_SLOTS:
        raise HTTPException(422, f"Unknown robot type {slot!r}.")
    other = (await s.execute(select(db.Robot).where(db.Robot.org_id == me.org_id,
                                                    db.Robot.slot == slot))).scalar()
    if other is not None and other.id != robot_id:
        raise HTTPException(
            409, f"{other.name} already uses the {ROBOT_SLOTS[slot]} connection. This server "
                 "drives one robot of each model for now.")


@router.post("/api/robots")
async def create_robot(body: RobotBody, conn: HTTPConnection):
    me = _principal(conn)
    if not body.name or not body.slot:
        raise HTTPException(422, "Give the robot a name and a type.")
    async with db.SessionLocal() as s:
        await _check_slot(s, me, body.slot)
        await _check_site(s, me, body.site_id)
        robot = db.Robot(org_id=me.org_id, name=body.name.strip(), slot=body.slot,
                         serial=(body.serial or "").strip(), site_id=body.site_id)
        s.add(robot)
        await s.commit()
    await auth.audit(conn, "fleet.robot_added", me, target=robot.name, detail={"slot": robot.slot})
    return _robot_json(robot)


@router.patch("/api/robots/{robot_id}")
async def update_robot(robot_id: str, body: RobotBody, conn: HTTPConnection):
    me = _principal(conn)
    async with db.SessionLocal() as s:
        robot = await s.get(db.Robot, robot_id)
        if robot is None or robot.org_id != me.org_id:
            raise HTTPException(404, "No such robot.")
        if body.slot is not None:
            await _check_slot(s, me, body.slot, robot.id)
            robot.slot = body.slot
        if body.site_id is not None:
            await _check_site(s, me, body.site_id)
            robot.site_id = body.site_id
        if body.unassign:
            robot.site_id = None
        if body.name is not None:
            robot.name = body.name.strip()
        if body.serial is not None:
            robot.serial = body.serial.strip()
        await s.commit()
    await auth.audit(conn, "fleet.robot_changed", me, target=robot.name)
    return _robot_json(robot)


@router.delete("/api/robots/{robot_id}")
async def delete_robot(robot_id: str, conn: HTTPConnection):
    me = _principal(conn)
    async with db.SessionLocal() as s:
        robot = await s.get(db.Robot, robot_id)
        if robot is None or robot.org_id != me.org_id:
            raise HTTPException(404, "No such robot.")
        await s.delete(robot)
        await s.commit()
    await auth.audit(conn, "fleet.robot_removed", me, target=robot.name)
    return {"ok": True}
