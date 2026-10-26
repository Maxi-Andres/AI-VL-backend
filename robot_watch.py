"""robot_watch.py — remember each robot's last known state, with or without anyone watching.

Every WATCH_EVERY seconds the gateway asks the executor how each robot is reached (its
/transport: the ping address, the link, and the relay's battery reading), pings each robot,
and writes what it saw onto the robots table: online since when, last seen, round-trip
latency, link, last battery and when it was read. The Fleet page reads it back, so a robot
that dropped two hours ago says "offline · 2 h" and still shows the battery it had.

A change of online/offline counts only after FLIP_AFTER samples in a row agree, so one lost
ping does not flap the card or flood the activity feed; a confirmed change is logged there
("Huesos went offline"). A robot with no ping address is never judged: unknown is not down.
The executor being unreachable is not evidence about any robot either, so that tick records
nothing.

One GET per tick on the executor, which sits on the control path: the same rate as one
browser's header poll, well inside what it serves (standard §1).
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from datetime import datetime

import httpx
from sqlalchemy import select

import auth
import db

logger = logging.getLogger("uvicorn.error")

WATCH_EVERY = float(os.environ.get("ROBOT_WATCH_SECONDS", "15"))  # 0 turns the watcher off
FLIP_AFTER = 2

_RTT = re.compile(rb"time[=<]([0-9.]+) ?ms")
# slot -> (the state seen last, how many samples in a row it has held)
_streak: dict[str, tuple[bool, int]] = {}


async def ping_rtt(ip: str, timeout: float = 1.5) -> float | None:  # noqa: ASYNC109  # bounded with asyncio.wait_for below
    """Round trip in ms if the host answers one ICMP echo, else None. Runs `ping` on the
    host: the executor's container has no raw sockets."""
    if not ip:
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "ping", "-c", "1", "-W", "1", ip,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return None
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    m = _RTT.search(out)
    return float(m.group(1)) if m else 0.0


def _confirmed(slot: str, online: bool, stored: bool | None) -> bool | None:
    """The state to record: the new one once FLIP_AFTER samples agree, the stored one until
    then. The first measurement ever is taken at once — there is nothing to flap from."""
    last, n = _streak.get(slot, (online, 0))
    n = n + 1 if last == online else 1
    _streak[slot] = (online, n)
    if stored is None or n >= FLIP_AFTER:
        return online
    return stored


def _fresh_battery(t: dict) -> dict | None:
    b = (t.get("relay") or {}).get("battery") or {}
    if isinstance(b.get("percent"), (int, float)) and not b.get("stale"):
        return b
    return None


async def record(transports: dict[str, dict], now: datetime | None = None) -> None:
    """Write one sample per robot slot. `transports` is the executor's /transport map."""
    now = now or db.utcnow()
    slots = [s for s, t in transports.items() if t.get("ping_ip")]
    rtts = await asyncio.gather(*(ping_rtt(transports[s]["ping_ip"]) for s in slots))
    changes: list[tuple[str, str, str]] = []
    async with db.SessionLocal() as s:
        for slot, rtt in zip(slots, rtts, strict=True):
            t = transports[slot]
            online = rtt is not None
            robots = (await s.execute(select(db.Robot).where(db.Robot.slot == slot))).scalars().all()
            state = None
            for robot in robots:
                state = _confirmed(slot, online, robot.online) if state is None else state
                if state != robot.online:
                    if robot.online is not None:
                        changes.append((robot.org_id, robot.name, "robot.online" if state else "robot.offline"))
                    robot.online, robot.status_since = state, now
                if online:
                    robot.last_seen_at, robot.latency_ms = now, rtt
                robot.link = str(t.get("mode") or "")[:16] or None
                if (b := _fresh_battery(t)) is not None:
                    robot.battery_pct = float(b["percent"])
                    robot.battery_charging = b.get("charging") if isinstance(b.get("charging"), bool) else None
                    robot.battery_at = now
            if not robots:
                _confirmed(slot, online, None)
        await s.commit()
    for org_id, name, action in changes:
        await auth.audit(None, action, target=name, org_id=org_id)


async def _tick(ec: httpx.AsyncClient, executor_url: str) -> None:
    try:
        r = await ec.get(f"{executor_url}/transport")
        transports = r.json().get("transports") or {}
    except Exception as e:
        logger.debug("robot_watch: executor unreachable: %s", e)
        return
    await record(transports)


async def _run(executor_url: str) -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=8.0, write=5.0, pool=3.0)) as ec:
        while True:
            try:
                await _tick(ec, executor_url)
            except Exception as e:  # one bad tick must not end the watch
                logger.warning("robot_watch: tick failed: %s", e)
            await asyncio.sleep(WATCH_EVERY)


def start(executor_url: str) -> asyncio.Task | None:
    if WATCH_EVERY <= 0:
        return None
    return asyncio.create_task(_run(executor_url), name="robot_watch")


async def stop(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
