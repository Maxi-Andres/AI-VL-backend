"""backup.py — a copy of everything Pilot keeps, once a day, the last few days kept.

What Pilot keeps is one directory, PILOT_DATA_DIR (~/.pilot): the SQLite database (accounts,
the team, sites and robots, conversations, the activity log) and a few files beside it (floor
plans, the camera bridge's key, the map key). A backup is a dated folder holding all of it:

  <PILOT_BACKUP_DIR>/2026-10-27/pilot.db       made with SQLite's own backup API — a consistent
                                               copy while the app keeps writing, unlike a cp
  <PILOT_BACKUP_DIR>/2026-10-27/site-plans/…   the other files, as they are

Once a day (PILOT_BACKUP_HOURS) and on startup when the newest backup is older than that; the
newest PILOT_BACKUP_KEEP are kept. The default folder is inside the data directory, which
survives a broken app but not a broken disk: point PILOT_BACKUP_DIR at another disk or a
mounted share for that. Restoring is stopping Pilot and copying a backup's files back.

A database that is not SQLite (DATABASE_URL to Postgres) is backed up by that server's own
tools; this then copies the files only.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import db

logger = logging.getLogger("uvicorn.error")

BACKUP_DIR = Path(os.environ.get("PILOT_BACKUP_DIR", "") or db.DATA_DIR / "backups").expanduser()
BACKUP_KEEP = max(1, int(os.environ.get("PILOT_BACKUP_KEEP", "7")))
BACKUP_HOURS = float(os.environ.get("PILOT_BACKUP_HOURS", "24"))  # 0 turns the daily run off
DB_FILE = "pilot.db"


def _sqlite_path() -> Path | None:
    url = db.DATABASE_URL
    if not url.startswith("sqlite"):
        return None
    return Path(url.split("///", 1)[1]) if "///" in url else None


def _run(now: datetime) -> Path:
    """Make one backup folder now (blocking: run it in a thread)."""
    BACKUP_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    name = now.strftime("%Y-%m-%d")
    dest = BACKUP_DIR / name
    if dest.exists():                       # a second run the same day replaces that day's copy
        shutil.rmtree(dest)
    tmp = BACKUP_DIR / f".{name}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(mode=0o700)
    src = _sqlite_path()
    if src and src.exists():
        with sqlite3.connect(src) as live, sqlite3.connect(tmp / DB_FILE) as copy:
            live.backup(copy)
    backups = BACKUP_DIR.resolve()
    for item in db.DATA_DIR.iterdir():
        if item.resolve() == backups or item.name.startswith(DB_FILE):  # pilot.db, -wal, -shm
            continue
        target = tmp / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        elif item.is_file():
            shutil.copy2(item, target)
    tmp.rename(dest)                        # a backup is either whole or not there
    for old in sorted(listing(), key=lambda b: b["name"], reverse=True)[BACKUP_KEEP:]:
        shutil.rmtree(BACKUP_DIR / old["name"], ignore_errors=True)
    return dest


def listing() -> list[dict]:
    """The backups there are, newest first: name (the day), when, size in bytes."""
    if not BACKUP_DIR.is_dir():
        return []
    out = []
    for d in BACKUP_DIR.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            out.append({"name": d.name, "at": datetime.fromtimestamp(d.stat().st_mtime, UTC).isoformat(),
                        "size": size})
    return sorted(out, key=lambda b: b["name"], reverse=True)


async def run_now() -> dict:
    dest = await asyncio.to_thread(_run, datetime.now(UTC))
    logger.info("backup: wrote %s", dest)
    return next(b for b in listing() if b["name"] == dest.name)


def _due(now: datetime) -> bool:
    newest = listing()
    if not newest:
        return True
    return now - datetime.fromisoformat(newest[0]["at"]) >= timedelta(hours=BACKUP_HOURS)


async def _loop() -> None:
    while True:
        try:
            if _due(datetime.now(UTC)):
                await run_now()
        except Exception as e:  # a failed backup must not take the app down; try again later
            logger.error("backup: failed: %s", e)
        await asyncio.sleep(min(3600.0, BACKUP_HOURS * 3600 / 4))


def start() -> asyncio.Task | None:
    if BACKUP_HOURS <= 0:
        return None
    return asyncio.create_task(_loop(), name="backup")


async def stop(task: asyncio.Task | None) -> None:
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
