"""Backups (backup.py), against the real auth.authorize (marker `real_auth`).

Proven here: a backup is a dated folder with a database that opens and holds the accounts, and
the files beside the database; only the newest BACKUP_KEEP are kept; only an owner sees or
makes them.
"""
from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

import app as gateway
import auth
import backup
import db

pytestmark = pytest.mark.real_auth

OWNER = {"org_name": "Acme Logistics", "name": "Ana Owner", "email": "ana@acme.example",
         "password": "correct horse battery"}


@pytest.fixture(scope="module")
def owner():
    with TestClient(gateway.app) as tc:
        if tc.get("/api/auth/state").json()["setup_required"]:
            r = tc.post("/api/auth/setup", json={**OWNER, "setup_code": auth.SETUP_CODE_PATH.read_text().strip()})
        else:
            r = tc.post("/api/auth/login", json={"email": OWNER["email"], "password": OWNER["password"]})
        assert r.status_code == 200, r.text
        yield tc


def test_a_backup_holds_the_database_and_the_files(owner, monkeypatch, tmp_path):
    monkeypatch.setattr(backup, "BACKUP_DIR", tmp_path / "backups")
    (db.DATA_DIR / "site-plans").mkdir(exist_ok=True)
    (db.DATA_DIR / "site-plans" / "plan-test").write_bytes(b"\x89PNG plan")
    made = owner.post("/api/backups").json()
    folder = tmp_path / "backups" / made["name"]
    with sqlite3.connect(folder / "pilot.db") as copy:
        emails = {r[0] for r in copy.execute("SELECT email FROM users")}
    assert OWNER["email"] in emails                                   # a database that opens
    assert (folder / "site-plans" / "plan-test").read_bytes() == b"\x89PNG plan"
    assert not (folder / "backups").exists()                          # never a backup of backups
    listed = owner.get("/api/backups").json()
    assert listed["backups"][0]["name"] == made["name"] and listed["keep"] == backup.BACKUP_KEEP


def test_only_the_newest_are_kept(monkeypatch, tmp_path):
    from datetime import UTC, datetime, timedelta
    monkeypatch.setattr(backup, "BACKUP_DIR", tmp_path / "backups")
    monkeypatch.setattr(backup, "BACKUP_KEEP", 3)
    day = datetime(2026, 10, 1, tzinfo=UTC)
    for i in range(5):
        backup._run(day + timedelta(days=i))
    assert [b["name"] for b in backup.listing()] == ["2026-10-05", "2026-10-04", "2026-10-03"]


def test_only_an_owner_sees_backups(owner):
    r = owner.post("/api/team/invitations", json={"email": "bk-admin@acme.example", "role": "admin"})
    token = r.json()["link_path"].split("token=")[1]
    admin = TestClient(gateway.app)
    admin.post("/api/invitations/accept", json={"token": token, "name": "Admin", "password": "another long password"})
    assert admin.get("/api/backups").status_code == 403
    assert admin.post("/api/backups").status_code == 403
