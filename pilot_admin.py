#!/usr/bin/env python3
"""pilot_admin.py — account administration from the server's shell.

For what the web app cannot do yet, and for getting back in when nobody can sign in — e.g.
the only owner forgot their password: `set-password` is the way back. It needs a shell on the
server as the account that runs the gateway, which can already read the database itself, so
it opens nothing new; it signs the person out everywhere and leaves an audit row. It
talks to the same database as the gateway (PILOT_DATA_DIR / DATABASE_URL), so run it on the
server, as the user that runs the gateway, from this directory:

    .venv/bin/python pilot_admin.py users
    .venv/bin/python pilot_admin.py set-password ana@acme.example
    .venv/bin/python pilot_admin.py set-email ana@acme.exmple ana@acme.example
    .venv/bin/python pilot_admin.py device-key create "camera bridge" --save
    .venv/bin/python pilot_admin.py device-key create "latency probe" --role viewer
    .venv/bin/python pilot_admin.py device-key list
    .venv/bin/python pilot_admin.py device-key revoke pk_Ab12Cd

The gateway creates the camera bridge's key by itself when the install is set up (in
PILOT_DATA_DIR/device-key); `device-key create ... --save` replaces it, e.g. to rotate it —
revoke the old one afterwards, and restart the bridge so it reads the new one.

A running gateway keeps validated credentials in memory for up to auth.CACHE_TTL seconds, so
a key revoked here stops working within that window, not instantly.
"""
from __future__ import annotations

import argparse
import getpass
import sys

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session

import auth
import db


def _session() -> Session:
    db.ensure_data_dir()
    db.migrate()
    return Session(create_engine(db.sync_url()))


def users(_args) -> int:
    with _session() as s:
        rows = s.execute(
            select(db.User, db.Membership).join(db.Membership, db.Membership.user_id == db.User.id)
        ).all()
        for user, member in rows:
            state = "active" if user.is_active else "disabled"
            print(f"{user.email:40} {member.role:9} {state:9} {user.name}")
        if not rows:
            print(f"No accounts yet: open the app and use the setup code in {auth.SETUP_CODE_PATH}")
    return 0


def set_email(args) -> int:
    """Correct an account's address — e.g. a typo made at first-run setup, which single sign-on
    would never match (it finds people by the address their work account has)."""
    old = args.old.strip().lower()
    try:
        new = auth.normalize_email(args.new)
    except Exception as e:
        print(getattr(e, "detail", str(e)), file=sys.stderr)
        return 1
    with _session() as s:
        user = s.execute(select(db.User).where(db.User.email == old)).scalar()
        if user is None:
            print(f"No account for {old}.", file=sys.stderr)
            return 1
        if s.execute(select(db.User.id).where(db.User.email == new)).first():
            print(f"{new} already has an account; nothing changed.", file=sys.stderr)
            return 1
        user.email = new
        member = s.execute(select(db.Membership).where(db.Membership.user_id == user.id)).scalar()
        s.add(db.AuditEvent(org_id=member.org_id if member else None, action="team.email_changed",
                            target=new, detail={"from": old, "via": "pilot_admin.py on the server"}))
        s.commit()
    print(f"{old} is now {new}. Sign in with the new address (the password is unchanged).")
    return 0


def set_password(args) -> int:
    email = args.email.strip().lower()
    password = getpass.getpass(f"New password for {email}: ")
    if password != getpass.getpass("Again: "):
        print("The two passwords differ; nothing changed.", file=sys.stderr)
        return 1
    try:
        auth.check_password_policy(password, email)
    except Exception as e:
        print(getattr(e, "detail", str(e)), file=sys.stderr)
        return 1
    with _session() as s:
        user = s.execute(select(db.User).where(db.User.email == email)).scalar()
        if user is None:
            print(f"No account for {email}.", file=sys.stderr)
            return 1
        user.password_hash = auth._hasher.hash(password)
        # A new password signs every existing session out.
        s.execute(update(db.AuthSession).where(db.AuthSession.user_id == user.id,
                                               db.AuthSession.revoked_at.is_(None))
                  .values(revoked_at=db.utcnow()))
        # Recorded like any other change, so a reset done on the server is visible in the app.
        member = s.execute(select(db.Membership).where(db.Membership.user_id == user.id)).scalar()
        s.add(db.AuditEvent(org_id=member.org_id if member else None, action="auth.password_reset",
                            target=email, detail={"via": "pilot_admin.py on the server"}))
        s.commit()
    print(f"Password changed for {email}; their open sessions were signed out.")
    return 0


def key_create(args) -> int:
    with _session() as s:
        org = s.execute(select(db.Organization).limit(1)).scalar()
        if org is None:
            print("Set the install up in the app first (there is no organization yet).",
                  file=sys.stderr)
            return 1
        key = auth.new_api_key()
        s.add(db.ApiKey(org_id=org.id, name=args.name, role=args.role,
                        key_hash=auth.token_hash(key), prefix=key[:9]))
        s.commit()
    if args.save:
        auth.write_secret_file(auth.DEVICE_KEY_PATH, key)
        print(f"Device key for {args.name!r} saved to {auth.DEVICE_KEY_PATH} (0600).")
        return 0
    # Printed once, on purpose, to the operator who asked for it: it is never stored or
    # shown again. Put it where the device reads it, e.g. PILOT_DEVICE_KEY.
    print(key)
    print(f"Device key for {args.name!r} created. Copy it now: it cannot be shown again.",
          file=sys.stderr)
    return 0


def key_list(_args) -> int:
    with _session() as s:
        for k in s.execute(select(db.ApiKey).order_by(db.ApiKey.created_at)).scalars():
            state = "revoked" if k.revoked_at else "active"
            used = k.last_used_at.strftime("%Y-%m-%d %H:%M") if k.last_used_at else "never"
            print(f"{k.prefix}…  {state:8} last used {used:16}  {k.name}")
    return 0


def key_revoke(args) -> int:
    with _session() as s:
        n = s.execute(update(db.ApiKey).where(db.ApiKey.prefix == args.prefix,
                                              db.ApiKey.revoked_at.is_(None))
                      .values(revoked_at=db.utcnow())).rowcount
        s.commit()
    print(f"Revoked {n} key(s) starting with {args.prefix}.")
    return 0 if n else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Silk Pilot account administration.")
    sub = p.add_subparsers(required=True)
    sub.add_parser("users", help="list accounts and roles").set_defaults(fn=users)
    sp = sub.add_parser("set-password", help="set a user's password (signs them out)")
    sp.add_argument("email")
    sp.set_defaults(fn=set_password)
    se = sub.add_parser("set-email", help="correct an account's email address")
    se.add_argument("old")
    se.add_argument("new")
    se.set_defaults(fn=set_email)
    keys = sub.add_parser("device-key", help="API keys for machines (the camera bridge)")
    ksub = keys.add_subparsers(required=True)
    kc = ksub.add_parser("create", help="create a key; prints it once")
    kc.add_argument("name")
    kc.add_argument("--role", choices=("device", "viewer"), default="device",
                    help="device = push video in (the camera bridge); viewer = read-only, e.g. "
                         "a measurement script that watches /ws/view")
    kc.add_argument("--save", action="store_true",
                    help=f"write it to {auth.DEVICE_KEY_PATH} (the bridge reads it there) "
                         "instead of printing it")
    kc.set_defaults(fn=key_create)
    ksub.add_parser("list", help="list keys (prefix, state, last use)").set_defaults(fn=key_list)
    kr = ksub.add_parser("revoke", help="revoke the key with this prefix")
    kr.add_argument("prefix")
    kr.set_defaults(fn=key_revoke)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
