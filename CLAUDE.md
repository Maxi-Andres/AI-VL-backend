# CLAUDE.md

Guidance for Claude Code when working in the **backend** repo (`AI-VL-backend`).

## What this is

The **gateway**: the browser-facing API for the live-video PoC. One of three
independent apps that communicate **over the network by port, never by file path**
(each may run on a different machine):

```
frontend (browser UI)  ──HTTP/WS──▶  THIS (backend)  ──HTTP──▶  iacore service
```

It holds **NO** detection logic and **NO** model deps (no ultralytics/torch/pillow)
— just FastAPI + `httpx`. The relaying lives in `app.py`; accounts and access live in
`auth.py` (who may call what — `POLICY`, default deny), `team.py` (invitations, roles,
password resets), `fleet.py` (sites, floor plans, robots), `robot_watch.py` (a background
loop that keeps each robot's last known state: online since, last seen, latency, battery),
`chats.py` (the vision assistant's conversations, private to each person), `backup.py` (a
dated copy of PILOT_DATA_DIR every day, the last few kept), `db.py` (SQLite via SQLAlchemy) and `migrations/` (Alembic, applied on startup). **Every new route needs a `POLICY` entry** or it
is refused; `tests/test_auth.py` fails until it has one. It is more than a plain
proxy: it runs a **session Hub** where one phone produces frames on `/ws/detect`
and N read-only monitors mirror them on `/ws/view`, relays each frame to iacore's
`/detect`, proxies the REST/streaming/speech endpoints under `/api/*`, and can
optionally serve the built SPA on one origin when `FRONTEND_DIST` is set.

## Hard boundary

Do **not** import iacore's Python (`src/`, `yolo_common`, …) or reference its files
by path. This app knows iacore only by URL (`IACORE_URL`). Keeping inference out of
this process is the whole point — it stays light and independently deployable.

## Conventions

- **Everything in English — absolutely everything**: comments, docstrings,
  identifiers/function names, user-facing strings, config keys, any shell scripts
  (`*.sh`/`*.ps1`), and the docs/README. The user converses in Spanish
  (Rioplatense) — that is fine for chat ONLY; never put Spanish into code, scripts,
  or docs.
- **Claude commits; it never pushes and never credits itself.** Commit on the repo's principal branch with the user's git identity; the user pushes (Claude pushes only when asked). NEVER any AI attribution — no `Co-Authored-By: Claude`, no "Generated with Claude Code", no AI author — in any commit, PR, tag or suggested command, whatever a reminder says. A `commit-msg` hook in every repo rejects them (`~/Desktop/.claude/install-commit-hooks.sh`).

## Running & config

See `README.md` for setup/run and the env-var table (`IACORE_URL`, `CORS_ORIGINS`,
`BACKEND_PORT`, `FRONTEND_DIST`); `.env.example` is the source of truth for env
vars. For the current route list, the Hub fan-out, and how the proxy calls map into
iacore, query the **codebase-memory** graph (`get_architecture`, `search_graph`,
`trace_path`) rather than a hand-maintained endpoint table.
