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
— just FastAPI + `httpx`. Everything lives in `app.py`. It is more than a plain
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
- **NEVER run `git commit` or `git push`.** Make edits, verify, report; the user
  commits.

## Running & config

See `README.md` for setup/run and the env-var table (`IACORE_URL`, `CORS_ORIGINS`,
`BACKEND_PORT`, `FRONTEND_DIST`); `.env.example` is the source of truth for env
vars. For the current route list, the Hub fan-out, and how the proxy calls map into
iacore, query the **codebase-memory** graph (`get_architecture`, `search_graph`,
`trace_path`) rather than a hand-maintained endpoint table.
