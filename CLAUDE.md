# CLAUDE.md

Guidance for Claude Code when working in the **backend** repo.

## What this is

The **gateway** app: the browser-facing API for the live-video PoC. It is one of
three independent apps that communicate **over the network by port, never by file
path** (each may run on a different machine):

```
frontend (browser UI)  ──HTTP/WS──▶  THIS (backend)  ──HTTP──▶  iacore service
```

- **This repo (backend)**: holds NO detection logic and NO model deps (no
  ultralytics/torch/pillow). It exposes a WebSocket (`/ws/detect`) the browser
  streams frames to, relays each frame to iacore's `/detect`, and relays the boxes
  back. It also proxies `/api/options`, `/api/classes`, and `/api/vlm` so the
  frontend only ever talks to the backend. Everything lives in `app.py`.
- **iacore**: the separate inference service that owns YOLO + the Ollama VLM.
  Reached via the `IACORE_URL` env var. Its own repo.
- **frontend**: the browser UI. Talks only to this gateway. Its own repo.

## Hard boundary

Do **not** import iacore's Python (`src/`, `yolo_common`, …) or reference its files
by path. This app knows iacore only by URL (`IACORE_URL`). Keeping inference out
of this process is the whole point — it stays light and independently deployable.

## Conventions

- **Everything in English — absolutely everything**: comments, docstrings,
  identifiers / function names, user-facing strings, config keys, any shell
  scripts (`*.sh`/`*.ps1`), and the docs/README. The user converses in Spanish
  (Rioplatense) — that is fine for chat ONLY; never put Spanish into code,
  scripts, or docs.
- **NEVER run `git commit` or `git push`.** Make edits, verify, report; the user
  commits.

## Config (env)

- `IACORE_URL` — base URL of the iacore service (default `http://localhost:8001`).
- `CORS_ORIGINS` — comma-separated browser origins allowed (default `*`).
- `BACKEND_PORT` — informational; pass the port to uvicorn yourself.

See `.env.example`.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export IACORE_URL=http://localhost:8001       # where the iacore service runs
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

| Method | Path           | Purpose                                              |
|--------|----------------|------------------------------------------------------|
| WS     | `/ws/detect`   | JPEG frames in → relay to iacore `/detect` → boxes   |
| POST   | `/api/vlm`     | proxy to iacore `/vlm`                                |
| GET    | `/api/options` | proxy to iacore `/options`                           |
| GET    | `/api/classes` | proxy to iacore `/classes`                           |
| GET    | `/api/health`  | backend liveness + iacore reachability               |
