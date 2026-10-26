# backend — gateway

The browser-facing API for the live-video PoC. One of **three independent apps**
that talk over the network **by port, never by file path** (each can run on a
different machine):

```
frontend (browser UI)  ──HTTP/WS──▶  backend (this repo)  ──HTTP──▶  iacore service
```

This app holds **no** detection logic or model deps. The browser opens a
WebSocket here and streams webcam frames; for each frame the gateway forwards the
bytes to the **iacore** service (`/detect`) and relays the boxes back. The "Ask
VLM" button and the options/classes lookups are proxied to iacore too, so the
frontend only ever knows the backend's URL.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # fastapi + uvicorn + httpx (no torch!)
```

## Run

Point it at wherever the iacore service runs, then start it:

```bash
export IACORE_URL=http://localhost:8001      # default; change for a remote iacore
uvicorn app:app --host 0.0.0.0 --port 8000 --reload   # --reload is for development only
```

The `--reload` flag is a development convenience (auto-restart on file changes); the
single-origin HTTPS deploy driven by the ecosystem `run.sh` does not use it.

Check it can reach iacore: <http://localhost:8000/api/health>.

## Config (env, see `.env.example`)

| Var             | Default                  | Meaning                                            |
|-----------------|--------------------------|----------------------------------------------------|
| `IACORE_URL`    | `http://localhost:8001`  | base URL of the iacore inference service           |
| `CORS_ORIGINS`  | `*`                      | comma-separated browser origins allowed            |
| `BACKEND_PORT`  | `8000`                   | informational; pass `--port` to uvicorn            |
| `FRONTEND_DIST` | (unset)                  | path to the built SPA; when set, served on one origin |
| `PILOT_DATA_DIR`| `~/.pilot`               | accounts database, first-run setup code, the camera bridge's device key (0700) |
| `DATABASE_URL`  | SQLite in `PILOT_DATA_DIR` | SQLAlchemy async URL, e.g. Postgres later       |
| `COOKIE_SECURE` | (auto: on over HTTPS)    | force the session cookie's Secure flag behind a TLS proxy |

## Accounts and access

Every route and WebSocket needs a signed-in person (session cookie) or a device key
(`Authorization: Bearer pk_…`); who may call what is the `POLICY` table in `auth.py`, default
deny, and `tests/test_auth.py` fails if a route has no entry. Roles per organization:
viewer (watch) < operator (+ ask the AI, drive, act) < admin (+ robot/video settings, people)
< owner.

- **First run:** with no accounts the gateway writes a one-time setup code to
  `~/.pilot/setup-code`; the app's setup screen asks for it, then creates the company and its
  owner. It also writes the camera bridge's key to `~/.pilot/device-key`.
- **Schema:** Alembic migrations in `migrations/`, applied on startup. New table or column:
  `alembic revision -m "..."` from this directory, then edit the generated file.
- **From the shell** (`pilot_admin.py`): list users, reset a password, create/revoke device keys.
- **Someone forgot their password:** an admin or owner opens Team → *Reset password* and hands
  them the one-time link (24 h). Later, with SMTP, the same link goes out by email.
- **The only owner forgot theirs (break-glass):** on the server, as the account that runs the
  gateway, from this directory:
  `.venv/bin/python pilot_admin.py set-password <email>`. It needs a shell on the server, which
  can already read the database, so it opens nothing new; it signs the person out everywhere
  and leaves an `auth.password_reset` audit row. Team warns while there is only one owner.
- **Dev with the vite server** (`:5173` calls `:8000` directly): start the gateway with
  `CORS_ORIGINS=http://localhost:5173` so the browser may send the session cookie.

## Endpoints

| Method | Path                | Purpose                                                   |
|--------|---------------------|-----------------------------------------------------------|
| WS     | `/ws/detect`        | phone streams JPEG frames in → relay to iacore `/detect` → boxes back |
| WS     | `/ws/view`          | read-only monitor: mirrors the phone's frames + boxes     |
| GET    | `/api/health`       | backend liveness + iacore reachability                    |
| GET    | `/api/options`      | proxy to iacore `/options`                                 |
| GET    | `/api/classes`      | proxy to iacore `/classes`                                 |
| POST   | `/api/vlm`          | proxy to iacore `/vlm`                                     |
| POST   | `/api/vlm/stream`   | streamed VLM answer relay                                  |
| POST   | `/api/transcribe`   | raw audio in → iacore `/transcribe` → text                |
| POST   | `/api/speak`        | text → iacore `/speak` → WAV audio                         |
| GET    | `/api/tts/voices`   | proxy to iacore `/tts/voices` (list neural voices)         |

When `FRONTEND_DIST` is set, the built SPA is also served here (catch-all), so the
whole app lives on one origin. The browser never talks to iacore directly — only
this gateway does. `app.py` runs a session Hub that fans one producer (`/ws/detect`)
out to N monitors (`/ws/view`).
