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
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Check it can reach iacore: <http://localhost:8000/api/health>.

## Config (env, see `.env.example`)

| Var             | Default                  | Meaning                                            |
|-----------------|--------------------------|----------------------------------------------------|
| `IACORE_URL`    | `http://localhost:8001`  | base URL of the iacore inference service           |
| `CORS_ORIGINS`  | `*`                      | comma-separated browser origins allowed            |
| `BACKEND_PORT`  | `8000`                   | informational; pass `--port` to uvicorn            |
| `FRONTEND_DIST` | (unset)                  | path to the built SPA; when set, served on one origin |

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
