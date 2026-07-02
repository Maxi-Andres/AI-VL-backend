#!/usr/bin/env python3
"""
app.py — backend GATEWAY (one of three independent apps).

The browser-facing API. It is the only app the frontend talks to, and the only
app exposed publicly; the heavy inference lives in a separate `iacore` service
that this gateway reaches over HTTP. The three apps communicate BY PORT, never by
file path, and may each run on a different machine:

    frontend (browser UI)  ──HTTP/WS──▶  THIS (backend)  ──HTTP──▶  iacore service

This app holds NO detection logic and NO model deps (no ultralytics/torch): it
just relays. For the live stream the browser opens a WebSocket here and streams
JPEG frames; for each frame the gateway POSTs the bytes to iacore's `/detect` and
relays the boxes back. The VLM button POSTs a frame to iacore's `/vlm`. The
options/classes calls are proxied so the frontend only ever knows the backend.

Config via env:
    IACORE_URL    base URL of the iacore service (default http://localhost:8001)
    CORS_ORIGINS  comma-separated allowed origins for the browser (default *)
    FRONTEND_DIST optional path to a built frontend (dist/); when set, this app
                  also SERVES that SPA so the app + /api + /ws live on one origin
                  (used for the single-origin HTTPS deploy, e.g. phone access).
                  When unset, this app stays a pure gateway.

Run (from the backend repo root, venv active):
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

IACORE_URL = os.environ.get("IACORE_URL", "http://localhost:8001").rstrip("/")
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
FRONTEND_DIST = os.environ.get("FRONTEND_DIST", "").strip()

# One shared async client (keep-alive to iacore). Generous read timeout because
# the VLM call can take many seconds; the connect timeout stays short.
client = httpx.AsyncClient(
    base_url=IACORE_URL,
    timeout=httpx.Timeout(connect=5.0, read=320.0, write=30.0, pool=5.0),
)


@asynccontextmanager
async def lifespan(app):
    yield
    await client.aclose()


app = FastAPI(title="backend — gateway", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Plain proxies (the frontend only ever talks to the backend)
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health():
    try:
        r = await client.get("/health")
        return {"backend": "ok", "iacore": r.json()}
    except Exception as e:
        return JSONResponse({"backend": "ok", "iacore_error": str(e)}, status_code=502)


@app.get("/api/options")
async def options():
    try:
        r = await client.get("/options")
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"iacore unreachable: {e}"}, status_code=502)


@app.get("/api/classes")
async def classes(model: str = ""):
    try:
        r = await client.get("/classes", params={"model": model})
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"iacore unreachable: {e}"}, status_code=502)


@app.post("/api/vlm")
async def vlm(payload: dict):
    try:
        r = await client.post("/vlm", json=payload)
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"iacore unreachable: {e}"}, status_code=502)


# --------------------------------------------------------------------------- #
# Live session hub: one producer (the phone at /ws/detect) + N read-only
# monitors (the server screen at /ws/view). The phone already uploads every
# frame here for detection, so mirroring it to a monitor costs the phone NOTHING
# extra — the gateway just re-sends the bytes it already has, downstream. And it
# only does so WHILE a monitor is attached: with no viewers, fan-out is skipped
# entirely, so an idle monitor uses zero bandwidth.
# --------------------------------------------------------------------------- #
class Viewer:
    """A single monitor connection. Its bounded queue holds only the freshest
    frame: if the monitor can't keep up, we drop stale frames instead of letting
    it back-pressure (and slow down) the phone's detection loop."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1)


class Hub:
    def __init__(self):
        self.viewers: set[Viewer] = set()
        # Shared YOLO config for the session. Whoever changes it last wins — the
        # phone OR a monitor — and the producer reads it for every frame, so a
        # monitor can drive detection just like the phone does.
        self.config = {"model": None, "conf": None, "imgsz": None, "classes": []}

    def apply_config(self, upd: dict) -> None:
        for k in ("model", "conf", "imgsz", "classes"):
            if k in upd and upd[k] is not None:
                self.config[k] = upd[k]

    def fanout(self, jpeg: bytes, det: dict) -> None:
        if not self.viewers:
            return  # nobody watching -> no extra bandwidth
        payload = {
            "type": "frame",
            "jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
            **det,
        }
        for v in self.viewers:
            if v.queue.full():  # drop the stale frame, keep only the newest
                try:
                    v.queue.get_nowait()
                except Exception:
                    pass
            try:
                v.queue.put_nowait(payload)
            except Exception:
                pass


hub = Hub()


# --------------------------------------------------------------------------- #
# Live YOLO over WebSocket (frames in -> iacore /detect -> boxes out)
# --------------------------------------------------------------------------- #
@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    """Producer: the phone streams frames here.

    The client sends a JSON text message to (re)configure {model, conf, imgsz,
    classes} and binary JPEG frames. For each frame we POST the bytes to iacore's
    /detect with the current params and relay the JSON reply. The client paces
    itself (one frame in flight), which throttles to the achievable rate. Each
    answered frame is also fanned out to any attached monitors.
    """
    await ws.accept()
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            text = msg.get("text")
            if text is not None:
                try:
                    upd = json.loads(text)
                except json.JSONDecodeError:
                    continue
                hub.apply_config(upd)
                await ws.send_json({"type": "config", "state": hub.config})
                continue

            data = msg.get("bytes")
            if not data:
                continue

            params = {}
            if hub.config["model"]:
                params["model"] = hub.config["model"]
            if hub.config["conf"] is not None:
                params["conf"] = hub.config["conf"]
            if hub.config["imgsz"] is not None:
                params["imgsz"] = hub.config["imgsz"]
            if hub.config["classes"]:
                params["classes"] = ",".join(hub.config["classes"])

            try:
                r = await client.post("/detect", content=data, params=params)
                d = r.json()
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"iacore unreachable: {e}"})
                continue

            if isinstance(d, dict) and "error" in d:
                await ws.send_json({"type": "error", "message": d["error"]})
            else:
                await ws.send_json({"type": "detections", **d})
                hub.fanout(data, d)
    except WebSocketDisconnect:
        pass


# --------------------------------------------------------------------------- #
# Read-only monitor over WebSocket: mirrors the phone's frames + boxes to a
# screen on the server, and can push YOLO config changes into the shared
# session. ATTACHING here is what turns fan-out on; detaching turns it off.
# --------------------------------------------------------------------------- #
@app.websocket("/ws/view")
async def ws_view(ws: WebSocket):
    await ws.accept()
    viewer = Viewer(ws)
    hub.viewers.add(viewer)

    async def sender():
        try:
            while True:
                payload = await viewer.queue.get()
                await ws.send_json(payload)
        except Exception:
            pass

    task = asyncio.create_task(sender())
    try:
        # Seed the monitor UI with the current shared config right away.
        await ws.send_json({"type": "config", "state": hub.config})
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text")
            if text is None:
                continue  # monitors never send frames, only config
            try:
                upd = json.loads(text)
            except json.JSONDecodeError:
                continue
            hub.apply_config(upd)
    except WebSocketDisconnect:
        pass
    finally:
        hub.viewers.discard(viewer)
        task.cancel()


# --------------------------------------------------------------------------- #
# Optional: serve a built frontend so the app + /api + /ws share ONE origin.
# Enabled only when FRONTEND_DIST points at a built SPA dir (set by run-phone.sh
# for the single-origin HTTPS deploy a phone can reach). This does NOT couple to
# the frontend repo — it just serves whatever directory it is handed, and is a
# no-op (pure gateway) when FRONTEND_DIST is unset. Registered LAST so the /api
# and /ws routes above always take precedence.
# --------------------------------------------------------------------------- #
if FRONTEND_DIST and os.path.isdir(FRONTEND_DIST):
    DIST_ABS = os.path.abspath(FRONTEND_DIST)

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        if full_path.startswith(("api/", "ws/")):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        candidate = os.path.normpath(os.path.join(DIST_ABS, full_path))
        # Serve a real asset if it exists (and stays inside dist), else fall back
        # to index.html for client-side routes (React Router).
        if (
            full_path
            and candidate.startswith(DIST_ABS + os.sep)
            and os.path.isfile(candidate)
        ):
            return FileResponse(candidate)
        return FileResponse(os.path.join(DIST_ABS, "index.html"))
