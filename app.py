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

Run (from the backend repo root, venv active):
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import json
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

IACORE_URL = os.environ.get("IACORE_URL", "http://localhost:8001").rstrip("/")
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]

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
# Live YOLO over WebSocket (frames in -> iacore /detect -> boxes out)
# --------------------------------------------------------------------------- #
@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    """Per-connection live detection relay.

    The client sends a JSON text message to (re)configure {model, conf, imgsz,
    classes} and binary JPEG frames. For each frame we POST the bytes to iacore's
    /detect with the current params and relay the JSON reply. The client paces
    itself (one frame in flight), which throttles to the achievable rate.
    """
    await ws.accept()
    state = {"model": None, "conf": None, "imgsz": None, "classes": []}
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
                for k in ("model", "conf", "imgsz", "classes"):
                    if k in upd and upd[k] is not None:
                        state[k] = upd[k]
                await ws.send_json({"type": "config", "state": state})
                continue

            data = msg.get("bytes")
            if not data:
                continue

            params = {}
            if state["model"]:
                params["model"] = state["model"]
            if state["conf"] is not None:
                params["conf"] = state["conf"]
            if state["imgsz"] is not None:
                params["imgsz"] = state["imgsz"]
            if state["classes"]:
                params["classes"] = ",".join(state["classes"])

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
    except WebSocketDisconnect:
        pass
