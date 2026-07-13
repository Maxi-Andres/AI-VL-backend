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
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

IACORE_URL = os.environ.get("IACORE_URL", "http://localhost:8001").rstrip("/")
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
FRONTEND_DIST = os.environ.get("FRONTEND_DIST", "").strip()

# Log through uvicorn's configured handler so gateway failures land in the same
# stream as the access logs (uvicorn owns the root logging config).
logger = logging.getLogger("uvicorn.error")

# One shared async client (keep-alive to iacore), built in the lifespan and closed
# on shutdown. Generous read timeout because the VLM call can take many seconds;
# the connect timeout stays short.
client: httpx.AsyncClient


@asynccontextmanager
async def lifespan(app):
    global client
    client = httpx.AsyncClient(
        base_url=IACORE_URL,
        timeout=httpx.Timeout(connect=5.0, read=320.0, write=30.0, pool=5.0),
    )
    try:
        yield
    finally:
        await client.aclose()


app = FastAPI(title="backend — gateway", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Request models. The gateway only relays, so fields are optional and forwarded
# with exclude_none=True — iacore applies its own config.json defaults. Typed
# bodies still reject malformed input at the edge (422) instead of deep in a call.
# --------------------------------------------------------------------------- #
class VlmRequest(BaseModel):
    image: str = ""
    model: str | None = None
    scope: str | None = None
    variant: str | None = None
    max_tokens: int | None = None
    num_ctx: int | None = None
    think: bool | None = None
    prompt: str | None = None


class VlmStreamRequest(BaseModel):
    image: str = ""
    prompt: str = ""
    model: str | None = None
    max_tokens: int | None = None
    num_ctx: int | None = None


class SpeakRequest(BaseModel):
    text: str = ""
    voice: str | None = None


async def proxy_json(call):
    """Await a downstream httpx call and relay its JSON body + status. Map any
    transport failure to a 502 with a clear message (and log it server-side)."""
    try:
        r = await call
    except Exception as e:
        logger.warning("iacore unreachable: %s", e)
        return JSONResponse({"error": f"iacore unreachable: {e}"}, status_code=502)
    return JSONResponse(r.json(), status_code=r.status_code)


# --------------------------------------------------------------------------- #
# Plain proxies (the frontend only ever talks to the backend)
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health():
    try:
        r = await client.get("/health")
        return {"backend": "ok", "iacore": r.json()}
    except Exception as e:
        logger.warning("iacore health check failed: %s", e)
        return JSONResponse({"backend": "ok", "iacore_error": str(e)}, status_code=502)


@app.get("/api/options")
async def options():
    return await proxy_json(client.get("/options"))


@app.get("/api/classes")
async def classes(model: str = ""):
    return await proxy_json(client.get("/classes", params={"model": model}))


@app.post("/api/vlm")
async def vlm(req: VlmRequest):
    return await proxy_json(client.post("/vlm", json=req.model_dump(exclude_none=True)))


@app.post("/api/vlm/stream")
async def vlm_stream(req: VlmStreamRequest):
    """Stream the free-prompt answer through to the browser as it is generated, so
    the UI can show it live and speak it sentence by sentence. Relays iacore's
    text chunks unchanged; still no model deps here."""

    async def gen():
        try:
            async with client.stream(
                "POST", "/vlm/stream", json=req.model_dump(exclude_none=True)
            ) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
        except Exception as e:
            logger.warning("iacore /vlm/stream failed: %s", e)
            yield f"\n[error] iacore unreachable: {e}".encode()

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")


@app.post("/api/transcribe")
async def transcribe(request: Request, language: str = "", translate: bool = False):
    """Speech-to-text proxy: relay the browser's raw audio bytes to iacore's
    /transcribe (like the /ws/detect frame relay) and return the transcript. No
    model deps here — this app stays a pure gateway."""
    data = await request.body()
    params = {"translate": translate}
    if language:
        params["language"] = language
    return await proxy_json(
        client.post(
            "/transcribe",
            content=data,
            headers={"Content-Type": request.headers.get("content-type", "application/octet-stream")},
            params=params,
        )
    )


@app.get("/api/tts/voices")
async def tts_voices():
    return await proxy_json(client.get("/tts/voices"))


@app.post("/api/speak")
async def speak(req: SpeakRequest):
    """Text-to-speech proxy: relay the synthesized audio (or a JSON error) back to
    the browser with iacore's own content type. Still no model deps here."""
    try:
        r = await client.post("/speak", json=req.model_dump(exclude_none=True))
        return Response(
            content=r.content,
            status_code=r.status_code,
            media_type=r.headers.get("content-type", "application/octet-stream"),
        )
    except Exception as e:
        logger.warning("iacore /speak failed: %s", e)
        return JSONResponse({"error": f"iacore unreachable: {e}"}, status_code=502)


# --------------------------------------------------------------------------- #
# Live session hub: one producer (the phone at /ws/detect) + N read-only
# monitors (the server screen at /ws/view). The phone already uploads every
# frame here for detection, so mirroring it to a monitor costs the phone NOTHING
# extra — the gateway just re-sends the bytes it already has, downstream. And it
# only does so WHILE a monitor is attached: with no viewers, fan-out is skipped
# entirely, so an idle monitor uses zero bandwidth.
# --------------------------------------------------------------------------- #
def _put_latest(q: "asyncio.Queue", item) -> None:
    """Enqueue without ever blocking: if the queue is full, drop the oldest item
    first. Keeps the freshest data flowing to a slow consumer."""
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass


class Conn:
    """A live-session connection. EVERYONE (phone + monitors) gets a config queue
    so a config change from any client is pushed to all the others and every UI
    stays in sync. Monitors additionally get a frame queue (bounded to the single
    freshest frame) for the video mirror. A per-connection send lock serializes
    the two sender tasks + the receive loop so their sends never interleave on the
    same socket."""

    def __init__(self, ws: WebSocket, is_viewer: bool):
        self.ws = ws
        self.is_viewer = is_viewer
        self.frame_q = asyncio.Queue(maxsize=1) if is_viewer else None
        self.config_q: asyncio.Queue = asyncio.Queue(maxsize=8)
        self.send_lock = asyncio.Lock()

    async def send(self, msg) -> None:
        async with self.send_lock:
            await self.ws.send_json(msg)


class Hub:
    def __init__(self):
        self.conns: set[Conn] = set()
        # Shared session config. Any client (phone OR a monitor) can change it; the
        # change is pushed to every OTHER client so all UIs stay in sync. The
        # producer reads model/conf/imgsz/classes for every frame; `max_fps` is a
        # client-side capture cap that we only relay (never sent to /detect).
        self.config = {
            "model": None,
            "conf": None,
            "imgsz": None,
            "classes": [],
            "max_fps": None,
        }

    def add(self, conn: Conn) -> None:
        self.conns.add(conn)

    def remove(self, conn: Conn) -> None:
        self.conns.discard(conn)

    def apply_config(self, upd: dict, origin: "Conn | None" = None) -> None:
        """Merge an update into the shared config; if anything actually changed,
        push the new config to every client except the origin. Only broadcasting
        on a real change is what stops the sync from looping between clients."""
        changed = False
        for k in ("model", "conf", "imgsz", "classes", "max_fps"):
            if k in upd and upd[k] is not None and self.config[k] != upd[k]:
                self.config[k] = upd[k]
                changed = True
        if changed:
            msg = {"type": "config", "state": dict(self.config)}
            for c in self.conns:
                if c is not origin:
                    _put_latest(c.config_q, msg)

    def fanout(self, jpeg: bytes, det: dict) -> None:
        payload = None
        for c in self.conns:
            if not c.is_viewer:
                continue
            if payload is None:  # encode once, and only if a monitor is attached
                payload = {
                    "type": "frame",
                    "jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
                    **det,
                }
            _put_latest(c.frame_q, payload)


hub = Hub()


async def _pump(conn: Conn, queue: "asyncio.Queue") -> None:
    """Forward everything from a per-connection queue to the socket until the task
    is cancelled or the socket dies. Cancellation propagates; anything else (e.g. a
    send on a closed socket) ends the loop quietly."""
    try:
        while True:
            await conn.send(await queue.get())
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug("sender loop stopped: %s", e)


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
    conn = Conn(ws, is_viewer=False)
    hub.add(conn)

    task = asyncio.create_task(_pump(conn, conn.config_q))
    try:
        # Seed the phone with the current shared config (so it adopts whatever a
        # monitor may already have set).
        await conn.send({"type": "config", "state": dict(hub.config)})
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
                hub.apply_config(upd, origin=conn)
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
                logger.warning("iacore /detect failed: %s", e)
                await conn.send({"type": "error", "message": f"iacore unreachable: {e}"})
                continue

            if isinstance(d, dict) and "error" in d:
                await conn.send({"type": "error", "message": d["error"]})
            else:
                await conn.send({"type": "detections", **d})
                hub.fanout(data, d)
    except WebSocketDisconnect:
        pass
    finally:
        hub.remove(conn)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


# --------------------------------------------------------------------------- #
# Read-only monitor over WebSocket: mirrors the phone's frames + boxes to a
# screen on the server, and can push YOLO config changes into the shared
# session. ATTACHING here is what turns fan-out on; detaching turns it off.
# --------------------------------------------------------------------------- #
@app.websocket("/ws/view")
async def ws_view(ws: WebSocket):
    await ws.accept()
    conn = Conn(ws, is_viewer=True)
    hub.add(conn)

    tasks = [
        asyncio.create_task(_pump(conn, conn.frame_q)),
        asyncio.create_task(_pump(conn, conn.config_q)),
    ]
    try:
        # Seed the monitor UI with the current shared config right away.
        await conn.send({"type": "config", "state": dict(hub.config)})
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
            hub.apply_config(upd, origin=conn)
    except WebSocketDisconnect:
        pass
    finally:
        hub.remove(conn)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


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
