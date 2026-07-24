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
# The robot executor (ROS2 skill executor) — a separate service, usually the
# unitree_ros2 devcontainer on this host (host-networked -> localhost). It turns a
# skill JSON into a real robot command. Off the hot path, so a one-shot client.
EXECUTOR_URL = os.environ.get("EXECUTOR_URL", "http://localhost:8090").rstrip("/")
# The robot camera bridge control (start/stop the robot-camera stream). Usually the
# unitree_ros2 devcontainer on this host (host-networked -> localhost).
CAMERA_CONTROL_URL = os.environ.get("CAMERA_CONTROL_URL", "http://localhost:8091").rstrip("/")
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


class CommandRequest(BaseModel):
    text: str = ""
    image: str | None = None
    model: str | None = None
    robot: str | None = None
    num_ctx: int | None = None
    max_tokens: int | None = None


class CommandExecuteRequest(BaseModel):
    robot: str | None = None
    skill: str = ""
    params: dict | None = None
    safe_mode: bool | None = None


class RobotCameraConfig(BaseModel):
    robot: str | None = None  # go2 | g1 | test — switches the camera source
    fps: float | None = None
    resolution: str | None = None  # native | 720p | 480p | 360p
    quality: int | None = None


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


@app.post("/api/command")
async def command(req: CommandRequest):
    """Unitree G1 command interpreter proxy: relay the transcribed text to iacore's
    /command and return the chosen skill JSON. No model deps here — pure gateway."""
    return await proxy_json(client.post("/command", json=req.model_dump(exclude_none=True)))


@app.post("/api/execute")
async def execute(req: CommandExecuteRequest):
    """Forward a chosen skill to the robot executor (ROS2) so the robot acts on it.
    Kept a one-shot httpx call (the executor may live in a different process/container
    than iacore). Relays the executor's JSON + status; 502 if it's unreachable."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=20.0, write=5.0, pool=3.0)
        ) as ec:
            r = await ec.post(f"{EXECUTOR_URL}/execute",
                              json=req.model_dump(exclude_none=True))
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        logger.warning("robot executor unreachable: %s", e)
        return JSONResponse(
            {"ok": False, "error": f"robot executor unreachable: {e}"}, status_code=502)


@app.post("/api/robot-camera/config")
async def robot_camera_config(req: RobotCameraConfig):
    """Reconfigure the shared robot-camera source (fps/resolution/quality) via the
    bridge. One call affects every viewer, since they all watch the same source.
    Declared BEFORE the /{action} route so "config" isn't captured as an action."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=3.0)
        ) as cc:
            r = await cc.post(f"{CAMERA_CONTROL_URL}/config",
                              json=req.model_dump(exclude_none=True))
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        logger.warning("camera bridge unreachable: %s", e)
        return JSONResponse(
            {"ok": False, "error": f"camera bridge unreachable: {e}"}, status_code=502)


@app.post("/api/robot-camera/{action}")
async def robot_camera(action: str):
    """Start/stop the robot camera stream via the camera bridge (in the devcontainer).
    The bridge, while streaming, feeds frames to /ws/robot-cam below."""
    if action not in ("start", "stop"):
        return JSONResponse({"ok": False, "error": "unknown action"}, status_code=400)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=3.0)
        ) as cc:
            r = await cc.post(f"{CAMERA_CONTROL_URL}/{action}")
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        logger.warning("camera bridge unreachable: %s", e)
        return JSONResponse(
            {"ok": False, "error": f"camera bridge unreachable: {e}"}, status_code=502)


@app.get("/api/robot-camera/status")
async def robot_camera_status():
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0)) as cc:
            r = await cc.get(f"{CAMERA_CONTROL_URL}/status")
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"ok": False, "streaming": False,
                             "error": f"camera bridge unreachable: {e}"}, status_code=502)


@app.get("/api/skills")
async def skills(robot: str = ""):
    """Proxy the robot skill catalog so the UI can show the available skills/params
    and the list of robots. `robot` selects the catalog (g1|go2)."""
    return await proxy_json(
        client.get("/skills", params={"robot": robot} if robot else None)
    )


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
def _detect_params(cfg: dict) -> dict:
    """Build the iacore /detect query params from the shared session config.
    Shared by the phone producer (/ws/detect) and the robot-camera producer
    (/ws/robot-cam) so both run YOLO with the same model/conf/imgsz/classes."""
    params: dict = {}
    if cfg.get("model"):
        params["model"] = cfg["model"]
    if cfg.get("conf") is not None:
        params["conf"] = cfg["conf"]
    if cfg.get("imgsz") is not None:
        params["imgsz"] = cfg["imgsz"]
    if cfg.get("classes"):
        params["classes"] = ",".join(cfg["classes"])
    return params


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

    async def send_bytes(self, data: bytes) -> None:
        async with self.send_lock:
            await self.ws.send_bytes(data)


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
            # Master YOLO on/off, shared across clients. Default OFF so the GPU
            # stays idle until someone turns detection on. The producers below
            # read it: /ws/detect is gated client-side; /ws/robot-cam checks it
            # to decide whether to run iacore on the robot frames.
            "enabled": False,
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
        for k in ("model", "conf", "imgsz", "classes", "max_fps", "enabled"):
            if k in upd and upd[k] is not None and self.config[k] != upd[k]:
                self.config[k] = upd[k]
                changed = True
        if changed:
            msg = {"type": "config", "state": dict(self.config)}
            for c in self.conns:
                if c is not origin:
                    _put_latest(c.config_q, msg)

    def fanout(self, jpeg: bytes, det: dict) -> None:
        # Enqueue the RAW jpeg bytes + detections (no base64, no JSON here). The
        # per-viewer frame pump sends the JPEG as a BINARY WebSocket frame and the
        # boxes as a tiny separate JSON message. This removes the per-viewer
        # base64+json.dumps of a huge string that used to hog the single event loop
        # (so extra viewers no longer add latency to control/other viewers).
        item = (jpeg, det)
        for c in self.conns:
            if c.is_viewer:
                _put_latest(c.frame_q, item)


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


async def _pump_frames(conn: Conn) -> None:
    """Send fanned-out frames to a viewer: the boxes as a tiny JSON message, then
    the JPEG as a BINARY WebSocket frame. The det message is sent every frame (even
    empty) so the overlay clears when detection turns off. This is the low-cost
    replacement for the old base64-in-JSON payload."""
    try:
        while True:
            jpeg, det = await conn.frame_q.get()
            await conn.send({"type": "det", **det})
            await conn.send_bytes(jpeg)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.debug("frame sender loop stopped: %s", e)


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

            params = _detect_params(hub.config)

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
        asyncio.create_task(_pump_frames(conn)),
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


@app.websocket("/ws/robot-cam")
async def ws_robot_cam(ws: WebSocket):
    """Robot-camera producer (the camera bridge connects here). Each binary JPEG
    frame is fanned out to the monitors (/ws/view).

    YOLO is OPT-IN and shared: when the session's `enabled` flag is off (the
    default) frames are relayed straight through with EMPTY detections, for a
    minimum-latency view and zero GPU use. When it's on, each frame is first sent
    to iacore's /detect (same model/conf/imgsz/classes as the phone) and fanned
    out WITH the boxes — so the robot video can show detections too."""
    await ws.accept()
    empty = {"objects": [], "n": 0, "elapsed_ms": 0}
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if not data:
                continue  # ignore any text/config; this producer only sends frames
            det = empty
            if hub.config.get("enabled"):
                try:
                    r = await client.post(
                        "/detect", content=data, params=_detect_params(hub.config))
                    d = r.json()
                    if not (isinstance(d, dict) and "error" in d):
                        det = d
                except Exception as e:
                    logger.warning("iacore /detect (robot-cam) failed: %s", e)
            hub.fanout(data, det)
    except WebSocketDisconnect:
        pass


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
