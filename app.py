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
    PILOT_DATA_DIR, DATABASE_URL, COOKIE_SECURE — accounts and sessions; see db.py and
                  auth.py.

Every route and WebSocket requires a signed-in person or a device key, per auth.POLICY
(default deny). The only public paths are the sign-in flow and the SPA shell.

Run (from the backend repo root, venv active):
    uvicorn app:app --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from websockets.asyncio.client import connect as ws_connect

import auth
import backup
import chats
import fleet
import memory
import robot_watch
import sso
import team

IACORE_URL = os.environ.get("IACORE_URL", "http://localhost:8001").rstrip("/")
# The robot executor (ROS2 skill executor) — a separate service, usually the
# unitree_ros2 devcontainer on this host (host-networked -> localhost). It turns a
# skill JSON into a real robot command. Off the hot path, so a one-shot client.
EXECUTOR_URL = os.environ.get("EXECUTOR_URL", "http://localhost:8090").rstrip("/")
# The robot camera bridge control (start/stop the robot-camera stream). Usually the
# unitree_ros2 devcontainer on this host (host-networked -> localhost).
CAMERA_CONTROL_URL = os.environ.get("CAMERA_CONTROL_URL", "http://localhost:8091").rstrip("/")
# mediamtx's WebRTC server, for the H.264 drive view (POST /api/whep/{robot}). On this host;
# its certificate is self-signed, which is fine on loopback.
# Frigate, the NVR: it reads EVERY robot's camera at once (the camera bridge streams only the
# robot picked), so the Fleet's thumbnails come from it, one per robot.
FRIGATE_URL = os.environ.get("FRIGATE_URL", "http://127.0.0.1:5000").rstrip("/")
FRIGATE_CAMERAS = {"go2": "go2", "g1": "g1"}   # robot slot -> Frigate camera name
MEDIAMTX_WEBRTC_URL = os.environ.get("MEDIAMTX_WEBRTC_URL", "https://127.0.0.1:8889").rstrip("/")
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()]
FRONTEND_DIST = os.environ.get("FRONTEND_DIST", "").strip()

# Presence (GET /api/presence, see the section at the bottom of this file).
# A robot counts as LIVE while a frame arrived within this window — at the lowest
# selectable capture rate (5 fps) that is still many frames, so a real stream never
# flickers to offline.
ROBOT_CAM_LIVE_WINDOW = 3.0
# Cache window for one presence probe round, shared by every polling browser.
PRESENCE_TTL = 1.5

# Largest body POST /api/detect will accept. A 1080p JPEG off this camera is
# ~200 KB, so 2 MB is generous headroom; what it buys is that an unbounded body
# stops being a memory lever on a public endpoint (engineering standard, §2).
MAX_DETECT_BODY = 2 * 1024 * 1024

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
    await auth.init()
    client = httpx.AsyncClient(
        base_url=IACORE_URL,
        timeout=httpx.Timeout(connect=5.0, read=320.0, write=30.0, pool=5.0),
    )
    watch = robot_watch.start(EXECUTOR_URL)
    backups = backup.start()
    try:
        yield
    finally:
        await backup.stop(backups)
        await robot_watch.stop(watch)
        await client.aclose()


# auth.authorize runs before EVERY route and WebSocket, including the ones added later:
# a route without an auth.POLICY entry is refused (and tests/test_auth.py fails).
# No /docs, /redoc or /openapi.json: FastAPI serves them outside the route dependencies, so
# auth.authorize never saw them and the whole API map was public, signed in or not (found
# 2026-10-27 while preparing to publish Pilot on the internet).
app = FastAPI(title="backend — gateway", lifespan=lifespan,
              dependencies=[Depends(auth.authorize)],
              docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(auth.router)
app.include_router(team.router)
app.include_router(sso.router)
app.include_router(fleet.router)
app.include_router(chats.router)
app.include_router(memory.router)
# Credentials (the session cookie) only for origins named explicitly — never with "*". The
# deploy is same-origin and needs no CORS at all; this is for the vite dev server on :5173,
# which calls :8000 directly: run the gateway with CORS_ORIGINS=http://localhost:5173.
# Headers that tell the browser what this site may do, on every HTTP answer (not WebSockets):
# - CSP: scripts only from this origin (an injected <script> does not run) plus WebAssembly
#   (three's meshopt decoder for the 3D robot); images and calls also to https (map tiles, the
#   address search, mediamtx's WebRTC on its own port); no one may frame Pilot (clickjacking).
# - HSTS on https: once seen, the browser never tries plain http for this name again.
# - nosniff, no referrer outside the site, and the camera/microphone/location for this origin only.
_CSP = ("default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob: https:; media-src 'self' blob: data:; font-src 'self' data:; "
        "connect-src 'self' https: wss:; worker-src 'self' blob:; object-src 'none'; "
        "base-uri 'self'; form-action 'self'; frame-ancestors 'none'")
_SECURITY_HEADERS = [
    (b"content-security-policy", _CSP.encode()),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"same-origin"),
    (b"permissions-policy", b"camera=(self), microphone=(self), geolocation=(self)"),
]


class SecurityHeaders:
    """Pure ASGI (no BaseHTTPMiddleware): it touches only the response's start, so the VLM's
    streamed answers and the WebSockets pass through untouched."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        secure = scope.get("scheme") == "https" or auth.FORCE_SECURE_COOKIE

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                have = {k.lower() for k, _ in headers}
                headers += [(k, v) for k, v in _SECURITY_HEADERS if k not in have]
                if secure:
                    headers.append((b"strict-transport-security", b"max-age=31536000"))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


app.add_middleware(SecurityHeaders)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials="*" not in CORS_ORIGINS,
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
    robot: str | None = None        # whose identity answers (iacore persona_for)


class ChatTurn(BaseModel):
    question: str = Field("", max_length=4000)
    answer: str = Field("", max_length=8000)


class VlmStreamRequest(BaseModel):
    image: str = ""
    prompt: str = ""
    model: str | None = None
    max_tokens: int | None = None
    num_ctx: int | None = None
    robot: str | None = None        # whose identity answers (iacore persona_for)
    # The conversation so far, oldest first; iacore sends the newest turns that fit along.
    history: list[ChatTurn] = Field(default_factory=list, max_length=50)
    # Filled HERE, never by the browser: what the robot knows about itself (robot_watch.state)
    # and what it was asked to remember (memory.notes).
    robot_state: dict | None = Field(default=None, exclude=True)
    robot_notes: list[str] | None = Field(default=None, exclude=True)
    # Set HERE for a signed-in person: iacore also decides whether the message asks to keep a
    # picture or a note, beside its other questions to the model, and says it first (KEEP_MARK).
    keep: bool = Field(default=False, exclude=True)


class SpeakRequest(BaseModel):
    text: str = ""
    voice: str | None = None


class RobotSayRequest(BaseModel):
    robot: str = "go2"
    text: str = ""
    voice: str | None = None


class RobotVolumeRequest(BaseModel):
    robot: str = "go2"
    level: int


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


class RobotNetConfig(BaseModel):
    """Where the robot lives on the network, for the DDS transport. `peers` are the
    robot's IPs for unicast discovery — required whenever the robot is not on the same
    subnet as the executor, since DDS discovery is multicast and multicast does not
    cross a router. Empty list = back to plain multicast."""
    peers: list[str] = []
    iface: str | None = None


class RobotTransportConfig(BaseModel):
    robot: str | None = None       # go2 | g1
    mode: str | None = None        # dds (from this machine) | relay (agent on the robot)
    url: str | None = None         # the robot-side relay, e.g. http://10.1.254.18:8092
    ping_ip: str | None = None     # address probed for the online/offline indicator


class RobotVideoConfig(BaseModel):
    """The robot's LIVE video knobs. Ranges are NOT declared here on purpose: the on-robot
    relay owns them, and a second copy in the gateway would drift from it."""

    robot: str | None = None       # go2 | g1
    # Applied instantly, no restart:
    fps: float | None = None       # cap for HTTP viewers; 0 = every frame
    width: int | None = None       # downscale width; 0 = native, no re-encode at all
    quality: int | None = None     # JPEG quality, only meaningful when width > 0
    h264_qp: int | None = None     # drive view (all-intra H.264): lower = sharper, bigger
    h264_width: int | None = None  # drive view width; the robot derives the 16:9 height
    camera: str | None = None      # front | rs_color | rs_depth: the ONE camera sent
    # Saved only — the gst pipeline reads these at launch, so they need a restart of the
    # robot's video service. The relay refuses them unless persist is set, because a knob
    # that silently does nothing is worse than one that says no.
    bitrate: int | None = None     # H.264 bitrate for the recording branch
    maxfps: int | None = None      # capture rate cap
    idr: int | None = None         # keyframe interval, in frames
    nvr: int | None = None         # 1/0: feed the recording branch at all
    persist: bool | None = None    # also write the robot's video.env


class RobotCameraConfig(BaseModel):
    robot: str | None = None  # go2 | g1 | test — switches the camera source
    fps: float | None = None
    resolution: str | None = None  # native | 720p | 480p | 360p
    quality: int | None = None


async def downstream(service: str, method: str, url: str, read: float = 10.0,
                     write: float = 5.0, **kwargs) -> JSONResponse:
    """One call to the robot executor or the camera bridge, relayed as its JSON + status; a
    502 that names the service when it cannot be reached. Both live on this machine, so a
    connection per call costs nothing — what this saves is nine copies of the same try/except."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=read, write=write, pool=3.0)
        ) as dc:
            r = await dc.request(method, url, **kwargs)
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        logger.warning("%s unreachable: %s", service, e)
        return JSONResponse({"ok": False, "error": f"{service} unreachable: {e}"}, status_code=502)


async def proxy_json(call):
    """Await a downstream httpx call and relay its JSON body + status. Map any
    transport failure to a 502 with a clear message (and log it server-side)."""
    try:
        r = await call
    except Exception as e:
        logger.warning("iacore unreachable: %s", e)
        return JSONResponse({"error": f"iacore unreachable: {e}"}, status_code=502)
    return JSONResponse(r.json(), status_code=r.status_code)


async def _bounded_body(request: Request, limit: int) -> "bytes | None":
    """Read a request body, giving up as soon as it passes `limit` (None then).

    `await request.body()` reads whatever the client sends, with no ceiling. And
    trusting Content-Length is not a fix: it is client-supplied and a chunked
    request has none, so the only honest check is counting bytes as they arrive."""
    size = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


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


KEEP_MARK = b"\x1ekeep "


async def _keep_mark(chunks) -> "tuple[dict | None, bytes]":
    """iacore's first words on a stream with `keep`: its answer {"decided", "keep", "note"}, and
    whatever came after it in the same chunks. (None, what came) when the stream does not start
    with it (an older iacore): memory.decide then falls back on its patterns."""
    head = b""
    async for chunk in chunks:
        head += chunk
        if not KEEP_MARK.startswith(head[:len(KEEP_MARK)]):
            return None, head
        end = head.find(b"\x1e", len(KEEP_MARK))
        if end >= 0:
            try:
                said = json.loads(head[len(KEEP_MARK):end])
            except ValueError:
                said = None
            return said, head[end + 1:]
    return None, head


@app.post("/api/vlm/stream")
async def vlm_stream(req: VlmStreamRequest, request: Request):
    """Stream the free-prompt answer through to the browser as it is generated, so
    the UI can show it live and speak it sentence by sentence. Relays iacore's
    text chunks unchanged; still no model deps here.

    The robot's own state (battery, link) goes along, from this company's robot in that slot:
    iacore says it only when the question asks ("¿cuánta batería tenés?"). So do the notes it
    was asked to keep (memory.py). "Guardá esta imagen" / "acordate que…" are done here, with
    no model: the answer says it was kept."""
    principal = request.scope.get("principal")
    body = req.model_dump(exclude_none=True)
    if principal is not None:
        body["keep"] = True
    if req.robot and principal is not None:
        state = await robot_watch.state(principal.org_id, req.robot)
        if state:
            body["robot_state"] = state
        if notes := await memory.notes(principal.org_id, req.robot):
            body["robot_notes"] = notes

    async def gen():
        try:
            async with client.stream("POST", "/vlm/stream", json=body) as r:
                chunks = r.aiter_bytes()
                if body.get("keep"):
                    said, rest = await _keep_mark(chunks)
                    cmd = memory.decide(req.prompt, said)
                    if cmd:
                        # Kept here, said so, and the model's answer (if any) is not read.
                        yield (await memory.handle(request, req.robot, cmd, req.image,
                                                   req.prompt)).encode()
                        return
                    if rest:
                        yield rest
                async for chunk in chunks:
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
    return await downstream("robot executor", "POST", f"{EXECUTOR_URL}/execute",
                            json=req.model_dump(exclude_none=True), read=20.0)


@app.get("/api/robot-net")
async def robot_net():
    """Current DDS transport of the robot executor (interface + unicast peers)."""
    return await downstream("robot executor", "GET", f"{EXECUTOR_URL}/dds", read=8.0)


@app.post("/api/robot-net")
async def set_robot_net(req: RobotNetConfig):
    """Point the stack at the robot on another network (e.g. its wlan0 on another VLAN).
    The executor persists it and RESTARTS itself to apply it — CycloneDDS reads its
    config once per process — so expect a few seconds of downtime after this returns."""
    return await downstream("robot executor", "POST", f"{EXECUTOR_URL}/dds",
                            json=req.model_dump(exclude_none=True), read=15.0)


# --- Robot reachability ---------------------------------------------------------------- #
# Probed HERE, on the host, not in the executor's container: ICMP needs raw sockets, which
# the container does not have. Cached briefly because the UI polls this and a ping costs a
# round trip.
_PING_CACHE: dict[str, tuple[float, bool]] = {}
_PING_TTL = 4.0


async def _ping(ip: str, timeout: float = 1.2) -> bool:  # noqa: ASYNC109  # passed to robot_watch.ping_rtt, which bounds it with asyncio.wait_for
    """True if the host answers ICMP. One packet, short deadline: this drives a UI dot."""
    if not ip:
        return False
    now = time.monotonic()
    hit = _PING_CACHE.get(ip)
    if hit and now - hit[0] < _PING_TTL:
        return hit[1]
    ok = await robot_watch.ping_rtt(ip, timeout) is not None
    _PING_CACHE[ip] = (now, ok)
    return ok


@app.get("/api/robot-transport")
async def robot_transport():
    """How commands reach each robot: DDS from this machine, or a relay ON the robot.
    DDS only works while the robot shares this machine's subnet; the relay works anywhere."""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=3.0, read=8.0, write=5.0, pool=3.0)
        ) as ec:
            r = await ec.get(f"{EXECUTOR_URL}/transport")
        data = r.json()
        # Enrich with reachability so the UI can show online/offline per robot without a
        # second endpoint. Probed in parallel: two sequential pings would be visible lag.
        transports = data.get("transports") or {}
        names = list(transports)
        results = await asyncio.gather(
            *(_ping(transports[n].get("ping_ip", "")) for n in names))
        for name, online in zip(names, results, strict=True):  # gather preserves order and length
            transports[name]["online"] = online
        return JSONResponse(data, status_code=r.status_code)
    except Exception as e:
        logger.warning("robot executor unreachable: %s", e)
        return JSONResponse(
            {"ok": False, "error": f"robot executor unreachable: {e}"}, status_code=502)


@app.post("/api/robot-transport")
async def set_robot_transport(req: RobotTransportConfig):
    """Switch a robot between DDS and the on-robot relay. The executor persists it and
    RESTARTS to apply it, so expect a few seconds of downtime after this returns."""
    return await downstream("robot executor", "POST", f"{EXECUTOR_URL}/transport",
                            json=req.model_dump(exclude_none=True), read=15.0)


# The H.264/WebRTC drive view, negotiated through here instead of with mediamtx directly.
# The browser used to POST its offer to mediamtx's own origin (:8889), whose self-signed
# certificate it refused with no prompt until someone opened that address once and accepted
# it, per browser ("Open this once and accept the certificate"); from the internet :8889 is
# not reachable at all, and mediamtx answers anyone who can. Here the offer rides Pilot's
# origin and certificate, behind its sign-in. Only the signalling: the video itself still
# flows mediamtx -> browser over ICE, so outside the office it needs mediamtx to announce an
# address reachable from there (docs/PUBLIC_ACCESS.md).
WHEP_PATHS = {"go2": "robot", "g1": "g1"}     # the mediamtx path of each robot's camera
_WHEP_SESSION = re.compile(r"[A-Za-z0-9-]{1,64}")


def _whep_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(verify=False, timeout=httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=3.0))  # noqa: S501 - loopback, mediamtx's own self-signed certificate


@app.post("/api/whep/{robot}")
async def whep_offer(robot: str, request: Request):
    path = WHEP_PATHS.get(robot)
    if path is None:
        return JSONResponse({"error": f"unknown robot {robot!r}"}, status_code=404)
    offer = await request.body()
    if not offer or len(offer) > 64_000:
        return JSONResponse({"error": "expected an SDP offer"}, status_code=400)
    try:
        async with _whep_client() as mc:
            r = await mc.post(f"{MEDIAMTX_WEBRTC_URL}/{path}/whep", content=offer,
                              headers={"Content-Type": "application/sdp"})
    except httpx.HTTPError as e:
        logger.warning("mediamtx unreachable: %s", e)
        return JSONResponse({"error": "video server unreachable"}, status_code=502)
    if r.status_code != 201:
        # 404 when the robot is not publishing: say so, the page falls back on its own.
        return Response(status_code=r.status_code if r.status_code < 500 else 502)
    headers = {}
    session = r.headers.get("location", "").rstrip("/").rsplit("/", 1)[-1]
    if _WHEP_SESSION.fullmatch(session):
        headers["Location"] = f"/api/whep/{robot}/{session}"
    return Response(r.content, status_code=201, media_type="application/sdp", headers=headers)


@app.delete("/api/whep/{robot}/{session}")
async def whep_close(robot: str, session: str):
    path = WHEP_PATHS.get(robot)
    if path is None or not _WHEP_SESSION.fullmatch(session):
        return Response(status_code=404)
    try:
        async with _whep_client() as mc:
            r = await mc.delete(f"{MEDIAMTX_WEBRTC_URL}/{path}/whep/{session}")
    except httpx.HTTPError:
        return Response(status_code=502)
    return Response(status_code=r.status_code if r.status_code < 500 else 502)


@app.get("/api/robot-video")
async def robot_video(robot: str = "go2"):
    """The robot's LIVE video knobs: what is running, what is saved, and the valid ranges.

    Running and saved are reported separately on purpose — editing the robot's video.env
    without restarting the publisher is exactly how this report used to lie."""
    return await downstream("robot executor", "GET", f"{EXECUTOR_URL}/video",
                            params={"robot": robot}, read=8.0)


@app.post("/api/robot-video")
async def set_robot_video(req: RobotVideoConfig):
    """Retune the robot's live view. Applies immediately, with NO restart and no gap in
    the stream; `persist` also writes the robot's video.env so it survives one.

    The gateway does not validate the values: the on-robot relay owns the allowlist and
    the ranges, and duplicating them here would give us two sources of truth for the same
    contract and one of them would drift."""
    return await downstream("robot executor", "POST", f"{EXECUTOR_URL}/video",
                            json=req.model_dump(exclude_none=True), read=10.0)


@app.post("/api/robot-camera/config")
async def robot_camera_config(req: RobotCameraConfig):
    """Reconfigure the shared robot-camera source (fps/resolution/quality) via the
    bridge. One call affects every viewer, since they all watch the same source.
    Declared BEFORE the /{action} route so "config" isn't captured as an action."""
    return await downstream("camera bridge", "POST", f"{CAMERA_CONTROL_URL}/config",
                            json=req.model_dump(exclude_none=True), read=10.0)


@app.post("/api/robot-camera/{action}")
async def robot_camera(action: str):
    """Start/stop the robot camera stream via the camera bridge (in the devcontainer).
    The bridge, while streaming, feeds frames to /ws/robot-cam below."""
    if action not in ("start", "stop"):
        return JSONResponse({"ok": False, "error": "unknown action"}, status_code=400)
    return await downstream("camera bridge", "POST", f"{CAMERA_CONTROL_URL}/{action}", read=10.0)


_frigate_cache: "tuple[float, dict[str, bool] | None]" = (0.0, None)


def _frigate_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_PROBE_TIMEOUT)


async def _frigate_live() -> "dict[str, bool] | None":
    """Which robots' cameras Frigate is receiving now (its capture rate above zero); None when
    Frigate does not answer. Cached briefly: every thumbnail and every presence round asks."""
    global _frigate_cache
    at, cached = _frigate_cache
    if time.monotonic() - at < PRESENCE_TTL:
        return cached
    stats = await _probe_json(f"{FRIGATE_URL}/api/stats")
    cams = (stats or {}).get("cameras")
    live = None if not isinstance(cams, dict) else {
        slot: (cams.get(name) or {}).get("camera_fps", 0) > 0 for slot, name in FRIGATE_CAMERAS.items()}
    _frigate_cache = (time.monotonic(), live)
    return live


@app.get("/api/robot-frame")
async def robot_frame(robot: str = ""):
    """The newest frame from the robot's camera, as JPEG — what the robot "sees" when the voice
    assistant is asked something from a page that shows no video. 503 when the camera bridge
    sent nothing in the last ROBOT_CAM_LIVE_WINDOW seconds: an old frame would be a lie.

    With `robot`, that robot's camera as Frigate sees it, small (a Fleet thumbnail): the bridge
    carries only the robot picked, so the other one never had a picture there. 503 the same
    way when Frigate is not receiving it — Frigate keeps showing the last frame it got."""
    if robot:
        name = FRIGATE_CAMERAS.get(robot)
        if name is None:
            return JSONResponse({"error": f"unknown robot {robot!r}"}, status_code=404)
        if not ((await _frigate_live()) or {}).get(robot):
            return JSONResponse({"error": "no live frame from this robot's camera"}, status_code=503)
        try:
            async with _frigate_client() as fc:
                r = await fc.get(f"{FRIGATE_URL}/api/{name}/latest.jpg", params={"h": 240})
        except httpx.HTTPError:
            return JSONResponse({"error": "the video recorder does not answer"}, status_code=502)
        if r.status_code != 200:
            return JSONResponse({"error": "no frame from the video recorder"}, status_code=502)
        return Response(content=r.content, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    if not robot_cam.last_jpeg or not robot_cam.snapshot()["live"]:
        return JSONResponse({"error": "no live frame from the robot's camera"}, status_code=503)
    return Response(content=robot_cam.last_jpeg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


# A robot's camera as a live MJPEG stream, from Frigate: what the Splunk dashboards show in an
# <img>. They used to point the browser at Frigate itself (http://192.168.20.99:5000), which an
# https page may not load and the internet cannot reach; through Pilot it is https, signed in
# (the session cookie rides along: splunk. and pilot. are the same site) and works from
# anywhere. The rate and size are capped: each viewer is one stream out of the office.
MJPEG_HEIGHTS = (240, 360, 480, 720)
MJPEG_MAX_FPS = 10


@app.get("/api/robot-mjpeg")
async def robot_mjpeg(robot: str, h: int = 480, fps: int = 5):
    name = FRIGATE_CAMERAS.get(robot)
    if name is None:
        return JSONResponse({"error": f"unknown robot {robot!r}"}, status_code=404)
    h = min(MJPEG_HEIGHTS, key=lambda x: abs(x - h))
    fps = max(1, min(MJPEG_MAX_FPS, fps))
    fc = httpx.AsyncClient(timeout=httpx.Timeout(connect=3.0, read=15.0, write=5.0, pool=3.0))
    try:
        upstream = await fc.send(fc.build_request("GET", f"{FRIGATE_URL}/api/{name}",
                                                  params={"fps": fps, "h": h}), stream=True)
    except httpx.HTTPError:
        await fc.aclose()
        return JSONResponse({"error": "the video recorder does not answer"}, status_code=502)
    if upstream.status_code != 200:
        await upstream.aclose()
        await fc.aclose()
        return JSONResponse({"error": "no stream from the video recorder"}, status_code=502)

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        except httpx.HTTPError:
            pass                    # the recorder went away: end the stream, the page shows it
        finally:
            await upstream.aclose()
            await fc.aclose()

    return StreamingResponse(relay(), media_type=upstream.headers.get("content-type", "multipart/x-mixed-replace"),
                             headers={"Cache-Control": "no-store"})


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
# The robot's own speaker. The voice is made HERE (iacore's Piper, the same voice the browser
# plays) and the robot only plays it: one TTS for both outputs, and nothing to install on the
# robot. The executor adds the relay token; the relay validates the WAV and the volume.
# --------------------------------------------------------------------------- #
async def _executor_audio(method: str, path: str, robot: str, content: bytes = b"",
                          ctype: str = "application/json") -> JSONResponse:
    return await downstream("robot executor", method, f"{EXECUTOR_URL}{path}",
                            params={"robot": robot}, content=content or None,
                            headers={"Content-Type": ctype}, read=25.0, write=10.0)


@app.post("/api/robot-audio/say")
async def robot_say(req: RobotSayRequest):
    """Say `text` through the robot's speaker. Replies once the robot has the audio, with
    how long it will play — not when it finishes."""
    text = req.text.strip()
    if not text:
        return JSONResponse({"ok": False, "error": "empty text"}, status_code=400)
    try:
        r = await client.post("/speak", json={"text": text, "voice": req.voice})
    except Exception as e:
        logger.warning("iacore /speak failed: %s", e)
        return JSONResponse({"ok": False, "error": f"iacore unreachable: {e}"},
                            status_code=502)
    if r.status_code != 200 or not r.content.startswith(b"RIFF"):
        return JSONResponse({"ok": False, "error": f"TTS failed: HTTP {r.status_code}"},
                            status_code=502)
    return await _executor_audio("POST", "/audio/say", req.robot, r.content, "audio/wav")


@app.post("/api/robot-audio/stop")
async def robot_audio_stop(robot: str = "go2"):
    return await _executor_audio("POST", "/audio/stop", robot)


@app.get("/api/robot-audio/volume")
async def robot_volume(robot: str = "go2"):
    return await _executor_audio("GET", "/audio/volume", robot)


@app.post("/api/robot-audio/volume")
async def set_robot_volume(req: RobotVolumeRequest):
    return await _executor_audio("POST", "/audio/volume", req.robot,
                                 json.dumps({"level": req.level}).encode())


# --------------------------------------------------------------------------- #
# Live session hub: one producer (the phone at /ws/detect) + N read-only
# monitors (the server screen at /ws/view). The phone already uploads every
# frame here for detection, so mirroring it to a monitor costs the phone NOTHING
# extra — the gateway just re-sends the bytes it already has, downstream. And it
# only does so WHILE a monitor is attached: with no viewers, fan-out is skipped
# entirely, so an idle monitor uses zero bandwidth.
# --------------------------------------------------------------------------- #
# What a frame carries when nobody detected on it. Handed out as a copy, never
# shared, so a consumer that mutates its boxes cannot poison the next frame.
EMPTY_DET = {"objects": [], "n": 0, "elapsed_ms": 0}


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
        # Does THIS viewer want boxes? Per connection and never shared, which is
        # the whole point: the drive view declaring `false` must not turn
        # detection off for the live view. Default off — a viewer that says
        # nothing gets the picture at minimum latency and costs no GPU.
        self.wants_boxes = False
        self.frame_q = asyncio.Queue(maxsize=1) if is_viewer else None
        self.config_q: asyncio.Queue = asyncio.Queue(maxsize=8)
        self.send_lock = asyncio.Lock()
        # Who is on this socket, for the header's "people online" list. Kept by reference,
        # so a name or colour changed in Settings shows on the next poll. Only a signed-in
        # person is someone: a device key (the camera bridge) is a machine, never listed.
        p = ws.scope.get("principal")
        self.person = p if p is not None and p.kind == "user" and p.user_id else None

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
        # producer reads model/conf/imgsz/classes for every frame. `max_fps` caps
        # detections per second: a browser's own camera applies it to what it sends,
        # and /ws/robot-cam applies it here, to how often it calls iacore (never sent
        # to /detect itself).
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

    def web_counts(self) -> dict:
        """How many browsers are on the session right now, split by role: producers
        stream their own camera in on /ws/detect, viewers only watch /ws/view. (The
        robot-camera producer is NOT a hub Conn — see RobotCamState.)"""
        viewers = sum(1 for c in self.conns if c.is_viewer)
        return {
            "producers": len(self.conns) - viewers,
            "viewers": viewers,
            "total": len(self.conns),
        }

    def web_people(self) -> list[dict]:
        """The people behind those browsers, one entry per person however many tabs they
        have open, and what they are doing: streaming their camera in, watching, or both.
        Name, colour and id only — enough for an avatar, never an email or a credential."""
        people: dict[str, dict] = {}
        for c in self.conns:
            p = c.person
            if p is None:
                continue
            e = people.setdefault(p.user_id, {
                "user_id": p.user_id, "name": p.name, "color": p.color or None,
                "streaming": False, "watching": False, "sockets": 0,
            })
            e["sockets"] += 1
            e["watching" if c.is_viewer else "streaming"] = True
        return sorted(people.values(), key=lambda e: (e["name"].casefold(), e["user_id"]))

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

    # Two delivery paths out of one producer, and the split is the whole design:
    # IMMEDIACY on the raw path, PAIRING on the annotated one. Both enqueue the
    # RAW jpeg bytes + boxes (no base64, no JSON here); the per-viewer pump sends
    # the JPEG as a BINARY frame and the boxes as a tiny separate JSON message,
    # which is what keeps an extra viewer from loading the single event loop.

    def wants_boxes(self) -> bool:
        """Is anyone asking for detection right now? When nobody is, the producer
        never calls iacore at all — the cost disappears instead of being tolerated."""
        return any(c.is_viewer and c.wants_boxes for c in self.conns)

    def fanout_raw(self, jpeg: bytes) -> None:
        """Deliver to the viewers that want the picture and nothing else (the drive
        view, and the live view with YOLO off). Nothing may be awaited before this."""
        item = (jpeg, dict(EMPTY_DET))
        for c in self.conns:
            if c.is_viewer and not c.wants_boxes:
                _put_latest(c.frame_q, item)

    def fanout_annotated(self, jpeg: bytes, det: dict) -> None:
        """Deliver a frame together with ITS OWN boxes. The pair is built by the
        caller and travels together, so a viewer can never be shown boxes that
        describe a different moment than the picture under them."""
        item = (jpeg, det)
        for c in self.conns:
            if c.is_viewer and c.wants_boxes:
                _put_latest(c.frame_q, item)


hub = Hub()


async def _detect(jpeg: bytes) -> dict:
    """Run one frame through iacore and return its boxes, or EMPTY on any failure.

    Fails OPEN on purpose: a detector that is down must degrade to "no boxes",
    never to "no picture"."""
    try:
        r = await client.post("/detect", content=jpeg, params=_detect_params(hub.config))
        d = r.json()
    except Exception as e:
        logger.warning("iacore /detect failed: %s", e)
        return dict(EMPTY_DET)
    if isinstance(d, dict) and "error" in d:
        logger.warning("iacore /detect returned an error: %s", d["error"])
        return dict(EMPTY_DET)
    return d


async def _detect_and_fanout(jpeg: bytes) -> None:
    """Detect on ONE frame, then hand that same frame and its boxes to the
    annotated viewers. Runs as a task precisely so the producer loop does not
    await it: awaiting here is what used to put iacore's latency on the drive view."""
    hub.fanout_annotated(jpeg, await _detect(jpeg))


@app.post("/api/detect")
async def detect(request: Request):
    """Boxes for ONE frame the caller already has. JPEG in, detections out.

    It lives here, next to `_detect`, because it answers with the same session
    params (model/conf/imgsz/classes) that the producers use — point it at a frame
    and it is analysed exactly as the live path would analyse it.

    It exists because of the H.264 transport: that picture goes browser <-> mediamtx
    and never reaches this gateway, so the gateway CANNOT pair it. The browser has
    to, and to do that it needs somewhere to send a frame that gives back only the
    answer.

    NOT /ws/detect, and this is a trap worth naming: that endpoint fans the frames
    it receives out to every monitor, so uploading grabs through it would put the
    live machine's screen onto the drive machine's — replacing the robot picture
    for the operator."""
    body = await _bounded_body(request, MAX_DETECT_BODY)
    if body is None:
        return JSONResponse({"error": "frame too large"}, status_code=413)
    return await proxy_json(
        client.post("/detect", content=body, params=_detect_params(hub.config))
    )


class RobotCamState:
    """Liveness of the robot-camera producer AS SEEN BY THIS GATEWAY: how many
    bridge sockets are attached and when a frame last actually arrived. Kept
    separate from what the bridge says about itself, so a hung bridge (or a robot
    that stopped publishing) can't keep claiming it is streaming."""

    def __init__(self):
        self.attached = 0
        self.frames = 0
        self.last_frame = 0.0
        self.last_jpeg = b""          # the newest frame, for /api/robot-frame

    def frame(self, jpeg: bytes = b"") -> None:
        self.frames += 1
        self.last_frame = time.monotonic()
        if jpeg:
            self.last_jpeg = jpeg

    def snapshot(self) -> dict:
        live = bool(self.attached) and (
            time.monotonic() - self.last_frame < ROBOT_CAM_LIVE_WINDOW
        )
        return {"attached": bool(self.attached), "live": live, "frames": self.frames}


robot_cam = RobotCamState()


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
                # The phone awaits its own detection, so this frame is ALREADY
                # paired: annotated viewers get the pair, raw viewers get the
                # picture with no boxes on it.
                hub.fanout_raw(data)
                hub.fanout_annotated(data, d)
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
    principal = ws.scope.get("principal")
    may_tune = principal is not None and "analyze" in principal.permissions
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
            # `boxes` is THIS connection's intent and never enters the shared
            # config. That separation is the fix for the drive view switching
            # detection off for everyone every time it reconnected.
            if isinstance(upd.get("boxes"), bool):
                conn.wants_boxes = upd["boxes"]
            # The shared config (model, threshold, classes…) changes detection for EVERY
            # viewer and costs GPU: it needs "analyze". A viewer only picks their own boxes.
            if may_tune:
                hub.apply_config(upd, origin=conn)
    except WebSocketDisconnect:
        pass
    finally:
        hub.remove(conn)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# --------------------------------------------------------------------------- #
# The DRIVE branch: all-intra H.264, relayed untouched.
#
# A SECOND CHANNEL AND NOT A MODE OF THE FIRST, deliberately. /ws/view carries JPEG and three
# consumers assume it: the browser's <img>, YOLO and the VLM. Putting H.264 in there would
# break all three at once. The two branches exist precisely so the drive view and the
# detectors stop having to be the same picture.
#
# THIS TIER DOES NOT DECODE. It relays bytes: the robot encodes, the browser decodes with
# WebCodecs. Measured 2026-09-21: 4207 B per frame against the JPEG's ~9000 at the same
# quality, and 0.7 ms to decode one in the browser.
#
# WIRE FORMAT: one binary message per frame = 8-byte little-endian double (the robot's clock
# when the camera produced it) followed by the access unit. One message, so a frame and its
# capture time cannot arrive out of order or be paired wrongly under load — which two
# messages would allow, and which would silently corrupt every latency measurement built on
# it.
# --------------------------------------------------------------------------- #
class H264Hub:
    """Viewers of the drive branch. Newest-only per viewer, never a queue."""

    def __init__(self):
        self.viewers: set[asyncio.Queue] = set()
        self.frames = 0
        self.last_at = 0.0

    def fanout(self, message: bytes) -> None:
        self.frames += 1
        self.last_at = time.time()
        for q in self.viewers:
            _put_latest(q, message)


h264_hub = H264Hub()


@app.websocket("/ws/robot-h264")
async def ws_robot_h264(ws: WebSocket):
    """Producer: the camera bridge relays the robot's all-intra H.264 here."""
    await ws.accept()
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data and len(data) > 8:
                h264_hub.fanout(data)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/view-h264")
async def ws_view_h264(ws: WebSocket):
    """Viewer: the drive page. Gets exactly what the robot produced, byte for byte."""
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    h264_hub.viewers.add(q)
    try:
        while True:
            await ws.send_bytes(await q.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        h264_hub.viewers.discard(q)


@app.websocket("/ws/robot-cam")
async def ws_robot_cam(ws: WebSocket):
    """Robot-camera producer (the camera bridge connects here). Each binary JPEG
    frame goes out on TWO paths, and which path a viewer sits on is its own choice.

    RAW viewers — the drive view, and the live view with YOLO off — are served
    FIRST and are never made to wait for anything. This loop does not await
    detection any more: it used to, and that put iacore's cost (12 ms at 480x270,
    25 ms at 1080p, measured) on the drive view of an operator who had not asked
    for boxes and had nothing in the UI telling them why.

    ANNOTATED viewers get a frame only once ITS OWN boxes are ready, so a box can
    never describe a moment the picture does not show. They therefore receive
    fewer frames — the detection rate — which is the intended trade.

    While a detection is in flight the next frames are SKIPPED, not queued. A
    queue is how latency accumulates; skipping is the same discipline
    `_put_latest` already enforces one layer down."""
    await ws.accept()
    robot_cam.attached += 1
    detect_task: asyncio.Task | None = None
    last_detect = float("-inf")
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if not data:
                continue  # ignore any text/config; this producer only sends frames
            robot_cam.frame(data)
            hub.fanout_raw(data)
            if not hub.wants_boxes():
                continue  # nobody asked: iacore is not called at all
            if detect_task is not None and not detect_task.done():
                continue  # one in flight is the cap
            # The panel's Max FPS: at most that many detections a second. Until it is set,
            # the GPU's own pace (one in flight) is the only limit — it used to be the only
            # one always, and the slider did nothing on the robot camera.
            cap = hub.config.get("max_fps") or 0
            now = time.monotonic()
            if cap > 0 and now - last_detect < 1.0 / cap:
                continue
            last_detect = now
            detect_task = asyncio.create_task(_detect_and_fanout(data))
    except WebSocketDisconnect:
        pass
    finally:
        robot_cam.attached = max(0, robot_cam.attached - 1)
        if detect_task is not None and not detect_task.done():
            detect_task.cancel()


# --------------------------------------------------------------------------- #
# Joint state (the Joints page) — READ-ONLY.
#
#   browser --WS /ws/joints--> THIS --HTTP stream--> executor --HTTP stream--> relay on PC2
#
# Each line is the robot's joint state as g1_joint_bridge printed it (robot-command-relay),
# passed through untouched: this gateway does not parse it, so the format has one owner.
# The executor holds the relay token, so it never reaches this process or the browser.
#
# Newest-wins, per viewer: a viewer that falls behind gets the latest line, never a backlog
# (engineering standard §3). Nothing a viewer sends is acted on — there is no command path
# here. Moving joints has its own socket, /ws/joint-control below, so that WATCHING can never
# move anything.
# --------------------------------------------------------------------------- #
# Read timeout well above the relay's 1 s heartbeat: a silent upstream for this long is a
# dead link, not a still robot.
JOINT_STREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=8.0, write=5.0, pool=5.0)


async def _pump_joint_stream(robot: str, q: "asyncio.Queue") -> None:
    """Upstream lines (str) into `q`, newest wins. Always ends with ONE final dict saying why
    the stream stopped, so the page can show it instead of a bare disconnect."""
    final = {"error": "the robot's joint stream ended"}
    try:
        async with (
            httpx.AsyncClient(timeout=JOINT_STREAM_TIMEOUT) as ec,
            ec.stream("GET", f"{EXECUTOR_URL}/joints/stream", params={"robot": robot}) as r,
        ):
            if r.status_code != 200:
                raw = (await r.aread())[:500]
                try:
                    detail = json.loads(raw).get("error") or raw.decode("utf-8", "replace")
                except ValueError:
                    detail = raw.decode("utf-8", "replace")
                final = {"error": detail, "status": r.status_code}
                return
            async for line in r.aiter_lines():
                if line.startswith("{"):
                    _put_latest(q, line)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        final = {"error": f"robot executor unreachable: {e}"}
    finally:
        _put_latest(q, {**final, "final": True})


@app.websocket("/ws/joints")
async def ws_joints(ws: WebSocket, robot: str = "g1"):
    """The Joints page's live feed: one text message per robot tick, newest wins. The last
    message before the socket closes is {"error": ..., "final": true}; the page reconnects."""
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    pump = asyncio.create_task(_pump_joint_stream(robot, q))

    async def send_loop() -> None:
        while True:
            item = await q.get()
            if isinstance(item, dict):          # the final message: say why, then stop
                await ws.send_text(json.dumps(item))
                return
            await ws.send_text(item)

    async def recv_loop() -> None:
        # Only here to notice the browser leaving; what it sends is ignored (read-only).
        while (await ws.receive()).get("type") != "websocket.disconnect":
            pass

    tasks = {asyncio.create_task(send_loop()), asyncio.create_task(recv_loop())}
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        for t in (*tasks, pump):
            t.cancel()
        try:
            await ws.close()
        except RuntimeError:
            pass        # already closed by the browser


# --------------------------------------------------------------------------- #
# The robot's ear (G1): its own microphone, listened to Alexa-style.
#
#   robot mic --relay--> executor --HTTP stream--> THIS --WS binary--> iacore /ws/ear
#   iacore's events (speech / heard / wake / command) --> THIS --WS text--> browser
#
# The page's "robot mic" switch IS this socket: opening it opens the robot's microphone (the
# relay turns mic mode on for an open stream) and closing it closes it, at once. The page's
# {"type": "mute", "on": bool, "barge": bool} goes on to iacore, so the ear is deaf while the
# robot speaks and never answers its own voice — but for its name, with "barge", to interrupt. The socket's last message is {"type": "error", "error": why}.
# Audio only flows here and to iacore; the browser gets text.
# --------------------------------------------------------------------------- #
EAR_TIMEOUT = httpx.Timeout(connect=3.0, read=10.0, write=5.0, pool=3.0)
# ONE ear per robot. Two pages listening (two tabs, a phone and a PC) each answered every
# command, and switching the mic off in one left the robot listening through the other
# (2026-10-19). A new page takes the ear from the old one, and {"type": "off"} from any page
# switches it off for all of them: like a physical switch.
_ears: dict = {}                     # robot -> {asyncio.Event: reason}, one per open socket


def _stop_ears(robot: str, reason: str) -> None:
    for stop in list(_ears.get(robot, {})):
        _ears[robot][stop] = reason
        stop.set()
IACORE_WS_URL = os.environ.get("IACORE_WS_URL", "ws" + IACORE_URL[4:]
                               if IACORE_URL.startswith("http") else IACORE_URL)


def _ear_connect():
    """The WebSocket to iacore's ear. A function so the tests can put a fake there."""
    return ws_connect(f"{IACORE_WS_URL}/ws/ear", max_size=2**20, open_timeout=5)


@app.websocket("/ws/robot-ear")
async def ws_robot_ear(ws: WebSocket, robot: str = "g1"):
    await ws.accept()
    final = {"type": "error", "error": "the robot's microphone stream ended"}
    _stop_ears(robot, "another page took the robot's microphone")
    stop = asyncio.Event()
    _ears.setdefault(robot, {})[stop] = ""
    try:
        async with (
            httpx.AsyncClient(timeout=EAR_TIMEOUT) as ec,
            ec.stream("GET", f"{EXECUTOR_URL}/audio/mic/stream", params={"robot": robot}) as r,
        ):
            if r.status_code != 200:
                raw = (await r.aread())[:500]
                try:
                    detail = json.loads(raw).get("error") or raw.decode("utf-8", "replace")
                except ValueError:
                    detail = raw.decode("utf-8", "replace")
                final = {"type": "error", "error": detail, "status": r.status_code}
                return
            try:
                ear_cm = _ear_connect()
                ear = await ear_cm.__aenter__()
            except Exception as e:
                final = {"type": "error", "error": f"speech-to-text (iacore) unreachable: {e}"}
                return
            try:
                await ws.send_text(json.dumps({"type": "listening", "robot": robot}))
                left = asyncio.Event()

                async def mic_to_ear() -> None:
                    async for chunk in r.aiter_bytes():
                        await ear.send(chunk)

                async def ear_to_browser() -> None:
                    async for msg in ear:
                        await ws.send_text(msg if isinstance(msg, str) else msg.decode())

                async def browser_to_ear() -> None:
                    while True:
                        msg = await ws.receive()
                        if msg.get("type") == "websocket.disconnect":
                            left.set()
                            return
                        try:
                            control = json.loads(msg.get("text") or "")
                        except ValueError:
                            continue
                        if isinstance(control, dict) and control.get("type") == "mute":
                            await ear.send(json.dumps({"type": "mute",
                                                       "on": bool(control.get("on")),
                                                       "barge": bool(control.get("barge"))}))
                        elif isinstance(control, dict) and control.get("type") == "off":
                            _stop_ears(robot, "the robot's microphone was switched off")
                            return

                tasks = {asyncio.create_task(mic_to_ear()),
                         asyncio.create_task(ear_to_browser()),
                         asyncio.create_task(browser_to_ear()),
                         asyncio.create_task(stop.wait())}
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                for t in done:
                    if not left.is_set() and not t.cancelled() and t.exception() is not None:
                        final = {"type": "error", "error": f"the robot's ear stopped: "
                                                           f"{t.exception()}"}
                if stop.is_set():
                    final = {"type": "error", "error": _ears.get(robot, {}).get(stop) or
                             "the robot's microphone was switched off"}
            finally:
                await ear_cm.__aexit__(None, None, None)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        final = {"type": "error", "error": f"robot executor unreachable: {e}"}
    finally:
        _ears.get(robot, {}).pop(stop, None)
        try:
            await ws.send_text(json.dumps(final))
            await ws.close()
        except (RuntimeError, WebSocketDisconnect):
            pass        # the browser already left: nobody to tell


# --------------------------------------------------------------------------- #
# Joint control (the Joints page) — MOVES the robot's arm joints.
#
#   browser --WS /ws/joint-control--> THIS --POST--> executor --POST--> relay --> g1_joint_sender
#
# Every limit is on the robot (src/joint_ctl_core.hpp in robot-command-relay). What this socket
# adds is a LIFETIME: the relay's session id is kept HERE, never sent to the browser, and when
# the socket closes for any reason — tab closed, network gone, page navigated away — the
# session is released. A browser cannot leave an arm held behind it.
#
# The browser sends {"op": "take" | "target" | "hand" | "keepalive" | "status" | "release",
# joint?, q?, side?, finger?, pos?} and gets back {"op", "status", ...the relay's answer}.
# Requests go upstream one at a time. While one is in flight, a newer target for the SAME joint
# (or finger) replaces the waiting one — newest wins, a dragged slider wants where it is now —
# but targets for different joints all go ("everything back to where it started" sends fifteen
# at once; until 2026-10-15 one slot per op kept only the last of them). Keepalives collapse.
# --------------------------------------------------------------------------- #
# `take` waits on the robot (the sender joins DDS and reads it once): up to ~10 s upstream.
JOINT_CONTROL_TIMEOUT = httpx.Timeout(connect=3.0, read=12.0, write=5.0, pool=3.0)
JOINT_CONTROL_OPS = ("take", "target", "hand", "keepalive", "status", "release")


class _JointControlLink:
    """One browser's control socket: its pending requests and the relay session it holds."""

    def __init__(self, robot: str):
        self.robot = robot
        self.session: str | None = None
        self.pending: dict = {}             # slot -> latest request (see _slot)
        self.wake = asyncio.Event()

    @staticmethod
    def _slot(msg: dict) -> tuple:
        if msg["op"] == "target":
            return ("target", str(msg.get("joint")))
        if msg["op"] == "hand":
            return ("hand", str(msg.get("side")), str(msg.get("finger")))
        return (msg["op"],)

    def submit(self, msg: dict) -> None:
        slot = self._slot(msg)
        self.pending.pop(slot, None)        # newest wins, and goes to the back of the queue
        self.pending[slot] = msg
        self.wake.set()

    def next_request(self) -> "dict | None":
        # A release goes first, always; then take; then targets and fingers in the order they
        # came; then the rest.
        for slot in (("release",), ("take",)):
            if slot in self.pending:
                return self.pending.pop(slot)
        for slot in list(self.pending):
            if slot[0] in ("target", "hand"):
                return self.pending.pop(slot)
        for slot in (("keepalive",), ("status",)):
            if slot in self.pending:
                return self.pending.pop(slot)
        return None

    async def call(self, ec: httpx.AsyncClient, msg: dict) -> "tuple[int, dict]":
        body = {k: msg[k] for k in ("op", "joint", "q", "side", "finger", "pos") if k in msg}
        if msg["op"] != "take":
            body["session"] = self.session
        try:
            r = await ec.post(f"{EXECUTOR_URL}/joints/control", params={"robot": self.robot},
                              json=body)
            out = r.json()
            status = r.status_code
        except Exception as e:
            return 502, {"ok": False, "error": f"robot executor unreachable: {e}"}
        if not isinstance(out, dict):
            return 502, {"ok": False, "error": "bad answer from the executor"}
        if msg["op"] == "take" and status == 200 and out.get("session"):
            self.session = out.pop("session")
        out.pop("session", None)
        # Only once the relay says the session is OVER — not when a release is asked for: the
        # hand-back ramp takes ~2 s, and the keepalives during it are how the page sees it end.
        if out.get("state") == "off" or (status == 409 and out.get("error") == "no control session"):
            self.session = None
        return status, out


# Releases in flight, held so the event loop does not drop a task nobody awaits any more.
_JOINT_RELEASES: "set[asyncio.Task]" = set()


async def _release_joint_control(link: _JointControlLink) -> None:
    async with httpx.AsyncClient(timeout=JOINT_CONTROL_TIMEOUT) as ec:
        status, out = await link.call(ec, {"op": "release"})
    logger.info("joint control: socket closed, released (%s %s)", status,
                out.get("reply") or out.get("error"))


@app.websocket("/ws/joint-control")
async def ws_joint_control(ws: WebSocket, robot: str = "g1"):
    """Move the G1's arm joints from the Joints page. Closing this socket releases the arms."""
    await ws.accept()
    link = _JointControlLink(robot)

    async def recv_loop() -> None:
        while True:
            raw = await ws.receive()
            if raw.get("type") == "websocket.disconnect":
                return
            try:
                msg = json.loads(raw.get("text") or "")
            except ValueError:
                continue
            if isinstance(msg, dict) and msg.get("op") in JOINT_CONTROL_OPS:
                link.submit(msg)

    async def work_loop(ec: httpx.AsyncClient) -> None:
        while True:
            await link.wake.wait()
            link.wake.clear()
            while (msg := link.next_request()) is not None:
                if msg["op"] != "take" and link.session is None:
                    await ws.send_text(json.dumps({"op": msg["op"], "status": 409, "ok": False,
                                                   "error": "no control session",
                                                   "state": "off"}))
                    continue
                status, out = await link.call(ec, msg)
                await ws.send_text(json.dumps({"op": msg["op"], "status": status, **out}))

    async with httpx.AsyncClient(timeout=JOINT_CONTROL_TIMEOUT) as ec:
        tasks = {asyncio.create_task(recv_loop()), asyncio.create_task(work_loop(ec))}
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            for t in tasks:
                t.cancel()
            # Whatever ended the socket, the arms go back to the robot's own controller. Its
            # own task, shielded: when the server cancels this handler (a disconnect can do
            # that), the release must still go out rather than be cancelled with it.
            if link.session is not None:
                release = asyncio.create_task(_release_joint_control(link))
                _JOINT_RELEASES.add(release)
                release.add_done_callback(_JOINT_RELEASES.discard)
                try:
                    await asyncio.shield(release)
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                await ws.close()
            except RuntimeError:
                pass


# --------------------------------------------------------------------------- #
# Presence: WHO is attached right now. This gateway is the only process that sees
# every participant (the robot-camera bridge, the browsers on the hub) and can
# reach the sidecars, so it aggregates the facts here and the UI renders one pill
# per participant. Deliberately only FACTS: the gateway does not know the robot
# registry (that lives in iacore), so it reports which robot the bridge selected
# as its source and lets the UI join that against the robot list.
# --------------------------------------------------------------------------- #
_presence_cache: "tuple[float, dict]" = (0.0, {})
_presence_lock = asyncio.Lock()

# Short probe timeouts: presence is polled, so it must never hang a UI.
_PROBE_TIMEOUT = httpx.Timeout(connect=1.0, read=2.0, write=2.0, pool=1.0)


async def _probe_json(url: str) -> "dict | None":
    """GET a small JSON status from a sidecar. None means it did not answer."""
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as c:
            r = await c.get(url)
        d = r.json()
        return d if isinstance(d, dict) else {}
    except Exception:
        return None


# iacore's /status probes the VLM server and runs nvidia-smi (up to ~2.5 s when one of them
# hangs), so it gets more read time than a sidecar's status.
_IACORE_STATUS_TIMEOUT = httpx.Timeout(connect=1.0, read=3.5, write=2.0, pool=1.0)


async def _probe_iacore() -> "dict | None":
    """None: iacore does not answer. Otherwise its /status — whether the VLM server answers and
    how full the GPU is — or {} from an older iacore that only has /health."""
    try:
        r = await client.get("/status", timeout=_IACORE_STATUS_TIMEOUT)
        if r.status_code < 400:
            d = r.json()
            return d if isinstance(d, dict) else {}
        r = await client.get("/health", timeout=_PROBE_TIMEOUT)
        return {} if r.status_code < 500 else None
    except Exception:
        return None


@app.get("/api/presence")
async def presence():
    """Who is connected right now, for the header pills: the robot-camera producer
    (and which robot it is streaming), the browsers on the hub and the people behind
    them, whether the executor / iacore / the VLM server answer, and the graphics cards.

    Cached for PRESENCE_TTL and serialized on a lock so N polling browsers cost ONE
    probe round — the bridge and the executor are single-threaded HTTP servers that
    sit on the robot control path, and must not be hammered."""
    global _presence_cache
    async with _presence_lock:
        ts, cached = _presence_cache
        now = time.monotonic()
        if cached and now - ts < PRESENCE_TTL:
            return cached

        bridge, executor, iacore, cameras = await asyncio.gather(
            _probe_json(f"{CAMERA_CONTROL_URL}/status"),
            _probe_json(f"{EXECUTOR_URL}/health"),
            _probe_iacore(),
            _frigate_live(),
        )
        b, e = bridge or {}, executor or {}
        data = {
            "web": {**hub.web_counts(), "people": hub.web_people()},
            "robot_cam": {
                **robot_cam.snapshot(),
                # `bridge`/`streaming` are the bridge's self-report; `live` above is
                # this gateway's own evidence (frames really arriving).
                "bridge": bridge is not None,
                "robot": b.get("robot"),
                "streaming": bool(b.get("streaming")),
                "fps": b.get("fps"),
                "resolution": b.get("resolution"),
            },
            "executor": {
                "online": executor is not None,
                "default_robot": e.get("default_robot"),
                "safe_mode": e.get("safe_mode"),
                "dry_run": e.get("dry_run"),
            },
            "iacore": {
                "online": iacore is not None,
                # The VLM server (llama-server or Ollama) is its own process: it can be down
                # while iacore answers. None when this iacore cannot tell.
                "vlm": (iacore or {}).get("vlm"),
                # The graphics cards the AI can use, each {vendor, name, used_mb, total_mb}
                # (memory None where the driver does not say); [] = none found, None = this
                # iacore cannot tell (down or older).
                "gpus": (iacore or {}).get("gpus"),
            },
            # Each robot's camera reaching the recorder (Frigate) — every robot at once, unlike
            # robot_cam, which is the one robot the bridge streams. None: Frigate is down.
            "cameras": cameras,
        }
        _presence_cache = (now, data)
        return data


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
        candidate = os.path.normpath(os.path.join(DIST_ABS, full_path))  # noqa: ASYNC240  # one local stat of a served file; an async FS layer is not worth a dependency
        # Serve a real asset if it exists (and stays inside dist), else fall back
        # to index.html for client-side routes (React Router).
        if (
            full_path
            and candidate.startswith(DIST_ABS + os.sep)
            and os.path.isfile(candidate)  # noqa: ASYNC240  # one local stat of a served file; an async FS layer is not worth a dependency
        ):
            return FileResponse(candidate)
        return FileResponse(os.path.join(DIST_ABS, "index.html"))
