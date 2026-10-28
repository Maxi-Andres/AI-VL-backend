"""/ws/robot-ear: the G1's microphone, relayed from the executor to iacore's ear, events back.

Proven here: the robot's audio reaches the ear and the ear's events reach the page; the page's
mute goes on to the ear; every way it can stop ends with ONE {"type": "error"} message that says
why; and a page that leaves releases the upstream stream — which is what switches the robot's
mic off. A real HTTP server stands in for the executor and a fake for iacore's WebSocket.

Every receive runs on a worker thread with a deadline (see `recv`): TestClient's websocket
receive blocks forever, and a hung test once took the whole machine with it.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

import app as gateway

PACKET = b"\x07\x00" * 2560          # 160 ms of 16 kHz s16le


class FakeExecutor(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_GET(self):
        srv = self.server
        srv.requests.append(self.path)
        if srv.mode == "refuse":
            body = json.dumps({"ok": False, "error": "no microphone for robot 'go2'"}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/L16;rate=16000;channels=1")
        self.end_headers()
        try:
            for _ in range(400):
                self.wfile.write(PACKET)
                self.wfile.flush()
                time.sleep(0.02)
        except OSError:
            srv.hung_up.set()


class FakeEar:
    """iacore's /ws/ear: records what it gets, answers with the events it is given."""

    def __init__(self, events):
        self.sent_bytes = 0
        self.texts = []
        self.events = list(events)
        self.heard_audio = asyncio.Event()      # set once three packets have arrived

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def send(self, msg):
        if isinstance(msg, bytes):
            self.sent_bytes += len(msg)
            if self.sent_bytes >= len(PACKET) * 3:
                self.heard_audio.set()
        else:
            self.texts.append(json.loads(msg))

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self.heard_audio.wait()             # answer only once audio has arrived
        if not self.events:
            await asyncio.sleep(3600)
        return json.dumps(self.events.pop(0))


@pytest.fixture
def executor(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeExecutor)
    srv.mode = "stream"
    srv.requests = []
    srv.hung_up = threading.Event()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(gateway, "EXECUTOR_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    yield srv
    srv.shutdown()


@pytest.fixture
def ear(monkeypatch):
    fake = FakeEar([{"type": "speech"}, {"type": "heard", "text": "Bender, qué ves"},
                    {"type": "command", "text": "qué ves"}])
    monkeypatch.setattr(gateway, "_ear_connect", lambda: fake)
    return fake


@pytest.fixture
def tc():
    with TestClient(gateway.app) as client:
        yield client


def recv(ws, timeout: float = 5.0) -> dict:
    box: dict = {}
    t = threading.Thread(target=lambda: box.update(msg=ws.receive_text()), daemon=True)
    t.start()
    t.join(timeout)
    assert "msg" in box, f"no message within {timeout:.1f}s"
    return json.loads(box["msg"])


def test_the_robots_audio_reaches_the_ear_and_its_events_reach_the_page(tc, executor, ear):
    with tc.websocket_connect("/ws/robot-ear?robot=g1") as ws:
        got = [recv(ws) for _ in range(4)]
    assert got[0] == {"type": "listening", "robot": "g1"}
    assert got[-1] == {"type": "command", "text": "qué ves"}
    assert ear.sent_bytes >= 3 * len(PACKET)
    assert executor.requests == ["/audio/mic/stream?robot=g1"]


def test_the_pages_mute_goes_on_to_the_ear(tc, executor, ear):
    """Catches: the robot answering its own voice — the page mutes the ear while it speaks."""
    with tc.websocket_connect("/ws/robot-ear?robot=g1") as ws:
        recv(ws)
        ws.send_text(json.dumps({"type": "mute", "on": True, "barge": True}))
        ws.send_text("not json")
        deadline = time.time() + 3
        while not ear.texts and time.time() < deadline:
            time.sleep(0.02)
    assert ear.texts == [{"type": "mute", "on": True, "barge": True}]


def test_an_executor_refusal_is_explained(tc, executor, ear):
    executor.mode = "refuse"
    with tc.websocket_connect("/ws/robot-ear?robot=go2") as ws:
        msg = recv(ws)
    assert msg == {"type": "error", "error": "no microphone for robot 'go2'", "status": 400}


def test_an_unreachable_ear_is_explained(tc, executor, monkeypatch):
    def refuse():
        raise OSError("connection refused")
    monkeypatch.setattr(gateway, "_ear_connect", refuse)
    with tc.websocket_connect("/ws/robot-ear?robot=g1") as ws:
        msg = recv(ws)
    assert msg["type"] == "error" and "iacore" in msg["error"]


def test_a_page_leaving_releases_the_robots_mic(tc, executor, ear):
    """Catches: the switch turned off in the page while the robot's mic stream stays open —
    and with it the robot's microphone."""
    with tc.websocket_connect("/ws/robot-ear?robot=g1") as ws:
        recv(ws)
    assert executor.hung_up.wait(5), "the upstream stream must close when the page leaves"


# --------------------------------------------------------------------------- #
# One ear per robot (2026-10-19: two tabs both answered, and switching one off left the robot
# listening through the other)
# --------------------------------------------------------------------------- #
@pytest.fixture
def quiet_ears(monkeypatch):
    made = []

    def factory():
        made.append(FakeEar([]))
        return made[-1]
    monkeypatch.setattr(gateway, "_ear_connect", factory)
    return made


def test_a_new_page_takes_the_ear_from_the_old_one(tc, executor, quiet_ears):
    with tc.websocket_connect("/ws/robot-ear?robot=g1") as first:
        assert recv(first)["type"] == "listening"
        with tc.websocket_connect("/ws/robot-ear?robot=g1") as second:
            msg = recv(first)
            assert msg["type"] == "error" and "another page" in msg["error"]
            assert recv(second)["type"] == "listening"


def test_off_from_one_page_switches_the_mic_off_for_every_page(tc, executor, quiet_ears):
    with tc.websocket_connect("/ws/robot-ear?robot=g1") as page:
        assert recv(page)["type"] == "listening"
        page.send_text(json.dumps({"type": "off"}))
        msg = recv(page)
    assert msg["type"] == "error" and "switched off" in msg["error"]
    assert executor.hung_up.wait(5), "off must close the robot's mic stream"
    assert gateway._ears.get("g1") == {}
