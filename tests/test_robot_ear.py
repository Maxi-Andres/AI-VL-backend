"""/ws/robot-ear: the G1's microphone, relayed from the executor to iacore's ear, events back;
and "Bender, ..." handed to the agent, whose reply the robot speaks (robot_ear.py, 2026-11-12).

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

    def do_POST(self):
        srv = self.server
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        if self.path.startswith("/walk/control"):
            body = json.loads(raw or b"{}")
            srv.walk_ops.append(body.get("op"))
            out = {"ok": True, "walk": srv.walk}
        elif self.path.startswith("/audio/say"):
            srv.said.append(len(raw))
            out = {"ok": True, "seconds": 0.05}
        else:
            out = {"ok": False}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
    srv.walk, srv.walk_ops, srv.said = {"state": "success"}, [], []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(gateway, "EXECUTOR_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    yield srv
    srv.shutdown()


@pytest.fixture
def ear(monkeypatch):
    fake = FakeEar([{"type": "speech"}, {"type": "heard", "text": "Bender, qué ves"},
                    {"type": "command", "text": "qué ves"}])
    monkeypatch.setattr(gateway, "_ear_connect", lambda: fake)
    heard = []

    async def to_agent(robot, text):
        heard.append((robot, text))
    monkeypatch.setattr(gateway, "_bender_heard", to_agent)
    fake.to_agent = heard
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
    assert ear.to_agent == [("g1", "qué ves")]        # the agent answers it, not the page


def test_the_pages_mute_goes_on_to_the_ear(tc, executor, ear):
    """Catches: the robot answering its own voice — the page mutes the ear while it speaks."""
    with tc.websocket_connect("/ws/robot-ear?robot=g1") as ws:
        recv(ws)
        ws.send_text(json.dumps({"type": "mute", "on": True, "barge": True}))
        ws.send_text("not json")
        deadline = time.time() + 3
        while len(ear.texts) < 2 and time.time() < deadline:
            time.sleep(0.02)
    # First the ear's state when it opened (nobody speaking), then the page's.
    # ... and when the page leaves, its mute goes with it.
    assert ear.texts[:2] == [{"type": "mute", "on": False, "barge": False},
                             {"type": "mute", "on": True, "barge": True}]


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


def test_two_pages_share_one_ear_and_one_mic_stream(tc, executor, quiet_ears):
    """Since the gateway holds the ear (2026-11-12) pages only watch it — the agent answers what
    it hears, once — so a second page joins instead of taking it over."""
    with tc.websocket_connect("/ws/robot-ear?robot=g1") as first:
        assert recv(first)["type"] == "listening"
        with tc.websocket_connect("/ws/robot-ear?robot=g1") as second:
            assert recv(second)["type"] == "listening"
        assert gateway._EARS["g1"].running(), "one page left: the other still listens"
    assert executor.hung_up.wait(5), "the last page leaving closes the robot's mic stream"
    assert executor.requests == ["/audio/mic/stream?robot=g1"] and len(quiet_ears) == 1


def test_off_from_one_page_switches_the_mic_off_for_every_page(tc, executor, quiet_ears):
    with tc.websocket_connect("/ws/robot-ear?robot=g1") as page:
        assert recv(page)["type"] == "listening"
        page.send_text(json.dumps({"type": "off"}))
        msg = recv(page)
    assert msg["type"] == "error" and "switched off" in msg["error"]
    assert executor.hung_up.wait(5), "off must close the robot's mic stream"
    assert not gateway._EARS["g1"].running()


# --------------------------------------------------------------------------- #
# "Bender, ..." -> the agent, the reply through the robot; and a blocked walk opens the ear
# --------------------------------------------------------------------------- #
class FakeIacore(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        pass

    def _json(self, code, out):
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        srv = self.server
        srv.asked.append((self.path, body))
        if self.path == "/speak":
            wav = b"RIFF" + b"\0" * 40
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)
            return
        if self.path == "/agent/stop":
            srv.busy = False
            self._json(200, {"ok": True})
            return
        if srv.busy:
            self._json(409, {"error": "the robot is already busy with an instruction"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for ev in ({"type": "heard", "text": body.get("text")},
                   {"type": "tool", "tool": "navigate", "args": {"turn_deg": 180}},
                   {"type": "reply", "text": "Listo, me doy la vuelta."}):
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"event: end\ndata: {}\n\n")


@pytest.fixture
def iacore(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeIacore)
    srv.asked, srv.busy = [], False
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(gateway, "IACORE_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    yield srv
    srv.shutdown()


def agent_events(ws, until="reply", n=12):
    got = []
    for _ in range(n):
        msg = recv(ws)
        if msg.get("type") == "agent":
            got.append(msg["event"])
            if msg["event"].get("type") == until:
                return got
    raise AssertionError(f"no agent {until!r} in {got}")


@pytest.fixture
def bender_ear(monkeypatch):
    fake = FakeEar([{"type": "command", "text": "no podés pasar, date la vuelta"}])
    monkeypatch.setattr(gateway, "_ear_connect", lambda: fake)
    return fake


def test_bender_goes_to_the_agent_and_the_robot_says_its_reply(executor, iacore, bender_ear):
    with TestClient(gateway.app) as tc, tc.websocket_connect("/ws/robot-ear?robot=g1") as ws:
        assert recv(ws)["type"] == "listening"
        events = agent_events(ws)
        deadline = time.time() + 3
        while not executor.said and time.time() < deadline:
            time.sleep(0.02)
    assert [e["type"] for e in events] == ["heard", "tool", "reply"]
    assert events[0]["text"] == "no podés pasar, date la vuelta"
    assert ("/agent", {"text": "no podés pasar, date la vuelta"}) in iacore.asked
    assert ("/speak", {"text": "Listo, me doy la vuelta.", "voice": None}) in iacore.asked
    assert executor.said, "the reply is spoken through the robot"
    assert "cancel" not in executor.walk_ops                  # it was not walking


def test_bender_while_blocked_cancels_the_walk_and_tells_the_agent_why(executor, iacore,
                                                                       bender_ear):
    executor.walk = {"state": "running", "blocked": {"reason": "something is in the way"}}
    iacore.busy = True                    # the agent's own walk is the one blocked: take over
    with TestClient(gateway.app) as tc, tc.websocket_connect("/ws/robot-ear?robot=g1") as ws:
        recv(ws)
        agent_events(ws)
    assert "cancel" in executor.walk_ops
    assert ("/agent/stop", {}) in iacore.asked
    asked = [b["text"] for p, b in iacore.asked if p == "/agent"]
    assert asked[-1].startswith("no podés pasar, date la vuelta\n(Context: you were walking")
    assert "blocking your way" in asked[-1]


def test_bender_instruction_says_why_only_when_it_was_walking():
    assert gateway.bender_instruction("qué ves", {"state": "success"}) == "qué ves"
    walking = gateway.bender_instruction("esperá", {"state": "running", "blocked": None})
    assert "walking somewhere; that walk is stopped" in walking


def test_a_blocked_walk_holds_the_ear_and_lets_go_a_while_after():
    hold, last = gateway.blocked_hold({"state": "running", "blocked": {"reason": "x"}}, 100.0, None)
    assert hold and last == 100.0
    hold, last = gateway.blocked_hold({"state": "running", "blocked": None}, 105.0, last)
    assert hold and last == 100.0                               # still within EAR_AFTER_BLOCKED_S
    hold, last = gateway.blocked_hold({"state": "success"}, 100.0 + gateway.EAR_AFTER_BLOCKED_S,
                                      last)
    assert not hold and last is None
    assert gateway.blocked_hold({}, 1.0, None) == (False, None)


def test_the_walk_watch_opens_the_ear_with_no_page_open(executor, quiet_ears, monkeypatch):
    executor.walk = {"state": "running", "blocked": {"reason": "something is in the way"}}
    monkeypatch.setattr(gateway, "WALK_WATCH_S", 0.05)
    with TestClient(gateway.app):
        deadline = time.time() + 5
        while not executor.requests and time.time() < deadline:
            time.sleep(0.02)
        assert executor.requests == ["/audio/mic/stream?robot=g1"]
        assert "blocked" in gateway._EARS["g1"].holds
        executor.walk = {"state": "success"}
        monkeypatch.setattr(gateway, "EAR_AFTER_BLOCKED_S", 0.0)
        assert executor.hung_up.wait(5), "the walk over, the mic is let go"
