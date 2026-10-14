"""/ws/joint-control: moving the G1's arm joints from the Joints page.

This socket moves the robot, so the property that matters most is its LIFETIME: the relay's
session id stays in this process (never sent to the browser), and closing the socket releases
the arms. Also proven: ops reach the executor with that session attached, a refused `take`
keeps its status, nothing but a `take` goes upstream without a session, and junk is ignored.

A real HTTP server stands in for the executor. Every receive has a deadline (see `recv`):
TestClient's websocket receive blocks forever, and a hung test once took the whole machine.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

import app as gateway

SESSION = "0123456789abcdef"


class FakeExecutor(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_POST(self):
        srv = self.server
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        srv.seen.append(body)
        op = body.get("op")
        if op == "take" and srv.refuse:
            code, out = 409, {"ok": False, "reply": "err fsm 1 not allowed", "state": "off"}
        elif op == "take":
            code, out = 200, {"ok": True, "session": SESSION, "state": "up", "w": 0.0,
                              "allowed": ["left_elbow", "right_elbow"]}
        elif body.get("session") != SESSION:
            code, out = 409, {"ok": False, "error": "another page holds the arms"}
        elif op == "release":
            code, out = 200, {"ok": True, "reply": "release", "state": "down", "w": 1.0}
        else:
            if op == "target":
                time.sleep(srv.slow)
            code, out = 200, {"ok": True, "reply": op, "state": "active", "w": 1.0}
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def executor(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeExecutor)
    srv.seen = []
    srv.refuse = False
    srv.slow = 0.0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(gateway, "EXECUTOR_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    yield srv
    srv.shutdown()


@pytest.fixture
def tc():
    with TestClient(gateway.app) as client:
        yield client


def recv(ws, timeout: float = 5.0) -> dict:
    box: dict = {}
    t = threading.Thread(target=lambda: box.update(msg=ws.receive_text()), daemon=True)
    t.start()
    t.join(timeout)
    assert "msg" in box, f"no message within {timeout} s"
    return json.loads(box["msg"])


def _wait(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_take_keeps_the_session_here_and_ops_carry_it(tc, executor):
    with tc.websocket_connect("/ws/joint-control?robot=g1") as ws:
        ws.send_text(json.dumps({"op": "take"}))
        out = recv(ws)
        assert out["op"] == "take" and out["status"] == 200 and out["ok"]
        assert "session" not in out, "the session id never reaches the browser"
        assert out["allowed"] == ["left_elbow", "right_elbow"]
        ws.send_text(json.dumps({"op": "target", "joint": "left_elbow", "q": 1.1}))
        out = recv(ws)
        assert out["op"] == "target" and out["status"] == 200
        assert executor.seen[-1] == {"op": "target", "joint": "left_elbow", "q": 1.1,
                                     "session": SESSION}
        ws.send_text(json.dumps({"op": "release"}))
        assert recv(ws)["state"] == "down"
        # The hand-back ramp: keepalives still carry the session until the relay says "off".
        ws.send_text(json.dumps({"op": "keepalive"}))
        assert recv(ws)["status"] == 200
        assert executor.seen[-1] == {"op": "keepalive", "session": SESSION}


def test_closing_the_socket_releases_the_arms(tc, executor):
    with tc.websocket_connect("/ws/joint-control?robot=g1") as ws:
        ws.send_text(json.dumps({"op": "take"}))
        assert recv(ws)["status"] == 200
    assert _wait(lambda: executor.seen and executor.seen[-1].get("op") == "release")
    assert executor.seen[-1]["session"] == SESSION


def test_no_release_sent_when_nothing_was_taken(tc, executor):
    with tc.websocket_connect("/ws/joint-control?robot=g1") as ws:
        ws.send_text(json.dumps({"op": "keepalive"}))
        out = recv(ws)
        assert out["status"] == 409 and out["error"] == "no control session"
    time.sleep(0.3)
    assert executor.seen == [], "nothing but a take goes upstream without a session"


def test_a_refused_take_keeps_its_status_and_holds_nothing(tc, executor):
    executor.refuse = True
    with tc.websocket_connect("/ws/joint-control?robot=g1") as ws:
        ws.send_text(json.dumps({"op": "take"}))
        out = recv(ws)
        assert out["status"] == 409 and not out["ok"] and "not allowed" in out["reply"]
    time.sleep(0.3)
    assert [b["op"] for b in executor.seen] == ["take"], "no release for a session never held"


def test_a_burst_of_targets_sends_the_newest(tc, executor):
    executor.slow = 0.3
    with tc.websocket_connect("/ws/joint-control?robot=g1") as ws:
        ws.send_text(json.dumps({"op": "take"}))
        recv(ws)
        for q in (0.5, 0.6, 0.7, 0.8):
            ws.send_text(json.dumps({"op": "target", "joint": "left_elbow", "q": q}))
        recv(ws)
        recv(ws)
        targets = [b["q"] for b in executor.seen if b["op"] == "target"]
        assert targets[-1] == 0.8, "where the slider is now"
        assert len(targets) < 4, "the ones overtaken while waiting were dropped"


def test_junk_is_ignored(tc, executor):
    with tc.websocket_connect("/ws/joint-control?robot=g1") as ws:
        ws.send_text("not json")
        ws.send_text(json.dumps({"op": "damp"}))
        ws.send_text(json.dumps([1, 2]))
        ws.send_text(json.dumps({"op": "take"}))
        assert recv(ws)["op"] == "take"
    assert executor.seen[0] == {"op": "take"}
