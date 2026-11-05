"""/api/walk and /ws/odom: the Navigation page's walk to a point and its live pose, relayed from
the executor (whose relay runs the loop on the robot). A real HTTP server stands in for the
executor; no robot is involved.

Proven: a goto reaches the executor with its numbers and nothing else, cancel/status carry no
numbers, a bad op never leaves this process, the executor's status is kept, and the odometry
arrives line by line from /odom/stream, not /joints/stream.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

import app as gateway


class FakeExecutor(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        self.server.posts.append((self.path, body))
        code = 409 if self.server.refuse else 200
        out = json.dumps({"ok": code == 200, "walk": {"state": "running"}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        self.server.gets.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for k in range(2):
            self.wfile.write(json.dumps({"x": k, "y": 0, "yaw": 0}).encode() + b"\n")
            self.wfile.flush()


@pytest.fixture
def executor(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeExecutor)
    srv.posts, srv.gets, srv.refuse = [], [], False
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(gateway, "EXECUTOR_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    yield srv
    srv.shutdown()


@pytest.fixture
def tc():
    with TestClient(gateway.app) as client:
        yield client


def recv(ws, timeout: float = 3.0) -> dict:
    box: dict = {}
    t = threading.Thread(target=lambda: box.update(msg=ws.receive_text()), daemon=True)
    t.start()
    t.join(timeout)
    assert "msg" in box, f"no message within {timeout:.1f}s"
    return json.loads(box["msg"])


def test_goto_reaches_the_executor_with_its_numbers(tc, executor):
    r = tc.post("/api/walk", json={"op": "goto", "dx": 2, "dyaw": 0.5})
    assert r.status_code == 200 and r.json()["walk"]["state"] == "running"
    assert executor.posts == [("/walk/control?robot=g1",
                               {"op": "goto", "dx": 2.0, "dy": 0.0, "dyaw": 0.5})]


def test_cancel_and_status_carry_no_numbers(tc, executor):
    tc.post("/api/walk", json={"op": "cancel", "dx": 9})
    tc.get("/api/walk")
    assert [b for _p, b in executor.posts] == [{"op": "cancel"}, {"op": "status"}]


def test_a_bad_op_never_leaves_the_gateway(tc, executor):
    assert tc.post("/api/walk", json={"op": "dance"}).status_code == 422
    assert executor.posts == []


def test_the_executor_status_is_kept(tc, executor):
    executor.refuse = True
    assert tc.post("/api/walk", json={"op": "goto", "dx": 1}).status_code == 409


def test_the_odometry_arrives_from_odom_stream(tc, executor):
    # Newest wins: a line may be skipped when two arrive together, never reordered.
    with tc.websocket_connect("/ws/odom?robot=g1") as ws:
        got = []
        while not (msg := recv(ws)).get("final"):
            got.append(msg["x"])
    assert got and got == sorted(got) and got[-1] == 1
    assert executor.gets == ["/odom/stream?robot=g1"]
