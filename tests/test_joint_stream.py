"""/ws/joints: the Joints page's read-only feed, relayed from the executor's /joints/stream.

Proven here: lines arrive as the robot sent them, every way the stream can stop ends with ONE
final message that says why (instead of a bare disconnect the page cannot explain), and a
viewer that leaves releases the upstream request. A real HTTP server stands in for the
executor; no robot, relay or iacore is involved.

Every receive runs on a worker thread with a deadline (see `recv`): TestClient's websocket
receive blocks forever, and a hung test here once took the whole machine with it.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

import app as gateway

LINES = ['{"t":1,"q":[0.1]}', '{"t":2,"q":[0.2]}', '{"stale":true}']


class FakeExecutor(BaseHTTPRequestHandler):
    """Serves /joints/stream the way the executor does. `mode` picks the behaviour."""

    def log_message(self, *_a):
        pass

    def do_GET(self):
        srv = self.server
        srv.requests.append(self.path)
        if srv.mode == "refuse":
            body = json.dumps({"ok": False, "error": "G1_RELAY_URL is unset"}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        try:
            for line in LINES:
                self.wfile.write((line + "\n").encode())
                self.wfile.flush()
                time.sleep(0.05)
            while srv.mode == "endless":
                self.wfile.write(b'{"stale":true}\n')
                self.wfile.flush()
                time.sleep(0.05)
        except OSError:
            srv.hung_up.set()           # the gateway dropped the upstream request


@pytest.fixture
def executor(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeExecutor)
    srv.mode = "finite"
    srv.requests = []
    srv.hung_up = threading.Event()
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


def test_lines_arrive_then_one_final_message(tc, executor):
    with tc.websocket_connect("/ws/joints?robot=g1") as ws:
        got = [recv(ws) for _ in range(4)]
    assert got[:3] == [json.loads(line) for line in LINES]
    assert got[3]["final"] is True and "ended" in got[3]["error"]
    assert executor.requests == ["/joints/stream?robot=g1"]


def test_an_executor_refusal_is_explained(tc, executor):
    executor.mode = "refuse"
    with tc.websocket_connect("/ws/joints?robot=g1") as ws:
        msg = recv(ws)
    assert msg == {"error": "G1_RELAY_URL is unset", "status": 503, "final": True}


def test_an_unreachable_executor_is_explained(tc, monkeypatch):
    monkeypatch.setattr(gateway, "EXECUTOR_URL", "http://127.0.0.1:9")   # nothing listens
    with tc.websocket_connect("/ws/joints?robot=g1") as ws:
        msg = recv(ws)
    assert msg["final"] is True and "unreachable" in msg["error"]


def test_a_viewer_leaving_releases_the_upstream(tc, executor):
    executor.mode = "endless"
    with tc.websocket_connect("/ws/joints?robot=g1") as ws:
        recv(ws)
    assert executor.hung_up.wait(5), \
        "the upstream stream must be closed when the viewer leaves, or each page reload " \
        "leaves a stream open all the way to the robot"
