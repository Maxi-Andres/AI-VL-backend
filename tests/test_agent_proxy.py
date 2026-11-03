"""/api/agent: iacore's live trace reaches the browser as it is made, and STOP gets through.

A real HTTP server stands in for iacore and streams Server-Sent Events slowly, the way the
agent does while the robot moves.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

import app as gateway


class FakeIacore(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        self.server.asked.append((self.path, json.loads(body or b"{}")))
        if self.path == "/agent/stop":
            out = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        if self.server.busy:
            out = b'{"error": "the robot is already busy with an instruction"}'
            self.send_response(409)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for ev in ({"type": "heard", "text": "agarrá la botella"},
                   {"type": "tool", "tool": "pick", "args": {"object": "la botella"}},
                   {"type": "reply", "text": "Listo."}):
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.05)
        self.wfile.write(b"event: end\ndata: {}\n\n")


@pytest.fixture
def iacore(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeIacore)
    srv.asked, srv.busy = [], False
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(gateway, "IACORE_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    with TestClient(gateway.app) as tc:
        yield tc, srv
    srv.shutdown()


def _events(body):
    return [json.loads(ln[6:]) for ln in body.splitlines() if ln.startswith("data: {\"")]


def test_the_trace_is_relayed_event_by_event(iacore):
    tc, srv = iacore
    with tc.stream("POST", "/api/agent", json={"text": "agarrá la botella"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    assert [e["type"] for e in _events(body)] == ["heard", "tool", "reply"]
    assert srv.asked == [("/agent", {"text": "agarrá la botella"})]


def test_a_busy_robot_is_an_error_in_the_trace(iacore):
    """The defect: a 409 from iacore relayed as an empty stream — the tab would wait forever."""
    tc, srv = iacore
    srv.busy = True
    with tc.stream("POST", "/api/agent", json={"text": "agarrá la botella"}) as r:
        body = "".join(r.iter_text())
    ev = _events(body)
    assert ev[0]["type"] == "error" and "409" in ev[0]["text"] and "event: end" in body


def test_iacore_down_is_an_error_in_the_trace(monkeypatch):
    monkeypatch.setattr(gateway, "IACORE_URL", "http://127.0.0.1:9")
    with TestClient(gateway.app) as tc, \
            tc.stream("POST", "/api/agent", json={"text": "hola"}) as r:
        body = "".join(r.iter_text())
    assert _events(body)[0]["type"] == "error" and "unreachable" in _events(body)[0]["text"]


def test_stop_reaches_iacore(iacore):
    tc, srv = iacore
    assert tc.post("/api/agent/stop").json() == {"ok": True}
    assert srv.asked[-1][0] == "/agent/stop"


def test_an_empty_instruction_is_refused(iacore):
    tc, _srv = iacore
    assert tc.post("/api/agent", json={"text": ""}).status_code == 422
