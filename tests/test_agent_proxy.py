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
        if self.path == "/agent/stop" or self.path.startswith(("/walk/control", "/execute")):
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
    # The same server stands in for the executor: STOP reaches it too.
    monkeypatch.setattr(gateway, "EXECUTOR_URL", f"http://127.0.0.1:{srv.server_address[1]}")
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
    assert srv.asked == [("/agent", {"text": "agarrá la botella", "history": []})]


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


def test_stop_stops_everything_at_once(iacore):
    """2026-11-13, "tiene que parar todo el stop": the run, the walk on the robot (whoever
    started it) and the robot itself — not only the run, which cancelled the walk later."""
    tc, srv = iacore
    assert tc.post("/api/agent/stop").json() == {"ok": True}
    asked = {path: body for path, body in srv.asked}
    assert "/agent/stop" in asked
    assert asked["/walk/control?robot=g1"] == {"op": "cancel"}
    assert asked["/execute"] == {"robot": "g1", "skill": "stop", "params": {}, "safe_mode": True}


def test_stop_still_ends_the_run_when_the_executor_is_down(iacore, monkeypatch):
    tc, srv = iacore
    monkeypatch.setattr(gateway, "EXECUTOR_URL", "http://127.0.0.1:9")
    assert tc.post("/api/agent/stop").json() == {"ok": True}
    assert [p for p, _b in srv.asked] == ["/agent/stop"]


def test_an_empty_instruction_is_refused(iacore):
    tc, _srv = iacore
    assert tc.post("/api/agent", json={"text": ""}).status_code == 422


def test_the_conversation_so_far_reaches_iacore(iacore):
    """Defect: the gateway's model did not declare `history`, so pydantic dropped it silently."""
    tc, srv = iacore
    earlier = [{"question": "agarrá la botella", "answer": "Listo.", "done": ["pick({}) -> ok"]}]
    tc.post("/api/agent", json={"text": "ahora dejala en la mesa", "history": earlier})
    assert srv.asked[-1] == ("/agent", {"text": "ahora dejala en la mesa", "history": earlier})
