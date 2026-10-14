"""/api/robot-audio/*: text becomes iacore's voice, and that voice goes to the robot's speaker.

Proven here: the WAV iacore makes reaches the executor byte for byte as audio/wav, for the
robot asked; nothing reaches the robot when the text is empty or the TTS fails; and the
executor's status codes survive. Real HTTP servers stand in for iacore and the executor.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

import app as gateway

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(range(40))


class FakeIacore(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.server.asked.append(json.loads(self.rfile.read(n)))
        body = WAV if self.server.mode == "ok" else b'{"error": "piper crashed"}'
        self.send_response(200 if self.server.mode == "ok" else 500)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeExecutor(BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def _handle(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        self.server.seen.append((self.command, self.path, self.headers.get("Content-Type"),
                                 self.rfile.read(n)))
        body = json.dumps(self.server.reply).encode()
        self.send_response(self.server.code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = _handle


def _serve(handler, **attrs):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    for k, v in attrs.items():
        setattr(srv, k, v)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def fakes(monkeypatch):
    iacore = _serve(FakeIacore, asked=[], mode="ok")
    executor = _serve(FakeExecutor, seen=[], code=200, reply={"ok": True, "seconds": 1.5})
    monkeypatch.setattr(gateway, "IACORE_URL", f"http://127.0.0.1:{iacore.server_address[1]}")
    monkeypatch.setattr(gateway, "EXECUTOR_URL",
                        f"http://127.0.0.1:{executor.server_address[1]}")
    with TestClient(gateway.app) as tc:
        yield tc, iacore, executor
    iacore.shutdown()
    executor.shutdown()


def test_text_is_spoken_by_the_robot(fakes):
    tc, iacore, executor = fakes
    r = tc.post("/api/robot-audio/say",
                json={"robot": "go2", "text": " hola ", "voice": "es_AR-daniela-high"})
    assert r.status_code == 200 and r.json() == {"ok": True, "seconds": 1.5}
    assert iacore.asked == [{"text": "hola", "voice": "es_AR-daniela-high"}]
    assert executor.seen == [("POST", "/audio/say?robot=go2", "audio/wav", WAV)]


def test_empty_text_never_reaches_the_robot(fakes):
    tc, iacore, executor = fakes
    assert tc.post("/api/robot-audio/say", json={"text": "   "}).status_code == 400
    assert iacore.asked == [] and executor.seen == []


def test_a_failed_tts_never_reaches_the_robot(fakes):
    """The defect this guards: forwarding iacore's JSON error body to the speaker as audio."""
    tc, iacore, executor = fakes
    iacore.mode = "fail"
    r = tc.post("/api/robot-audio/say", json={"text": "hola"})
    assert r.status_code == 502 and "TTS failed" in r.json()["error"]
    assert executor.seen == []


def test_volume_and_stop_go_through_the_executor(fakes):
    tc, _iacore, executor = fakes
    executor.reply = {"ok": True, "volume": 4}
    assert tc.get("/api/robot-audio/volume?robot=go2").json()["volume"] == 4
    tc.post("/api/robot-audio/volume", json={"robot": "go2", "level": 4})
    assert executor.seen[-1] == ("POST", "/audio/volume?robot=go2", "application/json",
                                 b'{"level": 4}')
    tc.post("/api/robot-audio/stop?robot=go2")
    assert executor.seen[-1][:2] == ("POST", "/audio/stop?robot=go2")


def test_the_robots_refusal_keeps_its_status(fakes):
    tc, _iacore, executor = fakes
    executor.code, executor.reply = 400, {"ok": False, "error": "'level' must be between 0 and 10"}
    r = tc.post("/api/robot-audio/volume", json={"level": 11})
    assert r.status_code == 400 and "between 0 and 10" in r.json()["error"]
