"""downstream(): the one place the gateway calls the robot executor and the camera bridge.
The success path is exercised end to end by test_robot_audio.py (a real fake executor)."""
from __future__ import annotations

import asyncio
import json
import socket

import app as gateway


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_an_unreachable_service_is_a_502_that_names_it():
    url = f"http://127.0.0.1:{_closed_port()}/execute"
    for service in ("robot executor", "camera bridge"):
        r = asyncio.run(gateway.downstream(service, "POST", url, json={"skill": "stand"}))
        body = json.loads(r.body)
        assert r.status_code == 502 and body["ok"] is False
        assert body["error"].startswith(f"{service} unreachable: ")
