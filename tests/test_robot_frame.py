"""/api/robot-frame: the newest robot-camera frame, for the voice assistant on pages without
video — and never a stale one. The camera bridge's socket is driven directly; no robot.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app as gateway

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"


@pytest.fixture
def tc(monkeypatch):
    monkeypatch.setattr(gateway, "robot_cam", gateway.RobotCamState())
    with TestClient(gateway.app) as client:
        yield client


def test_no_frame_yet_is_a_503(tc):
    r = tc.get("/api/robot-frame")
    assert r.status_code == 503 and "no live frame" in r.json()["error"]


def test_the_newest_frame_is_served(tc):
    gateway.robot_cam.attached = 1
    gateway.robot_cam.frame(b"old")
    gateway.robot_cam.frame(JPEG)
    r = tc.get("/api/robot-frame")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content == JPEG


def test_a_stale_frame_is_not_passed_off_as_now(tc, monkeypatch):
    """Catches: the robot answering about what it saw minutes ago, after its camera stopped."""
    gateway.robot_cam.attached = 1
    gateway.robot_cam.frame(JPEG)
    gateway.robot_cam.last_frame -= gateway.ROBOT_CAM_LIVE_WINDOW + 1
    assert tc.get("/api/robot-frame").status_code == 503


@pytest.fixture
def frigate(tc, monkeypatch):
    """Frigate stand-in: which cameras it receives, and what each latest.jpg asks for."""
    live = {"go2": True, "g1": True}
    asked = []

    async def frigate_live():
        return live

    def handle(req):
        asked.append(req.url)
        return gateway.httpx.Response(200, content=JPEG)

    monkeypatch.setattr(gateway, "_frigate_live", frigate_live)
    monkeypatch.setattr(gateway, "_frigate_client",
                        lambda: gateway.httpx.AsyncClient(transport=gateway.httpx.MockTransport(handle)))
    return live, asked


def test_each_robot_has_its_own_picture(tc, frigate):
    """Catches: the Fleet showing only the robot the camera bridge streams."""
    _, asked = frigate
    for slot in ("go2", "g1"):
        r = tc.get("/api/robot-frame", params={"robot": slot})
        assert r.status_code == 200 and r.content == JPEG
    assert [u.path for u in asked] == ["/api/go2/latest.jpg", "/api/g1/latest.jpg"]


def test_a_robot_frigate_does_not_receive_is_a_503(tc, frigate):
    """Catches: Frigate's last frame of a robot that went off, shown as live."""
    live, asked = frigate
    live["g1"] = False
    assert tc.get("/api/robot-frame", params={"robot": "g1"}).status_code == 503
    assert tc.get("/api/robot-frame", params={"robot": "b2"}).status_code == 404
    assert asked == []
