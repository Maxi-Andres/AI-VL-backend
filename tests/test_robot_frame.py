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
