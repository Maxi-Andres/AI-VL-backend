"""YOLO must never slow the drive view, and boxes must match their own frame.

Every test here names the defect it catches; the defects are the ones measured on the Go2 over
LTE on 2026-09-16 and written up in `AI-VL-ecosystem/docs/PLAN_YOLO_FRAME_PAIRING.md`.

WHAT WAS WRONG, in one line each:

  * The producer AWAITED `/detect` before fanning anything out, so turning YOLO on anywhere
    added iacore's cost to EVERY viewer — the drive view included, which had not asked for
    boxes and had nothing in the UI saying why it got slower. Measured: +12 ms at 480x270,
    and it scales with the frame (~25 ms at 1080p).
  * `enabled` was ONE shared value any client could write, and the drive page seeded it to
    `false` on every connect — so reloading the drive machine switched detection OFF on the
    live machine.
  * Boxes and picture came from different sources on the H.264 transport: the boxes described
    a moment ~260 ms newer than the frame they were drawn on, so a walking person's box sat
    ahead of them.

NO NETWORK, NO GPU, NO ROBOT. iacore is replaced by `FakeIacore`, which is also the instrument:
it counts the calls and can be held open mid-detection, which is how the "raw viewers are not
waiting" and "in-flight frames are skipped, not queued" claims are proven rather than asserted.

WHY POLLING HELPERS AND NOT `asyncio` PRIMITIVES. `TestClient` runs the app in its own thread
and its own event loop; an `asyncio.Event` set from the test thread would be touching futures
that belong to the other loop. `threading.Event` plus a short async poll is thread-safe and
costs milliseconds.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

import app as gateway

FRAME = b"\xff\xd8pretend-this-is-a-jpeg"


# --------------------------------------------------------------------------- #
# The stub iacore, and the instrument
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


class FakeIacore:
    """Stands in for the iacore HTTP client and records what the gateway asked of it.

    `hold()` makes the next detections block until `release()`, which is what lets a test
    observe the system WHILE a detection is in flight — the only moment at which "the raw path
    did not wait" and "the next frame was skipped" are observable at all.
    """

    def __init__(self):
        self.calls: list[bytes] = []
        self.params: list[dict] = []
        self._entered = threading.Event()   # a detection has started
        self._go = threading.Event()        # ...and may now finish
        self._go.set()

    def hold(self) -> None:
        self._go.clear()
        self._entered.clear()

    def release(self) -> None:
        self._go.set()

    def wait_until_detecting(self, timeout: float = 3.0) -> None:
        assert self._entered.wait(timeout), "no detection was ever started"

    @property
    def detecting(self) -> bool:
        return self._entered.is_set() and not self._go.is_set()

    async def post(self, path: str, content: bytes | None = None, params: dict | None = None,
                   **_kw) -> _Resp:
        assert path == "/detect", f"the suite only stubs /detect, got {path}"
        self.calls.append(content)
        self.params.append(params or {})
        self._entered.set()
        # ASYNC110 wants an asyncio.Event here and it is wrong for this case: the gate is
        # opened from the TEST thread, while this coroutine runs in TestClient's own loop.
        # Setting an asyncio.Event across loops touches futures that belong to the other one.
        # A threading.Event polled briefly is the thread-safe version of the same wait.
        while not self._go.is_set():  # noqa: ASYNC110
            await asyncio.sleep(0.005)
        # The label echoes the frame, which is what makes a mismatched pair detectable:
        # a box that says FRAME-1 drawn on FRAME-2 is the 260 ms desync, in miniature.
        label = (content or b"").decode("latin-1")
        return _Resp({"objects": [{"label": label}], "n": 1, "elapsed_ms": 1})


def until(pred, what: str, timeout: float = 3.0) -> None:
    """Wait for a condition the server thread will make true. Fails loudly, never hangs."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def iacore():
    return FakeIacore()


@pytest.fixture
def tc(iacore):
    """The running gateway with iacore stubbed out.

    `hub` is a module singleton, so the shared config is restored afterwards: a test that
    leaves `conf` set behind would silently change the next one's `/detect` params.
    """
    before = dict(gateway.hub.config)
    with TestClient(gateway.app) as client:
        # The lifespan built a real httpx client and will close it on the way out, so put it
        # back before leaving: swapping it away permanently would leak one per test.
        real = gateway.client
        gateway.client = iacore
        try:
            yield client
        finally:
            gateway.client = real
    gateway.hub.config.update(before)


def open_viewer(stack, tc: TestClient, *, boxes: bool):
    ws = stack.enter_context(tc.websocket_connect("/ws/view"))
    assert ws.receive_json()["type"] == "config"   # the seed every viewer gets on connect
    ws.send_json({"boxes": boxes})
    return ws


def read_frame(ws, timeout: float = 3.0) -> tuple[dict, bytes]:
    """A fanned-out frame is two messages: the boxes, then the JPEG as binary.

    Read on a worker thread with a deadline, because `TestClient`'s websocket receive blocks
    forever. The defect this suite exists to catch is "the frame never arrives, or arrives too
    late" — read it directly and a regression HANGS the suite instead of failing it, which in
    CI is indistinguishable from an infrastructure problem. Verified: putting the awaited
    detection back in the producer turns this into a 3 s failure with a readable message.
    """
    box: dict = {}

    def pull() -> None:
        det = ws.receive_json()
        box["det"] = det
        box["jpeg"] = ws.receive_bytes()

    t = threading.Thread(target=pull, daemon=True)
    t.start()
    t.join(timeout)
    assert "jpeg" in box, f"no frame reached this viewer within {timeout:.1f}s"
    assert box["det"]["type"] == "det"
    return box["det"], box["jpeg"]


# --------------------------------------------------------------------------- #
# §2.2 — the drive view used to switch detection off for everyone
# --------------------------------------------------------------------------- #
def test_drive_viewer_does_not_change_detection_state(tc):
    from contextlib import ExitStack

    with ExitStack() as stack:
        live = open_viewer(stack, tc, boxes=True)
        until(gateway.hub.wants_boxes, "the live viewer to be asking for boxes")

        # The drive view connects and declares what ControlPage declares.
        open_viewer(stack, tc, boxes=False)
        time.sleep(0.05)
        assert gateway.hub.wants_boxes() is True, "the drive view turned detection off"

        # And the old lever must be dead: writing the shared flag cannot reach it either.
        live.send_json({"enabled": False})
        time.sleep(0.05)
        assert gateway.hub.wants_boxes() is True, "`enabled` still drives the producer"


# --------------------------------------------------------------------------- #
# §2.1 — with nobody asking, iacore must not be called AT ALL
# --------------------------------------------------------------------------- #
def test_producer_does_not_detect_when_nobody_wants_boxes(tc, iacore):
    from contextlib import ExitStack

    with ExitStack() as stack:
        drive = open_viewer(stack, tc, boxes=False)
        cam = stack.enter_context(tc.websocket_connect("/ws/robot-cam"))
        until(lambda: not gateway.hub.wants_boxes(), "the viewer's declaration to land")

        cam.send_bytes(FRAME)
        det, jpeg = read_frame(drive)
        assert jpeg == FRAME
        assert det["objects"] == []
        assert iacore.calls == [], "iacore was called although nobody asked for boxes"


# --------------------------------------------------------------------------- #
# §3 — the whole point: the raw path does not wait for the annotated one
# --------------------------------------------------------------------------- #
def test_raw_viewer_gets_the_frame_before_detection_resolves(tc, iacore):
    from contextlib import ExitStack

    with ExitStack() as stack:
        open_viewer(stack, tc, boxes=True)          # someone wants boxes, so detection runs
        drive = open_viewer(stack, tc, boxes=False)
        cam = stack.enter_context(tc.websocket_connect("/ws/robot-cam"))
        until(gateway.hub.wants_boxes, "the live viewer to be asking for boxes")

        iacore.hold()
        cam.send_bytes(FRAME)

        det, jpeg = read_frame(drive)               # arrives with the detection still open
        assert jpeg == FRAME
        assert det["objects"] == []
        assert iacore.detecting, "the drive frame only arrived after detection finished"
        iacore.release()


# --------------------------------------------------------------------------- #
# §2.3 — the 260 ms desync: boxes must describe the frame they are drawn on
# --------------------------------------------------------------------------- #
def test_annotated_frame_is_the_one_that_was_detected(tc, iacore):
    from contextlib import ExitStack

    with ExitStack() as stack:
        live = open_viewer(stack, tc, boxes=True)
        cam = stack.enter_context(tc.websocket_connect("/ws/robot-cam"))
        until(gateway.hub.wants_boxes, "the live viewer to be asking for boxes")

        iacore.hold()
        cam.send_bytes(b"FRAME-1")
        iacore.wait_until_detecting()
        cam.send_bytes(b"FRAME-2")      # newer picture, while FRAME-1 is still being detected
        iacore.release()

        det, jpeg = read_frame(live)
        assert jpeg == b"FRAME-1"
        assert det["objects"] == [{"label": "FRAME-1"}], \
            "the viewer was shown boxes computed on a different frame"


# --------------------------------------------------------------------------- #
# §3 — skip, never queue: a queue is how latency accumulates
# --------------------------------------------------------------------------- #
def test_second_frame_is_skipped_while_a_detection_is_in_flight(tc, iacore):
    from contextlib import ExitStack

    with ExitStack() as stack:
        open_viewer(stack, tc, boxes=True)
        cam = stack.enter_context(tc.websocket_connect("/ws/robot-cam"))
        until(gateway.hub.wants_boxes, "the live viewer to be asking for boxes")

        iacore.hold()
        cam.send_bytes(b"FRAME-1")
        iacore.wait_until_detecting()
        cam.send_bytes(b"FRAME-2")
        cam.send_bytes(b"FRAME-3")
        time.sleep(0.1)
        assert len(iacore.calls) == 1, f"frames were queued, not skipped: {iacore.calls}"
        iacore.release()


# --------------------------------------------------------------------------- #
# §3.3 — the trap: POST /api/detect must answer, and ONLY answer
# --------------------------------------------------------------------------- #
def test_detect_endpoint_answers_without_fanning_the_frame_out(tc, iacore):
    from contextlib import ExitStack

    with ExitStack() as stack:
        drive = open_viewer(stack, tc, boxes=False)
        cam = stack.enter_context(tc.websocket_connect("/ws/robot-cam"))
        cam.send_bytes(FRAME)
        _, jpeg = read_frame(drive)
        assert jpeg == FRAME

        # A browser pairing the H.264 picture locally uploads its own grab here.
        r = tc.post("/api/detect", content=b"BROWSER-GRAB")
        assert r.status_code == 200
        assert r.json()["objects"] == [{"label": "BROWSER-GRAB"}]

        # That grab must NOT reach the drive view. `/ws/detect` would have sent it,
        # replacing the robot picture on the operator's screen.
        cam.send_bytes(b"NEXT-ROBOT-FRAME")
        _, jpeg = read_frame(drive)
        assert jpeg == b"NEXT-ROBOT-FRAME", "the browser's own grab was fanned out to the drive view"


def test_detect_endpoint_refuses_an_unbounded_body(tc, iacore):
    r = tc.post("/api/detect", content=b"x" * (gateway.MAX_DETECT_BODY + 1))
    assert r.status_code == 413
    assert iacore.calls == [], "an oversized body was relayed to iacore anyway"
