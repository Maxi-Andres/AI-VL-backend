"""Make `app.py` importable from the tests.

The sibling suite (`test_iacore_contract.py`) deliberately parses the source with `ast` and
imports nothing, because it only compares declarations. The pairing suite cannot do that: it
tests RUNTIME behaviour — who gets a frame, when, and how many times iacore is called — so it
drives the real ASGI app through Starlette's `TestClient`.

That needs `fastapi` and `httpx`, both of which are already in `requirements.txt` and both of
which CI installs before running pytest. Nothing else is required: iacore is stubbed in the
suite itself, so no model, no GPU and no robot are involved.

Same tactic as `unitree_ros2/robot_camera_bridge/tests/conftest.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
