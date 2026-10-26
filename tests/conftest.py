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

import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.requests import (
    HTTPConnection,  # module level: FastAPI resolves the override's annotation here
)

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

# The gateway keeps accounts in a database (db.py). Point it at a throwaway directory BEFORE
# anything imports db, so no test ever reads or writes the real ~/.pilot.
os.environ["PILOT_DATA_DIR"] = tempfile.mkdtemp(prefix="pilot-test-")
os.environ.pop("DATABASE_URL", None)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_auth: run against the real auth.authorize instead of a signed-in owner"
    )


@pytest.fixture(autouse=True)
def _signed_in_owner(request):
    """The relay suites test what the gateway does with a request, not who may send it, so
    they run as an owner. tests/test_auth.py opts out with @pytest.mark.real_auth and
    exercises sign-in and the policy for real."""
    if request.node.get_closest_marker("real_auth"):
        yield
        return
    import app as gateway
    import auth

    async def owner(conn: HTTPConnection):
        p = auth.Principal(kind="user", org_id="test", role="owner", credential_id="test",
                           user_id="test", name="Test Owner", email="owner@test.local")
        conn.scope["principal"] = p
        return p

    gateway.app.dependency_overrides[auth.authorize] = owner
    yield
    gateway.app.dependency_overrides.pop(auth.authorize, None)
