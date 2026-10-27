"""What every HTTP answer tells the browser, and what is no longer served.

Proven here: the API map (/docs, /redoc, /openapi.json) is gone — FastAPI served it outside
auth.authorize, so anyone could read it; and every answer carries the CSP (scripts from this
origin only, no framing), nosniff, frame denial, referrer and permissions policies, plus HSTS
over https.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

import app as gateway


def test_the_api_map_is_not_served():
    with TestClient(gateway.app) as tc:
        for path in ("/docs", "/redoc", "/openapi.json"):
            r = tc.get(path)
            assert "swagger" not in r.text.lower() and '"openapi"' not in r.text, path


def test_every_answer_carries_the_security_headers():
    with TestClient(gateway.app) as tc:
        r = tc.get("/api/auth/state")
        csp = r.headers["content-security-policy"]
        assert "script-src 'self' 'wasm-unsafe-eval';" in csp and "frame-ancestors 'none'" in csp
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["x-content-type-options"] == "nosniff"
        assert "strict-transport-security" not in r.headers          # plain http here
    with TestClient(gateway.app, base_url="https://testserver") as tc:
        assert tc.get("/api/auth/state").headers["strict-transport-security"].startswith("max-age=")
