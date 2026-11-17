"""API key check for the public server (api/web/api_keys.py, 2026-11-17)."""
import hashlib

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio

import httpx

from web.api_keys import ApiKeyMiddleware, parse_keys

KEY = "ck_muhayu_test-key-1234567890"
OTHER = "ck_other_another-key-0987654321"
ENTRY = f"muhayu:{hashlib.sha256(KEY.encode()).hexdigest()},other:{hashlib.sha256(OTHER.encode()).hexdigest()}"


class _Client:
    """FastAPI 0.109's TestClient needs an older httpx; call the app through httpx's ASGI transport instead."""

    def __init__(self, app):
        self.app = app

    def _call(self, method, url, **kw):
        async def go():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test") as c:
                return await c.request(method, url, **kw)
        return asyncio.run(go())

    def get(self, url, **kw):
        return self._call("GET", url, **kw)

    def post(self, url, **kw):
        return self._call("POST", url, **kw)

    def options(self, url, **kw):
        return self._call("OPTIONS", url, **kw)


def _app():
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware)                       # same order as web/app.py: key check inside CORS
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.post("/api/llm-extract")
    def extract():
        return {"ok": True}

    @app.get("/api/test/status")
    def console_status():
        return {"ok": True}

    @app.get("/download/{rid}")
    def download(rid: str):
        return {"rid": rid}

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.get("/test")
    def console_page():
        return {"page": "test"}

    @app.get("/api/stream")
    def stream():
        return StreamingResponse((f"data: {i}\n\n" for i in range(3)), media_type="text/event-stream")

    return app


def test_without_api_keys_everything_stays_open(monkeypatch):
    monkeypatch.delenv("API_KEYS", raising=False)
    c = _Client(_app())
    assert c.post("/api/llm-extract").status_code == 200
    assert c.get("/download/abc").status_code == 200


def test_protected_paths_need_a_valid_key(monkeypatch):
    monkeypatch.setenv("API_KEYS", ENTRY)
    c = _Client(_app())
    r = c.post("/api/llm-extract")
    assert r.status_code == 401 and r.json() == {"detail": "API key missing or invalid"}
    assert r.headers.get("www-authenticate") == "Bearer"
    assert c.post("/api/llm-extract", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/download/abc").status_code == 401
    assert c.post("/api/llm-extract", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200
    assert c.post("/api/llm-extract", headers={"authorization": f"bearer {OTHER}"}).status_code == 200
    assert c.get("/download/abc", headers={"X-API-Key": KEY}).json() == {"rid": "abc"}


def test_console_health_and_pages_stay_open(monkeypatch):
    monkeypatch.setenv("API_KEYS", ENTRY)
    c = _Client(_app())
    assert c.get("/health").status_code == 200
    assert c.get("/api/test/status").status_code == 200
    assert c.get("/test").status_code == 200


def test_browser_preflight_and_401_carry_cors_headers(monkeypatch):
    monkeypatch.setenv("API_KEYS", ENTRY)
    c = _Client(_app())
    pre = c.options("/api/llm-extract", headers={"Origin": "https://example.org", "Access-Control-Request-Method": "POST",
                                                  "Access-Control-Request-Headers": "authorization"})
    assert pre.status_code == 200 and pre.headers.get("access-control-allow-origin")
    denied = c.post("/api/llm-extract", headers={"Origin": "https://example.org"})
    assert denied.status_code == 401 and denied.headers.get("access-control-allow-origin")


def test_streaming_responses_pass_through(monkeypatch):
    monkeypatch.setenv("API_KEYS", ENTRY)
    c = _Client(_app())
    r = c.get("/api/stream", headers={"Authorization": f"Bearer {KEY}"})
    assert r.status_code == 200 and r.text == "data: 0\n\ndata: 1\n\ndata: 2\n\n"


@pytest.mark.parametrize("raw,n", [("", 0), ("bad", 0), ("x:nothex", 0), (f" a:{'a' * 64} ; b:{'B' * 64} ", 2)])
def test_parse_keys_is_tolerant(raw, n):
    assert len(parse_keys(raw)) == n
