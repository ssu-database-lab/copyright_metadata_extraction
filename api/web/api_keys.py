"""API 키 확인 — 공개 서버(dblab)용 (2026-11-17).

API_KEYS 환경 변수가 있을 때만 켜진다. 없으면 지금과 똑같이 동작한다 (로컬 PC · 평가일 콘솔은 그대로).

    API_KEYS="muhayu:<sha256 hex>,other:<sha256 hex>"

서버에는 키의 SHA-256 만 둔다 — 설정이 새도 키는 드러나지 않고, 고객별로 따로 폐기할 수 있다.
고객은 요청마다 `Authorization: Bearer <키>` (또는 `X-API-Key: <키>`) 헤더를 보낸다.

검사하는 경로: /api/* (단, 간소화 콘솔 /test 가 브라우저에서 부르는 /api/test/* 는 제외) 와 /download/*.
열어 두는 경로: /health, /test, /docs, 정적 파일. CORS 사전 요청(OPTIONS)은 검사하지 않는다.
키 만들기: `python3 api/tools/make_api_key.py <고객이름>`.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
from functools import lru_cache
from typing import Dict, Optional

from starlette.responses import JSONResponse

ENV = "API_KEYS"
PROTECTED_PREFIXES = ("/api/", "/download/")
EXEMPT_PREFIXES = ("/api/test/",)      # 간소화 콘솔 — 로그인 없이 쓰기로 함 (2026-11-17)
KEY_PREFIX = "ck_"

logger = logging.getLogger("api_keys")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.strip().encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def parse_keys(raw: Optional[str]) -> Dict[str, str]:
    """'name:hash, name2:hash2' → {hash: name}. 형식이 틀린 항목은 버린다."""
    out: Dict[str, str] = {}
    for part in re.split(r"[,;\n]", raw or ""):
        part = part.strip()
        if ":" not in part:
            continue
        name, digest = (x.strip() for x in part.split(":", 1))
        digest = digest.lower()
        if name and re.fullmatch(r"[0-9a-f]{64}", digest):
            out[digest] = name
    return out


def needs_key(path: str) -> bool:
    return path.startswith(PROTECTED_PREFIXES) and not path.startswith(EXEMPT_PREFIXES)


def presented_key(headers: Dict[str, str]) -> Optional[str]:
    auth = headers.get("authorization") or ""
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    alt = (headers.get("x-api-key") or "").strip()
    return alt or None


def client_for(key: Optional[str], keys: Dict[str, str]) -> Optional[str]:
    """키가 맞으면 고객 이름. 모든 저장 해시와 상수 시간으로 비교한다."""
    if not key:
        return None
    digest = hash_key(key)
    found = None
    for stored, name in keys.items():
        if hmac.compare_digest(stored, digest):
            found = name
    return found


class ApiKeyMiddleware:
    """순수 ASGI 미들웨어 — 스트리밍 응답(stream=true)을 그대로 통과시킨다."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        keys = parse_keys(os.environ.get(ENV))
        path = scope.get("path") or ""
        method = scope.get("method") or ""
        if not keys or method == "OPTIONS" or not needs_key(path):
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers") or []}
        name = client_for(presented_key(headers), keys)
        if name is None:
            logger.warning("API 키 거부 — %s %s (키 없음 또는 불일치)", method, path)
            response = JSONResponse({"detail": "API key missing or invalid"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
            await response(scope, receive, send)
            return
        scope.setdefault("state", {})["api_client"] = name
        logger.info("API 키 확인 — 고객 %s · %s %s", name, method, path)
        await self.app(scope, receive, send)
