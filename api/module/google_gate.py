"""Google Cloud Vision 차단 스위치 (2026-10-11).

Google Cloud 자격 증명은 개인 계정이라 과금되면 안 된다 — 그래서 Vision OCR 은 기본으로 꺼 두고,
환경 변수 GOOGLE_VISION_ENABLED=1 일 때만 쓴다. Gemma(Google AI Studio, GEMINI_API_KEY) 는 이 스위치와 무관하다.
"""
import os

ENV_NAME = "GOOGLE_VISION_ENABLED"
GOOGLE_DISABLED_MSG = ("Google Cloud Vision 은 꺼져 있습니다 (개인 계정 과금 방지) — "
                       "GOOGLE_VISION_ENABLED=1 일 때만 씁니다")


def google_vision_enabled() -> bool:
    try:
        from . import env_loader  # noqa: F401 — .env 를 먼저 읽는다
    except Exception:
        pass
    return os.getenv(ENV_NAME, "").strip().lower() in ("1", "true", "yes", "on")
