"""
Production VLM attribute extractor with an automatic fallback chain.

`compare.py` is the offline benchmark (run both models, score them). This is
the *production* counterpart: one entry point that picks the first reachable
VLM backend and, if a call fails mid-flight, transparently advances to the
next backend in the chain.

Default chain (as of 2026-07-08): Gemma 4 31B via **OpenRouter API** is primary
(운영 결정: API 우선), the lab vLLM Gemma server is second, Qwen3-VL (DashScope)
is third. All three are OpenAI-compatible; an unreachable backend (e.g. no
OPENROUTER_API_KEY yet) fails its ping and the chain advances automatically.
Fully configurable: pass `backends=[...]` or `prefer=` ("gemma"|"openrouter"|
"gemma-local"|"qwen").

NOTE on imports: this module deliberately loads its sibling VLM files
(client.py / prompts.py) by path and reads .env directly, rather than going
through the `api.module` package. Importing anything under `api.module`
triggers `api/module/__init__.py`, which eagerly loads the NER/torch stack
(~60s+ cold start) — a cost a pure VLM caller should never pay. Keeping the
imports lazy/path-based makes this safe to import from the web app or a CLI.

Usage:
    from api.module.clip_extraction.vlm.extractor import VLMExtractor

    ex = VLMExtractor()                 # OpenRouter Gemma → local Gemma → Qwen
    res = ex.extract("photo.jpg")       # VLMResult + .backend_used
    print(res.backend_used, res.parsed.get("work_type"))

    results = ex.extract_batch(["a.jpg", "b.jpg"])

CLI smoke test:
    python api/module/clip_extraction/vlm/extractor.py
    python -m api.module.clip_extraction.vlm.extractor --prefer gemma
    python api/module/clip_extraction/vlm/extractor.py --images /path/to/imgs
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
DEFAULT_IMAGES = THIS_DIR.parent / "test_data" / "sample_works"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Backend definitions — kept in sync with compare.py -----------------------
DASHSCOPE_BASE = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
DEFAULT_GEMMA_URL = "http://127.0.0.1:8001/v1"
GEMMA_MODEL = "google/gemma-4-31B-it"
QWEN_MODEL = "qwen3.7-plus"   # 2026-10-11: qwen3-vl-235b 퇴역(Alibaba 2026-10-10) → qwen3.7-plus, 생각 끔


# -- lightweight sibling imports (avoid the api.module NER stack) -----------

def _load_sibling(mod_name: str, filename: str):
    """
    Import a sibling file (client.py / prompts.py) by path. If the package is
    already importable cheaply we reuse the loaded module; otherwise we load
    the file directly. Either way we never trigger api/module/__init__.py.
    """
    existing = sys.modules.get(f"api.module.clip_extraction.vlm.{mod_name}")
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        f"_vlm_{mod_name}", THIS_DIR / filename)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load sibling {filename}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_client_mod = _load_sibling("client", "client.py")
_prompts_mod = _load_sibling("prompts", "prompts.py")

VLMClient = _client_mod.VLMClient
VLMResult = _client_mod.VLMResult
SYSTEM_PROMPT = _prompts_mod.SYSTEM_PROMPT
USER_PROMPT = _prompts_mod.USER_PROMPT


def _prompt_with_context(base: str, context) -> str:
    """계약서 문맥 {"title", "work_type"} 이 있으면 base 끝에 참고 줄을 붙인다 (M2 spec A4).

    context 가 None/빈 dict 이면 base 그대로 — 웹앱 등 기존 호출자의 프롬프트는 바뀌지 않는다.
    """
    if not context:
        return base
    if not isinstance(context, dict):
        raise TypeError("context 는 {'title': ..., 'work_type': ...} 형태의 dict 여야 합니다")
    return _prompts_mod.user_prompt_with_context(
        base, title=context.get("title"), work_type=context.get("work_type"))


_MANAGED_KEYS = ("OPENROUTER_API_KEY", "DASHSCOPE_API_KEY", "OPENROUTER_VLM_MODEL",
                 "OPENROUTER_BASE_URL", "DASHSCOPE_BASE_URL", "GEMMA_URL", "GEMINI_API_KEY")


def _dashscope_json_mode() -> bool:
    """DashScope 백엔드(omni·qwen3.7-plus)에 JSON 만 출력하게 강제하는가 — 환경변수 VLM_JSON_MODE (기본 on).
    무하유 2026-10-12 시험에서 omni-flash 응답 8건 중 2건을 JSON 으로 읽지 못해 2순위 모델로 넘어갔다(7.7초로 짧았다)."""
    return os.getenv("VLM_JSON_MODE", "on").strip().lower() not in ("off", "0", "false", "no")


def _parse_retries() -> int:
    """JSON 을 읽지 못한 응답이면 같은 백엔드로 몇 번 더 부르는가 — VLM_PARSE_RETRIES (기본 1). 그다음 폴백한다.
    파일럿 이미지 100건의 해석 실패 2건은 같은 모델로 다시 부르자 둘 다 읽혔다."""
    try:
        return max(0, int(os.getenv("VLM_PARSE_RETRIES", "1")))
    except ValueError:
        return 1


def _call_parse_retry(backend, call, label: str, fails: list):
    """call() 을 부르고, JSON 을 읽지 못하면 같은 백엔드로 _parse_retries() 번 더 부른다. 못 읽은 원문은 fails 에 남긴다
    (호출한 쪽이 결과 폴더에 저장 — 다음에 같은 일이 생기면 무엇이 왔는지 볼 수 있게). 예외는 (None, 오류문) 으로 돌려준다."""
    tries = 1 + _parse_retries()
    result, err = None, ""
    for k in range(tries):
        try:
            result = call()
        except Exception as e:  # noqa: BLE001 — defensive: client shouldn't raise
            return None, f"{type(e).__name__}: {e}"
        if not (result.ok and not result.parse_ok):
            return result, err
        raw = result.raw_text or ""
        fails.append({"backend": backend.model_label, "attempt": k + 1, "latency_s": result.latency_s,
                      "finish_reason": getattr(result, "finish_reason", None), "usage": result.usage,
                      "chars": len(raw), "raw": raw[:8000]})
        logger.warning("VLM 응답을 JSON 으로 읽지 못함 — %s · %s · %.1f초 · 종료 사유 %s · %d자 %r%s",
                       backend.model_label, label, result.latency_s or 0.0, getattr(result, "finish_reason", None),
                       len(raw), raw[:200], " → 같은 모델로 다시" if k + 1 < tries else "")
    return result, err


def _ensure_env_loaded() -> None:
    """
    Load .env so API keys (OPENROUTER_API_KEY / DASHSCOPE_API_KEY) are available,
    mirroring env_loader's search order (project root → api/ → api/web/). Done
    inline to avoid importing api.module (which loads the NER stack).

    Robust against the empty-placeholder trap: a shipped `OPENROUTER_API_KEY=`
    line (empty) must NOT poison os.environ such that a later real key in .env
    is ignored. So we read .env with dotenv_values (no mutation) and copy a
    managed key into os.environ only when the .env value is non-empty and the
    current process value is missing/empty. This also means a real key added to
    .env is picked up by a fresh VLMExtractor (or refresh(rebuild=True)) without
    a full process restart, while a real shell-provided key is never clobbered.
    """
    try:
        from dotenv import dotenv_values
    except Exception:  # noqa: BLE001 — dotenv optional; rely on shell env
        return
    for env_path in (ROOT / ".env", ROOT / "api" / ".env", ROOT / "api" / "web" / ".env"):
        if not env_path.exists():
            continue
        values = dotenv_values(env_path)
        for key in _MANAGED_KEYS:
            val = (values.get(key) or "").strip()
            if val and not (os.environ.get(key) or "").strip():
                os.environ[key] = val
        return


# -- backend factory --------------------------------------------------------

# OpenRouter (OpenAI-호환): Gemma 4 31B를 API로 제공 — 2026-07-08 결정으로 주(1순위) 백엔드.
# 키가 없으면 ping 실패 → 체인이 자동으로 자체 서버(Gemma)로 넘어가므로 키 없이도 동작.
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemma-4-31b-it"  # env OPENROUTER_VLM_MODEL 로 교체 가능(예: ...:free)


def make_openrouter_backend():
    """Gemma 4 31B via OpenRouter API. api_key from env OPENROUTER_API_KEY.

    OpenRouter의 /models 목록은 무인증 공개라 기본 ping이 키 없이도 성공한다 —
    그대로 두면 키 미설정 상태에서 체인이 OpenRouter를 선택해 매 호출 401→폴백을
    반복한다. 그래서 ping을 인증 검증형으로 교체: 키가 없으면 즉시 DOWN, 있으면
    /api/v1/key (인증 필요 엔드포인트)로 실제 키 유효성을 확인한다.
    """
    _ensure_env_loaded()
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    client = VLMClient(
        model_label="Gemma 4 31B (OpenRouter)",
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE),
        model=os.getenv("OPENROUTER_VLM_MODEL", OPENROUTER_MODEL),
        api_key=key or "not-set",
        image_first=True,  # same weights as local Gemma → same ordering
    )

    def _authed_ping():
        if not key:
            return False, "OPENROUTER_API_KEY not set (.env)"
        try:
            import requests
            r = requests.get("https://openrouter.ai/api/v1/key",
                             headers={"Authorization": f"Bearer {key}"}, timeout=10)
            if r.status_code == 200:
                usage = r.json().get("data", {})
                return True, f"key ok (usage=${usage.get('usage', '?')})"
            return False, f"key check HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"

    client.ping = _authed_ping
    return client


def make_qwen_backend():
    """Qwen3-VL on DashScope (cloud). api_key from env DASHSCOPE_API_KEY."""
    _ensure_env_loaded()
    return VLMClient(
        model_label="Qwen3.7-Plus (DashScope)",
        base_url=os.getenv("DASHSCOPE_BASE_URL", DASHSCOPE_BASE),
        model=QWEN_MODEL,
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        image_first=False,  # matches existing OCR pipeline ordering
        extra_body={"enable_thinking": False},
        json_mode=_dashscope_json_mode(),
    )


def make_gemma_backend(gemma_url: str | None = None):
    """Gemma 4 31B on the lab vLLM server. URL from env GEMMA_URL or default."""
    return VLMClient(
        model_label="Gemma 4 31B (vLLM)",
        base_url=gemma_url or os.getenv("GEMMA_URL", DEFAULT_GEMMA_URL),
        model=GEMMA_MODEL,
        api_key="not-needed",
        image_first=True,  # Gemma prefers image-first
        # 연구실 서버가 닿지 않으면 연결 단계에서 바로 넘긴다 — 예전에는 요청당 약 4분(257초 실측)을 기다렸다
        connect_timeout=float(os.getenv("GEMMA_CONNECT_TIMEOUT_SEC", "10")),
        max_retries=0,
    )


def make_google_gemma_backend():
    """Gemma 4 31B on Google AI Studio (Gemini API, OpenAI-compatible) — 무료(유료 등급 없음), 분당 30회(실측 2026-10-11).
    키: GEMINI_API_KEY. 추론 블록(<thought>)은 client._extract_json 이 지운다. 무료 등급 약관상 입력이 제품 개선·사람 검토에
    쓰일 수 있으므로 계약서·개인정보에는 쓰지 않는다 (저작물 이미지 전용). 기본 체인에는 아직 넣지 않는다."""
    _ensure_env_loaded()
    return VLMClient(
        model_label="Gemma 4 31B (Google AI Studio)",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model=os.getenv("GOOGLE_GEMMA_MODEL", "gemma-4-31b-it"),
        api_key=os.getenv("GEMINI_API_KEY", ""),
        image_first=True,
    )


def make_omni_backend():
    """qwen3.5-omni-plus on DashScope — **영상 트랙 1순위**.

    8개 모델 × 5영상 비교시험에서 정확도 92.3%(24/26)로 1위이면서 평균 5.0초로
    상위권 중 유일하게 빨랐다(2위 qwen3.7-plus 84.6%/15.0초).
    근거: docs/영상모델_비교보고서_20260803.md
    """
    _ensure_env_loaded()
    # 2026-10-11: 기본을 qwen3.8-omni-flash 로 — 파일럿 영상 100건 Macro 80.9% vs 3.5-omni-plus 78.6%
    # (+2.33점, 95% CI [+0.73, +3.99]), 단가 입력 $0.15 / 출력 $0.47 (3.5-omni-plus 는 $1.4 / $8.3). docs/v4/vision_backends_20261011.md
    omni_model = os.getenv("OMNI_MODEL", "qwen3.8-omni-flash")
    return VLMClient(
        # 라벨은 실제 모델을 따른다 — 예전에는 고정 문자열이라 OMNI_MODEL 을 바꿔도 결과에 3.5-Omni-Plus 로 기록됐다
        model_label="Qwen3.5-Omni-Plus (DashScope)" if omni_model == "qwen3.5-omni-plus" else f"{omni_model} (DashScope)",
        base_url=os.getenv("DASHSCOPE_BASE_URL", DASHSCOPE_BASE),
        model=omni_model,
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        image_first=False,
        json_mode=_dashscope_json_mode(),
    )


def make_omni_backup_backend():
    """영상 트랙 2순위 — qwen3.5-omni-plus (2026-10-10 까지의 1순위). OMNI_BACKUP_MODEL 로 바꿀 수 있다."""
    _ensure_env_loaded()
    model = os.getenv("OMNI_BACKUP_MODEL", "qwen3.5-omni-plus")
    return VLMClient(
        model_label="Qwen3.5-Omni-Plus (DashScope)" if model == "qwen3.5-omni-plus" else f"{model} (DashScope)",
        base_url=os.getenv("DASHSCOPE_BASE_URL", DASHSCOPE_BASE),
        model=model,
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        image_first=False,
        json_mode=_dashscope_json_mode(),
    )


def default_backends(prefer: str = "gemma", gemma_url: str | None = None) -> list:
    """
    Build the default ordered backend chain (2026-07-08 운영 결정):

    prefer="gemma" (기본)   → [Gemma@OpenRouter, Gemma@자체서버, Qwen@DashScope]
                              — 동일 모델(Gemma 4 31B)을 API 우선, 자체 서버 폴백으로.
    prefer="openrouter"     → 동일 ("gemma"의 별칭)
    prefer="gemma-local"    → [Gemma@자체서버, Gemma@OpenRouter, Qwen@DashScope]
                              — 자체 서버 우선(내부망/오프라인 운용 시)
    prefer="qwen"           → [Qwen@DashScope, Gemma@OpenRouter, Gemma@자체서버]

    OPENROUTER_API_KEY 미설정 시 OpenRouter ping이 실패하고 체인이 다음 백엔드로
    자동 진행되므로, 어느 구성에서도 코드 변경 없이 동작한다.
    """
    openrouter = make_openrouter_backend()
    gemma_local = make_gemma_backend(gemma_url)
    qwen = make_qwen_backend()
    if prefer == "omni":
        # 영상 트랙 기본 체인 — 비교시험 순위 그대로(92.3% → 80.8% → 76.9%)
        # 영상: qwen3.8-omni-flash → qwen3.5-omni-plus(이전 1순위, 백업) → qwen3.7-plus → Gemma
        return [make_omni_backend(), make_omni_backup_backend(), qwen, openrouter, gemma_local]
    google_gemma = make_google_gemma_backend()
    if prefer == "auto":
        # 이미지 기본 체인 (2026-10-11 결정): qwen3.8-omni-flash → qwen3.7-plus → Gemma 4(Google, 무료) → Gemma(OpenRouter) → Gemma(연구실)
        # 파일럿 이미지 100건: omni-flash 100/100·설명 통과 0.916·19초·₩1.34, qwen3.7-plus 98/100·0.863, Gemma(Google) 97/100·0.874·92초
        return [make_omni_backend(), qwen, google_gemma, openrouter, gemma_local]
    if prefer == "qwen":
        return [qwen, google_gemma, openrouter, gemma_local]
    if prefer == "gemma-local":
        return [gemma_local, google_gemma, openrouter, qwen]
    # "gemma" | "openrouter" → Gemma 우선: Google(무료) → OpenRouter → 연구실 → qwen3.7-plus
    return [google_gemma, openrouter, gemma_local, qwen]


@dataclass
class _ProbeResult:
    """Per-backend ping outcome, kept for error reporting."""
    label: str
    ok: bool
    detail: str


class VLMExtractor:
    """
    Ordered VLM backends with automatic reachability selection + per-call
    fallback. Probes lazily on first use; `refresh()` re-probes (e.g. after a
    downed Gemma server returns).
    """

    def __init__(
        self,
        backends: list | None = None,
        prefer: str = "auto",
        gemma_url: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self._prefer = prefer
        self._gemma_url = gemma_url
        self._custom_backends = backends is not None
        self.backends: list = backends or default_backends(prefer, gemma_url)
        if not self.backends:
            raise ValueError("VLMExtractor needs at least one backend")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._active = None
        self._active_index: int | None = None
        self._probe_log: list[_ProbeResult] = []

    # -- backend selection --------------------------------------------------

    def _probe_chain(self):
        """Ping backends in order; return the first reachable one + its index.

        첫 번째로 응답한 백엔드에서 **멈춘다**. 뒤쪽 백엔드는 실제 호출이 실패할 때 extract()/extract_video()
        가 그때 폴백하므로 미리 두드릴 이유가 없다. 예전에는 체인 전체를 두드려서, 자체 서버(GEMMA_URL)가
        SYN 단계에서 멎어 있으면 다리마다 클라이언트 타임아웃만큼 기다렸다(실측: 이미지 leg 272.85s 중 ~250s).
        `_probe_log` 에는 두드린 백엔드만 남는다.
        """
        self._probe_log = []
        selected = None
        selected_idx: int | None = None
        for idx, backend in enumerate(self.backends):
            t0 = time.perf_counter()
            ok, detail = backend.ping()
            sec = time.perf_counter() - t0
            self._probe_log.append(_ProbeResult(backend.model_label, ok, detail))
            if ok:
                selected, selected_idx = backend, idx
                logger.info("VLM 백엔드 확인 — %s 응답함 (%s, %.1f초) → 이 백엔드로 시작", backend.model_label,
                            str(detail)[:120], sec)
                break
            nxt = self.backends[idx + 1].model_label if idx + 1 < len(self.backends) else None
            logger.warning("VLM 백엔드 응답 없음 — %s: %s (%.1f초)%s", backend.model_label, str(detail)[:300], sec,
                           f" → 다음 {nxt}" if nxt else " — 남은 백엔드 없음")
        return selected, selected_idx

    def active_backend(self):
        """
        Return the first reachable backend (cached). Raises RuntimeError listing
        every backend tried if none respond.
        """
        if self._active is not None:
            return self._active
        selected, idx = self._probe_chain()
        if selected is None:
            tried = "; ".join(f"{p.label}: {p.detail}" for p in self._probe_log)
            raise RuntimeError(
                f"No reachable VLM backend. Tried {len(self.backends)} → {tried}"
            )
        # Note any backends ahead of the selected one that were skipped.
        for p in self._probe_log[:idx]:
            logger.debug("VLM 폴백 — 앞 순위 %s 응답 없음 → %s 사용", p.label, selected.model_label)
        self._active, self._active_index = selected, idx
        return selected

    def refresh(self, rebuild: bool = False):
        """
        Clear the cache and re-probe (e.g. after a downed server returns).

        rebuild=True also reconstructs the backend list from the factory, which
        re-reads env (API keys). Use this after adding OPENROUTER_API_KEY to .env
        on a long-lived extractor so the new key is picked up without a process
        restart. Ignored when the extractor was built with explicit backends=[...].
        """
        if rebuild and not self._custom_backends:
            self.backends = default_backends(self._prefer, self._gemma_url)
        self._active = None
        self._active_index = None
        return self.active_backend()

    @property
    def probe_log(self) -> list[dict[str, Any]]:
        """Last probe results as plain dicts (for reports / debugging)."""
        return [{"label": p.label, "ok": p.ok, "detail": p.detail}
                for p in self._probe_log]

    # -- extraction ---------------------------------------------------------

    def extract(self, image_path: str | Path, context: dict | None = None):
        """
        Extract attributes from one image using the active backend. If the call
        errors (exception or VLMResult.ok is False), advance to the next backend
        in the chain and retry. The returned VLMResult carries a `backend_used`
        field. If every backend fails, returns a VLMResult with ok=False.

        context: 계약서 참고 문맥 {"title", "work_type"} — 프롬프트 끝에 참고 줄로 붙는다 (A4).
                 None 이면 프롬프트는 기존과 동일하다.
        """
        image_path = Path(image_path)
        user_prompt = _prompt_with_context(USER_PROMPT, context)
        # Ensure a backend is selected (sets the start index) before we begin.
        if self._active is None:
            self.active_backend()
        start = self._active_index or 0

        last_error = ""
        fails: list = []                    # JSON 으로 읽지 못한 응답 원문 (결과의 parse_failures)
        for idx in range(start, len(self.backends)):
            backend = self.backends[idx]
            result, exc = _call_parse_retry(backend, lambda: backend.extract(
                image_path, SYSTEM_PROMPT, user_prompt,
                max_tokens=self.max_tokens, temperature=self.temperature,
            ), image_path.name, fails)
            if exc:
                last_error = exc

            if result is not None and result.ok and result.parse_ok:   # 다시 불러도 못 읽으면 실패로 보고 다음 백엔드로
                _attach_backend(result, backend.model_label)
                _attach_context(result, context)
                setattr(result, "parse_failures", fails)
                if idx != start:
                    logger.info("VLM 폴백 성공 — %s (%s, %.1f초)",
                                backend.model_label, image_path.name, getattr(result, "latency_s", 0.0) or 0.0)
                    # Promote the working backend so later calls start here.
                    self._active, self._active_index = backend, idx
                return result

            # This backend failed — log and try the next.
            err = (result.error or ("JSON 해석 실패" if result.ok and not result.parse_ok else None)) if result is not None else last_error
            last_error = err or "unknown error"
            lat = f", {result.latency_s}초" if result is not None and result.latency_s else ""
            if idx + 1 < len(self.backends):
                nxt = self.backends[idx + 1]
                logger.warning("VLM 추출 실패 — %s: %s%s → 폴백 %s",
                               backend.model_label, str(last_error)[:300], lat, nxt.model_label)
            else:
                logger.error("VLM 추출 실패 — %s: %s%s · 남은 백엔드 없음",
                             backend.model_label, str(last_error)[:300], lat)

        # Every backend in the chain failed.
        fail = VLMResult(
            model_label="none",
            image=image_path.name,
            ok=False,
            latency_s=0.0,
            error=f"all {len(self.backends)} backend(s) failed; last: {last_error}",
        )
        _attach_backend(fail, "none")
        setattr(fail, "parse_failures", fails)
        return fail

    def extract_video(self, frame_paths: list, frame_times: list | None = None,
                      label: str = "video", context: dict | None = None):
        """키프레임 여러 장을 한 번의 호출로 보내 영상 속성을 추출한다.

        extract() 와 동일한 폴백 체인 규칙을 따른다 — 실패하면 다음 백엔드로
        넘어가고, 성공한 백엔드를 승격시켜 이후 호출이 거기서 시작하게 한다.
        context 는 extract() 와 같다 (계약서 참고 문맥, 없으면 프롬프트 불변).
        """
        from .prompts import VIDEO_SYSTEM_PROMPT, video_user_prompt

        if not frame_paths:
            fail = VLMResult(model_label="none", image=label, ok=False, latency_s=0.0,
                             error="키프레임이 없습니다")
            _attach_backend(fail, "none")
            return fail

        user_prompt = _prompt_with_context(video_user_prompt(frame_times or []), context)
        if self._active is None:
            self.active_backend()
        start = self._active_index or 0

        last_error = ""
        fails: list = []
        for idx in range(start, len(self.backends)):
            backend = self.backends[idx]
            result, exc = _call_parse_retry(backend, lambda: backend.extract_frames(
                frame_paths, VIDEO_SYSTEM_PROMPT, user_prompt,
                max_tokens=self.max_tokens, temperature=self.temperature,
                label=label,
            ), label, fails)
            if exc:
                last_error = exc

            if result is not None and result.ok and result.parse_ok:   # 다시 불러도 못 읽으면 실패로 보고 다음 백엔드로
                _attach_backend(result, backend.model_label)
                _attach_context(result, context)
                setattr(result, "parse_failures", fails)
                if idx != start:
                    logger.info("VLM 영상 폴백 성공 — %s (%s, %.1f초)",
                                backend.model_label, label, getattr(result, "latency_s", 0.0) or 0.0)
                    self._active, self._active_index = backend, idx
                return result

            err = (result.error or ("JSON 해석 실패" if result.ok and not result.parse_ok else None)) if result is not None else last_error
            last_error = err or "unknown error"
            lat = f", {result.latency_s}초" if result is not None and result.latency_s else ""
            if idx + 1 < len(self.backends):
                logger.warning("VLM 영상 추출 실패 — %s: %s%s → 폴백 %s", backend.model_label,
                               str(last_error)[:300], lat, self.backends[idx + 1].model_label)
            else:
                logger.error("VLM 영상 추출 실패 — %s: %s%s · 남은 백엔드 없음",
                             backend.model_label, str(last_error)[:300], lat)

        fail = VLMResult(model_label="none", image=label, ok=False, latency_s=0.0,
                         error=f"all {len(self.backends)} backend(s) failed; last: {last_error}")
        _attach_backend(fail, "none")
        setattr(fail, "parse_failures", fails)
        return fail

    def extract_batch(self, image_paths: list) -> list:
        """Extract attributes from many images. Selects a backend once up front."""
        self.active_backend()  # probe once, fail fast if nothing is reachable
        return [self.extract(p) for p in image_paths]


def _attach_backend(result, backend_label: str) -> None:
    """Stamp the backend that actually produced a result onto the VLMResult.

    VLMResult is a dataclass without this field; setattr keeps client.py
    untouched while exposing `backend_used` to callers.
    """
    setattr(result, "backend_used", backend_label)


def _attach_context(result, context) -> None:
    """어떤 계약서 문맥이 프롬프트에 들어갔는지 결과에 남긴다 (`prompt_context`, 없으면 None)."""
    setattr(result, "prompt_context", dict(context) if context else None)


# -- CLI smoke test ---------------------------------------------------------

def _discover_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted(p for p in image_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Production VLM extractor (auto-fallback)")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES),
                        help="Directory of images to extract")
    parser.add_argument("--prefer", default="gemma",
                        choices=["gemma", "openrouter", "gemma-local", "qwen"],
                        help="Chain order (default gemma = OpenRouter→자체서버→Qwen)")
    parser.add_argument("--gemma-url", default=None,
                        help="Gemma vLLM base URL (env GEMMA_URL also works)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N images (0 = all)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ex = VLMExtractor(prefer=args.prefer, gemma_url=args.gemma_url)

    print("Probing backend chain...")
    try:
        active = ex.active_backend()
    except RuntimeError as e:
        print(f"FATAL: {e}")
        return 3
    for p in ex.probe_log:
        print(f"  {'OK  ' if p['ok'] else 'DOWN'} {p['label']}: {p['detail']}")
    print(f"Active backend: {active.model_label}\n")

    images = _discover_images(Path(args.images))
    if not images:
        print(f"No images in {args.images} — backend selection still verified above.")
        return 0
    if args.limit:
        images = images[:args.limit]

    print(f"Extracting {len(images)} image(s)...\n")
    for img in images:
        res = ex.extract(img)
        backend_used = getattr(res, "backend_used", "?")
        if res.ok:
            wt = (res.parsed or {}).get("work_type", "?") if res.parse_ok else "(parse failed)"
            print(f"  {img.name}")
            print(f"    backend_used: {backend_used}")
            print(f"    work_type:    {wt}  ({res.latency_s}s)")
        else:
            print(f"  {img.name}")
            print(f"    backend_used: {backend_used}")
            print(f"    ERROR: {res.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
