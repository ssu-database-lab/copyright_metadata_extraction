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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
DEFAULT_IMAGES = THIS_DIR.parent / "test_data" / "sample_works"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Backend definitions — kept in sync with compare.py -----------------------
DASHSCOPE_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_GEMMA_URL = "http://127.0.0.1:8001/v1"
GEMMA_MODEL = "google/gemma-4-31B-it"
QWEN_MODEL = "qwen3-vl-235b-a22b-instruct"


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


_MANAGED_KEYS = ("OPENROUTER_API_KEY", "DASHSCOPE_API_KEY", "OPENROUTER_VLM_MODEL",
                 "OPENROUTER_BASE_URL", "GEMMA_URL")


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
        model_label="Qwen3-VL-235B (DashScope)",
        base_url=DASHSCOPE_BASE,
        model=QWEN_MODEL,
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        image_first=False,  # matches existing OCR pipeline ordering
    )


def make_gemma_backend(gemma_url: str | None = None):
    """Gemma 4 31B on the lab vLLM server. URL from env GEMMA_URL or default."""
    return VLMClient(
        model_label="Gemma 4 31B (vLLM)",
        base_url=gemma_url or os.getenv("GEMMA_URL", DEFAULT_GEMMA_URL),
        model=GEMMA_MODEL,
        api_key="not-needed",
        image_first=True,  # Gemma prefers image-first
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
    if prefer == "qwen":
        return [qwen, openrouter, gemma_local]
    if prefer == "gemma-local":
        return [gemma_local, openrouter, qwen]
    # "gemma" | "openrouter" | anything else → API-first Gemma chain
    return [openrouter, gemma_local, qwen]


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
        prefer: str = "gemma",
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
        """Ping backends in order; return the first reachable one + its index."""
        self._probe_log = []
        selected = None
        selected_idx: int | None = None
        for idx, backend in enumerate(self.backends):
            ok, detail = backend.ping()
            self._probe_log.append(_ProbeResult(backend.model_label, ok, detail))
            if ok and selected is None:
                selected, selected_idx = backend, idx
                logger.info("VLM backend selected: %s (%s)", backend.model_label, detail)
            elif not ok:
                logger.warning("VLM backend unreachable: %s — %s",
                               backend.model_label, detail)
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
            logger.info("falling back: primary %s unreachable → using %s",
                        p.label, selected.model_label)
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

    def extract(self, image_path: str | Path):
        """
        Extract attributes from one image using the active backend. If the call
        errors (exception or VLMResult.ok is False), advance to the next backend
        in the chain and retry. The returned VLMResult carries a `backend_used`
        field. If every backend fails, returns a VLMResult with ok=False.
        """
        image_path = Path(image_path)
        # Ensure a backend is selected (sets the start index) before we begin.
        if self._active is None:
            self.active_backend()
        start = self._active_index or 0

        last_error = ""
        for idx in range(start, len(self.backends)):
            backend = self.backends[idx]
            try:
                result = backend.extract(
                    image_path, SYSTEM_PROMPT, USER_PROMPT,
                    max_tokens=self.max_tokens, temperature=self.temperature,
                )
            except Exception as e:  # noqa: BLE001 — defensive: client shouldn't raise
                last_error = f"{type(e).__name__}: {e}"
                result = None

            if result is not None and result.ok:
                _attach_backend(result, backend.model_label)
                if idx != start:
                    logger.info("fallback succeeded on %s for %s",
                                backend.model_label, image_path.name)
                    # Promote the working backend so later calls start here.
                    self._active, self._active_index = backend, idx
                return result

            # This backend failed — log and try the next.
            err = result.error if result is not None else last_error
            last_error = err or "unknown error"
            if idx + 1 < len(self.backends):
                nxt = self.backends[idx + 1]
                logger.warning("primary %s failed (%s) → falling back to %s",
                               backend.model_label, last_error, nxt.model_label)
            else:
                logger.error("backend %s failed (%s); no more fallbacks",
                             backend.model_label, last_error)

        # Every backend in the chain failed.
        fail = VLMResult(
            model_label="none",
            image=image_path.name,
            ok=False,
            latency_s=0.0,
            error=f"all {len(self.backends)} backend(s) failed; last: {last_error}",
        )
        _attach_backend(fail, "none")
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
