"""
Unified OpenAI-compatible VLM client.

Both target models speak the OpenAI Chat Completions API — they differ only
in base_url / model / api_key:

  Gemma 4 31B   base_url=http://<host>:8001/v1            api_key=(ignored)
  Qwen3-VL-235B base_url=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
                api_key=$DASHSCOPE_API_KEY

So one client class parametrized by those three values covers both.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:                                   # 큰 그림 풀기는 동시 개수를 제한한다 (module/local_limits.py)
    from ...local_limits import heavy_slot
except ImportError:                    # 스크립트로 직접 실행할 때 — 제한 없음
    import contextlib as _contextlib
    heavy_slot = _contextlib.nullcontext

# ping 전용 timeout(초). 추출 호출의 timeout(기본 300s)·재시도를 ping 에도 쓰면 SYN 단계에서 멎은
# 서버 하나에 5분을 기다린다. 환경변수 VLM_PING_TIMEOUT_SEC 로 바꿀 수 있다.
PING_TIMEOUT_SEC = 10.0


def _ping_timeout() -> float:
    try:
        v = float(os.getenv("VLM_PING_TIMEOUT_SEC", "") or PING_TIMEOUT_SEC)
    except ValueError:
        v = PING_TIMEOUT_SEC
    return v if v > 0 else PING_TIMEOUT_SEC


@dataclass
class VLMResult:
    model_label: str
    image: str
    ok: bool
    latency_s: float
    raw_text: str = ""
    parsed: dict[str, Any] | None = None
    parse_ok: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    finish_reason: str | None = None


# DashScope rejects images over ~10 MB ("Multimodal file size is too large").
# VLMs also internally downscale, so a long-edge cap costs no quality but keeps
# us well under provider limits and makes Gemma/Qwen see the same pixels.
_MAX_LONG_EDGE = 1536  # px
# JPEG 가 이보다 크면 축소해서 푼다(draft) — 22,991×30,000(6.9억 px) 원본을 다 풀면 2 GB 가 넘는다.
# 보통 그림은 지금과 똑같이 원본에서 줄인다 (모델 입력을 바꾸지 않는다).
_HUGE_PIXELS = 100_000_000


def _encode_image(image_path: Path) -> str:
    """
    Read an image, downscale if its long edge exceeds _MAX_LONG_EDGE, and
    return an OpenAI data: URL (base64 JPEG). Downscaling keeps large KOGL
    source files under the DashScope size limit.

    Pillow 의 압축 폭탄 한도(약 1.79억 px)를 여기서 끈다. 예전에는 미리보기 저장(pipeline._keep_work_preview)이
    전역으로 끄기 전까지 큰 그림이 DecompressionBombError 로 모든 백엔드에서 실패했다 — G4000 에 6.9억 px·2.6억 px
    그림이 있다(13266578, 13266658). 우리 시험 자료라 폭탄 걱정은 없고, 아래 draft 로 메모리도 묶는다.
    """
    from PIL import Image
    import io

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as im:
        if im.format == "JPEG" and im.width * im.height > _HUGE_PIXELS:
            im.draft("RGB", (_MAX_LONG_EDGE, _MAX_LONG_EDGE))   # 1/2·1/4·1/8 로 풀어 긴 변 1536 이상만 남긴다
        im = im.convert("RGB")
        long_edge = max(im.size)
        if long_edge > _MAX_LONG_EDGE:
            scale = _MAX_LONG_EDGE / long_edge
            new_size = (round(im.width * scale), round(im.height * scale))
            im = im.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


_THOUGHT_BLOCK = re.compile(r"<thought>.*?</thought>", re.S)


def _extract_json(text: str) -> dict[str, Any] | None:
    # Google AI Studio 의 Gemma 4 는 추론을 <thought>…</thought> 로 본문에 먼저 쓴다(끌 수 없음) — 그 안의 중괄호를 JSON 으로 잡지 않게 먼저 지운다.
    text = _THOUGHT_BLOCK.sub("", text or "")
    """
    Best-effort parse of a JSON object from model output. Handles bare JSON,
    ```json fenced blocks, and leading/trailing prose.
    """
    if not text:
        return None
    # Strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    # Fallback: grab the outermost {...}
    if not fence:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start:end + 1]
    try:
        obj = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    # 빈 객체({})·배열·문자열은 읽지 못한 것으로 본다 — 2026-10-30 영상 1건이 읽기 성공·내용 없음으로 와서
    # 같은 모델 재시도도 폴백도 없이 실패했다. None 이면 재시도 → 폴백 → vlm_unparsed.jsonl 에 원문이 남는다.
    return obj if isinstance(obj, dict) and obj else None


class VLMClient:
    """Thin wrapper over the openai SDK targeting one VLM backend."""

    def __init__(
        self,
        model_label: str,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 300.0,
        image_first: bool = True,
        extra_body: dict | None = None,
        connect_timeout: float | None = None,
        max_retries: int = 1,
        json_mode: bool = False,
        min_temperature: float | None = None,
    ):
        from openai import OpenAI
        # 이 백엔드만 온도 하한을 둔다 — Google AI Studio 의 Gemma 4 는 온도 0 에서 크레딧 자막의 점선(". . . .")을 반복하다
        # 출력 한도에 걸렸다 (G0050 Te-2 재실행 13223267, 2026-11-10: 2/2 실패). 0.5 에서는 같은 프레임 3/3 성공.
        self.min_temperature = min_temperature
        self.extra_body = extra_body   # 예: {"enable_thinking": False} — qwen3.7 계열은 기본이 생각 켬
        # JSON 만 출력하게 강제(response_format json_object) — omni-flash 가 가끔 읽을 수 없는 응답을 냈다(무하유 8건 중 2건)
        self.json_mode = json_mode
        self.model_label = model_label
        self.model = model
        self.image_first = image_first
        if connect_timeout is not None:
            import httpx   # 연결 단계만 짧게 — 닿지 않는 서버(SYN 무응답)에서 300초를 기다리지 않게
            timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed",
                             timeout=timeout, max_retries=max_retries)

    def ping(self) -> tuple[bool, str]:
        """Cheap reachability check — lists models. Returns (ok, detail).

        짧은 timeout(_ping_timeout, 기본 10s)·재시도 없음으로 돈다. 느려서 '불가' 로 보여도 실제 호출은
        extract()/extract_video() 가 폴백하므로 잃는 것은 체인의 시작 순서뿐이다.
        """
        try:
            client = self.client.with_options(timeout=_ping_timeout(), max_retries=0)
        except Exception:  # noqa: BLE001 — with_options 가 없는 구버전 SDK 면 기본 설정으로
            client = self.client
        try:
            models = client.models.list()
            ids = [m.id for m in models.data]
            return True, f"{len(ids)} model(s): {', '.join(ids[:3])}"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"

    def _temperature(self, temperature: float) -> float:
        return max(temperature, self.min_temperature) if self.min_temperature is not None else temperature

    def extract(
        self,
        image_path: Path,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VLMResult:
        """Run one image through the model. Times the call, parses JSON."""
        with heavy_slot():                 # 수백 MB TIF 를 푸는 일 — 동시에 몇 개만 (메모리)
            data_url = _encode_image(image_path)
        image_part = {"type": "image_url", "image_url": {"url": data_url}}
        text_part = {"type": "text", "text": user_prompt}
        content = [image_part, text_part] if self.image_first else [text_part, image_part]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                **({'extra_body': self.extra_body} if self.extra_body else {}),
                **({'response_format': {"type": "json_object"}} if self.json_mode else {}),
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self._temperature(temperature),
            )
            latency = time.perf_counter() - t0
            text = resp.choices[0].message.content or ""
            finish = getattr(resp.choices[0], "finish_reason", None)
            parsed = _extract_json(text)
            usage = {}
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                }
            return VLMResult(
                model_label=self.model_label,
                image=image_path.name,
                ok=True,
                latency_s=round(latency, 2),
                raw_text=text,
                parsed=parsed,
                parse_ok=parsed is not None,
                usage=usage,
                finish_reason=finish,
            )
        except Exception as e:  # noqa: BLE001
            return VLMResult(
                model_label=self.model_label,
                image=image_path.name,
                ok=False,
                latency_s=round(time.perf_counter() - t0, 2),
                error=f"{type(e).__name__}: {e}",
            )

    def extract_frames(
        self,
        frame_paths: list,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        label: str = "video",
    ) -> VLMResult:
        """여러 프레임을 **한 번의 호출**로 보내 영상 전체 속성을 추출한다.

        프레임을 한 장씩 따로 호출하면 모델이 '개별 이미지 N장'으로 볼 뿐 영상의
        전개를 보지 못한다. 영상 모델 비교시험도 이 방식(다중 이미지 1회 호출)으로
        측정했고 qwen3.5-omni-plus 가 92.3% 로 1위였다.

        네이티브 비디오 입력(type:"video")은 base64 크기 제한(9.4MB→HTTP 413)에
        걸려 쓰지 않는다.
        """
        parts = [{"type": "image_url", "image_url": {"url": _encode_image(Path(p))}}
                 for p in frame_paths]
        text_part = {"type": "text", "text": user_prompt}
        content = parts + [text_part] if self.image_first else [text_part] + parts
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        t0 = time.perf_counter()
        try:
            resp = self.client.chat.completions.create(
                **({'extra_body': self.extra_body} if self.extra_body else {}),
                **({'response_format': {"type": "json_object"}} if self.json_mode else {}),
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self._temperature(temperature),
            )
            latency = time.perf_counter() - t0
            text = resp.choices[0].message.content or ""
            finish = getattr(resp.choices[0], "finish_reason", None)
            parsed = _extract_json(text)
            usage = {}
            if resp.usage:
                usage = {"prompt_tokens": resp.usage.prompt_tokens,
                         "completion_tokens": resp.usage.completion_tokens}
            return VLMResult(
                model_label=self.model_label, image=label, ok=True,
                latency_s=round(latency, 2), raw_text=text, parsed=parsed,
                parse_ok=parsed is not None, usage=usage, finish_reason=finish,
            )
        except Exception as e:  # noqa: BLE001
            return VLMResult(
                model_label=self.model_label, image=label, ok=False,
                latency_s=round(time.perf_counter() - t0, 2),
                error=f"{type(e).__name__}: {e}",
            )
