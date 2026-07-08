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
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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


# DashScope rejects images over ~10 MB ("Multimodal file size is too large").
# VLMs also internally downscale, so a long-edge cap costs no quality but keeps
# us well under provider limits and makes Gemma/Qwen see the same pixels.
_MAX_LONG_EDGE = 1536  # px


def _encode_image(image_path: Path) -> str:
    """
    Read an image, downscale if its long edge exceeds _MAX_LONG_EDGE, and
    return an OpenAI data: URL (base64 JPEG). Downscaling keeps large KOGL
    source files under the DashScope size limit.
    """
    from PIL import Image
    import io

    with Image.open(image_path) as im:
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


def _extract_json(text: str) -> dict[str, Any] | None:
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
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None


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
    ):
        from openai import OpenAI
        self.model_label = model_label
        self.model = model
        self.image_first = image_first
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed",
                             timeout=timeout, max_retries=1)

    def ping(self) -> tuple[bool, str]:
        """Cheap reachability check — lists models. Returns (ok, detail)."""
        try:
            models = self.client.models.list()
            ids = [m.id for m in models.data]
            return True, f"{len(ids)} model(s): {', '.join(ids[:3])}"
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {e}"

    def extract(
        self,
        image_path: Path,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> VLMResult:
        """Run one image through the model. Times the call, parses JSON."""
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
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency = time.perf_counter() - t0
            text = resp.choices[0].message.content or ""
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
            )
        except Exception as e:  # noqa: BLE001
            return VLMResult(
                model_label=self.model_label,
                image=image_path.name,
                ok=False,
                latency_s=round(time.perf_counter() - t0, 2),
                error=f"{type(e).__name__}: {e}",
            )
