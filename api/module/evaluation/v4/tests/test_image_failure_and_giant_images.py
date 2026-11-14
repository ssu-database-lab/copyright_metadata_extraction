"""Image works: file attributes on a failed VLM call, and very large images (G1000 Te-2, 2026-11-14).

1. 13266336 (Rembrandt self-portrait): every VLM backend failed and the response had no resolution, although
   resolution comes from the file, not the model. The video branch already measured the file on failure.
2. G4000 holds 690 MP and 258 MP JPEGs (13266578, 13266658). _encode_image raised DecompressionBombError unless
   an earlier image's preview had already switched Pillow's limit off for the process.
"""
from __future__ import annotations

import base64
import io
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
API = ROOT / "api"


@pytest.fixture
def light_llm_extraction():
    """test_video_llm_extract 와 같다 — module.llm_extraction 의 __init__(torch) 없이 하위 모듈만 적재."""
    import module  # noqa: F401
    added_stub = False
    if "module.llm_extraction" not in sys.modules:
        stub = types.ModuleType("module.llm_extraction")
        stub.__path__ = [str(API / "module" / "llm_extraction")]  # type: ignore[attr-defined]
        sys.modules["module.llm_extraction"] = stub
        added_stub = True
    before = set(sys.modules)
    yield
    if added_stub:
        for name in [n for n in sys.modules if n == "module.llm_extraction" or n.startswith("module.llm_extraction.")]:
            if name not in before or name == "module.llm_extraction":
                sys.modules.pop(name, None)


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 80, 40)).save(buf, "PNG")
    return buf.getvalue()


def test_failed_image_still_gets_its_resolution_from_the_file(tmp_path, light_llm_extraction, monkeypatch):
    from api.web.pipeline import PipelineOrchestrator
    orch = PipelineOrchestrator(llm_processor=None, ner_predict_fn=None, available_ner_models={},
                                upload_dir=tmp_path / "uploads", results_dir=tmp_path / "results")
    monkeypatch.setattr(orch, "run_vlm", lambda ctx, prefer="auto", context=None, render=None: {
        "success": False, "metadata": {}, "confidence": 0.0, "model_used": "none",
        "error": "all 5 backend(s) failed", "extraction_time": 0.1})
    resp = orch.run(_png(640, 480), "portrait.png", consolidate_work=False)
    assert resp["success"] is False and resp["modality"] == "image"
    md = resp.get("consolidated_metadata") or resp.get("metadata") or {}
    assert md.get("resolution") == "640x480", md
    assert md.get("file_size") == len(_png(640, 480))


def test_encode_image_ignores_pillows_decompression_bomb_limit(tmp_path, monkeypatch):
    from module.clip_extraction.vlm import client as C
    p = tmp_path / "big.png"
    p.write_bytes(_png(400, 300))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)       # 120,000 px > 2 × 1,000 → would raise before the fix
    uri = C._encode_image(p)
    assert uri.startswith("data:image/jpeg;base64,")


def test_huge_jpeg_is_decoded_reduced_and_still_downscaled_to_1536(tmp_path, monkeypatch):
    from module.clip_extraction.vlm import client as C
    p = tmp_path / "huge.jpg"
    Image.new("RGB", (6000, 4000), (30, 90, 160)).save(p, "JPEG", quality=80)
    monkeypatch.setattr(C, "_HUGE_PIXELS", 1_000_000)           # treat this 24 MP JPEG as "huge" → draft path
    out = Image.open(io.BytesIO(base64.b64decode(C._encode_image(p).split(",", 1)[1])))
    assert max(out.size) == 1536 and out.size == (1536, 1024)
