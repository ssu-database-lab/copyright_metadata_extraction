"""영상 저작물 — /api/llm-extract 가 쓰는 pipeline.run_video_response (2026-10-30). 모델 호출 없음.

1. run() 이 영상을 run_video_response 로 보낸다 (스트리밍 경로도 같은 메서드를 부른다)
2. contract_metadata 를 주면 이미지와 같이 계약서 권리정보를 상속한다 — 전에는 영상 경로만 빠져 있었다
3. 영상 분석이 실패하면 상속하지 않는다 (실패 응답에 계약서 값만 들어가 성공처럼 보이지 않게)
4. 기본(consolidate_work=False 로 부를 때) 중재자를 부르지 않는다
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

ROOT = Path(__file__).resolve().parents[5]
API = ROOT / "api"

CONTRACT = {"copyright_holder": "한국저작권위원회", "author": "홍길동", "work_type": "영상저작물",
            "commercial_use": "허용", "license_start_date": "2026-01-01", "work_title": "다른 제목"}


@pytest.fixture
def light_llm_extraction():
    """test_m2_pipeline 과 같다 — module.llm_extraction 의 __init__(torch) 없이 하위 모듈만 적재."""
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


def _orch(tmp_path: Path, monkeypatch, ok: bool = True, calls: List[str] = None):
    from api.web.pipeline import PipelineOrchestrator
    orch = PipelineOrchestrator(llm_processor=None, ner_predict_fn=None, available_ner_models={},
                                upload_dir=tmp_path / "uploads", results_dir=tmp_path / "results")

    def fake_video(ctx, prefer="omni", context=None, frame_schedule="v3"):
        if calls is not None:
            calls.append(f"video:{prefer}:{frame_schedule}")
        if not ok:
            return {"success": False, "metadata": {}, "model_used": "qwen3.8-omni-flash (DashScope)",
                    "error": "프레임 추출 실패", "_keyframes": []}
        md = {"work_type": "영상저작물", "description": "바닷가 풍경", "keyword": ["바다"], "digital_format": "MP4"}
        return {"success": True, "metadata": md, "confidence": 0.7, "model_used": "qwen3.8-omni-flash (DashScope)",
                "extraction_time": 0.1, "_keyframes": [0.5, 3.0], "_file_meta": {}}

    def no_consolidation(*a, **k):
        raise AssertionError("consolidate_work=False 인데 중재자가 불렸다")

    monkeypatch.setattr(orch, "run_video", fake_video)
    monkeypatch.setattr(orch, "run_consolidation", no_consolidation)
    monkeypatch.setattr(orch, "_backfill_file_attributes", lambda ctx, response: None)   # ffprobe 없이
    return orch


def test_run_video_inherits_contract_metadata(tmp_path, light_llm_extraction, monkeypatch):
    calls: List[str] = []
    orch = _orch(tmp_path, monkeypatch, calls=calls)
    resp = orch.run(b"\x00\x00\x00\x18ftypmp42", "13309792_1_바닷가.mp4", consolidate_work=False,
                    contract_metadata=CONTRACT)
    assert calls == ["video:omni:v3"]
    assert resp["success"] is True and resp["modality"] == "video"
    assert resp["vlm_backend"] == "qwen3.8-omni-flash (DashScope)" and resp["keyframes"] == [0.5, 3.0]
    md = resp["consolidated_metadata"]
    assert md["copyright_holder"] == "한국저작권위원회" and md["author"] == "홍길동"
    assert md["commercial_use"] == "허용" and md["license_start_date"] == "2026-01-01"
    assert md["description"] == "바닷가 풍경"                      # 영상 분석 값은 그대로
    assert md["work_title"] == "다른 제목"                         # 계약서 저작물이 하나면 제목도 상속 (이미지와 같은 규칙)
    inh = resp["contract_inheritance"]
    assert inh["applied"] is True and inh["inherited"] >= 5 and inh["title_match"] == "single"
    fields = {d["field"] for d in resp["consolidation_decisions"]}
    assert {"copyright_holder", "author", "commercial_use"} <= fields
    assert (tmp_path / "results" / resp["request_id"] / "llm_metadata.json").is_file()


def test_run_video_without_contract_is_unchanged(tmp_path, light_llm_extraction, monkeypatch):
    orch = _orch(tmp_path, monkeypatch)
    resp = orch.run(b"video", "a.mov", consolidate_work=False)
    assert resp["success"] is True and resp["modality"] == "video"
    assert "contract_inheritance" not in resp
    assert resp["consolidated_metadata"] == resp["metadata"] and resp["work_consolidation"]["applied"] is False


def test_failed_video_does_not_inherit(tmp_path, light_llm_extraction, monkeypatch):
    orch = _orch(tmp_path, monkeypatch, ok=False)
    resp = orch.run(b"video", "broken.mp4", consolidate_work=False, contract_metadata=CONTRACT)
    assert resp["success"] is False and resp["modality"] == "video"
    assert "contract_inheritance" not in resp
    assert not (resp.get("consolidated_metadata") or {}).get("copyright_holder")


def test_run_video_response_is_callable_on_a_prepared_ctx(tmp_path, light_llm_extraction, monkeypatch):
    """스트리밍 경로는 setup()·라우터 뒤에 이 메서드를 직접 부른다 — 같은 결과여야 한다."""
    orch = _orch(tmp_path, monkeypatch)
    ctx = orch.setup(b"video", "clip.mp4")
    resp = orch.run_video_response(ctx, consolidate_work=False, contract_metadata=CONTRACT)
    assert resp["request_id"] == ctx["request_id"] and resp["modality"] == "video"
    assert resp["consolidated_metadata"]["copyright_holder"] == "한국저작권위원회"


def test_router_and_api_accept_the_same_video_extensions():
    """app.py 의 /api/llm-extract 허용 목록은 라우터의 영상 확장자를 그대로 가져온다 — 소스로 확인 (app 적재는 torch 때문에 무겁다)."""
    from module.clip_extraction.router import _VIDEO_EXTS, detect_modality
    src = (API / "web" / "app.py").read_text(encoding="utf-8")
    assert "from module.clip_extraction.router import _VIDEO_EXTS as VIDEO_EXTENSIONS" in src
    assert "allowed_file(filename, LLM_EXTRACT_EXTENSIONS)" in src
    assert "pipeline_orchestrator.run_video_response" in src          # 스트리밍 경로도 영상을 처리한다
    for ext in ("mp4", "mov", "avi", "mkv", "webm", "wmv", "mpg", "mpeg", "m4v"):
        assert ext in _VIDEO_EXTS and detect_modality(f"x.{ext}") == "video"
