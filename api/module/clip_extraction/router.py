"""
Modality router for the Year 2 multimodal pipeline.

파일 확장자를 기준으로 입력 파일의 modality(이미지/오디오/비디오/문서/텍스트)를
판별하고, 어떤 추출기(extractor) 체인을 태울지 dispatch 계획을 돌려준다.

설계 의도:
- 이미지/오디오/비디오 같은 멀티미디어 저작물 → 생성형 VLM(설명·속성) + 임베딩(유사도).
- 문서/텍스트(계약서·동의서·OCR 결과) → 기존 OCR → LLM → NER 파이프라인 그대로.

이 모듈은 STANDALONE 빌딩블록이다. app.py / pipeline.py는 건드리지 않으며,
나중에 PipelineOrchestrator에서 route()의 결과를 보고 분기하도록 배선한다.
무거운 의존성(torch, transformers, VLM 클라이언트)은 import하지 않는다 —
확장자 매칭만 하므로 순수 표준 라이브러리로 충분하다.
"""

from __future__ import annotations

import os
from typing import Dict, List

# ---------------------------------------------------------------------------
# 확장자 → modality 매핑 (점(.) 제외, 소문자 기준)
# ---------------------------------------------------------------------------
_IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp"}
_AUDIO_EXTS = {"mp3", "wav", "flac", "m4a", "ogg"}
_VIDEO_EXTS = {"mp4", "avi", "mov", "mkv", "webm"}
_DOCUMENT_EXTS = {"pdf", "hwp", "docx", "doc", "pptx", "xlsx"}
_TEXT_EXTS = {"txt", "md", "ocr"}

# modality → 추출기 체인 (dispatch 계획)
_EXTRACTOR_PLAN: Dict[str, List[str]] = {
    "image": ["vlm", "embedding"],
    "audio": ["vlm"],
    "video": ["vlm"],
    "document": ["ocr", "llm", "ner"],
    "text": ["ocr", "llm", "ner"],
    "unknown": [],
}

# 사람이 읽을 수 있는 분기 근거 (rationale)
_RATIONALE: Dict[str, str] = {
    "image": "이미지 저작물 — 생성형 VLM으로 설명/속성 추출 + 임베딩으로 유사도 계산",
    "audio": "오디오 저작물 — VLM(멀티모달)으로 속성 추출 (임베딩은 추후)",
    "video": "비디오 저작물 — 대표 프레임을 VLM으로 속성 추출",
    "document": "문서 — 기존 OCR → LLM → NER 파이프라인으로 메타데이터 추출",
    "text": "텍스트(OCR 결과 포함) — OCR 단계는 통과/스킵 후 LLM → NER 추출",
    "unknown": "지원하지 않는 확장자 — 처리할 추출기 없음",
}


def _extension(file_path: str) -> str:
    """파일 경로에서 확장자를 점 없이 소문자로 추출. 없으면 빈 문자열."""
    ext = os.path.splitext(str(file_path))[1]
    return ext[1:].lower() if ext.startswith(".") else ext.lower()


def detect_modality(file_path: str) -> str:
    """
    파일 확장자로 modality를 판별한다.

    반환값: 'image' | 'audio' | 'video' | 'document' | 'text' | 'unknown'
    """
    ext = _extension(file_path)
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _AUDIO_EXTS:
        return "audio"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _DOCUMENT_EXTS:
        return "document"
    if ext in _TEXT_EXTS:
        return "text"
    return "unknown"


def route(file_path: str) -> Dict[str, object]:
    """
    입력 파일에 대한 dispatch 계획을 돌려준다.

    예시 반환:
        {
            "modality": "image",
            "extractors": ["vlm", "embedding"],
            "rationale": "...",
            "file_path": "...",
            "extension": "jpg",
        }
    """
    modality = detect_modality(file_path)
    return {
        "modality": modality,
        # 리스트는 호출자가 변형해도 모듈 상수가 오염되지 않게 복사해서 반환
        "extractors": list(_EXTRACTOR_PLAN[modality]),
        "rationale": _RATIONALE[modality],
        "file_path": str(file_path),
        "extension": _extension(file_path),
    }


if __name__ == "__main__":
    import json

    examples = [
        "사진저작물__국물떡볶이.jpg",   # image
        "interview_recording.mp3",      # audio
        "promo_clip.mp4",               # video
        "저작재산권_이용허락_계약서.pdf",  # document
        "contract_scan.ocr",            # text (OCR 결과)
        "archive.zip",                  # unknown
    ]

    print("=" * 70)
    print("Modality Router — dispatch plan per example file")
    print("=" * 70)
    for fp in examples:
        plan = route(fp)
        print(f"\n[{fp}]")
        print(json.dumps(plan, ensure_ascii=False, indent=2))
