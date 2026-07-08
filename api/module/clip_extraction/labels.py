"""
Korean candidate-label vocabularies for zero-shot classification.

These labels mirror the unified-schema fields produced by the existing
LLM extraction pipeline, so a VLM verdict can be fused into the same
67-field metadata object.
"""

# --- work_type --- (저작권법 제4조: 저작물의 예시) -------------------------
WORK_TYPE_LABELS = [
    "사진저작물",
    "영상저작물",
    "어문저작물",
    "음악저작물",
    "미술저작물",
    "건축저작물",
    "도형저작물",
    "컴퓨터프로그램저작물",
    "연극저작물",
    "기타",
]

# English aliases for OpenAI CLIP (no Korean support) --------------------
WORK_TYPE_LABELS_EN = [
    "a photograph",
    "a video still or cinematographic frame",
    "a literary work or document scan",
    "a music score or audio waveform",
    "a painting, drawing, or other artwork",
    "an architectural work or building blueprint",
    "a chart, diagram, map, or technical drawing",
    "computer program source code",
    "a theatrical or stage performance still",
    "other miscellaneous work",
]

# --- coarse keyword tags (subject) for 사진/영상 works -------------------
SUBJECT_LABELS = [
    "인물",
    "풍경",
    "동물",
    "식물",
    "음식",
    "건물 또는 건축물",
    "문화재 또는 유물",
    "도시 거리",
    "자연 또는 야외",
    "예술 작품",
    "스포츠 또는 활동",
    "교통수단",
    "행사 또는 공연",
    "추상 또는 패턴",
]

SUBJECT_LABELS_EN = [
    "people or portrait",
    "landscape or scenery",
    "animals",
    "plants or flowers",
    "food",
    "buildings or architecture",
    "cultural heritage or artifacts",
    "city street",
    "nature or outdoors",
    "artwork",
    "sports or activity",
    "vehicle or transportation",
    "event or performance",
    "abstract or pattern",
]

# --- 라이선스 / 권리 유형 (for downstream rights classification) ---------
LICENSE_TYPE_LABELS = [
    "공공저작물 자유이용 (KOGL)",
    "크리에이티브 커먼즈 (CC)",
    "퍼블릭 도메인",
    "저작권 보호 대상",
    "저작권 정보 미상",
]


# Map Korean label list to English-aliased list (used by CLIP-en pipeline)
EN_ALIAS = {
    "work_type": dict(zip(WORK_TYPE_LABELS, WORK_TYPE_LABELS_EN)),
    "subject": dict(zip(SUBJECT_LABELS, SUBJECT_LABELS_EN)),
}


def label_set(name: str, language: str = "ko") -> list[str]:
    """Return a candidate label list by name + language ('ko' or 'en')."""
    mapping = {
        ("work_type", "ko"): WORK_TYPE_LABELS,
        ("work_type", "en"): WORK_TYPE_LABELS_EN,
        ("subject", "ko"): SUBJECT_LABELS,
        ("subject", "en"): SUBJECT_LABELS_EN,
        ("license", "ko"): LICENSE_TYPE_LABELS,
    }
    if (name, language) not in mapping:
        raise KeyError(f"Unknown label set: {name=} {language=}")
    return mapping[(name, language)]
