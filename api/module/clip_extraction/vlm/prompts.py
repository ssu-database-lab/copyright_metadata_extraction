"""
Shared extraction prompt for the VLM comparison.

The SAME prompt goes to both Gemma 4 and Qwen3-VL so the comparison is fair.
The prompt asks for structured JSON aligned to the unified-schema fields the
existing pipeline already produces, plus the open-vocabulary fields a VLM is
uniquely good at (description, keywords, visual properties).

KEY design choice: we explicitly tell the model to classify work_type by
MEDIUM (촬영/그림/영상), not by SUBJECT. This is the exact failure mode CLIP
zero-shot couldn't fix (a photo of a building scored as 건축저작물). A
generative VLM can just be told the rule.
"""

SYSTEM_PROMPT = (
    "당신은 공유저작물(공공저작물·CCL·퍼블릭 도메인) 메타데이터 추출 전문가입니다. "
    "주어진 이미지를 분석하여 저작물의 속성 정보를 정확하게 추출합니다. "
    "반드시 유효한 JSON 객체 하나만 출력하고, 그 외의 설명·코드블록·마크다운은 절대 출력하지 마세요."
)

# work_type candidate set mirrors labels.WORK_TYPE_LABELS (schema-aligned)
USER_PROMPT = """이 이미지를 분석하여 아래 JSON 스키마에 맞게 공유저작물 메타데이터를 추출하세요.

{
  "description": "이미지 내용을 한국어로 2~3문장으로 객관적으로 설명",
  "work_type": "아래 목록 중 '매체(medium)' 기준으로 하나만 선택",
  "work_type_reason": "그 매체로 판단한 근거 한 문장",
  "keywords": ["핵심 키워드 5~7개"],
  "main_subjects": ["이미지에 보이는 주요 객체/피사체 목록"],
  "dominant_colors": ["주요 색상 2~3개 (한국어)"],
  "text_in_image": "이미지 안에 보이는 글자(있으면 그대로, 없으면 null)",
  "scene_type": "실내/실외/스튜디오/그래픽 등",
  "estimated_quality": "고화질/중간/저화질 중 하나"
}

work_type 목록: 사진저작물, 영상저작물, 어문저작물, 음악저작물, 미술저작물, 건축저작물, 도형저작물, 컴퓨터프로그램저작물, 연극저작물, 기타

★ 매우 중요: work_type은 '무엇을 찍었는가(피사체)'가 아니라 '어떤 매체로 만들어졌는가'로 판단하세요.
  - 건물을 카메라로 촬영한 사진 → "사진저작물" (O), "건축저작물" (X)
  - 손으로 그린 그림/회화 → "미술저작물"
  - 지도·도표·설계도 → "도형저작물"
  - 동영상의 한 장면(프레임) → "영상저작물"

JSON 객체 하나만 출력하세요."""

# ── PENDING PROMPT TUNING (deferred 2026-06-08 by project decision) ──────────
# The 2026-06-08 Gemma-vs-Qwen 15-image run surfaced two work_type disambiguation
# gaps that cause systematic model divergence:
#   (1) LOGO / brand symbol / typography  → Gemma calls 미술저작물, Qwen calls 도형저작물
#   (2) TEXT-DOCUMENT SCAN (book/news/report) → Gemma over-calls 미술/사진, Qwen → 어문
# Adding disambiguation rules would converge the models, BUT the correct convention
# is a KOGL labeling decision. We are intentionally NOT adding rules until the
# requested 구분 명세서 (KOGL metadata field-definition doc) arrives — see
# docs/task_status_blockers_20260529.md and the data request to KCISA.
# DO NOT add logo / text-scan rules here without that spec.


# ── English-instruction variant (values still in Korean) ────────────────────
# Hypothesis: English instructions improve instruction-following (this project's
# OCR/LLM pipeline already uses an English prompt), while "write all values in
# Korean" keeps the metadata Korean and may reduce Qwen's foreign-script bleed.
# The work_type RULES are held IDENTICAL to the Korean version on purpose, so an
# A/B test isolates the effect of prompt LANGUAGE, not rule changes.
SYSTEM_PROMPT_EN = (
    "You are an expert metadata extractor for public-domain works (public-sector "
    "works, CC-licensed, public domain). Analyze the given image and extract its "
    "attribute metadata accurately. Output exactly ONE valid JSON object and nothing "
    "else — no explanation, no code fences, no markdown. IMPORTANT: write every field "
    "VALUE in Korean (한국어). Exceptions: 'work_type' must be copied verbatim from the "
    "provided Korean list, and 'text_in_image' must be transcribed exactly as it appears."
)

USER_PROMPT_EN = """Analyze this image and extract public-domain-work metadata into the JSON schema below. Write every value in Korean (한국어).

{
  "description": "<2-3 objective sentences in Korean describing the image>",
  "work_type": "<choose exactly ONE from the list below, by MEDIUM>",
  "work_type_reason": "<one Korean sentence justifying the medium choice>",
  "keywords": ["<5-7 key terms in Korean>"],
  "main_subjects": ["<main objects/subjects visible, in Korean>"],
  "dominant_colors": ["<2-3 dominant colors, in Korean>"],
  "text_in_image": "<text visible in the image, transcribed verbatim; null if none>",
  "scene_type": "<indoor/outdoor/studio/graphic etc., written in Korean>",
  "estimated_quality": "<one of: 고화질 / 중간 / 저화질>"
}

work_type list (Korean — copy one verbatim): 사진저작물, 영상저작물, 어문저작물, 음악저작물, 미술저작물, 건축저작물, 도형저작물, 컴퓨터프로그램저작물, 연극저작물, 기타

CRITICAL: classify work_type by the MEDIUM the work was created in, NOT by its subject.
  - a building captured with a camera → "사진저작물" (correct), NOT "건축저작물"
  - a hand-drawn picture/painting → "미술저작물"
  - a map/chart/blueprint → "도형저작물"
  - a single frame from a video → "영상저작물"

Output only one JSON object."""


def get_prompts(lang: str = "ko") -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for lang 'ko' (default) or 'en'."""
    if lang == "en":
        return SYSTEM_PROMPT_EN, USER_PROMPT_EN
    if lang == "ko":
        return SYSTEM_PROMPT, USER_PROMPT
    raise ValueError(f"unknown prompt lang: {lang!r} (use 'ko' or 'en')")


def build_messages(image_data_url: str, image_first: bool = True) -> list[dict]:
    """
    Build OpenAI-compatible chat messages for one image.

    image_first=True puts the image part before the text part — Gemma 4's
    documented preference. Qwen3-VL is order-agnostic, so this is safe for both.
    """
    image_part = {"type": "image_url", "image_url": {"url": image_data_url}}
    text_part = {"type": "text", "text": USER_PROMPT}
    content = [image_part, text_part] if image_first else [text_part, image_part]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
