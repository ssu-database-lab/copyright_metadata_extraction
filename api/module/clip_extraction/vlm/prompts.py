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
# "hashtags" 줄(M2b 수정 3): v4 T-002 키 18 — VLM 이 직접 '#태그' 목록을 내게 한다(이미지·영상 공통).
# 없으면 eval23 직렬화기가 keywords 로 대체하지만 그러면 hashtags_source='keywords' 로 남는다.
USER_PROMPT = """이 이미지를 분석하여 아래 JSON 스키마에 맞게 공유저작물 메타데이터를 추출하세요.

{
  "description": "이미지 내용을 한국어로 2~3문장으로 객관적으로 설명",
  "work_type": "아래 목록 중 '매체(medium)' 기준으로 하나만 선택",
  "work_type_reason": "그 매체로 판단한 근거 한 문장",
  "keywords": ["키워드 5~7개 — 저작물명에 주제어(대상·장소·행사·단어)가 있으면 화면과 모순되지 않는 한 먼저 넣고, 나머지는 화면의 주 피사체·장소·행위를 나타내는 일반 명사. 장르·분위기·평가·추측 단어는 쓰지 않는다"],
  "hashtags": ["해시태그 5~10개 — '#태그' 형식, 띄어쓰기 없이 붙여 쓴 명사. 화면에 보이는 대상·장소·행위·매체 특성을 나타낸다. 저작물명의 주제어(대상·단어·장소)는 화면과 모순되지 않으면 넣는다. 장르·분위기·평가·추측 태그(예: #액션 #여행 #감성 #자연 #일상 #풍경), 인명·나이, 라이선스·출처·기증 단어는 쓰지 않는다"],
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

키워드·해시태그 작성 예시 (형식과 고르는 방식만 참고하고, 내용은 이 저작물에 맞게 쓴다):
- 사진, 저작물명 '춘천댐 공사 전경' → 키워드: 춘천댐, 댐, 공사, 전경, 흑백사진, 철골, 설비, 강원도 / 해시태그: #춘천댐 #댐 #공사 #전경 #흑백사진 #철골 #설비 #계곡
- 사진, 저작물명 '성남 봉국사 대광명전' → 키워드: 봉국사, 대광명전, 사찰, 전각, 단청, 목조건물 / 해시태그: #사찰 #전각 #단청 #처마 #현판 #목조건물 #불교건축
- 영상, 저작물명 '전문용어_수어_교통_주취 운전 (酒醉運轉)' → 키워드: 수어, 주취운전, 운전, 교통, 전문용어 / 해시태그: #수어 #주취운전 #운전 #손동작 #정면구도 #초록배경 #검은셔츠 #상반신
- 영상, 저작물명 '정선군_산촌_마을' → 키워드: 정선, 산촌, 마을, 전경, 드론, 항공, 풍경, 강원도 / 해시태그: #정선군 #산촌 #마을 #드론 #항공 #산 #들판 #전경
규칙: 저작물명의 대상·장소를 먼저, 그다음 화면의 주 피사체(구체어와 그 상위어를 함께, 예: #춘천댐 #댐), 매체 특성(흑백사진·드론·항공), 구도·배경을 쉬운 일반 명사로 쓴다.

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


# ── 영상 트랙 프롬프트 ───────────────────────────────────────────────────────
# 이미지 프롬프트를 그대로 쓰면 모델이 프레임을 '개별 이미지 N장'으로 취급한다.
# 영상은 전개(시간 흐름)를 봐야 하므로 각 프레임의 시점을 알려주고 "영상 전체"를
# 파악하라고 명시한다. 영상 모델 비교시험(40콜)에서 이 형태로 측정했다.
VIDEO_SYSTEM_PROMPT = (
    "당신은 공유저작물(공공저작물·CCL·퍼블릭 도메인) 메타데이터 추출 전문가입니다. "
    "주어진 프레임들은 하나의 영상에서 시간순으로 추출한 대표 장면입니다. "
    "개별 이미지가 아니라 영상 전체의 내용을 파악하여 속성 정보를 추출합니다. "
    "반드시 유효한 JSON 객체 하나만 출력하고, 그 외의 설명·코드블록·마크다운은 절대 출력하지 마세요."
)

_VIDEO_USER_TEMPLATE = """다음 {n}장은 하나의 영상에서 시간순으로 추출한 대표 프레임입니다(각 시점: {times}).
개별 이미지가 아니라 **영상 전체**의 내용을 파악하여 아래 JSON 스키마로만 답하세요.

{{
  "description": "영상 내용을 한국어 2~3문장으로 구체적으로 (주 장면과, 시작부터 끝까지 화면이 어떻게 바뀌는지 — 카메라·피사체 이동, 장면 전환)",
  "work_type": "영상저작물",
  "work_type_reason": "그 매체로 판단한 근거 한 문장",
  "keywords": ["키워드 5~7개 — 저작물명에 주제어(대상·장소·행사·단어)가 있으면 화면과 모순되지 않는 한 먼저 넣고, 나머지는 화면의 주 피사체·장소·행위를 나타내는 일반 명사. 장르·분위기·평가·추측 단어는 쓰지 않는다"],
  "hashtags": ["해시태그 5~10개 — '#태그' 형식, 띄어쓰기 없이 붙여 쓴 명사. 화면에 보이는 대상·장소·행위·매체 특성을 나타낸다. 저작물명의 주제어(대상·단어·장소)는 화면과 모순되지 않으면 넣는다. 장르·분위기·평가·추측 태그(예: #액션 #여행 #감성 #자연 #일상 #풍경), 인명·나이, 라이선스·출처·기증 단어는 쓰지 않는다"],
  "main_subjects": ["영상에 등장하는 주요 객체/인물/피사체 목록"],
  "dominant_colors": ["주요 색상 2~3개 (한국어)"],
  "text_in_image": "화면에 보이는 글자·자막(있으면 그대로, 없으면 null)",
  "scene_type": "실내/실외/스튜디오/애니메이션/그래픽 등",
  "estimated_quality": "고화질/중간/저화질 중 하나"
}}

★ 고유명사(인물명·작품명)는 확실하지 않으면 지어내지 말고 일반 명사로 쓰세요.
   비교시험에서 모델들이 등장인물 이름을 창작하는 사례가 다수 확인되었습니다.
★ 줄거리·사건 전개·등장인물의 역할·장르(범죄·액션·드라마·모험 등)는 쓰지 마세요. 설명·키워드·해시태그 모두
   프레임에 실제로 보이는 대상·장소·행위·화면 특성(흑백, 자막, 구도 등)만으로 쓰세요.
★ 모든 값은 한국어로 작성하세요.

키워드·해시태그 작성 예시 (형식과 고르는 방식만 참고하고, 내용은 이 저작물에 맞게 쓴다):
- 사진, 저작물명 '춘천댐 공사 전경' → 키워드: 춘천댐, 댐, 공사, 전경, 흑백사진, 철골, 설비, 강원도 / 해시태그: #춘천댐 #댐 #공사 #전경 #흑백사진 #철골 #설비 #계곡
- 사진, 저작물명 '성남 봉국사 대광명전' → 키워드: 봉국사, 대광명전, 사찰, 전각, 단청, 목조건물 / 해시태그: #사찰 #전각 #단청 #처마 #현판 #목조건물 #불교건축
- 영상, 저작물명 '전문용어_수어_교통_주취 운전 (酒醉運轉)' → 키워드: 수어, 주취운전, 운전, 교통, 전문용어 / 해시태그: #수어 #주취운전 #운전 #손동작 #정면구도 #초록배경 #검은셔츠 #상반신
- 영상, 저작물명 '정선군_산촌_마을' → 키워드: 정선, 산촌, 마을, 전경, 드론, 항공, 풍경, 강원도 / 해시태그: #정선군 #산촌 #마을 #드론 #항공 #산 #들판 #전경
규칙: 저작물명의 대상·장소를 먼저, 그다음 화면의 주 피사체(구체어와 그 상위어를 함께, 예: #춘천댐 #댐), 매체 특성(흑백사진·드론·항공), 구도·배경을 쉬운 일반 명사로 쓴다.

JSON 객체 하나만 출력하세요."""


def video_user_prompt(frame_times) -> str:
    """프레임 시점 목록을 넣은 영상용 사용자 프롬프트."""
    times = ", ".join(f"{float(t):.0f}초" for t in frame_times) if frame_times else "미상"
    return _VIDEO_USER_TEMPLATE.format(n=len(frame_times or []), times=times)


# ── 계약서 문맥(선택) — v4 T-002 옵션 B ──────────────────────────────────────
# 계약서를 규칙으로 읽어 얻은 저작물명·종별을 VLM 에게 '참고'로 준다 (M2 spec A4,
# contract_context.parse 가 만든 값). 2026-10-10 변경: T-002 는 키 1–16 을 채점하지 않고, 정답 지침 v0.3 규칙 6-1 이
# 제목의 주제어(대상·단어·장소)를 설명·해시태그에 넣으라고 하므로, 화면과 모순되지 않는 제목 주제어를 키워드·해시태그에
# 쓰게 한다 (docs/v4/tags_investigation_20261010.md — 보류 파일럿에서 해시태그 0.281→0.394, 키워드 0.267→0.350).
# 문맥이 비어 있으면 프롬프트는 글자 하나 바뀌지 않는다 (웹앱 기본 동작 유지).
_CONTEXT_LINE = "참고(계약서): {items} — 설명은 화면에 보이는 것만 쓰되, 저작물명이 알려 주는 주제어(대상·단어·장소)는 화면과 모순되지 않으면 설명·키워드·해시태그에 넣는다. 수어 영상이면 저작물명의 단어를 수어로 표현하는 영상이다."


def _clean_context_value(value):
    """None/공백 → None, 그 외 → 공백을 한 칸으로 접은 문자열."""
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def user_prompt_with_context(base: str, title=None, work_type=None) -> str:
    """base 프롬프트 끝에 계약서 참고 줄을 붙인다. title/work_type 둘 다 없으면 base 그대로."""
    items = []
    title = _clean_context_value(title)
    work_type = _clean_context_value(work_type)
    if title:
        items.append(f"저작물명 '{title}'")
    if work_type:
        items.append(f"종별 '{work_type}'")
    if not items:
        return base
    return base.rstrip() + "\n\n" + _CONTEXT_LINE.format(items=", ".join(items))
