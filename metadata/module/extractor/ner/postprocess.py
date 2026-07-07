"""NER 출력 후처리 — closed-vocab 매칭 + form-cue line capture + 휴리스틱 필터.

ner_predict 가 OCR 텍스트와 NER 집계 결과를 가지고 호출.
NER 노이즈가 심한 라벨은 closed-vocab 키워드로 대체하고, 한국 양식의 cue
(성명:, 주소:, 전화번호:, 기관명: 등) 뒤 줄 끝까지를 entity 로 union 한다.
"""
from __future__ import annotations

import re
from typing import Dict, List


MetadataDict = Dict[str, List[str]]


# ---------------------------------------------------------------------------
# 공통 regex
# ---------------------------------------------------------------------------

_HANGUL_RE = re.compile(r"[가-힣]")
# 회사/기관은 한자 표기(帝國火災保險株式會社)도 유효 → CJK 통합 한자 포함.
_CJK_RE = re.compile(r"[가-힣㐀-䶿一-鿿]")
_PERIOD_CUE_RE = re.compile(
    r"(?:년|월|일|개월|시부터|만료|까지|보호기간|기한|동안|일\s*간|개월\s*간)"
)
_DATE_FULL_RE = re.compile(
    r"\d{4}\s?\.\s?\d{1,2}\s?\.\s?\d{1,2}\s?\.?|\d{4}년\s?\d{1,2}월\s?\d{1,2}일|\d{4}-\d{1,2}-\d{1,2}"
)
# ri_period 는 날짜 형식(유효기간 : 2080-04-30)도 유효값으로 허용.
_DATEISH_RE = re.compile(r"\d{4}\s?[.\-/]\s?\d{1,2}\s?[.\-/]\s?\d{1,2}|\d{4}년|\d{1,2}\s?개월|\d{1,2}\s?년")
# 파일형식(copyright_status = digital status / file type) — 닫힌 확장자 집합.
_FILE_EXT = (
    "jpg", "jpeg", "png", "gif", "bmp", "tif", "tiff", "webp", "svg", "psd", "ai", "eps",
    "pdf", "hwp", "hwpx", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "txt", "rtf", "csv",
    "mp3", "wav", "flac", "m4a", "aac", "mp4", "avi", "mov", "mkv", "wmv", "webm",
    "zip", "egg", "obj", "fbx", "glb", "gltf", "stl", "dwg", "json", "xml",
)
_FILE_EXT_RE = re.compile(
    r"\.(" + "|".join(_FILE_EXT) + r")\b", re.IGNORECASE
)
# 권리주체 cue (저작권자/기관/원본소유자 등) — 지자체·기관명 공통 앵커.
_AUTHOR_CUE = (
    r"(?:기관(?:명)?|저작(?:인접)?권자|원본\s*소유자|소장\s*처|소장\s*기관|제공\s*기관|"
    r"소유자|저작자|공동\s*저작자|발행\s*(?:처|기관)|제작\s*(?:사|기관))"
)
# 지자체(행정구역) = 소재지/주소 (사용자 확정: 지자체=address). cue 뒤 지역명 캡처.
# 뒤에 한글이 더 붙으면(도청/도교육청 등 기관명) 지역명으로 보지 않음 → 부정 룩어헤드.
_REGION_ADDR_RE = re.compile(
    _AUTHOR_CUE + r"\s*[:：]\s*"
    r"([가-힣]{2,7}(?:특별자치시|특별자치도|특별시|광역시|도|시|군|구)"
    r"(?:\s+[가-힣]{2,7}(?:시|군|구))?)"
    r"(?![가-힣])",
    re.MULTILINE,
)
# 기관/회사 접미사 — cue 뒤 값이 이 접미사로 끝나면 company. (시/군/구/도 는 address 로 별도 처리)
_ORG_SUFFIXES = (
    "교육지원청", "교육청", "시청", "도청", "구청", "군청", "문화재청", "청",
    "국립현충원", "현충원", "박물관", "미술관", "도서관", "기념관", "전시관", "문예회관",
    "체육관", "과학관", "문화원", "국악원", "연구원", "연구소", "진흥원", "대학원", "관",
    "대학교", "대학", "고등학교", "중학교", "초등학교", "학교",
    "문화재단", "장학재단", "재단법인", "사단법인", "재단", "협회", "조합", "연맹", "학회",
    "위원회", "보존회", "종친회", "동문회", "번영회", "회",
    "주식회사", "방송국", "방송", "신문사", "출판사", "컨설팅", "스튜디오", "프로덕션",
    "엔터테인먼트", "공사", "공단", "사업단", "센터", "종단", "교구", "본사", "본부", "지사",
    "화랑", "갤러리", "백화점", "은행", "극단", "악단", "합창단", "무용단",
    "대사전", "사전", "규장각",
    # 단자 접미사 — cue 앵커 + 사람이름 끝음절과 잘 안 겹치는 것만. (원/정/전/성 은
    # 김미정·이서원·김태성 처럼 이름 흔한 끝음절이라 제외 → 재학습 모델이 문맥으로 처리)
    "부", "처", "소", "사", "각", "당", "궁", "암",
    "株式會社", "有限會社", "社",
)
_ORG_CUE_VALUE_RE = re.compile(_AUTHOR_CUE + r"\s*[:：]\s*([^\n,/·]+?)\s*(?=[\n,/()·]|$)", re.MULTILINE)
# 기관 접두어 — 이 접두어로 시작하면 기관(공공/국립 계열).
_ORG_PREFIXES = (
    "국립", "공립", "시립", "도립", "구립", "군립", "사립", "국", "한국", "대한", "재단법인",
    "사단법인", "학교법인", "의료법인", "정부", "중앙", "서울대학교", "국가", "국제", "세계",
)
# 크레딧 형식 "미술-장종선 / 의상디자인·제작-홍장희" → 역할(미술/의상디자인·제작)=position.
_CREDIT_ROLE_RE = re.compile(r"([가-힣][가-힣·]{1,11})\s*[-–—]\s*[가-힣]")
_CREDIT_ROLE_KEYWORDS = ("디자인", "제작", "장치", "도안", "감독", "디렉터", "작화", "채색", "분장")
_CREDIT_ROLE_LEXICON = frozenset({
    "미술", "의상", "소품", "조명", "음악", "안무", "각본", "연출", "촬영", "편집", "분장",
    "가발", "효과", "무대", "세트", "특수", "작화", "채색", "장치", "도안", "제작", "디자인",
    "무대장치", "의상디자인", "소품디자인", "장치디자인", "장치도안", "무대장치제작",
    "소품제작", "의상제작", "미술감독", "무대미술",
})
# 광역 시·도 전체명 — 본문 어디에 있어도 소재지로 인정.
_SIDO_NAMES = (
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시",
    "울산광역시", "세종특별자치시", "경기도", "강원특별자치도", "강원도", "충청북도",
    "충청남도", "전북특별자치도", "전라북도", "전라남도", "경상북도", "경상남도",
    "제주특별자치도", "제주도",
)
# 직위/직책 lexicon — parenthetical (…연구위원) 등 cue 없는 직위 회수.
POSITION_LEXICON = (
    "대표이사", "부대표", "대표", "부사장", "사장", "부회장", "회장", "상무이사", "상무",
    "전무이사", "전무", "이사장", "이사", "감사", "총장", "부총장", "학장", "원장", "부원장",
    "센터장", "본부장", "실장", "국장", "과장", "부장", "차장", "팀장", "반장", "소장",
    "수석연구위원", "책임연구원", "선임연구원", "연구위원", "연구단장", "연구원", "연구사",
    "교수", "부교수", "조교수", "겸임교수", "초빙교수", "강사", "박사", "석사",
    "단장", "위원장", "위원", "주임", "대리", "주무관", "사무관", "서기관",
    "감독", "연출", "프로듀서", "촬영감독", "작가", "기자", "편집장", "아나운서",
    "큐레이터", "학예사", "학예연구사", "관장", "지휘자", "연주자",
)


# ---------------------------------------------------------------------------
# closed vocabulary — NER 가 노이즈 심해서 키워드 직접 매칭이 더 정확.
# ---------------------------------------------------------------------------

CLOSED_VOCAB: Dict[str, List[str]] = {
    "ri_copyright": [
        "저작재산권", "저작인접권", "복제권", "배포권", "대여권",
        "방송권", "전송권", "공중송신권", "전시권", "공연권",
        "2차 저작물 작성권", "2차저작물 작성권", "2차 저작물작성권",
        "성명표시권", "동일성유지권", "공표권", "저작인격권",
    ],
    "ri_contract_type": [
        "양도 계약", "양도계약", "이용허락", "위탁 계약", "위탁계약",
        "용역 계약", "용역계약",
    ],
    "ri_info": [
        "공공저작물 자유이용", "공공저작물자유이용",
        "CC BY", "CC-BY", "CC0",
    ],
    "copyright_type": [
        "사진저작물", "영상저작물", "음악저작물", "어문저작물",
        "미술저작물", "공연저작물", "방송저작물", "건축저작물",
        "도형저작물", "응용미술저작물", "컴퓨터프로그램저작물",
    ],
    "copyright_status": ["공개", "비공개", "온라인", "오프라인"],
    "copyright_language": ["한국어", "영어", "일본어", "중국어", "독일어", "프랑스어"],
}


# ---------------------------------------------------------------------------
# form-cue — 구조화된 한국 양식의 라벨 cue 뒤 "줄 끝까지" 캡처.
# 주의: \s 는 newline 을 포함 → inline whitespace 만 허용 ([ \t]).
# ---------------------------------------------------------------------------

FORM_CUE_PATTERNS: Dict[str, List[re.Pattern]] = {
    "name": [
        re.compile(r"성[ \t]*명[ \t]*[:：][ \t]+([^\n(]+?)(?:[ \t]*\(서명\)|[ \t]*$)",
                   re.MULTILINE),
        # `명` 필수 — "대표자:" 단독은 OCR 변형 (주소 잘못 라벨).
        re.compile(r"대표자[ \t]*명[ \t]*[:：][ \t]+([^\n(]+?)(?:[ \t]*\(|[ \t]*$)",
                   re.MULTILINE),
    ],
    "address": [
        re.compile(
            r"(?:^|\n)[ \t]*[○●·•∙*◦▫▪]?[ \t]*(?:대표자?[ \t]*)?주[ \t]*소[ \t]*[:：][ \t]+"
            r"(.+?)(?:[ \t]*전화번호|[ \t]*대표자[ \t]*연락처|[ \t]*$)",
            re.MULTILINE,
        ),
    ],
    "phone": [
        re.compile(r"전[ \t]*화[ \t]*번[ \t]*호[ \t]*[:：][ \t]+([^\n]+?)[ \t]*$",
                   re.MULTILINE),
        re.compile(r"대표자[ \t]*연락처[ \t]*[:：][ \t]+([^\n]+?)[ \t]*$",
                   re.MULTILINE),
    ],
    "company": [
        re.compile(
            r"(?:^|\n)[ \t]*[○●·•∙*◦▫▪]?[ \t]*기[ \t]*관[ \t]*명?[ \t]*[:：][ \t]+"
            r"([^\n]+?)(?:[ \t]+대표자|[ \t]+연락처|[ \t]+전화|[ \t]*$)",
            re.MULTILINE,
        ),
        re.compile(
            r"(?:^|\n)[ \t]*[○●·•∙*◦▫▪]?[ \t]*소[ \t]*속[ \t]*[:：][ \t]+([^\n(]+?)(?:[ \t]*\(|[ \t]*$)",
            re.MULTILINE,
        ),
    ],
    "department": [
        re.compile(r"(?:^|\n)[ \t]*(?:부서|부서명)[ \t]*[:：][ \t]+([^\n]+?)[ \t]*$",
                   re.MULTILINE),
    ],
    "position": [
        re.compile(r"(?:^|\n)[ \t]*(?:직위|직책|직급)[ \t]*[:：][ \t]+([^\n]+?)[ \t]*$",
                   re.MULTILINE),
    ],
    "email": [
        re.compile(r"이?\s*메일[ \t]*[:：][ \t]+([^\s\n]+@[^\s\n]+)"),
    ],
    "ri_period": [
        re.compile(r"보유\s*및\s*이용\s*기간[ \t]*[:：][ \t]+([^\n]{5,80})"),
    ],
    "ri_law_reference": [
        re.compile(r"(저작권법\s*제\s*\d+조(?:\s*의\s*\d+)?)"),
        re.compile(r"(공공데이터의\s*제공\s*및\s*이용활성화에\s*관한\s*법률)"),
    ],
    "copyright_kotitle": [
        re.compile(
            r"---\s*Page\s*1/\d+\s*---\s*\n+[ \t]*"
            r"([가-힣A-Za-z0-9 \t·,()'\"-]{5,80}(?:계약서|동의서|확약서|증서|신청서|양도서))",
        ),
    ],
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _dedup_keep_order(items: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _apply_closed_vocab(raw_text: str, label: str) -> List[str]:
    vocab = CLOSED_VOCAB.get(label, [])
    seen: set = set()
    found: List[str] = []
    for kw in vocab:
        if kw in raw_text and kw not in seen:
            seen.add(kw)
            found.append(kw)
    return found


def _apply_form_cues(raw_text: str, label: str) -> List[str]:
    out: List[str] = []
    seen: set = set()
    for pat in FORM_CUE_PATTERNS.get(label, []):
        for m in pat.finditer(raw_text):
            captured = (m.group(1) if m.groups() else m.group(0)).strip()
            if len(captured) >= 2 and captured not in seen:
                seen.add(captured)
                out.append(captured)
    return out


def _extract_file_ext(raw_text: str) -> List[str]:
    """copyright_status = digital status / file type → 파일 확장자 추출 (닫힌 집합)."""
    out: List[str] = []
    seen: set = set()
    for m in _FILE_EXT_RE.finditer(raw_text):
        ext = m.group(1).lower()
        if ext not in seen:
            seen.add(ext)
            out.append(ext)
    return out


def _extract_region_address(raw_text: str) -> List[str]:
    """지자체(행정구역명) → 소재지/주소. cue 뒤 지역명 + 광역시도 전체명."""
    out: List[str] = []
    seen: set = set()
    for m in _REGION_ADDR_RE.finditer(raw_text):
        val = re.sub(r"\s+", " ", m.group(1)).strip()
        if val and val not in seen:
            seen.add(val)
            out.append(val)
    for sido in _SIDO_NAMES:
        if sido in raw_text and sido not in seen:
            seen.add(sido)
            out.append(sido)
    return out


# 다인 표(연번 성명 주소 전화번호 서명) 행: "32 오금숙 청주시 … 010-5418-6104".
_TABLE_ROW_RE = re.compile(r"^\s*\d{1,3}\s+([가-힣]{2,4})(?:\s+(.*))?$")
_PHONE_IN_LINE_RE = re.compile(r"0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4}")


def _extract_table_rows(raw_text: str):
    """다인 표의 각 행을 컬럼(번호|이름|주소|전화)으로 분해.

    폼-cue 가 없는 표에서 NER/regex 가 행 구조(이름·전화 경계)를 못 잡는 문제 보정.
    번호로 시작+한글 이름+전화번호 를 가진 행이 3개 이상일 때만 발동(일반 문서 no-op).
    Returns {"name","address","phone"} 또는 None.
    """
    parsed = []
    for ln in raw_text.splitlines():
        m = _TABLE_ROW_RE.match(ln.strip())
        if not m:
            continue
        name = m.group(1)
        rest = (m.group(2) or "").strip()
        pm = _PHONE_IN_LINE_RE.search(rest)
        phone = pm.group(0).strip() if pm else None
        addr = (rest[:pm.start()] if pm else rest).strip(" -·,")
        parsed.append((name, addr, phone))
    if sum(1 for _, _, p in parsed if p) < 3:      # 전화 있는 행 3개 미만 → 표 아님
        return None
    return {
        "name":    _dedup_keep_order([n for n, _, _ in parsed]),
        "address": _dedup_keep_order([a for _, a, _ in parsed if a and len(a) >= 5]),
        "phone":   _dedup_keep_order([p for _, _, p in parsed if p]),
    }


def _extract_org_company(raw_text: str) -> List[str]:
    """권리주체 cue 뒤 값이 기관/회사 접미사로 끝나면 company 로 회수.

    NER 이 '문화재청'·'독립기념관'·'안성시청' 등 기관명을 name 으로 오태깅하는 문제 보정.
    """
    out: List[str] = []
    seen: set = set()
    for m in _ORG_CUE_VALUE_RE.finditer(raw_text):
        val = re.sub(r"\s+", " ", m.group(1)).strip()
        if not val or len(val) < 2 or val in seen:
            continue
        val_ns = val.replace(" ", "")
        is_org = (
            any(val.endswith(suf) for suf in _ORG_SUFFIXES)      # 기관 접미사
            or val.startswith(_ORG_PREFIXES)                     # 국립/한국/정부…
            or len(val_ns) >= 5                                  # 사람이름은 ≤4자
            or (" " in val and all(len(p) >= 2 for p in val.split()))  # 복합 기관명
        )
        # 사람이름 오탐 방지: 성명/대표자 cue 문맥의 값은 제외.
        if is_org and not re.search(r"(성\s*명|대표자|서명)\s*[:：]\s*" + re.escape(val), raw_text):
            seen.add(val)
            out.append(val)
    return out


def _extract_credit_roles(raw_text: str) -> List[str]:
    """'무대장치-장종선 / 의상디자인·제작-홍장희' 형식에서 역할(position) 추출."""
    out: List[str] = []
    seen: set = set()
    for m in _CREDIT_ROLE_RE.finditer(raw_text):
        role = m.group(1).strip(" ·-")
        if not role or role in seen:
            continue
        if role in _CREDIT_ROLE_LEXICON or role.endswith(_CREDIT_ROLE_KEYWORDS):
            seen.add(role)
            out.append(role)
    return out


# 직위 lexicon 뒤에 이 글자가 오면 cue/더 긴 단어의 일부 → 오추출 (대표자, 위원회/위원장).
_POS_TRAILING_BLOCK = ("자", "회", "장", "명")


def _extract_position_lexicon(raw_text: str) -> List[str]:
    """직위 lexicon 매칭 — 단, 뒤 글자가 cue/더 긴 단어를 이루면 제외 (대표자·위원회)."""
    found: List[str] = []
    for kw in POSITION_LEXICON:
        start = 0
        while True:
            i = raw_text.find(kw, start)
            if i < 0:
                break
            start = i + 1
            nxt = raw_text[i + len(kw): i + len(kw) + 1]
            if nxt and nxt in _POS_TRAILING_BLOCK:
                continue
            if kw not in found:
                found.append(kw)
            break
    # 부분문자열 제거: '연구원'이 '연구위원'/'선임연구원'의 부분이면 버림.
    out: List[str] = []
    for kw in found:
        if any(kw != other and kw in other for other in found):
            continue
        if kw not in out:
            out.append(kw)
    return out


def _find_lexicon(raw_text: str, lexicon) -> List[str]:
    """closed lexicon 매칭 — 다른 매칭의 부분문자열인 항목은 제거(최장 우선)."""
    found: List[str] = []
    for kw in lexicon:
        if kw in raw_text:
            found.append(kw)
    # 부분문자열 제거: '연구원'이 '연구위원'/'선임연구원'의 부분이면 버림.
    out: List[str] = []
    for kw in found:
        if any(kw != other and kw in other for other in found):
            continue
        if kw not in out:
            out.append(kw)
    return out


# 사람이름이 아닌데 NER 이 name 으로 자주 오태깅하는 양식 용어(동의서 문맥).
_NAME_STOPWORDS = frozenset({
    "초상", "성명", "대표", "대표자", "서명", "동의", "동의함", "양도", "양수",
    "양도인", "양수인", "본인", "실연", "실연자", "저작", "저작자", "인적사항",
    "개인정보", "연락처", "주소", "전화", "전화번호", "기관", "기관명", "별지", "붙임",
})


def _filter_korean_names(items: List[str]) -> List[str]:
    """이름 후보 — 한글 2자 이상 + 2-20자. 양식 용어(초상/성명 등)·구두점 조각 제거."""
    out: List[str] = []
    for s in items:
        s = s.strip()
        core = re.sub(r"[^가-힣]", "", s)          # 순수 한글만
        if len(s) < 2 or len(s) > 20:
            continue
        if len(core) < 2:                          # '나.' 처럼 한글 1자+구두점 조각 컷
            continue
        if core in _NAME_STOPWORDS:
            continue
        out.append(s)
    return _dedup_keep_order(out) or ["N/A"]


def _fix_company_paren(items: List[str], raw_text: str) -> List[str]:
    """`주`, `) foo` → `(주)foo` 로 합치기 (raw_text 에서 검증)."""
    cleaned: List[str] = []
    for s in items:
        s = s.strip()
        if not s or s == "N/A":
            continue
        s2 = re.sub(r"^\)\s*", "", s)
        if s2 in ("주", "(주)", "주식회사"):
            continue
        if s2 and f"(주){s2}" in raw_text.replace(" ", ""):
            s2 = f"(주){s2}"
        elif s2 and f"(주) {s2}" in raw_text:
            s2 = f"(주) {s2}"
        # 회사/기관명은 한글 또는 한자(株式會社 등) 포함이면 유효.
        if len(s2) >= 2 and _CJK_RE.search(s2):
            cleaned.append(s2)
    return _dedup_keep_order(cleaned) or ["N/A"]


def _filter_by_length(items: List[str], min_len: int, max_len: int) -> List[str]:
    out = [s.strip() for s in items if s and s != "N/A" and min_len <= len(s.strip()) <= max_len]
    return _dedup_keep_order(out) or ["N/A"]


def _drop_contained(items: List[str]) -> List[str]:
    """다른(더 긴) 값의 부분문자열인 항목 제거 — 전체주소 있으면 바닥 '서울특별시' 중복 제거."""
    vals = [s for s in items if s and s != "N/A"]
    out: List[str] = []
    for s in vals:
        s_ns = s.replace(" ", "")
        if any(s != other and s_ns in other.replace(" ", "") for other in vals):
            continue
        if s not in out:
            out.append(s)
    return out or ["N/A"]


def _filter_period_text(items: List[str]) -> List[str]:
    out: List[str] = []
    for s in items:
        s = s.strip()
        if not s or s == "N/A" or len(s) < 3:
            continue
        # 시간 표현("…만료일까지") 또는 날짜 형식("2080-04-30") 둘 다 유효기간으로 인정.
        if _PERIOD_CUE_RE.search(s) or _DATEISH_RE.search(s):
            out.append(s)
    return _dedup_keep_order(out) or ["N/A"]


def _extend_date_boundary(items: List[str], raw_text: str) -> List[str]:
    out: List[str] = []
    for s in items:
        s = s.strip()
        if not s or s == "N/A":
            continue
        if _DATE_FULL_RE.fullmatch(s):
            out.append(s)
            continue
        prefix = re.escape(s.rstrip("."))
        m = re.search(prefix + r"\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}\s*\.?", raw_text)
        out.append(m.group(0).strip() if m else s)
    return _dedup_keep_order(out) or ["N/A"]


def _filter_title(items: List[str]) -> List[str]:
    """제목 필터 — 일반 제목(제목: …)도 보존. 길이만 검증하고 문서접미사는 강제하지 않음.

    과거엔 '…계약서/동의서'로 끝나야만 통과시켜 일반 저작물 제목을 전부 버렸음.
    NER 제목 예측이 정확(gold relaxed≈0.99)하므로 접미사 강제 대신 길이/노이즈만 컷.
    """
    out: List[str] = []
    for s in items:
        s = s.strip()
        if not s or s == "N/A":
            continue
        if 3 <= len(s) <= 100:
            out.append(s)
    return _dedup_keep_order(out) or ["N/A"]


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------

def postprocess_metadata(meta: MetadataDict, raw_text: str) -> MetadataDict:
    """결정적 후처리.

    1. closed-vocab 라벨 (ri_copyright, ri_contract_type, ri_info, copyright_type,
       copyright_status, copyright_language) → keyword 매칭으로 NER 대체.
    2. form-cue 라벨 (name, address, phone, company, department, position, email,
       ri_period, ri_law_reference, copyright_kotitle) → cue 매칭 결과를 NER 와
       union.
    3. ri_data boundary 복원 + date fallback.
    4. 라벨별 휴리스틱 필터 (길이/한글 검증, period cue, title suffix).
    """
    # (1) closed-vocab — vocab 우선, 없으면 NER 폴백.
    #     (기존엔 vocab 로 무조건 대체 → 닫힌 목록에 없는 정답을 전부 버렸음.
    #      동의서엔 vocab 가 명확히 잡히고, 넓은 gold 엔 vocab 가 비어 NER 폴백으로 회수.)
    for label in CLOSED_VOCAB:
        vocab = _apply_closed_vocab(raw_text, label)
        if label == "copyright_status":
            vocab = _dedup_keep_order(vocab + _extract_file_ext(raw_text))
        ner_hits = [s for s in meta.get(label, []) if s and s != "N/A"]
        merged = vocab if vocab else ner_hits
        meta[label] = merged if merged else ["N/A"]

    # (2) form-cue + NER union
    for label in FORM_CUE_PATTERNS:
        cue_hits = _apply_form_cues(raw_text, label)
        ner_hits = [s for s in meta.get(label, []) if s and s != "N/A"]
        merged = _dedup_keep_order(ner_hits + cue_hits)
        meta[label] = merged if merged else ["N/A"]

    # (2b) 지자체(행정구역) → address (사용자 확정: 지자체=소재지).
    addr_region = _extract_region_address(raw_text)
    if addr_region:
        base = [s for s in meta.get("address", []) if s and s != "N/A"]
        meta["address"] = _dedup_keep_order(base + addr_region)

    # (2c) 직위 lexicon + 크레딧 역할 → position.
    pos_extra = _extract_position_lexicon(raw_text) + _extract_credit_roles(raw_text)
    if pos_extra:
        base = [s for s in meta.get("position", []) if s and s != "N/A"]
        meta["position"] = _dedup_keep_order(base + pos_extra)

    # (2d) 기관 접미사(청/관/원/회/재단…) → company (NER 이 name 으로 오태깅한 기관명 회수).
    org_company = _extract_org_company(raw_text)
    if org_company:
        base = [s for s in meta.get("company", []) if s and s != "N/A"]
        meta["company"] = _dedup_keep_order(base + org_company)

    # (2e) 다인 표(연번 성명 주소 전화번호) → 행별 name/address/phone.
    #      cue 없는 표는 NER/regex 가 이름·전화 경계를 못 잡음 → 컬럼 파싱으로 보정.
    table = _extract_table_rows(raw_text)
    if table:
        for lab in ("name", "address"):
            base = [s for s in meta.get(lab, []) if s and s != "N/A"]
            meta[lab] = _dedup_keep_order(base + table[lab]) or ["N/A"]
        # 표 전화는 컬럼 파싱이 정확(앞칸 번지 노이즈 없음) → regex 결과를 대체.
        meta["phone"] = table["phone"] or meta.get("phone", ["N/A"])

    # (3) ri_data boundary 복원 + date fallback
    if "ri_data" in meta and meta["ri_data"] != ["N/A"]:
        meta["ri_data"] = _extend_date_boundary(meta["ri_data"], raw_text)
    if meta.get("ri_data") == ["N/A"] and meta.get("date") and meta["date"] != ["N/A"]:
        meta["ri_data"] = list(meta["date"])

    # (4) 라벨별 휴리스틱
    if "name" in meta and meta["name"] != ["N/A"]:
        meta["name"] = _filter_korean_names(meta["name"])
    if "company" in meta and meta["company"] != ["N/A"]:
        meta["company"] = _fix_company_paren(meta["company"], raw_text)
    if "address" in meta and meta["address"] != ["N/A"]:
        meta["address"] = _drop_contained(_filter_by_length(meta["address"], 2, 200))
    if "ri_period" in meta and meta["ri_period"] != ["N/A"]:
        meta["ri_period"] = _filter_period_text(meta["ri_period"])
    if "copyright_kotitle" in meta and meta["copyright_kotitle"] != ["N/A"]:
        meta["copyright_kotitle"] = _filter_title(meta["copyright_kotitle"])

    return meta
