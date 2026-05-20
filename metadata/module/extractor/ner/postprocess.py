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
_PERIOD_CUE_RE = re.compile(
    r"(?:년|월|일|개월|시부터|만료|까지|보호기간|기한|동안|일\s*간|개월\s*간)"
)
_DATE_FULL_RE = re.compile(
    r"\d{4}\s?\.\s?\d{1,2}\s?\.\s?\d{1,2}\s?\.?|\d{4}년\s?\d{1,2}월\s?\d{1,2}일|\d{4}-\d{1,2}-\d{1,2}"
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
            r"(?:^|\n)[ \t]*[○●·•∙*◦▫▪]?[ \t]*기[ \t]*관[ \t]*명?[ \t]*[:：][ \t]+([^\n]+?)[ \t]*$",
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


def _filter_korean_names(items: List[str]) -> List[str]:
    """이름 후보 — 한글 1자 이상 포함 + 2-20자. 체크박스/punctuation 제거."""
    out: List[str] = []
    for s in items:
        s = s.strip()
        if len(s) < 2 or len(s) > 20:
            continue
        if not _HANGUL_RE.search(s):
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
        if len(s2) >= 2 and _HANGUL_RE.search(s2):
            cleaned.append(s2)
    return _dedup_keep_order(cleaned) or ["N/A"]


def _filter_by_length(items: List[str], min_len: int, max_len: int) -> List[str]:
    out = [s.strip() for s in items if s and s != "N/A" and min_len <= len(s.strip()) <= max_len]
    return _dedup_keep_order(out) or ["N/A"]


def _filter_period_text(items: List[str]) -> List[str]:
    out: List[str] = []
    for s in items:
        s = s.strip()
        if not s or s == "N/A" or len(s) < 3:
            continue
        if _PERIOD_CUE_RE.search(s):
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
    out: List[str] = []
    for s in items:
        s = s.strip()
        if not s or s == "N/A":
            continue
        if 5 <= len(s) <= 80 and any(s.endswith(suf) for suf in
                                       ("계약서", "동의서", "확약서", "증서", "신청서",
                                        "보고서", "협약서", "각서", "양도서")):
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
    # (1) closed-vocab
    for label in CLOSED_VOCAB:
        kws = _apply_closed_vocab(raw_text, label)
        meta[label] = kws if kws else ["N/A"]

    # (2) form-cue + NER union
    for label in FORM_CUE_PATTERNS:
        cue_hits = _apply_form_cues(raw_text, label)
        ner_hits = [s for s in meta.get(label, []) if s and s != "N/A"]
        merged = _dedup_keep_order(ner_hits + cue_hits)
        meta[label] = merged if merged else ["N/A"]

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
        meta["address"] = _filter_by_length(meta["address"], 2, 200)
    if "ri_period" in meta and meta["ri_period"] != ["N/A"]:
        meta["ri_period"] = _filter_period_text(meta["ri_period"])
    if "copyright_kotitle" in meta and meta["copyright_kotitle"] != ["N/A"]:
        meta["copyright_kotitle"] = _filter_title(meta["copyright_kotitle"])

    return meta
