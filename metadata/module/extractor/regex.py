r"""Regex 기반 추출기 — REGEX_LABEL_SET 9 라벨.

``regex_extract(text)[label]`` → ``List[str]`` (중복 제거, 원문 등장 순).
빈 결과는 ``["N/A"]``.
"""
from __future__ import annotations

import re
from typing import Dict, List

from module.parts.labels import REGEX_LABEL_SET


# ──────────────────────────────────────────────────────────────────────
# 라벨별 패턴.
#
# 모든 패턴은 ``re.findall`` 호환 — group 0 전체 매치를 추출값으로 사용한다.
# (group 이 정의된 패턴은 group 0 만 의도된 경우라도 findall 이 tuple 을
# 반환할 수 있어, 모두 non-capturing 으로 작성한다.)
# ──────────────────────────────────────────────────────────────────────


PATTERNS: Dict[str, re.Pattern] = {
    # 휴대전화 + 일반전화 (지역번호 0X / 0XX).
    # OCR 변형 폭넓게 허용: 구분자 ", " / ". " / 다중 공백; suffix 4-5자리.
    # `\b` 단어경계 + `[-.,\s]*` 다중 구분자.
    "phone": re.compile(
        r"\b0?\d{1,2}[-.,\s)]+\d{3,4}[-.,\s]+\d{4,5}\b"
        r"|\b01\d[-.,\s]*\d{3,4}[-.,\s]*\d{4,5}\b"
    ),
    # 이메일 — RFC822 단순화.
    "email": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    # URL — http(s) 또는 도메인-bearing www.
    "copyright_url": re.compile(
        r"https?://\S+"
        r"|www\.\S+\.[a-zA-Z]{2,}"
    ),
    # UCI — 한국저작권보호원 식별번호.  예: 'I410-EC-040-04-04-A0010101'.
    "copyright_uci": re.compile(
        r"\b[A-Za-z]\d{3}-[A-Za-z0-9\-]+\b"
    ),
    # 날짜 — 한국 공공저작물에서 자주 쓰이는 4 종.
    "date": re.compile(
        r"\d{4}\s?\.\s?\d{1,2}\s?\.\s?\d{1,2}\s?\.?"
        r"|\d{4}년\s?\d{1,2}월\s?\d{1,2}일"
        r"|\d{4}-\d{1,2}-\d{1,2}"
        r"|\d{4}/\d{1,2}/\d{1,2}"
    ),
    # 금액 — 숫자 + 통화 단위.  '5,000원', '100만원', '$1000', 'USD 500'.
    "ri_money": re.compile(
        r"[\d,]+\s?(?:원|만\s?원|억\s?원|천\s?원|백\s?만\s?원)"
        r"|(?:USD|\$)\s?[\d,]+(?:\.\d+)?"
    ),
    # 등록번호 — 한국저작권보호원 공식 패턴은 'C-YYYY-XXXXXX' / '제YYYY-XXXXXX호'.
    # phone 의 substring (`8024-9505` 등) 을 false-match 하지 않도록 prefix 필수.
    "copyright_num": re.compile(
        r"\b(?:C|A)-\d{4}-\d{4,}\b"
        r"|제\s?\d{4}\s?-\s?\d{4,}\s?호"
    ),
    # 식별번호 — 등록부/일련번호.  주민번호 형식 (YYMMDD-XXXXXXX) 또는
    # 13-18 자리 숫자열 (phone 10-11자리, OCR 노이즈 100+자리 모두 회피).
    "copyright_idnum": re.compile(
        r"\b\d{6}-\d{7}\b"
        r"|\b\d{13,18}\b"
    ),
    # 수량 — 숫자 + 단위.
    "copyright_quantity": re.compile(
        r"\b\d+\s?(?:개|건|장|점|편|회|페이지|쪽|편[성]?|copies?|items?)\b"
    ),
}

# 정합성 — 코드의 PATTERNS keys 가 labels.py 의 REGEX_LABEL_SET 과 일치해야 한다.
assert set(PATTERNS.keys()) == REGEX_LABEL_SET, (
    f"PATTERNS keys mismatch: {set(PATTERNS.keys())} vs REGEX_LABEL_SET {REGEX_LABEL_SET}"
)


def regex_extract(text: str) -> Dict[str, List[str]]:
    """텍스트에서 regex 9 라벨을 추출.

    Returns:
        ``{label: [str, ...]}``.  각 라벨 리스트는 중복 제거, 원문 등장 순서 유지.
        빈 결과는 ``["N/A"]`` (NER 출력 포맷과 호환).
    """
    out: Dict[str, List[str]] = {}
    for label, rx in PATTERNS.items():
        seen: set = set()
        hits: List[str] = []
        for m in rx.finditer(text):
            s = m.group(0).strip()
            if s and s not in seen:
                seen.add(s)
                hits.append(s)
        out[label] = hits if hits else ["N/A"]
    return out
