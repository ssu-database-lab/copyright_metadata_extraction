"""계약서 본문에서 당사자('이용자'·'저작자')를 규칙으로 찾는다 (M2 spec A5 — v4 키 5 저작권자).

새 서식(2026-09-22, 장세영)에는 '저작권자' 라는 낱말이 어디에도 인쇄되지 않는다. 원장의 권리자
열(한국저작권위원회)이 계약서에는 **이용자** 당사자로 인쇄되므로, v4 는 키 5 를 '이용자 당사자' 로
정의했다 (docs/v4/M1_evaluator_spec.md §1 #5). 모델(LLM/NER)이 뽑은 copyright_holder 는 v4 출력에
쓰지 않고 비교용(test_prediction._meta.model_copyright_holder)으로만 남긴다. 모델 호출 없음.

이용자는 두 곳에서 읽는다 — 둘 다 01_new_contracts_T001_keys.md §(a) 실측:
  ① 전문    `… 저작재산권 이용자 한국저작권위원회 (이하 “이용자”라 함)는 …`
             본문층은 `저작재산권 이\\n용자`, `(이하 “이용\\n자”라 함)` 처럼 낱말 안에서 줄이 넘어가고
             (OCR 단계의 _pdf_page_text 는 이를 되붙이지만 raw fitz 텍스트는 아니다), 따옴표는 U+201C/D 다.
  ② 서명란  `이용자 : 한국저작권위원회` (마지막 쪽). 이름이 여럿이면 다음 줄에 하나씩 이어진다
             (권리자 서명란에서 실측: `권리자 : 양지웅 (저작물 3 건)⏎이영지 (저작물 2 건)⏎신혜진`).
서명란이 한 줄에 깔끔하게 인쇄되므로 우선하고, 없으면 전문을 쓴다. 둘 다 있으면 일치 여부도 돌려준다.

저작자(키 4 비교용)는 제2조 `저작자 : …` 줄과 이어지는 이름 줄(다음 라벨 `종별 :` 전까지)에서 읽는다.
`(저작물 N 건)` 주석은 평가기(norm_name)가 떼므로 여기서는 인쇄된 그대로 둔다 (clean_names 로 뗄 수 있다).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_OPEN_Q = "“\"'‘「『"
_CLOSE_Q = "”\"'’」』"

# ① 전문: '저작재산권 이용자 <이름> (이하 "이용자"' — 글자 사이 \s* 는 낱말 안 줄바꿈 때문이다.
_PREAMBLE = re.compile(
    r"저\s*작\s*재\s*산\s*권\s*이\s*용\s*자\s*"
    r"(?P<name>.{1,80}?)"
    r"\s*\(\s*이\s*하\s*[" + _OPEN_Q + r"]?\s*이\s*용\s*자\s*[" + _CLOSE_Q + r"]?",
    re.S,
)
# ② 서명란: 줄 머리의 '이용자 : <이름>' (전문의 '이용자' 는 줄 머리에 오지 않는다)
_SIGNATURE = re.compile(r"^[ \t]*이\s*용\s*자\s*[:：][ \t]*(?P<name>[^\n]*)$", re.M)
# 제2조 '저작자 : <이름>'
_AUTHOR = re.compile(r"^[ \t]*저\s*작\s*자\s*[:：][ \t]*(?P<name>[^\n]*)$", re.M)

# 이어지는 이름 줄을 멈추는 라벨들. 서명란의 '성 명(인)' · '주 소' 는 자간 때문에 `성` / `명(인) : …`,
# `주` / `소 : …` 로 갈라져 나온다. 콜론이 있는 줄, 날짜 줄, 조문 줄도 멈춤 신호다.
_STOP_LINE = re.compile(
    r"^\s*(?:성\s*명|성|명\s*\(\s*인\s*\)|생\s*년\s*월\s*일|주\s*소|주|소|권\s*리\s*자|이\s*용\s*자|"
    r"저\s*작\s*자|종\s*별|권\s*리|대\s*리\s*인|사업자\s*등록\s*번호|주민\s*등록\s*번호|연락처|전화|"
    r"날인|서명|제\s*\d+\s*조|\(\s*\d+\s*\)|\d{4}\s*년)"
    r"\s*(?:[:：(]|$)"
)
_HAS_COLON = re.compile(r"[:：]")
_NAME_ANNOT = re.compile(r"\(\s*(?:저작물)?\s*\d+\s*건\s*\)")
_MAX_CONTINUATION = 6


def _clean(name: Optional[str]) -> Optional[str]:
    """줄바꿈 제거(낱말 안에서 넘어간 것) → 공백 접기 → 양끝 구두점·따옴표 제거. 비면 None."""
    if not name:
        return None
    s = name.replace("\n", "")
    s = " ".join(s.split())
    s = s.strip(" ,.;、" + _OPEN_Q + _CLOSE_Q)
    if not s or "이하" in s:
        return None
    return s


def _continuation(lines: List[str], start: int) -> List[str]:
    """start 줄 다음부터 라벨/빈 줄/콜론 줄이 나오기 전까지의 이름 줄들."""
    out: List[str] = []
    for line in lines[start + 1: start + 1 + _MAX_CONTINUATION]:
        if not line.strip() or _STOP_LINE.match(line) or _HAS_COLON.search(line):
            break
        out.append(line)
    return out


def _labelled_names(text: str, pattern: re.Pattern, last: bool) -> List[str]:
    """'라벨 : 이름' 줄(+이어지는 줄)에서 이름 목록. last=True 면 마지막 등장(서명란)을 쓴다."""
    lines = text.splitlines()
    hits = [(i, m) for i, line in enumerate(lines) for m in [pattern.match(line)] if m]
    if not hits:
        return []
    i, m = hits[-1] if last else hits[0]
    raw = [m.group("name")] + _continuation(lines, i)
    names = [n for n in (_clean(x) for x in raw) if n]
    # 중복 제거(순서 유지)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def find_licensee(text: str) -> Dict[str, Any]:
    """이용자 당사자. 반환:
        {"name": "한국저작권위원회" | None,   # 여럿이면 ', ' 로 이어 붙인 문자열
         "names": [...],                       # 이름 목록
         "source": "signature" | "preamble" | None,
         "signature": [...] | None,            # 서명란에서 읽은 이름들
         "preamble": str | None,               # 전문에서 읽은 이름
         "agree": bool | None}                 # 두 출처가 모두 있을 때 공백 제거 후 일치 여부
    """
    text = text or ""
    signature = _labelled_names(text, _SIGNATURE, last=True) or None
    m = _PREAMBLE.search(text)
    preamble = _clean(m.group("name")) if m else None

    if signature:
        names, source = signature, "signature"
    elif preamble:
        names, source = [preamble], "preamble"
    else:
        names, source = [], None

    agree = None
    if signature and preamble:
        agree = "".join(signature).replace(" ", "") == preamble.replace(" ", "")

    return {
        "name": ", ".join(names) if names else None,
        "names": names,
        "source": source,
        "signature": signature,
        "preamble": preamble,
        "agree": agree,
    }


def extract_licensee(text: str) -> Optional[str]:
    """이용자 당사자 이름만 (없으면 None)."""
    return find_licensee(text)["name"]


# ---------------------------------------------------------------- 권리자(이용허락자) 당사자 — v5 계약서 (2026-11-03)
# v5 시험 계약서(장세영 템플릿 2. 계약서.hwpx = A, 숭실대 B·C)는 원장의 권리자를 **이용허락자** 자리에 인쇄한다.
#   A  전문 `저작자 또는 저작재산권 이용허락자 <이름>(이하 “권리자”라 함)` · 서명란 `권리자 : <이름>`
#   B  전문 `저작재산권자 <이름>(이하 “권리자”라 한다)` · 서명란 `권리자` 다음 줄 `성명(상호) : <이름>`
#   C  표 `저작재산권자` 칸 다음 줄 `<이름>` · 전문 `<이름>(이하 “갑”이라 한다)`
# 키 5(저작권자)는 이 당사자다. 9월 서식(권리자 열이 이용자 자리에 인쇄됨)용 find_licensee 는 비교용으로 남긴다.
_LICENSOR_SIGNATURE = re.compile(r"^[ \t]*권\s*리\s*자\s*[:：][ \t]*(?P<name>[^\n]*)$", re.M)
_LICENSOR_BLOCK = re.compile(r"^[ \t]*권\s*리\s*자[ \t]*\n[ \t]*성\s*명\s*\(\s*상\s*호\s*\)\s*[:：][ \t]*(?P<name>[^\n]*?)"
                             r"(?:\s{2,}\(서명 또는 인\))?[ \t]*$", re.M)
_LICENSOR_TABLE = re.compile(r"^[ \t]*저\s*작\s*재\s*산\s*권\s*자[ \t]*\n(?P<name>[^\n]+)$", re.M)
_LICENSOR_PREAMBLE_LABELLED = re.compile(      # A·B: 라벨 뒤 이름 (이름이 줄을 넘을 수 있다)
    r"저\s*작\s*재\s*산\s*권\s*(?:이\s*용\s*허\s*락\s*자|자)\s+"
    r"(?P<name>[^\n(]{1,80}?(?:\n[^\n(]{1,40}?)?)"
    r"\s*\(\s*이\s*하\s*[" + _OPEN_Q + r"]?\s*권\s*리\s*자\s*[" + _CLOSE_Q + r"]?")
_LICENSOR_PREAMBLE_GAP = re.compile(           # C: 문단 첫머리 `<이름>(이하 “갑”` — 한 줄 안에서만
    r"^(?P<name>[^\n(]{1,80}?)\s*\(\s*이\s*하\s*[" + _OPEN_Q + r"]?\s*갑\s*[" + _CLOSE_Q + r"]?", re.M)


def _clean_party(name: Optional[str]) -> Optional[str]:
    """권리자 이름 정리 — 낱말 안 줄바꿈 제거, 공백 접기, 양끝 쉼표·일반 따옴표만 뗀다 (『』「」 는 이름의 일부일 수 있다)."""
    if not name:
        return None
    s = " ".join(name.replace("\n", "").split()).strip(" ,.;\"'“”‘’")
    return s if s and "이하" not in s else None


def find_licensor(text: str) -> Dict[str, Any]:
    """권리자(이용허락자) 당사자 — 서명란 → 표 → 전문 순. 반환은 find_licensee 와 같은 모양."""
    text = text or ""
    found: List[str] = []
    source = None
    lines = text.splitlines()
    sig = [(i, m) for i, line in enumerate(lines) for m in [_LICENSOR_SIGNATURE.match(line)] if m]
    if sig:                                    # A 서명란 `권리자 : 이름` (+ 다음 줄로 이어지는 이름들, 9월 서식)
        i, m = sig[-1]
        for raw in [m.group("name")] + _continuation(lines, i):
            n = _clean_party(raw)
            if n and n not in found:
                found.append(n)
        source = "signature" if found else None
    for name, pat, last in (("signature_block", _LICENSOR_BLOCK, True), ("table", _LICENSOR_TABLE, False)):
        if found:
            break
        hits = list(pat.finditer(text))
        if hits:
            n = _clean_party((hits[-1] if last else hits[0]).group("name"))
            if n:
                found, source = [n], name
    m = _LICENSOR_PREAMBLE_LABELLED.search(text) or _LICENSOR_PREAMBLE_GAP.search(text)
    preamble = _clean_party(m.group("name")) if m else None
    if not found and preamble:
        found, source = [preamble], "preamble"
    agree = None
    if found and preamble and source != "preamble":
        strip = lambda x: _NAME_ANNOT.sub("", x).replace(" ", "")
        agree = "".join(map(strip, found)) == strip(preamble)
    return {"name": ", ".join(found) if found else None, "names": found, "source": source,
            "preamble": preamble, "agree": agree}


def find_author(text: str) -> List[str]:
    """제2조 '저작자 : …' 의 이름 목록 (인쇄된 그대로 — `(저작물 N 건)` 주석 포함)."""
    return _labelled_names(text or "", _AUTHOR, last=False)


def clean_names(names: List[str]) -> List[str]:
    """`(저작물 N 건)` / `(N건)` 주석을 떼고 공백을 접은 이름 목록 (평가기 norm_name 과 같은 규칙의 표시용 버전)."""
    out: List[str] = []
    for n in names or []:
        s = " ".join(_NAME_ANNOT.sub("", n).split())
        if s and s not in out:
            out.append(s)
    return out


def find_parties(text: str) -> Dict[str, Any]:
    """이용자·저작자를 한 번에."""
    lic = find_licensee(text)
    authors = find_author(text)
    return {"licensee": lic, "author": authors, "author_clean": clean_names(authors)}
