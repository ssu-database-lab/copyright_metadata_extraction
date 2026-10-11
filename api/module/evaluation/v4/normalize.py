"""v4 평가 — 값 정규화 (docs/v4/M1_evaluator_spec.md §4 일치 규칙, §1 키별 규칙).

규약
  * 입력이 None / 공백만 있는 문자열 / 빈 리스트이면 None 을 돌려준다 (= 출력 없음).
  * 해석할 수 없는 값이면 ValueError 를 던진다 (= 평가기에서 오답 + 경고).
  * 예측값과 gold 값을 **같은 함수**로 정규화한 뒤 비교한다. 따라서 표기 변형
    (JPG/JPEG, 2024.1.5/2024-01-05, O/허용 …)은 양쪽 어디에 있어도 같게 취급된다.
"""
from __future__ import annotations

import datetime as _dt
import re
import unicodedata
from typing import Any, Iterable, Optional

from .keys23 import WORK_TYPES

# ---------------------------------------------------------------------------
# 공통: 텍스트
# ---------------------------------------------------------------------------
# 보이지 않는 문자 — 제로폭(U+200B~D, U+2060, U+FEFF)·양방향 제어(U+200E/F, U+202A~E, U+2066~9).
# 사이트 값에 섞여 들어와 비교를 조용히 실패시키므로 먼저 제거한다.
_INVISIBLE = re.compile("[​‌‍⁠﻿‎‏‪-‮⁦-⁩]")
_WS = re.compile(r"\s+")
_ID_PREFIX = re.compile(r"^\d{5,8}\s+")


def is_blank(v: Any) -> bool:
    """None, 공백만 있는 문자열, 빈 리스트/튜플/집합/딕셔너리 → '출력 없음'."""
    if v is None:
        return True
    if isinstance(v, str):
        return _INVISIBLE.sub("", v).strip() == ""
    if isinstance(v, (list, tuple, set, frozenset, dict)):
        return len(v) == 0
    return False


def scalar_input(v: Any) -> Any:
    """단일값 키에 리스트가 들어오면 원소 1개는 그 원소로, 여럿이면 ', ' 로 이어 붙인다."""
    if isinstance(v, (list, tuple)):
        items = [x for x in v if not is_blank(x)]
        if not items:
            return None
        return items[0] if len(items) == 1 else ", ".join(str(x) for x in items)
    return v


def norm_text(s: Any) -> Optional[str]:
    """NFC → 보이지 않는 문자 제거 → 공백 압축·양끝 제거. 대소문자는 유지한다(표시용)."""
    if is_blank(s):
        return None
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFC", s)
    s = _INVISIBLE.sub("", s)
    s = _WS.sub(" ", s).strip()
    return s or None


def text_key(s: Any) -> Optional[str]:
    """비교용 — norm_text 뒤 라틴 대소문자 무시(casefold)."""
    t = norm_text(s)
    return t.casefold() if t is not None else None


def strip_id_prefix(s: Any) -> Optional[str]:
    """제호 앞의 관리번호 토큰(`^\\d{5,8}\\s+`)을 뗀다: '13387752 운수 자판기' → '운수 자판기'."""
    t = norm_text(s)
    if t is None:
        return None
    stripped = _ID_PREFIX.sub("", t).strip()
    return stripped or t


# ---------------------------------------------------------------------------
# 이름 (저작자·저작권자) — §4 Names
# ---------------------------------------------------------------------------
_NAME_ANNOT = re.compile(r"\(\s*(?:저작물)?\s*\d+\s*건\s*\)")
_NAME_SPLIT = re.compile(r"[\n,;/·ㆍ•]")
_ALL_WS = re.compile(r"\s+")


def norm_name(s: Any) -> Optional[str]:
    """NFC, 공백 전부 제거, '(저작물 N 건)'/'(N건)' 주석 제거, casefold."""
    t = norm_text(s)
    if t is None:
        return None
    t = _NAME_ANNOT.sub("", t)
    t = _ALL_WS.sub("", t)
    return t.casefold() or None


def _iter_items(v: Any) -> Iterable[Any]:
    if isinstance(v, (list, tuple, set, frozenset)):
        return list(v)
    return [v]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def split_names(v: Any) -> list[str]:
    """문자열 또는 리스트 → 정규화된 이름 목록 (개행 , ; / · 로 분리, 중복 제거, 입력 순서 유지)."""
    if is_blank(v):
        return []
    names: list[str] = []
    for item in _iter_items(v):
        if is_blank(item):
            continue
        for piece in _NAME_SPLIT.split(str(item)):
            n = norm_name(piece)
            if n:
                names.append(n)
    return _dedupe(names)


# ---------------------------------------------------------------------------
# 태그 (키워드·해시태그) — §4 Lists
# ---------------------------------------------------------------------------
_TAG_SEP = re.compile(r"[,|;\n、]+")


def norm_tag(s: Any) -> Optional[str]:
    """'#' 제거, 공백·'_'·'-'·구두점 제거(영숫자·한글 등 isalnum 문자만 남김), casefold."""
    t = norm_text(s)
    if t is None:
        return None
    kept = "".join(ch for ch in t if ch.isalnum())
    return kept.casefold() or None


def _split_tag_string(s: str) -> list[str]:
    # '#바다 #해변' 처럼 '#' 로 이어진 문자열은 '#' 에서 끊고, 그 다음 , | ; 개행 으로 끊는다.
    # 공백으로는 끊지 않는다 — '경복궁 단청' 같은 두 단어 태그는 하나다.
    pieces: list[str] = []
    for chunk in s.split("#"):
        pieces.extend(p for p in _TAG_SEP.split(chunk) if p.strip())
    return pieces


def split_tags(v: Any) -> list[str]:
    """문자열 또는 리스트 → 정규화된 태그 목록 (중복 제거, 입력 순서 유지)."""
    if is_blank(v):
        return []
    tags: list[str] = []
    for item in _iter_items(v):
        if is_blank(item):
            continue
        for piece in _split_tag_string(str(item)):
            t = norm_tag(piece)
            if t:
                tags.append(t)
    return _dedupe(tags)


# ---------------------------------------------------------------------------
# 날짜·기간 — §4 Dates, §1 #14~16
# ---------------------------------------------------------------------------
_DATE_SEP = re.compile(r"^(\d{4})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})\s*\.?$")
_DATE_KO = re.compile(r"^(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일?$")
_DATE_COMPACT = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
_DATE_WITH_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ]\d{1,2}:\d{2}")
# 기간 문자열 안에서 날짜를 찾아내는 패턴 — 구분자(~ - – 부터/까지)에 상관없이 날짜 두 개를 뽑는다.
_DATE_ANY = re.compile(
    r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일?"
    r"|\d{4}\s*[.\-/]\s*\d{1,2}\s*[.\-/]\s*\d{1,2}"
    r"|\d{8}"
)


def _ymd(y: str, m: str, d: str, raw: Any) -> str:
    try:
        return _dt.date(int(y), int(m), int(d)).isoformat()
    except ValueError as e:  # 13월, 32일 등
        raise ValueError(f"날짜 해석 불가: {raw!r} ({e})") from None


def norm_date(v: Any) -> Optional[str]:
    """YYYY-MM-DD / YYYY.MM.DD / YYYY/MM/DD / YYYY년 M월 D일 / YYYYMMDD → 'YYYY-MM-DD'."""
    if is_blank(v):
        return None
    if isinstance(v, _dt.datetime):
        return v.date().isoformat()
    if isinstance(v, _dt.date):
        return v.isoformat()
    t = norm_text(scalar_input(v))
    if t is None:
        return None
    m_time = _DATE_WITH_TIME.match(t)
    if m_time:
        t = m_time.group(1)
    for pat in (_DATE_SEP, _DATE_KO, _DATE_COMPACT):
        m = pat.match(t)
        if m:
            return _ymd(m.group(1), m.group(2), m.group(3), v)
    raise ValueError(f"날짜 해석 불가: {v!r}")


def _period_parts(v: Any) -> tuple[Any, Any]:
    if isinstance(v, dict):
        start = v.get("start") or v.get("license_start_date") or v.get("시작일")
        end = v.get("end") or v.get("license_end_date") or v.get("종료일")
        if is_blank(start) or is_blank(end):
            raise ValueError(f"기간 해석 불가: {v!r}")
        return start, end
    if isinstance(v, (list, tuple)):
        items = [x for x in v if not is_blank(x)]
        if len(items) != 2:
            raise ValueError(f"기간 해석 불가: {v!r}")
        return items[0], items[1]
    t = norm_text(v)
    if t is None:
        raise ValueError(f"기간 해석 불가: {v!r}")
    found = _DATE_ANY.findall(t)
    if len(found) != 2:
        raise ValueError(f"기간 해석 불가(날짜 {len(found)}개): {v!r}")
    return found[0], found[1]


def norm_period(v: Any) -> Optional[str]:
    """'YYYY-MM-DD~YYYY-MM-DD' 로 정규화. ~ - – 부터/까지, 리스트[2], dict{start,end} 를 받는다."""
    if is_blank(v):
        return None
    start, end = _period_parts(v)
    s, e = norm_date(start), norm_date(end)
    if s is None or e is None:
        raise ValueError(f"기간 해석 불가: {v!r}")
    return f"{s}~{e}"


# ---------------------------------------------------------------------------
# 권리 — §1 Rights values
# ---------------------------------------------------------------------------
# 허용|불허|true|false|O|X|Y|N 의 8가지 (대소문자 무시). 문자열 'null'/'none' 은 null 로 본다.
# 체크 글리프(v/V/■/□)는 예측값으로 받지 않는다 — 원장(gold) 표기일 뿐이며 추출기는 허용/불허 로 내야 한다 (§1 Rights values).
_RIGHTS = {
    "허용": "허용", "불허": "불허",
    "true": "허용", "false": "불허",
    "o": "허용", "x": "불허",
    "y": "허용", "n": "불허",
}
_RIGHTS_NULL = {"null", "none"}
_TRAILING_PAREN = re.compile(r"\s*\(.*\)$", re.S)  # '허용(방송권, 디지털음성송신권)' 의 뒤 괄호


def norm_rights(v: Any) -> Optional[str]:
    """권리 값 → '허용' | '불허' | None."""
    if is_blank(v):
        return None
    if isinstance(v, bool):
        return "허용" if v else "불허"
    t = text_key(scalar_input(v))
    if t is None or t in _RIGHTS_NULL:
        return None
    # 공중송신권의 하위 권리(방송권·전송권·디지털음성송신권)는 별도 키가 아니므로(§1 #8) 뒤에 붙은 괄호는 떼고 본다
    t = _TRAILING_PAREN.sub("", t).strip() or t
    if t in _RIGHTS:
        return _RIGHTS[t]
    raise ValueError(f"권리 값 해석 불가: {v!r}")


# ---------------------------------------------------------------------------
# 저작물유형 — §1 #3 (닫힌 집합)
# ---------------------------------------------------------------------------
_WORK_TYPE_ALIASES = {
    "프로그램": "컴퓨터프로그램",
    "컴퓨터프로그램": "컴퓨터프로그램",
    "소프트웨어": "컴퓨터프로그램",
    "동영상": "영상",
    "영상물": "영상",
}
_PAREN = re.compile(r"\([^)]*\)")


def norm_work_type(v: Any) -> Optional[str]:
    """공백·괄호 제거, 끝의 '저작물' 제거, 변형 매핑 → WORK_TYPES 중 하나."""
    if is_blank(v):
        return None
    t = norm_text(scalar_input(v))
    if t is None:
        return None
    t = _PAREN.sub("", t)
    t = _ALL_WS.sub("", t)
    t = re.sub(r"저작물$", "", t)
    if t in WORK_TYPES:
        return t
    if t in _WORK_TYPE_ALIASES:
        return _WORK_TYPE_ALIASES[t]
    raise ValueError(f"저작물유형 해석 불가: {v!r}")


# ---------------------------------------------------------------------------
# 파일 기술정보 — §1 #19~21, #23
# ---------------------------------------------------------------------------
_FORMAT_ALIASES = {"JPEG": "JPG", "TIFF": "TIF", "MPEG4": "MP4", "MPEG-4": "MP4"}


def norm_format(v: Any) -> Optional[str]:
    """확장자 대문자(점 없이). 'image/jpeg', '.jpg', 'a.JPG' 도 받는다. JPEG→JPG, TIFF→TIF, MPEG4→MP4."""
    if is_blank(v):
        return None
    t = norm_text(scalar_input(v))
    if t is None:
        return None
    if "/" in t:  # MIME 타입 — 'application/x-hwp' 의 비표준 접두 'x-' 도 뗀다
        t = t.rsplit("/", 1)[1]
        if t.lower().startswith("x-"):
            t = t[2:]
    if "." in t:  # 파일명 또는 '.jpg'
        t = t.rsplit(".", 1)[1]
    t = re.sub(r"[^0-9A-Za-z\-]", "", t).upper()
    if not t:
        raise ValueError(f"파일포맷 해석 불가: {v!r}")
    return _FORMAT_ALIASES.get(t, t.replace("-", ""))


_BYTES = re.compile(r"^(\d[\d,]*)(?:\.0+)?\s*(?:bytes?|b|바이트)?$", re.IGNORECASE)


def norm_int_bytes(v: Any) -> Optional[int]:
    """정수 바이트. '1,234', '1234 bytes', '1234B', '1234.0' 허용; KB/MB 단위는 불허(ValueError)."""
    if is_blank(v):
        return None
    if isinstance(v, bool):
        raise ValueError(f"파일크기 해석 불가: {v!r}")
    if isinstance(v, int):
        if v < 0:
            raise ValueError(f"파일크기 해석 불가: {v!r}")
        return v
    if isinstance(v, float):
        if v.is_integer() and v >= 0:
            return int(v)
        raise ValueError(f"파일크기 해석 불가: {v!r}")
    t = norm_text(scalar_input(v))
    if t is None:
        return None
    m = _BYTES.match(t)
    if not m:
        raise ValueError(f"파일크기 해석 불가: {v!r}")
    return int(m.group(1).replace(",", ""))


_RESOLUTION = re.compile(r"^(\d+)\s*[xX×*]\s*(\d+)\s*(?:px|픽셀|pixels?)?$", re.IGNORECASE)


def _res_pair(w: Any, h: Any, raw: Any) -> str:
    try:
        wi, hi = int(str(w).strip()), int(str(h).strip())
    except ValueError:
        raise ValueError(f"해상도 해석 불가: {raw!r}") from None
    if wi <= 0 or hi <= 0:
        raise ValueError(f"해상도 해석 불가: {raw!r}")
    return f"{wi}x{hi}"


def _looks_like_number(x: Any) -> bool:
    """정수/실수(bool 제외) 또는 숫자만 있는 문자열('1920')."""
    if isinstance(x, bool):
        return False
    return isinstance(x, (int, float)) or (isinstance(x, str) and x.strip().isdigit())


def norm_resolution(v: Any) -> Optional[str]:
    """'1920 x 1080', '1920*1080', '1920×1080', [1920, 1080], ['1920', '1080'], {'width','height'} → '1920x1080'."""
    if is_blank(v):
        return None
    if isinstance(v, dict):
        return _res_pair(v.get("width"), v.get("height"), v)
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(_looks_like_number(x) for x in v):
        return _res_pair(v[0], v[1], v)  # [1920, 1080] 과 ['1920', '1080'] 모두 (문자열 하나짜리 리스트는 아래 일반 경로)
    t = norm_text(scalar_input(v))
    if t is None:
        return None
    m = _RESOLUTION.match(t)
    if not m:
        raise ValueError(f"해상도 해석 불가: {v!r}")
    return _res_pair(m.group(1), m.group(2), v)


_HASH = re.compile(r"^(?:sha-?256\s*[:=]?\s*)?([0-9a-fA-F]{64})$", re.IGNORECASE)


def norm_hash(v: Any) -> Optional[str]:
    """SHA-256 16진수 64자, 소문자. 'sha256:' 접두 허용."""
    if is_blank(v):
        return None
    t = norm_text(scalar_input(v))
    if t is None:
        return None
    m = _HASH.match(t)
    if not m:
        raise ValueError(f"해시 해석 불가: {v!r}")
    return m.group(1).lower()


__all__ = [
    "is_blank", "scalar_input", "norm_text", "text_key", "strip_id_prefix",
    "norm_name", "split_names", "norm_tag", "split_tags",
    "norm_date", "norm_period", "norm_rights", "norm_work_type",
    "norm_format", "norm_int_bytes", "norm_resolution", "norm_hash",
]
