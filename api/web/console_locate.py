"""시험 콘솔 — 원본 위치 표시 (계약서 쪽 미리보기 위에 '값이 읽힌 자리' 를 노랗게 칠한다).

docs/v4/M3_console_spec_addendum_highlight.md 구현. **표시 전용** — 추출 결과·점수는 바꾸지 않는다.

쪽마다 위치 출처를 고른다
  · 텍스트 레이어 쪽 (`len(page.get_text().strip()) >= 30` 이고 거의 다 읽을 수 있는 글자): PyMuPDF 글자 상자 → 정확.
  · 스캔·사진 쪽 (그 밖, 그리고 JPG/PNG/TIFF 계약서): Qwen 줄 단위 위치 찾기(qwen3.8-omni-flash → qwen3.6-flash,
    생각 끔). 응답은 줄마다 {"text", "bbox_2d"} (0–1000 상대 좌표) — 값의 상자는 줄 상자 안 글자 구간으로 어림한다 → 근사.
    쪽은 병렬로 한 번씩 부르고, 결과(줄 글자·상자)는 실행 폴더에 남겨 다시 부르지 않는다.
  · 둘 다 없음 (OCR 꺼짐 · DashScope 실패 · DOCX/HWP) → 위치 정보 없음.
Google 서비스는 부르지 않는다 (DashScope 의 Qwen 만).

찾는 대상(targets)
  · `k:<23키>` — 세트 예측 레코드의 23키 값 (T-001 1–16, T-002 는 계약서 문맥 키). 이름 목록은 이름마다, 날짜는 표기 변형,
    권리 6–12 는 체크 칸 항목(□/■/V + 권리 이름[+ 괄호 속 세부 권리]; 그 줄에 권리가 하나뿐이면 줄 전체), 유효기간은
    '시작일…종료일' 한 구간(없으면 두 날짜 따로).
  · `f:<통합 필드>` — 통합 검증(중재자) 결정 중 최종 값이 있는 필드. 23키로 곧장 이어지는 필드는 그 `k:` 대상을 가리킨다(alias).
  · 찾을 수 없는 값(요약·생성 값 / 정규화된 값 / 문서 밖 정보(파일 측정) / 찾지 못함)은 not_found 에 사유와 함께.

좌표: 쪽 기준 0–1 정규화 (x0, y0, x1, y1) — 미리보기 크기와 상관없이 겹친다. PDF 는 회전(page.rotation)을 반영한 화면 좌표,
그림 계약서는 미리보기와 같이 EXIF 회전을 적용하지 않은 원본 방향.
캐시: <run>/locate/<set>.json (표시 결과) · <set>.ocr.json (쪽별 OCR 줄). `?refresh=1` 은 표시를 다시 계산하고(OCR 줄은
재사용), `?refresh=ocr` 은 OCR 도 다시 부른다.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:                                     # 앱은 api/ 를 루트로 올린다(module.*) — 평가 CLI 는 api.module.*
    from module import log_context as LC
    import module.env_loader  # noqa: F401 — .env (DASHSCOPE_API_KEY · DASHSCOPE_BASE_URL) 를 한 번만 읽는다
except ImportError:                      # pragma: no cover
    from api.module import log_context as LC
    import api.module.env_loader  # noqa: F401

logger = logging.getLogger(__name__)

LOCATE_VERSION = 3                       # 계산 방식이 바뀌면 올린다 (캐시 무효화)
OCR_CACHE_VERSION = 1
OCR_DPI = 200                            # 스캔 쪽을 Qwen 에 보낼 때의 해상도 (검증 하네스와 같다) — 좌표는 정규화라 미리보기 DPI 와 무관
OCR_MAX_SIDE = 2400
IMAGE_MAX_SIDE = 2000
OCR_TIMEOUT_SEC = 120.0
OCR_MAX_TOKENS = 8000
OCR_MAX_WORKERS = 6
FAIL_RETRY_SEC = 300.0                   # OCR 이 실패한 결과는 이만큼 지나면 다시 시도한다
TEXT_LAYER_MIN_CHARS = 30
MIN_VALUE_CHARS = 2                      # 이보다 짧은 통합 필드 값은 찾지 않는다
DEFAULT_MODELS = ("qwen3.8-omni-flash", "qwen3.6-flash")
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
KRW_PER_USD = 1400.0
# 공개 단가 (USD / 1M 토큰: 입력, 출력) — 2026-10-11 비교 하네스와 같은 값
PRICE_USD_PER_M: Dict[str, Tuple[float, float]] = {
    "qwen3.8-omni-flash": (0.15, 0.47),
    "qwen3.6-flash": (0.25, 1.50),
    "qwen3.7-plus": (0.40, 1.60),
}
PROMPT_LINES = ("Detect every line of text in this scanned Korean document. Return ONLY a JSON array, one object per "
                'text line, in reading order: {"text": "<exact text>", "bbox_2d": [x1, y1, x2, y2]}. '
                "Copy the text exactly as printed.")

SRC_TEXT = "text_layer"
SRC_OCR = "ocr_qwen"
SRC_NONE = "none"

R_GENERATED = "요약·생성 값"
R_NORMALISED = "정규화된 값"
R_OUTSIDE = "문서 밖 정보(파일 측정)"
R_NOT_FOUND = "찾지 못함"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
NO_PAGE_EXT = {".docx", ".doc", ".hwp", ".hwpx"}

# 체크 칸 글자 (casefold 뒤) — 'v' 는 바로 앞이 라틴 글자가 아닐 때만
CHECK_GLYPHS = set("■□☑☐✓✔✗✘☒▣◼◻▪▫●○◉◆◇◾◽") | {"v"}
CHECKED_GLYPHS = set("■☑✓✔✗✘☒▣◼▪●◉◆◾") | {"v"}

# 23키 → 권리 이름 (계약서 서식의 체크 칸 이름)
RIGHT_LABELS: Dict[str, str] = {
    "reproduction_right": "복제권", "performance_right": "공연권", "public_transmission_right": "공중송신권",
    "exhibition_right": "전시권", "distribution_right": "배포권", "rental_right": "대여권",
    "derivative_work_right": "2차적저작물작성권",
}
_ALL_RIGHT_NAMES = tuple(RIGHT_LABELS.values()) + ("방송권", "전송권", "디지털음성송신권")
WORK_TYPES = ("어문", "음악", "연극", "미술", "건축", "사진", "영상", "도형", "컴퓨터프로그램", "기타")

# 23키 분류
KEY_GENERATED = {"keywords", "hashtags", "description"}
KEY_OUTSIDE = {"file_name", "file_format", "file_size", "resolution", "file_hash"}
KEY_SKIP = {"other_rights"}               # 서식에 칸이 없어 항상 NULL
# 통합 필드 분류
FIELD_GENERATED = {"description", "summary", "keyword", "keywords", "hashtag", "hashtags", "special_terms",
                   "termination_conditions", "contract_terms", "checkbox_info", "remaining_fields", "notes",
                   "remarks", "contract_purpose", "purpose", "usage_scope", "usage_purpose"}
FIELD_DATE_RX = re.compile(r"(?:^|_)(?:date|day)$|_date(?:$|_)|^date_")
FIELD_AMOUNT_RX = re.compile(r"amount|fee|price|payment|royalt|cost")
LEAF_KEYS = {"name", "address", "organization", "company", "agency", "phone", "email", "title"}

ANCHORS = {
    "title": ("제목(제호):", "제호:", "제목:"),
    "creator": ("저작자:",),
    "work_type": ("종별:", "저작물의종류:"),
    "rights": ("권리:",),
    "period": ("이용허락기간은", "이용허락기간:", "계약기간은", "계약기간:", "이용기간:"),
    "preamble_licensee": ("저작재산권이용자",),
}

_LOCK = threading.Lock()
_PATH_LOCKS: Dict[str, threading.Lock] = {}


# ---------------------------------------------------------------------------
# 정규화 · 표기 변형
# ---------------------------------------------------------------------------
_ZW_RX = re.compile(r"[\s​-‍⁠﻿­]+")
_PUNCT_MAP = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‟": '"', "‘": "'", "’": "'", "‚": "'", "‛": "'", "「": '"', "」": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-", "ㆍ": "·", "・": "·", "･": "·", "∙": "·",
    "〜": "~", "∼": "~",
})


def norm_text(s: Any) -> str:
    """NFKC · casefold · 따옴표/줄표 통일 · 공백 제거 (한국어 띄어쓰기 차이를 무시한다)."""
    t = unicodedata.normalize("NFKC", str(s or "")).casefold().translate(_PUNCT_MAP)
    return _ZW_RX.sub("", t)


_DATE_RX = re.compile(r"^\s*(\d{4})\s*(?:[-./]|년)\s*(\d{1,2})\s*(?:[-./]|월)\s*(\d{1,2})\s*(?:일|\.)?\s*$")


def parse_date(v: Any) -> Optional[Tuple[int, int, int]]:
    m = _DATE_RX.match(str(v or ""))
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return y, mo, d


def date_variants(v: Any) -> List[str]:
    """'2007-03-22' → 원문에 흔한 표기들 (정규화 전 문자열). 첫 항목이 입력 그대로."""
    p = parse_date(v)
    if not p:
        return [str(v)]
    y, m, d = p
    forms = [str(v).strip(), f"{y}-{m:02d}-{d:02d}", f"{y}년 {m}월 {d}일", f"{y}년 {m:02d}월 {d:02d}일",
             f"{y}. {m}. {d}.", f"{y}.{m:02d}.{d:02d}", f"{y}.{m}.{d}", f"{y}/{m:02d}/{d:02d}", f"{y}-{m}-{d}"]
    out, seen = [], set()
    for f in forms:
        n = norm_text(f).rstrip(".")
        if n and n not in seen:
            seen.add(n)
            out.append(f)
    return out


_AMOUNT_RX = re.compile(r"^\s*(?:금\s*)?([\d][\d,\s]*)\s*(원|₩)?\s*$")


def amount_variants(v: Any) -> Optional[List[str]]:
    """266464758 / '266,464,758원' → ['266,464,758원', '266,464,758', '266464758원', '266464758'] (긴 것 먼저)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if isinstance(v, float) and not v.is_integer():
            return None
        digits = str(int(v))
    else:
        m = _AMOUNT_RX.match(str(v or ""))
        if not m:
            return None
        digits = re.sub(r"\D", "", m.group(1))
    if len(digits) < 4:
        return None
    grouped = f"{int(digits):,}"
    return [f"{grouped}원", grouped, f"{digits}원", digits]


def work_type_variants(v: Any) -> List[str]:
    s = str(v or "").strip()
    short = s[:-3] if s.endswith("저작물") else s
    out = [s]
    for f in (f"{short}저작물", short):
        if f and f not in out:
            out.append(f)
    return out


def parse_jsonish(v: Any) -> Any:
    if isinstance(v, str):
        t = v.strip()
        if t[:1] in ("[", "{"):
            try:
                return json.loads(t)
            except (ValueError, TypeError):
                return v
    return v


def _scalar_list(v: Any) -> Optional[List[str]]:
    """스칼라 또는 스칼라 목록 → 문자열 목록 (사전·중첩은 None)."""
    v = parse_jsonish(v)
    if v is None or v == "" or v == []:
        return []
    if isinstance(v, (str, int, float)) and not isinstance(v, bool):
        return [str(v)]
    if isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float)) and not isinstance(x, bool)
                                            for x in v if x is not None):
        return [str(x) for x in v if x not in (None, "")]
    return None


def _leaves(v: Any, out: List[str], depth: int = 0) -> None:
    v = parse_jsonish(v)
    if depth > 3:
        return
    if isinstance(v, dict):
        for k, x in v.items():
            if str(k).lower() in LEAF_KEYS and isinstance(x, str) and len(norm_text(x)) >= 2:
                out.append(x)
            elif isinstance(x, (dict, list)):
                _leaves(x, out, depth + 1)
    elif isinstance(v, list):
        for x in v:
            if isinstance(x, (dict, list)):
                _leaves(x, out, depth + 1)


# ---------------------------------------------------------------------------
# 쪽 글자 흐름 (공백 없는 정규화 문자열 + 글자마다 정규화 상자 + 줄 번호)
# ---------------------------------------------------------------------------
Rect = Tuple[float, float, float, float]


def _union(rs: Iterable[Rect]) -> Rect:
    rs = list(rs)
    return (min(r[0] for r in rs), min(r[1] for r in rs), max(r[2] for r in rs), max(r[3] for r in rs))


def _clamp01(r: Rect) -> Rect:
    x0, y0, x1, y1 = (min(1.0, max(0.0, float(v))) for v in r)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def norm_rect(r: Sequence[float], width: float, height: float) -> Rect:
    """절대 좌표 상자 → 쪽 기준 0–1 (뒤집힌 상자는 바로 세우고 0–1 로 자른다)."""
    if not width or not height:
        raise ValueError("쪽 크기가 0입니다")
    x0, y0, x1, y1 = (float(v) for v in r[:4])
    return _clamp01((x0 / width, y0 / height, x1 / width, y1 / height))


@dataclass
class Stream:
    text: str = ""
    boxes: List[Rect] = dc_field(default_factory=list)
    line_of: List[int] = dc_field(default_factory=list)
    line_start: List[int] = dc_field(default_factory=list)       # 줄마다 흐름에서 시작 위치
    line_end: List[int] = dc_field(default_factory=list)
    line_rect: List[Optional[Rect]] = dc_field(default_factory=list)
    raw_lines: List[str] = dc_field(default_factory=list)       # 줄마다 원문 (공백 포함)
    raw_of: List[int] = dc_field(default_factory=list)          # 정규화 글자마다 그 줄 원문에서의 위치

    def add_line(self, chars: Sequence[Tuple[str, Rect]]) -> None:
        cs, bs, rs = _norm_chars(chars)
        li = len(self.line_start)
        self.line_start.append(len(self.boxes))
        self.text += "".join(cs)
        self.boxes.extend(bs)
        self.line_of.extend([li] * len(cs))
        self.raw_of.extend(rs)
        self.line_end.append(len(self.boxes))
        self.line_rect.append(_union([r for _, r in chars]) if chars else None)
        self.raw_lines.append("".join(c for c, _ in chars))

    def line_text(self, li: int) -> str:
        return self.text[self.line_start[li]:self.line_end[li]]

    def raw_text(self, s: int, e: int) -> str:
        """흐름 구간 → 원문 표기 (줄이 바뀌면 공백 하나)."""
        if e <= s or s >= len(self.raw_of):
            return ""
        e = min(e, len(self.raw_of))
        la, lb = self.line_of[s], self.line_of[e - 1]
        ra, rb = self.raw_of[s], self.raw_of[e - 1]
        if la == lb:
            return self.raw_lines[la][ra:rb + 1].strip()
        parts = [self.raw_lines[la][ra:]] + self.raw_lines[la + 1:lb] + [self.raw_lines[lb][:rb + 1]]
        return " ".join(x.strip() for x in parts if x.strip())

    def rects(self, s: int, e: int) -> List[Rect]:
        """흐름 구간 → 줄마다 하나씩 합친 상자."""
        by: Dict[int, List[Rect]] = {}
        for i in range(max(0, s), min(e, len(self.boxes))):
            by.setdefault(self.line_of[i], []).append(self.boxes[i])
        return [_union(by[li]) for li in sorted(by)]


def _norm_chars(chars: Sequence[Tuple[str, Rect]]) -> Tuple[List[str], List[Rect], List[int]]:
    """한 줄의 (글자, 상자) → 정규화 글자 · 그 상자 · 원문 위치. NFKC 가 앞 글자와 합치는 경우(분리 자모 등)도 상자를
    합쳐 따라간다."""
    out_c: List[str] = []
    out_b: List[Rect] = []
    out_r: List[int] = []
    acc = ""
    for ri, (ch, rect) in enumerate(chars):
        acc += ch
        n = norm_text(acc)
        cur = "".join(out_c)
        if n.startswith(cur):
            for c in n[len(cur):]:
                out_c.append(c)
                out_b.append(rect)
                out_r.append(ri)
            continue
        k = 0
        while k < min(len(n), len(cur)) and n[k] == cur[k]:
            k += 1
        merged = _union(out_b[k:] + [rect])
        r0 = out_r[k] if k < len(out_r) else ri
        del out_c[k:]
        del out_b[k:]
        del out_r[k:]
        for c in n[k:]:
            out_c.append(c)
            out_b.append(merged)
            out_r.append(r0)
    return out_c, out_b, out_r


def text_layer_ok(text: str) -> bool:
    """텍스트 레이어로 쓸 만한가 — 30자 이상이고 깨진 글자(대체 문자·사용자 영역·제어 문자)가 5% 이하."""
    s = (text or "").strip()
    if len(s) < TEXT_LAYER_MIN_CHARS:
        return False
    bad = sum(1 for c in s if c == "�" or 0xE000 <= ord(c) <= 0xF8FF
              or (unicodedata.category(c) == "Cc" and c not in "\n\r\t"))
    return bad / len(s) <= 0.05


def text_layer_stream(page) -> Stream:
    """PyMuPDF 쪽 → 글자 흐름. 상자는 회전을 반영한 화면 좌표를 쪽 크기로 나눈 0–1."""
    import fitz
    st = Stream()
    W, H = float(page.rect.width), float(page.rect.height)
    mat = page.rotation_matrix
    rd = page.get_text("rawdict")
    for b in rd.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            chars: List[Tuple[str, Rect]] = []
            for sp in line.get("spans", []):
                for ch in sp.get("chars", []):
                    c = ch.get("c") or ""
                    if not c:
                        continue
                    r = fitz.Rect(ch["bbox"]) * mat if page.rotation else fitz.Rect(ch["bbox"])
                    chars.append((c, norm_rect((r.x0, r.y0, r.x1, r.y1), W, H)))
            if chars:
                st.add_line(chars)
    return st


def _char_weight(c: str) -> float:
    if c.isspace():
        return 0.3
    ea = unicodedata.east_asian_width(c)
    if ea in ("W", "F") or (ea == "A" and not c.isascii()):    # 한글 글꼴은 □■“” 같은 모호 폭 기호도 전각으로 그린다
        return 1.0
    if c.isalnum():
        return 0.56
    return 0.4


def ocr_stream(lines: Sequence[Dict[str, Any]]) -> Stream:
    """OCR 줄 [{text, box(0–1)}] → 글자 흐름. 글자 상자는 줄 상자를 글자 폭 비율로 나눠 어림한다."""
    st = Stream()
    for ln in lines:
        text = str(ln.get("text") or "")
        box = ln.get("box")
        if not text.strip() or not box:
            continue
        x0, y0, x1, y1 = box
        ws = [_char_weight(c) for c in text]
        tot = sum(ws) or 1.0
        acc = 0.0
        chars = []
        for c, w in zip(text, ws):
            a = x0 + (x1 - x0) * acc / tot
            acc += w
            b = x0 + (x1 - x0) * acc / tot
            chars.append((c, (a, y0, b, y1)))
        st.add_line(chars)
    return st


# ---------------------------------------------------------------------------
# 찾기
# ---------------------------------------------------------------------------
def _digit_ok(hay: str, s: int, e: int, needle: str) -> bool:
    if needle[:1].isdigit() and s > 0 and hay[s - 1].isdigit():
        return False
    if needle[-1:].isdigit() and e < len(hay) and hay[e].isdigit():
        return False
    return True


def find_all(hay: str, needle: str) -> List[Tuple[int, int]]:
    """정규화 흐름에서 needle(정규화된) 을 겹치지 않게 모두 찾는다 (숫자 경계 확인)."""
    out: List[Tuple[int, int]] = []
    if not needle:
        return out
    i = hay.find(needle)
    while i >= 0:
        e = i + len(needle)
        if _digit_ok(hay, i, e, needle):
            out.append((i, e))
            i = hay.find(needle, e)
        else:
            i = hay.find(needle, i + 1)
    return out


def fuzzy_tolerance(n: int) -> int:
    return max(1, n // 6)


def _char_class(c: str) -> str:
    if "\uac00" <= c <= "\ud7a3" or "\u1100" <= c <= "\u11ff" or "\u3130" <= c <= "\u318f":
        return "h"
    if c.isdigit() or (c.isascii() and c.isalpha()):
        return "a"                                   # OCR 은 0/O · 1/l 을 섞는다 — 숫자·라틴은 한 묶음
    if c.isalpha():
        return "o"
    return "p"


def _sub_cost(a: str, b: str) -> int:
    """바꿔 읽기 비용 — 같은 글자 0, 같은 갈래(한글끼리·숫자/라틴끼리) 1, 갈래가 다르면(한글 ↔ 문장부호 등) 2."""
    if a == b:
        return 0
    ca = _char_class(a)
    return 1 if ca == _char_class(b) and ca != "p" else 2


def fuzzy_find(hay: str, needle: str, tol: Optional[int] = None) -> List[Tuple[int, int, int]]:
    """근사 부분 문자열 찾기 (편집 거리 ≤ tol, 기본 max(1, len/6)) → [(시작, 끝, 거리)] 겹치지 않게, 가까운 것 먼저.

    OCR 오독(양지웅 → 양지용)을 받아들이되, 글자를 문장부호로 바꿔 읽은 것은 2로 세어 짧은 값이 엉뚱한 자리에 붙지 않게 한다.
    """
    m, n = len(needle), len(hay)
    if m == 0 or n == 0:
        return []
    if tol is None:
        tol = fuzzy_tolerance(m)
    if len(set(needle) & set(hay)) < max(1, len(set(needle)) - tol):     # 빠른 거르기
        return []
    prev_d = [0] * (n + 1)
    prev_s = list(range(n + 1))
    for i in range(1, m + 1):
        cur_d = [i] * (n + 1)
        cur_s = [0] * (n + 1)
        nc = needle[i - 1]
        for j in range(1, n + 1):
            d = prev_d[j - 1] + _sub_cost(hay[j - 1], nc)
            s = prev_s[j - 1]
            if prev_d[j] + 1 < d:
                d, s = prev_d[j] + 1, prev_s[j]
            if cur_d[j - 1] + 1 < d:
                d, s = cur_d[j - 1] + 1, cur_s[j - 1]
            cur_d[j], cur_s[j] = d, s
        prev_d, prev_s = cur_d, cur_s
    cands = sorted(((prev_d[j], prev_s[j], j) for j in range(1, n + 1) if prev_d[j] <= tol),
                   key=lambda x: (x[0], abs((x[2] - x[1]) - m), x[1]))
    chosen: List[Tuple[int, int, int]] = []
    for d, s, e in cands:
        if e - s < max(1, m - tol):
            continue
        if any(not (e <= cs or s >= ce) for _, cs, ce in chosen):
            continue
        chosen.append((d, s, e))
    return sorted(((s, e, d) for d, s, e in chosen), key=lambda x: x[0])


def _glyph_before(st: Stream, s: int) -> int:
    """s 바로 앞이 체크 칸 글자면 그 위치, 아니면 s."""
    if s <= 0:
        return s
    c = st.text[s - 1]
    if c in CHECK_GLYPHS:
        if c == "v" and s >= 2 and st.text[s - 2].isascii() and st.text[s - 2].isalpha():
            return s
        return s - 1
    return s


def _is_checked(st: Stream, s: int) -> bool:
    g = _glyph_before(st, s)
    return g < s and st.text[g] in CHECKED_GLYPHS


# ---------------------------------------------------------------------------
# 대상
# ---------------------------------------------------------------------------
@dataclass
class Target:
    target: str
    label_ko: str
    key: Optional[str] = None
    no: Optional[int] = None
    field: Optional[str] = None
    kind: str = "text"                    # text | name | date | period | amount | work_type | right
    values: List[Any] = dc_field(default_factory=list)
    miss_reason: str = R_NOT_FOUND
    skip_reason: Optional[str] = None     # 찾지 않는 값 (요약·생성 / 문서 밖)
    primary: Dict[str, Any] = dc_field(default_factory=dict)   # {"mode": "anchor"|"last"|"checked", "anchors": (...)}
    alias: Optional[str] = None


def _keys23():
    try:
        from module.evaluation.v4.keys23 import KEYS
    except ImportError:                  # pragma: no cover
        from api.module.evaluation.v4.keys23 import KEYS
    return KEYS


def build_targets(record: Dict[str, Any], decisions: Sequence[Dict[str, Any]], test: Optional[str],
                  key23_of: Optional[Callable[[str], Optional[str]]] = None,
                  label_of: Optional[Callable[[str], Optional[str]]] = None) -> List[Target]:
    """세트 레코드(23키) + 통합 결정 → 찾을 대상."""
    meta = record.get("_meta") or {}
    pm = meta.get("pipeline_meta") or {}
    ctx = meta.get("contract_context") if isinstance(meta.get("contract_context"), dict) else {}
    out: List[Target] = []
    by_key: Dict[str, Target] = {}
    for K in _keys23():
        k = K.name
        v = record.get(k)
        if k in KEY_SKIP or v is None or v == "" or v == []:
            continue
        t = Target(target=f"k:{k}", key=k, no=K.no, label_ko=K.label_ko)
        if k in KEY_GENERATED:
            t.skip_reason, t.values = R_GENERATED, [_short(v)]
        elif k in KEY_OUTSIDE:
            t.skip_reason, t.values = R_OUTSIDE, [_short(v)]
        elif k == "work_title":
            t.values = [str(v)]
            src = pm.get("work_title_source") if test == "T-001" else (ctx or {}).get("source")
            if src == "rules":
                t.primary = {"mode": "anchor", "anchors": ANCHORS["title"], "window": 40}
        elif k == "work_type":
            t.kind, t.values, t.miss_reason = "work_type", [str(v)], R_NORMALISED
            t.primary = {"mode": "checked", "anchors": ANCHORS["work_type"], "window": 120}
        elif k in ("creator", "copyright_holder"):
            t.kind = "name"
            t.values = _scalar_list(v) or [str(v)]
            if k == "creator":
                t.primary = {"mode": "anchor", "anchors": ANCHORS["creator"], "window": 60}
            else:
                src = ((pm.get("licensee_rule") or {}) if isinstance(pm.get("licensee_rule"), dict) else {}).get("source")
                if src == "signature":
                    t.primary = {"mode": "last"}
                elif src == "preamble":
                    t.primary = {"mode": "anchor", "anchors": ANCHORS["preamble_licensee"], "window": 30}
        elif k in RIGHT_LABELS:
            t.kind, t.values = "right", [str(v)]
        elif k in ("license_start_date", "license_end_date"):
            t.kind, t.values, t.miss_reason = "date", [str(v)], R_NORMALISED
            t.primary = {"mode": "anchor", "anchors": ANCHORS["period"], "window": 40}
        elif k == "contract_validity_period":
            t.kind, t.values, t.miss_reason = "period", [str(v)], R_NORMALISED
            t.primary = {"mode": "anchor", "anchors": ANCHORS["period"], "window": 20}
        else:
            vals = _scalar_list(v)
            if vals is None:
                t.skip_reason, t.values = R_NORMALISED, [_short(v)]
            else:
                t.values = vals
        out.append(t)
        by_key[k] = t

    seen_f = set()
    for d in decisions or []:
        if not isinstance(d, dict):
            continue
        f = str(d.get("field") or "")
        fv = d.get("final_value")
        if not f or f in seen_f or fv is None or fv == "" or fv == []:
            continue
        seen_f.add(f)
        lab = (label_of(f) if label_of else None) or d.get("label_ko") or f
        t = Target(target=f"f:{f}", field=f, label_ko=str(lab))
        k23 = (key23_of(f) if key23_of else None) or d.get("key23")
        if k23 and k23 in by_key and not by_key[k23].skip_reason:
            t.alias = by_key[k23].target
            out.append(t)
            continue
        if k23:
            t.key = k23
        base = re.split(r"[\[.]", f, maxsplit=1)[0]
        pv = parse_jsonish(fv)
        abbreviated = isinstance(pv, str) and pv.strip()[:1] in ("[", "{") and ("..." in pv or "…" in pv)
        if base in FIELD_GENERATED or f in FIELD_GENERATED or abbreviated:
            t.skip_reason, t.values = R_GENERATED, [_short(fv)]
        elif isinstance(pv, (dict,)) or (isinstance(pv, list) and any(isinstance(x, (dict, list)) for x in pv)):
            leaves: List[str] = []
            _leaves(pv, leaves)
            if leaves:
                t.values = list(dict.fromkeys(leaves))
            else:
                t.skip_reason, t.values = R_GENERATED, [_short(fv)]
        else:
            vals = _scalar_list(fv) or []
            if vals and all(x in _ALL_RIGHT_NAMES or norm_text(x) in {norm_text(r) for r in _ALL_RIGHT_NAMES}
                            for x in vals):
                t.kind = "right_names"
            elif len(vals) == 1 and parse_date(vals[0]):
                t.kind, t.miss_reason = "date", R_NORMALISED
            elif len(vals) == 1 and (FIELD_AMOUNT_RX.search(f) or isinstance(pv, (int, float))) \
                    and amount_variants(pv if isinstance(pv, (int, float)) else vals[0]):
                t.kind, t.miss_reason = "amount", R_NORMALISED
                vals = [pv if isinstance(pv, (int, float)) else vals[0]]
            elif FIELD_DATE_RX.search(f) and vals and not parse_date(vals[0]):
                t.miss_reason = R_NORMALISED
            t.values = vals
            if t.kind not in ("amount", "date") and vals and all(len(norm_text(x)) < MIN_VALUE_CHARS for x in vals):
                t.skip_reason = R_NORMALISED          # '원' 같은 한 글자 값 — 문서 곳곳에 있어 자리를 가리키지 못한다
        out.append(t)
    return out


def _short(v: Any, n: int = 80) -> str:
    if isinstance(v, (list, tuple)):
        s = ", ".join(str(x) for x in v)
    elif isinstance(v, dict):
        s = json.dumps(v, ensure_ascii=False)
    else:
        s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# 대상 → 표시 (쪽 흐름에서)
# ---------------------------------------------------------------------------
@dataclass
class Hit:
    page: int
    s: int
    e: int
    value: Any
    match: str
    text: str
    rects: List[Rect]
    checked: bool = False


def _variants_for(t: Target, value: Any) -> List[Tuple[str, str]]:
    """(정규화 needle, match 종류) — 긴 것 먼저."""
    raw = norm_text(value)
    if t.kind == "date":
        vs = [norm_text(x).rstrip(".") for x in date_variants(value)]
    elif t.kind == "amount":
        vs = [norm_text(x) for x in (amount_variants(value) or [str(value)])]
    elif t.kind == "work_type":
        vs = [norm_text(x) for x in work_type_variants(value)]
    else:
        vs = [raw]
    out, seen = [], set()
    for v in sorted(vs, key=len, reverse=True):
        if v and v not in seen:
            seen.add(v)
            out.append((v, "exact" if v == raw.rstrip(".") or v == raw else "variant"))
    return out


def _pick_spans(cands: List[Tuple[int, int, str, str]]) -> List[Tuple[int, int, str, str]]:
    """겹치는 후보 중 긴 것(같으면 exact) 하나만."""
    order = {"exact": 0, "variant": 1, "fuzzy": 2}
    cands = sorted(cands, key=lambda c: (-(c[1] - c[0]), order.get(c[2], 3), c[0]))
    chosen: List[Tuple[int, int, str, str]] = []
    for c in cands:
        if any(not (c[1] <= x[0] or c[0] >= x[1]) for x in chosen):
            continue
        chosen.append(c)
    return sorted(chosen, key=lambda c: c[0])


def _allow_fuzzy(needle: str) -> bool:
    if len(needle) < 3:
        return False
    digits = sum(c.isdigit() for c in needle)
    return digits / len(needle) <= 0.3


def _search_value(t: Target, value: Any, pages: Dict[int, Tuple[Stream, str]]) -> List[Hit]:
    hits: List[Hit] = []
    variants = _variants_for(t, value)
    for pno, (st, src) in pages.items():
        cands: List[Tuple[int, int, str, str]] = []
        for needle, kind in variants:
            for s, e in find_all(st.text, needle):
                cands.append((s, e, kind, needle))
        if not cands and src == SRC_OCR and variants:
            needle = variants[0][0] if t.kind not in ("date", "amount") else ""
            if needle and _allow_fuzzy(needle):
                for li in range(len(st.line_start)):
                    a = st.line_start[li]
                    b = st.line_end[li + 1] if li + 1 < len(st.line_end) else st.line_end[li]
                    for s, e, d in fuzzy_find(st.text[a:b], needle):
                        cands.append((a + s, a + e, "fuzzy", needle))
        for s, e, kind, _n in _pick_spans(cands):
            gs = s
            if t.kind == "work_type":
                gs = _glyph_before(st, s)
            hits.append(Hit(page=pno, s=gs, e=e, value=value, match=kind, text=st.raw_text(gs, e), rects=st.rects(gs, e),
                            checked=_is_checked(st, s)))
    return hits


def _right_regex(label: str) -> re.Pattern:
    """권리 이름 → 정규식 (글자 사이 각주 표시 '1)' 허용) + 바로 뒤 괄호 속 세부 권리."""
    n = norm_text(label)
    foot = r"(?:\d{1,2}\)|[¹²³⁴⁵⁶⁷⁸⁹⁰]+)?"
    body = foot.join(re.escape(c) for c in n)
    return re.compile(body + r"(?:\([^()]{0,60}\))?")


def _search_right(t: Target, label: str, pages: Dict[int, Tuple[Stream, str]]) -> List[Hit]:
    rx = _right_regex(label)
    others = [norm_text(x) for x in RIGHT_LABELS.values() if x != label]
    hits: List[Hit] = []
    for pno, (st, src) in pages.items():
        anchors = [e for a in ANCHORS["rights"] for _, e in find_all(st.text, norm_text(a))]
        for m in rx.finditer(st.text):
            s, e = m.start(), m.end()
            gs = _glyph_before(st, s)
            in_box = gs < s or any(0 <= s - a <= 260 for a in anchors)
            if not in_box:
                continue
            rects = st.rects(gs, e)
            li = st.line_of[s] if s < len(st.line_of) else None
            if li is not None and not any(o in st.line_text(li) for o in others) and gs < s:
                # 그 줄에 이 권리 하나뿐 — 줄 전체 (한 줄에 하나씩 적는 서식)
                rects = [st.line_rect[li]] if st.line_rect[li] else rects
            hits.append(Hit(page=pno, s=gs, e=e, value=t.values[0] if t.values else None, match="exact",
                            text=st.raw_text(gs, e), rects=rects, checked=gs < s and st.text[gs] in CHECKED_GLYPHS))
    return hits


def _search_period(t: Target, value: Any, pages: Dict[int, Tuple[Stream, str]]) -> List[Hit]:
    parts = re.split(r"\s*[~∼〜]\s*|\s+-\s+", str(value or ""), maxsplit=1)
    if len(parts) != 2 or not (parse_date(parts[0]) and parse_date(parts[1])):
        sub = Target(target=t.target, label_ko=t.label_ko, kind="text", values=[value])
        return _search_value(sub, value, pages)
    a_vs = [norm_text(x).rstrip(".") for x in date_variants(parts[0])]
    b_vs = [norm_text(x).rstrip(".") for x in date_variants(parts[1])]
    alt = lambda vs: "|".join(re.escape(v) for v in sorted(set(vs), key=len, reverse=True))  # noqa: E731
    rx = re.compile(rf"(?<!\d)(?:{alt(a_vs)})\.?.{{0,14}}?(?:{alt(b_vs)})(?!\d)\.?")
    hits: List[Hit] = []
    for pno, (st, src) in pages.items():
        for m in rx.finditer(st.text):
            hits.append(Hit(page=pno, s=m.start(), e=m.end(), value=value, match="variant",
                            text=st.raw_text(m.start(), m.end()), rects=st.rects(m.start(), m.end())))
    if hits:
        return hits
    sub_a = Target(target=t.target, label_ko=t.label_ko, kind="date", values=[parts[0]])
    sub_b = Target(target=t.target, label_ko=t.label_ko, kind="date", values=[parts[1]])
    return _search_value(sub_a, parts[0], pages) + _search_value(sub_b, parts[1], pages)


def _choose_primary(t: Target, hits: List[Hit], pages: Dict[int, Tuple[Stream, str]]) -> Optional[int]:
    if not hits:
        return None
    mode = (t.primary or {}).get("mode")
    if mode == "last":
        return len(hits) - 1
    if mode == "checked":
        for i, h in enumerate(hits):
            if h.checked:
                return i
    anchors = (t.primary or {}).get("anchors") or ()
    window = int((t.primary or {}).get("window") or 60)
    if anchors:
        for i, h in enumerate(hits):
            st = pages[h.page][0]
            for a in anchors:
                for _, ae in find_all(st.text, norm_text(a)):
                    if 0 <= h.s - ae <= window:
                        return i
    return 0


def match_targets(targets: Sequence[Target], pages: Dict[int, Tuple[Stream, str]]
                  ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """대상 → (marks, not_found, target 요약)."""
    marks: List[Dict[str, Any]] = []
    not_found: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    order = sorted(pages)
    pages = {p: pages[p] for p in order}
    for t in targets:
        base = {"target": t.target, "key": t.key, "no": t.no, "field": t.field, "label_ko": t.label_ko}
        if t.alias:
            summary.append({**base, "alias": t.alias})
            continue
        row = {**base, "values": [_short(v) for v in t.values], "marks": 0, "pages": [], "primary": None,
               "reason": None}
        if t.skip_reason:
            row["reason"] = t.skip_reason
            for v in t.values or [None]:
                not_found.append({**base, "value": _short(v) if v is not None else None, "reason": t.skip_reason})
            summary.append(row)
            continue
        t_hits: List[Tuple[Hit, bool]] = []
        missing: List[Any] = []
        for v in t.values:
            if t.kind == "right":
                hits = _search_right(t, RIGHT_LABELS[t.key], pages) if t.key in RIGHT_LABELS else []
            elif t.kind == "right_names":
                lab = next((r for r in _ALL_RIGHT_NAMES if norm_text(r) == norm_text(v)), str(v))
                sub = Target(target=t.target, label_ko=t.label_ko, values=[v])
                hits = _search_right(sub, lab, pages)
            elif t.kind == "period":
                hits = _search_period(t, v, pages)
            else:
                hits = _search_value(t, v, pages)
            hits.sort(key=_reading_order)
            if not hits:
                missing.append(v)
                continue
            pi = _choose_primary(t, hits, pages)
            t_hits.extend((h, i == pi) for i, h in enumerate(hits))
        # 같은 대상의 겹친 표시(이름이 서로 포함되는 경우 등)는 하나로
        uniq: List[Tuple[Hit, bool]] = []
        for h, pri in sorted(t_hits, key=lambda x: (_reading_order(x[0]), -(x[0].e - x[0].s))):
            dup = next((j for j, (u, _) in enumerate(uniq) if u.page == h.page and not (h.e <= u.s or h.s >= u.e)), None)
            if dup is None:
                uniq.append((h, pri))
            elif pri and not uniq[dup][1]:
                uniq[dup] = (uniq[dup][0], True)
        prim_row = None
        for i, (h, pri) in enumerate(uniq):
            src = pages[h.page][1]
            m = {"id": f"{t.target}#{i}", **base, "value": _mark_value(t, h.value), "text": h.text, "page": h.page,
                 "rects": [[round(x, 4) for x in r] for r in h.rects], "match": h.match, "primary": bool(pri),
                 "source": src}
            marks.append(m)
            if pri and prim_row is None:
                prim_row = {"page": h.page, "mark": m["id"]}
        for v in missing:
            not_found.append({**base, "value": _short(v), "reason": t.miss_reason})
        row["marks"] = len(uniq)
        row["pages"] = sorted({h.page for h, _ in uniq})
        row["primary"] = prim_row or ({"page": uniq[0][0].page, "mark": f"{t.target}#0"} if uniq else None)
        if not uniq:
            row["reason"] = t.miss_reason
        summary.append(row)
    return marks, not_found, summary


def _reading_order(h: Hit) -> Tuple[int, float, float, int]:
    """읽는 순서 — 쪽, 위→아래(줄 높이의 절반 안이면 같은 줄), 왼→오른. 텍스트 레이어의 내용 순서와 OCR 줄 순서가 달라도
    '본문의 첫 자리' 가 같게 나온다."""
    r = h.rects[0] if h.rects else (0.0, 0.0, 0.0, 0.0)
    return (h.page, round(r[1] / 0.006), r[0], h.s)


def _mark_value(t: Target, v: Any) -> Any:
    if t.kind == "amount" and isinstance(v, (int, float)):
        return f"{int(v):,}"
    return _short(v, 120) if not isinstance(v, (int, float)) else v


# ---------------------------------------------------------------------------
# Qwen 줄 위치 찾기 (스캔 쪽)
# ---------------------------------------------------------------------------
class OcrParseError(ValueError):
    pass


def ocr_mode() -> str:
    v = (os.getenv("CONSOLE_LOCATE_OCR") or "qwen").strip().lower()
    return "off" if v in ("off", "0", "false", "no", "none") else "qwen"


def ocr_models() -> List[str]:
    raw = os.getenv("CONSOLE_LOCATE_MODELS") or ",".join(DEFAULT_MODELS)
    out = [m.strip() for m in raw.split(",") if m.strip()]
    return out or list(DEFAULT_MODELS)


def _thinking_off(model: str) -> bool:
    return bool(re.match(r"^qwen3\.\d+-(?:flash|plus|max|turbo)\b", model or "")) and "omni" not in model


def model_cost_krw(model: str, tokens_in: Optional[int], tokens_out: Optional[int]) -> Optional[float]:
    p = PRICE_USD_PER_M.get(model)
    if p is None or tokens_in is None or tokens_out is None:
        return None
    return round((tokens_in * p[0] + tokens_out * p[1]) / 1e6 * KRW_PER_USD, 3)


def _default_client():
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DashScope 키(DASHSCOPE_API_KEY)가 없습니다")
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL,
                  timeout=OCR_TIMEOUT_SEC, max_retries=1)


_CLIENT_FACTORY: Callable[[], Any] = _default_client


def set_client_factory(fn: Optional[Callable[[], Any]]) -> None:
    """테스트용 — DashScope 클라이언트를 바꿔 끼운다 (None → 기본)."""
    global _CLIENT_FACTORY
    _CLIENT_FACTORY = fn or _default_client


def _json_array(text: str) -> List[Any]:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    m = re.search(r"\[.*\]", t, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, list):
                return v
        except ValueError:
            pass
    # 잘린 응답 — 온전한 객체만 건진다
    objs = []
    for om in re.finditer(r"\{[^{}]*\}", t):
        try:
            objs.append(json.loads(om.group(0)))
        except ValueError:
            continue
    if objs:
        return objs
    raise OcrParseError("JSON 배열이 아닙니다")


def _box_of(L: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    b = L.get("bbox_2d") or L.get("bbox") or L.get("box")
    if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b):
        return tuple(float(x) for x in b)                                      # type: ignore[return-value]
    if isinstance(b, (list, tuple)) and len(b) == 8 and all(isinstance(x, (int, float)) for x in b):
        xs, ys = b[0::2], b[1::2]
        return (min(xs), min(ys), max(xs), max(ys))
    if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in b):
        xs, ys = [p[0] for p in b], [p[1] for p in b]
        return (min(xs), min(ys), max(xs), max(ys))
    rr = L.get("rotate_rect")
    if isinstance(rr, (list, tuple)) and len(rr) >= 4:
        cx, cy, w, h = (float(x) for x in rr[:4])
        a = float(rr[4]) if len(rr) > 4 else 0.0
        if abs(abs(a) - 90) < 20:                    # ~90° 회전 사각형 — 가로·세로가 뒤바뀌어 온다
            w, h = h, w
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    return None


def parse_ocr_lines(text: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """모델 응답 → [{text, box(0–1)}]. 좌표가 1000 을 넘지 않으면 0–1000 상대 좌표(Qwen3 계열), 넘으면 픽셀로 본다."""
    raw = _json_array(text)
    items = []
    for L in raw:
        if not isinstance(L, dict):
            continue
        t = L.get("text")
        if not isinstance(t, str) or not t.strip():
            continue
        b = _box_of(L)
        if b is None:
            continue
        items.append((t, b))
    if not items:
        raise OcrParseError("줄이 없습니다")
    mx = max(max(b) for _, b in items)
    rel = mx <= 1000.5
    out = []
    for t, (x0, y0, x1, y1) in items:
        if rel:
            r = _clamp01((x0 / 1000.0, y0 / 1000.0, x1 / 1000.0, y1 / 1000.0))
        else:
            r = norm_rect((x0, y0, x1, y1), img_w, img_h)
        if r[2] - r[0] <= 0 or r[3] - r[1] <= 0:
            continue
        out.append({"text": t, "box": [round(v, 5) for v in r]})
    if not out:
        raise OcrParseError("쓸 수 있는 상자가 없습니다")
    return out


def _err_text(e: BaseException) -> str:
    s = (str(e).splitlines() or [""])[0][:200]
    return LC.scrub_secret(f"{type(e).__name__}: {s}" if s else type(e).__name__)


def ocr_page(img: bytes, mime: str, img_w: int, img_h: int, page: int, models: Sequence[str]) -> Dict[str, Any]:
    """한 쪽 → {lines, model, tokens_in, tokens_out, seconds, cost_krw, errors}. 앞 모델이 실패(호출 오류·JSON 해석 실패)하면
    다음 모델로. 모두 실패하면 lines=[] 와 errors."""
    url = f"data:{mime};base64," + base64.b64encode(img).decode()
    errors: List[str] = []
    t_all = time.time()
    try:
        client = _CLIENT_FACTORY()
    except Exception as e:
        msg = _err_text(e)
        logger.warning("원본 위치 OCR 준비 실패 — %d쪽: %s", page, msg, extra={"component": "OCR"})
        return {"lines": [], "model": None, "errors": [msg], "seconds": 0.0, "cost_krw": 0.0}
    cost = 0.0
    for i, model in enumerate(models):
        nxt = models[i + 1] if i + 1 < len(models) else None
        t0 = time.time()
        try:
            kw: Dict[str, Any] = {}
            if _thinking_off(model):
                kw["extra_body"] = {"enable_thinking": False}
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=OCR_MAX_TOKENS,
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                       {"type": "text", "text": PROMPT_LINES}]}], **kw)
            u = getattr(r, "usage", None)
            tin = getattr(u, "prompt_tokens", None) if u is not None else None
            tout = getattr(u, "completion_tokens", None) if u is not None else None
            c = model_cost_krw(model, tin, tout)
            cost += c or 0.0
            content = r.choices[0].message.content if r.choices else ""
            if isinstance(content, list):                 # 일부 SDK 는 조각 목록으로 준다
                content = "".join(str(getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else "") or "")
                                  for p in content)
            try:
                lines = parse_ocr_lines(content or "", img_w, img_h)
            except OcrParseError as pe:
                errors.append(f"{model}: 응답 해석 실패 ({pe})")
                logger.warning("원본 위치 OCR 응답을 해석하지 못했습니다 — %d쪽 %s (%s)%s", page, model, pe,
                               f" → {nxt} 로 다시" if nxt else "", extra={"component": "OCR"})
                continue
            sec = time.time() - t0
            if i > 0:
                logger.warning("원본 위치 OCR 대체 모델 사용 — %d쪽 %s", page, model, extra={"component": "OCR"})
            return {"lines": lines, "model": model, "tokens_in": tin, "tokens_out": tout, "seconds": round(sec, 2),
                    "cost_krw": round(cost, 3), "errors": errors}
        except Exception as e:                           # 호출 실패 → 다음 모델
            msg = _err_text(e)
            errors.append(f"{model}: {msg}")
            logger.warning("원본 위치 OCR 실패 — %d쪽 %s: %s%s", page, model, msg, f" → {nxt} 로 다시" if nxt else "",
                           extra={"component": "OCR"})
    return {"lines": [], "model": None, "errors": errors, "seconds": round(time.time() - t_all, 2),
            "cost_krw": round(cost, 3)}


# ---------------------------------------------------------------------------
# 쪽 그림 (스캔 쪽 OCR 용)
# ---------------------------------------------------------------------------
def pdf_page_jpeg(page) -> Tuple[bytes, int, int]:
    import fitz
    r = page.rect
    dpi = OCR_DPI
    long_in = max(r.width, r.height) / 72.0
    if long_in * dpi > OCR_MAX_SIDE:
        dpi = max(72, int(OCR_MAX_SIDE / long_in))
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    from PIL import Image
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue(), pix.width, pix.height


def image_jpeg(path: Path) -> Tuple[bytes, int, int, int, int]:
    """그림 계약서 → (jpeg, 보낸 폭, 보낸 높이, 원본 폭, 원본 높이). 미리보기처럼 EXIF 회전을 적용하지 않는다."""
    from PIL import Image
    with Image.open(path) as im:
        ow, oh = im.size
        im = im.convert("RGB")
        im.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        return buf.getvalue(), im.size[0], im.size[1], ow, oh


# ---------------------------------------------------------------------------
# 캐시
# ---------------------------------------------------------------------------
def _path_lock(p: Path) -> threading.Lock:
    with _LOCK:
        lk = _PATH_LOCKS.get(str(p))
        if lk is None:
            lk = _PATH_LOCKS[str(p)] = threading.Lock()
        return lk


def _sig(p: Optional[Path]) -> Optional[List[int]]:
    try:
        st = p.stat() if p else None
        return [st.st_size, st.st_mtime_ns] if st else None
    except OSError:
        return None


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{threading.get_ident()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def _targets_sig(targets: Sequence[Target]) -> str:
    blob = json.dumps([[t.target, t.kind, t.alias, t.skip_reason, [str(v) for v in t.values], t.primary]
                       for t in targets], ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------------
def _unavailable(reason: str, pages=None, not_found=None, targets=None, engine=None) -> Dict[str, Any]:
    return {"available": False, "reason": reason, "pages": pages or [], "marks": [], "not_found": not_found or [],
            "targets": targets or [],
            "engine": engine or {"ocr": "none", "model": None, "pages_ocr": 0, "seconds": 0.0, "cost_krw": 0.0,
                                 "cached": False}}


def locate(contract: Optional[Path], record: Dict[str, Any], decisions: Sequence[Dict[str, Any]],
           test: Optional[str], cache_dir: Optional[Path], stem: str, refresh: Any = None,
           max_pages: int = 40, key23_of=None, label_of=None) -> Dict[str, Any]:
    """계약서 + 세트 레코드 + 통합 결정 → 위치 표시 응답 (spec §3). 캐시: <cache_dir>/<stem>.json · <stem>.ocr.json."""
    t_start = time.time()
    refresh_s = str(refresh or "").strip().lower()
    refresh_marks = refresh_s in ("1", "true", "yes", "ocr", "all")
    refresh_ocr = refresh_s in ("ocr", "all")
    mode = ocr_mode()
    models = ocr_models()
    targets = build_targets(record, decisions, test, key23_of=key23_of, label_of=label_of)
    if contract is None:
        return _unavailable("계약서 파일이 없습니다")
    try:
        ok_file = contract.is_file()
    except OSError:
        ok_file = False
    if not ok_file:
        return _unavailable("계약서 원본 파일을 찾을 수 없습니다")
    ext = contract.suffix.lower()
    if ext in NO_PAGE_EXT:
        return _unavailable("DOCX·HWP 계약서는 쪽 미리보기가 없어 위치를 표시할 수 없습니다")
    if ext != ".pdf" and ext not in IMAGE_EXT:
        return _unavailable("쪽 미리보기를 만들 수 없는 형식이라 위치를 표시할 수 없습니다")

    key = {"v": LOCATE_VERSION, "contract": _sig(contract), "targets": _targets_sig(targets), "ocr": mode,
           "models": models if mode == "qwen" else []}
    cache_p = (cache_dir / f"{stem}.json") if cache_dir else None
    ocr_p = (cache_dir / f"{stem}.ocr.json") if cache_dir else None
    lock = _path_lock(cache_p) if cache_p else threading.Lock()
    with lock:
        if cache_p and not refresh_marks:
            c = _read_json(cache_p)
            if isinstance(c, dict) and c.get("_key") == key and not (
                    c.get("_ocr_failed") and time.time() - float(c.get("_at") or 0) > FAIL_RETRY_SEC):
                out = {k: v for k, v in c.items() if not k.startswith("_")}
                out.setdefault("engine", {})["cached"] = True
                return out
        out = _compute(contract, ext, targets, mode, models, ocr_p, refresh_ocr, max_pages, t_start)
        failed = bool(out.pop("_ocr_failed", False))
        if cache_p:
            try:
                _write_json(cache_p, {**out, "_key": key, "_at": time.time(), "_ocr_failed": failed})
            except OSError as e:
                logger.warning("원본 위치 캐시를 쓰지 못했습니다: %s", e, extra={"component": "측정"})
        return out


def _compute(contract: Path, ext: str, targets: List[Target], mode: str, models: List[str], ocr_p: Optional[Path],
             refresh_ocr: bool, max_pages: int, t_start: float) -> Dict[str, Any]:
    pages_info: List[Dict[str, Any]] = []
    streams: Dict[int, Tuple[Stream, str]] = {}
    scan_jobs: List[Tuple[int, Callable[[], Tuple[bytes, int, int]]]] = []
    doc = None
    try:
        if ext == ".pdf":
            import fitz
            try:
                doc = fitz.open(str(contract))
            except Exception as e:
                return _unavailable(f"계약서 PDF 를 열 수 없습니다 — {_err_text(e)}")
            n = min(doc.page_count, max_pages)
            for i in range(n):
                pg = doc[i]
                info = {"page": i + 1, "width": round(float(pg.rect.width), 2), "height": round(float(pg.rect.height), 2),
                        "source": SRC_NONE}
                if text_layer_ok(pg.get_text()):
                    streams[i + 1] = (text_layer_stream(pg), SRC_TEXT)
                    info["source"] = SRC_TEXT
                else:
                    scan_jobs.append((i + 1, (lambda p=pg: pdf_page_jpeg(p))))
                pages_info.append(info)
        else:
            try:
                from PIL import Image
                with Image.open(contract) as im:
                    ow, oh = im.size
            except Exception as e:
                return _unavailable(f"계약서 그림을 열 수 없습니다 — {_err_text(e)}")
            pages_info.append({"page": 1, "width": float(ow), "height": float(oh), "source": SRC_NONE})
            scan_jobs.append((1, lambda: image_jpeg(contract)[:3]))

        engine: Dict[str, Any] = {"ocr": "none", "model": None, "pages_ocr": 0, "seconds": 0.0, "cost_krw": 0.0,
                                  "cached": False}
        ocr_failed = False
        n_text = sum(1 for p in pages_info if p["source"] == SRC_TEXT)
        if scan_jobs:
            engine["ocr"] = mode
            if mode == "off":
                for p in pages_info:
                    if p["source"] == SRC_NONE:
                        p["note"] = "스캔 쪽 — 위치 찾기(OCR)가 꺼져 있습니다"
            else:
                res = _ocr_pages(contract, scan_jobs, models, ocr_p, refresh_ocr)
                used, secs, cost, fresh = [], 0.0, 0.0, 0
                for p in pages_info:
                    r = res.get(p["page"])
                    if r is None:
                        continue
                    if r.get("lines"):
                        streams[p["page"]] = (ocr_stream(r["lines"]), SRC_OCR)
                        p["source"] = SRC_OCR
                        p["model"] = r.get("model")
                        p["lines"] = len(r["lines"])
                        if r.get("model") and r["model"] not in used:
                            used.append(r["model"])
                    else:
                        ocr_failed = True
                        p["note"] = "스캔 쪽 — 위치 찾기(OCR) 실패" + (f": {r['errors'][-1]}" if r.get("errors") else "")
                    if not r.get("cached"):
                        fresh += 1
                        secs = max(secs, float(r.get("seconds") or 0.0))
                        cost += float(r.get("cost_krw") or 0.0)
                engine.update(model=", ".join(used) or None, pages_ocr=sum(1 for p in pages_info if p["source"] == SRC_OCR),
                              cost_krw=round(cost, 2), ocr_seconds=round(secs, 1), pages_ocr_fresh=fresh)
                if fresh:
                    logger.info("원본 위치 — 스캔 %d쪽 %s (₩%.1f · %.1fs)", fresh, ", ".join(used) or "실패", cost,
                                time.time() - t_start, extra={"component": "OCR"})
    finally:
        if doc is not None:
            doc.close()

    marks, not_found, summary = match_targets(targets, streams)
    engine["seconds"] = round(time.time() - t_start, 2)
    n_ocr = sum(1 for p in pages_info if p["source"] == SRC_OCR)
    parts = []
    if n_text:
        parts.append(f"텍스트 레이어 {n_text}쪽")
    if n_ocr:
        parts.append(f"스캔 {n_ocr}쪽(OCR 좌표)")
    n_none = sum(1 for p in pages_info if p["source"] == SRC_NONE)
    if n_none:
        parts.append(f"위치 정보 없음 {n_none}쪽")
    if streams:
        logger.info("원본 위치 — %s · 표시 %d곳 · 못 찾음 %d", " · ".join(parts) or "쪽 없음", len(marks), len(not_found),
                    extra={"component": "측정"})
    else:
        logger.info("원본 위치 — %s · 표시할 위치 없음", " · ".join(parts) or "쪽 없음", extra={"component": "측정"})
    if not streams:
        if scan_jobs and mode == "off":
            reason = "스캔 문서인데 위치 찾기(OCR)가 꺼져 있습니다 (CONSOLE_LOCATE_OCR=off)"
        elif scan_jobs:
            errs = [p.get("note") for p in pages_info if p.get("note")]
            reason = "스캔 쪽 위치 찾기(Qwen OCR)가 실패했습니다" + (f" — {errs[0].split(': ', 1)[-1]}" if errs else "")
        else:
            reason = "계약서에 쪽이 없습니다"
        out = _unavailable(reason, pages=pages_info, not_found=[], targets=summary, engine=engine)
        out["_ocr_failed"] = ocr_failed
        return out
    return {"available": True, "reason": None, "pages": pages_info, "marks": marks, "not_found": not_found,
            "targets": summary, "engine": engine, "_ocr_failed": ocr_failed}


def _ocr_pages(contract: Path, jobs: List[Tuple[int, Callable[[], Tuple[bytes, int, int]]]], models: List[str],
               ocr_p: Optional[Path], refresh_ocr: bool) -> Dict[int, Dict[str, Any]]:
    """스캔 쪽들을 병렬로 OCR (캐시에 있는 쪽은 다시 부르지 않는다) → {쪽: 결과}."""
    sig = _sig(contract)
    cache = _read_json(ocr_p) if (ocr_p and not refresh_ocr) else None
    if not (isinstance(cache, dict) and cache.get("v") == OCR_CACHE_VERSION and cache.get("contract") == sig):
        cache = {"v": OCR_CACHE_VERSION, "contract": sig, "dpi": OCR_DPI, "pages": {}}
    cached_pages = cache.get("pages") or {}
    out: Dict[int, Dict[str, Any]] = {}
    todo = []
    for pno, render in jobs:
        c = cached_pages.get(str(pno))
        if isinstance(c, dict) and c.get("lines"):
            out[pno] = {**c, "cached": True}
        else:
            todo.append((pno, render))
    if todo:
        # PyMuPDF 문서 객체는 스레드 사이에 나눠 쓰지 않는다 — 그림은 여기서 차례로 만들고, 모델 호출만 병렬로
        rendered: List[Tuple[int, Optional[Tuple[bytes, int, int]], Optional[str]]] = []
        for pno, render in todo:
            try:
                rendered.append((pno, render(), None))
            except Exception as e:
                msg = _err_text(e)
                logger.warning("원본 위치 — %d쪽 그림을 만들지 못했습니다: %s", pno, msg, extra={"component": "OCR"})
                rendered.append((pno, None, msg))

        def one(pno: int, data: Optional[Tuple[bytes, int, int]], err: Optional[str]) -> Dict[str, Any]:
            if data is None:
                return {"lines": [], "model": None, "errors": [err or "그림 없음"], "seconds": 0.0, "cost_krw": 0.0}
            img, w, h = data
            return ocr_page(img, "image/jpeg", w, h, pno, models)

        with ThreadPoolExecutor(max_workers=max(1, min(OCR_MAX_WORKERS, len(rendered))),
                                thread_name_prefix="locate-ocr") as ex:
            futs = [(pno, LC.submit(ex, one, pno, data, err)) for pno, data, err in rendered]
            for pno, f in futs:
                r = f.result()
                out[pno] = r
                if r.get("lines"):
                    cached_pages[str(pno)] = {k: r.get(k) for k in ("lines", "model", "tokens_in", "tokens_out",
                                                                     "seconds", "cost_krw")}
        cache["pages"] = cached_pages
        if ocr_p:
            try:
                _write_json(ocr_p, cache)
            except OSError as e:
                logger.warning("원본 위치 OCR 캐시를 쓰지 못했습니다: %s", e, extra={"component": "OCR"})
    return out


# ---------------------------------------------------------------------------
# 콘솔 실행(run) 에 붙이기
# ---------------------------------------------------------------------------
def locate_set(run, set_id: str, rec: Dict[str, Any], refresh: Any = None) -> Dict[str, Any]:
    """console_api 의 실행·세트 → locate(). 계약서 경로는 레코드의 _meta.files 에서만 (요청에서 받지 않는다)."""
    from . import console_api as CA
    meta = rec.get("_meta") or {}
    test = rec.get("test") or meta.get("test") or run.test
    contract = CA._source_file(run, rec, "contract")       # 원본이 없으면 실행 폴더의 사본
    decisions: List[Dict[str, Any]] = []
    try:
        con = CA._consolidation_payload(run, rec, test)
        for leg in con.get("legs") or []:
            if leg.get("leg") in ("contract", "context"):
                decisions.extend(d for d in (leg.get("decisions") or []) if isinstance(d, dict))
    except Exception as e:                   # 통합 결정을 못 읽어도 23키 위치는 낸다
        logger.warning("원본 위치 — 통합 결정을 읽지 못했습니다: %s", e, extra={"component": "측정"})
    if contract is None and (meta.get("files") or {}).get("contract"):
        out = _unavailable("계약서 원본 파일을 찾을 수 없습니다")
    else:
        out = locate(contract, rec, decisions, test, run.dir / "locate", CA._cache_name(set_id), refresh=refresh,
                     max_pages=CA.MAX_CONTRACT_PAGES, key23_of=CA.field_key23, label_of=CA.field_label_ko)
    return CA.scrub_json(out)
