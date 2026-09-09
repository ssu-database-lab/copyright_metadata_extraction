"""속성별 채점기 — 11속성은 정답 비교 방식이 근본적으로 다르다.

계획서 2-4 평가 대상:
    텍스트 속성 : 제목, 저자, 설명, 라이선스 유형, 키워드
    시각적 속성 : 해상도, 주요 색상, 개체 범주
    파일정보    : 파일크기, 파일포맷, 파일 생성 날짜

같은 '정확도'라도 해상도는 완전일치, 키워드는 집합 F1, 설명은 자유서술이라
비교 방법을 하나로 쓸 수 없다. 여기서 속성마다 비교기를 지정한다.

⚠️ 설명(description) 채점 방식은 **미확정 사항**이다.
   정답 "카메라 묘기를 가장 먼저 사용한 영화…" vs 출력 "흑백 무성영화로 처형 장면이…"
   는 둘 다 맞지만 문자열 일치는 0이다. 현재 기본값은 `content_recall`
   (정답의 내용어가 출력에 얼마나 등장하는가)이며, 임계값으로 판정한다.
   재현 가능하고 감사 가능하지만 의미 동치를 완벽히 잡지는 못한다.
   시험기관과 합의되면 --desc-method 로 교체한다(embedding / llm_judge / item_hit).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 정규화
# ---------------------------------------------------------------------------
# 사이트 값에 섞이는 양방향 제어문자(U+202A~U+202E) — 눈에 안 보여 비교가 조용히 실패한다
_BIDI = dict.fromkeys(range(0x202A, 0x202F), None)
_PUNCT = re.compile(r"[\s\-_·,./\\()\[\]{}'\"“”‘’«»…:;!?]+")
# 한국어 조사/기능어 — 내용어 비교에서 제외
_STOP = {"이", "그", "저", "것", "수", "등", "및", "또는", "있는", "있다", "하는", "한다",
         "되는", "된다", "에서", "으로", "에게", "까지", "부터", "이다", "합니다", "그리고",
         "the", "a", "an", "of", "in", "on", "is", "are", "and", "or", "to", "for"}


def norm(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, (list, tuple)):
        s = " ".join(str(x) for x in s)
    return unicodedata.normalize("NFKC", str(s)).translate(_BIDI).strip()


def norm_key(s: Any) -> str:
    """비교용 강정규화 — 공백·구두점 제거 + 소문자."""
    return _PUNCT.sub("", norm(s)).lower()


def tokens(s: Any) -> List[str]:
    """내용어 토큰 — 조사/기능어와 1글자 토큰 제외."""
    raw = _PUNCT.sub(" ", norm(s)).lower().split()
    return [t for t in raw if len(t) > 1 and t not in _STOP]


def as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v if norm(x)]
    txt = norm(v)
    if not txt:
        return []
    return [p.strip() for p in re.split(r"[,;|/]", txt) if p.strip()]


# ---------------------------------------------------------------------------
# 비교기 — 각각 (match: bool, detail: dict) 반환
# ---------------------------------------------------------------------------
def cmp_norm_exact(gt, got) -> Tuple[bool, Dict]:
    g, o = norm_key(gt), norm_key(got)
    return (bool(g) and g == o), {"method": "norm_exact"}


def cmp_contains(gt, got) -> Tuple[bool, Dict]:
    """정답이 출력에 포함되거나 그 반대 — 제목처럼 접두/접미가 붙는 경우."""
    g, o = norm_key(gt), norm_key(got)
    if not g or not o:
        return False, {"method": "contains"}
    return (g in o or o in g), {"method": "contains"}


def cmp_numeric(gt, got) -> Tuple[bool, Dict]:
    try:
        return int(gt) == int(got), {"method": "numeric_exact"}
    except (TypeError, ValueError):
        return False, {"method": "numeric_exact"}


def cmp_resolution(gt, got) -> Tuple[bool, Dict]:
    """'1757 x 1172' / '600 * 498' / '1757x1172' 를 모두 같은 값으로 본다."""
    def parse(v):
        m = re.match(r"^(\d+)\s*[xX×*]\s*(\d+)$", norm(v))
        return (int(m.group(1)), int(m.group(2))) if m else None
    a, b = parse(gt), parse(got)
    return (a is not None and a == b), {"method": "resolution"}


def cmp_date(gt, got, year_only_ok: bool = True) -> Tuple[bool, Dict]:
    """YYYY-MM-DD 완전일치. 정답이 연도만이면(예 '1895') 연도 일치로 인정."""
    g, o = norm(gt), norm(got)
    if not g or not o:
        return False, {"method": "date"}
    if g == o:
        return True, {"method": "date"}
    gy = re.match(r"^(\d{4})", g)
    oy = re.match(r"^(\d{4})", o)
    if year_only_ok and gy and oy and len(g) == 4:
        return gy.group(1) == oy.group(1), {"method": "date(year)"}
    return False, {"method": "date"}


def cmp_set_recall(gt, got, threshold: float = 0.34) -> Tuple[bool, Dict]:
    """키워드 — 정답 용어가 출력에 얼마나 담겼는가(recall).

    ⚠️ F1을 쓰면 안 된다. 정답(공유마당 3단 분류 '어문' 1개)과 출력(VLM 자유 키워드 7개)은
       개수 자체가 다르다. 정답 1개를 출력이 **정확히 맞혀도** F1은
       2·(1/7)·1 / (1/7+1) = 0.25 로 임계값 아래가 되어 오답 처리된다.
       정확히 맞힌 답을 틀렸다고 하는 채점기는 쓸 수 없다.

       키워드를 더 많이 내는 것은 감점 사유가 아니므로 정밀도를 빼고 재현율만 본다.
       부분 문자열도 인정한다 — 정답 '회화'와 출력 '회화작품'은 같은 개념이다.
    """
    G = [norm_key(x) for x in as_list(gt) if norm_key(x)]
    O = [norm_key(x) for x in as_list(got) if norm_key(x)]
    if not G:
        return False, {"method": "set_recall", "recall": None}
    blob = " ".join(O)
    hit = [g for g in G if g in O or (len(g) > 1 and g in blob)]
    rec = len(hit) / len(G)
    return rec >= threshold, {"method": "set_recall", "recall": round(rec, 3),
                              "hit": hit[:6], "gt_n": len(G), "got_n": len(O)}


# 라이선스 표기 정규화 — 부분일치로 채점하면 CC BY 가 CC BY-NC 를 맞다고 한다
_LICENSE_CANON = [
    (re.compile(r"ccby[-_ ]?ncnd|ccbyncnd"), "CC BY-NC-ND"),
    (re.compile(r"ccby[-_ ]?ncsa|ccbyncsa"), "CC BY-NC-SA"),
    (re.compile(r"ccby[-_ ]?nc(?!nd|sa)"), "CC BY-NC"),
    (re.compile(r"ccby[-_ ]?nd"), "CC BY-ND"),
    (re.compile(r"ccby[-_ ]?sa"), "CC BY-SA"),
    (re.compile(r"ccby(?![-_ ]?(nc|nd|sa))"), "CC BY"),
    (re.compile(r"(공공누리)?제?1유형|kogl.?1|출처표시(?!.*(상업|변경))"), "KOGL 제1유형"),
    (re.compile(r"(공공누리)?제?2유형|kogl.?2"), "KOGL 제2유형"),
    (re.compile(r"(공공누리)?제?3유형|kogl.?3"), "KOGL 제3유형"),
    (re.compile(r"(공공누리)?제?4유형|kogl.?4"), "KOGL 제4유형"),
    (re.compile(r"자유이용만료|만료저작물|보호기간만료"), "만료저작물"),
    (re.compile(r"기증저작물자유이용|기증.*자유이용"), "기증-자유이용"),
    (re.compile(r"기증저작물이용허락|기증.*이용허락"), "기증-이용허락"),
]


_TRUE = {"v", "V", "■", "☑", "y", "yes", "true", "허락", "o", "O", "1"}
_FALSE = {"□", "n", "no", "false", "미허락", "x", "X", "0"}


def cmp_bool(gt, got) -> Tuple[bool, Dict]:
    """계약서 제2조의 권리 체크박스. 표기가 v / ■ / □ 로 섞여 있다."""
    def norm(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        t = str(v).strip()
        if t in _TRUE:
            return True
        if t in _FALSE:
            return False
        return None
    g, o = norm(gt), norm(got)
    return (g is not None and g == o), {"method": "bool", "gt": g, "got": o}


def cmp_license(gt, got) -> Tuple[bool, Dict]:
    """라이선스 — 표기 변형은 흡수하되 **다른 라이선스는 반드시 구분**한다.

    부분 문자열 비교는 여기서 치명적이다: 'CC BY' ⊂ 'CC BY-NC' 이므로
    비상업 전용 저작물을 상업 가능으로 잘못 추출해도 정답 처리된다.
    실제 권리 판단이 뒤집히는 오류라 반드시 정규형끼리 완전일치로 본다.
    """
    def canon(v):
        k = norm_key(v)
        if not k:
            return None
        for pat, name in _LICENSE_CANON:
            if pat.search(k):
                return name
        return k
    a, b = canon(gt), canon(got)
    return (a is not None and a == b), {"method": "license", "gt_canon": a, "got_canon": b}


def cmp_content_recall(gt, got, threshold: float = 0.30) -> Tuple[bool, Dict]:
    """설명 — 정답의 내용어가 출력에 얼마나 재현되는가(recall).

    정밀도가 아니라 재현율을 쓰는 이유: 출력이 정답보다 길고 상세한 것은
    감점 사유가 아니다. 정답이 말한 핵심을 담았는지가 관심사다.

    ⚠️ 토큰 완전일치를 쓰면 한국어에서 조사 때문에 무너진다. 실측 사례:
         정답 "…천사의 날개와 왕관 그리고 부를 상징하는 금목걸이를…"
         출력 "…커다란 날개가 … 왕관이 배치되어 … 금색 체인 목걸이가…"
       날개·왕관·목걸이가 다 들어 있는데 '날개와'≠'날개가' 라 recall 0.0 이 나왔다.
       형태소 분석기 의존성을 추가하지 않고, **2자 이상 어간이 상대 문자열에
       포함되면 적중**으로 본다(긴 어간부터 시도). 위 사례는 0.0 → 0.33 이 된다.
    """
    G = tokens(gt)
    if not G:
        return False, {"method": "content_recall", "recall": None}
    blob = "".join(tokens(got))
    hit = []
    for g in G:
        for n in range(len(g), 1, -1):
            if g[:n] in blob:
                hit.append(g)
                break
    rec = len(hit) / len(G)
    return rec >= threshold, {"method": "content_recall(stem)", "recall": round(rec, 3),
                              "hit": hit[:8], "gt_n": len(G)}


# ---------------------------------------------------------------------------
# 속성 정의 — 계획서 11속성
# ---------------------------------------------------------------------------
class Attr:
    """field 는 문자열 하나 또는 후보 튜플이다.

    튜플인 경우 앞에서부터 값이 있는 첫 필드를 쓴다. 스키마를 TTA 표준 이름으로
    옮기는 동안 신·구 이름이 공존하기 때문이다 (예: author 가 비어 있으면
    예전 결과의 copyright_holder 로 떨어진다). 이름을 한 번에 갈아끼우면
    기존 결과가 전부 오답 처리된다 — 실측으로 채점이 0.48 → 0.00 이 된다.
    """

    def __init__(self, name: str, field, cmp: Callable, group: str,
                 visual_only: bool = False):
        self.name, self.field, self.cmp = name, field, cmp
        self.group, self.visual_only = group, visual_only

    def pick(self, extracted: Dict[str, Any]) -> Any:
        e = extracted or {}
        fields = self.field if isinstance(self.field, tuple) else (self.field,)
        for f in fields:
            v = e.get(f)
            if v not in (None, "", [], {}):
                return v
        return e.get(fields[0])


ATTRIBUTES: List[Attr] = [
    Attr("제목",          "work_title",        cmp_contains,       "텍스트"),
    # 계약서 제2조가 인쇄하는 것은 저작자다. copyright_holder(저작권자)는 계약서에
    # 인쇄되지 않으며 저작자와 다른 값인 경우 0/57 로 전혀 맞지 않는다.
    Attr("저자",          ("author", "copyright_holder"), cmp_contains, "텍스트"),
    Attr("설명",          "description",       cmp_content_recall, "텍스트"),
    Attr("라이선스 유형",   "kogl_type",         cmp_license,        "텍스트"),
    Attr("키워드",         "keyword",           cmp_set_recall,     "텍스트"),
    Attr("해상도",         "resolution",        cmp_resolution,     "시각", visual_only=True),
    Attr("주요 색상",      "dominant_colors",   cmp_set_recall,     "시각", visual_only=True),
    Attr("개체 범주",      "main_subjects",     cmp_set_recall,     "시각", visual_only=True),
    Attr("파일크기",       "file_size",         cmp_numeric,        "파일"),
    Attr("파일포맷",       ("file_format", "digital_format"), cmp_norm_exact, "파일"),
    Attr("파일 생성 날짜",  "file_created_date", cmp_date,           "파일"),
]
ATTR_BY_NAME = {a.name: a for a in ATTRIBUTES}


# ---------------------------------------------------------------------------
# TTA 표준 채점 범위 — 계약서에서 실제로 읽어낼 수 있는 항목만
#
# 근거: 생성계약서 전수/표본 감사 결과.
#   · 계약서 5,716건은 전부 같은 5쪽 "저작재산권 이용허락 계약서" 서식이다.
#   · 권리자·저작자·종별·권리 체크박스·이용허락기간은 300/300 인쇄된다.
#   · 파일명은 0/300, 라이선스(공공누리·CC BY·기증·만료)는 0/300 인쇄되지 않는다.
#   · 저작권자명은 저작자·권리자와 다른 값일 때 0/57 로 전혀 인쇄되지 않는다
#     (81% 일치는 세 값이 우연히 같은 경우일 뿐이다).
# 명세서의 "실제 데이터 사용 여부" 열은 양방향으로 틀렸다 — file_name 은 O 인데
# 인쇄되지 않고, R_4~R_7·이용허락기간은 X 인데 300/300 인쇄된다. 실측을 따른다.
# ---------------------------------------------------------------------------
TTA_ATTRIBUTES: List[Attr] = [
    # --- 식별/유형 ---
    Attr("저작물명",            "work_title",                     cmp_contains,   "식별"),
    Attr("저작물 유형",          "work_type",                      cmp_norm_exact, "식별"),
    # --- 권리주체 (4.1/4.3/4.4) ---
    Attr("저작자",              ("author", "copyright_holder"),   cmp_contains,   "권리주체"),
    Attr("저작재산권자",         ("economic_rights_holder", "copyright_holder"), cmp_contains, "권리주체"),
    Attr("이용허락자",           ("licensor", "copyright_holder"), cmp_contains,   "권리주체"),
    # --- 저작재산권 세부 권리 (7.1~7.7) — 계약서 제2조 체크박스 ---
    Attr("복제권",              "reproduction_right",             cmp_bool,       "세부권리"),
    Attr("공연권",              "public_performance_right",       cmp_bool,       "세부권리"),
    Attr("공중송신권",           "public_transmission_right",      cmp_bool,       "세부권리"),
    Attr("전시권",              "exhibition_right",               cmp_bool,       "세부권리"),
    Attr("배포권",              "distribution_right",             cmp_bool,       "세부권리"),
    Attr("대여권",              "rental_right",                   cmp_bool,       "세부권리"),
    Attr("2차적저작물작성권",      "derivative_work_creation_right", cmp_bool,       "세부권리"),
    # --- 유효기간 (8.2/8.3) ---
    Attr("이용허락 시작일",       "license_start_date",             cmp_date,       "유효기간"),
    Attr("이용허락 종료일",       "license_end_date",               cmp_date,       "유효기간"),
]
TTA_ATTR_BY_NAME = {a.name: a for a in TTA_ATTRIBUTES}

# 계약서에 근거가 없어 채점에서 제외하는 항목과 그 사유.
# 리포트에 "미채점"으로 찍어야 하며, 0점으로 집계하면 안 된다.
TTA_EXCLUDED = {
    "file_name":        "계약서에 인쇄되지 않음 (0/300)",
    "copyright_holder": "저작자·권리자와 다를 때 인쇄되지 않음 (0/57)",
    "public_release_type": "라이선스 표기가 계약서에 없음 (0/300) — 저작물/카탈로그 경로에서 채점",
    "work_identifier":  "계약서 본문에 저작물 ID 없음 (0/5,714)",
}



def score_set(gt_attributes: Dict[str, Dict], extracted: Dict[str, Any],
              media: str = "image",
              overrides: Optional[Dict[str, Callable]] = None,
              skip_tiers: tuple = ("D",),
              attributes: Optional[List[Attr]] = None) -> Dict[str, Any]:
    """정답 레코드의 attributes 와 파이프라인 출력을 대조한다.

    채점에서 빠지는 경우 — 값을 지어내지 않고 사유를 남긴다:
      skipped_no_gt   : 정답 자체가 없음(주요 색상·개체 범주 등)
      not_applicable  : 어문 저작물의 시각적 속성(tier N/A)
      skipped_tier    : 정답과 추출값이 서로 다른 양이라 대조가 성립하지 않음

    `skip_tiers` 기본값 ("D",) 는 파일 생성 날짜를 제외한다. 정답은 저작물 창작 시점
    (창작년도 1895 등)이고 추출값은 파일 생성 시각이라, 같은 값으로 볼 근거가 없다.
    이를 채점하면 모든 세트에서 구조적으로 틀리게 되어 정확도가 왜곡된다.

    `attributes` 로 채점 기준을 바꿀 수 있다. 기본값은 계획서 11속성(ATTRIBUTES),
    TTA 표준 채점에는 TTA_ATTRIBUTES 를 넘긴다.

    Returns:
        {"per_attr": {...}, "n_scored": int, "n_match": int, "accuracy": float|None}
    """
    overrides = overrides or {}
    # 기본은 연구개발계획서 11속성. TTA 채점은 TTA_ATTRIBUTES 를 넘겨서 쓴다 —
    # 두 기준은 대상도 정답 출처도 달라서 한 리스트로 합칠 수 없다.
    attrs = attributes if attributes is not None else ATTRIBUTES
    per: Dict[str, Any] = {}
    n_scored = n_match = 0

    for a in attrs:
        gt_entry = (gt_attributes or {}).get(a.name) or {}
        tier = gt_entry.get("tier")
        gt_val = gt_entry.get("value")
        got_val = a.pick(extracted)

        if tier == "N/A" or (a.visual_only and media == "text"):
            per[a.name] = {"status": "not_applicable", "tier": tier}
            continue
        if tier in skip_tiers:
            per[a.name] = {"status": "skipped_tier", "tier": tier,
                           "reason": gt_entry.get("note") or "정답과 추출값의 정의가 다름"}
            continue
        if gt_val in (None, "", []):
            per[a.name] = {"status": "skipped_no_gt", "tier": tier,
                           "reason": gt_entry.get("note")}
            continue

        fn = overrides.get(a.name, a.cmp)
        match, detail = fn(gt_val, got_val)
        n_scored += 1
        n_match += bool(match)
        # 컬렉션 공통 태그는 배치를 설명할 뿐 개별 저작물 화면 내용이 아니다.
        # 채점에는 넣되 표시해 두어, 리포트에서 항목별/컬렉션별을 분리해 볼 수 있게 한다.
        if gt_entry.get("batch_level"):
            detail = {**detail, "batch_level": True,
                      "batch_size": gt_entry.get("batch_size")}
        per[a.name] = {"status": "scored", "match": bool(match), "tier": tier,
                       "gt": gt_val if not isinstance(gt_val, list) else gt_val[:5],
                       "got": got_val if not isinstance(got_val, list) else got_val[:5],
                       **detail}

    return {"per_attr": per, "n_scored": n_scored, "n_match": n_match,
            "accuracy": (n_match / n_scored) if n_scored else None}
