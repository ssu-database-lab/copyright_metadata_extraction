"""평면 추출 결과 → v4 시험평가용 23키 예측 레코드 (docs/v4/M2_runner_spec.md A2, M1 spec §1·§6).

tta_serializer 와 같은 자리에서 같은 입력(평면 metadata + 파이프라인 응답)을 받아, 평가기
(api/module/evaluation/v4)가 읽는 23개 키를 keys23.KEYS 순서로 낸다. 추출은 평면 그대로 두고
출력 직전에 한 번 바꾸는 이유는 tta_serializer 모듈 설명과 같다.

규칙 요약 (키 번호는 keys23):
  1  work_title       ← response["contract_context_rules"]["title"](규칙 파싱, 문서 경로, 관리번호 이미 제거)이 비어 있지
                        않으면 **그것**, 아니면 모델의 work_title(앞의 관리번호 토큰 ^\\d{5,8}\\s+ 제거, §12.7(c)).
                        출처는 _meta.work_title_source('rules'|'model'|'contract_context'|None), 모델 값은 _meta.model_work_title.
                        근거: T-001 개발 런에서 LLM 은 공백을 흔들었고('구축 전략'→'구축전략', '세상으로(희망가)'→'세상으로 (희망가)')
                        규칙 파서는 5/5 정확했다 (M2b 수정 2).
  2  file_name        ← response["file_name"] (setup 이 정제 전에 잡아 둔 원본 이름)
  3  work_type        ← 같은 우선순위(규칙 파싱 종별 → 모델 work_type) → 닫힌 집합(어문/음악/…/기타, 끝의 '저작물' 제거);
                        _meta.work_type_source · _meta.model_work_type
  4  creator          ← author (목록; `(저작물 N 건)` 주석 제거, 개행·쉼표로 분리). 다른 필드로 대체하지 않는다.
  5  copyright_holder ← **이용자 당사자**: metadata["licensee_party"](본문 규칙, party_rules) → licensee → user.
                        모델의 copyright_holder 는 쓰지 않고 _meta.model_copyright_holder 에만 남긴다.
  6~12 권리 7종       ← 개별 필드(bool/문자) 또는 granted_rights(tta_serializer._right_from_granted 재사용) → 허용/불허/null
  13 other_rights     ← 항상 null (모든 계약서가 상태 ③; 모델 값은 _meta.model_other_rights)
  14/15 시작·종료일   ← license_start_date / license_end_date (대체: effective_date / expiration_date) → YYYY-MM-DD
  16 contract_validity_period ← 14·15 가 모두 있으면 "시작~종료", 아니면 valid_period / contract_duration 을 정규화
  17 keywords         ← keyword (문자열이면 , | ; 개행으로 분리)
  18 hashtags         ← hashtags/hashtag 가 있으면 그것('#' 보장), 없으면 ["#"+k for k in keywords]
  19 file_format      ← digital_format (대문자, 점 없이; 없으면 file_name 의 확장자)
  20 file_size        ← file_size (정수 바이트; 없으면 response["file_size_bytes"])
  21 resolution       ← resolution → "WxH"
  22 description      ← description
  23 file_hash        ← response["file_hash"] (setup 의 SHA-256)
  test="T-001" → 17~23 을 null 로. test="T-002" → 1·3 이 비면 response["contract_context"](규칙 파싱) 값을 넣는다.
해석할 수 없는 값은 가능하면 원문 그대로 내보내고(평가기가 경고로 잡는다) _meta.unparsed 에 적는다.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# granted_rights 해석은 tta_serializer 의 것을 재사용한다 — 세 번째 사본을 만들지 않는다 (A2).
# llm_extraction 패키지의 __init__ 이 torch 를 올리기 때문에 이 파일을 경로로 직접 적재하는 호출자
# (단위 테스트)도 있다. 그때는 상대 import 가 실패하므로 옆 파일을 경로로 읽는다
# (clip_extraction/vlm/extractor.py 의 _load_sibling 과 같은 방식).
try:
    from .tta_serializer import _right_from_granted
except ImportError:
    def _load_sibling(mod_name: str, filename: str):
        for key in (f"module.llm_extraction.schemas.{mod_name}",
                    f"api.module.llm_extraction.schemas.{mod_name}"):
            if key in sys.modules:
                return sys.modules[key]
        spec = importlib.util.spec_from_file_location(f"_eval23_{mod_name}",
                                                      Path(__file__).with_name(filename))
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"cannot load sibling {filename}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    _right_from_granted = _load_sibling("tta_serializer", "tta_serializer.py")._right_from_granted

# 23개 키와 정규화 규칙은 평가기(v4)가 단일 출처다. 파이프라인은 평면 `module.*` 로, 테스트·CLI 는
# `api.module.*` 로 import 하므로 둘 다 받는다 (같은 파일이다).
try:
    from module.evaluation.v4 import keys23 as _keys23
    from module.evaluation.v4 import normalize as _norm
except ImportError:
    from api.module.evaluation.v4 import keys23 as _keys23      # type: ignore[no-redef]
    from api.module.evaluation.v4 import normalize as _norm     # type: ignore[no-redef]

KEY_NAMES: Tuple[str, ...] = _keys23.KEY_NAMES
T002_KEYS: Tuple[str, ...] = _keys23.T002_KEYS
TESTS = ("T-001", "T-002")

# 23키 → 평면 스키마 필드 (권리 7종). 이름이 다른 셋이 '재명명' 항목이다.
RIGHTS_MAP: Dict[str, str] = {
    "reproduction_right": "reproduction_right",
    "performance_right": "public_performance_right",
    "public_transmission_right": "public_transmission_right",
    "exhibition_right": "exhibition_right",
    "distribution_right": "distribution_right",
    "rental_right": "rental_right",
    "derivative_work_right": "derivative_work_creation_right",
}
# 모델이 권리 값으로 내는 표기들 → 허용/불허. 체크 글리프(v/■/□)는 평가기가 받지 않으므로 여기서 바꿔 준다.
_ALLOW = {"허용", "허락", "허가", "가능", "동의", "포함", "o", "y", "yes", "true", "1", "v", "■", "☑", "✓", "✔"}
_DENY = {"불허", "미허락", "미허용", "불가", "불허용", "없음", "미포함", "x", "n", "no", "false", "0", "□", "◻"}
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")
_NAME_ANNOT = re.compile(r"\(\s*(?:저작물)?\s*\d+\s*건\s*\)")
_NAME_SPLIT = re.compile(r"[\n,;/·ㆍ•]")
_TAG_SPLIT = re.compile(r"[,|;\n、]+")
_DATE_TAIL = re.compile(r"\s*(?:부터|까지|이후|이전|경)?\s*(?:\([^)]*\))?\s*$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ID_ONLY = re.compile(r"^\d{5,8}$")


# ----------------------------------------------------------------------------- 작은 도우미
def _blank(v: Any) -> bool:
    return v is None or v == "" or v == [] or v == {}


def _scalar(v: Any) -> Any:
    """단일값 키에 리스트가 오면 첫 비어 있지 않은 원소."""
    if isinstance(v, (list, tuple)):
        for x in v:
            if not _blank(x):
                return x
        return None
    return v


def _title(v: Any) -> Optional[str]:
    t = _norm.norm_text(_scalar(v))
    if t is None or _ID_ONLY.match(t):
        return t
    return _norm.strip_id_prefix(t)


def _work_type(v: Any) -> Tuple[Optional[str], Any]:
    t = _norm.norm_text(_scalar(v))
    if t is None:
        return None, None
    try:
        return _norm.norm_work_type(t), None
    except ValueError:
        # 닫힌 집합 밖('텍스트', '사진·영상' 등) — 끝의 '저작물' 만 떼고 원문을 남긴다
        return (re.sub(r"저작물$", "", t.replace(" ", "")) or t), t


def _names(v: Any) -> Optional[List[str]]:
    """문자열/목록/dict → 이름 목록. `(저작물 N 건)` 주석 제거, 개행·쉼표·가운뎃점으로 분리, 중복 제거."""
    items = list(v) if isinstance(v, (list, tuple)) else [v]
    out: List[str] = []
    for it in items:
        if _blank(it):
            continue
        if isinstance(it, dict):
            it = it.get("name") or it.get("이름") or it.get("성명") or ""
        for piece in _NAME_SPLIT.split(str(it)):
            piece = " ".join(_NAME_ANNOT.sub("", piece).split())
            if piece and piece not in out:
                out.append(piece)
    return out or None


def _rights_scalar(raw: Any) -> Tuple[Optional[str], Any]:
    if _blank(raw):
        return None, None
    if isinstance(raw, bool):
        return ("허용" if raw else "불허"), None
    if isinstance(raw, (int, float)):
        return ("허용" if raw else "불허"), None
    if isinstance(raw, dict):               # {"granted": true} / {"허락": "O"}
        inner = raw.get("granted", raw.get("허락", raw.get("value")))
        return _rights_scalar(inner)
    s = _TRAILING_PAREN.sub("", str(_scalar(raw))).strip().casefold()
    if s in _ALLOW:
        return "허용", None
    if s in _DENY:
        return "불허", None
    return None, raw                        # 해석 불가 → null, 원문은 _meta.unparsed 로


def _rights(md: Dict[str, Any], flat: str) -> Tuple[Optional[str], Any]:
    raw = md.get(flat)
    if _blank(raw):
        raw = _right_from_granted(md, flat)
    return _rights_scalar(raw)


def _date(v: Any) -> Tuple[Optional[str], Any]:
    """YYYY-MM-DD 로. '2009년 09월 27일부터' 같은 꼬리는 떼고 다시 시도; 그래도 안 되면 원문을 낸다."""
    if _blank(v):
        return None, None
    try:
        return _norm.norm_date(v), None
    except ValueError:
        pass
    t = _norm.norm_text(_scalar(v)) or ""
    try:
        return _norm.norm_date(_DATE_TAIL.sub("", t)), None
    except ValueError:
        return (t or None), v


def _period(md: Dict[str, Any], start: Optional[str], end: Optional[str]) -> Tuple[Optional[str], Any]:
    """14·15 가 모두 날짜면 '시작~종료'(제3조 정의와 일치), 아니면 valid_period/contract_duration 을 정규화."""
    if start and end and _ISO_DATE.match(start) and _ISO_DATE.match(end):
        return f"{start}~{end}", None
    for key in ("valid_period", "contract_duration"):
        v = md.get(key)
        if _blank(v):
            continue
        try:
            return _norm.norm_period(v), None
        except ValueError:
            return _norm.norm_text(_scalar(v)), v
    return None, None


def _tags(v: Any) -> Optional[List[str]]:
    if _blank(v):
        return None
    items = list(v) if isinstance(v, (list, tuple)) else _TAG_SPLIT.split(str(v))
    out: List[str] = []
    for it in items:
        if _blank(it):
            continue
        s = " ".join(str(it).split())
        if s and s not in out:
            out.append(s)
    return out or None


def _hashtags(md: Dict[str, Any], keywords: Optional[List[str]]) -> Tuple[Optional[List[str]], Optional[str]]:
    raw = md.get("hashtags")
    if _blank(raw):
        raw = md.get("hashtag")
    tags = _tags(raw)
    if tags:
        return [t if t.startswith("#") else "#" + t for t in tags], "model"
    if keywords:
        return ["#" + k for k in keywords], "keywords"
    return None, None


def _format(md: Dict[str, Any], file_name: Optional[str]) -> Optional[str]:
    v = _scalar(md.get("digital_format"))
    if not _blank(v):
        t = str(v).strip().lstrip(".").upper()
        if t:
            return t
    ext = os.path.splitext(str(file_name or ""))[1].lstrip(".").upper()
    return ext or None


def _int_bytes(v: Any) -> Optional[int]:
    if _blank(v) or isinstance(v, bool):
        return None
    try:
        return _norm.norm_int_bytes(v)
    except ValueError:
        return None


def _resolution(v: Any) -> Optional[str]:
    if _blank(v):
        return None
    try:
        return _norm.norm_resolution(v)
    except ValueError:
        return _norm.norm_text(_scalar(v))


def _licensee(md: Dict[str, Any]) -> Tuple[Optional[List[str]], Optional[str]]:
    """키 5 — 이용자 당사자. 본문 규칙(licensee_party) → 모델의 licensee → user 순."""
    for key in ("licensee_party", "licensee", "user"):
        v = md.get(key)
        if isinstance(v, dict):
            v = v.get("name") or v.get("이름") or v.get("성명")
        names = _names(v)
        if names:
            return names, key
    return None, None


# ----------------------------------------------------------------------------- 본체
def build(metadata: Optional[Dict[str, Any]],
          response: Optional[Dict[str, Any]] = None,
          test: Optional[str] = None) -> Dict[str, Any]:
    """평면 metadata(+파이프라인 응답) → 23키(keys23 순서) + `_meta`.

    metadata : 최종 평면 dict (consolidated_metadata 에 원본에서 되살린 필드를 합친 것)
    response : llm_metadata.json 전체 — file_name / file_hash / file_size_bytes / contract_context 를 여기서 읽는다
    test     : "T-001" (17~23 을 null) | "T-002" (1·3 을 계약서 문맥으로 보충) | None (모두 그대로)
    """
    if test is not None and test not in TESTS:
        raise ValueError(f"test 는 {TESTS} 또는 None 이어야 합니다: {test!r}")
    md: Dict[str, Any] = dict(metadata or {})
    resp: Dict[str, Any] = response or {}
    out: Dict[str, Any] = {k: None for k in KEY_NAMES}
    unparsed: Dict[str, Any] = {}
    meta: Dict[str, Any] = {"serializer": "eval23", "test": test}

    # 1~3 식별 — 제목·종별은 규칙 파싱(contract_context_rules: 문서 경로, 모델 호출 없음)이 있으면 모델 값보다 우선한다.
    # 모델 값은 비교용으로 _meta.model_work_title / model_work_type 에 남긴다 (M2b 수정 2).
    rules_raw = resp.get("contract_context_rules")
    rules: Dict[str, Any] = rules_raw if isinstance(rules_raw, dict) else {}
    model_title = _title(md.get("work_title"))
    rule_title = _title(rules.get("title"))          # 이미 관리번호가 떼어진 값 — 같은 정규화를 한 번 더 거쳐도 무해하다
    if rule_title:
        out["work_title"], meta["work_title_source"] = rule_title, "rules"
    else:
        out["work_title"], meta["work_title_source"] = model_title, ("model" if model_title else None)
    meta["model_work_title"] = md.get("work_title")
    out["file_name"] = resp.get("file_name") or resp.get("filename")
    rule_type, _ = _work_type(rules.get("work_type"))
    model_type, bad = _work_type(md.get("work_type"))
    if rule_type:
        out["work_type"], meta["work_type_source"] = rule_type, "rules"
    else:
        out["work_type"], meta["work_type_source"] = model_type, ("model" if model_type else None)
        if bad is not None:                          # 출력에 실린 값이 닫힌 집합 밖일 때만 unparsed 에 적는다
            unparsed["work_type"] = bad
    meta["model_work_type"] = md.get("work_type")

    # 4~5 권리주체
    out["creator"] = _names(md.get("author"))
    out["copyright_holder"], meta["copyright_holder_source"] = _licensee(md)
    meta["model_copyright_holder"] = md.get("copyright_holder")

    # 6~13 권리
    for key23, flat in RIGHTS_MAP.items():
        out[key23], bad = _rights(md, flat)
        if bad is not None:
            unparsed[key23] = bad
    out["other_rights"] = None
    if not _blank(md.get("other_rights")):
        meta["model_other_rights"] = md.get("other_rights")

    # 14~16 기간
    start_raw = md.get("license_start_date") if not _blank(md.get("license_start_date")) else md.get("effective_date")
    end_raw = md.get("license_end_date") if not _blank(md.get("license_end_date")) else md.get("expiration_date")
    out["license_start_date"], bad = _date(start_raw)
    if bad is not None:
        unparsed["license_start_date"] = bad
    out["license_end_date"], bad = _date(end_raw)
    if bad is not None:
        unparsed["license_end_date"] = bad
    out["contract_validity_period"], bad = _period(md, out["license_start_date"], out["license_end_date"])
    if bad is not None:
        unparsed["contract_validity_period"] = bad
    if not _blank(md.get("valid_period")):
        meta["model_valid_period"] = md.get("valid_period")

    # 17~23 내용·파일
    keywords = _tags(md.get("keyword") if not _blank(md.get("keyword")) else md.get("keywords"))
    out["keywords"] = keywords
    out["hashtags"], meta["hashtags_source"] = _hashtags(md, keywords)
    out["file_format"] = _format(md, out["file_name"])
    out["file_size"] = _int_bytes(md.get("file_size"))
    if out["file_size"] is None:
        out["file_size"] = _int_bytes(resp.get("file_size_bytes"))
    out["resolution"] = _resolution(md.get("resolution"))
    desc = md.get("description")
    out["description"] = str(_scalar(desc)).strip() if not _blank(desc) else None
    out["file_hash"] = resp.get("file_hash") or (md.get("_file_meta") or {}).get("sha256")

    # 시험별 널 처리 / 보충
    if test == "T-001":
        for k in T002_KEYS:
            out[k] = None
    elif test == "T-002":
        ctx = resp.get("contract_context") or {}
        if out["work_title"] is None and not _blank(ctx.get("title")):
            out["work_title"] = _title(ctx.get("title"))
            meta["work_title_source"] = "contract_context"
        if out["work_type"] is None and not _blank(ctx.get("work_type")):
            out["work_type"], _ = _work_type(ctx.get("work_type"))
            meta["work_type_source"] = "contract_context"

    # 출처 기록 — 어떤 문맥이 VLM 에 들어갔고 규칙 파서는 무엇을 읽었는지
    if resp.get("contract_context"):
        meta["contract_context"] = resp.get("contract_context")
    rules = resp.get("contract_context_rules")
    if isinstance(rules, dict):
        meta["rule_title"] = rules.get("title")
        meta["rule_work_type"] = rules.get("work_type")
    detail = resp.get("licensee_party_detail")
    if isinstance(detail, dict):
        meta["licensee_rule"] = detail
    if unparsed:
        meta["unparsed"] = unparsed
    out["_meta"] = meta
    return out
