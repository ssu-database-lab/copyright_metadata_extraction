"""
계약서 → 저작물(work) 메타데이터 상속 (contract→work inheritance).

배경 (2026-07 무하유 정합성 점검): 무하유 kogl-classifier의 '검사' 단위는
계약서 1건 + 저작물 파일 N건인데, 저작물 이미지의 VLM 추출은 시각 필드
(description/work_type/keyword/digital_format)만 채운다. 저작물 레코드의
권리정보(저작권자·이용허락·유효기간·공공누리 유형 등)는 '계약서'의 사실이므로,
클라이언트(HF Space)가 계약서 분석 결과 JSON을 함께 보내면 여기서 병합한다.

사용: POST /api/llm-extract 에 optional form field `contract_metadata`
(계약서 처리 결과의 consolidated_metadata 또는 metadata JSON 문자열).

병합 규칙:
- INHERITABLE_FIELDS 만 상속. 시각 필드(description, work_type, keyword 등)는
  절대 계약서 값으로 덮지 않는다 (저작물 자체 분석이 우선).
- 저작물(work) 측에 이미 값이 있으면 유지하고 계약서 값은 참고로 decision에만 기록.
- work_title: 계약서가 복수 저작물을 다루면(리스트/구분자) 파일명·VLM 설명과
  토큰 매칭으로 해당 저작물을 선택. 명확히 못 고르면 CONTRACT_AMBIGUOUS.
- 각 상속 필드는 consolidator 의 decisions 항목과 동일한 스키마로 provenance 를 남긴다:
  {field, llm_value, ner_value, final_value, decision, reasoning, confidence, evidence}
  decision ∈ {CONTRACT_INHERITED (0.8), CONTRACT_AMBIGUOUS (0.5)}.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# 계약서에서 저작물 레코드로 상속 가능한 권리/계약 필드 (67-field unified schema 부분집합).
# 시각/식별 필드(description, keyword, digital_format, language 등)는 제외.
INHERITABLE_FIELDS = [
    # 권리 주체
    "copyright_holder", "co_author", "neighboring_rights_holder",
    # 공개/이용 조건
    "disclosure_type", "commercial_use", "economic_rights", "kogl_type",
    "granted_rights", "portrait_rights", "third_party_rights", "co_author_consent",
    "copyrightability", "unprotected_work", "work_for_hire",
    # 기간/일자
    "valid_period", "contract_duration", "signature_date",
    "effective_date", "expiration_date", "created_date", "production_date",
    # 계약 정보
    "contract_type", "payment_amount", "payment_currency",
    "special_terms", "termination_conditions", "contract_terms",
    # 당사자/연락
    "parties", "contact_info",
    # 개인정보 관련 (동의서 계열)
    "personal_info", "consent_type", "consent_status", "consent_date",
]

# work_title 은 특별 취급: 매칭 성공 시에만 상속.
_TITLE_FIELD = "work_title"

_INHERITED_CONF = 0.8
_AMBIGUOUS_CONF = 0.5


def _norm(s: Any) -> str:
    return re.sub(r"[\s\W_]+", "", str(s or ""), flags=re.UNICODE).lower()


def _tokens(s: Any) -> set:
    # 파일명 구분자(_-.)와 괄호류까지 토큰 경계로 취급 ("제주_유채꽃.jpg" → 제주/유채꽃/jpg)
    return {t for t in re.split(r"[\s,;/·|_\-.()\[\]]+", str(s or "").lower()) if len(t) >= 2}


def _candidate_titles(value: Any) -> List[str]:
    """계약서의 저작물명 값에서 후보 목록 추출.

    리스트면 각 원소가 저작물 1건. 문자열이면 강한 구분자([;|]·개행·' 외 N건')로만
    분리한다 — 쉼표(,)·슬래시(/)·가운뎃점(·)은 한국어 제목 안에 흔히 등장하므로
    (예: "계원마을_김상채, 김종례_결혼식 사진") 구분자로 취급하지 않는다.
    복수 저작물은 HF Space 연동 시 리스트(work_names)로 오는 것이 정상 경로."""
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(v) for v in value if v]
    else:
        items = re.split(r"[;|\n]|\s외\s*\d*\s*[건점편]?", str(value))
    return [t.strip() for t in items if t and t.strip()]


def match_work_title(contract_title_value: Any, work_filename: str = "",
                     work_description: str = "") -> Tuple[Optional[str], str]:
    """
    계약서 저작물명(단일/복수)에서 이 저작물 파일에 해당하는 제목을 고른다.
    returns (선택된 제목 | None, 판정: matched|single|ambiguous|none)
    """
    cands = _candidate_titles(contract_title_value)
    if not cands:
        return None, "none"
    if len(cands) == 1:
        return cands[0], "single"
    # 복수 후보: 파일명/설명과 토큰 겹침 점수로 매칭
    hint = _tokens(work_filename) | _tokens(work_description)
    hint_norm = _norm(work_filename) + _norm(work_description)
    best, best_score = None, 0.0
    for c in cands:
        score = 0.0
        cn = _norm(c)
        if cn and cn in hint_norm:
            score += 2.0
        overlap = _tokens(c) & hint
        score += len(overlap) * 0.5
        if score > best_score:
            best, best_score = c, score
    if best is not None and best_score >= 1.0:
        return best, "matched"
    return None, "ambiguous"


def inherit_contract_fields(
    work_metadata: Dict[str, Any],
    contract_metadata: Dict[str, Any],
    work_filename: str = "",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    계약서 메타데이터를 저작물 메타데이터에 병합.

    Returns:
        merged: 병합된 저작물 메타데이터 (원본 dict 는 수정하지 않음)
        decisions: consolidator 형식의 provenance 항목 리스트 (상속 필드만)
        summary: {"inherited": n, "skipped_existing": n, "title_match": str}
    """
    merged = dict(work_metadata or {})
    contract = contract_metadata or {}
    decisions: List[Dict[str, Any]] = []
    inherited = skipped = 0

    def _empty(v: Any) -> bool:
        return v is None or v == "" or v == [] or v == {}

    def _add_decision(field: str, value: Any, decision: str, reasoning: str, conf: float):
        decisions.append({
            "field": field,
            "llm_value": None,
            "ner_value": None,
            "final_value": value,
            "decision": decision,
            "reasoning": reasoning,
            "confidence": conf,
            "evidence": {"source": "계약서 메타데이터 상속 (contract_metadata)"},
        })

    # 1) 일반 상속 필드
    for field in INHERITABLE_FIELDS:
        cval = contract.get(field)
        if _empty(cval):
            continue
        if not _empty(merged.get(field)):
            skipped += 1  # 저작물 자체 분석 값 우선 — 계약서 값은 덮지 않음
            continue
        merged[field] = cval
        inherited += 1
        _add_decision(field, cval, "CONTRACT_INHERITED",
                      "저작물 파일에서 추출 불가한 권리/계약 정보 — 연계된 계약서 메타데이터에서 상속",
                      _INHERITED_CONF)

    # 2) work_title 매칭 상속
    title_match = "none"
    if _empty(merged.get(_TITLE_FIELD)):
        title, title_match = match_work_title(
            contract.get(_TITLE_FIELD) or contract.get("work_names"),
            work_filename=work_filename,
            work_description=str(merged.get("description") or ""),
        )
        if title is not None:
            merged[_TITLE_FIELD] = title
            inherited += 1
            _add_decision(_TITLE_FIELD, title, "CONTRACT_INHERITED",
                          ("계약서의 단일 저작물명 상속" if title_match == "single"
                           else "계약서의 복수 저작물명 중 파일명/설명 매칭으로 선택"),
                          _INHERITED_CONF if title_match == "single" else _AMBIGUOUS_CONF + 0.2)
        elif title_match == "ambiguous":
            cands = _candidate_titles(contract.get(_TITLE_FIELD) or contract.get("work_names"))
            _add_decision(_TITLE_FIELD, None, "CONTRACT_AMBIGUOUS",
                          f"계약서에 저작물 {len(cands)}건 — 파일명/설명으로 특정 불가 (후보: {', '.join(cands[:5])})",
                          _AMBIGUOUS_CONF)

    summary = {"inherited": inherited, "skipped_existing": skipped, "title_match": title_match}
    return merged, decisions, summary
