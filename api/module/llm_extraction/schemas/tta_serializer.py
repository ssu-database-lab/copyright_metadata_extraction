"""평면 추출 결과를 TTA 표준 중첩 JSON(10그룹 · 66 하위요소)으로 직렬화한다.

왜 평면으로 뽑고 나중에 중첩하는가
--------------------------------
LLM 에게 처음부터 중첩 JSON 을 요구하면 통합(consolidator)·채점(scoring)·NER 매핑이
모두 경로 기반으로 바뀐다. 실측으로, 이름 대응표 없이 중첩만 시키면 채점이
0.48 → 0.00 으로 떨어진다(키가 하나도 안 맞으므로). 추출은 평면 그대로 두고
출력 직전에 한 번 접는 편이 파이프라인 전체에서 바뀌는 곳이 가장 적다.

빈도(cardinality)는 TTA 표에서 온다: 0:n = 배열, 0:1 = 선택 단일, 1 = 필수 단일.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# TTA 표준 "빈도" 열이 0:n 인 요소 — 항상 배열로 낸다.
_ARRAY_KEYS = {
    "file_name", "original_source_location", "collection_path", "source_description",
    "media_type", "file_format", "language",
    "author", "copyright_holder", "economic_rights_holder", "licensor",
    "co_authors", "rights_holder_identifier",
    "assessment_basis", "previous_public_release_type", "usage_restrictions",
    "other_rights", "contract_validity_period",
    "moral_rights_notes", "name_attribution_condition", "rights_restriction_description",
    "verification_date", "verifier", "error_type", "review_history",
}

# 그룹 → (TTA 하위요소 키, 평면 스키마에서 가져올 소스)
# 소스가 None 이면 아직 추출원이 없는 항목(스펙상 존재하지만 계약서에 없음).
_MAP: Dict[str, List[tuple]] = {
    "identification_information": [
        ("work_identifier", None),
        ("work_title", "work_title"),
        ("file_name", None),                     # 파이프라인 응답의 filename 에서 주입
        ("registration_number", None),
        ("registration_date", "registration_date"),
        ("managing_organization_identifier", None),
    ],
    "source_information": [
        ("providing_organization", "agency_name"),
        ("managing_organization", "site_name"),
        ("original_source_location", "url"),
        ("database_name", "board_name"),
        ("collection_path", "board_path"),
        ("source_description", "memo"),
    ],
    "work_type_information": [
        ("work_type", "work_type"),
        ("media_type", None),                    # digital_format 에서 파생
        ("file_format", "digital_format"),
        ("language", "language"),
        ("creation_date", "created_date"),
        ("publication_date", None),
    ],
    "rights_holder_information": [
        ("author", "author"),
        ("copyright_holder", "copyright_holder"),
        ("economic_rights_holder", "economic_rights_holder"),
        ("licensor", "licensor"),
        ("co_authors", "co_author"),
        ("rights_holder_identifier", "rights_holder_identifier"),
    ],
    "rights_assessment_information": [
        ("copyrightability_status", "copyrightability"),
        ("unprotected_work_status", "unprotected_work"),
        ("work_made_for_hire_status", "work_for_hire"),
        ("rights_ownership_status", None),
        ("assessment_basis", None),              # 통합 근거에서 주입
        ("assessment_date", None),
    ],
    "public_release_and_license_conditions": [
        ("public_release_type", "disclosure_type"),
        ("previous_public_release_type", None),
        ("final_public_release_type", "kogl_type"),
        ("attribution_required", None),
        ("commercial_use_permitted", "commercial_use"),
        ("modification_permitted", None),
        ("share_alike_required", None),
        ("usage_restrictions", "special_terms"),
    ],
    "detailed_economic_rights_information": [
        ("reproduction_right", "reproduction_right"),
        ("public_performance_right", "public_performance_right"),
        ("public_transmission_right", "public_transmission_right"),
        ("exhibition_right", "exhibition_right"),
        ("distribution_right", "distribution_right"),
        ("rental_right", "rental_right"),
        ("derivative_work_creation_right", "derivative_work_creation_right"),
        ("other_rights", "other_rights"),
    ],
    "validity_period_information": [
        ("copyright_expiration_date", None),
        ("license_start_date", "license_start_date"),
        ("license_end_date", "license_end_date"),
        ("contract_validity_period", "valid_period"),
        ("public_release_type_effective_date", None),
        ("validity_review_status", None),
    ],
    "moral_and_portrait_rights_information": [
        ("moral_rights_notes", None),
        ("name_attribution_condition", None),
        ("integrity_right_restrictions", None),
        ("portrait_rights_included", "portrait_rights"),
        ("third_party_rights_included", "third_party_rights"),
        ("rights_restriction_description", None),
    ],
    # 검증정보는 추출 대상이 아니라 파이프라인이 아는 사실이다 — build() 에서 주입한다.
    "verification_information": [
        ("verification_status", None),
        ("verification_result", None),
        ("verification_date", None),
        ("verifier", None),
        ("verification_method", None),
        ("error_type", None),
        ("review_history", None),
        ("last_updated_date", None),
    ],
}

# 파일 형식 → 매체 유형 (TTA 3.2). 확장자 문자열에서 결정적으로 유도한다.
_MEDIA_BY_FORMAT = {
    "jpg": "이미지", "jpeg": "이미지", "png": "이미지", "gif": "이미지", "bmp": "이미지",
    "tif": "이미지", "tiff": "이미지", "webp": "이미지",
    "mp4": "영상", "avi": "영상", "mov": "영상", "mkv": "영상", "wmv": "영상", "swf": "영상",
    "mp3": "음원", "wav": "음원", "flac": "음원",
    "pdf": "문서", "hwp": "문서", "hwpx": "문서", "docx": "문서", "doc": "문서", "txt": "문서",
}


def _as_array(v: Any) -> Optional[List[Any]]:
    if v is None:
        return None
    if isinstance(v, list):
        return [x for x in v if x not in (None, "")] or None
    return [v]


def _media_type(fmt: Any) -> Optional[List[str]]:
    if not fmt:
        return None
    key = str(fmt).strip().lower().lstrip(".")
    hit = _MEDIA_BY_FORMAT.get(key)
    return [hit] if hit else None


def build(metadata: Dict[str, Any],
          response: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """평면 metadata(+파이프라인 응답)를 TTA 중첩 구조로 접는다.

    metadata : 통합 결과의 평면 dict (consolidated_metadata 또는 llm metadata)
    response : llm_metadata.json 전체. 검증정보·파일명처럼 추출이 아니라
               파이프라인이 이미 아는 값들을 여기서 가져온다.
    """
    response = response or {}
    out: Dict[str, Any] = {}

    for group, leaves in _MAP.items():
        node: Dict[str, Any] = {}
        for leaf, src in leaves:
            val = metadata.get(src) if src else None
            node[leaf] = _as_array(val) if leaf in _ARRAY_KEYS else val
        out[group] = node

    # --- 추출이 아니라 파이프라인이 아는 값들 ---
    ident = out["identification_information"]
    ident["file_name"] = _as_array(response.get("filename"))
    ident["work_identifier"] = response.get("request_id")

    wt = out["work_type_information"]
    wt["media_type"] = _media_type(metadata.get("digital_format"))

    ra = out["rights_assessment_information"]
    ra["assessment_basis"] = _as_array(response.get("consolidation_summary"))

    ver = out["verification_information"]
    ok = bool(response.get("consolidation_success", response.get("success")))
    ver["verification_status"] = "완료" if ok else "실패"
    ver["verification_result"] = "정상" if ok else "오류"
    ver["verifier"] = _as_array(response.get("consolidation_model_used")
                                or response.get("model_used"))
    ver["verification_method"] = "자동 (OCR→LLM+NER→통합 중재)"
    ver["error_type"] = _as_array(response.get("consolidation_error")
                                  or response.get("error"))
    ver["last_updated_date"] = response.get("extraction_time")

    return out


def leaf_keys() -> List[str]:
    """66개 하위요소 키 (검증용)."""
    return [leaf for leaves in _MAP.values() for leaf, _ in leaves]
