"""
Schema mapping: VLM / 임베딩 출력 → 프로젝트의 기존 통합 67-필드 메타데이터 스키마.

생성형 VLM이 뱉은 JSON(prompts.py의 USER_PROMPT 키)을 기존 LLM 추출
파이프라인이 만드는 것과 *똑같은 모양*의 unified-schema dict로 변환한다.
이렇게 해야 VLM 결과가 NER/Consolidator 등 다운스트림에 그대로 흘러갈 수 있다.

★ 하드 규칙: 여기서 emit하는 키는 반드시 get_unified_schema()['properties']에
  실제로 존재하는 키여야 한다. 스키마에 없는 필드는 절대 만들어내지 않는다.
  스키마에 자리가 없는 정보(파일 해시, VLM 부가 메타)는 별도의 `_file_meta`
  dict에 담아 함께 돌려준다 — 가짜 스키마 필드에 욱여넣지 않는다.

무거운 의존성은 lazy import한다(스키마 모듈만 함수 안에서 불러옴).
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# VLM 출력 키(prompts.py USER_PROMPT) → 통합 스키마 키 매핑
# ---------------------------------------------------------------------------
# VLM이 내보내는 키:
#   description, work_type, work_type_reason, keywords, main_subjects,
#   dominant_colors, text_in_image, scene_type, estimated_quality
#
# 실제 스키마에 직접 1:1로 대응되는 것만 매핑한다:
#   description  -> description   (스키마: 설명/콘텐츠 설명)
#   work_type    -> work_type     (스키마: 유형/저작물 유형)
#   keywords     -> keyword       (스키마: 주제어 — string|array 허용)
#
# 스키마에 자리가 없어 의도적으로 버리는(또는 _file_meta로만 보존하는) 키:
#   work_type_reason, main_subjects, dominant_colors, text_in_image,
#   scene_type, estimated_quality
#   → keyword 필드를 main_subjects로 보강할 수는 있으나, keyword(주제어)는
#     "키워드/태그"이고 main_subjects는 "피사체 목록"이라 의미가 겹치므로
#     keywords가 비었을 때만 main_subjects를 fallback으로 사용한다.
_VLM_DIRECT_MAP: Dict[str, str] = {
    "description": "description",
    "work_type": "work_type",
    "keywords": "keyword",
}


def _empty_unified_schema() -> Dict[str, Any]:
    """
    모든 통합 스키마 필드를 null/빈 값으로 초기화한 dict를 만든다.

    파이프라인이 기대하는 것과 동일한 shape(스키마-complete)을 보장하기 위해,
    get_unified_schema()의 실제 properties 키만 사용한다.
    배열 타입 필드는 [] 로, 그 외에는 None 으로 초기화한다.
    """
    # lazy import — 스키마 모듈은 표준 라이브러리만 쓰지만 import 경로가 길어
    # 함수 안에서 불러 모듈 로드 시점을 늦춘다.
    from module.llm_extraction.schemas.document_schemas import DocumentSchemas

    props = DocumentSchemas.get_unified_schema()["properties"]
    out: Dict[str, Any] = {}
    for key, spec in props.items():
        types = spec.get("type", [])
        if isinstance(types, str):
            types = [types]
        # "array"만 가능한(다른 스칼라 대안이 없는) 필드는 []로 초기화
        if "array" in types and not ({"string", "number", "integer", "object"} & set(types)):
            out[key] = []
        else:
            out[key] = None
    return out


def _ext_to_digital_format(file_path: str) -> Optional[str]:
    """
    파일 확장자를 digital_format 값으로 변환.

    convention: 점 없는 대문자 확장자 (".jpg" -> "JPG", ".mp4" -> "MP4").
    스키마 설명("PDF, JPG, MP4, HWP 등")이 대문자 확장자 표기를 쓰므로 그에 맞춘다.
    """
    ext = os.path.splitext(str(file_path))[1]
    if not ext:
        return None
    return ext[1:].upper() if ext.startswith(".") else ext.upper()


def _sha256_of_file(file_path: str) -> Optional[str]:
    """파일 내용의 SHA256 해시(hex). 파일이 없거나 못 읽으면 None."""
    try:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):  # 1MB 단위 스트리밍
                h.update(chunk)
        return h.hexdigest()
    except (OSError, TypeError):
        return None


def map_vlm_to_unified(
    vlm_output: Dict[str, Any],
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    VLM JSON 출력을 통합 67-필드 스키마 dict로 변환한다.

    Args:
        vlm_output: VLM이 생성한 dict (prompts.py USER_PROMPT 키들).
        file_path:  원본 파일 경로(선택). 주어지면 digital_format을 채우고
                    파일 SHA256 해시를 계산해 `_file_meta`에 넣는다.

    Returns:
        스키마-complete dict. 모든 67개 필드가 존재(매핑 못한 건 null/[]),
        매핑 가능한 필드만 채워짐. 스키마에 자리 없는 부가정보(해시, VLM
        진단 키들)는 `_file_meta` 키 아래 별도 dict로 동봉된다.

        주의: 반환 dict는 스키마 67개 키 + (옵션) `_file_meta` 1개를 가진다.
        `_file_meta`는 스키마 필드가 아니며 언더스코어 접두사로 구분된다.
    """
    vlm_output = vlm_output or {}
    unified = _empty_unified_schema()

    # --- 1) 직접 1:1 매핑 ---------------------------------------------------
    for vlm_key, schema_key in _VLM_DIRECT_MAP.items():
        if vlm_key not in vlm_output:
            continue
        value = vlm_output[vlm_key]
        if value is None or value == "" or value == []:
            continue
        unified[schema_key] = value

    # --- 2) keyword 보강: keywords가 비었으면 main_subjects를 fallback ------
    #     (둘 다 스키마 keyword 필드는 string|array를 허용)
    if not unified.get("keyword"):
        subjects = vlm_output.get("main_subjects")
        if subjects:
            unified["keyword"] = subjects

    # --- 3) 파일 기반 필드 --------------------------------------------------
    file_meta: Dict[str, Any] = {}
    if file_path:
        digital_format = _ext_to_digital_format(file_path)
        if digital_format:
            unified["digital_format"] = digital_format

        sha256 = _sha256_of_file(file_path)
        # 스키마에는 해시/UCI 전용 필드가 없으므로 _file_meta로만 보존한다.
        file_meta["file_path"] = str(file_path)
        file_meta["file_name"] = os.path.basename(str(file_path))
        file_meta["sha256"] = sha256

    # --- 4) 스키마에 자리 없는 VLM 진단 키를 _file_meta에 보존(분실 방지) ----
    #     work_type_reason / scene_type / dominant_colors / text_in_image /
    #     estimated_quality / main_subjects 등은 다운스트림 디버깅/근거용으로만.
    vlm_extras = {
        k: vlm_output[k]
        for k in (
            "work_type_reason",
            "main_subjects",
            "dominant_colors",
            "text_in_image",
            "scene_type",
            "estimated_quality",
        )
        if k in vlm_output and vlm_output[k] not in (None, "", [])
    }
    if vlm_extras:
        file_meta["vlm_extras"] = vlm_extras

    if file_meta:
        unified["_file_meta"] = file_meta

    return unified


def map_embedding_to_unified(
    embedding: Any = None,
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    (스텁) 임베딩 → 통합 스키마.

    임베딩 벡터 자체는 스키마 필드에 매핑되지 않는다 — 임베딩은 '유사도 검색
    /중복 탐지'에 쓰이지 메타데이터 속성값을 만들지 않기 때문이다. 따라서
    스키마-complete 빈 dict를 돌려주고, 벡터/해시는 `_file_meta`에만 보존한다.

    실제 속성(work_type, keyword 등)이 필요하면 임베딩 최근접 라벨을 별도
    분류기로 뽑아 map_vlm_to_unified와 유사하게 채우는 식으로 추후 확장한다.
    """
    unified = _empty_unified_schema()
    file_meta: Dict[str, Any] = {}
    if file_path:
        file_meta["file_path"] = str(file_path)
        file_meta["sha256"] = _sha256_of_file(file_path)
    if embedding is not None:
        # numpy/torch에 의존하지 않도록 길이만 기록(있으면)
        try:
            file_meta["embedding_dim"] = len(embedding)
        except TypeError:
            file_meta["embedding_dim"] = None
        file_meta["note"] = "embeddings feed similarity/dedup, not schema fields"
    if file_meta:
        unified["_file_meta"] = file_meta
    return unified


if __name__ == "__main__":
    import glob
    import json
    import sys

    # 스키마 모듈 import 경로 보장 (api/ 를 sys.path에 추가)
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _API_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))  # .../api
    if _API_ROOT not in sys.path:
        sys.path.insert(0, _API_ROOT)

    # prompts.py 키에 맞춘 현실적인 샘플 VLM 출력
    sample_vlm_output = {
        "description": "야외 시장 좌판 위에 빨간 국물의 떡볶이가 담긴 접시가 놓여 있다. 김이 올라오고 있으며 배경은 흐릿하게 처리되어 있다.",
        "work_type": "사진저작물",
        "work_type_reason": "카메라로 촬영한 음식 사진이므로 매체 기준 사진저작물.",
        "keywords": ["떡볶이", "분식", "한국음식", "길거리음식", "야외시장"],
        "main_subjects": ["떡볶이", "접시", "좌판"],
        "dominant_colors": ["빨강", "주황", "갈색"],
        "text_in_image": None,
        "scene_type": "실외",
        "estimated_quality": "고화질",
    }

    # test_data/sample_works 에서 실제 샘플 파일 하나 집어오기(있으면)
    sample_dir = os.path.join(_HERE, "test_data", "sample_works")
    sample_files = sorted(glob.glob(os.path.join(sample_dir, "*.jpg")))
    sample_file = sample_files[0] if sample_files else None

    print("=" * 70)
    print("schema_mapping — map_vlm_to_unified() on a realistic sample")
    print("=" * 70)
    if sample_file:
        print(f"sample file: {os.path.basename(sample_file)}")
    else:
        print("sample file: <none found — running without file_path>")

    result = map_vlm_to_unified(sample_vlm_output, file_path=sample_file)
    print("\n--- unified-schema dict ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # ----- 검증: emit한 스키마 키가 전부 실제 스키마에 존재하는지 -----------
    from module.llm_extraction.schemas.document_schemas import DocumentSchemas

    schema_keys = set(DocumentSchemas.get_unified_schema()["properties"].keys())
    emitted_keys = set(result.keys())
    # _file_meta는 의도적으로 스키마 밖(언더스코어 접두) 부가 키이므로 제외하고 검증
    schema_emitted = {k for k in emitted_keys if not k.startswith("_")}
    non_null_filled = {
        k for k in schema_emitted if result[k] not in (None, "", [])
    }

    print("\n--- verification ---")
    print(f"unified schema field count : {len(schema_keys)}")
    print(f"emitted schema keys (count): {len(schema_emitted)}")
    print(f"non-schema extra keys      : {sorted(emitted_keys - schema_emitted)}")
    print(f"filled (non-null) fields   : {sorted(non_null_filled)}")
    invalid = schema_emitted - schema_keys
    print(f"INVALID keys (must be empty): {sorted(invalid)}")
    assert not invalid, f"emitted keys not in real schema: {invalid}"
    assert schema_emitted == schema_keys, (
        "output is not schema-complete: "
        f"missing={sorted(schema_keys - schema_emitted)}"
    )
    print("OK: all emitted schema keys exist in get_unified_schema() "
          "and output is schema-complete (67 fields).")
