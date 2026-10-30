"""v4 평가 — 23개 고정 키 정의 (docs/v4/M1_evaluator_spec.md §1).

이 파일이 키 이름·순서·타입·시험 소속의 단일 출처다. 다른 모듈은 KEYS 를 import 해서 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

KeyType = Literal["str", "name_list", "category", "rights", "date", "period", "tag_list", "format", "int", "resolution", "description", "hash", "na"]


@dataclass(frozen=True)
class Key:
    no: int
    name: str
    label_ko: str
    group_ko: str
    type: KeyType
    test: Literal["T-001", "T-002"]


KEYS: tuple[Key, ...] = (
    Key(1, "work_title", "저작물명", "식별정보", "str", "T-001"),
    Key(2, "file_name", "파일명", "식별정보", "str", "T-001"),
    Key(3, "work_type", "저작물유형", "식별정보", "category", "T-001"),
    Key(4, "creator", "저작자", "권리주체정보", "name_list", "T-001"),
    Key(5, "copyright_holder", "저작권자", "권리주체정보", "name_list", "T-001"),
    Key(6, "reproduction_right", "복제권", "권리정보", "rights", "T-001"),
    Key(7, "performance_right", "공연권", "권리정보", "rights", "T-001"),
    Key(8, "public_transmission_right", "공중송신권", "권리정보", "rights", "T-001"),
    Key(9, "exhibition_right", "전시권", "권리정보", "rights", "T-001"),
    Key(10, "distribution_right", "배포권", "권리정보", "rights", "T-001"),
    Key(11, "rental_right", "대여권", "권리정보", "rights", "T-001"),
    Key(12, "derivative_work_right", "2차적저작물작성권", "권리정보", "rights", "T-001"),
    Key(13, "other_rights", "기타 권리", "권리정보", "na", "T-001"),
    Key(14, "license_start_date", "이용허락 시작일", "이용허락·계약정보", "date", "T-001"),
    Key(15, "license_end_date", "이용허락 종료일", "이용허락·계약정보", "date", "T-001"),
    Key(16, "contract_validity_period", "계약 유효기간", "이용허락·계약정보", "period", "T-001"),
    Key(17, "keywords", "키워드", "내용·특징정보", "tag_list", "T-002"),
    Key(18, "hashtags", "해시태그", "내용·특징정보", "tag_list", "T-002"),
    Key(19, "file_format", "파일포맷", "파일·기술정보", "format", "T-002"),
    Key(20, "file_size", "파일크기", "파일·기술정보", "int", "T-002"),
    Key(21, "resolution", "해상도", "파일·기술정보", "resolution", "T-002"),
    Key(22, "description", "설명", "내용·특징정보", "description", "T-002"),
    Key(23, "file_hash", "Hash 데이터", "파일·기술정보", "hash", "T-002"),
)

KEY_NAMES: tuple[str, ...] = tuple(k.name for k in KEYS)
BY_NAME: dict[str, Key] = {k.name: k for k in KEYS}
T001_KEYS: tuple[str, ...] = tuple(k.name for k in KEYS if k.test == "T-001")
T002_KEYS: tuple[str, ...] = tuple(k.name for k in KEYS if k.test == "T-002")
# Te-2 가 계약서에서 받는 항목 (1~16번에서 file_name 을 뺀 것 — Te-2 의 파일명은 저작물 파일의 이름이다).
# Te-1 결과가 있으면 그 값을 쓰고, 없으면 Te-2 가 계약서를 직접 처리한다 (batch_runner, 2610028 최종 절차서)
CONTRACT_ITEM_KEYS: tuple[str, ...] = tuple(n for n in T001_KEYS if n != "file_name")
RIGHTS_KEYS: tuple[str, ...] = tuple(k.name for k in KEYS if k.type == "rights")
WORK_TYPES: tuple[str, ...] = ("어문", "음악", "연극", "미술", "건축", "사진", "영상", "도형", "컴퓨터프로그램", "기타")


def scored_keys(test: str) -> tuple[str, ...]:
    """시험별 채점 대상 키 — 2610028 최종 절차서 부록 2: Te-1 은 1~16번, Te-2 는 1~23번
    (other_rights 는 전 계약서에서 상태 ③ 이므로 평가기에서 제외된다)."""
    return T001_KEYS if test == "T-001" else KEY_NAMES


def empty_prediction() -> dict[str, None]:
    return {k: None for k in KEY_NAMES}
