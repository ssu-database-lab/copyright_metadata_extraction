"""35-label 스키마 + 3-way 분할 (REGEX / NER / LLM).

추출 파이프라인:
  1. REGEX  — 형식 규칙이 명확한 9 라벨. 정확도 100%.
  2. NER    — 자유 형식 한국어 텍스트 17 라벨. BERT token classification.
  3. LLM    — 문맥·정책 판정 9 라벨. (미구현)
"""
from __future__ import annotations

from typing import FrozenSet, Tuple


REGEX_LABELS: Tuple[str, ...] = (
    "phone",
    "email",
    "copyright_url",
    "copyright_uci",
    "date",
    "ri_money",
    "copyright_num",
    "copyright_idnum",
    "copyright_quantity",
)


NER_LABELS: Tuple[str, ...] = (
    # copyright_info (6)
    "copyright_kotitle",
    "copyright_status",
    "copyright_description",
    "copyright_Keyword",
    "copyright_language",
    "copyright_type",
    # author_info (5)
    "name",
    "company",
    "address",
    "position",
    "department",
    # rights_info (6) — ri_period 는 한국어 표현 ("동의 시부터 ... 만료일까지") 비중 커서 NER.
    "ri_data",
    "ri_period",
    "ri_info",
    "ri_contract_type",
    "ri_copyright",
    "ri_law_reference",
)


LLM_DELEGATED_LABELS: Tuple[str, ...] = (
    # copyright_info (3)
    "copyright_id",
    "copyright_Pname",
    "copyright_con_status",
    # rights_info (6) — 문맥 판정·정책
    "ri_cpcheck",
    "ri_uncopyright",
    "ri_workhire",
    "ri_consent_type",
    "ri_jch_conset",
    "ri_portrait",
)


REGEX_LABEL_SET: FrozenSet[str] = frozenset(REGEX_LABELS)
NER_LABEL_SET: FrozenSet[str] = frozenset(NER_LABELS)
LLM_DELEGATED_LABEL_SET: FrozenSet[str] = frozenset(LLM_DELEGATED_LABELS)


ALL_LABELS: Tuple[str, ...] = REGEX_LABELS + NER_LABELS + LLM_DELEGATED_LABELS


# 정합성 — 세 집합이 ALL_LABELS 를 disjoint 3-way 분할.
assert REGEX_LABEL_SET.isdisjoint(NER_LABEL_SET)
assert REGEX_LABEL_SET.isdisjoint(LLM_DELEGATED_LABEL_SET)
assert NER_LABEL_SET.isdisjoint(LLM_DELEGATED_LABEL_SET)
assert REGEX_LABEL_SET | NER_LABEL_SET | LLM_DELEGATED_LABEL_SET == set(ALL_LABELS)
