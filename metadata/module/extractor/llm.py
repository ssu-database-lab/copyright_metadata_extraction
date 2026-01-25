"""LLM 추출기: regular와 ner 결과 통합 및 최종 정리"""
from __future__ import annotations

from typing import Dict, Any, List

from module.parts.types import Decision


def llm_extractor(
    *,
    raw_text: str,
    sentences: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
    previous_decisions: List[Decision],
) -> List[Decision]:
    """
    LLM 기반 통합 및 최종 정리
    
    regular와 ner의 모든 결과를 받아서 LLM이 최종 판단
    우선순위: regular < ner < llm (llm이 최종 결정)
    충돌은 무시하고 모든 결과를 LLM에게 전달
    """
    if not previous_decisions:
        return []
    
    # regular와 ner 결과 분리 (충돌 무시, 모두 전달)
    regular_sources = {"regex", "datetime", "numeric"}
    regular_decisions = [d for d in previous_decisions if d.source in regular_sources]
    ner_decisions = [d for d in previous_decisions if d.source == "ner"]
    
    # 모든 결과를 LLM에게 전달 (충돌 무시)
    all_decisions = regular_decisions + ner_decisions
    
    # LLM이 문맥을 고려하여 최종 판단
    final_decisions = _llm_finalize(
        all_decisions=all_decisions,
        sentences=sentences,
        tokens=tokens,
        raw_text=raw_text
    )
    
    return final_decisions


def _llm_finalize(
    all_decisions: List[Decision],
    sentences: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
    raw_text: str
) -> List[Decision]:
    """
    LLM이 모든 결과를 받아서 최종 판단
    
    우선순위: regular < ner < llm
    - 같은 라벨에 여러 값이 있으면 LLM이 문맥을 고려하여 선택
    - 누락된 메타데이터 추론
    - 값 검증 및 보정
    """
    # 현재는 기본 통합만 수행 (향후 LLM 구현)
    # 같은 라벨에 여러 값이 있으면 우선순위: ner > regular
    final_decisions: List[Decision] = []
    label_to_decisions: Dict[str, List[Decision]] = {}
    
    # 라벨별로 그룹화
    for d in all_decisions:
        if not d.value or not d.value.strip():
            continue
        label = d.label
        label_to_decisions.setdefault(label, []).append(d)
    
    # 각 라벨에 대해 최종 결정
    for label, decisions in label_to_decisions.items():
        if len(decisions) == 1:
            # 값이 하나면 그대로 사용
            final_decisions.append(Decision(
                label=label,
                value=decisions[0].value.strip(),
                sent_id=decisions[0].sent_id,
                tok_id=decisions[0].tok_id,
                source="llm",
                meta={**decisions[0].meta, "integrated": True, "sources": [decisions[0].source]}
            ))
        else:
            # 여러 값이 있으면 우선순위: ner > regular
            # ner 결과 우선, 없으면 regular 결과
            ner_d = next((d for d in decisions if d.source == "ner"), None)
            if ner_d:
                final_decisions.append(Decision(
                    label=label,
                    value=ner_d.value.strip(),
                    sent_id=ner_d.sent_id,
                    tok_id=ner_d.tok_id,
                    source="llm",
                    meta={**ner_d.meta, "integrated": True, "sources": [d.source for d in decisions]}
                ))
            else:
                # regular 결과 사용
                regular_d = decisions[0]
                final_decisions.append(Decision(
                    label=label,
                    value=regular_d.value.strip(),
                    sent_id=regular_d.sent_id,
                    tok_id=regular_d.tok_id,
                    source="llm",
                    meta={**regular_d.meta, "integrated": True, "sources": [d.source for d in decisions]}
                ))
    
    # TODO: LLM을 사용한 문맥 분석
    # - sentences, tokens, raw_text를 활용하여 최종 판단
    # - 누락된 메타데이터 추론
    # - 값 검증 및 보정
    # - 충돌하는 값들 중 가장 적절한 것 선택
    
    return final_decisions


def merge_regular_ner(regular_decisions: List[Decision], ner_decisions: List[Decision]) -> List[Decision]:
    """
    regular와 ner 결과를 통합 (우선순위: ner > regular)
    
    같은 라벨에 여러 값이 있으면 ner 결과 우선
    """
    final_decisions: List[Decision] = []
    label_to_decisions: Dict[str, List[Decision]] = {}
    
    # 모든 결과를 라벨별로 그룹화
    for d in regular_decisions + ner_decisions:
        if not d.value or not d.value.strip():
            continue
        label_to_decisions.setdefault(d.label, []).append(d)
    
    # 각 라벨에 대해 최종 결정 (ner 우선)
    for label, decisions in label_to_decisions.items():
        if len(decisions) == 1:
            final_decisions.append(decisions[0])
        else:
            # ner 결과 우선, 없으면 regular 결과
            ner_d = next((d for d in decisions if d.source == "ner"), None)
            if ner_d:
                final_decisions.append(ner_d)
            else:
                final_decisions.append(decisions[0])  # regular 결과
    
    return final_decisions
