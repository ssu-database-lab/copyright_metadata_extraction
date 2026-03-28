#!/usr/bin/env python3
"""
Reasoning Generator - Generates evidence and reasoning for consolidation decisions

This module provides evidence generation and reasoning explanation
for the consolidation process.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ReasoningGenerator:
    """Generates evidence and reasoning for consolidation decisions"""
    
    def __init__(self):
        """Initialize reasoning generator"""
        pass
    
    def generate_evidence(
        self,
        field_name: str,
        llm_value: Any,
        ner_value: Any,
        final_value: Any,
        decision: str,
        ocr_text: str = "",
        confidence: float = 0.0
    ) -> Dict[str, Any]:
        """Generate evidence object for a consolidation decision."""
        evidence = {
            "field": field_name,
            "llm_value": llm_value,
            "ner_value": ner_value,
            "final_value": final_value,
            "decision": decision,
            "confidence": confidence,
            "reasoning": self._generate_reasoning(
                field_name, llm_value, ner_value, final_value, decision, confidence
            ),
            "ocr_excerpt": self._extract_ocr_excerpt(
                final_value, ocr_text, llm_value, ner_value
            ) if ocr_text else None
        }
        return evidence

    def _generate_reasoning(
        self,
        field_name: str,
        llm_value: Any,
        ner_value: Any,
        final_value: Any,
        decision: str,
        confidence: float = 0.0
    ) -> str:
        """Generate Korean reasoning explanation with confidence-aware messaging."""

        if decision == "AGREED":
            if confidence >= 0.9:
                return f"LLM과 NER 모두 '{final_value}' 값을 추출했습니다. 매우 높은 신뢰도입니다."
            elif confidence >= 0.7:
                return f"LLM과 NER이 유사한 값을 추출했습니다 ('{final_value}'). 신뢰도 양호."
            else:
                return f"LLM과 NER 결과가 일치하나 신뢰도가 낮습니다 ({confidence:.0%}). '{final_value}' 사용."

        elif decision == "CONFLICT":
            return (
                f"LLM은 '{llm_value}', NER은 '{ner_value}'를 추출했습니다. "
                f"신뢰도 비교 후 '{final_value}'를 선택했습니다."
            )

        elif decision == "LLM_ONLY":
            return f"NER에서 해당 필드를 추출하지 못해 LLM 값 '{final_value}'를 사용했습니다."

        elif decision == "NER_ONLY":
            return f"LLM에서 해당 필드를 추출하지 못해 NER 값 '{final_value}'를 사용했습니다."

        elif decision == "MISSING":
            return f"LLM과 NER 모두 '{field_name}' 필드를 추출하지 못했습니다."

        else:
            return f"필드 '{field_name}'에 대한 처리 결과입니다."

    def _extract_ocr_excerpt(
        self,
        final_value: Any,
        ocr_text: str,
        llm_value: Any = None,
        ner_value: Any = None,
        context_window: int = 100
    ) -> Optional[str]:
        """Extract relevant excerpt from OCR text by searching for the actual value.

        Searches for the final_value (or llm/ner values as fallback) in the OCR
        text and returns surrounding context. This finds real evidence instead of
        searching for English field names in Korean text.
        """
        if not ocr_text:
            return None

        # Build search terms: try final_value first, then llm/ner values
        search_terms = []
        for val in [final_value, llm_value, ner_value]:
            if val is not None and val != "":
                str_val = str(val).strip()
                if len(str_val) >= 2:  # skip single characters
                    search_terms.append(str_val)

        if not search_terms:
            return None

        # Search for each term in the OCR text
        for term in search_terms:
            idx = ocr_text.find(term)
            if idx == -1:
                continue

            start = max(0, idx - context_window)
            end = min(len(ocr_text), idx + len(term) + context_window)

            excerpt = ocr_text[start:end]
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(ocr_text):
                excerpt = excerpt + "..."

            return excerpt.strip()

        return None

