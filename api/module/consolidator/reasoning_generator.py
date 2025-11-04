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
        """
        Generate evidence object for a consolidation decision
        
        Args:
            field_name: Name of the field
            llm_value: Value from LLM extraction
            ner_value: Value from NER extraction
            final_value: Final selected value
            decision: Decision type (AGREED, CONFLICT, etc.)
            ocr_text: Original OCR text
            confidence: Confidence score
        
        Returns:
            Evidence dictionary
        """
        evidence = {
            "field": field_name,
            "llm_value": llm_value,
            "ner_value": ner_value,
            "final_value": final_value,
            "decision": decision,
            "confidence": confidence,
            "reasoning": self._generate_reasoning(
                field_name, llm_value, ner_value, final_value, decision
            ),
            "ocr_excerpt": self._extract_ocr_excerpt(field_name, ocr_text) if ocr_text else None
        }
        
        return evidence
    
    def _generate_reasoning(
        self,
        field_name: str,
        llm_value: Any,
        ner_value: Any,
        final_value: Any,
        decision: str
    ) -> str:
        """Generate Korean reasoning explanation"""
        
        if decision == "AGREED":
            return f"LLM과 NER 모두 '{final_value}' 값을 추출했습니다. 높은 신뢰도입니다."
        
        elif decision == "CONFLICT":
            return f"LLM은 '{llm_value}', NER은 '{ner_value}'를 추출했습니다. {final_value}를 선택했습니다."
        
        elif decision == "LLM_ONLY":
            return f"NER에서 해당 필드를 찾을 수 없어 LLM 값 '{final_value}'를 사용했습니다."
        
        elif decision == "NER_ONLY":
            return f"LLM에서 해당 필드를 찾을 수 없어 NER 값 '{final_value}'를 사용했습니다."
        
        else:
            return f"필드 '{field_name}'에 대한 처리 결과입니다."
    
    def _extract_ocr_excerpt(
        self,
        field_name: str,
        ocr_text: str,
        context_window: int = 50
    ) -> Optional[str]:
        """
        Extract relevant excerpt from OCR text
        
        Phase 1: Simple implementation
        Phase 2: Will add intelligent context extraction
        """
        if not ocr_text or not field_name:
            return None
        
        # Simple: find field name in text and extract surrounding context
        field_lower = field_name.lower()
        text_lower = ocr_text.lower()
        
        idx = text_lower.find(field_lower)
        if idx == -1:
            return None
        
        start = max(0, idx - context_window)
        end = min(len(ocr_text), idx + len(field_name) + context_window)
        
        excerpt = ocr_text[start:end]
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(ocr_text):
            excerpt = excerpt + "..."
        
        return excerpt.strip()

