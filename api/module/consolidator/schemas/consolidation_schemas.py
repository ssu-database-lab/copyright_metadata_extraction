#!/usr/bin/env python3
"""
Consolidation Schemas for Qwen3-Next-80B

Defines the JSON schema for consolidation output format
"""

from typing import Dict, Any

class ConsolidationSchemas:
    """Schemas for consolidation tasks"""
    
    @staticmethod
    def get_consolidation_schema() -> Dict[str, Any]:
        """
        Get the schema for consolidation output
        
        Returns:
            JSON schema for consolidation result
        """
        return {
            "type": "object",
            "properties": {
                "consolidated_metadata": {
                    "type": "object",
                    "description": "최종 통합된 메타데이터",
                    "additionalProperties": True
                },
                "decisions": {
                    "type": "array",
                    "description": "각 필드에 대한 결정 사항",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {
                                "type": "string",
                                "description": "필드 이름"
                            },
                            "llm_value": {
                                "description": "LLM에서 추출한 값 (null 가능)"
                            },
                            "ner_value": {
                                "description": "NER에서 추출한 값 (null 가능)"
                            },
                            "final_value": {
                                "description": "최종 선택된 값"
                            },
                            "decision": {
                                "type": "string",
                                "enum": ["AGREED", "CONFLICT", "LLM_ONLY", "NER_ONLY", "MISSING"],
                                "description": "결정 유형: AGREED(일치), CONFLICT(충돌), LLM_ONLY(LLM만), NER_ONLY(NER만), MISSING(둘다없음)"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "결정 이유 (한국어)"
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "신뢰도 점수"
                            },
                            "evidence": {
                                "type": "object",
                                "properties": {
                                    "llm_confidence": {
                                        "type": "number",
                                        "description": "LLM 신뢰도"
                                    },
                                    "ner_confidence": {
                                        "type": "number",
                                        "description": "NER 신뢰도"
                                    },
                                    "ocr_excerpt": {
                                        "type": "string",
                                        "description": "OCR 텍스트 일부 (관련 부분)"
                                    }
                                }
                            }
                        },
                        "required": ["field", "decision", "final_value", "reasoning", "confidence"]
                    }
                },
                "summary": {
                    "type": "object",
                    "properties": {
                        "total_fields": {
                            "type": "integer",
                            "description": "총 필드 수"
                        },
                        "agreed_fields": {
                            "type": "integer",
                            "description": "일치한 필드 수"
                        },
                        "conflicted_fields": {
                            "type": "integer",
                            "description": "충돌한 필드 수"
                        },
                        "llm_only_fields": {
                            "type": "integer",
                            "description": "LLM에만 있는 필드 수"
                        },
                        "ner_only_fields": {
                            "type": "integer",
                            "description": "NER에만 있는 필드 수"
                        },
                        "missing_fields": {
                            "type": "integer",
                            "description": "둘 다 없는 필드 수"
                        },
                        "overall_confidence": {
                            "type": "number",
                            "description": "전체 신뢰도 점수"
                        }
                    },
                    "required": ["total_fields", "agreed_fields", "conflicted_fields"]
                }
            },
            "required": ["consolidated_metadata", "decisions", "summary"]
        }
    
    @staticmethod
    def get_field_decision_schema() -> Dict[str, Any]:
        """Get schema for a single field decision"""
        return {
            "type": "object",
            "properties": {
                "field": {"type": "string"},
                "llm_value": {},
                "ner_value": {},
                "final_value": {},
                "decision": {
                    "type": "string",
                    "enum": ["AGREED", "CONFLICT", "LLM_ONLY", "NER_ONLY", "MISSING"]
                },
                "reasoning": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["field", "decision", "final_value", "reasoning", "confidence"]
        }

