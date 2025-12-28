#!/usr/bin/env python3
"""
Confidence Scoring System for Metadata Extraction
각 메타데이터 필드에 대한 신뢰도(confidence) 점수를 계산합니다.
"""
import re
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from collections import Counter


class ConfidenceScorer:
    """메타데이터 추출 결과에 대한 신뢰도 점수 계산"""
    
    # 필드별 가중치 (저작권 메타데이터에서의 중요도 기반)
    FIELD_IMPORTANCE = {
        # 필수 필드 (높은 가중치)
        "contract_type": 1.0,
        "rights_holder": 1.0,
        "user": 1.0,
        "granted_rights": 0.95,
        "signature_date": 0.9,
        
        # 중요 필드
        "work_title": 0.85,
        "payment_amount": 0.8,
        "contract_duration": 0.8,
        "parties": 0.85,
        
        # 선택 필드
        "work_category": 0.6,
        "special_terms": 0.5,
        "address": 0.5,
    }
    
    def __init__(self):
        """초기화"""
        pass
    
    def calculate_ner_confidence(
        self, 
        entity: str, 
        entity_type: str,
        context: str = "",
        model_logits: Optional[List[float]] = None
    ) -> float:
        """
        NER 추출 결과의 신뢰도 계산
        
        Args:
            entity: 추출된 엔티티 텍스트
            entity_type: 엔티티 타입 (NAME, DATE, COMPANY 등)
            context: 엔티티가 추출된 문맥
            model_logits: 모델 출력 logits (optional)
            
        Returns:
            0.0 ~ 1.0 사이의 신뢰도 점수
        """
        confidence_scores = []
        
        # 1. 패턴 매칭 신뢰도
        pattern_conf = self._pattern_confidence(entity, entity_type)
        confidence_scores.append(pattern_conf)
        
        # 2. 길이 기반 신뢰도 (너무 짧거나 긴 엔티티는 낮은 점수)
        length_conf = self._length_confidence(entity, entity_type)
        confidence_scores.append(length_conf)
        
        # 3. 문맥 일관성 신뢰도
        if context:
            context_conf = self._context_confidence(entity, entity_type, context)
            confidence_scores.append(context_conf)
        
        # 4. 모델 logits 기반 신뢰도
        if model_logits:
            logit_conf = self._logits_confidence(model_logits)
            confidence_scores.append(logit_conf * 1.2)  # 높은 가중치
        
        # 가중 평균
        return np.mean(confidence_scores)
    
    def calculate_llm_confidence(
        self,
        field_value: Any,
        field_name: str,
        schema_type: str,
        llm_response: Optional[Dict] = None
    ) -> float:
        """
        LLM 추출 결과의 신뢰도 계산
        
        Args:
            field_value: 추출된 필드 값
            field_name: 필드 이름
            schema_type: 스키마 타입 (contract, consent 등)
            llm_response: LLM 응답 전체 (logprobs 포함)
            
        Returns:
            0.0 ~ 1.0 사이의 신뢰도 점수
        """
        confidence_scores = []
        
        # 1. 필드 값 유효성 신뢰도
        validity_conf = self._field_validity_confidence(field_value, field_name)
        confidence_scores.append(validity_conf)
        
        # 2. 스키마 준수 신뢰도
        schema_conf = self._schema_compliance_confidence(field_value, field_name, schema_type)
        confidence_scores.append(schema_conf)
        
        # 3. LLM logprobs 기반 신뢰도
        if llm_response and "logprobs" in llm_response:
            logprobs_conf = self._llm_logprobs_confidence(llm_response["logprobs"])
            confidence_scores.append(logprobs_conf * 1.3)  # 높은 가중치
        
        # 4. 필드 중요도 가중치 적용
        importance = self.FIELD_IMPORTANCE.get(field_name, 0.5)
        base_confidence = np.mean(confidence_scores)
        
        return base_confidence * (0.7 + 0.3 * importance)
    
    def calculate_consolidated_confidence(
        self,
        ner_value: Any,
        ner_confidence: float,
        llm_value: Any,
        llm_confidence: float,
        final_value: Any,
        validation_result: Dict
    ) -> float:
        """
        통합(consolidation) 후 최종 신뢰도 계산
        
        Args:
            ner_value: NER 추출 값
            ner_confidence: NER 신뢰도
            llm_value: LLM 추출 값
            llm_confidence: LLM 신뢰도
            final_value: 최종 선택된 값
            validation_result: 검증 결과
            
        Returns:
            0.0 ~ 1.0 사이의 최종 신뢰도 점수
        """
        # 1. 출처별 신뢰도
        if final_value == ner_value and final_value == llm_value:
            # 양쪽 일치 -> 높은 신뢰도
            agreement_bonus = 0.2
            base_conf = max(ner_confidence, llm_confidence)
        elif final_value == ner_value:
            base_conf = ner_confidence
            agreement_bonus = -0.1  # 불일치 페널티
        elif final_value == llm_value:
            base_conf = llm_confidence
            agreement_bonus = -0.1
        else:
            # 새로운 값 생성 (consolidation agent의 결정)
            base_conf = (ner_confidence + llm_confidence) / 2
            agreement_bonus = -0.05
        
        # 2. 검증 결과 반영
        validation_bonus = 0.0
        if validation_result.get("is_valid", False):
            validation_bonus = 0.15
        elif validation_result.get("has_warnings", False):
            validation_bonus = -0.1
        
        # 3. 최종 신뢰도 계산
        final_confidence = min(1.0, max(0.0, base_conf + agreement_bonus + validation_bonus))
        
        return final_confidence
    
    def calculate_document_confidence(
        self,
        field_confidences: Dict[str, float]
    ) -> Dict[str, float]:
        """
        문서 전체 신뢰도 계산
        
        Args:
            field_confidences: 필드별 신뢰도 딕셔너리
            
        Returns:
            전체 통계 (평균, 최소, 필수필드 평균 등)
        """
        if not field_confidences:
            return {
                "overall_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "required_fields_confidence": 0.0,
                "quality_grade": "F"
            }
        
        values = list(field_confidences.values())
        overall_conf = np.mean(values)
        min_conf = np.min(values)
        max_conf = np.max(values)
        
        # 필수 필드만 신뢰도 계산
        required_fields = ["contract_type", "rights_holder", "user", "granted_rights", "signature_date"]
        required_confs = [field_confidences.get(f, 0.0) for f in required_fields if f in field_confidences]
        required_avg = np.mean(required_confs) if required_confs else 0.0
        
        # 품질 등급
        if overall_conf >= 0.9 and min_conf >= 0.7:
            grade = "A"
        elif overall_conf >= 0.8 and min_conf >= 0.6:
            grade = "B"
        elif overall_conf >= 0.7 and min_conf >= 0.5:
            grade = "C"
        elif overall_conf >= 0.6:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "overall_confidence": float(overall_conf),
            "min_confidence": float(min_conf),
            "max_confidence": float(max_conf),
            "required_fields_confidence": float(required_avg),
            "quality_grade": grade,
            "total_fields": len(field_confidences),
            "low_confidence_fields": [k for k, v in field_confidences.items() if v < 0.5]
        }
    
    # ==================== Private Helper Methods ====================
    
    def _pattern_confidence(self, entity: str, entity_type: str) -> float:
        """패턴 매칭 기반 신뢰도"""
        patterns = {
            "DATE": [
                r'\d{4}[년.-]\s*\d{1,2}[월.-]\s*\d{1,2}',
                r'\d{4}[-./]\d{2}[-./]\d{2}',
                r'\d{4}\.\s*\d{1,2}\.\s*\d{1,2}'
            ],
            "PHONE": [
                r'\d{2,3}[-\s]\d{3,4}[-\s]\d{4}',
                r'\d{10,11}'
            ],
            "EMAIL": [
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            ],
            "MONEY": [
                r'\d{1,3}(,\d{3})*원',
                r'\d+원'
            ],
            "ADDRESS": [
                r'.*[시도군구].*(로길동)'
            ]
        }
        
        if entity_type not in patterns:
            return 0.7  # 기본 점수
        
        for pattern in patterns[entity_type]:
            if re.search(pattern, entity):
                return 0.95
        
        return 0.5  # 패턴 불일치
    
    def _length_confidence(self, entity: str, entity_type: str) -> float:
        """길이 기반 신뢰도"""
        length = len(entity.strip())
        
        # 타입별 적정 길이 범위
        length_ranges = {
            "NAME": (2, 20),
            "COMPANY": (2, 50),
            "DATE": (8, 30),
            "PHONE": (9, 20),
            "EMAIL": (5, 50),
            "ADDRESS": (5, 100),
            "MONEY": (1, 20)
        }
        
        if entity_type not in length_ranges:
            return 0.8
        
        min_len, max_len = length_ranges[entity_type]
        
        if min_len <= length <= max_len:
            return 1.0
        elif length < min_len:
            return max(0.3, 0.5 + (length / min_len) * 0.5)
        else:  # length > max_len
            return max(0.3, 1.0 - (length - max_len) / max_len * 0.5)
    
    def _context_confidence(self, entity: str, entity_type: str, context: str) -> float:
        """문맥 일관성 신뢰도"""
        context_keywords = {
            "NAME": ["이름", "성명", "권리자", "저작자"],
            "COMPANY": ["회사", "법인", "단체", "기관"],
            "DATE": ["일자", "날짜", "년", "월", "일"],
            "PHONE": ["전화", "연락처", "Tel"],
            "ADDRESS": ["주소", "소재지", "거주지"],
            "RIGHT_INFO": ["권리", "저작", "이용", "허락"]
        }
        
        if entity_type not in context_keywords:
            return 0.7
        
        keywords = context_keywords[entity_type]
        context_lower = context.lower()
        
        matches = sum(1 for kw in keywords if kw in context_lower)
        
        if matches >= 2:
            return 0.95
        elif matches == 1:
            return 0.85
        else:
            return 0.6
    
    def _logits_confidence(self, logits: List[float]) -> float:
        """모델 logits 기반 신뢰도"""
        if not logits:
            return 0.5
        
        # Softmax 확률 계산
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        max_prob = np.max(probs)
        
        # 최대 확률을 신뢰도로 변환
        return float(max_prob)
    
    def _field_validity_confidence(self, value: Any, field_name: str) -> float:
        """필드 값 유효성 신뢰도"""
        if value is None or value == "":
            return 0.0
        
        # Boolean 필드
        if isinstance(value, bool):
            return 1.0
        
        # String 필드
        if isinstance(value, str):
            if len(value.strip()) == 0:
                return 0.0
            if len(value) > 500:  # 너무 긴 텍스트
                return 0.6
            return 0.9
        
        # Number 필드
        if isinstance(value, (int, float)):
            if value < 0 and field_name == "payment_amount":
                return 0.3  # 음수 금액은 의심스러움
            return 0.95
        
        # Array 필드
        if isinstance(value, list):
            if len(value) == 0:
                return 0.5
            return 0.85
        
        return 0.7
    
    def _schema_compliance_confidence(self, value: Any, field_name: str, schema_type: str) -> float:
        """스키마 준수 신뢰도"""
        # 간단한 타입 체크
        expected_types = {
            "contract_type": str,
            "rights_holder": str,
            "payment_amount": (int, float),
            "signature_date": str,
            "granted_rights": (list, dict)
        }
        
        if field_name in expected_types:
            expected = expected_types[field_name]
            if isinstance(value, expected):
                return 1.0
            else:
                return 0.5
        
        return 0.8
    
    def _llm_logprobs_confidence(self, logprobs: Dict) -> float:
        """LLM logprobs 기반 신뢰도"""
        # OpenAI style logprobs 처리
        if "token_logprobs" in logprobs:
            token_probs = logprobs["token_logprobs"]
            if token_probs:
                avg_logprob = np.mean([lp for lp in token_probs if lp is not None])
                # logprob을 확률로 변환 (exp)
                return float(min(1.0, np.exp(avg_logprob)))
        
        return 0.7


def add_confidence_to_extraction_result(
    extraction_result: Dict,
    ner_result: Optional[Dict] = None,
    llm_result: Optional[Dict] = None,
    validation_result: Optional[Dict] = None
) -> Dict:
    """
    추출 결과에 confidence scores 추가
    
    Args:
        extraction_result: 최종 추출 결과
        ner_result: NER 추출 결과 (optional)
        llm_result: LLM 추출 결과 (optional)
        validation_result: 검증 결과 (optional)
        
    Returns:
        Confidence scores가 추가된 결과
    """
    scorer = ConfidenceScorer()
    
    # 필드별 confidence 계산
    field_confidences = {}
    
    for field_name, field_value in extraction_result.items():
        if field_name.startswith("_"):  # 메타 필드 스킵
            continue
        
        # NER confidence
        ner_conf = 0.5
        if ner_result and field_name in ner_result:
            ner_conf = scorer.calculate_ner_confidence(
                entity=str(ner_result[field_name]),
                entity_type=field_name.upper()
            )
        
        # LLM confidence
        llm_conf = 0.5
        if llm_result and field_name in llm_result:
            llm_conf = scorer.calculate_llm_confidence(
                field_value=llm_result[field_name],
                field_name=field_name,
                schema_type=extraction_result.get("_schema_type", "unknown")
            )
        
        # 최종 confidence
        final_conf = scorer.calculate_consolidated_confidence(
            ner_value=ner_result.get(field_name) if ner_result else None,
            ner_confidence=ner_conf,
            llm_value=llm_result.get(field_name) if llm_result else None,
            llm_confidence=llm_conf,
            final_value=field_value,
            validation_result=validation_result or {}
        )
        
        field_confidences[field_name] = final_conf
    
    # 문서 전체 confidence
    doc_confidence = scorer.calculate_document_confidence(field_confidences)
    
    # 결과에 추가
    result_with_confidence = extraction_result.copy()
    result_with_confidence["_confidence_scores"] = {
        "fields": field_confidences,
        "document": doc_confidence
    }
    
    return result_with_confidence


if __name__ == "__main__":
    # 테스트
    scorer = ConfidenceScorer()
    
    # NER confidence 테스트
    ner_conf = scorer.calculate_ner_confidence(
        entity="2024년 1월 15일",
        entity_type="DATE",
        context="계약 체결 일자: 2024년 1월 15일"
    )
    print(f"NER Confidence (DATE): {ner_conf:.2%}")
    
    # LLM confidence 테스트
    llm_conf = scorer.calculate_llm_confidence(
        field_value="저작재산권 양도 계약서",
        field_name="contract_type",
        schema_type="contract"
    )
    print(f"LLM Confidence (contract_type): {llm_conf:.2%}")
    
    # 통합 confidence 테스트
    consolidated_conf = scorer.calculate_consolidated_confidence(
        ner_value="홍길동",
        ner_confidence=0.85,
        llm_value="홍길동",
        llm_confidence=0.92,
        final_value="홍길동",
        validation_result={"is_valid": True}
    )
    print(f"Consolidated Confidence: {consolidated_conf:.2%}")
    
    # 문서 confidence 테스트
    field_confs = {
        "contract_type": 0.95,
        "rights_holder": 0.88,
        "user": 0.91,
        "signature_date": 0.85,
        "payment_amount": 0.72
    }
    doc_conf = scorer.calculate_document_confidence(field_confs)
    print(f"\nDocument Confidence: {doc_conf}")
