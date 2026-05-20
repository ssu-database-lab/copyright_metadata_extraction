#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER 모델로 텍스트 추론 및 라벨 검증 스크립트
- 현재 훈련된 bert-base-multilingual-cased 모델 사용
- 텍스트의 NER 결과 출력
- 추출된 라벨 정리 (O/X)
"""

import sys
import json
from pathlib import Path
import torch

# Add api folder to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ner.ner_constants import ENTITY_TYPES, LABEL_TO_ID, ID_TO_LABEL
from src.ner.ner_system import load_model_for_inference, predict_texts


def main():
    print("=" * 80)
    print("NER 모델 테스트 - 라벨 추출 결과 분석")
    print("=" * 80)
    
    # 모델 로드
    model_dir = Path(__file__).parent / "models" / "bert-base-multilingual-cased"
    if not model_dir.exists():
        print(f"❌ 모델 디렉토리를 찾을 수 없습니다: {model_dir}")
        sys.exit(1)
    
    print(f"\n📁 모델 경로: {model_dir}")
    print(f"✅ 모델 로딩 중...\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, id2label, config, device = load_model_for_inference(str(model_dir), device)
    
    print(f"✅ 모델 로드 완료 (Device: {device})")
    print(f"✅ 엔티티 타입: {len(ENTITY_TYPES)}개")
    print(f"✅ 라벨 총 개수: {len(ID_TO_LABEL)}개\n")
    
    # 테스트 텍스트들
    test_texts = [
        "정보통신네트워크 이용촉진 및 정보보호 등에 관한 법률 제52조의2 제1항의 규정에 따라 성명 홍길동, 주소 서울시 강남구 테헤란로, 연락처 010-1234-5678로 본인은 동의합니다.",
        "계약금액: 50,000,000원, 계약일: 2024년 3월 19일, 계약자: (주)테크회사, 담당자 김철수, 이메일 kim@company.com, 직위 과장",
        "양도인: 이순신, 양수인: 강감찬, 계약유형: 저작권양도, 계약기간: 1년, 법령근거: 저작권법 제44조"
    ]
    
    # 모든 엔티티 타입 초기화
    extracted_entities = {entity_type: False for entity_type in ENTITY_TYPES}
    
    print("=" * 80)
    print("테스트 텍스트 NER 결과")
    print("=" * 80)
    
    for idx, text in enumerate(test_texts, 1):
        print(f"\n[테스트 {idx}]")
        print(f"원문: {text}\n")
        
        # 추론
        results = predict_texts(model, tokenizer, id2label, config, [text], device=device)
        result = results[0]
        
        # 토큰과 라벨 출력
        print("추출된 엔티티:")
        token_labels = result["tokens"]
        
        extracted_labels = set()
        current_entity = None
        current_label = None
        
        for token, label in token_labels:
            if label != "O":
                extracted_labels.add(label)
                
                # 엔티티 타입 추출 (B-NAME -> NAME)
                entity_type = label.split('-')[1]
                extracted_entities[entity_type] = True
                
                # 엔티티 부분 출력
                if label.startswith("B-"):
                    if current_entity:
                        print(f"  - {current_entity} [{current_label}]")
                    current_entity = token
                    current_label = label
                elif label.startswith("I-"):
                    current_entity += token
                    
        if current_entity:
            print(f"  - {current_entity} [{current_label}]")
        
        if not extracted_labels:
            print("  (엔티티 없음)")
    
    # 전체 결과 정리
    print("\n" + "=" * 80)
    print("라벨 추출 결과 (O/X 정리)")
    print("=" * 80)
    
    extracted_count = 0
    print(f"\n{'라벨 이름':<20} {'추출 여부':<10} {'상태':<5}")
    print("-" * 40)
    
    for entity_type in ENTITY_TYPES:
        status = "O" if extracted_entities[entity_type] else "X"
        if extracted_entities[entity_type]:
            extracted_count += 1
        print(f"{entity_type:<20} {'추출됨' if extracted_entities[entity_type] else '미추출':<10} {status:<5}")
    
    print("-" * 40)
    print(f"{'합계':<20} {extracted_count}/{len(ENTITY_TYPES)}")
    print("=" * 80)
    
    # 요약
    print(f"\n📊 최종 결과:")
    print(f"  - 총 엔티티 타입: {len(ENTITY_TYPES)}개")
    print(f"  - 추출된 타입: {extracted_count}개")
    print(f"  - 미추출 타입: {len(ENTITY_TYPES) - extracted_count}개")
    print(f"  - 추출률: {(extracted_count / len(ENTITY_TYPES) * 100):.1f}%")
    
    # 추출된 엔티티 타입 나열
    extracted_types = [et for et in ENTITY_TYPES if extracted_entities[et]]
    print(f"\n✅ 추출된 엔티티 타입: {', '.join(extracted_types)}")
    
    # 미추출 엔티티 타입 나열
    not_extracted = [et for et in ENTITY_TYPES if not extracted_entities[et]]
    if not_extracted:
        print(f"❌ 미추출 엔티티 타입: {', '.join(not_extracted)}")


if __name__ == "__main__":
    main()
