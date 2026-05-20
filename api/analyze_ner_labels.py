#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NER 라벨 추출 현황 분석
- 현재 훈련된 모델에서 사용된 23개 라벨 분석
- 각 라벨별 train/validation/test 데이터셋에서의 사용 여부 확인
"""

from pathlib import Path
from collections import defaultdict

def extract_labels_from_file(file_path):
    """BIO 형식 파일에서 라벨 추출"""
    labels_found = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        token, label = parts
                        if label != 'O':
                            # B-NAME -> NAME, I-PHONE -> PHONE
                            entity_type = label.split('-')[1]
                            labels_found.add(entity_type)
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {file_path} - {e}")
    
    return labels_found


def main():
    print("=" * 90)
    print("📊 NER 모델 라벨 추출 현황 분석")
    print("=" * 90)
    
    # 23개 라벨 (정의된 ENTITY_TYPES)
    ALL_ENTITY_TYPES = [
        "NAME", "PHONE", "ADDRESS", "DATE", "COMPANY", "EMAIL", "POSITION",
        "CONTRACT_TYPE", "CONSENT_TYPE", "RIGHT_INFO", "MONEY", "PERIOD",
        "PROJECT_NAME", "LAW_REFERENCE", "ID_NUM", "TITLE", "URL",
        "DESCRIPTION", "TYPE", "STATUS", "DEPARTMENT", "LANGUAGE", "QUANTITY"
    ]
    
    print(f"\n📋 정의된 총 라벨 개수: {len(ALL_ENTITY_TYPES)}개")
    
    # 훈련 데이터 경로
    training_dir = Path("/home/peppermint/copyright_metadata_extraction/api/src/ner/training/google-bert-bert-base-multilingual-cased")
    
    train_file = training_dir / "train.txt"
    val_file = training_dir / "validation.txt"
    test_file = training_dir / "test.txt"
    
    # 각 파일에서 라벨 추출
    print(f"\n🔍 분석 중...")
    train_labels = extract_labels_from_file(train_file)
    val_labels = extract_labels_from_file(val_file)
    test_labels = extract_labels_from_file(test_file)
    
    # 전체 사용된 라벨
    all_used_labels = train_labels | val_labels | test_labels
    
    print(f"✓ train.txt: {sorted(train_labels)}")
    print(f"✓ validation.txt: {sorted(val_labels)}")
    print(f"✓ test.txt: {sorted(test_labels)}")
    
    # 결과 정렬 출력
    print("\n" + "=" * 90)
    print("라벨별 추출 현황 (O = 추출됨, X = 미추출)")
    print("=" * 90)
    print(f"\n{'#':<3} {'라벨명':<20} {'Train':<8} {'Validation':<12} {'Test':<8} {'최종':<6}")
    print("-" * 90)
    
    extracted_count = 0
    for idx, entity_type in enumerate(ALL_ENTITY_TYPES, 1):
        in_train = "O" if entity_type in train_labels else "X"
        in_val = "O" if entity_type in val_labels else "X"
        in_test = "O" if entity_type in test_labels else "X"
        in_all = "O" if entity_type in all_used_labels else "X"
        
        if in_all == "O":
            extracted_count += 1
        
        print(f"{idx:<3} {entity_type:<20} {in_train:<8} {in_val:<12} {in_test:<8} {in_all:<6}")
    
    print("=" * 90)
    
    # 최종 요약
    not_extracted = [et for et in ALL_ENTITY_TYPES if et not in all_used_labels]
    extracted_labels_list = [et for et in ALL_ENTITY_TYPES if et in all_used_labels]
    
    print(f"\n📈 최종 결과:")
    print(f"  • 총 정의된 라벨: {len(ALL_ENTITY_TYPES)}개")
    print(f"  • 훈련 데이터에서 사용된 라벨: {extracted_count}개 ({(extracted_count/len(ALL_ENTITY_TYPES)*100):.1f}%)")
    print(f"  • 사용되지 않은 라벨: {len(not_extracted)}개 ({(len(not_extracted)/len(ALL_ENTITY_TYPES)*100):.1f}%)")
    
    print(f"\n✅ 추출된 라벨 ({extracted_count}개):")
    print(f"   {', '.join(extracted_labels_list)}")
    
    if not_extracted:
        print(f"\n❌ 미추출 라벨 ({len(not_extracted)}개):")
        print(f"   {', '.join(not_extracted)}")
    
    print("\n" + "=" * 90)
    
    # 데이터셋별 통계
    print(f"\n📊 데이터셋별 통계:")
    print(f"  • Train 전용: {len(train_labels - val_labels - test_labels)}개")
    print(f"  • Validation 전용: {len(val_labels - train_labels - test_labels)}개")
    print(f"  • Test 전용: {len(test_labels - train_labels - val_labels)}개")
    print(f"  • 모든 데이터셋에 포함: {len(train_labels & val_labels & test_labels)}개")


if __name__ == "__main__":
    main()
