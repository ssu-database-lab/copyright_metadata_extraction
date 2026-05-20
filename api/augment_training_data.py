#!/usr/bin/env python3
"""
훈련 데이터 대폭 증강 스크립트
- auto_extracted_ground_truth.json의 모든 샘플 활용
- 다양한 데이터 증강 기법 적용
- 최대한 많은 훈련 샘플 생성
"""
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import re

# 엔티티 타입별 대체 가능한 값들 (데이터 증강용)
ENTITY_REPLACEMENTS = {
    "PHONE": [
        "010-1234-5678", "010-9876-5432", "010-5555-1234", 
        "010-7777-8888", "010-2222-3333", "02-1234-5678",
        "031-123-4567", "010-4567-8901", "010-8765-4321"
    ],
    "DATE": [
        "2024.01.15", "2023.12.31", "2024.03.20", "2023.06.15",
        "2024.05.10", "2023.08.25", "2024.02.14", "2023.11.11",
        "2024.04.01", "2023.09.30", "2024.07.07", "2023.10.10"
    ],
    "MONEY": [
        "1,000,000원", "500,000원", "2,000,000원", "3,500,000원",
        "750,000원", "5,000,000원", "10,000,000원", "250,000원"
    ],
    "EMAIL": [
        "test@example.com", "user@company.co.kr", "admin@domain.com",
        "contact@office.kr", "info@business.com"
    ]
}

def tokenize_to_chars(text: str) -> List[str]:
    """텍스트를 문자 단위로 토큰화 (공백 포함)"""
    return list(text)

def create_bio_sample(text: str, entities: Dict[str, List[str]]) -> Dict:
    """텍스트와 엔티티로부터 BIO 형식 샘플 생성"""
    tokens = tokenize_to_chars(text)
    labels = ['O'] * len(tokens)
    
    # 각 엔티티 타입별로 처리
    for entity_type, entity_list in entities.items():
        if entity_type in ['N/A', 'metadata']:
            continue
        
        for entity_value in entity_list:
            if not entity_value or entity_value == 'N/A':
                continue
            
            # 텍스트에서 엔티티 찾기 (모든 출현 위치)
            entity_tokens = tokenize_to_chars(entity_value)
            entity_str = ''.join(entity_tokens)
            text_str = ''.join(tokens)
            
            start_pos = 0
            while True:
                pos = text_str.find(entity_str, start_pos)
                if pos == -1:
                    break
                
                # BIO 태깅
                if labels[pos] == 'O':  # 중복 태깅 방지
                    labels[pos] = f'B-{entity_type}'
                    for i in range(pos + 1, min(pos + len(entity_tokens), len(labels))):
                        if labels[i] == 'O':
                            labels[i] = f'I-{entity_type}'
                
                start_pos = pos + 1
    
    return {
        "tokens": tokens,
        "labels": labels
    }

def split_into_sentences(text: str, max_length: int = 150) -> List[str]:
    """텍스트를 문장 단위로 분할"""
    # 문장 종결 기호로 분할
    sentences = re.split(r'([.!?。]\s*)', text)
    
    result = []
    current = ""
    
    for i in range(0, len(sentences), 2):
        sent = sentences[i]
        punct = sentences[i+1] if i+1 < len(sentences) else ""
        full_sent = sent + punct
        
        if len(current) + len(full_sent) > max_length:
            if current:
                result.append(current.strip())
            current = full_sent
        else:
            current += full_sent
    
    if current:
        result.append(current.strip())
    
    return result

def augment_with_entity_replacement(sample: Dict, entity_type: str) -> List[Dict]:
    """특정 엔티티 타입을 다른 값으로 대체하여 증강"""
    augmented = []
    
    if entity_type not in ENTITY_REPLACEMENTS:
        return augmented
    
    text = ''.join(sample['tokens'])
    
    # 원본에서 해당 엔티티 타입 찾기
    original_entities = []
    i = 0
    while i < len(sample['labels']):
        if sample['labels'][i].startswith(f'B-{entity_type}'):
            start = i
            i += 1
            while i < len(sample['labels']) and sample['labels'][i] == f'I-{entity_type}':
                i += 1
            entity_value = ''.join(sample['tokens'][start:i])
            original_entities.append(entity_value)
        else:
            i += 1
    
    # 엔티티를 새 값으로 대체
    for replacement in ENTITY_REPLACEMENTS[entity_type][:3]:  # 최대 3개 변형
        new_text = text
        for orig_entity in original_entities:
            new_text = new_text.replace(orig_entity, replacement, 1)
        
        if new_text != text:
            # 새로운 샘플 생성 (간단한 토큰/라벨 생성)
            new_tokens = tokenize_to_chars(new_text)
            new_labels = ['O'] * len(new_tokens)
            
            # 대체된 엔티티에 라벨링
            for i, token in enumerate(new_tokens):
                if i < len(sample['labels']):
                    # 기존 라벨 구조 유지하되, 대체된 부분만 업데이트
                    new_labels[i] = sample['labels'][i]
            
            augmented.append({
                "tokens": new_tokens,
                "labels": new_labels
            })
    
    return augmented

def add_noise_to_sample(sample: Dict) -> Dict:
    """샘플에 노이즈 추가 (OCR 오류 시뮬레이션)"""
    tokens = sample['tokens'][:]
    labels = sample['labels'][:]
    
    # 10% 확률로 문자 변경
    for i in range(len(tokens)):
        if random.random() < 0.05 and tokens[i].strip():  # 5% 노이즈
            # 유사한 문자로 대체
            similar_chars = {
                'o': '0', '0': 'o', 'l': '1', '1': 'l',
                'O': '0', 'I': '1', 'S': '5', 's': '5'
            }
            if tokens[i] in similar_chars:
                tokens[i] = similar_chars[tokens[i]]
    
    return {
        "tokens": tokens,
        "labels": labels
    }

def shuffle_sample_order(samples: List[Dict]) -> List[Dict]:
    """샘플 순서 무작위 섞기"""
    shuffled = samples[:]
    random.shuffle(shuffled)
    return shuffled

def convert_to_bio_format(samples: List[Dict]) -> str:
    """BIO 샘플들을 훈련 형식으로 변환"""
    lines = []
    
    for sample in samples:
        tokens = sample['tokens']
        labels = sample['labels']
        
        for token, label in zip(tokens, labels):
            if token.strip():  # 공백이 아닌 토큰만
                lines.append(f"{token}\t{label}")
        lines.append("")  # 샘플 구분용 빈 줄
    
    return '\n'.join(lines)

def main():
    print("=" * 80)
    print("훈련 데이터 대폭 증강 시작")
    print("=" * 80)
    
    # 입력 파일 로드
    input_path = Path("data/in/auto_extracted_ground_truth.json")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_samples = data['samples']
    print(f"\n원본 샘플 수: {len(original_samples)}")
    
    all_training_samples = []
    
    # 1. 원본 샘플을 BIO 형식으로 변환
    print("\n[1단계] 원본 샘플 BIO 변환...")
    for idx, sample in enumerate(original_samples):
        if 'tokens' in sample and 'labels' in sample:
            all_training_samples.append({
                "tokens": sample['tokens'],
                "labels": sample['labels']
            })
            print(f"  샘플 {idx+1}: {len(sample['tokens'])} 토큰")
    
    print(f"  → BIO 변환 완료: {len(all_training_samples)} 샘플")
    
    # 2. 각 샘플을 문장 단위로 분할하여 더 많은 샘플 생성
    print("\n[2단계] 문장 단위 분할...")
    split_samples = []
    for sample in all_training_samples:
        text = ''.join(sample['tokens'])
        sentences = split_into_sentences(text, max_length=100)
        
        for sent in sentences:
            if len(sent.strip()) > 20:  # 너무 짧은 문장 제외
                sent_tokens = tokenize_to_chars(sent)
                # 기존 라벨에서 해당 부분 추출
                start_idx = text.find(sent)
                if start_idx >= 0:
                    end_idx = start_idx + len(sent)
                    sent_labels = sample['labels'][start_idx:end_idx]
                    
                    if len(sent_tokens) == len(sent_labels):
                        split_samples.append({
                            "tokens": sent_tokens,
                            "labels": sent_labels
                        })
    
    print(f"  → 분할 완료: {len(split_samples)} 샘플 추가")
    all_training_samples.extend(split_samples)
    
    # 3. 데이터 증강 - 엔티티 교체
    print("\n[3단계] 엔티티 교체 증강...")
    augmented_samples = []
    for sample in all_training_samples[:50]:  # 처음 50개만
        for entity_type in ['PHONE', 'DATE']:
            augmented = augment_with_entity_replacement(sample, entity_type)
            augmented_samples.extend(augmented)
    
    print(f"  → 증강 완료: {len(augmented_samples)} 샘플 추가")
    all_training_samples.extend(augmented_samples)
    
    # 4. OCR 노이즈 시뮬레이션
    print("\n[4단계] OCR 노이즈 추가...")
    noisy_samples = []
    for sample in random.sample(all_training_samples, min(100, len(all_training_samples))):
        noisy = add_noise_to_sample(sample)
        noisy_samples.append(noisy)
    
    print(f"  → 노이즈 샘플: {len(noisy_samples)} 샘플 추가")
    all_training_samples.extend(noisy_samples)
    
    # 5. 샘플 순서 무작위화
    print("\n[5단계] 샘플 섞기...")
    all_training_samples = shuffle_sample_order(all_training_samples)
    
    # 6. BIO 형식으로 변환 및 저장
    print("\n[6단계] 훈련 파일 생성...")
    
    # 기존 파일에 추가
    output_files = [
        "data/in/realistic_train.txt",
        "data/in/real_document_train.txt"
    ]
    
    bio_text = convert_to_bio_format(all_training_samples)
    
    for output_file in output_files:
        output_path = Path(output_file)
        
        # 기존 내용 로드
        existing_content = ""
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        
        # 새 데이터 추가
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(existing_content)
            if existing_content and not existing_content.endswith('\n\n'):
                f.write('\n\n')
            f.write(bio_text)
        
        # 통계
        total_lines = len(existing_content.split('\n')) + len(bio_text.split('\n'))
        print(f"  ✅ {output_path.name}: {total_lines:,} 라인")
    
    # 7. 최종 통계
    print("\n" + "=" * 80)
    print("훈련 데이터 증강 완료!")
    print("=" * 80)
    print(f"총 생성된 샘플: {len(all_training_samples):,}개")
    
    # 엔티티 통계
    entity_counts = {}
    for sample in all_training_samples:
        for label in sample['labels']:
            if label.startswith('B-'):
                entity_type = label[2:]
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    print(f"\n엔티티 통계:")
    for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count:,}개")
    
    print("\n✅ 완료!")

if __name__ == "__main__":
    random.seed(42)  # 재현성을 위한 시드
    main()
