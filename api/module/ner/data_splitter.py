#!/usr/bin/env python3
"""
NER 훈련 데이터 분할 및 관리 시스템
Train/Validation/Test 분할 (80/10/10) with Stratified Sampling
"""

import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
from collections import defaultdict, Counter


def load_conll_data(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """CoNLL 형식 데이터 로드"""
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:  # 빈 줄 = 문장 구분
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, label = parts
                    current_sentence.append(token)
                    current_labels.append(label)
        
        # 마지막 문장 처리
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
    
    return sentences, labels


def save_conll_data(sentences: List[List[str]], labels: List[List[str]], file_path: str):
    """CoNLL 형식으로 데이터 저장"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence, label_seq in zip(sentences, labels):
            for token, label in zip(sentence, label_seq):
                f.write(f"{token}\t{label}\n")
            f.write("\n")


def get_entity_types_in_sentence(labels: List[str]) -> set:
    """문장에 포함된 엔티티 타입 추출 (B- 접두사 기준)"""
    entity_types = set()
    for label in labels:
        if label.startswith('B-'):
            entity_types.add(label[2:])  # B- 제거
    return entity_types


def stratified_split(
    sentences: List[List[str]], 
    labels: List[List[str]], 
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[Dict[str, List], Dict[str, Any]]:
    """
    Stratified Split: 엔티티 타입별 균등 분배
    
    Args:
        sentences: 토큰 리스트의 리스트
        labels: 라벨 리스트의 리스트
        train_ratio: 훈련 데이터 비율 (기본 80%)
        val_ratio: 검증 데이터 비율 (기본 10%)
        test_ratio: 테스트 데이터 비율 (기본 10%)
        random_seed: 재현성을 위한 시드
    
    Returns:
        split_data: {'train': [...], 'validation': [...], 'test': [...]}
        split_info: 분할 정보 (통계)
    """
    random.seed(random_seed)
    
    # 1. 각 문장의 엔티티 타입 추출
    sentence_data = []
    entity_type_counter = Counter()
    
    for sent, labs in zip(sentences, labels):
        entity_types = get_entity_types_in_sentence(labs)
        sentence_data.append({
            'sentence': sent,
            'labels': labs,
            'entity_types': entity_types
        })
        entity_type_counter.update(entity_types)
    
    # 2. 엔티티 타입별로 문장 그룹화
    entity_to_sentences = defaultdict(list)
    for idx, data in enumerate(sentence_data):
        for entity_type in data['entity_types']:
            entity_to_sentences[entity_type].append(idx)
    
    # 3. 각 엔티티 타입별로 분할
    train_indices = set()
    val_indices = set()
    test_indices = set()
    
    for entity_type, sentence_indices in entity_to_sentences.items():
        random.shuffle(sentence_indices)
        
        n = len(sentence_indices)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_indices.update(sentence_indices[:train_end])
        val_indices.update(sentence_indices[train_end:val_end])
        test_indices.update(sentence_indices[val_end:])
    
    # 4. 엔티티가 없는 문장(O만 있는 문장) 처리
    no_entity_indices = []
    for idx, data in enumerate(sentence_data):
        if not data['entity_types']:
            no_entity_indices.append(idx)
    
    if no_entity_indices:
        random.shuffle(no_entity_indices)
        n = len(no_entity_indices)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_indices.update(no_entity_indices[:train_end])
        val_indices.update(no_entity_indices[train_end:val_end])
        test_indices.update(no_entity_indices[val_end:])
    
    # 5. 중복 제거 (validation과 test 우선)
    train_indices = train_indices - val_indices - test_indices
    
    # 6. 인덱스로 데이터 분할
    train_data = [sentence_data[i] for i in sorted(train_indices)]
    val_data = [sentence_data[i] for i in sorted(val_indices)]
    test_data = [sentence_data[i] for i in sorted(test_indices)]
    
    # 7. 분할 정보 생성
    def get_entity_distribution(data_list):
        counter = Counter()
        for item in data_list:
            counter.update(item['entity_types'])
        return dict(counter)
    
    split_info = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'random_seed': random_seed,
        'ratios': {
            'train': train_ratio,
            'validation': val_ratio,
            'test': test_ratio
        },
        'counts': {
            'train': len(train_data),
            'validation': len(val_data),
            'test': len(test_data),
            'total': len(sentences)
        },
        'entity_distributions': {
            'train': get_entity_distribution(train_data),
            'validation': get_entity_distribution(val_data),
            'test': get_entity_distribution(test_data),
            'total': dict(entity_type_counter)
        }
    }
    
    # 8. 결과 반환
    split_data = {
        'train': [(item['sentence'], item['labels']) for item in train_data],
        'validation': [(item['sentence'], item['labels']) for item in val_data],
        'test': [(item['sentence'], item['labels']) for item in test_data]
    }
    
    return split_data, split_info


def split_and_save_training_data(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    훈련 데이터를 분할하여 저장
    
    Args:
        input_file: 원본 CoNLL 형식 파일 경로
        output_dir: 분할된 파일을 저장할 디렉토리
        train_ratio, val_ratio, test_ratio: 분할 비율
        random_seed: 재현성 시드
    
    Returns:
        분할 정보 딕셔너리
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 데이터 로드
    print(f"데이터 로드 중: {input_file}")
    sentences, labels = load_conll_data(input_file)
    print(f"총 {len(sentences):,}개 문장 로드")
    
    # 2. Stratified Split
    print(f"Stratified Split 수행 중 (Train/Val/Test = {train_ratio}/{val_ratio}/{test_ratio})...")
    split_data, split_info = stratified_split(
        sentences, labels, 
        train_ratio, val_ratio, test_ratio, 
        random_seed
    )
    
    # 3. 각 세트를 파일로 저장
    for split_name, data_list in split_data.items():
        split_sentences = [item[0] for item in data_list]
        split_labels = [item[1] for item in data_list]
        
        output_file = output_path / f"{split_name}.txt"
        save_conll_data(split_sentences, split_labels, str(output_file))
        print(f"✓ {split_name:12s}: {len(data_list):,}개 문장 저장 → {output_file.name}")
    
    # 4. 분할 정보 저장
    info_file = output_path / "data_split_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    print(f"✓ 분할 정보 저장 → {info_file.name}")
    
    # 5. 통계 출력
    print("\n" + "=" * 70)
    print("데이터 분할 통계")
    print("=" * 70)
    for split_name, count in split_info['counts'].items():
        if split_name != 'total':
            percentage = (count / split_info['counts']['total']) * 100
            print(f"{split_name:12s}: {count:,}개 문장 ({percentage:.1f}%)")
    print(f"{'총 합계':12s}: {split_info['counts']['total']:,}개 문장")
    
    print("\n엔티티 타입별 분포:")
    print("-" * 70)
    print(f"{'엔티티':20s} {'Train':>10s} {'Val':>10s} {'Test':>10s} {'Total':>10s}")
    print("-" * 70)
    
    all_entity_types = set()
    for dist in split_info['entity_distributions'].values():
        all_entity_types.update(dist.keys())
    
    for entity_type in sorted(all_entity_types):
        train_count = split_info['entity_distributions']['train'].get(entity_type, 0)
        val_count = split_info['entity_distributions']['validation'].get(entity_type, 0)
        test_count = split_info['entity_distributions']['test'].get(entity_type, 0)
        total_count = split_info['entity_distributions']['total'].get(entity_type, 0)
        
        print(f"{entity_type:20s} {train_count:>10,} {val_count:>10,} {test_count:>10,} {total_count:>10,}")
    
    print("=" * 70)
    
    return split_info


if __name__ == "__main__":
    # 테스트
    import sys
    
    if len(sys.argv) < 3:
        print("사용법: python data_splitter.py <input_file> <output_dir>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    split_and_save_training_data(input_file, output_dir)
