#!/usr/bin/env python3
"""
고급 데이터 증강 - 더 많은 훈련 샘플 생성
- 문장 재조합
- 동의어 치환
- 순서 변경
- 엔티티 조합 변형
"""
import json
import random
from pathlib import Path
from typing import List, Dict
import itertools

# 한국어 동의어 사전
SYNONYMS = {
    "저작물": ["작품", "창작물", "저작"],
    "계약서": ["합의서", "협약서", "동의서"],
    "양도": ["이전", "양도"],
    "권리": ["권한", "권리"],
    "사업": ["프로젝트", "사업", "과제"],
    "구축": ["개발", "제작", "구축"],
    "디지털": ["전산", "디지털", "온라인"],
}

# 문서 템플릿들
DOCUMENT_TEMPLATES = [
    "{company}는(은) {project_name}의 결과물에 대한 {right_info}를(을) {date}에 양도한다.",
    "{company}와(과) 양도인은 {contract_type}을(를) {date}에 체결하였다.",
    "{position} {company} 담당자는 {address}에 위치하며, 연락처는 {phone}이다.",
    "본 {contract_type}은(는) {law_reference}에 근거하여 작성되었다.",
    "{project_name}의 {right_info}는(은) {company}에게 귀속된다.",
]

def load_entities_from_samples(samples: List[Dict]) -> Dict[str, List[str]]:
    """샘플들로부터 엔티티 추출"""
    entities = {}
    
    for sample in samples:
        tokens = sample.get('tokens', [])
        labels = sample.get('labels', [])
        
        i = 0
        while i < len(labels):
            if labels[i].startswith('B-'):
                entity_type = labels[i][2:]
                start = i
                i += 1
                while i < len(labels) and labels[i] == f'I-{entity_type}':
                    i += 1
                
                entity_value = ''.join(tokens[start:i])
                
                if entity_type not in entities:
                    entities[entity_type] = []
                if entity_value not in entities[entity_type]:
                    entities[entity_type].append(entity_value)
            else:
                i += 1
    
    return entities

def generate_template_samples(entities: Dict[str, List[str]], num_samples: int = 500) -> List[Dict]:
    """템플릿을 사용하여 새로운 샘플 생성"""
    samples = []
    
    for _ in range(num_samples):
        template = random.choice(DOCUMENT_TEMPLATES)
        
        # 템플릿에 필요한 엔티티 타입 찾기
        required_types = []
        for entity_type in entities.keys():
            placeholder = "{" + entity_type.lower() + "}"
            if placeholder in template:
                required_types.append(entity_type)
        
        # 필요한 엔티티가 모두 있는지 확인
        if not all(entity_type in entities and entities[entity_type] for entity_type in required_types):
            continue
        
        # 엔티티 값으로 템플릿 채우기
        text = template
        entity_map = {}
        
        for entity_type in required_types:
            placeholder = "{" + entity_type.lower() + "}"
            entity_value = random.choice(entities[entity_type])
            text = text.replace(placeholder, entity_value)
            entity_map[entity_type] = entity_value
        
        # BIO 태깅
        tokens = list(text)
        labels = ['O'] * len(tokens)
        
        for entity_type, entity_value in entity_map.items():
            pos = text.find(entity_value)
            if pos >= 0:
                labels[pos] = f'B-{entity_type}'
                for i in range(pos + 1, pos + len(entity_value)):
                    if i < len(labels):
                        labels[i] = f'I-{entity_type}'
        
        samples.append({
            "tokens": tokens,
            "labels": labels
        })
    
    return samples

def synonym_replacement(sample: Dict, replacement_prob: float = 0.3) -> Dict:
    """동의어 치환"""
    tokens = sample['tokens'][:]
    labels = sample['labels'][:]
    
    text = ''.join(tokens)
    
    for word, synonyms in SYNONYMS.items():
        if word in text and random.random() < replacement_prob:
            synonym = random.choice(synonyms)
            text = text.replace(word, synonym, 1)
    
    new_tokens = list(text)
    new_labels = labels[:len(new_tokens)] + ['O'] * max(0, len(new_tokens) - len(labels))
    new_labels = new_labels[:len(new_tokens)]
    
    return {
        "tokens": new_tokens,
        "labels": new_labels
    }

def combine_samples(samples: List[Dict], num_combinations: int = 200) -> List[Dict]:
    """여러 샘플을 조합하여 새로운 샘플 생성"""
    combined = []
    
    for _ in range(num_combinations):
        # 2-3개 샘플 선택
        num_to_combine = random.randint(2, 3)
        selected = random.sample(samples, min(num_to_combine, len(samples)))
        
        # 조합
        all_tokens = []
        all_labels = []
        
        for sample in selected:
            all_tokens.extend(sample['tokens'])
            all_labels.extend(sample['labels'])
            # 샘플 사이에 공백 추가
            all_tokens.extend([' ', ' '])
            all_labels.extend(['O', 'O'])
        
        combined.append({
            "tokens": all_tokens,
            "labels": all_labels
        })
    
    return combined

def back_translation_simulation(sample: Dict) -> List[Dict]:
    """역번역 시뮬레이션 (단순 변형)"""
    variations = []
    
    # 조사 변경
    text = ''.join(sample['tokens'])
    
    replacements = [
        ('는', '은'), ('은', '는'),
        ('가', '이'), ('이', '가'),
        ('를', '을'), ('을', '를'),
        ('와', '과'), ('과', '와'),
    ]
    
    for old, new in replacements:
        if old in text:
            new_text = text.replace(old, new, 1)
            new_tokens = list(new_text)
            new_labels = sample['labels'][:len(new_tokens)]
            
            if len(new_tokens) > len(new_labels):
                new_labels = new_labels + ['O'] * (len(new_tokens) - len(new_labels))
            else:
                new_labels = new_labels[:len(new_tokens)]
            
            variations.append({
                "tokens": new_tokens,
                "labels": new_labels
            })
            break  # 하나만 생성
    
    return variations

def convert_to_bio_format(samples: List[Dict]) -> str:
    """BIO 형식으로 변환"""
    lines = []
    
    for sample in samples:
        tokens = sample['tokens']
        labels = sample['labels']
        
        for token, label in zip(tokens, labels):
            if token.strip():
                lines.append(f"{token}\t{label}")
        lines.append("")
    
    return '\n'.join(lines)

def main():
    print("=" * 80)
    print("고급 데이터 증강 시작")
    print("=" * 80)
    
    # 기존 샘플 로드
    input_path = Path("data/in/auto_extracted_ground_truth.json")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_samples = data['samples']
    print(f"\n원본 샘플: {len(original_samples)}개")
    
    # 엔티티 추출
    print("\n[1단계] 엔티티 추출...")
    entities = load_entities_from_samples(original_samples)
    for entity_type, values in entities.items():
        print(f"  {entity_type}: {len(values)}개")
    
    all_samples = []
    
    # 템플릿 기반 샘플 생성
    print("\n[2단계] 템플릿 기반 샘플 생성...")
    template_samples = generate_template_samples(entities, num_samples=500)
    print(f"  → {len(template_samples)}개 생성")
    all_samples.extend(template_samples)
    
    # 동의어 치환
    print("\n[3단계] 동의어 치환...")
    synonym_samples = []
    for sample in original_samples[:100]:
        for _ in range(3):  # 각 샘플당 3개 변형
            synonym_sample = synonym_replacement(sample)
            synonym_samples.append(synonym_sample)
    print(f"  → {len(synonym_samples)}개 생성")
    all_samples.extend(synonym_samples)
    
    # 샘플 조합
    print("\n[4단계] 샘플 조합...")
    combined_samples = combine_samples(original_samples, num_combinations=300)
    print(f"  → {len(combined_samples)}개 생성")
    all_samples.extend(combined_samples)
    
    # 역번역 시뮬레이션
    print("\n[5단계] 역번역 시뮬레이션...")
    bt_samples = []
    for sample in original_samples:
        bt_variations = back_translation_simulation(sample)
        bt_samples.extend(bt_variations)
    print(f"  → {len(bt_samples)}개 생성")
    all_samples.extend(bt_samples)
    
    # 원본도 추가
    all_samples.extend(original_samples)
    
    # 무작위 섞기
    random.shuffle(all_samples)
    
    print(f"\n총 생성된 샘플: {len(all_samples)}개")
    
    # 파일로 저장
    print("\n[6단계] 파일 저장...")
    bio_text = convert_to_bio_format(all_samples)
    
    output_files = [
        "data/in/realistic_train.txt",
        "data/in/real_document_train.txt"
    ]
    
    for output_file in output_files:
        output_path = Path(output_file)
        
        # 기존 내용에 추가
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write('\n\n')
            f.write(bio_text)
        
        # 통계
        with open(output_path, 'r', encoding='utf-8') as f:
            total_lines = len(f.readlines())
        
        print(f"  ✅ {output_path.name}: {total_lines:,} 라인")
    
    # 엔티티 통계
    print(f"\n엔티티 통계:")
    entity_counts = {}
    for sample in all_samples:
        for label in sample['labels']:
            if label.startswith('B-'):
                entity_type = label[2:]
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count:,}개")
    
    print("\n✅ 고급 증강 완료!")

if __name__ == "__main__":
    random.seed(123)
    main()
