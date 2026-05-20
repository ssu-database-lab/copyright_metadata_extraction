#!/usr/bin/env python3
"""
대량 데이터 생성 - 패턴 기반 샘플 생성
실제 문서에서 자주 나타나는 패턴을 이용하여 대량의 훈련 샘플 생성
"""
import json
import random
from pathlib import Path
from typing import List, Dict
import itertools

# 회사명 목록
COMPANIES = [
    "한국문화정보원", "주스튜디오", "문화재청", "국립중앙박물관", 
    "한국저작권위원회", "충남문화관광재단", "국가유산청", "문화진흥원",
    "서울시립미술관", "한국문화예술위원회", "예술의전당", "세종문화회관",
    "국립현대미술관", "대한민국역사박물관", "한국영상자료원"
]

# 프로젝트명 목록
PROJECTS = [
    "2024년 공공저작물 디지털 전환 구축 사업",
    "2023년 문화재 디지털화 사업",
    "공공저작물 고품질 확충 사업",
    "디지털 아카이브 구축 사업",
    "문화유산 디지털 복원 프로젝트",
    "온라인 콘텐츠 제작 사업",
    "디지털 전시 콘텐츠 개발",
    "메타데이터 표준화 사업"
]

# 주소 목록
ADDRESSES = [
    "서울특별시 마포구 월드컵북로 400",
    "서울시 종로구 세종대로 209",
    "서울 용산구 서빙고로 137",
    "서울특별시 서초구 서초대로 398",
    "충청남도 천안시 동남구 목천읍",
    "경기도 과천시 광명로 313",
    "서울 종로구 사직로 161",
    "부산광역시 해운대구 우동"
]

# 직책 목록
POSITIONS = [
    "대표이사", "연구원", "책임연구원", "팀장", "과장",
    "부장", "실장", "본부장", "센터장", "관장"
]

# 권리 정보
RIGHTS = [
    "저작재산권", "저작인격권", "저작인접권", "2차저작물작성권",
    "복제권", "배포권", "공연권", "전시권", "공표권",
    "성명표시권", "동일성유지권", "공중송신권"
]

# 법률 참조
LAWS = [
    "저작권법 제24조의2", "저작권법 제101조", "저작권법 제45조",
    "공공데이터의 제공 및 이용활성화에 관한 법률",
    "저작권법 제11조", "저작권법 제12조", "저작권법 제13조"
]

# 계약 유형
CONTRACT_TYPES = [
    "저작재산권 양도계약", "초상권 이용동의서", "저작물 이용허락 계약",
    "저작인접권 양도계약서", "콘텐츠 제작 계약서"
]

# 문장 패턴들
SENTENCE_PATTERNS = [
    "{company}는 {project}를 수행하며 발생한 {right}를 양도한다.",
    "{position} {company} 소속은 {address}에 소재한다.",
    "본 {contract_type}는 {law}에 근거하여 작성되었다.",
    "{company}와 양도인은 {date}에 {contract_type}를 체결하였다.",
    "{project}의 결과물에 대한 {right}는 {company}에 귀속된다.",
    "양도인은 {company}에게 {right}를 양도하는 것에 동의한다.",
    "{law}에 따라 {right}를 {date}부터 양도한다.",
    "연락처 {phone}으로 문의 바랍니다.",
    "{address} 소재 {company}는 {project}를 추진한다.",
    "{position}은 {right}에 대한 권한을 가진다.",
]

def generate_phone():
    """전화번호 생성"""
    return f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"

def generate_date():
    """날짜 생성"""
    year = random.randint(2020, 2024)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    return f"{year}.{month:02d}.{day:02d}"

def create_sample_from_pattern(pattern: str) -> Dict:
    """패턴으로부터 샘플 생성"""
    entities = {}
    text = pattern
    
    # 엔티티 값 할당
    replacements = {
        '{company}': (random.choice(COMPANIES), 'COMPANY'),
        '{project}': (random.choice(PROJECTS), 'PROJECT_NAME'),
        '{right}': (random.choice(RIGHTS), 'RIGHT_INFO'),
        '{address}': (random.choice(ADDRESSES), 'ADDRESS'),
        '{position}': (random.choice(POSITIONS), 'POSITION'),
        '{law}': (random.choice(LAWS), 'LAW_REFERENCE'),
        '{contract_type}': (random.choice(CONTRACT_TYPES), 'CONTRACT_TYPE'),
        '{phone}': (generate_phone(), 'PHONE'),
        '{date}': (generate_date(), 'DATE'),
    }
    
    for placeholder, (value, entity_type) in replacements.items():
        if placeholder in text:
            text = text.replace(placeholder, value)
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(value)
    
    # BIO 태깅
    tokens = list(text)
    labels = ['O'] * len(tokens)
    
    for entity_type, values in entities.items():
        for value in values:
            pos = text.find(value)
            if pos >= 0:
                labels[pos] = f'B-{entity_type}'
                for i in range(pos + 1, min(pos + len(value), len(labels))):
                    labels[i] = f'I-{entity_type}'
    
    return {
        "tokens": tokens,
        "labels": labels
    }

def generate_complex_documents(num_docs: int = 1000) -> List[Dict]:
    """복잡한 문서 샘플 생성"""
    samples = []
    
    for _ in range(num_docs):
        # 2-4개 문장 조합
        num_sentences = random.randint(2, 4)
        selected_patterns = random.sample(SENTENCE_PATTERNS, num_sentences)
        
        all_tokens = []
        all_labels = []
        
        for pattern in selected_patterns:
            sample = create_sample_from_pattern(pattern)
            all_tokens.extend(sample['tokens'])
            all_labels.extend(sample['labels'])
            # 문장 사이 공백
            all_tokens.extend([' '])
            all_labels.extend(['O'])
        
        samples.append({
            "tokens": all_tokens,
            "labels": all_labels
        })
    
    return samples

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
    print("대량 패턴 기반 데이터 생성")
    print("=" * 80)
    
    all_samples = []
    
    # 1. 단순 패턴 샘플
    print("\n[1단계] 단순 패턴 샘플 생성...")
    for _ in range(2000):
        pattern = random.choice(SENTENCE_PATTERNS)
        sample = create_sample_from_pattern(pattern)
        all_samples.append(sample)
    print(f"  → {len(all_samples)}개 생성")
    
    # 2. 복잡한 문서 샘플
    print("\n[2단계] 복잡한 문서 샘플 생성...")
    complex_samples = generate_complex_documents(num_docs=1500)
    all_samples.extend(complex_samples)
    print(f"  → {len(complex_samples)}개 추가")
    
    print(f"\n총 샘플: {len(all_samples)}개")
    
    # 무작위 섞기
    random.shuffle(all_samples)
    
    # 파일로 저장
    print("\n[3단계] 파일 저장...")
    bio_text = convert_to_bio_format(all_samples)
    
    output_files = [
        "data/in/realistic_train.txt",
        "data/in/real_document_train.txt"
    ]
    
    for output_file in output_files:
        output_path = Path(output_file)
        
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write('\n\n')
            f.write(bio_text)
        
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
    
    print("\n✅ 대량 데이터 생성 완료!")

if __name__ == "__main__":
    random.seed(456)
    main()
