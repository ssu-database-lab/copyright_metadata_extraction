#!/usr/bin/env python3
"""
최종 대량 데이터 생성 - 모든 기법 통합
실제 사용 가능한 고품질 대량 훈련 데이터 생성
"""
import json
import random
from pathlib import Path
from typing import List, Dict
import itertools

# 확장된 데이터 풀
EXTENDED_COMPANIES = [
    "한국문화정보원", "주스튜디오", "문화재청", "국립중앙박물관", 
    "한국저작권위원회", "충남문화관광재단", "국가유산청", "문화진흥원",
    "서울시립미술관", "한국문화예술위원회", "예술의전당", "세종문화회관",
    "국립현대미술관", "대한민국역사박물관", "한국영상자료원",
    "한국예술종합학교", "국립박물관문화재단", "한국문화재재단",
    "서울문화재단", "경기문화재단", "인천문화재단", "부산문화재단",
    "광주문화재단", "대전문화재단", "울산문화재단", "강원문화재단"
]

EXTENDED_PROJECTS = [
    "2024년 공공저작물 디지털 전환 구축 사업",
    "2023년 문화재 디지털화 사업",
    "공공저작물 고품질 확충 사업",
    "디지털 아카이브 구축 사업",
    "문화유산 디지털 복원 프로젝트",
    "온라인 콘텐츠 제작 사업",
    "디지털 전시 콘텐츠 개발",
    "메타데이터 표준화 사업",
    "문화재 3D 스캔 사업",
    "VR/AR 콘텐츠 제작 프로젝트",
    "문화유산 기록화 사업",
    "박물관 소장품 디지털화",
    "문화재 보존처리 기록 사업",
    "전통문화 아카이빙 프로젝트"
]

DOCUMENT_CONTEXTS = [
    "본 계약은 저작권법과 관련 법규에 따라 체결되었으며, ",
    "양 당사자는 상호 협의 하에 다음과 같이 합의하였다. ",
    "본 동의서는 개인정보 보호법을 준수하여 작성되었으며, ",
    "저작재산권 양도에 관하여 다음과 같이 계약을 체결한다. ",
    "문화재 보호법 및 저작권법에 근거하여 ",
]

def generate_realistic_address():
    """실제 같은 주소 생성"""
    cities = ["서울특별시", "경기도", "부산광역시", "인천광역시", "대구광역시", "대전광역시", "광주광역시", "울산광역시", "충청남도", "충청북도", "전라남도", "전라북도", "경상남도", "경상북도", "강원도", "제주특별자치도"]
    districts = ["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구"]
    streets = ["세종대로", "태평로", "을지로", "종로", "남대문로", "소공로", "퇴계로", "청계천로", "강남대로", "테헤란로", "논현로", "영동대로", "올림픽대로", "한강대로"]
    
    city = random.choice(cities)
    district = random.choice(districts) if "시" in city else ""
    street = random.choice(streets)
    number = random.randint(1, 500)
    
    if district:
        return f"{city} {district} {street} {number}"
    else:
        return f"{city} {street} {number}"

def generate_phone_variations():
    """다양한 전화번호 형식 생성"""
    formats = [
        f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}",
        f"02-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
        f"031-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
        f"032-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
    ]
    return random.choice(formats)

def generate_date_range():
    """날짜 범위 생성"""
    year1 = random.randint(2020, 2024)
    month1 = random.randint(1, 12)
    day1 = random.randint(1, 28)
    
    year2 = year1 if random.random() > 0.3 else year1 + 1
    month2 = random.randint(month1, 12)
    day2 = random.randint(1, 28)
    
    return f"{year1}.{month1:02d}.{day1:02d} ~ {year2}.{month2:02d}.{day2:02d}"

COMPLEX_PATTERNS = [
    "{context}{company}는(은) {project}의 수행 결과물에 대한 {right1} 및 {right2}를(을) {date}에 양도한다.",
    "{company}의 {position}은 {address}에 소재하며, {project}를 담당한다. 연락처: {phone}",
    "본 {contract_type}는 {law1} 및 {law2}에 근거하여 {date}에 작성되었다.",
    "{context}{company}와 양수인은 {project}와 관련하여 {right1}을 양도하는 것에 합의하였다.",
    "{position} {company} 담당자는 {project}의 {right1} 및 {right2}에 관한 권한을 보유한다.",
    "{date}부터 {company}는 {project}의 결과물에 대한 모든 {right1}를 행사할 수 있다.",
    "{address} 소재 {company}는 {law1}에 따라 {contract_type}를 체결하였다.",
    "{context}양도인은 {company}에게 {right1}을 양도하며, 이에 대한 대가는 {date}까지 지급된다.",
]

RIGHTS_EXTENDED = [
    "저작재산권", "저작인격권", "저작인접권", "2차저작물작성권",
    "복제권", "배포권", "공연권", "전시권", "공표권",
    "성명표시권", "동일성유지권", "공중송신권", "전송권",
    "방송권", "대여권", "번역권", "각색권"
]

LAWS_EXTENDED = [
    "저작권법 제24조의2", "저작권법 제101조", "저작권법 제45조",
    "공공데이터의 제공 및 이용활성화에 관한 법률",
    "저작권법 제11조", "저작권법 제12조", "저작권법 제13조",
    "저작권법 제16조", "저작권법 제17조", "저작권법 제18조",
    "저작권법 제21조", "저작권법 제22조", "저작권법 제46조"
]

POSITIONS_EXTENDED = [
    "대표이사", "연구원", "책임연구원", "팀장", "과장",
    "부장", "실장", "본부장", "센터장", "관장",
    "주임연구원", "선임연구원", "수석연구원", "연구소장",
    "기획실장", "사업본부장", "운영팀장"
]

CONTRACT_TYPES_EXTENDED = [
    "저작재산권 양도계약", "초상권 이용동의서", "저작물 이용허락 계약",
    "저작인접권 양도계약서", "콘텐츠 제작 계약서",
    "저작권 양도 및 이용 계약서", "공동저작물 계약서",
    "저작물 사용 동의서", "초상 및 저작권 이용동의서"
]

def create_complex_sample(pattern: str) -> Dict:
    """복잡한 샘플 생성"""
    text = pattern
    entities = {}
    
    replacements = {
        '{context}': (random.choice(DOCUMENT_CONTEXTS), None),
        '{company}': (random.choice(EXTENDED_COMPANIES), 'COMPANY'),
        '{project}': (random.choice(EXTENDED_PROJECTS), 'PROJECT_NAME'),
        '{right1}': (random.choice(RIGHTS_EXTENDED), 'RIGHT_INFO'),
        '{right2}': (random.choice(RIGHTS_EXTENDED), 'RIGHT_INFO'),
        '{address}': (generate_realistic_address(), 'ADDRESS'),
        '{position}': (random.choice(POSITIONS_EXTENDED), 'POSITION'),
        '{law1}': (random.choice(LAWS_EXTENDED), 'LAW_REFERENCE'),
        '{law2}': (random.choice(LAWS_EXTENDED), 'LAW_REFERENCE'),
        '{contract_type}': (random.choice(CONTRACT_TYPES_EXTENDED), 'CONTRACT_TYPE'),
        '{phone}': (generate_phone_variations(), 'PHONE'),
        '{date}': (generate_date_range() if random.random() > 0.5 else f"{random.randint(2020, 2024)}.{random.randint(1, 12):02d}.{random.randint(1, 28):02d}", 'DATE'),
    }
    
    for placeholder, (value, entity_type) in replacements.items():
        if placeholder in text:
            text = text.replace(placeholder, value)
            if entity_type:
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(value)
    
    # BIO 태깅
    tokens = list(text)
    labels = ['O'] * len(tokens)
    
    for entity_type, values in entities.items():
        for value in values:
            start_pos = 0
            while True:
                pos = text.find(value, start_pos)
                if pos < 0:
                    break
                
                if labels[pos] == 'O':  # 중복 태깅 방지
                    labels[pos] = f'B-{entity_type}'
                    for i in range(pos + 1, min(pos + len(value), len(labels))):
                        if labels[i] == 'O':
                            labels[i] = f'I-{entity_type}'
                
                start_pos = pos + 1
    
    return {
        "tokens": tokens,
        "labels": labels
    }

def generate_multi_paragraph_documents(num_docs: int = 2000) -> List[Dict]:
    """다단락 문서 생성"""
    samples = []
    
    for _ in range(num_docs):
        num_paragraphs = random.randint(3, 6)
        all_tokens = []
        all_labels = []
        
        for _ in range(num_paragraphs):
            pattern = random.choice(COMPLEX_PATTERNS)
            sample = create_complex_sample(pattern)
            
            all_tokens.extend(sample['tokens'])
            all_labels.extend(sample['labels'])
            all_tokens.extend([' ', ' '])
            all_labels.extend(['O', 'O'])
        
        samples.append({
            "tokens": all_tokens,
            "labels": all_labels
        })
    
    return samples

def convert_to_bio_format(samples: List[Dict]) -> str:
    """BIO 형식으로 변환"""
    lines = []
    
    for sample in samples:
        for token, label in zip(sample['tokens'], sample['labels']):
            if token.strip():
                lines.append(f"{token}\t{label}")
        lines.append("")
    
    return '\n'.join(lines)

def main():
    print("=" * 80)
    print("최종 대량 데이터 생성")
    print("=" * 80)
    
    all_samples = []
    
    # 1. 복잡한 패턴 샘플
    print("\n[1단계] 복잡한 패턴 샘플 생성...")
    for _ in range(3000):
        pattern = random.choice(COMPLEX_PATTERNS)
        sample = create_complex_sample(pattern)
        all_samples.append(sample)
    print(f"  → {len(all_samples)}개 생성")
    
    # 2. 다단락 문서
    print("\n[2단계] 다단락 문서 생성...")
    multi_para = generate_multi_paragraph_documents(num_docs=2000)
    all_samples.extend(multi_para)
    print(f"  → {len(multi_para)}개 추가")
    
    print(f"\n총 샘플: {len(all_samples):,}개")
    
    # 무작위 섞기
    random.shuffle(all_samples)
    
    # 저장
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
        file_size = output_path.stat().st_size / (1024 * 1024)
        
        print(f"  ✅ {output_path.name}: {total_lines:,} 라인 ({file_size:.2f} MB)")
    
    # 통계
    entity_counts = {}
    for sample in all_samples:
        for label in sample['labels']:
            if label.startswith('B-'):
                entity_type = label[2:]
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    print(f"\n엔티티 통계:")
    for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count:,}개")
    
    print("\n" + "=" * 80)
    print("✅ 최종 대량 데이터 생성 완료!")
    print("=" * 80)

if __name__ == "__main__":
    random.seed(789)
    main()
