#!/usr/bin/env python3
"""
NER Fine-Tuning 시스템 - 고성능 버전
B-I-O 태깅 기법으로 23개 엔티티 타입 지원 + 실시간 성능 모니터링

디렉토리 구조:
    api/
    ├── models/
    │   └── ner/
    │       └── {model_name}/          # 예: klue-roberta-large
    │           ├── config.json
    │           ├── model.safetensors
    │           ├── label_map.json
    │           └── ...
    └── module/
        └── ner/
            └── training/
                └── {model_name}/      # 예: klue-roberta-large
                    ├── dynamic_train.txt
                    ├── logs/
                    └── checkpoints/
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, precision_score, recall_score
from seqeval.metrics import f1_score as seqeval_f1, precision_score as seqeval_precision, recall_score as seqeval_recall, accuracy_score as seqeval_accuracy
from datasets import Dataset as HFDataset
import warnings
warnings.filterwarnings("ignore")

# 추가 경고 메시지 숨기기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
import logging as transformers_logging
transformers_logging.getLogger("transformers").setLevel(transformers_logging.ERROR)

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 로깅 설정
logging.basicConfig(level=logging.WARNING, format='%(message)s')

@dataclass
class TrainingConfig:
    """Fine-Tuning 설정"""
    model_name: str = os.getenv("NER_MODEL_NAME", "klue/roberta-large")
    num_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 128
    gradient_accumulation_steps: int = 1
    eval_steps: int = 200
    save_steps: int = 500
    fp16: bool = True
    early_stopping_patience: int = 3
    dropout: float = 0.1
    label_smoothing: float = 0.0# 23개 엔티티 타입 (B-I-O 태깅)
ENTITY_TYPES = [
    "NAME",           # 이름 
    "PHONE",          # 전화번호
    "ADDRESS",        # 주소
    "DATE",           # 날짜
    "COMPANY",        # 회사/기관명
    "EMAIL",          # 이메일
    "POSITION",       # 직책/직위
    "CONTRACT_TYPE",  # 계약서 유형
    "CONSENT_TYPE",   # 동의서 유형
    "RIGHT_INFO",     # 권리정보
    "MONEY",          # 금액
    "PERIOD",         # 기간
    "PROJECT_NAME",   # 사업명
    "LAW_REFERENCE",  # 법령 근거
    "ID_NUM",         # 신분증번호
    "TITLE",          # 제목
    "URL",            # URL정보
    "DESCRIPTION",    # 설명
    "TYPE",           # 유형
    "STATUS",         # 상태
    "DEPARTMENT",     # 부서정보
    "LANGUAGE",       # 언어
    "QUANTITY"        # 수량정보
]

def create_bio_labels():
    """BIO 라벨 시스템 생성"""
    labels = ["O"]  # Outside
    for entity in ENTITY_TYPES:
        labels.extend([f"B-{entity}", f"I-{entity}"])
    return labels

BIO_LABELS = create_bio_labels()
LABEL_TO_ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

print(f"Total labels: {len(BIO_LABELS)}")
print(f"Label examples: {BIO_LABELS[:10]}...")

# 엔티티 데이터 풀 (전체 명사 데이터셋)
ENTITY_DATA_POOL = {
    "names": ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황", "안", "송", "류", "전", "홍", "고", "문", "양", "손", "배", "백", "허", "유", "남", "심", "노", "하", "곽", "성", "차", "주", "우", "구", "나", "민", "진", "지", "엄", "채", "원", "천", "방", "변", "여", "추", "도", "소", "석", "선", "설", "마", "길", "연"],
    "given_names": ["민수", "영희", "철수", "영수", "미영", "수진", "정훈", "지영", "성호", "은주", "태현", "소영", "준호", "혜진", "동현", "서연", "도윤", "시우", "하은", "주원", "지호", "은서", "예준", "지민", "서진", "예림", "지아", "현우", "채원", "시윤", "유진", "시은", "준혁", "예은", "도현", "채윤", "건우", "서우", "지율", "하윤", "준서", "서준", "하준", "지후", "민준", "선우", "연우", "정우", "승우", "지원", "서윤", "지우", "민서", "하린", "수아", "지유", "유나", "소율", "예서", "하율", "시연", "유주", "다은", "서현", "가윤", "나윤"],
    "companies": [
        "삼성전자", "LG전자", "현대자동차", "SK텔레콤", "네이버", "카카오", "한국전력공사", "포스코", "신한은행", "국민은행", "하나은행", "우리은행", "KB국민은행", "KT", "LG유플러스", "롯데그룹", "두산그룹", "GS그룹", "한화그룹", "CJ그룹",
        "서울시청", "부산시청", "인천시청", "대구시청", "광주시청", "대전시청", "울산시청", "세종시청", "경기도청", "강원도청", "충청북도청", "충청남도청", "전라북도청", "전라남도청", "경상북도청", "경상남도청", "제주도청",
        "한국문화예술위원회", "국가보훈처", "문화체육관광부", "교육부", "한국저작권위원회", "한국콘텐츠진흥원", "국립중앙도서관", "국사편찬위원회", "문화재청", "방송통신위원회", "한국정보화진흥원", "한국교육학술정보원",
        "KBS", "MBC", "SBS", "JTBC", "tvN", "채널A", "MBN", "TV조선", "YTN", "연합뉴스", "조선일보", "중앙일보", "동아일보", "한겨레", "경향신문",
        "국립극장", "세종문화회관", "예술의전당", "국립현대미술관", "국립중앙박물관", "대한민국역사박물관", "국립민속박물관", "서울시립미술관", "부산시립미술관", "광주시립미술관", "한국문화예술교육진흥원", "한국문화관광연구원", "한국예술종합학교", "국악방송", "아리랑TV",
        "나라지식정보", "한국문화정보원", "디지털헤리티지", "문화콘텐츠닷컴", "아트센터나비", "미디어아트센터", "콘텐츠웨이브", "크리에이티브그룹", "뉴미디어아트", "인터랙티브미디어", "스마트컬처", "디지털큐레이션"
    ],
    "cities": ["서울시", "부산시", "대구시", "인천시", "광주시", "대전시", "울산시", "세종시"],
    "districts": ["강남구", "서초구", "송파구", "영등포구", "마포구", "용산구", "중구", "종로구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "양천구", "강서구", "구로구", "금천구", "관악구", "동작구", "강동구"],
    "streets": ["테헤란로", "강남대로", "서초대로", "한강대로", "마포대로", "을지로", "종로", "세종대로", "청계천로", "남산터널로"]
}

def split_entity_data(train_ratio=0.8, random_seed=42):
    """명사 데이터를 훈련/평가용으로 분할"""
    random.seed(random_seed)
    
    train_data = {}
    eval_data = {}
    
    for key, data_list in ENTITY_DATA_POOL.items():
        shuffled = data_list.copy()
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        train_data[key] = shuffled[:split_idx]
        eval_data[key] = shuffled[split_idx:]
    
    return train_data, eval_data

# 고품질 엔티티 생성기들
def generate_random_name(data_source=None):
    """한국어 이름 생성"""
    if data_source is None:
        data_source = ENTITY_DATA_POOL
    surnames = data_source.get("names", ENTITY_DATA_POOL["names"])
    given_names = data_source.get("given_names", ENTITY_DATA_POOL["given_names"])
    return random.choice(surnames) + random.choice(given_names)

def generate_random_phone():
    """전화번호 생성"""
    patterns = [
        f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}",
        f"02-{random.randint(100,999)}-{random.randint(1000,9999)}",
        f"031-{random.randint(100,999)}-{random.randint(1000,9999)}",
        f"032-{random.randint(100,999)}-{random.randint(1000,9999)}",
        f"051-{random.randint(100,999)}-{random.randint(1000,9999)}",
        f"053-{random.randint(100,999)}-{random.randint(1000,9999)}",
        f"070-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
    ]
    return random.choice(patterns)

def generate_random_company(data_source=None):
    """회사/기관명 생성"""
    if data_source is None:
        data_source = ENTITY_DATA_POOL
    companies = data_source.get("companies", ENTITY_DATA_POOL["companies"])
    
    suffixes = {
        "corporation": ["㈜", "(주)", "주식회사"],
        "foundation": ["재단법인", "(재)"],
        "association": ["사단법인", "(사)"],
        "institute": ["연구원", "연구소", "원"],
        "center": ["센터", "기관"]
    }
    
    base_name = random.choice(companies)
    if random.random() < 0.3:
        suffix_type = random.choice(list(suffixes.keys()))
        suffix = random.choice(suffixes[suffix_type])
        if suffix_type in ["foundation", "association"]:
            return f"{suffix} {base_name}"
        else:
            return f"{base_name} {suffix}"
    return base_name

def generate_random_date():
    """날짜 생성"""
    year = random.randint(2020, 2025)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    
    patterns = [
        f"{year}년 {month}월 {day}일",
        f"{year}.{month:02d}.{day:02d}",
        f"{year}-{month:02d}-{day:02d}",
        f"{month}/{day}/{year}",
        f"{year}/{month:02d}/{day:02d}",
        f"{year}. {month}. {day}.",
        f"{month}월 {day}일",
        f"{year}년{month}월{day}일"
    ]
    return random.choice(patterns)

def generate_random_address(data_source=None):
    """주소 생성"""
    if data_source is None:
        data_source = ENTITY_DATA_POOL
    cities = data_source.get("cities", ENTITY_DATA_POOL["cities"])
    districts = data_source.get("districts", ENTITY_DATA_POOL["districts"])
    streets = data_source.get("streets", ENTITY_DATA_POOL["streets"])
    
    city = random.choice(cities)
    district = random.choice(districts)
    street = random.choice(streets)
    number = random.randint(1, 999)
    
    patterns = [
        f"{city} {district} {street} {number}",
        f"{city} {district} {street} {number}번길",
        f"{city} {district} {street}{number}",
        f"{district} {street} {number}",
        f"{city} {district}"
    ]
    return random.choice(patterns)

def generate_random_money():
    """금액 생성"""
    base_amounts = [10, 50, 100, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 100000, 200000, 300000, 500000, 1000000, 2000000]
    amount = random.choice(base_amounts) * random.randint(1, 50)
    
    patterns = [
        f"{amount:,}원",
        f"금 {amount:,}원",
        f"{amount:,}만원" if amount >= 10000 else f"{amount:,}원",
        f"￦{amount:,}",
        f"{amount//10000}만원" if amount >= 10000 else f"{amount:,}원",
        f"금{amount:,}원정",
        f"{amount:,}원정"
    ]
    return random.choice(patterns)

def generate_random_email():
    """이메일 생성"""
    domains = ["gmail.com", "naver.com", "daum.net", "hanmail.net", "yahoo.com", "nate.com", "hotmail.com", "outlook.com"]
    usernames = ["user", "test", "admin", "info", "contact", "support", "help", "service", "manager", "director"]
    numbers = str(random.randint(1, 9999))
    
    username = random.choice(usernames) + numbers
    domain = random.choice(domains)
    return f"{username}@{domain}"

def generate_random_position():
    """직책 생성"""
    positions = [
        "대표이사", "상무이사", "전무이사", "이사", "부장", "차장", "과장", "대리", "주임",
        "팀장", "실장", "본부장", "지점장", "센터장", "관리자", "담당자", "연구원", "개발자",
        "회장", "부회장", "전무", "상무", "사장", "부사장", "감독", "사원", "대표", "위원장", "위원", "회원", "간사", "총무", "서기"
    ]
    return random.choice(positions)

def generate_random_contract_type():
    """계약서 유형 생성"""
    contract_types = [
        "저작물 저작재산권 양도 계약서",
        "업무위탁 계약서", 
        "용역 계약서",
        "매매 계약서",
        "임대차 계약서",
        "라이선스 계약서",
        "콘텐츠 제작 계약서",
        "저작권 양도 계약서",
        "근로계약서",
        "협약서",
        "업무협약서",
        "상호협력협약서",
        "양해각서",
        "약정서",
        "합의서"
    ]
    return random.choice(contract_types)

def generate_random_consent_type():
    """동의서 유형 생성"""
    consent_types = [
        "저작인접권의 저작재산권 양도 및 초상 이용 동의서",
        "개인정보 수집 및 이용 동의서",
        "초상권 이용 동의서",
        "저작물 활용 동의서",
        "출연 동의서",
        "촬영 동의서",
        "개인정보 제3자 제공 동의서",
        "저작권 동의서",
        "승낙서",
        "확인서",
        "참여동의서",
        "녹화동의서"
    ]
    return random.choice(consent_types)

def generate_random_right_info():
    """권리정보 생성"""
    right_types = [
        "저작재산권",
        "저작인격권", 
        "저작인접권",
        "초상권",
        "성명표시권",
        "동일성유지권",
        "공표권",
        "복제권",
        "배포권",
        "대여권",
        "방송권",
        "전송권"
    ]
    return random.choice(right_types)

def generate_random_project_name():
    """사업명 생성"""
    project_names = [
        "2024년 공공저작물 디지털 전환 구축 사업",
        "2025년 공공저작물 고품질 확충 사업", 
        "문화콘텐츠 디지털화 사업",
        "국가문화유산 디지털 아카이브 구축사업",
        "공공 메타데이터 표준화 사업",
        "디지털 문화콘텐츠 활용 촉진 사업",
        "AI 기반 문화콘텐츠 분석 사업",
        "메타버스 문화콘텐츠 개발 사업"
    ]
    return random.choice(project_names)

def generate_random_law_reference():
    """법령 근거 생성"""
    law_refs = [
        "저작권법 제24조의 2",
        "공공데이터의 제공 및 이용활성화에 관한 법률",
        "개인정보보호법",
        "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
        "저작권법 제78조",
        "저작권법 제11조",
        "저작권법 제35조",
        "문화예술진흥법",
        "문화산업진흥 기본법"
    ]
    return random.choice(law_refs)

def generate_random_id_num():
    """신분증번호 생성"""
    year = str(random.randint(50, 99)) if random.choice([True, False]) else str(random.randint(0, 25)).zfill(2)
    month = str(random.randint(1, 12)).zfill(2)
    day = str(random.randint(1, 28)).zfill(2)
    gender = random.choice(["1", "2", "3", "4"])
    remaining = str(random.randint(100000, 999999))
    
    return f"{year}{month}{day}-{gender}{remaining}"

def generate_random_period():
    """기간 생성"""
    start_year = random.randint(2020, 2025)
    end_year = start_year + random.randint(1, 5)
    
    patterns = [
        f"{start_year}년부터 {end_year}년까지",
        f"{start_year}.01.01 ~ {end_year}.12.31",
        f"{start_year}-{end_year}",
        f"{end_year - start_year}년간",
        f"{start_year}년 {random.randint(1,12)}월부터 {end_year}년 {random.randint(1,12)}월까지",
        f"{end_year - start_year}개년"
    ]
    return random.choice(patterns)

# 추가 엔티티 생성기들
def generate_random_title():
    """제목 생성"""
    title_templates = [
        "{}에 대한 연구", "{} 현황 분석", "{} 관리 방안", "{} 활용 가이드라인", 
        "{} 표준화 연구", "{} 디지털화 전략", "{} 개발 계획", "{} 운영 방안",
        "{} 품질 관리", "{} 보존 전략", "{} 활용 촉진", "{} 서비스 개선"
    ]
    keywords = [
        "문화콘텐츠", "저작권", "메타데이터", "디지털 아카이브", "공공저작물",
        "콘텐츠 관리", "정보 서비스", "문화유산", "디지털 변환", "온라인 서비스",
        "사용자 인터페이스", "검색 시스템", "분류 체계", "품질 보증"
    ]
    template = random.choice(title_templates)
    keyword = random.choice(keywords)
    return template.format(keyword)

def generate_random_url():
    """URL 생성"""
    domains = ["koreafilm.or.kr", "culture.go.kr", "kcti.re.kr", "copyright.or.kr", "data.go.kr"]
    paths = ["archive", "collection", "metadata", "content", "search", "view", "detail"]
    return f"https://{random.choice(domains)}/{random.choice(paths)}/{random.randint(1000, 9999)}"

def generate_random_description():
    """설명 생성"""
    descriptions = [
        "공공기관에서 제작한 문화콘텐츠입니다",
        "디지털화 작업을 통해 보존된 자료입니다", 
        "메타데이터 표준에 따라 정리된 콘텐츠입니다",
        "저작권 처리가 완료된 공공저작물입니다",
        "국민 누구나 자유롭게 이용 가능한 자료입니다"
    ]
    return random.choice(descriptions)

def generate_random_type():
    """유형 생성"""
    types = ["사진", "영상", "음성", "텍스트", "이미지", "문서", "도서", "음향자료"]
    return random.choice(types)

def generate_random_status():
    """상태 생성"""
    statuses = ["완료", "진행중", "대기", "승인", "검토중", "보류", "취소"]
    return random.choice(statuses)

def generate_random_department():
    """부서 생성"""
    departments = [
        "문화정책과", "저작권정책과", "콘텐츠진흥과", "디지털정책과", 
        "문화예술정책과", "미디어콘텐츠과", "지역문화정책과", "문화산업정책과"
    ]
    return random.choice(departments)

def generate_random_language():
    """언어 생성"""
    languages = ["한국어", "영어", "일본어", "중국어", "프랑스어", "독일어", "스페인어"]
    return random.choice(languages)

def generate_random_quantity():
    """수량 생성"""
    units = ["건", "개", "부", "권", "점", "편", "장"]
    quantity = random.randint(1, 1000)
    unit = random.choice(units)
    return f"{quantity}{unit}"

def create_char_bio_tags(text, entities):
    """문자 단위로 정확한 BIO 태그 생성 - NER 표준 준수"""
    tags = ['O'] * len(text)
    
    # 엔티티를 길이 순으로 정렬 (긴 것부터 처리하여 겹침 방지)
    sorted_entities = sorted(entities, key=lambda x: len(x[0]), reverse=True)
    
    for entity_text, entity_type in sorted_entities:
        if entity_type not in ENTITY_TYPES or not entity_text.strip():
            continue
            
        # 정확한 매칭을 위해 단어 경계 고려
        start = 0
        while True:
            pos = text.find(entity_text, start)
            if pos == -1:
                break
            
            # 이미 다른 엔티티로 태깅된 위치인지 확인
            if all(tags[i] == 'O' for i in range(pos, min(pos + len(entity_text), len(tags)))):
                # B- 태그 (첫 번째 문자) - 공백은 제외
                first_non_space_idx = pos
                while first_non_space_idx < min(pos + len(entity_text), len(tags)) and not text[first_non_space_idx].strip():
                    first_non_space_idx += 1
                
                if first_non_space_idx < len(tags):
                    tags[first_non_space_idx] = f'B-{entity_type}'
                
                # I- 태그 (나머지 문자들) - 공백은 O로 유지
                for i in range(pos, min(pos + len(entity_text), len(tags))):
                    if i == first_non_space_idx:
                        continue  # 이미 B- 태그 부여됨
                    if text[i].strip():  # 공백이 아닌 문자만 I- 태그 적용
                        tags[i] = f'I-{entity_type}'
                    # 공백은 O로 유지 (토크나이저가 공백을 offset에 포함하지 않음)
            
            start = pos + 1
    
    return tags

def generate_rich_training_data(training_dir: Path, num_samples=3000, data_source=None):
    """훈련 데이터 생성"""
    if data_source is None:
        data_source = ENTITY_DATA_POOL
    
    training_data = []
    
    # 대폭 확장된 문서 템플릿들
    contract_templates = [
        # 기본 계약/동의서 템플릿 (30개)
        "본 {CONTRACT_TYPE}는 {NAME}과 {COMPANY} 간에 체결되는 계약입니다.",
        "{NAME}({POSITION})은 {COMPANY}에서 {PERIOD} 동안 근무하기로 합니다.",
        "계약금 {MONEY}는 {DATE}까지 입금해야 합니다.",
        "연락처: {PHONE}, 이메일: {EMAIL}, 주소: {ADDRESS}",
        "신분증번호 {ID_NUM}로 신원을 확인하였습니다.",
        "계약서명일: {DATE}, 계약자: {NAME}, 회사: {COMPANY}",
        "{NAME} {POSITION}의 계약 기간은 {PERIOD}이고 월급여는 {MONEY}입니다.",
        "본인 {NAME}({ID_NUM})은 {COMPANY}의 {CONTRACT_TYPE}에 서명합니다.",
        "{COMPANY} 소속 {NAME} {POSITION}의 연락처는 {PHONE}이며 주소는 {ADDRESS}입니다.",
        "계약 체결일: {DATE}, 만료일: {PERIOD}, 당사자: {NAME}, {COMPANY}",
        "{CONTRACT_TYPE} 체결을 위해 {NAME}님께서는 {EMAIL}로 연락 바랍니다.",
        "담당자 {NAME}({POSITION})이 {DATE}에 {COMPANY}와 계약을 체결했습니다.",
        "{NAME}은 주민등록번호 {ID_NUM}, 주소 {ADDRESS}, 전화번호 {PHONE}로 등록되었습니다.",
        "계약 당사자: 갑) {COMPANY} 을) {NAME}({POSITION})",
        "{COMPANY}에서 발송한 {CONTRACT_TYPE}를 {NAME}이 {DATE}에 수령했습니다.",
        "본 계약의 보증금은 {MONEY}이며 {NAME}({ID_NUM})이 {DATE}에 납부합니다.",
        "{NAME} {POSITION}과 {COMPANY} 간의 {CONTRACT_TYPE} 유효기간: {PERIOD}",
        "계약자 정보 - 성명: {NAME}, 직책: {POSITION}, 연락처: {PHONE}, 이메일: {EMAIL}",
        "{COMPANY} 대표 {NAME}이 {DATE}자로 {CONTRACT_TYPE}를 작성하였습니다.",
        "해당 {CONTRACT_TYPE}는 {NAME}({ID_NUM})과 {COMPANY} 간 {PERIOD} 유효합니다.",
        "갑: {COMPANY}\n  대표자: {NAME}\n  주소: {ADDRESS}\n  전화: {PHONE}",
        "을: {NAME}\n  주민등록번호: {ID_NUM}\n  주소: {ADDRESS}\n  연락처: {PHONE}",
        "계약 제목: {CONTRACT_TYPE}\n작성일: {DATE}\n만료일: {PERIOD}",
        "계약자: {NAME} (주민등록번호: {ID_NUM})\n주소: {ADDRESS}\n연락처: {PHONE}",
        "계약 조건:\n1. 계약금: {MONEY}\n2. 계약 기간: {PERIOD}\n3. 담당자: {NAME}",
        "연락처 정보:\n- 이름: {NAME}\n- 직책: {POSITION}\n- 전화: {PHONE}\n- 이메일: {EMAIL}",
        "당사자 정보\n\n성명: {NAME}\n직책: {POSITION}\n소속: {COMPANY}\n연락처: {PHONE}",
        "계약서 작성 정보:\n작성일: {DATE}\n작성자: {NAME}\n소속: {COMPANY}",
        "개인정보:\n- 성명: {NAME}\n- 주민등록번호: {ID_NUM}\n- 주소: {ADDRESS}\n- 전화: {PHONE}",
        "계약 내역\n계약명: {CONTRACT_TYPE}\n계약자: {NAME}\n계약금: {MONEY}\n기간: {PERIOD}",
        
        # 저작권 관련 템플릿 (40개)
        "{NAME}은 {RIGHT_INFO}을 {COMPANY}에게 양도합니다.",
        "{CONSENT_TYPE}에 {NAME}({ID_NUM})이 동의합니다.",
        "{PROJECT_NAME}과 관련하여 {RIGHT_INFO} 처리를 진행합니다.",
        "{LAW_REFERENCE}에 근거하여 {RIGHT_INFO}을 행사합니다.",
        "{NAME}의 {RIGHT_INFO} 양도 기간은 {PERIOD}입니다.",
        "{NAME}님의 {RIGHT_INFO}을 {COMPANY}에서 {PERIOD} 동안 사용합니다.",
        "{PROJECT_NAME} 사업에서 {NAME}의 {RIGHT_INFO} 활용에 {MONEY} 지급됩니다.",
        "{LAW_REFERENCE} 제2항에 따라 {NAME}({ID_NUM})의 {RIGHT_INFO}을 보호합니다.",
        "{CONSENT_TYPE}에 명시된 {RIGHT_INFO} 사용료는 {MONEY}입니다.",
        "{COMPANY}는 {NAME}에게 {RIGHT_INFO} 사용 대가로 {MONEY}를 {DATE}에 지급합니다.",
        "{NAME} {POSITION}이 작성한 저작물의 {RIGHT_INFO}을 {COMPANY}가 양도받습니다.",
        "{PROJECT_NAME}에서 {RIGHT_INFO} 처리를 담당하는 부서는 {DEPARTMENT}입니다.",
        "{LAW_REFERENCE}에 의거, {NAME}의 {RIGHT_INFO} 보호 기간은 {PERIOD}입니다.",
        "저작자 {NAME}({ID_NUM})이 {COMPANY}와 {RIGHT_INFO} 양도 계약을 체결했습니다.",
        "{CONSENT_TYPE} 작성일: {DATE}, 저작자: {NAME}, 양수인: {COMPANY}",
        "{PROJECT_NAME} 관련 {RIGHT_INFO} 처리 담당자: {NAME} {POSITION}({PHONE})",
        "{COMPANY}에서 {NAME}의 {RIGHT_INFO} 사용 승인을 {DATE}에 받았습니다.",
        "본 {CONSENT_TYPE}는 {LAW_REFERENCE}를 근거로 작성되었습니다.",
        "{NAME}이 창작한 저작물에 대한 {RIGHT_INFO}을 {PERIOD} 동안 양도합니다.",
        "{RIGHT_INFO} 양도 계약서에 {NAME}({ID_NUM})과 {COMPANY} 대표가 서명했습니다.",
        "{PROJECT_NAME}의 {RIGHT_INFO} 관리는 {COMPANY} {DEPARTMENT}에서 담당합니다.",
        "{NAME} {POSITION}의 {RIGHT_INFO}에 대한 사용료는 {MONEY}으로 책정되었습니다.",
        "{CONSENT_TYPE}에 따라 {NAME}의 {RIGHT_INFO}을 {COMPANY}에서 활용합니다.",
        "{DATE}부터 {PERIOD}까지 {NAME}의 {RIGHT_INFO}을 {COMPANY}가 사용합니다.",
        "{LAW_REFERENCE} 규정에 따라 {NAME}({ID_NUM})의 {RIGHT_INFO}을 보호합니다.",
        "저작권 정보:\n- 저작자: {NAME}\n- 권리 종류: {RIGHT_INFO}\n- 양수인: {COMPANY}",
        "{CONSENT_TYPE}\n\n성명: {NAME}\n생년월일: {ID_NUM}\n주소: {ADDRESS}",
        "{RIGHT_INFO} 양도 조건:\n- 양도인: {NAME}\n- 양수인: {COMPANY}\n- 기간: {PERIOD}",
        "{CONSENT_TYPE} 동의 내용:\n\n동의자: {NAME}({ID_NUM})\n동의일: {DATE}",
        "법적 근거: {LAW_REFERENCE}\n적용 기간: {PERIOD}\n담당 기관: {COMPANY}",
        "{PROJECT_NAME} 참여자 명단\n\n이름: {NAME}\n소속: {COMPANY}\n직책: {POSITION}",
        "저작물 정보:\n제목: {TITLE}\n저작자: {NAME}\n권리: {RIGHT_INFO}\n기간: {PERIOD}",
        "{LAW_REFERENCE}에 따라 {PROJECT_NAME}을 수행합니다.",
        "근거 법령: {LAW_REFERENCE}, 사업명: {PROJECT_NAME}",
        "{LAW_REFERENCE} 제정에 따른 {RIGHT_INFO} 처리 방침",
        "법적 근거: {LAW_REFERENCE}\n적용 대상: {PROJECT_NAME}\n담당: {COMPANY}",
        "{LAW_REFERENCE}의 규정에 따라 {NAME}의 {RIGHT_INFO}을 보호합니다.",
        "{PROJECT_NAME}은 {LAW_REFERENCE}를 근거로 {COMPANY}에서 수행합니다.",
        "{RIGHT_INFO} 관련 {LAW_REFERENCE} 적용으로 {NAME}의 권리를 보장합니다.",
        "{COMPANY}는 {LAW_REFERENCE}에 근거하여 {NAME}과 {PROJECT_NAME} 계약을 체결합니다.",
        
        # 메타데이터 기반 템플릿 (25개)
        "제목: {TITLE}",
        "URL: {URL}",  
        "설명: {DESCRIPTION}",
        "유형: {TYPE}, 상태: {STATUS}",
        "담당부서: {DEPARTMENT}, 언어: {LANGUAGE}",
        "수량: {QUANTITY}",
        "문서 제목: {TITLE}, 담당: {DEPARTMENT}, 상태: {STATUS}",
        "{TITLE} 관련 자료는 {URL}에서 확인 가능합니다.",
        "{DESCRIPTION} - 유형: {TYPE}, 수량: {QUANTITY}",
        "{DEPARTMENT}에서 관리하는 {LANGUAGE} 자료 현황: {STATUS}",
        "{TYPE} 자료 {QUANTITY}가 {DEPARTMENT}에 등록되었습니다.",
        "자료명: {TITLE}, 설명: {DESCRIPTION}, 언어: {LANGUAGE}",
        "{STATUS} 상태의 {TYPE} 자료가 {URL}에 업로드되었습니다.",
        "{DEPARTMENT} 담당 {TITLE} 자료의 수량은 {QUANTITY}입니다.",
        "메타데이터 - 제목: {TITLE}, 유형: {TYPE}, 상태: {STATUS}, URL: {URL}",
        "문서 제목: {TITLE}\n설명: {DESCRIPTION}\n유형: {TYPE}\n상태: {STATUS}",
        "{DEPARTMENT}에서 관리하는 {LANGUAGE} {TYPE} 자료 {QUANTITY}입니다.",
        "자료 관리 정보\n제목: {TITLE}\nURL: {URL}\n담당부서: {DEPARTMENT}\n상태: {STATUS}",
        "{LANGUAGE} 언어로 작성된 {TYPE} 자료가 {DEPARTMENT}에 {QUANTITY} 등록되었습니다.",
        "메타데이터 현황:\n- 제목: {TITLE}\n- 설명: {DESCRIPTION}\n- 상태: {STATUS}",
        "{DEPARTMENT} 관리 자료\n유형: {TYPE}\n수량: {QUANTITY}\n언어: {LANGUAGE}\nURL: {URL}",
        "자료 검색 정보: {TITLE} - {DESCRIPTION} ({TYPE}, {STATUS})",
        "{QUANTITY}개의 {TYPE} 자료가 {DEPARTMENT}에서 {STATUS} 상태로 관리중입니다.",
        "디지털 자료 정보\nURL: {URL}\n제목: {TITLE}\n유형: {TYPE}\n언어: {LANGUAGE}",
        "{DEPARTMENT}에서 제공하는 {TITLE} 자료의 상세 설명: {DESCRIPTION}",
        
        # 복합 정보 템플릿들 (50개)
        "{COMPANY} {NAME} {POSITION}입니다. 연락처는 {PHONE}이고 이메일은 {EMAIL}입니다.",
        "{PROJECT_NAME}에서 {NAME}님께서 {ADDRESS}에서 {PERIOD} 동안 {MONEY} 지급받기로 합니다.",
        "{DATE}부터 효력이 발생하는 {CONTRACT_TYPE}에 대해 {NAME}({ID_NUM})이 동의합니다.",
        "{COMPANY}에서 {NAME} {POSITION}이 {EMAIL}로 {CONSENT_TYPE}를 발송했습니다.",
        "{PROJECT_NAME} 담당자 {NAME}({PHONE})이 {COMPANY} {DEPARTMENT}에 소속되어 있습니다.",
        "{NAME} {POSITION}의 {CONTRACT_TYPE} 체결로 {MONEY}가 {DATE}에 지급됩니다.",
        "{COMPANY}의 {PROJECT_NAME}에서 {NAME}({ID_NUM})이 {RIGHT_INFO} 양도에 동의했습니다.",
        "{CONSENT_TYPE} 작성자: {NAME}, 소속: {COMPANY}, 연락처: {PHONE}, 이메일: {EMAIL}",
        "{DATE} 체결된 {CONTRACT_TYPE}에 따라 {NAME}이 {MONEY} 수령 예정입니다.",
        "{COMPANY} {DEPARTMENT} 소속 {NAME} {POSITION}이 {PROJECT_NAME}을 담당합니다.",
        "{NAME}({ID_NUM})이 거주하는 {ADDRESS}로 {CONTRACT_TYPE}를 {DATE}에 발송했습니다.",
        "{PROJECT_NAME}의 {RIGHT_INFO} 처리를 위해 {NAME}과 {COMPANY}가 계약했습니다.",
        "{CONSENT_TYPE}에 {NAME} {POSITION}({PHONE})이 {DATE}에 서명했습니다.",
        "{COMPANY}에서 발주한 {PROJECT_NAME}에 {NAME}이 {MONEY}로 참여합니다.",
        "{NAME}의 {ADDRESS} 주소지로 {CONTRACT_TYPE} 원본을 {DATE}에 우편발송했습니다.",
        "{RIGHT_INFO} 양도 계약자 {NAME}({ID_NUM})의 연락처는 {PHONE}입니다.",
        "{PROJECT_NAME} 관련 {CONSENT_TYPE}를 {NAME}이 {EMAIL}로 제출했습니다.",
        "{COMPANY} 대표 {NAME} {POSITION}이 {DATE}에 {CONTRACT_TYPE}를 체결했습니다.",
        "{NAME}({ID_NUM})과 {COMPANY} 간 {CONTRACT_TYPE}의 계약 기간은 {PERIOD}입니다.",
        "{PROJECT_NAME}에서 {RIGHT_INFO} 사용료 {MONEY}를 {NAME}에게 지급합니다.",
        "수탁기관: {COMPANY}\n담당자: {NAME} {POSITION}\n사업명: {PROJECT_NAME}",
        "본 {CONTRACT_TYPE}의 유효기간은 {PERIOD}이며, 계약일은 {DATE}입니다.",
        "{NAME}은 {COMPANY}와 체결한 {CONTRACT_TYPE}에 따라 {RIGHT_INFO}을 양도합니다.",
        "사업명: {PROJECT_NAME}\n수행기관: {COMPANY}\n책임자: {NAME} {POSITION}",
        "{PROJECT_NAME}\n\n가. 사업 기간: {PERIOD}\n나. 사업비: {MONEY}\n다. 담당: {DEPARTMENT}",
        "사업 담당자\n이름: {NAME}\n부서: {DEPARTMENT}\n연락처: {PHONE}\n이메일: {EMAIL}",
        "{COMPANY} 소속 직원 정보\n성명: {NAME}\n직책: {POSITION}\n입사일: {DATE}",
        "{CONSENT_TYPE} 제출 마감일은 {DATE}이며 담당자는 {NAME} {POSITION}입니다.",
        "{COMPANY} {DEPARTMENT}에서 {NAME}의 {CONTRACT_TYPE}를 {DATE}에 승인했습니다.",
        "{NAME} {POSITION}({EMAIL})이 {PROJECT_NAME}의 {RIGHT_INFO} 처리를 담당합니다.",
        "{ADDRESS}에 거주하는 {NAME}({ID_NUM})이 {COMPANY}와 계약을 체결했습니다.",
        "{DATE} 만료되는 {CONTRACT_TYPE}에 대해 {NAME}이 {MONEY} 청구할 예정입니다.",
        "{PROJECT_NAME}의 예산 {MONEY}로 {NAME}과 {COMPANY}가 {CONSENT_TYPE}를 작성했습니다.",
        "{COMPANY}의 {NAME} {POSITION}이 {PHONE}으로 {CONTRACT_TYPE} 관련 연락을 했습니다.",
        "{RIGHT_INFO} 관련 {LAW_REFERENCE}에 따라 {NAME}({ID_NUM})의 권리를 보호합니다.",
        "{PROJECT_NAME} 완료 후 {NAME}에게 {MONEY} 지급 예정이며 연락처는 {EMAIL}입니다.",
        "{CONSENT_TYPE} 서명자 {NAME}의 주소 {ADDRESS}로 계약서 사본을 발송했습니다.",
        "법령 적용: {LAW_REFERENCE}\n사업: {PROJECT_NAME}\n권리: {RIGHT_INFO}",
        "{LAW_REFERENCE}에 의거하여 {COMPANY}가 {PROJECT_NAME}을 진행합니다.",
        "{NAME}의 {RIGHT_INFO}은 {LAW_REFERENCE}에 따라 {PERIOD} 동안 보호됩니다.",
        "관련 법령: {LAW_REFERENCE}\n적용 기간: {PERIOD}\n적용 기관: {COMPANY}",
        "{PROJECT_NAME} 수행 시 {LAW_REFERENCE}를 준수하여 {RIGHT_INFO}을 처리합니다.",
        "{LAW_REFERENCE} 시행에 따라 {COMPANY}에서 {NAME}의 권리를 보호합니다.",
        "법적 보호: {LAW_REFERENCE} 적용으로 {PROJECT_NAME}의 {RIGHT_INFO} 보장",
        "관리 부서: {DEPARTMENT}\n자료명: {TITLE}\n설명: {DESCRIPTION}\n수량: {QUANTITY}",
        "{STATUS} 상태의 {TYPE} 자료를 {DEPARTMENT}에서 {QUANTITY} 관리하고 있습니다.",
        "온라인 자료 링크: {URL}\n자료 설명: {DESCRIPTION}\n담당: {DEPARTMENT}",
        "{TITLE} 관련 {TYPE} 자료가 {LANGUAGE}로 {QUANTITY} 제작되었습니다.",
        "자료 업로드 정보: {URL}, 제목: {TITLE}, 상태: {STATUS}, 담당: {DEPARTMENT}",
        "{DEPARTMENT}의 {LANGUAGE} {TYPE} 컬렉션에 {TITLE} 자료 {QUANTITY}가 추가되었습니다."
    ]
    
    # 통합 엔티티 생성기 딕셔너리
    entity_generators = {
        # 기본 엔티티
        "NAME": generate_random_name,
        "PHONE": generate_random_phone, 
        "COMPANY": generate_random_company,
        "ADDRESS": generate_random_address,
        "DATE": generate_random_date,
        "MONEY": generate_random_money,
        "EMAIL": generate_random_email,
        "POSITION": generate_random_position,
        "PERIOD": generate_random_period,
        "ID_NUM": generate_random_id_num,
        
        # OCR 기반 엔티티
        "CONTRACT_TYPE": generate_random_contract_type,
        "CONSENT_TYPE": generate_random_consent_type,
        "RIGHT_INFO": generate_random_right_info,
        "PROJECT_NAME": generate_random_project_name,
        "LAW_REFERENCE": generate_random_law_reference,
        
        # 메타데이터 기반 엔티티
        "TITLE": generate_random_title,
        "URL": generate_random_url,
        "DESCRIPTION": generate_random_description,
        "TYPE": generate_random_type,
        "STATUS": generate_random_status,
        "DEPARTMENT": generate_random_department,
        "LANGUAGE": generate_random_language,
        "QUANTITY": generate_random_quantity
    }
    
    for i in range(num_samples):
        if i % 5000 == 0 and i > 0:
            print(f"   진행: {i:,}/{num_samples:,} ({(i/num_samples*100):.0f}%)")
        
        # 다중 템플릿 조합 전략 (40% 확률로 복합 문서)
        if random.random() < 0.4:
            selected_templates = random.sample(contract_templates, k=random.randint(2, 4))
            template = "\n".join(selected_templates)
        else:
            template = random.choice(contract_templates)
        
        # 엔티티 생성
        entities = []
        text = template
        
        # 템플릿의 플레이스홀더를 실제 값으로 교체
        for placeholder_key in entity_generators.keys():
            placeholder = f"{{{placeholder_key}}}"
            while placeholder in text:
                entity_value = entity_generators[placeholder_key]()
                entities.append((entity_value, placeholder_key))
                text = text.replace(placeholder, entity_value, 1)
        
        # 추가 랜덤 엔티티들 삽입 (60% 확률)
        if random.random() < 0.6:
            available_entities = list(entity_generators.keys())
            additional_entities = random.sample(available_entities, k=random.randint(1, 8))  # 최대 8개
            for entity_type in additional_entities:
                if entity_type in entity_generators:
                    entity_value = entity_generators[entity_type]()
                    entities.append((entity_value, entity_type))
                    connectors = [" 또한 ", " 추가로 ", " 참고: ", " 기타 ", " 별첨: ", " 첨부: ", " 보충: ", " 부가정보: "]
                    text += f"{random.choice(connectors)}{entity_value}"
        
        # OCR 특성 반영 (20% 확률)
        if random.random() < 0.2:
            ocr_additions = [
                "\n□ 동의함 □ 동의하지 않음",
                "\n※ 주의사항",
                "\n- 첨부서류:",
                "\n* 특이사항:",
                "\n[인장]",
                "\n서명: ___________",
                "\n날짜: ___________",
                "\n위 내용에 동의하며 서명합니다.",
                "\n본 계약서는 2부를 작성하여 각각 보관합니다."
            ]
            text += random.choice(ocr_additions)
        
        # BIO 태깅
        bio_tags = create_char_bio_tags(text, entities)
        
        # 데이터 검증
        if len(text) != len(bio_tags):
            continue
        
        # 품질 체크: 최소 엔티티 개수와 다양성
        unique_entity_types = set([entity[1] for entity in entities])
        if len(entities) >= 3 and len(unique_entity_types) >= 2 and any(tag != 'O' for tag in bio_tags):
            training_data.append({
                'text': text,
                'bio_tags': bio_tags,
                'entities': entities,
                'entity_diversity': len(unique_entity_types)
            })
    
    # print(f"완료: {len(training_data):,}개의 고품질 학습 샘플 생성")
    
    # 훈련 데이터를 파일로 저장
    training_data_file = training_dir / f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(training_data_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    # print(f"훈련 데이터 저장: {training_data_file}")
    
    return training_data

# CoNLL 형식 저장/로드 함수들
def save_to_conll(data, filename):
    """CoNLL 형식으로 데이터 저장"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        for sample in data:
            text = sample['text']
            bio_tags = sample['bio_tags']
            
            for char, tag in zip(text, bio_tags):
                if char.strip():  # 공백이 아닌 문자만
                    f.write(f"{char}\t{tag}\n")
                else:
                    f.write(f"{char}\t{tag}\n")
            f.write("\n")  # 문장 구분
    
    # print(f"저장 완료: {len(data):,}개 샘플을 {filename}에 저장")

def load_conll_data(filename):
    """CoNLL 형식 데이터 읽기"""
    sentences = []
    labels = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        current_sentence = []
        current_labels = []
        
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    char, label = parts
                    current_sentence.append(char)
                    current_labels.append(label)
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
        
        # 마지막 문장 처리
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
    
    return sentences, labels

def align_labels_with_tokens(tokenizer, texts, labels, max_length=512):
    """토큰화된 입력과 라벨 정렬"""
    tokenized_inputs = tokenizer(
        [''.join(text) for text in texts],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_offsets_mapping=True,
        is_split_into_words=False
    )
    
    aligned_labels = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        offset_mapping = tokenized_inputs['offset_mapping'][i]
        
        aligned_label = []
        for j, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:  # [CLS], [SEP], [PAD]
                aligned_label.append(-100)
            else:
                # Offset 범위의 첫 번째 문자 라벨 사용
                # 예: (0,2) → label[0] 사용 (B-NAME)
                if start < len(label):
                    aligned_label.append(LABEL_TO_ID[label[start]])
                else:
                    aligned_label.append(LABEL_TO_ID['O'])
        
        aligned_labels.append(aligned_label)
    
    tokenized_inputs['labels'] = aligned_labels
    return tokenized_inputs

def compute_metrics(eval_pred):
    """고성능 평가 메트릭 계산 (seqeval 사용)"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = []
    true_labels = []
    
    for prediction, label in zip(predictions, labels):
        true_pred = []
        true_lab = []
        
        for p, l in zip(prediction, label):
            if l != -100:
                true_pred.append(ID_TO_LABEL[p])
                true_lab.append(ID_TO_LABEL[l])
        
        if true_pred and true_lab:
            true_predictions.append(true_pred)
            true_labels.append(true_lab)
    
    if not true_predictions or not true_labels:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0
        }
    
    try:
        results = {
            "precision": seqeval_precision(true_labels, true_predictions),
            "recall": seqeval_recall(true_labels, true_predictions),
            "f1": seqeval_f1(true_labels, true_predictions),
            "accuracy": seqeval_accuracy(true_labels, true_predictions)
        }
        return results
    except Exception as e:
        print(f"경고: 메트릭 계산 오류: {e}")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0
        }

def train_ner_model():
    """고성능 NER Fine-Tuning"""
    print("\n" + "=" * 60)
    print("NER 모델 훈련 시작")
    print("=" * 60)
    
    start_time = time.time()
    config = TrainingConfig()
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({memory_gb:.1f}GB)")
        torch.cuda.empty_cache()
    else:
        print("CPU 모드")
    
    # 경로 설정 - 모델명 기반 디렉토리 구조
    current_dir = Path(__file__).parent
    api_dir = current_dir.parent.parent  # api 디렉토리
    
    # 모델명 추출 (예: klue/roberta-large -> klue-roberta-large)
    model_name_safe = config.model_name.replace('/', '-')
    
    # 모델 저장 위치: api/models/ner/{model_name}
    models_base_dir = api_dir / "models" / "ner"
    models_base_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir = models_base_dir / model_name_safe
    model_output_dir.mkdir(exist_ok=True)
    
    # 훈련 데이터는 api/module/ner/training/{model_name}에 저장
    training_base_dir = current_dir / "training"
    training_base_dir.mkdir(exist_ok=True)
    training_dir = training_base_dir / model_name_safe
    training_dir.mkdir(exist_ok=True)
    
    print(f"모델: {config.model_name}")
    
    # 데이터 파일 경로
    temp_train_file = training_dir / "train.txt"
    temp_val_file = training_dir / "validation.txt"
    temp_test_file = training_dir / "test.txt"
    
    # 데이터가 이미 존재하면 재사용
    if temp_train_file.exists() and temp_val_file.exists() and temp_test_file.exists():
        print(f"\n기존 데이터 사용 중...")
        print(f"✓ 훈련 데이터: {temp_train_file}")
        print(f"✓ 평가 데이터: {temp_val_file}")
    else:
        # 1. 명사 데이터 분할 (훈련 80% / 평가 20%)
        print(f"\n명사 데이터 분할 중... (훈련 80% / 평가 20%)")
        train_data_source, eval_data_source = split_entity_data(train_ratio=0.8, random_seed=42)
        print(f"✓ 훈련용 명사: {sum(len(v) for v in train_data_source.values())}개")
        print(f"✓ 평가용 명사: {sum(len(v) for v in eval_data_source.values())}개")
        
        # 2. 훈련 데이터 생성 (80% 명사 사용)
        print(f"\n훈련 데이터 생성 중...")
        training_data = generate_rich_training_data(training_dir, num_samples=3000, data_source=train_data_source)
        
        if not training_data:
            print("오류: 학습 데이터 생성 실패!")
            return False
        
        # 3. 평가 데이터 생성 (20% 명사 + 훈련 데이터에서 랜덤 20%)
        print(f"\n평가 데이터 생성 중...")
        eval_sample_count = int(3000 * 0.2)
        
        # 평가용 데이터: 새로운 명사(20%) + 훈련 데이터 일부(20%)
        eval_data_new = generate_rich_training_data(training_dir, num_samples=eval_sample_count, data_source=eval_data_source)
        eval_data_overlap = random.sample(training_data, k=int(len(training_data) * 0.2))
        evaluation_data = eval_data_new + eval_data_overlap
        random.shuffle(evaluation_data)
        
        print(f"✓ 훈련 데이터: {len(training_data):,}개")
        print(f"✓ 평가 데이터: {len(evaluation_data):,}개 (신규 {len(eval_data_new):,} + 중복 {len(eval_data_overlap):,})")
        
        # 4. CoNLL 형식으로 저장
        save_to_conll(training_data, str(temp_train_file))
        save_to_conll(evaluation_data, str(temp_val_file))
        save_to_conll(evaluation_data, str(temp_test_file))
        
        print(f"데이터 저장 완료")
    
    # 5. 데이터 로드
    train_file = training_dir / "train.txt"
    val_file = training_dir / "validation.txt"
    test_file = training_dir / "test.txt"
    
    train_sentences, train_labels = load_conll_data(str(train_file))
    val_sentences, val_labels = load_conll_data(str(val_file))
    test_sentences, test_labels = load_conll_data(str(test_file))
    
    print(f"✓ 훈련 데이터: {len(train_sentences):,}개 문장")
    print(f"✓ 검증 데이터: {len(val_sentences):,}개 문장")
    print(f"✓ 테스트 데이터: {len(test_sentences):,}개 문장")
    
    # 데이터만 생성하고 종료
    if os.getenv("GENERATE_DATA_ONLY") == "1":
        print("\n데이터 생성 완료 (훈련 스킵)")
        return True
    
    # 라벨 분포 확인
    all_labels = [label for sentence_labels in train_labels for label in sentence_labels]
    entity_labels = [label for label in all_labels if label != 'O']
    
    if len(entity_labels) == 0:
        print("오류: 엔티티 라벨이 없습니다!")
        return False
    
    # 5. 모델 초기화
    print(f"\n모델 초기화 중... ({config.model_name})")
    
    # 로컬 모델 경로 사용 (test.py의 Step 2에서 복사됨)
    model_path = str(model_output_dir)
    
    # 로컬 파일이 없으면 온라인에서 다운로드
    if not (Path(model_path) / "config.json").exists():
        print(f"  로컬 모델 없음, 온라인에서 로드: {config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=len(BIO_LABELS),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout
        )
    else:
        print(f"  로컬 모델 로드: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(BIO_LABELS),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
            ignore_mismatched_sizes=True,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout
        )
    
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"모델 준비 완료: {total_params:,} 파라미터, {len(BIO_LABELS)}개 라벨")
    
    # 6. 데이터셋 준비 (이미 분할된 데이터 사용)
    
    print(f"데이터셋 토큰화 중...")
    train_encodings = align_labels_with_tokens(tokenizer, train_sentences, train_labels, config.max_length)
    val_encodings = align_labels_with_tokens(tokenizer, val_sentences, val_labels, config.max_length)
    
    # HuggingFace Dataset 생성
    train_dataset = HFDataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_encodings['labels']
    })
    
    val_dataset = HFDataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'], 
        'labels': val_encodings['labels']
    })
    
    print(f"데이터셋 준비 완료: 훈련 {len(train_dataset):,}개 / 검증 {len(val_dataset):,}개")
    
    # 7. 최적화된 훈련 설정
    
    # Epoch당 스텝 계산
    steps_per_epoch = len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)
    eval_steps = steps_per_epoch
    save_steps = steps_per_epoch * 2
    print(f"평가 주기: {eval_steps}스텝 (매 epoch)")
    print(f"저장 주기: {save_steps}스텝 (2 epoch마다)")
    
    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,  # 과적합 방지
        learning_rate=config.learning_rate,
        logging_dir=str(training_dir / 'logs'),
        logging_steps=eval_steps // 2,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=None,
        fp16=config.fp16,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        gradient_checkpointing=True if torch.cuda.is_available() else False,
        lr_scheduler_type="cosine",  # 안정적인 학습
        optim="adamw_torch",
        disable_tqdm=False,
        dataloader_drop_last=True,
        prediction_loss_only=False,
        label_smoothing_factor=config.label_smoothing  # 과적합 방지
    )
    
    # 8. 트레이너 설정
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)
    
    # Early Stopping 콜백 (성능 개선 없으면 조기 종료)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks  # 콜백 추가
    )
    
    # 9. 고성능 훈련 실행
    print(f"\n훈련 시작: {config.num_epochs} epochs, {len(train_sentences):,} 샘플")
    print(f"디바이스: {trainer.args.device}, FP16: {'ON' if config.fp16 else 'OFF'}")
    print(f"예상 시간: ~20-30분\n")
    
    try:
        # 훈련 시작
        trainer.train()
        
        # 10. 최종 모델 저장
        print(f"\n모델 저장 중...")
        trainer.save_model()
        tokenizer.save_pretrained(str(model_output_dir))
        
        # 라벨 매핑 저장
        label_info = {
            'id2label': ID_TO_LABEL,
            'label2id': LABEL_TO_ID,
            'label_list': BIO_LABELS,
            'entity_types': ENTITY_TYPES,
            'num_labels': len(BIO_LABELS)
        }
        
        with open(model_output_dir / 'label_map.json', 'w', encoding='utf-8') as f:
            json.dump(label_info, f, ensure_ascii=False, indent=2)
        
        # 11. 최종 평가
        eval_results = trainer.evaluate()
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("NER 모델 훈련 완료!")
        print("=" * 60)
        print(f"저장 위치: {model_output_dir}")
        print(f"훈련 데이터: {len(train_sentences):,}개 샘플")
        print(f"소요 시간: {total_time/60:.1f}분")
        print(f"F1 Score: {eval_results.get('eval_f1', 0):.4f}")
        print(f"Precision: {eval_results.get('eval_precision', 0):.4f}")
        print(f"Recall: {eval_results.get('eval_recall', 0):.4f}")
        print("=" * 60 + "\n")
        
        return True
            
    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = train_ner_model()
        if success:
            print("NER Fine-Tuning 성공!")
        else:
            print("NER Fine-Tuning 실패!")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()