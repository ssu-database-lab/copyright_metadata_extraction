#!/usr/bin/env python3
"""
통합 NER 시스템 - 완전 독립 버전
ner_train.py와 함께 사용하여 완전한 NER 시스템 구성
- 강화된 이중 예측 시스템 (B-I-O 모델 + 정규표현식)
- 자동 모델 훈련 기능
- 높은 정확도 보장
- 간단한 API 인터페이스
"""

import os
import sys
import json
import csv
import time
import logging
import re
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple
from tqdm import tqdm

# PyTorch 및 Transformers
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import warnings
warnings.filterwarnings("ignore")

# 로깅 설정
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 엔티티 타입 정의 (ner_train.py와 동일)
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

# 기본 설정
DEFAULT_MAX_LENGTH = 512

# model_config.json에서 기본 모델 로드
def load_default_model_name():
    """model_config.json에서 기본 모델 이름 로드"""
    try:
        config_path = Path(__file__).parent.parent.parent / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            default_model = config.get("ner", {}).get("default_model", "klue-roberta-large")
            print(f"✓ 기본 모델 설정: {default_model}")
            return default_model
    except Exception as e:
        print(f"⚠️  model_config.json 로드 실패: {e}, 기본값 사용")
    return "klue-roberta-large"

DEFAULT_MODEL_NAME = load_default_model_name()

def check_system_requirements(verbose: bool = False):
    """시스템 요구사항 확인"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU 사용 가능: {gpu_name} ({memory_gb:.1f}GB)")
        else:
            print("CPU 모드로 실행")
    
    return device

def get_model_path(model_name: str = DEFAULT_MODEL_NAME) -> Path:
    """
    모델 저장 경로 반환 및 준비
    
    로직:
    1. models/ner/{model_name}/ 확인
       - 있으면 → 사용
       - 없으면 → 2단계
    
    2. model_downloaded/{model_name}/ 확인
       - 있으면 → models/ner/로 복사 후 사용
       - 없으면 → 3단계
    
    3. Hugging Face에서 다운로드
       - model_downloaded/로 다운로드
       - models/ner/로 복사 후 사용
    
    최종적으로 항상 models/ner/{model_name}/ 경로 반환
    """
    current_dir = Path(__file__).parent
    api_dir = current_dir.parent.parent
    
    # 모델명에서 슬래시를 대시로 변경 (예: klue/roberta-large -> klue-roberta-large)
    model_name_safe = model_name.replace('/', '-')
    
    # 목표 경로: models/ner/{model_name}
    models_ner_dir = api_dir / "models" / "ner" / model_name_safe
    models_ner_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # 1단계: models/ner/{model_name}이 이미 있는지 확인
    if models_ner_dir.exists() and (models_ner_dir / "config.json").exists():
        print(f"✓ 모델 발견: {models_ner_dir}")
        return models_ner_dir
    
    # 2단계: model_downloaded/{model_name} 확인
    model_downloaded_dir = api_dir / "model_downloaded" / model_name_safe
    if model_downloaded_dir.exists() and (model_downloaded_dir / "config.json").exists():
        print(f"✓ 다운로드된 모델 발견: {model_downloaded_dir}")
        print(f"📦 모델 복사 중: {model_downloaded_dir} → {models_ner_dir}")
        
        import shutil
        if models_ner_dir.exists():
            shutil.rmtree(models_ner_dir)
        shutil.copytree(model_downloaded_dir, models_ner_dir)
        
        print(f"✓ 모델 복사 완료: {models_ner_dir}")
        return models_ner_dir
    
    # 3단계: Hugging Face에서 다운로드
    print(f"⚠️  로컬에 모델 없음: {model_name}")
    print(f"📥 Hugging Face에서 다운로드 중...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        # model_downloaded로 다운로드
        model_downloaded_dir.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"  → 다운로드 위치: {model_downloaded_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        # 저장
        tokenizer.save_pretrained(str(model_downloaded_dir))
        model.save_pretrained(str(model_downloaded_dir))
        
        print(f"✓ 다운로드 완료: {model_downloaded_dir}")
        
        # models/ner/로 복사
        print(f"📦 모델 복사 중: {model_downloaded_dir} → {models_ner_dir}")
        
        import shutil
        if models_ner_dir.exists():
            shutil.rmtree(models_ner_dir)
        shutil.copytree(model_downloaded_dir, models_ner_dir)
        
        print(f"✓ 모델 복사 완료: {models_ner_dir}")
        return models_ner_dir
        
    except Exception as e:
        print(f"❌ 모델 다운로드 실패: {e}")
        # 실패해도 경로는 반환 (나중에 훈련 시 재시도)
        return models_ner_dir

def load_model_and_tokenizer(model_path: Path, verbose: bool = True):
    """모델과 토크나이저 로드"""
    if verbose:
        print(f"모델 로드 중: {model_path}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    # 모델 로드
    model = AutoModelForTokenClassification.from_pretrained(str(model_path))
    
    # 라벨 맵 로드
    label_map_file = model_path / "label_map.json"
    if label_map_file.exists():
        with open(label_map_file, 'r', encoding='utf-8') as f:
            label_info = json.load(f)
        id2label = label_info['id2label']
        id2label = {int(k): v for k, v in id2label.items()}
    else:
        # 기본 라벨 맵 생성
        labels = ["O"]
        for entity in ENTITY_TYPES:
            labels.extend([f"B-{entity}", f"I-{entity}"])
        id2label = {i: label for i, label in enumerate(labels)}
    
    # GPU 사용 가능하면 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    if verbose:
        print(f"모델 로드 완료 ({device})")
    return tokenizer, model, id2label, device

def extract_entities_by_bio_tagging(text: str, tokenizer, model, id2label: dict, device) -> Set[Tuple[str, str]]:
    """B-I-O 태깅 기반 엔티티 추출 (강화된 버전)"""
    entities = set()
    
    # 텍스트를 적절한 크기로 분할
    sentences = split_text_smartly(text, DEFAULT_MAX_LENGTH)
    
    for sentence in sentences:
        if len(sentence.strip()) < 3:
            continue
            
        try:
            # 토큰화
            encoding = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=DEFAULT_MAX_LENGTH,
                add_special_tokens=True,
                return_offsets_mapping=True,
                padding=True
            )
            
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            offset_mapping = encoding["offset_mapping"][0]
            
            # 모델 예측
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.softmax(outputs.logits, dim=-1)
                predicted_labels = torch.argmax(predictions, dim=-1).squeeze(0).tolist()
                confidence_scores = torch.max(predictions, dim=-1)[0].squeeze(0).tolist()
            
            # B-I-O 태깅으로 엔티티 추출
            current_entity = ""
            current_type = None
            current_start = -1
            current_confidence = 0.0
            
            for token_idx, (pred_id, confidence, (start, end)) in enumerate(zip(predicted_labels, confidence_scores, offset_mapping)):
                if start == 0 and end == 0:  # 특수 토큰 건너뛰기
                    continue
                
                pred_label = id2label.get(pred_id, 'O')
                
                # 더 높은 신뢰도만 사용 (노이즈 감소)
                if confidence < 0.7:
                    pred_label = 'O'
                
                if pred_label.startswith('B-'):
                    # 이전 엔티티 저장
                    if current_entity and current_type and current_confidence > 0.75:
                        clean_entity = clean_entity_text(current_entity)
                        if is_valid_entity(clean_entity, current_type):
                            entities.add((clean_entity, current_type))
                    
                    # 새 엔티티 시작
                    current_entity = sentence[start:end]
                    current_type = pred_label[2:]
                    current_start = start
                    current_confidence = confidence
                    
                elif pred_label.startswith('I-') and current_type == pred_label[2:]:
                    # 현재 엔티티 확장
                    if current_start != -1:
                        current_entity = sentence[current_start:end]
                        current_confidence = min(current_confidence, confidence)
                else:
                    # 엔티티 종료
                    if current_entity and current_type and current_confidence > 0.75:
                        clean_entity = clean_entity_text(current_entity)
                        if is_valid_entity(clean_entity, current_type):
                            entities.add((clean_entity, current_type))
                    
                    current_entity = ""
                    current_type = None
                    current_start = -1
                    current_confidence = 0.0
            
            # 마지막 엔티티 처리
            if current_entity and current_type and current_confidence > 0.75:
                clean_entity = clean_entity_text(current_entity)
                if is_valid_entity(clean_entity, current_type):
                    entities.add((clean_entity, current_type))
                    
        except Exception as e:
            logger.warning(f"문장 처리 오류: {e}")
            continue
    
    return entities

def extract_entities_by_regex(text: str) -> Set[Tuple[str, str]]:
    """정규표현식 기반 백업 엔티티 추출"""
    entities = set()
    
    # 이름 패턴 (한국어 이름)
    name_patterns = [
        r'[가-힣]{2,4}(?=\s*(?:씨|님|선생|교수|박사|의원|대표|이사|부장|과장|대리|주임|팀장))',
        r'성명:\s*([가-힣]{2,4})',
        r'이름:\s*([가-힣]{2,4})',
        r'계약자:\s*([가-힣]{2,4})',
        r'(?:갑|을):\s*([가-힣]{2,4})'
    ]
    
    for pattern in name_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            name = match.group(1) if match.groups() else match.group(0)
            name = name.replace(':', '').strip()
            if len(name) >= 2 and name.isalpha() and is_valid_entity(name, 'NAME'):
                entities.add((name, 'NAME'))
    
    # 전화번호 패턴
    phone_patterns = [
        r'(\d{2,3}-\d{3,4}-\d{4})',
        r'(\d{3}-\d{4}-\d{4})',
        r'전화번호:\s*([0-9-]{10,15})',
        r'연락처:\s*([0-9-]{10,15})',
        r'TEL:\s*([0-9-]{10,15})'
    ]
    
    for pattern in phone_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            phone = match.group(1) if match.groups() else match.group(0)
            phone = phone.replace('전화번호:', '').replace('연락처:', '').replace('TEL:', '').strip()
            # 전화번호 유효성 강화
            if re.match(r'^[0-9-]{10,15}$', phone) and is_valid_entity(phone, 'PHONE'):
                entities.add((phone, 'PHONE'))
    
    # 주소 패턴
    address_patterns = [
        r'주소:\s*([가-힣0-9\s-]+(?:시|구|군|동|로|길)[가-힣0-9\s-]*)',
        r'([가-힣]+(?:시|도)\s+[가-힣]+(?:구|군)\s+[가-힣0-9\s-]*(?:로|길|동)[\s0-9]*)',
        r'(서울시\s+[가-힣]+구[가-힣0-9\s-]*)',
        r'(부산시\s+[가-힣]+구[가-힣0-9\s-]*)'
    ]
    
    for pattern in address_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            address = match.group(1) if match.groups() else match.group(0)
            address = address.replace('주소:', '').strip()
            if len(address) > 5 and is_valid_entity(address, 'ADDRESS'):
                entities.add((address, 'ADDRESS'))
    
    # 이메일 패턴
    email_pattern = r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
    matches = re.finditer(email_pattern, text)
    for match in matches:
        email = match.group(1)
        entities.add((email, 'EMAIL'))
    
    # 회사명 패턴
    company_patterns = [
        r'([가-힣]+(?:주식회사|㈜|회사|기관|재단|협회|센터|연구소|연구원|대학교|청|처|부|원))',
        r'소속:\s*([가-힣0-9\s]+(?:주식회사|㈜|회사|기관|재단|협회|센터|연구소|연구원|대학교|청|처|부|원))',
        r'회사:\s*([가-힣0-9\s]+)',
        r'기관:\s*([가-힣0-9\s]+)'
    ]
    
    for pattern in company_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            company = match.group(1) if match.groups() else match.group(0)
            company = company.replace('소속:', '').replace('회사:', '').replace('기관:', '').strip()
            if len(company) > 2 and is_valid_entity(company, 'COMPANY'):
                entities.add((company, 'COMPANY'))
    
    # 날짜 패턴
    date_patterns = [
        r'(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)',
        r'(\d{4}\.\d{1,2}\.\d{1,2})',
        r'(\d{4}-\d{1,2}-\d{1,2})',
        r'(\d{1,2}/\d{1,2}/\d{4})',
        r'작성일:\s*([0-9년월일.\s-]+)',
        r'계약일:\s*([0-9년월일.\s-]+)'
    ]
    
    for pattern in date_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            date = match.group(1) if match.groups() else match.group(0)
            date = date.replace('작성일:', '').replace('계약일:', '').strip()
            if len(date) > 4 and is_valid_entity(date, 'DATE'):
                entities.add((date, 'DATE'))
    
    # 금액 패턴
    money_patterns = [
        r'(\d{1,3}(?:,\d{3})*원)',
        r'금\s*(\d{1,3}(?:,\d{3})*원)',
        r'계약금:\s*([0-9,원\s]+)',
        r'사업비:\s*([0-9,원\s]+)'
    ]
    
    for pattern in money_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            money = match.group(1) if match.groups() else match.group(0)
            money = money.replace('금', '').replace('계약금:', '').replace('사업비:', '').strip()
            if '원' in money and len(money) > 2:
                entities.add((money, 'MONEY'))
    
    return entities

def split_text_smartly(text: str, max_length: int = 512) -> List[str]:
    """텍스트를 스마트하게 분할"""
    if len(text) <= max_length:
        return [text]
    
    sentences = []
    text_sentences = text.replace('\n', '. ').split('.')
    
    current_chunk = ""
    for sentence in text_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk + sentence) <= max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                sentences.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        sentences.append(current_chunk.strip())
    
    return sentences

def clean_entity_text(entity: str) -> str:
    """엔티티 텍스트 정리"""
    entity = entity.strip()
    entity = re.sub(r'^[:\s,.-]+', '', entity)
    entity = re.sub(r'[:\s,.-]+$', '', entity)
    return entity

def group_entities_by_type(entities: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """
    엔티티를 타입별로 그룹화
    
    Args:
        entities: [(값, 타입), ...] 형태의 엔티티 리스트
    
    Returns:
        {타입: [값1, 값2, ...], ...} 형태의 딕셔너리
    """
    grouped = {}
    for entity, entity_type in entities:
        if entity_type not in grouped:
            grouped[entity_type] = []
        grouped[entity_type].append(entity)
    
    # 알파벳 순으로 정렬
    return dict(sorted(grouped.items()))

def is_valid_entity(entity: str, entity_type: Optional[str] = None) -> bool:
    """유효한 엔티티인지 확인 - 정확도 개선 버전"""
    
    # 1. 기본 길이 체크 (최소 2자)
    if len(entity) < 2:
        return False
    
    # 2. 최대 길이 제한 (여러 줄 방지)
    if '\n' in entity:
        line_count = entity.count('\n')
        # ADDRESS는 최대 1개 줄바꿈, 나머지는 2개까지 허용
        if entity_type == 'ADDRESS':
            if line_count > 1:  # ADDRESS는 1줄까지만
                return False
        elif line_count > 2:  # 다른 타입은 2줄까지
            return False
    if len(entity) > 50:  # 최대 50자
        return False
    
    # 3. 불필요한 문자들 제외
    invalid_chars = ['□', '☑', '○', '●']
    if any(char in entity for char in invalid_chars):
        return False
    
    # 4. 숫자만으로는 안됨
    if entity.isdigit():
        return False
    
    # 5. 숫자+단위 패턴 제외 (예: "100원", "2024년")
    if re.match(r'^\d+[년월일원]$', entity):
        return False
    
    # 6. 불완전한 단어 제외 (마침표, 쉼표로 끝남)
    if entity.endswith(('.', ',', '·', ')', '(', ':')):
        return False
    if entity.startswith(('.', ',', '·', ')', '(')):
        return False
    
    # 7. 조사/접미사만 있는 경우 제외
    josa_list = ['을', '를', '가', '는', '은', '의', '이', '에', '에서', '에게', '부터', '까지', '으로', '로', '과', '와', '및']
    if entity in josa_list:
        return False
    
    # 8. 타입별 특수 검증
    if entity_type:
        # NAME 타입 검증
        if entity_type == 'NAME':
            # 동사형 제외
            if entity in ['양도', '양수', '제공', '수령', '대표', '담당', '관리', '저작', '회사', '기관']:
                return False
            # 역할 단어 제외 (끝이 자/인/처)
            if entity.endswith(('자', '인', '처')) and len(entity) <= 4:
                # 실제 이름 예외
                if entity not in ['김자', '이자', '박인', '한자']:
                    return False
            # 역할 단어 완전 매칭
            role_words = ['양도자', '양도인', '양수인', '양수자', '제공자', '이용자', '수령자', '수령인']
            if entity in role_words:
                return False
            # 중간에 마침표/쉼표 있는 경우 제외 (예: "양도. 자")
            if '.' in entity or ',' in entity:
                return False
        
        # COMPANY 타입 검증
        elif entity_type == 'COMPANY':
            # 일반 명사 제외
            general_nouns = ['연락처', '주소', '성명', '전화', '휴대', '담당', '회사', '기관', 
                           '공공기관', '방송사', '관할법원', '보전처', '관할', '보전', '확인',
                           '수행자로부', '저작인접권자로부', '저작권자로부', '권리자로부']
            if entity in general_nouns:
                return False
            # 부분 문자열 체크 (일반 명사 포함 제외)
            if '연락처' in entity or '주소' in entity:
                return False
            # "~로부"로 끝나는 경우 제외
            if entity.endswith('로부'):
                return False
            # "~법원" 포함 제외 (관할법원, 지방법원 등)
            if '법원' in entity and len(entity) <= 6:
                return False
            # "~처" 로 끝나는 관청 명사 제외
            if entity.endswith('처') and len(entity) <= 3:
                return False
            # 너무 짧은 회사명 제외 (4자 미만, 특수 예외 제외)
            if len(entity) < 4 and entity not in ['KBS', 'MBC', 'SBS', 'EBS']:
                return False
        
        # PHONE 타입 검증
        elif entity_type == 'PHONE':
            # 날짜 패턴 제외 (예: "2020. 6. 20")
            if re.match(r'\d{4}\.\s*\d{1,2}\.\s*\d{1,2}', entity):
                return False
            # 짧은 숫자 제외 (7자리 미만, 실제 전화번호는 보통 7-11자리)
            digits_only = ''.join(c for c in entity if c.isdigit())
            if len(digits_only) < 7:
                return False
            # 전화번호 뒤에 날짜가 붙은 경우 제외
            if re.search(r'\d{2,4}\.\s*\d{4}', entity):
                return False
        
        # POSITION 타입 검증
        elif entity_type == 'POSITION':
            invalid_positions = ['저작', '회사', '사업', '스튜디오', '대표', '담당', '상대방', '관계', '저작물']
            if entity in invalid_positions:
                return False
        
        # DESCRIPTION 타입 검증
        elif entity_type == 'DESCRIPTION':
            # 너무 짧은 설명 제외
            if len(entity) < 5:
                return False
            # 일반적인 단어 제외
            generic_words = ['공공', '저작물', '저작권', '사회통념상', '국민들이', '영리적으로도 이용할 수',
                           '공공저작물', '저작물을', '저작물 제작을', '저작물의']
            if entity in generic_words:
                return False
            # 포함 체크
            if '공공저작물' in entity and '목적으로' in entity:  # "공공저작물 제작을 목적으로" 같은 것
                return False
            # "~을/를/가/는"으로 끝나는 불완전한 설명 제외
            if entity.endswith(('을', '를', '가', '는', '이', '의', '으로')):
                return False
            # 중간에 마침표가 있는 불완전한 문장 제외 (예: "공공저작물을 자유롭. 게")
            if '. ' in entity or ' .' in entity:
                return False
        
        # DATE 타입 검증
        elif entity_type == 'DATE':
            # 파일 확장자 제외
            if entity.lower() in ['png', 'jpg', 'pdf', 'txt', 'doc', 'xlsx', 'jpeg', 'gif']:
                return False
        
        # CONSENT_TYPE, CONTRACT_TYPE 타입 검증
        elif entity_type in ['CONSENT_TYPE', 'CONTRACT_TYPE']:
            # 너무 짧은 것 제외 (최소 3자)
            if len(entity) < 3:
                return False
            # 불완전한 단어 제외 (예: "확인 및")
            if entity.endswith(' 및') or entity.endswith(' 와') or entity.endswith(' 또는'):
                return False
        
        # ADDRESS 타입 검증
        elif entity_type == 'ADDRESS':
            # 줄바꿈이 2개 이상인 경우 제외 (너무 긴 주소)
            if entity.count('\n') > 1:
                return False
    
    # 9. 공백만 있는 경우
    if entity.strip() == '':
        return False
    
    return True

def download_pretrained_model(model_name: str, model_path: Path, verbose: bool = True) -> bool:
    """
    Hugging Face에서 사전 훈련된 모델 다운로드
    
    Args:
        model_name: Hugging Face 모델 이름 (예: klue/roberta-large, xlm-roberta-large)
        model_path: 저장할 로컬 경로
        verbose: 로그 출력 여부
    
    Returns:
        bool: 다운로드 성공 여부
    """
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔽 Hugging Face에서 모델 다운로드 중...")
            print(f"   모델: {model_name}")
            print(f"   저장 경로: {model_path}")
            print(f"{'='*60}")
        
        # 모델명 정규화 (파일명 형식을 Hugging Face 형식으로 변환)
        # klue-roberta-large -> klue/roberta-large
        # xlm-roberta-large -> xlm-roberta-large (그대로)
        hf_model_name = model_name
        if model_name.startswith('klue-'):
            hf_model_name = model_name.replace('klue-', 'klue/', 1)
        elif model_name.startswith('bert-'):
            hf_model_name = model_name  # bert-base-multilingual-cased 등은 그대로
        
        print(f"📥 Hugging Face 모델명: {hf_model_name}")
        
        # Hugging Face에서 토크나이저와 모델 다운로드
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        print(f"📥 토크나이저 다운로드 중... ({hf_model_name})")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        
        print(f"📥 모델 다운로드 중... ({hf_model_name})")
        # NER용 모델 로드 (기본 레이블로 초기화)
        model = AutoModelForTokenClassification.from_pretrained(
            hf_model_name,
            num_labels=len(ENTITY_TYPES) * 2 + 1,  # B-I-O 태깅
            ignore_mismatched_sizes=True
        )
        
        # 로컬에 저장
        model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 모델 저장 중... ({model_path})")
        tokenizer.save_pretrained(str(model_path))
        model.save_pretrained(str(model_path))
        
        # 라벨 매핑 저장
        labels = ["O"]
        for entity in ENTITY_TYPES:
            labels.extend([f"B-{entity}", f"I-{entity}"])
        
        id2label = {i: label for i, label in enumerate(labels)}
        label2id = {label: i for i, label in enumerate(labels)}
        
        label_info = {
            'id2label': {str(k): v for k, v in id2label.items()},
            'label2id': label2id,
            'entity_types': ENTITY_TYPES
        }
        
        label_map_file = model_path / "label_map.json"
        with open(label_map_file, 'w', encoding='utf-8') as f:
            json.dump(label_info, f, ensure_ascii=False, indent=2)
        
        if verbose:
            print(f"✅ 모델 다운로드 완료!")
            print(f"   - config.json: {(model_path / 'config.json').exists()}")
            print(f"   - model.safetensors: {(model_path / 'model.safetensors').exists()}")
            print(f"   - tokenizer.json: {(model_path / 'tokenizer.json').exists()}")
            print(f"   - label_map.json: {label_map_file.exists()}")
            print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_model_if_needed(model_name: str, model_path: Path, verbose: bool = True, auto_train: bool = False, auto_download: bool = True) -> bool:
    """
    필요시 모델 훈련 또는 다운로드
    
    Args:
        model_name: 모델 이름 (Hugging Face 모델명)
        model_path: 모델 경로
        verbose: 로그 출력 여부
        auto_train: 자동 훈련 활성화 여부 (기본값: False)
        auto_download: Hugging Face에서 자동 다운로드 여부 (기본값: True)
    
    Returns:
        bool: 모델 사용 가능 여부
    """
    if model_path.exists() and (model_path / "config.json").exists():
        if verbose:
            print("✓ 기존 모델을 사용합니다.")
        return True
    
    # 1단계: Hugging Face에서 다운로드 시도 (auto_download=True인 경우)
    if auto_download:
        print(f"⚠️  로컬에 모델이 없습니다: {model_path}")
        print(f"🔍 Hugging Face에서 '{model_name}' 모델을 검색합니다...")
        
        if download_pretrained_model(model_name, model_path, verbose):
            print(f"✅ 모델 다운로드 완료! Fine-tuning 없이 사용 가능합니다.")
            return True
        else:
            print(f"⚠️  Hugging Face에서 모델을 다운로드할 수 없습니다.")
    
    # 2단계: 자동 훈련 시도 (auto_train=True인 경우)
    if not auto_train:
        print(f"⚠️  자동 훈련이 비활성화되어 있습니다.")
        print(f"⚠️  다음 중 하나를 선택하세요:")
        print(f"   1) auto_download=True로 설정하여 Hugging Face에서 다운로드")
        print(f"   2) auto_train=True로 설정하여 자동 훈련")
        print(f"   3) 수동 훈련: python api/module/ner/ner_train.py")
        return False
    
    # 자동 훈련 실행
    print("모델이 없습니다. 훈련을 시작합니다...")
    
    try:
        # ner_train.py 실행
        current_dir = Path(__file__).parent
        train_script = current_dir / "ner_train.py"
        
        if not train_script.exists():
            print(f"훈련 스크립트를 찾을 수 없습니다: {train_script}")
            return False
        
        print(f"훈련 스크립트 실행: {train_script}")
        
        # subprocess로 훈련 실행
        process = subprocess.Popen([
            sys.executable, str(train_script)
        ], cwd=str(current_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
        text=True, universal_newlines=True)
        
        # 실시간 출력 표시
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return_code = process.poll()
        
        if return_code == 0:
            print("모델 훈련이 완료되었습니다!")
            return True
        else:
            print(f"모델 훈련이 실패했습니다. (코드: {return_code})")
            return False
            
    except Exception as e:
        print(f"모델 훈련 중 오류: {e}")
        return False

def extract_entities_from_text(text: str, model_name: Optional[str] = None, debug: bool = False, train: bool = False) -> List[Tuple[str, str]]:
    """
    텍스트에서 엔티티 추출 (통합 메인 함수)
    
    Args:
        text: 입력 텍스트
        model_name: 모델 이름
        debug: 디버그 로그 출력
        train: True이면 훈련 수행, False이면 훈련 없이 예측 (기본값: False)
    """
    if debug:
        print(f"엔티티 추출 시작 (텍스트 길이: {len(text)}자)")
    
    # 모델 이름이 지정되지 않으면 기본 모델 사용
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
        if debug:
            print(f"모델 이름이 지정되지 않아 기본 모델 사용: {model_name}")
    
    if debug:
        print(f"사용 모델: {model_name}")
    
    all_entities = set()
    
    # 모델 경로 확인
    model_path = get_model_path(model_name)
    if debug:
        print(f"모델 경로: {model_path}")
    
    # 모델 존재 여부 확인 (config.json 확인)
    model_exists = model_path.exists() and (model_path / "config.json").exists()
    
    # 모델이 없으면 무조건 다운로드
    if not model_exists:
        if debug:
            print("=" * 60)
            print(f"모델 없음: {model_name} 다운로드를 시작합니다...")
            print("=" * 60)
        
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            
            # 모델 다운로드
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            # 모델 저장
            model_path.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            
            if debug:
                print(f"✓ 모델 다운로드 완료: {model_path}")
            
            model_exists = True
        except Exception as e:
            if debug:
                print(f"⚠️ 모델 다운로드 실패: {e}")
                print(f"정규표현식만 사용합니다.")
    
    # train=True이면 훈련 수행
    if train:
        if debug:
            print("=" * 60)
            print("train=True: 모델 훈련을 시작합니다...")
            print("=" * 60)
        
        # ner_train 함수 호출
        result = ner_train(
            model_name=model_name,
            epochs=2,
            force_retrain=True,  # train=True이면 무조건 훈련
            debug=debug
        )
        
        if not result.get('success', False):
            if debug:
                print(f"⚠️ 모델 훈련 실패: {result.get('error', 'Unknown error')}")
        else:
            if debug:
                print(f"✓ 모델 훈련 완료!")
    elif model_exists:
        if debug:
            print(f"✓ 기존 모델 사용 (훈련 없이 예측): {model_path}")
    
    try:
        # 1. B-I-O 모델 기반 예측
        model_exists = model_path.exists() and (model_path / "config.json").exists()
        
        if model_exists:
            if debug:
                print(f"파인튜닝 모델 파일 확인됨: {model_path}")
                print("B-I-O 태깅 기반 예측 시작...")
            
            tokenizer, model, id2label, device = load_model_and_tokenizer(model_path, verbose=debug)
            if debug:
                print(f"모델 로드 완료 - 라벨 수: {len(id2label)}개")
            
            bio_entities = extract_entities_by_bio_tagging(text, tokenizer, model, id2label, device)
            all_entities.update(bio_entities)
            
            if debug:
                print(f"B-I-O 예측 결과: {len(bio_entities)}개 엔티티")
        
        else:
            if debug:
                print(f"모델 파일 없음: {model_path}")
                print(f"   - 디렉토리 존재: {model_path.exists()}")
                if model_path.exists():
                    print(f"   - config.json 존재: {(model_path / 'config.json').exists()}")
        
    except Exception as e:
        if debug:
            print(f"모델 예측 오류: {e}")
            import traceback
            traceback.print_exc()
    
    # 2. 정규표현식 백업 예측
    if debug:
        print("정규표현식 백업 예측 시작...")
    
    regex_entities = extract_entities_by_regex(text)
    all_entities.update(regex_entities)
    
    if debug:
        print(f"정규표현식 예측 결과: {len(regex_entities)}개 엔티티")
    
    # 3. 결과 통합 및 중복 제거
    final_entities = []
    seen_entities = set()
    
    for entity, label in all_entities:
        entity_lower = entity.lower().strip()
        if entity_lower not in seen_entities:
            seen_entities.add(entity_lower)
            final_entities.append((entity, label))
    
    # 엔티티 타입별로 정렬
    final_entities.sort(key=lambda x: (x[1], x[0]))
    
    if debug:
        print(f"최종 예측 결과: {len(final_entities)}개 엔티티")
        for entity, label in final_entities[:10]:  # 처음 10개만 출력
            print(f"  - {entity} ({label})")
    
    return final_entities

def ner_predict(
    input_path: str,
    output_path: str,
    model_name: Optional[str] = None,
    confidence_threshold: float = 0.85,
    output_format: str = "both",
    save_statistics: bool = True,
    entity_filter: Optional[List[str]] = None,
    train: bool = False,
    debug: bool = False
) -> Dict[str, Any]:
    """
    디렉토리 또는 파일에 대한 NER 예측 수행
    
    Args:
        input_path: 입력 파일/디렉토리 경로
        output_path: 출력 디렉토리 경로
        model_name: 사용할 모델 이름 (기본값: model_config.json의 default_model)
        confidence_threshold: 신뢰도 임계값
        output_format: 출력 형식
        save_statistics: 통계 저장 여부
        entity_filter: 추출할 엔티티 타입 필터
        train: True이면 무조건 모델 훈련 후 예측 (기본값: False)
        debug: True이면 상세 로그 출력 (기본값: False)
    
    Returns:
        Dict[str, Any]: 예측 결과 정보
    """
    start_time = time.time()
    
    # 모델 이름이 지정되지 않으면 기본 모델 사용
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
        if debug:
            print(f"⚠️  모델 이름이 지정되지 않아 기본 모델 사용: {model_name}")
    
    try:
        if debug:
            print("=" * 60)
            print("NER 예측 시스템 시작")  
            print("=" * 60)
        
        # 1. 시스템 요구사항 확인
        device = check_system_requirements(verbose=debug)
        
        # 2. 입력 경로 확인
        input_path_obj = Path(input_path)
        output_path_obj = Path(output_path)
        
        if not input_path_obj.exists():
            return {
                "success": False,
                "error": f"입력 경로가 존재하지 않습니다: {input_path_obj}"
            }
        
        # 3. 출력 디렉토리 생성 (모델별로 분리)
        # ner/{model_name} 디렉토리 구조로 저장
        # 모델 이름에서 슬래시를 하이픈으로 변경 (파일시스템 호환)
        model_dir_name = model_name.replace('/', '-')
        ner_dir = output_path_obj / "ner" / model_dir_name
        ner_dir.mkdir(parents=True, exist_ok=True)
        
        if debug:
            print(f"출력 디렉토리: {ner_dir}")
            print(f"모델: {model_name}")
        
        # 4. 처리할 파일 목록 생성
        files_to_process = []
        
        if input_path_obj.is_file():
            if input_path_obj.suffix.lower() in ['.txt', '.md']:
                files_to_process = [input_path_obj]
        else:
            # 디렉토리인 경우 텍스트 파일들 찾기
            for ext in ['*.txt', '*.md']:
                files_to_process.extend(input_path_obj.glob(f"**/{ext}"))
        
        if not files_to_process:
            return {
                "success": False,
                "error": "처리할 텍스트 파일이 없습니다."
            }
        
        if debug:
            print(f"처리할 파일 수: {len(files_to_process)}")
        
        # 5. 모델 다운로드 및 훈련 (필요한 경우 한 번만 수행)
        model_path = get_model_path(model_name)
        model_exists = model_path.exists() and (model_path / "config.json").exists()
        
        # 모델이 없으면 무조건 다운로드
        if not model_exists:
            if debug:
                print("=" * 60)
                print(f"모델 없음: {model_name} 다운로드를 시작합니다...")
                print("=" * 60)
            
            try:
                from transformers import AutoTokenizer, AutoModelForTokenClassification
                
                # 모델 다운로드
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForTokenClassification.from_pretrained(model_name)
                
                # 모델 저장
                model_path.mkdir(parents=True, exist_ok=True)
                tokenizer.save_pretrained(model_path)
                model.save_pretrained(model_path)
                
                if debug:
                    print(f"✓ 모델 다운로드 완료: {model_path}")
                
                model_exists = True
            except Exception as e:
                print(f"⚠️ 모델 다운로드 실패: {e}")
                return {
                    "success": False,
                    "error": f"모델 다운로드 실패: {e}"
                }
        
        # train=True이면 훈련 수행
        if train:
            if debug:
                print("=" * 60)
                print(f"train=True: {model_name} 훈련을 시작합니다...")
                print("=" * 60)
            
            result = ner_train(
                model_name=model_name,
                epochs=2,
                force_retrain=True,  # train=True이면 무조건 훈련
                debug=debug
            )
            
            if not result.get('success', False):
                print(f"⚠️ 모델 훈련 실패: {result.get('error', 'Unknown error')}")
                print(f"훈련 없이 베이스 모델로 예측을 계속합니다.")
            else:
                if debug:
                    print(f"✓ 모델 훈련 완료!")
        elif model_exists:
            if debug:
                print(f"✓ 기존 모델 사용 (훈련 없이 예측): {model_path}")
        
        # 6. 엔티티 추출 시작
        if debug:
            print("엔티티 추출 시작...")
        
        all_entities = []
        processed_files = 0
        
        # 프로그레스 바는 항상 표시 (disable=False)
        import sys
        for file_path in tqdm(files_to_process, desc="파일 처리 중", disable=False, file=sys.stdout, ncols=80):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content.strip()) < 10:
                    continue
                
                # 엔티티 추출 (train=False로 고정 - 이미 위에서 훈련 완료 또는 스킵)
                entities = extract_entities_from_text(content, model_name=model_name, debug=False, train=False)
                
                # 결과 저장 - 입력 경로 구조 유지 (pdf_to_image와 동일한 패턴)
                if entities:
                    # 엔티티 리스트를 타입별로 그룹화
                    entities_grouped = group_entities_by_type(entities)
                    
                    # 결과 구조: 타입별로 그룹화된 형태
                    file_result = {
                        'file': str(file_path),
                        'entities': entities_grouped,
                        'entity_count': len(entities),
                        'entity_types': list(entities_grouped.keys())
                    }
                    all_entities.extend(entities)
                    
                    # 입력 경로 기준으로 상대 경로 계산하여 출력 구조 생성
                    file_path_obj = Path(file_path)
                    
                    if input_path_obj.is_file():
                        # 단일 파일인 경우 - 파일명으로 디렉토리 생성하지 않고 직접 저장
                        result_file = ner_dir / f"{file_path_obj.stem}_entities.json"
                    else:
                        # 디렉토리인 경우 - input_path를 기준으로 한 상대 경로 구조 유지
                        try:
                            relative_path = file_path_obj.relative_to(input_path_obj)
                            # 상대 경로 구조를 유지하면서 _entities.json 추가
                            if relative_path.parent != Path('.'):
                                result_dir = ner_dir / relative_path.parent
                                result_dir.mkdir(parents=True, exist_ok=True)
                                result_file = result_dir / f"{file_path_obj.stem}_entities.json"
                            else:
                                result_file = ner_dir / f"{file_path_obj.stem}_entities.json"
                        except ValueError:
                            # 상대 경로 계산 실패시 폴백
                            result_file = ner_dir / f"{file_path_obj.stem}_entities.json"
                    
                    # JSON 파일 저장
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(file_result, f, ensure_ascii=False, indent=2)
                
                processed_files += 1
                
            except Exception as e:
                logger.warning(f"파일 처리 오류 {file_path}: {e}")
                continue
        
        # 6. 전체 결과 요약
        entity_stats = defaultdict(int)
        for entity, entity_type in all_entities:
            entity_stats[entity_type] += 1
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary = {
            'total_files_processed': processed_files,
            'total_entities_found': len(all_entities),
            'unique_entities': len(set(all_entities)),
            'entity_types_count': dict(entity_stats),
            'processing_time': time.time() - start_time,
            'timestamp': timestamp
        }
        
        # 7. 요약 파일 저장
        summary_file = ner_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\nNER 예측 완료!")
        print(f"처리된 파일: {processed_files}/{len(files_to_process)}")
        print(f"추출된 엔티티: {len(all_entities)}개")
        print(f"결과 저장: {ner_dir}")
        print(f"소요 시간: {summary['processing_time']:.1f}초")
        
        return {
            "success": True,
            "processed_files": processed_files,
            "total_entities": len(all_entities),
            "extracted_entities": all_entities,
            "statistics": summary,
            "output_directory": str(ner_dir),
            "processing_time": summary['processing_time'],
            "summary_file": str(summary_file)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"예측 중 오류 발생: {str(e)}"
        }

def ensure_training_data_exists(model_name: str, num_samples: int = 7500, force_regenerate: bool = False) -> bool:
    """
    훈련/검증 데이터가 존재하는지 확인하고 없거나 force_regenerate=True면 새로 생성
    
    Args:
        model_name: 모델 이름
        num_samples: 생성할 총 샘플 수 (기본값: 7500, 범위: 5000~10000)
        force_regenerate: True면 기존 데이터를 삭제하고 재생성 (overfitting 방지)
    
    Returns:
        bool: 데이터 준비 성공 여부
    
    Note:
        - 생성된 데이터의 80%는 train.txt (훈련용)
        - 나머지 20%는 validation.txt (검증/평가용)
        - test.txt는 validation.txt와 동일하게 생성 (호환성)
    """
    try:
        current_dir = Path(__file__).parent
        model_name_safe = model_name.replace('/', '-')
        training_dir = current_dir / "training" / model_name_safe
        
        train_file = training_dir / "train.txt"
        val_file = training_dir / "validation.txt"
        test_file = training_dir / "test.txt"
        
        # force_regenerate=True 또는 파일이 없으면 생성
        if force_regenerate or not (train_file.exists() and val_file.exists()):
            if force_regenerate:
                print(f"\n{'='*60}")
                print(f"🔄 훈련 데이터 재생성 (Overfitting 방지)")
                print(f"{'='*60}")
            else:
                print(f"\n{'='*60}")
                print(f"📦 훈련 데이터 생성")
                print(f"{'='*60}")
            
            print(f"총 샘플 수: {num_samples:,}개")
            print(f"훈련 데이터: {int(num_samples * 0.8):,}개 (80%)")
            print(f"검증 데이터: {int(num_samples * 0.2):,}개 (20%)")
            print(f"{'='*60}")
            
            # 기존 파일 삭제
            if training_dir.exists():
                import shutil
                shutil.rmtree(training_dir)
            
            # 훈련 데이터 디렉토리 생성
            training_dir.mkdir(parents=True, exist_ok=True)
            
            # ner_train.py의 데이터 생성 함수 import
            try:
                from .ner_train import generate_rich_training_data
                
                # 훈련 데이터 생성
                result = generate_rich_training_data(training_dir, num_samples=num_samples)
                
                if result and train_file.exists() and val_file.exists():
                    # test.txt는 validation.txt와 동일하게 생성 (호환성)
                    if val_file.exists():
                        import shutil
                        shutil.copy(val_file, test_file)
                    
                    print(f"✓ 훈련 데이터 생성 완료!")
                    print(f"  - Train: {train_file} ({train_file.stat().st_size // 1024}KB)")
                    print(f"  - Validation: {val_file} ({val_file.stat().st_size // 1024}KB)")
                    print(f"  - Test: {test_file} (검증 데이터 복사본)")
                    return True
                else:
                    print(f"⚠️  데이터 생성 실패")
                    return False
                    
            except ImportError as e:
                print(f"⚠️  ner_train.py를 import할 수 없습니다: {e}")
                return False
        else:
            print(f"✓ 기존 훈련 데이터 사용: {training_dir}")
            return True
            
    except Exception as e:
        print(f"⚠️  훈련 데이터 생성 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def ner_train(
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: Optional[str] = None,
    enable_fp16: bool = True,
    max_length: int = 128,
    warmup_steps: int = 100,
    save_steps: int = 200,
    eval_steps: int = 100,
    force_retrain: bool = False,
    callback_url: Optional[str] = None,
    debug: bool = False,
    num_train_samples: int = 7500
) -> Dict[str, Any]:
    """
    NER 모델 훈련 API - 매번 새로운 데이터로 훈련 (overfitting 방지)
    
    Args:
        epochs: 훈련 에포크 수 (기본값: 3)
        batch_size: 배치 크기 (기본값: 8)
        learning_rate: 학습률 (기본값: 3e-5)
        model_name: 기본 모델명 (기본값: klue-roberta-large)
        output_dir: 출력 디렉토리 (None이면 자동 생성)
        enable_fp16: Mixed precision training 사용 여부
        max_length: 최대 토큰 길이
        warmup_steps: Warmup 스텝 수
        save_steps: 모델 저장 간격
        eval_steps: 평가 간격
        force_retrain: 기존 모델이 있어도 재훈련 여부
        callback_url: 훈련 상태 콜백 URL (옵션)
        debug: True이면 상세 로그 출력 (기본값: False)
        num_train_samples: 생성할 총 샘플 수 (기본값: 7500, 80%는 훈련, 20%는 검증)
    
    Returns:
        Dict[str, Any]: 훈련 결과 정보
    
    Note:
        - 훈련 시마다 새로운 데이터 생성 (force_regenerate=True)
        - 생성된 데이터의 80%는 훈련용, 20%는 검증용
        - 샘플 수 범위: 5,000 ~ 10,000개 권장
    """
    start_time = time.time()
    
    try:
        if debug:
            print("=" * 60)
            print("NER 모델 훈련 API 시작")
            print("=" * 60)
        
        # 1. 시스템 요구사항 확인
        device = check_system_requirements(verbose=debug)
        
        # 2. 훈련 데이터 생성 (매번 새로 생성 - overfitting 방지)
        if not ensure_training_data_exists(model_name, num_samples=num_train_samples, force_regenerate=True):
            return {
                "success": False,
                "error": "훈련 데이터 생성 실패",
                "training_time": time.time() - start_time
            }
        
        # 3. 모델 경로 설정
        if output_dir:
            model_path = Path(output_dir)
        else:
            model_path = get_model_path(model_name)
        
        # 기존 모델이 있어도 force_retrain=True이면 재훈련
        if not force_retrain and model_path.exists() and (model_path / "config.json").exists():
            if debug:
                print(f"⚠️  기존 모델 존재하지만 force_retrain=False라서 훈련 스킵")
            return {
                "success": True,
                "message": "기존 훈련된 모델을 사용합니다.",
                "model_path": str(model_path),
                "training_time": 0,
                "skipped": True
            }
        
        if debug:
            print(f"모델 저장 경로: {model_path}")
            print(f"훈련 설정:")
            print(f"  - Epochs: {epochs}")
            print(f"  - Batch Size: {batch_size}")
            print(f"  - Learning Rate: {learning_rate}")
            print(f"  - Max Length: {max_length}")
            print(f"  - FP16: {enable_fp16}")
            print(f"  - Training Samples: {int(num_train_samples * 0.8):,} (80%)")
            print(f"  - Validation Samples: {int(num_train_samples * 0.2):,} (20%)")
        
        # 4. ner_train.py 스크립트 실행
        current_dir = Path(__file__).parent
        train_script = current_dir / "ner_train.py"
        
        if not train_script.exists():
            return {
                "success": False,
                "error": f"훈련 스크립트를 찾을 수 없습니다: {train_script}"
            }
        
        # 5. 훈련 명령어 구성
        cmd_args = [
            sys.executable, str(train_script),
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--learning_rate", str(learning_rate),
            "--model_name", model_name,
            "--output_dir", str(model_path),
            "--max_length", str(max_length),
            "--warmup_steps", str(warmup_steps),
            "--save_steps", str(save_steps),
            "--eval_steps", str(eval_steps)
        ]
        
        if enable_fp16:
            cmd_args.append("--fp16")
        
        if debug:
            print(f"훈련 시작...")
        
        # 6. 훈련 실행
        process = subprocess.Popen(
            cmd_args,
            cwd=str(current_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True
        )
        
        # 실시간 출력 및 진행 상황 추적
        training_logs = []
        current_epoch = 0
        current_step = 0
        total_steps = 0
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
                
            if output:
                line = output.strip()
                print(line)
                training_logs.append(line)
                
                # 진행 상황 파싱
                if "Epoch" in line and "/" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "Epoch" in part and i + 1 < len(parts):
                                epoch_info = parts[i + 1]
                                if "/" in epoch_info:
                                    current_epoch = int(epoch_info.split("/")[0])
                    except:
                        pass
                
                if "Step" in line and "/" in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "Step" in part and i + 1 < len(parts):
                                step_info = parts[i + 1]
                                if "/" in step_info:
                                    current_step, total_steps = map(int, step_info.split("/"))
                    except:
                        pass
                
                # 콜백 URL이 있으면 진행 상황 전송 (실제 구현시)
                if callback_url and current_step > 0:
                    progress = {
                        "epoch": current_epoch,
                        "step": current_step,
                        "total_steps": total_steps,
                        "progress_percent": (current_step / max(total_steps, 1)) * 100
                    }
                    # 여기서 callback_url로 POST 요청 보낼 수 있음
        
        return_code = process.poll()
        training_time = time.time() - start_time
        
        # 7. 결과 처리
        if return_code == 0:
            # 훈련된 모델 검증
            if model_path.exists() and (model_path / "config.json").exists():
                # 훈련 정보를 JSON 파일로 저장
                training_info = {
                    "training_time": training_time,
                    "training_time_minutes": training_time / 60,
                    "final_epoch": current_epoch,
                    "total_steps": total_steps,
                    "completed_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "config": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "model_name": model_name,
                        "max_length": max_length,
                        "num_train_samples": num_train_samples
                    }
                }
                
                training_info_file = model_path / "training_info.json"
                with open(training_info_file, 'w', encoding='utf-8') as f:
                    json.dump(training_info, f, ensure_ascii=False, indent=2)
                
                if debug:
                    print(f"✓ 훈련 정보 저장: {training_info_file}")
                
                return {
                    "success": True,
                    "message": "NER 모델 훈련이 성공적으로 완료되었습니다!",
                    "model_path": str(model_path),
                    "training_time": training_time,
                    "final_epoch": current_epoch,
                    "total_steps": total_steps,
                    "training_logs": training_logs[-50:],  # 마지막 50줄만 반환
                    "config": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "model_name": model_name,
                        "max_length": max_length
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "훈련은 완료되었지만 모델 파일을 찾을 수 없습니다.",
                    "training_time": training_time,
                    "training_logs": training_logs[-20:]
                }
        else:
            return {
                "success": False,
                "error": f"모델 훈련이 실패했습니다. (Exit code: {return_code})",
                "training_time": training_time,
                "training_logs": training_logs[-20:]
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"훈련 중 예외 발생: {str(e)}",
            "training_time": time.time() - start_time
        }

def get_training_status(model_path: Optional[str] = None) -> Dict[str, Any]:
    """훈련 상태 확인"""
    if model_path:
        check_path = Path(model_path)
    else:
        check_path = get_model_path()
    
    status = {
        "model_exists": check_path.exists(),
        "model_path": str(check_path),
        "files": []
    }
    
    if check_path.exists():
        # 모델 파일들 확인
        important_files = ["config.json", "model.safetensors", "tokenizer.json", "label_map.json"]
        for file_name in important_files:
            file_path = check_path / file_name
            status["files"].append({
                "name": file_name,
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
            })
        
        # 체크포인트 확인
        checkpoints = list(check_path.glob("checkpoint-*"))
        status["checkpoints"] = len(checkpoints)
        status["latest_checkpoint"] = str(max(checkpoints, key=lambda p: p.stat().st_mtime)) if checkpoints else None
    
    return status

def ner_evaluate(
    output_path: str,
    model_name: Optional[str] = None,
    test_data_path: Optional[str] = None,
    verbose: bool = True,
    debug: bool = False,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    NER 모델 성능 평가 (F1 Score, Precision, Recall)
    훈련 데이터의 20% 검증 데이터(validation.txt)를 사용하여 평가
    
    Args:
        output_path: 평가 결과 저장 기본 경로 (예: "data/out")
                    {output_path}/debug/ 디렉토리에 {model_name}_evaluation.txt로 저장
        model_name: 평가할 모델 이름 (기본값: None이면 ner_predict와 동일한 기본 모델 사용)
        test_data_path: 테스트 데이터 경로 (BIO 포맷 .txt 파일)
                       None이면 자동으로 training/{model_name}/validation.txt 사용
        verbose: 상세 출력 여부 (기본값: True, debug 파라미터 통합)
        debug: 디버그 로그 출력 여부 (기본값: False, verbose보다 우선)
        max_samples: 평가할 최대 문장 수 (None이면 전체, 빠른 테스트용)
    
    Returns:
        Dict[str, Any]: 평가 결과 (F1, Precision, Recall, 엔티티별 점수)
    
    Usage:
        # ner_test.py 스타일 사용법
        ner_evaluate("data/out")  # 기본 모델, validation.txt 사용
        ner_evaluate("data/out", "FacebookAI/xlm-roberta-large")
        ner_evaluate("data/out", "google-bert/bert-base-multilingual-cased")
    
    Note:
        - validation.txt는 전체 생성 데이터의 20%로, 훈련에 사용되지 않은 홀드아웃 데이터입니다.
        - 매번 훈련할 때마다 새로운 데이터가 생성되므로, 평가 결과도 달라집니다.
    """
    # debug=True이면 verbose도 True로 설정
    if debug:
        verbose = True
    
    try:
        from sklearn.metrics import precision_recall_fscore_support, classification_report
    except ImportError:
        print("⚠️  scikit-learn이 설치되지 않았습니다. 설치 중...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
            from sklearn.metrics import precision_recall_fscore_support, classification_report
        except Exception as e:
            return {
                "success": False,
                "error": f"scikit-learn 설치 실패: {str(e)}"
            }
    
    overall_start_time = datetime.now()
    
    # 모델 이름 설정 (None이면 기본 모델 사용 - ner_predict와 동일)
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
        if verbose:
            print(f"⚠️  모델 이름이 지정되지 않아 기본 모델 사용: {model_name}")
    
    # 모델명 정규화 (klue/roberta-large → klue-roberta-large)
    model_name_safe = model_name.replace('/', '-')
    
    # 경로 설정
    current_dir = Path(__file__).parent
    api_dir = current_dir.parent.parent
    
    # 모델 경로에서 훈련 정보 로드
    model_path = get_model_path(model_name)
    training_info_file = model_path / "training_info.json"
    training_info = None
    
    if training_info_file.exists():
        try:
            with open(training_info_file, 'r', encoding='utf-8') as f:
                training_info = json.load(f)
            if verbose:
                print(f"✓ 훈련 정보 로드: 훈련 시간 {training_info.get('training_time', 0):.1f}초 ({training_info.get('training_time_minutes', 0):.1f}분)")
        except Exception as e:
            if debug:
                print(f"⚠️  훈련 정보 로드 실패: {e}")
    
    # 훈련 데이터가 없으면 자동 생성
    if not ensure_training_data_exists(model_name, num_samples=3000, force_regenerate=False):
        return {
            "success": False,
            "error": "훈련 데이터 생성 실패",
            "evaluation_time": (datetime.now() - overall_start_time).total_seconds()
        }
    
    # 검증 데이터 경로 자동 설정 (항상 validation.txt 사용)
    if test_data_path is None:
        training_dir = current_dir / "training" / model_name_safe
        test_data_path = str(training_dir / "validation.txt")
        eval_type = "Validation (훈련 데이터의 20%)"
    else:
        eval_type = "Custom"
    
    if verbose:
        print("=" * 80)
        print("NER 모델 성능 평가")
        print("=" * 80)
        print(f"✓ 사용 모델: {model_name}")
        print(f"✓ 평가 타입: {eval_type}")
        print(f"✓ 검증 데이터: {test_data_path}")
    
    # 1. 테스트 데이터 로드
    test_path = Path(test_data_path)
    if not test_path.exists():
        return {
            "success": False,
            "error": f"테스트 데이터 파일이 없습니다: {test_data_path}"
        }
    
    # BIO 포맷 파싱
    sentences = []
    current_sentence = []
    current_labels = []
    
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append((current_sentence.copy(), current_labels.copy()))
                    current_sentence = []
                    current_labels = []
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                token, label = parts[0], parts[1]
                current_sentence.append(token)
                current_labels.append(label)
    
    # 마지막 문장 처리
    if current_sentence:
        sentences.append((current_sentence.copy(), current_labels.copy()))
    
    if not sentences:
        return {
            "success": False,
            "error": "테스트 데이터가 비어있습니다."
        }
    
    # max_samples 적용
    original_count = len(sentences)
    if max_samples is not None and max_samples < len(sentences):
        sentences = sentences[:max_samples]
        if verbose:
            print(f"⚠️  샘플링: {max_samples}/{original_count}개 문장만 평가")
    
    if verbose:
        print(f"✓ 테스트 문장 수: {len(sentences):,}")
    
    # 2. 모델 로드
    model_path = get_model_path(model_name)
    
    # 로컬 모델이 없으면 Hugging Face에서 로드
    if not model_path.exists():
        if verbose:
            print(f"⚠️  로컬 모델이 없습니다. Hugging Face에서 로드: {model_name}")
        model_path_str = model_name  # Hugging Face 모델명 사용
        model_source = "huggingface"
    else:
        model_path_str = str(model_path)
        model_source = "local"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path_str)
        model = AutoModelForTokenClassification.from_pretrained(model_path_str)
        model.to(device)
        model.eval()
        
        # label_map 로드
        if model_source == "local":
            label_map_path = model_path / "label_map.json"
            if label_map_path.exists():
                with open(label_map_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                    if 'id2label' in label_data:
                        id2label = {int(k): v for k, v in label_data['id2label'].items()}
                    else:
                        id2label = {int(k): v for k, v in label_data.items()}
            else:
                id2label = {int(k): v for k, v in model.config.id2label.items()}
        else:
            id2label = {int(k): v for k, v in model.config.id2label.items()}
        
        if verbose:
            print(f"✓ 모델 로드 완료 (출처: {model_source}, 디바이스: {device})")
    except Exception as e:
        return {
            "success": False,
            "error": f"모델 로드 실패: {str(e)}"
        }
    
    # 3. 평가 시작 시간 측정
    eval_start_time = datetime.now()
    
    # 4. 예측 수행
    if verbose:
        print("예측 수행 중...")
    
    all_true_labels = []
    all_pred_labels = []
    successful_predictions = 0
    failed_predictions = 0
    
    with torch.no_grad():
        for idx, (sentence_tokens, sentence_labels) in enumerate(tqdm(sentences, disable=not verbose)):
            # 공백 없이 문장 구성
            text = ''.join(sentence_tokens)
            
            try:
                # 토큰화
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                    return_offsets_mapping=True
                )
                offset_mapping = inputs.pop('offset_mapping')[0]
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 예측
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                pred_label_ids = predictions[0].cpu().numpy()
                
                # Offset mapping을 사용하여 문자 단위 라벨 정렬
                char_pred_labels = ['O'] * len(text)
                
                for token_idx, (start, end) in enumerate(offset_mapping):
                    if start == 0 and end == 0:  # [CLS], [SEP], [PAD]
                        continue
                    
                    label_id = int(pred_label_ids[token_idx])
                    label = id2label.get(label_id, 'O')
                    
                    # 해당 offset 범위의 모든 문자에 라벨 할당
                    for char_idx in range(start, min(end, len(char_pred_labels))):
                        if char_idx == start and label.startswith('B-'):
                            char_pred_labels[char_idx] = label
                        elif label.startswith('B-'):
                            char_pred_labels[char_idx] = label.replace('B-', 'I-')
                        else:
                            char_pred_labels[char_idx] = label
                
                # Ground truth 라벨 (문자 단위)
                char_true_labels = []
                for token, label in zip(sentence_tokens, sentence_labels):
                    for i, char in enumerate(token):
                        if i == 0 and label.startswith('B-'):
                            char_true_labels.append(label)
                        elif label.startswith('B-'):
                            char_true_labels.append(label.replace('B-', 'I-'))
                        else:
                            char_true_labels.append(label)
                
                # 길이 검증 및 맞추기
                if len(char_true_labels) != len(char_pred_labels):
                    if debug:
                        print(f"\n⚠️  문장 {idx+1}: 라벨 길이 불일치")
                        print(f"  True: {len(char_true_labels)}, Pred: {len(char_pred_labels)}")
                    
                    # 짧은 쪽에 맞춤
                    min_len = min(len(char_true_labels), len(char_pred_labels))
                    char_true_labels = char_true_labels[:min_len]
                    char_pred_labels = char_pred_labels[:min_len]
                
                # 리스트에 추가
                all_true_labels.append(char_true_labels)
                all_pred_labels.append(char_pred_labels)
                successful_predictions += 1
                
            except Exception as e:
                if debug:
                    print(f"\n⚠️  문장 {idx+1} 예측 실패: {e}")
                failed_predictions += 1
                continue
    
    eval_end_time = datetime.now()
    eval_time = (eval_end_time - eval_start_time).total_seconds()
    
    if successful_predictions == 0:
        return {
            "success": False,
            "error": "모든 예측이 실패했습니다.",
            "evaluation_time": eval_time
        }
    
    # 5. 메트릭 계산 (seqeval 사용)
    try:
        from seqeval.metrics import precision_score as seqeval_precision
        from seqeval.metrics import recall_score as seqeval_recall
        from seqeval.metrics import f1_score as seqeval_f1
        
        # seqeval은 zero_division 파라미터를 지원하지 않으므로 제거
        precision = seqeval_precision(all_true_labels, all_pred_labels)
        recall = seqeval_recall(all_true_labels, all_pred_labels)
        f1 = seqeval_f1(all_true_labels, all_pred_labels)
        
        use_seqeval = True
    except ImportError:
        if verbose:
            print("⚠️  seqeval이 설치되지 않아 sklearn을 사용합니다.")
        
        # 평탄화
        flat_true = [label for sent in all_true_labels for label in sent]
        flat_pred = [label for sent in all_pred_labels for label in sent]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_true,
            flat_pred,
            average='weighted',
            zero_division=0
        )
        use_seqeval = False
    
    # 6. 결과 출력 (콘솔)
    if verbose:
        print("\n" + "=" * 80)
        print(f"{model_name} 평가 결과")
        print("=" * 80)
        print(f"평가 시간: {eval_time:.1f}s")
        print(f"성공: {successful_predictions:,}, 실패: {failed_predictions}")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")
    
    # 7. 결과 저장
    overall_end_time = datetime.now()
    
    results = {
        "success": True,
        "model_name": model_name,
        "test_data_path": str(test_path),
        "overall": {
            "precision": float(precision) * 100,
            "recall": float(recall) * 100,
            "f1_score": float(f1) * 100,
            "total_samples": successful_predictions
        },
        "evaluation_time": eval_time
    }
    
    # output_path/debug/ 디렉토리에 저장
    output_dir = Path(output_path) / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # {model_name}_evaluation.txt 파일 생성
    eval_file = output_dir / f"{model_name_safe}_evaluation.txt"
    
    # model_evaluation_log.txt 형식으로 저장
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NER Model Evaluation\n")
        f.write("=" * 80 + "\n")
        f.write(f"Start: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
        f.write("=" * 80 + "\n\n\n")
        
        f.write(f"{model_name}\n")
        f.write("=" * 80 + "\n")
        
        # 훈련 정보 추가
        if training_info:
            f.write(f"Training Time: {training_info.get('training_time', 0):.1f}s ({training_info.get('training_time_minutes', 0):.1f}min)\n")
            f.write(f"Training Completed: {training_info.get('completed_at', 'N/A')}\n")
            f.write(f"Training Epochs: {training_info.get('config', {}).get('epochs', 'N/A')}\n")
            f.write(f"Training Samples: {training_info.get('config', {}).get('num_train_samples', 'N/A'):,}\n")
            f.write("-" * 80 + "\n")
        
        f.write(f"Evaluation Time: {eval_time:.1f}s\n")
        f.write(f"Evaluation Samples: {successful_predictions:,}\n")
        f.write(f"Precision: {precision * 100:.2f}%\n")
        f.write(f"Recall: {recall * 100:.2f}%\n")
        f.write(f"F1 Score: {f1 * 100:.2f}%\n\n")
        
        f.write(f"Complete: {overall_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
        f.write("=" * 80 + "\n\n")
    
    if verbose:
        print(f"\n✓ 평가 결과 저장: {eval_file}")
        print("=" * 80)
    
    return results


def main():
    """메인 함수 - 사용 예제"""
    print("NER 시스템 테스트")
    
    # 1. 엔티티 추출 테스트
    test_text = """저작물 저작재산권 양도 계약서

계약자: 김민수
전화번호: 010-1234-5678
이메일: minsu.kim@gmail.com
주소: 서울시 강남구 테헤란로 123

수탁기관: 한국콘텐츠진흥원
담당자: 박영희 부장
계약금: 5,000,000원"""
    
    entities = extract_entities_from_text(test_text, debug=True)
    
    print(f"\n추출된 엔티티 ({len(entities)}개):")
    for entity, label in entities:
        print(f"  - {entity} ({label})")
    
    # 2. 훈련 상태 확인
    print(f"\n현재 모델 상태:")
    status = get_training_status()
    print(f"  - 모델 존재: {status['model_exists']}")
    print(f"  - 경로: {status['model_path']}")
    if status.get('checkpoints', 0) > 0:
        print(f"  - 체크포인트: {status['checkpoints']}개")

if __name__ == "__main__":
    main()

