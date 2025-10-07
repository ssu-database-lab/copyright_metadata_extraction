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

def check_system_requirements():
    """시스템 요구사항 확인"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU 사용 가능: {gpu_name} ({memory_gb:.1f}GB)")
    else:
        print("CPU 모드로 실행")
    
    return device

def get_model_path(model_name: str = DEFAULT_MODEL_NAME) -> Path:
    """
    모델 저장 경로 반환
    
    경로 구조: api/models/ner/{model_name}/
    예: api/models/ner/klue-roberta-large/
    """
    current_dir = Path(__file__).parent
    api_dir = current_dir.parent.parent
    
    # 새로운 경로 구조: models/ner/{model_name}
    models_base_dir = api_dir / "models" / "ner"
    models_base_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델명에서 슬래시를 대시로 변경 (예: klue/roberta-large -> klue-roberta-large)
    model_name_safe = model_name.replace('/', '-')
    
    model_path = models_base_dir / model_name_safe
    return model_path

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

def train_model_if_needed(model_path: Path, verbose: bool = True) -> bool:
    """필요시 모델 훈련"""
    if model_path.exists() and (model_path / "config.json").exists():
        if verbose:
            print("기존 모델을 사용합니다.")
        return True
    
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

def extract_entities_from_text(text: str, model_name: Optional[str] = None, debug: bool = False) -> List[Tuple[str, str]]:
    """텍스트에서 엔티티 추출 (통합 메인 함수)"""
    if debug:
        print(f"엔티티 추출 시작 (텍스트 길이: {len(text)}자)")
    
    # 모델 이름이 지정되지 않으면 기본 모델 사용
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
    
    if debug:
        print(f"사용 모델: {model_name}")
    
    all_entities = set()
    
    # 모델 경로 확인 및 훈련
    model_path = get_model_path(model_name)
    
    if not train_model_if_needed(model_path, verbose=debug):
        if debug:
            print("모델 훈련에 실패했습니다. 정규표현식만 사용합니다.")
    
    try:
        # 1. B-I-O 모델 기반 예측
        if model_path.exists() and (model_path / "config.json").exists():
            if debug:
                print("B-I-O 태깅 기반 예측 시작...")
            
            tokenizer, model, id2label, device = load_model_and_tokenizer(model_path, verbose=debug)
            bio_entities = extract_entities_by_bio_tagging(text, tokenizer, model, id2label, device)
            all_entities.update(bio_entities)
            
            if debug:
                print(f"B-I-O 예측 결과: {len(bio_entities)}개 엔티티")
        
    except Exception as e:
        if debug:
            print(f"모델 예측 오류: {e}")
    
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
    entity_filter: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    디렉토리 또는 파일에 대한 NER 예측 수행
    """
    start_time = time.time()
    
    try:
        print("=" * 60)
        print("NER 예측 시스템 시작")  
        print("=" * 60)
        
        # 1. 시스템 요구사항 확인
        device = check_system_requirements()
        
        # 2. 입력 경로 확인
        input_path_obj = Path(input_path)
        output_path_obj = Path(output_path)
        
        if not input_path_obj.exists():
            return {
                "success": False,
                "error": f"입력 경로가 존재하지 않습니다: {input_path_obj}"
            }
        
        # 3. 출력 디렉토리 생성 (pdf_to_image와 동일한 패턴)
        # ner 디렉토리를 최상단으로 바인딩
        ner_dir = output_path_obj / "ner"
        ner_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"출력 디렉토리: {ner_dir}")
        
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
        
        print(f"처리할 파일 수: {len(files_to_process)}")
        
        # 5. 엔티티 추출 시작
        print("엔티티 추출 시작...")
        
        all_entities = []
        processed_files = 0
        
        for file_path in tqdm(files_to_process, desc="파일 처리 중"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content.strip()) < 10:
                    continue
                
                # 엔티티 추출 (model_name 전달)
                entities = extract_entities_from_text(content, model_name=model_name, debug=False)
                
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
    callback_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    NER 모델 훈련 API
    
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
    
    Returns:
        Dict[str, Any]: 훈련 결과 정보
    """
    start_time = time.time()
    
    try:
        print("=" * 60)
        print("NER 모델 훈련 API 시작")
        print("=" * 60)
        
        # 1. 시스템 요구사항 확인
        device = check_system_requirements()
        
        # 2. 모델 경로 설정
        if output_dir:
            model_path = Path(output_dir)
        else:
            model_path = get_model_path(model_name)
        
        # 3. 기존 모델 확인
        if not force_retrain and model_path.exists() and (model_path / "config.json").exists():
            return {
                "success": True,
                "message": "기존 훈련된 모델을 사용합니다.",
                "model_path": str(model_path),
                "training_time": 0,
                "skipped": True
            }
        
        print(f"모델 저장 경로: {model_path}")
        print(f"훈련 설정:")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Learning Rate: {learning_rate}")
        print(f"  - Max Length: {max_length}")
        print(f"  - FP16: {enable_fp16}")
        
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