#!/usr/bin/env python3
"""
통합 NER 시스템 (Named Entity Recognition System)

Features:
- 이중 예측 시스템 (BERT-CRF 모델 + 정규표현식 패턴)
- 23가지 엔티티 타입 지원
- 자동 모델 훈련 및 다운로드
- 고성능 배치 처리

Usage:
    from module.ner.ner_system import ner_predict
    
    result = ner_predict(
        source_type="text",
        source_content="홍길동은 서울시 강남구에 거주합니다."
    )
"""

# ========== Standard Library Imports ==========
import os
import sys
import json
import time
import logging
import re
import subprocess
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set, Tuple

# ========== Third-party Imports ==========
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm

# ========== Configuration ==========
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ========== Constants ==========

# 엔티티 타입 정의 (ner_train.py와 동일한 순서 유지 필수)
ENTITY_TYPES = [
    "NAME",           # 이름
    "PHONE",          # 전화번호
    "ADDRESS",        # 주소
    "DATE",           # 날짜
    "COMPANY",        # 회사/기관명
    "EMAIL",          # 이메일
    "POSITION",       # 직책/직위
    "CONTRACT_TYPE",  # 계약서 유형
    "MONEY",          # 금액
    "PERIOD",         # 기간
    "ID_NUM",         # 신분증번호
    "CONSENT_TYPE",   # 동의서 유형
    "RIGHT_INFO",     # 권리정보
    "PROJECT_NAME",   # 사업명
    "LAW_REFERENCE",  # 법령 근거
    "TITLE",          # 제목
    "URL",            # URL정보
    "DESCRIPTION",    # 설명
    "TYPE",           # 유형
    "STATUS",         # 상태
    "DEPARTMENT",     # 부서정보
    "LANGUAGE",       # 언어
    "QUANTITY"        # 수량정보
]

DEFAULT_MAX_LENGTH = 512

# ========== Helper Functions - Configuration ==========

def load_default_model_name() -> str:
    """model_config.json에서 기본 모델 이름 로드"""
    try:
        config_path = Path(__file__).parent.parent.parent / "model_config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            default_model = config.get("ner", {}).get("default_model", "google-bert/bert-base-multilingual-cased")
            print(f"OK: 기본 모델 설정: {default_model}")
            return default_model
    except Exception as e:
        print(f"WARNING: model_config.json 로드 실패: {e}, 기본값 사용")
    return "google-bert/bert-base-multilingual-cased"


DEFAULT_MODEL_NAME = load_default_model_name()


# ========== Helper Functions - System ==========

def check_system_requirements(verbose: bool = False) -> torch.device:
    """시스템 요구사항 확인 및 디바이스 반환"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU 사용 가능: {gpu_name} ({memory_gb:.1f}GB)")
        else:
            print("CPU 모드로 실행")
    
    return device


# ========== Helper Functions - Model Path ==========

def get_model_path(model_name: str = DEFAULT_MODEL_NAME) -> Path:
    """
    모델 저장 경로 반환 및 준비
    
    경로 우선순위:
    1. models/ner/{model_name}/ - 사용 가능하면 반환
    2. model_downloaded/{model_name}/ - 있으면 복사 후 반환
    3. Hugging Face에서 다운로드 후 반환
    
    Returns:
        Path: models/ner/{model_name}/ 경로
    """
    current_dir = Path(__file__).parent
    api_dir = current_dir.parent.parent
    model_name_safe = model_name.replace('/', '-')
    
    # 목표 경로: models/ner/{model_name}
    models_ner_dir = api_dir / "models" / "ner" / model_name_safe
    models_ner_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. models/ner/{model_name} 확인
    if models_ner_dir.exists() and (models_ner_dir / "config.json").exists():
        print(f"OK: 모델 발견: {models_ner_dir}")
        return models_ner_dir
    
    # 2. model_downloaded/{model_name} 확인 및 복사
    model_downloaded_dir = api_dir / "model_downloaded" / model_name_safe
    if model_downloaded_dir.exists() and (model_downloaded_dir / "config.json").exists():
        print(f"OK: 다운로드된 모델 발견: {model_downloaded_dir}")
        print(f"[Package] 모델 복사 중: {model_downloaded_dir} → {models_ner_dir}")
        
        import shutil
        if models_ner_dir.exists():
            shutil.rmtree(models_ner_dir)
        shutil.copytree(model_downloaded_dir, models_ner_dir)
        
        print(f"OK: 모델 복사 완료: {models_ner_dir}")
        return models_ner_dir
    
    # 3. Hugging Face에서 다운로드
    print(f"WARNING: 로컬에 모델 없음: {model_name}")
    print(f"[Download] Hugging Face에서 다운로드 중...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        model_downloaded_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"  → 다운로드 위치: {model_downloaded_dir}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        tokenizer.save_pretrained(str(model_downloaded_dir))
        model.save_pretrained(str(model_downloaded_dir))
        print(f"✓ 다운로드 완료: {model_downloaded_dir}")
        
        # models/ner/로 복사
        import shutil
        if models_ner_dir.exists():
            shutil.rmtree(models_ner_dir)
        shutil.copytree(model_downloaded_dir, models_ner_dir)
        print(f"✓ 모델 복사 완료: {models_ner_dir}")
        
        return models_ner_dir
        
    except Exception as e:
        print(f"ERROR: 모델 다운로드 실패: {e}")
        return models_ner_dir


# ========== Helper Functions - Model Loading ==========

def load_model_and_tokenizer(model_path: Path, verbose: bool = True):
    """모델과 토크나이저 로드 (BERT-CRF 커스텀 모델 지원)"""
    if verbose:
        print(f"모델 로드 중: {model_path}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
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
    
    # BERT-CRF 커스텀 모델 로드
    model_pt_path = model_path / "model.pt"
    if model_pt_path.exists():
        # 커스텀 BERT-CRF 모델 사용
        from .ner_train import BertCrfForNER
        
        # config.json에서 모델 이름 확인
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            model_name = config.get('_name_or_path', 'google-bert/bert-base-multilingual-cased')
        else:
            model_name = 'google-bert/bert-base-multilingual-cased'
        
        num_labels = len(id2label)
        model = BertCrfForNER(model_name=model_name, num_labels=num_labels)
        
        # 훈련된 가중치 로드
        model.load_state_dict(torch.load(model_pt_path, map_location='cpu'))
        if verbose:
            print(f"   BERT-CRF 모델 로드 완료: {num_labels}개 라벨")
    else:
        # 표준 HuggingFace 모델
        model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        if verbose:
            print(f"   HuggingFace 모델 로드 완료")
    
    # GPU 사용 가능하면 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    if verbose:
        print(f"모델 로드 완료 ({device})")
    return tokenizer, model, id2label, device

def extract_entities_by_bio_tagging(text: str, tokenizer, model, id2label: dict, device) -> Set[Tuple[str, str]]:
    """B-I-O 태깅 기반 엔티티 추출 (강화된 버전 + 전체 엔티티 문맥 기반 보정)"""
    entities = set()
    
    # === 엔티티별 패턴 사전 ===
    import re
    
    # 한국어 성씨 (NAME용)
    korean_surnames = {
        "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
        "한", "오", "서", "신", "권", "황", "안", "송", "류", "전",
        "홍", "고", "문", "양", "손", "배", "백", "허", "남", "심",
        "노", "하", "곽", "성", "차", "주", "우", "구", "라", "진"
    }
    
    # 엔티티 타입별 패턴 감지 함수
    def detect_entity_type_by_pattern(token_text: str, context_before: str = "", context_after: str = "") -> tuple:
        """
        패턴 기반으로 엔티티 타입 감지
        Returns: (entity_type, confidence_boost)
        """
        # PHONE - 전화번호 패턴
        if re.match(r'^[\d-]+$', token_text) and len(token_text) >= 8:
            if re.search(r'\d{2,4}-?\d{3,4}-?\d{4}', token_text):
                return ('PHONE', 0.8)
        
        # EMAIL - 이메일 패턴
        if '@' in token_text or re.search(r'[a-zA-Z0-9._%+-]+@', context_before + token_text + context_after):
            return ('EMAIL', 0.8)
        
        # URL - URL 패턴
        if any(x in token_text.lower() for x in ['http', 'www', '.com', '.kr', '.net']):
            return ('URL', 0.7)
        
        # DATE - 날짜 패턴
        if re.search(r'\d{4}년', token_text) or re.search(r'\d{1,2}월', token_text) or re.search(r'\d{1,2}일', token_text):
            return ('DATE', 0.7)
        if re.match(r'^\d{4}-\d{1,2}-\d{1,2}$', token_text):
            return ('DATE', 0.9)
        
        # MONEY - 금액 패턴
        if '원' in token_text or '만원' in token_text or '천원' in token_text:
            return ('MONEY', 0.7)
        if re.search(r'\d+,\d+', token_text) and ('원' in context_after[:5] or '만' in context_after[:5]):
            return ('MONEY', 0.6)
        
        # LAW_REFERENCE - 법률 조항
        if '법' in token_text and '제' in token_text:
            return ('LAW_REFERENCE', 0.8)
        if re.search(r'제\s*\d+조', token_text):
            return ('LAW_REFERENCE', 0.9)
        
        # COMPANY - 회사명
        if any(x in token_text for x in ['㈜', '주식회사', '재단', '협회', '연구소', '센터']):
            return ('COMPANY', 0.7)
        
        # ID_NUM - 식별번호
        if re.match(r'^\d{6}-\d{7}$', token_text):  # 주민등록번호
            return ('ID_NUM', 0.9)
        if re.match(r'^\d{3}-\d{2}-\d{5}$', token_text):  # 사업자등록번호
            return ('ID_NUM', 0.9)
        
        # PERIOD - 기간
        if any(x in token_text for x in ['일간', '개월', '년간', '주간', '분기']):
            return ('PERIOD', 0.6)
        
        # ADDRESS - 주소
        if any(x in token_text for x in ['시', '도', '구', '동', '로', '길']):
            if '특별시' in token_text or '광역시' in token_text or '도' in context_before + token_text:
                return ('ADDRESS', 0.7)
        
        # NAME - 이름 (성씨 기반)
        if token_text and len(token_text) >= 2 and token_text[0] in korean_surnames:
            name_indicators = ['은', '는', '이', '가', '을', '를', '의', '님', '씨', '귀', '께']
            if any(ind in context_after[:2] for ind in name_indicators):
                return ('NAME', 0.6)
            if any(x in context_before for x in ['작성자', '저작자', '본인', '담당', '대표']):
                return ('NAME', 0.7)
        
        # POSITION - 직책
        if any(x in token_text for x in ['팀장', '과장', '부장', '사장', '대표', '이사', '사원', '주임']):
            return ('POSITION', 0.7)
        
        # TITLE - 제목
        if '프로젝트' in context_before and len(token_text) > 3:
            return ('TITLE', 0.5)
        
        return (None, 0.0)
    
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
                
                # BERT-CRF 모델인지 표준 모델인지 확인
                if hasattr(outputs, 'logits'):
                    # 표준 HuggingFace 모델
                    predictions = torch.softmax(outputs.logits, dim=-1)
                    predicted_labels = torch.argmax(predictions, dim=-1).squeeze(0).tolist()
                    confidence_scores = torch.max(predictions, dim=-1)[0].squeeze(0).tolist()
                    all_probs = predictions.squeeze(0)
                elif isinstance(outputs, dict):
                    if 'predictions' in outputs:
                        # BERT-CRF 모델 (inference mode - Viterbi decoding)
                        predicted_labels = outputs['predictions'].squeeze(0).tolist()
                        confidence_scores = [1.0] * len(predicted_labels)
                        all_probs = None
                    elif 'logits' in outputs:
                        # BERT-CRF 모델 (training mode - 학습 중 평가)
                        logits = outputs['logits']
                        predictions = torch.softmax(logits, dim=-1)
                        predicted_labels = torch.argmax(predictions, dim=-1).squeeze(0).tolist()
                        confidence_scores = torch.max(predictions, dim=-1)[0].squeeze(0).tolist()
                        all_probs = predictions.squeeze(0)
                    else:
                        raise ValueError(f"Unexpected output format: {outputs.keys()}")
                else:
                    raise ValueError(f"Unknown model output type: {type(outputs)}")
            
            # B-I-O 태깅으로 엔티티 추출
            current_entity = ""
            current_type = None
            current_start = -1
            current_confidence = 0.0
            
            for token_idx, (pred_id, confidence, (start, end)) in enumerate(zip(predicted_labels, confidence_scores, offset_mapping)):
                if start == 0 and end == 0:  # 특수 토큰 건너뛰기
                    continue
                
                pred_label = id2label.get(pred_id, 'O')
                token_text = sentence[start:end]
                
                # === 문맥 기반 능동적 보정 (모든 엔티티) ===
                context_before = sentence[max(0, start-10):start]
                context_after = sentence[end:min(len(sentence), end+10)]
                
                # 패턴 기반 엔티티 타입 감지
                detected_type, pattern_confidence = detect_entity_type_by_pattern(
                    token_text, context_before, context_after
                )
                
                if detected_type and all_probs is not None:
                    # 감지된 타입의 B-, I- 라벨 확률 체크
                    for label_id, label_name in id2label.items():
                        if label_name.endswith(f'-{detected_type}'):
                            type_prob = all_probs[token_idx][label_id].item()
                            # 패턴 매칭 + 모델 확률 조합
                            combined_confidence = max(type_prob, pattern_confidence * type_prob)
                            
                            # 낮은 확률이라도 패턴이 강하면 승격 (0.1 이상)
                            if combined_confidence > 0.1 or (pattern_confidence > 0.7 and type_prob > 0.05):
                                pred_label = label_name
                                confidence = max(confidence, combined_confidence)
                                break
                
                # 기존 threshold 체크 (완화: CRF 기반 모델용)
                if confidence < 1.0 and confidence < 0.10:
                    pred_label = 'O'
                
                if pred_label.startswith('B-'):
                    # 이전 엔티티 저장
                    if current_entity and current_type and current_confidence > 0.10:
                        clean_entity = clean_entity_text(current_entity)
                        if is_valid_entity(clean_entity, current_type):
                            entities.add((clean_entity, current_type))
                    
                    # 새 엔티티 시작
                    current_entity = sentence[start:end]
                    current_type = pred_label[2:]
                    current_start = start
                    current_confidence = confidence
                    
                elif pred_label.startswith('I-') and current_type == pred_label[2:]:
                    # 엔티티 연속
                    if current_start != -1:
                        current_entity = sentence[current_start:end]
                        current_confidence = max(current_confidence, confidence)
                    
                else:
                    # 엔티티 종료
                    if current_entity and current_type and current_confidence > 0.10:
                        clean_entity = clean_entity_text(current_entity)
                        if is_valid_entity(clean_entity, current_type):
                            entities.add((clean_entity, current_type))
                    
                    # 초기화
                    current_entity = ""
                    current_type = None
                    current_start = -1
                    current_confidence = 0.0
            
            # 마지막 엔티티 저장
            if current_entity and current_type and current_confidence > 0.10:
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
    """텍스트를 스마트하게 분할 (문장 단위)"""
    if len(text) <= max_length:
        return [text]
    
    text_sentences = text.replace('\n', '. ').split('.')
    sentences = []
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
    """엔티티 텍스트 정리 (앞뒤 특수문자 제거)"""
    entity = entity.strip()
    entity = re.sub(r'^[:\s,.-]+', '', entity)
    entity = re.sub(r'[:\s,.-]+$', '', entity)
    return entity


def group_entities_by_type(entities: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """엔티티를 타입별로 그룹화
    
    Args:
        entities: [(값, 타입), ...] 리스트
    
    Returns:
        {타입: [값1, 값2, ...], ...} 딕셔너리
    """
    grouped = {}
    for entity, entity_type in entities:
        if entity_type not in grouped:
            grouped[entity_type] = []
        grouped[entity_type].append(entity)
    
    # 알파벳 순으로 정렬
    return dict(sorted(grouped.items()))

def is_valid_entity(entity: str, entity_type: Optional[str] = None) -> bool:
    """엔티티 유효성 검증
    
    Args:
        entity: 검증할 엔티티 텍스트
        entity_type: 엔티티 타입 (선택)
    
    Returns:
        bool: 유효하면 True
    """
    # 기본 검증
    if len(entity) < 2 or len(entity) > 50:
        return False
    
    # 줄바꿈 제한
    if '\n' in entity:
        line_count = entity.count('\n')
        if entity_type == 'ADDRESS' and line_count > 1:
            return False
        elif line_count > 2:
            return False
    
    # 불필요한 문자 제외
    invalid_chars = ['□', '☑', '○', '●']
    if any(char in entity for char in invalid_chars):
        return False
    
    # 숫자만 제외
    if entity.isdigit():
        return False
    
    # 불완전한 단어 제외
    if entity.endswith(('.', ',', '·', ')', '(', ':')):
        return False
    if entity.startswith(('.', ',', '·', ')', '(')):
        return False
    # 조사만 있는 경우 제외
    josa_list = ['을', '를', '가', '는', '은', '의', '이', '에', '에서', '에게', '부터', '까지', '으로', '로', '과', '와', '및']
    if entity in josa_list:
        return False
    
    # 타입별 특수 검증
    if entity_type:
        if entity_type == 'NAME':
            if entity in ['양도', '양수', '제공', '수령', '대표', '담당', '관리', '저작', '회사', '기관']:
                return False
            if entity.endswith(('자', '인', '처')) and len(entity) <= 4:
                if entity not in ['김자', '이자', '박인', '한자']:
                    return False
            role_words = ['양도자', '양도인', '양수인', '양수자', '제공자', '이용자', '수령자', '수령인']
            if entity in role_words:
                return False
        
        elif entity_type == 'COMPANY':
            general_nouns = ['연락처', '주소', '성명', '전화', '휴대', '담당', '회사', '기관']
            if entity in general_nouns or any(x in entity for x in ['연락처', '주소']):
                return False
            if entity.endswith('로부') or ('법원' in entity and len(entity) <= 6):
                return False
            if len(entity) < 4 and entity not in ['KBS', 'MBC', 'SBS', 'EBS']:
                return False
        
        elif entity_type == 'PHONE':
            if re.match(r'\d{4}\.\s*\d{1,2}\.\s*\d{1,2}', entity):
                return False
            digits_only = ''.join(c for c in entity if c.isdigit())
            if len(digits_only) < 7:
                return False
        
        elif entity_type == 'DESCRIPTION':
            if len(entity) < 5:
                return False
            if entity.endswith(('을', '를', '가', '는', '이', '의', '으로')):
                return False
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
        
        print(f"모델 저장 중... ({model_path})")
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
            print(f"OK: 모델 다운로드 완료!")
            print(f"   - config.json: {(model_path / 'config.json').exists()}")
            print(f"   - model.safetensors: {(model_path / 'model.safetensors').exists()}")
            print(f"   - tokenizer.json: {(model_path / 'tokenizer.json').exists()}")
            print(f"   - label_map.json: {label_map_file.exists()}")
            print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"ERROR: 모델 다운로드 실패: {e}")
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
        print(f"[경고] 로컬에 모델이 없습니다: {model_path}")
        print(f"Hugging Face에서 '{model_name}' 모델을 검색합니다...")
        
        if download_pretrained_model(model_name, model_path, verbose):
            print(f"OK: 모델 다운로드 완료! Fine-tuning 없이 사용 가능합니다.")
            return True
        else:
            print(f"경고: Hugging Face에서 모델을 다운로드할 수 없습니다.")
    
    # 2단계: 자동 훈련 시도 (auto_train=True인 경우)
    if not auto_train:
        print(f"경고: 자동 훈련이 비활성화되어 있습니다.")
        print(f"[경고] 다음 중 하나를 선택하세요:")
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

def extract_entities_from_text(text: str, model_name: Optional[str] = None, debug: bool = False) -> List[Tuple[str, str]]:
    """
    텍스트에서 엔티티 추출 (통합 메인 함수)
    
    Args:
        text: 입력 텍스트
        model_name: 모델 이름
        debug: 디버그 로그 출력
        
    Note:
        훈련이 필요한 경우 ner_train() 함수를 별도로 호출하세요.
        이 함수는 예측만 수행합니다.
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
                print(f"[경고] 모델 다운로드 실패: {e}")
                print(f"정규표현식만 사용합니다.")
    
    # 모델이 없으면 에러 (train 파라미터 제거)
    if not model_exists:
        if debug:
            print("=" * 60)
            print(f"[경고] 모델 없음: {model_name}")
            print("ner_train() 함수를 먼저 호출하여 모델을 훈련하세요.")
            print("=" * 60)
    
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
    
    # 2. 정규표현식 백업 예측 (기본 패턴만)
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
    debug: bool = False
) -> Dict[str, Any]:
    """
    디렉토리 또는 파일에 대한 NER 예측 수행 (순수 예측만)
    
    Args:
        input_path: 입력 파일/디렉토리 경로
        output_path: 출력 디렉토리 경로
        model_name: 사용할 모델 이름 (기본값: klue/roberta-large)
        confidence_threshold: 신뢰도 임계값
        output_format: 출력 형식
        save_statistics: 통계 저장 여부
        entity_filter: 추출할 엔티티 타입 필터
        debug: 상세 로그 출력 여부
    
    Returns:
        Dict[str, Any]: 예측 결과 정보
    
    Note:
        - 훈련이 필요한 경우 ner_train() 함수를 먼저 호출하세요
        - 이 함수는 예측만 수행합니다
    """
    start_time = time.time()
    
    # 모델 이름이 지정되지 않으면 기본 모델 사용
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
        if debug:
            print(f"[경고] 모델 이름이 지정되지 않아 기본 모델 사용: {model_name}")
    
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
            for ext in ['*.txt', '*.md']:
                files_to_process.extend(input_path_obj.glob(f"**/{ext}"))
        
        if not files_to_process:
            return {
                "success": False,
                "error": "처리할 텍스트 파일이 없습니다."
            }
        
        if debug:
            print(f"처리할 파일 수: {len(files_to_process)}")
        
        # 5. 모델 확인 (없으면 에러)
        model_path = get_model_path(model_name)
        model_exists = model_path.exists() and (model_path / "config.json").exists()
        
        if not model_exists:
            return {
                "success": False,
                "error": f"모델이 없습니다: {model_name}. ner_train() 함수를 먼저 실행하세요."
            }
        
        if debug:
            print(f"✓ 모델 로드: {model_path}")
        
        # 6. 엔티티 추출 시작
        if debug:
            print("엔티티 추출 시작...")
        
        all_entities = []
        processed_files = 0
        
        import sys
        for file_path in tqdm(files_to_process, desc="파일 처리 중", disable=False, file=sys.stdout, ncols=80):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content.strip()) < 2:  # 최소 2글자 이상
                    continue
                
                entities = extract_entities_from_text(content, model_name=model_name, debug=debug)
                
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

def ensure_training_data_exists(model_name: str, num_samples: int = 7500, force_regenerate: bool = False, use_large_dataset: bool = True) -> bool:
    """
    훈련/검증 데이터가 존재하는지 확인하고 없거나 force_regenerate=True면 새로 생성
    
    Args:
        model_name: 모델 이름
        num_samples: 생성할 총 샘플 수 (기본값: 7500, 범위: 5000~10000)
        force_regenerate: True면 기존 데이터를 삭제하고 재생성 (overfitting 방지)
        use_large_dataset: True면 미리 생성된 대량 데이터 사용 (data/in/ner/large_training_data.txt)
    
    Returns:
        bool: 데이터 준비 성공 여부
    
    Note:
        - 생성된 데이터의 80%는 train.txt (훈련용)
        - 나머지 20%는 validation.txt (검증/평가용)
        - test.txt는 validation.txt와 동일하게 생성 (호환성)
        - use_large_dataset=True시 대량 데이터 우선 사용 (2,000+ 샘플)
    """
    try:
        current_dir = Path(__file__).parent
        model_name_safe = model_name.replace('/', '-')
        training_dir = current_dir / "training" / model_name_safe
        
        train_file = training_dir / "train.txt"
        val_file = training_dir / "validation.txt"
        test_file = training_dir / "test.txt"
        
        # 대량 데이터셋 경로 (realistic 데이터 우선!)
        realistic_dataset_path = Path(__file__).parent.parent.parent / "data" / "in" / "ner" / "realistic_training_data.txt"
        large_dataset_path = Path(__file__).parent.parent.parent / "data" / "in" / "ner" / "large_training_data.txt"
        
        # force_regenerate=True 또는 파일이 없으면 생성
        if force_regenerate or not (train_file.exists() and val_file.exists()):
            if force_regenerate:
                print(f"\n{'='*60}")
                print(f"훈련 데이터 재생성 (Overfitting 방지)")
                print(f"{'='*60}")
            else:
                print(f"\n{'='*60}")
                print(f"훈련 데이터 생성")
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
            
            # 🔥 항상 파라미터 개수만큼 새로 생성 (기존 파일 무시)
            print(f"\n[데이터 생성] 훈련 데이터 생성 중... (파라미터: {num_samples:,}개)")
            
            # ner_train.py의 데이터 생성 함수 import
            try:
                from .ner_train import generate_rich_training_data
                
                # 훈련 데이터 생성 (무작위)
                result = generate_rich_training_data(training_dir, num_samples=num_samples)
                
                if result and train_file.exists() and val_file.exists():
                    # test.txt는 validation.txt와 동일하게 생성 (호환성)
                    if val_file.exists():
                        import shutil
                        shutil.copy(val_file, test_file)
                    
                    print(f"훈련 데이터 생성 완료!")
                    print(f"  - Train: {train_file} ({train_file.stat().st_size // 1024}KB)")
                    print(f"  - Validation: {val_file} ({val_file.stat().st_size // 1024}KB)")
                    print(f"  - Test: {test_file} (검증 데이터 복사본)")
                    print(f"  - 실제 생성: {num_samples:,}개 샘플 (매번 무작위)")
                    return True
                else:
                    print(f"[경고] 데이터 생성 실패")
                    return False
                    
            except ImportError as e:
                print(f"[경고] ner_train.py를 import할 수 없습니다: {e}")
                return False
        else:
            print(f"✓ 기존 훈련 데이터 사용: {training_dir}")
            return True
            
    except Exception as e:
        print(f"[경고] 훈련 데이터 생성 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def ner_train(
    model_name: str = DEFAULT_MODEL_NAME,
    iterations: int = 1,
    epochs: int = 100,                # 기본값: 100 (충분한 학습)
    batch_size: int = 12,             # 기본값: 12 (메모리 효율)
    learning_rate: float = 1e-5,      # 기본값: 1e-5 (안정적 학습)
    num_train_samples: int = 30000,   # 기본값: 30,000 (대규모 데이터)
    enable_visualization: bool = True,
    enable_early_stopping: bool = False,
    debug: bool = False
) -> Dict[str, Any]:
    """
    NER 모델 훈련 전용 함수
    
    Args:
        model_name: 훈련할 모델 이름
        iterations: 훈련 반복 횟수 (각 반복마다 새 데이터 생성)
        epochs: 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
        num_train_samples: 생성할 샘플 수 (80/20 분할)
        enable_visualization: 학습 곡선 시각화 생성 여부
        enable_early_stopping: Early stopping 활성화 (기본 False)
        debug: 상세 로그 출력
    
    Returns:
        Dict[str, Any]: 훈련 결과 및 메트릭
    """
    overall_start = time.time()
    
    if debug:
        print("=" * 80)
        print("NER 모델 훈련 시작")
        print("=" * 80)
        print(f"모델: {model_name}")
        print(f"반복 횟수: {iterations}")
        print(f"에포크: {epochs}")
        print(f"학습률: {learning_rate}")
        print(f"샘플 수: {num_train_samples:,} (훈련 {int(num_train_samples*0.8):,} + 검증 {int(num_train_samples*0.2):,})")
        print("=" * 80)
    
    results = []
    all_metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'train_loss_history': [],
        'steps': []
    }
    
    for iteration in range(1, iterations + 1):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration}/{iterations}")
        print(f"{'='*80}")
        
        iteration_start = time.time()
        
        try:
            # 1. 모델 경로 준비
            model_path = get_model_path(model_name)
            
            # 2. 훈련 데이터 생성 (매번 새로 생성)
            print(f"\n훈련 데이터 재생성 (Iteration {iteration})")
            model_name_safe = model_name.replace('/', '-')
            training_dir = Path(__file__).parent / "training" / model_name_safe
            
            if not ensure_training_data_exists(model_name, num_samples=num_train_samples, force_regenerate=True):
                results.append({
                    "iteration": iteration,
                    "success": False,
                    "error": "훈련 데이터 생성 실패"
                })
                continue
            
            # 3. 훈련 직접 실행
            print(f"\n모델 훈련 시작 (Iteration {iteration})")
            
            try:
                from .ner_train import train_ner_model
                
                # 일반 모드
                model, tokenizer, metrics = train_ner_model(
                    model_name=model_name,
                    num_samples=num_train_samples,
                    num_epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    output_dir=str(model_path),
                    use_gpu=torch.cuda.is_available(),
                    use_realistic_data=True,  # 실전 기반 데이터 사용
                    enable_early_stopping=enable_early_stopping
                )
                
                iteration_time = time.time() - iteration_start
                
                # 훈련 정보 저장
                training_info = {
                    "iteration": iteration,
                    "training_time": iteration_time,
                    "training_time_minutes": iteration_time / 60,
                    "epochs": epochs,
                    "completed_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "config": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "model_name": model_name,
                        "num_train_samples": num_train_samples
                    },
                    "metrics": metrics
                }
                
                training_info_file = model_path / "training_info.json"
                with open(training_info_file, 'w', encoding='utf-8') as f:
                    json.dump(training_info, f, ensure_ascii=False, indent=2)
                
                # 최종 메트릭 추출 (history의 마지막 값)
                final_f1 = 0
                final_precision = 0
                final_recall = 0
                final_loss = 0
                if 'history' in metrics and metrics['history']:
                    hist = metrics['history']
                    if 'eval_f1' in hist and hist['eval_f1']:
                        final_f1 = hist['eval_f1'][-1]
                    if 'eval_precision' in hist and hist['eval_precision']:
                        final_precision = hist['eval_precision'][-1]
                    if 'eval_recall' in hist and hist['eval_recall']:
                        final_recall = hist['eval_recall'][-1]
                    if 'eval_loss' in hist and hist['eval_loss']:
                        final_loss = hist['eval_loss'][-1]
                
                # 메트릭 저장 (간단한 형식)
                all_metrics['epochs'].append(epochs)
                all_metrics['val_f1'].append(final_f1)
                all_metrics['val_precision'].append(final_precision)
                all_metrics['val_recall'].append(final_recall)
                all_metrics['val_loss'].append(final_loss)
                
                # 성공 기록
                results.append({
                    "iteration": iteration,
                    "success": True,
                    "metrics": metrics,
                    "training_time": iteration_time
                })
                
                print(f"\nIteration {iteration} 완료!")
                print(f"   - Best F1: {metrics.get('best_f1', 0):.4f}")
                print(f"   - Final F1: {final_f1:.4f}")
                print(f"   - Final Precision: {final_precision:.4f}")
                print(f"   - Final Recall: {final_recall:.4f}")
                print(f"   - Time elapsed: {iteration_time/60:.1f} min")
                
                # Generate visualization (save per iteration)
                if enable_visualization and 'history' in metrics:
                    try:
                        from .ner_train import plot_training_curves
                        
                        # Visualization directory structure
                        model_safe_name = model_name.replace('/', '-')
                        base_viz_dir = Path("data/out/ner_visualization") / model_safe_name / str(iteration)
                        base_viz_dir.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now().strftime("%y%m%d%H%M")
                        viz_path = base_viz_dir / f"{model_safe_name}_{timestamp}.png"
                        plot_training_curves(
                            metrics['history'],
                            viz_path,
                            model_name
                        )
                        print(f"   Visualization saved: {viz_path}")
                            
                    except Exception as e:
                        print(f"   Warning: Visualization generation failed: {e}")
                        import traceback
                        traceback.print_exc()
                
            except Exception as e:
                results.append({
                    "iteration": iteration,
                    "success": False,
                    "error": f"훈련 함수 호출 실패: {e}"
                })
                print(f"\nERROR: Iteration {iteration} 실패: {e}")
                import traceback
                traceback.print_exc()
        
        except Exception as e:
            results.append({
                "iteration": iteration,
                "success": False,
                "error": str(e)
            })
            print(f"\nERROR: Iteration {iteration} 오류: {e}")
    
    total_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print(f"전체 훈련 완료!")
    print(f"총 소요 시간: {total_time/60:.1f}분")
    print(f"완료된 에포크: {len(all_metrics['epochs'])}")
    print(f"성공: {sum(1 for r in results if r['success'])}/{iterations}")
    print(f"{'='*80}\n")
    
    return {
        "success": True,
        "model_name": model_name,
        "iterations": iterations,
        "results": results,
        "metrics": all_metrics,
        "total_time": total_time
    }


def extract_metrics_from_logs(logs: List[str]) -> Dict[str, Any]:
    """훈련 로그에서 메트릭 추출 (에포크별)"""
    import re
    import json
    
    metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'train_loss_history': [],
        'steps': []
    }
    
    current_epoch = None
    step_count = 0
    epoch_data = {}  # 에포크별로 데이터 수집
    
    for line in logs:
        # Transformers Trainer 로그 파싱: {'loss': 0.5, 'epoch': 1, ...}
        if "{'loss':" in line or '{"loss":' in line:
            try:
                # dict 형식 추출
                dict_match = re.search(r'\{[^}]+\}', line)
                if dict_match:
                    data_str = dict_match.group(0).replace("'", '"')
                    data = json.loads(data_str)
                    
                    # Epoch 정보
                    if 'epoch' in data:
                        current_epoch = int(float(data['epoch']))
                    
                    # Training loss (step별)
                    if 'loss' in data and 'eval' not in line:
                        loss_val = float(data['loss'])
                        metrics['train_loss_history'].append(loss_val)
                        step_count += 1
                        metrics['steps'].append(step_count)
            except Exception as e:
                pass
        
        # Evaluation 로그 파싱: eval_loss, eval_f1, etc.
        if 'eval_loss' in line or 'eval_f1' in line or 'eval_precision' in line:
            try:
                # Transformers 평가 결과: {'eval_loss': 0.234, 'eval_f1': 0.95, ...}
                dict_match = re.search(r'\{[^}]+\}', line)
                if dict_match:
                    data_str = dict_match.group(0).replace("'", '"')
                    data = json.loads(data_str)
                    
                    # Epoch 추정 (eval이 나올 때마다 epoch 증가)
                    if 'eval_loss' in data:
                        if current_epoch is None:
                            current_epoch = 1
                        
                        # 새로운 에포크 데이터 저장
                        if current_epoch not in epoch_data:
                            epoch_data[current_epoch] = {}
                        
                        epoch_data[current_epoch]['val_loss'] = float(data.get('eval_loss', 0))
                        epoch_data[current_epoch]['val_f1'] = float(data.get('eval_f1', 0)) * 100  # 0-1 -> 0-100
                        epoch_data[current_epoch]['val_precision'] = float(data.get('eval_precision', 0)) * 100
                        epoch_data[current_epoch]['val_recall'] = float(data.get('eval_recall', 0)) * 100
            except Exception as e:
                pass
        
        # 텍스트 형식 파싱 (명시적 출력)
        # "Epoch 1/3" 형식
        epoch_match = re.search(r'Epoch[:\s]+(\d+)', line, re.IGNORECASE)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        
        # "F1 Score: 0.9567" 형식
        if 'F1 Score:' in line and current_epoch:
            f1_match = re.search(r'F1 Score:\s*([\d.]+)', line)
            if f1_match:
                f1_val = float(f1_match.group(1))
                if f1_val <= 1.0:
                    f1_val *= 100
                if current_epoch not in epoch_data:
                    epoch_data[current_epoch] = {}
                epoch_data[current_epoch]['val_f1'] = f1_val
        
        # "Precision: 0.9567" 형식
        if 'Precision:' in line and current_epoch:
            prec_match = re.search(r'Precision:\s*([\d.]+)', line)
            if prec_match:
                prec_val = float(prec_match.group(1))
                if prec_val <= 1.0:
                    prec_val *= 100
                if current_epoch not in epoch_data:
                    epoch_data[current_epoch] = {}
                epoch_data[current_epoch]['val_precision'] = prec_val
        
        # "Recall: 0.9567" 형식
        if 'Recall:' in line and current_epoch:
            rec_match = re.search(r'Recall:\s*([\d.]+)', line)
            if rec_match:
                rec_val = float(rec_match.group(1))
                if rec_val <= 1.0:
                    rec_val *= 100
                if current_epoch not in epoch_data:
                    epoch_data[current_epoch] = {}
                epoch_data[current_epoch]['val_recall'] = rec_val
    
    # 에포크별 데이터를 리스트로 변환
    if epoch_data:
        sorted_epochs = sorted(epoch_data.keys())
        for epoch in sorted_epochs:
            metrics['epochs'].append(epoch)
            data = epoch_data[epoch]
            metrics['val_loss'].append(data.get('val_loss', 0))
            metrics['val_f1'].append(data.get('val_f1', 0))
            metrics['val_precision'].append(data.get('val_precision', 0))
            metrics['val_recall'].append(data.get('val_recall', 0))
            # Train loss는 에포크 평균 (나중에 계산)
            metrics['train_loss'].append(0)  # placeholder
    
    return metrics


def create_training_visualization(metrics: Dict, model_name: str) -> Optional[Path]:
    """
    학습 곡선 시각화 생성 (분류 모델 평가에 최적화)
    
    4개 그래프:
    1. Epoch별 Loss (Train vs Val) - 과적합 감지
    2. Epoch별 Performance Metrics (F1, Precision, Recall) - 성능 추이
    3. Steps별 Loss 수렴 - 상세 학습 과정
    4. Epoch별 F1 Score와 Loss 비교 - 성능-손실 관계
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import numpy as np
        
        # 한글 폰트 설정 (범용 폰트 사용)
        try:
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            
            # 한글 폰트 우선순위 리스트
            korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic']
            
            # 사용 가능한 한글 폰트 찾기
            font_found = False
            for font_name in korean_fonts:
                if font_name in available_fonts:
                    plt.rcParams['font.family'] = font_name
                    font_found = True
                    break
            
            if not font_found:
                # 한글 폰트가 없으면 DejaVu Sans 사용
                plt.rcParams['font.family'] = 'DejaVu Sans'
            
            plt.rcParams['axes.unicode_minus'] = False
            
        except Exception:
            # 폰트 설정 실패 시 기본 폰트 사용
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        epochs = metrics.get('epochs', [])
        
        # ==================== 1. Epoch별 Loss (Train vs Validation) ====================
        ax1 = fig.add_subplot(gs[0, 0])
        if epochs:
            if metrics.get('val_loss'):
                ax1.plot(epochs, metrics['val_loss'], 'r-o', label='Validation Loss', 
                        linewidth=3, markersize=10, markeredgewidth=2, markeredgecolor='white')
            if metrics.get('train_loss'):
                # train_loss가 있으면 표시 (없을 수도 있음)
                train_losses = [tl for tl in metrics['train_loss'] if tl > 0]
                if train_losses:
                    ax1.plot(epochs[:len(train_losses)], train_losses, 'b-s', 
                            label='Training Loss', linewidth=3, markersize=10, 
                            markeredgewidth=2, markeredgecolor='white', alpha=0.7)
        
        ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
        ax1.set_title('Loss Convergence (Train vs Validation)', fontsize=16, fontweight='bold', pad=15)
        ax1.legend(fontsize=12, loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        if epochs:
            ax1.set_xticks(epochs)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # ==================== 2. Epoch별 Performance Metrics (F1, Precision, Recall) ====================
        ax2 = fig.add_subplot(gs[0, 1])
        if epochs:
            if metrics.get('val_f1'):
                ax2.plot(epochs, metrics['val_f1'], 'g-o', label='F1 Score', 
                        linewidth=3, markersize=10, markeredgewidth=2, markeredgecolor='white')
            if metrics.get('val_precision'):
                ax2.plot(epochs, metrics['val_precision'], 'b-s', label='Precision', 
                        linewidth=3, markersize=10, markeredgewidth=2, markeredgecolor='white')
            if metrics.get('val_recall'):
                ax2.plot(epochs, metrics['val_recall'], 'orange', marker='^', label='Recall', 
                        linewidth=3, markersize=10, markeredgewidth=2, markeredgecolor='white')
        
        ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
        ax2.set_title('Performance Metrics (F1, Precision, Recall)', fontsize=16, fontweight='bold', pad=15)
        ax2.legend(fontsize=12, loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='--')
        if epochs:
            ax2.set_xticks(epochs)
            ax2.set_ylim([max(0, min([min(m) for m in [metrics.get('val_f1', [80]), 
                                                         metrics.get('val_precision', [80]), 
                                                         metrics.get('val_recall', [80])] if m]) - 5), 105])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # ==================== 3. Steps별 Training Loss (상세 수렴 과정) ====================
        ax3 = fig.add_subplot(gs[1, 0])
        if metrics.get('train_loss_history') and metrics.get('steps'):
            steps = metrics['steps']
            train_loss_hist = metrics['train_loss_history']
            
            # Raw data (반투명)
            ax3.plot(steps, train_loss_hist, 'b-', alpha=0.3, linewidth=1, label='Raw Loss')
            
            # 이동 평균 (강조)
            if len(train_loss_hist) > 20:
                window = min(100, max(20, len(train_loss_hist) // 20))
                smoothed = np.convolve(train_loss_hist, np.ones(window)/window, mode='valid')
                smooth_steps = steps[window-1:window-1+len(smoothed)]
                ax3.plot(smooth_steps, smoothed, 'r-', linewidth=3, 
                        label=f'Smoothed (window={window})')
        
        ax3.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Loss', fontsize=14, fontweight='bold')
        ax3.set_title('Training Loss Convergence (Step-by-Step)', fontsize=16, fontweight='bold', pad=15)
        ax3.legend(fontsize=12, loc='best', framealpha=0.9)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # ==================== 4. Epoch별 F1 Score vs Loss (듀얼 축) ====================
        ax4 = fig.add_subplot(gs[1, 1])
        ax4_twin = ax4.twinx()
        
        if epochs:
            # F1 Score (왼쪽 축)
            if metrics.get('val_f1'):
                line1 = ax4.plot(epochs, metrics['val_f1'], 'g-o', label='F1 Score', 
                                linewidth=3, markersize=10, markeredgewidth=2, markeredgecolor='white')
                ax4.set_ylabel('F1 Score (%)', fontsize=14, fontweight='bold', color='g')
                ax4.tick_params(axis='y', labelcolor='g')
            
            # Validation Loss (오른쪽 축)
            if metrics.get('val_loss'):
                line2 = ax4_twin.plot(epochs, metrics['val_loss'], 'r-s', label='Validation Loss', 
                                     linewidth=3, markersize=10, markeredgewidth=2, markeredgecolor='white')
                ax4_twin.set_ylabel('Loss', fontsize=14, fontweight='bold', color='r')
                ax4_twin.tick_params(axis='y', labelcolor='r')
        
        ax4.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax4.set_title('⚖️ F1 Score vs Loss Trade-off', fontsize=16, fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3, linestyle='--')
        if epochs:
            ax4.set_xticks(epochs)
        
        # 범례 통합
        if epochs and metrics.get('val_f1') and metrics.get('val_loss'):
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax4.legend(lines, labels, fontsize=12, loc='best', framealpha=0.9)
        
        ax4.spines['top'].set_visible(False)
        
        # 전체 타이틀
        fig.suptitle(f'Model Training Analysis: {model_name}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 저장
        model_name_safe = model_name.replace('/', '-')
        output_dir = Path(__file__).parent.parent.parent / "data" / "out" / "debug"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{model_name_safe}_training_curves.png"
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return output_file
        
    except Exception as e:
        print(f"[경고] 시각화 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def ner_train_legacy(
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
    레거시 NER 모델 훈련 API (하위 호환성)
    
    Note: 새 코드에서는 ner_train() 사용 권장
    """
    return ner_train(
        model_name=model_name,
        iterations=1,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_samples=num_train_samples,
        enable_visualization=True,
        debug=debug
    )

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


# ========== Main Function ==========

def main():
    """NER 시스템 테스트 및 사용 예제"""
    print("="*60)
    print("NER 시스템 테스트")
    print("="*60)
    
    # 엔티티 추출 테스트
    test_text = """저작물 저작재산권 양도 계약서

계약자: 김민수
전화번호: 010-1234-5678
이메일: minsu.kim@gmail.com
주소: 서울시 강남구 테헤란로 123

수탁기관: 한국콘텐츠진흥원
담당자: 박영희 부장
계약금: 5,000,000원"""
    
    entities = extract_entities_from_text(test_text, debug=True)
    
    print(f"\n✅ 추출된 엔티티 ({len(entities)}개):")
    for entity, label in entities:
        print(f"  - {entity} ({label})")
    
    # 훈련 상태 확인
    print(f"\n📊 현재 모델 상태:")
    status = get_training_status()
    print(f"  - 모델 존재: {status['model_exists']}")
    print(f"  - 경로: {status['model_path']}")
    if status.get('checkpoints', 0) > 0:
        print(f"  - 체크포인트: {status['checkpoints']}개")
    
    print("="*60)


if __name__ == "__main__":
    main()


