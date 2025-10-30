#!/usr/bin/env python3
"""
Token-level BIO + BERT-CRF NER Training System

주요 특징:
- BERT + CRF 모델 (시퀀스 레벨 최적화)
- Token-level BIO tagging (Character-level 대비 5배 빠름)
- OOV(Out-of-Vocabulary) 문제 해결 (컨텍스트 기반 인식)
"""

# ========== Imports ==========
import os
import json
import random
import math
import re
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from torch.utils.data import Dataset
from torchcrf import CRF
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 한글 폰트 설정
try:
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    korean_fonts = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic']
    
    font_found = False
    for font_name in korean_fonts:
        if font_name in available_fonts:
            plt.rcParams['font.family'] = font_name
            font_found = True
            print(f"✅ 한글 폰트 설정: {font_name}")
            break
    
    if not font_found:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("⚠️ 한글 폰트 없음. 기본 폰트 사용: DejaVu Sans")
    
    plt.rcParams['axes.unicode_minus'] = False
    
except Exception as e:
    print(f"⚠️ 폰트 설정 실패: {e}. 기본 폰트 사용")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False


# ========== Utility Functions ==========

def ensure_dir(path: Union[str, Path]) -> Path:
    """디렉토리가 존재하는지 확인하고 없으면 생성"""
    path_obj = Path(path) if isinstance(path, str) else path
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_device() -> torch.device:
    """사용 가능한 디바이스 반환 (CUDA > CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """JSON 파일 저장"""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """JSON 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ========== 모니터링 콜백 ==========

class MetricsMonitorCallback(TrainerCallback):
    """학습 중 메트릭 모니터링"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """로그 출력 시 호출"""
        if logs:
            # Epoch, Loss, Learning Rate 출력
            if 'loss' in logs:
                print(f"   [Step {state.global_step}] Loss={logs['loss']:.4f}", end='')
                if 'learning_rate' in logs:
                    print(f", LR={logs['learning_rate']:.2e}", end='')
                print()
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """평가 후 호출"""
        if metrics:
            print(f"\n{'='*60}")
            print(f"Epoch {int(metrics.get('epoch', 0))} 평가 결과:")
            print(f"{'='*60}")
            if 'eval_f1' in metrics:
                print(f"   F1 Score:  {metrics['eval_f1']:.4f}")
            if 'eval_precision' in metrics:
                print(f"   Precision: {metrics['eval_precision']:.4f}")
            if 'eval_recall' in metrics:
                print(f"   Recall:    {metrics['eval_recall']:.4f}")
            if 'eval_loss' in metrics:
                print(f"   Val Loss:  {metrics['eval_loss']:.4f}")
            if 'eval_tp' in metrics and 'eval_fp' in metrics and 'eval_fn' in metrics:
                print(f"   TP={metrics['eval_tp']}, FP={metrics['eval_fp']}, FN={metrics['eval_fn']}")
            print(f"{'='*60}\n")


# ========== 설정 ==========

@dataclass
class Config:
    """학습 설정"""
    model_name: str = "google-bert/bert-base-multilingual-cased"
    num_epochs: int = 300
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 1e-5
    max_length: int = 256
    dropout: float = 0.15
    warmup_ratio: float = 0.08
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    adaptive_grad_clip: bool = True
    agc_percentile: float = 10.0
    label_smoothing: float = 0.1
    ema_decay: float = 0.999
    layer_lr_decay: float = 0.95
    enable_loss_smoothing: bool = True


# 엔티티 타입 (23개)
ENTITY_TYPES = [
    "NAME", "PHONE", "ADDRESS", "DATE", "COMPANY", "EMAIL", "POSITION",
    "CONTRACT_TYPE", "MONEY", "PERIOD", "ID_NUM", "CONSENT_TYPE", "RIGHT_INFO",
    "PROJECT_NAME", "LAW_REFERENCE", "TITLE", "URL", "DESCRIPTION", "TYPE",
    "STATUS", "DEPARTMENT", "LANGUAGE", "QUANTITY"
]

# BIO 라벨 생성
BIO_LABELS = ["O"] + [f"{prefix}-{entity}" for entity in ENTITY_TYPES for prefix in ["B", "I"]]
LABEL_TO_ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

print(f"OK: {len(BIO_LABELS)}개 라벨 로드 완료")


# ========== BERT + CRF 모델 ==========

class BertCrfForNER(nn.Module):
    """
    BERT + CRF 모델
    
    구조:
    - BERT Encoder (768d)
    - Dropout
    - BiLSTM (256d, 3 layers)
    - Intermediate Layer (FFN + LayerNorm)
    - Classifier (BIO 태그)
    - CRF Layer (Viterbi decoding)
    """
    
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1, 
                 use_lstm: bool = True, lstm_hidden_dim: int = 256, lstm_layers: int = 3):
        super().__init__()
        self.num_labels = num_labels
        self.use_lstm = use_lstm
        
        # BERT Encoder
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config  # HuggingFace Trainer 호환성
        hidden_size = self.bert.config.hidden_size
        
        # Dropout (과적합 방지)
        self.dropout = nn.Dropout(dropout)
        
        # BiLSTM 레이어 (컨텍스트 양방향 학습)
        if use_lstm:
            self.lstm = nn.LSTM(
                hidden_size, 
                lstm_hidden_dim // 2,
                num_layers=lstm_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0
            )
            classifier_input_dim = lstm_hidden_dim
        else:
            self.lstm = None
            classifier_input_dim = hidden_size
        
        # Intermediate Layer
        self.intermediate = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim * 2, classifier_input_dim),
            nn.LayerNorm(classifier_input_dim)
        )
        
        # Classifier (BIO 태그)
        self.classifier = nn.Linear(classifier_input_dim, num_labels)
        
        # CRF Layer
        self.crf = CRF(num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        # BERT Encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # BiLSTM Layer
        if self.use_lstm and self.lstm is not None:
            lstm_output, _ = self.lstm(sequence_output)
            sequence_output = self.dropout(lstm_output)
        
        # Intermediate Layer
        sequence_output = self.intermediate(sequence_output)
        
        # Classification
        logits = self.classifier(sequence_output)
        
        # CRF: (seq_len, batch, num_labels) 형식 필요
        logits_crf = logits.transpose(0, 1)
        
        if labels is not None:
            # 학습: CRF loss
            labels_crf = labels.transpose(0, 1)
            mask_crf = attention_mask.transpose(0, 1).bool()
            
            # torchcrf.CRF.forward()는 log-likelihood 반환
            log_likelihood = self.crf(logits_crf, labels_crf, mask=mask_crf)
            
            # 배치 평균
            if isinstance(log_likelihood, torch.Tensor):
                if log_likelihood.dim() > 0:
                    log_likelihood = log_likelihood.mean()
            
            # CRF loss는 negative log-likelihood (항상 양수)
            loss = -log_likelihood
            
            return {'loss': loss, 'logits': logits}
        else:
            # 추론: CRF Viterbi decoding
            mask_crf = attention_mask.transpose(0, 1).bool()

            # torchcrf.CRF.decode(): List[List[int]] (Viterbi paths)
            best_paths = self.crf.decode(logits_crf, mask=mask_crf)
            
            # List[List[int]] → Tensor (batch, seq_len)
            batch_size = len(best_paths)
            max_len = logits.size(1)
            predictions = torch.zeros(batch_size, max_len, dtype=torch.long, device=logits.device)
            
            for i, path in enumerate(best_paths):
                predictions[i, :len(path)] = torch.tensor(path, device=logits.device, dtype=torch.long)
            
            return {'predictions': predictions, 'logits': logits}


# ========== 데이터 생성 (토큰 단위) ==========

# ========== Helper Functions ==========

def extract_entities_from_template(template: str) -> List[str]:
    """템플릿에서 엔티티 타입 추출"""
    return [match.group(1) for match in re.finditer(r'\{(\w+)\}', template)]


def generate_sample_from_template(template: str, entity_generators: Dict) -> Tuple[str, List[Tuple[str, str]]]:
    """템플릿으로부터 샘플 생성"""
    entities = {}
    for match in re.finditer(r'\{(\w+)\}', template):
        etype = match.group(1)
        if etype in entity_generators and etype not in entities:
            entities[etype] = entity_generators[etype]()
    
    # 템플릿 치환
    text = template
    for etype, value in entities.items():
        text = text.replace(f"{{{etype}}}", value)
    
    # 엔티티 리스트 생성
    entity_list = [(value, etype) for etype, value in entities.items() if value in text]
    
    return text, entity_list


def build_template_list(single_templates: Dict, dual_templates: List, multi_templates: List) -> List[Tuple[str, List[str]]]:
    """모든 템플릿을 (템플릿, 엔티티_리스트) 형태로 변환"""
    all_templates = []
    
    # 단일 엔티티 템플릿
    for entity_type, templates_list in single_templates.items():
        for tmpl in templates_list:
            all_templates.append((tmpl, [entity_type]))
    
    # 2개 및 다중 엔티티 템플릿
    for tmpl in dual_templates + multi_templates:
        entities = extract_entities_from_template(tmpl)
        all_templates.append((tmpl, entities))
    
    return all_templates


# ========== 한글 이름 생성 ==========

def generate_random_korean_name():
    """랜덤 한글 이름 생성 (OOV 대응용)"""
    surnames = [
        "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황", "안", "송", "류", "전",
        "유", "고", "문", "양", "손", "배", "백", "허", "남", "심", "노", "하", "곽", "성", "차", "주", "우", "구", "민", "홍",
        "진", "지", "엄", "원", "채", "천", "방", "공", "현", "함", "변", "염", "여", "추", "도", "소", "석", "선", "설", "마",
        "길", "연", "위", "표", "명", "기", "반", "라", "왕", "금", "옥", "육", "인", "맹", "제갈", "남궁", "독고", "사공", "선우", "황보"
    ]
    
    syllables = [
        "가", "각", "간", "갈", "감", "강", "개", "객", "건", "걸", "검", "겁", "게", "견", "결", "경", "계", "고", "곡", "곤", 
        "골", "공", "과", "관", "광", "교", "구", "국", "군", "굴", "궁", "권", "귀", "규", "균", "글", "금", "급", "기", "길",
        "나", "낙", "난", "날", "남", "납", "낭", "내", "냉", "너", "넉", "널", "네", "녀", "념", "녕", "노", "녹", "논", "놀",
        "농", "뇌", "누", "눈", "눌", "뉴", "늘", "능", "니", "님", "다", "단", "달", "담", "답", "당", "대", "댁", "더", "덕",
        "던", "덜", "덤", "덥", "데", "도", "독", "돈", "돌", "동", "돼", "두", "둔", "둘", "뒤", "드", "득", "들", "등", "디",
        "라", "락", "란", "랄", "람", "랑", "래", "랭", "량", "러", "럭", "런", "럴", "럼", "렁", "레", "려", "력", "련", "렬",
        "렴", "렵", "령", "례", "로", "록", "론", "롤", "롬", "롱", "뢰", "료", "룡", "루", "룩", "룬", "룰", "룸", "륙", "률",
        "륜", "르", "름", "릉", "리", "릭", "린", "릴", "림", "립", "마", "막", "만", "말", "맘", "망", "매", "맥", "맹", "머",
        "먹", "면", "멸", "명", "모", "목", "몬", "몰", "몽", "묘", "무", "묵", "문", "물", "미", "민", "밀", "바", "박", "반",
        "발", "밤", "방", "배", "백", "뱀", "버", "번", "벌", "범", "법", "벽", "변", "별", "병", "보", "복", "본", "봉", "부",
        "북", "분", "불", "비", "빈", "빌", "빔", "사", "삭", "산", "살", "삼", "상", "새", "색", "생", "서", "석", "선", "설",
        "섬", "섭", "성", "세", "소", "속", "손", "솔", "송", "쇄", "쇠", "수", "숙", "순", "술", "숨", "숭", "쉬", "스", "슬",
        "습", "승", "시", "식", "신", "실", "심", "십", "아", "악", "안", "알", "암", "압", "앙", "애", "액", "야", "약", "얀",
        "양", "어", "언", "얼", "엄", "업", "에", "여", "역", "연", "열", "염", "영", "예", "오", "옥", "온", "올", "옴", "옹",
        "와", "완", "왕", "외", "요", "욕", "용", "우", "욱", "운", "울", "움", "웅", "원", "월", "위", "유", "육", "윤", "율",
        "은", "을", "음", "읍", "응", "의", "이", "익", "인", "일", "임", "입", "자", "작", "잔", "잘", "잠", "장", "재", "쟁",
        "저", "적", "전", "절", "점", "접", "정", "제", "조", "족", "존", "졸", "종", "좌", "주", "죽", "준", "줄", "중", "증",
        "지", "직", "진", "질", "짐", "집", "징", "차", "착", "찬", "찰", "참", "창", "채", "책", "처", "척", "천", "철", "첨",
        "청", "체", "초", "촉", "총", "최", "추", "축", "춘", "출", "충", "취", "측", "치", "칙", "친", "칠", "침", "카", "쾌",
        "크", "큰", "클", "키", "타", "탁", "탄", "탈", "탑", "탕", "태", "택", "터", "테", "토", "통", "투", "특", "틀", "티",
        "파", "판", "팔", "패", "팽", "펴", "편", "평", "폐", "포", "폭", "표", "풍", "프", "플", "피", "필", "하", "학", "한",
        "할", "함", "합", "항", "해", "핵", "행", "향", "허", "헌", "험", "혁", "현", "혈", "협", "형", "혜", "호", "혹", "혼",
        "홀", "홍", "화", "확", "환", "활", "황", "회", "획", "횡", "효", "후", "훈", "휘", "휴", "흐", "흔", "흘", "흠", "흥",
        "희", "흰", "히"
    ]
    
    # 이름 길이: 2자 40%, 3자 55%, 4자 5%
    name_len_choice = random.random()
    if name_len_choice < 0.40:
        name_len = 2
    elif name_len_choice < 0.95:
        name_len = 3
    else:
        name_len = 4
    
    surname = random.choice(surnames)
    name_parts = [random.choice(syllables) for _ in range(name_len - 1)]
    
    return surname + ''.join(name_parts)


def load_bio_file(bio_file_path: str) -> List[Dict]:
    """
    BIO 형식 파일 로드
    
    Args:
        bio_file_path: BIO 파일 경로
    
    Returns:
        List[Dict]: [{'text': '...', 'entities': [('엔티티', '타입'), ...]}, ...]
    """
    samples = []
    current_tokens = []
    current_labels = []
    
    with open(bio_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 빈 줄 = 샘플 구분
            if not line:
                if current_tokens:
                    # 토큰 → 텍스트 복원
                    text = ' '.join(current_tokens)
                    
                    # BIO 태그 → 엔티티 추출
                    entities = []
                    current_entity_tokens = []
                    current_entity_type = None
                    
                    for token, label in zip(current_tokens, current_labels):
                        if label == 'O':
                            if current_entity_tokens:
                                entity_text = ' '.join(current_entity_tokens)
                                entities.append((entity_text, current_entity_type))
                                current_entity_tokens = []
                                current_entity_type = None
                        elif label.startswith('B-'):
                            if current_entity_tokens:
                                entity_text = ' '.join(current_entity_tokens)
                                entities.append((entity_text, current_entity_type))
                            current_entity_tokens = [token]
                            current_entity_type = label[2:]
                        elif label.startswith('I-'):
                            current_entity_tokens.append(token)
                    
                    # 마지막 엔티티 처리
                    if current_entity_tokens:
                        entity_text = ' '.join(current_entity_tokens)
                        entities.append((entity_text, current_entity_type))
                    
                    samples.append({
                        'text': text,
                        'entities': entities
                    })
                    
                    current_tokens = []
                    current_labels = []
            else:
                # 토큰 태그 분리
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    label = parts[1]
                    current_tokens.append(token)
                    current_labels.append(label)
    
    # 마지막 샘플 처리
    if current_tokens:
        text = ' '.join(current_tokens)
        entities = []
        current_entity_tokens = []
        current_entity_type = None
        
        for token, label in zip(current_tokens, current_labels):
            if label == 'O':
                if current_entity_tokens:
                    entity_text = ' '.join(current_entity_tokens)
                    entities.append((entity_text, current_entity_type))
                    current_entity_tokens = []
                    current_entity_type = None
            elif label.startswith('B-'):
                if current_entity_tokens:
                    entity_text = ' '.join(current_entity_tokens)
                    entities.append((entity_text, current_entity_type))
                current_entity_tokens = [token]
                current_entity_type = label[2:]
            elif label.startswith('I-'):
                current_entity_tokens.append(token)
        
        if current_entity_tokens:
            entity_text = ' '.join(current_entity_tokens)
            entities.append((entity_text, current_entity_type))
        
        samples.append({
            'text': text,
            'entities': entities
        })
    
    return samples


def generate_training_samples(num_samples: int = 5000, balanced: bool = True) -> List[Dict]:
    """
    학습 데이터 생성 (템플릿 기반) - 대규모 데이터 증강 지원 (최대 30만개)
    
    Args:
        num_samples: 생성할 샘플 수 (권장: 30만개)
        balanced: True면 모든 엔티티 타입을 골고루 포함
    """
    
    # 한국 성씨 확장 (200개)
    korean_surnames = [
        "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "오", "한", "신", "서", "권", "황", "안", "송", "전", "홍",
        "유", "고", "문", "양", "손", "배", "백", "허", "남", "심", "노", "하", "곽", "성", "차", "주", "우", "구", "신", "민",
        "진", "지", "엄", "원", "채", "천", "방", "공", "현", "함", "변", "염", "여", "추", "도", "소", "석", "선", "설", "마",
        "길", "연", "위", "표", "명", "기", "반", "라", "왕", "금", "옥", "육", "인", "맹", "제갈", "남궁", "독고", "사공", "선우", "황보"
    ]
    
    # 한국 이름 (500개 이상 조합)
    korean_name_parts = [
        "민수", "영희", "철수", "지훈", "수진", "하늘", "별", "달", "서연", "지우", "도윤", "예은", "시우", "서준", "하준",
        "지호", "준서", "민준", "지환", "윤서", "채원", "지안", "수빈", "현우", "예진", "서현", "유진", "지원", "하은", "태양",
        "소율", "건우", "시온", "태희", "나연", "정민", "승현", "민석", "재현", "다은", "예원", "지율", "우진", "하윤", "승우",
        "가은", "민지", "수아", "연우", "은서", "준혁", "예린", "서아", "지후", "유나", "수민", "태우", "시현", "주원", "다현",
        "민재", "서윤", "예나", "지민", "하린", "유찬", "소윤", "재윤", "시아", "준영", "채은", "지혁", "수현", "예준", "하영",
        "은지", "민규", "예서", "지성", "태윤", "소희", "준우", "시원", "지수", "하늘", "유빈", "재윤", "예림", "서진", "태민"
    ]
    
    # 엔티티 생성기 (대폭 확장)
    def random_phone():
        prefixes = ["010", "011", "016", "017", "018", "019"]
        return f"{random.choice(prefixes)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
    
    def random_company():
        # 500개 이상의 회사명 생성 가능
        company_types = ["주식회사", "(주)", "㈜", "유한회사", ""]
        company_names = [
            "삼성전자", "LG화학", "현대자동차", "SK하이닉스", "네이버", "카카오", "쿠팡", "배달의민족",
            "포스코", "한화", "롯데", "신세계", "한국전력", "KT", "SK텔레콤", "CJ제일제당",
            "현대중공업", "삼성SDI", "LG전자", "SK이노베이션", "한국타이어", "두산", "GS칼텍스", "S-Oil",
            "아모레퍼시픽", "LG생활건강", "오뚜기", "농심", "풀무원", "대상", "사조대림", "동원F&B",
            "넥슨", "엔씨소프트", "펄어비스", "크래프톤", "컴투스", "위메이드", "데브시스터즈", "스마일게이트",
            "셀트리온", "삼성바이오로직스", "녹십자", "유한양행", "종근당", "대웅제약", "한미약품", "일동제약",
            "현대건설", "삼성물산", "대림산업", "대우건설", "GS건설", "포스코건설", "롯데건설", "HDC현대산업개발"
        ]
        industries = ["테크놀로지", "바이오", "컨설팅", "솔루션즈", "시스템", "엔터프라이즈", "이노베이션", 
                     "크리에이티브", "디지털", "인터랙티브", "커뮤니케이션즈", "글로벌", "코리아", "인터내셔널"]
        
        if random.random() < 0.7:
            return random.choice(company_names)
        else:
            return f"{random.choice(company_types)}{random.choice(industries)}{random.randint(1,99)}"
    
    def random_date():
        year = random.randint(2015, 2025)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        formats = [
            f"{year}년 {month}월 {day}일",
            f"{year}.{month:02d}.{day:02d}",
            f"{year}-{month:02d}-{day:02d}",
            f"{year}/{month:02d}/{day:02d}",
            f"{month}월 {day}일",
            f"{year}년 {month}월",
            f"{month}/{day}",
            f"서기 {year}년 {month}월 {day}일"
        ]
        return random.choice(formats)
    
    def random_money():
        unit = random.choice(["원", "만원", "억원", "천원", "USD", "달러", "유로"])
        amount = random.randint(1, 999999)
        formats = [
            f"{amount:,}{unit}",
            f"금 {amount:,}{unit}",
            f"{amount}{unit} 정",
            f"￦{amount:,}",
            f"${amount:,}" if "USD" in unit or "달러" in unit else f"{amount:,}{unit}"
        ]
        return random.choice(formats)
    
    def random_address():
        # 전국 주요 도시/구/동 데이터
        cities = ["서울시", "부산시", "대구시", "인천시", "광주시", "대전시", "울산시", "세종시", 
                 "수원시", "성남시", "용인시", "고양시", "창원시", "청주시", "전주시", "천안시", "안산시", "안양시"]
        seoul_gu = ["강남구", "서초구", "송파구", "강동구", "마포구", "용산구", "종로구", "중구", "동작구",
                   "관악구", "서대문구", "은평구", "노원구", "도봉구", "강북구", "성북구", "중랑구", "광진구",
                   "성동구", "동대문구", "영등포구", "구로구", "금천구", "양천구", "강서구"]
        streets = ["테헤란로", "역삼로", "봉은사로", "선릉로", "논현로", "강남대로", "올림픽로", "한강대로",
                  "세종대로", "을지로", "종로", "퇴계로", "남대문로", "광화문로", "신촌로", "이대로", "연세로"]
        buildings = ["빌딩", "타워", "오피스", "센터", "플라자", "스퀘어", "파크"]
        
        if random.random() < 0.5:
            gu = random.choice(seoul_gu)
            street = random.choice(streets)
            return f"서울시 {gu} {street} {random.randint(1,999)}"
        else:
            city = random.choice(cities)
            return f"{city} {random.choice(streets)} {random.randint(1,999)} {random.choice(buildings)}"
    
    def random_email():
        domains = ["gmail.com", "naver.com", "kakao.com", "daum.net", "hanmail.net", "nate.com",
                  "outlook.com", "yahoo.com", "hotmail.com", "icloud.com", "company.co.kr", "example.com"]
        prefixes = ["user", "admin", "contact", "info", "support", "sales", "help", "service",
                   "team", "office", "manager", "director", "ceo", "cto", "hr", "pr"]
        names = ["kim", "lee", "park", "choi", "jung", "kang", "cho", "yoon", "jang", "lim"]
        
        if random.random() < 0.5:
            return f"{random.choice(prefixes)}{random.randint(1,9999)}@{random.choice(domains)}"
        else:
            return f"{random.choice(names)}.{random.choice(prefixes)}@{random.choice(domains)}"
    
    def random_position():
        positions = [
            "대표이사", "부사장", "전무", "상무", "이사", "부장", "차장", "과장", "대리", "주임", "사원",
            "팀장", "파트장", "실장", "본부장", "센터장", "지점장", "영업부장", "마케팅이사", "기술이사",
            "연구원", "선임연구원", "수석연구원", "책임연구원", "매니저", "시니어매니저", "프로젝트매니저",
            "컨설턴트", "시니어컨설턴트", "전문위원", "수석전문위원", "기술사", "변리사", "회계사", "세무사"
        ]
        return random.choice(positions)
    
    def random_contract_type():
        contracts = [
            "저작권 양도 계약", "이용허락 계약", "공동저작 계약", "출판권설정 계약", "배타적발행권 계약",
            "독점 라이선스", "비독점 라이선스", "전송권 계약", "배포권 계약", "2차저작물 작성권 계약",
            "실시권 설정", "통상실시권", "전용실시권", "독점적 통상실시권", "크로스 라이선스",
            "양해각서(MOU)", "업무협약(MOU)", "비밀유지계약(NDA)", "용역계약", "위탁계약"
        ]
        return random.choice(contracts)
    
    def random_period():
        units = ["일", "개월", "년", "주"]
        numbers = random.randint(1, 60)
        formats = [
            f"{numbers}{random.choice(units)}",
            f"{numbers}{random.choice(units)}간",
            f"계약일로부터 {numbers}{random.choice(units)}",
            f"{numbers}{random.choice(units)} 이내"
        ]
        return random.choice(formats)
    
    def random_id_num():
        # 주민등록번호 형식 (뒷자리 마스킹)
        front = f"{random.randint(0,99):02d}{random.randint(1,12):02d}{random.randint(1,28):02d}"
        formats = [
            f"{front}-*******",
            f"{front}-{random.randint(1,4)}******",
            f"주민등록번호: {front}-*******",
            f"사업자등록번호: {random.randint(100,999)}-{random.randint(10,99)}-{random.randint(10000,99999)}"
        ]
        return random.choice(formats)
    
    def random_consent_type():
        consents = [
            "개인정보 수집 및 이용 동의", "개인정보 제3자 제공 동의", "마케팅 정보 수신 동의",
            "저작권 이용 동의", "초상권 이용 동의", "정보 제공 동의", "서비스 이용 동의",
            "위치정보 이용 동의", "전자금융거래 이용 동의", "신용정보 조회 동의",
            "공공저작물 자유이용허락 동의", "CCL 라이선스 동의", "오픈소스 라이선스 동의"
        ]
        return random.choice(consents)
    
    def random_right_info():
        rights = [
            "저작인격권", "저작재산권", "2차적저작물작성권", "복제권", "배포권", "전송권", "공연권", 
            "전시권", "대여권", "방송권", "공중송신권", "디지털음성송신권", "실연자의 권리",
            "음반제작자의 권리", "방송사업자의 권리", "데이터베이스제작자의 권리", "특허권",
            "상표권", "디자인권", "실용신안권", "영업비밀", "초상권", "퍼블리시티권", "성명권"
        ]
        return random.choice(rights)
    
    def random_project():
        projects = [
            "디지털 아카이브 구축 사업", "저작권 관리 시스템 개발", "메타데이터 표준화 프로젝트",
            "문화재 디지털화 사업", "교육콘텐츠 개발 프로젝트", "AI 학습데이터 구축",
            "스마트시티 플랫폼 구축", "블록체인 기반 저작권 관리", "디지털 트윈 구축",
            "빅데이터 분석 플랫폼", "클라우드 마이그레이션", "보안관제 시스템 구축",
            "전자정부 표준프레임워크", "공공데이터 개방", "K-디지털 트레이닝",
            "문화예술 진흥 사업", "연구개발(R&D) 과제", "정보화 전략계획(ISP)"
        ]
        return random.choice(projects)
    
    def random_law():
        laws = [
            "저작권법 제10조", "저작권법 제45조", "저작권법 제25조", "저작권법 제136조",
            "민법 제114조", "민법 제750조", "민법 제393조", "민법 제398조",
            "개인정보보호법 제15조", "개인정보보호법 제17조", "개인정보보호법 제24조",
            "정보통신망법 제22조", "정보통신망법 제24조", "정보통신망법 제44조의7",
            "특허법 제2조", "상표법 제2조", "디자인보호법 제2조", "부정경쟁방지법 제2조",
            "공공기록물관리법 제3조", "국가정보화기본법 제5조", "전자문서법 제4조"
        ]
        return random.choice(laws)
    
    def random_title():
        titles = [
            "저작물 이용 계약서", "저작권 양도 계약서", "공동저작 계약서", "용역 계약서", "라이선스 계약서",
            "출판권 설정 계약서", "배타적 발행권 설정 계약서", "전송권 이용 계약서", "위탁 개발 계약서",
            "비밀유지 계약서", "업무협약서", "공동연구 계약서", "기술이전 계약서", "특허 실시 계약서",
            "상표권 사용 계약서", "프랜차이즈 계약서", "판매 대행 계약서", "유통 계약서", "제작 계약서",
            "공급 계약서", "구매 계약서", "임대차 계약서", "고용 계약서", "도급 계약서"
        ]
        return random.choice(titles)
    
    def random_url():
        domains = ["example.com", "sample.org", "test.net", "demo.co.kr", "site.com",
                  "portal.go.kr", "company.co.kr", "service.net", "platform.io", "app.com"]
        paths = ["page", "document", "file", "content", "resource", "api", "service", "data"]
        return f"https://www.{random.choice(domains)}/{random.choice(paths)}/{random.randint(1,9999)}"
    
    def random_description():
        descriptions = [
            "본 계약은 저작물 이용에 관한 사항입니다", "양 당사자 간 권리관계를 명확히 합니다",
            "저작권 귀속 및 이용범위를 정합니다", "계약 이행 조건을 규정합니다",
            "비밀유지 의무를 준수합니다", "손해배상 책임에 관한 사항입니다",
            "계약기간 및 갱신 조건을 명시합니다", "대가 지급 방법을 규정합니다",
            "해지 및 종료 사유를 정합니다", "분쟁 해결 방법을 명시합니다",
            "지적재산권의 귀속을 명확히 합니다", "보증 및 면책 조항입니다",
            "준거법 및 관할법원을 정합니다", "계약서 해석 기준을 명시합니다"
        ]
        return random.choice(descriptions)
    
    def random_type():
        types = [
            "문서", "이미지", "영상", "음원", "소프트웨어", "데이터베이스", "웹사이트", "애플리케이션",
            "사진저작물", "어문저작물", "미술저작물", "음악저작물", "영상저작물", "건축저작물",
            "도형저작물", "편집저작물", "2차적저작물", "공동저작물", "업무상저작물",
            "컴퓨터프로그램저작물", "데이터베이스", "멀티미디어", "게임", "캐릭터", "폰트"
        ]
        return random.choice(types)
    
    def random_status():
        statuses = [
            "진행중", "완료", "검토중", "승인대기", "반려", "수정요청", "보류", "취소",
            "계약체결", "이행중", "종료", "갱신", "해지", "분쟁중", "중재중", "소송중",
            "일시중단", "재개", "연장", "변경", "추가협의", "합의완료"
        ]
        return random.choice(statuses)
    
    def random_department():
        departments = [
            "법무팀", "지식재산팀", "경영지원팀", "기획팀", "연구개발팀", "마케팅팀", "인사팀",
            "재무팀", "회계팀", "영업팀", "생산팀", "품질관리팀", "구매팀", "물류팀", "CS팀",
            "IT팀", "보안팀", "홍보팀", "IR팀", "전략기획실", "감사실", "준법감시실",
            "R&D센터", "디자인센터", "고객지원센터", "콜센터", "물류센터"
        ]
        return random.choice(departments)
    
    def random_language():
        languages = [
            "한국어", "영어", "일본어", "중국어", "스페인어", "불어", "독어", "러시아어",
            "아랍어", "포르투갈어", "이탈리아어", "베트남어", "태국어", "인도네시아어",
            "한글", "영문", "일문", "중문", "다국어", "이중언어"
        ]
        return random.choice(languages)
    
    def random_quantity():
        units = ["개", "건", "부", "권", "장", "매", "점", "식", "종", "세트", "박스", "팩"]
        amount = random.randint(1, 9999)
        return f"{amount:,}{random.choice(units)}"
    
    def random_name():
        """다양한 한국인 이름 생성 (OOV 대응 강화)"""
        # 50% 확률로 완전 랜덤 이름 생성 (처음 보는 이름 패턴 학습)
        if random.random() < 0.5:
            return generate_random_korean_name()
        else:
            surname = random.choice(korean_surnames)
            name = random.choice(korean_name_parts)
            return f"{surname}{name}"
    
    # 대규모 템플릿 라이브러리 (카테고리별 분류)
    # 총 1000+ 템플릿 (단일 엔티티: 300, 2개 엔티티: 400, 3개 엔티티: 200, 다중 엔티티: 100)
    
    # === 1. 단일 엔티티 템플릿 (300개) ===
    single_entity_templates = {
        "NAME": [
            # 기본 패턴 (20개)
            "{NAME} 본인이 서명합니다.",
            "당사자: {NAME}",
            "작성자는 {NAME}입니다.",
            "{NAME}이(가) 권리를 행사합니다.",
            "저작자 {NAME}은",
            "{NAME} 귀하",
            "본인 {NAME}은 동의합니다.",
            "신청인: {NAME}",
            "{NAME}님의 요청",
            "담당자 {NAME}",
            "{NAME} 대표님",
            "계약당사자 {NAME}",
            "{NAME}의 서명으로",
            "권리자 {NAME}은",
            "{NAME} 선생님",
            "{NAME}씨",
            "작가 {NAME}",
            "감독 {NAME}",
            "연출 {NAME}",
            "제작자 {NAME}",
            # 문맥 강화 패턴 (30개 추가) - 처음 보는 이름도 문맥으로 인식
            "{NAME} 님께서",
            "{NAME}님이",
            "{NAME}씨가",
            "{NAME}이",
            "{NAME}가",
            "{NAME}은",
            "{NAME}는",
            "{NAME}을",
            "{NAME}를",
            "{NAME}과",
            "{NAME}와",
            "{NAME}에게",
            "{NAME}으로부터",
            "성명: {NAME}",
            "이름: {NAME}",
            "성함: {NAME}",
            "저작자명: {NAME}",
            "작성자명: {NAME}",
            "신청자: {NAME}",
            "제출자: {NAME}",
            "확인자: {NAME}",
            "승인자: {NAME}",
            "책임자: {NAME}",
            "{NAME}님은",
            "{NAME}님의",
            "{NAME}씨는",
            "{NAME}씨의",
            "{NAME}이라는 사람",
            "{NAME}라는 분",
            "{NAME} 작가님",
        ],
        "PHONE": [
            "연락처: {PHONE}",
            "전화번호 {PHONE}",
            "{PHONE}로 연락주세요.",
            "휴대전화: {PHONE}",
            "문의처 {PHONE}",
            "{PHONE}로 문의",
            "대표번호 {PHONE}",
            "{PHONE} 담당자",
            "전화: {PHONE}",
            "팩스: {PHONE}",
            "긴급연락처 {PHONE}",
            "{PHONE}으로 전화",
            "연락가능번호 {PHONE}",
            "휴대폰 {PHONE}",
            "{PHONE}로 회신",
        ],
        "COMPANY": [
            "{COMPANY}에서 발행",
            "주식회사 {COMPANY}",
            "{COMPANY} 법인",
            "회사명: {COMPANY}",
            "{COMPANY}와의 계약",
            "{COMPANY} 소속",
            "{COMPANY} 담당",
            "제작사 {COMPANY}",
            "발주처 {COMPANY}",
            "수주처 {COMPANY}",
            "{COMPANY}가 제공하는",
            "{COMPANY} 대표",
            "기업 {COMPANY}",
            "{COMPANY}의 서비스",
            "상호: {COMPANY}",
            "{COMPANY} 본사",
            "{COMPANY} 지사",
            "법인명 {COMPANY}",
            "{COMPANY}측",
            "{COMPANY} 브랜드",
        ],
        "ADDRESS": [
            "주소: {ADDRESS}",
            "소재지 {ADDRESS}",
            "{ADDRESS}에 위치",
            "사업장 {ADDRESS}",
            "본사 {ADDRESS}",
            "{ADDRESS} 소재",
            "배송지 {ADDRESS}",
            "거주지 {ADDRESS}",
            "{ADDRESS}로 발송",
            "주소지 {ADDRESS}",
            "등기주소 {ADDRESS}",
            "{ADDRESS} 사무실",
            "우편주소 {ADDRESS}",
            "{ADDRESS} 건물",
            "지번 {ADDRESS}",
        ],
        "DATE": [
            "계약일: {DATE}",
            "{DATE}에 체결",
            "일자 {DATE}",
            "{DATE} 기준",
            "작성일 {DATE}",
            "{DATE}부터",
            "{DATE}까지",
            "기간 {DATE}",
            "{DATE} 시작",
            "{DATE} 종료",
            "발효일 {DATE}",
            "{DATE} 현재",
            "체결일 {DATE}",
            "{DATE} 합의",
            "날짜: {DATE}",
            "{DATE}에 발생",
            "적용일 {DATE}",
            "{DATE} 시행",
            "갱신일 {DATE}",
            "{DATE} 연장",
        ],
        "EMAIL": [
            "이메일: {EMAIL}",
            "연락처 {EMAIL}",
            "{EMAIL}로 송부",
            "문의 {EMAIL}",
            "메일주소 {EMAIL}",
            "{EMAIL}로 전송",
            "담당자 이메일 {EMAIL}",
            "{EMAIL}로 회신",
            "전자우편 {EMAIL}",
            "{EMAIL} 수신",
            "이메일주소: {EMAIL}",
            "{EMAIL}로 통보",
            "메일 {EMAIL}",
            "{EMAIL}로 발송",
            "연락메일 {EMAIL}",
        ],
        "MONEY": [
            "금액: {MONEY}",
            "대가 {MONEY}",
            "{MONEY}를 지급",
            "보수 {MONEY}",
            "급여 {MONEY}",
            "{MONEY} 지불",
            "계약금 {MONEY}",
            "잔금 {MONEY}",
            "{MONEY}의 보상",
            "총액 {MONEY}",
            "{MONEY} 상당",
            "합계 {MONEY}",
            "{MONEY}를 수령",
            "비용 {MONEY}",
            "{MONEY} 산정",
            "금 {MONEY}",
            "{MONEY} 정",
            "가격 {MONEY}",
            "보상금 {MONEY}",
            "{MONEY}의 대가",
        ],
        "POSITION": [
            # 직위 (50개 - 제로샷 강화)
            "직위: {POSITION}",
            "{POSITION} 담당",
            "직책 {POSITION}",
            "{POSITION}으로 근무",
            "{POSITION} 임명",
            "직급 {POSITION}",
            "{POSITION} 역할",
            "{POSITION}이(가)",
            "{POSITION} 수행",
            "{POSITION} 보임",
            "소속 {POSITION}",
            "{POSITION} 발령",
            "직함 {POSITION}",
            "{POSITION} 임용",
            "{POSITION} 임직원",
            "{POSITION}의",
            "직위는 {POSITION}",
            "{POSITION}을",
            "직책: {POSITION}",
            "{POSITION}를",
            "직급은 {POSITION}",
            "{POSITION}이",
            "직함: {POSITION}",
            "{POSITION}가",
            "직위: {POSITION}입니다",
            "{POSITION}에",
            "직책은 {POSITION}",
            "{POSITION}은",
            "직급: {POSITION}이며",
            "{POSITION}와",
            "직위정보: {POSITION}",
            "{POSITION}로",
            "직함은 {POSITION}",
            "{POSITION}으로",
            "직책정보 {POSITION}",
            "{POSITION}과",
            "직위는 {POSITION}이며",
            "{POSITION}에서",
            "직급정보: {POSITION}",
            "{POSITION}를 맡은",
            "직함정보 {POSITION}",
            "{POSITION}에 따라",
            "직위: {POSITION}로",
            "{POSITION}에 의거",
            "직책은 {POSITION}이고",
            "{POSITION}로서",
            "직급: {POSITION}으로",
            "{POSITION}에 관한",
            "직함은 {POSITION}이며",
            "{POSITION}이라는",
        ],
        "RIGHT_INFO": [
            # 권리정보 (50개 - 제로샷 강화)
            "권리: {RIGHT_INFO}",
            "{RIGHT_INFO} 양도",
            "권리는 {RIGHT_INFO}",
            "{RIGHT_INFO}의",
            "{RIGHT_INFO} 이전",
            "권리정보: {RIGHT_INFO}",
            "{RIGHT_INFO}을",
            "{RIGHT_INFO} 보유",
            "권리내용 {RIGHT_INFO}",
            "{RIGHT_INFO}를",
            "{RIGHT_INFO} 행사",
            "권리: {RIGHT_INFO}입니다",
            "{RIGHT_INFO}이",
            "{RIGHT_INFO} 귀속",
            "권리형태: {RIGHT_INFO}",
            "{RIGHT_INFO}가",
            "{RIGHT_INFO} 소유",
            "권리종류 {RIGHT_INFO}",
            "{RIGHT_INFO}에",
            "{RIGHT_INFO} 양수",
            "권리는 {RIGHT_INFO}이며",
            "{RIGHT_INFO}은",
            "{RIGHT_INFO} 취득",
            "권리항목: {RIGHT_INFO}",
            "{RIGHT_INFO}와",
            "{RIGHT_INFO} 포기",
            "권리사항 {RIGHT_INFO}",
            "{RIGHT_INFO}로",
            "{RIGHT_INFO} 제한",
            "권리정보는 {RIGHT_INFO}",
            "{RIGHT_INFO}으로",
            "{RIGHT_INFO} 등록",
            "권리내용: {RIGHT_INFO}",
            "{RIGHT_INFO}과",
            "{RIGHT_INFO} 설정",
            "권리: {RIGHT_INFO}로서",
            "{RIGHT_INFO}에서",
            "{RIGHT_INFO} 확인",
            "권리형태 {RIGHT_INFO}",
            "{RIGHT_INFO}를 통해",
            "{RIGHT_INFO} 증명",
            "권리종류: {RIGHT_INFO}",
            "{RIGHT_INFO}에 따라",
            "{RIGHT_INFO} 명시",
            "권리항목 {RIGHT_INFO}이며",
            "{RIGHT_INFO}에 의거",
            "{RIGHT_INFO} 관계",
            "권리사항: {RIGHT_INFO}",
            "{RIGHT_INFO}로서",
            "{RIGHT_INFO} 발생",
        ],
        "PROJECT_NAME": [
            # 프로젝트명 (50개 - 제로샷 강화)
            "프로젝트: {PROJECT_NAME}",
            "{PROJECT_NAME} 수행",
            "과제명: {PROJECT_NAME}",
            "{PROJECT_NAME}의",
            "{PROJECT_NAME} 진행",
            "사업명 {PROJECT_NAME}",
            "{PROJECT_NAME}을",
            "프로젝트는 {PROJECT_NAME}",
            "{PROJECT_NAME}를",
            "과제: {PROJECT_NAME}",
            "{PROJECT_NAME}이",
            "사업 {PROJECT_NAME}",
            "{PROJECT_NAME}가",
            "프로젝트명: {PROJECT_NAME}",
            "{PROJECT_NAME}에",
            "과제명은 {PROJECT_NAME}",
            "{PROJECT_NAME}은",
            "사업명: {PROJECT_NAME}",
            "{PROJECT_NAME}와",
            "프로젝트 {PROJECT_NAME}로",
            "{PROJECT_NAME}로",
            "과제: {PROJECT_NAME}이며",
            "{PROJECT_NAME}으로",
            "사업명은 {PROJECT_NAME}",
            "{PROJECT_NAME}과",
            "프로젝트명 {PROJECT_NAME}",
            "{PROJECT_NAME}에서",
            "과제명 {PROJECT_NAME}",
            "{PROJECT_NAME}를 통해",
            "사업: {PROJECT_NAME}",
            "{PROJECT_NAME}에 따라",
            "프로젝트는 {PROJECT_NAME}이며",
            "{PROJECT_NAME}에 의거",
            "과제 {PROJECT_NAME}으로",
            "{PROJECT_NAME}로서",
            "사업명: {PROJECT_NAME}이고",
            "{PROJECT_NAME}에 관한",
            "프로젝트정보: {PROJECT_NAME}",
            "{PROJECT_NAME}이라는",
            "과제정보 {PROJECT_NAME}",
            "{PROJECT_NAME}라고",
            "사업정보: {PROJECT_NAME}",
            "{PROJECT_NAME}을 수행",
            "프로젝트 제목: {PROJECT_NAME}",
            "{PROJECT_NAME}를 진행",
            "과제제목 {PROJECT_NAME}",
            "{PROJECT_NAME}에 참여",
            "사업제목: {PROJECT_NAME}",
            "{PROJECT_NAME}으로 명명",
            "프로젝트 {PROJECT_NAME}이며",
        ],
        "LAW_REFERENCE": [
            # 법령참조 (50개 - 제로샷 강화)
            "법령: {LAW_REFERENCE}",
            "{LAW_REFERENCE}에 따라",
            "근거법령 {LAW_REFERENCE}",
            "{LAW_REFERENCE}의",
            "{LAW_REFERENCE}에 의거",
            "법: {LAW_REFERENCE}",
            "{LAW_REFERENCE}을",
            "법령은 {LAW_REFERENCE}",
            "{LAW_REFERENCE}를",
            "관련법: {LAW_REFERENCE}",
            "{LAW_REFERENCE}이",
            "{LAW_REFERENCE} 준수",
            "법령: {LAW_REFERENCE}입니다",
            "{LAW_REFERENCE}가",
            "{LAW_REFERENCE} 적용",
            "근거: {LAW_REFERENCE}",
            "{LAW_REFERENCE}에",
            "{LAW_REFERENCE} 규정",
            "법령정보: {LAW_REFERENCE}",
            "{LAW_REFERENCE}은",
            "{LAW_REFERENCE} 위반",
            "관련법령 {LAW_REFERENCE}",
            "{LAW_REFERENCE}와",
            "{LAW_REFERENCE} 조항",
            "법: {LAW_REFERENCE}이며",
            "{LAW_REFERENCE}로",
            "{LAW_REFERENCE} 근거",
            "법령근거: {LAW_REFERENCE}",
            "{LAW_REFERENCE}으로",
            "{LAW_REFERENCE} 기준",
            "관련법: {LAW_REFERENCE}로",
            "{LAW_REFERENCE}과",
            "{LAW_REFERENCE} 해석",
            "법령은 {LAW_REFERENCE}이며",
            "{LAW_REFERENCE}에서",
            "{LAW_REFERENCE} 조문",
            "근거법: {LAW_REFERENCE}",
            "{LAW_REFERENCE}를 통해",
            "{LAW_REFERENCE} 명시",
            "법령항목: {LAW_REFERENCE}",
            "{LAW_REFERENCE}에 따른",
            "{LAW_REFERENCE} 이행",
            "관련법령: {LAW_REFERENCE}이고",
            "{LAW_REFERENCE}에 의한",
            "{LAW_REFERENCE} 제정",
            "법령: {LAW_REFERENCE}로서",
            "{LAW_REFERENCE}로서",
            "{LAW_REFERENCE} 개정",
            "근거법령은 {LAW_REFERENCE}",
            "{LAW_REFERENCE}에 관한",
        ],
        "CONTRACT_TYPE": [
            # 계약 유형 (50개 - 제로샷 강화)
            "계약형태: {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}을(를) 체결",
            "본 {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}에 따라",
            "{CONTRACT_TYPE} 내용",
            "{CONTRACT_TYPE} 조건",
            "{CONTRACT_TYPE}에 의거",
            "{CONTRACT_TYPE} 이행",
            "{CONTRACT_TYPE}의 목적",
            "{CONTRACT_TYPE} 체결",
            "계약종류: {CONTRACT_TYPE}",
            "{CONTRACT_TYPE} 합의",
            "{CONTRACT_TYPE} 방식",
            "{CONTRACT_TYPE}로 진행",
            "{CONTRACT_TYPE} 절차",
            "{CONTRACT_TYPE}의",
            "계약: {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}을",
            "계약형태는 {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}를",
            "계약종류 {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}이",
            "계약: {CONTRACT_TYPE}입니다",
            "{CONTRACT_TYPE}가",
            "계약방식: {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}에",
            "계약유형 {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}은",
            "계약은 {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}와",
            "계약형식: {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}로",
            "계약종류는 {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}으로",
            "계약: {CONTRACT_TYPE}로서",
            "{CONTRACT_TYPE}과",
            "계약형태 {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}에서",
            "계약방식 {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}를 통해",
            "계약유형: {CONTRACT_TYPE}이며",
            "{CONTRACT_TYPE}에 의한",
            "계약형식 {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}로서",
            "계약은 {CONTRACT_TYPE}이고",
            "{CONTRACT_TYPE}에 관한",
            "계약종류: {CONTRACT_TYPE}로",
            "{CONTRACT_TYPE}이라는",
            "계약방식은 {CONTRACT_TYPE}",
            "{CONTRACT_TYPE}라고",
        ],
        "PERIOD": [
            # 기간 (50개 - 제로샷 강화)
            "기간: {PERIOD}",
            "{PERIOD} 동안",
            "{PERIOD}간",
            "계약기간 {PERIOD}",
            "{PERIOD} 유효",
            "존속기간 {PERIOD}",
            "{PERIOD} 기한",
            "{PERIOD} 내",
            "유효기간 {PERIOD}",
            "{PERIOD} 만료",
            "기한 {PERIOD}",
            "{PERIOD} 적용",
            "이용기간 {PERIOD}",
            "{PERIOD}의 기간",
            "{PERIOD} 효력",
            "{PERIOD}의",
            "기간은 {PERIOD}",
            "{PERIOD}을",
            "기한: {PERIOD}",
            "{PERIOD}를",
            "계약기간: {PERIOD}",
            "{PERIOD}이",
            "존속기간: {PERIOD}",
            "{PERIOD}가",
            "유효기간: {PERIOD}",
            "{PERIOD}에",
            "기간: {PERIOD}입니다",
            "{PERIOD}은",
            "이용기간: {PERIOD}",
            "{PERIOD}와",
            "기한은 {PERIOD}",
            "{PERIOD}로",
            "계약기간은 {PERIOD}",
            "{PERIOD}으로",
            "존속기간은 {PERIOD}",
            "{PERIOD}과",
            "유효기간은 {PERIOD}",
            "{PERIOD}에서",
            "기간정보: {PERIOD}",
            "{PERIOD}를 통해",
            "이용기간은 {PERIOD}",
            "{PERIOD}에 따라",
            "기한: {PERIOD}이며",
            "{PERIOD}에 의거",
            "계약기간: {PERIOD}로",
            "{PERIOD}로서",
            "존속기간: {PERIOD}이고",
            "{PERIOD}에 관한",
            "유효기간: {PERIOD}로",
            "{PERIOD}이라는",
        ],
        "ID_NUM": [
            # 주민등록번호/사업자번호 등 (50개 - 제로샷 강화)
            "주민등록번호: {ID_NUM}",
            "사업자등록번호 {ID_NUM}",
            "{ID_NUM} 본인",
            "등록번호 {ID_NUM}",
            "식별번호 {ID_NUM}",
            "{ID_NUM}로 확인",
            "번호: {ID_NUM}",
            "{ID_NUM} 해당",
            "고유번호 {ID_NUM}",
            "{ID_NUM} 등록",
            "주민번호 {ID_NUM}",
            "{ID_NUM} 신원",
            "ID: {ID_NUM}",
            "{ID_NUM} 인증",
            "법인번호 {ID_NUM}",
            "{ID_NUM}으로",
            "식별 {ID_NUM}",
            "{ID_NUM} 조회",
            "증번호 {ID_NUM}",
            "{ID_NUM} 기재",
            "주번 {ID_NUM}",
            "{ID_NUM}이",
            "등록 {ID_NUM}",
            "{ID_NUM}의",
            "고유 {ID_NUM}",
            "{ID_NUM}을",
            "번호는 {ID_NUM}",
            "{ID_NUM}로서",
            "주민등록 {ID_NUM}",
            "{ID_NUM} 번호",
            "사업자 {ID_NUM}",
            "{ID_NUM} 보유",
            "개인번호 {ID_NUM}",
            "{ID_NUM} 대조",
            "주민 {ID_NUM}",
            "{ID_NUM} 일치",
            "등록증 {ID_NUM}",
            "{ID_NUM} 부여",
            "신원번호 {ID_NUM}",
            "{ID_NUM} 명의",
            "식별코드 {ID_NUM}",
            "{ID_NUM} 소지",
            "법인 {ID_NUM}",
            "{ID_NUM} 기준",
            "고유ID {ID_NUM}",
            "{ID_NUM} 소유",
            "번호 {ID_NUM}이며",
            "{ID_NUM} 확인서",
            "주민증 {ID_NUM}",
            "{ID_NUM} 표기",
        ],
        "CONSENT_TYPE": [
            # 동의 유형 (50개 - 제로샷 강화)
            "동의유형: {CONSENT_TYPE}",
            "{CONSENT_TYPE}에 동의",
            "본 {CONSENT_TYPE}",
            "{CONSENT_TYPE} 획득",
            "{CONSENT_TYPE}을 체결",
            "동의: {CONSENT_TYPE}",
            "{CONSENT_TYPE} 완료",
            "{CONSENT_TYPE}에 서명",
            "{CONSENT_TYPE} 처리",
            "{CONSENT_TYPE}의 내용",
            "{CONSENT_TYPE} 승인",
            "{CONSENT_TYPE}에 따라",
            "동의형태 {CONSENT_TYPE}",
            "{CONSENT_TYPE} 제출",
            "{CONSENT_TYPE}을 작성",
            "{CONSENT_TYPE} 합의",
            "{CONSENT_TYPE}로",
            "동의종류: {CONSENT_TYPE}",
            "{CONSENT_TYPE} 이행",
            "{CONSENT_TYPE}에 의거",
            "{CONSENT_TYPE} 확인",
            "{CONSENT_TYPE}을",
            "동의서: {CONSENT_TYPE}",
            "{CONSENT_TYPE} 발급",
            "{CONSENT_TYPE}에",
            "{CONSENT_TYPE} 신청",
            "{CONSENT_TYPE}의",
            "동의절차 {CONSENT_TYPE}",
            "{CONSENT_TYPE} 접수",
            "{CONSENT_TYPE}을 통해",
            "{CONSENT_TYPE} 필요",
            "{CONSENT_TYPE}로서",
            "동의방식 {CONSENT_TYPE}",
            "{CONSENT_TYPE} 요청",
            "{CONSENT_TYPE}에 대한",
            "{CONSENT_TYPE} 수령",
            "{CONSENT_TYPE}은",
            "동의내용 {CONSENT_TYPE}",
            "{CONSENT_TYPE} 등록",
            "{CONSENT_TYPE}를 득함",
            "{CONSENT_TYPE} 진행",
            "{CONSENT_TYPE}가",
            "동의문서 {CONSENT_TYPE}",
            "{CONSENT_TYPE} 기재",
            "{CONSENT_TYPE}으로서",
            "{CONSENT_TYPE} 보관",
            "{CONSENT_TYPE}를",
            "동의조건 {CONSENT_TYPE}",
            "{CONSENT_TYPE} 명시",
            "{CONSENT_TYPE}와",
        ],
        "TITLE": [
            # 작품명/문서명 (50개 - 제로샷 강화)
            "제목: {TITLE}",
            "작품명 {TITLE}",
            "{TITLE}의 저작권",
            "문서명: {TITLE}",
            "{TITLE} 계약서",
            "본 {TITLE}",
            "{TITLE}에 대한",
            "작품 {TITLE}",
            "{TITLE}을",
            "제목은 {TITLE}",
            "{TITLE}의",
            "계약서명: {TITLE}",
            "{TITLE}를",
            "서류명 {TITLE}",
            "{TITLE}이",
            "저작물명: {TITLE}",
            "{TITLE}가",
            "문서 {TITLE}",
            "{TITLE}에",
            "제목이 {TITLE}",
            "{TITLE}은",
            "작품제목 {TITLE}",
            "{TITLE}와",
            "서명: {TITLE}",
            "{TITLE}라는",
            "저작물 {TITLE}",
            "{TITLE}로",
            "문서제목 {TITLE}",
            "{TITLE}과",
            "작 {TITLE}",
            "{TITLE}에서",
            "계약명: {TITLE}",
            "{TITLE}를 통해",
            "서류 {TITLE}",
            "{TITLE}으로",
            "작품명칭 {TITLE}",
            "{TITLE}에 의거",
            "제 {TITLE}",
            "{TITLE}의 내용",
            "문서명 {TITLE}이며",
            "{TITLE}에 따라",
            "저작 {TITLE}",
            "{TITLE}라고",
            "서류명칭 {TITLE}",
            "{TITLE}에 관한",
            "작품 제목은 {TITLE}",
            "{TITLE}을 대상으로",
            "문건 {TITLE}",
            "{TITLE}이라는",
            "제명: {TITLE}",
        ],
        "URL": [
            # URL (50개 - 제로샷 강화)
            "URL: {URL}",
            "웹사이트 {URL}",
            "{URL} 참조",
            "링크: {URL}",
            "{URL}에서",
            "주소 {URL}",
            "{URL} 확인",
            "사이트 {URL}",
            "{URL}을",
            "홈페이지 {URL}",
            "{URL}로",
            "웹주소 {URL}",
            "{URL}에",
            "링크주소: {URL}",
            "{URL}를",
            "인터넷주소 {URL}",
            "{URL}의",
            "접속 {URL}",
            "{URL}에서 확인",
            "웹 {URL}",
            "{URL}로 접속",
            "주소: {URL}입니다",
            "{URL}을 통해",
            "사이트주소 {URL}",
            "{URL}에 공개",
            "홈 {URL}",
            "{URL}로부터",
            "온라인 {URL}",
            "{URL}에 게시",
            "링크 {URL}로",
            "{URL}을 참고",
            "웹페이지 {URL}",
            "{URL}에 업로드",
            "URL주소: {URL}",
            "{URL}를 통해",
            "포털 {URL}",
            "{URL}에서 다운로드",
            "인터넷 {URL}",
            "{URL}로 연결",
            "웹링크 {URL}",
            "{URL}을 클릭",
            "주소는 {URL}",
            "{URL}에 접속",
            "사이트 링크 {URL}",
            "{URL}로 이동",
            "페이지 {URL}",
            "{URL}에 명시",
            "온라인주소 {URL}",
            "{URL}를 방문",
            "웹사이트주소: {URL}",
        ],
        "DESCRIPTION": [
            # 설명/내용 (50개 - 제로샷 강화)
            "내용: {DESCRIPTION}",
            "설명 {DESCRIPTION}",
            "{DESCRIPTION}에 관한",
            "목적: {DESCRIPTION}",
            "{DESCRIPTION}을",
            "용도 {DESCRIPTION}",
            "{DESCRIPTION}의",
            "내용은 {DESCRIPTION}",
            "{DESCRIPTION}를",
            "설명: {DESCRIPTION}입니다",
            "{DESCRIPTION}이",
            "용도: {DESCRIPTION}",
            "{DESCRIPTION}가",
            "목적 {DESCRIPTION}",
            "{DESCRIPTION}에",
            "내역 {DESCRIPTION}",
            "{DESCRIPTION}은",
            "사항: {DESCRIPTION}",
            "{DESCRIPTION}와",
            "설명내용 {DESCRIPTION}",
            "{DESCRIPTION}로",
            "목적내용: {DESCRIPTION}",
            "{DESCRIPTION}과",
            "용도설명 {DESCRIPTION}",
            "{DESCRIPTION}으로",
            "내용이 {DESCRIPTION}",
            "{DESCRIPTION}에서",
            "사유: {DESCRIPTION}",
            "{DESCRIPTION}에 대해",
            "설명문 {DESCRIPTION}",
            "{DESCRIPTION}을 위해",
            "목적은 {DESCRIPTION}",
            "{DESCRIPTION}를 통해",
            "내용설명 {DESCRIPTION}",
            "{DESCRIPTION}에 따라",
            "용도는 {DESCRIPTION}",
            "{DESCRIPTION}에 의거",
            "사항 {DESCRIPTION}",
            "{DESCRIPTION}로서",
            "내역: {DESCRIPTION}",
            "{DESCRIPTION}에 관하여",
            "설명항목 {DESCRIPTION}",
            "{DESCRIPTION}이라는",
            "목적사항 {DESCRIPTION}",
            "{DESCRIPTION}라고",
            "용도내용 {DESCRIPTION}",
            "{DESCRIPTION}을 목적으로",
            "내용항목: {DESCRIPTION}",
            "{DESCRIPTION}를 위한",
            "사유내용 {DESCRIPTION}",
        ],
        "TYPE": [
            # 저작물 유형 등 (50개 - 제로샷 강화)
            "유형: {TYPE}",
            "종류 {TYPE}",
            "{TYPE} 저작물",
            "형태: {TYPE}",
            "{TYPE}의",
            "타입 {TYPE}",
            "{TYPE}을",
            "종류는 {TYPE}",
            "{TYPE}를",
            "유형은 {TYPE}",
            "{TYPE}이",
            "분류: {TYPE}",
            "{TYPE}가",
            "타입: {TYPE}",
            "{TYPE}에",
            "종류: {TYPE}입니다",
            "{TYPE}은",
            "형태 {TYPE}",
            "{TYPE}와",
            "유형이 {TYPE}",
            "{TYPE}로",
            "카테고리: {TYPE}",
            "{TYPE}과",
            "분류 {TYPE}",
            "{TYPE}으로",
            "종류가 {TYPE}",
            "{TYPE}에서",
            "형식: {TYPE}",
            "{TYPE}에 해당",
            "타입은 {TYPE}",
            "{TYPE}를 통해",
            "유형: {TYPE}이며",
            "{TYPE}에 따라",
            "종류분류 {TYPE}",
            "{TYPE}에 의거",
            "분류는 {TYPE}",
            "{TYPE}로서",
            "형태는 {TYPE}",
            "{TYPE}에 관한",
            "타입분류 {TYPE}",
            "{TYPE}이라는",
            "유형분류: {TYPE}",
            "{TYPE}라고",
            "종류형태 {TYPE}",
            "{TYPE}을 기준",
            "분류형태: {TYPE}",
            "{TYPE}를 대상",
            "형식은 {TYPE}",
            "{TYPE}에 속하는",
            "카테고리 {TYPE}",
        ],
        "STATUS": [
            # 상태 (50개 - 제로샷 강화)
            "상태: {STATUS}",
            "현황 {STATUS}",
            "{STATUS} 단계",
            "진행상태: {STATUS}",
            "{STATUS}로",
            "상황 {STATUS}",
            "{STATUS}의",
            "상태는 {STATUS}",
            "{STATUS}을",
            "처리상태 {STATUS}",
            "{STATUS}를",
            "현황: {STATUS}",
            "{STATUS}이",
            "진행 {STATUS}",
            "{STATUS}가",
            "상황: {STATUS}",
            "{STATUS}에",
            "상태가 {STATUS}",
            "{STATUS}은",
            "현황은 {STATUS}",
            "{STATUS}와",
            "단계: {STATUS}",
            "{STATUS}로서",
            "진행도 {STATUS}",
            "{STATUS}으로",
            "처리 {STATUS}",
            "{STATUS}에서",
            "상태정보: {STATUS}",
            "{STATUS}에 해당",
            "현황상태 {STATUS}",
            "{STATUS}를 통해",
            "상황은 {STATUS}",
            "{STATUS}에 따라",
            "진행상황 {STATUS}",
            "{STATUS}에 의거",
            "단계는 {STATUS}",
            "{STATUS}로 변경",
            "처리현황: {STATUS}",
            "{STATUS}에 관한",
            "상태는 현재 {STATUS}",
            "{STATUS}이라는",
            "현황정보 {STATUS}",
            "{STATUS}라고",
            "진행단계: {STATUS}",
            "{STATUS}을 확인",
            "상황정보 {STATUS}",
            "{STATUS}를 유지",
            "처리단계 {STATUS}",
            "{STATUS}에 있음",
            "현재상태: {STATUS}",
        ],
        "DEPARTMENT": [
            # 부서 (50개 - 제로샷 강화)
            "부서: {DEPARTMENT}",
            "담당부서 {DEPARTMENT}",
            "{DEPARTMENT} 소속",
            "부서명: {DEPARTMENT}",
            "{DEPARTMENT}의",
            "담당 {DEPARTMENT}",
            "{DEPARTMENT}에서",
            "부서는 {DEPARTMENT}",
            "{DEPARTMENT}을",
            "소속부서 {DEPARTMENT}",
            "{DEPARTMENT}를",
            "부서명 {DEPARTMENT}",
            "{DEPARTMENT}이",
            "담당처 {DEPARTMENT}",
            "{DEPARTMENT}가",
            "부: {DEPARTMENT}",
            "{DEPARTMENT}에",
            "소속 {DEPARTMENT}",
            "{DEPARTMENT}은",
            "담당부: {DEPARTMENT}",
            "{DEPARTMENT}와",
            "부서 {DEPARTMENT}로",
            "{DEPARTMENT}으로",
            "주관부서 {DEPARTMENT}",
            "{DEPARTMENT}과",
            "담당: {DEPARTMENT}",
            "{DEPARTMENT}에서 담당",
            "소관부서 {DEPARTMENT}",
            "{DEPARTMENT}를 통해",
            "부서명은 {DEPARTMENT}",
            "{DEPARTMENT}에 따라",
            "관리부서: {DEPARTMENT}",
            "{DEPARTMENT}에 의거",
            "담당처: {DEPARTMENT}",
            "{DEPARTMENT}로서",
            "주관 {DEPARTMENT}",
            "{DEPARTMENT}에 관한",
            "부서정보: {DEPARTMENT}",
            "{DEPARTMENT}이라는",
            "소속: {DEPARTMENT}",
            "{DEPARTMENT}라고",
            "담당부서는 {DEPARTMENT}",
            "{DEPARTMENT}을 거쳐",
            "관할부서 {DEPARTMENT}",
            "{DEPARTMENT}를 경유",
            "부: {DEPARTMENT}이며",
            "{DEPARTMENT}에서 처리",
            "주관처: {DEPARTMENT}",
            "{DEPARTMENT}로 이관",
            "담당팀: {DEPARTMENT}",
        ],
        "LANGUAGE": [
            # 언어 (50개 - 제로샷 강화)
            "언어: {LANGUAGE}",
            "{LANGUAGE}로 작성",
            "사용언어 {LANGUAGE}",
            "{LANGUAGE} 번역",
            "언어는 {LANGUAGE}",
            "{LANGUAGE}의",
            "작성언어: {LANGUAGE}",
            "{LANGUAGE}을",
            "{LANGUAGE} 버전",
            "언어 {LANGUAGE}로",
            "{LANGUAGE}를",
            "사용 {LANGUAGE}",
            "{LANGUAGE}이",
            "언어: {LANGUAGE}입니다",
            "{LANGUAGE}가",
            "작성 {LANGUAGE}",
            "{LANGUAGE}에",
            "번역언어 {LANGUAGE}",
            "{LANGUAGE}은",
            "{LANGUAGE} 문서",
            "{LANGUAGE}와",
            "언어버전: {LANGUAGE}",
            "{LANGUAGE}로",
            "사용언어: {LANGUAGE}",
            "{LANGUAGE}으로",
            "{LANGUAGE} 텍스트",
            "{LANGUAGE}과",
            "작성언어 {LANGUAGE}로",
            "{LANGUAGE}에서",
            "언어형태: {LANGUAGE}",
            "{LANGUAGE}를 통해",
            "번역 {LANGUAGE}",
            "{LANGUAGE}에 따라",
            "{LANGUAGE} 원문",
            "{LANGUAGE}에 의거",
            "사용언어는 {LANGUAGE}",
            "{LANGUAGE}로서",
            "언어는 {LANGUAGE}이며",
            "{LANGUAGE}에 관한",
            "{LANGUAGE} 원본",
            "{LANGUAGE}이라는",
            "작성언어는 {LANGUAGE}",
            "{LANGUAGE}라고",
            "번역언어: {LANGUAGE}",
            "{LANGUAGE}을 사용",
            "{LANGUAGE} 표기",
            "{LANGUAGE}를 기준",
            "사용 언어는 {LANGUAGE}",
            "{LANGUAGE}로 번역",
            "언어코드: {LANGUAGE}",
        ],
        "QUANTITY": [
            # 수량 (50개 - 제로샷 강화)
            "수량: {QUANTITY}",
            "{QUANTITY} 제작",
            "수량은 {QUANTITY}",
            "{QUANTITY}의",
            "개수 {QUANTITY}",
            "{QUANTITY}을",
            "수량: {QUANTITY}입니다",
            "{QUANTITY}를",
            "총 {QUANTITY}",
            "{QUANTITY}이",
            "개수: {QUANTITY}",
            "{QUANTITY}가",
            "분량 {QUANTITY}",
            "{QUANTITY}에",
            "수량이 {QUANTITY}",
            "{QUANTITY}은",
            "총량: {QUANTITY}",
            "{QUANTITY}와",
            "개수는 {QUANTITY}",
            "{QUANTITY}로",
            "수: {QUANTITY}",
            "{QUANTITY}과",
            "분량: {QUANTITY}",
            "{QUANTITY}으로",
            "총수량 {QUANTITY}",
            "{QUANTITY}에서",
            "개수가 {QUANTITY}",
            "{QUANTITY}를 통해",
            "총 개수는 {QUANTITY}",
            "{QUANTITY}에 따라",
            "수량정보: {QUANTITY}",
            "{QUANTITY}에 의거",
            "분량은 {QUANTITY}",
            "{QUANTITY}로서",
            "총량은 {QUANTITY}",
            "{QUANTITY}에 관한",
            "개수정보 {QUANTITY}",
            "{QUANTITY}이라는",
            "수량은 총 {QUANTITY}",
            "{QUANTITY}라고",
            "총 분량: {QUANTITY}",
            "{QUANTITY}을 기준",
            "수량규모 {QUANTITY}",
            "{QUANTITY}를 대상",
            "개수: {QUANTITY}이며",
            "{QUANTITY}으로 제작",
            "총수: {QUANTITY}",
            "{QUANTITY}에 달하는",
            "분량정보: {QUANTITY}",
            "{QUANTITY} 단위",
        ],
    }
    
    # === 2. 2개 엔티티 조합 템플릿 (400개) ===
    dual_entity_templates = [
        # NAME + 다른 엔티티 (강화: 처음 보는 이름도 문맥으로 학습)
        "{NAME}의 연락처는 {PHONE}입니다.",
        "{NAME} 전화번호는 {PHONE}이고",
        "본인 {NAME}은 {COMPANY}에 근무합니다.",
        "{NAME}({POSITION}) 서명",
        "작성자 {NAME}, 주소 {ADDRESS}",
        "{NAME}은 {DATE}에 동의했습니다.",
        "{NAME} 이메일 {EMAIL}",
        "{NAME}에게 {MONEY}를 지급합니다.",
        "{NAME}의 {CONTRACT_TYPE}",
        "권리자 {NAME}, 기간 {PERIOD}",
        "{NAME}님 {PHONE}로 연락",
        "{NAME}씨 전화번호 {PHONE}",
        "{NAME} 님의 휴대전화 {PHONE}",
        "{NAME}({PHONE}) 담당자",
        "성명 {NAME} 연락처 {PHONE}",
        "{NAME} 전화 {PHONE}입니다",
        "{NAME}의 핸드폰은 {PHONE}",
        "작성자 {NAME} Tel. {PHONE}",
        "{NAME} 선생님 {PHONE}",
        "{NAME} 작가 연락처 {PHONE}",
        
        # COMPANY + 다른 엔티티
        "{COMPANY}의 주소는 {ADDRESS}입니다.",
        "{COMPANY} 대표전화 {PHONE}",
        "{COMPANY}와 {DATE}에 계약",
        "{COMPANY}의 {POSITION} 채용",
        "{COMPANY}({EMAIL}) 문의",
        "{COMPANY}는 {MONEY}를 투자했습니다.",
        "{COMPANY}의 {CONTRACT_TYPE} 체결",
        "{COMPANY} 계약기간 {PERIOD}",
        "{COMPANY}와 {TITLE} 합의",
        "{COMPANY}의 {PROJECT_NAME}",
        
        # DATE + 다른 엔티티
        "{DATE}부터 {PERIOD}간 유효",
        "{DATE}에 {MONEY} 지급",
        "{DATE} {CONTRACT_TYPE} 발효",
        "{DATE} {ADDRESS}에서 체결",
        "{DATE} {TITLE} 작성",
        "{DATE}에 {PHONE}로 통보",
        "{DATE}부터 {COMPANY} 소속",
        "{DATE} {STATUS} 변경",
        "{DATE} {POSITION} 임명",
        "{DATE}에 {EMAIL}로 송신",
        
        # CONTRACT_TYPE + 다른 엔티티
        "{CONTRACT_TYPE}으로 {MONEY} 합의",
        "{CONTRACT_TYPE} 체결일 {DATE}",
        "{CONTRACT_TYPE} 계약자 {NAME}",
        "{CONTRACT_TYPE} 체결기관 {COMPANY}",
        "{CONTRACT_TYPE}의 {RIGHT_INFO} 이전",
        "{CONTRACT_TYPE} 기간 {PERIOD}",
        "{CONTRACT_TYPE} 목적: {DESCRIPTION}",
        "{CONTRACT_TYPE} 대상 {TYPE}",
        "{CONTRACT_TYPE} 현황 {STATUS}",
        "{CONTRACT_TYPE} 담당 {DEPARTMENT}",
        
        # MONEY + 다른 엔티티
        "{MONEY}의 {CONTRACT_TYPE}",
        "{MONEY} {DATE} 지급",
        "{MONEY}를 {NAME}에게",
        "{MONEY} {PERIOD}간 분할",
        "{MONEY} 규모의 {PROJECT_NAME}",
        "{MONEY} {COMPANY} 납부",
        "{MONEY} 보상 {RIGHT_INFO}",
        "{MONEY}에 대한 {DESCRIPTION}",
        "{MONEY}의 {TYPE} 가격",
        "{MONEY} {QUANTITY} 단가",
        
        # RIGHT_INFO + 다른 엔티티
        "{RIGHT_INFO}을 {NAME}에게 양도",
        "{RIGHT_INFO} 보유자 {COMPANY}",
        "{RIGHT_INFO} 대가 {MONEY}",
        "{RIGHT_INFO}의 {CONTRACT_TYPE}",
        "{RIGHT_INFO} 유효기간 {PERIOD}",
        "{RIGHT_INFO} 근거 {LAW_REFERENCE}",
        "{RIGHT_INFO} 관련 {TITLE}",
        "{RIGHT_INFO} 이전일 {DATE}",
        "{RIGHT_INFO} 관리부서 {DEPARTMENT}",
        "{RIGHT_INFO}에 대한 {DESCRIPTION}",
        
        # PROJECT_NAME + 다른 엔티티
        "{PROJECT_NAME} 예산 {MONEY}",
        "{PROJECT_NAME} 기간 {PERIOD}",
        "{PROJECT_NAME} 담당사 {COMPANY}",
        "{PROJECT_NAME} 책임자 {NAME}",
        "{PROJECT_NAME} 시작일 {DATE}",
        "{PROJECT_NAME}의 {TYPE} 개발",
        "{PROJECT_NAME} 진행 {STATUS}",
        "{PROJECT_NAME} 소관부서 {DEPARTMENT}",
        "{PROJECT_NAME} 문의 {EMAIL}",
        "{PROJECT_NAME}에 대한 {DESCRIPTION}",
        
        # ADDRESS + 다른 엔티티
        "{ADDRESS} 소재 {COMPANY}",
        "{ADDRESS} 거주 {NAME}",
        "{ADDRESS} 대표전화 {PHONE}",
        "{ADDRESS}에서 {DATE} 개최",
        "{ADDRESS} 담당부서 {DEPARTMENT}",
        "{ADDRESS} 이메일 {EMAIL}",
        "{ADDRESS}의 {TYPE} 사업장",
        "{ADDRESS} 계약장소로 지정",
        "{ADDRESS} {STATUS} 확인",
        "{ADDRESS}에 대한 {DESCRIPTION}",
        
        # LAW_REFERENCE + 다른 엔티티
        "{LAW_REFERENCE}에 따른 {RIGHT_INFO}",
        "{LAW_REFERENCE} 위반시 {MONEY} 벌금",
        "{LAW_REFERENCE} 적용 {CONTRACT_TYPE}",
        "{LAW_REFERENCE} 근거 {TITLE}",
        "{LAW_REFERENCE}에 의한 {DESCRIPTION}",
        "{LAW_REFERENCE} 개정 {DATE}",
        "{LAW_REFERENCE} 관할 {DEPARTMENT}",
        "{LAW_REFERENCE}의 {TYPE} 규정",
        "{LAW_REFERENCE} 준수 {STATUS}",
        "{LAW_REFERENCE} 해석 {COMPANY} 의뢰",
        
        # 더 다양한 조합 (각 카테고리별 10개씩 추가)
        "{NAME}은 {ID_NUM} 본인임을 확인합니다.",
        "{COMPANY}의 {CONSENT_TYPE}을 득합니다.",
        "{TITLE} 문서의 {LANGUAGE} 번역본",
        "{TYPE} 저작물의 {RIGHT_INFO}",
        "{PROJECT_NAME} 수행사 {COMPANY}",
        "{DEPARTMENT}의 {POSITION} 담당",
        "{STATUS} 상태의 {CONTRACT_TYPE}",
        "{DESCRIPTION}에 관한 {TITLE}",
        "{QUANTITY}의 {TYPE} 제작",
        "{LANGUAGE}로 작성된 {TITLE}",
        "{PHONE} 연락 {NAME}",
        "{EMAIL} 담당자 {POSITION}",
        "{ADDRESS}에서 {COMPANY} 운영",
        "{DATE}까지 {STATUS}",
        "{PERIOD} 이내 {MONEY} 지급",
        "{CONTRACT_TYPE}에 따른 {RIGHT_INFO} 이전",
        "{LAW_REFERENCE} 제정 {DATE}",
        "{PROJECT_NAME}의 {TYPE}",
        "{CONSENT_TYPE} 일자 {DATE}",
        "{ID_NUM} 본인 {NAME}",
        "{URL} 참조 {DESCRIPTION}",
        "{TITLE}의 {LANGUAGE} 버전",
        "{COMPANY} {DEPARTMENT} 소속",
        "{POSITION} {NAME} 승인",
        "{MONEY} 규모 {PROJECT_NAME}",
        "{DATE} {STATUS} 전환",
        "{TYPE} {QUANTITY} 발주",
        "{RIGHT_INFO} 기간 {PERIOD}",
        "{CONTRACT_TYPE} 당사자 {NAME}",
        "{TITLE} 제출처 {COMPANY}",
        "{DESCRIPTION} 내용 {LANGUAGE}",
        "{ADDRESS} 팩스 {PHONE}",
        "{PROJECT_NAME} 예산 {MONEY}",
        "{DEPARTMENT} 이메일 {EMAIL}",
        "{LAW_REFERENCE} 적용 {TYPE}",
        "{CONSENT_TYPE} 대상 {NAME}",
        "{STATUS} 문서 {TITLE}",
        "{QUANTITY} 단위 {MONEY}",
        "{PERIOD} 계약 {COMPANY}",
        "{URL} 링크된 {TYPE}",
        
        # 의미있는 문장 형태 추가 (100개)
        "본 계약은 {NAME}과(와) {COMPANY} 간에 체결됩니다.",
        "{NAME}은(는) {POSITION}으로서 본 계약을 체결합니다.",
        "계약일자는 {DATE}이며 기간은 {PERIOD}입니다.",
        "{COMPANY}는 {NAME}에게 {MONEY}를 지급하기로 합니다.",
        "{CONTRACT_TYPE}에 따라 {RIGHT_INFO}를 양도합니다.",
        "{PROJECT_NAME} 수행을 위해 {COMPANY}와 계약합니다.",
        "본 계약서는 {TITLE}로서 {DATE}에 작성되었습니다.",
        "{NAME}의 주소는 {ADDRESS}이고 연락처는 {PHONE}입니다.",
        "{LAW_REFERENCE}에 근거하여 {RIGHT_INFO}를 행사합니다.",
        "{COMPANY}의 {DEPARTMENT}에서 {PROJECT_NAME}를 담당합니다.",
        "{CONSENT_TYPE}에 {NAME}이(가) 동의했습니다.",
        "{TYPE} 저작물의 {RIGHT_INFO}는 {COMPANY}에 귀속됩니다.",
        "{DATE}부터 {PERIOD}간 {CONTRACT_TYPE}이 유효합니다.",
        "대가로 {MONEY}를 {DATE}에 지급합니다.",
        "{TITLE} 계약서는 {LANGUAGE}로 작성됩니다.",
        "{COMPANY}는 {ADDRESS}에 본사를 두고 있습니다.",
        "{NAME} {POSITION}이(가) {PROJECT_NAME}를 총괄합니다.",
        "{RIGHT_INFO}의 대가는 {MONEY}로 산정됩니다.",
        "{CONTRACT_TYPE} 진행 {STATUS}는 정상입니다.",
        "{DEPARTMENT}에 {EMAIL}로 문의하시기 바랍니다.",
        "총 {QUANTITY}의 {TYPE}을(를) 제작합니다.",
        "{PROJECT_NAME}의 예산은 {MONEY}입니다.",
        "{LAW_REFERENCE}를 준수하여 {CONTRACT_TYPE}를 이행합니다.",
        "{NAME}은(는) {ID_NUM}으로 신원이 확인됩니다.",
        "자세한 내용은 {URL}을(를) 참조하세요.",
        "{DESCRIPTION} 조항은 {TITLE}에 명시되어 있습니다.",
        "{COMPANY}의 {POSITION} {NAME}이(가) 서명합니다.",
        "{TYPE} 형태의 {PROJECT_NAME}를 진행합니다.",
        "{CONSENT_TYPE}은(는) {DATE}에 취득했습니다.",
        "{PERIOD} 동안 {RIGHT_INFO}를 행사할 수 있습니다.",
        "{ADDRESS}에서 {DATE}에 계약을 체결했습니다.",
        "{MONEY}의 {QUANTITY}만큼 비용이 발생합니다.",
        "{LANGUAGE} 및 {LANGUAGE}로 이중 작성됩니다.",
        "{DEPARTMENT}는 {PHONE}으로 연락 가능합니다.",
        "{STATUS} 상태를 {DATE}에 확인했습니다.",
        "{LAW_REFERENCE} 개정으로 {RIGHT_INFO}가 변경되었습니다.",
        "{COMPANY}와 {CONTRACT_TYPE}을(를) {DATE}에 갱신합니다.",
        "{NAME}에게 {TYPE} 저작물의 {RIGHT_INFO}를 부여합니다.",
        "{PROJECT_NAME} 관련 {TITLE}을(를) 작성합니다.",
        "{PERIOD} 이내에 {MONEY}를 납부해야 합니다.",
        "{CONSENT_TYPE} 철회는 {EMAIL}로 요청하세요.",
        "{ADDRESS} 소재 {COMPANY}가 발주처입니다.",
        "{QUANTITY}에 해당하는 {DESCRIPTION}입니다.",
        "{TITLE}의 {URL}에서 확인 가능합니다.",
        "{DEPARTMENT} 소속 {NAME} {POSITION}",
        "{DATE} 현재 {STATUS}로 처리 중입니다.",
        "{CONTRACT_TYPE}의 {TYPE}은(는) {LANGUAGE}로 제공됩니다.",
        "{RIGHT_INFO} 행사기간은 {PERIOD}로 제한됩니다.",
        "{COMPANY}의 {PROJECT_NAME}에 {MONEY}를 배정합니다.",
        "{LAW_REFERENCE} 위반 시 {CONTRACT_TYPE}은(는) 해지됩니다.",
        "{NAME}은(는) {PHONE}과(와) {EMAIL}로 연락됩니다.",
        "{TYPE} 제작 수량은 {QUANTITY}입니다.",
        "{DESCRIPTION}에 관한 {CONSENT_TYPE}",
        "{ADDRESS}의 {DEPARTMENT}에서 관리합니다.",
        "{PROJECT_NAME} {STATUS}를 {DATE}에 보고합니다.",
        "{TITLE} 문서는 {COMPANY}가 보관합니다.",
        "{MONEY} 상당의 {RIGHT_INFO}를 양도받습니다.",
        "{PERIOD}간 {TYPE} 이용권을 부여합니다.",
        "{CONTRACT_TYPE} 해지 시 {MONEY}를 반환합니다.",
        "{LANGUAGE} 통역이 필요한 {TITLE}",
        "{URL}에 게시된 {PROJECT_NAME}",
        "{ID_NUM} 확인 후 {CONSENT_TYPE} 처리",
        "{PHONE} 또는 {EMAIL}로 문의하세요.",
        "{COMPANY}는 {ADDRESS}에서 {TYPE}을(를) 제작합니다.",
        "{NAME} {POSITION}의 {DESCRIPTION}",
        "{DATE}부터 {STATUS}가 변경됩니다.",
        "{QUANTITY}만큼 {PROJECT_NAME}에 투입됩니다.",
        "{DEPARTMENT}가 {LAW_REFERENCE}를 적용합니다.",
        "{RIGHT_INFO} 귀속은 {CONTRACT_TYPE}에 따릅니다.",
        "{PERIOD} 후 {TITLE}을(를) 갱신합니다.",
        "{COMPANY}의 {TYPE} {QUANTITY} 생산",
        "{NAME}이(가) {DATE}에 {CONSENT_TYPE}했습니다.",
        "{PROJECT_NAME} 참여자 {PHONE} 연락",
        "{MONEY} 예산으로 {DESCRIPTION} 수행",
        "{URL}에서 {LANGUAGE} 문서 다운로드",
        "{ADDRESS} 방문 시 {COMPANY} 확인",
        "{STATUS} 처리 완료된 {TITLE}",
        "{CONTRACT_TYPE} 기반 {RIGHT_INFO} 라이선스",
        "{POSITION} {NAME}의 {EMAIL} 주소",
        "{TYPE} 형식의 {QUANTITY} 납품",
        "{LAW_REFERENCE} 준거로 {DESCRIPTION} 해석",
        "{DEPARTMENT}에서 {PROJECT_NAME}을(를) 진행합니다.",
        "{DATE} 이후 {PERIOD}간 효력 유지",
        "{COMPANY}와 {NAME}의 {CONTRACT_TYPE}",
        "{MONEY} 규모의 {TYPE} 사업",
        "{CONSENT_TYPE} 서류는 {ADDRESS}로 송부",
        "{TITLE} {STATUS} 확인 요청",
        "{LANGUAGE} 버전 {URL} 참조",
        "{PROJECT_NAME}의 {QUANTITY} 목표 달성",
        "{RIGHT_INFO} 기간 {PERIOD} 연장 가능",
        "{PHONE}번으로 {NAME} 담당자 호출",
        "{ID_NUM} 소지자에 한해 {CONSENT_TYPE}",
        "{DEPARTMENT}의 {DESCRIPTION} 업무",
        "{DATE} 기준 {MONEY} 정산",
        "{COMPANY}가 {LAW_REFERENCE}를 적용합니다.",
        "{ADDRESS}에 {EMAIL} 서면 송부",
        "{TYPE} {QUANTITY}에 대한 {CONTRACT_TYPE}",
        "{PROJECT_NAME} {STATUS} 변동사항",
        "{TITLE}의 {LANGUAGE} 및 영문 버전",
        "{NAME} {POSITION}이 {PERIOD}간 근무",
    ]
    
    # === 3. 3개 이상 엔티티 조합 템플릿 (200개) ===
    multi_entity_templates = [
        "{NAME}은(는) {COMPANY}의 {POSITION}으로서 {DATE}에 계약합니다.",
        "{NAME}({PHONE})은(는) {ADDRESS}에 거주하며 {COMPANY}에 소속되어 있습니다.",
        "본 {CONTRACT_TYPE}은(는) {NAME}과(와) {COMPANY} 간에 {DATE}부터 {PERIOD}간 유효합니다.",
        "{COMPANY}는 {NAME}에게 {RIGHT_INFO}에 대한 대가로 {MONEY}를 {DATE}에 지급합니다.",
        "{PROJECT_NAME}은(는) {COMPANY}가 {MONEY} 예산으로 {PERIOD}간 진행합니다.",
        "{NAME} {POSITION}은(는) {EMAIL}과(와) {PHONE}로 {COMPANY}에서 연락 가능합니다.",
        "{TITLE} 계약서는 {LANGUAGE}로 작성되었으며 {DATE}에 {COMPANY}와 {NAME}이 체결했습니다.",
        "{RIGHT_INFO}는 {LAW_REFERENCE}에 따라 {NAME}으로부터 {COMPANY}에게 {DATE}에 양도됩니다.",
        "{COMPANY}의 {ADDRESS}에서 {NAME} {POSITION}이(가) {PROJECT_NAME}를 {DATE}부터 수행합니다.",
        "{TYPE} 저작물의 {RIGHT_INFO}는 {CONTRACT_TYPE}에 따라 {NAME}에서 {COMPANY}로 이전됩니다.",
        "{PROJECT_NAME} 관련하여 {NAME}은(는) {COMPANY} {DEPARTMENT}에 {EMAIL}로 {DATE}까지 제출합니다.",
        "{CONSENT_TYPE}에 {NAME}({ID_NUM})이(가) {DATE}에 동의하였으며 {COMPANY}가 관리합니다.",
        "{CONTRACT_TYPE}의 {TITLE}은(는) {DATE}에 작성되었고 {PERIOD}간 {MONEY}로 합의되었습니다.",
        "{COMPANY}는 {ADDRESS}에 본사를 두고 {NAME} {POSITION}이(가) {PROJECT_NAME}를 총괄합니다.",
        "{DATE}부터 {PERIOD}간 {NAME}은(는) {COMPANY}에서 {POSITION}으로 {MONEY}의 급여로 근무합니다.",
        "{LAW_REFERENCE}를 근거로 {RIGHT_INFO}는 {NAME}에게 {PERIOD}간 {MONEY}의 대가로 허락됩니다.",
        "{TITLE} 문서는 {LANGUAGE}로 작성되며 {COMPANY}의 {DEPARTMENT}에서 {DATE}까지 검토합니다.",
        "{PROJECT_NAME}의 {TYPE} 제작 수량은 {QUANTITY}이며 {COMPANY}가 {MONEY}로 수주했습니다.",
        "{NAME}은(는) {PHONE}과(와) {EMAIL}로 연락되며 {ADDRESS}에 거주하고 {COMPANY}에 재직 중입니다.",
        "{CONTRACT_TYPE}에 따라 {RIGHT_INFO}를 {NAME}으로부터 {COMPANY}가 {DATE}에 {MONEY}로 매입합니다.",
        "{COMPANY}의 {PROJECT_NAME}는 {DATE}에 시작하여 {PERIOD}간 진행되며 예산은 {MONEY}입니다.",
        "{DEPARTMENT}에서 {NAME} {POSITION}이(가) {PROJECT_NAME}를 담당하며 {EMAIL}로 문의 가능합니다.",
        "{TITLE}은(는) {CONTRACT_TYPE}으로서 {DATE}에 체결되었고 {COMPANY}와 {NAME}이 서명했습니다.",
        "{RIGHT_INFO}의 {CONTRACT_TYPE}은(는) {LAW_REFERENCE}에 근거하며 {PERIOD}간 {MONEY}로 합의되었습니다.",
        "{NAME}은(는) {ID_NUM}으로 신원확인되며 {CONSENT_TYPE}을(를) {DATE}에 {COMPANY}에 제출했습니다.",
        "{PROJECT_NAME}은(는) {COMPANY} {DEPARTMENT}가 {MONEY} 예산으로 {DATE}부터 {PERIOD}간 수행합니다.",
        "{TYPE} 형태의 {QUANTITY}를 {COMPANY}가 {MONEY}에 제작하여 {DATE}까지 납품합니다.",
        "{CONTRACT_TYPE}의 {STATUS}는 {DATE} 현재 진행중이며 {COMPANY} {DEPARTMENT}에서 관리합니다.",
        "{NAME} {POSITION}은(는) {ADDRESS}의 {COMPANY}에서 {PROJECT_NAME}를 {PERIOD}간 수행합니다.",
        "{TITLE} 문서의 {LANGUAGE} 버전은 {URL}에 게시되어 있으며 {DATE}에 업데이트되었습니다.",
        "{DESCRIPTION}에 관한 {TITLE}은(는) {COMPANY}가 {DATE}에 작성하여 {DEPARTMENT}에 제출했습니다.",
        "{RIGHT_INFO}는 {CONTRACT_TYPE}을(를) 통해 {NAME}에서 {COMPANY}로 {DATE}에 {MONEY}로 양도됩니다.",
        "{CONSENT_TYPE}은(는) {NAME}({PHONE})이(가) {DATE}에 {COMPANY}에 {EMAIL}로 제출했습니다.",
        "{PROJECT_NAME}의 {TYPE} 제작을 위해 {COMPANY}는 {NAME} {POSITION}에게 {MONEY}를 지급합니다.",
        "{LAW_REFERENCE}에 따라 {RIGHT_INFO}의 {CONTRACT_TYPE}은(는) {PERIOD}간 {MONEY}로 체결됩니다.",
        "{COMPANY}는 {ADDRESS}에서 {PROJECT_NAME}를 {DATE}부터 시작하며 {DEPARTMENT}가 담당합니다.",
        "{NAME}은(는) {POSITION}으로서 {COMPANY}의 {PROJECT_NAME}에 {PERIOD}간 참여합니다.",
        "{TITLE}의 {CONTRACT_TYPE}은(는) {DATE}에 체결되었으며 {MONEY}를 {PERIOD}에 걸쳐 분할 지급합니다.",
        "{TYPE} 저작물 {QUANTITY}의 {RIGHT_INFO}는 {NAME}에게 있으며 {LAW_REFERENCE}로 보호됩니다.",
        "{COMPANY} {DEPARTMENT}의 {NAME} {POSITION}은(는) {PHONE}과(와) {EMAIL}로 연락됩니다.",
        "{PROJECT_NAME} 진행을 위해 {CONSENT_TYPE}을(를) {NAME}으로부터 {DATE}에 취득했습니다.",
        "{CONTRACT_TYPE}에 기반한 {RIGHT_INFO} 이전은 {DATE}에 {MONEY}의 대가로 이루어집니다.",
        "{TITLE}은(는) {LANGUAGE}로 작성되었으며 {COMPANY}와 {NAME}이 {ADDRESS}에서 {DATE}에 체결했습니다.",
        "{PROJECT_NAME}의 {STATUS}는 {DATE} 기준 양호하며 {COMPANY} {DEPARTMENT}가 관리합니다.",
        "{NAME}은(는) {ADDRESS}에 거주하며 {ID_NUM}으로 {CONSENT_TYPE}에 {DATE}에 동의했습니다.",
        "{COMPANY}는 {PROJECT_NAME}에 {MONEY}를 투자하며 {TYPE} {QUANTITY}를 {PERIOD}간 제작합니다.",
        "{RIGHT_INFO}에 대한 {CONTRACT_TYPE}은(는) {LAW_REFERENCE}를 준거법으로 {DATE}에 체결됩니다.",
        "{DEPARTMENT}에서 {EMAIL}로 {TITLE}을(를) {DATE}까지 {LANGUAGE}로 제출해 주시기 바랍니다.",
        "{NAME} {POSITION}이(가) {PROJECT_NAME}를 총괄하며 {PHONE}로 {COMPANY}에서 연락 가능합니다.",
        "{TYPE} 형태의 {PROJECT_NAME}는 {COMPANY}가 {MONEY} 예산으로 {PERIOD}간 수행합니다.",
        "{CONTRACT_TYPE}의 {TITLE}은(는) {DATE}에 작성되었으며 {COMPANY}와 {NAME}이 {MONEY}로 합의했습니다.",
        "{CONSENT_TYPE} 문서는 {ADDRESS}로 {DATE}까지 {LANGUAGE}로 작성하여 제출해야 합니다.",
        "{PROJECT_NAME} 관련 {DESCRIPTION}은(는) {COMPANY} {DEPARTMENT}의 {NAME}에게 {EMAIL}로 문의하세요.",
        "{RIGHT_INFO}는 {LAW_REFERENCE}에 의해 {PERIOD}간 보호되며 {COMPANY}가 {MONEY}로 매입했습니다.",
        "{NAME}은(는) {COMPANY}의 {DEPARTMENT}에서 {POSITION}으로 {DATE}부터 {PERIOD}간 근무합니다.",
        "{TITLE} 계약서는 {CONTRACT_TYPE}으로 {DATE}에 체결되었고 {MONEY}를 {PERIOD}에 분할 지급합니다.",
        "{PROJECT_NAME}의 {TYPE} 제작 수량 {QUANTITY}는 {COMPANY}가 {DATE}까지 납품합니다.",
        "{COMPANY}는 {ADDRESS}에 본사를 두고 {PROJECT_NAME}를 {MONEY} 예산으로 진행합니다.",
        "{NAME}({ID_NUM})은(는) {CONSENT_TYPE}에 {DATE}에 동의하였으며 {PHONE}로 연락됩니다.",
        "{RIGHT_INFO}의 {CONTRACT_TYPE}은(는) {NAME}과(와) {COMPANY} 간에 {DATE}에 {MONEY}로 체결됩니다.",
        "{PROJECT_NAME} 수행을 위해 {DEPARTMENT}의 {NAME} {POSITION}이(가) {PERIOD}간 전담합니다.",
        "{TITLE}은(는) {LANGUAGE} 및 영문으로 작성되며 {DATE}에 {COMPANY}가 {ADDRESS}에서 발행합니다.",
        "{TYPE} 저작물의 {RIGHT_INFO}는 {LAW_REFERENCE}에 따라 {PERIOD}간 {COMPANY}에 귀속됩니다.",
        "{CONTRACT_TYPE}의 {STATUS}는 {DATE} 기준 정상이며 {COMPANY} {DEPARTMENT}에 {EMAIL}로 확인하세요.",
        "{NAME}은(는) {POSITION}으로서 {COMPANY}의 {PROJECT_NAME}에 참여하며 {MONEY}를 수령합니다.",
        "{CONSENT_TYPE} 취득 후 {PROJECT_NAME}를 {DATE}부터 {PERIOD}간 진행하며 {COMPANY}가 담당합니다.",
        "{TITLE}의 {CONTRACT_TYPE}은(는) {DATE}에 체결되었고 {RIGHT_INFO}는 {MONEY}로 산정됩니다.",
        "{COMPANY}는 {PROJECT_NAME}의 {TYPE} 제작을 {NAME} {POSITION}에게 {MONEY}에 의뢰합니다.",
        "{LAW_REFERENCE}를 준거로 {RIGHT_INFO}의 {CONTRACT_TYPE}은(는) {PERIOD}간 유효합니다.",
        "{NAME}({PHONE}, {EMAIL})은(는) {ADDRESS}에서 {COMPANY}의 {DEPARTMENT}에 소속되어 있습니다.",
        "{PROJECT_NAME}은(는) {MONEY} 규모로 {COMPANY}가 {DATE}에 착수하여 {PERIOD}간 수행합니다.",
        "{TITLE} 문서는 {LANGUAGE}로 작성되며 {URL}에서 확인 가능하고 {DATE}에 업데이트됩니다.",
        "{TYPE} {QUANTITY}의 {PROJECT_NAME}를 {COMPANY}가 {MONEY}에 수주하여 {DATE}까지 완료합니다.",
        "{NAME}은(는) {ID_NUM}으로 신원확인되며 {CONSENT_TYPE}을(를) {COMPANY}에 {DATE}에 제출했습니다.",
        "{RIGHT_INFO}는 {CONTRACT_TYPE}을(를) 통해 {DATE}에 {MONEY}로 {NAME}에서 {COMPANY}로 이전됩니다.",
        "{DEPARTMENT}에서 {PROJECT_NAME}를 담당하며 {NAME} {POSITION}에게 {PHONE}으로 문의하세요.",
        "{CONTRACT_TYPE}의 {TITLE}은(는) {DATE}에 작성되고 {PERIOD}간 {MONEY}로 계약되었습니다.",
        "{COMPANY}는 {ADDRESS}에서 {PROJECT_NAME}를 진행하며 {DEPARTMENT}의 {EMAIL}로 문의됩니다.",
        "{NAME} {POSITION}은(는) {COMPANY}의 {PROJECT_NAME}에서 {RIGHT_INFO} 관련 업무를 수행합니다.",
        "{TITLE}의 {CONTRACT_TYPE}은(는) {LAW_REFERENCE}에 근거하여 {DATE}에 체결되었습니다.",
        "{PROJECT_NAME}의 {TYPE} 제작은 {COMPANY}가 {MONEY}로 수주하여 {QUANTITY}만큼 납품합니다.",
        "{CONSENT_TYPE} 서류를 {NAME}으로부터 {DATE}에 취득하여 {COMPANY} {DEPARTMENT}에 보관합니다.",
        "{RIGHT_INFO}의 대가로 {MONEY}를 {PERIOD}에 걸쳐 {NAME}에게 지급하는 {CONTRACT_TYPE}입니다.",
        "{COMPANY}의 {PROJECT_NAME}는 {DATE}부터 {PERIOD}간 진행되며 {STATUS}는 정상입니다.",
        "{NAME}은(는) {PHONE}과(와) {EMAIL}로 연락되며 {ADDRESS}의 {COMPANY}에 재직 중입니다.",
        "{TITLE} 계약서는 {LANGUAGE}로 작성되고 {COMPANY}와 {NAME}이 {DATE}에 {MONEY}로 체결했습니다.",
        "{PROJECT_NAME}의 {DESCRIPTION}은(는) {COMPANY} {DEPARTMENT}에서 {DATE}까지 작성합니다.",
        "{TYPE} 저작물 {QUANTITY}의 {RIGHT_INFO}는 {LAW_REFERENCE}로 {PERIOD}간 보호됩니다.",
        "{CONTRACT_TYPE}에 따라 {NAME}은(는) {COMPANY}로부터 {MONEY}를 {DATE}에 수령합니다.",
        "{COMPANY}는 {ADDRESS}에 위치하며 {PROJECT_NAME}를 {MONEY} 예산으로 수행합니다.",
        "{NAME} {POSITION}이(가) {DATE}부터 {PERIOD}간 {COMPANY} {DEPARTMENT}에서 근무합니다.",
        "{TITLE}의 {STATUS}는 {DATE} 현재 완료 단계이며 {COMPANY}에서 관리합니다.",
        "{RIGHT_INFO}는 {CONTRACT_TYPE}을(를) 통해 {PERIOD}간 {MONEY}로 {NAME}에게 허락됩니다.",
        "{PROJECT_NAME} 관련 {CONSENT_TYPE}을(를) {NAME}으로부터 {DATE}에 {COMPANY}가 취득했습니다.",
        "{LANGUAGE}로 작성된 {TITLE}은(는) {URL}에서 다운로드 가능하며 {DATE}에 갱신되었습니다.",
        "{COMPANY}의 {DEPARTMENT}는 {PROJECT_NAME}를 {DATE}부터 {PERIOD}간 담당합니다.",
        "{NAME}은(는) {ID_NUM}으로 확인되며 {CONSENT_TYPE}에 {DATE}에 {PHONE}로 동의했습니다.",
        "{TYPE} {QUANTITY}를 {COMPANY}가 {MONEY}에 제작하여 {DATE}까지 {ADDRESS}로 납품합니다.",
        "{CONTRACT_TYPE}의 {TITLE}은(는) {LAW_REFERENCE}를 준거법으로 {DATE}에 체결됩니다.",
        "{PROJECT_NAME}의 {RIGHT_INFO}는 {NAME}에게 있으며 {COMPANY}가 {MONEY}로 매입합니다.",
        "{DEPARTMENT}의 {NAME} {POSITION}에게 {EMAIL}과(와) {PHONE}로 {TITLE} 관련 문의하세요.",
        "{COMPANY}는 {PROJECT_NAME}를 {MONEY} 예산으로 {DATE}부터 {PERIOD}간 진행합니다.",
        "{CONSENT_TYPE}은(는) {NAME}이(가) {DATE}에 제출하였고 {COMPANY}의 {ADDRESS}에 보관됩니다.",
        "{RIGHT_INFO}의 {CONTRACT_TYPE}은(는) {PERIOD}간 유효하며 {MONEY}를 {NAME}에게 지급합니다.",
        "{TITLE} 문서는 {LANGUAGE}로 작성되고 {COMPANY} {DEPARTMENT}에서 {DATE}까지 검토합니다.",
        "{PROJECT_NAME}의 {TYPE} 제작 수량은 {QUANTITY}이며 {COMPANY}가 {DATE}까지 완료합니다.",
        "{NAME}({PHONE})은(는) {POSITION}으로서 {COMPANY}의 {ADDRESS}에서 근무합니다.",
        "{CONTRACT_TYPE}에 따라 {RIGHT_INFO}는 {NAME}에서 {COMPANY}로 {DATE}에 {MONEY}로 양도됩니다.",
        "{PROJECT_NAME}의 {STATUS}는 {DATE} 기준 진행중이며 {DEPARTMENT}의 {EMAIL}로 확인하세요.",
        "{COMPANY}는 {PROJECT_NAME}에 {MONEY}를 투자하여 {TYPE} {QUANTITY}를 {PERIOD}간 제작합니다.",
        "{NAME} {POSITION}은(는) {COMPANY} {DEPARTMENT}에서 {PROJECT_NAME}를 담당하며 {PHONE}로 연락됩니다.",
        "{TITLE}의 {CONTRACT_TYPE}은(는) {DATE}에 체결되었고 {RIGHT_INFO}는 {LAW_REFERENCE}로 보호됩니다.",
        "{CONSENT_TYPE} 문서는 {LANGUAGE}로 작성하여 {ADDRESS}로 {DATE}까지 제출해야 합니다.",
        "{PROJECT_NAME}의 {DESCRIPTION}은(는) {COMPANY}가 {MONEY} 규모로 {PERIOD}간 수행합니다.",
        "{TYPE} 저작물의 {RIGHT_INFO}는 {CONTRACT_TYPE}에 따라 {PERIOD}간 {COMPANY}에 귀속됩니다.",
        "{NAME}은(는) {ID_NUM}으로 신원확인되며 {ADDRESS}에 거주하고 {COMPANY}에 재직합니다.",
        "{COMPANY} {DEPARTMENT}의 {PROJECT_NAME}는 {DATE}부터 시작하여 {MONEY} 예산으로 진행됩니다.",
        "{CONTRACT_TYPE}의 {TITLE}은(는) {DATE}에 작성되고 {PERIOD}간 {MONEY}를 {NAME}에게 지급합니다.",
        "{RIGHT_INFO}는 {LAW_REFERENCE}에 근거하여 {COMPANY}가 {PERIOD}간 {MONEY}로 매입합니다.",
        "{PROJECT_NAME} 관련 {TYPE} {QUANTITY}는 {DATE}까지 {COMPANY}가 납품해야 합니다.",
        "{NAME} {POSITION}이(가) {PHONE}과(와) {EMAIL}로 {COMPANY}의 {ADDRESS}에서 연락됩니다.",
        "{TITLE}은(는) {LANGUAGE}로 작성되며 {URL}에 게시되고 {DATE}에 업데이트됩니다.",
        "{COMPANY}는 {PROJECT_NAME}의 {CONTRACT_TYPE}을(를) {DATE}에 체결하여 {PERIOD}간 이행합니다.",
        "{CONSENT_TYPE}은(는) {NAME}으로부터 {DATE}에 취득하여 {COMPANY} {DEPARTMENT}에서 관리합니다.",
        "{RIGHT_INFO}의 대가로 {MONEY}를 {PERIOD}에 분할하여 {NAME}에게 지급하는 계약입니다.",
        "{PROJECT_NAME}의 {STATUS}는 {DATE} 현재 양호하며 {COMPANY}가 {ADDRESS}에서 진행합니다.",
        "{NAME}은(는) {COMPANY}의 {DEPARTMENT}에서 {POSITION}으로 {PROJECT_NAME}를 수행합니다.",
        "{TITLE} 계약서는 {CONTRACT_TYPE}으로 {DATE}에 체결되고 {MONEY}를 {PERIOD}간 지급합니다.",
        "{TYPE} {QUANTITY}의 {PROJECT_NAME}를 {COMPANY}가 {MONEY}에 수주하여 진행합니다.",
        "{LAW_REFERENCE}를 준거로 {RIGHT_INFO}의 {CONTRACT_TYPE}은(는) {PERIOD}간 {MONEY}로 체결됩니다.",
        "{COMPANY}는 {ADDRESS}에서 {PROJECT_NAME}를 담당하며 {DEPARTMENT}의 {EMAIL}로 문의하세요.",
        "{NAME}({ID_NUM})은(는) {CONSENT_TYPE}에 {DATE}에 동의하였고 {PHONE}로 연락 가능합니다.",
        "{PROJECT_NAME}의 {TYPE} 제작을 {COMPANY}가 {MONEY}로 수주하여 {QUANTITY}만큼 납품합니다.",
        "{RIGHT_INFO}는 {CONTRACT_TYPE}을(를) 통해 {DATE}에 {MONEY}로 {COMPANY}가 매입합니다.",
        "{TITLE}의 {LANGUAGE} 버전은 {URL}에서 확인 가능하며 {DATE}에 갱신되었습니다.",
        "{COMPANY} {DEPARTMENT}에서 {PROJECT_NAME}를 {PERIOD}간 진행하며 {NAME} {POSITION}이 담당합니다.",
        "{CONTRACT_TYPE}의 {TITLE}은(는) {LAW_REFERENCE}에 근거하여 {DATE}에 {MONEY}로 체결됩니다.",
        "{NAME}은(는) {ADDRESS}에 거주하며 {COMPANY}에서 {PERIOD}간 {POSITION}으로 근무합니다.",
        "{PROJECT_NAME}의 {STATUS}는 {DATE} 기준 완료되었으며 {COMPANY}가 {MONEY}를 지급했습니다.",
        "{CONSENT_TYPE} 서류를 {NAME}으로부터 {DATE}에 취득하여 {ADDRESS}에 보관합니다.",
        "{TYPE} 저작물의 {RIGHT_INFO}는 {LAW_REFERENCE}로 {PERIOD}간 보호되며 {COMPANY}에 귀속됩니다.",
        "{COMPANY}는 {PROJECT_NAME}에 {MONEY}를 투자하고 {DATE}부터 {PERIOD}간 수행합니다.",
        "{NAME} {POSITION}에게 {PHONE}과(와) {EMAIL}로 {TITLE} 관련 문의가 가능합니다.",
        "{CONTRACT_TYPE}에 따라 {RIGHT_INFO}는 {NAME}에서 {COMPANY}로 {DATE}에 {MONEY}로 이전됩니다.",
        "{PROJECT_NAME}의 {TYPE} {QUANTITY}는 {COMPANY}가 {DATE}까지 납품하기로 계약했습니다.",
        "{TITLE}은(는) {LANGUAGE}로 작성되고 {COMPANY}와 {NAME}이 {ADDRESS}에서 {DATE}에 체결했습니다.",
        "{DEPARTMENT}에서 {PROJECT_NAME}를 담당하며 {NAME} {POSITION}이(가) {PERIOD}간 전담합니다.",
        "{RIGHT_INFO}의 {CONTRACT_TYPE}은(는) {PERIOD}간 유효하며 {MONEY}를 {DATE}에 지급합니다.",
        "{COMPANY}는 {PROJECT_NAME}를 {MONEY} 예산으로 진행하며 {STATUS}는 {DATE} 현재 정상입니다.",
        "{NAME}은(는) {ID_NUM}으로 확인되며 {CONSENT_TYPE}에 {DATE}에 {COMPANY}에 동의했습니다.",
        "{TYPE} {QUANTITY}의 제작 비용은 {MONEY}이며 {COMPANY}가 {DATE}까지 완료합니다.",
        "{PROJECT_NAME} 관련 {DESCRIPTION}은(는) {COMPANY} {DEPARTMENT}에서 {LANGUAGE}로 작성됩니다.",
        "{CONTRACT_TYPE}의 {TITLE}은(는) {LAW_REFERENCE}를 준거법으로 {DATE}에 {PERIOD}간 체결됩니다.",
        "{COMPANY}는 {ADDRESS}에 본사를 두고 {PROJECT_NAME}를 {MONEY}로 진행합니다.",
        "{NAME} {POSITION}은(는) {PHONE}과(와) {EMAIL}로 {COMPANY} {DEPARTMENT}에서 연락됩니다.",
        "{RIGHT_INFO}는 {CONTRACT_TYPE}을(를) 통해 {PERIOD}간 {MONEY}로 {NAME}에게 허락됩니다.",
        "{TITLE} 문서는 {LANGUAGE}로 작성되며 {URL}에서 다운로드하고 {DATE}에 업데이트됩니다.",
        "{PROJECT_NAME}의 {TYPE} 제작 수량 {QUANTITY}는 {COMPANY}가 {MONEY}로 수주했습니다.",
        "{CONSENT_TYPE}은(는) {NAME}이(가) {DATE}에 제출하였고 {COMPANY}의 {DEPARTMENT}에 보관됩니다.",
        "{CONTRACT_TYPE}에 따라 {RIGHT_INFO}의 대가로 {MONEY}를 {PERIOD}에 걸쳐 {NAME}에게 지급합니다.",
        "{COMPANY} {DEPARTMENT}의 {PROJECT_NAME}는 {DATE}부터 {PERIOD}간 진행되며 {STATUS}는 양호합니다.",
        "{NAME}은(는) {ADDRESS}에 거주하며 {COMPANY}에서 {POSITION}으로 {PROJECT_NAME}를 담당합니다.",
        "{TITLE}의 {CONTRACT_TYPE}은(는) {DATE}에 체결되었고 {RIGHT_INFO}는 {LAW_REFERENCE}로 보호됩니다.",
        "{PROJECT_NAME}의 {TYPE} {QUANTITY}를 {COMPANY}가 {MONEY}에 제작하여 {DATE}까지 납품합니다.",
        "{RIGHT_INFO}는 {LAW_REFERENCE}에 근거하여 {PERIOD}간 {COMPANY}가 {MONEY}로 매입합니다.",
        "{NAME}({PHONE}, {EMAIL})은(는) {COMPANY}의 {ADDRESS}에서 {DEPARTMENT}에 소속되어 있습니다.",
        "{PROJECT_NAME}은(는) {MONEY} 규모로 {DATE}부터 {PERIOD}간 {COMPANY}가 수행합니다.",
        "{TITLE}은(는) {LANGUAGE}로 작성되며 {COMPANY}와 {NAME}이 {DATE}에 {MONEY}로 체결했습니다.",
        "{CONTRACT_TYPE}의 {STATUS}는 {DATE} 기준 진행중이며 {COMPANY} {DEPARTMENT}에서 관리합니다.",
        "{CONSENT_TYPE} 문서를 {NAME}으로부터 {DATE}에 취득하여 {ADDRESS}로 송부했습니다.",
        "{TYPE} 저작물의 {RIGHT_INFO}는 {CONTRACT_TYPE}에 따라 {PERIOD}간 {MONEY}로 계약됩니다.",
        "{COMPANY}는 {PROJECT_NAME}에 {MONEY}를 배정하고 {DATE}부터 {TYPE} {QUANTITY}를 제작합니다.",
        "{NAME} {POSITION}이(가) {COMPANY} {DEPARTMENT}에서 {PROJECT_NAME}를 {PERIOD}간 담당합니다.",
    ]
    
    # 엔티티 생성기 매핑 (이미 정의된 함수들을 매핑)
    entity_generators = {
        "NAME": random_name,
        "PHONE": random_phone,
        "COMPANY": random_company,
        "DATE": random_date,
        "MONEY": random_money,
        "ADDRESS": random_address,
        "EMAIL": random_email,
        "POSITION": random_position,
        "CONTRACT_TYPE": random_contract_type,
        "PERIOD": random_period,
        "ID_NUM": random_id_num,
        "CONSENT_TYPE": random_consent_type,
        "RIGHT_INFO": random_right_info,
        "PROJECT_NAME": random_project,
        "LAW_REFERENCE": random_law,
        "TITLE": random_title,
        "URL": random_url,
        "DESCRIPTION": random_description,
        "TYPE": random_type,
        "STATUS": random_status,
        "DEPARTMENT": random_department,
        "LANGUAGE": random_language,
        "QUANTITY": random_quantity
    }
    
    samples = []
    seen_texts = set()
    
    if balanced:
        # 균형잡힌 대규모 데이터 생성
        print(f"대규모 균형 데이터 생성 시작: {num_samples:,}개 샘플")
        samples_per_entity = max(1, math.ceil(num_samples / len(ENTITY_TYPES)))
        print(f"   → 엔티티당 {samples_per_entity:,}개씩 생성 (총 {len(ENTITY_TYPES)}개 타입)")
        
        # 템플릿 통합
        all_templates = build_template_list(single_entity_templates, dual_entity_templates, multi_entity_templates)
        print(f"   → 총 {len(all_templates):,}개 템플릿 사용 가능")
        
        # 각 엔티티 타입별로 샘플 생성
        for entity_type in tqdm(ENTITY_TYPES, desc="엔티티별 샘플 생성"):
            # 해당 엔티티를 포함하는 템플릿만 필터링
            relevant_templates = [(tmpl, entities) for tmpl, entities in all_templates if entity_type in entities]
            if not relevant_templates:
                relevant_templates = all_templates
            
            # 샘플 생성
            generated = 0
            attempts = 0
            max_attempts = samples_per_entity * 3
            
            while generated < samples_per_entity and attempts < max_attempts:
                attempts += 1
                template, _ = random.choice(relevant_templates)
                
                # 샘플 생성
                text, entity_list = generate_sample_from_template(template, entity_generators)
                
                # 중복 확인 및 추가
                if text not in seen_texts and entity_list:
                    seen_texts.add(text)
                    samples.append({"text": text, "entities": entity_list})
                    generated += 1
        
        # 요청 수보다 더 생성되었으면 무작위로 잘라냄
        if len(samples) > num_samples:
            random.shuffle(samples)
            samples = samples[:num_samples]
        
        print(f"생성 완료: {len(samples):,}개 샘플")
    
    else:
        # 비균형 모드
        print(f"비균형 모드로 {num_samples:,}개 샘플 생성")
        all_templates = build_template_list(single_entity_templates, dual_entity_templates, multi_entity_templates)
        
        for _ in range(num_samples):
            template, _ = random.choice(all_templates)
            text, entity_list = generate_sample_from_template(template, entity_generators)
            
            if entity_list:
                samples.append({"text": text, "entities": entity_list})
    
    return samples


def tokenize_and_align_labels(
    samples: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int = 256
) -> Dict:
    """
    토큰화 + BIO 태그 정렬
    
    핵심: offset_mapping으로 문자 위치 → 토큰 인덱스 매핑
    """
    
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    
    for sample in samples:
        text = sample['text']
        entities = sample['entities']
        
        # 토큰화
        tokenized = tokenizer(  # type: ignore[operator]
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True
        )
        
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        offset_mapping = tokenized['offset_mapping']
        
        # 엔티티 위치 매핑
        entity_positions = []
        for entity_text, entity_type in entities:
            # 엔티티 타입 검증
            if entity_type not in ENTITY_TYPES:
                continue
                
            start_pos = text.find(entity_text)
            if start_pos != -1:
                entity_positions.append({
                    'start': start_pos,
                    'end': start_pos + len(entity_text),
                    'type': entity_type
                })
        
        # BIO 태그 생성 (offset_mapping 길이에 맞춤)
        # CRF는 -100을 처리할 수 없으므로, 기본값을 'O'(0)로 설정
        # attention_mask가 0인 위치는 CRF가 자동으로 무시함
        labels = [LABEL_TO_ID['O']] * len(offset_mapping)  # 기본값 'O'로 초기화
        
        for idx, (start, end) in enumerate(offset_mapping):
            # 특수 토큰 또는 패딩 (start==end)
            # attention_mask가 0이면 CRF가 자동으로 무시하므로 'O' 유지
            if start == end:
                continue  # 'O' 그대로 유지
            
            # 일반 토큰: 엔티티 매칭
            matched = False
            for entity in entity_positions:
                # 토큰이 엔티티 범위에 포함되는지 확인
                token_in_entity = (entity['start'] <= start < entity['end']) or \
                                  (entity['start'] < end <= entity['end']) or \
                                  (start <= entity['start'] and end >= entity['end'])
                
                if token_in_entity:
                    # B-I-O 판단: 이전 토큰도 같은 엔티티인지 확인
                    is_begin = True
                    if idx > 0 and labels[idx-1] != LABEL_TO_ID['O']:  # 이전 라벨이 O가 아닌 경우
                        prev_label_id = labels[idx-1]
                        # 이전 토큰이 같은 엔티티 타입이면 I-
                        b_label_id = LABEL_TO_ID.get(f"B-{entity['type']}", -1)
                        i_label_id = LABEL_TO_ID.get(f"I-{entity['type']}", -1)
                        if prev_label_id in [b_label_id, i_label_id]:
                            is_begin = False
                    
                    if is_begin:
                        label_str = f"B-{entity['type']}"
                    else:
                        label_str = f"I-{entity['type']}"
                    
                    # 라벨이 존재하는지 확인
                    if label_str in LABEL_TO_ID:
                        labels[idx] = LABEL_TO_ID[label_str]
                        matched = True
                        break
            
            if not matched:
                labels[idx] = LABEL_TO_ID['O']
        
        # 검증: 모든 라벨이 유효한 범위인지 확인
        max_label_id = len(BIO_LABELS) - 1
        for i, label in enumerate(labels):
            if label != -100 and (label < 0 or label > max_label_id):
                print(f"경고: 유효하지 않은 라벨 인덱스 {label} 발견 (위치 {i}). O(0)로 변경.")
                labels[i] = LABEL_TO_ID['O']
        
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)
    
    return {
        'input_ids': all_input_ids,
        'attention_mask': all_attention_mask,
        'labels': all_labels
    }


# ========== 데이터셋 ==========

class ExponentialMovingAverage:
    """
    EMA (Exponential Moving Average) - 모델 파라미터의 이동평균
    과적합 방지 및 안정적인 예측
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 초기 파라미터 복사
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """매 스텝마다 호출: shadow = decay * shadow + (1 - decay) * param"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:  # 안전 검사
                    new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                    self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """평가 시 호출: 현재 파라미터를 백업하고 shadow로 교체"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """평가 후 호출: shadow를 제거하고 원래 파라미터로 복원"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}


class LossFilter:
    """
    Loss Smoothing Filter
    - Process Variance: 모델의 자연스러운 변동
    - Measurement Variance: Loss 측정의 노이즈
    - 급격한 loss spike를 완화하여 안정적인 학습
    """
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2):
        self.process_variance = process_variance  # Q: 프로세스 노이즈
        self.measurement_variance = measurement_variance  # R: 측정 노이즈
        self.estimate = None  # 추정값 (평활화된 loss)
        self.error_covariance = 1.0  # 추정 오차 공분산
    
    def update(self, measurement):
        """
        Filter 업데이트
        
        Args:
            measurement: 현재 측정된 loss 값
        
        Returns:
            smoothed_loss: 평활화된 loss 값
        """
        if self.estimate is None:
            # 초기화: 첫 측정값으로 시작
            self.estimate = measurement
            return measurement
        
        # Prediction step
        predicted_estimate = self.estimate
        predicted_error_cov = self.error_covariance + self.process_variance
        
        # Update step
        gain = predicted_error_cov / (predicted_error_cov + self.measurement_variance)
        self.estimate = predicted_estimate + gain * (measurement - predicted_estimate)
        self.error_covariance = (1 - gain) * predicted_error_cov
        
        return self.estimate


class AdaptiveGradientClipper(TrainerCallback):
    """
    Adaptive Gradient Clipping (AGC)
    - Gradient norm이 비정상적으로 클 때만 clipping
    - 정상 범위 내에서는 자유로운 학습 허용
    - Percentile 기반 동적 threshold 계산
    """
    def __init__(self, percentile=10.0, max_grad_norm=1.0, warmup_steps=100):
        self.percentile = percentile  # 상위 10% gradient만 clip
        self.max_grad_norm = max_grad_norm  # 최대 norm
        self.warmup_steps = warmup_steps  # 초기에는 고정 clipping
        self.grad_norms = []  # Gradient norm 히스토리
        self.dynamic_threshold = max_grad_norm
        self.step_count = 0
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """매 스텝 후 gradient clipping"""
        if model is None:
            return
        
        self.step_count += 1
        
        # 현재 gradient norm 계산
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # 히스토리에 추가
        self.grad_norms.append(total_norm)
        if len(self.grad_norms) > 1000:  # 최근 1000개만 유지
            self.grad_norms.pop(0)
        
        # Warmup 기간에는 고정 clipping
        if self.step_count <= self.warmup_steps:
            threshold = self.max_grad_norm
        else:
            # Percentile 기반 동적 threshold
            if len(self.grad_norms) >= 50:  # 충분한 데이터 수집 후
                self.dynamic_threshold = np.percentile(self.grad_norms, 100 - self.percentile)
                threshold = min(self.dynamic_threshold, self.max_grad_norm * 2)  # 최대 2배까지
            else:
                threshold = self.max_grad_norm
        
        # Clipping 수행
        if total_norm > threshold:
            clip_coef = threshold / (total_norm + 1e-6)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
            
            # 경고 출력 (너무 자주 나오지 않도록)
            if self.step_count % 100 == 0:
                print(f"\n[WARNING] Gradient clipped: {total_norm:.4f} → {threshold:.4f} (step {self.step_count})")


class AdvancedMetricsCallback(TrainerCallback):
    """
    고급 메트릭 추적 콜백
    - Best-so-far F1 추적
    - EMA 메트릭 추적
    - Precision spike detection
    - Loss Filter Smoothing
    - Save history for graphing
    """
    def __init__(self, enable_loss_smoothing=True):
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_loss_smoothed': [],  # Loss filter smoothed
            'eval_loss': [],
            'eval_loss_smoothed': [],  # Loss filter smoothed
            'eval_f1': [],
            'eval_precision': [],
            'eval_recall': [],
            'best_f1': [],  # Best-so-far
            'precision_spikes': []
        }
        self.best_f1 = 0.0
        self.last_precision = None
        self.spike_count = 0
        
        # Loss Filter for Loss Smoothing
        self.enable_smoothing = enable_loss_smoothing
        if self.enable_smoothing:
            self.train_loss_filter = LossFilter(
                process_variance=1e-5,  # 작은 값: 부드러운 변화
                measurement_variance=1e-2  # 큰 값: 측정 노이즈 크다고 가정
            )
            self.eval_loss_filter = LossFilter(
                process_variance=1e-5,
                measurement_variance=1e-2
            )
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics is None:
            return
        
        epoch = state.epoch
        eval_f1 = metrics.get('eval_f1', 0.0)
        eval_precision = metrics.get('eval_precision', 0.0)
        eval_recall = metrics.get('eval_recall', 0.0)
        eval_loss = metrics.get('eval_loss', 0.0)
        
        # Loss Filter로 eval loss 평활화
        if self.enable_smoothing:
            eval_loss_smoothed = self.eval_loss_filter.update(eval_loss)
        else:
            eval_loss_smoothed = eval_loss
        
        # Update Best-so-far F1 (monotonic non-decreasing)
        if eval_f1 > self.best_f1:
            self.best_f1 = eval_f1
        
        # Precision spike detection (Delta >= 0.25)
        precision_spike = False
        if self.last_precision is not None:
            precision_delta = abs(eval_precision - self.last_precision)
            if precision_delta >= 0.25:
                precision_spike = True
                self.spike_count += 1
                print(f"\n[WARNING] Precision spike detected! Epoch {epoch:.1f}: Delta = {precision_delta:.3f}")
        
        self.last_precision = eval_precision
        
        # Save history
        self.history['epoch'].append(epoch)
        self.history['eval_loss'].append(eval_loss)
        self.history['eval_loss_smoothed'].append(eval_loss_smoothed)
        self.history['eval_f1'].append(eval_f1)
        self.history['eval_precision'].append(eval_precision)
        self.history['eval_recall'].append(eval_recall)
        self.history['best_f1'].append(self.best_f1)
        self.history['precision_spikes'].append(1 if precision_spike else 0)
        
        # Print progress (smoothed loss 표시)
        loss_change = ""
        if len(self.history['eval_loss']) > 1:
            raw_delta = eval_loss - self.history['eval_loss'][-2]
            smoothed_delta = eval_loss_smoothed - self.history['eval_loss_smoothed'][-2]
            loss_change = f" | ΔLoss: {raw_delta:+.4f} (smoothed: {smoothed_delta:+.4f})"
        
        print(f"\n[Epoch {epoch:.1f}] "
              f"F1: {eval_f1:.4f} (Best: {self.best_f1:.4f}) | "
              f"P: {eval_precision:.4f} | "
              f"R: {eval_recall:.4f} | "
              f"Loss: {eval_loss:.4f} (smoothed: {eval_loss_smoothed:.4f}){loss_change} | "
              f"Spikes: {self.spike_count}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """로깅 시 호출 (train loss 추적)"""
        if logs and 'loss' in logs:
            train_loss = logs['loss']
            
            # Loss Filter로 train loss 평활화
            if self.enable_smoothing:
                train_loss_smoothed = self.train_loss_filter.update(train_loss)
            else:
                train_loss_smoothed = train_loss
            
            # Train loss는 step 단위로 오므로 마지막 값만 저장
            if len(self.history['train_loss']) < len(self.history['epoch']):
                self.history['train_loss'].append(train_loss)
                self.history['train_loss_smoothed'].append(train_loss_smoothed)


class NERDataset(Dataset):
    """PyTorch Dataset for NER"""
    
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.encodings['labels'][idx])
        }


# ========== 평가 ==========

def bio_to_entities(bio_tags):
    """
    BIO 태그 → 엔티티 span 변환
    
    Args:
        bio_tags: ['O', 'B-NAME', 'I-NAME', 'O', 'B-ADDRESS', 'I-ADDRESS', ...]
    
    Returns:
        set: {(start, end, type), ...}
        예: {(1, 3, 'NAME'), (4, 6, 'ADDRESS')}
    """
    entities = []
    current_entity = None
    
    for idx, tag in enumerate(bio_tags):
        if tag == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
        elif tag.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            entity_type = tag[2:]  # 'B-NAME' -> 'NAME'
            current_entity = {'start': idx, 'end': idx + 1, 'type': entity_type}
        elif tag.startswith('I-'):
            entity_type = tag[2:]
            if current_entity and current_entity['type'] == entity_type:
                current_entity['end'] = idx + 1
            else:
                # I-태그가 단독으로 나타남 (B- 없이)
                # 새로운 엔티티로 간주
                if current_entity:
                    entities.append(current_entity)
                current_entity = {'start': idx, 'end': idx + 1, 'type': entity_type}
    
    if current_entity:
        entities.append(current_entity)
    
    # set으로 변환 (중복 제거 + 비교 용이)
    return set((e['start'], e['end'], e['type']) for e in entities)


def compute_metrics(pred):
    """
    엔티티 단위(Span-level) F1, Precision, Recall 계산
    
    - 학습: BIO 태그로 학습 (B-NAME, I-NAME, ...)
    - 평가: 엔티티 전체 매칭 (span + type)
    - 예시: "대한민국 서울시" → (start, end, 'ADDRESS') 정확히 일치해야 함
    """
    predictions, labels = pred
    
    # CRF 예측 결과 처리
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Argmax (Softmax 출력인 경우)
    if len(predictions.shape) == 3:
        predictions = np.argmax(predictions, axis=2)
    
    # 엔티티 span 추출
    true_entities_list = []
    pred_entities_list = []
    
    for i in range(len(labels)):
        # 유효한 토큰만 필터링 (padding 제외: label != -100)
        true_tags = []
        pred_tags = []
        
        for j in range(len(labels[i])):
            # Padding 토큰 제외 (label == -100인 경우)
            if labels[i][j] != -100:
                true_tags.append(ID_TO_LABEL[labels[i][j]])
                pred_tags.append(ID_TO_LABEL[predictions[i][j]])
        
        # BIO → 엔티티 변환
        true_entities = bio_to_entities(true_tags)
        pred_entities = bio_to_entities(pred_tags)
        
        true_entities_list.append(true_entities)
        pred_entities_list.append(pred_entities)
    
    # Span-level Precision, Recall, F1
    tp = 0  # True Positive
    fp = 0  # False Positive
    fn = 0  # False Negative
    
    for true_ents, pred_ents in zip(true_entities_list, pred_entities_list):
        tp += len(true_ents & pred_ents)  # 교집합
        fp += len(pred_ents - true_ents)  # 예측만 있음
        fn += len(true_ents - pred_ents)  # 정답만 있음
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


# ========== 메인 학습 함수 ==========

def train_ner_model(
    model_name: str = "google-bert/bert-base-multilingual-cased",
    num_samples: int = 30000,         # 기본값: 30,000 (대규모 데이터)
    num_epochs: int = 20,            # 기본값: 20 (충분한 학습)
    batch_size: int = 12,             # 기본값: 12 (메모리 효율)
    learning_rate: float = 2e-5,      # 기본값: 2e-5 (안정적 학습)
    output_dir: Optional[str] = None,
    use_gpu: bool = True,
    use_realistic_data: bool = True,  # 실전 기반 데이터 사용
    enable_early_stopping: bool = False  # Early stopping 활성화 여부
):
    """
    NER 모델 학습 (재설계)
    
    주요 변경사항:
    - 실전 기반 데이터 우선 사용 (긴 문장, 복잡한 패턴)
    - Train/Val 완전 분리 (다른 문장 패턴)
    - Epoch마다 데이터 셔플 (과적합 방지)
    - 실전 테스트 샘플로 검증
    
    Args:
        model_name: HuggingFace 모델명
        num_samples: 학습 샘플 수
        num_epochs: 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
        output_dir: 모델 저장 경로
        use_gpu: GPU 사용 여부
        use_realistic_data: 실전 기반 데이터 사용 (기본 True)
        enable_early_stopping: Early stopping 활성화 (기본 False)
    """
    
    print("=" * 80)
    print("Token-level BERT-CRF NER 학습 시작")
    print("=" * 80)
    
    # 1. 설정
    config = Config(
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # output_dir 설정 및 Path 객체로 변환
    output_dir_path: Path
    if output_dir is None:
        model_safe_name = model_name.replace('/', '-')
        output_dir_path = Path(f"models/ner/{model_safe_name}")
    else:
        output_dir_path = Path(output_dir)
    
    ensure_dir(output_dir_path)
    
    print(f"\n설정:")
    print(f"   - 모델: {config.model_name}")
    print(f"   - 에포크: {config.num_epochs}")
    print(f"   - 배치 크기: {config.batch_size}")
    print(f"   - 학습률: {config.learning_rate}")
    print(f"   - 샘플 수: {num_samples}")
    
    # 2. 토크나이저 로드
    print(f"\n토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 3. 데이터 로드 (최적화된 BIO 데이터 사용)
    print(f"\n학습 데이터 로드 중...")
    
    # 모델명을 경로에 반영
    model_name_safe = config.model_name.replace('/', '-')
    train_data_path = Path(f"module/ner/training/{model_name_safe}/train.txt")
    test_data_path = Path(f"module/ner/training/{model_name_safe}/test.txt")

    if train_data_path.exists() and test_data_path.exists():
        print(f"   OK: 최적화 BIO 데이터 로드 ({model_name_safe})")
        print(f"      훈련: {train_data_path}")
        print(f"      평가: {test_data_path}")
        
        train_samples = load_bio_file(str(train_data_path))
        val_samples = load_bio_file(str(test_data_path))
        
        print(f"   훈련 샘플: {len(train_samples):,}개")
        print(f"   평가 샘플: {len(val_samples):,}개")
        print(f"   → 사전 분리된 데이터 사용 (재분할 안함)")
        
    else:
        # 대체 데이터 경로
        realistic_data_path = Path("data/in/ner/realistic_training_data.txt")
        large_data_path = Path("data/in/ner/large_training_data.txt")
        
        if use_realistic_data and realistic_data_path.exists():
            print(f"   실전 기반 데이터 로드: {realistic_data_path}")
            samples_loaded = load_bio_file(str(realistic_data_path))
            print(f"   전체 샘플: {len(samples_loaded):,}개")
            
            # 요청된 샘플 수만큼 무작위 선택
            if len(samples_loaded) > num_samples:
                random.shuffle(samples_loaded)
                samples = samples_loaded[:num_samples]
                print(f"   {num_samples:,}개 샘플 선택 (실전 데이터)")
            else:
                samples = samples_loaded
                
        elif large_data_path.exists():
            print(f"   대량 BIO 데이터 로드: {large_data_path}")
            samples_loaded = load_bio_file(str(large_data_path))
            print(f"   전체 샘플: {len(samples_loaded):,}개")
            
            # 요청된 샘플 수만큼 무작위 선택
            if len(samples_loaded) > num_samples:
                random.shuffle(samples_loaded)
                samples = samples_loaded[:num_samples]
                print(f"   {num_samples:,}개 샘플 선택")
            else:
                samples = samples_loaded
        else:
            print(f"   경고: 훈련 데이터 없음. 템플릿 데이터 생성 중...")
            samples = generate_training_samples(num_samples, balanced=True)
        
        # Train/Val split (80/20) with stratified sampling
        print(f"\n   엔티티별 균등 분배로 Train/Val split 수행...")
        
        # 엔티티 타입별로 샘플 그룹화
        entity_type_samples = {entity_type: [] for entity_type in ENTITY_TYPES}
        no_entity_samples = []
        
        for sample in samples:
            # 샘플에 포함된 엔티티 타입 확인
            entity_types_in_sample = set()
            for _, etype in sample.get('entities', []):
                if etype in ENTITY_TYPES:
                    entity_types_in_sample.add(etype)
            
            if entity_types_in_sample:
                # 첫 번째 엔티티 타입으로 분류
                primary_type = list(entity_types_in_sample)[0]
                entity_type_samples[primary_type].append(sample)
            else:
                no_entity_samples.append(sample)
        
        # 각 엔티티 타입별 80/20 분할
        train_samples = []
        val_samples = []
        
        for entity_type in ENTITY_TYPES:
            type_samples = entity_type_samples[entity_type]
            if len(type_samples) > 0:
                # 셔플 후 80/20 분할
                random.shuffle(type_samples)
                split_idx = int(len(type_samples) * 0.8)
                train_samples.extend(type_samples[:split_idx])
                val_samples.extend(type_samples[split_idx:])
        
        # 엔티티 없는 샘플도 분할
        if no_entity_samples:
            random.shuffle(no_entity_samples)
            split_idx = int(len(no_entity_samples) * 0.8)
            train_samples.extend(no_entity_samples[:split_idx])
            val_samples.extend(no_entity_samples[split_idx:])
        
        # 최종 셔플
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        
        print(f"   데이터 분할 완료: Train {len(train_samples):,}개 / Val {len(val_samples):,}개")
    
    # 4. 토큰화
    print(f"\n토큰화 중...")
    train_encodings = tokenize_and_align_labels(train_samples, tokenizer, config.max_length)
    val_encodings = tokenize_and_align_labels(val_samples, tokenizer, config.max_length)
    
    # 간단한 라벨 검증
    all_labels_flat = [label for labels in train_encodings['labels'] for label in labels if label != -100]
    if all_labels_flat:
        invalid_count = sum(1 for l in all_labels_flat if l < 0 or l >= len(BIO_LABELS))
        if invalid_count > 0:
            print(f"   [경고] 유효하지 않은 라벨 {invalid_count}개 발견")
    
    train_dataset = NERDataset(train_encodings)
    val_dataset = NERDataset(val_encodings)
    
    print(f"   ✅ 토큰화 완료")
    
    # 5. 모델 초기화 또는 로드
    print(f"\n모델 초기화 중...")
    model = BertCrfForNER(
        model_name=config.model_name,
        num_labels=len(BIO_LABELS),
        dropout=config.dropout
    )
    
    # K-Fold 모드에서는 각 fold마다 새로운 모델로 학습 (독립적 평가)
    # 기존 모델 로드 여부 확인 (이어 학습)
    model_pt_path = output_dir_path / "model.pt"
    is_continued_training = False
    
    if model_pt_path.exists():
        print(f"   기존 모델 로드 (이어 학습)")
        try:
            state_dict = torch.load(model_pt_path, map_location='cpu')
            model.load_state_dict(state_dict)
            is_continued_training = True
        except Exception as e:
            print(f"   [경고] 모델 로드 실패: {e}")
            print(f"   새 모델로 시작")
    else:
        print(f"   새 모델로 시작")
    
    # 학습률 조정 (이어 학습 시 감소)
    adjusted_learning_rate = learning_rate
    if is_continued_training:
        adjusted_learning_rate = learning_rate * 0.3  # 이어 학습 시 30%로 감소
        print(f"   학습률 조정: {learning_rate:.2e} → {adjusted_learning_rate:.2e}")
    
    # 6. Training Arguments
    print(f"\n학습 설정 구성 중...")
    
    # Layer-wise Learning Rate Decay (BERT 하위층은 작은 LR)
    optimizer_grouped_parameters = []
    num_layers = model.bert.config.num_hidden_layers
    
    # BERT Embeddings (가장 낮은 LR)
    optimizer_grouped_parameters.append({
        'params': [p for n, p in model.bert.embeddings.named_parameters() if p.requires_grad],
        'lr': adjusted_learning_rate * (config.layer_lr_decay ** num_layers)
    })
    
    # BERT Encoder Layers (Layer-wise decay)
    for layer_idx in range(num_layers):
        layer_lr = adjusted_learning_rate * (config.layer_lr_decay ** (num_layers - layer_idx - 1))
        optimizer_grouped_parameters.append({
            'params': [p for n, p in model.bert.encoder.layer[layer_idx].named_parameters() if p.requires_grad],
            'lr': layer_lr
        })
    
    # Classifier와 CRF는 원래 학습률 사용 (가장 높은 LR)
    classifier_params = []
    for n, p in model.named_parameters():
        if ('classifier' in n or 'crf' in n) and p.requires_grad:
            classifier_params.append(p)
    
    if classifier_params:
        optimizer_grouped_parameters.append({
            'params': classifier_params,
            'lr': adjusted_learning_rate
        })
    
    training_args = TrainingArguments(
        output_dir=str(output_dir_path),
        num_train_epochs=config.num_epochs,  # 300 epochs
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=adjusted_learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,  # 8% warmup
        lr_scheduler_type="cosine_with_restarts",  # Cosine Annealing with Restarts
        max_grad_norm=config.max_grad_norm,  # Gradient Clipping
        label_smoothing_factor=config.label_smoothing,  # Label Smoothing
        logging_strategy="epoch",  # 매 에포크 로깅
        eval_strategy="epoch",  # 매 에포크 평가
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",
        disable_tqdm=False,
        fp16=torch.cuda.is_available(),  # Mixed Precision (GPU에서만)
        use_cpu=not use_gpu,
        dataloader_num_workers=0,  # Windows 호환성
        gradient_accumulation_steps=1,  # 필요시 증가
    )
    
    print(f"   학습 설정 완료:")
    print(f"      - Scheduler: cosine_with_restarts")
    print(f"      - Label Smoothing: {config.label_smoothing}")
    print(f"      - FP16: {training_args.fp16}")
    print(f"      - Gradient Clipping: {config.max_grad_norm}")
    
    # 학습 설정 완료 (간소화)
    print(f"   설정 완료")
    
    # 7. Trainer (고급 콜백 + EMA + AGC + Loss Filter)
    advanced_callback = AdvancedMetricsCallback(
        enable_loss_smoothing=config.enable_loss_smoothing
    )
    callbacks: List[TrainerCallback] = [advanced_callback]  # 고급 메트릭 추적
    
    # Adaptive Gradient Clipping 추가
    agc_callback = None
    if config.adaptive_grad_clip:
        agc_callback = AdaptiveGradientClipper(
            percentile=config.agc_percentile,
            max_grad_norm=config.max_grad_norm,
            warmup_steps=100
        )
        callbacks.append(agc_callback)
    
    if not enable_early_stopping:
        # Early stopping 비활성화 메시지만 표시
        pass
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=(torch.optim.AdamW(optimizer_grouped_parameters, 
                                      lr=adjusted_learning_rate,
                                      weight_decay=config.weight_decay), None)
    )
    
    # EMA 초기화
    ema = ExponentialMovingAverage(model, decay=config.ema_decay)
    
    # 8. 학습
    print(f"\n{'='*80}")
    print(f"학습 시작 ({config.num_epochs} Epochs)")
    print(f"{'='*80}")
    print(f"Device: {trainer.args.device} | Epochs: {config.num_epochs} | Batch: {config.batch_size} | LR: {adjusted_learning_rate:.2e}")
    print(f"{'='*80}\n")
    
    # 커스텀 학습 루프 (EMA 업데이트 포함)
    class EMATrainerCallback(TrainerCallback):
        """EMA 업데이트 콜백"""
        def __init__(self, ema_model):
            self.ema = ema_model
        
        def on_step_end(self, args, state, control, **kwargs):
            """매 스텝 후 EMA 업데이트"""
            self.ema.update()
    
    trainer.add_callback(EMATrainerCallback(ema))
    
    trainer.train()
    
    # 9. 최종 평가 (EMA 모델, 에포크에서 제외)
    print(f"\n최종 평가 (EMA 모델)...")
    
    # 평가 데이터셋 준비
    eval_dataset = NERDataset(val_encodings)
    
    # 직접 평가 (콜백 없이)
    model.eval()
    ema.apply_shadow()  # EMA 파라미터 적용
    
    # 모델이 위치한 디바이스 확인
    device = next(model.parameters()).device
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in trainer.get_eval_dataloader(eval_dataset):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # labels 제거 (추론 모드)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # 모델 추론 (labels 없이)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # outputs 형식 확인: dict with 'predictions' or 'logits'
            if isinstance(outputs, dict):
                if 'predictions' in outputs:
                    # CRF Viterbi decoding 결과
                    predictions = outputs['predictions']
                elif 'logits' in outputs:
                    # logits에서 argmax (학습 중 평가 시)
                    predictions = torch.argmax(outputs['logits'], dim=-1)
                else:
                    raise ValueError(f"Unexpected dict keys: {outputs.keys()}")
            else:
                raise ValueError(f"Unexpected output type: {type(outputs)}")
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    ema.restore()  # 원래 파라미터 복원
    
    # 메트릭 계산
    final_eval_metrics = compute_metrics((np.array(all_preds), np.array(all_labels)))
    
    # 최종 평가 로그 저장
    eval_log_dir = ensure_dir(Path("data/out/ner/training"))
    
    # 모델명에서 / 제거
    safe_model_name = model_name.replace("/", "-")
    eval_log_path = eval_log_dir / f"{safe_model_name}_evaluation_log.json"
    
    eval_log = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "final_ema_evaluation": {
            "f1": float(final_eval_metrics['f1']),
            "precision": float(final_eval_metrics['precision']),
            "recall": float(final_eval_metrics['recall']),
            "tp": int(final_eval_metrics['tp']),
            "fp": int(final_eval_metrics['fp']),
            "fn": int(final_eval_metrics['fn'])
        },
        "best_during_training": {
            "f1": float(advanced_callback.best_f1),
            "precision": float(max(advanced_callback.history['eval_precision'])),
            "recall": float(max(advanced_callback.history['eval_recall']))
        },
        "precision_spike_count": advanced_callback.spike_count,
        "total_epochs": config.num_epochs,
        "num_samples": len(train_samples)
    }
    
    save_json(eval_log, eval_log_path)
    
    print(f"   최종 평가 로그 저장: {eval_log_path}")
    print(f"      - F1: {final_eval_metrics['f1']:.4f}")
    print(f"      - Precision: {final_eval_metrics['precision']:.4f}")
    print(f"      - Recall: {final_eval_metrics['recall']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"학습 완료!")
    print(f"{'='*80}")
    print(f"Best F1: {advanced_callback.best_f1:.4f} | Precision: {max(advanced_callback.history['eval_precision']):.4f} | Recall: {max(advanced_callback.history['eval_recall']):.4f}")
    print(f"{'='*80}\n")
    
    # 10. 저장
    print(f"모델 저장 중...")
    
    # 전체 모델 가중치 저장 (BERT + Classifier + CRF) - EMA 파라미터로
    ema.apply_shadow()
    torch.save(model.state_dict(), output_dir_path / "model.pt")
    ema.restore()
    
    # HuggingFace 호환성을 위해 BERT도 별도 저장
    model.bert.save_pretrained(output_dir_path)
    tokenizer.save_pretrained(output_dir_path)
    
    # Label map 저장
    save_json({"id2label": ID_TO_LABEL, "label2id": LABEL_TO_ID}, output_dir_path / "label_map.json")
    
    # 학습 히스토리 저장 (고급 메트릭 포함)
    history_metrics = {
        'history': advanced_callback.history,
        'best_f1': advanced_callback.best_f1,
        'precision_spike_count': advanced_callback.spike_count
    }
    
    # 히스토리 JSON 저장
    history_path = output_dir_path / "training_history.json"
    history_serializable = {k: [float(x) if isinstance(x, (np.floating, np.integer)) else x 
                                 for x in v] 
                           for k, v in advanced_callback.history.items()}
    save_json(history_serializable, history_path)
    
    print(f"저장 완료: {output_dir_path}\n")
    
    return model, tokenizer, history_metrics


# ========== 기존 호환성 래퍼 함수 ==========

def ner_train(
    model_name: str = "google-bert/bert-base-multilingual-cased",
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
    기존 ner_system.py와 호환되는 래퍼 함수
    
    Args:
        model_name: 훈련할 모델 이름 (예: "klue/roberta-large")
        iterations: 훈련 반복 횟수
        epochs: 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
        num_train_samples: 생성할 샘플 수
        enable_visualization: 시각화 여부 (현재 미지원, 추후 추가)
        enable_early_stopping: Early stopping 활성화 (기본 False)
        debug: 디버그 모드
        
    Returns:
        Dict[str, Any]: {
            'model_name': str,
            'epochs': int,
            'samples': int,
            'metrics': {...},
            'status': 'success'
        }
    """
    
    print(f"\n{'='*80}")
    print(f"NER 모델 훈련 시작 (Token-level + BERT-CRF)")
    print(f"{'='*80}")
    
    all_results = []
    last_model = None
    last_tokenizer = None
    val_samples = None
    viz_dir = None
    
    for iteration in range(iterations):
        if iterations > 1:
            print(f"\n반복 {iteration + 1}/{iterations}")
        
        # 검증 샘플 생성 (시각화용, 한 번만)
        if val_samples is None:
            all_samples = generate_training_samples(num_train_samples, balanced=True)
            split_idx = int(len(all_samples) * 0.8)
            val_samples = all_samples[split_idx:]
        
        # 모델 학습
        model, tokenizer, metrics = train_ner_model(
            model_name=model_name,
            num_samples=num_train_samples,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=None,  # 자동 생성
            enable_early_stopping=enable_early_stopping
        )
        
        last_model = model
        last_tokenizer = tokenizer
        
        all_results.append({
            'iteration': iteration + 1,
            'metrics': metrics
        })
    
    # 시각화 (enable_visualization=True일 때)
    if enable_visualization and 'history' in all_results[-1]['metrics']:
        print(f"\n{'='*80}")
        print(f"시각화 생성 중...")
        print(f"{'='*80}")
        
        model_safe_name = model_name.replace('/', '-')
        iteration_idx = all_results[-1]['iteration']
        viz_dir = ensure_dir(Path("data/out/ner_visualization") / model_safe_name / str(iteration_idx))
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        history = all_results[-1]['metrics']['history']
        
        # 학습 곡선 시각화 (1장만)
        try:
            plot_training_curves(
                history,
                viz_dir / f"{model_safe_name}_{timestamp}.png",
                model_name
            )
            print(f"\n시각화 완료! 저장 위치: {viz_dir}")
        except Exception as e:
            print(f"경고: 학습 곡선 생성 실패: {e}")
    
    # 결과 포맷 (기존 호환)
    final_metrics = all_results[-1]['metrics']
    
    return {
        'model_name': model_name,
        'epochs': epochs,
        'samples': num_train_samples,
        'iterations': iterations,
        'metrics': {
            'f1': final_metrics.get('eval_f1', 0),
            'precision': final_metrics.get('eval_precision', 0),
            'recall': final_metrics.get('eval_recall', 0),
            'val_loss': final_metrics.get('eval_loss', 0)
        },
        'all_iterations': all_results,
        'status': 'success',
        'model_type': 'bert-crf',  # 새로운 필드
        'architecture': 'token-level-bio',  # 새로운 필드
        'visualization_dir': str(viz_dir) if enable_visualization else None
    }


# ========== 기존 ner_system.py 호환성 함수 ==========

def generate_rich_training_data(output_dir: Path, num_samples: int = 7500):
    """
    대규모 균형 훈련 데이터 생성
    - 모든 엔티티 타입에 대해 균형있게 생성
    - 엔티티당 충분한 샘플 확보
    
    Args:
        output_dir: 출력 디렉토리 (train.txt, validation.txt 생성)
        num_samples: 총 샘플 수 (권장: 10,000 ~ 500,000)
    
    Returns:
        bool: 성공 여부
    """
    try:
        print(f"\n{'='*70}")
        print(f"대규모 훈련 데이터 생성 시작")
        print(f"{'='*70}")
        print(f"목표 샘플 수: {num_samples:,}개")
        print(f"엔티티 타입: {len(ENTITY_TYPES)}개")
        print(f"  → {', '.join(ENTITY_TYPES[:5])}... (외 {len(ENTITY_TYPES)-5}개)")
        
        # 균형잡힌 대규모 샘플 생성
        print(f"\n[1/3] 샘플 생성 중...")
        samples = generate_training_samples(num_samples, balanced=True)
        
        # 엔티티별 통계
        entity_counts = {etype: 0 for etype in ENTITY_TYPES}
        for sample in samples:
            for _, etype in sample['entities']:
                if etype in entity_counts:
                    entity_counts[etype] += 1
        
        print(f"\n생성된 샘플 통계:")
        print(f"  총 샘플: {len(samples):,}개")
        print(f"  엔티티별 분포:")
        sorted_counts = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        for etype, count in sorted_counts[:10]:
            print(f"    - {etype:15s}: {count:6,}개")
        if len(sorted_counts) > 10:
            print(f"    ... (외 {len(sorted_counts)-10}개 타입)")
        
        # 80/20 분할
        print(f"\n[2/3] 데이터 분할 중...")
        train_size = int(len(samples) * 0.8)
        train_samples = samples[:train_size]
        val_samples = samples[train_size:]
        
        print(f"  → Train: {len(train_samples):,}개 (80%)")
        print(f"  → Validation: {len(val_samples):,}개 (20%)")
        
        # BIO 포맷으로 저장
        print(f"\n[3/3] 파일 저장 중...")
        output_dir = ensure_dir(Path(output_dir))
        
        train_file = output_dir / "train.txt"
        val_file = output_dir / "validation.txt"
        
        # train.txt 저장
        print(f"  → {train_file}")
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(train_samples, desc="Train 저장", ncols=80):
                text = sample['text']
                entities = sample['entities']
                
                # 텍스트를 문자로 분리
                chars = list(text)
                labels = ['O'] * len(chars)
                
                # 엔티티 라벨링
                for entity_text, entity_type in entities:
                    start = text.find(entity_text)
                    if start != -1:
                        end = start + len(entity_text)
                        labels[start] = f'B-{entity_type}'
                        for i in range(start + 1, end):
                            labels[i] = f'I-{entity_type}'
                
                # BIO 포맷 출력
                for char, label in zip(chars, labels):
                    f.write(f"{char}\t{label}\n")
                f.write("\n")  # 샘플 구분
        
        # validation.txt 저장
        print(f"  → {val_file}")
        with open(val_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(val_samples, desc="Validation 저장", ncols=80):
                text = sample['text']
                entities = sample['entities']
                
                chars = list(text)
                labels = ['O'] * len(chars)
                
                for entity_text, entity_type in entities:
                    start = text.find(entity_text)
                    if start != -1:
                        end = start + len(entity_text)
                        labels[start] = f'B-{entity_type}'
                        for i in range(start + 1, end):
                            labels[i] = f'I-{entity_type}'
                
                for char, label in zip(chars, labels):
                    f.write(f"{char}\t{label}\n")
                f.write("\n")
        
        print(f"\n{'='*70}")
        print(f"✓ 훈련 데이터 생성 완료!")
        print(f"{'='*70}")
        print(f"파일 크기:")
        print(f"  - {train_file.name}: {train_file.stat().st_size // (1024*1024):.1f} MB")
        print(f"  - {val_file.name}: {val_file.stat().st_size // (1024*1024):.1f} MB")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: 데이터 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========== 시각화 함수 ==========

def plot_training_curves(history: Dict[str, List], save_path: Path, model_name: str):
    """
    고급 Training curve visualization (현재값, Best-so-far, EMA)
    
    Args:
        history: Training history with advanced metrics
        save_path: Save path
        model_name: Model name
    """
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle(f'NER Training Results (300 Epochs) - {model_name}', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    epochs = history['epoch']
    
    # 1. Loss curve
    ax1 = fig.add_subplot(gs[0, 0])
    if history.get('train_loss'):
        ax1.plot(epochs[:len(history['train_loss'])], history['train_loss'], 
                'b-', alpha=0.5, label='Train Loss', linewidth=1.5)
    ax1.plot(epochs, history['eval_loss'], 'r-', alpha=0.7, 
            label='Val Loss', linewidth=2.0)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training/Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. F1 Score (2 lines: Current, Best-so-far)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['eval_f1'], 'g-o', linewidth=2.0, markersize=4,
            label='Current F1', alpha=0.7)
    ax2.plot(epochs, history['best_f1'], 'r--', linewidth=2.5,
            label='Best-so-far F1', alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Score (Current, Best)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim((max(0, min(history['eval_f1']) - 0.1), 1.05))
    
    # 3. Precision with Spikes
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, history['eval_precision'], 'b-o', linewidth=1.5, markersize=3,
            label='Precision', alpha=0.7)
    # Precision spike display
    if sum(history['precision_spikes']) > 0:
        spike_epochs = [e for e, s in zip(epochs, history['precision_spikes']) if s == 1]
        spike_values = [p for p, s in zip(history['eval_precision'], history['precision_spikes']) if s == 1]
        ax3.scatter(spike_epochs, spike_values, c='red', s=150, marker='X',
                   zorder=10, label=f'Spikes ({sum(history["precision_spikes"])})', alpha=0.9)
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax3.set_title('Precision (with Spikes)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10, loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim((max(0, min(history['eval_precision']) - 0.1), 1.05))
    
    # 4. Recall
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, history['eval_recall'], 'g-o', linewidth=1.5, markersize=3,
            label='Recall', alpha=0.7)
    ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax4.set_title('Recall', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10, loc='best', framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim((max(0, min(history['eval_recall']) - 0.1), 1.05))
    
    # 5. Precision & Recall 비교
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, history['eval_precision'], 'b-', linewidth=2.0, label='Precision', alpha=0.7)
    ax5.plot(epochs, history['eval_recall'], 'r-', linewidth=2.0, label='Recall', alpha=0.7)
    ax5.plot(epochs, history['eval_f1'], 'g-', linewidth=2.5, label='F1', alpha=0.8)
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax5.set_title('Precision & Recall & F1', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10, loc='best', framealpha=0.9)
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # 6. Precision-Recall Curve (better for NER than ROC)
    ax6 = fig.add_subplot(gs[1, 2])
    if len(history['eval_f1']) > 1:
        precision = history['eval_precision']
        recall = history['eval_recall']
        sorted_pairs = sorted(zip(recall, precision))
        recall_sorted = [x[0] for x in sorted_pairs]
        precision_sorted = [x[1] for x in sorted_pairs]
        ap_score = np.trapz(precision_sorted, recall_sorted)
        ax6.plot(recall_sorted, precision_sorted, 'g-o', linewidth=2.5, markersize=4, 
                label=f'Model (AP={ap_score:.4f})', color='#2E86AB')
        
        # Add F1 iso-lines
        for f1 in [0.3, 0.5, 0.7, 0.9]:
            x = np.linspace(0.01, 1, 100)
            y = f1 * x / (2 * x - f1)
            y = np.clip(y, 0, 1)
            ax6.plot(x, y, '--', color='gray', alpha=0.2, linewidth=1)
    else:
        precision = history['eval_precision'][0]
        recall = history['eval_recall'][0]
        ap_score = precision * recall
        ax6.plot([recall], [precision], 'o', markersize=10, color='#2E86AB',
                label=f'Model (AP≈{ap_score:.4f})')
    
    ax6.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax6.set_title(f'Precision-Recall Curve (AP={ap_score:.4f})', fontsize=14, fontweight='bold')
    ax6.set_xlim((0, 1))
    ax6.set_ylim((0, 1))
    ax6.legend(fontsize=10, loc='lower left', framealpha=0.9)
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    # 7. Metrics Summary
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')
    summary_text = "=" * 50 + "\n"
    summary_text += "      Final Evaluation Results\n"
    summary_text += "=" * 50 + "\n\n"
    summary_text += f"Final F1 (Current):     {history['eval_f1'][-1]:.4f}\n"
    summary_text += f"Best F1 (Best-so-far):  {max(history['eval_f1']):.4f}\n\n"
    summary_text += f"Final Precision:        {history['eval_precision'][-1]:.4f}\n"
    summary_text += f"Final Recall:           {history['eval_recall'][-1]:.4f}\n"
    summary_text += f"Average Precision:      {ap_score:.4f}\n\n"
    summary_text += f"Precision Spikes:       {sum(history['precision_spikes'])} times\n"
    summary_text += "=" * 50
    
    ax7.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', 
             bbox=dict(boxstyle='round,pad=1.5', facecolor='lightblue', 
                      alpha=0.3, edgecolor='navy', linewidth=2.5))
    
    # 8. Training Info
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')
    info_text = "=" * 50 + "\n"
    info_text += "        Training Information\n"
    info_text += "=" * 50 + "\n\n"
    info_text += f"Model: {model_name.split('/')[-1]}\n"
    info_text += f"Total Epochs: {len(epochs)}\n\n"
    best_f1_idx = np.argmax(history['eval_f1'])
    info_text += f"Best F1 Epoch: {best_f1_idx + 1}\n"
    info_text += f"Best F1 Value: {history['eval_f1'][best_f1_idx]:.4f}\n\n"
    info_text += "Improvements:\n"
    info_text += "- Best-so-far tracking\n"
    info_text += "- Spike detection (Delta>=0.25)\n"
    info_text += "=" * 50
    
    ax8.text(0.05, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=1.5', facecolor='lightyellow', 
                      alpha=0.3, edgecolor='orange', linewidth=2.5))
    
    # 9. Convergence Analysis
    ax9 = fig.add_subplot(gs[2, 2])
    # F1 improvement rate (derivative)
    if len(history['eval_f1']) > 10:
        window = 10
        f1_smooth = np.convolve(history['eval_f1'], np.ones(window)/window, mode='valid')
        f1_derivative = np.diff(f1_smooth)
        derivative_epochs = epochs[window//2:window//2+len(f1_derivative)]
        ax9.plot(derivative_epochs, f1_derivative, 'b-', linewidth=2.0, alpha=0.8)
        ax9.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
        ax9.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax9.set_ylabel('F1 Improvement Rate', fontsize=12, fontweight='bold')
        ax9.set_title('Convergence Analysis (F1 Derivative)', fontsize=14, fontweight='bold')
        ax9.grid(True, alpha=0.3, linestyle='--')
    else:
        ax9.axis('off')
        ax9.text(0.5, 0.5, 'Not enough epochs\nfor convergence analysis', 
                ha='center', va='center', fontsize=12)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    plt.close()
    
    print(f"Advanced Training curves saved: {save_path}")

    """
    Training curve visualization (Loss, F1, Precision, Recall, PR Curve)
    
    Args:
        history: Training history {'train_loss': [...], 'eval_loss': [...], ...}
        save_path: Save path
        model_name: Model name
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # 2x3 grid for ROC curve
    fig.suptitle(f'NER Training Results - {model_name}', fontsize=18, fontweight='bold', y=0.995)
    
    epochs = range(1, len(history['eval_loss']) + 1)
    
    # Note: Loss can be negative when model is perfectly trained (log_likelihood ≈ 0)
    # This is normal behavior for CRF models
    
    # 1. Loss curve (scaled to Train Loss range)
    ax1 = axes[0, 0]
    if history['train_loss']:
        # Train loss has more points (every step)
        train_steps = np.linspace(1, len(epochs), len(history['train_loss']))
        ax1.plot(train_steps, history['train_loss'], 'b-', alpha=0.6, label='Train Loss', linewidth=2.0)
    ax1.plot(epochs, history['eval_loss'], 'r-o', linewidth=2.5, markersize=8, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax1.set_title('Training/Validation Loss', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.4, linestyle='--', linewidth=1.0)
    ax1.tick_params(labelsize=11)
    
    # Y-axis range adjustment (prioritize Train Loss range)
    all_loss_values = history['train_loss'] + history['eval_loss'] if history['train_loss'] else history['eval_loss']
    if all_loss_values and len(all_loss_values) > 0:
        loss_min = min(all_loss_values)
        loss_max = max(all_loss_values)
        loss_range = loss_max - loss_min
        
        # If change is very small, use fixed range
        if loss_range < 0.01:
            ax1.set_ylim([loss_min - 0.5, loss_max + 0.5])  # Allow negative if needed
        else:
            loss_margin = loss_range * 0.3  # 30% margin
            ax1.set_ylim([loss_min - loss_margin, loss_max + loss_margin])  # Allow negative
    
    # 2. F1 Score (without "Best:" text)
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['eval_f1'], 'g-o', linewidth=2.5, marker='o', markersize=8)
    ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    ax2.set_title('F1 Score (Span-level)', fontsize=15, fontweight='bold', pad=15)
    
    # Y-axis range dynamic adjustment
    f1_values = history['eval_f1']
    if f1_values and len(f1_values) > 0:
        f1_min = min(f1_values)
        f1_max = max(f1_values)
        f1_range = f1_max - f1_min
        
        # If change is very small (overfitting or very stable)
        if f1_range < 0.05:
            center = (f1_min + f1_max) / 2
            ax2.set_ylim([max(0, center - 0.15), min(1.0, center + 0.15)])
        else:
            f1_margin = f1_range * 0.3  # 30% margin
            ax2.set_ylim([max(0, f1_min - f1_margin), min(1.0, f1_max + f1_margin)])
    else:
        ax2.set_ylim([0, 1.05])
    
    ax2.grid(True, alpha=0.4, linestyle='--', linewidth=1.0)
    ax2.tick_params(labelsize=11)
    
    # Best F1 line only (no text to avoid overlap)
    best_f1_idx = np.argmax(history['eval_f1'])
    best_f1 = history['eval_f1'][best_f1_idx]
    ax2.axhline(y=best_f1, color='r', linestyle='--', alpha=0.5, linewidth=2)
    
    # 3. Precision & Recall
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['eval_precision'], 'b-o', linewidth=2.5, label='Precision', marker='s', markersize=8)
    ax3.plot(epochs, history['eval_recall'], 'r-o', linewidth=2.5, label='Recall', marker='^', markersize=8)
    ax3.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax3.set_title('Precision & Recall', fontsize=15, fontweight='bold', pad=15)
    
    # Y-axis range dynamic adjustment
    pr_values = history['eval_precision'] + history['eval_recall']
    if pr_values and len(pr_values) > 0:
        pr_min = min(pr_values)
        pr_max = max(pr_values)
        pr_range = pr_max - pr_min
        
        # If change is very small
        if pr_range < 0.05:
            center = (pr_min + pr_max) / 2
            ax3.set_ylim([max(0, center - 0.15), min(1.0, center + 0.15)])
        else:
            pr_margin = pr_range * 0.3  # 30% margin
            ax3.set_ylim([max(0, pr_min - pr_margin), min(1.0, pr_max + pr_margin)])
    else:
        ax3.set_ylim([0, 1.05])
    
    ax3.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax3.grid(True, alpha=0.4, linestyle='--', linewidth=1.0)
    ax3.tick_params(labelsize=11)
    
    # 4. Precision-Recall Curve and AP (Average Precision)
    ax4 = axes[0, 2]
    
    # Precision-Recall Curve (better for NER tasks than ROC)
    if len(history['eval_f1']) > 1:
        precision = history['eval_precision']
        recall = history['eval_recall']
        
        # Sort by recall for proper PR curve
        sorted_pairs = sorted(zip(recall, precision))
        recall_sorted = [x[0] for x in sorted_pairs]
        precision_sorted = [x[1] for x in sorted_pairs]
        
        # Calculate Average Precision (AP) using trapezoidal rule
        ap_score = np.trapz(precision_sorted, recall_sorted)
        
        # Plot PR curve
        ax4.plot(recall_sorted, precision_sorted, 'g-o', linewidth=2.5, markersize=6, 
                label=f'Model (AP={ap_score:.4f})', color='#2E86AB')
        
        # Add F1 iso-lines for reference
        f1_scores = [0.3, 0.5, 0.7, 0.9]
        for f1 in f1_scores:
            x = np.linspace(0.01, 1, 100)
            y = f1 * x / (2 * x - f1)
            y = np.clip(y, 0, 1)
            ax4.plot(x, y, '--', color='gray', alpha=0.3, linewidth=1)
            # Label position
            if f1 < 0.9:
                ax4.text(0.9, f1 * 0.9 / (2 * 0.9 - f1), f'F1={f1:.1f}', 
                        fontsize=8, color='gray', alpha=0.6)
        
    else:
        # Single epoch - show point
        precision = history['eval_precision'][0]
        recall = history['eval_recall'][0]
        ap_score = precision * recall  # Simple approximation
        ax4.plot([recall], [precision], 'o', markersize=12, color='#2E86AB',
                label=f'Model (AP≈{ap_score:.4f})')
    
    ax4.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax4.set_title(f'Precision-Recall Curve (AP={ap_score:.4f})', fontsize=15, fontweight='bold', pad=15)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    ax4.legend(fontsize=11, loc='lower left', framealpha=0.9)
    ax4.grid(True, alpha=0.4, linestyle='--', linewidth=1.0)
    ax4.tick_params(labelsize=11)
    
    # 5. Final Metrics Summary
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    final_metrics = {
        'F1 Score': history['eval_f1'][-1],
        'Precision': history['eval_precision'][-1],
        'Recall': history['eval_recall'][-1],
        'Best F1': max(history['eval_f1']),
        'Average Precision': ap_score,
        'Final Loss': history['eval_loss'][-1]  # Can be negative (good!)
    }
    
    summary_text = "=" * 50 + "\n"
    summary_text += "      Final Evaluation Results\n"
    summary_text += "=" * 50 + "\n\n"
    
    for metric, value in final_metrics.items():
        summary_text += f"{metric:.<35} {value:.4f}\n"
    
    summary_text += "\n" + "=" * 50 + "\n"
    summary_text += f"Total Epochs: {len(epochs)}\n"
    summary_text += f"Model: {model_name.split('/')[-1]}\n"
    summary_text += "=" * 50
    
    ax5.text(0.05, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center', 
             bbox=dict(boxstyle='round,pad=1.5', facecolor='lightblue', alpha=0.3, edgecolor='navy', linewidth=2.5))
    
    # 6. Additional info panel
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    info_text = "=" * 50 + "\n"
    info_text += "        Training Information\n"
    info_text += "=" * 50 + "\n\n"
    info_text += f"Model: {model_name}\n\n"
    info_text += f"Epoch with Best F1: {best_f1_idx + 1}\n"
    info_text += f"Best F1 Score: {best_f1:.4f}\n\n"
    info_text += f"Final F1: {history['eval_f1'][-1]:.4f}\n"
    info_text += f"Final Precision: {history['eval_precision'][-1]:.4f}\n"
    info_text += f"Final Recall: {history['eval_recall'][-1]:.4f}\n\n"
    info_text += "=" * 50 + "\n"
    info_text += "Note:\n"
    info_text += "- PR Curve shows Precision vs Recall\n"
    info_text += "- AP (Average Precision) is the\n"
    info_text += "  area under PR curve\n"
    info_text += "- F1 iso-lines shown for reference\n"
    info_text += "=" * 50
    
    ax6.text(0.05, 0.5, info_text, fontsize=11, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=1.5', facecolor='lightyellow', alpha=0.3, edgecolor='orange', linewidth=2.5))
    
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    plt.close()
    
    print(f"OK: Training curves saved: {save_path}")


def plot_entity_performance(val_dataset, model, tokenizer, save_path: Path, device='cuda'):
    """
    엔티티 타입별 성능 시각화
    
    Args:
        val_dataset: 검증 데이터셋
        model: 학습된 모델
        tokenizer: 토크나이저
        save_path: 저장 경로
        device: 디바이스
    """
    model.eval()
    model.to(device)
    
    # 엔티티별 통계 수집
    entity_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    with torch.no_grad():
        for item in val_dataset:
            # item['input_ids']는 이미 리스트이므로 그대로 사용
            input_ids = torch.tensor(item['input_ids']).unsqueeze(0).to(device)  # (1, seq_len)
            attention_mask = torch.tensor(item['attention_mask']).unsqueeze(0).to(device)
            labels = item['labels']
            
            # 예측 (모델의 forward 사용)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # CRF Viterbi decoding 결과 추출
            if isinstance(outputs, dict) and 'predictions' in outputs:
                predictions = outputs['predictions'].squeeze(0).tolist()  # (seq_len,)
            else:
                raise ValueError(f"Unexpected model output: {type(outputs)}")
            
            # BIO → 엔티티
            # labels가 tensor일 수 있으므로 변환
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            true_tags = [ID_TO_LABEL[l] for l in labels if l != -100]
            pred_tags = [ID_TO_LABEL[p] for p in predictions[:len(true_tags)]]
            
            true_entities = bio_to_entities(true_tags)
            pred_entities = bio_to_entities(pred_tags)
            
            # 엔티티별 집계
            for start, end, etype in true_entities:
                if (start, end, etype) in pred_entities:
                    entity_stats[etype]['tp'] += 1
                else:
                    entity_stats[etype]['fn'] += 1
            
            for start, end, etype in pred_entities:
                if (start, end, etype) not in true_entities:
                    entity_stats[etype]['fp'] += 1
    
    # F1 계산
    entity_f1 = {}
    for etype, stats in entity_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        entity_f1[etype] = {'f1': f1, 'precision': precision, 'recall': recall, 'count': tp + fn}
    
    # 시각화
    if not entity_f1:
        print("경고: 엔티티 성능 데이터 없음")
        return
    
    # Count 기준 정렬 (많이 나온 엔티티 우선)
    sorted_entities = sorted(entity_f1.items(), key=lambda x: x[1]['count'], reverse=True)
    top_entities = sorted_entities[:15]  # 상위 15개만
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))  # 크기 증가
    fig.suptitle('엔티티별 성능 분석', fontsize=18, fontweight='bold', y=0.98)
    
    # 1. F1 Score 막대 그래프
    entity_names = [e[0] for e in top_entities]
    f1_scores = [e[1]['f1'] for e in top_entities]
    precisions = [e[1]['precision'] for e in top_entities]
    recalls = [e[1]['recall'] for e in top_entities]
    
    x = np.arange(len(entity_names))
    width = 0.25
    
    ax1.bar(x - width, f1_scores, width, label='F1', color='green', alpha=0.8, edgecolor='black')
    ax1.bar(x, precisions, width, label='Precision', color='blue', alpha=0.8, edgecolor='black')
    ax1.bar(x + width, recalls, width, label='Recall', color='red', alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('엔티티 타입', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax1.set_title('엔티티별 성능 (F1, Precision, Recall)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(entity_names, rotation=45, ha='right', fontsize=11)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim([0, 1.1])
    ax1.tick_params(axis='y', labelsize=11)
    
    # 2. 엔티티 출현 빈도
    counts = [e[1]['count'] for e in top_entities]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(entity_names)))  # type: ignore[attr-defined]
    
    ax2.barh(entity_names, counts, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('출현 횟수', fontsize=14, fontweight='bold')
    ax2.set_ylabel('엔티티 타입', fontsize=14, fontweight='bold')
    ax2.set_title('엔티티 출현 빈도', fontsize=16, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax2.tick_params(axis='both', labelsize=11)
    
    # 값 표시
    for i, (name, count) in enumerate(zip(entity_names, counts)):
        ax2.text(count + max(counts) * 0.02, i, f' {count}', 
                va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # 여백 조정
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    
    print(f"OK: 엔티티 성능 저장: {save_path}")


def plot_prediction_examples(val_samples, model, tokenizer, save_path: Path, num_examples=5, device='cuda'):
    """
    예측 예시 시각화 (실제 vs 예측 비교)
    
    Args:
        val_samples: 검증 샘플
        model: 학습된 모델
        tokenizer: 토크나이저
        save_path: 저장 경로
        num_examples: 표시할 예시 수
        device: 디바이스
    """
    model.eval()
    model.to(device)
    
    # 랜덤 샘플 선택
    examples = random.sample(val_samples, min(num_examples, len(val_samples)))
    
    fig, axes = plt.subplots(num_examples, 1, figsize=(16, 4 * num_examples))
    if num_examples == 1:
        axes = [axes]
    
    for idx, (sample, ax) in enumerate(zip(examples, axes)):
        text = sample['text']
        true_entities = sample['entities']
        
        # 토큰화
        encoding = tokenizer(
            text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # 예측 (모델의 forward 사용)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # CRF Viterbi decoding 결과 추출
            if isinstance(outputs, dict) and 'predictions' in outputs:
                predictions = outputs['predictions'].squeeze(0).tolist()  # (seq_len,)
            else:
                raise ValueError(f"Unexpected model output: {type(outputs)}")
        
        # 토큰 → 문자 매핑
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # 예측 엔티티 추출 (간단히 B-/I- 태그만)
        pred_entities = []
        current_entity = None
        
        for token, pred_id in zip(tokens, predictions):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            pred_label = ID_TO_LABEL[pred_id]
            
            if pred_label.startswith('B-'):
                if current_entity:
                    pred_entities.append(current_entity)
                current_entity = {'text': token.replace('##', ''), 'type': pred_label[2:]}
            elif pred_label.startswith('I-') and current_entity:
                current_entity['text'] += token.replace('##', '')
        
        if current_entity:
            pred_entities.append(current_entity)
        
        # 시각화
        ax.axis('off')
        
        # 텍스트 표시 (더 간결하게)
        display_text = f"원문: {text[:80]}{'...' if len(text) > 80 else ''}\n\n"
        
        display_text += f"정답 엔티티:\n"
        for ent_text, ent_type in true_entities:
            display_text += f"   • {ent_text} ({ent_type})\n"
        
        display_text += f"\n예측 엔티티:\n"
        if pred_entities:
            for ent in pred_entities:
                display_text += f"   • {ent['text']} ({ent['type']})\n"
        else:
            display_text += "   (없음)\n"
        
        # 정확도 계산
        true_set = set((t, e) for t, e in true_entities)
        pred_set = set((e['text'], e['type']) for e in pred_entities)
        
        matches = len(true_set & pred_set)
        accuracy = matches / max(len(true_set), 1)
        
        display_text += f"\n정확도: {matches}/{len(true_set)} = {accuracy*100:.1f}%"
        
        # 배경색 (정확도에 따라)
        bg_color = 'lightgreen' if accuracy >= 0.8 else 'lightyellow' if accuracy >= 0.5 else 'lightcoral'
        
        ax.text(0.05, 0.5, display_text, fontsize=11, family='monospace',
                verticalalignment='center', wrap=True,
                bbox=dict(boxstyle='round,pad=1.2', facecolor=bg_color, alpha=0.4, 
                         edgecolor='black', linewidth=1.5))
    
    plt.suptitle('NER 예측 예시 (실제 vs 예측)', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    
    print(f"OK: 예측 예시 저장: {save_path}")


# ===========================
# 메인 실행
# ===========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER 모델 학습")
    parser.add_argument("--model", type=str, default="google-bert/bert-base-multilingual-cased")
    parser.add_argument("--samples", type=int, default=30000, help="학습 샘플 수")
    parser.add_argument("--epochs", type=int, default=10, help="에포크 수")
    parser.add_argument("--output", type=str, default=None, help="모델 저장 경로")
    
    args = parser.parse_args()
    
    train_ner_model(
        model_name=args.model,
        num_samples=args.samples,
        num_epochs=args.epochs,
        output_dir=args.output
    )
