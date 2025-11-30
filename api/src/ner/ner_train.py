#!/usr/bin/env python3
import argparse
import os
import random
import math
import re
import json
import time
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoTokenizer, 
    AutoModel,
    Trainer, 
    TrainingArguments, 
    TrainerCallback,
    get_linear_schedule_with_warmup
)
from torchcrf import CRF

try:
    from .ner_model import BertBiLstmCrf, NERConfig, save_ner_model
    from .ner_data import read_conll, build_label_map, NERDataset, evaluate_ner
except ImportError:
    from ner_model import BertBiLstmCrf, NERConfig, save_ner_model
    from ner_data import read_conll, build_label_map, NERDataset, evaluate_ner

# Entity Types
ENTITY_TYPES = [
    "NAME", "PHONE", "ADDRESS", "DATE", "COMPANY", "EMAIL", "POSITION",
    "CONTRACT_TYPE", "MONEY", "PERIOD", "ID_NUM", "CONSENT_TYPE", "RIGHT_INFO",
    "PROJECT_NAME", "LAW_REFERENCE", "TITLE", "URL", "DESCRIPTION", "TYPE",
    "STATUS", "DEPARTMENT", "LANGUAGE", "QUANTITY"
]

# BIO Labels
BIO_LABELS = ["O"] + [f"{prefix}-{entity}" for entity in ENTITY_TYPES for prefix in ["B", "I"]]
LABEL_TO_ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ========== 데이터 생성 헬퍼 함수 ==========

def extract_entities_from_template(template: str) -> List[str]:
    return [match.group(1) for match in re.finditer(r'\{(\w+)\}', template)]

def generate_sample_from_template(template: str, entity_generators: Dict) -> Tuple[str, List[Tuple[str, str]], str]:
    entities = {}
    for match in re.finditer(r'\{(\w+)\}', template):
        etype = match.group(1)
        if etype in entity_generators and etype not in entities:
            entities[etype] = entity_generators[etype]()
    
    text = template
    for etype, value in entities.items():
        text = text.replace(f"{{{etype}}}", value)
    
    entity_list = [(value, etype) for etype, value in entities.items() if value in text]
    return text, entity_list, template

def build_template_list(single_templates: Dict, dual_templates: List, multi_templates: List) -> List[Tuple[str, List[str]]]:
    all_templates = []
    for entity_type, templates_list in single_templates.items():
        for tmpl in templates_list:
            all_templates.append((tmpl, [entity_type]))
    for tmpl in dual_templates + multi_templates:
        entities = extract_entities_from_template(tmpl)
        all_templates.append((tmpl, entities))
    return all_templates

# ========== 생성기 함수들 ==========

def generate_random_korean_name():
    surnames = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황", "안", "송", "류", "전"]
    syllables = ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파", "하", "건", "성", "현", "우", "준", "규", "민"]
    surname = random.choice(surnames)
    name_len = 2 if random.random() < 0.6 else 3
    name = "".join([random.choice(syllables) for _ in range(name_len - 1)])
    return surname + name

def random_phone():
    return f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}"

def random_date():
    return f"{random.randint(2020,2025)}년 {random.randint(1,12)}월 {random.randint(1,28)}일"

def random_email():
    return f"user{random.randint(1,999)}@example.com"

def apply_ocr_noise(text: str, noise_prob: float) -> str:
    """
    Simulate OCR errors:
    - Space insertion (common in Korean OCR)
    - Character mutation (rare but possible)
    """
    if noise_prob <= 0:
        return text
        
    chars = list(text)
    new_chars = []
    
    for char in chars:
        # Apply noise with probability
        if random.random() < noise_prob:
            noise_type = random.choice(['space', 'char'])
            
            if noise_type == 'space':
                # Insert space (most common OCR error for Korean)
                new_chars.append(char + ' ')
            elif noise_type == 'char':
                # Dummy mutation (in real OCR, use visual similarity)
                # Just keep original to avoid semantic drift for now
                new_chars.append(char) 
        else:
            new_chars.append(char)
            
    # Cleanup excessive spaces
    return "".join(new_chars).replace("  ", " ")

def generate_training_samples(num_samples: int = 3000, balanced: bool = True, noise_level: float = 0.0, dataset_type: str = 'train') -> List[Dict]:
    """
    학습 데이터 생성 (다양성 강화 + OCR 노이즈 버전)
    dataset_type: 'train' or 'dev'/'test'. 
    'dev'/'test' will use a different set of templates to evaluate generalization.
    """
    # 엔티티 생성기 매핑
    entity_generators = {
        "NAME": generate_random_korean_name,
        "PHONE": random_phone,
        "DATE": random_date,
        "EMAIL": random_email,
        "COMPANY": lambda: f"주식회사 {generate_random_korean_name()}",
        "ADDRESS": lambda: f"서울시 강남구 {generate_random_korean_name()}로 {random.randint(1,100)}",
        "ID_NUM": lambda: f"{random.randint(0,99):02d}0101-{random.randint(1,4)}******",
        "MONEY": lambda: f"{random.randint(1,999)}만원",
        "PERIOD": lambda: f"{random.randint(1,12)}개월",
        "CONTRACT_TYPE": lambda: random.choice(["표준계약서", "양도계약서", "이용허락계약서", "비밀유지서약서", "근로계약서"]),
        "POSITION": lambda: random.choice(["팀장", "대표", "사원", "책임", "부장", "이사"]),
        "RIGHT_INFO": lambda: random.choice(["저작재산권", "배포권", "복제권", "전송권", "2차적저작물작성권"]),
        "PROJECT_NAME": lambda: f"프로젝트 {chr(random.randint(65, 90))}{random.randint(1,100)}",
        "LAW_REFERENCE": lambda: f"저작권법 제{random.randint(1,50)}조",
        "TITLE": lambda: f"{generate_random_korean_name()} 관련 합의서",
        "URL": lambda: f"http://www.{generate_random_korean_name()}{random.randint(1,99)}.com",
        "DESCRIPTION": lambda: "본 계약의 상세 내용은 별첨과 같다.",
        "TYPE": lambda: random.choice(["어문저작물", "사진저작물", "영상저작물", "소프트웨어"]),
        "STATUS": lambda: random.choice(["체결 완료", "검토 중", "해지", "갱신"]),
        "DEPARTMENT": lambda: random.choice(["인사팀", "개발팀", "법무팀", "영업팀", "기획팀"]),
        "LANGUAGE": lambda: random.choice(["한국어", "영어", "일본어"]),
        "QUANTITY": lambda: f"{random.randint(1,100)}건",
        "CONSENT_TYPE": lambda: random.choice(["개인정보 수집 이용 동의", "마케팅 수신 동의", "제3자 제공 동의"]),
    }

    # 1. Train Templates (Common patterns)
    train_single_templates = {
        "NAME": [
            "{NAME}입니다.", "{NAME} 님 안녕하세요.", "작성자: {NAME}", "본인은 {NAME}로서 서명합니다.",
            "{NAME} 귀하에게 알립니다.", "담당자는 {NAME}입니다.", "수신: {NAME}", "발신: {NAME}",
            "성명: {NAME}", "이름: {NAME}", "{NAME} (인)", "{NAME} (서명)", "대리인 {NAME}"
        ],
        "PHONE": [
            "연락처는 {PHONE}입니다.", "문의: {PHONE}", "Tel: {PHONE}", "비상연락망: {PHONE}",
            "{PHONE}으로 전화주세요.", "휴대전화 {PHONE} 기재 요망.", "H.P: {PHONE}", "전화: {PHONE}",
            "팩스: {PHONE}", "대표번호: {PHONE}", "고객센터: {PHONE}"
        ],
        "DATE": [
            "{DATE}에 만나요.", "기한: {DATE}", "날짜: {DATE}", "계약일: {DATE}",
            "{DATE}부터 효력이 발생합니다.", "마감일은 {DATE}까지입니다.", "작성일: {DATE}",
            "체결일자: {DATE}", "유효기간: {DATE}", "{DATE} 기준", "{DATE} 현재"
        ],
        "EMAIL": [
            "이메일 {EMAIL}로 보내주세요.", "E-mail: {EMAIL}", "문의 사항은 {EMAIL}로.",
            "회신 주소: {EMAIL}", "{EMAIL} (업무용)", "전자우편: {EMAIL}", "메일: {EMAIL}",
            "contact: {EMAIL}", "support: {EMAIL}"
        ],
        "COMPANY": [
            "{COMPANY}에서 왔습니다.", "소속: {COMPANY}", "{COMPANY} 대표이사 귀하",
            "당사자는 {COMPANY}입니다.", "{COMPANY}와의 협력.", "상호: {COMPANY}",
            "법인명: {COMPANY}", "업체명: {COMPANY}", "발주처: {COMPANY}", "수주처: {COMPANY}"
        ],
        "ADDRESS": [
            "주소는 {ADDRESS}입니다.", "위치: {ADDRESS}", "사업장 소재지: {ADDRESS}",
            "{ADDRESS}로 배송 바랍니다.", "본점: {ADDRESS}", "거주지: {ADDRESS}",
            "등록기준지: {ADDRESS}", "배달 장소: {ADDRESS}"
        ],
        "ID_NUM": [
            "주민번호: {ID_NUM}", "등록번호 {ID_NUM}입니다.", "사업자번호 {ID_NUM} 기재.",
            "신분증 번호: {ID_NUM}", "주민등록번호: {ID_NUM}", "법인등록번호: {ID_NUM}"
        ],
        "MONEY": [
            "가격은 {MONEY}입니다.", "비용: {MONEY}", "계약금 {MONEY}를 지급한다.",
            "총액 {MONEY} (VAT 별도)", "보상금 {MONEY} 산정.", "금액: {MONEY}",
            "일금 {MONEY}정", "합계: {MONEY}", "잔금: {MONEY}", "계약보증금: {MONEY}"
        ],
    }
    
    # Fill missing types for train
    for etype in ENTITY_TYPES:
        if etype not in train_single_templates:
            train_single_templates[etype] = [
                f"이것은 {etype} 예시인 {{{etype}}}입니다.", 
                f"{etype}: {{{etype}}}",
                f"상세 {etype} 정보: {{{etype}}}",
                f"{{{etype}}}에 관한 내용.",
                f"다음 {etype}을 확인하세요: {{{etype}}}",
                f"입력된 {etype} 값은 {{{etype}}} 입니다."
            ]

    train_dual_templates = [
        "{NAME}의 전화번호는 {PHONE}입니다.", "{DATE}까지 {EMAIL}로 제출하세요.",
        "{COMPANY}의 주소는 {ADDRESS}입니다.", "{NAME}님({ID_NUM}) 확인되었습니다.",
        "{COMPANY}는 {MONEY}를 {DATE}에 지급한다.", "{NAME} {POSITION}님의 연락처는 {PHONE}입니다.",
        "{CONTRACT_TYPE} 체결일은 {DATE}입니다.", "{PROJECT_NAME} 예산은 {MONEY}입니다.",
        "{DEPARTMENT} 소속 {NAME}입니다.", "{RIGHT_INFO} 양도 대가는 {MONEY}입니다.",
        "{LAW_REFERENCE}에 의거하여 {CONTRACT_TYPE}을 체결합니다.", "{NAME}은 {ADDRESS}에 거주합니다.",
        "{COMPANY} (대표: {NAME})", "{DATE} 자로 {COMPANY}와 계약함",
        "{NAME} ({PHONE})", "{EMAIL} / {PHONE}", "{ADDRESS} ({COMPANY})",
        "{MONEY} ({DATE} 지급)", "{CONTRACT_TYPE} ({DATE})", "{NAME} - {POSITION}"
    ]
    
    train_multi_templates = [
        "{COMPANY} {DEPARTMENT}의 {NAME} {POSITION}입니다.",
        "{DATE}에 {COMPANY}와 {NAME}은 {CONTRACT_TYPE}을 체결했다.",
        "본 {CONTRACT_TYPE}은 {DATE}부터 {PERIOD}간 유효하며 금액은 {MONEY}이다.",
        "{NAME}({ID_NUM})은 {ADDRESS}에 거주하며 {PHONE}을 사용한다.",
        "{PROJECT_NAME} 수행을 위해 {COMPANY}는 {MONEY}를 투자하고 {DATE}에 완료한다.",
        "{TITLE}에 명시된 {RIGHT_INFO}는 {LAW_REFERENCE}에 따라 {COMPANY}에 귀속된다.",
        "{NAME} {POSITION}은 {DATE}에 {CONSENT_TYPE}에 서명하고 {EMAIL}로 제출했다.",
        "{COMPANY}는 {ADDRESS}에 위치하며 대표전화는 {PHONE}, 홈페이지는 {URL}이다.",
        "{TYPE} 저작물 {QUANTITY}에 대한 {RIGHT_INFO}를 {MONEY}에 양도한다.",
        "갑: {COMPANY}, 을: {NAME}, 계약일: {DATE}",
        "{NAME} (주민번호: {ID_NUM}, 주소: {ADDRESS})",
        "1. {NAME} 2. {PHONE} 3. {EMAIL}",
        "상기 {NAME}은 {DATE}에 {COMPANY}에 입사하였음을 증명함.",
        "{COMPANY} 귀중. 참조: {DEPARTMENT} {NAME} {POSITION}",
        "계약금 {MONEY}는 {DATE}에 입금하고 잔금 {MONEY}는 {DATE}에 지급한다."
    ]

    # 2. Dev/Test Templates (Unseen patterns to test generalization)
    dev_single_templates = {
        "NAME": [
            "Who is {NAME}?", "Contact person: {NAME}", "{NAME} signed here.",
            "Approved by {NAME}", "To: {NAME}", "From: {NAME}", "User: {NAME}"
        ],
        "PHONE": [
            "Call {PHONE} now.", "Mobile: {PHONE}", "Phone Number: {PHONE}",
            "Reach me at {PHONE}", "Dial {PHONE}", "SMS: {PHONE}"
        ],
        "DATE": [
            "Due by {DATE}", "Date: {DATE}", "Effective from {DATE}",
            "Signed on {DATE}", "Expires: {DATE}", "Since {DATE}"
        ],
        "EMAIL": [
            "Send to {EMAIL}", "Email address: {EMAIL}", "Reply-To: {EMAIL}",
            "CC: {EMAIL}", "Mail: {EMAIL}"
        ],
        "COMPANY": [
            "Vendor: {COMPANY}", "Client: {COMPANY}", "Organization: {COMPANY}",
            "Made by {COMPANY}", "Copyright {COMPANY}"
        ],
        "ADDRESS": [
            "Located at {ADDRESS}", "Ship to: {ADDRESS}", "Office: {ADDRESS}",
            "Residence: {ADDRESS}", "Site: {ADDRESS}"
        ],
        "ID_NUM": [
            "ID: {ID_NUM}", "SSN: {ID_NUM}", "Reg No: {ID_NUM}",
            "License: {ID_NUM}"
        ],
        "MONEY": [
            "Cost: {MONEY}", "Price: {MONEY}", "Total: {MONEY}",
            "Fee: {MONEY}", "Payment: {MONEY}", "Amount: {MONEY}"
        ],
    }
    
    # Fill missing types for dev
    for etype in ENTITY_TYPES:
        if etype not in dev_single_templates:
            dev_single_templates[etype] = [
                f"Check {etype}: {{{etype}}}", 
                f"Value of {etype} is {{{etype}}}",
                f"Please provide {{{etype}}} for {etype}.",
                f"Missing {etype}: {{{etype}}}"
            ]

    dev_dual_templates = [
        "Please contact {NAME} at {PHONE}.",
        "Submit to {EMAIL} by {DATE}.",
        "{COMPANY} is located at {ADDRESS}.",
        "Identity verified: {NAME}, {ID_NUM}.",
        "Payment of {MONEY} due on {DATE}.",
        "{NAME} ({POSITION}) can be reached at {PHONE}.",
        "Agreement {CONTRACT_TYPE} signed on {DATE}.",
        "Budget for {PROJECT_NAME}: {MONEY}.",
        "{NAME} works at {DEPARTMENT}.",
        "Transfer fee for {RIGHT_INFO} is {MONEY}.",
        "Under {LAW_REFERENCE}, we sign {CONTRACT_TYPE}.",
        "{NAME} lives in {ADDRESS}.",
        "{COMPANY} CEO {NAME}",
        "{DATE}: {COMPANY}",
        "{PHONE} ({NAME})",
        "{EMAIL}, {PHONE}",
        "{COMPANY} - {ADDRESS}",
        "{DATE} / {MONEY}",
        "{DATE} - {CONTRACT_TYPE}",
        "{POSITION}: {NAME}"
    ]
    
    dev_multi_templates = [
        "{NAME} ({POSITION}) from {COMPANY} {DEPARTMENT}.",
        "On {DATE}, {COMPANY} and {NAME} signed {CONTRACT_TYPE}.",
        "This {CONTRACT_TYPE} is valid from {DATE} for {PERIOD}, value {MONEY}.",
        "{NAME} (ID: {ID_NUM}) resides at {ADDRESS}, phone: {PHONE}.",
        "{COMPANY} invests {MONEY} in {PROJECT_NAME}, completion {DATE}.",
        "{RIGHT_INFO} in {TITLE} belongs to {COMPANY} per {LAW_REFERENCE}.",
        "{NAME} ({POSITION}) signed {CONSENT_TYPE} on {DATE} -> {EMAIL}.",
        "{COMPANY} @ {ADDRESS}, Tel: {PHONE}, Web: {URL}.",
        "Transfer {RIGHT_INFO} of {QUANTITY} {TYPE} for {MONEY}.",
        "Party A: {COMPANY}, Party B: {NAME}, Date: {DATE}",
        "Details: {NAME}, {ID_NUM}, {ADDRESS}",
        "Info: 1.{NAME} 2.{PHONE} 3.{EMAIL}",
        "Certificate: {NAME} joined {COMPANY} on {DATE}.",
        "Attn: {NAME} {POSITION}, {DEPARTMENT}, {COMPANY}",
        "Deposit {MONEY} on {DATE}, Balance {MONEY} on {DATE}."
    ]

    # Select templates based on dataset_type
    if dataset_type == 'train':
        single = train_single_templates
        dual = train_dual_templates
        multi = train_multi_templates
    else:
        # For dev/test, use different templates to test generalization
        single = dev_single_templates
        dual = dev_dual_templates
        multi = dev_multi_templates
    
    all_templates = build_template_list(single, dual, multi)
    
    # Add "Negative" samples (sentences with NO entities)
    # This helps precision by teaching the model what is NOT an entity.
    negative_templates = [
        "안녕하세요, 반갑습니다.", "오늘 날씨가 참 좋네요.", "식사는 하셨나요?",
        "회의는 2시에 시작합니다.", "문서를 검토해 주세요.", "확인 부탁드립니다.",
        "감사합니다.", "수고하셨습니다.", "다음에 뵙겠습니다.",
        "이 내용은 중요합니다.", "참고하시기 바랍니다.", "문의사항이 있으시면 연락주세요.",
        "첨부파일을 확인하세요.", "작업이 완료되었습니다.", "오류가 발생했습니다.",
        "시스템 점검 중입니다.", "잠시 후 다시 시도해 주세요.", "로그인이 필요합니다.",
        "회원가입을 환영합니다.", "비밀번호를 변경해 주세요."
    ]
    
    samples = []
    seen_texts = set()
    
    # Add negative samples (approx 10% of data)
    num_negatives = int(num_samples * 0.1)
    for _ in range(num_negatives):
        text = random.choice(negative_templates)
        # Add some random noise to negative samples too
        if random.random() < 0.5:
            text += f" ({random.randint(1,100)})"
        # Use the text itself as the template key for negatives
        samples.append({"text": text, "entities": [], "template": "NEGATIVE_SAMPLE"})
        seen_texts.add(text)
        
    remaining_samples = num_samples - len(samples)
    
    if balanced:
        # 균형 데이터 생성 로직
        print(f"[DataGen] Generating balanced data for {len(ENTITY_TYPES)} entity types ({dataset_type})...")
        samples_per_entity = max(1, math.ceil(remaining_samples / len(ENTITY_TYPES)))
        
        for entity_type in ENTITY_TYPES:
            # 해당 엔티티를 포함하는 템플릿 필터링
            relevant = [(t, e) for t, e in all_templates if entity_type in e]
            if not relevant: relevant = all_templates
            
            count = 0
            attempts = 0
            random.shuffle(relevant)
            
            while count < samples_per_entity and attempts < samples_per_entity * 5:
                attempts += 1
                template, _ = random.choice(relevant)
                text, entity_list, tmpl_str = generate_sample_from_template(template, entity_generators)
                
                if text not in seen_texts:
                    seen_texts.add(text)
                    samples.append({"text": text, "entities": entity_list, "template": tmpl_str})
                    count += 1
                    
        # 부족하거나 넘치는 경우 처리
        if len(samples) > num_samples:
            random.shuffle(samples)
            samples = samples[:num_samples]
        elif len(samples) < num_samples:
            while len(samples) < num_samples:
                template, _ = random.choice(all_templates)
                text, entity_list, tmpl_str = generate_sample_from_template(template, entity_generators)
                if text not in seen_texts:
                    seen_texts.add(text)
                    samples.append({"text": text, "entities": entity_list, "template": tmpl_str})
    else:
        # 단순 랜덤 생성
        while len(samples) < num_samples:
            template, _ = random.choice(all_templates)
            text, entity_list, tmpl_str = generate_sample_from_template(template, entity_generators)
            if text not in seen_texts:
                seen_texts.add(text)
                samples.append({"text": text, "entities": entity_list, "template": tmpl_str})
                
    print(f"[DataGen] Generated {len(samples)} unique samples ({dataset_type}).")
    return samples

def write_bio_file(samples: List[Dict], filepath: Union[str, Path]) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        for sample in samples:
            text = sample.get('text', '')
            entities = sample.get('entities', [])
            
            # 문자 단위 라벨링
            labels = ['O'] * len(text)
            for entity_text, entity_type in entities:
                start = text.find(entity_text)
                if start != -1:
                    end = start + len(entity_text)
                    labels[start] = f"B-{entity_type}"
                    for i in range(start+1, end):
                        labels[i] = f"I-{entity_type}"
            
            # 토큰 단위 (어절) 변환
            tokens = text.split()
            for token in tokens:
                # Placeholder (실제 로직은 ner_test.py의 write_bio_word_level에서 수행)
                f.write(f"{token}\tO\n") 
            f.write("\n")

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    pass

if __name__ == "__main__":
    main()
