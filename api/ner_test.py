#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stable NER Training/Eval/Inference (PyTorch + Transformers)
- mBERT(local path supported) + BiLSTM + token classifier
- BIO scheme for 23 entity types
- Synthetic data generation with entity pools + templates
- Deterministic, cacheable, robust token/label alignment (Fast tokenizer only)

CLI:
  python ner_test.py --mode train
  python ner_test.py --mode eval
  python ner_test.py --mode both

Env vars:
  MODE=both|train|eval
  CONTINUE_TRAINING=true|false
  BERT_DIR=/path/to/pretrained_bert
  STRICT_LOCAL_BERT=1   (if set, do NOT fallback to HF; error out instead)
  TRANSFORMERS_OFFLINE=1 / HF_HUB_OFFLINE=1  (offline mode)
"""

import os
import re
import json
import time
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel

# Simple stopwords and heuristics for post-processing
NAME_STOPWORDS = {"나", "인력", "사업", "가", "동의", "양도자", "양수자", "본", "대표자", "이름", "성명", "양도인", "양수인", "대표자명"}

def is_korean_name(s: str) -> bool:
    """Check if string is a valid Korean name (2-5 Hangul chars)"""
    s = s.strip().replace(" ", "").replace("\n", "")
    return 2 <= len(s) <= 5 and re.fullmatch(r"[가-힣]+", s) is not None

def extract_names_with_regex(text: str) -> List[str]:
    """Extract names using multiple regex patterns and heuristics"""
    # Normalize text: remove excessive whitespace/newlines for matching
    norm_text = re.sub(r'\s+', ' ', text)
    
    found: List[str] = []
    
    # Pattern 1: "성명: NAME" or "성명 NAME" (with flexible whitespace)
    for m in re.finditer(r"성명[:\s]+([가-힣]{2,5})", norm_text, re.IGNORECASE):
        found.append(m.group(1))
    
    # Pattern 2: "대표자명: NAME"
    for m in re.finditer(r"대표자명[:\s]+([가-힣]{2,5})", norm_text, re.IGNORECASE):
        found.append(m.group(1))
    
    # Pattern 3: "양도인" block - NAME appears after "양도인" within 10 chars
    for m in re.finditer(r"양도인[:\s\(]*([가-힣]{2,5})?", norm_text):
        if m.group(1):
            found.append(m.group(1))
        else:
            # Try to get next word after "양도인"
            start = m.end()
            rest = norm_text[start:start+20]
            name_match = re.search(r"[^\s]*([가-힣]{2,5})[^\s]*", rest)
            if name_match:
                found.append(name_match.group(1))
    
    # Pattern 4: "이름: NAME" or "이름 NAME"
    for m in re.finditer(r"이름[:\s]+([가-힣]{2,5})", norm_text, re.IGNORECASE):
        found.append(m.group(1))
    
    # Pattern 5: "동의자: NAME"
    for m in re.finditer(r"동의자[:\s]+([가-힣]{2,5})", norm_text, re.IGNORECASE):
        found.append(m.group(1))
    
    # Pattern 6: Repeated name "NAME NAME" (일반적 패턴)
    for m in re.finditer(r"([가-힣]{2,5})\s+\1", norm_text):
        found.append(m.group(1))
    
    # Pattern 7: "대표자" 다음에 오는 2-5글자 한글
    for m in re.finditer(r"대표자[:\s]*([가-힣]{2,5})", norm_text):
        found.append(m.group(1))
    
    # Pattern 8: "(서명)" 앞 한글 (서명 직전 이름)
    for m in re.finditer(r"([가-힣]{2,5})\s*\(서명\)", norm_text):
        found.append(m.group(1))
    
    # 기관명과 붙어있는 회사명 앞의 이름 제거 (나라지식정 같은 것)
    found = [f.strip() for f in found]
    found = [f for f in found if is_korean_name(f) and f not in NAME_STOPWORDS and len(f) <= 5]
    
    return sorted(set(found))

def cleanup_entities(all_entities: Dict[str, List[str]], text: str) -> Dict[str, List[str]]:
    """
    추출된 엔티티를 정제하고 검증합니다.
    """
    # Normalize text for pattern matching
    norm_text = re.sub(r'\s+', ' ', text)
    
    # NAME cleanup + regex union
    names = []
    for n in all_entities.get("NAME", []):
        n_clean = n.replace(".", "").strip()
        # 한글 이름 검증
        if is_korean_name(n_clean) and n_clean not in NAME_STOPWORDS:
            names.append(n_clean)
    
    # Regex로 이름 추출 (보조 수단)
    regex_names = extract_names_with_regex(text)
    names = sorted(set(names) | set(regex_names))
    all_entities["NAME"] = names if names else ["N/A"]

    # COMPANY cleanup
    comp = []
    comp_stop = {"기관명", "대표자", "1층", "본", "주", "이름", "대표자명", "연락처", "주소", "소속"}
    for c in all_entities.get("COMPANY", []):
        c_clean = c.strip()
        if len(c_clean) < 2:
            continue
        # 조사 제거 (는, 은, 이, 가, 에, 에서, 에게, 의 등)
        c_clean = re.sub(r'[는은이가에에서에게의과와]$', '', c_clean).strip()
        if c_clean in comp_stop:
            continue
        if not re.search(r"[가-힣]", c_clean):
            continue
        if re.match(r"^[가-힣]$", c_clean):
            continue
        comp.append(c_clean)
    
    # Regex for company names
    regex_comps: List[str] = []
    
    # Pattern 1: "기관명 : COMPANY" 또는 "기관명: COMPANY"
    for m in re.finditer(r"기관명\s*[:：]\s*(?:\(주\))?\s*([가-힣A-Za-z0-9()]{2,})", norm_text):
        comp_name = m.group(1).strip()
        if comp_name not in comp_stop:
            regex_comps.append(comp_name)
    
    # Pattern 2: "소속 : COMPANY"
    for m in re.finditer(r"소속\s*[:：]\s*([가-힣A-Za-z0-9()]{2,})", norm_text):
        comp_name = m.group(1).strip()
        if comp_name not in comp_stop:
            regex_comps.append(comp_name)
    
    # Pattern 3: "(주)..." 형태
    for m in re.finditer(r"\(주\)\s*([가-힣A-Za-z0-9]{2,})", norm_text):
        regex_comps.append(m.group(1).strip())
    
    all_comp = sorted(set(comp) | set(regex_comps))
    all_comp = [c for c in all_comp if len(c.strip()) > 1 and c not in comp_stop]
    all_entities["COMPANY"] = all_comp if all_comp else ["N/A"]

    # PHONE cleanup
    phones = []
    phone_pattern = re.compile(r'^0\d{1,2}-?\d{3,4}-?\d{4}$|^\d{2,3}-?\d{3,4}-?\d{4}$')
    for p in all_entities.get("PHONE", []):
        p_clean = re.sub(r'[^\d-]', '', p.strip())
        if phone_pattern.match(p_clean) or (p_clean.replace('-', '').isdigit() and 10 <= len(p_clean.replace('-', '')) <= 11):
            phones.append(p_clean)
    all_entities["PHONE"] = sorted(set(phones)) if phones else ["N/A"]

    # ADDRESS cleanup - 주소 형식 검증
    addresses = []
    for addr in all_entities.get("ADDRESS", []):
        addr_clean = addr.strip()
        # 최소 길이 및 한글 포함 검증
        if len(addr_clean) >= 5 and re.search(r"[가-힣]", addr_clean):
            # 시/도, 시/군/구 등이 포함되어야 함
            if re.search(r'(시|도|구|군|읍|면|동|리)', addr_clean):
                addresses.append(addr_clean)
    all_entities["ADDRESS"] = sorted(set(addresses)) if addresses else ["N/A"]

    # CONSENT_TYPE cleanup - 너무 짧거나 이상한 것 제거
    consent_types = []
    for ct in all_entities.get("CONSENT_TYPE", []):
        ct_clean = ct.strip()
        # 최소 길이 및 의미 있는 단어 포함
        if len(ct_clean) >= 3 and re.search(r'[가-힣]', ct_clean):
            # 너무 긴 것 제거 (OCR 오류 가능성)
            if len(ct_clean) <= 50:
                consent_types.append(ct_clean)
    all_entities["CONSENT_TYPE"] = sorted(set(consent_types)) if consent_types else ["N/A"]

    # DESCRIPTION cleanup
    descriptions = []
    for desc in all_entities.get("DESCRIPTION", []):
        desc_clean = desc.strip()
        # 최소 길이 및 의미 있는 내용
        if len(desc_clean) >= 3:
            # 너무 긴 것 제거
            if len(desc_clean) <= 100:
                descriptions.append(desc_clean)
    all_entities["DESCRIPTION"] = sorted(set(descriptions)) if descriptions else ["N/A"]

    return all_entities


# -----------------------------
# 0) Repro / GPU setup
# -----------------------------
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_gpu() -> torch.device:
    """
    GPU 전용 모드. CUDA를 사용할 수 없으면 에러 발생.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("[ERROR] CUDA를 사용할 수 없습니다. GPU가 필요합니다.")
    
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[INFO] ✅ GPU 사용 모드: {gpu_name}", flush=True)
    print(f"[INFO] CUDA version: {torch.version.cuda}", flush=True)
    print(f"[INFO] PyTorch version: {torch.__version__}", flush=True)
    return device


def gpu_warmup(device: torch.device) -> None:
    """Warmup GPU to ensure everything is initialized."""
    try:
        a = torch.randn(1024, 1024, device=device)
        b = torch.matmul(a, a)
        _ = b.cpu()
        print("[INFO] GPU warmup OK", flush=True)
    except Exception as e:
        raise RuntimeError(f"[ERROR] GPU warmup failed: {e}")


# -----------------------------
# 1) Config
# -----------------------------
@dataclass
class Config:
    root_dir: Path = Path(__file__).resolve().parent

    # Data
    # Data - 대폭 증강
    num_samples: int = 50000  # 10,000 → 50,000
    min_entities_per_type: int = 200  # 100 → 200
    max_len: int = 128
    train_ratio: float = 0.85
    seed: int = 42

    # Model
    lr: float = 3e-5
    epochs: int = 15  # 10 → 15 (더 많은 학습)
    batch_size: int = 32
    lstm_units: int = 256
    dropout: float = 0.2

    # Paths
    model_dir: Path = root_dir / "models/ner_bilstm_pytorch"
    cache_dir: Path = root_dir / "data/cache_ner"
    ocr_dir: Path = root_dir / "data/out/ocr/naver"
    out_dir: Path = root_dir / "conc/final"

    # BERT local priority
    container_bert: Path = Path("/app/models/pretrained_bert")
    volume_bert: Path = root_dir / "models/pretrained_bert"
    hf_fallback: str = "bert-base-multilingual-cased"


CFG = Config()

ENTITY_TYPES = [
    "NAME","PHONE","ADDRESS","DATE","COMPANY","EMAIL","POSITION","CONTRACT_TYPE","CONSENT_TYPE",
    "RIGHT_INFO","MONEY","PERIOD","PROJECT_NAME","LAW_REFERENCE","ID_NUM","TITLE","URL",
    "DESCRIPTION","TYPE","STATUS","DEPARTMENT","LANGUAGE","QUANTITY"
]

BIO_LABELS = ["O"]
for t in ENTITY_TYPES:
    BIO_LABELS.append(f"B-{t}")
    BIO_LABELS.append(f"I-{t}")

LABEL2ID = {l: i for i, l in enumerate(BIO_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
IGNORE_INDEX = -100


# -----------------------------
# 2) Utilities
# -----------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _is_valid_bert_dir(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    has_cfg = (p / "config.json").exists()
    has_tok = (p / "vocab.txt").exists() or (p / "tokenizer.json").exists()
    return has_cfg and has_tok


def pick_bert_path(override: Optional[str] = None) -> str:
    if override:
        p = Path(override).expanduser()
        if _is_valid_bert_dir(p):
            print(f"[INFO] ✅ --bert-dir 사용: {p}", flush=True)
            return str(p)
        raise RuntimeError(f"--bert-dir is set but invalid: {p}")

    env_dir = os.getenv("BERT_DIR")
    if env_dir:
        p = Path(env_dir).expanduser()
        if _is_valid_bert_dir(p):
            print(f"[INFO] ✅ BERT_DIR 사용: {p}", flush=True)
            return str(p)
        raise RuntimeError(f"BERT_DIR is set but invalid: {p}")

    candidates = [CFG.container_bert, CFG.volume_bert]
    for c in candidates:
        if _is_valid_bert_dir(Path(c)):
            print(f"[INFO] ✅ 로컬 BERT 발견: {c}", flush=True)
            return str(c)

    strict = os.getenv("STRICT_LOCAL_BERT", "").strip().lower() in ("1", "true", "yes", "y", "on")
    if strict:
        raise RuntimeError(
            "STRICT_LOCAL_BERT=1 이지만 로컬 BERT를 찾지 못했습니다.\n"
            f"후보 경로: {candidates}\n"
            "해결: BERT_DIR=/path/to/pretrained_bert 를 지정하거나 models/pretrained_bert를 준비하세요."
        )

    print(f"[INFO] 로컬 BERT 없음. HuggingFace에서 모델 다운로드/사용: {CFG.hf_fallback}", flush=True)
    return CFG.hf_fallback


def warn_if_running_on_mntc() -> None:
    p = str(CFG.root_dir).replace("\\", "/")
    if p.startswith("/mnt/"):
        print("[WARN] WSL에서 /mnt/*(Windows 드라이브)에서 실행 중입니다. "
              "데이터 생성/캐시/저장 속도가 느려 '멈춘 것처럼' 보일 수 있습니다.\n"
              "가능하면 프로젝트를 ~/work/... 같은 리눅스 홈으로 복사해서 실행하세요.", flush=True)


# -----------------------------
# 3) Entity pools (synthetic)
# -----------------------------
KOR_SYLLABLES = list("가나다라마바사아자차카타파하거너더러머버서어저처커터퍼허")
KOR_SURNAMES = ["김","이","박","최","정","강","조","윤","장","임","오","한","신","서","권","황","안","송","류","전","홍"]


def rnd_kor_name(rng: np.random.RandomState) -> str:
    surname = rng.choice(KOR_SURNAMES)
    given_len = 2 if rng.rand() < 0.6 else 3
    given = "".join(rng.choice(KOR_SYLLABLES) for _ in range(given_len))
    return surname + given


def rnd_phone(rng: np.random.RandomState) -> str:
    if rng.rand() < 0.7:
        return f"010-{rng.randint(1000,9999)}-{rng.randint(1000,9999)}"
    area = rng.choice(["02","031","032","051","053","062","064","070"])
    mid = rng.randint(100,9999)
    last = rng.randint(1000,9999)
    if mid < 1000:
        return f"{area}-{mid:03d}-{last:04d}"
    return f"{area}-{mid:04d}-{last:04d}"


def rnd_date(rng: np.random.RandomState) -> str:
    y = rng.randint(2018, 2027)
    m = rng.randint(1, 13)
    d = rng.randint(1, 29)
    if rng.rand() < 0.5:
        return f"{y}년 {m}월 {d}일"
    return f"{y}-{m:02d}-{d:02d}"


def rnd_email(rng: np.random.RandomState) -> str:
    user = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), size=rng.randint(6,12)))
    dom = rng.choice(["gmail.com","naver.com","daum.net","company.co.kr","example.com"])
    return f"{user}@{dom}"


def rnd_url(rng: np.random.RandomState) -> str:
    dom = rng.choice(["example.com","company.com","service.kr","my-page.net"])
    path = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), size=rng.randint(4,10)))
    if rng.rand() < 0.5:
        return f"https://{dom}/{path}"
    return f"http://{dom}/{path}"


def rnd_money(rng: np.random.RandomState) -> str:
    v = rng.randint(1_000, 500_000_000)
    if rng.rand() < 0.5:
        return f"{v:,}원"
    return f"₩{v:,}"


def rnd_quantity(rng: np.random.RandomState) -> str:
    n = rng.randint(1, 5000)
    unit = rng.choice(["건","명","개","회","페이지","GB","MB","TB"])
    return f"{n}{unit}"


def rnd_address(rng: np.random.RandomState) -> str:
    si = rng.choice([
        "서울특별시","부산광역시","대구광역시","인천광역시","광주광역시","대전광역시","울산광역시",
        "경기도","강원도","충청북도","충청남도","전라북도","전라남도","경상북도","경상남도","제주특별자치도"
    ])
    gu = rng.choice(["중구","서구","남구","북구","동구","수성구","해운대구","마포구","강남구","은평구","수원시","성남시","고양시","창원시","청주시","전주시","포항시"])
    road = rng.choice(["테헤란로","세종대로","충정로","월드컵북로","성미산로","중앙로","산업로","학동로","강변북로"])
    num1 = rng.randint(1, 300)
    num2 = rng.randint(1, 50)
    if rng.rand() < 0.5:
        return f"{si} {gu} {road} {num1}-{num2}"
    return f"{si} {gu} {road} {num1}"


FIXED_POOLS = {
    "NAME": [
        # 실제 문서에서 발견된 이름들
        "강희주","손동수","이긴구","이진구","이한울","서필원","남광호","김진성","김주완",
        # 추가 한국 이름 (다양성)
        "김민준","이서연","박지훈","최수빈","정예린","조민서","윤하준","장서영","임도윤","오채원",
        "한지우","신유진","권준혁","황서현","안민우","송지민","류하윤","전준서","홍서아","노예준",
        "배지호","곽시우","성서우","구민재","방하은","표지안","탁소율","석우진","선다인","진아윤",
    ],
    "COMPANY": [
        # 실제 문서에서 발견된 회사들
        "(주)스튜디오수집","한국문화정보원","에이드미디어","제주콘텐츠진흥원","충남문화관광재단","국가유산청",
        # 추가 회사명
        "(주)데이터랩","(주)테크놀로지","네이버","카카오","삼성전자","LG전자","현대자동차","SK텔레콤",
        "한국저작권위원회","문화체육관광부","한국콘텐츠진흥원","한국저작권보호원",
        "(주)디지털큐브","(주)아트컴퍼니","(주)미디어랩","(주)크리에이티브","(주)비전소프트",
        "서울시청","경기도청","부산시청","대전시청","광주시청","인천시청",
    ],
    "ADDRESS": [
        # 실제 문서 기반 주소
        "서울특별시 마포구 월드컵북로 400, 6층","서울시 마포구 월드컵북로 400, 8층",
        "경기도 고양시 덕양구 청초로 66, B동 1006호","경기도 남양주시 천마산로 65-2",
        "경기도 고양시 품질보증, 1384번길 29","서울시 은평구 증산로9길 36-5",
        "서울시 마포구 성미산로89","청주시 흥덕구 복대동 충북 영조2차 아파트 203-1204",
        "경기도 안양시 동안구 달안로 153","서울 마포구 망원로6길 14",
        "제주특별자치도 제주시 정실3길 104 민포레 1층","제주특별자치도 제주시 구산서길 29",
        # 추가 주소
        "서울특별시 강남구 테헤란로 152","서울특별시 서초구 서초대로 398",
        "부산광역시 해운대구 센텀중앙로 79","대전광역시 유성구 대학로 99",
        "경기도 성남시 분당구 판교역로 235","경기도 수원시 영통구 광교중앙로 140",
    ],
    "PHONE": [
        "010-4560-5825","010-1234-5678","010-9876-5432","010-5555-1234","010-7890-4567",
        "02-1234-5678","031-123-4567","051-890-1234","053-456-7890","062-789-0123",
        "064-123-0456","070-1234-5678","032-567-8901","042-234-5678","052-345-6789",
    ],
    "EMAIL": [
        "contact@studio.co.kr","info@culture.or.kr","admin@media.com","support@company.kr",
        "manager@project.com","team@digital.co.kr","director@agency.kr","staff@creative.com",
    ],
    "POSITION": [
        "대표","이사","팀장","부장","과장","차장","연구원","주임","매니저","법무담당",
        "책임자","대표자","사업관리","메타관리","품질보증","사진영상촬영","사진촬영","영상촬영",
        "항공영상촬영","사진스캔","복원","총괄","담당자","실장","본부장","센터장",
    ],
    "CONTRACT_TYPE": [
        "양도","위탁","도급","용역","자문","라이선스","사용허락","계약","합의","협약","위임",
        "저작재산권양도","저작권양도","사용권허락","2차저작물작성권허락",
    ],
    "CONSENT_TYPE": [
        "개인정보 수집·이용 동의","제3자 제공 동의","처리위탁 동의","마케팅 수신 동의",
        "국외이전 동의","보유·이용기간 동의","저작권 양도 동의","초상권 사용 동의",
        "참여 동의","사업 참여 동의","계약 동의","협력 동의",
    ],
    "RIGHT_INFO": [
        "저작권","저작재산권","저작인격권","사용권","2차적저작물작성권","복제권","배포권",
        "공중송신권","전시권","공연권","대여권","공표권","성명표시권","동일성유지권",
        "초상권","퍼블리시티권","상표권","특허권","디자인권",
    ],
    "MONEY": [
        "10,000,000원","5,000,000원","3,000,000원","1,500,000원","500,000원",
        "₩10,000,000","₩5,000,000","₩3,000,000","₩1,000,000","₩500,000",
    ],
    "PERIOD": [
        "1년","2년","3년","6개월","3개월","1개월","계약기간 내","사업종료시까지",
        "2024년","2025년","2026년","2024년 말까지","2025년 12월 31일까지",
        "영구","무기한","5년","10년",
    ],
    "PROJECT_NAME": [
        "공공저작물 디지털 전환 구축 사업","공공저작물 디지털전환구축 사업","공공저작물디지털전환구축사업",
        "2024년 공공저작물 디지털 전환 사업","저작권 메타데이터 추출","문서 NER 고도화",
        "OCR 파이프라인 개선","계약서 분석 자동화","권리정보 정규화",
        "디지털 아카이브 구축","문화유산 디지털화","콘텐츠 메타데이터 관리",
    ],
    "LAW_REFERENCE": [
        "저작권법","저작권법 제24조","저작권법 제101조","개인정보 보호법","정보통신망법",
        "민법","상법","전자문서 및 전자거래 기본법","공공데이터의 제공 및 이용 활성화에 관한 법률",
        "문화예술진흥법","문화재보호법",
    ],
    "ID_NUM": [
        "ID-123456","ID-789012","ID-345678","DOC-2024-001","DOC-2024-002",
        "CONTRACT-2024-001","AGREEMENT-2024-001","REF-2024-001",
    ],
    "TITLE": [
        "계약서","양도계약서","저작재산권 양도 계약서","동의서","합의서","확약서",
        "요청서","신청서","위임장","보고서","사업계획서","제안서","협약서",
    ],
    "URL": [
        "https://www.culture.go.kr","http://www.copyright.or.kr","https://example.com/project",
        "http://company.co.kr/contract","https://portal.kr/document",
    ],
    "DESCRIPTION": [
        "사업의 사업관리 총괄","사업의 메타관리 품질보증","사업의 사진영상 촬영",
        "사업의 사진 촬영","사업의 사진스캔 복원","사업의 영상촬영","사업의 항공영상 촬영",
        "본 계약은 당사자 간의 권리·의무를 규정한다","세부 내용은 별첨을 따른다",
        "분쟁 발생 시 관할은 서울중앙지방법원으로 한다","저작재산권의 전부를 양도한다",
        "개인정보를 안전하게 보관한다","프로젝트 수행 및 관리","기술 자문 및 지원",
    ],
    "TYPE": [
        "문서","이미지","영상","음원","데이터","소프트웨어","소스코드",
        "사진저작물","영상저작물","음악저작물","어문저작물","미술저작물",
        "산출물","결과물","보고서","계약서","동의서",
    ],
    "STATUS": [
        "유효","만료","해지","갱신","진행중","완료","보류","승인","반려","검토중",
        "대기","처리완료","제출완료","확인중",
    ],
    "DEPARTMENT": [
        "기획팀","개발팀","연구팀","법무팀","인사팀","재무팀","영업팀","운영팀",
        "사업팀","관리팀","디자인팀","마케팅팀","총무팀","홍보팀",
    ],
    "LANGUAGE": [
        "한국어","영어","일본어","중국어","스페인어","프랑스어","독일어","러시아어",
        "한글","영문","일문","중문",
    ],
    "QUANTITY": [
        "1건","5건","10건","50건","100건","1,000건","10,000건",
        "1명","5명","10명","50명","100명",
        "1개","10개","100개","1,000개",
        "1GB","10GB","100GB","1TB","10TB",
        "1페이지","10페이지","100페이지","500페이지",
    ],
}


def build_entity_pool(rng: np.random.RandomState) -> Dict[str, List[str]]:
    """
    엔티티 풀을 효율적으로 구축합니다.
    FIXED_POOLS가 충분하면 그대로 사용하고, 부족한 경우에만 생성합니다.
    """
    pool: Dict[str, List[str]] = {}
    print(f"[INFO] 엔티티 풀 구축 중... (각 타입당 최소 {CFG.min_entities_per_type}개)", flush=True)
    
    # 생성 함수 매핑
    generators = {
        "NAME": lambda: rnd_kor_name(rng),
        "PHONE": lambda: rnd_phone(rng),
        "ADDRESS": lambda: rnd_address(rng),
        "DATE": lambda: rnd_date(rng),
        "EMAIL": lambda: rnd_email(rng),
        "URL": lambda: rnd_url(rng),
        "MONEY": lambda: rnd_money(rng),
        "QUANTITY": lambda: rnd_quantity(rng),
        "ID_NUM": lambda: f"ID-{rng.randint(100000,999999)}",
    }
    
    for idx, t in enumerate(ENTITY_TYPES, 1):
        n = CFG.min_entities_per_type
        vals = set()
        
        # FIXED_POOLS에 있으면 우선 사용
        if t in FIXED_POOLS:
            base = FIXED_POOLS[t]
            vals.update(base)
            
            # 부족하면 벡터화된 생성으로 빠르게 채움
            if len(vals) < n:
                needed = n - len(vals)
                # 한 번에 필요한 만큼 + 여유분 생성 (중복 고려)
                batch_size = min(needed * 3, 1000)  # 최대 1000개까지 한 번에 생성
                
                # 생성 함수 선택
                if t in generators:
                    gen_func = generators[t]
                else:
                    gen_func = lambda: f"{t}_{rng.randint(1, 999999)}"
                
                # 벡터화된 생성
                generated = [gen_func() for _ in range(batch_size)]
                
                # 중복 제거하면서 추가
                for g in generated:
                    if len(vals) >= n:
                        break
                    vals.add(g)
                
                # 여전히 부족하면 추가 생성 (최대 100회 시도)
                attempts = 0
                while len(vals) < n and attempts < 100:
                    vals.add(gen_func())
                    attempts += 1
            
            print(f"[INFO] {idx}/{len(ENTITY_TYPES)} {t}: {len(vals)}개 (고정풀 + 생성)", flush=True)
        
        # FIXED_POOLS에 없으면 벡터화된 생성
        else:
            # 생성 함수 선택
            if t in generators:
                gen_func = generators[t]
            else:
                gen_func = lambda: f"{t}_{rng.randint(1, 999999)}"
            
            # 한 번에 필요한 만큼 + 여유분 생성
            batch_size = min(n * 3, 1000)  # 최대 1000개까지 한 번에 생성
            
            # 벡터화된 생성
            generated = [gen_func() for _ in range(batch_size)]
            
            # 중복 제거하면서 추가
            for g in generated:
                if len(vals) >= n:
                    break
                vals.add(g)
            
            # 여전히 부족하면 추가 생성 (최대 100회 시도)
            attempts = 0
            while len(vals) < n and attempts < 100:
                vals.add(gen_func())
                attempts += 1
            
            print(f"[INFO] {idx}/{len(ENTITY_TYPES)} {t}: {len(vals)}개 (생성)", flush=True)
        
        pool[t] = list(vals)
    
    print(f"[INFO] ✅ 엔티티 풀 구축 완료", flush=True)
    return pool


# -----------------------------
# 4) Template-based sentence synthesis + BIO tagging (대폭 확장)
# -----------------------------
TEMPLATES = [
    # 기본 문장 (15개)
    "{NAME}은 {COMPANY}에서 {PHONE} 번호를 사용한다.",
    "{NAME}님의 이메일은 {EMAIL}이고 연락처는 {PHONE}입니다.",
    "본 {TITLE}는 {COMPANY}와 {NAME} 간 {CONTRACT_TYPE} 계약입니다.",
    "{DATE}에 {NAME}이 {COMPANY}에서 {PROJECT_NAME} 프로젝트를 시작했다.",
    "담당자 {NAME}({POSITION})의 연락처는 {PHONE}이며 주소는 {ADDRESS}입니다.",
    "{COMPANY} {DEPARTMENT} {POSITION} {NAME}이 {DATE}에 계약금 {MONEY}을 지불했다.",
    "{NAME}은 {COMPANY}에 {TYPE} 자료 {QUANTITY}를 {LANGUAGE}로 제공한다.",
    "참조 URL은 {URL}이고 문서번호는 {ID_NUM}이다.",
    "{CONTRACT_TYPE} 계약서에 따라 {NAME}이 {RIGHT_INFO}를 {COMPANY}에 양도한다.",
    "{DATE}부터 {PERIOD} 동안 {NAME}은 {COMPANY}에서 {POSITION}으로 근무한다.",
    "{CONSENT_TYPE}에 동의한 {NAME}의 개인정보는 {COMPANY}가 {PERIOD} 보관한다.",
    "관련 법령 {LAW_REFERENCE}에 따라 {COMPANY}는 {NAME}에게 {MONEY}를 지급한다.",
    "{COMPANY}의 {DEPARTMENT}는 {PROJECT_NAME}에 대해 {STATUS} 상태로 보고했다.",
    "{NAME}이 작성한 {TYPE} 문서는 {DATE}에 {COMPANY}에 제출되었다.",
    "주소 {ADDRESS}에 위치한 {COMPANY}는 {PHONE}로 연락 가능하다.",
    
    # 법률 문서 스타일 (30개)
    "책임자(대표자) {POSITION} {NAME} (인)",
    "기관(개인)명 : {NAME} (인) {COMPANY} 소속 : ○",
    "{NAME} {COMPANY} {ADDRESS} {PHONE} 사업의 {DESCRIPTION}",
    "이름 {NAME} 소속 {COMPANY} 주소 {ADDRESS} 주요 업무 {DESCRIPTION}",
    "{NAME}은 {COMPANY}의 {PROJECT_NAME}에 참여하여 {DESCRIPTION}를 수행한다.",
    "저작재산권을 {NAME}으로부터 {COMPANY}에게 {CONTRACT_TYPE}한다.",
    "{NAME}이 {COMPANY}에 {RIGHT_INFO}를 {CONTRACT_TYPE}하기로 합의하였다.",
    "{DATE}부터 {PERIOD}까지 {NAME}은 {PROJECT_NAME} 프로젝트 {POSITION}으로 활동한다.",
    "{COMPANY} {DEPARTMENT} {NAME}({PHONE})이 {DATE}에 {TYPE} 문서를 제출했다.",
    "연락처 {PHONE} {NAME} {POSITION} {COMPANY} {ADDRESS}",
    "{NAME} {COMPANY} 본 {PROJECT_NAME} {POSITION} {DESCRIPTION}",
    "참여인력 {NAME} {COMPANY} {ADDRESS} {PHONE} 주요업무 {DESCRIPTION}",
    "{NAME}은(는) {DATE}에 {COMPANY}와 {CONTRACT_TYPE} 계약을 체결하였다.",
    "{CONSENT_TYPE} 동의서에 {NAME}이 {DATE}에 서명하였다.",
    "{NAME}의 {RIGHT_INFO}는 {COMPANY}가 {PERIOD} 동안 보유한다.",
    "{LAW_REFERENCE}에 의거하여 {NAME}은 {COMPANY}에 {MONEY}를 지급한다.",
    "{PROJECT_NAME} 사업 {POSITION} {NAME} {COMPANY} {PHONE}",
    "대표자 {NAME} 주소 {ADDRESS} 연락처 {PHONE} 소속 {COMPANY}",
    "{NAME}이 {COMPANY}의 {POSITION}으로 {DATE}부터 근무한다.",
    "{COMPANY} 소속 {NAME}은 {PROJECT_NAME}의 {POSITION}을 담당한다.",
    
    # 표 형식 스타일 (20개)
    "번호 이름 {NAME} 소속 {COMPANY} 연락처 {PHONE}",
    "{NAME} {COMPANY} {PHONE} {ADDRESS} {POSITION}",
    "성명 {NAME} 생년월일 {DATE} 전화번호 {PHONE}",
    "{NAME} (인) {COMPANY} {DATE} {CONTRACT_TYPE}",
    "참여자 {NAME} 기관 {COMPANY} 업무 {DESCRIPTION}",
    "{POSITION} {NAME} {PHONE} {EMAIL} {ADDRESS}",
    "{NAME} | {COMPANY} | {PHONE} | {POSITION}",
    "이름: {NAME} 회사: {COMPANY} 연락처: {PHONE}",
    "{NAME} - {POSITION} - {COMPANY} - {PHONE}",
    "담당 {NAME} 부서 {DEPARTMENT} 직책 {POSITION}",
    
    # OCR 오류 시뮬레이션 (띄어쓰기 없음/이상한 띄어쓰기) (20개)
    "{NAME}은{COMPANY}에서{POSITION}으로근무한다.",
    "{NAME} 님 의 연 락 처 는 {PHONE} 입 니 다.",
    "{COMPANY}{NAME}{PHONE}{ADDRESS}",
    "기관명:{COMPANY}담당자:{NAME}전화:{PHONE}",
    "{NAME}이{DATE}에{COMPANY}와계약체결하였다.",
    "{PROJECT_NAME}사업{NAME}{COMPANY}{POSITION}",
    "{NAME} ( {POSITION} ) {COMPANY} {PHONE}",
    "성 명{NAME}소 속{COMPANY}연락처{PHONE}",
    "{NAME}은 {COMPANY} 의{PROJECT_NAME}에 참 여한다.",
    "{CONTRACT_TYPE}계약서 {NAME} {COMPANY} {DATE}",
    
    # 복잡한 복합 문장 (30개)
    "{NAME}은 {COMPANY}의 {DEPARTMENT} {POSITION}으로서 {PROJECT_NAME} 프로젝트를 {DATE}부터 {PERIOD}까지 수행하며, 연락처는 {PHONE}이고 이메일은 {EMAIL}이다.",
    "{DATE}에 {NAME}({POSITION})은 {COMPANY}와 {CONTRACT_TYPE} 계약을 체결하였으며, {RIGHT_INFO}에 대한 권리를 {PERIOD} 동안 양도하기로 합의하였다.",
    "{COMPANY} 소속 {NAME}은 {ADDRESS}에 거주하며, {PHONE}로 연락 가능하고, {PROJECT_NAME}의 {POSITION}을 맡고 있다.",
    "{CONSENT_TYPE}에 동의한 {NAME}은 {COMPANY}에 {DATE}부터 {PERIOD}까지 개인정보 활용을 허가하였으며, 연락처는 {PHONE}이다.",
    "{LAW_REFERENCE}에 따라 {COMPANY}는 {NAME}에게 {MONEY}를 {DATE}까지 지급해야 하며, 주소는 {ADDRESS}이다.",
    "{NAME}이 {DATE}에 제출한 {TYPE} 문서는 {COMPANY}의 {DEPARTMENT}에서 {STATUS} 상태로 검토 중이다.",
    "{PROJECT_NAME} 사업의 참여자 {NAME}은 {COMPANY} {POSITION}으로 {DESCRIPTION}를 담당하며, {PHONE}로 연락 가능하다.",
    "{COMPANY}와 {NAME} 간 {CONTRACT_TYPE} 계약에 따라 {RIGHT_INFO}가 {DATE}부터 {PERIOD} 동안 양도되었다.",
    "{NAME}({PHONE})은 {COMPANY}의 {PROJECT_NAME}에서 {POSITION}으로 활동하며, 주소는 {ADDRESS}이다.",
    "{DATE}에 {NAME}이 {COMPANY}에 제출한 {CONSENT_TYPE} 동의서는 {DEPARTMENT}에서 {STATUS} 처리되었다.",
    
    # 불완전한 문장 / 단편적 표현 (20개)
    "{NAME} {COMPANY} {POSITION}",
    "담당: {NAME} ({PHONE})",
    "{NAME}, {COMPANY}, {DATE}",
    "연락처 {PHONE} 담당자 {NAME}",
    "{COMPANY} {NAME} {ADDRESS}",
    "{PROJECT_NAME} - {NAME} - {POSITION}",
    "{NAME} (인) {DATE}",
    "{COMPANY} 소속: {NAME}",
    "{POSITION} {NAME} 연락 {PHONE}",
    "{NAME} / {COMPANY} / {PHONE}",
    "책임자 {NAME} {COMPANY}",
    "{NAME} 주소 {ADDRESS}",
    "{PHONE} {EMAIL} {NAME}",
    "{NAME} ({COMPANY}) {POSITION}",
    "{DATE} {NAME} {CONTRACT_TYPE}",
    "{PROJECT_NAME} 담당 {NAME}",
    "{NAME} | {POSITION} | {PHONE}",
    "대표 {NAME} {COMPANY}",
    "{NAME} 이메일: {EMAIL}",
    "{COMPANY} {NAME} 참여",

    # 서명/양도인/양수인/대표자 블록 (실제 양식 대응)
    "양도인 {NAME} (서명) 성명: {NAME} 주소: {ADDRESS} 전화번호: {PHONE}",
    "양수인 {COMPANY} 대표자명: {NAME} 기관명: {COMPANY} 주소: {ADDRESS} 연락처: {PHONE}",
    "양도인 {NAME} 주소 {ADDRESS} 연락처 {PHONE} 양수인 {COMPANY} 대표 {NAME}",
    "대표자명: {NAME} 기관명: {COMPANY} 주소: {ADDRESS} 전화: {PHONE}",
    "성명 {NAME} 연락처 {PHONE} 주소 {ADDRESS} 양도인 서명",
    "성명: {NAME} (인) 전화번호: {PHONE} 주소: {ADDRESS}",
    "대표자 {NAME} (인) 회사 {COMPANY} 전화 {PHONE} 주소 {ADDRESS}",
    "양도자 {NAME} 양수자 {COMPANY} 대표 {NAME} 연락처 {PHONE}",
    "양도인 이름 {NAME} 전화 {PHONE} 주소 {ADDRESS}",
    "양수인 {COMPANY} 대표자 {NAME} 전화번호 {PHONE} 주소 {ADDRESS}",
    "서명: {NAME} / 소속: {COMPANY} / 주소: {ADDRESS} / 연락처: {PHONE}",
    "서명자 {NAME} ({PHONE}) {ADDRESS} 소속 {COMPANY}",

    # 개인정보/동의 양식 문장 (실제 문서 패턴)
    "개인정보 항목 성명: {NAME}, 주소: {ADDRESS}, 전화번호: {PHONE}",
    "개인정보 수집 및 제공에 동의합니다. 성명 {NAME} 연락처 {PHONE}",
    "제3자 제공 대상: {COMPANY}, 제공 항목: 성명 {NAME}, 연락처 {PHONE}, 주소 {ADDRESS}",
    "개인정보 보유기간: {PERIOD}, 담당자: {NAME}, 연락처: {PHONE}",
    "동의자: {NAME} 서명(인) 연락처: {PHONE}",
    "본 동의서는 {DATE}에 작성되었으며, 작성자 {NAME}, 연락처 {PHONE}",
    "본인은 {COMPANY}에 개인정보 제공(성명 {NAME}, 연락처 {PHONE}, 주소 {ADDRESS})에 동의합니다.",

    # Entity 없는 일반 문장 추가 (O 태그 비율 확대)
    "본 동의서는 개인정보 처리에 관한 일반적인 내용을 담고 있습니다.",
    "아래 빈칸에 필요한 내용을 기입하십시오.",
    "상기 사항을 확인하였으며 별도의 문의사항은 없습니다.",
    "첨부된 서류를 확인하시고 서명해 주세요.",
    "계약 조건 및 조항을 숙지하였습니다.",
    "관련 법령에 따라 처리됩니다.",
    "본 문서는 참고용으로 제공됩니다.",
    "필요 시 추가 정보를 요청할 수 있습니다.",
    "작성일자를 기입하고 서명하십시오.",
    "본 문서의 일부는 생략되었습니다.",
    "서명란에 자필 서명을 해 주세요.",
    
    # Entity 없는 일반 문장 (10개 - 중요!)
    "저작재산권 양도 계약서",
    "참여 인력 명단",
    "아래와 같이 계약을 체결한다.",
    "본 사업의 목적은 다음과 같다.",
    "관련 법령에 따라 처리한다.",
    "상기 내용에 동의합니다.",
    "주요 업무 내용",
    "사업 개요",
    "계약 조건",
    "첨부 서류 목록",
]

def simple_word_tokenize(text: str) -> List[str]:
    """
    공백 기반 단어 분리 - BERT tokenizer가 내부적으로 subword 처리
    한국어는 형태소 단위가 아닌 어절(공백 단위) 분리
    """
    # 공백으로 분리
    tokens = text.split()
    # 빈 토큰 제거
    return [t for t in tokens if t.strip()]


def render_template(template: str, pool: Dict[str, List[str]], rng: np.random.RandomState) -> Tuple[List[str], List[str]]:
    """
    템플릿을 렌더링하고 BIO 태그를 생성합니다.
    OCR 오류를 시뮬레이션하기 위해 일부 템플릿에서는 띄어쓰기를 제거합니다.
    """
    used: Dict[str, str] = {}
    text = template
    for t in ENTITY_TYPES:
        key = "{" + t + "}"
        if key in text:
            used[t] = rng.choice(pool[t])
            text = text.replace(key, used[t])

    # OCR 오류 시뮬레이션: 일부 템플릿에서 띄어쓰기 제거 (30% 확률)
    if rng.rand() < 0.3:
        # 엔티티 사이의 띄어쓰기만 제거 (엔티티 내부는 유지)
        for ent_type, ent_text in used.items():
            # 엔티티 앞뒤의 띄어쓰기 제거
            text = text.replace(f" {ent_text} ", f"{ent_text}")
            text = text.replace(f" {ent_text}", f"{ent_text}")
            text = text.replace(f"{ent_text} ", f"{ent_text}")

    words = simple_word_tokenize(text)
    labels = ["O"] * len(words)

    # 엔티티 매칭: 정확한 매칭과 부분 매칭 모두 시도
    for ent_type, ent_text in used.items():
        ent_words = simple_word_tokenize(ent_text)
        if not ent_words:
            continue
        
        # 정확한 매칭 시도
        i = 0
        matched = False
        while i <= len(words) - len(ent_words):
            if words[i:i+len(ent_words)] == ent_words:
                labels[i] = f"B-{ent_type}"
                for j in range(1, len(ent_words)):
                    labels[i+j] = f"I-{ent_type}"
                i += len(ent_words)
                matched = True
            else:
                i += 1
        
        # 정확한 매칭 실패 시, 엔티티가 하나의 단어로 합쳐진 경우 찾기
        if not matched and len(ent_text.replace(" ", "")) > 0:
            ent_combined = ent_text.replace(" ", "")
            for i, word in enumerate(words):
                if ent_combined in word or word in ent_combined:
                    # 부분 매칭: 단어가 엔티티를 포함하거나 그 반대인 경우
                    if len(ent_combined) >= len(word) * 0.7:  # 70% 이상 일치
                        labels[i] = f"B-{ent_type}"
                        matched = True
                        break

    return words, labels


def generate_bio_samples(num_samples: int, seed: int) -> List[Dict]:
    rng = np.random.RandomState(seed)
    pool = build_entity_pool(rng)
    print(f"[INFO] 샘플 생성 시작: {num_samples}개", flush=True)
    samples: List[Dict] = []

    initial_count = min(500, num_samples // 2)
    print(f"[INFO] 초기 샘플 생성: {initial_count}개", flush=True)
    for i in range(initial_count):
        tmpl = rng.choice(TEMPLATES)
        w, y = render_template(tmpl, pool, rng)
        samples.append({"tokens": w, "labels": y})
        if (i + 1) % 100 == 0:
            print(f"[INFO] 초기 샘플 진행: {i+1}/{initial_count}", flush=True)

    print(f"[INFO] 추가 샘플 생성: {num_samples - len(samples)}개", flush=True)
    last = time.time()
    while len(samples) < num_samples:
        tmpl = rng.choice(TEMPLATES)
        w, y = render_template(tmpl, pool, rng)
        samples.append({"tokens": w, "labels": y})

        if len(samples) % 500 == 0:
            now = time.time()
            dt = now - last
            last = now
            print(f"[INFO] 샘플 진행: {len(samples)}/{num_samples} (+500 in {dt:.1f}s)", flush=True)

    print(f"[INFO] ✅ 샘플 생성 완료: {len(samples)}개", flush=True)
    return samples[:num_samples]


# -----------------------------
# 5) Tokenization + label alignment
# -----------------------------
def load_tokenizer(bert_path: str):
    tok = AutoTokenizer.from_pretrained(bert_path, use_fast=True)
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("Tokenizer must be a Fast tokenizer (use_fast=True).")
    return tok


def align_labels_with_word_ids(encodings, word_labels: List[List[int]], max_len: int) -> np.ndarray:
    out = []
    for i in range(len(word_labels)):
        word_ids = encodings.word_ids(batch_index=i)
        aligned = np.full((max_len,), IGNORE_INDEX, dtype=np.int64)
        prev_wid = None
        for j, wid in enumerate(word_ids):
            if j >= max_len:
                break
            if wid is None:
                continue
            if wid != prev_wid:
                if wid < len(word_labels[i]):
                    aligned[j] = int(word_labels[i][wid])
            prev_wid = wid
        out.append(aligned)
    return np.stack(out, axis=0)


def tokenize_bio_samples(samples: List[Dict], tokenizer, max_len: int):
    tokens_batch = [s["tokens"] for s in samples]
    labels_batch = [[LABEL2ID.get(l, 0) for l in s["labels"]] for s in samples]

    enc = tokenizer(
        tokens_batch,
            is_split_into_words=True,
            truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="np",
    )

    input_ids = enc["input_ids"].astype(np.int64)
    attention_mask = enc["attention_mask"].astype(np.int64)
    labels = align_labels_with_word_ids(enc, labels_batch, max_len)

    supervised = (labels != IGNORE_INDEX).sum(axis=1)
    keep = supervised > 0

    input_ids = input_ids[keep]
    attention_mask = attention_mask[keep]
    labels = labels[keep]
    return input_ids, attention_mask, labels


def compute_label_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.
    labels: (N, max_len) with label IDs and IGNORE_INDEX for padding.
    """
    flat = labels.reshape(-1)
    mask = flat != IGNORE_INDEX
    vals = flat[mask]
    counts = np.bincount(vals, minlength=len(BIO_LABELS)).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# -----------------------------
# 6) Balanced split
# -----------------------------
def sample_signature(labels_row: np.ndarray) -> Tuple[int, ...]:
    present = set()
    for lid in labels_row:
        if lid == IGNORE_INDEX:
            continue
        lab = ID2LABEL.get(int(lid), "O")
        if lab.startswith(("B-", "I-")):
            ent = lab[2:]
            if ent in ENTITY_TYPES:
                present.add(ENTITY_TYPES.index(ent))
    return tuple(sorted(present))


def split_train_val(input_ids, attention_mask, labels, train_ratio: float, seed: int):
    rng = np.random.RandomState(seed)
    n = labels.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    sigs = [sample_signature(labels[i]) for i in idx]
    want_val = max(1, int((1.0 - train_ratio) * n))

    val_idx = []
    covered = set()
    for i, s in zip(idx, sigs):
        if len(val_idx) >= want_val:
            break
        new = set(s) - covered
        if new:
            val_idx.append(i)
            covered |= set(s)

    remaining = [i for i in idx if i not in set(val_idx)]
    rng.shuffle(remaining)
    for i in remaining:
        if len(val_idx) >= want_val:
            break
        val_idx.append(i)

    val_set = set(val_idx)
    train_idx = [i for i in idx if i not in val_set]

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    
    return (
        input_ids[train_idx], attention_mask[train_idx], labels[train_idx],
        input_ids[val_idx], attention_mask[val_idx], labels[val_idx],
    )


# -----------------------------
# 7) PyTorch Dataset
# -----------------------------
class NERDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


# -----------------------------
# 8) PyTorch Model (mBERT + BiLSTM)
# -----------------------------
class BertBiLSTMNER(nn.Module):
    def __init__(self, bert_path: str, num_labels: int, lstm_units: int, dropout: float):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            self.bert.config.hidden_size,
            lstm_units,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.classifier = nn.Linear(lstm_units * 2, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, hidden)
        sequence_output = self.dropout(sequence_output)
        lstm_output, _ = self.lstm(sequence_output)  # (batch, seq_len, lstm_units*2)
        lstm_output = self.dropout(lstm_output)
        logits = self.classifier(lstm_output)  # (batch, seq_len, num_labels)
        return logits


# -----------------------------
# 9) Metrics (entity-level F1)
# -----------------------------
def decode_bio(label_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
    out = []
    label_ids = label_ids.cpu().numpy()
    attention_mask = attention_mask.cpu().numpy()
    for lid, m in zip(label_ids.tolist(), attention_mask.tolist()):
        if m == 0:
            break
        if lid == IGNORE_INDEX:
            out.append("O")
    else:
            out.append(ID2LABEL.get(int(lid), "O"))
    return out


def bio_to_entities(seq: List[str]) -> List[Tuple[str, int, int]]:
    ents = []
    i = 0
    while i < len(seq):
        tag = seq[i]
        if tag.startswith("B-"):
            t = tag[2:]
            j = i + 1
            while j < len(seq) and seq[j] == f"I-{t}":
                j += 1
            ents.append((t, i, j))
            i = j
        else:
            i += 1
    return ents


def entity_f1(y_true_seqs: List[List[str]], y_pred_seqs: List[List[str]]) -> Dict[str, float]:
    """
    엔티티 레벨 F1 - 타입과 위치가 정확히 일치하는 엔티티만 TP로 계산
    (타입, 시작위치, 끝위치) 튜플이 완전히 일치해야 정확한 매칭으로 간주
    """
    tp = fp = fn = 0
    
    for yt, yp in zip(y_true_seqs, y_pred_seqs):
        # 엔티티를 (타입, 시작위치, 끝위치) 튜플로 추출
        true_entities = set(bio_to_entities(yt))
        pred_entities = set(bio_to_entities(yp))
        
        # 정확히 일치하는 것만 TP (타입과 위치가 모두 일치)
        tp += len(true_entities & pred_entities)
        # 예측했지만 실제로는 없는 것 = FP
        fp += len(pred_entities - true_entities)
        # 실제로는 있지만 예측하지 못한 것 = FN
        fn += len(true_entities - pred_entities)
    
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1}


# -----------------------------
# 10) Save / Load
# -----------------------------
def save_artifacts(model: nn.Module, tokenizer, cfg: Config):
    ensure_dir(cfg.model_dir)
    torch.save(model.state_dict(), cfg.model_dir / "model.pt")
    tokenizer.save_pretrained(cfg.model_dir / "tokenizer")
    (cfg.model_dir / "labels.json").write_text(
        json.dumps(
            {"BIO_LABELS": BIO_LABELS, "ENTITY_TYPES": ENTITY_TYPES, "max_len": cfg.max_len},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[INFO] ✅ 모델 저장 완료: {cfg.model_dir}", flush=True)


def load_artifacts(cfg: Config, bert_path: str, device: torch.device):
    model_path = cfg.model_dir / "model.pt"
    tok_dir = cfg.model_dir / "tokenizer"
    labels_path = cfg.model_dir / "labels.json"
    if not model_path.exists() or not tok_dir.exists() or not labels_path.exists():
        return None, None

    meta = json.loads(labels_path.read_text(encoding="utf-8"))
    if meta.get("BIO_LABELS") != BIO_LABELS or meta.get("max_len") != cfg.max_len:
        print("[WARN] Saved label schema/max_len differs. Refusing to continue-training.", flush=True)
        return None, None

    model = BertBiLSTMNER(bert_path, len(BIO_LABELS), cfg.lstm_units, cfg.dropout)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    return model, tokenizer


# -----------------------------
# 11) Cache
# -----------------------------
def cache_key(cfg: Config, bert_path: str) -> str:
    s = f"{cfg.num_samples}|{cfg.min_entities_per_type}|{cfg.max_len}|{cfg.seed}|{bert_path}|pytorch_v1"
    return sha1_text(s)[:16]


def load_cached_dataset(cfg: Config, key: str):
    p = cfg.cache_dir / f"dataset_{key}.npz"
    if not p.exists():
        return None
    data = np.load(p)
    return data["input_ids"], data["attention_mask"], data["labels"]


def save_cached_dataset(cfg: Config, key: str, input_ids, attention_mask, labels):
    ensure_dir(cfg.cache_dir)
    p = cfg.cache_dir / f"dataset_{key}.npz"
    np.savez_compressed(p, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(f"[INFO] ✅ Cached dataset: {p}", flush=True)


# -----------------------------
# 12) Train
# -----------------------------
def train(continue_from_existing: bool, bert_dir_override: Optional[str] = None, do_warmup: bool = True):
    print("\n[INFO] 훈련 모드 시작...", flush=True)
    set_global_seed(CFG.seed)
    warn_if_running_on_mntc()
    device = setup_gpu()
    if do_warmup:
        gpu_warmup(device)

    bert_path = pick_bert_path(override=bert_dir_override)
    key = cache_key(CFG, bert_path)

    model = None
    tokenizer = None

    if continue_from_existing:
        print(f"[INFO] 기존 모델 확인 중...", flush=True)
        model, tokenizer = load_artifacts(CFG, bert_path, device)
        if model is not None:
            print(f"[INFO] ✅ 기존 모델을 계속 학습합니다.", flush=True)

    if tokenizer is None:
        print(f"[INFO] 토큰라이저 로드 중...", flush=True)
        tokenizer = load_tokenizer(bert_path)

    cached = load_cached_dataset(CFG, key)
    if cached is None:
        print(f"\n[INFO] 학습 데이터 생성 중... (23개 라벨, {CFG.num_samples}개 샘플)", flush=True)
        samples = generate_bio_samples(CFG.num_samples, seed=CFG.seed)
        print("[INFO] 토큰화 및 라벨 정렬 중...", flush=True)
        input_ids, attention_mask, labels = tokenize_bio_samples(samples, tokenizer, CFG.max_len)
        print(f"[INFO] ✅ 유효한 샘플: {labels.shape[0]}개", flush=True)
        save_cached_dataset(CFG, key, input_ids, attention_mask, labels)
    else:
        input_ids, attention_mask, labels = cached
        print(f"[INFO] ✅ 캠시된 데이터 로드완료: {labels.shape[0]}개", flush=True)

    tr_ids, tr_mask, tr_y, va_ids, va_mask, va_y = split_train_val(
        input_ids, attention_mask, labels, CFG.train_ratio, CFG.seed
    )
    print(f"[INFO] Train={len(tr_ids)}, Val={len(va_ids)}", flush=True)

    train_dataset = NERDataset(tr_ids, tr_mask, tr_y)
    val_dataset = NERDataset(va_ids, va_mask, va_y)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False)

    if model is None:
        print(f"[INFO] Building model...", flush=True)
        model = BertBiLSTMNER(bert_path, len(BIO_LABELS), CFG.lstm_units, CFG.dropout)
        print(f"[INFO] 모델을 GPU로 이동 중...", flush=True)
        model.to(device)
        print(f"[INFO] ✅ Model moved to {device}", flush=True)

    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    # Compute class weights to improve minority label learning
    weights = compute_label_weights(tr_y)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=weights.to(device))

    print(f"\n{'='*70}", flush=True)
    print(f"[INFO] 훈련 시작: {CFG.epochs} 에포크, 배치={CFG.batch_size}", flush=True)
    print(f"[INFO] 훈련 배치: {len(train_loader)}/에포크 | 검증 배치: {len(val_loader)}", flush=True)
    print(f"{'='*70}\n", flush=True)
    
    for epoch in range(CFG.epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        print(f"\n{'='*70}", flush=True)
        print(f"[EPOCH {epoch+1}/{CFG.epochs}] 학습 시작...", flush=True)
        print(f"{'='*70}", flush=True)
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            
            # Reshape for loss calculation
            loss = criterion(logits.view(-1, len(BIO_LABELS)), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # 10개 배치마다 진행 상황 출력
            if batch_count % 10 == 0:
                avg_loss = total_loss / batch_count
                progress = (batch_count / len(train_loader)) * 100
                print(f"  [{epoch+1}/{CFG.epochs}] Batch {batch_count:3d}/{len(train_loader)} ({progress:5.1f}%) | Loss: {avg_loss:.4f}", flush=True)
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"\n[EPOCH {epoch+1}/{CFG.epochs}] 훈련 완료 - Train Loss: {avg_train_loss:.4f}", flush=True)
        
        # Validation with metrics
        print(f"\n[EPOCH {epoch+1}/{CFG.epochs}] 검증 시작...", flush=True)
        model.eval()
        val_loss = 0.0
        y_true_seqs, y_pred_seqs = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits.view(-1, len(BIO_LABELS)), labels.view(-1))
                val_loss += loss.item()
                
                pred = torch.argmax(logits, dim=-1)
                for i in range(pred.shape[0]):
                    yt = decode_bio(labels[i].cpu(), attention_mask[i].cpu())
                    yp = decode_bio(pred[i].cpu(), attention_mask[i].cpu())
                    y_true_seqs.append(yt)
                    y_pred_seqs.append(yp)
        
        avg_val_loss = val_loss / len(val_loader)
        metrics = entity_f1(y_true_seqs, y_pred_seqs)
        
        print(f"\n{'='*70}", flush=True)
        print(f"[EPOCH {epoch+1}/{CFG.epochs}] 검증 완료", flush=True)
        print(f"  Train Loss: {avg_train_loss:.4f}", flush=True)
        print(f"  Val Loss:   {avg_val_loss:.4f}", flush=True)
        print(f"  Precision:  {metrics['precision']:.4f}", flush=True)
        print(f"  Recall:     {metrics['recall']:.4f}", flush=True)
        print(f"  F1 Score:   {metrics['f1']:.4f}", flush=True)
        print(f"{'='*70}\n", flush=True)

    # Save final model
    save_artifacts(model, tokenizer, CFG)


# -----------------------------
# 13) Inference
# -----------------------------
def predict_text(model: nn.Module, tokenizer, text: str, device: torch.device) -> Dict[str, List[str]]:
    """
    텍스트에서 엔티티를 추출합니다.
    원본 텍스트에서 직접 추출하여 띄어쓰기를 보존합니다.
    """
    model.eval()
    all_entities: Dict[str, List[str]] = {t: [] for t in ENTITY_TYPES}
    
    # 텍스트를 청크로 나누기 (max_len을 고려하여)
    max_chunk_chars = (CFG.max_len - 16) * 10
    text_chunks = []
    chunk_offsets = []  # 각 청크의 원본 텍스트에서의 시작 위치
    for i in range(0, len(text), max_chunk_chars):
        chunk = text[i:i + max_chunk_chars]
        text_chunks.append(chunk)
        chunk_offsets.append(i)

    with torch.no_grad():
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_offset = chunk_offsets[chunk_idx]
            
            # BERT tokenizer로 직접 인코딩 (offsets_mapping 포함)
            enc = tokenizer(
                chunk_text,
                truncation=True,
                padding="max_length",
                max_length=CFG.max_len,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            offsets = enc["offset_mapping"][0].cpu().numpy()  # (seq_len, 2) 형태: (start, end)
            
            logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            
            # word_ids를 사용하여 subword를 word로 그룹화
            try:
                word_ids = enc.word_ids(batch_index=0)
            except:
                word_ids = None
            
            # 엔티티 추출: word_ids를 사용하여 원본 텍스트에서 직접 추출
            if word_ids is not None:
                prev_wid = None
                cur_type = None
                cur_start = None
                cur_end = None
                
                for j in range(len(pred)):
                    if j >= len(attention_mask[0]) or attention_mask[0, j] == 0:
                        break
                    if j >= len(word_ids) or j >= len(offsets):
                        break
                    
                    wid = word_ids[j]
                    offset = offsets[j]
                    label_id = pred[j]
                    
                    # Special tokens 스킵
                    if wid is None or offset[0] == offset[1] == 0:
                        if cur_type and cur_start is not None and cur_end is not None:
                            # 이전 엔티티 저장
                            entity_text = chunk_text[cur_start:cur_end].strip()
                            if entity_text and len(entity_text) > 1:
                                all_entities[cur_type].append(entity_text)
                            cur_type = None
                            cur_start = None
                            cur_end = None
                        continue
                    
                    lab = ID2LABEL.get(int(label_id), "O")
                    
                    # 같은 word의 첫 subword만 처리
                    if wid == prev_wid:
                        # 같은 word의 subword인 경우, end 위치만 업데이트
                        if cur_type and cur_start is not None:
                            cur_end = offset[1]
                        continue
                    prev_wid = wid
                    
                    if lab.startswith("B-"):
                        # 이전 엔티티 저장
                        if cur_type and cur_start is not None and cur_end is not None:
                            entity_text = chunk_text[cur_start:cur_end].strip()
                            if entity_text and len(entity_text) > 1:
                                all_entities[cur_type].append(entity_text)
                        cur_type = lab[2:]
                        cur_start = offset[0]
                        cur_end = offset[1]
                    elif lab.startswith("I-") and cur_type == lab[2:]:
                        # 연속된 엔티티: end 위치만 업데이트
                        if cur_start is not None:
                            cur_end = offset[1]
                    else:
                        # 이전 엔티티 저장
                        if cur_type and cur_start is not None and cur_end is not None:
                            entity_text = chunk_text[cur_start:cur_end].strip()
                            if entity_text and len(entity_text) > 1:
                                all_entities[cur_type].append(entity_text)
                        cur_type = None
                        cur_start = None
                        cur_end = None
                
                # 마지막 엔티티 저장
                if cur_type and cur_start is not None and cur_end is not None:
                    entity_text = chunk_text[cur_start:cur_end].strip()
                    if entity_text and len(entity_text) > 1:
                        all_entities[cur_type].append(entity_text)
            
            else:
                # word_ids가 없으면 offset을 사용하여 처리
                cur_type = None
                cur_start = None
                cur_end = None
                
                for j in range(len(pred)):
                    if j >= len(attention_mask[0]) or attention_mask[0, j] == 0:
                        break
                    if j >= len(offsets):
                        break
                    
                    offset = offsets[j]
                    label_id = pred[j]
                    
                    # Special tokens 스킵
                    if offset[0] == offset[1] == 0:
                        if cur_type and cur_start is not None and cur_end is not None:
                            entity_text = chunk_text[cur_start:cur_end].strip()
                            if entity_text and len(entity_text) > 1:
                                all_entities[cur_type].append(entity_text)
                            cur_type = None
                            cur_start = None
                            cur_end = None
                        continue
                    
                    lab = ID2LABEL.get(int(label_id), "O")
                    
                    if lab.startswith("B-"):
                        # 이전 엔티티 저장
                        if cur_type and cur_start is not None and cur_end is not None:
                            entity_text = chunk_text[cur_start:cur_end].strip()
                            if entity_text and len(entity_text) > 1:
                                all_entities[cur_type].append(entity_text)
                        cur_type = lab[2:]
                        cur_start = offset[0]
                        cur_end = offset[1]
                    elif lab.startswith("I-") and cur_type == lab[2:]:
                        # 연속된 엔티티: end 위치만 업데이트
                        if cur_start is not None:
                            cur_end = offset[1]
                    else:
                        # 이전 엔티티 저장
                        if cur_type and cur_start is not None and cur_end is not None:
                            entity_text = chunk_text[cur_start:cur_end].strip()
                            if entity_text and len(entity_text) > 1:
                                all_entities[cur_type].append(entity_text)
                        cur_type = None
                        cur_start = None
                        cur_end = None
                
                # 마지막 엔티티 저장
                if cur_type and cur_start is not None and cur_end is not None:
                    entity_text = chunk_text[cur_start:cur_end].strip()
                    if entity_text and len(entity_text) > 1:
                        all_entities[cur_type].append(entity_text)

    # 중복 제거 및 필터링
    for t in ENTITY_TYPES:
        filtered = []
        for e in all_entities[t]:
            e_clean = e.strip()
            if not e_clean or len(e_clean) < 2:
                continue
            # 숫자만 있는 경우 제거 (일부 타입 제외)
            if t not in ["PHONE", "ID_NUM", "QUANTITY", "MONEY"] and e_clean.isdigit():
                continue
            # 파일 확장자 제거
            if e_clean.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.txt')):
                continue
            filtered.append(e_clean)
        
        dedup = sorted(set(filtered))
        all_entities[t] = dedup if dedup else ["N/A"]
    
    # Apply cleanup and regex-based enrichment
    all_entities = cleanup_entities(all_entities, text)
    return all_entities


def run_prediction():
    print("\n[INFO] 예측 모드 시작...", flush=True)
    device = setup_gpu()
    bert_path = pick_bert_path()
    
    print(f"[INFO] 모델 로드 중...", flush=True)
    model, tokenizer = load_artifacts(CFG, bert_path, device)
    if model is None or tokenizer is None:
        print("\n[ERROR] 학습된 모델이 없습니다. 먼저 --mode train을 실행하세요.", flush=True)
        return
    
    in_dir = CFG.ocr_dir
    out_dir = CFG.out_dir
    
    if not in_dir.exists():
        print(f"\n[ERROR] OCR 폴더를 찾을 수 없습니다: {in_dir}", flush=True)
        return

    # 하위 디렉토리 포함 모든 txt 파일 검색
    txts = sorted(in_dir.glob("**/*.txt"))
    if not txts:
        print(f"\n[WARN] {in_dir} 폴더(하위 포함)에 .txt 파일이 없습니다.", flush=True)
        return

    print(f"\n[INFO] {len(txts)}개의 텍스트 파일을 처리합니다...\n", flush=True)
    
    for idx, p in enumerate(txts, 1):
        print(f"[{idx}/{len(txts)}] {p.name} 처리 중...", flush=True)
        text = p.read_text(encoding="utf-8", errors="ignore")
        ents = predict_text(model, tokenizer, text, device)
        
        # 파일명으로 폴더 생성 (확장자 제거)
        folder_name = p.stem
        
        # conc/final이 없으면 생성 시도
        try:
            doc_out = ensure_dir(out_dir / folder_name)
        except (PermissionError, OSError) as e:
            # Windows 드라이브 권한 문제 시 임시 경로 사용
            fallback_dir = CFG.root_dir / "data/out/ner_results"
            doc_out = ensure_dir(fallback_dir / folder_name)
            print(f"  ⚠️ 경로 변경: {fallback_dir.relative_to(CFG.root_dir)}", flush=True)
        
        (doc_out / "predicted_entities.json").write_text(
            json.dumps(ents, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (doc_out / "original_text.txt").write_text(text, encoding="utf-8")
        
        # 메타데이터 저장
        metadata = {
            "original_filename": p.name,
            "original_path": str(p),
            "source_directory": str(p.parent)
        }
        (doc_out / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  ✅ 저장됨: {folder_name}/", flush=True)
    
    print(f"\n[INFO] ✅ 예측 완료! 결과: {out_dir}", flush=True)


# -----------------------------
# 14) Main
# -----------------------------
def parse_bool_env(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def main():
    import argparse

    env_mode = os.getenv("MODE")
    env_continue = parse_bool_env(os.getenv("CONTINUE_TRAINING"))

    parser = argparse.ArgumentParser(description="Stable NER (mBERT+BiLSTM PyTorch) - synthetic BIO")
    parser.add_argument("--mode", choices=["train", "predict"], default=env_mode or "predict")
    parser.add_argument("--bert-dir", type=str, default=None, help="override local BERT dir (highest priority)")
    args = parser.parse_args()
    
    print("=" * 80)
    print(" NER: mBERT + BiLSTM")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print("=" * 80, flush=True)

    if args.mode == "train":
        train(continue_from_existing=True, bert_dir_override=args.bert_dir, do_warmup=True)
    elif args.mode == "predict":
        run_prediction()


if __name__ == "__main__":
    main()
