#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stable NER Training/Eval/Inference (TF + Transformers)
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

# -----------------------------
# MUST set TF env BEFORE tf import
# -----------------------------
import os

# RTX 5070 (Compute Capability 12.0) 호환성 문제 해결:
# TensorFlow 2.20은 아직 CUDA 13.1 및 CC 12.0을 완전히 지원하지 않음
# 해결책: CPU 모드 사용 (USE_CPU_ONLY=1 환경변수로 제어, 기본값 1)
USE_CPU_ONLY = os.getenv("USE_CPU_ONLY", "1").strip() in ("1", "true", "yes", "y", "on")

if USE_CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU 완전 비활성화
    print("[INFO] CPU 모드로 실행 (RTX 5070 호환성 문제 회피)", flush=True)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")          # reduce TF logs
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")           # For Transformers + Keras 3 compatibility
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")         # Disable oneDNN for stability
# If you want to disable oneDNN (CPU optimizations) for determinism/log noise:
# os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import re
import json
import time
import random
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel


# -----------------------------
# 0) Repro / TF setup
# -----------------------------
def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_gpu_memory_growth() -> None:
    if USE_CPU_ONLY:
        print("[INFO] GPU setup skipped (CPU mode)", flush=True)
        return
    
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[WARN] No GPU detected. Using CPU.", flush=True)
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] GPUs detected: {gpus}", flush=True)
        print("[INFO] GPU memory growth enabled", flush=True)
    except Exception as e:
        print(f"[WARN] GPU setup failed: {e}", flush=True)


def gpu_warmup() -> None:
    """Helps differentiate 'hang' vs PTX JIT compile on new GPUs."""
    if USE_CPU_ONLY:
        print("[INFO] GPU warmup skipped (CPU mode)", flush=True)
        return
    
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        a = tf.random.uniform([1024, 1024])
        b = tf.matmul(a, a)
        _ = b.numpy()
        print("[INFO] GPU warmup OK", flush=True)
    except Exception as e:
        print(f"[WARN] GPU warmup failed: {e}", flush=True)


# -----------------------------
# 1) Config
# -----------------------------
@dataclass
class Config:
    root_dir: Path = Path(__file__).resolve().parent  # 안정적인 기준 경로

    # Data
    num_samples: int = 10000  # 증강 줄이고 epochs로 보완
    min_entities_per_type: int = 100  # 최소 엔티티 수 감소
    max_len: int = 128
    train_ratio: float = 0.85
    seed: int = 42

    # Model
    lr: float = 3e-5
    epochs: int = 20  # 데이터는 적게, 학습은 많이
    batch_size: int = 32
    lstm_units: int = 256
    dropout: float = 0.2

    # Paths (root_dir 기준)
    model_dir: Path = root_dir / "models/ner_bilstm_tf_stable"
    cache_dir: Path = root_dir / "data/cache_ner"
    ocr_dir: Path = root_dir / "data/in/ocr"
    out_dir: Path = root_dir / "data/out/ner_results"

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
    # 최소한 config + (vocab 또는 tokenizer.json)
    has_cfg = (p / "config.json").exists()
    has_tok = (p / "vocab.txt").exists() or (p / "tokenizer.json").exists()
    return has_cfg and has_tok


def pick_bert_path(override: Optional[str] = None) -> str:
    """
    Resolution order:
      1) override arg (CLI --bert-dir)
      2) env BERT_DIR
      3) /app/models/pretrained_bert (docker-style)
      4) <project>/models/pretrained_bert
      5) HF fallback (unless STRICT_LOCAL_BERT=1)
    """
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

    print(f"[INFO] ⚠️ 로컬 BERT 미발견. HuggingFace 모델 사용: {CFG.hf_fallback}", flush=True)
    return CFG.hf_fallback


def warn_if_running_on_mntc() -> None:
    # /mnt/c는 WSL에서 IO가 매우 느릴 수 있음
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
    "POSITION": ["대표","이사","팀장","부장","과장","차장","연구원","주임","매니저","법무담당"],
    "CONTRACT_TYPE": ["위탁","도급","용역","자문","라이선스","양도","사용허락"],
    "CONSENT_TYPE": ["개인정보 수집·이용 동의","제3자 제공 동의","처리위탁 동의","마케팅 수신 동의","국외이전 동의","보유·이용기간 동의"],
    "RIGHT_INFO": ["저작권","사용권","2차적저작물작성권","복제권","배포권","공중송신권","전시권","공연권"],
    "LAW_REFERENCE": ["개인정보 보호법","저작권법","정보통신망법","민법","상법","전자문서 및 전자거래 기본법"],
    "TITLE": ["계약서","합의서","동의서","확약서","요청서","신청서","위임장","보고서"],
    "DESCRIPTION": ["본 계약은 당사자 간의 권리·의무를 규정한다","세부 내용은 별첨을 따른다","분쟁 발생 시 관할은 서울중앙지방법원으로 한다"],
    "TYPE": ["문서","이미지","영상","음원","데이터","소프트웨어","소스코드"],
    "STATUS": ["유효","만료","해지","갱신","진행중","완료","보류"],
    "DEPARTMENT": ["기획팀","개발팀","연구팀","법무팀","인사팀","재무팀","영업팀","운영팀"],
    "LANGUAGE": ["한국어","영어","일본어","중국어","스페인어","프랑스어"],
    "COMPANY": ["(주)테스트","(주)데이터랩","오픈AI코리아","네이버","카카오","삼성전자","LG전자","현대자동차"],
    "PROJECT_NAME": ["저작권 메타데이터 추출","문서 NER 고도화","OCR 파이프라인 개선","계약서 분석 자동화","권리정보 정규화"],
    "PERIOD": ["1년","2년","6개월","3개월","계약기간 내","2025년 말까지","서비스 종료 시까지"],
}


def build_entity_pool(rng: np.random.RandomState) -> Dict[str, List[str]]:
    pool: Dict[str, List[str]] = {}
    print(f"[INFO] 엔티티 풀 구축 중... (각 타입당 최소 {CFG.min_entities_per_type}개)", flush=True)
    for idx, t in enumerate(ENTITY_TYPES, 1):
        n = CFG.min_entities_per_type
        vals = set()
        if t in FIXED_POOLS:
            base = FIXED_POOLS[t]
            # 고정 풀의 경우, 풀 크기만큼만 사용 (중복 허용하지 않음)
            if len(base) >= n:
                # 풀이 충분히 크면 n개만 선택
                vals = set(rng.choice(base, size=n, replace=False))
            else:
                # 풀이 작으면 풀 크기만큼만 사용 (중복 없이)
                vals = set(base)
            print(f"[INFO] {idx}/{len(ENTITY_TYPES)} {t}: {len(vals)}개 (고정 풀)", flush=True)
        elif t == "NAME":
            while len(vals) < n:
                vals.add(rnd_kor_name(rng))
            print(f"[INFO] {idx}/{len(ENTITY_TYPES)} {t}: {len(vals)}개", flush=True)
        elif t == "PHONE":
            while len(vals) < n:
                vals.add(rnd_phone(rng))
            print(f"[INFO] {idx}/{len(ENTITY_TYPES)} {t}: {len(vals)}개", flush=True)
        elif t == "ADDRESS":
            while len(vals) < n:
                vals.add(rnd_address(rng))
            print(f"[INFO] {idx}/{len(ENTITY_TYPES)} {t}: {len(vals)}개", flush=True)
        elif t == "DATE":
            while len(vals) < n:
                vals.add(rnd_date(rng))
            print(f"[INFO] {idx}/{len(ENTITY_TYPES)} {t}: {len(vals)}개", flush=True)
        elif t == "EMAIL":
            while len(vals) < n:
                vals.add(rnd_email(rng))
            print(f"[INFO] {idx}/{len(ENTITY_TYPES)} {t}: {len(vals)}개", flush=True)
        elif t == "URL":
            while len(vals) < n:
                vals.add(rnd_url(rng))
            print(f"[INFO] {idx}/{len(ENTITY_TYPES)} {t}: {len(vals)}개", flush=True)
        elif t == "MONEY":
            while len(vals) < n:
                vals.add(rnd_money(rng))
            print(f"[INFO] {idx}/{len(ENTITY_TYPES)} {t}: {len(vals)}개", flush=True)
        elif t == "QUANTITY":
            while len(vals) < n:
                vals.add(rnd_quantity(rng))
            print(f"[INFO] {idx}/{len(ENTITY_TYPES)} {t}: {len(vals)}개", flush=True)
        elif t == "ID_NUM":
            while len(vals) < n:
                vals.add(f"ID-{rng.randint(100000,999999)}")
            print(f"[INFO] {idx}/{len(ENTITY_TYPES)} {t}: {len(vals)}개", flush=True)
        else:
            while len(vals) < n:
                vals.add(f"{t}_{rng.randint(1, 999999)}")
            print(f"[INFO] {idx}/{len(ENTITY_TYPES)} {t}: {len(vals)}개", flush=True)
        pool[t] = list(vals)
    print(f"[INFO] ✅ 엔티티 풀 구축 완료", flush=True)
    return pool


# -----------------------------
# 4) Template-based sentence synthesis + BIO tagging
# -----------------------------
TEMPLATES = [
    "본 {TITLE}는 {COMPANY}와 {NAME} 간 {CONTRACT_TYPE} 관련 사항을 규정한다.",
    "담당자 {NAME} (연락처: {PHONE}, 이메일: {EMAIL})",
    "주소: {ADDRESS} / 시행일: {DATE} / 상태: {STATUS}",
    "프로젝트명: {PROJECT_NAME} / 수량: {QUANTITY} / 금액: {MONEY}",
    "관련 법령: {LAW_REFERENCE} / 권리정보: {RIGHT_INFO}",
    "{DEPARTMENT}에서 {TYPE} 자료를 {LANGUAGE}로 제공한다. 자세한 내용: {DESCRIPTION}",
    "참조 URL: {URL} / 문서 번호: {ID_NUM}",
    "{COMPANY} {DEPARTMENT} {POSITION} {NAME}는 본 건을 검토한다.",
]

TOKEN_SPLIT_RE = re.compile(r"\s+|(?=[()/:,.-])|(?<=[()/:,.-])")


def simple_word_tokenize(text: str) -> List[str]:
    return [p for p in TOKEN_SPLIT_RE.split(text) if p and not p.isspace()]


def render_template(template: str, pool: Dict[str, List[str]], rng: np.random.RandomState) -> Tuple[List[str], List[str]]:
    used: Dict[str, str] = {}
    text = template
    for t in ENTITY_TYPES:
        key = "{" + t + "}"
        if key in text:
            used[t] = rng.choice(pool[t])
            text = text.replace(key, used[t])

    words = simple_word_tokenize(text)
    labels = ["O"] * len(words)

    for ent_type, ent_text in used.items():
        ent_words = simple_word_tokenize(ent_text)
        if not ent_words:
            continue
        i = 0
        while i <= len(words) - len(ent_words):
            if words[i:i+len(ent_words)] == ent_words:
                labels[i] = f"B-{ent_type}"
                for j in range(1, len(ent_words)):
                    labels[i+j] = f"I-{ent_type}"
                i += len(ent_words)
            else:
                i += 1
    return words, labels


def generate_bio_samples(num_samples: int, seed: int) -> List[Dict]:
    rng = np.random.RandomState(seed)
    pool = build_entity_pool(rng)
    print(f"[INFO] 샘플 생성 시작: {num_samples}개", flush=True)
    samples: List[Dict] = []

    # 초기 샘플 생성 (템플릿 기반)
    initial_count = min(500, num_samples // 2)  # 초기 샘플 수 감소
    print(f"[INFO] 초기 샘플 생성: {initial_count}개", flush=True)
    for i in range(initial_count):
        tmpl = rng.choice(TEMPLATES)
        w, y = render_template(tmpl, pool, rng)
        samples.append({"tokens": w, "labels": y})
        if (i + 1) % 100 == 0:
            print(f"[INFO] 초기 샘플 진행: {i+1}/{initial_count}", flush=True)

    # 나머지 샘플 생성 (간단하게)
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
# 5) Tokenization + label alignment (FAST tokenizer only)
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
        aligned = np.full((max_len,), IGNORE_INDEX, dtype=np.int32)
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

    input_ids = enc["input_ids"].astype(np.int32)
    attention_mask = enc["attention_mask"].astype(np.int32)
    labels = align_labels_with_word_ids(enc, labels_batch, max_len)

    supervised = (labels != IGNORE_INDEX).sum(axis=1)
    keep = supervised > 0

    input_ids = input_ids[keep]
    attention_mask = attention_mask[keep]
    labels = labels[keep]
    return input_ids, attention_mask, labels


# -----------------------------
# 6) Balanced split (label coverage aware)
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

    train_idx = np.array(train_idx, dtype=np.int32)
    val_idx = np.array(val_idx, dtype=np.int32)
    
    return (
        input_ids[train_idx], attention_mask[train_idx], labels[train_idx],
        input_ids[val_idx], attention_mask[val_idx], labels[val_idx],
    )


# -----------------------------
# 7) Model (mBERT + BiLSTM)
# -----------------------------
class BertEncoder(tf.keras.layers.Layer):
    def __init__(self, bert_path: str, **kwargs):
        super().__init__(**kwargs)
        self.bert_path = bert_path
        self.bert = TFAutoModel.from_pretrained(bert_path)

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, training=training)
        return out.last_hidden_state

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"bert_path": self.bert_path})
        return cfg


def build_model(bert_path: str, num_labels: int, max_len: int, lstm_units: int, dropout: float):
    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    x = BertEncoder(bert_path, name="bert_encoder")([input_ids, attention_mask])
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True),
        name="bilstm",
    )(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    logits = tf.keras.layers.Dense(num_labels, name="classifier")(x)
    return tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)


class MaskedSparseCELoss(tf.keras.losses.Loss):
    def __init__(self, ignore_index: int = IGNORE_INDEX, name="masked_sparse_ce"):
        super().__init__(name=name)
        self.ignore_index = ignore_index
        self.ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def call(self, y_true, y_pred):
        mask = tf.not_equal(y_true, self.ignore_index)
        y_true_safe = tf.where(mask, y_true, tf.zeros_like(y_true))
        loss = self.ce(y_true_safe, y_pred)
        loss = tf.where(mask, loss, tf.zeros_like(loss))
        denom = tf.reduce_sum(tf.cast(mask, tf.float32)) + 1e-8
        return tf.reduce_sum(loss) / denom


# -----------------------------
# 8) Metrics (entity-level F1 simple)
# -----------------------------
def decode_bio(label_ids: np.ndarray, attention_mask: np.ndarray) -> List[str]:
    out = []
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
    tp = fp = fn = 0
    for yt, yp in zip(y_true_seqs, y_pred_seqs):
        tset = set(bio_to_entities(yt))
        pset = set(bio_to_entities(yp))
        tp += len(tset & pset)
        fp += len(pset - tset)
        fn += len(tset - pset)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1}


# -----------------------------
# 9) Save / Load
# -----------------------------
def save_artifacts(model: tf.keras.Model, tokenizer, cfg: Config):
    ensure_dir(cfg.model_dir)
    model.save(cfg.model_dir / "model.keras")
    tokenizer.save_pretrained(cfg.model_dir / "tokenizer")
    (cfg.model_dir / "labels.json").write_text(
        json.dumps(
            {"BIO_LABELS": BIO_LABELS, "ENTITY_TYPES": ENTITY_TYPES, "max_len": cfg.max_len},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] ✅ Saved model to: {cfg.model_dir}", flush=True)


def load_artifacts(cfg: Config):
    model_path = cfg.model_dir / "model.keras"
    tok_dir = cfg.model_dir / "tokenizer"
    labels_path = cfg.model_dir / "labels.json"
    if not model_path.exists() or not tok_dir.exists() or not labels_path.exists():
        return None, None

    meta = json.loads(labels_path.read_text(encoding="utf-8"))
    if meta.get("BIO_LABELS") != BIO_LABELS or meta.get("max_len") != cfg.max_len:
        print("[WARN] Saved label schema/max_len differs. Refusing to continue-training.", flush=True)
        return None, None

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"BertEncoder": BertEncoder, "MaskedSparseCELoss": MaskedSparseCELoss},
    )
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    print(f"[INFO] ✅ Loaded existing model from: {cfg.model_dir}", flush=True)
    return model, tokenizer


# -----------------------------
# 10) Cache
# -----------------------------
def cache_key(cfg: Config, bert_path: str) -> str:
    s = f"{cfg.num_samples}|{cfg.min_entities_per_type}|{cfg.max_len}|{cfg.seed}|{bert_path}|v3"
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
# 11) Train / Eval / Inference
# -----------------------------
def make_tf_dataset(input_ids, attention_mask, labels, batch_size: int, shuffle: bool, seed: int):
    ds = tf.data.Dataset.from_tensor_slices(((input_ids, attention_mask), labels))
    # CUDA 오류 방지: shuffle을 완전히 제거
    # 이미 split_train_val에서 seed 기반 shuffle을 했으므로, dataset 레벨 shuffle 불필요
    # GPU에서 shuffle이 CUDA_ERROR_INVALID_HANDLE 오류를 일으키므로 제거
    # if shuffle:
    #     ds = ds.shuffle(buffer_size=min(20000, len(input_ids)), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def train(continue_from_existing: bool, bert_dir_override: Optional[str] = None, do_warmup: bool = True):
    set_global_seed(CFG.seed)
    warn_if_running_on_mntc()
    setup_gpu_memory_growth()
    if do_warmup:
        gpu_warmup()

    bert_path = pick_bert_path(override=bert_dir_override)
    key = cache_key(CFG, bert_path)

    model = None
    tokenizer = None

    if continue_from_existing:
        model, tokenizer = load_artifacts(CFG)

    if tokenizer is None:
        tokenizer = load_tokenizer(bert_path)

    cached = load_cached_dataset(CFG, key)
    if cached is None:
        print(f"[INFO] Generating synthetic BIO samples: {CFG.num_samples}", flush=True)
        samples = generate_bio_samples(CFG.num_samples, seed=CFG.seed)
        print("[INFO] Tokenizing & aligning labels...", flush=True)
        input_ids, attention_mask, labels = tokenize_bio_samples(samples, tokenizer, CFG.max_len)
        print(f"[INFO] Valid samples after alignment: {labels.shape[0]}", flush=True)
        save_cached_dataset(CFG, key, input_ids, attention_mask, labels)
    else:
        input_ids, attention_mask, labels = cached
        print(f"[INFO] Loaded cached dataset. n={labels.shape[0]}", flush=True)

    tr_ids, tr_mask, tr_y, va_ids, va_mask, va_y = split_train_val(
        input_ids, attention_mask, labels, CFG.train_ratio, CFG.seed
    )
    print(f"[INFO] Train={len(tr_ids)}, Val={len(va_ids)}", flush=True)

    train_ds = make_tf_dataset(tr_ids, tr_mask, tr_y, CFG.batch_size, shuffle=True, seed=CFG.seed)
    val_ds = make_tf_dataset(va_ids, va_mask, va_y, CFG.batch_size, shuffle=False, seed=CFG.seed)

    if model is None:
        model = build_model(
            bert_path, num_labels=len(BIO_LABELS), max_len=CFG.max_len,
            lstm_units=CFG.lstm_units, dropout=CFG.dropout
        )

    loss_fn = MaskedSparseCELoss()
    optimizer = tf.keras.optimizers.Adam(learning_rate=CFG.lr)
    model.compile(optimizer=optimizer, loss=loss_fn)

    print(f"[INFO] Training epochs={CFG.epochs}, batch={CFG.batch_size}, max_len={CFG.max_len}", flush=True)
    model.fit(train_ds, validation_data=val_ds, epochs=CFG.epochs)

    y_true_seqs, y_pred_seqs = [], []
    for (x, y) in val_ds:
        logits = model(x, training=False)
        pred = tf.argmax(logits, axis=-1).numpy()
        y_np = y.numpy()
        mask_np = x[1].numpy()
        for i in range(pred.shape[0]):
            yt = decode_bio(y_np[i], mask_np[i])
            yp = decode_bio(pred[i], mask_np[i])
            y_true_seqs.append(yt)
            y_pred_seqs.append(yp)

    m = entity_f1(y_true_seqs, y_pred_seqs)
    print(f"[INFO] Val entity-level: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}", flush=True)

    save_artifacts(model, tokenizer, CFG)


def predict_text(model, tokenizer, text: str) -> Dict[str, List[str]]:
    tokens = simple_word_tokenize(text)
    all_entities: Dict[str, List[str]] = {t: [] for t in ENTITY_TYPES}
    step = CFG.max_len - 16

    for start in range(0, len(tokens), step):
        chunk = tokens[start:start + step]
        enc = tokenizer(
            [chunk],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=CFG.max_len,
            return_tensors="np",
        )
        ids = enc["input_ids"].astype(np.int32)
        mask = enc["attention_mask"].astype(np.int32)
        logits = model([ids, mask], training=False).numpy()
        pred = logits.argmax(axis=-1)[0]

        word_ids = enc.word_ids(batch_index=0)
        prev = None
        cur_type = None
        cur_words: List[str] = []

        for j, wid in enumerate(word_ids):
            if wid is None or mask[0, j] == 0:
                continue
            if wid == prev:
                continue
            prev = wid
            if wid >= len(chunk):
                continue
        
            lab = ID2LABEL.get(int(pred[j]), "O")
            w = chunk[wid]

            if lab.startswith("B-"):
                if cur_type and cur_words:
                    all_entities[cur_type].append(" ".join(cur_words).strip())
                cur_type = lab[2:]
                cur_words = [w]
            elif lab.startswith("I-") and cur_type == lab[2:]:
                cur_words.append(w)
            else:
                if cur_type and cur_words:
                    all_entities[cur_type].append(" ".join(cur_words).strip())
                cur_type = None
                cur_words = []

        if cur_type and cur_words:
            all_entities[cur_type].append(" ".join(cur_words).strip())

    for t in ENTITY_TYPES:
        dedup = sorted(set([e for e in all_entities[t] if e and len(e) > 1]))
        all_entities[t] = dedup if dedup else ["N/A"]
    return all_entities


def run_evaluation():
    model, tokenizer = load_artifacts(CFG)
    if model is None or tokenizer is None:
        print("[ERROR] No saved model. Train first.", flush=True)
        return

    in_dir = CFG.ocr_dir
    out_dir = ensure_dir(CFG.out_dir)
    if not in_dir.exists():
        print(f"[WARN] OCR dir not found: {in_dir} (skip)", flush=True)
        return

    txts = sorted(in_dir.glob("*.txt"))
    if not txts:
        print(f"[WARN] No .txt files in: {in_dir}", flush=True)
        return

    for p in txts:
        text = p.read_text(encoding="utf-8", errors="ignore")
        ents = predict_text(model, tokenizer, text)
        doc_out = ensure_dir(out_dir / p.stem)
        (doc_out / "predicted_entities.json").write_text(
            json.dumps(ents, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (doc_out / "original_text.txt").write_text(text, encoding="utf-8")
        print(f"[INFO] Saved: {doc_out}", flush=True)


# -----------------------------
# 12) Main (mode + env)
# -----------------------------
def parse_bool_env(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def main():
    import argparse

    env_mode = os.getenv("MODE")
    env_continue = parse_bool_env(os.getenv("CONTINUE_TRAINING"))

    parser = argparse.ArgumentParser(description="Stable NER (mBERT+BiLSTM) - synthetic BIO")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default=env_mode or "both")
    parser.add_argument("--continue-training", action="store_true", default=env_continue)
    parser.add_argument("--no-continue-training", action="store_true", help="force fresh training (ignore saved model)")
    parser.add_argument("--bert-dir", type=str, default=None, help="override local BERT dir (highest priority)")
    parser.add_argument("--no-warmup", action="store_true", help="disable GPU warmup step")
    args = parser.parse_args()
    
    print("=" * 80)
    print(" Stable NER: mBERT + BiLSTM (BIO)")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    cont = args.continue_training and (not args.no_continue_training)
    print(f"Continue training: {cont}")
    print(f"Root dir: {CFG.root_dir}")
    print(f"Model dir: {CFG.model_dir}")
    print("=" * 80, flush=True)

    if args.mode in ("train", "both"):
        train(continue_from_existing=cont, bert_dir_override=args.bert_dir, do_warmup=(not args.no_warmup))
    if args.mode in ("eval", "both"):
        run_evaluation()


if __name__ == "__main__":
    main()
