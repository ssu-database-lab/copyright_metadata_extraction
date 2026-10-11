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
import threading
import warnings
from contextlib import contextmanager
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
            logger.debug(f"NER 기본 모델 설정: {default_model}")
            return default_model
    except Exception as e:
        logger.warning(f"NER model_config.json 을 읽지 못해 기본값을 씁니다: {type(e).__name__}: {e}")
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
        logger.debug(f"NER 모델 폴더 확인: {models_ner_dir}")
        return models_ner_dir
    
    # 2. model_downloaded/{model_name} 확인 및 복사
    model_downloaded_dir = api_dir / "model_downloaded" / model_name_safe
    if model_downloaded_dir.exists() and (model_downloaded_dir / "config.json").exists():
        logger.info(f"NER 모델 복사 — {model_downloaded_dir} → {models_ner_dir}")
        
        import shutil
        if models_ner_dir.exists():
            shutil.rmtree(models_ner_dir)
        shutil.copytree(model_downloaded_dir, models_ner_dir)
        
        logger.info(f"NER 모델 복사 완료: {models_ner_dir}")
        return models_ner_dir
    
    # 3. Hugging Face에서 다운로드
    logger.warning(f"NER 모델이 로컬에 없습니다: {model_name} — Hugging Face 에서 내려받습니다 ({model_downloaded_dir})")
    
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        model_downloaded_dir.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        tokenizer.save_pretrained(str(model_downloaded_dir))
        model.save_pretrained(str(model_downloaded_dir))
        logger.info(f"NER 모델 내려받기 완료: {model_downloaded_dir}")
        
        # models/ner/로 복사
        import shutil
        if models_ner_dir.exists():
            shutil.rmtree(models_ner_dir)
        shutil.copytree(model_downloaded_dir, models_ner_dir)
        logger.info(f"NER 모델 복사 완료: {models_ner_dir}")
        
        return models_ner_dir
        
    except Exception as e:
        logger.error(f"NER 모델 내려받기 실패 ({model_name}): {type(e).__name__}: {e}")
        return models_ner_dir


# ========== Helper Functions - Model Loading ==========

def load_model_and_tokenizer(model_path: Path, verbose: bool = True):
    """모델과 토크나이저 로드 (BERT-CRF 커스텀 모델 지원)

    BERT-CRF 뼈대는 모델 폴더의 config.json 으로 만든다 (2026-10-11 수정). 예전에는 config.json 의
    _name_or_path 를 기반 모델 이름으로 썼는데, KLUE·XLM-R 폴더의 config.json 에는 그 값이 없어 mBERT 뼈대가
    만들어졌고 model.pt 와 구조가 맞지 않아 적재가 늘 실패했다 — NER 이 조용히 정규식만 돌았다.
    config.json 이 없을 때만 예전처럼 기본 기반 모델(mBERT)을 쓴다.
    """
    logger.debug(f"NER 모델 파일 읽기: {model_path}")
    
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
            model_name = str(model_path)
            name_source = f"모델 폴더 config.json · model_type {config.get('model_type', '?')}"
        else:
            model_name = 'google-bert/bert-base-multilingual-cased'
            name_source = "기본값 — config.json 없음"
        
        num_labels = len(id2label)
        logger.info(f"NER BERT-CRF 뼈대 — 기반 모델 {model_name} ({name_source}) · 라벨 {num_labels}개 · 가중치 {model_pt_path.name}")
        model = BertCrfForNER(model_name=model_name, num_labels=num_labels)
        
        # 훈련된 가중치 로드
        model.load_state_dict(torch.load(model_pt_path, map_location='cpu'))
        logger.debug(f"NER BERT-CRF 가중치 적재 완료: {num_labels}개 라벨")
    else:
        # 표준 HuggingFace 모델
        model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        logger.debug("NER HuggingFace 토큰 분류 모델 적재 완료")
    
    # GPU 사용 가능하면 모델을 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    logger.debug(f"NER 모델 적재 완료 ({device})")
    return tokenizer, model, id2label, device


# ========== Model Cache (process-wide, thread-safe) ==========
#
# 예전에는 extract_entities_from_text 가 호출될 때마다 load_model_and_tokenizer 를 불러
# KLUE-RoBERTa-Large(+CRF, RAM ~1.4GB) 를 새로 적재했다. 배치 실행기(--workers N)와
# 파이프라인의 LLM ∥ NER 동시 실행이 겹치면 세트마다 사본이 하나씩 생겨 RAM 이 워커 수를
# 묶었고, 호출마다 적재 시간(/mnt/c 에서 15~30초)을 다시 치렀다.
#
# 지금은 (해석된 모델 경로, 디바이스) 를 키로 프로세스당 한 번만 적재해서 모든 스레드가
# 같은 객체를 쓴다.
#   - 적재: 프로세스 전역 적재 잠금 + 이중 확인. 동시에 들어온 첫 호출들 중 하나만 적재하고
#     나머지는 기다렸다 같은 객체를 받는다. 다른 모델의 적재도 서로 겹치지 않게 한 줄로 세운다 —
#     transformers 4.57 의 from_pretrained 는 프로세스 전역 meta-device 문맥에서 모델을 만들어서,
#     두 스레드가 동시에 적재하면 둘 다 "Cannot copy out of meta tensor" 로 실패한다 (실측: 3스레드
#     9/9 실패). 예전 경로에서는 그 실패가 아래 except 에 삼켜져 NER 이 조용히 정규식만 돌렸다.
#   - 적재 실패도 기억한다(NER_MODEL_RETRY_SEC 동안). 예전 경로는 실패를 삼키고 정규식만
#     돌렸는데 호출마다 같은 실패를 다시 겪었다 — 결과는 같고 시간만 버렸다.
#   - 토크나이저: HF fast 토크나이저는 호출 때 truncation/padding 설정을 Rust 객체에 써 넣는다.
#     다른 스레드가 encode 하는 동안 그걸 바꾸면 "Already borrowed" 가 나고, 그 문장은 아래
#     except 에서 조용히 빠진다. 그래서 토큰화 한 줄만 잠금으로 감싼다 (문장당 ~1ms).
#   - forward: eval() + no_grad() 의 forward 는 상태를 바꾸지 않아 동시에 불러도 안전하다.
#     다만 CPU 에서는 호출마다 torch 가 코어 수만큼 스레드를 쓰므로 N 개를 동시에 돌리면
#     N×코어 스레드로 과다 구독된다. NER_MAX_CONCURRENCY 세마포어로 동시 forward 수를 묶는다.
#     torch.set_num_threads 는 프로세스 전역이라 여기서 건드리지 않는다.
#
# 환경 변수 (호출 시점에 읽는다):
#   NER_MODEL_CACHE=0        캐시를 끄고 예전처럼 호출마다 적재한다.
#   NER_MAX_CONCURRENCY=K    동시에 도는 forward 수 상한. 0 = 제한 없음. 기본은
#                            _default_max_concurrency() 참조.
#   NER_MODEL_RETRY_SEC=S    적재 실패를 기억하는 시간(초, 기본 300). 0 = 매번 다시 시도.
# 테스트는 clear_ner_model_cache() 로 비운다.

_FALSEY = ('0', 'false', 'no', 'off')


class NERModelLoadError(RuntimeError):
    """캐시에 기억된 적재 실패를 다시 알릴 때 쓴다 (원래 예외는 __cause__).

    brief: 기억된 오류의 한 줄 요약 · failed_ago: 실패한 지 몇 초 · retry_in: 몇 초 뒤 다시 적재를 시도하는지.
    """

    def __init__(self, message: str, brief: str = "", failed_ago: float = 0.0, retry_in: float = 0.0):
        super().__init__(message)
        self.brief = brief
        self.failed_ago = failed_ago
        self.retry_in = retry_in


_DEFAULT_BASE_MODEL = 'google-bert/bert-base-multilingual-cased'


def _error_brief(error: Any, limit: int = 300) -> str:
    """예외(또는 그 문자열) → 한 줄 요약: '클래스: 첫 줄' + (가중치 모양 불일치면) 건수와 첫 예.

    load_state_dict 의 RuntimeError 는 첫 줄이 'Error(s) in loading state_dict for BertCrfForNER:' 뿐이라
    무엇이 어긋났는지 보이지 않는다 — 첫 'size mismatch' 줄을 붙인다.
    """
    if isinstance(error, BaseException):
        text = f"{type(error).__name__}: {error}"
    else:
        text = str(error or "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    brief = lines[0]
    mism = [ln for ln in lines if ln.startswith("size mismatch")]
    if mism:
        m = re.search(r"size mismatch for (\S+?): copying a param with shape torch\.Size\(\[([^\]]*)\]\) from checkpoint, "
                      r"the shape in current model is torch\.Size\(\[([^\]]*)\]\)", mism[0])
        example = (f"{m.group(1)} 가중치 [{m.group(2)}] ≠ 뼈대 [{m.group(3)}]" if m else mism[0][:160])
        brief += f" · 크기 불일치 {len(mism)}건 (예: {example})"
    else:
        missing = [ln for ln in lines if ln.startswith(("Missing key", "Unexpected key"))]
        if missing:
            brief += f" · {missing[0][:180]}"
    return brief[:limit]


def _explain_load_error(error: Any, model_path: Optional[Path]) -> str:
    """적재 실패를 한 줄 한국어로 풀어 쓴다 (원인 추정)."""
    text = str(error or "")
    name_or_path = None
    has_cfg = False
    try:
        cfg_path = Path(model_path) / "config.json" if model_path else None
        if cfg_path is not None and cfg_path.exists():
            has_cfg = True
            name_or_path = json.loads(cfg_path.read_text(encoding="utf-8")).get('_name_or_path')
    except Exception:
        pass
    if "state_dict" in text or "size mismatch" in text or "Missing key" in text or "Unexpected key" in text:
        if has_cfg:                              # 뼈대는 모델 폴더 config.json 으로 만든다 (2026-10-11~)
            return ("저장된 가중치(model.pt)가 모델 폴더 config.json 의 구조와 맞지 않아 모델을 올리지 못했습니다 "
                    "— 다른 모델로 학습한 model.pt 인지 확인하세요 · 정규표현식 추출만 씁니다")
        return f"저장된 가중치(model.pt)가 기본 기반 모델({_DEFAULT_BASE_MODEL})의 구조와 맞지 않습니다 — 정규표현식 추출만 씁니다"
    if "meta tensor" in text:
        return "모델을 여러 스레드가 동시에 적재하다 meta-device 경쟁으로 실패했습니다 — 정규표현식 추출만 씁니다"
    if isinstance(error, MemoryError) or "out of memory" in text.lower():
        return "메모리가 부족해 모델을 올리지 못했습니다 — 정규표현식 추출만 씁니다"
    if isinstance(error, (FileNotFoundError, PermissionError)) or "No such file" in text:
        return "모델 파일을 읽을 수 없습니다 — 정규표현식 추출만 씁니다"
    if "Can't load" in text or "couldn't connect" in text.lower() or "offline" in text.lower():
        return "기반 모델(Hugging Face)을 내려받거나 찾지 못했습니다 — 정규표현식 추출만 씁니다"
    return "모델을 올리지 못했습니다 — 정규표현식 추출만 씁니다"


class _CacheEntry:
    __slots__ = ('bundle', 'error', 'failed_at', 'load_seconds', 'loaded_at', 'brief')

    def __init__(self, bundle=None, error: Optional[BaseException] = None,
                 load_seconds: float = 0.0, brief: str = ""):
        self.bundle = bundle                  # (tokenizer, model, id2label, device) 또는 None
        self.error = error
        self.failed_at = time.monotonic() if error is not None else None
        self.loaded_at = time.time()
        self.load_seconds = load_seconds
        self.brief = brief                    # 실패 한 줄 요약 (_error_brief)


_MODEL_CACHE: Dict[Tuple[str, str], _CacheEntry] = {}
_MODEL_CACHE_LOCK = threading.Lock()          # _MODEL_CACHE 와 _CACHE_GENERATION 을 지킨다 (짧게만 잡는다)
_MODEL_LOAD_LOCK = threading.Lock()           # 적재는 한 번에 하나 (위 meta-device 경쟁)
_CACHE_GENERATION = 0                         # clear 후에 끝난 적재가 캐시를 되살리지 않도록

_TOKENIZER_LOCK = threading.Lock()

_FORWARD_SEM_LOCK = threading.Lock()
# (세마포어를 만든 설정값, 세마포어) — 한 번에 읽히도록 튜플 하나로 둔다. 설정값 0 = 제한 없음(None).
_FORWARD_SEM_STATE: Tuple[Optional[int], Optional[threading.BoundedSemaphore]] = (None, None)


def _model_cache_enabled() -> bool:
    return os.getenv('NER_MODEL_CACHE', '1').strip().lower() not in _FALSEY


def _model_retry_seconds() -> float:
    try:
        return max(0.0, float(os.getenv('NER_MODEL_RETRY_SEC', '300')))
    except ValueError:
        return 300.0


def _default_device() -> torch.device:
    """load_model_and_tokenizer 와 같은 규칙 — 캐시 키를 적재 전에 정하려고 따로 둔다."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cache_key(model_path: Path) -> Tuple[str, str]:
    return (str(Path(model_path).resolve()), str(_default_device()))


def _load_logged(model_path: Path, model_name: Optional[str], verbose: bool, cached: bool):
    """load_model_and_tokenizer 를 부르며 시작(INFO)·완료(INFO)·실패(ERROR)를 남긴다. 실패는 그대로 다시 던진다.

    실패 로그: 예외 클래스와 첫 줄(+ 가중치 모양 불일치 예), 모델 경로, 한 줄 한국어 설명.
    """
    device = _default_device()
    label = model_name or Path(model_path).name
    logger.info(f"NER 모델 적재 시작 — 모델 {label} · 경로 {model_path} · 장치 {device}"
                + ("" if cached else " (캐시 끔 — 호출마다 적재)"))
    started = time.time()
    try:
        # 모듈 전역 이름으로 부른다 — 테스트가 가짜 적재기로 바꿔 끼울 수 있게.
        bundle = load_model_and_tokenizer(Path(model_path), verbose=verbose)
    except Exception as exc:
        brief = _error_brief(exc)
        retry = (f" · {_model_retry_seconds():.0f}초 동안 다시 시도하지 않음" if cached and _model_retry_seconds() > 0
                 else "")
        logger.error(f"NER 모델 적재 실패 — {brief} · 경로 {model_path} · {time.time() - started:.1f}초{retry} · "
                     f"{_explain_load_error(exc, Path(model_path))}")
        raise
    elapsed = time.time() - started
    try:
        n_labels = len(bundle[2])
    except Exception:
        n_labels = None
    logger.info(f"NER 모델 적재 완료 — 모델 {label} · {elapsed:.1f}초 · 라벨 {n_labels if n_labels is not None else '?'}개 · "
                f"장치 {bundle[3] if isinstance(bundle, tuple) and len(bundle) > 3 else device}")
    return bundle, elapsed


def get_cached_model_and_tokenizer(model_path: Path, verbose: bool = False, *, model_name: Optional[str] = None):
    """load_model_and_tokenizer 와 같은 4-튜플을 돌려주되, 프로세스당 한 번만 적재한다.

    같은 (경로, 디바이스) 로 들어온 동시 첫 호출들은 하나만 적재하고 나머지는 그 결과를 받는다.
    적재가 실패하면 NER_MODEL_RETRY_SEC 동안은 다시 적재하지 않고 NERModelLoadError 를 낸다.
    model_name 은 로그에 적을 이름일 뿐이다 (캐시 키는 경로·디바이스).
    """
    key = _cache_key(model_path)

    entry = _MODEL_CACHE.get(key)              # 빠른 경로: 잠금 없이 (dict 조회는 원자적)
    if entry is not None and entry.bundle is not None:
        logger.debug(f"NER 모델 캐시 사용 ({key[0]}, {key[1]})")
        return entry.bundle

    with _MODEL_LOAD_LOCK:
        generation = _CACHE_GENERATION
        entry = _MODEL_CACHE.get(key)          # 이중 확인: 기다리는 동안 다른 스레드가 적재했을 수 있다
        if entry is not None:
            if entry.bundle is not None:
                logger.debug(f"NER 모델 캐시 사용 ({key[0]}, {key[1]}) — 적재를 기다린 호출")
                return entry.bundle
            ago = time.monotonic() - entry.failed_at
            window = _model_retry_seconds()
            if ago < window:
                raise NERModelLoadError(
                    f"NER 모델 적재가 앞서 실패함 ({key[0]}, {key[1]}): {entry.error}",
                    brief=entry.brief or _error_brief(entry.error), failed_ago=ago, retry_in=window - ago,
                ) from entry.error

        try:
            bundle, elapsed = _load_logged(Path(model_path), model_name, verbose, cached=True)
        except Exception as exc:
            # 원래 예외는 기억하지 않는다 — __traceback__ 이 적재 중이던 프레임(반쯤 만든 모델,
            # torch.load 한 state_dict 등 GB 단위)을 붙잡아 재시도 창 내내 RAM 을 묶는다.
            summary = RuntimeError(f"{type(exc).__name__}: {str(exc)[:2000]}")
            with _MODEL_CACHE_LOCK:
                if generation == _CACHE_GENERATION:
                    _MODEL_CACHE[key] = _CacheEntry(error=summary, brief=_error_brief(exc))
            raise

        with _MODEL_CACHE_LOCK:
            if generation == _CACHE_GENERATION:
                _MODEL_CACHE[key] = _CacheEntry(bundle=bundle, load_seconds=elapsed)
        return bundle


def clear_ner_model_cache() -> int:
    """캐시(성공·실패 기록 모두)와 forward 세마포어를 비운다. 비운 항목 수를 돌려준다.

    이미 모델을 들고 있는 호출은 끝날 때까지 그 객체를 계속 쓴다. 진행 중이던 적재는
    끝나도 캐시에 넣지 않는다.
    """
    global _CACHE_GENERATION, _FORWARD_SEM_STATE
    with _MODEL_CACHE_LOCK:
        n = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()
        _CACHE_GENERATION += 1
    with _FORWARD_SEM_LOCK:
        _FORWARD_SEM_STATE = (None, None)
    import gc
    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    return n


def ner_model_cache_info() -> List[Dict[str, Any]]:
    """진단용: 캐시에 든 항목 (경로, 디바이스, 상태, 적재 시간)."""
    with _MODEL_CACHE_LOCK:
        items = list(_MODEL_CACHE.items())
    return [{
        'model_path': path,
        'device': device,
        'status': 'loaded' if e.bundle is not None else 'failed',
        'load_seconds': round(e.load_seconds, 2),
        'loaded_at': datetime.fromtimestamp(e.loaded_at).isoformat(timespec='seconds'),
        'error': None if e.error is None else f"{type(e.error).__name__}: {str(e.error)[:200]}",
    } for (path, device), e in items]


def _default_max_concurrency() -> int:
    """NER_MAX_CONCURRENCY 기본값 — CUDA 면 1, CPU 면 쓸 수 있는 CPU 4개당 1개 (1~4).

    실측 (KLUE-RoBERTa-Large+CRF, 계약서 본문 ~3.5천 자 = 8 청크, 캐시 적재 후, 8 스레드):
      - CUDA (RTX 4070S): 동시 1 → 0.6초/건, 2 → 1.2초, 제한 없음 → 2.3초. forward 가 커널 발사·GIL 에
        묶여 있어 겹치면 오히려 느려진다.
      - CPU 28 논리코어 (i7-14700F, WSL2, torch 14 스레드): 1 → 7.9초/건, 2 → 5.2, 4 → 4.1
        (순차 1 스레드 4.2). 1 개 forward 가 코어를 다 쓰지 못한다 — torch 스레드를 줄이면 더 좋다
        (OMP_NUM_THREADS=4 + 동시 4 → 2.5초/건; 배포 설정 몫이라 여기서 정하지 않는다).
      - CPU 2 vCPU (Oracle 서버 모사, taskset 0,1): 1·2·제한 없음 모두 6.3~6.9초/건 — CPU 가 이미 포화.
    """
    if torch.cuda.is_available():
        return 1
    try:
        n_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n_cpus = os.cpu_count() or 1
    return max(1, min(4, n_cpus // 4))


def _max_concurrency() -> int:
    raw = os.getenv('NER_MAX_CONCURRENCY')
    if raw is None or not raw.strip():
        return _default_max_concurrency()
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        logger.warning(f"NER_MAX_CONCURRENCY={raw!r} 를 읽을 수 없음 — 기본값 사용")
        return _default_max_concurrency()
    return value


def _forward_semaphore() -> Optional[threading.BoundedSemaphore]:
    """설정값이 바뀌면 새로 만든다 (이미 들어간 호출은 예전 세마포어로 나간다)."""
    global _FORWARD_SEM_STATE
    size = _max_concurrency()
    state = _FORWARD_SEM_STATE
    if state[0] == size:
        return state[1]
    with _FORWARD_SEM_LOCK:
        if _FORWARD_SEM_STATE[0] != size:
            _FORWARD_SEM_STATE = (size, threading.BoundedSemaphore(size) if size > 0 else None)
        return _FORWARD_SEM_STATE[1]


@contextmanager
def _ner_forward_slot():
    """동시에 도는 모델 forward 수를 NER_MAX_CONCURRENCY 로 묶는다."""
    sem = _forward_semaphore()
    if sem is None:
        yield
        return
    with sem:
        yield


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
            # 토큰화 — 공유 토크나이저의 truncation/padding 설정 변경이 다른 스레드의 encode 와
            # 겹치지 않도록 잠근다 (모델 캐시 절 참조).
            with _TOKENIZER_LOCK:
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
                with _ner_forward_slot():
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

        if process.stdout is None:
            raise RuntimeError("훈련 프로세스에서 표준 출력 스트림을 확보하지 못했습니다.")
        
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

def extract_entities_with_status(
    text: str,
    model_name: Optional[str] = None,
    *,
    model_path: Optional[Path] = None,
    debug: bool = False
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    """텍스트에서 엔티티 추출 + 어떻게 추출했는지(상태).

    상태: {"mode": "model" | "regex_fallback", "model_loaded": bool, "load_error": str|None, "model": 이름,
           "model_path": 경로, "model_entities": 모델이 낸 수, "regex_entities": 정규식이 낸 수, "entity_count": 최종 수}
    모델을 올리지 못하면 정규표현식만 쓴다 — 그 사실을 WARNING 으로 **호출마다** 남긴다 (앞선 적재 실패를 캐시가
    기억하고 있어도). 적재 실패 자체는 적재기가 ERROR 로 남긴다.
    """
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME
        logger.debug(f"NER 모델 이름이 지정되지 않아 기본 모델 사용: {model_name}")

    if model_path is not None:
        resolved_model_path = Path(model_path)
    else:
        resolved_model_path = get_model_path(model_name)

    logger.debug(f"NER 엔티티 추출 시작 — 본문 {len(text)}자 · 모델 {model_name} · 경로 {resolved_model_path}")

    status: Dict[str, Any] = {
        "mode": "regex_fallback", "model_loaded": False, "load_error": None, "model": model_name,
        "model_path": str(resolved_model_path), "model_entities": 0, "regex_entities": 0, "entity_count": 0,
    }
    remembered: Optional[NERModelLoadError] = None
    all_entities: Set[Tuple[str, str]] = set()
    bio_entities: Set[Tuple[str, str]] = set()

    model_exists = resolved_model_path.exists() and (resolved_model_path / "config.json").exists()

    if not model_exists:
        logger.warning(f"NER 모델 폴더가 없습니다: {resolved_model_path} — Hugging Face 에서 {model_name} 을 내려받습니다")
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)

            resolved_model_path.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(resolved_model_path)
            model.save_pretrained(resolved_model_path)

            logger.info(f"NER 모델 내려받기 완료: {resolved_model_path}")
            model_exists = True
        except Exception as exc:
            status["load_error"] = f"모델 파일 없음 · 내려받기 실패: {_error_brief(exc)}"
            logger.error(f"NER 모델 내려받기 실패 — {_error_brief(exc)} · 경로 {resolved_model_path} · "
                         f"모델 파일이 없어 정규표현식 추출만 씁니다 (ner_train() 으로 먼저 훈련하세요)")

    try:
        model_exists = resolved_model_path.exists() and (resolved_model_path / "config.json").exists()

        if model_exists:
            if _model_cache_enabled():
                tokenizer, model, id2label, device = get_cached_model_and_tokenizer(
                    resolved_model_path, verbose=debug, model_name=model_name)
            else:   # NER_MODEL_CACHE=0 — 예전처럼 호출마다 적재
                (tokenizer, model, id2label, device), _ = _load_logged(
                    resolved_model_path, model_name, debug, cached=False)
            status["model_loaded"] = True

            bio_entities = extract_entities_by_bio_tagging(text, tokenizer, model, id2label, device)
            all_entities.update(bio_entities)
            status["mode"] = "model"
            status["model_entities"] = len(bio_entities)
            logger.debug(f"NER B-I-O 예측 결과: {len(bio_entities)}개 엔티티 (라벨 {len(id2label)}개)")

        elif status["load_error"] is None:
            status["load_error"] = f"모델 파일 없음: {resolved_model_path}"

    except NERModelLoadError as exc:           # 앞선 적재 실패를 캐시가 기억하고 있다 — 아래 WARNING 에 그 사유를 적는다
        remembered = exc
        status["load_error"] = exc.brief or _error_brief(exc.__cause__ or exc)
    except Exception as exc:
        if status["model_loaded"]:              # 모델은 올라왔는데 예측 중에 실패
            status["load_error"] = f"예측 오류: {_error_brief(exc)}"
            logger.error(f"NER 모델 예측 오류 — {_error_brief(exc)} · 정규표현식 결과만 씁니다", exc_info=debug)
        else:                                   # 적재 실패 — 적재기(_load_logged)가 이미 ERROR 로 남겼다
            status["load_error"] = _error_brief(exc)

    regex_entities = extract_entities_by_regex(text)
    all_entities.update(regex_entities)
    status["regex_entities"] = len(regex_entities)

    final_entities: List[Tuple[str, str]] = []
    seen_entities: Set[str] = set()

    model_set = set(bio_entities)
    for entity, label in sorted(all_entities, key=lambda x: (x not in model_set, x[0], x[1])):
        entity_lower = entity.lower().strip()
        if entity_lower not in seen_entities:
            seen_entities.add(entity_lower)
            final_entities.append((entity, label))

    final_entities.sort(key=lambda x: (x[1], x[0]))
    status["entity_count"] = len(final_entities)

    counts = (f"모델 {status['model_entities']}개 · 정규식 {status['regex_entities']}개 · 최종 {len(final_entities)}개")
    if status["mode"] != "model":
        if remembered is not None:
            why = (f" · 앞선 적재 실패 기억됨({remembered.failed_ago:.0f}초 전, {remembered.retry_in:.0f}초 뒤 다시 시도): "
                   f"{status['load_error']}")
        else:
            why = f" · 사유: {status['load_error']}" if status["load_error"] else ""
        # 실패를 기억해 다시 적재하지 않은 호출은 원래 WARNING 이지만, 시험 실행마다 첫 번째는 ERROR 로 올린다 —
        # 앞선 실행에서 난 적재 실패가 이번 실행의 로그에서 경고로만 보여 묻히지 않게 한다
        level = logging.ERROR if remembered is not None and _first_fallback_in_run() else logging.WARNING
        logger.log(level, f"모델 미적재 — 정규표현식 대체 ({counts}) · 모델 {model_name}{why}")
    else:
        logger.debug(f"NER 예측 — {counts}")

    if debug:
        for entity, label in final_entities[:10]:
            logger.debug(f"  - {entity} ({label})")

    return final_entities, status


_FALLBACK_RUNS: "set[str]" = set()
_FALLBACK_RUNS_LOCK = threading.Lock()


def _first_fallback_in_run() -> bool:
    """지금 문맥이 시험 실행에 묶여 있고, 그 실행에서 처음 맞는 정규표현식 대체이면 True (실행 밖이면 False)."""
    try:
        try:
            from module import log_context as _lc
        except ImportError:
            from api.module import log_context as _lc
        b = _lc.current()
    except Exception:
        return False
    rid = getattr(b, "run_id", None) if b else None
    if not rid:
        return False
    with _FALLBACK_RUNS_LOCK:
        if rid in _FALLBACK_RUNS:
            return False
        if len(_FALLBACK_RUNS) > 1000:
            _FALLBACK_RUNS.clear()
        _FALLBACK_RUNS.add(rid)
        return True


def extract_entities_from_text(
    text: str,
    model_name: Optional[str] = None,
    *,
    model_path: Optional[Path] = None,
    debug: bool = False
) -> List[Tuple[str, str]]:
    """텍스트에서 엔티티 추출 (통합 메인 함수) — 상태가 필요하면 extract_entities_with_status 를 쓴다."""
    return extract_entities_with_status(text, model_name, model_path=model_path, debug=debug)[0]

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
        - 결과에는 추출 방식이 함께 붙는다: mode("model" | "regex_fallback"; 아무 파일도 처리하지 못했으면 None),
          model_loaded(bool), load_error(str|None — 모델을 못 올린 사유), entity_count(추출 엔티티 수)
    """
    start_time = time.time()

    def _failed(error: str) -> Dict[str, Any]:
        return {"success": False, "error": error, "mode": None, "model_loaded": False,
                "load_error": None, "entity_count": 0}
    
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
            return _failed(f"입력 경로가 존재하지 않습니다: {input_path_obj}")
        
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
            if input_path_obj.suffix.lower() in ['.ocr', '.txt', '.md']:
                files_to_process = [input_path_obj]
        else:
            for ext in ['*.ocr', '*.txt', '*.md']:
                files_to_process.extend(input_path_obj.glob(f"**/{ext}"))
        
        if not files_to_process:
            return _failed("처리할 텍스트 파일이 없습니다.")
        
        if debug:
            print(f"처리할 파일 수: {len(files_to_process)}")
            print(f"처리할 파일: {files_to_process}")
            print("-"*100)
        
        # 5. 모델 확인 (없으면 에러)
        model_path = get_model_path(model_name)
        model_exists = model_path.exists() and (model_path / "config.json").exists()
        
        if not model_exists:
            logger.error(f"NER 모델이 없습니다: {model_name} ({model_path}) — ner_train() 함수를 먼저 실행하세요")
            out = _failed(f"모델이 없습니다: {model_name}. ner_train() 함수를 먼저 실행하세요.")
            out["load_error"] = f"모델 파일 없음: {model_path}"
            return out
        
        if debug:
            print(f"✓ 모델 로드: {model_path}")
        
        # 6. 엔티티 추출 시작
        if debug:
            print("엔티티 추출 시작...")
        
        all_entities = []
        processed_files = 0
        statuses: List[Dict[str, Any]] = []
        
        import sys
        for file_path in tqdm(files_to_process, desc="파일 처리 중", disable=False, file=sys.stdout, ncols=80):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if len(content.strip()) < 2:  # 최소 2글자 이상
                    continue
                
                entities, ner_status = extract_entities_with_status(
                    content,
                    model_name=model_name,
                    model_path=model_path,
                    debug=debug
                )
                statuses.append(ner_status)
                
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
        
        # 추출 방식 — 파일 하나라도 정규식으로 대체했으면 regex_fallback
        if statuses:
            mode = "model" if all(st.get("mode") == "model" for st in statuses) else "regex_fallback"
        else:
            mode = None
        model_loaded = bool(statuses) and all(st.get("model_loaded") for st in statuses)
        load_error = next((st.get("load_error") for st in statuses if st.get("load_error")), None)
        logger.debug(f"NER 예측 완료 — 파일 {processed_files}/{len(files_to_process)} · 엔티티 {len(all_entities)}개 · "
                     f"방식 {mode} · {summary['processing_time']:.1f}초 · 저장 {ner_dir}")
        
        return {
            "success": True,
            "mode": mode,
            "model_loaded": model_loaded,
            "load_error": load_error,
            "entity_count": len(all_entities),
            "processed_files": processed_files,
            "total_entities": len(all_entities),
            "extracted_entities": all_entities,
            "statistics": summary,
            "output_directory": str(ner_dir),
            "processing_time": summary['processing_time'],
            "summary_file": str(summary_file)
        }
        
    except Exception as e:
        logger.error(f"NER 예측 중 오류: {type(e).__name__}: {e}")
        return _failed(f"예측 중 오류 발생: {str(e)}")

def ensure_training_data_exists(model_name: str, num_samples: int = 7500, force_regenerate: bool = False, use_large_dataset: bool = True) -> bool:
    """prepare_training_data 헬퍼를 사용해 BIO 데이터셋을 준비"""

    model_name_safe = model_name.replace('/', '-')
    training_dir = Path(__file__).parent / "training" / model_name_safe

    try:
        from .ner_train import prepare_training_data
    except ImportError as exc:
        print(f"[경고] prepare_training_data import 실패: {exc}")
        return False

    try:
        prepare_training_data(
            model_name=model_name,
            num_samples=num_samples,
            force_regenerate=force_regenerate,
            use_realistic_data=use_large_dataset,
            balanced=True,
            training_root=training_dir
        )
    except Exception as exc:
        print(f"[경고] 훈련 데이터 준비 실패: {exc}")
        import traceback
        traceback.print_exc()
        return False

    train_file = training_dir / "train.txt"
    val_file = training_dir / "validation.txt"

    if train_file.exists() and val_file.exists():
        print(f"✓ BIO 데이터 준비 완료: {training_dir}")
        return True

    print(f"[경고] BIO 데이터 파일이 누락되었습니다: {training_dir}")
    return False

def ner_train(
    model_name: str = DEFAULT_MODEL_NAME,
    iterations: int = 1,
    epochs: int = 100,
    batch_size: int = 12,
    learning_rate: float = 1e-5,
    num_train_samples: int = 30000,
    enable_visualization: bool = True,
    enable_early_stopping: bool = False,
    enable_balanced_sampling: bool = True,
    debug: bool = False,
    force_regenerate_data: bool = True
) -> Dict[str, Any]:
    """Wrapper that delegates NER training to ``ner_train.py`` implementation."""

    overall_start = time.time()

    from . import ner_train as training_module

    training_result = training_module.ner_train(
        model_name=model_name,
        iterations=iterations,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_samples=num_train_samples,
        enable_visualization=enable_visualization,
        enable_early_stopping=enable_early_stopping,
        enable_balanced_sampling=enable_balanced_sampling,
        debug=debug,
        force_regenerate_data=force_regenerate_data
    )

    results = training_result.get('all_iterations', [])
    latest_metrics = results[-1].get('metrics', {}) if results else {}
    history = latest_metrics.get('history', {}) if isinstance(latest_metrics, dict) else {}

    all_metrics = {
        'epochs': list(history.get('epoch', [])),
        'train_loss': list(history.get('train_loss', [])),
        'val_loss': list(history.get('eval_loss', [])),
        'val_f1': list(history.get('eval_f1', [])),
        'val_precision': list(history.get('eval_precision', [])),
        'val_recall': list(history.get('eval_recall', [])),
        'train_loss_history': list(history.get('train_loss', [])),
        'steps': list(range(1, len(history.get('train_loss', [])) + 1))
    }

    total_time = training_result.get('total_time', time.time() - overall_start)

    try:
        model_path = get_model_path(model_name)
        if results:
            last_iteration = results[-1]
            training_info = {
                "iteration": last_iteration.get('iteration'),
                "training_time": last_iteration.get('training_time', 0.0),
                "training_time_minutes": last_iteration.get('training_time', 0.0) / 60,
                "epochs": epochs,
                "completed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "config": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "model_name": model_name,
                    "num_train_samples": num_train_samples
                },
                "metrics": latest_metrics
            }

            training_info_file = model_path / "training_info.json"
            with open(training_info_file, 'w', encoding='utf-8') as f:
                json.dump(training_info, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[경고] training_info 저장 실패: {exc}")

    success = training_result.get('status', '').lower() == 'success'

    return {
        "success": success,
        "status": training_result.get('status', 'success' if success else 'failed'),
        "model_name": model_name,
        "iterations": iterations,
        "results": results,
        "metrics": all_metrics,
        "total_time": total_time
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


