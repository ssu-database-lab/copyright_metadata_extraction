"""
NER 모듈 공통 설정 및 상수
- labels.yaml -> NER 라벨 목록 로드
- BIO_LABELS / LABEL_TO_ID / ID_TO_LABEL 생성
- 경로: model_downloaded / models / configs/training
"""
from __future__ import annotations

from pathlib import Path
from typing import List

CONFIG_PATH = Path("configs/labels.yaml")

# 사용자 요청 반영
DOWNLOADED_MODEL_DIR = Path("model_downloaded")      # 원본 모델 저장
MODEL_DIR = Path("models/ner")                       # 학습 산출물(어댑터 등)
ADAPTER_DIR = MODEL_DIR / "adapters"                 # 최종 어댑터 저장
TRAINING_DATA_DIR = Path("configs/training")         # 학습 데이터 모음


def load_ner_labels() -> List[str]:
    """labels.yaml에서 NER 라벨을 로드하여 BIO 태깅 스키마를 생성"""
    import yaml
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        ner_config = config.get("ner", {})
        ner_labels = ner_config.get("labels", []) if isinstance(ner_config, dict) else []
    except Exception:
        return []

    if not ner_labels:
        return []

    bio_labels = ["O"]
    for label in ner_labels:
        bio_labels.extend([f"B-{label}", f"I-{label}"])
    return bio_labels


BIO_LABELS = load_ner_labels()
LABEL_TO_ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
