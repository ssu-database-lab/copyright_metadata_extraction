"""NER 모듈 공통 설정 및 상수"""
from __future__ import annotations

from pathlib import Path
from typing import List

CONFIG_PATH = Path("configs/labels.yaml")
MODEL_DIR = Path("models/ner")


def load_ner_labels() -> List[str]:
    """labels.yaml에서 NER 라벨을 로드하여 BIO 태깅 스키마를 생성"""
    import yaml
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        ner_labels = config.get("ner_labels", [])
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

