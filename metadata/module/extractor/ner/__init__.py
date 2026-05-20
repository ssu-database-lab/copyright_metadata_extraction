"""NER 추출기 — 학습 + 예측."""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .base import ner_predict, ner_train  # noqa: F401

__all__ = ["ner_predict", "ner_train"]
