"""NER 추출기 모듈 (Token Classification / BERT 계열).

``ner_train``은 순환 import 방지를 위해 지연 로드한다.
"""
from .base import (
    detect_model_type,
    model_display_name,
    ner_predict,
)

__all__ = [
    "detect_model_type",
    "model_display_name",
    "ner_predict",
    "ner_train",
]


def __getattr__(name: str):
    if name == "ner_train":
        from .train import ner_train

        return ner_train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
