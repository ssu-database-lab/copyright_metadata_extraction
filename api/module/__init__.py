"""
AI API 모듈 패키지
PDF 처리, OCR, NER 기능을 제공하는 모듈화된 API

하위 모듈은 **지연 로드**한다 (PEP 562). `module.ner.ner_system` 이 torch/transformers 를
올리는 데 ~30초가 걸려, 스키마 직렬화기·라우터·평가 CLI 처럼 모델이 필요 없는 import 까지
그 비용을 치르고 있었다. `from module import ner_predict` / `module.ocr_complete` 는
그대로 동작하고, `module.ocr`, `module.evaluation` 같은 하위 패키지 import 도 영향이 없다.
"""

import importlib

# 이름 → 그 이름을 정의한 하위 모듈 (첫 접근 때 import 한다)
_LAZY_ATTRS = {
    'pdf_to_image': '.pdf_to_image',
    'ocr_naver': '.ocr_system',
    'ocr_mistral': '.ocr_system',
    'ocr_google': '.ocr_system',
    'ocr_complete': '.ocr_system',
    'ner_predict': '.ner.ner_system',
    'ner_train': '.ner.ner_system',
}

__all__ = [
    'pdf_to_image',
    'ocr_naver', 'ocr_mistral', 'ocr_google', 'ocr_complete',
    'ner_predict', 'ner_train'
]


def __getattr__(name):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(target, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))
