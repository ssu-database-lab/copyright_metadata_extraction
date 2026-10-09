"""
API 모듈 초기화
NER, OCR, PDF 처리 통합 API

무거운 하위 스택(NER 의 torch/transformers)은 **지연 로드**한다 (PEP 562 `__getattr__`).
`import api` 만으로 30초가 걸리면 평가 CLI(`python -m api.module.evaluation.v4.evaluate`)가
쓸 수 없다. `from api import ner_predict` 처럼 실제로 이름을 꺼낼 때 비로소 `api/api.py`
(→ module.ner.ner_system → torch) 가 올라오므로 웹앱(web/app.py:47)·CLI 는 그대로 동작한다.
"""

import importlib

__version__ = "1.0.0"
__author__ = "SSU Database Lab"

# api.py 가 제공하는 함수들 — 이름만 미리 선언하고 본체는 첫 접근 때 가져온다.
_LAZY_NAMES = (
    'pdf_to_image',
    'ocr_naver',
    'ocr_mistral',
    'ocr_google',
    'ocr_complete',
    'ner_predict',
    'ner_train',
    'process_pdf_to_ner',
    'get_api_info',
)

__all__ = list(_LAZY_NAMES)


def __getattr__(name):
    """`from api import ner_predict` → 이때 처음으로 api/api.py(와 NER/OCR 스택)를 적재한다."""
    if name in _LAZY_NAMES:
        impl = importlib.import_module(".api", __name__)
        value = getattr(impl, name)
        globals()[name] = value          # 두 번째부터는 일반 속성 조회로 끝난다
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_NAMES))
