"""
추출기 팩토리 모듈.
다음 함수를 외부에 노출한다:
- ner_extractor
- ocr (extract_text_from_file)
"""

from module.parts.types import Decision
from .ner import ner_extractor
from . import ocr

__all__ = ["Decision", "ner_extractor", "ocr"]