"""
API 모듈 초기화
NER, OCR, PDF 처리 통합 API
"""

__version__ = "1.0.0"
__author__ = "SSU Database Lab"

# api.py의 모든 함수를 __init__.py에서 제공
from .api import (
    pdf_to_image,
    ocr_naver,
    ocr_mistral,
    ocr_google,
    ocr_complete,
    ner_predict,
    ner_train,
    process_pdf_to_ner,
    get_api_info
)

__all__ = [
    'pdf_to_image',
    'ocr_naver',
    'ocr_mistral', 
    'ocr_google',
    'ocr_complete',
    'ner_predict',
    'ner_train',
    'process_pdf_to_ner',
    'get_api_info'
]
