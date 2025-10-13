"""
OCR Module for API
Provides unified OCR functionality with multiple providers
"""

from .universal_ocr import UniversalOCRProcessor, OCRProvider
from .google_ocr import GoogleCloudOCRProvider
from .mistral_ocr import MistralOCRProvider
from .naver_ocr import NaverOCRProvider
from .alibaba_ocr import AlibabaCloudOCRProvider

__all__ = [
    'UniversalOCRProcessor',
    'OCRProvider',
    'GoogleCloudOCRProvider',
    'MistralOCRProvider', 
    'NaverOCRProvider',
    'AlibabaCloudOCRProvider'
]
