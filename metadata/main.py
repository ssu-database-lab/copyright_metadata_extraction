"""Production inference — ``python main.py``.

단일 진입점: ``module.api.extract_metadata`` 만 사용.
End-to-end 파이프라인: OCR → regex(9) → NER(17) → LLM(9, placeholder) → 35-label JSON.

OCR 캐시(``data/out/ocr/result/*.txt``)가 입력 문서 수만큼 있으면 OCR 단계 자동 스킵.
``FORCE_OCR=1 python main.py`` 로 강제 재실행 가능.
"""
from module.api import extract_metadata

if __name__ == "__main__":
    extract_metadata()
