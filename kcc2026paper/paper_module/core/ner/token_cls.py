"""Compat shim — token_cls.py 본체는 module/extractor/ner/token_cls.py 로 이동.

paper{1..6}.py / smoke 등 기존 임포트 호환용 re-export.
신규 코드는 module/extractor/ner/token_cls.py 를 직접 import.
"""
from module.extractor.ner.token_cls import *  # noqa: F401,F403
from module.extractor.ner.token_cls import TokenClassNER  # noqa: F401
