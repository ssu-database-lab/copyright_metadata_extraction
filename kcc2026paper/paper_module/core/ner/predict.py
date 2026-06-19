"""Compat shim — predict.py 본체는 module/extractor/ner/predict.py 로 이동.

paper{1..6}.py 호환용. 신규 코드는 ``from module.api import ner_predict`` 사용.
"""
from module.extractor.ner.predict import *  # noqa: F401,F403
from module.extractor.ner.predict import ner_predict  # noqa: F401
