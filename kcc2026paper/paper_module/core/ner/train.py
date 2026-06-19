"""Compat shim — train.py 본체는 module/extractor/ner/train.py 로 이동.

paper{1..6}.py 의 ``from paper_module.core.ner.train import ner_train`` 호환용.
신규 코드는 ``from module.api import ner_train`` 사용.
"""
from module.extractor.ner.train import *  # noqa: F401,F403
from module.extractor.ner.train import ner_train  # noqa: F401
