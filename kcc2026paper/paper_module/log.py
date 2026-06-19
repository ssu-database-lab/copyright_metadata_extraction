"""Compat shim — log.py 본체는 module/extractor/ner/full_logger.py 로 이동."""
from module.extractor.ner.full_logger import *  # noqa: F401,F403
from module.extractor.ner.full_logger import from_env  # noqa: F401
