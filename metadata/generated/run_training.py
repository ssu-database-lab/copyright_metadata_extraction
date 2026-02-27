#!/usr/bin/env python3
"""Quick script placeholder for NER training (zero-shot mode)."""

# import
import sys
import os

sys.path.insert(0, os.getcwd())
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'

from module.extractor.ner import base as ner_base


# -----------------------------------------------------------------------------
# 실행
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        ner_base.train()
    except RuntimeError as exc:
        print(f"[안내] {exc}")
