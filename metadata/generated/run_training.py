#!/usr/bin/env python3
"""Quick script placeholder for NER training (zero-shot mode)."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

# Set environment variable to avoid distributed tensor import issues
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'

# Zero-shot 모드에서는 학습을 수행하지 않습니다.
from module.extractor.ner import base as ner_base

if __name__ == "__main__":
    try:
        ner_base.train()
    except RuntimeError as exc:
        print(f"[안내] {exc}")
