#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BERT 모델 정보 확인 스크립트"""

from transformers import AutoConfig
from pathlib import Path

bert_path = Path("models/pretrained_bert")

if bert_path.exists():
    print(f"[INFO] 로컬 BERT 모델 발견: {bert_path.absolute()}")
    try:
        config = AutoConfig.from_pretrained(str(bert_path))
        print(f"  모델명: {config.name_or_path if hasattr(config, 'name_or_path') else 'bert-base-multilingual-cased'}")
        print(f"  Vocab 크기: {config.vocab_size}")
        print(f"  Hidden 크기: {config.hidden_size}")
        print(f"  레이어 수: {config.num_hidden_layers}")
        print(f"  어텐션 헤드 수: {config.num_attention_heads}")
        print(f"  ✅ 로컬 BERT 모델이 정상적으로 로드됩니다.")
    except Exception as e:
        print(f"  ❌ 모델 로드 실패: {e}")
else:
    print(f"[INFO] 로컬 BERT 모델 없음: {bert_path}")
    print(f"  HuggingFace에서 다운로드하여 사용: bert-base-multilingual-cased")

