#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BERT 모델 다운로드 스크립트"""

from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import os

def download_bert():
    model_name = "bert-base-multilingual-cased"
    output_dir = Path("models/pretrained_bert")
    
    print(f"[INFO] BERT 모델 다운로드 중: {model_name}")
    print(f"[INFO] 저장 경로: {output_dir.absolute()}")
    
    # 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 토큰라이저 다운로드
    print("[INFO] 토큰라이저 다운로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 모델 다운로드
    print("[INFO] 모델 다운로드 중... (시간이 걸릴 수 있습니다)")
    model = AutoModel.from_pretrained(model_name)
    
    # 저장
    print("[INFO] 모델 저장 중...")
    tokenizer.save_pretrained(str(output_dir))
    model.save_pretrained(str(output_dir))
    
    # 확인
    config_file = output_dir / "config.json"
    vocab_file = output_dir / "vocab.txt"
    
    if config_file.exists() and vocab_file.exists():
        print(f"[INFO] ✅ 로컬 BERT 모델 저장 완료!")
        print(f"[INFO] 위치: {output_dir.absolute()}")
        print(f"[INFO] 모델: {model_name}")
    else:
        print("[ERROR] 모델 저장 실패 - 파일 확인 필요")

if __name__ == "__main__":
    download_bert()

