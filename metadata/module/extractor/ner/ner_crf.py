"""NER 모델 클래스 (외부 모델: BERT 등)"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from .config import BIO_LABELS, ID_TO_LABEL


class NER:
    """외부 NER 모델 클래스 (BERT 등)"""
    
    def __init__(self, model_name: str = "google-bert/bert-base-multilingual-cased", 
                 model_path: Optional[str] = None):
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """모델 및 토크나이저 로드"""
        try:
            if self.model_path and Path(self.model_path).exists():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_path,
                    num_labels=len(BIO_LABELS),
                    ignore_mismatched_sizes=True
                )
            else:
                print(f"[NER] 모델 다운로드 중: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_name,
                    num_labels=len(BIO_LABELS),
                    ignore_mismatched_sizes=True
                )
            
            self.model.to(self.device)
            self.model.eval()
            print(f"[NER] 모델 로드 완료: {self.model_name} (device: {self.device})")
        except Exception as e:
            raise RuntimeError(f"모델 로드 실패: {e}")
    
    def predict(self, texts: List[List[str]]) -> List[List[str]]:
        """NER 예측"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        
        predictions = []
        for tokens in texts:
            encoded = self.tokenizer(
                tokens,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                encoded_device = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded_device)
                pred_ids = outputs.logits.argmax(dim=-1).cpu().numpy()[0]
            
            input_ids = encoded["input_ids"][0].cpu().numpy()
            aligned_preds = self._align_predictions(tokens, input_ids, pred_ids)
            predictions.append(aligned_preds)
        
        return predictions
    
    def _align_predictions(self, tokens: List[str], input_ids: np.ndarray, 
                          pred_ids: np.ndarray) -> List[str]:
        """subword 토큰 예측을 원본 토큰에 정렬"""
        token_to_subwords = []
        for token in tokens:
            subwords = self.tokenizer.tokenize(token)
            token_to_subwords.append(len(subwords) if subwords else 1)
        
        aligned = []
        subword_idx = 1  # [CLS] 건너뛰기
        
        for num_subwords in token_to_subwords:
            if subword_idx >= len(pred_ids):
                aligned.append("O")
                continue
            aligned.append(ID_TO_LABEL.get(pred_ids[subword_idx], "O"))
            subword_idx += num_subwords
        
        while len(aligned) < len(tokens):
            aligned.append("O")
        
        return aligned[:len(tokens)]
    
    def save(self, save_path: str):
        """모델 저장"""
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"[NER] 모델 저장 완료: {save_path}")
    
    def load(self, load_path: str):
        """모델 로드"""
        self.model_path = load_path
        self._load_model()

