"""NER 추출기 메인 인터페이스"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..types import Decision

# train.py는 여기서만 import (접근 제어)
from . import train as train_module


# ========== 설정 및 상수 ==========

from .config import BIO_LABELS, LABEL_TO_ID, ID_TO_LABEL, MODEL_DIR


# ========== 모델 팩토리 ==========


def _get_model(model_type: str = "ner", model_name: Optional[str] = None, 
               model_path: Optional[str] = None):
    """모델 타입에 따라 적절한 모델 인스턴스 반환"""
    if model_type == "ner":
        from .ner_crf import NER
        return NER(model_name=model_name or "google-bert/bert-base-multilingual-cased", 
                   model_path=model_path)
    elif model_type == "bilstm_crf":
        from .bilstm_crf import BiLSTMCRF
        return BiLSTMCRF()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'ner' or 'bilstm_crf'.")


# ========== 메인 함수들 ==========


def predict(
    sentences: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
    model_type: str = "ner",
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    **kwargs
) -> List[Decision]:
    """NER 예측"""
    if not BIO_LABELS:
        return []
    
    model = _get_model(model_type=model_type, model_name=model_name, model_path=model_path)
    decisions: List[Decision] = []
    
    # 토큰을 문장별로 그룹화
    token_groups: Dict[int, List[Dict[str, Any]]] = {}
    for token in tokens:
        if (sid := token.get("sent_id")) is not None:
            token_groups.setdefault(int(sid), []).append(token)
    
    # 문장별로 토큰 텍스트 수집
    sentence_texts: List[List[str]] = []
    sentence_info: List[Tuple[int, List[Dict[str, Any]]]] = []
    
    for sentence in sentences:
        if (sid := sentence.get("sent_id")) is None:
            continue
        
        sent_tokens = token_groups.get(int(sid), [])
        token_texts = [t.get("text", "") for t in sent_tokens]
        
        if token_texts:
            sentence_texts.append(token_texts)
            sentence_info.append((sid, sent_tokens))
    
    if not sentence_texts:
        return decisions
    
    # 모델 예측
    predicted_labels_list = model.predict(sentence_texts)
    
    # 예측 결과를 Decision으로 변환
    for (sid, sent_tokens), predicted_labels in zip(sentence_info, predicted_labels_list):
        current_entity = None
        current_label = None
        current_start_idx = None
        
        for token, label in zip(sent_tokens, predicted_labels):
            if label == "O":
                if current_entity:
                    decisions.append(Decision(
                        label=current_label,
                        value=current_entity,
                        sent_id=sid,
                        tok_id=current_start_idx,
                        source="ner"
                    ))
                    current_entity = None
            elif label.startswith("B-"):
                if current_entity:
                    decisions.append(Decision(
                        label=current_label,
                        value=current_entity,
                        sent_id=sid,
                        tok_id=current_start_idx,
                        source="ner"
                    ))
                current_label = label.replace("B-", "")
                current_entity = token.get("text", "")
                current_start_idx = token.get("tok_id")
            elif label.startswith("I-") and current_entity:
                current_entity += " " + token.get("text", "")
        
        if current_entity:
            decisions.append(Decision(
                label=current_label,
                value=current_entity,
                sent_id=sid,
                tok_id=current_start_idx,
                source="ner"
            ))
    
    return decisions


def train(
    model_type: str = "ner",
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    dataset_size: Optional[int] = None,
    samples_per_file: Optional[int] = None,
    sample_ratio_per_file: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """NER 모델 학습 (train.py 호출)"""
    return train_module.train_model(
        model_type=model_type,
        model_name=model_name,
        model_path=model_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_ratio=train_ratio,
        random_seed=random_seed,
        dataset_size=dataset_size,
        samples_per_file=samples_per_file,
        sample_ratio_per_file=sample_ratio_per_file,
        **kwargs
    )


def validate(
    model_type: str = "ner",
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    **kwargs
) -> Dict[str, float]:
    """NER 모델 검증"""
    return train_module.validate_model(
        model_type=model_type,
        model_name=model_name,
        model_path=model_path,
        val_data_path=val_data_path,
        **kwargs
    )


# ========== NER Extractor (기존 인터페이스 유지) ==========


def ner_extractor(
    *,
    sentences: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
) -> List[Decision]:
    """NER 기반 추출 (api에서 호출)"""
    return predict(sentences=sentences, tokens=tokens)

