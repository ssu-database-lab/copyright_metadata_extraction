"""
NER 추출기 메인 인터페이스
- predict(): API에서 들어온 sentences/tokens를 AdapterNER 입력 형태로 변환
- train(): configs/training 기반으로 라벨별 어댑터 학습
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from module.parts.types import Decision
from .adapter import train as train_module
from .config import BIO_LABELS, TRAINING_DATA_DIR


def _get_model(model_name: Optional[str] = None,
               model_path: Optional[str] = None,
               adapter_dir: Optional[str] = None):
    from .adapter.ner import AdapterNER
    adapter_path = adapter_dir or "models/ner/adapters"
    return AdapterNER(
        model_name=model_name or "bert-base-multilingual-cased",
        model_path=model_path,
        adapter_dir=adapter_path
    )


def predict(
    sentences: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    model_path: Optional[str] = None,
    adapter_dir: Optional[str] = None,
    **kwargs
) -> List[Decision]:
    if not BIO_LABELS:
        return []

    # 기본값
    if adapter_dir is None:
        adapter_dir = "models/ner/adapters"
    if model_name is None:
        model_name = "bert-base-multilingual-cased"
    if model_path is None:
        model_path = f"model_downloaded/{model_name}"
        if not Path(model_path).exists():
            model_path = model_name

    model = _get_model(model_name=model_name, model_path=model_path, adapter_dir=adapter_dir)

    # token을 sentence별로 묶기 (기존 로직 유지)
    decisions: List[Decision] = []
    token_groups: Dict[int, List[Dict[str, Any]]] = {}
    for token in tokens:
        if (sid := token.get("sent_id")) is not None:
            token_groups.setdefault(int(sid), []).append(token)

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

    predicted_labels_list = model.predict(sentence_texts)

    # BIO -> Decision 변환 (기존 로직 유지/개선 가능)
    for (sid, sent_tokens), predicted_labels in zip(sentence_info, predicted_labels_list):
        cur_val = None
        cur_label = None
        cur_tok_id = None

        for tok, tag in zip(sent_tokens, predicted_labels):
            if tag == "O":
                if cur_val:
                    decisions.append(Decision(label=cur_label, value=cur_val, sent_id=sid, tok_id=cur_tok_id, source="ner"))
                    cur_val = None
                continue

            if tag.startswith("B-"):
                if cur_val:
                    decisions.append(Decision(label=cur_label, value=cur_val, sent_id=sid, tok_id=cur_tok_id, source="ner"))
                cur_label = tag[2:]
                cur_val = tok.get("text", "")
                cur_tok_id = tok.get("tok_id")

            elif tag.startswith("I-") and cur_val is not None:
                cur_val += " " + tok.get("text", "")

        if cur_val:
            decisions.append(Decision(label=cur_label, value=cur_val, sent_id=sid, tok_id=cur_tok_id, source="ner"))

    return decisions


def train(
    model_name: str = "bert-base-multilingual-cased",
    model_path: Optional[str] = None,
    adapter_dir: str = "models/ner/adapters",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    train_data_path: Optional[str] = None,
    train_ratio: float = 0.8,
    random_seed: int = 42,
    **kwargs
) -> Dict[str, Any]:
    # 기본 데이터 폴더: configs/training
    if train_data_path is None:
        train_data_path = str(TRAINING_DATA_DIR)

    return train_module.train_adapter_ner(
        model_name=model_name,
        model_path=model_path,
        adapter_dir=adapter_dir,
        train_data_path=train_data_path,
        train_ratio=train_ratio,
        random_seed=random_seed,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        **kwargs
    )


def ner_extractor(*, sentences: List[Dict[str, Any]], tokens: List[Dict[str, Any]]):
    return predict(sentences=sentences, tokens=tokens)
