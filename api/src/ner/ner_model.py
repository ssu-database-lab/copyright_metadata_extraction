# ner_model.py
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import json
import os

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer


@dataclass
class NERConfig:
    """
    NER 모델 전체 설정.
    """
    model_name_or_path: str = "bert-base-multilingual-cased"
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 1
    dropout: float = 0.1
    max_length: int = 128

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "NERConfig":
        return cls(**data)


class BertBiLstmCrf(nn.Module):
    """
    BERT + BiLSTM + CRF 기반 NER 모델.
    """
    def __init__(self, config: NERConfig, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels

        # BERT backbone
        self.bert = AutoModel.from_pretrained(config.model_name_or_path)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(config.dropout)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=config.lstm_hidden_size // 2,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Linear layer to num_labels
        self.classifier = nn.Linear(config.lstm_hidden_size, num_labels)

        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        labels가 주어지면 loss 반환,
        아니면 CRF decode 결과(각 시퀀스별 label id 리스트) 반환.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, H)
        sequence_output = self.dropout(sequence_output)

        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)

        emissions = self.classifier(lstm_output)  # (B, L, num_labels)
        mask = attention_mask.bool()

        if labels is not None:
            # CRF는 -100을 처리할 수 없으므로 0("O")으로 치환하여 계산
            # 실제 평가 시에는 -100 부분을 제외하고 평가하므로 문제 없음
            labels_for_crf = labels.clone()
            labels_for_crf[labels_for_crf == -100] = 0
            
            loss = -self.crf(emissions, labels_for_crf, mask=mask, reduction="mean")
            return loss

        # decode: List[List[int]] 형태
        predictions = self.crf.decode(emissions, mask=mask)
        return predictions


def save_ner_model(
    output_dir: str,
    model: BertBiLstmCrf,
    tokenizer,
    config: NERConfig,
    label2id: Dict[str, int],
) -> None:
    """
    모델 가중치 + 토크나이저 + NER 설정 + 라벨 매핑 저장.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)

    # 토크나이저는 HuggingFace 방식 그대로 저장
    tokenizer.save_pretrained(output_dir)

    id2label = {int(v): k for k, v in label2id.items()}

    config_path = os.path.join(output_dir, "ner_config.json")
    to_save = {
        "ner_config": config.to_dict(),
        "label2id": label2id,
        "id2label": id2label,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, indent=2)


def load_ner_model(
    model_dir: str,
    device: torch.device = None,
) -> Tuple[BertBiLstmCrf, AutoTokenizer, Dict[str, int], Dict[int, str], NERConfig]:
    """
    저장된 모델 디렉토리에서 모델/토크나이저/라벨 매핑/설정을 로드.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(model_dir, "ner_config.json")
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    ner_cfg = NERConfig.from_dict(data["ner_config"])
    label2id = {k: int(v) for k, v in data["label2id"].items()}
    id2label = {int(k): v for k, v in data["id2label"].items()}

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = BertBiLstmCrf(ner_cfg, num_labels=len(label2id))

    model_path = os.path.join(model_dir, "pytorch_model.bin")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, tokenizer, label2id, id2label, ner_cfg