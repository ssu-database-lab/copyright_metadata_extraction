# module/ner/ner_constants.py
from dataclasses import dataclass

# 엔티티 타입 (23개) - 한 곳만 진실의 원천으로
ENTITY_TYPES = [
    "NAME", "PHONE", "ADDRESS", "DATE", "COMPANY", "EMAIL", "POSITION",
    "CONTRACT_TYPE", "MONEY", "PERIOD", "ID_NUM", "CONSENT_TYPE", "RIGHT_INFO",
    "PROJECT_NAME", "LAW_REFERENCE", "TITLE", "URL", "DESCRIPTION", "TYPE",
    "STATUS", "DEPARTMENT", "LANGUAGE", "QUANTITY",
]

BIO_LABELS = ["O"] + [
    f"{prefix}-{entity}"
    for entity in ENTITY_TYPES
    for prefix in ["B", "I"]
]
LABEL_TO_ID = {label: i for i, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


@dataclass
class TrainConfig:
    model_name: str = "google-bert/bert-base-multilingual-cased"
    num_epochs: int = 300
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 1e-5
    max_length: int = 256
    dropout: float = 0.15
    warmup_ratio: float = 0.08
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    adaptive_grad_clip: bool = True
    agc_percentile: float = 10.0
    label_smoothing: float = 0.1
    ema_decay: float = 0.999
    layer_lr_decay: float = 0.95
    enable_loss_smoothing: bool = True
    enable_balanced_sampling: bool = True
