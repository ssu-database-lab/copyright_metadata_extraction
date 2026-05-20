"""모델·학습 하이퍼파라미터 설정."""
from __future__ import annotations

from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# integrated: 모델 × 학습 방법 × LoRA rank
# ---------------------------------------------------------------------------

BERT_MODELS: Tuple[str, ...] = (
    "google-bert/bert-base-multilingual-cased",   # mBERT
    # "FacebookAI/xlm-roberta-large",             # 학습 속도 과다 → 제외
    "klue/bert-base",                              # KLUE BERT (한국어 특화)
    "microsoft/deberta-v3-base",                   # DeBERTa-v3
    "monologg/koelectra-base-v3-discriminator",    # KoELECTRA
)

FINE_TUNING_METHODS: Tuple[str, ...] = ("lora", "full")
LORA_R_VALUES: Tuple[int, ...] = (4, 8, 16, 32)


# ---------------------------------------------------------------------------
# 학습 하이퍼파라미터 (fallback 기본값)
# ---------------------------------------------------------------------------

EPOCHS = 10
BATCH_SIZE = 8
LR = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
LORA_R = 8
LORA_ALPHA = 16
EARLY_STOPPING_PATIENCE = 3

TRAIN_RATIO = 8 / 12
VAL_RATIO = 2 / 12
TEST_RATIO = 2 / 12
SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# 튜닝된 하이퍼파라미터 프로파일
# ---------------------------------------------------------------------------

TUNED_HPARAMS: Dict[str, Dict[str, float]] = {
    "lora": {
        "lr": 3e-5, "epochs": 15, "warmup_ratio": 0.1,
        "weight_decay": 0.01, "lora_r": 16, "lora_alpha": 32,
        "early_stopping_patience": 4,
    },
    "full": {
        "lr": 2e-5, "epochs": 5, "warmup_ratio": 0.1,
        "weight_decay": 0.1,
        "early_stopping_patience": 2,
    },
}

LORA_RANK_HPARAMS: Dict[str, float] = {
    "lr": 3e-5, "epochs": 10, "warmup_ratio": 0.1,
    "weight_decay": 0.01, "lora_alpha": 32,
    "early_stopping_patience": 3,
}


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

THRESHOLD_SWEEP: Tuple[float, ...] = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)

# 라벨당 최대 학습 레코드 (None = 제한 없음)
# 모든 라벨 동일하게 MAX_PER_LABEL건 → 가중치 균등
MAX_PER_LABEL = 300
