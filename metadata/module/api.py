"""공개 API.

- ocr_extract       : module.extractor.ocr        → PDF/이미지 → 텍스트
- regex_extract     : module.extractor.regex      → 9 strict-format 라벨
- ner_predict       : module.extractor.ner.base   → 35-라벨 메타데이터 (regex+NER+후처리+LLM placeholder)
- ner_train         : module.extractor.ner.base   → NER 학습 (저수준, 인자 직접 지정)
- llm_extract       : module.extractor.llm.llm    → 9 위임 라벨 (미구현)
- extract_metadata  : ★ OCR + NER + LLM end-to-end 오케스트레이터 (main.py)
- train_metadata    : ★ 배포 NER 모델 재학습 (train.py) — silver + 증강 full FT
"""
from module.extractor.ner import base as ner_base
from module.extractor.llm import llm
from module.extractor import regex
from module.extractor import ocr


def ocr_extract(**kwargs):
    return ocr.ocr_extract(**kwargs)


def regex_extract(text):
    return regex.regex_extract(text)


def ner_predict(**kwargs):
    return ner_base.ner_predict(**kwargs)


def ner_train(**kwargs):
    return ner_base.ner_train(**kwargs)


def train_metadata(**overrides):
    """배포 NER 모델 재학습 — ``train.py`` 의 단일 진입점.

    ``configs/labels.yaml::ner.model_name`` 백본을 ``configs/integrated/silver``
    (654k BIO) + ``configs/integrated/silver_aug`` (지자체→address, cue+기관→company
    증강) 로 full fine-tune 한다. 기본값은 현재 배포 모델(xlm-roberta-base)을 재현하는
    설정. 학습 산출물은 gitignore 대상이라, 새 환경에서는 ``python train.py`` 로 학습한
    뒤 ``python main.py`` 로 예측한다.

    ``force=False`` (기본): silver 서명이 같고 어댑터가 이미 있으면 스킵.
    하이퍼파라미터는 keyword 로 override 가능 (예: ``train_metadata(epochs=5)``).
    """
    from module.extractor.ner._runtime import load_ner_defaults
    model_name = load_ner_defaults().get("model_name") or "FacebookAI/xlm-roberta-base"
    cfg = dict(
        model_name=model_name,
        input_path="configs/integrated/silver",
        extra_input_paths=["configs/integrated/silver_aug"],
        fine_tuning_method="full",
        epochs=3,
        batch_size=32,
        lr=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_per_label=10000,
        split_seed=42,
    )
    cfg.update(overrides)
    return ner_train(**cfg)


def llm_extract(*args, **kwargs):
    return llm.llm_extract(*args, **kwargs)


def metadata_extract(**kwargs):
    """End-to-end: OCR → NER (regex + 후처리 + LLM placeholder 포함) → 35-라벨 JSON.

    인자는 ocr_extract / ner_predict 가 받는 키를 그대로 전달.
    OCR 캐시 (``ocr_output_path/result/``) 가 입력 문서 수만큼 있으면 OCR 단계 자동 스킵.
    ``input_text`` 또는 NER용 ``input_path`` 를 직접 넘기면 OCR 단계는 건너뛴다.
    OCR 입력을 직접 지정하려면 ``in_path`` 를 사용한다.
    """
    ocr_kwargs = _ocr_args(kwargs)
    should_run_ocr = bool(ocr_kwargs) or (
        "input_text" not in kwargs and "input_path" not in kwargs
    )
    if should_run_ocr:
        ocr_extract(**ocr_kwargs)
    return ner_predict(**_ner_args(kwargs))
    # llm_extract(...)  # 미구현 — 9 위임 라벨 placeholder 는 ner_predict 내부에서 처리


def extract_metadata(**kwargs):
    """Public API alias for the end-to-end metadata extraction pipeline."""
    return metadata_extract(**kwargs)


# ---------------------------------------------------------------------------
# 인자 라우팅 (각 stage 가 받는 키만 선별)
# ---------------------------------------------------------------------------

_OCR_KEYS = ("in_path", "out_path", "metadata_path", "device")
_NER_KEYS = (
    "model_name", "model_path", "input_path", "input_text", "output_path",
    "threshold", "thresholds", "result_phase", "llm_fn", "log_adapter_status",
    "debug", "debug_path",
)


def _ocr_args(kw):
    return {k: kw[k] for k in _OCR_KEYS if k in kw}


def _ner_args(kw):
    return {k: kw[k] for k in _NER_KEYS if k in kw}
