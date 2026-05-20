"""공개 API.

- ocr_extract       : module.extractor.ocr        → PDF/이미지 → 텍스트
- regex_extract     : module.extractor.regex      → 9 strict-format 라벨
- ner_predict       : module.extractor.ner.base   → 35-라벨 메타데이터 (regex+NER+후처리+LLM placeholder)
- ner_train         : module.extractor.ner.base   → NER 학습
- llm_extract       : module.extractor.llm.llm    → 9 위임 라벨 (미구현)
- metadata_extract  : ★ OCR + NER + LLM end-to-end 오케스트레이터
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


def llm_extract(*args, **kwargs):
    return llm.llm_extract(*args, **kwargs)


def metadata_extract(**kwargs):
    """End-to-end: OCR → NER (regex + 후처리 + LLM placeholder 포함) → 35-라벨 JSON.

    인자는 ocr_extract / ner_predict 가 받는 키를 그대로 전달.
    OCR 캐시 (``ocr_output_path/result/``) 가 입력 문서 수만큼 있으면 OCR 단계 자동 스킵.
    """
    ocr_extract(**_ocr_args(kwargs))
    ner_predict(**_ner_args(kwargs))
    # llm_extract(...)  # 미구현 — 9 위임 라벨 placeholder 는 ner_predict 내부에서 처리


# ---------------------------------------------------------------------------
# 인자 라우팅 (각 stage 가 받는 키만 선별)
# ---------------------------------------------------------------------------

_OCR_KEYS = ("in_path", "out_path", "metadata_path", "device")
_NER_KEYS = (
    "model_name", "model_path", "input_path", "input_text", "output_path",
    "threshold", "thresholds", "result_phase", "log_adapter_status",
    "debug", "debug_path",
)


def _ocr_args(kw):
    return {k: kw[k] for k in _OCR_KEYS if k in kw}


def _ner_args(kw):
    return {k: kw[k] for k in _NER_KEYS if k in kw}
