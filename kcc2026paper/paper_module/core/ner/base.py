"""Compat shim — base.py 본체는 module/extractor/ner/ 로 이동했다.

이 모듈은 paper{1..6}.py / ner_robust-distillbert.py 등 기존 임포트를 끊지
않기 위한 re-export 레이어. **신규 코드는 module/extractor/ner/ 를 직접 import** 할 것.

2026-06 metadata 리팩터(base.py → _runtime.py 분리)에 맞춰, 공개 진입점
(ner_predict/ner_train)은 slim `base` 에서, 나머지 런타임 surface 는 `_runtime`
에서 가져온다. (kcc2026paper 로 분리하며 현행 metadata API 에 재정합.)
"""
from module.extractor.ner.base import ner_predict, ner_train  # noqa: F401
from module.extractor.ner import _runtime as _rt

# 현행 metadata 리팩터 대응: _runtime 의 모든 공개 심볼(DEFAULT_MODEL,
# detect_model_type, model_display_name, get_model_dir, ner_predict_at_thresholds …)
# 을 base 네임스페이스로 재노출하여 옛 `from paper_module.core.ner.base import X`
# 임포트를 그대로 유지한다.
for _name in dir(_rt):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_rt, _name)
del _name
