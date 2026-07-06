"""Production training — ``python train.py``.

단일 진입점: ``module.api.train_metadata`` 만 사용.
``configs/labels.yaml::ner.model_name`` 백본(기본 xlm-roberta-base)을
``configs/integrated/silver`` + ``configs/integrated/silver_aug`` 로 full fine-tune.

학습 산출물(``models/<id>/adapter/``)은 gitignore 대상 → 새 환경에서는
``python train.py`` (학습) 후 ``python main.py`` (예측) 순서로 실행한다.
이미 학습돼 있으면(silver 서명 동일 + 어댑터 존재) 자동 스킵.
강제 재학습: ``python train.py`` 대신 ``train_metadata(force=True)`` 호출.
"""
from module.api import train_metadata

if __name__ == "__main__":
    train_metadata()
