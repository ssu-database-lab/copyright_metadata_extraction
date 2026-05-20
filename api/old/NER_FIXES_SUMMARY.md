# NER 모델 문제점 수정 요약

## 수정 완료된 항목

### 1. 출력 디렉토리 통일 ✅
**문제**: `data/out/results` 디렉토리를 사용하고 있었으나, 실제 프로젝트 구조는 `data/out/ner_visualization`을 사용

**수정사항**:
- `ner_test.py`의 모든 출력 경로를 `data/out/ner_visualization`로 변경
- 디렉토리 생성 코드 추가: `report_path.parent.mkdir(parents=True, exist_ok=True)`

**영향받는 파일**:
- `api/ner_test.py` (7개 위치 수정)

### 2. 모델 차원 불일치 처리 ✅
**문제**: klue/roberta-large (hidden_size=1024)와 bert-base (hidden_size=768) 간의 차원 불일치

**현재 상태**:
- `module/ner/ner_system.py`의 `load_model_and_tokenizer` 함수에서 이미 처리됨
- 차원 불일치 감지 시 자동으로 HuggingFace 표준 모델로 fallback
- 에러 메시지 개선됨

**코드 위치**: `api/module/ner/ner_system.py` lines 220-260

## 추가 권장 수정사항

### 3. 모델 아키텍처 정보 저장
**문제**: 모델을 저장할 때 아키텍처 정보(hidden_size, use_lstm 등)를 함께 저장하지 않아서 로드 시 정확한 아키텍처 재현이 어려움

**권장 수정**:
`module/ner/ner_train.py`의 모델 저장 부분(line ~3395)에 아키텍처 정보 추가:

```python
# 모델 아키텍처 정보 저장
model_architecture = {
    "model_type": "BertCrfForNER",
    "model_name": config.model_name,
    "hidden_size": model.bert.config.hidden_size,
    "num_labels": len(BIO_LABELS),
    "dropout": config.dropout,
    "use_lstm": True,
    "lstm_hidden_dim": 256,
    "lstm_layers": 3
}
save_json(model_architecture, output_dir_path / "model_architecture.json")
```

### 4. 시각화 파일 저장 개선
**현재 문제**: 일부 시각화가 저장되지 않거나 경로가 일관되지 않음

**권장 수정**:
- 모든 시각화 저장 전에 디렉토리 존재 확인
- 파일명 규칙 통일: `{model_safe_name}_{visualization_type}.png`

### 5. 로그 출력 개선
**문제**: 너무 많은 체크포인트가 저장되어 디스크 공간 낭비

**권장 수정**:
`module/ner/ner_train.py`의 TrainingArguments에서:
```python
save_strategy="epoch",  # 현재
save_total_limit=3,     # 현재 - 좋음
# 추가 권장:
save_only_model=True,   # optimizer 상태 저장 안 함
```

## 실행 시 주의사항

### 모델별 권장 배치 크기
- **google-bert/bert-base-multilingual-cased**: 
  - 순수 모델: 32
  - 시스템 모델: 12

- **klue/roberta-large** (hidden_size=1024, 파라미터 수 많음):
  - 순수 모델: 16 (32는 OOM 위험)
  - 시스템 모델: 8 (12는 OOM 위험)

- **FacebookAI/xlm-roberta-large** (hidden_size=1024):
  - 순수 모델: 16
  - 시스템 모델: 8

### GPU 메모리 관리
- Large 모델 학습 시 `fp16=True` 권장 (이미 적용됨)
- 메모리 부족 시: `gradient_accumulation_steps=2` 추가

## 테스트 실행 방법

```bash
# 단일 모델 테스트
cd api
python ner_test.py

# 또는 직접 함수 호출
python -c "from ner_test import compare_models; compare_models('google-bert/bert-base-multilingual-cased', epochs=5)"
```

## 파일 구조 확인

```
api/
├── data/
│   └── out/
│       ├── ner/                    # NER 학습 중간 결과
│       ├── ner_visualization/      # ✅ 모든 리포트와 시각화 저장 위치
│       ├── ocr/                    # OCR 관련
│       └── pdf_convert/            # PDF 변환 관련
├── models/
│   └── ner/                        # 훈련된 NER 모델 저장
│       ├── google-bert-bert-base-multilingual-cased/
│       ├── klue-roberta-large/
│       └── FacebookAI-xlm-roberta-large/
└── module/
    └── ner/
        ├── ner_train.py           # 모델 훈련
        ├── ner_system.py          # 모델 로드 및 예측
        ├── ner_evaluate.py        # 모델 평가
        └── training/              # BIO 학습 데이터
```

## 디버깅 팁

### 차원 불일치 에러 발생 시
```
경고: BERT-CRF 모델 로드 실패, HuggingFace 표준 모델 사용
     오류: 차원 불일치: checkpoint=1024, model=768
```

**해결 방법**:
1. 모델 디렉토리에서 `model.pt` 파일 삭제
2. 모델을 처음부터 재학습하거나
3. HuggingFace 표준 모델로 fallback (자동)

### 메모리 부족 (OOM) 에러
```
CUDA out of memory
```

**해결 방법**:
1. 배치 크기 줄이기: `batch_size=16` → `batch_size=8`
2. 시퀀스 길이 줄이기: `max_length=512` → `max_length=256`
3. Gradient accumulation 사용

## 수정 완료 체크리스트

- [x] 출력 디렉토리를 `data/out/ner_visualization`로 통일
- [x] 모든 파일 저장 전 디렉토리 생성 확인
- [x] 모델 차원 불일치 처리 확인
- [x] 에러 메시지 개선
- [ ] 모델 아키텍처 정보 저장 (권장)
- [ ] 배치 크기 자동 조정 (권장)

## 참고 자료

- HuggingFace Transformers 문서: https://huggingface.co/docs/transformers
- PyTorch CRF 라이브러리: https://pytorch-crf.readthedocs.io/
- 프로젝트 wiki (있다면 링크 추가)
