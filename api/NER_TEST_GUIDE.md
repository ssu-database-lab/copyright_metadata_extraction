# NER 테스트 실행 가이드

## 빠른 시작

### 1. 단일 모델 테스트 (5 에포크)
```bash
cd c:\Users\peppermint\Desktop\copyright_metadata_extraction\api
python -c "from ner_test import compare_models; compare_models('google-bert/bert-base-multilingual-cased', epochs=5)"
```

### 2. 전체 테스트 (20 에포크)
```bash
python ner_test.py
```

### 3. 특정 모델만 테스트
```python
# Python에서 실행
from ner_test import compare_models

# BERT Base
compare_models('google-bert/bert-base-multilingual-cased', epochs=20)

# KLUE RoBERTa Large
compare_models('klue/roberta-large', epochs=20, batch_size_single=16, batch_size_system=8)

# XLM-RoBERTa Large  
compare_models('FacebookAI/xlm-roberta-large', epochs=20, batch_size_single=16, batch_size_system=8)
```

## 모델별 권장 설정

### BERT Base (hidden_size=768)
```python
compare_models(
    model_name='google-bert/bert-base-multilingual-cased',
    epochs=20,
    batch_size_single=32,    # 순수 모델
    batch_size_system=12,    # 시스템 모델
    learning_rate=2e-5
)
```
- GPU 메모리: ~4GB
- 학습 시간: ~30-40분 (20 epochs)

### KLUE RoBERTa Large (hidden_size=1024)
```python
compare_models(
    model_name='klue/roberta-large',
    epochs=20,
    batch_size_single=16,    # 메모리 절약
    batch_size_system=8,     # 메모리 절약
    learning_rate=2e-5
)
```
- GPU 메모리: ~8GB
- 학습 시간: ~60-80분 (20 epochs)

### XLM-RoBERTa Large (hidden_size=1024)
```python
compare_models(
    model_name='FacebookAI/xlm-roberta-large',
    epochs=20,
    batch_size_single=16,
    batch_size_system=8,
    learning_rate=2e-5
)
```
- GPU 메모리: ~8GB
- 학습 시간: ~60-80분 (20 epochs)

## 메모리 부족 시 대처

### OOM (Out of Memory) 에러 발생 시
```python
# 배치 크기를 더 줄이기
compare_models(
    model_name='klue/roberta-large',
    epochs=20,
    batch_size_single=8,     # 16 → 8
    batch_size_system=4,     # 8 → 4
    learning_rate=2e-5
)

# 또는 gradient accumulation 사용
# ner_train.py의 TrainingArguments에서:
# gradient_accumulation_steps=2
```

### 메모리 최적화 옵션
```python
# ner_train.py Config 클래스 수정:
Config(
    batch_size=8,                    # 기본값 16 → 8
    eval_batch_size=16,              # 기본값 32 → 16
    max_length=256,                  # 기본값 256 (또는 128로 줄이기)
    gradient_accumulation_steps=2,   # 추가 (효과적 배치=8*2=16)
)
```

## 결과 확인

### 1. 리포트 파일 위치
```
data/out/ner_visualization/
├── pure_model_train_report_google-bert_bert-base-multilingual-cased.txt
├── pure_model_validate_report_google-bert_bert-base-multilingual-cased.txt
├── system_model_train_report_google-bert_bert-base-multilingual-cased.txt
├── system_model_validate_report_google-bert_bert-base-multilingual-cased.txt
├── comparison_metrics_google-bert_bert-base-multilingual-cased.png
├── comparison_training_google-bert_bert-base-multilingual-cased.png
├── ner_comparison_results_google-bert_bert-base-multilingual-cased.json
└── ner_comparison_report_google-bert_bert-base-multilingual-cased.txt
```

### 2. 모델 파일 위치
```
models/ner/
├── google-bert-bert-base-multilingual-cased/
│   ├── config.json
│   ├── model.pt                      # BERT-CRF 가중치
│   ├── model_architecture.json       # 아키텍처 정보 (새로 추가)
│   ├── label_map.json                # 라벨 맵
│   ├── training_history.json         # 학습 히스토리
│   ├── pytorch_model.bin             # BERT만 (HuggingFace 호환)
│   ├── tokenizer_config.json
│   └── ...
├── klue-roberta-large/
└── FacebookAI-xlm-roberta-large/
```

### 3. 주요 메트릭 확인
```bash
# 텍스트 리포트에서 확인
cat data/out/ner_visualization/ner_comparison_report_*.txt

# JSON에서 확인
python -c "import json; print(json.dumps(json.load(open('data/out/ner_visualization/ner_comparison_results_google-bert_bert-base-multilingual-cased.json')), indent=2))"
```

## 문제 해결

### 차원 불일치 에러
```
경고: BERT-CRF 모델 로드 실패, HuggingFace 표준 모델 사용
     오류: 차원 불일치: checkpoint=1024, model=768
```

**해결 방법**:
1. 자동 fallback 동작 확인 (정상)
2. 재학습하면 `model_architecture.json` 생성되어 해결
3. 또는 기존 `model.pt` 삭제 후 재학습

### 파일 없음 에러
```
FileNotFoundError: [Errno 2] No such file or directory
```

**해결 방법**:
- 디렉토리 자동 생성이 이미 적용되어 있으므로 재실행
- 수동 생성: `mkdir -p data/out/ner_visualization`

### Import 에러
```
ModuleNotFoundError: No module named 'torchcrf'
```

**해결 방법**:
```bash
pip install torchcrf
# 또는
pip install -r requirements.txt
```

## 성능 벤치마크

### 예상 F1 Score (20 epochs, 3개 train samples, 7개 val samples)
- **순수 BERT**: 0.00-0.10 (매우 작은 데이터셋)
- **BERT-CRF**: 0.20-0.35 (시스템 모델, 개선됨)

### 실제 데이터로 학습 시 (권장)
```python
compare_models(
    model_name='google-bert/bert-base-multilingual-cased',
    epochs=50,
    dataset_size=10000,      # 더 많은 데이터
    batch_size_single=32,
    batch_size_system=12
)
```
- 예상 F1: 0.70-0.85

## 추가 옵션

### 시각화 비활성화
```python
compare_models(
    model_name='...',
    plot=False  # 시각화 생성 안 함
)
```

### Early Stopping 활성화
```python
# ner_train.py Config에서:
enable_early_stopping = True  # 5 epoch 개선 없으면 중단
```

### 데이터 재생성
```python
compare_models(
    model_name='...',
    force_regenerate_data=True  # BIO 데이터 재생성
)
```

## 참고 사항

- **CPU 모드**: GPU 없어도 작동하지만 매우 느림 (10배 이상)
- **혼합 정밀도**: `fp16=True` 기본 활성화 (GPU 메모리 절약)
- **체크포인트**: 최대 3개 유지 (`save_total_limit=3`)
- **로그**: 터미널 출력 + 파일 저장

## 문의

- 문제 발생: GitHub Issues
- 긴급: 팀 채널
