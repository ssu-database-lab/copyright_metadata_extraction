# NER 테스트 결과 파일 구조

## 개요

3개의 BERT 모델(google-bert/bert-base-multilingual-cased, klue/roberta-large, FacebookAI/xlm-roberta-large)에 대해 다음 2가지 구성으로 훈련 및 평가합니다:

1. **순수 모델 (Pure Model)**: BERT 모델만 사용 (AutoModelForTokenClassification)
2. **혼합 모델 (Mixed Model)**: BERT + BiLSTM + CRF Layers

## 파일 명명 규칙

### 순수 모델 (Pure Model)
- `pure_{model_name}_*.{확장자}`

### 혼합 모델 (Mixed Model)  
- `mixed_{model_name}_*.{확장자}`

## 각 모델당 생성되는 파일 (총 12개)

### 1. 훈련 리포트 (Training Report) - 2개
```
pure_{model_name}_train_report.txt      # 순수 모델 훈련 결과
mixed_{model_name}_train_report.txt     # 혼합 모델 훈련 결과
```

**내용**:
- 모델 타입 및 이름
- 훈련 완료 시간
- 에포크 수
- 학습/검증 샘플 수
- 배치 크기, 학습률
- 모델 경로
- 훈련 히스토리 (loss, accuracy 등)

### 2. 검증 리포트 (Validation Report) - 2개
```
pure_{model_name}_validate_report.txt   # 순수 모델 검증 결과
mixed_{model_name}_validate_report.txt  # 혼합 모델 검증 결과
```

**내용**:
- 모델 타입 및 이름
- 검증 완료 시간
- Precision, Recall, F1 Score
- TP, FP, FN
- 엔티티 타입별 성능

### 3. 종합 테스트 리포트 (Comprehensive Test) - 4개
```
pure_{model_name}_comprehensive.txt     # 순수 모델 종합 테스트 (텍스트)
pure_{model_name}_comprehensive.json    # 순수 모델 종합 테스트 (JSON)
mixed_{model_name}_comprehensive.txt    # 혼합 모델 종합 테스트 (텍스트)
mixed_{model_name}_comprehensive.json   # 혼합 모델 종합 테스트 (JSON)
```

**내용**:
- **[1/3] 테스트 결과**: 샘플 텍스트에 대한 예측 결과
- **[2/3] 성능 메트릭**: Precision, Recall, F1, 엔티티 타입별 성능
- **[3/3] 예측 결과**: 모든 샘플에 대한 상세 예측 결과

### 4. 시각화 파일 (Visualization) - 2개
```
pure_{model_name}_train.png             # 순수 모델 훈련 곡선 (선택사항)
pure_{model_name}_validate.png          # 순수 모델 검증 시각화 (선택사항)
```

### 5. 비교 결과 (Comparison) - 4개
```
comparison_{model_name}_metrics.png     # 메트릭 비교 시각화
comparison_{model_name}_training.png    # 훈련 곡선 비교
comparison_{model_name}_results.json    # 비교 결과 (JSON)
comparison_{model_name}_report.txt      # 비교 리포트 (텍스트)
```

**내용**:
- 순수 vs 혼합 모델 성능 비교
- Precision, Recall, F1 비교
- 개선율 계산
- 훈련 히스토리 비교

## 예제: google-bert/bert-base-multilingual-cased

```
data/out/ner_visualization/
├── pure_google-bert_bert-base-multilingual-cased_train_report.txt
├── pure_google-bert_bert-base-multilingual-cased_validate_report.txt
├── pure_google-bert_bert-base-multilingual-cased_comprehensive.txt
├── pure_google-bert_bert-base-multilingual-cased_comprehensive.json
├── pure_google-bert_bert-base-multilingual-cased_train.png
├── pure_google-bert_bert-base-multilingual-cased_validate.png
├── mixed_google-bert_bert-base-multilingual-cased_train_report.txt
├── mixed_google-bert_bert-base-multilingual-cased_validate_report.txt
├── mixed_google-bert_bert-base-multilingual-cased_comprehensive.txt
├── mixed_google-bert_bert-base-multilingual-cased_comprehensive.json
├── comparison_google-bert_bert-base-multilingual-cased_metrics.png
├── comparison_google-bert_bert-base-multilingual-cased_training.png
├── comparison_google-bert_bert-base-multilingual-cased_results.json
└── comparison_google-bert_bert-base-multilingual-cased_report.txt
```

## 전체 3개 모델에 대한 결과 (총 42개 파일)

### Google BERT Base (14개)
- 순수 모델: 6개 파일
- 혼합 모델: 4개 파일  
- 비교 결과: 4개 파일

### KLUE RoBERTa Large (14개)
- 순수 모델: 6개 파일
- 혼합 모델: 4개 파일
- 비교 결과: 4개 파일

### XLM-RoBERTa Large (14개)
- 순수 모델: 6개 파일
- 혼합 모델: 4개 파일
- 비교 결과: 4개 파일

## 파일 내용 상세

### 훈련 리포트 예시
```text
================================================================================
NER 모델 성능 리포트
================================================================================

모델 타입:
  순수 모델 (레이어 추가 없음)

모델명:
  google-bert/bert-base-multilingual-cased

훈련 완료 시간:
  2025-11-25T14:30:00

에포크:
  20

학습 샘플 수:
  30000

배치 크기:
  32

학습률:
  2e-05

모델 경로:
  models/ner_pure/google-bert-bert-base-multilingual-cased

히스토리:
  - loss: [3.8389, 3.8674, ...]
  - eval_loss: [3.8718, 3.5130, ...]
  - eval_f1: [0.0, 0.0, 0.1081, ...]
```

### 검증 리포트 예시
```text
================================================================================
NER 모델 성능 리포트
================================================================================

모델 타입:
  순수 모델 (레이어 추가 없음)

모델명:
  google-bert/bert-base-multilingual-cased

검증 완료 시간:
  2025-11-25T14:45:00

Precision:
  0.0833

Recall:
  0.1538

F1 Score:
  0.1081

전체 메트릭:
  - TP: 2
  - FP: 22
  - FN: 11
```

### 종합 테스트 JSON 예시
```json
{
  "model_name": "google-bert/bert-base-multilingual-cased",
  "model_type": "pure",
  "timestamp": "2025-11-25T14:50:00",
  "test_results": [
    {
      "sample_idx": 1,
      "input_text": "저작물 저작재산권 양도 계약서...",
      "predicted_entities": [
        ["서울시 강남구 테헤란로 123", "ADDRESS"],
        ["김민수", "NAME"],
        ["010-1234-5678", "PHONE"]
      ],
      "entity_count": 9
    }
  ],
  "performance_metrics": {
    "precision": 0.0833,
    "recall": 0.1538,
    "f1": 0.1081,
    "tp": 2,
    "fp": 22,
    "fn": 11
  },
  "prediction_results": [...]
}
```

### 비교 리포트 예시
```text
================================================================================
NER 모델 성능 리포트
================================================================================

model_name:
  google-bert/bert-base-multilingual-cased

timestamp:
  2025-11-25T15:00:00

pure_model:
  - train: {...}
  - validate: {...}
  - comprehensive: {...}

system_model:
  - train: {...}
  - validate: {...}
  - comprehensive: {...}

comparison:
  - validate:
    - pure:
      - precision: 0.0833
      - recall: 0.1538
      - f1: 0.1081
    - system:
      - precision: 0.2778
      - recall: 0.3846
      - f1: 0.3226
    - f1_diff: 0.2145
    - f1_improvement: 198.52%
```

## 확인 체크리스트

### 각 모델당 확인 항목 (12개 파일)

#### 순수 모델 (6개)
- [ ] `pure_{model}_train_report.txt` - 훈련 리포트
- [ ] `pure_{model}_validate_report.txt` - 검증 리포트
- [ ] `pure_{model}_comprehensive.txt` - 종합 테스트 (텍스트)
- [ ] `pure_{model}_comprehensive.json` - 종합 테스트 (JSON)
- [ ] `pure_{model}_train.png` - 훈련 시각화 (선택)
- [ ] `pure_{model}_validate.png` - 검증 시각화 (선택)

#### 혼합 모델 (4개)
- [ ] `mixed_{model}_train_report.txt` - 훈련 리포트
- [ ] `mixed_{model}_validate_report.txt` - 검증 리포트
- [ ] `mixed_{model}_comprehensive.txt` - 종합 테스트 (텍스트)
- [ ] `mixed_{model}_comprehensive.json` - 종합 테스트 (JSON)

#### 비교 결과 (4개)
- [ ] `comparison_{model}_metrics.png` - 메트릭 비교
- [ ] `comparison_{model}_training.png` - 훈련 비교
- [ ] `comparison_{model}_results.json` - 비교 결과 (JSON)
- [ ] `comparison_{model}_report.txt` - 비교 리포트 (텍스트)

## 파일 위치
```
copyright_metadata_extraction/
└── api/
    └── data/
        └── out/
            └── ner_visualization/
                ├── pure_*.txt
                ├── pure_*.json
                ├── pure_*.png
                ├── mixed_*.txt
                ├── mixed_*.json
                ├── comparison_*.png
                ├── comparison_*.json
                └── comparison_*.txt
```

## 실행 명령

```bash
# 단일 모델 테스트
cd api
python -c "from ner_test import compare_models; compare_models('google-bert/bert-base-multilingual-cased', epochs=20)"

# 전체 3개 모델 테스트
python ner_test.py
```

## 예상 실행 시간

- **BERT Base**: ~40분 (20 epochs)
- **RoBERTa Large**: ~80분 (20 epochs)
- **XLM-RoBERTa Large**: ~80분 (20 epochs)
- **전체**: ~3-4시간

## 주의사항

1. **메모리 부족**: Large 모델은 배치 크기를 줄여야 할 수 있음
2. **디스크 공간**: 각 모델당 ~2-4GB 필요
3. **GPU**: CUDA 사용 가능 시 자동으로 GPU 사용
4. **시각화**: `plot=False`로 시각화 파일 생성 비활성화 가능

## 문제 해결

### 파일이 생성되지 않는 경우
```bash
# 디렉토리 확인
ls data/out/ner_visualization/

# 수동 생성
mkdir -p data/out/ner_visualization
```

### 파일명이 다른 경우
- 코드가 업데이트되었는지 확인
- `ner_test.py` 최신 버전 사용 확인

### 일부 파일만 생성되는 경우
- 에러 로그 확인
- 메모리 부족 여부 확인
- 모델 경로 확인
