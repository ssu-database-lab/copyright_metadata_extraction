# NER 시스템 수정 완료 보고서

## 수정 날짜
2025-11-25

## 수정 사항 요약

### 1. 출력 디렉토리 통일 ✅
**변경 파일**: `api/ner_test.py`

**수정 내용**:
- 모든 출력 경로를 `data/out/results` → `data/out/ner_visualization`로 변경
- 총 7개 위치 수정
- 각 파일 저장 전 디렉토리 자동 생성 추가

**변경된 위치**:
```python
# 1. 순수 모델 훈련 리포트
report_path = Path("data/out/ner_visualization/pure_model_train_report_{model_safe_name}.txt")
report_path.parent.mkdir(parents=True, exist_ok=True)

# 2. 순수 모델 검증 리포트
report_path = Path("data/out/ner_visualization/pure_model_validate_report_{model_name}.txt")

# 3. 시스템 모델 훈련 리포트  
report_path = Path("data/out/ner_visualization/system_model_train_report_{model_safe_name}.txt")

# 4. 시스템 모델 검증 리포트
report_path = Path("data/out/ner_visualization/system_model_validate_report_{model_safe_name}.txt")

# 5-7. 비교 결과 디렉토리
output_dir = Path("data/out/ner_visualization")
```

### 2. 모델 아키텍처 정보 저장 추가 ✅
**변경 파일**: `api/module/ner/ner_train.py` (line ~3395)

**추가 내용**:
```python
# 모델 아키텍처 정보 저장 (모델 로딩 시 정확한 아키텍처 재현을 위해)
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

**효과**:
- 모델 로딩 시 정확한 hidden_size 정보 활용
- klue/roberta-large (1024) vs bert-base (768) 차원 불일치 문제 해결
- 재현 가능한 모델 아키텍처 보장

### 3. 모델 로딩 로직 개선 ✅
**변경 파일**: `api/module/ner/ner_system.py` (line ~210)

**개선 내용**:
- `model_architecture.json` 파일 우선 로드
- 저장된 아키텍처 정보로 정확한 모델 인스턴스 생성
- Fallback 메커니즘 유지 (3단계):
  1. BERT-CRF 모델 with architecture.json
  2. HuggingFace 표준 모델 (local)
  3. HuggingFace 원본 모델 다운로드

**코드 구조**:
```python
# 1단계: model_architecture.json 확인
arch_path = model_path / "model_architecture.json"
if arch_path.exists():
    # 저장된 설정으로 정확한 모델 생성
    arch_config = json.load(...)
    model = BertCrfForNER(
        model_name=arch_config['model_name'],
        dropout=arch_config['dropout'],
        use_lstm=arch_config['use_lstm'],
        lstm_hidden_dim=arch_config['lstm_hidden_dim'],
        lstm_layers=arch_config['lstm_layers']
    )
    # 가중치 로드
    model.load_state_dict(state_dict)
```

## 해결된 문제들

### 문제 1: 출력 파일 경로 불일치
**증상**: 
- 파일이 `data/out/results`에 저장되었으나 실제 프로젝트 구조는 `data/out/ner_visualization` 사용
- 사용자가 결과 파일을 찾을 수 없음

**해결**:
- 모든 경로를 `data/out/ner_visualization`로 통일
- 디렉토리 자동 생성으로 에러 방지

### 문제 2: 모델 차원 불일치
**증상**:
```
경고: BERT-CRF 모델 로드 실패, HuggingFace 표준 모델 사용
     오류: 차원 불일치: checkpoint=1024, model=768
```

**원인**:
- klue/roberta-large는 hidden_size=1024
- bert-base는 hidden_size=768
- 저장된 모델 정보 없이 기본값(768)으로 로드 시도

**해결**:
1. 모델 저장 시 `model_architecture.json` 생성
2. 로딩 시 정확한 hidden_size로 모델 인스턴스 생성
3. 차원 불일치 시 자동 fallback

### 문제 3: 디렉토리 생성 에러
**증상**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/out/results/...'
```

**해결**:
```python
report_path.parent.mkdir(parents=True, exist_ok=True)
```

## 테스트 방법

### 1. 단일 모델 테스트
```bash
cd api
python ner_test.py

# 또는
python -c "from ner_test import compare_models; compare_models('google-bert/bert-base-multilingual-cased', epochs=5)"
```

### 2. 모델별 권장 설정
```python
# bert-base (hidden_size=768)
compare_models(
    model_name='google-bert/bert-base-multilingual-cased',
    epochs=20,
    batch_size_single=32,  # 순수 모델
    batch_size_system=12   # 시스템 모델 (CRF+BiLSTM)
)

# roberta-large (hidden_size=1024)
compare_models(
    model_name='klue/roberta-large',
    epochs=20,
    batch_size_single=16,  # 메모리 고려
    batch_size_system=8    # 메모리 고려
)
```

### 3. 결과 확인
```bash
# 생성된 파일 확인
ls data/out/ner_visualization/

# 예상 파일:
# - pure_model_train_report_*.txt
# - pure_model_validate_report_*.txt
# - system_model_train_report_*.txt
# - system_model_validate_report_*.txt
# - comparison_metrics_*.png
# - comparison_training_*.png
# - ner_comparison_results_*.json
# - ner_comparison_report_*.txt
```

## 파일 구조 (수정 후)

```
api/
├── data/
│   └── out/
│       ├── ner/                          # 학습 데이터
│       ├── ner_visualization/            # ✅ 모든 결과 파일
│       │   ├── *_train_report_*.txt
│       │   ├── *_validate_report_*.txt
│       │   ├── comparison_*.png
│       │   └── ner_comparison_*.json
│       ├── ocr/
│       └── pdf_convert/
├── models/
│   └── ner/
│       └── {model_name}/
│           ├── config.json
│           ├── model.pt
│           ├── model_architecture.json   # ✅ 새로 추가
│           ├── label_map.json
│           ├── training_history.json
│           └── ...
├── module/
│   └── ner/
│       ├── ner_train.py                  # ✅ 수정됨
│       ├── ner_system.py                 # ✅ 수정됨
│       └── ner_evaluate.py
├── ner_test.py                           # ✅ 수정됨
└── NER_FIXES_SUMMARY.md                  # ✅ 문서
```

## 변경된 파일 목록

1. ✅ `api/ner_test.py` - 출력 디렉토리 통일
2. ✅ `api/module/ner/ner_train.py` - 아키텍처 정보 저장
3. ✅ `api/module/ner/ner_system.py` - 모델 로딩 개선
4. ✅ `api/NER_FIXES_SUMMARY.md` - 수정 가이드

## 주의사항

### GPU 메모리 관리
- **Large 모델** (klue/roberta-large, xlm-roberta-large):
  - 순수 모델: batch_size=16 권장
  - 시스템 모델: batch_size=8 권장
  - 필요시 `gradient_accumulation_steps=2` 추가

### 기존 모델 재사용
- 이전에 학습된 모델은 `model_architecture.json`이 없을 수 있음
- 문제 없이 fallback 메커니즘으로 로드됨
- 재학습 시 자동으로 `model_architecture.json` 생성

### 디스크 공간
- 각 모델 체크포인트: ~1-3GB (모델 크기에 따라)
- `save_total_limit=3`으로 최대 3개 체크포인트 유지
- 불필요한 체크포인트 삭제 가능

## 검증 완료

- [x] 출력 디렉토리 통일
- [x] 파일 저장 전 디렉토리 생성
- [x] 모델 아키텍처 정보 저장
- [x] 차원 불일치 자동 처리
- [x] Fallback 메커니즘 동작
- [x] 에러 메시지 개선
- [x] 문서화 완료

## 후속 작업 (선택 사항)

1. **배치 크기 자동 조정**
   - 모델 크기에 따라 자동으로 배치 크기 조정
   - GPU 메모리 체크 및 OOM 방지

2. **시각화 개선**
   - 더 많은 메트릭 시각화
   - 모델 비교 대시보드

3. **학습 재개 기능**
   - 중단된 학습 자동 재개
   - 체크포인트 관리 개선

## 참고 자료

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PyTorch CRF](https://pytorch-crf.readthedocs.io/)
- [프로젝트 Wiki](링크가 있다면 추가)

## 문의
- 문제 발생 시: GitHub Issues 또는 팀 채널
- 수정 제안: Pull Request
