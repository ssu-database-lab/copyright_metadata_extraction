# NER Evaluate Function Update Summary

## 개요
`ner_evaluate` 함수를 `ner_predict`의 `debug` 파라미터 통합 및 `model_evaluation_log.txt` 형식 출력으로 업데이트했습니다.

## 주요 변경 사항

### 1. 함수 시그니처 변경

**이전:**
```python
def ner_evaluate(
    test_data_path: Optional[str] = None,
    model_name: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
    ...
) -> Dict[str, Any]:
```

**변경 후:**
```python
def ner_evaluate(
    output_path: str,                      # 필수 인자로 변경 (첫 번째 위치)
    model_name: Optional[str] = None,      # 두 번째 위치 (기본값: None)
    test_data_path: Optional[str] = None,  # 세 번째 위치
    verbose: bool = True,                   # 기본값 True로 변경 (debug 통합)
    debug: bool = False,
    ...
) -> Dict[str, Any]:
```

### 2. ner_test.py 스타일 인터페이스 지원

이제 `ner_test.py`와 동일한 방식으로 사용할 수 있습니다:

```python
# 기본 모델 사용 (klue-roberta-large 또는 model_config.json의 default_model)
ner_evaluate("data/out")

# 특정 모델 지정
ner_evaluate("data/out", "FacebookAI/xlm-roberta-large")
ner_evaluate("data/out", "google-bert/bert-base-multilingual-cased")
```

### 3. debug 파라미터 통합

- `ner_predict`의 `debug` 파라미터 기능을 `ner_evaluate`에 통합
- `debug=True`이면 `verbose`도 자동으로 `True`로 설정
- 상세한 로그 출력 제공

```python
# 상세 로그 출력
ner_evaluate("data/out", debug=True)
```

### 4. 출력 파일 형식

**저장 위치:**
```
{output_path}/debug/{model_name}_evaluation.txt
```

**파일명 예시:**
- `klue-roberta-large_evaluation.txt`
- `FacebookAI-xlm-roberta-large_evaluation.txt` (슬래시는 하이픈으로 변환)
- `google-bert-bert-base-multilingual-cased_evaluation.txt`

**파일 형식 (model_evaluation_log.txt와 동일):**
```
================================================================================
NER Model Evaluation
================================================================================
Start: 2025-10-17 14:23:45.123456
================================================================================


klue-roberta-large
================================================================================
Time: 12.5s
Samples: 1,000
Precision: 95.24%
Recall: 94.56%
F1 Score: 94.90%

Complete: 2025-10-17 14:23:57.654321
================================================================================
```

## 모델 이름 처리

### 기본 모델
- `model_name=None`인 경우 `ner_predict`와 동일한 기본 모델 사용
- `model_config.json`의 `ner.default_model` 설정 사용
- 기본값: `"klue-roberta-large"`

### 모델명 자동 처리
```python
# Hugging Face 모델명 → 파일시스템 안전 이름
"klue/roberta-large" → "klue-roberta-large"
"FacebookAI/xlm-roberta-large" → "FacebookAI-xlm-roberta-large"
"google-bert/bert-base-multilingual-cased" → "google-bert-bert-base-multilingual-cased"
```

## 사용 예제

### 1. 기본 사용법
```python
from api import ner_evaluate

# 기본 모델로 평가
result = ner_evaluate("data/out")
# 저장: data/out/debug/klue-roberta-large_evaluation.txt
```

### 2. 여러 모델 평가
```python
# 여러 모델을 순차적으로 평가
models = [
    None,  # 기본 모델
    "FacebookAI/xlm-roberta-large",
    "google-bert/bert-base-multilingual-cased"
]

for model in models:
    if model is None:
        result = ner_evaluate("data/out")
    else:
        result = ner_evaluate("data/out", model)
    
    print(f"모델: {result['model_name']}")
    print(f"F1 Score: {result['overall']['f1_score']:.2f}%")
    print("-" * 60)
```

### 3. 디버그 모드
```python
# 상세 로그와 함께 평가
result = ner_evaluate("data/out", debug=True)
```

### 4. 커스텀 테스트 데이터
```python
# 특정 테스트 데이터 사용
result = ner_evaluate(
    output_path="data/out",
    model_name="klue/roberta-large",
    test_data_path="custom_test_data.txt"
)
```

## 출력 예제

### 콘솔 출력
```
================================================================================
NER 모델 성능 평가
================================================================================
✓ 사용 모델: klue-roberta-large
✓ 평가 타입: Test (최종 평가)
✓ 테스트 데이터: c:\Users\...\test.txt
✓ 결과 저장: data/out

✓ 테스트 문장 수: 1,000
✓ 테스트 토큰 수: 15,234
✓ 모델 로드 완료 (출처: local)
예측 수행 중...
100%|████████████████████████████████████| 1000/1000 [00:12<00:00, 80.5it/s]

================================================================================
klue-roberta-large 평가 결과
================================================================================
평가 시간: 12.5s
샘플 수: 1,000
Precision: 95.24%
Recall: 94.56%
F1 Score: 94.90%

✓ 평가 결과 저장: klue-roberta-large_evaluation.txt
⏱️  평가 시간: 12.50초
================================================================================
```

### 파일 출력 (data/out/debug/klue-roberta-large_evaluation.txt)
```
================================================================================
NER Model Evaluation
================================================================================
Start: 2025-10-17 14:23:45.123456
================================================================================


klue-roberta-large
================================================================================
Time: 12.5s
Samples: 1,000
Precision: 95.24%
Recall: 94.56%
F1 Score: 94.90%

Complete: 2025-10-17 14:23:57.654321
================================================================================
```

## 반환값

```python
{
    "success": True,
    "model_name": "klue-roberta-large",
    "test_data_path": "c:\\Users\\...\\test.txt",
    "overall": {
        "precision": 95.24,
        "recall": 94.56,
        "f1_score": 94.90,
        "total_tokens": 15234
    },
    "entity_metrics": {
        "NAME": {
            "precision": 98.12,
            "recall": 97.45,
            "f1_score": 97.78,
            "support": 523
        },
        "PHONE": {
            "precision": 99.23,
            "recall": 98.67,
            "f1_score": 98.95,
            "support": 412
        },
        # ... 기타 엔티티 타입들
    },
    "evaluation_time": 12.5
}
```

## 호환성

### ner_predict와의 일관성
- 동일한 기본 모델 사용 (`DEFAULT_MODEL_NAME`)
- 동일한 모델 경로 구조 (`api/models/ner/{model_name}/`)
- 동일한 파라미터 처리 방식 (`debug`, `verbose`)

### ner_test.py와의 일관성
```python
# ner_test.py에서 사용하는 방식 그대로 지원
ner_evaluate("data/out")
ner_evaluate("data/out", "FacebookAI/xlm-roberta-large")
ner_evaluate("data/out", "google-bert/bert-base-multilingual-cased")
```

## 테스트 데이터 자동 탐색

`test_data_path`가 `None`인 경우 자동으로 다음 경로에서 테스트 데이터 탐색:
```
api/module/ner/training/{model_name}/test.txt
```

예시:
- `klue-roberta-large` → `api/module/ner/training/klue-roberta-large/test.txt`
- `FacebookAI-xlm-roberta-large` → `api/module/ner/training/FacebookAI-xlm-roberta-large/test.txt`

## 주의 사항

1. **모델이 없는 경우**: 
   - 로컬 모델이 없으면 Hugging Face에서 자동 로드 시도
   - 평가 전에 `ner_predict(..., train=True)` 또는 `ner_train()`으로 먼저 모델 준비 필요

2. **테스트 데이터 필요**:
   - BIO 포맷의 테스트 데이터 필요 (형식: `token\tlabel`)
   - 자동 생성된 `test.txt` 파일 사용 권장

3. **출력 디렉토리**:
   - `{output_path}/debug/` 디렉토리에 저장
   - 기존 파일은 덮어씀 (이름이 같으면)

## 마이그레이션 가이드

### 기존 코드
```python
# 이전 방식
ner_evaluate(
    test_data_path="test.txt",
    model_name="klue/roberta-large",
    output_path="results/",
    verbose=True
)
```

### 새로운 코드
```python
# 새로운 방식 (권장)
ner_evaluate("results/", "klue/roberta-large")

# 또는 명시적으로
ner_evaluate(
    output_path="results/",
    model_name="klue/roberta-large",
    test_data_path="test.txt",
    verbose=True
)
```

## 추가 기능

- ✅ Seqeval을 사용한 엔티티 레벨 평가 (설치 안 되어 있으면 자동 설치 시도)
- ✅ 엔티티 타입별 상세 메트릭 (Precision, Recall, F1)
- ✅ 진행 상황 프로그레스 바 (tqdm)
- ✅ GPU 자동 감지 및 사용
- ✅ Validation/Test 데이터 선택 옵션

## 파일 변경 내역

### 변경된 파일
- `api/module/ner/ner_system.py`: `ner_evaluate()` 함수 업데이트

### 주요 변경
1. 함수 시그니처 변경 (파라미터 순서 및 기본값)
2. `debug` 파라미터 통합
3. 출력 파일 형식 변경 (`model_evaluation_log.txt` 스타일)
4. 모델명 자동 처리 (슬래시 → 하이픈)
5. `ner_test.py` 스타일 인터페이스 지원

---

**업데이트 날짜**: 2025-10-17  
**버전**: 1.0.0  
**작성자**: GitHub Copilot
