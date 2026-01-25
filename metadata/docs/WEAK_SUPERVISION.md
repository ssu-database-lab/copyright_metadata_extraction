# Weak-Supervision 시스템 사용 가이드

## 개요

Weak-Supervision 시스템은 `label_data.json`에 저장된 예시 문장/단어를 기반으로 자동 라벨링을 수행하는 시스템입니다. Online learning을 통해 `label_data.json`이 업데이트되면 자동으로 모델이 갱신됩니다.

## 파일 구조

- `configs/label_data.json`: 예시 문장/단어와 라벨 매핑 저장
- `data/out/label_pg_data.json`: 예측 결과 저장
- `configs/labels.yaml`: 라벨 스키마 정의 (기존)

## label_data.json 구조

간단한 구조로 각 라벨에 대해 문장 또는 단어 리스트를 저장합니다:

```json
{
  "work_title": [
    "저작물명: 전통민요 모음집",
    "저작물제목: 경상북도 민요",
    "제목:",
    "작품명:"
  ],
  "copyright_holder": [
    "저작권자: 홍길동",
    "원본소유자: 경상북도청",
    "저작자:"
  ],
  "production_date": [
    "제작일: 2024년 1월 15일",
    "촬영년도: 2023",
    "발행일:"
  ]
}
```

각 라벨은 문자열 리스트로 구성되며, 문장 전체 또는 키워드만 포함할 수 있습니다.

## 사용 방법

### 1. 기본 예측

```python
from module import api

# 텍스트에 대해 Weak-Supervision 라벨링 수행
text = "저작물명: 전통민요 모음집\n저작권자: 홍길동"
predictions = api.weak_supervision_predict(text)
print(predictions)
```

### 2. OCR 결과와 연동

```python
from module import api

# OCR 결과(metadata 디렉토리)에서 자동으로 라벨링 수행
api.weak_supervision_predict_from_ocr(
    ocr_metadata_dir="data/out/ocr/metadata",
    save_to_pg=True
)
```

### 3. label_data.json에 예시 추가

```python
from module import api

# 새로운 예시 추가 (간단하게)
api.weak_supervision_update_label_data(
    label="work_title",
    example_text="저작물제목: 경상북도 민요"
)
```

### 4. Online Learning

```python
from module import api

# label_data.json 변경 감지 시 자동 갱신
updated = api.weak_supervision_train_online()
if updated:
    print("모델이 갱신되었습니다.")
```

## Labeling Functions

시스템은 다음 3가지 유형의 labeling function을 사용합니다:

1. **키워드 기반**: 특정 키워드가 포함된 텍스트를 라벨링 (confidence: 0.7)
2. **패턴 기반**: 정규표현식 패턴으로 매칭 (confidence: 0.9)
3. **예시 기반**: 예시 문장과의 유사도로 판단 (confidence: 0.6-0.8)

## 예측 결과 저장

예측 결과는 `label_pg_data.json`에 다음과 같은 형식으로 저장됩니다:

```json
{
  "version": "1.0.0",
  "predictions": [
    {
      "timestamp": "2024-01-15T10:00:00",
      "source_file": "data/out/ocr/metadata/doc001.json",
      "predictions": {
        "work_title": [
          {
            "text": "저작물명: 전통민요 모음집",
            "label": "work_title",
            "confidence": 0.9,
            "source": "weak_supervision"
          }
        ]
      }
    }
  ]
}
```

## Online Learning 워크플로우

1. `label_data.json`에 새로운 예시 추가
2. `weak_supervision_train_online()` 호출 또는 다음 예측 시 자동 감지
3. Labeling functions 재컴파일
4. 이후 예측에 새 예시 반영

## 통합 워크플로우 예시

```python
from module import api

# 1. OCR 수행
api.ocr_extract("data/in/document", "data/out/ocr")

# 2. Weak-Supervision 라벨링 (자동으로 online learning 체크)
api.weak_supervision_predict_from_ocr("data/out/ocr/metadata")

# 3. 필요시 label_data.json 수동 업데이트
api.weak_supervision_update_label_data(
    label="copyright_holder",
    example_text="저작권자: 김철수",
    confidence=1.0
)

# 4. 다음 예측 시 자동으로 갱신된 모델 사용
```

