# NER Model Validation

이 디렉토리는 NER 모델 검증 및 테스트 관련 파일들을 포함합니다.

## 📁 파일 목록

### 메인 테스트 스크립트
- **test.py** (10KB)
  - 3개 모델 자동 다운로드, 훈련, 평가 전체 프로세스
  - Before/After Training 비교 (1-6번)
  - 로그 파일: `model_evaluation_log.txt`

### 디버깅 스크립트
- **debug_prediction.py** (3KB)
  - 단일 문장 예측 과정 상세 디버깅
  - 토큰화, offset mapping, 라벨 정렬 확인

- **debug_complex.py** (3.6KB)
  - 복잡한 문장(공백, 긴 엔티티) 예측 테스트
  - 케이스별 정확도 계산

- **debug_space.py** (1.4KB)
  - 공백 처리 방식 확인
  - BERT 토크나이저의 offset mapping 분석

- **debug_tokenization.py** (1KB)
  - 다양한 텍스트의 토큰화 패턴 확인
  - 서브워드 분리 방식 분석

### 로그 파일
- **model_evaluation_log.txt** (0.9KB)
  - 모델 평가 결과 기록
  - Precision, Recall, F1 Score

- **test_output.log** (5.2KB)
  - test.py 실행 중 전체 출력 로그
  - 훈련 진행률, 속도, 메모리 사용량

## 🔧 발견 및 수정된 버그

### 1. 평가 함수 텍스트 처리 불일치
**문제**: 
```python
# 훈련: ''.join(text) → "석수아"
# 평가: ' '.join(sentence) → "석 수 아"  ❌
```

**해결**:
```python
# 평가도 동일하게
text = ''.join(sentence)  ✅
```

### 2. 공백 처리 문제
**문제**: BERT 토크나이저가 공백을 offset에 포함하지 않음
- 훈련 데이터: "2022년 12월" → 공백에도 `I-DATE` 태그 ❌
- 모델 예측: 공백 위치는 라벨 없음 → `O` 예측
- 결과: 불일치!

**해결**:
```python
# 공백은 O 태그로 유지
if text[i].strip():  # 공백이 아닌 문자만
    tags[i] = f'I-{entity_type}'
# 공백은 O로 유지 ✅
```

### 3. 다중 문자 토큰의 B/I 태그 문제
**문제**: "연합뉴스" → 1개 토큰 → 모든 문자에 `B-COMPANY` ❌

**해결**:
```python
for char_idx in range(start, end):
    if char_idx == start:
        char_labels[char_idx] = label  # 첫 문자: B-
    else:
        if label.startswith('B-'):
            char_labels[char_idx] = label.replace('B-', 'I-')  ✅
```

### 4. 훈련 라벨 정렬 버그
**문제**:
```python
char_idx = start  # offset (0,2)일 때 label[0]만 사용 ❌
```

**해결**:
```python
if start < len(label):
    aligned_label.append(LABEL_TO_ID[label[start]])  ✅
```

## 📊 예상 결과

### Before Training (1-3번)
- F1 Score: 0% (정상 - 훈련 안 된 모델)

### After Training (4-6번)
- **목표**: F1 Score 95-99% 🎯
- 이전 결과: 21% → 44% (개선)
- 모든 버그 수정 후: **95%+ 기대**

## 🚀 실행 방법

```powershell
# validation 폴더에서 실행
cd validation
python test.py

# 또는 api 폴더에서
cd api
python validation/test.py

# 디버그 스크립트
cd validation
python debug_prediction.py
python debug_complex.py
python debug_space.py
python debug_tokenization.py
```

**주의**: 모든 스크립트는 `validation/` 폴더에서 실행하도록 설계되었습니다.
- 상대 경로가 자동으로 부모 디렉토리(`api/`)를 기준으로 조정됩니다.
- `sys.path`에 부모 디렉토리가 자동 추가되어 `api` 모듈을 import할 수 있습니다.

## ⏱️ 예상 소요 시간
- 데이터 생성: 1-2분
- bert 훈련: ~4분
- klue 훈련: ~20분
- xlm 훈련: ~20분
- **총**: 약 50분

## 📝 수정 날짜
- 2025-10-11 02:00 AM: 모든 버그 수정 및 재훈련 시작
