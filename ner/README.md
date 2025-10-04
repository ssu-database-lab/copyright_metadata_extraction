# NER 시스템 - 저작권 동의서 및 계약서 자동 처리 시스템

이 디렉토리는 저작권 관련 문서(동의서, 계약서)에서 주요 정보를 자동으로 추출하는 명명 개체 인식(NER) 시스템을 포함하고 있습니다.

## 시스템 기능

- OCR 처리된 텍스트 파일에서 개체(Entity) 추출
- 동의서와 계약서 문서 유형 지원
- 한국어 텍스트 분석을 위한 NER 모델 학습
- 학습된 모델을 이용한 예측 수행
- 대화형 및 자동 학습 데이터 생성

## 디렉토리 구조

```
ner/
├── out/                        # 출력 디렉토리
│   ├── train_consent_results.csv   # 동의서 학습 데이터
│   ├── train_contract_results.csv  # 계약서 학습 데이터
│   ├── consent_prediction_results.csv  # 동의서 예측 결과
│   ├── contract_prediction_results.csv # 계약서 예측 결과
│   ├── consent_model/          # 동의서 모델
│   └── contract_model/         # 계약서 모델
├── create_training_data.py     # 학습 데이터 생성 스크립트
├── train_model.py              # 모델 학습 스크립트
├── ner.py                      # 예측 실행 스크립트
├── entity_extraction.py        # 개체 추출 유틸리티
└── requirements.txt            # 필요 패키지 목록
```

## 개체 레이블

### 동의서 레이블
- 양수인 성명 (Assignee Name)
- 양도인 주소 (Assignor Address)
- 양도인 전화번호 (Assignor Phone Number)
- 양수인 기관명 (Assignee Organization Name)
- 양수인 대표자명 (Assignee Representative Name)
- 양수인 대표자 주소 (Assignee Representative Address)
- 양수인 대표자 연락처 (Assignee Representative Contact)
- 동의여부 (Consent Status)
- 동의날짜 (Consent Date)

### 계약서 레이블
- 저작물명 (Work Title)
- 대상 저작물 상세정보 (Work Details)
- 양수자 기관명 (Assignee Organization Name)
- 양수자 주소 (Assignee Address)
- 양도자 기관(개인)명 (Assignor Organization/Individual Name)
- 양도자 소속 (Assignor Affiliation)
- 양도자 대표주소 (Assignor Representative Address)
- 양도자 연락처 (Assignor Contact)
- 동의여부 (Consent Status)
- 날짜 (Date)

## 사용 방법

### 설치

```bash
pip install -r requirements.txt
```

### 학습 데이터 생성

1. **대화형 모드**:
   ```bash
   python create_training_data.py
   ```

2. **자동 생성 모드**:
   ```bash
   # 동의서 학습 데이터 생성
   python create_training_data.py --auto --doc-type 동의서
   
   # 계약서 학습 데이터 생성
   python create_training_data.py --auto --doc-type 계약서
   ```

### 모델 학습

```bash
# 모든 모델 학습
python train_model.py --doc-type all

# 동의서 모델만 학습
python train_model.py --doc-type 동의서

# 계약서 모델만 학습
python train_model.py --doc-type 계약서
```

### 예측 실행

```bash
# 모든 문서 예측
python ner.py --predict-all

# 동의서 예측
python ner.py --predict-consent

# 계약서 예측
python ner.py --predict-contract
```

## 시스템 요구사항

- Python 3.8 이상
- PyTorch
- Transformers (Hugging Face)
- Pandas
- Datasets (Hugging Face)
- tqdm

## 주의사항

- OCR 처리된 텍스트 파일은 `../OCR/ocr_document` 디렉토리에 위치해야 합니다.
- 동의서 파일은 `동의서` 디렉토리 또는 이름에 '동의서'가 포함된 디렉토리에 위치해야 합니다.
- 계약서 파일은 `계약서` 디렉토리 또는 이름에 '계약서'가 포함된 디렉토리에 위치해야 합니다.
- 학습 데이터와 예측 결과는 `out` 디렉토리에 저장됩니다.
