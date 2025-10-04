# NER 시스템

문서(계약서/동의서)에서 필드를 추출하는 NER(Named Entity Recognition) 시스템입니다.

## 주요 기능

- 문서 유형 자동 감지 (계약서/동의서)
- 문서 유형별 맞춤 필드 추출
- 유연한 텍스트 청크 분할 및 분석
- 문맥 기반 필드 값 추출
- 피드백 학습 시스템 (사용자 수정을 학습하여 성능 개선)
- 이미지 파일 OCR 처리 지원
- **단일 CSV 파일에 모든 문서 유형 결과 저장 및 재학습**

## 시스템 구조

```
/ner/                        # NER 패키지 디렉토리
├── __init__.py              # 패키지 초기화
├── ner_core.py              # 핵심 NER 프로세서
├── ner.py                   # 간소화된 NER 모듈
├── run_ner.py               # 명령줄 실행 스크립트
├── example.py               # 사용 예제 스크립트
├── combined_csv_demo.py     # 단일 CSV 재학습 예제
├── utils.py                 # 유틸리티 함수
├── document_type.py         # 문서 유형 감지 모듈
├── feedback.py              # 피드백 학습 모듈
├── ner_image.py             # 이미지 처리 모듈
├── ner_contract/            # 계약서 관련 모듈
│   └── contract_labels.py   # 계약서 레이블 정의
├── ner_consent/             # 동의서 관련 모듈
│   └── consent_labels.py    # 동의서 레이블 정의
├── out/                     # 결과 저장 디렉토리
│   └── results.csv         # 통합 결과 파일
└── learning_data/           # 피드백 학습 데이터
```

## 설치 및 요구사항

### 필수 라이브러리

```bash
pip install sentence-transformers pandas numpy
```

### 이미지 처리 관련 (선택)

```bash
pip install pytesseract pillow
```

- Tesseract OCR 엔진 설치 필요: 
  - [Windows 설치 링크](https://github.com/UB-Mannheim/tesseract/wiki)
  - Linux: `sudo apt install tesseract-ocr tesseract-ocr-kor`
  - Mac: `brew install tesseract tesseract-lang`

### 권장 환경

- Python 3.8 이상
- 최소 4GB RAM

## 사용 방법

### 명령줄에서 실행

```bash
# 기본 사용법
python -m ner.run_ner --input "데이터_디렉토리" --output "결과_디렉토리"

# 다양한 옵션 사용
python -m ner.run_ner --input "데이터_디렉토리" --output "결과_디렉토리" --model "모델명" --feedback --pattern "**/*.txt"

# 이미지 처리
python -m ner.ner_image --dir "이미지_디렉토리" --output "결과_CSV"
```

### 코드에서 사용

```python
from ner import NERProcessor

# 프로세서 초기화
processor = NERProcessor(use_feedback=True)

# 텍스트 처리
results, doc_type = processor.process_text("문서 내용...")
print(f"문서 유형: {doc_type}")
print("추출 결과:", results)

# 디렉토리 처리
results = processor.process_directory("데이터_디렉토리")

# 결과 저장
saved_file = processor.save_results(results, output_dir="결과_디렉토리", results_csv="results.csv")
```

## 주요 명령줄 옵션

| 옵션 | 설명 |
|------|------|
| `--input`, `-i` | 입력 디렉토리 경로 (필수) |
| `--output`, `-o` | 출력 디렉토리 경로 (기본값: ./out) |
| `--model`, `-m` | 사용할 문장 임베딩 모델 이름 (기본값: BAAI/bge-m3) |
| `--feedback`, `-f` | 피드백 학습 사용 여부 (플래그) |
| `--feedback-dir` | 피드백 학습 데이터 저장 디렉토리 (기본값: ./learning_data) |
| `--pattern`, `-p` | 처리할 파일 패턴 (기본값: **/*.txt) |
| `--results-csv` | 결과 파일 이름 (기본값: results.csv) |
| `--learn`, `-l` | 이전 결과에서 학습 (플래그) |
| `--original-csv` | 원본 결과 파일 경로 (비교용) |

## 피드백 학습 시스템

1. 초기 처리: `python -m ner.run_ner -i "데이터_디렉토리" -o "결과_디렉토리" -f`
2. 결과 수정: 생성된 results.csv 파일에서 잘못된 값 수정
3. 수정 학습: `python -m ner.run_ner -i "데이터_디렉토리" -o "결과_디렉토리" -f -l --original-csv "원본_results.csv"`
4. 다시 처리: `python -m ner.run_ner -i "데이터_디렉토리" -o "결과_디렉토리" -f`

## 단일 CSV 파일 저장 및 재학습

모든 문서 유형(계약서, 동의서)의 결과가 하나의 CSV 파일에 저장되어, 통합된 방식으로 필드 값을 관리하고 재학습할 수 있습니다. 문서 유형별로 적절한 레이블만 처리됩니다.

단일 CSV 파일의 구조:
- DocumentID: 문서 식별자
- DocumentType: 문서 유형 (계약서/동의서)
- 모든 가능한 레이블 컬럼 포함 (계약서와 동의서 레이블 모두)
- 각 문서 유형에 해당하는 레이블만 값이 채워짐

단일 CSV 파일에서의 재학습 예제는 다음 파일을 참조하세요:
```
ner/combined_csv_demo.py
```

실행 방법:
```bash
python -m ner.combined_csv_demo
```

## 문서 유형별 레이블

### 계약서 레이블

- 저작물명
- 대상 저작물 상세정보
- 양수자 기관명
- 양수자 주소
- 양도자 기관(개인)명
- 양도자 소속
- 양도자 대표주소
- 양도자 연락처
- 계약내용
- 계약일자

### 동의서 레이블

- 저작물명
- 저작물 설명
- 동의자 성명
- 동의자 주소
- 동의자 연락처
- 동의자 소속
- 저작권자
- 동의내용
- 동의여부
- 동의일자

## 예제 실행

간단한 예제 실행은 다음 명령어로 가능합니다:

```bash
python -m ner.example
```

## 개선사항

- 피드백 학습 시스템을 통한 지속적인 성능 개선
- 문서 유형 감지 알고리즘 강화
- 청크 분할 알고리즘 최적화
- OCR 처리 개선
