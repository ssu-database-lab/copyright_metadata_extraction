# 장세영 연구원님께 회신 메일

---

**To:** 장세영 <seyjang216@naver.com>

**Subject:** Re: [무하유] 실증 서비스 관련 사항 — CLI 도구 전달

---

안녕하세요, 장세영 연구원님.

밀란두입니다.

왕일 프로님 메일 내용과 연구원님의 회신 확인하였습니다.

말씀하신 실행 구조에 맞춰 CLI 도구를 구현 완료하여 공유드립니다.

---

## 1. 실행 구조

```
python extract.py [PDF 파일 경로] -t [문서유형] -o [출력 경로]
```

**입력:** PDF 파일 (또는 이미지 파일)

**출력:**
```
출력폴더/문서명/
├── ocr_text.txt                  — OCR 추출 텍스트
├── llm_metadata.json             — LLM 구조화 메타데이터 (67개 필드)
├── ner_entities.json             — NER 개체명 추출 결과
├── consolidated_metadata.json    — 최종 통합 메타데이터 (LLM + NER 병합)
└── full_response.json            — 전체 파이프라인 응답
```

### 실행 예시

```bash
# 단일 파일 (전체 파이프라인)
python extract.py document.pdf

# 문서 유형 지정
python extract.py contract.pdf -t 계약서

# 일괄 처리 (폴더 내 모든 PDF)
python extract.py ./documents/ -o ./results/

# OCR만 실행
python extract.py document.pdf -s ocr

# OCR + NER만 실행 (클라우드 LLM 불필요)
python extract.py document.pdf -s ocr+ner
```

Windows의 경우 `run_extract.bat` 더블 클릭으로도 실행 가능합니다.

---

## 2. 처리 파이프라인

```
PDF 업로드
    │
    ▼
[1단계] OCR 텍스트 추출 (Alibaba Cloud Qwen3-VL-235B)
    │
    ├─────────────────┐
    │                 │
[2단계] LLM 추출    [3단계] NER 추출    ← 동시 실행
    │                 │
    ├─────────────────┘
    │
[4단계] 통합 검증 (LLM + NER 결과 병합)
    │
    ▼
결과 파일 저장
```

---

## 3. 사용 모델

| 단계 | 모델 | 실행 위치 |
|------|------|-----------|
| OCR | Qwen3-VL-235B | Alibaba Cloud API |
| LLM 메타데이터 추출 | Qwen3.5-122B-A10B | Alibaba Cloud API |
| NER 개체명 추출 | KLUE-RoBERTa-Large | 로컬 CPU (패키지 포함) |
| 통합 검증 | Qwen3.5-122B-A10B | Alibaba Cloud API |

---

## 4. 설치 방법

1. 첨부한 `copyright_extraction_cli.tar.gz` (약 1.2GB) 압축 해제
2. Python 3.9 이상 설치 (https://www.python.org/downloads/)
3. `pip install -r requirements.txt` 실행 (약 5분)
4. `python extract.py document.pdf` 실행

상세한 설치 및 실행 방법은 패키지 내 `설치_안내서.md`를 참조해 주시기 바랍니다.

---

## 5. 패키지 내용물

- CLI 도구 소스코드 (`extract.py`)
- Windows 실행 파일 (`run_extract.bat`)
- NER 모델 (KLUE-RoBERTa-Large, 로컬 실행용)
- API 키 설정 완료 (`.env`)
- 설치 안내서 (`설치_안내서.md`)
- 추출 결과 예시 (`sample_output/`)

---

## 6. 테스트 요청

테스트 진행 후 문제가 있으시면 알려주시기 바랍니다.

또한, Python 설치 없이 더 간편하게 실행할 수 있는 방법도 별도로 준비 중이오니, 완료되는 대로 추가 공유드리겠습니다.

왕일 프로님 메일에서 말씀하신 실제 추출 예시는 패키지 내 `sample_output/` 폴더에 포함되어 있으며, 추가 문서에 대한 추출 결과가 필요하시면 말씀해 주세요.

감사합니다.

밀란두 드림
