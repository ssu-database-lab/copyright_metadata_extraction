# 저작물 메타데이터 추출 API 명세서

**문서 버전:** 1.0  
**최종 수정일:** 2026-02-06  
**API 버전:** 2.0.0  
**프레임워크:** FastAPI  

---

## 1. 개요

본 API는 **저작물(계약서·동의서 등) 업로드 시 메타데이터 추출 및 자동 분류**를 위한 프로토타입 수준의 REST API입니다.

### 주요 기능
- **OCR**: PDF/이미지에서 텍스트 추출 (Google, Naver, Mistral, Alibaba Cloud 지원)
- **메타데이터 추출**: LLM 기반 구조화된 메타데이터 추출
- **NER 엔티티 추출**: 인명, 날짜, 전화번호, 회사명 등 개체명 인식
- **메타데이터 통합**: LLM 결과와 NER 결과를 비교·병합하여 최종 메타데이터 생성
- **문서 유형 자동 분류**: 계약서, 동의서, 기타문서 등 유형별 맞춤 스키마 적용

### 지원 문서 유형
| 문서 유형 | 설명 |
|----------|------|
| 계약서 | 저작재산권 이용허락 계약서 등 |
| 동의서 | 개인정보 수집·이용 동의서 등 |
| 저작재산권 양도동의서 | 저작권 양도 관련 동의서 |
| 공공저작물 자유이용허락 동의서 | 공공누리 등 공공저작물 동의서 |
| 기타문서 | 일반 문서 (공문서, 사업계획서 등) |
| 디지털 콘텐츠 | 디지털 콘텐츠 관리 메타데이터 (42개 필드) |

---

## 2. API 엔드포인트 목록

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | 웹 UI 메인 페이지 |
| GET | `/health` | 서버 상태 확인 |
| GET | `/docs` | Swagger API 문서 (자동 생성) |
| GET | `/redoc` | ReDoc API 문서 |
| GET | `/api/info` | API 정보 및 사용 가능 모델 목록 |
| POST | `/api/llm-extract` | **핵심** 저작물 메타데이터 추출 (OCR + LLM + NER + 통합) |
| POST | `/api/ner-extract` | NER 엔티티 추출 전용 |
| POST | `/api/ocr-universal` | OCR 텍스트 추출 전용 |
| GET | `/api/llm-models` | 사용 가능 LLM 모델 목록 |
| GET | `/download/{request_id}` | 결과 파일 다운로드 |
| GET | `/api/list-files/{request_id}` | 결과 디렉토리 파일 목록 |

---

## 3. 핵심 API 상세 명세

### 3.1 메타데이터 추출 API (권장)

**`POST /api/llm-extract`**

저작물 업로드 시 메타데이터 추출 및 자동 분류를 수행하는 **통합 엔드포인트**입니다.

#### 요청 (multipart/form-data)

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `file` | File | ✅ | - | 업로드 파일 (PDF, PNG, JPG, JPEG, TIF, TIFF) |
| `model_name` | string | ❌ | solar-ko | LLM 모델 (solar-ko, alibaba-qwen-max, alibaba-qwen3-next-80b-a3b-instruct 등) |
| `document_type` | string | ❌ | 기타문서 | 문서 유형 (계약서, 동의서, 기타문서, 저작재산권 양도동의서, 공공저작물 자유이용허락 동의서, 디지털 콘텐츠) |
| `ocr_provider` | string | ❌ | google | OCR 엔진 (google, mistral, alibaba) |
| `ocr_model` | string | ❌ | null | OCR 모델 (Alibaba 사용 시) |
| `ner_model` | string | ❌ | klue-roberta-large | NER 모델 (klue-roberta-large, google-bert, xlm-roberta) |
| `consolidate` | boolean | ❌ | true | LLM+NER 메타데이터 통합 수행 여부 |
| `consolidation_model` | string | ❌ | alibaba-qwen3-next-80b-a3b-instruct | 통합용 LLM 모델 |
| `stream` | boolean | ❌ | false | SSE 스트리밍 응답 여부 |

#### 응답 (200 OK)

```json
{
  "success": true,
  "request_id": "20260206_123456_123456",
  "filename": "저작물계약서.pdf",
  "file_size_mb": 1.2,
  "model_used": "solar-ko",
  "document_type": "계약서",
  "metadata": {
    "contract_type": "저작재산권 비독점적 이용허락 계약서",
    "rights_holder": "홍길동",
    "user": "국립생태원",
    "work_title": "멸종위기 야생생물 홍보물",
    "contract_duration": "2024년 1월 1일부터 2024년 12월 31일까지",
    "signature_date": "2024-01-15",
    "parties": [...]
  },
  "consolidated_metadata": { ... },
  "consolidation_decisions": [ ... ],
  "entities": { "NAME": 5, "DATE": 3, "PHONE": 2 },
  "entity_count": 10,
  "processing_time": 12.5,
  "consolidation_success": true
}
```

#### 에러 응답

| 상태 코드 | 설명 |
|-----------|------|
| 400 | 잘못된 요청 (파일 형식, 모델 선택 등) |
| 500 | 서버 내부 오류 |

---

### 3.2 NER 엔티티 추출 API

**`POST /api/ner-extract`**

OCR + NER만 수행하는 엔드포인트입니다.

#### 요청 (multipart/form-data)

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `file` | File | ✅ | - | 업로드 파일 |
| `model` | string | ❌ | klue-roberta-large | NER 모델 |
| `ocr_provider` | string | ❌ | google | OCR 엔진 |
| `ocr_model` | string | ❌ | null | OCR 모델 |

#### 응답 (200 OK)

```json
{
  "success": true,
  "request_id": "20260206_123456_123456",
  "entities": { "NAME": 5, "DATE": 3, "PHONE": 2, "COMPANY": 1 },
  "entity_count": 11,
  "processing_time": 3.2
}
```

---

### 3.3 OCR 전용 API

**`POST /api/ocr-universal`**

텍스트 추출만 수행합니다. PDF, DOCX, HWP, 이미지 등 다양한 형식 지원.

#### 요청 (multipart/form-data)

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|----------|------|------|--------|------|
| `file` | File | ✅ | - | 업로드 파일 |
| `provider` | string | ❌ | google | google, mistral, naver, alibaba |
| `model` | string | ❌ | null | Alibaba 사용 시 모델명 |
| `stream` | boolean | ❌ | false | 스트리밍 출력 |

#### 응답 (200 OK)

```json
{
  "request_id": "20260206_123456",
  "success": true,
  "total_pages": 5,
  "total_text_length": 3500,
  "extracted_text": "...",
  "processing_time": 2.1
}
```

---

### 3.4 결과 다운로드 API

**`GET /download/{request_id}?type={entities|llm|stats}`**

| type | 반환 파일 |
|------|-----------|
| entities | NER 엔티티 JSON |
| llm | LLM 메타데이터 JSON |
| stats | 요약 통계 JSON |

---

### 3.5 헬스체크 및 정보 API

**`GET /health`**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-06T12:00:00",
  "available_models": ["klue-roberta-large", "google-bert", "xlm-roberta"],
  "universal_ocr_providers": ["google", "alibaba"]
}
```

**`GET /api/info`**
```json
{
  "title": "NER 엔티티 추출 API",
  "version": "2.0.0",
  "framework": "FastAPI",
  "models": { ... },
  "llm_models": { ... }
}
```

---

## 4. 메타데이터 스키마 (문서 유형별)

### 4.1 계약서 (contract_type, rights_holder, user, work_title, contract_duration, signature_date, parties 등)

### 4.2 동의서 (consent_type, data_controller, data_subject, collection_purpose, retention_period 등)

### 4.3 기타문서 (document_type, title, main_content, key_dates, key_amounts 등)

상세 필드 정의는 `api/module/llm_extraction/METADATA_SCHEMA_DOCUMENTATION.md` 참조.

---

## 5. 기술 스택 및 의존성

- **Python**: 3.8+
- **FastAPI**: 웹 프레임워크
- **OCR**: Google Vision, Naver CLOVA, Mistral, Alibaba Cloud Qwen3-VL
- **LLM**: SOLAR-Ko (로컬), Alibaba Qwen 시리즈 (클라우드)
- **NER**: KLUE RoBERTa, Google mBERT, XLM-RoBERTa (Hugging Face)

---

## 6. 프로토타입 범위 및 제한사항

- **현재 수준**: 프로토타입 (연구·시범용)
- **인증**: API 키/토큰 기반 인증 미구현 (내부 테스트용)
- **Rate Limit**: 미적용
- **배포**: 로컬/내부 서버 실행 기준
- **프로토타입 수준 및 범위**: 협의 가능

---

## 7. 연락처

본 API에 대한 문의는 과제 담당자에게 연락 부탁드립니다.
