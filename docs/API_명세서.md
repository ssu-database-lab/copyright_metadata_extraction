# 저작권 메타데이터 추출 API 명세서

**버전:** 2.0.0
**작성자:** 숭실대학교 DB연구실
**작성일:** 2026-03-18
**프레임워크:** FastAPI (Python)
**기본 URL:** `http://{server}:5000`
**API 문서:** `http://{server}:5000/docs` (Swagger UI)

---

## 1. 시스템 개요

PDF/이미지 문서를 업로드하면 OCR → LLM 메타데이터 추출 → NER 엔티티 추출 → 통합 검증(Consolidation)까지 수행하여 구조화된 메타데이터를 JSON으로 반환하는 API입니다.

### 처리 파이프라인

```
입력 (PDF/이미지)
    ↓
[1] OCR 텍스트 추출 (Qwen3-VL-235B, Alibaba Cloud)
    ↓
[2] LLM 메타데이터 추출 (Qwen3-Next-80B, Alibaba Cloud)
    ↓
[3] NER 엔티티 추출 (KLUE-RoBERTa-Large, 로컬)
    ↓
[4] 통합 검증 (Qwen3-Next-80B → LLM/NER 결과 비교·판정)
    ↓
출력 (JSON)
```

### 사용 모델

| 단계 | 모델 | 호스팅 |
|------|------|--------|
| OCR | Qwen3-VL-235B-A22B-Instruct | Alibaba Cloud (DashScope API) |
| LLM 메타데이터 추출 | Qwen3-Next-80B-A3B-Instruct | Alibaba Cloud (DashScope API) |
| 통합 검증 | Qwen3-Next-80B-A3B-Instruct | Alibaba Cloud (DashScope API) |
| NER 엔티티 추출 | KLUE-RoBERTa-Large | 로컬 서버 |

### 지원 파일 형식

PDF, PNG, JPG, JPEG, TIF, TIFF

---

## 2. API 엔드포인트

### 2.1 `POST /api/llm-extract` — 통합 메타데이터 추출 (메인)

전체 파이프라인 (OCR + LLM + NER + 통합 검증)을 수행합니다.

**요청 파라미터 (multipart/form-data):**

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `file` | File | (필수) | 업로드 파일 (PDF/이미지) |
| `model_name` | string | `"solar-ko"` | LLM 모델명 |
| `document_type` | string | `"기타문서"` | 문서 유형 (계약서, 동의서, 저작재산권 양도동의서, 공공저작물 자유이용허락 동의서, 기타문서) |
| `ocr_provider` | string | `"google"` | OCR 제공자 (google, mistral, naver, alibaba) |
| `ocr_model` | string | `null` | OCR 모델명 (예: qwen3-vl-235b-a22b-instruct) |
| `ner_model` | string | `"klue-roberta-large"` | NER 모델 (klue-roberta-large, google-bert-mbert, xlm-roberta-large) |
| `consolidate` | bool | `true` | 통합 검증 수행 여부 |
| `consolidation_model` | string | `"alibaba-qwen3-next-80b-a3b-instruct"` | 통합 검증 모델 |
| `stream` | bool | `false` | SSE 스트리밍 활성화 |

**응답 (JSON):**

```json
{
  "success": true,
  "request_id": "20251110_120246_846222",
  "filename": "문서파일명.pdf",
  "file_size_mb": 0.35,
  "model_used": "모델명",
  "document_type": "동의서",

  "metadata": { ... },
  "confidence": 1.0,
  "extraction_time": 14.2,

  "ocr_text": "OCR로 추출된 전체 텍스트",
  "ocr_provider": "alibaba",
  "ocr_model": "qwen3-vl-235b-a22b-instruct",

  "ner_model": "KLUE RoBERTa Large",
  "entities": { "NAME": 3, "PHONE": 2, "ADDRESS": 1 },
  "entity_count": 6,
  "ner_success": true,

  "consolidate": true,
  "consolidation_success": true,
  "consolidated_metadata": { ... },
  "consolidation_decisions": [ ... ],
  "consolidation_summary": {
    "total_fields": 27,
    "agreed_fields": 12,
    "conflicted_fields": 6,
    "llm_only_fields": 8,
    "ner_only_fields": 0,
    "missing_fields": 1,
    "overall_confidence": 0.87
  },
  "consolidation_confidence": 0.87,
  "consolidation_model_used": "alibaba-qwen3-next-80b-a3b-instruct",
  "consolidation_fallback_used": false,

  "processing_time": 42.41
}
```

**응답 파라미터 설명:**

| 파라미터 | 타입 | 예시값 | 설명 |
|---------|------|--------|------|
| `success` | boolean | `true` | 요청 처리 성공 여부 |
| `request_id` | string | `"20251110_120246_846222"` | 요청 추적용 고유 ID |
| `filename` | string | `"문서파일명.pdf"` | 업로드된 원본 파일명 |
| `file_size_mb` | number | `0.35` | 파일 크기 (MB) |
| `model_used` | string | `"모델명"` | 문서 분석에 사용된 LLM 모델명 |
| `document_type` | string | `"동의서"` | 판별된 문서 유형 |
| `metadata` | object | `{ ... }` | 문서에서 추출한 메타데이터 (문서 유형별 스키마에 따라 구조가 다름) |
| `confidence` | number | `1.0` | 전체 추출 신뢰도 (0.0~1.0) |
| `extraction_time` | number | `14.2` | LLM 추출 처리 시간 (초) |
| `ocr_text` | string | `"OCR로 추출된 전체 텍스트"` | OCR로 추출한 전체 본문 텍스트 |
| `ocr_provider` | string | `"alibaba"` | 사용한 OCR 제공자 |
| `ocr_model` | string | `"qwen3-vl-235b-a22b-instruct"` | 사용한 OCR 모델명 |
| `ner_model` | string | `"KLUE RoBERTa Large"` | 개체명 인식(NER) 모델명 |
| `entities` | object | `{ "NAME": 3, "PHONE": 2 }` | 개체 타입별 추출 개수 |
| `entity_count` | integer | `6` | 총 추출 개체 수 |
| `ner_success` | boolean | `true` | NER 처리 성공 여부 |
| `consolidate` | boolean | `true` | 통합 검증 수행 여부 |
| `consolidation_success` | boolean | `true` | 통합 처리 성공 여부 |
| `consolidated_metadata` | object | `{ ... }` | 통합된 최종 메타데이터 |
| `consolidation_decisions` | array | `[ ... ]` | 필드별 통합 판단 결과 (하위 구조 참조) |
| `consolidation_summary` | object | `{ ... }` | 통합 결과 요약 정보 (하위 구조 참조) |
| `consolidation_confidence` | number | `0.87` | 통합 결과 신뢰도 (0.0~1.0) |
| `consolidation_model_used` | string | `"alibaba-qwen3-next-80b-a3b-instruct"` | 통합 처리에 사용한 모델 |
| `consolidation_fallback_used` | boolean | `false` | fallback 모델 사용 여부 |
| `processing_time` | number | `42.41` | 전체 처리 시간 (초) |

**`consolidation_summary` 하위 구조:**

| 파라미터 | 타입 | 예시값 | 설명 |
|---------|------|--------|------|
| `total_fields` | integer | `27` | 검증 대상 전체 필드 수 |
| `agreed_fields` | integer | `12` | LLM과 NER 결과가 일치한 필드 수 |
| `conflicted_fields` | integer | `6` | LLM과 NER 결과가 충돌하여 판정이 필요한 필드 수 |
| `llm_only_fields` | integer | `8` | LLM만 추출한 필드 수 (NER 결과 없음) |
| `ner_only_fields` | integer | `0` | NER만 추출한 필드 수 (LLM 결과 없음) |
| `missing_fields` | integer | `1` | LLM·NER 모두 추출하지 못한 필드 수 |
| `overall_confidence` | number | `0.87` | 통합 검증 종합 신뢰도 (0.0~1.0) |

**`consolidation_decisions[]` 배열 항목 구조:**

| 파라미터 | 타입 | 예시값 | 설명 |
|---------|------|--------|------|
| `field` | string | `"data_subject"` | 필드명 |
| `llm_value` | any | `"박광수"` | LLM이 추출한 값 |
| `ner_value` | any | `"박광수"` | NER이 추출한 값 |
| `final_value` | any | `"박광수"` | 통합 판정 후 최종 채택값 |
| `decision` | string | `"AGREED"` | 판정 결과 (AGREED, CONFLICT, LLM_ONLY, NER_ONLY, MISSING) |
| `confidence` | number | `1.0` | 해당 필드의 판정 신뢰도 (0.0~1.0) |
| `reasoning` | string | `"LLM과 NER 결과 일치"` | 판정 근거 설명 |

---

### 2.2 `POST /api/ocr-universal` — OCR 텍스트 추출

문서에서 텍스트만 추출합니다.

**요청 파라미터 (multipart/form-data):**

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `file` | File | (필수) | 업로드 파일 |
| `provider` | string | `"google"` | OCR 제공자 |
| `model` | string | `null` | 모델명 |
| `stream` | bool | `false` | 스트리밍 출력 |

**응답 (JSON):**

```json
{
  "request_id": "20251110_120246_846222",
  "filename": "문서파일명.pdf",
  "provider": "alibaba",
  "model": "qwen3-vl-235b-a22b-instruct",
  "success": true,
  "total_pages": 1,
  "total_text_length": 871,
  "processing_time": 8.5,
  "extracted_text": "전체 추출 텍스트...",
  "pages": [
    {
      "page_number": 1,
      "extracted_text": "페이지 1 텍스트...",
      "text_length": 871,
      "status": "success"
    }
  ]
}
```

**응답 파라미터 설명:**

| 파라미터 | 타입 | 예시값 | 설명 |
|---------|------|--------|------|
| `request_id` | string | `"20251110_120246_846222"` | 요청 추적용 고유 ID |
| `filename` | string | `"문서파일명.pdf"` | 업로드된 원본 파일명 |
| `provider` | string | `"alibaba"` | 사용한 OCR 제공자 |
| `model` | string | `"qwen3-vl-235b-a22b-instruct"` | 사용한 OCR 모델명 |
| `success` | boolean | `true` | 요청 처리 성공 여부 |
| `total_pages` | integer | `1` | 문서 총 페이지 수 |
| `total_text_length` | integer | `871` | 추출된 전체 텍스트 길이 (문자 수) |
| `processing_time` | number | `8.5` | OCR 처리 시간 (초) |
| `extracted_text` | string | `"전체 추출 텍스트..."` | 모든 페이지의 텍스트를 합친 전체 본문 |
| `pages` | array | `[ ... ]` | 페이지별 추출 결과 배열 (하위 구조 참조) |

**`pages[]` 배열 항목 구조:**

| 파라미터 | 타입 | 예시값 | 설명 |
|---------|------|--------|------|
| `page_number` | integer | `1` | 페이지 번호 (1부터 시작) |
| `extracted_text` | string | `"페이지 1 텍스트..."` | 해당 페이지에서 추출된 텍스트 |
| `text_length` | integer | `871` | 해당 페이지 텍스트 길이 (문자 수) |
| `status` | string | `"success"` | 페이지별 처리 상태 (success, error) |

---

### 2.3 `POST /api/ner-extract` — NER 엔티티 추출

OCR 후 NER 모델로 엔티티(인명, 기관명, 날짜, 연락처 등)를 추출합니다.

**요청 파라미터 (multipart/form-data):**

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `file` | File | (필수) | 업로드 파일 |
| `model` | string | `"klue-roberta-large"` | NER 모델 |
| `ocr_provider` | string | `"google"` | OCR 제공자 |
| `ocr_model` | string | `null` | OCR 모델명 |

**응답 (JSON):**

```json
{
  "success": true,
  "request_id": "20251110_120246_846222",
  "filename": "문서파일명.pdf",
  "file_size_mb": 0.35,
  "model": "KLUE RoBERTa Large",
  "entities": {
    "NAME": 3,
    "PHONE": 2,
    "ADDRESS": 2,
    "COMPANY": 1
  },
  "entity_count": 8,
  "steps": {
    "ocr": { "success": true, "time": 8.5 },
    "ner": { "success": true, "entity_count": 8, "time": 3.2 }
  },
  "processing_time": 11.7
}
```

**응답 파라미터 설명:**

| 파라미터 | 타입 | 예시값 | 설명 |
|---------|------|--------|------|
| `success` | boolean | `true` | 요청 처리 성공 여부 |
| `request_id` | string | `"20251110_120246_846222"` | 요청 추적용 고유 ID |
| `filename` | string | `"문서파일명.pdf"` | 업로드된 원본 파일명 |
| `file_size_mb` | number | `0.35` | 파일 크기 (MB) |
| `model` | string | `"KLUE RoBERTa Large"` | 사용한 NER 모델명 |
| `entities` | object | `{ "NAME": 3, "PHONE": 2 }` | 개체 타입별 추출 개수 (키: 개체 타입, 값: 개수) |
| `entity_count` | integer | `8` | 총 추출 개체 수 |
| `steps` | object | `{ ... }` | 단계별 처리 결과 (하위 구조 참조) |
| `processing_time` | number | `11.7` | 전체 처리 시간 (초) |

**`steps` 하위 구조:**

| 파라미터 | 타입 | 예시값 | 설명 |
|---------|------|--------|------|
| `steps.ocr.success` | boolean | `true` | OCR 단계 성공 여부 |
| `steps.ocr.time` | number | `8.5` | OCR 처리 시간 (초) |
| `steps.ner.success` | boolean | `true` | NER 단계 성공 여부 |
| `steps.ner.entity_count` | integer | `8` | NER 추출 개체 수 |
| `steps.ner.time` | number | `3.2` | NER 처리 시간 (초) |

---

### 2.4 `GET /download/{request_id}` — 결과 다운로드

추출 결과 JSON 파일을 다운로드합니다.

**경로 파라미터:**

| 파라미터 | 설명 |
|---------|------|
| `request_id` | 요청 ID (예: 20251110_120246_846222) |

**쿼리 파라미터:**

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `type` | `"entities"` | 다운로드 유형 (entities, metadata) |

**응답:** 해당 요청 ID의 결과 JSON 파일이 다운로드됩니다.

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| (파일) | JSON file | `type=entities`: NER 엔티티 결과 파일, `type=metadata`: LLM 메타데이터 결과 파일 |

---

### 2.5 `GET /health` — 서버 상태

**응답 (JSON):**

```json
{
  "status": "healthy",
  "timestamp": "2026-03-18T10:00:00",
  "available_ocr_engines": { "google": true, "alibaba": true },
  "available_models": { ... }
}
```

**응답 파라미터 설명:**

| 파라미터 | 타입 | 예시값 | 설명 |
|---------|------|--------|------|
| `status` | string | `"healthy"` | 서버 상태 (healthy, unhealthy) |
| `timestamp` | string | `"2026-03-18T10:00:00"` | 응답 시점 타임스탬프 (ISO 8601) |
| `available_ocr_engines` | object | `{ "google": true, "alibaba": true }` | 사용 가능한 OCR 엔진 목록 및 활성화 여부 |
| `available_models` | object | `{ ... }` | 사용 가능한 NER/LLM 모델 목록 |

---

## 3. 실제 처리 결과 예시

### 3.1 입력 문서

**파일명:** `샘플_저작물-20250812T232645Z-1-001_doc00180820201224100019.pdf`
**문서 유형:** 개인정보 수집 및 이용 동의서
**파일 크기:** 0.35 MB

### 3.2 OCR 추출 결과

**OCR 모델:** Qwen3-VL-235B (Alibaba Cloud)

```
4. 개인정보 수집 및 이용 동의

○ 개인정보 수집 및 이용 목적 : 저작인접권의 저작재산권 양도 의사표시 확인 및
  초상 공개·사용 등 의사표시 확인

○ 수집하는 개인정보 항목 : 성명, 전화번호(휴대전화), 주소

○ 개인정보 보유 및 이용 기간 : 동의 시부터 저작인접권 보호기간 만료일까지

○ 개인정보 수집 및 이용에 동의하지 않을 권리가 있으며, 본 동의를 거절하실 경우에는
  저작물의 제작이 불가함을 알려드립니다.

양도자 본인은 개인정보 수집 및 이용에 동의합니다. 동의함 ☑ 동의하지 않음 □

5. 개인정보 제3자 제공에 대한 동의

○ 개인정보를 제공받는 자: 한국문화정보원
○ 개인정보를 제공받는 자의 개인정보 이용 목적: 저작물의 관리
○ 제공하는 개인정보의 항목: 성명, 전화번호, 주소

양도자 본인은 개인정보의 제3자 제공에 동의합니다. 동의함 ☑ 동의하지 않음 □

양도인
성명: 박광수 (서명)
주소: 진천읍 외동1길8 전화번호: 010-8024-17505

양수인
기관명: 주) 나라지식정보 대표자명: 손영호
대표자 주소: 서울특별시 종로구 삼봉로 81, 409호 대표자 연락처: 20-3141-7644

2020. 6. 20.
```

### 3.3 LLM 메타데이터 추출 결과

**LLM 모델:** Qwen-Max (Alibaba Cloud)
**추출 시간:** 14.2초
**신뢰도:** 1.0

```json
{
  "consent_type": "개인정보 수집 및 이용 동의서",
  "data_controller": "주) 나라지식정보",
  "data_subject": "박광수",
  "collection_purpose": "저작인접권의 저작재산권 양도 의사표시 확인 및 초상 공개·사용 등 의사표시 확인",
  "collected_data_types": ["성명", "전화번호(휴대전화)", "주소"],
  "retention_period": "동의 시부터 저작인접권 보호기간 만료일까지",
  "third_party_sharing": {
    "recipient": "한국문화정보원",
    "purpose": "저작물의 관리",
    "data_types": ["성명", "전화번호", "주소"]
  },
  "consent_status": "동의함",
  "consent_date": "2020-06-20",
  "signature": "박광수",
  "contact_info": {
    "phone": "010-8024-17505",
    "address": "진천읍 외동1길8",
    "email": null
  },
  "parties": [
    { "name": "박광수", "phone": "010-8024-17505", "address": "진천읍 외동1길8", "role": "정보주체" },
    { "name": "주) 나라지식정보", "phone": "20-3141-7644", "address": "서울특별시 종로구 삼봉로 81, 409호", "role": "처리자" }
  ]
}
```

### 3.4 NER 엔티티 추출 결과

**NER 모델:** KLUE-RoBERTa-Large (로컬)

```json
{
  "entities": [
    ["서울특별시 종로구 삼봉로 81", "ADDRESS"],
    ["진천읍 외동1길8 전화번호", "ADDRESS"],
    ["한국문화정보원", "COMPANY"],
    ["박광수", "NAME"],
    ["손영호", "NAME"],
    ["지식정보", "NAME"],
    ["010-8024-1750", "PHONE"],
    ["010-8024-17505", "PHONE"],
    ["20-3141-7644", "PHONE"]
  ],
  "entity_count": 9,
  "entity_types": ["ADDRESS", "COMPANY", "NAME", "PHONE"]
}
```

> **참고:** 통합 검증(Consolidation) 결과 상세 내용은 별도 문서 `통합검증_결과_상세.md`를 참조하시기 바랍니다.

---

## 4. 결과 파일 구조

각 요청에 대해 `results/{request_id}/` 디렉토리에 결과가 저장됩니다:

```
results/{request_id}/
├── llm_metadata.json              # LLM 추출 + NER + 통합 전체 결과
├── consolidated_metadata.json     # 통합 검증 상세 결과
└── ner/
    └── ner/
        └── klue-roberta-large/
            └── {filename}_entities.json   # NER 엔티티 상세 결과
```

---

## 5. 에러 응답

모든 엔드포인트는 실패 시 아래 형식으로 응답합니다:

```json
{
  "success": false,
  "error": "에러 메시지",
  "request_id": "20251110_120246_846222"
}
```

**주요 에러 코드:**
- `400` — 지원하지 않는 파일 형식
- `500` — 서버 내부 오류 (OCR 실패, 모델 로딩 실패 등)
