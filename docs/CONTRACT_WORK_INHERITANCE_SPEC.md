# 계약서→저작물 메타데이터 상속 API 연동 명세 (HF Space `ilwang-kogl-pipeline` 대상)

> 작성: 숭실대 (2026-07-16) · 대상: 무하유/왕일 프로 (HF Space 파이프라인)
> 목적: kogl-classifier의 저작물(work) 레코드 20개 항목을 실제 데이터로 채우기 위한 연동 방법.
> 배경: 저작물 이미지의 VLM 분석은 시각 필드만 채울 수 있고, 권리정보 9개 항목은 **계약서의 사실**이므로
> 계약서 분석 결과를 함께 보내면 서버가 병합(상속)해 반환합니다.

---

## 1. 무엇이 바뀌었나

`POST /api/llm-extract` 에 **선택(optional) 파라미터 1개**가 추가되었습니다. 기존 호출은 전혀 영향 없습니다.

| 파라미터 | 타입 | 설명 |
|---|---|---|
| `contract_metadata` | Form 필드, JSON **문자열** (≤200KB) | 해당 저작물이 속한 **계약서의 분석 결과** — 계약서 처리 응답의 `consolidated_metadata`(권장) 또는 `metadata` 를 그대로 직렬화해 전달 |

이미지 파일 + `contract_metadata` 가 함께 오면, 응답의 메타데이터에 계약서의 권리정보가
**상속 병합**되고, 각 상속 필드는 기존 `consolidation_decisions` 와 동일한 형식의 항목으로
출처가 표기됩니다 (`decision: "CONTRACT_INHERITED"`).

## 2. 권장 호출 순서 (검사 1건 = 계약서 1 + 저작물 N)

### 방식 A — 순차 (단순)

```
1) 계약서 파일 → POST /api/llm-extract            → 응답 A (기존과 동일)
2) 각 저작물 파일 → POST /api/llm-extract
     -F file=@work_image.jpg
     -F contract_metadata=<응답 A 의 consolidated_metadata JSON 문자열>
   → 응답 B: 시각 필드(VLM) + 권리 필드(계약서 상속) 병합 완료 레코드
3) 응답 B 를 works 테이블 20개 컬럼에 매핑 (아래 §4 표)
```

### 방식 B — 병렬 + 사후 병합 (권장: 검사 1건 전체 소요시간 최소화)

계약서 분석(~2분)을 기다리지 않고 저작물 분석을 **동시에** 시작한 뒤,
두 결과가 모두 도착하면 병합 전용 엔드포인트로 합칩니다 (LLM 호출 없음, 수 ms).

```
1) 계약서   → POST /api/llm-extract (document_type=계약서)      ─┐ 동시 실행
2) 각 저작물 → POST /api/llm-extract (contract_metadata 없이)    ─┘
3) 두 응답이 모두 도착하면:
   POST /api/apply-inheritance      (Content-Type: application/json)
   Body: { "contract_metadata": <응답 A 의 consolidated_metadata>,
           "work_response":     <응답 B 전체> }
   → 방식 A 의 2) 와 동일한 스키마의 병합 완료 레코드
```

- 전체 소요시간 ≈ max(계약서, 가장 느린 저작물) — 순차 방식 대비 저작물 N건일수록 이득.
- `/api/apply-inheritance` 오류 응답: 400(JSON 파싱 실패·필수 키 누락), 413(본문 >2MB).
- 데모 UI: `GET /pair` 가 이 방식으로 동작 (단일 제출 → 동시 분석 → 완료 즉시 자동 병합).

## 3. 병합 규칙 (서버 동작)

- **시각 필드는 절대 계약서 값으로 덮지 않음**: `description`, `work_type`, `keyword`,
  `digital_format` 등은 저작물 자체 분석(VLM) 결과 유지.
- **저작물 측에 이미 값이 있으면 유지** (계약서 값은 무시).
- **`work_title` 매칭**: 계약서가 저작물 여러 건을 다루는 경우(값이 리스트이거나 `외 N건`),
  저작물 **파일명·VLM 설명**과 토큰 매칭으로 해당 제목을 선택.
  - 명확히 못 고르면 값을 강제하지 않고 `decision: "CONTRACT_AMBIGUOUS"` 로 후보를 알려줌.
  - **복수 저작물 계약은 `work_title`(또는 `work_names`)을 배열로 보내는 것을 권장** —
    문자열 쉼표는 제목 일부로 취급합니다 (한국어 제목에 쉼표가 흔하므로).
- 상속 confidence: `CONTRACT_INHERITED` 0.8 (제목 매칭 선택 시 0.7), `CONTRACT_AMBIGUOUS` 0.5.

## 4. 응답 → works 테이블 20개 항목 매핑

응답에서 읽을 곳: `consolidated_metadata` (없으면 `metadata`). **(R)** = 키 이름이 다름 — 주의.

| # | works 컬럼 (kogl-classifier) | 응답 키 (숭실대 API) | 값 출처 |
|---|---|---|---|
| 1 | `work_name` | `work_title` **(R)** | 계약서 상속(매칭) |
| 2 | `work_type` | `work_type` | 저작물 VLM |
| 3 | `digital_format` | `digital_format` | 저작물 VLM |
| 4 | `description` | `description` | 저작물 VLM |
| 5 | `keywords` | `keyword` **(R)** | 저작물 VLM |
| 6 | `language` | `language` | VLM(이미지 내 문자 존재 시) |
| 7 | `created_date` | `created_date` | 계약서 상속 |
| 8 | `creator` | — (계약서 파일명) | 클라이언트에서 기입 |
| 9 | `copyright_holder` | `copyright_holder` | 계약서 상속 |
| 10 | `co_authors` | `co_author` **(R)** | 계약서 상속 |
| 11 | `neighboring_rights_holder` | `neighboring_rights_holder` | 계약서 상속 |
| 12 | `disclosure_type` | `disclosure_type` | 계약서 상속 |
| 13 | `copyrightability` | `copyrightability` | 계약서 상속 |
| 14 | `non_protected_work` | `unprotected_work` **(R)** | 계약서 상속 |
| 15 | `work_for_hire` | `work_for_hire` | 계약서 상속 |
| 16 | `commercial_use` | `commercial_use` | 계약서 상속 |
| 17 | `property_rights` | `economic_rights` **(R)** | 계약서 상속 |
| 18 | `co_author_consent` | `co_author_consent` | 계약서 상속 |
| 19 | `validity_period` | `valid_period` **(R)** | 계약서 상속 |
| 20 | `portrait_rights` | `portrait_rights` | 계약서 상속 |

→ 20개 중 **19개가 API 응답으로 채워짐** (4 VLM + 15 상속, `language` 조건부), `creator` 만 클라이언트 기입.

## 5. 응답 추가 필드

```jsonc
{
  "...": "기존 응답 그대로",
  "modality": "image",
  "vlm_backend": "Gemma 4 31B (OpenRouter)",
  "contract_inheritance": {           // NEW — 상속 요약
    "applied": true,
    "inherited": 7,                    // 상속된 필드 수
    "skipped_existing": 0,             // 저작물 값이 있어 건너뛴 수
    "title_match": "single"            // single | matched | ambiguous | none
  },
  "consolidation_decisions": [
    { "field": "copyright_holder", "final_value": "김종례",
      "decision": "CONTRACT_INHERITED", "confidence": 0.8,
      "reasoning": "저작물 파일에서 추출 불가한 권리/계약 정보 — 연계된 계약서 메타데이터에서 상속",
      "evidence": {"source": "계약서 메타데이터 상속 (contract_metadata)"},
      "llm_value": null, "ner_value": null }
  ]
}
```

## 6. 호출 예시 (curl)

```bash
# 1) 계약서
curl -X POST http://150.230.114.9:5000/api/llm-extract \
  -F "file=@contract.pdf" -F "document_type=계약서" -F "stream=false" > contract_result.json

# 2) 저작물 이미지 (+ 계약서 메타데이터)
CM=$(python3 -c "import json;print(json.dumps(json.load(open('contract_result.json'))['consolidated_metadata'],ensure_ascii=False))")
curl -X POST http://150.230.114.9:5000/api/llm-extract \
  -F "file=@work_photo.jpg" -F "contract_metadata=$CM" -F "stream=false"
```

## 7. 오류 처리

| 상황 | 동작 |
|---|---|
| `contract_metadata` 가 잘못된 JSON / 객체 아님 / >200KB | HTTP 400 + 오류 메시지 (파일 처리 전 조기 반환) |
| 상속 처리 중 내부 오류 | 응답은 정상 반환, `contract_inheritance: {applied:false, error:...}` |
| 문서(계약서 등) 파일에 `contract_metadata` 동봉 | 무시됨 (이미지 경로에서만 적용) |
| 영상/음성 | 기존과 동일 (P3 예정, 미지원 응답) |

## 8. 검증 상태

- 단위 테스트 8건 (단일/복수 저작물 매칭, 쉼표 포함 제목, 기존값 보존, 스키마 정합) 통과
- 실서버 E2E: 비스트리밍 + SSE 스트리밍 경로 모두 실이미지·실계약(합성 1500건 코퍼스) 검증 완료
- 배포: 2026-07-16 (150.230.114.9:5000)
