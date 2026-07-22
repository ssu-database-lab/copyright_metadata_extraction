# 저작권 메타데이터 추출 API 사용 명세서 (v3 — 멀티모달 포함)

**버전:** 3.0.0
**작성:** 숭실대학교 DB연구실
**작성일:** 2026-07-23
**프레임워크:** FastAPI (Python)
**기본 URL:** `http://{server}:5000` (운영: `http://150.230.114.9:5000`)
**API 문서(Swagger):** `http://{server}:5000/docs`
**코드:** GitHub `ssu-database-lab/copyright_metadata_extraction` (main)

> 본 문서는 무하유 통합 시스템 연동을 위한 API 사용 명세입니다. v2 대비 **① 멀티모달(이미지 저작물) 처리**, **② VLM 백엔드로 OpenRouter Gemma 4 도입**, **③ 계약서→저작물 메타데이터 상속(/api/apply-inheritance)** 이 추가되었습니다. 변경 요약은 §2 참조.

---

## 1. 시스템 개요

파일(계약서·동의서 등 **문서**, 또는 저작물 **이미지**)을 업로드하면 파일 종류(modality)를 판별해 알맞은 파이프라인으로 처리하고, 통합 67필드 스키마의 구조화 메타데이터를 JSON으로 반환합니다.

### 처리 파이프라인 (modality 라우팅)

```
입력 파일
   ↓
[0] Modality 라우터 — 확장자/내용으로 문서 vs 이미지 판별
   │
   ├─(문서: PDF/이미지 스캔/DOCX/HWP)────────────────────────────┐
   │   [1] OCR 텍스트 추출 (Qwen3-VL-235B)                       │
   │   [2] LLM 메타데이터 추출 (Qwen3.5-122B) ∥ [3] NER (KLUE-RoBERTa) │
   │   [4] 통합 검증 (Qwen3.5-122B, LLM/NER 판정)                 │
   │                                                            │
   └─(이미지: PNG/JPG 등 저작물 파일)────────────────────────────┤
       [1'] VLM 속성 추출 (Gemma 4 31B @ OpenRouter → 자체서버 → Qwen3-VL)
       [2'] (선택) 계약서 메타데이터 상속 병합                      │
                                                                ↓
                                                          출력 (JSON, 67필드 통합 스키마)
```

### 사용 모델

| 단계 | 모델 | 호스팅 | 비고 |
|------|------|--------|------|
| OCR | Qwen3-VL-235B-A22B-Instruct | Alibaba Cloud (DashScope) | fallback: Qwen3.5-Flash |
| LLM 메타데이터 추출 | Qwen3.5-122B-A10B | Alibaba Cloud (DashScope) | |
| 통합 검증(Consolidation) | Qwen3.5-122B-A10B | Alibaba Cloud (DashScope) | fallback: Qwen3.5-Plus |
| NER 엔티티 추출 | KLUE-RoBERTa-Large | 로컬 서버 (CPU) | |
| **VLM (이미지 저작물 속성)** | **Gemma-4-31B-it** | **OpenRouter API (1순위)** | **자체 vLLM(2순위) → Qwen3-VL-235B(3순위) 자동 폴백** |

### 지원 파일 형식

- **문서:** PDF, PNG, JPG, JPEG, TIF, TIFF, DOCX, HWP
- **이미지 저작물:** PNG, JPG, JPEG, WEBP, BMP, GIF, TIF, TIFF

---

## 2. v2 → v3 변경 사항 (연동 시 확인)

1. **멀티모달(이미지 저작물) 처리 추가** — 이미지 파일 업로드 시 OCR/NER 대신 VLM 속성추출 경로로 자동 라우팅. 응답에 `modality`, `vlm_backend` 필드 추가. (§4)
2. **VLM 백엔드 = OpenRouter Gemma 4 31B (기본)** — 자체 GPU 호스팅을 OpenRouter API 호출로 전환(운영 안정성·비용). 자체서버·Qwen3-VL 자동 폴백. (§5)
3. **계약서→저작물 메타데이터 상속** — 계약서의 권리정보(저작권자·공공누리유형·유효기간 등)를 저작물 이미지 결과에 병합. 두 방식 제공: (A) `/api/llm-extract` 의 `contract_metadata` 폼 파라미터, (B) 신규 **`/api/apply-inheritance`** (계약서·저작물 동시 분석 후 사후 병합). (§6, 별도 문서 `CONTRACT_WORK_INHERITANCE_SPEC.md`)
4. **모델 갱신** — 추출/통합 모델을 Qwen3.5-122B로, DashScope 전용 엔드포인트/워크스페이스 키로 이전(코드 내 환경변수화, 연동 측 변경 불필요).

---

## 3. API 엔드포인트

### 3.1 `POST /api/llm-extract` — 통합 메타데이터 추출 (메인)

문서/이미지 공통 진입점. modality를 자동 판별해 문서면 OCR+LLM+NER+통합, 이미지면 VLM 속성추출을 수행합니다.

**요청 파라미터 (multipart/form-data):**

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `file` | File | (필수) | 업로드 파일 (문서 또는 이미지) |
| `document_type` | string | `"기타문서"` | 문서 유형 (계약서, 동의서, 저작재산권 양도동의서, 공공저작물 자유이용허락 동의서, 기타문서). **이미지는 무시됨** |
| `model_name` | string | `"alibaba-qwen3.5-122b-a10b"` | LLM 추출 모델 |
| `ocr_provider` | string | `"alibaba"` | OCR 제공자 (alibaba, mistral, google, naver) — 문서 전용 |
| `ocr_model` | string | `null` | OCR 모델명 |
| `ner_model` | string | `"klue-roberta-large"` | NER 모델 — 문서 전용 |
| `consolidate` | bool | `true` | 통합 검증 수행 여부 |
| `consolidation_model` | string | `"alibaba-qwen3.5-122b-a10b"` | 통합 검증 모델 |
| `vlm_prefer` | string | `"gemma"` | **[신규]** VLM 우선 백엔드 (gemma=OpenRouter우선, gemma-local=자체서버우선, qwen=Qwen우선) — 이미지 전용 |
| `contract_metadata` | string(JSON) | `null` | **[신규]** 계약서 분석 결과(consolidated_metadata)의 JSON 문자열. 이미지 처리 시 권리정보를 상속 병합 (최대 200KB). (§6) |
| `stream` | bool | `false` | SSE 스트리밍 활성화 |

**응답 (문서, JSON):** v2와 동일 구조(`metadata`, `ocr_text`, `entities`, `consolidated_metadata`, `consolidation_decisions[]`, `consolidation_summary`, `processing_time` 등). 상세는 v2 명세 §2.1 참조.

**응답 (이미지 저작물, JSON):** 아래 필드가 추가/변형됩니다.

```json
{
  "success": true,
  "request_id": "20260723_140312_882110",
  "filename": "제주_유채꽃.jpg",
  "modality": "image",
  "vlm_backend": "Gemma 4 31B (OpenRouter)",
  "ocr_provider": "(image/VLM)",
  "ner_model": null,

  "metadata": {
    "description": "노란 유채꽃밭이 펼쳐진 풍경 사진입니다. ...",
    "work_type": "사진저작물",
    "keyword": ["유채꽃", "제주", "봄", "풍경"],
    "digital_format": "JPG"
  },
  "consolidated_metadata": { "...": "위 metadata + (상속 시) 권리정보 필드" },
  "consolidation_decisions": [
    { "field": "description", "final_value": "...", "decision": "LLM_ONLY", "confidence": 0.6, "reasoning": "..." }
  ],
  "contract_inheritance": { "applied": false },
  "processing_time": 7.3
}
```

**이미지 응답 추가 필드:**

| 파라미터 | 타입 | 예시값 | 설명 |
|---------|------|--------|------|
| `modality` | string | `"image"` | 판별된 파일 종류 (image / document) |
| `vlm_backend` | string | `"Gemma 4 31B (OpenRouter)"` | 실제 속성추출에 사용된 VLM 백엔드(폴백 반영) |
| `contract_inheritance` | object | `{ "applied": true, "inherited": 18, "title_match": "single" }` | 계약서 상속 병합 결과 요약 (상속 미적용 시 `applied:false`) |

> 이미지 결과의 시각 필드(description/work_type/keyword/digital_format)는 VLM 산출이므로 통합 판정은 `LLM_ONLY`(신뢰도 0.5~0.7)로 표기됩니다. work_type은 **매체(medium) 기준**(피사체가 아니라 '무엇으로 만들어졌는가')으로 분류됩니다. (§4)

---

### 3.2 `POST /api/apply-inheritance` — 계약서→저작물 상속 병합 (신규)

계약서와 저작물을 **동시에** 분석한 뒤, 두 결과가 모두 도착하면 계약서 권리정보를 저작물 결과에 병합하는 **순수 병합 엔드포인트(LLM 호출 없음, 수 ms)**. 무하유 검사(계약서 1 + 저작물 N)의 병렬 처리에 사용합니다.

**요청 (application/json):**

```json
{
  "contract_metadata": { "copyright_holder": "제주관광공사", "kogl_type": "제1유형", "valid_period": "2026-12-31", "...": "..." },
  "work_response":     { "success": true, "filename": "제주_유채꽃.jpg", "consolidated_metadata": { "...": "..." }, "...": "저작물 /api/llm-extract 응답 전체" }
}
```

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `contract_metadata` | object (필수) | 계약서 분석 결과의 `consolidated_metadata`(또는 `metadata`) |
| `work_response` | object (필수) | 저작물 `/api/llm-extract` 응답 전체 |

**응답:** `work_response` 에 상속을 적용한 레코드. `/api/llm-extract` 에 `contract_metadata` 를 준 것과 동일 스키마(상속 필드는 `consolidation_decisions` 에 `CONTRACT_INHERITED`(0.8)/`CONTRACT_AMBIGUOUS`(0.5) 로 추가, `contract_inheritance` 요약 포함).

**오류:** `400`(JSON 파싱 실패·필수 키 누락), `413`(본문 > 2MB).

> 권장 호출 순서(병렬): ① 계약서 → `/api/llm-extract`(document_type=계약서), ② 각 저작물 → `/api/llm-extract`(contract_metadata **없이**) 를 계약서와 동시에, ③ 둘 다 도착하면 → `/api/apply-inheritance`. 데모 UI: `GET /pair`.

---

### 3.3 `POST /api/ocr-universal` — OCR 텍스트 추출
### 3.4 `POST /api/ner-extract` — NER 엔티티 추출
### 3.5 `GET /download/{request_id}` · `GET /health`

v2 명세와 동일합니다(§2.2~2.5). `/health` 응답의 `available_ocr_engines` 에 `alibaba` 포함, 모델 목록은 위 §1 기준.

---

## 4. 멀티모달(이미지 저작물) 처리 상세

이미지 파일은 OCR/NER 대신 **VLM 속성추출** 경로로 처리되어 다음 통합 스키마 필드를 채웁니다.

| 통합 스키마 필드 | 출처(VLM) | 설명 |
|---|---|---|
| `description` | VLM | 이미지 내용 한국어 서술(2~3문장) |
| `work_type` | VLM | 저작물 유형 — **매체 기준** 분류 (사진/영상/어문/음악/미술/건축/도형/컴퓨터프로그램/연극/기타) |
| `keyword` | VLM | 핵심 주제어 5~7개 |
| `digital_format` | 파일 확장자 | JPG/PNG 등 |

- **work_type 판단 원칙:** '무엇을 찍었는가(피사체)'가 아니라 '어떤 매체로 만들어졌는가'로 분류합니다. (예: 건물 사진 → 건축저작물이 아니라 **사진저작물**)
- **VLM이 채우지 못하는 권리정보**(저작권자·공공누리유형·유효기간·상업적이용 등)는 저작물 이미지만으로는 알 수 없으므로, **계약서로부터 상속**받습니다. (§6)
- 계약서+저작물 통합 검사 데모: **`GET /pair`** (단일 제출 → 계약서·저작물 동시 분석 → 계약서 완료 즉시 자동 상속). 각 결과 카드에서 OCR·NER·LLM·통합·상속 전체 결과 열람 가능.

---

## 5. OpenRouter Gemma 4 31B 통합 (VLM 백엔드)

### 5.1 무엇을 / 왜 바꿨나

이미지 저작물의 속성(description·work_type·keyword) 추출에 **Gemma 4 31B** 를 사용하며, 기본 호출을 **OpenRouter API** 로 전환했습니다.

**전환 이유:**
- **운영 안정성:** 자체 GPU 서버(vLLM) 호스팅은 전용 GPU·상시 구동·교차망 방화벽(Tailscale) 관리가 필요해 상시 가용성 확보가 어려움. OpenRouter는 온디맨드 호출로 GPU 운영 부담이 없음.
- **비용 효율:** 유료 기준 입력 $0.12/M · 출력 $0.35/M ≈ **이미지 1장당 약 ₩0.4**, 144K 저작물 전량 ≈ **약 $38**. 테스트 호출은 장당 ~$0.0004 수준.
- **품질:** 사전 비교에서 Gemma 4 31B가 한국어 서술·이미지 내 문자 인식(OCR) 충실도에서 Qwen3-VL 대비 우위. (근거: 자체 VLM 비교 리포트)
- **폴백 이중화:** OpenRouter 장애 시 자체 vLLM Gemma → Qwen3-VL(DashScope) 로 자동 폴백하여 단일 장애점 제거.

### 5.2 백엔드 폴백 체인 (`vlm_prefer="gemma"` 기본)

```
1순위: Gemma 4 31B  @ OpenRouter API        (google/gemma-4-31b-it)
2순위: Gemma        @ 자체 vLLM 서버          (GEMMA_URL)
3순위: Qwen3-VL-235B @ Alibaba DashScope     (백업)
```
각 요청 시작 시 상위 백엔드부터 헬스체크(authed ping)하여 첫 번째 가용 백엔드를 사용하고, 호출 실패 시 다음 순위로 폴백합니다. 응답의 `vlm_backend` 에 실제 사용 백엔드가 기록됩니다.

### 5.3 환경변수 설정 (`.env`, 프로젝트 루트)

```bash
# OpenRouter (VLM 1순위)
OPENROUTER_API_KEY=<발급받은 키>            # sk-or-v1-********  ← §5.4 참조
OPENROUTER_VLM_MODEL=google/gemma-4-31b-it # 유료. 개발용 무료 티어는 ':free' 접미사(google/gemma-4-31b-it:free, 20/min·50/day)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1   # (기본값, 생략 가능)

# (선택) 자체 Gemma 서버 — 2순위 폴백. 미설정 시 자동 skip
GEMMA_URL=http://<자체서버>:8001/v1

# Qwen3-VL 백업 — 3순위. 기존 DashScope 키 재사용
DASHSCOPE_API_KEY=<DashScope 키>
DASHSCOPE_BASE_URL=<DashScope 엔드포인트>
```

- OpenRouter 키 발급: <https://openrouter.ai/keys> (가입 후 즉시 발급, 잔액/크레딧 충전으로 rate limit 상향).
- 모델 식별자: `google/gemma-4-31b-it` (유료), `google/gemma-4-31b-it:free` (개발용 무료, 20 req/min · 50 req/day, $10 충전 시 1,000/day).

### 5.4 API 키 값 전달 (보안)

> ⚠️ **API 키 값은 본 문서 및 GitHub 저장소(공개)에 포함하지 않습니다.** 키가 공개 저장소에 커밋되면 자동 스캐너에 의해 즉시 무효화되고, 크레딧이 소진될 수 있습니다.
>
> **연동 시 키 확보 방법 (택1):**
> 1. **(권장) 무하유 측 자체 키 발급** — 최종 시스템은 무하유가 운영하므로 <https://openrouter.ai/keys> 에서 무하유 계정의 키를 발급해 `.env` 의 `OPENROUTER_API_KEY` 에 설정. 비용/한도를 자체 관리 가능.
> 2. **숭실대 테스트 키 사용** — 통합 테스트용으로 숭실대가 발급한 OpenRouter 키를 **별도 보안 채널(이메일/메신저 직접 전달)** 로 제공. 이 키는 저장소에 커밋 금지.

---

## 6. 계약서 → 저작물 메타데이터 상속

저작물 이미지만으로는 확인 불가한 **권리정보 28필드**(copyright_holder, kogl_type, commercial_use, valid_period, parties 등)를 계약서 분석 결과로부터 저작물 레코드에 병합합니다. 시각 필드(description/work_type/keyword)는 절대 덮어쓰지 않으며, 저작물 자체 값이 있으면 유지합니다. 다건 계약의 경우 파일명·설명 토큰으로 저작물 제목을 매칭(`CONTRACT_INHERITED` 0.8 / `CONTRACT_AMBIGUOUS` 0.5).

- 방식 A(순차): `/api/llm-extract` 에 `contract_metadata` 폼 파라미터 전달.
- 방식 B(병렬): 계약서·저작물 동시 분석 후 `/api/apply-inheritance` 로 사후 병합.
- 무하유 20필드 ↔ 우리 67필드 매핑(5개 필드명 상이: work_name↔work_title, keywords↔keyword, property_rights↔economic_rights, validity_period↔valid_period, non_protected_work↔unprotected_work)은 **별도 문서 `docs/CONTRACT_WORK_INHERITANCE_SPEC.md`** 참조.

---

## 7. 에러 응답

모든 엔드포인트는 실패 시 아래 형식으로 응답합니다:

```json
{ "success": false, "error": "에러 메시지", "request_id": "20260723_140312_882110" }
```

**주요 에러 코드:** `400`(지원하지 않는 파일 형식 / JSON 파싱 실패 / 필수 파라미터 누락), `413`(요청 본문 초과), `500`(서버 내부 오류 — OCR/모델 호출 실패 등).
