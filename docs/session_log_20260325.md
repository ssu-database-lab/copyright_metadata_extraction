# Session Log — 2026-03-25

## Session Overview
Two connected sessions covering: API 명세서 작성, 메타데이터 매핑, 배포 준비, 무하유 연동

---

## Part 1: API 명세서 & 메타데이터 (Previous Session)

### 1. API 명세서 Word 파일 생성
- `docs/API_명세서.md` → `docs/API_명세서.docx` 변환
- 서버 실행하여 스크린샷 촬영

### 2. 피드백 반영 (4차례)

**피드백 1: 응답 파라미터 설명 추가**
- 5개 엔드포인트 전체에 응답 파라미터 테이블 추가
- 중첩 객체(consolidation_summary, consolidation_decisions, pages[], steps) 서브 테이블 포함

**피드백 2: 통합검증결과 분리**
- 3.5절(통합검증결과)을 별도 문서로 분리
- `docs/통합검증_결과_상세.md` + `.docx` 생성
- 판정 유형 설명, 신뢰도 범위, 판정 근거 컬럼, 효과성 분석 포함

**피드백 3: 27 필드 vs 50 필드 수정**
- 실제 구현은 27개 필드 (동의서 기준 consolidation 결과)
- 27개 필드 목록: consent_type, consenter_name, consenter_phone, consenter_address, collector_name, collector_company, collector_department, collection_purpose, collection_items, usage_purpose, retention_period, third_party_provision, marketing_consent, sensitive_info_consent, consent_date, consent_signature, withdrawal_method, legal_representative, representative_relationship, data_categories, processing_method, consent_withdrawal_procedure, refusal_disadvantage, data_protection_officer, emergency_contact, collection_source, automated_decision

**피드백 4: 비용 정보 추가**
- Alibaba Cloud DashScope API 과금 정보 추가
- Qwen3-VL-235B: $0.40/$1.60 per 1M tokens (input/output)
- Qwen3-Next-80B: $0.15/$1.20 per 1M tokens
- 문서당 약 $0.008

### 3. 메타데이터 매핑 작업

**20개 필수 메타데이터 라벨 (원본 기준):**
1. 저작물명
2. 유형
3. 디지털화 형태
4. 설명
5. 주제어
6. 언어
7. 제작일
8. 계약서
9. 저작권자
10. 공동저작자
11. 저작인접권자
12. 공개유형
13. 저작물성
14. 비보호저작물
15. 업무상저작물
16. 상업적 이용허락
17. 저작재산권
18. 공동저작자 동의
19. 유효기간
20. 초상권

**23개 NER 엔티티 (학습 현황):**
- 학습 완료 (16개): NAME, PHONE, ADDRESS, DATE, COMPANY, EMAIL, POSITION, CONTRACT_TYPE, CONSENT_TYPE, RIGHT_INFO, MONEY, PERIOD, PROJECT_NAME, LAW_REFERENCE, ID_NUM, TITLE
- 학습 미완료 (7개): URL, DESCRIPTION, TYPE, STATUS, DEPARTMENT, LANGUAGE, QUANTITY

**생성된 파일:**
- `docs/메타데이터_추출결과_27필드.xlsx` — 27개 필드 O/X 추출 결과
- `docs/메타데이터_추출결과_20항목.xlsx` — 20개 라벨 ↔ NER/LLM 매핑
- `docs/통합_메타데이터_추출현황.xlsx` — 56개 필드 통합 (필수 20 + 확장 36)
- `docs/메타데이터_확장_설명.md` + `.docx` — 확장 사유 설명
- `docs/메타데이터_확장_설명_요약.md` + `.docx` — 요약 버전

### 4. 이메일 업데이트
- `docs/email_draft_muhayu_reply.md` 섹션 3: "50개 필드" → "필수 20개 + 확장 36개 = 총 56개 필드" 수정
- 섹션 4.1: API 호출 비용 정보 추가
- Word 파일 재생성

### 5. 이해관계자 이메일 분석
- 왕일 (무하유): 프로토타입 연동 준비 요청
- 장세영: 데모 계정 공유, 무하유 프론트엔드 HTML 전달
- 왕일: kogl-classifier 프론트엔드 리포 공유

### 6. 배포 설정 (ngrok)
- ngrok 설치: `~/bin/ngrok`
- 인증 토큰 설정 완료
- 터널 URL: `https://cottony-speedingly-adalynn.ngrok-free.dev`
- 사용자가 별도 터미널에서 서버/ngrok 직접 관리

### 7. End-to-End 테스트
- 첫 시도: `model_name=solar-ko` → 로컬 모델 미설치 오류
- 두 번째: `model_name=alibaba-qwen3-next-80b-a3b-instruct` → **DashScope Arrearage 오류**
- 원인: Alibaba Cloud 계정 결제 문제 (미납)

---

## Part 2: 배포 긴급 대응 (Current Session — 2026-03-25)

### 왕일의 5가지 요청 사항

**1. API 서버 접속 정보**
- 테스트 가능한 서버 주소 (IP:포트 또는 도메인)
- 인증 방식 (API Key 등)

**2. DashScope API 키**
- OCR/LLM 호출용 키 공유 또는 별도 발급 여부

**3. 연동 흐름 확인**
- 무하유 → POST /api/llm-extract → OCR + 메타데이터 + 통합검증 JSON → OCR 텍스트를 HMC에 전달 → 공공누리 분류

**4. 테스트용 샘플**
- 샘플 PDF 1~2건 + 예상 응답 JSON

**5. 서버 가동 일정**
- 테스트 가능 시점

### Priority Table

| Priority | Task | Urgency |
|----------|------|---------|
| 1 | 서버 배포 — 공개 접근 가능한 서버 확보 | Immediate |
| 2 | Reply email — 왕일에게 5개 질문 답변 | Immediate |
| 3 | Sample JSON 준비 | Immediate |
| 4 | Frontend 통합 (kogl-classifier) | This week |
| 5 | API 응답 필드명 정렬 | This week |
| 6 | 테스트 PDF 준비 | This week |
| 7 | DashScope 키 결정 | Decision needed |

### 긴급 상황: 내일(3/26)까지 API 구축 완료 필요

연구원 간 대화에서 확인된 사항:
- **최선:** 무하유에게서 서버를 받는 것 (예상대로 진행될 것)
- **차선:** 선배 컴퓨터 24시간 상시 가동 (어려울 것으로 판단)

### 장세영 연구원 전달용 3가지 답변

**1. 서버 스펙 — GPU 불필요**

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| CPU | 2코어 | 4코어 |
| RAM | 4GB | 8GB |
| 디스크 | 20GB | 50GB |
| GPU | 불필요 | 불필요 |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |
| Python | 3.9+ | 3.10+ |

이유: OCR(Qwen3-VL-235B)과 LLM(Qwen3-Next-80B)은 Alibaba Cloud API 호출. 로컬 NER 모델(KLUE-RoBERTa-Large)은 CPU 가능.

**2. 서버 설치 소요 시간 — 약 30분**

| 단계 | 소요 시간 |
|------|----------|
| Python + 의존성 설치 | ~15분 |
| 코드 배포 + .env 설정 | ~5분 |
| NER 모델 다운로드 | ~5분 |
| 서버 실행 + 테스트 | ~5분 |

배포 스크립트 미리 준비 시 15~20분 단축 가능.

**3. 필요한 서비스**
- Python 3.9+ (pip 포함)
- 외부 네트워크 접근 (DashScope API 호출용 outbound HTTPS)
- 포트 개방 (5000 또는 80/443)
- Git (코드 배포용, 선택)
- Docker, DB, GPU, Nginx — **불필요**

### NER 모델 GPU 관련 확인

GPU는 선택사항. CPU fallback 코드 내장:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

| 모델 | 파라미터 | CPU RAM |
|------|---------|---------|
| mBERT-base (기본) | ~110M | ~1.2GB |
| KLUE-RoBERTa-Large | ~340M | ~3.5GB |
| XLM-RoBERTa-Large | ~550M | ~5.5GB |

### 100명 동시 접속 시 분석

- OCR/LLM/Consolidation: Alibaba Cloud API → 서버 부하 없음
- **NER만 로컬 처리** → 병목 지점
- 프로토타입(3-5명): 현재 4core/8GB 충분
- 운영(100명): 8+ core / 16GB / GPU T4 권장

### Docker 필요 여부
- 프로토타입: **불필요** (`git clone` → `pip install` → `uvicorn` 3줄로 배포)
- 운영 단계: 유용 (자동 배포, 서비스 격리 등)

---

## 미해결 사항

1. **DashScope Arrearage 오류** — 미납 요금 결제 필요 (곧 처리 예정)
2. **왕일 답변 이메일** — 서버 정보 확정 후 발송
3. **배포 스크립트** — 서버 받기 전 미리 준비 필요
4. **Frontend 통합** — kogl-classifier 클론 + FastAPI 마운트
5. **Schema alignment** — 공식 50개 필드 스펙 매핑 (장기)

---

## 생성된 주요 파일 목록

| 파일 | 설명 |
|------|------|
| `docs/API_명세서.md` + `.docx` | API 명세서 (피드백 4차 반영) |
| `docs/통합검증_결과_상세.md` + `.docx` | 통합검증 결과 별도 문서 |
| `docs/메타데이터_추출결과_27필드.xlsx` | 27필드 O/X 추출 결과 |
| `docs/메타데이터_추출결과_20항목.xlsx` | 20개 라벨 ↔ NER/LLM 매핑 |
| `docs/통합_메타데이터_추출현황.xlsx` | 56필드 통합 현황 |
| `docs/메타데이터_확장_설명.md` + `.docx` | 확장 사유 설명 |
| `docs/메타데이터_확장_설명_요약.md` + `.docx` | 확장 사유 요약 |
| `docs/email_draft_muhayu_reply.md` + `.docx` | 무하유 답신 이메일 |
| `docs/session_log_20260325.md` | 이 세션 로그 |
