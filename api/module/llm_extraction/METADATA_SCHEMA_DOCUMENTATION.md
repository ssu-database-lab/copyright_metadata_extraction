# LLM 메타데이터 추출 스키마 문서

이 문서는 LLM 메타데이터 추출 시스템에서 사용하는 모든 JSON 스키마의 필드에 대한 상세 설명을 제공합니다.

## 목차

1. [계약서 (Contract) Schema](#1-계약서-contract-schema)
2. [계약서 Enhanced Schema](#2-계약서-enhanced-schema)
3. [동의서 (Consent) Schema](#3-동의서-consent-schema)
4. [동의서 Enhanced Schema](#4-동의서-enhanced-schema)
5. [기타문서 (General) Schema](#5-기타문서-general-schema)
6. [기타문서 Enhanced Schema](#6-기타문서-enhanced-schema)
7. [저작재산권 양도동의서 Schema](#7-저작재산권-양도동의서-schema)
8. [공공저작물 자유이용허락 동의서 Schema](#8-공공저작물-자유이용허락-동의서-schema)
9. [공통 필드 (Parties)](#9-공통-필드-parties)
10. [체크박스 처리 정보](#10-체크박스-처리-정보)
11. [종합 필드 참조표 (Comprehensive Field Reference)](#11-종합-필드-참조표-comprehensive-field-reference)

---

## 1. 계약서 (Contract) Schema

### 기본 정보 필드

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `contract_type` | string | ✅ | - | 계약서 유형 | Contract type | "저작재산권 비독점적 이용허락 계약서" |
| `rights_holder` | string | ✅ | - | 권리자 (저작자 및 저작권 이용허락자) | Rights holder (author and copyright licensor) | "홍길동" |
| `user` | string | ✅ | - | 이용자 (저작권 이용자) | User (copyright user) | "국립생태원 멸종위기종복원센터" |
| `work_title` | string | ⚪ | - | 저작물 제목 | Work title | "멸종위기 야생생물 대국민 온라인 홍보물 제작" |
| `work_category` | string | ⚪ | - | 저작물 종별 | Work category | "어문저작물, 사진저작물" |

### 권리 및 권한 필드

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `granted_rights` | array | ⚪ | Array of strings | 허락된 권리 | Granted rights | `["복제권", "공중송신권"]` |

### 계약 조건 필드

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `contract_purpose` | string | ⚪ | - | 계약의 목적 | Contract purpose | "저작물 이용허락" |
| `contract_duration` | string | ⚪ | - | 계약 기간 | Contract duration | "2024년 1월 1일부터 2024년 12월 31일까지" |
| `special_terms` | array | ⚪ | Array of strings | 특별 약정 사항 | Special terms | `["비상업적 이용만 허용", "출처 표기 필수"]` |
| `termination_conditions` | array | ⚪ | Array of strings | 계약 해지 조건 | Termination conditions | `["상대방의 중대한 위약 시", "30일 전 서면 통지"]` |

### 날짜 필드

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `signature_date` | string | ⚪ | YYYY-MM-DD | 계약 체결일 | Contract signature date | "2024-01-15" |
| `effective_date` | string | ⚪ | YYYY-MM-DD | 계약 효력 발생일 | Contract effective date | "2024-01-15" |
| `expiration_date` | string | ⚪ | YYYY-MM-DD | 계약 만료일 | Contract expiration date | "2024-12-31" |

### 금액 정보 필드

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `payment_amount` | number | ⚪ | 숫자만 | 지급 금액 | Payment amount | `1000000` |
| `payment_currency` | string | ⚪ | - | 통화 | Currency | "원" |

### 당사자 정보

`parties` 배열의 각 객체는 [공통 필드 (Parties)](#9-공통-필드-parties) 섹션을 참조하세요.

---

## 2. 계약서 Enhanced Schema

Enhanced Schema는 기본 Schema의 모든 필드를 포함하며, 추가로 체크박스 기반 필드와 상세 정보를 제공합니다.

### 기본 정보 필드

기본 Schema와 동일합니다. [계약서 (Contract) Schema](#1-계약서-contract-schema) 섹션을 참조하세요.

### 권리 및 권한 필드 (체크박스 기반)

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example | Notes |
|------------|------|----------|--------|---------------------|----------------------|---------|-------|
| `granted_rights.reproduction_right` | boolean | ✅ | true/false | 복제권 (체크박스 상태) | Reproduction right (checkbox state) | `true` | 체크박스 패턴 자동 감지 |
| `granted_rights.performance_right` | boolean | ✅ | true/false | 공연권 (체크박스 상태) | Performance right (checkbox state) | `false` | 체크박스 패턴 자동 감지 |
| `granted_rights.broadcasting_right` | boolean | ✅ | true/false | 공중송신권 (체크박스 상태) | Broadcasting right (checkbox state) | `true` | 체크박스 패턴 자동 감지 |
| `granted_rights.exhibition_right` | boolean | ✅ | true/false | 전시권 (체크박스 상태) | Exhibition right (checkbox state) | `false` | 체크박스 패턴 자동 감지 |
| `granted_rights.distribution_right` | boolean | ✅ | true/false | 배포권 (체크박스 상태) | Distribution right (checkbox state) | `false` | 체크박스 패턴 자동 감지 |
| `granted_rights.rental_right` | boolean | ✅ | true/false | 대여권 (체크박스 상태) | Rental right (checkbox state) | `false` | 체크박스 패턴 자동 감지 |
| `granted_rights.derivative_work_right` | boolean | ✅ | true/false | 2차적저작물작성권 (체크박스 상태) | Derivative work right (checkbox state) | `false` | 체크박스 패턴 자동 감지 |

**체크박스 패턴 인식:**
- 체크됨: `📧`, `☑`, `✓`, `■`, `●`, `◼`, `◉`
- 체크 안됨: `☐`, `□`, `○`, `◯`, `◻`, `◦`

### 계약 조건 필드 (체크박스 기반)

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `contract_terms.contract_type_selection.exclusive` | boolean | ⚪ | true/false | 독점적 계약 (체크박스) | Exclusive contract (checkbox) | `false` |
| `contract_terms.contract_type_selection.non_exclusive` | boolean | ⚪ | true/false | 비독점적 계약 (체크박스) | Non-exclusive contract (checkbox) | `true` |
| `contract_terms.payment_terms.prepaid` | boolean | ⚪ | true/false | 선불 (체크박스) | Prepaid (checkbox) | `true` |
| `contract_terms.payment_terms.postpaid` | boolean | ⚪ | true/false | 후불 (체크박스) | Postpaid (checkbox) | `false` |
| `contract_terms.payment_terms.installment` | boolean | ⚪ | true/false | 할부 (체크박스) | Installment (checkbox) | `false` |
| `contract_terms.renewal_options.auto_renewal` | boolean | ⚪ | true/false | 자동갱신 (체크박스) | Auto renewal (checkbox) | `false` |
| `contract_terms.renewal_options.manual_renewal` | boolean | ⚪ | true/false | 수동갱신 (체크박스) | Manual renewal (checkbox) | `true` |

### 체크박스 처리 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `checkbox_info.pattern_detected` | string | ⚪ | enum | 감지된 체크박스 패턴 | Detected checkbox pattern | `"pattern_b"` |
| `checkbox_info.extraction_confidence` | number | ⚪ | 0.0-1.0 | 체크박스 추출 신뢰도 | Checkbox extraction confidence | `0.95` |
| `checkbox_info.checkbox_fields_found` | array | ⚪ | Array of strings | 발견된 체크박스 필드들 | Found checkbox fields | `["reproduction_right", "broadcasting_right"]` |

**패턴 유형:**
- `pattern_a`: 📧/☐
- `pattern_b`: ☑/□
- `pattern_c`: ✓/○
- `pattern_d`: ■/□

---

## 3. 동의서 (Consent) Schema

### 기본 정보 필드

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `consent_type` | string | ✅ | - | 동의서 유형 | Consent type | "개인정보 수집 및 이용 동의서" |
| `data_controller` | string | ✅ | - | 개인정보 처리자 (기관명) | Data controller (institution name) | "국립생태원" |
| `data_subject` | string | ⚪ | - | 정보주체 (동의자) | Data subject (consenter) | "홍길동" |

### 개인정보 수집 및 이용 필드

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `collection_purpose` | string | ⚪ | - | 개인정보 수집 및 이용 목적 | Personal information collection and use purpose | "서비스 제공 및 고객 지원" |
| `collected_data_types` | array | ⚪ | Array of strings | 수집하는 개인정보 항목 | Collected personal information items | `["성명", "전화번호", "주소"]` |
| `retention_period` | string | ⚪ | - | 개인정보 보유 및 이용 기간 | Personal information retention and use period | "서비스 이용 기간 동안" |

### 제3자 제공 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `third_party_sharing.recipient` | string | ⚪ | - | 제공받는 자 | Recipient | "마케팅 파트너사" |
| `third_party_sharing.purpose` | string | ⚪ | - | 이용 목적 | Purpose | "마케팅 및 광고" |
| `third_party_sharing.data_types` | array | ⚪ | Array of strings | 제공하는 개인정보 항목 | Provided personal information items | `["성명", "이메일"]` |

### 동의 상태 및 날짜

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `consent_status` | string | ✅ | enum | 동의 여부 | Consent status | `"동의함"` |
| `consent_date` | string | ⚪ | YYYY-MM-DD | 동의일 | Consent date | "2024-01-15" |
| `signature` | string | ⚪ | - | 서명자 정보 | Signer information | "홍길동 (서명)" |

**동의 상태 값:**
- `"동의함"`
- `"동의하지 않음"`
- `null`

### 연락처 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `contact_info.phone` | string | ⚪ | 숫자와 하이픈만 | 연락처 | Contact phone | "010-1234-5678" |
| `contact_info.address` | string | ⚪ | - | 주소 | Address | "서울특별시 강남구 테헤란로 123" |
| `contact_info.email` | string | ⚪ | - | 이메일 | Email | "example@email.com" |

### 권리 및 안내

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `withdrawal_rights` | string | ⚪ | - | 동의 철회 권리에 대한 안내 | Withdrawal rights information | "언제든지 동의를 철회할 수 있습니다" |
| `consequences_of_refusal` | string | ⚪ | - | 동의 거부 시 불이익 | Consequences of refusal | "서비스 이용이 제한될 수 있습니다" |

---

## 4. 동의서 Enhanced Schema

Enhanced Schema는 기본 Schema의 모든 필드를 포함하며, 체크박스 처리 정보를 추가로 제공합니다.

### 기본 정보 필드

기본 Schema와 동일합니다. [동의서 (Consent) Schema](#3-동의서-consent-schema) 섹션을 참조하세요.

### 체크박스 처리 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `checkbox_info.pattern_detected` | string | ⚪ | enum | 감지된 체크박스 패턴 | Detected checkbox pattern | `"pattern_b"` |
| `checkbox_info.extraction_confidence` | number | ⚪ | 0.0-1.0 | 체크박스 추출 신뢰도 | Checkbox extraction confidence | `0.92` |
| `checkbox_info.checkbox_fields_found` | array | ⚪ | Array of strings | 발견된 체크박스 필드들 | Found checkbox fields | `["consent_status"]` |

---

## 5. 기타문서 (General) Schema

### 기본 정보 필드

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `document_type` | string | ✅ | - | 문서 유형 | Document type | "공문서" |
| `title` | string | ✅ | - | 문서 제목 | Document title | "2024년도 사업계획서" |

### 날짜 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `key_dates[].date` | string | ⚪ | YYYY-MM-DD | 중요한 날짜 | Important date | "2024-01-15" |
| `key_dates[].description` | string | ⚪ | - | 날짜 설명 | Date description | "사업 시작일" |

### 금액 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `key_amounts[].amount` | number | ⚪ | 숫자만 | 중요한 금액 | Important amount | `5000000` |
| `key_amounts[].currency` | string | ⚪ | - | 통화 | Currency | "원" |
| `key_amounts[].description` | string | ⚪ | - | 금액 설명 | Amount description | "연간 예산" |

### 문서 내용

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `main_content` | string | ⚪ | - | 문서의 주요 내용 요약 | Main content summary | "2024년도 사업계획에 대한 상세 내용..." |
| `important_terms` | array | ⚪ | Array of strings | 중요한 조항이나 조건들 | Important terms or conditions | `["예산 집행 시 사전 승인 필요", "분기별 보고 의무"]` |

---

## 6. 기타문서 Enhanced Schema

Enhanced Schema는 기본 Schema의 모든 필드를 포함하며, 체크박스 데이터와 처리 정보를 추가로 제공합니다.

### 기본 정보 필드

기본 Schema와 동일합니다. [기타문서 (General) Schema](#5-기타문서-general-schema) 섹션을 참조하세요.

### 체크박스 데이터

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `checkbox_data.status_indicators` | object | ⚪ | - | 상태 표시 (승인, 대기, 완료 등) | Status indicators (approval, pending, completed) | `{"approved": true, "pending": false}` |
| `checkbox_data.priority_levels` | object | ⚪ | - | 우선순위 (높음, 보통, 낮음 등) | Priority levels (high, medium, low) | `{"high": true, "medium": false}` |
| `checkbox_data.category_selections` | object | ⚪ | - | 카테고리 선택 (유형별 분류) | Category selections (type-based classification) | `{"type_a": true, "type_b": false}` |
| `checkbox_data.approval_states` | object | ⚪ | - | 승인 상태 (승인, 거부, 검토중 등) | Approval states (approved, rejected, under review) | `{"approved": true, "rejected": false}` |
| `checkbox_data.service_options` | object | ⚪ | - | 서비스 옵션 (기본, 프리미엄, 기업 등) | Service options (basic, premium, enterprise) | `{"premium": true, "basic": false}` |
| `checkbox_data.contact_preferences` | object | ⚪ | - | 연락처 선호도 (이메일, 전화, SMS 등) | Contact preferences (email, phone, SMS) | `{"email": true, "phone": false}` |

### 체크박스 처리 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `checkbox_info.pattern_detected` | string | ⚪ | enum | 감지된 체크박스 패턴 | Detected checkbox pattern | `"pattern_c"` |
| `checkbox_info.extraction_confidence` | number | ⚪ | 0.0-1.0 | 체크박스 추출 신뢰도 | Checkbox extraction confidence | `0.88` |
| `checkbox_info.checkbox_fields_found` | array | ⚪ | Array of strings | 발견된 체크박스 필드들 | Found checkbox fields | `["status_indicators", "approval_states"]` |

---

## 7. 저작재산권 양도동의서 Schema

### 기본 정보 필드

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `document_type` | string | ✅ | - | 문서 유형 | Document type | "저작재산권 양도동의서" |
| `document_title` | string | ⚪ | - | 문서 제목 | Document title | "저작재산권 양도동의서" |
| `work_category` | string | ⚪ | - | 작품 카테고리 | Work category | "출판도서" |

### 저작물 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `work_info.work_title` | string | ✅ | - | 저작물 제목 | Work title | "세계속담" |
| `work_info.work_subtitle` | string | ⚪ | - | 저작물 부제목 | Work subtitle | "우리옛이야기" |
| `work_info.work_series` | string | ⚪ | - | 작품 시리즈 | Work series | "세계속담, 우리옛이야기" |
| `work_info.publication_year` | string | ⚪ | - | 출판년도 | Publication year | "2023" |
| `work_info.work_type` | string | ✅ | enum | 저작물 유형 | Work type | `"도서"` |

**저작물 유형 값:**
- `"도서"`
- `"음악"`
- `"미술"`
- `"영상"`
- `"기타"`

### 저작재산권 양도 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `copyright_transfer.transfer_type` | string | ✅ | enum | 양도 유형 | Transfer type | `"전체양도"` |
| `copyright_transfer.transfer_scope.reproduction_right` | boolean | ✅ | true/false | 복제권 | Reproduction right | `true` |
| `copyright_transfer.transfer_scope.performance_right` | boolean | ✅ | true/false | 공연권 | Performance right | `false` |
| `copyright_transfer.transfer_scope.broadcasting_right` | boolean | ✅ | true/false | 공중송신권 | Broadcasting right | `true` |
| `copyright_transfer.transfer_scope.exhibition_right` | boolean | ✅ | true/false | 전시권 | Exhibition right | `false` |
| `copyright_transfer.transfer_scope.distribution_right` | boolean | ✅ | true/false | 배포권 | Distribution right | `false` |
| `copyright_transfer.transfer_scope.rental_right` | boolean | ✅ | true/false | 대여권 | Rental right | `false` |
| `copyright_transfer.transfer_scope.derivative_work_right` | boolean | ✅ | true/false | 2차적저작물작성권 | Derivative work right | `false` |
| `copyright_transfer.transfer_scope.moral_rights` | boolean | ✅ | true/false | 인격권 | Moral rights | `false` |
| `copyright_transfer.transfer_conditions` | array | ⚪ | Array of strings | 양도 조건 | Transfer conditions | `["비상업적 이용만 허용"]` |
| `copyright_transfer.compensation.amount` | number | ⚪ | 숫자만 | 보상 금액 | Compensation amount | `500000` |
| `copyright_transfer.compensation.currency` | string | ⚪ | - | 통화 | Currency | "원" |
| `copyright_transfer.compensation.payment_method` | string | ⚪ | - | 지급 방법 | Payment method | "계좌이체" |
| `copyright_transfer.compensation.payment_schedule` | string | ⚪ | - | 지급 일정 | Payment schedule | "계약 체결 후 30일 이내" |

**양도 유형 값:**
- `"전체양도"`
- `"부분양도"`
- `"이용허락"`

### 공공누리 라이선스 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `public_nuri_license.nuri_type` | string | ⚪ | enum | 공공누리 유형 | Public Nuri license type | `"제1유형"` |
| `public_nuri_license.license_conditions.attribution_required` | boolean | ⚪ | true/false | 저작자표시 | Attribution required | `true` |
| `public_nuri_license.license_conditions.commercial_use` | boolean | ⚪ | true/false | 상업적이용 | Commercial use | `false` |
| `public_nuri_license.license_conditions.modification_allowed` | boolean | ⚪ | true/false | 변경허용 | Modification allowed | `true` |
| `public_nuri_license.license_conditions.share_alike` | boolean | ⚪ | true/false | 동일조건변경허락 | Share alike | `false` |
| `public_nuri_license.license_duration` | string | ⚪ | - | 라이선스 기간 | License duration | "저작권 보호기간 동안" |

**공공누리 유형 값:**
- `"제1유형"`
- `"제2유형"`
- `"제3유형"`
- `"제4유형"`

### 동의 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `consent_info.consent_status` | string | ✅ | enum | 동의 상태 | Consent status | `"동의함"` |
| `consent_info.consent_date` | string | ✅ | YYYY-MM-DD | 동의일 | Consent date | "2024-01-15" |
| `consent_info.consent_scope` | array | ⚪ | Array of strings | 동의 범위 | Consent scope | `["전체 저작재산권", "복제권 및 공중송신권"]` |
| `consent_info.withdrawal_conditions` | string | ⚪ | - | 동의 철회 조건 | Withdrawal conditions | "계약 체결 후 7일 이내" |

**동의 상태 값:**
- `"동의함"`
- `"동의하지 않음"`
- `"조건부동의"`

### 계약 조건

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `contract_terms.effective_date` | string | ⚪ | YYYY-MM-DD | 계약 효력 발생일 | Contract effective date | "2024-01-15" |
| `contract_terms.expiration_date` | string | ⚪ | YYYY-MM-DD | 계약 만료일 | Contract expiration date | "2024-12-31" |
| `contract_terms.territory` | string | ⚪ | - | 적용 지역 | Applicable territory | "대한민국 전역" |
| `contract_terms.language` | string | ⚪ | - | 적용 언어 | Applicable language | "한국어" |
| `contract_terms.special_conditions` | array | ⚪ | Array of strings | 특별 조건 | Special conditions | `["출처 표기 필수", "수정 금지"]` |
| `contract_terms.termination_conditions` | array | ⚪ | Array of strings | 계약 해지 조건 | Termination conditions | `["상대방의 중대한 위약 시"]` |

### 체크박스 처리 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `checkbox_info.pattern_detected` | string | ⚪ | enum | 감지된 체크박스 패턴 | Detected checkbox pattern | `"pattern_b"` |
| `checkbox_info.extraction_confidence` | number | ⚪ | 0.0-1.0 | 체크박스 추출 신뢰도 | Checkbox extraction confidence | `0.93` |
| `checkbox_info.checkbox_fields_found` | array | ⚪ | Array of strings | 발견된 체크박스 필드들 | Found checkbox fields | `["transfer_scope", "license_conditions"]` |

---

## 8. 공공저작물 자유이용허락 동의서 Schema

### 기본 정보 필드

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `consent_type` | string | ✅ | - | 동의서 유형 | Consent type | "공공저작물 자유이용허락 동의서" |
| `data_controller` | string | ⚪ | - | 개인정보 처리자 (기관명) | Data controller (institution name) | "국립극단" |
| `data_subject` | string | ⚪ | - | 정보주체 (동의자) | Data subject (consenter) | "홍길동" |

### 저작물 표시 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `work_display.work_names` | array | ✅ | Array of strings | 저작물명 목록 | Work names list | `["오페라 작품", "연극 작품"]` |
| `work_display.institution` | string | ⚪ | - | 기관명 | Institution name | "국립극단" |
| `work_display.work_category` | string | ✅ | - | 저작물 종별 | Work category | "우대미술, 의상디자인" |
| `work_display.work_details.stage` | boolean | ⚪ | true/false | 무대 | Stage | `true` |
| `work_display.work_details.lighting` | boolean | ⚪ | true/false | 장치 | Lighting | `true` |
| `work_display.work_details.costume` | boolean | ⚪ | true/false | 의상 | Costume | `true` |
| `work_display.work_details.accessories` | boolean | ⚪ | true/false | 장신구 | Accessories | `false` |
| `work_display.work_details.props` | boolean | ⚪ | true/false | 소품 | Props | `false` |
| `work_display.work_details.meditation` | boolean | ⚪ | true/false | 명상 | Meditation | `false` |
| `work_display.work_details.sound` | boolean | ⚪ | true/false | 음향 | Sound | `true` |
| `work_display.work_details.video` | boolean | ⚪ | true/false | 영상 | Video | `false` |
| `work_display.work_details.lighting_equipment` | boolean | ⚪ | true/false | 조명 | Lighting equipment | `true` |
| `work_display.detailed_info` | string | ⚪ | - | 상세정보 | Detailed information | "별지 저작물 목록 참조" |

### 저작재산권 이용허락 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `copyright_license.license_purpose` | string | ✅ | - | 이용허락 목적 | License purpose | "공공저작물 자유이용" |
| `copyright_license.licensing_institution` | string | ✅ | - | 이용허락 기관 | Licensing institution | "국립극장" |
| `copyright_license.granted_rights.reproduction_right` | boolean | ✅ | true/false | 복제권 (목제권 포함) | Reproduction right (including 목제권) | `true` |
| `copyright_license.granted_rights.performance_right` | boolean | ✅ | true/false | 공연권 (공면권 포함) | Performance right (including 공면권) | `true` |
| `copyright_license.granted_rights.broadcasting_right` | boolean | ✅ | true/false | 공중송신권 | Broadcasting right | `true` |
| `copyright_license.granted_rights.exhibition_right` | boolean | ✅ | true/false | 전시권 | Exhibition right | `false` |
| `copyright_license.granted_rights.distribution_right` | boolean | ✅ | true/false | 배포권 | Distribution right | `false` |
| `copyright_license.granted_rights.rental_right` | boolean | ✅ | true/false | 대여권 | Rental right | `false` |
| `copyright_license.granted_rights.derivative_work_right` | boolean | ✅ | true/false | 2차적저작물작성권 | Derivative work right | `false` |
| `copyright_license.license_type` | string | ⚪ | - | 이용허락 유형 | License type | "비독점적" |

**OCR 오류 고려:**
- "목제권" → "복제권"으로 해석
- "공면권" → "공연권"으로 해석

### 공공누리 적용 동의 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `public_nuri_license.license_purpose` | string | ✅ | - | 공공누리 적용 목적 | Public Nuri license purpose | "공공저작물 자유이용" |
| `public_nuri_license.nuri_type` | string | ✅ | enum | 선택된 공공누리 유형 | Selected Public Nuri type | `"제1유형"` |
| `public_nuri_license.available_types` | array | ⚪ | Array of strings | 사용 가능한 공공누리 유형들 | Available Public Nuri types | `["제1유형", "제2유형"]` |
| `public_nuri_license.modification_rights.integrity_right_waiver` | boolean | ⚪ | true/false | 동일성유지권 행사 제안 동의 | Integrity right waiver consent | `true` |
| `public_nuri_license.modification_rights.modification_allowed` | boolean | ⚪ | true/false | 변경 이용 가능 여부 | Modification allowed | `true` |
| `public_nuri_license.modification_rights.conditions` | string | ⚪ | - | 변경 이용 조건 | Modification conditions | "출처 표기 필수" |

**공공누리 유형 값:**
- `"제1유형"`
- `"제2유형"`
- `"제3유형"`
- `"제4유형"`

### 개인정보 제공 동의 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `personal_info_consent.collected_data_types` | array | ✅ | Array of strings | 수집하는 개인정보 항목 | Collected personal information items | `["성명", "전화번호"]` |
| `personal_info_consent.collection_purpose` | string | ✅ | - | 개인정보 수집 이용목적 | Personal information collection purpose | "공공저작물 이용허락 처리" |
| `personal_info_consent.retention_period` | string | ✅ | - | 개인정보 보유, 이용기간 | Personal information retention period | "이용허락 기간 동안" |

### 날짜 및 서명 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `consent_date` | string | ⚪ | YYYY-MM-DD | 동의일 | Consent date | "2024-01-15" |
| `signature` | string | ⚪ | - | 서명자 정보 | Signer information | "홍길동 (서명)" |
| `utilizing_institution` | string | ⚪ | - | 활용기관 | Utilizing institution | "국립극장" |

### 처리 정보

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `processing_info.checkbox_pattern_detected` | string | ⚪ | enum | 감지된 체크박스 패턴 | Detected checkbox pattern | `"pattern_b"` |
| `processing_info.extraction_confidence` | number | ⚪ | 0.0-1.0 | 추출 신뢰도 | Extraction confidence | `0.94` |

---

## 9. 공통 필드 (Parties)

모든 문서 유형에서 사용되는 당사자 정보 필드입니다.

### Parties 배열 필드

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Example |
|------------|------|----------|--------|---------------------|----------------------|---------|
| `parties[].name` | string | ✅ | - | 당사자 이름 또는 회사명 | Party name or company name | "홍길동" |
| `parties[].phone` | string | ⚪ | 숫자와 하이픈만 | 전화번호 | Phone number | "010-1234-5678" |
| `parties[].address` | string | ⚪ | - | 주소 | Address | "서울특별시 강남구 테헤란로 123" |
| `parties[].registration_no` | string | ⚪ | 숫자와 하이픈만 | 사업자등록번호 또는 주민등록번호 | Business registration number or resident registration number | "123-45-67890" |
| `parties[].role` | string | ⚪ | - | 문서에서의 역할 | Role in document | "권리자", "이용자", "증인", "대리인", "저작자", "출판사", "기관", "정보주체", "처리자", "발신자", "수신자" |

**역할 값 (문서 유형별):**
- 계약서: `"권리자"`, `"이용자"`, `"증인"`
- 동의서: `"정보주체"`, `"처리자"`, `"대리인"`
- 저작재산권 양도동의서: `"저작자"`, `"출판사"`, `"기관"`, `"대리인"`, `"증인"`
- 공공저작물 동의서: `"저작자"`, `"활용기관"`
- 기타문서: `"발신자"`, `"수신자"`, `"증인"`, `"대리인"`

---

## 10. 체크박스 처리 정보

### 체크박스 패턴

시스템은 다음 체크박스 패턴을 자동으로 감지하고 처리합니다:

| Pattern ID | 체크됨 표시 | 체크 안됨 표시 | Description |
|------------|------------|---------------|-------------|
| `pattern_a` | 📧 | ☐ | 이메일/체크박스 패턴 |
| `pattern_b` | ☑ | □ | 일반 체크박스 패턴 |
| `pattern_c` | ✓ | ○ | 체크마크/원형 패턴 |
| `pattern_d` | ■ | □ | 사각형/체크박스 패턴 |

### 체크박스 처리 규칙

1. **자동 감지**: 시스템은 문서에서 사용된 체크박스 패턴을 자동으로 감지합니다.
2. **일관성 처리**: 동일한 문서 내에서 일관된 패턴으로 처리합니다.
3. **OCR 오류 고려**: 
   - "목제권" → "복제권"으로 해석
   - "공면권" → "공연권"으로 해석
4. **신뢰도 점수**: 각 체크박스 필드에 대해 0.0-1.0 범위의 신뢰도 점수를 제공합니다.

### 체크박스 필드 구조

모든 체크박스 필드는 `boolean` 타입이며:
- `true`: 체크됨 상태
- `false`: 체크 안됨 상태
- `null`: 정보가 없거나 불분명한 경우

---

## 필드 타입 설명

### 기본 타입

| Type | Description | Example |
|------|-------------|---------|
| `string` | 문자열 | `"홍길동"` |
| `number` | 숫자 (금액은 숫자만, 단위 제외) | `1000000` |
| `boolean` | 불린 값 (true/false) | `true`, `false` |
| `array` | 배열 | `["항목1", "항목2"]` |
| `object` | 객체 (중첩된 구조) | `{"key": "value"}` |

### 특수 형식

| Format | Description | Example |
|--------|-------------|---------|
| `YYYY-MM-DD` | 날짜 형식 (ISO 8601) | `"2024-01-15"` |
| `enum` | 열거형 값 (제한된 값 목록) | `"동의함"`, `"동의하지 않음"` |
| 숫자와 하이픈만 | 전화번호, 사업자등록번호 형식 | `"010-1234-5678"` |

---

## 필수 필드 (Required Fields)

각 문서 유형별 필수 필드:

### 계약서
- `contract_type`
- `rights_holder`
- `user`

### 계약서 Enhanced
- `contract_type`
- `rights_holder`
- `user`
- `granted_rights` (객체)

### 동의서
- `consent_type`
- `data_controller`
- `consent_status`

### 기타문서
- `document_type`
- `title`

### 저작재산권 양도동의서
- `document_type`
- `work_info` (객체, `work_title`, `work_type` 필수)
- `copyright_transfer` (객체, `transfer_type`, `transfer_scope` 필수)
- `consent_info` (객체, `consent_status`, `consent_date` 필수)
- `parties` (배열, 각 객체의 `name`, `role` 필수)

### 공공저작물 자유이용허락 동의서
- `consent_type`
- `work_display` (객체, `work_names`, `work_category` 필수)
- `copyright_license` (객체, `license_purpose`, `licensing_institution`, `granted_rights` 필수)
- `public_nuri_license` (객체, `nuri_type`, `license_purpose` 필수)

---

## 주의사항

1. **null 값 처리**: 정보가 명시적으로 존재하지 않거나 불분명한 경우 `null`을 사용합니다. 추측은 하지 않습니다.

2. **날짜 형식**: 모든 날짜는 `YYYY-MM-DD` 형식으로 변환됩니다.

3. **금액 형식**: 금액은 숫자만 추출하며, 통화 단위는 별도 필드에 저장됩니다.

4. **전화번호 및 등록번호**: 숫자와 하이픈(-)만 포함합니다.

5. **체크박스 처리**: 다양한 체크박스 패턴을 자동으로 감지하고 일관되게 처리합니다.

6. **OCR 오류**: OCR 오류로 인한 잘못된 텍스트를 자동으로 보정합니다 (예: "목제권" → "복제권").

---

## 버전 정보

- **문서 버전**: 1.0.0
- **스키마 버전**: 2.0.0
- **최종 업데이트**: 2024-01-15

---

## 11. 종합 필드 참조표 (Comprehensive Field Reference)

이 섹션은 모든 문서 유형의 모든 필드를 단일 평면 테이블로 제공합니다. 특정 필드를 빠르게 검색하고 참조할 수 있습니다.

### 사용 방법

- **Field Path**: JSON 경로를 사용하여 필드를 찾습니다 (예: `contract_type`, `parties[].name`)
- **Document Types**: 해당 필드가 사용되는 문서 유형을 확인합니다
- **Required**: ✅ 표시는 필수 필드, ⚪ 표시는 선택 필드입니다
- **Format**: 날짜 형식, enum 값, 정규식 패턴 등을 확인합니다

### 필드 목록

| Field Path | Type | Required | Format | Description (한국어) | Description (English) | Document Types | Example | Notes |
|------------|------|----------|--------|---------------------|----------------------|----------------|----------|-------|
| `contract_type` | string | ✅ | - | 계약서 유형 | Contract type | 계약서, 계약서 Enhanced | "저작재산권 비독점적 이용허락 계약서" | - |
| `rights_holder` | string | ✅ | - | 권리자 (저작자 및 저작권 이용허락자) | Rights holder (author and copyright licensor) | 계약서, 계약서 Enhanced | "홍길동" | - |
| `user` | string | ✅ | - | 이용자 (저작권 이용자) | User (copyright user) | 계약서, 계약서 Enhanced | "국립생태원 멸종위기종복원센터" | - |
| `work_title` | string | ⚪ | - | 저작물 제목 | Work title | 계약서, 계약서 Enhanced, 저작재산권 양도동의서 | "멸종위기 야생생물 대국민 온라인 홍보물 제작" | - |
| `work_category` | string | ⚪ | - | 저작물 종별 | Work category | 계약서, 계약서 Enhanced, 저작재산권 양도동의서, 공공저작물 동의서 | "어문저작물, 사진저작물" | - |
| `granted_rights` | array | ⚪ | Array of strings | 허락된 권리 | Granted rights | 계약서 | `["복제권", "공중송신권"]` | 기본 Schema |
| `granted_rights.reproduction_right` | boolean | ✅ | true/false | 복제권 (체크박스 상태) | Reproduction right (checkbox state) | 계약서 Enhanced, 저작재산권 양도동의서, 공공저작물 동의서 | `true` | 체크박스 패턴 자동 감지 |
| `granted_rights.performance_right` | boolean | ✅ | true/false | 공연권 (체크박스 상태) | Performance right (checkbox state) | 계약서 Enhanced, 저작재산권 양도동의서, 공공저작물 동의서 | `false` | 체크박스 패턴 자동 감지 |
| `granted_rights.broadcasting_right` | boolean | ✅ | true/false | 공중송신권 (체크박스 상태) | Broadcasting right (checkbox state) | 계약서 Enhanced, 저작재산권 양도동의서, 공공저작물 동의서 | `true` | 체크박스 패턴 자동 감지 |
| `granted_rights.exhibition_right` | boolean | ✅ | true/false | 전시권 (체크박스 상태) | Exhibition right (checkbox state) | 계약서 Enhanced, 저작재산권 양도동의서, 공공저작물 동의서 | `false` | 체크박스 패턴 자동 감지 |
| `granted_rights.distribution_right` | boolean | ✅ | true/false | 배포권 (체크박스 상태) | Distribution right (checkbox state) | 계약서 Enhanced, 저작재산권 양도동의서, 공공저작물 동의서 | `false` | 체크박스 패턴 자동 감지 |
| `granted_rights.rental_right` | boolean | ✅ | true/false | 대여권 (체크박스 상태) | Rental right (checkbox state) | 계약서 Enhanced, 저작재산권 양도동의서, 공공저작물 동의서 | `false` | 체크박스 패턴 자동 감지 |
| `granted_rights.derivative_work_right` | boolean | ✅ | true/false | 2차적저작물작성권 (체크박스 상태) | Derivative work right (checkbox state) | 계약서 Enhanced, 저작재산권 양도동의서, 공공저작물 동의서 | `false` | 체크박스 패턴 자동 감지 |
| `contract_terms.contract_type_selection.exclusive` | boolean | ⚪ | true/false | 독점적 계약 (체크박스) | Exclusive contract (checkbox) | 계약서 Enhanced | `false` | 체크박스 패턴 자동 감지 |
| `contract_terms.contract_type_selection.non_exclusive` | boolean | ⚪ | true/false | 비독점적 계약 (체크박스) | Non-exclusive contract (checkbox) | 계약서 Enhanced | `true` | 체크박스 패턴 자동 감지 |
| `contract_terms.payment_terms.prepaid` | boolean | ⚪ | true/false | 선불 (체크박스) | Prepaid (checkbox) | 계약서 Enhanced | `true` | 체크박스 패턴 자동 감지 |
| `contract_terms.payment_terms.postpaid` | boolean | ⚪ | true/false | 후불 (체크박스) | Postpaid (checkbox) | 계약서 Enhanced | `false` | 체크박스 패턴 자동 감지 |
| `contract_terms.payment_terms.installment` | boolean | ⚪ | true/false | 할부 (체크박스) | Installment (checkbox) | 계약서 Enhanced | `false` | 체크박스 패턴 자동 감지 |
| `contract_terms.renewal_options.auto_renewal` | boolean | ⚪ | true/false | 자동갱신 (체크박스) | Auto renewal (checkbox) | 계약서 Enhanced | `false` | 체크박스 패턴 자동 감지 |
| `contract_terms.renewal_options.manual_renewal` | boolean | ⚪ | true/false | 수동갱신 (체크박스) | Manual renewal (checkbox) | 계약서 Enhanced | `true` | 체크박스 패턴 자동 감지 |
| `contract_purpose` | string | ⚪ | - | 계약의 목적 | Contract purpose | 계약서, 계약서 Enhanced | "저작물 이용허락" | - |
| `contract_duration` | string | ⚪ | - | 계약 기간 | Contract duration | 계약서, 계약서 Enhanced | "2024년 1월 1일부터 2024년 12월 31일까지" | - |
| `payment_amount` | number | ⚪ | 숫자만 | 지급 금액 | Payment amount | 계약서, 계약서 Enhanced, 저작재산권 양도동의서 | `1000000` | 단위 제외 |
| `payment_currency` | string | ⚪ | - | 통화 | Currency | 계약서, 계약서 Enhanced, 저작재산권 양도동의서 | "원" | - |
| `signature_date` | string | ⚪ | YYYY-MM-DD | 계약 체결일 | Contract signature date | 계약서, 계약서 Enhanced | "2024-01-15" | ISO 8601 형식 |
| `effective_date` | string | ⚪ | YYYY-MM-DD | 계약 효력 발생일 | Contract effective date | 계약서, 계약서 Enhanced, 저작재산권 양도동의서 | "2024-01-15" | ISO 8601 형식 |
| `expiration_date` | string | ⚪ | YYYY-MM-DD | 계약 만료일 | Contract expiration date | 계약서, 계약서 Enhanced, 저작재산권 양도동의서 | "2024-12-31" | ISO 8601 형식 |
| `special_terms` | array | ⚪ | Array of strings | 특별 약정 사항 | Special terms | 계약서, 계약서 Enhanced, 저작재산권 양도동의서 | `["비상업적 이용만 허용", "출처 표기 필수"]` | - |
| `termination_conditions` | array | ⚪ | Array of strings | 계약 해지 조건 | Termination conditions | 계약서, 계약서 Enhanced, 저작재산권 양도동의서 | `["상대방의 중대한 위약 시", "30일 전 서면 통지"]` | - |
| `parties[].name` | string | ✅ | - | 당사자 이름 또는 회사명 | Party name or company name | 모든 문서 유형 | "홍길동" | 공통 필드 |
| `parties[].phone` | string | ⚪ | 숫자와 하이픈만 | 전화번호 | Phone number | 모든 문서 유형 | "010-1234-5678" | 공통 필드, 정규식: `^\d+(-\d+)*$` |
| `parties[].address` | string | ⚪ | - | 주소 | Address | 모든 문서 유형 | "서울특별시 강남구 테헤란로 123" | 공통 필드 |
| `parties[].registration_no` | string | ⚪ | 숫자와 하이픈만 | 사업자등록번호 또는 주민등록번호 | Business registration number or resident registration number | 모든 문서 유형 | "123-45-67890" | 공통 필드, 정규식: `^\d+(-\d+)*$` |
| `parties[].role` | string | ⚪ | - | 문서에서의 역할 | Role in document | 모든 문서 유형 | "권리자", "이용자", "증인" | 공통 필드, 문서 유형별로 다른 값 |
| `parties[].email` | string | ⚪ | - | 이메일 | Email | 저작재산권 양도동의서, 공공저작물 동의서 | "example@email.com" | - |
| `parties[].organization` | string | ⚪ | - | 소속 기관 | Organization | 저작재산권 양도동의서 | "국립생태원" | - |
| `checkbox_info.pattern_detected` | string | ⚪ | enum | 감지된 체크박스 패턴 | Detected checkbox pattern | 계약서 Enhanced, 동의서 Enhanced, 기타문서 Enhanced, 저작재산권 양도동의서 | `"pattern_b"` | enum: pattern_a, pattern_b, pattern_c, pattern_d |
| `checkbox_info.extraction_confidence` | number | ⚪ | 0.0-1.0 | 체크박스 추출 신뢰도 | Checkbox extraction confidence | 계약서 Enhanced, 동의서 Enhanced, 기타문서 Enhanced, 저작재산권 양도동의서 | `0.95` | 범위: 0.0 ~ 1.0 |
| `checkbox_info.checkbox_fields_found` | array | ⚪ | Array of strings | 발견된 체크박스 필드들 | Found checkbox fields | 계약서 Enhanced, 동의서 Enhanced, 기타문서 Enhanced, 저작재산권 양도동의서 | `["reproduction_right", "broadcasting_right"]` | - |
| `consent_type` | string | ✅ | - | 동의서 유형 | Consent type | 동의서, 동의서 Enhanced, 공공저작물 동의서 | "개인정보 수집 및 이용 동의서" | - |
| `data_controller` | string | ✅ | - | 개인정보 처리자 (기관명) | Data controller (institution name) | 동의서, 동의서 Enhanced, 공공저작물 동의서 | "국립생태원" | - |
| `data_subject` | string | ⚪ | - | 정보주체 (동의자) | Data subject (consenter) | 동의서, 동의서 Enhanced, 공공저작물 동의서 | "홍길동" | - |
| `collection_purpose` | string | ⚪ | - | 개인정보 수집 및 이용 목적 | Personal information collection and use purpose | 동의서, 동의서 Enhanced, 공공저작물 동의서 | "서비스 제공 및 고객 지원" | - |
| `collected_data_types` | array | ⚪ | Array of strings | 수집하는 개인정보 항목 | Collected personal information items | 동의서, 동의서 Enhanced, 공공저작물 동의서 | `["성명", "전화번호", "주소"]` | - |
| `retention_period` | string | ⚪ | - | 개인정보 보유 및 이용 기간 | Personal information retention and use period | 동의서, 동의서 Enhanced, 공공저작물 동의서 | "서비스 이용 기간 동안" | - |
| `third_party_sharing.recipient` | string | ⚪ | - | 제공받는 자 | Recipient | 동의서, 동의서 Enhanced | "마케팅 파트너사" | - |
| `third_party_sharing.purpose` | string | ⚪ | - | 이용 목적 | Purpose | 동의서, 동의서 Enhanced | "마케팅 및 광고" | - |
| `third_party_sharing.data_types` | array | ⚪ | Array of strings | 제공하는 개인정보 항목 | Provided personal information items | 동의서, 동의서 Enhanced | `["성명", "이메일"]` | - |
| `consent_status` | string | ✅ | enum | 동의 여부 | Consent status | 동의서, 동의서 Enhanced, 저작재산권 양도동의서 | `"동의함"` | enum: "동의함", "동의하지 않음", "조건부동의", null |
| `consent_date` | string | ⚪ | YYYY-MM-DD | 동의일 | Consent date | 동의서, 동의서 Enhanced, 저작재산권 양도동의서, 공공저작물 동의서 | "2024-01-15" | ISO 8601 형식, 저작재산권 양도동의서는 필수 |
| `signature` | string | ⚪ | - | 서명자 정보 | Signer information | 동의서, 동의서 Enhanced, 공공저작물 동의서 | "홍길동 (서명)" | - |
| `contact_info.phone` | string | ⚪ | 숫자와 하이픈만 | 연락처 | Contact phone | 동의서, 동의서 Enhanced | "010-1234-5678" | 정규식: `^\d+(-\d+)*$` |
| `contact_info.address` | string | ⚪ | - | 주소 | Address | 동의서, 동의서 Enhanced | "서울특별시 강남구 테헤란로 123" | - |
| `contact_info.email` | string | ⚪ | - | 이메일 | Email | 동의서, 동의서 Enhanced | "example@email.com" | - |
| `withdrawal_rights` | string | ⚪ | - | 동의 철회 권리에 대한 안내 | Withdrawal rights information | 동의서, 동의서 Enhanced | "언제든지 동의를 철회할 수 있습니다" | - |
| `consequences_of_refusal` | string | ⚪ | - | 동의 거부 시 불이익 | Consequences of refusal | 동의서, 동의서 Enhanced | "서비스 이용이 제한될 수 있습니다" | - |
| `document_type` | string | ✅ | - | 문서 유형 | Document type | 기타문서, 기타문서 Enhanced, 저작재산권 양도동의서 | "공문서", "저작재산권 양도동의서" | - |
| `document_title` | string | ⚪ | - | 문서 제목 | Document title | 저작재산권 양도동의서 | "저작재산권 양도동의서" | - |
| `title` | string | ✅ | - | 문서 제목 | Document title | 기타문서, 기타문서 Enhanced | "2024년도 사업계획서" | - |
| `key_dates[].date` | string | ⚪ | YYYY-MM-DD | 중요한 날짜 | Important date | 기타문서, 기타문서 Enhanced | "2024-01-15" | ISO 8601 형식 |
| `key_dates[].description` | string | ⚪ | - | 날짜 설명 | Date description | 기타문서, 기타문서 Enhanced | "사업 시작일" | - |
| `key_amounts[].amount` | number | ⚪ | 숫자만 | 중요한 금액 | Important amount | 기타문서, 기타문서 Enhanced | `5000000` | 단위 제외 |
| `key_amounts[].currency` | string | ⚪ | - | 통화 | Currency | 기타문서, 기타문서 Enhanced, 저작재산권 양도동의서 | "원" | - |
| `key_amounts[].description` | string | ⚪ | - | 금액 설명 | Amount description | 기타문서, 기타문서 Enhanced | "연간 예산" | - |
| `main_content` | string | ⚪ | - | 문서의 주요 내용 요약 | Main content summary | 기타문서, 기타문서 Enhanced | "2024년도 사업계획에 대한 상세 내용..." | - |
| `important_terms` | array | ⚪ | Array of strings | 중요한 조항이나 조건들 | Important terms or conditions | 기타문서, 기타문서 Enhanced | `["예산 집행 시 사전 승인 필요", "분기별 보고 의무"]` | - |
| `checkbox_data.status_indicators` | object | ⚪ | - | 상태 표시 (승인, 대기, 완료 등) | Status indicators (approval, pending, completed) | 기타문서 Enhanced | `{"approved": true, "pending": false}` | 체크박스 기반 |
| `checkbox_data.priority_levels` | object | ⚪ | - | 우선순위 (높음, 보통, 낮음 등) | Priority levels (high, medium, low) | 기타문서 Enhanced | `{"high": true, "medium": false}` | 체크박스 기반 |
| `checkbox_data.category_selections` | object | ⚪ | - | 카테고리 선택 (유형별 분류) | Category selections (type-based classification) | 기타문서 Enhanced | `{"type_a": true, "type_b": false}` | 체크박스 기반 |
| `checkbox_data.approval_states` | object | ⚪ | - | 승인 상태 (승인, 거부, 검토중 등) | Approval states (approved, rejected, under review) | 기타문서 Enhanced | `{"approved": true, "rejected": false}` | 체크박스 기반 |
| `checkbox_data.service_options` | object | ⚪ | - | 서비스 옵션 (기본, 프리미엄, 기업 등) | Service options (basic, premium, enterprise) | 기타문서 Enhanced | `{"premium": true, "basic": false}` | 체크박스 기반 |
| `checkbox_data.contact_preferences` | object | ⚪ | - | 연락처 선호도 (이메일, 전화, SMS 등) | Contact preferences (email, phone, SMS) | 기타문서 Enhanced | `{"email": true, "phone": false}` | 체크박스 기반 |
| `work_info.work_title` | string | ✅ | - | 저작물 제목 | Work title | 저작재산권 양도동의서 | "세계속담" | - |
| `work_info.work_subtitle` | string | ⚪ | - | 저작물 부제목 | Work subtitle | 저작재산권 양도동의서 | "우리옛이야기" | - |
| `work_info.work_series` | string | ⚪ | - | 작품 시리즈 | Work series | 저작재산권 양도동의서 | "세계속담, 우리옛이야기" | - |
| `work_info.publication_year` | string | ⚪ | - | 출판년도 | Publication year | 저작재산권 양도동의서 | "2023" | - |
| `work_info.work_type` | string | ✅ | enum | 저작물 유형 | Work type | 저작재산권 양도동의서 | `"도서"` | enum: "도서", "음악", "미술", "영상", "기타" |
| `copyright_transfer.transfer_type` | string | ✅ | enum | 양도 유형 | Transfer type | 저작재산권 양도동의서 | `"전체양도"` | enum: "전체양도", "부분양도", "이용허락" |
| `copyright_transfer.transfer_scope.reproduction_right` | boolean | ✅ | true/false | 복제권 | Reproduction right | 저작재산권 양도동의서 | `true` | 체크박스 기반 |
| `copyright_transfer.transfer_scope.performance_right` | boolean | ✅ | true/false | 공연권 | Performance right | 저작재산권 양도동의서 | `false` | 체크박스 기반 |
| `copyright_transfer.transfer_scope.broadcasting_right` | boolean | ✅ | true/false | 공중송신권 | Broadcasting right | 저작재산권 양도동의서 | `true` | 체크박스 기반 |
| `copyright_transfer.transfer_scope.exhibition_right` | boolean | ✅ | true/false | 전시권 | Exhibition right | 저작재산권 양도동의서 | `false` | 체크박스 기반 |
| `copyright_transfer.transfer_scope.distribution_right` | boolean | ✅ | true/false | 배포권 | Distribution right | 저작재산권 양도동의서 | `false` | 체크박스 기반 |
| `copyright_transfer.transfer_scope.rental_right` | boolean | ✅ | true/false | 대여권 | Rental right | 저작재산권 양도동의서 | `false` | 체크박스 기반 |
| `copyright_transfer.transfer_scope.derivative_work_right` | boolean | ✅ | true/false | 2차적저작물작성권 | Derivative work right | 저작재산권 양도동의서 | `false` | 체크박스 기반 |
| `copyright_transfer.transfer_scope.moral_rights` | boolean | ✅ | true/false | 인격권 | Moral rights | 저작재산권 양도동의서 | `false` | 체크박스 기반 |
| `copyright_transfer.transfer_conditions` | array | ⚪ | Array of strings | 양도 조건 | Transfer conditions | 저작재산권 양도동의서 | `["비상업적 이용만 허용"]` | - |
| `copyright_transfer.compensation.amount` | number | ⚪ | 숫자만 | 보상 금액 | Compensation amount | 저작재산권 양도동의서 | `500000` | 단위 제외 |
| `copyright_transfer.compensation.currency` | string | ⚪ | - | 통화 | Currency | 저작재산권 양도동의서 | "원" | - |
| `copyright_transfer.compensation.payment_method` | string | ⚪ | - | 지급 방법 | Payment method | 저작재산권 양도동의서 | "계좌이체" | - |
| `copyright_transfer.compensation.payment_schedule` | string | ⚪ | - | 지급 일정 | Payment schedule | 저작재산권 양도동의서 | "계약 체결 후 30일 이내" | - |
| `public_nuri_license.nuri_type` | string | ⚪ | enum | 공공누리 유형 | Public Nuri license type | 저작재산권 양도동의서, 공공저작물 동의서 | `"제1유형"` | enum: "제1유형", "제2유형", "제3유형", "제4유형" |
| `public_nuri_license.license_conditions.attribution_required` | boolean | ⚪ | true/false | 저작자표시 | Attribution required | 저작재산권 양도동의서 | `true` | 체크박스 기반 |
| `public_nuri_license.license_conditions.commercial_use` | boolean | ⚪ | true/false | 상업적이용 | Commercial use | 저작재산권 양도동의서 | `false` | 체크박스 기반 |
| `public_nuri_license.license_conditions.modification_allowed` | boolean | ⚪ | true/false | 변경허용 | Modification allowed | 저작재산권 양도동의서 | `true` | 체크박스 기반 |
| `public_nuri_license.license_conditions.share_alike` | boolean | ⚪ | true/false | 동일조건변경허락 | Share alike | 저작재산권 양도동의서 | `false` | 체크박스 기반 |
| `public_nuri_license.license_duration` | string | ⚪ | - | 라이선스 기간 | License duration | 저작재산권 양도동의서 | "저작권 보호기간 동안" | - |
| `consent_info.consent_status` | string | ✅ | enum | 동의 상태 | Consent status | 저작재산권 양도동의서 | `"동의함"` | enum: "동의함", "동의하지 않음", "조건부동의" |
| `consent_info.consent_date` | string | ✅ | YYYY-MM-DD | 동의일 | Consent date | 저작재산권 양도동의서 | "2024-01-15" | ISO 8601 형식, 필수 필드 |
| `consent_info.consent_scope` | array | ⚪ | Array of strings | 동의 범위 | Consent scope | 저작재산권 양도동의서 | `["전체 저작재산권", "복제권 및 공중송신권"]` | - |
| `consent_info.withdrawal_conditions` | string | ⚪ | - | 동의 철회 조건 | Withdrawal conditions | 저작재산권 양도동의서 | "계약 체결 후 7일 이내" | - |
| `contract_terms.effective_date` | string | ⚪ | YYYY-MM-DD | 계약 효력 발생일 | Contract effective date | 저작재산권 양도동의서 | "2024-01-15" | ISO 8601 형식 |
| `contract_terms.expiration_date` | string | ⚪ | YYYY-MM-DD | 계약 만료일 | Contract expiration date | 저작재산권 양도동의서 | "2024-12-31" | ISO 8601 형식 |
| `contract_terms.territory` | string | ⚪ | - | 적용 지역 | Applicable territory | 저작재산권 양도동의서 | "대한민국 전역" | - |
| `contract_terms.language` | string | ⚪ | - | 적용 언어 | Applicable language | 저작재산권 양도동의서 | "한국어" | - |
| `contract_terms.special_conditions` | array | ⚪ | Array of strings | 특별 조건 | Special conditions | 저작재산권 양도동의서 | `["출처 표기 필수", "수정 금지"]` | - |
| `contract_terms.termination_conditions` | array | ⚪ | Array of strings | 계약 해지 조건 | Termination conditions | 저작재산권 양도동의서 | `["상대방의 중대한 위약 시"]` | - |
| `work_display.work_names` | array | ✅ | Array of strings | 저작물명 목록 | Work names list | 공공저작물 동의서 | `["오페라 작품", "연극 작품"]` | - |
| `work_display.institution` | string | ⚪ | - | 기관명 | Institution name | 공공저작물 동의서 | "국립극단" | - |
| `work_display.work_category` | string | ✅ | - | 저작물 종별 | Work category | 공공저작물 동의서 | "우대미술, 의상디자인" | - |
| `work_display.work_details.stage` | boolean | ⚪ | true/false | 무대 | Stage | 공공저작물 동의서 | `true` | 체크박스 기반 |
| `work_display.work_details.lighting` | boolean | ⚪ | true/false | 장치 | Lighting | 공공저작물 동의서 | `true` | 체크박스 기반 |
| `work_display.work_details.costume` | boolean | ⚪ | true/false | 의상 | Costume | 공공저작물 동의서 | `true` | 체크박스 기반 |
| `work_display.work_details.accessories` | boolean | ⚪ | true/false | 장신구 | Accessories | 공공저작물 동의서 | `false` | 체크박스 기반 |
| `work_display.work_details.props` | boolean | ⚪ | true/false | 소품 | Props | 공공저작물 동의서 | `false` | 체크박스 기반 |
| `work_display.work_details.meditation` | boolean | ⚪ | true/false | 명상 | Meditation | 공공저작물 동의서 | `false` | 체크박스 기반 |
| `work_display.work_details.sound` | boolean | ⚪ | true/false | 음향 | Sound | 공공저작물 동의서 | `true` | 체크박스 기반 |
| `work_display.work_details.video` | boolean | ⚪ | true/false | 영상 | Video | 공공저작물 동의서 | `false` | 체크박스 기반 |
| `work_display.work_details.lighting_equipment` | boolean | ⚪ | true/false | 조명 | Lighting equipment | 공공저작물 동의서 | `true` | 체크박스 기반 |
| `work_display.detailed_info` | string | ⚪ | - | 상세정보 | Detailed information | 공공저작물 동의서 | "별지 저작물 목록 참조" | - |
| `copyright_license.license_purpose` | string | ✅ | - | 이용허락 목적 | License purpose | 공공저작물 동의서 | "공공저작물 자유이용" | - |
| `copyright_license.licensing_institution` | string | ✅ | - | 이용허락 기관 | Licensing institution | 공공저작물 동의서 | "국립극장" | - |
| `copyright_license.granted_rights.reproduction_right` | boolean | ✅ | true/false | 복제권 (목제권 포함) | Reproduction right (including 목제권) | 공공저작물 동의서 | `true` | OCR 오류 고려: "목제권" → "복제권" |
| `copyright_license.granted_rights.performance_right` | boolean | ✅ | true/false | 공연권 (공면권 포함) | Performance right (including 공면권) | 공공저작물 동의서 | `true` | OCR 오류 고려: "공면권" → "공연권" |
| `copyright_license.granted_rights.broadcasting_right` | boolean | ✅ | true/false | 공중송신권 | Broadcasting right | 공공저작물 동의서 | `true` | 체크박스 기반 |
| `copyright_license.granted_rights.exhibition_right` | boolean | ✅ | true/false | 전시권 | Exhibition right | 공공저작물 동의서 | `false` | 체크박스 기반 |
| `copyright_license.granted_rights.distribution_right` | boolean | ✅ | true/false | 배포권 | Distribution right | 공공저작물 동의서 | `false` | 체크박스 기반 |
| `copyright_license.granted_rights.rental_right` | boolean | ✅ | true/false | 대여권 | Rental right | 공공저작물 동의서 | `false` | 체크박스 기반 |
| `copyright_license.granted_rights.derivative_work_right` | boolean | ✅ | true/false | 2차적저작물작성권 | Derivative work right | 공공저작물 동의서 | `false` | 체크박스 기반 |
| `copyright_license.license_type` | string | ⚪ | - | 이용허락 유형 | License type | 공공저작물 동의서 | "비독점적" | - |
| `public_nuri_license.license_purpose` | string | ✅ | - | 공공누리 적용 목적 | Public Nuri license purpose | 공공저작물 동의서 | "공공저작물 자유이용" | - |
| `public_nuri_license.available_types` | array | ⚪ | Array of strings | 사용 가능한 공공누리 유형들 | Available Public Nuri types | 공공저작물 동의서 | `["제1유형", "제2유형"]` | - |
| `public_nuri_license.modification_rights.integrity_right_waiver` | boolean | ⚪ | true/false | 동일성유지권 행사 제안 동의 | Integrity right waiver consent | 공공저작물 동의서 | `true` | 체크박스 기반 |
| `public_nuri_license.modification_rights.modification_allowed` | boolean | ⚪ | true/false | 변경 이용 가능 여부 | Modification allowed | 공공저작물 동의서 | `true` | 체크박스 기반 |
| `public_nuri_license.modification_rights.conditions` | string | ⚪ | - | 변경 이용 조건 | Modification conditions | 공공저작물 동의서 | "출처 표기 필수" | - |
| `personal_info_consent.collected_data_types` | array | ✅ | Array of strings | 수집하는 개인정보 항목 | Collected personal information items | 공공저작물 동의서 | `["성명", "전화번호"]` | - |
| `personal_info_consent.collection_purpose` | string | ✅ | - | 개인정보 수집 이용목적 | Personal information collection purpose | 공공저작물 동의서 | "공공저작물 이용허락 처리" | - |
| `personal_info_consent.retention_period` | string | ✅ | - | 개인정보 보유, 이용기간 | Personal information retention period | 공공저작물 동의서 | "이용허락 기간 동안" | - |
| `utilizing_institution` | string | ⚪ | - | 활용기관 | Utilizing institution | 공공저작물 동의서 | "국립극장" | - |
| `processing_info.checkbox_pattern_detected` | string | ⚪ | enum | 감지된 체크박스 패턴 | Detected checkbox pattern | 공공저작물 동의서 | `"pattern_b"` | enum: pattern_a, pattern_b, pattern_c, pattern_d |
| `processing_info.extraction_confidence` | number | ⚪ | 0.0-1.0 | 추출 신뢰도 | Extraction confidence | 공공저작물 동의서 | `0.94` | 범위: 0.0 ~ 1.0 |

### 필드 검색 팁

1. **문서 유형별 필터링**: "Document Types" 컬럼을 사용하여 특정 문서 유형에 사용되는 필드만 확인할 수 있습니다.
2. **필수 필드 확인**: "Required" 컬럼에서 ✅ 표시된 필드는 해당 문서 유형에서 필수입니다.
3. **체크박스 필드**: "Notes" 컬럼에 "체크박스 패턴 자동 감지"가 표시된 필드는 체크박스 기반으로 추출됩니다.
4. **공통 필드**: "Document Types"에 "모든 문서 유형"이 표시된 필드는 모든 스키마에서 사용됩니다.
5. **중첩 필드**: 점(.) 표기법을 사용하여 중첩된 객체의 필드를 참조합니다 (예: `parties[].name`).

### 체크박스 패턴 참조

| Pattern ID | 체크됨 표시 | 체크 안됨 표시 | Description |
|------------|------------|---------------|-------------|
| `pattern_a` | 📧 | ☐ | 이메일/체크박스 패턴 |
| `pattern_b` | ☑ | □ | 일반 체크박스 패턴 |
| `pattern_c` | ✓ | ○ | 체크마크/원형 패턴 |
| `pattern_d` | ■ | □ | 사각형/체크박스 패턴 |

### 특수 처리 규칙

1. **OCR 오류 보정**:
   - "목제권" → "복제권"으로 자동 해석
   - "공면권" → "공연권"으로 자동 해석

2. **날짜 형식**: 모든 날짜는 `YYYY-MM-DD` 형식으로 변환됩니다.

3. **금액 형식**: 금액은 숫자만 추출하며, 통화 단위는 별도 필드에 저장됩니다.

4. **전화번호 및 등록번호**: 숫자와 하이픈(-)만 포함합니다 (정규식: `^\d+(-\d+)*$`).

5. **null 값 처리**: 정보가 명시적으로 존재하지 않거나 불분명한 경우 `null`을 사용합니다.

---

## 참고 자료

- [JSON Schema Specification](https://json-schema.org/)
- [ISO 8601 Date Format](https://www.iso.org/iso-8601-date-and-time-format.html)
- [공공누리 라이선스](https://www.kogl.or.kr/)

