# NER Entity Labels Mapping to Digital Content Schema

## Overview

This document maps the NER entity labels from the reference table to both the existing schemas (contracts, consents) and the new digital content schema.

---

## Complete Entity Label Mapping

### ✅ Already Mapped Entity Types

| NER Label | Korean | Current Mappings | Digital Content Schema Fields |
|-----------|--------|-----------------|------------------------------|
| **NAME** | 이름 | `rights_holder`, `user`, `parties[].name` | `copyright_holder`, `co_author` |
| **PHONE** | 전화번호 | `parties[].phone`, `contact_info.phone` | `phone` |
| **ADDRESS** | 주소 | `parties[].address`, `contact_info.address` | - |
| **DATE** | 날짜 | `signature_date`, `effective_date`, `expiration_date` | `created_date`, `registration_date`, `production_date`, `valid_period` |
| **COMPANY** | 기관, 회사명 | `user`, `rights_holder`, `data_controller` | `agency_name`, `site_name`, `board_name` |
| **EMAIL** | 이메일 | `parties[].email`, `contact_info.email` | - |
| **POSITION** | 직위, 직책 | `parties[].role`, `position` | - |
| **MONEY** | 금액 | `payment_amount`, `price`, `fee` | - |
| **ID_NUM** | 식별번호 | `parties[].registration_no` | - |
| **TITLE** | 제목, 문서명 | `work_title`, `document_title`, `title` | `work_title` |
| **URL** | 링크, 주소 | `website`, `homepage` | `url` |
| **DESCRIPTION** | 설명, 내용 | `contract_purpose`, `collection_purpose` | `description`, `memo` |
| **CONTRACT_TYPE** | 계약 유형 | `contract_type` | - |
| **CONSENT_TYPE** | 동의 유형 | `consent_type` | - |

### ✅ Newly Added Entity Types

| NER Label | Korean | Digital Content Schema Fields | Priority |
|-----------|--------|------------------------------|----------|
| **PERIOD** | 기간 | `valid_period`, `contract_duration` | High |
| **RIGHT_INFO** | 권리 정보 | `economic_rights`, `third_party_rights`, `portrait_rights`, `neighboring_rights_holder` | High |
| **PROJECT_NAME** | 프로젝트명, 사업명 | `work_title`, `title` | Medium |
| **LAW_REFERENCE** | 법령, 조항 | `special_terms`, `important_terms` | Low |
| **TYPE** | 유형, 종류 | `work_type`, `category`, `work_category` | High |
| **STATUS** | 상태, 진행현황 | `disclosure_type`, `review_impossible` | Medium |
| **DEPARTMENT** | 담당부서 | `agency_name`, `data_controller` | Medium |
| **LANGUAGE** | 언어 | `language` | High |
| **QUANTITY** | 수량, 분량 | `quantity`, `video_count`, `photo_count`, `document_count`, `view_count` | High |

---

## Detailed Field Mappings

### 1. NAME (이름)
**Description**: 문서의 성명

**Mappings**:
- Contract/Consent: `rights_holder`, `user`, `data_controller`, `data_subject`, `parties[].name`
- **Digital Content**: `copyright_holder`, `co_author`

**Priority**: `copyright_holder` = 10, `co_author` = 8

---

### 2. PHONE (전화번호)
**Description**: 연락처, 휴대전화 등의 숫자

**Mappings**:
- Contract/Consent: `parties[].phone`, `contact_info.phone`
- **Digital Content**: `phone`

**Priority**: `phone` = 8

---

### 3. ADDRESS (주소)
**Description**: 지리적 위치

**Mappings**:
- Contract/Consent: `parties[].address`, `contact_info.address`
- Digital Content: (No direct mapping, but can be extracted)

---

### 4. DATE (날짜)
**Description**: 일자/기간

**Mappings**:
- Contract/Consent: `signature_date`, `effective_date`, `expiration_date`, `consent_date`
- **Digital Content**: `created_date`, `registration_date`, `production_date`, `valid_period`

**Priority**: `created_date` = 9, `registration_date` = 8, `production_date` = 7

---

### 5. COMPANY (기관, 회사명)
**Description**: 법인, 단체, 발주처 등의 명칭

**Mappings**:
- Contract/Consent: `user`, `rights_holder`, `data_controller`
- **Digital Content**: `agency_name`, `site_name`, `board_name`

**Priority**: `agency_name` = 10, `site_name` = 9, `board_name` = 8

---

### 6. EMAIL (이메일)
**Description**: 전자우편 주소

**Mappings**:
- Contract/Consent: `parties[].email`, `contact_info.email`
- Digital Content: (No direct mapping)

---

### 7. POSITION (직위, 직책)
**Description**: 담당자 역할, 직급, 직함 등

**Mappings**:
- Contract/Consent: `parties[].role`, `position`
- Digital Content: (No direct mapping)

---

### 8. CONTRACT_TYPE (계약 유형)
**Description**: 저작권/라이선스 등의 문서 종류 및 형태

**Mappings**:
- Contract/Consent: `contract_type`
- Digital Content: (No direct mapping)

---

### 9. MONEY (금액)
**Description**: 대가, 보수 등의 돈

**Mappings**:
- Contract/Consent: `payment_amount`, `price`, `fee`
- Digital Content: (No direct mapping)

---

### 10. PERIOD (기간) ⭐ NEW
**Description**: 시간적 범위

**Mappings**:
- Contract/Consent: `contract_duration`, `retention_period`
- **Digital Content**: `valid_period`

**Priority**: `valid_period` = 10

---

### 11. ID_NUM (식별번호)
**Description**: 신원 및 법인의 식별값

**Mappings**:
- Contract/Consent: `parties[].registration_no`, `registration_no`
- Digital Content: (No direct mapping)

---

### 12. CONSENT_TYPE (동의 유형)
**Description**: 이용 허락과 같은 문서 종류

**Mappings**:
- Contract/Consent: `consent_type`
- Digital Content: (No direct mapping)

---

### 13. RIGHT_INFO (권리 정보) ⭐ NEW
**Description**: 지적재산권, 인접권 등의 권리 항목

**Mappings**:
- Contract/Consent: `granted_rights`
- **Digital Content**: `economic_rights`, `third_party_rights`, `portrait_rights`, `neighboring_rights_holder`

**Priority**: `economic_rights` = 10, `third_party_rights` = 8, `portrait_rights` = 7

---

### 14. PROJECT_NAME (프로젝트명, 사업명) ⭐ NEW
**Description**: 수행 과제, 사업 등의 명칭

**Mappings**:
- Contract/Consent: `work_title`, `title`
- **Digital Content**: `work_title` (same)

**Priority**: `work_title` = 10

---

### 15. LAW_REFERENCE (법령, 조항) ⭐ NEW
**Description**: 법령, 규정 등의 조문 인용

**Mappings**:
- Contract/Consent: `special_terms`, `important_terms`, `termination_conditions`
- Digital Content: (No direct mapping, but can be in `memo`)

---

### 16. TITLE (제목, 문서명)
**Description**: 계약서명, 저작물명 등

**Mappings**:
- Contract/Consent: `work_title`, `document_title`, `title`
- **Digital Content**: `work_title` (same)

---

### 17. URL (링크, 주소)
**Description**: 인터넷 경로

**Mappings**:
- Contract/Consent: `website`, `homepage`
- **Digital Content**: `url`

**Priority**: `url` = 10

---

### 18. DESCRIPTION (설명, 내용)
**Description**: 목적, 조항, 내역 등등의 서술형 텍스트

**Mappings**:
- Contract/Consent: `contract_purpose`, `collection_purpose`
- **Digital Content**: `description`, `memo`

**Priority**: `description` = 10, `memo` = 8

---

### 19. TYPE (유형, 종류) ⭐ NEW
**Description**: 저작물 형태 분류 카테고리

**Mappings**:
- Contract/Consent: `work_category`, `document_type`
- **Digital Content**: `work_type`, `category`

**Priority**: `work_type` = 10, `category` = 9

---

### 20. STATUS (상태, 진행현황) ⭐ NEW
**Description**: 처리중인 단계

**Mappings**:
- Contract/Consent: (No direct mapping)
- **Digital Content**: `disclosure_type`, `review_impossible`

**Priority**: `disclosure_type` = 10, `review_impossible` = 8

---

### 21. DEPARTMENT (담당부서) ⭐ NEW
**Description**: 계약서명 등의 담당 조직명

**Mappings**:
- Contract/Consent: `data_controller`, `organization`
- **Digital Content**: `agency_name`

**Priority**: `agency_name` = 10

---

### 22. LANGUAGE (언어) ⭐ NEW
**Description**: 문서에 사용된 언어 형태

**Mappings**:
- Contract/Consent: (No direct mapping)
- **Digital Content**: `language`

**Priority**: `language` = 10

---

### 23. QUANTITY (수량, 분량) ⭐ NEW
**Description**: 항목의 개수 및 단위

**Mappings**:
- Contract/Consent: (No direct mapping)
- **Digital Content**: `quantity`, `video_count`, `photo_count`, `document_count`, `view_count`

**Priority**: `quantity` = 10, `video_count` = 9, `photo_count` = 9, `document_count` = 9, `view_count` = 8

---

## Summary

### Total Entity Types: 23
- ✅ **Already Mapped**: 14 types
- ⭐ **Newly Added**: 9 types (PERIOD, RIGHT_INFO, PROJECT_NAME, LAW_REFERENCE, TYPE, STATUS, DEPARTMENT, LANGUAGE, QUANTITY)

### Digital Content Schema Coverage
- **Direct Mappings**: 20+ fields from digital content schema now have NER entity mappings
- **Priority Scores**: All new mappings include priority scores for intelligent field matching

### Benefits
1. **Better Consolidation**: NER entities can now be mapped to all relevant digital content fields
2. **Comprehensive Coverage**: All 23 entity types from the reference table are now supported
3. **Intelligent Matching**: Priority scores ensure the best field matches are selected
4. **Future-Proof**: Easy to add more mappings as new schemas are introduced

---

## Usage Example

When NER extracts:
- `("한국문화정보원", "COMPANY")` → Maps to `agency_name` (priority 10)
- `("2024-11-10", "DATE")` → Maps to `created_date` (priority 9) or `registration_date` (priority 8)
- `("https://example.com", "URL")` → Maps to `url` (priority 10)
- `("한국어", "LANGUAGE")` → Maps to `language` (priority 10)
- `("15", "QUANTITY")` → Maps to `quantity` (priority 10) or `video_count` (priority 9)

The FieldMapper will intelligently select the best matching field based on:
1. Priority scores
2. Existing LLM metadata values
3. Document type context
4. OCR text context

