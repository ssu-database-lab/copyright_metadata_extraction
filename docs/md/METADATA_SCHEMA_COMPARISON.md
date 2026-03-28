# Metadata Schema Comparison: Current vs New Requirements

## Overview

The current LLM extraction schemas are designed for **legal documents** (contracts, consent forms), while the new metadata list appears to be for **digital content management** or **public content archive** systems (websites, boards, digital works).

---

## Current Schema Fields (Legal Documents)

### Contract Schema (계약서)
- `contract_type` - 계약서 유형
- `rights_holder` - 권리자
- `user` - 이용자
- `work_title` - 저작물 제목
- `work_category` - 저작물 종별
- `granted_rights` - 허락된 권리
- `contract_purpose` - 계약 목적
- `contract_duration` - 계약 기간
- `payment_amount` - 지급 금액
- `payment_currency` - 통화
- `signature_date` - 계약 체결일
- `effective_date` - 효력 발생일
- `expiration_date` - 만료일
- `special_terms` - 특별 약정
- `termination_conditions` - 해지 조건
- `parties[]` - 당사자 정보 (name, phone, address, role)

### Consent Schema (동의서)
- Similar structure with consent-specific fields

### General Document Schema (기타문서)
- `document_type` - 문서 유형
- `title` - 문서 제목
- `parties[]` - 당사자 정보
- `key_dates[]` - 중요 날짜들
- `key_amounts[]` - 중요 금액들
- `main_content` - 주요 내용
- `important_terms[]` - 중요 조항

---

## New Metadata List (Digital Content Management)

The new list contains **42 fields** focused on digital content metadata:

### Content Organization Fields
- `seq_number` - 순번
- `site_name` - 사이트명
- `agency_name` - 기관명
- `board_name` - 게시판명
- `board_path` - 게시판 진입 과정
- `category` - 카테고리
- `work_title` - 저작물명 ✅ (overlaps with current)
- `url` - URL
- `description` - 설명

### Date Fields
- `created_date` - 작성일
- `registration_date` - 등록일
- `production_date` - 제작일
- `valid_period` - 유효기간

### File/Media Fields
- `attachment` - 첨부파일
- `video_count` - 영상
- `photo_count` - 사진
- `document_count` - 문서
- `quantity` - 수량

### Statistics Fields
- `view_count` - 조회수

### Copyright & Licensing Fields
- `kogl_type` - 공공누리유형 (Korea Open Government License)
- `disclosure_type` - 공개유형
- `copyrightability` - 저작물성
- `unprotected_work` - 비보호저작물
- `work_for_hire` - 업무상저작물
- `copyright_holder` - 저작권자 ✅ (similar to rights_holder)
- `co_author` - 공동저작자
- `neighboring_rights_holder` - 저작인접권자
- `co_author_consent` - 공동저작자동의
- `third_party_rights` - 제3자 권리
- `economic_rights` - 저작재산권
- `commercial_use` - 상업적이용허락
- `portrait_rights` - 초상권
- `personal_info` - 개인정보

### Additional Fields
- `contract` - 계약서
- `review_impossible` - 검토불가
- `work_type` - 유형
- `digital_format` - 디지털형태
- `keyword` - 주제어
- `language` - 언어
- `phone` - 전화번호 ✅ (exists in parties[].phone)
- `memo` - 비고

---

## Key Differences

| Aspect | Current Schemas | New Metadata List |
|--------|----------------|-------------------|
| **Purpose** | Legal documents (contracts, consents) | Digital content management/archive |
| **Focus** | Contract terms, parties, rights | Content organization, copyright, statistics |
| **Structure** | Nested (parties array, dates array) | Flat structure (mostly) |
| **Fields** | ~15-30 fields per schema | 42 fields |
| **Document Type** | Contracts, consent forms | Web content, digital works, archives |
| **Copyright Info** | Basic (rights_holder, user) | Comprehensive (KOGL, disclosure, multiple rights) |
| **Organization** | Parties-based | Site/board/category-based |
| **Statistics** | None | View counts, media counts |
| **URL/Web** | None | URL, site_name, board_name |

---

## Overlapping Fields

| New Field | Current Equivalent | Notes |
|-----------|-------------------|-------|
| `work_title` | `work_title` | Same concept |
| `copyright_holder` | `rights_holder` | Similar but different context |
| `phone` | `parties[].phone` | Exists in nested structure |
| `description` | `main_content` or `contract_purpose` | Similar concept |

---

## Fields Only in New List

**Content Organization (9 fields):**
- seq_number, site_name, agency_name, board_name, board_path, category, url, description, memo

**Statistics (5 fields):**
- video_count, photo_count, document_count, quantity, view_count

**Advanced Copyright (13 fields):**
- kogl_type, disclosure_type, copyrightability, unprotected_work, work_for_hire, co_author, neighboring_rights_holder, co_author_consent, third_party_rights, economic_rights, commercial_use, portrait_rights, personal_info

**Dates (4 fields):**
- created_date, registration_date, production_date, valid_period

**Other (4 fields):**
- attachment, contract, review_impossible, work_type, digital_format, keyword, language

---

## Fields Only in Current Schemas

**Contract-specific:**
- contract_type, contract_purpose, contract_duration, signature_date, effective_date, expiration_date, special_terms, termination_conditions

**Parties structure:**
- parties[] array with nested name, phone, address, registration_no, role

**Rights structure:**
- granted_rights array, contract_terms object

---

## Integration Strategy

### Option 1: Create New Schema Type
Create a new schema type (e.g., `digital_content_schema`) for this use case.

### Option 2: Extend General Schema
Add these fields to the general document schema as optional fields.

### Option 3: Hybrid Schema
Create a schema that supports both legal documents and digital content.

**Recommendation: Option 1** - Create a dedicated schema for digital content management, as it has a fundamentally different structure and purpose.








