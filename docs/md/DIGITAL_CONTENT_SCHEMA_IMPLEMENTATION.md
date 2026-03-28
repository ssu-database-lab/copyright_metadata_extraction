# Digital Content Schema Implementation Guide

## Summary

A new schema has been added to support **digital content management and archive systems**. This schema includes 42 metadata fields for extracting information from web-based content platforms, public archives, and digital content management systems.

---

## What Was Added

### 1. New Schema Method
- **`get_digital_content_schema()`** - Returns schema with 42 fields for digital content metadata

### 2. Schema Detection Updates
- Updated `get_schema_by_document_type()` to detect digital content documents
- Updated `detect_document_type_from_title()` to recognize digital content keywords

### 3. Frontend Updates
- Added "디지털 콘텐츠" (Digital Content) option to document type selection

---

## New Schema Fields (42 Total)

### Content Organization (9 fields)
- `seq_number` - 순번
- `site_name` - 사이트명
- `agency_name` - 기관명
- `board_name` - 게시판명
- `board_path` - 게시판 진입 과정
- `category` - 카테고리
- `work_title` - 저작물명
- `url` - URL
- `description` - 설명

### Date Fields (4 fields)
- `created_date` - 작성일
- `registration_date` - 등록일
- `production_date` - 제작일
- `valid_period` - 유효기간

### File/Media Fields (5 fields)
- `attachment` - 첨부파일
- `video_count` - 영상 개수
- `photo_count` - 사진 개수
- `document_count` - 문서 개수
- `quantity` - 수량

### Statistics (1 field)
- `view_count` - 조회수

### Copyright & Licensing (15 fields)
- `kogl_type` - 공공누리유형
- `disclosure_type` - 공개유형
- `copyrightability` - 저작물성
- `unprotected_work` - 비보호저작물
- `work_for_hire` - 업무상저작물
- `copyright_holder` - 저작권자
- `co_author` - 공동저작자
- `neighboring_rights_holder` - 저작인접권자
- `co_author_consent` - 공동저작자동의
- `third_party_rights` - 제3자 권리
- `economic_rights` - 저작재산권
- `commercial_use` - 상업적이용허락
- `portrait_rights` - 초상권
- `personal_info` - 개인정보

### Additional Fields (8 fields)
- `contract` - 계약서
- `review_impossible` - 검토불가
- `work_type` - 유형
- `digital_format` - 디지털형태
- `keyword` - 주제어
- `language` - 언어
- `phone` - 전화번호
- `memo` - 비고

---

## How to Use

### 1. Via API

```python
# The schema will be automatically selected based on document_type
POST /api/llm-extract
{
    "document_type": "디지털 콘텐츠",
    "model_name": "alibaba-qwen-max",
    ...
}
```

### 2. Via Frontend

1. Select "LLM 메타데이터 추출" as processing type
2. Select "디지털 콘텐츠" as document type
3. Upload your document
4. The system will automatically use the digital content schema

### 3. Programmatically

```python
from module.llm_extraction.schemas.document_schemas import DocumentSchemas

# Get the schema directly
schema = DocumentSchemas.get_digital_content_schema()

# Or let the system auto-detect
schema = DocumentSchemas.get_schema_by_document_type("디지털 콘텐츠")
```

---

## Schema Detection Keywords

The system will automatically use the digital content schema if the document type contains:

**Korean Keywords:**
- 디지털 콘텐츠
- 공공저작물
- 콘텐츠 관리
- 아카이브
- 게시판
- 사이트
- 공공누리
- 디지털

**English Keywords:**
- digital content
- public content
- content management
- archive
- board
- site
- kogl

---

## Field Types

Most fields support multiple types for flexibility:

- **String fields**: Can be `string` or `null`
- **Numeric fields**: Can be `integer` or `null`
- **Boolean fields**: Can be `boolean`, `string`, or `null`
- **Array fields**: Can be `string`, `array`, or `null`
- **Date fields**: Format `YYYY-MM-DD` or `null`

This allows the LLM to extract data in various formats while maintaining schema compliance.

---

## Required Fields

Only `work_title` is required. All other fields are optional, allowing for partial extraction when some information is not available in the source document.

---

## Integration with Consolidation

The new schema works seamlessly with the consolidation module:

1. **LLM Extraction**: Uses digital content schema to extract metadata
2. **NER Extraction**: Extracts entities (names, dates, addresses, etc.)
3. **Consolidation**: Merges and validates both sources
4. **Final Result**: Provides consolidated metadata with decisions and reasoning

---

## Example Output

```json
{
  "seq_number": 1,
  "site_name": "공공데이터포털",
  "agency_name": "한국문화정보원",
  "board_name": "공공저작물",
  "work_title": "2024년 공공저작물 디지털 전환 구축 사업",
  "url": "https://example.com/content/123",
  "description": "공공저작물 디지털 전환 관련 콘텐츠",
  "created_date": "2024-11-10",
  "registration_date": "2024-11-15",
  "kogl_type": "제1유형",
  "copyright_holder": "한국문화정보원",
  "view_count": 1250,
  "video_count": 3,
  "photo_count": 15,
  "document_count": 2,
  ...
}
```

---

## Next Steps

1. **Test the Schema**: Upload a digital content document and verify extraction
2. **Adjust Field Types**: Modify field types if needed based on actual data
3. **Add Validation**: Add specific validation rules for fields like `kogl_type`, `url`, etc.
4. **Update Documentation**: Update API documentation with the new schema

---

## Notes

- The schema is designed to be flexible and accommodate various document formats
- All fields except `work_title` are optional to handle incomplete documents
- The schema supports both single values and arrays where appropriate
- Date fields use ISO format (YYYY-MM-DD) for consistency








