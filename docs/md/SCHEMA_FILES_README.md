# Schema Files Documentation

This directory contains complete JSON schema files for both the NER and LLM models used in the copyright metadata extraction system.

## Files Created

### 1. `NER_SCHEMA.json` (12 KB)
Complete schema for the Named Entity Recognition (NER) model.

**Contents:**
- **Entity Types**: All 23 entity types supported by the NER model
- **Extraction Methods**: BERT-based and regex-based extraction configurations
- **Validation Rules**: Entity validation criteria and type-specific rules
- **Document-Specific Labels**: Contract and consent document label mappings
- **Output Format**: Structure of NER extraction results

**Key Features:**
- 23 entity types: NAME, PHONE, ADDRESS, DATE, COMPANY, EMAIL, POSITION, CONTRACT_TYPE, MONEY, PERIOD, ID_NUM, CONSENT_TYPE, RIGHT_INFO, PROJECT_NAME, LAW_REFERENCE, TITLE, URL, DESCRIPTION, TYPE, STATUS, DEPARTMENT, LANGUAGE, QUANTITY
- Dual extraction methods (BERT + Regex)
- Complete validation rules
- Document-specific label mappings

### 2. `LLM_SCHEMA.json` (44 KB)
Complete schema collection for the LLM-based metadata extraction model.

**Contents:**
- **6 Document Type Schemas**:
  1. **contract** (계약서) - Contract documents with checkbox support
  2. **consent** (동의서) - Consent forms with checkbox support
  3. **general** (기타문서) - General/unknown document types
  4. **copyright_transfer** (저작재산권 양도동의서) - Copyright transfer consent forms
  5. **public_copyright** (공공저작물 자유이용허락 동의서) - Public copyright consent with 공공누리 support
  6. **digital_content** (디지털 콘텐츠) - Digital content management systems

**Key Features:**
- Universal checkbox support (4 patterns: pattern_a, pattern_b, pattern_c, pattern_d)
- Comprehensive field definitions with Korean descriptions
- Required field specifications for each document type
- Nested object structures for complex data
- Date format specifications (YYYY-MM-DD)
- Support for arrays, objects, and primitive types

## Schema Structure

### NER Schema Structure
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NER Model Entity Schema",
  "properties": {
    "extracted_entities": [...],
    "entity_types_count": {...},
    "total_entities": 0,
    "processing_time": 0.0
  },
  "definitions": {
    "entity_types": {...},
    "document_specific_labels": {...},
    "extraction_methods": {...},
    "validation_rules": {...}
  }
}
```

### LLM Schema Structure
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "LLM Model Metadata Extraction Schema",
  "document_types": {
    "contract": {...},
    "consent": {...},
    ...
  },
  "schemas": {
    "contract": {
      "type": "object",
      "properties": {...},
      "required": [...]
    },
    ...
  }
}
```

## Usage

### Validating NER Results
```python
import json
from jsonschema import validate

# Load NER schema
with open('NER_SCHEMA.json', 'r', encoding='utf-8') as f:
    ner_schema = json.load(f)

# Validate NER extraction result
ner_result = {
    "extracted_entities": [
        {"entity_text": "김민수", "entity_type": "NAME"},
        {"entity_text": "010-1234-5678", "entity_type": "PHONE"}
    ],
    "total_entities": 2
}

validate(instance=ner_result, schema=ner_schema)
```

### Validating LLM Results
```python
import json
from jsonschema import validate

# Load LLM schema
with open('LLM_SCHEMA.json', 'r', encoding='utf-8') as f:
    llm_schema_data = json.load(f)

# Get specific document type schema
contract_schema = llm_schema_data['schemas']['contract']

# Validate LLM extraction result
llm_result = {
    "contract_type": "저작재산권 비독점적 이용허락 계약서",
    "rights_holder": "집건에",
    "user": "국립생태원",
    "granted_rights": {
        "reproduction_right": True,
        "performance_right": False
    }
}

validate(instance=llm_result, schema=contract_schema)
```

## Entity Type Mappings

### NER → LLM Field Mappings

| NER Entity Type | LLM Field Examples |
|----------------|-------------------|
| NAME | `rights_holder`, `user`, `parties[].name`, `data_subject` |
| PHONE | `parties[].phone`, `contact_info.phone` |
| ADDRESS | `parties[].address`, `contact_info.address` |
| DATE | `signature_date`, `effective_date`, `consent_date`, `created_date` |
| COMPANY | `user`, `rights_holder`, `data_controller`, `agency_name` |
| EMAIL | `parties[].email`, `contact_info.email` |
| MONEY | `payment_amount`, `compensation.amount` |
| TITLE | `work_title`, `document_title`, `title` |
| URL | `url` |
| DESCRIPTION | `contract_purpose`, `description`, `main_content` |

## Document Type Selection

The LLM model automatically selects the appropriate schema based on document type detection:

```python
from api.module.llm_extraction.schemas.document_schemas import DocumentSchemas

# Automatic schema selection
schema = DocumentSchemas.get_schema_by_document_type("계약서")
# Returns: contract schema

schema = DocumentSchemas.get_schema_by_document_type("동의서")
# Returns: consent schema

schema = DocumentSchemas.get_schema_by_document_type("저작재산권 양도동의서")
# Returns: copyright_transfer schema
```

## Checkbox Support

Both schemas support checkbox extraction:

### NER Schema
- Uses pattern-based detection for checkbox symbols
- Validates checkbox entities separately

### LLM Schema
- **4 Checkbox Patterns**:
  - `pattern_a`: 📧/☐
  - `pattern_b`: ☑/□
  - `pattern_c`: ✓/○
  - `pattern_d`: ■/□
- Checkbox fields are boolean types
- Checkbox detection metadata included in results

## Required Fields

### Contract Schema
- `contract_type`
- `rights_holder`
- `user`
- `granted_rights`

### Consent Schema
- `consent_type`
- `data_controller`
- `consent_status`

### Copyright Transfer Schema
- `document_type`
- `work_info`
- `copyright_transfer`
- `consent_info`
- `parties`

### Public Copyright Schema
- `consent_type`
- `work_display`
- `copyright_license`
- `public_nuri_license`

### Digital Content Schema
- `work_title` (only required field)

## Notes

1. **Date Format**: All dates in LLM schemas use `YYYY-MM-DD` format
2. **Phone Format**: Phone numbers should contain only digits and hyphens
3. **Null Values**: Most fields allow `null` values when information is not available
4. **Arrays**: Many fields support arrays for multiple values
5. **Nested Objects**: Complex data structures use nested objects (e.g., `parties[]`, `granted_rights`)

## Version Information

- **NER Schema Version**: 1.0.0
- **LLM Schema Version**: 1.0.0
- **Last Updated**: 2025-12-29

## Related Documentation

- `REGEX_PATTERNS_DOCUMENTATION.md` - Regex patterns used by both models
- `HOW_NER_EXTRACTS_METADATA.md` - How NER extraction works
- `NER_LABELS_MAPPING.md` - NER to LLM field mappings

