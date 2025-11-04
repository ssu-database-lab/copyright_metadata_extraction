# Metadata Consolidation Agent - Design Proposal

## 1. Architecture Overview

### Module Structure: `api/module/consolidator/`

```
api/module/consolidator/
├── __init__.py
├── consolidation_agent.py      # Main LLM agent orchestrator
├── field_mapper.py             # Maps NER entities to LLM fields
├── validation_engine.py         # Validates & cross-checks data
├── reasoning_generator.py      # Generates evidence & reasoning
└── schemas/
    └── consolidation_schemas.py  # Schema definitions
```

## 2. Core Components

### 2.1 Field Mapper (`field_mapper.py`)
**Purpose**: Maps NER entities to LLM metadata fields intelligently

**Key Mappings**:
- `NAME` entity → `rights_holder`, `user`, `parties[].name`
- `DATE` entity → `signature_date`, `effective_date`, `expiration_date`
- `PHONE` entity → `parties[].phone`
- `ADDRESS` entity → `parties[].address`
- `COMPANY` entity → `rights_holder`, `user`, `data_controller`
- `MONEY` entity → `payment_amount`
- `EMAIL` entity → `parties[].email`

**Logic**:
- Infer field from NER entity position and context
- Search for matches (fuzzy if needed)
- Score candidates by proximity and semantic similarity

### 2.2 Validation Engine (`validation_engine.py`)
**Purpose**: Cross-validates LLM vs NER results

**Validation Rules**:
1. **Format consistency**: Dates in YYYY-MM-DD; phone with digits/hyphens only
2. **Logical consistency**: Start < End date; phone length constraints
3. **Entity presence**: NER entity exists in OCR text (verify source)
4. **LLM confidence**: Use when LLM confidence > threshold
5. **NER confidence**: Prefer multi-label or high-score NER entities

### 2.3 Consolidation Agent (`consolidation_agent.py`)
**Purpose**: Main LLM agent that performs intelligent merging

**Key Features**:
1. **Comparison Layer**: Compares LLM vs NER findings field-by-field
2. **Decision Layer**: Selects best value based on rules + evidence
3. **Reasoning Layer**: Generates explanations for each decision
4. **Output Layer**: Produces consolidated metadata + evidence report

**Decision Logic**:
- If **agreement** → Use value (confidence++)
- If **conflict** → Use NER if present in OCR; prefer higher-confidence source
- If **missing LLM** → Use NER
- If **missing NER** → Use LLM with caution flag

### 2.4 Reasoning Generator (`reasoning_generator.py`)
**Purpose**: Generates human-readable evidence for each decision

**Output Format**:
```json
{
  "field_name": "signature_date",
  "llm_value": "2024-01-15",
  "ner_value": "2024-01-15",
  "final_value": "2024-01-15",
  "decision": "AGREED",
  "confidence": 1.0,
  "reasoning": "LLM and NER both extracted identical date from OCR text at position [120:130]. High confidence.",
  "evidence": {
    "llm_confidence": 0.95,
    "ner_confidence": 0.90,
    "source_text": "... 계약 체결일: 2024-01-15 ..."
  }
}
```

## 3. LLM Agent Prompt Strategy

### 3.1 Prompt Structure
```python
PROMPT_TEMPLATE = """
당신은 한국어 문서 메타데이터 추출 전문가입니다. 
두 가지 다른 추출 방법(LLM 추출과 NER 추출)의 결과를 비교하고 통합하는 것이 임무입니다.

**텍스트**: {ocr_text}

**LLM 추출 결과**:
{llm_metadata}

**NER 추출 결과**:
{ner_entities}

**작업**:
1. 각 필드를 비교하여 일치/불일치/누락 여부를 판단
2. 더 신뢰할 수 있는 값을 선택
3. 선택 이유를 설명
4. 최종 통합 메타데이터 생성

**출력 형식**:
- 각 필드에 대해 decision, value, reasoning 제공
- 전체 consolidated_metadata와 validation_report 제공
"""
```

### 3.2 Output Schema
```json
{
  "consolidated_metadata": { ... },
  "validation_report": {
    "total_fields": 15,
    "agreed_fields": 10,
    "conflicted_fields": 3,
    "llm_only_fields": 2,
    "ner_only_fields": 0,
    "decisions": [
      { "field": "...", "decision": "...", "reasoning": "..." }
    ]
  },
  "confidence_score": 0.92,
  "processing_time": 2.3
}
```

## 4. Implementation Steps

### Phase 1: Basic Mapping & Validation
1. ✅ Create `field_mapper.py` with basic entity-to-field mappings
2. ✅ Create `validation_engine.py` with format checks
3. ✅ Test on sample data

### Phase 2: LLM Agent Integration
1. ✅ Create `consolidation_agent.py` using existing LLM infrastructure
2. ✅ Create prompt templates
3. ✅ Implement decision logic

### Phase 3: Reasoning & Reporting
1. ✅ Create `reasoning_generator.py`
2. ✅ Generate evidence report
3. ✅ Integration testing

### Phase 4: Production Integration
1. ✅ Integrate into `/api/llm-extract` endpoint
2. ✅ Update response format
3. ✅ Add configurable confidence thresholds

## 5. Best Practices

### 5.1 Error Handling
- Graceful degradation if NER fails
- Fallback to LLM-only if consolidation fails
- Log all conflicts for analysis

### 5.2 Performance
- Cache field mappings
- Use async LLM calls
- Parallel validation where possible

### 5.3 Extensibility
- Plugin architecture for custom validators
- Configurable thresholds
- Support multiple document types

### 5.4 Testing
- Unit tests for each component
- Integration tests with real documents
- Performance benchmarking

## 6. Configuration

```yaml
# config/consolidation_config.yaml
consolidation:
  confidence_threshold: 0.7
  prefer_ner_for: ["PHONE", "EMAIL", "DATE"]  # Prefer NER for these
  prefer_llm_for: ["contract_type", "work_category"]  # Prefer LLM for these
  fuzzy_match_threshold: 0.85
  max_date_variance_days: 1
  
validation:
  strict_date_format: true
  validate_phone_length: true
  validate_email_format: true
  check_logical_consistency: true

reasoning:
  include_ocr_excerpt: true
  max_excerpt_length: 50
  language: "ko"  # Korean explanations
```

## 7. Example Output

```json
{
  "consolidated_metadata": {
    "contract_type": "저작재산권 비독점적 이용허락 계약서",
    "rights_holder": "집건에",
    "user": "국립생태원 멸종위기종복원센터",
    "signature_date": "2024-01-15",
    ...
  },
  "validation_report": {
    "confidence_score": 0.92,
    "field_decisions": [
      {
        "field": "signature_date",
        "llm_value": "2024-01-15",
        "ner_value": ["2024-01-15"],
        "final_value": "2024-01-15",
        "decision": "AGREED",
        "conflict_resolved": false,
        "reasoning": "LLM and NER both extracted identical date",
        "confidence": 1.0
      },
      {
        "field": "rights_holder",
        "llm_value": "집건에",
        "ner_value": ["집건에"],
        "final_value": "집건에",
        "decision": "AGREED",
        "conflict_resolved": false,
        "reasoning": "NER NAME entity '집건에' matches LLM value",
        "confidence": 0.95
      }
    ],
    "summary": {
      "total_fields": 15,
      "agreed": 12,
      "conflicted": 2,
      "llm_only": 1,
      "ner_only": 0
    }
  },
  "timestamp": "2025-01-28T10:30:00",
  "processing_time": 2.3
}
```

## 8. Integration with Existing System

### In `app.py`:
```python
# After LLM and NER extraction:
from module.consolidator import ConsolidationAgent

agent = ConsolidationAgent(model_name=model_name)
consolidated_result = agent.consolidate(
    llm_result=llm_result,
    ner_result=ner_result,
    ocr_text=ocr_text,
    document_type=document_type
)

# Add to response
response["metadata"] = consolidated_result.consolidated_metadata
response["validation_report"] = consolidated_result.validation_report
```

## 9. Future Enhancements

1. **Learning from corrections**: Save user corrections to improve future consolidations
2. **Active learning**: Request human validation for low-confidence fields
3. **Multi-language support**: Extend to English/Chinese documents
4. **Advanced NLP**: Use semantic similarity (e.g., Sentence-BERT) for better matching
5. **Performance optimization**: Batch processing, caching, parallel validation

