# Detailed Technical Diagrams - Metadata Consolidation

## 1. Module Structure & File Organization

```
api/module/
├── consolidator/                    ← NEW MODULE
│   ├── __init__.py
│   ├── consolidation_agent.py      ← Main orchestrator
│   ├── field_mapper.py             ← NER→LLM mapping
│   ├── validation_engine.py        ← Cross-validation
│   ├── reasoning_generator.py      ← Evidence creation
│   ├── config/
│   │   └── consolidation_config.yaml
│   └── schemas/
│       └── consolidation_schemas.py
│
├── llm_extraction/                  ← EXISTING
│   └── (current structure)
│
├── ner/                             ← EXISTING
│   └── (current structure)
│
└── ocr/                             ← EXISTING
    └── (current structure)
```

## 2. Class Diagram (UML-style)

```
┌─────────────────────────────────────────────────────────────┐
│                  ConsolidationAgent                          │
├─────────────────────────────────────────────────────────────┤
│ - llm_extractor: LLMExtractionProcessor                     │
│ - field_mapper: FieldMapper                                 │
│ - validator: ValidationEngine                               │
│ - reasoner: ReasoningGenerator                              │
│ - config: Dict[str, Any]                                    │
├─────────────────────────────────────────────────────────────┤
│ + consolidate(llm_result, ner_result, ocr_text)            │
│ + _compare_fields(llm_metadata, ner_entities)               │
│ + _make_decision(field, llm_val, ner_val)                   │
│ + _generate_reasoning(decision, evidence)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ uses
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FieldMapper                             │
├─────────────────────────────────────────────────────────────┤
│ - entity_to_field_map: Dict[str, List[str]]                 │
│ - context_window: int                                        │
│ - fuzzy_threshold: float                                     │
├─────────────────────────────────────────────────────────────┤
│ + map_entities_to_fields(ner_entities, llm_metadata)      │
│ + _find_best_match(entity, candidate_fields)                │
│ + _extract_context(entity, ocr_text)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ uses
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  ValidationEngine                            │
├─────────────────────────────────────────────────────────────┤
│ - date_format_validator: DateValidator                       │
│ - phone_validator: PhoneValidator                            │
│ - email_validator: EmailValidator                            │
│ - logical_validator: LogicalValidator                       │
├─────────────────────────────────────────────────────────────┤
│ + validate_format(field, value)                             │
│ + validate_logic(metadata)                                  │
│ + check_consistency(llm_val, ner_val)                       │
└─────────────────────────────────────────────────────────────┘
```

## 3. Data Structure Comparison

```
LLM RESULT STRUCTURE
═══════════════════════════════════════════════════════════════
{
  "success": true,
  "metadata": {
    "rights_holder": "집건에",
    "user": "국립생태원 멸종위기종복원센터",
    "signature_date": "2024-01-15",
    "granted_rights": {
      "reproduction_right": true,
      "broadcasting_right": true
    }
  },
  "confidence": 0.95,
  "model_used": "solar-ko",
  "extraction_time": 2.3
}


NER RESULT STRUCTURE
═══════════════════════════════════════════════════════════════
{
  "success": true,
  "extracted_entities": [
    ("집건에", "NAME"),
    ("국립생태원", "COMPANY"),
    ("2024-01-15", "DATE"),
    ("010-1234-5678", "PHONE")
  ],
  "statistics": {
    "entity_types_count": {
      "NAME": 1,
      "COMPANY": 1,
      "DATE": 1,
      "PHONE": 1
    }
  },
  "total_entities": 4
}


CONSOLIDATED RESULT STRUCTURE
═══════════════════════════════════════════════════════════════
{
  "consolidated_metadata": {
    "rights_holder": "집건에",
    "user": "국립생태원 멸종위기종복원센터",
    "signature_date": "2024-01-15",
    "parties": [{
      "phone": "010-1234-5678"
    }],
    "granted_rights": {
      "reproduction_right": true,
      "broadcasting_right": true
    }
  },
  "validation_report": {
    "confidence_score": 0.92,
    "total_fields": 15,
    "agreed_fields": 12,
    "conflicted_fields": 2,
    "llm_only_fields": 1,
    "ner_only_fields": 0,
    "decisions": [
      {
        "field": "signature_date",
        "llm_value": "2024-01-15",
        "ner_value": "2024-01-15",
        "final_value": "2024-01-15",
        "decision": "AGREED",
        "reasoning": "Both sources extracted identical date...",
        "confidence": 1.0,
        "evidence": {
          "llm_confidence": 0.95,
          "ner_confidence": 0.88,
          "ocr_excerpt": "... 계약 체결일: 2024-01-15 ..."
        }
      }
    ]
  },
  "processing_time": 2.5,
  "timestamp": "2025-01-28T10:30:00"
}
```

## 4. Prompt Engineering Template

```
CONSOLIDATION PROMPT STRUCTURE
═══════════════════════════════════════════════════════════════

System Message:
─────────────────────────────────────────────────────────────
당신은 한국어 문서 메타데이터 추출 전문가입니다.
두 가지 다른 추출 방법(LLM 추출과 NER 추출)의 결과를 비교하고
통합하는 것이 임무입니다.


User Message:
─────────────────────────────────────────────────────────────
다음은 문서의 OCR 텍스트입니다:

[OCR_TEXT]

다음은 LLM으로 추출한 구조화된 메타데이터입니다:

{
  "rights_holder": "집건에",
  "user": "국립생태원...",
  "signature_date": "2024-01-15",
  ...
}

다음은 NER으로 추출한 엔티티 목록입니다:

[
  ("집건에", "NAME"),
  ("국립생태원", "COMPANY"),
  ("2024-01-15", "DATE"),
  ...
]

작업:
1. 각 필드를 비교하여 일치/불일치/누락 여부를 판단하세요
2. 더 신뢰할 수 있는 값을 선택하세요
3. 선택 이유를 한국어로 설명하세요
4. 최종 통합 메타데이터를 생성하세요

출력 형식 (반드시 유효한 JSON만):
{
  "consolidated_metadata": { ... },
  "decisions": [
    {
      "field": "signature_date",
      "llm_value": "2024-01-15",
      "ner_value": "2024-01-15",
      "final_value": "2024-01-15",
      "decision": "AGREED",
      "reasoning": "...",
      "confidence": 1.0
    }
  ]
}
```

## 5. Field Mapping Rules

```
ENTITY TYPE → LLM FIELD MAPPING RULES
═══════════════════════════════════════════════════════════════

NAME Entity:
├─ Priority 1: rights_holder (if near "권리자" or "저작자")
├─ Priority 2: user (if near "이용자")
├─ Priority 3: parties[].name (if in parties section)
└─ Priority 4: Any name field

DATE Entity:
├─ Priority 1: signature_date (if near "체결일" or "서명일")
├─ Priority 2: effective_date (if near "효력발생일")
├─ Priority 3: expiration_date (if near "만료일")
└─ Priority 4: Any date field

COMPANY Entity:
├─ Priority 1: user (if organization type)
├─ Priority 2: rights_holder (if in rights section)
└─ Priority 3: parties[].company

PHONE Entity:
├─ Priority 1: parties[].phone (match with nearby NAME)
└─ Priority 2: Any phone field

EMAIL Entity:
├─ Priority 1: parties[].email (match with nearby NAME)
└─ Priority 2: Any email field

ADDRESS Entity:
├─ Priority 1: parties[].address (match with nearby NAME)
└─ Priority 2: Any address field

MONEY Entity:
├─ Priority 1: payment_amount
└─ Priority 2: Any money field
```

## 6. Validation Rules Matrix

```
VALIDATION RULE MATRIX
═══════════════════════════════════════════════════════════════

Field Type      │ Format Rule              │ Logic Rule
─────────────────────────────────────────────────────────────────
DATE            │ YYYY-MM-DD               │ start < end
PHONE           │ ^[0-9\-]+$               │ length: 10-15
EMAIL           │ email regex              │ contains @ and .
ADDRESS         │ non-empty string        │ min length: 5
MONEY           │ numeric                  │ >= 0
NAME            │ non-empty string         │ min length: 2
COMPANY         │ non-empty string         │ min length: 2
BOOLEAN         │ true/false only          │ checkbox validated
```

## 7. Decision Tree (Detailed)

```
FIELD DECISION TREE
═══════════════════════════════════════════════════════════════

Start: Compare LLM Value vs NER Value
│
├─ Both Present?
│  ├─ Yes → Continue
│  └─ No → Check Individual
│      ├─ LLM Only → Use LLM + Low Confidence Flag
│      └─ NER Only → Use NER + Medium Confidence Flag
│
├─ Values Match? (exact or fuzzy)
│  ├─ Yes → AGREED Decision
│  │   ├─ Validate Format
│  │   │   ├─ Valid → Use Value + High Confidence (0.9-1.0)
│  │   │   └─ Invalid → Flag Error + Low Confidence (0.3)
│  │   └─ Skip Logic Check (already validated)
│  │
│  └─ No → CONFLICT Decision
│      ├─ Check OCR Source
│      │   ├─ NER in OCR → Prefer NER (reason: verified source)
│      │   └─ LLM in OCR → Prefer LLM (reason: structured context)
│      │
│      ├─ Check Confidence Scores
│      │   ├─ LLM > NER + 0.2 → Use LLM (reason: higher confidence)
│      │   └─ NER > LLM + 0.2 → Use NER (reason: higher confidence)
│      │
│      └─ Check Format Validation
│          ├─ LLM Valid, NER Invalid → Use LLM
│          ├─ NER Valid, LLM Invalid → Use NER
│          └─ Both Valid → Use Higher Confidence
│
└─ Final Decision
    ├─ Generate Reasoning
    ├─ Calculate Confidence
    └─ Create Evidence Object
```

## 8. API Endpoint Integration Flow

```
BEFORE INTEGRATION (Current)
═══════════════════════════════════════════════════════════════

@app.post("/api/llm-extract")
async def llm_extract_metadata(...):
    # 1. OCR
    ocr_result = processor.process_single_file(...)
    
    # 2. LLM Extraction
    llm_result = llm_processor.extract_metadata_from_text(...)
    
    # 3. NER Extraction
    ner_result = ner_predict(...)
    
    # 4. Return Both Separately
    return {
        "metadata": llm_result.metadata,
        "entities": ner_result.entities,
        ...
    }


AFTER INTEGRATION (Enhanced)
═══════════════════════════════════════════════════════════════

@app.post("/api/llm-extract")
async def llm_extract_metadata(...):
    # 1. OCR
    ocr_result = processor.process_single_file(...)
    
    # 2. LLM Extraction
    llm_result = llm_processor.extract_metadata_from_text(...)
    
    # 3. NER Extraction
    ner_result = ner_predict(...)
    
    # 4. NEW: Consolidation
    from module.consolidator import ConsolidationAgent
    
    agent = ConsolidationAgent(model_name=model_name)
    consolidated = agent.consolidate(
        llm_result=llm_result,
        ner_result=ner_result,
        ocr_text=ocr_text,
        document_type=document_type
    )
    
    # 5. Enhanced Return
    return {
        "metadata": consolidated.consolidated_metadata,  # PRIMARY
        "validation_report": consolidated.validation_report,
        "llm_metadata": llm_result.metadata,            # ORIGINAL
        "ner_entities": ner_result.entities,            # ORIGINAL
        ...
    }
```

## 9. Performance Optimization Strategy

```
PERFORMANCE OPTIMIZATION
═══════════════════════════════════════════════════════════════

1. CACHING
   ───────────────────────────────────────────────────────────
   - Cache field mappings (rarely change)
   - Cache validation rules
   - Cache LLM model instances

2. PARALLEL PROCESSING
   ───────────────────────────────────────────────────────────
   - Validate multiple fields in parallel
   - Process entity mappings concurrently
   - Async LLM calls where possible

3. EARLY EXIT
   ───────────────────────────────────────────────────────────
   - Skip validation if both sources agree
   - Skip mapping if no NER entities
   - Fast-fail on obvious errors

4. BATCH PROCESSING
   ───────────────────────────────────────────────────────────
   - Batch field comparisons
   - Batch evidence generation
   - Batch confidence calculations

Target Performance:
   - Single document: < 2 seconds
   - Batch (10 docs): < 15 seconds
```

## 10. Testing Strategy

```
TESTING PYRAMID
═══════════════════════════════════════════════════════════════

        /\
       /  \    E2E Tests (5%)
      /____\      Full pipeline integration
     /      \     Real document samples
    /        \    
   /__________\  Integration Tests (20%)
  /            \    Component interaction
 /              \   LLM + NER + Consolidator
/________________\ Unit Tests (75%)
                      Field mapper
                      Validator
                      Reasoner
                      Confidence calculator

TEST COVERAGE
═══════════════════════════════════════════════════════════════

Unit Tests:
├─ Field mapping accuracy
├─ Validation rules
├─ Confidence calculation
└─ Reasoning generation

Integration Tests:
├─ End-to-end consolidation
├─ Error handling
├─ Fallback scenarios
└─ Performance benchmarks

E2E Tests:
├─ Real document processing
├─ Various document types
├─ Edge cases (OCR errors, hallucinations)
└─ Production-like scenarios
```

## 11. Configuration Schema

```yaml
# consolidation_config.yaml

consolidation:
  # Model Settings
  model_name: "solar-ko"  # Reuse extraction model
  cache_model: true
  
  # Confidence Settings
  confidence_threshold: 0.7
  llm_weight: 0.6
  ner_weight: 0.4
  agreement_bonus: 0.1
  
  # Field Preference
  prefer_ner_for:
    - "PHONE"
    - "EMAIL"
    - "DATE"
    - "MONEY"
  
  prefer_llm_for:
    - "contract_type"
    - "work_category"
    - "granted_rights"
  
  # Matching Settings
  fuzzy_match_threshold: 0.85
  context_window_size: 100  # characters around entity
  max_date_variance_days: 1

validation:
  # Format Validation
  strict_date_format: true
  date_pattern: "^\\d{4}-\\d{2}-\\d{2}$"
  phone_pattern: "^[0-9\\-]+$"
  email_pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
  
  # Logic Validation
  validate_phone_length: true
  min_phone_length: 10
  max_phone_length: 15
  validate_email_format: true
  check_logical_consistency: true
  
  # Date Validation
  validate_date_range: true
  max_future_date_years: 10
  min_past_date_years: 100

reasoning:
  # Output Settings
  include_ocr_excerpt: true
  max_excerpt_length: 50
  language: "ko"  # Korean explanations
  
  # Evidence Settings
  include_confidence_scores: true
  include_source_positions: true
  include_validation_status: true

performance:
  # Optimization
  enable_caching: true
  cache_ttl: 3600  # seconds
  parallel_validation: true
  max_parallel_fields: 5
```

These diagrams provide comprehensive visual documentation for your team presentation and implementation reference.

