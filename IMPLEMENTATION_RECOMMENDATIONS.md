# Implementation Recommendations for Metadata Consolidation Module

## Executive Summary

After analyzing your codebase, I recommend creating a **metadata consolidation agent** that intelligently merges and validates results from both LLM extraction and NER systems. This addresses accuracy issues, hallucinations, and inconsistencies by making an LLM the "expert judge" to compare, merge, and provide reasoning.

## Key Findings from Codebase Analysis

### Current State:
1. **LLM Extraction** (`llm_extraction` module):
   - Extracts structured metadata using JSON schemas
   - Returns: `ExtractionResult` with `metadata` (Dict), `confidence`, `model_used`
   - Document types: contracts, consent forms, copyright transfers
   - Built on SOLAR-Ko, Qwen, Alibaba cloud models

2. **NER Extraction** (`ner` module):
   - Extracts 23 entity types (NAME, DATE, PHONE, etc.)
   - Returns: `extracted_entities` (List[Tuple[text, type]]), `statistics`
   - B-I-O tagging + regex fallback
   - Models: klue-roberta-large, google-bert, xlm-roberta

3. **Current Gap**:
   - Both results collected but NOT consolidated
   - `/api/llm-extract` endpoint returns both separately
   - No validation or conflict resolution
   - No reasoning/evidence for decisions

## Recommended Approach

### Phase 1: Core Architecture (Week 1)
Create the basic consolidation infrastructure:

```python
# api/module/consolidator/consolidation_agent.py

class ConsolidationAgent:
    def __init__(self, model_name="solar-ko"):
        self.llm_extractor = LLMExtractionProcessor()
        self.field_mapper = FieldMapper()
        self.validator = ValidationEngine()
    
    def consolidate(self, llm_result, ner_result, ocr_text, document_type):
        """
        1. Map NER entities to LLM fields
        2. Compare LLM vs NER per field
        3. Generate decisions with reasoning
        4. Output consolidated metadata + evidence
        """
        pass
```

### Phase 2: Field Mapping (Week 2)
Implement intelligent entity-to-field mapping:

**Key Challenges**:
- NER extracts raw entities: `[("집건에", "NAME"), ("010-1234-5678", "PHONE")]`
- LLM extracts structured fields: `{"rights_holder": "집건에"}`
- Need to map NAME → rights_holder, user, parties[].name

**Solutions**:
1. **Context-aware mapping**: Use position in text to infer field
2. **Fuzzy matching**: Handle OCR errors (e.g., "목제권" → "복제권")
3. **Semantic similarity**: Use embeddings to find best match
4. **Confidence scoring**: Combine LLM confidence + NER confidence

### Phase 3: LLM Agent Integration (Week 3)
Create the reasoning engine:

**Prompt Strategy**:
```python
consolidation_prompt = f"""
당신은 한국어 문서 메타데이터 추출 전문가입니다.

OCR 텍스트: {ocr_text}

LLM 추출 결과 (구조화된 메타데이터):
{llm_metadata}

NER 추출 결과 (엔티티):
{ner_entities}

**작업**:
1. 각 필드를 비교하여 일치/불일치 여부 판단
2. 더 신뢰할 수 있는 값 선택
3. 선택 이유 설명
4. 최종 통합 메타데이터 생성

출력 형식:
{{
  "consolidated_metadata": {{ ... }},
  "decisions": [
    {{
      "field": "rights_holder",
      "llm_value": "집건에",
      "ner_value": "집건에",
      "final_value": "집건에",
      "decision": "AGREED",
      "reasoning": "Both sources extracted identical value from OCR.",
      "confidence": 0.95
    }}
  ]
}}
"""
```

### Phase 4: Validation Engine (Week 4)
Cross-validation and quality checks:

**Validation Rules**:
1. **Format validation**: Dates in YYYY-MM-DD, phone patterns
2. **Logical validation**: start_date < end_date
3. **Presence validation**: NER entity exists in OCR text
4. **Confidence validation**: Weight by source confidence
5. **Consistency validation**: Check for contradictions

### Phase 5: Production Integration (Week 5)
Integrate into existing workflow:

**Modified `/api/llm-extract` endpoint**:
```python
# After LLM and NER extraction:
agent = ConsolidationAgent(model_name=model_name)
consolidated_result = agent.consolidate(
    llm_result=llm_result,
    ner_result=ner_result,
    ocr_text=ocr_text,
    document_type=document_type
)

# Enhanced response
response = {
    "metadata": consolidated_result.consolidated_metadata,
    "validation_report": consolidated_result.report,
    "llm_metadata": llm_result.metadata,  # Original LLM result
    "ner_entities": ner_result.entities,    # Original NER result
    "consolidation_decisions": consolidated_result.decisions,
    "confidence_score": consolidated_result.confidence,
    # ... rest of response
}
```

## Technical Recommendations

### 1. Use Existing LLM Infrastructure
**Leverage**: Your existing `LLMExtractionProcessor` and `BaseLLMExtractor`
- Reuse model loading and caching
- Use same prompt engineering patterns
- Share config management

**Benefits**:
- Consistency with existing system
- Proven reliability
- Less code duplication

### 2. Add Field Mapping Layer
**Purpose**: Bridge NER's flat entity list to LLM's structured fields

**Implementation**:
```python
class FieldMapper:
    def map_entities_to_fields(self, ner_entities, llm_metadata):
        """Map NER entities to LLM metadata fields"""
        mappings = {
            ("NAME", "rights_holder"): self._match_by_position,
            ("DATE", "signature_date"): self._match_by_date_format,
            ("PHONE", "parties[].phone"): self._match_by_pattern,
        }
        # ... implementation
```

### 3. Implement Confidence Scoring
**Formula**:
```python
final_confidence = (
    llm_confidence * llm_weight 
    + ner_confidence * ner_weight 
    + agreement_bonus
)
```

Where:
- `agreement_bonus` = 0.2 if both sources agree
- `llm_weight` = 0.6 (default, configurable)
- `ner_weight` = 0.4 (default, configurable)

### 4. Generate Evidence Report
**Purpose**: Provide audit trail for downstream systems

**Format**:
```json
{
  "field": "signature_date",
  "llm_value": "2024-01-15",
  "ner_value": "2024-01-15",
  "final_value": "2024-01-15",
  "decision": "AGREED",
  "reasoning": "Both LLM and NER extracted '2024-01-15' from OCR text at position [120:130].",
  "evidence": {
    "llm_confidence": 0.95,
    "ner_confidence": 0.88,
    "ocr_excerpt": "... 계약 체결일: 2024-01-15 ..."
  }
}
```

## Suggested File Structure

```
api/module/consolidator/
├── __init__.py
├── consolidation_agent.py       # Main orchestrator (LLM agent)
├── field_mapper.py              # NER→LLM field mapping
├── validation_engine.py         # Cross-validation
├── reasoning_generator.py       # Evidence generation
├── config/
│   └── consolidation_config.yaml
└── schemas/
    └── consolidation_schemas.py
```

## Best Practices

### 1. Graceful Degradation
- If NER fails → use LLM result with warning
- If consolidation fails → return both results separately
- Log all failures for debugging

### 2. Performance Optimization
- Cache field mappings (rarely change)
- Use async LLM calls
- Parallel validation where possible
- Target <2s consolidation time

### 3. Extensibility
- Make thresholds configurable
- Support custom field mappings
- Plugin architecture for validators
- Document type-specific rules

### 4. Testing Strategy
- Unit tests: each component
- Integration tests: full pipeline
- Edge cases: OCR errors, hallucinations
- Performance tests: large documents

## Expected Output Example

```json
{
  "request_id": "20251027_072025",
  "filename": "contract_sample.pdf",
  
  "consolidated_metadata": {
    "contract_type": "저작재산권 비독점적 이용허락 계약서",
    "rights_holder": "집건에",
    "user": "국립생태원 멸종위기종복원센터",
    "signature_date": "2024-01-15",
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
        "reasoning": "Both sources extracted identical date from OCR text.",
        "confidence": 1.0
      }
    ]
  },
  
  "llm_metadata": { ... },  // Original LLM result
  "ner_entities": [ ... ]   // Original NER result
}
```

## Success Metrics

- **Accuracy**: >95% field accuracy (measured vs human labels)
- **Conflicts resolved**: >90% conflict resolution rate
- **Processing time**: <2s for typical document
- **Confidence correlation**: Consolidated confidence correlates with actual accuracy

## Next Steps

1. **Review this proposal** with your team
2. **Choose implementation priority** (basic → advanced features)
3. **Create initial implementation** following Phase 1
4. **Test on real documents** from your system
5. **Iterate based on results**

## Open Questions for Discussion

1. **Model choice**: Same LLM as extraction, or dedicated model?
   - **My recommendation**: Same model (leverages caching)

2. **When to use**: Always consolidate, or configurable?
   - **My recommendation**: Make it optional with `consolidate=true` flag

3. **Threshold tuning**: How to determine confidence thresholds?
   - **My recommendation**: Start with 0.7, tune based on validation results

4. **Storage**: Store reasoning/evidence for audit trail?
   - **My recommendation**: Yes, store in `validation_report.json`

5. **UI display**: Show consolidated or original results in frontend?
   - **My recommendation**: Show both (consolidated as primary, original on hover/expand)

