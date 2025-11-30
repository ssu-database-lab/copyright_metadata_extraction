# Consolidator Module

Intelligent metadata consolidation module that merges and validates results from both LLM extraction and NER extraction systems.

## Overview

The consolidator module uses **Qwen3-Next-80B** (via Alibaba Cloud API) as an expert judge to:
- Compare LLM and NER extraction results
- Resolve conflicts intelligently
- Generate evidence and reasoning for each decision
- Produce validated, consolidated metadata

## Components

### 1. FieldMapper
Maps NER entities to LLM metadata fields.

**Features:**
- Maps 13+ entity types to 30+ LLM fields
- Priority-based matching
- Confidence scoring
- Nested field support

### 2. ValidationEngine
Validates metadata formats and logical consistency.

**Features:**
- Format validation (dates, phones, emails, money)
- Logical consistency checks
- Cross-validation between sources
- Auto field type detection

### 3. ConsolidationAgent
Main orchestrator that uses Qwen3-Next-80B.

**Features:**
- Integrates all components
- LLM-based consolidation
- Error handling
- Result formatting

### 4. ReasoningGenerator
Generates evidence and reasoning for decisions.

**Features:**
- Korean reasoning explanations
- Evidence object creation
- OCR excerpt extraction

## Installation

No additional installation required. Uses existing dependencies.

## Configuration

### Environment Variables

```bash
# Required for Qwen3-Next-80B (Alibaba Cloud)
export DASHSCOPE_API_KEY="your_api_key_here"
```

Or in `.env` file:
```
DASHSCOPE_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
from module.consolidator import ConsolidationAgent

# Initialize agent
agent = ConsolidationAgent(
    model_name="alibaba-qwen3-next-80b-a3b-instruct"
)

# Consolidate results
result = agent.consolidate(
    llm_result=llm_result,
    ner_result=ner_result,
    ocr_text=ocr_text,
    document_type="계약서"
)

# Access results
consolidated_metadata = result["consolidated_metadata"]
validation_report = result["validation_report"]
```

### Individual Components

```python
from module.consolidator import FieldMapper, ValidationEngine, ReasoningGenerator

# Field Mapping
mapper = FieldMapper()
mappings = mapper.map_entities_to_fields(
    ner_entities=[("집건에", "NAME"), ("2024-01-15", "DATE")],
    llm_metadata={"rights_holder": "집건에"},
    ocr_text="...",
    document_type="계약서"
)

# Validation
validator = ValidationEngine()
errors = validator.validate_logic(metadata, "계약서")
is_valid, error = validator.validate_format("signature_date", "2024-01-15", "date")

# Reasoning
reasoner = ReasoningGenerator()
evidence = reasoner.generate_evidence(
    field_name="signature_date",
    llm_value="2024-01-15",
    ner_value="2024-01-15",
    final_value="2024-01-15",
    decision="AGREED",
    ocr_text="...",
    confidence=1.0
)
```

## Testing

Run the test suite:

```bash
cd api/module/consolidator
python test_consolidator.py
```

Or from project root:

```bash
python api/module/consolidator/test_consolidator.py
```

## Output Format

```json
{
  "success": true,
  "consolidated_metadata": {
    "contract_type": "저작재산권 비독점적 이용허락 계약서",
    "rights_holder": "집건에",
    "signature_date": "2024-01-15",
    ...
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
        "reasoning": "Both sources extracted identical date",
        "confidence": 1.0,
        "evidence": {
          "ocr_excerpt": "... 계약 체결일: 2024-01-15 ..."
        }
      }
    ],
    "validation_errors": []
  },
  "model_used": "alibaba-qwen3-next-80b-a3b-instruct"
}
```

## Implementation Status

### ✅ Phase 1: Core Architecture (Complete)
- Module structure
- Field mapping
- Validation engine
- Agent skeleton

### ⏳ Phase 3: LLM Integration (In Progress)
- Consolidation prompt (skeleton)
- Schema definition
- Response parsing
- Decision logic

## Known Limitations

1. **LLM Consolidation**: Full implementation pending Phase 3
2. **Field Mapping**: Basic implementation (context-aware matching in Phase 2)
3. **Fuzzy Matching**: OCR error tolerance pending Phase 2

## Error Handling

The module handles errors gracefully:
- Missing API key → Returns error with helpful message
- Network issues → Returns error result
- Invalid inputs → Validates and returns errors
- Missing fields → Uses available sources with warnings

## Integration

To integrate with `/api/llm-extract` endpoint:

```python
from module.consolidator import ConsolidationAgent

# After LLM and NER extraction
agent = ConsolidationAgent()
consolidated = agent.consolidate(
    llm_result=llm_result,
    ner_result=ner_result,
    ocr_text=ocr_text,
    document_type=document_type
)

# Add to response
response["metadata"] = consolidated["consolidated_metadata"]
response["validation_report"] = consolidated["validation_report"]
```

## Support

For issues or questions, refer to:
- `DESIGN_PROPOSAL.md` - Full design documentation
- `IMPLEMENTATION_RECOMMENDATIONS.md` - Implementation guide
- `CONSOLIDATOR_PHASE1_COMPLETE.md` - Phase 1 completion status

