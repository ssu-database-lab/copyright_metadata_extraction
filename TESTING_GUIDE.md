# Testing Guide for Consolidator Module

## Quick Start

### Test Components Only (No API Key Required)

This tests FieldMapper, ValidationEngine, and ReasoningGenerator:

```bash
cd api/module/consolidator
python test_components_only.py
```

Or from project root:

```bash
python api/module/consolidator/test_components_only.py
```

**Expected Output:**
- ✅ Field Mapper: Maps NER entities to LLM fields
- ✅ Validation Engine: Validates formats and logic
- ✅ Reasoning Generator: Generates evidence and reasoning

### Full Test Suite (Requires API Key)

This includes testing the ConsolidationAgent with Qwen3-Next-80B:

```bash
# Set API key first
export DASHSCOPE_API_KEY="your_api_key_here"

# Run full test
cd api/module/consolidator
python test_consolidator.py
```

**Note:** This will make actual API calls to Alibaba Cloud.

## Test Components

### 1. Field Mapper Test

Tests entity-to-field mapping:

```python
from module.consolidator import FieldMapper

mapper = FieldMapper()
mappings = mapper.map_entities_to_fields(
    ner_entities=[("집건에", "NAME"), ("2024-01-15", "DATE")],
    llm_metadata={"rights_holder": "집건에"},
    ocr_text="...",
    document_type="계약서"
)
```

**What it tests:**
- Entity type recognition
- Field name mapping
- Confidence scoring
- Nested field support

### 2. Validation Engine Test

Tests format and logic validation:

```python
from module.consolidator import ValidationEngine

validator = ValidationEngine()
is_valid, error = validator.validate_format("signature_date", "2024-01-15", "date")
errors = validator.validate_logic(metadata, "계약서")
```

**What it tests:**
- Date format validation (YYYY-MM-DD)
- Phone number validation
- Email validation
- Money validation
- Date range logic (start < end)
- Required field validation

### 3. Reasoning Generator Test

Tests evidence generation:

```python
from module.consolidator import ReasoningGenerator

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

**What it tests:**
- Korean reasoning generation
- Evidence object creation
- OCR excerpt extraction

### 4. Consolidation Agent Test

Tests full consolidation flow:

```python
from module.consolidator import ConsolidationAgent

agent = ConsolidationAgent()
result = agent.consolidate(
    llm_result=llm_result,
    ner_result=ner_result,
    ocr_text=ocr_text,
    document_type="계약서"
)
```

**What it tests:**
- Component integration
- Field mapping integration
- Validation integration
- Basic consolidation flow (Phase 1)
- Error handling

## Sample Data

The test scripts use sample data:

```python
# Sample NER entities
ner_entities = [
    ("집건에", "NAME"),
    ("국립생태원", "COMPANY"),
    ("2024-01-15", "DATE"),
    ("010-1234-5678", "PHONE"),
]

# Sample LLM metadata
llm_metadata = {
    "contract_type": "저작재산권 비독점적 이용허락 계약서",
    "rights_holder": "집건에",
    "user": "국립생태원 멸종위기종복원센터",
    "signature_date": "2024-01-15",
}

# Sample OCR text
ocr_text = """
저작재산권 비독점적 이용허락 계약서
저작자 및 저작권 이용허락자: 집건에
계약 체결일: 2024-01-15
"""
```

## Troubleshooting

### Import Errors

If you get import errors:

```bash
# Make sure you're in the right directory
cd api/module/consolidator

# Or run from project root with full path
python -m api.module.consolidator.test_components_only
```

### API Key Issues

For full tests, make sure API key is set:

```bash
# Check if set
echo $DASHSCOPE_API_KEY

# Set if not
export DASHSCOPE_API_KEY="your_key_here"
```

### No Mappings Found

If field mapper returns no mappings:

1. Check entity types match expected types (NAME, DATE, etc.)
2. Check LLM metadata has matching field names
3. Check confidence threshold (default: 0.3)

### Validation Errors

If validation fails:

1. Check date format is YYYY-MM-DD
2. Check phone format (digits and hyphens only)
3. Check date ranges (start < end)

## Expected Test Results

### Component Tests (No API Key)
```
✅ PASS: Field Mapper
✅ PASS: Validation Engine
✅ PASS: Reasoning Generator
```

### Full Tests (With API Key)
```
✅ Field Mapper: Working
✅ Validation Engine: Working
✅ Reasoning Generator: Working
✅ Consolidation Agent: Structure ready (LLM pending Phase 3)
```

## Next Steps After Testing

1. ✅ **If tests pass**: Components are working correctly
2. ✅ **If Field Mapper works**: Ready for Phase 3 (LLM integration)
3. ✅ **If Validation works**: Ready to validate real metadata
4. ✅ **If Agent initializes**: Ready to test with real API calls

## Integration Testing

To test with real data from your system:

```python
# In your application code
from module.consolidator import ConsolidationAgent

# After LLM and NER extraction
agent = ConsolidationAgent()
result = agent.consolidate(
    llm_result=llm_result,  # From llm_processor
    ner_result=ner_result,  # From ner_predict
    ocr_text=ocr_text,       # From OCR processor
    document_type="계약서"
)

# Check result
if result["success"]:
    metadata = result["consolidated_metadata"]
    report = result["validation_report"]
    print(f"Confidence: {report['confidence_score']}")
```

## Performance Notes

- **Field Mapper**: Fast (< 1ms per entity)
- **Validation Engine**: Fast (< 1ms per field)
- **Reasoning Generator**: Fast (< 1ms per evidence)
- **Consolidation Agent**: Depends on API call (~2-5 seconds)

## Success Criteria

✅ **Component Tests**: All pass without errors  
✅ **Field Mapping**: Entities map to correct fields  
✅ **Validation**: Invalid data detected correctly  
✅ **Reasoning**: Evidence generated correctly  
✅ **Agent**: Initializes without errors  

If all criteria met, you're ready for Phase 3!

