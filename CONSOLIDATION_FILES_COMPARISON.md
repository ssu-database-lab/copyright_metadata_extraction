# Consolidation Files Comparison

## Overview

When consolidation is enabled, the system generates two JSON files:
1. **`consolidated_metadata.json`** - Raw output from ConsolidationAgent
2. **`llm_metadata.json`** - Complete API response from `/api/llm-extract`

## File Structure Comparison

### `consolidated_metadata.json` (ConsolidationAgent Output)

This is the **raw output** directly from `ConsolidationAgent.consolidate()`. It contains:

```json
{
  "success": true,
  "consolidated_metadata": { ... },      // ✅ FINAL RESULT - Validated & merged metadata
  "validation_report": {
    "confidence_score": 0.72,
    "total_fields": 43,
    "agreed_fields": 2,
    "conflicted_fields": 5,
    "llm_only_fields": 32,
    "ner_only_fields": 0,
    "missing_fields": 6,
    "validation_errors": [],
    "decisions": [ ... ],                // Field-by-field decisions with reasoning
    "summary": { ... }                   // Summary statistics
  },
  "llm_metadata": { ... },               // Original LLM extraction (for reference)
  "ner_entities": [ ... ],              // Original NER extraction (for reference)
  "model_used": "alibaba-qwen3-next-80b-a3b-instruct",
  "status": "completed",
  "llm_confidence": 0.72,
  "fallback_used": false
}
```

**Purpose**: Internal consolidation result with detailed validation report

---

### `llm_metadata.json` (API Response)

This is the **complete API response** from `/api/llm-extract`. It contains:

```json
{
  "success": true,
  "request_id": "20251110_133106_992874",
  "filename": "...",
  "file_size_mb": 5.23,
  "model_used": "Alibaba-qwen3-next-80b-a3b-instruct",
  "document_type": "계약서",
  
  "metadata": { ... },                   // Original LLM extraction (for reference)
  
  "consolidated_metadata": { ... },     // ✅ FINAL RESULT - Same as consolidated_metadata.json
  
  "consolidation_decisions": [ ... ],    // Same as validation_report.decisions
  "consolidation_summary": { ... },      // Same as validation_report.summary
  "consolidation_confidence": 0.72,      // Same as validation_report.confidence_score
  "consolidation_model_used": "alibaba-qwen3-next-80b-a3b-instruct",
  "consolidation_fallback_used": false,
  
  "ocr_text": "...",                     // Full OCR text
  "ocr_provider": "alibaba",
  "ocr_model": "qwen3-vl-235b-a22b-instruct",
  "entities": { ... },                   // NER entity counts
  "entity_count": 15,
  "ner_model": "Google mBERT",
  "processing_time": 102.66,
  ...
}
```

**Purpose**: Complete API response with all context (OCR, NER, LLM, consolidation)

---

## Key Differences

| Aspect | `consolidated_metadata.json` | `llm_metadata.json` |
|--------|------------------------------|---------------------|
| **Source** | Direct ConsolidationAgent output | Complete API response |
| **Final Metadata** | `consolidated_metadata` | `consolidated_metadata` (same content) |
| **Original Metadata** | `llm_metadata` | `metadata` (same content) |
| **Decisions** | `validation_report.decisions` | `consolidation_decisions` (same content) |
| **Summary** | `validation_report.summary` | `consolidation_summary` (same content) |
| **Confidence** | `validation_report.confidence_score` | `consolidation_confidence` (same value) |
| **Additional Context** | NER entities only | OCR text, NER stats, processing info |
| **Request Info** | None | request_id, filename, file_size_mb |
| **Processing Info** | None | processing_time, extraction_time |
| **Model Info** | `model_used`, `fallback_used` | `consolidation_model_used`, `consolidation_fallback_used` |

---

## Which File Should You Use?

### ✅ **Use `llm_metadata.json` as the Final Result**

**Reasons:**

1. **Complete Context**: Contains all information in one place:
   - Original LLM metadata (`metadata`)
   - Final consolidated metadata (`consolidated_metadata`) ✅
   - Consolidation decisions and reasoning
   - OCR text (for verification)
   - NER entities and statistics
   - Processing metadata (request_id, timestamps, etc.)

2. **API Standard**: This is the official API response format
   - All frontend/backend integrations expect this structure
   - Consistent with API documentation

3. **Traceability**: Includes `request_id` for tracking and debugging

4. **Self-Contained**: Everything needed is in one file

### 📋 **Use `consolidated_metadata.json` for:**

- **Deep Analysis**: When you need the raw consolidation output
- **Debugging**: To understand consolidation internals
- **Development**: When working directly with ConsolidationAgent

---

## Final Result Location

**The final consolidated metadata is located at:**

```json
// In llm_metadata.json (RECOMMENDED)
{
  "consolidated_metadata": { ... }  // ✅ Use this as final result
}

// OR in consolidated_metadata.json
{
  "consolidated_metadata": { ... }  // Same content, but less context
}
```

**Both files contain the same `consolidated_metadata` content**, but `llm_metadata.json` provides more complete context.

---

## Example Usage

### Recommended: Use `llm_metadata.json`

```python
import json

# Load the complete API response
with open('llm_metadata.json', 'r', encoding='utf-8') as f:
    result = json.load(f)

# Get final consolidated metadata
final_metadata = result['consolidated_metadata']  # ✅ Final result

# Compare with original if needed
original_metadata = result['metadata']  # Original LLM extraction

# Review consolidation decisions
decisions = result['consolidation_decisions']  # Field-by-field reasoning

# Check consolidation quality
summary = result['consolidation_summary']
confidence = result['consolidation_confidence']

print(f"Final metadata: {len(final_metadata)} fields")
print(f"Confidence: {confidence:.2%}")
print(f"Agreed fields: {summary['agreed_fields']}")
print(f"Conflicted fields: {summary['conflicted_fields']}")
```

### Alternative: Use `consolidated_metadata.json`

```python
import json

# Load consolidation result
with open('consolidated_metadata.json', 'r', encoding='utf-8') as f:
    result = json.load(f)

# Get final consolidated metadata
final_metadata = result['consolidated_metadata']  # ✅ Final result

# Review detailed validation report
validation_report = result['validation_report']
decisions = validation_report['decisions']
summary = validation_report['summary']
confidence = validation_report['confidence_score']
```

---

## Summary

| Question | Answer |
|----------|--------|
| **Which file has the final result?** | Both files have the same `consolidated_metadata` |
| **Which file should I use?** | **`llm_metadata.json`** (recommended) - more complete |
| **What's the difference?** | `llm_metadata.json` includes API context (OCR, NER stats, request info) |
| **Can I use either?** | Yes, but `llm_metadata.json` is more comprehensive |

---

## Quick Reference

```json
// Final consolidated metadata (same in both files)
llm_metadata.json["consolidated_metadata"]           ✅ FINAL RESULT
consolidated_metadata.json["consolidated_metadata"]   ✅ FINAL RESULT (same content)

// Original LLM extraction (for comparison)
llm_metadata.json["metadata"]                        📋 Original
consolidated_metadata.json["llm_metadata"]            📋 Original (same content)

// Consolidation decisions (same in both)
llm_metadata.json["consolidation_decisions"]         🔍 Decisions
consolidated_metadata.json["validation_report"]["decisions"]  🔍 Decisions (same content)
```

**Recommendation**: Use `llm_metadata.json` as your primary source for the final result, as it contains all necessary context in a single, well-structured file.

