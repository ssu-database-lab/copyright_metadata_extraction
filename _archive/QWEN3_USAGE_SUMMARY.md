# Qwen3-Next-80B Usage Summary

## Quick Answer

**Qwen3-Next-80B is used in the Consolidation Agent step**, which occurs AFTER both LLM extraction and NER extraction are complete.

## Visual Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  COMPLETE PROCESS FLOW                       │
└─────────────────────────────────────────────────────────────┘

1. Document Upload
   │
   ▼
2. OCR Processing
   ├─ Google / Mistral / Alibaba OCR
   └─ Output: OCR Text
   │
   ├─────────────────────────────┐
   │                             │
   ▼                             ▼
3. LLM Extraction           4. NER Extraction
   ├─ Model: User Choice         ├─ Model: klue-roberta-large
   │  (e.g., solar-ko)          │  (or google-bert, etc.)
   └─ Output: Structured         └─ Output: Entity List
      Metadata                       [(text, type), ...]
   │                             │
   │                             │
   └──────────────┬──────────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  PRE-CONSOLIDATION│
         │   (No LLM used)  │
         ├──────────────────┤
         │ • Field Mapping  │
         │ • Format Checks   │
         └──────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────────┐
    │  CONSOLIDATION AGENT                │
    │  🤖 Qwen3-Next-80B                  │
    │  (Alibaba Cloud API)                │
    │                                     │
    │  • Compares LLM vs NER             │
    │  • Makes intelligent decisions      │
    │  • Generates reasoning              │
    │  • Produces consolidated result     │
    └─────────────────────────────────────┘
                  │
                  ▼
         ┌──────────────────┐
         │   POST-PROCESS   │
         │   (No LLM used)  │
         ├──────────────────┤
         │ • Parse response │
         │ • Generate evidence│
         │ • Calculate confidence│
         └──────────────────┘
                  │
                  ▼
         5. Final Output
         ├─ Consolidated Metadata
         ├─ Validation Report
         └─ Evidence & Reasoning
```

## Exact Location in Code Flow

### Step-by-Step Breakdown:

**STEP 1-2: Extraction (User's Choice Models)**
```python
# In app.py /api/llm-extract endpoint

# Step 1: OCR (no LLM)
ocr_result = processor.process_single_file(str(upload_path))

# Step 2: LLM Extraction (USER'S CHOICE)
llm_result = llm_processor.extract_metadata_from_text(
    text=ocr_text,
    model_name=model_name  # ← User chooses (e.g., "solar-ko")
)

# Step 3: NER Extraction (NER MODELS)
ner_result = ner_predict(
    str(ocr_dir),
    str(ner_dir),
    model_name=ner_model_name  # ← User chooses (e.g., "klue-roberta-large")
)
```

**STEP 3: Consolidation (Qwen3-Next-80B - FIXED)**
```python
# Step 4: Consolidation (QWEN3-NEXT-80B - FIXED MODEL)
from module.consolidator import ConsolidationAgent

# Initialize with Qwen3-Next-80B (fixed, not user choice)
agent = ConsolidationAgent(
    model_name="alibaba-qwen3-next-80b-a3b-instruct"  # ← ALWAYS THIS MODEL
)

# Call consolidation (this is where Qwen3-Next-80B is invoked)
consolidated = agent.consolidate(
    llm_result=llm_result,
    ner_result=ner_result,
    ocr_text=ocr_text,
    document_type=document_type
)
```

## What Qwen3-Next-80B Does

When the Consolidation Agent calls Qwen3-Next-80B, it:

1. **Receives**:
   - LLM extracted metadata (structured JSON)
   - NER extracted entities (list of tuples)
   - Original OCR text (for context)

2. **Processes**:
   - Compares each field from LLM vs NER
   - Makes decisions on which value to use
   - Generates reasoning in Korean
   - Calculates confidence scores

3. **Returns**:
   - Consolidated metadata (final merged result)
   - Decision log (for each field)
   - Reasoning explanations
   - Evidence objects

## Model Configuration

**Model ID**: `alibaba-qwen3-next-80b-a3b-instruct`
- **Provider**: Alibaba Cloud (DashScope)
- **API Endpoint**: `https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation`
- **Region**: Singapore
- **API Key**: `DASHSCOPE_API_KEY` environment variable
- **Status**: ✅ Already configured in your codebase

## Key Points

1. **Qwen3-Next-80B is ONLY used for consolidation**, not extraction
2. **It's a fixed choice** - not user-configurable (ensures consistent quality)
3. **Called via API** - no local model loading needed
4. **Runs once per document** - after both extractions complete
5. **Purpose**: Intelligent comparison and merging with reasoning

## Why This Model?

- **High Performance**: 80B parameters for excellent reasoning
- **Cloud-Based**: Fast API calls, no GPU needed
- **Korean Support**: Excellent Korean language understanding
- **Consistent**: Fixed model ensures reproducible consolidation logic
- **Dedicated**: Specialized for consolidation task (not used for extraction)

## Environment Setup

Make sure you have the API key set:

```bash
export DASHSCOPE_API_KEY="your_alibaba_api_key_here"
```

Or in `.env` file:
```
DASHSCOPE_API_KEY=your_alibaba_api_key_here
```

## Summary Table

| Phase | Models Used | User Choice? | Qwen3-Next-80B? |
|-------|-------------|--------------|-----------------|
| OCR | Google/Mistral/Alibaba | ✅ Yes | ❌ No |
| LLM Extraction | SOLAR-Ko/Qwen/etc. | ✅ Yes | ❌ No |
| NER Extraction | klue-roberta/google-bert | ✅ Yes | ❌ No |
| **Consolidation** | **Qwen3-Next-80B** | ❌ **No (Fixed)** | ✅ **Yes** |
| Post-Processing | None | N/A | ❌ No |

**Answer**: Qwen3-Next-80B is used **ONLY in the Consolidation Agent** step, which is the intelligent merging step after all extractions are complete.

