# Pipeline Orchestrator Guide

## Background: The Problem Before

`app.py` was a 1,420-line "god file" where the `/api/llm-extract` handler alone was 385 lines doing everything: file validation, file saving, OCR processing, LLM extraction, NER extraction, consolidation, response building, file saving, SSE progress updates, and error handling — all in one function.

```
app.py (1 file, 1,420 lines)
└── /api/llm-extract handler (385 lines!)
    ├── File validation
    ├── File saving
    ├── OCR processing (40 lines)
    ├── LLM extraction (15 lines)
    ├── NER extraction (50 lines)
    ├── Consolidation (70 lines)
    ├── Response building (35 lines)
    ├── File saving (15 lines)
    ├── SSE progress updates (scattered everywhere)
    ├── Error handling (scattered try/except)
    └── Non-streaming collector (40 lines of SSE parsing hack)
```

Problems:
- Business logic mixed with HTTP handling
- `locals()` checks like `request_id if 'request_id' in locals() else None` — a sign of unmanageable scope
- Bare `except:` with `continue` silently swallowing errors
- Two near-duplicate pipeline functions sharing ~70% code
- Impossible to test pipeline logic without starting the web server

---

## The Solution: Separation of Concerns

The **Service Layer Pattern** separates:
- **Route handlers** (`app.py`) — HTTP concerns: read request, validate input, call service, return response
- **Business logic** (`pipeline.py`) — the actual processing pipeline

```
pipeline.py (business logic)          app.py (HTTP handling)
┌─────────────────────────┐          ┌──────────────────────┐
│ PipelineOrchestrator    │          │ /api/llm-extract     │
│                         │          │                      │
│  setup()       → ctx    │ ←────── │  Reads file          │
│  run_ocr()     → text   │          │  Validates input     │
│  run_llm()     → meta   │          │  Calls pipeline      │
│  run_ner()     → ents   │          │  Sends SSE updates   │
│  run_consolidation()    │          │  Returns JSON        │
│  build_response() → {}  │          │                      │
│  save_results()         │          │  (~70 lines total)   │
│                         │          └──────────────────────┘
│  run() → full pipeline  │
│  (non-streaming shortcut)│
└─────────────────────────┘
```

---

## PipelineOrchestrator API

### Constructor

```python
pipeline = PipelineOrchestrator(
    llm_processor=llm_processor,           # LLMExtractionProcessor instance
    ner_predict_fn=ner_predict,            # NER prediction function
    available_ner_models=AVAILABLE_MODELS, # Dict of NER model configs
    upload_dir=UPLOAD_DIR,                 # Path to uploads directory
    results_dir=RESULTS_DIR,              # Path to results directory
)
```

### Methods

#### `setup(file_bytes, filename) → ctx`
Saves the uploaded file and prepares directories. Returns a **context dict** that all subsequent stages use.

```python
ctx = pipeline.setup(file_bytes, filename)
# ctx = {
#     "request_id": "20260327_123456_789012",
#     "filename": "contract.pdf",
#     "upload_path": Path("uploads/20260327_.../contract.pdf"),
#     "result_dir": Path("results/20260327_.../"),
#     "file_size_mb": 0.35,
#     "start_time": datetime(2026, 3, 27, 12, 34, 56),
# }
```

#### `run_ocr(ctx, provider, model) → (ocr_text, ocr_result)`
Stage 1: Run OCR on the uploaded file. Returns the extracted text and the raw OCR result dict.

```python
ocr_text, ocr_result = pipeline.run_ocr(ctx, "alibaba", "qwen3-vl-235b-a22b-instruct")
# ocr_text = "저작물 저작재산권 양도 계약서\n※ 저작재산권 양도 취지..."
# ocr_result = {"status": "success", "full_text": "...", "total_pages": 3, ...}
```

#### `run_llm(ocr_text, document_type, filename, model_name) → llm_result`
Stage 2: Run LLM metadata extraction on OCR text.

```python
llm_result = pipeline.run_llm(ocr_text, "계약서", "contract.pdf", "alibaba-qwen3-next-80b-a3b-instruct")
# llm_result = {
#     "success": True,
#     "metadata": {"work_title": "...", "copyright_holder": "...", ...},
#     "confidence": 0.85,
#     "extraction_time": 14.2,
#     "model_used": "Alibaba-qwen3-next-80b-a3b-instruct",
# }
```

#### `run_ner(ocr_result, result_dir, ner_model, ocr_text) → ner_result`
Stage 3: Run NER entity extraction.

```python
ner_result = pipeline.run_ner(ocr_result, ctx["result_dir"], "klue-roberta-large", ocr_text)
# ner_result = {
#     "success": True,
#     "total_entities": 15,
#     "entities": {"NAME": ["박광수"], "PHONE": ["010-1234-5678"], ...},
#     "extracted_entities": [("박광수", "NAME"), ("010-1234-5678", "PHONE"), ...],
# }
```

#### `run_consolidation(llm_result, ner_result, ocr_text, document_type, result_dir, consolidation_model) → (result, success, error)`
Stage 4: Consolidate LLM + NER results using an LLM arbiter.

```python
con_result, con_success, con_error = pipeline.run_consolidation(
    llm_result, ner_result, ocr_text, "계약서",
    ctx["result_dir"], "alibaba-qwen3-next-80b-a3b-instruct"
)
# con_result = {
#     "consolidated_metadata": {"work_title": "...", "copyright_holder": "...", ...},
#     "validation_report": {
#         "decisions": [{"field": "...", "decision": "AGREED", ...}, ...],
#         "summary": {"total_fields": 27, "agreed_fields": 18, ...},
#     },
#     "success": True,
# }
# con_success = True
# con_error = None
```

#### `build_response(ctx, **kwargs) → response_dict`
Assembles the complete API response from all stage results.

```python
response = pipeline.build_response(
    ctx,
    model_name=..., document_type=...,
    ocr_text=..., ocr_provider=..., ocr_model=...,
    llm_result=..., ner_model=..., ner_result=...,
    consolidate=..., consolidation_model=...,
    consolidation_result=..., consolidation_success=..., consolidation_error=...,
)
# response = the full JSON structure returned by the API
```

#### `save_results(result_dir, response, consolidation_result, consolidation_success)`
Writes `llm_metadata.json` and `consolidated_metadata.json` to disk.

#### `run(file_bytes, filename, **kwargs) → response_dict`
Convenience method: runs the full pipeline in one call (non-streaming).
Internally calls: `setup → run_ocr → run_llm → run_ner → run_consolidation → build_response → save_results`

---

## How Each Mode Works in app.py

### Non-streaming mode (simple API call → JSON response)

The client sends a POST request and waits for the complete result:

```python
# app.py — just 6 lines of business logic
result = pipeline_orchestrator.run(
    file_content, filename,
    model_name=..., document_type=..., ...
)
return JSONResponse(content=result)
```

The `run()` method handles everything internally:
```
run() → setup() → run_ocr() → run_llm() → run_ner() → run_consolidation() → build_response() → save_results()
```

### SSE streaming mode (real-time progress to frontend)

The client gets Server-Sent Events (SSE) with progress updates as each stage completes:

```python
# app.py — async generator calls stages one by one with yields between
async def process_with_progress():
    ctx = pipeline_orchestrator.setup(file_content, filename)
    yield progress("파일 업로드 완료", 10%)       # ← client sees this immediately

    ocr_text, ocr_result = pipeline_orchestrator.run_ocr(ctx, ...)
    yield progress("OCR 완료", 40%)               # ← client sees this after OCR finishes

    llm_result = pipeline_orchestrator.run_llm(ocr_text, ...)
    yield progress("LLM 완료", 70%)               # ← client sees this after LLM finishes

    ner_result = pipeline_orchestrator.run_ner(...)
    yield progress("NER 완료", 90%)               # ← client sees this after NER finishes

    con_result = pipeline_orchestrator.run_consolidation(...)
    yield progress("통합 완료", 98%)              # ← client sees this after consolidation

    response = pipeline_orchestrator.build_response(ctx, ...)
    yield progress("처리 완료", 100%, result=response)  # ← final result
```

The `yield` between stages is what makes it real-time — each SSE message is sent to the client as soon as a stage completes, so the frontend can update a progress bar.

---

## Data Flow Through the Pipeline

```
User uploads PDF
       │
       ▼
   setup()
   ├── save file to uploads/{request_id}/
   ├── create results/{request_id}/
   └── return ctx = {request_id, filename, upload_path, result_dir, ...}
       │
       ▼
   run_ocr(ctx, provider, model)
   ├── Creates UniversalOCRProcessor
   ├── Calls Alibaba/Google/Mistral API to extract text from images
   ├── If primary provider fails → automatic fallback chain (alibaba → google → mistral → naver)
   ├── Timeout: 60s per request, 3 retries on transient errors
   └── Returns (ocr_text: str, ocr_result: dict)
       │
       ▼
   run_llm(ocr_text, document_type, filename, model_name)
   ├── LLMExtractionProcessor sends OCR text + unified schema to Alibaba Cloud API
   ├── Uses the unified schema (67 fields covering all document types)
   ├── The LLM fills in fields it can find, returns null for the rest
   ├── Timeout: 60s, 3 retries
   └── Returns llm_result: {success, metadata: {67 fields}, confidence, ...}
       │
       ▼
   run_ner(ocr_result, result_dir, ner_model, ocr_text)
   ├── Loads BERT-CRF model (KLUE-RoBERTa-Large by default)
   ├── Runs BIO tagging on OCR text → extracts named entities
   ├── Regex fallback for entities the model misses
   ├── Runs on CPU (no GPU needed, ~6s per document)
   └── Returns ner_result: {entities: {NAME: [...], PHONE: [...]}, total_entities: N}
       │
       ▼
   run_consolidation(llm_result, ner_result, ocr_text, ...)
   ├── ConsolidationAgent receives both LLM metadata and NER entities
   ├── Calls Qwen3-Next-80B as an "expert judge" to compare and merge
   ├── For each field, decides: AGREED / CONFLICT / LLM_ONLY / NER_ONLY / MISSING
   ├── If JSON parsing fails → fallback to Qwen-Max
   ├── Timeout: 90s (longer responses)
   └── Returns (consolidated_result, success: bool, error: str|None)
       │
       ▼
   build_response(ctx, all_results...)
   ├── Assembles the complete API response from all stages
   ├── Calculates total processing time
   ├── Formats NER entities for display
   └── Returns the response dict (same structure as api_response_structure.json)
       │
       ▼
   save_results(result_dir, response, ...)
   ├── Writes results/{request_id}/llm_metadata.json (full response)
   └── Writes results/{request_id}/consolidated_metadata.json (if consolidation succeeded)
```

---

## Why This Design?

### 1. Route handlers stay thin
The `/api/llm-extract` handler in `app.py` is now ~70 lines instead of 385. It only handles HTTP concerns: reading the file, validating input, calling the pipeline, and formatting the response.

### 2. Pipeline logic is testable
`PipelineOrchestrator` can be tested independently — just instantiate it with mock dependencies and call `run()`. No need to start a web server.

### 3. Stages are independently callable
For SSE streaming, the handler calls each stage separately with `yield` between them. For non-streaming, `run()` calls all stages in sequence. Same logic, two modes.

### 4. No more `locals()` hacks
Before: `request_id if 'request_id' in locals() else None` — a fragile check because variables might not be defined if an earlier step failed.
After: `ctx["request_id"]` — always defined because `setup()` runs first and creates the context dict.

### 5. No more bare except
Before: `except: continue` in the SSE parser silently swallowed all errors.
After: Non-streaming mode doesn't parse SSE at all — it just calls `run()` and returns the result.

---

## File Locations

| File | Lines | Purpose |
|------|-------|---------|
| `api/web/pipeline.py` | ~280 | PipelineOrchestrator class |
| `api/web/app.py` | ~1,165 | FastAPI routes (down from 1,420) |
| `docs/api_response_structure.json` | — | Expected API response format |
