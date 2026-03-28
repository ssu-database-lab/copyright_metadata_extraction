# Architecture Fixes Guide

This document explains every fix applied from the Architecture Assessment (19 out of 22 issues resolved). Each section describes: what the problem was, why it mattered, and how it was fixed.

---

## Phase 1: Eliminate Confusion

### Fix #1 & #2: Delete Legacy Code

**Problem:** Three parallel implementations of LLM extraction existed:
- `/llm_extraction/` (legacy, 3,657 lines)
- `/extract/llm_extraction/` (nested legacy, ~500 lines)
- `/api/module/llm_extraction/` (active, 16,054 lines)

Same duplication for NER: `/ner/` (legacy) and `/api/module/ner/` (active). Also `api/call.py` (old Flask server superseded by FastAPI).

**Why it mattered:** Someone could accidentally edit the wrong copy, thinking they're fixing a bug but actually modifying dead code. The legacy code would inevitably diverge from the active code.

**Fix:** Deleted all legacy directories (`/ner/`, `/llm_extraction/`, `/extract/`) and legacy API files (`api/call.py`, `api/ner_test.py`). Only `/api/module/` remains as the single active codebase.

---

### Fix #3: Consolidate `.env` Files

**Problem:** 6 duplicate `.env` files scattered across directories:
- `api/.env`, `api/.env_alibaba`
- `api/web/.env`, `api/web/.env_alibaba`
- `OCR/google_vision/.env`, `OCR/google_vision/.env_alibaba`

Each OCR provider independently hunted for `.env` files across 3-5 different paths:
```python
# This was repeated in every OCR provider file
env_paths = [
    Path(__file__).parent.parent / ".env",
    Path(__file__).parent.parent / "web" / ".env",
    Path(__file__).parent.parent / ".env_alibaba",
    Path(__file__).parent.parent.parent / "OCR" / "google_vision" / ".env",
]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break
```

**Why it mattered:** Update one `.env` file, forget the others — keys get out of sync. Different providers might load different files. Confusing for deployment.

**Fix:**
1. Created single `.env` at project root with all API keys
2. Created `.env.example` template (safe to commit, no secrets)
3. Created `api/module/env_loader.py` — a shared loader that searches project root first, loaded automatically on import
4. Replaced the 10-15 line loading blocks in all 5 OCR providers with one line: `import module.env_loader`
5. Updated `app.py` to load from project root first

**Files:** `.env`, `.env.example`, `api/module/env_loader.py`, `app.py`, all 5 OCR provider files

---

### Fix #4: Fix model_config.yaml Path

**Problem:** The default config path was `"config/model_config.yaml"` — a relative path. This only works if the working directory happens to be `api/module/llm_extraction/`. Running from `api/` (the normal case with `uvicorn`) would cause `FileNotFoundError`.

```python
# Before — breaks when running from api/
class BaseLLMExtractor:
    def __init__(self, config_path="config/model_config.yaml"):
        self.cache_manager = ModelCacheManager(config_path)  # fails!
```

**Why it mattered:** The server would crash when trying to use local models, and the error message ("file not found: config/model_config.yaml") wouldn't clearly explain why.

**Fix:** Changed the default to be computed from `__file__` — always resolves to the absolute path regardless of working directory:
```python
# After — works from any directory
class BaseLLMExtractor:
    _DEFAULT_CONFIG = str(Path(__file__).parent.parent / "config" / "model_config.yaml")

    def __init__(self, config_path=None):
        self.cache_manager = ModelCacheManager(config_path or self._DEFAULT_CONFIG)
```

**Files:** `base_extractor.py`, `model_cache.py`

---

## Phase 2: Fix Reliability

### Fix #5: Timeout and Retry on Alibaba API Calls

**Problem:** All 3 Alibaba Cloud API call points had no timeout and no retry:
- OCR (`alibaba_ocr.py`) — calls DashScope for image-to-text
- LLM extraction (`cloud_extractor.py`) — calls DashScope for metadata extraction
- Consolidation (`consolidation_agent.py`) — calls DashScope for LLM+NER merging

Without timeout, a hanging API would block the entire pipeline **forever**. Without retry, a transient network blip would fail immediately.

**Why it mattered:** In production, network issues happen. An API might be slow (not down), and without timeout the user's request just hangs with no error. The OCR fallback chain (#7) can't trigger if the primary provider never returns an error — it just waits.

**Fix:** Added `timeout` and `max_retries` to all OpenAI client instances:
```python
# Before
self.client = OpenAI(api_key=api_key, base_url="https://dashscope-intl...")

# After
self.client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope-intl...",
    timeout=60.0,    # 60s timeout per request
    max_retries=3    # retry on 429, 500, 502, 503, 504
)
```

The OpenAI SDK handles retry with exponential backoff automatically. Consolidation gets 90s timeout (longer responses).

| Component | Timeout | Retry |
|-----------|---------|-------|
| OCR (alibaba_ocr.py) | 60s | 3x |
| LLM extraction (cloud_extractor.py) | 60s | 3x |
| Consolidation (consolidation_agent.py) | 90s | 3x (inherited) |

**Files:** `alibaba_ocr.py`, `cloud_extractor.py`, `consolidation_agent.py`

---

### Fix #6: Duplicate PHONE Priority Key

**Problem:** In `field_mapper.py`, the `PHONE` entity type was defined twice in a Python dictionary:
```python
FIELD_PRIORITY = {
    'PHONE': {                    # line 241 — first definition
        'parties[].phone': 10,
        'contact_info.phone': 9,
        'phone': 5                # phone priority = 5
    },
    'COMPANY': { ... },
    'PHONE': {                    # line 256 — second definition (OVERWRITES first!)
        'parties[].phone': 10,
        'contact_info.phone': 9,
        'phone': 8                # phone priority = 8
    },
}
```

**Why it mattered:** Python dictionaries silently overwrite duplicate keys. The first `PHONE` definition was invisible — only the second one took effect. The developer who added the second definition likely didn't realize the first existed.

**Fix:** Removed the duplicate, kept the merged version with `phone: 8` (the value Python was actually using).

**File:** `field_mapper.py`

---

### Fix #7: OCR Provider Fallback Chain

**Problem:** `UniversalOCRProcessor` initialized exactly one OCR provider. If that provider failed (API down, quota exceeded, Arrearage billing issue), the entire pipeline returned an error. Despite having 4 providers available, there was no attempt to try alternatives.

**Why it mattered:** Our DashScope account had an Arrearage error blocking all API calls. With no fallback, the entire system was unusable even though Google Vision and Mistral were available.

**Fix:** Added automatic fallback to `UniversalOCRProcessor`:
```python
class UniversalOCRProcessor:
    DEFAULT_FALLBACK_ORDER = ["alibaba", "google", "mistral", "naver"]

    def process_single_file(self, file_path):
        result = self._process_with_current_provider(file_path)

        if result["status"] == "failed" and self.fallback_enabled:
            for fallback_provider in self.fallback_chain[1:]:
                self.ocr_provider = self._create_provider(fallback_provider)
                result = self._process_with_current_provider(file_path)
                if result["status"] != "failed":
                    result["fallback_used"] = True
                    result["original_provider"] = primary_provider
                    return result

        return result
```

When the primary provider fails for an entire document, the processor tries the next one in the chain. The response includes fallback metadata so you can see what happened. Can be disabled with `fallback=False`.

**File:** `universal_ocr.py`

---

### Fix #8: Replace `locals()` Checks and Bare `except:`

**Problem:** The `/api/llm-extract` handler used fragile patterns:
```python
# locals() check — variable might not be defined if earlier step failed
"request_id": request_id if 'request_id' in locals() else None

# Bare except — silently swallows ALL errors including KeyboardInterrupt
except:
    continue
```

**Why it mattered:** `locals()` checks indicate the function has outgrown its scope — too many variables, too many paths. Bare `except:` hides bugs by swallowing errors silently.

**Fix:** Resolved by the PipelineOrchestrator refactoring (#14). The `setup()` method creates a `ctx` dict early, so `ctx["request_id"]` is always defined. The non-streaming path no longer parses SSE messages (no `except: continue` needed).

**File:** `app.py` (via fix #14)

---

## Phase 3: Reduce Duplication

### Fix #9: Collapse 9 Model Extractors into 1 Class

**Problem:** `base_extractor.py` contained 9 model-specific classes that were ~80% identical:
- `SOLARKoExtractor`, `QwenExtractor`, `LlamaExtractor`, `Qwen72BExtractor`, `QwenVLExtractor`, `Qwen3Extractor`, `Gemma3Extractor`, `MixtralExtractor`

Each repeated the same `_load_model()` and `extract_metadata()` with only the model name and cache key different. Total: 1,215 lines.

**Why it mattered:** Adding a new model required copy-pasting ~100 lines and changing 3 values. Fixing a bug in extraction logic required changing it in 9 places.

**Fix:** Created a single `LocalModelExtractor` class with a model registry:
```python
class LocalModelExtractor(BaseLLMExtractor):
    def __init__(self, model_config, model_display_name=None, cache_key=None):
        # One class handles all models — differences are parameters
        ...

LOCAL_MODEL_REGISTRY = {
    "qwen":    ("secondary", "secondary", "Qwen2.5-7B"),
    "qwen3":   ("qwen3",    "qwen3",     "Qwen3-4B"),
}
```

Also removed unused models (SOLAR, Llama, Mixtral, Gemma3, etc.) that were never used in production — only cloud models (Alibaba) are used. Changed default from `solar-ko` to `alibaba-qwen3-next-80b-a3b-instruct` everywhere.

Result: **1,215 → 490 lines** (60% reduction), **10 classes → 4 classes**.

**Files:** `base_extractor.py`, `llm_extractor.py`, `app.py`, `index.html`

---

### Fix #10: Extract Shared Prompt Template

**Problem:** The extraction prompt (instructions telling the LLM how to extract metadata) was copy-pasted in 5 different places:
- `base_extractor.py` — 1 copy (bilingual Korean+English, 60 lines)
- `cloud_extractor.py` — 2 Korean copies + 2 English copies

Changing one instruction required editing 5 locations. The bilingual prompt also doubled token cost for no benefit.

**Why it mattered:** Instructions could easily drift between providers. The bilingual Korean+English prompt was wasteful — English instructions are equally effective for Qwen models and use fewer tokens.

**Fix:** Created a single `create_extraction_prompt()` function in `cloud_extractor.py`:
```python
def create_extraction_prompt(text, schema, document_type) -> str:
    """Single source of truth for extraction instructions."""
    return f"""You are a metadata extraction assistant for Korean documents...
    Rules:
    1. Extract information exactly as it appears...
    ..."""
```

All 5 `_create_prompt()` methods now delegate to it:
```python
def _create_prompt(self, text, schema, document_type):
    return create_extraction_prompt(text, schema, document_type)
```

Also switched to English-only instructions (~30% fewer tokens, equally effective).

**Files:** `cloud_extractor.py`, `base_extractor.py`

---

### Fix #11: Merge Duplicate Pipeline Functions

**Problem:** `app.py` had two near-duplicate pipeline functions:
- `process_document` (line 194) — older version
- `process_document_with_universal_ocr` (line 340) — newer version
They shared ~70% code overlap.

**Fix:** Resolved by the PipelineOrchestrator (#14). Both functions replaced by a single `PipelineOrchestrator.run()` method.

**File:** `app.py`, `pipeline.py` (via fix #14)

---

### Fix #12: Unify Document Type Detection

**Problem:** Document type detection existed in 3 different places with different logic:
- `llm_extractor.py` — mapped types to schemas
- `document_extractors.py` — detected type from filename
- `document_schemas.py` — mapped types to per-type schemas

Each had its own if/elif chains and could disagree.

**Fix:** Created a unified schema that's used for ALL document types. `get_schema_by_document_type()` now always returns the same 67-field schema regardless of input. Document type detection still exists for logging/routing but no longer affects which fields are extracted.

**File:** `document_schemas.py`

---

## Phase 4: Structural Health

### Fix #13: `pyproject.toml` and Eliminate `sys.path` Hacks

**Problem:** 8 files used `sys.path.insert(0, ...)` or `sys.path.append(...)` to make imports work:
```python
# Every module guessed where it was relative to project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from module.llm_extraction import ...
```

**Why it mattered:** This breaks when running from a different directory, deploying to a server, or packaging the code. It's also confusing — some modules add the same path that other modules already added.

**Fix:** Two-part solution:

1. **`.pth` file** — a file in Python's `site-packages/` that adds `api/` to the Python path permanently:
```
# /path/to/site-packages/copyright-metadata.pth
/path/to/project/api
```
After this, `from module.ocr import ...` works from any directory.

2. **`pyproject.toml`** — standard Python project file for `pip install -e .` (editable install). Works on normal filesystems; WSL2+OneDrive is too slow for the build step, so the `.pth` file is the primary mechanism.

3. **Removed all 8 `sys.path` hacks** from: `app.py`, `consolidation_agent.py`, `extract_metadata.py`, `checkbox_extractor.py`, `cleanup_cache.py`, `test_consolidator.py`, `test_components_only.py`, `test_phase3.py`.

For server deployment, one command sets it up:
```bash
echo "$(pwd)/api" > $(python3 -c "import site; print(site.getsitepackages()[0])")/copyright-metadata.pth
```

**Files:** `pyproject.toml`, `.pth` file, all 8 files with sys.path hacks

---

### Fix #14: Extract PipelineOrchestrator from `app.py`

**Problem:** `app.py` was a 1,420-line "god file". The `/api/llm-extract` handler alone was 385 lines doing everything: file validation, saving, OCR, LLM, NER, consolidation, response building, file saving, SSE streaming, and error handling — all in one function.

**Why it mattered:** Impossible to test pipeline logic without starting the web server. Impossible to reuse pipeline stages. Every change risked breaking unrelated functionality.

**Fix:** Applied the **Service Layer Pattern** — separated HTTP handling from business logic:

1. Created `api/web/pipeline.py` with `PipelineOrchestrator` class (~280 lines)
2. Each pipeline stage is a public method:
   - `setup(file_bytes, filename)` → saves file, creates directories, returns context dict
   - `run_ocr(ctx, provider, model)` → returns (ocr_text, ocr_result)
   - `run_llm(ocr_text, doc_type, filename, model)` → returns llm_result
   - `run_ner(ocr_result, result_dir, ner_model, ocr_text)` → returns ner_result
   - `run_consolidation(...)` → returns (result, success, error)
   - `build_response(ctx, ...)` → returns complete API response dict
   - `save_results(...)` → writes JSON to disk
   - `run(file_bytes, filename, **kwargs)` → runs full pipeline in one call

3. The `/api/llm-extract` handler in `app.py` is now ~70 lines:
   - **Non-streaming:** calls `pipeline_orchestrator.run()` → returns JSON
   - **SSE streaming:** async generator calls each stage with `yield` between them for real-time progress updates

**How streaming works:**
```python
async def process_with_progress():
    ctx = pipeline_orchestrator.setup(file_content, filename)
    yield progress("파일 업로드 완료", 10%)       # sent to client immediately

    ocr_text, _ = pipeline_orchestrator.run_ocr(ctx, ...)
    yield progress("OCR 완료", 40%)               # sent after OCR finishes

    llm_result = pipeline_orchestrator.run_llm(...)
    yield progress("LLM 완료", 70%)               # sent after LLM finishes

    # ... etc
```

The `yield` between stages is what makes it real-time — each SSE message is sent to the client as soon as a stage completes.

**Files:** `api/web/pipeline.py` (new), `api/web/app.py` (refactored)

See `docs/PIPELINE_ORCHESTRATOR_GUIDE.md` for detailed architecture documentation.

---

### Fix #16: Consolidate `requirements.txt` Files

**Problem:** 4 requirements files with overlapping and conflicting dependencies:
- `api/requirements.txt` — 25 packages (included stale `flask`)
- `api/module/llm_extraction/requirements.txt` — 10 packages
- `OCR/google_vision/requirements.txt` — 7 packages
- `requirements_hf_downloader.txt` — 4 packages

`protobuf>=4.21.0` in one file, `protobuf>=3.20.0` in another. `torch` and `transformers` listed in 3 files.

**Why it mattered:** Which file do you install? If you run `pip install -r api/requirements.txt`, you miss LLM deps like `bitsandbytes` and `pydantic`. Version conflicts could cause subtle bugs.

**Fix:** Created single `requirements.txt` at project root with 30 deduplicated packages organized by category. Removed stale `flask` dependency. Install with one command: `pip install -r requirements.txt`.

**File:** `requirements.txt` (project root)

---

## Phase 5: Code Quality

### Fix #18: Standardize OCR Metadata Structure

**Problem:** Each OCR provider returned different metadata fields:

| Provider | Extra fields | Missing fields |
|----------|-------------|----------------|
| Google | `text_annotations` (huge nested object) | `model` |
| Naver | `raw_response` (entire API dump) | `model` |
| Mistral | — | `model` in error case |
| Alibaba | `region`, `processing_mode` | `model` in error case |

**Why it mattered:** Code consuming OCR results couldn't rely on consistent fields. Google's `text_annotations` dumped megabytes of annotation data into the result JSON unnecessarily.

**Fix:** Standardized all 4 providers to return identical structure:
```python
# Success
{"extracted_text": "...", "metadata": {"provider": "...", "model": "...", "confidence": 0.8, "processing_time": None}}

# Error
{"extracted_text": "",    "metadata": {"provider": "...", "model": "...", "error": "...", "confidence": 0.0}}
```

Removed provider-specific dumps (`text_annotations`, `raw_response`, `region`, `processing_mode`).

**Files:** `google_ocr.py`, `naver_ocr.py`, `mistral_ocr.py`, `alibaba_ocr.py`

---

### Fix #19: Korean Date Format Support

**Problem:** The `ValidationEngine` only accepted `YYYY-MM-DD` dates. But Korean OCR text produces dates like:
- `2024년 1월 15일`
- `2024. 1. 15.`
- `2024/01/15`

These were all rejected as invalid, causing the consolidation engine to distrust date fields.

**Fix:** Updated `_validate_date()` to accept any format that `_normalize_date()` can parse. Added Korean date formats to `_normalize_date()`:
```python
# Now handles:
"2024-01-15"      # standard
"2024/01/15"      # slash
"2024.01.15"      # dot
"2024년 1월 15일"   # Korean
"2024년 01월 15일"  # Korean padded
"2024. 1. 15."    # Korean dot-spaced
"20240115"        # compact
```

All formats normalize to `YYYY-MM-DD`.

**File:** `validation_engine.py`

---

### Fix #21: Fix Reasoning Generator OCR Excerpt Search

**Problem:** Three issues in `reasoning_generator.py`:

1. **Searched for field names instead of values.** The OCR excerpt search looked for the English field name (e.g., `"rights_holder"`) in Korean OCR text — this never matches. Korean text doesn't contain English variable names.

2. **Same reasoning for all AGREED cases.** Always said "높은 신뢰도입니다" (high confidence) regardless of whether confidence was 0.5 or 1.0.

3. **Context window too small.** 50 characters is barely a sentence fragment in Korean.

**Fix:**

1. Now searches for the **actual extracted value** (e.g., `"나라지식정보"`) instead of the field name. Falls back through final_value → llm_value → ner_value.

2. Confidence-aware reasoning messages:
   - 0.9+ → "매우 높은 신뢰도" (very high)
   - 0.7-0.9 → "신뢰도 양호" (good)
   - < 0.7 → "신뢰도가 낮습니다 (50%)" (low, with percentage)
   - Added MISSING case reasoning

3. Context window increased from 50 → 100 characters.

**File:** `reasoning_generator.py`

---

### Fix #22: Remove Dead Code

**Problem:** Several pieces of dead code:
- `HybridModelExtractor` class in `hybrid_extractor.py` — never imported or instantiated
- Base schemas (non-enhanced versions) — never returned by `get_schema_by_document_type()`

**Fix:**
- Deleted `hybrid_extractor.py` entirely (never imported anywhere)
- Base schemas replaced by unified schema (fix #12) — old per-type schemas remain as legacy methods but are no longer called

**File:** Deleted `hybrid_extractor.py`

---

## Summary

| Phase | Issues | Fixed | Remaining |
|-------|--------|-------|-----------|
| 1: Eliminate Confusion | 4 | 4 | 0 |
| 2: Fix Reliability | 4 | 4 | 0 |
| 3: Reduce Duplication | 4 | 4 | 0 |
| 4: Structural Health | 5 | 4 | #15 (pytest) |
| 5: Code Quality | 5 | 4 | #20 (confidence threshold) |
| **Total** | **22** | **19** | **3** |

### Remaining Items
- **#15** — Add pytest configuration (medium effort, not urgent for deployment)
- **#17** — Move remaining docs to `docs/` directory (cosmetic)
- **#20** — Raise confidence threshold from 0.3 to 0.5 in field mapper (5 min quick win)

### Key Metrics
- `app.py`: 1,420 → 1,165 lines
- `base_extractor.py`: 1,215 → 490 lines (60% reduction)
- `.env` files: 6 → 1
- `requirements.txt` files: 4 → 1
- `sys.path` hacks: 8 → 0
- Model extractor classes: 10 → 4
- Prompt copies: 5 → 1
