# Architecture Assessment: Copyright Metadata Extraction System

## Executive Summary

The system implements a four-stage pipeline (OCR → LLM → NER → Consolidation) for extracting structured metadata from Korean copyright documents. The design intent is sound — using an LLM arbiter to merge two independent extraction strategies is a legitimate architecture. However, the implementation carries significant technical debt from organic growth without refactoring passes, which now impedes development velocity and reliability.

**Overall Health Score: 4/10** — Promising architecture undermined by structural debt.

---

## What Works Well

### 1. NER Module — Best-Engineered Component

The NER system (`api/module/ner/ner_system.py`) demonstrates the strongest engineering in the codebase:

- **Dual-pipeline architecture**: BERT-CRF (BiLSTM + CRF Viterbi decoding) with regex fallback guarantees output even when the model is unavailable
- **Pattern-based correction**: Low-confidence tokens from the neural model are cross-checked against regex patterns, combining statistical and rule-based approaches
- **Graceful degradation**: If the model can't be downloaded, the system falls back to regex extraction transparently
- **23 entity types** with type-specific validation and false positive filtering

### 2. Consolidation Concept — Sophisticated Design

The `ConsolidationAgent` uses Qwen3-Next-80B as an "expert judge" to arbitrate between LLM and NER extractions:

- **5-layer JSON parsing fallback**: Direct parse → markdown extraction → brace matching → truncation repair → regex extraction
- **Model fallback chain**: Qwen3-Next-80B → Qwen-Max → basic consolidation
- **Component separation**: FieldMapper, ValidationEngine, ReasoningGenerator are cleanly isolated

### 3. OCR Provider Abstraction — Right Direction

Four providers (Google Vision, Mistral, Naver, Alibaba) behind an ABC with `UniversalOCRProcessor` as coordinator. Clean one-file-per-provider organization.

---

## Critical Problems

### 1. The God File — `api/web/app.py` (1,400+ lines)

This single file contains route definitions, business logic, file I/O, SSE streaming, response serialization, and pipeline orchestration.

**Specific issues:**

- **Two near-duplicate pipeline functions** (`process_document` at line 194 and `process_document_with_universal_ocr` at line 340) share ~70% code overlap
- **The `/api/llm-extract` handler is 385 lines** (lines 807-1191) — it does OCR, LLM, NER, consolidation, progress tracking, file writing, and error handling in one function
- **`locals()` variable existence checks** at line 1133: `request_id if 'request_id' in locals() else None` — a sign that control flow has outgrown the function scope
- **Blocking file I/O in async context** at line 878: synchronous `open()` + `write()` inside an async handler
- **Hardcoded progress percentages** (10, 20, 40, 50, 70, 80, 90, 95, 98, 100) scattered across 9 `asyncio.sleep(0.01)` calls that don't reflect actual work progress
- **Print statements mixed with logging** at lines 974-976: `print(f"NER 모델 이름: {ner_model_name}")` alongside `logger.info()` calls elsewhere
- **Bare except** at line 1176: `except:` with `continue` silently swallows all errors during JSON parsing
- **Repeated file validation** at lines 517, 860, 1268 — same extension check done three different ways
- **Repeated directory creation** at 10+ locations with no helper function

**Recommendation**: Extract a `PipelineOrchestrator` class. Route handlers should be <30 lines. Each pipeline stage should pass results through a well-defined data structure, not scattered local variables.

### 2. Dual Codebase — Legacy vs Active

Three parallel implementations of LLM extraction exist:

| Location | Status | LOC |
|---|---|---|
| `/llm_extraction/` | Legacy, obsolete | 3,657 |
| `/extract/llm_extraction/` | Nested legacy, obsolete | ~500 |
| `/api/module/llm_extraction/` | Active | 16,054 |

Similarly for NER: root `/ner/` (legacy) and `/api/module/ner/` (active). The legacy `base_extractor.py` is byte-for-byte identical (46,706 bytes) to the active copy. They will inevitably diverge when someone fixes a bug in the wrong copy.

**Also legacy**: `api/call.py` (Flask server superseded by FastAPI), `api/api.py` (old wrapper), the outdated `api/__init__.py` public API that doesn't reflect the modern module structure.

**Recommendation**: Delete `/ner/`, `/llm_extraction/`, `/extract/` entirely.

### 3. Massive Duplication in LLM Extractors

`api/module/llm_extraction/models/base_extractor.py` contains **9 model-specific extractor classes** (~1,000 lines) that are ~80% identical:

- `SOLARKoExtractor`, `QwenExtractor`, `LlamaExtractor`, `Qwen72BExtractor`, `QwenVLExtractor`, `Qwen3Extractor`, `Gemma3Extractor`, `MixtralExtractor`
- Each repeats: `__init__`, `load_model()`, `extract_metadata()` with the only real differences being the model name and `max_new_tokens`
- Inconsistent loading patterns: some use `_load_model()` + pipeline, others have separate `load_model()` methods that aren't called by the base class contract

The 11-point extraction prompt is copy-pasted into **5 different cloud extractors** (`base_extractor.py`, `HuggingFaceInferenceExtractor`, `OpenAIExtractor`, `AlibabaCloudExtractor`, `TogetherAIExtractor`). Changing one instruction requires updating 5 locations.

**Recommendation**: Collapse into a single configurable class + a model registry dict. Extract prompts to a shared template.

### 4. No OCR Provider Fallback

Despite having 4 OCR providers, `UniversalOCRProcessor` initializes exactly **one**. If Google Vision is down or over quota, the entire pipeline fails — no attempt to try Mistral or Naver.

Additionally, DOCX/PPTX/XLSX/HWP file conversion all return empty lists silently:
```python
logger.warning("DOCX to image conversion not implemented yet")
return []
```
A user uploading a DOCX gets zero extracted text and no clear error.

Contrast with the NER module, which has excellent fallback (BERT-CRF → regex). OCR should follow the same pattern.

**Recommendation**: Implement a provider chain (primary → secondary → tertiary).

### 5. `sys.path` Manipulation Everywhere

Almost every module does path manipulation:

| File | Pattern |
|---|---|
| `api/web/app.py` | `sys.path.insert(0, str(api_dir))` |
| `consolidation_agent.py` | `sys.path.insert(0, str(module_dir))` then `from module.llm_extraction...` |
| `extract_metadata.py` | `sys.path.append(os.path.dirname(...))` |
| `checkbox_extractor.py` | `sys.path.append(os.path.dirname(...))` |
| Each OCR provider | Hardcoded `.env` path search up the directory tree |

This breaks when: running tests from different directories, packaging the code, or deploying to a different filesystem layout.

**Recommendation**: Create `pyproject.toml`, make the project installable, use relative imports.

### 6. Configuration Chaos

**6 `.env` files** scattered across directories:
- `api/.env`, `api/.env_alibaba`
- `api/web/.env`, `api/web/.env_alibaba`
- `OCR/google_vision/.env`, `OCR/google_vision/.env_alibaba`

**8 `requirements.txt` files** with overlapping, uncoordinated dependencies across `api/`, `llm_extraction/`, `ner/`, `extract/`, `OCR/google_vision/`.

Model config (`model_config.yaml`) exists in the **legacy** directory but the `config/` directory in the **active** module is empty — code can't find models at runtime.

**Recommendation**: Single `.env` at project root, single `requirements.txt`, move model config to active directory.

---

## Module-Level Issues

### Consolidation Agent (`api/module/consolidator/`)

**Alibaba Cloud API calls lack resilience** (`consolidation_agent.py` line 307):
- No timeout specified — if Alibaba hangs, consolidation blocks indefinitely
- No retry logic — transient network errors immediately trigger fallback
- No rate limiting — batch processing could hit API limits
- Generic `except Exception` treats auth errors, network timeouts, and malformed responses identically
- No circuit breaker pattern — if API is down, every document attempt wastes time

**Field Mapper bugs** (`field_mapper.py`):
- `PHONE` priority is defined **twice** (lines 241 and 256) — Python dict silently overwrites the first definition
- Confidence threshold of 0.3 is too permissive — a 3/10 priority score can cause wrong field assignment
- No OCR error tolerance (Levenshtein distance) for fuzzy matching
- No contextual awareness — doesn't consider document type when mapping

**Validation gaps** (`validation_engine.py`):
- Only accepts YYYY-MM-DD dates, but OCR produces "2024년 1월 15일", "2024.1.15" — these get rejected
- Only 2 document types have required field definitions (contracts and consent)
- No validation for Korean business registration numbers, name character validation, or cross-field relationships

**Reasoning quality** (`reasoning_generator.py`):
- OCR excerpt search looks for literal field names like "rights_holder" in Korean text — will never match
- Same reasoning message for all AGREED cases regardless of match quality
- 50-character context window too small for meaningful evidence

### LLM Extraction (`api/module/llm_extraction/`)

**Schema design**:
- Base + Enhanced versions of same schemas (contract, consent, general) where only the enhanced are actually used — base schemas are dead code
- `get_schema_by_document_type()` never returns base versions
- Digital content schema has 30+ fields with only `work_title` required — too permissive
- `copyrightability` allows `["string", "boolean", "null"]` — ambiguous type semantics
- Document type detection exists in **3 different places** with different logic: `llm_extractor.py`, `document_extractors.py`, `document_schemas.py`

**Error handling**:
- `ExtractionResult` has an optional `error` field but no invariant (error set + non-zero confidence is valid)
- `HybridModelExtractor` exists but is never instantiated by `create_extractor()` — dead code
- `_parse_response()` accepts partial JSON extractions silently with no schema validation
- `LLMExtractionProcessor` returns a custom dict instead of `ExtractionResult`, creating type inconsistency

### NER Module (`api/module/ner/`)

Generally well-engineered, with minor issues:
- CRF always returns confidence 1.0, making the threshold check at line 423 (`if confidence < 0.10`) effectively dead code
- `extract_entities_by_bio_tagging()` is 235 lines — should be split into smaller functions
- Confidence threshold magic number `0.10` used in 5+ places without constants

### OCR Module (`api/module/ocr/`)

- ABC is minimal (only `process_image` and `get_provider_name`) — missing abstraction for streaming (uses `hasattr()` runtime check instead)
- Each provider independently loads `.env` from hardcoded paths — should be centralized
- Inconsistent metadata structures across providers (Google returns `fullTextAnnotation`, Mistral returns simple dict, Naver returns `raw_response`)
- No structured error codes — Alibaba uses string pattern matching (`'Arrearage' in error_str`)

---

## Testing Infrastructure

**Status: No formal test framework.**

- No `pytest.ini`, `conftest.py`, `setup.cfg`, or `pyproject.toml`
- 16 ad-hoc test files scattered across directories, run via `python script.py`
- Tests require live API keys — can't run in CI/CD
- No mocking, no coverage reporting, no test discovery
- The consolidator has the best test setup (component tests without API keys), but still standalone scripts

---

## Documentation

**35+ markdown files in the project root** — extensive but disorganized:
- `API_SPECIFICATION.md`, `DIAGRAMS_ARCHITECTURE.md`, `SCHEMA_FILES_README.md`, `NER_LABELS_MAPPING.md`, `HOW_NER_EXTRACTS_METADATA.md`, `REGEX_PATTERNS_DOCUMENTATION.md`, `CONSOLIDATION_*.md`, `RUNPOD_DEPLOYMENT.md`, `TESTING_GUIDE.md`, etc.
- No `docs/` directory structure
- Several appear to be one-off reference documents rather than maintained docs

---

## Prioritized Recommendations

### Phase 1: Eliminate Confusion (Immediate, ~4 hours)

1. Delete legacy directories: `/ner/`, `/llm_extraction/`, `/extract/`
2. Delete legacy API files: `api/call.py` (Flask), update `api/__init__.py`
3. Consolidate to one `.env` at project root with a `.env.example` template
4. Move `model_config.yaml` to the active `api/module/llm_extraction/config/`

### Phase 2: Fix Reliability (High Priority, ~8 hours)

5. Add timeout (30s) and retry (3x with exponential backoff) to Alibaba API calls
6. Fix duplicate `PHONE` priority key in `field_mapper.py`
7. Implement OCR provider fallback chain
8. Replace `locals()` checks and bare `except:` in `app.py`

### Phase 3: Reduce Duplication (High Priority, ~12 hours)

9. Collapse 9 model extractors into 1 configurable class + model registry
10. Extract shared prompt template (currently duplicated 5x)
11. Merge duplicate pipeline functions in `app.py`
12. Unify document type detection (currently in 3 places)

### Phase 4: Structural Health (Medium Priority, ~16 hours)

13. Create `pyproject.toml`, eliminate all `sys.path` hacks
14. Extract `PipelineOrchestrator` from `app.py`
15. Add pytest configuration with conftest fixtures and API mocking
16. Consolidate 8 `requirements.txt` files into one
17. Move documentation to `docs/` directory

### Phase 5: Code Quality (Medium Priority, ~8 hours)

18. Standardize OCR metadata structure across providers
19. Add Korean date format support to `ValidationEngine`
20. Raise confidence threshold from 0.3 to 0.5 in field mapper
21. Fix reasoning generator OCR excerpt search (search for values, not field names)
22. Remove dead code: base schemas, `HybridModelExtractor`, unreachable CRF confidence checks
