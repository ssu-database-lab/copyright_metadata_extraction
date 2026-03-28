# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Copyright metadata extraction system for Korean documents (contracts, consent forms). Processes documents through a pipeline: OCR → LLM extraction → NER extraction → Consolidation, producing structured metadata with confidence scores and evidence.

## Commands

### Run the API server
```bash
cd api && python -m uvicorn web.app:app --host 0.0.0.0 --port 5000
```
Swagger docs at http://localhost:5000/docs

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run tests
```bash
# Consolidator component tests (no API key needed)
python api/module/consolidator/test_components_only.py

# Full consolidator tests (requires DASHSCOPE_API_KEY)
python api/module/consolidator/test_consolidator.py

# LLM extractor tests
python api/module/llm_extraction/extractors/document_extractors.py
```

### Legacy NER CLI
```bash
python -m ner.run_ner --input "data_dir" --output "result_dir"
```

## Architecture

### Processing Pipeline

The main endpoint `POST /api/llm-extract` drives the full pipeline:

1. **File Upload** → saved to `api/web/uploads/{request_id}/`
2. **OCR** (`api/module/ocr/`) → `UniversalOCRProcessor` coordinates multiple providers (Google Vision, Mistral, Naver, Alibaba) with fallback
3. **LLM Extraction** (`api/module/llm_extraction/`) → `LLMExtractionProcessor` extracts structured metadata using document-type-specific schemas
4. **NER Extraction** (`api/module/ner/`) → BERT-CRF token classification (KLUE-RoBERTa, mBERT, XLM-RoBERTa)
5. **Consolidation** (`api/module/consolidator/`) → `ConsolidationAgent` merges LLM + NER results using Qwen3-Next-80B as arbiter, with fallback to Qwen-Max
6. **Results** → JSON saved to `api/web/results/{request_id}/`

Responses use SSE streaming for real-time progress.

### Key Modules

- **`api/web/app.py`** — FastAPI application (~1400 lines). All API endpoints. Entry point imports from `api/__init__.py` which exposes `pdf_to_image` and `ner_predict`.
- **`api/module/ocr/`** — Multi-provider OCR with `OCRProvider` ABC. Handles PDF/DOCX/HWP/image conversion via `FileProcessor`. Config in `api/ocr_config.json`.
- **`api/module/llm_extraction/`** — 18 model variants (local + Alibaba Cloud). Factory pattern via `create_extractor()`. JSON schemas in `schemas/document_schemas.py`. Model config in `config/model_config.yaml`.
- **`api/module/consolidator/`** — Four components: `FieldMapper` (NER→LLM field mapping), `ValidationEngine` (format/logic validation), `ReasoningGenerator` (Korean evidence generation), `ConsolidationAgent` (orchestrator calling Alibaba Cloud API).
- **`api/module/ner/`** — `ner_system.py` for prediction/training with pytorch-crf.

### Document Types

The system handles five Korean document types, each with specific schemas:
- 저작재산권 이용허락 계약서 (Copyright License Agreement)
- 저작재산권 양도동의서 (Copyright Assignment Agreement)
- 개인정보 수집·이용 동의서 (Personal Information Consent)
- 공공저작물 자유이용허락 동의서 (Public Work License)
- Digital content metadata

### Legacy vs Active Code

- **Active:** `api/` directory (FastAPI server, all modules under `api/module/`)
- **Legacy:** Root-level `ner/`, `llm_extraction/`, `extract/` directories (older implementations, superseded by `api/module/` equivalents)

## Environment Variables

Required in `.env` at project root:
- `DASHSCOPE_API_KEY` — Alibaba Cloud (consolidation + cloud LLM models)
- `GOOGLE_APPLICATION_CREDENTIALS` — Google Vision OCR (`api/google_credentials.json`)
- `MISTRAL_API_KEY` — Mistral OCR
- `NAVER_OCR_API_URL`, `NAVER_OCR_SECRET_KEY` — Naver OCR
- `GRPC_DNS_RESOLVER=native` — Required on WSL2 for Google API IPv6 fix

## Conventions

- Korean comments and logging throughout the codebase
- `sys.path` manipulation in `app.py` to add `api/` parent directory
- Results and uploads use `{request_id}` UUID-based directory structure
- Models are cached locally in `api/module/llm_extraction/models/hf_models/`
- No linter or formatter configured; no CI/CD pipeline
