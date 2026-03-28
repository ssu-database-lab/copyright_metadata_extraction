# Copyright Metadata Extraction CLI Tool

Korean document metadata extraction system. Processes PDF/image files through:
**OCR → LLM Extraction → NER → Consolidation**

---

## Quick Start

### 1. Install Python 3.9+

Download from https://www.python.org/downloads/ (check "Add to PATH" during install)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
# Single file (full pipeline)
python extract.py document.pdf

# Specify document type
python extract.py contract.pdf -t 계약서

# Batch mode (whole folder)
python extract.py ./documents/ -o ./results/
```

Or on Windows, double-click **`run_extract.bat`** and follow the prompts.

---

## Usage

### Basic Commands

```bash
# Full pipeline (OCR + LLM + NER + Consolidation)
python extract.py document.pdf

# OCR only
python extract.py document.pdf -s ocr

# OCR + NER only (no cloud LLM needed)
python extract.py document.pdf -s ocr+ner

# OCR + LLM only
python extract.py document.pdf -s ocr+llm

# All extraction, skip consolidation
python extract.py document.pdf -s ocr+llm+ner

# Batch mode
python extract.py ./folder_of_pdfs/ -o ./results/

# List available models
python extract.py --list-models
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output` | `./extraction_results` | Output directory |
| `-t, --document-type` | `기타문서` | Document type: 계약서, 동의서, 저작재산권 양도동의서, 공공저작물 자유이용허락 동의서, 기타문서 |
| `-s, --stages` | `all` | Pipeline stages: `ocr`, `ocr+ner`, `ocr+llm`, `ocr+llm+ner`, `all` |
| `-m, --llm-model` | `alibaba-qwen3.5-122b-a10b` | LLM model for extraction |
| `--ner-model` | `klue-roberta-large` | NER model |
| `--ocr-provider` | `alibaba` | OCR provider: alibaba, google, mistral, naver |
| `--no-consolidate` | - | Skip consolidation step |

### Output Files

```
extraction_results/
└── document_name/
    ├── ocr_text.txt                  — raw OCR extracted text
    ├── llm_metadata.json             — LLM structured metadata
    ├── ner_entities.json             — NER entity extraction
    ├── consolidated_metadata.json    — final merged metadata
    └── full_response.json            — complete pipeline response
```

---

## Models Used

| Stage | Model | Type |
|-------|-------|------|
| OCR | Qwen3-VL-235B | Alibaba Cloud API |
| LLM Extraction | Qwen3.5-122B-A10B | Alibaba Cloud API |
| NER | KLUE-RoBERTa-Large | Local (CPU, no GPU needed) |
| Consolidation | Qwen3.5-122B-A10B | Alibaba Cloud API |
| Consolidation fallback | Qwen3.5-Plus (397B) | Alibaba Cloud API |

---

## Document Types

| Type | Korean | Description |
|------|--------|-------------|
| Contract | 계약서 | Copyright license agreements |
| Consent | 동의서 | Personal information consent forms |
| Transfer | 저작재산권 양도동의서 | Copyright transfer agreements |
| Public License | 공공저작물 자유이용허락 동의서 | Public work license agreements |
| Other | 기타문서 | General documents |

---

## System Requirements

| Requirement | Minimum |
|-------------|---------|
| Python | 3.9+ |
| RAM | 4GB (8GB recommended) |
| GPU | Not required |
| Internet | Required (cloud API calls) |
| Disk | ~2GB (NER models + dependencies) |

---

## Environment Configuration

API keys are pre-configured in `.env`. To modify:

```
# .env
DASHSCOPE_API_KEY=your_key_here     # Required: Alibaba Cloud
MISTRAL_API_KEY=your_key_here       # Optional: Mistral OCR fallback
```

---

## Supported File Types

PDF, JPG, JPEG, PNG, GIF, BMP, TIF, TIFF

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `DASHSCOPE_API_KEY not set` | Check `.env` file exists at project root |
| OCR returns empty text | Try different OCR provider: `--ocr-provider google` |
| Consolidation fails | Try with `--no-consolidate` to get LLM + NER results |
| Slow processing | Use `-s ocr+ner` to skip cloud LLM calls |

---

## Contact

- Soongsil University Database Lab
- Project: Copyright Metadata Extraction System
