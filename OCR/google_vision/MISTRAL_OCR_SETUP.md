# Mistral OCR Setup Guide

## Prerequisites

1. **Mistral API Key**: Get your API key from [Mistral AI](https://console.mistral.ai/)
2. **Python Dependencies**: Install required packages

## Installation

1. **Install dependencies**:
```bash
pip install mistralai python-dotenv PyMuPDF
```

2. **Set up API key**:
```bash
# Create .env file in OCR/google_vision directory
echo "MISTRAL_API_KEY=your_mistral_api_key_here" > .env
```

## Usage Options

### Option 1: Use Existing Categorized Batch OCR
```python
from categorized_batch_ocr import CategorizedPDFOCR

ocr_processor = CategorizedPDFOCR(
    base_output_dir="ocr_results",
    ocr_provider="mistral"
)

results = ocr_processor.process_pdf_document("your_document.pdf")
```

### Option 2: Use Simple Mistral OCR
```python
from mistral_ocr_simple import SimpleMistralOCR

ocr_processor = SimpleMistralOCR()
results = ocr_processor.process_pdf_document("your_document.pdf")
```

### Option 3: Use Test Script
```bash
python test_mistral_ocr.py
```

## Features

- **PDF to Image Conversion**: High-resolution PNG conversion (2x zoom)
- **Mistral OCR API**: Uses official Mistral SDK
- **Batch Processing**: Process multiple PDFs
- **Comprehensive Output**: JSON and text file results
- **Error Handling**: Robust error management
- **Logging**: Detailed processing logs

## Output Structure

```
mistral_ocr_results/
├── converted_images/
│   ├── document_page_001.png
│   ├── document_page_002.png
│   └── ...
├── document_page_001_mistral_ocr.json
├── document_page_002_mistral_ocr.json
├── document_complete_mistral_ocr.json
└── document_mistral_extracted_text.txt
```

## API Limits

- Be aware of Mistral API rate limits and quotas
- Process documents sequentially to avoid overwhelming the API
- Monitor your API usage in the Mistral console

## Troubleshooting

1. **API Key Issues**: Ensure MISTRAL_API_KEY is set correctly
2. **Network Issues**: Check internet connection
3. **File Permissions**: Ensure write access to output directory
4. **PDF Issues**: Verify PDF files are not corrupted
