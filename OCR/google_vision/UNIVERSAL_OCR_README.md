# Universal OCR Processor

A comprehensive OCR solution that can process any file type including documents and images using multiple OCR providers.

## 🚀 Features

### Supported File Types

**Documents:**
- PDF (.pdf)
- Microsoft Word (.docx, .doc)
- Microsoft PowerPoint (.pptx, .ppt)
- Microsoft Excel (.xlsx, .xls)
- Hancom Office (.hwp) - *planned*

**Images:**
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tif, .tiff)

### OCR Providers

1. **Google Cloud Vision API** - Excellent for Korean text
2. **Mistral OCR API** - Fast and accurate
3. **Naver Clova OCR API** - Specialized for Korean documents

## 📦 Installation

### 1. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Additional dependencies for universal processing
pip install -r universal_requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the `google_vision` directory:

```bash
# Mistral API Key
MISTRAL_API_KEY=your_mistral_api_key_here

# Naver Clova OCR API Keys (optional)
NAVER_API_KEY=your_naver_api_key_here
NAVER_SECRET_KEY=your_naver_secret_key_here
```

### 3. Google Cloud Setup

Ensure your Google Cloud credentials file is in the same directory as the scripts.

## 🎯 Usage

### Method 1: Easy Interactive Interface

```bash
cd /home/mbmk92/copyright/Project/OCR/google_vision
python easy_ocr_processor.py
```

This will show an interactive menu where you can:
- Process single files
- Process entire directories
- Choose OCR providers
- View supported file types

### Method 2: Command Line Interface

```bash
# Process a single file
python universal_ocr_processor.py /path/to/file.pdf --provider mistral

# Process a directory
python universal_ocr_processor.py /path/to/directory --provider google_cloud --recursive

# Process with custom output directory
python universal_ocr_processor.py /path/to/files --provider mistral --output my_results
```

### Method 3: Programmatic Usage

```python
from universal_ocr_processor import UniversalOCRProcessor

# Initialize processor
processor = UniversalOCRProcessor(
    provider="mistral",  # or "google_cloud" or "naver"
    output_dir="my_ocr_results"
)

# Process single file
result = processor.process_single_file("document.pdf")

# Process directory
results = processor.process_directory("my_documents", recursive=True)
```

## 📊 Output Structure

```
universal_ocr_results/
├── converted_images/           # Converted images from documents
├── document1_ocr_result.json  # Structured results for each file
├── document1_extracted_text.txt # Plain text extraction
├── document2_ocr_result.json
├── document2_extracted_text.txt
├── batch_results_20240922_123456.json # Batch processing results
└── processing_summary.json    # Summary of batch processing
```

## 🔧 Configuration

### OCR Provider Selection

**Google Cloud Vision API:**
- ✅ Excellent Korean text recognition
- ✅ High accuracy
- ❌ Requires Google Cloud setup
- ❌ API costs

**Mistral OCR API:**
- ✅ Fast processing
- ✅ Good accuracy
- ✅ Easy setup
- ❌ API costs

**Naver Clova OCR:**
- ✅ Specialized for Korean
- ✅ Good for Korean documents
- ❌ Requires Naver account
- ❌ API costs

### File Processing

**PDF Files:**
- Converted to high-resolution images (300 DPI)
- Each page processed separately
- Results combined into single document

**DOCX Files:**
- Text extracted directly (no OCR needed)
- Preserves formatting structure
- Faster processing than OCR

**Image Files:**
- Processed directly with OCR
- No conversion needed
- Supports all common image formats

## 📝 Example Results

### JSON Output Structure

```json
{
  "file_name": "document.pdf",
  "file_path": "/path/to/document.pdf",
  "file_type": ".pdf",
  "ocr_provider": "mistral",
  "total_pages": 3,
  "processing_timestamp": "2024-09-22T12:34:56",
  "pages": [
    {
      "page_number": 1,
      "image_path": "/path/to/converted_images/document_page_001.png",
      "extracted_text": "Page 1 content...",
      "text_length": 1500,
      "method": "ocr"
    }
  ],
  "full_text": "Combined text from all pages...",
  "total_text_length": 4500,
  "status": "success"
}
```

### Text Output

Plain text files are saved with the same name as the original file plus `_extracted_text.txt`.

## 🚨 Error Handling

The processor handles various error conditions:

- **Unsupported file types**: Skipped with warning
- **Corrupted files**: Error logged, processing continues
- **API failures**: Retry logic and fallback options
- **Memory issues**: Large files processed in chunks

## 📈 Performance Tips

1. **For Korean text**: Use Google Cloud Vision API
2. **For speed**: Use Mistral OCR API
3. **For large batches**: Process in smaller chunks
4. **For mixed content**: Use recursive directory processing

## 🔍 Troubleshooting

### Common Issues

**Import Errors:**
```bash
pip install python-docx openpyxl pandas
```

**API Key Issues:**
- Check `.env` file exists
- Verify API keys are correct
- Ensure environment variables are loaded

**File Processing Errors:**
- Check file permissions
- Verify file is not corrupted
- Ensure sufficient disk space

**Memory Issues:**
- Process files individually
- Use smaller batch sizes
- Increase system memory

## 📚 Advanced Usage

### Custom OCR Providers

You can extend the system by implementing the `OCRProvider` interface:

```python
class CustomOCRProvider(OCRProvider):
    def process_image(self, image_path: str) -> Dict:
        # Your OCR implementation
        pass
    
    def get_provider_name(self) -> str:
        return "custom"
```

### Batch Processing with Custom Logic

```python
processor = UniversalOCRProcessor("mistral")
files = processor.find_files("/path/to/documents", recursive=True)

# Custom filtering
filtered_files = [f for f in files if "important" in f.lower()]

# Process filtered files
for file_path in filtered_files:
    result = processor.process_single_file(file_path)
    # Custom processing logic
```

## 📄 License

This project is part of the copyright processing system and follows the same license terms.

## 🤝 Contributing

To add support for new file types or OCR providers:

1. Implement the `OCRProvider` interface for new providers
2. Add file processing logic in `FileProcessor` class
3. Update supported extensions list
4. Add tests for new functionality

## 📞 Support

For issues or questions:
1. Check the logs in the `logs/` directory
2. Verify API keys and credentials
3. Test with simple files first
4. Check file permissions and disk space
