# PDF Document OCR with Google Cloud Vision API

This project provides a comprehensive OCR solution for scanned paper documents in PDF format using Google Cloud Vision API. It handles PDF to image conversion and performs high-quality OCR with document text detection.

## Features

- **PDF to Image Conversion**: Converts PDF pages to high-resolution PNG images for better OCR quality
- **Google Cloud Vision API Integration**: Uses the latest Vision API with document text detection
- **Batch Processing**: Process multiple PDF documents in a directory
- **Comprehensive Output**: Saves results in both JSON and text formats
- **Error Handling**: Robust error handling with detailed logging
- **WSL2 Compatibility**: Includes fixes for WSL2 IPv6 issues

## Prerequisites

1. **Google Cloud Project**: You need a Google Cloud project with Vision API enabled
2. **Service Account**: A service account with Vision API permissions
3. **Credentials File**: JSON credentials file for authentication
4. **Python Dependencies**: Required Python packages (see requirements.txt)

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd Project/OCR/google_vision
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google Cloud credentials**:
   - Place your service account JSON file in the project directory
   - Update the path in the code or set the environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
   ```

## Usage

### Basic Usage

```python
from pdf_document_ocr import PDFDocumentOCR

# Initialize OCR processor
ocr_processor = PDFDocumentOCR(output_dir="ocr_results")

# Process a single PDF document
results = ocr_processor.process_pdf_document("path/to/your/document.pdf")

# Access results
print(f"Document: {results['document_name']}")
print(f"Total Pages: {results['total_pages']}")
print(f"Full Text: {results['full_text']}")
```

### Advanced Usage

```python
# Process multiple PDFs in a directory
results = ocr_processor.batch_process_pdfs("path/to/pdf/directory")

# Convert PDF to images only (without OCR)
image_paths = ocr_processor.convert_pdf_to_images("path/to/document.pdf")

# Custom OCR settings
page_result = ocr_processor.ocr_image("image.png", use_document_detection=True)
```

### Command Line Testing

Run the test suite to try different functionalities:

```bash
python test_pdf_ocr.py
```

This provides an interactive menu to test:
1. Single PDF OCR processing
2. Batch PDF processing
3. PDF to image conversion

## Output Structure

The solution generates several output files:

### Individual Page Results
- `{document_name}_page_{page_number}_ocr.json`: OCR results for each page
- `{document_name}_page_{page_number}.png`: Converted image for each page

### Complete Document Results
- `{document_name}_complete_ocr.json`: Complete OCR results for the entire document
- `{document_name}_extracted_text.txt`: Full extracted text from all pages

### Output Directory Structure
```
ocr_results/
├── converted_images/
│   ├── document_page_001.png
│   ├── document_page_002.png
│   └── ...
├── document_page_001_ocr.json
├── document_page_002_ocr.json
├── document_complete_ocr.json
└── document_extracted_text.txt
```

## Configuration

### Environment Variables

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your service account JSON file
- `GRPC_DNS_RESOLVER`: Set to "native" for WSL2 compatibility

### OCR Settings

- **Document Text Detection**: Enabled by default for better scanned document results
- **Image Resolution**: 2x zoom for higher quality OCR
- **Output Format**: PNG format for best OCR accuracy

## Error Handling

The solution includes comprehensive error handling:

- **PDF Conversion Errors**: Handles corrupted PDFs and conversion issues
- **API Errors**: Manages Vision API rate limits and authentication issues
- **File System Errors**: Handles permission and disk space issues
- **Logging**: Detailed logging for debugging and monitoring

## Performance Considerations

- **Image Resolution**: Higher resolution images provide better OCR but increase processing time
- **API Limits**: Be aware of Google Cloud Vision API quotas and rate limits
- **Memory Usage**: Large PDFs with many pages may require significant memory
- **Batch Processing**: Process multiple documents sequentially to avoid overwhelming the API

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Verify your credentials file path
   - Check service account permissions
   - Ensure Vision API is enabled

2. **WSL2 Issues**:
   - The solution includes IPv4 forcing for gRPC
   - Ensure proper network configuration

3. **Memory Issues**:
   - Reduce image resolution if processing large documents
   - Process documents one at a time

4. **API Rate Limits**:
   - Implement delays between requests
   - Use batch processing with appropriate intervals

### Debug Mode

Enable detailed logging by modifying the logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Reference

### PDFDocumentOCR Class

#### Methods

- `__init__(output_dir)`: Initialize the OCR processor
- `convert_pdf_to_images(pdf_path, output_dir)`: Convert PDF to images
- `ocr_image(image_path, use_document_detection)`: Perform OCR on an image
- `process_pdf_document(pdf_path, save_images, save_individual_results)`: Complete workflow
- `batch_process_pdfs(pdf_directory, pattern)`: Process multiple PDFs

#### Parameters

- `pdf_path`: Path to the PDF file
- `output_dir`: Directory for output files
- `use_document_detection`: Whether to use document text detection
- `save_images`: Whether to save converted images
- `save_individual_results`: Whether to save individual page results

## Contributing

To contribute to this project:

1. Follow the existing code style
2. Add appropriate error handling
3. Include logging for new functionality
4. Update documentation for new features
5. Test with various PDF formats and sizes

## License

This project is provided as-is for educational and development purposes. Ensure compliance with Google Cloud Vision API terms of service and usage limits.
