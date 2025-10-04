# PDF Extraction from Compressed Archives

This project provides efficient tools to extract only PDF files from large collections of compressed ZIP archives without decompressing everything.

## Why This Approach?

- **Efficient**: Only extracts PDF files, not entire archives
- **Resource-friendly**: Minimal disk space and memory usage
- **Fast**: No need to decompress non-PDF content
- **Organized**: Maintains traceability to source archives

## Files

- `simple_extract.py` - Simple, focused PDF extractor
- `extract_pdfs.py` - Advanced extractor with logging and detailed reporting
- `requirements.txt` - Dependencies (currently none - uses standard library)

## Quick Start

### 1. First, explore what's in your archives (dry run):

```bash
# Navigate to your project directory
cd /path/to/your/compressed/files

# List contents without extracting (dry run)
python3 simple_extract.py . /path/to/output/directory --dry-run
```

### 2. Extract PDFs:

```bash
# Extract all PDFs from ZIP files in current directory
python3 simple_extract.py . /path/to/output/directory
```

## Usage Examples

### Basic Usage
```bash
# Extract PDFs from ZIP files in /data/archives to /output/pdfs
python3 simple_extract.py /data/archives /output/pdfs
```

### Dry Run (Preview)
```bash
# See what PDFs are available without extracting
python3 simple_extract.py /data/archives /output/pdfs --dry-run
```

### Advanced Usage (with logging)
```bash
# Use the advanced extractor with detailed logging
python3 extract_pdfs.py /data/archives /output/pdfs
```

## Output Structure

```
output_directory/
├── extracted_pdfs/          # All extracted PDF files
│   ├── archive1_document1.pdf
│   ├── archive1_document2.pdf
│   └── archive2_document1.pdf
└── extraction_logs/         # Detailed extraction logs
    └── extraction_summary.txt
```

## Features

- **Selective Extraction**: Only PDF files are extracted
- **Archive Traceability**: Output filenames include source archive names
- **Error Handling**: Continues processing even if individual files fail
- **Progress Reporting**: Shows extraction progress and results
- **Dry Run Mode**: Preview contents before extraction

## Performance Tips

1. **Start with dry-run** to see what PDFs are available
2. **Process in batches** if you have thousands of archives
3. **Monitor disk space** - PDFs can still be large
4. **Use SSD storage** for better I/O performance

## Future Metadata Extraction

Once you have the PDFs extracted, you can use these libraries for metadata extraction:

```bash
pip install PyPDF2 pdfplumber pdf2image
```

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure you have read access to ZIP files and write access to output directory
2. **Disk Space**: Check available space before extraction
3. **Corrupted Archives**: Script will skip corrupted files and continue

### Error Messages

- `Error reading archive`: ZIP file is corrupted or inaccessible
- `Error extracting file`: Individual file extraction failed
- `No ZIP files found`: Check source directory path

## License

This project is open source and available under the MIT License.
