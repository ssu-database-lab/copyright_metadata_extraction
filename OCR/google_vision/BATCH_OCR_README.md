# Categorized Batch PDF OCR Processing

This enhanced solution processes all PDF documents from multiple directories and organizes the OCR results by document category (계약서 and 동의서) with a structured folder hierarchy.

## 🎯 **Features**

- **Categorized Processing**: Automatically categorizes documents as 계약서 or 동의서
- **Structured Output**: Organizes results in a clear folder hierarchy
- **Batch Processing**: Processes all PDFs from multiple source directories
- **High-Quality OCR**: Uses Google Cloud Vision API with document text detection
- **Comprehensive Logging**: Detailed progress tracking and error reporting
- **Summary Reports**: Generates processing summaries and statistics

## 📁 **Output Structure**

The results will be organized as follows:

```
ocr_results/
├── 계약서/
│   ├── [PDF_NAME_1]/
│   │   ├── converted_images/
│   │   │   ├── page_001.png
│   │   │   ├── page_002.png
│   │   │   └── ...
│   │   ├── page_results/
│   │   │   ├── page_001_ocr.json
│   │   │   ├── page_002_ocr.json
│   │   │   └── ...
│   │   ├── [PDF_NAME_1]_complete_ocr.json
│   │   └── [PDF_NAME_1]_extracted_text.txt
│   └── [PDF_NAME_2]/
│       └── ...
├── 동의서/
│   ├── [PDF_NAME_1]/
│   │   ├── converted_images/
│   │   ├── page_results/
│   │   ├── [PDF_NAME_1]_complete_ocr.json
│   │   └── [PDF_NAME_1]_extracted_text.txt
│   └── ...
└── batch_processing_summary.json
```

## 🚀 **Quick Start**

### 1. Install Dependencies
```bash
pip install PyMuPDF google-cloud-vision python-dotenv requests
```

### 2. Setup API Keys (for Mistral OCR)
```bash
# Option A: Use the setup script
python setup_env.py

# Option B: Create .env file manually
cp env_example.txt .env
# Edit .env and add your Mistral API key
```

### 3. Run Batch Processing
```bash
python run_batch_ocr.py
```

### 3. Follow the Prompts
- The script will check dependencies
- Ask for confirmation before processing
- Show progress and results

## 📋 **Source Directories**

The script processes PDFs from these directories:
- `Project/OCR/document/동의서/` - 동의서 documents
- `Project/data/pdf/계약서/` - 계약서 documents  
- `Project/data/pdf/동의서/` - 동의서 documents

## 🔧 **Configuration**

### Customizing Source Directories
Edit `run_batch_ocr.py` and modify the `source_directories` list:

```python
source_directories = [
    "Project/OCR/document/동의서",
    "Project/data/pdf/계약서", 
    "Project/data/pdf/동의서",
    # Add more directories as needed
]
```

### Output Directory
Change the output directory in `run_batch_ocr.py`:

```python
ocr_processor = CategorizedPDFOCR(base_output_dir="your/custom/path")
```

### Environment Variables (.env file)

Create a `.env` file in the `google_vision` directory with your API keys:

```bash
# Mistral API Key for OCR processing
MISTRAL_API_KEY=your_mistral_api_key_here
```

**Security Notes:**
- Never commit the `.env` file to version control
- Keep your API keys secure
- The `.env` file is automatically ignored by git

### Other Environment Variables

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your service account JSON file
- `GRPC_DNS_RESOLVER`: Set to "native" for WSL2 compatibility

## 📊 **Processing Results**

### Individual Document Results
Each processed PDF creates:
- **Converted Images**: High-resolution PNG files of each page
- **Page Results**: JSON files with detailed OCR data for each page
- **Complete OCR**: JSON file with all document metadata and results
- **Extracted Text**: Plain text file with all extracted content

### Batch Summary
The `batch_processing_summary.json` contains:
- Total files processed
- Success/failure counts
- Category breakdown (계약서 vs 동의서)
- Processing timestamps
- Detailed file-by-file results

## 🎯 **Category Detection**

The system automatically categorizes documents based on:
1. **Path Analysis**: Checks if "계약서" or "동의서" is in the file path
2. **Filename Analysis**: Looks for keywords in the filename
3. **Directory Structure**: Uses parent directory names
4. **Default Fallback**: Defaults to 동의서 if uncertain

## ⚠️ **Important Notes**

### Processing Time
- Large PDFs may take several minutes each
- Total processing time depends on number and size of documents
- Consider running during off-peak hours

### API Costs
- Google Cloud Vision API charges per page processed
- Monitor your usage in Google Cloud Console
- Consider processing in smaller batches if needed

### Storage Requirements
- Converted images can be large (high-resolution)
- Ensure sufficient disk space
- Consider cleanup of intermediate files after processing

## 🔍 **Troubleshooting**

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install PyMuPDF google-cloud-vision
   ```

2. **Credentials Error**
   - Ensure credentials file is in the correct location
   - Check file permissions
   - Verify Google Cloud project setup

3. **Memory Issues**
   - Process fewer files at once
   - Reduce image resolution in code
   - Close other applications

4. **API Rate Limits**
   - Add delays between requests
   - Process in smaller batches
   - Check Google Cloud quotas

### Debug Mode
Enable detailed logging by modifying the logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 **Performance Optimization**

### For Large Batches
1. **Process in Chunks**: Modify the script to process subsets
2. **Parallel Processing**: Add threading for multiple documents
3. **Image Optimization**: Reduce resolution for faster processing
4. **Storage Management**: Clean up intermediate files

### Monitoring Progress
- Check the log output for real-time progress
- Monitor the summary file for overall statistics
- Use Google Cloud Console to track API usage

## 🎉 **Success Indicators**

When processing completes successfully, you should see:
- ✅ All dependencies installed
- ✅ Source directories found
- ✅ OCR processor initialized
- ✅ Documents categorized correctly
- ✅ Results organized in proper structure
- ✅ Summary file generated

## 📞 **Support**

If you encounter issues:
1. Check the log output for error messages
2. Verify all dependencies are installed
3. Ensure Google Cloud credentials are correct
4. Check available disk space
5. Monitor Google Cloud API quotas

---

**Happy OCR Processing! 🚀**
