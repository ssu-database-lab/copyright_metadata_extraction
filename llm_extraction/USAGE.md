# Korean Document Metadata Extraction - Usage Guide

## 🚀 Quick Start

### 1. Setup
```bash
cd Project/llm_extraction
python setup.py
```

### 2. Test the System (First run will download models)
```bash
python extract_metadata.py --test
```

### 3. Process OCR Results
```bash
python extract_metadata.py --ocr-results-dir ../OCR/google_vision/ocr_results --output-dir metadata_results
```

## 🗂️ Model Cache Management

### **First-Time Setup**
When you run the system for the first time, models will be automatically downloaded:
- **SOLAR-Ko-10.7B**: ~20GB download
- **Qwen2.5-7B**: ~15GB download  
- **SOLAR-Ko-1.7B**: ~3GB download

### **Cache Commands**
```bash
# List all cached models
python cleanup_cache.py --list

# Show cache statistics
python cleanup_cache.py --stats

# Clean up models unused for 30 days
python cleanup_cache.py --cleanup --days 30

# Remove all cached models (frees up ~38GB)
python cleanup_cache.py --cleanup-all --yes

# Verify model integrity
python cleanup_cache.py --verify primary
```

### **Cache Benefits**
- **⚡ Faster Loading**: Subsequent runs load from local cache
- **🌐 Offline Mode**: Works without internet after initial download
- **💾 Storage Control**: Models stored in `models/hf_models/`
- **🔄 Consistent Versions**: Same model version across runs

## 📋 Available Models

### **SOLAR-Ko-10.7B** (Default - Recommended)
- **Best for**: Korean documents, legal contracts
- **Accuracy**: High
- **Speed**: Medium
- **Memory**: ~16GB RAM
```bash
python extract_metadata.py --model solar-ko --test
```

### **Qwen2.5-7B** (Alternative)
- **Best for**: Multilingual documents
- **Accuracy**: High
- **Speed**: Fast
- **Memory**: ~12GB RAM
```bash
python extract_metadata.py --model qwen --test
```

### **SOLAR-Ko-1.7B** (Lightweight)
- **Best for**: Resource-constrained environments
- **Accuracy**: Medium
- **Speed**: Very Fast
- **Memory**: ~6GB RAM
```bash
python extract_metadata.py --model lightweight --test
```

## 🎯 Usage Examples

### Test with Sample Data
```bash
python extract_metadata.py --test --model solar-ko
```

### Process All OCR Results
```bash
python extract_metadata.py \
  --ocr-results-dir ../OCR/google_vision/ocr_results \
  --output-dir metadata_results \
  --model solar-ko
```

### Process Specific Provider Results
```bash
python extract_metadata.py \
  --ocr-results-dir ../OCR/google_vision/ocr_results/mistral_ocr \
  --output-dir mistral_metadata \
  --model qwen
```

## 📊 Output Structure

```
metadata_results/
├── extraction_summary.json          # Overall statistics and results
├── google_cloud_ocr_계약서_[doc]_metadata.json
├── google_cloud_ocr_동의서_[doc]_metadata.json
├── mistral_ocr_계약서_[doc]_metadata.json
└── mistral_ocr_동의서_[doc]_metadata.json
```

## 🏗️ System Architecture

The system follows a modular architecture:

```
📁 llm_extraction/
├── 📄 extract_metadata.py          # Main CLI interface
├── 📄 setup.py                     # Automated setup
├── 📁 models/
│   └── 📄 base_extractor.py        # LLM model implementations
├── 📁 extractors/
│   └── 📄 document_extractors.py   # Document-specific logic
├── 📁 schemas/
│   └── 📄 document_schemas.py      # JSON schema definitions
└── 📁 config/
    └── 📄 model_config.yaml        # Model configurations
```

### **Processing Flow:**
1. **Input**: OCR text files from `../OCR/google_vision/ocr_results/`
2. **Schema Selection**: Choose appropriate JSON schema based on document type
3. **LLM Processing**: SOLAR-Ko/Qwen/Qwen2.5-72B/Qwen3-4B/Llama/Gemma3/Mixtral extracts structured metadata
4. **Post-processing**: Clean and validate extracted data
5. **Output**: Save structured JSON metadata files

## ✅ Implementation Status

**Fully Implemented Components:**
- ✅ **Core LLM Extractors**: SOLAR-Ko, Qwen, Qwen2.5-72B, Qwen3-4B, Llama, Gemma 3, Mixtral, and lightweight models
- ✅ **JSON Schema System**: Contract and consent form schemas
- ✅ **Document Extractors**: Specialized processing for different document types
- ✅ **Batch Processing**: Automated processing of OCR results
- ✅ **CLI Interface**: Command-line interface with multiple options
- ✅ **Error Handling**: Comprehensive error handling and retry mechanisms
- ✅ **Logging**: Detailed logging for debugging and monitoring
- ✅ **Setup Script**: Automated dependency installation and setup

**Ready for Production Use!** 🚀

## 🔍 Extracted Metadata Fields

### **Contracts (계약서)**
- `contract_type`: 계약서 유형
- `rights_holder`: 권리자
- `user`: 이용자
- `work_title`: 저작물 제목
- `work_category`: 저작물 종별
- `granted_rights`: 허락된 권리
- `contract_purpose`: 계약의 목적
- `payment_amount`: 지급 금액
- `signature_date`: 계약 체결일
- `special_terms`: 특별 약정 사항

### **Consent Forms (동의서)**
- `consent_type`: 동의서 유형
- `data_controller`: 개인정보 처리자
- `data_subject`: 정보주체
- `collection_purpose`: 수집 목적
- `collected_data_types`: 수집 항목
- `retention_period`: 보유 기간
- `consent_status`: 동의 여부
- `consent_date`: 동의일
- `contact_info`: 연락처 정보

## ⚙️ Configuration

### Model Configuration (`config/model_config.yaml`)
```yaml
models:
  primary:
    name: "SOLAR-Ko-10.7B"
    model_id: "upstage/SOLAR-10.7B-Instruct-v1.0"
    max_length: 4096
    temperature: 0.1
    top_p: 0.9
```

### Custom Schema
You can modify schemas in `schemas/document_schemas.py` to add new fields or change extraction logic.

## 🛠️ Troubleshooting

### **Out of Memory Error**
- Use `--model lightweight` for smaller model
- Reduce `max_length` in config
- Close other applications

### **Slow Processing**
- Use GPU if available
- Try `--model qwen` for faster processing
- Process smaller batches

### **Poor Extraction Quality**
- Use `--model solar-ko` for better Korean understanding
- Check OCR text quality
- Adjust `temperature` in config (lower = more focused)

## 📈 Performance Tips

1. **GPU Usage**: Ensure CUDA is available for faster processing
2. **Batch Processing**: Process multiple documents together
3. **Model Selection**: Choose appropriate model for your needs
4. **Text Preprocessing**: Clean OCR text before extraction

## 🔧 Advanced Usage

### Custom Extraction
```python
from models.base_extractor import create_extractor
from extractors.document_extractors import DocumentMetadataExtractor

# Create extractor
extractor = create_extractor("solar-ko")
doc_extractor = DocumentMetadataExtractor(extractor)

# Extract metadata
result = doc_extractor.extract_metadata(text, "계약서", "document_name")
print(result.metadata)
```

### Batch Processing with Custom Logic
```python
# Process specific document types only
results = []
for doc_type in ["계약서"]:  # Only contracts
    result = doc_extractor.extract_metadata(text, doc_type, doc_name)
    results.append(result)
```

## 📞 Support

For issues or questions:
1. Check logs in `logs/` directory
2. Run with `--test` to verify setup
3. Check GPU availability with `python -c "import torch; print(torch.cuda.is_available())"`
