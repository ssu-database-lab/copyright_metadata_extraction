# LLM Metadata Extraction for Korean Documents

## Overview
This directory contains scripts for extracting metadata from Korean OCR results using open-source Large Language Models (LLMs) with intelligent local caching for improved performance.

## 🚀 Key Features

- **Smart Model Caching**: Downloads models locally to avoid repeated downloads
- **On-Demand Loading**: Models are downloaded only when needed
- **Cache Management**: Built-in cleanup tools for unused models
- **Multiple Model Support**: SOLAR-Ko, Qwen2.5, Qwen2.5-72B, Qwen3-4B, Llama 3.1-70B, Gemma 3 12B, Mixtral 8x7B, and other Korean-optimized models
- **JSON Schema Extraction**: Structured metadata extraction using JSON schemas
- **Progress Tracking**: Visual progress bars and detailed logging

## Recommended Open-Source LLMs for Korean Text Processing

### 1. **Korean-Specialized Models** (Recommended)

#### **A. SOLAR-Ko (Solar-10.7B-Instruct-v1.0)**
- **Best for**: Korean text understanding and metadata extraction
- **Size**: 10.7B parameters
- **Strengths**: 
  - Excellent Korean language understanding
  - Good at structured data extraction
  - Available on Hugging Face
- **Implementation**: `transformers` library
- **Model ID**: `upstage/SOLAR-10.7B-Instruct-v1.0`

#### **B. Qwen2.5-Ko (Qwen2.5-7B-Instruct)**
- **Best for**: Multilingual with strong Korean support
- **Size**: 7B parameters
- **Strengths**:
  - Strong Korean capabilities
  - Good instruction following
  - Efficient inference
- **Implementation**: `transformers` library
- **Model ID**: `Qwen/Qwen2.5-7B-Instruct`

#### **C. Llama-3.1-Korean (Llama-3.1-8B-Instruct)**
- **Best for**: General-purpose Korean text processing
- **Size**: 8B parameters
- **Strengths**:
  - Good Korean language support
  - Strong reasoning capabilities
  - Well-documented
- **Implementation**: `transformers` library
- **Model ID**: `meta-llama/Llama-3.1-8B-Instruct`

### 2. **Multilingual Models** (Alternative)

#### **A. Gemma-2-9B-Instruct**
- **Best for**: Multilingual text processing
- **Size**: 9B parameters
- **Strengths**:
  - Good multilingual support
  - Efficient inference
  - Google's open model
- **Implementation**: `transformers` library
- **Model ID**: `google/gemma-2-9b-instruct`

#### **B. Mistral-7B-Instruct-v0.3**
- **Best for**: General text processing
- **Size**: 7B parameters
- **Strengths**:
  - Strong instruction following
  - Good multilingual capabilities
  - Efficient
- **Implementation**: `transformers` library
- **Model ID**: `mistralai/Mistral-7B-Instruct-v0.3`

### 3. **Smaller Models** (For Resource-Constrained Environments)

#### **A. SOLAR-Ko-1.7B**
- **Best for**: Lightweight Korean processing
- **Size**: 1.7B parameters
- **Strengths**:
  - Fast inference
  - Good Korean support
  - Lower resource requirements
- **Implementation**: `transformers` library
- **Model ID**: `upstage/SOLAR-1.7B-Instruct-v1.0`

## Implementation Recommendations

### **Primary Recommendation: SOLAR-Ko-10.7B**
For Korean document metadata extraction, I recommend starting with **SOLAR-Ko-10.7B** because:

1. **Korean-Optimized**: Specifically trained for Korean language
2. **Structured Output**: Good at extracting structured metadata
3. **Contract Understanding**: Excellent at understanding legal documents
4. **Available**: Easy to access via Hugging Face

### **Secondary Recommendation: Qwen2.5-7B**
As a backup option:
1. **Multilingual**: Good Korean + English support
2. **Efficient**: Smaller model with good performance
3. **Reliable**: Well-tested and documented

## Hardware Requirements

### **Minimum Requirements**:
- **RAM**: 16GB+ (for 7B models)
- **GPU**: NVIDIA RTX 3080/4080 or better
- **Storage**: 20GB+ free space

### **Recommended Requirements**:
- **RAM**: 32GB+ (for 10B+ models)
- **GPU**: NVIDIA RTX 4090 or A100
- **Storage**: 50GB+ free space

## Next Steps

1. **Choose Model**: Start with SOLAR-Ko-10.7B
2. **Setup Environment**: Install required libraries
3. **Create Extraction Scripts**: Implement metadata extraction
4. **Test on Sample Data**: Validate with OCR results
5. **Optimize Performance**: Fine-tune for your specific use case

## System Architecture

```
Project/llm_extraction/
├── 📄 README.md                    # System overview and LLM recommendations
├── 📄 USAGE.md                     # Detailed usage guide and examples
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Setup script for installation
├── 📄 extract_metadata.py          # Main execution script
├── 📄 cleanup_cache.py             # Model cache management tool
│
├── 📁 config/
│   └── 📄 model_config.yaml        # Model configurations with cache settings
│
├── 📁 models/
│   ├── 📄 base_extractor.py        # Core LLM extractor classes
│   ├── 📄 model_cache.py           # Model cache management system
│   └── 📁 hf_models/               # Local model cache directory
│       ├── SOLAR-10.7B-Instruct-v1.0/
│       ├── Qwen2.5-7B-Instruct/
│       ├── SOLAR-1.7B-Instruct-v1.0/
│       └── cache_metadata.json     # Cache tracking metadata
│
├── 📁 extractors/
│   └── 📄 document_extractors.py   # Document-specific extractors
│       ├── ContractExtractor
│       ├── ConsentExtractor
│       └── DocumentMetadataExtractor
│
├── 📁 schemas/
│   └── 📄 document_schemas.py      # JSON schemas for different document types
│       ├── get_contract_schema()
│       ├── get_consent_schema()
│       ├── get_general_document_schema()
│       └── get_schema_by_document_type()
│
└── 📁 logs/                        # Log files (created at runtime)
```

## 🗂️ Model Caching System

### **How It Works**
1. **First Run**: Models are downloaded from Hugging Face Hub to `models/hf_models/`
2. **Subsequent Runs**: Models are loaded from local cache (much faster!)
3. **Automatic Management**: Cache tracks usage, size, and access patterns
4. **Cleanup Tools**: Remove unused models to free up disk space

### **Cache Benefits**
- **⚡ Faster Loading**: No repeated downloads (saves 10-20GB per model)
- **🌐 Offline Capability**: Works without internet after initial download
- **💾 Storage Control**: Know exactly where models are stored
- **🔄 Version Management**: Consistent model versions across runs

### **Storage Requirements**
- **SOLAR-Ko-10.7B**: ~20GB
- **Qwen2.5-7B**: ~15GB  
- **SOLAR-Ko-1.7B**: ~3GB
- **Total**: ~38GB for all models

### **Cache Management Commands**
```bash
# List cached models
python cleanup_cache.py --list

# Show cache statistics
python cleanup_cache.py --stats

# Clean up models unused for 30 days
python cleanup_cache.py --cleanup --days 30

# Remove all cached models
python cleanup_cache.py --cleanup-all

# Verify model integrity
python cleanup_cache.py --verify primary
```

## Implementation Status

✅ **Completed Components:**
- Core LLM extractor classes with SOLAR-Ko, Qwen, Qwen2.5-72B, Qwen3-4B, Llama, Gemma 3, and Mixtral support
- JSON schema-based extraction for contracts and consent forms
- Document-specific extractors with preprocessing/post-processing
- Batch processing system for OCR results
- Comprehensive error handling and logging
- CLI interface with multiple model options
- Setup script and dependency management

## System Flow

```
OCR Results → DocumentMetadataExtractor → BaseLLMExtractor → LLM Model → JSON Schema → Structured Metadata
```

1. **Input**: OCR text files from `../OCR/google_vision/ocr_results/`
2. **Processing**: Document-specific extractors apply appropriate schemas
3. **LLM**: SOLAR-Ko/Qwen/Qwen2.5-72B/Qwen3-4B/Llama/Gemma3/Mixtral models extract structured metadata
4. **Output**: JSON files with extracted metadata in `metadata_results/`
