# DeepSeek-OCR Integration Guide

## Overview

DeepSeek-OCR is a powerful vision-language model specifically designed for OCR tasks. It offers excellent performance for document processing and text extraction with multiple resolution modes.

**Model Information:**
- **Model**: [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **Size**: 3B parameters
- **License**: MIT
- **Paper**: [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234)

## Features

### 🎯 **Multiple Processing Modes**
- **Tiny**: 512×512 (64 vision tokens) - Fastest processing
- **Small**: 640×640 (100 vision tokens) - Fast processing  
- **Base**: 1024×1024 (256 vision tokens) - Balanced performance (recommended)
- **Large**: 1280×1280 (400 vision tokens) - High quality
- **Gundam**: Dynamic resolution (n×640×640 + 1×1024×1024) - Best quality

### 🔧 **Key Capabilities**
- **Document OCR**: Convert documents to markdown format
- **Free OCR**: Extract text without layout preservation
- **Figure Parsing**: Analyze charts, diagrams, and figures
- **Multilingual Support**: Excellent Korean text recognition
- **Layout Understanding**: Preserves document structure

## Installation

### 1. **Install Dependencies**

```bash
# Install PyTorch with CUDA support (recommended)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install transformers==4.46.3 tokenizers==0.20.3 einops addict easydict

# Install flash attention for performance
pip install flash-attn==2.7.3 --no-build-isolation
```

### 2. **System Requirements**
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Memory**: At least 8GB VRAM for base mode
- **Storage**: ~6GB for model download
- **Python**: 3.12.9+ (tested)

## Usage

### **Method 1: Easy Interactive Interface**

```bash
cd /home/mbmk92/copyright/copyright_metadata_extraction/OCR/google_vision
python easy_ocr_processor.py
# Select option 1 (Process single file)
# Select option 5 (DeepSeek-OCR)
# Choose processing mode (1-5)
```

### **Method 2: Command Line Interface**

```bash
# Process single file with base mode
python universal_ocr_processor.py /path/to/image.jpg --provider deepseek --model base

# Process with high-quality mode
python universal_ocr_processor.py /path/to/document.pdf --provider deepseek --model large

# Process directory with tiny mode (fastest)
python universal_ocr_processor.py /path/to/documents --provider deepseek --model tiny --recursive
```

### **Method 3: Programmatic Usage**

```python
from universal_ocr_processor import UniversalOCRProcessor

# Initialize with base mode (recommended)
processor = UniversalOCRProcessor(
    provider="deepseek",
    output_dir="deepseek_results",
    model="base"
)

# Process single file
result = processor.process_single_file("document.jpg")

# Process directory
results = processor.process_directory("documents_folder", recursive=True)
```

## Configuration

### **Environment Variables**

Create a `.env` file or set environment variables:

```bash
# DeepSeek-OCR Configuration
DEEPSEEK_MODE=base                    # Processing mode: tiny, small, base, large, gundam
DEEPSEEK_DEVICE=cuda                  # Device: cuda or cpu
DEEPSEEK_PROMPT=<image>\n<|grounding|>Convert the document to markdown.
```

### **Custom Prompts**

DeepSeek-OCR supports various prompt templates:

```python
# Document to markdown (default)
prompt = "<image>\n<|grounding|>Convert the document to markdown."

# Free OCR (no layout preservation)
prompt = "<image>\nFree OCR."

# Figure parsing
prompt = "<image>\nParse the figure."

# General description
prompt = "<image>\nDescribe this image in detail."

# Reference location
prompt = "<image>\nLocate <|ref|>specific_text<|/ref|> in the image."
```

## Performance Comparison

| Mode | Resolution | Vision Tokens | Speed | Quality | Memory |
|------|------------|---------------|-------|---------|--------|
| Tiny | 512×512 | 64 | ⚡⚡⚡ | ⭐⭐ | 4GB |
| Small | 640×640 | 100 | ⚡⚡ | ⭐⭐⭐ | 6GB |
| Base | 1024×1024 | 256 | ⚡ | ⭐⭐⭐⭐ | 8GB |
| Large | 1280×1280 | 400 | 🐌 | ⭐⭐⭐⭐⭐ | 12GB |
| Gundam | Dynamic | Variable | 🐌 | ⭐⭐⭐⭐⭐ | 16GB |

## Output Structure

```
deepseek_results/
└── deepseek_ocr/
    ├── base/                          # Mode-specific directory
    │   ├── converted_images/
    │   ├── document1_ocr_result.json
    │   ├── document1_extracted_text.txt
    │   └── ...
    ├── batch_results_20241023_123456.json
    └── processing_summary.json
```

## Example Results

### **JSON Output Structure**

```json
{
  "file_name": "document.jpg",
  "file_path": "/path/to/document.jpg",
  "file_type": ".jpg",
  "ocr_provider": "deepseek",
  "total_pages": 1,
  "processing_timestamp": "2024-10-23T12:34:56",
  "pages": [
    {
      "page_number": 1,
      "image_path": "/path/to/converted_images/document_page_001.png",
      "extracted_text": "Extracted text content...",
      "text_length": 1500,
      "method": "ocr",
      "metadata": {
        "model": "deepseek-ai/DeepSeek-OCR",
        "mode": "base",
        "device": "cuda",
        "prompt": "<image>\n<|grounding|>Convert the document to markdown.",
        "processing_params": {
          "base_size": 1024,
          "image_size": 1024,
          "crop_mode": false
        }
      }
    }
  ],
  "full_text": "Complete extracted text...",
  "total_text_length": 1500,
  "status": "success"
}
```

## Troubleshooting

### **Common Issues**

1. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Use a smaller mode (tiny/small) or reduce batch size

2. **Model Loading Failed**
   ```
   OSError: Can't load tokenizer for 'deepseek-ai/DeepSeek-OCR'
   ```
   **Solution**: Check internet connection and try again

3. **Flash Attention Error**
   ```
   ImportError: flash_attn not found
   ```
   **Solution**: Install flash attention: `pip install flash-attn==2.7.3 --no-build-isolation`

4. **Slow Processing on CPU**
   ```
   Warning: CUDA not available, using CPU (slower)
   ```
   **Solution**: Install CUDA-enabled PyTorch or use GPU

### **Performance Tips**

1. **For Speed**: Use `tiny` or `small` mode
2. **For Quality**: Use `large` or `gundam` mode  
3. **For Balance**: Use `base` mode (recommended)
4. **For Large Files**: Process individually to avoid memory issues
5. **For Batch Processing**: Use smaller modes for efficiency

## Comparison with Other Providers

| Provider | Speed | Quality | Korean Support | Local Processing | Cost |
|----------|-------|---------|----------------|------------------|------|
| **DeepSeek-OCR** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | Free |
| Google Cloud Vision | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | Paid |
| Mistral OCR | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ❌ | Paid |
| Alibaba Cloud | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | Paid |

## Advanced Usage

### **Custom Model Loading**

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load model with custom settings
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='flash_attention_2', 
    trust_remote_code=True, 
    use_safetensors=True
)

# Configure for your hardware
model = model.eval().cuda().to(torch.bfloat16)
```

### **Batch Processing with Custom Logic**

```python
processor = UniversalOCRProcessor("deepseek", model="base")
files = processor.find_files("/path/to/documents", recursive=True)

# Process files with custom filtering
for file_path in files:
    if file_path.endswith('.pdf'):
        result = processor.process_single_file(file_path)
        # Custom processing logic here
```

## References

- **Hugging Face Model**: [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **GitHub Repository**: [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- **Paper**: [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234)
- **Documentation**: [DeepSeek-OCR GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)

## Support

For issues or questions:
1. Check the logs in the `logs/` directory
2. Verify GPU/CUDA setup
3. Test with simple images first
4. Check available memory
5. Review the [GitHub issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)

---

**DeepSeek-OCR is now fully integrated into your Universal OCR Processor!** 🎉

