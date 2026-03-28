# Hugging Face Model Downloader

This script downloads models from Hugging Face Hub and stores them in a specified directory.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_hf_downloader.txt
```

## Usage

### Python Script

```bash
# Basic usage - downloads to C:\hf_models
python download_hf_model.py microsoft/DialoGPT-medium

# Specify custom output directory
python download_hf_model.py bert-base-uncased --output-dir C:\my_models

# Download private model with token
python download_hf_model.py meta-llama/Llama-2-7b-hf --token YOUR_HF_TOKEN

# Enable verbose logging
python download_hf_model.py microsoft/DialoGPT-medium --verbose
```

### Batch Script (Windows)

```bash
# Basic usage
download_hf_model.bat microsoft/DialoGPT-medium

# Specify custom output directory
download_hf_model.bat bert-base-uncased C:\my_models
```

## Examples

### Download popular models:

```bash
# BERT base model
python download_hf_model.py bert-base-uncased

# GPT-2 model
python download_hf_model.py gpt2

# DistilBERT
python download_hf_model.py distilbert-base-uncased

# RoBERTa
python download_hf_model.py roberta-base

# T5 model
python download_hf_model.py t5-small
```

### Download multilingual models:

```bash
# Multilingual BERT
python download_hf_model.py bert-base-multilingual-cased

# XLM-RoBERTa
python download_hf_model.py xlm-roberta-base
```

## Features

- **Resume Downloads**: Automatically resumes interrupted downloads
- **Progress Logging**: Shows download progress and file information
- **Error Handling**: Graceful error handling with informative messages
- **Token Support**: Supports Hugging Face tokens for private models
- **Flexible Output**: Customizable output directory
- **File Listing**: Shows key downloaded files after completion

## Output Structure

Models are downloaded to subdirectories named after the model ID:

```
C:\hf_models\
├── microsoft_DialoGPT-medium\
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
├── bert-base-uncased\
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── vocab.txt
│   └── ...
└── ...
```

## Requirements

- Python 3.7+
- huggingface_hub>=0.19.0
- torch>=2.0.0
- transformers>=4.30.0
- safetensors>=0.3.0

## Troubleshooting

### Common Issues:

1. **Permission Denied**: Make sure you have write permissions to the output directory
2. **Network Issues**: Check your internet connection and try again
3. **Model Not Found**: Verify the model ID is correct on Hugging Face Hub
4. **Private Models**: Use the `--token` parameter for private models

### Getting Help:

- Check the Hugging Face Hub documentation: https://huggingface.co/docs/hub/index
- Verify model availability: https://huggingface.co/models
