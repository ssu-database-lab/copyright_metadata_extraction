# Cloud Alternatives to Local Model Storage

This document outlines alternatives to storing LLM models locally in the `hf_models` directory.

## 🚀 **Available Alternatives**

### **1. Cloud-based Model APIs**

#### **Hugging Face Inference API**
```bash
# Set up environment
export HUGGINGFACE_API_KEY="your_api_key_here"

# Use cloud extractor
python -c "
from models.cloud_extractor import create_cloud_extractor
extractor = create_cloud_extractor('huggingface', 'your_api_key', 'upstage/SOLAR-10.7B-Instruct-v1.0')
result = extractor.extract_metadata(text, schema, '계약서')
print(result)
"
```

#### **OpenAI API**
```bash
# Set up environment
export OPENAI_API_KEY="your_api_key_here"

# Use cloud extractor
python -c "
from models.cloud_extractor import create_cloud_extractor
extractor = create_cloud_extractor('openai', 'your_api_key', 'gpt-4o-mini')
result = extractor.extract_metadata(text, schema, '계약서')
print(result)
"
```

#### **Together AI**
```bash
# Set up environment
export TOGETHER_API_KEY="your_api_key_here"

# Use cloud extractor
python -c "
from models.cloud_extractor import create_cloud_extractor
extractor = create_cloud_extractor('together', 'your_api_key', 'meta-llama/Llama-2-7b-chat-hf')
result = extractor.extract_metadata(text, schema, '계약서')
print(result)
"
```

### **2. Hybrid Approach (Recommended)**

The hybrid approach tries local models first, then falls back to cloud APIs:

```bash
# Set up environment variables
export HUGGINGFACE_API_KEY="your_api_key_here"
export OPENAI_API_KEY="your_api_key_here"
export EXTRACTION_STRATEGY="hybrid"  # or "local", "cloud"

# Use hybrid extractor
python -c "
from models.hybrid_extractor import create_hybrid_extractor
extractor = create_hybrid_extractor()

# Check available models
models = extractor.get_available_models()
print('Available models:', list(models.keys()))

# Extract metadata
result = extractor.extract_metadata(text, schema, '계약서')
print(f'Used model: {result.model} ({result.provider})')
"
```

### **3. Serverless Deployment**

#### **AWS Lambda Deployment**
```python
# lambda_handler.py
import json
from models.cloud_extractor import create_cloud_extractor

def lambda_handler(event, context):
    # Extract text and schema from event
    text = event['text']
    schema = event['schema']
    document_type = event.get('document_type', '계약서')
    
    # Use cloud extractor
    extractor = create_cloud_extractor('huggingface', os.environ['HUGGINGFACE_API_KEY'], 'upstage/SOLAR-10.7B-Instruct-v1.0')
    result = extractor.extract_metadata(text, schema, document_type)
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

#### **Google Cloud Functions**
```python
# main.py
import functions_framework
from models.cloud_extractor import create_cloud_extractor

@functions_framework.http
def extract_metadata(request):
    request_json = request.get_json()
    text = request_json['text']
    schema = request_json['schema']
    document_type = request_json.get('document_type', '계약서')
    
    extractor = create_cloud_extractor('openai', os.environ['OPENAI_API_KEY'], 'gpt-4o-mini')
    result = extractor.extract_metadata(text, schema, document_type)
    
    return result
```

## 💰 **Cost Comparison**

### **Local Models (Current)**
- **Storage**: ~50GB for all models
- **Compute**: Free (uses your hardware)
- **Latency**: Fast (no network calls)
- **Scalability**: Limited by local resources

### **Cloud APIs**

#### **Hugging Face Inference API**
- **Cost**: ~$0.0001 per 1K tokens
- **Latency**: ~2-5 seconds
- **Scalability**: High
- **Models**: 100+ available

#### **OpenAI API**
- **Cost**: ~$0.00015-0.005 per 1K tokens
- **Latency**: ~1-3 seconds
- **Scalability**: Very high
- **Models**: GPT-4o, GPT-4o-mini

#### **Together AI**
- **Cost**: ~$0.0002 per 1K tokens
- **Latency**: ~2-4 seconds
- **Scalability**: High
- **Models**: Llama, Mistral, Qwen, etc.

## 🔧 **Configuration Options**

### **Environment Variables**
```bash
# Required for cloud APIs
export HUGGINGFACE_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
export TOGETHER_API_KEY="your_key"

# Optional configuration
export EXTRACTION_STRATEGY="hybrid"  # local, cloud, hybrid
export CLOUD_TIMEOUT="30"
export CLOUD_RETRY_ATTEMPTS="3"
```

### **Configuration Files**

#### **cloud_model_config.yaml**
```yaml
cloud:
  enabled: true
  fallback_to_local: true
  timeout: 30

providers:
  huggingface:
    enabled: true
    api_key_env: "HUGGINGFACE_API_KEY"
    
models:
  solar_ko_cloud:
    name: "SOLAR-Ko-10.7B-Cloud"
    provider: "huggingface"
    model_id: "upstage/SOLAR-10.7B-Instruct-v1.0"
    cost_per_token: 0.0001
```

## 📊 **Performance Comparison**

| Approach | Latency | Cost | Scalability | Reliability |
|----------|---------|------|-------------|-------------|
| Local | 0.1-1s | Free | Low | High |
| Hugging Face | 2-5s | Low | High | Medium |
| OpenAI | 1-3s | Medium | Very High | High |
| Together AI | 2-4s | Low | High | Medium |
| Hybrid | 0.1-5s | Variable | High | Very High |

## 🎯 **Recommendations**

### **For Development/Testing**
- Use **local models** for fast iteration
- Use **cloud APIs** for testing different models

### **For Production**
- Use **hybrid approach** for best reliability
- Use **cloud APIs** for high-volume processing
- Use **serverless** for cost optimization

### **For Cost Optimization**
- Use **Hugging Face** for cheapest option
- Use **Together AI** for good balance
- Use **OpenAI** for highest quality

## 🚀 **Migration Guide**

### **Step 1: Set up Cloud APIs**
```bash
# Get API keys from providers
# Hugging Face: https://huggingface.co/settings/tokens
# OpenAI: https://platform.openai.com/api-keys
# Together AI: https://api.together.xyz/settings/api-keys

# Set environment variables
export HUGGINGFACE_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
export TOGETHER_API_KEY="your_key"
```

### **Step 2: Test Cloud Extractors**
```bash
cd /home/mbmk92/copyright/copyright_metadata_extraction/llm_extraction

# Test cloud extractor
python -c "
from models.cloud_extractor import create_cloud_extractor
extractor = create_cloud_extractor('huggingface', 'your_key', 'upstage/SOLAR-10.7B-Instruct-v1.0')
print('Cloud extractor initialized successfully!')
"
```

### **Step 3: Use Hybrid Approach**
```bash
# Test hybrid extractor
python -c "
from models.hybrid_extractor import create_hybrid_extractor
extractor = create_hybrid_extractor()
models = extractor.get_available_models()
print('Available models:', len(models))
"
```

### **Step 4: Update Main Pipeline**
```python
# In extract_metadata.py, replace:
# from models.base_extractor import create_extractor
# with:
from models.hybrid_extractor import create_hybrid_extractor

# Replace:
# extractor = create_extractor(model_name, config_path)
# with:
extractor = create_hybrid_extractor(config_path, "config/cloud_model_config.yaml")
```

## 🔍 **Monitoring and Logging**

### **Cost Monitoring**
```python
# Monitor API costs
extractor = create_hybrid_extractor()
cost = extractor.estimate_cost(text, "cloud_solar_ko_cloud")
print(f"Estimated cost: ${cost:.4f}")
```

### **Performance Monitoring**
```python
# Monitor performance
import time
start_time = time.time()
result = extractor.extract_metadata(text, schema, "계약서")
processing_time = time.time() - start_time
print(f"Processing time: {processing_time:.2f}s")
```

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **API Key Not Found**
```bash
# Check environment variables
echo $HUGGINGFACE_API_KEY
echo $OPENAI_API_KEY
echo $TOGETHER_API_KEY
```

#### **API Rate Limits**
```python
# Add retry logic
import time
import random

def retry_api_call(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
```

#### **Network Timeouts**
```python
# Increase timeout
extractor = create_cloud_extractor('huggingface', api_key, model_id)
# Set timeout in requests.post(..., timeout=60)
```

## 📚 **Additional Resources**

- [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Together AI Documentation](https://docs.together.ai/)
- [AWS Lambda Python Guide](https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html)
- [Google Cloud Functions Python Guide](https://cloud.google.com/functions/docs/writing)
