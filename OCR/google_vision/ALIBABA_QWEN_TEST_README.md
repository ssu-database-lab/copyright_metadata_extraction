# Alibaba Cloud Model Studio - Qwen-Plus Test Script

This test script allows you to test the Alibaba Cloud Model Studio API with the Qwen-Plus model.

## 🎯 Features

- **Simple Chat Testing**: Test basic conversation functionality
- **Korean Language Support**: Verify Korean text processing capabilities
- **Custom Prompts**: Interactive mode for testing your own questions
- **API Parameter Testing**: Test different temperature and token limits
- **Error Handling**: Comprehensive error reporting and troubleshooting

## 🔧 Setup

### 1. Get Your API Key

1. Visit [Alibaba Cloud DashScope Console](https://dashscope.console.aliyun.com/)
2. Create an account or log in
3. Generate an API key for Model Studio
4. Copy your API key

### 2. Set Up Environment

**Option A: Environment Variable**
```bash
export DASHSCOPE_API_KEY=your_actual_api_key_here
# OR use alternative name:
export ALIBABA_API_KEY=your_actual_api_key_here
```

**Option B: Interactive Input**
Use the script - it will ask for your API key if not found in environment.

**Option C: Environment File**
Copy `alibaba_env_template.txt` to `.env_alibaba` and edit:
```bash
cp alibaba_env_template.txt .env_alibaba
# Edit .env_alibaba and add your API key
```

**Option D: Region Selection**
The script now supports both Singapore and China (Beijing) regions according to [official documentation](https://www.alibabacloud.com/help/en/model-studio/use-qwen-by-calling-api):
- Singapore: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- China: `https://dashscope.aliyuncs.com/compatible-mode/v1`

### 3. Install Required Dependencies

The only required dependency is `requests`:
```bash
pip install requests
```

## 🚀 Usage

### Run the Test Script
```bash
python test_alibaba_qwen.py
```

### What It Tests

1. **Simple Chat**: Basic conversation with the model
2. **Korean Language**: Korean text generation and translation
3. **Custom Prompts**: Interactive testing mode
4. **API Parameters**: Testing with different temperature settings

### Example Output

```
🚀 Alibaba Cloud Model Studio - Qwen-Plus Test Suite
============================================================
✅ requests library available
✅ Found Alibaba API key in environment variables
✅ Alibaba Cloud client initialized
🤖 Model: qwen-plus

🧪 Testing Simple Chat
========================================
🔄 Sending request to Qwen-Plus model...
📝 Model: qwen-plus
🌡️ Temperature: 0.7
📊 Max tokens: 1000
💬 Messages: 1 message(s)
📡 Response status: 200
✅ Request successful!

📋 Qwen-Plus Response:
----------------------------------------
안녕하세요! 저는 Qwen-Plus라고 불리는 인공지능 모델입니다...
----------------------------------------
```

## 📝 Features Demonstrated

### 1. Simple Chat Test
Tests basic conversation capabilities and model responsiveness.

### 2. Korean Language Test
Verifies Korean language support with:
- Korean text generation
- Korean-to-English translation

### 3. Custom Prompt Testing
Interactive mode where you can:
- Ask any questions
- Test creative writing
- Verify factual responses
- Test different conversation styles

### 4. API Parameter Testing
Tests different model behaviors with:
- Low temperature (0.1) - focused responses
- Medium temperature (0.5) - balanced creativity
- High temperature (1.0) - more creative responses

## 🔍 Troubleshooting

### Common Issues

**1. API Key Invalid**
```
❌ Request failed: 401
```
Solution: Verify your API key is correct

**2. Rate Limiting**
```
❌ Request failed: 429
```
Solution: Wait a moment and try again

**3. Network Issues**
```
🔌 Connection error - Could not reach the API
```
Solution: Check your internet connection

**4. Missing Dependencies**
```
❌ requests library not found
```
Solution: `pip install requests`

### Debug Information

The script provides detailed logging:
- Request parameters
- Response status codes
- Error details
- API response structure

## 📊 Expected Response Format

The Qwen-Plus model returns responses in OpenAI compatible format:
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "qwen-plus",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Model response text here"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  }
}
```

## ⚠️ Important Notes

1. **API Costs**: Each request may consume API credits
2. **Rate Limits**: Be mindful of request frequency
3. **Security**: Never share your API key publicly
4. **Data**: Be careful not to send sensitive information in prompts

## 🎉 Success Indicators

When everything works correctly, you should see:
- ✅ Successful API responses
- 🤖 Properly formatted Korean and English text
- 📊 Token usage information
- 🔄 Smooth interactive conversation flow

This confirms your Alibaba Cloud Model Studio integration is working properly!
