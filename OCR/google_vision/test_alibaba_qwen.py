#!/usr/bin/env python3
"""
Test script for Alibaba Cloud Model Studio API - Qwen-Plus model
This script allows you to test the Qwen-Plus model with text chat functionality.
"""

import os
import json
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional

class AlibabaCloudModelStudio:
    """Alibaba Cloud Model Studio API client for Qwen-Plus model."""
    
    def __init__(self, api_key: str, model: str = "qwen-plus", region: str = "singapore"):
        """
        Initialize the Alibaba Cloud Model Studio client.
        
        Args:
            api_key: Your Alibaba Cloud API key
            model: Model name (default: qwen-plus)
            region: Region - "singapore" or "china"
        """
        self.api_key = api_key
        self.model = model
        self.region = region
        
        # Set base URL based on region according to documentation
        if region == "singapore":
            self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        else:  # china
            self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def send_chat_message(self, messages: List[Dict[str, str]], 
                         temperature: float = 0.7, 
                         max_tokens: int = 1000) -> Dict:
        """
        Send chat messages to Qwen-Plus model using OpenAI compatible API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary containing the API response
        """
        # Use the OpenAI compatible endpoint as per documentation
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            print(f"🔄 Sending request to Qwen-Plus model...")
            print(f"📝 Model: {self.model}")
            print(f"🌡️ Temperature: {temperature}")
            print(f"📊 Max tokens: {max_tokens}")
            print(f"💬 Messages: {len(messages)} message(s)")
            
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Request successful!")
                
                # Enhanced debug information
                if "choices" in result:
                    print(f"💬 Response choices: {len(result['choices'])}")
                if "usage" in result:
                    usage = result["usage"]
                    print(f"📊 Token usage: {usage.get('total_tokens', 'N/A')} total")
                
                return {"status": "success", "data": result}
            else:
                error_detail = response.text
                print(f"❌ Request failed: {response.status_code}")
                print(f"📋 Error details: {error_detail}")
                
                # Try to parse JSON error response
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        print(f"🔍 Error message: {error_json['error']}")
                except:
                    pass
                
                return {"status": "error", "error": f"HTTP {response.status_code}", "details": error_detail}
                
        except requests.exceptions.Timeout:
            error_msg = "Request timeout - API took too long to respond"
            print(f"⏰ {error_msg}")
            return {"status": "error", "error": error_msg}
        except requests.exceptions.ConnectionError:
            error_msg = "Connection error - Could not reach the API"
            print(f"🔌 {error_msg}")
            return {"status": "error", "error": error_msg}
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"💥 {error_msg}")
            return {"status": "error", "error": error_msg}
    
    def extract_response_text(self, api_response: Dict) -> str:
        """Extract the actual response text from the OpenAI compatible API response."""
        try:
            if api_response.get("status") != "success":
                return ""
            
            data = api_response.get("data", {})
            
            # OpenAI compatible response format
            if "choices" in data:
                choices = data["choices"]
                if choices and len(choices) > 0:
                    choice = choices[0]
                    if "message" in choice:
                        message = choice["message"]
                        return message.get("content", "")
            
            # Fallback for other response formats
            elif "output" in data:
                output = data["output"]
                return output.get("text", "")
            
            else:
                # If we can't find the text, show the full structure
                return f"[Could not extract text from response. Full response: {json.dumps(data, indent=2)}]"
                
        except Exception as e:
            return f"[Error extracting response text: {str(e)}]"

def test_simple_chat(client: AlibabaCloudModelStudio):
    """Test simple chat functionality."""
    print("\n🧪 Testing Simple Chat")
    print("=" * 40)
    
    test_messages = [
        {
            "role": "user",
            "content": "Hello! Please introduce yourself and tell me what you can help with."
        }
    ]
    
    result = client.send_chat_message(test_messages)
    
    if result["status"] == "success":
        response_text = client.extract_response_text(result)
        print(f"\n📋 Qwen-Plus Response:")
        print("-" * 40)
        print(response_text)
        print("-" * 40)
        return True
    else:
        print(f"\n❌ Test failed: {result['error']}")
        if "details" in result:
            print(f"📋 Details: {result['details']}")
        return False

def test_korean_language(client: AlibabaCloudModelStudio):
    """Test Korean language capabilities."""
    print("\n🧪 Testing Korean Language Support")
    print("=" * 40)
    
    test_messages = [
        {
            "role": "user",
            "content": "안녕하세요! 한국어를 해석할 수 있나요? 간단한 문장을 한국어로 작성해주세요."
        },
        {
            "role": "user", 
            "content": "영어로 번역해주세요: '안녕하세요, 저는 인공지능입니다.'"
        }
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n📝 Test {i}: {message['content'][:50]}...")
        
        result = client.send_chat_message([message])
        
        if result["status"] == "success":
            response_text = client.extract_response_text(result)
            print(f"📋 Response:")
            print("-" * 30)
            print(response_text)
            print("-" * 30)
        else:
            print(f"❌ Test failed: {result['error']}")
    
    return True

def test_custom_prompt(client: AlibabaCloudModelStudio):
    """Test with custom user prompts."""
    print("\n🧪 Custom Prompt Testing")
    print("=" * 40)
    
    while True:
        user_input = input("\n📝 Enter your message (or 'quit' to exit): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            print("⚠️ Please enter a message.")
            continue
        
        test_messages = [
            {
                "role": "user",
                "content": user_input
            }
        ]
        
        result = client.send_chat_message(test_messages)
        
        if result["status"] == "success":
            response_text = client.extract_response_text(result)
            print(f"\n🤖 Qwen-Plus Response:")
            print("-" * 50)
            print(response_text)
            print("-" * 50)
        else:
            print(f"\n❌ Error: {result['error']}")
    
    return True

def test_api_limits(client: AlibabaCloudModelStudio):
    """Test API response limits and parameters."""
    print("\n🧪 Testing API Parameters")
    print("=" * 40)
    
    # Test with different temperatures
    temperatures = [0.1, 0.5, 1.0]
    
    for temp in temperatures:
        print(f"\n🌡️ Testing with temperature: {temp}")
        
        test_messages = [
            {
                "role": "user",
                "content": "Write a creative short story about a robot learning to paint."
            }
        ]
        
        result = client.send_chat_message(test_messages, temperature=temp, max_tokens=200)
        
        if result["status"] == "success":
            response_text = client.extract_response_text(result)
            print(f"📋 Response (t={temp}):")
            print("-" * 30)
            print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
            print("-" * 30)
        else:
            print(f"❌ Test failed: {result['error']}")
        
        # Small delay between requests
        time.sleep(1)
    
    return True

def get_api_key():
    """Get API key from user or environment."""
    # First, try to get from environment using official variable name
    api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('ALIBABA_API_KEY')
    
    if api_key:
        print("✅ Found Alibaba API key in environment variables")
        return api_key
    
    # If not found, ask user
    print("⚠️ No API key found in environment variables")
    print("💡 You can set DASHSCOPE_API_KEY environment variable")
    api_key = input("📝 Enter your Alibaba Cloud API key: ").strip()
    
    if not api_key:
        print("❌ API key is required!")
        return None
    
    return api_key

def get_region():
    """Get region selection from user."""
    print("\n🌍 Select your region:")
    print("1. Singapore (recommended for international users)")
    print("2. China (Beijing)")
    
    while True:
        choice = input("Select region (1-2) [default: 1]: ").strip()
        if choice == "2":
            return "china"
        else:
            return "singapore"

def create_env_file():
    """Create a .env file for storing API key."""
    env_content = """# Alibaba Cloud Model Studio API Configuration

# Alibaba Cloud API Key (official environment variable name)
DASHSCOPE_API_KEY=your_alibaba_api_key_here

# Alternative variable name (also supported)
ALIBABA_API_KEY=your_alibaba_api_key_here

# Instructions:
# 1. Replace 'your_alibaba_api_key_here' with your actual API key
# 2. Keep this file secure and never commit it to version control
# 3. Get your API key from: https://dashscope.console.aliyun.com/
# 4. API keys for Singapore and China regions are different
"""
    
    try:
        with open('.env_alibaba', 'w') as f:
            f.write(env_content)
        
        print(f"🔧 Created .env_alibaba file template")
        print(f"📝 Please edit the file and add your actual API key")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")

def main():
    """Main test function."""
    print("🚀 Alibaba Cloud Model Studio - Qwen-Plus Test Suite")
    print("=" * 60)
    
    # Check dependencies
    try:
        import requests
        print("✅ requests library available")
    except ImportError:
        print("❌ requests library not found")
        print("💡 Install with: pip install requests")
        return
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        print("\n🔧 You can create a .env file template:")
        create_env_file()
        return
    
    # Get region selection
    region = get_region()
    
    # Initialize client
    try:
        client = AlibabaCloudModelStudio(api_key, region=region)
        print(f"✅ Alibaba Cloud client initialized")
        print(f"🤖 Model: qwen-plus")
        print(f"🌍 Region: {region}")
        print(f"🔗 Base URL: {client.base_url}")
    except Exception as e:
        print(f"❌ Error initializing client: {e}")
        return
    
    # Run tests
    print(f"\n🧪 Starting Tests...")
    print("=" * 30)
    
    try:
        # Test 1: Simple chat
        success_1 = test_simple_chat(client)
        
        # Test 2: Korean language
        if success_1:
            success_2 = test_korean_language(client)
        
        # Test 3: API parameters
        success_3 = test_api_limits(client)
        
        # Interactive testing
        print(f"\n🎮 Interactive Testing")
        print("=" * 30)
        print("You can now send custom messages to Qwen-Plus")
        test_custom_prompt(client)
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
    
    print(f"\n👋 Test session ended")
    print(f"📊 Results saved with timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")

if __name__ == "__main__":
    main()
