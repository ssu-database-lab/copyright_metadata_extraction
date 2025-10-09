"""
Cloud-based Model Extractors for LLM Metadata Extraction

This module provides alternatives to local model storage by using cloud APIs
and inference services for metadata extraction.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path

# Load environment variables from .env file
def load_env_file(env_path: str = None):
    """Load environment variables from .env file"""
    if env_path is None:
        # Default to the OCR directory .env file
        env_path = Path(__file__).parent.parent.parent / "OCR" / "google_vision" / ".env"
    
    env_path = Path(env_path)
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value
        return True
    return False

# Load environment variables on import
load_env_file()

logger = logging.getLogger(__name__)


class CloudExtractor(ABC):
    """Abstract base class for cloud-based extractors."""
    
    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
    
    @abstractmethod
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """Extract metadata using cloud API."""
        pass


class HuggingFaceInferenceExtractor(CloudExtractor):
    """Hugging Face Inference API extractor."""
    
    def __init__(self, api_key: str, model_id: str):
        super().__init__(api_key, model_id)
        self.base_url = "https://api-inference.huggingface.co/models"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """Extract metadata using Hugging Face Inference API."""
        prompt = self._create_prompt(text, schema, document_type)
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2048,
                "temperature": 0.1,
                "top_p": 0.8,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/{self.model_id}",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            extracted_text = result[0]["generated_text"]
            
            # Parse JSON response
            try:
                metadata = json.loads(extracted_text)
                return {
                    "metadata": metadata,
                    "model": self.model_id,
                    "provider": "huggingface_inference",
                    "processing_time": None,
                    "confidence": None
                }
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {extracted_text}")
                return {"error": "Invalid JSON response"}
                
        except requests.RequestException as e:
            logger.error(f"Hugging Face API error: {e}")
            return {"error": str(e)}
    
    def _create_prompt(self, text: str, schema: Dict[str, Any], document_type: str) -> str:
        """Create a structured prompt for metadata extraction."""
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        return f"""당신은 한국어 문서(계약서, 동의서, 기타)에서 정보를 추출하는 도우미입니다.
반드시 유효한 JSON만 출력하세요. 설명·주석·마크다운·코드블록 금지.

다음은 {document_type} 문서의 OCR 텍스트입니다. 주어진 JSON 스키마에 따라 메타데이터를 추출해주세요.

문서 텍스트:
{text}

추출할 메타데이터 스키마:
{schema_str}

지시사항:
1. 텍스트에서 각 필드에 해당하는 정보를 정확히 찾아 추출하세요
2. 정보가 명시적으로 존재하지 않거나 불분명한 경우 반드시 null을 사용하세요 (추측 금지)
3. 날짜는 YYYY-MM-DD 형식으로 변환하세요
4. 금액은 숫자만 추출하세요 (단위 제외)
5. 전화번호는 숫자와 하이픈(-)만 포함하세요
6. 주소는 전체 주소를 정확히 추출하세요
7. 사업자등록번호는 숫자와 하이픈(-)만 포함하세요
8. 체크박스 정보 처리:
   - 체크박스가 체크된 상태(📧, ☑, ✓, ■, ●, ◼, ◉)인 경우 true로 설정
   - 체크박스가 체크되지 않은 상태(☐, □, ○, ◯, ◻, ◦)인 경우 false로 설정
   - 체크박스 패턴을 자동으로 감지하여 일관성 있게 처리
   - OCR 오류 고려: "목제권"은 "복제권"으로 해석
9. 반드시 유효한 JSON 형식으로 응답하세요
10. 추가 정보나 설명은 포함하지 마세요
11. ```json이나 ``` 같은 마크다운 문법 사용 금지

응답 (JSON만):"""


class OpenAIExtractor(CloudExtractor):
    """OpenAI API extractor."""
    
    def __init__(self, api_key: str, model_id: str = "gpt-4o-mini"):
        super().__init__(api_key, model_id)
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def extract_metadata(self, text: str, schema: Dict[str,Any], document_type: str) -> Dict[str, Any]:
        """Extract metadata using OpenAI API."""
        prompt = self._create_prompt(text, schema, document_type)
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are a Korean document metadata extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "top_p": 0.8,
            "max_tokens": 2048
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            extracted_text = result["choices"][0]["message"]["content"]
            
            # Parse JSON response
            try:
                metadata = json.loads(extracted_text)
                return {
                    "metadata": metadata,
                    "model": self.model_id,
                    "provider": "openai",
                    "processing_time": None,
                    "confidence": None
                }
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {extracted_text}")
                return {"error": "Invalid JSON response"}
                
        except requests.RequestException as e:
            logger.error(f"OpenAI API error: {e}")
            return {"error": str(e)}
    
    def _create_prompt(self, text: str, schema: Dict[str, Any], document_type: str) -> str:
        """Create a structured prompt for metadata extraction."""
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        return f"""Extract metadata from this Korean {document_type} document text according to the provided JSON schema.

Document text:
{text}

Schema:
{schema_str}

Instructions:
1. Extract information for each field in the schema
2. Use null for missing or unclear information
3. Convert dates to YYYY-MM-DD format
4. Extract only numbers for amounts
5. Handle checkbox states (☑/☐, ✓/○, ■/□, etc.)
6. Return only valid JSON without markdown formatting

Response (JSON only):"""


class AlibabaCloudExtractor(CloudExtractor):
    """Alibaba Cloud DashScope API extractor using OpenAI-compatible interface."""
    
    def __init__(self, api_key: str, model_id: str = "qwen-plus", region: str = "singapore", 
                 temperature: float = 1.0, top_p: float = 0.8, top_k: int = None):
        super().__init__(api_key, model_id)
        self.region = region
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # Available models for metadata extraction (verified working models)
        self.available_models = {
            "qwen-plus": "Qwen-Plus",
            "qwen-max": "Qwen-Max", 
            "qwen-turbo": "Qwen-Turbo",
            "qwen-vl-plus": "Qwen-VL-Plus",
            "qwen3-next-80b-a3b-instruct": "Qwen3-Next-80B-A3B-Instruct",
            "qwen3-vl-235b-a22b-instruct": "Qwen3-VL-235B-A22B-Instruct",
            "qwen3-235b-a22b-instruct-2507": "Qwen3-235B-A22B-Instruct-2507"
        }
        
        # Map model names to DashScope model IDs
        self.model_mapping = {
            "qwen-plus": "qwen-plus",
            "qwen-max": "qwen-max",
            "qwen-turbo": "qwen-turbo", 
            "qwen-vl-plus": "qwen-vl-plus",
            "qwen3-next-80b-a3b-instruct": "qwen3-next-80b-a3b-instruct",
            "qwen3-vl-235b-a22b-instruct": "qwen3-vl-235b-a22b-instruct",
            "qwen3-235b-a22b-instruct-2507": "qwen3-235b-a22b-instruct-2507"
        }
        
        # Validate model
        if model_id not in self.available_models:
            available_models = ", ".join(self.available_models.keys())
            raise ValueError(f"Unsupported model: {model_id}. Available models: {available_models}")
        
        self.dashscope_model_id = self.model_mapping.get(model_id, model_id)
        
        # Initialize OpenAI client for DashScope compatible API
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
        except ImportError:
            raise ImportError("openai package not found. Install with: pip install openai")
    
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """Extract metadata using Alibaba Cloud DashScope API."""
        prompt = self._create_prompt(text, schema, document_type)
        
        messages = [
            {
                "role": "system",
                "content": "You are a Korean document metadata extraction assistant specialized in contracts, consent forms, and copyright documents. Extract structured information according to the provided JSON schema."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # Prepare generation parameters
        generation_params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": 2048
        }
        
        # Add top_k if specified
        if self.top_k is not None:
            generation_params["top_k"] = self.top_k
        
        try:
            response = self.client.chat.completions.create(
                model=self.dashscope_model_id,
                messages=messages,
                **generation_params
            )
            
            extracted_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                # Clean markdown formatting if present
                cleaned_text = self._clean_markdown_formatting(extracted_text)
                metadata = json.loads(cleaned_text)
                return {
                    "metadata": metadata,
                    "model": self.model_id,
                    "provider": "alibaba_cloud",
                    "processing_time": None,
                    "confidence": None
                }
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {extracted_text}")
                return {"error": "Invalid JSON response"}
                
        except Exception as e:
            logger.error(f"Alibaba Cloud API error: {e}")
            return {"error": str(e)}
    
    def _create_prompt(self, text: str, schema: Dict[str, Any], document_type: str) -> str:
        """Create a structured prompt for metadata extraction."""
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        return f"""당신은 한국어 문서(계약서, 동의서, 기타)에서 정보를 추출하는 도우미입니다.
반드시 유효한 JSON만 출력하세요. 설명·주석·마크다운·코드블록 금지.

다음은 {document_type} 문서의 OCR 텍스트입니다. 주어진 JSON 스키마에 따라 메타데이터를 추출해주세요.

문서 텍스트:
{text}

추출할 메타데이터 스키마:
{schema_str}

지시사항:
1. 텍스트에서 각 필드에 해당하는 정보를 정확히 찾아 추출하세요
2. 정보가 명시적으로 존재하지 않거나 불분명한 경우 반드시 null을 사용하세요 (추측 금지)
3. 날짜는 YYYY-MM-DD 형식으로 변환하세요
4. 금액은 숫자만 추출하세요 (단위 제외)
5. 전화번호는 숫자와 하이픈(-)만 포함하세요
6. 주소는 전체 주소를 정확히 추출하세요
7. 사업자등록번호는 숫자와 하이픈(-)만 포함하세요
8. 체크박스 정보 처리:
   - 체크박스가 체크된 상태(📧, ☑, ✓, ■, ●, ◼, ◉)인 경우 true로 설정
   - 체크박스가 체크되지 않은 상태(☐, □, ○, ◯, ◻, ◦)인 경우 false로 설정
   - 체크박스 패턴을 자동으로 감지하여 일관성 있게 처리
   - OCR 오류 고려: "목제권"은 "복제권"으로 해석
9. 반드시 유효한 JSON 형식으로 응답하세요
10. 추가 정보나 설명은 포함하지 마세요
11. ```json이나 ``` 같은 마크다운 문법 사용 금지

응답 (JSON만):"""
    
    def _clean_markdown_formatting(self, text: str) -> str:
        """Remove markdown formatting from API response."""
        import re
        
        # Remove markdown code blocks
        text = re.sub(r'```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'```\n?', '', text)
        
        # Remove other markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`([^`]*)`', r'\1', text)      # Inline code
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = text.strip()
        
        return text


class TogetherAIExtractor(CloudExtractor):
    """Together AI API extractor."""
    
    def __init__(self, api_key: str, model_id: str):
        super().__init__(api_key, model_id)
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def extract_metadata(self, text: str, schema: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """Extract metadata using Together AI API."""
        prompt = self._create_prompt(text, schema, document_type)
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are a Korean document metadata extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "top_p": 0.8,
            "max_tokens": 2048
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            extracted_text = result["choices"][0]["message"]["content"]
            
            # Parse JSON response
            try:
                metadata = json.loads(extracted_text)
                return {
                    "metadata": metadata,
                    "model": self.model_id,
                    "provider": "together_ai",
                    "processing_time": None,
                    "confidence": None
                }
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {extracted_text}")
                return {"error": "Invalid JSON response"}
                
        except requests.RequestException as e:
            logger.error(f"Together AI API error: {e}")
            return {"error": str(e)}
    
    def _create_prompt(self, text: str, schema: Dict[str, Any], document_type: str) -> str:
        """Create a structured prompt for metadata extraction."""
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        
        return f"""Extract metadata from this Korean {document_type} document text according to the provided JSON schema.

Document text:
{text}

Schema:
{schema_str}

Instructions:
1. Extract information for each field in the schema
2. Use null for missing or unclear information
3. Convert dates to YYYY-MM-DD format
4. Extract only numbers for amounts
5. Handle checkbox states (☑/☐, ✓/○, ■/□, etc.)
6. Return only valid JSON without markdown formatting

Response (JSON only):"""


def create_cloud_extractor(provider: str, api_key: str, model_id: str, **kwargs) -> CloudExtractor:
    """Factory function to create cloud extractors."""
    if provider.lower() == "huggingface":
        return HuggingFaceInferenceExtractor(api_key, model_id)
    elif provider.lower() == "openai":
        return OpenAIExtractor(api_key, model_id)
    elif provider.lower() == "together":
        return TogetherAIExtractor(api_key, model_id)
    elif provider.lower() in ["alibaba", "alibaba_cloud", "dashscope"]:
        # Extract Alibaba-specific parameters
        region = kwargs.get("region", "singapore")
        temperature = kwargs.get("temperature", 1.0)
        top_p = kwargs.get("top_p", 0.8)
        top_k = kwargs.get("top_k", None)
        return AlibabaCloudExtractor(api_key, model_id, region, temperature, top_p, top_k)
    else:
        raise ValueError(f"Unsupported cloud provider: {provider}")


# Example usage and configuration
if __name__ == "__main__":
    # Example configuration for cloud extractors
    cloud_config = {
        "huggingface": {
            "api_key": os.getenv("HUGGINGFACE_API_KEY"),
            "model_id": "upstage/SOLAR-10.7B-Instruct-v1.0"
        },
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model_id": "gpt-4o-mini"
        },
        "together": {
            "api_key": os.getenv("TOGETHER_API_KEY"),
            "model_id": "meta-llama/Llama-2-7b-chat-hf"
        },
        "alibaba": {
            "api_key": os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_API_KEY"),
            "model_id": "qwen-plus",
            "region": "singapore",
            "temperature": 1.0,
            "top_p": 0.8
        }
    }
    
    # Test cloud extractor
    provider = "huggingface"
    if cloud_config[provider]["api_key"]:
        extractor = create_cloud_extractor(
            provider,
            cloud_config[provider]["api_key"],
            cloud_config[provider]["model_id"]
        )
        
        # Test extraction
        test_text = "계약서 내용..."
        test_schema = {"title": "string", "date": "string"}
        
        result = extractor.extract_metadata(test_text, test_schema, "계약서")
        print(f"Cloud extraction result: {result}")
    else:
        print(f"No API key found for {provider}")
