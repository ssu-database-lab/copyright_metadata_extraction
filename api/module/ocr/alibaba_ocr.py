#!/usr/bin/env python3
"""
Alibaba Cloud OCR Provider
"""

import os
import base64
import logging
import re
from pathlib import Path
from typing import Dict, Generator
from openai import OpenAI
import module.env_loader  # noqa: F401 — loads .env on import

# Configure logging first
logger = logging.getLogger(__name__)

class AlibabaCloudOCRProvider:
    """Alibaba Cloud DashScope OCR provider."""
    
    AVAILABLE_MODELS = {
        # Qwen3-VL (dedicated vision-language)
        "qwen-vl-ocr": "Qwen-VL-OCR (Dedicated OCR)",
        "qwen-vl-plus": "Qwen3-VL-Plus",
        "qwen3-vl-30b-a3b-instruct": "Qwen3-VL-30B-A3B",
        "qwen3-vl-235b-a22b-instruct": "Qwen3-VL-235B-A22B",
        # Qwen3.5 (natively multimodal — can do OCR)
        "qwen3.5-plus": "Qwen3.5-Plus (397B, natively multimodal)",
        "qwen3.5-flash": "Qwen3.5-Flash (35B, natively multimodal)",
    }
    
    # Model-level fallback order for OCR
    OCR_MODEL_FALLBACK = {
        "qwen3-vl-235b-a22b-instruct": "qwen3.5-flash",
        "qwen3-vl-30b-a3b-instruct": "qwen3.5-flash",
        "qwen-vl-ocr": "qwen3.5-flash",
        "qwen-vl-plus": "qwen3.5-flash",
        "qwen3.5-plus": "qwen3.5-flash",
    }

    def __init__(self, api_key: str, model: str = "qwen3-vl-235b-a22b-instruct", region: str = "singapore",
                 temperature: float = 1.0, top_p: float = 0.8, top_k: int = None):
        self.api_key = api_key
        self.model = model
        self.region = region
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # Validate model
        if model not in self.AVAILABLE_MODELS:
            available_models = ", ".join(self.AVAILABLE_MODELS.keys())
            raise ValueError(f"Unsupported model: {model}. Available models: {available_models}")
        
        # Map model names to DashScope model IDs
        self.model_mapping = {
            "qwen-vl-ocr": "qwen-vl-ocr",
            "qwen-vl-plus": "qwen-vl-plus",
            "qwen3-vl-30b-a3b-instruct": "qwen3-vl-30b-a3b-instruct",
            "qwen3-vl-235b-a22b-instruct": "qwen3-vl-235b-a22b-instruct",
            "qwen3.5-plus": "qwen3.5-plus",
            "qwen3.5-flash": "qwen3.5-flash",
        }
        
        self.dashscope_model_id = self.model_mapping.get(model, model)
        
        # Initialize OpenAI client for DashScope compatible API
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
                timeout=60.0,       # 60s timeout per request
                max_retries=3       # retry up to 3 times on transient errors
            )
        except ImportError:
            raise ImportError("openai package not found. Install with: pip install openai")
    
    def process_image(self, image_path: str) -> Dict:
        """Process an image using Alibaba Cloud DashScope API, with model-level fallback."""
        result = self._process_image_with_model(image_path, self.dashscope_model_id)

        # If primary model failed and a fallback exists, try it
        if result.get('metadata', {}).get('error') and self.model in self.OCR_MODEL_FALLBACK:
            fallback_model = self.OCR_MODEL_FALLBACK[self.model]
            primary_error = result['metadata']['error']
            logger.warning(f"OCR model '{self.model}' failed: {primary_error}. Trying fallback: {fallback_model}")
            result = self._process_image_with_model(image_path, fallback_model)
            if not result.get('metadata', {}).get('error'):
                result['metadata']['fallback_used'] = True
                result['metadata']['original_model'] = self.model
                logger.info(f"OCR fallback to '{fallback_model}' succeeded")

        return result

    def _process_image_with_model(self, image_path: str, model_id: str) -> Dict:
        """Process an image with a specific model."""
        try:
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            
            # Determine image format
            image_format = image_path.split('.')[-1].lower()
            if image_format in ['jpg', 'jpeg']:
                image_format = 'jpeg'
            elif image_format == 'png':
                image_format = 'png'
            else:
                image_format = 'jpeg'  # Default fallback
            
            # Prepare messages for OpenAI-compatible API
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert OCR (Optical Character Recognition) assistant specialized in Korean and multilingual document processing. Your task is to accurately extract all text content from images while preserving the original layout, formatting, and structure. Pay special attention to Korean text recognition, checkbox states (☑, ☐, ✓, ○, ■, □), and maintain proper line breaks and spacing. IMPORTANT: Output only raw text content without any markdown formatting, code blocks (```), or special formatting symbols."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all the text from the uploaded document. Output only the raw text content without any markdown formatting, code blocks, or special formatting."},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_data}"}}
                    ]
                }
            ]
            
            # Prepare generation parameters
            generation_params = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": 2048
            }
            
            # Note: top_k is not supported by the OpenAI-compatible DashScope endpoint
            
            # Make API call using OpenAI-compatible client
            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                **generation_params
            )
            
            extracted_text = response.choices[0].message.content
            
            # Clean markdown formatting
            extracted_text = self._clean_markdown_formatting(extracted_text)
            
            return {
                'extracted_text': extracted_text,
                'metadata': {
                    'provider': 'alibaba_cloud',
                    'model': self.dashscope_model_id,
                    'confidence': 0.8,
                    'processing_time': None
                }
            }
            
        except Exception as e:
            # Extract detailed error information
            error_str = str(e)
            error_details = error_str
            
            # Check for common Alibaba Cloud error codes and provide user-friendly messages
            if 'Arrearage' in error_str or 'arrearage' in error_str.lower():
                error_details = "Alibaba Cloud account billing issue: Access denied due to outstanding payment. Please check your account billing status in the Alibaba Cloud console."
            elif 'InvalidApiKey' in error_str or 'invalid' in error_str.lower() and 'key' in error_str.lower():
                error_details = "Invalid Alibaba Cloud API key. Please verify your DASHSCOPE_API_KEY or ALIBABA_API_KEY environment variable."
            elif 'QuotaExceeded' in error_str or 'quota' in error_str.lower():
                error_details = "Alibaba Cloud API quota exceeded. Please check your usage limits."
            elif 'code' in error_str.lower() and ('400' in error_str or '401' in error_str or '403' in error_str):
                # Try to extract error details from string representation
                try:
                    import ast
                    # Look for dictionary in error string
                    if "'error':" in error_str or '"error":' in error_str:
                        # Extract error message from string
                        if "'message':" in error_str:
                            # Find the message part
                            msg_start = error_str.find("'message':") + len("'message':")
                            msg_end = error_str.find(",", msg_start)
                            if msg_end == -1:
                                msg_end = error_str.find("}", msg_start)
                            if msg_end > msg_start:
                                msg = error_str[msg_start:msg_end].strip().strip("'\"")
                                if msg:
                                    error_details = f"Alibaba Cloud API Error: {msg}"
                except:
                    pass
            
            logger.error(f"Alibaba Cloud OCR processing error: {error_details}")
            return {
                'extracted_text': '',
                'metadata': {
                    'provider': 'alibaba_cloud',
                    'model': self.dashscope_model_id,
                    'error': error_details,
                    'confidence': 0.0
                }
            }
    
    def process_image_streaming(self, image_path: str) -> Generator[str, None, None]:
        """Process an image with streaming output using OpenAI compatible interface."""
        try:
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            
            # Determine image format
            image_format = image_path.split('.')[-1].lower()
            if image_format in ['jpg', 'jpeg']:
                image_format = 'jpeg'
            elif image_format == 'png':
                image_format = 'png'
            else:
                image_format = 'jpeg'
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert OCR (Optical Character Recognition) assistant specialized in Korean and multilingual document processing. Your task is to accurately extract all text content from images while preserving the original layout, formatting, and structure. Pay special attention to Korean text recognition, checkbox states (☑, ☐, ✓, ○, ■, □), and maintain proper line breaks and spacing. IMPORTANT: Output only raw text content without any markdown formatting, code blocks (```), or special formatting symbols."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all the text from the uploaded document. Output only the raw text content without any markdown formatting, code blocks, or special formatting."},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_data}"}}
                    ]
                }
            ]
            
            # Prepare generation parameters
            generation_params = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": 2048,
                "stream": True
            }
            
            # Note: top_k is not supported by the OpenAI-compatible DashScope endpoint
            
            # Make streaming API call
            completion = self.client.chat.completions.create(
                model=self.dashscope_model_id,
                messages=messages,
                **generation_params
            )
            
            full_content = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    yield content  # Stream output
            
            # Post-process the complete content to remove markdown formatting
            if full_content:
                cleaned_content = self._clean_markdown_formatting(full_content)
                logger.info(f"Streaming completed. Cleaned content length: {len(cleaned_content)}")
            
            logger.info(f"Streaming Alibaba Cloud Qwen-OCR processed {image_path} - {len(full_content)} characters")
            
        except Exception as e:
            logger.error(f"Alibaba Cloud streaming OCR error: {e}")
            yield f"Error: {str(e)}"
    
    def process_image_api_client(self, image_path: str) -> Dict:
        """Process image using API Client approach (non-streaming) with OpenAI compatible interface."""
        try:
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            
            # Determine image format
            image_format = image_path.split('.')[-1].lower()
            if image_format in ['jpg', 'jpeg']:
                image_format = 'jpeg'
            elif image_format == 'png':
                image_format = 'png'
            else:
                image_format = 'jpeg'
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert OCR (Optical Character Recognition) assistant specialized in Korean and multilingual document processing. Your task is to accurately extract all text content from images while preserving the original layout, formatting, and structure. Pay special attention to Korean text recognition, checkbox states (☑, ☐, ✓, ○, ■, □), and maintain proper line breaks and spacing. IMPORTANT: Output only raw text content without any markdown formatting, code blocks (```), or special formatting symbols."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all the text from the uploaded document. Output only the raw text content without any markdown formatting, code blocks, or special formatting."},
                        {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{image_data}"}}
                    ]
                }
            ]
            
            # Prepare generation parameters
            generation_params = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": 2048
            }
            
            # Note: top_k is not supported by the OpenAI-compatible DashScope endpoint
            
            # Make API call
            completion = self.client.chat.completions.create(
                model=self.dashscope_model_id,
                messages=messages,
                **generation_params
            )
            
            extracted_text = completion.choices[0].message.content
            
            # Post-process to remove markdown formatting
            extracted_text = self._clean_markdown_formatting(extracted_text)
            
            return {
                'extracted_text': extracted_text,
                'metadata': {
                    'provider': 'alibaba_cloud',
                    'model': self.dashscope_model_id,
                    'confidence': 0.8,
                    'processing_time': None
                }
            }

        except Exception as e:
            logger.error(f"Alibaba Cloud API Client OCR error: {e}")
            return {
                'extracted_text': '',
                'metadata': {
                    'provider': 'alibaba_cloud',
                    'model': self.dashscope_model_id,
                    'error': str(e),
                    'confidence': 0.0
                }
            }
    
    def _clean_markdown_formatting(self, text: str) -> str:
        """Remove markdown formatting from OCR output."""
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
    
    def get_provider_name(self) -> str:
        """Return the name of the OCR provider."""
        return f"Alibaba Cloud ({self.dashscope_model_id})"
