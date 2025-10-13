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
from dotenv import load_dotenv

# Configure logging first
logger = logging.getLogger(__name__)

# Load environment variables
env_paths = [
    Path(__file__).parent.parent / ".env",  # API directory
    Path(__file__).parent.parent / "web" / ".env",  # Web directory
    Path(__file__).parent.parent / ".env_alibaba",  # Alibaba specific
    Path(__file__).parent.parent / "web" / ".env_alibaba",  # Web Alibaba specific
    Path(__file__).parent.parent.parent / "OCR" / "google_vision" / ".env",  # OCR directory
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from: {env_path}")
        break
else:
    logger.warning("No .env file found. Using system environment variables only.")

class AlibabaCloudOCRProvider:
    """Alibaba Cloud DashScope OCR provider."""
    
    AVAILABLE_MODELS = {
        "qwen-vl-ocr": "Qwen-VL-OCR (Original)",
        "qwen-vl-plus": "Qwen3-VL-Plus",
        "qwen3-vl-30b-a3b-instruct": "Qwen/Qwen3-VL-30B-A3B-Instruct", 
        "qwen3-vl-235b-a22b-instruct": "Qwen/Qwen3-VL-235B-A22B-Instruct"
    }
    
    def __init__(self, api_key: str, model: str = "qwen-vl-ocr", region: str = "singapore", 
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
            "qwen3-vl-235b-a22b-instruct": "qwen3-vl-235b-a22b-instruct"
        }
        
        self.dashscope_model_id = self.model_mapping.get(model, model)
        
        # Initialize OpenAI client for DashScope compatible API
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
        except ImportError:
            raise ImportError("openai package not found. Install with: pip install openai")
    
    def process_image(self, image_path: str) -> Dict:
        """Process an image using Alibaba Cloud DashScope API."""
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
            
            # Add top_k if specified
            if self.top_k is not None:
                generation_params["top_k"] = self.top_k
            
            # Make API call using OpenAI-compatible client
            response = self.client.chat.completions.create(
                model=self.dashscope_model_id,
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
                    'confidence': 0.8,  # Alibaba doesn't provide detailed confidence scores
                    'processing_time': None,
                    'region': self.region
                }
            }
            
        except Exception as e:
            logger.error(f"Alibaba Cloud OCR processing error: {e}")
            return {
                'extracted_text': '',
                'metadata': {
                    'provider': 'alibaba_cloud',
                    'error': str(e),
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
            
            # Add top_k if specified
            if self.top_k is not None:
                generation_params["top_k"] = self.top_k
            
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
            
            # Add top_k if specified
            if self.top_k is not None:
                generation_params["top_k"] = self.top_k
            
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
                    'processing_time': None,
                    'region': self.region,
                    'processing_mode': 'api_client'
                }
            }
            
        except Exception as e:
            logger.error(f"Alibaba Cloud API Client OCR error: {e}")
            return {
                'extracted_text': '',
                'metadata': {
                    'provider': 'alibaba_cloud',
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
