#!/usr/bin/env python3
"""
Universal OCR Processor
Handles all file types: documents (PDF, DOCX, DOC, PPTX, XLS, XLSX, PPT, HWP) and images (GIF, JPG, JPEG, PNG, BMP, TIF, TIFF)
Supports multiple OCR providers: Google Cloud Vision, Mistral, Naver Clova OCR
"""

import os
import sys
import json
import base64
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime
from abc import ABC, abstractmethod
import argparse

# Document processing imports
import fitz  # PyMuPDF
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# OCR provider imports
from google.cloud import vision
from google.protobuf.json_format import MessageToDict
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = log_dir / f"universal_ocr_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Force IPv4 for gRPC (fixes WSL2 IPv6 issues)
os.environ["GRPC_DNS_RESOLVER"] = "native"

# Set up Google Cloud credentials
script_dir = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(script_dir, "semiotic-pager-466612-t0-c587b9296fb8.json")

if os.path.exists(credentials_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    logger.info(f"Using Google Cloud credentials from: {credentials_path}")
else:
    logger.warning("Google Cloud credentials not found. Google OCR will not be available.")

class OCRProvider(ABC):
    """Abstract base class for OCR providers."""
    
    @abstractmethod
    def process_image(self, image_path: str) -> Dict:
        """Process an image and return OCR results."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the OCR provider."""
        pass

class GoogleCloudOCRProvider(OCRProvider):
    """Google Cloud Vision API OCR provider."""
    
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()
    
    def process_image(self, image_path: str) -> Dict:
        """Process an image using Google Cloud Vision API."""
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            response = self.client.document_text_detection(image=image)
            
            if response.error.message:
                raise RuntimeError(f"Google Vision API Error: {response.error.message}")
            
            response_dict = MessageToDict(response._pb)
            
            if response.text_annotations:
                extracted_text = response.text_annotations[0].description
                response_dict['extracted_text'] = extracted_text
                response_dict['text_length'] = len(extracted_text)
            else:
                response_dict['extracted_text'] = ""
                response_dict['text_length'] = 0
            
            logger.info(f"Google Cloud Vision processed {image_path} - {response_dict['text_length']} characters")
            return response_dict
            
        except Exception as e:
            logger.error(f"Error processing {image_path} with Google Cloud Vision: {e}")
            raise
    
    def get_provider_name(self) -> str:
        return "google_cloud"

class MistralOCRProvider(OCRProvider):
    """Mistral OCR API provider."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = Mistral(api_key=api_key)
    
    def encode_image(self, image_path: str) -> str:
        """Encode the image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise
    
    def process_image(self, image_path: str) -> Dict:
        """Process an image using Mistral OCR API."""
        try:
            base64_image = self.encode_image(image_path)
            
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64_image}" 
                },
                include_image_base64=True
            )
            
            # Extract text from Mistral's response
            extracted_text = ""
            if hasattr(ocr_response, 'pages') and ocr_response.pages:
                text_parts = []
                for page in ocr_response.pages:
                    if hasattr(page, 'markdown') and page.markdown:
                        text_parts.append(page.markdown)
                    elif hasattr(page, 'text') and page.text:
                        text_parts.append(page.text)
                extracted_text = "\n\n".join(text_parts)
            elif hasattr(ocr_response, 'text') and ocr_response.text:
                extracted_text = ocr_response.text
            else:
                extracted_text = str(ocr_response)
            
            response_dict = {
                'extracted_text': extracted_text,
                'text_length': len(extracted_text),
                'provider': 'mistral',
                'raw_response': {
                    'response_type': type(ocr_response).__name__,
                    'model': getattr(ocr_response, 'model', 'unknown')
                }
            }
            
            logger.info(f"Mistral OCR processed {image_path} - {len(extracted_text)} characters")
            return response_dict
            
        except Exception as e:
            logger.error(f"Error processing {image_path} with Mistral OCR: {e}")
            raise
    
    def get_provider_name(self) -> str:
        return "mistral"

class NaverClovaOCRProvider(OCRProvider):
    """Naver Clova OCR API provider."""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.api_url = "https://7b8b8b8b8b8b8b8b8b8b8b8b8b8b8b8b.apigw.ntruss.com/ocr/v1/general"
    
    def process_image(self, image_path: str) -> Dict:
        """Process an image using Naver Clova OCR API."""
        try:
            import requests
            
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
            
            headers = {
                'X-OCR-SECRET': self.secret_key
            }
            
            files = {
                'message': (None, json.dumps({
                    'version': 'V2',
                    'requestId': str(datetime.now().timestamp()),
                    'timestamp': int(datetime.now().timestamp()),
                    'images': [{'format': 'jpg', 'name': 'demo'}]
                })),
                'file': (image_path, image_data, 'image/jpeg')
            }
            
            response = requests.post(self.api_url, headers=headers, files=files)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract text from Naver Clova response
            extracted_text = ""
            if 'images' in result and result['images']:
                text_parts = []
                for image in result['images']:
                    if 'fields' in image:
                        for field in image['fields']:
                            if 'inferText' in field:
                                text_parts.append(field['inferText'])
                extracted_text = '\n'.join(text_parts)
            
            response_dict = {
                'extracted_text': extracted_text,
                'text_length': len(extracted_text),
                'provider': 'naver_clova',
                'raw_response': result
            }
            
            logger.info(f"Naver Clova OCR processed {image_path} - {len(extracted_text)} characters")
            return response_dict
            
        except Exception as e:
            logger.error(f"Error processing {image_path} with Naver Clova OCR: {e}")
            raise
    
    def get_provider_name(self) -> str:
        return "naver_clova"

class AlibabaCloudOCRProvider(OCRProvider):
    """Alibaba Cloud Model Studio OCR provider using Qwen-OCR models with DashScope SDK."""
    
    # Available Qwen3-VL models
    AVAILABLE_MODELS = {
        "qwen-vl-ocr": "Qwen-VL-OCR (Original)",
        "qwen-vl-plus": "Qwen3-VL-Plus",
        "qwen3-vl-30b-a3b-instruct": "Qwen/Qwen3-VL-30B-A3B-Instruct", 
        "qwen3-vl-235b-a22b-instruct": "Qwen/Qwen3-VL-235B-A22B-Instruct"
    }
    
    # File size limits according to Alibaba Cloud documentation
    MAX_FILE_SIZE_MB = 10  # Maximum file size in MB
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
    
    def __init__(self, api_key: str, model: str = "qwen-vl-ocr", region: str = "singapore", 
                 temperature: float = 1.0, top_p: float = 0.8, top_k: int = None,
                 min_pixels: int = None, max_pixels: int = None):
        """
        Initialize the Alibaba Cloud Model Studio OCR client.
        
        Args:
            api_key: Your Alibaba Cloud API key
            model: Model name - choose from:
                - "qwen-vl-ocr" (default, original Qwen-VL-OCR)
                - "qwen-vl-plus" (Qwen3-VL-Plus)
                - "qwen3-vl-30b-a3b-instruct" (Qwen3-VL-30B-A3B-Instruct)
                - "qwen3-vl-235b-a22b-instruct" (Qwen3-VL-235B-A22B-Instruct)
            region: Region - "singapore" or "china"
            temperature: Controls randomness (0.0-2.0, default: 1.0)
            top_p: Controls nucleus sampling (0.0-1.0, default: 0.8)
            top_k: Controls candidate set size (optional)
            min_pixels: Minimum pixel threshold for image scaling (default: 28*28*4 = 3136)
            max_pixels: Maximum pixel threshold for image scaling (default: 28*28*8192 = 6422528)
        """
        try:
            import dashscope
            self.dashscope = dashscope
        except ImportError:
            raise ImportError("dashscope package not found. Install with: pip install dashscope")
        
        # Validate and map model name
        if model not in self.AVAILABLE_MODELS:
            available_models = ", ".join(self.AVAILABLE_MODELS.keys())
            raise ValueError(f"Unsupported model: {model}. Available models: {available_models}")
        
        self.api_key = api_key
        self.model = model
        self.region = region
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # Set default pixel thresholds according to Alibaba Cloud documentation
        # Alibaba Cloud requires min_pixels >= 65536 (256*256)
        self.min_pixels = min_pixels or (256 * 256)  # Default: 65536 pixels
        self.max_pixels = max_pixels or (28 * 28 * 8192)  # Default: 6422528 pixels
        
        # Map model names to actual DashScope model IDs
        self.model_mapping = {
            "qwen-vl-ocr": "qwen-vl-ocr",
            "qwen-vl-plus": "qwen-vl-plus", 
            "qwen3-vl-30b-a3b-instruct": "qwen3-vl-30b-a3b-instruct",
            "qwen3-vl-235b-a22b-instruct": "qwen3-vl-235b-a22b-instruct"
        }
        
        self.dashscope_model_id = self.model_mapping.get(model, model)
        
        # Set base URL based on region
        if region == "singapore":
            self.dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
        else:  # china
            self.dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
    
    def process_image_streaming(self, image_path: str):
        """Process image with streaming support using OpenAI compatible interface."""
        try:
            # Import OpenAI client for streaming
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package not found. Install with: pip install openai")
            
            # Initialize OpenAI client with DashScope compatible endpoint
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
            
            # Read and encode image as base64
            import base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine image format from file extension
            image_format = Path(image_path).suffix.lower().lstrip('.')
            if image_format == 'tif':
                image_format = 'tiff'  # Convert tif to tiff for MIME type
            
            # Prepare messages in OpenAI format with base64 image
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert OCR (Optical Character Recognition) assistant specialized in Korean and multilingual document processing. Your task is to accurately extract all text content from images while preserving the original layout, formatting, and structure. Pay special attention to Korean text recognition, checkbox states (☑, ☐, ✓, ○, ■, □), and maintain proper line breaks and spacing. IMPORTANT: Output only raw text content without any markdown formatting, code blocks (```), or special formatting symbols."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all the text from the uploaded document. Output only the raw text content without any markdown formatting, code blocks, or special formatting."},
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
            
            logger.info(f"Making streaming Alibaba Cloud Qwen-OCR request for {image_path}")
            
            # Prepare generation parameters
            generation_params = {
                "temperature": self.temperature,
                "top_p": self.top_p
            }
            
            # Add top_k if specified
            if self.top_k is not None:
                generation_params["top_k"] = self.top_k
            
            # Call OpenAI compatible interface with streaming
            completion = client.chat.completions.create(
                model=self.dashscope_model_id,
                messages=messages,
                stream=True,
                **generation_params
            )
            
            # Stream the response
            full_content = ""
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    yield content  # Stream output
            
            # Post-process the complete content to remove markdown formatting
            if full_content:
                cleaned_content = self._clean_markdown_formatting(full_content)
                # Note: This is for logging purposes only, streaming has already yielded content
                logger.info(f"Streaming completed. Cleaned content length: {len(cleaned_content)}")
            
            logger.info(f"Streaming Alibaba Cloud Qwen-OCR processed {image_path} - {len(full_content)} characters")
            
            return full_content
            
        except Exception as e:
            logger.error(f"Error in streaming OCR processing for {image_path}: {e}")
            raise
    
    def process_image_api_client(self, image_path: str) -> Dict:
        """Process image using API Client approach (non-streaming) with OpenAI compatible interface."""
        try:
            # Import OpenAI client for API client approach
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package not found. Install with: pip install openai")
            
            # Initialize OpenAI client with DashScope compatible endpoint
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
            
            # Read and encode image as base64
            import base64
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine image format from file extension
            image_format = Path(image_path).suffix.lower().lstrip('.')
            if image_format == 'tif':
                image_format = 'tiff'  # Convert tif to tiff for MIME type
            
            # Prepare messages in OpenAI format with base64 image
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert OCR (Optical Character Recognition) assistant specialized in Korean and multilingual document processing. Your task is to accurately extract all text content from images while preserving the original layout, formatting, and structure. Pay special attention to Korean text recognition, checkbox states (☑, ☐, ✓, ○, ■, □), and maintain proper line breaks and spacing. IMPORTANT: Output only raw text content without any markdown formatting, code blocks (```), or special formatting symbols."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all the text from the uploaded document. Output only the raw text content without any markdown formatting, code blocks, or special formatting."},
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
            
            logger.info(f"Making API Client Alibaba Cloud Qwen-OCR request for {image_path}")
            
            # Prepare generation parameters
            generation_params = {
                "temperature": self.temperature,
                "top_p": self.top_p
            }
            
            # Add top_k if specified
            if self.top_k is not None:
                generation_params["top_k"] = self.top_k
            
            # Call OpenAI compatible interface without streaming
            completion = client.chat.completions.create(
                model=self.dashscope_model_id,
                messages=messages,
                stream=False,  # Non-streaming
                **generation_params
            )
            
            # Extract text from complete response
            extracted_text = completion.choices[0].message.content
            
            # Post-process to remove markdown formatting
            extracted_text = self._clean_markdown_formatting(extracted_text)
            
            logger.info(f"API Client Alibaba Cloud Qwen-OCR processed {image_path} - {len(extracted_text)} characters")
            
            return {
                "extracted_text": extracted_text,
                "model": self.dashscope_model_id,
                "provider": "alibaba_api_client",
                "processing_time": None,  # Could be added if needed
                "confidence": None,  # Could be added if needed
                "metadata": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "api_approach": "openai_compatible",
                    "streaming": False,
                    "image_format": image_format
                }
            }
            
        except Exception as e:
            logger.error(f"Error in API Client OCR processing for {image_path}: {e}")
            raise
    
    def _clean_markdown_formatting(self, text: str) -> str:
        """Remove markdown formatting from OCR output."""
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
    
    def _check_file_size(self, image_path: str) -> Dict[str, any]:
        """
        Check if the image file meets Alibaba Cloud size requirements.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with file size information and recommendations
        """
        try:
            file_size = os.path.getsize(image_path)
            file_size_mb = file_size / (1024 * 1024)
            
            result = {
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size_mb, 2),
                "within_limit": file_size <= self.MAX_FILE_SIZE_BYTES,
                "recommendation": ""
            }
            
            if file_size > self.MAX_FILE_SIZE_BYTES:
                result["recommendation"] = f"File size ({file_size_mb:.2f}MB) exceeds the 10MB limit. Consider compressing the image or splitting it into smaller parts."
                logger.warning(f"File {image_path} is {file_size_mb:.2f}MB, exceeding the 10MB limit")
            else:
                result["recommendation"] = "File size is within acceptable limits."
                logger.info(f"File {image_path} is {file_size_mb:.2f}MB, within the 10MB limit")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking file size for {image_path}: {e}")
            return {
                "file_size_bytes": 0,
                "file_size_mb": 0,
                "within_limit": False,
                "recommendation": f"Error checking file size: {e}"
            }
    
    def _get_image_dimensions(self, image_path: str) -> Dict[str, any]:
        """
        Get image dimensions to help with pixel calculations.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image dimension information
        """
        try:
            from PIL import Image
            
            with Image.open(image_path) as img:
                width, height = img.size
                total_pixels = width * height
                
                return {
                    "width": width,
                    "height": height,
                    "total_pixels": total_pixels,
                    "aspect_ratio": round(width / height, 2) if height > 0 else 0,
                    "within_pixel_limits": self.min_pixels <= total_pixels <= self.max_pixels,
                    "recommendation": ""
                }
                
        except ImportError:
            logger.warning("PIL (Pillow) not available for image dimension checking")
            return {
                "width": 0,
                "height": 0,
                "total_pixels": 0,
                "aspect_ratio": 0,
                "within_pixel_limits": True,
                "recommendation": "PIL not available for dimension checking"
            }
        except Exception as e:
            logger.error(f"Error getting image dimensions for {image_path}: {e}")
            return {
                "width": 0,
                "height": 0,
                "total_pixels": 0,
                "aspect_ratio": 0,
                "within_pixel_limits": True,
                "recommendation": f"Error getting dimensions: {e}"
            }
    
    def process_image(self, image_path: str) -> Dict:
        """Process image using Alibaba Cloud Qwen-OCR model with DashScope SDK."""
        try:
            # Check file size before processing
            file_size_info = self._check_file_size(image_path)
            processed_image_path = image_path
            
            # If file exceeds size limit, automatically compress it
            if not file_size_info["within_limit"]:
                logger.info(f"File size ({file_size_info['file_size_mb']}MB) exceeds 10MB limit. Automatically compressing...")
                
                try:
                    # Import compression utility
                    from compress_images import ImageCompressor
                    
                    # Create temporary compressed file
                    temp_dir = Path(image_path).parent / "temp_compressed"
                    temp_dir.mkdir(exist_ok=True)
                    compressed_path = temp_dir / f"{Path(image_path).stem}_auto_compressed.jpg"
                    
                    # Compress the image
                    compressor = ImageCompressor(str(temp_dir))
                    processed_image_path = compressor.compress_image(image_path, str(compressed_path))
                    
                    # Update file size info for compressed file
                    file_size_info = self._check_file_size(processed_image_path)
                    logger.info(f"Auto-compression complete: {processed_image_path} ({file_size_info['file_size_mb']}MB)")
                    
                except Exception as e:
                    error_msg = f"Auto-compression failed: {e}"
                    logger.error(error_msg)
                    return {
                        "provider": "alibaba_cloud",
                        "extracted_text": "",
                        "text_length": 0,
                        "error": error_msg,
                        "file_size_info": file_size_info,
                        "status": "error"
                    }
            
            # Get image dimensions for better processing
            dimension_info = self._get_image_dimensions(processed_image_path)
            
            # Convert image path to file:// format for local file upload
            image_file_path = f"file://{os.path.abspath(processed_image_path)}"
            
            # Prepare image content with proper pixel parameters
            image_content = {
                "image": image_file_path,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "enable_rotate": True
            }
            
            # Prepare messages for DashScope SDK
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert OCR (Optical Character Recognition) assistant specialized in Korean and multilingual document processing. Your task is to accurately extract all text content from images while preserving the original layout, formatting, and structure. Pay special attention to Korean text recognition, checkbox states (☑, ☐, ✓, ○, ■, □), and maintain proper line breaks and spacing. IMPORTANT: Output only raw text content without any markdown formatting, code blocks (```), or special formatting symbols."
                },
                {
                    "role": "user",
                    "content": [
                        image_content,
                        {"text": "Extract all the text from the uploaded document. Output only the raw text content without any markdown formatting, code blocks, or special formatting."}
                    ]
                }
            ]
            
            logger.info(f"Making Alibaba Cloud Qwen-OCR request using DashScope SDK for {processed_image_path}")
            
            # Prepare generation parameters
            generation_params = {
                "temperature": self.temperature,
                "top_p": self.top_p
            }
            
            # Add top_k if specified
            if self.top_k is not None:
                generation_params["top_k"] = self.top_k
            
            # Call DashScope MultiModalConversation
            response = self.dashscope.MultiModalConversation.call(
                api_key=self.api_key,
                model=self.dashscope_model_id,
                messages=messages,
                ocr_options={"task": "multi_lan"},
                **generation_params
            )
            print(response)
            print("-"*100)

            # Extract text from response
            extracted_text = ""
            if "output" in response and "choices" in response["output"]:
                choices = response["output"]["choices"]
                if choices and len(choices) > 0:
                    choice = choices[0]
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                        if isinstance(content, list) and len(content) > 0:
                            # Extract text from content array
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    extracted_text += item["text"]
                        elif isinstance(content, str):
                            extracted_text = content
            
            logger.info(f"Alibaba Cloud Qwen-OCR processed {processed_image_path} - {len(extracted_text)} characters")
            
            return {
                "provider": "alibaba_cloud",
                "extracted_text": extracted_text.strip(),
                "text_length": len(extracted_text.strip()),
                "raw_response": response,
                "file_size_info": file_size_info,
                "dimension_info": dimension_info,
                "processing_params": {
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "model": self.dashscope_model_id
                },
                "status": "success"
            }
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Alibaba Cloud Qwen-OCR error: {error_msg}")
            
            # Try to get file size info even if processing failed
            try:
                file_size_info = self._check_file_size(processed_image_path)
                dimension_info = self._get_image_dimensions(processed_image_path)
            except:
                file_size_info = {"file_size_mb": 0, "within_limit": False}
                dimension_info = {"total_pixels": 0, "within_pixel_limits": False}
            
            return {
                "provider": "alibaba_cloud",
                "extracted_text": "",
                "text_length": 0,
                "error": error_msg,
                "file_size_info": file_size_info,
                "dimension_info": dimension_info,
                "status": "error"
            }
    
    def get_provider_name(self) -> str:
        return "alibaba_cloud"
    
    @classmethod
    def list_available_models(cls) -> Dict[str, str]:
        """Return dictionary of available models."""
        return cls.AVAILABLE_MODELS.copy()

class DeepSeekOCRProvider(OCRProvider):
    """DeepSeek-OCR provider using Hugging Face transformers."""
    
    # Available DeepSeek-OCR modes
    AVAILABLE_MODES = {
        "tiny": "Tiny (512×512, 64 vision tokens)",
        "small": "Small (640×640, 100 vision tokens)", 
        "base": "Base (1024×1024, 256 vision tokens)",
        "large": "Large (1280×1280, 400 vision tokens)",
        "gundam": "Gundam (Dynamic resolution: n×640×640 + 1×1024×1024)"
    }
    
    def __init__(self, mode: str = "base", device: str = "cuda", 
                 prompt: str = "<image>\n<|grounding|>Convert the document to markdown."):
        """
        Initialize DeepSeek-OCR provider.
        
        Args:
            mode: Processing mode - "tiny", "small", "base", "large", or "gundam"
            device: Device to run on - "cuda" or "cpu"
            prompt: OCR prompt template
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers and torch packages not found. Install with: pip install transformers torch")
        
        self.mode = mode
        self.device = device
        self.prompt = prompt
        
        # Check for local model first, then fall back to Hugging Face
        script_dir = Path(__file__).parent
        local_model_path = script_dir.parent.parent / "api" / "models" / "ocr" / "deepseek-ai_DeepSeek-OCR"
        
        if local_model_path.exists():
            self.model_name = str(local_model_path)
            logger.info(f"Using local DeepSeek-OCR model: {self.model_name}")
        else:
            self.model_name = 'deepseek-ai/DeepSeek-OCR'
            logger.info(f"Local model not found, using Hugging Face: {self.model_name}")
        
        # Set CUDA device if available
        if device == "cuda" and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.device = "cuda"
        else:
            self.device = "cpu"
            logger.warning("CUDA not available, using CPU (slower)")
        
        # Initialize model and tokenizer
        try:
            logger.info(f"Loading DeepSeek-OCR model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                _attn_implementation='flash_attention_2', 
                trust_remote_code=True, 
                use_safetensors=True
            )
            
            if self.device == "cuda":
                self.model = self.model.eval().cuda().to(torch.bfloat16)
            else:
                self.model = self.model.eval().to(torch.float32)
                
            logger.info(f"DeepSeek-OCR model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load DeepSeek-OCR model: {e}")
            raise
    
    def _get_mode_params(self) -> Dict[str, int]:
        """Get processing parameters based on mode."""
        mode_params = {
            "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
            "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
            "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
            "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
            "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True}
        }
        return mode_params.get(self.mode, mode_params["base"])
    
    def process_image(self, image_path: str) -> Dict:
        """Process an image using DeepSeek-OCR."""
        try:
            logger.info(f"Processing image with DeepSeek-OCR: {image_path}")
            
            # Get mode parameters
            params = self._get_mode_params()
            
            # Create temporary output directory
            temp_output_dir = Path(image_path).parent / "deepseek_temp"
            temp_output_dir.mkdir(exist_ok=True)
            
            # Run inference
            result = self.model.infer(
                tokenizer=self.tokenizer,
                prompt=self.prompt,
                image_file=image_path,
                output_path=str(temp_output_dir),
                base_size=params["base_size"],
                image_size=params["image_size"],
                crop_mode=params["crop_mode"],
                save_results=True,
                test_compress=False
            )
            
            # Extract text from result
            extracted_text = ""
            if isinstance(result, str):
                extracted_text = result
            elif isinstance(result, dict) and "text" in result:
                extracted_text = result["text"]
            elif hasattr(result, 'text'):
                extracted_text = result.text
            
            # Clean up temporary files
            try:
                import shutil
                shutil.rmtree(temp_output_dir)
            except:
                pass  # Ignore cleanup errors
            
            logger.info(f"DeepSeek-OCR processed {image_path} - {len(extracted_text)} characters")
            
            return {
                "provider": "deepseek_ocr",
                "extracted_text": extracted_text.strip(),
                "text_length": len(extracted_text.strip()),
                "model": self.model_name,
                "mode": self.mode,
                "device": self.device,
                "prompt": self.prompt,
                "processing_params": params,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path} with DeepSeek-OCR: {e}")
            return {
                "provider": "deepseek_ocr",
                "extracted_text": "",
                "text_length": 0,
                "error": str(e),
                "model": self.model_name,
                "mode": self.mode,
                "status": "error"
            }
    
    def get_provider_name(self) -> str:
        return "deepseek_ocr"
    
    @classmethod
    def list_available_modes(cls) -> Dict[str, str]:
        """Return dictionary of available processing modes."""
        return cls.AVAILABLE_MODES.copy()

class PaddleOCRVLProvider(OCRProvider):
    """PaddleOCR-VL provider using local model from api/models/ocr/."""
    
    def __init__(self, device: str = "cuda", task: str = "ocr"):
        """
        Initialize PaddleOCR-VL provider.
        
        Args:
            device: Device to run on - "cuda" or "cpu"
            task: Task type - "ocr", "table", "chart", or "formula"
        """
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            from PIL import Image
            import torch
        except ImportError:
            raise ImportError("transformers, torch, and PIL packages not found. Install with: pip install transformers torch pillow")
        
        self.device = device
        self.task = task
        
        # Check for local model
        script_dir = Path(__file__).parent
        local_model_path = script_dir.parent.parent / "api" / "models" / "ocr" / "PaddlePaddle_PaddleOCR-VL"
        
        if local_model_path.exists():
            self.model_name = str(local_model_path)
            logger.info(f"Using local PaddleOCR-VL model: {self.model_name}")
        else:
            self.model_name = 'PaddlePaddle/PaddleOCR-VL'
            logger.info(f"Local model not found, using Hugging Face: {self.model_name}")
        
        # Set device
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            logger.warning("CUDA not available, using CPU (slower)")
        
        # Define prompts for different tasks
        self.PROMPTS = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
        }
        
        # Check for torchvision NMS operator (required for some PaddleOCR-VL operations)
        try:
            import torchvision
            if not hasattr(torchvision.ops, 'nms'):
                logger.warning("torchvision.ops.nms not available. This may cause issues with PaddleOCR-VL.")
                logger.warning("Try reinstalling torchvision: pip install --upgrade torchvision")
        except ImportError:
            logger.warning("torchvision not installed. Some PaddleOCR-VL features may not work.")
        
        # Initialize model and processor
        try:
            logger.info(f"Loading PaddleOCR-VL model: {self.model_name}")
            
            # Try loading the processor first (less likely to trigger torchvision issues)
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
                logger.info("Processor loaded successfully")
            except Exception as proc_error:
                logger.warning(f"Failed to load processor first: {proc_error}")
                # Continue anyway, will try again after model loads
            
            # Try loading with different strategies
            model_loaded = False
            load_strategies = [
                # Strategy 1: Standard load with device_map
                {
                    "name": "device_map auto",
                    "kwargs": {
                        "trust_remote_code": True,
                        "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32,
                        "device_map": "auto" if self.device == "cuda" else None
                    }
                },
                # Strategy 2: Standard load without device_map
                {
                    "name": "standard load",
                    "kwargs": {
                        "trust_remote_code": True,
                        "torch_dtype": torch.bfloat16 if self.device == "cuda" else torch.float32
                    }
                },
                # Strategy 3: Load with float32 (more compatible)
                {
                    "name": "float32 load",
                    "kwargs": {
                        "trust_remote_code": True,
                        "torch_dtype": torch.float32
                    }
                },
            ]
            
            for strategy in load_strategies:
                try:
                    logger.info(f"Trying load strategy: {strategy['name']}")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **strategy['kwargs']
                    )
                    if self.device == "cpu" or not hasattr(self.model, 'device'):
                        self.model = self.model.to(self.device)
                    self.model = self.model.eval()
                    model_loaded = True
                    logger.info(f"Successfully loaded using strategy: {strategy['name']}")
                    break
                except Exception as strategy_error:
                    logger.warning(f"Strategy '{strategy['name']}' failed: {strategy_error}")
                    continue
            
            if not model_loaded:
                raise RuntimeError("All model loading strategies failed")
            
            # Load processor if not already loaded
            if not hasattr(self, 'processor') or self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            
            logger.info(f"PaddleOCR-VL model loaded successfully on {self.device}")
            
        except Exception as e:
            error_msg = str(e)
            if "torchvision::nms" in error_msg or "nms" in error_msg.lower():
                logger.error(f"Failed to load PaddleOCR-VL model: {error_msg}")
                logger.error("\n" + "="*60)
                logger.error("TORCHVISION NMS OPERATOR ERROR")
                logger.error("="*60)
                logger.error("This error occurs when PaddleOCR-VL's custom code tries to use")
                logger.error("torchvision operations that aren't available in your environment.")
                logger.error("\nTo fix this issue, try one of the following:")
                logger.error("\n1. Reinstall torchvision with CUDA support:")
                logger.error("   pip uninstall torchvision")
                logger.error("   pip install torchvision --index-url https://download.pytorch.org/whl/cu128")
                logger.error("\n2. Or reinstall both PyTorch and torchvision together:")
                logger.error("   pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128")
                logger.error("\n3. Try using CPU mode (set PADDLEOCR_DEVICE=cpu)")
                logger.error("\n4. Check if the model's custom code has compatibility issues")
                logger.error("   The error might be in the model's trust_remote_code components")
                logger.error("="*60)
            else:
                logger.error(f"Failed to load PaddleOCR-VL model: {error_msg}")
            raise
    
    def process_image(self, image_path: str) -> Dict:
        """Process an image using PaddleOCR-VL."""
        try:
            logger.info(f"Processing image with PaddleOCR-VL: {image_path}")
            
            from PIL import Image
            import torch
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Prepare messages
            messages = [
                {"role": "user",
                 "content": [
                     {"type": "image", "image": image},
                     {"type": "text", "text": self.PROMPTS.get(self.task, "OCR:")},
                 ]
                }
            ]
            
            # Process inputs
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=1024)
            
            # Decode output
            extracted_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Remove the prompt from the output
            prompt_text = self.PROMPTS.get(self.task, "OCR:")
            if prompt_text in extracted_text:
                extracted_text = extracted_text.split(prompt_text, 1)[-1].strip()
            
            logger.info(f"PaddleOCR-VL processed {image_path} - {len(extracted_text)} characters")
            
            return {
                "provider": "paddleocr_vl",
                "extracted_text": extracted_text.strip(),
                "text_length": len(extracted_text.strip()),
                "model": self.model_name,
                "task": self.task,
                "device": self.device,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path} with PaddleOCR-VL: {e}")
            return {
                "provider": "paddleocr_vl",
                "extracted_text": "",
                "text_length": 0,
                "error": str(e),
                "model": self.model_name,
                "task": self.task,
                "status": "error"
            }
    
    def get_provider_name(self) -> str:
        return "paddleocr_vl"

class FileProcessor:
    """Handles different file types and converts them to images for OCR."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def process_pdf(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to images."""
        image_paths = []
        try:
            doc = fitz.open(pdf_path)
            pdf_name = Path(pdf_path).stem
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                
                image_path = self.output_dir / f"{pdf_name}_page_{page_num + 1:03d}.png"
                pix.save(str(image_path))
                image_paths.append(str(image_path))
            
            doc.close()
            logger.info(f"Converted PDF {pdf_path} to {len(image_paths)} images")
            
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path}: {e}")
            raise
        
        return image_paths
    
    def process_docx(self, docx_path: str) -> List[str]:
        """Convert DOCX to images (requires additional processing)."""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not available. Install with: pip install python-docx")
        
        # For now, we'll extract text directly from DOCX
        # In a full implementation, you'd convert to images first
        try:
            doc = Document(docx_path)
            text_content = []
            for paragraph in doc.paragraphs:
                text_content.append(paragraph.text)
            
            # Save as text file for now
            text_path = self.output_dir / f"{Path(docx_path).stem}_extracted.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(text_content))
            
            logger.info(f"Extracted text from DOCX {docx_path}")
            return [str(text_path)]
            
        except Exception as e:
            logger.error(f"Error processing DOCX {docx_path}: {e}")
            raise
    
    def process_image(self, image_path: str) -> List[str]:
        """Process image files directly."""
        return [image_path]
    
    def process_file(self, file_path: str) -> List[str]:
        """Process any file type and return list of image paths."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self.process_pdf(str(file_path))
        elif extension in ['.docx']:
            return self.process_docx(str(file_path))
        elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff']:
            return self.process_image(str(file_path))
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return []

class UniversalOCRProcessor:
    """Main OCR processor that handles all file types."""
    
    def __init__(self, provider: str, output_dir: str = "universal_ocr_results", model: str = None):
        """
        Initialize Universal OCR Processor.
        
        Args:
            provider: OCR provider ("google_cloud", "mistral", "naver", "alibaba")
            output_dir: Base output directory for results
            model: Model name for providers that support multiple models (e.g., alibaba)
        """
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Initialize OCR provider
        self.provider_name = provider.lower()
        if self.provider_name == "google_cloud":
            self.ocr_provider = GoogleCloudOCRProvider()
        elif self.provider_name == "mistral":
            api_key = os.getenv('MISTRAL_API_KEY')
            if not api_key:
                raise ValueError("MISTRAL_API_KEY not found in environment variables")
            self.ocr_provider = MistralOCRProvider(api_key)
        elif self.provider_name == "naver":
            api_key = os.getenv('NAVER_API_KEY')
            secret_key = os.getenv('NAVER_SECRET_KEY')
            if not api_key or not secret_key:
                raise ValueError("NAVER_API_KEY and NAVER_SECRET_KEY not found in environment variables")
            self.ocr_provider = NaverClovaOCRProvider(api_key, secret_key)
        elif self.provider_name == "alibaba":
            api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('ALIBABA_API_KEY')
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY or ALIBABA_API_KEY not found in environment variables")
            # Get region from environment or default to singapore
            region = os.getenv('ALIBABA_REGION', 'singapore')
            # Use provided model or default to qwen-vl-ocr
            alibaba_model = model or os.getenv('ALIBABA_MODEL', 'qwen-vl-ocr')
            
            # Get generation parameters from environment or use defaults
            temperature = float(os.getenv('ALIBABA_TEMPERATURE', '1.0'))
            top_p = float(os.getenv('ALIBABA_TOP_P', '0.8'))
            top_k = os.getenv('ALIBABA_TOP_K')
            if top_k:
                top_k = int(top_k)
            
            # Get pixel parameters from environment or use defaults
            min_pixels = os.getenv('ALIBABA_MIN_PIXELS')
            if min_pixels:
                min_pixels = int(min_pixels)
            
            max_pixels = os.getenv('ALIBABA_MAX_PIXELS')
            if max_pixels:
                max_pixels = int(max_pixels)
            
            self.ocr_provider = AlibabaCloudOCRProvider(
                api_key, 
                model=alibaba_model, 
                region=region,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_pixels=min_pixels,
                max_pixels=max_pixels
            )
        elif self.provider_name == "deepseek":
            # Get DeepSeek-OCR mode from environment or use provided model parameter
            deepseek_mode = model or os.getenv('DEEPSEEK_MODE', 'base')
            # Get device preference from environment
            device = os.getenv('DEEPSEEK_DEVICE', 'cuda')
            # Get custom prompt from environment
            prompt = os.getenv('DEEPSEEK_PROMPT', '<image>\n<|grounding|>Convert the document to markdown.')
            
            self.ocr_provider = DeepSeekOCRProvider(
                mode=deepseek_mode,
                device=device,
                prompt=prompt
            )
        elif self.provider_name == "paddleocr_vl":
            # Get task type from model parameter or use default
            task = model or os.getenv('PADDLEOCR_TASK', 'ocr')
            # Get device preference from environment, with fallback to CPU if CUDA fails
            device = os.getenv('PADDLEOCR_DEVICE', 'cuda')
            
            try:
                self.ocr_provider = PaddleOCRVLProvider(
                    device=device,
                    task=task
                )
            except Exception as e:
                # If CUDA fails with torchvision error, try CPU as fallback
                if "torchvision::nms" in str(e) and device == "cuda":
                    logger.warning("CUDA mode failed, trying CPU mode as fallback...")
                    device = "cpu"
                    self.ocr_provider = PaddleOCRVLProvider(
                        device=device,
                        task=task
                    )
                else:
                    raise
        else:
            raise ValueError(f"Unsupported OCR provider: {provider}")
        
        # Create provider-specific output directory with model subdirectory for Alibaba, DeepSeek, and PaddleOCR-VL
        if self.provider_name == "alibaba" and hasattr(self.ocr_provider, 'model'):
            # Create model-specific subdirectory for Alibaba Qwen3-VL models
            model_name = self.ocr_provider.model.replace('-', '_')
            self.provider_output_dir = self.base_output_dir / f"{self.provider_name}_ocr" / model_name
        elif self.provider_name == "deepseek" and hasattr(self.ocr_provider, 'mode'):
            # Create mode-specific subdirectory for DeepSeek-OCR
            mode_name = self.ocr_provider.mode.replace('-', '_')
            self.provider_output_dir = self.base_output_dir / f"{self.provider_name}_ocr" / mode_name
        elif self.provider_name == "paddleocr_vl" and hasattr(self.ocr_provider, 'task'):
            # Create task-specific subdirectory for PaddleOCR-VL
            task_name = self.ocr_provider.task.replace('-', '_')
            self.provider_output_dir = self.base_output_dir / f"{self.provider_name}_ocr" / task_name
        else:
            self.provider_output_dir = self.base_output_dir / f"{self.provider_name}_ocr"
        
        self.provider_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize file processor with provider-specific directory
        self.file_processor = FileProcessor(self.provider_output_dir / "converted_images")
        
        logger.info(f"Initialized Universal OCR Processor with {self.provider_name} provider")
        if self.provider_name == "alibaba" and hasattr(self.ocr_provider, 'model'):
            logger.info(f"Model: {self.ocr_provider.model}")
            logger.info(f"Temperature: {self.ocr_provider.temperature}")
            logger.info(f"Top-p: {self.ocr_provider.top_p}")
            if self.ocr_provider.top_k is not None:
                logger.info(f"Top-k: {self.ocr_provider.top_k}")
            logger.info(f"Min pixels: {self.ocr_provider.min_pixels}")
            logger.info(f"Max pixels: {self.ocr_provider.max_pixels}")
            logger.info(f"Max file size: {self.ocr_provider.MAX_FILE_SIZE_MB}MB")
        elif self.provider_name == "deepseek" and hasattr(self.ocr_provider, 'mode'):
            logger.info(f"Model: {self.ocr_provider.model_name}")
            logger.info(f"Mode: {self.ocr_provider.mode}")
            logger.info(f"Device: {self.ocr_provider.device}")
            logger.info(f"Prompt: {self.ocr_provider.prompt}")
        logger.info(f"Provider output directory: {self.provider_output_dir}")
    
    def process_single_file_streaming(self, file_path: str):
        """Process a single file with streaming output."""
        file_path = Path(file_path)
        logger.info(f"Processing file with streaming: {file_path.name}")
        
        try:
            # Create structured output paths
            output_paths = self.create_structured_output_paths(str(file_path))
            
            # Create a temporary file processor for this specific file
            temp_file_processor = FileProcessor(output_paths['images_dir'])
            
            # Convert file to images
            image_paths = temp_file_processor.process_file(str(file_path))
            
            if not image_paths:
                yield f"Error: Unsupported file type or conversion failed for {file_path.name}"
                return
            
            # Process each image with streaming OCR
            all_text = []
            page_results = []
            
            for i, image_path in enumerate(image_paths):
                try:
                    logger.info(f"Processing page {i+1}/{len(image_paths)}: {Path(image_path).name}")
                    
                    # Check if provider supports streaming
                    if hasattr(self.ocr_provider, 'process_image_streaming'):
                        # Use streaming OCR
                        page_text = ""
                        for chunk in self.ocr_provider.process_image_streaming(image_path):
                            page_text += chunk
                            yield chunk  # Stream the chunk
                        
                        # Store the complete text for this page
                        all_text.append(page_text)
                        page_results.append({
                            'page_number': i + 1,
                            'image_path': image_path,
                            'extracted_text': page_text,
                            'text_length': len(page_text),
                            'status': 'success'
                        })
                    else:
                        # Fallback to regular OCR
                        result = self.ocr_provider.process_image(image_path)
                        page_text = result.get('extracted_text', '')
                        all_text.append(page_text)
                        page_results.append({
                            'page_number': i + 1,
                            'image_path': image_path,
                            'extracted_text': page_text,
                            'text_length': len(page_text),
                            'status': 'success'
                        })
                        yield page_text  # Stream the complete text
                
                except Exception as e:
                    logger.error(f"Error processing page {i+1}: {e}")
                    page_results.append({
                        'page_number': i + 1,
                        'image_path': image_path,
                        'extracted_text': '',
                        'text_length': 0,
                        'status': 'failed',
                        'error': str(e)
                    })
                    yield f"Error processing page {i+1}: {e}"
            
            # Save results
            full_text = '\n\n'.join(all_text)
            
            # Save extracted text
            with open(output_paths['text_file'], 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            # Save detailed results
            result_data = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'total_pages': len(image_paths),
                'total_text_length': len(full_text),
                'pages': page_results,
                'full_text': full_text,
                'ocr_provider': self.ocr_provider.get_provider_name(),
                'processing_mode': 'streaming',
                'status': 'success'
            }
            
            with open(output_paths['result_file'], 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Streaming processing complete for {file_path.name}: {len(full_text)} characters")
            
        except Exception as e:
            logger.error(f"Error in streaming file processing for {file_path}: {e}")
            yield f"Error processing file {file_path.name}: {e}"
    
    def process_single_file_api_client(self, file_path: str) -> Dict:
        """Process a single file using API Client approach (non-streaming)."""
        file_path = Path(file_path)
        logger.info(f"Processing file with API Client: {file_path.name}")
        
        try:
            # Create structured output paths
            output_paths = self.create_structured_output_paths(str(file_path))
            
            # Create a temporary file processor for this specific file
            temp_file_processor = FileProcessor(output_paths['images_dir'])
            
            # Convert file to images
            image_paths = temp_file_processor.process_file(str(file_path))
            
            if not image_paths:
                return {
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'status': 'failed',
                    'error': 'Unsupported file type or conversion failed',
                    'total_pages': 0,
                    'total_text_length': 0,
                    'pages': [],
                    'full_text': '',
                    'ocr_provider': self.ocr_provider.get_provider_name(),
                    'processing_mode': 'api_client'
                }
            
            # Process each image with API Client OCR
            all_text = []
            page_results = []
            
            for i, image_path in enumerate(image_paths):
                try:
                    logger.info(f"Processing page {i+1}/{len(image_paths)}: {Path(image_path).name}")
                    
                    # Check if provider supports API Client approach
                    if hasattr(self.ocr_provider, 'process_image_api_client'):
                        # Use API Client OCR
                        result = self.ocr_provider.process_image_api_client(image_path)
                        page_text = result.get('extracted_text', '')
                        all_text.append(page_text)
                        page_results.append({
                            'page_number': i + 1,
                            'image_path': image_path,
                            'extracted_text': page_text,
                            'text_length': len(page_text),
                            'status': 'success',
                            'metadata': result.get('metadata', {})
                        })
                    else:
                        # Fallback to regular OCR
                        result = self.ocr_provider.process_image(image_path)
                        page_text = result.get('extracted_text', '')
                        all_text.append(page_text)
                        page_results.append({
                            'page_number': i + 1,
                            'image_path': image_path,
                            'extracted_text': page_text,
                            'text_length': len(page_text),
                            'status': 'success'
                        })
                
                except Exception as e:
                    logger.error(f"Error processing page {i+1}: {e}")
                    page_results.append({
                        'page_number': i + 1,
                        'image_path': image_path,
                        'extracted_text': '',
                        'text_length': 0,
                        'status': 'failed',
                        'error': str(e)
                    })
            
            # Save results
            full_text = '\n\n'.join(all_text)
            
            # Save extracted text
            with open(output_paths['text_file'], 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            # Save detailed results
            result_data = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'total_pages': len(image_paths),
                'total_text_length': len(full_text),
                'pages': page_results,
                'full_text': full_text,
                'ocr_provider': self.ocr_provider.get_provider_name(),
                'processing_mode': 'api_client',
                'status': 'success'
            }
            
            with open(output_paths['result_file'], 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"API Client processing complete for {file_path.name}: {len(full_text)} characters")
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error in API Client file processing for {file_path}: {e}")
            return {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'status': 'failed',
                'error': str(e),
                'total_pages': 0,
                'total_text_length': 0,
                'pages': [],
                'full_text': '',
                'ocr_provider': self.ocr_provider.get_provider_name(),
                'processing_mode': 'api_client'
            }
    
    def create_structured_output_paths(self, file_path: str) -> Dict[str, Path]:
        """Create structured output paths based on source file path."""
        file_path = Path(file_path)
        
        # Create directory structure based on source path
        # Example: /path/to/data/nii/01. 개인정보 이미지 파일 -> nii/01. 개인정보 이미지 파일
        source_parts = file_path.parts
        
        # Find the base directory (usually 'data' or similar)
        base_index = -1
        for i, part in enumerate(source_parts):
            if part in ['data', 'documents', 'files', 'images']:
                base_index = i
                break
        
        if base_index >= 0 and base_index < len(source_parts) - 1:
            # Create structure: provider_ocr/category/filename/
            category = source_parts[base_index + 1] if base_index + 1 < len(source_parts) else "misc"
            filename = file_path.stem
            
            # Create the structured directory
            structured_dir = self.provider_output_dir / category / filename
            structured_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            images_dir = structured_dir / "converted_images"
            images_dir.mkdir(exist_ok=True)
            
            return {
                'structured_dir': structured_dir,
                'images_dir': images_dir,
                'result_file': structured_dir / f"{filename}_ocr_result.json",
                'text_file': structured_dir / f"{filename}_extracted_text.txt"
            }
        else:
            # Fallback: use filename directly
            filename = file_path.stem
            structured_dir = self.provider_output_dir / filename
            structured_dir.mkdir(parents=True, exist_ok=True)
            
            images_dir = structured_dir / "converted_images"
            images_dir.mkdir(exist_ok=True)
            
            return {
                'structured_dir': structured_dir,
                'images_dir': images_dir,
                'result_file': structured_dir / f"{filename}_ocr_result.json",
                'text_file': structured_dir / f"{filename}_extracted_text.txt"
            }
    
    def find_files(self, directory: str, recursive: bool = True) -> List[str]:
        """Find all supported files in directory."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        supported_extensions = {
            # Documents
            '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xls', '.xlsx',
            # Images
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff'
        }
        
        files = []
        if recursive:
            for ext in supported_extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in supported_extensions:
                files.extend(directory.glob(f"*{ext}"))
        
        return [str(f) for f in files]
    
    def process_single_file(self, file_path: str) -> Dict:
        """Process a single file."""
        file_path = Path(file_path)
        logger.info(f"Processing file: {file_path.name}")
        
        try:
            # Create structured output paths
            output_paths = self.create_structured_output_paths(str(file_path))
            
            # Create a temporary file processor for this specific file
            temp_file_processor = FileProcessor(output_paths['images_dir'])
            
            # Convert file to images
            image_paths = temp_file_processor.process_file(str(file_path))
            
            if not image_paths:
                return {
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'status': 'skipped',
                    'reason': 'Unsupported file type or conversion failed'
                }
            
            # Process each image with OCR
            all_text = []
            page_results = []
            
            for i, image_path in enumerate(image_paths):
                try:
                    if image_path.endswith('.txt'):
                        # Direct text file (from DOCX)
                        with open(image_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        page_result = {
                            'page_number': i + 1,
                            'extracted_text': text,
                            'text_length': len(text),
                            'method': 'direct_extraction'
                        }
                    else:
                        # OCR processing
                        ocr_result = self.ocr_provider.process_image(image_path)
                        page_result = {
                            'page_number': i + 1,
                            'image_path': image_path,
                            'extracted_text': ocr_result.get('extracted_text', ''),
                            'text_length': ocr_result.get('text_length', 0),
                            'method': 'ocr'
                        }
                    
                    page_results.append(page_result)
                    all_text.append(page_result['extracted_text'])
                    
                except Exception as e:
                    logger.error(f"Error processing page {i + 1} of {file_path.name}: {e}")
                    page_results.append({
                        'page_number': i + 1,
                        'error': str(e),
                        'extracted_text': '',
                        'text_length': 0
                    })
            
            # Combine all text
            full_text = '\n\n--- PAGE BREAK ---\n\n'.join(all_text)
            
            result = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'file_type': file_path.suffix.lower(),
                'ocr_provider': self.provider_name,
                'total_pages': len(image_paths),
                'processing_timestamp': datetime.now().isoformat(),
                'output_directory': str(output_paths['structured_dir']),
                'pages': page_results,
                'full_text': full_text,
                'total_text_length': len(full_text),
                'status': 'success'
            }
            
            # Save individual file results in structured directory
            with open(output_paths['result_file'], 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Save extracted text in structured directory
            with open(output_paths['text_file'], 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            logger.info(f"Successfully processed {file_path.name} - {len(full_text)} characters")
            logger.info(f"Results saved to: {output_paths['structured_dir']}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'status': 'failed',
                'error': str(e)
            }
    
    def process_directory(self, directory: str, recursive: bool = True) -> List[Dict]:
        """Process all files in a directory."""
        logger.info(f"Processing directory: {directory}")
        
        files = self.find_files(directory, recursive)
        if not files:
            logger.warning(f"No supported files found in {directory}")
            return []
        
        logger.info(f"Found {len(files)} files to process")
        
        results = []
        for i, file_path in enumerate(files, 1):
            logger.info(f"Processing file {i}/{len(files)}: {Path(file_path).name}")
            result = self.process_single_file(file_path)
            results.append(result)
        
        # Save batch results in provider-specific directory
        batch_file = self.provider_output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Generate summary
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'failed']
        
        summary = {
            'total_files': len(files),
            'successful': len(successful),
            'failed': len(failed),
            'total_text_length': sum(r.get('total_text_length', 0) for r in successful),
            'ocr_provider': self.provider_name,
            'processing_timestamp': datetime.now().isoformat(),
            'results_file': str(batch_file),
            'provider_output_directory': str(self.provider_output_dir)
        }
        
        summary_file = self.provider_output_dir / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing complete: {len(successful)} successful, {len(failed)} failed")
        logger.info(f"Results saved to: {self.provider_output_dir}")
        return results

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Universal OCR Processor")
    parser.add_argument("path", help="Path to file or directory to process")
    parser.add_argument("--provider", "-p", choices=["google_cloud", "mistral", "naver", "alibaba", "deepseek", "paddleocr_vl"], 
                       default="mistral", help="OCR provider to use")
    parser.add_argument("--output", "-o", default="universal_ocr_results", 
                       help="Output directory for results")
    parser.add_argument("--recursive", "-r", action="store_true", 
                       help="Process subdirectories recursively")
    parser.add_argument("--single-file", "-f", action="store_true", 
                       help="Process as single file (not directory)")
    parser.add_argument("--model", "-m", 
                       help="Model name for providers that support multiple models (e.g., alibaba: qwen-vl-ocr, qwen-vl-plus, qwen-vl-30b, qwen-vl-235b; deepseek: tiny, small, base, large, gundam; paddleocr_vl: ocr, table, chart, formula)")
    parser.add_argument("--stream", "-s", action="store_true",
                       help="Enable streaming output (real-time processing)")
    
    args = parser.parse_args()
    
    print("🚀 Universal OCR Processor")
    print("=" * 50)
    print(f"Provider: {args.provider}")
    if args.model:
        print(f"Model: {args.model}")
    print(f"Path: {args.path}")
    print(f"Output: {args.output}")
    print(f"Recursive: {args.recursive}")
    if args.stream:
        print(f"Streaming: Enabled")
    print("=" * 50)
    
    try:
        processor = UniversalOCRProcessor(args.provider, args.output, args.model)
        
        if args.single_file or Path(args.path).is_file():
            # Process single file
            if args.stream:
                # Streaming processing
                print(f"\n🔄 Streaming Processing Started!")
                print(f"File: {Path(args.path).name}")
                print("=" * 50)
                print("Streaming output:")
                print("-" * 30)
                
                full_content = ""
                for chunk in processor.process_single_file_streaming(args.path):
                    print(chunk, end='', flush=True)
                    full_content += chunk
                
                print("\n" + "-" * 30)
                print(f"\n✅ Streaming Processing Complete!")
                print(f"Total characters: {len(full_content)}")
            else:
                # Regular processing
                result = processor.process_single_file(args.path)
                print(f"\n✅ Processing Complete!")
                print(f"File: {result['file_name']}")
                print(f"Status: {result['status']}")
                if result['status'] == 'success':
                    print(f"Text Length: {result['total_text_length']} characters")
                    print(f"Pages: {result['total_pages']}")
        else:
            # Process directory
            results = processor.process_directory(args.path, args.recursive)
            
            successful = [r for r in results if r.get('status') == 'success']
            failed = [r for r in results if r.get('status') == 'failed']
            
            print(f"\n🎉 Batch Processing Complete!")
            print(f"Total files: {len(results)}")
            print(f"Successful: {len(successful)}")
            print(f"Failed: {len(failed)}")
            print(f"Total text extracted: {sum(r.get('total_text_length', 0) for r in successful)} characters")
            print(f"Results saved to: {processor.provider_output_dir}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main processing error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
