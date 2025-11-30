#!/usr/bin/env python3
"""
Mistral AI OCR Provider
"""

import os
import base64
import logging
from pathlib import Path
from typing import Dict
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables
env_paths = [
    Path(__file__).parent.parent / ".env",  # API directory
    Path(__file__).parent.parent.parent / "OCR" / "google_vision" / ".env",  # OCR directory
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break

logger = logging.getLogger(__name__)

class MistralOCRProvider:
    """Mistral AI OCR provider."""
    
    def __init__(self):
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set")
        
        self.client = Mistral(api_key=api_key)
        self.model_name = os.getenv('MISTRAL_MODEL', 'mistral-ocr-latest')
    
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
                model=self.model_name,
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
                logger.warning("No text found in Mistral OCR response")
            
            return {
                'extracted_text': extracted_text,
                'metadata': {
                    'provider': 'mistral_ai',
                    'model': self.model_name,
                    'confidence': 0.8,  # Mistral doesn't provide confidence scores
                    'processing_time': None
                }
            }
            
        except Exception as e:
            logger.error(f"Mistral AI processing error: {e}")
            return {
                'extracted_text': '',
                'metadata': {
                    'provider': 'mistral_ai',
                    'error': str(e),
                    'confidence': 0.0
                }
            }
    
    def get_provider_name(self) -> str:
        """Return the name of the OCR provider."""
        return f"Mistral AI ({self.model_name})"
