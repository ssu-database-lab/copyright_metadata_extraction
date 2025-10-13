#!/usr/bin/env python3
"""
Google Cloud Vision OCR Provider
"""

import os
import logging
from pathlib import Path
from typing import Dict
from google.cloud import vision
from google.protobuf.json_format import MessageToDict
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

class GoogleCloudOCRProvider:
    """Google Cloud Vision API OCR provider."""
    
    def __init__(self):
        # Set up Google Cloud credentials
        script_dir = os.path.dirname(os.path.abspath(__file__))
        credentials_path = os.path.join(script_dir, "..", "..", "..", "OCR", "google_vision", "semiotic-pager-466612-t0-c587b9296fb8.json")
        
        if os.path.exists(credentials_path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            logger.info(f"Using Google Cloud credentials from: {credentials_path}")
        else:
            logger.warning("Google Cloud credentials not found. Google OCR will not be available.")
        
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
            
            # Extract text from response
            extracted_text = response.full_text_annotation.text if response.full_text_annotation else ""
            
            # Convert response to dict for metadata
            response_dict = MessageToDict(response._pb)
            
            return {
                'extracted_text': extracted_text,
                'metadata': {
                    'provider': 'google_cloud_vision',
                    'confidence': self._calculate_confidence(response_dict),
                    'text_annotations': response_dict.get('fullTextAnnotation', {}),
                    'processing_time': None
                }
            }
            
        except Exception as e:
            logger.error(f"Google Vision API processing error: {e}")
            return {
                'extracted_text': '',
                'metadata': {
                    'provider': 'google_cloud_vision',
                    'error': str(e),
                    'confidence': 0.0
                }
            }
    
    def _calculate_confidence(self, response_dict: Dict) -> float:
        """Calculate average confidence from Google Vision response."""
        try:
            full_text_annotation = response_dict.get('fullTextAnnotation', {})
            pages = full_text_annotation.get('pages', [])
            
            if not pages:
                return 0.0
            
            total_confidence = 0.0
            total_blocks = 0
            
            for page in pages:
                blocks = page.get('blocks', [])
                for block in blocks:
                    paragraphs = block.get('paragraphs', [])
                    for paragraph in paragraphs:
                        words = paragraph.get('words', [])
                        for word in words:
                            symbols = word.get('symbols', [])
                            for symbol in symbols:
                                confidence = symbol.get('confidence', 0.0)
                                total_confidence += confidence
                                total_blocks += 1
            
            return total_confidence / total_blocks if total_blocks > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.0
    
    def get_provider_name(self) -> str:
        """Return the name of the OCR provider."""
        return "Google Cloud Vision"
