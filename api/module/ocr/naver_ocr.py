#!/usr/bin/env python3
"""
Naver Clova OCR Provider
"""

import os
import base64
import requests
import time
import logging
from pathlib import Path
from typing import Dict
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

class NaverOCRProvider:
    """Naver Clova OCR provider."""
    
    def __init__(self):
        self.api_url = os.getenv('NAVER_OCR_API_URL')
        self.secret_key = os.getenv('NAVER_OCR_SECRET_KEY')
        
        if not self.api_url or not self.secret_key:
            raise ValueError("NAVER_OCR_API_URL and NAVER_OCR_SECRET_KEY environment variables must be set")
        
        self.template_ids = os.getenv('NAVER_OCR_TEMPLATE_IDS', '').split(',') if os.getenv('NAVER_OCR_TEMPLATE_IDS') else []
    
    def process_image(self, image_path: str) -> Dict:
        """Process an image using Naver Clova OCR."""
        try:
            # Read and encode image
            with open(image_path, 'rb') as image_file:
                file_data = base64.b64encode(image_file.read()).decode()
            
            # Prepare request data
            request_json = {
                'images': [{
                    'format': image_path.split('.')[-1].lower(),
                    'name': os.path.basename(image_path),
                    'data': file_data
                }],
                'requestId': os.path.splitext(os.path.basename(image_path))[0],
                'version': 'V2',
                'timestamp': int(time.time() * 1000)
            }
            
            if self.template_ids:
                request_json['templates'] = [{'id': tid} for tid in self.template_ids]
            
            # Make API call
            headers = {
                'X-OCR-SECRET': self.secret_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(self.api_url, headers=headers, json=request_json)
            response.raise_for_status()
            
            result_data = response.json()
            
            # Extract text with coordinate-based alignment
            extracted_text = ""
            if 'images' in result_data:
                for image in result_data['images']:
                    if 'fields' in image:
                        # Sort text fields by coordinates (Y first, then X)
                        fields_with_coords = []
                        for field in image['fields']:
                            if 'boundingPoly' in field and 'vertices' in field['boundingPoly']:
                                vertices = field['boundingPoly']['vertices']
                                if vertices:
                                    y_coord = vertices[0].get('y', 0)
                                    x_coord = vertices[0].get('x', 0)
                                    text = field.get('inferText', '').strip()
                                    if text:
                                        fields_with_coords.append((y_coord, x_coord, text))
                        
                        # Sort by coordinates
                        fields_with_coords.sort(key=lambda x: (x[0], x[1]))
                        
                        # Group by lines and construct text
                        if fields_with_coords:
                            current_line_y = fields_with_coords[0][0]
                            line_threshold = 20  # Same line threshold
                            current_line_texts = []
                            
                            for y, x, text in fields_with_coords:
                                if abs(y - current_line_y) <= line_threshold:
                                    current_line_texts.append(text)
                                else:
                                    if current_line_texts:
                                        extracted_text += " ".join(current_line_texts) + "\n"
                                    current_line_texts = [text]
                                    current_line_y = y
                            
                            # Last line
                            if current_line_texts:
                                extracted_text += " ".join(current_line_texts) + "\n"
            
            return {
                'extracted_text': extracted_text,
                'metadata': {
                    'provider': 'naver_clova',
                    'confidence': self._calculate_confidence(result_data),
                    'raw_response': result_data,
                    'processing_time': None
                }
            }
            
        except Exception as e:
            logger.error(f"Naver Clova OCR processing error: {e}")
            return {
                'extracted_text': '',
                'metadata': {
                    'provider': 'naver_clova',
                    'error': str(e),
                    'confidence': 0.0
                }
            }
    
    def _calculate_confidence(self, result_data: Dict) -> float:
        """Calculate average confidence from Naver OCR response."""
        try:
            if 'images' not in result_data:
                return 0.0
            
            total_confidence = 0.0
            total_fields = 0
            
            for image in result_data['images']:
                if 'fields' in image:
                    for field in image['fields']:
                        confidence = field.get('inferConfidence', 0.0)
                        total_confidence += confidence
                        total_fields += 1
            
            return total_confidence / total_fields if total_fields > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.0
    
    def get_provider_name(self) -> str:
        """Return the name of the OCR provider."""
        return "Naver Clova OCR"
