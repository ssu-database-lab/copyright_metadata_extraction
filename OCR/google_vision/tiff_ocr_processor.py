#!/usr/bin/env python3
"""
TIFF Image OCR Processor
Processes TIFF images using Google Cloud Vision API or Mistral OCR
Supports both single files and batch processing
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime
import base64
from google.cloud import vision
from google.protobuf.json_format import MessageToDict
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TIFFOCRProcessor:
    """
    OCR processor specifically designed for TIFF images.
    Supports both Google Cloud Vision API and Mistral OCR.
    """
    
    def __init__(self, provider: str = "google_cloud", output_dir: str = "tiff_ocr_results"):
        """
        Initialize TIFF OCR processor.
        
        Args:
            provider: OCR provider ("google_cloud" or "mistral")
            output_dir: Directory to save OCR results
        """
        self.provider = provider.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize OCR provider
        if self.provider == "google_cloud":
            self._init_google_cloud()
        elif self.provider == "mistral":
            self._init_mistral()
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'google_cloud' or 'mistral'")
        
        logger.info(f"TIFF OCR processor initialized with {self.provider} provider")
    
    def _init_google_cloud(self):
        """Initialize Google Cloud Vision API client."""
        # Force IPv4 for gRPC (fixes WSL2 IPv6 issues)
        os.environ["GRPC_DNS_RESOLVER"] = "native"
        
        # Set up Google Cloud credentials
        script_dir = os.path.dirname(os.path.abspath(__file__))
        credentials_path = os.path.join(script_dir, "semiotic-pager-466612-t0-c587b9296fb8.json")
        
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"Google Cloud credentials file not found at: {credentials_path}")
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        self.client = vision.ImageAnnotatorClient()
        logger.info(f"Google Cloud Vision API initialized")
    
    def _init_mistral(self):
        """Initialize Mistral API client."""
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY in .env file.")
        
        self.client = Mistral(api_key=api_key)
        logger.info("Mistral API initialized")
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API calls."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise
    
    def ocr_image_google_cloud(self, image_path: str) -> Dict:
        """Perform OCR using Google Cloud Vision API."""
        try:
            # Read image file
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Use document text detection for better results with scanned documents
            response = self.client.document_text_detection(image=image)
            
            # Check for errors
            if response.error.message:
                raise RuntimeError(f"Vision API Error: {response.error.message}")
            
            # Convert response to dictionary
            response_dict = MessageToDict(response._pb)
            
            # Extract text content
            if response.text_annotations:
                extracted_text = response.text_annotations[0].description
                response_dict['extracted_text'] = extracted_text
                response_dict['text_length'] = len(extracted_text)
            else:
                response_dict['extracted_text'] = ""
                response_dict['text_length'] = 0
            
            return response_dict
            
        except Exception as e:
            logger.error(f"Error processing image {image_path} with Google Cloud Vision: {e}")
            raise
    
    def ocr_image_mistral(self, image_path: str) -> Dict:
        """Perform OCR using Mistral API."""
        try:
            # Encode image to base64
            base64_image = self.encode_image(image_path)
            
            logger.info(f"Processing image with Mistral OCR: {image_path}")
            
            # Use Mistral's OCR endpoint
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/tiff;base64,{base64_image}" 
                },
                include_image_base64=True
            )
            
            # Extract text from response
            extracted_text = ""
            confidence = 1.0
            
            if hasattr(ocr_response, 'pages') and ocr_response.pages:
                # Mistral OCR returns pages with markdown content
                text_parts = []
                for page in ocr_response.pages:
                    if hasattr(page, 'markdown') and page.markdown:
                        text_parts.append(page.markdown)
                    elif hasattr(page, 'text') and page.text:
                        text_parts.append(page.text)
                extracted_text = "\n\n".join(text_parts)
            elif hasattr(ocr_response, 'text') and ocr_response.text:
                extracted_text = ocr_response.text
            elif hasattr(ocr_response, 'content') and ocr_response.content:
                extracted_text = ocr_response.content
            else:
                # Fallback: convert response to string
                extracted_text = str(ocr_response)
            
            # Create response dictionary
            response_dict = {
                'extracted_text': extracted_text,
                'text_length': len(extracted_text),
                'confidence': confidence,
                'provider': 'mistral',
                'model': 'mistral-ocr-latest',
                'processing_timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'raw_response': str(ocr_response) if hasattr(ocr_response, '__dict__') else None
            }
            
            logger.info(f"Mistral OCR completed - extracted {len(extracted_text)} characters")
            return response_dict
            
        except Exception as e:
            logger.error(f"Error processing image {image_path} with Mistral OCR: {e}")
            raise
    
    def ocr_image(self, image_path: str) -> Dict:
        """
        Perform OCR on a TIFF image using the configured provider.
        
        Args:
            image_path: Path to the TIFF image file
            
        Returns:
            Dictionary containing OCR results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not image_path.lower().endswith(('.tif', '.tiff')):
            logger.warning(f"File {image_path} may not be a TIFF image")
        
        logger.info(f"Processing TIFF image: {image_path}")
        
        if self.provider == "google_cloud":
            return self.ocr_image_google_cloud(image_path)
        elif self.provider == "mistral":
            return self.ocr_image_mistral(image_path)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def process_single_tiff(self, tiff_path: str, save_results: bool = True) -> Dict:
        """
        Process a single TIFF image.
        
        Args:
            tiff_path: Path to the TIFF file
            save_results: Whether to save results to files
            
        Returns:
            Dictionary containing OCR results
        """
        tiff_name = Path(tiff_path).stem
        logger.info(f"Processing single TIFF: {tiff_name}")
        
        try:
            # Perform OCR
            result = self.ocr_image(tiff_path)
            result['image_name'] = tiff_name
            result['image_path'] = tiff_path
            result['processing_timestamp'] = datetime.now().isoformat()
            
            # Save results if requested
            if save_results:
                # Save individual result
                result_path = self.output_dir / f"{tiff_name}_{self.provider}_ocr.json"
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                # Save extracted text
                text_path = self.output_dir / f"{tiff_name}_{self.provider}_extracted_text.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(result.get('extracted_text', ''))
                
                logger.info(f"Results saved to {result_path} and {text_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing TIFF {tiff_path}: {e}")
            raise
    
    def process_tiff_directory(self, directory_path: str, pattern: str = "*.tif*") -> List[Dict]:
        """
        Process all TIFF files in a directory.
        
        Args:
            directory_path: Directory containing TIFF files
            pattern: File pattern to match (default: *.tif*)
            
        Returns:
            List of results for each processed file
        """
        tiff_dir = Path(directory_path)
        tiff_files = list(tiff_dir.glob(pattern))
        
        if not tiff_files:
            logger.warning(f"No TIFF files found in {directory_path} matching pattern {pattern}")
            return []
        
        logger.info(f"Found {len(tiff_files)} TIFF files to process")
        
        results = []
        for tiff_file in tiff_files:
            try:
                logger.info(f"Processing {tiff_file.name}")
                result = self.process_single_tiff(str(tiff_file))
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {tiff_file.name}: {e}")
                results.append({
                    'image_name': tiff_file.stem,
                    'image_path': str(tiff_file),
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Save batch results summary
        batch_summary_path = self.output_dir / f"batch_tiff_ocr_{self.provider}_summary.json"
        with open(batch_summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_files': len(tiff_files),
                'successful': len([r for r in results if 'error' not in r]),
                'failed': len([r for r in results if 'error' in r]),
                'processing_timestamp': datetime.now().isoformat(),
                'provider': self.provider,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing complete. Summary saved to {batch_summary_path}")
        return results

def main():
    """Example usage of TIFFOCRProcessor."""
    print("🚀 TIFF Image OCR Processor")
    print("=" * 50)
    
    # Get user preferences
    provider = input("Choose OCR provider (google_cloud/mistral): ").strip().lower()
    if provider not in ["google_cloud", "mistral"]:
        print("Invalid provider. Using Google Cloud Vision.")
        provider = "google_cloud"
    
    try:
        # Initialize processor
        processor = TIFFOCRProcessor(provider=provider)
        
        # Get processing mode
        print("\n📋 Processing Options:")
        print("1. Process single TIFF file")
        print("2. Process all TIFF files in directory")
        
        choice = input("Select option (1-2): ").strip()
        
        if choice == "1":
            # Single file processing
            tiff_path = input("Enter path to TIFF file: ").strip()
            if not os.path.exists(tiff_path):
                print(f"❌ Error: File {tiff_path} not found!")
                return
            
            result = processor.process_single_tiff(tiff_path)
            
            print(f"\n✅ TIFF OCR Processing Complete!")
            print(f"📄 Image: {result['image_name']}")
            print(f"📝 Text Length: {result.get('text_length', 0)} characters")
            print(f"💾 Results saved to: {processor.output_dir}")
            
            # Show sample text
            if result.get('extracted_text'):
                sample_text = result['extracted_text'][:200]
                print(f"\n📋 Sample Text (first 200 chars):")
                print(sample_text + "..." if len(sample_text) == 200 else sample_text)
        
        elif choice == "2":
            # Directory processing
            directory_path = input("Enter directory path containing TIFF files: ").strip()
            if not os.path.exists(directory_path):
                print(f"❌ Error: Directory {directory_path} not found!")
                return
            
            results = processor.process_tiff_directory(directory_path)
            
            print(f"\n✅ Batch TIFF OCR Processing Complete!")
            print(f"📊 Processed {len(results)} files")
            
            successful = [r for r in results if 'error' not in r]
            failed = [r for r in results if 'error' in r]
            
            print(f"✅ Successful: {len(successful)}")
            print(f"❌ Failed: {len(failed)}")
            
            if failed:
                print("\nFailed files:")
                for result in failed:
                    print(f"  - {result['image_name']}: {result['error']}")
        
        else:
            print("❌ Invalid choice.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main processing error: {e}")

if __name__ == "__main__":
    main()
