#!/usr/bin/env python3
"""
Simple Mistral OCR Implementation for PDF Documents
A streamlined version for easy OCR processing using Mistral API
"""

import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime
import base64
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleMistralOCR:
    """
    Simple OCR solution using Mistral API for PDF documents.
    """
    
    def __init__(self, api_key: str = None, output_dir: str = "mistral_ocr_results"):
        """
        Initialize Mistral OCR processor.
        
        Args:
            api_key: Mistral API key (if not provided, will try to get from environment)
            output_dir: Directory to save OCR results
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY in .env file or pass as parameter.")
        
        # Initialize Mistral client
        self.client = Mistral(api_key=self.api_key)
        
        # Setup output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("Mistral OCR processor initialized successfully")
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Convert PDF pages to PNG images for OCR processing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of paths to converted image files
        """
        pdf_name = Path(pdf_path).stem
        images_dir = self.output_dir / "converted_images"
        images_dir.mkdir(exist_ok=True)
        
        image_paths = []
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            logger.info(f"Converting PDF: {pdf_path} ({len(doc)} pages)")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Create high-resolution image (2x zoom for better OCR)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                
                # Save as PNG
                image_path = images_dir / f"{pdf_name}_page_{page_num + 1:03d}.png"
                pix.save(str(image_path))
                image_paths.append(str(image_path))
                
                logger.info(f"Converted page {page_num + 1} to {image_path}")
            
            doc.close()
            logger.info(f"Successfully converted {len(image_paths)} pages")
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
            
        return image_paths
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for Mistral API."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise
    
    def ocr_image(self, image_path: str) -> Dict:
        """
        Perform OCR on a single image using Mistral API.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing OCR results
        """
        try:
            # Encode image to base64
            base64_image = self.encode_image(image_path)
            
            logger.info(f"Processing image with Mistral OCR: {image_path}")
            
            # Use Mistral's OCR endpoint
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64_image}" 
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
    
    def process_pdf_document(self, pdf_path: str, save_images: bool = True) -> Dict:
        """
        Complete workflow: Convert PDF to images and perform OCR on each page.
        
        Args:
            pdf_path: Path to the PDF file
            save_images: Whether to save converted images
            
        Returns:
            Dictionary containing complete OCR results
        """
        pdf_name = Path(pdf_path).stem
        logger.info(f"Starting Mistral OCR processing for document: {pdf_name}")
        
        # Convert PDF to images
        image_paths = self.convert_pdf_to_images(pdf_path)
        
        # Process each image with OCR
        document_results = {
            'document_name': pdf_name,
            'total_pages': len(image_paths),
            'processing_timestamp': datetime.now().isoformat(),
            'provider': 'mistral',
            'pages': []
        }
        
        all_text = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing page {i + 1}/{len(image_paths)}")
            
            try:
                # Perform OCR
                page_result = self.ocr_image(image_path)
                page_result['page_number'] = i + 1
                
                document_results['pages'].append(page_result)
                all_text.append(page_result.get('extracted_text', ''))
                
                # Save individual page results
                page_output_path = self.output_dir / f"{pdf_name}_page_{i + 1:03d}_mistral_ocr.json"
                with open(page_output_path, 'w', encoding='utf-8') as f:
                    json.dump(page_result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved page {i + 1} results to {page_output_path}")
                
            except Exception as e:
                logger.error(f"Error processing page {i + 1}: {e}")
                document_results['pages'].append({
                    'page_number': i + 1,
                    'image_path': image_path,
                    'error': str(e),
                    'extracted_text': '',
                    'text_length': 0
                })
        
        # Combine all text
        document_results['full_text'] = '\n\n--- PAGE BREAK ---\n\n'.join(all_text)
        document_results['total_text_length'] = len(document_results['full_text'])
        
        # Save complete document results
        doc_output_path = self.output_dir / f"{pdf_name}_complete_mistral_ocr.json"
        with open(doc_output_path, 'w', encoding='utf-8') as f:
            json.dump(document_results, f, indent=2, ensure_ascii=False)
        
        # Save full text as separate file
        text_output_path = self.output_dir / f"{pdf_name}_mistral_extracted_text.txt"
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(document_results['full_text'])
        
        logger.info(f"Complete Mistral OCR results saved to {doc_output_path}")
        logger.info(f"Extracted text saved to {text_output_path}")
        
        return document_results

def main():
    """Example usage of SimpleMistralOCR."""
    print("🚀 Simple Mistral OCR Processor")
    print("=" * 50)
    
    try:
        # Initialize Mistral OCR processor
        ocr_processor = SimpleMistralOCR()
        
        # Get PDF path from user
        pdf_path = input("Enter the path to your PDF document: ").strip()
        
        if not os.path.exists(pdf_path):
            print(f"❌ Error: File {pdf_path} not found!")
            return
        
        # Process the document
        print(f"📄 Processing PDF: {pdf_path}")
        results = ocr_processor.process_pdf_document(pdf_path)
        
        print(f"\n✅ Mistral OCR Processing Complete!")
        print(f"📄 Document: {results['document_name']}")
        print(f"📖 Total Pages: {results['total_pages']}")
        print(f"📝 Total Text Length: {results['total_text_length']} characters")
        print(f"💾 Results saved to: {ocr_processor.output_dir}")
        
        # Show sample text from first page
        if results['pages'] and results['pages'][0].get('extracted_text'):
            sample_text = results['pages'][0]['extracted_text'][:200]
            print(f"\n📋 Sample Text (first 200 chars from page 1):")
            print(sample_text + "..." if len(sample_text) == 200 else sample_text)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main processing error: {e}")

if __name__ == "__main__":
    main()
