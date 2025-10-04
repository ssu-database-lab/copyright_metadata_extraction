#!/usr/bin/env python3
"""
Enhanced Batch PDF OCR Processor for Categorized Documents
Processes PDFs from multiple directories and organizes results by category.
"""

import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from google.cloud import vision
from google.protobuf.json_format import MessageToDict
import logging
import shutil
from datetime import datetime
import time
import requests
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import base64
from mistralai import Mistral

# Load environment variables from .env file
load_dotenv()

# Configure logging

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = log_dir / f"batch_ocr_{timestamp}.log"

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # Console output
    ]
)
logger = logging.getLogger(__name__)

# Log the start of the session
logger.info(f"Starting batch OCR processing session")
logger.info(f"Log file: {log_filename}")
logger.info(f"Working directory: {os.getcwd()}")

# Force IPv4 for gRPC (fixes WSL2 IPv6 issues)
os.environ["GRPC_DNS_RESOLVER"] = "native"

# Set up Google Cloud credentials
script_dir = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(script_dir, "semiotic-pager-466612-t0-c587b9296fb8.json")

# Check if credentials file exists
if not os.path.exists(credentials_path):
    raise FileNotFoundError(f"Google Cloud credentials file not found at: {credentials_path}\n"
                           f"Please ensure the credentials file is in the same directory as this script.")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
logger.info(f"Using Google Cloud credentials from: {credentials_path}")

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
            # Read image file
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Use document text detection for better results with scanned documents
            response = self.client.document_text_detection(image=image)
            logger.info(f"Using Google Cloud Vision API for {image_path}")
            
            # Check for errors
            if response.error.message:
                raise RuntimeError(f"Google Vision API Error: {response.error.message}")
            
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
    
    def get_provider_name(self) -> str:
        return "google_cloud"

class MistralOCRProvider(OCRProvider):
    """Mistral OCR API provider using official SDK."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = Mistral(api_key=api_key)
    
    def encode_image(self, image_path: str) -> str:
        """Encode the image to base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            logger.error(f"Error: The file {image_path} was not found.")
            raise
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise
    
    def process_image(self, image_path: str) -> Dict:
        """Process an image using Mistral OCR API."""
        try:
            # Encode image to base64
            base64_image = self.encode_image(image_path)
            
            # Use Mistral's dedicated OCR endpoint
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64_image}" 
                },
                include_image_base64=True
            )
            
            # Debug: Log response structure
            logger.debug(f"Mistral OCR response type: {type(ocr_response)}")
            logger.debug(f"Mistral OCR response attributes: {[attr for attr in dir(ocr_response) if not attr.startswith('_')]}")
            
            # Extract text from OCR response - Mistral returns pages with markdown content
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
            
            # Create a serializable raw response
            try:
                # Extract serializable data from the response
                raw_response = {
                    'response_type': type(ocr_response).__name__,
                    'text': extracted_text,
                    'confidence': confidence,
                    'has_pages': hasattr(ocr_response, 'pages'),
                    'page_count': len(ocr_response.pages) if hasattr(ocr_response, 'pages') else 1,
                    'model': getattr(ocr_response, 'model', 'unknown')
                }
                
                # Handle usage_info safely
                usage_info = getattr(ocr_response, 'usage_info', None)
                if usage_info:
                    try:
                        # Try to extract serializable data from usage_info
                        raw_response['usage_info'] = {
                            'type': type(usage_info).__name__,
                            'attributes': [attr for attr in dir(usage_info) if not attr.startswith('_')]
                        }
                        # Try to get common usage fields
                        for attr in ['input_tokens', 'output_tokens', 'total_tokens', 'prompt_tokens', 'completion_tokens']:
                            if hasattr(usage_info, attr):
                                raw_response['usage_info'][attr] = getattr(usage_info, attr)
                    except Exception:
                        raw_response['usage_info'] = str(usage_info)
                else:
                    raw_response['usage_info'] = None
                
                # Add page information if available
                if hasattr(ocr_response, 'pages') and ocr_response.pages:
                    page_info = []
                    for i, page in enumerate(ocr_response.pages):
                        page_data = {
                            'index': getattr(page, 'index', i),
                            'text_length': len(getattr(page, 'markdown', '')),
                            'has_images': len(getattr(page, 'images', [])) > 0,
                            'dimensions': {}
                        }
                        
                        # Add dimensions if available
                        if hasattr(page, 'dimensions'):
                            dims = page.dimensions
                            page_data['dimensions'] = {
                                'width': getattr(dims, 'width', 0),
                                'height': getattr(dims, 'height', 0),
                                'dpi': getattr(dims, 'dpi', 0)
                            }
                        
                        page_info.append(page_data)
                    
                    raw_response['pages'] = page_info
                
            except Exception as e:
                # Fallback to string representation
                raw_response = {
                    'response_type': type(ocr_response).__name__,
                    'text': extracted_text,
                    'confidence': confidence,
                    'error': f'Could not serialize response object: {str(e)}'
                }
            
            # Create a standardized response format
            response_dict = {
                'extracted_text': extracted_text,
                'text_length': len(extracted_text),
                'provider': 'mistral',
                'raw_response': raw_response,
                'confidence': confidence
            }
            
            logger.info(f"Using Mistral OCR API (official SDK) for {image_path} - extracted {len(extracted_text)} characters")
            return response_dict
            
        except Exception as e:
            logger.error(f"Error processing image {image_path} with Mistral OCR: {e}")
            raise
    
    def get_provider_name(self) -> str:
        return "mistral"

class ProgressBar:
    """Simple progress bar for console output."""
    
    def __init__(self, total: int, width: int = 50, prefix: str = "Progress"):
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0
        self.start_time = time.time()
    
    def update(self, current: int, status: str = ""):
        """Update the progress bar."""
        self.current = current
        percent = (current / self.total) * 100
        filled = int(self.width * current // self.total)
        bar = "█" * filled + "░" * (self.width - filled)
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        if current > 0:
            eta = (elapsed / current) * (self.total - current)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        print(f"\r{self.prefix}: |{bar}| {percent:.1f}% ({current}/{self.total}) {eta_str} {status}", end="", flush=True)
    
    def finish(self, status: str = "Complete!"):
        """Finish the progress bar."""
        elapsed = time.time() - self.start_time
        print(f"\r{self.prefix}: |{'█' * self.width}| 100.0% ({self.total}/{self.total}) Time: {elapsed:.1f}s {status}")
        print()  # New line after completion

class CategorizedPDFOCR:
    """
    Enhanced OCR solution for categorized PDF documents supporting multiple OCR providers.
    Organizes results by OCR provider, document category (계약서, 동의서) and document name.
    """
    
    def __init__(self, base_output_dir: str = "ocr_results", ocr_provider: str = "google_cloud", mistral_api_key: str = None):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Store log file path for reference
        self.log_file = log_filename
        
        # Initialize OCR provider
        self.ocr_provider_name = ocr_provider.lower()
        if self.ocr_provider_name == "google_cloud":
            self.ocr_provider = GoogleCloudOCRProvider()
        elif self.ocr_provider_name == "mistral":
            # Try to get API key from parameter first, then from environment
            api_key = mistral_api_key or os.getenv('MISTRAL_API_KEY')
            if not api_key:
                raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY in .env file or pass as parameter.")
            self.ocr_provider = MistralOCRProvider(api_key)
        else:
            raise ValueError(f"Unsupported OCR provider: {ocr_provider}. Supported providers: 'google_cloud', 'mistral'")
        
        # Create provider-specific output directory
        self.provider_output_dir = self.base_output_dir / f"{self.ocr_provider_name}_ocr"
        self.provider_output_dir.mkdir(exist_ok=True)
        
        # Create category directories within provider directory
        self.categories = ["계약서", "동의서"]
        for category in self.categories:
            category_dir = self.provider_output_dir / category
            category_dir.mkdir(exist_ok=True)
            logger.info(f"Created category directory: {category_dir}")
        
        logger.info(f"Initialized OCR processor with {self.ocr_provider_name} provider")
    
    def get_log_file_path(self) -> str:
        """Get the path to the current log file."""
        return str(self.log_file)
    
    def determine_category(self, pdf_path: str) -> str:
        """
        Determine the category of a PDF based on its path.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Category string (계약서 or 동의서)
        """
        pdf_path_lower = pdf_path.lower()
        
        # Check for category indicators in the path
        if "계약서" in pdf_path or "contract" in pdf_path_lower:
            return "계약서"
        elif "동의서" in pdf_path or "consent" in pdf_path_lower:
            return "동의서"
        else:
            # Default based on directory structure
            if "계약서" in str(Path(pdf_path).parent):
                return "계약서"
            elif "동의서" in str(Path(pdf_path).parent):
                return "동의서"
            else:
                # Try to infer from filename
                filename = Path(pdf_path).stem.lower()
                if any(keyword in filename for keyword in ["계약", "contract", "agreement"]):
                    return "계약서"
                elif any(keyword in filename for keyword in ["동의", "consent", "approval"]):
                    return "동의서"
                else:
                    # Default to 동의서 if uncertain
                    logger.warning(f"Could not determine category for {pdf_path}, defaulting to 동의서")
                    return "동의서"
    
    def create_document_output_structure(self, pdf_path: str, category: str) -> Dict[str, Path]:
        """
        Create the organized output structure for a document.
        
        Args:
            pdf_path: Path to the PDF file
            category: Document category (계약서 or 동의서)
            
        Returns:
            Dictionary with paths for different output types
        """
        pdf_name = Path(pdf_path).stem
        category_dir = self.provider_output_dir / category
        document_dir = category_dir / pdf_name
        document_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        images_dir = document_dir / "converted_images"
        json_dir = document_dir / "page_results"
        images_dir.mkdir(exist_ok=True)
        json_dir.mkdir(exist_ok=True)
        
        return {
            'document_dir': document_dir,
            'images_dir': images_dir,
            'json_dir': json_dir,
            'complete_json': document_dir / f"{pdf_name}_complete_ocr.json",
            'extracted_text': document_dir / f"{pdf_name}_extracted_text.txt"
        }
    
    def convert_pdf_to_images(self, pdf_path: str, output_dir: Path) -> List[str]:
        """
        Convert PDF pages to high-quality PNG images.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save converted images
            
        Returns:
            List of paths to converted image files
        """
        image_paths = []
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            logger.info(f"Processing PDF: {pdf_path} with {len(doc)} pages")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Create high-resolution pixmap for better OCR quality
                # Method 1: Using DPI (more precise control)
                pix = page.get_pixmap(dpi=400)  # 300 DPI for high quality
                
                # Method 2: Using Matrix (zoom factor) - uncomment to use instead
                # mat = fitz.Matrix(5.0, 5.0)
                # pix = page.get_pixmap(matrix=mat)
                
                # Save as PNG with high quality
                image_path = output_dir / f"page_{page_num + 1:03d}.png"
                pix.save(str(image_path))
                image_paths.append(str(image_path))
                
                logger.info(f"Converted page {page_num + 1} to {image_path}")
            
            doc.close()
            logger.info(f"Successfully converted {len(image_paths)} pages")
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
            
        return image_paths
    
    def ocr_image(self, image_path: str) -> Dict:
        """
        Perform OCR on a single image using the configured OCR provider.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing OCR results
        """
        try:
            # Use the configured OCR provider
            result = self.ocr_provider.process_image(image_path)
            
            # Add provider information to the result
            result['provider'] = self.ocr_provider_name
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path} with {self.ocr_provider_name}: {e}")
            raise
    
    def process_single_pdf(self, pdf_path: str) -> Dict:
        """
        Process a single PDF document with organized output structure.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing complete OCR results for the document
        """
        pdf_name = Path(pdf_path).stem
        logger.info(f"Starting OCR processing for document: {pdf_name}")
        
        # Determine category
        category = self.determine_category(pdf_path)
        logger.info(f"Document category: {category}")
        
        # Create output structure
        output_paths = self.create_document_output_structure(pdf_path, category)
        
        # Convert PDF to images
        image_paths = self.convert_pdf_to_images(pdf_path, output_paths['images_dir'])
        
        # Process each image with OCR
        document_results = {
            'document_name': pdf_name,
            'category': category,
            'ocr_provider': self.ocr_provider_name,
            'original_path': pdf_path,
            'total_pages': len(image_paths),
            'processing_timestamp': datetime.now().isoformat(),
            'output_directory': str(output_paths['document_dir']),
            'pages': []
        }
        
        all_text = []
        
        # Initialize page progress bar for documents with multiple pages
        if len(image_paths) > 1:
            page_progress = ProgressBar(len(image_paths), prefix=f"Pages ({pdf_name})")
        
        for i, image_path in enumerate(image_paths):
            if len(image_paths) > 1:
                page_progress.update(i + 1, f"Page {i + 1}")
            
            logger.info(f"Processing page {i + 1}/{len(image_paths)}")
            
            try:
                # Perform OCR
                page_result = self.ocr_image(image_path)
                
                # Add page metadata
                page_result['page_number'] = i + 1
                page_result['image_path'] = image_path
                
                document_results['pages'].append(page_result)
                all_text.append(page_result.get('extracted_text', ''))
                
                # Save individual page results
                page_output_path = output_paths['json_dir'] / f"page_{i + 1:03d}_ocr.json"
                with open(page_output_path, 'w', encoding='utf-8') as f:
                    json.dump(page_result, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved page {i + 1} results to {page_output_path}")
                
            except Exception as e:
                logger.error(f"Error processing page {i + 1}: {e}")
                # Add error information to results
                document_results['pages'].append({
                    'page_number': i + 1,
                    'image_path': image_path,
                    'error': str(e),
                    'extracted_text': '',
                    'text_length': 0
                })
        
        # Finish page progress bar if it was used
        if len(image_paths) > 1:
            page_progress.finish("Document complete!")
        
        # Combine all text
        document_results['full_text'] = '\n\n--- PAGE BREAK ---\n\n'.join(all_text)
        document_results['total_text_length'] = len(document_results['full_text'])
        
        # Save complete document results
        with open(output_paths['complete_json'], 'w', encoding='utf-8') as f:
            json.dump(document_results, f, indent=2, ensure_ascii=False)
        
        # Save full text as separate file
        with open(output_paths['extracted_text'], 'w', encoding='utf-8') as f:
            f.write(document_results['full_text'])
        
        logger.info(f"Complete OCR results saved to {output_paths['complete_json']}")
        logger.info(f"Extracted text saved to {output_paths['extracted_text']}")
        
        return document_results
    
    def find_pdf_files(self, directories: List[str]) -> List[str]:
        """
        Find all PDF files in the specified directories.
        
        Args:
            directories: List of directory paths to search
            
        Returns:
            List of PDF file paths
        """
        pdf_files = []
        
        for directory in directories:
            if os.path.exists(directory):
                logger.info(f"Searching for PDFs in: {directory}")
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.lower().endswith('.pdf'):
                            pdf_path = os.path.join(root, file)
                            pdf_files.append(pdf_path)
                            logger.info(f"Found PDF: {pdf_path}")
            else:
                logger.warning(f"Directory not found: {directory}")
        
        logger.info(f"Total PDF files found: {len(pdf_files)}")
        return pdf_files
    
    def batch_process_categorized_pdfs(self, source_directories: List[str]) -> Dict:
        """
        Process all PDFs from multiple source directories with organized output.
        
        Args:
            source_directories: List of directories containing PDF files
            
        Returns:
            Dictionary containing summary of processing results
        """
        logger.info(f"Starting batch processing of PDFs from {len(source_directories)} directories")
        
        # Find all PDF files
        pdf_files = self.find_pdf_files(source_directories)
        
        if not pdf_files:
            logger.warning("No PDF files found to process")
            return {'status': 'no_files_found', 'processed': 0}
        
        # Process each PDF
        results = {
            'total_files': len(pdf_files),
            'processed_files': 0,
            'failed_files': 0,
            'categories': {'계약서': 0, '동의서': 0},
            'processing_summary': [],
            'start_time': datetime.now().isoformat()
        }
        
        # Initialize progress bar
        progress_bar = ProgressBar(len(pdf_files), prefix="Processing PDFs")
        print(f"\n🚀 Starting batch processing of {len(pdf_files)} PDF files...")
        
        for i, pdf_path in enumerate(pdf_files, 1):
            try:
                pdf_name = Path(pdf_path).name
                progress_bar.update(i, f"Processing: {pdf_name[:30]}...")
                
                # Process the PDF
                document_result = self.process_single_pdf(pdf_path)
                
                # Update summary
                category = document_result['category']
                results['categories'][category] += 1
                results['processed_files'] += 1
                
                results['processing_summary'].append({
                    'file': pdf_path,
                    'category': category,
                    'pages': document_result['total_pages'],
                    'text_length': document_result['total_text_length'],
                    'status': 'success',
                    'output_dir': document_result['output_directory']
                })
                
                logger.info(f"✅ Successfully processed: {pdf_name} ({category})")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {pdf_path}: {e}")
                results['failed_files'] += 1
                results['processing_summary'].append({
                    'file': pdf_path,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Finish progress bar
        progress_bar.finish("Batch processing complete!")
        
        results['end_time'] = datetime.now().isoformat()
        results['status'] = 'completed'
        
        # Save batch processing summary
        summary_path = self.base_output_dir / "batch_processing_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing completed!")
        logger.info(f"Total files: {results['total_files']}")
        logger.info(f"Successfully processed: {results['processed_files']}")
        logger.info(f"Failed: {results['failed_files']}")
        logger.info(f"계약서: {results['categories']['계약서']}")
        logger.info(f"동의서: {results['categories']['동의서']}")
        logger.info(f"Summary saved to: {summary_path}")
        
        return results

def main():
    """Main function to run the categorized batch OCR processing."""
    
    # Define source directories
    source_directories = [
        "Project/data/pdf/계약서", 
        "Project/data/pdf/동의서"
    ]
    
    # Initialize OCR processor
    ocr_processor = CategorizedPDFOCR(base_output_dir="Project/OCR/google_vision/ocr_results")
    
    try:
        # Process all PDFs
        results = ocr_processor.batch_process_categorized_pdfs(source_directories)
        
        print(f"\n🎉 Batch OCR Processing Complete!")
        print(f"📊 Summary:")
        print(f"   Total files: {results['total_files']}")
        print(f"   Successfully processed: {results['processed_files']}")
        print(f"   Failed: {results['failed_files']}")
        print(f"   계약서: {results['categories']['계약서']}")
        print(f"   동의서: {results['categories']['동의서']}")
        print(f"📁 Results organized in: {ocr_processor.base_output_dir}")
        
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        logger.error(f"Main processing error: {e}")

if __name__ == "__main__":
    main()
