#!/usr/bin/env python3
"""
Universal OCR Processor for API Module
Handles all file types: documents (PDF, DOCX, DOC, PPTX, XLS, XLSX, PPT, HWP) and images (GIF, JPG, JPEG, PNG, BMP, TIF, TIFF)
Supports multiple OCR providers: Google Cloud Vision, Mistral, Naver Clova OCR, Alibaba Cloud
"""

import os
import sys
import json
import base64
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Generator
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

# Configure logging first
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
import os

# Load environment variables from multiple locations
env_paths = [
    Path(__file__).parent.parent / ".env",  # API directory
    Path(__file__).parent.parent.parent / "OCR" / "google_vision" / ".env",  # OCR directory
    Path(__file__).parent.parent / ".env_alibaba",  # Alibaba specific
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from: {env_path}")
        break
else:
    logger.warning("No .env file found. Using system environment variables only.")

# Force IPv4 for gRPC (fixes WSL2 IPv6 issues)
os.environ["GRPC_DNS_RESOLVER"] = "native"

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

class FileProcessor:
    """Handles file conversion to images for OCR processing."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_file(self, file_path: str) -> List[str]:
        """Convert file to images and return image paths."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        # Determine file type and process accordingly
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self._process_pdf(file_path)
        elif suffix in ['.docx', '.doc']:
            return self._process_docx(file_path)
        elif suffix in ['.pptx', '.ppt']:
            return self._process_pptx(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return self._process_xlsx(file_path)
        elif suffix in ['.hwp']:
            return self._process_hwp(file_path)
        elif suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff']:
            return [str(file_path)]
        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return []
    
    def _process_pdf(self, file_path: Path) -> List[str]:
        """Convert PDF to images."""
        try:
            doc = fitz.open(file_path)
            image_paths = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                
                image_path = self.output_dir / f"{file_path.stem}_page_{page_num + 1}.png"
                pix.save(str(image_path))
                image_paths.append(str(image_path))
            
            doc.close()
            logger.info(f"PDF converted to {len(image_paths)} images")
            return image_paths
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []
    
    def _process_docx(self, file_path: Path) -> List[str]:
        """Convert DOCX to images (placeholder - would need additional libraries)."""
        if not DOCX_AVAILABLE:
            logger.warning("python-docx not available. DOCX processing skipped.")
            return []
        
        # Placeholder implementation
        logger.warning("DOCX to image conversion not implemented yet")
        return []
    
    def _process_pptx(self, file_path: Path) -> List[str]:
        """Convert PPTX to images (placeholder)."""
        logger.warning("PPTX to image conversion not implemented yet")
        return []
    
    def _process_xlsx(self, file_path: Path) -> List[str]:
        """Convert XLSX to images (placeholder)."""
        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available. XLSX processing skipped.")
            return []
        
        logger.warning("XLSX to image conversion not implemented yet")
        return []
    
    def _process_hwp(self, file_path: Path) -> List[str]:
        """Convert HWP to images (placeholder)."""
        logger.warning("HWP to image conversion not implemented yet")
        return []

class UniversalOCRProcessor:
    """Universal OCR processor supporting multiple providers and file types."""
    
    def __init__(self, provider: str, output_dir: str = "universal_ocr_results", model: str = None):
        self.provider_name = provider.lower()
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OCR provider
        if self.provider_name == "google":
            from .google_ocr import GoogleCloudOCRProvider
            self.ocr_provider = GoogleCloudOCRProvider()
        elif self.provider_name == "mistral":
            from .mistral_ocr import MistralOCRProvider
            self.ocr_provider = MistralOCRProvider()
        elif self.provider_name == "naver":
            from .naver_ocr import NaverOCRProvider
            self.ocr_provider = NaverOCRProvider()
        elif self.provider_name == "alibaba":
            from .alibaba_ocr import AlibabaCloudOCRProvider
            api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('ALIBABA_API_KEY')
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY or ALIBABA_API_KEY environment variable not set")
            
            region = os.getenv('ALIBABA_REGION', 'singapore')
            alibaba_model = model or os.getenv('ALIBABA_MODEL', 'qwen-vl-ocr')
            
            temperature = float(os.getenv('ALIBABA_TEMPERATURE', '1.0'))
            top_p = float(os.getenv('ALIBABA_TOP_P', '0.8'))
            top_k = os.getenv('ALIBABA_TOP_K')
            if top_k:
                top_k = int(top_k)
            
            self.ocr_provider = AlibabaCloudOCRProvider(
                api_key, 
                model=alibaba_model, 
                region=region,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
        else:
            raise ValueError(f"Unsupported OCR provider: {provider}")
        
        # Create provider-specific output directory
        if self.provider_name == "alibaba" and hasattr(self.ocr_provider, 'model'):
            model_name = self.ocr_provider.model.replace('-', '_')
            self.provider_output_dir = self.base_output_dir / f"{self.provider_name}_ocr" / model_name
        else:
            self.provider_output_dir = self.base_output_dir / f"{self.provider_name}_ocr"
        self.provider_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {self.provider_name} OCR processor")
        logger.info(f"Output directory: {self.provider_output_dir}")
        if hasattr(self.ocr_provider, 'model'):
            logger.info(f"Model: {self.ocr_provider.model}")
        if hasattr(self.ocr_provider, 'temperature'):
            logger.info(f"Temperature: {self.ocr_provider.temperature}, Top-p: {self.ocr_provider.top_p}")
    
    def create_structured_output_paths(self, file_path: str) -> Dict[str, Path]:
        """Create structured output paths for a file."""
        file_path = Path(file_path)
        
        # Create category-specific subdirectory
        category = self._determine_category(file_path)
        category_dir = self.provider_output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Create document-specific subdirectory
        doc_dir = category_dir / file_path.stem
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        # Create images subdirectory
        images_dir = doc_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            'category_dir': category_dir,
            'doc_dir': doc_dir,
            'images_dir': images_dir,
            'text_file': doc_dir / f"{file_path.stem}_extracted_text.txt",
            'result_file': doc_dir / f"{file_path.stem}_ocr_result.json"
        }
    
    def _determine_category(self, file_path: Path) -> str:
        """Determine document category based on file path."""
        # Simple category detection based on filename patterns
        filename = file_path.name.lower()
        
        if any(keyword in filename for keyword in ['계약서', 'contract', 'agreement']):
            return 'contract'
        elif any(keyword in filename for keyword in ['동의서', 'consent', 'agreement']):
            return 'consent'
        elif any(keyword in filename for keyword in ['저작권', 'copyright', '양도']):
            return 'copyright'
        else:
            return 'general'
    
    def process_single_file(self, file_path: str) -> Dict:
        """Process a single file with OCR."""
        file_path = Path(file_path)
        logger.info(f"Processing file: {file_path.name}")
        
        try:
            output_paths = self.create_structured_output_paths(str(file_path))
            temp_file_processor = FileProcessor(output_paths['images_dir'])
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
                    'ocr_provider': self.ocr_provider.get_provider_name()
                }
            
            all_text = []
            page_results = []
            
            for i, image_path in enumerate(image_paths):
                try:
                    result = self.ocr_provider.process_image(image_path)
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
            
            full_text = '\n\n'.join(all_text)
            
            # Save extracted text
            with open(output_paths['text_file'], 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            # Prepare result data
            result_data = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'total_pages': len(image_paths),
                'total_text_length': len(full_text),
                'pages': page_results,
                'full_text': full_text,
                'ocr_provider': self.ocr_provider.get_provider_name(),
                'status': 'success'
            }
            
            # Save result JSON
            with open(output_paths['result_file'], 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"OCR processing complete for {file_path.name}: {len(full_text)} characters")
            return result_data
            
        except Exception as e:
            logger.error(f"Error in file processing for {file_path}: {e}")
            return {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'status': 'failed',
                'error': str(e),
                'total_pages': 0,
                'total_text_length': 0,
                'pages': [],
                'full_text': '',
                'ocr_provider': self.ocr_provider.get_provider_name()
            }
    
    def process_directory(self, directory_path: str) -> Dict:
        """Process all files in a directory."""
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            return {
                'status': 'failed',
                'error': f"Directory not found: {directory_path}",
                'processed_files': 0,
                'results': []
            }
        
        # Find all supported files
        supported_extensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.hwp',
                              '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff']
        
        files_to_process = []
        for ext in supported_extensions:
            files_to_process.extend(directory_path.rglob(f"*{ext}"))
        
        if not files_to_process:
            return {
                'status': 'failed',
                'error': 'No supported files found',
                'processed_files': 0,
                'results': []
            }
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        results = []
        successful_count = 0
        
        for file_path in files_to_process:
            try:
                result = self.process_single_file(str(file_path))
                results.append(result)
                
                if result['status'] == 'success':
                    successful_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append({
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'status': 'failed',
                    'error': str(e)
                })
        
        return {
            'status': 'success',
            'total_files': len(files_to_process),
            'processed_files': successful_count,
            'failed_files': len(files_to_process) - successful_count,
            'results': results,
            'ocr_provider': self.ocr_provider.get_provider_name()
        }
    
    def process_single_file_streaming(self, file_path: str) -> Generator[str, None, None]:
        """Process a single file with streaming output."""
        file_path = Path(file_path)
        logger.info(f"Processing file with streaming: {file_path.name}")
        
        try:
            output_paths = self.create_structured_output_paths(str(file_path))
            temp_file_processor = FileProcessor(output_paths['images_dir'])
            image_paths = temp_file_processor.process_file(str(file_path))
            
            if not image_paths:
                yield f"Error: Unsupported file type or conversion failed for {file_path.name}"
                return
            
            all_text = []
            
            for i, image_path in enumerate(image_paths):
                if hasattr(self.ocr_provider, 'process_image_streaming'):
                    page_text = ""
                    for chunk in self.ocr_provider.process_image_streaming(image_path):
                        page_text += chunk
                        yield chunk
                    all_text.append(page_text)
                else:
                    # Fallback to regular OCR
                    result = self.ocr_provider.process_image(image_path)
                    page_text = result.get('extracted_text', '')
                    all_text.append(page_text)
                    yield page_text
            
            # Save full text
            full_text = '\n\n'.join(all_text)
            with open(output_paths['text_file'], 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            logger.info(f"Streaming OCR processing complete for {file_path.name}: {len(full_text)} characters")
            
        except Exception as e:
            logger.error(f"Error in streaming file processing for {file_path}: {e}")
            yield f"Error: {str(e)}"
    
    def process_single_file_api_client(self, file_path: str) -> Dict:
        """Process a single file using API Client approach (non-streaming)."""
        file_path = Path(file_path)
        logger.info(f"Processing file with API Client: {file_path.name}")
        
        try:
            output_paths = self.create_structured_output_paths(str(file_path))
            temp_file_processor = FileProcessor(output_paths['images_dir'])
            image_paths = temp_file_processor.process_file(str(file_path))
            
            if not image_paths:
                return {
                    'file_name': file_path.name, 'file_path': str(file_path), 'status': 'failed',
                    'error': 'Unsupported file type or conversion failed', 'total_pages': 0,
                    'total_text_length': 0, 'pages': [], 'full_text': '',
                    'ocr_provider': self.ocr_provider.get_provider_name(), 'processing_mode': 'api_client'
                }
            
            all_text = []
            page_results = []
            
            for i, image_path in enumerate(image_paths):
                try:
                    if hasattr(self.ocr_provider, 'process_image_api_client'):
                        result = self.ocr_provider.process_image_api_client(image_path)
                        page_text = result.get('extracted_text', '')
                        all_text.append(page_text)
                        page_results.append({
                            'page_number': i + 1, 'image_path': image_path, 'extracted_text': page_text,
                            'text_length': len(page_text), 'status': 'success', 'metadata': result.get('metadata', {})
                        })
                    else:
                        result = self.ocr_provider.process_image(image_path)
                        page_text = result.get('extracted_text', '')
                        all_text.append(page_text)
                        page_results.append({
                            'page_number': i + 1, 'image_path': image_path, 'extracted_text': page_text,
                            'text_length': len(page_text), 'status': 'success'
                        })
                except Exception as e:
                    logger.error(f"Error processing page {i+1}: {e}")
                    page_results.append({
                        'page_number': i + 1, 'image_path': image_path, 'extracted_text': '',
                        'text_length': 0, 'status': 'failed', 'error': str(e)
                    })
            
            full_text = '\n\n'.join(all_text)
            with open(output_paths['text_file'], 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            result_data = {
                'file_name': file_path.name, 'file_path': str(file_path), 'total_pages': len(image_paths),
                'total_text_length': len(full_text), 'pages': page_results, 'full_text': full_text,
                'ocr_provider': self.ocr_provider.get_provider_name(), 'processing_mode': 'api_client',
                'status': 'success'
            }
            
            with open(output_paths['result_file'], 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"API Client processing complete for {file_path.name}: {len(full_text)} characters")
            return result_data
            
        except Exception as e:
            logger.error(f"Error in API Client file processing for {file_path}: {e}")
            return {
                'file_name': file_path.name, 'file_path': str(file_path), 'status': 'failed',
                'error': str(e), 'total_pages': 0, 'total_text_length': 0, 'pages': [],
                'full_text': '', 'ocr_provider': self.ocr_provider.get_provider_name(),
            }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Universal OCR Processor")
    parser.add_argument("--provider", required=True, choices=["google", "mistral", "naver", "alibaba"],
                       help="OCR provider to use")
    parser.add_argument("--input", required=True, help="Input file or directory path")
    parser.add_argument("--output", default="universal_ocr_results", help="Output directory")
    parser.add_argument("--model", help="Model name (for Alibaba Cloud)")
    parser.add_argument("--stream", "-s", action="store_true", help="Enable streaming output")
    
    args = parser.parse_args()
    
    try:
        processor = UniversalOCRProcessor(args.provider, args.output, args.model)
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            if args.stream:
                print("Streaming OCR processing...")
                for chunk in processor.process_single_file_streaming(str(input_path)):
                    print(chunk, end='', flush=True)
            else:
                result = processor.process_single_file(str(input_path))
                print(f"Processing complete: {result['status']}")
                if result['status'] == 'success':
                    print(f"Text length: {result['total_text_length']} characters")
        elif input_path.is_dir():
            result = processor.process_directory(str(input_path))
            print(f"Directory processing complete: {result['status']}")
            print(f"Processed: {result['processed_files']}/{result['total_files']} files")
        else:
            print(f"Invalid input path: {input_path}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
