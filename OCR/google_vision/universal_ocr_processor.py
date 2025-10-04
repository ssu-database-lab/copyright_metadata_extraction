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
    """Alibaba Cloud Model Studio OCR provider using Qwen-OCR model with DashScope SDK."""
    
    def __init__(self, api_key: str, model: str = "qwen-vl-ocr", region: str = "singapore"):
        """
        Initialize the Alibaba Cloud Model Studio OCR client.
        
        Args:
            api_key: Your Alibaba Cloud API key
            model: Model name (default: qwen-vl-ocr)
            region: Region - "singapore" or "china"
        """
        try:
            import dashscope
            self.dashscope = dashscope
        except ImportError:
            raise ImportError("dashscope package not found. Install with: pip install dashscope")
        
        self.api_key = api_key
        self.model = model
        self.region = region
        
        # Set base URL based on region
        if region == "singapore":
            self.dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
        else:  # china
            self.dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
    
    def process_image(self, image_path: str) -> Dict:
        """Process image using Alibaba Cloud Qwen-OCR model with DashScope SDK."""
        try:
            # Convert image path to file:// format for local file upload
            image_file_path = f"file://{os.path.abspath(image_path)}"
            
            # Prepare messages for DashScope SDK
            messages = [{
                    "role": "user",
                    "content": [{
                            "image": image_file_path,
                            # Minimum pixel threshold for the input image
                            # "min_pixels": 28 * 28 * 4,
                            # Maximum pixel threshold for the input image
                            # "max_pixels": 28 * 28 * 8192,
                            "enable_rotate": True},
                        {"text": "Please output only the text content from the image without any additional descriptions but try your best to mainting the original layout."}]
                }]
            
            logger.info(f"Making Alibaba Cloud Qwen-OCR request using DashScope SDK for {image_path}")
            
            # Call DashScope MultiModalConversation
            response = self.dashscope.MultiModalConversation.call(
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                ocr_options={"task": "multi_lan"}
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
            
            logger.info(f"Alibaba Cloud Qwen-OCR processed {image_path} - {len(extracted_text)} characters")
            
            return {
                "provider": "alibaba_cloud",
                "extracted_text": extracted_text.strip(),
                "text_length": len(extracted_text.strip()),
                "raw_response": response,
                "status": "success"
            }
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Alibaba Cloud Qwen-OCR error: {error_msg}")
            return {
                "provider": "alibaba_cloud",
                "extracted_text": "",
                "text_length": 0,
                "error": error_msg,
                "status": "error"
            }
    
    def get_provider_name(self) -> str:
        return "alibaba_cloud"

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
    
    def __init__(self, provider: str, output_dir: str = "universal_ocr_results"):
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
            self.ocr_provider = AlibabaCloudOCRProvider(api_key, region=region)
        else:
            raise ValueError(f"Unsupported OCR provider: {provider}")
        
        # Create provider-specific output directory
        self.provider_output_dir = self.base_output_dir / f"{self.provider_name}_ocr"
        self.provider_output_dir.mkdir(exist_ok=True)
        
        # Initialize file processor with provider-specific directory
        self.file_processor = FileProcessor(self.provider_output_dir / "converted_images")
        
        logger.info(f"Initialized Universal OCR Processor with {self.provider_name} provider")
        logger.info(f"Provider output directory: {self.provider_output_dir}")
    
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
    parser.add_argument("--provider", "-p", choices=["google_cloud", "mistral", "naver"], 
                       default="mistral", help="OCR provider to use")
    parser.add_argument("--output", "-o", default="universal_ocr_results", 
                       help="Output directory for results")
    parser.add_argument("--recursive", "-r", action="store_true", 
                       help="Process subdirectories recursively")
    parser.add_argument("--single-file", "-f", action="store_true", 
                       help="Process as single file (not directory)")
    
    args = parser.parse_args()
    
    print("🚀 Universal OCR Processor")
    print("=" * 50)
    print(f"Provider: {args.provider}")
    print(f"Path: {args.path}")
    print(f"Output: {args.output}")
    print(f"Recursive: {args.recursive}")
    print("=" * 50)
    
    try:
        processor = UniversalOCRProcessor(args.provider, args.output)
        
        if args.single_file or Path(args.path).is_file():
            # Process single file
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
