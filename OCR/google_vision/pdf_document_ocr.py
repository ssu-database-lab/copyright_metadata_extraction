import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from google.cloud import vision
from google.protobuf.json_format import MessageToDict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Force IPv4 for gRPC (fixes WSL2 IPv6 issues)
os.environ["GRPC_DNS_RESOLVER"] = "native"

# Set up Google Cloud credentials
# Get the directory where this script is located and construct the credentials path
script_dir = os.path.dirname(os.path.abspath(__file__))
credentials_path = os.path.join(script_dir, "semiotic-pager-466612-t0-c587b9296fb8.json")

# Check if credentials file exists
if not os.path.exists(credentials_path):
    raise FileNotFoundError(f"Google Cloud credentials file not found at: {credentials_path}\n"
                           f"Please ensure the credentials file is in the same directory as this script.")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
logger.info(f"Using Google Cloud credentials from: {credentials_path}")

class PDFDocumentOCR:
    """
    A comprehensive OCR solution for scanned PDF documents using Google Cloud Vision API.
    Handles PDF to image conversion and OCR processing with document text detection.
    """
    
    def __init__(self, output_dir: str = "ocr_results"):
        self.client = vision.ImageAnnotatorClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def convert_pdf_to_images(self, pdf_path: str, output_dir: str = None) -> List[str]:
        """
        Convert PDF pages to high-quality PNG images for better OCR results.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save converted images (optional)
            
        Returns:
            List of paths to converted image files
        """
        if output_dir is None:
            output_dir = self.output_dir / "converted_images"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pdf_name = Path(pdf_path).stem
        image_paths = []
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            logger.info(f"Processing PDF: {pdf_path} with {len(doc)} pages")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Create high-resolution pixmap for better OCR quality
                # Use 2x zoom for higher resolution
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                
                # Save as PNG with high quality
                image_path = output_path / f"{pdf_name}_page_{page_num + 1:03d}.png"
                pix.save(str(image_path))
                image_paths.append(str(image_path))
                
                logger.info(f"Converted page {page_num + 1} to {image_path}")
            
            doc.close()
            logger.info(f"Successfully converted {len(image_paths)} pages")
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise
            
        return image_paths
    
    def ocr_image(self, image_path: str, use_document_detection: bool = True) -> Dict:
        """
        Perform OCR on a single image using Google Cloud Vision API.
        
        Args:
            image_path: Path to the image file
            use_document_detection: Whether to use document text detection (better for scanned docs)
            
        Returns:
            Dictionary containing OCR results
        """
        try:
            # Read image file
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            if use_document_detection:
                # Use document text detection for better results with scanned documents
                response = self.client.document_text_detection(image=image)
                logger.info(f"Using document text detection for {image_path}")
            else:
                # Use regular text detection for general images
                response = self.client.text_detection(image=image)
                logger.info(f"Using general text detection for {image_path}")
            
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
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def process_pdf_document(self, pdf_path: str, save_images: bool = True, 
                           save_individual_results: bool = True) -> Dict:
        """
        Complete workflow: Convert PDF to images and perform OCR on each page.
        
        Args:
            pdf_path: Path to the PDF file
            save_images: Whether to save converted images
            save_individual_results: Whether to save individual page results
            
        Returns:
            Dictionary containing complete OCR results for the document
        """
        pdf_name = Path(pdf_path).stem
        logger.info(f"Starting OCR processing for document: {pdf_name}")
        
        # Convert PDF to images
        image_paths = self.convert_pdf_to_images(pdf_path)
        
        # Process each image with OCR
        document_results = {
            'document_name': pdf_name,
            'total_pages': len(image_paths),
            'processing_timestamp': str(Path().cwd()),
            'pages': []
        }
        
        all_text = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing page {i + 1}/{len(image_paths)}")
            
            try:
                # Perform OCR
                page_result = self.ocr_image(image_path)
                
                # Add page metadata
                page_result['page_number'] = i + 1
                page_result['image_path'] = image_path
                
                document_results['pages'].append(page_result)
                all_text.append(page_result.get('extracted_text', ''))
                
                # Save individual page results if requested
                if save_individual_results:
                    page_output_path = self.output_dir / f"{pdf_name}_page_{i + 1:03d}_ocr.json"
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
        
        # Combine all text
        document_results['full_text'] = '\n\n--- PAGE BREAK ---\n\n'.join(all_text)
        document_results['total_text_length'] = len(document_results['full_text'])
        
        # Save complete document results
        doc_output_path = self.output_dir / f"{pdf_name}_complete_ocr.json"
        with open(doc_output_path, 'w', encoding='utf-8') as f:
            json.dump(document_results, f, indent=2, ensure_ascii=False)
        
        # Save full text as separate file
        text_output_path = self.output_dir / f"{pdf_name}_extracted_text.txt"
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(document_results['full_text'])
        
        logger.info(f"Complete OCR results saved to {doc_output_path}")
        logger.info(f"Extracted text saved to {text_output_path}")
        
        return document_results
    
    def batch_process_pdfs(self, pdf_directory: str, pattern: str = "*.pdf") -> List[Dict]:
        """
        Process multiple PDF files in a directory.
        
        Args:
            pdf_directory: Directory containing PDF files
            pattern: File pattern to match (default: *.pdf)
            
        Returns:
            List of results for each processed document
        """
        pdf_dir = Path(pdf_directory)
        pdf_files = list(pdf_dir.glob(pattern))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory} matching pattern {pattern}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = []
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}")
                result = self.process_pdf_document(str(pdf_file))
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                results.append({
                    'document_name': pdf_file.stem,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return results

def main():
    """Example usage of the PDFDocumentOCR class."""
    
    # Initialize OCR processor
    ocr_processor = PDFDocumentOCR()
    
    # Example 1: Process a single PDF
    pdf_path = input("Enter the path to your PDF document: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found!")
        return
    
    try:
        # Process the document
        results = ocr_processor.process_pdf_document(pdf_path)
        
        print(f"\n✅ OCR Processing Complete!")
        print(f"📄 Document: {results['document_name']}")
        print(f"📖 Total Pages: {results['total_pages']}")
        print(f"📝 Total Text Length: {results['total_text_length']} characters")
        print(f"💾 Results saved to: {ocr_processor.output_dir}")
        
        # Show first 200 characters of extracted text
        if results['full_text']:
            print(f"\n📋 Sample Text (first 200 chars):")
            print(results['full_text'][:200] + "..." if len(results['full_text']) > 200 else results['full_text'])
        
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        logger.error(f"Main processing error: {e}")

if __name__ == "__main__":
    main()
