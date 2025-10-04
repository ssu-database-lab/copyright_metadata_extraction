#!/usr/bin/env python3
"""
Test script for PDF Document OCR using Google Cloud Vision API.
This script demonstrates the basic functionality of the PDFDocumentOCR class.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_document_ocr import PDFDocumentOCR

def test_single_pdf():
    """Test processing a single PDF document."""
    print("🧪 Testing Single PDF OCR Processing")
    print("=" * 50)
    
    # Initialize OCR processor
    ocr_processor = PDFDocumentOCR(output_dir="test_ocr_results")
    
    # Test with a sample PDF (you'll need to provide your own)
    pdf_path = input("Enter the path to your test PDF document: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File {pdf_path} not found!")
        return False
    
    try:
        print(f"📄 Processing PDF: {pdf_path}")
        
        # Process the document
        results = ocr_processor.process_pdf_document(
            pdf_path=pdf_path,
            save_images=True,
            save_individual_results=True
        )
        
        print(f"\n✅ OCR Processing Complete!")
        print(f"📄 Document: {results['document_name']}")
        print(f"📖 Total Pages: {results['total_pages']}")
        print(f"📝 Total Text Length: {results['total_text_length']} characters")
        print(f"💾 Results saved to: {ocr_processor.output_dir}")
        
        # Show sample text from each page
        for i, page in enumerate(results['pages']):
            if 'extracted_text' in page and page['extracted_text']:
                print(f"\n📋 Page {i+1} Sample (first 100 chars):")
                text = page['extracted_text'][:100]
                print(text + "..." if len(page['extracted_text']) > 100 else text)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        return False

def test_batch_processing():
    """Test batch processing of multiple PDFs in a directory."""
    print("\n🧪 Testing Batch PDF OCR Processing")
    print("=" * 50)
    
    pdf_directory = input("Enter the directory containing PDF files: ").strip()
    
    if not os.path.exists(pdf_directory):
        print(f"❌ Error: Directory {pdf_directory} not found!")
        return False
    
    try:
        # Initialize OCR processor
        ocr_processor = PDFDocumentOCR(output_dir="batch_ocr_results")
        
        print(f"📁 Processing PDFs in directory: {pdf_directory}")
        
        # Process all PDFs in the directory
        results = ocr_processor.batch_process_pdfs(pdf_directory)
        
        if results:
            print(f"\n✅ Batch Processing Complete!")
            print(f"📊 Processed {len(results)} documents")
            
            for result in results:
                if 'error' in result:
                    print(f"❌ {result['document_name']}: {result['error']}")
                else:
                    print(f"✅ {result['document_name']}: {result['total_pages']} pages, {result['total_text_length']} chars")
            
            return True
        else:
            print("⚠️ No PDF files found to process")
            return False
            
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        return False

def test_image_conversion():
    """Test PDF to image conversion functionality."""
    print("\n🧪 Testing PDF to Image Conversion")
    print("=" * 50)
    
    pdf_path = input("Enter the path to your test PDF document: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File {pdf_path} not found!")
        return False
    
    try:
        # Initialize OCR processor
        ocr_processor = PDFDocumentOCR(output_dir="conversion_test_results")
        
        print(f"🔄 Converting PDF to images: {pdf_path}")
        
        # Convert PDF to images
        image_paths = ocr_processor.convert_pdf_to_images(pdf_path)
        
        print(f"✅ Conversion Complete!")
        print(f"🖼️ Generated {len(image_paths)} images:")
        
        for i, image_path in enumerate(image_paths):
            print(f"   Page {i+1}: {image_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting PDF to images: {e}")
        return False

def main():
    """Main test function with menu options."""
    print("🚀 PDF Document OCR Test Suite")
    print("Using Google Cloud Vision API")
    print("=" * 50)
    
    while True:
        print("\n📋 Test Options:")
        print("1. Test Single PDF OCR Processing")
        print("2. Test Batch PDF Processing")
        print("3. Test PDF to Image Conversion")
        print("4. Exit")
        
        choice = input("\nSelect an option (1-4): ").strip()
        
        if choice == "1":
            test_single_pdf()
        elif choice == "2":
            test_batch_processing()
        elif choice == "3":
            test_image_conversion()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
