#!/usr/bin/env python3
"""
Test script for PDF Document OCR using Mistral API.
This script demonstrates the basic functionality of Mistral OCR.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mistral_ocr_simple import SimpleMistralOCR

def test_single_pdf():
    """Test processing a single PDF document with Mistral OCR."""
    print("🧪 Testing Single PDF OCR Processing with Mistral")
    print("=" * 50)
    
    try:
        # Initialize Mistral OCR processor
        ocr_processor = SimpleMistralOCR(output_dir="test_mistral_ocr_results")
        
        # Test with a sample PDF
        pdf_path = input("Enter the path to your test PDF document: ").strip()
        
        if not os.path.exists(pdf_path):
            print(f"❌ Error: File {pdf_path} not found!")
            return False
        
        print(f"📄 Processing PDF: {pdf_path}")
        
        # Process the document
        results = ocr_processor.process_pdf_document(pdf_path)
        
        print(f"\n✅ Mistral OCR Processing Complete!")
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
    """Test batch processing of multiple PDFs with Mistral OCR."""
    print("\n🧪 Testing Batch PDF OCR Processing with Mistral")
    print("=" * 50)
    
    pdf_directory = input("Enter the directory containing PDF files: ").strip()
    
    if not os.path.exists(pdf_directory):
        print(f"❌ Error: Directory {pdf_directory} not found!")
        return False
    
    try:
        # Initialize Mistral OCR processor
        ocr_processor = SimpleMistralOCR(output_dir="batch_mistral_ocr_results")
        
        print(f"📁 Processing PDFs in directory: {pdf_directory}")
        
        # Get all PDF files
        pdf_dir = Path(pdf_directory)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("⚠️ No PDF files found to process")
            return False
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        results = []
        for pdf_file in pdf_files:
            try:
                print(f"\nProcessing {pdf_file.name}...")
                result = ocr_processor.process_pdf_document(str(pdf_file))
                results.append(result)
                print(f"✅ {pdf_file.name}: {result['total_pages']} pages, {result['total_text_length']} chars")
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {e}")
                results.append({
                    'document_name': pdf_file.stem,
                    'error': str(e),
                    'status': 'failed'
                })
        
        print(f"\n✅ Batch Processing Complete!")
        print(f"📊 Processed {len(results)} documents")
        
        return True
            
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
        # Initialize Mistral OCR processor
        ocr_processor = SimpleMistralOCR(output_dir="conversion_test_results")
        
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
    print("Using Mistral API")
    print("=" * 50)
    
    while True:
        print("\n📋 Test Options:")
        print("1. Test Single PDF OCR Processing (Mistral)")
        print("2. Test Batch PDF Processing (Mistral)")
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
