#!/usr/bin/env python3
"""
Simple script to run the categorized batch OCR processing.
This script will process all PDFs from the specified directories and organize results by category.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_batch_ocr(ocr_provider: str = "google_cloud"):
    """Run the batch OCR processing for all PDFs."""
    
    print("🚀 Starting Categorized Batch OCR Processing")
    print("=" * 60)
    
    try:
        # Import the categorized OCR processor
        from categorized_batch_ocr import CategorizedPDFOCR
        
        # Define source directories (using absolute paths)
        base_path = Path(__file__).parent.parent.parent.parent  # Go up to ssu directory
        source_directories = [
            str(base_path / "Project/data/pdf/계약서"), 
            str(base_path / "Project/data/pdf/동의서"),
        ]
        
        # Verify directories exist
        print("📁 Checking source directories...")
        for directory in source_directories:
            if os.path.exists(directory):
                print(f"✅ Found: {directory}")
            else:
                print(f"⚠️ Not found: {directory}")
        
        # Initialize OCR processor
        print(f"\n🔄 Initializing OCR processor with {ocr_provider} provider...")
        output_dir = str(base_path / "Project/OCR/google_vision/ocr_results")
        ocr_processor = CategorizedPDFOCR(
            base_output_dir=output_dir, 
            ocr_provider=ocr_provider
        )
        
        # Show log file location
        log_file = ocr_processor.get_log_file_path()
        print(f"📝 Log file: {log_file}")
        print("💡 All processing details will be saved to this log file")
        
        # Process all PDFs
        print("\n🔄 Starting batch processing...")
        print("📊 This may take a while depending on the number and size of PDFs...")
        print("💡 Progress bars will show real-time status\n")
        
        results = ocr_processor.batch_process_categorized_pdfs(source_directories)
        
        # Display results
        print(f"\n🎉 Batch OCR Processing Complete!")
        print("=" * 60)
        print(f"📊 Processing Summary:")
        print(f"   OCR Provider: {ocr_provider}")
        print(f"   Total files found: {results['total_files']}")
        print(f"   Successfully processed: {results['processed_files']}")
        print(f"   Failed: {results['failed_files']}")
        print(f"   계약서 documents: {results['categories']['계약서']}")
        print(f"   동의서 documents: {results['categories']['동의서']}")
        print(f"\n📁 Results organized in: {ocr_processor.provider_output_dir}")
        print(f"📋 Summary file: {ocr_processor.base_output_dir}/batch_processing_summary.json")
        print(f"📝 Detailed log file: {log_file}")
        
        # Show output structure
        print(f"\n📂 Output Structure:")
        print(f"   ocr_results/")
        print(f"   └── {ocr_provider}_ocr/")
        print(f"       ├── 계약서/")
        print(f"       │   └── [PDF_NAME]/")
        print(f"       │       ├── converted_images/")
        print(f"       │       ├── page_results/")
        print(f"       │       ├── [PDF_NAME]_complete_ocr.json")
        print(f"       │       └── [PDF_NAME]_extracted_text.txt")
        print(f"       └── 동의서/")
        print(f"           └── [PDF_NAME]/")
        print(f"               ├── converted_images/")
        print(f"               ├── page_results/")
        print(f"               ├── [PDF_NAME]_complete_ocr.json")
        print(f"               └── [PDF_NAME]_extracted_text.txt")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required dependencies are installed:")
        print("   pip install PyMuPDF google-cloud-vision requests")
        return False
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    try:
        import fitz
        print("✅ PyMuPDF (fitz) - OK")
    except ImportError:
        print("❌ PyMuPDF (fitz) - Missing")
        return False
    
    try:
        from google.cloud import vision
        print("✅ Google Cloud Vision - OK")
    except ImportError:
        print("❌ Google Cloud Vision - Missing")
        return False
    
    try:
        from google.protobuf.json_format import MessageToDict
        print("✅ Google Protobuf - OK")
    except ImportError:
        print("❌ Google Protobuf - Missing")
        return False
    
    return True

def main():
    """Main function."""
    print("🚀 Categorized Batch OCR Processor")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them first:")
        print("   pip install PyMuPDF google-cloud-vision requests")
        return
    
    # Ask for OCR provider selection
    print("\n🔧 OCR Provider Selection:")
    print("1. Google Cloud Vision API (default)")
    print("2. Mistral OCR API")
    
    provider_choice = input("\nSelect OCR provider (1-2) [default: 1]: ").strip()
    
    ocr_provider = "google_cloud"
    
    if provider_choice == "2":
        ocr_provider = "mistral"
        print("✅ Using Mistral OCR API (reading API key from .env file)")
    else:
        print("✅ Using Google Cloud Vision API")
    
    # Ask for confirmation
    print(f"\n⚠️ This will process ALL PDF files using {ocr_provider} OCR.")
    print("   This may take a long time and use API credits.")
    
    response = input("\nDo you want to continue? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        success = run_batch_ocr(ocr_provider=ocr_provider)
        if success:
            print("\n✅ Processing completed successfully!")
        else:
            print("\n❌ Processing failed. Check the error messages above.")
    else:
        print("\n⏹️ Processing cancelled.")

if __name__ == "__main__":
    main()
