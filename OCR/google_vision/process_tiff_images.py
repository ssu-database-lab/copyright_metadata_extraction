#!/usr/bin/env python3
"""
Process TIFF Images in data/images/동의서 Directory
Simple script to OCR the specific TIFF files you have
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tiff_ocr_processor import TIFFOCRProcessor

def process_consent_forms():
    """Process the TIFF consent forms in the data/images/동의서 directory."""
    print("🚀 Processing TIFF Consent Forms")
    print("=" * 50)
    
    # Path to your TIFF images
    tiff_directory = Path("../../data/images/동의서")
    
    if not tiff_directory.exists():
        print(f"❌ Error: Directory {tiff_directory} not found!")
        print("Make sure you're running this from the OCR/google_vision directory")
        return False
    
    # List TIFF files
    tiff_files = list(tiff_directory.glob("*.tif*"))
    if not tiff_files:
        print(f"❌ No TIFF files found in {tiff_directory}")
        return False
    
    print(f"📁 Found {len(tiff_files)} TIFF files:")
    for tiff_file in tiff_files:
        print(f"  - {tiff_file.name}")
    
    # Choose OCR provider
    print("\n🔧 Choose OCR Provider:")
    print("1. Google Cloud Vision API (recommended for Korean text)")
    print("2. Mistral OCR API")
    
    choice = input("Select provider (1-2): ").strip()
    
    if choice == "1":
        provider = "google_cloud"
        print("✅ Using Google Cloud Vision API")
    elif choice == "2":
        provider = "mistral"
        print("✅ Using Mistral OCR API")
    else:
        print("Invalid choice. Using Google Cloud Vision API.")
        provider = "google_cloud"
    
    try:
        # Initialize processor
        processor = TIFFOCRProcessor(
            provider=provider,
            output_dir=f"consent_forms_{provider}_ocr_results"
        )
        
        print(f"\n🔄 Processing TIFF files...")
        
        # Process all TIFF files
        results = processor.process_tiff_directory(str(tiff_directory))
        
        # Display results
        print(f"\n✅ Processing Complete!")
        print(f"📊 Total files: {len(results)}")
        
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        if successful:
            print(f"\n📋 Successfully processed files:")
            for result in successful:
                text_length = result.get('text_length', 0)
                print(f"  - {result['image_name']}: {text_length} characters")
        
        if failed:
            print(f"\n❌ Failed files:")
            for result in failed:
                print(f"  - {result['image_name']}: {result['error']}")
        
        print(f"\n💾 Results saved to: {processor.output_dir}")
        
        # Show sample text from first successful result
        if successful and successful[0].get('extracted_text'):
            sample_text = successful[0]['extracted_text'][:300]
            print(f"\n📋 Sample Text from {successful[0]['image_name']} (first 300 chars):")
            print("-" * 50)
            print(sample_text + "..." if len(sample_text) == 300 else sample_text)
            print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing TIFF files: {e}")
        return False

def process_single_tiff():
    """Process a single TIFF file."""
    print("🚀 Process Single TIFF File")
    print("=" * 50)
    
    tiff_path = input("Enter the path to your TIFF file: ").strip()
    
    if not os.path.exists(tiff_path):
        print(f"❌ Error: File {tiff_path} not found!")
        return False
    
    # Choose OCR provider
    print("\n🔧 Choose OCR Provider:")
    print("1. Google Cloud Vision API")
    print("2. Mistral OCR API")
    
    choice = input("Select provider (1-2): ").strip()
    provider = "mistral" if choice == "2" else "google_cloud"
    
    try:
        processor = TIFFOCRProcessor(provider=provider)
        result = processor.process_single_tiff(tiff_path)
        
        print(f"\n✅ Processing Complete!")
        print(f"📄 File: {result['image_name']}")
        print(f"📝 Text Length: {result.get('text_length', 0)} characters")
        print(f"💾 Results saved to: {processor.output_dir}")
        
        # Show sample text
        if result.get('extracted_text'):
            sample_text = result['extracted_text'][:300]
            print(f"\n📋 Sample Text (first 300 chars):")
            print("-" * 50)
            print(sample_text + "..." if len(sample_text) == 300 else sample_text)
            print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False

def main():
    """Main function with menu options."""
    print("🚀 TIFF OCR Processor")
    print("Process TIFF images using Google Cloud Vision or Mistral OCR")
    print("=" * 60)
    
    while True:
        print("\n📋 Options:")
        print("1. Process consent forms in data/images/동의서")
        print("2. Process single TIFF file")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            process_consent_forms()
        elif choice == "2":
            process_single_tiff()
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-3.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()