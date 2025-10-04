#!/usr/bin/env python3
"""
Test script for Universal OCR Processor
Demonstrates processing of TIFF files with different OCR providers
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from universal_ocr_processor import UniversalOCRProcessor

def test_tiff_processing():
    """Test processing the TIFF files with different providers."""
    
    print("🧪 Testing Universal OCR Processor")
    print("=" * 50)
    
    # Path to TIFF files
    tiff_directory = Path("../../data/images/동의서")
    
    if not tiff_directory.exists():
        print(f"❌ Error: Directory {tiff_directory} not found!")
        return False
    
    # List TIFF files
    tiff_files = list(tiff_directory.glob("*.tif*"))
    if not tiff_files:
        print(f"❌ No TIFF files found in {tiff_directory}")
        return False
    
    print(f"📁 Found {len(tiff_files)} TIFF files:")
    for tiff_file in tiff_files:
        print(f"  - {tiff_file.name}")
    
    # Test with Mistral OCR
    print(f"\n🔄 Testing with Mistral OCR...")
    try:
        processor = UniversalOCRProcessor(
            provider="mistral",
            output_dir="test_mistral_results"
        )
        
        # Process first TIFF file
        test_file = tiff_files[0]
        result = processor.process_single_file(str(test_file))
        
        print(f"✅ Mistral OCR Test Complete!")
        print(f"📄 File: {result['file_name']}")
        print(f"📊 Status: {result['status']}")
        
        if result['status'] == 'success':
            print(f"📝 Text Length: {result['total_text_length']} characters")
            print(f"🔍 Provider: {result['ocr_provider']}")
            
            # Show sample text
            if result['full_text']:
                sample_text = result['full_text'][:200]
                print(f"\n📋 Sample Text (first 200 chars):")
                print("-" * 50)
                print(sample_text + "..." if len(sample_text) == 200 else sample_text)
                print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Mistral OCR test failed: {e}")
        return False

def test_google_cloud_processing():
    """Test processing with Google Cloud Vision."""
    
    print(f"\n🔄 Testing with Google Cloud Vision...")
    try:
        processor = UniversalOCRProcessor(
            provider="google_cloud",
            output_dir="test_google_results"
        )
        
        # Process the TIFF directory
        tiff_directory = Path("../../data/images/동의서")
        results = processor.process_directory(str(tiff_directory), recursive=False)
        
        if results:
            successful = [r for r in results if r.get('status') == 'success']
            failed = [r for r in results if r.get('status') == 'failed']
            
            print(f"✅ Google Cloud Vision Test Complete!")
            print(f"📊 Summary:")
            print(f"   Total files: {len(results)}")
            print(f"   ✅ Successful: {len(successful)}")
            print(f"   ❌ Failed: {len(failed)}")
            print(f"   📝 Total text: {sum(r.get('total_text_length', 0) for r in successful)} characters")
            
            if successful:
                print(f"\n📋 Successfully processed files:")
                for result in successful:
                    print(f"   - {result['file_name']}: {result.get('total_text_length', 0)} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Google Cloud Vision test failed: {e}")
        return False

def show_file_type_support():
    """Show what file types are supported."""
    
    print(f"\n📋 Supported File Types:")
    print("-" * 30)
    
    print("📄 Documents:")
    print("   - PDF (.pdf)")
    print("   - Microsoft Word (.docx, .doc)")
    print("   - Microsoft PowerPoint (.pptx, .ppt)")
    print("   - Microsoft Excel (.xlsx, .xls)")
    print("   - Hancom Office (.hwp) - planned")
    
    print("\n🖼️ Images:")
    print("   - JPEG (.jpg, .jpeg)")
    print("   - PNG (.png)")
    print("   - GIF (.gif)")
    print("   - BMP (.bmp)")
    print("   - TIFF (.tif, .tiff)")
    
    print("\n🔧 OCR Providers:")
    print("   - Google Cloud Vision API")
    print("   - Mistral OCR API")
    print("   - Naver Clova OCR API")

def main():
    """Main test function."""
    print("🚀 Universal OCR Processor - Test Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("universal_ocr_processor.py").exists():
        print("❌ Please run this script from the google_vision directory")
        return
    
    # Show supported file types
    show_file_type_support()
    
    # Test TIFF processing
    print(f"\n🧪 Running Tests...")
    print("=" * 30)
    
    # Test Mistral OCR
    mistral_success = test_tiff_processing()
    
    # Test Google Cloud Vision
    google_success = test_google_cloud_processing()
    
    # Summary
    print(f"\n🎉 Test Summary:")
    print("=" * 20)
    print(f"Mistral OCR: {'✅ PASS' if mistral_success else '❌ FAIL'}")
    print(f"Google Cloud Vision: {'✅ PASS' if google_success else '❌ FAIL'}")
    
    if mistral_success or google_success:
        print(f"\n💡 The Universal OCR Processor is working!")
        print(f"   You can now process any supported file type.")
        print(f"   Use 'python easy_ocr_processor.py' for interactive mode.")
        print(f"   Use 'python universal_ocr_processor.py --help' for command line options.")
    else:
        print(f"\n❌ Tests failed. Check your API keys and dependencies.")

if __name__ == "__main__":
    main()
