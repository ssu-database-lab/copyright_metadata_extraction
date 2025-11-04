#!/usr/bin/env python3
"""
Test script for Alibaba Cloud OCR with file size checking and pixel parameter handling.
This script demonstrates the enhanced file size validation and image scaling features.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from universal_ocr_processor import UniversalOCRProcessor

def test_file_size_checking():
    """Test the file size checking functionality."""
    print("🔍 Testing Alibaba Cloud OCR File Size Checking")
    print("=" * 60)
    
    # Initialize processor
    try:
        processor = UniversalOCRProcessor(provider="alibaba", output_dir="test_results")
        print("✅ Alibaba Cloud OCR processor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        return
    
    # Test with the large file mentioned by the user
    large_file = "/home/mbmk92/copyright/copyright_metadata_extraction/data/nii/02. 개인정보 이미지 용량별 파일/22.8MB.tif"
    
    if os.path.exists(large_file):
        print(f"\n📁 Testing with large file: {Path(large_file).name}")
        
        # Check file size using the provider's method
        file_size_info = processor.ocr_provider._check_file_size(large_file)
        dimension_info = processor.ocr_provider._get_image_dimensions(large_file)
        
        print(f"📊 File Size Information:")
        print(f"   Size: {file_size_info['file_size_mb']} MB")
        print(f"   Within limit: {file_size_info['within_limit']}")
        print(f"   Recommendation: {file_size_info['recommendation']}")
        
        print(f"\n📐 Image Dimension Information:")
        print(f"   Dimensions: {dimension_info['width']} x {dimension_info['height']}")
        print(f"   Total pixels: {dimension_info['total_pixels']:,}")
        print(f"   Aspect ratio: {dimension_info['aspect_ratio']}")
        print(f"   Within pixel limits: {dimension_info['within_pixel_limits']}")
        
        print(f"\n⚙️ Processing Parameters:")
        print(f"   Min pixels: {processor.ocr_provider.min_pixels:,}")
        print(f"   Max pixels: {processor.ocr_provider.max_pixels:,}")
        print(f"   Max file size: {processor.ocr_provider.MAX_FILE_SIZE_MB} MB")
        
        # Try to process the file (this should fail due to size)
        print(f"\n🔄 Attempting to process the large file...")
        try:
            result = processor.process_single_file(large_file)
            if result.get('status') == 'error':
                print(f"❌ Processing failed as expected: {result.get('error')}")
            else:
                print(f"✅ Processing succeeded unexpectedly!")
        except Exception as e:
            print(f"❌ Processing failed with exception: {e}")
    else:
        print(f"❌ Large file not found: {large_file}")
    
    # Test with a smaller file if available
    print(f"\n📁 Testing with smaller files...")
    test_files = [
        "/home/mbmk92/copyright/copyright_metadata_extraction/data/nii/01. 개인정보 이미지 파일",
        "/home/mbmk92/copyright/copyright_metadata_extraction/data/pdf/계약서"
    ]
    
    for test_dir in test_files:
        if os.path.exists(test_dir):
            print(f"\n🔍 Checking files in: {test_dir}")
            for file_path in Path(test_dir).rglob("*.tif"):
                if file_path.stat().st_size < 10 * 1024 * 1024:  # Less than 10MB
                    print(f"   📄 {file_path.name}: {file_path.stat().st_size / (1024*1024):.2f} MB")
                    break

def test_environment_configuration():
    """Test environment variable configuration."""
    print(f"\n🔧 Environment Configuration Test")
    print("=" * 40)
    
    env_vars = [
        'DASHSCOPE_API_KEY',
        'ALIBABA_API_KEY', 
        'ALIBABA_REGION',
        'ALIBABA_MODEL',
        'ALIBABA_TEMPERATURE',
        'ALIBABA_TOP_P',
        'ALIBABA_MIN_PIXELS',
        'ALIBABA_MAX_PIXELS'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            if 'KEY' in var:
                print(f"✅ {var}: {'*' * 10} (hidden)")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")

def show_recommendations():
    """Show recommendations for handling large files."""
    print(f"\n💡 Recommendations for Large Files")
    print("=" * 40)
    
    recommendations = [
        "1. Compress the image using image processing tools",
        "2. Split large images into smaller sections",
        "3. Reduce image resolution while maintaining readability",
        "4. Use alternative OCR providers for very large files",
        "5. Consider using the API Client approach with base64 encoding",
        "6. Adjust min_pixels and max_pixels parameters for better scaling"
    ]
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print(f"\n📝 Environment Variables for Fine-tuning:")
    print(f"   ALIBABA_MIN_PIXELS=3136    # Minimum pixels (default)")
    print(f"   ALIBABA_MAX_PIXELS=6422528 # Maximum pixels (default)")
    print(f"   ALIBABA_TEMPERATURE=1.0    # Randomness control")
    print(f"   ALIBABA_TOP_P=0.8          # Nucleus sampling")

def main():
    """Main test function."""
    print("🚀 Alibaba Cloud OCR File Size Testing Suite")
    print("=" * 60)
    
    # Test environment configuration
    test_environment_configuration()
    
    # Test file size checking
    test_file_size_checking()
    
    # Show recommendations
    show_recommendations()
    
    print(f"\n✅ Testing complete!")
    print(f"📚 For more information, see:")
    print(f"   - https://www.alibabacloud.com/help/en/model-studio/qwen-vl-ocr")
    print(f"   - https://www.alibabacloud.com/help/en/model-studio/vision")

if __name__ == "__main__":
    main()
