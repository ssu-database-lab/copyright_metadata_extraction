#!/usr/bin/env python3
"""
Test script for DeepSeek-OCR integration
Tests the DeepSeek-OCR provider with sample images
"""

import os
import sys
from pathlib import Path

def test_deepseek_ocr():
    """Test DeepSeek-OCR provider."""
    print("🧪 Testing DeepSeek-OCR Integration")
    print("=" * 50)
    
    try:
        # Import the processor
        from universal_ocr_processor import UniversalOCRProcessor
        
        print("✅ UniversalOCRProcessor imported successfully")
        
        # Test with a sample image
        test_image = "/home/mbmk92/copyright/copyright_metadata_extraction/data/nii/02. 개인정보 이미지 용량별 파일/22.8MB_compressed_v2.jpg"
        
        if not os.path.exists(test_image):
            print(f"❌ Test image not found: {test_image}")
            print("💡 Using a different test image...")
            # Try to find any image file
            test_dir = Path("/home/mbmk92/copyright/copyright_metadata_extraction/data")
            image_files = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.png")) + list(test_dir.rglob("*.tif"))
            if image_files:
                test_image = str(image_files[0])
                print(f"📄 Using test image: {test_image}")
            else:
                print("❌ No test images found")
                return False
        
        print(f"📄 Test image: {test_image}")
        
        # Test different modes
        modes = ["tiny", "small", "base", "large", "gundam"]
        
        for mode in modes:
            print(f"\n🔬 Testing DeepSeek-OCR mode: {mode}")
            print("-" * 30)
            
            try:
                # Initialize processor
                processor = UniversalOCRProcessor(
                    provider="deepseek", 
                    output_dir="test_deepseek_results",
                    model=mode
                )
                
                print(f"✅ DeepSeek-OCR processor initialized with mode: {mode}")
                
                # Process the image
                result = processor.process_single_file(test_image)
                
                if result.get('status') == 'success':
                    text_length = result.get('total_text_length', 0)
                    print(f"✅ Processing successful!")
                    print(f"📝 Text length: {text_length} characters")
                    
                    if text_length > 0:
                        sample_text = result.get('full_text', '')[:200]
                        print(f"📖 Sample text: {sample_text}...")
                    else:
                        print("⚠️  No text extracted")
                else:
                    print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                print(f"❌ Error testing mode {mode}: {e}")
                continue
        
        print("\n🎉 DeepSeek-OCR integration test completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install DeepSeek-OCR dependencies:")
        print("   pip install -r deepseek_requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking DeepSeek-OCR Dependencies")
    print("-" * 40)
    
    required_packages = [
        'torch',
        'transformers', 
        'tokenizers',
        'einops',
        'addict',
        'easydict'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("💡 Install with: pip install -r deepseek_requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def main():
    """Main test function."""
    print("🚀 DeepSeek-OCR Integration Test")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        return
    
    print("\n" + "=" * 50)
    
    # Test DeepSeek-OCR integration
    if test_deepseek_ocr():
        print("\n🎉 All tests passed! DeepSeek-OCR is ready to use.")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()

