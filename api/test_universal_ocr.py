#!/usr/bin/env python3
"""
Test script for Universal OCR Module
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_paths = [
    Path(__file__).parent / ".env",  # API directory
    Path(__file__).parent / ".env_alibaba",  # Alibaba specific
    Path(__file__).parent.parent / "OCR" / "google_vision" / ".env",  # OCR directory
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from: {env_path}")
        break
else:
    print("Warning: No .env file found. Using system environment variables only.")

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from module.ocr import UniversalOCRProcessor

def test_google_ocr():
    """Test Google Cloud Vision OCR"""
    print("Testing Google Cloud Vision OCR...")
    
    try:
        processor = UniversalOCRProcessor("google", "test_results/google")
        
        # Test with a sample PDF (if available)
        test_files = [
            "../샘플_저작물-20250812T232645Z-1-001_7.저작물양도계약서.pdf",
            "test.pdf",
            "sample.pdf"
        ]
        
        for test_file in test_files:
            if Path(test_file).exists():
                print(f"Processing: {test_file}")
                result = processor.process_single_file(test_file)
                print(f"Status: {result['status']}")
                print(f"Pages: {result.get('total_pages', 0)}")
                print(f"Text length: {result.get('total_text_length', 0)}")
                if result['status'] == 'success':
                    print("✅ Google OCR test successful")
                    return True
                else:
                    print(f"❌ Google OCR test failed: {result.get('error', 'Unknown error')}")
                    return False
        
        print("❌ No test files found")
        return False
        
    except Exception as e:
        print(f"❌ Google OCR test error: {e}")
        return False

def test_alibaba_ocr():
    """Test Alibaba Cloud OCR"""
    print("Testing Alibaba Cloud OCR...")
    
    try:
        # Check API key
        api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('ALIBABA_API_KEY')
        if not api_key or api_key == 'your_alibaba_api_key_here':
            print("❌ DASHSCOPE_API_KEY or ALIBABA_API_KEY not set")
            print("   Please set your API key in .env_alibaba file")
            return False
        
        processor = UniversalOCRProcessor("alibaba", "test_results/alibaba", "qwen-vl-plus")
        
        # Test with a sample PDF (if available)
        test_files = [
            "../샘플_저작물-20250812T232645Z-1-001_7.저작물양도계약서.pdf",
            "test.pdf",
            "sample.pdf"
        ]
        
        for test_file in test_files:
            if Path(test_file).exists():
                print(f"Processing: {test_file}")
                result = processor.process_single_file(test_file)
                print(f"Status: {result['status']}")
                print(f"Pages: {result.get('total_pages', 0)}")
                print(f"Text length: {result.get('total_text_length', 0)}")
                if result['status'] == 'success':
                    print("✅ Alibaba OCR test successful")
                    return True
                else:
                    print(f"❌ Alibaba OCR test failed: {result.get('error', 'Unknown error')}")
                    return False
        
        print("❌ No test files found")
        return False
        
    except Exception as e:
        print(f"❌ Alibaba OCR test error: {e}")
        return False

def test_streaming():
    """Test streaming OCR"""
    print("Testing streaming OCR...")
    
    try:
        # Check API key
        api_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('ALIBABA_API_KEY')
        if not api_key or api_key == 'your_alibaba_api_key_here':
            print("❌ DASHSCOPE_API_KEY or ALIBABA_API_KEY not set")
            print("   Please set your API key in .env_alibaba file")
            return False
        
        processor = UniversalOCRProcessor("alibaba", "test_results/streaming", "qwen-vl-plus")
        
        # Test with a sample PDF (if available)
        test_files = [
            "../샘플_저작물-20250812T232645Z-1-001_7.저작물양도계약서.pdf",
            "test.pdf",
            "sample.pdf"
        ]
        
        for test_file in test_files:
            if Path(test_file).exists():
                print(f"Streaming processing: {test_file}")
                chunk_count = 0
                total_text = ""
                
                for chunk in processor.process_single_file_streaming(test_file):
                    chunk_count += 1
                    total_text += chunk
                    if chunk_count <= 5:  # Show first 5 chunks
                        print(f"Chunk {chunk_count}: {chunk[:100]}...")
                
                print(f"Total chunks: {chunk_count}")
                print(f"Total text length: {len(total_text)}")
                print("✅ Streaming OCR test successful")
                return True
        
        print("❌ No test files found")
        return False
        
    except Exception as e:
        print(f"❌ Streaming OCR test error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Universal OCR Module Test Suite")
    print("=" * 60)
    
    # Create test results directory
    test_results_dir = Path("test_results")
    test_results_dir.mkdir(exist_ok=True)
    
    tests = [
        ("Google Cloud Vision", test_google_ocr),
        ("Alibaba Cloud", test_alibaba_ocr),
        ("Streaming", test_streaming)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
