#!/usr/bin/env python3
"""
Simple test script to verify Google Cloud Vision API credentials are working.
"""

import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_credentials():
    """Test if Google Cloud credentials are working."""
    print("🔐 Testing Google Cloud Vision API Credentials")
    print("=" * 50)
    
    try:
        # Import the OCR class
        from pdf_document_ocr import PDFDocumentOCR
        print("✅ Successfully imported PDFDocumentOCR class")
        
        # Try to initialize the OCR processor
        print("🔄 Initializing OCR processor...")
        ocr_processor = PDFDocumentOCR(output_dir="test_credentials_results")
        print("✅ OCR processor initialized successfully")
        
        # Test if we can create a Vision API client
        print("🔄 Testing Vision API client creation...")
        client = ocr_processor.client
        print("✅ Vision API client created successfully")
        
        print("\n🎉 All credentials tests passed!")
        print("✅ Google Cloud Vision API is ready to use")
        print(f"📁 Output directory: {ocr_processor.output_dir}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Credentials file error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

def test_simple_ocr():
    """Test a simple OCR operation with a test image."""
    print("\n🧪 Testing Simple OCR Operation")
    print("=" * 50)
    
    try:
        from pdf_document_ocr import PDFDocumentOCR
        
        # Initialize OCR processor
        ocr_processor = PDFDocumentOCR(output_dir="test_credentials_results")
        
        # Check if there are any test images in the current directory
        import glob
        test_images = glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*.jpeg")
        
        if test_images:
            print(f"📸 Found test images: {test_images}")
            test_image = test_images[0]
            print(f"🔄 Testing OCR with: {test_image}")
            
            # Test OCR on the first image
            result = ocr_processor.ocr_image(test_image)
            print("✅ OCR test successful!")
            print(f"📝 Extracted text length: {result.get('text_length', 0)} characters")
            
            if result.get('extracted_text'):
                sample_text = result['extracted_text'][:100]
                print(f"📋 Sample text: {sample_text}...")
            
            return True
        else:
            print("⚠️ No test images found in current directory")
            print("💡 To test OCR, place a PNG, JPG, or JPEG image in this directory")
            return False
            
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Google Cloud Vision API Credentials Test Suite")
    print("=" * 60)
    
    # Test 1: Basic credentials
    print("\n1️⃣ Testing Basic Credentials...")
    creds_ok = test_credentials()
    
    if creds_ok:
        # Test 2: Simple OCR operation
        print("\n2️⃣ Testing OCR Operation...")
        ocr_ok = test_simple_ocr()
        
        if ocr_ok:
            print("\n🎉 All tests passed! Your setup is ready for PDF OCR processing.")
        else:
            print("\n⚠️ Credentials work but OCR test failed. Check your test images.")
    else:
        print("\n❌ Credentials test failed. Please check your setup.")
        print("\n🔧 Troubleshooting tips:")
        print("1. Ensure the credentials file exists in this directory")
        print("2. Check that the credentials file has the correct format")
        print("3. Verify that Vision API is enabled in your Google Cloud project")
        print("4. Check that your service account has the necessary permissions")

if __name__ == "__main__":
    main()
