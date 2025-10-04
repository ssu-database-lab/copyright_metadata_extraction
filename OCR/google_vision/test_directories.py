#!/usr/bin/env python3
"""
Test script to verify directory paths and check what PDFs are available.
"""

import os
from pathlib import Path

def test_directories():
    """Test if the directories exist and show what PDFs are available."""
    
    print("🔍 Testing Directory Paths")
    print("=" * 50)
    
    # Get the base path (go up to ssu directory)
    base_path = Path(__file__).parent.parent.parent.parent  # Go up to ssu directory
    print(f"Base path: {base_path}")
    
    # Define source directories
    source_directories = [
        str(base_path / "Project/data/pdf/계약서"), 
        str(base_path / "Project/data/pdf/동의서"),
        # str(base_path / "Project/OCR/document/동의서")
    ]
    
    total_pdfs = 0
    
    for directory in source_directories:
        print(f"\n📁 Checking: {directory}")
        
        if os.path.exists(directory):
            print(f"✅ Directory exists")
            
            # Count PDFs in this directory
            pdf_count = 0
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_count += 1
                        print(f"   📄 {os.path.join(root, file)}")
            
            print(f"   📊 Found {pdf_count} PDF files")
            total_pdfs += pdf_count
            
        else:
            print(f"❌ Directory not found")
    
    print(f"\n📊 Total PDFs found: {total_pdfs}")
    
    # Test output directory
    output_dir = str(base_path / "Project/OCR/google_vision/ocr_results")
    print(f"\n📁 Output directory: {output_dir}")
    
    if os.path.exists(output_dir):
        print("✅ Output directory exists")
    else:
        print("⚠️ Output directory will be created")
    
    return total_pdfs > 0

def main():
    """Main function."""
    print("🚀 Directory Path Test")
    print("=" * 30)
    
    has_pdfs = test_directories()
    
    if has_pdfs:
        print("\n✅ Ready to run batch OCR processing!")
        print("💡 Run: python run_batch_ocr.py")
    else:
        print("\n⚠️ No PDF files found in the specified directories.")
        print("💡 Check if the directories contain PDF files.")

if __name__ == "__main__":
    main()
