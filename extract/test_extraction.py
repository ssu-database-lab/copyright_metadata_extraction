#!/usr/bin/env python3
"""
Test script to demonstrate PDF extraction functionality.
This creates a sample ZIP file with mixed content and then extracts only the PDFs.
"""

import zipfile
import tempfile
import os
from pathlib import Path
from simple_extract import extract_pdfs_from_zip

def create_test_zip():
    """Create a test ZIP file with mixed content."""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create some test files
    test_files = {
        "document.pdf": b"%PDF-1.4\nTest PDF content",
        "image.jpg": b"fake jpeg data",
        "spreadsheet.xlsx": b"fake excel data",
        "another_document.pdf": b"%PDF-1.4\nAnother test PDF",
        "video.mp4": b"fake video data",
        "text.txt": b"plain text content"
    }
    
    # Create the test files
    for filename, content in test_files.items():
        with open(test_dir / filename, 'wb') as f:
            f.write(content)
    
    # Create ZIP file
    zip_path = test_dir / "test_archive.zip"
    with zipfile.ZipFile(zip_path, 'w') as zip_ref:
        for filename in test_files.keys():
            zip_ref.write(test_dir / filename, filename)
    
    print(f"Created test ZIP file: {zip_path}")
    print(f"Contains: {list(test_files.keys())}")
    
    return zip_path

def test_extraction():
    """Test the PDF extraction functionality."""
    print("=== PDF Extraction Test ===\n")
    
    # Create test data
    zip_path = create_test_zip()
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nExtracting PDFs from: {zip_path}")
    print(f"Output directory: {output_dir}")
    
    # Extract PDFs
    extracted_count = extract_pdfs_from_zip(zip_path, output_dir)
    
    print(f"\nExtraction complete!")
    print(f"PDFs extracted: {extracted_count}")
    
    # List extracted files
    if output_dir.exists():
        extracted_files = list(output_dir.glob("*.pdf"))
        print(f"\nExtracted PDF files:")
        for pdf_file in extracted_files:
            print(f"  - {pdf_file.name}")
    
    # Cleanup
    print(f"\nCleaning up test files...")
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_extraction()
