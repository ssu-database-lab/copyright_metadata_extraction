#!/usr/bin/env python3
"""
Practical PDF Extraction Runner
This script shows you exactly how to run the PDF extraction on your data.
"""

import os
import sys
from pathlib import Path
import subprocess

def get_user_input():
    """Get user input for source and destination directories."""
    print("=== PDF Extraction Setup ===\n")
    
    # Get source directory (where your ZIP files are)
    print("Where are your compressed ZIP files located?")
    print("(e.g., /path/to/compressed/files or . for current directory)")
    
    while True:
        source_dir = input("Source directory: ").strip()
        if not source_dir:
            source_dir = "."
        
        source_path = Path(source_dir).resolve()
        if source_path.exists():
            # Check if it contains ZIP files
            zip_files = list(source_path.glob("*.zip"))
            if zip_files:
                print(f"✅ Found {len(zip_files)} ZIP files in {source_path}")
                break
            else:
                print(f"❌ No ZIP files found in {source_path}")
                print("Please check the directory path and try again.")
        else:
            print(f"❌ Directory {source_path} does not exist")
            print("Please check the path and try again.")
    
    # Get destination directory
    print(f"\nWhere should the extracted PDFs be saved?")
    print("(e.g., /path/to/output/pdfs)")
    
    while True:
        dest_dir = input("Destination directory: ").strip()
        if not dest_dir:
            dest_dir = "./extracted_pdfs"
        
        dest_path = Path(dest_dir).resolve()
        
        # Create directory if it doesn't exist
        try:
            dest_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Output directory: {dest_path}")
            break
        except Exception as e:
            print(f"❌ Cannot create directory {dest_path}: {e}")
            print("Please check permissions and try again.")
    
    return source_path, dest_path

def run_dry_run(source_dir, dest_dir):
    """Run a dry-run to see what PDFs are available."""
    print(f"\n=== Dry Run - Exploring Archives ===")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print("-" * 50)
    
    try:
        # Run the simple extractor in dry-run mode
        cmd = [
            sys.executable, "simple_extract.py",
            str(source_dir), str(dest_dir), "--dry-run"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dry run completed successfully!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("❌ Dry run failed!")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running dry run: {e}")
        return False
    
    return True

def run_extraction(source_dir, dest_dir):
    """Run the actual PDF extraction."""
    print(f"\n=== Running PDF Extraction ===")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print("-" * 50)
    
    # Ask user if they want to proceed
    response = input("Proceed with PDF extraction? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Extraction cancelled.")
        return False
    
    try:
        # Run the simple extractor
        cmd = [
            sys.executable, "simple_extract.py",
            str(source_dir), str(dest_dir)
        ]
        
        print("Starting extraction... This may take a while for large collections.")
        print("Press Ctrl+C to stop at any time.\n")
        
        # Run with real-time output
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ PDF extraction completed successfully!")
            return True
        else:
            print("\n❌ PDF extraction failed!")
            return False
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Extraction interrupted by user.")
        return False
    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        return False

def show_next_steps():
    """Show what to do next after extraction."""
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS FOR METADATA EXTRACTION")
    print("=" * 60)
    
    print("\n1. 📊 Review Extracted PDFs:")
    print("   - Check the output directory for extracted PDFs")
    print("   - Verify file counts and organization")
    
    print("\n2. 🔍 Install Metadata Extraction Libraries:")
    print("   pip install PyPDF2 pdfplumber pdf2image")
    
    print("\n3. 📝 Extract Metadata:")
    print("   - Use PyPDF2 for basic metadata (author, title, dates)")
    print("   - Use pdfplumber for text content extraction")
    print("   - Use pdf2image for OCR processing if needed")
    
    print("\n4. 🗄️  Organize Results:")
    print("   - Create a database or spreadsheet of metadata")
    print("   - Categorize documents by type or content")
    print("   - Plan your analysis workflow")
    
    print("\n5. 📈 Scale Up:")
    print("   - Use batch_extract.py for larger collections")
    print("   - Implement parallel processing for metadata extraction")
    print("   - Set up automated workflows")

def main():
    """Main function."""
    print("🚀 PDF Extraction Tool - Ready to Process Your 200GB Collection!\n")
    
    try:
        # Get user input
        source_dir, dest_dir = get_user_input()
        
        # Run dry-run first
        if not run_dry_run(source_dir, dest_dir):
            print("❌ Dry run failed. Please check your setup and try again.")
            return
        
        # Run actual extraction
        if run_extraction(source_dir, dest_dir):
            show_next_steps()
        else:
            print("❌ Extraction failed. Please check the errors above.")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
