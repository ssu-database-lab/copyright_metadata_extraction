#!/usr/bin/env python3
"""
Simple PDF Extractor from ZIP files
Extracts only PDF files without decompressing everything.
"""

import zipfile
import os
from pathlib import Path
import argparse

def extract_pdfs_from_zip(zip_path, output_dir):
    """Extract PDF files from a single ZIP file."""
    pdf_count = 0
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of all files in the ZIP
            file_list = zip_ref.namelist()
            
            # Filter for PDF files only
            pdf_files = [f for f in file_list if f.lower().endswith('.pdf')]
            
            print(f"Found {len(pdf_files)} PDF files in {zip_path.name}")
            
            # Extract each PDF
            for pdf_file in pdf_files:
                try:
                    # Create output filename (archive_name + pdf_name)
                    archive_name = Path(zip_path).stem
                    pdf_name = Path(pdf_file).name
                    output_filename = f"{archive_name}_{pdf_name}"
                    output_path = Path(output_dir) / output_filename
                    
                    # Extract the PDF
                    with zip_ref.open(pdf_file) as source, open(output_path, 'wb') as target:
                        target.write(source.read())
                    
                    pdf_count += 1
                    print(f"  Extracted: {pdf_file} -> {output_filename}")
                    
                except Exception as e:
                    print(f"  Error extracting {pdf_file}: {e}")
                    
    except Exception as e:
        print(f"Error reading {zip_path}: {e}")
    
    return pdf_count

def main():
    parser = argparse.ArgumentParser(description='Extract PDF files from ZIP archives')
    parser.add_argument('source_dir', help='Directory containing ZIP files')
    parser.add_argument('output_dir', help='Output directory for extracted PDFs')
    parser.add_argument('--dry-run', action='store_true', help='List contents without extracting')
    
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Find all ZIP files
    zip_files = list(source_dir.glob("*.zip"))
    
    if not zip_files:
        print("No ZIP files found in source directory")
        return
    
    print(f"Found {len(zip_files)} ZIP files")
    
    if args.dry_run:
        # Just list contents
        for zip_path in zip_files:
            print(f"\nArchive: {zip_path.name}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    pdf_files = [f for f in file_list if f.lower().endswith('.pdf')]
                    print(f"  Total files: {len(file_list)}")
                    print(f"  PDF files: {len(pdf_files)}")
                    if pdf_files:
                        print("  PDF files found:")
                        for pdf in pdf_files:
                            print(f"    - {pdf}")
            except Exception as e:
                print(f"  Error reading archive: {e}")
    else:
        # Extract PDFs
        total_extracted = 0
        for zip_path in zip_files:
            print(f"\nProcessing: {zip_path.name}")
            extracted = extract_pdfs_from_zip(zip_path, output_dir)
            total_extracted += extracted
        
        print(f"\nExtraction complete! Total PDFs extracted: {total_extracted}")
        print(f"PDFs saved to: {output_dir}")

if __name__ == "__main__":
    main()
