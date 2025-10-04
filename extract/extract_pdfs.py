#!/usr/bin/env python3
"""
PDF Extractor from Compressed Archives
Extracts only PDF files from multiple ZIP archives without decompressing everything.
"""

import os
import zipfile
import shutil
from pathlib import Path
import argparse
from typing import List, Dict
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organization
        self.pdfs_dir = self.output_dir / "extracted_pdfs"
        self.pdfs_dir.mkdir(exist_ok=True)
        
        self.logs_dir = self.output_dir / "extraction_logs"
        self.logs_dir.mkdir(exist_ok=True)
        
    def find_zip_files(self) -> List[Path]:
        """Find all ZIP files in the source directory."""
        zip_files = list(self.source_dir.glob("*.zip"))
        logger.info(f"Found {len(zip_files)} ZIP files")
        return zip_files
    
    def list_archive_contents(self, zip_path: Path) -> Dict[str, List[str]]:
        """List contents of a ZIP file and categorize files."""
        contents = {
            'pdfs': [],
            'other_files': [],
            'total_files': 0
        }
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                contents['total_files'] = len(file_list)
                
                for file_path in file_list:
                    if file_path.lower().endswith('.pdf'):
                        contents['pdfs'].append(file_path)
                    else:
                        contents['other_files'].append(file_path)
                        
        except Exception as e:
            logger.error(f"Error reading {zip_path}: {e}")
            
        return contents
    
    def extract_pdfs_from_archive(self, zip_path: Path, pdf_files: List[str]) -> int:
        """Extract PDF files from a specific ZIP archive."""
        extracted_count = 0
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for pdf_file in pdf_files:
                    try:
                        # Create output path
                        output_path = self.pdfs_dir / f"{zip_path.stem}_{pdf_file.split('/')[-1]}"
                        
                        # Extract the PDF
                        with zip_ref.open(pdf_file) as source, open(output_path, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        
                        extracted_count += 1
                        logger.info(f"Extracted: {pdf_file} -> {output_path}")
                        
                    except Exception as e:
                        logger.error(f"Error extracting {pdf_file}: {e}")
                        
        except Exception as e:
            logger.error(f"Error opening {zip_path}: {e}")
            
        return extracted_count
    
    def process_all_archives(self):
        """Process all ZIP files and extract PDFs."""
        zip_files = self.find_zip_files()
        
        if not zip_files:
            logger.warning("No ZIP files found in source directory")
            return
        
        total_pdfs_found = 0
        total_pdfs_extracted = 0
        
        # Create summary log
        summary_log = self.logs_dir / "extraction_summary.txt"
        
        with open(summary_log, 'w', encoding='utf-8') as f:
            f.write("PDF Extraction Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for zip_path in zip_files:
                logger.info(f"Processing: {zip_path.name}")
                
                # List contents
                contents = self.list_archive_contents(zip_path)
                pdf_count = len(contents['pdfs'])
                total_pdfs_found += pdf_count
                
                # Log archive info
                f.write(f"Archive: {zip_path.name}\n")
                f.write(f"  Total files: {contents['total_files']}\n")
                f.write(f"  PDF files: {pdf_count}\n")
                f.write(f"  Other files: {len(contents['other_files'])}\n")
                
                if pdf_count > 0:
                    f.write(f"  PDF files found:\n")
                    for pdf in contents['pdfs']:
                        f.write(f"    - {pdf}\n")
                    
                    # Extract PDFs
                    extracted = self.extract_pdfs_from_archive(zip_path, contents['pdfs'])
                    total_pdfs_extracted += extracted
                    f.write(f"  PDFs extracted: {extracted}\n")
                
                f.write("\n")
                
            # Final summary
            f.write("=" * 50 + "\n")
            f.write(f"Total PDFs found across all archives: {total_pdfs_found}\n")
            f.write(f"Total PDFs extracted: {total_pdfs_extracted}\n")
            f.write(f"Extraction completed at: {datetime.now()}\n")
        
        logger.info(f"Extraction complete! Summary saved to: {summary_log}")
        logger.info(f"Total PDFs found: {total_pdfs_found}")
        logger.info(f"Total PDFs extracted: {total_pdfs_extracted}")
        logger.info(f"PDFs saved to: {self.pdfs_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract PDF files from ZIP archives')
    parser.add_argument('source_dir', help='Directory containing ZIP files')
    parser.add_argument('output_dir', help='Output directory for extracted PDFs')
    parser.add_argument('--dry-run', action='store_true', help='List contents without extracting')
    
    args = parser.parse_args()
    
    extractor = PDFExtractor(args.source_dir, args.output_dir)
    
    if args.dry_run:
        # Just list contents without extracting
        zip_files = extractor.find_zip_files()
        for zip_path in zip_files:
            print(f"\nArchive: {zip_path.name}")
            contents = extractor.list_archive_contents(zip_path)
            print(f"  Total files: {contents['total_files']}")
            print(f"  PDF files: {len(contents['pdfs'])}")
            if contents['pdfs']:
                print("  PDF files found:")
                for pdf in contents['pdfs']:
                    print(f"    - {pdf}")
    else:
        # Extract PDFs
        extractor.process_all_archives()

if __name__ == "__main__":
    main()
