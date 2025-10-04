#!/usr/bin/env python3
"""
Batch PDF Extractor for Large Collections
Processes ZIP files in batches to manage memory and provide progress updates.
"""

import zipfile
import os
from pathlib import Path
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class BatchPDFExtractor:
    def __init__(self, source_dir, output_dir, batch_size=10, max_workers=4):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Thread-safe counters
        self.lock = threading.Lock()
        self.total_processed = 0
        self.total_pdfs_found = 0
        self.total_pdfs_extracted = 0
        
    def find_zip_files(self):
        """Find all ZIP files in the source directory."""
        zip_files = list(self.source_dir.glob("*.zip"))
        zip_files.sort()  # Process in consistent order
        return zip_files
    
    def process_single_archive(self, zip_path):
        """Process a single ZIP archive and extract PDFs."""
        pdf_count = 0
        extracted_count = 0
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get list of all files
                file_list = zip_ref.namelist()
                
                # Filter for PDF files
                pdf_files = [f for f in file_list if f.lower().endswith('.pdf')]
                pdf_count = len(pdf_files)
                
                # Extract each PDF
                for pdf_file in pdf_files:
                    try:
                        # Create output filename
                        archive_name = zip_path.stem
                        pdf_name = Path(pdf_file).name
                        output_filename = f"{archive_name}_{pdf_name}"
                        output_path = self.output_dir / output_filename
                        
                        # Extract the PDF
                        with zip_ref.open(pdf_file) as source, open(output_path, 'wb') as target:
                            target.write(source.read())
                        
                        extracted_count += 1
                        
                    except Exception as e:
                        print(f"  Error extracting {pdf_file} from {zip_path.name}: {e}")
                        
        except Exception as e:
            print(f"Error processing {zip_path.name}: {e}")
        
        # Update counters thread-safely
        with self.lock:
            self.total_pdfs_found += pdf_count
            self.total_pdfs_extracted += extracted_count
            self.total_processed += 1
        
        return {
            'archive': zip_path.name,
            'pdfs_found': pdf_count,
            'pdfs_extracted': extracted_count
        }
    
    def process_batch(self, batch):
        """Process a batch of ZIP files."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files in the batch
            future_to_zip = {executor.submit(self.process_single_archive, zip_path): zip_path 
                           for zip_path in batch}
            
            # Process completed tasks
            for future in as_completed(future_to_zip):
                zip_path = future_to_zip[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Progress update
                    progress = (self.total_processed / len(self.find_zip_files())) * 100
                    print(f"Progress: {progress:.1f}% - {result['archive']}: "
                          f"{result['pdfs_extracted']}/{result['pdfs_found']} PDFs extracted")
                    
                except Exception as e:
                    print(f"Error processing {zip_path.name}: {e}")
        
        return results
    
    def run(self):
        """Run the batch extraction process."""
        zip_files = self.find_zip_files()
        
        if not zip_files:
            print("No ZIP files found in source directory")
            return
        
        print(f"Found {len(zip_files)} ZIP files")
        print(f"Processing in batches of {self.batch_size} with {self.max_workers} workers")
        print(f"Output directory: {self.output_dir}")
        print("-" * 60)
        
        start_time = time.time()
        
        # Process in batches
        for i in range(0, len(zip_files), self.batch_size):
            batch = zip_files[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(zip_files) + self.batch_size - 1) // self.batch_size
            
            print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} files)")
            print("-" * 40)
            
            self.process_batch(batch)
            
            # Batch summary
            elapsed = time.time() - start_time
            avg_time_per_archive = elapsed / self.total_processed if self.total_processed > 0 else 0
            remaining_archives = len(zip_files) - self.total_processed
            estimated_remaining_time = remaining_archives * avg_time_per_archive
            
            print(f"\nBatch {batch_num} complete!")
            print(f"Processed: {self.total_processed}/{len(zip_files)} archives")
            print(f"PDFs found: {self.total_pdfs_found}")
            print(f"PDFs extracted: {self.total_pdfs_extracted}")
            print(f"Elapsed time: {elapsed:.1f}s")
            print(f"Estimated remaining time: {estimated_remaining_time:.1f}s")
        
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE!")
        print("=" * 60)
        print(f"Total archives processed: {self.total_processed}")
        print(f"Total PDFs found: {self.total_pdfs_found}")
        print(f"Total PDFs extracted: {self.total_pdfs_extracted}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per archive: {total_time/self.total_processed:.2f}s")
        print(f"PDFs saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Batch extract PDF files from ZIP archives')
    parser.add_argument('source_dir', help='Directory containing ZIP files')
    parser.add_argument('output_dir', help='Output directory for extracted PDFs')
    parser.add_argument('--batch-size', type=int, default=10, 
                       help='Number of archives to process in each batch (default: 10)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of worker threads (default: 4)')
    
    args = parser.parse_args()
    
    extractor = BatchPDFExtractor(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_workers=args.max_workers
    )
    
    extractor.run()

if __name__ == "__main__":
    main()
