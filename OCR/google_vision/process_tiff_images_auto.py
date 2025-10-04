#!/usr/bin/env python3
"""
Non-interactive script to process TIFF images using Mistral OCR.
Processes the attached TIFF images from the 동의서 directory.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def process_tiff_images():
    """Process the TIFF images using Mistral OCR."""
    
    print("🚀 Processing TIFF Images with Mistral OCR")
    print("=" * 50)
    
    try:
        # Import the OCR processor
        from categorized_batch_ocr import CategorizedPDFOCR
        
        # Initialize OCR processor with Mistral
        ocr_processor = CategorizedPDFOCR(
            base_output_dir="ocr_results",
            ocr_provider="mistral"
        )
        
        # Define TIFF image paths
        tiff_files = [
            "/home/mbmk92/copyright/Project/data/images/동의서/공공저작물 자유이용허락 동의서(박동우).tif",
            "/home/mbmk92/copyright/Project/data/images/동의서/공공저작물 자유이용허락 동의서(정경희).tif"
        ]
        
        # Check if files exist
        print("📁 Checking TIFF files...")
        existing_files = []
        for tiff_file in tiff_files:
            if os.path.exists(tiff_file):
                print(f"✅ Found: {Path(tiff_file).name}")
                existing_files.append(tiff_file)
            else:
                print(f"❌ Not found: {tiff_file}")
        
        if not existing_files:
            print("❌ No TIFF files found to process!")
            return
        
        # Process each TIFF image
        results = []
        for i, tiff_file in enumerate(existing_files, 1):
            print(f"\n🔄 Processing image {i}/{len(existing_files)}: {Path(tiff_file).name}")
            
            try:
                # Process the image
                result = ocr_processor.ocr_image(tiff_file)
                
                # Add file information
                result['file_name'] = Path(tiff_file).name
                result['file_path'] = tiff_file
                result['processing_timestamp'] = datetime.now().isoformat()
                
                results.append(result)
                
                print(f"✅ Successfully processed!")
                print(f"   📝 Extracted text length: {result['text_length']} characters")
                print(f"   🔍 Provider: {result['provider']}")
                
                # Show first 200 characters of extracted text
                if result['extracted_text']:
                    print(f"   📋 Sample text: {result['extracted_text'][:200]}...")
                
            except Exception as e:
                print(f"❌ Error processing {Path(tiff_file).name}: {e}")
                results.append({
                    'file_name': Path(tiff_file).name,
                    'file_path': tiff_file,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Save results
        output_file = "ocr_results/tiff_ocr_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 Processing Complete!")
        print(f"📊 Summary:")
        print(f"   Total files: {len(existing_files)}")
        print(f"   Successfully processed: {len([r for r in results if 'error' not in r])}")
        print(f"   Failed: {len([r for r in results if 'error' in r])}")
        print(f"📁 Results saved to: {output_file}")
        
        # Save individual text files
        for result in results:
            if 'error' not in result and result['extracted_text']:
                text_file = f"ocr_results/{result['file_name']}_extracted_text.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(result['extracted_text'])
                print(f"📄 Text saved to: {text_file}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all required dependencies are installed:")
        print("   pip install PyMuPDF google-cloud-vision mistralai python-dotenv")
        return False
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TIFF Image OCR Processor")
    print("=" * 30)
    
    # Check if we're in the right directory
    if not Path("categorized_batch_ocr.py").exists():
        print("❌ Please run this script from the google_vision directory")
        sys.exit(1)
    
    print("\n⚠️ Processing TIFF images using Mistral OCR API...")
    print("   This will use API credits.")
    
    success = process_tiff_images()
    if success:
        print("\n✅ Processing completed successfully!")
    else:
        print("\n❌ Processing failed. Check the error messages above.")
        sys.exit(1)
