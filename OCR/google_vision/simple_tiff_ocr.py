#!/usr/bin/env python3
"""
Simple TIFF OCR using existing infrastructure
Uses the existing categorized_batch_ocr.py which already supports TIFF
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def process_tiff_with_existing_code():
    """Process TIFF files using the existing categorized_batch_ocr.py"""
    print("🚀 Processing TIFF Images with Existing Code")
    print("=" * 50)
    
    # Path to your TIFF images
    tiff_directory = Path("../../data/images/동의서")
    
    if not tiff_directory.exists():
        print(f"❌ Error: Directory {tiff_directory} not found!")
        print("Make sure you're running this from the OCR/google_vision directory")
        return False
    
    # List TIFF files
    tiff_files = list(tiff_directory.glob("*.tif*"))
    if not tiff_files:
        print(f"❌ No TIFF files found in {tiff_directory}")
        return False
    
    print(f"📁 Found {len(tiff_files)} TIFF files:")
    for tiff_file in tiff_files:
        print(f"  - {tiff_file.name}")
    
    print("\n🔧 Available OCR Options:")
    print("1. Use existing categorized_batch_ocr.py with Google Cloud Vision")
    print("2. Use existing categorized_batch_ocr.py with Mistral OCR")
    print("3. Use existing text_converter_directory.py (Naver Clova OCR)")
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        print("✅ Using Google Cloud Vision API via categorized_batch_ocr.py")
        return use_categorized_batch_ocr(tiff_directory, "google_cloud")
    elif choice == "2":
        print("✅ Using Mistral OCR API via categorized_batch_ocr.py")
        return use_categorized_batch_ocr(tiff_directory, "mistral")
    elif choice == "3":
        print("✅ Using Naver Clova OCR via text_converter_directory.py")
        return use_naver_clova_ocr(tiff_directory)
    else:
        print("Invalid choice. Using Google Cloud Vision.")
        return use_categorized_batch_ocr(tiff_directory, "google_cloud")

def use_categorized_batch_ocr(tiff_directory, provider):
    """Use the existing categorized_batch_ocr.py for TIFF processing."""
    try:
        # Import the existing categorized batch OCR
        from categorized_batch_ocr import CategorizedPDFOCR
        
        print(f"\n🔄 Processing TIFF files with {provider}...")
        
        # Initialize OCR processor
        ocr_processor = CategorizedPDFOCR(
            base_output_dir=f"tiff_ocr_{provider}_results",
            ocr_provider=provider
        )
        
        # Process each TIFF file individually
        results = []
        for tiff_file in tiff_files:
            try:
                print(f"Processing {tiff_file.name}...")
                
                # Use the OCR provider's process_image method directly
                if provider == "google_cloud":
                    result = ocr_processor.ocr_provider.process_image(str(tiff_file))
                elif provider == "mistral":
                    result = ocr_processor.ocr_provider.process_image(str(tiff_file))
                
                # Add file metadata
                result['file_name'] = tiff_file.name
                result['file_path'] = str(tiff_file)
                results.append(result)
                
                print(f"✅ {tiff_file.name}: {result.get('text_length', 0)} characters")
                
            except Exception as e:
                print(f"❌ Error processing {tiff_file.name}: {e}")
                results.append({
                    'file_name': tiff_file.name,
                    'file_path': str(tiff_file),
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Save results
        import json
        from datetime import datetime
        
        output_dir = Path(f"tiff_ocr_{provider}_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save individual results
        for result in results:
            if 'error' not in result:
                file_name = result['file_name'].replace('.tif', '').replace('.tiff', '')
                result_path = output_dir / f"{file_name}_{provider}_ocr.json"
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                # Save extracted text
                text_path = output_dir / f"{file_name}_{provider}_extracted_text.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(result.get('extracted_text', ''))
        
        # Save batch summary
        summary_path = output_dir / f"batch_tiff_ocr_{provider}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_files': len(tiff_files),
                'successful': len([r for r in results if 'error' not in r]),
                'failed': len([r for r in results if 'error' in r]),
                'processing_timestamp': datetime.now().isoformat(),
                'provider': provider,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Processing Complete!")
        print(f"📊 Total files: {len(results)}")
        print(f"✅ Successful: {len([r for r in results if 'error' not in r])}")
        print(f"❌ Failed: {len([r for r in results if 'error' in r])}")
        print(f"💾 Results saved to: {output_dir}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing categorized_batch_ocr: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install google-cloud-vision mistralai python-dotenv")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def use_naver_clova_ocr(tiff_directory):
    """Use the existing Naver Clova OCR for TIFF processing."""
    try:
        # Import the existing text converter
        sys.path.append("../../OCR")
        from text_converter_directory import process_directory, Path as PathConverter
        
        print(f"\n🔄 Processing TIFF files with Naver Clova OCR...")
        
        # Convert paths for the text converter
        src_root = tiff_directory
        text_ocr_root = Path("naver_clova_tiff_ocr_results")
        
        # Process the directory
        processed_files = process_directory(tiff_directory, src_root, text_ocr_root)
        
        print(f"\n✅ Processing Complete!")
        print(f"📊 Processed {len(processed_files)} files")
        print(f"💾 Results saved to: {text_ocr_root}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing text_converter_directory: {e}")
        print("Make sure you're running from the correct directory")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function."""
    print("🚀 TIFF OCR Processor")
    print("Process TIFF images using your existing OCR infrastructure")
    print("=" * 60)
    
    success = process_tiff_with_existing_code()
    
    if success:
        print("\n🎉 TIFF OCR processing completed successfully!")
    else:
        print("\n❌ TIFF OCR processing failed.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
