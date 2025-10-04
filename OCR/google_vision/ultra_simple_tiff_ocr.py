#!/usr/bin/env python3
"""
Ultra Simple TIFF OCR
Direct usage of Google Cloud Vision API for TIFF images
"""

import os
import json
from pathlib import Path
from datetime import datetime

def process_tiff_with_google_vision():
    """Process TIFF files using Google Cloud Vision API directly."""
    print("🚀 Simple TIFF OCR with Google Cloud Vision")
    print("=" * 50)
    
    # Path to your TIFF images
    tiff_directory = Path("../../data/images/동의서")
    
    if not tiff_directory.exists():
        print(f"❌ Error: Directory {tiff_directory} not found!")
        return False
    
    # List TIFF files
    tiff_files = list(tiff_directory.glob("*.tif*"))
    if not tiff_files:
        print(f"❌ No TIFF files found in {tiff_directory}")
        return False
    
    print(f"📁 Found {len(tiff_files)} TIFF files:")
    for tiff_file in tiff_files:
        print(f"  - {tiff_file.name}")
    
    try:
        # Import Google Cloud Vision
        from google.cloud import vision
        from google.protobuf.json_format import MessageToDict
        
        # Set up credentials (same as your existing code)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        credentials_path = os.path.join(script_dir, "semiotic-pager-466612-t0-c587b9296fb8.json")
        
        if not os.path.exists(credentials_path):
            print(f"❌ Google Cloud credentials not found at: {credentials_path}")
            return False
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        
        # Initialize client
        client = vision.ImageAnnotatorClient()
        
        # Create output directory
        output_dir = Path("simple_tiff_ocr_results")
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n🔄 Processing TIFF files...")
        
        results = []
        for tiff_file in tiff_files:
            try:
                print(f"Processing {tiff_file.name}...")
                
                # Read image file
                with open(tiff_file, 'rb') as image_file:
                    content = image_file.read()
                
                # Create image object
                image = vision.Image(content=content)
                
                # Perform OCR using document text detection
                response = client.document_text_detection(image=image)
                
                # Check for errors
                if response.error.message:
                    raise RuntimeError(f"Vision API Error: {response.error.message}")
                
                # Convert response to dictionary
                response_dict = MessageToDict(response._pb)
                
                # Extract text content
                extracted_text = ""
                if response.text_annotations:
                    extracted_text = response.text_annotations[0].description
                
                # Create result
                result = {
                    'file_name': tiff_file.name,
                    'file_path': str(tiff_file),
                    'extracted_text': extracted_text,
                    'text_length': len(extracted_text),
                    'processing_timestamp': datetime.now().isoformat(),
                    'provider': 'google_cloud_vision',
                    'raw_response': response_dict
                }
                
                results.append(result)
                
                # Save individual result
                file_name = tiff_file.stem
                result_path = output_dir / f"{file_name}_google_vision_ocr.json"
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                # Save extracted text
                text_path = output_dir / f"{file_name}_google_vision_extracted_text.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                
                print(f"✅ {tiff_file.name}: {len(extracted_text)} characters")
                
            except Exception as e:
                print(f"❌ Error processing {tiff_file.name}: {e}")
                results.append({
                    'file_name': tiff_file.name,
                    'file_path': str(tiff_file),
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Save batch summary
        summary_path = output_dir / "batch_tiff_ocr_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_files': len(tiff_files),
                'successful': len([r for r in results if 'error' not in r]),
                'failed': len([r for r in results if 'error' in r]),
                'processing_timestamp': datetime.now().isoformat(),
                'provider': 'google_cloud_vision',
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Processing Complete!")
        print(f"📊 Total files: {len(results)}")
        print(f"✅ Successful: {len([r for r in results if 'error' not in r])}")
        print(f"❌ Failed: {len([r for r in results if 'error' in r])}")
        print(f"💾 Results saved to: {output_dir}")
        
        # Show sample text from first successful result
        successful_results = [r for r in results if 'error' not in r]
        if successful_results and successful_results[0].get('extracted_text'):
            sample_text = successful_results[0]['extracted_text'][:300]
            print(f"\n📋 Sample Text from {successful_results[0]['file_name']} (first 300 chars):")
            print("-" * 50)
            print(sample_text + "..." if len(sample_text) == 300 else sample_text)
            print("-" * 50)
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing Google Cloud Vision: {e}")
        print("Install dependencies with: pip install google-cloud-vision")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function."""
    success = process_tiff_with_google_vision()
    
    if success:
        print("\n🎉 TIFF OCR processing completed successfully!")
    else:
        print("\n❌ TIFF OCR processing failed.")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
