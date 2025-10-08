#!/usr/bin/env python3
"""
Universal OCR Processor - Easy Interface
Simple wrapper for the universal OCR processor with interactive menu
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from universal_ocr_processor import UniversalOCRProcessor

def show_menu():
    """Display the main menu."""
    print("\n" + "="*60)
    print("🚀 Universal OCR Processor")
    print("Process any file type: PDF, DOCX, DOC, PPTX, XLS, XLSX, PPT, HWP")
    print("Images: JPG, JPEG, PNG, GIF, BMP, TIF, TIFF")
    print("="*60)
    print("📋 Options:")
    print("1. Process single file")
    print("2. Process directory (all files)")
    print("3. Process directory recursively (including subdirectories)")
    print("4. Show supported file types")
    print("5. Exit")
    print("="*60)

def get_ocr_provider():
    """Get OCR provider choice from user."""
    print("\n🔧 Choose OCR Provider:")
    print("1. Google Cloud Vision API (recommended for Korean text)")
    print("2. Mistral OCR API")
    print("3. Naver Clova OCR API")
    print("4. Alibaba Cloud Model Studio (Qwen3-VL models)")
    
    while True:
        choice = input("Select provider (1-4): ").strip()
        if choice == "1":
            return "google_cloud", None
        elif choice == "2":
            return "mistral", None
        elif choice == "3":
            return "naver", None
        elif choice == "4":
            return "alibaba", get_alibaba_model()
        else:
            print("❌ Invalid choice. Please select 1-4.")

def get_alibaba_model():
    """Get Alibaba model choice from user."""
    print("\n🤖 Choose Alibaba Qwen3-VL Model:")
    print("1. Qwen-VL-OCR (Original) - General OCR tasks")
    print("2. Qwen3-VL-Plus - Enhanced performance")
    print("3. Qwen3-VL-30B-A3B-Instruct - Balanced performance")
    print("4. Qwen3-VL-235B-A22B-Instruct - Highest accuracy")
    
    while True:
        choice = input("Select model (1-4): ").strip()
        if choice == "1":
            return "qwen-vl-ocr"
        elif choice == "2":
            return "qwen-vl-plus"
        elif choice == "3":
            return "qwen3-vl-30b-a3b-instruct"
        elif choice == "4":
            return "qwen3-vl-235b-a22b-instruct"
        else:
            print("❌ Invalid choice. Please select 1-4.")

def get_processing_mode():
    """Get processing mode choice from user."""
    print("\n⚡ Choose Processing Mode:")
    print("1. Batch Processing - Complete response (DashScope SDK)")
    print("2. Streaming Processing - Real-time output (API Client)")
    print("3. API Client Batch - Complete response (API Client)")
    
    while True:
        choice = input("Select mode (1-3): ").strip()
        if choice == "1":
            return "batch"  # DashScope SDK batch processing
        elif choice == "2":
            return "streaming"   # API Client streaming processing
        elif choice == "3":
            return "api_client"  # API Client batch processing
        else:
            print("❌ Invalid choice. Please select 1-3.")

def process_single_file():
    """Process a single file."""
    print("\n📄 Process Single File")
    print("-" * 30)
    
    file_path = input("Enter the path to your file: ").strip()
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} not found!")
        return False
    
    provider, model = get_ocr_provider()
    
    # Ask for processing mode only for Alibaba provider (multiple processing options)
    if provider == "alibaba":
        processing_mode = get_processing_mode()
    else:
        processing_mode = "batch"  # Other providers only support batch processing
    
    try:
        processor = UniversalOCRProcessor(provider, model=model)
        
        if processing_mode == "streaming":
            # Streaming processing
            print(f"\n🔄 Streaming Processing Started!")
            print(f"📄 File: {Path(file_path).name}")
            print("=" * 50)
            print("Streaming output:")
            print("-" * 30)
            
            full_content = ""
            for chunk in processor.process_single_file_streaming(file_path):
                print(chunk, end='', flush=True)
                full_content += chunk
            
            print("\n" + "-" * 30)
            print(f"\n✅ Streaming Processing Complete!")
            print(f"📝 Total characters: {len(full_content)}")
            print(f"🔍 Provider: {provider}")
            if model:
                print(f"🤖 Model: {model}")
            
            # Show sample text
            if full_content:
                sample_text = full_content[:300]
                print(f"\n📋 Sample Text (first 300 chars):")
                print("-" * 50)
                print(sample_text + "..." if len(sample_text) == 300 else sample_text)
                print("-" * 50)
                
        elif processing_mode == "api_client":
            # API Client batch processing
            print(f"\n🔧 API Client Batch Processing Started!")
            print(f"📄 File: {Path(file_path).name}")
            print("=" * 50)
            
            result = processor.process_single_file_api_client(file_path)
            
            print(f"\n✅ API Client Processing Complete!")
            print(f"📄 File: {result['file_name']}")
            print(f"📊 Status: {result['status']}")
            
            if result['status'] == 'success':
                print(f"📝 Text Length: {result['total_text_length']} characters")
                print(f"📖 Pages: {result['total_pages']}")
                print(f"🔍 Provider: {result['ocr_provider']}")
                print(f"⚙️ Processing Mode: {result['processing_mode']}")
                
                # Show sample text
                if result['full_text']:
                    sample_text = result['full_text'][:300]
                    print(f"\n📋 Sample Text (first 300 chars):")
                    print("-" * 50)
                    print(sample_text + "..." if len(sample_text) == 300 else sample_text)
                    print("-" * 50)
            
            elif result['status'] == 'failed':
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                
        else:  # processing_mode == "batch"
            # DashScope SDK batch processing
            print(f"\n📦 DashScope SDK Batch Processing Started!")
            print(f"📄 File: {Path(file_path).name}")
            print("=" * 50)
            
            result = processor.process_single_file(file_path)
            
            print(f"\n✅ Batch Processing Complete!")
            print(f"📄 File: {result['file_name']}")
            print(f"📊 Status: {result['status']}")
            
            if result['status'] == 'success':
                print(f"📝 Text Length: {result['total_text_length']} characters")
                print(f"📖 Pages: {result['total_pages']}")
                print(f"🔍 Provider: {result['ocr_provider']}")
                
                # Show sample text
                if result['full_text']:
                    sample_text = result['full_text'][:300]
                    print(f"\n📋 Sample Text (first 300 chars):")
                    print("-" * 50)
                    print(sample_text + "..." if len(sample_text) == 300 else sample_text)
                    print("-" * 50)
            
            elif result['status'] == 'failed':
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False

def process_directory(recursive=False):
    """Process all files in a directory."""
    print(f"\n📁 Process Directory {'(Recursive)' if recursive else ''}")
    print("-" * 40)
    
    dir_path = input("Enter the path to your directory: ").strip()
    
    if not os.path.exists(dir_path):
        print(f"❌ Error: Directory {dir_path} not found!")
        return False
    
    if not os.path.isdir(dir_path):
        print(f"❌ Error: {dir_path} is not a directory!")
        return False
    
    provider, model = get_ocr_provider()
    
    try:
        processor = UniversalOCRProcessor(provider, model=model)
        results = processor.process_directory(dir_path, recursive)
        
        if not results:
            print("❌ No supported files found in the directory!")
            return False
        
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'failed']
        
        print(f"\n🎉 Batch Processing Complete!")
        print(f"📊 Summary:")
        print(f"   Total files: {len(results)}")
        print(f"   ✅ Successful: {len(successful)}")
        print(f"   ❌ Failed: {len(failed)}")
        print(f"   📝 Total text extracted: {sum(r.get('total_text_length', 0) for r in successful)} characters")
        print(f"   🔍 Provider: {provider}")
        
        if successful:
            print(f"\n📋 Successfully processed files:")
            for result in successful[:10]:  # Show first 10
                print(f"   - {result['file_name']}: {result.get('total_text_length', 0)} chars")
            if len(successful) > 10:
                print(f"   ... and {len(successful) - 10} more files")
        
        if failed:
            print(f"\n❌ Failed files:")
            for result in failed[:5]:  # Show first 5
                print(f"   - {result['file_name']}: {result.get('error', 'Unknown error')}")
            if len(failed) > 5:
                print(f"   ... and {len(failed) - 5} more files")
        
        print(f"\n💾 Results saved to: {processor.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing directory: {e}")
        return False

def show_supported_types():
    """Show supported file types."""
    print("\n📋 Supported File Types")
    print("-" * 30)
    
    print("📄 Documents:")
    print("   - PDF (.pdf)")
    print("   - Microsoft Word (.docx, .doc)")
    print("   - Microsoft PowerPoint (.pptx, .ppt)")
    print("   - Microsoft Excel (.xlsx, .xls)")
    print("   - Hancom Office (.hwp)")
    
    print("\n🖼️ Images:")
    print("   - JPEG (.jpg, .jpeg)")
    print("   - PNG (.png)")
    print("   - GIF (.gif)")
    print("   - BMP (.bmp)")
    print("   - TIFF (.tif, .tiff)")
    
    print("\n🔧 OCR Providers:")
    print("   - Google Cloud Vision API")
    print("   - Mistral OCR API")
    print("   - Naver Clova OCR API")
    print("   - Alibaba Cloud Model Studio (Qwen3-VL models)")
    print("     • Qwen-VL-OCR (Original)")
    print("     • Qwen3-VL-Plus")
    print("     • Qwen3-VL-30B-A3B-Instruct")
    print("     • Qwen3-VL-235B-A22B-Instruct")
    
    print("\n⚡ Processing Modes:")
    print("   - Batch Processing - Complete response (DashScope SDK)")
    print("   - Streaming Processing - Real-time output (API Client)")
    print("   - API Client Batch - Complete response (API Client)")

def main():
    """Main function with interactive menu."""
    print("🚀 Universal OCR Processor - Easy Interface")
    print("Process any file type with multiple OCR providers")
    
    while True:
        show_menu()
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            process_single_file()
        elif choice == "2":
            process_directory(recursive=False)
        elif choice == "3":
            process_directory(recursive=True)
        elif choice == "4":
            show_supported_types()
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-5.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
