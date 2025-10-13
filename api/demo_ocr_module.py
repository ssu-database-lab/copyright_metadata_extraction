#!/usr/bin/env python3
"""
Demo script for Universal OCR Module
Shows how to use the OCR module without requiring API keys
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_paths = [
    Path(__file__).parent / ".env",  # API directory
    Path(__file__).parent / ".env_alibaba",  # Alibaba specific
    Path(__file__).parent.parent / "OCR" / "google_vision" / ".env",  # OCR directory
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment variables from: {env_path}")
        break
else:
    print("Warning: No .env file found. Using system environment variables only.")

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def demo_module_structure():
    """Demonstrate the OCR module structure"""
    print("=" * 60)
    print("Universal OCR Module Demo")
    print("=" * 60)
    
    print("\n📁 Module Structure:")
    print("api/module/ocr/")
    print("├── __init__.py              # Module exports")
    print("├── universal_ocr.py         # Main processor")
    print("├── google_ocr.py            # Google Cloud Vision")
    print("├── mistral_ocr.py           # Mistral AI")
    print("├── naver_ocr.py              # Naver Clova")
    print("└── alibaba_ocr.py           # Alibaba Cloud")
    
    print("\n🔧 Available Providers:")
    providers = [
        ("Google Cloud Vision", "google", "Requires credentials JSON file"),
        ("Mistral AI", "mistral", "Requires MISTRAL_API_KEY"),
        ("Naver Clova", "naver", "Requires NAVER_OCR_API_URL and NAVER_OCR_SECRET_KEY"),
        ("Alibaba Cloud", "alibaba", "Requires DASHSCOPE_API_KEY")
    ]
    
    for name, code, requirement in providers:
        print(f"  • {name} ({code}): {requirement}")
    
    print("\n📄 Supported File Types:")
    print("  Documents: PDF, DOCX, DOC, PPTX, XLS, XLSX, PPT, HWP")
    print("  Images: JPG, JPEG, PNG, GIF, BMP, TIF, TIFF")

def demo_usage_examples():
    """Show usage examples"""
    print("\n" + "=" * 60)
    print("Usage Examples")
    print("=" * 60)
    
    print("\n🐍 Python API:")
    print("""
from module.ocr import UniversalOCRProcessor

# Initialize processor
processor = UniversalOCRProcessor("alibaba", "output_dir", "qwen-vl-plus")

# Process single file
result = processor.process_single_file("document.pdf")

# Process directory
result = processor.process_directory("documents/")

# Streaming processing
for chunk in processor.process_single_file_streaming("document.pdf"):
    print(chunk, end='')
""")
    
    print("\n🌐 Web API:")
    print("""
# Start web server
cd web && python app.py

# Test via curl
curl -X POST "http://localhost:5000/api/ocr-universal" \\
  -F "file=@sample.pdf" \\
  -F "provider=alibaba" \\
  -F "model=qwen-vl-plus"
""")
    
    print("\n💻 Command Line:")
    print("""
# Direct usage
python -m module.ocr.universal_ocr \\
  --provider alibaba \\
  --input sample.pdf \\
  --output results/ \\
  --model qwen-vl-plus
""")

def demo_configuration():
    """Show configuration options"""
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    
    print("\n🔑 Environment Variables:")
    print("""
# Main .env file
MISTRAL_API_KEY=your_mistral_api_key_here
NAVER_OCR_API_URL=your_naver_api_url_here
NAVER_OCR_SECRET_KEY=your_naver_secret_key_here

# Alibaba Cloud .env_alibaba file
DASHSCOPE_API_KEY=your_alibaba_api_key_here
ALIBABA_API_KEY=your_alibaba_api_key_here
""")
    
    print("\n📁 Google Cloud Credentials:")
    print("""
# Place credentials JSON file in:
api/google_credentials.json
# or
OCR/google_vision/semiotic-pager-466612-t0-c587b9296fb8.json
""")
    
    print("\n⚙️ Provider-Specific Settings:")
    print("""
# Alibaba Cloud models
qwen-vl-ocr              # Original Qwen-VL-OCR
qwen-vl-plus             # Qwen3-VL-Plus
qwen3-vl-30b-a3b-instruct # Qwen3-VL-30B
qwen3-vl-235b-a22b-instruct # Qwen3-VL-235B

# Generation parameters
temperature: 1.0         # Default
top_p: 0.8              # Default
top_k: None             # Optional
""")

def demo_features():
    """Show key features"""
    print("\n" + "=" * 60)
    print("Key Features")
    print("=" * 60)
    
    features = [
        "✅ Universal file support (PDF, DOCX, images, etc.)",
        "✅ Multiple OCR providers (Google, Mistral, Naver, Alibaba)",
        "✅ Automatic file conversion to images",
        "✅ Streaming output support",
        "✅ Structured output directories",
        "✅ Error handling and logging",
        "✅ Web API integration",
        "✅ Command-line interface",
        "✅ Environment variable configuration",
        "✅ Markdown formatting cleanup"
    ]
    
    for feature in features:
        print(f"  {feature}")
    
    print("\n🎯 Processing Modes:")
    print("  • Regular processing: Complete result at once")
    print("  • Streaming processing: Real-time output")
    print("  • API client mode: Non-streaming API calls")
    print("  • Batch processing: Multiple files/directories")

def demo_next_steps():
    """Show next steps"""
    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    
    print("\n1. 🔑 Set up API keys:")
    print("   • Edit .env_alibaba with your Alibaba Cloud API key")
    print("   • Add Google Cloud credentials JSON file")
    print("   • Configure Mistral/Naver API keys if needed")
    
    print("\n2. 🧪 Test the module:")
    print("   python test_universal_ocr.py")
    
    print("\n3. 🌐 Start web interface:")
    print("   cd web && python app.py")
    print("   Visit: http://localhost:5000/docs")
    
    print("\n4. 🔗 Integrate with NER:")
    print("   • Use OCR output as input for NER processing")
    print("   • Combine OCR + NER + LLM metadata extraction")
    
    print("\n5. 🚀 Production deployment:")
    print("   • Set up proper environment variables")
    print("   • Configure logging and monitoring")
    print("   • Deploy web API with proper security")

def main():
    """Main demo function"""
    try:
        demo_module_structure()
        demo_usage_examples()
        demo_configuration()
        demo_features()
        demo_next_steps()
        
        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print("The Universal OCR Module is ready to use.")
        print("Set up your API keys and start processing documents!")
        
    except Exception as e:
        print(f"\nError during demo: {e}")

if __name__ == "__main__":
    main()
