#!/usr/bin/env python3
"""
Setup script for Korean Document Metadata Extraction
Installs dependencies and sets up the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "logs",
        "metadata_results",
        "models/__init__.py",
        "extractors/__init__.py", 
        "schemas/__init__.py"
    ]
    
    for dir_path in directories:
        if dir_path.endswith("__init__.py"):
            # Create __init__.py files
            Path(dir_path).parent.mkdir(parents=True, exist_ok=True)
            Path(dir_path).touch()
        else:
            # Create directories
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created successfully")

def check_gpu():
    """Check GPU availability"""
    print("🖥️ Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️ No GPU detected. CPU processing will be used (slower)")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Korean Document Metadata Extraction System")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        sys.exit(1)
    
    # Check GPU
    check_gpu()
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Test the system: python extract_metadata.py --test")
    print("2. Process OCR results: python extract_metadata.py --ocr-results-dir ../OCR/google_vision/ocr_results")
    print("3. Choose different model: python extract_metadata.py --model qwen --test")
    
    print("\n🔧 Available models:")
    print("- solar-ko: SOLAR-Ko-10.7B (recommended for Korean)")
    print("- qwen: Qwen2.5-7B (multilingual)")
    print("- qwen72b: Qwen2.5-72B (high performance, large model)")
    print("- qwen3: Qwen3-4B (latest generation)")
    print("- llama: Llama-3.1-70B (high performance, large model)")
    print("- gemma3: Gemma 3 12B (latest Google model, requires auth)")
    print("- mixtral: Mixtral 8x7B (mixture of experts, requires auth)")
    print("- lightweight: SOLAR-Ko-1.7B (faster, less accurate)")

if __name__ == "__main__":
    main()
