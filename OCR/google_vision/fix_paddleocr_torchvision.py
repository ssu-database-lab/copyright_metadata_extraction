#!/usr/bin/env python3
"""
Quick fix script for PaddleOCR-VL torchvision NMS operator error
"""

import subprocess
import sys

def check_torchvision():
    """Check if torchvision NMS is available."""
    try:
        import torch
        import torchvision
        from torchvision.ops import nms
        
        print("✅ torchvision.ops.nms is available")
        print(f"PyTorch: {torch.__version__}")
        print(f"torchvision: {torchvision.__version__}")
        return True
    except ImportError as e:
        print(f"❌ torchvision.ops.nms is NOT available: {e}")
        return False

def fix_torchvision():
    """Try to fix torchvision installation."""
    print("\n" + "="*60)
    print("Attempting to fix torchvision installation...")
    print("="*60)
    
    commands = [
        ["pip", "uninstall", "-y", "torchvision"],
        ["pip", "install", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"]
    ]
    
    for cmd in commands:
        print(f"\nRunning: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Success")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    return True

def main():
    print("PaddleOCR-VL torchvision NMS Fix Script")
    print("="*60)
    
    if check_torchvision():
        print("\n✅ torchvision appears to be working correctly.")
        print("If you're still getting the error, it might be a model-specific issue.")
        print("\nTry using CPU mode as a workaround:")
        print("  export PADDLEOCR_DEVICE=cpu")
        return
    
    print("\n❌ torchvision NMS operator is missing or not working.")
    
    response = input("\nWould you like to try to fix it automatically? (y/n): ").strip().lower()
    
    if response == 'y':
        if fix_torchvision():
            print("\n✅ torchvision reinstalled. Please test again.")
            if check_torchvision():
                print("✅ Verification successful!")
            else:
                print("❌ Still having issues. You may need to:")
                print("   1. Restart your Python environment")
                print("   2. Use CPU mode: export PADDLEOCR_DEVICE=cpu")
        else:
            print("\n❌ Automatic fix failed. Try manual steps:")
            print("   1. pip uninstall torchvision")
            print("   2. pip install torchvision --index-url https://download.pytorch.org/whl/cu128")
    else:
        print("\nManual fix steps:")
        print("1. pip uninstall torchvision")
        print("2. pip install torchvision --index-url https://download.pytorch.org/whl/cu128")
        print("\nOr use CPU mode as workaround:")
        print("  export PADDLEOCR_DEVICE=cpu")

if __name__ == "__main__":
    main()

