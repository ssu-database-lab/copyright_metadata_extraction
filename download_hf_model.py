#!/usr/bin/env python3
"""
Hugging Face Model Downloader Script

This script downloads models from Hugging Face Hub and stores them in a specified directory.
Usage: python download_hf_model.py <model_id> [--output-dir OUTPUT_DIR]

Example:
    python download_hf_model.py microsoft/DialoGPT-medium
    python download_hf_model.py bert-base-uncased --output-dir C:\\my_models
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_output_directory(output_dir: str) -> Path:
    """Create the output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")
    return output_path

def download_model(model_id: str, output_dir: str, token: str = None) -> bool:
    """
    Download a model from Hugging Face Hub.
    
    Args:
        model_id: The model identifier (e.g., 'microsoft/DialoGPT-medium')
        output_dir: Directory to store the downloaded model
        token: Hugging Face token for private models (optional)
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        logger.info(f"Starting download of model: {model_id}")
        
        # Create output directory
        output_path = create_output_directory(output_dir)
        
        # Create model-specific subdirectory
        model_name = model_id.replace('/', '_')
        model_dir = output_path / model_name
        
        # Download the model
        logger.info(f"Downloading to: {model_dir}")
        
        # Use snapshot_download to get the entire model repository
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            token=token,
            resume_download=True,  # Resume interrupted downloads
            local_dir_use_symlinks=False  # Use actual files instead of symlinks
        )
        
        logger.info(f"Successfully downloaded model to: {downloaded_path}")
        
        # List downloaded files
        model_files = list(Path(downloaded_path).rglob('*'))
        logger.info(f"Downloaded {len(model_files)} files")
        
        # Show some key files
        key_files = [f for f in model_files if f.suffix in ['.json', '.txt', '.bin', '.safetensors', '.pt', '.pth']]
        if key_files:
            logger.info("Key files downloaded:")
            for file in key_files[:10]:  # Show first 10 key files
                logger.info(f"  - {file.name}")
            if len(key_files) > 10:
                logger.info(f"  ... and {len(key_files) - 10} more files")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {str(e)}")
        return False

def main():
    """Main function to handle command line arguments and execute download."""
    parser = argparse.ArgumentParser(
        description="Download models from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_hf_model.py microsoft/DialoGPT-medium
  python download_hf_model.py bert-base-uncased --output-dir C:\\\\my_models
  python download_hf_model.py meta-llama/Llama-2-7b-hf --token YOUR_HF_TOKEN
        """
    )
    
    parser.add_argument(
        'model_id',
        help='Hugging Face model identifier (e.g., microsoft/DialoGPT-medium)'
    )
    
    parser.add_argument(
        '--output-dir',
        default=r'C:\hf_models',
        help='Output directory for downloaded models (default: C:\\hf_models)'
    )
    
    parser.add_argument(
        '--token',
        help='Hugging Face token for private models (optional)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate model_id format
    if '/' not in args.model_id:
        logger.warning(f"Model ID '{args.model_id}' doesn't contain '/'. "
                      "Most models use format 'organization/model-name'")
    
    # Download the model
    success = download_model(args.model_id, args.output_dir, args.token)
    
    if success:
        logger.info("Download completed successfully!")
        sys.exit(0)
    else:
        logger.error("Download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
