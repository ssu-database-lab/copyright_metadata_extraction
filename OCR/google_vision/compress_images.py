#!/usr/bin/env python3
"""
Image Compression Utility for Alibaba Cloud OCR
Compresses large images to meet the 10MB file size limit while maintaining OCR quality.
"""

import os
import sys
from pathlib import Path
from PIL import Image
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageCompressor:
    """Compress images to meet Alibaba Cloud OCR requirements."""
    
    MAX_FILE_SIZE_MB = 9.5  # Leave some margin below 10MB limit
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    
    def __init__(self, output_dir: str = "compressed_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB."""
        return os.path.getsize(file_path) / (1024 * 1024)
    
    def compress_image(self, input_path: str, output_path: str = None, 
                      quality: int = 85, max_width: int = None, max_height: int = None) -> str:
        """
        Compress an image to meet size requirements.
        
        Args:
            input_path: Path to input image
            output_path: Path for compressed image (auto-generated if None)
            quality: JPEG quality (1-100)
            max_width: Maximum width for resizing
            max_height: Maximum height for resizing
            
        Returns:
            Path to compressed image
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Generate output path if not provided
        if output_path is None:
            output_path = self.output_dir / f"{input_path.stem}_compressed.jpg"
        else:
            output_path = Path(output_path)
        
        # Check if compression is needed
        current_size_mb = self.get_file_size_mb(input_path)
        if current_size_mb <= self.MAX_FILE_SIZE_MB:
            logger.info(f"File {input_path.name} ({current_size_mb:.2f}MB) is already within size limit")
            return str(input_path)
        
        logger.info(f"Compressing {input_path.name} ({current_size_mb:.2f}MB) to meet size requirements")
        
        try:
            with Image.open(input_path) as img:
                # Convert to RGB if necessary (JPEG doesn't support transparency)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparent images
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate compression ratio if needed
                if current_size_mb > self.MAX_FILE_SIZE_MB:
                    # Estimate compression ratio needed
                    target_ratio = self.MAX_FILE_SIZE_MB / current_size_mb
                    
                    # Apply size reduction
                    if max_width and max_height:
                        # Use provided dimensions
                        new_size = (max_width, max_height)
                    else:
                        # Calculate new size based on compression ratio
                        new_size = (int(img.width * target_ratio), int(img.height * target_ratio))
                    
                    # Ensure minimum size for OCR quality
                    min_size = 1000  # Minimum dimension for good OCR
                    if new_size[0] < min_size or new_size[1] < min_size:
                        # Maintain aspect ratio while ensuring minimum size
                        aspect_ratio = img.width / img.height
                        if new_size[0] < min_size:
                            new_size = (min_size, int(min_size / aspect_ratio))
                        if new_size[1] < min_size:
                            new_size = (int(min_size * aspect_ratio), min_size)
                    
                    logger.info(f"Resizing from {img.size} to {new_size}")
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save with compression
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
                
                # Check final size
                final_size_mb = self.get_file_size_mb(output_path)
                logger.info(f"Compressed to {output_path.name} ({final_size_mb:.2f}MB)")
                
                if final_size_mb > self.MAX_FILE_SIZE_MB:
                    logger.warning(f"Compressed file still exceeds size limit. Try reducing quality or dimensions.")
                
                return str(output_path)
                
        except Exception as e:
            logger.error(f"Error compressing {input_path}: {e}")
            raise
    
    def batch_compress(self, input_dir: str, pattern: str = "*.tif", 
                      quality: int = 85) -> list:
        """
        Compress all images in a directory.
        
        Args:
            input_dir: Directory containing images
            pattern: File pattern to match
            quality: JPEG quality
            
        Returns:
            List of compressed file paths
        """
        input_dir = Path(input_dir)
        compressed_files = []
        
        for file_path in input_dir.rglob(pattern):
            if file_path.is_file():
                try:
                    compressed_path = self.compress_image(file_path, quality=quality)
                    compressed_files.append(compressed_path)
                except Exception as e:
                    logger.error(f"Failed to compress {file_path}: {e}")
        
        return compressed_files

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Compress images for Alibaba Cloud OCR")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-q", "--quality", type=int, default=85, 
                       help="JPEG quality (1-100, default: 85)")
    parser.add_argument("--max-width", type=int, 
                       help="Maximum width for resizing")
    parser.add_argument("--max-height", type=int, 
                       help="Maximum height for resizing")
    parser.add_argument("-d", "--output-dir", default="compressed_images",
                       help="Output directory for batch processing")
    parser.add_argument("--pattern", default="*.tif",
                       help="File pattern for batch processing")
    
    args = parser.parse_args()
    
    compressor = ImageCompressor(args.output_dir)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file compression
        try:
            compressed_path = compressor.compress_image(
                args.input, 
                args.output, 
                quality=args.quality,
                max_width=args.max_width,
                max_height=args.max_height
            )
            print(f"✅ Compressed: {compressed_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    elif input_path.is_dir():
        # Batch compression
        try:
            compressed_files = compressor.batch_compress(
                args.input,
                pattern=args.pattern,
                quality=args.quality
            )
            print(f"✅ Compressed {len(compressed_files)} files")
            for file_path in compressed_files:
                print(f"   {file_path}")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    
    else:
        print(f"❌ Input path not found: {args.input}")
        sys.exit(1)

if __name__ == "__main__":
    main()
