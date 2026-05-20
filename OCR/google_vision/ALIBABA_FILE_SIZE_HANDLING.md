# Alibaba Cloud OCR File Size Handling

## Overview

The Alibaba Cloud Qwen-OCR models have specific file size and image dimension requirements. This document explains how the enhanced Universal OCR Processor handles these limitations and provides solutions for processing large files.

## File Size Limits

According to the [Alibaba Cloud documentation](https://www.alibabacloud.com/help/en/model-studio/qwen-vl-ocr):

- **Maximum file size**: 10 MB per image
- **Minimum dimensions**: Width and height must both be greater than 10 pixels
- **Aspect ratio**: Must not exceed 200:1 or 1:200
- **No strict pixel limit**: The model performs automatic scaling

## Enhanced Features

### 1. File Size Validation

The enhanced `AlibabaCloudOCRProvider` now includes:

```python
def _check_file_size(self, image_path: str) -> Dict[str, any]:
    """Check if the image file meets Alibaba Cloud size requirements."""
    file_size = os.path.getsize(image_path)
    file_size_mb = file_size / (1024 * 1024)
    
    return {
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size_mb, 2),
        "within_limit": file_size <= self.MAX_FILE_SIZE_BYTES,
        "recommendation": "File size is within acceptable limits." if within_limit else "Consider compressing..."
    }
```

### 2. Image Dimension Analysis

```python
def _get_image_dimensions(self, image_path: str) -> Dict[str, any]:
    """Get image dimensions to help with pixel calculations."""
    with Image.open(image_path) as img:
        width, height = img.size
        total_pixels = width * height
        
        return {
            "width": width,
            "height": height,
            "total_pixels": total_pixels,
            "aspect_ratio": round(width / height, 2),
            "within_pixel_limits": self.min_pixels <= total_pixels <= self.max_pixels
        }
```

### 3. Pixel Parameter Control

The processor now supports configurable pixel parameters:

- **`min_pixels`**: Ensures small images are enlarged to recognize details (default: 3,136)
- **`max_pixels`**: Prevents oversized images from consuming excessive resources (default: 6,422,528)

## Configuration

### Environment Variables

Add these to your `.env` file:

```bash
# File size and pixel parameters
ALIBABA_MIN_PIXELS=3136
ALIBABA_MAX_PIXELS=6422528

# Generation parameters
ALIBABA_TEMPERATURE=1.0
ALIBABA_TOP_P=0.8
ALIBABA_MODEL=qwen-vl-ocr
```

### Programmatic Configuration

```python
from universal_ocr_processor import UniversalOCRProcessor

# Initialize with custom pixel parameters
processor = UniversalOCRProcessor(
    provider="alibaba",
    output_dir="results"
)

# The processor automatically uses environment variables or defaults
```

## Handling Large Files

### Problem: File Exceeds 10MB Limit

When processing the file `/home/mbmk92/copyright/copyright_metadata_extraction/data/nii/02. 개인정보 이미지 용량별 파일/22.8MB.tif`:

```
❌ File size (22.8MB) exceeds the 10MB limit for Alibaba Cloud Qwen-OCR
```

### Solutions

#### 1. Image Compression

Use image processing tools to reduce file size:

```python
from PIL import Image

def compress_image(input_path, output_path, quality=85, max_size_mb=9):
    """Compress image to meet size requirements."""
    with Image.open(input_path) as img:
        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        # Calculate compression ratio
        current_size_mb = os.path.getsize(input_path) / (1024 * 1024)
        if current_size_mb > max_size_mb:
            ratio = max_size_mb / current_size_mb
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save with compression
        img.save(output_path, 'JPEG', quality=quality, optimize=True)
```

#### 2. Image Splitting

Split large images into smaller sections:

```python
def split_large_image(image_path, output_dir, max_width=2000, max_height=2000):
    """Split large image into smaller sections."""
    with Image.open(image_path) as img:
        width, height = img.size
        
        sections = []
        for y in range(0, height, max_height):
            for x in range(0, width, max_width):
                # Calculate crop box
                right = min(x + max_width, width)
                bottom = min(y + max_height, height)
                
                # Crop section
                section = img.crop((x, y, right, bottom))
                
                # Save section
                section_path = output_dir / f"section_{x}_{y}.jpg"
                section.save(section_path, 'JPEG', quality=90)
                sections.append(section_path)
        
        return sections
```

#### 3. Alternative OCR Providers

For very large files, consider using other providers:

```python
# Use Google Cloud Vision (no 10MB limit)
processor = UniversalOCRProcessor(provider="google_cloud")

# Use Mistral OCR
processor = UniversalOCRProcessor(provider="mistral")
```

#### 4. API Client Approach

Use the base64 encoding approach for better handling:

```python
# The processor automatically uses the API Client approach when needed
result = processor.process_single_file_api_client(large_file_path)
```

## Testing

### Test File Size Handling

Run the test script to verify file size handling:

```bash
cd /home/mbmk92/copyright/copyright_metadata_extraction/OCR/google_vision
python test_alibaba_file_size.py
```

### Expected Output

```
🔍 Testing Alibaba Cloud OCR File Size Checking
============================================================
✅ Alibaba Cloud OCR processor initialized successfully

📁 Testing with large file: 22.8MB.tif
📊 File Size Information:
   Size: 22.8 MB
   Within limit: False
   Recommendation: File size (22.8MB) exceeds the 10MB limit. Consider compressing the image or splitting it into smaller parts.

📐 Image Dimension Information:
   Dimensions: 3000 x 4000
   Total pixels: 12,000,000
   Aspect ratio: 0.75
   Within pixel limits: False

⚙️ Processing Parameters:
   Min pixels: 3,136
   Max pixels: 6,422,528
   Max file size: 10 MB

🔄 Attempting to process the large file...
❌ Processing failed as expected: File size (22.8MB) exceeds the 10MB limit for Alibaba Cloud Qwen-OCR
```

## Best Practices

### 1. Pre-processing Checklist

Before processing with Alibaba Cloud OCR:

- [ ] Check file size (< 10MB)
- [ ] Verify image dimensions (> 10x10 pixels)
- [ ] Check aspect ratio (< 200:1)
- [ ] Ensure image quality is sufficient for OCR

### 2. Optimization Strategies

- **For small images**: Increase `min_pixels` to ensure details are recognized
- **For large images**: Decrease `max_pixels` to reduce processing time
- **For high-resolution images**: Compress before processing
- **For very large files**: Split into sections or use alternative providers

### 3. Error Handling

The enhanced processor provides detailed error information:

```python
result = processor.process_single_file(large_file)

if result['status'] == 'error':
    print(f"Error: {result['error']}")
    print(f"File size: {result['file_size_info']['file_size_mb']} MB")
    print(f"Recommendation: {result['file_size_info']['recommendation']}")
```

## References

- [Alibaba Cloud Qwen-OCR Documentation](https://www.alibabacloud.com/help/en/model-studio/qwen-vl-ocr)
- [Alibaba Cloud Vision Documentation](https://www.alibabacloud.com/help/en/model-studio/vision)
- [Image Processing Best Practices](https://www.alibabacloud.com/help/en/model-studio/qwen-vl-ocr#da33480805fjh)

## Troubleshooting

### Common Issues

1. **"File size exceeds 10MB limit"**
   - Solution: Compress or split the image

2. **"Image too small"**
   - Solution: Increase `min_pixels` parameter

3. **"Processing timeout"**
   - Solution: Reduce `max_pixels` parameter

4. **"Poor OCR quality"**
   - Solution: Ensure image quality and appropriate pixel parameters

### Support

For additional help:
- Check the logs in the `logs/` directory
- Run the test script for diagnostics
- Review the Alibaba Cloud documentation
- Consider using alternative OCR providers for problematic files
