# 🚀 Quick Start Guide - PDF Extraction

## ⚡ Immediate Action Steps

### 1. **Explore Your Archives (Dry Run)**
```bash
cd /home/mbmk92/projects/ssu/Project/extract
python simple_extract.py /path/to/your/zip/files /path/to/output --dry-run
```

### 2. **Extract PDFs**
```bash
python simple_extract.py /path/to/your/zip/files /path/to/output
```

### 3. **Interactive Setup**
```bash
python run_extraction.py
```

## 📁 What You Have

- **`simple_extract.py`** - Start here! Basic PDF extractor
- **`extract_pdfs.py`** - Advanced extractor with logging
- **`batch_extract.py`** - Multi-threaded for large collections
- **`run_extraction.py`** - Interactive setup wizard
- **`demo.py`** - See all features

## 🎯 Key Benefits

✅ **Efficient**: Only extracts PDFs, not entire archives  
✅ **Fast**: No decompression of non-PDF content  
✅ **Organized**: Maintains source archive traceability  
✅ **Safe**: Dry-run mode to preview first  
✅ **Scalable**: Handles thousands of archives  

## 💡 Pro Tips

1. **Always start with `--dry-run`** to see what's available
2. **Use batch processing** for collections >1000 archives
3. **Monitor disk space** - PDFs can still be large
4. **Check logs** for any extraction issues

## 🔧 Requirements

- Python 3.6+ ✅ (You have 3.12.9)
- Read access to ZIP files
- Write access to output directory
- No external dependencies needed

## 📊 Expected Results

From your 200GB collection, you'll get:
- **Organized PDF collection** ready for metadata extraction
- **Significant space savings** compared to full extraction
- **Detailed logs** of the extraction process
- **Traceable files** linking back to source archives

## 🚨 If Something Goes Wrong

1. Check file permissions
2. Verify directory paths exist
3. Ensure sufficient disk space
4. Run with `--dry-run` first
5. Check the logs in `extraction_logs/`

---

**Ready to extract PDFs? Run `python run_extraction.py` for the interactive setup!**
