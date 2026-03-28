#!/bin/bash
# ============================================
# Package CLI Tool for Distribution
# Creates a clean zip with only essential files
# ============================================

set -e

PKG="copyright_extraction_cli"
SRC="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Packaging CLI Tool"
echo "  Source: $SRC"
echo "============================================"

# Clean previous build
rm -rf "$SRC/$PKG" "$SRC/$PKG.zip"
mkdir -p "$SRC/$PKG"

# ── Root files ──
echo "[1/8] Copying root files..."
for f in extract.py run_extract.bat README_CLI.md requirements.txt .env .env.example pyproject.toml; do
    if [ -f "$SRC/$f" ]; then
        cp "$SRC/$f" "$SRC/$PKG/"
        echo "  + $f"
    fi
done

# ── api/ core files ──
echo "[2/8] Copying api/ core..."
mkdir -p "$SRC/$PKG/api"
cp "$SRC/api/__init__.py" "$SRC/$PKG/api/"
cp "$SRC/api/api.py" "$SRC/$PKG/api/"
# setup_env.py might be needed
[ -f "$SRC/api/setup_env.py" ] && cp "$SRC/api/setup_env.py" "$SRC/$PKG/api/"

# ── api/module/ ──
echo "[3/8] Copying api/module/..."
mkdir -p "$SRC/$PKG/api/module"
cp "$SRC/api/module/__init__.py" "$SRC/$PKG/api/module/" 2>/dev/null || touch "$SRC/$PKG/api/module/__init__.py"
cp "$SRC/api/module/env_loader.py" "$SRC/$PKG/api/module/"

# OCR module
echo "  + module/ocr/"
mkdir -p "$SRC/$PKG/api/module/ocr"
cp "$SRC"/api/module/ocr/*.py "$SRC/$PKG/api/module/ocr/"

# LLM extraction module (code only, no cached models or results)
echo "  + module/llm_extraction/"
find "$SRC/api/module/llm_extraction" -name "*.py" -o -name "*.yaml" -o -name "*.json" | while read f; do
    rel="${f#$SRC/api/module/llm_extraction/}"
    # Skip hf_models, metadata_results, __pycache__
    case "$rel" in
        models/hf_models/*|metadata_results/*|__pycache__/*) continue ;;
    esac
    dir="$SRC/$PKG/api/module/llm_extraction/$(dirname "$rel")"
    mkdir -p "$dir"
    cp "$f" "$dir/"
done

# Consolidator module
echo "  + module/consolidator/"
mkdir -p "$SRC/$PKG/api/module/consolidator/schemas"
cp "$SRC"/api/module/consolidator/*.py "$SRC/$PKG/api/module/consolidator/"
cp "$SRC"/api/module/consolidator/schemas/*.py "$SRC/$PKG/api/module/consolidator/schemas/" 2>/dev/null || true

# NER module (code only, no training data JSONs)
echo "  + module/ner/"
mkdir -p "$SRC/$PKG/api/module/ner"
cp "$SRC"/api/module/ner/*.py "$SRC/$PKG/api/module/ner/"
# Copy training dir structure but skip large JSON files
mkdir -p "$SRC/$PKG/api/module/ner/training"
find "$SRC/api/module/ner/training" -name "*.py" -o -name "*.txt" -o -name "*.csv" | while read f; do
    cp "$f" "$SRC/$PKG/api/module/ner/training/" 2>/dev/null || true
done

# ── api/web/pipeline.py only ──
echo "[4/8] Copying api/web/pipeline.py..."
mkdir -p "$SRC/$PKG/api/web"
cp "$SRC/api/web/pipeline.py" "$SRC/$PKG/api/web/"
touch "$SRC/$PKG/api/web/__init__.py"

# ── NER Model (KLUE-RoBERTa-Large, final files only) ──
echo "[5/8] Copying NER model (KLUE-RoBERTa-Large)..."
MODEL_DIR="$SRC/$PKG/api/models/ner/klue-roberta-large"
mkdir -p "$MODEL_DIR"
for f in model.pt config.json label_map.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json vocab.txt training_info.json training_history.json; do
    if [ -f "$SRC/api/models/ner/klue-roberta-large/$f" ]; then
        echo "  + $f ($(du -h "$SRC/api/models/ner/klue-roberta-large/$f" | cut -f1))"
        cp "$SRC/api/models/ner/klue-roberta-large/$f" "$MODEL_DIR/"
    fi
done
echo "  (skipped: model.safetensors, checkpoint-*/)"

# ── Sample output ──
echo "[6/8] Copying sample output..."
if [ -d "$SRC/sample_output" ]; then
    cp -r "$SRC/sample_output" "$SRC/$PKG/sample_output"
    # Remove internal _results and _uploads directories
    find "$SRC/$PKG/sample_output" -type d -name "_results" -exec rm -rf {} + 2>/dev/null || true
    find "$SRC/$PKG/sample_output" -type d -name "_uploads" -exec rm -rf {} + 2>/dev/null || true
fi

# ── Documentation ──
echo "[7/8] Copying documentation..."
if [ -d "$SRC/docs" ]; then
    mkdir -p "$SRC/$PKG/docs"
    # Copy all docs except very large files
    find "$SRC/docs" -maxdepth 1 -type f -size -10M | while read f; do
        cp "$f" "$SRC/$PKG/docs/"
    done
    # Copy md subdirectory
    [ -d "$SRC/docs/md" ] && cp -r "$SRC/docs/md" "$SRC/$PKG/docs/md"
fi

# ── Clean up __pycache__ ──
find "$SRC/$PKG" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ── Summary ──
echo "[8/8] Package summary..."
echo ""
echo "============================================"
echo "  Package: $PKG/"
echo "============================================"
echo ""
du -sh "$SRC/$PKG"
echo ""
echo "Directory structure:"
find "$SRC/$PKG" -maxdepth 3 -type d | sed "s|$SRC/$PKG|.|" | sort
echo ""
echo "Large files:"
find "$SRC/$PKG" -type f -size +1M -exec du -h {} \; | sort -rh
echo ""
echo "Total files: $(find "$SRC/$PKG" -type f | wc -l)"
echo ""

# ── Create zip ──
echo "Creating zip archive..."
cd "$SRC"
# Use zip if available, otherwise tar
if command -v zip &>/dev/null; then
    zip -r "$PKG.zip" "$PKG" -x "*/__pycache__/*"
    echo ""
    echo "============================================"
    echo "  DONE: $PKG.zip ($(du -h "$PKG.zip" | cut -f1))"
    echo "============================================"
else
    tar czf "$PKG.tar.gz" "$PKG" --exclude="__pycache__"
    echo ""
    echo "============================================"
    echo "  DONE: $PKG.tar.gz ($(du -h "$PKG.tar.gz" | cut -f1))"
    echo "============================================"
fi
