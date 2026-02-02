#!/bin/bash
# Simple training runner for unified adapter NER

set -e

cd /mnt/c/Users/peppermint/Desktop/copyright_metadata_extraction/metadata
source .venv/bin/activate

echo "============================================"
echo "[통합 어댑터 NER 훈련]"
echo "============================================"
echo ""
echo "설정:"
echo "  - 혼합 데이터: 모든 7개 라벨 (4,100 샘플)"
echo "  - 어댑터: 1개 통합 어댑터"
echo "  - 라벨 수: 15개 BIO 라벨"
echo "  - 훈련 에포크: 5"
echo ""

python3 main.py

echo ""
echo "============================================"
echo "[훈련 완료]"
echo "============================================"
