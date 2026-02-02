#!/bin/bash
# Run NER training to completion and capture results

cd /mnt/c/Users/peppermint/Desktop/copyright_metadata_extraction/metadata
source .venv/bin/activate

echo "[시작] 혼합 데이터 + 라벨별 어댑터 훈련 시작..."
python3 main.py

echo ""
echo "[완료] 훈련 종료"
echo "결과 확인: models/ner/adapters/training_results.json"
