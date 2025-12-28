#!/usr/bin/env python3
import numpy as np
from pathlib import Path

cache_dir = Path('data/cache_ner')
npz_file = list(cache_dir.glob('dataset_*.npz'))[0]
print(f"파일: {npz_file}")
data = np.load(npz_file, allow_pickle=True)

print("Keys:", list(data.keys()))
print("\n입력 샘플 5개:")
print(data['input_ids'][:5, :20])
print("\n라벨 샘플 5개:")
print(data['labels'][:5, :20])
