#!/usr/bin/env python3
import numpy as np
import os
from collections import Counter

cache_dir = "/mnt/c/Users/peppermint/Desktop/copyright_metadata_extraction/api/data/cache_ner"
dataset_files = [f for f in os.listdir(cache_dir) if f.startswith("dataset_") and f.endswith(".npz")]
latest = sorted(dataset_files)[-1]
path = os.path.join(cache_dir, latest)

print(f"📊 분석 중: {latest}")
data = np.load(path)
labels = data['labels']

total_tokens = 0
total_o_tokens = 0
total_entity_tokens = 0
entity_per_sample = []
samples_zero_entities = 0
IGNORE_INDEX = -100

for label_seq in labels:
    valid_labels = label_seq[label_seq != IGNORE_INDEX]
    if len(valid_labels) == 0:
        continue
    
    total_tokens += len(valid_labels)
    o_count = np.sum(valid_labels == 0)
    e_count = np.sum(valid_labels != 0)
    
    total_o_tokens += o_count
    total_entity_tokens += e_count
    entity_per_sample.append(e_count)
    
    if e_count == 0:
        samples_zero_entities += 1

entity_ratio = 100 * total_entity_tokens / total_tokens
zero_pct = 100 * samples_zero_entities / len(labels)

print("\n" + "="*70)
print("📈 데이터셋 라벨 분포 분석")
print("="*70)
print(f"\n총 토큰 (패딩 제외): {total_tokens:,}")
print(f"O (Outside) 토큰: {total_o_tokens:,} ({100*total_o_tokens/total_tokens:.1f}%)")
print(f"엔티티 토큰 (B-/I-): {total_entity_tokens:,} ({entity_ratio:.1f}%)")

print(f"\n총 샘플: {len(labels):,}")
print(f"엔티티 없는 샘플: {samples_zero_entities:,} ({zero_pct:.1f}%)")
print(f"엔티티 있는 샘플: {len(labels)-samples_zero_entities:,} ({100-zero_pct:.1f}%)")

entity_counts = np.array(entity_per_sample)
print(f"\n평균 엔티티 토큰/샘플: {np.mean(entity_counts):.2f}")
print(f"중간값: {np.median(entity_counts):.0f}")
print(f"최소: {np.min(entity_counts)}, 최대: {np.max(entity_counts)}")

print("\n" + "="*70)
print("🔍 진단:")
print("="*70)

if entity_ratio > 50:
    print(f"⚠️  심각: 전체 토큰의 {entity_ratio:.1f}%가 엔티티!")
    print("   → 모델이 모든 것을 엔티티로 예측")
    print("   → 결과: 높은 recall, 낮은 precision (~15-20%)")
elif entity_ratio > 35:
    print(f"⚠️  높음: 전체 토큰의 {entity_ratio:.1f}%가 엔티티")
    print("   → 엔티티 편향 데이터셋")
else:
    print(f"✅ 엔티티 비율 {entity_ratio:.1f}% - 상대적으로 균형잡음")

print()

if zero_pct < 10:
    print(f"⚠️  심각: 겨우 {zero_pct:.1f}%의 샘플만 엔티티가 없음!")
    print("   → 모델이 '엔티티 없음' 사례를 거의 보지 못함")
    print("   → 언제 예측하지 말아야 하는지 학습 불가")
elif zero_pct < 20:
    print(f"⚠️  낮음: 겨우 {zero_pct:.1f}%의 샘플만 엔티티가 없음")
    print("   → 더 많은 '엔티티 없음' 샘플이 필요")
else:
    print(f"✅ {zero_pct:.1f}% 샘플에 엔티티 없음 - 괜찮음")

print("\n" + "="*70)
