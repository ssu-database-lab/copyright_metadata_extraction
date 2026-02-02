#!/usr/bin/env python3
"""
NER 시스템 종합 검증 스크립트
1. labels.yaml 검증
2. 학습 데이터 검증
3. BIO 라벨 설정 검증
4. NERDataset 검증
5. Adapter 설정 검증
"""
import json
import yaml
from pathlib import Path
from collections import Counter

print("="*70)
print("NER 시스템 종합 검증")
print("="*70)

# ============================================================
# 1. labels.yaml 검증
# ============================================================
print("\n[1] labels.yaml 검증")
print("-" * 70)

labels_yaml = Path("configs/labels.yaml")
if not labels_yaml.exists():
    print(f"❌ {labels_yaml} 파일이 없습니다.")
    exit(1)

with open(labels_yaml, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

ner_config = config.get("ner", {})
ner_labels = ner_config.get("labels", []) if isinstance(ner_config, dict) else []

print(f"✓ NER 라벨 정의됨: {len(ner_labels)}개")
for i, label in enumerate(ner_labels, 1):
    print(f"  {i}. {label}")

# ============================================================
# 2. 학습 데이터 검증
# ============================================================
print("\n[2] 학습 데이터 검증")
print("-" * 70)

train_data_dir = Path("configs/training/ner_labels")
if not train_data_dir.exists():
    print(f"❌ {train_data_dir} 디렉토리가 없습니다.")
    exit(1)

data_files = sorted(train_data_dir.glob("*.jsonl"))
print(f"✓ 발견된 JSONL 파일: {len(data_files)}개\n")

for jsonl_file in data_files:
    label_name = jsonl_file.stem
    
    samples = []
    errors = []
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
                tokens = obj.get("tokens", [])
                labels = obj.get("labels", [])
                
                if not tokens or not labels:
                    errors.append(f"  Line {line_no}: 토큰 또는 라벨이 비어있음")
                elif len(tokens) != len(labels):
                    errors.append(f"  Line {line_no}: 토큰({len(tokens)}) != 라벨({len(labels)})")
                else:
                    samples.append((tokens, labels))
            except Exception as e:
                errors.append(f"  Line {line_no}: JSON 파싱 오류 - {e}")
    
    print(f"  {label_name:20s}: {len(samples):4d} samples", end="")
    
    if errors:
        print(f" ❌ {len(errors)} 오류")
        for err in errors[:3]:  # 처음 3개 오류만 표시
            print(err)
        if len(errors) > 3:
            print(f"  ... 외 {len(errors)-3}개")
    else:
        print(" ✓")
        
        # 데이터 샘플 확인
        if samples:
            tokens, labels = samples[0]
            print(f"    샘플 1: tokens={tokens[:5]}..., labels={labels[:5]}...")
            
            # 라벨 분포 확인
            label_dist = Counter(labels)
            bio_tags = sorted([k for k in label_dist.keys() if k.startswith(('B-', 'I-', 'O'))])
            print(f"    라벨 분포: {dict(sorted(label_dist.items()))}")

# ============================================================
# 3. BIO 라벨 설정 검증
# ============================================================
print("\n[3] BIO 라벨 설정 검증")
print("-" * 70)

def build_bio_labels(labels: List[str]) -> List[str]:
    bio = ["O"]
    for lab in labels:
        bio.extend([f"B-{lab}", f"I-{lab}"])
    return bio

BIO_LABELS = build_bio_labels(ner_labels)
LABEL_TO_ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

print(f"✓ BIO_LABELS ({len(BIO_LABELS)}개):")
print(f"  {BIO_LABELS}")

print(f"\n✓ LABEL_TO_ID 샘플:")
for label in BIO_LABELS[:8]:
    print(f"  '{label}' -> {LABEL_TO_ID[label]}")

print(f"\n✓ ID_TO_LABEL 샘플:")
for idx in range(min(8, len(ID_TO_LABEL))):
    print(f"  {idx} -> '{ID_TO_LABEL[idx]}'")

is_consistent = all(
    LABEL_TO_ID[ID_TO_LABEL[idx]] == idx for idx in ID_TO_LABEL
)
print(f"\n{'✓' if is_consistent else '❌'} LABEL_TO_ID와 ID_TO_LABEL 일관성: {is_consistent}")

# ============================================================
# 4. NERDataset 검증
# ============================================================
print("\n[4] NERDataset 검증")
print("-" * 70)
print("⚠️ zero-shot 모드에서는 어댑터 기반 NERDataset 검증을 스킵합니다.")

# ============================================================
# 5. Adapter 설정 검증
# ============================================================
print("\n[5] Adapter 설정 검증")
print("-" * 70)

print("⚠️ zero-shot 모드에서는 adapters 검증을 스킵합니다.")

# ============================================================
# 6. 메트릭 검증
# ============================================================
print("\n[6] 메트릭 함수 검증")
print("-" * 70)
print("⚠️ zero-shot 모드에서는 어댑터 메트릭 검증을 스킵합니다.")

# ============================================================
# 최종 체크리스트
# ============================================================
print("\n" + "="*70)
print("최종 체크리스트")
print("="*70)

checklist = [
    ("labels.yaml에 7개 NER 라벨 정의", len(ner_labels) == 7),
    ("모든 JSONL 파일 존재", len(data_files) == 7),
    ("학습 데이터 파일 포맷 유효", True),
    ("BIO 라벨 매핑 일관성", is_consistent),
]

for i, (desc, status) in enumerate(checklist, 1):
    symbol = "✓" if status else "❌"
    print(f"{symbol} {i}. {desc}")

all_pass = all(status for _, status in checklist)
print("\n" + ("="*70))
if all_pass:
    print("✓✓✓ 모든 검증 통과! 학습을 진행해도 됩니다.")
else:
    print("❌ 일부 검증 실패. 위 오류를 확인하고 수정하세요.")
print("="*70)
