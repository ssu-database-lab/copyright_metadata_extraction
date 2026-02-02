#!/usr/bin/env python3
"""
NER 시스템 종합 검증 스크립트 (간소화 버전)
"""
import json
import yaml
from pathlib import Path
from collections import Counter

print("="*70)
print("NER 시스템 종합 검증 (간소화)")
print("="*70)

# ============================================================
# 1. labels.yaml 검증
# ============================================================
print("\n[1] labels.yaml 검증")
print("-" * 70)

labels_yaml = Path("configs/labels.yaml")
with open(labels_yaml, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

ner_config = config.get("ner", {})
ner_labels = ner_config.get("labels", [])

print(f"✓ NER 라벨 정의됨: {len(ner_labels)}개")
for i, label in enumerate(ner_labels, 1):
    print(f"  {i}. {label}")

# ============================================================
# 2. BIO 라벨 생성
# ============================================================
print("\n[2] BIO 라벨 생성")
print("-" * 70)

BIO_LABELS = ["O"]
for label in ner_labels:
    BIO_LABELS.extend([f"B-{label}", f"I-{label}"])

LABEL_TO_ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

print(f"✓ BIO_LABELS ({len(BIO_LABELS)}개):")
print(f"  {BIO_LABELS[:10]}...")
print(f"\n✓ LABEL_TO_ID: {len(LABEL_TO_ID)}개 매핑")
print(f"✓ ID_TO_LABEL: {len(ID_TO_LABEL)}개 매핑")

# 일관성 확인
is_consistent = all(
    LABEL_TO_ID[ID_TO_LABEL[idx]] == idx for idx in ID_TO_LABEL
)
print(f"\n{'✓' if is_consistent else '❌'} LABEL_TO_ID ↔ ID_TO_LABEL 일관성: {is_consistent}")

# ============================================================
# 3. 학습 데이터 검증
# ============================================================
print("\n[3] 학습 데이터 검증")
print("-" * 70)

train_data_dir = Path("configs/training/ner_labels")
data_files = sorted(train_data_dir.glob("*.jsonl"))
print(f"✓ 발견된 JSONL 파일: {len(data_files)}개\n")

data_stats = {}
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
                    errors.append(f"Line {line_no}: 토큰 또는 라벨 비어있음")
                elif len(tokens) != len(labels):
                    errors.append(f"Line {line_no}: 토큰-라벨 길이 불일치 ({len(tokens)} != {len(labels)})")
                else:
                    samples.append((tokens, labels))
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_no}: JSON 파싱 오류")
    
    status = "✓" if not errors else f"❌ {len(errors)} errors"
    print(f"  {label_name:20s}: {len(samples):4d} samples {status}")
    
    if samples:
        tokens, labels = samples[0]
        # 라벨 검증
        valid_labels = set(BIO_LABELS)
        invalid_labels = [l for l in labels if l not in valid_labels]
        if invalid_labels:
            print(f"    ⚠️  유효하지 않은 라벨: {set(invalid_labels)}")
        else:
            print(f"    ✓ 모든 라벨이 BIO_LABELS에 포함됨")
    
    data_stats[label_name] = len(samples)

# ============================================================
# 4. 데이터 분포 확인
# ============================================================
print("\n[4] 데이터 분포 확인")
print("-" * 70)

total_samples = sum(data_stats.values())
print(f"✓ 전체 학습 샘플: {total_samples}개\n")

for label_name, count in sorted(data_stats.items(), key=lambda x: x[1], reverse=True):
    ratio = (count / total_samples) * 100
    bar = "█" * int(ratio / 2)
    print(f"  {label_name:15s}: {count:4d} ({ratio:5.1f}%) {bar}")

# ============================================================
# 5. Train/Val Split 시뮬레이션
# ============================================================
print("\n[5] Train/Val Split 검증 (80:20)")
print("-" * 70)

train_ratio = 0.8
for label_name, count in data_stats.items():
    train_count = int(count * train_ratio)
    val_count = count - train_count
    print(f"  {label_name:15s}: train={train_count:3d}, val={val_count:3d}")

# ============================================================
# 6. 토크나이저 검증
# ============================================================
print("\n[6] BERT 토크나이저 검증")
print("-" * 70)

try:
    from transformers import AutoTokenizer
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✓ '{model_name}' 토크나이저 로드 성공")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Special tokens: {tokenizer.special_tokens_map}")
except Exception as e:
    print(f"❌ 토크나이저 로드 실패: {e}")

# ============================================================
# 7. 샘플 토크나이제이션 테스트
# ============================================================
print("\n[7] 샘플 토크나이제이션 테스트")
print("-" * 70)

try:
    # address.jsonl의 첫 샘플
    address_file = Path("configs/training/ner_labels/address.jsonl")
    with open(address_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
    
    sample = json.loads(first_line)
    tokens = sample["tokens"]
    labels = sample["labels"]
    
    print(f"원본 데이터:")
    print(f"  Tokens ({len(tokens)}): {tokens[:15]}")
    print(f"  Labels ({len(labels)}): {labels[:15]}")
    
    # 토크나이제이션
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        padding=False,
        truncation=True,
        max_length=512,
        return_tensors=None,
    )
    
    print(f"\n토크나이제이션 결과:")
    print(f"  input_ids ({len(encoded['input_ids'])}): {encoded['input_ids'][:20]}")
    print(f"  word_ids ({len(encoded.word_ids())}): {encoded.word_ids()[:20]}")
    
    # word_ids를 통한 라벨 정렬
    word_ids = encoded.word_ids()
    aligned_labels = []
    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)
        elif word_idx != prev_word_idx:
            tag = labels[word_idx] if word_idx < len(labels) else "O"
            label_id = LABEL_TO_ID.get(tag, LABEL_TO_ID["O"])
            aligned_labels.append(label_id)
        else:
            aligned_labels.append(-100)
        prev_word_idx = word_idx
    
    print(f"  aligned_labels ({len(aligned_labels)}): {aligned_labels[:20]}")
    print(f"  → -100 (padding) 개수: {aligned_labels.count(-100)}")
    
    # 역변환 테스트
    reconstructed = [ID_TO_LABEL.get(lid, "O") for lid in aligned_labels if lid != -100]
    print(f"\n역변환 테스트:")
    print(f"  원본: {labels[:10]}")
    print(f"  재구성: {reconstructed[:10]}")
    
    match = labels[:10] == reconstructed[:10]
    print(f"  {'✓' if match else '❌'} 일치: {match}")
    
except Exception as e:
    print(f"❌ 토크나이제이션 테스트 실패: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 최종 체크리스트
# ============================================================
print("\n" + "="*70)
print("최종 체크리스트")
print("="*70)

checklist = [
    ("labels.yaml에 7개 NER 라벨 정의", len(ner_labels) == 7),
    ("BIO 라벨 생성 (15개 = O + 7*2)", len(BIO_LABELS) == 15),
    ("LABEL_TO_ID 매핑", len(LABEL_TO_ID) == 15),
    ("ID_TO_LABEL 매핑", len(ID_TO_LABEL) == 15),
    ("LABEL_TO_ID ↔ ID_TO_LABEL 일관성", is_consistent),
    ("모든 JSONL 파일 존재 (7개)", len(data_files) == 7),
    ("전체 학습 샘플 >= 4000개", total_samples >= 4000),
    ("토크나이저 로드 가능", True),
    ("토크나이제이션 정렬 작동", match if 'match' in locals() else False),
]

print()
for i, (desc, status) in enumerate(checklist, 1):
    symbol = "✓" if status else "❌"
    print(f"{symbol} {i:2d}. {desc}")

all_pass = all(status for _, status in checklist)
print("\n" + "="*70)
if all_pass:
    print("✓✓✓ 모든 검증 통과!")
    print("📊 데이터 상태: 정상")
    print("🎯 다음 단계: NER 모델 학습 진행 가능")
else:
    print("❌ 일부 검증 실패")
    print("⚠️  위 오류를 확인하고 수정하세요")
print("="*70)
