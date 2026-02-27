#!/usr/bin/env python3
"""
학습 데이터 형식 검증 + 병합/시그니처 테스트 (GLiNER2 불필요).
실제 학습은: .venv/bin/python -m module.extractor.ner.train
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 프로젝트 루트
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module.extractor.ner.base import (
    bio_to_ner_spans,
    get_gliner_train_dir,
    get_training_data_signature,
    load_labels_from_predict,
)


def main() -> int:
    train_dir = get_gliner_train_dir()
    if not train_dir.exists():
        print(f"❌ 학습 디렉터리 없음: {train_dir}")
        return 1

    labels, _ = load_labels_from_predict()
    if not labels:
        print("❌ predict 에서 라벨을 읽지 못함 (configs/gliner/predict/*.txt)")
        return 1
    print(f"✓ predict 라벨: {labels}")

    jsonl_files = list(train_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"❌ {train_dir} 에 .jsonl 파일 없음")
        return 1
    print(f"✓ 학습 파일: {[p.name for p in jsonl_files]}")

    allowed = set(labels)
    total = 0
    for p in sorted(train_dir.glob("*.jsonl")):
        if p.stem not in allowed:
            print(f"  ⚠ 건너뜀 (predict에 없음): {p.name}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    tokens = obj.get("tokens", [])
                    labels_list = obj.get("labels", [])
                    if len(tokens) != len(labels_list):
                        print(f"  ❌ {p.name} 줄 {i}: tokens({len(tokens)}) != labels({len(labels_list)})")
                        return 1
                    ner = bio_to_ner_spans(labels_list)
                    total += 1
                except Exception as e:
                    print(f"  ❌ {p.name} 줄 {i}: {e}")
                    return 1
        print(f"  ✓ {p.name}: {sum(1 for _ in open(p, encoding='utf-8') if _.strip())} 줄")
    print(f"✓ 총 병합 예제 수: {total}")

    sig = get_training_data_signature(train_dir)
    print(f"✓ 학습 데이터 시그니처: {sig[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
