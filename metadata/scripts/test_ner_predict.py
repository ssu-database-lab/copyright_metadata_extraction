#!/usr/bin/env python3
"""
NER predict 테스트 (문장 1개). auto=False 로 기존 어댑터만 사용.
학습 완료 후 auto=True 로 실행하면 데이터 변경 시 자동 학습 후 predict.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module.extractor.ner import ner_extractor


def main() -> int:
    # 문장 1개: 대표 이름 홍길동, 연락처 help@company.co.kr
    sentences = [{"sent_id": 0, "text": "대표 이름 홍길동, 연락처 help@company.co.kr"}]
    tokens = [
        {"sent_id": 0, "tok_id": 0, "text": "대표"},
        {"sent_id": 0, "tok_id": 1, "text": "이름"},
        {"sent_id": 0, "tok_id": 2, "text": "홍길동"},
        {"sent_id": 0, "tok_id": 3, "text": ","},
        {"sent_id": 0, "tok_id": 4, "text": "연락처"},
        {"sent_id": 0, "tok_id": 5, "text": "help@company.co.kr"},
    ]
    decisions = ner_extractor(sentences=sentences, tokens=tokens, auto=False)
    print("auto=False (학습 검사 없이 predict):")
    for d in decisions:
        print(f"  {d.label}: {d.value}")
    if not decisions:
        print("  (추출된 엔티티 없음 — zero-shot 또는 어댑터 미로드)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
