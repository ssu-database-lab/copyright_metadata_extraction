"""silver BIO 데이터 로딩 + 라벨 맵 구축 — TokenClassNER 학습 입력 준비."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)


def _load_records_by_label(dirs: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
    """디렉터리 목록에서 파일명(stem) → 레코드 리스트 매핑으로 로드.

    각 .jsonl 파일의 stem을 라벨 그룹 식별자로 사용한다.
    예) ch_co_address.jsonl → {"ch_co_address": [{...}, ...]}
    라벨은 정규화하지 않고 원본 그대로 유지 (role-specific 학습 지원).
    """
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for d in dirs:
        for p in sorted(d.glob("*.jsonl")):
            label_group = p.stem
            recs: List[Dict[str, Any]] = []
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    tokens = obj.get("tokens", [])
                    labels = obj.get("labels", [])
                    if tokens and len(tokens) == len(labels):
                        recs.append({"tokens": tokens, "labels": labels})
                except Exception:
                    continue
            if recs:
                by_label.setdefault(label_group, []).extend(recs)
    return by_label


def build_label_map(
    records: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, int]]:
    """BIO 레코드에서 고유 라벨 → label_list, label2id."""
    all_labels: set[str] = set()
    for rec in records:
        all_labels.update(rec["labels"])
    label_list = sorted(all_labels)
    if "O" in label_list:
        label_list.remove("O")
        label_list = ["O"] + label_list
    label2id = {label: idx for idx, label in enumerate(label_list)}
    return label_list, label2id


# ═══════════════════════════════════════════════════════════════════════
# TokenClassNER
