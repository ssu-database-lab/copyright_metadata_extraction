from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Optional
import json
import re

from .config import TRAINING_DATA_DIR


@dataclass
class PseudoEntity:
    label: str          # e.g. "phone_number"
    start: int          # token start idx
    end: int            # token end idx (exclusive)


class WeakSupervisionLabeler:
    """
    1) tokens(단어 단위)에서 정규식/룰로 엔티티 span을 잡고
    2) BIO labels를 만들어
    3) jsonl로 저장
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else TRAINING_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def label_with_regex(
        self,
        tokens: List[str],
        regex_map: Dict[str, str],   # {"phone_number": r"...", "email": r"..."}
    ) -> List[str]:
        text = " ".join(tokens)
        entities: List[PseudoEntity] = []

        # 아주 단순한 방식: 매치된 문자열을 다시 토큰 시퀀스에서 찾음
        # (고도화하려면 char-span 기반 토크나이즈 매핑 필요)
        for lab, pat in regex_map.items():
            for m in re.finditer(pat, text):
                matched = m.group(0).strip()
                if not matched:
                    continue
                mtoks = matched.split()
                if not mtoks:
                    continue

                # 토큰 subseq 검색
                for i in range(0, len(tokens) - len(mtoks) + 1):
                    if tokens[i : i + len(mtoks)] == mtoks:
                        entities.append(PseudoEntity(label=lab, start=i, end=i + len(mtoks)))
                        break

        labels = ["O"] * len(tokens)
        for ent in entities:
            if ent.start < 0 or ent.end > len(tokens) or ent.start >= ent.end:
                continue
            labels[ent.start] = f"B-{ent.label}"
            for j in range(ent.start + 1, ent.end):
                labels[j] = f"I-{ent.label}"
        return labels

    def append_jsonl(
        self,
        file_name: str,
        sample_id: str,
        tokens: List[str],
        labels: List[str],
    ) -> Path:
        out = self.output_dir / file_name
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": sample_id, "tokens": tokens, "labels": labels}, ensure_ascii=False) + "\n")
        return out
