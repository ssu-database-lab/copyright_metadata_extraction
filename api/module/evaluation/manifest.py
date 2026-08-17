"""세트 매니페스트 — 계약서 1건 + 저작물 1건을 하나의 평가 단위로 묶는다.

**세트별 압축파일 대신 매니페스트를 쓰는 이유**
  - 4,000개 압축 해제는 순수 오버헤드다. 9p(/mnt/*)에서는 특히 비싸고, 재실행마다 반복된다.
  - 재개하려면 "2,847번 세트"에 바로 접근해야 하는데 압축은 풀어야 찾는다.
  - 파일럿 50건·특정 권리유형만·실패분 재시도가 매니페스트에서는 한 줄 필터다.
  - 짝짓기 근거가 파일시스템 구조가 아니라 **데이터로 남아** 감사 가능하다.

압축이 맞는 곳은 **시험기관 납품**이다 — 권리유형별 4개 아카이브에
contracts/ · works/ · manifest.jsonl · ground_truth.jsonl · checksums.sha256 를 담는다.

세트 ID는 **원문인덱스**를 쓴다. 파일명 접두(0034_)는 20건이 공유하고,
원본파일명 자체도 6,847건이 중복이라 키로 쓸 수 없다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class ManifestEntry:
    set_id: str
    work: str                      # 저작물 파일 경로 (매니페스트 기준 상대 또는 절대)
    contract: Optional[str] = None  # 계약서 경로 — 없으면 저작물 단독 평가
    media: str = "image"           # image | video | text
    license_bucket: str = ""       # expired | donated | ccl | kogl
    document_type: str = "저작재산권 이용허락 계약서"
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolve(self, root: Path) -> "ManifestEntry":
        """상대 경로를 매니페스트 위치 기준 절대 경로로 바꾼다."""
        def _abs(p):
            if not p:
                return p
            q = Path(p)
            return str(q if q.is_absolute() else (root / q))
        return ManifestEntry(
            set_id=self.set_id, work=_abs(self.work), contract=_abs(self.contract),
            media=self.media, license_bucket=self.license_bucket,
            document_type=self.document_type, extra=self.extra,
        )


def load_manifest(path: str | Path, resolve: bool = True) -> List[ManifestEntry]:
    p = Path(path)
    root = p.parent
    out: List[ManifestEntry] = []
    with p.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno} JSON 파싱 실패: {e}") from e
            known = {"set_id", "work", "contract", "media", "license_bucket",
                     "document_type", "extra"}
            entry = ManifestEntry(
                set_id=str(d.get("set_id") or d.get("id") or lineno),
                work=d.get("work") or d.get("work_file") or "",
                contract=d.get("contract") or d.get("contract_file"),
                media=d.get("media") or "image",
                license_bucket=d.get("license_bucket") or "",
                document_type=d.get("document_type") or "저작재산권 이용허락 계약서",
                extra={k: v for k, v in d.items() if k not in known},
            )
            out.append(entry.resolve(root) if resolve else entry)
    return out


def validate_manifest(entries: List[ManifestEntry]) -> Dict[str, Any]:
    """실행 전 점검 — 없는 파일·중복 ID를 미리 잡는다.

    4,000건을 돌리다 2,900번째에서 파일 없음으로 죽는 것보다 시작 전에 아는 게 낫다.
    """
    seen, dup = set(), []
    missing_work, missing_contract, no_contract = [], [], []
    for e in entries:
        if e.set_id in seen:
            dup.append(e.set_id)
        seen.add(e.set_id)
        if not e.work or not os.path.isfile(e.work):
            missing_work.append(e.set_id)
        if e.contract:
            if not os.path.isfile(e.contract):
                missing_contract.append(e.set_id)
        else:
            no_contract.append(e.set_id)
    return {
        "total": len(entries),
        "duplicate_ids": dup[:20], "duplicate_count": len(dup),
        "missing_work": missing_work[:20], "missing_work_count": len(missing_work),
        "missing_contract": missing_contract[:20],
        "missing_contract_count": len(missing_contract),
        "work_only": len(no_contract),
        "ok": not dup and not missing_work and not missing_contract,
    }


def load_ground_truth(path: str | Path) -> Dict[str, Dict]:
    """ground_truth.jsonl → {set_id: record}. id 또는 set_id 키를 모두 받는다."""
    gt: Dict[str, Dict] = {}
    p = Path(path)
    if not p.exists():
        return gt
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(r.get("set_id") or r.get("id") or "")
            if key:
                gt[key] = r
    return gt


def write_manifest(entries: List[ManifestEntry], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for e in entries:
            d = asdict(e)
            if not d.get("extra"):
                d.pop("extra", None)
            if d.get("contract") is None:
                d.pop("contract", None)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return len(entries)
