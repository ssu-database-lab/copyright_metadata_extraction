"""Te-2 의 1~16번 — 같은 계약서를 이미 처리한 Te-1 결과를 찾아 쓴다 (2610028 최종 절차서: Te-2 는 1~23번 평가).

절차서 순서대로 Te-1 을 먼저 돌리면 Te-2 세트 대부분(4,000 중 3,800)은 같은 계약서를 Te-1 에서 이미 처리했다.
그 결과를 다시 계산하지 않고 쓴다. 쓰는 조건:
  - 같은 세트 ID 의 Te-1 예측(test_predictions/<set_id>.json)이 있고 성공(_meta.ok)했다
  - 그때 처리한 계약서가 지금 Te-2 의 계약서와 **같은 파일**이다 (sha256 — 이름만 같은 다른 계약서는 쓰지 않는다)
여러 Te-1 실행에 있으면 앞(가장 최근) 실행의 것을 쓴다. 찾지 못한 세트는 Te-2 가 계약서를 직접 처리한다 (batch_runner).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .keys23 import CONTRACT_ITEM_KEYS


def set_filename(set_id: Any) -> str:
    """<set_id>.json 의 파일명 — 평가기는 파일 stem 을 set_id 로 읽으므로 가능한 한 그대로 둔다."""
    s = str(set_id).strip() or "_"
    return s.replace(os.sep, "_").replace("/", "_")


def file_sha256(path: Any) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def is_ok_te1(rec: Any) -> bool:
    return isinstance(rec, dict) and rec.get("test") == "T-001" and (rec.get("_meta") or {}).get("ok") is True


def contract_items(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Te-1 예측 레코드 → Te-2 가 받는 항목 (1~16번, file_name 제외)."""
    return {k: rec.get(k) for k in CONTRACT_ITEM_KEYS}


def ok_contracts(results_path: Path) -> Dict[str, str]:
    """Te-1 실행의 results.jsonl → {set_id: 계약서 경로} (성공한 세트만, 같은 세트는 마지막 줄).
    계획(불러오기)용 — 세트 파일을 하나씩 열지 않고 실행마다 한 파일만 읽는다. 실제로 쓸 때는 find() 가 해시로 다시 확인한다."""
    out: Dict[str, str] = {}
    try:
        with Path(results_path).open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(r, dict) or r.get("set_id") is None:
                    continue
                sid = str(r["set_id"])
                if r.get("ok") and r.get("contract"):
                    out[sid] = str(r["contract"])
                else:
                    out.pop(sid, None)
    except OSError:
        pass
    return out


def probably_same(te1_contract: Optional[str], contract: str) -> bool:
    """계획용 어림 — 이름이 같고, Te-1 때의 계약서가 남아 있으면 크기도 같다 (해시는 실행 때 find() 가 본다)."""
    if not te1_contract or os.path.basename(te1_contract) != os.path.basename(contract):
        return False
    try:
        return os.path.getsize(te1_contract) == os.path.getsize(contract)
    except OSError:
        return True                                   # Te-1 때의 파일이 없으면 이름만으로 어림한다


class Te1Index:
    """Te-1 예측 폴더 여럿(가장 최근 실행이 앞) — find() 는 작업 스레드 여럿이 함께 불러도 된다."""

    def __init__(self, sources: Sequence[Tuple[str, Path]]):
        self.sources: List[Tuple[str, Path]] = [(str(r), Path(d)) for r, d in sources]
        self._sha: Dict[str, Optional[str]] = {}
        self._lock = threading.Lock()

    def sha_of(self, path: Any) -> Optional[str]:
        key = str(path)
        with self._lock:
            if key in self._sha:
                return self._sha[key]
        h = file_sha256(path)
        with self._lock:
            self._sha[key] = h
        return h

    @staticmethod
    def _load(f: Path) -> Optional[Dict[str, Any]]:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return r if isinstance(r, dict) else None

    def find(self, set_id: Any, contract_path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Te-2 세트의 계약서와 같은 파일을 처리한 가장 최근 Te-1 결과 → {record, run_id, path, contract_sha256} 또는 None."""
        if not contract_path:
            return None
        want = self.sha_of(contract_path)
        if not want:
            return None
        name = set_filename(set_id)
        for run_id, d in self.sources:
            f = d / f"{name}.json"
            if not f.is_file():
                continue
            rec = self._load(f)
            if not is_ok_te1(rec) or str(rec.get("set_id")) != str(set_id):
                continue
            meta = rec.get("_meta") or {}
            got = meta.get("file_sha256")
            if not got:                                   # 해시가 없는 옛 기록 — 그때 계약서가 남아 있으면 직접 잰다
                cp = (meta.get("files") or {}).get("contract")
                got = self.sha_of(cp) if cp and os.path.isfile(cp) else None
            if got == want:
                return {"record": rec, "run_id": run_id, "path": str(f), "contract_sha256": want}
        return None


def pred_dir_of(path: Any) -> Optional[Path]:
    """실행 폴더 · out 폴더 · test_predictions 폴더 어느 것을 줘도 예측 폴더를 찾는다 (CLI --te1-predictions)."""
    p = Path(path)
    for c in (p, p / "test_predictions", p / "out" / "test_predictions"):
        if c.is_dir() and (c.name == "test_predictions" or any(c.glob("*.json"))):
            return c
    return None


def index_from_paths(paths: Iterable[Any]) -> Te1Index:
    srcs: List[Tuple[str, Path]] = []
    for p in paths:
        d = pred_dir_of(p)
        if d is None:
            raise FileNotFoundError(f"Te-1 예측 폴더를 찾을 수 없습니다: {p}")
        srcs.append((d.parent.parent.name if d.parent.name == "out" else d.parent.name, d))
    return Te1Index(srcs)
