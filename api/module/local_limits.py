"""시험 PC 의 무거운 로컬 작업 동시 개수 — 작업자(세트) 수와 따로 둔다.

DashScope 응답을 기다리는 일은 많이 겹쳐도 되지만, 시험 자료 드라이브를 읽는 일과 메모리를 크게 쓰는 일은 몇 개씩만 해야
빠르고 안전하다. 근거 — 2026-10-19/20 부하 시험 (docs/v4/load_test_20261019.md):

* 드라이브 읽기 (disk_slot) — WSL 에서 E:(외장 USB 하드)를 읽으면 동시에 1개일 때 102 MB/s, 2개 15 MB/s, 8개 16 MB/s
  (윈도에서 직접 읽으면 8개도 104 MB/s — WSL 파일 다리가 동시 읽기에서 무너진다). 그래서 드라이브마다 한 번에 하나씩,
  처음부터 끝까지 순서대로 읽어 작업 폴더로 복사한다(그동안 SHA-256 도 계산). 드라이브가 다르면 서로 기다리지 않는다
  (Te-1 계약서가 Te-2 영상 복사 뒤에 줄 서지 않게). 환경변수 LOCAL_DISK_SLOTS (드라이브당, 기본 1, 0 = 제한 없음).
* 무거운 처리 (heavy_slot) — 큰 그림 풀기, ffmpeg 키프레임. Te-2 150 작업자에서 WSL 메모리 15 GB 에 닿았다.
  환경변수 LOCAL_HEAVY_SLOTS (기본 8, 0 = 제한 없음).
"""
from __future__ import annotations

import contextlib
import os
import re
import threading
from typing import Dict, Iterator, Optional, Tuple

DEFAULT_DISK_SLOTS = 1
DEFAULT_HEAVY_SLOTS = 8

_LOCK = threading.Lock()
_SEMS: Dict[Tuple[str, str], Tuple[int, threading.BoundedSemaphore]] = {}


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def slots(kind: str = "heavy") -> int:
    """지금 설정된 동시 작업 수 (0 = 제한 없음). kind: 'heavy' | 'disk'."""
    if kind == "disk":
        return _env_int("LOCAL_DISK_SLOTS", DEFAULT_DISK_SLOTS)
    return _env_int("LOCAL_HEAVY_SLOTS", DEFAULT_HEAVY_SLOTS)


def drive_key(path: Optional[os.PathLike]) -> str:
    """경로가 있는 드라이브 — WSL /mnt/e/… → 'mnt/e', 윈도 E:\\… → 'E:', 그 밖은 'local'. 링크는 실제 파일을 따라간다."""
    p = str(path or "")
    if p and os.path.islink(p):
        p = os.path.realpath(p)
    m = re.match(r"^/mnt/([A-Za-z])(/|$)", p)
    if m:
        return "mnt/" + m.group(1).lower()
    m = re.match(r"^([A-Za-z]):[\\/]", p)
    if m:
        return m.group(1).upper() + ":"
    return "local"


@contextlib.contextmanager
def _slot(kind: str, key: str) -> Iterator[None]:
    n = slots(kind)
    if n <= 0:
        yield
        return
    with _LOCK:
        cur = _SEMS.get((kind, key))
        if cur is None or cur[0] != n:          # 설정이 바뀌면 새 세마포어 — 쓰던 쪽은 잡아 둔 객체에 돌려준다
            cur = (n, threading.BoundedSemaphore(n))
            _SEMS[(kind, key)] = cur
        sem = cur[1]
    with sem:
        yield


def heavy_slot() -> contextlib.AbstractContextManager:
    """큰 그림 풀기·ffmpeg 하나를 감싼다 — 자리가 날 때까지 기다린다."""
    return _slot("heavy", "all")


def disk_slot(path: Optional[os.PathLike] = None) -> contextlib.AbstractContextManager:
    """path 가 있는 드라이브에서 파일 하나를 통째로 읽는 일을 감싼다 — 드라이브마다 따로 줄 선다."""
    return _slot("disk", drive_key(path))
