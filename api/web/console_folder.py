"""시험 콘솔 — 폴더로 시험 (시험 PC 의 폴더를 읽어 세트를 만들고 정답 파일을 읽는다).

시험 PC 에는 저작물 원본이 이미 있다 (T-002 4,000건 = 376 GB, E: 네 폴더). 브라우저로 올리거나 한 폴더로 모을 수
없으므로, 콘솔이 **폴더를 제자리에서 읽는다**.

- 위치 제한: 허용된 위치(`CONSOLE_DATA_ROOTS`, 세미콜론 구분) 안쪽만 열고 읽는다. 값이 없으면 WSL 에서는 윈도 드라이브
  (/mnt/c, /mnt/d …), 그 밖(공개 서버 등)에서는 **꺼짐**. 경로는 `E:\\시험자료` 처럼 윈도 형식으로 받아도 된다.
- 세트 ID: 파일 이름의 5~8자리 숫자 — 정답(또는 시험 목록)에 있는 묶음이 먼저, 없으면 0 으로 시작하지 않는 첫 묶음
  ('00002_13387752 운수 자판기.pdf' → 13387752). 파일 이름에 숫자가 없으면 바로 위 폴더 이름에서 찾는다.
- 짝 맞추기: T-001 = 계약서 하나가 세트 하나. T-002 = 저작물 하나가 세트 하나, 같은 ID 의 계약서가 있으면 붙인다(없어도 됨).
- 정답: 기본 정답(dataset/test_set_v4) · 정답 파일(.jsonl · .json · .csv · .xlsx) · 정답 없이. 정답이 있으면 **정답에 있는
  세트만** 실행한다(폴더에 더 많은 파일이 있어도 된다). 같은 ID 의 파일이 여럿이면 정답의 파일명(file_name)과 같은 파일을 쓴다.

이 모듈은 FastAPI 를 모른다 — 실행 만들기·엔드포인트는 console_api.py 가 한다.
"""
from __future__ import annotations

import csv
import io
import json
import os
import platform
import re
import string
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

ENV_ROOTS = "CONSOLE_DATA_ROOTS"
ANSWER_EXT = (".jsonl", ".json", ".csv", ".xlsx")
MAX_LIST_DIRS = 500
MAX_ANSWER_FILES = 200
MAX_SCAN_FILES = 300_000
MAX_EXAMPLES = 50
SAMPLE_ROWS = 20
EMPTY_MARKERS = {"(빈값)", "빈값", "empty", "<empty>", "(empty)"}
_ID_RX = re.compile(r"(?<!\d)(\d{5,8})(?!\d)")
_SKIP_DIRS = {"$recycle.bin", "system volume information", "__pycache__", "node_modules", ".git"}


class FolderError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra


# ---------------------------------------------------------------------------
# 위치 — 윈도 경로 변환 · 허용 위치
# ---------------------------------------------------------------------------
def is_wsl() -> bool:
    return "microsoft" in platform.release().lower() or Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists()


def data_roots() -> List[Path]:
    """허용 위치. 환경변수가 있으면 그 값(빈 값이면 꺼짐), 없으면 WSL 의 윈도 드라이브."""
    raw = os.environ.get(ENV_ROOTS)
    if raw is not None:
        out = []
        for part in re.split(r"[;\n]", raw):
            part = part.strip()
            if part:
                try:
                    out.append(Path(os.path.realpath(to_local_path(part))))
                except FolderError:
                    continue
        return out
    if not is_wsl():
        return []
    return [Path(f"/mnt/{d}") for d in string.ascii_lowercase if os.path.ismount(f"/mnt/{d}")]


def to_local_path(raw: Any) -> Path:
    """'E:\\시험자료' · 'E:/시험자료' · '/mnt/e/시험자료' · '\\\\wsl.localhost\\Ubuntu\\home\\…' → 이 서버의 경로."""
    s = str(raw or "").strip().strip('"').strip("'").strip()
    if not s:
        raise FolderError(400, "no_path", "폴더 경로를 입력하세요")
    if "\x00" in s:
        raise FolderError(400, "bad_path", "경로에 쓸 수 없는 문자가 있습니다")
    m = re.match(r"^([A-Za-z]):(?:[\\/](.*))?$", s)
    if m:
        s = f"/mnt/{m.group(1).lower()}/" + (m.group(2) or "").replace("\\", "/")
    else:
        m = re.match(r"^\\\\wsl(?:\.localhost|\$)\\[^\\]+(\\.*)?$", s, re.I)
        if m:
            s = (m.group(1) or "\\").replace("\\", "/")
        elif not s.startswith("/"):
            raise FolderError(400, "relative_path", "전체 경로를 입력하세요 (예: E:\\시험자료)")
    return Path(os.path.normpath(s))


def display_path(p: os.PathLike) -> str:
    """/mnt/e/시험자료 → E:\\시험자료 (윈도 사용자에게 익숙한 모양). 그 밖은 그대로."""
    s = str(p)
    m = re.match(r"^/mnt/([a-z])(?:/(.*))?$", s)
    if m:
        return f"{m.group(1).upper()}:\\" + (m.group(2) or "").replace("/", "\\")
    return s


def _inside(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def resolve_allowed(raw: Any, roots: Sequence[Path], *, want: str = "dir") -> Path:
    if not roots:
        raise FolderError(403, "folders_off", "이 서버에서는 폴더로 시험을 쓸 수 없습니다 (허용된 위치가 없습니다)")
    p = Path(os.path.realpath(to_local_path(raw)))
    if not any(_inside(p, r) for r in roots):
        raise FolderError(403, "outside_roots", "콘솔에서 열 수 없는 위치입니다 — 허용된 위치: "
                          + ", ".join(display_path(r) for r in roots))
    if want == "dir" and not p.is_dir():
        raise FolderError(404, "no_dir", f"폴더가 없습니다: {display_path(p)}")
    if want == "file" and not p.is_file():
        raise FolderError(404, "no_file", f"파일이 없습니다: {display_path(p)}")
    return p


def roots_view(roots: Sequence[Path]) -> List[Dict[str, str]]:
    return [{"label": display_path(r).rstrip("\\") or display_path(r), "path": display_path(r)} for r in roots]


def _skip_dir(name: str) -> bool:
    return name.startswith((".", "$")) or name.lower() in _SKIP_DIRS


def list_dir(raw: Any, roots: Sequence[Path], contract_ext: Iterable[str], work_ext: Iterable[str]) -> Dict[str, Any]:
    """폴더 고르기 창 — 하위 폴더와 이 폴더의 파일 수(계약서·저작물·정답), 정답 파일 이름."""
    p = resolve_allowed(raw, roots)
    cext, wext = set(contract_ext), set(work_ext)
    dirs, answers = [], []
    counts = {"contract": 0, "work": 0, "answer": 0, "other": 0}
    try:
        with os.scandir(p) as it:
            items = sorted(it, key=lambda e: e.name.lower())
    except OSError as e:
        raise FolderError(403, "unreadable", f"폴더를 읽을 수 없습니다: {display_path(p)} ({e.strerror or e})")
    for e in items:
        try:
            if e.is_dir(follow_symlinks=True):
                if not _skip_dir(e.name) and len(dirs) < MAX_LIST_DIRS:
                    dirs.append({"name": e.name, "path": display_path(p / e.name)})
                continue
            if not e.is_file(follow_symlinks=True):
                continue
        except OSError:
            continue
        ext = os.path.splitext(e.name)[1].lower()
        if ext in ANSWER_EXT:
            counts["answer"] += 1
            if len(answers) < MAX_ANSWER_FILES:
                answers.append({"name": e.name, "path": display_path(p / e.name)})
        elif ext in cext and ext in wext:          # 이미지 — 계약서(스캔)일 수도, 저작물일 수도
            counts["work"] += 1
        elif ext in cext:
            counts["contract"] += 1
        elif ext in wext:
            counts["work"] += 1
        else:
            counts["other"] += 1
    parent = p.parent
    up = display_path(parent) if parent != p and any(_inside(parent, r) for r in roots) else None
    return {"path": display_path(p), "parent": up, "dirs": dirs, "more_dirs": len(dirs) >= MAX_LIST_DIRS,
            "counts": counts, "answer_files": answers, "roots": roots_view(roots)}


# ---------------------------------------------------------------------------
# 파일 찾기 · 세트 ID
# ---------------------------------------------------------------------------
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def walk_files(dirs: Sequence[Path], exts: Iterable[str], recursive: bool = True) -> Iterator[Path]:
    ex = {e.lower() for e in exts}
    seen: set = set()
    n = 0
    for d in dirs:
        stack = [d]
        while stack:
            cur = stack.pop()
            try:
                with os.scandir(cur) as it:
                    items = sorted(it, key=lambda e: e.name)
            except OSError:
                continue
            subdirs = []
            for e in items:
                try:
                    if e.is_dir(follow_symlinks=False):
                        if recursive and not _skip_dir(e.name):
                            subdirs.append(Path(e.path))
                        continue
                    if not e.is_file(follow_symlinks=True):
                        continue
                except OSError:
                    continue
                if os.path.splitext(e.name)[1].lower() not in ex:
                    continue
                rp = os.path.realpath(e.path)
                if rp in seen:                       # 겹치는 폴더를 두 번 고른 경우
                    continue
                seen.add(rp)
                n += 1
                if n > MAX_SCAN_FILES:
                    raise FolderError(400, "too_many_files", f"파일이 너무 많습니다 ({MAX_SCAN_FILES:,}개 초과) — 폴더를 좁혀 주세요")
                yield Path(e.path)
            stack.extend(reversed(subdirs))


def id_groups(path: Path) -> List[str]:
    g = _ID_RX.findall(nfc(path.stem))
    if not g:
        g = _ID_RX.findall(nfc(path.parent.name))     # 세트마다 폴더를 나눈 경우: 13387752/계약서.pdf
    return g


def choose_id(groups: Sequence[str], prefer: Optional[set] = None) -> Optional[str]:
    if not groups:
        return None
    if prefer:
        for g in groups:
            if g in prefer:
                return g
    for g in groups:
        if not g.startswith("0"):
            return g
    return groups[0]


# ---------------------------------------------------------------------------
# 정답 파일 읽기 (jsonl · json · csv · xlsx → 평가기의 gold 레코드)
# ---------------------------------------------------------------------------
_SET_ID_HEADERS = {"set_id", "setid", "세트id", "세트", "work_id", "workid", "id", "원문인덱스", "관리번호", "저작물id"}
_TEST_HEADERS = {"test", "시험", "시험구분"}
# 시험 이름 — 절차서는 Te-1 · Te-2, 내부 코드(정답·목록 파일·API)는 T-001 · T-002. 입력은 둘 다 받는다.
_TEST_CODES = {"T-001": "T-001", "T001": "T-001", "TE-1": "T-001", "TE1": "T-001",
               "T-002": "T-002", "T002": "T-002", "TE-2": "T-002", "TE2": "T-002"}
TEST_LABELS = {"T-001": "Te-1", "T-002": "Te-2"}


def norm_test(v: Any) -> Optional[str]:
    """시험 구분 → 내부 코드 (T-001 · T-002). Te-1 · te1 · T001 등도 받는다. 모르는 값은 None."""
    if v is None:
        return None
    return _TEST_CODES.get(re.sub(r"\s+", "", str(v)).upper())


def test_label(t: Optional[str]) -> Optional[str]:
    """화면·이름에 쓰는 시험 이름 (T-001 → Te-1)."""
    return TEST_LABELS.get(t, t) if t else t
_MEDIA_HEADERS = {"media", "매체"}
_BUCKET_HEADERS = {"bucket", "license_bucket", "구분", "이용조건"}


def _hkey(h: Any) -> str:
    return re.sub(r"[\s_\-·.()]", "", nfc(str(h or ""))).lower()


def _key_headers(keys: Sequence[Any]) -> Dict[str, str]:
    """머리글 → 키 이름. 영문 키 이름, 한국어 항목명, '1 저작물명'·'1. 저작물명' 모두 받는다."""
    out: Dict[str, str] = {}
    for k in keys:
        for h in (k.name, k.label_ko, f"{k.no}{k.label_ko}", f"{k.no}{k.name}"):
            out[_hkey(h)] = k.name
    return out


def _split_list(v: str, hashtag: bool) -> List[str]:
    parts = [p.strip() for p in re.split(r"[;|\n]|,(?![^()]*\))", v)]
    out = []
    for p in parts:
        if hashtag:
            p = p.lstrip("#").strip()
        if p:
            out.append(p)
    return out


def _cell_value(key, raw: Any) -> Tuple[Any, str, Optional[str]]:
    """표의 한 칸 → (값, 상태, 문제). 빈 칸 = na(채점 제외), '(빈값)' = empty(값이 없어야 정답)."""
    if raw is None:
        return None, "na", None
    if isinstance(raw, (datetime, date)):
        raw = (raw.date() if isinstance(raw, datetime) else raw).isoformat()
    if isinstance(raw, float) and raw.is_integer():
        raw = int(raw)
    s = nfc(str(raw)).strip()
    if not s or s.upper() in ("NULL", "NONE", "N/A", "NA", "-"):
        return None, "na", None
    if s.lower() in EMPTY_MARKERS:
        return None, "empty", None
    if key.type in ("name_list", "tag_list"):
        items = _split_list(s, hashtag=(key.name == "hashtags"))
        return (items or None), ("value" if items else "na"), None
    if key.type == "int":
        digits = re.sub(r"[,\s]|bytes?$|바이트$", "", s, flags=re.I)
        if digits.isdigit():
            return int(digits), "value", None
        return s, "value", f"{key.name} 값이 숫자가 아닙니다: {s[:30]}"
    return s, "value", None


def _record_from_flat(row: Dict[str, Any], keys, kmap: Dict[str, str], source: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    problems: List[str] = []
    sid = test = media = bucket = None
    vals: Dict[str, Any] = {}
    for h, v in row.items():
        hk = _hkey(h)
        if hk in _SET_ID_HEADERS and sid is None:
            sid = v
        elif hk in _TEST_HEADERS:
            test = v
        elif hk in _MEDIA_HEADERS:
            media = v
        elif hk in _BUCKET_HEADERS:
            bucket = v
        elif hk in kmap:
            vals[kmap[hk]] = v
    if sid is None or str(sid).strip() == "":
        return None, problems
    sid = str(int(sid)) if isinstance(sid, float) and sid.is_integer() else nfc(str(sid)).strip()
    rec: Dict[str, Any] = {"set_id": sid, "source": source, "keys": {}}
    if norm_test(test):
        rec["test"] = norm_test(test)
    if media and str(media).strip():
        rec["media"] = str(media).strip()
    if bucket and str(bucket).strip():
        rec["bucket"] = str(bucket).strip()
    for k in keys:
        value, state, prob = _cell_value(k, vals.get(k.name))
        rec["keys"][k.name] = {"value": value, "state": state}
        if prob:
            problems.append(f"{sid}: {prob}")
    return rec, problems


def _record_from_json(obj: Dict[str, Any], keys, kmap, source: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    if not isinstance(obj, dict):
        return None, []
    if isinstance(obj.get("keys"), dict):               # 평가기 형식 — 그대로 (값만 다듬는다)
        sid = obj.get("set_id")
        if sid is None or str(sid).strip() == "":
            return None, []
        rec = {**obj, "set_id": nfc(str(sid)).strip(), "keys": {}}
        if rec.get("test") is not None:
            rec["test"] = norm_test(rec["test"]) or rec["test"]
        for k in keys:
            e = obj["keys"].get(k.name)
            if isinstance(e, dict):
                rec["keys"][k.name] = e
            elif e is None:
                rec["keys"][k.name] = {"value": None, "state": "na"}
            else:
                v, st, _ = _cell_value(k, e if not isinstance(e, list) else ";".join(map(str, e)))
                rec["keys"][k.name] = {"value": v, "state": st}
        rec.setdefault("source", source)
        return rec, []
    return _record_from_flat(obj, keys, kmap, source)


def _read_table_csv(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise FolderError(400, "answer_encoding", "정답 CSV 의 글자 인코딩을 읽을 수 없습니다 (UTF-8 또는 CP949)")
    return list(csv.DictReader(io.StringIO(text)))


def _read_table_xlsx(path: Path) -> List[Dict[str, Any]]:
    try:
        import openpyxl
    except ImportError:
        raise FolderError(400, "answer_xlsx", "엑셀 정답을 읽을 수 없습니다 (openpyxl 없음) — CSV 로 저장해 주세요")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        ws = wb.worksheets[0]
        rows = ws.iter_rows(values_only=True)
        header = None
        out = []
        for r in rows:
            if header is None:
                if r and any(c not in (None, "") for c in r):
                    header = [str(c).strip() if c is not None else "" for c in r]
                continue
            if not r or all(c in (None, "") for c in r):
                continue
            out.append({header[i]: r[i] for i in range(min(len(header), len(r))) if header[i]})
        return out
    finally:
        wb.close()


@dataclass
class Answers:
    path: Optional[Path]
    fmt: str
    records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    skipped_rows: int = 0


def load_answers(path: Path, keys, test: Optional[str] = None) -> Answers:
    """정답 파일 → Answers. test 를 주면 그 시험 레코드만 (test 칸이 없는 레코드는 함께 쓴다)."""
    ext = path.suffix.lower()
    kmap = _key_headers(keys)
    src = f"answer:{path.name}"
    ans = Answers(path, ext.lstrip("."))
    rows: List[Any]
    if ext == ".jsonl":
        rows = []
        with path.open(encoding="utf-8-sig") as f:
            for no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    ans.problems.append(f"{no}번째 줄을 JSON 으로 읽을 수 없습니다")
                    ans.skipped_rows += 1
    elif ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = data.get("records") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise FolderError(400, "answer_format", "정답 JSON 은 레코드 목록이어야 합니다")
    elif ext == ".csv":
        rows = _read_table_csv(path)
    elif ext == ".xlsx":
        rows = _read_table_xlsx(path)
    else:
        raise FolderError(400, "answer_format", "정답 파일은 .jsonl · .json · .csv · .xlsx 만 받습니다")
    if ext in (".csv", ".xlsx") and rows:
        heads = {_hkey(h) for h in rows[0].keys()}
        if not heads & _SET_ID_HEADERS:
            raise FolderError(400, "answer_no_id", "정답 표에 세트 ID 열(set_id)이 없습니다")
        have = {kmap[h] for h in heads if h in kmap}
        ans.missing_columns = [k.name for k in keys if k.name not in have and (test is None or k.test == test)]
    for r in rows:
        rec, probs = _record_from_json(r, keys, kmap, src)
        ans.problems.extend(probs)
        if rec is None:
            ans.skipped_rows += 1
            continue
        if test and rec.get("test") and rec["test"] != test:
            continue
        ans.records[rec["set_id"]] = rec                  # 같은 ID 가 여러 줄이면 마지막 줄 (평가기와 같다)
    return ans


def load_gold_jsonl(path: Path, test: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("set_id") is not None and (not test or rec.get("test") is None or norm_test(rec.get("test")) == test):
                out[str(rec["set_id"])] = rec
    return out


# ---------------------------------------------------------------------------
# 세트 만들기
# ---------------------------------------------------------------------------
def _gold_value(rec: Optional[Dict[str, Any]], key: str) -> Any:
    e = ((rec or {}).get("keys") or {}).get(key)
    return e.get("value") if isinstance(e, dict) else None


DOC_EXT = (".pdf", ".hwp", ".hwpx", ".docx")


def _pick(files: List[Path], want_name: Any, doc_first: bool = False) -> Tuple[Path, bool]:
    """같은 ID 의 파일 여럿 → 정답 파일명과 같은 것, 없으면 이름순 첫 파일(계약서는 문서 형식 먼저).
    (고른 파일, 정답 파일명으로 골랐는지)."""
    if len(files) > 1 and isinstance(want_name, str) and want_name.strip():
        w = nfc(want_name.strip())
        for f in files:
            if nfc(f.name) == w:
                return f, True
    return sorted(files, key=lambda p: (doc_first and p.suffix.lower() not in DOC_EXT, nfc(p.name), str(p)))[0], False


@dataclass
class TestScan:
    test: str
    entries: list = field(default_factory=list)
    scored: int = 0
    missing: List[Dict[str, str]] = field(default_factory=list)
    duplicates: List[Dict[str, Any]] = field(default_factory=list)
    unkeyed: List[str] = field(default_factory=list)
    no_id: List[str] = field(default_factory=list)
    n_contract_files: int = 0
    n_work_files: int = 0
    paired: int = 0


def build_sets(test: str, contracts: List[Path], works: List[Path], answers: Dict[str, Dict[str, Any]],
               *, known: Optional[set] = None, include_unkeyed: bool = False, media_of=None,
               manifest_entry=None) -> TestScan:
    """폴더에서 찾은 파일 + 정답 → 매니페스트 항목. media_of(path) 와 manifest_entry(**kw) 는 호출자가 넘긴다."""
    ts = TestScan(test)
    prefer = set(answers) | (known or set())
    if test == "T-002":                                   # 같은 폴더를 둘 다 골랐을 때 저작물 그림을 계약서로 잡지 않게
        wset = {str(p) for p in works}
        contracts = [p for p in contracts if str(p) not in wset]
    by_c: Dict[str, List[Path]] = {}
    by_w: Dict[str, List[Path]] = {}
    for files, bucket_map, counter in ((contracts, by_c, "n_contract_files"), (works if test == "T-002" else [], by_w, "n_work_files")):
        for p in files:
            setattr(ts, counter, getattr(ts, counter) + 1)
            sid = choose_id(id_groups(p), prefer)
            if sid is None:
                if len(ts.no_id) < MAX_EXAMPLES * 4:
                    ts.no_id.append(display_path(p))
                continue
            bucket_map.setdefault(sid, []).append(p)
    primary = by_c if test == "T-001" else by_w
    if answers:
        ids = sorted(answers)
        extra = sorted(set(primary) - set(answers))
        ts.unkeyed = extra
        if include_unkeyed:
            ids += extra
    else:
        ids = sorted(primary)
    for sid in ids:
        rec = answers.get(sid)
        files = primary.get(sid)
        if not files:
            ts.missing.append({"set_id": sid, "reason": "계약서 파일 없음" if test == "T-001" else "저작물 파일 없음"})
            continue
        main, by_name = _pick(files, _gold_value(rec, "file_name"), doc_first=(test == "T-001"))
        if len(files) > 1:
            ts.duplicates.append({"set_id": sid, "files": [display_path(f) for f in files][:6], "used": display_path(main),
                                  "by_answer": by_name})
        contract = work = None
        if test == "T-001":
            contract = main
        else:
            work = main
            cfiles = by_c.get(sid)
            if cfiles:
                contract, _ = _pick(cfiles, None, doc_first=True)
                ts.paired += 1
        media = str((rec or {}).get("media") or (media_of(work) if work else "text") or "image")
        if test == "T-002" and work is not None:
            media = media_of(work) or media
        bucket = str((rec or {}).get("bucket") or "")
        title = _gold_value(rec, "work_title")
        extra_info: Dict[str, Any] = {"cell": f"{media}/{bucket}" if bucket else media, "file": main.name}
        if isinstance(title, str) and title.strip():
            extra_info["title"] = title.strip()
        if rec is not None:
            ts.scored += 1
        ts.entries.append(manifest_entry(set_id=sid, contract=str(contract) if contract else None,
                                         work=str(work) if work else None, media=media, license_bucket=bucket,
                                         test=test, extra=extra_info))
    return ts
