"""시험 콘솔 API — 로그인한 운영자만 쓰는 /console 화면과 /api/test/* (시험 실행·진행·결과·근거).

무엇을 하는가
  · 시험 모드: 계약서(Te-1) 또는 저작물(Te-2, 계약서는 문맥용) 파일 1건을 올려 바로 돌리고 23개 항목을 본다.
  (화면의 시험 이름은 절차서의 Te-1 · Te-2, 내부 코드·파일·API 는 T-001 · T-002 — 입력은 둘 다 받는다)
  · 일괄 시험: 고정 시험 목록(csv)에서 셀·건수를 골라 돌린다.
  · 진행: SSE 로 세트별 시작·종료·로그·단계를 흘려보낸다 (`after` 로 이어 붙인다 — /api/batch/{id}/stream 과 같은 방식).
  · 결과·지표: 평가기 산출물(test_metrics.json · per_set.jsonl)을 그대로 읽어 세트×항목 판정 격자와 근거를 만든다.
  · 실행 이력: 실행 폴더가 디스크에 남아 있어 서버를 다시 띄워도 결과와 근거가 그대로 열린다.

**채점은 다시 구현하지 않는다** — BatchRunner(실행·기록) + finalize_eval23(평가기)를 그대로 부른다. 실행마다
module.evaluation.cli.build_orchestrator(out_dir) 로 오케스트레이터를 따로 만들어 단계별 산출물이 그 실행 폴더
(out/runs/{uploads,results,llm_results}) 에 모인다.

저장 위치: api/web/test_runs/<run_id>/
  run.json · manifest.jsonl · inputs/(단건 업로드) · out/(BatchRunner 출력) · cache/(미리보기)
run_id = YYYYMMDD_HHMMSS_{single|batch}_{hex4}. 상태: queued → running → scoring → done | cancelled | error | interrupted.

이 모듈은 torch 를 올리지 않는다 — 평가·실행 모듈은 실행 스레드·핸들러 안에서만 import 한다.
화면에 보이는 문자열은 한국어이고 버전 표기는 "v1.0" 이다 — 내부 단계 이름은 응답·내려받기에서 지운다(_scrub_*).
"""
from __future__ import annotations

import asyncio
import io
import json
import math
import logging
import os
import random
import re
import secrets
import shutil
import threading
import time
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Iterator, List, Optional, Tuple
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response,
                               StreamingResponse)
from fastapi.routing import APIRoute

from . import console_auth as auth
from . import console_folder as CF

try:                                     # 앱은 api/ 를 루트로 올린다(module.*) — 다른 경로로 올려도 같은 묶음 변수를 쓴다
    from module import log_context as LC
except ImportError:                      # pragma: no cover
    from api.module import log_context as LC

logger = logging.getLogger(__name__)

VERSION = "v1.0"
SYSTEM_NAME = "저작물 속성정보 추출 시스템"
API = "/api/test"
PASS_MARK = 0.85
ACTIVE_STATES = ("queued", "running", "scoring")
TERMINAL_STATES = ("done", "cancelled", "error", "interrupted")
TESTS = ("T-001", "T-002")
EVENT_BUFFER = 20000
LOG_REPLAY_LINES = 400
MAX_UPLOAD_BYTES = 4 * 1024 ** 3
MAX_WORKERS = 200
# 기본 작업자 — 부하 시험 (docs/v4/load_test_20261019.md). Te-1 은 생각 끔(기본, 2026-10-20)에서 50: 60 이면 토큰이 분당
# 0.82–0.93M 으로 한도(1M/분)에 닿아 429 가 났다(200건 중 8회) — 50 ≈ 85 %. Te-2 는 E:(외장 USB 하드) 읽기가 병목이라
# 작업자를 늘려도 빨라지지 않는다 (재시험 값으로 정함).
DEFAULT_WORKERS = {"T-001": 50, "T-002": 32}
DEFAULT_TAU = 0.65
PING_TTL_SEC = 60.0
PREVIEW_MAX_SIDE = 1280
CONTRACT_DPI = 120
MAX_CONTRACT_PAGES = 40

# 추정치 (2026-10-10/11 실측). 시간은 세트당 '실효' 시간 — 기준 workers 에서 잰 값이고 workers 에 반비례로 본다.
# Te-1 은 생각(thinking) 설정에 따라 다르다 — 기본 끔. 켬(EXTRACTION_THINKING / CONSOLIDATION_THINKING=on) 값은 2026-10-19 실측.
EST_T001_KRW = 25.5            # 끔: 추출 ₩10.8 + 중재 ₩14.7 (계약서 200건 토큰 실측) · 켬 ₩44.8
EST_T001_KRW_THINKING = 44.8
# Te-1 계약서 1건의 소요(중앙값, 초) — 작업자 수별 실측. 작업자가 늘면 공용 GPU(NER)·중재자 응답이 느려진다.
T001_LATENCY = ((40, 50.0), (60, 65.0))          # 끔: 200건(스캔 25 %) 40·60 작업자
T001_LATENCY_THINKING = ((3, 85.0), (16, 77.0), (32, 87.0), (48, 91.0), (60, 102.0), (80, 117.0), (100, 127.0))
T001_TOKENS_PER_SET = 17000    # 끔: 추출+중재 2회 호출의 입력+출력 (실측 16.7~17.0k) · 켬 24.4k
T001_TOKENS_PER_SET_THINKING = 24400
DASHSCOPE_TPM = 1_000_000     # qwen3.5-122b-a10b 계정 한도 — 넘으면 429 로 기다리므로 처리량 상한이 된다
EST_T002_REF_WORKERS = 6      # 아래 EST_T002_SEC 를 잰 작업자 수
# Te-2 는 저작물 드라이브 읽기가 상한 — 드라이브마다 한 번에 하나씩 순서대로 복사(+SHA-256)하면 E:(외장 USB) 에서 약 68 MB/s
# (2026-10-20 실측, 100건 7.5 GB). 장세영 목록 평균 크기: 이미지 4.6 MB · 영상 188 MB (1,990건 375 GB).
EST_T002_MB = {"image": 4.6, "video": 188.0}
EST_DISK_MBPS = 68.0
EST_T002_SEC = 5.0            # 6 workers 기준 (대량 실행 처리량)
EST_T002_SET_SEC = 37.0       # 세트 하나의 소요(저작물 leg, 실측 2026-10-11 콘솔 점검)
EST_WORK_CONSOLIDATION_SEC = 55.0   # 저작물 leg 중재자를 켜면 세트당 더해지는 시간(직렬 기준)

MODELS = {"ocr": "qwen3.7-plus", "ocr_fallback": "qwen3.8-max", "extraction": "qwen3.5-122b-a10b",
          "ner": "klue-roberta-large"}
_IMAGE_CHAIN_FALLBACK = ["qwen3.8-omni-flash (DashScope)", "Qwen3.7-Plus (DashScope)",
                         "Gemma 4 31B (Google AI Studio)", "Gemma 4 31B (OpenRouter)", "Gemma 4 31B (연구실 서버)"]
_VIDEO_CHAIN_FALLBACK = ["qwen3.8-omni-flash (DashScope)", "Qwen3.5-Omni-Plus (DashScope)",
                         "Qwen3.7-Plus (DashScope)", "Gemma 4 31B (OpenRouter)", "Gemma 4 31B (연구실 서버)"]

_HERE = Path(os.path.abspath(__file__)).parent         # api/web (심볼릭 링크를 풀지 않는다)
REPO_ROOT = _HERE.parents[1]
DEFAULT_RUNS_ROOT = _HERE / "test_runs"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".webm", ".m4v", ".mpg", ".mpeg", ".flv", ".swf"}
PPTX_EXT = {".pptx"}
WORK_EXT = IMAGE_EXT | VIDEO_EXT | PPTX_EXT
CONTRACT_EXT = {".pdf", ".docx", ".hwp", ".hwpx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_MEDIA_FILE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mkv", ".mov", ".wmv", ".avi", ".zip", ".gz"}

RIGHTS_MAP = {   # 23키 → 평면 스키마 필드 (eval23_serializer.RIGHTS_MAP 과 같다)
    "reproduction_right": "reproduction_right",
    "performance_right": "public_performance_right",
    "public_transmission_right": "public_transmission_right",
    "exhibition_right": "exhibition_right",
    "distribution_right": "distribution_right",
    "rental_right": "rental_right",
    "derivative_work_right": "derivative_work_creation_right",
}
_TAG_KEYS = ("keywords", "hashtags")
VERDICTS = ("NA", "TP", "PARTIAL", "WRONG", "FP", "FN", "EMPTY_OK")


# ---------------------------------------------------------------------------
# 내부 이름 지우기 — 화면·내려받기에 'v4'·단계 이름이 보이지 않게
# ---------------------------------------------------------------------------
_SCRUB_COMMON = [
    (re.compile(r"docs/v4/[\w./-]+\.md"), "평가 규칙서"),
    (re.compile(r"[\w./-]*_v4/"), ""),                                    # dataset/test_set_v4/… → 파일 이름만
    (re.compile(r"\s*\((?:v4|M\d[a-z]?)\b[^()\n]{0,40}\)"), ""),           # "(v4 A3)", "(M2 A2)", "(M2b 수정 4)"
]
_SCRUB_TEXT_ONLY = [
    (re.compile(r"(?m)^(#+\s*)v4\s+"), r"\1"),                              # 보고서 제목 "# v4 평가 보고서"
    (re.compile(r"(?m)^(\s*)python -m api\.module\.evaluation\.v4\.evaluate\b.*$"),
     r"\1(관리자 평가 명령으로 다시 채점할 수 있습니다)"),
    (re.compile(r"\bv4\b"), "v1.0"),
    (re.compile(r"\bM[1-3]b?\b ?"), ""),
]


def scrub_text(s: Optional[str]) -> Optional[str]:
    """화면·내려받기용 문자열 — 비밀값을 가리고(log_context.scrub_secret) 내부 단계 이름을 지운다."""
    if not isinstance(s, str) or not s:
        return s
    s = LC.scrub_secret(s)
    for rx, rep in _SCRUB_COMMON:
        s = rx.sub(rep, s)
    for rx, rep in _SCRUB_TEXT_ONLY:
        s = rx.sub(rep, s)
    return s


def scrub_json(obj: Any) -> Any:
    """JSON 값 — 문자열이 정확히 'v4' 면 'v1.0', 그 밖에는 내부 문서·단계 표기만 지운다 (내용 문자열은 건드리지 않는다)."""
    if isinstance(obj, str):
        if obj == "v4":
            return "v1.0"
        if "v4" in obj or "M" in obj:
            for rx, rep in _SCRUB_COMMON:
                obj = rx.sub(rep, obj)
        return obj
    if isinstance(obj, dict):
        return {k: scrub_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub_json(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# 오류 · 라우트 클래스
# ---------------------------------------------------------------------------
class ConsoleError(Exception):
    def __init__(self, status: int, error: str, message: str = "", headers: Optional[Dict[str, str]] = None,
                 **extra: Any):
        super().__init__(message or error)
        self.status = status
        self.headers = headers
        self.payload = {"error": error, **({"message": message} if message else {}), **extra}


class _ConsoleRoute(APIRoute):
    """ConsoleError → {"error": …} JSON. 의존성(로그인 확인)에서 던진 것도 여기서 잡힌다."""

    def get_route_handler(self) -> Callable:
        handler = super().get_route_handler()

        async def wrapped(request: Request) -> Response:
            try:
                return await handler(request)
            except ConsoleError as e:
                return JSONResponse(e.payload, status_code=e.status, headers=e.headers)
        return wrapped


router = APIRouter(route_class=_ConsoleRoute)


def _not_found(message: str = "찾을 수 없습니다") -> ConsoleError:
    return ConsoleError(404, "not_found", message)


# ---------------------------------------------------------------------------
# 전역 상태 · 초기화
# ---------------------------------------------------------------------------
_ROOT: Path = DEFAULT_RUNS_ROOT
_REPO: Path = REPO_ROOT
_TEMPLATES = None
_SOURCES: List[Dict[str, Any]] = []
_SINGLE_GOLD: Dict[str, Optional[Path]] = {}
_RUNS: Dict[str, "TestRun"] = {}
_RUNS_LOCK = threading.RLock()


def _default_orchestrator(out_dir: Path):
    from module.evaluation.cli import build_orchestrator      # 무거운 import (torch) — 실행 스레드에서만
    return build_orchestrator(out_dir)


def _default_runner(orch, cfg, out_dir: Path, progress_cb):
    from module.evaluation.batch_runner import BatchRunner
    return BatchRunner(orch, cfg, out_dir, progress_cb=progress_cb)


_ORCH_FACTORY: Callable = _default_orchestrator
_RUNNER_FACTORY: Callable = _default_runner


def default_sources(repo: Path) -> List[Dict[str, Any]]:
    tv = repo / "dataset" / "test_set_v4"
    return [
        {"id": "t001_new5", "label": "Te-1 신규 양식 계약서", "test": "T-001",
         "csv": [tv / "t001_new5_ids.csv"], "gold": tv / "gold23_t001_new5.jsonl"},
        {"id": "t002", "label": "Te-2 고정 시험 목록", "test": "T-002",
         "csv": [tv / "t002_ids.csv"], "gold": tv / "gold23_t002.jsonl"},
        {"id": "pilot200", "label": "Te-2 파일럿 200", "test": "T-002",
         "csv": [repo / "dataset" / "pilot200_v4" / "pilot_ids.csv"], "gold": tv / "gold23_t002.jsonl"},
        {"id": "t001", "label": "Te-1 고정 시험 목록", "test": "T-001",
         "csv": [tv / "t001_ids.csv"], "gold": None, "enabled": False, "reason": "신규 양식 계약서 대기"},
        {"id": "ids", "label": "ID 목록 직접 입력", "test": None, "by_ids": True,
         # T-001 은 신규 양식 계약서가 있는 목록을 먼저 본다 — 같은 id 의 옛 양식 계약서로 채점하지 않게
         "csv_by_test": {"T-001": [tv / "t001_new5_ids.csv", tv / "t001_ids.csv"], "T-002": [tv / "t002_ids.csv"]},
         "gold_by_test": {"T-001": tv / "gold23_t001_new5.jsonl", "T-002": tv / "gold23_t002.jsonl"}},
    ]


def default_single_gold(repo: Path) -> Dict[str, Optional[Path]]:
    tv = repo / "dataset" / "test_set_v4"
    return {"T-001": tv / "gold23_t001_new5.jsonl", "T-002": tv / "gold23_t002.jsonl"}


def init_console_api(templates=None, runs_root: Optional[os.PathLike] = None,
                     orchestrator_factory: Optional[Callable] = None,
                     runner_factory: Optional[Callable] = None,
                     repo_root: Optional[os.PathLike] = None,
                     sources: Optional[List[Dict[str, Any]]] = None,
                     single_gold: Optional[Dict[str, Optional[os.PathLike]]] = None,
                     env_path: Optional[os.PathLike] = None) -> None:
    """app.py 에서 1회 호출 — 화면 템플릿·저장 위치를 정하고 디스크의 실행을 되살린다. 테스트는 가짜 공장을 넘긴다."""
    global _TEMPLATES, _ROOT, _REPO, _SOURCES, _SINGLE_GOLD, _ORCH_FACTORY, _RUNNER_FACTORY
    _TEMPLATES = templates
    _ROOT = Path(runs_root) if runs_root is not None else DEFAULT_RUNS_ROOT
    _REPO = Path(repo_root) if repo_root is not None else REPO_ROOT
    _SOURCES = sources if sources is not None else default_sources(_REPO)
    sg = single_gold if single_gold is not None else default_single_gold(_REPO)
    _SINGLE_GOLD = {k: (Path(v) if v else None) for k, v in sg.items()}
    _ORCH_FACTORY = orchestrator_factory or _default_orchestrator
    _RUNNER_FACTORY = runner_factory or _default_runner
    auth.configure(env_path)
    _ENTRY_CACHE.clear()
    try:
        _ROOT.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error("시험 실행 폴더를 만들 수 없습니다 %s: %s", _ROOT, e)
    with _RUNS_LOCK:
        _RUNS.clear()
    install_log_capture()
    _recover_runs()


# ---------------------------------------------------------------------------
# 작은 도구
# ---------------------------------------------------------------------------
def _iso(ts: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(ts) if ts is not None else datetime.now()
    return dt.astimezone().isoformat(timespec="seconds")


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{threading.get_ident()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    try:
        f = path.open(encoding="utf-8")
    except OSError:
        return
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _sig(path: Path) -> Optional[Tuple[int, int]]:
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _inside(path: os.PathLike, root: os.PathLike) -> bool:
    try:
        p = os.path.realpath(path)
        r = os.path.realpath(root)
    except (OSError, ValueError):
        return False
    return p == r or p.startswith(r.rstrip(os.sep) + os.sep)


def _safe_set_filename(set_id: str) -> str:
    """batch_runner._safe_set_filename 과 같은 규칙 — test_predictions/<set_id>.json."""
    s = str(set_id).strip() or "_"
    return s.replace(os.sep, "_").replace("/", "_")


def _cache_name(set_id: str) -> str:
    return re.sub(r"[^\w.-]", "_", str(set_id))[:80] or "_"


_SET_ID_RX = re.compile(r"^[\w.\-]{1,80}$")
_RID_RX = re.compile(r"^[\w.\-]{1,80}$")
_FRAME_RX = re.compile(r"^[\w.\-]{1,120}\.(?:jpg|jpeg|png)$", re.I)
_FRAME_T_RX = re.compile(r"_t(\d+(?:\.\d+)?)s\.", re.I)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _test_param(v: Any) -> Optional[str]:
    """요청의 시험 구분 → 내부 코드 (Te-1 · T-001 · t001 → T-001). 모르는 값은 대문자 그대로 (뒤에서 bad_test)."""
    return CF.norm_test(v) or (str(v or "").strip().upper() or None)


def _int_param(v: Any, name: str, default: Optional[int], lo: int, hi: int) -> Optional[int]:
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    try:
        n = int(str(v).strip())
    except (TypeError, ValueError):
        raise ConsoleError(400, "bad_request", f"{name} 는 정수여야 합니다")
    return int(_clamp(n, lo, hi))


def _float_param(v: Any, name: str, default: Optional[float], lo: float, hi: float) -> Optional[float]:
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        raise ConsoleError(400, "bad_request", f"{name} 는 숫자여야 합니다")
    if x != x:  # NaN
        raise ConsoleError(400, "bad_request", f"{name} 는 숫자여야 합니다")
    return _clamp(x, lo, hi)


def _bool_param(v: Any, default: bool) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on", "y"):
        return True
    if s in ("0", "false", "no", "off", "n", ""):
        return False
    return default


def _list_param(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        items = [str(x) for x in v]
    else:
        items = re.split(r"[\s,;]+", str(v))
    out, seen = [], set()
    for x in items:
        x = x.strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _value_short(v: Any, n: int = 60) -> Optional[str]:
    if v is None or v == "" or v == []:
        return None
    if isinstance(v, (list, tuple)):
        s = ", ".join(str(x) for x in v)
    elif isinstance(v, dict):
        s = json.dumps(v, ensure_ascii=False)
    else:
        s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt(v: Any) -> str:
    if v is None or v == "" or v == [] or v == {}:
        return "—"
    if isinstance(v, bool):
        return "예" if v else "아니오"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, (list, tuple)):
        parts = []
        for x in v:
            if isinstance(x, (list, tuple)) and len(x) == 2:
                parts.append(f"{x[0]} = {x[1]}")
            else:
                parts.append(_fmt(x))
        return ", ".join(parts)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _media_of_path(path: str) -> Optional[str]:
    ext = os.path.splitext(str(path))[1].lower()
    if ext in VIDEO_EXT:
        return "video"
    if ext in IMAGE_EXT or ext in PPTX_EXT:
        return "image"
    return None


def _work_kind(path: Optional[str], media: Optional[str]) -> Optional[str]:
    if not path:
        return None
    ext = os.path.splitext(str(path))[1].lower()
    if ext in PPTX_EXT:
        return "pptx"
    if ext in VIDEO_EXT or media == "video":
        return "video"
    if ext in IMAGE_EXT or media == "image":
        return "image"
    return None


# ---------------------------------------------------------------------------
# 실행 1건
# ---------------------------------------------------------------------------
class TestRun:
    __test__ = False          # pytest 가 테스트 클래스로 모으지 않게

    def __init__(self, run_id: str, kind: str, root: Path):
        self.run_id = run_id
        self.kind = kind                      # single | batch
        self.dir = Path(root) / run_id
        self.out_dir = self.dir / "out"
        self.name = run_id
        self.test: Optional[str] = None
        self.source: Optional[str] = None
        self.state = "queued"
        self.created_at = _iso()
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.n_sets = 0
        self.config: Dict[str, Any] = {}      # 화면에 보이는 설정
        self.cfg_kwargs: Dict[str, Any] = {}  # RunConfig 인자 (내부)
        self.estimate: Optional[Dict[str, Any]] = None
        self.progress: Dict[str, Any] = {"i": 0, "total": 0, "failed": 0, "cost_krw_total": 0.0, "last_set": None}
        self.summary: Optional[Dict[str, Any]] = None
        self.final: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.set_id: Optional[str] = None     # 단건
        self.gold = False
        self.gold_path: Optional[str] = None
        self.runner = None
        self.cancel_requested = False
        self.cost_exceeded = False
        self.threads: Dict[int, str] = {}     # (예전 방식, 쓰지 않음) — 세트 묶음은 log_context(contextvars) 로 한다
        self._pred_count: Tuple[Any, float, int] = (None, 0.0, 0)   # (상태·진행, 센 시각, 개수) — 상세 조회마다 4,000개를 세지 않게
        self.file_to_set: Dict[str, str] = {}
        self.running_sets: set = set()
        self.set_info: Dict[str, Dict[str, Any]] = {}   # set_id → {cell, media, title} (set_start·set_done 이벤트에 붙인다)
        self._events: Deque[Dict[str, Any]] = deque(maxlen=EVENT_BUFFER)
        self._seq = 0
        self._lock = threading.RLock()
        self._last_save = 0.0
        self._log_handler: Optional[logging.Handler] = None
        self._index: Optional[Tuple[Any, Dict[str, Any]]] = None

    # -- 이벤트 --------------------------------------------------------------
    def push(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._seq += 1
            ev = {**ev, "seq": self._seq, "ts": round(time.time(), 3)}
            self._events.append(ev)
            return ev

    def events_after(self, seq: int) -> List[Dict[str, Any]]:
        with self._lock:
            if not self._events or self._events[-1]["seq"] <= seq:
                return []
            return [e for e in self._events if e["seq"] > seq]

    def set_state(self, state: str, **extra: Any) -> None:
        with self._lock:
            if state == "running" and not self.started_at:
                self.started_at = _iso()
            if state in TERMINAL_STATES:
                self.finished_at = self.finished_at or _iso()
            # 이벤트를 먼저 쌓고 상태를 바꾼다 — 스트림이 '끝난 상태 + 남은 이벤트 없음' 을 보고 닫을 때 마지막 상태 이벤트를 놓치지 않게
            self.push({"type": "state", "state": state, **extra})
            self.state = state
            self.save()

    def finish(self, state: str) -> None:
        self.running_sets.clear()
        self.set_state(state, error=self.error)

    # -- BatchRunner 콜백 ----------------------------------------------------
    def on_runner_event(self, ev: Dict[str, Any]) -> None:
        t = ev.get("type")
        if t == "start":
            with self._lock:
                self.progress["total"] = int(ev.get("total") or 0)
            self.push({"type": "start", "total": ev.get("total"), "already_done": ev.get("already_done"),
                       "workers": ev.get("workers")})
        elif t == "progress":
            sid = str(ev.get("set_id"))
            self.running_sets.discard(sid)
            self.push({"type": "set_done", "set_id": sid, **self.set_info.get(sid, {}), "ok": bool(ev.get("ok")),
                       "elapsed_sec": ev.get("elapsed_sec"), "cost_krw": self._set_cost(sid),
                       "error": scrub_text(LC.scrub_secret(ev.get("error"))) if ev.get("error") else None})
            with self._lock:
                # 작업 스레드가 진행 이벤트를 잠금 밖에서 보내므로 순서가 뒤바뀔 수 있다 — 더 늦은(작은 i) 이벤트로 되돌리지 않는다
                if int(ev.get("i") or 0) >= int(self.progress.get("i") or 0):
                    self.progress = {"i": ev.get("i", 0), "total": ev.get("total", self.progress.get("total", 0)),
                                     "failed": ev.get("failed", 0), "cost_krw_total": ev.get("cost_krw_total", 0.0),
                                     "last_set": sid}
                prog = dict(self.progress)
            self.push({"type": "progress", **prog})
            self.save(throttle=True)
        elif t == "cost_limit":
            self.cost_exceeded = True

    def _set_cost(self, set_id: str) -> Optional[float]:
        rec = _read_json(self.out_dir / "test_predictions" / f"{_safe_set_filename(set_id)}.json")
        try:
            return float(((rec or {}).get("_meta") or {}).get("cost_krw"))
        except (TypeError, ValueError):
            return None

    # -- test.log → log 이벤트 ------------------------------------------------
    # log 이벤트: {"type":"log", "line": "HH:MM:SS LEVEL [set_id] [구성요소] 메시지", "level", "component", "set_id",
    #             "logger"?, "stage"?(진행 단계), "replayed"?(디스크에서 다시 읽은 줄)}
    def push_line(self, line: str, **extra: Any) -> None:
        """test.log 의 한 줄(새 형식·예전 형식) → log 이벤트 (level·component·set_id 를 줄에서 읽는다)."""
        f = LC.parse_line(line)
        self.push({"type": "log", "line": scrub_text(LC.scrub_secret(line)), "level": f["level"],
                   "component": f["component"], "set_id": f["set_id"], **extra})

    def attach_runner_log(self, runner) -> None:
        lg = getattr(runner, "log", None)
        log_path = getattr(runner, "log_path", self.out_dir / "test.log")
        try:
            for line in Path(log_path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.push_line(line)
        except OSError:
            pass
        if isinstance(lg, logging.Logger):
            h = _RunLogHandler(self)
            lg.addHandler(h)
            self._log_handler = h

    def detach_runner_log(self, runner) -> None:
        lg = getattr(runner, "log", None)
        if self._log_handler is not None and isinstance(lg, logging.Logger):
            lg.removeHandler(self._log_handler)
        self._log_handler = None

    def log_record(self, record: logging.LogRecord, desc: Dict[str, Any]) -> None:
        """파이프라인·모델 모듈의 기록(이 실행에 묶인 것) → test.log 한 줄 + log 이벤트."""
        w = getattr(self.runner, "write_log_record", None)
        if callable(w):
            try:
                w(record)
            except Exception:
                pass
        ev: Dict[str, Any] = {"type": "log", "line": scrub_text(_LINE_FMT.format(record)), "level": desc["level"],
                              "component": desc["component"], "set_id": desc["set_id"]}
        if desc.get("stage"):
            ev["stage"] = desc["stage"]
        self.push(ev)

    # -- 저장 · 복원 ------------------------------------------------------------
    def to_json(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "version": VERSION, "run_id": self.run_id, "kind": self.kind, "name": self.name,
                "test": self.test, "source": self.source, "state": self.state,
                "created_at": self.created_at, "started_at": self.started_at, "finished_at": self.finished_at,
                "n_sets": self.n_sets, "config": self.config, "estimate": self.estimate,
                "progress": self.progress, "summary": self.summary, "final": self.final, "error": self.error,
                "set_id": self.set_id, "gold": self.gold,
                "_internal": {"cfg_kwargs": self.cfg_kwargs, "gold_path": self.gold_path,
                              "file_to_set": self.file_to_set},
            }

    def save(self, throttle: bool = False) -> None:
        now = time.time()
        if throttle and now - self._last_save < 2.0:
            return
        self._last_save = now
        try:
            _write_json_atomic(self.dir / "run.json", self.to_json())
        except OSError as e:
            logger.warning("run.json 저장 실패 %s: %s", self.run_id, e)

    @classmethod
    def load(cls, d: Path) -> Optional["TestRun"]:
        data = _read_json(d / "run.json")
        if not isinstance(data, dict):
            return None
        run = cls(d.name, str(data.get("kind") or "batch"), d.parent)
        for k in ("name", "test", "source", "state", "created_at", "started_at", "finished_at", "estimate",
                  "summary", "final", "error", "set_id"):
            if k in data:
                setattr(run, k, data.get(k))
        run.n_sets = int(data.get("n_sets") or 0)
        run.config = data.get("config") or {}
        run.gold = bool(data.get("gold"))
        prog = data.get("progress")
        if isinstance(prog, dict):
            run.progress = {**run.progress, **prog}
        internal = data.get("_internal") or {}
        run.cfg_kwargs = internal.get("cfg_kwargs") or {}
        run.gold_path = internal.get("gold_path")
        run.file_to_set = internal.get("file_to_set") or {}
        run.name = run.name or run.run_id
        return run

    @property
    def pred_dir(self) -> Path:
        return self.out_dir / "test_predictions"

    def n_predictions(self) -> int:
        """생성된 속성정보 파일(test_predictions/<set_id>.json) 개수 — 실패한 세트도 한 파일씩 남는다. 실행 중에는 5초마다만 센다."""
        key = (self.state, self.progress.get("i"))
        k0, at, n = self._pred_count
        if k0 == key and (self.state in TERMINAL_STATES or time.time() - at < 5.0):
            return n
        try:
            n = sum(1 for e in os.scandir(self.pred_dir) if e.name.endswith(".json") and ".tmp" not in e.name)
        except OSError:
            n = 0
        self._pred_count = (key, time.time(), n)
        return n

    # -- 화면용 ----------------------------------------------------------------
    def row(self) -> Dict[str, Any]:
        f = self.final or {}
        ev = bool(f.get("evaluated"))
        with self._lock:
            prog = dict(self.progress)
        return {
            "run_id": self.run_id, "kind": self.kind, "name": self.name, "test": self.test,
            "source": self.source, "state": self.state, "created_at": self.created_at,
            "started_at": self.started_at, "finished_at": self.finished_at, "n_sets": self.n_sets,
            "done": prog.get("i", 0), "failed": prog.get("failed", 0),
            "cost_krw": round(float(prog.get("cost_krw_total") or 0.0) + float(self.config.get("cost_prev_krw") or 0.0), 2),   # 다시 실행한 회차 포함
            "macro_f1": f.get("macro_f1") if ev else None, "micro_f1": f.get("micro_f1") if ev else None,
            "passed": f.get("passed") if ev else None,
            "set_id": self.set_id, "gold": self.gold,
        }

    def detail(self) -> Dict[str, Any]:
        d = self.row()
        n_pred = self.n_predictions()
        with self._lock:
            prog = dict(self.progress)
        d.update({
            "config": self.config, "estimate": self.estimate,
            "progress": {k: prog.get(k) for k in ("i", "total", "failed", "cost_krw_total", "last_set")},
            "running_sets": sorted(self.running_sets),
            "summary": {**(self.summary or {}), "evaluation": self.final} if (self.summary or self.final) else None,
            "error": self.error,
            "downloads": [n for n in DOWNLOADS if n == "run.zip" or (self.out_dir / n).is_file()
                          or n == PREDICTIONS_ZIP and n_pred],
            "version": VERSION,
        })
        d = scrub_json(d)
        d["n_predictions"] = n_pred                                   # 절차서: 생성된 속성정보 파일(*.json)을 확인한다
        d["predictions_dir"] = CF.display_path(self.pred_dir)        # 시험 PC 의 폴더 (경로는 지우지 않는다)
        return d


_LINE_FMT = LC.ConsoleLineFormatter()


class _LocklessHandler(logging.Handler):
    """handle() 에서 핸들러 잠금을 잡지 않는다 — emit 안에서 실행(run)의 잠금을 잡기 때문에, 핸들러 잠금까지 겹쳐 잡으면
    '실행 잠금을 쥔 채 로그를 남기는' 경로(set_state → save 실패 경고)와 교착할 수 있다. emit 은 스스로 스레드 안전하다."""

    def handle(self, record: logging.LogRecord) -> bool:
        rv = self.filter(record)
        if rv:
            self.emit(record)
        return bool(rv)


class _RunLogHandler(_LocklessHandler):
    """BatchRunner 의 실행 로거(test.log)에 붙어 줄마다 log 이벤트를 만든다 (실행기 자신의 줄)."""
    _START_RX = re.compile(r"^\[(.+?)\] 시작 —")

    def __init__(self, run: TestRun):
        super().__init__(level=logging.INFO)
        self.run = run
        self.setFormatter(_LINE_FMT)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            d = LC.describe(record)
            ev: Dict[str, Any] = {"type": "log", "line": scrub_text(self.format(record)), "level": d["level"],
                                  "component": d["component"], "set_id": d["set_id"]}
            stage = d.get("stage") or getattr(record, "console_stage", None)
            if stage:
                ev["stage"] = stage
            else:
                m = self._START_RX.match(record.getMessage())
                if m:
                    sid = m.group(1)
                    self.run.running_sets.add(sid)
                    self.run.push({"type": "set_start", "set_id": sid, **self.run.set_info.get(sid, {})})
            self.run.push(ev)
        except Exception:      # 로그 처리 실패가 실행을 멈추면 안 된다
            pass


# ---------------------------------------------------------------------------
# 로그 수집 — 실행 로그(test.log · SSE) · 시스템 로그(링 버퍼, /api/test/syslog)
# ---------------------------------------------------------------------------
# 로거 이름은 import 경로를 따른다: 앱(api/ 에서 web.app:app)은 web.* · module.*, 평가 CLI 는 api.module.* —
# 세 갈래를 모두 잡는다. HTTP 호출은 httpx(요청 한 줄)·openai(재시도). 시스템 로그는 uvicorn.error 도 받는다.
CAPTURE_LOGGERS = ("web", "module", "api", "httpx", "openai")
SYSLOG_ONLY_LOGGERS = ("uvicorn.error",)
SYSLOG_MAX = 5000
_LEVEL_NO = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "WARN": 30, "ERROR": 40, "CRITICAL": 50}


class _SysLog:
    """서버 전체 로그의 마지막 SYSLOG_MAX 건 (seq 는 1부터 늘어난다 — after 로 이어 받는다)."""

    def __init__(self, maxlen: int = SYSLOG_MAX):
        self._buf: Deque[Dict[str, Any]] = deque(maxlen=maxlen)
        self._seq = 0
        self._lock = threading.Lock()

    def append(self, item: Dict[str, Any]) -> None:
        with self._lock:
            self._seq += 1
            item["seq"] = self._seq
            self._buf.append(item)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def query(self, after: int = 0, levels: Optional[set] = None, min_level: Optional[int] = None,
              components: Optional[set] = None, text: str = "", run_id: Optional[str] = None,
              set_id: Optional[str] = None, limit: int = 500) -> Dict[str, Any]:
        with self._lock:
            items = list(self._buf)
            last = self._seq
        first = items[0]["seq"] if items else last + 1
        text = (text or "").lower()

        def ok(it: Dict[str, Any]) -> bool:
            if levels and it["level"] not in levels:
                return False
            if min_level is not None and _LEVEL_NO.get(it["level"], 0) < min_level:
                return False
            if components and it["component"] not in components:
                return False
            if run_id and it.get("run_id") != run_id:
                return False
            if set_id and it.get("set_id") != set_id:
                return False
            if text and text not in " ".join(str(it.get(k) or "") for k in ("message", "logger", "set_id",
                                                                              "run_id")).lower():
                return False
            return True

        more = False
        if after <= 0:                           # 처음 열 때 — 조건에 맞는 마지막 limit 건
            hits = [it for it in items if ok(it)]
            more = len(hits) > limit
            out, scanned = hits[-limit:], last
        else:                                    # 이어 받기 — after 다음부터 오래된 순으로 limit 건
            out, scanned = [], last
            for it in items:
                if it["seq"] <= after or not ok(it):
                    continue
                if len(out) >= limit:
                    more = True
                    break
                out.append(it)
                scanned = it["seq"]
            if not more:
                scanned = last
        return {"items": [dict(it) for it in out], "last_seq": scanned, "first_seq": first, "more": more,
                "gap": bool(after > 0 and after + 1 < first)}


_SYSLOG = _SysLog()


_INTERNAL_SEGMENT = re.compile(r"^(?:v\d+|M\d[a-z]?)$")


def display_logger(name: Optional[str]) -> Optional[str]:
    """화면에 보일 로거 이름 — 'api.' 접두와 내부 단계 이름 조각('v4' 등)을 뺀다 (module.evaluation.v4.evaluator →
    module.evaluation.evaluator). 실행기 전용 로거는 'module.evaluation.batch_runner' 로 줄인다."""
    if not name:
        return name
    n = name[4:] if name.startswith("api.") else name
    if LC.is_runner_logger(n):
        n = n.split(".run.", 1)[0]
    return ".".join(seg for seg in n.split(".") if not _INTERNAL_SEGMENT.match(seg)) or n


class _CaptureHandler(_LocklessHandler):
    """web · module · api · httpx · openai (+ uvicorn.error) 의 기록 → 시스템 로그, 그리고 실행에 묶인 기록은 그 실행의
    test.log 와 SSE 로. 실행기 자신의 줄(test.log FileHandler 가 달린 로거)은 _RunLogHandler 가 맡으므로 시스템 로그에만 넣는다."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self._guard = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._guard, "busy", False):          # 이 안에서 남긴 로그가 다시 들어오면 버린다
            return
        self._guard.busy = True
        try:
            d = LC.describe(record)
            msg = d["message"]
            exc = record.exc_info[1] if record.exc_info else None
            if exc is not None:                          # 시스템 로그는 한 줄 — traceback 대신 예외 이름과 첫 줄
                first = (str(exc).splitlines() or [""])[0][:200]
                msg = f"{msg} ({type(exc).__name__}{': ' + first if first else ''})"
            _SYSLOG.append({"ts": d["ts"], "time": d["time"], "level": d["level"], "component": d["component"],
                            "logger": display_logger(d["logger"]), "run_id": d["run_id"], "set_id": d["set_id"],
                            "message": scrub_text(LC.scrub_secret(msg))})
            rid = d["run_id"]
            if rid and not LC.is_runner_logger(record.name):
                run = _RUNS.get(rid)
                if run is not None:
                    run.log_record(record, d)
        except Exception:
            pass
        finally:
            self._guard.busy = False


class _QuietRootFilter(logging.Filter):
    """수집하려고 INFO 로 내린 로거의 기록이 루트 핸들러(uvicorn stdout · systemd journal)로 새지 않게 한다.

    기록을 남긴 로거에서 위로 올라가며, 직접 레벨을 정한 로거를 먼저 만나면(예: 실행기 로거 INFO) 그 레벨이 예전에도
    거른 것이므로 그대로 통과시키고, 우리가 내린 로거를 먼저 만나면 내리기 전의 실효 레벨로 거른다.
    """

    def __init__(self, lowered: Dict[str, int]):
        super().__init__()
        self.lowered = lowered

    def _threshold(self, name: str) -> Optional[int]:
        # 기억해 두지 않는다 — 로거 레벨은 실행 중에도 바뀔 수 있다(예: caplog.at_level). 위로 몇 단계 dict 조회일 뿐이다.
        if not self.lowered:
            return None
        loggers = logging.Logger.manager.loggerDict
        n = name or ""
        while n and n != "root":
            if n in self.lowered:
                return self.lowered[n]
            node = loggers.get(n)
            if isinstance(node, logging.Logger) and node.level != logging.NOTSET:
                return None
            n = n.rpartition(".")[0]
        return None

    def filter(self, record: logging.LogRecord) -> bool:
        thr = self._threshold(record.name)
        return thr is None or record.levelno >= thr


_CAPTURE: Dict[str, Any] = {"handler": None, "lowered": {}, "filter": None, "root_handlers": []}
_CAPTURE_LOCK = threading.Lock()


def install_log_capture() -> None:
    """수집 핸들러를 단다 (여러 번 불러도 한 번만). 앱 시작 때 init_console_api 가 부른다."""
    with _CAPTURE_LOCK:
        if _CAPTURE["handler"] is not None:
            return
        h = _CaptureHandler()
        lowered: Dict[str, int] = {}
        original: Dict[str, int] = {}
        for name in CAPTURE_LOGGERS + SYSLOG_ONLY_LOGGERS:
            lg = logging.getLogger(name)
            eff = lg.getEffectiveLevel()
            if eff > logging.INFO:
                lowered[name] = eff
                original[name] = lg.level
                lg.setLevel(logging.INFO)
            lg.addHandler(h)
        f = _QuietRootFilter(lowered)
        roots = list(logging.getLogger().handlers)
        for rh in roots:
            rh.addFilter(f)
        _CAPTURE.update(handler=h, lowered=lowered, original=original, filter=f, root_handlers=roots)
    logger.info("로그 수집 설치 — 로거 %s · 시스템 로그 최근 %d건 보관 · 레벨을 INFO 로 내린 로거 %s (루트 출력은 예전 레벨 유지)",
                ", ".join(CAPTURE_LOGGERS + SYSLOG_ONLY_LOGGERS), SYSLOG_MAX,
                ", ".join(f"{k}({logging.getLevelName(v)})" for k, v in lowered.items()) or "없음",
                extra={"component": "시스템"})


def uninstall_log_capture() -> None:
    """테스트용 — 핸들러·필터를 떼고 내린 레벨을 되돌린다."""
    with _CAPTURE_LOCK:
        h = _CAPTURE["handler"]
        if h is None:
            return
        for name in CAPTURE_LOGGERS + SYSLOG_ONLY_LOGGERS:
            lg = logging.getLogger(name)
            lg.removeHandler(h)
            if name in _CAPTURE.get("original", {}):
                lg.setLevel(_CAPTURE["original"][name])
        for rh in _CAPTURE["root_handlers"]:
            rh.removeFilter(_CAPTURE["filter"])
        _CAPTURE.update(handler=None, lowered={}, original={}, filter=None, root_handlers=[])


class _OrchProxy:
    """오케스트레이터 대리 — run() 을 부른 워커 스레드를 세트에 묶어 둔다(단계 로그를 세트에 붙이려고)."""

    def __init__(self, orch, run: TestRun):
        self._orch = orch
        self._run = run

    def run(self, file_bytes, filename, **kw):
        # 실행기(BatchRunner)가 이미 세트를 묶었으면 그 세트, 아니면(가짜 실행기 등) 파일 이름으로 찾은 세트
        b = LC.current()
        sid = (b.set_id if b is not None and b.set_id else None) or self._run.file_to_set.get(filename) or str(filename)
        with LC.bind(run_id=self._run.run_id, set_id=sid):
            return self._orch.run(file_bytes, filename, **kw)

    def __getattr__(self, name):
        return getattr(self._orch, name)


# ---------------------------------------------------------------------------
# 실행 스레드
# ---------------------------------------------------------------------------
class _Rejected(Exception):
    pass


def _public_summary(s: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    s = s or {}
    return {k: s.get(k) for k in ("total", "already_done", "ok", "failed", "cost_krw", "elapsed_sec",
                                  "cancelled", "cost_exceeded")}


def _public_final(fin: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    fin = fin or {}
    out = {k: fin.get(k) for k in ("evaluated", "test", "predictions", "predictions_ok", "n_sets", "n_predictions",
                                   "n_gold_dropped", "macro_f1", "micro_f1", "passed", "per_key_f1", "excluded_keys",
                                   "warning_counts", "only_predicted", "tau", "contract_items")}
    out["error"] = scrub_text(fin.get("error")) if fin.get("error") else None
    out["warning"] = scrub_text(fin.get("warning")) if fin.get("warning") else None
    out["pass_mark"] = PASS_MARK
    return out


# ---------------------------------------------------------------------------
# Te-2 의 1~16번 — 같은 계약서를 처리한 Te-1 결과 (2610028 최종 절차서: Te-2 는 1~23번 평가)
# ---------------------------------------------------------------------------
TE1_WAIT_POLL_SEC = 3.0


def _te1_runs(exclude: Optional[str] = None) -> List["TestRun"]:
    """끝났거나 도는 중인 Te-1 일괄 실행 — 가장 최근 것이 앞."""
    with _RUNS_LOCK:
        runs = [r for r in _RUNS.values() if r.kind == "batch" and r.test == "T-001" and r.run_id != exclude]
    runs.sort(key=lambda r: (r.created_at or "", r.run_id), reverse=True)
    return runs


def _contract_items_plan(entries: list, partner_ids: Optional[set] = None) -> Dict[str, Any]:
    """불러오기 화면용 — Te-2 세트마다 1~16번을 어디서 받을지 센다 (세트 ID + 계약서 파일 이름·크기로 어림).
    실행 중에는 세트마다 계약서 sha256 으로 다시 확인하고, 맞지 않으면 계약서를 직접 처리한다."""
    from module.evaluation.v4.te1_reuse import ok_contracts, probably_same
    srcs = [(r, ok_contracts(r.out_dir / "results.jsonl")) for r in _te1_runs() if r.state not in ACTIVE_STATES]
    partner_ids = partner_ids or set()
    n = {"te1": 0, "partner": 0, "process": 0, "no_contract": 0}
    by_run: Dict[str, int] = {}
    for e in entries:
        if not e.contract:
            n["no_contract"] += 1
            continue
        if e.set_id in partner_ids:
            n["partner"] += 1
            continue
        hit = next((r for r, ok in srcs if probably_same(ok.get(e.set_id), e.contract)), None)
        if hit is None:
            n["process"] += 1
        else:
            n["te1"] += 1
            by_run[hit.run_id] = by_run.get(hit.run_id, 0) + 1
    names = {r.run_id: r.name for r, _ in srcs}
    return {**n, "runs": [{"run_id": rid, "name": names.get(rid), "n": k}
                          for rid, k in sorted(by_run.items(), key=lambda kv: -kv[1])]}


def _te1_lookup(run: "TestRun"):
    """Te-2 실행기의 te1_lookup — 함께 도는 Te-1(둘 다)이 있으면 그 결과를 먼저 보고, 아직 처리 전이면 기다린다
    (같은 계약서를 두 번 처리하지 않게). 그다음 지난 Te-1 실행을 최근 것부터."""
    from module.evaluation.v4.te1_reuse import Te1Index
    partners = [o for o in _partner_runs(run) if o.test == "T-001"]
    rest = [r for r in _te1_runs(exclude=run.run_id) if r not in partners]
    idx = Te1Index([(r.run_id, r.pred_dir) for r in partners + rest])

    def lookup(entry) -> Optional[Dict[str, Any]]:
        hit = idx.find(entry.set_id, entry.contract)
        if hit is not None:
            return hit
        waiting = [o for o in partners if entry.set_id in o.set_info]
        if waiting:
            logger.info("[%s] 함께 도는 Te-1 이 이 계약서를 처리하기를 기다립니다", entry.set_id, extra={"component": "계약서"})
        while waiting and not run.cancel_requested:
            if all(o.state not in ACTIVE_STATES for o in waiting):
                break
            time.sleep(TE1_WAIT_POLL_SEC)
            hit = idx.find(entry.set_id, entry.contract)
            if hit is not None:
                return hit
        return idx.find(entry.set_id, entry.contract) if waiting else None

    return lookup


def _execute(run: TestRun, entries: list) -> None:
    # 이 스레드(와 여기서 문맥을 복사해 간 작업 스레드)가 남기는 로그는 이 실행에 붙는다 — 세트는 실행기가 묶는다
    with LC.bind(run_id=run.run_id, set_id=None):
        _execute_bound(run, entries)


def _execute_bound(run: TestRun, entries: list) -> None:
    runner = None
    finalized = False
    try:
        run.set_state("running")
        if run.kind == "batch":
            from module.evaluation.manifest import validate_manifest
            v = validate_manifest(entries, test=run.test)
            if not v.get("ok"):
                raise _Rejected("목록 검증 실패 — 실행하지 않습니다: " + " / ".join(v.get("problems") or ["(사유 없음)"]))
        logger.info("실행 준비 — 파이프라인 구성 중 (%s · %d세트)", run.test, len(entries), extra={"component": "시스템"})
        from module.evaluation.batch_runner import RunConfig
        t0 = time.perf_counter()
        orch = _ORCH_FACTORY(run.out_dir)
        logger.info("파이프라인 구성 완료 — %.1f초", time.perf_counter() - t0, extra={"component": "시스템"})
        cfg = RunConfig(**run.cfg_kwargs)
        runner = _RUNNER_FACTORY(_OrchProxy(orch, run), cfg, run.out_dir, run.on_runner_event)
        if run.kind == "batch" and run.test == "T-002" and getattr(cfg, "contract_items", "te1") == "te1":
            runner.te1_lookup = _te1_lookup(run)
        run.runner = runner
        run.attach_runner_log(runner)
        if run.cancel_requested:
            runner.cancel()
        summary = runner.run(entries, {}) or {}
        run.summary = _public_summary(summary)
        processed = int(summary.get("ok") or 0) + int(summary.get("failed") or 0)
        rcfg = getattr(runner, "cfg", None)
        if processed == 0 and rcfg is not None:
            rcfg.gold = None                            # 예측이 하나도 없으면 채점하지 않는다
        if getattr(rcfg, "gold", None):
            run.set_state("scoring")
        fin = runner.finalize_eval23()
        finalized = True
        run.final = _public_final(fin)
        run.push({"type": "finalized", **{k: run.final.get(k) for k in
                                          ("evaluated", "macro_f1", "micro_f1", "passed", "error", "warning")}})
        if summary.get("cost_exceeded") or run.cost_exceeded:
            run.error = "비용 상한에 도달해 중단했습니다 (처리한 세트의 결과는 보존)"
        elif run.final.get("error"):
            run.error = f"채점 실패 — {run.final['error']}"
        run.finish("cancelled" if summary.get("cancelled") else "done")
    except _Rejected as e:
        run.error = str(e)[:600]
        run.push({"type": "error", "error": run.error})
        run.finish("error")
    except Exception as e:
        logger.error("시험 실행 실패 %s: %s", run.run_id, e, exc_info=True, extra={"component": "시스템"})
        run.error = scrub_text(f"실행 오류 — {type(e).__name__}: {e}")[:600]
        run.push({"type": "error", "error": run.error})
        run.finish("error")
    finally:
        if runner is not None:
            run.detach_runner_log(runner)
            if not finalized and hasattr(runner, "close_log"):
                try:
                    runner.close_log()
                except Exception:
                    pass
        run.runner = None
        run.threads.clear()


def _set_info(entries: list) -> Dict[str, Dict[str, Any]]:
    out = {}
    for e in entries:
        info = {"cell": _entry_cell(e), "media": e.media}
        title = (e.extra or {}).get("title")
        if title:
            info["title"] = title
        out[e.set_id] = info
    return out


def _start(run: TestRun, entries: list) -> None:
    run.set_info = _set_info(entries)
    threading.Thread(target=_execute, args=(run, entries), daemon=True, name=f"test-run-{run.run_id}").start()


def _new_run_id(kind: str) -> str:
    for _ in range(50):
        rid = f"{datetime.now():%Y%m%d_%H%M%S}_{kind}_{secrets.token_hex(2)}"
        if rid not in _RUNS and not (_ROOT / rid).exists():
            return rid
    raise ConsoleError(500, "internal", "실행 번호를 만들 수 없습니다")


def _active_runs(kind: str) -> List["TestRun"]:
    """진행 중인 실행 (등록 순서) — 둘 다(동시 실행)면 일괄 실행이 두 건이다."""
    with _RUNS_LOCK:
        return [r for r in _RUNS.values() if r.kind == kind and r.state in ACTIVE_STATES]


def _active_run(kind: str) -> Optional["TestRun"]:
    runs = _active_runs(kind)
    return runs[0] if runs else None


def _busy_error(active: "TestRun") -> ConsoleError:
    what = "일괄 시험" if active.kind == "batch" else "단건 시험"
    return ConsoleError(409, "busy", f"이미 진행 중인 {what}이 있습니다 ({active.run_id})", active=active.run_id)


def _register(*runs: TestRun) -> None:
    """실행을 등록한다 — 같은 종류의 실행이 진행 중이면 409. 여러 건(둘 다)은 한꺼번에 등록한다."""
    with _RUNS_LOCK:
        active = _active_run(runs[0].kind)
        if active is not None:
            raise _busy_error(active)
        for run in runs:
            _RUNS[run.run_id] = run
    for run in runs:
        run.save()


def _get_run(run_id: str) -> TestRun:
    with _RUNS_LOCK:
        run = _RUNS.get(run_id)
    if run is None:
        raise _not_found("실행을 찾을 수 없습니다")
    return run


# ---------------------------------------------------------------------------
# 복원 (서버 재시작)
# ---------------------------------------------------------------------------
def _progress_from_disk(run: TestRun) -> None:
    per: Dict[str, Dict[str, Any]] = {}
    for r in _iter_jsonl(run.out_dir / "results.jsonl"):
        per[str(r.get("set_id"))] = {"ok": bool(r.get("ok")), "cost": float(r.get("cost_krw") or 0.0)}
    if not per:
        return
    run.progress = {"i": len(per), "total": max(run.n_sets, len(per)),
                    "failed": sum(1 for v in per.values() if not v["ok"]),
                    "cost_krw_total": round(sum(v["cost"] for v in per.values()), 2),
                    "last_set": run.progress.get("last_set")}


def _replay_log(run: TestRun) -> None:
    path = run.out_dir / "test.log"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines[-LOG_REPLAY_LINES:]:
        if line.strip():
            run.push_line(line, replayed=True)
    run.push({"type": "state", "state": run.state, "error": run.error, "replayed": True})


def _recover_runs() -> None:
    if not _ROOT.is_dir():
        return
    n = 0
    for d in sorted(_ROOT.iterdir()):
        if not d.is_dir() or not (d / "run.json").is_file():
            continue
        run = TestRun.load(d)
        if run is None:
            continue
        if run.state in ACTIVE_STATES:
            run.state = "interrupted"
            run.error = run.error or "서버 재시작으로 중단됨 (처리한 세트의 결과는 보존)"
            run.finished_at = run.finished_at or _iso((d / "run.json").stat().st_mtime)
            _progress_from_disk(run)
            run.save()
        _replay_log(run)
        with _RUNS_LOCK:
            _RUNS[run.run_id] = run
        n += 1
    if n:
        logger.info("시험 실행 %d건 복원 (%s)", n, _ROOT)


# ---------------------------------------------------------------------------
# 시험 목록 · 선택 · 추정
# ---------------------------------------------------------------------------
_ENTRY_CACHE: Dict[Tuple[str, str], Tuple[Any, list]] = {}
_ENTRY_LOCK = threading.Lock()


def _csv_entries(csv: Path, test: str) -> list:
    sig = _sig(Path(csv))
    if sig is None:
        return []
    key = (str(csv), test)
    with _ENTRY_LOCK:
        hit = _ENTRY_CACHE.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    from module.evaluation.manifest_v4 import entries_from_ids_csv
    entries = entries_from_ids_csv(csv, test, repo_root=_REPO)
    with _ENTRY_LOCK:
        _ENTRY_CACHE[key] = (sig, entries)
    return entries


def _entry_cell(e) -> str:
    return (e.extra or {}).get("cell") or f"{e.media}/{e.license_bucket}"


def _source(source_id: Optional[str]) -> Dict[str, Any]:
    for s in _SOURCES:
        if s["id"] == source_id:
            return s
    raise ConsoleError(400, "unknown_source", "알 수 없는 시험 목록입니다")


def _source_view(src: Dict[str, Any]) -> Dict[str, Any]:
    enabled = src.get("enabled", True) is not False
    out = {"id": src["id"], "label": src.get("label"), "test": src.get("test"), "n": None, "cells": {},
           "gold": False, "enabled": enabled, "reason": src.get("reason")}
    if src.get("by_ids"):
        out["gold"] = any(p and Path(p).is_file() for p in (src.get("gold_by_test") or {}).values())
        out["tests"] = list(TESTS)
        return out
    try:
        entries = []
        for csv in src.get("csv") or []:
            entries.extend(_csv_entries(Path(csv), src["test"]))
        cells: Dict[str, int] = {}
        for e in entries:
            c = _entry_cell(e)
            cells[c] = cells.get(c, 0) + 1
        out["n"], out["cells"] = len(entries), dict(sorted(cells.items()))
    except Exception as e:
        logger.warning("시험 목록 %s 를 읽지 못했습니다: %s", src["id"], e)
        out["enabled"], out["reason"] = False, out["reason"] or "목록 파일을 읽을 수 없습니다"
    g = src.get("gold")
    out["gold"] = bool(g and Path(g).is_file())
    return out


def _select(test: Optional[str], source_id: Optional[str], ids: List[str], cells: List[str], n: Optional[int],
            sample: str, seed: Optional[int]):
    src = _source(source_id)
    if src.get("enabled", True) is False:
        raise ConsoleError(400, "source_disabled", f"지금은 쓸 수 없는 목록입니다 — {src.get('reason') or '사유 없음'}")
    stest = src.get("test")
    if stest:
        if test and test != stest:
            raise ConsoleError(400, "test_mismatch", f"이 목록은 {stest} 전용입니다")
        test = stest
    if test not in TESTS:
        raise ConsoleError(400, "bad_test", "시험 구분(Te-1 또는 Te-2)을 고르세요")
    missing: List[str] = []
    if src.get("by_ids"):
        bad = [i for i in ids if not _SET_ID_RX.match(i)]
        if bad:
            raise ConsoleError(400, "bad_ids", f"ID 형식이 올바르지 않습니다: {', '.join(bad[:5])}")
        if not ids:
            raise ConsoleError(400, "no_ids", "ID 목록을 입력하세요")
        pool: Dict[str, Any] = {}
        for csv in (src.get("csv_by_test") or {}).get(test) or []:
            for e in _csv_entries(Path(csv), test):
                pool.setdefault(e.set_id, e)
        entries = [pool[i] for i in ids if i in pool]
        missing = [i for i in ids if i not in pool]
        gold = (src.get("gold_by_test") or {}).get(test)
    else:
        entries = []
        for csv in src.get("csv") or []:
            entries.extend(_csv_entries(Path(csv), test))
        gold = src.get("gold")
    if cells:
        cs = set(cells)
        entries = [e for e in entries if _entry_cell(e) in cs]
    if n:
        if sample == "random" and n < len(entries):
            rng = random.Random(seed)
            idx = sorted(rng.sample(range(len(entries)), n))
            entries = [entries[i] for i in idx]
        else:
            entries = entries[:n]
    gold_path = Path(gold) if gold and Path(gold).is_file() else None
    return test, entries, missing, gold_path, src


def _vlm_unit_costs() -> Tuple[float, float, float]:
    try:
        from module.evaluation import batch_runner as BR
        return float(BR.COST_VLM_IMAGE), float(BR.COST_VLM_VIDEO), float(BR.COST_CONSOLIDATION_IMG)
    except Exception:
        return 1.34, 1.77, 35.4


def _interp(table, x: float) -> float:
    """(x, y) 표에서 직선 보간 — 표 밖은 끝 값."""
    if x <= table[0][0]:
        return table[0][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def _estimate(test: str, entries: list, workers: Optional[int] = None,
              consolidate_work: bool = False, contract_sets: int = 0) -> Dict[str, Any]:
    """예상 소요·비용. contract_sets = Te-2 가 계약서를 직접 처리할 세트 수 (Te-1 결과가 없는 세트)."""
    by_cell: Dict[str, int] = {}
    for e in entries:
        c = _entry_cell(e)
        by_cell[c] = by_cell.get(c, 0) + 1
    n = len(entries)
    w = int(_clamp(workers or DEFAULT_WORKERS[test], 1, MAX_WORKERS))
    if test == "T-001":
        think = os.getenv("CONSOLIDATION_THINKING", "off").strip().lower() not in ("off", "0", "false", "no")
        krw = n * (EST_T001_KRW_THINKING if think else EST_T001_KRW)
        lat, tok = (T001_LATENCY_THINKING, T001_TOKENS_PER_SET_THINKING) if think else (T001_LATENCY, T001_TOKENS_PER_SET)
        # 작업자 수만큼씩 묶어 도는 묶음 수 × 그 작업자 수에서의 1건 소요 — 토큰 한도가 처리량의 상한
        sec = max(math.ceil(n / w) * _interp(lat, w), n * tok / DASHSCOPE_TPM * 60) if n else 0
    else:
        img, vid, con = _vlm_unit_costs()
        krw = sum(vid if e.media == "video" else img for e in entries)
        sec = n * EST_T002_SEC * EST_T002_REF_WORKERS / w
        sec = max(sec, sum(EST_T002_MB.get(e.media, EST_T002_MB["image"]) for e in entries) / EST_DISK_MBPS)
        if consolidate_work:
            krw += n * con
            sec += n * EST_WORK_CONSOLIDATION_SEC / w
        # 소량 실행의 하한: 작업자 수만큼씩 묶어 도는 세트 묶음 수 × 세트 하나의 소요
        per_set = EST_T002_SET_SEC + (EST_WORK_CONSOLIDATION_SEC if consolidate_work else 0)
        sec = max(sec, math.ceil(n / w) * per_set) if n else 0
        if contract_sets:
            think = os.getenv("CONSOLIDATION_THINKING", "off").strip().lower() not in ("off", "0", "false", "no")
            krw += contract_sets * (EST_T001_KRW_THINKING if think else EST_T001_KRW)
            sec += math.ceil(contract_sets / w) * _interp(T001_LATENCY_THINKING if think else T001_LATENCY, w)
    return {"n_sets": n, "by_cell": dict(sorted(by_cell.items())), "est_sec": int(round(sec)),
            "est_krw": round(krw, 2), "workers": w}


# ---------------------------------------------------------------------------
# gold 색인 (줄 위치만 기억 — 4,000줄 파일을 통째로 들고 있지 않는다)
# ---------------------------------------------------------------------------
_GOLD_INDEX: Dict[str, Tuple[Any, Dict[str, int]]] = {}
_GOLD_LOCK = threading.Lock()
_GOLD_SID_RX = re.compile(rb'"set_id"\s*:\s*"([^"]+)"')


def _gold_offsets(path: Path) -> Dict[str, int]:
    sig = _sig(path)
    if sig is None:
        return {}
    key = str(path)
    with _GOLD_LOCK:
        hit = _GOLD_INDEX.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    offs: Dict[str, int] = {}
    with path.open("rb") as f:
        pos = 0
        for line in f:
            m = _GOLD_SID_RX.search(line[:300])
            if m:
                offs[m.group(1).decode("utf-8", "replace")] = pos      # 같은 id 가 여러 줄이면 마지막 줄 (평가기와 같다)
            pos += len(line)
    with _GOLD_LOCK:
        _GOLD_INDEX[key] = (sig, offs)
    return offs


def gold_record(path: Optional[os.PathLike], set_id: str) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    off = _gold_offsets(p).get(str(set_id))
    if off is None:
        return None
    try:
        with p.open("rb") as f:
            f.seek(off)
            rec = json.loads(f.readline().decode("utf-8"))
        return rec if isinstance(rec, dict) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


_ID_CACHE: Dict[Tuple[str, str], Tuple[Any, frozenset]] = {}


def _csv_ids(csv: Path, test: str) -> frozenset:
    """목록 csv 의 work_id (계약서 유무와 상관없이 — 계약서가 아직 없던 세트에 계약서를 올리는 경우도 알아본다)."""
    sig = _sig(Path(csv))
    if sig is None:
        return frozenset()
    key = (str(csv), test)
    hit = _ID_CACHE.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    from module.evaluation.manifest_v4 import read_ids_csv
    ids = frozenset(str(r.get("work_id") or "").strip() for r in read_ids_csv(csv)
                    if (r.get("test") or test).strip() == test and str(r.get("work_id") or "").strip())
    _ID_CACHE[key] = (sig, ids)
    return ids


def _known_ids(test: str) -> set:
    """시험 구분에 맞는 목록(t001_new5·t001 / t002·pilot200, ID 입력 목록)의 set_id."""
    out: set = set()
    for src in _SOURCES:
        csvs = (src.get("csv_by_test") or {}).get(test) if src.get("by_ids") else (
            src.get("csv") if src.get("test") == test else None)
        for csv in csvs or []:
            try:
                out |= _csv_ids(Path(csv), test)
            except Exception as e:
                logger.warning("목록 %s 의 ID 를 읽지 못했습니다: %s", csv, e)
    return out


def _single_set_id(filename: str, gold_path: Optional[Path], known: Optional[set] = None) -> Tuple[str, bool]:
    """파일 이름의 5~8자리 숫자 묶음 가운데 gold·시험 목록에 있는 첫 번째 → set_id, 없으면 마지막 묶음, 그것도 없으면 'upload'.

    신규 양식 계약서 이름은 '00002_13387752 운수 자판기 (1).pdf' 처럼 생성 순번이 앞에 붙는다 — 앞의 번호는 set_id 가 아니다.
    두 번째 값은 그 set_id 가 gold 에 있는지(판정 가능 여부).
    """
    groups = re.findall(r"(?<!\d)(\d{5,8})(?!\d)", Path(filename).stem)
    if not groups:
        return "upload", False
    offs = _gold_offsets(gold_path) if gold_path else {}
    known = known or set()
    for g in groups:
        if g in offs or g in known:
            return g, g in offs
    return groups[-1], groups[-1] in offs


# ---------------------------------------------------------------------------
# 판정 (per_set.jsonl 한 줄 → 화면 판정)
# ---------------------------------------------------------------------------
def verdict_of(row: Optional[Dict[str, Any]]) -> Optional[str]:
    """NA · TP · PARTIAL · WRONG · FP · FN · EMPTY_OK. 행이 없으면(비채점 키·gold 없음) None."""
    if not row:
        return None
    state = row.get("state")
    if state == "na":
        return "NA"
    tp, fp, fn = int(row.get("tp") or 0), int(row.get("fp") or 0), int(row.get("fn") or 0)
    if state == "empty":
        return "EMPTY_OK" if fp == 0 else "FP"
    if fp == 0 and fn == 0:
        return "TP" if (tp > 0 or row.get("matched")) else None
    if row.get("key") in _TAG_KEYS and tp > 0:
        return "PARTIAL"
    if fp > 0 and fn > 0:
        return "WRONG"
    if fp > 0:
        return "FP"
    return "FN"


# ---------------------------------------------------------------------------
# 세트 색인 (실행 폴더의 jsonl 4개 → 목록·격자)
# ---------------------------------------------------------------------------
def _run_index(run: TestRun) -> Dict[str, Any]:
    paths = {"manifest": run.dir / "manifest.jsonl", "results": run.out_dir / "results.jsonl",
             "preds": run.out_dir / "test_predictions.jsonl", "per_set": run.out_dir / "per_set.jsonl"}
    sig = tuple(_sig(p) for p in paths.values())
    hit = run._index
    if hit and hit[0] == sig:
        return hit[1]
    order: List[str] = []
    sets: Dict[str, Dict[str, Any]] = {}
    for d in _iter_jsonl(paths["manifest"]):
        sid = str(d.get("set_id") or "")
        if not sid or sid in sets:
            continue
        extra = d.get("extra") if isinstance(d.get("extra"), dict) else {}
        test = d.get("test") or run.test
        src = d.get("contract") if test == "T-001" else d.get("work")
        media = d.get("media") or "image"
        bucket = d.get("license_bucket") or ""
        sets[sid] = {"set_id": sid, "cell": extra.get("cell") or (f"{media}/{bucket}" if bucket else media),
                     "media": media, "title": extra.get("title") or None,
                     "file_name": os.path.basename(src) if src else None}
        order.append(sid)
    results: Dict[str, Dict[str, Any]] = {}
    attempts: Dict[str, int] = {}
    for r in _iter_jsonl(paths["results"]):           # 다시 실행하면 같은 세트의 줄이 늘어난다 — 마지막 줄이 현재 상태
        sid = str(r.get("set_id"))
        attempts[sid] = attempts.get(sid, 0) + 1
        results[sid] = {"ok": bool(r.get("ok")), "error": scrub_text(r.get("error")),
                        "elapsed_sec": r.get("elapsed_sec"), "cost_krw": r.get("cost_krw"), "attempts": attempts[sid]}
    preds: Dict[str, Dict[str, Any]] = {}
    for r in _iter_jsonl(paths["preds"]):
        sid = str(r.get("set_id") or (r.get("_meta") or {}).get("set_id") or "")
        if sid:
            preds[sid] = {k: v for k, v in r.items() if k != "_meta"}
    verdicts: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in _iter_jsonl(paths["per_set"]):
        verdicts.setdefault(str(r.get("set_id")), {})[str(r.get("key"))] = r
    for sid in list(results) + list(preds):
        if sid not in sets:                     # 매니페스트에 없는 결과(예외적) — 목록 끝에 붙인다
            sets[sid] = {"set_id": sid, "cell": None, "media": None, "title": None, "file_name": None}
            order.append(sid)
    data = {"order": order, "sets": sets, "results": results, "preds": preds, "verdicts": verdicts}
    run._index = (sig, data)
    return data


def _set_items(run: TestRun) -> List[Dict[str, Any]]:
    from module.evaluation.v4.keys23 import KEY_NAMES
    idx = _run_index(run)
    items = []
    running = set(run.running_sets)
    for sid in idx["order"]:
        base = idx["sets"][sid]
        res = idx["results"].get(sid)
        pred = idx["preds"].get(sid) or {}
        vrows = idx["verdicts"].get(sid) or {}
        if res is None:
            state = "running" if sid in running else "pending"
        else:
            state = "ok" if res["ok"] else "failed"
        keys = {}
        for k in KEY_NAMES:
            row = vrows.get(k)
            v = verdict_of(row)
            ent: Dict[str, Any] = {"verdict": v, "value_short": _value_short(pred.get(k))}
            if v == "PARTIAL" or (row and k in _TAG_KEYS and (row.get("detail") or {}).get("set_f1") is not None):
                ent["set_f1"] = (row.get("detail") or {}).get("set_f1")
            keys[k] = ent
        items.append({
            "set_id": sid, "cell": base.get("cell"), "media": base.get("media"),
            "title": pred.get("work_title") or base.get("title"),
            "file_name": pred.get("file_name") or base.get("file_name"),
            "state": state, "ok": None if res is None else res["ok"],
            "error": None if res is None else res.get("error"),
            "elapsed_sec": None if res is None else res.get("elapsed_sec"),
            "cost_krw": None if res is None else res.get("cost_krw"),
            "attempts": 0 if res is None else res.get("attempts", 1),
            "keys": keys,
        })
    return items


# ---------------------------------------------------------------------------
# 세트 상세 — 레코드 · 23키 · 근거 · 미리보기
# ---------------------------------------------------------------------------
def _record_path(run: TestRun, set_id: str) -> Path:
    if not _SET_ID_RX.match(str(set_id)) or set(str(set_id)) <= {"."}:
        raise _not_found("세트를 찾을 수 없습니다")
    p = run.out_dir / "test_predictions" / f"{_safe_set_filename(set_id)}.json"
    if not _inside(p, run.dir):
        raise _not_found("세트를 찾을 수 없습니다")
    return p


def _load_record(run: TestRun, set_id: str) -> Dict[str, Any]:
    rec = _read_json(_record_path(run, set_id))
    if not isinstance(rec, dict):
        raise _not_found("세트 결과가 아직 없습니다")
    return rec


def _leg_dir(run: TestRun, rid: Optional[str]) -> Optional[Path]:
    if not rid or not _RID_RX.match(str(rid)) or str(rid) in (".", ".."):
        return None
    d = run.out_dir / "runs" / "results" / str(rid)
    if not d.is_dir() or not _inside(d, run.dir):
        return None
    return d


def _record_file(rec: Dict[str, Any], which: str) -> Optional[Path]:
    """레코드의 _meta.files 에 적힌 파일만 (경로를 요청에서 받지 않는다)."""
    p = (((rec.get("_meta") or {}).get("files")) or {}).get(which)
    if not p or not isinstance(p, str):
        return None
    try:
        return Path(p) if os.path.isfile(p) else None
    except (OSError, ValueError):
        return None


# 결과 폴더 source/ 에 남긴 사본 (web/pipeline.py _keep_source·_keep_work_preview, batch_runner._keep_contract_copy)
_KEPT = {"contract": ("document.pdf", "contract.pdf"), "work": ("work_preview.jpg",)}


def _source_file(run: TestRun, rec: Dict[str, Any], which: str) -> Optional[Path]:
    """미리보기·위치 표시용 원본 — 원본 파일이 있으면 그것, 없으면(옮기거나 지움) 실행 폴더에 남긴 사본.
    사본은 레코드의 request_ids 가 가리키는 leg 폴더 안에서만 찾는다 (경로를 요청에서 받지 않는다)."""
    p = _record_file(rec, which)
    if p is not None:
        return p
    meta = rec.get("_meta") or {}
    if not (meta.get("files") or {}).get(which):
        return None
    rids = meta.get("request_ids") or {}
    for leg in ("contract", "work"):
        d = _leg_dir(run, rids.get(leg))
        if d is None:
            continue
        for name in _KEPT.get(which, ()):
            c = d / "source" / name
            if c.is_file() and _inside(c, run.dir):
                return c
    return None


def _frames(leg: Optional[Path]) -> List[Dict[str, Any]]:
    if leg is None:
        return []
    kf = leg / "keyframes"
    if not kf.is_dir():
        return []
    out = []
    for name in sorted(os.listdir(kf)):
        if not _FRAME_RX.match(name) or not (kf / name).is_file():
            continue
        m = _FRAME_T_RX.search(name)
        out.append({"name": name, "t": float(m.group(1)) if m else None})
    return out


_PAGE_CACHE: Dict[Tuple[str, Any], int] = {}


def _page_count(path: Path) -> int:
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        return 1
    if ext != ".pdf":
        return 0
    key = (str(path), _sig(path))
    if key in _PAGE_CACHE:
        return _PAGE_CACHE[key]
    try:
        import fitz
        with fitz.open(str(path)) as doc:
            n = int(doc.page_count)
    except Exception:
        n = 0
    _PAGE_CACHE[key] = n
    return n


def _previews(run: TestRun, set_id: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    meta = rec.get("_meta") or {}
    test = rec.get("test") or meta.get("test") or run.test
    base = f"{API}/runs/{run.run_id}/sets/{quote(str(set_id), safe='')}/preview"
    work: Dict[str, Any] = {"kind": None, "image_url": None, "frames": []}
    files = meta.get("files") or {}
    if test == "T-002" and files.get("work"):
        kind = _work_kind(files.get("work"), meta.get("media"))
        work["kind"] = kind
        wdir = _leg_dir(run, (meta.get("request_ids") or {}).get("work"))
        if kind == "video":
            work["frames"] = [{"url": f"{base}/frames/{quote(f['name'])}", "t": f["t"], "name": f["name"]}
                              for f in _frames(wdir)]
        elif kind == "pptx":
            if wdir is not None and (wdir / "pptx_slide1.png").is_file():
                work["image_url"] = f"{base}/work.jpg"
        elif kind == "image" and _source_file(run, rec, "work") is not None:
            work["image_url"] = f"{base}/work.jpg"
    contract: Dict[str, Any] = {"pages": []}
    cpath = _source_file(run, rec, "contract")
    if cpath is not None:
        n = min(_page_count(cpath), MAX_CONTRACT_PAGES)
        contract["pages"] = [{"page": i, "url": f"{base}/contract/{i}.png"} for i in range(1, n + 1)]
    return {"work": work, "contract": contract}


# ---------------------------------------------------------------------------
# 통합 검증(중재자) 상세 — 필드 이름 · 23키 대응
# ---------------------------------------------------------------------------
# 통합 스키마 필드 → 23키 (판정 행의 "key23"). 23키 하나로 곧장 이어지는 필드만 — 나머지는 None.
FIELD_KEY23: Dict[str, str] = {
    "work_title": "work_title", "file_name": "file_name", "work_type": "work_type",
    "author": "creator", "creator": "creator",
    "copyright_holder": "copyright_holder", "licensee_party": "copyright_holder", "licensee": "copyright_holder",
    "user": "copyright_holder",
    **{flat: k for k, flat in RIGHTS_MAP.items()}, **{k: k for k in RIGHTS_MAP},
    "other_rights": "other_rights",
    "license_start_date": "license_start_date", "license_end_date": "license_end_date",
    "valid_period": "contract_validity_period", "contract_duration": "contract_validity_period",
    "contract_validity_period": "contract_validity_period",
    "keyword": "keywords", "keywords": "keywords", "hashtags": "hashtags", "hashtag": "hashtags",
    "digital_format": "file_format", "file_format": "file_format", "file_size": "file_size",
    "resolution": "resolution", "description": "description", "file_hash": "file_hash",
}
# 23키 여러 개에 걸친(또는 다른 키의 재료가 되는) 필드 — 판정 행의 "key23_related", 그리고 그 키들의 근거에 붙는다
FIELD_KEY23_RELATED: Dict[str, Tuple[str, ...]] = {
    "economic_rights": tuple(RIGHTS_MAP), "granted_rights": tuple(RIGHTS_MAP), "rights_granted": tuple(RIGHTS_MAP),
    "license_start_date": ("contract_validity_period",), "license_end_date": ("contract_validity_period",),
}
_SUBFIELD_KO = {"name": "이름", "address": "주소", "phone": "전화번호", "role": "역할", "registration_no": "등록번호",
                "email": "이메일", "title": "제목", "type": "유형"}
_FIELD_LABELS: Dict[str, str] = {}
_FIELD_LABEL_LOCK = threading.Lock()


def _field_labels() -> Dict[str, str]:
    """통합 스키마 설명의 앞부분(괄호·대시 앞) → 필드 한국어 이름. 스키마 모듈만 파일 경로로 올린다(패키지 import 는 무겁다)."""
    with _FIELD_LABEL_LOCK:
        if _FIELD_LABELS:
            return _FIELD_LABELS
        mod = None
        import sys
        for name in ("module.llm_extraction.schemas.document_schemas", "api.module.llm_extraction.schemas.document_schemas"):
            if name in sys.modules:
                mod = sys.modules[name]
                break
        try:
            if mod is None:
                import importlib.util
                path = _HERE.parent / "module" / "llm_extraction" / "schemas" / "document_schemas.py"
                spec = importlib.util.spec_from_file_location("_console_document_schemas", str(path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)        # type: ignore[union-attr]
            props = mod.DocumentSchemas.get_unified_schema().get("properties") or {}
            for k, v in props.items():
                desc = str((v or {}).get("description") or "").strip()
                label = re.split(r"\s*[(（—]|\s+-\s+|\.\s", desc, maxsplit=1)[0].strip() if desc else ""
                if label:
                    _FIELD_LABELS[k] = label
        except Exception as e:
            logger.warning("통합 스키마 필드 이름을 읽지 못했습니다: %s", e)
        return _FIELD_LABELS


def field_label_ko(field: Any) -> Optional[str]:
    """'author' → '저작자', 'parties[].address' → '당사자들의 정보 · 주소'. 모르면 None."""
    f = str(field or "")
    base = re.split(r"[\[.]", f, maxsplit=1)[0]
    lab = _field_labels().get(f) or _field_labels().get(base)
    if lab is None:
        return None
    if base != f:
        tail = f[len(base):].lstrip("[].")
        tail = _SUBFIELD_KO.get(tail, tail)
        return f"{lab} · {tail}" if tail else lab
    return lab


def field_key23(field: Any) -> Optional[str]:
    return FIELD_KEY23.get(str(field or ""))


def field_key23_related(field: Any) -> List[str]:
    f = str(field or "")
    primary = FIELD_KEY23.get(f)
    return [k for k in FIELD_KEY23_RELATED.get(f, ()) if k != primary]


def _decisions_by_key(decisions: Any) -> Dict[str, List[Dict[str, Any]]]:
    """판정 행 → {23키: [행…]} (곧장 이어지는 키 먼저, 걸친 키는 뒤에)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not isinstance(decisions, list):
        return out
    for d in decisions:
        if not isinstance(d, dict):
            continue
        k = field_key23(d.get("field"))
        if k:
            out.setdefault(k, []).append(d)
    for d in decisions:
        if isinstance(d, dict):
            for k in field_key23_related(d.get("field")):
                out.setdefault(k, []).append(d)
    return out


_LEG_KO = {"contract": "계약서", "work": "저작물"}
_DECISIONS = ("AGREED", "CONFLICT", "LLM_ONLY", "NER_ONLY", "MISSING")
# 근거 탭에 쓰는 판정 이름 — 통합 검증 탭(DEC)과 같은 한국어 이름, 원래 코드는 괄호로 남깁니다
_DECISION_KO = {"AGREED": "일치", "CONFLICT": "충돌", "LLM_ONLY": "LLM 전용", "NER_ONLY": "NER 전용", "MISSING": "누락"}


def _decision_text(d: Dict[str, Any]) -> str:
    code = str(d.get("decision") or "")
    conf = d.get("confidence")
    name = f"{_DECISION_KO[code]} ({code})" if code in _DECISION_KO else code
    return name + (f" · 신뢰도 {conf}" if conf is not None else "")


MAX_NER_ENTITIES = 500


def _consolidation_leg(leg: str, con: Dict[str, Any], resp: Dict[str, Any]) -> Dict[str, Any]:
    """consolidated_metadata.json (+ llm_metadata.json) → A6 의 <leg>."""
    vr = con.get("validation_report") if isinstance(con.get("validation_report"), dict) else {}
    summ = vr.get("summary") if isinstance(vr.get("summary"), dict) else {}
    summary = {k: summ.get(k, vr.get(k)) for k in ("total_fields", "agreed_fields", "conflicted_fields",
                                                   "llm_only_fields", "ner_only_fields", "missing_fields")}
    summary["overall_confidence"] = summ.get("overall_confidence", vr.get("confidence_score"))
    decisions = []
    counts = {k: 0 for k in _DECISIONS}
    for d in vr.get("decisions") or []:
        if not isinstance(d, dict):
            continue
        dec = str(d.get("decision") or "")
        if dec in counts:
            counts[dec] += 1
        decisions.append({
            "field": d.get("field"), "label_ko": field_label_ko(d.get("field")), "key23": field_key23(d.get("field")),
            "key23_related": field_key23_related(d.get("field")),
            "llm_value": scrub_json(d.get("llm_value")), "ner_value": scrub_json(d.get("ner_value")),
            "final_value": scrub_json(d.get("final_value")), "decision": d.get("decision"),
            "confidence": d.get("confidence"), "reasoning": scrub_text(d.get("reasoning")),
        })
    ents = []
    raw_ents = con.get("ner_entities") if isinstance(con.get("ner_entities"), list) else []
    for e in raw_ents[:MAX_NER_ENTITIES]:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            ents.append({"text": scrub_json(e[0]), "type": e[1]})
        elif isinstance(e, dict):
            ents.append({"text": scrub_json(e.get("text") or e.get("entity")), "type": e.get("type") or e.get("label")})
    ner_ran = "ner_mode" in resp or bool(resp.get("ner_model_key"))
    ner = {
        "mode": resp.get("ner_mode"),
        "model": resp.get("ner_model") or resp.get("ner_model_key"),
        "loaded": (bool(resp.get("ner_model_loaded")) if "ner_model_loaded" in resp else None) if ner_ran else False,
        "load_error": scrub_text(LC.scrub_secret(resp.get("ner_load_error"))) if resp.get("ner_load_error") else None,
        "entity_count": resp.get("ner_entity_count", resp.get("entity_count", len(raw_ents))),
        "entities": ents,
        "entities_truncated": len(raw_ents) > MAX_NER_ENTITIES,
    }
    secs = con.get("consolidation_seconds")
    if secs is None:
        secs = (resp.get("stage_seconds") or {}).get("consolidation") if isinstance(resp.get("stage_seconds"), dict) else None
    return {
        "leg": leg, "leg_ko": _LEG_KO.get(leg, leg),
        "model_used": con.get("model_used") or resp.get("consolidation_model_used"),
        "requested_model": con.get("requested_model") or resp.get("consolidation_model"),
        "fallback_used": bool(con.get("fallback_used")),
        "degraded": bool(con.get("consolidation_degraded")),
        "degraded_reason": scrub_text(con.get("degraded_reason")) if con.get("degraded_reason") else None,
        "status": con.get("status"),
        "seconds": secs,
        "summary": summary,
        "decision_counts": counts,
        "decisions": decisions,
        "validation_errors": [scrub_text(str(x)) for x in (vr.get("validation_errors") or [])],
        "fields_before": con.get("consolidation_fields_before"),
        "fields_after": con.get("consolidation_fields_after"),
        "lost_fields": list(con.get("consolidation_lost_fields") or []),
        "ner": ner,
    }


def _no_consolidation_reason(leg: str, resp: Dict[str, Any], run: "TestRun") -> str:
    if not resp:
        return f"{_LEG_KO.get(leg, leg)} 결과(llm_metadata.json)가 없습니다"
    wc = resp.get("work_consolidation") if isinstance(resp.get("work_consolidation"), dict) else {}
    if leg == "work" and (wc.get("applied") is False or resp.get("consolidate_work") is False
                          or (run.cfg_kwargs or {}).get("consolidate_work") is False):
        return "저작물 통합 검증을 끈 실행입니다 (consolidate_work=False) — VLM 결과와 파일 측정값을 그대로 썼습니다"
    if resp.get("consolidation_error"):
        return f"통합 검증 실패 — {scrub_text(LC.scrub_secret(str(resp.get('consolidation_error'))))}"
    if resp.get("consolidate") is False:
        if leg == "work" and not resp.get("success"):
            return "저작물 추출이 실패해 통합 검증을 하지 않았습니다"
        return "통합 검증을 끈 실행입니다"
    return "통합 검증 결과 파일(consolidated_metadata.json)이 없습니다"


def _consolidation_payload(run: "TestRun", rec: Dict[str, Any], test: Optional[str]) -> Dict[str, Any]:
    """A6 — {"available", "reason"?, "legs": [<leg>]} (통합 검증을 돌린 leg 마다 하나)."""
    meta = rec.get("_meta") or {}
    rids = meta.get("request_ids") or {}
    order = (["contract", "work"] if test == "T-001" else ["work", "contract"])
    order += [k for k in rids if k not in order]
    legs, reasons = [], []
    for leg in order:
        rid = rids.get(leg)
        if not rid:
            continue
        d = _leg_dir(run, rid)
        resp = (_read_json(d / "llm_metadata.json") if d else None) or {}
        con = _read_json(d / "consolidated_metadata.json") if d else None
        if isinstance(con, dict):
            legs.append(_consolidation_leg(leg, con, resp if isinstance(resp, dict) else {}))
        else:
            reasons.append(_no_consolidation_reason(leg, resp if isinstance(resp, dict) else {}, run))
    if legs:
        return {"available": True, "legs": legs}
    if not rids:
        reasons.append("이 세트에는 파이프라인 결과가 없습니다" + (f" — {scrub_text(meta.get('error'))}" if meta.get("error") else ""))
    return {"available": False, "reason": " · ".join(reasons) or "통합 검증 기록이 없습니다", "legs": []}


_SOURCE_KO = {"rules": "계약서 규칙 파싱", "model": "모델 추출", "contract_context": "계약서 문맥",
              "manifest_title": "시험 목록 제목", "licensee_party": "본문 규칙(이용자 당사자)",
              "licensee": "모델 추출(이용자)", "user": "모델 추출(이용자)", "keywords": "키워드에서 변환",
              "signature": "서명란", "preamble": "전문(前文)"}
_KIND_KO = {"generated": "생성 정답", "caption": "캡션 정답", "label": "라벨 정답", "site": "사이트 정답"}


def _ev(source_label: Optional[str], summary: str, items: Iterable[Tuple[str, Any]],
        skip_missing: Iterable[str] = ()) -> Dict[str, Any]:
    skip = set(skip_missing)
    out = []
    for label, value in items:
        if label in skip and (value is None or value == "" or value == [] or value == {}):
            continue
        out.append({"label": label, "value": _fmt(value)})
    return {"source_label": source_label, "summary": summary, "items": out}


def _decision(decisions: Any, fields: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    if not isinstance(decisions, list):
        return None
    for d in decisions:
        if isinstance(d, dict) and d.get("field") in fields:
            return d
    return None


def _build_evidence(test: str, rec: Dict[str, Any], resp: Dict[str, Any], vrows: Dict[str, Dict[str, Any]],
                    work_kind: Optional[str], tau: Optional[float]) -> Dict[str, Dict[str, Any]]:
    from module.evaluation.v4.keys23 import KEYS
    meta = rec.get("_meta") or {}
    tp_meta = (resp.get("test_prediction") or {}).get("_meta") if isinstance(resp.get("test_prediction"), dict) else None
    pm = meta.get("pipeline_meta") or tp_meta or {}
    raw_md = resp.get("metadata") if isinstance(resp.get("metadata"), dict) else {}
    con_md = resp.get("consolidated_metadata") if isinstance(resp.get("consolidated_metadata"), dict) else {}
    md = dict(raw_md)
    md.update({k: v for k, v in con_md.items() if v not in (None, "", [], {})})
    rules = resp.get("contract_context_rules") if isinstance(resp.get("contract_context_rules"), dict) else {}
    ctx = meta.get("contract_context") if isinstance(meta.get("contract_context"), dict) else {}
    lp = resp.get("licensee_party_detail") or pm.get("licensee_rule") or {}
    lp = lp if isinstance(lp, dict) else {}
    unparsed = pm.get("unparsed") if isinstance(pm.get("unparsed"), dict) else {}
    decisions = resp.get("consolidation_decisions")
    consolidated = bool(resp.get("consolidation_success")) or bool(con_md and resp.get("consolidate"))
    by_key = _decisions_by_key(decisions)

    def detail(k: str) -> Dict[str, Any]:
        return ((vrows.get(k) or {}).get("detail")) or {}

    from module.evaluation.v4.keys23 import CONTRACT_ITEM_KEYS
    ci = meta.get("contract_items") if isinstance(meta.get("contract_items"), dict) else {}
    ev: Dict[str, Dict[str, Any]] = {}
    for key in KEYS:
        k = key.name
        v = rec.get(k)
        if test == "T-001" and key.test != test:
            ev[k] = _ev(None, f"해당 없음 — {key.test} 전용 항목이라 이 시험({test})에서는 채점하지 않습니다", [])
            continue
        if test == "T-002" and k in CONTRACT_ITEM_KEYS and ci.get("source") in ("te1", "contract_leg"):
            if ci["source"] == "te1":
                ev[k] = _ev("Te-1 결과", f"같은 계약서(sha256 일치)를 처리한 Te-1 실행 {ci.get('run_id')} 의 값을 그대로 씁니다.",
                            [("Te-1 실행", ci.get("run_id")), ("계약서 sha256", ci.get("contract_sha256"))])
            else:
                ev[k] = _ev("계약서 처리", f"Te-1 결과가 없어 이 세트에서 계약서를 처리한 값입니다 ({ci.get('reason') or ''}).",
                            [("계약서 요청 ID", ci.get("request_id"))])
            continue
        if test == "T-002" and k in CONTRACT_ITEM_KEYS and k not in ("work_title", "work_type"):
            ev[k] = _ev(None, "계약서 결과가 없어 비워 둡니다 — Te-1 결과도, 이 세트의 계약서 처리 결과도 없습니다.", [])
            continue
        if k == "work_title":
            if test == "T-001":
                src = pm.get("work_title_source")
                lab = _SOURCE_KO.get(src) if src else None
                summ = ("계약서의 제호 줄을 규칙으로 읽은 값입니다 (관리번호는 떼어 냄). 모델 값은 대조용으로만 남깁니다."
                        if src == "rules" else "규칙 파싱이 제호를 찾지 못해 모델 추출값을 씁니다." if src == "model"
                        else "제호를 찾지 못했습니다.")
                ev[k] = _ev(lab, summ, [("규칙 파싱 원문", rules.get("title_raw")),
                                       ("규칙 파싱 제호", rules.get("title") or pm.get("rule_title")),
                                       ("모델 추출값", pm.get("model_work_title")), ("채택", lab)])
            else:
                lab = _SOURCE_KO.get(ctx.get("source")) if ctx else None
                ev[k] = _ev(lab, "계약서 결과가 없어 계약서 규칙 파싱(또는 시험 목록 제목)으로 얻은 제호를 적습니다 "
                                 "(VLM 참고 문맥과 같은 값)." if ctx else "계약서 문맥이 없어 비워 둡니다.",
                            [("계약서 문맥 제호", ctx.get("title")), ("문맥 출처", lab),
                             ("모델 값", pm.get("model_work_title"))])
        elif k == "file_name":
            ev[k] = _ev("업로드 파일명", "업로드한 파일의 원래 이름(경로 제외, 공백 포함)을 그대로 씁니다.",
                        [("원본 파일명", v), ("파이프라인 저장 이름", resp.get("filename"))])
        elif k == "work_type":
            if test == "T-001":
                src = pm.get("work_type_source")
                lab = _SOURCE_KO.get(src) if src else None
                summ = ("계약서 '저작물의 종류' 체크 칸을 규칙으로 읽었습니다." if src == "rules"
                        else "체크 칸을 읽지 못해 모델 값을 10개 종류로 맞춰 씁니다." if src == "model"
                        else "종류를 정하지 못했습니다.")
                items = [("체크된 칸", rules.get("work_type_ticked")),
                         ("규칙 종별", rules.get("work_type") or pm.get("rule_work_type")),
                         ("모델 추출값", pm.get("model_work_type")), ("채택", lab),
                         ("해석 불가 원문", unparsed.get("work_type"))]
                ev[k] = _ev(lab, summ, items, skip_missing=("해석 불가 원문",))
            else:
                lab = _SOURCE_KO.get(ctx.get("source")) if ctx else None
                ev[k] = _ev(lab, "계약서 결과가 없어 계약서 문맥의 종류를 적습니다." if ctx
                            else "계약서 문맥이 없어 비워 둡니다.",
                            [("계약서 문맥 종류", ctx.get("work_type")), ("모델 값", pm.get("model_work_type"))])
        elif k == "creator":
            d = _decision(decisions, ("author", "creator"))
            items = [("모델 추출값(LLM)", raw_md.get("author")), ("통합 결과", con_md.get("author") if con_md else None)]
            if d:
                items += [("NER 값", d.get("ner_value")),
                          ("통합 판정", _decision_text(d)),
                          ("판정 근거", d.get("reasoning"))]
            ev[k] = _ev("모델 추출 + 통합 검증" if consolidated else "모델 추출",
                        "계약서 '저작자' 항목을 모델이 읽고" + (" LLM·NER 대조(통합 검증)를 거친" if consolidated else "")
                        + " 값을 이름 목록으로 냅니다.", items)
        elif k == "copyright_holder":
            src = pm.get("copyright_holder_source")
            ev[k] = _ev(_SOURCE_KO.get(src) if src else None,
                        "저작권자는 계약서의 이용자 당사자입니다 — 본문 규칙(서명란 → 전문 순)으로 읽고, 모델의 저작권자 값은 "
                        "대조용으로만 남깁니다.",
                        [("서명란", lp.get("signature")), ("전문(前文)", lp.get("preamble")),
                         ("두 출처 일치", lp.get("agree")), ("규칙 채택 출처", _SOURCE_KO.get(lp.get("source"))),
                         ("모델 저작권자 값(쓰지 않음)", pm.get("model_copyright_holder"))])
        elif k in RIGHTS_MAP:
            raw = md.get(RIGHTS_MAP[k])
            ev[k] = _ev("모델 추출(체크 칸)", f"계약서 {key.label_ko} 체크 칸의 표시를 허용/불허로 바꿉니다.",
                        [("모델이 읽은 표시", raw), ("정규화", v), ("해석 불가 원문", unparsed.get(k))],
                        skip_missing=("해석 불가 원문",))
        elif k == "other_rights":
            ev[k] = _ev("고정값", "계약서 서식에 '기타 권리' 칸이 없어 항상 NULL 로 냅니다 (채점 제외).",
                        [("모델 값(쓰지 않음)", pm.get("model_other_rights"))], skip_missing=("모델 값(쓰지 않음)",))
        elif k in ("license_start_date", "license_end_date"):
            raw = md.get(k) if md.get(k) not in (None, "") else md.get(
                "effective_date" if k == "license_start_date" else "expiration_date")
            ev[k] = _ev("모델 추출(정규화)", f"계약서의 {key.label_ko}을 YYYY-MM-DD 로 맞춥니다.",
                        [("모델 원문", raw), ("정규화", v), ("해석 불가 원문", unparsed.get(k))],
                        skip_missing=("해석 불가 원문",))
        elif k == "contract_validity_period":
            raw = pm.get("model_valid_period") or md.get("valid_period")
            ev[k] = _ev("모델 추출(정규화)", "기간 원문을 '시작~종료' 로 맞춥니다 (원문이 없으면 시작일·종료일로 만듭니다).",
                        [("기간 원문", raw), ("시작일", rec.get("license_start_date")),
                         ("종료일", rec.get("license_end_date")), ("정규화", v),
                         ("해석 불가 원문", unparsed.get(k))], skip_missing=("해석 불가 원문",))
        elif k in _TAG_KEYS:
            dt = detail(k)
            items: List[Tuple[str, Any]] = []
            if k == "keywords":
                items.append(("모델 키워드", md.get("keyword") or md.get("keywords")))
                lab, summ = "모델 추출(VLM)", ("VLM 이 그림(영상은 프레임)에서 뽑은 키워드입니다. 정규화한 뒤 정답 목록과 "
                                              "1:1 로 짝짓습니다.")
            else:
                hs = pm.get("hashtags_source")
                lab = "모델 생성(VLM)" if hs == "model" else _SOURCE_KO.get(hs) if hs else None
                items += [("모델 해시태그", md.get("hashtags") or md.get("hashtag")), ("생성 방식", lab)]
                summ = ("VLM 이 만든 해시태그입니다." if hs == "model" else
                        "모델 해시태그가 없어 키워드 앞에 #을 붙였습니다." if hs == "keywords" else "해시태그가 없습니다.")
            if dt:
                items += [("일치한 짝", dt.get("pairs")), ("정답에만 있음", dt.get("unmatched_gold")),
                          ("예측에만 있음", dt.get("unmatched_pred")), ("세트 F1", dt.get("set_f1"))]
            ev[k] = _ev(lab, summ, items)
        elif k == "file_format":
            ext = os.path.splitext(str(rec.get("file_name") or ""))[1].lstrip(".").upper() or None
            ev[k] = _ev("파일 측정", "파일 형식 — 모델의 디지털 형식 값이 있으면 그 값을, 없으면 확장자를 대문자로 씁니다.",
                        [("측정값", v), ("확장자", ext), ("모델 형식 값", raw_md.get("digital_format"))])
        elif k == "file_size":
            items = [("바이트 수", f"{v:,}" if isinstance(v, int) else v),
                     ("MB", f"{v / 1048576:.2f}" if isinstance(v, (int, float)) else None),
                     ("방법", "원본 파일의 바이트 수 (파일 시스템)")]
            ev[k] = _ev("파일 측정", "원본 파일 크기를 바이트로 잽니다.", items)
        elif k == "resolution":
            how = {"image": "이미지 헤더 (Pillow)", "video": "영상 스트림 (ffprobe)",
                   "pptx": "슬라이드 문서 — 해상도 없음"}.get(work_kind or "", "파일 헤더")
            ev[k] = _ev("파일 측정", f"해상도를 {how} 에서 읽습니다.", [("측정값", v), ("방법", how)])
        elif k == "file_hash":
            ev[k] = _ev("파일 측정", "원본 파일 전체 바이트의 SHA-256 입니다.",
                        [("SHA-256", v), ("방법", "SHA-256 (원본 파일 전체)")])
        elif k == "description":
            dt = detail(k)
            cos = dt.get("cosine")
            t = dt.get("tau", tau)
            row = vrows.get(k) or {}
            kind = row.get("kind")
            items = [("설명", v), ("코사인 유사도", cos), ("임계값 τ", t)]
            if cos is not None and t is not None:
                items.append(("판정", "통과" if row.get("matched") else "미달"))
            items += [("공유 내용어", dt.get("shared_tokens")), ("내용 검사", dt.get("content_ok")),
                      ("정답 종류", _KIND_KO.get(kind, kind) if kind else None)]
            if cos is not None and t is not None:
                summ = (f"임베딩(KURE-v1) 코사인 {cos:.3f} {'≥' if cos >= t else '<'} τ {t}"
                        + (" · 내용어 검사 통과" if dt.get("content_ok") else " · 내용어 검사 미달"
                           if dt.get("content_ok") is False else ""))
            else:
                summ = "VLM 이 만든 설명문입니다 (정답이 없어 판정하지 않았습니다)." if not vrows.get(k) else \
                    "설명 판정 기록입니다."
            ev[k] = _ev("모델 생성(VLM)", summ, items)
        else:
            ev[k] = _ev(None, "", [])
        _add_decision_items(ev[k], k, by_key.get(k) or [])
    return ev


_CONSOLIDATION_HINT = "필드 전체 대조는 '통합 검증' 탭에 있습니다"


def _add_decision_items(entry: Dict[str, Any], key: str, rows: List[Dict[str, Any]]) -> None:
    """이 23키의 재료가 된 필드에 중재자 판정이 있으면 근거에 'NER 값 · 통합 판정 · 판정 근거' 를 붙인다 (+ 통합 검증 탭 안내).

    저작자(creator)는 예전부터 author 판정을 직접 붙여 두었으므로 안내만 더한다. 행이 여럿이면(예: 유효기간 ← 유효기간·시작일·종료일)
    라벨 뒤에 필드 이름을 붙여 가른다.
    """
    if not rows:
        return
    items = entry.setdefault("items", [])
    if key != "creator":
        many = len(rows) > 1
        for d in rows:
            sfx = f" · {field_label_ko(d.get('field')) or d.get('field')}" if many else ""
            items += [{"label": f"NER 값{sfx}", "value": _fmt(d.get("ner_value"))},
                      {"label": f"통합 판정{sfx}", "value": _decision_text(d)},
                      {"label": f"판정 근거{sfx}", "value": _fmt(d.get("reasoning"))}]
    items.append({"label": "통합 검증", "value": _CONSOLIDATION_HINT})


def _display(key, value: Any, test: str) -> str:
    if value is None or value == "" or value == []:
        # 다른 시험의 항목은 이 시험에서 추출·채점하지 않으므로 NULL(값 없음)로 보인다 — "해당 없음" 은
        # 이 문서에 그런 항목이 없는 경우(기타 권리 등)에만 쓴다 (M3 spec §5)
        return "해당 없음" if key.name == "other_rights" else "NULL"
    if key.name == "file_size" and isinstance(value, (int, float)):
        return f"{int(value):,} 바이트 ({value / 1048576:.2f} MB)"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(x) for x in value)
    return str(value)


def _partner_runs(run: TestRun) -> List[TestRun]:
    """같은 폴더 시험에서 함께 돈 다른 시험의 실행 (Te-1 ↔ Te-2). 동시 실행은 config.parallel, 예전 순차 실행은 after/next."""
    ids = [p.get("run_id") for p in (run.config.get("parallel") or []) if isinstance(p, dict)]
    ids += [run.config.get("after"), run.config.get("next")]
    out: List[TestRun] = []
    for rid in ids:
        if not rid or str(rid) == run.run_id:
            continue
        with _RUNS_LOCK:
            o = _RUNS.get(str(rid))
        if o is not None and o.test and o.test != run.test and o not in out:
            out.append(o)
    return out


def _set_detail(run: TestRun, set_id: str, merge: bool = True) -> Dict[str, Any]:
    """세트 하나의 23개 항목·근거·미리보기. merge 면 이 시험이 채점하지 않는 항목(Te-1 의 17~23번)을 짝 실행(Te-2)의
    값·판정·근거로 채운다 — 한 화면에서 23개를 모두 본다. Te-2 는 1~23번을 모두 채점하므로(1~16번은 Te-1 결과를
    쓰거나 계약서를 직접 처리) 자기 값을 그대로 보인다. 각 행의 from_test·from_run 이 어느 실행의 결과인지 알린다."""
    from module.evaluation.v4.keys23 import KEYS, scored_keys
    rec = _load_record(run, set_id)
    meta = rec.get("_meta") or {}
    test = rec.get("test") or meta.get("test") or run.test
    idx = _run_index(run)
    vrows = idx["verdicts"].get(str(set_id)) or {}
    grec = gold_record(run.gold_path, set_id) if run.gold_path else None
    gkeys = (grec or {}).get("keys") or {}
    rids = meta.get("request_ids") or {}
    primary = "contract" if test == "T-001" else "work"
    pdir = _leg_dir(run, rids.get(primary))
    resp = (_read_json(pdir / "llm_metadata.json") if pdir else None) or {}
    work_kind = _work_kind((meta.get("files") or {}).get("work"), meta.get("media")) if test == "T-002" else None
    tau = (run.cfg_kwargs or {}).get("tau", DEFAULT_TAU)
    evidence = _build_evidence(test, rec, resp, vrows, work_kind, tau)

    keys = []
    for key in KEYS:
        row = vrows.get(key.name)
        g = gkeys.get(key.name) if isinstance(gkeys.get(key.name), dict) else None
        det, counts = None, None
        if row:
            det = row.get("detail") if isinstance(row.get("detail"), dict) else {}
            counts = {"tp": row.get("tp"), "fp": row.get("fp"), "fn": row.get("fn"), "state": row.get("state"),
                      "kind": row.get("kind"), "matched": row.get("matched")}
        keys.append({
            "no": key.no, "key": key.name, "label_ko": key.label_ko, "group_ko": key.group_ko, "test": key.test,
            "scored": key.name in scored_keys(test) and key.type != "na",
            "value": rec.get(key.name), "display": _display(key, rec.get(key.name), test),
            "source": (evidence.get(key.name) or {}).get("source_label"),
            "verdict": verdict_of(row),
            "gold": (g or {}).get("value") if g else (row.get("gold") if row else None),
            "gold_state": (g or {}).get("state") if g else (row.get("state") if row else None),
            "detail": det, "counts": counts,
        })

    legs: Dict[str, Any] = {}
    for leg, info in (meta.get("legs") or {}).items():
        entry = dict(info) if isinstance(info, dict) else {"value": info}
        d = _leg_dir(run, entry.get("request_id"))
        if d is not None:
            files = []
            for f in sorted(d.rglob("*")):
                if f.is_file() and len(files) < 500:
                    files.append({"path": str(f.relative_to(d)), "bytes": f.stat().st_size})
            entry["files"] = files
        legs[leg] = entry
    # 모델 — 배포 구성(MODELS) + 이 세트에서 실제로 쓰인 것. 레코드의 "alibaba/(기본)" 같은 설정 문자열은 config 에만 둔다.
    rm = meta.get("models") or {}
    pleg = (meta.get("legs") or {}).get(primary) or {}
    models: Dict[str, Any] = dict(MODELS)
    models["model_used"] = rm.get("model_used") or pleg.get("model_used")
    models["consolidation"] = rm.get("consolidation")
    if pleg.get("ocr_provider_used") == "direct_text":
        models["ocr_used"] = "PDF 텍스트층 (OCR 생략)"
    elif pleg.get("ocr_model"):
        models["ocr_used"] = pleg.get("ocr_model")
    if pleg.get("vlm_backend"):
        models["vlm_backend"] = pleg.get("vlm_backend")
    models["config"] = rm
    previews = _previews(run, set_id, rec)
    partner: Optional[Dict[str, Any]] = None
    if merge:
        for o in _partner_runs(run):
            if not _record_path(o, set_id).is_file():
                partner = partner or {"run_id": o.run_id, "test": o.test, "available": False}
                continue
            try:
                pd = _set_detail(o, set_id, merge=False)
            except Exception as e:                  # 짝 실행을 못 읽어도 이 실행의 상세는 그대로 보인다
                logger.warning("짝 실행 %s 의 세트 %s 를 읽지 못했습니다: %s", o.run_id, set_id, e)
                continue
            ptest = pd.get("test")
            pkeys = {k.get("key"): k for k in pd.get("keys") or []}
            pev = pd.get("evidence") or {}
            for i, k in enumerate(keys):
                if k["test"] == ptest and k["key"] not in scored_keys(test) and k["key"] in pkeys:
                    keys[i] = dict(pkeys[k["key"]], from_run=o.run_id, from_test=ptest)
                    if k["key"] in pev:
                        evidence[k["key"]] = pev[k["key"]]
            ppv = pd.get("previews") or {}
            if ptest == "T-002" and ((ppv.get("work") or {}).get("image_url") or (ppv.get("work") or {}).get("frames")):
                previews["work"] = ppv["work"]
            if ptest == "T-001" and not (previews.get("contract") or {}).get("pages"):
                previews["contract"] = ppv.get("contract") or {"pages": []}
            partner = {"run_id": o.run_id, "test": ptest, "available": True, "ok": pd.get("ok"),
                       "error": pd.get("error"), "gold": pd.get("gold"), "record": pd.get("record")}
            break
    out = {
        "run_id": run.run_id, "set_id": str(set_id), "test": test,
        "ok": meta.get("ok"), "error": scrub_text(meta.get("error")) if meta.get("error") else None,
        "gold": bool(grec), "record": rec, "keys": keys, "evidence": evidence,
        "previews": previews, "legs": legs, "models": models,
        "consolidation": _consolidation_payload(run, rec, test),
        "partner": partner,
    }
    return scrub_json(out)


# ---------------------------------------------------------------------------
# 미리보기 파일
# ---------------------------------------------------------------------------
def _downscale_jpeg(src: Path, dest: Path, max_side: int = PREVIEW_MAX_SIDE) -> Path:
    try:
        if dest.is_file() and dest.stat().st_mtime >= src.stat().st_mtime:
            return dest
    except OSError:
        pass
    from PIL import Image, ImageOps
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".tmp{threading.get_ident()}")
    with Image.open(src) as im:
        try:
            im.draft("RGB", (max_side, max_side))
        except Exception:
            pass
        try:
            im = ImageOps.exif_transpose(im)
        except Exception:
            pass
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        im.save(tmp, "JPEG", quality=85)
    os.replace(tmp, dest)
    return dest


def _contract_png(src: Path, page: int, dest: Path) -> Path:
    try:
        if dest.is_file() and dest.stat().st_mtime >= src.stat().st_mtime:
            return dest
    except OSError:
        pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + f".tmp{threading.get_ident()}.png")
    if src.suffix.lower() == ".pdf":
        import fitz
        with fitz.open(str(src)) as doc:
            if page < 1 or page > doc.page_count:
                raise IndexError(page)
            doc[page - 1].get_pixmap(dpi=CONTRACT_DPI).save(str(tmp))
    else:
        if page != 1:
            raise IndexError(page)
        from PIL import Image
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((1600, 1600))
            im.save(tmp, "PNG")
    os.replace(tmp, dest)
    return dest


# ---------------------------------------------------------------------------
# 내려받기 (zip 은 흘려보낸다 — 4,000세트 프레임까지 메모리에 올리지 않는다)
# ---------------------------------------------------------------------------
PREDICTIONS_ZIP = "predictions_json.zip"   # 속성정보 파일(*.json) — test_predictions/<set_id>.json 만 묶는다 (절차서 6단계)
DOWNLOADS = ("test_predictions.jsonl", "test_metrics.json", "report.md", "test.log", "per_set.jsonl", "run.zip",
             PREDICTIONS_ZIP)
UNFINISHED_CSV = "unfinished.csv"          # 실패·미실행 세트 목록 (만들어서 준다)


class _ZipSink(io.RawIOBase):
    def __init__(self):
        self._buf = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:
        self._buf.extend(b)
        return len(b)

    def take(self) -> bytes:
        out = bytes(self._buf)
        self._buf.clear()
        return out


def _scrubbed_bytes(path: Path) -> Optional[bytes]:
    """텍스트 산출물은 내부 이름을 지운 사본으로 — 50MB 넘는 파일은 그대로."""
    ext = path.suffix.lower()
    if ext not in (".json", ".jsonl", ".md", ".log", ".txt"):
        return None
    try:
        if path.stat().st_size > 50 * 1024 * 1024:
            return None
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if ext == ".json":
        try:
            return json.dumps(scrub_json(json.loads(text)), ensure_ascii=False, indent=1).encode("utf-8")
        except json.JSONDecodeError:
            return scrub_text(text).encode("utf-8")
    if ext == ".jsonl":
        out = []
        for line in text.splitlines():
            try:
                out.append(json.dumps(scrub_json(json.loads(line)), ensure_ascii=False) if line.strip() else line)
            except json.JSONDecodeError:
                out.append(line)
        return ("\n".join(out) + "\n").encode("utf-8")
    return scrub_text(text).encode("utf-8")


def _zip_stream(items: Iterable[Tuple[str, Any]]) -> Iterator[bytes]:
    sink = _ZipSink()
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for arc, src in items:
            if isinstance(src, (bytes, bytearray)):
                zf.writestr(arc, bytes(src))
            else:
                p = Path(src)
                data = _scrubbed_bytes(p)
                if data is not None:
                    zf.writestr(arc, data)
                else:
                    zi = zipfile.ZipInfo.from_file(str(p), arc)
                    zi.compress_type = (zipfile.ZIP_STORED if p.suffix.lower() in _MEDIA_FILE_EXT
                                        else zipfile.ZIP_DEFLATED)
                    with p.open("rb") as f, zf.open(zi, "w", force_zip64=True) as dst:
                        while True:
                            chunk = f.read(1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)
                            out = sink.take()
                            if out:
                                yield out
            out = sink.take()
            if out:
                yield out
    out = sink.take()
    if out:
        yield out


def _run_zip_items(run: TestRun) -> Iterator[Tuple[str, Any]]:
    prefix = run.run_id
    yield f"{prefix}/run.json", json.dumps(run.detail(), ensure_ascii=False, indent=1).encode("utf-8")
    if (run.dir / "manifest.jsonl").is_file():
        yield f"{prefix}/manifest.jsonl", run.dir / "manifest.jsonl"
    skip = run.out_dir / "runs" / "uploads"
    for base in (run.out_dir, run.dir / "inputs"):
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if not f.is_file() or _inside(f, skip) or ".tmp" in f.name or not _inside(f, run.dir):
                continue
            yield f"{prefix}/{f.relative_to(run.dir).as_posix()}", f


def _predictions_zip_items(run: TestRun) -> Iterator[Tuple[str, Any]]:
    """세트마다 하나인 속성정보 파일(*.json) — 압축을 풀면 test_predictions/<set_id>.json (콘솔이 보여 주는 값과 같다)."""
    if not run.pred_dir.is_dir():
        return
    for f in sorted(run.pred_dir.glob("*.json")):
        if f.is_file() and ".tmp" not in f.name and _inside(f, run.dir):
            yield f"test_predictions/{f.name}", f


def _attachment(name: str) -> Dict[str, str]:
    safe = re.sub(r"[^\w.\-]", "_", name)
    return {"Content-Disposition": f'attachment; filename="{safe}"', "Cache-Control": "no-store"}


def _metrics_view(m: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(m)
    m.pop("spec", None)
    inputs = m.get("inputs")
    if isinstance(inputs, dict):
        m["inputs"] = {k: (os.path.basename(str(v).rstrip("/")) if v else v) for k, v in inputs.items()}
    m.pop("outputs", None)
    m["pass_mark"] = PASS_MARK
    return scrub_json(m)


# ---------------------------------------------------------------------------
# 상태 — 모델 체인 · DashScope 확인
# ---------------------------------------------------------------------------
_CHAINS: Dict[str, List[str]] = {}


def _chain_labels() -> Tuple[List[str], List[str]]:
    if not _CHAINS:
        try:
            from module.clip_extraction.vlm.extractor import default_backends
            _CHAINS["image"] = [b.model_label for b in default_backends("auto")]
            _CHAINS["video"] = [b.model_label for b in default_backends("omni")]
        except Exception as e:
            logger.info("VLM 체인 라벨을 읽지 못해 기본 목록을 씁니다: %s", type(e).__name__)
            _CHAINS["image"], _CHAINS["video"] = list(_IMAGE_CHAIN_FALLBACK), list(_VIDEO_CHAIN_FALLBACK)
    return _CHAINS["image"], _CHAINS["video"]


def _ping_dashscope() -> str:
    """DashScope 인증 확인 — 키로 /models 를 부른다. 키는 어디에도 남기지 않는다 (로그에는 호스트·상태만)."""
    key = (os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if not key:
        logger.warning("DashScope 확인 — DASHSCOPE_API_KEY 가 없습니다", extra={"component": "API"})
        return "fail"
    base = (os.environ.get("DASHSCOPE_BASE_URL") or "").strip() or \
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    host = urlsplit(base).hostname or "?"
    t0 = time.perf_counter()
    try:
        import httpx
        r = httpx.get(base.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=4.0)
        ok = r.status_code == 200
        (logger.info if ok else logger.warning)("DashScope 확인 — %s · HTTP %d · %.1f초 → %s", host, r.status_code,
                                                time.perf_counter() - t0, "정상" if ok else "실패",
                                                extra={"component": "API"})
        return "ok" if ok else "fail"
    except Exception as e:
        logger.warning("DashScope 확인 — %s · %s · %.1f초 → 실패", host, type(e).__name__, time.perf_counter() - t0,
                       extra={"component": "API"})
        return "fail"


_PING: Dict[str, Any] = {"value": "unknown", "at": 0.0, "busy": False, "event": None}
_PING_LOCK = threading.Lock()


def _ping_worker(ev: threading.Event) -> None:
    try:
        v = _ping_dashscope()
    except Exception:
        v = "fail"
    with _PING_LOCK:
        _PING.update(value=v if v in ("ok", "fail") else "unknown", at=time.time(), busy=False)
    ev.set()


def _dashscope_status(wait: float = 4.5) -> str:
    now = time.time()
    with _PING_LOCK:
        if _PING["at"] and now - _PING["at"] < PING_TTL_SEC:
            return _PING["value"]
        if not _PING["busy"]:
            _PING["busy"] = True
            _PING["event"] = threading.Event()
            threading.Thread(target=_ping_worker, args=(_PING["event"],), daemon=True,
                             name="console-ping").start()
        ev, first = _PING["event"], not _PING["at"]
    if first and ev is not None:
        ev.wait(wait)
    with _PING_LOCK:
        return _PING["value"]


# ---------------------------------------------------------------------------
# 로그인 확인
# ---------------------------------------------------------------------------
def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin or origin == "null":
        return origin != "null"
    return urlsplit(origin).netloc.lower() == (request.headers.get("host") or "").lower()


# 시험 PC 에서는 로그인 없이 — 시험원이 같은 PC 의 브라우저로 여는 간소화 콘솔(/test)용.
# 켜짐 조건: CONSOLE_NO_LOGIN (값이 없으면 WSL 에서만 켬 — 공개 서버는 기본 꺼짐) + 같은 PC(루프백)에서 온 요청 +
# 프록시 머리말 없음(nginx 같은 앞단을 거친 요청은 루프백처럼 보여도 바깥 요청이다).
NO_LOGIN_ENV = "CONSOLE_NO_LOGIN"
LOCAL_USER = "시험"
_LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
_PROXY_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded", "x-forwarded-host")


def no_login_enabled() -> bool:
    v = (os.environ.get(NO_LOGIN_ENV) or "").strip().lower()
    if v:
        return v in ("1", "on", "true", "yes")
    return CF.is_wsl()


def _client_host(request: Request) -> str:
    return request.client.host if request.client else ""


def _open_local(request: Request) -> bool:
    if not no_login_enabled() or _client_host(request) not in _LOOPBACK:
        return False
    if any(request.headers.get(h) for h in _PROXY_HEADERS):
        return False
    # 브라우저가 쓴 주소도 이 PC 여야 한다 — 다른 사이트의 도메인을 127.0.0.1 로 돌려(DNS 리바인딩) 들어오는 것을 막는다
    host = (request.headers.get("host") or "").strip().lower()
    name = host[1:host.index("]")] if host.startswith("[") and "]" in host else host.rsplit(":", 1)[0]
    return name in ("localhost", "127.0.0.1", "::1")


def _api_user(request: Request) -> str:
    if _open_local(request):
        if request.method not in ("GET", "HEAD", "OPTIONS") and not _same_origin(request):
            raise ConsoleError(403, "forbidden", "다른 사이트에서 보낸 요청은 받지 않습니다")
        return LOCAL_USER
    s = auth.get_settings()
    if not s.configured:
        raise ConsoleError(503, "console_not_configured",
                           f"콘솔 비밀번호가 설정되지 않았습니다 — 서버에서 `{auth.SET_PASSWORD_HINT}` 를 실행하세요")
    user = auth.session_user(request, s)
    if not user:
        raise ConsoleError(401, "unauthorized", "로그인이 필요합니다")
    if request.method not in ("GET", "HEAD", "OPTIONS") and not _same_origin(request):
        raise ConsoleError(403, "forbidden", "다른 사이트에서 보낸 요청은 받지 않습니다")
    return user


def _render(name: str, ctx: Dict[str, Any], status_code: int = 200,
            headers: Optional[Dict[str, str]] = None) -> Response:
    hdrs = {"Cache-Control": "no-store", **(headers or {})}
    if _TEMPLATES is not None:
        try:
            return _TEMPLATES.TemplateResponse(name, ctx, status_code=status_code, headers=hdrs)
        except Exception as e:                       # 템플릿이 아직 없거나 깨졌으면 최소 화면으로
            if type(e).__name__ != "TemplateNotFound":
                logger.error("콘솔 화면 렌더링 실패 %s: %s", name, e, exc_info=True)
    return HTMLResponse(_fallback_page(name, ctx), status_code=status_code, headers=hdrs)


def _esc(s: Any) -> str:
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


_PAGE_CSS = ("body{margin:0;background:#0f1216;color:#d8dde3;font:15px/1.6 system-ui,'Malgun Gothic',sans-serif;"
             "display:flex;min-height:100vh;align-items:center;justify-content:center;padding:16px}"
             ".box{max-width:560px;width:100%;background:#171b21;border:1px solid #2a3038;border-radius:8px;"
             "padding:28px}h1{font-size:18px;margin:0 0 4px}small{color:#8a96a3}code{background:#0b0e12;"
             "padding:2px 6px;border-radius:4px;color:#f0b350;word-break:break-all}input{width:100%;"
             "box-sizing:border-box;margin:6px 0 12px;padding:8px;background:#0b0e12;color:#d8dde3;"
             "border:1px solid #2a3038;border-radius:4px}button{padding:8px 18px;background:#f0b350;border:0;"
             "border-radius:4px;font-weight:600}.err{color:#ff8a7a}")


def _fallback_page(name: str, ctx: Dict[str, Any]) -> str:
    title = f"{SYSTEM_NAME} {VERSION} — 시험 콘솔"
    if name == "console_login.html":
        err = f'<p class="err">{_esc(ctx.get("error"))}</p>' if ctx.get("error") else ""
        body = (f'<h1>{_esc(title)}</h1><small>로그인</small>{err}'
                f'<form method="post" action="/console/login"><input type="hidden" name="next" '
                f'value="{_esc(ctx.get("next"))}"><label>아이디<input name="user" autocomplete="username"></label>'
                f'<label>비밀번호<input name="password" type="password" autocomplete="current-password"></label>'
                f'<button type="submit">로그인</button></form>')
    else:
        body = (f'<h1>{_esc(title)}</h1><p>화면 파일(console.html)을 찾을 수 없습니다. 로그인 사용자: '
                f'{_esc(ctx.get("user"))}</p><p><a href="/console/logout" style="color:#f0b350">로그아웃</a></p>')
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{_esc(title)}</title><style>{_PAGE_CSS}</style>'
            f'</head><body><div class="box">{body}</div></body></html>')


def _not_configured_page() -> HTMLResponse:
    title = f"{SYSTEM_NAME} {VERSION} — 시험 콘솔"
    body = (f'<h1>{_esc(title)}</h1><small>초기 설정 필요</small>'
            '<p>콘솔 비밀번호가 아직 설정되지 않았습니다. 서버(저장소 루트)에서 다음 명령을 실행해 아이디와 비밀번호를 '
            '정하세요. 비밀번호는 입력 창에서 받으며 화면에 남지 않습니다.</p>'
            f'<p><code>{_esc(auth.SET_PASSWORD_HINT)}</code></p>'
            '<p>설정은 저장소 루트의 <code>.env</code> 에 저장되고, 실행 중인 서버는 다음 요청부터 적용합니다. '
            '설정한 뒤 이 화면을 새로 고치세요.</p>')
    html = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{_esc(title)}</title><style>{_PAGE_CSS}</style>'
            f'</head><body><div class="box">{body}</div></body></html>')
    return HTMLResponse(html, status_code=503, headers={"Cache-Control": "no-store"})


def _login_redirect(request: Request) -> RedirectResponse:
    target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    return RedirectResponse(f"/console/login?next={quote(auth.safe_next(target), safe='/')}", status_code=303)


# ---------------------------------------------------------------------------
# 화면 라우트
# ---------------------------------------------------------------------------
@router.get("/console", include_in_schema=False)
async def console_page(request: Request):
    s = auth.get_settings()
    if not s.configured:
        return _not_configured_page()
    user = auth.session_user(request, s)
    if not user:
        return _login_redirect(request)
    return _render("console.html", {"request": request, "user": user, "version": VERSION})


@router.get("/console/login", include_in_schema=False)
async def console_login_page(request: Request, next: str = "/console"):
    s = auth.get_settings()
    if not s.configured:
        return _not_configured_page()
    nxt = auth.safe_next(next)
    if auth.session_user(request, s):
        return RedirectResponse(nxt, status_code=303)
    return _render("console_login.html", {"request": request, "error": None, "next": nxt, "version": VERSION})


@router.post("/console/login", include_in_schema=False)
async def console_login_submit(request: Request):
    s = auth.get_settings()
    if not s.configured:
        return _not_configured_page()
    try:
        form = await request.form()
    except Exception:
        form = {}
    user = str(form.get("user") or "").strip()
    password = str(form.get("password") or "")
    nxt = auth.safe_next(str(form.get("next") or ""))
    ip = request.client.host if request.client else "-"
    wait = auth.THROTTLE.retry_after(ip)
    if wait > 0:
        secs = max(1, int(wait + 0.999))
        return _render("console_login.html",
                       {"request": request, "next": nxt, "version": VERSION,
                        "error": f"로그인 시도가 너무 많습니다. {secs}초 뒤에 다시 시도하세요."},
                       status_code=429, headers={"Retry-After": str(secs)})
    ok = await run_in_threadpool(auth.check_credentials, s, user, password)
    if not ok:
        auth.THROTTLE.record_failure(ip)
        await asyncio.sleep(auth.FAIL_DELAY_SEC)
        return _render("console_login.html",
                       {"request": request, "next": nxt, "version": VERSION,
                        "error": "아이디 또는 비밀번호가 올바르지 않습니다."}, status_code=401)
    auth.THROTTLE.reset(ip)
    resp = RedirectResponse(nxt, status_code=303)
    auth.set_session_cookie(resp, request, auth.make_session_token(s, s.user))
    return resp


@router.get("/console/logout", include_in_schema=False)
async def console_logout(request: Request):
    resp = RedirectResponse("/console/login", status_code=303)
    auth.clear_session_cookie(resp, request)
    return resp


@router.get("/test", include_in_schema=False)
async def simple_console_page(request: Request):
    """간소화 시험 콘솔 (시험 · 진행 · 결과·지표) — 시험 PC 에서는 로그인 없이, 그 밖에는 콘솔 로그인으로."""
    if _open_local(request):
        user, open_mode = LOCAL_USER, True
    else:
        s = auth.get_settings()
        if not s.configured:
            return _not_configured_page()
        user, open_mode = auth.session_user(request, s), False
        if not user:
            return _login_redirect(request)
    return _render("console_simple.html", {"request": request, "user": user, "version": VERSION, "open_mode": open_mode})


@router.get("/console/{rest:path}", include_in_schema=False)
async def console_deep_link(request: Request, rest: str):
    """/console/… 의 나머지 경로 — 로그인 전이면 로그인으로, 로그인했으면 같은 콘솔 화면 (새로 고침해도 열린다)."""
    return await console_page(request)


# ---------------------------------------------------------------------------
# API — 상태 · 추정
# ---------------------------------------------------------------------------
@router.get(API + "/status")
def api_status(user: str = Depends(_api_user)):
    image_chain, video_chain = _chain_labels()
    batches, a_single = _active_runs("batch"), _active_run("single")
    return {
        "version": VERSION, "system": SYSTEM_NAME, "user": user,
        "models": {**MODELS, "image_chain": image_chain, "video_chain": video_chain},
        "services": {"dashscope": _dashscope_status()},
        "active": {"batch": batches[0].run_id if batches else None,
                   "batches": [{"run_id": r.run_id, "test": r.test} for r in batches],
                   "single": a_single.run_id if a_single else None},
        "sources": [_source_view(s) for s in _SOURCES],
        "defaults": {"tau": DEFAULT_TAU, "workers": dict(DEFAULT_WORKERS), "max_workers": MAX_WORKERS},
        "pass_mark": PASS_MARK,
    }


@router.get(API + "/estimate")
def api_estimate(request: Request, user: str = Depends(_api_user)):
    q = request.query_params
    test = _test_param(q.get("test"))
    n = _int_param(q.get("n"), "n", None, 1, 1_000_000)
    seed = _int_param(q.get("seed"), "seed", 0, 0, 2 ** 31 - 1)
    workers = _int_param(q.get("workers"), "workers", None, 1, MAX_WORKERS)
    sample = "random" if (q.get("sample") or "").strip() == "random" else "first"
    test, entries, missing, gold, _src = _select(test, q.get("source"), _list_param(q.get("ids")),
                                                 _list_param(q.get("cells")), n, sample, seed)
    est = _estimate(test, entries, workers, _bool_param(q.get("consolidate_work"), False))
    return {**est, "test": test, "gold": bool(gold), "missing_ids": missing}


# ---------------------------------------------------------------------------
# API — 시스템 로그
# ---------------------------------------------------------------------------
@router.get(API + "/syslog")
def api_syslog(request: Request, user: str = Depends(_api_user)):
    """서버 전체 로그(최근 SYSLOG_MAX 건). after=<seq> 이면 그 다음부터, 없거나 0 이면 조건에 맞는 마지막 limit 건.

    level: 하나면 '그 레벨 이상'(예: WARNING → WARNING·ERROR·CRITICAL), 쉼표로 여럿이면 그 레벨들만.
    component: 구성요소(쉼표로 여럿). q: 메시지·로거·세트에서 찾을 문자열. run_id·set_id: 그 실행·세트만.
    """
    q = request.query_params
    after = _int_param(q.get("after"), "after", 0, 0, 2 ** 62)
    limit = _int_param(q.get("limit"), "limit", 500, 1, SYSLOG_MAX)
    lv = [x.upper() for x in _list_param(q.get("level"))]
    bad = [x for x in lv if x not in _LEVEL_NO]
    if bad:
        raise ConsoleError(400, "bad_request", f"알 수 없는 레벨입니다: {', '.join(bad)}")
    levels = {("WARNING" if x == "WARN" else x) for x in lv} if len(lv) > 1 else None
    min_level = _LEVEL_NO[lv[0]] if len(lv) == 1 else None
    comps = set(_list_param(q.get("component"))) or None
    if comps and not comps <= set(LC.COMPONENTS):
        raise ConsoleError(400, "bad_request", f"알 수 없는 구성요소입니다: {', '.join(sorted(comps - set(LC.COMPONENTS)))}")
    out = _SYSLOG.query(after=after, levels=levels, min_level=min_level, components=comps,
                        text=(q.get("q") or "").strip(), run_id=(q.get("run_id") or "").strip() or None,
                        set_id=(q.get("set_id") or "").strip() or None, limit=limit)
    out["components"] = list(LC.COMPONENTS)
    return out


# ---------------------------------------------------------------------------
# API — 실행 만들기
# ---------------------------------------------------------------------------
def _upload_field(v: Any):
    return v if (v is not None and hasattr(v, "filename") and hasattr(v, "read") and v.filename) else None


def _clean_filename(name: str) -> str:
    n = str(name or "").replace("\\", "/").split("/")[-1]
    n = "".join(c for c in n if ord(c) >= 32 and ord(c) != 127).strip()
    if n in ("", ".", ".."):
        n = "upload"
    if len(n.encode("utf-8")) > 200:
        stem, ext = os.path.splitext(n)
        while len((stem + ext).encode("utf-8")) > 200 and stem:
            stem = stem[:-1]
        n = (stem or "upload") + ext[:12]
    return n


async def _save_upload(up, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / _clean_filename(up.filename)
    size = 0
    with path.open("wb") as f:
        while True:
            chunk = await up.read(4 * 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise ConsoleError(413, "too_large", "파일이 너무 큽니다 (4GB 초과)")
            f.write(chunk)
    if size == 0:
        raise ConsoleError(400, "empty_file", f"빈 파일입니다: {path.name}")
    return path


@router.post(API + "/single")
async def api_single(request: Request, user: str = Depends(_api_user)):
    from module.evaluation.manifest import ManifestEntry, write_manifest
    try:
        form = await request.form()
    except Exception:
        raise ConsoleError(400, "bad_request", "업로드 양식을 읽을 수 없습니다")
    test = _test_param(form.get("test")) or ""
    if test not in TESTS:
        raise ConsoleError(400, "bad_test", "시험 구분(Te-1 또는 Te-2)을 고르세요")
    contract, work = _upload_field(form.get("contract")), _upload_field(form.get("work"))
    if test == "T-001":
        work = None                                    # T-001 은 계약서만 본다
        if contract is None:
            raise ConsoleError(400, "missing_file", "Te-1 은 계약서 파일이 필요합니다")
    elif work is None:
        raise ConsoleError(400, "missing_file", "Te-2 는 저작물 파일이 필요합니다")
    if contract is not None and os.path.splitext(contract.filename)[1].lower() not in CONTRACT_EXT:
        raise ConsoleError(400, "bad_file", "계약서는 PDF·DOCX·HWP·이미지 파일만 받습니다")
    if work is not None and os.path.splitext(work.filename)[1].lower() not in WORK_EXT:
        raise ConsoleError(400, "bad_file", "저작물은 이미지·영상·PPTX 파일만 받습니다")
    tau = _float_param(form.get("tau"), "tau", DEFAULT_TAU, 0.0, 1.0)
    consolidate_work = _bool_param(form.get("consolidate_work"), False)
    active = _active_run("single")
    if active is not None:
        raise _busy_error(active)

    run = TestRun(_new_run_id("single"), "single", _ROOT)
    run.dir.mkdir(parents=True, exist_ok=True)
    try:
        cpath = await _save_upload(contract, run.dir / "inputs" / "contract") if contract is not None else None
        wpath = await _save_upload(work, run.dir / "inputs" / "work") if work is not None else None
        primary = cpath if test == "T-001" else wpath
        gold_path = _SINGLE_GOLD.get(test)
        gold_path = gold_path if gold_path and gold_path.is_file() else None
        known = await run_in_threadpool(_known_ids, test)
        sid, has_gold = await run_in_threadpool(_single_set_id, primary.name, gold_path, known)
        grec = gold_record(gold_path, sid) if has_gold else None
        bucket = str((grec or {}).get("bucket") or "")
        if test == "T-001":
            media = str((grec or {}).get("media") or "text")
        else:
            media = str((grec or {}).get("media") or _media_of_path(primary.name) or "image")
        title = None
        if grec:
            title = (((grec.get("keys") or {}).get("work_title") or {}).get("value"))
        entry = ManifestEntry(set_id=sid, contract=str(cpath) if cpath else None, work=str(wpath) if wpath else None,
                              media=media, license_bucket=bucket, test=test,
                              extra={"cell": f"{media}/{bucket}" if bucket else media,
                                     **({"title": title} if title else {}), "upload": primary.name})
        run.name = f"{CF.test_label(test)} 단건 · {primary.name}"[:120]
        run.test, run.source, run.set_id, run.gold, run.n_sets = test, "upload", sid, has_gold, 1
        run.gold_path = str(gold_path) if has_gold else None
        run.file_to_set = {primary.name: sid}
        # 단건 Te-2 의 계약서는 제목·종별 힌트로만 쓴다 (화면 안내 그대로) — 1~16번은 일괄 시험에서 채운다
        run.cfg_kwargs = {"workers": 1, "scoring_set": "eval23", "test": test, "frame_schedule": "v4",
                          "consolidate_work": consolidate_work, "only_predicted": True, "title_fallback": False,
                          "tau": tau, "gold": run.gold_path, "contract_items": "off"}
        run.config = {"test": test, "source": "upload", "set_id": sid, "workers": 1, "tau": tau, "only_predicted": True,
                      "title_fallback": False, "consolidate_work": consolidate_work,
                      "contract_file": cpath.name if cpath else None, "work_file": wpath.name if wpath else None,
                      "gold": os.path.basename(run.gold_path) if run.gold_path else None}
        write_manifest([entry], run.dir / "manifest.jsonl")
        _register(run)
    except BaseException:
        shutil.rmtree(run.dir, ignore_errors=True)
        raise
    _start(run, [entry])
    return {"run_id": run.run_id, "set_id": sid, "gold": has_gold}


@router.post(API + "/batch")
async def api_batch(request: Request, user: str = Depends(_api_user)):
    from module.evaluation.manifest import write_manifest
    try:
        body = await request.json()
    except Exception:
        raise ConsoleError(400, "bad_request", "요청 본문(JSON)을 읽을 수 없습니다")
    if not isinstance(body, dict):
        raise ConsoleError(400, "bad_request", "요청 본문은 JSON 객체여야 합니다")
    test = _test_param(body.get("test"))
    source = body.get("source")
    ids = _list_param(body.get("ids"))
    cells = _list_param(body.get("cells"))
    n = _int_param(body.get("n"), "n", None, 1, 1_000_000)
    sample = "random" if str(body.get("sample") or "first") == "random" else "first"
    seed = _int_param(body.get("seed"), "seed", None, 0, 2 ** 31 - 1)
    if sample == "random" and seed is None:
        seed = secrets.randbelow(2 ** 31)              # 재현할 수 있게 실제로 쓴 seed 를 설정에 남긴다
    active = _active_run("batch")
    if active is not None:
        raise _busy_error(active)
    test, entries, missing, gold_path, src = await run_in_threadpool(_select, test, source, ids, cells, n, sample, seed)
    if not entries:
        raise ConsoleError(400, "empty", "선택한 조건에 맞는 세트가 없습니다", missing_ids=missing)
    workers = _int_param(body.get("workers"), "workers", DEFAULT_WORKERS[test], 1, MAX_WORKERS)
    tau = _float_param(body.get("tau"), "tau", DEFAULT_TAU, 0.0, 1.0)
    only_predicted = _bool_param(body.get("only_predicted"), True)
    title_fallback = _bool_param(body.get("title_fallback"), False)
    consolidate_work = _bool_param(body.get("consolidate_work"), False)
    max_cost = _float_param(body.get("max_cost"), "max_cost", None, 0.0, 1e9)
    max_cost = max_cost if max_cost and max_cost > 0 else None
    plan = await run_in_threadpool(_contract_items_plan, entries) if test == "T-002" else None
    est = _estimate(test, entries, workers, consolidate_work, (plan or {}).get("process", 0))
    name = re.sub(r"\s+", " ", str(body.get("name") or "")).strip()[:80] or f"{src.get('label')} {len(entries)}건"

    run = TestRun(_new_run_id("batch"), "batch", _ROOT)
    run.dir.mkdir(parents=True, exist_ok=True)
    try:
        run.name, run.test, run.source, run.n_sets = name, test, src["id"], len(entries)
        run.gold = gold_path is not None
        run.gold_path = str(gold_path) if gold_path else None
        run.estimate = {"sec": est["est_sec"], "krw": est["est_krw"], "by_cell": est["by_cell"]}
        run.file_to_set = {os.path.basename((e.contract if test == "T-001" else e.work) or ""): e.set_id
                           for e in entries}
        run.cfg_kwargs = {"workers": workers, "scoring_set": "eval23", "test": test, "frame_schedule": "v4",
                          "consolidate_work": consolidate_work, "only_predicted": only_predicted,
                          "title_fallback": title_fallback, "tau": tau, "gold": run.gold_path,
                          "max_cost_krw": max_cost}
        run.config = {"test": test, "source": src["id"], "source_label": src.get("label"), "cells": cells,
                      "n": n, "sample": sample, "seed": seed, "ids_count": len(ids) if ids else None,
                      "ids": ids[:50] if ids else None, "workers": workers, "tau": tau,
                      "only_predicted": only_predicted, "title_fallback": title_fallback,
                      "consolidate_work": consolidate_work, "max_cost": max_cost,
                      "gold": os.path.basename(run.gold_path) if run.gold_path else None,
                      "missing_ids": missing[:50] if missing else None}
        if plan is not None:
            run.config["contract_items"] = plan
        write_manifest(entries, run.dir / "manifest.jsonl")
        _register(run)
    except BaseException:
        shutil.rmtree(run.dir, ignore_errors=True)
        raise
    _start(run, entries)
    return {"run_id": run.run_id, "n_sets": len(entries), "estimate": {"sec": est["est_sec"], "krw": est["est_krw"]},
            "missing_ids": missing}


# ---------------------------------------------------------------------------
# API — 폴더로 시험 (시험 PC 의 폴더를 제자리에서 읽는다 · web/console_folder.py)
#   GET  /fs/roots · /fs/list?path=   폴더 고르기
#   POST /folder/scan                 불러오기(등록 확인) — 짝 맞추기·정답 대조·예상치 (실행하지 않는다)
#   POST /folder/start                확인한 그대로 실행 (둘 다 = T-001 과 T-002 를 동시에, 실행 2건)
# ---------------------------------------------------------------------------
FOLDER_SCAN_TTL_SEC = 1800
FOLDER_SOURCE_LABEL = "폴더로 시험"
_FOLDER_SCANS: Dict[str, Dict[str, Any]] = {}
_FOLDER_LOCK = threading.Lock()


def _folder(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except CF.FolderError as e:
        raise ConsoleError(e.status, e.code, e.message, **e.extra)


@router.get(API + "/fs/roots")
def api_fs_roots(user: str = Depends(_api_user)):
    roots = CF.data_roots()
    return {"enabled": bool(roots), "roots": CF.roots_view(roots), "env": CF.ENV_ROOTS}


@router.get(API + "/fs/list")
def api_fs_list(path: str = "", user: str = Depends(_api_user)):
    roots = CF.data_roots()
    if not str(path).strip():
        if not roots:
            raise ConsoleError(403, "folders_off", "이 서버에서는 폴더로 시험을 쓸 수 없습니다 (허용된 위치가 없습니다)")
        path = CF.display_path(roots[0])
    return _folder(CF.list_dir, path, roots, CONTRACT_EXT, WORK_EXT)


def _folder_workers(v: Any, test: str) -> int:
    if isinstance(v, dict):
        v = v.get(test)
    return int(_int_param(v, "workers", DEFAULT_WORKERS[test], 1, MAX_WORKERS))


def _folder_scan(body: Dict[str, Any]) -> Dict[str, Any]:
    from module.evaluation.manifest import ManifestEntry
    from module.evaluation.v4.keys23 import KEYS
    t = _test_param(body.get("test")) or ""
    tests = TESTS if t in ("BOTH", "ALL", "T-001+T-002") else ((t,) if t in TESTS else ())
    if not tests:
        raise ConsoleError(400, "bad_test", "시험(Te-1 · Te-2 · 둘 다)을 고르세요")
    roots = CF.data_roots()
    recursive = _bool_param(body.get("recursive"), True)
    include_unkeyed = _bool_param(body.get("include_unkeyed"), False)
    cdirs = [_folder(CF.resolve_allowed, p, roots) for p in _list_param(body.get("contract_dirs"))]
    wdirs = [_folder(CF.resolve_allowed, p, roots) for p in _list_param(body.get("work_dirs"))]
    if "T-001" in tests and not cdirs:
        raise ConsoleError(400, "no_contract_dir", "계약서 폴더를 고르세요 (Te-1 은 계약서가 필요합니다)")
    if "T-002" in tests and not wdirs:
        raise ConsoleError(400, "no_work_dir", "저작물 폴더를 고르세요 (Te-2 는 저작물 파일이 필요합니다)")
    t0 = time.perf_counter()
    contracts = list(_folder(lambda: list(CF.walk_files(cdirs, CONTRACT_EXT, recursive)))) if cdirs else []
    works = list(_folder(lambda: list(CF.walk_files(wdirs, WORK_EXT, recursive)))) if wdirs else []
    answers_in = body.get("answers") if isinstance(body.get("answers"), dict) else {}
    consolidate_work = _bool_param(body.get("consolidate_work"), False)
    views: Dict[str, Any] = {}
    internal: Dict[str, Any] = {}
    for test in tests:
        spec = answers_in.get(test) if isinstance(answers_in.get(test), dict) else {}
        mode = str(spec.get("mode") or "builtin")
        if mode not in ("builtin", "file", "none"):
            raise ConsoleError(400, "bad_answer_mode", "정답은 기본 정답 · 정답 파일 · 정답 없이 가운데 하나입니다")
        records: Dict[str, Dict[str, Any]] = {}
        aview: Dict[str, Any] = {"mode": mode}
        gold_path: Optional[Path] = None
        answer_file: Optional[Path] = None
        if mode == "builtin":
            gp = _SINGLE_GOLD.get(test)
            if gp and gp.is_file():
                records, gold_path = CF.load_gold_jsonl(gp, test), gp
            aview.update(label="기본 정답", file=gp.name if gp else None, n=len(records), available=gold_path is not None)
        elif mode == "file":
            answer_file = _folder(CF.resolve_allowed, spec.get("path"), roots, want="file")
            a = _folder(CF.load_answers, answer_file, KEYS, test)
            records = a.records
            aview.update(label="정답 파일", file=answer_file.name, path=CF.display_path(answer_file), n=len(records),
                         format=a.fmt, problems=a.problems[:20], n_problems=len(a.problems),
                         missing_columns=a.missing_columns, skipped_rows=a.skipped_rows)
        else:
            aview.update(label="정답 없이 (채점 안 함)", n=0)
        ts = CF.build_sets(test, contracts, works, records, known=_known_ids(test), include_unkeyed=include_unkeyed,
                           media_of=_media_of_path, manifest_entry=ManifestEntry)
        w = _folder_workers(body.get("workers"), test)
        plan = None
        if test == "T-002":                          # 1~16번 — Te-1 결과를 쓸 세트 · 계약서를 직접 처리할 세트
            pids = {e.set_id for e in internal["T-001"]["entries"]} if "T-001" in internal else None
            plan = _contract_items_plan(ts.entries, pids)
        est = _estimate(test, ts.entries, w, consolidate_work and test == "T-002", (plan or {}).get("process", 0))
        media: Dict[str, int] = {}
        for e in ts.entries:
            media[e.media] = media.get(e.media, 0) + 1
        sample = [{"set_id": e.set_id, "contract": os.path.basename(e.contract) if e.contract else None,
                   "work": os.path.basename(e.work) if e.work else None, "media": e.media,
                   "title": (e.extra or {}).get("title"), "scored": e.set_id in records}
                  for e in ts.entries[:CF.SAMPLE_ROWS]]
        views[test] = {
            "test": test, "n_sets": len(ts.entries), "n_scored": ts.scored, "media": media, "by_cell": est["by_cell"],
            "n_contract_files": ts.n_contract_files, "n_work_files": ts.n_work_files, "paired": ts.paired,
            "answer": aview,
            "missing": ts.missing[:CF.MAX_EXAMPLES], "n_missing": len(ts.missing),
            "duplicates": ts.duplicates[:CF.MAX_EXAMPLES], "n_duplicates": len(ts.duplicates),
            "unkeyed": ts.unkeyed[:CF.MAX_EXAMPLES], "n_unkeyed": len(ts.unkeyed), "include_unkeyed": include_unkeyed,
            "no_id": ts.no_id[:CF.MAX_EXAMPLES], "n_no_id": len(ts.no_id),
            "sample": sample,
            "estimate": {"sec": est["est_sec"], "krw": est["est_krw"], "workers": est["workers"]},
            "contract_items": plan,
        }
        internal[test] = {"entries": ts.entries, "gold_path": gold_path, "answer_file": answer_file,
                          "records": records if mode == "file" else None, "mode": mode, "contract_items": plan}
    # 시험 대상 = 고른 시험들의 세트 ID 합집합 (Te-1 4,000 + Te-2 4,000 → 어문 200 은 Te-1 에만 있어 4,200)
    item_media: Dict[str, str] = {}
    for test in tests:
        for e in internal[test]["entries"]:
            if e.set_id not in item_media or test == "T-002":          # 저작물 파일이 있는 쪽(Te-2)의 매체를 따른다
                item_media[e.set_id] = e.media or "text"
    items_media: Dict[str, int] = {}
    for m in item_media.values():
        items_media[m] = items_media.get(m, 0) + 1
    scan_id = secrets.token_hex(8)
    folders = {"contract_dirs": [CF.display_path(d) for d in cdirs], "work_dirs": [CF.display_path(d) for d in wdirs],
               "recursive": recursive}
    out = {"scan_id": scan_id, "test": "BOTH" if len(tests) > 1 else tests[0], "tests": views, "order": list(tests),
           "folders": folders, "n_contract_files": len(contracts), "n_work_files": len(works),
           "scan_sec": round(time.perf_counter() - t0, 1), "consolidate_work": consolidate_work,
           "n_items": len(item_media), "items_media": items_media, "expires_in": FOLDER_SCAN_TTL_SEC}
    now = time.time()
    with _FOLDER_LOCK:
        for k in [k for k, v in _FOLDER_SCANS.items() if now - v["at"] > FOLDER_SCAN_TTL_SEC]:
            _FOLDER_SCANS.pop(k, None)
        _FOLDER_SCANS[scan_id] = {"at": now, "view": out, "internal": internal, "body": body}
    return out


@router.post(API + "/folder/scan")
async def api_folder_scan(request: Request, user: str = Depends(_api_user)):
    try:
        body = await request.json()
    except Exception:
        raise ConsoleError(400, "bad_request", "요청 본문(JSON)을 읽을 수 없습니다")
    if not isinstance(body, dict):
        raise ConsoleError(400, "bad_request", "요청 본문은 JSON 객체여야 합니다")
    return await run_in_threadpool(_folder_scan, body)          # 경로는 바꾸지 않는다 (scrub 하지 않음)


PEEK_ROWS = 300


def _gold_filled(rec: Dict[str, Any], names: Iterable[str]) -> int:
    ks = rec.get("keys") or {}
    return sum(1 for n in names if isinstance(ks.get(n), dict) and ks[n].get("state") in ("value", "empty"))


def _folder_peek(body: Dict[str, Any]) -> Dict[str, Any]:
    """간소화 콘솔 — 경로 하나의 미리보기 (계약서 폴더 · 저작물 폴더 · 정답 파일). 세트는 만들지 않는다 (그건 scan)."""
    kind = str(body.get("kind") or "")
    roots = CF.data_roots()
    t0 = time.perf_counter()
    if kind in ("contract", "work"):
        d = _folder(CF.resolve_allowed, body.get("path"), roots)
        files = list(_folder(lambda: list(CF.walk_files([d], CONTRACT_EXT if kind == "contract" else WORK_EXT, True))))
        by_ext: Dict[str, int] = {}
        by_media: Dict[str, int] = {}
        ids: set = set()
        items: List[Dict[str, Any]] = []
        no_id = 0
        for p in files:
            ext = p.suffix.lower().lstrip(".")
            by_ext[ext] = by_ext.get(ext, 0) + 1
            media = (_media_of_path(str(p)) or "text") if kind == "work" else None
            if media:
                by_media[media] = by_media.get(media, 0) + 1
            sid = CF.choose_id(CF.id_groups(p))
            if sid is None:
                no_id += 1
            else:
                ids.add(sid)
            if len(items) < PEEK_ROWS:
                try:
                    size = p.stat().st_size
                except OSError:
                    size = None
                rel = os.path.relpath(p.parent, d)
                items.append({"name": p.name, "dir": "" if rel == "." else rel.replace(os.sep, "\\"), "set_id": sid,
                              "size": size, "media": media})
        return {"kind": kind, "path": CF.display_path(d), "n_files": len(files), "n_ids": len(ids), "n_no_id": no_id,
                "by_ext": dict(sorted(by_ext.items(), key=lambda kv: -kv[1])), "by_media": by_media,
                "items": items, "truncated": len(files) > len(items), "sec": round(time.perf_counter() - t0, 1)}
    if kind == "answers":
        from module.evaluation.v4.keys23 import KEYS
        f = _folder(CF.resolve_allowed, body.get("path"), roots, want="file")
        test = _test_param(body.get("test"))
        test = test if test in TESTS else None              # 시험을 주면 scan 과 같게 — 그 시험 레코드 + 시험 표시 없는 레코드
        try:
            a = _folder(CF.load_answers, f, KEYS, test)
        except (ValueError, UnicodeDecodeError) as e:          # 깨진 JSON · 인코딩
            raise ConsoleError(400, "answer_unreadable", f"정답 파일을 읽을 수 없습니다 — {type(e).__name__}: {str(e)[:160]}")
        k1 = [k.name for k in KEYS if k.test == "T-001"]
        k2 = [k.name for k in KEYS if k.test == "T-002"]
        by_test: Dict[str, int] = {}
        n_k1 = n_k2 = 0
        items = []
        for sid in sorted(a.records):
            rec = a.records[sid]
            t = rec.get("test") or "-"
            by_test[t] = by_test.get(t, 0) + 1
            f1, f2 = _gold_filled(rec, k1), _gold_filled(rec, k2)
            n_k1 += f1 > 0
            n_k2 += f2 > 0
            if len(items) < PEEK_ROWS:
                items.append({"set_id": sid, "test": rec.get("test"), "work_title": CF._gold_value(rec, "work_title"),
                              "work_type": CF._gold_value(rec, "work_type"), "file_name": CF._gold_value(rec, "file_name"),
                              "n_contract_keys": f1, "n_work_keys": f2})
        return {"kind": kind, "path": CF.display_path(f), "file": f.name, "format": a.fmt, "n": len(a.records), "test": test,
                "by_test": by_test, "n_with_contract_keys": n_k1, "n_with_work_keys": n_k2,
                "n_contract_keys": len(k1), "n_work_keys": len(k2),
                "problems": a.problems[:20], "n_problems": len(a.problems), "missing_columns": a.missing_columns,
                "skipped_rows": a.skipped_rows, "items": items, "truncated": len(a.records) > len(items),
                "sec": round(time.perf_counter() - t0, 1)}
    raise ConsoleError(400, "bad_kind", "미리보기 종류는 contract · work · answers 가운데 하나입니다")


@router.post(API + "/folder/peek")
async def api_folder_peek(request: Request, user: str = Depends(_api_user)):
    try:
        body = await request.json()
    except Exception:
        raise ConsoleError(400, "bad_request", "요청 본문(JSON)을 읽을 수 없습니다")
    if not isinstance(body, dict):
        raise ConsoleError(400, "bad_request", "요청 본문은 JSON 객체여야 합니다")
    return await run_in_threadpool(_folder_peek, body)


def _make_folder_run(test: str, sc: Dict[str, Any], view: Dict[str, Any], scan: Dict[str, Any],
                     opts: Dict[str, Any]) -> TestRun:
    from module.evaluation.manifest import write_manifest
    entries = sc["entries"]
    tv = view["tests"][test]
    run = TestRun(_new_run_id("batch"), "batch", _ROOT)
    run.dir.mkdir(parents=True, exist_ok=True)
    try:
        gold_path: Optional[Path] = None
        if sc["mode"] == "file" and sc.get("records") is not None:
            gold_path = run.dir / "answers.jsonl"          # 실행 폴더에 정답을 고정해 둔다 (원본도 함께)
            with gold_path.open("w", encoding="utf-8") as f:
                for sid in sorted(sc["records"]):
                    f.write(json.dumps(sc["records"][sid], ensure_ascii=False) + "\n")
            if sc.get("answer_file"):
                shutil.copy2(sc["answer_file"], run.dir / ("answers_source" + Path(sc["answer_file"]).suffix.lower()))
        elif sc["mode"] == "builtin" and sc.get("gold_path"):
            gold_path = Path(sc["gold_path"])
        w = _folder_workers(opts.get("workers"), test)
        cw = bool(opts.get("consolidate_work")) and test == "T-002"
        plan = sc.get("contract_items")
        est = _estimate(test, entries, w, cw, (plan or {}).get("process", 0))
        base = opts.get("name") or ""
        label = CF.test_label(test)
        run.name = (f"{base} · {label}" if base and len(view["order"]) > 1 else base) or f"{label} {FOLDER_SOURCE_LABEL} {len(entries)}건"
        run.test, run.source, run.n_sets = test, "folder", len(entries)
        run.gold = gold_path is not None
        run.gold_path = str(gold_path) if gold_path else None
        run.estimate = {"sec": est["est_sec"], "krw": est["est_krw"], "by_cell": est["by_cell"]}
        run.file_to_set = {os.path.basename((e.contract if test == "T-001" else e.work) or ""): e.set_id for e in entries}
        run.cfg_kwargs = {"workers": w, "scoring_set": "eval23", "test": test, "frame_schedule": "v4",
                          "consolidate_work": cw, "only_predicted": opts["only_predicted"], "title_fallback": False,
                          "tau": opts["tau"], "gold": run.gold_path, "max_cost_krw": opts.get("max_cost")}
        if test == "T-002":
            run.cfg_kwargs["contract_items"] = "te1"
        ans = tv["answer"]
        run.config = {"test": test, "source": "folder", "source_label": FOLDER_SOURCE_LABEL, **view["folders"],
                      "answer": {k: ans.get(k) for k in ("mode", "label", "file", "n")},
                      "workers": w, "tau": opts["tau"], "only_predicted": opts["only_predicted"], "consolidate_work": cw,
                      "max_cost": opts.get("max_cost"), "gold": os.path.basename(run.gold_path) if run.gold_path else None,
                      "n_missing": tv["n_missing"], "n_duplicates": tv["n_duplicates"], "n_unkeyed": tv["n_unkeyed"],
                      "include_unkeyed": tv["include_unkeyed"], "media": tv.get("media") or {},
                      "n_items": view.get("n_items"), "items_media": view.get("items_media") or {}}
        if plan is not None:
            run.config["contract_items"] = plan          # 불러오기 때의 어림 — 실제 출처는 채점 결과의 contract_items
        write_manifest(entries, run.dir / "manifest.jsonl")
        _write_json_atomic(run.dir / "registration.json", {"version": VERSION, "registered_at": _iso(),
                                                            "folders": view["folders"], **tv})
    except BaseException:
        shutil.rmtree(run.dir, ignore_errors=True)
        raise
    return run


@router.post(API + "/folder/start")
async def api_folder_start(request: Request, user: str = Depends(_api_user)):
    try:
        body = await request.json()
    except Exception:
        raise ConsoleError(400, "bad_request", "요청 본문(JSON)을 읽을 수 없습니다")
    if not isinstance(body, dict):
        raise ConsoleError(400, "bad_request", "요청 본문은 JSON 객체여야 합니다")
    sid = str(body.get("scan_id") or "")
    with _FOLDER_LOCK:
        scan = _FOLDER_SCANS.get(sid)
    if scan is None or time.time() - scan["at"] > FOLDER_SCAN_TTL_SEC:
        raise ConsoleError(404, "scan_expired", "불러오기 결과가 없거나 만료되었습니다 — 불러오기를 다시 하세요")
    active = _active_run("batch")
    if active is not None:
        raise _busy_error(active)
    max_cost = _float_param(body.get("max_cost"), "max_cost", None, 0.0, 1e9)
    opts = {"workers": body.get("workers"), "tau": _float_param(body.get("tau"), "tau", DEFAULT_TAU, 0.0, 1.0),
            "only_predicted": _bool_param(body.get("only_predicted"), True),
            "consolidate_work": _bool_param(body.get("consolidate_work"), scan["view"].get("consolidate_work", False)),
            "max_cost": max_cost if max_cost and max_cost > 0 else None,
            "name": re.sub(r"\s+", " ", str(body.get("name") or "")).strip()[:80]}
    view = scan["view"]
    order = [t for t in view["order"] if scan["internal"][t]["entries"]]
    if not order:
        raise ConsoleError(400, "empty", "실행할 세트가 없습니다 — 불러오기 결과를 보세요")
    runs: List[Tuple[TestRun, list]] = []
    for test in order:
        run = await run_in_threadpool(_make_folder_run, test, scan["internal"][test], view, scan, opts)
        runs.append((run, scan["internal"][test]["entries"]))
    # 둘 다: 두 시험을 동시에 돌린다. Te-2 의 1~16번은 함께 도는 Te-1 의 결과를 쓴다 — Te-2 세트가 먼저 끝나면
    # Te-1 이 그 계약서를 처리할 때까지 기다린다(_te1_lookup). 실행마다 스레드·작업자·로그·비용 상한이 따로이고,
    # 한쪽을 취소해도 다른 쪽은 계속한다 (Te-1 이 그 세트를 처리하지 못하고 끝나면 Te-2 가 계약서를 직접 처리한다).
    if len(runs) > 1:
        for r, _ in runs:
            r.config["parallel"] = [{"run_id": o.run_id, "test": o.test} for o, _ in runs if o is not r]
    try:
        _register(*(r for r, _ in runs))
    except BaseException:
        for r, _ in runs:
            shutil.rmtree(r.dir, ignore_errors=True)
        raise
    for r, e in runs:
        _start(r, e)
    with _FOLDER_LOCK:
        _FOLDER_SCANS.pop(sid, None)
    return {"run_id": runs[0][0].run_id, "parallel": len(runs) > 1,
            "runs": [{"run_id": r.run_id, "test": r.test, "n_sets": r.n_sets,
                      "estimate": {"sec": (r.estimate or {}).get("sec"), "krw": (r.estimate or {}).get("krw")}}
                     for r, _ in runs]}


# ---------------------------------------------------------------------------
# API — 실행 조회 · 진행 · 취소
# ---------------------------------------------------------------------------
@router.get(API + "/runs")
def api_runs(user: str = Depends(_api_user)):
    with _RUNS_LOCK:
        runs = list(_RUNS.values())
    rows = [r.row() for r in runs]
    rows.sort(key=lambda r: (r.get("created_at") or "", r.get("run_id") or ""), reverse=True)
    return scrub_json(rows)


@router.get(API + "/runs/{run_id}")
def api_run(run_id: str, user: str = Depends(_api_user)):
    run = _get_run(run_id)
    d = run.detail()
    if run.kind == "batch":
        d["remaining"] = _remaining(run)
    return d


@router.get(API + "/runs/{run_id}/stream")
async def api_run_stream(run_id: str, request: Request, user: str = Depends(_api_user)):
    run = _get_run(run_id)
    after = request.query_params.get("after")
    if after is None:
        after = request.headers.get("last-event-id")
    start = _int_param(after, "after", 0, 0, 2 ** 62)

    async def gen():
        seq = start
        idle = 0
        while True:
            evs = run.events_after(seq)
            for e in evs:
                seq = e["seq"]
                yield f"id: {seq}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n"
            if evs:
                idle = 0
            else:
                idle += 1
                if idle % 10 == 0:                       # ~5초마다 연결 유지 주석
                    yield ": keep-alive\n\n"
            if run.state in TERMINAL_STATES and not run.events_after(seq):
                yield f"data: {json.dumps({'type': 'closed', 'state': run.state, 'seq': seq})}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})


@router.post(API + "/runs/{run_id}/cancel")
def api_run_cancel(run_id: str, user: str = Depends(_api_user)):
    run = _get_run(run_id)
    if run.state in ACTIVE_STATES:
        run.cancel_requested = True
        runner = run.runner
        if runner is not None:
            runner.cancel()
        with LC.bind(run_id=run.run_id, set_id=None):
            logger.warning("취소 요청 — 진행 중인 세트만 마치고 멈춥니다", extra={"component": "시스템"})
    return {"ok": True, "state": run.state}


RETRYABLE_STATES = ("done", "cancelled", "error", "interrupted")


def _remaining(run: TestRun) -> Dict[str, int]:
    """매니페스트 기준 — 성공 · 실패(마지막 시도가 실패) · 미실행(결과 없음) 세트 수."""
    idx = _run_index(run)
    failed = sum(1 for s in idx["order"] if (idx["results"].get(s) or {}).get("ok") is False)
    pending = sum(1 for s in idx["order"] if s not in idx["results"])
    return {"ok": len(idx["order"]) - failed - pending, "failed": failed, "pending": pending,
            "retries": len(run.config.get("retries") or [])}


def _unfinished_csv(run: TestRun) -> bytes:
    """실패·미실행 세트 목록 (엑셀에서 바로 열리게 UTF-8 BOM)."""
    import csv as _csv
    idx = _run_index(run)
    buf = io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["set_id", "시험", "셀", "상태", "시도 횟수", "파일", "마지막 오류"])
    label = CF.test_label(run.test)
    for sid in idx["order"]:
        res = idx["results"].get(sid)
        if res is not None and res["ok"]:
            continue
        base = idx["sets"][sid]
        w.writerow([sid, label, base.get("cell") or "", "실패" if res is not None else "미실행",
                    0 if res is None else res.get("attempts", 1), base.get("file_name") or "",
                    "" if res is None else (res.get("error") or "")])
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def _retry_entries(run: TestRun) -> Optional[list]:
    """다시 실행할 매니페스트(전체) — 실행기가 성공한 세트는 건너뛴다(resume). 다시 할 것이 없으면 None."""
    rem = _remaining(run)
    if not rem["failed"] and not rem["pending"]:
        return None
    from module.evaluation.manifest import load_manifest
    return load_manifest(run.dir / "manifest.jsonl")


@router.post(API + "/runs/{run_id}/retry")
async def api_run_retry(run_id: str, request: Request, user: str = Depends(_api_user)):
    """실패·미실행 세트만 같은 실행 폴더에서 다시 돌리고 전체를 다시 채점한다.

    둘 다(동시 실행)로 함께 시작한 실행도 끝났고 다시 할 세트가 있으면 같이 다시 시작한다 (include_parallel, 기본 켬).
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    body = body if isinstance(body, dict) else {}
    run = _get_run(run_id)
    if run.kind != "batch":
        raise ConsoleError(400, "not_batch", "일괄 시험만 다시 실행할 수 있습니다")
    if run.state not in RETRYABLE_STATES:
        raise ConsoleError(409, "not_finished", "실행이 끝난 뒤에 다시 실행할 수 있습니다")
    targets = [run]
    if _bool_param(body.get("include_parallel"), True):
        for p in run.config.get("parallel") or []:
            with _RUNS_LOCK:
                o = _RUNS.get(str(p.get("run_id")))
            if o is not None and o.kind == "batch" and o.state in RETRYABLE_STATES:
                targets.append(o)
    plan: List[Tuple[TestRun, list, Dict[str, int]]] = []
    for r in targets:
        entries = await run_in_threadpool(_retry_entries, r)
        if entries:
            plan.append((r, entries, _remaining(r)))
    if not plan or plan[0][0] is not run:
        raise ConsoleError(400, "nothing_to_retry", "다시 실행할 세트가 없습니다 — 모든 세트가 성공했습니다")
    with _RUNS_LOCK:
        active = _active_run("batch")
        if active is not None:
            raise _busy_error(active)
        for r, _, rem in plan:
            prev_cost = float((r.progress or {}).get("cost_krw_total") or 0.0)
            r.config["cost_prev_krw"] = round(float(r.config.get("cost_prev_krw") or 0.0) + prev_cost, 2)
            r.config["retries"] = (r.config.get("retries") or []) + [
                {"at": _iso(), "failed": rem["failed"], "pending": rem["pending"], "state_before": r.state}]
            r.cancel_requested = r.cost_exceeded = False
            r.error = r.summary = r.final = r.finished_at = None
            r.progress = {"i": 0, "total": rem["failed"] + rem["pending"], "failed": 0, "cost_krw_total": 0.0,
                          "last_set": None}
            r._index = None
            r.state = "queued"
    for r, entries, rem in plan:
        r.save()
        with LC.bind(run_id=r.run_id, set_id=None):
            logger.info("다시 실행 — 실패 %d · 미실행 %d세트 (성공한 %d세트는 건너뜀, 끝나면 전체를 다시 채점)",
                        rem["failed"], rem["pending"], rem["ok"], extra={"component": "시스템"})
        _start(r, entries)
    return {"ok": True, "runs": [{"run_id": r.run_id, "test": r.test, "failed": rem["failed"], "pending": rem["pending"]}
                                 for r, _, rem in plan]}


@router.get(API + "/runs/{run_id}/metrics")
def api_run_metrics(run_id: str, user: str = Depends(_api_user)):
    run = _get_run(run_id)
    m = _read_json(run.out_dir / "test_metrics.json")
    if not isinstance(m, dict):
        raise ConsoleError(404, "not_scored", "아직 채점되지 않았습니다")
    return _metrics_view(m)


@router.get(API + "/runs/{run_id}/sets")
def api_run_sets(run_id: str, request: Request, user: str = Depends(_api_user)):
    run = _get_run(run_id)
    q = request.query_params
    cell = (q.get("cell") or "").strip() or None
    key = (q.get("key") or "").strip() or None
    verdicts = {v.upper() for v in _list_param(q.get("verdict"))}
    failed = _bool_param(q.get("failed"), False)
    text = (q.get("q") or "").strip().lower()
    offset = _int_param(q.get("offset"), "offset", 0, 0, 10 ** 9)
    limit = _int_param(q.get("limit"), "limit", 100, 1, 5000)
    items = _set_items(run)
    out = []
    for it in items:
        if cell and it.get("cell") != cell:
            continue
        if failed and it.get("ok") is not False:
            continue
        if key and verdicts:
            if ((it["keys"].get(key) or {}).get("verdict") or "") not in verdicts:
                continue
        elif key:
            if (it["keys"].get(key) or {}).get("verdict") is None:
                continue
        elif verdicts:
            if not any((e.get("verdict") or "") in verdicts for e in it["keys"].values()):
                continue
        if text:
            hay = " ".join(str(x or "") for x in (it.get("set_id"), it.get("title"), it.get("file_name"))).lower()
            if text not in hay:
                continue
        out.append(it)
    return scrub_json({"total": len(out), "offset": offset, "limit": limit, "items": out[offset: offset + limit]})


@router.get(API + "/runs/{run_id}/sets/{set_id}")
def api_run_set(run_id: str, set_id: str, user: str = Depends(_api_user)):
    return _set_detail(_get_run(run_id), set_id)


@router.get(API + "/runs/{run_id}/sets/{set_id}/locate")
def api_run_set_locate(run_id: str, set_id: str, request: Request, user: str = Depends(_api_user)):
    """원본 위치 표시 — 계약서 쪽 미리보기 위에 값이 읽힌 자리 (web/console_locate.py, 표시 전용)."""
    from . import console_locate as CL
    run = _get_run(run_id)
    rec = _load_record(run, set_id)
    with LC.bind(run_id=run.run_id, set_id=str(set_id)):
        try:
            return CL.locate_set(run, set_id, rec, refresh=request.query_params.get("refresh"))
        except Exception as e:           # 위치 표시가 실패해도 상세 보기는 그대로 — 사유만 돌려준다
            logger.warning("원본 위치 계산 실패 %s/%s: %s", run_id, set_id, LC.scrub_secret(str(e))[:200],
                           exc_info=True, extra={"component": "측정"})
            return scrub_json({"available": False, "reason": f"위치 계산 중 오류 — {type(e).__name__}",
                               "pages": [], "marks": [], "not_found": [], "targets": [],
                               "engine": {"ocr": "none", "model": None, "pages_ocr": 0, "seconds": 0.0,
                                          "cost_krw": 0.0, "cached": False}})


@router.get(API + "/runs/{run_id}/sets/{set_id}/preview/work.jpg")
def api_preview_work(run_id: str, set_id: str, user: str = Depends(_api_user)):
    run = _get_run(run_id)
    rec = _load_record(run, set_id)
    meta = rec.get("_meta") or {}
    kind = _work_kind((meta.get("files") or {}).get("work"), meta.get("media"))
    wdir = _leg_dir(run, (meta.get("request_ids") or {}).get("work"))
    src: Optional[Path] = None
    if kind == "pptx" and wdir is not None and (wdir / "pptx_slide1.png").is_file():
        src = wdir / "pptx_slide1.png"
    elif kind == "video":
        frames = _frames(wdir)
        src = (wdir / "keyframes" / frames[0]["name"]) if (wdir is not None and frames) else None
    elif kind == "image":
        src = _source_file(run, rec, "work")
    if src is None or not src.is_file():
        raise _not_found("미리보기할 저작물 그림이 없습니다")
    if src != _record_file(rec, "work") and not _inside(src, run.dir):
        raise _not_found("미리보기할 저작물 그림이 없습니다")
    try:
        out = _downscale_jpeg(src, run.dir / "cache" / f"{_cache_name(set_id)}_work.jpg")
    except Exception as e:
        logger.warning("미리보기 생성 실패 %s/%s: %s", run_id, set_id, e)
        raise _not_found("미리보기를 만들 수 없습니다")
    return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@router.get(API + "/runs/{run_id}/sets/{set_id}/preview/frames/{name}")
def api_preview_frame(run_id: str, set_id: str, name: str, user: str = Depends(_api_user)):
    run = _get_run(run_id)
    rec = _load_record(run, set_id)
    if not _FRAME_RX.match(name or ""):
        raise _not_found("프레임을 찾을 수 없습니다")
    wdir = _leg_dir(run, ((rec.get("_meta") or {}).get("request_ids") or {}).get("work"))
    if wdir is None or name not in {f["name"] for f in _frames(wdir)}:
        raise _not_found("프레임을 찾을 수 없습니다")
    p = wdir / "keyframes" / name
    if not _inside(p, run.dir):
        raise _not_found("프레임을 찾을 수 없습니다")
    media = "image/png" if name.lower().endswith(".png") else "image/jpeg"
    return FileResponse(p, media_type=media, headers={"Cache-Control": "no-cache"})


@router.get(API + "/runs/{run_id}/sets/{set_id}/preview/contract/{page}.png")
def api_preview_contract(run_id: str, set_id: str, page: str, user: str = Depends(_api_user)):
    run = _get_run(run_id)
    rec = _load_record(run, set_id)
    if not re.fullmatch(r"\d{1,3}", page or ""):
        raise _not_found("계약서 쪽을 찾을 수 없습니다")
    pg = int(page)
    src = _source_file(run, rec, "contract")
    if src is None or pg < 1 or pg > min(_page_count(src), MAX_CONTRACT_PAGES):
        raise _not_found("계약서 쪽을 찾을 수 없습니다")
    try:
        out = _contract_png(src, pg, run.dir / "cache" / f"{_cache_name(set_id)}_contract_p{pg}.png")
    except Exception as e:
        logger.warning("계약서 미리보기 실패 %s/%s p%d: %s", run_id, set_id, pg, e)
        raise _not_found("계약서 쪽을 그릴 수 없습니다")
    return FileResponse(out, media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.get(API + "/runs/{run_id}/download/{name}")
def api_download(run_id: str, name: str, user: str = Depends(_api_user)):
    run = _get_run(run_id)
    if name == UNFINISHED_CSV:
        return Response(_unfinished_csv(run), media_type="text/csv; charset=utf-8",
                        headers=_attachment(f"{run.run_id}_{name}"))
    if name not in DOWNLOADS:
        raise _not_found("내려받을 수 없는 파일입니다")
    fname = f"{run.run_id}_{name}"
    if name == "run.zip":
        return StreamingResponse(_zip_stream(_run_zip_items(run)), media_type="application/zip",
                                 headers=_attachment(fname))
    if name == PREDICTIONS_ZIP:
        if not run.n_predictions():
            raise _not_found("속성정보 파일이 아직 없습니다")
        return StreamingResponse(_zip_stream(_predictions_zip_items(run)), media_type="application/zip",
                                 headers=_attachment(fname))
    path = run.out_dir / name
    if not path.is_file():
        raise _not_found("파일이 아직 없습니다")
    if name == "test_metrics.json":
        m = _read_json(path)
        if not isinstance(m, dict):
            raise _not_found("파일을 읽을 수 없습니다")
        return Response(json.dumps(_metrics_view(m), ensure_ascii=False, indent=1).encode("utf-8"),
                        media_type="application/json", headers=_attachment(fname))
    if name == "per_set.jsonl":
        return FileResponse(path, media_type="application/x-ndjson", headers=_attachment(fname))
    data = _scrubbed_bytes(path)
    if data is None:
        return FileResponse(path, headers=_attachment(fname))
    media = {"report.md": "text/markdown; charset=utf-8", "test.log": "text/plain; charset=utf-8",
             "test_predictions.jsonl": "application/x-ndjson"}.get(name, "application/octet-stream")
    return Response(data, media_type=media, headers=_attachment(fname))


@router.get(API + "/runs/{run_id}/sets/{set_id}/download")
def api_set_download(run_id: str, set_id: str, user: str = Depends(_api_user)):
    run = _get_run(run_id)
    rec = _load_record(run, set_id)
    meta = rec.get("_meta") or {}
    idx = _run_index(run)
    items: List[Tuple[str, Any]] = [
        ("prediction.json", json.dumps(scrub_json(rec), ensure_ascii=False, indent=1).encode("utf-8"))]
    vrows = idx["verdicts"].get(str(set_id))
    if vrows:
        items.append(("verdicts.json", json.dumps(scrub_json(list(vrows.values())), ensure_ascii=False,
                                                  indent=1).encode("utf-8")))
    for leg, rid in (meta.get("request_ids") or {}).items():
        d = _leg_dir(run, rid)
        if d is None:
            continue
        for f in sorted(d.rglob("*")):
            if f.is_file() and _inside(f, run.dir):
                items.append((f"{re.sub(r'[^\w-]', '_', str(leg))}/{f.relative_to(d).as_posix()}", f))
    safe_sid = re.sub(r"[^\w-]", "_", str(set_id))
    return StreamingResponse(_zip_stream(items), media_type="application/zip",
                             headers=_attachment(f"{run.run_id}_set_{safe_sid}.zip"))
