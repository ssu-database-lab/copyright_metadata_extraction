"""실행 로그 공통 도구 — 세트 묶기 · 구성요소 태그 · 비밀값 가리기 · 한 줄 형식 (M3 콘솔 로그 보강, 2026-10-11).

무엇을 하는가
  · 세트 묶기: `bind(run_id=…, set_id=…)` 로 지금 스레드(정확히는 contextvars 문맥)를 시험 실행·세트에 묶는다.
    파이프라인 안의 작업 스레드로 넘길 때는 `submit(executor, fn, …)` 로 문맥을 복사해 넘긴다 — 그래야 LLM ∥ NER
    스레드에서 남긴 로그도 같은 세트로 붙는다. 묶이지 않은 기록은 `[-]` 로 적힌다.
  · 구성요소: 로거 이름(과 기록의 extra `component`)으로 시스템·업로드·문맥·OCR·LLM·NER·통합·VLM·API·측정·채점 중 하나를 붙인다.
  · 비밀값 가리기 `scrub_secret()`: URL 쿼리 문자열을 떼고 sk-… · sk-ws-… · AIza… · AQ.… · Bearer … · key=… · api_key=… 와
    이름에 KEY/SECRET/TOKEN/PASSWORD 가 든 환경 변수의 값을 가린다. 콘솔이 받는 모든 줄(test.log · SSE · 시스템 로그)이 거친다.
  · 한 줄 형식 `ConsoleLineFormatter`: `HH:MM:SS LEVEL [set_id] [구성요소] 메시지`.

이 모듈은 표준 라이브러리만 쓴다 — 실행기(module.evaluation.batch_runner)·파이프라인(web.pipeline)·콘솔(web.console_api)이
모두 가볍게 import 한다. 앱은 `module.log_context`, 평가 CLI 는 `api.module.log_context` 로 올리므로 모듈 사본이 둘 생길 수
있다 — 그래도 같은 묶음을 보도록 ContextVar 는 logging 모듈 속성에 하나만 둔다.
"""
from __future__ import annotations

import contextvars
import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, NamedTuple, Optional, Tuple
from urllib.parse import urlsplit

__all__ = [
    "Binding", "COMPONENTS", "ConsoleLineFormatter", "bind", "bind_set", "component_for", "current", "describe",
    "is_runner_logger", "parse_line", "scrub_secret", "submit", "wrap",
]

COMPONENTS: Tuple[str, ...] = ("시스템", "업로드", "문맥", "OCR", "LLM", "NER", "통합", "VLM", "API", "측정", "채점")


# ---------------------------------------------------------------------------
# 세트 묶기 (contextvars)
# ---------------------------------------------------------------------------
class Binding(NamedTuple):
    run_id: Optional[str]
    set_id: Optional[str]


_ATTR = "_copyright_console_binding"
_CTX: "contextvars.ContextVar[Optional[Binding]]" = getattr(logging, _ATTR, None)  # type: ignore[assignment]
if _CTX is None:                                     # 프로세스에 하나 — import 경로가 달라도 같은 변수
    _CTX = contextvars.ContextVar("console_binding", default=None)
    setattr(logging, _ATTR, _CTX)

_KEEP = object()


def current() -> Optional[Binding]:
    """지금 문맥의 묶음 (없으면 None)."""
    return _CTX.get()


@contextmanager
def bind(run_id: Any = _KEEP, set_id: Any = _KEEP) -> Iterator[Binding]:
    """이 블록 안에서 남기는 로그를 실행·세트에 묶는다. 넘기지 않은 쪽은 바깥 묶음을 이어받는다."""
    cur = _CTX.get()
    rid = (cur.run_id if cur else None) if run_id is _KEEP else (str(run_id) if run_id is not None else None)
    sid = (cur.set_id if cur else None) if set_id is _KEEP else (str(set_id) if set_id is not None else None)
    b = Binding(rid, sid)
    token = _CTX.set(b)
    try:
        yield b
    finally:
        _CTX.reset(token)


def bind_set(set_id: Optional[str]):
    """실행 묶음은 그대로 두고 세트만 묶는다."""
    return bind(set_id=set_id)


def submit(executor, fn: Callable, *args, **kwargs):
    """executor.submit 과 같되 지금 문맥(세트 묶음)을 복사해 작업 스레드로 넘긴다 — 호출마다 새 사본."""
    ctx = contextvars.copy_context()
    return executor.submit(ctx.run, fn, *args, **kwargs)


def wrap(fn: Callable) -> Callable:
    """threading.Thread(target=…) 용 — 지금 문맥을 복사해 두었다가 그 안에서 fn 을 돌린다 (한 번만 부를 것)."""
    ctx = contextvars.copy_context()

    def runner(*a, **kw):
        return ctx.run(fn, *a, **kw)
    return runner


# ---------------------------------------------------------------------------
# 비밀값 가리기
# ---------------------------------------------------------------------------
MASK = "***"
_SECRET_ENV_NAME = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|PASSWD", re.I)
_NOT_SECRET_VALUES = {"true", "false", "none", "null", "not-set", "not-needed"}
_ENV_CACHE: Dict[str, Any] = {"sig": None, "rx": None}

_URL_QUERY_RX = re.compile(r"(?i)\b((?:https?|wss?)://[^\s?#\"'<>]+)\?[^\s#\"'<>]*")
_PATH_QUERY_RX = re.compile(r"((?<![\w/])/[\w.~%/\-]*)\?([^\s\"'<>]*=[^\s\"'<>]*)")
_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=\-]{6,}"), "Bearer " + MASK),
    (re.compile(r"\bsk-ws-[A-Za-z0-9_\-]{6,}"), "sk-ws-" + MASK),
    (re.compile(r"\bsk-(?!ws-)[A-Za-z0-9_\-]{8,}"), "sk-" + MASK),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{10,}"), "AIza" + MASK),
    (re.compile(r"(?<![\w.])AQ\.[A-Za-z0-9_\-.]{8,}"), "AQ." + MASK),
    # key=… · api_key: … · "token": "…" · 'password': '…'
    (re.compile(r"(?i)(?<![A-Za-z0-9_])((?:x-)?api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|id[_-]?token"
                r"|auth[_-]?token|token|client[_-]?secret|secret|password|passwd|key|sig|signature)"
                r"(\s*[=:]\s*['\"]?|['\"]\s*:\s*['\"]?)([^\s&'\",;{}()\[\]<>]+)"), r"\1\2" + MASK),
)


def _env_secret_rx() -> Optional[re.Pattern]:
    vals = []
    for k, v in os.environ.items():
        if not v or len(v) < 8 or not _SECRET_ENV_NAME.search(k):
            continue
        v = v.strip()
        if len(v) >= 8 and v.lower() not in _NOT_SECRET_VALUES:
            vals.append(v)
    sig = tuple(sorted(set(vals)))
    if sig != _ENV_CACHE["sig"]:
        _ENV_CACHE["sig"] = sig
        _ENV_CACHE["rx"] = (re.compile("|".join(re.escape(v) for v in sorted(sig, key=len, reverse=True)))
                            if sig else None)
    return _ENV_CACHE["rx"]


def scrub_secret(text: Any) -> Any:
    """로그 한 줄에서 비밀값을 가린다 (문자열이 아니면 그대로)."""
    if not isinstance(text, str) or not text:
        return text
    rx = _env_secret_rx()
    if rx is not None:
        text = rx.sub(MASK, text)
    text = _URL_QUERY_RX.sub(r"\1", text)
    text = _PATH_QUERY_RX.sub(r"\1", text)
    for pat, rep in _PATTERNS:
        text = pat.sub(rep, text)
    return text


# ---------------------------------------------------------------------------
# 구성요소
# ---------------------------------------------------------------------------
_COMPONENT_PREFIXES: Tuple[Tuple[str, str], ...] = tuple(sorted((
    ("module.ocr", "OCR"),
    ("module.llm_extraction.party_rules", "문맥"),
    ("module.llm_extraction", "LLM"),
    ("module.ner", "NER"),
    ("module.consolidator", "통합"),
    ("module.clip_extraction.contract_context", "문맥"),
    ("module.clip_extraction.contract_inheritance", "통합"),
    ("module.clip_extraction.technical_metadata", "측정"),
    ("module.clip_extraction.router", "업로드"),
    ("module.clip_extraction", "VLM"),
    ("module.evaluation.v4", "채점"),
    ("module.evaluation", "시스템"),
    ("module", "시스템"),
    ("web", "시스템"),
    ("httpx", "API"), ("httpcore", "API"), ("openai", "API"), ("urllib3", "API"), ("requests", "API"),
    ("uvicorn", "시스템"),
), key=lambda p: -len(p[0])))

_STAGE_COMPONENT = {"upload": "업로드", "route": "업로드", "ocr": "OCR", "consolidate": "통합",
                    "consolidate_done": "통합", "extract_done": "LLM"}


def is_runner_logger(name: str) -> bool:
    """BatchRunner 의 실행 전용 로거(test.log FileHandler 가 달린 자식 로거)인가."""
    return ".batch_runner.run." in (name or "")


def _runner_component(msg: str) -> str:
    if msg.startswith(("평가", "gold 미지정")):
        return "채점"
    if "문맥" in msg:
        return "문맥"
    return "시스템"


def component_for(name: str, msg: str = "") -> str:
    """로거 이름 → 구성요소. 'api.' 로 시작하는 이름(평가 CLI 의 import 경로)은 떼고 본다."""
    n = name or ""
    if n.startswith("api."):
        n = n[4:]
    if is_runner_logger(n):
        return _runner_component(msg)
    for prefix, comp in _COMPONENT_PREFIXES:
        if n == prefix or n.startswith(prefix + "."):
            return comp
    return "시스템"


# ---------------------------------------------------------------------------
# 기록 → 화면용 필드
# ---------------------------------------------------------------------------
_SET_PREFIX_RX = re.compile(r"^\[([^\]\s]{1,80})\] ")
_OPENAI_RETRY_RX = re.compile(r"^Retrying request to (\S+?) in ([\d.]+) seconds")
_LEGACY_STAGES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"^Pipeline setup: "), "upload"),
    (re.compile(r"^Router: .*modality=\w+"), "route"),
    (re.compile(r"^OCR complete: \d+ chars"), "ocr"),
    (re.compile(r"^OCR returned empty text"), "ocr"),
    (re.compile(r"^OCR (?:failed|error)"), "ocr"),
    (re.compile(r"^Consolidation start"), "consolidate"),
    (re.compile(r"^Consolidation (?:complete|failed)"), "consolidate_done"),
    (re.compile(r"^(?:VLM extraction complete via|Video extraction via) "), "extract_done"),
)


def _host_path(url: Any) -> str:
    try:
        u = urlsplit(str(url))
        return f"{u.hostname or ''}{u.path or '/'}"
    except Exception:
        return str(url).split("?", 1)[0]


def _message(record: logging.LogRecord) -> str:
    name = record.name or ""
    args = record.args if isinstance(record.args, tuple) else ()
    if name.startswith("httpx") and str(record.msg).startswith("HTTP Request") and len(args) >= 4:
        method, url, _ver, status = args[:4]
        reason = args[4] if len(args) > 4 else ""
        return f"HTTP 요청 {method} {_host_path(url)} → {status} {reason}".rstrip()
    try:
        msg = record.getMessage()
    except Exception:                                   # 형식 인자가 맞지 않는 기록도 줄은 남긴다
        msg = str(record.msg)
    if name.startswith("openai"):
        m = _OPENAI_RETRY_RX.match(msg)
        if m:
            return f"API 재시도 — {m.group(1).split('?', 1)[0]} · {float(m.group(2)):.1f}초 후"
    return msg


def legacy_stage(msg: str) -> Optional[str]:
    for rx, stage in _LEGACY_STAGES:
        if rx.search(msg):
            return stage
    return None


def describe(record: logging.LogRecord) -> Dict[str, Any]:
    """기록 → {time, level, run_id, set_id, component, stage, logger, message}. 기록에 붙여 두어 핸들러끼리 다시 계산하지 않는다.

    묶음은 **기록을 남긴 스레드**에서 읽어야 하므로 이 함수는 emit/format 안(같은 스레드)에서 부른다.
    """
    d = record.__dict__.get("_console_desc")
    if d is not None:
        return d
    msg = _message(record)
    b = _CTX.get()
    run_id = record.__dict__.get("console_run") or (b.run_id if b else None)
    set_id = record.__dict__.get("console_set")
    if set_id is None and is_runner_logger(record.name):
        m = _SET_PREFIX_RX.match(msg)
        if m:
            set_id, msg = m.group(1), msg[m.end():]
    elif set_id is not None and msg.startswith(f"[{set_id}] "):
        msg = msg[len(str(set_id)) + 3:]
    if set_id is None and b is not None:
        set_id = b.set_id
    stage = record.__dict__.get("stage")
    if stage is None and (record.name or "").endswith("web.pipeline"):
        stage = legacy_stage(msg)
    comp = record.__dict__.get("component")
    if comp not in COMPONENTS:
        comp = None
    if comp is None and stage and (record.name or "").endswith("web.pipeline"):
        comp = "VLM" if stage == "extract_done" and re.search(r"VLM|Video|영상|프레임", msg) else _STAGE_COMPONENT.get(stage)
    if comp is None:
        comp = component_for(record.name, msg)
    d = {
        "time": datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
        "ts": round(record.created, 3),
        "level": record.levelname,
        "run_id": run_id,
        "set_id": str(set_id) if set_id is not None else None,
        "component": comp,
        "stage": stage,
        "logger": record.name,
        "message": scrub_secret(msg),
    }
    record.__dict__["_console_desc"] = d
    return d


class ConsoleLineFormatter(logging.Formatter):
    """`HH:MM:SS LEVEL [set_id] [구성요소] 메시지` (+ 예외가 있으면 그 아래 traceback) — 비밀값은 가린다."""

    def __init__(self, traceback: bool = True):
        super().__init__()
        self.traceback = traceback

    def format(self, record: logging.LogRecord) -> str:
        d = describe(record)
        line = f"{d['time']} {d['level']} [{d['set_id'] or '-'}] [{d['component']}] {d['message']}"
        if self.traceback:
            if record.exc_info and not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                line += "\n" + record.exc_text
            if record.stack_info:
                line += "\n" + self.formatStack(record.stack_info)
        return scrub_secret(line)


# ---------------------------------------------------------------------------
# test.log 한 줄 읽기 (새 형식 · 예전 형식 모두) — 재시작 뒤 다시 흘려보낼 때
# ---------------------------------------------------------------------------
_NEW_LINE_RX = re.compile(r"^(\d\d:\d\d:\d\d) (DEBUG|INFO|WARNING|ERROR|CRITICAL) \[([^\]]*)\] \[([^\]]*)\] (.*)$", re.S)
_OLD_LINE_RX = re.compile(r"^(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)(?:[,.]\d+)? (DEBUG|INFO|WARNING|ERROR|CRITICAL) (.*)$",
                          re.S)
_OLD_STAGE_COMPONENT = ((re.compile(r"^단계 (?:업로드|경로)"), "업로드"), (re.compile(r"^단계 텍스트·OCR"), "OCR"),
                        (re.compile(r"^단계 .*통합"), "통합"), (re.compile(r"^단계 추출 완료 —"), "VLM"))


def parse_line(line: str) -> Dict[str, Any]:
    """test.log 한 줄 → {time, level, set_id, component, message} (알 수 없는 줄은 level·component 가 None)."""
    m = _NEW_LINE_RX.match(line or "")
    if m:
        sid = m.group(3)
        return {"time": m.group(1), "level": m.group(2), "set_id": None if sid in ("", "-") else sid,
                "component": m.group(4) or None, "message": m.group(5)}
    m = _OLD_LINE_RX.match(line or "")
    if m:
        msg = m.group(4)
        sid = None
        p = _SET_PREFIX_RX.match(msg)
        if p:
            sid, msg = p.group(1), msg[p.end():]
        comp = None
        for rx, c in _OLD_STAGE_COMPONENT:
            if rx.search(msg):
                comp = c
                break
        return {"time": m.group(2), "level": m.group(3), "set_id": sid,
                "component": comp or _runner_component(msg), "message": msg}
    return {"time": None, "level": None, "set_id": None, "component": None, "message": line}
