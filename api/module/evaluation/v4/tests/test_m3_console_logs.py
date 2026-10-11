"""시험 콘솔 로그 보강 테스트 — 줄 형식·구성요소·세트 묶기(스레드 넘김)·비밀값 가리기·시스템 로그·통합 검증 상세
(docs/v4/M3_console_spec_addendum_logs.md §A·§C·§D).

모델·API 호출 없음: 파이프라인은 실제 PipelineOrchestrator 를 쓰되 LLM 은 가짜, NER 는 실제 ner_predict 에 실패하는 가짜 적재기를
끼우고(그래서 '모델 미적재 — 정규표현식 대체' 경로를 그대로 탄다), 입력은 OCR 공급자를 부르지 않는 .txt 다. 통합 검증 상세는
디스크에 남은 실제 T-001 실행(api/web/test_runs/20261011_082121_batch_0bb0)을 임시 폴더로 복사해 읽는다(없으면 건너뜀).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

import pytest

from .test_m3_console import (CA, ROOT, Console, _httpx_testclient_compat, console, logged_in,  # noqa: F401
                              no_internal, sse_events, wait_done)

from module import log_context as LC                         # noqa: E402

REAL_RUN = ROOT / "api/web/test_runs/20261011_082121_batch_0bb0"
LINE_RX = re.compile(r"^(\d\d:\d\d:\d\d) (DEBUG|INFO|WARNING|ERROR|CRITICAL) \[([^\]]+)\] \[([^\]]+)\] (.+)$", re.S)
needs_real_run = pytest.mark.skipif(not (REAL_RUN / "run.json").is_file(), reason="실제 T-001 실행 폴더 없음")


def _record(name: str, msg: str, level: int = logging.INFO, args=(), **extra) -> logging.LogRecord:
    rec = logging.getLogger(name).makeRecord(name, level, __file__, 1, msg, args, None)
    rec.__dict__.update(extra)
    return rec


# ---------------------------------------------------------------------------
# §A1 줄 형식 · 구성요소
# ---------------------------------------------------------------------------
def test_line_format_set_and_component():
    fmt = LC.ConsoleLineFormatter()
    with LC.bind(run_id="r1", set_id="13387752"):
        line = fmt.format(_record("module.ocr.alibaba_ocr", "OCR 쪽 완료 — %d자", args=(120,)))
    m = LINE_RX.match(line)
    assert m and m.group(2) == "INFO" and m.group(3) == "13387752" and m.group(4) == "OCR"
    assert m.group(5) == "OCR 쪽 완료 — 120자"
    # 묶이지 않은 기록은 [-]
    assert LINE_RX.match(fmt.format(_record("web.console_api", "x"))).group(3) == "-"
    # 실행기 줄 — 메시지 앞의 '[set_id] ' 는 세트 칸으로 옮긴다 (중복 없음)
    line = fmt.format(_record("module.evaluation.batch_runner.run.out.ab12cd", "[90000001] 시작 — test=T-002"))
    assert LINE_RX.match(line).group(3) == "90000001" and line.endswith("[90000001] [시스템] 시작 — test=T-002")
    assert "[-] [채점] 평가 완료" in fmt.format(_record("module.evaluation.batch_runner.run.o.x", "평가 완료 — T-001"))
    assert "[문맥] 계약서 문맥" in fmt.format(_record("api.module.evaluation.batch_runner.run.o.x", "[w0] 계약서 문맥 — 제목"))
    # extra component 가 이긴다 · 알 수 없는 값은 무시
    assert "[측정]" in fmt.format(_record("web.pipeline", "파일 측정", component="측정"))
    assert "[시스템]" in fmt.format(_record("web.pipeline", "기타", component="없는구성요소"))
    # 예외는 아래 줄에 traceback
    try:
        raise ValueError("boom")
    except ValueError:
        rec = logging.getLogger("module.ner.ner_system").makeRecord("module.ner.ner_system", logging.ERROR, __file__, 1,
                                                                    "실패", (), sys.exc_info())
    out = fmt.format(rec)
    assert out.splitlines()[0].endswith("[-] [NER] 실패") and "ValueError: boom" in out


@pytest.mark.parametrize("name,comp", [
    ("web.console_api", "시스템"), ("web.app", "시스템"), ("uvicorn.error", "시스템"),
    ("web.pipeline", "시스템"), ("module.clip_extraction.router", "업로드"),
    ("module.clip_extraction.contract_context", "문맥"), ("module.llm_extraction.party_rules", "문맥"),
    ("module.ocr.universal_ocr", "OCR"), ("api.module.ocr.alibaba_ocr", "OCR"),
    ("module.llm_extraction.models.cloud_extractor", "LLM"), ("module.ner.ner_system", "NER"),
    ("api.module.ner.ner_system", "NER"), ("module.consolidator.consolidation_agent", "통합"),
    ("module.clip_extraction.vlm.extractor", "VLM"), ("module.clip_extraction.keyframes", "VLM"),
    ("httpx", "API"), ("openai._base_client", "API"), ("module.clip_extraction.technical_metadata", "측정"),
    ("module.evaluation.v4.evaluator", "채점"), ("api.module.evaluation.v4.embed_judge", "채점"),
    ("something.else", "시스템"),
])
def test_component_for_logger_names(name, comp):
    assert LC.component_for(name) == comp
    assert comp in LC.COMPONENTS


def test_pipeline_stage_and_legacy_stage_detection():
    d = LC.describe(_record("web.pipeline", "텍스트·OCR 완료 — 3,508자", component="OCR", stage="ocr"))
    assert d["stage"] == "ocr" and d["component"] == "OCR"
    d = LC.describe(_record("web.pipeline", "OCR complete: 1234 chars"))           # 예전 영어 문구
    assert d["stage"] == "ocr" and d["component"] == "OCR"
    d = LC.describe(_record("web.pipeline", "Video extraction via fake-omni: 3 frames"))
    assert d["stage"] == "extract_done" and d["component"] == "VLM"
    assert LC.describe(_record("module.ocr.x", "OCR complete: 1 chars"))["stage"] is None   # web.pipeline 만


def test_parse_line_new_and_old_formats():
    assert LC.parse_line("19:37:41 WARNING [13387752] [NER] 모델 미적재") == {
        "time": "19:37:41", "level": "WARNING", "set_id": "13387752", "component": "NER", "message": "모델 미적재"}
    f = LC.parse_line("19:37:41 INFO [-] [시스템] 배치 시작")
    assert f["set_id"] is None and f["component"] == "시스템"
    f = LC.parse_line("2026-10-11 08:21:21,271 INFO [13333561] 단계 텍스트·OCR 완료 — 3,562자")
    assert (f["level"], f["set_id"], f["component"]) == ("INFO", "13333561", "OCR")
    f = LC.parse_line("2026-10-11 08:23:56,484 INFO 평가 완료 — T-001")
    assert (f["set_id"], f["component"]) == (None, "채점")
    assert LC.parse_line("Traceback (most recent call last):")["level"] is None


# ---------------------------------------------------------------------------
# §A3 세트 묶기 — 스레드 넘김
# ---------------------------------------------------------------------------
def test_binding_crosses_threads_only_with_copied_context():
    seen: Dict[str, Any] = {}
    with LC.bind(run_id="R", set_id=None):
        with LC.bind_set("S1"):
            with ThreadPoolExecutor(max_workers=2) as ex:
                seen["copied"] = LC.submit(ex, LC.current).result()
                seen["plain"] = ex.submit(LC.current).result()
            t = threading.Thread(target=LC.wrap(lambda: seen.setdefault("thread", LC.current())))
            t.start()
            t.join()
        seen["after_set"] = LC.current()
    seen["outside"] = LC.current()
    assert seen["copied"] == LC.Binding("R", "S1") and seen["thread"] == LC.Binding("R", "S1")
    assert seen["plain"] is None                                     # 문맥을 복사하지 않으면 묶음이 없다
    assert seen["after_set"] == LC.Binding("R", None) and seen["outside"] is None


def test_batch_runner_binds_set_per_worker(tmp_path):
    """BatchRunner 작업 스레드(workers>1)에서 오케스트레이터 안의 기록이 자기 세트로 묶인다."""
    from module.evaluation.batch_runner import BatchRunner, RunConfig
    from module.evaluation.manifest import ManifestEntry
    seen = {}

    class Orch:
        def run(self, file_bytes, filename, **kw):
            b = LC.current()
            with ThreadPoolExecutor(max_workers=1) as ex:          # 파이프라인 안의 작업 스레드
                inner = LC.submit(ex, LC.current).result()
            seen[filename] = (b, inner)
            return {"success": True, "request_id": "rid-" + filename, "test_prediction": {}}

    files = []
    for i in range(4):
        f = tmp_path / f"c{i}.pdf"
        f.write_bytes(b"%PDF fake")
        files.append(f)
    entries = [ManifestEntry(set_id=f"s{i}", contract=str(f), media="text", test="T-001") for i, f in enumerate(files)]
    runner = BatchRunner(Orch(), RunConfig(workers=3, scoring_set="eval23", test="T-001"), tmp_path / "out")
    with LC.bind(run_id="RUN"):
        runner.run(entries, {})
    runner.close_log()
    for i, f in enumerate(files):
        b, inner = seen[f.name]
        assert b == LC.Binding("RUN", f"s{i}") and inner == b
    log = (tmp_path / "out" / "test.log").read_text(encoding="utf-8")
    for line in log.splitlines():
        assert LINE_RX.match(line), line
    assert "INFO [s0] [시스템] 시작 — test=T-001" in log and "INFO [-] [시스템] 배치 시작" in log


# ---------------------------------------------------------------------------
# §D 비밀값 가리기
# ---------------------------------------------------------------------------
def test_scrub_secret_patterns(monkeypatch):
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "tok-very-secret-value-42")
    monkeypatch.setenv("PLAIN_DISPLAY_SETTING", "plain-value-shown-123")
    cases = [
        "GET https://generativelanguage.googleapis.com/v1beta/models/gemma:generateContent?key=AIzaSyD-abcdefghijklmnop12345 200",
        "POST /v1beta/openai/chat/completions?key=AIzaSyD-abcdefghijklmnop12345&alt=sse",
        "api key sk-0123456789abcdefABCDEF in error",
        "workspace sk-ws-" + "A1b2C3d4" * 13 + " rejected",
        "oauth AQ.Zz0TESTKxyzABCDEFGHIJKLMNOP_qrstu-vwx used",
        "Authorization: Bearer abc.def-ghi_jkl123456",
        "api_key=raw-key-value-123 and token: ttt-999999 and {'password': 'hunter2!!'}",
        'json {"api_key": "zzzz-1234-yyyy"}',
        "env value tok-very-secret-value-42 leaked",
        "standalone AIzaSyD-abcdefghijklmnop12345",
    ]
    for c in cases:
        out = LC.scrub_secret(c)
        for secret in ("AIzaSyD-abcdefghijklmnop12345", "sk-0123456789abcdefABCDEF", "A1b2C3d4A1b2C3d4", "AQ.Zz0TESTKxyz",
                       "abc.def-ghi_jkl123456", "raw-key-value-123", "ttt-999999", "hunter2!!", "zzzz-1234-yyyy",
                       "tok-very-secret-value-42", "alt=sse"):
            assert secret not in out, (c, out)
    assert LC.scrub_secret(cases[0]) == ("GET https://generativelanguage.googleapis.com/v1beta/models/gemma:generateContent"
                                         " 200")
    assert "sk-ws-***" in LC.scrub_secret(cases[3]) and "Bearer ***" in LC.scrub_secret(cases[5])
    # 가리지 말아야 할 것
    keep = "모델 qwen3.5-122b-a10b · 토큰 입력 4,210 / 출력 812 · prompt_tokens=10 · plain-value-shown-123 · KeyError: 'x'"
    assert LC.scrub_secret(keep) == keep
    assert LC.scrub_secret(None) is None and LC.scrub_secret("") == ""
    # 콘솔의 scrub_text(화면·내려받기)도 비밀값을 가린다 — 예전 test.log 를 내려받거나 다시 흘려보낼 때
    assert "AIza" not in CA.scrub_text(cases[0]) and "sk-ws-***" in CA.scrub_text(cases[3] + " (v4)")


def test_httpx_record_is_rewritten_without_query():
    rec = _record("httpx", 'HTTP Request: %s %s "%s %d %s"',
                  args=("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions?key=AIzaSyXYZ"
                                "0123456789abcdef", "HTTP/1.1", 200, "OK"))
    d = LC.describe(rec)
    assert d["component"] == "API"
    assert d["message"] == "HTTP 요청 POST generativelanguage.googleapis.com/v1beta/openai/chat/completions → 200 OK"
    rec = _record("openai._base_client", "Retrying request to %s in %f seconds", args=("/chat/completions?x=1", 0.4321))
    assert LC.describe(rec)["message"] == "API 재시도 — /chat/completions · 0.4초 후"


# ---------------------------------------------------------------------------
# §A4·A5 수집 · 시스템 로그
# ---------------------------------------------------------------------------
def test_capture_does_not_leak_info_to_root_handlers(console):
    console.init()                                             # 수집 설치 (idempotent)
    got = []

    class Root(logging.Handler):
        def emit(self, record):
            got.append((record.name, record.levelname))

    h = Root()
    f = CA._CAPTURE["filter"]
    h.addFilter(f)                                             # 서버의 루트 핸들러와 같은 필터
    root = logging.getLogger()
    root.addHandler(h)
    try:
        assert logging.getLogger("httpx").isEnabledFor(logging.INFO)
        assert logging.getLogger("module.ocr.x").isEnabledFor(logging.INFO)
        logging.getLogger("httpx").info("HTTP Request: GET https://x.example/?key=1 \"HTTP/1.1 200 OK\"")
        logging.getLogger("module.ocr.x").info("info 줄")
        logging.getLogger("module.ocr.x").warning("warning 줄")
        rl = logging.getLogger("module.evaluation.batch_runner.run.leak.abc123")
        rl.setLevel(logging.INFO)                              # 실행기 로거처럼 직접 INFO 로 정한 로거는 예전처럼 통과
        rl.info("[s] 시작 — 실행기 줄")
    finally:
        root.removeHandler(h)
    if CA._CAPTURE["lowered"].get("httpx", 0) > logging.INFO:
        assert ("httpx", "INFO") not in got and ("module.ocr.x", "INFO") not in got
    assert ("module.ocr.x", "WARNING") in got
    assert ("module.evaluation.batch_runner.run.leak.abc123", "INFO") in got


def test_syslog_endpoint_auth_filters_after(console):
    c = console.client()
    console.set_password()
    assert c.get("/api/test/syslog").status_code == 401
    c = logged_in(console)
    CA._SYSLOG.clear()
    lg = logging.getLogger("module.ocr.alibaba_ocr")
    lg.info("첫 줄 OCR")
    with LC.bind(run_id="20990101_000000_batch_abcd", set_id="S9"):
        logging.getLogger("module.ner.ner_system").warning("모델 미적재 — 정규표현식 대체 (모델 0개 · 정규식 3개)")
        logging.getLogger("module.ner.ner_system").error("NER 모델 적재 실패 — RuntimeError: x")
        logging.getLogger("httpx").info('HTTP Request: %s %s "%s %d %s"', "GET",
                                        "https://api.example.com/v1/models?key=AIzaSECRET0123456789", "HTTP/1.1",
                                        401, "Unauthorized")
    logging.getLogger("module.evaluation.v4.evaluator").info("v4 평가 완료")

    def get(qs: str = "") -> Dict[str, Any]:
        out = c.get("/api/test/syslog" + qs).json()
        # TestClient 도 httpx 라 요청마다 'HTTP 요청 GET testserver/…' 줄이 쌓인다 — 이 테스트가 남긴 줄만 본다
        out["mine"] = [it for it in out.get("items", []) if "testserver" not in it["message"]]
        return out

    r = get()
    assert set(r) >= {"items", "last_seq", "first_seq", "more", "components"}
    items = r["mine"]
    assert [it["message"] for it in items][:2] == ["첫 줄 OCR", "모델 미적재 — 정규표현식 대체 (모델 0개 · 정규식 3개)"]
    assert len(items) == 5
    for it in r["items"]:
        assert set(it) >= {"seq", "ts", "level", "component", "logger", "run_id", "set_id", "message"}
    by_msg = {it["message"]: it for it in items}
    ner = by_msg["NER 모델 적재 실패 — RuntimeError: x"]
    assert ner["level"] == "ERROR" and ner["component"] == "NER" and ner["set_id"] == "S9" and \
        ner["run_id"] == "20990101_000000_batch_abcd"
    http = next(it for it in items if it["component"] == "API")
    assert http["message"] == "HTTP 요청 GET api.example.com/v1/models → 401 Unauthorized"
    assert "AIza" not in json.dumps(r) and "key=" not in json.dumps(r)
    ev = next(it for it in items if it["component"] == "채점")
    assert ev["message"] == "v1.0 평가 완료" and ev["logger"] == "module.evaluation.evaluator"   # 내부 이름 지움
    no_internal(r)

    assert [it["level"] for it in get("?level=WARNING")["mine"]] == ["WARNING", "ERROR"]
    assert [it["level"] for it in get("?level=ERROR,INFO")["mine"]] == ["INFO", "ERROR", "INFO", "INFO"]
    assert {it["component"] for it in get("?component=NER,API")["mine"]} == {"NER", "API"}
    assert len(get("?q=정규표현식")["items"]) == 1
    assert len(get("?set_id=S9")["items"]) == 3
    assert len(get("?run_id=20990101_000000_batch_abcd&component=NER")["items"]) == 2
    assert c.get("/api/test/syslog?level=LOUD").status_code == 400
    assert c.get("/api/test/syslog?component=모름").status_code == 400

    # after — 다음 것만, limit 으로 잘리면 more=True 와 이어 받을 last_seq (구성요소 NER·OCR·채점 만 — 요청 줄 제외)
    f = "&component=OCR,NER,채점"
    first = items[0]["seq"]
    r2 = get(f"?after={first}&limit=1{f}")
    assert [it["message"] for it in r2["items"]] == ["모델 미적재 — 정규표현식 대체 (모델 0개 · 정규식 3개)"]
    assert r2["more"] is True and r2["last_seq"] == r2["items"][0]["seq"]
    r3 = get(f"?after={r2['last_seq']}{f}")
    assert [it["component"] for it in r3["items"]] == ["NER", "채점"] and r3["more"] is False
    assert get(f"?after={r3['last_seq']}{f}")["items"] == []
    tail = get(f"?limit=2{f}")                                    # 처음 열 때는 조건에 맞는 마지막 limit 건
    assert [it["component"] for it in tail["items"]] == ["NER", "채점"] and tail["more"] is True


# ---------------------------------------------------------------------------
# 실제 파이프라인(가짜 LLM · NER 적재 실패) → 실행 로그
# ---------------------------------------------------------------------------
@pytest.fixture
def light_llm_extraction():
    """module.llm_extraction 패키지 __init__(torch) 없이 party_rules·schemas 만 올린다 (test_m2_pipeline 과 같은 방식)."""
    import module  # noqa: F401
    added = False
    if "module.llm_extraction" not in sys.modules:
        stub = types.ModuleType("module.llm_extraction")
        stub.__path__ = [str(ROOT / "api" / "module" / "llm_extraction")]  # type: ignore[attr-defined]
        sys.modules["module.llm_extraction"] = stub
        added = True
    before = set(sys.modules)
    yield
    if added:
        for name in [n for n in sys.modules if n == "module.llm_extraction" or n.startswith("module.llm_extraction.")]:
            if name not in before or name == "module.llm_extraction":
                sys.modules.pop(name, None)


class FakeLLM:
    def extract_metadata_from_text(self, text, document_type, document_name, model_name):
        logging.getLogger("module.llm_extraction.models.cloud_extractor").info(
            "LLM 응답 — 모델 fake · 0.0초 · 토큰 입력 10 / 출력 5 · 재시도 0회")
        return {"success": True, "metadata": {"work_title": "운수 자판기", "author": "김은혜"}, "confidence": 0.8,
                "model_used": "fake-llm", "extraction_time": 0.01}


def _failing_loader(model_path, verbose=True):
    raise RuntimeError("Error(s) in loading state_dict for BertCrfForNER:\n\tsize mismatch for "
                       "bert.embeddings.word_embeddings.weight: copying a param with shape torch.Size([32000, 1024]) "
                       "from checkpoint, the shape in current model is torch.Size([119547, 768]).")


def test_pipeline_logs_reach_run_with_set_and_ner_failure(console, tmp_path, monkeypatch, light_llm_extraction):
    """실제 PipelineOrchestrator 문서 경로: LLM ∥ NER 작업 스레드의 기록도 세트에 붙고, NER 적재 실패가 ERROR,
    정규식 대체가 WARNING 으로 실행 로그(test.log · SSE)에 남으며 응답에 ner_mode 등이 실린다. API 호출 없음(.txt 입력)."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-ws-TESTSECRETKEY0123456789abcdef")
    # OCR 모듈은 import 때 저장소 .env 를 os.environ 에 올린다(module.env_loader) — 시험 프로세스에 실제 키·콘솔 비밀번호가
    # 들어오지 않게 빈 모듈로 막는다 (이미 올라와 있으면 그대로)
    if "module.env_loader" not in sys.modules:
        monkeypatch.setitem(sys.modules, "module.env_loader", types.ModuleType("module.env_loader"))
    from module.ner import ner_system as ns
    from web.pipeline import PipelineOrchestrator
    from module.ocr.alibaba_ocr import AlibabaCloudOCRProvider
    monkeypatch.setattr(AlibabaCloudOCRProvider, "process_image",
                        lambda self, p: (_ for _ in ()).throw(AssertionError("OCR API 를 부르면 안 된다")))
    model_dir = tmp_path / "fake-ner-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ns, "get_model_path", lambda name=None: model_dir)
    monkeypatch.setattr(ns, "load_model_and_tokenizer", _failing_loader)
    ns.clear_ner_model_cache()

    def orch_factory(out_dir):
        runs = Path(out_dir) / "runs"
        return PipelineOrchestrator(llm_processor=FakeLLM(), ner_predict_fn=ns.ner_predict,
                                    available_ner_models={"klue-roberta-large": {"name": "fake/ner-model",
                                                                                 "display_name": "가짜 NER"}},
                                    upload_dir=runs / "uploads", results_dir=runs / "results")

    console.set_password()
    CA.init_console_api(templates=None, runs_root=console.runs, orchestrator_factory=orch_factory,
                        sources=console.sources(), single_gold={"T-001": None, "T-002": None}, env_path=console.env)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    app = FastAPI()
    app.include_router(CA.router)
    c = TestClient(app)
    assert c.post("/console/login", data={"user": "admin", "password": "testpass123", "next": "/console"},
                  follow_redirects=False).status_code == 303

    # T-001 단건은 계약서 확장자만 받는다 — 실행기를 직접 꾸며 .txt 계약서를 넣는다
    from module.evaluation.manifest import ManifestEntry, write_manifest
    txt = tmp_path / "13387752 계약서.txt"
    txt.write_text("저작재산권 이용허락 계약서\n제호: 운수 자판기\n저작자: 김은혜\n연락처 010-1234-5678\n"
                   "이용자: 한국저작권위원회 (서명)\n", encoding="utf-8")
    run = CA.TestRun(CA._new_run_id("single"), "single", console.runs)
    run.dir.mkdir(parents=True)
    run.test, run.set_id, run.n_sets, run.source = "T-001", "13387752", 1, "upload"
    run.file_to_set = {txt.name: "13387752"}
    run.cfg_kwargs = {"workers": 1, "scoring_set": "eval23", "test": "T-001", "consolidate": False, "gold": None}
    entry = ManifestEntry(set_id="13387752", contract=str(txt), media="text", test="T-001")
    write_manifest([entry], run.dir / "manifest.jsonl")
    CA._register(run)
    CA._start(run, [entry])
    d = wait_done(c, run.run_id)
    ns.clear_ner_model_cache()
    assert d["state"] == "done", d

    log = (run.out_dir / "test.log").read_text(encoding="utf-8")
    lines = [ln for ln in log.splitlines() if LINE_RX.match(ln)]
    assert lines and all(LINE_RX.match(ln) for ln in log.splitlines() if ln[:2].isdigit())

    def has(level, comp, text, sid="13387752"):
        return any(ln.split(" ", 2)[1] == level and f"[{sid}] [{comp}] " in ln and text in ln for ln in lines)

    assert has("INFO", "업로드", "파일 준비 — 13387752 계약서.txt")
    assert has("INFO", "업로드", "경로 — 매체 텍스트 (text)")
    assert has("INFO", "OCR", "텍스트·OCR 완료") and has("INFO", "OCR", "파일 본문 텍스트 사용 (OCR 생략)")
    assert has("INFO", "LLM", "LLM 응답 — 모델 fake")                   # 작업 스레드(LLM)의 기록
    assert has("INFO", "NER", "NER 모델 적재 시작 — 모델 fake/ner-model")  # 작업 스레드(NER)의 기록
    assert has("ERROR", "NER", "NER 모델 적재 실패 — RuntimeError: Error(s) in loading state_dict for BertCrfForNER:")
    assert has("ERROR", "NER", "bert.embeddings.word_embeddings.weight 가중치 [32000, 1024] ≠ 뼈대 [119547, 768]")
    assert has("ERROR", "NER", "모델 폴더 config.json 의 구조와 맞지 않아")
    assert has("WARNING", "NER", "모델 미적재 — 정규표현식 대체 (모델 0개 · 정규식")
    assert has("INFO", "NER", "NER 추출 완료 — 방식 정규표현식 대체")
    assert has("INFO", "통합", "통합 생략 — consolidate=False")
    assert has("INFO", "측정", "파일 측정 — 형식 TXT")
    assert has("INFO", "문맥", "이용자 당사자 규칙")
    assert "TESTSECRETKEY" not in log

    resp = json.loads(next((run.out_dir / "runs" / "results").glob("*/llm_metadata.json")).read_text(encoding="utf-8"))
    assert resp["ner_mode"] == "regex_fallback" and resp["ner_model_loaded"] is False
    assert resp["ner_load_error"].startswith("RuntimeError: Error(s) in loading state_dict")
    assert resp["ner_entity_count"] == resp["entity_count"] > 0
    assert set(resp["stage_seconds"]) >= {"sha256", "ocr", "extraction", "ner"}

    evs = sse_events(c, run.run_id)
    logs = [e for e in evs if e["type"] == "log"]
    assert all({"line", "level", "component", "set_id"} <= set(e) for e in logs)
    err = [e for e in logs if e["level"] == "ERROR" and e["component"] == "NER"]
    assert err and err[0]["set_id"] == "13387752" and "적재 실패" in err[0]["line"]
    assert any(e.get("stage") == "ocr" and e["set_id"] == "13387752" for e in logs)
    assert any(e.get("stage") == "upload" for e in logs) and any(e.get("stage") == "route" for e in logs)
    assert any(e["component"] == "시스템" and e["set_id"] is None and "실행 준비" in e["line"] for e in logs)
    assert "TESTSECRETKEY" not in json.dumps(evs, ensure_ascii=False)
    no_internal(logs)
    # 시스템 로그에도 같은 실행·세트로 남는다
    sl = c.get(f"/api/test/syslog?run_id={run.run_id}&component=NER&level=ERROR").json()["items"]
    assert sl and sl[0]["set_id"] == "13387752"


# ---------------------------------------------------------------------------
# §A6 통합 검증 상세 (실제 T-001 실행 복사본) · 근거의 판정 행
# ---------------------------------------------------------------------------
def _copy_real_run(dst_root: Path) -> Path:
    dst = dst_root / REAL_RUN.name
    shutil.copytree(REAL_RUN, dst, ignore=shutil.ignore_patterns("cache", "ocr", "ner", "uploads"))
    return dst


@needs_real_run
def test_consolidation_payload_from_real_run(console):
    _copy_real_run(console.runs)
    c = logged_in(console)
    det = c.get(f"/api/test/runs/{REAL_RUN.name}/sets/13387752").json()
    con = det["consolidation"]
    assert con["available"] is True and len(con["legs"]) == 1
    leg = con["legs"][0]
    assert set(leg) >= {"leg", "model_used", "requested_model", "fallback_used", "degraded", "degraded_reason", "status",
                        "seconds", "summary", "decisions", "validation_errors", "fields_before", "fields_after",
                        "lost_fields", "ner"}
    assert leg["leg"] == "contract" and leg["model_used"] == "alibaba-qwen3.5-122b-a10b" and leg["fallback_used"] is False
    assert leg["degraded"] is False and leg["status"] == "completed" and leg["seconds"] is None   # 예전 실행: 소요 미기록
    assert set(leg["summary"]) == {"total_fields", "agreed_fields", "conflicted_fields", "llm_only_fields",
                                   "ner_only_fields", "missing_fields", "overall_confidence"}
    raw = json.loads(next(p for p in (console.runs / REAL_RUN.name / "out/runs/results").glob("*/consolidated_metadata.json")
                          if json.loads(p.read_text(encoding="utf-8"))["validation_report"]["decisions"]
                          and p.parent.name == det["record"]["_meta"]["request_ids"]["contract"]).read_text(encoding="utf-8"))
    assert len(leg["decisions"]) == len(raw["validation_report"]["decisions"])
    for dd in leg["decisions"]:
        assert set(dd) >= {"field", "label_ko", "key23", "llm_value", "ner_value", "final_value", "decision", "confidence",
                           "reasoning"}
        assert dd["decision"] in ("AGREED", "CONFLICT", "LLM_ONLY", "NER_ONLY", "MISSING")
    by = {dd["field"]: dd for dd in leg["decisions"]}
    assert by["author"]["label_ko"] == "저작자" and by["author"]["key23"] == "creator"
    assert by["work_title"]["key23"] == "work_title" and by["license_start_date"]["key23"] == "license_start_date"
    assert by["license_start_date"]["key23_related"] == ["contract_validity_period"]
    nested = [dd for dd in leg["decisions"] if "[" in dd["field"] or "." in dd["field"]]
    assert all(dd["key23"] is None for dd in nested)
    assert any(dd["label_ko"] and " · 주소" in dd["label_ko"] for dd in nested)
    assert sum(leg["decision_counts"].values()) == len(leg["decisions"])
    ner = leg["ner"]
    assert set(ner) >= {"mode", "model", "loaded", "load_error", "entity_count", "entities"}
    assert ner["model"] == "KLUE RoBERTa Large" and ner["entity_count"] == len(ner["entities"]) > 0
    assert all(set(e) == {"text", "type"} for e in ner["entities"])
    assert ner["mode"] is None and ner["loaded"] is None                 # 예전 실행은 NER 방식을 기록하지 않았다
    no_internal(con)

    # 근거 — 판정 행이 있는 23키마다 NER 값 · 통합 판정 · 판정 근거 + 안내
    ev = det["evidence"]
    items = lambda k: [i["label"] for i in ev[k]["items"]]               # noqa: E731
    for k in ("work_title", "license_start_date", "license_end_date"):
        assert {"NER 값", "통합 판정", "판정 근거", "통합 검증"} <= set(items(k)), (k, items(k))
    labels = items("contract_validity_period")                          # 시작일·종료일 판정 둘 → 필드 이름으로 가른다
    assert "통합 판정 · 이용허락 시작일" in labels and "통합 판정 · 이용허락 종료일" in labels
    assert any(i["value"] == "필드 전체 대조는 '통합 검증' 탭에 있습니다" for i in ev["creator"]["items"])
    assert items("creator").count("통합 판정") == 1                        # 저작자는 예전 항목 그대로 + 안내만
    assert "통합 검증" not in items("keywords")                           # T-002 항목(해당 없음)에는 붙이지 않는다


def test_consolidation_payload_unavailable_reasons(console, tmp_path):
    """T-002 저작물 통합을 끈 실행 → available=False 와 이유 (가짜 오케스트레이터의 저작물 leg)."""
    c = logged_in(console)
    r = c.post("/api/test/batch", json={"source": "syn", "workers": 1, "n": 1})
    assert r.status_code == 200, r.text
    rid = r.json()["run_id"]
    assert wait_done(c, rid)["state"] == "done"
    det = c.get(f"/api/test/runs/{rid}/sets/90000001").json()
    con = det["consolidation"]
    assert con["available"] is False and con["legs"] == [] and "consolidate_work=False" in con["reason"]
    assert all(i["label"] != "통합 검증" for k in det["evidence"] for i in det["evidence"][k]["items"])

    # 단위: 이유 고르기
    run = CA._get_run(rid)
    assert "통합 검증 실패 — 시간 초과" in CA._no_consolidation_reason(
        "contract", {"consolidate": True, "consolidation_error": "시간 초과"}, run)
    assert CA._no_consolidation_reason("contract", {}, run).endswith("가 없습니다")
    assert CA._consolidation_payload(run, {"_meta": {}}, "T-001")["available"] is False


def test_evidence_decision_items_unit():
    rows = [{"field": "economic_rights", "llm_value": "[\"복제권\"]", "ner_value": None, "decision": "LLM_ONLY",
             "confidence": 0.7, "reasoning": "NER 없음"},
            {"field": "reproduction_right", "ner_value": "복제", "decision": "AGREED", "confidence": 0.9,
             "reasoning": "같음"}]
    by = CA._decisions_by_key(rows)
    assert [r["field"] for r in by["reproduction_right"]] == ["reproduction_right", "economic_rights"]
    assert [r["field"] for r in by["rental_right"]] == ["economic_rights"]
    entry = {"items": []}
    CA._add_decision_items(entry, "rental_right", by["rental_right"])
    assert [i["label"] for i in entry["items"]] == ["NER 값", "통합 판정", "판정 근거", "통합 검증"]
    assert entry["items"][1]["value"] == "LLM_ONLY (신뢰도 0.7)"
    entry = {"items": []}
    CA._add_decision_items(entry, "reproduction_right", by["reproduction_right"])
    assert "통합 판정 · 복제권 허락 여부" in [i["label"] for i in entry["items"]]
    entry = {"items": []}
    CA._add_decision_items(entry, "work_type", [])
    assert entry["items"] == []
