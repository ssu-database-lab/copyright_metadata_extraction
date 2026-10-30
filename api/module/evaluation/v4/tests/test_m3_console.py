"""시험 콘솔 백엔드 테스트 — 로그인·실행·판정·근거·미리보기·내려받기·복원 (docs/v4/M3_console_spec.md §2~§4, §6).

모델 호출 없음: 오케스트레이터는 가짜(ConsoleOrch)이고 실행기는 실제 BatchRunner 다. T-001 채점은 실제 평가기를 쓰고
(설명 키가 없어 임베딩 모델을 올리지 않는다), T-002 채점은 평가기에 스텁 판정기를 끼운다. 계약서 PDF 는
dataset/new_contracts/ 를 읽기만 하고, 저작물·gold·목록·.env 는 전부 임시 폴더에 만든다.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

ROOT = Path(__file__).resolve().parents[5]
_API_DIR = str(ROOT / "api")
if _API_DIR not in sys.path:            # 콘솔은 app.py 처럼 api/ 를 루트로 import 된다 (web.*, module.*)
    sys.path.insert(0, _API_DIR)

from fastapi import FastAPI                                  # noqa: E402
from fastapi.templating import Jinja2Templates               # noqa: E402
from fastapi.testclient import TestClient                    # noqa: E402

from web import console_api as CA                            # noqa: E402
from web import console_auth as AU                           # noqa: E402
from module.evaluation import batch_runner as BR             # noqa: E402

NEW_CONTRACT = ROOT / "dataset/new_contracts/00002_13387752 운수 자판기 (1).pdf"
GOLD_T001_NEW5 = ROOT / "dataset/test_set_v4/gold23_t001_new5.jsonl"
PASSWORD = "testpass123"
INTERNAL_RX = re.compile(r"\bv4\b|\bM[1-3]b?\b")

needs_contract = pytest.mark.skipif(not NEW_CONTRACT.is_file(), reason="신규 양식 계약서 PDF 없음")


# ---------------------------------------------------------------------------
# 가짜 오케스트레이터
# ---------------------------------------------------------------------------
def _gold_values(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["set_id"]] = {k: v.get("value") for k, v in r["keys"].items()}
    return out


class ConsoleOrch:
    """PipelineOrchestrator 대역 — results/<rid>/llm_metadata.json(+영상 keyframes/)을 쓰고 web.pipeline 로거에 단계 로그를 남긴다."""

    def __init__(self, out_dir: Path, t001_gold: Dict[str, Dict[str, Any]], t002_preds: Dict[str, Dict[str, Any]],
                 block: Optional[threading.Event] = None, fail: bool = False):
        self.results = Path(out_dir) / "runs" / "results"
        self.t001_gold = t001_gold
        self.t002_preds = t002_preds
        self.block = block
        self.fail = fail
        self.n = 0
        self.lock = threading.Lock()

    def run(self, file_bytes: bytes, filename: str, **kw) -> Dict[str, Any]:
        if not file_bytes and kw.get("source_path"):          # 실행기는 파일을 제자리로 넘긴다 (source_path)
            file_bytes = Path(kw["source_path"]).read_bytes()
        if self.block is not None:
            self.block.wait(20)
        if self.fail:
            raise RuntimeError("가짜 파이프라인 오류")
        with self.lock:
            self.n += 1
            rid = f"20261011_000000_{self.n:06d}"
        d = self.results / rid
        d.mkdir(parents=True, exist_ok=True)
        test = kw.get("test")
        plog = logging.getLogger("web.pipeline")
        plog.info("Pipeline setup: %s (0.10MB)", filename)
        sha = hashlib.sha256(file_bytes).hexdigest()
        resp: Dict[str, Any] = {"success": True, "request_id": rid, "file_name": filename,
                                "filename": filename.replace(" ", "_"), "file_hash": sha,
                                "file_size_bytes": len(file_bytes), "model_used": "fake", "test": test}
        if test == "T-001":
            plog.info("Router: %s → modality=document", filename)
            plog.info("OCR complete: 1234 chars")
            plog.info("Consolidation start (LLM: OK, NER: OK)")
            plog.info("Consolidation complete")
            tok = next((t for t in re.findall(r"\d{5,8}", filename) if t in self.t001_gold), None)
            pred = dict(self.t001_gold.get(tok) or {})
            resp.update({
                "ocr_provider": "alibaba", "ocr_provider_used": "direct_text", "ocr_pages": 5, "consolidate": True,
                "consolidation_success": True,
                "metadata": {"author": "김은혜", "copyright_holder": "김은혜", "rental_right": "△",
                             "reproduction_right": "□", "valid_period": "2009.9.27 ~ 2033.4.14"},
                "consolidated_metadata": {"author": "김은혜"},
                "consolidation_decisions": [{"field": "author", "llm_value": "김은혜", "ner_value": "김은혜",
                                             "final_value": "김은혜", "decision": "AGREED",
                                             "reasoning": "두 값이 같음", "confidence": 1.0}],
                "contract_context_rules": {"title": "운수 자판기", "work_type": "어문", "source": "rules",
                                           "title_raw": "13387752 운수 자판기", "work_type_ticked": ["어문"]},
                "licensee_party_detail": {"source": "signature", "signature": ["한국저작권위원회"],
                                          "preamble": "한국저작권위원회", "agree": True},
                "test_prediction": {**pred, "_meta": {
                    "serializer": "eval23", "test": "T-001", "work_title_source": "rules",
                    "model_work_title": "운수자판기", "work_type_source": "rules", "model_work_type": "어문저작물",
                    "copyright_holder_source": "licensee_party", "model_copyright_holder": "김은혜",
                    "unparsed": {"rental_right": "△"}}},
            })
        else:
            sid = next((t for t in re.findall(r"\d{5,8}", filename) if t in self.t002_preds), None)
            p = dict(self.t002_preds.get(sid) or {})
            video = filename.lower().endswith(".mp4")
            if video:
                kf = d / "keyframes"
                kf.mkdir()
                from PIL import Image
                for i, t in enumerate((0.5, 1.5, 2.5), 1):
                    Image.new("RGB", (32, 18), (i * 60, 40, 90)).save(kf / f"frame{i:02d}_t{t}s.jpg")
                plog.info("Video extraction via fake-omni: 3 frames (schedule v4)")
            else:
                plog.info("VLM extraction complete via fake-vlm: 7 fields")
            p.setdefault("file_size", len(file_bytes))          # None 이면(FN 시나리오) 그대로 둔다
            p["file_hash"] = sha
            resp.update({
                "ocr_provider": "(video/keyframes)" if video else "(image/VLM)", "consolidate": False,
                "modality": "video" if video else "image", "vlm_backend": "fake-vlm",
                "metadata": {"keyword": p.get("keywords"), "hashtags": p.get("hashtags"),
                             "digital_format": p.get("file_format"), "description": p.get("description")},
                "work_consolidation": {"applied": False, "reason": "consolidate_work=False — 직접 사용 (v4 A3)"},
                "test_prediction": {**p, "_meta": {"serializer": "eval23", "test": "T-002", "hashtags_source": "model"}},
            })
            if video:
                resp["keyframes"] = {"kept": 3, "target": 3, "sampled": 4, "duration": 3.0, "schedule": "v4"}
        (d / "llm_metadata.json").write_text(json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8")
        return resp


class StubJudge:
    def decide(self, p: str, g: str) -> Dict[str, Any]:
        same = p.strip() == g.strip()
        return {"match": same, "cosine": 0.91 if same else 0.12, "shared_tokens": ["바다"] if same else [],
                "tau": 0.65, "content_check": True, "content_ok": same}

    def info(self) -> Dict[str, Any]:
        return {"model_id": "stub", "revision": None, "device": "cpu", "tau": 0.65, "content_check": True,
                "max_length": 512, "pooling": "cls", "cache_path": None}


# ---------------------------------------------------------------------------
# 합성 T-002 목록 · gold
# ---------------------------------------------------------------------------
def _na_keys(test_of_na: str) -> Dict[str, Any]:
    from module.evaluation.v4.keys23 import KEYS
    return {k.name: {"value": None, "state": "na", "source": "비채점", "tier": "", "kind": None}
            for k in KEYS if k.test == test_of_na}


def _make_t002(tmp: Path) -> Dict[str, Any]:
    from PIL import Image
    works = tmp / "works"
    works.mkdir()
    img = works / "90000001_1_테스트 사진.jpg"
    Image.new("RGB", (64, 48), (10, 120, 200)).save(img, "JPEG")
    big = works / "90000003_1_큰 사진.jpg"
    Image.new("RGB", (2000, 1000), (200, 120, 10)).save(big, "JPEG")
    vid = works / "90000002_1_테스트 영상.mp4"
    vid.write_bytes(b"fakevideo" * 100)
    preds = {
        "90000001": {"keywords": ["바다", "하늘", "갈매기"], "hashtags": ["#바다", "#하늘"], "file_format": "JPG",
                     "resolution": "64x48", "description": "푸른 바다와 하늘"},
        "90000002": {"keywords": None, "hashtags": ["#영상"], "file_format": "MP4", "resolution": "1280x720",
                     "description": "영상 설명", "file_size": None},
        "90000003": {"keywords": ["노을"], "hashtags": ["#노을"], "file_format": "JPG", "resolution": "2000x1000",
                     "description": "노을"},
    }
    header = ("test,cell,sub_source,work_id,media,license_bucket,source,title,work_file,has_contract,contract_pdf,"
              "work_설명_tier,work_해시태그_tier,work_해상도_tier")
    rows = [f"T-002,image/ccl,합성,90000001,image,ccl,합성,테스트 사진,{img},False,,A,,A",
            f"T-002,video/kogl,합성,90000002,video,kogl,합성,테스트 영상,{vid},False,,A,,A",
            f"T-002,image/ccl,합성,90000003,image,ccl,합성,큰 사진,{big},False,,A,,A"]
    csv = tmp / "syn_ids.csv"
    csv.write_text("﻿" + header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    def g(v, kind=None, state="value"):
        return {"value": v, "state": state, "source": "합성", "tier": "A", "kind": kind}
    gold = [
        {"set_id": "90000001", "test": "T-002", "media": "image", "bucket": "ccl", "source": "합성",
         "keys": {**_na_keys("T-001"), "file_name": g(img.name),
                  "keywords": g(["바다", "하늘", "모래"]), "hashtags": g(["바다", "하늘"], "generated"),
                  "file_format": g("JPG"), "file_size": g(img.stat().st_size), "resolution": g("64x48"),
                  "description": g("푸른 바다와 하늘", "generated"),
                  "file_hash": g(hashlib.sha256(img.read_bytes()).hexdigest())}},
        {"set_id": "90000002", "test": "T-002", "media": "video", "bucket": "kogl", "source": "합성",
         "keys": {**_na_keys("T-001"), "file_name": g(vid.name),
                  "keywords": g(None, state="empty"), "hashtags": g(None, state="na"),
                  "file_format": g("MP4"), "file_size": g(vid.stat().st_size), "resolution": g("1920x1080"),
                  "description": g("다른 설명", "generated"),
                  "file_hash": g(hashlib.sha256(vid.read_bytes()).hexdigest())}},
        {"set_id": "90000003", "test": "T-002", "media": "image", "bucket": "ccl", "source": "합성",
         "keys": {**_na_keys("T-001"), "file_name": g(big.name),
                  "keywords": g(["노을"]), "hashtags": g(["노을"], "generated"),
                  "file_format": g("JPG"), "file_size": g(big.stat().st_size), "resolution": g("2000x1000"),
                  "description": g("노을", "generated"),
                  "file_hash": g(hashlib.sha256(big.read_bytes()).hexdigest())}},
    ]
    gold_path = tmp / "gold_syn.jsonl"
    gold_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in gold) + "\n", encoding="utf-8")
    return {"csv": csv, "gold": gold_path, "preds": preds, "img": img, "vid": vid, "big": big}


# ---------------------------------------------------------------------------
# 앱 · 로그인 도구
# ---------------------------------------------------------------------------
# 텍스트층 계약서 1건의 추정 비용 — qwen3.5 생각 끔(기본)이면 추출 + 통합만, OCR 요금 없음
TEXT_CONTRACT_KRW = BR.COST_LLM_EXTRACT_NO_THINK + BR.COST_CONSOLIDATION_DOC_NO_THINK


class Console:
    def __init__(self, tmp: Path, monkeypatch):
        self.tmp = tmp
        self.mp = monkeypatch
        self.env = tmp / "console.env"
        self.runs = tmp / "runs"
        self.tpl = tmp / "tpl"
        self.tpl.mkdir()
        (self.tpl / "console.html").write_text("<html><title>콘솔</title>사용자={{ user }} 버전={{ version }}</html>",
                                               encoding="utf-8")
        (self.tpl / "console_login.html").write_text(
            "<html>로그인|{{ error or '' }}|next={{ next }}|{{ version }}</html>", encoding="utf-8")
        self.syn = _make_t002(tmp)
        self.t001_gold = _gold_values(GOLD_T001_NEW5) if GOLD_T001_NEW5.is_file() else {}
        self.block: Optional[threading.Event] = None
        self.fail = False
        self.fail_factory = False
        self.orchs = []

    def orch_factory(self, out_dir: Path):
        if self.fail_factory:
            raise RuntimeError("오케스트레이터 준비 실패")
        o = ConsoleOrch(out_dir, self.t001_gold, self.syn["preds"], block=self.block, fail=self.fail)
        self.orchs.append(o)
        return o

    def sources(self):
        return CA.default_sources(ROOT) + [
            {"id": "syn", "label": "합성 T-002", "test": "T-002", "csv": [self.syn["csv"]], "gold": self.syn["gold"]},
            {"id": "syn_ids", "label": "합성 ID 입력", "test": None, "by_ids": True,
             "csv_by_test": {"T-002": [self.syn["csv"]]}, "gold_by_test": {"T-002": self.syn["gold"]}},
        ]

    def init(self, templates=None):
        CA.init_console_api(templates=templates or Jinja2Templates(directory=str(self.tpl)), runs_root=self.runs,
                            orchestrator_factory=self.orch_factory, sources=self.sources(),
                            single_gold={"T-001": GOLD_T001_NEW5, "T-002": self.syn["gold"]}, env_path=self.env)
        app = FastAPI()
        app.include_router(CA.router)
        return app

    def client(self, https: bool = False, templates=None) -> TestClient:
        return TestClient(self.init(templates), base_url="https://testserver" if https else "http://testserver")

    def set_password(self, pw: str = PASSWORD, user: str = "admin"):
        AU.set_password(self.env, user, pw)


@pytest.fixture(autouse=True)
def _httpx_testclient_compat(monkeypatch):
    """starlette 0.35 의 TestClient 는 httpx.Client(app=…) 를 부르는데 httpx 0.28 에서 그 인자가 빠졌다 — 테스트에서만 걸러 준다."""
    import inspect
    import httpx
    if "app" not in inspect.signature(httpx.Client.__init__).parameters:
        orig = httpx.Client.__init__

        def init(self, *a, app=None, **kw):
            orig(self, *a, **kw)
        monkeypatch.setattr(httpx.Client, "__init__", init)


@pytest.fixture
def console(tmp_path, monkeypatch):
    for k in (AU.KEY_USER, AU.KEY_HASH, AU.KEY_SECRET):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(AU, "FAIL_DELAY_SEC", 0.0)
    AU.THROTTLE.reset()
    monkeypatch.setattr(CA, "_ping_dashscope", lambda: "ok")
    monkeypatch.setattr(CA, "_chain_labels", lambda: (["qwen3.8-omni-flash (DashScope)"], ["qwen3.8-omni-flash (DashScope)"]))
    CA._PING.update(value="unknown", at=0.0, busy=False, event=None)
    # T-002 채점 — 실제 평가기에 스텁 판정기를 끼운다 (임베딩 모델을 올리지 않는다)
    import module.evaluation.v4.evaluate as EVM
    real = EVM.evaluate

    def fake_evaluate(test, gold, pred, out, **kw):
        kw.pop("embed_cache", None)
        kw.pop("embed_revision", None)
        kw.setdefault("n_boot", 50)
        return real(test, gold, pred, out, judge=StubJudge(), **kw)
    monkeypatch.setattr(EVM, "evaluate", fake_evaluate)
    c = Console(tmp_path, monkeypatch)
    yield c
    if c.block is not None:
        c.block.set()
    # 다음 테스트로 넘어가기 전에 실행 스레드가 끝나기를 기다린다
    deadline = time.time() + 30
    while time.time() < deadline and any(r.state in CA.ACTIVE_STATES for r in list(CA._RUNS.values())):
        time.sleep(0.05)
    AU.configure(AU.DEFAULT_ENV_PATH)
    CA.uninstall_log_capture()          # 로그 수집(레벨 낮춤·루트 필터)을 다음 테스트로 끌고 가지 않는다 (caplog 핸들러는 세션 공용)


def login(client: TestClient, pw: str = PASSWORD, nxt: str = "/console", user: str = "admin"):
    return client.post("/console/login", data={"user": user, "password": pw, "next": nxt}, follow_redirects=False)


def logged_in(console: Console, **kw) -> TestClient:
    console.set_password()
    c = console.client(**kw)
    r = login(c)
    assert r.status_code == 303, r.text
    return c


def wait_done(client: TestClient, run_id: str, timeout: float = 60.0) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/test/runs/{run_id}")
        assert r.status_code == 200, r.text
        d = r.json()
        if d["state"] in CA.TERMINAL_STATES:
            return d
        time.sleep(0.05)
    raise AssertionError(f"실행이 끝나지 않았습니다: {run_id}")


def wait_state(client: TestClient, run_id: str, states, timeout: float = 20.0) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        d = client.get(f"/api/test/runs/{run_id}").json()
        if d["state"] in states:
            return d
        time.sleep(0.02)
    raise AssertionError(f"상태 {states} 에 이르지 않았습니다: {run_id}")


def run_single_t001(client: TestClient) -> Dict[str, Any]:
    r = client.post("/api/test/single", data={"test": "T-001"},
                    files={"contract": (NEW_CONTRACT.name, NEW_CONTRACT.read_bytes(), "application/pdf")})
    assert r.status_code == 200, r.text
    return r.json()


def sse_events(client: TestClient, run_id: str, after: int = 0):
    evs = []
    with client.stream("GET", f"/api/test/runs/{run_id}/stream?after={after}") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                evs.append(ev)
                if ev.get("type") == "closed":
                    break
    return evs


def no_internal(obj) -> None:
    s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    m = INTERNAL_RX.search(s)
    assert m is None, f"내부 표기가 보입니다: …{s[max(0, m.start() - 60): m.end() + 60]}…"


# ---------------------------------------------------------------------------
# §2 인증
# ---------------------------------------------------------------------------
def test_password_hash_format_and_verify():
    h = AU.hash_password("correct horse")
    algo, it, salt, dk = h.split("$")
    assert algo == "pbkdf2_sha256" and int(it) >= 200_000 and salt and dk
    assert AU.verify_password("correct horse", h)
    assert not AU.verify_password("wrong horse", h)
    assert not AU.verify_password("x", "pbkdf2_sha256$abc$$")
    assert AU.parse_hash("md5$1$a$b") is None
    with pytest.raises(ValueError):
        AU.hash_password("x", iterations=1000)


def test_not_configured_page_and_api_503(console):
    c = console.client()
    r = c.get("/console", follow_redirects=False)
    assert r.status_code == 503 and "set-password" in r.text and "v1.0" in r.text
    assert c.get("/console/login").status_code == 503
    r = c.get("/api/test/status")
    assert r.status_code == 503 and r.json()["error"] == "console_not_configured"


def test_unauthenticated_redirects_and_401(console):
    console.set_password()
    c = console.client()
    r = c.get("/console", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/console/login?next=/console"
    r = c.get("/console/runs/abc?x=1", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/console/login?next=/console/runs/abc")
    for path in ("/api/test/status", "/api/test/runs", "/api/test/runs/x/sets"):
        r = c.get(path)
        assert r.status_code == 401 and r.json()["error"] == "unauthorized"
    assert c.post("/api/test/batch", json={}).status_code == 401
    r = c.get("/console/login?next=/console")
    assert r.status_code == 200 and "next=/console" in r.text and "v1.0" in r.text


def test_login_success_cookie_flags_and_logout(console):
    console.set_password()
    c = console.client()
    r = login(c, nxt="/console/runs?x=1")
    assert r.status_code == 303 and r.headers["location"] == "/console/runs?x=1"
    sc = r.headers["set-cookie"]
    assert "console_session=" in sc and "HttpOnly" in sc and "samesite=lax" in sc.lower() and "Max-Age=43200" in sc
    assert "Secure" not in sc
    r = c.get("/console")
    assert r.status_code == 200 and "사용자=admin" in r.text and "버전=v1.0" in r.text
    st = c.get("/api/test/status").json()
    assert st["version"] == "v1.0" and st["user"] == "admin"
    r = c.get("/console/logout", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/console/login"
    c.cookies.clear()
    assert c.get("/api/test/status").status_code == 401


def test_secure_cookie_only_over_https(console):
    console.set_password()
    c = console.client(https=True)
    r = login(c)
    assert r.status_code == 303 and "Secure" in r.headers["set-cookie"]


def test_login_failure_delay_and_unsafe_next(console, monkeypatch):
    console.set_password()
    c = console.client()
    monkeypatch.setattr(AU, "FAIL_DELAY_SEC", 0.3)
    t0 = time.perf_counter()
    r = login(c, pw="wrong-password")
    assert r.status_code == 401 and "올바르지 않습니다" in r.text
    assert time.perf_counter() - t0 >= 0.3
    monkeypatch.setattr(AU, "FAIL_DELAY_SEC", 0.0)
    assert login(c, user="root").status_code == 401                     # 아이디가 다르면 실패
    for bad in ("//evil.example/x", "https://evil.example/", "/\\evil.example", "javascript:alert(1)"):
        AU.THROTTLE.reset()
        r = login(c, nxt=bad)
        assert r.status_code == 303 and r.headers["location"] == "/console", bad


def test_login_throttle_after_five_failures(console, monkeypatch):
    console.set_password()
    c = console.client()
    now = [time.time()]
    monkeypatch.setattr(AU, "_now", lambda: now[0])
    for _ in range(5):
        assert login(c, pw="nope-nope").status_code == 401
    r = login(c)                                                         # 맞는 비밀번호도 30초는 기다린다
    assert r.status_code == 429 and "초 뒤에 다시" in r.text and int(r.headers["retry-after"]) >= 29
    now[0] += 31
    assert login(c).status_code == 303
    assert c.get("/api/test/status").status_code == 200


def test_session_expiry_tamper_and_password_change(console):
    console.set_password()
    c = console.client()
    s = AU.get_settings()
    c.cookies.set(AU.SESSION_COOKIE, AU.make_session_token(s, "admin", now=time.time() - 13 * 3600))
    assert c.get("/api/test/status").status_code == 401                  # 12시간 지난 쿠키
    good = AU.make_session_token(s, "admin")
    c.cookies.set(AU.SESSION_COOKIE, good[:-2] + ("AA" if not good.endswith("AA") else "BB"))
    assert c.get("/api/test/status").status_code == 401                  # 서명이 틀린 쿠키
    c.cookies.set(AU.SESSION_COOKIE, good)
    assert c.get("/api/test/status").status_code == 200
    time.sleep(0.01)
    console.set_password("brand-new-pass")                               # 비밀번호를 바꾸면 기존 세션이 끊긴다
    assert c.get("/api/test/status").status_code == 401
    assert login(c, pw="brand-new-pass").status_code == 303


def test_env_override_password(console, monkeypatch):
    console.set_password("file-password")
    monkeypatch.setenv(AU.KEY_HASH, AU.hash_password("env-password"))   # 시험용 덮어쓰기 (.env 와 다르다)
    c = console.client()
    assert login(c, pw="file-password").status_code == 401
    assert login(c, pw="env-password").status_code == 303


def test_cross_site_post_rejected(console):
    c = logged_in(console)
    r = c.post("/api/test/batch", json={"source": "t001_new5"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and r.json()["error"] == "forbidden"


def test_cli_set_password_writes_env_without_printing_secrets(tmp_path):
    envf = tmp_path / ".env"
    envf.write_text("FOO=bar\n# 주석\nCONSOLE_USER=old\n", encoding="utf-8")
    cmd = [sys.executable, "-m", "api.web.console_auth", "set-password", "--user", "admin",
           "--password", "cli-secret-pass", "--env-file", str(envf)]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    text = envf.read_text(encoding="utf-8")
    from dotenv import dotenv_values
    vals = dotenv_values(envf)
    assert vals["FOO"] == "bar" and "# 주석" in text and vals["CONSOLE_USER"] == "admin"
    assert text.count("CONSOLE_USER=") == 1
    assert AU.verify_password("cli-secret-pass", vals["CONSOLE_PASSWORD_HASH"])
    assert len(vals["CONSOLE_SECRET"]) >= 64
    assert "cli-secret-pass" not in p.stdout + p.stderr
    assert vals["CONSOLE_PASSWORD_HASH"] not in p.stdout + p.stderr and vals["CONSOLE_SECRET"] not in p.stdout
    secret = vals["CONSOLE_SECRET"]
    # 표준입력으로 — 서명 키는 유지된다
    p2 = subprocess.run(cmd[:6] + ["--env-file", str(envf)], cwd=str(ROOT), capture_output=True, text=True,
                        input="stdin-pass-123\n", timeout=60)
    assert p2.returncode == 0, p2.stderr
    vals2 = dotenv_values(envf)
    assert AU.verify_password("stdin-pass-123", vals2["CONSOLE_PASSWORD_HASH"])
    assert vals2["CONSOLE_SECRET"] == secret and "stdin-pass-123" not in p2.stdout
    p3 = subprocess.run(cmd[:6] + ["--password", "short", "--env-file", str(envf)], cwd=str(ROOT),
                        capture_output=True, text=True, timeout=60)
    assert p3.returncode == 2 and "8자" in p3.stderr


def test_router_import_does_not_load_torch():
    code = ("import sys; sys.path.insert(0, 'api'); import web.console_api, web.console_auth; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules, 'torch loaded'")
    p = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr


def test_vlm_unit_costs_updated():
    from module.evaluation import batch_runner as BR
    assert BR.COST_VLM_IMAGE == pytest.approx(1.34) and BR.COST_VLM_VIDEO == pytest.approx(1.77)
    assert BR._estimate_cost({"success": True}, "video", False, consolidated=False) == pytest.approx(1.77)


# ---------------------------------------------------------------------------
# §4 상태 · 추정
# ---------------------------------------------------------------------------
def test_status_sources_models_and_no_internal_names(console):
    c = logged_in(console)
    st = c.get("/api/test/status").json()
    assert st["version"] == "v1.0" and st["services"] == {"dashscope": "ok"}
    assert st["models"]["ocr"] == "qwen3.7-plus" and st["models"]["ocr_fallback"] == "qwen3.8-max"
    assert st["models"]["extraction"] == "qwen3.5-122b-a10b" and st["models"]["ner"] == "klue-roberta-large"
    assert st["defaults"] == {"tau": 0.65, "workers": {"T-001": 50, "T-002": 32}, "max_workers": 200}
    assert st["active"] == {"batch": None, "batches": [], "single": None}
    src = {s["id"]: s for s in st["sources"]}
    assert set(src) >= {"t001_new5", "t002", "pilot200", "t001", "ids", "syn"}
    assert src["t001"]["enabled"] is False and src["t001"]["reason"] == "신규 양식 계약서 대기"
    if GOLD_T001_NEW5.is_file():
        assert src["t001_new5"]["n"] == 5 and src["t001_new5"]["gold"] is True
        assert src["t001_new5"]["cells"] == {"text/donated": 5}
    assert src["syn"]["n"] == 3 and src["syn"]["cells"] == {"image/ccl": 2, "video/kogl": 1}
    for s in st["sources"]:
        assert set(s) >= {"id", "label", "test", "n", "cells", "gold", "enabled", "reason"}
    no_internal(st)


def test_estimate(console):
    c = logged_in(console)
    if GOLD_T001_NEW5.is_file():
        e = c.get("/api/test/estimate?test=T-001&source=t001_new5").json()
        # 기본 50명(생각 끔): 한 묶음 × 57.5 s · 건당 ₩25.5. 측정 범위 밖(40명 미만)은 끝 값 50 s
        assert e["n_sets"] == 5 and e["est_krw"] == pytest.approx(5 * 25.5) and e["est_sec"] == 58
        assert e["by_cell"] == {"text/donated": 5}
        assert c.get("/api/test/estimate?source=t001_new5&n=2&workers=3").json()["est_sec"] == 50
        assert c.get("/api/test/estimate?source=t001_new5&n=2&workers=60").json()["est_sec"] == 65
    e = c.get("/api/test/estimate?source=syn").json()
    assert e["n_sets"] == 3 and e["est_krw"] == pytest.approx(1.34 * 2 + 1.77) and e["est_sec"] == 37   # 한 묶음 × 37 s
    e = c.get("/api/test/estimate?source=syn&cells=video/kogl").json()
    assert e["n_sets"] == 1 and e["by_cell"] == {"video/kogl": 1}
    e = c.get("/api/test/estimate?source=syn&consolidate_work=true").json()
    assert e["est_krw"] == pytest.approx(1.34 * 2 + 1.77 + 3 * 35.4)
    e = c.get("/api/test/estimate?source=syn_ids&test=T-002&ids=90000002,99999999").json()
    assert e["n_sets"] == 1 and e["missing_ids"] == ["99999999"]
    r = c.get("/api/test/estimate?source=t001")
    assert r.status_code == 400 and r.json()["error"] == "source_disabled" and "신규 양식" in r.json()["message"]
    r = c.get("/api/test/estimate?source=nope")
    assert r.status_code == 400 and r.json()["error"] == "unknown_source"
    r = c.get("/api/test/estimate?source=syn&n=abc")
    assert r.status_code == 400 and "정수" in r.json()["message"]


# ---------------------------------------------------------------------------
# 판정 매핑
# ---------------------------------------------------------------------------
def test_t001_estimate_follows_load_test(monkeypatch):
    E = type("E", (), {"media": "text", "license_bucket": "ccl", "extra": {}, "set_id": "x"})
    four_k = [E() for _ in range(4000)]
    monkeypatch.delenv("CONSOLIDATION_THINKING", raising=False)                              # 기본: 생각 끔
    assert CA._estimate("T-001", four_k, 50)["est_sec"] == round(80 * 57.5)                  # ≈77분
    assert CA._estimate("T-001", four_k, 50)["est_krw"] == pytest.approx(4000 * 25.5)
    assert CA._estimate("T-001", four_k, 100)["est_sec"] == round(4000 * 17000 / 1_000_000 * 60)   # 토큰 한도가 상한
    monkeypatch.setenv("CONSOLIDATION_THINKING", "on")
    assert CA._estimate("T-001", four_k, 60)["est_sec"] == 67 * 102                         # 켬 ≈1.9 h
    assert CA._estimate("T-001", four_k, 100)["est_sec"] == round(4000 * 24400 / 1_000_000 * 60)
    assert CA._estimate("T-001", four_k, 500)["workers"] == CA.MAX_WORKERS
    assert CA._interp(CA.T001_LATENCY_THINKING, 54) == pytest.approx(96.5)
    W = type("W", (), {"license_bucket": "ccl", "extra": {}, "set_id": "x"})
    works = [type("I", (W,), {"media": "image"})() for _ in range(2000)] + [type("V", (W,), {"media": "video"})() for _ in range(2000)]
    assert CA._estimate("T-002", works, 32)["est_sec"] == round((2000 * 4.6 + 2000 * 188.0) / 68.0)   # 드라이브 읽기가 상한 ≈1.6 h


def test_verdict_mapping():
    v = CA.verdict_of
    assert v(None) is None
    assert v({"key": "work_title", "state": "na", "tp": 0, "fp": 0, "fn": 0}) == "NA"
    assert v({"key": "work_title", "state": "value", "tp": 1, "fp": 0, "fn": 0, "matched": True}) == "TP"
    assert v({"key": "keywords", "state": "value", "tp": 2, "fp": 1, "fn": 0, "matched": False}) == "PARTIAL"
    assert v({"key": "hashtags", "state": "value", "tp": 1, "fp": 0, "fn": 3, "matched": False}) == "PARTIAL"
    assert v({"key": "keywords", "state": "value", "tp": 0, "fp": 2, "fn": 3, "matched": False}) == "WRONG"
    assert v({"key": "resolution", "state": "value", "tp": 0, "fp": 1, "fn": 1, "matched": False}) == "WRONG"
    assert v({"key": "file_size", "state": "value", "tp": 0, "fp": 0, "fn": 1, "matched": False}) == "FN"
    assert v({"key": "description", "state": "empty", "tp": 0, "fp": 1, "fn": 0, "matched": False}) == "FP"
    assert v({"key": "keywords", "state": "empty", "tp": 0, "fp": 0, "fn": 0, "matched": None}) == "EMPTY_OK"
    assert v({"key": "rental_right", "state": "value", "tp": 0, "fp": 1, "fn": 0}) == "FP"


def test_single_set_id_rules(tmp_path):
    gold = tmp_path / "g.jsonl"
    gold.write_text('{"set_id": "13387752", "keys": {}}\n', encoding="utf-8")
    sid = CA._single_set_id
    assert sid("00002_13387752 운수 자판기 (1).pdf", gold) == ("13387752", True)       # 앞의 생성 순번이 아니다
    assert sid("00002_13387752 운수 자판기 (1).pdf", None) == ("13387752", False)      # 목록에 없으면 마지막 묶음
    assert sid("00002_55555555 x.pdf", gold, {"00002"}) == ("00002", False)          # 목록에 있는 첫 번째
    assert sid("11066733_1_서울대공원24210.jpg", gold, {"11066733"}) == ("11066733", False)
    assert sid("11066733_1_서울대공원24210.jpg", gold) == ("24210", False)
    assert sid("서울대공원.jpg", gold) == ("upload", False)
    assert sid("1234_짧은번호.jpg", gold) == ("upload", False)
    assert sid("123456789_길다.jpg", gold) == ("upload", False)                     # 9자리는 묶음이 아니다


def test_known_ids_cover_rows_without_contract(console):
    console.init()
    assert {"90000001", "90000002", "90000003"} <= CA._known_ids("T-002")
    t001 = ROOT / "dataset/test_set_v4/t001_ids.csv"
    if t001.is_file():
        import csv
        rows = list(csv.DictReader(open(t001, encoding="utf-8-sig")))
        no_contract = next(r["work_id"] for r in rows if not r["contract_pdf"].strip())
        assert no_contract in CA._known_ids("T-001")                    # 계약서가 아직 없던 세트도 알아본다


def test_scrubbers():
    assert CA.scrub_text("# v4 평가 보고서 — T-002") == "# 평가 보고서 — T-002"
    assert "평가 규칙서" in CA.scrub_text("판정 규칙: `docs/v4/M1_evaluator_spec.md`")
    assert CA.scrub_text("gold: `dataset/test_set_v4/gold23_t002.jsonl`") == "gold: `gold23_t002.jsonl`"
    assert CA.scrub_text("직렬화기(M2 A2) 미반영") == "직렬화기 미반영"
    assert CA.scrub_text("프레임 v4 · 출력") == "프레임 v1.0 · 출력"
    assert CA.scrub_json({"a": "v4", "b": ["직접 사용 (v4 A3)"], "c": "BMW M3 사진"}) == \
        {"a": "v1.0", "b": ["직접 사용"], "c": "BMW M3 사진"}                 # 내용 문자열은 그대로


# ---------------------------------------------------------------------------
# §3 실행 — 단건 T-001 (실제 평가기)
# ---------------------------------------------------------------------------
@needs_contract
def test_single_t001_end_to_end_record_evidence_previews(console):
    c = logged_in(console)
    created = run_single_t001(c)
    assert created["set_id"] == "13387752" and created["gold"] is True
    rid = created["run_id"]
    assert re.fullmatch(r"\d{8}_\d{6}_single_[0-9a-f]{4}", rid)
    d = wait_done(c, rid)
    assert d["state"] == "done", d
    assert d["macro_f1"] == pytest.approx(1.0) and d["passed"] is True and d["n_sets"] == 1
    assert d["cost_krw"] == pytest.approx(TEXT_CONTRACT_KRW) and d["done"] == 1 and d["failed"] == 0
    assert d["set_id"] == "13387752" and d["gold"] is True and d["config"]["set_id"] == "13387752"
    assert {"test_metrics.json", "per_set.jsonl", "report.md", "test.log", "run.zip"} <= set(d["downloads"])
    no_internal(d)
    run_dir = console.runs / rid
    assert (run_dir / "run.json").is_file() and (run_dir / "manifest.jsonl").is_file()
    assert (run_dir / "inputs" / "contract" / NEW_CONTRACT.name).is_file()
    assert (run_dir / "out" / "runs" / "results").is_dir()
    log = (run_dir / "out" / "test.log").read_text(encoding="utf-8")
    # 줄 형식 HH:MM:SS LEVEL [set_id] [구성요소] 메시지 — 파이프라인 기록(가짜 오케스트레이터의 web.pipeline 줄)도 세트·구성요소가 붙어 test.log 에 들어간다
    assert re.search(r"(?m)^\d\d:\d\d:\d\d INFO \[13387752\] \[OCR\] OCR complete: 1234 chars$", log), log
    assert "INFO [13387752] [통합] Consolidation complete" in log and "INFO [13387752] [시스템] 시작 — test=T-001" in log
    assert "INFO [-] [채점] 평가 완료 — T-001" in log

    m = c.get(f"/api/test/runs/{rid}/metrics").json()
    assert m["pass_mark"] == 0.85 and m["macro_f1"] == pytest.approx(1.0) and "spec" not in m
    no_internal(m)

    sets = c.get(f"/api/test/runs/{rid}/sets").json()
    assert sets["total"] == 1
    it = sets["items"][0]
    assert it["set_id"] == "13387752" and it["state"] == "ok" and it["cell"] == "text/donated"
    assert it["keys"]["work_title"] == {"verdict": "TP", "value_short": "운수 자판기"}
    assert it["keys"]["other_rights"]["verdict"] == "NA" and it["keys"]["keywords"]["verdict"] is None
    assert len(it["keys"]) == 23

    det = c.get(f"/api/test/runs/{rid}/sets/13387752").json()
    keys = det["keys"]
    assert [k["no"] for k in keys] == list(range(1, 24))
    by = {k["key"]: k for k in keys}
    assert by["work_title"]["verdict"] == "TP" and by["work_title"]["gold"] == "운수 자판기"
    assert by["work_title"]["source"] == "계약서 규칙 파싱" and by["work_title"]["group_ko"] == "식별정보"
    assert by["other_rights"]["display"] == "해당 없음" and by["other_rights"]["scored"] is False   # 기타 권리는 늘 해당 없음
    assert by["keywords"]["display"] == "NULL" and by["keywords"]["scored"] is False   # 다른 시험 항목은 NULL
    assert by["copyright_holder"]["display"] == "한국저작권위원회"
    assert by["reproduction_right"]["display"] in ("허용", "불허")
    ev = det["evidence"]
    assert set(ev) == set(by)
    items = lambda k: {i["label"]: i["value"] for i in ev[k]["items"]}           # noqa: E731
    assert items("work_title")["규칙 파싱 원문"] == "13387752 운수 자판기"
    assert items("work_title")["모델 추출값"] == "운수자판기"
    assert items("work_type")["체크된 칸"] == "어문"
    ch = items("copyright_holder")
    assert ch["서명란"] == "한국저작권위원회" and ch["두 출처 일치"] == "예" and ch["모델 저작권자 값(쓰지 않음)"] == "김은혜"
    assert ev["copyright_holder"]["source_label"] == "본문 규칙(이용자 당사자)"
    assert items("creator")["통합 판정"] == "일치 (AGREED) · 신뢰도 1.0"
    assert items("rental_right")["해석 불가 원문"] == "△"
    assert items("contract_validity_period")["기간 원문"] == "2009.9.27 ~ 2033.4.14"
    assert items("file_name")["원본 파일명"] == NEW_CONTRACT.name
    assert ev["keywords"]["source_label"] is None and "해당 없음" in ev["keywords"]["summary"]
    assert det["record"]["_meta"]["set_id"] == "13387752"
    assert det["legs"]["contract"]["request_id"] and any(
        f["path"] == "llm_metadata.json" for f in det["legs"]["contract"]["files"])
    assert det["models"]["ocr"] == "qwen3.7-plus"
    pages = det["previews"]["contract"]["pages"]
    import fitz
    with fitz.open(str(NEW_CONTRACT)) as doc:
        assert len(pages) == doc.page_count
    assert det["previews"]["work"]["kind"] is None
    r = c.get(pages[0]["url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
    assert (run_dir / "cache" / "13387752_contract_p1.png").is_file()
    assert c.get(f"/api/test/runs/{rid}/sets/13387752/preview/contract/99.png").status_code == 404
    assert c.get(f"/api/test/runs/{rid}/sets/13387752/preview/contract/abc.png").status_code == 404
    assert c.get(f"/api/test/runs/{rid}/sets/13387752/preview/work.jpg").status_code == 404
    no_internal(det)

    # SSE — 시작·세트·진행·로그(단계 포함)·상태·채점·닫힘, 그리고 after 로 이어 붙이기
    evs = sse_events(c, rid)
    types = [e["type"] for e in evs]
    for t in ("start", "set_start", "set_done", "progress", "log", "state", "finalized", "closed"):
        assert t in types, (t, types)
    sd = next(e for e in evs if e["type"] == "set_done")
    assert sd["set_id"] == "13387752" and sd["ok"] is True and sd["cost_krw"] == pytest.approx(TEXT_CONTRACT_KRW)
    ss = next(e for e in evs if e["type"] == "set_start")
    for e in (sd, ss):
        assert e["cell"] == "text/donated" and e["media"] == "text" and e["title"] == "운수 자판기"
    assert all(isinstance(e["line"], str) for e in evs if e["type"] == "log")
    assert set(sd) >= {"set_id", "ok", "elapsed_sec", "cost_krw", "error", "seq", "ts"}
    assert any(e["type"] == "log" and e.get("stage") == "ocr" and e.get("set_id") == "13387752" for e in evs)
    assert [e["state"] for e in evs if e["type"] == "state"] == ["running", "scoring", "done"]
    seqs = [e["seq"] for e in evs if e["type"] != "closed"]
    assert seqs == sorted(seqs)
    mid = seqs[len(seqs) // 2]
    evs2 = sse_events(c, rid, after=mid)
    assert all(e["seq"] > mid for e in evs2 if e["type"] != "closed") and evs2[-1]["type"] == "closed"
    no_internal([e for e in evs if e["type"] == "log"])

    runs = c.get("/api/test/runs").json()
    assert runs[0]["run_id"] == rid and runs[0]["kind"] == "single"
    assert set(runs[0]) >= {"run_id", "kind", "name", "test", "source", "state", "created_at", "started_at",
                            "finished_at", "n_sets", "done", "failed", "cost_krw", "macro_f1", "micro_f1", "passed"}


@needs_contract
def test_downloads_and_zips(console):
    c = logged_in(console)
    rid = run_single_t001(c)["run_id"]
    assert wait_done(c, rid)["state"] == "done"
    for name in ("test_predictions.jsonl", "test_metrics.json", "report.md", "test.log", "per_set.jsonl"):
        r = c.get(f"/api/test/runs/{rid}/download/{name}")
        assert r.status_code == 200, name
        cd = r.headers["content-disposition"]
        assert cd == f'attachment; filename="{rid}_{name}"'
        no_internal(cd)
        if name != "per_set.jsonl":
            no_internal(r.content.decode("utf-8"))
    rep = c.get(f"/api/test/runs/{rid}/download/report.md").text
    assert rep.startswith("# 평가 보고서") and "평가 규칙서" in rep
    assert "pass_mark" in c.get(f"/api/test/runs/{rid}/download/test_metrics.json").json()
    for bad in ("run.json", "..%2Frun.json", "secrets.env", "manifest.jsonl"):
        assert c.get(f"/api/test/runs/{rid}/download/{bad}").status_code == 404, bad
    r = c.get(f"/api/test/runs/{rid}/download/run.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    assert f"{rid}/run.json" in names and f"{rid}/out/test_predictions.jsonl" in names
    assert f"{rid}/out/test_metrics.json" in names and f"{rid}/manifest.jsonl" in names
    assert any(n.startswith(f"{rid}/inputs/contract/") for n in names)
    assert not any("/runs/uploads/" in n for n in names)
    assert any(n.endswith("llm_metadata.json") for n in names)
    assert json.loads(z.read(f"{rid}/run.json"))["run_id"] == rid
    no_internal(z.read(f"{rid}/out/report.md").decode("utf-8"))
    r = c.get(f"/api/test/runs/{rid}/sets/13387752/download")
    assert r.status_code == 200 and r.headers["content-disposition"] == f'attachment; filename="{rid}_set_13387752.zip"'
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert {"prediction.json", "verdicts.json", "contract/llm_metadata.json"} <= set(z.namelist())
    assert c.get(f"/api/test/runs/{rid}/sets/99999999/download").status_code == 404


@needs_contract
def test_single_busy_409_and_failed_set(console):
    c = logged_in(console)
    console.block = threading.Event()
    first = run_single_t001(c)
    wait_state(c, first["run_id"], ("running",))
    r = c.post("/api/test/single", data={"test": "T-001"},
               files={"contract": (NEW_CONTRACT.name, NEW_CONTRACT.read_bytes(), "application/pdf")})
    assert r.status_code == 409 and r.json() == {"error": "busy", "message": r.json()["message"],
                                                 "active": first["run_id"]}
    assert c.get("/api/test/status").json()["active"]["single"] == first["run_id"]
    assert len([p for p in console.runs.iterdir() if p.is_dir()]) == 1       # 거절된 업로드는 남기지 않는다
    console.block.set()
    assert wait_done(c, first["run_id"])["state"] == "done"
    console.block = None
    console.fail = True                                                      # 세트 1건 실패 — 실행은 끝까지
    rid = run_single_t001(c)["run_id"]
    d = wait_done(c, rid)
    assert d["state"] == "done" and d["failed"] == 1
    it = c.get(f"/api/test/runs/{rid}/sets?failed=true").json()["items"][0]
    assert it["ok"] is False and "가짜 파이프라인 오류" in it["error"]
    console.fail, console.fail_factory = False, True                         # 준비 단계 실패 → error 상태
    rid = run_single_t001(c)["run_id"]
    d = wait_done(c, rid)
    assert d["state"] == "error" and d["error"].startswith("실행 오류 —") and "오케스트레이터 준비 실패" in d["error"]


def test_single_validation_errors(console):
    c = logged_in(console)
    assert c.post("/api/test/single", data={"test": "T-003"}).json()["error"] == "bad_test"
    r = c.post("/api/test/single", data={"test": "T-001"})
    assert r.status_code == 400 and "계약서" in r.json()["message"]
    r = c.post("/api/test/single", data={"test": "T-002"},
               files={"contract": ("a.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 400 and "저작물" in r.json()["message"]
    r = c.post("/api/test/single", data={"test": "T-002"}, files={"work": ("a.exe", b"MZ", "application/x")})
    assert r.status_code == 400 and r.json()["error"] == "bad_file"
    assert not console.runs.exists() or not any(console.runs.iterdir())


# ---------------------------------------------------------------------------
# 일괄 · T-002 (합성 목록, 스텁 판정기)
# ---------------------------------------------------------------------------
def test_batch_t002_verdicts_filters_evidence_previews(console):
    c = logged_in(console)
    r = c.post("/api/test/batch", json={"source": "syn", "workers": 2, "tau": 0.65, "name": "합성 일괄"})
    assert r.status_code == 200, r.text
    created = r.json()
    assert created["n_sets"] == 3 and created["estimate"]["krw"] == pytest.approx(1.34 * 2 + 1.77)
    rid = created["run_id"]
    assert "_batch_" in rid
    d = wait_done(c, rid)
    assert d["state"] == "done" and d["n_sets"] == 3 and d["done"] == 3 and d["name"] == "합성 일괄"
    assert d["cost_krw"] == pytest.approx(1.34 * 2 + 1.77, abs=0.05)
    assert d["config"]["workers"] == 2 and d["config"]["gold"] == "gold_syn.jsonl"
    assert d["macro_f1"] is not None and d["micro_f1"] is not None
    no_internal(d)

    sets = c.get(f"/api/test/runs/{rid}/sets").json()
    assert sets["total"] == 3
    by = {it["set_id"]: it for it in sets["items"]}
    k1, k2 = by["90000001"]["keys"], by["90000002"]["keys"]
    assert k1["keywords"]["verdict"] == "PARTIAL" and k1["keywords"]["set_f1"] == pytest.approx(2 / 3)
    assert k1["hashtags"]["verdict"] == "TP" and k1["description"]["verdict"] == "TP"
    assert k1["file_hash"]["verdict"] == "TP" and k1["work_title"]["verdict"] is None
    assert k2["keywords"]["verdict"] == "EMPTY_OK" and k2["hashtags"]["verdict"] == "NA"
    assert k2["resolution"]["verdict"] == "WRONG" and k2["description"]["verdict"] == "WRONG"
    assert k2["file_size"]["verdict"] == "FN" and k2["file_format"]["verdict"] == "TP"
    assert by["90000002"]["media"] == "video" and by["90000002"]["cell"] == "video/kogl"
    q = lambda s: c.get(f"/api/test/runs/{rid}/sets?{s}").json()                 # noqa: E731
    assert q("verdict=PARTIAL")["total"] == 1
    assert [i["set_id"] for i in q("key=resolution&verdict=WRONG")["items"]] == ["90000002"]
    assert q("key=resolution&verdict=WRONG,FN")["total"] == 1
    assert q("cell=image/ccl")["total"] == 2 and q("failed=true")["total"] == 0
    assert q("q=테스트")["total"] == 2 and q("q=90000003")["total"] == 1
    page = q("limit=1&offset=1")
    assert page["total"] == 3 and len(page["items"]) == 1 and page["items"][0]["set_id"] == "90000002"

    det = c.get(f"/api/test/runs/{rid}/sets/90000001").json()
    kd = {k["key"]: k for k in det["keys"]}
    assert kd["keywords"]["verdict"] == "PARTIAL" and kd["keywords"]["gold"] == ["바다", "하늘", "모래"]
    assert kd["keywords"]["detail"]["set_f1"] == pytest.approx(2 / 3)
    assert set(kd["keywords"]["detail"]) == {"pairs", "set_f1", "unmatched_gold", "unmatched_pred"}   # per_set 의 detail 그대로
    assert kd["keywords"]["counts"] == {"tp": 2, "fp": 1, "fn": 1, "state": "value", "kind": None, "matched": False}
    assert kd["resolution"]["detail"] == {"gold_norm": "64x48", "pred_norm": "64x48"}
    assert set(kd["description"]["detail"]) >= {"cosine", "tau", "shared_tokens"}
    assert kd["creator"]["detail"] is None and kd["creator"]["counts"] is None
    assert kd["creator"]["display"] == "NULL" and kd["file_name"]["display"] == "90000001_1_테스트 사진.jpg"
    items = lambda k: {i["label"]: i["value"] for i in det["evidence"][k]["items"]}   # noqa: E731
    assert items("keywords")["일치한 짝"] == "바다 = 바다, 하늘 = 하늘"
    assert items("keywords")["정답에만 있음"] == "모래" and items("keywords")["예측에만 있음"] == "갈매기"
    assert items("hashtags")["생성 방식"] == "모델 생성(VLM)"
    assert items("description")["코사인 유사도"] == "0.910" and items("description")["판정"] == "통과"
    assert "τ 0.65" in det["evidence"]["description"]["summary"]
    assert items("resolution")["방법"] == "이미지 헤더 (Pillow)"
    assert items("file_hash")["SHA-256"] == hashlib.sha256(console.syn["img"].read_bytes()).hexdigest()
    pw = det["previews"]["work"]
    assert pw["kind"] == "image" and pw["image_url"].endswith("/preview/work.jpg") and pw["frames"] == []
    assert det["previews"]["contract"]["pages"] == []
    assert det["models"]["vlm_backend"] == "fake-vlm"
    r = c.get(pw["image_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    no_internal(det)

    from PIL import Image
    big = c.get(f"/api/test/runs/{rid}/sets/90000003/preview/work.jpg")
    assert big.status_code == 200 and max(Image.open(io.BytesIO(big.content)).size) == 1280

    vd = c.get(f"/api/test/runs/{rid}/sets/90000002").json()
    pv = vd["previews"]["work"]
    assert pv["kind"] == "video" and pv["image_url"] is None
    assert [f["t"] for f in pv["frames"]] == [0.5, 1.5, 2.5]
    assert [f["name"] for f in pv["frames"]] == ["frame01_t0.5s.jpg", "frame02_t1.5s.jpg", "frame03_t2.5s.jpg"]
    for f in pv["frames"]:
        r = c.get(f["url"])
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    vitems = {i["label"]: i["value"] for i in vd["evidence"]["resolution"]["items"]}
    assert vitems["방법"] == "영상 스트림 (ffprobe)"
    assert {i["label"]: i["value"] for i in vd["evidence"]["description"]["items"]}["판정"] == "미달"
    no_internal(vd)                                                   # "(v4 A3)"·frame_schedule "v4" 가 지워졌다

    # 경로 안전 — 프레임 이름·세트 id·레코드의 request_id 를 조작해도 실행 폴더 밖을 읽지 못한다
    base = f"/api/test/runs/{rid}/sets/90000002/preview/frames"
    for bad in ("frame09_t9.0s.jpg", "..%2F..%2F..%2Frun.json", "%2E%2E", "llm_metadata.json"):
        assert c.get(f"{base}/{bad}").status_code == 404, bad
    for bad_sid in ("%2E%2E", "..%2F..%2Frun", "x" * 100, "a%20b"):
        assert c.get(f"/api/test/runs/{rid}/sets/{bad_sid}").status_code == 404, bad_sid
    run_obj = CA._get_run(rid)
    for bad_sid in ("..", ".", "../run", "a/b", ""):
        with pytest.raises(CA.ConsoleError):
            CA._record_path(run_obj, bad_sid)
    assert CA._leg_dir(run_obj, "..") is None and CA._leg_dir(run_obj, "../../x") is None
    pred_file = console.runs / rid / "out" / "test_predictions" / "90000002.json"
    rec = json.loads(pred_file.read_text(encoding="utf-8"))
    rec["_meta"]["request_ids"]["work"] = "../../.."
    pred_file.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    assert c.get(f"{base}/frame01_t0.5s.jpg").status_code == 404
    assert c.get(f"/api/test/runs/{rid}/sets/90000002").json()["previews"]["work"]["frames"] == []
    assert c.get(f"/api/test/runs/x{rid}/sets").status_code == 404


def test_batch_errors_ids_and_busy(console):
    c = logged_in(console)
    r = c.post("/api/test/batch", json={"source": "t001"})
    assert r.status_code == 400 and r.json()["error"] == "source_disabled"
    r = c.post("/api/test/batch", json={"source": "syn", "test": "T-001"})
    assert r.status_code == 400 and r.json()["error"] == "test_mismatch"
    r = c.post("/api/test/batch", json={"source": "syn", "cells": ["text/expired"]})
    assert r.status_code == 400 and r.json()["error"] == "empty"
    r = c.post("/api/test/batch", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"] == "bad_request"
    r = c.post("/api/test/batch", json={"source": "syn_ids", "ids": ["90000003", "99999999"]})
    assert r.status_code == 400 and r.json()["error"] == "bad_test"
    console.block = threading.Event()
    r = c.post("/api/test/batch", json={"source": "syn_ids", "test": "T-002", "ids": "90000003, 99999999",
                                        "workers": 1})
    assert r.status_code == 200, r.text
    created = r.json()
    assert created["n_sets"] == 1 and created["missing_ids"] == ["99999999"]
    rid = created["run_id"]
    wait_state(c, rid, ("running",))
    r = c.post("/api/test/batch", json={"source": "syn"})
    assert r.status_code == 409 and r.json()["error"] == "busy" and r.json()["active"] == rid
    assert c.get("/api/test/status").json()["active"]["batch"] == rid
    console.block.set()
    d = wait_done(c, rid)
    assert d["state"] == "done" and d["config"]["ids"] == ["90000003", "99999999"]
    assert d["config"]["missing_ids"] == ["99999999"]


def test_batch_cancel(console):
    c = logged_in(console)
    console.block = threading.Event()
    r = c.post("/api/test/batch", json={"source": "syn", "workers": 1, "sample": "first"})
    rid = r.json()["run_id"]
    wait_state(c, rid, ("running",))
    assert c.post(f"/api/test/runs/{rid}/cancel").json()["ok"] is True
    console.block.set()
    d = wait_done(c, rid)
    assert d["state"] == "cancelled" and d["done"] < 3
    assert c.post(f"/api/test/runs/{rid}/cancel").json() == {"ok": True, "state": "cancelled"}
    items = c.get(f"/api/test/runs/{rid}/sets").json()["items"]
    assert {i["state"] for i in items} <= {"ok", "pending"} and any(i["state"] == "pending" for i in items)


def test_batch_random_sample_is_reproducible(console):
    c = logged_in(console)
    e1 = c.get("/api/test/estimate?source=syn&n=2&sample=random&seed=7").json()
    e2 = c.get("/api/test/estimate?source=syn&n=2&sample=random&seed=7").json()
    assert e1 == e2 and e1["n_sets"] == 2
    r = c.post("/api/test/batch", json={"source": "syn", "n": 2, "sample": "random"})
    d = wait_done(c, r.json()["run_id"])
    assert d["config"]["sample"] == "random" and isinstance(d["config"]["seed"], int) and d["n_sets"] == 2


def test_single_t002_with_gold(console):
    c = logged_in(console)
    img = console.syn["img"]
    r = c.post("/api/test/single", data={"test": "T-002"}, files={"work": (img.name, img.read_bytes(), "image/jpeg")})
    assert r.status_code == 200, r.text
    created = r.json()
    assert created["set_id"] == "90000001" and created["gold"] is True
    d = wait_done(c, created["run_id"])
    assert d["state"] == "done" and d["macro_f1"] is not None
    det = c.get(f"/api/test/runs/{created['run_id']}/sets/90000001").json()
    assert det["previews"]["work"]["kind"] == "image"
    assert {k["key"]: k["verdict"] for k in det["keys"]}["keywords"] == "PARTIAL"
    # gold 없는 단건 — 채점 없이 기록만
    r = c.post("/api/test/single", data={"test": "T-002"}, files={"work": ("무제.png", img.read_bytes(), "image/png")})
    created = r.json()
    assert created == {"run_id": created["run_id"], "set_id": "upload", "gold": False}
    d = wait_done(c, created["run_id"])
    assert d["state"] == "done" and d["macro_f1"] is None
    assert c.get(f"/api/test/runs/{created['run_id']}/metrics").status_code == 404
    det = c.get(f"/api/test/runs/{created['run_id']}/sets/upload").json()
    assert all(k["verdict"] is None for k in det["keys"]) and det["gold"] is False
    rep = c.get(f"/api/test/runs/{created['run_id']}/download/report.md").text
    no_internal(rep)


# ---------------------------------------------------------------------------
# 복원 (서버 재시작)
# ---------------------------------------------------------------------------
def test_recovery_after_restart(console):
    c = logged_in(console)
    r = c.post("/api/test/batch", json={"source": "syn", "workers": 2})
    done_id = r.json()["run_id"]
    assert wait_done(c, done_id)["state"] == "done"
    # 실행 도중에 서버가 죽은 실행을 흉내 낸다 — run.json 은 running, 결과 2건
    dead = console.runs / "20261011_090000_batch_dead"
    (dead / "out").mkdir(parents=True)
    src = json.loads((console.runs / done_id / "run.json").read_text(encoding="utf-8"))
    src.update(run_id=dead.name, state="running", finished_at=None, final=None, summary=None,
               progress={"i": 0, "total": 3, "failed": 0, "cost_krw_total": 0.0, "last_set": None})
    (dead / "run.json").write_text(json.dumps(src, ensure_ascii=False), encoding="utf-8")
    (dead / "manifest.jsonl").write_bytes((console.runs / done_id / "manifest.jsonl").read_bytes())
    (dead / "out" / "results.jsonl").write_text(
        json.dumps({"set_id": "90000001", "ok": True, "cost_krw": 1.34}) + "\n"
        + json.dumps({"set_id": "90000002", "ok": False, "cost_krw": 0.0, "error": "x"}) + "\n", encoding="utf-8")
    (dead / "out" / "test.log").write_text("2026-10-11 09:00:00,000 INFO 실행기 준비 — 프레임 v4\n", encoding="utf-8")

    c2 = TestClient(console.init())                    # 같은 저장 위치로 다시 띄운다 (재시작)
    assert login(c2).status_code == 303
    rows = {r["run_id"]: r for r in c2.get("/api/test/runs").json()}
    assert rows[done_id]["state"] == "done" and rows[done_id]["macro_f1"] is not None
    d = c2.get(f"/api/test/runs/{dead.name}").json()
    assert d["state"] == "interrupted" and "재시작" in d["error"]
    assert d["progress"]["i"] == 2 and d["progress"]["failed"] == 1 and d["cost_krw"] == pytest.approx(1.34)
    assert json.loads((dead / "run.json").read_text(encoding="utf-8"))["state"] == "interrupted"
    # 끝난 실행은 결과·근거·미리보기가 그대로 열린다
    assert c2.get(f"/api/test/runs/{done_id}/metrics").status_code == 200
    assert c2.get(f"/api/test/runs/{done_id}/sets").json()["total"] == 3
    det = c2.get(f"/api/test/runs/{done_id}/sets/90000002").json()
    assert len(det["previews"]["work"]["frames"]) == 3
    assert c2.get(det["previews"]["work"]["frames"][0]["url"]).status_code == 200
    evs = sse_events(c2, dead.name)
    assert any(e["type"] == "log" and "프레임 v1.0" in e["line"] for e in evs)
    assert evs[-1] == {"type": "closed", "state": "interrupted", "seq": evs[-2]["seq"]}
    assert c2.get(f"/api/test/runs/{dead.name}/sets").json()["total"] == 3


def test_real_templates_render_if_present(console):
    tdir = ROOT / "api" / "web" / "templates"
    if not (tdir / "console.html").is_file() or not (tdir / "console_login.html").is_file():
        pytest.skip("콘솔 화면 템플릿이 아직 없습니다")
    console.set_password()
    c = console.client(templates=Jinja2Templates(directory=str(tdir)))
    r = c.get("/console/login?next=/console")
    assert r.status_code == 200 and "v1.0" in r.text
    no_internal(re.sub(r"<script.*?</script>|<style.*?</style>", "", r.text, flags=re.S))
    assert login(c).status_code == 303
    r = c.get("/console")
    assert r.status_code == 200 and "v1.0" in r.text and "시험 콘솔" in r.text


# ---------------------------------------------------------------------------
# 다시 실행 — 실패·미실행 세트만 같은 실행에서 (4,000세트 규모에서 빠진 세트를 끝까지 채운다)
# ---------------------------------------------------------------------------
def test_retry_reruns_only_failed_sets_and_rescores(console):
    c = logged_in(console)
    console.fail = True                                                      # 첫 회차: 세트 전부 실패
    rid = c.post("/api/test/batch", json={"source": "syn", "workers": 1}).json()["run_id"]
    d = wait_done(c, rid)
    assert d["state"] == "done" and d["remaining"] == {"ok": 0, "failed": 3, "pending": 0, "retries": 0}
    csv_r = c.get(f"/api/test/runs/{rid}/download/unfinished.csv")
    assert csv_r.status_code == 200 and csv_r.text.startswith("﻿") and csv_r.text.count("실패") == 3
    console.fail = False                                                     # 두 번째 회차는 성공
    r = c.post(f"/api/test/runs/{rid}/retry", json={})
    assert r.status_code == 200 and r.json()["runs"] == [{"run_id": rid, "test": "T-002", "failed": 3, "pending": 0}]
    d2 = wait_done(c, rid)
    assert d2["state"] == "done" and d2["remaining"] == {"ok": 3, "failed": 0, "pending": 0, "retries": 1}
    assert d2["macro_f1"] is not None and d2["config"]["retries"][0]["failed"] == 3
    items = c.get(f"/api/test/runs/{rid}/sets").json()["items"]
    assert {i["state"] for i in items} == {"ok"} and {i["attempts"] for i in items} == {2}
    again = c.post(f"/api/test/runs/{rid}/retry", json={})
    assert again.status_code == 400 and again.json()["error"] == "nothing_to_retry"
    assert c.get(f"/api/test/runs/{rid}/download/unfinished.csv").text.count("\n") == 1     # 머리글만


def test_retry_finishes_cancelled_run_and_guards(console):
    c = logged_in(console)
    console.block = threading.Event()
    rid = c.post("/api/test/batch", json={"source": "syn", "workers": 1, "sample": "first"}).json()["run_id"]
    wait_state(c, rid, ("running",))
    assert c.post(f"/api/test/runs/{rid}/retry", json={}).json()["error"] == "not_finished"
    c.post(f"/api/test/runs/{rid}/cancel")
    console.block.set()
    d = wait_done(c, rid)
    assert d["state"] == "cancelled" and d["remaining"]["pending"] >= 1
    first_done = d["remaining"]["ok"]
    console.block = None
    r = c.post(f"/api/test/runs/{rid}/retry", json={}).json()
    assert r["runs"][0]["pending"] == 3 - first_done
    d2 = wait_done(c, rid)
    assert d2["state"] == "done" and d2["remaining"]["pending"] == 0 and d2["remaining"]["failed"] == 0
    items = {i["set_id"]: i["attempts"] for i in c.get(f"/api/test/runs/{rid}/sets").json()["items"]}
    assert sorted(items.values()) == [1, 1, 1]                               # 성공한 세트는 다시 돌리지 않았다
