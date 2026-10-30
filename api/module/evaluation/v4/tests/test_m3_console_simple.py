"""간소화 시험 콘솔 (/test) — 로그인 없는 시험 PC 모드 · 화면 라우트 · 경로 미리보기(/folder/peek). 모델 호출 없음.

- 로그인 없이 쓰는 조건: CONSOLE_NO_LOGIN 켬 + 같은 PC(루프백) + 프록시 머리말 없음 + 브라우저 주소가 localhost.
  하나라도 어긋나면 지금처럼 로그인이 필요하다.
- /folder/peek: 계약서·저작물 폴더는 파일 수·세트 ID·확장자·매체, 정답 파일은 시험별 레코드와 항목 수.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from .test_m3_console import CA, Console, _httpx_testclient_compat, console, logged_in  # noqa: F401

from web import console_folder as CF                          # noqa: E402


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv(CF.ENV_ROOTS, str(tmp_path))
    return tmp_path


def _tpl(console: Console):
    (console.tpl / "console_simple.html").write_text("<html>간소화|사용자={{ user }}|열림={{ open_mode }}|{{ version }}</html>",
                                                     encoding="utf-8")


def _local(console: Console, monkeypatch, host: str = "127.0.0.1", flag: str = "1") -> TestClient:
    monkeypatch.setenv(CA.NO_LOGIN_ENV, flag)
    monkeypatch.setattr(CA, "_client_host", lambda request: host)
    _tpl(console)
    return TestClient(console.init(), base_url="http://localhost:5000")


# ---------------------------------------------------------------------------
# 로그인 없는 시험 PC 모드
# ---------------------------------------------------------------------------
def test_open_local_api_and_page_without_login(console, monkeypatch):
    c = _local(console, monkeypatch)
    r = c.get("/api/test/status")
    assert r.status_code == 200 and r.json()["user"] == CA.LOCAL_USER
    p = c.get("/test")
    assert p.status_code == 200 and "간소화|사용자=시험|열림=True" in p.text


def test_open_local_refuses_without_flag_remote_proxy_or_foreign_host(console, monkeypatch):
    c = _local(console, monkeypatch, flag="0")
    assert c.get("/api/test/status").status_code == 503          # 비밀번호도 없으면 설정 안내 (로그인 모드 그대로)
    c = _local(console, monkeypatch, host="10.0.0.5")
    assert c.get("/api/test/status").status_code in (401, 503)
    c = _local(console, monkeypatch)
    assert c.get("/api/test/status", headers={"X-Forwarded-For": "1.2.3.4"}).status_code in (401, 503)
    assert c.get("/api/test/status", headers={"Host": "evil.example:5000"}).status_code in (401, 503)   # DNS 리바인딩
    assert c.get("/api/test/status", headers={"Host": "127.0.0.1:5000"}).status_code == 200
    assert c.get("/api/test/status", headers={"Host": "[::1]:5000"}).status_code == 200
    # 다른 사이트에서 보낸 POST 는 로그인 없는 모드에서도 막는다
    r = c.post("/api/test/folder/peek", json={"kind": "contract", "path": "/x"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_default_is_on_only_on_wsl(monkeypatch):
    monkeypatch.delenv(CA.NO_LOGIN_ENV, raising=False)
    monkeypatch.setattr(CF, "is_wsl", lambda: False)
    assert CA.no_login_enabled() is False
    monkeypatch.setattr(CF, "is_wsl", lambda: True)
    assert CA.no_login_enabled() is True
    monkeypatch.setenv(CA.NO_LOGIN_ENV, "off")
    assert CA.no_login_enabled() is False


def test_simple_page_needs_login_when_not_local(console, monkeypatch):
    monkeypatch.setenv(CA.NO_LOGIN_ENV, "0")
    _tpl(console)
    c = logged_in(console)
    assert "간소화|사용자=admin|열림=False" in c.get("/test").text
    anon = console.client()
    r = anon.get("/test", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/console/login?next=/test")


# ---------------------------------------------------------------------------
# /folder/peek
# ---------------------------------------------------------------------------
def _files(root):
    c = root / "계약서"
    (c / "하위").mkdir(parents=True)
    (c / "00002_13387752 운수 자판기 (1).pdf").write_bytes(b"%PDF-1.4 x")
    (c / "하위" / "13333800 F!RE 가사.hwp").write_bytes(b"h" * 2048)
    (c / "메모 계약서.pdf").write_bytes(b"%PDF")                   # 세트 ID 없음
    (c / "readme.txt").write_text("x", encoding="utf-8")           # 형식 밖 — 세지 않음
    w = root / "저작물"
    w.mkdir()
    (w / "13387752_1_사진.jpg").write_bytes(b"\xff\xd8" + b"0" * 3000)
    (w / "13333800_1_영상.mp4").write_bytes(b"0" * 10)
    return c, w


def test_peek_contract_and_work_folders(console, roots, monkeypatch):
    c = logged_in(console)
    cdir, wdir = _files(roots)
    r = c.post("/api/test/folder/peek", json={"kind": "contract", "path": str(cdir), "test": "Te-1"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["n_files"] == 3 and d["n_ids"] == 2 and d["n_no_id"] == 1
    assert d["by_ext"] == {"pdf": 2, "hwp": 1} and d["truncated"] is False
    hwp = next(x for x in d["items"] if x["name"].endswith(".hwp"))
    assert hwp["set_id"] == "13333800" and hwp["dir"] == "하위" and hwp["size"] == 2048
    w = c.post("/api/test/folder/peek", json={"kind": "work", "path": str(wdir)}).json()
    assert w["n_files"] == 2 and w["by_media"] == {"image": 1, "video": 1}
    assert {x["set_id"] for x in w["items"]} == {"13387752", "13333800"}
    monkeypatch.setattr(CA, "PEEK_ROWS", 1)
    t = c.post("/api/test/folder/peek", json={"kind": "contract", "path": str(cdir)}).json()
    assert len(t["items"]) == 1 and t["truncated"] is True and t["n_files"] == 3


def test_peek_answers_filters_by_test_like_scan(console, roots):
    c = logged_in(console)
    g = roots / "정답.jsonl"
    rows = [{"set_id": "13387752", "test": "T-001", "work_title": "운수 자판기", "work_type": "어문", "creator": "김은혜",
             "reproduction_right": "허용"},
            {"set_id": "13387752", "test": "T-002", "keywords": "자판기;운수", "file_format": "JPG"},
            {"set_id": "13333800", "work_title": "F!RE 가사", "hashtags": "#불"}]          # 시험 표시 없음 → 두 시험 모두
    g.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    d1 = c.post("/api/test/folder/peek", json={"kind": "answers", "path": str(g), "test": "T-001"}).json()
    assert d1["n"] == 2 and d1["test"] == "T-001" and d1["format"] == "jsonl"
    first = next(x for x in d1["items"] if x["set_id"] == "13387752")
    assert first["work_title"] == "운수 자판기" and first["n_contract_keys"] >= 3 and first["n_work_keys"] == 0
    assert d1["n_contract_keys"] == 16 and d1["n_work_keys"] == 7
    d2 = c.post("/api/test/folder/peek", json={"kind": "answers", "path": str(g), "test": "Te-2"}).json()
    assert d2["n"] == 2 and next(x for x in d2["items"] if x["set_id"] == "13387752")["n_work_keys"] == 2
    assert d2["n_with_work_keys"] == 2


def test_peek_errors(console, roots):
    c = logged_in(console)
    cdir, _ = _files(roots)
    assert c.post("/api/test/folder/peek", json={"kind": "nope", "path": str(cdir)}).json()["error"] == "bad_kind"
    r = c.post("/api/test/folder/peek", json={"kind": "contract", "path": "/etc"})
    assert r.status_code == 403
    r = c.post("/api/test/folder/peek", json={"kind": "answers", "path": str(cdir)})          # 폴더를 정답 파일로
    assert r.status_code in (400, 404)
    bad = roots / "깨진.json"
    bad.write_text("{not json", encoding="utf-8")
    r = c.post("/api/test/folder/peek", json={"kind": "answers", "path": str(bad)})
    assert r.status_code == 400 and r.json()["error"] == "answer_unreadable"
    r = c.post("/api/test/folder/peek", content=b"[]", headers={"Content-Type": "application/json"})
    assert r.status_code == 400
