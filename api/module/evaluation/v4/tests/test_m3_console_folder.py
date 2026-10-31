"""시험 콘솔 — 폴더로 시험 (web/console_folder.py + /api/test/fs · /folder) 테스트.

모델 호출 없음: test_m3_console 의 가짜 오케스트레이터·스텁 판정기를 그대로 쓴다. 폴더·정답은 임시 폴더에 만들고,
허용 위치(CONSOLE_DATA_ROOTS)는 그 임시 폴더로 좁힌다.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import threading
from pathlib import Path

import pytest

from .test_m3_console import (CA, GOLD_T001_NEW5, NEW_CONTRACT, Console, _httpx_testclient_compat, console,  # noqa: F401
                              logged_in, needs_contract, wait_done, wait_state)

from web import console_folder as CF                          # noqa: E402
from module.evaluation.manifest import ManifestEntry           # noqa: E402
from module.evaluation.v4.keys23 import KEYS                   # noqa: E402


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv(CF.ENV_ROOTS, str(tmp_path))
    return tmp_path


def _media(p):
    return CA._media_of_path(str(p))


# ---------------------------------------------------------------------------
# 경로 · 허용 위치
# ---------------------------------------------------------------------------
def test_windows_paths_convert_and_display():
    assert str(CF.to_local_path(r"E:\시험자료\계약서")) == "/mnt/e/시험자료/계약서"
    assert str(CF.to_local_path("e:/a/b/")) == "/mnt/e/a/b"
    assert str(CF.to_local_path("E:")) == "/mnt/e"
    assert str(CF.to_local_path(r"\\wsl.localhost\Ubuntu\home\u\x")) == "/home/u/x"
    assert str(CF.to_local_path('"/mnt/e/x y"')) == "/mnt/e/x y"
    assert CF.display_path("/mnt/e/시험자료/계약서") == "E:\\시험자료\\계약서"
    assert CF.display_path("/mnt/e") == "E:\\"
    assert CF.display_path("/home/u") == "/home/u"
    for bad in ("", "시험자료\\계약서", "a/b"):
        with pytest.raises(CF.FolderError):
            CF.to_local_path(bad)


def test_roots_env_limits_and_turns_off(tmp_path, monkeypatch):
    monkeypatch.setenv(CF.ENV_ROOTS, f"{tmp_path};/nonexistent/also")
    assert Path(os.path.realpath(tmp_path)) in CF.data_roots()
    monkeypatch.setenv(CF.ENV_ROOTS, "")
    assert CF.data_roots() == []
    with pytest.raises(CF.FolderError) as e:
        CF.resolve_allowed(str(tmp_path), [])
    assert e.value.status == 403 and e.value.code == "folders_off"


def test_resolve_allowed_blocks_outside_traversal_and_symlinks(roots):
    inside = roots / "a"
    inside.mkdir()
    rs = CF.data_roots()
    assert CF.resolve_allowed(str(inside), rs) == Path(os.path.realpath(inside))
    for bad in ("/etc", str(roots / "a" / ".." / ".."), "/"):
        with pytest.raises(CF.FolderError) as e:
            CF.resolve_allowed(bad, rs)
        assert e.value.status == 403
    link = roots / "escape"
    os.symlink("/etc", link)
    with pytest.raises(CF.FolderError) as e:
        CF.resolve_allowed(str(link), rs)
    assert e.value.status == 403
    with pytest.raises(CF.FolderError) as e:
        CF.resolve_allowed(str(roots / "missing"), rs)
    assert e.value.status == 404


def test_list_dir_counts_and_answer_files(roots):
    d = roots / "data"
    (d / "sub").mkdir(parents=True)
    (d / "$RECYCLE.BIN").mkdir()
    (d / "13387752 계약서.pdf").write_bytes(b"%PDF")
    (d / "90000001.jpg").write_bytes(b"x")
    (d / "정답.csv").write_text("set_id\n1\n", encoding="utf-8")
    (d / "memo.txt").write_text("x", encoding="utf-8")
    out = CF.list_dir(str(d), CF.data_roots(), CA.CONTRACT_EXT, CA.WORK_EXT)
    assert [x["name"] for x in out["dirs"]] == ["sub"]
    assert out["counts"] == {"contract": 1, "work": 1, "answer": 1, "other": 1}
    assert [x["name"] for x in out["answer_files"]] == ["정답.csv"]
    assert out["parent"] is not None
    top = CF.list_dir(str(roots), CF.data_roots(), CA.CONTRACT_EXT, CA.WORK_EXT)
    assert top["parent"] is None                                   # 허용 위치 밖으로는 올라가지 않는다


# ---------------------------------------------------------------------------
# 세트 ID · 정답 파일
# ---------------------------------------------------------------------------
def test_test_names_follow_procedure():
    for v in ("T-001", "t-001", "T001", "Te-1", "TE-1", "te1", " Te - 1 "):
        assert CF.norm_test(v) == "T-001", v
    for v in ("T-002", "Te-2", "te2"):
        assert CF.norm_test(v) == "T-002", v
    assert CF.norm_test("Te-3") is None and CF.norm_test(None) is None and CF.norm_test("") is None
    assert CF.test_label("T-001") == "Te-1" and CF.test_label("T-002") == "Te-2" and CF.test_label(None) is None


def test_answers_csv_test_column_accepts_te_names(tmp_path):
    p = tmp_path / "a.csv"
    _write_csv(p, [{"set_id": "90000003", "시험": "Te-2", "파일포맷": "JPG"},
                   {"set_id": "90000004", "시험": "Te-1", "파일포맷": "PNG"}])
    from module.evaluation.v4.keys23 import KEYS
    a = CF.load_answers(p, KEYS, "T-002")
    assert set(a.records) == {"90000003"} and a.records["90000003"]["test"] == "T-002"


def test_scan_accepts_te_name(console, roots):
    c = logged_in(console)
    sc = _scan(c, test="Te-2", work_dirs=[str(console.syn["img"].parent)])
    assert sc["order"] == ["T-002"] and sc["tests"]["T-002"]["n_sets"] == 3


def test_set_id_rules():
    assert CF.choose_id(CF.id_groups(Path("00002_13387752 운수 자판기 (1).pdf"))) == "13387752"
    assert CF.choose_id(CF.id_groups(Path("13178069_1_인도네시아 (309).JPG"))) == "13178069"
    assert CF.choose_id(CF.id_groups(Path("76128.jpg"))) == "76128"
    assert CF.choose_id(["13387752", "20231015"]) == "13387752"
    assert CF.choose_id(["13387752", "20231015"], prefer={"20231015"}) == "20231015"
    assert CF.choose_id(CF.id_groups(Path("13387752/계약서.pdf"))) == "13387752"     # 세트마다 폴더
    assert CF.choose_id(CF.id_groups(Path("계약서.pdf"))) is None


def _write_csv(path: Path, rows, encoding="utf-8-sig"):
    with path.open("w", encoding=encoding, newline="") as f:
        names = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)


def test_answers_csv_korean_and_english_headers(tmp_path):
    p = tmp_path / "정답.csv"
    _write_csv(p, [
        {"원문인덱스": "13387752", "1 저작물명": "운수 자판기", "파일명": "13387752 운수 자판기.pdf", "저작물유형": "어문",
         "저작자": "김은혜; 홍길동", "복제권": "허용", "기타 권리": "(빈값)", "license_start_date": "2009-09-27",
         "keywords": "", "hashtags": "#바다, #하늘", "file_size": "4,530,544"},
        {"원문인덱스": "", "1 저작물명": "ID 없음"},
    ])
    a = CF.load_answers(p, KEYS, "T-001")
    assert a.fmt == "csv" and set(a.records) == {"13387752"} and a.skipped_rows == 1
    k = a.records["13387752"]["keys"]
    assert k["work_title"] == {"value": "운수 자판기", "state": "value"}
    assert k["creator"]["value"] == ["김은혜", "홍길동"]
    assert k["other_rights"]["state"] == "empty"
    assert k["keywords"]["state"] == "na" and k["keywords"]["value"] is None
    assert k["hashtags"]["value"] == ["바다", "하늘"]
    assert k["file_size"]["value"] == 4530544
    assert k["copyright_holder"]["state"] == "na"
    assert "copyright_holder" in a.missing_columns and "work_title" not in a.missing_columns
    assert all(len(r["keys"]) == 23 for r in a.records.values())


def test_answers_cp949_csv_and_test_filter(tmp_path):
    p = tmp_path / "answers.csv"
    _write_csv(p, [{"set_id": "1001", "test": "T-001", "저작물명": "가"},
                   {"set_id": "2002", "test": "T-002", "설명": "나"}], encoding="cp949")
    assert set(CF.load_answers(p, KEYS, "T-001").records) == {"1001"}
    assert set(CF.load_answers(p, KEYS, "T-002").records) == {"2002"}
    assert set(CF.load_answers(p, KEYS).records) == {"1001", "2002"}


def test_answers_xlsx(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from datetime import datetime
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["set_id", "저작물명", "이용허락 시작일", "파일크기"])
    ws.append([13387752, "운수 자판기", datetime(2009, 9, 27), 1234.0])
    p = tmp_path / "정답.xlsx"
    wb.save(p)
    a = CF.load_answers(p, KEYS)
    k = a.records["13387752"]["keys"]
    assert k["work_title"]["value"] == "운수 자판기"
    assert k["license_start_date"]["value"] == "2009-09-27"
    assert k["file_size"]["value"] == 1234


@pytest.mark.skipif(not GOLD_T001_NEW5.is_file(), reason="gold 없음")
def test_answers_jsonl_passthrough_matches_gold(tmp_path):
    a = CF.load_answers(GOLD_T001_NEW5, KEYS, "T-001")
    g = CF.load_gold_jsonl(GOLD_T001_NEW5, "T-001")
    assert set(a.records) == set(g) and len(g) == 5
    sid = sorted(g)[0]
    assert a.records[sid]["keys"] == g[sid]["keys"]


def test_answers_reject_table_without_id_and_bad_format(tmp_path):
    p = tmp_path / "a.csv"
    _write_csv(p, [{"저작물명": "가"}])
    with pytest.raises(CF.FolderError) as e:
        CF.load_answers(p, KEYS)
    assert e.value.code == "answer_no_id"
    q = tmp_path / "a.txt"
    q.write_text("x", encoding="utf-8")
    with pytest.raises(CF.FolderError):
        CF.load_answers(q, KEYS)


# ---------------------------------------------------------------------------
# 짝 맞추기
# ---------------------------------------------------------------------------
def _rec(sid, file_name=None, title=None):
    keys = {k.name: {"value": None, "state": "na"} for k in KEYS}
    if file_name:
        keys["file_name"] = {"value": file_name, "state": "value"}
    if title:
        keys["work_title"] = {"value": title, "state": "value"}
    return {"set_id": sid, "keys": keys}


def test_build_sets_t002_pairs_duplicates_and_unkeyed(tmp_path):
    w, c = tmp_path / "w", tmp_path / "c"
    w.mkdir(), c.mkdir()
    for n in ("13385261_1_템플릿.pptx", "13385261_2_템플릿.jpg", "90000001_1_사진.jpg", "90000002.mp4", "99999999.jpg"):
        (w / n).write_bytes(b"x")
    (c / "00002_13385261 템플릿.pdf").write_bytes(b"%PDF")
    answers = {"13385261": _rec("13385261", "13385261_1_템플릿.pptx", "템플릿"), "90000001": _rec("90000001"),
               "90000002": _rec("90000002"), "90000009": _rec("90000009")}
    works = list(CF.walk_files([w], CA.WORK_EXT))
    contracts = list(CF.walk_files([c], CA.CONTRACT_EXT))
    ts = CF.build_sets("T-002", contracts, works, answers, media_of=_media, manifest_entry=ManifestEntry)
    by = {e.set_id: e for e in ts.entries}
    assert set(by) == {"13385261", "90000001", "90000002"}
    assert Path(by["13385261"].work).name == "13385261_1_템플릿.pptx"          # 정답 파일명으로 고름
    assert Path(by["13385261"].contract).name == "00002_13385261 템플릿.pdf"
    assert by["13385261"].extra["title"] == "템플릿" and by["90000002"].media == "video"
    assert ts.paired == 1 and ts.scored == 3
    assert ts.missing == [{"set_id": "90000009", "reason": "저작물 파일 없음"}]
    assert ts.duplicates[0]["by_answer"] is True and ts.unkeyed == ["99999999"]
    ts2 = CF.build_sets("T-002", contracts, works, answers, include_unkeyed=True, media_of=_media, manifest_entry=ManifestEntry)
    assert "99999999" in {e.set_id for e in ts2.entries} and ts2.scored == 3
    ts3 = CF.build_sets("T-002", contracts, works, {}, media_of=_media, manifest_entry=ManifestEntry)
    assert {e.set_id for e in ts3.entries} == {"13385261", "90000001", "90000002", "99999999"} and ts3.scored == 0


def test_build_sets_same_folder_and_t001_doc_first(tmp_path):
    d = tmp_path / "mix"
    d.mkdir()
    for n in ("13387752 계약서.pdf", "13387752 계약서 스캔.png", "13387752_1_작품.jpg"):
        (d / n).write_bytes(b"x")
    files_c = list(CF.walk_files([d], CA.CONTRACT_EXT))
    files_w = list(CF.walk_files([d], CA.WORK_EXT))
    t1 = CF.build_sets("T-001", files_c, [], {}, media_of=_media, manifest_entry=ManifestEntry)
    assert [Path(e.contract).name for e in t1.entries] == ["13387752 계약서.pdf"]      # 문서 형식 먼저
    t2 = CF.build_sets("T-002", files_c, files_w, {"13387752": _rec("13387752", "13387752_1_작품.jpg")},
                       media_of=_media, manifest_entry=ManifestEntry)
    e = t2.entries[0]
    assert Path(e.work).name == "13387752_1_작품.jpg" and Path(e.contract).name == "13387752 계약서.pdf"


# ---------------------------------------------------------------------------
# API — 등록 확인 · 시작 · 둘 다 · 취소
# ---------------------------------------------------------------------------
def _scan(c, **body):
    r = c.post("/api/test/folder/scan", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_fs_endpoints_and_off_switch(console, roots, monkeypatch):
    c = logged_in(console)
    r = c.get("/api/test/fs/roots").json()
    assert r["enabled"] and r["roots"][0]["path"] == CF.display_path(os.path.realpath(roots))
    assert c.get("/api/test/fs/list").status_code == 200
    assert c.get("/api/test/fs/list", params={"path": "/etc"}).status_code == 403
    monkeypatch.setenv(CF.ENV_ROOTS, "")
    assert c.get("/api/test/fs/roots").json()["enabled"] is False
    r = c.post("/api/test/folder/scan", json={"test": "T-002", "work_dirs": [str(roots)]})
    assert r.status_code == 403 and r.json()["error"] == "folders_off"


def test_scan_errors(console, roots):
    c = logged_in(console)
    assert c.post("/api/test/folder/scan", json={"test": "T-009"}).status_code == 400
    assert c.post("/api/test/folder/scan", json={"test": "T-002"}).json()["error"] == "no_work_dir"
    assert c.post("/api/test/folder/scan", json={"test": "T-001"}).json()["error"] == "no_contract_dir"
    r = c.post("/api/test/folder/scan", json={"test": "T-002", "work_dirs": ["/etc"]})
    assert r.status_code == 403
    r = c.post("/api/test/folder/start", json={"scan_id": "nope"})
    assert r.status_code == 404 and r.json()["error"] == "scan_expired"


def test_t002_folder_run_builtin_answers(console, roots):
    syn_works = console.syn["img"].parent                            # tmp/works (허용 위치 안)
    c = logged_in(console)
    sc = _scan(c, test="T-002", work_dirs=[str(syn_works)], answers={"T-002": {"mode": "builtin"}})
    v = sc["tests"]["T-002"]
    assert v["n_sets"] == 3 and v["n_scored"] == 3 and v["media"] == {"image": 2, "video": 1}
    assert v["answer"]["mode"] == "builtin" and v["answer"]["n"] == 3 and v["n_missing"] == 0
    assert v["estimate"]["krw"] > 0 and len(v["sample"]) == 3
    r = c.post("/api/test/folder/start", json={"scan_id": sc["scan_id"], "name": "폴더 점검"})
    assert r.status_code == 200, r.text
    rid = r.json()["run_id"]
    d = wait_done(c, rid)
    assert d["state"] == "done" and d["source"] == "folder" and d["n_sets"] == 3
    assert d["config"]["source_label"] == "폴더로 시험" and d["config"]["answer"]["mode"] == "builtin"
    assert d["macro_f1"] is not None
    reg = json.loads((console.runs / rid / "registration.json").read_text(encoding="utf-8"))
    assert reg["n_sets"] == 3 and reg["folders"]["work_dirs"]
    assert c.post("/api/test/folder/start", json={"scan_id": sc["scan_id"]}).status_code == 404   # 한 번만 쓴다


def test_t002_folder_run_with_csv_answers_scores_like_jsonl(console, roots):
    syn = console.syn
    gold = [json.loads(x) for x in syn["gold"].read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = []
    for g in gold:
        k = g["keys"]
        rows.append({"set_id": g["set_id"], "test": "T-002", "media": g["media"], "bucket": g["bucket"],
                     "파일명": k["file_name"]["value"],
                     "키워드": ";".join(k["keywords"]["value"] or []) if k["keywords"]["state"] == "value"
                     else ("(빈값)" if k["keywords"]["state"] == "empty" else ""),
                     "해시태그": ";".join(k["hashtags"]["value"] or []), "파일포맷": k["file_format"]["value"],
                     "파일크기": k["file_size"]["value"], "해상도": k["resolution"]["value"],
                     "설명": k["description"]["value"], "Hash 데이터": k["file_hash"]["value"]})
    ans = roots / "정답_T-002.csv"
    _write_csv(ans, rows)
    c = logged_in(console)
    sc = _scan(c, test="T-002", work_dirs=[str(syn["img"].parent)], answers={"T-002": {"mode": "file", "path": str(ans)}})
    v = sc["tests"]["T-002"]
    # 최종 절차서: Te-2 는 1~23번 — 저작물 열만 있는 표는 1~16번 열(파일명 제외)이 없다고 알린다
    from module.evaluation.v4.keys23 import CONTRACT_ITEM_KEYS
    assert v["answer"]["format"] == "csv" and v["n_scored"] == 3
    assert v["answer"]["missing_columns"] == list(CONTRACT_ITEM_KEYS)
    rid = c.post("/api/test/folder/start", json={"scan_id": sc["scan_id"]}).json()["run_id"]
    d1 = wait_done(c, rid)
    assert (console.runs / rid / "answers.jsonl").is_file() and (console.runs / rid / "answers_source.csv").is_file()
    # 같은 정답을 기본(jsonl)으로 돌린 결과와 같은 점수
    sc2 = _scan(c, test="T-002", work_dirs=[str(syn["img"].parent)], answers={"T-002": {"mode": "builtin"}})
    d2 = wait_done(c, c.post("/api/test/folder/start", json={"scan_id": sc2["scan_id"]}).json()["run_id"])
    assert d1["macro_f1"] == pytest.approx(d2["macro_f1"])


def test_no_answer_runs_unscored(console, roots):
    c = logged_in(console)
    sc = _scan(c, test="T-002", work_dirs=[str(console.syn["img"].parent)], answers={"T-002": {"mode": "none"}})
    assert sc["tests"]["T-002"]["n_scored"] == 0 and sc["tests"]["T-002"]["n_sets"] == 3
    d = wait_done(c, c.post("/api/test/folder/start", json={"scan_id": sc["scan_id"]}).json()["run_id"])
    assert d["state"] == "done" and d["gold"] is False and d["macro_f1"] is None


@needs_contract
def test_both_runs_t001_and_t002_in_parallel(console, roots):
    cdir = roots / "계약서"
    cdir.mkdir()
    shutil.copy2(NEW_CONTRACT, cdir / NEW_CONTRACT.name)
    c = logged_in(console)
    sc = _scan(c, test="BOTH", contract_dirs=[str(cdir)], work_dirs=[str(console.syn["img"].parent)],
               answers={"T-001": {"mode": "builtin"}, "T-002": {"mode": "builtin"}})
    assert sc["order"] == ["T-001", "T-002"]
    t1 = sc["tests"]["T-001"]
    assert t1["n_sets"] == 1 and t1["n_scored"] == 1 and t1["n_missing"] == 4      # 기본 정답 5건 중 파일은 1건
    console.block = threading.Event()                                    # 두 실행 모두 첫 세트에서 멈춰 둔다
    r = c.post("/api/test/folder/start", json={"scan_id": sc["scan_id"], "name": "둘 다"}).json()
    assert r["parallel"] is True and [x["test"] for x in r["runs"]] == ["T-001", "T-002"]
    a, b = (x["run_id"] for x in r["runs"])
    wait_state(c, a, ("running",))
    wait_state(c, b, ("running",))                                       # T-001 이 끝나기 전에 T-002 도 돈다
    act = c.get("/api/test/status").json()["active"]
    assert act["batch"] == a and act["batches"] == [{"run_id": a, "test": "T-001"}, {"run_id": b, "test": "T-002"}]
    console.block.set()
    da = wait_done(c, a, timeout=120)
    db = wait_done(c, b, timeout=120)
    assert da["state"] == "done" and db["state"] == "done"
    assert da["config"]["parallel"] == [{"run_id": b, "test": "T-002"}]
    assert db["config"]["parallel"] == [{"run_id": a, "test": "T-001"}]
    assert "next" not in da["config"] and "after" not in db["config"]
    assert da["name"] == "둘 다 · Te-1" and db["name"] == "둘 다 · Te-2"          # 화면 이름은 절차서를 따른다
    assert db["started_at"] <= da["finished_at"]                      # 초 단위 시각 — 동시 실행은 위의 running 두 건으로 확인
    assert c.get("/api/test/status").json()["active"] == {"batch": None, "batches": [], "single": None}


def test_cancel_one_parallel_run_keeps_the_other(console, roots):
    cdir = roots / "계약서"
    cdir.mkdir()
    (cdir / "90000001 계약서.pdf").write_bytes(b"%PDF-1.4\n")          # 정답 없이 → 채점 없는 T-001 1건
    console.block = threading.Event()
    c = logged_in(console)
    sc = _scan(c, test="BOTH", contract_dirs=[str(cdir)], work_dirs=[str(console.syn["img"].parent)],
               answers={"T-001": {"mode": "none"}, "T-002": {"mode": "builtin"}})
    r = c.post("/api/test/folder/start", json={"scan_id": sc["scan_id"], "workers": {"T-002": 1}}).json()
    a, b = (x["run_id"] for x in r["runs"])
    wait_state(c, a, ("running",))
    wait_state(c, b, ("running",))
    busy = c.post("/api/test/folder/start", json={"scan_id": _scan(c, test="T-002", work_dirs=[str(console.syn["img"].parent)])["scan_id"]})
    assert busy.status_code == 409 and busy.json()["active"] == a              # 동시 실행 중에는 새 일괄 시험을 받지 않는다
    assert c.post(f"/api/test/runs/{b}/cancel").json()["ok"] is True
    console.block.set()
    da, db = wait_done(c, a), wait_done(c, b)
    assert da["state"] == "done"                                         # 다른 쪽 취소와 상관없이 끝까지 돈다
    assert db["state"] == "cancelled" and db["done"] < 3


def test_retry_parallel_pair_reruns_both(console, roots):
    cdir = roots / "계약서"
    cdir.mkdir()
    (cdir / "90000001 계약서.pdf").write_bytes(b"%PDF-1.4\n")
    c = logged_in(console)
    console.fail = True                                                      # 첫 회차: 두 시험 모두 실패
    sc = _scan(c, test="BOTH", contract_dirs=[str(cdir)], work_dirs=[str(console.syn["img"].parent)],
               answers={"T-001": {"mode": "none"}, "T-002": {"mode": "builtin"}})
    r = c.post("/api/test/folder/start", json={"scan_id": sc["scan_id"]}).json()
    a, b = (x["run_id"] for x in r["runs"])
    da, db = wait_done(c, a), wait_done(c, b)
    assert da["remaining"]["failed"] == 1 and db["remaining"]["failed"] == 3
    console.fail = False
    rr = c.post(f"/api/test/runs/{a}/retry", json={}).json()                # 함께 실행한 Te-2 도 같이 다시
    assert [(x["run_id"], x["failed"]) for x in rr["runs"]] == [(a, 1), (b, 3)]
    da, db = wait_done(c, a), wait_done(c, b)
    assert da["remaining"]["failed"] == 0 and db["remaining"]["failed"] == 0
    assert da["state"] == "done" and db["state"] == "done"
