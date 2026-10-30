"""시험 콘솔 — 절차서 3~7단계 화면에 필요한 값: 시험 대상 합계(불러오기), 속성정보 파일(*.json) 개수·폴더·묶음 내려받기.

모델 호출 없음: test_m3_console 의 가짜 오케스트레이터를 쓴다 (test_m3_console_folder 와 같은 방식).
"""
from __future__ import annotations

import io
import json
import threading
import zipfile

from .test_m3_console import Console, _httpx_testclient_compat, console, logged_in, wait_done, wait_state  # noqa: F401
from .test_m3_console_folder import _scan, roots  # noqa: F401

from web import console_api as CA                             # noqa: E402
from web import console_folder as CF                          # noqa: E402


def test_scan_counts_distinct_items_across_both_tests(console, roots):
    cdir = roots / "계약서"
    cdir.mkdir()
    (cdir / "90000001 계약서.pdf").write_bytes(b"%PDF-1.4\n")          # Te-2 저작물과 같은 ID
    (cdir / "90000009 계약서.pdf").write_bytes(b"%PDF-1.4\n")          # Te-1 에만 있는 세트 (어문처럼)
    c = logged_in(console)
    sc = _scan(c, test="BOTH", contract_dirs=[str(cdir)], work_dirs=[str(console.syn["img"].parent)],
               answers={"T-001": {"mode": "none"}, "T-002": {"mode": "builtin"}})
    assert sc["tests"]["T-001"]["n_sets"] == 2 and sc["tests"]["T-002"]["n_sets"] == 3
    assert sc["n_items"] == 4                                          # 2 + 3 이 아니라 합집합 — 90000001 은 한 번
    assert sc["items_media"] == {"image": 2, "video": 1, "text": 1}    # 겹치는 세트는 저작물(Te-2) 쪽 매체
    r = c.post("/api/test/folder/start", json={"scan_id": sc["scan_id"]}).json()
    a, b = (x["run_id"] for x in r["runs"])
    da, db = wait_done(c, a), wait_done(c, b)
    for d in (da, db):                                                 # 진행 · 완료 화면이 같은 합계를 쓴다
        assert d["config"]["n_items"] == 4 and d["config"]["items_media"] == {"image": 2, "video": 1, "text": 1}
    assert db["config"]["media"] == {"image": 2, "video": 1}
    assert da["config"]["media"] == {"text": 2}                        # 정답 없이 → Te-1 은 계약서만 보고 '어문'으로 센다
    one = _scan(c, test="T-002", work_dirs=[str(console.syn["img"].parent)])
    assert one["n_items"] == 3 and one["n_items"] == one["tests"]["T-002"]["n_sets"]


def test_predictions_count_dir_and_json_zip(console, roots):
    c = logged_in(console)
    console.block = threading.Event()                                   # 첫 세트에서 멈춰 둔다 — 아직 파일 없음
    sc = _scan(c, test="T-002", work_dirs=[str(console.syn["img"].parent)], answers={"T-002": {"mode": "builtin"}})
    rid = c.post("/api/test/folder/start", json={"scan_id": sc["scan_id"]}).json()["run_id"]
    wait_state(c, rid, ("running",))
    d0 = c.get(f"/api/test/runs/{rid}").json()
    assert d0["n_predictions"] == 0 and CA.PREDICTIONS_ZIP not in d0["downloads"]
    r = c.get(f"/api/test/runs/{rid}/download/{CA.PREDICTIONS_ZIP}")
    assert r.status_code == 404
    console.block.set()
    d = wait_done(c, rid)
    assert d["state"] == "done" and d["n_predictions"] == 3
    assert d["predictions_dir"] == CF.display_path(console.runs / rid / "out" / "test_predictions")
    assert CA.PREDICTIONS_ZIP in d["downloads"]
    r = c.get(f"/api/test/runs/{rid}/download/{CA.PREDICTIONS_ZIP}")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert f'{rid}_predictions_json.zip' in r.headers["content-disposition"]
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = sorted(zf.namelist())
    assert names == [f"test_predictions/{s}.json" for s in ("90000001", "90000002", "90000003")]
    for n in names:                                                    # 세트마다 완결된 JSON — 콘솔 상세와 같은 레코드
        rec = json.loads(zf.read(n).decode("utf-8"))
        assert rec["set_id"] == n.rsplit("/", 1)[1][:-5] and "file_hash" in rec


def test_predictions_count_cached_while_running(tmp_path):
    run = CA.TestRun("r1", "batch", tmp_path)
    run.pred_dir.mkdir(parents=True)
    (run.pred_dir / "1.json").write_text("{}", encoding="utf-8")
    run.state = "running"
    assert run.n_predictions() == 1
    (run.pred_dir / "2.json").write_text("{}", encoding="utf-8")
    (run.pred_dir / "3.json.tmp").write_text("{}", encoding="utf-8")
    assert run.n_predictions() == 1                                    # 진행이 그대로면 5초 동안 다시 세지 않는다
    run.progress["i"] = 2
    assert run.n_predictions() == 2                                    # 세트가 끝나면 다시 센다 (.tmp 는 빼고)
    run.state = "done"
    assert run.n_predictions() == 2


def test_set_detail_shows_both_tests_for_the_same_item(console, roots):
    """Te-1 은 1~16번만 채점하므로 상세 화면이 17~23번을 짝 실행(Te-2)의 결과로 채운다. Te-2 는 1~23번을 채점하고
    1~16번은 함께 도는 Te-1 의 결과를 기다려 그대로 쓴다 (같은 계약서를 두 번 처리하지 않는다)."""
    from module.evaluation.v4.keys23 import KEYS
    cdir = roots / "계약서"
    cdir.mkdir()
    (cdir / "90000001 계약서.pdf").write_bytes(b"%PDF-1.4\n")          # Te-1·Te-2 둘 다 있는 항목
    (cdir / "90000009 계약서.pdf").write_bytes(b"%PDF-1.4\n")          # Te-1 에만 있는 항목
    c = logged_in(console)
    sc = _scan(c, test="BOTH", contract_dirs=[str(cdir)], work_dirs=[str(console.syn["img"].parent)],
               answers={"T-001": {"mode": "none"}, "T-002": {"mode": "builtin"}})
    r = c.post("/api/test/folder/start", json={"scan_id": sc["scan_id"]}).json()
    run = {x["test"]: x["run_id"] for x in r["runs"]}
    wait_done(c, run["T-001"]), wait_done(c, run["T-002"])
    own = {t: {k["key"]: k for k in CA._set_detail(CA._get_run(run[t]), "90000001", merge=False)["keys"]} for t in run}

    d1 = c.get(f"/api/test/runs/{run['T-001']}/sets/90000001").json()
    assert d1["partner"]["available"] and d1["partner"]["test"] == "T-002" and d1["partner"]["run_id"] == run["T-002"]
    k1 = {k["key"]: k for k in d1["keys"]}
    for K in KEYS:
        if K.test == "T-002":                                        # 17~23번은 Te-2 실행의 값·판정
            assert k1[K.name]["from_test"] == "T-002" and k1[K.name]["value"] == own["T-002"][K.name]["value"]
            assert k1[K.name]["verdict"] == own["T-002"][K.name]["verdict"]
        else:                                                        # 1~16번은 이 실행(Te-1) 그대로
            assert "from_test" not in k1[K.name] and k1[K.name]["value"] == own["T-001"][K.name]["value"]
    assert any(k1[K.name]["value"] not in (None, "", []) for K in KEYS if K.test == "T-002")   # 예전에는 모두 NULL
    assert d1["record"]["set_id"] == "90000001" and d1["partner"]["record"]["test"] == "T-002"

    d2 = c.get(f"/api/test/runs/{run['T-002']}/sets/90000001").json()   # Te-2 — 1~16번도 자기 값(= Te-1 결과)
    k2 = {k["key"]: k for k in d2["keys"]}
    assert d2["partner"]["test"] == "T-001"
    assert d2["record"]["_meta"]["contract_items"]["source"] == "te1"
    assert d2["record"]["_meta"]["contract_items"]["run_id"] == run["T-001"]
    for K in KEYS:
        if K.test == "T-001" and K.name != "file_name":
            assert "from_test" not in k2[K.name] and k2[K.name]["value"] == own["T-001"][K.name]["value"], K.name
    assert k2["file_name"]["value"] == own["T-002"]["file_name"]["value"]    # 파일명은 저작물 파일
    fin = c.get(f"/api/test/runs/{run['T-002']}").json()["summary"]["evaluation"]
    assert fin["contract_items"]["te1"] == fin["contract_items"]["runs"][run["T-001"]] >= 1

    only = c.get(f"/api/test/runs/{run['T-001']}/sets/90000009").json()  # Te-2 에 없는 항목 — 알리기만 한다
    assert only["partner"] == {"run_id": run["T-002"], "test": "T-002", "available": False}
    assert not any("from_test" in k for k in only["keys"])


def test_te2_after_te1_reuses_its_results_and_processes_the_rest(console, roots):
    """절차서 순서 — Te-1 을 먼저 돌리고 Te-2 를 따로 돌린다. Te-2 의 1~16번: Te-1 이 처리한 같은 계약서는 그 결과,
    Te-1 에 없던 계약서나 이름만 같고 내용이 다른 계약서는 Te-2 가 직접 처리한다. 불러오기 화면은 미리 센다."""
    from module.evaluation.v4.keys23 import CONTRACT_ITEM_KEYS
    te1_dir, te2_dir = roots / "Te-1 계약서", roots / "Te-2 계약서"
    te1_dir.mkdir()
    te2_dir.mkdir()
    (te1_dir / "90000001 계약서.pdf").write_bytes(b"%PDF-1.4\nsame contract\n")
    (te2_dir / "90000001 계약서.pdf").write_bytes(b"%PDF-1.4\nsame contract\n")     # 같은 파일 (다른 폴더)
    (te2_dir / "90000002 계약서.pdf").write_bytes(b"%PDF-1.4\nonly in Te-2\n")      # Te-1 에 없던 계약서
    (te1_dir / "90000003 계약서.pdf").write_bytes(b"%PDF-1.4\nold\n")
    (te2_dir / "90000003 계약서.pdf").write_bytes(b"%PDF-1.4\nnew version\n")          # 이름만 같고 내용이 다르다
    c = logged_in(console)
    s1 = _scan(c, test="T-001", contract_dirs=[str(te1_dir)], answers={"T-001": {"mode": "none"}})
    te1 = c.post("/api/test/folder/start", json={"scan_id": s1["scan_id"]}).json()["run_id"]
    wait_done(c, te1)

    s2 = _scan(c, test="T-002", contract_dirs=[str(te2_dir)], work_dirs=[str(console.syn["img"].parent)],
               answers={"T-002": {"mode": "builtin"}})
    plan = s2["tests"]["T-002"]["contract_items"]
    assert (plan["te1"], plan["process"], plan["no_contract"], plan["partner"]) == (1, 2, 0, 0)
    assert plan["runs"] == [{"run_id": te1, "name": CA._get_run(te1).name, "n": 1}]
    te2 = c.post("/api/test/folder/start", json={"scan_id": s2["scan_id"]}).json()["run_id"]
    d = wait_done(c, te2)
    assert d["config"]["contract_items"]["te1"] == 1
    assert d["summary"]["evaluation"]["contract_items"] == {"te1": 1, "contract_leg": 2, "none": 0, "runs": {te1: 1}}

    rec = lambda rid, sid: json.loads((console.runs / rid / "out" / "test_predictions" / f"{sid}.json").read_text(encoding="utf-8"))   # noqa: E731
    r1, mine = rec(te2, "90000001"), rec(te1, "90000001")
    assert r1["_meta"]["contract_items"]["source"] == "te1" and r1["_meta"]["contract_items"]["run_id"] == te1
    assert all(r1[k] == mine[k] for k in CONTRACT_ITEM_KEYS if k not in ("work_title", "work_type") or mine[k])
    assert r1["file_name"] == "90000001_1_테스트 사진.jpg"
    assert rec(te2, "90000002")["_meta"]["contract_items"]["source"] == "contract_leg"
    assert rec(te2, "90000003")["_meta"]["contract_items"]["source"] == "contract_leg"      # 해시가 달라 다시 처리
    log = (console.runs / te2 / "out" / "test.log").read_text(encoding="utf-8")
    assert f"Te-1 실행 {te1} 의 결과를 씁니다" in log and "Te-1 결과 없음 → 계약서" in log
