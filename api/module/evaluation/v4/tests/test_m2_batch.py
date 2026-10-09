"""M2 배치 측(B1~B5) 단위 테스트 — docs/v4/M2_runner_spec.md §1.B, §2, §3.1.

모델 호출 없음: 오케스트레이터는 가짜(FakeOrch)다. 계약서 PDF 는 저장소의 dataset/new_contracts/*.pdf 를
읽기만 하고(쪽수 세기·sha256), 저작물은 임시 파일이다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from api.module.evaluation import batch_runner as BR
from api.module.evaluation.batch_runner import (BatchRunner, RunConfig, _estimate_cost,
                                                build_prediction_record)
from api.module.evaluation.manifest import (ManifestEntry, effective_test, load_manifest,
                                            validate_manifest, write_manifest)
from api.module.evaluation.manifest_v4 import REPO_ROOT, build_manifest_v4, entries_from_ids_csv
from api.module.evaluation.v4.keys23 import KEY_NAMES, T001_KEYS, T002_KEYS

ROOT = Path(__file__).resolve().parents[5]
T001_NEW5 = ROOT / "dataset/test_set_v4/t001_new5_ids.csv"
T002_IDS = ROOT / "dataset/test_set_v4/t002_ids.csv"
GOLD_T001_NEW5 = ROOT / "dataset/test_set_v4/gold23_t001_new5.jsonl"
NEW_CONTRACT = ROOT / "dataset/new_contracts/00002_13387752 운수 자판기 (1).pdf"

_API_DIR = str(ROOT / "api")
if _API_DIR not in sys.path:           # web.batch_api 는 app.py 처럼 api/ 를 루트로 import 한다
    sys.path.insert(0, _API_DIR)


# ---------------------------------------------------------------------------
# 도구
# ---------------------------------------------------------------------------
def _gold_values() -> Dict[str, Dict[str, Any]]:
    out = {}
    for line in GOLD_T001_NEW5.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["set_id"]] = {k: v.get("value") for k, v in r["keys"].items()}
    return out


class FakeOrch:
    """PipelineOrchestrator 대역 — kwargs 를 기록하고 M2 A1/A2 인터페이스(test_prediction·file_name·file_hash)를 흉내 낸다.

    실제 build_response 와 같은 계약을 지킨다: `ocr_provider` 는 **요청한** 공급자(기본 'alibaba') 그대로,
    실제로 쓰인 공급자는 `ocr_provider_used`(PDF 텍스트층이면 'direct_text') 에 적는다. 실행기의 비용 추정은
    후자를 봐야 한다 — 전자만 보던 버그가 이 가짜가 'direct_text' 를 ocr_provider 에 직접 내서 가려졌다.
    """

    def __init__(self, pred_fn=None, success: bool = True, with_prediction: bool = True,
                 ocr_provider: str = "alibaba", ocr_provider_used: str = "direct_text"):
        self.calls: list = []
        self.pred_fn = pred_fn or (lambda filename: {})
        self.success = success
        self.with_prediction = with_prediction
        self.ocr_provider = ocr_provider
        self.ocr_provider_used = ocr_provider_used

    def run(self, file_bytes: bytes, filename: str, **kw) -> Dict[str, Any]:
        self.calls.append({"filename": filename, **kw})
        low = filename.lower()
        is_pdf = low.endswith(".pdf")
        is_video = low.endswith((".mp4", ".mov", ".avi", ".mkv"))
        resp = {
            "success": self.success, "request_id": f"rid-{len(self.calls)}",
            "filename": filename.replace(" ", "_"),          # 파이프라인의 정제된 이름(공백 → _)
            "file_name": filename, "file_hash": hashlib.sha256(file_bytes).hexdigest(),
            "ocr_provider": self.ocr_provider if is_pdf else ("(video/keyframes)" if is_video else "(image/VLM)"),
            "ocr_provider_used": (self.ocr_provider_used if is_pdf
                                  else ("(video/keyframes)" if is_video else "(image/VLM)")),
            "ocr_pages": 5 if is_pdf else None,
            "ocr_model": None, "model_used": "mock-model", "processing_time": 1.5, "extraction_time": 0.5,
            "consolidate": bool(kw.get("consolidate_work", kw.get("consolidate", True))),
            "metadata": {"work_title": "제목", "author": "저자", "description": "설명문"},
            "consolidated_metadata": None, "error": None if self.success else "mock failure",
            "modality": "document" if is_pdf else ("video" if is_video else "image"),
            "test": kw.get("test"),                            # 파이프라인은 받은 test 를 응답에 되돌려 준다 (A2)
        }
        if is_video:                                           # 실제 응답: frames 목록을 뺀 dict (pipeline.py run_video)
            resp["keyframes"] = {"duration": 7.0, "target": 3, "sampled": 4, "kept": 3, "error": None,
                                 "note": None, "schedule": kw.get("frame_schedule", "v3")}
            resp["frame_schedule"] = kw.get("frame_schedule", "v3")
        if self.with_prediction:
            resp["test_prediction"] = self.pred_fn(filename)
        return resp

    def apply_contract_inheritance(self, wresp, contract_meta):
        return wresp


def _gold_pred_fn():
    gold = _gold_values()

    def fn(filename: str) -> Dict[str, Any]:
        # 파일명은 '00002_13387752 …' 처럼 생성 순번이 앞에 붙는다 — gold 에 있는 토큰을 고른다
        for tok in re.findall(r"\d{5,8}", filename):
            if tok in gold:
                return dict(gold[tok])
        return {}
    return fn


def _t002_pred(filename: str) -> Dict[str, Any]:
    return {"keywords": ["바다", "하늘"], "hashtags": ["#바다", "#하늘"], "file_format": "JPG",
            "file_size": 1234, "resolution": "1920x1080", "description": "푸른 바다 위의 하늘",
            "file_hash": "ab" * 32, "creator": ["VLM 이 지어낸 값"], "work_title": "VLM 제목",
            "_meta": {"model_copyright_holder": ["모델값"]}}


@pytest.fixture
def work_file(tmp_path: Path) -> Path:
    p = tmp_path / "works" / "13178069_1_인도네시아 (309).JPG"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"\x89PNG fake image bytes")
    return p


# ---------------------------------------------------------------------------
# B1 — 매니페스트 검증 (contract_only / work_only / pair / neither)
# ---------------------------------------------------------------------------
def test_validate_manifest_kinds_and_gating(work_file: Path):
    e_contract = ManifestEntry(set_id="c1", contract=str(NEW_CONTRACT), media="text")
    e_work = ManifestEntry(set_id="w1", work=str(work_file), media="image")
    e_pair = ManifestEntry(set_id="p1", work=str(work_file), contract=str(NEW_CONTRACT))
    e_neither = ManifestEntry(set_id="n1")
    assert [e.kind for e in (e_contract, e_work, e_pair, e_neither)] == ["contract_only", "work_only", "pair", "neither"]

    v = validate_manifest([e_contract, e_work, e_pair, e_neither])
    assert (v["contract_only"], v["work_only"], v["pairs"], v["neither"]) == (1, 1, 1, 1)
    assert v["neither_ids"] == ["n1"] and v["ok"] is False
    assert any("계약서도 저작물도 없는" in p for p in v["problems"])

    v = validate_manifest([e_contract, e_work, e_pair])
    assert v["ok"] is True and v["problems"] == []
    assert v["missing_work_count"] == 0 and v["missing_contract_count"] == 0

    # 작업 수준 test 와 구성이 어긋나면 오류 — 요금을 쓰기 전에 잡는다
    v = validate_manifest([e_contract, e_work], test="T-001")
    assert v["ok"] is False and v["t001_without_contract"] == ["w1"]
    v = validate_manifest([e_contract, e_work], test="T-002")
    assert v["ok"] is False and v["t002_without_work"] == ["c1"]
    # 엔트리의 test 가 작업 수준보다 우선한다
    e_work_t2 = ManifestEntry(set_id="w2", work=str(work_file), test="T-002")
    # w2 는 엔트리 test=T-002 이고 저작물이 있으니 통과, c1 은 작업 수준 T-001 이고 계약서가 있으니 통과
    assert validate_manifest([e_contract, e_work_t2], test="T-001")["ok"] is True
    assert validate_manifest([e_work, e_work_t2], test="T-001")["t001_without_contract"] == ["w1"]
    assert validate_manifest([ManifestEntry(set_id="x", contract=str(NEW_CONTRACT), test="T-9")])["invalid_test"] == ["x"]
    assert validate_manifest([e_contract], test="T-9")["ok"] is False

    # 없는 파일
    v = validate_manifest([ManifestEntry(set_id="m", work="/nonexistent/a.jpg", contract="/nonexistent/b.pdf")])
    assert v["missing_work"] == ["m"] and v["missing_contract"] == ["m"] and v["ok"] is False
    # 중복 ID
    assert validate_manifest([e_contract, e_contract])["duplicate_count"] == 1


def test_effective_test_inference(work_file: Path):
    assert effective_test(ManifestEntry(set_id="a", contract="c.pdf")) == "T-001"
    assert effective_test(ManifestEntry(set_id="a", work="w.jpg")) == "T-002"
    assert effective_test(ManifestEntry(set_id="a", work="w.jpg", contract="c.pdf")) == "T-002"
    assert effective_test(ManifestEntry(set_id="a")) is None
    assert effective_test(ManifestEntry(set_id="a", work="w.jpg"), job_test="T-001") == "T-001"
    assert effective_test(ManifestEntry(set_id="a", work="w.jpg", test="T-002"), job_test="T-001") == "T-002"


def test_load_write_manifest_roundtrip(tmp_path: Path):
    entries = [ManifestEntry(set_id="1", contract=str(NEW_CONTRACT), media="text", test="T-001"),
               ManifestEntry(set_id="2", work="/mnt/e/x.jpg", media="image", extra={"title": "x"})]
    p = tmp_path / "m.jsonl"
    assert write_manifest(entries, p) == 2
    lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert "work" not in lines[0] and lines[0]["test"] == "T-001"       # None 키는 쓰지 않는다
    assert "contract" not in lines[1] and "test" not in lines[1] and lines[1]["extra"] == {"title": "x"}
    back = load_manifest(p)
    assert back[0].work is None and back[0].contract == str(NEW_CONTRACT) and back[0].test == "T-001"
    assert back[1].work == "/mnt/e/x.jpg" and back[1].test is None and back[1].extra == {"title": "x"}
    # 예전 형식(work 빈 문자열)도 None 으로 읽는다
    (tmp_path / "old.jsonl").write_text(json.dumps({"set_id": "9", "work": "", "contract": "c.pdf"}) + "\n", encoding="utf-8")
    old = load_manifest(tmp_path / "old.jsonl")[0]
    assert old.work is None and old.kind == "contract_only" and old.contract == str(tmp_path / "c.pdf")


# ---------------------------------------------------------------------------
# B5 — build_manifest_v4
# ---------------------------------------------------------------------------
def test_build_manifest_v4_t001_new5(tmp_path: Path):
    entries = entries_from_ids_csv(T001_NEW5, "T-001")
    assert len(entries) == 5
    assert {e.set_id for e in entries} == {"13387752", "13333800", "13333561", "13304492", "13333560"}
    for e in entries:
        assert e.kind == "contract_only" and e.work is None and e.test == "T-001" and e.media == "text"
        assert Path(e.contract).is_absolute() and e.contract.endswith(".pdf")
        assert "dataset/new_contracts/" in e.contract.replace("\\", "/")
        assert os.path.isfile(e.contract), e.contract
        assert e.license_bucket == "donated" and e.extra["title"]
    assert any(e.contract.endswith("00002_13387752 운수 자판기 (1).pdf") for e in entries)

    out = build_manifest_v4(T001_NEW5, "T-001", tmp_path / "t001" / "manifest.jsonl")
    assert out == tmp_path / "t001" / "manifest.jsonl" and out.exists()
    loaded = load_manifest(out)
    assert len(loaded) == 5 and all(e.work is None for e in loaded)
    v = validate_manifest(loaded, test="T-001")
    assert v["ok"] is True and v["contract_only"] == 5 and v["work_only"] == 0 and v["pairs"] == 0


def test_build_manifest_v4_t002_first3(tmp_path: Path):
    import csv
    with open(T002_IDS, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))[:3]
    entries = entries_from_ids_csv(T002_IDS, "T-002", limit=3)
    assert [e.set_id for e in entries] == [r["work_id"] for r in rows]
    for e, r in zip(entries, rows):
        assert e.test == "T-002" and e.media == r["media"] and e.kind == "pair"
        assert e.work == r["work_file"]                                   # 이미 절대 경로(/mnt/e/…)
        assert e.contract == str(REPO_ROOT / r["contract_pdf"])           # 상대 → 저장소 루트 기준
        assert e.extra["title"] == r["title"] and e.license_bucket == r["license_bucket"]
    out = build_manifest_v4(T002_IDS, "T-002", tmp_path / "t002.jsonl", limit=3)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3
    # work_ids 로 고르기
    picked = entries_from_ids_csv(T002_IDS, "T-002", work_ids=["13385261", "87248"])
    assert {e.set_id for e in picked} == {"13385261", "87248"}
    assert next(e for e in picked if e.set_id == "13385261").work.lower().endswith(".pptx")
    with pytest.raises(ValueError):
        entries_from_ids_csv(T002_IDS, "T-003")


# ---------------------------------------------------------------------------
# §2 — 예측 레코드 모양
# ---------------------------------------------------------------------------
def test_build_prediction_record_shape():
    full = {k: f"v_{k}" for k in KEY_NAMES}
    rec = build_prediction_record("T-001", "13387752", full, file_name="00002_13387752 운수 자판기 (1).pdf")
    assert set(rec) == {"set_id", "test", *KEY_NAMES}
    assert rec["set_id"] == "13387752" and rec["test"] == "T-001"
    assert all(rec[k] is None for k in T002_KEYS)                     # T-001 → 17~23 null
    assert rec["other_rights"] is None                                 # 상태 ③ → 항상 null
    assert rec["work_title"] == "v_work_title" and rec["creator"] == "v_creator"
    assert rec["file_name"] == "00002_13387752 운수 자판기 (1).pdf"      # 원본 파일명이 응답값을 이긴다

    rec2 = build_prediction_record("T-002", "13178069", _t002_pred("x"), file_name="13178069_1_인도네시아 (309).JPG",
                                   contract_context={"title": "인도네시아 (309)", "work_type": "사진", "source": "rules"})
    assert set(rec2) == {"set_id", "test", *KEY_NAMES}
    assert rec2["keywords"] == ["바다", "하늘"] and rec2["file_size"] == 1234 and rec2["file_hash"] == "ab" * 32
    assert rec2["work_title"] == "인도네시아 (309)" and rec2["work_type"] == "사진"   # 계약서 규칙 파싱에서
    assert rec2["creator"] is None and rec2["reproduction_right"] is None           # 나머지 1~16 은 null
    assert rec2["file_name"] == "13178069_1_인도네시아 (309).JPG"
    assert "_meta" not in rec2                                                        # _meta 는 실행기가 붙인다

    rec3 = build_prediction_record("T-002", "1", _t002_pred("x"))
    assert rec3["work_title"] is None and rec3["work_type"] is None
    rec4 = build_prediction_record(None, "1", full)
    assert all(rec4[k] is None for k in KEY_NAMES)


# ---------------------------------------------------------------------------
# B4 — 비용 추정 (direct_text 는 OCR 쪽당 요금 없음, token_usage.cost_krw 분기 제거)
# ---------------------------------------------------------------------------
def test_estimate_cost_direct_text_and_consolidation():
    direct = {"success": True, "ocr_provider": "direct_text"}
    assert _estimate_cost(direct, "text", True, pages=5) == pytest.approx(BR.COST_LLM_EXTRACT + BR.COST_CONSOLIDATION_DOC)
    ocr = {"success": True, "ocr_provider": "alibaba"}
    assert _estimate_cost(ocr, "text", True, pages=5) == pytest.approx(5 * BR.COST_OCR_PER_PAGE + BR.COST_LLM_EXTRACT + BR.COST_CONSOLIDATION_DOC)
    nested = {"success": True, "ocr_provider": "alibaba", "ocr_result": {"ocr_provider": "direct_text"}}
    assert _estimate_cost(nested, "text", True, pages=5) == pytest.approx(BR.COST_LLM_EXTRACT + BR.COST_CONSOLIDATION_DOC)
    # 죽은 분기 제거 — 응답의 token_usage.cost_krw 는 더 이상 믿지 않는다
    assert _estimate_cost({"success": True, "ocr_provider": "direct_text", "token_usage": {"cost_krw": 999}}, "text", True, pages=1) != 999
    # 요청한 통합검증 여부
    assert _estimate_cost(direct, "text", True, pages=5, consolidated=False) == pytest.approx(BR.COST_LLM_EXTRACT)
    assert _estimate_cost({"success": True}, "image", False, consolidated=False) == pytest.approx(BR.COST_VLM_IMAGE)
    assert _estimate_cost({"success": True}, "video", False, consolidated=False) == pytest.approx(BR.COST_VLM_VIDEO)
    assert _estimate_cost({"success": True}, "image", False) == pytest.approx(BR.COST_VLM_IMAGE + BR.COST_CONSOLIDATION_IMG)
    assert _estimate_cost({"success": True, "consolidate": False}, "image", False) == pytest.approx(BR.COST_VLM_IMAGE)
    # 실패 응답은 모델을 부르지 않았다
    assert _estimate_cost({"success": False}, "image", False) == 0.0
    assert _estimate_cost({"success": False, "ocr_text": "abc"}, "text", True) == pytest.approx(BR.COST_OCR_PER_PAGE)


def test_estimate_cost_reads_ocr_provider_used_from_real_pipeline_shape():
    """회귀(BUG): 실제 파이프라인은 ocr_provider 에 요청값('alibaba')을 두고 ocr_provider_used 에 'direct_text' 를 적는다.

    실측 T-001 leg(13387752): ocr_provider_used='direct_text' 인데 cost_krw=58.3 (=2.7×5 + 10.2 + 34.6) — 44.8 이어야 한다.
    """
    real = {"success": True, "ocr_provider": "alibaba", "ocr_provider_used": "direct_text", "ocr_pages": 5,
            "consolidate": True}
    assert BR._is_direct_text(real) is True
    assert _estimate_cost(real, "text", True, pages=5, consolidated=True) == pytest.approx(44.8)
    assert _estimate_cost(real, "text", True, pages=5, consolidated=True) == pytest.approx(BR.COST_LLM_EXTRACT + BR.COST_CONSOLIDATION_DOC)
    # 실제로 OCR 모델을 부른 경우(스캔 PDF)는 쪽당 요금이 붙는다
    scanned = {"success": True, "ocr_provider": "alibaba", "ocr_provider_used": "alibaba", "ocr_pages": 5}
    assert BR._is_direct_text(scanned) is False
    assert _estimate_cost(scanned, "text", True, pages=5) == pytest.approx(5 * BR.COST_OCR_PER_PAGE + BR.COST_LLM_EXTRACT + BR.COST_CONSOLIDATION_DOC)
    # 옛 모양(ocr_provider 자체가 direct_text)·ocr_result 중첩도 계속 인식한다
    assert BR._is_direct_text({"ocr_provider": "direct_text"}) is True
    assert BR._is_direct_text({"ocr_result": {"ocr_provider": "direct_text"}}) is True
    assert BR._is_direct_text({}) is False and BR._is_direct_text(None) is False


def test_leg_info_records_provider_used_and_keyframe_summary():
    """회귀(NIT): response['keyframes'] 는 frames 를 뺀 dict 라 len() 기반 요약은 항상 None 이었다."""
    resp = {"success": True, "request_id": "r", "ocr_provider": "alibaba", "ocr_provider_used": "direct_text",
            "ocr_pages": 5, "keyframes": {"duration": 61.0, "target": 8, "sampled": 10, "kept": 8, "error": None,
                                           "note": None, "schedule": "v4", "frames": "(없음)"}}
    info = BR._leg_info(resp, 1.234)
    assert info["ocr_provider"] == "alibaba" and info["ocr_provider_used"] == "direct_text" and info["ocr_pages"] == 5
    assert info["keyframes"] == {"kept": 8, "target": 8, "sampled": 10, "duration": 61.0, "schedule": "v4"}
    assert info["elapsed_sec"] == 1.23
    assert BR._leg_info({"keyframes": [1, 2, 3]}, 0)["keyframes"]["kept"] == 3       # 옛 모양(목록)
    assert BR._leg_info({"keyframes": None}, 0)["keyframes"] is None
    assert BR._leg_info(None, 0)["ocr_provider_used"] is None


def test_context_payload_requires_title_or_work_type():
    assert BR._context_payload(None) is None
    assert BR._context_payload({"title": None, "work_type": None, "source": "rules"}) is None
    assert BR._context_payload({"title": "", "work_type": "", "source": "rules"}) is None
    assert BR._context_payload({"title": "운수 자판기", "work_type": None}) == {"title": "운수 자판기", "work_type": None}
    assert BR._context_payload({"work_type": "사진", "source": "rules"}) == {"title": None, "work_type": "사진"}


# ---------------------------------------------------------------------------
# B2/B3 — eval23 T-001 계약서 단독: 레코드·test.log·finalize(M1 평가기)
# ---------------------------------------------------------------------------
def test_runner_eval23_t001_contract_only_end_to_end(tmp_path: Path):
    out = tmp_path / "t001"
    entries = load_manifest(build_manifest_v4(T001_NEW5, "T-001", out / "manifest.jsonl"))
    orch = FakeOrch(pred_fn=_gold_pred_fn())
    cfg = RunConfig(workers=2, scoring_set="eval23", test="T-001", gold=str(GOLD_T001_NEW5))
    runner = BatchRunner(orch, cfg, out)
    summary = runner.run(entries, {})
    assert summary["ok"] == 5 and summary["failed"] == 0
    assert summary["predictions_dir"] == str(out / "test_predictions")
    # 계약서 leg 만 불렀고, v3 kwargs 그대로다(저작물 전용 kwargs 없음)
    assert len(orch.calls) == 5
    for c in orch.calls:
        assert c["document_type"] == "저작재산권 이용허락 계약서" and "consolidate_work" not in c and "vlm_prefer" not in c
        assert " " in c["filename"]                                   # 원본 basename(공백 포함)을 넘긴다
        assert c["test"] == "T-001"                                    # 파이프라인 쪽 test_prediction 도 같은 규칙을 따르게
    # 비용: 텍스트층(ocr_provider_used == direct_text; ocr_provider 는 요청값 'alibaba') → OCR 쪽당 요금 없음
    assert summary["cost_krw"] == pytest.approx(5 * (BR.COST_LLM_EXTRACT + BR.COST_CONSOLIDATION_DOC), abs=0.1)

    # 출력 배치
    pred_dir = out / "test_predictions"
    files = sorted(pred_dir.glob("*.json"))
    assert [f.stem for f in files] == sorted(e.set_id for e in entries)
    assert len((out / "test_predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 5
    rec = json.loads((pred_dir / "13387752.json").read_text(encoding="utf-8"))
    assert set(rec) == {"set_id", "test", "_meta", *KEY_NAMES}
    assert rec["test"] == "T-001" and all(rec[k] is None for k in T002_KEYS)
    assert rec["file_name"] == "00002_13387752 운수 자판기 (1).pdf" and rec["creator"] == ["김은혜"]
    meta = rec["_meta"]
    assert meta["ok"] is True and meta["error"] is None
    assert meta["legs"]["contract"]["request_id"].startswith("rid-") and meta["legs"]["contract"]["elapsed_sec"] >= 0
    assert meta["request_ids"] == {"contract": meta["legs"]["contract"]["request_id"]}
    assert meta["file_sha256"] == hashlib.sha256(NEW_CONTRACT.read_bytes()).hexdigest()
    assert meta["models"]["extraction"] == cfg.model_name and meta["models"]["consolidation"] == cfg.consolidation_model
    assert meta["prompt"] == {"contract_context_used": False, "frame_schedule": None, "consolidate_work": None}
    assert meta["files"]["contract"].endswith(".pdf") and meta["files"]["work"] is None

    # results.jsonl — 재개 규칙(ok=True 만 완료) 그대로
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5 and all(r["ok"] and r["scoring_set"] == "eval23" and r["test"] == "T-001" for r in rows)
    assert all("accuracy" not in r and set(r["extracted"]) == set(KEY_NAMES) for r in rows)
    assert rows[0]["stages"]["contract"]["ocr_provider"] == "alibaba"            # 요청한 공급자
    assert rows[0]["stages"]["contract"]["ocr_provider_used"] == "direct_text"   # 실제로 쓰인 공급자
    assert rows[0]["stages"]["contract"]["ocr_pages"] == 5
    assert meta["legs"]["contract"]["ocr_provider_used"] == "direct_text"
    assert runner._completed_ids() == {e.set_id for e in entries}

    # test.log — 한국어 세트별 시작·종료·타이밍, 핸들러는 finalize 뒤에 닫힌다
    log = (out / "test.log").read_text(encoding="utf-8")
    assert "실행기 준비" in log and "배치 시작 — 대상 5세트" in log and "배치 종료 — 성공 5 · 실패 0" in log
    assert "[13387752] 시작 — test=T-001" in log and "[13387752] 종료 — OK" in log and "contract " in log
    assert runner._log_handler is not None                              # eval23 은 finalize 가 닫는다

    # finalize → M1 평가기: 예측 = gold 값이므로 Macro-F1 100%
    fin = runner.finalize_eval23()
    assert fin["evaluated"] is True and fin["test"] == "T-001" and fin["predictions"] == 5 and fin["predictions_ok"] == 5
    assert fin["macro_f1"] == pytest.approx(1.0) and fin["micro_f1"] == pytest.approx(1.0) and fin["passed"] is True
    assert fin["per_key_f1"]["work_title"] == pytest.approx(1.0) and "other_rights" not in fin["per_key_f1"] or fin["per_key_f1"].get("other_rights") in (None, 1.0)
    assert (out / "test_metrics.json").exists() and (out / "report.md").exists() and (out / "per_set.jsonl").exists()
    assert (out / "report.json").exists() and json.loads((out / "report.json").read_text(encoding="utf-8"))["evaluated"] is True
    metrics = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["test"] == "T-001" and metrics["warnings"]["missing_predictions"] == []
    assert runner._log_handler is None and not runner.log.handlers
    assert "평가 완료 — T-001 Macro-F1 100.0%" in (out / "test.log").read_text(encoding="utf-8")


def test_runner_eval23_without_gold_writes_summary_report(tmp_path: Path):
    out = tmp_path / "nogold"
    entries = load_manifest(build_manifest_v4(T001_NEW5, "T-001", out / "manifest.jsonl", limit=2))
    runner = BatchRunner(FakeOrch(pred_fn=_gold_pred_fn()), RunConfig(workers=1, scoring_set="eval23"), out)
    runner.run(entries, {})
    fin = runner.finalize_eval23()
    assert fin["evaluated"] is False and fin["test"] == "T-001" and fin["predictions"] == 2   # test 는 레코드에서 추론
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "gold 가 지정되지 않아" in md and "--pred" in md
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["scoring_set"] == "eval23"


# ---------------------------------------------------------------------------
# B2 — eval23 T-002 저작물 leg: consolidate_work=False · contract_context · frame_schedule="v4"
# ---------------------------------------------------------------------------
def test_runner_eval23_t002_work_leg_kwargs_and_context(tmp_path: Path, work_file: Path, monkeypatch):
    monkeypatch.setattr(BR, "parse_contract_context",
                        lambda path: {"title": "인도네시아 (309)", "work_type": "사진", "source": "rules"})
    out = tmp_path / "t002"
    entry = ManifestEntry(set_id="13178069", work=str(work_file), contract=str(NEW_CONTRACT), media="image")
    orch = FakeOrch(pred_fn=_t002_pred)
    runner = BatchRunner(orch, RunConfig(workers=1, scoring_set="eval23", test="T-002"), out)
    summary = runner.run([entry], {})
    assert summary["ok"] == 1 and len(orch.calls) == 1                 # 계약서 leg 는 부르지 않는다(규칙 문맥만)
    c = orch.calls[0]
    assert c["filename"] == "13178069_1_인도네시아 (309).JPG" and c["document_type"] == "기타문서"
    assert c["consolidate_work"] is False and c["frame_schedule"] == "v4" and c["media"] == "image"
    assert c["test"] == "T-002"
    assert c["contract_context"] == {"title": "인도네시아 (309)", "work_type": "사진"}
    assert c["vlm_prefer"] == "gemma"
    assert summary["cost_krw"] == pytest.approx(BR.COST_VLM_IMAGE)      # 중재자 생략 → VLM 요금만

    rec = json.loads((out / "test_predictions" / "13178069.json").read_text(encoding="utf-8"))
    assert rec["test"] == "T-002" and rec["keywords"] == ["바다", "하늘"] and rec["description"] == "푸른 바다 위의 하늘"
    assert rec["work_title"] == "인도네시아 (309)" and rec["work_type"] == "사진"
    assert rec["creator"] is None and rec["file_name"] == "13178069_1_인도네시아 (309).JPG"
    m = rec["_meta"]
    assert m["contract_context"] == {"title": "인도네시아 (309)", "work_type": "사진", "source": "rules"}
    assert m["prompt"] == {"contract_context_used": True, "frame_schedule": "v4", "consolidate_work": False}
    assert m["models"]["consolidation"] is None and m["models"]["vlm_prefer"] == "gemma"
    assert m["pipeline_meta"] == {"model_copyright_holder": ["모델값"]}
    assert m["legs"]["work"]["modality"] == "image" and m["request_ids"] == {"work": "rid-1"}
    assert m["file_sha256"] == hashlib.sha256(work_file.read_bytes()).hexdigest()
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["stages"]["context"]["ok"] is True and rows[0]["stages"]["context"]["title"] == "인도네시아 (309)"
    assert rows[0]["work_request_id"] == "rid-1" and "contract_request_id" not in rows[0]
    runner.finalize_eval23()


def test_runner_eval23_t002_consolidate_work_and_no_parser(tmp_path: Path, work_file: Path, monkeypatch):
    monkeypatch.setattr(BR, "parse_contract_context", None)              # A4 가 아직 없을 때
    monkeypatch.setattr(BR, "_CONTEXT_PARSER_LOADED", True)              # (지연 import 도 막는다 — A4 가 들어와도 같은 시나리오)
    entry = ManifestEntry(set_id="s", work=str(work_file), contract=str(NEW_CONTRACT), media="image")
    orch = FakeOrch(pred_fn=_t002_pred)
    runner = BatchRunner(orch, RunConfig(workers=1, scoring_set="eval23", consolidate_work=True), tmp_path / "o")
    summary = runner.run([entry], {})
    c = orch.calls[0]
    assert c["consolidate_work"] is True and c["contract_context"] is None
    assert summary["cost_krw"] == pytest.approx(BR.COST_VLM_IMAGE + BR.COST_CONSOLIDATION_IMG)
    rec = json.loads((tmp_path / "o" / "test_predictions" / "s.json").read_text(encoding="utf-8"))
    assert rec["test"] == "T-002" and rec["work_title"] is None and rec["_meta"]["prompt"]["contract_context_used"] is False
    runner.close_log()


def test_runner_eval23_t002_empty_rule_parse_means_no_context(tmp_path: Path, work_file: Path, monkeypatch):
    """규칙 파서가 제목·종별을 하나도 못 읽으면 프롬프트는 바뀌지 않는다 — contract_context_used 는 False 여야 한다."""
    monkeypatch.setattr(BR, "parse_contract_context", lambda path: {"title": None, "work_type": None, "source": "rules"})
    entry = ManifestEntry(set_id="e", work=str(work_file), contract=str(NEW_CONTRACT), media="image")
    orch = FakeOrch(pred_fn=_t002_pred)
    runner = BatchRunner(orch, RunConfig(workers=1, scoring_set="eval23", test="T-002"), tmp_path / "e")
    runner.run([entry], {})
    assert orch.calls[0]["contract_context"] is None
    rec = json.loads((tmp_path / "e" / "test_predictions" / "e.json").read_text(encoding="utf-8"))
    assert rec["_meta"]["prompt"]["contract_context_used"] is False
    assert rec["_meta"]["contract_context"] == {"title": None, "work_type": None, "source": "rules"}   # 파싱 결과는 그대로 남긴다
    assert rec["work_title"] is None and rec["work_type"] is None
    runner.close_log()


def test_runner_eval23_t002_video_leg_meta(tmp_path: Path, monkeypatch):
    """영상 leg: 프레임 요약(kept/target/schedule)이 _meta.legs.work.keyframes 에 남고, test·frame_schedule 이 전달된다."""
    monkeypatch.setattr(BR, "parse_contract_context", None)
    monkeypatch.setattr(BR, "_CONTEXT_PARSER_LOADED", True)
    vid = tmp_path / "87248.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftyp fake")
    entry = ManifestEntry(set_id="87248", work=str(vid), media="video", test="T-002")
    orch = FakeOrch(pred_fn=_t002_pred)
    runner = BatchRunner(orch, RunConfig(workers=1, scoring_set="eval23"), tmp_path / "v")
    summary = runner.run([entry], {})
    assert summary["ok"] == 1 and orch.calls[0]["test"] == "T-002" and orch.calls[0]["frame_schedule"] == "v4"
    assert summary["cost_krw"] == pytest.approx(BR.COST_VLM_VIDEO)
    rec = json.loads((tmp_path / "v" / "test_predictions" / "87248.json").read_text(encoding="utf-8"))
    leg = rec["_meta"]["legs"]["work"]
    assert leg["modality"] == "video" and leg["keyframes"] == {"kept": 3, "target": 3, "sampled": 4, "duration": 7.0, "schedule": "v4"}
    assert leg["ocr_provider_used"] == "(video/keyframes)"
    runner.close_log()


def test_runner_eval23_failures_are_recorded_and_retryable(tmp_path: Path, work_file: Path):
    entries = [ManifestEntry(set_id="fail", work=str(work_file), media="image", test="T-002"),
               ManifestEntry(set_id="nopred", contract=str(NEW_CONTRACT), media="text", test="T-001")]
    out = tmp_path / "f"
    # 1) success=False → ok=False, 레코드는 전부 null + _meta.error
    runner = BatchRunner(FakeOrch(success=False), RunConfig(workers=1, scoring_set="eval23"), out)
    summary = runner.run(entries[:1], {})
    assert summary["failed"] == 1
    rec = json.loads((out / "test_predictions" / "fail.json").read_text(encoding="utf-8"))
    assert all(rec[k] is None for k in KEY_NAMES if k != "file_name") and rec["_meta"]["ok"] is False
    assert "저작물 처리 실패" in rec["_meta"]["error"]
    assert runner._completed_ids() == set()                             # 재개 때 다시 시도된다
    runner.close_log()
    # 2) test_prediction 없음(A2 미반영) → 명확한 오류
    runner2 = BatchRunner(FakeOrch(with_prediction=False), RunConfig(workers=1, scoring_set="eval23"), out)
    runner2.run(entries[1:], {})
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["set_id"] == "nopred" and rows[-1]["ok"] is False and "test_prediction" in rows[-1]["error"]
    assert "[nopred] 종료 — FAIL" in (out / "test.log").read_text(encoding="utf-8")
    runner2.close_log()
    # 3) 예외(파일 없음)도 레코드로 남는다
    runner3 = BatchRunner(FakeOrch(), RunConfig(workers=1, scoring_set="eval23"), out)
    runner3.run([ManifestEntry(set_id="gone", work="/nonexistent/x.jpg", test="T-002")], {})
    rec = json.loads((out / "test_predictions" / "gone.json").read_text(encoding="utf-8"))
    assert rec["_meta"]["error"].startswith("FileNotFoundError") and rec["_meta"]["ok"] is False
    runner3.close_log()


# ---------------------------------------------------------------------------
# v3 경로 유지 (plan 채점) + 계약서 단독 세트는 돈을 쓰기 전에 실패
# ---------------------------------------------------------------------------
def test_runner_v3_pair_still_scores_and_contract_only_refused(tmp_path: Path, work_file: Path):
    gt = {"p1": {"attributes": {"제목": {"value": "제목", "tier": "A"}, "저자": {"value": "저자", "tier": "A"}}}}
    entries = [ManifestEntry(set_id="p1", work=str(work_file), contract=str(NEW_CONTRACT), media="image"),
               ManifestEntry(set_id="c1", contract=str(NEW_CONTRACT), media="text")]
    orch = FakeOrch()
    runner = BatchRunner(orch, RunConfig(workers=1, scoring_set="plan"), tmp_path / "v3")
    summary = runner.run(entries, gt)
    assert summary["ok"] == 1 and summary["failed"] == 1
    assert len(orch.calls) == 2                                          # p1 의 계약서+저작물 leg 만; c1 은 호출 없음
    assert all("consolidate_work" not in c and "frame_schedule" not in c for c in orch.calls)
    rows = {json.loads(l)["set_id"]: json.loads(l) for l in (tmp_path / "v3" / "results.jsonl").read_text(encoding="utf-8").splitlines()}
    assert rows["p1"]["ok"] and rows["p1"]["scoring_set"] == "plan" and "per_attr" in rows["p1"] and rows["p1"]["n_scored"] >= 1
    assert rows["c1"]["ok"] is False and "T-001" in rows["c1"]["error"] and rows["c1"]["cost_krw"] == 0.0
    assert not (tmp_path / "v3" / "test_predictions").exists()           # v3 는 23키 레코드를 쓰지 않는다
    assert runner._log_handler is None                                   # v3 는 run() 끝에 로그를 닫는다
    assert "[p1] 종료 — OK" in (tmp_path / "v3" / "test.log").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# B4 — 검증 실패 시 시작 거부 (cli.main · batch_api._run_job)
# ---------------------------------------------------------------------------
def test_cli_refuses_invalid_manifest_and_validates(tmp_path: Path, capsys):
    from api.module.evaluation import cli
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"set_id": "x", "contract": "/nonexistent/c.pdf"}) + "\n", encoding="utf-8")
    out = tmp_path / "out"
    rc = cli.main(["--manifest", str(bad), "--test", "T-001", "--out", str(out)])
    assert rc == 2 and not (out / "runs").exists()                       # 오케스트레이터를 만들지 않았다
    assert "검증 실패 — 실행하지 않습니다" in capsys.readouterr().out
    assert cli.main(["--manifest", str(bad), "--test", "T-001", "--out", str(out), "--validate-only"]) == 1
    # v3 모드도 같은 규칙 (work 없는 세트 → T-002/T-001 추론은 되지만 파일이 없다)
    assert cli.main(["--manifest", str(bad), "--out", str(out)]) == 2

    good = build_manifest_v4(T001_NEW5, "T-001", tmp_path / "good" / "manifest.jsonl")
    assert cli.main(["--manifest", str(good), "--test", "T-001", "--out", str(out), "--validate-only"]) == 0
    text = capsys.readouterr().out
    assert "계약서단독 5" in text and "채점 기준 eval23" in text
    # 기존 플래그는 그대로 파싱된다
    ns = cli.build_parser().parse_args(["--manifest", "m.jsonl", "--scoring-set", "tta", "--workers", "6",
                                        "--max-cost", "60000", "--no-consolidate", "--limit", "20", "--report-only",
                                        "--vlm-prefer", "qwen", "--ocr-provider", "mistral"])
    assert ns.scoring_set == "tta" and ns.workers == 6 and ns.test is None and ns.gold is None
    ns2 = cli.build_parser().parse_args(["--manifest", "m.jsonl", "--test", "T-002", "--gold", "g.jsonl",
                                         "--consolidate-work", "--tau", "0.7"])
    assert ns2.test == "T-002" and ns2.consolidate_work is True and ns2.tau == 0.7 and ns2.workers is None


def test_cli_evaluator_options_reach_evaluate(tmp_path: Path, monkeypatch):
    """--content-check/--embed-cache/--embed-revision/--n-boot 가 RunConfig 를 거쳐 M1 evaluate() 까지 간다 (B3 'τ etc.')."""
    from api.module.evaluation import cli
    from api.module.evaluation.v4 import evaluate as ev_mod
    ns = cli.build_parser().parse_args(["--manifest", "m.jsonl", "--test", "T-002", "--gold", "g.jsonl",
                                        "--content-check", "off", "--embed-cache", "/tmp/c.jsonl",
                                        "--embed-revision", "abc123", "--n-boot", "50"])
    assert ns.content_check == "off" and ns.embed_cache == "/tmp/c.jsonl" and ns.embed_revision == "abc123" and ns.n_boot == 50
    ns0 = cli.build_parser().parse_args(["--manifest", "m.jsonl"])
    assert ns0.content_check == "on" and ns0.embed_cache is None and ns0.embed_revision is None and ns0.n_boot is None
    assert RunConfig().content_check is True and RunConfig().n_boot is None

    seen: Dict[str, Any] = {}

    def fake_evaluate(test, gold, pred, out, **kw):
        seen.update(test=test, gold=str(gold), **kw)
        return {"n_sets": 0, "n_predictions": 0, "macro_f1": None, "micro_f1": None, "passed": False, "keys": {}}

    monkeypatch.setattr(ev_mod, "evaluate", fake_evaluate)
    out = tmp_path / "opts"
    entries = load_manifest(build_manifest_v4(T001_NEW5, "T-001", out / "manifest.jsonl", limit=1))
    cfg = RunConfig(workers=1, scoring_set="eval23", test="T-001", gold=str(GOLD_T001_NEW5), tau=0.7,
                    content_check=False, embed_cache="/tmp/c.jsonl", embed_revision="abc123", n_boot=50)
    runner = BatchRunner(FakeOrch(pred_fn=_gold_pred_fn()), cfg, out)
    runner.run(entries, {})
    fin = runner.finalize_eval23()
    assert fin["evaluated"] is True
    assert seen == {"test": "T-001", "gold": str(GOLD_T001_NEW5), "tau": 0.7, "cells": True, "content_check": False,
                    "embed_cache": "/tmp/c.jsonl", "embed_revision": "abc123", "n_boot": 50}
    assert fin["content_check"] is False and fin["embed_cache"] == "/tmp/c.jsonl" and fin["n_boot"] == 50
    # n_boot 미지정이면 평가기 기본값을 건드리지 않는다
    seen.clear()
    runner2 = BatchRunner(FakeOrch(pred_fn=_gold_pred_fn()),
                          RunConfig(workers=1, scoring_set="eval23", test="T-001", gold=str(GOLD_T001_NEW5)), tmp_path / "opts2")
    runner2.run(entries, {})
    runner2.finalize_eval23()
    assert "n_boot" not in seen and seen["content_check"] is True and seen["embed_cache"] is None


def test_repo_root_does_not_resolve_symlinks():
    """REPO_ROOT 는 import 된 경로 그대로(abspath) — /home/…/copyright → /mnt/c 심볼릭 링크를 풀지 않는다."""
    from api.module.evaluation import manifest_v4
    f = manifest_v4.__file__
    assert REPO_ROOT == Path(os.path.abspath(f)).parents[3]
    assert (REPO_ROOT / "api" / "module" / "evaluation" / "manifest_v4.py").is_file()
    if os.path.realpath(f) != os.path.abspath(f):          # 링크를 통해 import 된 경우에만 구분이 생긴다
        assert REPO_ROOT != Path(f).resolve().parents[3]
    e = entries_from_ids_csv(T001_NEW5, "T-001", limit=1)[0]
    assert e.contract.startswith(str(REPO_ROOT)) and os.path.isfile(e.contract)


def test_cli_report_only_reevaluates_existing_predictions(tmp_path: Path, capsys):
    from api.module.evaluation import cli
    out = tmp_path / "ro"
    m = build_manifest_v4(T001_NEW5, "T-001", out / "manifest.jsonl")
    runner = BatchRunner(FakeOrch(pred_fn=_gold_pred_fn()), RunConfig(workers=1, scoring_set="eval23", test="T-001"), out)
    runner.run(load_manifest(m), {})
    runner.close_log()
    rc = cli.main(["--manifest", str(m), "--test", "T-001", "--out", str(out), "--gold", str(GOLD_T001_NEW5), "--report-only"])
    assert rc == 0 and (out / "test_metrics.json").exists()
    text = capsys.readouterr().out
    assert "Macro-F1 100.0%" in text and "합격" in text


def test_batch_api_run_job_gates_on_validation(tmp_path: Path):
    from web import batch_api
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"set_id": "x", "work": "/nonexistent/w.jpg"}) + "\n", encoding="utf-8")
    job = batch_api.BatchJob("job-bad", tmp_path / "job-bad")
    job.out_dir.mkdir(parents=True)
    batch_api._run_job(job, bad, {"workers": 1, "scoring_set": "eval23", "test": "T-002"})
    assert job.state == "error" and "검증 실패" in job.error and "저작물 파일 없음" in job.error
    assert job.validation["ok"] is False and job.validation["scoring_set"] == "eval23"
    saved = json.loads((job.work_dir / "job.json").read_text(encoding="utf-8"))
    assert saved["validation"]["ok"] is False
    assert [e["type"] for e in job.events_after(0)] == ["validated", "error", "finished"]
    assert not (job.out_dir / "results.jsonl").exists()                   # 세트를 하나도 돌리지 않았다


def test_batch_api_run_job_eval23_happy_path(tmp_path: Path, monkeypatch):
    from web import batch_api
    m = build_manifest_v4(T001_NEW5, "T-001", tmp_path / "sets" / "manifest.jsonl", limit=2)
    monkeypatch.setattr(batch_api, "_ORCH", FakeOrch(pred_fn=_gold_pred_fn()))
    job = batch_api.BatchJob("job-ok", tmp_path / "job-ok")
    cfg = {"workers": 1, "scoring_set": "eval23", "test": "T-001", "gold": str(GOLD_T001_NEW5),
           "consolidate_work": False, "tau": 0.65}
    job.config = dict(cfg)
    batch_api._run_job(job, m, cfg)
    assert job.state == "done", job.error
    assert job.validation["ok"] is True and job.validation["gt_attr_overlap"] == "-"
    assert job.summary["eval23"]["evaluated"] is True and job.summary["aggregate"]["scoring_set"] == "eval23"
    assert (job.out_dir / "test_metrics.json").exists() and (job.out_dir / "report.json").exists()
    assert (job.out_dir / "test_predictions").is_dir() and (job.out_dir / "test.log").exists()
    assert batch_api._scoring_set(job) == "eval23"
    order, leg, field = batch_api._attr_spec(job)
    assert order == list(KEY_NAMES) and leg["work_title"] == "contract" and leg["description"] == "work"
    assert field["file_hash"] == "file_hash"
    # 설정이 사라진(복구된) 작업도 결과 줄의 scoring_set 으로 되짚는다
    job.config = {}
    assert batch_api._scoring_set(job) == "eval23"


# ---------------------------------------------------------------------------
# M2b 수정 1 — 계약서 문맥 파서 지연 import 의 스레드 안전성 (workers > 1) + 세트별 경고
# ---------------------------------------------------------------------------
class _CatchLogs(logging.Handler):
    """특정 로거에 직접 달아 메시지를 모은다 — root 전파 여부(api 패키지가 root 를 건드린다)에 기대지 않는다."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def br_logs():
    h = _CatchLogs()
    lg = BR.logger
    old = lg.level
    lg.addHandler(h)
    lg.setLevel(logging.DEBUG)
    yield h
    lg.removeHandler(h)
    lg.setLevel(old)


def _reset_context_parser(monkeypatch) -> None:
    monkeypatch.setattr(BR, "parse_contract_context", None)
    monkeypatch.setattr(BR, "_CONTEXT_PARSER_LOADED", False)
    monkeypatch.setattr(BR, "_CONTEXT_PARSER_FAILURE", None)


def _rule_ctx(path):
    return {"title": "인도네시아 (309)", "work_type": "사진", "source": "rules"}


def test_load_context_parser_is_thread_safe_with_slow_import(monkeypatch):
    """회귀(RACE): 예전 코드는 import 가 끝나기 전에 적재 플래그를 세웠다 — 느린 import 동안 들어온 스레드는 None 을 받았다."""
    import threading
    import time
    _reset_context_parser(monkeypatch)
    imports: list = []

    def slow_import():
        imports.append(threading.get_ident())
        time.sleep(0.4)                                   # fitz 등을 올리는 수 초를 흉내 낸다
        return _rule_ctx

    monkeypatch.setattr(BR, "_import_context_parser", slow_import)
    n = 8
    gate = threading.Barrier(n)
    got: list = [None] * n

    def worker(i: int) -> None:
        gate.wait()
        got[i] = BR._load_context_parser()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert all(not t.is_alive() for t in threads)
    assert all(fn is _rule_ctx for fn in got), got                       # 아무도 None 을 받지 않는다
    assert len(imports) == 1                                              # import 는 정확히 한 번
    assert BR._CONTEXT_PARSER_LOADED is True and BR.parse_contract_context is _rule_ctx
    assert BR._load_context_parser() is _rule_ctx and len(imports) == 1    # 이후 호출은 잠금 없이 바로


def test_runner_t002_workers_all_get_context_with_slow_import(tmp_path: Path, work_file: Path, monkeypatch):
    """같은 경쟁을 실행기 수준에서: workers=4, 계약서 있는 T-002 세트 4개가 동시에 첫 사용 — 전부 문맥을 받아야 한다."""
    import time
    _reset_context_parser(monkeypatch)

    def slow_import():
        time.sleep(0.4)
        return _rule_ctx

    monkeypatch.setattr(BR, "_import_context_parser", slow_import)
    entries = [ManifestEntry(set_id=f"r{i}", work=str(work_file), contract=str(NEW_CONTRACT), media="image", test="T-002")
               for i in range(4)]
    orch = FakeOrch(pred_fn=_t002_pred)
    runner = BatchRunner(orch, RunConfig(workers=4, scoring_set="eval23", test="T-002"), tmp_path / "race")
    summary = runner.run(entries, {})
    runner.close_log()
    assert summary["ok"] == 4 and len(orch.calls) == 4
    assert all(c["contract_context"] == {"title": "인도네시아 (309)", "work_type": "사진"} for c in orch.calls)
    for e in entries:
        rec = json.loads((tmp_path / "race" / "test_predictions" / f"{e.set_id}.json").read_text(encoding="utf-8"))
        assert rec["work_title"] == "인도네시아 (309)" and rec["work_type"] == "사진"
        assert rec["_meta"]["prompt"]["contract_context_used"] is True and rec["_meta"]["legs"]["context"]["ok"] is True
    assert "얻지 못했습니다" not in (tmp_path / "race" / "test.log").read_text(encoding="utf-8")


def test_load_context_parser_failure_logs_once_or_reraises(monkeypatch, br_logs):
    # 1) 모듈 없음(ImportError) → 경고 한 번, 이후 조용히 None, 플래그는 세우지 않는다
    _reset_context_parser(monkeypatch)
    calls = {"n": 0}

    def missing():
        calls["n"] += 1
        raise ImportError("No module named 'api.module.clip_extraction.contract_context'")

    monkeypatch.setattr(BR, "_import_context_parser", missing)
    assert BR._load_context_parser() is None
    assert BR._load_context_parser() is None
    assert BR._load_context_parser() is None
    warned = [r for r in br_logs.records if "계약서 문맥 파서" in r.getMessage()]
    assert len(warned) == 1 and warned[0].levelno == logging.WARNING and "올릴 수 없습니다" in warned[0].getMessage()
    assert calls["n"] == 1 and BR._CONTEXT_PARSER_LOADED is False and BR.parse_contract_context is None
    assert isinstance(BR._CONTEXT_PARSER_FAILURE, ImportError)

    # 2) 파서 모듈 자체가 깨짐(ImportError 아님) → 한 번 기록하고 다시 던진다; 이후 호출도 조용히 None 이 되지 않는다
    _reset_context_parser(monkeypatch)
    br_logs.records.clear()

    def broken():
        calls["n"] += 1
        raise NameError("name 'fitz' is not defined")

    monkeypatch.setattr(BR, "_import_context_parser", broken)
    with pytest.raises(NameError):
        BR._load_context_parser()
    with pytest.raises(RuntimeError, match="NameError"):
        BR._load_context_parser()
    assert calls["n"] == 2                                                # 두 번째는 다시 import 하지 않는다
    errs = [r for r in br_logs.records if "import 중 예외" in r.getMessage()]
    assert len(errs) == 1 and errs[0].levelno == logging.ERROR
    assert BR._CONTEXT_PARSER_LOADED is False


def test_runner_t002_warns_per_set_when_contract_yields_no_context(tmp_path: Path, work_file: Path, monkeypatch):
    """계약서가 있는데 문맥을 못 얻으면(파서 미적재 / 파싱 실패 / 빈 파싱) 세트마다 WARNING 이 test.log 에 남는다."""
    # (a) 파서 미적재
    monkeypatch.setattr(BR, "parse_contract_context", None)
    monkeypatch.setattr(BR, "_CONTEXT_PARSER_LOADED", True)
    entries = [ManifestEntry(set_id=f"w{i}", work=str(work_file), contract=str(NEW_CONTRACT), media="image", test="T-002")
               for i in range(2)]
    runner = BatchRunner(FakeOrch(pred_fn=_t002_pred), RunConfig(workers=1, scoring_set="eval23"), tmp_path / "nw")
    runner.run(entries, {})
    runner.close_log()
    log = (tmp_path / "nw" / "test.log").read_text(encoding="utf-8")
    for e in entries:
        assert (f"WARNING [{e.set_id}] 계약서 {NEW_CONTRACT.name} 가 있지만 문맥(제목·종별)을 얻지 못했습니다(문맥 파서 미적재)"
                " — 저작물 leg 를 계약서 문맥 없이 실행합니다") in log
    # (b) 파서는 있지만 파싱이 예외를 낸다
    def boom(path):
        raise RuntimeError("PDF 깨짐")

    monkeypatch.setattr(BR, "parse_contract_context", boom)
    runner = BatchRunner(FakeOrch(pred_fn=_t002_pred), RunConfig(workers=1, scoring_set="eval23"), tmp_path / "pf")
    runner.run(entries[:1], {})
    runner.close_log()
    log = (tmp_path / "pf" / "test.log").read_text(encoding="utf-8")
    assert "계약서 문맥 파싱 실패" in log and "WARNING [w0] 계약서" in log and "(규칙 파싱 실패)" in log
    # (c) 파싱은 됐지만 제목·종별이 비었다
    monkeypatch.setattr(BR, "parse_contract_context", lambda path: {"title": None, "work_type": None, "source": "rules"})
    runner = BatchRunner(FakeOrch(pred_fn=_t002_pred), RunConfig(workers=1, scoring_set="eval23"), tmp_path / "ep")
    runner.run(entries[:1], {})
    runner.close_log()
    log = (tmp_path / "ep" / "test.log").read_text(encoding="utf-8")
    assert "WARNING [w0] 계약서" in log and "제목·종별을 하나도 읽지 못했습니다" in log
    # (d) 계약서가 없는 T-002 세트·문맥을 얻은 세트에는 경고가 없다
    monkeypatch.setattr(BR, "parse_contract_context", _rule_ctx)
    runner = BatchRunner(FakeOrch(pred_fn=_t002_pred), RunConfig(workers=1, scoring_set="eval23"), tmp_path / "nc")
    runner.run([ManifestEntry(set_id="x", work=str(work_file), media="image", test="T-002"), entries[0]], {})
    runner.close_log()
    log = (tmp_path / "nc" / "test.log").read_text(encoding="utf-8")
    assert "얻지 못했습니다" not in log and "읽지 못했습니다" not in log


# ---------------------------------------------------------------------------
# M2b 수정 4 — --only-predicted (CLI → RunConfig → finalize_eval23 → evaluate)
# ---------------------------------------------------------------------------
def test_only_predicted_flag_reaches_evaluate_only_when_set(tmp_path: Path, monkeypatch):
    from api.module.evaluation import cli
    from api.module.evaluation.v4 import evaluate as ev_mod
    ns = cli.build_parser().parse_args(["--manifest", "m.jsonl", "--test", "T-001", "--gold", "g.jsonl", "--only-predicted"])
    assert ns.only_predicted is True
    assert cli.build_parser().parse_args(["--manifest", "m.jsonl"]).only_predicted is False
    assert RunConfig().only_predicted is False                              # 기본 off = M1 동작

    seen: Dict[str, Any] = {}

    def fake_evaluate(test, gold, pred, out, **kw):
        seen.clear()
        seen.update(kw)
        return {"n_sets": 2, "n_predictions": 2, "n_gold_dropped": 3, "macro_f1": 1.0, "micro_f1": 1.0, "passed": True, "keys": {}}

    monkeypatch.setattr(ev_mod, "evaluate", fake_evaluate)
    entries = load_manifest(build_manifest_v4(T001_NEW5, "T-001", tmp_path / "op" / "manifest.jsonl", limit=2))
    runner = BatchRunner(FakeOrch(pred_fn=_gold_pred_fn()),
                         RunConfig(workers=1, scoring_set="eval23", test="T-001", gold=str(GOLD_T001_NEW5), only_predicted=True),
                         tmp_path / "op")
    runner.run(entries, {})
    fin = runner.finalize_eval23()
    assert seen["only_predicted"] is True and fin["only_predicted"] is True and fin["n_gold_dropped"] == 3
    assert "gold 범위 예측 있는 세트만(--only-predicted)" in (tmp_path / "op" / "test.log").read_text(encoding="utf-8")
    # 기본값이면 평가기 호출에 키 자체가 없다 (M1 기본 동작을 건드리지 않는다)
    runner2 = BatchRunner(FakeOrch(pred_fn=_gold_pred_fn()),
                          RunConfig(workers=1, scoring_set="eval23", test="T-001", gold=str(GOLD_T001_NEW5)), tmp_path / "op2")
    runner2.run(entries, {})
    fin2 = runner2.finalize_eval23()
    assert "only_predicted" not in seen and fin2["only_predicted"] is False
    assert "gold 범위 전체" in (tmp_path / "op2" / "test.log").read_text(encoding="utf-8")


def test_cli_only_predicted_end_to_end_report_only(tmp_path: Path, capsys):
    """실제 평가기까지: gold 5세트에 예측 2세트 — 기본은 3세트가 누락(FN), --only-predicted 면 2세트만 채점·누락 0."""
    from api.module.evaluation import cli
    out = tmp_path / "op3"
    m = build_manifest_v4(T001_NEW5, "T-001", out / "manifest.jsonl", limit=2)
    runner = BatchRunner(FakeOrch(pred_fn=_gold_pred_fn()), RunConfig(workers=1, scoring_set="eval23", test="T-001"), out)
    runner.run(load_manifest(m), {})
    runner.close_log()
    base = ["--manifest", str(m), "--test", "T-001", "--out", str(out), "--gold", str(GOLD_T001_NEW5), "--report-only"]
    assert cli.main(base) == 0
    metrics = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_sets"] == 5 and len(metrics["warnings"]["missing_predictions"]) == 3 and metrics["n_gold_dropped"] == 0
    assert metrics["keys"]["work_title"]["n_scored"] == 5 and metrics["keys"]["work_title"]["fn"] == 3
    assert metrics["settings"]["only_predicted"] is False
    capsys.readouterr()
    assert cli.main(base + ["--only-predicted"]) == 0
    text = capsys.readouterr().out
    assert "gold 제외 3건 (--only-predicted)" in text and "Macro-F1 100.0%" in text
    metrics = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_sets"] == 2 and metrics["warnings"]["missing_predictions"] == [] and metrics["n_gold_dropped"] == 3
    assert metrics["keys"]["work_title"]["n_scored"] == 2 and metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["settings"]["only_predicted"] is True
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["only_predicted"] is True
    assert "gold 제외 3개 (--only-predicted" in (out / "report.md").read_text(encoding="utf-8")
