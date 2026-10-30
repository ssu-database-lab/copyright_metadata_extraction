"""폴더 시험의 원본 사본 — 원본 폴더를 옮기거나 지워도 콘솔이 계약서 쪽 보기·위치 표시와 저작물 그림을 보여 준다.

2026-10-20: 임시 폴더의 시험 자료가 지워진 실행에서 계약서 미리보기와 노란 칠이 모두 사라졌다(원본 경로만 기록했기 때문).
모델 호출 없음.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

_API_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _API_DIR not in sys.path:
    sys.path.insert(0, _API_DIR)

import fitz                                                    # noqa: E402
from PIL import Image                                          # noqa: E402

from web import console_api as CA                             # noqa: E402
from web.pipeline import PipelineOrchestrator                 # noqa: E402


def _pdf(path: Path, pages: int = 2) -> Path:
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    doc.save(path); doc.close()
    return path


def test_pipeline_keeps_pdf_and_image_preview(tmp_path):
    res = tmp_path / "res"; res.mkdir()
    staged = _pdf(tmp_path / "staged.pdf")
    ctx = {"staged": True, "upload_path": staged, "result_dir": res, "file_size": staged.stat().st_size}
    PipelineOrchestrator._keep_source(ctx)
    assert (res / "source" / "document.pdf").read_bytes() == staged.read_bytes()
    big = tmp_path / "big.png"
    Image.new("RGB", (4000, 2000), "white").save(big)
    PipelineOrchestrator._keep_work_preview({"staged": True, "result_dir": res}, big)
    with Image.open(res / "source" / "work_preview.jpg") as im:
        assert max(im.size) == 1280
    other = tmp_path / "res2"; other.mkdir()                   # 업로드(폴더 시험 아님)는 사본을 만들지 않는다
    PipelineOrchestrator._keep_source({"staged": False, "upload_path": staged, "result_dir": other, "file_size": 1})
    assert not (other / "source").exists()


def test_console_falls_back_to_kept_copy(tmp_path):
    run_dir = tmp_path / "run"; out = run_dir / "out"
    run = SimpleNamespace(dir=run_dir, out_dir=out, run_id="r1", test="T-001")
    leg = out / "runs" / "results" / "20261020_000000_000001"
    orig = _pdf(tmp_path / "11066748 서울대공원24225.pdf", pages=3)
    (leg / "source").mkdir(parents=True)
    _pdf(leg / "source" / "document.pdf", pages=3)
    rec = {"test": "T-001", "_meta": {"test": "T-001", "files": {"contract": str(orig), "work": None},
                                       "request_ids": {"contract": leg.name}}}
    assert CA._source_file(run, rec, "contract") == orig                       # 원본이 있으면 원본
    orig.unlink()
    assert CA._source_file(run, rec, "contract") == leg / "source" / "document.pdf"
    assert [p["page"] for p in CA._previews(run, "11066748", rec)["contract"]["pages"]] == [1, 2, 3]
    rec["_meta"]["request_ids"] = {"contract": "../../escape"}                 # leg 폴더 밖은 보지 않는다
    assert CA._source_file(run, rec, "contract") is None


def test_te2_work_preview_and_paired_contract(tmp_path):
    run_dir = tmp_path / "run"; out = run_dir / "out"
    run = SimpleNamespace(dir=run_dir, out_dir=out, run_id="r2", test="T-002")
    leg = out / "runs" / "results" / "20261020_000000_000002"
    (leg / "source").mkdir(parents=True)
    Image.new("RGB", (640, 480), "white").save(leg / "source" / "work_preview.jpg")
    _pdf(leg / "source" / "contract.pdf", pages=1)
    rec = {"test": "T-002", "_meta": {"test": "T-002", "media": "image",
                                       "files": {"contract": str(tmp_path / "gone.pdf"), "work": str(tmp_path / "gone.tif")},
                                       "request_ids": {"work": leg.name}}}
    pv = CA._previews(run, "90000001", rec)
    assert pv["work"]["image_url"] and len(pv["contract"]["pages"]) == 1
    assert CA._source_file(run, rec, "work") == leg / "source" / "work_preview.jpg"
