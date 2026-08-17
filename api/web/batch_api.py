"""배치 평가 API — 매니페스트 기반 계약서+저작물 세트 일괄 처리.

**왜 별도 엔드포인트가 필요한가**
브라우저가 4,000세트를 하나씩 올리며 몇 시간 도는 구조는 못 쓴다 — 탭을 닫거나
새로고침하면 진행분이 날아가고, 재개할 방법도 없다. 그래서 작업을 서버가 소유한다:
업로드는 1회(ZIP), 실행은 백그라운드 스레드, UI는 진행률만 구독한다. 탭을 닫아도
작업은 계속되고 나중에 다시 붙을 수 있다.

CLI 하네스(`module.evaluation.cli`)와 **같은 BatchRunner**를 쓰므로 UI로 돌리든
CLI로 돌리든 결과가 같다.

엔드포인트:
  POST /api/batch/create          ZIP 업로드 또는 서버경로 → 작업 생성
  GET  /api/batch/jobs            작업 목록
  GET  /api/batch/{id}/status     상태·진행률
  GET  /api/batch/{id}/stream     SSE 진행률 스트림
  POST /api/batch/{id}/cancel     중단
  GET  /api/batch/{id}/report     집계 리포트(markdown/json)
  GET  /api/batch/{id}/results    results.jsonl 다운로드
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
import time
import uuid
import zipfile
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/batch", tags=["batch"])

MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024   # 8GB
_JOBS: Dict[str, "BatchJob"] = {}
_JOBS_LOCK = threading.Lock()

BATCH_ROOT: Path = Path("batch_jobs")        # app.py 가 init_batch_api() 로 덮어쓴다
RESULTS_ROOT: Path = Path("results")         # 파이프라인 단계별 산출물 위치
_ORCH = None                                 # 파이프라인 오케스트레이터(공유)


def init_batch_api(orchestrator, batch_root: Path, results_root: Path = None):
    """app.py 에서 1회 호출 — 오케스트레이터와 저장 위치를 주입한다."""
    global _ORCH, BATCH_ROOT, RESULTS_ROOT
    _ORCH = orchestrator
    BATCH_ROOT = Path(batch_root)
    BATCH_ROOT.mkdir(parents=True, exist_ok=True)
    if results_root is not None:
        RESULTS_ROOT = Path(results_root)
    _recover_jobs()


def _recover_jobs():
    """서버 재기동 시 디스크에 남은 작업을 목록에 되살린다.

    작업 상태는 메모리에만 있어서 재시작·배포 한 번에 목록이 통째로 사라졌다.
    실행 자체는 재개할 수 없지만(스레드가 죽었으므로), 결과와 산출물은 디스크에
    그대로 있으므로 **리포트 조회와 세트별 다운로드는 되어야 한다**.
    4,000세트를 몇 시간 돌린 뒤 배포 한 번에 접근 경로를 잃으면 곤란하다.
    """
    if not BATCH_ROOT.is_dir():
        return
    recovered = 0
    for d in sorted(BATCH_ROOT.iterdir()):
        if not d.is_dir() or d.name in _JOBS:
            continue
        res = d / "_eval_out" / "results.jsonl"
        if not res.exists():
            continue
        job = BatchJob(d.name, d, name=d.name)
        try:
            job.created_at = d.stat().st_mtime
        except OSError:
            pass
        n = sum(1 for _ in res.open(encoding="utf-8"))
        job.total = n
        job.progress = {"i": n, "total": n, "failed": 0, "cost_krw_total": 0.0}
        rep = d / "_eval_out" / "report.json"
        if rep.exists():
            try:
                job.summary = {"aggregate": json.loads(rep.read_text(encoding="utf-8")),
                               "recovered": True}
            except (OSError, json.JSONDecodeError):
                pass
        # 실행 스레드는 없다. 리포트가 있으면 완료로, 없으면 중단된 것으로 표시한다.
        job.state = "done" if rep.exists() else "cancelled"
        job.error = None if rep.exists() else "서버 재시작으로 중단됨 (결과는 보존)"
        with _JOBS_LOCK:
            _JOBS[d.name] = job
        recovered += 1
    if recovered:
        logger.info(f"배치 작업 {recovered}건 복구 (디스크 기준)")


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Zip-slip 방어 — 아카이브 밖으로 나가는 경로를 거부한다.

    사용자가 올린 ZIP을 그대로 풀면 '../../etc/...' 같은 항목이 서버 파일을 덮어쓸 수 있다.
    """
    dest = dest.resolve()
    for member in zf.infolist():
        name = member.filename
        if name.startswith("/") or ".." in Path(name).parts:
            raise HTTPException(400, f"아카이브에 위험한 경로가 있습니다: {name}")
        target = (dest / name).resolve()
        if not str(target).startswith(str(dest)):
            raise HTTPException(400, f"아카이브 경로가 대상 폴더를 벗어납니다: {name}")
    zf.extractall(dest)


def _find_manifest(root: Path) -> Optional[Path]:
    direct = root / "manifest.jsonl"
    if direct.exists():
        return direct
    hits = sorted(root.rglob("manifest.jsonl"))
    return hits[0] if hits else None


class BatchJob:
    """작업 1건 — 상태·진행률·이벤트 버퍼를 소유한다."""

    def __init__(self, job_id: str, work_dir: Path, name: str = ""):
        self.job_id = job_id
        self.name = name or job_id
        self.work_dir = work_dir
        self.out_dir = work_dir / "_eval_out"
        self.state = "pending"          # pending|running|done|cancelled|error
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.error: Optional[str] = None
        self.total = 0
        self.progress: Dict[str, Any] = {"i": 0, "total": 0, "failed": 0,
                                         "cost_krw_total": 0.0}
        self.summary: Optional[Dict] = None
        self.validation: Optional[Dict] = None
        self.config: Dict[str, Any] = {}     # 어떤 모델로 돌렸는지 리포트에 남긴다
        self.runner = None
        # 작업 생성 직후~러너 할당 사이에 취소가 들어오면 조용히 무시되고 그대로 완주해버린다.
        # 취소 의사를 여기에 먼저 남겨두고, 러너가 붙는 즉시 반영한다.
        self.cancel_requested = False
        self._events: Deque[Dict] = deque(maxlen=2000)
        self._seq = 0
        self._lock = threading.Lock()

    def push(self, ev: Dict):
        with self._lock:
            self._seq += 1
            ev = {**ev, "seq": self._seq, "ts": time.time()}
            self._events.append(ev)
            if ev.get("type") == "start":
                # 실제 실행 대상 수로 교체한다. job.total 은 매니페스트 전체 건수라
                # --limit 이나 재개(이미 완료분 제외) 시 "1 / 20" 처럼 어긋난다.
                self.total = ev.get("total", self.total)
            if ev.get("type") == "progress":
                self.progress = {
                    "i": ev.get("i", 0), "total": ev.get("total", self.total),
                    "failed": ev.get("failed", 0),
                    "cost_krw_total": ev.get("cost_krw_total", 0.0),
                    "last_set": ev.get("set_id"), "last_ok": ev.get("ok"),
                    "last_accuracy": ev.get("accuracy"),
                }

    def events_after(self, seq: int):
        with self._lock:
            return [e for e in self._events if e["seq"] > seq]

    def info(self) -> Dict:
        return {
            "job_id": self.job_id, "name": self.name, "state": self.state,
            "total": self.total, "progress": self.progress,
            "summary": self.summary, "validation": self.validation,
            "config": self.config,
            "error": self.error, "created_at": self.created_at,
            "started_at": self.started_at, "finished_at": self.finished_at,
        }


def _run_job(job: BatchJob, manifest_path: Path, cfg_kwargs: Dict):
    """백그라운드 스레드 본체."""
    from module.evaluation.batch_runner import BatchRunner, RunConfig
    from module.evaluation.manifest import (load_ground_truth, load_manifest,
                                            validate_manifest)
    try:
        job.state = "running"
        job.started_at = time.time()
        entries = load_manifest(manifest_path)
        gt = load_ground_truth(manifest_path.parent / "ground_truth.jsonl")
        job.validation = validate_manifest(entries)
        job.validation["ground_truth"] = len(gt)
        job.total = len(entries)
        job.push({"type": "validated", **job.validation})

        cfg = RunConfig(**cfg_kwargs)
        runner = BatchRunner(_ORCH, cfg, job.out_dir, progress_cb=job.push)
        job.runner = runner
        if job.cancel_requested:          # 러너 할당 전에 들어온 취소를 반영
            runner.cancel()
        summary = runner.run(entries, gt)
        job.summary = summary

        from module.evaluation.report import aggregate, load_results, render_markdown
        agg = aggregate(load_results(job.out_dir / "results.jsonl"))
        (job.out_dir / "report.md").write_text(render_markdown(agg), encoding="utf-8")
        (job.out_dir / "report.json").write_text(
            json.dumps(agg, ensure_ascii=False, indent=1), encoding="utf-8")
        job.summary["aggregate"] = agg
        job.state = "cancelled" if summary.get("cancelled") else "done"
    except Exception as e:
        logger.error(f"배치 작업 실패 {job.job_id}: {e}", exc_info=True)
        job.state = "error"
        job.error = f"{type(e).__name__}: {e}"[:400]
        job.push({"type": "error", "error": job.error})
    finally:
        job.finished_at = time.time()
        job.push({"type": "finished", "state": job.state})


@router.post("/create")
async def create_batch(
    archive: UploadFile = File(default=None),
    manifest_path: str = Form(default=None),
    name: str = Form(default=""),
    workers: int = Form(default=4),
    limit: int = Form(default=0),
    max_cost_krw: float = Form(default=0),
    vlm_prefer: str = Form(default="gemma"),
    # 단계별 모델 — 미지정 시 /api/llm-extract 단일처리와 동일한 기본값
    model_name: str = Form(default="alibaba-qwen3.5-122b-a10b"),
    ocr_provider: str = Form(default="alibaba"),
    ocr_model: str = Form(default=""),
    ner_model: str = Form(default="klue-roberta-large"),
    consolidate: bool = Form(default=True),
    consolidation_model: str = Form(default="alibaba-qwen3.5-122b-a10b"),
):
    """ZIP(manifest.jsonl + contracts/ + works/ + ground_truth.jsonl) 또는 서버 경로로 작업 생성."""
    if _ORCH is None:
        raise HTTPException(503, "배치 API가 초기화되지 않았습니다")
    if not archive and not manifest_path:
        raise HTTPException(400, "archive(ZIP) 또는 manifest_path 중 하나가 필요합니다")

    job_id = uuid.uuid4().hex[:12]
    work_dir = BATCH_ROOT / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    if archive:
        if not archive.filename.lower().endswith(".zip"):
            raise HTTPException(400, "ZIP 파일만 지원합니다")
        zpath = work_dir / "upload.zip"
        size = 0
        with zpath.open("wb") as f:
            while chunk := await archive.read(4 * 1024 * 1024):
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    f.close()
                    shutil.rmtree(work_dir, ignore_errors=True)
                    raise HTTPException(413, "아카이브가 너무 큽니다 (>8GB)")
                f.write(chunk)
        extract_dir = work_dir / "sets"
        extract_dir.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(zpath) as zf:
                _safe_extract(zf, extract_dir)
        except zipfile.BadZipFile:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(400, "손상된 ZIP 파일입니다")
        zpath.unlink(missing_ok=True)
        mpath = _find_manifest(extract_dir)
        if not mpath:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(400, "아카이브에 manifest.jsonl 이 없습니다")
    else:
        mpath = Path(manifest_path)
        if not mpath.is_file():
            raise HTTPException(400, f"매니페스트를 찾을 수 없습니다: {manifest_path}")

    job = BatchJob(job_id, work_dir, name=name or mpath.parent.name)
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    cfg_kwargs = {
        "workers": max(1, min(16, workers)),
        "limit": limit or None,
        "max_cost_krw": max_cost_krw or None,
        "vlm_prefer": vlm_prefer,
        "model_name": model_name,
        "ocr_provider": ocr_provider,
        "ocr_model": ocr_model or None,
        "ner_model": ner_model,
        "consolidate": consolidate,
        "consolidation_model": consolidation_model,
    }
    job.config = {k: v for k, v in cfg_kwargs.items()}
    threading.Thread(target=_run_job, args=(job, mpath, cfg_kwargs),
                     daemon=True, name=f"batch-{job_id}").start()
    return {"job_id": job_id, "state": job.state, "manifest": str(mpath)}


@router.get("/jobs")
async def list_jobs():
    with _JOBS_LOCK:
        jobs = [j.info() for j in _JOBS.values()]
    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    return {"jobs": jobs}


def _get(job_id: str) -> BatchJob:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"작업을 찾을 수 없습니다: {job_id}")
    return job


@router.get("/{job_id}/status")
async def job_status(job_id: str):
    return _get(job_id).info()


@router.post("/{job_id}/cancel")
async def job_cancel(job_id: str):
    job = _get(job_id)
    job.cancel_requested = True     # 러너가 아직 없어도 의사를 남긴다
    if job.runner:
        job.runner.cancel()
    if job.state == "pending":
        job.state = "cancelled"
    job.push({"type": "cancel_requested"})
    return {"ok": True, "job_id": job_id, "state": job.state}


@router.get("/{job_id}/stream")
async def job_stream(job_id: str, after: int = 0):
    """SSE — 진행률 이벤트를 순번(seq) 이후부터 흘려보낸다.

    새로고침해도 `after`로 이어붙일 수 있어 UI가 상태를 잃지 않는다.
    """
    job = _get(job_id)

    async def gen():
        seq = after
        idle = 0
        while True:
            evs = job.events_after(seq)
            for e in evs:
                seq = e["seq"]
                yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
            if evs:
                idle = 0
            else:
                idle += 1
                if idle % 10 == 0:                       # 프록시 타임아웃 방지 하트비트
                    yield ": keep-alive\n\n"
            if job.state in ("done", "error", "cancelled") and not job.events_after(seq):
                yield f"data: {json.dumps({'type': 'closed', 'state': job.state})}\n\n"
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})


@router.get("/{job_id}/report")
async def job_report(job_id: str, fmt: str = "md"):
    job = _get(job_id)
    path = job.out_dir / ("report.json" if fmt == "json" else "report.md")
    if not path.exists():
        raise HTTPException(404, "리포트가 아직 생성되지 않았습니다")
    if fmt == "json":
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    return PlainTextResponse(path.read_text(encoding="utf-8"),
                             media_type="text/markdown; charset=utf-8")


@router.get("/{job_id}/results")
async def job_results(job_id: str, limit: int = 0):
    """results.jsonl — limit>0 이면 앞에서 N건만(UI 표 미리보기용)."""
    job = _get(job_id)
    path = job.out_dir / "results.jsonl"
    if not path.exists():
        raise HTTPException(404, "결과가 아직 없습니다")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if limit and len(rows) >= limit:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r.pop("extracted", None)       # 표 미리보기에는 과하다
            r.pop("traceback", None)
            rows.append(r)
    return {"job_id": job_id, "count": len(rows), "results": rows}


# ---------------------------------------------------------------------------
# 세트별 파이프라인 산출물 — OCR 텍스트 · NER 엔티티 · LLM 메타데이터 · 통합검증
# ---------------------------------------------------------------------------
# 파이프라인은 실행할 때마다 results/{request_id}/ 에 단계별 산출물을 남긴다.
# 배치 결과에 그 request_id 를 기록해 두었으므로(contract/work 각각) 세트 단위로
# 되찾을 수 있다. 채점 결과만 보고 "왜 틀렸는지" 확인할 방법이 없으면 곤란하다.

def _set_result(job: "BatchJob", set_id: str) -> Dict[str, Any]:
    path = job.out_dir / "results.jsonl"
    if not path.exists():
        raise HTTPException(404, "결과가 아직 없습니다")
    found = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("set_id")) == str(set_id):
                found = r          # 재개로 여러 번 있으면 마지막 것
    if not found:
        raise HTTPException(404, f"세트를 찾을 수 없습니다: {set_id}")
    return found


def _leg_dirs(rec: Dict[str, Any]) -> Dict[str, Path]:
    legs = {}
    for leg, key in (("contract", "contract_request_id"), ("work", "work_request_id")):
        rid = rec.get(key)
        if rid:
            d = RESULTS_ROOT / str(rid)
            if d.is_dir():
                legs[leg] = d
    return legs


@router.get("/{job_id}/set/{set_id}/artifacts")
async def set_artifacts(job_id: str, set_id: str):
    """세트의 단계별 산출물 파일 목록 (계약서 leg + 저작물 leg)."""
    job = _get(job_id)
    rec = _set_result(job, set_id)
    legs = _leg_dirs(rec)
    out = {"set_id": set_id, "job_id": job_id,
           "contract_request_id": rec.get("contract_request_id"),
           "work_request_id": rec.get("work_request_id"),
           "legs": {}}
    for leg, d in legs.items():
        files = []
        for f in sorted(d.rglob("*")):
            if f.is_file():
                files.append({"path": str(f.relative_to(d)), "bytes": f.stat().st_size})
        out["legs"][leg] = {"request_id": str(d.name), "files": files}
    if not legs:
        out["note"] = "산출물 디렉터리가 없습니다 (구버전 실행이거나 정리됨)"
    return out


@router.get("/{job_id}/set/{set_id}/download")
async def set_download(job_id: str, set_id: str):
    """세트의 모든 파이프라인 산출물을 ZIP 하나로 내려받는다.

    contract/ 와 work/ 하위에 OCR·NER·llm_metadata.json·consolidated_metadata.json 이
    그대로 들어간다 — 채점 결과를 다시 따져볼 때 필요한 원본이다.
    """
    import io
    job = _get(job_id)
    rec = _set_result(job, set_id)
    legs = _leg_dirs(rec)
    if not legs:
        raise HTTPException(404, "이 세트의 산출물을 찾을 수 없습니다")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for leg, d in legs.items():
            for f in sorted(d.rglob("*")):
                if f.is_file():
                    zf.writestr(f"{leg}/{f.relative_to(d)}", f.read_bytes())
        # 채점 결과도 같이 넣어 둔다 — 산출물과 판정을 따로 보관하면 대조가 번거롭다
        zf.writestr("scoring.json", json.dumps(rec, ensure_ascii=False, indent=1))
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="set_{set_id}_artifacts.zip"'})
