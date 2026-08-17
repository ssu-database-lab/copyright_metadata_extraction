"""배치 실행기 — 계약서+저작물 세트를 동시 처리하고 채점한다.

CLI 하네스와 웹 배치 API가 **이 클래스 하나**를 공유한다. 실행 경로가 달라도
같은 코드가 돌아야 "UI로 돌린 결과"와 "CLI로 돌린 결과"가 갈리지 않는다.

세트 1건 처리 순서:
    1) 계약서 → 파이프라인(OCR → LLM ∥ NER → 통합검증)  … 제목·저자·라이선스·권리
    2) 저작물 → 파이프라인(VLM 또는 OCR 경로)            … 설명·키워드·해상도·파일속성
    3) 상속 병합(LLM 호출 없음)                          … 계약서 권리정보를 저작물에 병합
    4) 정답 대조 채점

계약서가 있어야 제목·저자·라이선스가 **문서에서 추출된 값**이 된다.
저작물 파일만 넣으면 이 3속성은 비거나(과소평가) 입력 메타데이터를 그대로
되돌려주게 되어(과대평가) 어느 쪽도 실제 성능이 아니다.

설계 요점:
  - 세트 단위 체크포인트(results.jsonl append) → 중단·재개 안전. 4,000건은 반드시 끊긴다.
  - 세트 단위 예외 격리 → 파일 1건이 깨져도 나머지 3,999건이 살아남는다.
  - 누적 비용 상한 → 폭주 루프가 곧 요금이다.
  - 취소 이벤트 → UI에서 중단 가능.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .manifest import ManifestEntry
from .scoring import score_set

# 검증된 단가 (docs/API_비용산정서_20260730.md · 환율 $1=₩1,400)
COST_OCR_PER_PAGE = 2.7
COST_LLM_EXTRACT = 10.2
COST_CONSOLIDATION_DOC = 34.6
COST_CONSOLIDATION_IMG = 35.4
COST_VLM_IMAGE = 0.2
COST_VLM_VIDEO = 2.01      # qwen3.5-omni-plus 실측


@dataclass
class RunConfig:
    workers: int = 4
    limit: Optional[int] = None
    resume: bool = True
    max_cost_krw: Optional[float] = None
    model_name: str = "alibaba-qwen3.5-122b-a10b"
    ocr_provider: str = "alibaba"
    ocr_model: Optional[str] = None
    ner_model: str = "klue-roberta-large"
    consolidate: bool = True
    consolidation_model: str = "alibaba-qwen3.5-122b-a10b"
    vlm_prefer: str = "gemma"
    contract_document_type: str = "저작재산권 이용허락 계약서"
    work_document_type: str = "기타문서"
    # tier D(파일 생성 날짜)는 기본 제외 — 정답은 '저작물 창작 시점'이고 추출값은
    # '파일 생성 시각'이라 서로 다른 양이다. 같은 값으로 채점하면 구조적 오답이 된다.
    # 시험기관과 정의가 합의되면 빈 튜플로 두어 채점에 포함시킨다.
    skip_tiers: tuple = ("D",)


def _pdf_page_count(path: str) -> int:
    """PDF 페이지 수. OCR 비용이 페이지 비례라 추정의 정확도를 좌우한다.

    PyMuPDF(fitz) → pypdf → pdfinfo 순으로 시도한다. 전부 실패하면 1로 본다
    (과소추정이지만 비용 상한이 조기에 걸리는 것보다 낫다).
    """
    p = str(path)
    if not p.lower().endswith(".pdf"):
        return 1
    try:
        import fitz                                   # PyMuPDF
        with fitz.open(p) as doc:
            return max(1, doc.page_count)
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        return max(1, len(PdfReader(p).pages))
    except Exception:
        pass
    try:
        out = subprocess.run(["pdfinfo", p], capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            if line.lower().startswith("pages:"):
                return max(1, int(line.split()[1]))
    except Exception:
        pass
    return 1


def _estimate_cost(resp: Dict[str, Any], media: str, is_contract: bool,
                   pages: Optional[int] = None) -> float:
    """실측 단가 기반 추정치. 토큰 실측이 응답에 있으면 그것을 우선한다."""
    # 실패한 처리는 실제로 모델을 부르지 않았다. 정상 요금을 매기면 비용 상한이
    # 헛돌고(예: 영상 P3 no-op 에 ₩37.41) 리포트의 비용도 부풀려진다.
    if resp is not None and resp.get("success") is False:
        ocr_len = len(str(resp.get("ocr_text") or ""))
        return COST_OCR_PER_PAGE if ocr_len else 0.0
    usage = (resp or {}).get("token_usage") or {}
    if usage.get("cost_krw"):
        try:
            return float(usage["cost_krw"])
        except (TypeError, ValueError):
            pass
    if is_contract or media == "text":
        n = pages if pages else 1
        for key in ("page_count", "pages", "total_pages"):
            if not pages and (resp or {}).get(key):
                try:
                    n = max(1, int(resp[key])); break
                except (TypeError, ValueError):
                    pass
        return COST_OCR_PER_PAGE * n + COST_LLM_EXTRACT + COST_CONSOLIDATION_DOC
    vlm = COST_VLM_VIDEO if media == "video" else COST_VLM_IMAGE
    return vlm + COST_CONSOLIDATION_IMG


def _final_metadata(resp: Dict[str, Any]) -> Dict[str, Any]:
    """파이프라인 응답에서 최종 메타데이터를 꺼낸다(통합 결과 우선)."""
    if not isinstance(resp, dict):
        return {}
    for key in ("consolidated_metadata", "metadata"):
        v = resp.get(key)
        if isinstance(v, dict) and v:
            return v
    res = resp.get("result")
    if isinstance(res, dict):
        for key in ("consolidated_metadata", "metadata"):
            v = res.get(key)
            if isinstance(v, dict) and v:
                return v
    return {}


class BatchRunner:
    def __init__(self, orchestrator, config: RunConfig, out_dir: str | Path,
                 progress_cb: Optional[Callable[[Dict], None]] = None):
        self.orch = orchestrator
        self.cfg = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.out_dir / "results.jsonl"
        self.progress_cb = progress_cb
        self.cancel_event = threading.Event()

        self._lock = threading.Lock()
        self._cost = 0.0
        self._done = 0
        self._failed = 0
        self._cost_exceeded = False

    # -- 재개 ---------------------------------------------------------------
    def _completed_ids(self) -> set:
        done = set()
        if not (self.cfg.resume and self.results_path.exists()):
            return done
        with self.results_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 실패는 재시도 대상이므로 완료로 치지 않는다
                if r.get("ok"):
                    done.add(str(r.get("set_id")))
        return done

    def cancel(self):
        self.cancel_event.set()

    def _emit(self, event: Dict):
        if self.progress_cb:
            try:
                self.progress_cb(event)
            except Exception:  # 진행률 콜백 실패가 실행을 죽이면 안 된다
                pass

    # -- 세트 1건 -----------------------------------------------------------
    def _run_one(self, entry: ManifestEntry, gt: Dict[str, Any],
                 cost_box: Dict[str, float]) -> Dict[str, Any]:
        """세트 1건 처리.

        `cost_box["v"]` 에 누적 비용을 즉시 기록한다 — 도중에 예외가 나도 이미 쓴 돈은
        회계에 남아야 한다. 반환값에만 담으면 실패 시 비용이 증발해 상한이 무력해진다.
        """
        t0 = time.perf_counter()
        out: Dict[str, Any] = {"set_id": entry.set_id, "media": entry.media,
                               "license_bucket": entry.license_bucket,
                               "contract": entry.contract, "work": entry.work}
        contract_meta = None
        stages: Dict[str, Any] = {}

        # 1) 계약서
        if entry.contract:
            with open(entry.contract, "rb") as f:
                cbytes = f.read()
            cresp = self.orch.run(
                cbytes, os.path.basename(entry.contract),
                model_name=self.cfg.model_name,
                document_type=self.cfg.contract_document_type,
                ocr_provider=self.cfg.ocr_provider, ocr_model=self.cfg.ocr_model,
                ner_model=self.cfg.ner_model, consolidate=self.cfg.consolidate,
                consolidation_model=self.cfg.consolidation_model,
            )
            contract_meta = _final_metadata(cresp)
            cost_box["v"] += _estimate_cost(cresp, "text", True,
                                            pages=_pdf_page_count(entry.contract))
            # request_id 로 results/{id}/ 에 OCR·NER·LLM·통합검증 산출물이 저장돼 있다.
            # 이걸 남겨야 세트별로 파이프라인 원본 출력을 되찾을 수 있다.
            out["contract_request_id"] = cresp.get("request_id")
            stages["contract"] = {"ok": bool(cresp.get("success")) and bool(contract_meta),
                                  "error": cresp.get("error"),
                                  "fields": sum(1 for v in contract_meta.values()
                                                if v not in (None, "", []))}

        # 2) 저작물
        with open(entry.work, "rb") as f:
            wbytes = f.read()
        wresp = self.orch.run(
            wbytes, os.path.basename(entry.work),
            model_name=self.cfg.model_name,
            document_type=self.cfg.work_document_type,
            ocr_provider=self.cfg.ocr_provider, ocr_model=self.cfg.ocr_model,
            ner_model=self.cfg.ner_model, consolidate=self.cfg.consolidate,
            consolidation_model=self.cfg.consolidation_model,
            vlm_prefer=self.cfg.vlm_prefer,
        )
        cost_box["v"] += _estimate_cost(wresp, entry.media, False)
        out["work_request_id"] = wresp.get("request_id")

        # ⚠️ 저작물 처리가 실제로 성공했는지 반드시 확인한다.
        #    파이프라인은 실패해도 예외를 던지지 않고 success=False + metadata={} 를 돌려준다:
        #      · 영상/음성 → P3 가드로 미구현 (build_response(success=False))
        #      · .wmv/.swf → router 가 unknown 으로 보내 문서 경로 → OCR 공백 가드
        #    이걸 ok=True 로 기록하면 전 속성이 '오답'으로 집계되어 정확도가 통째로 무너지고,
        #    게다가 재개 시 완료로 간주돼 영영 재시도되지 않는다.
        work_ok = bool(wresp.get("success")) and bool(_final_metadata(wresp))
        stages["work"] = {"ok": work_ok, "error": wresp.get("error")}
        if not work_ok:
            out.update({
                "ok": False,
                "error": (f"저작물 처리 실패(media={entry.media}): "
                          f"{wresp.get('error') or 'success=False / 메타데이터 없음'}")[:300],
                "elapsed_sec": round(time.perf_counter() - t0, 2),
                "cost_krw": round(cost_box["v"], 2), "stages": stages,
            })
            return out

        # 3) 상속 병합 — LLM 호출 없음
        if contract_meta:
            try:
                wresp = self.orch.apply_contract_inheritance(wresp, contract_meta)
                stages["inheritance"] = {"ok": True}
            except Exception as e:
                stages["inheritance"] = {"ok": False, "error": str(e)[:120]}

        extracted = _final_metadata(wresp)

        # 4) 채점
        scored = score_set(gt.get("attributes", {}), extracted, media=entry.media,
                           skip_tiers=self.cfg.skip_tiers)

        out.update({
            "ok": True,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "cost_krw": round(cost_box["v"], 2),
            "stages": stages,
            "extracted": extracted,
            **scored,
        })
        return out

    # -- 전체 실행 ----------------------------------------------------------
    def run(self, entries: List[ManifestEntry],
            ground_truth: Dict[str, Dict]) -> Dict[str, Any]:
        done_ids = self._completed_ids()
        todo = [e for e in entries if e.set_id not in done_ids]
        if self.cfg.limit:
            todo = todo[: self.cfg.limit]

        total = len(todo)
        t_start = time.perf_counter()
        self._emit({"type": "start", "total": total, "already_done": len(done_ids),
                    "workers": self.cfg.workers})

        if not total:
            return {"total": 0, "already_done": len(done_ids), "ok": 0, "failed": 0,
                    "cost_krw": 0.0, "elapsed_sec": 0.0, "cancelled": False}

        fh = self.results_path.open("a", encoding="utf-8")

        def work(entry: ManifestEntry) -> Optional[Dict]:
            if self.cancel_event.is_set() or self._cost_exceeded:
                return None
            cost_box = {"v": 0.0}
            try:
                gt = ground_truth.get(entry.set_id, {})
                res = self._run_one(entry, gt, cost_box)
            except Exception as e:
                res = {"set_id": entry.set_id, "ok": False,
                       "error": f"{type(e).__name__}: {e}"[:300],
                       "traceback": traceback.format_exc()[-800:],
                       "media": entry.media, "license_bucket": entry.license_bucket,
                       "cost_krw": round(cost_box["v"], 2)}
            with self._lock:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
                self._done += 1
                # 실패해도 이미 호출한 API 요금은 발생했다 — 상한 계산에서 빼면 안 된다.
                self._cost += res.get("cost_krw", 0.0)
                if not res.get("ok"):
                    self._failed += 1
                if (self.cfg.max_cost_krw is not None
                        and self._cost >= self.cfg.max_cost_krw
                        and not self._cost_exceeded):
                    self._cost_exceeded = True
                    self.cancel_event.set()
                    self._emit({"type": "cost_limit", "cost_krw": round(self._cost, 1),
                                "limit": self.cfg.max_cost_krw})
                snapshot = {"type": "progress", "i": self._done, "total": total,
                            "set_id": res.get("set_id"), "ok": bool(res.get("ok")),
                            "accuracy": res.get("accuracy"),
                            "n_match": res.get("n_match"), "n_scored": res.get("n_scored"),
                            "elapsed_sec": res.get("elapsed_sec"),
                            "cost_krw_total": round(self._cost, 1),
                            "failed": self._failed,
                            "error": res.get("error")}
            self._emit(snapshot)
            return res

        try:
            with ThreadPoolExecutor(max_workers=self.cfg.workers) as ex:
                futures = [ex.submit(work, e) for e in todo]
                for _ in as_completed(futures):
                    pass
        finally:
            fh.close()

        summary = {
            "total": total, "already_done": len(done_ids),
            "ok": self._done - self._failed, "failed": self._failed,
            "cost_krw": round(self._cost, 2),
            "elapsed_sec": round(time.perf_counter() - t_start, 1),
            "cancelled": self.cancel_event.is_set(),
            "cost_exceeded": self._cost_exceeded,
            "results_path": str(self.results_path),
        }
        self._emit({"type": "done", **summary})
        return summary
