"""
Pipeline Orchestrator — coordinates OCR → LLM → NER → Consolidation.

Extracted from app.py to separate business logic from route handling.
Each stage returns a typed result dict, and the orchestrator composes
them into the final API response.

Usage (non-streaming)::

    result = pipeline.run(file_bytes, filename, model_name=..., ...)

Usage (SSE streaming — call stages individually)::

    ctx = pipeline.setup(file_bytes, filename)
    ocr_text, ocr_result = pipeline.run_ocr(ctx, provider, model)
    llm_result = pipeline.run_llm(ocr_text, doc_type, ctx["filename"], model)
    ner_result = pipeline.run_ner(ocr_result, ctx["result_dir"], ner_model, ocr_text)
    con_result, success, error = pipeline.run_consolidation(...)
    response = pipeline.build_response(ctx, ...)
    pipeline.save_results(ctx["result_dir"], response, con_result, success)

v4 시험평가(docs/v4/M2_runner_spec.md §1.A) 가 더한 run() 키워드 — 기본값은 모두 오늘의 동작::

    run(..., consolidate_work=True,   # False: 이미지/영상 경로에서 중재자(통합검증) 생략 (A3)
             contract_context=None,   # {"title","work_type"}: VLM 프롬프트에 계약서 참고 줄 (A4)
             frame_schedule="v3",     # "v4": 긴 영상 키프레임 16/24 (A6)
             media=None,              # "image": .pptx 를 1쪽 렌더링 → 이미지 경로 (A7)
             test=None)               # "T-001"/"T-002": response["test_prediction"] 의 널 처리 (A2)
    응답에 더해진 키: file_name(원본 이름), file_hash(SHA-256), file_size_bytes, licensee_party,
    test_prediction(23키), contract_context, contract_context_rules, ocr_provider_used, ocr_pages.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable

import re
import unicodedata

try:                                     # 무거운 로컬 작업 동시 개수 (LOCAL_HEAVY_SLOTS, module/local_limits.py)
    from module.local_limits import disk_slot, heavy_slot
except ImportError:
    from api.module.local_limits import disk_slot, heavy_slot  # type: ignore

# 시험 자료(폴더 시험)를 잠깐 복사해 두는 곳 — 빠른 로컬 디스크(WSL 이면 ext4 의 /tmp). 세트가 끝나면 지운다.
# 실행 폴더(api/web/test_runs, 윈도 C:)에 두면 WSL 파일 다리로 쓰게 되어 느리고 C: 를 채운다.
INPUT_STAGING_DIR = Path(os.getenv("INPUT_STAGING_DIR") or (Path(__import__("tempfile").gettempdir()) / "console_staging"))
# 폴더 시험에서 결과 폴더의 source/ 에 남기는 사본 — 원본 폴더를 옮기거나 지워도 콘솔이 계약서 쪽 보기·위치 표시(노란 칠)와
# 저작물 그림을 보여 주게. 문서는 그대로(계약서 1건 ≈100 KB → 4,000건 ≈400 MB), 그림은 긴 변 1280 px JPEG 만.
KEEP_SOURCE_EXT = {".pdf"}
KEEP_SOURCE_MAX_MB = float(os.getenv("KEEP_SOURCE_MAX_MB") or 50)
WORK_PREVIEW_SIDE = 1280
try:                                     # 앱·CLI 는 api/ 를 루트로(module.*), 일부 테스트는 api.web.pipeline 으로 올린다
    from module import log_context as LC
except ImportError:                      # pragma: no cover
    from api.module import log_context as LC

logger = logging.getLogger(__name__)

# 실행 로그 한 줄 = 단계 하나 (M3 콘솔 로그 보강). component 는 시스템·업로드·문맥·OCR·LLM·NER·통합·VLM·API·측정·채점,
# stage 는 콘솔의 진행 단계 이벤트(upload · route · ocr · consolidate · consolidate_done · extract_done)다.
_MODALITY_KO = {"image": "이미지", "video": "영상", "document": "문서", "text": "텍스트", "audio": "음성"}
_NER_MODE_KO = {"model": "모델 사용", "regex_fallback": "정규표현식 대체"}


def _log(level: int, component: str, msg: str, stage: Optional[str] = None, exc_info: bool = False) -> None:
    extra: Dict[str, Any] = {"component": component}
    if stage:
        extra["stage"] = stage
    logger.log(level, msg, extra=extra, exc_info=exc_info)


def _filled(md: Any) -> int:
    if not isinstance(md, dict):
        return 0
    return sum(1 for k, v in md.items() if not str(k).startswith("_") and v not in (None, "", [], {}))


def _short(v: Any, n: int = 160) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def safe_filename(filename: str) -> str:
    """Sanitize filename while preserving Korean characters.

    werkzeug's secure_filename strips all non-ASCII, breaking Korean filenames.
    This version keeps Korean (Hangul), alphanumeric, dots, hyphens, underscores.
    """
    # Normalize unicode
    filename = unicodedata.normalize("NFC", filename)
    # Replace path separators
    filename = filename.replace("/", "_").replace("\\", "_")
    # Keep Korean (Hangul: U+AC00-U+D7AF, Jamo: U+1100-U+11FF, Compat Jamo: U+3130-U+318F),
    # alphanumeric, dots, hyphens, underscores, spaces
    filename = re.sub(r"[^\w\s\.\-\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF]", "", filename)
    # Collapse whitespace
    filename = re.sub(r"\s+", "_", filename).strip("_")
    return filename or "unnamed"


class PipelineOrchestrator:
    """Orchestrates the full metadata extraction pipeline."""

    def __init__(
        self,
        llm_processor,
        ner_predict_fn: Callable,
        available_ner_models: Dict[str, Dict],
        upload_dir: Path,
        results_dir: Path,
    ):
        self.llm_processor = llm_processor
        self.ner_predict = ner_predict_fn
        self.available_ner_models = available_ner_models
        self.upload_dir = upload_dir
        self.results_dir = results_dir

    # ------------------------------------------------------------------
    # Setup — shared by both streaming and non-streaming paths
    # ------------------------------------------------------------------

    def setup(self, file_bytes: bytes, filename: str, source_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
        """Save the uploaded file and prepare directories.

        source_path: 이미 디스크에 있는 파일(시험 PC 의 폴더). 메모리에 통째로 올리지 않고, 드라이브마다 한 번에 하나씩
        처음부터 끝까지 순서대로 읽어 INPUT_STAGING_DIR 로 복사하면서 SHA-256 을 계산한다(disk_slot). 뒤 단계(ffmpeg
        키프레임처럼 파일 안을 뛰어다니며 읽는 일)는 빠른 로컬 사본을 읽고, 세트가 끝나면 run() 이 사본을 지운다.
        이때 file_bytes 는 쓰지 않는다(빈 값이어도 된다). 근거: 통째 적재는 150 작업자에서 메모리 부족, 제자리 읽기는
        WSL 에서 동시 읽기가 무너져(1개 102 MB/s → 2개 15 MB/s) 느렸다 (2026-10-19/20 부하 시험).

        Returns a context dict used by all subsequent stages.
        """
        sanitized = safe_filename(filename)
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        original_filename = unicodedata.normalize("NFC", os.path.basename(filename or "")) or sanitized

        # 원본 파일명과 SHA-256 은 정제(safe_filename) **전에** 여기서 잡아 둔다 (v4 키 2·23, A1).
        # 저장용 이름은 공백이 '_' 로 바뀌고 쉼표 등이 빠지므로 파일명 채점에 그대로 쓰면 틀린다.
        # 해시는 여기서 한 번만 계산한다 — 큰 영상을 다시 읽지 않는다 (VLM 매핑도 이 값을 쓴다).
        t0 = time.perf_counter()
        if source_path is not None:
            src = Path(source_path)
            if not src.is_file():
                raise FileNotFoundError(f"입력 파일이 없습니다: {src}")
            upload_path = INPUT_STAGING_DIR / request_id / sanitized
            upload_path.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            size = 0
            wait0 = time.perf_counter()
            with disk_slot(src):                    # 이 드라이브를 읽는 일은 한 번에 하나 — 순서대로 끝까지
                waited = time.perf_counter() - wait0
                with src.open("rb") as fi, upload_path.open("wb") as fo:
                    for chunk in iter(lambda: fi.read(8 << 20), b""):
                        h.update(chunk)
                        fo.write(chunk)
                        size += len(chunk)
            shutil.copystat(src, upload_path)      # 수정 시각 유지 (파일 생성일 대체값)
            sha256 = h.hexdigest()
            where = f"작업 사본 {upload_path.parent.name} (드라이브 대기 {waited:.1f}초, 세트가 끝나면 지움)"
        else:
            upload_path = self.upload_dir / request_id / sanitized
            upload_path.parent.mkdir(parents=True, exist_ok=True)
            upload_path.write_bytes(file_bytes)
            size = len(file_bytes)
            sha256 = hashlib.sha256(file_bytes).hexdigest()
            where = f"저장 이름 {sanitized}"
        hash_sec = time.perf_counter() - t0

        result_dir = self.results_dir / request_id
        result_dir.mkdir(parents=True, exist_ok=True)

        file_size_mb = size / (1024 * 1024)
        _log(logging.INFO, "업로드",
             f"파일 준비 — {original_filename} · {file_size_mb:.2f} MB ({size:,} 바이트) · "
             f"{where} · SHA-256 {hash_sec:.3f}초", stage="upload")
        return {
            "request_id": request_id,
            "filename": sanitized,
            "original_filename": original_filename,
            "sha256": sha256,
            "file_size": size,
            "staged": source_path is not None,
            "source_path": str(source_path) if source_path is not None else None,
            "upload_path": upload_path,
            "result_dir": result_dir,
            "file_size_mb": file_size_mb,
            "start_time": datetime.now(),
            "timings": {"sha256": round(hash_sec, 4)},
        }

    # ------------------------------------------------------------------
    # Stage 1: OCR
    # ------------------------------------------------------------------

    def run_ocr(self, ctx: Dict, ocr_provider: str, ocr_model: str = None) -> Tuple[str, Dict]:
        """Run OCR on the uploaded file. Returns (ocr_text, ocr_result_dict)."""
        from module.ocr import UniversalOCRProcessor

        ocr_dir = ctx["result_dir"] / "ocr"
        ocr_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        try:
            processor = UniversalOCRProcessor(
                provider=ocr_provider, output_dir=str(ocr_dir), model=ocr_model
            )
            model_used = getattr(processor.ocr_provider, "model", None)
            _log(logging.INFO, "OCR",
                 f"OCR 준비 — 공급자 {ocr_provider} · 모델 {model_used or ocr_model or '(공급자 기본)'} · "
                 f"폴백 체인 {' → '.join(processor.fallback_chain)} (PDF 텍스트층이 있으면 OCR 을 부르지 않음)")
            ocr_result = processor.process_single_file(str(ctx["upload_path"]))
            sec = time.perf_counter() - t0

            if ocr_result.get("status") == "success":
                ocr_text = ocr_result.get("full_text", "")
                _log(logging.INFO, "OCR",
                     f"텍스트·OCR 완료 — {len(ocr_text):,}자 · {ocr_result.get('total_pages') or '?'}쪽 · "
                     f"{self._ocr_method(ctx, ocr_result, model_used)} · {sec:.1f}초 · 추출 시작", stage="ocr")
            else:
                ocr_text = ""
                _log(logging.WARNING, "OCR",
                     f"텍스트·OCR 실패 — {_short(ocr_result.get('error'), 300)} · {sec:.1f}초"
                     + (" · 모든 폴백 공급자 실패" if ocr_result.get("fallback_exhausted") else ""), stage="ocr")

        except Exception as e:
            sec = time.perf_counter() - t0
            _log(logging.WARNING, "OCR", f"텍스트·OCR 오류 — {type(e).__name__}: {_short(e, 300)} · {sec:.1f}초",
                 stage="ocr")
            ocr_text = ""
            ocr_result = {"status": "failed", "error": str(e)}

        ctx.setdefault("timings", {})["ocr"] = round(sec, 2)
        return ocr_text, ocr_result

    @staticmethod
    def _ocr_method(ctx: Dict, ocr_result: Dict, model: Optional[str]) -> str:
        """OCR 결과 → '무엇으로 읽었는지' 한 구절 (텍스트층·OCR 공급자/모델·폴백)."""
        provider = ocr_result.get("ocr_provider")
        if provider == "direct_text":
            ext = os.path.splitext(str(ctx.get("upload_path") or ""))[1].lower()
            return "PDF 텍스트층 사용 (OCR 생략)" if ext == ".pdf" else "파일 본문 텍스트 사용 (OCR 생략)"
        pages = ocr_result.get("pages") or []
        models = sorted({str((pg.get("metadata") or {}).get("model")) for pg in pages
                         if isinstance(pg, dict) and (pg.get("metadata") or {}).get("model")})
        out = f"OCR {provider or '?'} · 모델 {', '.join(models) or model or '?'}"
        fb_pages = [pg for pg in pages if isinstance(pg, dict) and (pg.get("metadata") or {}).get("fallback_used")]
        if fb_pages:
            orig = (fb_pages[0].get("metadata") or {}).get("original_model")
            out += f" · 모델 폴백 {len(fb_pages)}쪽 ({orig} → {(fb_pages[0].get('metadata') or {}).get('model')})"
        rl = sum(1 for pg in pages if isinstance(pg, dict) and (pg.get("metadata") or {}).get("rate_limited"))
        if rl:
            out += f" · 속도 제한 실패 {rl}쪽"
        if ocr_result.get("fallback_used"):
            out += (f" · 공급자 폴백 {ocr_result.get('original_provider')} → {provider} "
                    f"(사유: {_short(ocr_result.get('fallback_reason'), 120)})")
        if ocr_result.get("status") == "partial" or ocr_result.get("failed_pages"):
            out += f" · 실패 {ocr_result.get('failed_pages')}쪽"
        return out

    # ------------------------------------------------------------------
    # Stage 2: LLM extraction
    # ------------------------------------------------------------------

    # 파일에서 결정적으로 산출되는 필드 — 문서 본문에서 추론할 대상이 아니다.
    # 스키마 설명에도 "추론하지 말 것"이라 적었지만 프롬프트 지시는 보장이 아니므로,
    # LLM이 값을 지어내도 여기서 걷어낸다. 이 값들은 technical_metadata가 채운다.
    _FILE_DERIVED_FIELDS = ("resolution", "file_size", "file_created_date")

    def run_llm(self, ocr_text: str, document_type: str,
                filename: str, model_name: str) -> Dict[str, Any]:
        """Run LLM metadata extraction on OCR text."""
        _log(logging.INFO, "LLM", f"LLM 추출 시작 — 모델 {model_name} · 문서 유형 {document_type} · 본문 {len(ocr_text or ''):,}자")
        t0 = time.perf_counter()
        result = self.llm_processor.extract_metadata_from_text(
            text=ocr_text,
            document_type=document_type,
            document_name=filename,
            model_name=model_name,
        )
        sec = time.perf_counter() - t0
        metadata = (result or {}).get("metadata")
        if isinstance(metadata, dict):
            hallucinated = [f for f in self._FILE_DERIVED_FIELDS
                            if metadata.get(f) not in (None, "", [])]
            for field in hallucinated:
                metadata[field] = None
            if hallucinated:
                _log(logging.WARNING, "LLM", f"LLM 이 파일 유래 필드를 본문에서 추론함 → 제거: {hallucinated}")
        if isinstance(result, dict):
            result["_seconds"] = round(sec, 2)
            if result.get("success"):
                _log(logging.INFO, "LLM",
                     f"LLM 추출 완료 — 모델 {result.get('model_used') or model_name} · {sec:.1f}초 · "
                     f"채운 항목 {_filled(metadata)}개")
            else:
                _log(logging.WARNING, "LLM", f"LLM 추출 실패 — 모델 {model_name} · {_short(result.get('error'), 300)} · {sec:.1f}초")
        return result

    # ------------------------------------------------------------------
    # Stage 3: NER
    # ------------------------------------------------------------------

    def run_ner(self, ocr_result: Dict, result_dir: Path,
                ner_model: str, ocr_text: str) -> Dict[str, Any]:
        """Run NER entity extraction."""
        ner_dir = result_dir / "ner"
        t0 = time.perf_counter()

        try:
            ocr_output_dir = ocr_result.get("output_directory")

            if not ocr_output_dir or ocr_result.get("status") != "success":
                ocr_output_dir = result_dir / "ocr"
                ocr_output_dir.mkdir(parents=True, exist_ok=True)
                (ocr_output_dir / "temp.ocr").write_text(ocr_text, encoding="utf-8")

            ner_model_name = self.available_ner_models[ner_model]["name"]
            _log(logging.INFO, "NER", f"NER 추출 시작 — 모델 {ner_model} ({ner_model_name})")
            ner_result = self.ner_predict(
                str(ocr_output_dir), str(ner_dir),
                model_name=ner_model_name, debug=False,
            )
            sec = time.perf_counter() - t0
            if isinstance(ner_result, dict):
                ner_result["_seconds"] = round(sec, 2)
                mode = ner_result.get("mode")
                if ner_result.get("success", False):
                    why = " · 모델 미적재 (사유는 위 NER 줄)" if mode != "model" and ner_result.get("load_error") else ""
                    _log(logging.INFO, "NER",
                         f"NER 추출 완료 — 방식 {_NER_MODE_KO.get(mode, mode or '알 수 없음')} · 모델 {ner_model_name} · "
                         f"엔티티 {ner_result.get('total_entities', 0)}개 · {sec:.1f}초{why}")
                else:
                    _log(logging.WARNING, "NER", f"NER 실패 — {_short(ner_result.get('error'), 300)} · {sec:.1f}초")
            return ner_result

        except Exception as e:
            sec = time.perf_counter() - t0
            _log(logging.WARNING, "NER", f"NER 오류 — {type(e).__name__}: {_short(e, 300)} · {sec:.1f}초")
            return {
                "success": False, "error": str(e),
                "entity_types": {}, "total_entities": 0,
                "extracted_entities": [], "_seconds": round(sec, 2),
                "mode": None, "model_loaded": False, "load_error": None, "entity_count": 0,
            }

    # ------------------------------------------------------------------
    # Stage 4: Consolidation
    # ------------------------------------------------------------------

    def run_consolidation(
        self, llm_result: Dict, ner_result: Dict, ocr_text: str,
        document_type: str, result_dir: Path,
        consolidation_model: str,
    ) -> Tuple[Optional[Dict], bool, Optional[str]]:
        """Consolidate LLM + NER results. Returns (result, success, error).

        Handles partial failures gracefully:
        - Both succeeded → full consolidation (compare + merge)
        - LLM only → use LLM metadata as-is (all decisions = LLM_ONLY)
        - NER only → use NER entities as-is (all decisions = NER_ONLY)
        - Both failed → skip consolidation
        """
        from module.consolidator import ConsolidationAgent
        from module.consolidator.consolidation_agent import _thinking_off, consolidation_timeout_sec

        llm_ok = llm_result and llm_result.get("success", False)
        ner_ok = ner_result and ner_result.get("success", False)

        if not llm_ok and not ner_ok:
            _log(logging.WARNING, "통합", "통합 생략 — LLM·NER 추출이 모두 실패했습니다", stage="consolidate_done")
            return None, False, "Both LLM and NER extraction failed"

        if not llm_ok:
            _log(logging.WARNING, "통합", "LLM 추출 실패 — 통합은 NER 결과만으로 진행합니다")
            llm_result = llm_result or {"success": False, "metadata": {}}

        if not ner_ok:
            # 이미지·영상 경로는 NER 를 돌리지 않는다(빈 NER 결과를 넘긴다) — 실패가 아니라 정보로 남긴다
            _log(logging.WARNING if ocr_text else logging.INFO, "통합",
                 "NER 결과 없음 — 통합은 LLM(VLM) 결과만으로 진행합니다")
            ner_result = ner_result or {"success": False, "entities": {}, "total_entities": 0, "extracted_entities": []}

        fallback = os.getenv("CONSOLIDATION_FALLBACK_MODEL", "alibaba-qwen3.5-plus")
        if fallback.lower() in ("", "none", "off"):
            fallback = None
        ner_mode = ner_result.get("mode") if isinstance(ner_result, dict) else None
        _log(logging.INFO, "통합",
             f"추출 완료 · 통합 시작 — 모델 {consolidation_model} · 폴백 {fallback or '없음'} · "
             f"LLM {'OK' if llm_ok else '실패'} · NER {'OK' if ner_ok else '없음/실패'}"
             + (f" ({_NER_MODE_KO.get(ner_mode, ner_mode)})" if ner_ok and ner_mode else "")
             + f" · 생각 {'끔' if _thinking_off('CONSOLIDATION_THINKING') else '켬'}"
             + f" · 시간 제한 {consolidation_timeout_sec():g}초" + ("" if os.getenv("CONSOLIDATION_TIMEOUT_SEC") else "(기본)"),
             stage="consolidate")
        t0 = time.perf_counter()

        try:
            # Ensure ner_result has extracted_entities
            if ner_result and "extracted_entities" not in ner_result:
                entities_data = ner_result.get("entities", {})
                extracted = []
                if isinstance(entities_data, dict):
                    for etype, elist in entities_data.items():
                        if isinstance(elist, list):
                            for entity in elist:
                                if isinstance(entity, dict):
                                    text = entity.get("text", entity.get("entity", ""))
                                    extracted.append((text, etype))
                                elif isinstance(entity, str):
                                    extracted.append((entity, etype))
                ner_result["extracted_entities"] = extracted

            llm_with_ocr = llm_result.copy()
            llm_with_ocr["ocr_text"] = ocr_text

            # 폴백 중재자(fallback, 위에서 읽음). 환경변수로 바꿀 수 있게 한다 — 기본값 qwen3.5-plus 는
            # 2026-09 측정에서 통합검증 프롬프트를 90초 제한 안에 끝내지 못해
            # (실측 94.9초) 전 호출이 실패하고 있었다. CONSOLIDATION_TIMEOUT_SEC 을
            # 함께 올리지 않으면 이 폴백은 동작하지 않는다.
            agent = ConsolidationAgent(
                model_name=consolidation_model,
                output_dir=str(result_dir),
                fallback_model=fallback,
                enable_hybrid=True,
            )
            result = agent.consolidate(
                llm_result=llm_with_ocr,
                ner_result=ner_result,
                ocr_text=ocr_text,
                document_type=document_type,
            )

            sec = time.perf_counter() - t0
            if isinstance(result, dict):
                result["consolidation_seconds"] = round(sec, 2)
            if result.get("success", False):
                counts = self._consolidation_counts(result)
                used = result.get("model_used") or consolidation_model
                fb = " (폴백)" if result.get("fallback_used") else ""
                if result.get("consolidation_degraded"):
                    # LLM 중재 없이 규칙 기반으로 병합된 결과다. 예외가 아니라 정상 반환이므로
                    # 여기서 명시하지 않으면 저장 결과만 보고는 구분할 수 없다.
                    _log(logging.ERROR, "통합",
                         f"통합 완료(저하) — LLM 중재 없이 규칙 기반 병합 · 사유 {_short(result.get('degraded_reason'), 300)} · "
                         f"{counts} · {sec:.1f}초", stage="consolidate_done")
                else:
                    _log(logging.INFO, "통합", f"통합 완료 — 모델 {used}{fb} · {counts} · {sec:.1f}초",
                         stage="consolidate_done")
                lost = result.get("consolidation_lost_fields")
                if lost:
                    _log(logging.WARNING, "통합",
                         f"통합 결과에서 필드가 사라짐 {result.get('consolidation_fields_before')}→"
                         f"{result.get('consolidation_fields_after')}: {lost}")
                return result, True, None
            else:
                error = result.get("error", "Consolidation failed")
                _log(logging.WARNING, "통합", f"통합 실패 — {_short(error, 300)} · {sec:.1f}초", stage="consolidate_done")
                return result, False, error

        except Exception as e:
            sec = time.perf_counter() - t0
            _log(logging.ERROR, "통합", f"통합 실패 — {type(e).__name__}: {_short(e, 300)} · {sec:.1f}초",
                 stage="consolidate_done", exc_info=True)
            return None, False, str(e)

    @staticmethod
    def _consolidation_counts(result: Dict[str, Any]) -> str:
        """중재자 판정 수 — 일치·충돌·LLM 전용·NER 전용·누락 (판정 목록 기준) + 요약 신뢰도."""
        vr = result.get("validation_report") or {}
        decisions = vr.get("decisions") or []
        c: Dict[str, int] = {}
        for d in decisions:
            if isinstance(d, dict):
                c[str(d.get("decision"))] = c.get(str(d.get("decision")), 0) + 1
        summ = vr.get("summary") or {}
        conf = summ.get("overall_confidence", vr.get("confidence_score"))
        out = (f"판정 {len(decisions)}건: 일치 {c.get('AGREED', 0)} · 충돌 {c.get('CONFLICT', 0)} · "
               f"LLM 전용 {c.get('LLM_ONLY', 0)} · NER 전용 {c.get('NER_ONLY', 0)} · 누락 {c.get('MISSING', 0)}")
        if isinstance(conf, (int, float)):
            out += f" · 신뢰도 {conf:.2f}"
        errs = vr.get("validation_errors") or []
        if errs:
            out += f" · 검증 오류 {len(errs)}건"
        return out

    # ------------------------------------------------------------------
    # Multimodal: modality routing + VLM (image) extraction
    # ------------------------------------------------------------------

    def run_router(self, ctx: Dict, media_hint: Optional[str] = None) -> Dict[str, Any]:
        """Detect modality (image/video/audio/document/text) + extractor plan.

        media_hint: 호출자(배치 매니페스트의 media)가 아는 매체. "image" 이면 .pptx 를 이미지
        경로로 보내고 plan["render"]="pptx_slide1" 을 돌려준다 (A7). None 이면 확장자만 본다.
        """
        from module.clip_extraction.router import route
        plan = route(str(ctx["upload_path"]), media_hint=media_hint)
        modality = plan["modality"]
        extra = " · 1쪽 슬라이드를 그림으로 렌더링" if plan.get("render") == "pptx_slide1" else (
            f" · 렌더링 {plan['render']}" if plan.get("render") else "")
        nxt = ("텍스트·OCR 시작" if modality in ("document", "text") else
               "추출 시작" if modality in ("image", "video") else "")
        _log(logging.INFO, "업로드",
             f"경로 — 매체 {_MODALITY_KO.get(modality, modality)} ({modality}){extra}"
             + (f" · 매체 지정 {media_hint}" if media_hint else "") + (f" · {nxt}" if nxt else ""), stage="route")
        return plan

    @staticmethod
    def _keep_source(ctx: Dict[str, Any]) -> None:
        """폴더 시험의 문서(계약서 PDF)를 결과 폴더 source/document.pdf 로 남긴다 — 작업 사본은 세트 끝에 지워진다."""
        src = Path(ctx["upload_path"])
        if not ctx.get("staged") or src.suffix.lower() not in KEEP_SOURCE_EXT:
            return
        if (ctx.get("file_size") or 0) > KEEP_SOURCE_MAX_MB * 1048576:
            _log(logging.INFO, "업로드", f"원본 사본 생략 — {ctx.get('file_size_mb', 0):.1f} MB > {KEEP_SOURCE_MAX_MB:g} MB")
            return
        try:
            dst = ctx["result_dir"] / "source" / ("document" + src.suffix.lower())
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except Exception as e:  # noqa: BLE001 — 미리보기용 사본이 본 처리를 막지 않게
            _log(logging.WARNING, "업로드", f"원본 사본 저장 실패 — {type(e).__name__}: {_short(e, 200)}")

    @staticmethod
    def _keep_work_preview(ctx: Dict[str, Any], image_path) -> None:
        """폴더 시험의 그림 저작물은 미리보기 JPEG(긴 변 1280 px)만 source/work_preview.jpg 로 남긴다 — 원본(수백 MB TIF 도
        있다)은 복사하지 않는다. 작업 사본을 읽으므로 E: 를 다시 읽지 않는다."""
        if not ctx.get("staged"):
            return
        try:
            from PIL import Image
            dst = ctx["result_dir"] / "source" / "work_preview.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            with heavy_slot():
                Image.MAX_IMAGE_PIXELS = None                 # 큰 TIF 도 연다 (VLM 인코딩과 같다)
                with Image.open(image_path) as im:
                    im.draft("RGB", (WORK_PREVIEW_SIDE, WORK_PREVIEW_SIDE))   # JPEG 는 축소해서 푼다
                    im = im.convert("RGB")
                    im.thumbnail((WORK_PREVIEW_SIDE, WORK_PREVIEW_SIDE))
                    im.save(dst, "JPEG", quality=85)
        except Exception as e:  # noqa: BLE001
            _log(logging.WARNING, "VLM", f"미리보기 그림 저장 실패 — {type(e).__name__}: {_short(e, 200)}")

    @staticmethod
    def _save_vlm_unparsed(ctx: Dict[str, Any], res) -> None:
        """JSON 으로 읽지 못한 VLM 응답 원문을 결과 폴더의 vlm_unparsed.jsonl 에 남긴다 — 예전에는 버려서 원인을 볼 수 없었다."""
        fails = getattr(res, "parse_failures", None)
        if not fails:
            return
        try:
            with open(ctx["result_dir"] / "vlm_unparsed.jsonl", "a", encoding="utf-8") as f:
                for x in fails:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            _log(logging.WARNING, "VLM", f"JSON 으로 읽지 못한 응답 {len(fails)}건 — 원문을 vlm_unparsed.jsonl 에 저장"
                 + (" · 다시 불러 성공" if res.ok and res.parse_ok else ""))
        except Exception as e:  # noqa: BLE001 — 진단용 기록이 본 처리를 막지 않게
            _log(logging.WARNING, "VLM", f"vlm_unparsed.jsonl 저장 실패 — {type(e).__name__}: {_short(e, 200)}")

    @staticmethod
    def _vlm_chain_line(extractor, what: str, context: Optional[Dict[str, Any]]) -> None:
        labels = [getattr(b, "model_label", "?") for b in getattr(extractor, "backends", []) or []]
        ctx_txt = ""
        if context:
            ctx_txt = (f" · 계약서 문맥 제목 {context.get('title')!r}" if context.get("title") else "") + \
                      (f" · 종류 {context.get('work_type')}" if context.get("work_type") else "")
        _log(logging.INFO, "VLM", f"{what} 시작 — 백엔드 체인 {' → '.join(labels) or '(없음)'}{ctx_txt}")

    @staticmethod
    def _vlm_done_line(res, backend: str, sec: float, unified: Dict[str, Any], extra: str = "") -> None:
        usage = getattr(res, "usage", None) or {}
        tok = (f" · 토큰 입력 {usage.get('prompt_tokens'):,} / 출력 {usage.get('completion_tokens'):,}"
               if isinstance(usage.get("prompt_tokens"), int) and isinstance(usage.get("completion_tokens"), int) else "")
        lat = getattr(res, "latency_s", None)
        _log(logging.INFO, "VLM",
             f"VLM 추출 완료 — {extra}{backend} · 응답 {lat if lat is not None else '?'}초{tok} · "
             f"유형 {unified.get('work_type') or '없음'} · 채운 항목 {_filled(unified)}개 · 전체 {sec:.1f}초",
             stage="extract_done")

    @staticmethod
    def _measure_line(ctx: Dict, md: Dict[str, Any], file_meta: Optional[Dict[str, Any]], sec: Optional[float]) -> None:
        """파일 측정 결과 한 줄 — 형식·크기·해상도·생성일과 잰 방법."""
        tech = (file_meta or {}).get("technical") or {}
        how = [str(v) for k, v in tech.items() if k.endswith("_source") and v]
        size = md.get("file_size")
        _log(logging.INFO, "측정",
             f"파일 측정 — 형식 {md.get('digital_format') or '?'} · 크기 "
             f"{f'{size:,} 바이트' if isinstance(size, int) else (size or '?')} · 해상도 {md.get('resolution') or '없음'} · "
             f"생성일 {md.get('file_created_date') or '없음'}"
             + (f" · 방법 {', '.join(how)}" if how else "")
             + (f" · SHA-256 {ctx.get('timings', {}).get('sha256'):.3f}초" if isinstance(ctx.get('timings', {}).get('sha256'), float) else "")
             + (f" · {sec:.2f}초" if sec is not None else ""))

    def run_vlm(self, ctx: Dict, prefer: str = "auto",
                context: Optional[Dict[str, Any]] = None,
                render: Optional[str] = None) -> Dict[str, Any]:
        """Extract visual metadata from an image work via the Gemma(primary)→
        Qwen(backup) VLM fallback chain, mapped to the unified schema.

        Returns an llm_result-shaped dict so it slots straight into
        run_consolidation() and build_response() with no arbiter changes.

        context: 계약서 참고 문맥 {"title", "work_type"} — 프롬프트 끝에 참고 줄로 붙는다 (A4).
        render : "pptx_slide1" 이면 1쪽 슬라이드를 PNG 로 렌더링해 그 그림을 VLM 에 넣는다 (A7).
                 파일포맷·크기·해시 같은 기술 메타데이터는 **원본 .pptx** 에서 계산하고 해상도는 비운다.
                 렌더링이 불가능한 환경이면 success=False 와 명확한 error 로 돌려준다 — 예외가 아니다.
        """
        import time
        from module.clip_extraction.vlm.extractor import VLMExtractor
        from module.clip_extraction.schema_mapping import map_vlm_to_unified

        t0 = time.perf_counter()
        try:
            image_path = ctx["upload_path"]
            render_info = None
            if render == "pptx_slide1" or str(image_path).lower().endswith(".pptx"):
                from module.clip_extraction.pptx_render import PptxRenderError, render_first_slide
                png = ctx["result_dir"] / "pptx_slide1.png"
                t_r = time.perf_counter()
                try:
                    render_first_slide(image_path, png)
                    _log(logging.INFO, "VLM", f"PPTX 1쪽 렌더링 완료 — {time.perf_counter() - t_r:.1f}초 · 그림을 VLM 에 넣습니다")
                except PptxRenderError as e:
                    _log(logging.ERROR, "VLM", f"PPTX 1쪽 렌더링 실패 — VLM 생략: {e}", stage="extract_done")
                    return {"success": False, "metadata": {}, "confidence": 0.0,
                            "model_used": "none", "error": f"PPTX 1쪽 렌더링 불가: {e}",
                            "extraction_time": round(time.perf_counter() - t0, 2),
                            "_render": {"method": "pptx_slide1", "ok": False}}
                image_path = png
                render_info = {"method": "pptx_slide1", "ok": True, "png": str(png)}

            extractor = VLMExtractor(prefer=prefer, max_tokens=4096)   # omni-flash 출력이 2048 을 넘는 경우 8% (2026-10-11 실측)
            self._vlm_chain_line(extractor, "VLM 이미지 추출", context)
            res = extractor.extract(image_path, context=context)
            self._save_vlm_unparsed(ctx, res)
            if render_info is None:                   # PPTX 는 pptx_slide1.png 가 이미 결과 폴더에 있다
                self._keep_work_preview(ctx, image_path)
            backend = getattr(res, "backend_used", "?")
            if not res.ok or not res.parse_ok or not res.parsed:
                err = res.error or "VLM ran but JSON parse failed"
                _log(logging.ERROR, "VLM", f"VLM 추출 실패 — {_short(err, 300)} · {time.perf_counter() - t0:.1f}초",
                     stage="extract_done")
                return {"success": False, "metadata": {}, "confidence": 0.0,
                        "model_used": backend,
                        "error": err,
                        "extraction_time": round(time.perf_counter() - t0, 2),
                        "_seconds": round(time.perf_counter() - t0, 2),
                        "_render": render_info}
            # 기술 메타데이터·SHA256 은 항상 **원본** 파일에서 — 렌더링한 PNG 가 아니다.
            t_m = time.perf_counter()
            unified = map_vlm_to_unified(res.parsed, str(ctx["upload_path"]), sha256=ctx.get("sha256"))
            file_meta = unified.pop("_file_meta", {})
            self._vlm_done_line(res, backend, time.perf_counter() - t0, unified)
            self._measure_line(ctx, unified, file_meta, time.perf_counter() - t_m)
            return {
                "success": True,
                "metadata": unified,
                "confidence": 0.7,  # VLM does not self-score; fixed visual-extraction prior
                "model_used": backend,
                "extraction_time": round(time.perf_counter() - t0, 2),
                "_seconds": round(time.perf_counter() - t0, 2),
                "_file_meta": file_meta,
                "_vlm_raw": res.parsed,
                "_render": render_info,
                "_prompt_context": dict(context) if context else None,
            }
        except Exception as e:
            _log(logging.ERROR, "VLM", f"VLM 추출 오류 — {type(e).__name__}: {_short(e, 300)}", stage="extract_done",
                 exc_info=True)
            return {"success": False, "metadata": {}, "confidence": 0.0,
                    "model_used": "none", "error": str(e),
                    "extraction_time": round(time.perf_counter() - t0, 2)}

    def run_video(self, ctx: Dict, prefer: str = "omni",
                  context: Optional[Dict[str, Any]] = None,
                  frame_schedule: str = "v3") -> Dict[str, Any]:
        """영상 저작물 → 키프레임 추출 → 다중 프레임 VLM 1회 호출 → 통합 스키마.

        run_vlm(이미지)과 같은 llm_result 모양을 돌려주므로 통합검증·응답조립
        경로를 그대로 재사용한다. 새로 필요한 것은 키프레임 추출뿐이다.

        context       : 계약서 참고 문맥 (run_vlm 과 같다, A4)
        frame_schedule: 키프레임 수 표 "v3"(기본) | "v4"(긴 영상 16/24, A6)
        """
        import time
        from module.clip_extraction.keyframes import extract_keyframes
        from module.clip_extraction.schema_mapping import map_vlm_to_unified
        from module.clip_extraction.vlm.extractor import VLMExtractor

        t0 = time.perf_counter()
        try:
            frame_dir = ctx["result_dir"] / "keyframes"
            with heavy_slot():                       # ffmpeg 키프레임 — 동시에 몇 개만 (하드 읽기·메모리)
                kf = extract_keyframes(str(ctx["upload_path"]), str(frame_dir), schedule=frame_schedule)
            kf_sec = time.perf_counter() - t0
            if not kf.get("frames"):
                _log(logging.ERROR, "VLM",
                     f"키프레임 추출 실패 — {kf.get('error') or '프레임 없음'} · 길이 {kf.get('duration')}초 · {kf_sec:.1f}초",
                     stage="extract_done")
                return {"success": False, "metadata": {}, "confidence": 0.0,
                        "model_used": "none",
                        "error": f"키프레임 추출 실패: {kf.get('error') or '프레임 없음'}",
                        "extraction_time": round(time.perf_counter() - t0, 2),
                        "_keyframes": kf}

            paths = [f["path"] for f in kf["frames"]]
            times = [f["t"] for f in kf["frames"]]
            _log(logging.INFO, "VLM",
                 f"키프레임 {kf.get('kept', len(paths))}장 추출 — 길이 {kf.get('duration')}초 · 표 {kf.get('schedule')} · "
                 f"목표 {kf.get('target')} · 후보 {kf.get('sampled')} · 시각 {', '.join(f'{t}s' for t in times)} · {kf_sec:.1f}초")
            extractor = VLMExtractor(prefer=prefer, max_tokens=4096)   # omni-flash 출력이 2048 을 넘는 경우 8% (2026-10-11 실측)
            self._vlm_chain_line(extractor, f"VLM 영상 추출({len(paths)}프레임 1회 호출)", context)
            res = extractor.extract_video(paths, times, label=ctx["filename"], context=context)
            self._save_vlm_unparsed(ctx, res)
            backend = getattr(res, "backend_used", "?")
            if not res.ok or not res.parse_ok or not res.parsed:
                err = res.error or "VLM ran but JSON parse failed"
                _log(logging.ERROR, "VLM", f"VLM 영상 추출 실패 — {_short(err, 300)} · {time.perf_counter() - t0:.1f}초",
                     stage="extract_done")
                return {"success": False, "metadata": {}, "confidence": 0.0,
                        "model_used": backend,
                        "error": err,
                        "extraction_time": round(time.perf_counter() - t0, 2),
                        "_seconds": round(time.perf_counter() - t0, 2),
                        "_keyframes": kf}

            t_m = time.perf_counter()
            unified = map_vlm_to_unified(res.parsed, str(ctx["upload_path"]), sha256=ctx.get("sha256"))
            file_meta = unified.pop("_file_meta", {})
            self._vlm_done_line(res, backend, time.perf_counter() - t0, unified,
                                extra=f"영상 {kf.get('kept', len(paths))}프레임 · ")
            self._measure_line(ctx, unified, file_meta, time.perf_counter() - t_m)
            return {
                "success": True, "metadata": unified, "confidence": 0.7,
                "model_used": backend,
                "extraction_time": round(time.perf_counter() - t0, 2),
                "_seconds": round(time.perf_counter() - t0, 2),
                "_file_meta": file_meta, "_vlm_raw": res.parsed,
                "_keyframes": {k: v for k, v in kf.items() if k != "frames"},
                "_frame_times": times,
                "_prompt_context": dict(context) if context else None,
            }
        except Exception as e:
            _log(logging.ERROR, "VLM", f"영상 추출 오류 — {type(e).__name__}: {_short(e, 300)}", stage="extract_done",
                 exc_info=True)
            return {"success": False, "metadata": {}, "confidence": 0.0,
                    "model_used": "none", "error": str(e),
                    "extraction_time": round(time.perf_counter() - t0, 2)}

    def apply_contract_inheritance(self, response: Dict[str, Any],
                                   contract_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """계약서 메타데이터를 저작물(이미지) 응답에 상속 병합 (post-consolidation).

        consolidated_metadata 가 있으면 거기에, 없으면 metadata 에 병합하고,
        상속 provenance 를 consolidation_decisions 와 동일 스키마로 덧붙인다.
        response 를 in-place 수정 후 반환. 실패해도 응답 자체는 깨지 않는다.
        """
        try:
            from module.clip_extraction.contract_inheritance import inherit_contract_fields
            target_key = "consolidated_metadata" if response.get("consolidated_metadata") else "metadata"
            merged, decisions, summary = inherit_contract_fields(
                response.get(target_key) or {},
                contract_metadata,
                work_filename=response.get("filename", ""),
            )
            response[target_key] = merged
            if decisions:
                existing = response.get("consolidation_decisions") or []
                # 상속이 발생한 필드는 원래 값이 비어있던 필드다 — arbiter 가 남긴 기존
                # 항목(MISSING 또는 null 값 LLM_ONLY 등)을 종류 불문 제거하고 상속
                # 항목으로 대체해야 UI/클라이언트의 field 단건 조회가 올바른 출처를 본다.
                inherited_fields = {d["field"] for d in decisions}
                existing = [d for d in existing if d.get("field") not in inherited_fields]
                response["consolidation_decisions"] = existing + decisions
            response["contract_inheritance"] = {"applied": True, **summary}
            _log(logging.INFO, "통합", f"계약서 상속 — 필드 +{summary['inherited']}개 (제목 일치 {summary['title_match']})")
        except Exception as e:
            _log(logging.ERROR, "통합", f"계약서 상속 실패 — {type(e).__name__}: {e}", exc_info=True)
            response["contract_inheritance"] = {"applied": False, "error": str(e)}
        return response

    # ------------------------------------------------------------------
    # Response building
    # ------------------------------------------------------------------

    def build_response(
        self, ctx: Dict, *,
        model_name: str, document_type: str,
        ocr_text: str, ocr_provider: str, ocr_model: str,
        llm_result: Dict, ner_model: str, ner_result: Dict,
        consolidate: bool, consolidation_model: str,
        consolidation_result: Optional[Dict],
        consolidation_success: bool, consolidation_error: Optional[str],
        ocr_result: Optional[Dict] = None,
        licensee_party: Optional[str] = None,
        test: Optional[str] = None,
        contract_context: Optional[Dict[str, Any]] = None,
        contract_context_rules: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Assemble the final API response.

        v4 선택 인자(모두 기본 None — 기존 호출자는 그대로):
          ocr_result            OCR 단계 결과 dict → ocr_provider_used('direct_text' 등)·ocr_pages
          licensee_party        이용자 당사자 이름. None 이고 ocr_text 가 있으면 party_rules 로 본문에서 찾는다 (A5)
          test                  "T-001"/"T-002" → test_prediction 의 시험별 널 처리 (A2)
          contract_context      VLM 에 넣은 계약서 문맥 {"title","work_type"} — 출처 기록용 (A4)
          contract_context_rules 계약서 PDF 를 규칙으로 읽은 결과 — 모델 값과 대조용 (§12.7(a))
        """
        total_time = (datetime.now() - ctx["start_time"]).total_seconds()

        response = {
            "success": llm_result.get("success", False),
            "request_id": ctx["request_id"],
            "filename": ctx["filename"],
            "file_size_mb": round(ctx["file_size_mb"], 2),
            # v4 A1: 원본 파일명·SHA-256·바이트 크기 — setup 이 정제 전에 잡아 둔 값 (키 2·20·23)
            "file_name": ctx.get("original_filename", ctx["filename"]),
            "file_hash": ctx.get("sha256"),
            "file_size_bytes": ctx.get("file_size"),
            "model_used": llm_result.get("model_used", model_name),
            "document_type": document_type,
            "metadata": llm_result.get("metadata", {}),
            "confidence": llm_result.get("confidence", 0.0),
            "extraction_time": llm_result.get("extraction_time", 0.0),
            "ocr_text": ocr_text,
            "ocr_provider": ocr_provider,
            "ocr_model": ocr_model,
            # 실제로 쓰인 공급자 — PDF 본문층을 읽었으면 'direct_text' (비용 추정이 쪽당 OCR 을 더하지 않게, B4)
            "ocr_provider_used": (ocr_result or {}).get("ocr_provider") or ocr_provider,
            "ocr_pages": (ocr_result or {}).get("total_pages"),
            "error": llm_result.get("error"),
            # None-safe: image/video (VLM) works have no NER model
            "ner_model": (self.available_ner_models.get(ner_model, {}).get("display_name")
                          if ner_model else None),
            "ner_model_key": ner_model,
            "entities": self._format_ner_entities(ner_result),
            "entity_count": self._count_ner_entities(ner_result),
            "ner_success": ner_result.get("success", False) if ner_result else False,
            "ner_error": ner_result.get("error") if ner_result and not ner_result.get("success", False) else None,
            # NER 추출 방식 — 모델을 올리지 못하면 정규표현식만 쓴다 (ner_system.ner_predict 의 mode/model_loaded/load_error)
            "ner_mode": (ner_result or {}).get("mode"),
            "ner_model_loaded": bool((ner_result or {}).get("model_loaded")),
            "ner_load_error": (ner_result or {}).get("load_error"),
            "ner_entity_count": int((ner_result or {}).get("entity_count")
                                    or (ner_result or {}).get("total_entities") or 0),
            "processing_time": round(total_time, 2),
            "consolidate": consolidate,
            "consolidation_model": consolidation_model if consolidate else None,
            "consolidation_success": consolidation_success,
            "consolidation_error": consolidation_error,
            "consolidated_metadata": consolidation_result.get("consolidated_metadata", {}) if consolidation_result else None,
            "consolidation_decisions": consolidation_result.get("validation_report", {}).get("decisions", []) if consolidation_result else None,
            "consolidation_summary": consolidation_result.get("validation_report", {}).get("summary", {}) if consolidation_result else None,
            "consolidation_confidence": consolidation_result.get("validation_report", {}).get("confidence_score", 0.0) if consolidation_result else None,
            "consolidation_model_used": consolidation_result.get("model_used", consolidation_model) if consolidation_result else None,
            "consolidation_fallback_used": consolidation_result.get("fallback_used", False) if consolidation_result else False,
            "consolidation_degraded": bool(consolidation_result.get("consolidation_degraded")) if consolidation_result else False,
            # 단계별 소요(초) — 실행 로그와 같은 값
            "stage_seconds": {k: v for k, v in {
                "sha256": (ctx.get("timings") or {}).get("sha256"),
                "ocr": (ctx.get("timings") or {}).get("ocr"),
                "extraction": (llm_result or {}).get("_seconds"),
                "ner": (ner_result or {}).get("_seconds"),
                "consolidation": (consolidation_result or {}).get("consolidation_seconds"),
                "contract_rules": (ctx.get("timings") or {}).get("contract_rules"),
            }.items() if v is not None},
            # v4 출처 기록
            "test": test,
            "contract_context": contract_context,
            "contract_context_rules": contract_context_rules,
        }

        # v4 A5: 이용자 당사자(키 5) — 모델 값이 아니라 본문 규칙으로. 새 서식은 '저작권자' 를 인쇄하지
        # 않고 원장의 권리자(한국저작권위원회)가 이용자 당사자로 인쇄된다. 본문이 없으면(이미지·영상) None.
        if licensee_party is None and ocr_text:
            try:
                from module.llm_extraction.party_rules import find_licensee
                found = find_licensee(ocr_text)
                licensee_party = found.get("name")
                response["licensee_party_detail"] = {
                    k: found.get(k) for k in ("source", "signature", "preamble", "agree")}
                src_ko = {"signature": "서명란", "preamble": "전문(前文)"}.get(found.get("source"), found.get("source"))
                if licensee_party:
                    agree = found.get("agree")
                    _log(logging.INFO, "문맥",
                         f"이용자 당사자 규칙 — {licensee_party!r} (출처 {src_ko}"
                         + (" · 서명란·전문 일치" if agree is True else " · 서명란·전문 불일치" if agree is False else "")
                         + ") → 저작권자(키 5)")
                else:
                    _log(logging.WARNING, "문맥", "이용자 당사자 규칙 — 서명란·전문에서 이용자를 찾지 못했습니다 (저작권자 키 5 는 모델 값 대체)")
            except Exception as e:                  # 규칙 실패가 응답을 깨면 안 된다
                _log(logging.WARNING, "문맥", f"이용자 당사자 규칙 추출 실패: {e}")
        response["licensee_party"] = licensee_party
        if licensee_party:
            for _key in ("metadata", "consolidated_metadata"):
                _meta = response.get(_key)
                if isinstance(_meta, dict):
                    _meta["licensee_party"] = licensee_party

        # TTA 표준 중첩 구조(10그룹·66 하위요소)를 함께 낸다. 추출은 평면 그대로 두고
        # 출력 직전에 한 번 접는다 — 추출 자체를 중첩시키면 통합·NER 매핑·채점이 모두
        # 경로 기반으로 바뀌고, 이름 대응표 없이는 채점이 0.48 → 0.00 으로 무너진다.
        # 기존 metadata/consolidated_metadata 는 그대로 두므로 하위 호환이 깨지지 않는다.
        _src = self._final_source(response)
        try:
            from module.llm_extraction.schemas import tta_serializer
            response["tta_metadata"] = tta_serializer.build(_src, response)
        except Exception as e:                      # 직렬화 실패로 추출 결과를 잃지 않는다
            _log(logging.WARNING, "시스템", f"TTA 직렬화 실패: {e}")
            response["tta_metadata"] = None

        # v4 A2: 시험평가용 23키 예측 (keys23 순서). 같은 _src 에서 한 번 더 접는다.
        self._attach_test_prediction(response, _src)

        return response

    @staticmethod
    def _final_source(response: Dict[str, Any], warn: bool = True) -> Dict[str, Any]:
        """직렬화 원본 — 통합 결과를 쓰되, 중재자가 지워버린 필드는 원본 추출로 되살린다.

        비율로 거르면 9→4 같은 손실(설명·키워드·유형이 통째로 사라짐)이 통과한다.
        """
        con = response.get("consolidated_metadata") or {}
        raw = response.get("metadata") or {}
        src = dict(con) if con else dict(raw)
        lost = []
        for k, v in raw.items():
            if v not in (None, "", [], {}) and src.get(k) in (None, "", [], {}):
                src[k] = v
                lost.append(k)
        if lost and warn:
            _log(logging.WARNING, "통합", f"직렬화: 통합에서 사라진 필드를 원본으로 복원 {lost}")
        return src

    def _attach_test_prediction(self, response: Dict[str, Any],
                                src: Optional[Dict[str, Any]] = None) -> None:
        """response["test_prediction"] = eval23 23키 레코드. 실패해도 응답은 깨지지 않는다.

        run() 의 각 분기는 파일 기술속성 보충·계약서 상속·중재자 생략 뒤에 이걸 다시 불러
        최종 metadata 를 반영한다 (멱등).
        """
        try:
            from module.llm_extraction.schemas import eval23_serializer
            if src is None:
                src = self._final_source(response, warn=False)
            response["test_prediction"] = eval23_serializer.build(src, response, response.get("test"))
        except Exception as e:
            _log(logging.WARNING, "시스템", f"23개 항목 직렬화 실패: {e}")
            response["test_prediction"] = None

    @staticmethod
    def save_results(result_dir: Path, response: Dict,
                     consolidation_result: Optional[Dict], consolidation_success: bool):
        """Save results to disk."""
        with open(result_dir / "llm_metadata.json", "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        if consolidation_result and consolidation_success:
            with open(result_dir / "consolidated_metadata.json", "w", encoding="utf-8") as f:
                json.dump(consolidation_result, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Full pipeline (non-streaming convenience method)
    # ------------------------------------------------------------------

    def run_video_response(self, ctx: Dict[str, Any], *, document_type: str = "기타문서",
                           consolidate: bool = True, consolidate_work: bool = True,
                           consolidation_model: str = "alibaba-qwen3.5-122b-a10b",
                           video_prefer: str = "omni", contract_context: Optional[Dict] = None,
                           frame_schedule: str = "v3", test: Optional[str] = None,
                           contract_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """영상 저작물 1건 — 키프레임 → 다중 프레임 VLM 1회 호출 → 응답 조립·저장.

        run() 과 /api/llm-extract 의 스트리밍 경로가 함께 쓴다. 이미지 경로와 같은 모양을 돌려주므로
        통합검증·응답 조립·계약서 상속은 그대로 재사용된다."""
        vid_result = self.run_video(ctx, prefer=video_prefer, context=contract_context,
                                    frame_schedule=frame_schedule)
        vid_ok = vid_result.get("success", False)
        do_con = consolidate and consolidate_work and vid_ok
        if do_con:
            con_result, con_success, con_error = self.run_consolidation(
                vid_result,
                {"success": False, "entities": {}, "total_entities": 0,
                 "extracted_entities": []},
                "", document_type, ctx["result_dir"], consolidation_model,
            )
        else:
            con_result, con_success, con_error = None, False, None
        response = self.build_response(
            ctx,
            model_name=vid_result.get("model_used", "VLM"), document_type=document_type,
            ocr_text="", ocr_provider="(video/keyframes)", ocr_model=None,
            llm_result=vid_result, ner_model=None, ner_result=None,
            consolidate=do_con, consolidation_model=consolidation_model,
            consolidation_result=con_result, consolidation_success=con_success,
            consolidation_error=con_error,
            test=test, contract_context=contract_context,
        )
        response["modality"] = "video"
        response["vlm_backend"] = vid_result.get("model_used")
        response["keyframes"] = vid_result.get("_keyframes")
        response["frame_schedule"] = frame_schedule
        response["consolidate_work"] = consolidate_work
        if vid_ok and not consolidate_work:
            self._skip_work_consolidation(response)
        self._backfill_file_attributes(ctx, response)
        # 계약서→저작물 권리정보 상속 — 이미지와 같다 (HF Space 연동: contract_metadata 동봉 시)
        if vid_ok and contract_metadata:
            self.apply_contract_inheritance(response, contract_metadata)
        self._attach_test_prediction(response)
        self.save_results(ctx["result_dir"], response, con_result, con_success)
        return response

    def run(self, file_bytes: bytes, filename: str, **kwargs) -> Dict[str, Any]:
        """Run the full pipeline in one call (non-streaming). 폴더 시험의 작업 사본은 끝나면(실패해도) 지운다."""
        staged: list = []
        try:
            return self._run(file_bytes, filename, _staged=staged, **kwargs)
        finally:
            for p in staged:
                shutil.rmtree(Path(p).parent, ignore_errors=True)

    def _run(self, file_bytes: bytes, filename: str, _staged: Optional[list] = None, **kwargs) -> Dict[str, Any]:
        model_name = kwargs.get("model_name", "alibaba-qwen3.5-122b-a10b")
        document_type = kwargs.get("document_type", "기타문서")
        ocr_provider = kwargs.get("ocr_provider", "alibaba")
        ocr_model = kwargs.get("ocr_model")
        ner_model = kwargs.get("ner_model", "klue-roberta-large")
        consolidate = kwargs.get("consolidate", True)
        consolidation_model = kwargs.get("consolidation_model", "alibaba-qwen3.5-122b-a10b")
        # --- v4 시험평가 키워드 (기본값 = 오늘의 동작; docs/v4/M2_runner_spec.md §1.A) ---
        consolidate_work = kwargs.get("consolidate_work", True)    # A3: False → 저작물 경로에서 중재자 생략
        contract_context = kwargs.get("contract_context") or None   # A4: {"title","work_type"} → VLM 참고 줄
        frame_schedule = kwargs.get("frame_schedule", "v3")         # A6: 키프레임 수 표
        media_hint = kwargs.get("media")                             # A7: "image" → .pptx 를 이미지로
        test = kwargs.get("test")                                    # A2: "T-001"/"T-002"/None
        if test not in (None, "T-001", "T-002"):
            raise ValueError(f"test 는 'T-001'/'T-002'/None 이어야 합니다: {test!r}")

        ctx = self.setup(file_bytes, filename, source_path=kwargs.get("source_path"))   # 폴더 시험: 작업 사본
        if ctx.get("staged") and _staged is not None:
            _staged.append(ctx["upload_path"])
        self._keep_source(ctx)

        # --- Modality routing: image → VLM (Gemma→Qwen); document/text → existing path ---
        plan = self.run_router(ctx, media_hint=media_hint)
        modality = plan["modality"]

        if modality == "image":
            vlm_result = self.run_vlm(ctx, prefer=kwargs.get("vlm_prefer", "auto"),
                                      context=contract_context, render=plan.get("render"))
            vlm_ok = vlm_result.get("success", False)
            do_con = consolidate and consolidate_work and vlm_ok
            if do_con:
                con_result, con_success, con_error = self.run_consolidation(
                    vlm_result,
                    {"success": False, "entities": {}, "total_entities": 0, "extracted_entities": []},
                    "", document_type, ctx["result_dir"], consolidation_model,
                )
            else:
                con_result, con_success, con_error = None, False, None
            response = self.build_response(
                ctx,
                model_name=vlm_result.get("model_used", "VLM"), document_type=document_type,
                ocr_text="", ocr_provider="(image/VLM)", ocr_model=None,
                llm_result=vlm_result, ner_model=None, ner_result=None,
                consolidate=do_con, consolidation_model=consolidation_model,
                consolidation_result=con_result, consolidation_success=con_success,
                consolidation_error=con_error,
                test=test, contract_context=contract_context,
            )
            response["modality"] = "image"
            response["vlm_backend"] = vlm_result.get("model_used")
            response["vlm_raw"] = vlm_result.get("_vlm_raw")
            response["render"] = vlm_result.get("_render")
            response["consolidate_work"] = consolidate_work
            if vlm_ok and not consolidate_work:
                self._skip_work_consolidation(response)
            # 계약서→저작물 권리정보 상속 (HF Space 연동: contract_metadata 동봉 시)
            if kwargs.get("contract_metadata"):
                self.apply_contract_inheritance(response, kwargs["contract_metadata"])
            self._attach_test_prediction(response)
            self.save_results(ctx["result_dir"], response, con_result, con_success)
            return response

        if modality == "video":
            return self.run_video_response(
                ctx, document_type=document_type, consolidate=consolidate,
                consolidate_work=consolidate_work, consolidation_model=consolidation_model,
                video_prefer=kwargs.get("video_prefer", "omni"), contract_context=contract_context,
                frame_schedule=frame_schedule, test=test, contract_metadata=kwargs.get("contract_metadata"))

        if modality == "audio":
            # ASR 은 1차 구현 제외(계획서 §5): 한국어 음성이 데이터의 0.19%뿐이고
            # 환각 위험이 실측 확인됐다. 기술 메타데이터만 채우고 종료한다.
            response = self.build_response(
                ctx,
                model_name="(none)", document_type=document_type,
                ocr_text="", ocr_provider="(none)", ocr_model=None,
                llm_result={"success": False, "metadata": {},
                            "error": "audio track not enabled (ASR excluded in v1 — 구현계획서 §5)"},
                ner_model=None, ner_result=None,
                consolidate=False, consolidation_model=consolidation_model,
                consolidation_result=None, consolidation_success=False,
                consolidation_error="audio not yet supported",
            )
            response["modality"] = modality
            self._backfill_file_attributes(ctx, response)
            self.save_results(ctx["result_dir"], response, None, False)
            return response

        # --- Document/text branch: existing OCR → LLM ∥ NER → consolidation (unchanged) ---
        ocr_text, ocr_result = self.run_ocr(ctx, ocr_provider, ocr_model)

        # Guard: stop early if OCR returned no text
        if not ocr_text or not ocr_text.strip():
            _log(logging.WARNING, "OCR", "텍스트·OCR — 텍스트 없음 (LLM·NER·통합 생략)", stage="ocr")
            response = self.build_response(
                ctx,
                model_name=model_name, document_type=document_type,
                ocr_text="", ocr_provider=ocr_provider, ocr_model=ocr_model,
                llm_result={"success": False, "metadata": {}, "error": "OCR에서 텍스트를 추출하지 못했습니다"},
                ner_model=ner_model,
                ner_result={"success": False, "entities": {}, "total_entities": 0},
                consolidate=False, consolidation_model=consolidation_model,
                consolidation_result=None, consolidation_success=False,
                consolidation_error="OCR 텍스트 없음",
            )
            self.save_results(ctx["result_dir"], response, None, False)
            return response

        # Run LLM and NER concurrently — they're independent, both only need OCR text
        # LLM is I/O-bound (cloud API), NER is CPU-bound (local model) — no resource conflict
        # 작업 스레드로 문맥(세트 묶음)을 복사해 넘긴다 — 두 스레드의 로그도 같은 세트로 붙는다
        with ThreadPoolExecutor(max_workers=2) as executor:
            llm_future = LC.submit(
                executor, self.run_llm, ocr_text, document_type, ctx["filename"], model_name
            )
            ner_future = LC.submit(
                executor, self.run_ner, ocr_result, ctx["result_dir"], ner_model, ocr_text
            )
            llm_result = llm_future.result()
            ner_result = ner_future.result()

        if consolidate:
            con_result, con_success, con_error = self.run_consolidation(
                llm_result, ner_result, ocr_text, document_type,
                ctx["result_dir"], consolidation_model,
            )
        else:
            _log(logging.INFO, "통합", "통합 생략 — consolidate=False (LLM 결과를 그대로 사용)")
            con_result, con_success, con_error = None, False, None

        response = self.build_response(
            ctx,
            model_name=model_name, document_type=document_type,
            ocr_text=ocr_text, ocr_provider=ocr_provider, ocr_model=ocr_model,
            llm_result=llm_result, ner_model=ner_model, ner_result=ner_result,
            consolidate=consolidate, consolidation_model=consolidation_model,
            consolidation_result=con_result,
            consolidation_success=con_success, consolidation_error=con_error,
            ocr_result=ocr_result, test=test, contract_context=contract_context,
            # 계약서 PDF 면 제호·종별을 규칙으로도 읽어 둔다 — 모델 값과의 대조용(§12.7(a)), 예측값은 아니다
            contract_context_rules=self._contract_rules(ctx),
        )

        # 문서 경로에도 파일 기술속성을 채운다. 이미지 경로는 map_vlm_to_unified 가
        # 이미 채우지만 문서 경로는 비어 있어, 평가에서 파일크기·파일포맷·해상도가
        # 구조적으로 오답이 된다(어문 저작물 825건 해당). 값은 파일에서 결정적으로 나오므로
        # 계약서든 저작물이든 채워 두는 편이 일관된다.
        self._backfill_file_attributes(ctx, response)
        self._attach_test_prediction(response)

        self.save_results(ctx["result_dir"], response, con_result, con_success)
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _skip_work_consolidation(response: Dict) -> None:
        """consolidate_work=False (A3): VLM 매핑 결과(+파일 기술속성)를 그대로 최종값으로 쓴다.

        저작물 경로의 중재자는 NER 입력이 비어 있어 중재할 것이 없는데도 저작물 레그 비용의 97%
        (₩35.4/₩35.6) 와 ~55초를 쓰고, 필드를 지운 적도 있다(9→4). consolidated_metadata 에
        원본을 그대로 복사해 두면 '통합 결과 우선' 으로 읽는 호출자(_final_metadata)도 같은 값을 본다.
        """
        _log(logging.INFO, "통합", "통합 생략 — 저작물 통합 검증 끔 (consolidate_work=False) · VLM 결과와 파일 측정값을 그대로 씁니다")
        response["consolidated_metadata"] = dict(response.get("metadata") or {})
        response["work_consolidation"] = {
            "applied": False,
            "reason": "consolidate_work=False — VLM 결과와 파일 기술속성을 직접 사용 (v4 A3)",
        }

    @staticmethod
    def _contract_rules(ctx: Dict) -> Optional[Dict[str, Any]]:
        """PDF 면 제호·종별을 규칙으로 읽는다 (contract_context.parse). 실패/비PDF → None."""
        if not str(ctx["upload_path"]).lower().endswith(".pdf"):
            return None
        t0 = time.perf_counter()
        try:
            from module.clip_extraction.contract_context import parse
            out = parse(ctx["upload_path"])
            sec = time.perf_counter() - t0
            ctx.setdefault("timings", {})["contract_rules"] = round(sec, 3)
            if isinstance(out, dict):
                # 진단용 'pdf' 는 저장용 정제 이름('…_운수_자판기_1.pdf')이 아니라 원본 파일명으로 — 응답의 file_name 과 같게
                out["pdf"] = ctx.get("original_filename") or out.get("pdf")
                title, wtype = out.get("title"), out.get("work_type")
                if title or wtype:
                    ticked = out.get("work_type_ticked") or []
                    msg = (f"계약서 규칙 파싱 — 제호 {repr(title) if title else '못 찾음'} · 종별 {wtype or '못 찾음'}"
                           + (f" (체크 {', '.join(map(str, ticked))})" if ticked else "") + f" · {sec:.2f}초")
                else:
                    err = f" ({out.get('error')})" if out.get("error") else ""
                    msg = f"계약서 규칙 파싱 — 제호·종별을 찾지 못했습니다{err} · {sec:.2f}초"
                _log(logging.INFO if (title and wtype) else logging.WARNING, "문맥", msg)
            return out
        except Exception as e:
            _log(logging.WARNING, "문맥", f"계약서 규칙 파싱 실패: {e}")
            return None

    @staticmethod
    def _backfill_file_attributes(ctx: Dict, response: Dict) -> None:
        """업로드 파일에서 해상도·파일크기·파일생성일을 산출해 빈 값만 채운다.

        모델이 만든 값을 덮어쓰지 않는다 — 비어 있을 때만 넣는다.
        """
        t0 = time.perf_counter()
        try:
            from module.clip_extraction.technical_metadata import extract_technical_metadata
            tech = extract_technical_metadata(str(ctx["upload_path"]))
        except Exception as e:  # 기술속성 산출 실패가 파이프라인을 죽이면 안 된다
            _log(logging.WARNING, "측정", f"파일 기술속성 산출 실패: {e}")
            return
        sec = time.perf_counter() - t0
        for key in ("metadata", "consolidated_metadata"):
            meta = response.get(key)
            if not isinstance(meta, dict):
                continue
            for field in ("resolution", "file_size", "file_created_date"):
                if meta.get(field) in (None, "", []) and tech.get(field) is not None:
                    meta[field] = tech[field]
            # digital_format 도 이미지 경로(map_vlm_to_unified)에서만 채워지고 있었다.
            # 문서 경로에서 비면 평가의 '파일포맷' 속성이 통째로 오답이 된다.
            if meta.get("digital_format") in (None, "", []):
                ext = os.path.splitext(str(ctx["upload_path"]))[1].lstrip(".").upper()
                if ext:
                    meta["digital_format"] = ext
        shown = response.get("consolidated_metadata") or response.get("metadata") or {}
        if isinstance(shown, dict):
            PipelineOrchestrator._measure_line(
                ctx, {**{k: tech.get(k) for k in ("resolution", "file_size", "file_created_date")},
                      **{k: v for k, v in shown.items() if k in ("digital_format", "resolution", "file_size",
                                                                  "file_created_date") and v not in (None, "", [])}},
                {"technical": tech.get("_technical") or {}}, sec)

    @staticmethod
    def _format_ner_entities(ner_result: Optional[Dict]) -> Dict:
        if not ner_result:
            return {}
        stats = ner_result.get("statistics", {})
        counts = stats.get("entity_types_count", {})
        if counts:
            return counts
        entities = ner_result.get("entities", {})
        if isinstance(entities, dict):
            return {k: len(v) if isinstance(v, list) else v for k, v in entities.items()}
        return {}

    @staticmethod
    def _count_ner_entities(ner_result: Optional[Dict]) -> int:
        if not ner_result:
            return 0
        return ner_result.get("total_entities", 0)
