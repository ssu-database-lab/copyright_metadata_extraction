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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
import re
import unicodedata

logger = logging.getLogger(__name__)


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

    def setup(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Save the uploaded file and prepare directories.

        Returns a context dict used by all subsequent stages.
        """
        sanitized = safe_filename(filename)
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        upload_path = self.upload_dir / request_id / sanitized
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(file_bytes)

        result_dir = self.results_dir / request_id
        result_dir.mkdir(parents=True, exist_ok=True)

        file_size_mb = len(file_bytes) / (1024 * 1024)
        logger.info(f"Pipeline setup: {sanitized} ({file_size_mb:.2f}MB)")

        # 원본 파일명과 SHA-256 은 정제(safe_filename) **전에** 여기서 잡아 둔다 (v4 키 2·23, A1).
        # 저장용 이름은 공백이 '_' 로 바뀌고 쉼표 등이 빠지므로 파일명 채점에 그대로 쓰면 틀린다.
        # 해시는 바이트가 손에 있는 지금 한 번만 계산한다 — 큰 영상을 다시 읽지 않는다.
        original_filename = unicodedata.normalize("NFC", os.path.basename(filename or "")) or sanitized
        return {
            "request_id": request_id,
            "filename": sanitized,
            "original_filename": original_filename,
            "sha256": hashlib.sha256(file_bytes).hexdigest(),
            "file_size": len(file_bytes),
            "upload_path": upload_path,
            "result_dir": result_dir,
            "file_size_mb": file_size_mb,
            "start_time": datetime.now(),
        }

    # ------------------------------------------------------------------
    # Stage 1: OCR
    # ------------------------------------------------------------------

    def run_ocr(self, ctx: Dict, ocr_provider: str, ocr_model: str = None) -> Tuple[str, Dict]:
        """Run OCR on the uploaded file. Returns (ocr_text, ocr_result_dict)."""
        from module.ocr import UniversalOCRProcessor

        ocr_dir = ctx["result_dir"] / "ocr"
        ocr_dir.mkdir(parents=True, exist_ok=True)

        try:
            processor = UniversalOCRProcessor(
                provider=ocr_provider, output_dir=str(ocr_dir), model=ocr_model
            )
            ocr_result = processor.process_single_file(str(ctx["upload_path"]))

            if ocr_result.get("status") == "success":
                ocr_text = ocr_result.get("full_text", "")
                logger.info(f"OCR complete: {len(ocr_text)} chars")
            else:
                ocr_text = ""
                logger.warning(f"OCR failed: {ocr_result.get('error')}")

        except Exception as e:
            logger.warning(f"OCR error: {e}")
            ocr_text = ""
            ocr_result = {"status": "failed", "error": str(e)}

        return ocr_text, ocr_result

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
        logger.info(f"LLM extraction start: model={model_name}")
        result = self.llm_processor.extract_metadata_from_text(
            text=ocr_text,
            document_type=document_type,
            document_name=filename,
            model_name=model_name,
        )
        metadata = (result or {}).get("metadata")
        if isinstance(metadata, dict):
            hallucinated = [f for f in self._FILE_DERIVED_FIELDS
                            if metadata.get(f) not in (None, "", [])]
            for field in hallucinated:
                metadata[field] = None
            if hallucinated:
                logger.warning(
                    f"LLM이 파일 유래 필드를 본문에서 추론함 → 제거: {hallucinated}")
        return result

    # ------------------------------------------------------------------
    # Stage 3: NER
    # ------------------------------------------------------------------

    def run_ner(self, ocr_result: Dict, result_dir: Path,
                ner_model: str, ocr_text: str) -> Dict[str, Any]:
        """Run NER entity extraction."""
        logger.info(f"NER start: model={ner_model}")
        ner_dir = result_dir / "ner"

        try:
            ocr_output_dir = ocr_result.get("output_directory")

            if not ocr_output_dir or ocr_result.get("status") != "success":
                ocr_output_dir = result_dir / "ocr"
                ocr_output_dir.mkdir(parents=True, exist_ok=True)
                (ocr_output_dir / "temp.ocr").write_text(ocr_text, encoding="utf-8")

            ner_model_name = self.available_ner_models[ner_model]["name"]
            ner_result = self.ner_predict(
                str(ocr_output_dir), str(ner_dir),
                model_name=ner_model_name, debug=False,
            )
            logger.info(f"NER complete: {ner_result.get('total_entities', 0)} entities")
            return ner_result

        except Exception as e:
            logger.warning(f"NER error: {e}")
            return {
                "success": False, "error": str(e),
                "entity_types": {}, "total_entities": 0,
                "extracted_entities": [],
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

        llm_ok = llm_result and llm_result.get("success", False)
        ner_ok = ner_result and ner_result.get("success", False)

        if not llm_ok and not ner_ok:
            logger.warning("Both LLM and NER failed — skipping consolidation")
            return None, False, "Both LLM and NER extraction failed"

        if not llm_ok:
            logger.warning("LLM extraction failed — consolidation will use NER results only")
            llm_result = llm_result or {"success": False, "metadata": {}}

        if not ner_ok:
            logger.warning("NER extraction failed — consolidation will use LLM results only")
            ner_result = ner_result or {"success": False, "entities": {}, "total_entities": 0, "extracted_entities": []}

        logger.info(f"Consolidation start (LLM: {'OK' if llm_ok else 'FAILED'}, NER: {'OK' if ner_ok else 'FAILED'})")

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

            # 폴백 중재자. 환경변수로 바꿀 수 있게 한다 — 기본값 qwen3.5-plus 는
            # 2026-09 측정에서 통합검증 프롬프트를 90초 제한 안에 끝내지 못해
            # (실측 94.9초) 전 호출이 실패하고 있었다. CONSOLIDATION_TIMEOUT_SEC 을
            # 함께 올리지 않으면 이 폴백은 동작하지 않는다.
            fallback = os.getenv("CONSOLIDATION_FALLBACK_MODEL", "alibaba-qwen3.5-plus")
            if fallback.lower() in ("", "none", "off"):
                fallback = None
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

            if result.get("success", False):
                if result.get("consolidation_degraded"):
                    # LLM 중재 없이 규칙 기반으로 병합된 결과다. 예외가 아니라 정상 반환이므로
                    # 여기서 명시하지 않으면 저장 결과만 보고는 구분할 수 없다.
                    logger.error(
                        "통합검증이 저하 모드로 완료됨 — LLM 중재 없음. "
                        f"사유: {result.get('degraded_reason')}"
                    )
                else:
                    logger.info("Consolidation complete")
                return result, True, None
            else:
                error = result.get("error", "Consolidation failed")
                logger.warning(f"Consolidation failed: {error}")
                return result, False, error

        except Exception as e:
            logger.error(f"Consolidation error: {e}", exc_info=True)
            return None, False, str(e)

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
        extra = f" (render={plan['render']})" if plan.get("render") else ""
        logger.info(f"Router: {ctx['filename']} → modality={plan['modality']}{extra}")
        return plan

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
                try:
                    render_first_slide(image_path, png)
                except PptxRenderError as e:
                    logger.error(f"PPTX 1쪽 렌더링 실패 — VLM 생략: {e}")
                    return {"success": False, "metadata": {}, "confidence": 0.0,
                            "model_used": "none", "error": f"PPTX 1쪽 렌더링 불가: {e}",
                            "extraction_time": round(time.perf_counter() - t0, 2),
                            "_render": {"method": "pptx_slide1", "ok": False}}
                image_path = png
                render_info = {"method": "pptx_slide1", "ok": True, "png": str(png)}

            extractor = VLMExtractor(prefer=prefer, max_tokens=4096)   # omni-flash 출력이 2048 을 넘는 경우 8% (2026-10-11 실측)
            res = extractor.extract(image_path, context=context)
            backend = getattr(res, "backend_used", "?")
            if not res.ok or not res.parse_ok or not res.parsed:
                return {"success": False, "metadata": {}, "confidence": 0.0,
                        "model_used": backend,
                        "error": res.error or "VLM ran but JSON parse failed",
                        "extraction_time": round(time.perf_counter() - t0, 2),
                        "_render": render_info}
            # 기술 메타데이터·SHA256 은 항상 **원본** 파일에서 — 렌더링한 PNG 가 아니다.
            unified = map_vlm_to_unified(res.parsed, str(ctx["upload_path"]))
            file_meta = unified.pop("_file_meta", {})
            logger.info(f"VLM extraction complete via {backend}: "
                        f"work_type={unified.get('work_type')}")
            return {
                "success": True,
                "metadata": unified,
                "confidence": 0.7,  # VLM does not self-score; fixed visual-extraction prior
                "model_used": backend,
                "extraction_time": round(time.perf_counter() - t0, 2),
                "_file_meta": file_meta,
                "_vlm_raw": res.parsed,
                "_render": render_info,
                "_prompt_context": dict(context) if context else None,
            }
        except Exception as e:
            logger.error(f"VLM extraction error: {e}", exc_info=True)
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
            kf = extract_keyframes(str(ctx["upload_path"]), str(frame_dir), schedule=frame_schedule)
            if not kf.get("frames"):
                return {"success": False, "metadata": {}, "confidence": 0.0,
                        "model_used": "none",
                        "error": f"키프레임 추출 실패: {kf.get('error') or '프레임 없음'}",
                        "extraction_time": round(time.perf_counter() - t0, 2),
                        "_keyframes": kf}

            paths = [f["path"] for f in kf["frames"]]
            times = [f["t"] for f in kf["frames"]]
            extractor = VLMExtractor(prefer=prefer, max_tokens=4096)   # omni-flash 출력이 2048 을 넘는 경우 8% (2026-10-11 실측)
            res = extractor.extract_video(paths, times, label=ctx["filename"], context=context)
            backend = getattr(res, "backend_used", "?")
            if not res.ok or not res.parse_ok or not res.parsed:
                return {"success": False, "metadata": {}, "confidence": 0.0,
                        "model_used": backend,
                        "error": res.error or "VLM ran but JSON parse failed",
                        "extraction_time": round(time.perf_counter() - t0, 2),
                        "_keyframes": kf}

            unified = map_vlm_to_unified(res.parsed, str(ctx["upload_path"]))
            file_meta = unified.pop("_file_meta", {})
            logger.info(f"Video extraction via {backend}: {kf['kept']} frames "
                        f"({kf['duration']}s), work_type={unified.get('work_type')}")
            return {
                "success": True, "metadata": unified, "confidence": 0.7,
                "model_used": backend,
                "extraction_time": round(time.perf_counter() - t0, 2),
                "_file_meta": file_meta, "_vlm_raw": res.parsed,
                "_keyframes": {k: v for k, v in kf.items() if k != "frames"},
                "_frame_times": times,
                "_prompt_context": dict(context) if context else None,
            }
        except Exception as e:
            logger.error(f"Video extraction error: {e}", exc_info=True)
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
            logger.info(f"Contract inheritance: +{summary['inherited']} fields "
                        f"(title_match={summary['title_match']})")
        except Exception as e:
            logger.error(f"Contract inheritance failed: {e}", exc_info=True)
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
            except Exception as e:                  # 규칙 실패가 응답을 깨면 안 된다
                logger.warning(f"이용자 당사자 규칙 추출 실패: {e}")
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
            logger.warning(f"TTA 직렬화 실패: {e}")
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
            logger.warning(f"직렬화: 통합에서 사라진 필드를 원본으로 복원 {lost}")
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
            logger.warning(f"eval23 직렬화 실패: {e}")
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

    def run(self, file_bytes: bytes, filename: str, **kwargs) -> Dict[str, Any]:
        """Run the full pipeline in one call (non-streaming)."""
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

        ctx = self.setup(file_bytes, filename)

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
            # 키프레임 → 다중 프레임 VLM 1회 호출. 이미지 경로와 동일한 모양을
            # 돌려주므로 통합검증·응답조립은 그대로 재사용된다.
            vid_result = self.run_video(ctx, prefer=kwargs.get("video_prefer", "omni"),
                                        context=contract_context, frame_schedule=frame_schedule)
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
            response["modality"] = modality
            response["keyframes"] = vid_result.get("_keyframes")
            response["frame_schedule"] = frame_schedule
            response["consolidate_work"] = consolidate_work
            if vid_ok and not consolidate_work:
                self._skip_work_consolidation(response)
            self._backfill_file_attributes(ctx, response)
            self._attach_test_prediction(response)
            self.save_results(ctx["result_dir"], response, con_result, con_success)
            return response

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
            logger.warning("OCR returned empty text — skipping LLM, NER, and consolidation")
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
        with ThreadPoolExecutor(max_workers=2) as executor:
            llm_future = executor.submit(
                self.run_llm, ocr_text, document_type, ctx["filename"], model_name
            )
            ner_future = executor.submit(
                self.run_ner, ocr_result, ctx["result_dir"], ner_model, ocr_text
            )
            llm_result = llm_future.result()
            ner_result = ner_future.result()

        if consolidate:
            con_result, con_success, con_error = self.run_consolidation(
                llm_result, ner_result, ocr_text, document_type,
                ctx["result_dir"], consolidation_model,
            )
        else:
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
        try:
            from module.clip_extraction.contract_context import parse
            out = parse(ctx["upload_path"])
            if isinstance(out, dict):
                # 진단용 'pdf' 는 저장용 정제 이름('…_운수_자판기_1.pdf')이 아니라 원본 파일명으로 — 응답의 file_name 과 같게
                out["pdf"] = ctx.get("original_filename") or out.get("pdf")
            return out
        except Exception as e:
            logger.warning(f"계약서 규칙 파싱 실패: {e}")
            return None

    @staticmethod
    def _backfill_file_attributes(ctx: Dict, response: Dict) -> None:
        """업로드 파일에서 해상도·파일크기·파일생성일을 산출해 빈 값만 채운다.

        모델이 만든 값을 덮어쓰지 않는다 — 비어 있을 때만 넣는다.
        """
        try:
            from module.clip_extraction.technical_metadata import extract_technical_metadata
            tech = extract_technical_metadata(str(ctx["upload_path"]))
        except Exception as e:  # 기술속성 산출 실패가 파이프라인을 죽이면 안 된다
            logger.warning(f"파일 기술속성 산출 실패: {e}")
            return
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
