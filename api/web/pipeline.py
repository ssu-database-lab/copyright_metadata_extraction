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
"""

import json
import logging
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

        return {
            "request_id": request_id,
            "filename": sanitized,
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

    def run_llm(self, ocr_text: str, document_type: str,
                filename: str, model_name: str) -> Dict[str, Any]:
        """Run LLM metadata extraction on OCR text."""
        logger.info(f"LLM extraction start: model={model_name}")
        return self.llm_processor.extract_metadata_from_text(
            text=ocr_text,
            document_type=document_type,
            document_name=filename,
            model_name=model_name,
        )

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

            fallback = (
                "alibaba-qwen3.5-plus"
                if consolidation_model in ("alibaba-qwen3.5-122b-a10b", "alibaba-qwen3-next-80b-a3b-instruct")
                else None
            )
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

    def run_router(self, ctx: Dict) -> Dict[str, Any]:
        """Detect modality (image/video/audio/document/text) + extractor plan."""
        from module.clip_extraction.router import route
        plan = route(str(ctx["upload_path"]))
        logger.info(f"Router: {ctx['filename']} → modality={plan['modality']}")
        return plan

    def run_vlm(self, ctx: Dict, prefer: str = "gemma") -> Dict[str, Any]:
        """Extract visual metadata from an image work via the Gemma(primary)→
        Qwen(backup) VLM fallback chain, mapped to the unified schema.

        Returns an llm_result-shaped dict so it slots straight into
        run_consolidation() and build_response() with no arbiter changes.
        """
        import time
        from module.clip_extraction.vlm.extractor import VLMExtractor
        from module.clip_extraction.schema_mapping import map_vlm_to_unified

        t0 = time.perf_counter()
        try:
            extractor = VLMExtractor(prefer=prefer, max_tokens=2048)
            res = extractor.extract(ctx["upload_path"])
            backend = getattr(res, "backend_used", "?")
            if not res.ok or not res.parse_ok or not res.parsed:
                return {"success": False, "metadata": {}, "confidence": 0.0,
                        "model_used": backend,
                        "error": res.error or "VLM ran but JSON parse failed",
                        "extraction_time": round(time.perf_counter() - t0, 2)}
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
            }
        except Exception as e:
            logger.error(f"VLM extraction error: {e}", exc_info=True)
            return {"success": False, "metadata": {}, "confidence": 0.0,
                    "model_used": "none", "error": str(e),
                    "extraction_time": round(time.perf_counter() - t0, 2)}

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
    ) -> Dict[str, Any]:
        """Assemble the final API response."""
        total_time = (datetime.now() - ctx["start_time"]).total_seconds()

        return {
            "success": llm_result.get("success", False),
            "request_id": ctx["request_id"],
            "filename": ctx["filename"],
            "file_size_mb": round(ctx["file_size_mb"], 2),
            "model_used": llm_result.get("model_used", model_name),
            "document_type": document_type,
            "metadata": llm_result.get("metadata", {}),
            "confidence": llm_result.get("confidence", 0.0),
            "extraction_time": llm_result.get("extraction_time", 0.0),
            "ocr_text": ocr_text,
            "ocr_provider": ocr_provider,
            "ocr_model": ocr_model,
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
        }

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

        ctx = self.setup(file_bytes, filename)

        # --- Modality routing: image → VLM (Gemma→Qwen); document/text → existing path ---
        modality = self.run_router(ctx)["modality"]

        if modality == "image":
            vlm_result = self.run_vlm(ctx, prefer=kwargs.get("vlm_prefer", "gemma"))
            vlm_ok = vlm_result.get("success", False)
            do_con = consolidate and vlm_ok
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
            )
            response["modality"] = "image"
            response["vlm_backend"] = vlm_result.get("model_used")
            response["vlm_raw"] = vlm_result.get("_vlm_raw")
            self.save_results(ctx["result_dir"], response, con_result, con_success)
            return response

        if modality in ("video", "audio"):
            # Video keyframe track = P3; audio has no image-VLM path → guarded out.
            response = self.build_response(
                ctx,
                model_name="(multimodal)", document_type=document_type,
                ocr_text="", ocr_provider="(none)", ocr_model=None,
                llm_result={"success": False, "metadata": {},
                            "error": f"{modality} track not enabled in this build (multimodal P3)"},
                ner_model=None, ner_result=None,
                consolidate=False, consolidation_model=consolidation_model,
                consolidation_result=None, consolidation_success=False,
                consolidation_error=f"{modality} not yet supported",
            )
            response["modality"] = modality
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
        )

        self.save_results(ctx["result_dir"], response, con_result, con_success)
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
