#!/usr/bin/env python3
"""
CLI tool for copyright metadata extraction.

Processes document files (PDF, images) through the full pipeline:
OCR → LLM Extraction → NER → Consolidation

Usage:
    # Single file
    python extract.py document.pdf

    # With options
    python extract.py document.pdf --document-type 계약서 --output ./results/

    # Batch mode (folder)
    python extract.py ./documents/ --output ./results/

    # List available models
    python extract.py --list-models

Output files (per document):
    {output}/{document_name}/
    ├── {document_name}.ocr           — raw OCR extracted text
    ├── llm_metadata.json             — LLM extraction result
    ├── ner_entities.json             — NER entity extraction result
    ├── consolidated_metadata.json    — final consolidated metadata
    └── full_response.json            — complete API response
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure api/ is importable
api_dir = Path(__file__).parent / "api"
if api_dir.exists() and str(api_dir) not in sys.path:
    sys.path.insert(0, str(api_dir))

# Load environment variables
try:
    import module.env_loader  # noqa: F401
except ImportError:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)


# ── NER model registry (same as app.py) ──
NER_MODELS = {
    "klue-roberta-large": {
        "name": "klue/roberta-large",
        "display_name": "KLUE RoBERTa Large",
    },
    "google-bert": {
        "name": "google-bert/bert-base-multilingual-cased",
        "display_name": "Google mBERT",
    },
    "xlm-roberta": {
        "name": "FacebookAI/xlm-roberta-large",
        "display_name": "XLM-RoBERTa Large",
    },
}

# ── Supported file extensions ──
SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"}

# ── Document types ──
DOCUMENT_TYPES = ["계약서", "동의서", "저작재산권 양도동의서", "공공저작물 자유이용허락 동의서", "기타문서"]


def list_models():
    """Print available models."""
    print("\n=== LLM Models (Alibaba DashScope) ===")
    print("  alibaba-qwen3.5-122b-a10b    Qwen3.5-122B (recommended)")
    print("  alibaba-qwen3.5-plus         Qwen3.5-Plus 397B (flagship)")
    print("  alibaba-qwen3.5-flash        Qwen3.5-Flash 35B (cost-effective)")
    print("  alibaba-qwen3-next-80b-a3b-instruct  Qwen3-Next-80B (previous gen)")

    print("\n=== NER Models (local) ===")
    for key, info in NER_MODELS.items():
        print(f"  {key:30s} {info['display_name']}")

    print("\n=== OCR Providers ===")
    print("  alibaba    Alibaba Cloud Qwen3-VL (default)")
    print("  google     Google Cloud Vision")
    print("  mistral    Mistral AI")
    print("  naver      Naver CLOVA OCR")

    print("\n=== Document Types ===")
    for dt in DOCUMENT_TYPES:
        print(f"  {dt}")


def process_file(
    file_path: Path,
    output_dir: Path,
    model_name: str,
    document_type: str,
    ocr_provider: str,
    ocr_model: str,
    ner_model: str,
    consolidate: bool,
    consolidation_model: str,
    run_llm: bool = True,
    run_ner: bool = True,
) -> bool:
    """Process a single file through the pipeline.

    Stages can be selectively enabled:
    - OCR always runs (required for everything else)
    - LLM and NER can be toggled independently
    - Consolidation requires both LLM and NER, or at least one
    """
    from api import ner_predict
    from module.llm_extraction import LLMExtractionProcessor
    from web.pipeline import PipelineOrchestrator

    # Create output directory named after the document
    doc_output = output_dir / file_path.stem
    doc_output.mkdir(parents=True, exist_ok=True)

    # Determine active stages
    stages = ["OCR"]
    if run_llm:
        stages.append("LLM")
    if run_ner:
        stages.append("NER")
    if consolidate and (run_llm or run_ner):
        stages.append("Consolidation")
    elif consolidate and not run_llm and not run_ner:
        consolidate = False  # can't consolidate without any extraction

    print(f"\n{'='*70}")
    print(f"  Processing: {file_path.name}")
    print(f"  Stages: {' → '.join(stages)}")
    print(f"  Document type: {document_type}")
    if run_llm:
        print(f"  LLM model: {model_name}")
    if run_ner:
        print(f"  NER model: {ner_model}")
    print(f"  OCR provider: {ocr_provider}")
    print(f"  Output: {doc_output}")
    print(f"{'='*70}")

    # Initialize pipeline
    llm_processor = LLMExtractionProcessor(output_dir=str(doc_output))
    pipeline = PipelineOrchestrator(
        llm_processor=llm_processor,
        ner_predict_fn=ner_predict,
        available_ner_models=NER_MODELS,
        upload_dir=doc_output / "_uploads",
        results_dir=doc_output / "_results",
    )

    # Read file
    file_bytes = file_path.read_bytes()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    print(f"\n  File size: {file_size_mb:.2f} MB")

    # Run pipeline
    start_time = time.time()
    step = 0
    total_steps = len(stages) + 1  # +1 for saving

    try:
        # ── Stage: OCR (always runs) ──
        step += 1
        print(f"\n  [{step}/{total_steps}] OCR ({ocr_provider})...", end="", flush=True)
        ctx = pipeline.setup(file_bytes, file_path.name)
        ocr_text, ocr_result = pipeline.run_ocr(ctx, ocr_provider, ocr_model)
        print(f" {len(ocr_text)} chars extracted")
        (doc_output / f"{file_path.stem}.ocr").write_text(ocr_text, encoding="utf-8")

        # Guard: stop early if OCR returned no text
        if not ocr_text or not ocr_text.strip():
            elapsed = time.time() - start_time
            print(f"\n\n  ERROR: OCR에서 텍스트를 추출하지 못했습니다.")
            print(f"  다른 OCR 제공자를 선택하거나 파일을 확인해 주세요.")
            print(f"  Time elapsed: {elapsed:.1f}s")
            return False

        # ── Stage: LLM + NER ──
        llm_result = None
        ner_result = None

        if run_llm and run_ner:
            # Both enabled — run concurrently
            step += 1
            print(f"  [{step}/{total_steps}] LLM + NER (concurrent)...", end="", flush=True)
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                llm_future = executor.submit(
                    pipeline.run_llm, ocr_text, document_type, ctx["filename"], model_name
                )
                ner_future = executor.submit(
                    pipeline.run_ner, ocr_result, ctx["result_dir"], ner_model, ocr_text
                )
                llm_result = llm_future.result()
                ner_result = ner_future.result()

            llm_ok = llm_result.get("success", False)
            ner_entities = ner_result.get("total_entities", 0) if ner_result else 0
            print(f" LLM: {'OK' if llm_ok else 'FAILED'}, NER: {ner_entities} entities")

        elif run_llm:
            # LLM only
            step += 1
            print(f"  [{step}/{total_steps}] LLM extraction...", end="", flush=True)
            llm_result = pipeline.run_llm(ocr_text, document_type, ctx["filename"], model_name)
            llm_ok = llm_result.get("success", False)
            print(f" {'OK' if llm_ok else 'FAILED'} ({len(llm_result.get('metadata', {}))} fields)")

        elif run_ner:
            # NER only
            step += 1
            print(f"  [{step}/{total_steps}] NER extraction...", end="", flush=True)
            ner_result = pipeline.run_ner(ocr_result, ctx["result_dir"], ner_model, ocr_text)
            ner_entities = ner_result.get("total_entities", 0) if ner_result else 0
            print(f" {ner_entities} entities")

        # Save extraction results
        if llm_result:
            with open(doc_output / "llm_metadata.json", "w", encoding="utf-8") as f:
                json.dump(llm_result, f, ensure_ascii=False, indent=2)

        if ner_result:
            with open(doc_output / "ner_entities.json", "w", encoding="utf-8") as f:
                json.dump(ner_result, f, ensure_ascii=False, indent=2)

        # ── Stage: Consolidation ──
        con_result, con_success, con_error = None, False, None
        if consolidate and (llm_result or ner_result):
            step += 1
            print(f"  [{step}/{total_steps}] Consolidation...", end="", flush=True)
            # Provide empty defaults for missing extraction
            _llm = llm_result or {"success": False, "metadata": {}}
            _ner = ner_result or {"success": False, "entities": {}, "total_entities": 0, "extracted_entities": []}
            con_result, con_success, con_error = pipeline.run_consolidation(
                _llm, _ner, ocr_text, document_type,
                ctx["result_dir"], consolidation_model,
            )
            print(f" {'OK' if con_success else f'FAILED: {con_error}'}")

            if con_result and con_success:
                with open(doc_output / "consolidated_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(con_result.get("consolidated_metadata", {}), f, ensure_ascii=False, indent=2)

        # ── Save full response ──
        step += 1
        print(f"  [{step}/{total_steps}] Saving results...", end="", flush=True)
        response = pipeline.build_response(
            ctx,
            model_name=model_name, document_type=document_type,
            ocr_text=ocr_text, ocr_provider=ocr_provider, ocr_model=ocr_model,
            llm_result=llm_result or {"success": False, "metadata": {}},
            ner_model=ner_model,
            ner_result=ner_result or {"success": False, "entities": {}, "total_entities": 0},
            consolidate=consolidate, consolidation_model=consolidation_model,
            consolidation_result=con_result,
            consolidation_success=con_success, consolidation_error=con_error,
        )

        with open(doc_output / "full_response.json", "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        print(f" done")

        # Summary
        print(f"\n  --- Results ---")
        print(f"  OCR text:      {len(ocr_text)} chars")
        if llm_result:
            print(f"  LLM metadata:  {'OK' if llm_result.get('success') else 'FAILED'} ({len(llm_result.get('metadata', {}))} fields)")
        else:
            print(f"  LLM metadata:  SKIPPED")
        if ner_result:
            print(f"  NER entities:  {ner_result.get('total_entities', 0)} entities")
        else:
            print(f"  NER entities:  SKIPPED")
        if consolidate:
            print(f"  Consolidation: {'OK' if con_success else 'FAILED'}")
        else:
            print(f"  Consolidation: SKIPPED")
        print(f"  Total time:    {elapsed:.1f}s")
        print(f"  Output:        {doc_output}")

        # List output files
        print(f"\n  Files created:")
        for f in sorted(doc_output.glob("*")):
            if f.is_file() and not f.name.startswith("_"):
                size = f.stat().st_size
                print(f"    {f.name:40s} {size:>8,} bytes")

        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n\n  ERROR: {e}")
        print(f"  Time elapsed: {elapsed:.1f}s")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Copyright metadata extraction CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract.py document.pdf                          # full pipeline
  python extract.py document.pdf -t 계약서                 # specify document type
  python extract.py document.pdf -s ocr                   # OCR only
  python extract.py document.pdf -s ocr+ner               # OCR + NER only
  python extract.py document.pdf -s ocr+llm               # OCR + LLM only
  python extract.py document.pdf -s ocr+llm+ner           # all extraction, no consolidation
  python extract.py document.pdf --no-consolidate         # same as ocr+llm+ner
  python extract.py document.pdf -m alibaba-qwen3.5-plus  # use flagship model
  python extract.py ./documents/ -o ./results/             # batch mode
  python extract.py --list-models                          # list available models
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Input file (PDF, image) or directory for batch processing",
    )
    parser.add_argument(
        "--output", "-o",
        default="./extraction_results",
        help="Output directory (default: ./extraction_results)",
    )
    parser.add_argument(
        "--document-type", "-t",
        default="기타문서",
        choices=DOCUMENT_TYPES,
        help="Document type (default: 기타문서)",
    )
    parser.add_argument(
        "--llm-model", "-m",
        default="alibaba-qwen3.5-122b-a10b",
        help="LLM model for extraction (default: alibaba-qwen3.5-122b-a10b)",
    )
    parser.add_argument(
        "--ner-model",
        default="klue-roberta-large",
        choices=list(NER_MODELS.keys()),
        help="NER model (default: klue-roberta-large)",
    )
    parser.add_argument(
        "--ocr-provider",
        default="alibaba",
        choices=["alibaba", "google", "mistral", "naver"],
        help="OCR provider (default: alibaba)",
    )
    parser.add_argument(
        "--ocr-model",
        default=None,
        help="OCR model override (default: provider's default)",
    )
    parser.add_argument(
        "--consolidation-model",
        default="alibaba-qwen3.5-122b-a10b",
        help="Consolidation model (default: alibaba-qwen3.5-122b-a10b)",
    )
    parser.add_argument(
        "--stages", "-s",
        default="all",
        choices=["all", "ocr", "ocr+ner", "ocr+llm", "ocr+llm+ner"],
        help="Pipeline stages to run (default: all = ocr+llm+ner+consolidation)",
    )
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Skip consolidation step (same as --stages ocr+llm+ner)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit",
    )

    args = parser.parse_args()

    # List models
    if args.list_models:
        list_models()
        return

    # Validate input
    if not args.input:
        parser.error("Input file or directory is required (use --list-models to see options)")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check API key
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIBABA_API_KEY")
    if not api_key and args.llm_model.startswith("alibaba"):
        print("Error: DASHSCOPE_API_KEY not set. Configure in .env file at project root.")
        sys.exit(1)

    # Collect files to process
    if input_path.is_dir():
        files = [
            f for f in sorted(input_path.iterdir())
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            print(f"No supported files found in {input_path}")
            print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
            sys.exit(1)
        print(f"Found {len(files)} file(s) to process")
    else:
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"Unsupported file type: {input_path.suffix}")
            print(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
            sys.exit(1)
        files = [input_path]

    # Determine which stages to run
    stages = args.stages
    do_llm = stages in ("all", "ocr+llm", "ocr+llm+ner")
    do_ner = stages in ("all", "ocr+ner", "ocr+llm+ner")
    do_consolidate = stages == "all" and not args.no_consolidate

    # API key only needed if running LLM or consolidation
    if not api_key and do_llm:
        print("Error: DASHSCOPE_API_KEY not set. Configure in .env file at project root.")
        sys.exit(1)

    # Process files
    total_start = time.time()
    success_count = 0
    fail_count = 0

    for i, file_path in enumerate(files, 1):
        if len(files) > 1:
            print(f"\n[{i}/{len(files)}]", end="")

        ok = process_file(
            file_path=file_path,
            output_dir=output_dir,
            model_name=args.llm_model,
            document_type=args.document_type,
            ocr_provider=args.ocr_provider,
            ocr_model=args.ocr_model,
            ner_model=args.ner_model,
            consolidate=do_consolidate,
            consolidation_model=args.consolidation_model,
            run_llm=do_llm,
            run_ner=do_ner,
        )

        if ok:
            success_count += 1
        else:
            fail_count += 1

    # Final summary
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  COMPLETE")
    print(f"  Processed: {len(files)} file(s)")
    print(f"  Success: {success_count}, Failed: {fail_count}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Output: {output_dir.resolve()}")
    print(f"{'='*70}")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
