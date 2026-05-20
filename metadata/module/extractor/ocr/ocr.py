"""OCR 추출 — Qwen3-VL-8B-Instruct-AWQ-4bit (로컬 GPU, Apache-2.0).

엔진 선택지 없음. 한국어 OCR 성능과 상업 사용을 모두 만족하는 단일 엔진.
PDF 는 PyMuPDF 로 페이지 이미지화 → Qwen3-VL chat-VL inference 로 텍스트 추출.
"""
from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from module.parts import directory


class OCRDeviceError(RuntimeError):
    """OCR 이 GPU 로 실행될 수 없을 때 발생."""


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

QWEN3VL_DEFAULT_REPO = "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"
QWEN3VL_DEFAULT_PROMPT = (
    "이 문서 이미지의 모든 텍스트를 정확히 추출해주세요. "
    "한국어 원문 그대로, 줄바꿈과 단락 구조를 유지하고, "
    "추가 설명 없이 텍스트만 출력하세요."
)
QWEN3VL_MAX_NEW_TOKENS = 4096
QWEN3VL_RENDER_ZOOM = 2.0  # PDF 2x = ~144 DPI


def _load_full_config() -> Dict[str, Any]:
    config_path = Path("configs/labels.yaml")
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_ocr_config() -> Dict[str, Any]:
    cfg = _load_full_config()
    return cfg.get("ocr", {}) if isinstance(cfg, dict) else {}


def _default_paths() -> Dict[str, Any]:
    """configs/labels.yaml::paths — 인자 미지정 시 ocr_extract 가 사용하는 기본 path."""
    cfg = _load_full_config()
    return cfg.get("paths", {}) if isinstance(cfg, dict) else {}


def _qwen3vl_settings() -> Dict[str, Any]:
    cfg = _load_ocr_config()
    qw = cfg.get("qwen3vl") if isinstance(cfg.get("qwen3vl"), dict) else {}
    return {
        "model_id": os.getenv("METADATA_QWEN3VL_REPO")
            or qw.get("model_id") or QWEN3VL_DEFAULT_REPO,
        "prompt": os.getenv("METADATA_QWEN3VL_PROMPT")
            or qw.get("prompt") or QWEN3VL_DEFAULT_PROMPT,
        "max_new_tokens": int(
            os.getenv("METADATA_QWEN3VL_MAX_NEW")
            or qw.get("max_new_tokens") or QWEN3VL_MAX_NEW_TOKENS
        ),
        "render_zoom": float(
            os.getenv("METADATA_QWEN3VL_ZOOM")
            or qw.get("render_zoom") or QWEN3VL_RENDER_ZOOM
        ),
    }


def _resolve_device(device: Optional[str] = None) -> str:
    """GPU device 결정. CPU 는 거부."""
    chosen = device or os.getenv("METADATA_OCR_DEVICE") or _load_ocr_config().get("device") or "cuda:0"
    resolved = str(chosen).strip()
    kind = resolved.split(":", 1)[0].lower()
    if kind == "cpu":
        raise OCRDeviceError(
            f"OCR device must be GPU, got {resolved!r}. CPU OCR is disabled."
        )
    if kind == "gpu":
        resolved = "cuda" + resolved[3:]
        kind = "cuda"
    if kind != "cuda":
        raise OCRDeviceError(f"OCR device must be cuda:*, got {resolved!r}.")
    return resolved


def _require_torch_cuda(device: str) -> None:
    try:
        import torch
    except Exception as exc:
        raise OCRDeviceError("OCR requires CUDA-enabled PyTorch.") from exc
    if not torch.cuda.is_available():
        raise OCRDeviceError("OCR requires CUDA but torch.cuda.is_available() is False.")
    if ":" in device:
        idx = int(device.split(":", 1)[1])
        if idx >= torch.cuda.device_count():
            raise OCRDeviceError(
                f"OCR device {device!r} unavailable; torch sees "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )


# ---------------------------------------------------------------------------
# Qwen3-VL pipeline (singleton)
# ---------------------------------------------------------------------------

_QWEN3VL_RUNTIME: Optional[Dict[str, Any]] = None


def _create_ocr_pipeline(device: Optional[str] = None) -> Dict[str, Any]:
    """Qwen3-VL 모델/프로세서 로드. 동일 (model_id, device) 면 캐시 재사용."""
    global _QWEN3VL_RUNTIME

    settings = _qwen3vl_settings()
    cuda_device = _resolve_device(device)
    _require_torch_cuda(cuda_device)

    if (
        _QWEN3VL_RUNTIME is not None
        and _QWEN3VL_RUNTIME["model_id"] == settings["model_id"]
        and _QWEN3VL_RUNTIME["device"] == cuda_device
    ):
        _QWEN3VL_RUNTIME["settings"] = settings
        return _QWEN3VL_RUNTIME

    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise OCRDeviceError(
            "Qwen3-VL needs transformers>=4.57. Run: pip install -r requirements.txt"
        ) from exc

    try:
        import fitz  # noqa: F401 — used by _pdf_to_pil_pages
    except ImportError as exc:
        raise OCRDeviceError("Qwen3-VL OCR needs PyMuPDF (pip install pymupdf).") from exc

    try:
        import qwen_vl_utils  # noqa: F401
    except ImportError as exc:
        raise OCRDeviceError("Qwen3-VL OCR needs qwen-vl-utils.") from exc

    print(f"OCR engine: Qwen3-VL ({settings['model_id']}, device={cuda_device})")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        settings["model_id"],
        torch_dtype="auto",
        device_map={"": cuda_device} if ":" in cuda_device else "auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(settings["model_id"])

    _QWEN3VL_RUNTIME = {
        "model": model,
        "processor": processor,
        "device": cuda_device,
        "model_id": settings["model_id"],
        "settings": settings,
        "torch": torch,
    }
    return _QWEN3VL_RUNTIME


# ---------------------------------------------------------------------------
# inference helpers
# ---------------------------------------------------------------------------

def _pdf_to_pil_pages(path: Path, zoom: float) -> List[Any]:
    import fitz
    from PIL import Image

    pages: List[Any] = []
    with fitz.open(str(path)) as doc:
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            if img.mode != "RGB":
                img = img.convert("RGB")
            pages.append(img)
    return pages


def _load_image_pages(path: Path) -> List[Any]:
    from PIL import Image

    img = Image.open(str(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return [img]


def _qwen3vl_ocr_one_image(runtime: Dict[str, Any], image: Any) -> str:
    from qwen_vl_utils import process_vision_info

    processor = runtime["processor"]
    model = runtime["model"]
    settings = runtime["settings"]
    torch = runtime["torch"]

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": settings["prompt"]},
        ],
    }]

    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=settings["max_new_tokens"],
            do_sample=False,
        )
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return (text or "").strip()


def _run_ocr_for_file(
    runtime: Dict[str, Any], file_path: Path,
) -> tuple[List[str], Dict[str, List[Dict[str, Any]]], int]:
    settings = runtime["settings"]
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        pil_pages = _pdf_to_pil_pages(file_path, settings["render_zoom"])
    else:
        pil_pages = _load_image_pages(file_path)

    total = len(pil_pages)
    page_texts: List[str] = []
    labeled_metadata: Dict[str, List[Dict[str, Any]]] = {}

    for idx, image in enumerate(pil_pages):
        text = _qwen3vl_ocr_one_image(runtime, image)
        if not text:
            continue
        page_texts.append(f"--- Page {idx + 1}/{total} ---\n{text}")
        labeled_metadata.setdefault("text", []).append({
            "page_index": idx,
            "item_index": 0,
            "label": "text",
            "content": text,
            "bbox": None,
        })

    return page_texts, labeled_metadata, total


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def needs_ocr(file_path: Path) -> bool:
    """파일이 OCR 이 필요한가 (PDF·이미지: True, txt/md/csv: False)."""
    if not file_path.exists():
        return False
    text_ext = {".txt", ".md", ".csv"}
    ocr_ext = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}
    ext = file_path.suffix.lower()
    if ext in text_ext:
        return False
    if ext in ocr_ext:
        return True
    return False


def extract_text_from_file(
    runtime: Dict[str, Any],
    file_path: str,
    save_path: Optional[str] = None,
) -> tuple[str, Dict[str, Any]]:
    """파일 또는 디렉터리 OCR. 평문은 ``save_path/result/`` 에 저장하고 메타데이터를 반환.

    단일 파일 → ``(text, metadata)``, 디렉터리 → ``("", {rel_path: metadata})``.
    """
    input_p = Path(file_path)
    if not input_p.exists():
        raise ValueError(f"Input path does not exist: {file_path}")
    if not save_path:
        raise ValueError("save_path must be provided")

    output_root = Path(save_path)
    result_dir = output_root / "result"

    if input_p.is_file():
        page_texts, labeled_metadata, total_pages = _run_ocr_for_file(runtime, input_p)
        plain_file = result_dir / f"{input_p.stem}.txt"
        plain_file.parent.mkdir(parents=True, exist_ok=True)
        plain_file.write_text("\n\n".join(page_texts), encoding="utf-8")
        metadata = {
            "source_file": str(input_p),
            "total_pages": total_pages,
            "labels": labeled_metadata,
        }
        return "\n\n".join(page_texts), metadata

    if input_p.is_dir():
        files = list(directory.iter_document_files(input_p))
        if not files:
            print("No supported document files found.")
            return "", {}
        print(f"Found {len(files)} files.")
        all_metadata: Dict[str, Dict[str, Any]] = {}
        for file in files:
            print(f"Processing: {file.relative_to(input_p)}")
            try:
                page_texts, labeled_metadata, total_pages = _run_ocr_for_file(runtime, file)
                rel_path = file.relative_to(input_p)
                plain_file = result_dir / rel_path.with_suffix(".txt")
                plain_file.parent.mkdir(parents=True, exist_ok=True)
                plain_file.write_text("\n\n".join(page_texts), encoding="utf-8")
                all_metadata[str(rel_path)] = {
                    "source_file": str(file),
                    "total_pages": total_pages,
                    "labels": labeled_metadata,
                }
            except OCRDeviceError:
                raise
            except Exception as e:
                print(f"Failed: {e}")
        return "", all_metadata

    raise ValueError(f"Input path does not exist: {file_path}")


def process_file_for_metadata(
    file_path: Path,
    output_path: Optional[str] = None,
) -> tuple[str, Dict[str, Any]]:
    """단일 파일 OCR (NER predict 단계가 사용).

    Args:
        file_path: 처리할 파일.
        output_path: OCR 결과 저장 루트. ``None`` 이면 시스템 임시 디렉터리에
            저장 후 자동 cleanup.

    Returns:
        ``(raw_text, ocr_labeled_metadata)``. 텍스트 파일이면 metadata 는 빈 dict.
        OCR 실패 시 ``("", {})``.
    """
    if not needs_ocr(file_path):
        return file_path.read_text(encoding="utf-8"), {}

    print(f"OCR required for: {file_path}")
    try:
        runtime = _create_ocr_pipeline()

        def _run(save_root: Path) -> tuple[str, Dict[str, Any]]:
            text, ocr_metadata = extract_text_from_file(
                runtime, str(file_path), save_path=str(save_root),
            )
            labels = ocr_metadata.get("labels", {}) if ocr_metadata else {}
            if ocr_metadata:
                mfile = save_root / "result" / "metadata" / f"{file_path.stem}.json"
                mfile.parent.mkdir(parents=True, exist_ok=True)
                with open(mfile, "w", encoding="utf-8") as f:
                    json.dump(ocr_metadata, f, ensure_ascii=False, indent=2)
            return text, labels

        if output_path is None:
            with tempfile.TemporaryDirectory() as tmp:
                return _run(Path(tmp))
        save_root = Path(output_path)
        save_root.mkdir(parents=True, exist_ok=True)
        return _run(save_root)

    except OCRDeviceError:
        raise
    except Exception as e:
        print(f"⚠️ OCR 실패 (정규식 처리만 진행): {e}")
        return "", {}


def ocr_extract(
    in_path: Optional[str] = None,
    out_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    """디렉터리/파일 일괄 OCR. 모델은 1회만 로드되어 모든 파일에 재사용.

    인자 미지정 시 ``configs/labels.yaml::paths`` 의 ``input`` / ``ocr_output`` 사용.
    텍스트는 ``out_path/result/`` 에 저장.
    ``metadata_path`` 가 지정되면 페이지별 라벨 JSON 을 그 아래에 저장.
    """
    paths = _default_paths()
    if in_path is None:
        in_path = paths.get("input")
    if out_path is None:
        out_path = paths.get("ocr_output")
    if in_path is None or out_path is None:
        raise ValueError(
            "ocr_extract: in_path / out_path 가 None 이고 configs/labels.yaml::paths "
            "에도 input/ocr_output 가 없습니다."
        )

    input_p = Path(in_path)
    output_p = Path(out_path)
    if not input_p.exists():
        print(f"Error: Input path does not exist: {in_path}")
        return

    metadata_dir = Path(metadata_path) if metadata_path else None
    runtime = _create_ocr_pipeline(device)

    try:
        if input_p.is_file():
            print(f"OCR Processing: {input_p}")
            save_root = output_p.parent if output_p.suffix else output_p
            _, metadata = extract_text_from_file(runtime, str(in_path), save_path=str(save_root))
            print(f"Plain saved to: {save_root / 'result' / (input_p.stem + '.txt')}")
            if metadata_dir is not None:
                mfile = metadata_dir / f"{input_p.stem}.json"
                mfile.parent.mkdir(parents=True, exist_ok=True)
                with open(mfile, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                print(f"Metadata saved to: {mfile}")
        else:
            print(f"OCR Processing(directory): {input_p}")
            _, all_metadata = extract_text_from_file(runtime, str(in_path), save_path=str(out_path))
            print(f"Plain saved to: {output_p / 'result'}")
            if metadata_dir is not None:
                saved = 0
                for rel, metadata in all_metadata.items():
                    mfile = metadata_dir / Path(rel).with_suffix(".json")
                    mfile.parent.mkdir(parents=True, exist_ok=True)
                    with open(mfile, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    saved += 1
                print(f"Metadata saved to: {metadata_dir} ({saved} files)")
        print("OCR extraction completed.")
    except Exception as e:
        traceback.print_exc()
        print(f"OCR extraction failed: {e}")
