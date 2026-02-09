"""API 모듈: main.py에서 사용하는 함수만 노출"""
import json

from pathlib import Path
from typing import Optional, Dict, Any, List

from module.extractor import ocr as ocr_module
from module.parts import directory
from module.extractor import text as text_module
from module.extractor.ner import ner_extractor
from module.extractor.ner.base import load_labels_from_yaml
from module.extractor import regular as regular_module
from module.extractor.llm import merge_regular_ner
from module.parts.types import Decision


def ocr_extract(
    in_path: str,
    out_path: str,
    metadata_path: Optional[str] = None,
) -> None:
    """
    OCR 추출 API: extractor.ocr.ocr_extract 래퍼.
    """
    return ocr_module.ocr_extract(
        in_path=in_path,
        out_path=out_path,
        metadata_path=metadata_path,
    )


def ner_predict(
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    sentences: Optional[List[Dict[str, Any]]] = None,
    tokens: Optional[List[Dict[str, Any]]] = None,
    out_dir: str = "data/out/results",
    **kwargs,
) -> Dict[str, Any]:
    if text is None and file_path is None and (sentences is None or tokens is None):
        raise ValueError("text/file_path 또는 sentences/tokens 중 하나는 제공되어야 합니다.")

    file_path_obj = Path(file_path) if file_path else None
    if text is None and file_path_obj:
        if file_path_obj.suffix.lower() in [".txt", ".md"]:
            text = file_path_obj.read_text(encoding="utf-8")
        else:
            text, _ = ocr_module.process_file_for_metadata(file_path_obj, use_temp_dir=True)

    if text is not None:
        struct = text_module.read_text(text)
        sentences = struct.get("sentences", [])
        tokens = struct.get("tokens", [])

    decisions = ner_extractor(sentences=sentences or [], tokens=tokens or [], **kwargs)
    print(
        f"NER: sentences={len(sentences or [])}, tokens={len(tokens or [])}, decisions={len(decisions)}"
    )
    ner_labels, _ = load_labels_from_yaml()
    ner_labels = ner_labels or sorted({d.label for d in decisions})
    aggregated = {label: [] for label in ner_labels}

    for decision in decisions:
        label = decision.label
        value = decision.value
        if not isinstance(value, str):
            value = str(value)
        if label in aggregated and value and value.strip() and value not in aggregated[label]:
            aggregated[label].append(value)

    for label in aggregated:
        if not aggregated[label]:
            aggregated[label] = ["N/A"]

    out_dir_path = directory.ensure_outdir(out_dir)
    out_file = directory.default_outfile(
        file_path=str(file_path_obj) if file_path_obj else None,
        out_dir=out_dir_path,
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)

    print(f"NER 결과 저장: {out_file}")
    return aggregated


def ner_metadata_extract(
    *,
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    out_dir: str = "data/out/results",
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    OCR + NER 통합 추출 (metadata_extract 별칭).
    """
    return metadata_extract(
        text=text,
        file_path=file_path,
        out_dir=out_dir,
        threshold=threshold,
    )


def llm_predict(
    *,
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    LLM 단독 추출 (현재는 빈 함수).
    """
    print("llm_predict: 아직 구현되지 않았습니다.")
    return {}


def llm_metadata_extract(
    *,
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    out_dir: str = "data/out/results",
    **kwargs,
) -> Dict[str, Any]:
    """
    OCR + NER + LLM 통합 추출 (현재는 빈 함수).
    """
    print("llm_metadata_extract: 아직 구현되지 않았습니다.")
    return {}


def _aggregate_decisions(
    decisions: List[Decision],
    labels: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    if labels is None:
        labels = sorted({d.label for d in decisions})
    aggregated = {label: [] for label in labels}
    for decision in decisions:
        label = decision.label
        value = decision.value
        if not isinstance(value, str):
            value = str(value)
        if label in aggregated and value and value.strip() and value not in aggregated[label]:
            aggregated[label].append(value)
    for label in aggregated:
        if not aggregated[label]:
            aggregated[label] = ["N/A"]
    return aggregated


def _aggregate_ocr_metadata(ocr_labeled_metadata: Dict[str, Any]) -> Dict[str, List[str]]:
    aggregated: Dict[str, List[str]] = {}
    for label, items in (ocr_labeled_metadata or {}).items():
        values: List[str] = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    content = item.get("content")
                else:
                    content = None
                if content is None:
                    content = str(item)
                if isinstance(content, str) and content.strip() and content not in values:
                    values.append(content)
        aggregated[label] = values if values else ["N/A"]
    return aggregated


def _merge_metadata(
    gliner_metadata: Dict[str, List[str]],
    ocr_metadata: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    merged = {**gliner_metadata}
    for label, values in ocr_metadata.items():
        if label not in merged:
            merged[label] = values
            continue
        if merged[label] == ["N/A"]:
            merged[label] = []
        for value in values:
            if value != "N/A" and value not in merged[label]:
                merged[label].append(value)
        if not merged[label]:
            merged[label] = ["N/A"]
    return merged


def metadata_extract(
    *,
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    out_dir: str = "data/out/results",
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    메타데이터 추출:
    - 텍스트 파일은 OCR 없이 GLiNER2 추출
    - 이미지/PDF는 OCR 후 임시 저장(/temp), GLiNER2 추출과 병합
    """
    if text is None and file_path is None:
        raise ValueError("text 또는 file_path 중 하나는 제공되어야 합니다.")

    file_path_obj = Path(file_path) if file_path else None
    ocr_labeled_metadata: Dict[str, Any] = {}

    if text is None and file_path_obj and file_path_obj.is_dir():
        results: Dict[str, Any] = {}
        text_exts = ["txt", "md"]
        text_files = list(directory.iter_files_by_ext(file_path_obj, text_exts))
        doc_files = list(directory.iter_document_files(file_path_obj))
        all_files = text_files + [f for f in doc_files if f not in text_files]
        all_files = sorted(set(all_files), key=lambda p: str(p))
        total = len(all_files)
        print(f"metadata_extract: 디렉토리 처리 시작 ({total} files) - {file_path_obj}")
        for idx, fpath in enumerate(all_files, start=1):
            rel_path = str(fpath.relative_to(file_path_obj))
            print(f"[{idx}/{total}] 처리 중: {rel_path}")
            results[rel_path] = metadata_extract(
                text=None,
                file_path=str(fpath),
                out_dir=out_dir,
                threshold=threshold,
            )
        return {
            "directory": str(file_path_obj),
            "results": results,
        }

    if text is None and file_path_obj:
        if file_path_obj.suffix.lower() in [".txt", ".md"]:
            text = file_path_obj.read_text(encoding="utf-8")
        else:
            text, ocr_labeled_metadata = ocr_module.process_file_for_metadata(
                file_path_obj, use_temp_dir=True, temp_root="temp"
            )

    raw_text = text or ""
    struct = text_module.read_text(raw_text)
    sentences = struct.get("sentences", [])
    tokens = struct.get("tokens", [])

    regular_decisions = regular_module.regular_extractor(sentences=sentences, tokens=tokens)
    ner_decisions = ner_extractor(sentences=sentences, tokens=tokens, threshold=threshold)
    merged_decisions = merge_regular_ner(regular_decisions, ner_decisions)

    ner_labels, _ = load_labels_from_yaml()
    gliner_metadata = _aggregate_decisions(ner_decisions, ner_labels)
    ocr_metadata = _aggregate_ocr_metadata(ocr_labeled_metadata)
    merged_metadata = _merge_metadata(gliner_metadata, ocr_metadata)

    out_dir_path = directory.ensure_outdir(out_dir)
    out_file = directory.default_outfile(
        file_path=str(file_path_obj) if file_path_obj else None,
        out_dir=out_dir_path,
    )
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(merged_metadata, f, ensure_ascii=False, indent=2)

    print(
        f"metadata_extract: sentences={len(sentences)}, tokens={len(tokens)}, "
        f"regular={len(regular_decisions)}, ner={len(ner_decisions)}"
    )
    print(f"Metadata 저장: {out_file}")

    return {
        "raw_text": raw_text,
        "decisions": merged_decisions,
        "ner_decisions": ner_decisions,
        "regular_decisions": regular_decisions,
        "gliner_metadata": gliner_metadata,
        "ocr_metadata": ocr_metadata,
        "merged_metadata": merged_metadata,
        "out_file": str(out_file),
    }