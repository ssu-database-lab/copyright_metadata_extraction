"""API 모듈: main.py에서 사용하는 함수만 노출"""
import json

from pathlib import Path
from typing import Optional, Dict, Any, List

from module.extractor import ocr as ocr_module
from module.parts import directory
from module.extractor import text as text_module
from module.extractor.ner import ner_extractor
from module.extractor.ner.base import load_labels_from_yaml


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
