"""Zero-shot NER (GLiNER2 기반)"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import re
import shutil
import unicodedata
import sys

from module.parts.types import Decision

try:
    from gliner2 import GLiNER2
except ImportError as e:
    raise RuntimeError(
        "GLiNER2가 필요합니다. 현재 실행중인 Python: "
        f"{sys.executable} (prefix: {sys.prefix})"
    ) from e


DEFAULT_MODEL_ID = "fastino/gliner2-base-v1"
DEFAULT_THRESHOLD = 0.55
MODEL_DIR = "models"
DOWNLOADED_MODEL_DIR = "model_downloaded"


@dataclass(frozen=True)
class EntitySpan:
    label: str
    text: str
    start: int
    end: int
    confidence: float


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _models_root() -> Path:
    return _project_root() / MODEL_DIR


def _model_downloaded_roots() -> List[Path]:
    root = _project_root()
    return [root / DOWNLOADED_MODEL_DIR, root / "module" / DOWNLOADED_MODEL_DIR]


def _migrate_legacy_downloaded() -> None:
    root = _project_root()
    legacy = root / "module" / DOWNLOADED_MODEL_DIR
    target = root / DOWNLOADED_MODEL_DIR
    if legacy.exists() and not target.exists():
        try:
            shutil.move(str(legacy), str(target))
        except Exception:
            pass


def _ensure_model_in_models(model_id: str) -> Optional[Path]:
    target = _models_root() / model_id
    if target.exists() and (target / "config.json").exists():
        return target
    for src_root in _model_downloaded_roots():
        candidate = src_root / model_id
        if candidate.exists():
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(candidate, target, dirs_exist_ok=True)
                return target
            except Exception:
                return None
    return None


def _find_latest_model_dir(base_dir: Path) -> Optional[Path]:
    if not base_dir.exists():
        return None
    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _find_model_dir_with_config(base_dir: Path) -> Optional[Path]:
    if not base_dir.exists():
        return None
    for path in base_dir.rglob("config.json"):
        return path.parent
    return None


def _resolve_model_source(model_id: str) -> str:
    _migrate_legacy_downloaded()
    root = _project_root()
    legacy_root = root / "module"
    models_root = root / MODEL_DIR
    legacy_models_root = legacy_root / MODEL_DIR

    latest = (
        _find_model_dir_with_config(models_root / "ner")
        or _find_model_dir_with_config(models_root)
        or _find_model_dir_with_config(legacy_models_root / "ner")
        or _find_model_dir_with_config(legacy_models_root)
        or _find_latest_model_dir(models_root / "ner")
        or _find_latest_model_dir(models_root)
        or _find_latest_model_dir(legacy_models_root / "ner")
        or _find_latest_model_dir(legacy_models_root)
    )
    if latest:
        return str(latest)

    ensured = _ensure_model_in_models(model_id)
    if ensured is not None and ensured.exists():
        return str(ensured)

    for dl_root in _model_downloaded_roots():
        local_dir = dl_root / model_id
        if local_dir.exists():
            return str(local_dir)

    return model_id


def load_labels_from_yaml(
    yaml_path: Optional[Union[str, Path]] = None,
    key: str = "ner",
) -> Tuple[List[str], Dict[str, str]]:
    """
    labels.yaml에서 라벨을 읽습니다.
    - 지원 형태:
      1) ner: {labels: ["email", "phone_number", ...]}
      2) ner: {labels: {email: "이메일 주소", phone_number: "전화번호", ...}}  # 설명 포함
    반환:
      (labels_list, label_descriptions_dict)
    """
    import yaml  # 지연 import

    if yaml_path is None:
        yaml_path = _project_root() / "configs" / "labels.yaml"
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        labels = ["address", "company_name", "person_name", "phone_number", "email", "url", "date"]
        return labels, {}

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    ner_cfg = cfg.get(key, {}) if isinstance(cfg, dict) else {}
    labels_obj = ner_cfg.get("labels")

    if isinstance(labels_obj, dict):
        labels = list(labels_obj.keys())
        return labels, {str(k): str(v) for k, v in labels_obj.items()}
    if isinstance(labels_obj, list):
        labels = [str(x) for x in labels_obj]
        return labels, _load_label_descriptions_from_files(labels)
    labels = ["address", "company_name", "person_name", "phone_number", "email", "url", "date"]
    return labels, _load_label_descriptions_from_files(labels)


def _load_label_descriptions_from_files(labels: List[str]) -> Dict[str, str]:
    """
    configs/gliner/ 또는 configs/training/ner_labels/ 내부의 라벨 설명 파일을 읽습니다.
    - 지원 확장자: .txt, .md
    - 파일명: {label}.txt 또는 {label}.md
    """
    root = _project_root()
    candidates = [
        root / "configs" / "gliner",
        root / "configs" / "training" / "ner_labels",
    ]
    descriptions: Dict[str, str] = {}
    for label in labels:
        for base_dir in candidates:
            for ext in (".txt", ".md"):
                path = base_dir / f"{label}{ext}"
                if path.exists():
                    text = path.read_text(encoding="utf-8").strip()
                    if text:
                        descriptions[label] = text
                    break
            if label in descriptions:
                break
    return descriptions


def _normalize_ocr_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\u00A0", " ")).strip()


def _join_tokens_with_spans(tokens: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    spans: List[Tuple[int, int]] = []
    parts: List[str] = []
    cur = 0
    for i, tok in enumerate(tokens):
        if i > 0:
            parts.append(" ")
            cur += 1
        start = cur
        parts.append(tok)
        cur += len(tok)
        end = cur
        spans.append((start, end))
    return "".join(parts), spans


def _span_to_token_indices(ent_start: int, ent_end: int, token_spans: List[Tuple[int, int]]) -> List[int]:
    idxs: List[int] = []
    for i, (ts, te) in enumerate(token_spans):
        if te <= ent_start:
            continue
        if ts >= ent_end:
            break
        # overlap
        if ts < ent_end and te > ent_start:
            idxs.append(i)
    return idxs


def _build_decisions_from_bio(
    sentence_tokens: List[Dict[str, Any]],
    predicted_labels: List[str],
    sent_id: int,
) -> List[Decision]:
    decisions: List[Decision] = []
    cur_val: Optional[str] = None
    cur_label: Optional[str] = None
    cur_tok_id: Optional[int] = None

    for tok, tag in zip(sentence_tokens, predicted_labels):
        tok_text = tok.get("text", "")
        tok_id = tok.get("tok_id")

        if tag == "O":
            if cur_val and cur_label:
                decisions.append(
                    Decision(
                        label=cur_label,
                        value=cur_val,
                        sent_id=sent_id,
                        tok_id=cur_tok_id,
                        source="ner",
                    )
                )
                cur_val = None
                cur_label = None
            continue

        if tag.startswith("B-"):
            if cur_val and cur_label:
                decisions.append(
                    Decision(
                        label=cur_label,
                        value=cur_val,
                        sent_id=sent_id,
                        tok_id=cur_tok_id,
                        source="ner",
                    )
                )
            cur_label = tag[2:]
            cur_val = tok_text
            cur_tok_id = tok_id
        elif tag.startswith("I-"):
            if cur_val is not None and cur_label is not None:
                cur_val += " " + tok_text
            else:
                cur_label = tag[2:]
                cur_val = tok_text
                cur_tok_id = tok_id

    if cur_val and cur_label:
        decisions.append(
            Decision(
                label=cur_label,
                value=cur_val,
                sent_id=sent_id,
                tok_id=cur_tok_id,
                source="ner",
            )
        )

    return decisions


class ZeroShotNER:
    """
    - 입력: List[List[str]] (토큰 단위)
    - 출력: List[List[str]] (BIO)
    - 내부: GLiNER2 zero-shot extract_entities 사용 :contentReference[oaicite:4]{index=4}
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        labels: Optional[Sequence[str]] = None,
        label_descriptions: Optional[Dict[str, str]] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.model_id = model_id
        self.threshold = float(threshold)

        if labels is None:
            labels, desc = load_labels_from_yaml()
            self.labels = labels
            self.label_descriptions = label_descriptions or desc
        else:
            self.labels = list(labels)
            self.label_descriptions = label_descriptions or {}

        model_source = _resolve_model_source(self.model_id)
        self.extractor = GLiNER2.from_pretrained(model_source)

    def predict(
        self,
        texts: List[List[str]],
        threshold: Optional[float] = None,
        use_descriptions: bool = True,
    ) -> List[List[str]]:
        th = self.threshold if threshold is None else float(threshold)

        schema: Union[List[str], Dict[str, str]] = (
            {k: self.label_descriptions.get(k, k) for k in self.labels}
            if use_descriptions and self.label_descriptions
            else self.labels
        )

        outputs: List[List[str]] = []

        for tokens in texts:
            raw_text, token_spans = _join_tokens_with_spans(tokens)
            text = _normalize_ocr_text(raw_text)

            res = self.extractor.extract_entities(
                text,
                schema,
                include_confidence=True,
                include_spans=True,
            )

            entities: List[EntitySpan] = []
            ent_map = (res or {}).get("entities", {}) or {}
            for lab, items in ent_map.items():
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    conf = float(it.get("confidence", 0.0))
                    if conf < th or "start" not in it or "end" not in it:
                        continue
                    entities.append(
                        EntitySpan(
                            label=str(lab),
                            text=str(it.get("text", "")),
                            start=int(it["start"]),
                            end=int(it["end"]),
                            confidence=conf,
                        )
                    )

            entities.sort(key=lambda x: x.confidence, reverse=True)

            bio = ["O"] * len(tokens)
            occupied = [False] * len(tokens)

            for ent in entities:
                tok_ids = _span_to_token_indices(ent.start, ent.end, token_spans)
                if not tok_ids:
                    continue
                tok_ids = [i for i in tok_ids if 0 <= i < len(tokens) and not occupied[i]]
                if not tok_ids:
                    continue

                bio[tok_ids[0]] = f"B-{ent.label}"
                for j in tok_ids[1:]:
                    bio[j] = f"I-{ent.label}"
                for j in tok_ids:
                    occupied[j] = True

            outputs.append(bio)

        return outputs


@lru_cache(maxsize=1)
def _get_zeroshot_model() -> ZeroShotNER:
    return ZeroShotNER()


def ner_extractor(
    *,
    sentences: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
    threshold: Optional[float] = None,
) -> List[Decision]:
    if not sentences or not tokens:
        return []

    token_groups: Dict[int, List[Dict[str, Any]]] = {}
    for token in tokens:
        if (sid := token.get("sent_id")) is not None:
            token_groups.setdefault(int(sid), []).append(token)

    sentence_texts: List[List[str]] = []
    sentence_info: List[Tuple[int, List[Dict[str, Any]]]] = []
    for sentence in sentences:
        if (sid := sentence.get("sent_id")) is None:
            continue
        sent_tokens = token_groups.get(int(sid), [])
        token_texts = [_clean_token_text(t.get("text", "")) for t in sent_tokens]
        if token_texts:
            sentence_texts.append(token_texts)
            sentence_info.append((sid, sent_tokens))

    if not sentence_texts:
        return []

    predicted_labels_list = _get_zeroshot_model().predict(sentence_texts, threshold=threshold)

    decisions: List[Decision] = []
    for (sid, sent_tokens), predicted_labels in zip(sentence_info, predicted_labels_list):
        decisions.extend(_build_decisions_from_bio(sent_tokens, predicted_labels, int(sid)))

    return decisions


def _clean_token_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.translate(str.maketrans({"－": "-", "—": "-", "–": "-", "‐": "-"}))
    return re.sub(r"\s+", " ", t).strip()
