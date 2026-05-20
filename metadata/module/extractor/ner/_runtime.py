"""모델 경로, 다운로드, 라벨 로딩, 디버그 세션, predict 공유 유틸."""
from __future__ import annotations

import bisect
from module.parts.paths import project_root  # noqa: E402 (paths.py에 단일 정의)
import hashlib
import json
import logging
import os
import re
import sys
import shutil
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from module.parts.text import clean_token_text
from module.parts.types import Decision

log = logging.getLogger(__name__)

# NER 디버그: 로거 이름 (configure_ner_debug)
NER_DEBUG_LOGGER_NAMES = (
    "module.extractor.ner",
    "module.extractor.ner.base",
    "module.extractor.ner.train",
    "module.extractor.ner.token_cls",
)

_ner_debug_stream_handler: Optional[logging.Handler] = None


def configure_ner_debug(enabled: bool) -> None:
    """``debug=True``일 때 NER 관련 로거를 DEBUG로 올리고 콘솔 핸들러를 붙인다."""
    global _ner_debug_stream_handler

    pkg = logging.getLogger("module.extractor.ner")
    if enabled:
        for name in NER_DEBUG_LOGGER_NAMES:
            logging.getLogger(name).setLevel(logging.DEBUG)
        if _ner_debug_stream_handler is None:
            h = logging.StreamHandler()
            h.setLevel(logging.DEBUG)
            h.setFormatter(
                logging.Formatter("[NER-DEBUG] %(levelname)s %(name)s: %(message)s")
            )
            _ner_debug_stream_handler = h
        if _ner_debug_stream_handler not in pkg.handlers:
            pkg.addHandler(_ner_debug_stream_handler)
        pkg.setLevel(logging.DEBUG)
        pkg.propagate = True
        log.debug("configure_ner_debug(True)")
    else:
        detach_ner_debug_file_logging()
        for name in NER_DEBUG_LOGGER_NAMES:
            logging.getLogger(name).setLevel(logging.INFO)
        pkg.setLevel(logging.INFO)
        if _ner_debug_stream_handler is not None and _ner_debug_stream_handler in pkg.handlers:
            pkg.removeHandler(_ner_debug_stream_handler)


# --- debug 세션: debug_root / model_display_name / train|predict / YYYYMMDDHHmm / threshold_dir

NER_DEBUG_SESSION_DIR: ContextVar[Optional[Path]] = ContextVar(
    "ner_debug_session_dir", default=None
)
_ner_debug_file_handlers: List[logging.Handler] = []
# 콘솔 Tee: (원본 stdout, 원본 stderr, console.log 파일 핸들)
_ner_debug_console_state: Optional[Tuple[Any, Any, Any]] = None


class _ConsoleTee:
    """``sys.stdout`` / ``sys.stderr``를 원래 스트림과 ``console.log``에 동시에 기록."""

    __slots__ = ("_stream", "_extra")

    def __init__(self, stream: Any, extra: Any) -> None:
        self._stream = stream
        self._extra = extra

    def write(self, s: Any) -> int:
        n: Any = None
        try:
            n = self._stream.write(s)
        except Exception:
            pass
        try:
            if isinstance(s, str):
                self._extra.write(s)
            else:
                self._extra.write(str(s))
            self._extra.flush()
        except Exception:
            pass
        if isinstance(n, int):
            return n
        try:
            return len(s) if isinstance(s, str) else len(str(s))
        except Exception:
            return 0

    def flush(self) -> None:
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._extra.flush()
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def resolve_ner_debug_root(debug_path: Optional[str]) -> Path:
    """``debug_path``가 None/빈 문자열이면 ``<project>/debug``, 아니면 절대/프로젝트 상대 경로."""
    if debug_path is None or str(debug_path).strip() == "":
        return (project_root() / "debug").resolve()
    p = Path(debug_path)
    return p.resolve() if p.is_absolute() else (project_root() / p).resolve()


def make_ner_debug_session_dir(
    debug_path: Optional[str],
    model_id: str,
    *,
    debug_kind: Literal["train", "predict"],
    threshold_dir: str = "na",
) -> Path:
    """``…/ model_name / train|predict / YYYYMMDDHHmm / threshold_dir /`` 생성 후 반환."""
    root = resolve_ner_debug_root(debug_path)
    stamp = datetime.now().strftime("%Y%m%d%H%M")
    folder = model_display_name(model_id).replace("/", "_").replace("\\", "_").strip() or "model"
    safe_th = (threshold_dir or "na").replace("/", "_").replace("\\", "_").strip() or "na"
    session = (root / folder / debug_kind / stamp / safe_th).resolve()
    session.mkdir(parents=True, exist_ok=True)
    return session


def ner_debug_print(msg: str) -> None:
    """콘솔 출력 + (세션 디렉터리가 있으면) ``trace.log``에 append."""
    print(msg)
    d = NER_DEBUG_SESSION_DIR.get()
    if d is None:
        return
    trace = d / "trace.log"
    try:
        with open(trace, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")
    except OSError:
        pass


def attach_ner_debug_file_logging(session_dir: Path) -> None:
    """``module.extractor.ner`` 패키지 로그를 ``session_dir/package.log``에 기록하고,
    ``sys.stdout`` / ``sys.stderr`` 전체를 ``session_dir/console.log``에도 기록한다."""
    global _ner_debug_file_handlers, _ner_debug_console_state
    path = session_dir / "package.log"
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    pkg = logging.getLogger("module.extractor.ner")
    pkg.addHandler(fh)
    _ner_debug_file_handlers.append(fh)

    if _ner_debug_console_state is None:
        console_path = session_dir / "console.log"
        try:
            cf = open(console_path, "a", encoding="utf-8", buffering=1)
        except OSError:
            return
        out0 = sys.stdout
        err0 = sys.stderr
        try:
            cf.write(
                f"\n===== console capture start {datetime.now().isoformat(timespec='seconds')} =====\n"
            )
            cf.flush()
        except OSError:
            try:
                cf.close()
            except Exception:
                pass
            return
        sys.stdout = _ConsoleTee(out0, cf)
        sys.stderr = _ConsoleTee(err0, cf)
        _ner_debug_console_state = (out0, err0, cf)


def detach_ner_debug_file_logging() -> None:
    global _ner_debug_file_handlers, _ner_debug_console_state
    pkg = logging.getLogger("module.extractor.ner")
    for h in list(_ner_debug_file_handlers):
        try:
            pkg.removeHandler(h)
        except ValueError:
            pass
        try:
            h.close()
        except Exception:
            pass
    _ner_debug_file_handlers.clear()

    if _ner_debug_console_state is not None:
        out0, err0, cf = _ner_debug_console_state
        _ner_debug_console_state = None
        try:
            sys.stdout = out0
            sys.stderr = err0
        except Exception:
            pass
        try:
            cf.write(
                f"===== console capture end {datetime.now().isoformat(timespec='seconds')} =====\n"
            )
            cf.flush()
            cf.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# 상수 / 레지스트리
# ═══════════════════════════════════════════════════════════════════════

MODEL_TYPE_TOKEN_CLS = "token_cls"

# code-level fallback — configs/labels.yaml::ner 가 없을 때만 사용.
DEFAULT_MODEL = "google-bert/bert-base-multilingual-cased"
DEFAULT_THRESHOLD = 0.25
MODEL_DIR = "models"
DOWNLOAD_DIR = "model_downloaded"
LABELS_YAML = "configs/labels.yaml"


def load_ner_defaults() -> Dict[str, Any]:
    """configs/labels.yaml 의 ner / paths 섹션을 합쳐 ner_predict 기본값으로 사용."""
    import yaml as _yaml
    yaml_path = project_root() / LABELS_YAML
    if not yaml_path.exists():
        return {}
    cfg = _yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        return {}
    out = dict(cfg.get("ner", {}))
    out["_paths"] = cfg.get("paths", {})
    return out


# ═══════════════════════════════════════════════════════════════════════
# 모델 해석 (범용 — 레지스트리 없음)
# ═══════════════════════════════════════════════════════════════════════


def model_display_name(model_id: str) -> str:
    """HuggingFace ID → 사람이 읽기 좋은 짧은 이름."""
    return model_id.rsplit("/", 1)[-1]


def model_dir_name(model_id: str) -> str:
    """HuggingFace ID → models/ 하위 디렉토리명."""
    return model_id.replace("/", "--")


def detect_model_type(model_dir: Path) -> str:
    """Hugging Face Token Classification 계열만 지원."""
    _ = model_dir  # 호환용 인자
    return MODEL_TYPE_TOKEN_CLS


# ═══════════════════════════════════════════════════════════════════════
# 경로 관리 + 모델 다운로드
# ═══════════════════════════════════════════════════════════════════════


def get_model_dir(model_id: str, model_path: Optional[str] = None) -> Path:
    """models/[dir_name] — model_path 지정 시 해당 경로 기준."""
    base = Path(model_path) if model_path is not None else project_root() / MODEL_DIR
    if not base.is_absolute():
        base = project_root() / base
    return base / model_dir_name(model_id)


def get_download_dir(model_id: str) -> Path:
    """model_downloaded/[model_id]"""
    return project_root() / DOWNLOAD_DIR / model_id


def _has_model(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def _materialize_model_dir(src: Path, dst: Path) -> str:
    """`src` 의 내용을 `dst` 로 옮긴다 (hard-link 시도, 실패 시 copy).

    HF Trainer 가 base 파일을 수정하지 않으므로 hard-link 안전.
    학습 산출물(adapter/checkpoint) 은 dst 에 새 sub-dir 로 추가되어 src 에 영향 없음.
    cross-filesystem 이거나 link 불가 시 자동 copy fallback.

    Returns:
        "hardlink" | "copy" | "mixed" — 실제 사용된 방식.
    """
    used = {"hardlink": 0, "copy": 0}
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(str(item), str(target))
            used["hardlink"] += 1
        except OSError:
            shutil.copy2(str(item), str(target))
            used["copy"] += 1
    if used["hardlink"] and not used["copy"]:
        return "hardlink"
    if used["copy"] and not used["hardlink"]:
        return "copy"
    return "mixed"


def ensure_model_ready(model_id: str, model_path: Optional[str] = None) -> Path:
    """모델 디렉터리가 사용 가능하도록 보장.

    1) model_path(기본 ``models``)/[dir_name] 확인 → 있으면 그대로 사용
    2) model_downloaded/[model_id] 확인 → 없으면 HuggingFace 다운로드
    3) model_downloaded → model_path 로 hard-link (실패 시 copy) — 디스크 절약.

    model_path: None이면 프로젝트 루트의 ``models``. 절대/상대 경로 모두 지원.
    """
    model_dir = get_model_dir(model_id, model_path)

    if _has_model(model_dir):
        return model_dir

    dl_dir = get_download_dir(model_id)
    if not _has_model(dl_dir):
        print(f"  HuggingFace 다운로드: {model_id}")
        _download_from_hf(model_id, dl_dir)

    if not _has_model(dl_dir):
        raise FileNotFoundError(
            f"모델을 찾을 수 없습니다: {model_id} "
            f"(models/{model_dir_name(model_id)}, "
            f"model_downloaded/{model_id} 모두 없음)"
        )

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    method = _materialize_model_dir(dl_dir, model_dir)
    print(f"  모델 준비: model_downloaded → {model_dir}  ({method})")
    return model_dir


def _download_from_hf(model_id: str, target_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(target_dir))


# ═══════════════════════════════════════════════════════════════════════
# 학습 데이터 상태
# ═══════════════════════════════════════════════════════════════════════


def get_train_dir() -> Path:
    return project_root() / "configs" / "train"


def get_train_raw_dir() -> Path:
    """BIO jsonl이 모여 있는 디렉터리 (있으면 ``configs/train/raw``)."""
    return project_root() / "configs" / "train" / "raw"


def resolve_bio_train_dir(train_dir: Path) -> Path:
    """상위에만 ``raw/*.jsonl``이 있을 때 학습·시그니처용 경로를 ``raw``로 맞춤."""
    raw = train_dir / "raw"
    if raw.is_dir() and any(raw.glob("*.jsonl")) and not any(train_dir.glob("*.jsonl")):
        return raw
    return train_dir


def get_training_data_signature(train_dir: Union[str, Path]) -> str:
    root = Path(train_dir)
    if not root.exists():
        return ""
    parts: List[str] = []
    for p in sorted(root.glob("*.jsonl")):
        try:
            parts.append(p.name)
            parts.append(str(p.stat().st_mtime_ns))
        except OSError:
            pass
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def get_train_state_path(model_id: str) -> Path:
    state_dir = project_root() / MODEL_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"train_state_{model_dir_name(model_id)}.json"


def read_train_state(model_id: str) -> Optional[Dict[str, Any]]:
    path = get_train_state_path(model_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_train_state(
    signature: str,
    adapter_path: Union[str, Path],
    model_id: str,
) -> None:
    path = get_train_state_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"signature": signature, "adapter_path": str(adapter_path)},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════════════════
# 라벨 로딩
# ═══════════════════════════════════════════════════════════════════════


def get_ner_label_descriptions_dir() -> Path:
    """라벨별 설명 텍스트(선택). ``configs/ner/labels/<라벨>.txt``."""
    return project_root() / "configs" / "ner" / "labels"


def load_labels_from_train_raw(
    raw_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """``configs/train/raw/*.jsonl`` stem → NER 라벨 목록."""
    root = Path(raw_dir) if raw_dir is not None else get_train_raw_dir()
    if not root.is_dir():
        return [], {}
    labels = sorted(
        p.stem for p in root.glob("*.jsonl") if p.is_file() and not p.name.startswith(".")
    )
    return labels, _load_label_descriptions_from_files(labels)


def load_labels_from_predict(
    predict_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """설명 전용 디렉터리(``configs/ner/labels``)에서 ``*.txt``/``*.md`` 파일명으로 라벨 수집."""
    root = Path(predict_dir) if predict_dir is not None else get_ner_label_descriptions_dir()
    if not root.exists():
        return [], {}
    descriptions: Dict[str, str] = {}
    for ext in (".txt", ".md"):
        for path in sorted(root.glob(f"*{ext}")):
            label = path.stem
            if not label.startswith("."):
                try:
                    text = path.read_text(encoding="utf-8").strip()
                    if text:
                        descriptions[label] = text
                except Exception:
                    pass
    return sorted(descriptions.keys()), descriptions


def _load_label_descriptions_from_files(labels: List[str]) -> Dict[str, str]:
    desc_root = get_ner_label_descriptions_dir()
    descriptions: Dict[str, str] = {}
    for label in labels:
        for ext in (".txt", ".md"):
            path = desc_root / f"{label}{ext}"
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8").strip()
                    if text:
                        descriptions[label] = text
                except Exception:
                    pass
                break
    return descriptions


def load_labels_from_yaml(
    yaml_path: Optional[Union[str, Path]] = None,
    key: str = "ner",
) -> Tuple[List[str], Dict[str, str]]:
    import yaml

    if yaml_path is None:
        yaml_path = project_root() / LABELS_YAML
    yaml_path = Path(yaml_path)

    if yaml_path.exists():
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        ner_cfg = cfg.get(key, {}) if isinstance(cfg, dict) else {}
        labels_obj = ner_cfg.get("labels")

        if labels_obj == "auto":
            names, desc = load_labels_from_train_raw()
            if names:
                return names, desc
            return load_labels_from_predict()
        if isinstance(labels_obj, dict):
            return list(labels_obj.keys()), {str(k): str(v) for k, v in labels_obj.items()}
        if isinstance(labels_obj, list):
            labels = [str(x) for x in labels_obj]
            return labels, _load_label_descriptions_from_files(labels)

    names, desc = load_labels_from_train_raw()
    if names:
        return names, desc
    labels, desc = load_labels_from_predict()
    if labels:
        return labels, desc
    from module.parts.labels import ALL_LABELS
    labels = list(ALL_LABELS)
    return labels, _load_label_descriptions_from_files(labels)


def load_ner_thresholds_from_yaml(
    yaml_path: Optional[Union[str, Path]] = None,
) -> Tuple[float, Dict[str, float]]:
    import yaml

    if yaml_path is None:
        yaml_path = project_root() / LABELS_YAML
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        return DEFAULT_THRESHOLD, {}

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    ner_cfg = cfg.get("ner", {}) if isinstance(cfg, dict) else {}
    default = float(ner_cfg.get("threshold", DEFAULT_THRESHOLD))
    per_label = ner_cfg.get("threshold_per_label")
    per_label = (
        {str(k): float(v) for k, v in per_label.items()}
        if isinstance(per_label, dict)
        else {}
    )
    return default, per_label


# ═══════════════════════════════════════════════════════════════════════
# BIO ↔ span 변환 (train.py에서도 사용)
# ═══════════════════════════════════════════════════════════════════════


def bio_to_ner_spans(labels: List[str]) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    i = 0
    while i < len(labels):
        tag = labels[i]
        if tag.startswith("B-"):
            label_name = tag[2:]
            start = i
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{label_name}":
                j += 1
            spans.append((start, j - 1, label_name))
            i = j
        else:
            i += 1
    return spans


# ═══════════════════════════════════════════════════════════════════════
# 전처리 + Decision 빌더
# ═══════════════════════════════════════════════════════════════════════


def _prepare_sentence_input(
    sentences: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
) -> Tuple[List[List[str]], List[Tuple[int, List[Dict[str, Any]]]]]:
    if not sentences or not tokens:
        return [], []
    token_groups: Dict[int, List[Dict[str, Any]]] = {}
    for t in tokens:
        sid = t.get("sent_id")
        if sid is not None:
            token_groups.setdefault(int(sid), []).append(t)
    sentence_texts: List[List[str]] = []
    sentence_info: List[Tuple[int, List[Dict[str, Any]]]] = []
    for s in sentences:
        sid = s.get("sent_id")
        if sid is None:
            continue
        sent_tokens = token_groups.get(int(sid), [])
        token_texts = [clean_token_text(t.get("text", "")) for t in sent_tokens]
        if token_texts:
            sentence_texts.append(token_texts)
            sentence_info.append((sid, sent_tokens))
    return sentence_texts, sentence_info


def _build_decisions_from_bio(
    sentence_tokens: List[Dict[str, Any]],
    predicted_labels: List[str],
    sent_id: int,
) -> List[Decision]:
    decisions: List[Decision] = []
    cur_val: Optional[str] = None
    cur_label: Optional[str] = None
    cur_tok_id: Optional[int] = None

    def _flush():
        nonlocal cur_val, cur_label, cur_tok_id
        if cur_val and cur_label:
            decisions.append(Decision(
                label=cur_label, value=cur_val,
                sent_id=sent_id, tok_id=cur_tok_id, source="ner",
            ))
        cur_val = cur_label = cur_tok_id = None

    for tok, tag in zip(sentence_tokens, predicted_labels):
        tok_text = tok.get("text", "")
        tok_id = tok.get("tok_id")

        if tag == "O":
            _flush()
            continue

        if tag.startswith("B-"):
            _flush()
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

    _flush()
    return decisions


# ═══════════════════════════════════════════════════════════════════════
# 모델 캐시 + 통합 예측
# ═══════════════════════════════════════════════════════════════════════

_token_cls_cache: Dict[str, Any] = {}


def invalidate_model_cache(model_id: str) -> None:
    """학습 후 캐시 무효화 (train.py에서 호출)."""
    _token_cls_cache.pop(model_id, None)


def clear_all_ner_model_caches() -> None:
    """메모리에 올라간 모든 NER 추론기 캐시 제거."""
    _token_cls_cache.clear()


def iter_installed_ner_model_ids() -> List[str]:
    """``models/`` 아래에 ``config.json``이 있는 로컬 모델 디렉터리 → HF 스타일 ID 목록."""
    root = project_root() / MODEL_DIR
    if not root.is_dir():
        return []
    out: List[str] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and (p / "config.json").exists():
            out.append(p.name.replace("--", "/"))
    return out


def disk_adapter_report(model_id: str) -> Dict[str, Any]:
    """디스크 기준: 어댑터 파일이 있는지(추론 시 로드 가능 여부는 별도)."""
    model_dir = ensure_model_ready(model_id)
    mt = detect_model_type(model_dir)
    adapter = model_dir / "adapter"
    report: Dict[str, Any] = {
        "model_id": model_id,
        "model_dir": str(model_dir),
        "model_type": mt,
        "adapter_config_path": None,
        "ready_for_adapter_load": False,
    }
    cfg = adapter / "adapter_config.json"
    tm = adapter / "train_method.json"
    lm = adapter / "label_map.json"
    if lm.exists() and (cfg.exists() or tm.exists()):
        marker = cfg if cfg.exists() else tm
        report["adapter_config_path"] = str(marker.resolve())
        report["ready_for_adapter_load"] = True
        if tm.exists():
            try:
                report["fine_tuning_method"] = json.loads(
                    tm.read_text(encoding="utf-8")
                ).get("fine_tuning_method", "lora")
            except Exception:
                pass
    return report


def runtime_adapter_report(model_id: str) -> Dict[str, Any]:
    """현재 프로세스 캐시에 올라간 추론기 기준 실제 적용 여부."""
    model_dir = get_model_dir(model_id)
    mt = detect_model_type(model_dir)
    cache_key = str(model_dir)
    tc = _token_cls_cache.get(cache_key) or _token_cls_cache.get(model_id)
    if tc is None:
        return {
            "model_id": model_id,
            "model_type": mt,
            "inference_cache_hit": False,
            "adapter_applied_path": None,
        }
    path = getattr(tc, "adapter_load_path", None)
    return {
        "model_id": model_id,
        "model_type": mt,
        "inference_cache_hit": True,
        "adapter_applied_path": path,
    }


def reset_ner_models(model_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """로컬 ``adapter/`` 및 학습 상태 파일 삭제 후 추론 캐시 전부 비움.

    ``model_ids``가 None이면 ``models/``에 설치된 모든 NER 모델을 대상으로 함.
    베이스 가중치(``models/.../`` 본체)는 삭제하지 않음.
    """
    actions: List[str] = []
    targets = list(model_ids) if model_ids is not None else iter_installed_ner_model_ids()
    for mid in targets:
        model_dir = get_model_dir(mid)
        if not model_dir.is_dir():
            actions.append(f"{mid}: models 폴더 없음 — 스킵")
            continue
        ad = model_dir / "adapter"
        if ad.exists():
            shutil.rmtree(ad, ignore_errors=True)
            actions.append(f"{mid}: adapter/ 삭제")
        tsp = get_train_state_path(mid)
        if tsp.exists():
            tsp.unlink(missing_ok=True)
            actions.append(f"{mid}: train_state 삭제")
        invalidate_model_cache(mid)
    clear_all_ner_model_caches()
    return {"targets": targets, "actions": actions}


def _predict_decisions(
    *,
    sentences: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
    threshold: Optional[float] = None,
    model: str = DEFAULT_MODEL,
    model_path: Optional[str] = None,
    debug: bool = False,
) -> List[Decision]:
    """NER 예측 (통합). model = HuggingFace ID 또는 로컬 이름.

    NER 은 role-free 라벨만 반환. 역할 부여 (ch_co / ch_ja / ch_nr) 는 LLM stage 전담.
    """
    if debug:
        configure_ner_debug(True)
        log.debug("ner_predict 시작 model=%s threshold=%s model_path=%s", model, threshold, model_path)

    model_dir = ensure_model_ready(model, model_path=model_path)
    model_type = detect_model_type(model_dir)

    sentence_texts, sentence_info = _prepare_sentence_input(sentences, tokens)
    if not sentence_texts:
        if debug:
            ner_debug_print("[NER debug] 예측 스킵: 문장/토큰 입력 없음")
        return []

    effective_threshold: float = (
        threshold if threshold is not None else DEFAULT_THRESHOLD
    )

    if debug:
        n_words = sum(len(s) for s in sentence_texts)
        ner_debug_print(
            f"[NER debug] model_dir={model_dir} type={model_type} "
            f"sentences={len(sentence_texts)} word_tokens≈{n_words} "
            f"yaml_tokens={len(tokens)} threshold={effective_threshold}"
        )

    from module.extractor.ner.token_cls import TokenClassNER

    cache_key = str(model_dir)  # model_path가 다르면 다른 캐시 항목
    if cache_key not in _token_cls_cache:
        _token_cls_cache[cache_key] = TokenClassNER(model_dir)
    predicted = _token_cls_cache[cache_key].predict(
        sentence_texts, threshold=effective_threshold,
    )

    decisions: List[Decision] = []
    for (sid, sent_tokens), pred_labels in zip(sentence_info, predicted):
        decisions.extend(_build_decisions_from_bio(sent_tokens, pred_labels, int(sid)))

    if debug:
        ner_debug_print(f"[NER debug] BIO 시퀀스 수={len(predicted)} decisions={len(decisions)}")
        for i, d in enumerate(decisions[:8]):
            ner_debug_print(f"    [{i}] {d.label!r} = {str(d.value)[:80]!r}")
        if len(decisions) > 8:
            ner_debug_print(f"    ... 외 {len(decisions) - 8}건")

    if debug:
        log.debug("ner_predict 완료 decisions=%d", len(decisions))

    return decisions


def ner_predict_at_thresholds(
    *,
    sentences: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
    thresholds: List[float],
    model: str = DEFAULT_MODEL,
    model_path: Optional[str] = None,
) -> Dict[float, List[Any]]:
    """NER 예측 — inference 1회로 여러 threshold 결과를 한번에 반환.

    Returns:
        {threshold → List[Decision]}
    """
    from module.extractor.ner.token_cls import TokenClassNER

    model_dir = ensure_model_ready(model, model_path=model_path)
    sentence_texts, sentence_info = _prepare_sentence_input(sentences, tokens)
    if not sentence_texts:
        return {thr: [] for thr in thresholds}

    cache_key = str(model_dir)
    if cache_key not in _token_cls_cache:
        _token_cls_cache[cache_key] = TokenClassNER(model_dir)
    ner_obj = _token_cls_cache[cache_key]

    # inference 1회: List[{thr: bio_list}]
    per_text_thr = ner_obj.predict_at_thresholds(sentence_texts, thresholds)

    result: Dict[float, list] = {}
    for thr in thresholds:
        decisions: list = []
        for (sid, sent_tokens), thr_bio_map in zip(sentence_info, per_text_thr):
            decisions.extend(_build_decisions_from_bio(sent_tokens, thr_bio_map[thr], int(sid)))

        result[thr] = decisions

    return result


