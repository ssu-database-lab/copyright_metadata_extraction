"""Zero-shot NER (GLiNER2 기반). auto 시 학습 데이터 변경 시에만 학습 후 predict."""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from module.parts.text import (
    clean_token_text,
    join_tokens_with_spans,
    normalize_ocr_text,
    span_to_token_indices,
)
from module.parts.types import Decision

try:
    import torch
except ImportError:
    torch = None

try:
    from gliner2 import GLiNER2
except ImportError as e:
    raise RuntimeError(
        "GLiNER2가 필요합니다. 현재 실행중인 Python: "
        f"{sys.executable} (prefix: {sys.prefix})"
    ) from e

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 환경 변수 기반 설정
# -----------------------------------------------------------------------------

DEFAULT_MODEL_ID = "fastino/gliner2-large-v1"
DEFAULT_THRESHOLD = 0.55
MODEL_DIR = "models"
DOWNLOADED_MODEL_DIR = "model_downloaded"


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


_SCHEMA_CHUNK_SIZE_DEFAULT = 10
_SCHEMA_CHUNK_OVERLAP = 0


def _max_chunk_tokens() -> int:
    """DeBERTa O(n^2) attention 특성 상 256 권장 (라벨 50개 기준). env GLINER_MAX_CHUNK_TOKENS."""
    return _env_int("GLINER_MAX_CHUNK_TOKENS", 256, 64)


def _ner_batch_size() -> int:
    """gliner2 내부 encoder batch size. env GLINER_NER_BATCH_SIZE."""
    return _env_int("GLINER_NER_BATCH_SIZE", 1)


def _micro_batch_size() -> int:
    """우리쪽 마이크로 배치 크기. env GLINER_MICRO_BATCH_SIZE."""
    return _env_int("GLINER_MICRO_BATCH_SIZE", 2)


def _schema_chunk_size() -> int:
    """한 번 추론에 넘길 최대 라벨 수. env GLINER_SCHEMA_CHUNK_SIZE."""
    return _env_int("GLINER_SCHEMA_CHUNK_SIZE", _SCHEMA_CHUNK_SIZE_DEFAULT, 1)


def _cuda_memory_fraction() -> Optional[float]:
    val = os.environ.get("GLINER_CUDA_MEMORY_FRACTION", "").strip()
    if not val:
        return None
    try:
        f = float(val)
        return f if 0 < f <= 1 else None
    except ValueError:
        return None


def _auto_train_cooldown_sec() -> int:
    return _env_int("GLINER_AUTO_TRAIN_COOLDOWN_SEC", 300, 0)


def _auto_train_lock_wait_sec() -> float:
    return _env_float("GLINER_AUTO_TRAIN_LOCK_WAIT_SEC", 2.0)


def _auto_train_enabled() -> bool:
    return os.environ.get("GLINER_AUTO_TRAIN", "1") not in ("0", "false", "False", "no", "NO")


# -----------------------------------------------------------------------------
# 데이터 클래스
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class EntitySpan:
    label: str
    text: str
    start: int
    end: int
    confidence: float


# -----------------------------------------------------------------------------
# ZeroShotNER
# -----------------------------------------------------------------------------

class ZeroShotNER:
    """GLiNER2 기반 zero-shot NER. auto=True 시 학습 데이터 변경 감지 → 재학습."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        labels: Optional[Sequence[str]] = None,
        label_descriptions: Optional[Dict[str, str]] = None,
        threshold: float = DEFAULT_THRESHOLD,
        adapter_path: Optional[str] = None,
        auto: bool = False,
    ):
        self.model_id = model_id
        self.threshold = float(threshold)
        self.auto = bool(auto)
        self._loaded_adapter_path: Optional[str] = None
        self._last_auto_train_attempt_ts: float = 0.0
        self._cuda_frac_applied = False

        if labels is None:
            labels, desc = load_labels_from_yaml()
            self.labels = labels
            self.label_descriptions = label_descriptions or desc
            _default_th, _per_label = load_ner_thresholds_from_yaml()
            self.threshold = float(threshold) if threshold != DEFAULT_THRESHOLD else _default_th
            self.threshold_per_label = _per_label
        else:
            self.labels = list(labels)
            self.label_descriptions = label_descriptions or {}
            self.threshold = float(threshold)
            self.threshold_per_label = {}

        model_source = _resolve_model_source(self.model_id)
        self.extractor = GLiNER2.from_pretrained(model_source)

        self.device = "cpu"
        self._preferred_device = "cpu"
        if torch is not None and torch.cuda.is_available() and hasattr(self.extractor, "to"):
            self._preferred_device = "cuda"

        if not self.auto:
            adapter = adapter_path or os.environ.get("GLINER_ONLINE_ADAPTER")
            if _try_load_adapter(self.extractor, adapter):
                self._loaded_adapter_path = str(Path(adapter).resolve())
            self._ensure_on_preferred_device()
        else:
            self._ensure_on_cpu()

        if hasattr(self.extractor, "eval"):
            self.extractor.eval()

    # --- device ---

    def _ensure_on_cpu(self) -> None:
        if self.device == "cpu":
            return
        try:
            if hasattr(self.extractor, "to"):
                self.extractor = self.extractor.to("cpu")
        finally:
            self.device = "cpu"
            _empty_cuda_cache()

    def _ensure_on_cuda(self) -> None:
        if self.device == "cuda":
            return
        if torch is None or not torch.cuda.is_available() or not hasattr(self.extractor, "to"):
            self.device = "cpu"
            return
        if not self._cuda_frac_applied:
            frac = _cuda_memory_fraction()
            if frac is None:
                frac = 0.80
            try:
                torch.cuda.set_per_process_memory_fraction(frac, 0)
                log.info("[GPU] 추론 메모리 제한 %.0f%% 적용", frac * 100)
            except Exception:
                pass
            import os as _os
            _os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            self._cuda_frac_applied = True

        _force_gc()
        self.extractor = self.extractor.to("cuda")
        self.device = "cuda"
        _log_vram("모델 GPU 로드 후")

    def _ensure_on_preferred_device(self) -> None:
        if self._preferred_device == "cuda":
            self._ensure_on_cuda()
        else:
            self._ensure_on_cpu()

    # --- auto adapter ---

    def _maybe_load_adapter(self, adapter_path: Optional[str]) -> None:
        if not adapter_path or self._loaded_adapter_path == adapter_path:
            return
        if _try_load_adapter(self.extractor, adapter_path):
            self._loaded_adapter_path = adapter_path

    def _load_best_adapter_and_move(self, override: Optional[str] = None) -> None:
        path = override
        if path is None:
            best = get_best_adapter_path()
            path = str(best) if best else None
        self._maybe_load_adapter(path)
        self._ensure_on_preferred_device()

    def _refresh_adapter_auto(self) -> None:
        train_dir = get_gliner_train_dir()
        current_sig = get_training_data_signature(train_dir)
        state = read_train_state() or {}

        if not current_sig or state.get("signature") == current_sig:
            self._load_best_adapter_and_move(state.get("adapter_path"))
            return

        if not _auto_train_enabled():
            self._load_best_adapter_and_move()
            return

        now = time.time()
        if (now - self._last_auto_train_attempt_ts) < float(_auto_train_cooldown_sec()):
            self._load_best_adapter_and_move()
            return

        self._last_auto_train_attempt_ts = now
        self._ensure_on_cpu()
        _ensure_trained_if_auto()

        state2 = read_train_state() or {}
        self._load_best_adapter_and_move(state2.get("adapter_path"))

    # --- predict ---

    def predict(
        self,
        texts: List[List[str]],
        threshold: Optional[float] = None,
        use_descriptions: bool = True,
    ) -> List[List[str]]:
        if self.auto:
            self._refresh_adapter_auto()
        else:
            self._ensure_on_preferred_device()

        th = self.threshold if threshold is None else float(threshold)
        full_schema: Union[List[str], Dict[str, str]] = (
            {k: self.label_descriptions.get(k, k) for k in self.labels}
            if use_descriptions and self.label_descriptions
            else self.labels
        )
        max_chunk = _max_chunk_tokens()
        batch_size = _ner_batch_size()
        micro_bs = _micro_batch_size()
        th_per_label = getattr(self, "threshold_per_label", None)

        jobs: List[Tuple[List[str], int]] = []
        for out_idx, tokens in enumerate(texts):
            if len(tokens) <= max_chunk:
                jobs.append((tokens, out_idx))
            else:
                for start in range(0, len(tokens), max_chunk):
                    jobs.append((tokens[start : start + max_chunk], out_idx))

        if not jobs:
            return [[] for _ in texts]

        batch_texts: List[str] = []
        job_spans: List[List[Tuple[int, int]]] = []
        for tokens, _ in jobs:
            raw_text, spans = join_tokens_with_spans(tokens)
            batch_texts.append(normalize_ocr_text(raw_text))
            job_spans.append(spans)

        schema_chunks = _split_schema(full_schema, _schema_chunk_size())
        n_chunks = len(schema_chunks)
        log.info(
            "추론 시작: texts=%d, labels=%d, schema_chunks=%d(×%d)",
            len(batch_texts),
            len(self.labels),
            n_chunks,
            _schema_chunk_size(),
        )

        merged_results: List[Dict[str, Any]] = [{} for _ in batch_texts]

        with _inference_context():
            for sc_idx, sub_schema in enumerate(schema_chunks):
                log.debug("schema chunk %d/%d (%d labels)", sc_idx + 1, n_chunks, _schema_len(sub_schema))

                for i in range(0, len(batch_texts), micro_bs):
                    chunk = batch_texts[i : i + micro_bs]
                    chunk_results = self._infer_chunk(chunk, sub_schema, batch_size, th)

                    for j, res in enumerate(chunk_results):
                        _merge_entity_results(merged_results[i + j], res)

                    _force_gc()

        outputs: List[List[str]] = [[] for _ in range(len(texts))]
        for idx, (tokens, out_idx) in enumerate(jobs):
            res = merged_results[idx] if idx < len(merged_results) else {}
            entities = _parse_entities_from_res(res, th, th_per_label)
            entities.sort(key=lambda x: x.confidence, reverse=True)
            bio = _entities_to_bio(entities, job_spans[idx], len(tokens))
            outputs[out_idx].extend(bio)

        return outputs

    def _infer_chunk(
        self,
        texts: List[str],
        schema: Union[List[str], Dict[str, str]],
        batch_size: int,
        threshold: float,
    ) -> List[Any]:
        """OOM / 타임아웃 발생 시 batch_size → 1 → 개별 처리로 자동 폴백."""
        for attempt_bs in (batch_size, 1):
            try:
                result = self.extractor.batch_extract_entities(
                    texts, schema,
                    batch_size=attempt_bs,
                    threshold=threshold,
                    include_confidence=True,
                    include_spans=True,
                )
                _cuda_sync_safe()
                return result if isinstance(result, list) else [result]
            except (RuntimeError, Exception) as e:
                if not _is_oom_or_cuda_error(e):
                    raise
                log.warning("CUDA 오류 (batch_size=%d, texts=%d) → 폴백: %s", attempt_bs, len(texts), e)
                _force_gc()
                if attempt_bs == 1:
                    break

        results: List[Any] = []
        for text in texts:
            try:
                r = self.extractor.batch_extract_entities(
                    [text], schema,
                    batch_size=1,
                    threshold=threshold,
                    include_confidence=True,
                    include_spans=True,
                )
                _cuda_sync_safe()
                results.extend(r if isinstance(r, list) else [r])
            except (RuntimeError, Exception) as e:
                if not _is_oom_or_cuda_error(e):
                    raise
                log.warning("CUDA 오류 (single text, len=%d) → 빈 결과: %s", len(text), e)
                _force_gc()
                results.append({})
        return results


# -----------------------------------------------------------------------------
# 스키마(라벨) 분할 헬퍼
# -----------------------------------------------------------------------------

def _schema_len(schema: Union[List[str], Dict[str, str]]) -> int:
    return len(schema)


def _split_schema(
    schema: Union[List[str], Dict[str, str]],
    chunk_size: int,
) -> List[Union[List[str], Dict[str, str]]]:
    """라벨 목록/사전을 chunk_size 단위로 분할. TDR 및 VRAM 부담 감소."""
    if _schema_len(schema) <= chunk_size:
        return [schema]

    if isinstance(schema, dict):
        keys = list(schema.keys())
        chunks: List[Dict[str, str]] = []
        for i in range(0, len(keys), chunk_size):
            sub = {k: schema[k] for k in keys[i : i + chunk_size]}
            chunks.append(sub)
        return chunks
    else:
        chunks_list: List[List[str]] = []
        for i in range(0, len(schema), chunk_size):
            chunks_list.append(schema[i : i + chunk_size])
        return chunks_list


def _merge_entity_results(target: Dict[str, Any], source: Any) -> None:
    """source의 entities를 target에 병합 (동일 라벨은 리스트 확장)."""
    if not source or not isinstance(source, dict):
        return
    src_ents = source.get("entities")
    if not src_ents or not isinstance(src_ents, dict):
        return
    if "entities" not in target:
        target["entities"] = {}
    for label, items in src_ents.items():
        if not isinstance(items, list):
            continue
        if label not in target["entities"]:
            target["entities"][label] = []
        target["entities"][label].extend(items)


# -----------------------------------------------------------------------------
# CUDA / 추론 헬퍼
# -----------------------------------------------------------------------------

def _log_vram(label: str = "") -> None:
    if torch is None or not torch.cuda.is_available():
        return
    try:
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        log.info("[VRAM %s] allocated=%.0fMiB reserved=%.0fMiB total=%.0fMiB", label, alloc, reserved, total)
    except Exception:
        pass


def _is_oom_or_cuda_error(exc: BaseException) -> bool:
    if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ("out of memory", "cuda error", "device-side assert"))


def _cuda_sync_safe() -> None:
    """CUDA 동기화 — 커널이 정상 완료되었는지 확인. TDR 감지."""
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except RuntimeError as e:
        log.error("CUDA synchronize 실패 (TDR 또는 드라이버 오류 가능): %s", e)
        raise


def _force_gc() -> None:
    """GPU 캐시 해제 + Python GC."""
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()


def _empty_cuda_cache() -> None:
    _force_gc()


class _inference_context:
    """torch.inference_mode() 안전 래퍼 (컨텍스트 매니저)."""

    def __enter__(self):
        self._ctx = None
        if torch is not None:
            try:
                self._ctx = torch.inference_mode()
                self._ctx.__enter__()
            except Exception:
                self._ctx = None
        return self

    def __exit__(self, *exc):
        if self._ctx is not None:
            try:
                self._ctx.__exit__(*exc)
            except Exception:
                pass
        _force_gc()
        return False


# -----------------------------------------------------------------------------
# 엔티티 파싱 / BIO 변환
# -----------------------------------------------------------------------------

def _parse_entities_from_res(
    res: Any,
    threshold: float,
    threshold_per_label: Optional[Dict[str, float]] = None,
) -> List[EntitySpan]:
    entities: List[EntitySpan] = []
    for lab, items in ((res or {}).get("entities", {}) or {}).items():
        if not isinstance(items, list):
            continue
        th = (threshold_per_label or {}).get(str(lab), threshold)
        for it in items:
            if not isinstance(it, dict):
                continue
            conf = float(it.get("confidence", 0.0))
            if conf < th or "start" not in it or "end" not in it:
                continue
            entities.append(EntitySpan(
                label=str(lab),
                text=str(it.get("text", "")),
                start=int(it["start"]),
                end=int(it["end"]),
                confidence=conf,
            ))
    return entities


def _entities_to_bio(
    entities: List[EntitySpan],
    token_spans: List[Tuple[int, int]],
    num_tokens: int,
) -> List[str]:
    bio = ["O"] * num_tokens
    occupied = [False] * num_tokens
    for ent in entities:
        tok_ids = span_to_token_indices(ent.start, ent.end, token_spans)
        tok_ids = [i for i in tok_ids if 0 <= i < num_tokens and not occupied[i]]
        if not tok_ids:
            continue
        bio[tok_ids[0]] = f"B-{ent.label}"
        for j in tok_ids[1:]:
            bio[j] = f"I-{ent.label}"
        for j in tok_ids:
            occupied[j] = True
    return bio


# -----------------------------------------------------------------------------
# 모델/어댑터 경로 헬퍼
# -----------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_model_source(model_id: str) -> str:
    root = _project_root()
    legacy = root / "module" / DOWNLOADED_MODEL_DIR
    target_dl = root / DOWNLOADED_MODEL_DIR
    if legacy.exists() and not target_dl.exists():
        try:
            shutil.move(str(legacy), str(target_dl))
        except Exception:
            pass
    for base in (root / MODEL_DIR, target_dl, root / "module" / DOWNLOADED_MODEL_DIR):
        path = base / model_id
        if path.exists() and (path / "config.json").exists():
            return str(path)
    return model_id


def _adapter_ready(path: Path) -> bool:
    for d in (path, path / "final"):
        if (d / "adapter_config.json").exists() or (d / "adapter_model.safetensors").exists():
            return True
    return False


def _resolve_adapter_dir(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    for d in (path, path / "final"):
        if (d / "adapter_config.json").exists() or (d / "adapter_model.safetensors").exists():
            return d
    return None


def _try_load_adapter(extractor: Any, adapter_path: Optional[Union[str, Path]]) -> bool:
    if not adapter_path or not hasattr(extractor, "load_adapter"):
        return False
    adp = Path(adapter_path)
    resolved = _resolve_adapter_dir(adp)
    if resolved is None or not _adapter_ready(adp):
        return False
    extractor.load_adapter(str(resolved))
    return True


# -----------------------------------------------------------------------------
# 학습 디렉터리 / 상태 관리
# -----------------------------------------------------------------------------

def get_gliner_train_dir() -> Path:
    return _project_root() / "configs" / "gliner" / "train"


def get_adapter_dir() -> Path:
    root = _project_root()
    for base in (root / MODEL_DIR, root / DOWNLOADED_MODEL_DIR):
        model_path = base / DEFAULT_MODEL_ID
        if model_path.exists():
            return model_path / "adapter"
    return root / DOWNLOADED_MODEL_DIR / DEFAULT_MODEL_ID / "adapter"


def get_adapters_dir() -> Path:
    return get_adapter_dir() / "adapters"


def get_optimized_dir() -> Path:
    return get_adapter_dir() / "optimized"


def list_adapter_runs() -> List[Path]:
    root = get_adapters_dir()
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_") and p.name[4:].isdigit()]
    runs.sort(key=lambda x: int(x.name[4:]))
    return runs


def get_next_run_id() -> int:
    runs = list_adapter_runs()
    return int(runs[-1].name[4:]) + 1 if runs else 1


def get_best_adapter_path() -> Optional[Path]:
    opt = get_optimized_dir()
    if opt.exists() and _adapter_ready(opt):
        return opt
    runs = list_adapter_runs()
    if runs and _adapter_ready(runs[-1]):
        return runs[-1]
    return None


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


def get_train_state_path() -> Path:
    return get_gliner_train_dir() / "train_state.json"


def read_train_state() -> Optional[Dict[str, Any]]:
    path = get_train_state_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_train_state(signature: str, adapter_path: Union[str, Path]) -> None:
    path = get_train_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"signature": signature, "adapter_path": str(adapter_path)}, ensure_ascii=False),
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# auto 학습 락 / 실행
# -----------------------------------------------------------------------------

def _auto_lock_path() -> Path:
    return get_adapter_dir() / ".auto_train.lock"


def _acquire_auto_lock(wait_sec: float) -> Optional[int]:
    lock_path = _auto_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + max(0.0, wait_sec)
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            return fd
        except FileExistsError:
            if time.time() >= deadline:
                return None
            time.sleep(0.1)


def _release_auto_lock(fd: Optional[int]) -> None:
    if fd is None:
        return
    try:
        os.close(fd)
    except Exception:
        pass
    try:
        _auto_lock_path().unlink(missing_ok=True)
    except Exception:
        pass


def _ensure_trained_if_auto() -> None:
    train_dir = get_gliner_train_dir()
    current_sig = get_training_data_signature(train_dir)
    if not current_sig:
        return

    state = read_train_state()
    if state and state.get("signature") == current_sig:
        return

    fd = _acquire_auto_lock(_auto_train_lock_wait_sec())
    if fd is None:
        return

    try:
        state2 = read_train_state()
        current_sig2 = get_training_data_signature(train_dir)
        if not current_sig2 or (state2 and state2.get("signature") == current_sig2):
            return

        subprocess.run(
            [
                sys.executable, "-m", "module.extractor.ner.train",
                "--train_dir", str(train_dir),
                "--out_dir", str(get_adapter_dir()),
            ],
            cwd=str(_project_root()),
            check=False,
        )
    finally:
        _release_auto_lock(fd)


# -----------------------------------------------------------------------------
# BIO 변환 (train.py에서도 사용)
# -----------------------------------------------------------------------------

def bio_to_ner_spans(labels: List[str]) -> List[List[Union[int, str]]]:
    spans: List[List[Union[int, str]]] = []
    i = 0
    while i < len(labels):
        tag = labels[i]
        if tag.startswith("B-"):
            label_name = tag[2:]
            start = i
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{label_name}":
                j += 1
            spans.append([start, j - 1, label_name])
            i = j
        else:
            i += 1
    return spans


# -----------------------------------------------------------------------------
# 라벨 로딩
# -----------------------------------------------------------------------------

def _get_predict_dir() -> Path:
    return _project_root() / "configs" / "gliner" / "predict"


def load_labels_from_predict(
    predict_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    root = Path(predict_dir) if predict_dir is not None else _get_predict_dir()
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
    root = _project_root()
    candidates = [
        root / "configs" / "gliner" / "predict",
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


def load_labels_from_yaml(
    yaml_path: Optional[Union[str, Path]] = None,
    key: str = "ner",
) -> Tuple[List[str], Dict[str, str]]:
    import yaml

    if yaml_path is None:
        yaml_path = _project_root() / "configs" / "labels.yaml"
    yaml_path = Path(yaml_path)

    if yaml_path.exists():
        cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        ner_cfg = cfg.get(key, {}) if isinstance(cfg, dict) else {}
        labels_obj = ner_cfg.get("labels")

        if labels_obj == "auto":
            return load_labels_from_predict()
        if isinstance(labels_obj, dict):
            return list(labels_obj.keys()), {str(k): str(v) for k, v in labels_obj.items()}
        if isinstance(labels_obj, list):
            labels = [str(x) for x in labels_obj]
            return labels, _load_label_descriptions_from_files(labels)

    labels, desc = load_labels_from_predict()
    if labels:
        return labels, desc
    labels = ["ch_co_address", "ch_co_company", "ch_co_name", "ch_co_phone", "ch_co_email", "copyright_url", "copyright_date"]
    return labels, _load_label_descriptions_from_files(labels)


def load_ner_thresholds_from_yaml(
    yaml_path: Optional[Union[str, Path]] = None,
) -> Tuple[float, Dict[str, float]]:
    import yaml

    if yaml_path is None:
        yaml_path = _project_root() / "configs" / "labels.yaml"
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        return DEFAULT_THRESHOLD, {}

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    ner_cfg = cfg.get("ner", {}) if isinstance(cfg, dict) else {}
    default = float(ner_cfg.get("threshold", DEFAULT_THRESHOLD))
    per_label = ner_cfg.get("threshold_per_label")
    per_label = {str(k): float(v) for k, v in per_label.items()} if isinstance(per_label, dict) else {}
    return default, per_label


# -----------------------------------------------------------------------------
# Decision 빌더
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# 모델 싱글턴
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_zeroshot_model() -> ZeroShotNER:
    return ZeroShotNER()


@lru_cache(maxsize=1)
def _get_auto_model() -> ZeroShotNER:
    for _name in ("gliner2", "gliner2.training", "gliner2.inference", "transformers"):
        logging.getLogger(_name).setLevel(logging.WARNING)
    return ZeroShotNER(auto=True)


# -----------------------------------------------------------------------------
# 전처리
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# export
# -----------------------------------------------------------------------------

def ner_extractor(
    *,
    sentences: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
    threshold: Optional[float] = None,
    auto: bool = True,
) -> List[Decision]:
    sentence_texts, sentence_info = _prepare_sentence_input(sentences, tokens)
    if not sentence_texts:
        return []

    model = _get_auto_model() if auto else _get_zeroshot_model()
    predicted_labels_list = model.predict(sentence_texts, threshold=threshold)

    decisions: List[Decision] = []
    for (sid, sent_tokens), predicted_labels in zip(sentence_info, predicted_labels_list):
        decisions.extend(_build_decisions_from_bio(sent_tokens, predicted_labels, int(sid)))
    return decisions
