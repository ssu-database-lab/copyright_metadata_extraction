"""Zero-shot NER (GLiNER2 기반). auto 시 학습 데이터 변경 시에만 학습 후 predict."""
from __future__ import annotations

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


# -----------------------------------------------------------------------------
# 변수 선언
# -----------------------------------------------------------------------------

DEFAULT_MODEL_ID = "fastino/gliner2-large-v1"
DEFAULT_THRESHOLD = 0.55
# 한 번에 NER에 넣을 최대 토큰 수. 초과 시 잘라서 처리해 메모리/멈춤 방지. env GLINER_MAX_CHUNK_TOKENS 로 변경 가능.
def _max_chunk_tokens() -> int:
    try:
        return max(64, int(os.environ.get("GLINER_MAX_CHUNK_TOKENS", "512")))
    except ValueError:
        return 512


# 배치 추론 시 한 번에 넣을 문장/청크 개수. env GLINER_NER_BATCH_SIZE 로 변경 가능.
# 12GB GPU OOM 시 1~2 유지. 긴 청크는 자동으로 batch_size=1 로 처리.
def _ner_batch_size() -> int:
    try:
        return max(1, int(os.environ.get("GLINER_NER_BATCH_SIZE", "2")))
    except ValueError:
        return 2

# 마이크로 배치(우리쪽 분할) 크기. env GLINER_MICRO_BATCH_SIZE 로 변경 가능.
# 12GB GPU OOM 시 2~4 유지. 긴 청크는 1건씩 처리.
def _micro_batch_size() -> int:
    try:
        return max(1, int(os.environ.get("GLINER_MICRO_BATCH_SIZE", "4")))
    except ValueError:
        return 4


# GPU 메모리 상한(비율 0~1). 설정 시 torch.cuda.set_per_process_memory_fraction 적용.
# WSL2 등에서 cudaErrorUnknown 완화용. 토큰/청크 제한은 그대로 두는 것이 안전함.
def _cuda_memory_fraction() -> Optional[float]:
    val = os.environ.get("GLINER_CUDA_MEMORY_FRACTION", "").strip()
    if not val:
        return None
    try:
        f = float(val)
        return None if f <= 0 or f > 1 else f
    except ValueError:
        return None


# auto 학습 쿨다운(초). 너무 자주 학습 띄우는 것 방지.
def _auto_train_cooldown_sec() -> int:
    try:
        return max(0, int(os.environ.get("GLINER_AUTO_TRAIN_COOLDOWN_SEC", "300")))
    except ValueError:
        return 300


# 락 경합 시 최대 대기(초). 길게 대기하면 추론이 멎을 수 있어 짧게.
def _auto_train_lock_wait_sec() -> float:
    try:
        return max(0.0, float(os.environ.get("GLINER_AUTO_TRAIN_LOCK_WAIT_SEC", "2.0")))
    except ValueError:
        return 2.0


# auto 모델에서 '학습까지' 수행할지. (운영 추론 환경에서 0으로 끄면 학습 subprocess를 띄우지 않음)
def _auto_train_enabled() -> bool:
    return os.environ.get("GLINER_AUTO_TRAIN", "1") not in ("0", "false", "False", "no", "NO")



MAX_CHUNK_TOKENS = 512
MODEL_DIR = "models"
DOWNLOADED_MODEL_DIR = "model_downloaded"


# -----------------------------------------------------------------------------
# class 선언 (데코레이터 있는 것 먼저)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class EntitySpan:
    label: str
    text: str
    start: int
    end: int
    confidence: float



class ZeroShotNER:
    """
    - 입력: List[List[str]] (토큰 단위)
    - 출력: List[List[str]] (BIO)
    - 내부: GLiNER2 zero-shot extract_entities 사용

    auto=True 모드 동작:
    - 학습 데이터(signature)가 변하지 않았으면: adapter 로드/유지 후 추론만
    - 변했으면:
      1) 현재 프로세스가 잡고 있던 GPU VRAM을 먼저 비움(모델 offload)
      2) 학습 subprocess 실행 (module.extractor.ner.train)
      3) train_state.json 기준으로 최신 adapter 재로드 후 추론

    폭주 방지:
    - 학습 전 offload + empty_cache
    - 학습 중복 실행 방지: lockfile + 재검증(락 획득 후 다시 signature 비교)
    - 추론 VRAM 피크 완화: batch_texts를 마이크로 배치로 쪼개 batch_extract_entities 호출
    """

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

        # preferred device
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
            # auto: 초기에는 CPU 유지 (학습 필요 시 VRAM 경합 방지)
            self._ensure_on_cpu()

        if hasattr(self.extractor, "eval"):
            self.extractor.eval()

    # ---------------------------
    # device helpers
    # ---------------------------

    def _ensure_on_cpu(self) -> None:
        if self.device == "cpu":
            return
        try:
            if hasattr(self.extractor, "to"):
                self.extractor = self.extractor.to("cpu")
        finally:
            self.device = "cpu"
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def _ensure_on_cuda(self) -> None:
        if self.device == "cuda":
            return
        if torch is None or not torch.cuda.is_available() or not hasattr(self.extractor, "to"):
            self.device = "cpu"
            return
        frac = _cuda_memory_fraction()
        if frac is not None:
            try:
                torch.cuda.set_per_process_memory_fraction(frac, 0)
            except Exception:
                pass
        self.extractor = self.extractor.to("cuda")
        self.device = "cuda"

    def _ensure_on_preferred_device(self) -> None:
        if self._preferred_device == "cuda":
            self._ensure_on_cuda()
        else:
            self._ensure_on_cpu()

    # ---------------------------
    # auto adapter / training
    # ---------------------------

    def _maybe_load_adapter(self, adapter_path: Optional[str]) -> None:
        if not adapter_path:
            return
        if self._loaded_adapter_path == adapter_path:
            return
        if _try_load_adapter(self.extractor, adapter_path):
            self._loaded_adapter_path = adapter_path

    def _refresh_adapter_auto(self) -> None:
        train_dir = get_gliner_train_dir()
        current_sig = get_training_data_signature(train_dir)
        state = read_train_state() or {}
        state_sig = state.get("signature")
        adapter_path = state.get("adapter_path")

        # 변경 없음 → adapter만 최신으로 맞춤
        if current_sig and state_sig == current_sig:
            if adapter_path is None:
                best = get_best_adapter_path()
                adapter_path = str(best) if best is not None else None
            self._maybe_load_adapter(adapter_path)
            self._ensure_on_preferred_device()
            return

        # 학습 데이터 자체가 없으면 adapter만
        if not current_sig:
            best = get_best_adapter_path()
            self._maybe_load_adapter(str(best) if best else None)
            self._ensure_on_preferred_device()
            return

        # 변경됨 → 쿨다운/활성화 체크
        if not _auto_train_enabled():
            best = get_best_adapter_path()
            self._maybe_load_adapter(str(best) if best else None)
            self._ensure_on_preferred_device()
            return

        now = time.time()
        if (now - self._last_auto_train_attempt_ts) < float(_auto_train_cooldown_sec()):
            best = get_best_adapter_path()
            self._maybe_load_adapter(str(best) if best else None)
            self._ensure_on_preferred_device()
            return

        self._last_auto_train_attempt_ts = now

        # ✅ 학습 전에 VRAM 경합 방지: CPU offload
        self._ensure_on_cpu()

        _ensure_trained_if_auto()

        # 학습 결과 반영
        state2 = read_train_state() or {}
        adapter2 = state2.get("adapter_path")
        if adapter2 is None:
            best = get_best_adapter_path()
            adapter2 = str(best) if best is not None else None
        self._maybe_load_adapter(adapter2)

        self._ensure_on_preferred_device()

    # ---------------------------
    # predict
    # ---------------------------

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
        schema: Union[List[str], Dict[str, str]] = (
            {k: self.label_descriptions.get(k, k) for k in self.labels}
            if use_descriptions and self.label_descriptions
            else self.labels
        )
        max_chunk = getattr(self, "max_chunk_tokens", None) or _max_chunk_tokens()
        batch_size = _ner_batch_size()
        micro_bs = _micro_batch_size()
        th_per_label = getattr(self, "threshold_per_label", None)

        # (tokens, out_idx) 단위로 작업 목록 생성
        jobs: List[Tuple[List[str], int]] = []
        for out_idx, tokens in enumerate(texts):
            if len(tokens) <= max_chunk:
                jobs.append((tokens, out_idx))
            else:
                for start in range(0, len(tokens), max_chunk):
                    jobs.append((tokens[start : start + max_chunk], out_idx))

        if not jobs:
            return []

        batch_texts: List[str] = []
        for tokens, _ in jobs:
            raw_text, _ = join_tokens_with_spans(tokens)
            batch_texts.append(normalize_ocr_text(raw_text))

        # 마이크로 배치로 분할 호출 → VRAM 피크 완화
        results_all: List[Any] = []
        ctx = None
        if torch is not None:
            try:
                ctx = torch.inference_mode()
                ctx.__enter__()
            except Exception:
                ctx = None

        # 긴 텍스트(문자 수 기준)는 VRAM 폭증 방지를 위해 1건씩, batch_size=1 로 처리
        long_char_threshold = 500
        try:
            for i in range(0, len(batch_texts), micro_bs):
                chunk = batch_texts[i : i + micro_bs]
                max_chunk_len = max(len(s) for s in chunk) if chunk else 0
                use_batch_size = 1 if max_chunk_len > long_char_threshold else batch_size
                if max_chunk_len > long_char_threshold:
                    for text in chunk:
                        one_result = self.extractor.batch_extract_entities(
                            [text],
                            schema,
                            batch_size=1,
                            threshold=th,
                            include_confidence=True,
                            include_spans=True,
                        )
                        if isinstance(one_result, list):
                            results_all.extend(one_result)
                        else:
                            results_all.append(one_result)
                else:
                    chunk_results = self.extractor.batch_extract_entities(
                        chunk,
                        schema,
                        batch_size=use_batch_size,
                        threshold=th,
                        include_confidence=True,
                        include_spans=True,
                    )
                    if isinstance(chunk_results, list):
                        results_all.extend(chunk_results)
                    else:
                        results_all.append(chunk_results)
        finally:
            if ctx is not None:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass

        outputs: List[List[str]] = [[] for _ in range(len(texts))]
        for i, (tokens, out_idx) in enumerate(jobs):
            res = results_all[i] if i < len(results_all) else {}
            _, token_spans = join_tokens_with_spans(tokens)
            entities = _parse_entities_from_res(res, th, th_per_label)
            entities.sort(key=lambda x: x.confidence, reverse=True)
            bio = _entities_to_bio(entities, token_spans, len(tokens))
            outputs[out_idx].extend(bio)

        return outputs


# -----------------------------------------------------------------------------
# function (private 우선)
# -----------------------------------------------------------------------------

def _parse_entities_from_res(
    res: Any,
    threshold: float,
    threshold_per_label: Optional[Dict[str, float]] = None,
) -> List[EntitySpan]:
    """extract_entities 결과에서 EntitySpan 리스트 추출 (라벨별 threshold 적용)."""
    entities: List[EntitySpan] = []
    ent_map = (res or {}).get("entities", {}) or {}
    for lab, items in ent_map.items():
        if not isinstance(items, list):
            continue
        th = (threshold_per_label or {}).get(str(lab), threshold)
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
    return entities


def _entities_to_bio(
    entities: List[EntitySpan],
    token_spans: List[Tuple[int, int]],
    num_tokens: int,
) -> List[str]:
    """엔티티 스팬 리스트를 BIO 시퀀스로 변환 (겹치면 confidence 순 적용됨)."""
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


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_model_source(model_id: str) -> str:
    """사용할 모델 경로 결정: 로컬에 있으면 그 경로, 없으면 model_id(HuggingFace) 반환."""
    root = _project_root()
    # 예전에 module/model_downloaded 에 받아둔 경우 프로젝트 루트로 한 번만 이전
    legacy = root / "module" / DOWNLOADED_MODEL_DIR
    target_dl = root / DOWNLOADED_MODEL_DIR
    if legacy.exists() and not target_dl.exists():
        try:
            shutil.move(str(legacy), str(target_dl))
        except Exception:
            pass

    # model_id에 해당하는 로컬 디렉터리만 사용 (학습 시 쓴 base와 동일해야 LoRA shape 일치)
    for base in (root / MODEL_DIR, target_dl, root / "module" / DOWNLOADED_MODEL_DIR):
        path = base / model_id
        if path.exists() and (path / "config.json").exists():
            return str(path)

    return model_id


def _adapter_ready(path: Path) -> bool:
    """어댑터 파일이 path 또는 path/final 에 있는지."""
    if (path / "adapter_config.json").exists() or (path / "adapter_model.safetensors").exists():
        return True
    sub = path / "final"
    return (sub / "adapter_config.json").exists() or (sub / "adapter_model.safetensors").exists()


def _resolve_adapter_dir(path: Path) -> Optional[Path]:
    """실제 로드할 어댑터 디렉터리 (본인 또는 final/)."""
    if not path.exists():
        return None
    if (path / "adapter_config.json").exists() or (path / "adapter_model.safetensors").exists():
        return path
    sub = path / "final"
    if sub.exists() and (
        (sub / "adapter_config.json").exists() or (sub / "adapter_model.safetensors").exists()
    ):
        return sub
    return None


def _try_load_adapter(extractor: Any, adapter_path: Optional[Union[str, Path]]) -> bool:
    """어댑터 경로가 유효하면 로드. 성공 여부 반환."""
    if not adapter_path or not hasattr(extractor, "load_adapter"):
        return False
    adp = Path(adapter_path)
    resolved = _resolve_adapter_dir(adp)
    if resolved is None or not _adapter_ready(adp):
        return False
    extractor.load_adapter(str(resolved))
    return True


def get_gliner_train_dir() -> Path:
    """학습 데이터 디렉터리 (configs/gliner/train). 라벨별 .jsonl."""
    return _project_root() / "configs" / "gliner" / "train"


def get_adapter_dir() -> Path:
    """어댑터 루트 (configs/gliner/train/adapter). 아래에 adapters/run_*, optimized/."""
    return get_gliner_train_dir() / "adapter"


def get_adapters_dir() -> Path:
    """run 누적 디렉터리 (adapter/adapters/)."""
    return get_adapter_dir() / "adapters"


def get_optimized_dir() -> Path:
    """최적화 후 단일 어댑터 (adapter/optimized/)."""
    return get_adapter_dir() / "optimized"


def list_adapter_runs() -> List[Path]:
    """adapters/ 아래 run_XXXXXX 목록 (숫자 순)."""
    root = get_adapters_dir()
    if not root.exists():
        return []
    runs: List[Path] = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("run_") and p.name[4:].isdigit():
            runs.append(p)
    runs.sort(key=lambda x: int(x.name[4:]))
    return runs


def get_next_run_id() -> int:
    """다음 run 번호 (1부터)."""
    runs = list_adapter_runs()
    return int(runs[-1].name[4:]) + 1 if runs else 1


def get_best_adapter_path() -> Optional[Path]:
    """예측 시 사용할 어댑터: optimized/ 우선, 없으면 최신 run."""
    if get_optimized_dir().exists() and _adapter_ready(get_optimized_dir()):
        return get_optimized_dir()
    runs = list_adapter_runs()
    if not runs:
        return None
    latest = runs[-1]
    return latest if _adapter_ready(latest) else None


def get_training_data_signature(train_dir: Union[str, Path]) -> str:
    """학습 데이터 디렉터리 내용의 해시. 변경 여부 검사용."""
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
    """학습 상태 파일 (configs/gliner/train/train_state.json)."""
    return get_gliner_train_dir() / "train_state.json"


def read_train_state() -> Optional[Dict[str, Any]]:
    """저장된 학습 상태. {"signature": ..., "adapter_path": ...} 또는 None."""
    path = get_train_state_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_train_state(signature: str, adapter_path: Union[str, Path]) -> None:
    """학습 후 상태 저장. train.py 에서 호출."""
    path = get_train_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"signature": signature, "adapter_path": str(adapter_path)}, ensure_ascii=False),
        encoding="utf-8",
    )


def _auto_lock_path() -> Path:
    return get_adapter_dir() / ".auto_train.lock"


def _acquire_auto_lock(wait_sec: float) -> Optional[int]:
    """단순 lockfile (O_EXCL)로 중복 학습 방지. 성공 시 fd 반환."""
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
    """어댑터 확인 → 학습 데이터 변경 여부 확인 → 변경 시 추가 학습, 없으면 그대로."""
    train_dir = get_gliner_train_dir()
    current_sig = get_training_data_signature(train_dir)
    if not current_sig:
        return

    state = read_train_state()
    if state and state.get("signature") == current_sig:
        return

    fd = _acquire_auto_lock(_auto_train_lock_wait_sec())
    if fd is None:
        # 다른 프로세스가 학습 중일 가능성이 높음 → 여기서는 멎지 않도록 그냥 반환
        return

    try:
        # 락 획득 후 재검증 (경합 상황 방지)
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

def bio_to_ner_spans(labels: List[str]) -> List[List[Union[int, str]]]:
    """BIO 시퀀스 → GLiNER2 ner 형식 [[start, end, label], ...] (end 포함). train.py 병합용."""
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
            end = j - 1
            spans.append([start, end, label_name])
            i = j
        else:
            i += 1
    return spans


def _get_predict_dir() -> Path:
    """추론용 라벨 설명 디렉터리 (configs/gliner/predict). train도 이 라벨 집합에 맞춰 동작."""
    return _project_root() / "configs" / "gliner" / "predict"


def load_labels_from_predict(
    predict_dir: Optional[Union[str, Path]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """
    configs/gliner/predict/ 에서 라벨을 자동 수집합니다.
    .txt, .md 파일명의 stem 이 라벨명, 파일 내용이 설명입니다.
    반환: (정렬된 labels_list, label_descriptions_dict).
    predict만 쓰면 train도 이 라벨 집합으로 은연중에 동작하게 할 때 사용.
    """
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
    labels = sorted(descriptions.keys())
    return labels, descriptions


def _load_label_descriptions_from_files(labels: List[str]) -> Dict[str, str]:
    """
    configs/gliner/predict/ (추천), configs/gliner/, configs/training/ner_labels/ 에서
    라벨 설명 파일을 읽습니다. .txt, .md 지원, 파일명: {label}.txt 또는 {label}.md
    """
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


@lru_cache(maxsize=1)
def _get_zeroshot_model() -> ZeroShotNER:
    return ZeroShotNER()


@lru_cache(maxsize=1)
def _get_auto_model() -> ZeroShotNER:
    """auto 모델 1회 로드 후 재사용 (파일마다 재로드 방지)."""
    for _name in ("gliner2", "gliner2.training", "gliner2.inference", "transformers"):
        logging.getLogger(_name).setLevel(logging.WARNING)
    return ZeroShotNER(auto=True)


# -----------------------------------------------------------------------------
# public function (내부/유틸)
# -----------------------------------------------------------------------------

def load_labels_from_yaml(
    yaml_path: Optional[Union[str, Path]] = None,
    key: str = "ner",
) -> Tuple[List[str], Dict[str, str]]:
    """
    labels.yaml에서 라벨을 읽습니다.
    - ner.labels 가 "auto" 이면 configs/gliner/predict/ 에서만 라벨 수집 (predict = 소스, train도 같은 라벨로 동작).
    - 그 외:
      1) ner: {labels: ["email", ...]}
      2) ner: {labels: {email: "이메일 주소", ...}}
    반환: (labels_list, label_descriptions_dict)
    """
    import yaml  # 지연 import

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
            labels = list(labels_obj.keys())
            return labels, {str(k): str(v) for k, v in labels_obj.items()}
        if isinstance(labels_obj, list):
            labels = [str(x) for x in labels_obj]
            return labels, _load_label_descriptions_from_files(labels)

    # 기본값: yaml 없거나 labels 없으면 predict 에서 시도, 없으면 fallback
    labels, desc = load_labels_from_predict()
    if labels:
        return labels, desc
    labels = ["address", "company_name", "person_name", "phone_number", "email", "url", "date"]
    return labels, _load_label_descriptions_from_files(labels)


def load_ner_thresholds_from_yaml(
    yaml_path: Optional[Union[str, Path]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    labels.yaml의 ner.threshold, ner.threshold_per_label을 읽습니다.
    zero-shot에서 트레이닝 없이 라벨별로 recall(낮춤)/precision(높임) 조정용.
    반환: (default_threshold, {label: threshold, ...})
    """
    import yaml  # 지연 import

    if yaml_path is None:
        yaml_path = _project_root() / "configs" / "labels.yaml"
    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        return DEFAULT_THRESHOLD, {}

    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    ner_cfg = cfg.get("ner", {}) if isinstance(cfg, dict) else {}
    default = float(ner_cfg.get("threshold", DEFAULT_THRESHOLD))
    per_label = ner_cfg.get("threshold_per_label")
    if isinstance(per_label, dict):
        per_label = {str(k): float(v) for k, v in per_label.items()}
    else:
        per_label = {}
    return default, per_label


def _prepare_sentence_input(
    sentences: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
) -> Tuple[List[List[str]], List[Tuple[int, List[Dict[str, Any]]]]]:
    """sentences + tokens → (sentence_texts, sentence_info). 비어 있으면 ([], [])."""
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
