"""v4 평가 — 설명(description) 임베딩 판정기 (docs/v4/M1_evaluator_spec.md §5 Description rule).

판정 = 코사인(nlpai-lab/KURE-v1, last_hidden_state 의 CLS 토큰, L2 정규화) ≥ τ
       AND (content_check 가 켜져 있으면) 예측·gold 가 내용 토큰을 1개 이상 공유.
torch/transformers 는 첫 임베딩 때 지연 import 한다. `embedder` 속성은 호출 가능 객체이므로
테스트는 고정 벡터를 돌려주는 스텁을 주입해 모델을 올리지 않는다.
임베딩 캐시: sha1(정규화 텍스트) → 벡터, jsonl 한 줄씩 추가 기록. 각 줄에 model/revision/device 를 함께 적고,
모든 텍스트가 캐시에서 나와 모델을 올리지 않는 재실행에서는 그 값을 넘겨받아 결과(settings.embed)에 기록한다
— 그래야 따뜻한 캐시 재실행이 처음 실행과 같은 test_metrics.json 을 낸다 (§5 'record the revision', §7 결정성).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .normalize import norm_tag, norm_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "nlpai-lab/KURE-v1"
DEFAULT_TAU = 0.65
MAX_LENGTH = 512

# §5 불용어 — 그대로 옮겼다. 토큰 길이 ≥ 2 규칙 때문에 1글자 항목은 사실상 걸러지지만 명시해 둔다.
STOPLIST_RAW: tuple[str, ...] = (
    "이", "그", "저", "있다", "있는", "보인다", "모습", "사진", "영상", "이미지", "장면",
    "것", "수", "등", "및", "와", "과", "의", "를", "을", "은", "는", "가", "이다", "한다", "된다",
)
STOPLIST: frozenset[str] = frozenset(t for t in (norm_tag(w) for w in STOPLIST_RAW) if t)
# 내용 토큰 = 한글·라틴·숫자의 최대 연속 구간 (길이 ≥ 2)
_CONTENT_RUN = re.compile(r"[0-9A-Za-z가-힣]+")

Embedder = Callable[[str], np.ndarray]


def content_tokens(text: Any) -> list[str]:
    """§5 내용 토큰: 한글/라틴/숫자 연속 구간(≥2자) → norm_tag → 불용어 제외. 정렬·중복 제거."""
    t = norm_text(text) or ""
    toks: set[str] = set()
    for run in _CONTENT_RUN.findall(t):
        if len(run) < 2:
            continue
        tok = norm_tag(run)
        if tok and len(tok) >= 2 and tok not in STOPLIST:
            toks.add(tok)
    return sorted(toks)


def shared_content_tokens(a: Any, b: Any) -> list[str]:
    return sorted(set(content_tokens(a)) & set(content_tokens(b)))


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _adopt(current: Optional[str], found: set[str], label: str) -> Optional[str]:
    """캐시 줄에서 읽은 provenance(revision/device)를 넘겨받는다 — 이미 값이 있으면 그대로, 하나로 모일 때만 채택.

    여러 값이 섞여 있으면(모델 버전이 다른 실행이 같은 캐시 파일에 쓴 경우) 경고하고 기록하지 않는다(None)."""
    if current is not None or not found:
        return current
    if len(found) > 1:
        logger.warning("임베딩 캐시에 %s 가 %d개 섞여 있다 %s — 결과에 기록하지 않는다 "
                       "(revision 은 --embed-revision 으로 고정하면 다른 revision 줄은 무시된다)", label, len(found), sorted(found))
        return None
    return next(iter(found))


class EmbedJudge:
    """설명 판정기. decide(pred, gold) → {match, cosine, shared_tokens, tau, content_check, content_ok}."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        revision: Optional[str] = None,
        device: Optional[str] = None,
        cache_path: Optional[str | Path] = None,
        tau: float = DEFAULT_TAU,
        content_check: bool = True,
        embedder: Optional[Embedder] = None,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.model_id = model_id
        self.revision = revision                            # --embed-revision 고정값(요청)
        self.resolved_revision: Optional[str] = revision    # 결과에 기록: 고정값 > 적재한 모델 > 캐시 줄
        self.device = device                                # 모델을 올릴 장치(요청; None 이면 자동)
        self.resolved_device: Optional[str] = None          # 결과에 기록: 적재한 모델 > 캐시 줄 (요청값은 사실이 아니라 안 쓴다)
        self.tau = float(tau)
        self.content_check = bool(content_check)
        self.max_length = max_length
        self.cache_path = Path(cache_path) if cache_path else None
        self.embedder: Embedder = embedder if embedder is not None else self._hf_embed
        self._model: Any = None
        self._tokenizer: Any = None
        self._cache: dict[str, np.ndarray] = {}
        self.n_embed_calls = 0
        self._load_cache()

    # ------------------------------------------------------------------ 판정
    def embed(self, text: Any) -> np.ndarray:
        """정규화 텍스트의 임베딩 — 캐시에 있으면 재사용, 없으면 embedder 호출 후 캐시에 기록."""
        t = norm_text(text) or ""
        key = text_sha1(t)
        vec = self._cache.get(key)
        if vec is None:
            vec = np.asarray(self.embedder(t), dtype=np.float32).ravel()
            self.n_embed_calls += 1
            self._cache[key] = vec
            self._append_cache(key, vec)
        return vec

    def decide(self, pred_text: Any, gold_text: Any) -> dict[str, Any]:
        p = norm_text(pred_text) or ""
        g = norm_text(gold_text) or ""
        cos = cosine(self.embed(p), self.embed(g)) if p and g else 0.0
        shared = shared_content_tokens(p, g)
        content_ok = (not self.content_check) or bool(shared)
        return {
            "match": bool(cos >= self.tau and content_ok),
            "cosine": cos,
            "shared_tokens": shared,
            "tau": self.tau,
            "content_check": self.content_check,
            "content_ok": content_ok,
        }

    def info(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.resolved_revision,
            "device": self.resolved_device,
            "tau": self.tau,
            "content_check": self.content_check,
            "max_length": self.max_length,
            "pooling": "cls",
            "cache_path": str(self.cache_path) if self.cache_path else None,
        }

    # ------------------------------------------------------------------ HF 모델 (지연)
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch  # 지연 import — 테스트/T-001 경로에서는 불필요
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs: dict[str, Any] = {"revision": self.revision} if self.revision else {}
        try:
            tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True, **kwargs)
            model = AutoModel.from_pretrained(self.model_id, local_files_only=True, **kwargs)
        except OSError:
            logger.info("로컬 캐시에 없어 허브에서 1회 내려받습니다: %s", self.model_id)
            tok = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
            model = AutoModel.from_pretrained(self.model_id, **kwargs)
        model = model.eval().to(self.device)
        if str(self.device).startswith("cuda"):
            model = model.half()
        model_revision = getattr(model.config, "_commit_hash", None) or self.revision
        if model_revision and self.resolved_revision and model_revision != self.resolved_revision:
            logger.warning("적재한 모델 revision(%s)이 캐시 줄의 revision(%s)과 다르다 — 이번 실행의 벡터가 섞인다",
                           model_revision, self.resolved_revision)
        self.resolved_revision = model_revision or self.resolved_revision
        self.resolved_device = str(self.device)
        self._tokenizer, self._model = tok, model
        logger.info("임베딩 모델 적재: %s (revision=%s, device=%s)", self.model_id, self.resolved_revision, self.resolved_device)

    def _hf_embed(self, text: str) -> np.ndarray:
        self._ensure_model()
        import torch

        with torch.inference_mode():
            enc = self._tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            hidden = self._model(**enc).last_hidden_state[:, 0, :]  # CLS 토큰
            vec = torch.nn.functional.normalize(hidden.float(), p=2, dim=-1)[0]
        return vec.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ 캐시
    def _parse_cache_line(self, line: str) -> Optional[dict[str, Any]]:
        """캐시 한 줄 → 레코드. 깨진 줄, 다른 모델, (고정했을 때) 다른 revision 의 줄은 None."""
        line = line.strip()
        if not line:
            return None
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(rec, dict) or rec.get("model") != self.model_id or "sha1" not in rec or "vec" not in rec:
            return None
        if self.revision and rec.get("revision") and rec["revision"] != self.revision:
            return None
        return rec

    def _load_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        n = 0
        revisions: set[str] = set()
        devices: set[str] = set()
        with self.cache_path.open(encoding="utf-8") as f:
            for line in f:
                rec = self._parse_cache_line(line)
                if rec is None:
                    continue
                self._cache[rec["sha1"]] = np.asarray(rec["vec"], dtype=np.float32)
                n += 1
                if rec.get("revision"):
                    revisions.add(str(rec["revision"]))
                if rec.get("device"):
                    devices.add(str(rec["device"]))
        # 모델을 올리지 않는 재실행에서도 provenance 가 비지 않도록 캐시 줄의 값을 넘겨받는다
        self.resolved_revision = _adopt(self.resolved_revision, revisions, "revision")
        self.resolved_device = _adopt(self.resolved_device, devices, "device")
        logger.info("임베딩 캐시 적재: %d개 (%s, revision=%s, device=%s)",
                    n, self.cache_path, self.resolved_revision, self.resolved_device)

    def _append_cache(self, key: str, vec: np.ndarray) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        # float32 → Python float 는 무손실이므로 반올림하지 않는다 (재실행 때 코사인이 같아야 한다)
        rec = {"sha1": key, "model": self.model_id, "revision": self.resolved_revision, "device": self.resolved_device,
               "vec": [float(x) for x in vec.tolist()]}
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


__all__ = [
    "DEFAULT_MODEL_ID", "DEFAULT_TAU", "MAX_LENGTH", "STOPLIST", "Embedder",
    "content_tokens", "shared_content_tokens", "cosine", "text_sha1", "EmbedJudge",
]
