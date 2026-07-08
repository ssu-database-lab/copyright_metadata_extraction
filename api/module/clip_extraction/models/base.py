"""
Common interface for all vision-language models in the benchmark.

Each concrete model implements load/classify/encode_image/encode_text/unload.
This keeps the benchmark runner model-agnostic.
"""

from __future__ import annotations

import abc
import gc
import time
from typing import Any

import numpy as np
import torch
from PIL import Image


class BaseVLM(abc.ABC):
    """Abstract base for any model the benchmark can run."""

    # Subclasses set these
    name: str = ""
    hf_id: str = ""
    license: str = ""
    params: str = ""        # e.g. "428M"
    korean_support: str = ""  # "native" | "via translation" | "none"

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False
        self._load_time_s: float | None = None

    # ---- lifecycle -------------------------------------------------------
    @abc.abstractmethod
    def _load(self) -> None:
        """Load weights + processor. Sets self.model, self.processor."""

    def load(self) -> None:
        if self._loaded:
            return
        t0 = time.perf_counter()
        self._load()
        self._load_time_s = time.perf_counter() - t0
        self._loaded = True

    def unload(self) -> None:
        for attr in ("model", "processor", "tokenizer", "translator"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False

    # ---- core inference --------------------------------------------------
    @abc.abstractmethod
    def classify(
        self,
        image: Image.Image,
        candidate_labels: list[str],
    ) -> dict[str, float]:
        """Return a dict {label: probability} summing to ~1.0."""

    @abc.abstractmethod
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Return a normalized image embedding vector (1D np.ndarray)."""

    @abc.abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Return a normalized text embedding vector (1D np.ndarray)."""

    # ---- helpers ---------------------------------------------------------
    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hf_id": self.hf_id,
            "license": self.license,
            "params": self.params,
            "korean_support": self.korean_support,
            "device": self.device,
            "load_time_s": round(self._load_time_s, 2) if self._load_time_s else None,
        }


def softmax(x: np.ndarray, dim: int = -1) -> np.ndarray:
    """Plain-numpy stable softmax."""
    x = x - x.max(axis=dim, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=dim, keepdims=True)


def normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector or row-wise for a matrix."""
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 0, v / n, v)
