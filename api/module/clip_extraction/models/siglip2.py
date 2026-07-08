"""
Google SigLIP 2 — multilingual vision-language encoder, Feb 2025.

109 languages, sigmoid loss instead of softmax-CE. Outperforms SigLIP-1
at every model size on zero-shot classification + retrieval.

Default checkpoint: SO400M-patch14-384 (good accuracy/cost tradeoff).
Swap to siglip2-base-patch16-512 if VRAM-limited (~85M params).

Requires transformers >= 4.49 (released Feb 2025).
"""

import numpy as np
import torch
from PIL import Image

from .base import BaseVLM, normalize


class SigLIP2(BaseVLM):
    name = "SigLIP 2 SO400M/14-384"
    hf_id = "google/siglip2-so400m-patch14-384"
    license = "Apache-2.0"
    params = "400M"
    korean_support = "native (109 langs)"

    def _load(self) -> None:
        # transformers 4.49+ exposes SigLIP 2 via AutoModel/AutoProcessor
        from transformers import AutoModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = AutoModel.from_pretrained(self.hf_id).to(self.device).eval()

    @torch.inference_mode()
    def classify(self, image: Image.Image, candidate_labels: list[str]) -> dict[str, float]:
        # SigLIP 2 requires padding="max_length" for correct text tokenization
        inputs = self.processor(
            text=candidate_labels,
            images=image,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**inputs)
        # Sigmoid (NOT softmax) is the SigLIP scoring function. We still
        # softmax across the candidate set to get a label distribution.
        logits = out.logits_per_image.squeeze(0).cpu().numpy()
        # Use temperature 1.0 since SigLIP logits are already calibrated.
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return dict(zip(candidate_labels, probs.tolist()))

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        v = self.model.get_image_features(**inputs).squeeze(0).cpu().numpy()
        return normalize(v)

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text], padding="max_length", return_tensors="pt"
        ).to(self.device)
        v = self.model.get_text_features(**inputs).squeeze(0).cpu().numpy()
        return normalize(v)
