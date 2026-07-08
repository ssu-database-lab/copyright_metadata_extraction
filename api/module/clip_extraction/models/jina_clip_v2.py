"""
Jina CLIP v2 — 89 languages incl. Korean, 0.9B params, Matryoshka.

WARNING: CC BY-NC 4.0 license. Benchmark-only; cannot ship commercially
without going through Jina's hosted API or marketplaces. Included as a
"strong reference ceiling" — if other Apache-licensed models are close,
prefer those for production.
"""

import numpy as np
import torch
from PIL import Image

from .base import BaseVLM, normalize, softmax


class JinaCLIPv2(BaseVLM):
    name = "Jina CLIP v2"
    hf_id = "jinaai/jina-clip-v2"
    license = "CC BY-NC 4.0 (non-commercial)"
    params = "0.9B"
    korean_support = "native (89 langs)"

    def _load(self) -> None:
        from transformers import AutoModel
        self.model = (
            AutoModel.from_pretrained(self.hf_id, trust_remote_code=True)
            .to(self.device).eval()
        )

    @torch.inference_mode()
    def classify(self, image: Image.Image, candidate_labels: list[str]) -> dict[str, float]:
        img_emb = np.array(self.model.encode_image([image], truncate_dim=512))[0]
        txt_emb = np.array(self.model.encode_text(candidate_labels, truncate_dim=512))
        img_emb = normalize(img_emb)
        txt_emb = normalize(txt_emb)
        logits = txt_emb @ img_emb
        probs = softmax(logits * 100.0)
        return dict(zip(candidate_labels, probs.tolist()))

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        v = np.array(self.model.encode_image([image], truncate_dim=512))[0]
        return normalize(v)

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        v = np.array(self.model.encode_text([text], truncate_dim=512))[0]
        return normalize(v)
