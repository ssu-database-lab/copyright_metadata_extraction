"""
sentence-transformers multilingual CLIP — 50+ languages incl. Korean.

Uses a distilled multilingual text encoder paired with the OpenAI
CLIP-B/32 image tower. Same image embedding space as OpenAI CLIP-B/32,
so cross-model image embeddings are NOT compatible with the L/14 models.
"""

import numpy as np
import torch
from PIL import Image

from .base import BaseVLM, normalize, softmax


class MultilingualCLIP(BaseVLM):
    name = "Multilingual CLIP (ViT-B/32 + DistilBERT)"
    hf_id = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    image_hf_id = "sentence-transformers/clip-ViT-B-32"  # paired image tower
    license = "Apache-2.0"
    params = "~280M (text) + ~150M (image)"
    korean_support = "native (50+ langs)"

    def _load(self) -> None:
        # Text tower (multilingual) and image tower are separate models.
        from sentence_transformers import SentenceTransformer
        self.text_model = SentenceTransformer(self.hf_id, device=self.device)
        self.image_model = SentenceTransformer(self.image_hf_id, device=self.device)

    @torch.inference_mode()
    def classify(self, image: Image.Image, candidate_labels: list[str]) -> dict[str, float]:
        img_emb = self.image_model.encode(
            [image], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        txt_emb = self.text_model.encode(
            candidate_labels, convert_to_numpy=True, normalize_embeddings=True
        )
        # Cosine similarity (both already L2-normalized)
        logits = txt_emb @ img_emb
        # Temperature ≈ 100 mimics CLIP's logit scale
        probs = softmax(logits * 100.0)
        return dict(zip(candidate_labels, probs.tolist()))

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        v = self.image_model.encode(
            [image], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return normalize(v)

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        v = self.text_model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return normalize(v)

    def unload(self) -> None:
        for attr in ("text_model", "image_model"):
            if hasattr(self, attr):
                delattr(self, attr)
        super().unload()
