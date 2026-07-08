"""OpenAI CLIP ViT-L/14 — English-only baseline."""

import numpy as np
import torch
from PIL import Image

from .base import BaseVLM, normalize


class OpenAICLIP(BaseVLM):
    name = "OpenAI CLIP ViT-L/14"
    hf_id = "openai/clip-vit-large-patch14"
    license = "MIT"
    params = "428M"
    korean_support = "none"

    def _load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained(self.hf_id)
        self.model = CLIPModel.from_pretrained(self.hf_id).to(self.device).eval()

    @torch.inference_mode()
    def classify(self, image: Image.Image, candidate_labels: list[str]) -> dict[str, float]:
        inputs = self.processor(
            text=candidate_labels,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        out = self.model(**inputs)
        probs = out.logits_per_image.softmax(dim=-1).squeeze(0).cpu().numpy()
        return dict(zip(candidate_labels, probs.tolist()))

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        v = self.model.get_image_features(**inputs).squeeze(0).cpu().numpy()
        return normalize(v)

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        v = self.model.get_text_features(**inputs).squeeze(0).cpu().numpy()
        return normalize(v)
