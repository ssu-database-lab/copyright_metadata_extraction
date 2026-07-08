"""
Gemma 4 (E4B/26B/31B) — Google's multimodal LLM, April 2026.

DIFFERENT PARADIGM than CLIP-family: image-in, text-out generative VLM.
For zero-shot classification we prompt it like "What kind of work is
this image? Choose one of: ..." and parse the answer. For embedding-
based similarity it's not directly usable — pair with a separate CLIP
encoder if FAISS is needed.

Default checkpoint: google/gemma-4-e4b (4B effective params, on-device
friendly). Use 26b-a4b for higher accuracy at MoE-priced inference.

Stub kept here so the benchmark can include it once Gemma 4 weights are
mirrored locally (HF auth + model download is the gating step).
"""

import re

import numpy as np
import torch
from PIL import Image

from .base import BaseVLM


class Gemma4VLM(BaseVLM):
    name = "Gemma 4 E4B (VLM)"
    hf_id = "google/gemma-4-e4b"
    license = "Apache-2.0"
    params = "4B"
    korean_support = "native (140+ langs)"

    def _load(self) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                self.hf_id, torch_dtype=torch.bfloat16
            ).to(self.device).eval()
        )

    @torch.inference_mode()
    def classify(self, image: Image.Image, candidate_labels: list[str]) -> dict[str, float]:
        prompt = (
            "다음 이미지가 어떤 유형의 저작물인지 가장 적합한 하나만 골라서 "
            "라벨 이름만 정확하게 답하세요.\n"
            f"선택지: {', '.join(candidate_labels)}\n답:"
        )
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        answer = self.processor.decode(
            out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        ).strip()

        # One-hot the matched label (VLMs don't give calibrated probs without logprobs)
        scores = {lbl: 0.0 for lbl in candidate_labels}
        for lbl in candidate_labels:
            if re.search(re.escape(lbl), answer):
                scores[lbl] = 1.0
                break
        else:
            # Nothing matched — split evenly so downstream code doesn't crash
            for lbl in candidate_labels:
                scores[lbl] = 1.0 / len(candidate_labels)
        return scores

    def encode_image(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError(
            "Gemma 4 is a generative VLM, not an embedding model. "
            "Pair it with a CLIP/SigLIP encoder for FAISS-style similarity."
        )

    def encode_text(self, text: str) -> np.ndarray:
        raise NotImplementedError(
            "Gemma 4 is a generative VLM, not an embedding model."
        )
