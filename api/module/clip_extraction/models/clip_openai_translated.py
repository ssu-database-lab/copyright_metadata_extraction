"""
OpenAI CLIP + Korean→English translation layer.

Translates candidate labels (and free-form query text) to English on the
fly using a small NLLB-200 distilled model, then runs OpenAI CLIP. Lets
us measure how much of the multilingual-CLIP gain is *just* translation.
"""

from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from .base import BaseVLM, normalize

_TRANSLATOR_ID = "facebook/nllb-200-distilled-600M"


class OpenAICLIPTranslated(BaseVLM):
    name = "OpenAI CLIP + NLLB translation"
    hf_id = "openai/clip-vit-large-patch14 + facebook/nllb-200-distilled-600M"
    license = "MIT + MIT (NLLB CC BY-NC for some weights — distilled-600M is permissive)"
    params = "428M + 600M"
    korean_support = "via translation"

    def _load(self) -> None:
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            CLIPModel,
            CLIPProcessor,
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = (
            CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            .to(self.device).eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(_TRANSLATOR_ID)
        self.translator = (
            AutoModelForSeq2SeqLM.from_pretrained(_TRANSLATOR_ID)
            .to(self.device).eval()
        )

    @lru_cache(maxsize=512)
    def _translate(self, text: str) -> str:
        # NLLB uses BCP-47 codes — Korean = "kor_Hang", English = "eng_Latn"
        tok = self.tokenizer(text, return_tensors="pt", src_lang="kor_Hang").to(self.device)
        with torch.inference_mode():
            out = self.translator.generate(
                **tok,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_new_tokens=64,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    @torch.inference_mode()
    def classify(self, image: Image.Image, candidate_labels: list[str]) -> dict[str, float]:
        en_labels = [self._translate(lbl) for lbl in candidate_labels]
        inputs = self.processor(
            text=en_labels, images=image, return_tensors="pt", padding=True
        ).to(self.device)
        out = self.model(**inputs)
        probs = out.logits_per_image.softmax(dim=-1).squeeze(0).cpu().numpy()
        return dict(zip(candidate_labels, probs.tolist()))  # report under Korean labels

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        v = self.model.get_image_features(**inputs).squeeze(0).cpu().numpy()
        return normalize(v)

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        en = self._translate(text)
        inputs = self.processor(text=[en], return_tensors="pt", padding=True).to(self.device)
        v = self.model.get_text_features(**inputs).squeeze(0).cpu().numpy()
        return normalize(v)
