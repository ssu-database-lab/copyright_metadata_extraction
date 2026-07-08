"""
Model registry. Each entry is a (key, factory) pair.

The benchmark walks this dict in order, loads each, runs the test set,
unloads, and writes a report. Add or comment-out lines to control which
models run.
"""

from .base import BaseVLM
from .clip_openai import OpenAICLIP
from .clip_openai_translated import OpenAICLIPTranslated
from .koclip_bingsu import KoCLIPBingsu
from .multilingual_clip import MultilingualCLIP
from .siglip2 import SigLIP2
from .jina_clip_v2 import JinaCLIPv2
from .gemma4_vlm import Gemma4VLM

# Order matters: small/fast models first so the benchmark fails-fast on the
# easy ones before spending time on the heavy ones.
REGISTRY: dict[str, type[BaseVLM]] = {
    "openai-clip-vit-l14":      OpenAICLIP,
    "openai-clip-vit-l14+translate": OpenAICLIPTranslated,
    "koclip-bingsu-vit-l14":    KoCLIPBingsu,
    "multilingual-clip-vit-b32": MultilingualCLIP,
    "siglip2-so400m-patch14":   SigLIP2,
    "jina-clip-v2":             JinaCLIPv2,
    # Deferred — Gemma 4 is a VLM (image-in, text-out), not an embedding
    # model. Different paradigm; benchmark separately. Uncomment when ready.
    # "gemma4-e4b":               Gemma4VLM,
}

__all__ = ["REGISTRY", "BaseVLM"]
