# CLIP / Multimodal Extraction (Year 2 prototype)

Year 2 of the 공유저작물 글로벌 project requires a CLIP-based multimodal
metadata extractor (proposal §2-2 task 4.1, milestone 4.1 at page 32+42).
This directory holds the **model-selection benchmark** that runs first —
we pick the production model based on measured accuracy/latency/license
trade-offs, *then* wire it into the main pipeline.

## What's in here

```
clip_extraction/
├── README.md               ← this file
├── requirements.txt        ← extra deps (transformers≥4.49, etc.)
├── labels.py               ← Korean candidate label vocabularies
├── benchmark.py            ← runner: loads each model, writes report
├── fetch_samples.py        ← download a small public-domain test set
├── models/
│   ├── base.py             ← common interface (BaseVLM)
│   ├── clip_openai.py      ← OpenAI CLIP ViT-L/14   (baseline)
│   ├── clip_openai_translated.py  ← + NLLB Korean translator
│   ├── koclip_bingsu.py    ← Bingsu/clip-vit-large-patch14-ko
│   ├── multilingual_clip.py ← sentence-transformers multilingual CLIP
│   ├── siglip2.py          ← Google SigLIP 2 SO400M/14-384
│   ├── jina_clip_v2.py     ← Jina CLIP v2  (CC BY-NC — non-commercial!)
│   └── gemma4_vlm.py       ← Gemma 4 generative VLM (stub, opt-in)
├── test_data/
│   ├── README.md           ← naming convention for ground-truth scoring
│   └── sample_works/       ← drop images here
└── reports/                ← benchmark output (JSON + Markdown)
```

## Quick start

```bash
# 1. Install extra deps (this upgrades transformers to 4.49+ for SigLIP 2)
pip install -r api/module/clip_extraction/requirements.txt

# 2. Fetch a tiny public-domain test set (or drop your own into test_data/sample_works/)
python -m api.module.clip_extraction.fetch_samples

# 3. Run the full benchmark
python -m api.module.clip_extraction.benchmark --labels-from-filename

# 4. View the report
ls api/module/clip_extraction/reports/
# benchmark_20260428_103045.md   ← human-readable comparison table
# benchmark_20260428_103045.json ← machine-readable, full per-image scores
```

## Running a subset

```bash
# Test just two models:
python -m api.module.clip_extraction.benchmark \
    --models openai-clip-vit-l14,siglip2-so400m-patch14 \
    --labels-from-filename

# Use the "subject" label set instead of "work_type":
python -m api.module.clip_extraction.benchmark \
    --label-set subject \
    --labels-from-filename
```

## Model registry

| Key | License | Korean | Note |
|---|---|---|---|
| `openai-clip-vit-l14` | MIT | none | Baseline, English-only |
| `openai-clip-vit-l14+translate` | MIT | via NLLB | Tests if translation alone suffices |
| `koclip-bingsu-vit-l14` | MIT | native | Korean knowledge-distilled CLIP |
| `multilingual-clip-vit-b32` | Apache-2.0 | native (50+) | Older but well-tested |
| `siglip2-so400m-patch14` | Apache-2.0 | native (109) | SOTA Feb 2025 |
| `jina-clip-v2` | **CC BY-NC** | native (89) | Strong but non-commercial — flag |
| `gemma4-e4b` (deferred) | Apache-2.0 | native (140+) | Generative VLM, different paradigm |

Edit `models/__init__.py` REGISTRY to add/remove entries.

## Why this benchmark exists

The proposal Year 1 deliverable for 숭실대 task 4.3 is *"CLIP 적합성 모의
테스트"* — a feasibility study. The numbers this benchmark produces become
the evidence for the Year 1 연구보고서, and the basis for choosing the
Year 2 production model (milestone 4.1).

## Adding a new model

1. Create `models/your_model.py` extending `BaseVLM`.
2. Implement `_load`, `classify`, `encode_image`, `encode_text`.
3. Register it in `models/__init__.py`.
4. Re-run `benchmark.py` — your model appears in the next report.

## Cost considerations

All models in the registry are open-weight (downloaded from HF). No
external API spend during the benchmark. GPU is optional but strongly
recommended for SigLIP 2 SO400M and Jina CLIP v2 (each ~3-4 GB VRAM).
On CPU, allow ~30 sec/image for the larger models.
