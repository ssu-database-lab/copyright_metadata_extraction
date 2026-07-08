"""
VLM (generative vision-language model) attribute extraction.

Unlike the CLIP-family benchmark (embedding models, fixed labels), this
compares *generative* VLMs that describe image content open-vocabulary and
extract structured metadata via prompting.

Used to choose the production VLM for description / attribute extraction:
- Gemma 4 31B   — local vLLM server (OpenAI-compatible)
- Qwen3-VL-235B — Alibaba DashScope (OpenAI-compatible, already in OCR pipeline)
"""
