# VLM P1 Baseline — 2026-06-28T09:36:10

- images: 50 | models: Gemma 4 31B (vLLM), Qwen3-VL-235B (DashScope) | manifest labels: 1500 works
- weak-label keyword coverage: 50/50 images have 주제어/해시태그

## Parse + latency
| Model | OK | parsed | avg latency | avg tokens |
|---|---|---|---|---|
| Gemma 4 31B (vLLM) | 50/50 | 50/50 | 7.31s | 241 |
| Qwen3-VL-235B (DashScope) | 50/50 | 50/50 | 5.98s | 279 |

## work_type distribution
- **Gemma 4 31B (vLLM)**: {'사진저작물': 41, '도형저작물': 1, '미술저작물': 7, '어문저작물': 1}
- **Qwen3-VL-235B (DashScope)**: {'사진저작물': 35, '어문저작물': 7, '도형저작물': 1, '미술저작물': 7}

## Gemma-vs-Qwen work_type agreement: **43/50 = 86.0%**

## Keyword set-F1 vs manifest 주제어/해시태그 (weak labels)
- **Gemma 4 31B (vLLM)**: mean F1 0.010 over 50 labeled images
- **Qwen3-VL-235B (DashScope)**: mean F1 0.013 over 50 labeled images