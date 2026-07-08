# Multimodal Model Tests — Consolidated Results (side-by-side)

**Project:** AI-based Public-Domain Work Analysis (MCST/KOCCA R&D) — Soongsil University
**Compiled:** 2026-06-08 · all numbers pulled from the raw JSON reports in `api/module/clip_extraction/reports/`. Tested images: `docs/test_images/`.

## Experiment index

| # | Experiment | Models | Images | Raw report |
|---|---|---|---|---|
| A | CLIP zero-shot type classification | multilingual-CLIP | 8 (all 사진) | `benchmark_20260518_161102` |
| B | VLM attribute extraction | Gemma 4 vs Qwen3-VL | 8 (all 사진) | `vlm_compare_20260524_012605` |
| C | Embedding duplicate-detection | 4 CLIP-family | 8 | `embed_benchmark_20260526_124650` |
| D | Embedding duplicate-detection | 3 CLIP-family | 80 diverse | `embed_benchmark_20260601_124525` |
| E | VLM attribute extraction | Qwen3-VL | 12 diverse | `vlm_compare_20260601_124517` |
| F | VLM Gemma vs Qwen (KO prompt) | Gemma 4 vs Qwen3-VL | 15 diverse | `vlm_compare_20260608_131551` |
| G | VLM Gemma vs Qwen (EN prompt) | Gemma 4 vs Qwen3-VL | 15 diverse | `vlm_compare_20260608_134205` |

## Experiment A — CLIP zero-shot type classification

- Model: `multilingual-clip-vit-b32` · label set: work_type · 8 images (all 사진저작물)
- **Top-1 accuracy: 50%** — CLIP classifies by subject not medium (structural limit).

| image | ground truth | CLIP top-1 | correct |
|---|---|---|---|
| 사진저작물__197.국물떡볶이.jpg | 사진저작물 | 사진저작물 | ✓ |
| 사진저작물__2026. 2월-문화가-있는-날.jpg | 사진저작물 | 연극저작물 | ✗ |
| 사진저작물__사진갤러리_1106_제주 감귤.jpg | 사진저작물 | 사진저작물 | ✓ |
| 사진저작물__엑스포1.jpg | 사진저작물 | 컴퓨터프로그램저작물 | ✗ |
| 사진저작물__완도 해조류센터.jpg | 사진저작물 | 건축저작물 | ✗ |
| 사진저작물__참다래 '해금'.jpg | 사진저작물 | 사진저작물 | ✓ |
| 사진저작물__포토갤러리_0325_[생태달력] 동물 -  | 사진저작물 | 사진저작물 | ✓ |
| 사진저작물__한국민속예술제사진5.jpg | 사진저작물 | 연극저작물 | ✗ |

## Experiment B — VLM Gemma vs Qwen (8 images, all 사진)

Both 6/8 medium-based agreement. Gemma better OCR (WANDO/66TH); Qwen misread (ANO/60th) + language bleed.

| model | ok | parsed | avg latency | avg tokens |
|---|---|---|---|---|
| Gemma 4 31B (vLLM) | 8/8 | 8 | 7.47s | 276 |
| Qwen3-VL-235B (DashScope) | 8/8 | 8 | 6.08s | 312 |

**work_type per image:**

| image | Gemma | Qwen3-VL-235B | agree |
|---|---|---|---|
| 사진저작물__197.국물떡볶이.jpg | 영상저작물 | 영상저작물 | ✓ |
| 사진저작물__2026. 2월-문화가-있는-날.j | 미술저작물 | 도형저작물 | ✗ |
| 사진저작물__사진갤러리_1106_제주 감귤.jp | 사진저작물 | 사진저작물 | ✓ |
| 사진저작물__엑스포1.jpg | 사진저작물 | 사진저작물 | ✓ |
| 사진저작물__완도 해조류센터.jpg | 사진저작물 | 사진저작물 | ✓ |
| 사진저작물__참다래 '해금'.jpg | 사진저작물 | 사진저작물 | ✓ |
| 사진저작물__포토갤러리_0325_[생태달력] 동 | 사진저작물 | 사진저작물 | ✓ |
| 사진저작물__한국민속예술제사진5.jpg | 사진저작물 | 사진저작물 | ✓ |

## Experiment C — Embedding duplicate-detection (8 images)

All 100% — easy set, can't differentiate. KoCLIP ≡ OpenAI (same image tower).

| model | dim | dup rank-1 | true sim | distractor sim | margin | embed/img |
|---|---|---|---|---|---|---|
| siglip2-so400m-patch14 | 1152 | 100.0% | 0.9835 | 0.4878 | 0.4957 | 0.795s |
| koclip-bingsu-vit-l14 | 768 | 100.0% | 0.9812 | 0.4499 | 0.5313 | 0.54s |
| multilingual-clip-vit-b32 | 512 | 100.0% | 0.9768 | 0.4274 | 0.5494 | 0.26s |
| openai-clip-vit-l14 | 768 | 100.0% | 0.9812 | 0.4499 | 0.5313 | 0.499s |

## Experiment D — Embedding duplicate-detection (80 diverse)

~65% all tied — NOT model weakness: KOGL has byte-identical thumbnails under different IDs (verified). Pick on efficiency → multilingual-CLIP (smallest/fastest).

| model | dim | dup rank-1 | true sim | distractor sim | margin | embed/img |
|---|---|---|---|---|---|---|
| siglip2-so400m-patch14 | 1152 | 64.7% | 0.9707 | 0.5385 | 0.4322 | 0.116s |
| koclip-bingsu-vit-l14 | 768 | 65.0% | 0.9679 | 0.5485 | 0.4194 | 0.028s |
| multilingual-clip-vit-b32 | 512 | 64.7% | 0.9731 | 0.5348 | 0.4383 | 0.015s |

## Experiment E — VLM Qwen on 12 diverse works

8/8 illustrations → 미술저작물; Qwen more accurate than KOGL label on 가례증해→어문, 만자→도형.

| model | ok | parsed | avg latency | avg tokens |
|---|---|---|---|---|
| Qwen3-VL-235B (DashScope) | 12/12 | 12 | 5.95s | 277 |

**work_type per image:**

| image | Qwen3-VL-235B |
|---|---|
| 2324.jpg | 사진저작물 |
| 63323.jpg | 미술저작물 |
| 63342.jpg | 미술저작물 |
| 63490.jpg | 미술저작물 |
| 63678.jpg | 미술저작물 |
| 63685.jpg | 미술저작물 |
| 63688.jpg | 미술저작물 |
| 63693.jpg | 미술저작물 |
| 63699.jpg | 미술저작물 |
| 67096.jpg | 어문저작물 |
| 70757.jpg | 도형저작물 |
| 74418.jpg | 사진저작물 |

## Experiments F & G — Gemma vs Qwen, KO prompt vs EN prompt (15 diverse)

Same 15 images, prompt language is the only change (work_type rules identical).

| image | Gemma KO | Gemma EN | Qwen KO | Qwen EN |
|---|---|---|---|---|
| 113140.jpg (어문) | 미술저작물 | 도형저작물 | 도형저작물 | 도형저작물 |
| 122108.jpg (문서) | 도형저작물 | 도형저작물 | 도형저작물 | 도형저작물 |
| 2324.jpg (사진) | 사진저작물 | 사진저작물 | 사진저작물 | 사진저작물 |
| 63323.jpg (미술) | 미술저작물 | 미술저작물 | 미술저작물 | 미술저작물 |
| 63342.jpg (미술) | 미술저작물 | 미술저작물 | 미술저작물 | 미술저작물 |
| 63490.jpg (미술) | 미술저작물 | 미술저작물 | 미술저작물 | 미술저작물 |
| 63678.jpg (미술) | 미술저작물 | 미술저작물 | 미술저작물 | 미술저작물 |
| 64794.jpg (어문) | 미술저작물 | PARSE✗ | 어문저작물 | 어문저작물 |
| 67096.jpg (사진) | 사진저작물 | 어문저작물 | 어문저작물 | 어문저작물 |
| 70757.jpg (사진) | 도형저작물 | 미술저작물 | 도형저작물 | 도형저작물 |
| 74418.jpg (사진) | 사진저작물 | 사진저작물 | 사진저작물 | 사진저작물 |
| 78079.jpg (사진) | 사진저작물 | 사진저작물 | 어문저작물 | 어문저작물 |
| 97530.jpg (문서) | 미술저작물 | 도형저작물 | 도형저작물 | 도형저작물 |
| 98400.jpg () | 도형저작물 | 도형저작물 | 도형저작물 | 도형저작물 |
| 99867.jpg (어문) | 미술저작물 | 미술저작물 | 도형저작물 | 도형저작물 |

**Verdict:** Qwen 0/15 changed (robust); Gemma 5/15 changed but mixed (1 parse-fail, 1 regression, slower). **Korean prompt kept as default.** Logo→도형/미술 & text→어문/사진 conventions deferred to KOGL 구분 명세서.

---
## Conclusions

- **Type classification:** generative VLM (medium-based) >> CLIP zero-shot (subject-based, ~50%).
- **VLM choice:** Gemma 4 production (OCR/cost/privacy), Qwen fallback — complementary; reconcile for `work_type`.
- **Embedding/FAISS:** all CLIP-family tie on dedup → **multilingual-CLIP** (cheapest 512-dim, fastest).
- **Prompt language:** Korean default (English A/B gave no clear win).
- **Data caveats (verified):** KOGL 정보유형 labels coarse (~all 사진 for images); KOGL has duplicate thumbnails; many thumbnails are logos/representative images, not the work → derive work_type from file medium.

_Images: `docs/test_images/` (INDEX.md). Raw reports: `api/module/clip_extraction/reports/`._