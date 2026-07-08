# Phase 2 Multimodal Model Experiments — Results Report

**Project:** AI-based Public-Domain Work Analysis & Type-Determination System (MCST / KOCCA R&D) — Soongsil University
**Scope:** Phase 2 image·text multimodal analysis — model selection experiments
**Date:** 2026-05-29 · **Test data:** 8 real KOGL (공공누리) work images (all 사진저작물 / photographic-work type)

---

## 0. One-line summary

> **Empirically showed CLIP-family models are unsuitable for type *classification* (~50%), and solved the problem by switching to a generative VLM.**
> Architecture direction fixed as a **hybrid**: **attribute extraction = VLM (Gemma/Qwen), similarity·duplicate-detection = embeddings (SigLIP/CLIP).**

---

## 1. Background & objective

- Phase 2 goal: automatic **attribute extraction (type · description · keywords)** from image/text public-domain works + **similarity·duplicate detection**.
- The proposal specifies **CLIP-based multimodal** → so we first empirically validated CLIP's suitability.
- Three experiments:
  1. **Experiment A** — CLIP zero-shot type-classification suitability
  2. **Experiment B** — generative-VLM attribute extraction comparison (Gemma 4 vs Qwen3-VL)
  3. **Experiment C** — embedding-model duplicate-detection (FAISS) performance

---

## 2. Experiment A — CLIP zero-shot type classification (suitability test)

**Method:** Zero-shot classification of 8 KOGL images against 10 work-type labels (photo/video/text/art/architecture/diagram…) using multilingual-CLIP. Korean labels, with/without a hypothesis template.

**Result: 50% accuracy (4/8)** — unchanged at 50% even with the template.


| Image                     | Ground truth  | CLIP prediction          | OK  |
| ------------------------- | ------------- | ------------------------ | --- |
| Tteokbokki                | 사진저작물 (photo) | 사진저작물                    | ✓   |
| Jeju tangerine            | 사진저작물         | 사진저작물                    | ✓   |
| White-naped crane         | 사진저작물         | 사진저작물                    | ✓   |
| Korean Folk Arts Festival | 사진저작물         | 사진저작물                    | ✓   |
| Wando Seaweed Center      | 사진저작물         | **건축저작물 (architecture)** | ✗   |
| Expo booth                | 사진저작물         | **건축저작물**                | ✗   |
| "Culture Day" poster      | 사진저작물         | **연극저작물 (theatrical)**   | ✗   |
| Kiwi 'Haegeum'            | 사진저작물         | **도형저작물 (diagram)**      | ✗   |


**Key finding (root cause):**

- CLIP classifies by **subject**, not by **medium**.
- E.g., a building photographed with a camera → CLIP sees a building and mislabels it "architectural work." But by copyright law the medium is "photographic work."
- This is a structural limit of CLIP — it's an **embedding-similarity model** that only measures visual similarity to fixed labels. The hypothesis template did not help (50%→50%).

**→ Conclusion: CLIP is unsuitable for type *classification*. You cannot *instruct* it on the classification criterion (medium vs subject).**

---

## 3. Experiment B — generative-VLM attribute extraction (Gemma 4 vs Qwen3-VL)

**Method:** Same 8 images, **same prompt** (JSON output, explicitly "classify by medium") → compare two VLMs.

- Gemma 4 31B (local vLLM server) vs Qwen3-VL-235B (Alibaba DashScope)

**Summary:**


| Model         | Processed | JSON parsed | Avg latency | Avg tokens | Type agreement |
| ------------- | --------- | ----------- | ----------- | ---------- | -------------- |
| Gemma 4 31B   | 8/8       | 8/8         | 7.47s       | 276        | 6/8            |
| Qwen3-VL-235B | 8/8       | 8/8         | 6.08s       | 312        | 6/8            |


**Key findings:**

1. **VLM solves the medium-based classification problem.** With only the instruction "classify by medium," it correctly labeled the Wando Seaweed Center (a building photo, which CLIP got wrong) as 사진저작물 — with reasoning: *"a still image of a real location captured with a camera."*
2. **Rich Korean description + keywords + objects/colors** generated automatically.
3. **Simultaneous OCR of in-image text** (poster film details, "2022 Wando Int'l Seaweed Expo," "66th KOREAN FOLK ARTS FESTIVAL," etc.).
4. **AI-generated-content detection** 🎯 — both models read the tteokbokki image's *"This video was created with the help of generative AI"* caption and judged it 영상저작물 (AI-generated) → directly relevant to the project's AI-generated-content detection requirement.

**Differences between models:**


| Aspect         | Gemma 4 31B                                | Qwen3-VL-235B                                    |
| -------------- | ------------------------------------------ | ------------------------------------------------ |
| OCR accuracy   | **"WANDO", "66TH" correct**                | "ANO", "60th" **misread**                        |
| Language bleed | none (clean Korean)                        | "테ント" (Japanese), "cylindric" (English) bleed-in |
| Hallucination  | none                                       | "drone photo" keyword hallucinated (1 case)      |
| Deployment     | **local (zero cost, data stays internal)** | cloud API (per-call billing, data leaves)        |
| License        | Apache-2.0                                 | commercial API                                   |


**→ Conclusion (provisional): Gemma 4 31B = recommended for production (cost·privacy·OCR edge), Qwen3-VL = fallback.**
**※ Gemma server is temporarily unavailable → Qwen3-VL is the current default model, with automatic fallback logic implemented.**

---

## 4. Experiment C — embedding-model duplicate detection (FAISS role)

**Method:** Evaluate CLIP/SigLIP as **embedding models, not classifiers** (for the duplicate-detection KPI 89→92%).
8 images × 4 near-duplicate perturbations (resize/recompress/crop/brightness) = 32 duplicate queries → measure rank-1 accuracy of retrieving the source.

**Result:**


| Model                  | License    | Dim  | Dup rank-1 | True-pair sim | Distractor sim | Margin | Embed speed |
| ---------------------- | ---------- | ---- | ---------- | ------------- | -------------- | ------ | ----------- |
| **SigLIP 2** (so400m)  | Apache-2.0 | 1152 | **100%**   | 0.984         | 0.488          | 0.496  | 0.80s       |
| **KoCLIP** (Bingsu)    | MIT        | 768  | **100%**   | 0.981         | 0.450          | 0.531  | 0.54s       |
| **multilingual-CLIP**  | Apache-2.0 | 512  | **100%**   | 0.977         | 0.427          | 0.549  | 0.26s       |
| **OpenAI CLIP** (L/14) | MIT        | 768  | **100%**   | 0.981         | 0.450          | 0.531  | 0.50s       |


All models scored rank-1 100% across all 4 perturbation types (resize/recompress/crop/brightness all 1.0).

**Key findings:**

1. **All models are suitable for duplicate detection** (true pairs ~0.98 vs distractors ~0.45 — clear separation).
2. **Small, single-type data cannot differentiate the models** → final selection requires large, diverse KOGL data.
3. KoCLIP ≡ OpenAI CLIP (same image tower) → identical for image-only duplicate detection; KoCLIP only adds value for Korean text-image matching.
4. **Differentiation candidates:** SigLIP 2 = newest·multilingual·highest dimension (precise); multilingual-CLIP = smallest dimension·fastest (indexing-cost advantage).

---

## 5. Overall conclusion — hybrid architecture

```
Public-domain work file
   ├─ image/audio/video → VLM (Gemma/Qwen) ──→ description·type·keywords (Exp B)
   ├─ image             → SigLIP/CLIP embedding → FAISS → similarity·duplicate detection (Exp C)
   └─ text/contract     → OCR + LLM + NER  ──→ rights info (Phase-1 pipeline)
                                    └→ unified schema (20 elements)
```


| Task                                             | Adopted technology               | Rationale                                                           |
| ------------------------------------------------ | -------------------------------- | ------------------------------------------------------------------- |
| Attribute extraction (description·type·keywords) | **generative VLM** (Gemma/Qwen)  | medium-based classification possible, no training needed (Exp A, B) |
| Similarity·duplicate detection                   | **embedding + FAISS** (SigLIP 2) | the native strength of embeddings (Exp C)                           |
| Rights info (text/contracts)                     | **OCR+LLM+NER** (existing)       | Phase-1 pipeline                                                    |


**Key message:** The initial plan to use CLIP for "classification" was empirically revised into a **"classify with VLM, find similarity with CLIP/SigLIP"** hybrid — placing each technology where its real strength lies.

---

## 5.5 Additional experiment — large·diverse-data validation (KOGL 80 / 12 works)

To address the 8-image (all-photo) limitation, we resampled diverse types from real KOGL data and re-validated. **Confirmed that the public thumbnail URLs are fetchable for all 144k works** (usable immediately).

### (1) Embedding duplicate detection — 80 diverse images


| Model             | Dim  | Dup rank-1 | Margin    | Embed speed |
| ----------------- | ---- | ---------- | --------- | ----------- |
| SigLIP 2          | 1152 | 64.7%      | 0.432     | 0.116s      |
| KoCLIP            | 768  | 65.0%      | 0.419     | 0.028s      |
| multilingual-CLIP | 512  | **64.7%**  | **0.438** | **0.015s**  |


- The three models are **statistically tied** (~65%). rank-1 dropping 100%→65% is **not** a model weakness but because **KOGL has identical images registered under different IDs/titles** (verified: four National Theater works — 꿈하늘/들오리/말괄량이/팔곡병풍 — share a **byte-identical thumbnail**, matching SHA256). → This is a real data-quality issue directly tied to the project's **duplicate-detection / false-registration-prevention** goal.
- Since accuracy is tied, **select on efficiency: multilingual-CLIP** (smallest 512-dim = minimal FAISS index, fastest). SigLIP 2's higher dimension did not translate into an accuracy gain.

### (2) VLM attribute extraction — 12 diverse works (Qwen3-VL)

- **8 illustrations (Daegu landmarks) → all correctly classified as 미술저작물 (artwork)** (reasoning correctly identifies "line-drawing technique," "watercolor," "illustration").
- **4 photos:** building photos → 사진저작물 ✓ / and **cases where the VLM was *more* accurate than the KOGL label** — a Chinese-character classical text (가례증해, KOGL=photo) → **어문저작물 (literary work)**, a geometric-pattern chart (만자의 여러 유형, KOGL=photo) → **도형저작물 (diagram work)**.
- **Implications:** ① VLM medium-based classification holds on diverse real data; ② **KOGL's 정보유형 label is coarse** (documents/diagrams stored as images are tagged "photo") → unsuitable as type ground-truth; work_type is better derived from the original file format.

### (3) Gemma vs Qwen detailed comparison — 15 diverse works (illustrations·photos·document scans·logos)

After restoring the Gemma server (permanent Tailscale link), both models were re-compared on the same 15 works. **15/15 processed, 9/15 (60%) work_type agreement.** Latency: Gemma 7.44s / Qwen 6.68s, tokens ~equal.

- **The 6 disagreements are mostly NOT errors** — two patterns:
  - **① Logo thumbnails (3)**: the press-release / deck / news thumbnails are actually **logos**. **Gemma consistently → 미술저작물 (art), Qwen consistently → 도형저작물 (figure/graphic)** → logo classification is a **labeling convention KOGL must decide**.
  - **② Text-document scans (3)**: ICT report, classical text (가례증해), newspaper (제국신문) → **Qwen classifies the medium more accurately (어문, literary)**, while Gemma over-calls 미술/사진.
- **Complementary strengths:** Gemma = better photo OCR + cleaner Korean / Qwen = better document·text medium classification.
- ⚠ **Caution (verified):** both models confidently **misidentified the 제국신문 first issue as 독립신문** — VLMs can be confidently wrong on specific historical artifacts.
- ⚠ **Reconfirmed:** many KOGL thumbnails are **logos / representative images, not the work** → **work_type must be derived from the actual file/medium, not from classifying the thumbnail.**

> Conclusion: Gemma stays the production recommendation (OCR·cost·privacy), but the gap is small and the two are complementary. **Consider reconciling both models (consolidation) for the type field.** (Source: `reports/vlm_compare_20260608_131551.md`)

---

## 6. Next steps

- **Build a large evaluation set** — balanced per-type sample from the 144k KOGL works (77k images, 29k with type labels) → large-scale Gemma/Qwen evaluation + final embedding-model selection.
- **Train a type classifier** — SigLIP embedding + lightweight classification head (using KOGL labels, to satisfy the official-certification KPI).
- **Pipeline integration** — modality router + unified-schema mapping + **VLM automatic fallback**.
- **Official certification (KTC/TTA)** — verify attribute-info accuracy 80→85% (KOGL 1,000 works per type).

---

## Appendix — reproducibility

- Code: `api/module/clip_extraction/` (benchmarks + `vlm/` VLM comparison)
- Raw results:
  - Experiment A: `reports/benchmark_20260518_*.md`
  - Experiment B: `reports/vlm_compare_20260524_012605.md`
  - Experiment C: `reports/embed_benchmark_20260526_124650.md`
  - Additional (diverse): `reports/embed_benchmark_20260601_*.md`, `reports/vlm_compare_20260601_*.md`
- Test environment: NVIDIA RTX 4070 SUPER (local benchmarks), Gemma 4 31B (RTX PRO 6000 ×2, remote vLLM), Qwen3-VL-235B (Alibaba DashScope)
- ⚠ Experiments A–C are **directional validation** on **8 KOGL images (all photographic works)**; statistical confirmation will be done on the large evaluation set (§5.5 begins this).

