# Model Evaluation & Data-Source Investigation — Consolidated Report

**Date:** 2026-07-23 · **Author:** Soongsil Univ. DB Lab
**Scope:** (A) Whether new models (Qwen3.6/3.7 etc.) are worth swapping into the pipeline, (B) sourcing GOLD descriptions for VLM evaluation (KOGL detail pages · 공유마당), and (C) the evaluation infrastructure (code) built for this, with measured results, conclusions, and recommendations.

> This single document lets you understand "what we did, what we tested, and the results/conclusions/recommendations." Detailed evidence is in each section's referenced files.

---

## 0. Bottom line at a glance (recommendation by role)

| Pipeline role | Current model | Verdict | Recommendation |
|---|---|---|---|
| **LLM metadata extraction** | qwen3.5-122b-a10b | **Keep** | Little to gain from switching. "Invalid JSON" is not a model problem but a lack of schema-constrained decoding → the real fix is decoding/validation hardening. If cost-driven, pilot qwen3.7-plus. |
| **Consolidation arbiter** | qwen3.5-122b-a10b (fb qwen3.5-plus) | **Worth reviewing (highest value)** | Reasoning-intensive → **pilot qwen3.7-plus** (maintains quality, −38% output cost). Use qwen3.7-max only if top-tier quality is empirically justified (expensive). |
| **OCR** | qwen3-vl-235b-a22b-instruct | **Keep (conditional)** | On clean printed contracts it's on par with the new models (~98%). **qwen3.6-35b-a3b** matches accuracy at 2.6× the speed → a swap candidate *if* it passes a degraded-scan robustness test. |
| **Image attributes (VLM, works)** | Gemma 4 31B (OpenRouter) → qwen3-vl-235b | **Keep Gemma** | Only Gemma recognizes a Korean hanok correctly (every Qwen model mislabels it "Japanese-style"). The Qwen fallback can move to qwen3.6-35b-a3b (cheaper, equal). |

**One-line conclusion:** *Keep extraction & OCR as-is for safety; only the consolidation/arbiter step is worth a qwen3.7-plus pilot.* The new flagships (qwen3.7-max, qwen3.6-max-preview) **cannot do vision (text-only)**, so they are not OCR/VLM replacement candidates at all.

---

## 1. Accessible models (Alibaba DashScope workspace key)

**149 models total** accessible with the key. Pipeline-relevant candidates:

| Role | Current | Accessible new candidates |
|---|---|---|
| Text (extract/consolidate) | qwen3.5-122b-a10b | qwen3.6-plus, qwen3.6-max-preview, qwen3.6-35b-a3b, qwen3.6-27b, qwen3.6-flash, **qwen3.7-max**, **qwen3.7-plus**, glm-5.1, glm-5.2 |
| Vision (OCR/image) | qwen3-vl-235b-a22b-instruct | qwen3-vl-235b-a22b-**thinking**, qwen3-vl-plus/flash (there is **no** 3.6/3.7-VL) |

**Key point:** qwen3.6-VL / qwen3.7-VL **do not exist** (not on our key; "3.6-VL" claims come only from content-farm pages). The official vision line is still **Qwen3-VL**.

---

## 2. Multimodal-capability test (the key twist)

Sent the same image to each model directly to test whether it accepts image input:

| Model | Accepts images | Note |
|---|---|---|
| qwen3-vl-235b-a22b-instruct (current OCR) | ✅ | OCR-specialized |
| **qwen3.7-max** (flagship) | ❌ **text-only** | HTTP 400 on image input |
| **qwen3.6-max-preview** (flagship) | ❌ **text-only** | HTTP 400 on image input |
| qwen3.6-plus | ✅ | general multimodal (not OCR-tuned) |
| qwen3.7-plus | ✅ | multimodal but slow (17s+/image) |
| qwen3.6-35b-a3b | ✅ | general multimodal, fast/cheap |
| qwen3.5-122b-a10b (current extractor) | ✅ | (note) current extractor also accepts images |

→ **The smartest flagships (3.7-max, 3.6-max-preview) can't do OCR at all.** So for "can a new model replace qwen3-vl?", the flagships aren't even candidates — only the multimodal plus/mid-tier models are.

---

## 3. OCR test on a real scanned contract

- **Test document:** `data/pdf/계약서/20251110_133106_992874/샘플_저작물..._7.저작물양도계약서.pdf` (4 pages, scan with no text layer, rendered at 200 DPI).
- **Scoring:** page 1 human-transcribed as GROUND TRUTH → character similarity (difflib) + recall of 15 anchors (title, dates, org names, the `㈜` symbol, address, etc.) + latency.

| Model | Char accuracy | Anchors | Latency/page | Notable |
|---|---|---|---|---|
| qwen3-vl-235b (current) | 97.8% | 14/15 | 13s | dropped the `㈜` symbol |
| **qwen3.6-plus** | **98.7%** | **15/15** | 24s | best accuracy (got `㈜`) |
| **qwen3.6-35b-a3b** | 97.8% | 14/15 | **5s** | **same accuracy as current, 2.6× faster** |
| qwen3.7-plus | 98.1% | 15/15 | 58s | accurate but far too slow for bulk |

**Conclusion:** on clean printed Korean contracts all four are excellent (~98%); differences are marginal. The `㈜` corporate symbol is the one differentiator (only 3.6-plus/3.7-plus get it). **qwen3.6-35b-a3b matches current accuracy while being much faster and cheaper** → attractive for 144K-scale bulk OCR.

**⚠ Limitation (important):** the above is **one clean printed page only**. The reason qwen3-vl-235b is a "specialist" is robustness on **degraded inputs** (low-light, blur, skew, stamps, handwriting, complex tables) — this test does not verify that. **Before any swap, re-test on a degraded/handwritten scan set.**

- Artifacts: `<scratch>/ocrtest/` (rendered images + each model's OCR output in `ocr_outputs.json`).

---

## 4. Image-description (VLM) quality test

- **Test:** generate a 3–4 sentence Korean description for 2 KOGL gold images (Suwon Hwaseong, Garden of Morning Calm) → score subject-consistency + descriptive-quality (1–5) with a qwen-max judge against the GOLD (subject background text).

| Model | subject(1-5) | quality(1-5) | Latency |
|---|---|---|---|
| qwen3-vl-235b (current) | 1.5 | 4.5 | 4s |
| qwen3.6-plus | 2.0 | 5.0 | 6s |
| qwen3.7-plus | 2.5 | 4.0 | 17s |
| qwen3.6-35b-a3b | 2.5 | 4.0 | 4s |
| **Gemma 4 31B (current primary)** | **correct** | high | (OpenRouter) |

**Key finding:** **every Qwen model (including the current qwen3-vl) mislabeled a traditional Korean hanok as a "Japanese-style building."** **Only Gemma 4 identified it correctly as a Korean house.** → This reconfirms that **keeping Gemma primary** for image-attribute extraction is justified. The Qwen fallback can move to qwen3.6-35b-a3b (fast, cheap, equal quality).

---

## 5. Text models (web benchmark research + cost)

> ⚠ **Reliability caveat:** all Qwen3.6/3.7·GLM-5.2 numbers are third-party and official cross-checking partially failed (no Qwen3.6 post on the official blog; sources disagree on release dates, context length, intelligence index). **Korean-specific benchmarks, IFEval, and JSON-reliability figures essentially don't exist publicly.** Treat the below as directional only — do not cite as fact; confirm with our own A/B test.

**Research summary:**
- qwen3.7-max = current Qwen **text flagship** (GPQA 92.4, multilingual WMT24++ 85.8 #1). **Text-only.**
- qwen3.7-plus = **IFEval 94.6 (edges max's 94.3)**, near-flagship reasoning, **~half the output cost**. Multimodal.
- qwen3.6-max-preview = coding-focused (tops 6 coding benches), overall behind 3.7-max. **preview = do not use in production** (non-reproducible, forced migration).
- GLM-5.2 = MIT open-weight/self-hostable, top intelligence but **weaker multilingual (83.1 vs Qwen 87)** → 2nd choice for Korean, only if on-prem PII is required.
- **Why keep extraction:** Qwen3.5 family ranks #1 open on Structured Output value-accuracy (**0.801**). "Invalid JSON" is caused by not applying schema-constrained decoding, not by the model.

**Cost (≈144K works, one full pass, text leg only; assuming ~12K input / ~5K output tokens per doc):**

| Model | in $/1M | out $/1M | 144K pass | vs current |
|---|---|---|---|---|
| qwen3.6-flash | 0.25 | 1.50 | ~$1,512 | −50% |
| **qwen3.7-plus** | 0.40 | **1.60** | **~$1,843** | **−38%** |
| qwen3.6-35b-a3b | 0.375 | 2.25 | ~$2,268 | −24% |
| **qwen3.5-122b-a10b (current)** | 0.40 | 3.20 | **~$2,995** | baseline |
| qwen3.6-plus | 0.50 | 3.00 | ~$3,024 | +1% |
| qwen3.7-max (discounted) | 1.25 | 3.75 | ~$4,860 | +62% (+224% if discount ends) |
| glm-5.2 | 1.40 | 4.40 | ~$5,587 | +87% |
| qwen3.6-max-preview | 1.30 | 7.80 | ~$7,862 | +163% |

- Our workload is **output-heavy** (JSON + Korean rationale), so output price dominates → qwen3.7-plus's −50% output price is decisive.
- Rate limits: plus 15,000 RPM / 5M TPM ≫ max 600 RPM / 1M TPM → **plus is better for bulk throughput**.
- **Incomplete:** an A/B extraction benchmark of 5 models (3.5-122b / 3.6-35b-a3b / 3.6-plus / 3.6-max-preview / 3.7-max) on our contracts was started but **did not finish** (reasoning-model latency / rate limits) → see §9.

- Reference: workflow output `<session>/tasks/wih1c6pz6.output` (4 research agents + synthesis, with source URLs and figures).

---

## 6. GOLD-description data-source investigation

**Goal:** obtain a "gold" reference description per image to evaluate VLM (Gemma/Qwen) description quality.

### 6.1 Our export/manifest has no descriptions
- `docs/붙임1_원문저작물 메타데이터.xlsx` (144K, 75 cols): **no description column.** The only prose column (`공공저작물활용안내`) is filled just 8% of the time. The rest are short tags (subject keywords, hashtags).
- Our `dataset/manifest.xlsx` (1,500 works): **`저작물 설명` ~93% empty (`-`)** (museum artifacts / historical photos were registered without descriptions).

### 6.2 KOGL detail pages DO have `저작물 설명` (a field absent from the export)
- The KOGL web detail page `recommend/recommendDivView.do?recommendIdx=N&division=img` has a **`저작물 설명` modal** with a real description. **Accessible anonymously (no login).**
- The representative image is also **anonymously downloadable** from `/upload_recommend/...jpg` (~700px).
- Rich descriptions come from **tourism/heritage/local-government collections** (Suwon Hwaseong 2,313 chars, Garden of Morning Calm 1,469 chars). Museum artifacts are mostly empty.
- ★ Each recommendIdx = a **gallery** of multiple images sharing one description → take one representative image.

### 6.3 공유마당 (gongu.copyright.or.kr) — a separate new source
- Anonymous download verified. Details: **`docs/gongu_download_research_20260720.md`** + downloader `api/module/dataset_builder/gongu_downloader.py`.
- Summary: 1.39M images / 128K videos; `licenseCd=97` = expired (public domain). CCL (21/24) works are the sweet spot for file+description together. The AI source-data board = 7.25M application-based records.

### 6.4 GOLD set obtained (done)
- A workflow collected **79 candidates** across tourism/heritage/nature themes → the scraper filtered (description ≥80 chars · dedup · image download success) → **72 final works**.
- Location: **`dataset/kogl_gold/kogl_gold.xlsx`** (+ `images/*.jpg`, 72 files). Description length median 321 chars, range 88–3,028.
- Examples: Jeonju Hanok Village, Eunpyeong Hanok Village, Gokseong Rose Festival, Mokpo Marine Cable Car, Gaesimsa Temple (Seosan), Surak Falls (Gurye), etc.

### 6.5 ⚠ Key methodological finding: KOGL GOLD is a "background essay," not a "caption"
- KOGL `저작물 설명` is **historical/tourism background about the subject**, not a visual caption of the photo. (Suwon Hwaseong → text about King Taejong/Jeongjo; the photo = a snow-lit fortress at night.)
- So scoring a VLM's (accurate) visual description against this essay by exact match is meaningless (initially 1/5). → The rubric was redesigned to **subject-consistency + descriptive-quality** (score by subject match / error, not essay reproduction). This rubric correctly catches real errors like the "Japanese-style" misread in §4.
- Note: the only true image-caption GOLD is **AI-Hub Korean Image Description (dataSetSn=261)**, but it's gated to Korean nationals with approval. An alternative rich source = the National Heritage Portal OpenAPI.

---

## 7. Evaluation infrastructure built & validated (code)

| File | Role | Status |
|---|---|---|
| `api/module/dataset_builder/kogl_gold_scraper.py` | KOGL detail page → GOLD description + representative image, produces `kogl_gold.xlsx` | ✅ validated (72 works) |
| `api/module/clip_extraction/vlm/eval_gold_descriptions.py` | Gemma vs Qwen description generation → score vs GOLD (subject/quality via qwen-max judge + text-embedding-v3 cosine + head-to-head) | ✅ 2-item smoke test done, **72-work full run pending** |
| `api/module/dataset_builder/gongu_downloader.py` | 공유마당 anonymous downloader (listing crawl → detail parse → multi-file download; handles gzip / %-filename / soft-404) | ✅ dry-run validated |
| `<scratch>/modeltest/bench.py` | Text-extraction A/B (5 models vs contracts_index ground truth) | ⚠ incomplete (latency / rate limit) |

**Judge/embedding models (validated):** judge = `qwen-max`, embedding = `text-embedding-v3` (1024-dim). (`qwen3.5-max` is not available on our key.)

---

## 8. Conclusions & recommendations (execution view)

1. **OCR:** keeping `qwen3-vl-235b-instruct` is safe. But `qwen3.6-35b-a3b` matches accuracy at 2.6× speed → **swap if it passes a degraded-scan robustness test** (bulk cost/throughput win).
2. **Extraction:** keep `qwen3.5-122b-a10b`. The real improvement is not a model swap but **schema-constrained decoding + value validation** (eliminate Invalid JSON). If cost-driven, pilot qwen3.7-plus.
3. **Consolidation/arbiter:** highest-value swap → **pilot qwen3.7-plus** (expected equal-or-better quality + −38% output cost). Use qwen3.7-max only if high-difficulty arbitration quality is empirically justified, limited to the consolidation leg.
4. **Image attributes (VLM):** **keep Gemma 4 primary (confirmed)** — the only model that identifies the hanok correctly. The Qwen fallback can move to qwen3.6-35b-a3b.
5. **Flagship misconception corrected:** qwen3.7-max & qwen3.6-max-preview are **text-only** → not OCR/VLM candidates. `-preview` is banned for production.

---

## 9. Open items / follow-ups

- [ ] **Finish the text-extraction A/B benchmark** — re-run `bench.py` with batching/low concurrency (focus current vs qwen3.7-plus); confirm field accuracy · JSON validity · latency vs our contract ground truth.
- [ ] **OCR robustness test** — re-test qwen3-vl-235b vs qwen3.6-35b-a3b on degraded/handwritten/complex-table scans (resolve the §3 limitation), then decide the swap.
- [ ] **Run the full GOLD description eval** — run `eval_gold_descriptions.py` on all 72 works → the full Gemma vs Qwen table (extends §4).
- [ ] **qwen3.7-plus consolidation pilot** — 500 works, current vs 3.7-plus on arbitration accuracy · Korean rationale quality · cost.
- [ ] (optional) If a true caption GOLD is needed, apply for AI-Hub 261 access or integrate the National Heritage Portal OpenAPI.

---

## Reference files

- 공유마당 research: `docs/gongu_download_research_20260720.md`
- API spec v3 (multimodal + OpenRouter): `docs/API_명세서_v3.md`
- Contract→work inheritance spec: `docs/CONTRACT_WORK_INHERITANCE_SPEC.md`
- GOLD set: `dataset/kogl_gold/kogl_gold.xlsx` (+ `images/`)
- Workflow raw outputs: `<session>/tasks/` — `wih1c6pz6` (text-model research), `wu2980axb` (GOLD pool), `w1y98p7vj` (공유마당 GOLD feasibility)
- OCR/model test artifacts: `<session>/tmp/ocrtest/`, `<session>/tmp/modeltest/`, `<session>/tmp/kgold_test/`
- Korean version of this report: `docs/model_eval_and_datasource_report_20260723.md`
