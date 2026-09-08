# OCR Robustness and Consolidation-Arbiter Comparison

**Date:** 2026-09-08 · **Subject:** the two pipeline stages that had never been tested
**Predecessor:** `docs/모델현황_및_신형대조_20260907.md` — §8-1 and §8-2 listed both as never run

---

## 0. Final recommendation — main and fallback per stage

| Stage | **Main** | **Fallback** | Basis |
|---|---|---|---|
| ① **OCR** | `qwen3-vl-235b-a22b-instruct` — **keep** | `qwen3.5-flash` → **change to `qwen3.8-flash`** | 87.4% across 720 degraded-input calls. The old fallback had never been measured; 3.8-flash is 86.4% at the same latency class |
| ② **LLM extraction** | `qwen3.5-122b-a10b` — **keep** | `qwen3.5-plus` — **keep (unmeasured; deferred)** | T1, the scored tier, is **identical** to every challenger (67/90) and the incumbent is 2.4× faster |
| ③ **NER** | `klue-roberta-large` — **keep** | none | the only local stage — a privacy design decision |
| ④ **Consolidation** | `qwen3.5-122b-a10b` — **keep** | `qwen3.5-plus` — **keep, but the timeout fix is mandatory** | best T1, fastest. The fallback has the best T2 (+9.7pp) but **had never once executed** under the 90s limit |
| ⑤-a Image VLM | `google/gemma-4-31b-it` — keep | Gemma@vLLM → Qwen3-VL-235B | out of scope here |
| ⑤-b Video VLM | `qwen3.5-omni-plus` — keep | Qwen3-VL-235B → Gemma | out of scope here |

**Combinations that must not be used** (all measured)

| Model | Stage | Why |
|---|---|---|
| `qwen3.7-plus` | consolidation | T1 **57.8%** — **16.6pp below doing no consolidation at all** (74.4%). This was July's recommendation; had it shipped it would have been a regression |
| `qwen3.8-flash` | consolidation | T2 64.4% — **3.5pp below baseline** |
| `qwen3-next-80b-a3b-instruct` | consolidation | HTTP **403 Free quota exhausted** → no-op in 0.6s. **This is the ConsolidationAgent constructor default** |
| `qwen3.8-2.4t-a95b`, `qwen3.7-max` | OCR | blind — accepts images, returns 200, says it cannot see them |
| `qwen-vl-ocr-2025-11-20` | OCR | the dedicated OCR model, but near-last on degraded input (80.2%) |

---

## 1. Why these two tests

Both were left open by the previous report.

- **OCR robustness** — every OCR comparison so far used clean printed contracts. The original reason for choosing the incumbent was robustness to low light, blur and skew, and that had never been tested.
- **Consolidation arbiter** — the only one of the six stages that had **never been compared at all**. July's report named it the stage with the most to gain and recommended a `qwen3.7-plus` pilot. That pilot never ran, and this test also explains why (§4-1).

---

## 2. OCR robustness

### 2-1. Design

**Degrade pages whose ground truth is already fixed.** Bringing in new documents would mix the effect of degradation with differences in ground-truth quality. Anchors were established by magnifying the original image and reading it — not by model consensus.

- Round 1 (2026-09-07): 1 document, 2 pages × 27 conditions × 6 models × **1 repeat** = 162 calls
- Round 2 (2026-09-08): **2 documents, 4 pages** × the 10 conditions that discriminated × 6 models × **3 repeats** = 720 calls

Every round-2 change targets a specific round-1 weakness:

| Round-1 weakness | Round-2 fix |
|---|---|
| one call per cell — run-to-run variance unmeasured | **3 repeats** |
| one source document (all 30 p2 anchors from one table) | **2 documents**; p5 is a different form (consent form, creased corner, bleed-through, handwriting) |
| 15 of 25 conditions saturated | only the **10 that discriminated** |
| boilerplate anchors a language model can fill from prior | proper nouns, numbers and dates only |
| one anchor was inverted | corrected and re-scored — see §2-3 |

### 2-2. Results (720 calls, zero failures)

| Model | Accuracy | Run-to-run spread | Traps | Median s |
|---|---|---|---|---|
| **qwen3.8-max-0902** | **89.3%** | 0.7pp | 34 | 38.1 |
| **qwen3-vl-235b (current)** | **87.4%** | 0.5pp | 87 | **12.7** |
| qwen3.8-flash | 86.4% | 0.7pp | 48 | 11.4 |
| qwen3.8-27b | 83.2% | 0.8pp | 65 | 12.9 |
| qwen3-vl-flash | 80.7% | 0.3pp | 91 | 7.1 |
| qwen-vl-ocr-2025-11-20 (dedicated OCR) | 80.2% | 0.4pp | 77 | **4.8** |

A trap is a *proven* misread: the correct string is absent and a specific known misreading is present instead. It matters more than a plain miss, because a miss leaves a field empty while a trap writes a **wrong name or address** into the metadata, indistinguishable downstream from a correct one.

Spreads of 0.3–0.8pp mean three repeats made the measurement stable.

**Per page — does the round-1 finding hold on another document?**

| Model | p1 (doc A) | **p2 (doc A)** | p4 (doc A) | **p5 (doc B)** |
|---|---|---|---|---|
| qwen3.8-flash | 90.2% | **75.9%** | 97.2% | 93.7% |
| qwen3-vl-235b (current) | 92.4% | 73.2% | 98.9% | **100.0%** |
| qwen3.8-max-0902 | 97.1% | 75.7% | 99.2% | 99.4% |

**qwen3.8-flash's advantage existed only on p2** — the personnel table every round-1 anchor came from. On the second document the incumbent leads at 100.0%.

**Only three conditions discriminate**

| Condition | Lowest | Highest | Spread |
|---|---|---|---|
| `phone_2중` (dim phone photo) | 38.7% | 58.1% | 19.4pp |
| `blur_3심` | 51.4% | 75.7% | 24.3pp |
| `lowres_3심` | 70.7% | 90.5% | 19.8pp |
| the rest (low light, skew, noise, JPEG) | — | — | 3–7pp |

Low light, skew, noise and JPEG compression move no model. **Anyone specifying acceptance-test conditions should concentrate on blur, low resolution and the combined dim-photo case; the others carry no information.**

### 2-3. Correction — one round-1 anchor was inverted

Round 1 treated `중산로9길` as correct and scored `증산로` as a misread. **It was the other way round.**

- Same-scan, same-typeface controls: 품질보`증` has no stem below the horizontal bar; 수`중`영상 does. The disputed glyph has **none** → `증`.
- External check: 은평구 contains **증산로9길, and no 중산로 exists there**.

That anchor is scored in all 21 legible p2 conditions, so it moved the entire round-1 ranking. The affected table in `docs/모델현황_및_신형대조_20260907.md` has been corrected.

**Lesson:** ground truth read by eye must be checked against **control glyphs from the same document**. A standalone reading can be wrong, and one character can invert the conclusion.

### 2-4. Verdict

**Keep `qwen3-vl-235b-a22b-instruct` as the OCR model.**

- Round 1's case for switching to qwen3.8-flash was a single-document artifact and is now **refuted**.
- `qwen3.8-max-0902` is significantly the most accurate (+1.8pp, CI [−3.57, −0.27]). But it costs roughly **10× (₩120,000 vs ₩12,600 per 4,000 pages)** at 3× the latency, and there is no evidence that 1.8pp propagates to the 11 certified attributes.
- **Change the fallback**: the existing `qwen3.5-flash` has never been measured on degraded input. `qwen3.8-flash` is 1.0pp off the incumbent at the same latency.

---

## 3. Consolidation arbiter

### 3-1. Design

The three inputs (OCR text, LLM metadata, NER entities) were built **once per contract and frozen**. FieldMapper, ValidationEngine and ReasoningGenerator are deterministic Python, so the only variable is the arbiter LLM. OCR reused the August cache, making the input byte-identical.

30 contracts × 6 arbiters × 2 repeats = 360 calls, scored with the same T1/T2 tiers as `ext30`. An **LLM-only baseline** (extraction with no consolidation) was scored alongside — "does consolidation help at all?" comes before "which arbiter is best?".

### 3-2. Results

Baseline, no consolidation: **T1 74.4% · T2 67.8%**

| Arbiter | T1 (scored tier) | T2 (party values) | 95% CI | vs LLM-only | Median s |
|---|---|---|---|---|---|
| **qwen3.5-plus** (current fallback) | 73.9% | **77.5%** | [73.9, 80.7] | **+9.7pp** | 149 |
| **qwen3.5-122b-a10b** (current main) | **74.4%** | 71.8% | [68.0, 75.3] | +4.0pp | **68** |
| qwen3.7-plus (July's pick) | **57.8%** | 68.3% | [64.4, 72.0] | +0.5pp | 219 |
| qwen3.5-flash | 70.6% | 68.2% | [64.3, 71.8] | +0.3pp | 67 |
| qwen3-next-80b (constructor default) | 74.4% | 67.8% | — | +0.0pp | 0.6 |
| qwen3.8-flash | 68.3% | 64.4% | [60.4, 68.2] | **−3.5pp** | 246 |

**Paired bootstrap vs the current arbiter** (clustered on contract, n=30)

| Arbiter | T2 Δ | 95% CI | P(Δ≤0) | T1 Δ | 95% CI |
|---|---|---|---|---|---|
| qwen3.5-plus | **+5.64pp** | [−0.55, +11.50] | 0.039 | −0.56pp | [−1.67, 0.00] |
| qwen3.7-plus | −3.36pp | [−17.48, +9.11] | 0.683 | **−16.59pp** | [−30.00, −5.56] |
| qwen3.5-flash | −3.61pp | [−6.73, −0.87] | 0.998 | −3.90pp | [−7.78, −1.11] |
| qwen3.8-flash | −7.48pp | [−14.80, −1.02] | 0.992 | −6.10pp | [−11.67, −1.11] |

### 3-3. Three findings

**① Consolidation helps only on T2.** The current arbiter's T1 is 74.4% — **identical to the baseline**. The fields the certification scores do not change whether consolidation runs or not. All of the benefit is in party values (contact, phone, business registration number, address), which matter to the Muhayu rights-inheritance flow.

**② The configured fallback is the best arbiter, and it has never run.** `qwen3.5-plus` delivers +9.7pp on T2, more than double the current arbiter's +4.0pp — while failing on every production call because of the 90s limit (§4-2). The paired CI narrowly includes zero (P=0.039), so n=30 is not enough to promote it; that needs a larger sample.

**③ July's recommendation would have been a regression.** `qwen3.7-plus` scores T1 57.8%, 16.6pp below not consolidating at all (CI [−30.00, −5.56], significant).

---

## 4. Four operational defects this exposed

### 4-1. A hardcoded model allowlist frozen at qwen3.5

`cloud_extractor.py:261` raised `ValueError` on any unlisted model name, and the list had no 3.6/3.7/3.8 entries — so **no current-generation model could be selected as extractor or arbiter at all.** This appears to be why July's recommended `qwen3.7-plus` pilot never ran. The OCR path (`alibaba_ocr.py:21`) had the same gate.

**Action:** both gates downgraded to warnings; the DashScope API decides what exists. **One guard was kept on the OCR path** — text-only models are still refused, because `qwen3.8-2.4t-a95b` and similar accept images, return HTTP 200 and reply that they cannot see them, which would silently empty the metadata for every document.

### 4-2. A 90-second hardcoded timeout was killing the fallback

`consolidation_agent.py` used `timeout=90.0` with the client's `max_retries=3`. Consolidation generates up to 8,192 tokens of JSON, and 90s is too tight:

| Arbiter | Median | p95 | Under the 90s limit |
|---|---|---|---|
| qwen3.5-122b (main) | 68s | 108s | mostly fits; retries on the tail |
| **qwen3.5-plus (fallback)** | **149s** | **180s** | **fails on every call** |
| qwen3.7-plus | 219s | 499s | fails on every call |

Each failure burns 90s × 4 = **365s** before returning a no-op.

**Action:** exposed as `CONSOLIDATION_TIMEOUT_SEC` and **raised the default from 90 to 240s** (headroom over the fallback's 180s p95).

### 4-3. Degraded results were indistinguishable from real ones

When LLM arbitration fails, a rule-based fallback runs. It used to return:

| Field | Before | Now |
|---|---|---|
| `success` | `True` | `True` (a result is still produced) |
| `consolidation_degraded` | *(absent)* | **`True`** |
| `model_used` | the model that failed | **`(rule-based fallback — no LLM arbitration)`** |
| `confidence_score` | **`0.6`** | **`0.0`** |
| `degraded_reason` | *(absent)* | the actual exception |
| log level | INFO | **ERROR** |

`0.6` cleared the 0.5 confidence threshold, so degraded results were stored as normal ones. A `status: "fallback"` field already existed but **nothing in the pipeline, API or UI ever checked it**.

The fallback model choice was also hardcoded to two specific primaries, so any other arbiter silently ran with no fallback. It is now `CONSOLIDATION_FALLBACK_MODEL`.

### 4-4. Some account models are out of free quota

`qwen3-next-80b-a3b-instruct` and `qwen3-vl-plus` return HTTP 403 *"Free quota exhausted … disable the 'use free tier only' mode"*. The first is the **`ConsolidationAgent` constructor default**, so any call path that does not name a model gets a no-op in 0.6s. This needs clearing in the console.

Separately, **13 of 360 responses were truncated at `max_tokens=8192`** and failed JSON parsing.

---

## 5. Open work

| Priority | Item | Basis |
|---|---|---|
| 1 | **Decide on promoting `qwen3.5-plus` to main arbiter** — re-test on the 100-document corpus | the paired CI narrowly includes zero (P=0.039). `dataset/e2e_eval/runs/results/` already holds 100 documents with OCR and NER cached |
| 2 | Clear the free-tier-only mode on the two 403 models | the constructor default does not work |
| 3 | Raise `max_tokens` or shrink the schema | 13 of 360 truncated |
| 4 | Handwriting and stamp OCR test | p5 contains handwriting but it was **excluded from scoring — ground truth could not be established**. Needs its own answer key |
| 5 | Whether the OCR 1.8pp gap propagates to the 11 certified attributes | the only thing that would justify adopting max-0902 |

---

## Appendix. Artifacts

| Test | Files |
|---|---|
| OCR round 1 (27 conditions) | `/home/mbmk92/eval_staging/ocr_robust/{results.jsonl, rescore.py, img/}` |
| OCR round 2 (720 calls) | `/home/mbmk92/eval_staging/ocr_confirm/{results.jsonl, anchors.py, report.py, img/}` |
| Arbiter (360 calls) | `/home/mbmk92/eval_staging/arbiter/{results.jsonl, report2.py, llm_cache/, ner_entities.json}` |
| The 90s-limit run (reference) | `/home/mbmk92/eval_staging/arbiter/results_timeout90.jsonl` |
