# Model Stack and Qwen3.8 Upgrade Review

**Date:** 2026-09-07 · **Basis:** direct code inspection + same-day measurements on three tracks (OCR, extraction, video)
**Scope:** (A) confirm the current model at each of the six pipeline stages, (B) record the 2026-08 Qwen3.8 comparison that was never written up, (C) measure the models released since, (D) decide on replacement

---

## 0. One-line conclusion

**Keep all six current models.** The new `qwen3.8-flash` is overwhelmingly better at extracting contract party PII — 70% → 98% — but on the fields the certification actually scores (T1) **all three models produced an identical 67/90**. There is no reason to pay 2.4× the runtime for a gain that is not scored.

The more important output of this review is the **two methodology corrections in §6**. Some of the August verdicts rested on weaker evidence than reported.

---

## 1. Current model stack (verified in code, 2026-09-07)

| # | Stage | Model | Runs on | Code location |
|---|---|---|---|---|
| ① | **OCR** | `qwen3-vl-235b-a22b-instruct` | Alibaba | `api/module/ocr/universal_ocr.py:197` |
| ② | **LLM metadata extraction** | `qwen3.5-122b-a10b` | Alibaba | `api/web/pipeline.py:486` |
| ③ | **NER** | `klue-roberta-large` | **local CPU** | `api/web/pipeline.py:490` |
| ④ | **Consolidation arbiter** [통합검증 중재자] | `qwen3.5-122b-a10b` | Alibaba | `api/web/pipeline.py:492` |
| ⑤-a | **Image VLM** | `google/gemma-4-31b-it` | OpenRouter | `api/module/clip_extraction/vlm/extractor.py:130` |
| ⑤-b | **Video VLM** | `qwen3.5-omni-plus` | Alibaba | `api/module/clip_extraction/vlm/extractor.py:203` |

### Fallback chains

```
① OCR      providers: alibaba → mistral → google → naver   (universal_ocr.py:148)
           models   : qwen3-vl-235b → qwen3.5-flash        (alibaba_ocr.py:34)
②/④       qwen3.5-122b-a10b → qwen3.5-plus
⑤-a image  Gemma@OpenRouter → Gemma@self-hosted vLLM → Qwen3-VL-235B
⑤-b video  qwen3.5-omni-plus → Qwen3-VL-235B → Gemma@OpenRouter → Gemma@vLLM
```

**NER is the only local stage.** It is the one point where names and contact details never leave the machine, so it is a privacy design decision, not a performance choice.

---

## 2. Why this document exists

Three Qwen3.8 comparison tracks were run on 2026-08-17/18 and decided as "keep everything", but **the results were never written up.** They existed only as raw JSON:

```
/home/mbmk92/eval_staging/ocrbench/ocr_outputs.json   OCR, 5 models
/home/mbmk92/eval_staging/vid38_results.json          video
/home/mbmk92/eval_staging/ext38_results.json          extraction pilot, 4 contracts
/home/mbmk92/eval_staging/ext30/results.jsonl         extraction, 30 contracts (+OCR cache)
```

The three existing model documents each cover a different slice:

| Document | Covers | Missing |
|---|---|---|
| `docs/model_eval_and_datasource_report_20260723.md` | 3.6 / 3.7 comparison | all of 3.8 |
| `docs/영상모델_비교보고서_20260803.md` | video, 8 models | OCR, extraction |
| `docs/파이프라인_다이어그램_20260731.md` | stack diagram | the evidence behind it |

This document becomes the **single reference for the model stack**.

---

## 3. The 2026-08 Qwen3.8 comparison (recorded retroactively)

At that time only two 3.8 models existed on our key: `qwen3.8-max` and `qwen3.8-2.4t-a95b`.

| Track | Incumbent | Challenger | Result | Verdict |
|---|---|---|---|---|
| OCR | qwen3-vl-235b · 12.7 s | qwen3.8-max · 50.0 s | output **100% identical** to qwen3.7-plus, 4× slower | keep |
| Extraction T1 | qwen3.5-122b · 92.9% | qwen3.8-max · 97.1% | +4.2 pts, **overlapping CIs** | keep |
| Extraction T2 | 65% | 3.8 family 98% | **33-pt gap, non-overlapping CIs** | not a scored attribute |
| Video | qwen3.5-omni-plus 92.3% | qwen3.8-max 84.6% | challenger worse, 5× slower | keep |

### The finding from that round that still stands

**`qwen3.8-2.4t-a95b` accepts image input without error, returns HTTP 200, and says in fluent Korean that it cannot see the image.** 25 characters on OCR, 0/26 on video. It was the best *text* extractor tested. Swapping it in on leaderboard scores alone would have **silently emptied the metadata for every image and video**. This is the basis for the rule that a vision check is the first gate for any new model.

---

## 4. What is new as of September 2026

### 4-1. Public information

| Model | Released | Character |
|---|---|---|
| **Qwen3.8-Max** | 2026-08-03 | 2.4T MoE, 1M context, $2 / $6 per 1M |
| **Qwen3.8-2.4T-A95B** | 2026-08-12 | Max-class open weights |
| **Qwen3.8-27B** | 2026-08-14 | small dense, Apache 2.0 |
| **Qwen3.8-Flash / Flash-Next** | 2026-08-26 | multimodal MoE, **~6B active**, 1M context, quoted $0.16 / $0.47 |
| **Qwen3.8-Max-0902** | 2026-09-01 | Max refresh — **improved document parsing, chart reasoning, multimodal perception**; same price |

### 4-2. What is actually reachable on our key

The workspace key now lists **165 models** (149 in July, +16). All five 3.8 models are reachable:

```
qwen3.8-2.4t-a95b   qwen3.8-27b   qwen3.8-flash   qwen3.8-max   qwen3.8-max-0902
```

**Three of these did not exist when we tested in August: `qwen3.8-flash`, `qwen3.8-27b`, `qwen3.8-max-0902`.**
`qwen3.8-max-0902` targets precisely the reason 3.8 lost the OCR track (document parsing);
`qwen3.8-flash` targets precisely the other reason (latency and cost). Retesting was justified.

> **On the vision line:** neither `qwen3.8-VL` nor `qwen3.8-Omni` exists — not on our key, not in the official documentation. The dedicated vision line is still Qwen3-VL and the dedicated omni line is still Qwen3.5-Omni.

### 4-3. Vision check (first gate)

```
qwen3.8-flash       HTTP 200  ✅ "빨간색 SSU-2026"   2.8 s
qwen3.8-max-0902    HTTP 200  ✅ "빨강, SSU-2026"    6.3 s
qwen3.8-27b         HTTP 200  ✅ "빨간색 SSU-2026"   3.6 s
qwen3-vl-235b (cur) HTTP 200  ✅ "빨간색 SSU-2026"   1.4 s
```

All three genuinely see. The `2.4t-a95b` failure mode did not recur (that model was excluded from this round).

---

## 5. Measurements

### 5-1. OCR — same two contract pages, same prompt

Ground truth was established by **magnifying and reading the source image directly**, not by model majority vote. A majority vote would have flipped two cells where only the minority was right — which actually happened here.

**Page 2, anchor scoring (30 items: personnel table, addresses, phone numbers)**

| Model | Anchors hit | Accuracy | Trap misreads | Signature column |
|---|---|---|---|---|
| **qwen3-vl-235b (current)** | 28/30 | **93.3%** | 이긴구→이진구, 복대동→북대동 | 6/10 |
| **qwen3.8-flash** | 28/30 | **93.3%** | 중산로→증산로, 이한울→이한율 | 10/10 |
| **qwen3.8-27b** | 28/30 | **93.3%** | 이긴구→이진구, 복대동→북대동 | 9/10 |
| qwen3.8-max-0902 | 26/30 | 86.7% | all three of the above | 0/10 |
| qwen3.8-max | 26/30 | 86.7% | all three of the above | 0/10 |

**Three models tie exactly; only the cells they miss differ.** The Max variants are **worse** — the 0902 document-parsing improvement is not observable on this document.

In the consent column [동의 확인], all 10 rows of the original are signed (rows 3, 5, 6, 7 with illegible handwriting). The incumbent leaves 4 rows **blank**; flash fills all 10 **with the printed name** — that is inference, not reading. Neither is correct, which means **signature presence in a contract must not be trusted from OCR output**.

**Latency and cost (median of 3 repeats per page)**

| Model | Median s | p90 | Mean output tok | 4,000 pages |
|---|---|---|---|---|
| qwen3-vl-235b (current) | 17.2 | 17.9 | 762 | **$9.02** (₩12,633) |
| qwen3.8-flash | 13.2 | 14.9 | 1,454 | $4.42 (₩6,182) |
| **qwen3.8-27b** | **12.6** | **13.2** | **645** | **$2.89** (₩4,052) |
| qwen3.8-max-0902 | 27.8 | 84.5 | 2,708 | $85.70 (₩119,986) |

> Pricing: the incumbent's rate is the confirmed figure from the cost sheet (2026-07-15). The 3.8 rates are **third-party quotes** and must be confirmed officially before any contract. Only the token counts are our own measurements.

**Verdict: keep.** `qwen3.8-27b` matching accuracy while running 27% faster and 3× cheaper is worth recording, but **n=2 pages cannot justify a switch**, and the accuracy gain is zero.

### 5-2. Metadata extraction — 30 contracts, same OCR cache as August

Every model received **byte-identical input**. The incumbent was re-run **in the same execution** (see §6-1 for why).

| Model | T1 schema (scored) | T2 party values | 95% CI | Mean s | 4,000 contracts |
|---|---|---|---|---|---|
| **qwen3.8-flash** | **67/90** | **283/289 (98%)** | 96–99% | 137 | 38.0 h |
| qwen3.8-27b | **67/90** | 270/289 (93%) | 90–96% | 217 | 60.3 h |
| **qwen3.5-122b (current)** | **67/90** | 203/289 (70%) | 65–75% | **87** | **24.0 h** |

**T1 is 67/90 for all three — and identical field by field:** title 28/30, rights holder 29/30, KOGL type 10/30 in every case. This is not "statistically indistinguishable"; it is literally the same.

**T2 does separate them — but it is not what the certification scores**

| Field | Current | qwen3.8-flash | Δ |
|---|---|---|---|
| 을_사업자등록번호 (licensee business reg. no.) | 1/18 | 16/18 | **+83 pts** |
| 갑_사업자등록번호 (licensor business reg. no.) | 3/25 | 22/25 | **+76 pts** |
| 을_주소 (licensee address) | 12/30 | 29/30 | +57 pts |
| 갑_담당자 (licensor contact person) | 21/30 | 30/30 | +30 pts |
| 갑·을 이메일 (both e-mails) | 23/30 | 30/30 | +23 pts |

The current model **essentially never extracts business registration numbers** — 4 of 43. That is a systematic blind spot, not variance.

**Verdict: keep.** Of the 11 certified attributes, only title, author and licence come from the contract — which is exactly the T1 measurement where all three models are identical. **We cannot pay 2.4× runtime (24.0 h → 38.0 h) for a gain that is not scored.**

### 5-3. Video — same frames, same prompt, same rubric (26 items)

| Model | Hits | Accuracy | Mean s |
|---|---|---|---|
| **qwen3.5-omni-plus (current)** | 22/26 | **84.6%** | **4.2** |
| qwen3.8-flash | 22/26 | **84.6%** | 7.8 |
| qwen3.8-max-0902 | 22/26 | **84.6%** | 20.2 |
| qwen3.8-27b | 21/26 | 80.8% | 12.8 |

**Three-way tie.** The incumbent is the fastest and cheapest. **Verdict: keep.**

---

## 6. Methodology corrections — the most important output of this review

### 6-1. Results vary between runs even at temperature = 0

Same model, same input, same prompt:

| Measurement | Earlier | Re-measured today | Δ |
|---|---|---|---|
| qwen3.5-omni-plus, video | 92.3% (July) | 84.6% | **−7.7 pts** |
| qwen3.5-122b, T2 extraction | 65% (August) | 70% | **+5 pts** |
| qwen3-vl-235b, OCR page 1 | `결한다` | `체결한다` | August error **did not reproduce** |

This looks like MoE routing non-determinism. Two practical consequences:

1. **Model comparisons must happen inside a single execution.** Comparing new calls against cached earlier figures is invalid.
2. **n=26 (video) and n=2 pages (OCR) have no power to separate models.** Run-to-run variance exceeds the gaps we were reporting.

### 6-2. The August video verdict rested on invalid evidence

That test called only the new models and **reused July's figures for the incumbent**. "3.8-max 84.6% vs incumbent 92.3%" was a **cross-run comparison**. Re-running the incumbent in the same execution today gives **84.6% — identical to 3.8-max**.

**The conclusion (keep the incumbent) does not change**, because it is 3–5× faster and cheaper. But the justification is corrected from "higher accuracy" to **"equal accuracy, better speed and cost"**.

### 6-3. The August OCR verdict is also partly weakened

August reported that the incumbent dropped a character in `체결한다` — "a genuine OCR error, not symbol pedantry". Today the incumbent produced `체결한다` correctly. It was **non-reproducible sampling variance, not a systematic defect**.

---

## 7. Final verdicts

| Stage | Current | Challengers | Verdict | Basis |
|---|---|---|---|---|
| ① OCR | qwen3-vl-235b | 3.8-flash / 27b / max-0902 | **keep** | anchors tie at 93.3%; Max variants worse |
| ② Extraction | qwen3.5-122b | 3.8-flash / 27b | **keep** | **T1 identical at 67/90**, 2.4× slower |
| ③ NER | klue-roberta-large | — | **keep** | only local stage (privacy design) |
| ④ Consolidation | qwen3.5-122b | — | **keep** | never tested — see §8-2 |
| ⑤-a Image | Gemma 4 31B | — | **keep** | hanok recognition advantage (July) |
| ⑤-b Video | qwen3.5-omni-plus | 3.8-flash / 27b / max-0902 | **keep** | 84.6% tie, 2–5× faster |

---

## 8. Open work, in priority order

### 8-1. OCR robustness on degraded input (never run, deferred three times)

Every OCR test so far has used **clean printed contracts**. The original reason for choosing `qwen3-vl-235b` was its robustness on **low light, blur, skew, stamps and handwriting** — and that condition has never been tested. Tying with the new models on clean input says nothing about it. **Until this test runs, any OCR replacement discussion is meaningless.**

### 8-2. Consolidation (④) has never been compared at all

The July report named it the stage with the **most to gain** from a change and recommended a `qwen3.7-plus` pilot. That pilot was never run. It is a reasoning-heavy task with different characteristics from extraction, so **the extraction results cannot be transferred to it.**

### 8-3. Hybrid routing for Muhayu rights inheritance — conditionally deferred

For any flow that needs contract party PII, `qwen3.8-flash` is clearly superior (70% → 98%), and the cost has improved since August (4.8× → 2.4× latency). But there is a precondition:

> A model that lifts business registration number recall from 3/25 to 22/25 will be correspondingly better at **resident registration numbers** [주민등록번호]. PIPA §24-2 prohibits processing them, and masking is still unimplemented. "Extracts PII far more reliably" is not unambiguously an improvement — it increases both what flows to a foreign API and what lands in stored results. **Settle the masking design first.**

### 8-4. Procedure for adopting any new model (established by this review)

```
1. Vision check       one synthetic image — HTTP 200 AND actual reading   (the 2.4t-a95b trap)
2. Same-run control   always re-measure the incumbent in the same run     (§6-1)
3. Score what counts  decide only on fields inside the 11 attributes      (§5-2)
4. Repeat latency     no single-shot timings; report median and p90       (§5-1)
```

---

## Appendix. Artifact paths

| Track | File |
|---|---|
| Vision check | `/home/mbmk92/eval_staging/probe38_sept.json` |
| OCR pages 1–2 | `/home/mbmk92/eval_staging/ocrbench/ocr_outputs_sept{,_p2}.json` |
| OCR latency | `/home/mbmk92/eval_staging/ocr_latency_sept.json` |
| Extraction, 30 | `/home/mbmk92/eval_staging/ext30b/results.jsonl` · aggregator `ext30b_report.py` |
| Video | `/home/mbmk92/eval_staging/vid38_sept_results.json` · scorer `api/module/clip_extraction/vlm/score_video_models.py` |
| August (retroactive) | `ocrbench/ocr_outputs.json` · `vid38_results.json` · `ext30/results.jsonl` |

### Sources (public information)

- Qwen3.8-Max release and specs — https://www.scmp.com/tech/article/3362738/alibabas-ai-model-qwen38-max-made-widely-accessible-ahead-open-weights-release
- Qwen3.8 line-up — https://codersera.com/blog/qwen-3-8-model-lineup-2026/
- Qwen3.8-Flash specs and pricing — https://www.datacamp.com/blog/qwen3-8-flash-next
- Qwen3.8-Max-0902 changes — https://datanorth.ai/news/alibaba-releases-qwen3-8-max-0902
- Qwen3-VL OCR, 32 languages — https://github.com/QwenLM/Qwen3-VL
