# API Cost Evaluation — Copyright Metadata Extraction Pipeline

**Author** Soongsil University DB Lab · **Date** 2026-07-30
**Method** Official published rates (vendor documentation) × **measured token usage** (live pipeline instrumentation)
**FX** $1 = ₩1,400

---

## ⚠️ First: correction to the "₩34,000/month" figure

The **₩34,000 monthly figure** in the earlier infrastructure proposal has **two problems** and is superseded by this document.

| Aspect | Previous | Corrected |
|---|---|---|
| Nature | Presented as a single "monthly operating cost," **conflating API cost with server cost** | **API cost and server cost fully separated** (§5) |
| Amount | Based on **estimated** token usage | Based on **measured** tokens → for the same volume, actually **≈ ₩120,000/month** (~3.5× higher) |
| Cause | Output tokens of the consolidation stage (④) were underestimated | Measured: ④ accounts for **60.7%** of total cost |

**API cost is usage-based with no cap.** It is therefore presented as a **formula plus a volume table**, not as a single number.

---

## 1. Official rates (vendor documentation)

| Model | Stage | Input $/1M tok | Output $/1M tok | Source |
|---|---|---|---|---|
| qwen3-vl-235b-a22b-instruct | ① OCR | **$0.40** | **$1.60** | Alibaba Model Studio (updated 2026-07-15) |
| qwen3.5-122b-a10b | ② extraction, ④ consolidation | **$0.40** | **$3.20** | same |
| google/gemma-4-31b-it | ⑤ image VLM | **$0.10** | **$0.34** | OpenRouter model page / API |
| KLUE-RoBERTa-Large | ③ NER | — | — | **local CPU inference → ₩0** |

- Region: **Singapore (International)** — our endpoint. The same models are cheaper in Frankfurt/Virginia/Beijing (see §7).
- OpenRouter's 5.5% card top-up fee is included in the ⑤ rate.
- Sources: `alibabacloud.com/help/en/model-studio/model-pricing` · `openrouter.ai/google/gemma-4-31b-it`

## 2. Measured token usage

Obtained by instrumenting the live pipeline (collecting the `usage` field from API responses). These are not estimates.

**Measurement 1: 4-page document (copyright assignment contract, 117.5 s total)**

| Stage | Model | Calls | Input tok | Output tok |
|---|---|---|---|---|
| ① OCR | qwen3-vl-235b | **4** (1 per page) | 8,404 (**2,101/page**) | 2,780 (695/page) |
| ② Extraction | qwen3.5-122b | 1 | 6,181 | 1,782 |
| ④ Consolidation | qwen3.5-122b | 1 | 4,355 | **7,176** |
| ③ NER | local | — | — | — |

**Measurement 2: one image work (92.3 s total)**

| Stage | Model | Calls | Input tok | Output tok |
|---|---|---|---|---|
| ⑤ VLM | gemma-4-31b-it | 1 | 789 | 218 |
| ④ Consolidation | qwen3.5-122b | 1 | 1,300 | **7,731** |

**Fixed vs. variable split of the extraction prompt (separately measured)**

| Input text | Prompt tokens | Interpretation |
|---|---|---|
| none (schema + instructions only) | **4,045** | fixed cost per document |
| 1 page (1,055 chars) | 4,654 | |
| 4 pages (4,221 chars) | 6,277 | → **≈558 tokens** added per page |

## 3. Cost per API call

| Stage | Model | In / Out tok | USD/call | **KRW/call** |
|---|---|---|---|---|
| **① OCR (1 page)** | qwen3-vl-235b | 2,101 / 695 | $0.00195 | **₩2.7** |
| **② Extraction** (fixed part) | qwen3.5-122b | 4,045 / 1,782 | $0.00732 | **₩10.2** |
| **④ Consolidation (document)** | qwen3.5-122b | 4,355 / 7,176 | $0.02471 | **₩34.6** |
| **④ Consolidation (image)** | qwen3.5-122b | 1,300 / 7,731 | $0.02526 | **₩35.4** |
| **⑤ VLM (1 image)** | gemma-4-31b-it | 789 / 218 | $0.00016 | **₩0.2** |
| **③ NER** | local CPU | — | $0 | **₩0** |

## 4. Cost formula ★ (the core of this document)

Cost is determined by **two variables: page count and document count.** OCR is incurred per page, but extraction and consolidation run **once per document** regardless of page count.

> ### Cost = **₩3.05 × pages** + **₩44.8 × documents**
>
> - **₩3.05/page** = OCR call ₩2.7 + extraction prompt growth ₩0.35
> - **₩44.8/document** = consolidation ₩34.6 + extraction fixed part ₩10.2

**Validation:** the measured 4-page document cost $0.04069; the formula gives $0.04073 → **0.1% error**.

### Why page count alone cannot price the workload

The fixed cost per document (₩44.8) is roughly **15× the per-page cost** (₩3.05). Therefore, for the same page count, cost varies greatly with document count.

| Same 100 pages, different document shape | Cost |
|---|---|
| 100 pages as 1 document | ₩362 |
| 100 pages as 10 documents (10p each) | ₩753 |
| 100 pages as 100 documents (1p each) | **₩4,788** |

→ **Identical page count, 13× cost difference.** Document count must always be quoted alongside page count.

### Reference: actual page distribution (93 PDFs measured in the project)

- **Median: 1 page** (74 of 93 are single-page)
- **Mean: 4.0 pages** (raised by bundled documents of 16p, 18p, 22p, 39p, 139p)

## 5. Cost by unit and by volume

### 5-1. Per processing unit

| Unit | USD | **KRW** |
|---|---|---|
| Document, 1 page | $0.0342 | **₩48** |
| Document, 2 pages | $0.0364 | **₩51** |
| Document, 4 pages | $0.0407 | **₩57** |
| Document, 10 pages | $0.0538 | **₩75** |
| Image work, 1 file | $0.0254 | **₩36** |

### 5-2. For 1,000 pages

| Document shape | USD | **KRW** |
|---|---|---|
| 250 docs × 4 pages | $10.18 | **₩14,255** |
| 500 docs × 2 pages | $18.19 | **₩25,464** |
| 1,000 docs × 1 page | $34.20 | **₩47,882** |
| (ref.) 1,000 image works | $25.42 | ₩35,589 |

### 5-3. Monthly cost table (20 working days, KRW)

| Documents/day | 1p/doc | 2p/doc | 4p/doc | 10p/doc |
|---|---|---|---|---|
| **10** | 9,576 | 10,186 | 11,404 | 15,059 |
| **50** | 47,882 | 50,928 | 57,019 | 75,294 |
| **100** | 95,763 | 101,855 | 114,038 | 150,588 |
| **250** | 239,408 | 254,638 | 285,096 | 376,471 |
| **500** | 478,817 | 509,275 | 570,192 | 752,942 |

**User count does not directly affect cost.** One user processing 50 documents costs exactly the same as five users processing 10 each (= 50 documents/day). Cost is proportional to **volume only**.

**Example: 5 users × 50 docs/day = 250 docs/day**

| Pages per document | Monthly cost |
|---|---|
| 1 page (measured median) | **₩239,408** |
| 4 pages (measured mean) | **₩285,096** |

## 6. Separating API cost from server cost

| Category | Nature | Amount |
|---|---|---|
| **API cost** | **Usage-based, no cap** | See §5 (e.g., 50 docs/day at 1p → ₩47,882/month) |
| **Server cost** | Fixed monthly | **Separate** — currently ₩0, as an existing Oracle Cloud VM is reused. A new 4 vCPU / 16 GB / 100 GB instance would run roughly **₩45,000–100,000/month** depending on provider |

- The earlier ₩34,000 figure conflated these two items and should be discarded in favour of this separated presentation.
- **Free tier:** Alibaba Model Studio grants new accounts 1,000,000 tokens per model (90 days, Singapore region only) → about **51 four-page documents** free. Adequate for initial demo validation, insufficient for pilot operation.

## 7. Cost-reduction opportunities (reference)

The figures above reflect the current implementation and should be treated as an **upper bound**. The following measures would reduce them materially.

| Item | Detail | Saving |
|---|---|---|
| **Optimize ④ consolidation** | Accounts for **60.7%** of document cost and **99.4%** of image cost. For images there is no NER result to arbitrate, yet all 67 fields are regenerated with rationale | Image cost ₩36 → **₩0.2** (≈99% reduction) |
| **Change region** | The same qwen3.5-122b is $0.115/$0.917 under Global scope (Frankfurt/Virginia) vs. our $0.40/$3.20 | Up to **≈70% reduction** (free tier not applicable; the cross-border personal-data issue is unchanged) |
| **Lower-tier model** | Replace consolidation with e.g. qwen3.5-flash ($0.10/$0.40) | Large saving, **requires quality validation** |

## 8. Summary

1. **Per-call rates:** OCR ₩2.7/page · extraction ₩10.2/document · consolidation ₩34.6/document · VLM ₩0.2/image · NER ₩0
2. **Formula: Cost = ₩3.05 × pages + ₩44.8 × documents** (0.1% error vs. measurement)
3. **Page count alone is insufficient** — the per-document fixed cost is 15× the per-page cost, so **document count and pages-per-document must both be specified**.
4. **API cost is uncapped and usage-based**; until expected volume is confirmed, quoting the range in §5-3 is the safe approach.
5. **Server cost is separate from API cost** and is currently ₩0 through reuse of the existing VM.
