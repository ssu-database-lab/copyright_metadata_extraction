# Infrastructure Specification Proposal — Demonstration Environment

**Author** Soongsil University DB Lab · **Date** 2026-07-29
**Requested items** ① System architecture ② Minimum/recommended specs per component ③ API replaceability
**Assumption** Demonstration scale — 1–5 concurrent users, tens to hundreds of documents/day (bulk batch of 144K estimated separately)

---

## Summary (3 lines)

1. **The current configuration (commercial APIs) runs on 4 vCPU / 16 GB RAM with no GPU.** This is backed by live measurements of the server in production (4 vCPU / 31 GB / no GPU; API process uses 1.12 GB). Monthly operating cost ≈ **₩34,000 (~$24)**.
2. **Full self-hosting (GPU purchase) does not mean "moving the same system" — it means downgrading the models.** The current OCR and extraction models require 4–8 GPUs, which is outside the demonstration budget, and replacing them with smaller models may reduce accuracy (unverified). Initial investment ₩11.7M–18.1M + 4–8 weeks of setup.
3. **Recommendation: hybrid** — self-host NER only (no commercial equivalent exists); use APIs for everything else. Re-prioritizing OCR/LLM to **domestic vendors** removes the cross-border personal-data risk **at the same cost**.

---

# 📋 Summary for Reply (this section alone is sufficient to forward)

## Specifications and API Replaceability by Component

| System | Minimum Spec | Recommended Spec | API Replaceable? |
|---|---|---|---|
| **① OCR server** | No local resources | No local resources | **✅ Yes** — Upstage·CLOVA (domestic), Google, Azure, current Qwen3-VL |
| **② Extraction server (LLM)** | No local resources | No local resources | **✅ Yes** — current Qwen, Solar Pro (domestic), GPT, Claude, Gemini |
| **③ NER server** | 2 vCPU / 4 GB RAM | 4 vCPU / 8 GB RAM | **❌ No — self-hosting required** (no vendor provides copyright-domain tags) |
| **④ Consolidation server** | No local resources | No local resources | **✅ Yes** — same model as ② |
| **⑤ Image analysis (VLM)** | No local resources | No local resources | **✅ Yes** — Gemma 4, Gemini, GPT-4o |
| **⑥ API/Web server** | 2 vCPU / 8 GB RAM / 40 GB | 4 vCPU / 16 GB RAM / 100 GB SSD | N/A (the service itself) |

## ✅ Final Recommended Spec (③ + ⑥ combined = the one server to prepare)

> ### **4 vCPU / 16 GB RAM / 100 GB SSD / No GPU**
> **Minimum:** 2 vCPU / 8 GB RAM / 40 GB / No GPU
> **Monthly cost:** ≈ ₩34,000 (~$24, usage-based API) · **Initial investment: ₩0**

**Evidence (measured):** the identical system is currently running in production on a **4 vCPU / 31 GB RAM / no-GPU** server, with the API process consuming only **1.12 GB**. The 31 GB of RAM is in fact more than needed.

---

## 🔴 Strong Recommendation: Use commercial LLM APIs — do NOT self-host the models

Self-hosting LLMs for the demonstration environment is **not recommended**, for the following reasons.

| # | Reason |
|---|---|
| **1. The current models physically cannot be self-hosted** | The OCR model (235B) requires **470 GB of VRAM**; the extraction model (122B) requires **244 GB**. That means **4–8 H100 GPUs (₩200M+ / $150K+)** — far outside the demonstration budget. |
| **2. Even two top-end GPUs (~₩50M) are insufficient** | Two RTX PRO 6000 96 GB cards (**192 GB total**) still cannot host the three current models simultaneously — they need **194 GB**. |
| **3. Self-hosting guarantees a performance downgrade** | Even after buying GPUs, the models must be swapped for versions **~1/7 the size** (235B → 32B). Whether the currently verified **98% OCR character accuracy** would be preserved is **unverified**. |
| **4. The investment cannot be recovered** | ₩18.1M hardware ÷ ₩34,000/month API = **break-even ≈ 44 years**. For a 2-GPU configuration it is **122 years**. |
| **5. Setup time and ancillary requirements** | GPU deployment requires **4–8 weeks**, plus a public IP (campus networks typically block inbound traffic → separate application), a dedicated power circuit, a UPS, and ongoing administration. |
| **6. The API approach is already proven** | It is **running in production today**. Self-hosting would require rebuilding and re-validating from scratch. |
| **7. Security concerns are also solvable via API** | If cross-border transfer of personal data is a concern, switching to **domestic vendors (Upstage, Naver CLOVA)** resolves it — at **₩32,000/month, which is actually cheaper**. The risk can be removed at no additional cost. |

### Conclusion

> **Use commercial APIs for LLM, OCR, and VLM; self-host only the NER component.**
> This configuration requires **no GPU at all**, costs **₩0 upfront and ~₩34,000/month**, and is already validated in live operation.
>
> **Purchasing GPUs should only be considered if the client institution explicitly requires fully air-gapped (network-isolated) processing as a contractual condition.** No other justification — cost savings, performance, or throughput — supports it.

---

# Detailed Content (for reference)

---

## 1. System Architecture

```
                    [User browser / Muhayu HF Space]
                                 │  ① File upload (PDF, image, DOCX, HWP)
                                 ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  ⑥ API/Web server — FastAPI + SSE streaming + Web UI              │
 │     PipelineOrchestrator (stage orchestration, progress push)     │
 └───┬──────────────────────────────────────────────────┬───────────┘
     │ File conversion (PDF → image)                     │
     ▼                                                   │
 ┌──────────────────────────┐                            │
 │ ① OCR                    │                            │
 │   Qwen3-VL-235B (API)    │                            │
 │   fallback: mistral→google→naver                      │
 └──────────┬───────────────┘                            │
            │ Korean text                                 │
       ┌────┴────────────────┐  (parallel execution)     │
       ▼                     ▼                           │
 ┌──────────────────┐  ┌──────────────────────┐          │
 │ ② LLM extraction │  │ ③ NER ★self-hosted   │          │
 │   qwen3.5-122b   │  │   KLUE-RoBERTa 1.3GB │          │
 │   → 67-field JSON│  │   names, orgs, phones│          │
 └────────┬─────────┘  └──────────┬───────────┘          │
          └───────────┬───────────┘                      │
                      ▼                                   │
        ┌───────────────────────────────┐                │
        │ ④ Consolidation               │                │
        │   qwen3.5-122b (API)          │                │
        │   → per-field confidence +    │                │
        │     Korean rationale          │                │
        └──────────────┬────────────────┘                │
                       │        ┌────────────────────────┴──┐
                       │        │ ⑤ Image work attributes   │
                       │        │   Gemma 4 31B (API)       │
                       │        │   → description/type/tags │
                       │        └────────────┬──────────────┘
                       └─────────┬───────────┘
                                 ▼
                  Final consolidated metadata JSON (SSE)

[External APIs]  ① ② ④ ⑤ — DashScope (Singapore) · OpenRouter · Google · Mistral · Naver
[Local compute]  ③ NER + ⑥ server + file conversion only
```

> **Key observation:** only **③ NER and ⑥ the server** run locally; the other four stages are external API calls. That is why the system runs on a low-spec VM with no GPU.

---

## 2. Scenario B — Commercial APIs (current, recommended)

### 2-1. Specs and API replaceability by component

| System | Minimum | Recommended | API replaceable? |
|---|---|---|---|
| **① OCR** | No local compute (outbound HTTPS only) | Same | **✅ Yes** — Upstage $1.50/1,000p (domestic), CLOVA (domestic), Google Vision $1.50, Azure $1.50, Mistral $2.00, current Qwen3-VL ≈$1.4 |
| **② LLM extraction** | No local compute | Same | **✅ Yes** — current $0.26/$0.90 per 1M tokens; alternative Solar Pro 3 $0.15/$0.60 (domestic) |
| **③ NER** | **2 vCPU / 4 GB RAM** | **4 vCPU / 8 GB RAM** | **❌ No (the only mandatory self-hosted component)** — no commercial API provides copyright-domain tags (rights holder, work title, usage scope, usage period). Google NL API offers only 8 generic types |
| **④ Consolidation** | No local compute | Same | **✅ Yes** — low token volume; under $10/month even with premium models |
| **⑤ VLM attributes** | No local compute | Same | **✅ Yes** — Gemma 4 31B $0.09/$0.34 (lowest tier). ⚠️ OpenRouter cannot guarantee processing jurisdiction → switch to a direct contract |
| **⑥ API/Web server** | **2 vCPU / 8 GB / 40 GB** | **4 vCPU / 16 GB / 100 GB SSD** | N/A |

### 2-2. Combined server spec (③ + ⑥)

| Tier | vCPU | RAM | Disk | GPU |
|---|---|---|---|---|
| **Minimum** | 2 | 8 GB | 40 GB | **Not required** |
| **Recommended** | **4** | **16 GB** | **100 GB SSD** | **Not required** |
| *Current, measured (validated)* | *4 (Xeon 8358)* | *31 GB* | *45 GB* | *none* |

**Measured basis:** API process RSS **1.12 GB**, NER model 1.3 GB, runtime 319 MB → ~2.8 GB total.
31 GB RAM is excessive; 16 GB suffices. Disk should go from 45 GB to 100 GB to absorb accumulated results.

**Measured processing time:** one document (OCR → extraction ∥ NER → consolidation) median **109 s**; one image median **81 s**.

### 2-3. Monthly operating cost (2,000 documents / 10,000 pages)

| Stage | Service | Monthly |
|---|---|---|
| ① OCR | Qwen3-VL-235B (10,000 p) | ≈ $14 |
| ② LLM extraction | qwen3.5-122b-a10b | $4.8 |
| ③ NER | local CPU | **$0** |
| ④ Consolidation | qwen3.5-122b-a10b | ≈ $4.8 |
| ⑤ VLM | Gemma 4 31B | $0.5 |
| ⑥ Server | existing VM | ≈ $0 |
| **Total** | | **≈ $24/month (~₩34,000)** |

**Full switch to domestic vendors:** Upstage OCR + Solar Pro 3 + Gemini direct = **≈ $23/month (~₩32,000)**
→ **$1 cheaper.** The cross-border data risk can be removed at no additional cost.

### 2-4. ⚠️ Cross-border personal-data review (required before the demonstration contract)

The target documents contain **names, resident registration numbers, dates of birth, phone numbers, and addresses**.

| Level | Service | Status |
|---|---|---|
| 🟢 Safe | Naver CLOVA, Upstage | Processed domestically |
| 🟡 Manageable | Google (asia-northeast3), Azure (Korea Central) | Region can be pinned, but foreign entity |
| 🔴 Caution | **DashScope (current, Singapore)**, Mistral (EU) | Cross-border transfer — must be disclosed in the privacy policy |
| 🔴 Highest risk | **OpenRouter (current ⑤)** | Routes across many providers → processing country/entity cannot be identified; likely to be flagged in public-sector review |

---

## 3. Scenario A — Full self-hosting

### 3-1. ⚠️ Premise — the current models cannot be self-hosted

| Current model | Params | BF16 VRAM | GPUs required | Verdict |
|---|---|---|---|---|
| Qwen3-VL-235B (① OCR) | 235B MoE | **~470 GB** | H100 80GB **×8** | ❌ GPUs alone $200K+ (cloud $14,400/month) |
| qwen3.5-122b-a10b (② ④) | 122B MoE | **~244 GB** | H100 **×4** | ❌ Not viable |
| Gemma 4 31B (⑤) | 31B | 61 GB (INT4 18.7 GB) | 1 × 48 GB | ✅ Feasible |
| KLUE-RoBERTa (③) | 0.35B | 1.3 GB | CPU | ✅ Keep as is |

> **The MoE trap:** even though only 22B/10B parameters are active, **all weights must reside in VRAM**.

**→ Self-hosting mandates model replacement, with accuracy risk.**

| Stage | Current → Replacement | VRAM | Risk |
|---|---|---|---|
| ① OCR | 235B → **Qwen3-VL-8B-FP8** | 9 GB | ⚠️ Accuracy must be re-validated (best open Korean OCR model) |
| ① aux | → **PaddleOCR-VL 0.9B** | 2 GB | Korean MDPBench **86.0** (exceeds Gemini-3-pro's 74.8) |
| ② ④ | 122B → **Qwen3-32B-FP8** or **EXAONE 4.0 32B** (LG, Korean-specialized) | 33 GB | ⚠️ **67-field extraction accuracy loss — the largest risk** |
| ③ NER | **unchanged** | CPU | None |
| ⑤ VLM | Gemma 4 31B **retained** | 18.7 GB (INT4) | Low |

> 🚨 **Important: Gemma cannot replace OCR.**
> On the Korean OCR benchmark (KO-OCRAG), **Gemma-3-27B scores 3.63 and Gemma-3-12B scores 0.50** — effectively failure (Qwen3-VL-8B scores 56.26). Gemma must be restricted to **image attributes/captioning**. This confirms the current architecture (OCR = Qwen, image attributes = Gemma) was correct.

### 3-2. Hardware options (Korean retail prices, July 2026)

| Item | A-1 Minimum | **A-2 Recommended** ★ | A-3 Headroom |
|---|---|---|---|
| GPU | RTX 5090 32 GB | **RTX 6000 Ada 48 GB** | RTX PRO 6000 Blackwell 96 GB |
| GPU price | ₩7.54M | **₩11.3M** | ₩21.16M |
| TDP | 575 W (loud, open-air) | **300 W** | 300 W (Max-Q) |
| CPU / RAM | Ryzen 9 / 128 GB | Threadripper / 256 GB ECC | Threadripper PRO / 256 GB |
| Storage, PSU, UPS, etc. | ~₩3M | ~₩6.8M | ~₩10.4M |
| **Total initial investment** | **≈ ₩11.7M** | **≈ ₩18.1M** | **≈ ₩31.6M** |
| Concurrent users | 1–2 | **3–5** ← matches demo target | 5–10 |

**Two notes**
- **Do not buy A100 80 GB** — ₩38.65M domestically, yet lower performance than RTX PRO 6000 (₩21.16M) and requires a server chassis.
- **This is a poor time to buy GPUs** — the memory shortage has pushed RTX 5090 to 2.7× MSRP and RTX PRO 6000 to +55%.

### 3-2-1. Higher-end review — RTX PRO 6000 Blackwell 96 GB, 1 card vs 2

**VRAM requirement by precision**

| Model | Params | BF16 | FP8 | NVFP4 |
|---|---|---|---|---|
| Qwen3-VL-235B-A22B (current OCR) | 235B | 470 GB | 235 GB | **118 GB** |
| qwen3.5-122b-a10b (current extraction) | 122B | 244 GB | 122 GB | **61 GB** |
| Gemma 4 31B (current image VLM) | 31B | 62 GB | 31 GB | **16 GB** |
| Qwen3-VL-32B (OCR alternative) | 32B | 64 GB | 32 GB | 16 GB |
| Qwen3-VL-8B (lightweight OCR) | 8B | 16 GB | 8 GB | 4 GB |

**① Single card (96 GB)**

| Combination | Total | KV headroom | Verdict |
|---|---|---|---|
| Current OCR 235B (any precision) | ≥118 GB | negative | ❌ **Impossible — no card fixes this** |
| Current extraction 122B (NVFP4) | 61 GB | 35 GB | ✅ Feasible |
| OCR 8B FP8 + extraction 32B FP8 + VLM FP4 | 58 GB | 38 GB | ✅ **Recommended single-card build** |

**② Two cards (192 GB)**

| Combination | Total | KV headroom | Verdict |
|---|---|---|---|
| **All three current models (235B+122B+31B, all NVFP4)** | **194 GB** | **−2 GB** | ❌ **Fails by 2 GB** |
| OCR 32B FP8 + extraction 122B FP8 + VLM FP8 | 185 GB | 7 GB | ⚠️ Insufficient KV cache |
| **OCR 32B FP8 + extraction 122B FP4 + VLM FP4** | **108 GB** | **84 GB** | ✅ **Recommended two-card build** |
| Current 235B OCR (FP4) + VLM FP4 (extraction swapped) | 133 GB | 59 GB | ✅ Possible but not advised |

> **Key conclusion:** even two cards (192 GB) **cannot host all three current models simultaneously** (194 GB needed). One of the three must be replaced or swapped in and out.

**③ Placement strategy given the absence of NVLink (important)**

RTX PRO 6000 Blackwell **does not support NVLink** (verify before purchase). Therefore:

- **If a single model exceeds 96 GB**, tensor parallelism across both cards is forced, and inter-GPU traffic goes over **PCIe 5.0 (~64 GB/s)** instead of NVLink (900 GB/s+). MoE models suffer most, since expert routing is communication-heavy → applies to the current 235B (118 GB).
- **If every model fits under 61 GB**, each model sits entirely on one card → **zero inter-GPU traffic**, full speed.

```
Recommended two-card placement (no tensor parallelism)
  GPU 0 │ qwen3.5-122b (FP4) 61 GB                     → 35 GB KV headroom
  GPU 1 │ Qwen3-VL-32B (FP8) 32 GB + Gemma (FP4) 16 GB = 48 GB → 48 GB KV headroom
```

**④ Cost and power**

| Configuration | GPU cost | With system | GPU power | Break-even vs API |
|---|---|---|---|---|
| RTX PRO 6000 96 GB ×1 | ₩21.16M | ≈ ₩31.6M | 600 W (Max-Q 300 W) | ≈ 77 years |
| **RTX PRO 6000 96 GB ×2** | **₩42.32M** | **≈ ₩50M** | **1,200 W (Max-Q 600 W)** | **≈ 122 years** |

- A two-card build needs a 1,600–2,000 W PSU and a dedicated circuit. For a lab environment, **the Max-Q variant (300 W × 2) is strongly recommended**.
- ⚠️ **MoE quantization risk:** both primary models (235B-A22B, 122B-A10B) are **MoE**, and MoE is sensitive to INT4/FP4 quantization. Whether the measured 98% OCR accuracy and 67-field extraction accuracy survive FP4 compression is **unverified**.

**⑤ Assessment**

- At demo scale (1–5 concurrent users), two cards buy **VRAM capacity, not compute** — one card already has throughput to spare.
- **The only reason to buy two cards is to retain the current 122B extraction model at FP4.**
- Since Korean OCR does not benefit from larger models (Qwen3-VL-**8B** 56.26 > Qwen2.5-VL-**32B** 33.36; PaddleOCR-VL **0.9B** 86.0), **a two-card build to preserve the 235B is not recommended.**
- ₩50M can only be justified by an **air-gap requirement**, not by performance or cost.

### 3-3. Ancillary requirements (not included in the investment figures)

| Item | Requirement |
|---|---|
| **Power** | 400–700 W continuous → ₩50,000–90,000/month; dedicated circuit needed |
| **Public IP** | Required for external (Muhayu) calls. **Campus networks typically block inbound** → separate application, possibly weeks |
| **Security** | HTTPS, auth tokens, blocked admin endpoints |
| **Heat / noise** | 300 W continuous → cooling required if located in the lab |
| **Setup time** | **4–8 weeks** (GPU supply and campus network approval are variables) |

---

## 4. Scenario comparison

| Criterion | A: Self-hosted | B: Commercial APIs (current) | Winner |
|---|---|---|---|
| Initial cost | ₩11.7M–31.6M | **₩0** | **B** |
| Monthly cost | ₩50–90K power + admin labor | **₩32,000–34,000** | **B** |
| Break-even | — | — | **≈ 44 years → unrecoverable at demo scale** |
| Performance | ⚠️ Downgrade unavoidable (unverified) | **Top-tier models, validated in production** | **B** |
| **Data security** | **✅ Air-gap possible (A's only decisive advantage)** | 🔴 Cross-border transfer (but resolved by domestic vendors at identical cost) | **A** |
| Setup time | 4–8 weeks | **0 days (already running)** | **B** |
| Maintenance | High (drivers, OOM tuning, power, model updates) | Low (vendor-managed; fallback already implemented) | **B** |
| Scalability | Requires more GPUs | Immediate | **B** |

> **Conclusion:** B wins on 6 of 8 criteria. A's sole decisive advantage — air-gapped processing — is **largely resolved by switching to domestic vendors at the same ₩32,000/month**.
> **Self-hosting is justified by data sovereignty, not by cost or performance.**

---

## 5. Final recommendation — hybrid

| Stage | Placement | Rationale |
|---|---|---|
| **① OCR** | **API — re-prioritize to domestic vendors** | Upstage/CLOVA first, DashScope as fallback. The fallback chain is already implemented → **minimal code change** |
| ① aux (optional) | PaddleOCR PP-OCRv5 korean, **local CPU** | Apache-2.0, ₩0, no GPU. Provides an offline/air-gap fallback path |
| **② LLM extraction** | **Keep API** | Replacing with a self-hosted 32B carries accuracy risk with no benefit |
| **③ NER** | **✅ Self-hosted (unchanged)** | No commercial equivalent; ₩0; personal data stays local |
| **④ Consolidation** | **Keep API** | Under $10/month |
| **⑤ VLM** | **Keep API — but leave OpenRouter** | Processing jurisdiction cannot be identified → switch to a Google direct contract |
| **⑥ Server** | **Self-hosted (current VM)** | 4 vCPU / 16 GB / 100 GB |

**Hybrid final spec: 4 vCPU / 16 GB RAM / 100 GB SSD / No GPU · ₩0 upfront · ₩32,000/month**

### Conditions that would warrant revisiting GPU purchase (any one)
1. The client institution mandates **fully air-gapped processing** as a contractual condition
2. Sustained bulk processing (100K+ documents/month) is confirmed → break-even shortens to ~15 months
3. Self-model fine-tuning becomes a core research contribution

| Purpose | Recommended GPU | Cost (with system) | Models supported |
|---|---|---|---|
| **Standard (recommended)** | RTX 6000 Ada 48 GB ×1 | ≈ ₩18.1M | OCR (PaddleOCR-VL/8B) + LLM (32B) + VLM (Gemma INT4) co-resident |
| Higher | RTX PRO 6000 96 GB ×1 | ≈ ₩31.6M | Above + **current 122B extraction model (FP4, 61 GB)** retained |
| Highest | RTX PRO 6000 96 GB ×2 | ≈ ₩50M | Above with headroom (84 GB KV). **Still cannot host all three current models (194 GB)** |

> **Note:** even with two cards, the current OCR 235B + extraction 122B + VLM **cannot be co-resident** (194 GB > 192 GB). **OCR model replacement and accuracy re-validation are unavoidable in every configuration.**

---

## 6. Items requiring confirmation

Please confirm the following before the meeting — the recommendation depends on them.

1. **Is transmitting documents (containing personal data) to overseas APIs permitted in the demonstration environment?**
   - Permitted → keep Scenario B (current)
   - Not permitted → switch to domestic vendors (same cost) or self-host
2. **What is the expected throughput** (documents per day/month)? Above ~100K/month, GPU investment becomes worth evaluating.
3. **Will external systems (Muhayu) call our server directly?** Self-hosting would then require a public IP.
