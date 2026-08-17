# Demonstration Test Cost Estimate and Commercial API Data-Policy Review

**Author** Soongsil University DB Lab · **Date** 2026-07-31
**Requested** ① Cost table by processing volume ② Documentation of the costing method ③ Verification of commercial APIs regarding external transmission, training use, and personal-data handling

---

## Summary

| Item | Conclusion |
|---|---|
| **Cost** | 1 unit (5-page contract + 1 image work) = **₩96**. 1,000 units = **₩95,654**; 6,000 units = **₩573,924** |
| **External transmission** | **Yes.** Alibaba (stored in Singapore; inference on global nodes excluding mainland China) · OpenRouter (US + third-party providers) |
| **Training use** | Alibaba: **not used** (contractual, opt-in structure) / OpenRouter itself: not used, but **sub-providers default to "training allowed"** → setting change required |
| **Personal data** | ⚠️ **Sending real contracts containing resident registration numbers under the current configuration would violate the Personal Information Protection Act.** Consent does not cure this for RRNs → **masking before transmission is mandatory** |

> ⚠️ **Most urgent action:** if real contracts are to be used in the demonstration, **RRN masking** and the **OpenRouter data-policy setting** must be applied first. See §4–§5.

---

## 1. Demonstration Test Cost

### 1-1. Test configuration assumptions

| Item | Detail |
|---|---|
| Processing unit (1 case) | **5-page standard contract + 1 image work** |
| Submission method | Contract and work submitted **together as one set** (same flow as the Muhayu UI) |
| API calls per unit | **9** — OCR 5 + extraction 1 + consolidation 1 + VLM 1 + consolidation 1 |
| NER | Local CPU → **₩0, no external transmission** |
| Contract→work inheritance merge | Pure computation (no LLM call) → **₩0** |

### 1-2. Cost composition per unit

| Component | USD | KRW | Share |
|---|---|---|---|
| Contract OCR (5 pages × 5 calls) | $0.00976 | ₩13.7 | 14.3% |
| Contract metadata extraction (1 call) | $0.00844 | ₩11.8 | 12.3% |
| Contract consolidation (1 call) | $0.02471 | ₩34.6 | 36.2% |
| Work image VLM analysis (1 call) | $0.00016 | ₩0.2 | 0.2% |
| Work consolidation (1 call) | $0.02526 | ₩35.4 | 37.0% |
| Contract→work inheritance merge | $0 | ₩0 | 0% |
| NER (local) | $0 | ₩0 | 0% |
| **Total** | **$0.06832** | **₩95.7** | 100% |

### 1-3. ★ Cost by processing volume (requested summary table)

| Processing unit | USD | KRW |
|---|---|---|
| **1 case** | **$0.07** | **₩96** |
| **1,000 cases** | **$68.32** | **₩95,654** |
| **2,000 cases** | **$136.65** | **₩191,308** |
| **3,000 cases** | **$204.97** | **₩286,962** |
| **4,000 cases** | **$273.30** | **₩382,616** |
| **6,000 cases** | **$409.95** | **₩573,924** |

### 1-4. Variation by work type (contract fixed at 5 pages)

| Work type | 1 case | 1,000 cases | Note |
|---|---|---|---|
| **Image** (baseline) | ₩95.7 | ₩95,654 | VLM + consolidation |
| **Text (어문), 1 page** | ₩107.9 | ₩107,947 | Uses the document path (OCR+extraction+consolidation) |
| **Video — short clip** | ₩95.9 | ₩95,861 | ⚠️ **Not implemented · estimate** (6 keyframes) |
| **Video — typical** | ₩96.3 | ₩96,274 | ⚠️ **Not implemented · estimate** (16 keyframes) |
| **Video — long (78 min)** | ₩100.6 | ₩100,575 | ⚠️ **Not implemented · estimate** (120 keyframes) |

**Basis for the video estimate**

Video is costed assuming **keyframe extraction analysed in a single multi-image VLM call** (frame extraction runs locally at zero cost).

| Keyframes | VLM cost | Consolidation | Total |
|---|---|---|---|
| 6 (short clip) | ₩0.3 | ₩35.4 | ₩35.7 |
| 16 | ₩0.8 | ₩35.4 | ₩36.2 |
| 120 (78-min video) | ₩5.1 | ₩35.4 | ₩40.5 |

> **Frame count barely affects cost.** Each frame is roughly 280 tokens at $0.10/1M input, so going from 1 to 120 frames adds only **₩4.9**. Since **88–99% of the work-side cost is consolidation**, image, video and text differ only marginally.
>
> **Assumptions:** ① keyframe sampling (1 frame per 10 s, capped at 120) ② ~280 tokens per frame ③ **audio (ASR) not included** — if speech-derived metadata is required, local Whisper adds ₩0 in API cost.
>
> **Reference:** measuring 40 actual 공유마당 videos gave a median of 6 seconds and a mean of 13.9 minutes — a **bimodal** distribution (73% under 1 minute, 27% over 15 minutes). For a "few clips" scenario, use **₩96 per case**.

> ⚠️ **Video work processing is not implemented.**
> The pipeline halts with `video track not enabled` when it detects video/audio files (planned as multimodal phase 3).
> **If video works are to be included in the demonstration, development must precede it — please confirm the target work types with Muhayu first.**

---

## 2. Costing Method (requested documentation)

No estimates were used. Costs were derived from **official published rates × measured token usage**.

### 2-1. Rates — official vendor documentation

| Model | Role | Input $/1M tok | Output $/1M tok | Source |
|---|---|---|---|---|
| qwen3-vl-235b-a22b-instruct | OCR | $0.40 | $1.60 | Alibaba Model Studio official pricing (updated 2026-07-15) |
| qwen3.5-122b-a10b | Extraction, consolidation | $0.40 | $3.20 | same |
| google/gemma-4-31b-it | Image VLM | $0.10 | $0.34 | OpenRouter model page / API |

- Region: **Singapore (International)** — our current endpoint
- OpenRouter's 5.5% card top-up fee included
- FX: $1 = ₩1,400

### 2-2. Token usage — live pipeline instrumentation

Real documents were run through the pipeline and the **`usage` field of each API response was collected** to measure per-stage token consumption.

| Stage | Model | Calls | Input tokens | Output tokens |
|---|---|---|---|---|
| OCR | qwen3-vl-235b | 1 per page | **2,101/page** | 695/page |
| Extraction | qwen3.5-122b | 1 per document | **4,045 fixed + 558/page** | 1,782 |
| Consolidation (document) | qwen3.5-122b | 1 per document | 4,355 | **7,176** |
| VLM | gemma-4-31b-it | 1 per image | 789 | 218 |
| Consolidation (image) | qwen3.5-122b | 1 per image | 1,300 | **7,731** |

- Subjects: a real 4-page copyright assignment contract and one image work
- The fixed/variable split of the extraction prompt was separately verified with three measurements at different input lengths (no text 4,045 → 1 page 4,654 → 4 pages 6,277)

### 2-3. Formula

Cost is determined by **two variables: page count and document count.** OCR is incurred per page, while extraction and consolidation run **once per document**.

> **Cost = ₩3.05 × pages + ₩44.8 × documents**
> (+ ₩35.6 per image work)

**Validation:** the measured 4-page document cost $0.04069; the formula yields $0.04073 → **0.1% error**.

### 2-4. Notes on the estimate

- **API cost is usage-based with no cap.** It rises proportionally with volume.
- **User count does not directly affect cost.** One user processing 50 cases costs the same as five users processing 10 each. Cost is proportional to **volume only**.
- The above covers **API usage only**; **server rental is separate** (currently ₩0 through reuse of an existing VM).
- Retries and errors add calls, so budgeting **105–110%** of the figures above is prudent.

---

## 3. Commercial API Data-Policy Findings

### 3-1. Overview

| Item | ①②④ Alibaba Model Studio | ⑤ OpenRouter (Gemma) | ③ NER |
|---|---|---|---|
| **External transmission** | **Yes** — stored in Singapore region | **Yes** — US + third-party providers | **None** (local CPU) |
| **Processing country identifiable** | Storage = Singapore (stated). Inference = **global nodes excluding mainland China** (third countries possible) | ❌ **Cannot be specified** (no guarantee beyond an EU-only option) | N/A |
| **Training use** | ✅ **Not used** (contractual) | OpenRouter itself: not used / ⚠️ **sub-providers default to allowing training** | N/A |
| **Retention period** | ⚠️ Storage confirmed, **specific period undisclosed** | Not stored by default (logging is opt-in) | N/A |
| **DPA** | ✅ Auto-incorporated into the membership agreement | ✅ Published / ⚠️ **sensitive data excluded by default** | N/A |

### 3-2. Alibaba Model Studio (OCR, extraction, consolidation)

**Training use — not used (strong basis)**

> "Alibaba Cloud will not use your Member Content to develop or improve the models on Model Studio, **unless you separately provide your consent**."
> — Product Terms §4.48(e)

This is an **opt-in structure** — no toggle needs to be disabled. The technical documentation reconfirms: "will never use your data for model training."

**Data location — technical docs and contract conflict (caution)**

- Technical docs: the Singapore region stores data in Singapore, with inference on **"global nodes excluding the Chinese mainland."** Mainland China is excluded, but **computation may occur in third countries** (EU, US, Japan, etc.).
- Contract (Product Terms §4.48(g)(viii)): a broad cross-border transfer clause that does not specify countries, and §4.48(g)(vii) **explicitly disclaims all warranties, including compliance with law**.

→ **To assure a public institution that data will not be transferred to China, a written vendor confirmation or contractual special term is required.** The standard terms alone are insufficient.

**Retention period — could not be confirmed**

Storage is acknowledged in the official FAQ ("will store data generated from model and application calls"), but **no official document states the retention period for API inputs/outputs.** Only monitoring data (30 days) is specified. → **Written inquiry required before contracting.**

**Responsibility for personal data — entirely ours**

> "You represent and warrant that **you have obtained the necessary consents from relevant individuals** … in relation to Member Content"
> — Product Terms §4.48(b)

We are the controller and Alibaba the processor. **The contract explicitly places the burden of obtaining data-subject consent entirely on us.**

### 3-3. OpenRouter (image VLM) — ⚠️ immediate action required

**Sub-provider training is currently permitted by default.**

> `data_collection` — **Default `"allow"`**
> "**allow**: (default) allow providers which store user data non-transiently and **may train on it**"
> — Provider Routing documentation

> "OpenRouter **cannot control Model Provider-side training** once user data is transmitted to a training-permitted Model Provider."
> — Privacy Policy §4

**Our code does not currently specify this setting, so it is at the default (allow).** → See §5.

**The processing country cannot be specified.** The provider *company* can be pinned via `provider.only`, but **country-level guarantees are unavailable except for an EU-only option.** This is likely to be flagged in public-sector review.

**Sensitive data is excluded by default in the DPA**, so transmitting documents containing resident registration numbers may breach the agreement.

**One non-opt-out item:** a sample of prompts is sent to a separate categorization model (not stored, anonymized). Use of the service constitutes consent and it cannot be refused, so it must be disclosed in the demonstration filing.

---

## 4. Legal Review — using real contracts

> The following is a technical review based on published statutes and guidelines. **Final determination should be made through legal counsel.**

### 4-1. Core conclusion

**Transmitting real contracts containing resident registration numbers (RRNs) under the current configuration would violate the Personal Information Protection Act (PIPA).**

Before the cross-border transfer requirements (Art. 28-8) even apply, **the processing of RRNs itself (Art. 24-2)** is restricted — and this restriction **is not cured by data-subject consent**.

For reference, the EU adequacy decision covering 30 countries that took effect in 2025 — Korea's most permissive cross-border transfer route — **explicitly excludes resident registration numbers**.

### 4-2. Applicability of cross-border rules

PIPA Art. 28-8(1) defines "transfer" to include **provision, processing consignment, storage — and even "cases where data is merely accessed."**
→ The argument "we only make API calls and do not store data" does not hold; API calls constitute cross-border transfer.

The only practically applicable exception is **item 1 (separate consent from the data subject)**. The others (statutory provision, contract performance, certification, adequacy decision) do not apply to this demonstration.

### 4-3. Five mandatory disclosures if obtaining consent (Art. 28-8(2))

| Disclosure item | Content for the demonstration |
|---|---|
| 1. Items transferred | Name, date of birth, address, contact number, signature contained in contract images/OCR text (**RRN excluded after masking**) |
| 2. Destination country, timing, method | Singapore (during document processing, HTTPS) / United States (during image analysis, HTTPS) |
| 3. Recipient | Alibaba Cloud (Singapore) Private Ltd. / OpenRouter, Inc. — with contact details |
| 4. Purpose and retention period | Purpose: OCR and metadata extraction computation (training use prohibited) / Retention: per the vendor's official policy |
| 5. Right to refuse, procedure, effect | Refusal excludes the document from the demonstration with no disadvantage |

**Notes:** ① Consent must be obtained **separately** from other consents (Art. 22). ② A change of vendor or country requires re-notification and re-consent, so **all vendors and countries reachable through the fallback chain must be enumerated up front**.

---

## 5. Recommended Actions

### 5-1. Mandatory before the demonstration

| # | Action | Reason | Owner / Duration |
|---|---|---|---|
| **1** | **Implement RRN masking** — regex-based detection and masking immediately after OCR, before external API transmission | Art. 24-2; not curable by consent | Lab / 2–3 days |
| **2** | **Set OpenRouter data policy** — specify `data_collection: "deny"` (currently at default "allow") | Blocks sub-provider training use | Lab / 1 day |
| **3** | **Prepare cross-border transfer consent form** — containing the five items in §4-3, as a separate consent | Art. 28-8(1)(i) | With client institution |
| **4** | **Obtain written confirmation from Alibaba** — retention period, assurance of no mainland-China routing | Standard terms provide no warranty | Contracting |

### 5-2. Alternative (fundamental resolution)

If RRN masking and consent procedures prove burdensome, **switching to domestically-processed services** eliminates the cross-border issue entirely.

| Alternative | Domestic processing | No training use | Public procurement | Note |
|---|---|---|---|---|
| **Naver Cloud (public)** | ✅ Stated (Pyeongchon IDC) | ⚠️ Written confirmation needed for specific terms | ✅ **CSAP·ISMS-P certified, listed on the Digital Service Mall** | Multimodal (HCX-005) available → can also replace image analysis |
| Upstage (SaaS) | ❌ US (AWS/Azure) | ✅ Stated in terms | ❌ No CSAP | Domestic processing **only via on-premise deployment** |
| Google/Azure Seoul region | ✅ Region selectable | Requires separate confirmation | ⚠️ **CSAP "Low" grade** = for systems without personal data → **insufficient grade** | |

> **Cost impact is minimal.** Our earlier review found domestic vendors to be slightly cheaper at comparable volumes.

### 5-3. Reference — cost optimization headroom

**73% of the current cost is the consolidation stage** (contract 36.2% + work 37.0%). In particular, consolidation of work images has no NER result to arbitrate, yet regenerates all fields.
→ Optimizing this could reduce the 1,000-case cost from **₩95,654 to roughly ₩60,000**. The figures above should therefore be read as an **upper bound on the current implementation**.

---

## 6. Items Requiring Confirmation

1. **Target work types for the demonstration** — whether video works are included (development required if so)
2. **Scope of real contract usage** — whether documents containing RRNs will actually be included
3. **Cross-border consent procedure** — whether the client institution can obtain consent, or whether a switch to domestic services should be considered
