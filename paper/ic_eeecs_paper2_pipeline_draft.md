# IC-EEECS 2026 — Paper 2 Draft (2 pages): The Deployed NER–LLM Pipeline

> **Status:** REAL NUMBERS filled — end-to-end evaluation completed on **N=100 contracts** (100/100 OK; `dataset/e2e_eval/report.md`, raw `results.jsonl`). No `[EVAL]` slots remain. Remaining: Fig 1, venue template, 2-page trim. Camera-ready 2026-07-10.
> **Authorship:** [User] (lead — LLM extraction, consolidation, deployment), 황성훈 (co — NER stage), [advisor]. Pairs with Paper 1 (황성훈 lead — NER training-data composition study). The two papers cross-cite; no content overlap.
> **Integrity note:** describes the production pipeline as deployed (KLUE-RoBERTa-Large NER + Qwen LLM + Qwen arbiter). Paper 1's M1/M2/M3 models are NOT in this system; the only link stated is that the production NER's training data follows the context-inclusive composition Paper 1 recommends.

---

## STORYLINE (one paragraph — the spine)

Korea's public-domain ecosystem needs **machine-actionable rights metadata** locked inside noisy, heterogeneous Korean rights documents (이용허락 계약서 / 양도동의서 / 동의서). Neither ingredient alone is trustworthy: off-the-shelf NER suffers severe domain shift on legal text, and pure-LLM extraction is costly and hard to audit in a legally sensitive domain — trust requires **value + confidence + evidence**. We present a **division-of-labor pipeline**, deployed as a streaming web service: multi-provider OCR (fallback chain) feeds a **fine-tuned Korean NER (KLUE-RoBERTa-Large)** and a **schema-guided LLM extractor (Qwen3.5)** running **concurrently**, whose outputs an **LLM consolidation arbiter** merges field-by-field into a unified 67-field schema — every field stamped with a **provenance decision** (AGREED / CONFLICT / LLM_ONLY / NER_ONLY / MISSING), a **confidence score**, and **Korean evidence grounded in the OCR text**. We evaluate end-to-end on a KOGL-grounded synthetic-contract corpus with exact ground truth (real work metadata + generated parties), reporting field-level accuracy, decision distribution, and per-stage latency. The system runs in production (REST/SSE API + web UI); the NER stage's training data follows the context-inclusive composition recommended by our companion study [Paper 1].

## SECTION MAP (2-page budget)

| § | Section | ~space | content |
|---|---------|--------|---------|
| I | Introduction | 0.5 col | need; why NER-only and LLM-only both fail trust; division-of-labor thesis; 3 contributions |
| II | Related Work | 0.3 col | 1 paragraph: PURE (staged>joint), Ma et al. (filter-rerank), VerifiNER (verify-one-model), NESTLE — we *merge two independent extractors* with per-field provenance |
| III | System | 1.0 col | Fig 1 architecture; stages (OCR fallback chain → concurrent LLM ∥ NER → arbiter); 67-field schema; decision policy + confidence + evidence; deployment (SSE, fallbacks) |
| IV | Evaluation | 0.9 col | corpus (1,500 synthetic contracts, ground-truth index); Table I field-level accuracy (overall + key rights fields); Table II decision distribution + confidence; latency/stage; (opt.) LLM-only vs full ablation |
| V | Conclusion | 0.3 col | contributions restated; limitations (synthetic PII, arbiter uses cloud LLM); future: multimodal (VLM) track, journal version |
| — | Ack + Refs | 0.3 col | grant 2024-0-00071; ~8 refs |

---

# Title

**Arbitrating Two Extractors: A Deployed NER–LLM Consolidation Pipeline for Metadata Extraction from Korean Public-Domain Rights Documents**

*(Alt: "Division of Labor for Trustworthy Metadata: A Deployed Korean Rights-Document Extraction Pipeline with Per-Field Confidence and Evidence")*

**Authors:** [User]¹, 황성훈¹, [advisor]¹  ¹School of Computer Science and Engineering, Soongsil University, Seoul, Korea

---

## Abstract

*(~130 words)*
Public-domain works in Korea require structured rights metadata that is locked inside heterogeneous Korean rights documents. Fine-tuned NER models suffer domain shift and produce spans, not schema-ready values; large language models extract flexibly but are costly to trust in a legally sensitive domain. We present a deployed pipeline that pairs both: multi-provider OCR feeds a fine-tuned Korean NER (KLUE-RoBERTa-Large) and a schema-guided LLM extractor (Qwen3.5) running concurrently, and an LLM consolidation arbiter merges their outputs field-by-field into a unified 67-field schema — each field carrying a provenance decision (AGREED/CONFLICT/LLM_ONLY/NER_ONLY/MISSING), a confidence score, and Korean evidence grounded in the source text. On 100 documents from a KOGL-grounded synthetic-contract corpus with exact ground truth, the full pipeline reaches 73.7% field-level accuracy/value-recall (88.0% on work metadata), outperforming LLM-only extraction (71.1%) — with the largest gain on contact fields (+5.2pp) — while 31.2% of all fields are dual-confirmed by both extractors (AGREED, mean confidence 0.96) at a median 115 s/document. The system is deployed as a streaming REST/web service; its NER training data follows the context-inclusive composition recommended by our companion study.

**Keywords:** information extraction, named entity recognition, large language models, consolidation, metadata, Korean rights documents

---

## I. Introduction

*(Need + failure of single approaches + thesis; compressed from the journal plan.)*

Korean public-domain works (KOGL) are governed by rights documents whose metadata — rightsholders, license type/scope, term, compensation, PII-consent — must be extracted into a machine-actionable catalog. Two single-model approaches both fall short of *trustworthy* extraction. Fine-tuned NER is fast and local but suffers domain shift on legal text [E-NER] and emits token spans, not normalized schema values. LLM extraction produces schema-shaped records directly [Dunn et al. 2024] but is expensive, and in a legally sensitive domain its unaudited output cannot be accepted: users need to know **which value came from where, with what confidence, on what evidence**.

Contributions: (1) an **integrated, deployed pipeline** — multi-provider OCR with automatic fallback → **concurrent** fine-tuned-NER ∥ schema-guided-LLM extraction → an **LLM consolidation arbiter**; (2) a **per-field trust model**: provenance decisions (AGREED/CONFLICT/LLM_ONLY/NER_ONLY/MISSING), calibrated-band confidence, and OCR-grounded Korean evidence for every one of 67 unified-schema fields; (3) an **end-to-end evaluation** on a 1,500-document KOGL-grounded synthetic-contract corpus with exact ground truth.

## II. Related Work

Staged pipelines beat joint models for entity/relation extraction [PURE]. For LLM–small-model division of labor, filter-then-rerank uses an LLM only on hard samples [Ma et al. 2023]; VerifiNER uses an LLM to verify a *single* NER model's outputs [Kim et al. 2024]; NESTLE pairs a small IE model with a commercial LLM for Korean legal statistics [Cho et al. 2024]. Unlike these, our arbiter **merges two independent, heterogeneous extractors** (fine-tuned NER and schema-guided LLM) at the *field level*, emitting provenance and evidence rather than only refined labels — and the whole pipeline is deployed, not a benchmark harness.

## III. System

**Architecture (Fig. 1).** Upload → OCR → concurrent {LLM extraction ∥ NER} → consolidation → schema-complete JSON, served over a streaming (SSE) REST API with a step-wise web UI.

**OCR.** Multi-provider chain with automatic fallback (Alibaba Qwen-VL → Mistral → Google Vision → Naver CLOVA); early-stop guard when no text is recoverable.

**LLM extraction.** Schema-guided prompting (unified 67-field schema covering five document types); Qwen3.5-122B primary with model fallback; outputs field values + self-assessed confidence.

**NER.** KLUE-RoBERTa-Large [KLUE] token classifier (26 labels), fine-tuned on context-inclusive B-I-O data (composition per [Paper 1]); runs locally on CPU concurrently with the LLM call.

**Consolidation arbiter.** An LLM (Qwen3.5, with fallback) receives both extractors' outputs plus the OCR text and produces, per field: the merged value; a **decision** — AGREED (both concur), CONFLICT (arbiter adjudicates), LLM_ONLY / NER_ONLY (single-source), MISSING; a **confidence** in banded ranges (AGREED 0.9–1.0, CONFLICT 0.7–0.9, LLM_ONLY 0.5–0.7, NER_ONLY 0.6–0.8); and a one-sentence **Korean evidence** quoting the source text. A validation engine (format/logic checks) and field mapper (NER label → schema field) precede the arbiter; graceful degradation covers single-extractor failure.

**Deployment.** FastAPI + SSE streaming, request-scoped result store, CLI batch mode; deployed on cloud infrastructure with provider/model fallback chains throughout. *(One sentence; deployment is applicability, not a claim.)*

## IV. Evaluation

**Corpus.** 1,500 synthetic 저작재산권 이용허락 계약서 built from real KOGL work metadata (titles, rightsholders, license types from the official 144k export) with generated parties (synthetic PII), each paired with exact ground truth (`contracts_index`: 18 fields incl. 갑/을 identities, contacts, business-registration numbers). Documents are rendered .docx→PDF (MS Word) and enter the pipeline through OCR like any upload. We evaluate a stratified sample of **N=100** documents (100/100 processed successfully). Scoring: strict normalized field match for schema-anchored fields (제목/저작권자/공공누리 유형); normalized value recall for the 13 party-PII fields (digit-sequence match for numbers/phones, case-folded for emails). 1,344 scoreable field checks in total.

**Table I. Field-level accuracy / value recall (%) — LLM-only vs. full pipeline.**
| Field group | n | LLM-only | Consolidated |
|---|---|---|---|
| Work metadata (제목, 저작권자, 유형) | 300 | 87.7 | **88.0** |
| Party identities (담당자, 이용자, 대표자) | 258 | 36.8 | **38.8** |
| Contacts (전화, 이메일, 주소) | 558 | 75.4 | **80.6** |
| Identifiers (사업자/주민번호, 생년월일) | 228 | 77.2 | 77.2 |
| **Overall** | 1,344 | 71.1 | **73.7** |

Consolidation improves every recoverable group, with the largest gain on contacts (+5.2pp) — the arbiter recovers values the LLM extracted into wrong/missing fields by cross-checking NER's phone/email/address entities. The weakest group is party sub-identities (담당자/대표자 inside contract tables) — a systematic extraction gap, not scoring noise, and the clearest target for future work.

**Table II. Consolidation decision distribution (per-field provenance, 100 docs).**
| Decision | share | mean confidence |
|---|---|---|
| LLM_ONLY | 46.7% | 0.68 |
| AGREED | 31.2% | 0.96 |
| CONFLICT | 11.3% | 0.83 |
| MISSING | 9.2% | 0.00 |
| NER_ONLY | 1.6% | 0.73 |

Nearly a third of all fields are **dual-confirmed** (AGREED) and carry near-certain confidence — the trust signal single-extractor systems cannot provide; explicit MISSING (9.2%) makes gaps auditable rather than silent.

**Latency.** Median seconds/doc: OCR 33.3 · LLM 16.6 · NER 17.9 · consolidation 55.0 · **end-to-end 115.0** (mean 128.5). Running LLM∥NER concurrently saves a median 15.6 s/doc; the arbiter is the dominant cost — a candidate for lighter arbitration on AGREED-heavy documents.

## V. Conclusion

We presented a deployed division-of-labor pipeline for Korean rights-document metadata extraction whose consolidation arbiter yields per-field provenance, confidence, and evidence — the trust layer single-extractor systems lack — and validated it end-to-end on a ground-truthed contract corpus. **Limitations:** synthetic party PII (real work metadata, generated identities); arbiter uses a commercial cloud LLM; confidence bands are heuristic (calibration studied in the journal version). **Future work:** a multimodal (VLM) track for image/video works, and the extended journal study including composition-effect propagation and confidence calibration.

## Acknowledgment
IITP/MSIT SW Star-Lab / SW-oriented University program (2024-0-00071).

## References (EEECS template style; numbered by first appearance in text)
*In-text tag → number: [E-NER]=1, [Dunn]=2, [PURE]=3, [Ma]=4, [VerifiNER/Kim]=5, [NESTLE/Cho]=6, [KLUE]=7, [Paper 1]=8.*
[1] Au, T. W. T., Lampos, V. and Cox, I., (2022). E-NER — an annotated named entity recognition corpus of legal text, Proceedings of the NLLP Workshop at EMNLP.
[2] Dunn, A. et al., (2024). Structured information extraction from complex scientific text with fine-tuned large language models, Nature Communications, 15, 1418.
[3] Zhong, Z. and Chen, D., (2021). A frustratingly easy approach for entity and relation extraction, Proceedings of NAACL-HLT, 50-61.
[4] Ma, Y., Cao, Y., Hong, Y. and Sun, A., (2023). Large language model is not a good few-shot information extractor, but a good reranker for hard samples!, Findings of EMNLP, 10572-10601.
[5] Kim, S., Seo, K., Chae, H., Yeo, J. and Lee, D., (2024). VerifiNER: Verification-augmented NER via knowledge-grounded reasoning with large language models, Proceedings of ACL, 2441-2461.
[6] Cho, K., Han, S., Choi, Y. R. and Hwang, W., (2024). NESTLE: A no-code tool for statistical analysis of legal corpus, Proceedings of EACL System Demonstrations.
[7] Park, S. et al., (2021). KLUE: Korean language understanding evaluation, Proceedings of the NeurIPS Datasets and Benchmarks Track.
[8] Hwang, S., [User] and [advisor], (2026). Does context composition generalize? A multi-encoder study of B-I-O training-input composition for low-resource Korean rights-document NER, IC-EEECS 2026 (companion paper).

---
## TODO
- [x] **End-to-end eval DONE** (N=100, 100/100 OK; harness `api/module/dataset_builder/eval_e2e_contracts.py`; report `dataset/e2e_eval/report.md`, raw per-doc `results.jsonl`). All `[EVAL]` filled.
- [x] **Fig 1 DONE**: `paper/figures/fig1_architecture.{png,pdf,svg}` (16cm × 6.5cm, 300dpi PNG for Word, grayscale-safe, Times-like serif; regenerate via `paper/figures/make_fig1.py`). Suggested caption: *"Figure 1. The deployed extraction pipeline. Multi-provider OCR feeds a schema-guided LLM extractor and a fine-tuned Korean NER concurrently; an LLM consolidation arbiter merges their outputs field-by-field into the unified 67-field schema, stamping each field with a provenance decision, a confidence score, and source-grounded Korean evidence."*
- [x] Ablation included (Table I dual columns).
- [ ] **Format per official template** (`paper/eeecs_template/PaperFormat.doc(x)`, verified 2026-07-06): TWO-PAGE extended abstract — A4, 1-column, 2.5cm margins, Times New Roman; title 14pt CAPS bold centered; authors 10pt; ABSTRACT ~200 words @10pt; Keywords ≤5 @10pt; body 11pt w/ 12pt CAPS section titles (INTRODUCTION/METHOD/FINDINGS/CONCLUSION); ≤2 pages INCL. references; no page numbers; PDF submission. → compress §I–V into this shape (~1,100–1,300 words + Fig 1 + Tables I/II).
- [x] References converted to the template's own style (not IEEE).
- [x] Cross-cited with Paper 1 both ways.
- [ ] Author names/emails/order (placeholders remain).
