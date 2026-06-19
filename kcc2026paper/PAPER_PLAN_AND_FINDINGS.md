# Paper Plan, Findings & Recommendations
**Project:** Metadata extraction for Korean public-domain rights documents (계약서/동의서/양도서)
**Authors:** 황성훈 (NER) + user (LLM extraction, consolidation, deployment) · 숭실대 · SW중심대학 2024-0-00071
**Prepared:** 2026-06-19 (multi-agent analysis of KCC2026 paper + related-work survey + Nature exemplar + web survey of NER+LLM pipelines, venue, and Korean rights-IE)

> Two deliverables flow from this plan: **(1) JOURNAL paper** (main goal — full integrated pipeline) and **(2) IC-EEECS 2026 short paper** (urgent, 2 pages, ~1-week extension — drafted separately in `ic_eeecs_short_paper_draft.md`).

---

## 0. CRITICAL FINDING — fix this first (blocks both papers)

The on-disk multi-encoder NER eval results are **invalid**. All three files —
`api/module/ner/validate/{klue-roberta-large, google-bert-bert-base-multilingual-cased, FacebookAI-xlm-roberta-large}/eval_results_*.json` —
report **precision = recall = F1 = 0.0** with `support: 0` for every one of the ~45 labels, and **all three point to the same `test_data_path`** (`module/ner/training/google-bert-bert-base-multilingual-cased/validation.txt`, the mBERT validation file) even for the KLUE and XLM-R runs. The eval ran each model against the wrong/mismatched test file → label–tokenization mismatch → zero matches → 0.0 F1.

**Implication:** the multi-encoder M1/M2/M3 numbers that both papers lean on **do not yet exist** in usable form. (황성훈 reportedly has the real experiment data; once pushed to GitHub, reconcile against these broken artifacts and re-run cleanly per model with the correct per-model test file.) **This is the single most important next action.**

---

## 1. Source-material assessment

- **KCC2026 (황성훈, `kcc2026_extracted.txt`):** a clean, well-scoped *NER-only* short paper. Core result is genuinely interesting — the **M1/M2/M3 context-composition finding**: all modes look equal on the AI-augmented **Silver** test set, but on the real-proof **Gold** set **M1 (answer-only) collapses while M2≈M3**, and the collapse is **label-type-asymmetric** (Free + Regular labels break in M1; Semi-Regular survive). Stated limitation: **single model (KLUE-BERT)** → the natural journal/conference hook. A strong *seed*, not a journal alone.
- **Related work (`relatedwork_short_english.txt`):** excellent and nearly publication-ready. 7 themes that already imply the whole positioning (fine-tuned NER + LLM verification/extraction + consolidation + rights-schema). A real asset; reuse near-verbatim.
- **Nature exemplar** = Dunn, Dagdelen et al., *"Structured information extraction from scientific text with LLMs,"* Nature Communications 15:1418 (2024), DOI 10.1038/s41467-024-45563-x (arXiv:2212.05238). Useful as a **structural + evaluation template** (low-annotation recipe; human-in-the-loop flywheel; **dual evaluation** = stringent exact-match F1 + relaxed/expert score). **But their architecture is the OPPOSITE of ours** — they fold NER *into* the LLM ("NERRE"); we keep a dedicated KLUE-BERT NER *and* an LLM and reconcile them with an arbiter. That contrast is the cleanest framing for our novelty. Adopt its discipline, not its shape.

---

## 2. Recommended framing — hybrid, led by SYSTEM/PIPELINE

**Adopt the system/pipeline ("division-of-labor") framing as the spine; graft on the science framing's empirical heart and the domain framing's gap-explicit positioning + taxonomy.**

Judge scores (3 independent reviewers, overall /100):
| Proposal | novelty | coherence | feasibility | venue_fit | contrib_balance | overall |
|---|---|---|---|---|---|---|
| **P1 System/pipeline** | 5 | 8 | 6 | 7 | 7 | **64** ← winner |
| P2 Science (NER-finding-as-paper) | 5 | 6 | 7 | 5 | 4 | 54 |
| P3 Domain/resource | 6 | 7 | 6 | 6 | 5 | 62 |

Why P1: it dominates on the axes that decide journal acceptance (coherence, venue-fit, two-author balance). P2 was penalized for being thin as a sole contribution, for over-claiming a near-tautological distribution-shift result as a "law," and for subordinating the user (lead author) to the co-author. P3 had the highest novelty (gap-explicit "first" + the Free/Regular/Semi-Regular taxonomy) — import both.

**Three corrections every judge demanded (bake in):**
1. **Soften "arbitrate-and-merge."** Present it honestly as **LLM-based late-fusion arbitration of two independent extractors with confidence + OCR-grounded evidence** — a real engineering/system contribution, not a new learning method. The confidence bands (0.9–1.0 / 0.7–0.9 / …) are **hand-set heuristics**; say so plainly or add a calibration check.
2. **Demote ODRL/ccREL "normalization"** to a **schema-mapping layer** (Problem Setting / applicability), not a co-equal contribution leg — the system currently outputs a flat schema.
3. **Don't over-claim model-independence** — frame as *"the effect generalizes across the encoders we tested"* and pre-empt the distribution-shift reading in Discussion.

**One-line framing:** *an integrated division-of-labor pipeline (fine-tuned Korean B-I-O NER ∥ LLM extraction → LLM consolidation arbiter) for trustworthy rights metadata, whose NER component rests on a multi-encoder context-composition study (empirical heart) and whose arbiter is the system novelty, validated end-to-end on a KOGL-grounded corpus.*

---

## 3. Title

**Recommended:** **Division of Labor for Trustworthy Metadata: Consolidating Fine-Tuned Korean NER and LLM Extraction for Public-Domain Rights Documents**

Alternatives:
1. *From Tokens to Trustworthy Metadata: A Fine-Tuned-NER and LLM-Arbiter Pipeline for Korean Public-Domain Rights Documents*
2. *Context Matters, Then Consolidation Matters: An Integrated NER–LLM System for Rights-Metadata Extraction from Low-Resource Korean Documents*
3. *Arbitrating Two Extractors: Confidence- and Evidence-Grounded Rights-Metadata Extraction for Korean Public-Domain Works*

(Avoid "law", "empirical law", "deployed" in the title.)

---

## 4. Storyline (journal narrative arc)

1. **Need** — KOGL/public-domain ecosystem needs structured, machine-actionable rights metadata locked inside noisy, heterogeneous Korean rights documents.
2. **Why naive fails** — off-the-shelf NER suffers domain shift (E-NER: 29.4–60.4% F1 drop on legal text); pure-LLM extraction is costly, English-biased, hard to trust in a legal domain. **Trust = value + confidence + evidence.**
3. **Principle** — division of labor (PURE: staged > joint; Ma et al.; VerifiNER): fine-tuned Korean B-I-O NER does first-pass recognition, an LLM extracts in parallel against the schema, an LLM arbiter merges both into one confidence-scored, evidence-grounded record.
4. **Empirical heart (NER)** — BIO training context (M1 answer-only / M2 answer+context / M3 +negatives) is decisive in the low-resource regime: **Silver hides all differences; Gold exposes the M1 collapse; M2≈M3.** The collapse is **label-type-asymmetric** (Free/Regular/Semi-Regular taxonomy predicts which labels break). **We close KCC2026's single-model limitation by showing this across multiple encoders.**
5. **System novelty (arbiter)** — late-fusion consolidation arbiter emits per-field decisions (AGREED/CONFLICT/LLM_ONLY/NER_ONLY/MISSING) with confidence + Korean evidence — more trustworthy than either component or an LLM-only baseline.
6. **The bridge** — feeding a context-aware (M2/M3) NER rather than answer-only (M1) into the arbiter measurably improves end-to-end metadata quality — empirically linking science to system.
7. **Realization** — runs as a real streaming service; report viability as *applicability*, not a research claim.

---

## 5. Journal section structure

| # | Section | Contents | Owner |
|---|---------|----------|-------|
| 1 | **Introduction** | KOGL need; low-resource + domain-shift; trust=value+confidence+evidence; division-of-labor thesis; **4 contributions** (pipeline; multi-encoder M1/M2/M3 finding + taxonomy; consolidation arbiter; released datasets). End-to-end framing figure. | joint (user lead) |
| 2 | **Related Work** | 7-theme survey. Camp A (GPT-NER/InstructUIE/UniversalNER) = English/general straw men; Camp B (Ma et al., VerifiNER, NESTLE, PURE) = basis. Domain-shift (E-NER, CUAD). Korean (KLUE, KBMC, Song 2024). Schema (ODRL/ccREL/KOGL, Pr²Graph). **Comparison table** vs VerifiNER, Pr²Graph, **SMALLM**. Draft: `relatedwork_short_english.txt`. | joint |
| 3 | **Problem Setting & Rights Schema** | 5 doc types; 67-field unified schema; **full 26-label NER enumeration**; **formalize Free/Regular/Semi-Regular and assign all 26 labels**; schema→ODRL/ccREL/KOGL mapping layer (modest). | user (taxonomy joint) |
| 4 | **System Architecture** | Multi-provider OCR (fallback) → concurrent LLM ∥ B-I-O NER → arbiter → schema map → JSON. Rationale via PURE + Ma et al. Architecture figure. | user |
| 5 | **Datasets** | Gold (~50k, 26 labels, web-crawl + provided docs, JSONL, answer-only, real-proof eval). Silver (AI-augmented ~5×, balanced, BIO, capped 10k/label, 8/2/2). **CRITICAL: document augmentation (model, prompt, BIO synthesis, balancing, QC) + reconcile ~50k-total vs 10k/label.** Synthetic corpus (1001 .docx + `contracts_index.xlsx`). | 황성훈 (Gold/Silver); joint (synthetic) |
| 6 | **NER: B-I-O Context-Composition (M1/M2/M3)** | Empirical heart. Modes; fixed recipe (full FT, AdamW, bs 32, 3 ep, early stop). **Multi-encoder: KLUE-RoBERTa + mBERT + XLM-RoBERTa (+ KoELECTRA ideally).** Silver-eval vs Gold-eval; label-type asymmetry; cross-encoder consistency; per-label tables, significance, error analysis. | 황성훈 |
| 7 | **LLM Extraction & Consolidation Arbiter** | Schema-guided extractor; FieldMapper; ValidationEngine; arbiter decision policy + confidence bands (state heuristic origin); ReasoningGenerator (Korean evidence w/ OCR excerpts). Honest contrast with VerifiNER / Ma et al. / SMALLM. | user |
| 8 | **End-to-End Evaluation** | (a) doc-level accuracy on synthetic corpus vs ground truth; (b) ablations full vs LLM-only vs NER-only vs no-consolidation; (c) **BRIDGE: M1- vs M2/M3-NER → downstream quality**; (d) arbiter eval: conflict-resolution accuracy + confidence calibration (reliability diagram, ECE/Brier) + small human eval of evidence; (e) error analysis by doc type / label class. | joint |
| 9 | **System Realization (Applicability)** | SSE REST API, web UI, CLI, Oracle Cloud, fallback chains, concurrency, latency/throughput/cost. **Explicitly NOT a novelty claim.** Short. | user |
| 10 | **Discussion** | Why O-context carries signal; **pre-empt the distribution-shift critique** via the label-type asymmetry; annotation-budget implications; consolidation as principled fusion; human-in-the-loop for a legal domain. | joint |
| 11 | **Limitations & Future Work** | small real Gold; synthetic PII is fake; ODRL = mapping not full graph; arbiter uses commercial cloud LLM; effect shown only on tested encoders. **Multimodal = one paragraph forward pointer.** | joint |
| 12 | **Conclusion** | restate integrated contribution + generalized O-token finding + arbiter + viability. | joint |
| — | **Appendices** | A: 26-label list + taxonomy (황); B: schema→ODRL/ccREL map (user); C: Silver augmentation prompts + QC (황); D: arbiter prompt + decision policy (user); E: per-label metric tables, all models (황). | joint |

**Authorship balance:** title/intro lead with the user's integrated system (user = lead); §5–6 + appendices A/C/E give 황성훈's NER study thesis-level real estate.

---

## 6. Decisions — web deployment? multimodal?
- **Web deployment: INCLUDE as §9 "Applicability" only** (latency/throughput/cost = production-viability evidence) — *not* a novelty axis (zero deployment precedent in the literature). Keep short.
- **Multimodal (CLIP/SigLIP+FAISS, Gemma/Qwen VLM): EXCLUDE from body; one paragraph in Future Work.** Incomplete + scope-bloat risk. Save as a standalone Year-2 paper.

---

## 7. Novelty & who to position against
**Novelty sentence (Intro):** *We present the first integrated system for Korean public-domain **rights** documents that pairs a fine-tuned domain B-I-O NER with an LLM extractor and fuses them via an **LLM consolidation arbiter producing per-field confidence and OCR-grounded Korean evidence**; and we show, across multiple encoders, that **B-I-O training-context composition governs low-resource generalization in a label-type-asymmetric way** that an augmented test set conceals and a real-document test set exposes.*

Position directly against:
1. **VerifiNER (Kim et al., ACL 2024)** — closest *method* twin (LLM verifies/refines one NER model). We **merge two independent extractors** (not verify one), extend to extraction+arbitration with confidence+evidence, target Korean rights entities. High-prestige, Korean-authored.
2. **Pr²Graph (2025, arXiv:2509.01716)** — closest *end-to-end* analogue (doc → LLM → ODRL). It is **LLM-only, English, privacy-policy** domain; we add the fine-tuned NER stage + arbiter, Korean copyright.
3. **SMALLM (Complex & Intelligent Systems, Sept 2025, doi 10.1007/s40747-025-02074-6)** — closest *fusion competitor* (logit-level O_Fusion = O_GPT + β·O_BERT + CRF for token tagging). We consolidate at the **field/metadata level** with provenance + confidence + human-readable Korean evidence (not logit fusion for tagging). *If feasible, implement a SMALLM-style logit-fusion baseline.*
4. **NESTLE (Cho et al., EACL 2024)** — Korean + commercial-LLM + small internal IE, but court-precedent stats, no rights-schema normalization.

Together VerifiNER (method axis) + Pr²Graph (task axis) bracket the gap; SMALLM is the sharpest contrast for "why field-level consolidation, not logit fusion."

---

## 8. What exists vs. what to add
**Exists (de-risks):** full consolidation stack (`api/module/consolidator/`), 67-field schema (`document_schemas.py`), Silver training data (`api/module/ner/training/training_data_*.json`), per-encoder split dirs already laid down, KOGL-grounded synthetic corpus (`dataset/` — 1000 contracts + `contracts_index.xlsx`), deployed pipeline + CLI + web UI, single-model KLUE-BERT M1/M2/M3 result, drafted related work.

**Must add (journal-strong):**
1. **Fix broken eval + clean multi-encoder M1/M2/M3 runs** (KLUE-RoBERTa + mBERT + XLM-R, ideally + KoELECTRA). ← load-bearing; see §0.
2. Real metric tables + per-label numbers + significance tests + confusion/error analysis (KCC figures are placeholders).
3. Full 26-label enumeration + Free/Regular/Semi-Regular assignment.
4. Silver augmentation methodology write-up + count reconciliation + QC protocol (most likely rejection lever if undocumented).
5. End-to-end ablations (full vs LLM-only vs NER-only vs no-consolidation).
6. **BRIDGE experiment** (M1 vs M2/M3 NER → downstream metadata quality) — run early; the parallel LLM + overriding arbiter may wash out the NER signal.
7. Arbiter eval: conflict-resolution accuracy + confidence-calibration reliability diagram (ECE/Brier) + small human eval of evidence.

---

## 9. IC-EEECS 2026 short paper (urgent) — scope summary
*(Full draft: `ic_eeecs_short_paper_draft.md`.)*

**Venue facts (from CFP):** IC-EEECS = "18th EEECS 2026," Osaka, Japan, **July 21–24, 2026**. Organizers: KOCTA (Seoul), CTRC (Gachon), ICDC (Kwangwoon), CCC (Mae Fah Luang, Thailand). **2-page short papers.** Tracks include "Artificial Intelligence and Applications" + "Big Data and Data Centric Computing." Extended deadlines: paper **2026-06-15**, notify 06-30, camera-ready 07-10. Indexing (Scopus/IEEE) **not advertised** → treat as a priority-staking teaser, not a prestige credit. CFP: https://ic-eeecs.org/call-for-papers/

**Recommended scope:** the **self-contained multi-encoder NER context-composition finding** (the clean closure of KCC2026's single-model limitation). Most tractable in 2 pages, stands alone, uses the incoming NER data, and gives **황성훈 the lead** on the conference piece (user co-author) — balancing the journal where the user leads. One forward-pointing paragraph to the consolidation pipeline stakes journal priority. **Cut:** arbiter experiments, schema/ODRL, deployment, end-to-end ablations, human eval.
*Fallback if multi-encoder runs aren't ready: KLUE-RoBERTa + mBERT (two encoders still beats KCC's one).*
*(Alternative framing, if preferred: a 2-page integrated-pipeline teaser, user-led — see Appendix A, P1 short_conf.)*

---

## 10. Target-journal shortlist
| Journal | Fit |
|---|---|
| **ACM TALLIP** | Best topical fit — Korean low-resource NER/IE; values the data-composition finding + language angle. |
| **IEEE Access** | Best for the integrated-system framing; broad, fast, tolerant of the applicability section. Good given timeline pressure. |
| **Information Processing & Management** (Elsevier) | Strong for IE w/ rigorous eval; higher bar (wants full ablation + calibration). |
| **Expert Systems with Applications** (Elsevier) | Good for applied, deployed AI pipeline + end-to-end eval. |
| **ETRI Journal** | Korean, English-language, applied; friendly to gov-funded Korean-domain work. Solid backup. |
| **Applied Sciences** (MDPI) | Fast, system-oriented, OA; speed/breadth over prestige. |

**Primary:** **IEEE Access** (if integrated/deployed story leads) or **ACM TALLIP** (if the Korean NER science leads).

---

## 11. Risks & the one next action
**Risks:** (a) generalization may not replicate across encoders (mitigate: a partial/asymmetric replication is still publishable via the taxonomy, not a "law"); (b) the finding reads as a distribution-shift tautology (mitigate: foreground the **Semi-Regular-survives** asymmetry as the non-obvious part); (c) Silver-vs-Gold validity is the top rejection lever if augmentation/QC undocumented; (d) the BRIDGE experiment may wash out (run early; soften the claim if so); (e) annotation labor for calibration/human-eval strains a 2-person team (scope small); (f) novelty ceiling — the win is coherence/balance/venue-fit, not breakthrough; don't over-claim.

**#1 ACTION:** **Fix the broken NER evaluation and produce clean multi-encoder M1/M2/M3 numbers** (see §0). It simultaneously unblocks the IC-EEECS deadline and de-risks the journal's central thesis. Reconcile with 황성훈's pushed data when available.

---

## Appendix A — the three design framings (for reference)
- **P1 System/pipeline (winner, 64):** title *"From Tokens to Trustworthy Metadata: A Deployed NER-LLM Consolidation Pipeline…"*; thesis = division-of-labor pipeline > any single-component/off-the-shelf/LLM-only. short_conf = integrated-pipeline teaser using existing single-model KLUE-BERT + 1 extra encoder.
- **P2 Science (54):** title *"Outside Tokens Are Not Optional: A Multi-Model Study of BIO Training-Data Composition…"*; thesis = O-token composition is the dominant, label-type-asymmetric, (claimed) model-independent driver. short_conf = standalone NER-only multi-encoder extension. **Penalized** for thin-as-sole-contribution + over-claim + author imbalance — but its *empirical core* is the journal's heart.
- **P3 Domain (62):** title *"From Rights Documents to Machine-Readable Licenses: An End-to-End NER–LLM Consolidation Pipeline…"*; thesis = first system to extract AND normalize Korean public-domain rights metadata (dataset + 26-label schema + method + normalization). **Import** its gap-explicit positioning + taxonomy.

## Appendix B — key comparators/refs discovered
- **Nature exemplar:** Dunn et al., Nature Communications 15:1418 (2024), DOI 10.1038/s41467-024-45563-x — dual-evaluation (exact-match F1 + relaxed/expert) template; NERRE = NER folded into LLM (opposite of ours).
- **SMALLM** — Complex & Intelligent Systems (Springer), Sept 2025, doi 10.1007/s40747-025-02074-6 — logit-level BERT+LLM fusion + CRF (closest fusion competitor).
- **Xu et al.**, "LLMs for Generative IE: A Survey," Frontiers of Computer Science, arXiv:2312.17617 — Related-Work taxonomy anchor.
- Evaluation suite to adopt: per-field exact + fuzzy P/R/F1 (define Korean normalization), Schema Compliance, Field Completeness, Hallucination Rate vs OCR source; calibration: ECE/Brier + reliability diagram; agreement: Cohen/Fleiss κ.
- Core lineage (from `relatedwork_short_english.txt`): Ramshaw&Marcus 1995 (BIO), Lample 2016, PURE 2021, Ma et al. 2023, VerifiNER 2024, NESTLE 2024, KLUE 2021, KBMC 2024, Song 2024, E-NER 2022, CUAD 2021, ODRL 2.2, ccREL, Pr²Graph 2025.

## Appendix C — verified facts
- NER eval artifacts broken (0.0 F1, mismatched test_data_path) — verified by reading the three `eval_results_*.json` on 2026-06-19.
- IC-EEECS = 2-page papers, Osaka Jul 21–24 2026 — from venue CFP survey.
- KCC2026 recipe: KLUE-BERT, full FT, AdamW, batch 32, 3 epochs, early stopping; Silver capped 10k/label, split 6,666/1,666/1,668; hardware Intel Ultra 5 225F / 64GB / RTX 5070 / Ubuntu 24.04.
