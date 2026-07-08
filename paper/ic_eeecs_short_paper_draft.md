# IC-EEECS 2026 — Short Paper Draft (2 pages)

> **Status:** REAL NUMBERS filled (2026-06-29) from 황성훈's `apiBackup` branch (`kcc2026paper/paper1/data/runs/20260513_paper1_{klue,mbert,koelectra}` — a uniform 3-backbone × 3-mode grid; replications: 20260427/20260430/20260619). **No `[FILL]` slots remain.** Remaining: 황성훈 confirms the 20260513 set as canonical, venue template (IEEE 2-col), trim to 2 pages. Camera-ready 2026-07-10.
> **Authorship:** 황성훈 (lead — NER experiments), user (co-author — application framing). Balances the journal, where the user leads.
> **Integrity note (agreed framing):** this is a *training-data-composition study*, NOT a system paper. The M1/M2/M3 models are experimental artifacts; they are **not deployed** in the production pipeline. The pipeline appears only as application context + design guidance ("informs the NER stage"), never as an integration claim. See §V wording.

---

## STORYLINE (one paragraph — the spine of the paper)

Korean public-domain rights documents (license/assignment/consent forms) are a **low-resource NER** problem: 26 fine-grained labels, scarce gold data. KCC2026 found, on a single encoder (KLUE-BERT), that **how the B-I-O training input is composed** — answer tokens only (M1) vs. answers in natural context (M2) vs. context + 25% true negatives at equal volume (M3) — is decisive: an **AI-augmented (Silver) validation set hides all differences (F1 0.989–0.993 for every mode), but a human-written (Gold) cross-distribution test exposes an M1 collapse (0.46 vs 0.87)**, and the collapse is **format-class-asymmetric** (format-regular and format-free labels break under M1; structurally-cued semi-regular labels survive). The open question — KCC2026's stated limitation — was **whether this is a property of one model or a general phenomenon.** This paper answers it: we run the full 3×3 grid across **KLUE-BERT, mBERT, and KoELECTRA** and show the pattern is **encoder-independent** — M1 collapses on every encoder (0.44/0.45/0.50), M2 is best on every encoder within 0.4pp of each other (0.869–0.872), M3 tracks M2, and the format-class asymmetry reproduces throughout. The practical payload: in low-resource Korean IE, **annotate surrounding context (O tokens), not just answers**; **true negatives buy little over context**; and **synthetic-data validation scores cannot be trusted without a real-distribution test set**. We close with the application context: these findings set the training-data design for the NER stage of a Korean rights-metadata extraction pipeline.

## SECTION MAP (2-page budget)

| §   | Section                 | ~space  | content |
| --- | ----------------------- | ------- | -------- |
| I   | Introduction            | 0.6 col | low-resource Korean rights NER; the composition question; KCC2026 single-model gap; **3 contributions** |
| II  | Related Work            | 0.4 col | 1 compressed paragraph: BIO, augmentation, domain shift, Korean NER |
| III | Method                  | 0.8 col | Silver/Gold data, 26 labels, M1/M2/M3 defn (substitutive negatives), 14/6/6 taxonomy, 3 encoders, fixed recipe, lenient-match metric |
| IV  | Results                 | 1.2 col | Table I Silver (near-identical, 9 runs); Table II Gold 3×3 grid (M1 collapse, encoder-independent); Fig 1 per-format-class asymmetry |
| V   | Discussion & Conclusion | 0.5 col | why O-context carries the signal; annotation + evaluation guidance; application context (design guidance, NOT integration); limitations |
| —   | Ack + Refs              | 0.3 col | grant 2024-0-00071; ~10 refs |

---

# Title

**Does Context Composition Generalize? A Multi-Encoder Study of B-I-O Training-Input Composition for Low-Resource Korean Rights-Document NER**

*(Alt: "Outside Tokens Matter Across Encoders: B-I-O Training-Data Composition in Low-Resource Korean Rights-Document NER")*

**Authors:** 황성훈¹, [User]¹, [advisor]¹  ¹School of Computer Science and Engineering, Soongsil University, Seoul, Korea
{emails}

---

## Abstract

*(~130 words)*
Extracting metadata from Korean public-domain rights documents (license agreements, assignment and consent forms) is a low-resource named-entity-recognition (NER) task with 26 fine-grained labels and scarce gold annotations. Prior work on a single Korean encoder showed that the composition of B-I-O training input — answer tokens only (M1), answers in natural context (M2), or context plus 25% true negatives at equal volume (M3) — strongly affects real-world extraction, but left generalization beyond one model unverified. We run the full mode-by-encoder grid on three encoders (KLUE-BERT, multilingual BERT, KoELECTRA). On every encoder, an augmented (Silver) validation set conceals the differences (F1 0.989–0.993 in all nine runs) while a human-written (Gold) test set exposes a severe M1 degradation (0.44–0.50 vs. 0.869–0.872 for M2, a ≥37pp gap), with M3 tracking M2. The degradation is format-class-asymmetric: structurally-cued labels survive answer-only training while format-regular and free-form labels collapse. The effect is encoder-independent. Training data for low-resource Korean information extraction should therefore include surrounding context, and synthetic-data validation scores must be checked against real-distribution tests.

**Keywords:** named entity recognition, B-I-O tagging, low-resource Korean NLP, training-data composition, information extraction, public-domain rights documents

---

## I. Introduction

Public-domain works in Korea (under the Korea Open Government License, KOGL) are governed by rights documents — 이용허락 계약서 (license agreements), 양도동의서 (assignment forms), and 개인정보 수집·이용 동의서 (personal-information consent forms) — whose metadata (rightsholders, license type, term, usage conditions) must be extracted to make the works machine-actionable. This is a **low-resource NER** problem: 26 domain-specific labels co-occur in one document, and human-annotated data is scarce, so training data is largely synthesized by augmentation.

A recurring practical question in this regime is **how to compose the B-I-O training input**: should each training record contain only the target answer tokens, the answers embedded in natural context, or additionally entity-free negative sentences? On a single Korean encoder (KLUE-BERT), prior work [KCC2026] reported a striking result: three composition modes — **M1** (answer-only), **M2** (answer in natural context), and **M3** (75% M2 records + 25% true negatives at equal total volume) — are **indistinguishable on the augmented validation set but diverge sharply on a human-written test set**, where M1 collapses while M2 and M3 remain close; moreover the collapse depends on the label's format regularity. That study's explicit limitation was that **only one backbone was tested**, leaving open whether the phenomenon is general or a model artifact.

**This paper closes that gap.** Our contributions:

1. We run the **full 3-mode × 3-encoder grid** (KLUE-BERT, multilingual BERT, KoELECTRA) on the same Silver/Gold rights-document data with an identical recipe — nine controlled runs.
2. We show the **Silver-hides / Gold-exposes pattern and the M1 collapse are encoder-independent** — M1 scores 0.44–0.50 on every encoder while M2 scores 0.869–0.872 (a spread of only 0.4pp across encoders) — and that the degradation is **format-class-asymmetric**, captured by a 14/6/6 Regular/Semi-Regular/Free label taxonomy.
3. We distill **actionable guidance** for low-resource Korean IE: annotate surrounding context (O tokens); true negatives add little once context is present; and augmented-data validation scores are unreliable without a real-distribution test.

## II. Related Work

The B-I-O (IOB) tagging scheme recasts NER as token-level sequence labeling [Ramshaw&Marcus 1995], the basis for neural [Lample 2016] and pretrained-transformer NER, including Korean encoders [KLUE 2021; KoELECTRA]. In low-resource settings, data augmentation is common [Dai&Adel 2020], and LLM-assisted labeling has improved Korean specialized-domain NER (e.g., medical [KBMC 2024]). Applying general-domain NER to specialized text causes severe degradation — up to 29–60% F1 on legal text [E-NER 2022] — motivating domain-specific data and careful evaluation. Prior NER work varies backbones, fine-tuning methods, and data volume; ablations of the *form of the training input itself* are rare, and studies of negative sampling often conflate added volume with the negative effect. Unlike prior work, we isolate **training-input composition** as the controlled variable — including a volume-controlled (substitutive) negative condition — and test its **cross-encoder** behavior on Korean rights documents.

## III. Method

**Data.** Two datasets over **26 rights-metadata labels** in B-I-O JSONL format. The **Silver** set is augmentation-synthesized (answers placed in generated contexts), capped at **10,000 records per label**, split **8/2/2** (train/val/test). The **Gold** set is human-written text with the same entity vocabulary, held out entirely as a **cross-distribution real-proof** evaluation: it measures whether a model trained on synthetic contexts identifies the same entities in natural human-written contexts (a context-level distribution-shift evaluation, not full OOD).

**Composition modes** (equal entity vocabulary; M2/M3 equal total volume):
- **M1 answer-only** — records contain only the B/I answer tokens, no context (193,752 records).
- **M2 context** — answers embedded in natural sentence context (192,927 records).
- **M3 context + true negatives (substitutive)** — 75% M2 records + 25% entity-free natural sentences made by *removing* entity spans from M2 records (192,927 records total). Negatives are true negatives (no label noise), and total volume equals M2, removing the volume confound.

**Label taxonomy.** The 26 labels divide into **14 format-regular** (regex-tight surface forms: phone, email, UCI, dates…), **6 format-semi-regular** (strong structural cues, e.g. values following "저작물명 :"), and **6 format-free** (open vocabulary: company, description, position…).

**Encoders & recipe.** **KLUE-BERT, multilingual BERT (mBERT), KoELECTRA**, each fine-tuned identically per mode (full fine-tuning, AdamW, warmup 0.1, weight decay 0.01, early stopping, seed 42; single seed justified by prior 3-seed variance σ≈0.011 vs. mode effects ≥4pp). Hardware: RTX 5070, 64GB RAM, Ubuntu 24.04.

**Metric.** Per-label **lenient-match accuracy** on Gold (a prediction hits if the gold span is matched at threshold 0.25), averaged over the 26 labels; Silver-side standard token-classification accuracy/F1 on the validation split.

## IV. Results

**Silver validation hides the differences — on every encoder.** All nine runs are near-identical in-distribution (Table I): accuracy 0.995–0.997, F1 0.989–0.993. Mode choice is invisible on augmented data.

**Table I. Silver (in-distribution) validation — accuracy / F1.**
| Encoder | M1 | M2 | M3 |
|---|---|---|---|
| KLUE-BERT | 0.9951 / 0.9926 | 0.9970 / 0.9904 | 0.9972 / 0.9891 |
| mBERT | 0.9947 / 0.9922 | 0.9971 / 0.9905 | 0.9972 / 0.9896 |
| KoELECTRA | 0.9948 / 0.9924 | 0.9970 / 0.9907 | 0.9971 / 0.9885 |

**Gold evaluation exposes the M1 collapse — on every encoder.** On the human-written Gold set (Table II), **M1 collapses on all three encoders while M2 is best on all three**, and the M2 scores agree across encoders within 0.4pp. M3 tracks M2 (−0.9 to −3.8pp) — far above M1 (−37 to −43pp).

**Table II. Gold (cross-distribution) lenient-match accuracy — overall and by format class.**
| Encoder | Mode | Overall | Regular (14) | Semi-Reg (6) | Free (6) |
|---|---|---|---|---|---|
| KLUE-BERT | M1 | 0.4409 | 0.2818 | 0.8812 | 0.3718 |
| | **M2** | **0.8690** | 0.8583 | 0.9980 | 0.7649 |
| | M3 | 0.8314 | 0.7958 | 0.9976 | 0.7481 |
| mBERT | M1 | 0.4478 | 0.3019 | 0.9479 | 0.2880 |
| | **M2** | **0.8724** | 0.8528 | 0.9980 | 0.7928 |
| | M3 | 0.8456 | 0.8141 | 0.9980 | 0.7667 |
| KoELECTRA | M1 | 0.5005 | 0.3780 | 0.8926 | 0.3942 |
| | **M2** | **0.8717** | 0.8627 | 0.9979 | 0.7667 |
| | M3 | 0.8605 | 0.8384 | 0.9977 | 0.7747 |

**The collapse is format-class-asymmetric — on every encoder.** Under M1, **format-semi-regular** labels stay high (0.88–0.95) because their structural cues are encoded in the labeled tokens themselves, while **format-regular** (0.28–0.38) and **format-free** (0.29–0.39) labels collapse: even regex-tight labels (email, URL, phone) fail without surrounding context, because B-I-O learning requires the context tokens (O) jointly with the answer tokens. M2 restores all classes (Regular 0.85–0.86, Free 0.76–0.79, Semi ≈1.0). *(Fig. 1: grouped bars per format class × mode, averaged over encoders.)*

**Replication.** The ordering (M2 ≥ M3 ≫ M1 on Gold; all-equal on Silver) reproduces in independent runs of the same grid (two additional run sets on different dates, including a re-run 30.6 min on RTX 5070), with run-to-run variance ≤ ±3pp — an order of magnitude smaller than the M1 gap.

## V. Discussion & Conclusion

**Why context matters.** B-I-O training couples each answer token with its surrounding O tokens; answer-only records deprive the model of the positional and structural signals it needs to *locate* entities in running text. Semi-regular labels survive because their cue is inside the labeled span; everything else needs the O-context. **Evaluation implication:** augmented validation data — the standard low-resource practice — completely masked a 37–43pp real-world failure in nine out of nine runs; low-resource studies must include a real-distribution test set. **Annotation implication:** label context, not just answers; adding true negatives is optional (−0.9 to −3.8pp, within noise of M2 in replications).

**Application context.** These findings set the training-data design for the NER stage of a Korean public-domain rights-metadata extraction pipeline deployed by our group (OCR → LLM ∥ NER → LLM-arbitrated consolidation; described and evaluated in the companion paper [Paper 2, this volume]): its NER training data follows the context-inclusive (M2) composition recommended here. Evaluating composition effects end-to-end inside that pipeline is ongoing work and out of scope for this paper.

**Limitations.** Results cover three encoders, one seed (variance bounded by replications), one domain (Korean rights documents), and lenient-match accuracy; a controlled study of context length and end-to-end propagation is future work.

## Acknowledgment
This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) / MSIT, SW Star-Lab / SW-oriented University program (2024-0-00071).

## References (short list — expand to venue style)
[1] Ramshaw & Marcus, "Text Chunking using Transformation-Based Learning," WVLC 1995.
[2] Dai & Adel, "An Analysis of Simple Data Augmentation for NER," COLING 2020.
[3] Park et al., "KLUE: Korean Language Understanding Evaluation," NeurIPS D&B 2021.
[4] Devlin et al., "BERT," NAACL-HLT 2019.
[5] Lample et al., "Neural Architectures for NER," NAACL-HLT 2016.
[6] Park, "KoELECTRA: Pretrained ELECTRA Model for Korean," 2020.
[7] Au et al., "E-NER: An Annotated NER Corpus of Legal Text," NLLP@EMNLP 2022.
[8] Byun et al., "KBMC: Korean Bio-Medical Corpus for Medical NER," LREC-COLING 2024.
[9] Loshchilov & Hutter, "Decoupled Weight Decay Regularization (AdamW)," ICLR 2019.
[10] Clark et al., "ELECTRA: Pre-training Text Encoders as Discriminators," ICLR 2020.
[—] [KCC2026] 황성훈 et al., "한국어 공공저작물 권리문서 메타데이터 추출을 위한 B-I-O 태깅 기반 학습데이터 구성에 따른 KLUE BERT 성능 분석," KCC 2026. (single-encoder precursor)
[—] [Paper 2] [User], 황성훈, and [advisor], "Arbitrating Two Extractors: A Deployed NER–LLM Consolidation Pipeline for Metadata Extraction from Korean Public-Domain Rights Documents," IC-EEECS 2026 (companion paper, this volume).

---
## TODO before submission
- [ ] **황성훈 confirms** the 20260513 uniform grid as the canonical run set (vs 20260427 KLUE / 20260430 mBERT+KoELECTRA replications) and re-generates Fig. 1 from it (`kcc2026paper/paper1/scripts/make_backbone_comparison.py`).
- [ ] **Format per official template** (`paper/eeecs_template/PaperFormat.doc(x)`): TWO-PAGE extended abstract — A4 1-column 2.5cm margins, Times New Roman, title 14pt CAPS, ABSTRACT ~200w @10pt, Keywords ≤5, body 11pt w/ 12pt CAPS section titles, ≤2 pages incl. refs, refs in template style (Surname, F., (Year). Title, Venue, pages — NOT IEEE), PDF submission. Finalize author order/emails.
- [ ] Decide whether to include the substitutive-vs-additive negative ablation sentence (KCC2026 §; −0.72pp true-negative effect vs −1.92pp confounded) — nice but space-limited.
- [ ] Cross-check KCC2026 camera-ready numbers (0427 run: M1 0.4635 / M2 0.8745 / M3 0.8673) are cited consistently as "prior single-encoder result".
- Data/provenance: all numbers extracted from `origin/apiBackup:kcc2026paper/paper1/data/runs/20260513_paper1_{klue,mbert,koelectra}/rule_m{1,2,3}_*/run.txt` (+ `log/scalars.jsonl` for Silver) on 2026-06-29.
