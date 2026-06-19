# IC-EEECS 2026 — Short Paper Draft (2 pages)

> **Status:** DRAFT scaffold. Prose is written; **numbers in `[FILL: …]` come from 황성훈's multi-encoder NER experiments** (KLUE-BERT exists from KCC2026; mBERT / XLM-R / KoELECTRA pending GitHub push). Target: 2-page IEEE-style short paper, "Artificial Intelligence and Applications" track. Camera-ready due 2026-07-10.
> **Authorship:** 황성훈 (lead — NER experiments), user (co-author — pipeline framing/forward pointer). Balances the journal, where the user leads.

---

## STORYLINE (one paragraph — the spine of the paper)

Korean public-domain rights documents (license/assignment/consent forms) are a **low-resource NER** problem: fine-grained labels, little gold data. KCC2026 found, on a single model (KLUE-BERT), that **how the B-I-O training context is composed** — answer spans only (M1) vs. answers + surrounding context (M2) vs. + negative samples (M3) — is decisive: an **AI-augmented (Silver) test set hides all differences, but a real-document (Gold) test set exposes a sharp M1 collapse, with M2≈M3**, and the collapse is **label-type-asymmetric** (free-form and regular labels break under M1; structurally-cued semi-regular labels survive). The open question — and KCC2026's stated limitation — is **whether this is a property of one model or a general phenomenon.** This paper answers it: we **reproduce the effect across multiple Korean encoders** (KLUE-RoBERTa, mBERT, XLM-RoBERTa), showing the Silver-hides / Gold-exposes pattern and the label-type asymmetry are **encoder-independent**, not a KLUE-BERT artifact. The practical payload: in low-resource Korean IE, **annotators should label surrounding context (O), not just answers**, and **adding negatives buys little over context**. We close with a one-paragraph pointer to the integrated NER→LLM→consolidation pipeline this NER feeds (the journal).

## SECTION MAP (2-page budget)


| §   | Section                 | ~space  | content                                                                                                                        |
| --- | ----------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------ |
| I   | Introduction            | 0.6 col | low-resource Korean rights NER; the M1/M2/M3 question; KCC2026 single-model gap; **3 contributions**                           |
| II  | Related Work            | 0.4 col | 1 compressed paragraph: BIO, augmentation, domain shift, Korean NER                                                            |
| III | Method                  | 0.8 col | Gold/Silver data, 26 labels, M1/M2/M3 defn, Free/Regular/Semi-Regular taxonomy, encoders, fixed recipe                         |
| IV  | Results                 | 1.2 col | Table I Silver (near-identical); Table II Gold (M1 collapse, M2≈M3); Fig 1 per-label-type asymmetry; cross-encoder consistency |
| V   | Discussion & Conclusion | 0.5 col | why O matters; annotation guidance; pipeline forward pointer; limitations                                                      |
| —   | Ack + Refs              | 0.3 col | grant 2024-0-00071; ~10 refs                                                                                                   |


---

# Title

**Does Context Composition Generalize? A Multi-Encoder Study of B-I-O Training Context for Low-Resource Korean Rights-Document NER**

*(Alt: "Outside Tokens Matter Across Models: B-I-O Training-Data Composition in Low-Resource Korean Rights-Document NER")*

**Authors:** 황성훈¹, [User]¹, [advisor]¹  ¹School of Computer Science and Engineering, Soongsil University, Seoul, Korea
{emails}

---

## Abstract

*(~120 words)*
Extracting metadata from Korean public-domain rights documents (license agreements, assignment and consent forms) is a low-resource named-entity-recognition (NER) task with fine-grained labels and scarce gold annotations. Prior work on a single Korean encoder showed that the composition of B-I-O training data—whether it contains only answer spans (M1), answers with surrounding context (M2), or additional negative samples (M3)—strongly affects real-world extraction, but left generalization to a single model unverified. We reproduce the study across [FILL: N] Korean encoders (KLUE-RoBERTa, multilingual BERT, XLM-RoBERTa). Across all encoders, an augmented (Silver) test set conceals the differences while a real-document (Gold) test set exposes a sharp M1 degradation with M2≈M3, and the degradation is label-type-asymmetric. The effect is therefore encoder-independent. We conclude that, for low-resource Korean information extraction, training data should include surrounding context, and that this NER feeds a downstream NER–LLM consolidation pipeline.

**Keywords:** named entity recognition, B-I-O tagging, low-resource Korean NLP, training-data composition, information extraction, public-domain rights documents

---

## I. Introduction

Public-domain works in Korea (under the Korea Open Government License, KOGL) are governed by rights documents—이용허락 계약서 (license agreements), 양도동의서 (assignment forms), and 개인정보 수집·이용 동의서 (personal-information consent forms)—whose metadata (rightsholders, license type, term, usage conditions) must be extracted to make the works machine-actionable. This is a **low-resource NER** problem: the entities are domain-specific and fine-grained, and annotated data is scarce.

A recurring practical question in low-resource NER is **how to compose the B-I-O training data**: should each training example contain only the target answer spans, or also the surrounding context, or additional examples with no answer at all? On a single Korean encoder (KLUE-BERT), prior work [KCC2026] reported a striking result: three composition modes—**M1** (answer-only), **M2** (answer + surrounding context), and **M3** (M2 + ~25% negative samples)—are **indistinguishable on an AI-augmented test set but diverge sharply on a real-document test set**, where M1 collapses while M2 and M3 are comparable; moreover the collapse depends on label type. That study's explicit limitation was that **only one model was tested**, leaving open whether the phenomenon is general or a model artifact.

**This paper closes that gap.** Our contributions:

1. We **reproduce the M1/M2/M3 context-composition study across [FILL: N] Korean encoders** (KLUE-RoBERTa, multilingual BERT, XLM-RoBERTa[, KoELECTRA]) on the same Gold/Silver rights-document data.
2. We show the **Silver-hides / Gold-exposes pattern and the M1 collapse (with M2≈M3) are encoder-independent**, and that the degradation is **label-type-asymmetric**, captured by a Free / Regular / Semi-Regular label taxonomy.
3. We distill **actionable guidance** for low-resource Korean IE annotation (label surrounding context; negatives add little), and situate the NER within a downstream NER–LLM consolidation pipeline.

## II. Related Work

The B-I-O (IOB) tagging scheme recasts NER as token-level sequence labeling [Ramshaw&Marcus 1995], the basis for neural [Lample 2016] and pretrained-transformer NER, including Korean encoders [KLUE 2021]. In low-resource settings, data augmentation is common [Dai&Adel 2020], and LLM-assisted labeling has improved Korean specialized-domain NER (e.g., medical [KBMC 2024]). Applying general-domain NER to specialized text causes severe degradation—up to 29–60% F1 on legal text [E-NER 2022]—motivating domain-specific data and careful evaluation. Unlike prior work, we isolate **training-context composition** as the variable and test its **cross-encoder** behavior on Korean rights documents, rather than proposing a new model or augmentation method.

## III. Method

**Data.** We use two datasets over **26 rights-metadata labels** in JSONL B-I-O format. The **Gold** set (~~[FILL: 50,000] tokens/items; web-crawled + provided real license/consent documents) is held out as a **real-proof** evaluation set. The **Silver** set is produced by AI augmentation [Dai&Adel 2020] (~~5× expansion, class-balanced), capped at **10,000 instances per label** and split **8/2/2** (train [FILL: 6,666] / test [FILL: 1,666] / validation [FILL: 1,668]).

**Context-composition modes.** From the same source we build three training variants: **M1** = answer spans only (tokens are B/I; minimal O); **M2** = answers embedded in surrounding context (natural O tokens around answers); **M3** = M2 plus ~25% negative samples (sentences with no target entity).

**Label-type taxonomy.** We group the 26 labels into **Free** (free-form surface, e.g., company name, job title), **Regular** (rigid format, e.g., phone, e-mail), and **Semi-Regular** (strong structural/textual cues, e.g., legal references, work keywords). *(Full assignment in the appendix / journal.)*

**Encoders & recipe.** We fine-tune [FILL: N] Korean/multilingual encoders—**KLUE-RoBERTa-large, multilingual BERT (mBERT), XLM-RoBERTa-large**[, KoELECTRA]—each as B-I-O token classifiers under an **identical recipe** (full fine-tuning, AdamW, batch size 32, 3 epochs, early stopping). Hardware: Intel Core Ultra 5 225F, 64 GB RAM, NVIDIA RTX 5070, Ubuntu 24.04. We evaluate each mode (M1/M2/M3) on **both** the Silver test split and the Gold real-proof set, reporting token-level Precision/Recall/F1 (and per-label-type accuracy).

## IV. Results

**Silver evaluation hides the differences.** On the augmented Silver test set, M1, M2, and M3 are near-identical for every encoder (Table I): all P/R/F1 within [FILL: ±X.X] points.

**Table I. Silver test set — overall token-level P / R / F1 (%).**


| Encoder       | M1 (P/R/F1)          | M2 (P/R/F1) | M3 (P/R/F1) |
| ------------- | -------------------- | ----------- | ----------- |
| KLUE-RoBERTa  | [FILL]/[FILL]/[FILL] | …           | …           |
| mBERT         | …                    | …           | …           |
| XLM-RoBERTa   | …                    | …           | …           |
| *(KoELECTRA)* | …                    | …           | …           |


**Gold evaluation exposes the M1 collapse — across all encoders.** On the real-document Gold set, **M1 degrades sharply while M2≈M3**, for *every* encoder (Table II) — the effect is **not specific to KLUE-BERT**.

**Table II. Gold (real-proof) set — overall token-level F1 (%), Δ vs. M2.**


| Encoder       | M1     | M2     | M3     | M2−M1             |
| ------------- | ------ | ------ | ------ | ----------------- |
| KLUE-RoBERTa  | [FILL] | [FILL] | [FILL] | **[FILL: large]** |
| mBERT         | [FILL] | [FILL] | [FILL] | **[FILL: large]** |
| XLM-RoBERTa   | [FILL] | [FILL] | [FILL] | **[FILL: large]** |
| *(KoELECTRA)* | …      | …      | …      | …                 |


**The collapse is label-type-asymmetric.** Breaking M1's Gold degradation down by label type (Fig. 1): **Free** and **Regular** labels drop steeply under M1, while **Semi-Regular** labels—whose structural cues survive without surrounding context—remain high. This pattern holds across encoders, indicating the asymmetry is a property of the *task/label structure*, not the model.

**Fig. 1.** Per-label-type accuracy by mode (M1/M2/M3), averaged over encoders: Free ↓↓, Regular ↓↓, Semi-Regular ≈ stable under M1. *[insert grouped bar chart from 황성훈's results]*

**Cross-encoder consistency.** The rank order (M2≈M3 > M1 on Gold; all-equal on Silver) is identical across the [FILL: N] encoders, with [FILL: e.g., consistent direction; optionally a significance/variance note]. We therefore find the context-composition effect **encoder-independent** within the models tested.

## V. Discussion & Conclusion

Two takeaways. **(1) Evaluation:** augmented test data can completely mask a model's real-world failure mode—**M1 looks fine on Silver and fails on Gold**—so low-resource studies must evaluate on real documents. **(2) Annotation:** the surrounding-context (O) tokens carry the signal that lets a model locate free-form and rigidly-formatted entities in running text; **labeling context (M2) is what matters, while extra negatives (M3) add little.** The label-type asymmetry explains *why*: semi-regular labels carry their own structural cues, so they survive context-free training, whereas free/regular labels rely on contextual position.

These results generalize KCC2026's single-model finding to multiple Korean encoders, establishing the context-composition effect as a robust, encoder-independent property of low-resource Korean rights-document NER. This NER component is the recognition stage of a larger **NER–LLM consolidation pipeline** that pairs it with a schema-guided LLM extractor and an LLM arbiter producing confidence-scored, evidence-grounded metadata; that integrated system is the subject of an extended journal version. **Limitations:** results cover the tested encoders and one rights-document corpus; a controlled study of context *quantity* and of downstream propagation into consolidated metadata is left to the journal.

## Acknowledgment

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) / MSIT, SW Star-Lab / SW-oriented University program (2024-0-00071).

## References (short list — expand to venue style)

[1] Ramshaw & Marcus, "Text Chunking using Transformation-Based Learning," WVLC 1995.
[2] Dai & Adel, "An Analysis of Simple Data Augmentation for NER," COLING 2020.
[3] Park et al., "KLUE: Korean Language Understanding Evaluation," NeurIPS D&B 2021.
[4] Devlin et al., "BERT," NAACL-HLT 2019.
[5] Lample et al., "Neural Architectures for NER," NAACL-HLT 2016.
[6] Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)," ACL 2020.
[7] Au et al., "E-NER: An Annotated NER Corpus of Legal Text," NLLP@EMNLP 2022.
[8] Byun et al., "KBMC: Korean Bio-Medical Corpus for Medical NER," LREC-COLING 2024.
[9] Loshchilov & Hutter, "Decoupled Weight Decay Regularization (AdamW)," ICLR 2019.
[10] Prechelt, "Early Stopping — But When?," in Neural Networks: Tricks of the Trade, 2012.
[—] [KCC2026] 황성훈 et al., "한국어 공공저작물 권리문서 메타데이터 추출을 위한 B-I-O 태깅 기반 학습데이터 구성에 따른 KLUE BERT 성능 분석," KCC 2026. (self-cite of the single-model precursor)

---

## TODO before submission

- [ ] **Insert real numbers** from 황성훈's multi-encoder runs (after GitHub push) — Tables I, II, Fig. 1; resolve the `[FILL]` markers and the abstract's `N` encoders.
- [ ] **Verify the runs** are clean (the repo's current `eval_results_*.json` are 0.0-F1 with a mismatched `test_data_path` — do NOT use those; see PAPER_PLAN_AND_FINDINGS.md §0).
- [ ] Confirm IC-EEECS template (likely IEEE 2-column) + exact page/format limits; trim to 2 pages.
- [ ] Finalize author order/emails; confirm 황성훈 as lead.
- [ ] Add a small significance/variance note if seeds/multiple runs are available.
- [ ] Optional: 1 schematic figure of the downstream pipeline (if space) to stake journal priority.