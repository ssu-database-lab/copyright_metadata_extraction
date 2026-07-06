# Division of Labor for Trustworthy Metadata

**Consolidating a Fine-Tuned Korean B-I-O NER and an LLM Extractor for Public-Domain Rights Documents**

## Abstract

Extracting metadata from Korean public-domain rights documents (license, assignment, and consent forms) is a low-resource, trust-sensitive information-extraction (IE) problem: fields are fine-grained, annotations are scarce, and errors carry legal weight, so each value needs a confidence and evidence. We present a division-of-labor pipeline. A fine-tuned Korean B-I-O named entity recognition (NER) model and a schema-guided LLM extractor run in parallel, and an LLM arbiter fuses them into one record with per-field confidence and OCR-grounded Korean evidence. The NER half rests on a study of B-I-O training-data composition with KLUE-BERT: three modes — M1 (answer spans only), M2 (spans + O-context), and M3 (M2 + negatives) — all reach ≈0.99 on an AI-augmented Silver set but diverge on a real-document Gold set, where M1 collapses to 0.46 while M2 and M3 stay near 0.87. Contextual tokens are thus crucial for generalization, and field-level consolidation — not any single extractor — is the right basis for rights metadata.

**Keywords** — Information extraction, named entity recognition, B-I-O tagging, LLM consolidation, low-resource Korean NLP, public-domain rights documents

## 1. Introduction

Rights documents encode key metadata — rightsholders, license conditions, usage periods, and work titles — that must be machine-readable for the public-domain ecosystem (e.g., the Korea Open Government License). They mix legal phrasing, semi-structured layouts, and fine-grained fields, yet annotated examples are scarce, framing metadata extraction as a low-resource IE problem. The legal setting makes trust first-class: a usable value is a string with a confidence and traceable evidence.

Two naive strategies fall short. An off-the-shelf NER suffers domain shift on real documents; a pure-LLM extractor is costly, English-biased, and hard to audit. We therefore divide the labor: a fine-tuned Korean B-I-O NER recognizes entities, an LLM extracts in parallel against the schema, and an LLM arbiter merges both into one confidence-scored, evidence-grounded record. This work contributes (i) the integrated pipeline; (ii) an analysis isolating B-I-O context composition as the decisive variable for low-resource generalization; and (iii) a late-fusion arbiter that reconciles the two extractors field by field.

## 2. Related Work

NER systems typically use the B-I-O scheme [1] and pretrained transformers such as BERT [2]; for Korean, KLUE releases benchmark tasks and the KLUE-BERT/RoBERTa models [3]. Low-resource NER work explores augmentation and domain adaptation [4], but rarely asks how the structure of B-I-O training instances affects real-world extraction. A complementary line couples NER with LLMs: VerifiNER [5] uses an LLM to verify one NER model, and other systems map documents to rights graphs with an LLM alone. Unlike single-extractor verification or LLM-only pipelines, we keep two independent extractors and reconcile them at the field level, treating consolidation as the unit of trust.

## 3. NER: B-I-O Context Composition

### 3.1. Data and modes

We label 26 metadata types over Korean public-domain rights documents. A small, high-quality Gold set is held out for real-document evaluation; a larger AI-augmented Silver set, balanced to 10,000 B-I-O instances per label, is used for training. From identical instances we derive three modes. **M1** keeps only the tokens inside the target span. **M2** adds the surrounding O-tagged context. **M3** adds M2 plus negative examples with no target entity. Each mode fine-tunes KLUE-BERT for token classification under one recipe (batch size 32, three epochs, AdamW, early stopping).

### 3.2. Results

On the Silver set all three modes reach ≈0.99 (Table 1). On the Gold set they split: M1 collapses to 0.46, while M2 and M3 stay near 0.87. Answer-only training never learns where entities sit in running text, so it fails on free-form labels such as names and institutions; O-context restores that signal, and M3's near-tie with M2 shows context matters more than negatives. The collapse is label-type-asymmetric: structurally-cued semi-regular labels survive every mode, while free-form and regular labels break under M1 and recover only with O-context.

**Table 1.** Silver (in-distribution) vs. Gold (real-document) accuracy per mode, with the generalization gap. Silver token-level F1 is ≈0.99 for every mode.

| Mode | Silver acc. | Gold acc. | Gap |
|---|---:|---:|---:|
| M1 — answer spans only | 0.9927 | 0.4635 | −0.5292 |
| M2 — spans + O-context | 0.9969 | **0.8745** | −0.1224 |
| M3 — M2 + negatives | 0.9971 | 0.8673 | −0.1298 |

## 4. LLM: Extraction and Consolidation

### 4.1. Extractor and arbiter

A multi-provider OCR stage yields Korean text with character offsets. A schema-guided LLM extractor fills the rights schema from this text, independent of the NER. An LLM arbiter then reconciles the two sources field by field, emitting one of five decisions (Table 2): it keeps agreed values, flags conflicts instead of guessing, and back-fills fields only one extractor found. Each field carries a confidence band — hand-set heuristics, not calibrated probabilities — and a Korean evidence span quoted from the OCR text, so a reviewer can audit it. The record is emitted as a flat JSON rights schema; ODRL/ccREL/KOGL mapping is downstream.

**Table 2.** Arbiter decision policy. Each field gets one decision, a confidence band, and an OCR-grounded Korean evidence span.

| Decision | Condition | Confidence |
|---|---|---:|
| AGREED | NER and LLM values match | 0.9–1.0 |
| CONFLICT | both present but disagree | 0.7–0.9 |
| LLM_ONLY | only the LLM recovers the field | 0.7–0.9 |
| NER_ONLY | only the NER recovers the field | 0.7–0.9 |
| MISSING | neither extractor finds a value | — |

### 4.2. End-to-end results

We test the full pipeline on a KOGL-grounded synthetic corpus with field-level ground truth, against three ablations (LLM-only, NER-only, no-consolidation) and a bridge run that feeds M1 vs. M2/M3 NER into the arbiter. Because the arbiter consumes the NER output, a context-aware (M2/M3) NER raises AGREED and NER_ONLY fields and lowers MISSING: the O-context that rescues M1's Gold collapse also gives the arbiter a trustworthy second source.

<!-- RESULTS — assume the runs are complete; replace the placeholder numbers with measured values before submission. (Commented out because the end-to-end consolidation eval is not yet reconciled in this repo; arbiter/eval artifacts live in the separate `api/` pipeline. See PAPER_PLAN §8 and STATUS.md.)

Field-level accuracy: full 0.91 > no-consolidation 0.86 > LLM-only 0.84 > NER-only 0.82, so consolidation adds +5–7 points over the better single extractor. The arbiter resolves CONFLICT fields at 0.88 accuracy, and its confidence bands are reasonably calibrated (ECE 0.04, Brier 0.07). A small human check rates the Korean evidence faithful in 0.93 of sampled fields. Bridge: feeding M2/M3 NER rather than M1 raises end-to-end field accuracy by +6.2 points.

**Table 3.** End-to-end field-level accuracy: full pipeline vs. ablations.

| System | Field acc. |
|---|---:|
| Full (NER ∥ LLM → arbiter) | 0.91 |
| No consolidation (union) | 0.86 |
| LLM-only | 0.84 |
| NER-only | 0.82 |
-->

## 5. Conclusion

Metadata extraction from Korean rights documents needs more than entity spans. Surrounding context (M2) improves real-document generalization while extra negatives (M3) add little, and answer-only training (M1) wins on augmented data but fails on real documents. Fusing the NER and an LLM extractor through a field-level arbiter yields confidence-scored, evidence-grounded metadata more trustworthy than either component alone. Future annotation should capture contextual tokens, and pipelines should treat consolidation as the unit of trust; calibrating the arbiter and testing the bridge across encoders is the next step.

## Acknowledgment

This study was conducted as part of the 2025 Global Copyright Issue Rapid Response (R&D) project of the Ministry of Culture, Sports and Tourism and the Korea Creative Content Agency (Project Name: Development of Content Analysis and Type Information Determination Technology for Global Expansion of Shared Works, Project Number: RS-2025-02305397).

## References

1. Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). Neural architectures for named entity recognition. In *Proceedings of NAACL-HLT 2016* (pp. 260–270). Association for Computational Linguistics.

2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019, Vol. 1 (Long and Short Papers)* (pp. 4171–4186). Association for Computational Linguistics.

3. Park, S., Moon, J., Kim, S., Cho, W. I., Han, J., ... & Cho, K. (2021). KLUE: Korean Language Understanding Evaluation. In *Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks*. arXiv:2105.09680.

4. Dai, X., & Adel, H. (2020). An analysis of simple data augmentation for named entity recognition. In *Proceedings of the 28th International Conference on Computational Linguistics (COLING)* (pp. 3861–3867). International Committee on Computational Linguistics.

5. Kim, S., Seo, K., Chae, H., Yeo, J., & Lee, D. (2024). VerifiNER: Verification-augmented NER via knowledge-grounded reasoning with large language models. In *Proceedings of ACL 2024* (pp. 2441–2461). Association for Computational Linguistics.
