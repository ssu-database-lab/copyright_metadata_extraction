# Data Gap Audit — what we lack (verified)

**Date:** 2026-06-15 · **Method:** 3 parallel auditors → synthesis → adversarial verifier that independently recomputed all 143,946 Excel rows (3 passes) + read the 20-element spec. **All 18 gaps confirmed**; 3 corrections + 3 additions below.
**Constraint:** KCISA provides NO more data (PII). Every gap must be self-sourced (Excel-derive / web-crawl / build-own) or accepted as a documented limitation. See `project_self_reliant_data` memory.

**Bottom line:** We have the **catalog** metadata (~100%) and **partial rights** (73,760 works, 51%). We **lack ground-truth labels / eval sets for nearly every KPI, the real content for non-image modalities, and everything for AI-detection** — none of which KCISA will provide, so all must be self-built.

---

## CRITICAL

1. **Rights labels: missing for ~49%, agency-biased, and degenerate per-right** — rights block filled on exactly 73,760 rows (51.2%), all-or-nothing per row; fill is agency-correlated (~100% 한국문화예술교육진흥원/세종학당/한국학중앙연구원; 0% 제주도/국립국악원/KCISA-self; **국가유산청 9.8%** [corrected, not 0%]). 6 of 7 저작재산권 sub-rights are identical (99.2% Y) → no model beats "always Y"; only 2차적저작물작성권 varies. → **Self-source:** tabular rights predictor on the 73k with **agency-stratified** splits (report out-of-agency accuracy honestly); rule-mine 공공누리 유형→license (note: **5 types incl. 유형9**, not 1–4); publish the 99.2%-Y prior. **Cannot** recover the blank 70k or synthesize absent variation.
2. **Zero real contracts / no contract→rights pairs** — 계약문서 is only a flag (유 21,374 / 무 33,667 / 업무상 14,999). The OCR+LLM+NER Stage-2 path has no training/eval GT. → **Self-source:** synthetic/templated Korean contract corpus (flagged non-certifiable); reframe primary rights estimation as tabular on the 73k. **Cannot** get real-contract eval except user-uploads at inference.
3. **No full document body text (33,094 어문/문서, 23% of corpus)** — public thumbnail is a cover only; PDFs login-gated; 사이트URL on just 6.7%. → **Self-source:** build our own public-domain doc corpus (국립중앙도서관, Gutenberg KR) + Excel text fields. **Cannot** get KOGL PDFs.
4. **No AI-generated vs human labeled dataset** — required for the AI-content-detection deliverable; KOGL has none. (Verifier note: "pre-gen-AI" is loose — some 2023+ uploads exist.) → **Self-source:** fully build-own (generate AI class with SD/FLUX/LLMs; pair vs KOGL as human class) + public benchmarks (GenImage, DiffusionForensics). No KOGL dependency.
5. **KTC/TTA 유형별 1,000건 cert sets — rare types unreachable + no human-verified gold labels** — can't get 1,000 each for 미술 (26)/3D (18)/복합 (57)/글꼴 (464)/동영상 (705); raw KOGL labels are known-wrong. → **Self-source:** crawl category-labeled public-domain works for rare types + a **manual gold-annotation pass** (inter-annotator agreement), or scope the certified taxonomy to reachable types (사진/문서/어문/음성/영상). Human effort mandatory.

## HIGH

6. **No clean, balanced multi-class work-type ground truth (유형분류 90%)** — 분류 is 6 coarse modality buckets; 정보유형 51% filled and 99.8% 사진 for images → classifier collapses to always-사진. → **Self-source:** VLM silver-label 144k + human spot-verify ~2,000 stratified gold; crawl rare classes.
7. **Mandatory rights/author sub-elements sparse/absent** — 유효기간 13.8%, 공동저작자 20%, 저작인접권자 17.3%, 소속 17.3%, **초상권 0.02%**, all person contact PII absent. → **Self-source:** small supervised slices + 저작권만료일 (47.7%) proxy; face-detection proxy for 초상권; contract-NER at inference. **Cannot** backfill PII.
8. **No reference 설명 (description) to score the attribute KPIs (85%)** — keyword GT is only noisy 해시태그. → **Self-source:** human-write ~1,000/type gold descriptions (doubles as cert set) OR LLM-judge rubric instead of exact-match.
9. **No labeled similarity/relevance pairs (유사도 94%)** — no human-judged GT, no query set, no cross-domain labels. → **Self-source:** silver positives from 주제어/제목 + ~300 human-judged gold queries. Cross-domain blocked by missing audio/video.
10. **Audio/video media + missing ASR/video models (멀티모달 85% / 유사도 94%)** — no ASR/audio-tagging/video models in stack; ~47% audio have no media file; video login-gated. → **Self-source:** harvest ~14k public MP3s via 게시글URL + add Whisper/keyframe sampling; scope video as a documented limitation.

## MEDIUM

11. **No UCI / SHA256 / version columns (중복검출 92% + KTC 시험결과서)** → compute SHA256 + perceptual-hash on the 144k thumbnails (satisfies KTC field + thumbnail-dedup); assign own version; derive 디지털화형태 from MIME. **Cannot** mint canonical UCI (needs registry).
12. **No labeled near-duplicate benchmark, esp. hard near-dups (중복검출 92%)** → synthesize near-dup positives by augmenting thumbnails (defines GT) + same-제목 series; exclude logo-collisions. Fully self-buildable.
13. **Only 330×220 thumbnails — no high-res originals (속성/OCR/중복검출)** → 144k thumbnails now; higher-res for text-heavy from Wikimedia/Europeana duplicates. **Cannot** get KOGL full-res.
14. **No 3D model files (1,478 works)** → scrape ~291 public WebGL assets / render multi-views. ~80% have only a single render.
15. **Headline trainable model lost its label basis (AI 모델 + SW 등록)** → train rights-flag/license classifiers on the 73k (agency-stratified) + CLIP/SigLIP contrastive fine-tune on thumbnails+주제어 (a documented null result is still a valid deliverable).

## LOW

16. **No font binaries (.ttf/.otf, 464 works)** → collect from ~67% institution font pages + fonttools extractor.
17. **Trivially-derivable catalog fields 0%/sparse** (디지털화형태·저작권기관명·언어·제작일) → derive from 원본파일명 ext / 원본소유자 / script detection / 촬영년도. Low effort.

## Verifier-added gaps (auditors missed)

18. **Global/multilingual mandate + 영문명 element have ~zero data** — 언어 is 97% Korean; 영문명 absent. Yet ~5 self-source plans depend on **unvalidated machine back-translation** → validate back-translation quality before relying on it.
19. **No OCR/transcription ground truth** — the whole pipeline sits on OCR, but there's no GT to bound OCR accuracy. → build a small human-corrected OCR eval set.
20. **Spec/registration technical fields absent** — 수량(quantity), 용량(KB file size), media duration/resolution/bitrate not in the export (low; derivable from fetched files).

---

## Self-source verdict summary

| Can self-build fully | Partial / documented limitation | Genuinely cannot |
|---|---|---|
| AI-gen dataset, near-dup benchmark, SHA256/perceptual-hash, synthetic contracts, derivable catalog fields, silver type/keyword labels, fine-tune set | Document corpus (own sources), audio (harvest ~53%), gold annotation (human effort), rights prediction (agency-biased), high-res (some via duplicates) | KOGL full-res originals, KOGL PDFs, canonical UCI, contact PII, true per-work rights for blank 70k, real-contract eval, video (login-gated) |

**Strategic implication:** the biggest credibility risk is **honest KPI measurement** — several "90%/94%" targets currently have no trustworthy gold set, and the rights 51% is agency-biased. Priority self-build order: (1) gold eval/cert sets (human annotation, agency-stratified) — unblocks honest measurement of nearly every KPI; (2) AI-gen dataset (fully self-buildable, own deliverable); (3) audio harvest + ASR; (4) document corpus.
