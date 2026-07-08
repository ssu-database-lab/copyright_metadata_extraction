# Task Status & Blockers — Year 2 Multimodal Work

**As of:** 2026-05-29 · **Basis:** the KOGL 144k Excel (`붙임1_원문저작물 메타데이터.xlsx`) + 장세영 email (2026-05-28) + current assets
**Context:** 장세영's 2026-05-28 email confirms the data requests are formally in motion (무하유 → KCISA), and adds a new ask — **구분 명세서** (metadata field-definition doc). No new data attached (same 144k export).

---

## TL;DR

- **Most Stage-1 work and all the prep/eval scaffolding can proceed now** with the Excel + existing assets.
- **Stage-2 (rights from contracts) is genuinely blocked** on actual 계약서/동의서 files and the 구분 명세서.
- **The single highest-leverage unknown:** are KOGL thumbnails/originals publicly fetchable via the URLs in the Excel? If yes, the multimodal eval is unblocked TODAY (not "after KCISA replies").

---

## ✅ CAN DO NOW

### A. Needs only the Excel
1. **Stratified KOGL eval manifest** (task #8) — sample N per `정보유형` (사진 29,291 / 문서 17,818 / 음성 14,212 / 어문 10,287 / 영상 1,338 / 동영상 705 / 미술 26 / 3D 18); include `게시글URL` + `썸네일웹경로` + ground-truth labels (정보유형·주제어·rights flags). xlsx → CSV. Foundation for downstream benchmarks.
2. **Stage-1 catalog auto-generation validation** — compare our pipeline outputs (제목·유형·키워드·설명·디지털화형태·hash) against KOGL catalog fields (~100% filled in the Excel). Yields a Stage-1 accuracy number with no new data.
3. **Train rights-flag classifiers** — labels present at scale on rights-complete subset: `상업적이용허락` (가능 57.8k / 금지 11.9k / 불가 3.4k, 3-class), `비보호저작물` (보호 63.5k / 만료 9.7k = our `unprotected_work` element). SigLIP image features or text(제목·주제어) + head.
4. **Refine VLM prompt** — align work_type list to KOGL's actual `정보유형` vocabulary (사진/문서/음성/어문/영상/동영상/미술/3D/복합).

### B. Needs only public web (likely no blocker — TEST FIRST)
5. **Fetch KOGL thumbnails** via `썸네일웹경로` (`/upload_recommend/thumb_L/…`) — probably public assets on kogl.or.kr. If public → ~144k image URLs instantly = partial unblock of the multimodal eval.
6. **Scrape `게시글URL` pages** — public KOGL pages may expose the original image.

### C. Pure code (no data dependency)
7. **Pipeline integration** — modality router (file-type dispatch) into `PipelineOrchestrator`; VLM/embedding/NER plumbing.
8. **Unified-schema mapping** — function: VLM/embedding outputs → 67-field schema (work_type, keyword, description, digital_format, hash/UCI).
9. **Embed-benchmark v2** — harder perturbation/semantic-near-duplicate set so SigLIP/KoCLIP actually separate (current 8-img test = all 100%, can't differentiate).

### D. User action (not data-blocked, pending you)
10. **Re-establish Gemma access** (cloudflared quick OR Tailscale permanent) — needed for VLM-side eval.
11. **KTC/TTA cert form scheduling** (유인재·유석 visit; not Wednesday).

---

## ❌ CANNOT DO (blocked on data)

> **⚠️ UPDATE 2026-06-08: KCISA will NOT provide any more data (PII/privacy).** Nothing below is "waiting on KCISA" — all of it must be **self-solved** from the Excel + web crawling + our own dataset building. See `project_self_reliant_data` memory.

| Task | Was blocked by | Self-reliant solution (no KCISA) |
|---|---|---|
| Stage-2 권리정보 추정 | contracts (not coming) | The Excel ALREADY has rights for **73,760 works** (저작권자명·권리·유효기간·상업적이용·비보호 ~51%). Reframe: **predict rights from metadata/owner/license patterns** learned on the 73k filled rows (tabular ML / rule-mining), not contract-NER. Contract-NER only when a user uploads a contract. |
| Full-resolution multimodal eval | originals (not coming) | Use 144k public thumbnails (fetchable). Fine-print OCR limited — accept, or crawl higher-res from public sources. |
| Field-level schema audit / 구분 명세서 | spec (not coming) | Infer field semantics ourselves from the Excel data + 붙임2 guide + public KOGL site. |
| VLM prompt work_type tuning (logo, text-scan) | 구분 명세서 (not coming) | Decide our OWN convention (document it), since no official spec is coming. Revisit the deferred `vlm/prompts.py` rules. |
| Decode 대/중/소분류 codes | codebook (not coming) | Reverse-engineer codes from the Excel (code↔제목↔분류 co-occurrence) + crawl KOGL site category names. |
| work_type classifier labels | richer labels (not coming) | Train on `분류` column (well-distributed) or VLM silver labels, or derive from file medium; augment with crawled type-labeled images. |

---

## 🧭 SELF-RELIANT DATA STRATEGY (new primary direction)

1. **Excel-derived:** rights (73k labeled), type (`분류`), keywords (주제어), code reverse-engineering.
2. **Web crawl:** KOGL public thumbnails (144k) + other public-domain sources (Wikimedia Commons KR, Europeana, Project Gutenberg); **back-translate non-Korean captions** to Korean.
3. **Our own dataset:** curated/labeled sets for type classification, CLIP fine-tuning (thumbnails + 주제어/제목 captions), and rights pattern-learning.

## 🚧 REMAINING BLOCKING POINTS (real, non-KCISA)

1. **Logo/text-scan work_type convention** — now OUR decision to make (no spec coming); document it.
2. **(none data-blocked anymore)** — Gemma access solved (Tailscale); thumbnails fetchable; rights data in-hand.

---

## ▶ RECOMMENDED SEQUENCE (all self-reliant)

1. ✅ Probe KOGL public access — DONE (thumbnails fetchable).
2. ✅ Eval manifest + diverse eval — DONE.
3. **Pipeline integration** (router + schema_mapping + VLM extractor + embedding into PipelineOrchestrator).
4. **Self-built datasets** — type-label set (분류/silver), CLIP fine-tune set (thumbnails+주제어), rights pattern-learning set (73k Excel rows).
5. ✅ Gemma access — DONE (Tailscale).

---

## Related docs
- Canonical data analysis: `docs/analysis_muhayu_kcisa_20260524.md`
- Data request to KCISA: `docs/data_request_to_kcisa_20260524.md` (note: add 구분 명세서 to the list)
- Year 2 design + roadmap: `docs/YEAR2_MULTIMODAL_DESIGN.md`
- Meeting brief: `docs/meeting_20260526_progress.md`
