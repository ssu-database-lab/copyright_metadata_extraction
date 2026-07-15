# 무하유 kogl-classifier ↔ 숭실대 파이프라인 정합성 점검 (2026-07-10)
_소스: github lemonsnack-rgb/kogl-classifier (public, 2026-04-22 push) + 라이브 Vercel 번들 + 우리 API 계약. 4-agent workflow._

---

# Alignment Verdict: Muhayu kogl-classifier ↔ SSU Extraction Pipeline

## 1. Their flow vs our flow

| Step | kogl-classifier (Muhayu) | Our /v2 UI (SSU) | Match? |
|---|---|---|---|
| Entry | `/works/new` 3-step wizard: 검사 명칭 + 문서 유형 + contract file (required) + consent file (optional) (`src/app/works/new/page.tsx:25-29`) | Step 1 파일 업로드: **single file**, no title/grouping concept (`api/web/templates/index2.html:1741`) | **Diverge** — theirs is a multi-document "검사" unit; ours is one file per request |
| Work upload | Step 2: multi-file drag&drop, accept `.pdf,.jpg,.jpeg,.png,.tiff,.tif` only (`works/new/page.tsx:533`) — stored to Supabase `works` bucket, `ocr_status='pending'`, **never sent to any API** (`works/new/page.tsx:204-222`) | Same file types (+webp/gif/bmp server-side, `app.py:97`); images route to VLM track (`pipeline.py:277-315`) | **Diverge** — they collect works but analyze nothing; we can analyze images but they never call us for them |
| Model config | None (fixed pipeline in HF Space) | Steps 2-3: OCR provider/model, LLM, NER, `vlm_prefer`, consolidation toggle | Diverge (cosmetic — our params all have defaults their pipeline can omit) |
| Analysis | Fire-and-forget browser POST to `https://ilwang-kogl-pipeline.hf.space/process` (multipart `file, contract_id, document_type, file_name`, `works/new/page.tsx:228-240`); HF Space then calls SSU `/api/llm-extract` → HMC `/api/predict` → writes Supabase | Direct SSE call to `/api/llm-extract` (OCR → LLM ∥ NER → consolidation) | **Match on the contract leg** — our API is stage 1 of their pipeline. The HF Space (separate repo, not audited) is the actual integration point |
| Result | Per-contract: 공공누리 KOGL-1..4 + confidence + evidence clauses (HMC) + contract metadata JSONB (us) + a 20-field per-work table that renders **"미식별"** for everything (`works/[id]/page.tsx:1346-1441`) | 67-field unified metadata + consolidation decisions; **no KOGL classification anywhere in our API** (`kogl_type` is a passive never-guess field, `cloud_extractor.py:46`) | **Partial** — contract metadata flows; KOGL is HMC's job (correct division of labor); the 20 work fields are dead on both ends |

Live-site check: kogl-classifier.vercel.app talks only to Supabase (`eodxyyyzvxoemryrmpna.supabase.co`); no `150.230.114.9`, no `/api/llm-extract` in any public bundle. This is consistent with the repo — the browser only hits Supabase + the HF Space; our URL lives in the HF Space / Vercel env, not client code. No contradiction, but it means **we cannot see or verify our own integration from either the frontend repo or the live site — the HF Space `ilwang-kogl-pipeline` repo is the missing artifact.**

## 2. API contract alignment

**Yes, they call our API — indirectly.** Two call sites consume `POST {SSU}/api/llm-extract`:
- Legacy (superseded): `src/app/api/pipeline/process/route.ts:78` — multipart `file, document_type, consolidate:"true"`; reads `ocr_text`, `consolidated_metadata || metadata`, `processing_time`.
- Live: the HF Space (per commit `3226ee4` "파이프라인을 HF Spaces로 이전"), presumably the same call.

**Param compatibility: fully compatible.** They send 3 of our 10 Form params (`app.py:836-847`); the other 7 (`model_name`, `ocr_provider`, `ner_model`, `vlm_prefer`, etc.) default sensibly. No unknown params sent.

**Response compatibility: compatible for what they read.** `ocr_text`, `consolidated_metadata`, `metadata`, `processing_time` all exist in `build_response` (`api/web/pipeline.py:321-366`). Their typed `SSUExtractResponse` (`src/lib/api/types.ts`) including `consolidation_decisions` matches our shape. Gaps they don't know about: `modality`/`vlm_backend`/`vlm_raw` (image path), `evidence` inside decision items — absent from our own `docs/API_명세서.md` (stale, 2026-03-18) too.

**Field-name mismatches (their `FIELD_LABELS` / `Work` interface vs our 67-field schema, `document_schemas.py:1174`):**

| Their key | In our 67-field schema? |
|---|---|
| `work_names` (labeled 저작물명, commit-recent) | **NO** — ours is `work_title` (singular). If their UI shows this label populated, the HF Space is renaming keys, or they're labeling a key we never emit |
| `integrity_right_waiver` (동일성유지권 포기) | **NO** — closest is `special_terms`/`contract_terms` free text |
| `modification_allowed` (변경 허용) | **NO** — not a schema field |
| `nuri_type`, `public_nuri_license` | **NO** — ours is `kogl_type` |
| `keywords`, `co_authors`, `non_protected_work`, `property_rights`, `validity_period` (Work interface) | **NO** — ours are `keyword`, `co_author`, `unprotected_work`, `economic_rights`, `valid_period`. Their `mapSSUToWorkFields()` (`src/lib/api/ocr.ts:207-251`) has candidate-key fallbacks that cover `work_title` and `valid_period`, but that function is **never called in their repo** |
| `granted_rights`, `contract_type`, `signature_date`, `consent_type`, `data_subject`, `data_controller`, `consent_date`, `kogl_type` label absent but `contract_terms` etc. | YES — match ours |

Also: their `DOCUMENT_TYPES` (`src/lib/api/config.ts:27-33`) include `공공저작물 자유이용허락 동의서`; our v2 UI offers `디지털 콘텐츠` instead. API accepts any string, so harmless, but the prompt steering differs per vocabulary.

**KOGL classification:** correctly not ours. HMC's `POST /api/predict` (live pipeline) does it. Note the vendored `kogl-classifier/` in our repo expects a *different* HMC endpoint (`/api/classify`, different response shape) — our vendored copy is stale relative to their `4ba08e4` "HMC 실제 스펙 반영".

## 3. The contract+work pairing question — **real gap, on their side of the boundary, but our side is not ready either**

Facts: works are paired to contracts only by `works.contract_id` in Supabase; work files are uploaded and abandoned (`ocr_status` stays `"pending"` forever); the 20-field work metadata table renders 미식별; their own `docs/mcpark-전달자료.md` §4-1 shows the aspirational combined API (`POST /api/classify` with `contract_file` + `work_files[]` → per-work metadata) — **not implemented anywhere**.

What's missing, by owner:
- **Muhayu/HF Space:** a per-work loop that POSTs each work file to `/api/llm-extract` and writes the 20 columns (they already wrote `mapSSUToWorkFields()` — it's just never wired up).
- **Us:** even if they call us per work file, our VLM image path fills only `description`, `work_type`, `keyword` (+`digital_format`) of 67 fields (`schema_mapping.py`) — 4-ish of their 20 fields; the rights fields (저작권자, 공개유형, 상업적이용허락, 유효기간…) are *contract* facts, not derivable from an image. Filling their 권리정보 section requires **inheriting contract metadata into work rows** — i.e., a paired endpoint (contract metadata + work file → merged 20-field record) or a documented client-side join rule. No such endpoint exists; nothing links two request_ids.
- **Nobody:** video/audio works — their upload filter excludes them and our pipeline returns `success=false` "P3 예정", so the gap is consistent (deferred on both sides).
- 공공누리 consistency check (contract terms vs work content) exists nowhere and is not in any requirement doc cited — an idea, not a committed gap.

## 4. Verdict: **PARTIALLY ALIGNED**

The contract-analysis leg is aligned and working: their pipeline consumes exactly our `/api/llm-extract` contract, params and response shapes are compatible. Not aligned: (a) work files — the core "저작물" in a 공공저작물 system — are never analyzed by anyone; (b) field vocabulary drift (`work_names`/`nuri_type`/`keywords` vs our schema) means even the wired-up parts show raw or mislabeled keys; (c) the integration lives in an HF Space repo neither analysis could inspect.

**Top actions:**

1. **[Muhayu, M] Wire the work-file loop in `ilwang-kogl-pipeline`**: for each `works` row, POST to our `/api/llm-extract` (images → VLM path automatically), map via their existing `mapSSUToWorkFields()`, write the 20 columns, flip `ocr_status`. This alone turns 미식별 into data for ~4 fields per image.
2. **[Us, M] Define and ship the contract→work inheritance rule** — either a paired endpoint (`contract_metadata` JSON + work file in one request, returns merged 20-field record) or a published mapping table (which of the 20 fields come from the work's VLM output vs the parent contract's `copyright_holder`/`commercial_use`/`valid_period`/`kogl_type`). Without this, 권리정보 (9 of 20 fields) stays empty forever.
3. **[Both, S] Freeze the field vocabulary**: joint mapping doc for `work_title↔work_names`, `keyword↔keywords`, `valid_period↔validity_period`, `unprotected_work↔non_protected_work`, `economic_rights↔property_rights`, `kogl_type↔nuri_type`; decide whether `integrity_right_waiver`/`modification_allowed` become real schema fields (we'd add them) or die.
4. **[Us, S] Update `docs/API_명세서.md`** — stale defaults (solar-ko, google), missing `vlm_prefer`, `modality`/`vlm_backend`/`vlm_raw`, `evidence` in decisions. This doc is what the HF Space integrates against.
5. **[Muhayu, S] Give us read access to the `ilwang-kogl-pipeline` HF Space repo** — it is the only place our API is actually called in production and currently a black box to us; every future contract change is unverifiable without it.