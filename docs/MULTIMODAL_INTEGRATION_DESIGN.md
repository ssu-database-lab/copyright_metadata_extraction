# Year-2 Multimodal Integration Design
_Generated 2026-06-28 via multi-agent design workflow (code-verified against the clip_extraction/vlm prototype). Gemma = primary VLM, Qwen3-VL = backup. Builds on `api/module/clip_extraction/`._

---

I have verified the load-bearing code claims. The extractor confirms `prefer="qwen"` default (extractor.py:168), image-only `SUPPORTED_EXTS` (line 52), sequential `extract_batch` (line 291), and `schema_mapping` passes VLM `work_type` straight through (`_VLM_DIRECT_MAP` line 40, no medium-derivation). Here is the final brief.

---

# Year-2 Multimodal Integration — Final Design Brief

## 1) Recommended architecture + why

**Adopt the parallel-track-by-modality design with a single consolidation tail, plus four corrections the verdicts demand.** A modality router at the head of the pipeline forks each work by media type; both branches emit the same unified 67-field dict and feed the same `ConsolidationAgent`. This scored highest (fit 9, reuse 9) because it is verified true against the code: `consolidate(llm_result, ner_result, ...)` reads only `llm_result['metadata']/['confidence']`, so a VLM result wrapped in the `llm_result` slot needs **zero arbiter changes**, and `run_consolidation` already synthesizes an empty `ner_result` → `LLM_ONLY` decisions for sourceless works.

Four corrections the verdicts forced (do not skip these — they are where the top proposal lost points):

- **work_type comes from the VLM, NOT from file extension.** The verdicts were decisive: a `.jpg` cannot distinguish 사진저작물 from a scanned 미술저작물 or a digitized 도형저작물 (map), and the Korean prompt was *purpose-built* to make exactly those medium-based distinctions. Forcing `work_type=사진저작물` by extension would **regress the KPI it was meant to protect.** → Keep the VLM's `work_type` as the primary value; use file medium only as a sanity floor for video (`영상저작물`, since per-frame guesses are noisy) and as a tiebreak when the VLM abstains.
- **`build_response` and the SSE generator are NOT "unchanged."** Code inspection (cited in verdicts) shows `build_response` calls `self.available_ner_models[ner_model]['display_name']` → KeyErrors on an image work with no NER model, and the SSE generator hardcodes OCR/LLM/NER step messages. Both need small but real None-safe edits. Treat "minimal wiring" as honest only for the *arbiter*, not these two.
- **Flip `prefer='gemma'` and raise `max_tokens` to 2048.** The current `prefer="qwen"` default (extractor.py:168, documented at lines 9-13) is an *operational* state from when Gemma was down, not a design decision. The task mandates Gemma-primary.
- **Embedding model: standardize on SigLIP 2.** The team's own canonical `YEAR2_MULTIMODAL_DESIGN.md` commits to SigLIP 2 (so400m, 1152-dim, Apache-2.0, native 109 langs). The "multilingual-CLIP on efficiency" note is a benchmark observation, not a decision. Honor the design doc; revisit only if index latency becomes a real constraint.

CLIP/SigLIP is **not** in either attribute track — it is a separate similarity/dedup service over image embeddings only (zero-shot medium classification scored ~50% because CLIP keys on visual *subject*, not legal *medium* — a structural limit, not a tuning problem).

## 2) Where it plugs in + routing

**Plug-in point:** `api/web/pipeline.py` `PipelineOrchestrator` + the SSE generator in `api/web/app.py`.

1. **Route at the head.** After `setup()` saves the file, call `router.route(ctx['upload_path'])` (pure stdlib, import at module top, no torch cost) → `{modality, extractors, rationale, extension}`.
2. **Branch in `run()`:**
   - `modality ∈ {document, text}` → existing path **byte-for-byte unchanged**: `run_ocr` → concurrent `run_llm`+`run_ner` → `run_consolidation`.
   - `modality ∈ {image, video}` → new path: `run_vlm()` → (image only, non-blocking) `run_similarity()` → `run_consolidation()` fed the VLM dict in the `llm_result` slot.
   - `modality == audio` → **guarded out** (see §9). Router lists `audio→[vlm]` but Gemma-4/Qwen3-VL are image VLMs; route audio to a "not supported in v1" terminal state, do not send it to the VLM.
3. **Bypass the empty-OCR early-stop guard** on the VLM branch (there is no OCR).
4. **New orchestrator methods** mirror the existing stage-method shape and return-dict convention: `run_router(ctx)`, `run_vlm(ctx)` (returns `{"success":True, "metadata":unified, "confidence":..., "_file_meta":..., "model_used":backend_used}`), `run_similarity(ctx)` (embedding/dedup only, never schema attributes).
5. **SSE wiring** in `process_with_progress()`: emit a routing event after upload, then branch the message stream — VLM track emits `"시각 메타데이터 추출 중 (VLM)..."` with `.backend_used` (Gemma/Qwen), skips OCR/NER messages, and converges on the same consolidate/complete events via the existing `_send_progress_update` helper.

Routing-by-modality is the single decision gate: documents never reach the VLM, images never reach OCR.

## 3) Data flow

**Image work:**
```
file → VLMExtractor.extract(path)            # vlm/extractor.py, Gemma→Qwen chain
     → VLMResult.parsed (JSON: description, work_type, work_type_reason,
       keywords, main_subjects, dominant_colors, text_in_image,
       scene_type, estimated_quality)
     → schema_mapping.map_vlm_to_unified(parsed, path)   # → 67-field dict + _file_meta
     → ConsolidationAgent.consolidate(llm_result=dict, ner_result={}, ...)
```
In parallel, non-blocking: `image → SigLIP 2 → L2-normalized vector → FAISS`. The embedding never fills attributes (`map_embedding_to_unified` is intentionally schema-empty); vector + sha256 live only in `_file_meta`.

**Video work (keyframe strategy):**
```
file → keyframe sampler (BUILD: ffmpeg/PyAV)
       → uniform N frames (start with N=3: 10%/50%/90% of duration),
         OR scene-change detection for variable-length clips; cap frame budget
     → VLMExtractor.extract_batch(frames)    # NOTE: sequential today (line 291)
     → aggregate (BUILD): union+dedupe keywords/main_subjects;
       pick longest/most-complete description (or concatenate per-frame);
       work_type = 영상저작물 (forced from medium — do NOT trust per-frame guess);
       keep first reachable .backend_used
     → schema_mapping.map_vlm_to_unified(aggregated, path)
     → consolidation
```
`client._encode_image` already downscales any frame to 1536px long-edge (under DashScope's 10MB limit), so the sampler is the only new file-handling code. Optionally embed one representative keyframe for dedup.

**CLIP/SigLIP+FAISS role (explicit):** similarity/dedup ONLY, over image embeddings, in a service decoupled from the attribute pipeline. It answers "is this work a near-duplicate?" (true-pair sim ~0.98 vs distractor ~0.45; threshold ~0.85–0.90). It produces no description/work_type/keyword. The dedup verdict is handed to 무하유's 중복판별. **Use FULL-RES originals on `/mnt/e/kogl_originals/`, not thumbnails** — the earlier 65% rank-1 miss was a data artifact (byte-identical KOGL thumbnails under different IDs, SHA256-verified), not a model weakness.

## 4) Gemma-PRIMARY / Qwen-BACKUP fallback design

**Reuse `vlm/extractor.py` `VLMExtractor` as-is — it already IS the chain the task asks for**, built to mirror the OCR (`UniversalOCRProcessor`) and LLM (`create_extractor` + fallback) patterns. Verified mechanics:
- `default_backends(prefer=)` builds the ordered list (extractor.py:134-145).
- Lazy ping-probe (`_probe_chain` via `VLMClient.ping → models.list`) selects the first reachable backend (lines 182-216).
- Per-call: if `extract()` raises or returns `ok=False`, it transparently advances to the next backend and **promotes** the working one (lines 246-264).
- `refresh()` re-probes when a downed Gemma server returns (lines 218-222).
- Every result carries `.backend_used` for provenance.

**Config to honor the intent:** `VLMExtractor(prefer="gemma", max_tokens=2048)` → chain `[Gemma 4 31B (local vLLM), Qwen3-VL-235B (DashScope)]`. The `max_tokens` bump (from 1024) prevents the long-document JSON truncation seen in the English-prompt run.

**Cross-network Gemma host:** `make_gemma_backend()` reads `GEMMA_URL` (default `http://127.0.0.1:8001/v1`, model `google/gemma-4-31B-it`). Point `GEMMA_URL` at the vLLM host over the **permanent Tailscale link** (retire the cloudflared quick-tunnel — see `vlm/standalone_gemma_host.py` + `docs/CLIENT_GUIDE.md`). `VLMClient.ping()` is the reachability gate: if Tailscale is down, the probe fails and the chain **auto-selects Qwen3-VL on DashScope** (`DASHSCOPE_BASE`, `api_key=DASHSCOPE_API_KEY`) with **zero code change** — exactly like OCR falling alibaba→mistral. On total VLM failure, `run_vlm` returns `success=False` and consolidation degrades gracefully.

## 5) Schema mapping

Reuse `schema_mapping.map_vlm_to_unified` — it is genuinely schema-validated (asserts every emitted key exists in `DocumentSchemas.get_unified_schema()['properties']`). Visual metadata → unified/official fields:

| VLM output | Unified field | Official KOGL element |
|---|---|---|
| `description` | `description` | 콘텐츠 설명 |
| `keywords` (fallback `main_subjects` when empty) | `keyword` | 주제어 / 해시태그 |
| `work_type` (VLM medium-classification) | `work_type` | 저작물 종별/유형 |
| file extension uppercased (JPG/MP4) | `digital_format` | 디지털화형태 (known KOGL gap) |
| from `manifest.xlsx`/filename (Stage-1) | `title` | 제목 (VLM does not reliably produce title) |

**Not schema fields** → `_file_meta`: `sha256` (serves UCI / dedup hash), `file_path`/`file_name`, and `vlm_extras` (`work_type_reason`, `scene_type`, `dominant_colors`, `text_in_image`, `estimated_quality`) preserved as evidence.

These fill **Stage-1 catalog elements** (제목/유형/주제어/설명/디지털화형태). **Rights elements** (저작권자/공동·인접권자/권리/유효기간/권리근거 = Stage 2) remain the contract OCR→LLM→NER path's job, now validatable against `dataset/contracts/contracts_index.xlsx` ground truth.

Optional later: if `text_in_image` is non-trivial, route it to `run_ner` so in-image text yields entities (turns `LLM_ONLY` into real arbitration).

## 6) REUSE vs BUILD

**REUSE (verified, as-is):**
- `api/module/clip_extraction/vlm/extractor.py` — Gemma→Qwen fallback chain. Set `prefer="gemma", max_tokens=2048`.
- `api/module/clip_extraction/vlm/client.py` — OpenAI-compatible client for both backends; 1536px downscale; `ping()` gate. No changes.
- `api/module/clip_extraction/vlm/prompts.py` — `SYSTEM_PROMPT`/`USER_PROMPT` (Korean default; medium-based work_type rule; 9-key schema). *(extractor imports these constants directly, not `get_prompts('ko')` — functionally identical.)*
- `api/module/clip_extraction/router.py` — `route()`/`detect_modality()`, pure stdlib.
- `api/module/clip_extraction/schema_mapping.py` — `map_vlm_to_unified()`/`map_embedding_to_unified()`, schema-validated.
- `api/module/clip_extraction/embed_benchmark.py` + `models/` (`base.py`, `siglip2.py`, etc.) — encoders for the FAISS service.
- `api/module/clip_extraction/vlm/compare.py` + `run_extended_comparison.py` + `summarize_vlm_compare.py` — offline Gemma-vs-Qwen eval harness.
- `api/module/clip_extraction/build_eval_manifest.py` + `fetch_samples.py` — eval-manifest builder; re-point at `dataset/manifest.xlsx` + `/mnt/e/kogl_originals/`.
- `api/module/consolidator/` `ConsolidationAgent.consolidate()` — same arbiter, both tracks.
- Pattern (not code): `api/module/ocr/` + `api/module/llm_extraction/` fallback shapes; `pipeline.py:236` `run_consolidation(fallback_model=...)`.

**BUILD:**
- **Wiring** in `pipeline.py`: `run_router`, `run_vlm` (returns `llm_result`-shaped dict), `run_similarity` (non-blocking); modality branch in `run()`; bypass empty-OCR guard on VLM branch.
- **None-safe `build_response`** (`pipeline.py:296`): guard `available_ner_models[ner_model]` and NER-entity formatting when `ner_model is None` (image/video works).
- **SSE branching** in `app.py process_with_progress()`: routing event + VLM-track messages, report `.backend_used`, converge on existing consolidate/complete events.
- **Video keyframe sampler** (ffmpeg/PyAV): uniform N=3 or scene-change; extend `SUPPORTED_EXTS` (extractor.py:52, image-only today); multi-frame aggregation (union keywords, merge description, force `work_type=영상저작물`). Consider parallelizing `extract_batch` (line 291 is sequential — per-video latency multiplies across frames).
- **Video work_type floor**: force `영상저작물` from medium post-aggregation; keep image `work_type` from the VLM.
- **Consolidation adapter**: wrap unified dict as `{"metadata": unified, "success": True, "confidence": ...}`; decide whether `text_in_image` → `run_ner`.
- **FAISS similarity/dedup service**: build a SigLIP-2 index over `/mnt/e` image embeddings; set threshold (~0.85–0.90); expose a small API; define the 무하유 handoff interface.
- **Audio guard**: route audio to a "not supported v1" terminal state (defer ASR→text path).
- **Eval run** + **Stage-2 rights validation** (see §8).
- **Config**: `prefer='gemma'`, `GEMMA_URL`=Tailscale host, `max_tokens=2048`; surface `.backend_used` + dedup verdict in `build_response`/`save_results`.

## 7) Phased plan (evaluable on the 500 img / 500 video on E:)

- **P0 — Restore Gemma + config (PREREQUISITE).** Permanent Tailscale link to the vLLM host; set `GEMMA_URL`; `VLMExtractor(prefer='gemma', max_tokens=2048)`; smoke-test against a handful of `/mnt/e/kogl_originals/이미지` samples (confirm Gemma primary, Qwen auto-fallback when `GEMMA_URL` unreachable).
- **P1 — Eval baseline (now unblocked).** Re-point `build_eval_manifest.py`/`compare.py` at `dataset/manifest.xlsx` + `/mnt/e` originals; run Gemma-vs-Qwen on the 500 images; score description/work_type/keyword; establish the 80→85% KPI baseline. **Ships first because it needs no pipeline changes.**
- **P2 — Wire the IMAGE track** into `PipelineOrchestrator` behind a feature flag: `run_router` + `run_vlm` + `map_vlm_to_unified` + consolidation adapter + None-safe `build_response` + SSE events. Document/text path stays byte-for-byte unchanged.
- **P3 — VIDEO track**: keyframe sampler + multi-frame aggregation; extend `SUPPORTED_EXTS`; run on the 500 videos (`work_type=영상저작물`); measure.
- **P4 — Similarity/dedup service**: SigLIP-2 FAISS index over image embeddings; set threshold; expose dedup API; agree 무하유 handoff.
- **P5 — Stage-2 rights**: run OCR→LLM→NER over the 1500 synthetic 이용허락 계약서; validate against `contracts_index.xlsx`; close rights elements.
- **P6 — Certification + deliverable**: KTC/TTA 공인시험인증 on 1,000 works/type; run the documented (evidence-gated) CLIP fine-tune experiment to satisfy proposal p.32 regardless of outcome.

## 8) How to evaluate

- **Weak labels from `manifest.xlsx`:** the manifest carries 제목/저작권자/공공누리유형/**주제어**/**해시태그** per 원문인덱스. Score VLM `keyword` against 주제어∪해시태그 with token-overlap / set-F1 (synonym-tolerant, since VLM keywords are open-vocabulary). Score `work_type` against 공공누리유형/정보유형 where present — but treat KOGL 정보유형 as noisy (images are ~all 사진: 28,692 vs 26 미술 / 41 음성), so report per-class, not just aggregate accuracy.
- **Small gold set (recommended, ~50–100 works/type):** hand-label description adequacy (3-point: complete/partial/wrong), work_type correctness by *medium*, and keyword precision. This is the only trustworthy measure for the 80→85% attribute-accuracy KPI, because manifest 주제어 is sparse and the medium-vs-subject distinction needs human judgment.
- **Gemma-vs-Qwen agreement** via `compare.py`/`summarize_vlm_compare.py`: expect ~60% on diverse sets; flag disagreements for the gold-set review (these are the hard cases). Qwen is historically better on document/text-medium (어문) — consider Gemma+Qwen `work_type` reconciliation via the consolidation pattern if disagreement clusters there.
- **Dedup eval**: on full-res `/mnt/e` originals, measure rank-1 retrieval + true-pair/distractor separation; set the FAISS threshold where the margin is cleanest (~0.85–0.90).
- **Stage-2**: exact/fuzzy field match of extracted rights vs `contracts_index.xlsx`.

## 9) Honest risks + the single most important first action

**Risks:**
- **Video is the weakest, load-bearing area.** No sampler exists; `SUPPORTED_EXTS` is image-only; `extract_batch` is sequential, so per-video latency = N frames × per-call time across 500 clips. Frame budget and aggregation rules (how to merge conflicting per-frame descriptions) are unresolved. Treat P3 as genuine net-new work, not wiring.
- **Both Gemma and Qwen can be confidently wrong on historical/niche artifacts** (both misread 제국신문 as 독립신문) — directly threatens the work_type/description KPI. Mitigate with the gold-set review and surfacing `work_type_reason` as evidence.
- **Cross-network Gemma link instability**: probe-based fallback covers single-backend loss (auto-switch to Qwen DashScope), but if *both* are down an image/video work yields no visual metadata — monitor `VLMExtractor.probe_log` and surface `.backend_used`.
- **Cost/awkwardness of running the full cloud arbiter on pure-image works** (every decision is `LLM_ONLY` — nothing to arbitrate, yet one cloud call per image at 500-image scale). Consider a cheaper single-source pass-through for VLM-only works in a later pass.
- **Audio has no real path** with the chosen image-VLM stack — guard it out in v1.
- **Embedding-model / work_type-source decisions** were genuinely contested across docs; this brief resolves them (SigLIP 2; VLM-derived work_type) — flag both to stakeholders so they aren't silently reversed.

**Single most important first action: P0 — establish the permanent Tailscale link to the Gemma vLLM host, set `GEMMA_URL`, and flip `VLMExtractor(prefer='gemma', max_tokens=2048)`, then smoke-test the chain on a few `/mnt/e/kogl_originals/` images.** Everything else (the entire eval baseline, the image track, the KPI measurement) is blocked on having Gemma reachable as primary; the fallback machinery is already built, so this is a config + network task, not code.

Key files: `api/module/clip_extraction/vlm/extractor.py`, `api/module/clip_extraction/vlm/client.py`, `api/module/clip_extraction/router.py`, `api/module/clip_extraction/schema_mapping.py`, `api/web/pipeline.py`, `api/web/app.py`, `api/module/consolidator/consolidation_agent.py`, `docs/YEAR2_MULTIMODAL_DESIGN.md`.