# Video Work Track — Implementation Plan

**Author** Soongsil University DB Lab · **Date** 2026-07-31 · **Revised** 2026-08-03 · **Status** Design finalised / Implementation pending
**Scope** The video processing path of the multimodal pipeline (currently blocked by a P3 guard)
**Basis** Full survey of 2,209 held videos + live ffmpeg/ffprobe measurements + live Gemma/Qwen token measurements
 + an **8-model × 5-video comparison test** (`docs/video_model_comparison_report_20260803_EN.md`)

> **Revision note (2026-08-03)** — the first edition of this plan was written **before** the model
> comparison test and named Gemma 4 31B as primary in §1 and §4. A subsequent 40-call comparison found
> **`qwen3.5-omni-plus` first at 92.3% accuracy / 5.0 s**, with Gemma at 76.9% / 20.2 s, so the model
> statements in §1 and §4 have been updated. All other sections (frame strategy, ASR exclusion rationale,
> cost structure) are unchanged. Note that the test harness did **not** implement the §3-4 farthest-point
> selection — see comparison report §5 before implementing.

---

## 1. Summary


| Item                    | Decision                                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------------------- |
| **Processing approach** | Keyframe extraction → single multi-frame VLM call → reuse the existing consolidation path               |
| **Frame selection**     | **Uniform seek anchors + local** `thumbnail` **selection** (hybrid)                                     |
| **Frame count**         | 3–16 by duration (floor of 2)                                                                           |
| **Audio (ASR)**         | ❌ **Excluded from v1** — only 0.19% of the data has Korean audio, and hallucination risk is measured    |
| **Model**               | **`qwen3.5-omni-plus` primary** (92.3% · 5.0 s) → `qwen3-vl-235b` → Gemma 4 31B<br>*(per the 2026-08-03 comparison; Gemma scored 76.9%/20.2 s on video → image-only)* |
| **Technical metadata**  | ffprobe auto-extraction of duration, resolution, aspect ratio, codec (15/15 match vs 공유마당 ground truth) |
| **Estimated cost**      | **₩98 per case** (5-page contract + 1 video) — +₩1.74 vs Gemma after the omni-plus switch                |
| **Estimated duration**  | **2–3 weeks**                                                                                           |


---



## 2. Measured Dataset Profile (all 2,209 files)


| Bucket    | Count       | Duration      | Resolution      | Codec | Character                                           |
| --------- | ----------- | ------------- | --------------- | ----- | --------------------------------------------------- |
| `ccl`     | 1,003 (73%) | **1.8–6 s**   | 1920×1080       | h264  | Sign-language clips, **single fixed shot, no cuts** |
| `donated` | 707         | 4–105 s       | 720×480–1080p   | h264  | Shorts / promotional                                |
| `expired` | 499 (27%)   | **50–80 min** | 320×240–662×480 | wmv2  | Public-domain black-and-white films                 |


**Containers**: mkv 825 · mp4 654 · wmv 503 · **swf 125 (⚠️ Flash, not video — must be excluded)** · mov 99 · avi 3

> **Design implication**: the data is **bimodal** (73% single-shot clips of a few seconds ↔ 27% hour-long films). No single strategy satisfies both, so **duration-based branching is mandatory**.

---



## 3. Keyframe Extraction Design



### 3-1. Why scene-change detection is not used alone


| Method                                    | 70-min film, 12 frames | Short sign clip              | Defect                                  |
| ----------------------------------------- | ---------------------- | ---------------------------- | --------------------------------------- |
| (a) Uniform `fps` filter                  | 26.9 s                 | 0.19 s                       | Full decode → cost scales with duration |
| (b) Scene change `select=gt(scene,0.4)`   | 41.7 s (full)          | **0 frames — total failure** | Two fatal defects below                 |
| (d) Fixed first/middle/last               | 0.2 s                  | 0.1 s                        | First frame measured pblack=100 (black) |
| **(e) Uniform seek + thumbnail (chosen)** | **0.97 s**             | 0.7 s                        | None                                    |


**Two measured defects of scene-change detection**

1. **Returns 0 frames on single-shot video** — a 2.4 s sign clip (fixed camera) yielded zero frames. **73% of our data is this type**, so scene detection alone fails on three quarters of the corpus.
2. **Temporal bias when combined with** `-frames:v N` — on a 70-minute film, all 12 extracted frames fell within the **first 6 minutes** (ffmpeg stops as soon as N is satisfied), whereas the actual 257 cuts were spread evenly across 70 minutes. Achieving even coverage requires a full detection pass (41.7 s) plus re-selection — **43× more expensive than the hybrid**.



### 3-2. Chosen method — uniform anchors + local representative selection

```bash
# One invocation per anchor (anchors = evenly spaced points across the full duration)
ffmpeg -hide_banner -loglevel error -nostdin \
  -ss "$T" -t 2 -i "$VIDEO" \
  -vf "thumbnail=n=30,scale=768:768:force_original_aspect_ratio=decrease" \
  -frames:v 1 -q:v 3 -y "frame_%02d.jpg"
```

**Three essentials**

- `-ss` **must precede** `-i` → input seek (keyframe jump). Placing it after forces sequential decoding, tens of times slower.
- `-t 2` decodes only a 2-second window; `thumbnail=n=30` then picks the histogram-optimal frame within it (avoiding blurry / mid-transition frames).
- `scale=768:768:force_original_aspect_ratio=decrease` caps the **long edge at 768** for both portrait and landscape.

> ⚠️ `-vsync` **is deprecated in ffmpeg 6.1.** Most web examples still use `-vsync vfr`; write `-fps_mode vfr`.



### 3-3. Frame count by duration

Research basis: Qwen3-VL-class models peak at **8 frames**, with no gain beyond 16; improvement runs from 8→32 then plateaus. Our task is **metadata extraction**, not VQA, so fewer frames than the benchmarks suffice.


| Duration    | Applicable data             | Target N | Oversample | Rationale                                            |
| ----------- | --------------------------- | -------- | ---------- | ---------------------------------------------------- |
| **< 10 s**  | ccl sign clips, 1,003 (73%) | **3**    | 5          | Single fixed shot; measured 2 of 8 frames duplicated |
| 10–60 s     | part of donated             | **5**    | 9          |                                                      |
| 1–5 min     | longer donated              | **8**    | 14         | VLM optimum                                          |
| 5–15 min    | few                         | **12**   | 21         |                                                      |
| **15 min+** | expired, 499 (27%)          | **16**   | 28         | Before the plateau                                   |


**Enforce a floor of 2 frames** — deduplication was measured reducing a static video to a single frame.

### 3-4. Quality filtering (measured thresholds)


| Filter                  | Threshold                           | Measured basis                                                                                                                                                     |
| ----------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Black-frame removal** | `pblack >= 95`                      | Fade-in t=0–1.5 s = 100; end credits = 97–100. ⚠️ **A genuine night scene measured pblack=60**, so a mean-luminance threshold is dangerous (discards real content) |
| **Deduplication**       | dHash(8×8) Hamming distance **> 6** | 70-min film 12 frames → 12/12 kept (no false positives) · sign clip 8 → 6/8 (rest-pose duplicates removed) · static night shot 5 → 2                               |
| **Fade / low contrast** | `YMAX − YMIN < 20`                  |                                                                                                                                                                    |


> A full `blackdetect` scan costs 113 s on a 70-minute video — **do not use it as a pre-pass**. Post-checking the 12 extracted JPEGs takes about 1 second.
> `mpdecimate` showed no discriminating power on extracted stills (12/12 passed) → **use dHash**.

**Structure**: oversample to **1.75×** target → quality filter → farthest-point (max-min) diversity selection → N frames

Measured effect: a 105 s video whose tail was black previously yielded only 7 frames; the revised flow secures the full 8.

```
13002255_....mp4  dur=104.8s  target=8  sampled=14  passedQC=13  final=8   2.81s  335KB
test_long.wmv     dur=4241s   target=16 sampled=28  passedQC=28  final=16  1.93s  419KB
```

---



## 4. Model Choice

> **Conclusion (updated 2026-08-03):** an 8-model × 5-video comparison (40 live calls) established
> **`qwen3.5-omni-plus`** as primary. Sections 4-1 and 4-2 below remain **valid measurements of tokens
> and input modes**, but the first edition's recommendation (Gemma primary) is superseded in 4-3.
> Full evidence: `docs/video_model_comparison_report_20260803_EN.md`.

### 4-0. Comparison test results (by accuracy)

| Rank | Model | Accuracy (26 items) | Mean s | Per video |
|---|---|---|---|---|
| **1** | **`qwen3.5-omni-plus`** | **92.3%** (24/26) | **5.0** | ₩2.01 |
| 2 | `qwen3.7-plus` | 84.6% | 15.0 | ₩2.58 |
| 2 | `qwen3.6-plus` | 84.6% | 28.0 | — |
| 4 | `qwen3-vl-235b-a22b-instruct` | 80.8% | 4.0 | ₩1.51 |
| 5 | `google/gemma-4-31b-it` | 76.9% | 20.2 | ₩0.27 |

The 2nd and 3rd place models are 3× and 5.6× slower. **Only omni-plus avoids the speed/accuracy
trade-off.** The gap on proper nouns is equally wide — `qwen3-vl-flash` invented *퍼시발* for Popeye and
*패트리시아* for Olive, while omni-plus named all three characters correctly.



### 4-1. Measured token comparison (same video, 5 min 03 s)


| Frames | Gemma prompt_tokens | Qwen `type:"video"` | of which video_tokens |
| ------ | ------------------- | ------------------- | --------------------- |
| 4      | 1,084               | 262                 | 242                   |
| 8      | 2,148               | 502                 | 482                   |
| 16     | 4,276               | 982                 | 962                   |
| 32     | 8,532               | 1,942               | 1,922                 |


→ **Gemma ≈ 266 tok/frame · Qwen video mode = exactly 60 tok/frame (4.4× cheaper)**

### 4-2. Model characteristics


| Item                    | Gemma 4 31B (OpenRouter)                                          | Qwen3-VL-235B (DashScope)                       |
| ----------------------- | ----------------------------------------------------------------- | ----------------------------------------------- |
| Video handling          | **Multi-image** (Google model card: *"process videos as frames"*) | **Native video** (`type:"video"` / `video_url`) |
| Multi-image verified    | ✅ **32 images succeeded**                                         | ✅ 8 verified (docs: up to 250–2000)             |
| **Resolution effect**   | **None — fixed 280 tokens** (high-res is free)                    | Proportional → **must downscale to ≤768px**     |
| Frame limit             | Recommended 60 s @1fps ≈ 60 frames                                | Min 4 / max 2000                                |
| Direct video file input | ⚠️ HTTP 402 (requires $1 balance)                                 | ✅ base64 ≤10 MB                                 |
| Temporal awareness      | Frame order only (needs prompt labels)                            | Timestamp alignment supported                   |
| In-video audio          | Not supported                                                     | **Not supported** (documented)                  |




### 4-3. Recommendation ~~(1st ed.: Gemma primary)~~ → **superseded 2026-08-03**

**Confirmed model chain:**

```
1st  qwen3.5-omni-plus              92.3% · 5.0 s · ₩2.01
2nd  qwen3-vl-235b-a22b-instruct    80.8% · 4.0 s · ₩1.51  (already held for OCR)
3rd  google/gemma-4-31b-it          76.9% · 20.2 s · ₩0.27 (shared with the image track)
```

- **Why 1st** — highest accuracy and the only fast model among the leaders. Its 629 output tokens are the fewest of all (2nd place 3,162; 3rd 5,271), so latency and cost are low simultaneously. Reasoning models' long thinking traces **did not convert into accuracy**.
- **Why 2nd** — 4th on accuracy but **the fastest overall (4.0 s)**, and already proven in the OCR stage, so it adds no new dependency.
- **Gemma demoted** — less accurate than omni-plus and 4× slower on video. It still has the lowest token count (fixed 280/frame) and **remains in use for the image track**, so it stays as final fallback.
- **Cost is not the deciding factor** — Gemma→omni-plus adds **₩1.74 per video**, 1.8% of the ₩96 per-case total. Cost is dominated by consolidation (₩35.4).

**Relation to the 1st-edition recommendation** — the "Qwen video mode is 4.4× cheaper per token" finding in 4-1 still holds, but the test above compared all models **under identical frame (multi-image) conditions**. Moving to native video mode is a separate decision, justified only by **timestamp-grounding quality**, not cost.

**⚠️ Caveat** — the sample is small (5 videos, 26 items) and the 1st/2nd gap is 2 items. The measurement also predates proper §3-4 frame selection, so re-confirm with 15–20 videos afterwards (comparison report §7-2).

---



## 5. Audio (ASR) — Excluded from v1 (measured basis)



### 5-1. The data effectively has no speech (all 2,209 files)


| Bucket              | Has audio stream | **Actually audible** (peak > −40 dB) | Digital silence (< −60 dB) | Subtitles    |
| ------------------- | ---------------- | ------------------------------------ | -------------------------- | ------------ |
| ccl (sign language) | 100.0%           | **0.6%**                             | 99.4%                      | 0            |
| donated             | 80.9%            | 39.0%                                | 57.0%                      | 0            |
| expired             | 98.6%            | 94.7%                                | 5.3%                       | 0            |
| **Total**           | **93.6%**        | **33.4%**                            | **65.7%**                  | **0 (0.0%)** |


- **"93.6% have audio" is a trap** — two-thirds of those are −91 dB digital silence (empty tracks added at encoding).
- Of 429 audible hours, **99.8% are foreign (English) public-domain films from 1895–1962**.
- **Korean + audible = 191 files / 0.88 hours = 0.19% of 452.5 total hours.**
- **Zero subtitle tracks** (embedded or sidecar) → no route to text without ASR.



### 5-2. ⚠️ ASR hallucination is measured, not hypothetical

Running whisper base on all 191 Korean audible clips: 52.9% produced zero Hangul, 36.6% produced 1–9 characters of noise, 10.5% produced ≥10 characters.
**Re-verifying with large-v3 produced more fluent and more dangerous fabrications:**


| Actual video            | large-v3 output                                                                                            |
| ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| Subway ambient noise    | "Hello everyone, today we're at the most famous place in **Jeju**, a public transit station in **Busan**…" |
| Sinchon church exterior | "Hello everyone! Today we're at the most famous place in **Jeju**, in **Busan**…"                          |
| Overcast sky            | "Today, this place is leisurely leisurely leisurely…"                                                      |


Three unrelated videos received **the same fabricated sentence**, internally contradictory (Jeju ↔ Busan). Feeding this into `description`/`keyword` would register **a Sinchon church video as a "Busan transit station."**
**Consolidation cross-checks LLM against NER, so an audio-only hallucination has no counterpart to catch it.**

> Control: 30 fully silent files produced **0/30 hallucinations**. The danger is **ambient noise**, which is the dominant mode of our Korean audible clips.



### 5-3. What to add instead — an ffprobe gate (zero cost)

ffprobe is called anyway for keyframe extraction. Capture **audio presence / audibility / subtitle presence** in the same call as technical metadata, so a future ASR addition requires no rework.

```bash
# Stream presence
ffprobe -v quiet -print_format json -show_format -show_streams "$F"
# Actual audibility (excludes silent tracks) — 30 s sample
ffmpeg -v info -nostats -t 30 -i "$F" -map 0:a:0 -af volumedetect -f null -
#   → max_volume < -40dB means effectively silent
```

> **Note**: static-image-plus-music videos (e.g. "8-track music compilation") cannot be understood from keyframes — **but ASR would not solve them either** (music, not speech). The honest handling is to flag `unique_scenes ≤ 2` in the result as "insufficient visual information."

---



## 6. Technical Metadata (ffprobe)



### 6-1. Availability and accuracy (60 files probed + 15 cross-validated against 공유마당)


| ffprobe path                      | Available     | Maps to                    | Validation       |
| --------------------------------- | ------------- | -------------------------- | ---------------- |
| `streams[v].width/height`         | 60/60         | quality `"W x H"`          | **15/15 match**  |
| `format.duration`                 | 57/60         | duration                   | **15/15 match**  |
| `streams[v].r_frame_rate`         | 60/60         | fps                        |                  |
| `streams[v].codec_name`           | 60/60         | codec                      |                  |
| `streams[a]` present              | 57 yes / 3 no | audio presence             |                  |
| `format.tags`                     | 57/60         | title, year, keyword bonus |                  |
| `streams[v].display_aspect_ratio` | **49/60**     | ⚠️ **do not use**          | **2 mismatches** |




### 6-2. ⚠️ Do not use `display_aspect_ratio` for aspect ratio

Cases were found returning `5:4` and `15:11` for a 640×480 video (non-square SAR tags). **GCD reduction of width/height matched 15/15**:

```python
g = math.gcd(w, h); aspect = f"{w//g}:{h//g}"
# 852x480 → "71:40" (identical to 공유마당), 1920x1080 → "16:9"
```

공유마당's convention is a **storage-ratio reduction**, not standard DAR.

### 6-3. Other cautions

- `.swf` **(125 files, 6%) returns** `format.duration` **= None** — flv1+mp3 streams are detected but duration is absent. Needs a fallback (`-count_frames`) or null tolerance.
- `format.tags` yields free metadata — e.g. `WM/Category: "1890s;video;short film;history;horror;black-and-white;silent film…"` → **keyword seed**, plus `title` and `WM/Year` → `created_date`.
- **No official standard requires video technical metadata** (the 58-row public-works metadata element list has no duration/resolution; the KCISA operational DB's 5,010 video rows have `digital format` 100% empty). → Position ffprobe extraction as **"filling empty fields for free," not regulatory compliance**.
- **KOGL registration requires two thumbnail sizes** (list 294×220 / detail 720×400) → **our extracted keyframes satisfy this requirement as a side effect**.

---



## 7. Implementation Items



### 7-1. New modules

```
api/module/clip_extraction/vlm/
├── video_frames.py     # ffprobe probing + keyframe extraction + quality filter + dHash dedup
└── prompts.py          # (modify) add video prompt
```



### 7-2. Modifications to existing files


| File                                | Change                                            | Difficulty |
| ----------------------------------- | ------------------------------------------------- | ---------- |
| `vlm/client.py` L124-127            | **Multi-image support** (below)                   | Low        |
| `vlm/client.py` `_MAX_LONG_EDGE`    | Override to 768 on the video path (parameterise)  | Low        |
| `vlm/extractor.py`                  | Add `extract_video()`                             | Medium     |
| `clip_extraction/schema_mapping.py` | `map_video_to_unified()` — merge ffprobe metadata | Medium     |
| `web/pipeline.py` L461              | **Remove P3 guard → implement** `run_video()`     | Medium     |
| `clip_extraction/router.py`         | Exclude `.swf`                                    | Low        |


**Multi-image extension (client.py) — minimal structural change**

```python
# Current (single image)
image_part = {"type": "image_url", "image_url": {"url": data_url}}
content = [image_part, text_part] if self.image_first else [text_part, image_part]

# Revised (N images)
image_parts = [{"type": "image_url", "image_url": {"url": _encode_image(p, max_edge)}}
               for p in image_paths]
content = image_parts + [text_part] if self.image_first else [text_part] + image_parts
```

Also required: extend `VLMResult.image` to hold multiple frame identifiers.

### 7-3. Prompt design essentials

The prompt must state that the frames are **time-ordered representatives of one video, not separate images**, and supply timestamp labels.

> "The following N images are representative frames extracted in chronological order from a single video (timestamps: 0 s, 18 s, 36 s…). Describe **the video as a whole**, not the individual images."

Extend the image output schema: existing 8 fields + `duration_desc` (how it develops), `scene_count` (degree of scene change), `video_type` (documentary / advertisement / animation / record, etc.).

---



## 8. ⚠️ Principal Operational Warning — `/mnt/e` I/O bottleneck

Identical 99 MB 1080p file, identical 8 seeks:


| Location                  | Time       |
| ------------------------- | ---------- |
| `/mnt/e` (external drive) | **16.5 s** |
| WSL native disk           | **1.46 s** |


**11× difference.** Measured `/mnt/e` read throughput ≈ **4.7 MB/s**. The bottleneck is disk, not CPU, so **raising parallelism will not help**.

- Batch-processing 2,000 files directly from `/mnt/e` averages 0.83 s/file → about 30 minutes (tolerable)
- Many large files (expired, 300–500 MB) mean **I/O contention risk when parallelising**
- **Recommendation: copy targets to the native path (**`/home/mbmk92/...`**) before processing**

---



## 9. Phased Plan


| Week   | Work                                                                                                                           | Deliverable         |
| ------ | ------------------------------------------------------------------------------------------------------------------------------ | ------------------- |
| **W1** | Implement `video_frames.py` — ffprobe probing, uniform-anchor + thumbnail extraction, pblack/dHash filters, duration-based N   | Module + unit tests |
| **W1** | Extend `client.py` for multi-image; parameterise `_MAX_LONG_EDGE`                                                              | PR                  |
| **W2** | Write video prompt, `extract_video()`, `map_video_to_unified()` (ffprobe merge, GCD aspect ratio)                              | PR                  |
| **W2** | Implement `pipeline.run_video()`, remove the P3 guard, exclude `.swf`                                                          | PR                  |
| **W3** | Validate on a stratified sample of 100 from the 2,000 held videos — frame quality, description accuracy, processing time, cost | Validation report   |
| **W3** | Regression-test the existing image path; deploy to the Oracle server                                                           | Deployment          |


**Total 2–3 weeks.** Audio is excluded, but the ffprobe gate captures `has_audio` / `audible` / `has_subtitle` so a future ASR addition requires no rework.

---



## 10. Open Items

1. **Whether video works are actually in the demonstration scope** — needs confirmation from Muhayu (if not, deprioritise)
2. **Whether to switch to Qwen video mode** — cost benefit is under ₩1 per case, so decide on quality grounds after testing
3. **Policy for static-image videos** (music compilations etc.) — finalise how the `unique_scenes ≤ 2` flag is surfaced
4. **The 125** `.swf` **files** — confirm exclusion (Flash cannot be processed by ffmpeg)

---



## Appendix — Research Artefacts

- Keyframe measurements: ffmpeg 6.1.1 against 2,084 dataset files
- Audio survey: all 2,209 files via ffprobe + volumedetect; whisper base and large-v3 executed
- Token measurements: Gemma (OpenRouter) 1–32 frames; Qwen (DashScope) `type:"video"` 4–32 frames
- ffprobe accuracy: 60 files probed, 15 cross-validated against 공유마당 ground truth

*Korean version of this document:* `docs/영상트랙_구현계획_20260731.md`