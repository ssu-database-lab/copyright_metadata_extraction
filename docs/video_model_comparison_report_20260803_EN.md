# Video Track Model Comparison Report

**Author** Soongsil University DB Lab · **Test date** 2026-07-31 · **Re-scored** 2026-08-03 · **Status** Conclusion fixed
**Scope** Selecting the VLM used to auto-generate descriptions and keywords for video works
**Basis** 8 models × 5 videos = **40 live API calls** (0 failures), scored against a published 26-item rubric

---

## 1. Summary

| Item | Conclusion |
|---|---|
| **Primary model** | **`qwen3.5-omni-plus`** — accuracy **92.3%** (24/26), mean **5.0 s** |
| **Fallback** | `qwen3-vl-235b-a22b-instruct` — 80.8%, 4.0 s (already in the stack for OCR) |
| **Gemma 4 31B** | 76.9% / 20.2 s on video → **keep as image-only**, drop from the video primary slot |
| **vs. existing plan** | The implementation plan's §1 *"keep Gemma 4 31B"* was written **before this test** → superseded by this report |
| **Cost impact** | ₩0.27 (Gemma) → ₩2.01 (omni-plus) per video. Against ₩96 total per work, **+1.8%** — negligible |
| **Side finding** | ⚠️ The test harness's frame sampling is **front-biased** (only the first 19–75% observed). The implementation plan's §3-4 farthest-point design does not have this problem, so **building to the plan resolves it**. Accuracy figures are a lower bound |

**Core judgment:** `qwen3.5-omni-plus` ranks first on accuracy while being **the only fast model among the top scorers**. The next-best models — `qwen3.7-plus` and `qwen3.6-plus` (both 84.6%) — are 3× and 5.6× slower. It is the only choice that requires no speed/accuracy trade-off.

---

## 2. Test design

### 2-1. The eight models

| Model | Route | Character |
|---|---|---|
| `qwen3.5-omni-plus` | DashScope | Omni (multimodal) flagship |
| `qwen3.6-plus` / `qwen3.7-plus` | DashScope | Latest flagships (reasoning) |
| `qwen3-vl-235b-a22b-instruct` | DashScope | Current OCR model |
| `qwen3-vl-flash` | DashScope | Lightweight VL |
| `qwen3.6-35b-a3b` | DashScope | Lightweight MoE |
| `qwen3-omni-flash` | DashScope | Lightweight omni |
| `google/gemma-4-31b-it` | OpenRouter | Current image VLM |

### 2-2. Five test videos (stratified by length)

| Class | Title | Length | Resolution | Codec | Frames | Audio |
|---|---|---|---|---|---|---|
| A very short | 시장의 반찬가게_002 | 7.0 s | 1920×1080 | h264 | 3 | aac |
| B medium | 바람에 흔들리는 튤립 배경 소스 | 61.7 s | 1920×1088 | h264 | 5 | aac |
| C long | Fright to the Finish 1954 | 380.4 s | 720×480 | wmv2 | 12 | wmav2 |
| D landscape | 어느 마을 어느 도로 | 14.0 s | 1920×1080 | h264 | **2** | aac |
| E night | 도시의 밤과 차 | 16.3 s | 1920×1080 | h264 | 5 | aac |

All sourced from 공유마당. **All inputs were keyframes (images)** — native video input was excluded because base64 size (9.4 MB) exceeded the request limit (HTTP 413).

### 2-3. Frame extraction (same settings as implementation plan §3)

```bash
ffmpeg -hide_banner -loglevel error -nostdin \
  -ss "$T" -t 2 -i "$VIDEO" \
  -vf "thumbnail=n=30,scale=768:768:force_original_aspect_ratio=decrease" \
  -frames:v 1 -q:v 3 -y "frame_%02d.jpg"
```
Target frames by duration: `<10s→3 · 10–60s→5 · 1–5m→8 · 5–15m→12 · 15m+→16` (floor of 2). Oversample 1.75×, then deduplicate by dHash (8×8) Hamming distance > 6.

### 2-4. Prompt (identical for all models, temperature = 0)

```
system: 당신은 저작물 메타데이터 추출 전문가입니다. 유효한 JSON 객체 하나만 출력하세요.
user:   다음 N장은 하나의 영상에서 시간순으로 추출한 대표 프레임입니다(각 시점: …초).
        개별 이미지가 아니라 **영상 전체**의 내용을 파악해 아래 JSON으로만 답하세요.
        {"description":"영상 내용을 한국어 2~3문장으로 구체적으로",
         "keywords":["핵심 키워드 5~7개"]}
```

**All 40 calls returned both fields** (1 description + 7 keywords; 0 parse failures).

### 2-5. Scoring rubric (26 items)

Items were derived from the 공유마당 GOLD descriptions. Scoring matches `description + keywords` as whitespace-stripped substrings, with alternates absorbing spelling variation (e.g. 검정콩조림 ← 검은콩 · 콩자반 · 블랙빈).

| Video | Items | Basis | Items |
|---|---|---|---|
| A | 6 | GOLD | 마늘 · 깻잎 · 검정콩조림 · 반찬 · 플라스틱용기 · 시장 |
| B | 3 | **title** | 튤립 · 바람·흔들림 · 꽃밭/정원 |
| C | 6 | GOLD | 뽀빠이 · 블루토 · 올리브 · 유령/공포 · 장난 · 애니메이션 |
| D | 6 | GOLD | 마을 · 산 · 바다 · 도로 · 구름 · 차 |
| E | 5 | GOLD | 지하철역 · 도로 · 오토바이 · 차 · 밤 |

> **Why B's GOLD was not used:** the description 공유마당 supplied reads *"기증식별번호(FC130026), 기증신청일시(2013년 10월 12일), 신청인구분(저작재산권자 본인)"* — **donation administrative data, not a visual description.** It cannot serve as visual ground truth, so the title was used instead. This is direct evidence that the collected `요약정보` field **needs filtering** before use as training or evaluation ground truth.
>
> **Mis-transliterations were not credited.** 파파이 / 퍼시발 / 페퍼 suggest the model recognized Popeye, but the value that would be **stored as metadata is wrong**, so they do not count as hits.

The scorer, with the rubric inline, is stored at `api/module/clip_extraction/vlm/score_video_models.py`. Re-running it regenerates every table in this report.

---

## 3. Overall results

| Rank | Model | Hits | Accuracy | Mean s | In tok | Out tok | Per video |
|---|---|---|---|---|---|---|---|
| **1** | **`qwen3.5-omni-plus`** | **24/26** | **92.3%** | **5.0** | 10,373 | **629** | ₩2.01 |
| 2 | `qwen3.7-plus` | 22/26 | 84.6% | 15.0 | 10,363 | 3,162 | ₩2.58 |
| 2 | `qwen3.6-plus` | 22/26 | 84.6% | 28.0 | 10,363 | 5,271 | rate unknown |
| 4 | `qwen3-vl-235b-a22b-instruct` | 21/26 | 80.8% | 4.0 | 10,463 | 764 | ₩1.51 |
| 5 | `qwen3.6-35b-a3b` | 20/26 | 76.9% | 6.6 | 10,363 | 3,691 | rate unknown |
| 5 | `google/gemma-4-31b-it` | 20/26 | 76.9% | 20.2 | 7,451 | 606 | ₩0.27 |
| 7 | `qwen3-vl-flash` | 17/26 | 65.4% | 3.3 | 10,463 | 964 | rate unknown |
| 8 | `qwen3-omni-flash` | 15/26 | 57.7% | 2.8 | 10,463 | 826 | rate unknown |

**Token counts are measured in this test**; unit prices are quoted from public sources (see §6).

### Per-video breakdown

| Model | A 반찬(6) | B 튤립(3) | C 뽀빠이(6) | D 마을(6) | E 야경(5) |
|---|---|---|---|---|---|
| `qwen3.5-omni-plus` | **6** | **3** | **6** | 5 | **4** |
| `qwen3.7-plus` | 5 | 2 | **6** | 5 | **4** |
| `qwen3.6-plus` | 4 | 2 | **6** | **6** | **4** |
| `qwen3-vl-235b` | 4 | **3** | 4 | **6** | **4** |
| `qwen3.6-35b-a3b` | 4 | **3** | 5 | 4 | **4** |
| `gemma-4-31b-it` | 4 | 2 | **6** | 4 | **4** |
| `qwen3-vl-flash` | 3 | 2 | 3 | **6** | 3 |
| `qwen3-omni-flash` | 3 | 2 | 2 | 4 | **4** |

`qwen3.5-omni-plus` places joint-first or better on 4 of 5 videos, and is the only model to score full marks on both A (6/6) and C (6/6).

### Actual output of the winning model (video A)

> **Description:** 투명 플라스틱 용기에 담긴 다양한 한국식 밑반찬들이 차례로 클로즈업됩니다. 노란 마늘 장아찌, 초록색 깻잎 무침, 검은 콩 조림 등 세 가지 반찬의 모습과 질감이 선명하게 보입니다.
> **Keywords:** 밑반찬, 한국음식, 마늘장아찌, 깻잎무침, 콩조림, 밀키트, 반찬가게
> **GOLD:** 전통시장에서 플라스틱 통에 담겨 판매되는 다양한 반찬들을 촬영한 7초 영상이다. 마늘장아찌, 깻잎조림, 검정콩조림이 보인다.

All three specific dishes identified correctly — from **only 3 frames** of a 7-second video.

---

## 4. Qualitative analysis

### 4-1. Proper-noun errors — the widest divergence between models

Character names in video C (a 1954 Popeye cartoon):

| Model | Output | Verdict |
|---|---|---|
| `qwen3.5-omni-plus` · `qwen3.6-plus` · `qwen3.7-plus` · `gemma-4-31b` | 뽀빠이 · 블루토 · 올리브 | ✅ correct |
| `qwen3.6-35b-a3b` | Popeye · Bluto · Olive Oyl | △ correct but **in English** |
| `qwen3-vl-235b-a22b-instruct` | **파파이** · 블루토 · **올리버** | ❌ mis-transliterated |
| `qwen3-vl-flash` | **퍼시발** · 블루토 · **패트리시아** | ❌ invented |
| `qwen3-omni-flash` | **페퍼** · 블루토 | ❌ invented |

**Implication:** work titles and person names are **fields the VLM must never populate**. `work_title` must come from the filename or database; VLM output should feed only `description` and `keyword`.

### 4-2. Language consistency

`qwen3.6-35b-a3b` returned its entire keyword list in English for video C despite a Korean prompt (`Popeye, Bluto, Olive Oyl, Halloween, Classic Cartoon, Comedy, Animation`). None of the other seven models did this. Adopting a lightweight model would require **forced-language post-processing**.

### 4-3. Gemma is not competitive on video

It is the current image-track model, but on video it scores **76.9% at 20.2 s** — less accurate *and* 4× slower than the winner. It did score 6/6 on video C and uses the fewest tokens (7,451 — a fixed 280 tok/frame). **Keep it for images**; drop it from the video primary slot.

### 4-4. Cost and latency structure of reasoning models

Input tokens are nearly identical across all eight models (7.4k–10.5k — dominated by frames). **Output tokens vary 8× (629–5,271).** The latency of `qwen3.6-plus` (5,271 tok, 28.0 s) and `qwen3.6-35b-a3b` (3,691 tok) comes from reasoning tokens, and **it did not translate into accuracy** (84.6% / 76.9%). Reasoning-heavy models offer no advantage for metadata extraction.

---

## 5. ⚠️ Front-biased frame sampling in the test harness (deviates from the plan)

Discovered during re-scoring. This is **not a flaw in the implementation plan's design** — it is the test harness failing to follow that design.

| Video | Length | Last frame at | **Observed range** |
|---|---|---|---|
| A 반찬 | 7.0 s | 3.5 s | first **50%** |
| B 튤립 | 61.7 s | 46.3 s | first **75%** |
| C 뽀빠이 | 380.4 s | 208.3 s | first **55%** |
| D 마을 | 14.0 s | 2.6 s | first **19%** |
| E 야경 | 16.3 s | 9.2 s | first **56%** |

**All five videos** missed their later portions. The cause is early termination in the harness's **greedy sequential** selection:

```python
for t, f in raw:                      # raw anchors are spread evenly across the whole video
    if all(ham(d, dhash(k[1])) > 6 for k in kept):
        kept.append((t, f, d))
    if len(kept) >= target: break     # ← stops on reaching target → later anchors never reached
```

The anchors themselves span the full duration, but iterating from the start and stopping as soon as the target count is met means **later anchors are never extracted**. Video D compounded this: slow panning made dHash reject nearly every frame, and the floor-of-2 rule then picked the **two earliest** anchors (0.9 s and 2.6 s), leaving only the first 19% of a 14-second video.

### The plan's design already avoids this

Implementation plan §3-4 specifies the selection structure as:

> **Structure**: oversample to **1.75×** target → quality filter → farthest-point (max-min) diversity selection → N frames

**Farthest-point selection considers the entire oversampled set and picks mutually distant frames, which naturally spreads across the timeline.** The test harness did not implement this, substituting a simpler greedy sequential pass. The correct response is therefore **not to change the plan, but to implement §3-4 literally.**

**Impact:** all eight models received **identical frames**, so **the ranking remains valid**. Absolute accuracy should be read as a **lower bound**; scores would likely rise once §3-4 is properly implemented.

**Checks at implementation time:**
1. Actually implement the §3-4 farthest-point (max-min) selection — do not substitute greedy sequential
2. When the floor of 2 applies, pick the **first and last** frames, not the first two
3. Review the dHash Hamming threshold (currently 6) for over-aggressive cases like video D
4. Re-measure the same 5 videos after the above → update absolute accuracy

---

## 6. Cost

Per video (keyframe mode, based on the measured tokens above):

| Model | Rate source | Per video |
|---|---|---|
| `google/gemma-4-31b-it` | OpenRouter (cost report) | ₩0.27 |
| `qwen3-vl-235b-a22b-instruct` | Alibaba official (cost report, 2026-07-15) | ₩1.51 |
| **`qwen3.5-omni-plus`** | third-party aggregators $0.40/$4.80 — **confirm officially before contracting** | **₩2.01** |
| `qwen3.7-plus` | Alibaba announced $0.40/$1.60 | ₩2.58 |

The remaining four models have **no cost figure** because no official rate could be confirmed (measured tokens only, §3).

**Switching Gemma → omni-plus adds ₩1.74 per video.** Against the plan's ₩96 per work (5-page contract + 1 video), that is **+1.8%**, and cost is dominated by the consolidation step (₩35.4) — so **cost cannot decide model selection here**. A 15.4 pp accuracy gain (76.9% → 92.3%) is worth far more than that amount.

Exchange rate $1 = ₩1,400 (same as the cost report).

---

## 7. Conclusion and recommendation

### 7-1. Video track model chain (changed)

```
1st  qwen3.5-omni-plus              92.3% · 5.0 s · ₩2.01
2nd  qwen3-vl-235b-a22b-instruct    80.8% · 4.0 s · ₩1.51   (already in the stack for OCR)
3rd  google/gemma-4-31b-it          76.9% · 20.2 s · ₩0.27  (shared with the image track)
```

`qwen3-vl-235b` is second because it is 4th on accuracy but **the fastest of all (4.0 s)** and already proven in the OCR stage, adding no new dependency. Gemma remains the final fallback since the image track uses it anyway.

### 7-2. Limits — how far this conclusion carries

- **The sample is small.** 5 videos, 26 items. The gap between 1st and 2nd is **2 items**, so a different sample could reorder them.
- **Measured with the harness's frame bias present** (§5). Re-measurement after building to plan §3-4 is required.
- **Automated scoring is coarse** — substring matching can credit a contextually wrong answer.
- **Audio was not used.** Every test video has an audio track, but only frames were sent. The ASR track remains out of scope per implementation plan §5.

This conclusion is therefore **sufficient grounds to begin implementation with omni-plus as primary**, but is not a final determination.

### 7-3. Follow-up actions

| # | Action | Priority |
|---|---|---|
| 1 | Implement plan §3-4 farthest-point selection for real (do not copy the harness) + first/last on floor | **High — before implementation** |
| 2 | Re-measure the same 5 videos afterwards → update absolute accuracy | High |
| 3 | Expand the sample to 15–20 videos to re-confirm the 1st/2nd gap | Medium |
| 4 | Confirm the official rate for `qwen3.5-omni-plus` (currently third-party) | Medium |
| 5 | Forced-language post-processing if a lightweight model is adopted (§4-2) | Low |
| 6 | Filter administrative records out of the `요약정보` field (§2-5, video B) | Medium |

---

## Appendix A. Reproduction

```bash
# Re-run scoring (rubric included in the script)
python api/module/clip_extraction/vlm/score_video_models.py \
       /mnt/d/copyright_dataset_metadata/_video_model_test/results.json
```

## Appendix B. Artifact locations

| Item | Path |
|---|---|
| Raw results (all 40 calls) | `E:\gongu_dataset\_video_model_test\results.json` |
| ↳ copy | `D:\copyright_dataset_metadata\_video_model_test\results.json` |
| ↳ copy | `D:\copyright_video_samples\model_comparison_results.json` |
| Frames sent to the models | `E:\gongu_dataset\_video_model_test\<video>\frames\` + `_contactsheet.jpg` |
| ↳ copy | `D:\copyright_video_samples\frames\<video>\` |
| The 5 test videos | `D:\copyright_video_samples\videos\` (163 MB) |
| Scorer + rubric | `api/module/clip_extraction/vlm/score_video_models.py` |

## Appendix C. Related documents

- `docs/video_track_implementation_plan_20260731_EN.md` — this report's conclusion is reflected in §1 and §4
- `docs/pipeline_diagram_20260731_EN.md` — video path model labels updated
- `docs/api_cost_evaluation_20260730_EN.md` — source for §6 rates and exchange rate
