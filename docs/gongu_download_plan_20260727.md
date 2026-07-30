# 공유마당 Bulk Download — Verified Feasibility & Execution Plan

**Date:** 2026-07-27 · **Author:** Soongsil Univ. DB Lab
**Target:** 6,000 files = 3 categories (이미지·영상·어문) × 4 licenses (만료·기증·CCL·KOGL) × 500
**Verification:** 6-agent workflow, all live-tested (full census of 627+556 video works; 465 어문/KOGL probes; browser UI click-through)

---

## 0. Verdict

**Ready to start — after 3 mandatory code patches (P1/P2/P3).**
**Realistically obtainable: 4,822 of 6,000 (80%)** — the 1,178 shortfall is because **the data does not exist**, not a crawling limitation.

| | Count | Note |
|---|---|---|
| Target | 6,000 | 12 cells × 500 |
| **Strict max (cells as defined)** | **4,322** | image 2,000 + video 1,500 + text 822 |
| **Recommended (in-category backfill)** | **4,822** | video CCL raised to 1,000 to keep video at 2,000 |
| Total size | **≈184 GB** | E: has 420 GB free ✅ |
| Total time | **33–44 h** | video = 92% of it (bandwidth-bound) |

---

## 1. Access & 검색 flow — VERIFIED (browser click-through)

Playwright drove the real UI: selected each license in `#licenseCd` and **clicked the 검색 button**, capturing the resulting request.

- The license filter is a **`<select name="licenseCd">`** inside `form#wrtClForm`, `method="get"` → **plain GET**, no POST. Our URL-parameter approach is the *identical code path* the site uses.
- **12/12 cells matched exactly** — both the `licenseCd` value and the displayed total count.

| License (UI label) | licenseCd value (site's own option value) |
|---|---|
| 만료저작물 | `97` |
| 기증저작물 | `98,99` |
| CCL | `20,21,22,23,24,25,26,27` |
| KOGL | `01,02,03,04` |

Endpoints (anonymous, no login): `listWrtImage.do?menuNo=200018` · `listWrtVideo.do?menuNo=200026` · `listWrtText.do?menuNo=200019`

---

## 2. 12-cell feasibility matrix (measured)

| # | Category | License | Total | hosted rate | **Max obtainable** | 500? | Notes |
|---|---|---|---|---|---|---|---|
| 1 | 이미지 | 만료 | 13,160 | **100%** (23/23) | 13,160 | ✅ | avg 0.82 MB |
| 2 | 이미지 | 기증 | 14,555 | **100%** | 14,555 | ✅ | avg 8.76 MB (high-res) |
| 3 | 이미지 | CCL | 373,572 | **100%** | 373,572 | ✅ | avg 11.05 MB |
| 4 | 이미지 | KOGL | 987,965 | **100%** | 987,965 | ✅ | avg 0.17 MB (thumbnail-grade) |
| 5 | 영상 | 만료 | **627** | **100%** (627/627 full census) | **627** | ✅ (margin 127) | all `.wmv`, avg 432.7 MB |
| 6 | 영상 | 기증 | **556** | **100%** (556/556 census) | **556** | ✅ (margin 56) | mp4 430 / swf 125, avg 54.8 MB |
| 7 | 영상 | CCL | 8,601 | **100%** (89/89) | 8,601 | ✅ | ~5,800 sign-language `.mkv` @ 0.79 MB |
| 8 | 영상 | KOGL | 118,720 | **0%** (0/70) | **0** | ❌ | `원문제공=원문URL` **with empty URL**; 24/24 direct downloads failed |
| 9 | 어문 | 만료 | 52,110 | **48.3%** (143/296) | **≈25,200** | ✅ | pages 94–661 are 100% hosted |
| 10 | 어문 | 기증 | **49** | 100% (49/49 census) | **49** | ❌ | population is only 49 |
| 11 | 어문 | CCL | **273** | 100% (273/273 census) | **273** | ❌ | population is only 273 |
| 12 | 어문 | KOGL | 1,038,997 | **0%** (0/465, 4-axis cross-check) | **0** | ❌ | metadata-only records; `wrtFileDownload` string absent from HTML |

### The 4 structural shortfalls (data doesn't exist)
- **영상/KOGL = 0.** 70 works checked across pages 1→4947, 3 sort orders — all `원문URL` with an *empty* URL value; direct download 24/24 soft-404.
- **어문/KOGL = 0.** 465 works checked across page depths (1→43,290), sort orders, and all 17 provider batches — hosted 0. The download button isn't even rendered. These 1M records are harvested press-release/document *titles+summaries* with no source file.
- **어문/기증 = 49 total**, **어문/CCL = 273 total** → collecting them *in full* is the only option (and gives a 100% sample, zero sampling bias).

---

## 3. Size & time (E: 420 GB free)

| Category | Cell | Count | **Size** | Selection method |
|---|---|---|---|---|
| 이미지 | 만료/기증/CCL/KOGL | 2,000 | **≈10.4 GB** | stratified 5 pages/cell |
| 영상 | 만료 500 | 500 | **162.1 GB** | census-sorted smallest-500 (page-order = 221.8 GB) |
| 영상 | 기증 500 | 500 | **10.0 GB** | census-sorted smallest-500 |
| 영상 | CCL 1,000 | 1,000 | **0.8 GB** | sign-language `.mkv` block |
| 어문 | 만료 500 + CCL 273 + 기증 49 | 822 | **0.08 GB** | hosted band |
| **Total** | | **4,822** | **≈184 GB** | leaves ~236 GB on E: ✅ |

⚠ **Without census-based selection, video jumps to ~339 GB** (leaves only 70 GB). Selection lists are already generated:
`/mnt/e/gongu_dataset/_select/video_expired_500.txt` (162.1 GB) · `video_donated_500.txt` (10.0 GB)

**Throughput (measured):** single stream 0.70–0.78 MB/s; **saturates at 4 workers = 1.78 MB/s**; 8 workers gave zero gain and +27% latency → **never exceed 8**.
**No rate limiting observed** (520 requests / 40 min; zero 429/403; no Retry-After).

| Stage | Time |
|---|---|
| Listing crawl (12 cells) | < 2 min |
| Detail parse + hosted gate | 20–30 min |
| Image download 2,000 | 1.7–2.5 h |
| Text download 822 | ~10 min |
| **Video download 2,000** | **30–40 h** (bandwidth-bound) |
| **Total** | **33–44 h** |

→ Images + text (2,822 files) finish in **~3 h on day 1**; video runs 2 nights in background.

---

## 4. Mandatory code patches before running (`gongu_downloader.py`)

| # | Problem | Fix | Priority |
|---|---|---|---|
| **P1** | `r.content` loads whole file into memory + `timeout=120` → **video (avg 433 MB, max 7.8 GB) fails 100%** at 0.7 MB/s | `stream=True` + `iter_content(1MB)` → `.part` file → rename; `timeout=(10,300)`; check soft-404 sentinel only **after ≥60 bytes** (sentinel is exactly 60 B) | **BLOCKING** |
| **P2** | Content-Disposition is **%-encoded** for 어문 (and many images), but current code only does latin-1→utf-8 → saves `%ED%97%88…` as the filename | `if "%" in raw: return unquote(raw)` **before** the latin-1 path | **BLOCKING** |
| **P3** | First 500 results are dominated by one uploader (image/CCL 483/500 one company; image/KOGL 500/500 one agency) → dataset diversity collapse | add `--pages "1,700,1400,…"` (multi start-page stratification); default `--page-unit 100` (500 works too) | **BLOCKING** |
| **P4** | No size control for video; **no resume possible** (server ignores Range, no Accept-Ranges) | `--max-file-mb` using container-header size probe (ASF/MP4/MKV/SWF — validated 5/5 exact) | required (video) |
| **P5** | index xlsx **overwritten** each run; `_existing()` treats a truncated part-file as complete | append JSONL → build xlsx at end (upsert by wrtSn); use `.part` and rename only on completion | required |
| P6 | `accepted += 1 if not args.hosted_only else 0` is dead code (always +0); external skips never counted | separate `external_skipped` counter + hosted-rate in summary | high |
| P7 | Running an external-only cell wastes 4,000+ requests | auto-abort if hosted rate < 5% in first 20 samples | high |
| P8 | Fully sequential + `sleep 1.0` → time estimates assume workers | `--workers` (detail 4 / image·text 4 / video 2), guard >8 | high |
| P9 | Listing card already yields (wrtSn, title, author, licenseCd) → pre-filter without detail requests | extend listing parser | medium |
| P10 | Some 어문 `hwp`+`txt` pairs are **byte-identical** (verified wrtSn 13313933) | sha1 dedupe within a work; add `md5` column | medium |
| P11 | No way to feed the video census | `--wrtsn-file <path>` (newline-separated wrtSn, skips listing crawl) | medium (video) |
| P12 | `--exclude-title` default (`템플릿|서식|폰트|글꼴`) silently drops 어문 works | default empty for text/video; keep for image | low |
| P13 | No progress/ETA — fatal for a 40 h job | parse total count → `[n/500, 12.3GB/174GB, ETA 21h]` | low |

---

## 5. Execution order

```bash
cd /home/mbmk92/copyright/copyright_metadata_extraction
export G="venv/bin/python -m api.module.dataset_builder.gongu_downloader"
export E=/mnt/e/gongu_dataset
mkdir -p $E/{image,video,text}/{expired,donated,ccl,kogl} $E/_select $E/_log
```

**STEP 0 — dry-run (5 min, no downloads):** 20 works/cell, verify `external`=0, license codes in-group, author diversity.

**STEP 1 — 어문 3 cells (10 min, 0.08 GB)** — cheapest and most fragile, do first:
```bash
$G --menu text --license 98,99                   --page-unit 100 --pages 1 --limit 49  --hosted-only --exclude-title "" --workers 4 --sleep 0.5 --out $E/text/donated
$G --menu text --license 20,21,22,23,24,25,26,27 --page-unit 500 --pages 1 --limit 273 --hosted-only --exclude-title "" --workers 4 --sleep 0.5 --out $E/text/ccl
$G --menu text --license 97 --page-unit 100 --pages 25,50,75,100,125,150 --limit 500 --hosted-only --exclude-title "" --workers 4 --sleep 0.5 --out $E/text/expired
# 어문/KOGL: DO NOT RUN (hosted 0/465). Substitute 500 from 만료 if desired → $E/text/kogl_substitute_97
```

**STEP 2 — 이미지 4 cells (2–3 h, 10.4 GB)** — stratified pages:
```bash
$G --menu image --license 97                     --page-unit 100 --pages 1,27,53,79,105       --limit 500 --hosted-only --workers 4 --sleep 0.5 --out $E/image/expired
$G --menu image --license 98,99                  --page-unit 100 --pages 1,30,59,88,117       --limit 500 --hosted-only --workers 4 --sleep 0.5 --out $E/image/donated
$G --menu image --license 20,21,22,23,24,25,26,27 --page-unit 100 --pages 1,700,1400,2100,2800 --limit 500 --hosted-only --workers 4 --sleep 0.5 --out $E/image/ccl
$G --menu image --license 01,02,03,04            --page-unit 100 --pages 1,1900,3800,5700,7600 --limit 500 --hosted-only --workers 4 --sleep 0.5 --out $E/image/kogl
```

**STEP 3 — 영상 (30–40 h, 173 GB)** — selection lists already generated; run in background, expired alone:
```bash
nohup $G --menu video --wrtsn-file $E/_select/video_donated_500.txt --limit 500 --hosted-only --max-file-mb 800 --workers 2 --sleep 0.5 --out $E/video/donated > $E/_log/vid_donated.log 2>&1 &
nohup $G --menu video --license 20,21,22,23,24,25,26,27 --page-unit 100 --pages 3,9,15,21,27,33,39,45,51,57 --limit 1000 --hosted-only --max-file-mb 50 --workers 4 --sleep 0.5 --out $E/video/ccl > $E/_log/vid_ccl.log 2>&1 &
wait
nohup $G --menu video --wrtsn-file $E/_select/video_expired_500.txt --limit 500 --hosted-only --max-file-mb 800 --workers 2 --sleep 0.5 --out $E/video/expired > $E/_log/vid_expired.log 2>&1 &
# 영상/KOGL: DO NOT RUN (hosted 0/70). CCL 1,000 covers its share.
```

**STEP 4 — master index:** merge all `gongu_index.xlsx` → `$E/gongu_master_index.xlsx`.

---

## 6. Risks

| Risk | Evidence | Mitigation |
|---|---|---|
| Rate limit / block | 520 req/40 min, zero 429/403 | workers ≤4 (video 2), sleep 0.5 s, keep UA+Referer, run at night |
| **Bandwidth is the real bottleneck** | 0.70–0.78 MB/s single, 1.78 MB/s at w=4 (client-side, not server) | video as 2-night background job. **Oracle server unusable — only 2.8 GB disk** |
| **No resume within a file** | server ignores Range, no Accept-Ranges | `.part` + rename (P5). A failed 400 MB file costs ~9 min retry. Per-work resume works fine |
| Disk | E: 420 GB free, plan 184 GB | **census-sorted selection is mandatory** (page-order = 339 GB). Never use C: (20 GB) or /tmp |
| Terms / bulk collection | anonymous GET, identical code path to the UI (12/12 match); no explicit crawl prohibition found in 이용약관; robots.txt 404 | ~16,000 requests (~2 req/s) is normal-browsing range. State non-commercial research purpose, no redistribution, credit 공유마당 |
| License compliance | codes differ per cell (CCL 8 types, KOGL 4 types, 기증 98/99 differ) | index must store `license_code`·`license_name`·`저작권자`·`출처`·`detail_url` (already in schema). KOGL requires attribution → prepare auto-credit script |
| Representativeness bias | first 500 dominated by single uploaders | stratified pages (verified: author changes at p100/p1200/p3000); post-verify author counts from index |
| 어문 reference-only records | 50.5% of 어문/만료 are 국립중앙도서관 bibliographic records with no file; **list text cannot detect them** (1 of 2,500) | gate on detail `출처` + `원문제공`; use hosted band pages 23–158 @100 |
| Streaming-iframe false positive | all 4 video cells embed the same recommendation carousel | judge hosted **only** by the `원문제공` dd (current code is correct) |

---

## 7. Assets already prepared

- **Video census (full, with exact sizes):** `/mnt/e/gongu_dataset/_select/census_expired.jsonl` (627 works, 252.7 GB total), `census_donated.jsonl` (556 works, 28.4 GB total) — copied out of volatile /tmp.
- **Selection lists:** `video_expired_500.txt` (162.1 GB), `video_donated_500.txt` (10.0 GB).
- **Downloader:** `api/module/dataset_builder/gongu_downloader.py` (needs P1/P2/P3 patches).
- Prior research: `docs/gongu_download_research_20260720.md`.

---

## 8. Open decision for the user

The 1,178-file shortfall is unavoidable. Choose the reporting stance:

- **(A) Report as-is (recommended):** 4,822 files; document the 4 empty/short cells as *verified findings* ("KOGL video/text have no source files on 공유마당"; "어문 기증·CCL populations are 49·273 and were collected in full"). Scientifically the strongest — full-population coverage means zero sampling bias.
- **(B) Backfill to 6,000:** fill the gaps from 어문/만료 (25,200 available) and 영상/CCL. Hits the round number but breaks license balance — must be flagged in the index (`substitute=KOGL→97`).
