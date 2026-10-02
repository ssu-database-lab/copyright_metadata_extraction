# Ground-truth files — inventory (2026-10-02)

There is **no single ground-truth file**. GT is split by **source** (공유마당 vs KOGL), and for
공유마당 also by **media × license bucket**. It is also split into two layers that answer different
questions:

- **Work-side GT** — what is in the file itself (title, author, description, keywords,
  resolution, … — the 11 attributes of the 연구개발계획서).
- **Contract-side GT** — what is printed in the contract (parties, the 7 rights, licence dates,
  title, work type — the TTA 14 attributes).

---

## 1. Work-side GT

| Source | Files | Rows | Layout |
|---|---|---|---|
| 공유마당 (gongu.copyright.or.kr) | 10 files: `D:\copyright_dataset_metadata\{image,video,text}\{expired,donated,ccl,kogl}\ground_truth.jsonl` | 4,834 | **one file per media × license bucket** |
| KOGL 공공누리 (www.kogl.or.kr) | 1 file: `E:\kogl_originals\ground_truth.jsonl` | 1,500 | **all three media in one file**, told apart by the `media` field |

### 공유마당 — per-cell row counts

| Media | expired | donated | ccl | kogl | Total |
|---|---|---|---|---|---|
| image | 503 | 503 | 503 | 500 | 2,012 |
| video | 499 | 498 | 1,003 | — | 2,000 |
| text | 503 | 49 | 273 | — | 825 |

There is no `video/kogl` or `text/kogl` file: 공유마당 holds no original files for KOGL video and
text works (audit trail: `D:\copyright_dataset_metadata\*\kogl\records.jsonl`, 46
`external_no_file` records).

The GT files exist only on D:. `E:\gongu_dataset\` holds the media plus `records.jsonl`,
`issues.jsonl` and `gongu_index.xlsx`.

### KOGL — per-media row counts

| image | video | text | Total |
|---|---|---|---|
| 500 | 500 | 500 | 1,500 |

### What each work-side record contains

Both sources use the same record shape: `id`, `media`, `license_bucket`, `file`, `file_exists`,
`attributes{…}`, and optionally `extra_reference{…}` and `classification{…}`. Every attribute is
`{value, schema_field, source, tier, note}`.

| Attribute | 공유마당 source | KOGL source |
|---|---|---|
| 제목 | site 저작물명 (A) | export 제목 (A) |
| 저자 | site 저작(권)자 (A) | export 저작권자명 (A) |
| 설명 | site 요약정보, filtered (A) | detail-page 저작물 설명, filtered (A) — **251/1,500** (img 83 · video 131 · 어문 37) |
| 라이선스 유형 | site license (A) | 공공누리 유형 (A) |
| 키워드 | site search tags (A) | 주제어, parsed on `#` (A) + `tag_classes` |
| 해시태그 | — (no such field) | 해시태그 as published + `page_value` — **reference only, not scored** |
| 해상도 | site 화질 (A) where it matches the file, else ffprobe (B) — video 2,000/2,000 | image PIL (B), video ffprobe (B) |
| 주요 색상 · 개체 범주 | none (C) | none (C) |
| 파일크기 · 파일포맷 | measured from the file (B) | measured from the file (B) |
| 파일 생성 날짜 | site 창작년도 (D — different meaning, skipped) | none (D) |

**Tiers:**
- **A** — external answer, independent of our extractor.
- **B** — measured from the file. The extractor reads the same file, so this checks that the
  value survives the pipeline, not accuracy.
- **C** — no answer exists.
- **D** — meaning differs; skipped by the scorer.
- **N/A** — not applicable (visual attributes of text works).
- **제외** — excluded; `value` is null and the reason is in `note`. Any collected text is kept
  under `raw`.

**Things to know before reading the scores:**
- The scorer scores any non-empty `value` regardless of tier. Excluded answers must therefore
  stay null.
- **Keywords are mostly collection tags, not content.** `batch_level: true` marks a tag list
  shared word-for-word by five or more works: 2,862 of 4,816 공유마당 records, and 1,056 of
  1,500 KOGL records. Report these separately.
- **99 KOGL image pages return an identical empty page** (87,458 bytes), all from
  한국학중앙연구원, so their description is recorded as *unknown*.
- Video `extra_reference` holds `실측_fps` and `실측_재생시간_sec` (ffprobe), plus the site's
  `재생시간` for 공유마당. The site floors duration to whole seconds.

---

## 2. Contract-side GT

| File | Rows | Layout |
|---|---|---|
| `dataset\contract_work_manifest.csv` | 5,834 | **one CSV for all media** (image 2,415 · video 2,244 · text 1,116 · 59 with no matched work file), 28 `gt_*` columns |
| `dataset\생성계약서_메타데이터_항목.xlsx` | — | the ledger the contracts were generated from — the source of the `gt_*` values |
| `dataset\contracts\contracts_index.xlsx` | — | synthetic personal-data GT (injected names, IDs, contacts) for the older 1,500 `.docx` contracts |

TTA-14 GT records are generated from the manifest at ZIP build time
(`api/module/evaluation/tta_ground_truth.py`). All 14 attributes are contract-sourced; each record
names its own source (e.g. `계약서:제2조 저작자`).

The manifest also carries 9 video technical columns, measured on the file actually shipped in the
ZIP: `gt_width`, `gt_height`, `gt_resolution`, `gt_fps`, `gt_duration_sec`,
`gt_resolution_source`, `gt_resolution_site`, `gt_duration_site_sec`,
`gt_resolution_site_matches_file`.

---

## 3. Eval-set copies

Every eval ZIP or staging folder (e.g. `test_zips\*.zip`, `/home/mbmk92/eval_staging/*/`)
contains its own `ground_truth.jsonl`. These mix media, but each covers only that set's works and
is assembled from the files above. **Edit the sources, not these copies.**

---

## 4. How the two layers link

- **공유마당 works:** each manifest row's `gt_source_record` column points to its per-cell file
  on D:.
- **KOGL works:** no pointer column yet; join on the work ID (`set_id` = `원문인덱스` = `id`).

---

## 5. Supporting indexes and raw data

| File | What it holds |
|---|---|
| `E:\kogl_originals\originals_index.xlsx` | KOGL saved file ↔ original filename; width/height (image PIL, video ffprobe), fps, duration_sec, video_codec, `dims_source` |
| `{E:\gongu_dataset, D:\copyright_dataset_metadata}\video\{bucket}\gongu_index.xlsx` | 공유마당 per-work index + measured video columns + site value + `resolution_match` |
| `D:\copyright_dataset_metadata\video\site_vs_ffprobe.csv` | 공유마당 site resolution/duration vs ffprobe, 2,000 rows (758/762 resolutions exact) |
| `D:\copyright_dataset_metadata\{media}\{bucket}\detail_meta.jsonl` | 공유마당 detail-page fields (화질, 재생시간, keywords, …) |
| `dataset\kogl_gold\kogl_detail_1500_20261002.jsonl` | KOGL detail-page scan: 저작물 설명, page tags, file size (1,500 pages) |
| `dataset\_probe\video_ffprobe.jsonl` | ffprobe cache, 2,711 video files |
| `docs\붙임1_원문저작물 메타데이터.xlsx` | KOGL export (143,946 rows × 75 cols) — source of KOGL title/author/licence/tags |

Backups taken before the 2026-10-02 edits are in `_backup_20261002\` under `E:\kogl_originals`,
`E:\gongu_dataset`, `D:\copyright_dataset_metadata` and `dataset\`.

---

## 6. Regenerating

```bash
# measure video technical metadata → indexes + contract manifest
python -m api.module.dataset_builder.measure_video_technical

# 공유마당 work-side GT (rebuild ALL cells — batch_level counts span every cell)
python -m api.module.dataset_builder.build_ground_truth

# KOGL work-side GT
python -m api.module.dataset_builder.build_kogl_ground_truth

# contract-side manifest (run from api/)
python -m module.dataset_builder.build_contract_work_manifest

# stage an eval set from either source
python -m api.module.dataset_builder.build_eval_manifest --source {gongu|kogl} --out <dir>
```
