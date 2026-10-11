# M3 console — addendum: source highlighting on the contract (원본 위치 표시) (2026-10-11)

Show, on the contract page previews the console already renders, a yellow highlight where each extracted value appears in the document (idea from design option E "작업대", whose mockup faked it with retyped HTML). Hover/click a key → its places light up; click a highlight → its key is selected. Display only: extraction results and scores never change.

## 1. Two position sources, chosen per page (no Google — project DashScope account only)
| Page kind | Detection | Positions from | Precision label | Cost |
|---|---|---|---|---|
| Text layer (all current test contracts: new5 5/5, T-002 sample 60/60) | `len(page.get_text().strip()) >= 30` and mostly printable | PyMuPDF `page.search_for()` / words with boxes | 정확 (텍스트 레이어) | none |
| Scan / photo / image-only page (examiner uploads, future contracts) | otherwise; also JPG/PNG/TIFF contracts | **Qwen text-line grounding** on the same page image the preview shows: `qwen3.8-omni-flash` (primary) → `qwen3.6-flash` (thinking off, fallback). Prompt asks for a JSON array `{"text","bbox_2d":[x1,y1,x2,y2]}` per text line, coordinates relative 0–1000; value box = its character span inside the line box | 근사 (OCR 좌표) | ₩1.3/page (omni-flash), ₩3.6/page (3.6-flash) |
| Neither available (DashScope failing, `off`) | — | — | 위치 정보 없음 | none |

Measured 2026-10-11 on a simulated scan (200 dpi, grayscale, blur, JPEG q70) of all 5 pages of `00003_13333800 F!RE 가사.pdf`, 12 value kinds, ground truth = text-layer positions (harness `/home/mbmk92/.claude/jobs/99c123af/tmp/scan/qwen_locate.py`):

| Model | Places found | IoU median (min) | s/page | ₩/page |
|---|---|---|---|---|
| qwen3.8-omni-flash | 30/30 | 0.71 (0.33) | 13.7 | 1.29 |
| qwen3.6-flash | 30/30 | 0.71 (0.42) | 6.8 | 3.64 |
| qwen-vl-ocr-2025-11-20 | 30/30 | 0.62 (0.37) | 6.9 | price not checked |
| qwen3.7-plus (page 1 only) | 10/10 | 0.50–0.73 | 30.6 | ~5.3 |

`qwen-vl-ocr` misreads (양지웅 → 양지용) are absorbed by the fuzzy match (≤ len/6 differing characters); its `rotate_rect` output must swap w/h when the angle is ~90°. Pages are sent **in parallel** (one call per page) so a 5-page scan takes about one page's time (~15 s), once — then cached.

Qwen is used **only for positions**, separate from the extraction OCR chain (qwen3.7-plus → qwen3.8-max), **lazily** when a set's detail is opened (not during batch runs), cached per run. Env `CONSOLE_LOCATE_OCR=qwen|off` (default `qwen`), `CONSOLE_LOCATE_MODELS=qwen3.8-omni-flash,qwen3.6-flash`. Uses the same DashScope key/base URL as the rest of the pipeline (`DASHSCOPE_API_KEY`, `DASHSCOPE_BASE_URL`). **Never call Google services** (the Google credentials belong to a private account).

## 2. What is located (targets)
- 23-key values of the set (prediction record), T-001 keys 1–16 and for T-002 the contract-context keys: 제목(1), 종별(3), 저작자(4, each name), 저작권자(5), rights 6–13 (the right's label on its checkbox line, rect widened to the whole checkbox line incl. ■/□/☑/V glyph), 시작일(14), 종료일(15), 유효기간(16 → both dates).
- 통합 검증 decision fields with a final value (payment amount, signature date, agency, parties, addresses, …).
- Value variants: names split from lists; dates `YYYY-MM-DD` → `YYYY년 M월 D일`, `YYYY년 MM월 DD일`, `YYYY. M. D.`, `YYYY.MM.DD`, `YYYY-MM-DD`; amounts with/without commas, with `원`; 종별 short ↔ long form (어문 ↔ 어문저작물); whitespace-insensitive (Korean spacing differs), NFKC.
- Matching: text layer exact (whitespace-insensitive); OCR path additionally fuzzy (edit distance ≤ max(1, len/6)) → `match: "fuzzy"`.
- Several occurrences: mark all; flag one `primary` — prefer the place the rule/model actually read from when the pipeline records it (e.g. `copyright_holder_source` = 서명란 → signature-block occurrence; title rule → title line), else the first occurrence in the body.
- Not locatable (summaries, keywords, normalised/inferred values, file name/size/hash): listed in `not_found` with a Korean reason (요약·생성 값 / 정규화된 값 / 문서 밖 정보(파일 측정) / 찾지 못함).

## 3. API
`GET /api/test/runs/{run_id}/sets/{set_id}/locate[?refresh=1]` (login required) →
```json
{"available": true, "reason": null,
 "pages": [{"page": 1, "width": 595.0, "height": 842.0, "source": "text_layer|ocr_qwen|none"}],
 "marks": [{"id": "k:creator#0", "target": "k:creator", "key": "creator", "no": 4, "field": "author",
            "label_ko": "저작자", "value": "양지웅", "page": 1, "rects": [[0.437, 0.162, 0.497, 0.177]],
            "match": "exact|variant|fuzzy", "primary": true, "source": "text_layer|ocr_qwen"}],
 "not_found": [{"target": "k:description", "key": "description", "value": "…", "reason": "요약·생성 값"}],
 "engine": {"ocr": "qwen|off|none", "model": "qwen3.8-omni-flash", "pages_ocr": 0, "seconds": 0.02, "cost_krw": 0.0, "cached": false}}
```
Rects normalised 0–1 to the page (x0,y0,x1,y1), so they overlay any preview size. Targets: `k:<key23 name>` for 23 keys, `f:<unified field>` for 통합 검증 fields. Cache: `<run>/locate/<set_id>.json` (+ OCR words per page). Log lines (component 측정 or OCR): `원본 위치 — 텍스트 레이어 5쪽 · 표시 23곳 · 못 찾음 6` / `원본 위치 — 스캔 2쪽 qwen3.8-omni-flash (₩2.6 · 14.1s)`; Qwen errors/fallbacks as WARNING. Works for single runs and batch sets; for contracts uploaded as images/DOCX/HWP use whatever page images the preview renders; if none → `available:false` + reason. Secrets never logged.

## 4. Frontend (console.html)
- Overlay highlight boxes on the contract page thumbnails in the evidence panel and on the large page in the lightbox (`#lb`): absolutely positioned boxes from normalised rects; yellow highlighter in light mode, a dark-mode variant; primary occurrence slightly stronger; fuzzy (OCR) marks dashed outline.
- Hover a row in the 23-key table or the 통합 검증 table → its marks get an outline and the preview scrolls to that page; click → open the lightbox at that page with the marks and a small label (`#4 저작자`). Click a mark → select that key's row (and its 근거).
- Caption/legend under the previews: `노란 표시 = 값이 읽힌 자리` + source per document (`텍스트 레이어 · 정확` / `스캔 · Qwen OCR 좌표 · 근사` / `위치 정보 없음 — 사유`), toggle 표시 켜기/끄기, count `표시 23곳 · 못 찾음 6`.
- In 근거 for a key: `원본 위치: 1쪽 (외 2곳)` link, or `원본에서 찾지 못함 — <reason>`.
- Loading indicator while `/locate` runs (scan path ~15 s for the whole document, pages in parallel); failure → message, previews still shown.
- Style C tokens; Korean; v1.0 only; no horizontal overflow at 1280/1920.

## 5. Verification
1. Text layer: the five new5 contracts (run `api/web/test_runs/20261011_082121_batch_0bb0`) — every T-001 key with a literal value found on the right page; rights on checkbox lines.
2. Scan: rasterise a new5 contract into an image-only PDF (200 dpi, grayscale, slight rotation/blur, JPEG) → same keys found with `source: ocr_qwen`, rects visually on the right words (screenshot check).
3. Locate OCR off (`CONSOLE_LOCATE_OCR=off`) → scan shows `위치 정보 없음`, no errors.
4. Unit tests: variants, normalisation, fuzzy, rect normalisation, cache, auth, not_found reasons. Full suite passes.
