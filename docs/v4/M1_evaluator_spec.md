# v4 — M1 spec: 23-key gold builder + evaluator (2026-10-09)

Branch `eval-harness-v4`. Package: `api/module/evaluation/v4/` (new; the existing `api/module/evaluation/*.py` = v3 harness, do not modify).
Authority for every rule below: `docs/test_procedure_review_20261004.md` §8, §10, §12 (12.1, 12.3, 12.7, 12.9) and
`docs/test_prep_investigations_20261007/08_description_pilot/README.md` §5–§5d. When this spec and those differ, this spec wins (it is the condensed, decided version).

## 0. The two tests

| | T-001 공유저작물 속성정보 정확도 | T-002 멀티모달 저작물 속성정보 정확도 |
|---|---|---|
| input | contract PDF only | work file (image / video) [+ contract parsed by rules for context] |
| scored keys | 1–16 | 17–23 |
| unscored keys | 17–23 → must be `null` in the prediction; ignored | 1–16 → ignored (may be copied from the contract parse) |
| set id | work_id of the work the contract is about (= 원문인덱스 / wrtSn) | work_id |

Pass mark: **Macro-F1 ≥ 85 %** over the scored keys. Micro-F1 reported as reference.

## 1. The 23 keys (`keys23.py`)

| # | key | 하위요소 | type | T | match rule |
|---|---|---|---|---|---|
| 1 | work_title | 저작물명 | str | 1 | `norm_text` exact; prediction may carry a leading `^\d{5,8}\s+` id token → strip before comparing |
| 2 | file_name | 파일명 | str | 1 (and 2, unscored) | exact after NFC + whitespace collapse (Latin case-insensitive); **T-001 gold = basename of the contract PDF actually given to the system**: `gold23` takes it from the frozen ids csv `contract_pdf` column when present, else `<ledger 파일명>.pdf` (`--file-name-from auto`, default; `ids` / `ledger` force one source). The two agree once the requested naming `<원문인덱스> <제목>.pdf` is honoured and differ for the 5 interim contracts (`00002_13387752 운수 자판기 (1).pdf`) — the ids csv wins, so the frozen list must point at the files actually fed to the extractor. `.pdf` is part of the value (the contract prints the 제호 without an extension; the extractor must emit the input file's basename). T-002 gold = work file basename |
| 3 | work_type | 저작물유형 | str (closed set) | 1 | normalized to one of 어문/음악/연극/미술/건축/사진/영상/도형/컴퓨터프로그램/기타 (strip trailing 저작물); exact |
| 4 | creator | 저작자 | list[str] | 1 | set equality after `norm_name` (NFC, remove spaces, drop `(저작물 N 건)` annotations, split on newline/`,`/`·`) |
| 5 | copyright_holder | 저작권자 | list[str] | 1 | **= the 이용자 party of the contract** (preamble "저작재산권 이용자 X (이하 "이용자"라 함)" / signature "이용자 : X"); same matching as 4 |
| 6 | reproduction_right | 복제권 | rights | 1 | value match 허용/불허 |
| 7 | performance_right | 공연권 | rights | 1 | value match |
| 8 | public_transmission_right | 공중송신권 | rights | 1 | value match; the sub-rights 방송권·전송권·디지털음성송신권 are NOT separate keys and never count as 기타 권리 |
| 9 | exhibition_right | 전시권 | rights | 1 | value match |
| 10 | distribution_right | 배포권 | rights | 1 | value match |
| 11 | rental_right | 대여권 | rights | 1 | value match |
| 12 | derivative_work_right | 2차적저작물작성권 | rights | 1 | value match |
| 13 | other_rights | 기타 권리 | — | 1 | **gold state ③ for every contract** (no 기타 box): prediction must be `null`; excluded from scoring (never FP) |
| 14 | license_start_date | 이용허락 시작일 | date | 1 | `norm_date` → YYYY-MM-DD exact |
| 15 | license_end_date | 이용허락 종료일 | date | 1 | same |
| 16 | contract_validity_period | 계약 유효기간 | str | 1 | `YYYY-MM-DD~YYYY-MM-DD` built from 제3조 (= 14 and 15); exact after `norm_period` (accept `~`, `-`, `–`, `부터/까지` forms) |
| 17 | keywords | 키워드 | list[str] | 2 | set-F1 with containment (§4) |
| 18 | hashtags | 해시태그 | list[str] | 2 | set-F1 with containment after stripping `#` |
| 19 | file_format | 파일포맷 | str | 2 | upper-cased extension without dot, aliases JPG=JPEG, TIF=TIFF, MPEG4=MP4; exact |
| 20 | file_size | 파일크기 | int bytes | 2 | exact integer (accept "1,234", "1234 bytes", "1234B"; KB/MB strings → not accepted) |
| 21 | resolution | 해상도 | str `WxH` | 2 | exact after `norm_resolution` (`1920 x 1080`, `1920*1080`, `1920×1080` → `1920x1080`) |
| 22 | description | 설명 | str | 2 | embedding rule (§5) |
| 23 | file_hash | Hash 데이터 | str | 2 | SHA-256 hex, case-insensitive exact; accept `sha256:` prefix |

Rights values: prediction may be `허용|불허|null|true|false|"O"|"X"|"Y"|"N"` → normalize to 허용/불허/null. A trailing parenthetical is stripped first (`허용(방송권, 디지털음성송신권)` → 허용 — the sub-rights are not keys, #8). **Checkbox glyphs (`v`/`V`/`■`/`□`) are not accepted prediction values**: they are ledger/gold notation only; an extractor that echoes them is scored wrong (unparseable → FP+FN).
Dates (#14/#15) must be bare dates in one of the §4 forms — `2009년 09월 27일부터`, `2009-09-27 (계약일)` are unparseable (= wrong); the 부터/까지 form is accepted only inside #16.

## 2. Gold states (per set × key) — `gold23.jsonl`

Each gold record: `{"set_id", "test": "T-001"|"T-002", "media", "bucket", "source", "keys": {key: {"value": ..., "state": "value"|"empty"|"na", "source": str, "tier": str, "kind": str|null}}}`

- `value` — gold exists → scored normally.
- `empty` (② 확정 빈값) — gold confirmed empty: prediction `null` = TN (not counted), prediction non-null = **FP only**.
- `na` (③ 미수집/비적용) — excluded from both numerator and denominator for that key (e.g. other_rights everywhere; hashtags where no gold; resolution for 어문; description where only a label/plot/background text exists and no verified generated gold).

Description `kind` ∈ {caption, generated, label, plot, background} and hashtag `kind` ∈ {site, generated}; the evaluator reports description/hashtag F1 **per kind** (two tracks: site vs generated) as well as combined.

## 3. Gold sources (`gold23.py`) — build from the frozen lists

Inputs (all exist; read-only):
- `dataset/test_set_v4/t002_ids.csv`, `t001_ids.csv` (columns: test, cell, sub_source, work_id, media, license_bucket, source, title, work_file, has_contract, contract_pdf, work_설명_tier, work_해시태그_tier, work_해상도_tier).
- `dataset/gt_merged_index.csv` (6,394 rows; key columns: work_id, media, source [공유마당|공공누리(KOGL)], license_bucket, title, work_file, work_저자, work_설명, work_설명_tier, work_키워드 (`|`-separated), work_키워드_tier, work_키워드_batch_level, work_키워드_내용태그, work_해시태그, work_해시태그_tier, work_해상도, work_해상도_tier, work_파일크기, work_파일포맷, video_duration_sec, contract_* columns incl. contract_pdf, contract_gt_title, contract_gt_rights_holder, contract_gt_licensee, contract_gt_*_right, contract_license_start/end (inspect the header for exact names), contract_work_content_dup_group).
- Contract ledger `dataset/생성계약서_메타데이터_항목.xlsx` sheet `데이터 랜덤 값` (5,834 rows; columns 파일명, 권리자, 이용자, 저작권자명, 제목, 저작자, R_1..R_7, R_3_1..3, C_1..C_10, C_기타, 이용허락기간_SY/SM/SD/EY/EM/ED (inspect exact names), 양도_*). **Mapping for the NEW template (장세영 2026-09-22+):** work_title ← 제목 (bare); creator ← 저작자; copyright_holder ← **권리자 column** (printed as the 이용자 party in the new template; `-` or empty → state ③); rights 6–12 ← R_1..R_7 (`v`/`V`/`■` = 허용, `□` = 불허); other_rights ← state ③; 14/15 ← 이용허락기간 start/end; 16 ← `start~end`; file_name ← see §1 #2 (`--file-name-from auto`: ids csv `contract_pdf` basename, else `<파일명>.pdf` where 파일명 = ledger 파일명). Ledger 파일명 is `<id> <제목>`, optionally with a generation sequence prefix and a download suffix (`00002_13387752 운수 자판기 (1)`); the id is the 5–8 digit token. `load_ledger` raises (ValueError) when it can read an id from none or from fewer than half of the non-empty rows — a changed naming convention must fail loudly, not yield an all-na T-001 gold. work_type ← ledger C_1..C_10 checked category (map to the closed set). A `--ledger` path option must allow swapping in 장세영's new ledger when it arrives (same columns expected; log unknown columns).
- Generated descriptions/hashtags: `dataset/gold_descriptions/t002_full/drafts.jsonl` (fields: work_id, draft_text, draft_hashtags, status, site_gold_kind) — **provisional** gold (kind `generated`, tier `A-검수(대기)`) until `gold_descriptions.jsonl` (verified; fields description, hashtags, verification_outcome) exists in the same folder — prefer verified when present (`--generated verified|draft|none`).
- Site description: index `work_설명` with tier A → kind from `site_gold_kind` logic: label (same text ≥3 rows in media×bucket×source), plot (video/expired), background (video, KOGL), else caption. Only `caption` is scoreable as site gold; others → state `na` unless generated gold exists.
- Hashtags: KOGL originals have site hashtags (`work_해시태그`, kind site); 공유마당 works → generated (kind generated) or `na`.
- Keywords: `work_키워드_내용태그` if present else `work_키워드` with batch-level lists (`work_키워드_batch_level == True`) → state `na` (collection tags, review §8.3); tags normalized by `norm_tag`.
- file_format / file_size / resolution: index values (`work_파일포맷`, `work_파일크기`, `work_해상도`); resolution for text media → `na`; missing → `na`.
- file_hash: SHA-256 of `work_file`, computed once and cached in `dataset/_probe/file_sha256.jsonl` (path, size, mtime, sha256); `--hash` flag required to compute (slow on /mnt/e); otherwise state `na` with source "not computed".
- file_name T-002: basename of `work_file`; T-001: §1 #2 — `--file-name-from auto` (default) | `ids` | `ledger`.

CLI: `python -m api.module.evaluation.v4.gold23 --test T-002 --ids dataset/test_set_v4/t002_ids.csv --out dataset/test_set_v4/gold23_t002.jsonl [--generated draft|verified|none] [--hash] [--ledger path] [--file-name-from auto|ids|ledger]` and the same for T-001. Print a coverage table: per cell × key → counts of value/empty/na.

## 4. Matching rules (`normalize.py`, `matchers.py`)

- `norm_text(s)`: NFC, strip, collapse whitespace, remove zero-width/bidi chars; comparisons case-insensitive for Latin.
- Names (`norm_name`): NFC, remove ALL whitespace, remove `(저작물 N 건)`/`(N건)` annotations, split lists on `\n , ; · /`; set equality → match (whole key, not per element).
- Lists (keywords, hashtags): `norm_tag` = strip `#`, remove whitespace/`_`/`-`/punctuation, casefold. Element match = equal OR one contains the other with len ≥ 2; one-to-one greedy matching (longest first). TP = matched pairs, FP = unmatched predicted, FN = unmatched gold; **the key's F1 for that set = 2TP/(2TP+FP+FN)** and these TP/FP/FN also feed the micro totals. If gold list is `empty` state and prediction non-empty → FP = len(pred).
- Dates: accept `YYYY-MM-DD`, `YYYY.MM.DD`, `YYYY/MM/DD`, `YYYY년 M월 D일`, `YYYYMMDD` → `YYYY-MM-DD`.
- Scalar keys: match → TP (1), mismatch → FP 1 + FN 1 (wrong value = both), prediction null with gold value → FN 1, prediction non-null with gold `empty` → FP 1, gold `na` → skipped.

## 5. Description rule (`embed_judge.py`)

- Model: `nlpai-lab/KURE-v1`, CLS token of `last_hidden_state`, L2-normalised, fp16 on CUDA if available else CPU; `max_length` 512; pin the HF revision via `--embed-revision` (record it in the output); cache embeddings by sha1(text) in `--cache` jsonl. Every cache line also carries `model`, `revision`, `device`; when a run serves every text from the cache (model never loaded) the recorded `revision`/`device` are taken from the cache lines — adopted only when unambiguous, mixed values → warning and `null` — so a warm re-run writes the same `settings.embed` as the cold run (§7 determinism). An explicit `--embed-revision` is recorded as given and lines of another revision are skipped.
- Match = cosine ≥ τ (**default τ = 0.65, provisional** — calibration evidence and the pending §5d sweep are in `M1_review_fixes_20261009.md`; freeze τ only after the sweep over all caption-gold T-002 sets; CLI `--tau`) **AND** content check (CLI `--content-check on|off`, default on): at least one shared content token between prediction and gold, where tokens = `norm_tag` of maximal runs of Hangul/Latin/digits of length ≥ 2, excluding a small stoplist (이, 그, 저, 있다, 있는, 보인다, 모습, 사진, 영상, 이미지, 장면, 것, 수, 등, 및, 와, 과, 의, 를, 을, 은, 는, 가, 이다, 한다, 된다). Report both the cosine and the token overlap per set in the per-set output.
- Scalar accounting as in §4 (match TP / mismatch FP+FN / null FN).
- Two tracks: compute description F1 separately for gold kind `caption` and `generated` (and `combined`).

## 6. Prediction input

`test_predictions/<set_id>.json` (one per set) or `test_predictions.jsonl` — object with the 23 keys (missing key = null) plus optional `_meta` (request ids, timings). Unknown extra keys ignored; key order irrelevant. T-001 predictions must have keys 17–23 null — the evaluator reports violations (non-null unscored keys) as warnings, never as FP.

## 7. Metrics (`evaluator.py`) — `test_metrics.json` + `report.md`

For the chosen test: per key → TP, FP, FN, n_scored (sets with gold state value/empty), precision, recall, F1, 95 % CI (bootstrap over sets, 1,000 resamples, seed 0); **Macro-F1 = mean of per-key F1 over scored keys** (keys with n_scored = 0 are excluded and listed), Micro-F1 from summed TP/FP/FN; the same per cell (media × bucket); description/hashtag by kind; counts of gold states per key; list of sets missing a prediction (counted as all-FN for keys with gold value); warnings (unscored keys non-null, unparseable values). `report.md`: the tables above in Korean headers matching the HWP (항목/정밀도/재현율/F1-score/최종 지표). Deterministic output (sorted keys, fixed seeds).

CLI: `python -m api.module.evaluation.v4.evaluate --test T-002 --gold dataset/test_set_v4/gold23_t002.jsonl --pred <dir or jsonl> --out <dir> [--tau 0.65] [--content-check on] [--cells] [--embed-cache ...]`.

## 8. Tests (`api/module/evaluation/v4/tests/`, pytest)

Hand-made fixtures, no network: (a) every normalizer (dates in 5 forms, rights in 8 forms, resolution in 4 forms, names with annotations, tags with # and spaces, id-prefixed titles); (b) matchers (scalar match/mismatch/null/empty/na; list containment one-to-one; set equality for names); (c) end-to-end evaluate on 6 synthetic sets × 23 keys with known TP/FP/FN totals → exact Macro/Micro values; (d) embedding judge with a stub embedder (inject a function) to test τ and the content check; (e) gold23 builder on a 10-row synthetic index/ledger (temp files) covering each gold state. All tests must pass with `venv/bin/python -m pytest api/module/evaluation/v4/tests -q`.

## 9. Constraints

Python 3.12, pandas, openpyxl, numpy, torch/transformers (only in embed_judge, imported lazily). No network except the one-time HF model download (already cached). Korean comments/log messages as in the rest of the repo. No edits outside `api/module/evaluation/v4/`, `docs/v4/`. Do not read or modify `.env`, `jiran.env`, or anything under `/mnt/d`, `/mnt/e` except reading the index/lists under `dataset/`. Keep functions small and typed; a module docstring in each file stating the rule section it implements.
