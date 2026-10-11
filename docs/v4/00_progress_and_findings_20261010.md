# v4 certification-test preparation: progress and findings

Status as of **2026-10-10**, branch `eval-harness-v4`. One reference for what is decided, what exists, what the first runs showed, and what is open and who owns it. Numbers in this file were computed from the files on disk when it was written; companion documents are in §0.

## 0. Where things are

| Need | Location |
|---|---|
| Official procedure (shared with 장세영) | `docs/261006_시험평가 절차서.hwp` + `_검토용.pdf`, v1.0 public, 13 pages |
| Our review of the procedure, decisions §12 to §12.9 | `docs/test_procedure_review_20261004.md` |
| Execution plan v2 and run log | `docs/test_execution_plan_20261006.md` |
| Round-3 summary and the message to 장세영 | `docs/test_prep_round3_summary_20261007.md`, `docs/message_to_jang_20261007.md` |
| Investigations 01 to 08 (new-contract keys, pipeline audit, description pilot, …) | `docs/test_prep_investigations_20261007/` |
| M1 evaluator spec and review fixes | `docs/v4/M1_evaluator_spec.md`, `docs/v4/M1_review_fixes_20261009.md` |
| M2 runner spec and review fixes | `docs/v4/M2_runner_spec.md`, `docs/v4/M2_review_fixes_20261010.md` |
| Pilot 200 and τ calibration (started 2026-10-10) | `docs/v4/pilot200_tau_calibration_20261010.md` |
| Evaluator, gold builder | `api/module/evaluation/v4/` |
| Runner, manifest builder, CLI | `api/module/evaluation/{manifest_v4,batch_runner,cli}.py` |
| Test-set draw, description generator | `api/module/dataset_builder/{build_test_set,generate_descriptions}.py` |
| Frozen lists, gold, contracts needed | `dataset/test_set_v4/` |
| Description and hashtag drafts, review workbooks | `dataset/gold_descriptions/t002_full/`, `pilot_200/` |
| Original files | 공유마당 `/mnt/e/gongu_dataset/<media>/<bucket>/`, KOGL `/mnt/e/kogl_originals/` |
| Merged ground-truth index (one row per work) | `dataset/gt_merged_index.csv` (6,398 works) |
| Heavy Python (torch) | run from `/home/mbmk92/copyright/copyright_metadata_extraction`, not from `/mnt/c` |

## 1. Status at a glance

| Workstream | Status |
|---|---|
| Test design (T-001 / T-002, 23 keys, scoring) | Settled; 장세영's answers of 2026-10-08 incorporated (§2) |
| Procedure HWP | v1.0 ready: `#` column, 시험별 평가 항목 rule, Ta-1/Ta-2 rows, NER and similarity models named. Four definition notes kept internal (§2.4). Sending status: user |
| Frozen test lists | T-002 4,000 works (8 cells × 500), T-001 4,000 contracts (200 text + 3,800 image/video drawn inside T-002); seed 20261009, exclusion policy "none" |
| T-002 gold | Complete for the file keys (4,000/4,000); resolution 3,991; description and hashtags 3,996 on the draft tier; keywords 1,809 (§3.3) |
| T-001 gold | Only the 5 new-template contracts; T-001 at scale waits on 장세영 (§3.2) |
| Description and hashtag drafts | 4,000 works drafted (3,996 ok, 4 flagged); human verification not started (174 rows in the verification workbook, 0 verdicts) |
| Code | M1 evaluator and gold builder (commit 9be59c4), M2 runner and adapter (commit 6e9e624); 274 tests green |
| Dev runs | T-001 on 5 contracts: Macro-F1 1.00. T-002 on 3 works: file keys 1.0, description 3/3, hashtags 0.22, keywords 0.17 (§6) |
| Next | Pilot 200 + τ calibration (started 2026-10-10); choose the v4 visual style, then build M3; full runs when contracts arrive (§10) |

## 2. The test as agreed

### 2.1 Two tests, one system
- **T-001**: input is the contract PDF only. Scored keys 1 to 16. Keys 17 to 23 must be null.
- **T-002**: input is the work file (image or video) plus its contract. The contract is read by **rules only** (title and 종별) and passed to the vision model as a hint. Scored keys 17 to 23. Keys 1 to 16 are not scored.
- Pass criterion per test: **Macro-F1 ≥ 85 %** over the scored keys; Micro-F1 is reported as reference.
- The arbiter (LLM consolidation) runs on the contract leg only; work legs run without it in v4.

### 2.2 The 23 keys

| # | key | 항목 | 그룹 | test |
|---|---|---|---|---|
| 1 | work_title | 저작물명 | 식별정보 | T-001 |
| 2 | file_name | 파일명 | 식별정보 | T-001 |
| 3 | work_type | 저작물유형 | 식별정보 | T-001 |
| 4 | creator | 저작자 | 권리주체정보 | T-001 |
| 5 | copyright_holder | 저작권자 | 권리주체정보 | T-001 |
| 6–12 | reproduction / performance / public_transmission / exhibition / distribution / rental / derivative_work _right | 복제권 공연권 공중송신권 전시권 배포권 대여권 2차적저작물작성권 | 권리정보 | T-001 |
| 13 | other_rights | 기타 권리 | 권리정보 | T-001 (N/A everywhere, excluded) |
| 14 | license_start_date | 이용허락 시작일 | 이용허락·계약정보 | T-001 |
| 15 | license_end_date | 이용허락 종료일 | 이용허락·계약정보 | T-001 |
| 16 | contract_validity_period | 계약 유효기간 | 이용허락·계약정보 | T-001 |
| 17 | keywords | 키워드 | 내용·특징정보 | T-002 |
| 18 | hashtags | 해시태그 | 내용·특징정보 | T-002 |
| 19 | file_format | 파일포맷 | 파일·기술정보 | T-002 |
| 20 | file_size | 파일크기 | 파일·기술정보 | T-002 |
| 21 | resolution | 해상도 | 파일·기술정보 | T-002 |
| 22 | description | 설명 | 내용·특징정보 | T-002 |
| 23 | file_hash | Hash 데이터 | 파일·기술정보 | T-002 |

Single source in code: `api/module/evaluation/v4/keys23.py`.

### 2.3 Scoring rules
- Slot F1 per key: a wrong value counts as one FP and one FN; a missing value as FN; a value where gold is empty as FP.
- Three gold states: value, empty (②, the contract has no such item), N/A (③, not applicable, excluded from the count).
- Name lists (저작자, 저작권자): whole-set equality after normalisation; annotations such as "(저작물 3 건)" are stripped.
- Rights: 허용 / 불허 / null. 기타 권리 is N/A on every contract because the generator skips that box, so it is excluded.
- Dates YYYY-MM-DD; 계약 유효기간 = the 제3조 dates as `YYYY-MM-DD~YYYY-MM-DD`.
- keywords and hashtags: set F1 with containment matching.
- description: KURE-v1 sentence-embedding cosine ≥ τ (working value 0.65) plus a content-token check (§2.5).
- 저작권자 (key 5) is the contract's **이용자** party, which equals the ledger 권리자 column; 저작자 (key 4) is the author list.

### 2.4 Definitions we rely on but did not write into the HWP (user decision 2026-10-09)
1. 저작물명: the 제목(제호) value with the leading 5-to-8-digit work id stripped.
2. 파일명: the uploaded file's own name including the extension, as given.
3. 계약 유효기간: 제3조 start and end dates, `YYYY-MM-DD~YYYY-MM-DD`.
4. 저작권자: the 이용자 party; 기타 권리: excluded. (Both confirmed by 장세영.)

### 2.5 Description judge and its calibration
- Embedding model `nlpai-lab/KURE-v1` (pinned revision), CLS cosine; bge-m3 agrees within ±0.02.
- Calibration from the 200-work pilot: shuffled (mismatched) pairs p95 0.516, max 0.571; matched site captions median 0.706; same-style proxy ≈ 0.80. Working threshold **τ = 0.65**, to be re-checked on verified gold (§10.2).
- Two gold tracks: site caption, or generated-then-verified text. Site texts that are only labels, plots or background notes are N/A.

### 2.6 Frames for video (v4 schedule, stills only, no audio)
<10 s: 3 · <60 s: 5 · <300 s: 8 · <900 s: 16 · ≥900 s: 24. Native clips were never better in the pilot and cost about 4× the tokens; OpenAI and Anthropic models take no video input. The gold generator used a superset (4/8/16/24/32).

## 3. Test data

### 3.1 Frozen lists (`dataset/test_set_v4/`, seed 20261009, per-cell random streams)
- T-002: 8 cells × 500 = 4,000 works. image: ccl, donated, expired (공유마당) and kogl = 250 KOGL originals + 250 공유마당 kogl. video: ccl, donated, expired (공유마당) and kogl = 500 KOGL originals.
- T-001: 4,000 contracts = 4 text cells × 50 (the 50th 기증 어문 is 13262821, added on 장세영's advice) + 8 image/video cells × 475 drawn **inside** T-002.
- Exclusion policy "none": the tuning (19), pinned (43) and old dev (29) works are kept because the cell pools hold only 500 to 503 works; overlaps are reported in `summary.json`.
- Interim T-001 list: `t001_new5_ids.csv` = the 5 new-template contracts in `dataset/new_contracts/`.

### 3.2 Contracts: what exists today and what 장세영 must deliver
All 4,000 T-002 work files are on disk. Contract coverage of the T-002 list today (old-template PDFs from the earlier generation):

| cell | works | with contract | without |
|---|---|---|---|
| image/ccl | 500 | 500 | 0 |
| image/donated | 500 | 500 | 0 |
| image/expired | 500 | 500 | 0 |
| image/kogl | 500 | 446 | 54 |
| video/ccl | 500 | 500 | 0 |
| video/donated | 500 | 498 | 2 |
| video/expired | 500 | 499 | 1 |
| video/kogl | 500 | 242 | 258 |
| **total** | **4,000** | **3,685** | **315** |

`contracts_needed.csv` lists 335 works without any contract ({'video': 261, 'image': 54, 'text': 20}). The contract template changed in October, so **T-001 at scale needs 장세영's regenerated contracts for all T-001 works**; today only 5 new-template contracts exist. For T-002 the contract is only a hint, so a T-002 run today would be a shakedown under old or missing hints, not the final measurement (§10.3).

### 3.3 Gold for T-002 (`gold23_t002.jsonl`, 4,000 records)

| key | gold present | N/A | empty |
|---|---|---|---|
| keywords | 1,809 | 2,191 | 0 |
| hashtags | 3,996 | 4 | 0 |
| file_format | 4,000 | 0 | 0 |
| file_size | 4,000 | 0 | 0 |
| resolution | 3,991 | 9 | 0 |
| description | 3,996 | 4 | 0 |
| file_hash | 4,000 | 0 | 0 |

N/A reasons: resolution for the 9 PowerPoint files; keywords where the site tags are collection-level; description and hashtags for the 4 drafts the generator flagged. Tiers: A = site value, B = measured (ffprobe, PIL, SHA-256), draft = model-generated and not yet verified.

### 3.4 Data fixes made in this round
- 4 files had NFD (decomposed) Unicode names on disk while every index used NFC: 3 만료 text PDFs (晋山家狀) and the 기증 video 13288275. Renamed to NFC, records updated.
- 13223002 (만료 video) had no extension; renamed `.wmv`, format gold WMV.
- KOGL video 87248 downloaded and added (spares 87249, 87250 in `/mnt/e/kogl_originals_scratch/`); video/donated +2, video/expired +1 from 공유마당.
- `originals_index.xlsx` on E: restored from backup after the downloader overwrote it; all 1,501 KOGL rows carry status `downloaded`.
- SHA-256 cache: `dataset/_probe/file_sha256.jsonl`; video probes: `dataset/_probe/video_ffprobe.jsonl`.

## 4. Description and hashtag gold generation
- **Why**: site descriptions are missing or unusable for many works, site keywords are often collection tags, and KOGL images have no hashtags. 장세영 agreed that generated and human-verified text is used as gold, hashtags included.
- **Pilot** (`docs/test_prep_investigations_20261007/08_description_pilot/README.md`): Gemma 4 31B, gpt-5.4-mini, Kanana-1.5-V-3B, Qwen3.5-omni-plus, qwen3-vl-235b, qwen3.8-omni-flash and Qwen-VL-Max compared with and without the guideline, frames vs native clips. Verdict: **gpt-5.4-mini** (reasoning medium, ≥ 4,000 completion tokens, JSON mode) drafts; Kanana as a local cross-check; Qwen and Gemma excluded because they are the system's own model family; the Anthropic key had no credits.
- **Guideline v0.3**: minimum 30 characters, 5 to 10 hashtags in `#태그` form in the same call, title-subject rule for 수어 clips, no author-name tags.
- **Full run**: 4,000 works; final status per work: ok 3,996, needs_review 4 (the 9 pptx works failed as unsupported on the first pass, were rendered through PowerPoint and redrafted); about USD 60.5; pptx first slides rendered through PowerPoint; swf/flv frames extracted sequentially.
- **Verification path**: `review_verification_pilot_200.xlsx` (174 rows) then `review.xlsx` (verdict dropdown, `final_text`, `final_hashtags`) → `generate_descriptions.py --import-review` → `gold_descriptions.jsonl` → `gold23 --generated verified`.
- **Pilot 200**: `dataset/gold_descriptions/pilot_200/` = 200 works, 25 per cell, all drafted; this is the calibration set of §10.2.

## 5. Code delivered (branch `eval-harness-v4`)

Recent commits:
```
6e9e624 2026-10-10 v4 M2: test runner, eval23 adapter and outputs for T-001/T-002
9be59c4 2026-10-09 v4 M1: 23-key evaluator and gold builder, frozen test lists, description/hashtag generator
6e06f3a 2026-10-02 Build one ground-truth index across 공유마당 and KOGL, as CSV and Excel
659521e 2026-10-02 Write down where every ground-truth file lives
d80544b 2026-10-02 Give KOGL a ground-truth file of its own, with descriptions, keywords and hashtags
0aa411b 2026-10-02 Measure every video's width, height, fps and duration, and carry them into the ground truth
```

### 5.1 M1, evaluator and gold builder (`api/module/evaluation/v4/`, 197 tests)
keys23 (single source of the 23 keys), normalize, matchers (names, rights, dates, periods, sets with containment), embed_judge (KURE-v1), evaluator (per key, per cell, per gold kind; Macro and Micro F1; bootstrap CI; warnings), evaluate CLI (`test_metrics.json`, `report.md`, `per_set.jsonl`), gold23 (gold from the merged index, generated drafts or verified text, SHA-256 with cache, ledger names).

### 5.2 M2, runner and adapter (77 tests)
- Pipeline: original file name and SHA-256 captured before sanitisation; `eval23_serializer` emits the 23 keys (rights 허용/불허/null, dates normalised, 유효기간 built from the two dates); `party_rules` finds the 이용자 party; `contract_context` parses title and 종별 by rules; rule-parsed title and 종별 take precedence over the model value; work legs skip the arbiter; VLM prompts receive the contract hint and now ask for hashtags; v4 frame schedule; `.pptx` image works render slide 1 through PowerPoint COM; `import api` no longer loads torch.
- Batch: contract-only, work-only and pair manifests; `manifest_v4` builds them from the frozen lists; eval23 scoring set; outputs `test_predictions/<set>.json`, `test_predictions.jsonl`, `test.log`, `test_metrics.json`, `report.md`; validation gating; direct-text PDFs not billed as OCR; thread-safe context parser; `--only-predicted` restricts gold to the predicted sets for dev and pilot runs.
- Schema: the flat unified schema grew from 86 to 87 fields (`hashtags`), which also moved the web app's 추출 신뢰도 denominator. **Decided 2026-10-10: keep 87 fields.**

### 5.3 Known limitations before the full runs
- The NER model (KLUE-RoBERTa-Large, about 1.4 GB) is loaded per call, so T-001 runs are capped at about 3 workers until it is cached.
- PowerPoint rendering exists only on the Windows workstation; the 9 pptx sets cannot run on the Oracle server.
- OpenRouter spend (Gemma 4 for images) is not in the ₩ estimate; the estimator books images at the DashScope unit price.
- The self-hosted Gemma URL is unreachable from the workstation; image legs go through OpenRouter.
- Qwen outputs vary between runs; single runs carry noise on the content keys.
- Manifest paths are recorded as physical `/mnt/c/...` paths; cosmetic.

## 6. Dev-run results (2026-10-10, after the follow-up fixes)

### 6.1 T-001, 5 new-template contracts, contract-only input
| metric | value |
|---|---|
| Macro-F1 / Micro-F1 | 1.00 / 1.00 |
| per key | all 15 scored keys 1.0; 기타 권리 excluded (N/A on all 5) |
| 저작권자 | 이용자 rule → 한국저작권위원회 on 5/5 |
| 저작물명 source | rules on 5/5 (model titles had whitespace drift on 3/5) |
| OCR | text layer used (direct_text), 5 pages each, no OCR cost |
| time | 75 to 107 s per contract at 3 workers (LLM ∥ NER, then arbiter) |
| cost | ₩44.8 per contract (estimator) |

### 6.2 T-002, 3 works (안동한지 image + contract, 반찬가게 video + contract, KOGL 4K video without contract)
| key | F1 | note |
|---|---|---|
| file_format, file_size, resolution, file_hash | 1.0 | identical regardless of contract |
| description | 1.0 (3/3 at τ 0.65) | cosines 0.68, 0.70, 0.76 |
| hashtags | 0.22 | KOGL site hashtags (충남, 홍성, 광천, 4K) are invisible in the frames; generated-kind track 0.34 |
| keywords | 0.17 | gold keywords are mostly collection tags |
| Macro-F1 | 0.77 | n = 3, content keys dominate |
Contract hint parsed by rules in 0.13 s on both contract sets; work legs 7.6 to 11.8 s; ₩4.2 total. Earlier 6-work passes (with the context race and without rule precedence) scored Macro-F1 0.70 to 0.73.

## 7. Findings and gotchas worth keeping
1. **LLM title drift**: qwen3.5-122b re-typed titles without a space where the PDF line wrapped, and added one before "(". The rule parser was exact. Fix: rule precedence for title and 종별.
2. **Lazy import under threads**: the context parser's lazy import had a race; at more than one worker most sets silently lost the hint. Fix: lock plus per-set warning.
3. **Phantom OCR cost**: text-layer PDFs were billed as OCR pages (₩58.3 instead of ₩44.8 per contract). Fix: bill only when a provider ran.
4. **Backend ping stall**: an unreachable self-hosted Gemma URL cost about 250 s per image leg. Fix: 10 s ping timeout and stop at the first reachable backend.
5. **Partial runs vs full gold**: scoring 6 predictions against 4,000 gold sets counts 3,994 misses. Use `--only-predicted` for dev and pilot runs.
6. **Site hashtags are not visual**: place and campaign tags cannot be recovered from pixels; generated and verified hashtags are the right gold.
7. **PowerPoint files under 이미지**: 9 works in image/ccl; slide 1 is the image for both gold and system; Windows only.
8. **Unicode file names**: NFD names on disk break NFC-keyed indexes; normalise when building indexes.
9. **Downloader gotchas**: KOGL `download_originals.py` defaults `--out` to `dataset/originals` and rewrites `originals_index.xlsx` with processed rows only (use a scratch dir); 공유마당 `--limit` counts existing works; take the last record line per wrtSn.
10. **HWP editing**: the user re-saves the file from 한글 between edits; compare with the installed copy before editing; headers carry a literal page total; `AllReplace` returns False even when it worked.
11. **Generator details**: gpt-5.4-mini at medium reasoning needs ≥ 4,000 completion tokens; Kanana needs `timm` and an SDPA patch; swf has no `format.duration`, so fall back to stream then index; seek-based extraction drops swf frames, so extract sequentially.
12. **Secrets**: credentials live only in gitignored env files and are never printed or committed; secrets of other projects do not belong in this repository folder.
13. **Schema growth**: adding `hashtags` moved the unified schema to 87 fields and the 추출 신뢰도 denominator with it.

## 8. Open items and owners

| Item | Owner | Blocks |
|---|---|---|
| Regenerate contracts (new template) for all T-001 works; 335 works have none (`contracts_needed.csv`) | 장세영 | T-001 full run, final T-002 hints |
| Confirm 이용자 = ledger 권리자 on a regenerated sample where 저작권자명 ≠ 저작자; PDF naming `<원문인덱스> <제목>.pdf` | 장세영 | T-001 gold at scale |
| Send the HWP (status unknown to this log) | user | lab scheduling |
| Human verification of the 200 pilot drafts, then all 4,000 | lab member | verified gold, τ calibration |
| τ sweep on verified gold; keywords N/A policy for collection tags | us | description and keywords scoring |
| Cache the NER model so T-001 can run at 8 workers | us | T-001 run time |
| Meter OpenRouter spend; exercise the 16/24-frame path on a long video | us | cost report, video legs |
| Choose the v4 visual style from the style options page, then build M3 (§10.1) | user, then us | lab-facing interface |

## 9. Cost and time, spent and projected (estimator figures, not invoices)

| Item | Spent | Projected |
|---|---|---|
| Description and hashtag drafts (4,000) | ≈ USD 60.5 + small pilots | verification is human time |
| Dev runs (T-001 ×5, T-002 ×3, final pass) | ₩228; earlier passes ≈ ₩300 | |
| T-001 full run (4,000 contracts) | | ≈ ₩179,000 at ₩44.8 each; ≈ 32 h at 3 workers, ≈ 12 h at 8 workers with cached NER |
| T-002 full run (4,000 works) | | ≈ ₩4,500 DashScope (videos ₩2, images ₩0.2) + OpenRouter for 2,000 images (unmetered, order of tens of USD); ≈ 4 to 5 h at 6 workers |

## 10. Next milestones

### 10.1 M3, the web interface for the test (not started; visual style chosen first)
Operator-facing pages inside the existing FastAPI app, Korean UI. The visual style is chosen first from a set of style options; the single-document pages stay as they are. A first M3 attempt was stopped on 2026-10-10 before any file was written.
- 시험 모드 (one set): choose T-001 or T-002, upload the contract (T-001) or the work plus its contract (T-002), run, see the 23 keys in the official order and grouping with 허용/불허/NULL/N/A rendering, download the JSON. This is what the lab examiner uses.
- 일괄 시험 (batch): build a manifest from the frozen lists (whole test, one cell, N sets, or an uploaded id list), set workers, τ, gold file and only-predicted, start, watch per-set progress live, cancel; results as a per-set table of prediction vs gold with key-level verdicts, a metrics page (Macro and Micro F1, per key, per cell, pass/fail at 85 %), downloads of `test_predictions.jsonl`, `test_metrics.json`, `report.md`, `test.log`; run history.
- Backend endpoints over `manifest_v4`, the eval23 `BatchRunner` and `evaluate`; the v3 batch API stays.

### 10.2 Pilot 200 with τ calibration (started 2026-10-10; analysis in `docs/v4/pilot200_tau_calibration_20261010.md`)
1. A human verifies the 200 pilot drafts (accept, edit or reject) → verified gold for 200 works, 25 per cell.
2. Run T-002 on the same 200 works → 200 system descriptions and hashtag sets.
3. Compute the KURE-v1 cosine for each system/gold pair and have the human mark which pairs truly describe the same content.
4. Choose τ where the human yes/no separates best, compare with 0.65, and report the per-cell description pass rate at that τ.
5. By-product: a first T-002 Macro-F1 on 200 sets before spending on 4,000. Cost ≈ ₩250 plus one or two reviewer days.

### 10.3 Full runs
- T-002 on all 4,000 can run technically today (files and gold exist), but with old or missing contract hints for the vision model; the 4 file keys would be final, the 3 content keys would be re-run once 장세영's contracts arrive. Treat such a run as a shakedown, not the measurement.
- T-001 on 4,000 waits on the regenerated contracts.

## Appendix A. Commands (run from the WSL-native repo path)
```bash
cd /home/mbmk92/copyright/copyright_metadata_extraction
venv/bin/python -m pytest api/module/evaluation/v4/tests -q -p no:cacheprovider        # 274 passed

# T-001 dev run (5 contract-only sets)
venv/bin/python -m api.module.evaluation.manifest_v4 --ids dataset/test_set_v4/t001_new5_ids.csv --test T-001 --out runs/t001/manifest.jsonl --validate
venv/bin/python -m api.module.evaluation.cli --manifest runs/t001/manifest.jsonl --test T-001 --out runs/t001 --gold dataset/test_set_v4/gold23_t001_new5.jsonl --workers 3 --only-predicted

# T-002 dev run (chosen works, contract hint where a contract exists)
venv/bin/python -m api.module.evaluation.manifest_v4 --ids dataset/test_set_v4/t002_ids.csv --test T-002 --work-ids 13304428,13313249,87248 --out runs/t002/manifest.jsonl --validate
venv/bin/python -m api.module.evaluation.cli --manifest runs/t002/manifest.jsonl --test T-002 --out runs/t002 --gold dataset/test_set_v4/gold23_t002.jsonl --workers 6 --tau 0.65 --only-predicted

# Rebuild the T-002 gold (draft tier, with SHA-256)
venv/bin/python -m api.module.evaluation.v4.gold23 --test T-002 --ids dataset/test_set_v4/t002_ids.csv --out dataset/test_set_v4/gold23_t002.jsonl --generated draft --hash

# Re-draw the frozen lists (membership is stable; refreshes file paths)
venv/bin/python api/module/dataset_builder/build_test_set.py --exclude none --out dataset/test_set_v4
```
Outputs of a run: `<out>/test_predictions/<set_id>.json`, `test_predictions.jsonl`, `test.log`, `test_metrics.json`, `report.md`, `per_set.jsonl`.

## Appendix B. People
장세영 연구원 (coordinator, generates the contracts) · 무하유 왕일 프로 (platform) · HM컴퍼니 박명찬 이사 (type classification) · 수요기관 한국문화정보원, 한국저작권위원회.
