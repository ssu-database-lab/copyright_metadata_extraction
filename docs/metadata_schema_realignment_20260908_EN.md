# Metadata Schema Realignment to the TTA Standard

**Date** 2026-09-08
**New spec** `docs/new metadatalist.xlsx` — sheet `메타데이터 표준`, three side-by-side tables
**Target** nested JSON: **10 groups / 66 leaves** (the `JSON 구조도` table)
**Code** `api/module/llm_extraction/schemas/tta_serializer.py` · `document_schemas.py` · `api/module/evaluation/scoring.py`

---

## 1. Summary

| | before | after |
|---|---:|---:|
| Flat extraction schema | 72 fields | **86 fields** |
| TTA leaves covered | — | **50 / 66** |
| TTA evaluation attributes | — | **14** |
| Manifest sets runnable | 5,621 | **5,774 (99.0%)** |

The final output is **nested**; our extraction stays **flat**. The LLM keeps emitting a flat object
and `tta_serializer.build()` folds it into the 10-group structure at the end. This was measured, not
assumed: nesting the extraction directly without a complete rename map drops evaluation accuracy
from 0.483 to **0.000**, because not one key still matches. Folding at the boundary leaves the
consolidator, the NER field-mapper and the scorer untouched.

## 2. Coverage of the 66 target leaves

| status | count | meaning |
|---|---:|---|
| **EXACT** | 19 | our field already has the target name |
| **RENAME** | 19 | same meaning, different name — handled by the serializer map |
| **INJECT** | 12 | not extracted; the pipeline already knows it (filename, model used, consolidation result…) |
| **NEW** | 16 | no source at all — the contract does not carry it |
| **total** | 66 | |

So **50 of 66 (76%)** are produced today. The 16 NEW are mostly group 5 (권리판단) and group 9
(인격권/초상권) items plus `registration_number`, `publication_date` and
`managing_organization_identifier` — none of which appear in the generated contracts.

### Full leaf-by-leaf mapping

| # | group | target leaf | 한글 | our field | status |
|---|---|---|---|---|---|
| 1.1 | identification | `work_identifier` | 저작물 식별자 | — | INJECT |
| 1.2 | identification | `work_title` | 저작물명 | `work_title` | EXACT |
| 1.3 | identification | `file_name` | 파일명 | — | INJECT |
| 1.4 | identification | `registration_number` | 등록번호 | — | NEW |
| 1.5 | identification | `registration_date` | 등록일 | `registration_date` | EXACT |
| 1.6 | identification | `managing_organization_identifier` | 관리기관 식별자 | — | NEW |
| 2.1 | source | `providing_organization` | 제공기관 | `agency_name` | RENAME |
| 2.2 | source | `managing_organization` | 관리기관 | `site_name` | RENAME |
| 2.3 | source | `original_source_location` | 원문 위치 | `url` | RENAME |
| 2.4 | source | `database_name` | 데이터베이스명 | `board_name` | RENAME |
| 2.5 | source | `collection_path` | 수집 경로 | `board_path` | RENAME |
| 2.6 | source | `source_description` | 원천 설명 | `memo` | RENAME |
| 3.1 | work | `work_type` | 저작물 유형 | `work_type` | EXACT |
| 3.2 | work | `media_type` | 매체 유형 | — | INJECT |
| 3.3 | work | `file_format` | 파일 형식 | `digital_format` | RENAME |
| 3.4 | work | `language` | 언어 | `language` | EXACT |
| 3.5 | work | `creation_date` | 창작일 | `created_date` | RENAME |
| 3.6 | work | `publication_date` | 공표일 | — | NEW |
| 4.1 | rights | `author` | 저작자 | `author` | EXACT |
| 4.2 | rights | `copyright_holder` | 저작권자 | `copyright_holder` | EXACT |
| 4.3 | rights | `economic_rights_holder` | 저작재산권자 | `economic_rights_holder` | EXACT |
| 4.4 | rights | `licensor` | 이용허락자 | `licensor` | EXACT |
| 4.5 | rights | `co_authors` | 공동저작자 | `co_author` | RENAME |
| 4.6 | rights | `rights_holder_identifier` | 권리주체 식별자 | `rights_holder_identifier` | EXACT |
| 5.1 | rights | `copyrightability_status` | 저작물성 여부 | `copyrightability` | RENAME |
| 5.2 | rights | `unprotected_work_status` | 비보호저작물 여부 | `unprotected_work` | RENAME |
| 5.3 | rights | `work_made_for_hire_status` | 업무상저작물 여부 | `work_for_hire` | RENAME |
| 5.4 | rights | `rights_ownership_status` | 권리보유 여부 | — | NEW |
| 5.5 | rights | `assessment_basis` | 판단근거 | — | INJECT |
| 5.6 | rights | `assessment_date` | 판단일 | — | NEW |
| 6.1 | public | `public_release_type` | 공개유형 | `disclosure_type` | RENAME |
| 6.2 | public | `previous_public_release_type` | 이전 공개유형 | — | NEW |
| 6.3 | public | `final_public_release_type` | 최종 공개유형 | `kogl_type` | RENAME |
| 6.4 | public | `attribution_required` | 출처표시 필요 여부 | — | NEW |
| 6.5 | public | `commercial_use_permitted` | 상업적 이용 가능 여부 | `commercial_use` | RENAME |
| 6.6 | public | `modification_permitted` | 변경 가능 여부 | — | NEW |
| 6.7 | public | `share_alike_required` | 동일조건 적용 여부 | — | NEW |
| 6.8 | public | `usage_restrictions` | 이용 제한사항 | `special_terms` | RENAME |
| 7.1 | detailed | `reproduction_right` | 복제권 | `reproduction_right` | EXACT |
| 7.2 | detailed | `public_performance_right` | 공연권 | `public_performance_right` | EXACT |
| 7.3 | detailed | `public_transmission_right` | 공중송신권 | `public_transmission_right` | EXACT |
| 7.4 | detailed | `exhibition_right` | 전시권 | `exhibition_right` | EXACT |
| 7.5 | detailed | `distribution_right` | 배포권 | `distribution_right` | EXACT |
| 7.6 | detailed | `rental_right` | 대여권 | `rental_right` | EXACT |
| 7.7 | detailed | `derivative_work_creation_right` | 2차적저작물작성권 | `derivative_work_creation_right` | EXACT |
| 7.8 | detailed | `other_rights` | 기타 권리 | `other_rights` | EXACT |
| 8.1 | validity | `copyright_expiration_date` | 저작권 만료일 | — | NEW |
| 8.2 | validity | `license_start_date` | 이용허락 시작일 | `license_start_date` | EXACT |
| 8.3 | validity | `license_end_date` | 이용허락 종료일 | `license_end_date` | EXACT |
| 8.4 | validity | `contract_validity_period` | 계약 유효기간 | `valid_period` | RENAME |
| 8.5 | validity | `public_release_type_effective_date` | 공개유형 적용일 | — | NEW |
| 8.6 | validity | `validity_review_status` | 유효기간 검토상태 | — | NEW |
| 9.1 | moral | `moral_rights_notes` | 저작인격권 유의사항 | — | NEW |
| 9.2 | moral | `name_attribution_condition` | 성명표시 조건 | — | NEW |
| 9.3 | moral | `integrity_right_restrictions` | 동일성유지 제한 | — | NEW |
| 9.4 | moral | `portrait_rights_included` | 초상권 포함 여부 | `portrait_rights` | RENAME |
| 9.5 | moral | `third_party_rights_included` | 제3자 권리 포함 여부 | `third_party_rights` | RENAME |
| 9.6 | moral | `rights_restriction_description` | 권리 제한 설명 | — | NEW |
| 10.1 | verification | `verification_status` | 검증상태 | — | INJECT |
| 10.2 | verification | `verification_result` | 검증결과 | — | INJECT |
| 10.3 | verification | `verification_date` | 검증일 | — | INJECT |
| 10.4 | verification | `verifier` | 검증자 | — | INJECT |
| 10.5 | verification | `verification_method` | 검증 방식 | — | INJECT |
| 10.6 | verification | `error_type` | 오류 유형 | — | INJECT |
| 10.7 | verification | `review_history` | 검토 이력 | — | INJECT |
| 10.8 | verification | `last_updated_date` | 최종 갱신일 | — | INJECT |

## 3. Fields added to the flat schema (72 → 86)

### 3.1 Rights-holder roles — TTA 4.1/4.3/4.4/4.6

The old schema had one `copyright_holder` described as *"저작권자 (저작권 보유자, 권리자, 양수기관 등)"*,
collapsing four legally distinct roles. Measured across all 5,834 contracts: 권리자 = 저작자 in 71%,
저작자 = 저작권자명 in 68%, all three identical in only 51%. They are not interchangeable.

| new field | TTA | Korean |
|---|---|---|
| `author` | 4.1 | 저작자 — the creator |
| `economic_rights_holder` | 4.3 | 저작재산권자 |
| `licensor` | 4.4 | 이용허락자 |
| `rights_holder_identifier` | 4.6 | 권리주체 식별자 (**not** a birth date — the spec says so explicitly) |

`copyright_holder` (4.2) stays, narrowed to 저작권자 alone.

### 3.2 The seven statutory rights — TTA 7.1–7.7, plus 7.8

`economic_rights` was free text, so per-right accuracy could not be scored at all. The contract's
제2조 prints seven checkboxes, so they are now seven booleans:
`reproduction_right`, `public_performance_right`, `public_transmission_right`, `exhibition_right`,
`distribution_right`, `rental_right`, `derivative_work_creation_right`, plus `other_rights` (7.8).

### 3.3 License period — TTA 8.2/8.3

`valid_period` was a free-text blob. Added `license_start_date` and `license_end_date` as dates.

> **Side effect:** `추출 신뢰도` is `filled_fields / total_schema_fields`, so the denominator moved
> 72 → 86 and the reported number drops for the same extraction. Compare only within one schema version.

## 4. Evaluation changes

### 4.1 The blocker: 저자 was scored against a field the contract never prints

`scoring.py` mapped the ground-truth attribute 저자 to `copyright_holder`. That field is not printed
in the contract — where it differs from 저작자/권리자 it matches **0 of 57** times, and its apparent
81% hit rate is coincidence. It now resolves to `author` first.

To avoid invalidating existing results, `Attr.field` accepts a **tuple of candidates** and takes the
first non-empty one (`author` → `copyright_holder`). Old result files keep scoring; new ones score
the correct field.

### 4.2 New: `TTA_ATTRIBUTES` — 14 attributes

The 계획서 11속성 set is unchanged and still drives our own evaluation. A separate `TTA_ATTRIBUTES`
list defines the TTA scope, with a new `cmp_bool` comparator that accepts the contract's mixed
checkbox glyphs (`v` / `■` / `□`).

| # | target key | Korean | comparator | group | why in scope |
|---|---|---|---|---|---|
| 1 | `work_title` | 저작물명 | contains | 식별 | printed 296/300 (1.6% glyph dropout) |
| 2 | `work_type` | 저작물 유형 | exact | 식별 | 종별 checkbox, 300/300 exact |
| 3 | `author` | 저작자 | contains | 권리주체 | printed 300/300 |
| 4 | `economic_rights_holder` | 저작재산권자 | contains | 권리주체 | 권리자 printed 5,714/5,714 |
| 5 | `licensor` | 이용허락자 | contains | 권리주체 | same source, spec says 저작재산권자 first then 이용허락자 |
| 6 | `reproduction_right` | 복제권 | bool | 세부권리 | R_1, 300/300 |
| 7 | `public_performance_right` | 공연권 | bool | 세부권리 | R_2, 300/300 |
| 8 | `public_transmission_right` | 공중송신권 | bool | 세부권리 | R_3, 300/300 |
| 9 | `exhibition_right` | 전시권 | bool | 세부권리 | R_4, 300/300 |
| 10 | `distribution_right` | 배포권 | bool | 세부권리 | R_5, 300/300 |
| 11 | `rental_right` | 대여권 | bool | 세부권리 | R_6, 300/300 |
| 12 | `derivative_work_creation_right` | 2차적저작물작성권 | bool | 세부권리 | R_7, 300/300 |
| 13 | `license_start_date` | 이용허락 시작일 | date | 유효기간 | 300/300 |
| 14 | `license_end_date` | 이용허락 종료일 | date | 유효기간 | 300/300 |

### 4.3 Excluded, with the reason (report as 미채점, never as 0)

| key | why excluded |
|---|---|
| `file_name` | **never printed** — 0/300, despite the spec marking it 실제 사용 O |
| `copyright_holder` | not printed when it differs from 저작자/권리자 — 0/57 |
| `public_release_type` | no contract states its license — 0/300 mention 공공누리 / CC BY / 기증 / 만료. Score from the work-file/catalog leg instead |
| `work_identifier` | the work ID appears in the filename only — 0/5,714 in the body |

> **The spec's `실제 데이터 사용 여부` column is wrong in both directions.** `file_name` is marked O but
> is never printed; `R_4`–`R_7` and the license dates are marked X but are printed and correct in
> 300/300. The table above follows measurement, not the column.

## 5. Fixes to the blocked evaluation sets

### 5.1 `WORK_EXT_UNPROCESSABLE` (40) — **fixed**

`universal_ocr._process_hwp` was a stub returning `[]`, so every `.hwp` work failed with empty OCR.
`.hwp` is not a scan — it is a binary document that already contains its text — so rasterising it
through OCR would degrade the original and cost API credits for nothing. Added a direct-text
short-circuit (`FileProcessor.extract_text()` → pyhwp) ahead of the image path, plus:

- a magic-byte sniff in `_process_hwp`, which recovers 1 file that is a PDF named `.HWP`
- a guard that rejects text which is mostly `<표>` placeholders (pyhwp drops table cell contents;
  Korean contracts are table-dominant, and the empty-OCR guard cannot catch a non-empty string of
  placeholders) and falls back to the OCR path
- `pyhwp>=0.1b15` in `requirements.txt` — **install it on the Oracle server before the next deploy**,
  or the handler silently reverts to the old behaviour

Measured: **39/40 extract directly** (157–33,431 chars, median 2,879), the 40th routes through PDF.
`.txt` works were broken the same way and are fixed as a side effect.

### 5.2 `CONTRACT_PDF_EMPTY` (118) — **resolved 2026-09-09: 117 recovered**

All 118 HWPX are intact (118 distinct filled contracts). What is missing is only the PDF rendering.

- LibreOffice **cannot** do this — verified by unpacking the packages: only a HWP-97 import filter, no OWPML.
- The already-registered `HwpAutomationApp2.HwpAutomation` instantiates from 32-bit PowerShell but
  **crashes with an AccessViolationException** on `Open()` — it is an in-process server. Do not use it.
- The working route is `HWPFrame.HwpObject`, which needs a one-time COM registration.

The 118 HWPX are staged at `C:\Users\user\AppData\Local\Temp\hwpx_convert\in` and the converter is
`api/module/dataset_builder/tools/convert_118_hwpx.ps1` (also copied next to the staged files).
Register once, then run it from **32-bit** PowerShell — `Hwp.exe` is 32-bit, so the class lands in
the WOW6432Node view and 64-bit hosts cannot see it. Accept only outputs that are non-zero **and
exactly 5 pages**. Estimated ₩6,900 to re-run the contract leg for the 118.

## 6. Files changed

| file | change |
|---|---|
| `schemas/document_schemas.py` | +14 fields (rights-holder roles, 7 statutory rights, license dates) |
| `schemas/tta_serializer.py` | **new** — flat → nested 10×66, cardinality from the spec's 빈도 column (25 arrays) |
| `evaluation/scoring.py` | `Attr.field` accepts fallback tuples; 저자 → `author`; `cmp_bool`; `TTA_ATTRIBUTES` (14) + `TTA_EXCLUDED` |
| `ocr/universal_ocr.py` | direct-text extraction for `.hwp`/`.txt`, magic-byte sniff, placeholder guard |
| `dataset_builder/build_contract_work_manifest.py` | `.hwp` no longer pre-excluded |
| `dataset_builder/tools/convert_118_hwpx.ps1` | **new** — HWPX→PDF recovery |
| `requirements.txt` | `pyhwp>=0.1b15` |

## 6b. Wiring (2026-09-09) — the TTA path now actually runs

Both components were built but unreachable; that is closed.

| gap | fix |
|---|---|
| Serializer not imported anywhere | `PipelineOrchestrator.build_response` folds the flat result and emits **`tta_metadata`** (10 groups / 66 leaves) alongside the existing `metadata`. Additive — no downstream consumer breaks. Wrapped in try/except so a serialization fault cannot lose an extraction. |
| `score_set` hardcoded `ATTRIBUTES` | now takes `attributes=`; `RunConfig.scoring_set` selects `"plan"` (default, the 11 계획서 attributes) or `"tta"` (the 14). |
| TTA attributes had **no ground truth** | the existing `ground_truth.jsonl` is keyed by the 11 계획서 names, so all 14 TTA attributes scored `None`. New `evaluation/tta_ground_truth.py` builds TTA-shaped GT from the manifest. |

The manifest gained the seven 제2조 rights checkboxes as booleans (`gt_reproduction_right` … 
`gt_derivative_work_creation_right`); the source marks them with four glyphs (`■`/`v`/`V` checked, `□` unchecked).

**Verified end-to-end on 5,774 eval-ready sets:** a perfect extraction scores **14/14 = 1.0**, a wrong
one **0/14 = 0.0**, and legacy `score_set` calls still return the same 11 attributes.

Ground-truth coverage:

| attribute | scorable | excluded |
|---|---:|---:|
| 저작물 유형 · 복제권 · 공연권 · 공중송신권 · 전시권 · 배포권 · 대여권 · 2차적저작물작성권 · 이용허락 시작/종료일 | 5,774 | 0 |
| 저작물명 | 5,768 | 6 (accented-glyph titles) |
| 저작재산권자 · 이용허락자 | 5,651 | 123 |
| 저작자 | 5,438 | 336 (`-` / `소속없음`) |

Exclusions carry a reason and tier `제외`, so they report as **미채점**, never as zeros.

Run the TTA evaluation with `RunConfig(scoring_set="tta")` and GT from
`tta_ground_truth.build_records("dataset/contract_work_manifest.csv")`.

## 7. Still open

1. ~~Run `convert_118_hwpx.ps1`~~ **done** — 117 recovered, 5,657 → 5,774 (99.0%).
2. `pip install -r requirements.txt` on the Oracle server (pyhwp).
3. The 16 NEW leaves need a source decision — most are 권리판단/인격권 items absent from the contracts.
4. `app.py:98 ALLOWED_EXTENSIONS` still rejects `.hwp`/`.txt` for **web uploads**. The batch runner
   bypasses it, so the TTA evaluation is unaffected; allowing them in the UI is a separate decision.
5. `.hwpx` extraction is a ~15-line sibling of the `.hwp` handler and would let the generated
   contracts be read directly from the ZIP.
