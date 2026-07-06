# AGENTS.md - Product Metadata Extraction Context

This file is the product-side context handoff for agents working inside
`metadata/`.

## Scope

`metadata/` is now the product/API tree. Keep it focused on the runnable
metadata extraction pipeline, current training inputs, current model artifacts,
and current OCR/NER outputs.

Research drafts, experiment sweeps, thesis materials, archived runs, and old
model comparisons have been moved to the sibling `../paper/` tree. The old
`metadata/old/` tree and remaining old/backup config/output directories were
archived under `../paper/old/metadata/`.

## Current Product State

- The public API facade is `module/api.py`.
- Live extractors live under `module/extractor/`.
- Shared schema/path helpers live under `module/parts/`.
- `main.py` is the single local inference entry point for the current extraction pipeline.
- Current OCR inputs and derived OCR metadata are under `data/out/ocr/`.
- Current 35-label metadata outputs are under `data/out/metadata/`.
- Historical/review NER metadata snapshots remain under `data/out/ner/`,
  including the `bert-base/clean_20260516` snapshot currently used for review.
- Current product model artifacts remain under `models/`.
- Downloaded base models remain under `model_downloaded/`.

## Pipeline

The product pipeline is a one-way extraction flow exposed as a single function
`module.api.extract_metadata`. `main.py` calls only that function.

1. OCR (`module/extractor/ocr/ocr.py`) — Qwen3-VL-2B-Instruct (Apache-2.0, BF16,
   GPU-only). PyMuPDF renders PDFs to PIL images; Qwen3-VL returns Korean text
   per page. CPU is rejected (`OCRDeviceError`).
2. Regex (`module/extractor/regex.py`) — 9 strict-format labels in
   `REGEX_LABEL_SET` (phone, email, copyright_url, copyright_uci, date,
   ri_money, copyright_num, copyright_idnum, copyright_quantity).
3. NER (`module/extractor/ner/`) — 17 free-form span labels in `NER_LABEL_SET`,
   token classification. **Default: `FacebookAI/xlm-roberta-base`, threshold 0.25**
   (`configs/labels.yaml::ner.model_name`). Chosen by a 2026-07 backbone tournament
   scored on per-label gold accuracy (not silver seqeval); full fine-tune on silver
   + augmentation. Other backbones were removed — retrain from silver if needed.
4. **Post-process** (`module/extractor/ner/postprocess.py::postprocess_metadata`) —
   deterministic cleanup (drives gold 17/17 NER labels to relaxed ≥0.90):
   - `CLOSED_VOCAB`: keyword search for ri_copyright, ri_contract_type, ri_info,
     copyright_type, copyright_status, copyright_language — **vocab-first, NER
     fallback** (was replace-NER, which discarded any answer outside the closed
     list). `copyright_status` also accepts file extensions (`_extract_file_ext`).
   - `FORM_CUE_PATTERNS`: line-capture cues (성명:, 주소:, 전화번호:, 기관명:, …)
     unioned with NER output.
   - Gazetteer/lexicon recovery: 지자체→address (`_extract_region_address`),
     org suffix/prefix→company (`_extract_org_company`), position lexicon + credit
     roles→position. Fixes NER mislabeling 지자체/기관 as name, name-precision-safe.
   - Heuristic filters (length, Hangul/CJK, period-or-date, substring dedup);
     `ri_data` falls back to regex `date`.
   Note: training augmentation lifted the model itself (address raw 0.60→0.99,
   company 0.60→0.74), so these rules now correct a smaller residual gap.
5. LLM extraction is still a stub in `module/extractor/llm/`. `extract_metadata`
   accepts an `llm_fn` callback; without it, the 9 delegated labels are filled
   with `["N/A"]` placeholders.

## Schema And Data

- `module/parts/labels.py` is the source of truth for label sets.
- Current training silver data is `configs/integrated/silver/`.
- Current external evaluation gold data is `configs/integrated/gold/`.
- Historical config backups were removed from this product tree and archived in
  `../paper/old/metadata/current_tree/`.

The clean schema is a disjoint 3-way split of 35 labels:

- 9 REGEX labels (strict surface format).
- 17 NER labels (free-form Korean spans).
- 9 LLM-delegated labels (policy / context judgment).

The final xlsx export maps these 35 clean labels back to 50 fields. The 7
role-free author labels (`name`, `address`, `phone`, `company`, `department`,
`email`, `position`) are expanded by the LLM/export stage into 21
role-prefixed `ch_co_*` / `ch_ja_*` / `ch_nr_*` fields. Source xlsx typos
(e.g. `ch_co_addres`, leading-space ` ri_portrait`) must be preserved in
output.

## Operational Notes

- Prefer editing `module/api.py`, `module/extractor/`, and `module/parts/` for
  product behavior.
- Keep research-only analysis, paper writing, sweeps, and large archived
  comparison results out of `metadata/`; use `../paper/` for that work.
- Do not recreate `old/` or `backup/` directories in the product tree unless the
  user explicitly asks for a new local archive.
- If training defaults change, update `main.py` and this file together.
- If schema labels change, update `module/parts/labels.py`,
  `NER_LLM_METADATA_CONNECTION.md`, and this file together.
- Keep historical label-decision notes and source-audit details in
  `../paper/docs/`, and label-level Gold source distributions in the paper
  config snapshot, not in this product tree.

## Next Product Work

- Continue reviewing the `data/out/ner/bert-base/clean_20260516` metadata JSON
  outputs against OCR metadata.
- Decide whether the regex-first hybrid path should become the default inference
  path for all current document batches.
- Implement the LLM stage only after the clean NER/regex output contract is
  stable.
