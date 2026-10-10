# OCR model re-test after the incumbent was withdrawn

2026-10-11. Our OCR model `qwen3-vl-235b-a22b-instruct` now returns 404 "Model not exist" on our DashScope account, and it stays gone after the billing fix. Qwen staff recommended `qwen3.7-plus` and offered to raise its rate limit as usage grows. This re-test decides the replacement.

## Method

The same benchmark that chose the incumbent in September (`docs/ocr_robustness_and_arbiter_20260908_EN.md` §2): 40 degraded page images from two documents (the 10 conditions that discriminate, including dim phone photo, heavy blur and low resolution), anchors of proper nouns, numbers and dates read from magnified originals, known misreadings counted as traps, the same prompt, temperature 0, 3 repeats. Five new configurations × 120 calls = 600 calls, all successful. September's results for the incumbent and five other models are the reference, from the same images and anchors. Harness: `/home/mbmk92/eval_staging/ocr_confirm/` (`run_round3.py`, `report_round3.py`, `results_round3.jsonl`); full output in `docs/v4/ocr_retest_20261011_raw.txt`.

## Results

| Model | Accuracy | vs incumbent, paired 95 % CI | Traps | Median s | p90 s | Output tokens |
|---|---|---|---|---|---|---|
| **qwen3.7-plus, thinking off** | **91.7 %** | **+4.26 pp [+2.75, +5.86]** | **36** | **12.6** | 29.9 | 448 |
| qwen3.7-plus, default (thinking on) | 90.0 % | +2.48 pp [+0.54, +4.40] | 39 | 25.9 | 40.5 | 1,291 |
| qwen3.8-max-0902 (Sep) | 89.3 % | +1.82 pp [+0.28, +3.54] | 34 | 38.1 | — | 3,263 |
| qwen3.7-flash, thinking off | 87.9 % | +0.46 pp [−1.55, +2.28] | 34 | 10.3 | 27.5 | 460 |
| qwen3-vl-235b-a22b-instruct (Sep, withdrawn) | 87.4 % | — | 87 | 12.7 | — | 586 |
| qwen3.8-flash (current fallback) | 86.5 % | −0.93 pp [−2.85, +0.93] | 42 | 15.5 | 42.9 | 931 |
| qwen3-vl-plus | 86.5 % | −0.92 pp [−2.67, +0.79] | 57 | 12.9 | 34.1 | 590 |

qwen3.8-flash scored 86.5 % now against 86.4 % in September, so the benchmark has not drifted and the comparison with September's numbers holds.

Hardest conditions (dim phone photo / heavy blur / low resolution): qwen3.7-plus thinking off 63.5 % / 80.6 % / 89.6 %, against the incumbent's 58.1 % / 75.2 % / 82.9 %.

## Cost per 4,000 pages

qwen3.7-plus at $0.40 / $1.60 per million input / output tokens (July price list), with the measured tokens: **about $7.0 (₩9,800) with thinking off**, $12.4 with thinking on. September's figures for comparison: incumbent ₩12,633, qwen3.8-flash ₩6,182, qwen3.8-max-0902 ₩119,986. Prices for qwen3-vl-plus and qwen3.7-flash were not available; confirm with Qwen staff, along with whether thinking and non-thinking output are priced the same.

## Recommendation

- **Switch the OCR model to `qwen3.7-plus` with thinking explicitly off.** It is significantly more accurate than the withdrawn model, makes less than half the traps (wrong names or numbers written into metadata), runs at the same speed and costs less.
- **Thinking must be disabled in the call** (`extra_body={"enable_thinking": False}`). The model thinks by default: twice the time, three times the output tokens and lower accuracy.
- **Keep `qwen3.8-flash` as the fallback.** It is wired, measured twice and stable. `qwen3.7-flash` (thinking off) is slightly better and faster but not significantly so.
- Code changes for the switch: `api/module/ocr/alibaba_ocr.py` (default model, model list, thinking flag, fallback map entry `qwen3.7-plus → qwen3.8-flash`), `api/ocr_config.json`, and the image-VLM fallback `QWEN_MODEL` in `api/module/clip_extraction/vlm/extractor.py`, which also points to the withdrawn model. Note the production OCR call uses temperature 1.0 and top_p 0.8, not 0 as in this benchmark.

## Round 3b (same day): fallback candidates and production sampling

Alibaba's notice of 2026-07-08 confirms the incumbent was retired on 2026-10-10 together with the whole qwen3-vl open-weight line, qwen-vl-plus/max and qwen3-vl-flash. Four more configurations were run on the same benchmark (480 calls, all successful):

| Model | Accuracy | vs incumbent, 95 % CI | Traps | Median s | p90 s | Output tokens |
|---|---|---|---|---|---|---|
| **qwen3.8-max, thinking off** | **92.0 %** | +4.54 pp [+2.33, +7.19] | **20** | 15.0 | 29.6 | 458 |
| qwen3.7-plus, thinking off, production sampling (temperature 1.0, top_p 0.8) | 91.6 % | +4.10 pp [+2.50, +5.65] | 39 | 12.9 | 30.0 | 448 |
| qwen3.6-plus, thinking off | 90.0 % | +2.58 pp [−0.15, +4.75] | 35 | 13.6 | 25.5 | 442 |
| qwen3.8-omni-flash | 81.8 % | −5.57 pp [−7.30, −3.87] | 62 | 13.4 | 31.9 | 767 |

- **qwen3.8-max with thinking off** is the most accurate model measured and makes the fewest traps; with thinking off it runs at 15 s instead of 38 s. Against qwen3.7-plus (thinking off) the accuracy difference is not significant (+0.33 pp, CI [−1.18, +1.93]) but the trap difference is (−16 traps over 120 calls, CI [−31, −3]).
- **Production sampling is fine:** qwen3.7-plus at temperature 1.0 / top_p 0.8 scores 91.6 %, the same as at temperature 0.
- **Cost per 4,000 pages** at list prices: qwen3.7-plus ≈ $7.0 (₩9,800); qwen3.8-max ($2 / $6 per million tokens) ≈ $31.7 (₩44,400).

## Configuration applied (2026-10-11)

- **Primary OCR: `qwen3.7-plus`, thinking off. Fallback: `qwen3.8-max`, thinking off.** The fallback runs only when the primary fails, so its higher price matters little, and it belongs to a different generation, which protects against a 3.7-only outage. If misreads matter more than cost, the two can be swapped.
- `api/module/ocr/alibaba_ocr.py`: `DEFAULT_MODEL`, `FALLBACK_MODEL`, `THINKING_OFF_MODELS` (flag applied on all three request paths), `RETIRED_MODELS` (old names such as `qwen3-vl-235b-a22b-instruct` or `qwen-vl-plus`, for example from partner clients, are mapped to the primary with a warning instead of failing), fallback map.
- `api/module/ocr/universal_ocr.py`, `api/web/app.py`, `.env` (`ALIBABA_MODEL=qwen3.7-plus`), batch CLI help.
- Vision pipeline: the Qwen image backend (`vlm/extractor.py`, also `compare.py`, `standalone_gemma_host.py`) now uses `qwen3.7-plus` with thinking off; `VLMClient` accepts `extra_body`.
- Web pages (`index.html`, `index2.html`, `v3.html`): OCR choices are now Qwen3.7-Plus (recommended, default), Qwen3.8-Max and Qwen3.8-Flash; all three old choices were retired models.
- Tests: 287 pass, including the new checks for the default, retired-name mapping, thinking flag and fallback. Live check through the real module: thinking off, no fallback, 13 to 15 s per page, scores in line with the benchmark.
- Not changed: retired names still listed in the LLM model catalogs (`cloud_extractor.py`, `base_extractor.py`, `llm_extractor.py`); no default uses them. The Oracle server still runs the old code; its OCR falls back to `qwen3.8-flash` until it is redeployed.
