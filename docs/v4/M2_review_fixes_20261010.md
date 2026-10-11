# v4 — M2 review fixes (2026-10-10)

Branch `eval-harness-v4`. Fixes applied to the M2 implementation (`docs/v4/M2_runner_spec.md`) after the review of the T-001 (13387752) and T-002 (13304428) smoke runs. Tests: `venv/bin/python -m pytest api/module/evaluation/v4/tests -q -p no:cacheprovider`.

## Fixed

| Sev | Where | Change |
|---|---|---|
| BUG | `batch_runner._is_direct_text` | Reads `response["ocr_provider_used"]` (the provider actually used — `build_response` keeps `ocr_provider` = the *requested* provider, `alibaba`). Text-layer contracts no longer pay 2.7 ₩/page of phantom OCR (T-001 leg 58.3 → 44.8 ₩; ~54,000 ₩ over 4,000 contracts, and `--max-cost` tripped early). `_leg_info` now records `ocr_provider_used` and `ocr_pages`. The test double `FakeOrch` emits `ocr_provider="alibaba"` + `ocr_provider_used="direct_text"` like the real pipeline. |
| RISK | `batch_runner._run_one_eval23` | Both `orch.run(...)` calls pass `test=test`, so the pipeline-side `test_prediction` in `runs/results/<rid>/llm_metadata.json` follows the same T-001/T-002 null rules as the scored record (`build_prediction_record` stays the authority). |
| RISK | `vlm/extractor.VLMExtractor._probe_chain` | Stops at the first reachable backend (per-call fallback in `extract()`/`extract_video()` already covers the rest). `probe_log` lists only the backends actually pinged. |
| RISK | `vlm/client.VLMClient.ping` | Pings with `with_options(timeout=VLM_PING_TIMEOUT_SEC (default 10 s), max_retries=0)`; extraction keeps its own 300 s/1-retry client. A `GEMMA_URL` stuck in SYN-SENT cost ~250 s per image leg (272.85 s observed) — now ≤ 10 s, and only when it is ahead of the selected backend. Operational note still applies: point `GEMMA_URL` at a reachable server or leave the default `http://127.0.0.1:8001/v1` (instant ECONNREFUSED) before the T-002 run. |
| NIT | `batch_runner._leg_info` | `keyframes` records `{kept, target, sampled, duration, schedule}` from the response dict (was always `None`: `len()` of a dict-shaped value guarded by `isinstance(list)`). |
| NIT | `batch_runner._meta` / `_context_payload` | `prompt.contract_context_used` is True only when the rule parse yielded a title or a 종별 (otherwise the prompt is unchanged and `contract_context=None` is passed). |
| NIT | `pipeline._contract_rules` | `contract_context_rules["pdf"]` carries the original file name (`ctx["original_filename"]`), not the sanitised storage name. |
| NIT | `batch_runner.RunConfig` / `finalize_eval23` / `cli` | `content_check`, `embed_cache`, `embed_revision`, `n_boot` are forwarded to `v4.evaluate.evaluate()`; CLI flags `--content-check on|off --embed-cache --embed-revision --n-boot`. The web batch API keeps the defaults (no new form fields). |
| NIT (partial) | `manifest_v4.REPO_ROOT`, `cli._HERE` | `os.path.abspath` instead of `Path.resolve()` — harmless, and keeps a symlinked import path as given. **Limitation:** for `python -m …` started from the symlinked cwd (`/home/mbmk92/copyright/…`) Python's `os.getcwd()` already returns the physical `/mnt/c/…` path, so module `__file__`s and therefore manifest/`_meta.files` paths are still `/mnt/c/…` (verified after the fix). Recording WSL paths would need an explicit root option; not worth it for ~100 KB contracts. |
| NIT | `docs/v4/M2_runner_spec.md` A2 | Key 5 wording reconciled with the code and A5: rule result `licensee_party` first, model `licensee`/`user` only as fallback. |

## Scope note (A8)

`api/module/__init__.py` was changed in addition to the `api/__init__.py` named in spec §4: the old `module/__init__.py` imported `module.ner.ner_system` (torch) eagerly, so `python -m api.module.evaluation.v4.evaluate` could not start in < 3 s without the same PEP 562 lazy-attribute treatment. Behaviour for `from module import ner_predict` / `module.ocr_complete` is unchanged.

## Deliberately not changed

- **T-002 result finding (keywords F1 0.133 / hashtags 0.308 / description cosine 0.657 on 13304428)** — not a code defect; gold keywords are collection tags the VLM cannot see. Feeds the M1 §5d τ sweep and the keywords `na` policy (M1 §3). Asking the VLM for hashtags explicitly is a prompt decision for the dev T-002 run, not a fix.
- **`api/web/app.py` SSE path** (`build_response` without `ocr_result=`/`contract_context_rules=`) — outside the M2 write scope; `ocr_provider_used` on the web SSE path equals the requested provider until app.py is changed.
- **`pipeline.build_response` injecting `licensee_party` into the live `metadata`/`consolidated_metadata` dicts** — copying would also detach `consolidated_metadata.json` from the later `_backfill_file_attributes` (same-object side effect the web app relies on today); needs a decision on what that file should contain, so left as is.
