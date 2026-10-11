# M3 — 시험 콘솔 (test console) spec, 2026-10-11

Operator-facing web interface for the certification test, run on the test workstation DESKTOP-NLP-24 by the project lead, who also shows it to the examiners. Decisions (user, 2026-10-11): **style C (계측 콘솔)** from `docs/v4/design/v4_style_options.html` + **style E's evidence panel** (근거 / JSON / 판정 tabs) with **previews**: the image for image works, **every frame sent to the model** for videos, the contract pages for contracts. **Login page** for access control. **Visible version label is "v1.0"** — no "v4", "M3" or internal milestone names anywhere a user can see (titles, headings, labels, file names offered for download).

The existing pages and APIs (`/`, `/v2`, `/v3`, `/pair`, `/api/llm-extract`, `/api/batch/*`) stay unchanged and unauthenticated (a partner service calls `/api/llm-extract`). Only the console routes below require login.

## 1. Files and ownership

| Owner | Files |
|---|---|
| Backend | `api/web/console_api.py` (router `/api/test` + page routes `/console…`), `api/web/console_auth.py` (login, sessions, CLI), `api/web/app.py` (only: include the console router + init), `.gitignore` (`api/web/test_runs/`), cost/time constants in `api/module/evaluation/batch_runner.py` (`_estimate_cost` only), tests `api/module/evaluation/v4/tests/test_m3_console.py` |
| Frontend | `api/web/templates/console.html`, `api/web/templates/console_login.html` (self-contained: inline CSS/JS like the other pages; wrap JS in `{% raw %}…{% endraw %}` where needed) |
| NER | `api/module/ner/ner_system.py` (+ its own tests) |

## 2. Authentication (`console_auth.py`)
- Credentials in `.env`: `CONSOLE_USER` (default `admin`), `CONSOLE_PASSWORD_HASH` = `pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>` (≥ 200,000 iterations), `CONSOLE_SECRET` (random hex ≥ 32 bytes, signs cookies).
- CLI: `python -m api.web.console_auth set-password --user admin [--password …]` (reads stdin when `--password` is absent) writes/updates the three lines in `.env` (creates `CONSOLE_SECRET` if missing; never prints the password or hash).
- Session cookie `console_session` = signed `user|expiry` (itsdangerous `TimestampSigner` or HMAC-SHA256), 12 h lifetime, `HttpOnly`, `SameSite=Lax`, `Secure` only when the request is HTTPS.
- If no password is configured, `/console` shows a page explaining the CLI command; `/api/test/*` returns 503 `{"error":"console_not_configured"}`.
- Throttle: after 5 failed logins from one IP within 10 minutes, further attempts wait 30 s (429 with Korean message). Failed login responds after a fixed ~0.5 s delay.
- Pages: `GET /console/login` (form), `POST /console/login` (`user`, `password`, `next`) → 303 to `next` (only same-site paths), `GET /console/logout` → clears cookie, 303 to login. Unauthenticated `GET /console*` → 303 to `/console/login?next=…`; unauthenticated `/api/test/*` → 401 JSON `{"error":"unauthorized"}`.

## 3. Runs and storage
- Root `api/web/test_runs/<run_id>/`: `run.json` (state, config, timestamps, summary, headline metrics), `manifest.jsonl`, `inputs/` (uploads of single runs), `out/` = BatchRunner out_dir (`test_predictions/`, `test_predictions.jsonl`, `test.log`, `results.jsonl`, `test_metrics.json`, `report.md`, `per_set.jsonl`, `runs/{results,uploads,llm_results}/<request_id>/…`), `cache/` (previews).
- `run_id` = `YYYYMMDD_HHMMSS_{single|batch}_{4 hex}`. States: `queued → running → scoring → done | cancelled | error | interrupted` (interrupted = found running on disk after a restart).
- Each run builds its own orchestrator like `api/module/evaluation/cli.py::build_orchestrator(out_dir)`, so all leg artifacts live under the run. Execution in a background thread with `BatchRunner(..., progress_cb=…)`, `RunConfig(scoring_set="eval23", test=…, frame_schedule="v4", consolidate_work=False by default, only_predicted=True, title_fallback as chosen, tau)`, then `finalize_eval23()` when a gold file applies. **Never re-implement scoring.**
- Concurrency: at most one batch and one single run at a time; otherwise 409 `{"error":"busy","active":"<run_id>"}`.
- Sources (frozen lists; read-only): `t001_new5` → `dataset/test_set_v4/t001_new5_ids.csv`, gold `gold23_t001_new5.jsonl`; `t002` → `t002_ids.csv`, gold `gold23_t002.jsonl`; `pilot200` → `dataset/pilot200_v4/pilot_ids.csv`, gold `gold23_t002.jsonl`; `t001` → `t001_ids.csv` listed but **disabled** with reason `신규 양식 계약서 대기` (contracts pending); `ids` → user-supplied id list resolved against `t002_ids.csv`/`t001_ids.csv`. Manifests via `api/module/evaluation/manifest_v4.py`.
- Single runs: set id = leading 5–8 digit number in the uploaded file name (work file for T-002, contract for T-001), else `upload`. If that id has a gold record in the test's gold file, the run is scored against that one record (only_predicted) so the 판정 tab works; otherwise no metrics.
- Estimates (`GET /api/test/estimate`): per set ₩44.8 and ~30 s effective at 3 workers for T-001; ₩1.34 per image and ₩1.77 per video and ~5 s effective per set at 6 workers for T-002 (measured 2026-10-10/11). Update `batch_runner._estimate_cost` image/video unit costs to these values (qwen3.8-omni-flash).

## 4. API (all under `/api/test`, JSON, login required)
- `GET /status` → `{"version":"v1.0","user":…, "models":{"ocr":"qwen3.7-plus","ocr_fallback":"qwen3.8-max","extraction":"qwen3.5-122b-a10b","ner":"klue-roberta-large","image_chain":[labels…],"video_chain":[labels…]}, "services":{"dashscope":"ok|fail|unknown"}, "active":{"batch":run_id|null,"single":run_id|null}, "sources":[{"id","label","test","n","cells":{cell:n},"gold":bool,"enabled":bool,"reason"}], "defaults":{"tau":0.65,"workers":{"T-001":3,"T-002":6}}}` (service ping cached ≤ 60 s).
- `POST /single` multipart `test`, `contract` (file, T-001 required), `work` (file, T-002 required; contract optional = context only) → `{"run_id","set_id","gold":bool}`.
- `POST /batch` JSON `{"test","source","ids"?:[…],"cells"?:[…],"n"?:int,"sample":"first"|"random","seed"?:int,"workers":int,"tau":float,"only_predicted":bool,"title_fallback":bool,"consolidate_work":bool,"max_cost"?:number,"name"?:str}` → `{"run_id","n_sets","estimate":{"sec","krw"}}`.
- `GET /estimate?test=&source=&cells=&n=` → `{"n_sets","by_cell":{…},"est_sec","est_krw"}`.
- `GET /runs` → `[{"run_id","kind","name","test","source","state","created_at","started_at","finished_at","n_sets","done","failed","cost_krw","macro_f1","micro_f1","passed"}]` newest first.
- `GET /runs/{id}` → that row + `"config"`, `"progress":{"i","total","failed","cost_krw_total","last_set"}`, `"summary"`, `"error"`.
- `GET /runs/{id}/stream?after=<seq>` → SSE `data: {…}` events with `seq`, `ts`, `type` ∈ `start | set_start | set_done | progress | log | state | finalized | error | closed`. `set_done` carries `set_id, ok, elapsed_sec, cost_krw, error`. `log` carries one new line of `out/test.log`. Keep-alive comments every ~5 s. Same resumable `after` semantics as `/api/batch/{id}/stream`.
- `POST /runs/{id}/cancel` → `{"ok":true}`.
- `GET /runs/{id}/metrics` → `test_metrics.json` content plus `"pass_mark":0.85` (404 until scored).
- `GET /runs/{id}/sets?cell=&key=&verdict=&failed=&q=&offset=&limit=` → `{"total","items":[{"set_id","cell","media","title","ok","error","elapsed_sec","cost_krw","keys":{key:{"verdict","value_short"}}}]}`. Verdict per scored key from `per_set.jsonl`: `NA` (gold state na), `TP` (matched, no fp/fn), `PARTIAL` (tag lists with tp>0 and fp or fn; include `set_f1`), `WRONG` (fp>0 and fn>0), `FP` (fp only), `FN` (fn only), `EMPTY_OK` (gold empty, prediction empty); `null` when the key is unscored or no gold.
- `GET /runs/{id}/sets/{set_id}` → `{"record":<prediction json incl. _meta>, "keys":[{"no","key","label_ko","group_ko","test","scored","value","display","source","verdict","gold","detail"}], "evidence":{key:{"source_label","summary","items":[{"label","value"}]}}, "previews":{"work":{"kind":"image|video|pptx|null","image_url","frames":[{"url","t","name"}]}, "contract":{"pages":[{"page","url"}]}}, "legs":{…}, "models":{…}}` (23 keys in `keys23.KEYS` order).
  Evidence sources (assemble from the record's `_meta` and the leg folders' `llm_metadata.json` / `consolidated_metadata.json`): work_title / work_type (rule parse `title_raw`, ticked boxes, model value, chosen source); file_name (uploaded name); creator (model value, consolidation decision if present); copyright_holder (이용자 rule: signature line, preamble, agreement; model value shown as ignored); rights (model value, raw text if unparsed); dates and period (raw `valid_period` text → normalised); keywords / hashtags (model lists, `hashtags_source`, matched pairs and unmatched items from the verdict detail); file_format / file_size / resolution / file_hash (measured values and how: file header, PIL, ffprobe, SHA-256); description (text, cosine vs τ, shared content words, gold kind).
- `GET /runs/{id}/sets/{set_id}/preview/work.jpg` (downscaled ≤ 1280 px; pptx → rendered slide 1), `…/preview/frames/{name}` (files in the work leg's `keyframes/`), `…/preview/contract/{page}.png` (PyMuPDF, 120 dpi, cached under `cache/`). Paths must be resolved inside the run folder or the record's `_meta.files` only (no traversal).
- `GET /runs/{id}/download/{name}` for `test_predictions.jsonl | test_metrics.json | report.md | test.log | per_set.jsonl | run.zip` (zip excludes `runs/uploads`); `GET /runs/{id}/sets/{set_id}/download` → zip of that set's leg artifacts.

## 5. Frontend (`console.html`)
- Base: the **C 계측 콘솔** look from `docs/v4/design/v4_style_options.html` (dark by default, light toggle remembered in localStorage with try/catch; icon rail; monospace amber numerals; status lights; compact tables), reusing its CSS/JS where possible. Korean UI. Title `저작물 속성정보 추출 시스템 v1.0 — 시험 콘솔`.
- Rail: 시험 모드 · 일괄 시험 · 진행 · 결과·지표 · 실행 이력; top bar: system name **v1.0**, service lights from `/status`, user, 로그아웃.
- 시험 모드: T-001 / T-002 toggle, upload zones (contract; or work + optional contract), 실행, live progress from SSE (steps 업로드 → 텍스트·OCR → 추출 → 통합 → 23개 항목, driven by events and log lines), then the **23-key table** (official order and grouping 식별정보 · 권리주체정보 · 권리정보 · 이용허락·계약정보 · 내용·특징정보 · 파일·기술정보; 채점/채점 제외 marking; 허용/불허 chips; NULL; 해당 없음) beside the **evidence panel**: preview area on top (image; video frames strip with timestamps, click to enlarge; contract page thumbnails), tabs **근거** (per-key evidence; clicking a key row scrolls/highlights its evidence), **JSON** (the record, copy/download), **판정** (per-key verdicts with gold values and detail, when gold exists).
- 일괄 시험: source, cells, n + first/random, workers, τ, only-predicted, title fallback, consolidate_work, max cost, name; live estimate; 시작.
- 진행: overall progress, per-set rows with status lights, elapsed, cost; live log tail; 취소.
- 결과·지표: Macro-F1 / Micro-F1 with the 85 % line and 합격/불합격, per-key F1 bars with tooltip, per-cell table, **set × key verdict grid** with filters (cell, key, verdict, failed only, search), row click → the same evidence panel for that set; downloads.
- 실행 이력: runs list, reopen any run (results and evidence still work after a restart).
- Must work at 1440 and 1920 wide; usable at 1280. No external network requests (no CDN, no web fonts).

## 6. Verification
Backend tests with FastAPI `TestClient`, a fake runner and fake orchestrator (no model calls): auth flow (redirects, 401, throttle, cookie expiry), single/batch creation and 409, verdict mapping, evidence assembly on a recorded prediction, preview path safety, downloads, recovery of runs on restart. Final integration on the workstation with real runs: single T-001 (new contract), single T-002 image, single T-002 video, batch T-001 interim 5, batch T-002 small sample; browser checks with Playwright.
