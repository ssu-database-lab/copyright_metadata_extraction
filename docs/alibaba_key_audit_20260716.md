# Alibaba 구 API 키 사용 추적 가이드 (2026-06 감사)
_웹 조사(4-agent workflow) + 로컬 포렌식. **근본 원인: 구 키가 public GitHub + CLI 배포판에 노출.**_

## ⚠️ 로컬 포렌식 결과 (최우선)
- `OCR/google_vision/test_alibaba_api_call.py` (public repo, Initial commit부터)에 키 2개 하드코딩:
  - `sk-95ba…` (35자) → 구 운영키, 현재 **폐기됨**(rotated). CLI tarball `.env`에도 동봉되어 장세영 연구원에 배포됨.
  - `sk-ad15…` (주석 처리) → **아직 LIVE** (dashscope.aliyuncs.com/CN 엔드포인트 HTTP 200). 즉시 콘솔에서 삭제 필요.
- 6월 청구 모델(qwen3.7-max/qwen3.6-plus/happyhorse-1.1-r2v)은 우리 코드가 쓰지 않음 → **유출 키 오남용 지문**.

---

# Alibaba Cloud Model Studio (Singapore/intl) — June 2026 Rogue-Key Audit Guide

## 1. What IS traceable — ranked by usefulness

### #1 — Billing detail export, broken down by API Key ID (the core evidence; retroactive, permanent)
**What it proves:** WHICH API key (including the rotated/dead one — its ApiKeyID persists in historical bills), WHICH model, WHICH workspace, WHICH day (per-minute billing cycle since 2025-11-06), WHICH invocation channel (app / SDK / OpenAI-compatible), and token volumes/cost. This alone attributes the qwen3.7-max / qwen3.6-plus / happyhorse-1.1-r2v charges to the specific key.

**Exact path (console login only, no RAM AK needed):**
1. User Center → Expenses and Costs → Bill Details: `https://usercenter2-intl.console.alibabacloud.com/finance/expense-report/expense-detail-by-instance`
2. Filter: Product = "Alibaba Cloud Model Studio", billing cycle = June 2026, Statistic Item = **Billable Item**
3. **Export Bill (CSV)**. The Instance ID column is a semicolon-separated string: `billing_type;workspace_ID;model_name;input/output_type;invocation_channel` (e.g. `text_token;llm-xxx;qwen-max;output_token;app`). Bill rows also carry the **ApiKeyID** — the docs explicitly say: "Copy the ApiKeyID from your bill and go to the Model Studio API Key Management page to find the corresponding key name." (https://www.alibabacloud.com/help/en/model-studio/bill-query-and-cost-management)
4. Group rows by ApiKeyID × model × day → your full June attribution table.

Granular per-key billing has existed since 2024-09-07, so June 2026 is fully covered. Billing records do not expire like monitoring data — this is your durable evidence.

**Programmatic equivalent (needs RAM AK/SK):** BSS OpenAPI `DescribeInstanceBill` / `QueryInstanceBill` (bssopenapi 2017-12-14) returns the same instance-level rows incl. InstanceID (up to 50k rows, 10 QPS). https://www.alibabacloud.com/help/en/user-center/developer-reference/api-bssopenapi-2017-12-14-queryinstancebill

### #2 — Cost Overview daily trend, filtered by API Key ID
**What it proves:** day-by-day spend profile of the rogue key vs. your keys — the fastest visual confirmation.
**Path:** Model Studio console → Dashboard → Usage & Billing → Cost overview: `https://modelstudio.console.alibabacloud.com/?tab=dashboard#/costing-balance/overview` — "filter by monthly or daily statistics, product category, API Key ID, or model." (https://www.alibabacloud.com/help/en/model-studio/model-usage-statistics)

### #3 — Model Monitoring / Call Statistics (30-day retention — EXPIRING NOW)
**What it proves:** per-API-key, per-model call counts, tokens, failures, latency at **minute/hour granularity**. Retention is **30 days** — as of 2026-07-14, only calls after ~June 14 remain. June 1–13 is already gone. **Do this today.**
**Path:** Model Studio console → target workspace → Monitoring: `https://modelstudio.console.alibabacloud.com/ap-southeast-1/?tab=model#/model-telemetry` → find each rogue model → "Monitor" → Call Statistics tab → filter by API key + time range. (https://www.alibabacloud.com/help/en/model-studio/model-telemetry) Screenshot/export everything before it ages out.

### #4 — ActionTrail: console logins + RAM/control-plane events, WITH source IPs (90-day retention)
**What it proves:** whether the leak was account compromise rather than a leaked key. Every `ConsoleSignin` event (root account AND RAM users, including **failed** attempts) carries `sourceIpAddress`, `userAgent`, `mfaChecked`. RAM events (CreateUser, CreateAccessKey, policy changes) likewise. (https://www.alibabacloud.com/help/en/actiontrail/product-overview/audit-events-for-ram-user-logon, https://www.alibabacloud.com/help/en/actiontrail/user-guide/logon-events-by-using-an-alibaba-cloud-account)
**Path:** ActionTrail console → Events → Event Query, **Singapore region** (global events like ConsoleSignin are queryable only there — conveniently your home region). Filter June 2026. Retention: 90 days default → June evidence starts expiring ~Aug 30 and is fully gone ~Sep 28, 2026. To preserve it permanently, create a trail to SLS/OSS now (a trail only captures post-creation events; historical backfill requires a support ticket). (https://www.alibabacloud.com/help/en/actiontrail/user-guide/query-events-in-the-actiontrail-console)
Also filter Service Name = `bailian` and search for Token/key-related events — the CN doc lists Bailian control-plane events (CreateToken etc.) but the intl supported-services list omits Model Studio, so coverage on your account is unverified; check empirically. (https://help.aliyun.com/zh/actiontrail/product-overview/the-audit-events-of-the-big-model-service-platform)

### #5 — Support ticket for gateway-side caller IPs
The **only** possible source of per-request IPs. Open a ticket citing the ApiKeyID, the exported bill rows, and the June time windows, and ask Alibaba Cloud to pull DashScope/MaaS gateway access logs for that key. Undocumented, not guaranteed — but the bill export gives them exactly what they need to search.

## 2. What is NOT possible — bluntly

- **You cannot get caller IPs for the inference calls yourself. Full stop.** No console page, log, or API exposes per-request source IP for data-plane calls to `dashscope-intl.aliyuncs.com` or `llm-*.maas.aliyuncs.com`.
- **ActionTrail does NOT record DashScope/Model Studio inference calls.** Model Studio is entirely absent from the intl supported-services list (only DashVector appears under AI), and ActionTrail data events cover only OSS and Tablestore. sk- keys are not RAM credentials — they never appear in ActionTrail's AccessKey dimension. There is no eventName, sourceIpAddress, or userAgent anywhere for your June inference traffic. (https://www.alibabacloud.com/help/en/actiontrail/product-overview/services-that-work-with-actiontrail)
- **`GetRequestLog` (OpenAPI Explorer diagnostics) is the wrong tool.** It's a point lookup by a single management-plane (POP gateway) RequestId — it does return callerIp/AK/userAgent, but only for RAM-signed OpenAPI calls, not Bearer-sk data-plane inference, and it cannot enumerate calls by key/time anyway. (https://www.alibabacloud.com/help/en/openapi/developer-reference/api-openapiexplorer-2024-11-30-getrequestlog)
- **Inference logs (request/response records) cannot be recovered for June.** The Logs/inference-log feature must be enabled **before** the calls (nothing retroactive), and content capture is documented as limited to some models in China (Beijing) — not Singapore. Even where available, the log fields do not include caller IP. (https://www.alibabacloud.com/help/en/model-studio/model-telemetry)
- **The rotated key no longer appears in API Key Management** (deleted keys vanish from the list; no "last used" attribute exists) — but its ApiKeyID survives in bills, which is why #1 works. (https://www.alibabacloud.com/help/en/model-studio/get-api-key)
- **June 1–13 minute-level monitoring data is already gone** (30-day retention).

## 3. Practical attribution plan

**Step A — Per-key split (bill export).** Pivot the June CSV by ApiKeyID. If ALL rogue-model rows (qwen3.7-max, qwen3.6-plus, happyhorse-1.1-r2v) carry the old key's ID and none carry your current key's ID, the leak is confined to the old key and rotation closed it.

**Step B — Model-mix fingerprint.** Your pipeline only ever calls: `qwen3.5-122b-a10b` (extraction + consolidation), `qwen3-vl-235b-a22b-instruct` (OCR), `qwen3.5-flash` (OCR fallback), `qwen3.5-plus` (consolidation fallback). Any bill row for another model = not your traffic, period. `happyhorse-1.1-r2v` (image generation) is especially damning — your system has zero image-gen code. Also check the **invocation_channel** component of Instance ID: your code uses the OpenAI-compatible/SDK channel; rows via `app` (Model Studio application channel) would be another foreign fingerprint.

**Step C — Temporal correlation with your own logs.** Build the per-day (bill) and per-hour (monitoring, ≥June 14 only) profile of the rogue key, then diff against:
- Oracle server: `journalctl -u copyright-api --since "2026-06-01" --until "2026-07-01"` on 150.230.114.9 (`ssh oracle`) — timestamps of your real `/api/llm-extract` runs.
- Local WSL runs: shell history / any run logs under `api/web/results/{request_id}/` (directory mtimes give you invocation times).
Rogue activity at hours when neither machine ran anything (e.g., your KST off-hours vs. a foreign timezone's working hours) strengthens the case and hints at the abuser's timezone — the closest you'll get to "where" without gateway logs.

**Step D — Rule out account compromise.** ActionTrail June query: every ConsoleSignin (success + failed) with source IP. If all logins are your IPs, the vector was key leakage (git history, .env in the 1.2GB CLI tarball, Colab notebook, shared machine), not the account. Note: the old key traveled in `copyright_extraction_cli.tar.gz` and possibly the Colab notebook — check what was distributed.

**Step E — Escalate.** Ticket with bill rows + ApiKeyID + time windows requesting gateway IP logs; if the spend is significant, also request a billing dispute/goodwill credit for unauthorized use — the per-key bill attribution is exactly the evidence they ask for.

## 4. Next 15 minutes in the console + programmatic prep

**Do now (console login only):**
1. **(5 min)** Bill Details → Product = Model Studio, June 2026, Statistic Item = Billable Item → **Export CSV**. This is the one irreplaceable artifact — get it saved locally.
2. **(3 min)** Monitoring (`ap-southeast-1` workspace) → each rogue model → Call Statistics → filter old-key + June 14–30, hourly → **screenshot/export before it expires** (loses another day, daily).
3. **(3 min)** Cost Overview → daily trend filtered by old ApiKeyID → screenshot.
4. **(2 min)** ActionTrail → Event Query (Singapore) → June → eyeball ConsoleSignin IPs; also try Service Name = bailian.
5. **(2 min)** Harden the CURRENT key: API Key Management (`https://modelstudio.console.alibabacloud.com/?tab=dashboard#/api-key`) → set the **IP whitelist** (up to 20 IPv4/IPv6/CIDR; default is 0.0.0.0/0 = open to the world) to your Oracle server IP (150.230.114.9) + home/lab IPs, and restrict the **model scope** to your four models. Enable Monitoring (audit + inference logs) so any future misuse IS logged. Consider ActionTrail trail → SLS for permanent history.

**Prepare for programmatic pulls (if you want us to script it):**
- Create a RAM user (console → RAM → Users → Create), attach `AliyunBSSReadOnlyAccess` (bill APIs) + `AliyunActionTrailReadOnlyAccess`, generate an AK/SK pair, put it in `.env` as `ALIBABA_CLOUD_ACCESS_KEY_ID` / `ALIBABA_CLOUD_ACCESS_KEY_SECRET`. With that we can script `QueryInstanceBill` (full June rows, grouped/pivoted automatically) and `LookupEvents` (2 QPS) instead of manual console work. Do NOT reuse the sk- Model Studio key for this — RAM AK and sk- keys are entirely separate credential systems.

**Deadline summary:** monitoring data — dying daily, June 14+ only; ActionTrail June events — safe until ~Aug 30; bill export — permanent, but export it now anyway as the anchor for the support ticket.