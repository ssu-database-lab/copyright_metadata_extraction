# Vision backends: free Gemma and the video model (2026-10-11)

## Images: Gemma 4

- **OpenRouter free variant (`google/gemma-4-31b-it:free`) is not usable.** 0 of 9 calls succeeded over five minutes, every one refused with `limit_source: upstream_provider_shared_pool`: the free variant runs on one Google AI Studio quota shared by all OpenRouter users, and it was exhausted. Our account also has the 50-requests-a-day cap for accounts that never bought $10 of credit.
- **Gemma 4 through Google AI Studio directly works and is free of charge.** Google's pricing page lists Gemma 4 as "Free of charge" on the free tier and does not offer it on the paid tier. Google's terms for unpaid services: Google "uses the content you submit … to provide, improve, and develop" its products, "human reviewers may read, annotate, and process your API input and output", and "Do not submit sensitive, confidential, or personal information". Images of public works are acceptable; contracts are not. The only key available is `GOOGLE_API_KEY` in `jiran.env`, which belongs to another project; a dedicated key for this project should be created.
- **Gemma on Google always writes its reasoning inline** (`<thought>…</thought>` before the answer); switching it off is rejected ("Thinking budget is not supported for this model"), so the client must strip the block before parsing.

40 pilot images (10 per image cell), current system prompt with title context, scored against the current gold:

| Image backend | Keywords | Hashtags | Description pass at τ 0.65 | Mean cosine | Median time | Failures |
|---|---|---|---|---|---|---|
| Gemma 4 31B, Google direct | 0.486 | 0.482 | 0.875 | 0.755 | 36.3 s | 0 / 40 |
| Qwen3.7-Plus, thinking off | 0.417 | 0.449 | 0.875 | 0.740 | 9.1 s | 0 / 40 |

Small sample; the tag edge for Gemma is a tendency, not a proven difference. No rate-limit errors at four parallel calls.

## Videos: Omni models

Our account offers no Qwen3.8-Omni-Plus; the Omni models are Qwen3-Omni-Flash, Qwen3.5-Omni-Flash, Qwen3.5-Omni-Plus and Qwen3.8-Omni-Flash. The pilot's 100 videos were run through the pipeline with each model (same prompts, v4 frames, no arbiter) and scored against the current gold:

| Video model | Video Macro-F1 | Keywords | Hashtags | Description | ccl | donated | expired | kogl |
|---|---|---|---|---|---|---|---|---|
| **qwen3.8-omni-flash** | **80.9 %** | **0.374** | **0.494** | **0.798** | 95.4 % | 83.9 % | 69.9 % | 80.0 % |
| qwen3.5-omni-plus (current) | 78.6 % | 0.308 | 0.437 | 0.758 | 92.6 % | 81.7 % | 67.1 % | 78.0 % |
| qwen3.5-omni-flash | 76.7 % | 0.264 | 0.389 | 0.717 | 92.2 % | 78.3 % | 67.3 % | 75.6 % |

- qwen3.8-omni-flash − qwen3.5-omni-plus: **+2.33 points, paired bootstrap 95 % CI [+0.73, +3.99]** over 99 videos; better in every key and every cell.
- 2 of 100 qwen3.8-omni-flash calls returned invalid JSON the first time and succeeded on retry; the others had none.
- Times are not comparable from these runs (the two Flash runs ran simultaneously, the baseline alone). Prices for the Flash models need confirming.
- Bug fixed: the Omni backend's label was a fixed string, so results recorded "Qwen3.5-Omni-Plus" whatever `OMNI_MODEL` said; the label now follows the model actually used.

## Follow-up (same day): project key, rate limit, image models, costs

### Google AI Studio key of this project
- Stored as `GEMINI_API_KEY` in `.env` (gitignored). It reaches 62 models including `gemma-4-31b-it` and `gemma-4-26b-a4b-it`.
- **Rate limit measured from Google's own quota error: 30 requests per minute** (`GenerateRequestsPerMinutePerProjectPerModel-FreeTier`, model gemma-4-31b). The daily cap is not published by Google; third-party sources say 1,500 or 3,000 requests per day; the project's live numbers are at aistudio.google.com/rate-limit.
- Under a 40-request burst, 16 succeeded, 9 hit the rate limit and **15 returned Google "Internal error" 500s**.

### Image description: 100 pilot images, current system prompt with title context
Accuracy on the 95 images all three models answered; cost at list prices and ₩1,400 per USD.

| Image model | Answered | Keywords | Hashtags | Description pass at τ 0.65 | Mean cosine | Median s | p90 s | ₩ per image | ₩ per 2,000 images |
|---|---|---|---|---|---|---|---|---|---|
| Gemma 4 31B, Google direct | 97 / 100 | **0.463** | 0.441 | 0.874 | **0.755** | 92.1 | 206.9 | 0 | 0 |
| **qwen3.8-omni-flash** | **100 / 100** | 0.424 | **0.463** | **0.916** | 0.745 | 19.0 | 39.3 | 1.34 | 2,684 |
| qwen3.7-plus, thinking off | 98 / 100 | 0.403 | 0.419 | 0.863 | 0.744 | 14.4 | 28.8 | 2.23 | 4,464 |

- Accuracy differences are small and within noise on this sample; no model leads on every key.
- **Gemma's reliability and speed are the problem:** 42 of 100 images needed retries after Google 500 errors and 3 failed after five attempts; at eight parallel calls the median rose to 92 s because of the 30-per-minute limit and retries (36 s unloaded). A 2,000-image run would also meet the daily cap.
- qwen3.7-plus was refused on 2 images by DashScope's content inspection (`data_inspection_failed`); the chain's next backend now takes such cases.
- Google's free-tier terms allow use of inputs for product improvement and human review; acceptable for images of public works, not for contracts.

### Video cost: 20 pilot videos, same frames
| Video model | ₩ per video | ₩ per 2,000 videos | Input tokens | Output tokens |
|---|---|---|---|---|
| **qwen3.8-omni-flash** | **1.77** | **3,542** | 4,860 | 1,141 |
| qwen3.5-omni-plus | 13.70 | 27,403 | 4,824 | 366 |

qwen3.8-omni-flash is about **7.7× cheaper** per video ($0.15 / $0.47 against $1.40 / $8.30 per million tokens) and more accurate on the 100-video test. The earlier ₩2.01 per video for qwen3.5-omni-plus used an unofficial price; at the official price it is ₩13.70.

### Changes applied
- **Video chain:** qwen3.8-omni-flash → qwen3.5-omni-plus → qwen3.7-plus → Gemma (OpenRouter) → Gemma (lab server). `OMNI_MODEL` / `OMNI_BACKUP_MODEL` override the first two.
- **An unreadable (non-JSON) answer now moves to the next backend** instead of failing the set; this covers the occasional invalid JSON from qwen3.8-omni-flash and content-inspection refusals.
- **Lab Gemma server:** 10-second connect timeout and no retry (`GEMMA_CONNECT_TIMEOUT_SEC`); an image on the default chain now reaches its fallback in 18 s instead of 257 s.
- **`make_google_gemma_backend()`** (Gemma 4 via Google, `GEMINI_API_KEY`) is available but not yet in any chain; the client strips Gemma's `<thought>` block before parsing.
- Tests: 291 pass.
