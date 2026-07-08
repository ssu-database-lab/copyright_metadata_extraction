# Gemma 4 31B (vLLM) — Client Implementation Guide

A self-contained guide for calling the locally-hosted `google/gemma-4-31B-it`
vLLM server. Written to be readable by humans and feed-able to an LLM that will
implement the client end-to-end.

> **Last verified: 2026-05-24** against the live server. Current config:
> 128K context, two served model names, **tool/function calling enabled**.

## TL;DR

- **Model:** `google/gemma-4-31B-it` — Google's instruction-tuned multimodal
  Gemma 4 31B. General-purpose: text Q&A, code, summarization, reasoning,
  multilingual (Korean-strong), vision (OCR, VQA, image description), and
  **function/tool calling**. Also served under the short alias **`gemma-4-31B-it`**
  (both strings work in the `model` field).
- **Endpoint (on the server box):** `http://127.0.0.1:8001/v1/chat/completions`
  (OpenAI-compatible Chat Completions API). No auth. Synchronous responses (streaming optional).
  **From another machine, `127.0.0.1` will NOT work** — use this server's IP
  `110.10.125.88` or an SSH tunnel; see "External access" below.
- **Context window:** **131,072 tokens (128K)** total, input + output combined.
- **Verify it's up:** `curl http://127.0.0.1:8001/v1/models`
- **One-line Python (with `openai` SDK):**
  ```python
  from openai import OpenAI
  client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="not-needed")
  resp = client.chat.completions.create(
      model="google/gemma-4-31B-it",   # or "gemma-4-31B-it"
      messages=[{"role": "user", "content": "Hello"}],
  )
  print(resp.choices[0].message.content)
  ```

## What the server is

`vllm serve` running `google/gemma-4-31B-it` from `/shared/models/gemma-4-31B-it`,
tensor-parallel across 2× RTX PRO 6000 Blackwell (96 GB each), bf16 weights,
fp8 KV cache, **128K context**, with **auto tool-choice + the `gemma4` tool-call
parser** enabled. Launched from `/shared/vllm_gemma_env/launch_gemma4.sh`. Logs
stream to `/tmp/gemma4_server.log`.

The vLLM project ships an **OpenAI-compatible HTTP API** — any OpenAI Chat
Completions client (`openai` Python SDK, raw `httpx`/`requests`, `curl`,
LangChain `ChatOpenAI`, etc.) works by pointing `base_url` at this server and
passing any non-empty string for `api_key` (it's ignored — there is no auth).

## Capabilities

| Task | Supported | Notes |
|---|---|---|
| Text-only chat / Q&A / reasoning | ✅ | Just `content: "..."` strings. |
| Code generation / explanation | ✅ | Standard Gemma 4 instruction-following. |
| Multilingual (esp. Korean) | ✅ | Strong on Korean; handles CJK, English, mixed. |
| Multimodal: image + text question | ✅ | Up to 4 images per prompt (`limit-mm-per-prompt`). |
| OCR | ✅ | High `image_seq_length=1120` makes it strong on dense docs. |
| System prompts | ✅ | Standard `{"role":"system","content":"..."}` works. |
| Multi-turn (assistant memory) | ✅ | Pass full message history; server is stateless. |
| Streaming (SSE) | ✅ | Pass `"stream": true`. |
| **Tool / function calling** | ✅ | **Enabled** via `--tool-call-parser gemma4`. Pass OpenAI-style `tools` + `tool_choice:"auto"`; you get standard `message.tool_calls` back. See the dedicated section below. |
| Logprobs | ✅ | `"logprobs": true, "top_logprobs": N`. |
| Audio input | ❌ | Server launched with `audio=0`. |
| Embeddings (`/v1/embeddings`) | ❌ | This is a chat-completions server only; embeddings require a separate model. |

## Server address

```
URL:    http://127.0.0.1:8001
Path:   /v1/chat/completions   (also /v1/models, /v1/completions, /health)
Auth:   none — pass any non-empty string as api_key; it is ignored
Model:  "google/gemma-4-31B-it"  OR  "gemma-4-31B-it"   (both accepted)
```

The server binds `0.0.0.0:8001`, so it's reachable from other machines on the
same LAN — use the workstation's IP or set up an SSH tunnel. **There is no
authentication**, so don't expose it beyond a trusted network.

## External access (from another machine)

`127.0.0.1` only works **on the server box itself**. From any other workstation, target this machine's network IP.

- **This server's IP:** `110.10.125.88` (interface `enp211s0`). vLLM binds `0.0.0.0:8001`, so it listens on all interfaces.
- **⚠️ Security:** `110.10.125.88` is a *public-range* IP (not a private `10.` / `192.168.` / `172.16–31.` address) and **the server has no authentication**. If port 8001 is reachable from the internet, anyone who finds it can use the GPU/model and read your prompts. Treat raw direct access as exposed unless you know the network path is locked down.

### Option A — SSH tunnel (recommended; safe default)
On the **client** workstation, forward a local port to the server over SSH, then point your code at `127.0.0.1` (tunneled — so all the localhost examples in this guide work unchanged):
```bash
ssh -N -L 8001:127.0.0.1:8001 <user>@110.10.125.88
# leave this running; in your app use base_url  http://127.0.0.1:8001/v1
```
No public exposure; auth is handled by SSH.

### Option B — direct (only if both machines are on the same trusted LAN)
Point the client straight at the server IP:
```python
from openai import OpenAI
client = OpenAI(base_url="http://110.10.125.88:8001/v1", api_key="not-needed")
```
```python
BASE_URL = "http://110.10.125.88:8001"   # raw httpx variant
```

### Verify reachability — run this ON the client workstation
```bash
curl -s --max-time 5 http://110.10.125.88:8001/v1/models
```
- Returns the two model IDs → reachable, you're set (Option B works).
- Hangs / "connection refused" → a firewall (OS `ufw`/`iptables` on the server, or an upstream network firewall) is blocking port 8001. Use the SSH tunnel (Option A), or ask the server admin to allow the client's IP to 8001.

> If the server's IP ever changes, re-check on the server box with `hostname -I` (take the non-loopback, non-Docker/`172.x` address).

## Request format (Chat Completions API)

`POST http://127.0.0.1:8001/v1/chat/completions`

### Minimum body

```json
{
  "model": "google/gemma-4-31B-it",
  "messages": [
    {"role": "user", "content": "Explain transformers in two sentences."}
  ]
}
```

### Full body schema (every field you'll typically use)

```json
{
  "model": "google/gemma-4-31B-it",
  "messages": [
    {"role": "system", "content": "Optional system prompt."},
    {"role": "user",   "content": "Or a list (see below) for vision."},
    {"role": "assistant", "content": "Prior assistant turn (multi-turn)."}
  ],
  "max_tokens": 1024,
  "temperature": 0.7,
  "top_p": 0.95,
  "stop": ["</done>"],
  "stream": false,
  "seed": 42,
  "logprobs": false,
  "top_logprobs": 5,
  "tools": [],
  "tool_choice": "auto"
}
```

### Vision: image + text in one user message

Images are passed as **OpenAI-style content parts** in the user message. Each
image is a `{"type": "image_url", ...}` part with a `data:` URL (base64 PNG/JPEG)
or an `http(s)://` URL the server can fetch.

**Recommended: image-first, text-second** (Google's guidance for Gemma 4):

```json
{
  "model": "google/gemma-4-31B-it",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAA..."}},
        {"type": "text", "text": "What's in this image?"}
      ]
    }
  ],
  "max_tokens": 1024
}
```

Up to 4 images per prompt. Mix `image_url` parts and `text` parts freely.

## Tool / function calling (ENABLED)

The server runs with `--enable-auto-tool-choice --tool-call-parser gemma4`, so
**standard OpenAI tool-calling works**. Gemma 4 emits a custom internal format
(`<|tool_call>call:func{...}<tool_call|>`), but vLLM's `gemma4` parser converts
it to the standard OpenAI `tool_calls` shape — clients use the normal API and
never see the raw format.

**Request:** add `tools` (OpenAI function-schema list) and `tool_choice` (`"auto"`,
`"none"`, `"required"`, or `{"type":"function","function":{"name":"..."}}`):

```json
{
  "model": "gemma-4-31B-it",
  "messages": [{"role": "user", "content": "What's the weather in Seoul?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the weather for a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }],
  "tool_choice": "auto",
  "max_tokens": 200
}
```

**Response** when the model calls a tool — `content` is `null`,
`finish_reason` is `"tool_calls"`, and `message.tool_calls[*].function.arguments`
is a **JSON string** (parse it):

```json
{"choices": [{"message": {"role": "assistant", "content": null,
  "tool_calls": [{"id": "chatcmpl-tool-...", "type": "function",
    "function": {"name": "get_weather", "arguments": "{\"city\": \"Seoul\"}"}}]},
  "finish_reason": "tool_calls"}]}
```

**Multi-turn tool loop** (standard OpenAI pattern): append the assistant's
`tool_calls` message, then a `{"role":"tool","tool_call_id":..,"content":..}`
message with the tool result, and call again. Verified working end-to-end
(it drives a full RAG agent loop). The `openai` SDK's tool-calling flow works
unchanged against this endpoint.

```python
resp = client.chat.completions.create(
    model="gemma-4-31B-it", messages=msgs, tools=tools, tool_choice="auto",
    max_tokens=300,
)
msg = resp.choices[0].message
if msg.tool_calls:
    import json
    for tc in msg.tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments)   # arguments is a JSON string
        # ... run the tool, then append {"role":"tool","tool_call_id":tc.id,"content":result}
```

## Response format (plain completion)

Standard OpenAI Chat Completions shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1779094718,
  "model": "google/gemma-4-31B-it",
  "choices": [
    {"index": 0,
     "message": {"role": "assistant", "content": "...the answer..."},
     "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 34, "completion_tokens": 19, "total_tokens": 53}
}
```

**What to parse:**
- `choices[0].message.content` — the assistant's reply text (`null` if it called a tool).
- `choices[0].message.tool_calls` — present when the model invokes a tool.
- `choices[0].finish_reason` — `"stop"` (normal), `"length"` (hit `max_tokens`),
  `"tool_calls"` (called a tool), `"content_filter"` (rare).
- `usage.prompt_tokens` / `usage.completion_tokens` — for cost/length tracking.

**Failure modes:**
- HTTP 400 — bad request (e.g. `max_tokens` exceeds remaining context budget, or
  malformed `messages`/`tools`). Body has `{"error":{"message":...}}`.
- HTTP 404 — wrong model id (use `google/gemma-4-31B-it` or `gemma-4-31B-it`).
- HTTP 500 — server-side (model crash, OOM). Check `/tmp/gemma4_server.log`.
- Connection refused — server is down. Restart with `/shared/vllm_gemma_env/launch_gemma4.sh`.

## Parameters — what to set, what to leave alone

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `model` | str | required | `"google/gemma-4-31B-it"` or `"gemma-4-31B-it"`. |
| `messages` | list | required | See above. |
| `max_tokens` | int | unset (≈ model max) | **Always set this.** Bounds output. `prompt_tokens + max_tokens ≤ 131072`. |
| `temperature` | float | 1.0 | `0.0` for deterministic / OCR / classification; `0.7` creative; `1.0` varied. |
| `top_p` | float | 1.0 | Nucleus sampling. Leave at 1.0 unless you have a reason. |
| `tools` / `tool_choice` | list / str | none | Function calling — see the tool-calling section. |
| `seed` | int | none | For reproducibility at non-zero temperature. |
| `stop` | str / list[str] | none | Stop sequences. |
| `stream` | bool | false | See "Streaming". |
| `logprobs` | bool | false | Returns token logprobs. |
| `presence_penalty`, `frequency_penalty` | float | 0.0 | Usually leave alone. |

**Context-budget rule:** prompt + `max_tokens` ≤ **131,072**. Exceeding it returns
HTTP 400 (`"maximum context length is 131072 tokens..."`). For images, each page
of a typical document at `image_seq_length=1120` contributes ~1,100–1,300 tokens.

## Server-baked settings the caller can't override

Set at server launch; apply to every request:

| Setting | Value | Why it matters |
|---|---|---|
| `max-model-len` | **131,072 (128K)** | Hard context ceiling (input+output). |
| `tool-call-parser` | **gemma4** | Enables OpenAI-style tool calling (with `--enable-auto-tool-choice`). |
| `image_seq_length` | 1120 | Vision-token budget per image — tuned for OCR quality. Higher than vLLM's default (280); every image call pays this. |
| `limit-mm-per-prompt` | image=4, audio=0 | ≤4 images per request; no audio. |
| `dtype` | bfloat16 | Native Gemma 4 precision. |
| `kv-cache-dtype` | fp8 | ~50% KV memory saving; quality-neutral in practice. |
| `tensor-parallel-size` | 2 | Sharded across 2 GPUs. |
| `max-num-seqs` | 16 | Up to 16 concurrent requests; beyond that, requests queue. (At 128K the KV pool allows ~12× full-length concurrency, so `max-num-seqs` is the binding limit.) |

Changing `image_seq_length` / context / tool parser requires relaunching the
server — talk to whoever owns it first.

## Code samples

### Python — `openai` SDK (easiest)

```python
# pip install openai>=1.0
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8001/v1",
    api_key="not-needed",  # vLLM ignores it, but the SDK requires non-empty
)

# Text-only
resp = client.chat.completions.create(
    model="google/gemma-4-31B-it",   # or "gemma-4-31B-it"
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Summarize the theory of relativity in 2 sentences."},
    ],
    max_tokens=200,
    temperature=0.3,
)
print(resp.choices[0].message.content)

# Image + text (vision)
import base64
with open("page.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("ascii")

resp = client.chat.completions.create(
    model="google/gemma-4-31B-it",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": "Describe this image."},
        ],
    }],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

### Python — raw `httpx`

```python
# pip install httpx
import base64, httpx

BASE_URL = "http://127.0.0.1:8001"
MODEL = "gemma-4-31B-it"   # or "google/gemma-4-31B-it"

def ask_text(question: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    body = {"model": MODEL, "messages": [{"role": "user", "content": question}],
            "max_tokens": max_tokens, "temperature": temperature}
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=body, timeout=120.0)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def ask_about_image(image_path: str, question: str, max_tokens: int = 1024) -> str:
    img_b64 = base64.b64encode(open(image_path, "rb").read()).decode("ascii")
    body = {"model": MODEL, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": question}]}],
            "max_tokens": max_tokens, "temperature": 0.0}
    r = httpx.post(f"{BASE_URL}/v1/chat/completions", json=body, timeout=300.0)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# Async: use httpx.AsyncClient + await client.post(...). Same body.
```

### curl (quick testing)

```bash
# Text-only
curl -s http://127.0.0.1:8001/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-31B-it","messages":[{"role":"user","content":"Hello"}],"max_tokens":50,"temperature":0}' \
  | python3 -m json.tool

# Tool calling
curl -s http://127.0.0.1:8001/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-31B-it","messages":[{"role":"user","content":"Weather in Seoul?"}],
       "tools":[{"type":"function","function":{"name":"get_weather","description":"weather for a city",
       "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
       "tool_choice":"auto","max_tokens":120}' | python3 -m json.tool
```

## Streaming (SSE)

Add `"stream": true`. The response becomes an SSE stream of `data: {...}\n\n`
lines (OpenAI delta shape), terminated by `data: [DONE]\n\n`.

```python
# openai SDK
stream = client.chat.completions.create(
    model="gemma-4-31B-it",
    messages=[{"role": "user", "content": "Tell me a short story."}],
    max_tokens=400, stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

```python
# httpx (manual SSE parse)
import json, httpx
with httpx.stream("POST", "http://127.0.0.1:8001/v1/chat/completions",
    json={"model":"gemma-4-31B-it","messages":[{"role":"user","content":"Tell me a short story."}],
          "max_tokens":400,"stream":True}, timeout=300.0) as r:
    for line in r.iter_lines():
        if not line or not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ").strip()
        if payload == "[DONE]":
            break
        delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
        if delta:
            print(delta, end="", flush=True)
print()
```

## Performance reference

Measured on this server (Gemma 4 31B, TP=2, bf16, fp8 KV, 128K, `image_seq_length=1120`):

| Task | Typical latency |
|---|---|
| Short text Q&A (~50 output tokens) | < 1 s |
| Medium text response (~500 output tokens) | 3–6 s |
| Tool call (decide + emit) | 1–3 s |
| OCR a dense Korean PDF page (~1,300 image-tokens in + ~1,000 out) | 22–38 s/page |
| Throughput | up to 16 concurrent requests (`max-num-seqs=16`) |

Cold first request after server start is slower (CUDA graph capture). Subsequent
requests are steady. Generation throughput ~28–31 tok/s/request.

## Quirks specific to Gemma 4

1. **Image-first ordering.** When sending image + text, put the `image_url`
   part *before* the `text` part. Google's docs and our benchmarking agree it
   beats text-first.
2. **`中` glyph artifact (OCR only).** Gemma 4 sometimes renders the CJK char
   `中` as `℄` (U+2104) or `℘` (U+2118) in OCR output. One-line fix:
   ```python
   import re
   _FIX = re.compile(r"[℄℘]")
   def normalize_gemma_ocr(text: str) -> str:
       return _FIX.sub("中", text)
   ```
   Only relevant for Korean OCR containing `中`; ignore otherwise.
3. **`temperature=0` + agent loops can repeat.** At greedy decoding, if a tool
   result doesn't change the model's context, it may re-issue the *identical*
   tool call. Always set `max_tokens`, and add a loop guard / dedup (or a small
   temperature like 0.3) in multi-round agent loops. (At 128K this no longer
   causes context-overflow crashes, but it can still waste rounds.)
4. **Tool-call arguments are a JSON *string*.** `message.tool_calls[*].function.arguments`
   is a serialized JSON string — `json.loads()` it before use.

## Health check / diagnostics

```bash
# Up? (lists both model ids)
curl -s http://127.0.0.1:8001/v1/models | python3 -m json.tool

tail -f /tmp/gemma4_server.log          # logs
pgrep -af "vllm serve" | head           # process alive?
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv   # GPU
```

If `/v1/models` returns nothing or times out, the server is down. Restart:

```bash
nohup /shared/vllm_gemma_env/launch_gemma4.sh >> /tmp/gemma4_server.log 2>&1 &
# Wait ~75–90 s for model load before requests succeed.
```

## Common pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Forgetting `max_tokens` | Output runs long / cut off | Always set `max_tokens`. |
| `prompt + max_tokens > 131072` | HTTP 400 "maximum context length" | Keep `max_tokens` ≤ `131072 − prompt_tokens`. |
| Wrong model id | HTTP 404 "model not found" | Use `google/gemma-4-31B-it` or `gemma-4-31B-it`. |
| Treating `tool_calls.arguments` as a dict | `TypeError` | It's a JSON **string** — `json.loads()` it. |
| Empty `api_key` with `openai` SDK | SDK raises before sending | Pass any non-empty string (`"not-needed"`). |
| Image as raw bytes in JSON | serialization fails | Base64 + `"data:image/png;base64,..."` URL. |
| Sending audio | rejected | Server launched with `audio=0`. |
| Short client timeout on long OCR | hangs / times out | `timeout=300`+ on `httpx`. |
| Server on a different host | connection refused | Binds `0.0.0.0:8001` — use the host's real IP, not `127.0.0.1`. |

## Minimal "implement this client" spec for an LLM

> Implement a client for an OpenAI-compatible Chat Completions endpoint at
> `http://127.0.0.1:8001/v1/chat/completions`. Model id: `"google/gemma-4-31B-it"`
> (or the alias `"gemma-4-31B-it"`). Standard body
> `{model, messages, max_tokens, temperature, top_p, tools, tool_choice, stream, ...}`;
> standard OpenAI response (`choices[0].message.content`, or `.tool_calls` when a
> tool is invoked with `finish_reason:"tool_calls"`). Auth is a no-op — pass any
> non-empty `api_key`. **Context window is 131,072 tokens**; always set
> `max_tokens` so `prompt_tokens + max_tokens ≤ 131072`. **Tool calling is
> enabled** (gemma4 parser): pass OpenAI `tools` + `tool_choice:"auto"`, and
> `json.loads()` the `tool_calls[*].function.arguments` JSON string. For vision,
> include image parts ordered image-first as
> `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` (≤4
> images). `temperature=0` for deterministic output; `stream:true` for SSE. No
> audio. Up to 16 concurrent requests.
```
