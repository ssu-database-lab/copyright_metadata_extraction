# VLM 모델 조사 브리프 (2026-07-07)
_멀티모달(이미지→한국어 설명/유형/키워드) 최적 모델 웹조사 — 15-agent workflow, 후보 존재/라이선스/vLLM/한국어 근거 전수 검증._

---

# VLM Recommendation Brief — Korean Visual Metadata Extraction (KOGL), July 2026

## 1) Verdict: KEEP Gemma 4 31B as self-hosted primary

Every load-bearing claim verified against primary sources. It is still the newest open generation (2026-04-02), genuinely Apache 2.0 (official model card: ai.google.dev/gemma/docs/core/model_card_4 — a deliberate break from the old custom Gemma Terms), day-0 vLLM with documented JSON constrained decoding + tool calling (docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html). Benchmarks confirmed on the official card: **MMMU-Pro 76.9 (leads all open weights)**, MATH-Vision 85.6, **OmniDocBench 1.5 error 0.131** (top-tier scanned-doc parsing), 256K ctx. Korean is explicitly listed among 35+ out-of-the-box languages (140+ pretrained), Google published a dedicated Korean launch post, and — decisively — it already won our own 50-image A/B vs Qwen3-VL-235B on Korean fluency, in-image Korean text, zero script leaks, 100% JSON parse. Nothing released since decisively beats it for Korean-output visual metadata. Challenge it; don't replace it preemptively.

## 2) A/B candidates (run on compare.py + 100-image gold set)

**A. Qwen3.6-27B — the primary challenger (full 100-image A/B)**
- Why it might win on OUR task: **OCRBench 89.4** and the Qwen family's confirmed edge on document-medium classification (어문 vs 미술 on scanned docs) target our one known Gemma weakness. All seven claimed vision numbers confirmed verbatim on the official HF card (MMMU 82.9, MMMU-Pro 75.8, MMBench 92.3-EN, MMStar 81.4; huggingface.co/Qwen/Qwen3.6-27B). Note MMMU-Pro 75.8 is *below* Gemma's 76.9 — the case rests on documents/OCR, not general vision.
- Size/license/serving: 27B dense natively multimodal, **Apache 2.0** (confirmed HF + GitHub), official vLLM ≥0.19 recipe; official FP8 runs with vision in ~22GB.
- Effort: low (~half day) — pull FP8, vLLM serve, port prompts. **Must set `enable_thinking: false`** (on by default) and use guided_json.
- Known risk (confirmed, not hypothetical): hanja/Chinese-script leakage in Korean output is documented for this exact model by Korean communities (dcinside guide describes banning ~55k hanja tokens + GBNF grammar to suppress it). Score leak rate explicitly.

**B. Gemma 4 26B-A4B (MoE) — the throughput/video engine (full A/B)**
- Why: only ~3 pts behind 31B on MMMU-Pro (73.8 vs 76.9, official card) at **3.8B active params** → large batch-throughput gain for the 144k backlog and video keyframes. Same Korean training lineage, same chat template, same vLLM server — **zero new serving work**, the cheapest possible A/B.
- License: Apache 2.0. ~50GB BF16 (all MoE params resident) or QAT q4_0 ~14-16GB.
- Effort: trivial (~1 hour).
- Caveat: **OmniDocBench is worse than the 31B (0.149 vs 0.131)** — scanned-doc/어문 classification may regress, and Korean quality at 3.8B active compute is unmeasured. Likely role: video/keyframe engine, 31B stays on stills.

**C. Qwen3-VL-32B-Instruct — the measured Korean SOTA in our GPU class (full A/B)**
- Why: in **Naver's own Jan-2026 tech report** (arXiv 2601.03286) it tops the Korean benchmarks — K-MMBench 91.1, **K-DTCBench 95.4** — beating even Naver's Korean-specialized HyperCLOVAX-SEED-Think-32B. It's the only candidate with *published* Korean multimodal numbers in our class. Same family as our DashScope backup → prompts/JSON schemas port with zero work.
- 32B dense, Apache 2.0 (Oct 2025), standard vLLM. Effort: low.
- Caveat: one generation older than Qwen3.6/Gemma 4; same lineage script-leak risk as our current backup.

**D. SKT A.X-4.0-VL-Light (8B) — optional Korean-document specialist, second opinion**
- Why: purpose-built for Korean documents — KoBizDoc 89.8, K-DTCBench 90.0, OutdoorKorean (Korean scene text) 97.3, K-Handwriting 84.3; Korean-image avg 79.4 vs Qwen2.5-VL-32B's 73.4 at 1/4 the size. **Apache 2.0, the cleanest license of any Korean VLM** (SKT explicitly: 연구·상업 이용). `vllm serve "skt/A.X-4.0-VL-Light"` one-liner; 8B fits *beside* Gemma on the same server.
- Role: not a primary candidate (Qwen2.5-era base; K-LLaVA-W 83.2 — free-form descriptions will trail Gemma 31B). Test as a work_type router/second-opinion on document-medium images only. Effort: trivial.

*Screened out / bench only:* **GLM-4.6V-Flash** (9B, MIT weights, native structured output) — zh/en only, no Korean claim; do a 20-image Korean screen at most before spending an A/B slot. **VARCO-VISION-2.0-14B** — best Korean-specialized VLM (K-LLaVA-W 96.5) but **CC BY-NC 4.0 → disqualified** for government use; usable only as offline eval reference. **HyperCLOVAX-SEED-Think-32B** — conditional-commercial license our use satisfies, strong KoNET 75.1, but it's a reasoning model with unproven JSON-with-thinking-suppressed behavior and is beaten on K-benchmarks by the Apache-licensed Qwen3-VL-32B anyway; revisit only if C disappoints. **EXAONE 4.5 VLM** — non-commercial license, out. **InternVL3.5, MiniCPM-V, Molmo 2** — superseded / below quality bar / English-centric.

## 3) Cloud fallback: stay on DashScope, swap model string to qwen3.7-plus

Keep the existing account (DashScope free dev tier ended 2026-04-15; new accounts get only a one-time 1M-token trial — switching providers costs money and a new integration for zero gain). Upgrade the backup from the two-generations-old qwen3-vl-235b-a22b-instruct to **qwen3.7-plus** (unified image+video multimodal, same multimodal-generation endpoint → literally a model-string swap). Pricing anchor: flagship qwen3.7-max lists $2.50/$7.50 per 1M tok with a 50% promo and 90% cached-input discount; Plus is the cheaper mid-tier (alibabacloud.com/help/en/model-studio/model-pricing). Re-verify JSON parse rate on 20 images after the swap. Risk profile (occasional script leaks) unchanged vs today's backup.

## 4) Video phase

Keyframes-through-the-same-model is the right architecture: both Gemma 4 variants take video/image input natively, and **26B-A4B is the natural keyframe engine** (many frames/video × 3.8B active params). Qwen3.6-27B posts VideoMME 87.7 if it wins the stills A/B. No video-native model justifies a second stack for *metadata generation*; Molmo 2 (Ai2, Dec 2025) is worth revisiting only as an auxiliary keyframe-selection/tracking tool, not for Korean output. qwen3.7-plus fallback also accepts video directly.

## 5) One-week A/B plan (existing harness, 100-image gold set)

- **Day 1:** Serve Gemma 4 26B-A4B on the existing vLLM stack; full 100-image run vs 31B baseline. Metrics: work_type accuracy (esp. 어문/scanned docs), description quality (human-rated), keyword precision, JSON parse %, latency/img.
- **Day 2:** Pull Qwen3.6-27B-FP8; configure `enable_thinking:false` + guided_json; 20-image smoke test with explicit **hanja-leak counter** added to compare.py.
- **Day 3:** If no leaks at >2% of images → full 100-image Qwen3.6 run. In parallel: swap backup string to qwen3.7-plus, 20-image JSON-parse verification.
- **Day 4:** Qwen3-VL-32B-Instruct full 100-image run (same guided decoding config). Optional: A.X-4.0-VL-Light on the document-image subset only.
- **Day 5:** Optional KOFFVQA run (github.com/maum-ai/KOFFVQA — open eval code; leaderboard predates this generation so we must score candidates ourselves) on the top 2 finishers as an external Korean check.
- **Days 6-7:** Human review of disagreement cases; decision matrix (work_type acc / Korean fluency / leak rate / JSON % / latency). **Decision rules:** challenger displaces Gemma 31B only if it beats it on work_type+document accuracy *with zero script leaks and ≥ Korean fluency*; 26B-A4B becomes video engine if within ~2 pts of 31B on gold-set quality.

## 6) Honest caveats

- **No published Korean multimodal score exists for any 2026-generation candidate** (Gemma 4, Qwen3.6) — KOFFVQA's 81-model leaderboard and KMMMU predate them. Our gold set is the real arbiter.
- Qwen3.6-27B's Chinese-script leakage is **confirmed by community reports**, with workarounds (token bans/guided decoding) that add complexity; budget for it or expect disqualification.
- Gemma 26B-A4B's "several-fold latency cut" is architecture-derived, not an independent measurement; and its OmniDocBench regression (0.149) hits our weakest category.
- Minor overstatements found in the research: Gemma "MMLU 87.1" is third-party (card says MMLU-Pro 85.2); "40+ languages" is actually 35+ out-of-the-box.
- GLM-4.6V license discrepancy (weights MIT vs repo Apache-2.0) — both permissive, but record the shipped LICENSE file at download time for the compliance record; same practice for Gemma 4 (card still links a prohibited-use page).
- The verification pass on GLM-4.6V-Flash and the qwen3.7-plus pricing/Korean-model verdicts was partially truncated in our research record — re-confirm qwen3.7-plus Plus-tier pricing on the DashScope console before committing budget numbers.
- Stale-SEO hazard: aggregator pages still calling Qwen2.5-VL-72B the open leader "as of May 2026" are outdated; trust HF model cards and official vendor posts.

**Bottom line:** Gemma 4 31B stays primary today. The realistic outcomes of the week are (a) Qwen3.6-27B or Qwen3-VL-32B takes over *document-medium* routing or the primary slot if leak-free, and (b) Gemma 4 26B-A4B becomes the video/batch engine. Fallback becomes qwen3.7-plus on the existing DashScope account.