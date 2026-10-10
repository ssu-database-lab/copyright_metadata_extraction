# Why keywords and hashtags fail, and what fixes them

Investigation of 2026-10-10, branch `eval-harness-v4`. Follows the pilot in `docs/v4/pilot200_tau_calibration_20261010.md`. All fixes were developed on works **outside** the frozen test lists and checked once on the pilot 200.

## 1. Answer in brief

The system's tags are mostly correct, but overlap scoring against one reference cannot reward them. A blind judge found 91.9 % of the system's hashtags valid for their work, the same rate as the gold hashtags (91.8 %). Overlap F1 is low because two annotators who both tag correctly still pick different words. The gold drafter, run twice on identical inputs, agrees with itself at only **0.69** hashtag F1. A prompt change gives a real, held-out gain of +0.11 on hashtags and +0.08 on keywords. With the gold fixes below, the projected T-002 Macro-F1 reaches only about 79.5 %. **Under the current scoring rule for tag keys, T-002 cannot reach 85 %.** Passing needs a change to how tag keys are scored, which is a decision for us and 장세영.

## 2. Why they fail: four causes

| Cause | Evidence |
|---|---|
| 1. **The gold had information the system did not.** The gold drafter saw the title and the site's content tags. The system saw the work and the contract, and its prompt said to describe the screen only, ignoring the title. | Gold hashtags from title words were hit 48 % of the time, those from site tags 46 %. On 수어 clips the system never named the signed word, although the title carries it. |
| 2. **Free vocabulary.** Correct tags differ between annotators: synonyms (녹색배경 / 초록배경, 흑백영화 / 흑백영상), aspect choice (구도, 배경, 매체), and specific versus general terms (은행나무 / 나무). | Gold drafter against itself: 0.69. Semantic matching of near-synonyms recovers only +0.03 to +0.04 (§4.3), so most of the gap is aspect choice, not spelling. |
| 3. **Much of the gold is institutional categories, not content.** KOGL site hashtags and keywords are provider tags such as 4k, 문화, 역사, 제주영상, 현충원, 수중문화재, 해양유물. 공유마당 uploaders add habit words such as 사진, 인물, 서울. | Of KOGL site-hashtag uses, 92 % are tags carried by ten or more works; only 68.8 % were judged valid content tags. KOGL keyword F1 on the development set: 0.04. The existing collection-tag rule, five or more works sharing an identical list, runs only for 공유마당. |
| 4. **System habits the guideline forbids.** Genre, mood and plot tags (#범죄영화, #액션, #여행, #감성), mainly on old films. | video/expired: 72 % of system tags valid, against 88 to 99 % in the other cells. |

## 3. What was tried

Development set: 96 works outside both test lists (79 images: 40 KOGL originals, 30 공유마당 kogl, 9 공유마당 ccl/donated/expired; 17 videos, 15 of them 수어 clips, because almost every non-수어 video sits inside the test lists). Gold hashtags were drafted for them with the same drafter and guideline v0.3. A second, independent draft gave the ceiling. Keyword gold follows the test rules. Four of the works serve as worked examples in variant v2 and are excluded from every score. Images ran on Qwen3-VL-235B, because OpenRouter (Gemma 4) is out of credit; videos ran on Qwen3.5-Omni-Plus, as in the system.

| Variant | What changes | Hashtags, generated gold | Hashtags, all | Keywords |
|---|---|---|---|---|
| v0 | current system prompt | 0.417 | 0.307 | 0.330 |
| v1 | tag rules copied from guideline v0.3 (rules 6-1 and 7); title subject words allowed in tags | 0.531 | 0.377 | 0.340 |
| **v2** | v1 + four worked examples from development works | **0.556** | **0.380** | **0.361** |
| v2, images on Omni | model change only | 0.564 | 0.385 | 0.361 |
| v3 | v2 + a second pass in which the system's text model (qwen3.5-122b) rewrites the tags | 0.596 | 0.415 | 0.377 |
| v4 | v3 + extra keyword rules (medium word, 인물, province of a named place) | 0.584 | 0.406 | 0.388 |
| ceiling | gold drafter against itself, same inputs | 0.69 | — | — |

Cutting tag lists to 5 to 8 items changed nothing material. The image model choice made no difference.

## 4. Held-out check on the pilot 200

Run once, on the same 200 test works as the pilot, with the same backends for both prompts.

| | Keywords | Hashtags, all | Hashtags, generated gold | Hashtags, KOGL site gold |
|---|---|---|---|---|
| v0, current prompt | 0.267 | 0.281 | 0.317 | 0.100 |
| **v2, new prompt** | **0.350** | **0.394** | **0.446** | 0.135 |
| v3, v2 + rewrite | 0.343 | 0.397 | 0.447 | 0.142 |

The rewrite step's development gain did not hold on the held-out set, so **v2 alone is the recommendation**. Per cell, on the 196 works both runs completed:

| Cell | Hashtags v0 → v2 | Keywords v0 → v2 |
|---|---|---|
| image/ccl | 0.369 → 0.506 | 0.241 → 0.462 |
| image/donated | 0.354 → 0.413 | 0.338 → 0.423 |
| image/expired | 0.288 → 0.339 | 0.0 → 0.083 (2 sets) |
| image/kogl | 0.288 → 0.340 | 0.340 → 0.395 |
| video/ccl (수어) | **0.278 → 0.706** | — |
| video/donated | 0.405 → 0.485 | 0.304 → 0.385 |
| video/expired | 0.118 → 0.165 | 1 set |
| video/kogl | 0.107 → 0.174 | 0.141 → 0.204 |

### 4.1 Projected T-002 Macro-F1 on the pilot

| Setting | Macro-F1 |
|---|---|
| Current prompt, current gold | 75.5 % |
| New prompt (v2), current gold | 78.4 % |
| New prompt + collection rule on KOGL gold + resolution gold fix | **79.5 %** |

The other keys are held at their pilot values: description 0.759 and file keys 1.0. Reaching 85 % would need the three content keys to sum to 1.95, for example description 0.85, hashtags 0.60 and keywords 0.50.

### 4.2 Validity check (blind)

For every pilot work, gpt-5.4-mini saw the image or six video frames with the title, plus a shuffled, unlabelled list mixing the system's v2 hashtags and the gold hashtags. It marked each tag as valid or not for the work.

| Tags | Judged valid |
|---|---|
| System, v2 | **1,661 / 1,808 = 91.9 %** |
| Gold, all | 1,325 / 1,444 = 91.8 % |
| Gold, generated | 95.2 % |
| Gold, KOGL site categories | 68.8 % |

System validity by cell ranges from 88 % to 99 %, except video/expired at 72 %. There the invalid tags are plot words such as 폭발, 형사 and 추격전. The judge belongs to the gold drafter's model family, so it may slightly favour the generated gold; the system's tags still come out level with the gold overall.

### 4.3 Semantic matching (pilot, current prompt)

Counting near-synonyms as matches with KURE-v1 tag embeddings lifts hashtag F1 from 0.304 to 0.337 at a similarity of 0.85 and 0.341 at 0.80. Accepted pairs at 0.85 are sound (흑백영상 / 흑백영화, 초록배경 / 녹색배경, 검은셔츠 / 검정셔츠). At 0.75 wrong pairs appear (조명 / 빛, 램프 / 조명). Keywords move by +0.01. This alone does not close the gap.

## 5. Recommendations

| # | Proposal | Effect | Decision by |
|---|---|---|---|
| 1 | **Adopt the v2 prompt** for work legs: guideline-aligned tag rules, title subject words allowed in keywords and hashtags, four worked examples from non-test works. The exact text is variant `v2` in `api/module/evaluation/v4/tag_lab.py`. | +0.11 hashtags, +0.08 keywords on the held-out pilot; +2.9 Macro points | user |
| 2 | Add a rule against plot and genre tags for videos ("no story, characters' roles or events not visible in the frames") | targets video/expired, where 28 % of tags are invalid | user |
| 3 | **Extend the collection-tag rule to KOGL**: identical keyword lists shared by five or more works become N/A; KOGL site hashtags of that kind are replaced by generated gold, as 장세영 asked for KOGL images | 636 of 750 KOGL hashtag sets and 65 keyword sets change; +1.1 Macro points with the resolution fix | user |
| 4 | **Change how tag keys are scored**, before the test. Options: (a) **validity precision**, the share of predicted tags judged valid for the work by the human reviewer or a blind judge, scoring about 92 % today; (b) overlap F1 with semantic matching at 0.85, scoring about 0.34 to 0.44, not enough; (c) keep overlap F1, in which case T-002 cannot pass. | (a) is the only option that measures whether the tags are correct and lets T-002 pass | user, 장세영, the test lab |
| 5 | Keep video legs at six concurrent calls or fewer. Two parallel runs of six hit DashScope's per-minute quota (HTTP 429, "Allocated quota exceeded"); retried at two workers they passed. | avoids failed sets in the 4,000-set run | — |

If option 4(a) is chosen, the stage-1 review workbook of the pilot can collect the human validity judgments for system tags at the same time as the gold verification, and the blind-judge script below can serve as a second rater.

## 6. Files and commands

| What | Where |
|---|---|
| Experiment harness (variants v0 to v2, rewrite, scoring) | `api/module/evaluation/v4/tag_lab.py` |
| Development set and gold drafts | `dataset/tag_dev_v4/dev_ids.csv`; `dataset/gold_descriptions/tag_dev_a/` (gold), `tag_dev_b/` (ceiling) |
| Development runs and scores | `dataset/tag_dev_v4/runs/`, `score_dev.csv` |
| Pilot runs and scores | `dataset/pilot200_v4/tag_runs/`, `tag_score_pilot.csv` |
| Validity judgments | `dataset/pilot200_v4/analysis/tag_validity.jsonl` |

```bash
python -m api.module.evaluation.v4.tag_lab run --variant v2 --ids dataset/tag_dev_v4/dev_ids.csv --out dataset/tag_dev_v4/runs --workers 6
python -m api.module.evaluation.v4.tag_lab score --runs dataset/tag_dev_v4/runs --ids dataset/tag_dev_v4/dev_ids.csv --gold dev
python -m api.module.evaluation.v4.tag_lab score --runs dataset/pilot200_v4/tag_runs --ids dataset/pilot200_v4/pilot_ids.csv --gold dataset/test_set_v4/gold23_t002.jsonl --exclude ""
```

Spend: two development gold drafts, USD 2.09; validity judge, about 190,000 input and 63,000 output tokens on gpt-5.4-mini; DashScope runs, a few hundred won by the estimator's unit prices.

## 7. Step-by-step results through the real pipeline (2026-10-10 evening)

Each proposal was put into the system one at a time and the T-002 pipeline was run on the pilot 200 (no arbiter on work legs, v4 frames, images on Qwen3-VL-235B because OpenRouter has no credit, videos on Qwen3.5-Omni-Plus), then scored by the evaluator. Failed sets were retried with the same code so every step has 0 failures. Scores under the current gold and under the gold with the KOGL collection rule (step 3):

| Step | What changed | Macro, current gold | Macro, KOGL rule | Keywords* | Hashtags* | Description |
|---|---|---|---|---|---|---|
| 0 | system as it was | 76.0 % | 76.9 % | 0.320 | 0.350 | 0.739 |
| 1 | new tag prompt: guideline rules, title subject words, four worked examples (images and videos) | 79.1 % | 80.2 % | 0.385 | 0.451 | 0.804 |
| 2 | + no plot, character-role or genre words in video descriptions and tags | 79.3 % | 80.4 % | 0.373 | 0.457 | 0.819 |
| 3 | + KOGL collection rule in the gold (the column "KOGL rule") | — | see above | | | |
| 4 | + title for works without a contract (17 KOGL videos), as every work will have one in the test | **79.8 %** | **80.7 %** | 0.380 | 0.462 | 0.829 |

\* under the KOGL-rule gold. File format, size and hash stay at 1.0 and resolution at 0.975 throughout; fixing the resolution gold (§2, finding 5 of the pilot document) adds about +0.4 points, for about **81.1 %**.

Per cell, KOGL-rule gold, Macro / hashtags / description:

| Cell | Step 0 | Step 1 | Step 2 | Step 4 |
|---|---|---|---|---|
| image/ccl | 77.3 % / 0.41 / 0.72 | 83.7 % / 0.53 / 0.92 | same | same |
| image/donated | 77.8 % / 0.44 / 0.84 | 79.5 % / 0.46 / 0.96 | same | same |
| image/expired | 74.8 % / 0.38 / 0.76 | 76.4 % / 0.42 / 0.84 | same | same |
| image/kogl | 78.1 % / 0.41 / 0.68 | 82.8 % / 0.50 / 0.80 | same | same |
| video/ccl | 86.9 % / 0.29 / 0.92 | 93.2 % / 0.67 / 0.92 | 92.6 % / 0.67 / 0.88 | same |
| video/donated | 81.1 % / 0.43 / 0.88 | 81.3 % / 0.48 / 0.80 | 81.7 % / 0.45 / 0.88 | same |
| video/expired | 65.1 % / 0.13 / 0.17 | 64.6 % / 0.17 / 0.25 | 67.1 % / 0.23 / 0.42 | same |
| video/kogl | 77.5 % / 0.33 / 0.92 | 79.4 % / 0.40 / 0.92 | 78.0 % / 0.40 / 0.84 | 80.6 % / 0.43 / 0.92 |

Reading the steps:
- **Step 1 is the big one (+3.1 points).** It also lifts the description (+0.065), because the new context line lets the title's subject words into the description, which the gold guideline asks for too.
- **Step 2 does what it targets** (old films: description 0.25 → 0.42, hashtags 0.17 → 0.23) but costs a little elsewhere (video/donated hashtags, video/ccl and video/kogl descriptions), so the net is +0.2 points.
- **Step 3** adds about +1 point on every step by no longer scoring institutional category tags.
- **Step 4** helps where titles were missing (video/kogl +2.6 points). The 4 image/kogl works without a contract could not be rerun because no image backend was reachable (below); they keep their step-2 predictions.
- **Images in step 2 and step 4 reuse step 1's image predictions.** Step 2 changes only the video prompt, and the image backend failed during the step-2 run (below).

### 7.1 Infrastructure problems found during the runs

- **DashScope is in free-tier-only mode or out of funds.** Around 22:30 KST `qwen3-vl-235b-a22b-instruct`, the system's OCR model and image model, started returning 404 "Model not exist" and disappeared from the model list (99 models listed, against 149 in July). `qwen3-vl-plus` answers 403: "Free quota exhausted. To continue accessing the model on a paid basis, please add funds or disable the 'use free tier only' mode in the management console." The video model and the text models still answer, but only until their free quota runs out. **This needs action in the Alibaba Cloud console before any further run.**
- **OpenRouter credit is exhausted** (402), so Gemma 4 is unavailable, and the self-hosted Gemma URL is unreachable. With DashScope's vision model gone, the system currently has **no working image backend**.
- **Two parallel runs of six video legs hit DashScope's per-minute quota** (429), and a burst of five video timeouts hit one run. Retried at two to four workers, all passed.
- **Invalid JSON from the vision model** failed 1 to 2 sets per run (no retry in the pipeline). A single retry on a parse failure would remove these.

### 7.2 Where this leaves T-002

With all accepted changes, the pilot projects to about **81 %** Macro-F1 under the current scoring rule, against 85 % needed. The remaining gap sits in hashtags (0.46, ceiling 0.69 under overlap scoring) and keywords (0.38). Recommendation 4 (validity scoring for tag keys) remains the change that decides whether T-002 can pass.
