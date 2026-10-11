# Remaining Work — Attribute Extraction Certification

**Author** Soongsil University DB Lab · **Date** 2026-08-17 · **Status** Working document
**Scope** What is left before the 공인시험인증 of our two indicators
**Target** 85% accuracy — 공유저작물 속성정보 정확도 (weight 15) + 멀티모달 저작물 속성정보 정확도 (weight 15)

---

## 1. Where we stand

| | |
|---|---|
| **Deployed** | `/v3` batch console + batch API live at `150.230.114.9:5000` (2026-08-17) |
| **Schema** | 67 → **72 fields**; all 11 evaluated attributes now have a home |
| **Ground truth** | **4,834 records** built for the 공유마당 corpus |
| **Harness** | Pair-mode batch runner + CLI + web console, sharing one core |
| **Measured** | Resolution extraction **96.8%** against independent site-published values |

What is **not** yet true: no contracts exist for the evaluation sets, the video track is still
blocked, and no pilot has been run — so **we still do not know our actual accuracy**.

---

## 2. Blocking items, in dependency order

### 2-1. Agree the test input format — *highest priority, not ours to decide alone*

Three of the eleven attributes (제목, 저자, 라이선스 유형) are not derivable from a work file
alone. How the test body supplies input decides everything downstream:

| Input | Effect |
|---|---|
| Work file only | Those 3 are structurally empty → ceiling **72.7%**, below target regardless of quality |
| File + source metadata | Those 3 are passed through unchanged → free 100%, **measures nothing** |
| **File + contract document** | All 3 are genuinely extracted from a document → a real test |

The third option is the only one that produces a defensible number, which is why contract
generation (2-3) matters. This has to be settled with 콘진원 and the 시험기관 in writing before
the procedure is finalised.

**Effort:** coordination only. **Blocks:** everything.

### 2-2. Implement the video track — *41% of the corpus is currently unevaluable*

Video and audio are still behind the P3 guard, and `router._VIDEO_EXTS` does not list `.wmv` or
`.swf`, so those fall through to the document path and die on the OCR-empty guard.

Confirmed by a live run: a `.wmv` set returned
`저작물 처리 실패(media=video): OCR에서 텍스트를 추출하지 못했습니다`.

**2,000 of 4,834 sets (41%)** cannot be evaluated at all today.

Everything needed is already prepared:
- Model selected and evidenced — `qwen3.5-omni-plus`, 92.3% / 5.0 s (`docs/video_model_comparison_report_20260803_EN.md`)
- Implementation plan written (`docs/video_track_implementation_plan_20260731_EN.md`)
- Keyframe strategy specified in plan §3-4 (farthest-point selection — **implement it literally**;
  the comparison harness used a simpler greedy pass that was front-biased)

**Effort:** 2–3 weeks per the plan. **Blocks:** the multimodal indicator entirely.

### 2-3. Generate contracts for the evaluation sets

Without a contract, 제목·저자·라이선스 cannot be scored honestly (see 2-1).

**Generate from 공유마당 metadata, not KOGL.** Ground-truth quality differs sharply:

| | 공유마당 (4,834) | KOGL 붙임1 (144k) |
|---|---|---|
| 제목 · 저자 · 라이선스 | 100% | available |
| **설명** | **73%** | **79 records total** — no description column in the export |
| **해상도** | **Tier A 64%** (site-published, independent of our extractor) | not in export → self-measured = circular |
| 파일 생성 날짜 | 89% | 제작일자 is "조선" etc. — not a date |

Two consistency issues seen in the sample contract (`0034_다산박물관_소장유물_이학편(천자).pdf`):

1. **Rights disagree with the source.** The metadata grants all 7 rights; the contract checks only
   4 (복제·배포·대여·2차적저작물). If that is deliberate variation, fine — but then the *contract*
   is the ground truth for rights, and the generator must record what it actually wrote.
2. **License semantics contradict the contract type.** The source is `공공누리 제1유형` /
   `만료저작물` / `계약문서: 무` — a free-to-use expired public work — yet the document is an
   **exclusive licence with a ₩2,700,000 fee**. Condition the template on `공공누리 유형` so
   expired/Type-1 works get a royalty-free licence instead.

**Also:** name files by `원문인덱스` (`63155.pdf`), not by 원본파일명. The `0034_` prefix is shared
by 20 works and full 원본파일명 collides 6,847 times in the export.

**Effort:** template work + a generation run. **Blocks:** honest scoring of 3 attributes.

### 2-4. Cap OCR pages — *measured, not theoretical*

A single 어문 work took **1,407 seconds (23 minutes)** end-to-end in the live run. Our 어문 PDFs
average **93 pages** (median 101; 52% exceed 100 pages), and OCR bills per page.

| | Full OCR | First 5 pages |
|---|---|---|
| Cost per 어문 work | ₩296 | ₩58 |
| 4,000-set run | ~₩324,000 | ~₩161,000 |

We extract 제목·저자·설명·키워드 — OCR'ing all 209 pages of a book to read its title is waste.
A page cap roughly halves both cost and wall-clock.

**Effort:** ~half a day. **Blocks:** any run of meaningful size.

### 2-5. Run a stratified pilot

400 sets (100 per rights type) gives a **±3.5%** confidence interval at p≈0.85 — enough to know
whether we are near target — for about a tenth of the cost of the full 4,000.

| Sample | 95% CI | Est. cost |
|---|---|---|
| 200 | ±5.0% | ~₩18,000 |
| **400** | **±3.5%** | **~₩35,000** |
| 4,000 | ±1.1% | ~₩352,000 |

Reserve the full 4,000 for the official test, where completeness is the requirement.

---

## 3. Ground-truth quality — issues that distort scoring

### 3-1. Keyword ground truth is the wrong shape

Observed in the live run:

```
GT (분류_장르): ['어문']
Extracted:      ['詩選', '고전시가', '한문', '중국문학', '시가']
Score:          0
```

The extracted keywords are **better than the answer key**, and score zero. 공유마당's `분류_장르`
is a 3-level taxonomy (`회화 일반 · 회화 · 미술`), not free keywords.

KOGL's `주제어` / `해시태그` (`#소장품 #유물`) is much closer to what a keyword extractor produces.
**Recommendation:** split the sources — 공유마당 for 설명 and 해상도, KOGL-style 주제어 for 키워드.

### 3-2. Hanja/Hangul breaks title matching

```
GT: 시선     Extracted: 詩選     → scored wrong
```

The same title in different scripts. Normalization does not bridge Hanja and Hangul. Needs either
a transliteration step or an explicit equivalence rule.

### 3-3. 주요 색상 / 개체 범주 have no ground truth anywhere

Not in 공유마당, not in KOGL, not in the file. The metadata re-scrape closed every other gap but
could not close this one. Options: label a subset by hand, or define a computable rubric (dominant
colour is checkable against an image histogram; 개체 범주 is not).

**Until resolved, 2 of 11 attributes cannot be scored.**

### 3-4. Description scoring method is unsettled

Ground truth *"카메라 묘기를 가장 먼저 사용한 영화…"* vs output *"흑백 무성영화로 처형 장면이…"*
— both correct, zero string overlap. Current default is `content_recall` (does the output cover
the GT's content words, threshold 0.30): reproducible and auditable, but blind to correct
paraphrase in different vocabulary. Alternatives: embedding similarity, an LLM judge, or the
per-work item rubric used in the video model comparison. The scorer accepts a drop-in replacement.

### 3-5. Smaller gaps

- **KOGL is 500 short** of the 1,000-per-rights-type requirement. 공유마당 holds no original files
  for KOGL 어문·영상 (46 `external_no_file` evidence records); the shortfall must come from the
  KOGL site directly.
- **`work_title` filename fallback** — the title appears in the filename for **75.5%** of works.
  Deterministic and hallucination-free, but 공유마당 names carry a `wrtSn_fileSn_` prefix needing
  a strip rule.

---

## 4. Requires agreement with 콘진원 / 시험기관

| Question | Why it matters |
|---|---|
| **Test input format** (§2-1) | Decides whether the number means anything |
| 4,000 per indicator, or 4,000 shared? | The plan's wording is identical for both indicators; 8,000 doubles cost and time |
| Definition of 파일 생성 날짜 | Plan says *file* creation; the site publishes *work* creation (창작년도 1895). Currently excluded from scoring as a definition mismatch |
| Scoring rules per attribute | Our rubric must match theirs, or our pilot number does not predict the official result |
| Which attributes apply to 어문 | The site itself publishes no 해상도 for 어문 — supporting 8 attributes for text, 11 for visual |
| 라이선스 유형 overlap | It is one of our 11 attributes *and* the substance of HM컴퍼니's 권리유형 분류 indicator |

---

## 5. Operational

- **Commit the backlog** — 18 uncommitted paths spanning this session and earlier work.
- **Update `docs/평가대응_준비현황_20260803.md`** with measured figures (resolution 96.8%,
  per-attribute GT coverage, the 23-minute document).
- **Preserve throwaway scripts** — `vidtest_full.py`, `vidbench.py`, `tokmeter.py` still sit in a
  disposable job directory.
- **Set OpenRouter `data_collection: "deny"`** — still at the default `allow`.
- **RRN masking** not implemented (3–4 weeks; design in `docs/주민등록번호_마스킹_설계검토_20260731.md`).
- **Oracle disk** — now 93% (3.5 GB free) after reclaiming the 1.2 GB stale tarball. Still tight;
  the box is shared with tfg-app, Docker/postgres and bondeal.

---

## 6. Suggested sequence

1. **Send the input-format question** to 콘진원 / 시험기관 today — it gates everything and is
   pure lead time.
2. **Implement the video track** (2–3 weeks) — highest value work we can do unblocked; recovers
   41% of the corpus, and both the model decision and the plan are already written.
3. **Add the OCR page cap** (~half a day) — halves pilot cost and wall-clock.
4. **Generate contracts from 공유마당 metadata**, named by `원문인덱스`, with the licence-consistency
   fix.
5. **Fix keyword GT source and Hanja normalization** — otherwise the pilot under-reports.
6. **Run the 400-set pilot** → first real accuracy number.
7. Decide on 주요 색상 / 개체 범주 and the description scoring method based on what the pilot shows.

Steps 2 and 3 are independent and can run in parallel.

---

## Appendix. Measured facts referenced above

| Fact | Value | Source |
|---|---|---|
| Resolution extraction accuracy | 96.8% (242/250) | Cross-check vs site-published values |
| GT records built | 4,834 | `<cell>/ground_truth.jsonl` |
| Metadata re-scrape | 4,834 pages, 0 failures, 16.1 min | `rescrape_details.py` |
| 설명 GT coverage | 73.0% | 공유마당 요약정보, filtered |
| 파일 생성 날짜 GT | 89% (0% before re-scrape) | 창작년도/공표년도 |
| Video model | qwen3.5-omni-plus 92.3% / 5.0 s | 8 models × 5 videos, 40 calls |
| 어문 PDF length | median 101 pages, mean 93 | 50-file sample |
| Single document runtime | 1,407 s | Live batch run |
| Title in filename | 75.5% | All 4,834 |
| ext4 vs 9p small-file I/O | 10× write, 39× read | 200 × 300 KB benchmark |

## Related documents

- `docs/평가대응_준비현황_20260803.md` — readiness status (needs updating with the above)
- `docs/video_track_implementation_plan_20260731_EN.md` — video track design
- `docs/video_model_comparison_report_20260803_EN.md` — model selection evidence
- `docs/api_cost_evaluation_20260730_EN.md` — verified unit costs
- `docs/시험인증요청서_TTA_초안.md` / `_KTC_초안.md` — certification request drafts
