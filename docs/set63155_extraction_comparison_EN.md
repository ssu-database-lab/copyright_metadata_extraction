# Set 63155 — Contract vs Image Extraction Comparison

**Author** Soongsil University DB Lab · **Date** 2026-08-18
**Subject** Why the evaluation score is what it is, and where extracted values diverge from the answer key
**Set** `63155` — 이학편(천자), the only test set that currently has a real generated contract

> **Note on Korean terms.** Attribute names (제목, 저자, …) are kept in Korean because they are the
> official wording of the R&D plan's §2-4 attribute list, and all extracted values and ground-truth
> values are Korean data. Everything else is English.

---

## 1. First, the score is not low

Be careful *which* score is being read:

| Scope | Scored | Accuracy |
|---|---|---|
| **Set 63155 alone** (contract + image) | 6/7 | **85.7%** — at target |
| Same set, re-run one day later | 5/7 | **71.4%** |
| 8-set batch average | 29/63 | **46.0%** |

The 46% is an average dominated by **seven sets that have no contract**. With no contract document
there is nothing to extract 제목 (title), 저자 (author) or 라이선스 유형 (licence type) from, so those
three come back empty every time — a structural ceiling, not an extraction failure.

```
With contract    (63155)    6/7   = 85.7%
Without contract (7 sets)  23/56  = 41.1%
Combined                   29/63  = 46.0%
```

**The denominator matters more than the numerator.** Only **7 of the 11** evaluated attributes were
scored at all; four were excluded before any comparison happened (§4).

---

## 2. What each leg actually extracted

The set runs two independent pipeline passes, then merges them.

### 2-1. Contract leg — `63155.pdf` (OCR → LLM ∥ NER → consolidation)

OCR produced **3,556 characters**; NER found **12 entities**; the LLM filled **30 fields**.

| Field | Value |
|---|---|
| work_title | 이학편 (천자) |
| work_type | 사진저작물 |
| digital_format | **PDF** |
| file_size | **97757** |
| description | 저작재산권 이용허락과 관련하여 권리자와 이용자 사이의 권리관계를 명확히 하는 것을 목적으로 한다. |
| keyword | ['저작재산권', '독점적 이용허락', '계약서', '이학편', '강진군'] |
| copyright_holder | ['전라남도 강진군'] |
| economic_rights | ['복제권', '배포권', '대여권', '2 차적저작물작성권'] |
| granted_rights | ['복제권', '배포권', '대여권', '2 차적저작물작성권'] |
| contract_type | 저작재산권 독점적 이용허락 계약서 |
| valid_period / contract_duration | 2026-08-13~2030-08-12 |
| payment_amount / payment_currency | 2700000 / 원 |
| signature_date / effective_date / expiration_date | 2026-08-13 / 2026-08-13 / 2030-08-12 |
| commercial_use | 허락 |
| copyrightability / unprotected_work | 인정 / 비해당 |
| special_terms | ['이용료 지급 후 다음날부터 기간 계산 가능 (제 3 조)', …] |
| termination_conditions | ['천재지변 또는 기타 불가항력으로…', …] |
| signature | ['전라남도 강진군', '합성이용기관 디지털아카이브원'] |
| personal_info | ['성명', '생년월일', '주소'] |
| language | 한국어 |

### 2-2. Image leg — `63155.jpg` (VLM → schema mapping → consolidation)

Filled **9 fields**.

| Field | Value |
|---|---|
| work_type | 사진저작물 |
| digital_format | **JPG** |
| resolution | 1757x1172 |
| file_size | **311813** |
| file_created_date | 2026-08-18 |
| dominant_colors | ['갈색', '흰색'] |
| main_subjects | ['고서 표지', '안내 문구 종이'] |
| description | 오래된 고서의 표지를 촬영한 사진입니다. 표지에는 한자가 적혀 있으며, 상단에는 '13 아학편 (천자)'이라는… |
| keyword | ['고서', '아학편', '천자문', '한자', '고문헌', '전통서적'] |

### 2-3. Both legs fill the same fields with different meanings

This is the structural point. Five fields are populated by **both** legs, each correct for its own
input but describing a different object:

| Field | Contract leg | Image leg | Merged | Correct choice? |
|---|---|---|---|---|
| `digital_format` | PDF | JPG | **JPG** | ✅ the work, not the contract |
| `file_size` | 97,757 (contract file) | 311,813 (work file) | **311813** | ✅ |
| `description` | purpose of the agreement | photo of an old book cover | **image leg** | ✅ |
| `keyword` | contract vocabulary | work-content vocabulary | **image leg** | ✅ |
| `work_title` | 이학편 (천자) | (none) | **contract leg** | ✅ |

The merge resolves all five correctly today, but **the precedence is implicit**. Nothing in the code
states "for a work set, file-derived and visual fields come from the work leg." If a future change
altered merge order, `digital_format` would silently become `PDF` and `file_size` the contract's —
both would then score as wrong, with no error raised anywhere.

**Recommendation:** make the precedence explicit and documented rather than emergent.

---

## 3. Expected vs extracted — attribute by attribute

| # | Attribute | Ground truth | Extracted | Verdict |
|---|---|---|---|---|
| 1 | 제목 (title) | `이학편(천자)` | `이학편 (천자)` | ✓ formatting difference |
| 2 | 저자 (author) | `전라남도 강진군` | `['전라남도 강진군']` | ✓ type difference |
| 3 | 설명 (description) | — | photo description | · no ground truth |
| 4 | 라이선스 유형 (licence type) | `제1유형` | `None` | ✗ **not extracted** |
| 5 | 키워드 (keywords) | `['소장품','유물']` | `['고서','아학편','천자문','한자','고문헌','전통서적']` | ✗ vocabulary mismatch |
| 6 | 해상도 (resolution) | `1757x1172` | `1757x1172` | ✓ exact |
| 7 | 주요 색상 (dominant colours) | — | `['갈색','흰색']` | · no ground truth |
| 8 | 개체 범주 (object categories) | — | `['고서 표지','안내 문구 종이']` | · no ground truth |
| 9 | 파일크기 (file size) | `311813` | `311813` | ✓ exact |
| 10 | 파일포맷 (file format) | `JPG` | `JPG` | ✓ exact |
| 11 | 파일 생성 날짜 (file creation date) | — | `2026-08-18` | · definition mismatch |

---

## 4. Mismatch taxonomy — the actionable part

### (A) Exact match — 3 attributes
해상도, 파일크기, 파일포맷. All file-derived and deterministic. **These are the only attributes that
will never drift.**

### (B) Formatting and type differences — 2 attributes (pass today, fail under strict scoring)

```
제목   GT "이학편(천자)"       extracted "이학편 (천자)"        ← space before the bracket
저자   GT "전라남도 강진군"     extracted ["전라남도 강진군"]     ← string vs array
```

Both pass only because the comparator normalizes whitespace and unwraps single-element lists. Under
exact string matching, **both would fail**. The space in 제목 originates in the OCR/extraction of the
contract, not in the source metadata. The same artifact appears inside `economic_rights`:
`2 차적저작물작성권` (space after the numeral).

**Action:** if the certification body scores by exact match, these become losses — roughly 2 of 7
attributes on this set. It must be settled in the test procedure document.

### (C) Vocabulary mismatch — 키워드 (structural problem)

```
GT:        ['소장품', '유물']                                    ← classification vocabulary
Extracted: ['고서','아학편','천자문','한자','고문헌','전통서적']      ← content vocabulary
```

The extraction is arguably **more useful** than the answer key: it identifies the actual work
(아학편/천자문, a Joseon-era primer for learning Chinese characters), while the key says only
"collection item, artefact". The two share nothing.

This is the same taxonomy-versus-content mismatch already recorded for 공유마당's `분류_장르` field.
**The keyword answer key is the problem, not the model.**

### (D) Not extracted — 라이선스 유형 (structurally unavailable)

The document is an exclusive copyright licence agreement (저작재산권 독점적 이용허락 계약서). It never
states a 공공누리 (Korea Open Government Licence) type, because that is a licence designation
attached to the work in the KOGL registry, not a term of the contract. No amount of extraction
quality changes this.

Note also that this attribute sits simultaneously in our 11 evaluated attributes **and** in
HM컴퍼니's rights-type classification indicator — an overlap that needs resolving with 콘진원.

### (E) No ground truth — 3 attributes

설명, 주요 색상, 개체 범주. The pipeline produced plausible and apparently correct values for all
three. **None can be credited**, because no ground truth exists in any source: the KOGL export has no
description column, and neither colour nor object category exists anywhere.

### (F) Definition mismatch — 파일 생성 날짜

The plan asks for the *file* creation date; the only available truth is 제작일자 = `조선` (a dynasty,
not a date). Excluded by default as a definition mismatch rather than scored as wrong.

---

## 5. The finding that matters most: the same set scored twice, differently

Identical set, identical files, identical models, run one day apart:

| Run | Extracted keywords | Contains 유물 | Score |
|---|---|---|---|
| 2026-08-17 | `['고서','천자문','전통서적','한자','박물관','유물','오래된 책']` | ✅ | **6/7 = 85.7%** |
| 2026-08-18 | `['고서','아학편','천자문','한자','고문헌','전통서적']` | ❌ | **5/7 = 71.4%** |

**A 14-point swing on one set, decided by whether the model happened to emit the single word 유물.**

Two consequences:

1. **Single-set and small-batch numbers are not measurements.** With 7 scored attributes, one
   attribute is worth 14 points. The 400-set stratified pilot (±3.5%) is not optional rigour — it is
   the minimum for any number anyone acts on.
2. **The keyword comparator sits on a cliff.** Passing depended on one term crossing a 0.34 recall
   threshold. Given (C) — that the answer key and the extraction use different vocabularies by
   design — keyword scoring today is closer to a coin flip than a measurement.

---

## 6. What to fix, in order

| # | Item | Rationale |
|---|---|---|
| 1 | **Replace the keyword ground-truth source** | (C), §5 — classification vocabulary vs content vocabulary. KOGL `주제어` is closer, but did not overlap even here |
| 2 | **Agree strict vs fuzzy scoring** | (B) — whitespace and array differences alone flip 2 attributes |
| 3 | **Decide how 라이선스 유형 is handled** | (D) — not extractable from a contract; supply it as input or exclude it from scoring |
| 4 | **Obtain ground truth for 주요 색상 and 개체 범주** | (E) — human labelling or a computable rubric |
| 5 | **Make merge precedence explicit** | §2-3 — currently correct by accident |
| 6 | **Agree the definition of 파일 생성 날짜** | (F) |

Items 1–3 alone determine whether this set scores 71% or 100%.

---

## Appendix — artefacts

| Item | Path |
|---|---|
| Re-run results (with per-leg request IDs) | `/home/mbmk92/eval_staging/cmp63155/_out/results.jsonl` |
| Contract-leg artefacts | `_out/runs/results/20260818_150417_810381/` (ocr · ner · llm_metadata.json) |
| Image-leg artefacts | `_out/runs/results/20260818_150636_369505/` |
| Ground truth | `/home/mbmk92/eval_staging/cmp63155/ground_truth.jsonl` |
| Per-leg field dump | `/home/mbmk92/eval_staging/cmp63155/legs.txt` |

The same artefacts are downloadable from each result row's **ZIP** link in the `/v3` console.
