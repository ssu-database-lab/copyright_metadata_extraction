# Set 63155 — Contract vs Image Extraction Comparison

**Author** Soongsil University DB Lab · **Date** 2026-08-18
**Subject** Why the evaluation score is what it is, and where extracted values diverge from the answer key
**Set** `63155` — 이학편(천자), the only test set that currently has a real generated contract

---

## 1. First, the score is not low

The number to be careful with is **which** score:

| | 채점 | 정확도 |
|---|---|---|
| **Set 63155 alone** (contract + image) | 6/7 | **85.7%** — at target |
| Same set, re-run 2026-08-18 | 5/7 | **71.4%** |
| 8-set batch average | 29/63 | **46.0%** |

The 46% figure is an average dominated by **seven sets that have no contract**. Without a contract
document there is nothing to extract 제목·저자·라이선스 from, so those three come back empty every
time — a structural ceiling, not an extraction failure.

```
계약서 있음 (63155)   6/7   = 85.7%
계약서 없음 (7건)    23/56  = 41.1%
전체                29/63  = 46.0%
```

**The denominator matters more than the numerator.** Only **7 of the 11** evaluated attributes were
scored at all. Four were excluded before any comparison happened (§4).

---

## 2. What each leg actually extracted

The set runs two independent pipeline passes, then merges.

### 2-1. Contract leg — `63155.pdf` (OCR → LLM ∥ NER → 통합검증)

OCR produced **3,556 characters**; NER found **12 entities**; the LLM filled **30 fields**.

| 필드 | 값 |
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
| payment_amount / currency | 2700000 / 원 |
| signature_date / effective_date / expiration_date | 2026-08-13 / 2026-08-13 / 2030-08-12 |
| commercial_use | 허락 |
| copyrightability / unprotected_work | 인정 / 비해당 |
| special_terms | ['이용료 지급 후 다음날부터 기간 계산 가능 (제 3 조)', …] |
| termination_conditions | ['천재지변 또는 기타 불가항력으로…', …] |
| signature | ['전라남도 강진군', '합성이용기관 디지털아카이브원'] |
| personal_info | ['성명', '생년월일', '주소'] |
| language | 한국어 |

### 2-2. Image leg — `63155.jpg` (VLM → 스키마 매핑 → 통합검증)

Filled **9 fields**.

| 필드 | 값 |
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

This is the structural point. Five fields are populated by **both** legs, with values that are each
correct for their own input but describe different objects:

| 필드 | 계약서 leg | 이미지 leg | 병합 결과 | 올바른 선택? |
|---|---|---|---|---|
| `digital_format` | PDF | JPG | **JPG** | ✅ 저작물 기준 |
| `file_size` | 97,757 (계약서 파일) | 311,813 (저작물 파일) | **311813** | ✅ |
| `description` | 계약의 목적 | 고서 표지 사진 설명 | **이미지 쪽** | ✅ |
| `keyword` | 계약 용어 | 저작물 내용어 | **이미지 쪽** | ✅ |
| `work_title` | 이학편 (천자) | (없음) | **계약서 쪽** | ✅ |

The merge resolves all five correctly today, but **the precedence is implicit**. Nothing in the code
states "for a work set, file-derived and visual fields come from the work leg." If a future change
altered merge order, `digital_format` would silently become `PDF` and `file_size` the contract's —
both would then score as wrong with no error anywhere.

**Recommendation:** make the precedence explicit and documented, not emergent.

---

## 3. Expected vs extracted — attribute by attribute

| # | 속성 | 정답 (GT) | 추출 | 판정 |
|---|---|---|---|---|
| 1 | 제목 | `이학편(천자)` | `이학편 (천자)` | ✓ 표기차 |
| 2 | 저자 | `전라남도 강진군` | `['전라남도 강진군']` | ✓ 타입차 |
| 3 | 설명 | — | 고서 표지 사진 설명 | · 정답 없음 |
| 4 | 라이선스 유형 | `제1유형` | `None` | ✗ **미추출** |
| 5 | 키워드 | `['소장품','유물']` | `['고서','아학편','천자문','한자','고문헌','전통서적']` | ✗ 어휘 불일치 |
| 6 | 해상도 | `1757x1172` | `1757x1172` | ✓ 완전일치 |
| 7 | 주요 색상 | — | `['갈색','흰색']` | · 정답 없음 |
| 8 | 개체 범주 | — | `['고서 표지','안내 문구 종이']` | · 정답 없음 |
| 9 | 파일크기 | `311813` | `311813` | ✓ 완전일치 |
| 10 | 파일포맷 | `JPG` | `JPG` | ✓ 완전일치 |
| 11 | 파일 생성 날짜 | — | `2026-08-18` | · 정의 불일치 |

---

## 4. Mismatch taxonomy — the actionable part

### (A) 완전일치 — 3속성
해상도 · 파일크기 · 파일포맷. All file-derived and deterministic. **These are the only attributes
that will never drift.**

### (B) 표기·타입 차이 — 2속성 (현재는 통과, 엄격 채점 시 실패)

```
제목   정답 "이학편(천자)"      추출 "이학편 (천자)"        ← 괄호 앞 공백
저자   정답 "전라남도 강진군"    추출 ["전라남도 강진군"]     ← 문자열 vs 배열
```

Both pass only because the comparator normalizes whitespace and unwraps single-element lists. Under
exact string matching both would **fail**. The space in 제목 originates in the OCR/extraction of the
contract, not in the source metadata. The same artifact appears inside
`economic_rights`: `2 차적저작물작성권` (space after the numeral).

**Action:** if the 시험기관 scores by exact match, these become losses. This must be settled in the
시험 절차서 — it is worth roughly 2 of 7 attributes on this set.

### (C) 어휘 불일치 — 키워드 (구조적 문제)

```
정답: ['소장품', '유물']                       ← 공유마당/KOGL 주제어 (분류 어휘)
추출: ['고서','아학편','천자문','한자','고문헌','전통서적']  ← 내용 기반 키워드
```

The extraction is arguably **more useful** than the answer key — it identifies the actual work
(아학편/천자문, a Joseon-era primer) while the key says only "소장품, 유물". They share nothing.

This is the same taxonomy-vs-content mismatch already recorded for 공유마당's `분류_장르`.
**The keyword answer key is the problem, not the model.**

### (D) 미추출 — 라이선스 유형 (추출 불가능한 속성)

The contract is a **저작재산권 독점적 이용허락 계약서**. It never states a 공공누리 유형, because
that is a KOGL licence designation, not a contract term. No extraction quality improves this.

Note also this attribute is simultaneously in our 11 evaluated attributes **and** the substance of
HM컴퍼니's 권리유형 분류 지표 — an overlap that needs resolving with 콘진원.

### (E) 정답 부재 — 3속성

설명 · 주요 색상 · 개체 범주. The pipeline produced plausible, apparently correct values for all
three. **None can be credited**, because no ground truth exists anywhere (붙임1 has no description
column; colour and object category exist in no source).

### (F) 정의 불일치 — 파일 생성 날짜

계획서 asks for *file* creation date; the available truth is 제작일자 = `조선` (a dynasty, not a
date). Excluded by default as a definition mismatch rather than scored as wrong.

---

## 5. The finding that matters most: the same set scored twice, differently

The identical set, identical files, identical models, run one day apart:

| | 키워드 추출 | '유물' 포함 | 점수 |
|---|---|---|---|
| 2026-08-17 | `['고서','천자문','전통서적','한자','박물관','유물','오래된 책']` | ✅ | **6/7 = 85.7%** |
| 2026-08-18 | `['고서','아학편','천자문','한자','고문헌','전통서적']` | ❌ | **5/7 = 71.4%** |

**A 14-point swing on one set, decided by whether the VLM happened to emit the single word 유물.**

Two consequences:

1. **Single-set and small-batch numbers are not measurements.** With 7 scored attributes, one
   attribute is worth 14 points. The 400-set stratified pilot (±3.5%) is not optional rigour — it is
   the minimum for any number anyone acts on.
2. **The keyword comparator sits on a cliff.** Passing depended on one term crossing a 0.34 recall
   threshold. Given (C) above — that the answer key vocabulary differs from the extraction vocabulary
   by design — keyword scoring is currently closer to a coin flip than a measurement.

---

## 6. What to fix, in order

| # | 항목 | 근거 |
|---|---|---|
| 1 | **키워드 정답 출처 교체** | (C)(§5) — 분류 어휘 vs 내용 어휘. KOGL `주제어` 가 더 근접하나 이 사례에서도 겹치지 않았다 |
| 2 | **엄격/유사 채점 기준 합의** | (B) — 공백·배열 차이만으로 2속성이 뒤집힌다 |
| 3 | **라이선스 유형 취급 결정** | (D) — 계약서에서 추출 불가. 입력으로 제공하거나 채점에서 제외해야 한다 |
| 4 | **주요 색상·개체 범주 정답 확보** | (E) — 사람 라벨링 또는 계산 가능한 루브릭 |
| 5 | **병합 우선순위 명시화** | §2-3 — 지금은 우연히 맞다 |
| 6 | **파일 생성 날짜 정의 합의** | (F) |

Attributes 1–3 alone determine whether this set scores 71% or 100%.

---

## 부록. 산출물

| 항목 | 경로 |
|---|---|
| 재실행 결과 (leg별 request_id 포함) | `/home/mbmk92/eval_staging/cmp63155/_out/results.jsonl` |
| 계약서 leg 산출물 | `_out/runs/results/20260818_150417_810381/` (ocr · ner · llm_metadata.json) |
| 이미지 leg 산출물 | `_out/runs/results/20260818_150636_369505/` |
| 정답 | `/home/mbmk92/eval_staging/cmp63155/ground_truth.jsonl` |
| leg별 필드 덤프 | `/home/mbmk92/eval_staging/cmp63155/legs.txt` |

UI 에서는 각 세트 행의 **ZIP** 링크로 동일한 산출물을 내려받을 수 있다.
