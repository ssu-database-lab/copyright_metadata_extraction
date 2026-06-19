# Paper 1 — NER Context-Length Sensitivity (§24-2)

## 가설

> "NER은 `name` 같은 짧은 스팬은 잘 분류하지만, 앞뒤로 context가 늘어날수록 분류 성능이 떨어진다."

검증: span 길이가 짧은 라벨일수록 context-length 증가 시 F1 하락폭이 클 것으로 예상.
span 길이가 긴 라벨은 context의 영향이 작을 것으로 예상.

## 실험 대상 라벨 (11개)

| 라벨 | 분류 | Silver | Gold | 비고 |
|---|---|---:|---:|---|
| `name` | 짧은 span | 20,863 | 2,366 | 2-4자 한글 |
| `phone` | 짧은 span | 19,310 | 1,000 | 전화번호 패턴 |
| `email` | 짧은 span | 17,223 | 532 | email 패턴 |
| `date` | 짧은 span | 6,000 | 2,067 | 날짜 포맷 |
| `address` | 긴 span | 18,477 | 203 | 여러 어절 주소 |
| `company` | 긴 span | 22,807 | 1,721 | 기관명 |
| `copyright_url` | 긴 span | 5,928 | 2,000 | URL |
| `copyright_Keyword` | 긴 span | 4,000 | 1,924 | #태그 모음 |
| `copyright_kotitle` | 매우 긴 span | 5,114 | 14,937 | 저작물 제목 |
| `copyright_description` | 매우 긴 span | 4,800 | 2,000 | 설명 문장 |
| `ri_period` | 짧은 span | 6,000 | 2,000 | 기간 연도 |

## Case Level (어절 기준 ±N)

| 레벨 | 의미 | 예 (`"저작자 : 주거지 : 전라북도 노원구 … 627호"` 에서 `address` 스팬) |
|---|---|---|
| `case0` | target span만 | `"전라북도 노원구 판교역중앙로 491 26동 627호"` |
| `case1` | ±1 어절 | `": 전라북도 노원구 판교역중앙로 491 26동 627호"` |
| `case3` | ±3 어절 | `"저작자 : 주거지 : 전라북도 노원구 … 627호"` |
| `case5` | ±5 어절 | 원본에 가까움 |
| `casefull` | 원본 그대로 | 축소 없음 |

**생성 규칙**:
- Silver (BIO 포맷): `labels` 에서 `B-{label}` ~ 연속된 `I-{label}` 스팬 위치 탐지 → 주변 ±N 토큰만 남기고 tokens/labels 동시 축소.
- Gold (`{text, answer, source}`): text 어절 split → answer 어절과 일치하는 구간 탐지 → 주변 ±N 어절만 남기고 text 재구성, answer/source 유지.

## 데이터셋 파일

```
paper1/configs/case_test/
├── silver/
│   ├── {label}_case0/{label}.jsonl       (훈련용 BIO)
│   ├── {label}_case1/{label}.jsonl
│   ├── {label}_case3/{label}.jsonl
│   ├── {label}_case5/{label}.jsonl
│   └── {label}_casefull/{label}.jsonl
└── gold/
    ├── {label}_case0/{label}.jsonl       (평가용 {text, answer, source})
    ├── {label}_case1/{label}.jsonl
    ├── {label}_case3/{label}.jsonl
    ├── {label}_case5/{label}.jsonl
    └── {label}_casefull/{label}.jsonl
```

11 labels × 5 case levels × 2 (silver/gold) = **110개 서브디렉토리**.

## 훈련·평가 설정

- 모델: BERT (KLUE) / Full FT / Integrated (§24-5·§24-7 고정)
- Split: 8:2:2 (train/val/test, `SPLIT_SEED=42`)
- 평가: Silver train → Gold test, per-label F1
- 로깅: `paper_module/core/run_logger` 로 step별 loss·eval·hparam 전량 기록

## 결과 분석 플롯

| 파일 | 내용 |
|---|---|
| `label_caseN_f1.png` | 각 case level의 라벨별 F1 막대 (5장) |
| `case_vs_f1_bylabel.png` | x=case level, y=F1, 라벨별 선 — 핵심 결과 |
| `short_vs_long_spans.png` | 짧은 span 평균 vs 긴 span 평균, case 따라 비교 |
| `training_curve_{label}_caseN.png` | 학습 곡선 (디버깅용) |

## 예상 결과 패턴 (가설)

- `name`, `phone`, `email`: casefull → case0 로 갈수록 F1 상승 (context가 오히려 방해)
- `copyright_kotitle`, `copyright_description`: case0 에서는 span 자체가 커 context 영향이 작음
- 중간 라벨 (`address`, `company`): 중간 경사

## 학술대회 논문 구조 (4-8p)

1. 서론 — NER의 context 영향 문제
2. 관련연구 — BIO NER, 공공저작물 도메인
3. 실험 설계 — 위 표/경로
4. 결과 — 핵심 플롯 2-3장
5. 논의 — 가설 검증, 실무 시사점
6. 한계·결론 — 단일 모델·도메인 한정

---

**체크리스트** (실행 순서):
- [x] 데이터 생성 스크립트 (`scripts/build_case_test.py`)
- [x] 생성 후 샘플 수 확인 (각 파일 200+ 목표)
- [x] 훈련 실행 (라벨 × case 조합 반복) — 275/275 완료 (pre-§24-14 스냅샷)
- [ ] §24-14 Silver 기반 재런 + 로그 수집 → 플롯 생성
- [ ] 논문 초고 (`paper1.md`)
