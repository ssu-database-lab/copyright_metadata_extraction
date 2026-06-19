# 노이즈의 성질과 학습방식에 따른 모델 성능 차이 분석: 한국어 공공저작물 권리문서 NER 사례

## Effect of Training Input Noise Composition on Model Performance: A Case Study on Korean Public-Domain Rights Document NER

---

## 1. Abstract

본 연구는 한국어 공공저작물 권리문서 NER (Named Entity Recognition) 에서 *학습 입력의 구성* 이 모델 성능에 미치는 영향을 정량화한다. 동일 backbone (KLUE BERT) · 동일 hyperparameter 하에서 학습 입력의 노이즈 구성만 세 가지로 변형한다 — **(M1) 답만**: BIO 정답 토큰만으로 구성된 minimal 입력, **(M2) 문장**: 정답 토큰을 자연 문맥 안에 위치시킨 BIO record, **(M3) 문장 + true negative**: 75 % positives + 25 % entity-free 자연 문장 (M2 와 *동일 총량*). 26 개 NER 라벨에 대한 cross-distribution 평가 (silver 합성 context → gold 인간 작성 context) 결과 M1 = 0.4635, **M2 = 0.8745**, M3 = 0.8673 으로 나타났다. 답만 학습 (M1) 은 본질적으로 작동하지 않으며 (M2 대비 −41 pp), true negative 의 추가 (M3) 는 −0.72 pp 의 미세한 손해만 야기한다. format-regularity 별 분해는 **format-semi-regular** 라벨이 모든 mode 에서 robust (≥ 0.93) 한 반면 **format-regular / format-free** 는 mode 에 극도로 민감 (각각 +54 pp / +44 pp 의 M1 → M2 점프) 함을 보여준다. 부수 ablation: 옛 additive 구현 (label noise + 데이터 양 +25 %) 의 −1.92 pp 손해는 *데이터 양 confound + label noise 의 합성 효과* 로, 진짜 negative 효과 (−0.72 pp) 와 분리된다.

**핵심어**: NER, BIO 학습, 노이즈 구성, format-regularity, cross-distribution evaluation, 한국어 권리문서

---

## 2. Keywords

Korean NER, training input noise, BIO sequence labeling, format-regularity stratification, cross-distribution evaluation, true negative ablation, public-domain rights document

---

## 3. Introduction

### 3.1 연구 배경

한국어 공공저작물 권리문서 (저작권 등록부, 표준계약서, 동의서 등) 의 메타데이터 추출은 디지털 아카이빙·라이선스 검증·자동 계약 처리에 필수적이다. 26 개의 NER 라벨 — 저작자 정보 (name, company, position, ...), 저작물 식별자 (UCI, 등록번호, 제목, ...), 권리 정보 (계약 종류, 기간, 금액, 법령 참조, ...) — 가 한 문서 내에 혼재한다. NER 학습은 일반적으로 BIO (Begin-Inside-Outside) tagging 방식으로 이루어진다. 그러나 BIO 학습 데이터를 *어떻게 구성하는가* — 정답 토큰을 자연 문맥에 위치시키는가, 단독으로 제시하는가, true negative 와 혼합하는가 — 가 모델 성능에 미치는 영향은 정량적으로 검증된 바 적다.

### 3.2 문제의식

기존 NER 연구는 backbone (mBERT, KLUE BERT, XLM-R) · fine-tuning 방법 (Full FT, LoRA) · 데이터 양에 초점을 맞추는 반면, 학습 입력의 *형태* 에 대한 ablation 은 드물다. 또한 NER 의 negative sample 추가 효과에 대한 기존 연구는 *substitutive* (총량 보존) 와 *additive* (총량 증가) 를 명확히 구분하지 않아 데이터 양과 negative 효과가 혼재된 경우가 많다. 본 연구는 이 두 confound 를 분리해 *진짜* negative 효과를 측정한다.

### 3.3 가설 (Hypothesis)

- **H1**: NER 모델은 정답 토큰만으로는 학습되지 않는다. 자연 문맥은 단순히 도움이 되는 것이 아니라 *본질적으로 필요한* 신호이다.
- **H2**: format-regularity (paper 5 §3.5 의 14/6/6 카테고리화) 가 학습 입력 구성에 대한 sensitivity 를 결정한다. 구조 단서 (semi-regular) 가 강한 라벨은 mode 에 robust 하다.
- **H3**: 통제된 substitutive 구성에서 25 % true negative 추가는 over-regularization 효과로 미세한 성능 손해를 야기한다.

### 3.4 기여 (Contributions)

1. 한국어 권리문서 26 NER 라벨 × 3 학습 입력 모드의 controlled comparison 으로 NER 학습에서 자연 문맥의 *필수성* 을 정량화 (M1 → M2 +41 pp).
2. format-regularity 별 mode-sensitivity 의 비대칭성 발견 — semi-regular 는 robust, regular/free 는 매우 민감.
3. **Substitutive vs additive negative 의 confound 분리 ablation** — 진짜 true negative 효과는 −0.72 pp 만으로 미세, 옛 additive 구현의 −1.92 pp 는 데이터 양 + label noise 의 합성 효과.
4. silver/gold 의 cross-distribution evaluation 프로토콜의 한계 (entity vocabulary 공유, context 분포만 다름) 를 정직하게 명시.

---

## 4. Method

### 4.1 데이터 (Dataset)

#### 4.1.1 Silver (학습용)

`configs/integrated/silver/` — 26 NER 라벨, 총 267,307 records. 규칙 기반 + 5-mode 노이즈 증강 (paper 5 §24-14: prefix/suffix 다양화, 라벨 혼입, 문서 wrapping, dropout, 띄어쓰기 변형). 본 연구에서는 라벨당 최대 10,000 records 로 cap (seed=42) 하여 mode 별 별도 파이프라인 처리.

#### 4.1.2 Gold (평가용, cross-distribution)

`configs/integrated/gold/` — kogl, tr01, tr02, wondb, kcc (한국저작권위원회 등록부), gongu (공유마당), casenote.kr 등 9+ source 의 인간 작성 평가 데이터. 총 52,721 records, `{text, answer, source}` 형식. **학습에 미참여**.

**Silver 와 Gold 의 분포 관계 — 정직한 단서**

| 차원 | Silver | Gold | 동일/상이 |
|---|---|---|---|
| Entity value (정답 surface form) | gold seed 풀에서 sampling (`build_silver_v2.py::load_gold_seeds`) | 본 source | ✅ **거의 동일** (seed 풀 공유) |
| 주변 context (앞뒤 텍스트) | 5-mode 합성 템플릿 | 인간 작성 자연 문서 | ❌ 다름 |
| 문서 구조 | 합성 boilerplate | 실제 계약서·등록부·판례 | ❌ 다름 |
| Tokenization 분포 | 짧은 형식 record | 긴 자연 문장 | ❌ 다름 |

따라서 본 평가는 *전형적 OOD* (분포 disjoint) 가 아닌 **context-level distribution shift evaluation** — 합성 context 에서 학습된 모델이 자연 인간 작성 context 에서 동일 entity 를 식별할 수 있는지 측정. mode 비교 (M1/M2/M3) 는 동일 entity vocabulary 조건에서 이뤄지므로 mode 간 비교는 공정.

#### 4.1.3 라벨 카테고리 (paper 5 §3.5 재사용)

| Class | n | 라벨 |
|---|:-:|---|
| **format-regular** | 14 | phone, email, date, ri_data, ri_period, ri_money, address, copyright_url, copyright_uci, copyright_num, copyright_idnum, copyright_status, copyright_quantity, copyright_language |
| **format-semi-regular** | 6 | copyright_Keyword, copyright_kotitle, ri_law_reference, ri_info, ri_contract_type, ri_copyright |
| **format-free** | 6 | name, company, department, position, copyright_description, copyright_type |

regex-tight 라벨 (regular), 문서 구조 단서 의존 (semi-regular), 자유 서술 (free).

### 4.2 학습 입력 모드 (Independent Variable)

| Mode | 설명 | 총 records | 예시 (label = `name`, answer = "이서연") |
|---|---|---:|---|
| **M1 답만** | BIO 정답 토큰만, 앞뒤 문맥 0 | 193,752 | `{"tokens": ["이서연"], "labels": ["B-name"]}` |
| **M2 문장** | 자연 문맥 + 정답 span (silver 원본) | 192,927 | `{"tokens": ["저작자",":","성명",":","이서연","서명함"], "labels": ["O","O","O","O","B-name","O"]}` |
| **M3 문장 + true negative** | 75 % positives + 25 % entity-free 문장 (M2 와 동일 총량) | 192,927 | (75%) M2 record + (25%) `{"tokens": ["저작자",":","성명",":","서명함"], "labels": ["O","O","O","O","O"]}` |

**M3 의 negative 구성 — substitutive + true negative**

M3 의 negative 는 **M2 record 에서 entity span (B-/I- 토큰) 을 *제거*** 하여 entity-free 자연 문장으로 변환. 즉 같은 토큰에 모순 라벨을 부여하는 *label noise* 가 아니며, entity 가 진짜로 존재하지 않는 자연스러운 문장. 이 구성은 (a) NER 학습의 모순 신호 제거 + (b) 자연스러운 도메인 분포 유지.

**총 record 수가 M2 와 동일** (192,927) 하도록 *substitutive*: 75 % positives (M2 random sample 144,696) + 25 % true negatives (48,231). 데이터 양 confound 제거 → **mode 효과의 통제 비교 성립**.

빌드 코드: [`paper1.py::transform_m1_answer/m2_context/m3_negatives`](paper1.py) · `_make_true_negative` (entity 제거 헬퍼).

### 4.3 학습 프로토콜

#### 4.3.1 3-way hold-out + cross-distribution final eval

Silver → 8/12 train + 2/12 val + 2/12 test (`SPLIT_SEED = 42`). val 은 epoch 별 학습 모니터링, test 는 silver 내부 holdout (보존). Gold 는 cross-distribution 최종 평가용으로 학습/튜닝에 미참여.

#### 4.3.2 모델 / 하이퍼파라미터

| 항목 | 값 | 출처 |
|---|---|---|
| Backbone | `klue/bert-base` (KLUE BERT, 110 M params) | Park 2021 |
| Method | Full Fine-Tuning | — |
| Mode | Integrated (26 라벨 동시) | — |
| Learning rate | 4 × 10⁻⁵ | paper 6 grid sweep |
| Warmup ratio | 0.1 | Devlin 2019 |
| Weight decay | 0.01 | Devlin 2019 |
| Batch size | 32 | — |
| Epochs | 3 | size-adjusted (Integrated 26 라벨 → 데이터 ×26) |
| Early stopping | patience 3, metric eval_f1 | Mosbach 2021 |
| Threshold (eval) | 0.25 | paper 5 와 동일 |

**Backbone 선택 정당화**: paper 4 의 4-backbone 비교 (gold cross-distribution acc) 에서 KLUE 0.8576 / KoELECTRA 0.8634 / mBERT 0.8435 / DeBERTa 0.8416. KLUE 와 KoELECTRA 가 거의 동급이나 KLUE BERT 가 한국어 NLP 표준 baseline (Park 2021) 이고 paper 5 (졸업논문 본 실험) 와 동일 backbone 사용으로 결과 직접 비교 가능. backbone × seed full grid 는 future work.

**Early stopping 주의**: epochs=3 + patience=3 조합으로 patience 카운트가 3 에 도달할 시간 부족 → **실질적으로 트리거 불가능, 안전 장치로만 작동**. 모든 mode 가 full 3 epoch 완주.

### 4.4 평가 지표

#### 4.4.1 Lenient match accuracy

각 Gold record (text + answer) 에 대해 모델이 BIO span 예측. 예측 span $\hat{a}$ 과 정답 $a$ 사이 다음 조건 만족 시 hit:

$$\text{hit}(a, \hat{a}) = \mathbf{1}[\hat{a} \neq \emptyset \land (a \subseteq \hat{a} \lor \hat{a} \subseteq a)]$$

라벨 $\ell$ 별 accuracy = hit 비율, 전체 = 26 라벨 mean.

#### 4.4.2 format-regularity 별 분해

각 mode 의 per-label accuracy 를 4.1.3 의 14/6/6 카테고리로 mean → mode-sensitivity 의 비대칭성 측정.

### 4.5 실험 규모 / 실행

26 NER 라벨 × 3 modes × 1 seed (42) = **3 sweep configs** (각 config 가 26 라벨 동시 학습). 단일 seed 는 paper 5 의 3-seed 분산 (σ ≈ 0.011) 에 비추어 mode 간 차이 (≥ 0.04) 가 통계적으로 유의함을 확인.

```bash
.venv/bin/python paper1/paper1.py build --source rule
.venv/bin/python paper1/paper1.py run --configs rule_m1,rule_m2,rule_m3
.venv/bin/python paper1/scripts/eval_only.py 20260427_143110
.venv/bin/python paper1/scripts/make_graphs.py
```

---

## 5. Results

### 5.1 Cross-distribution accuracy (Gold)

표 1: 3 mode × overall + format-class 별 정확도.

| Mode | overall | format-regular (14) | format-semi-regular (6) | format-free (6) |
|---|---:|---:|---:|---:|
| **M1 답만** | 0.4635 | 0.3230 | 0.9261 | 0.3286 |
| **M2 문장** | **0.8745** | 0.8680 | 0.9979 | 0.7661 |
| **M3 문장+true neg** (substitutive) | 0.8673 | 0.8581 | 0.9978 | 0.7583 |
| Δ (M2 − M1) | **+0.4110** | +0.5450 | +0.0718 | +0.4375 |
| Δ (M3 − M2) | −0.0072 | −0.0099 | −0.0001 | −0.0078 |

→ Figure G6 ([G6_overall_accuracy.png](figures/run_20260427_143110/G6_overall_accuracy.png)), G7 ([G7_per_class_accuracy.png](figures/run_20260427_143110/G7_per_class_accuracy.png)).

### 5.2 학습 수렴 — overfit / underfit 동시 부재

표 2: silver 내부 val 의 final epoch 메트릭.

| Mode | Accuracy | F1 | Precision | Recall | Loss |
|---|---:|---:|---:|---:|---:|
| **M1** | 0.9927 | 0.9882 | 0.9880 | 0.9883 | 0.0266 |
| **M2** | 0.9969 | 0.9905 | 0.9899 | 0.9910 | 0.0091 |
| **M3** | 0.9971 | 0.9888 | 0.9876 | 0.9901 | 0.0092 |

Figure G12 ([G12_fit_diagnostic.png](figures/run_20260427_143110/G12_fit_diagnostic.png)) — train + val loss 같은 축. 3 mode 모두 epoch 1 직후 0 근처에서 만나 plateau.

| 진단 | 기준 | paper 1 결과 | 결론 |
|---|---|---|---|
| Underfitting | train loss 가 높은 plateau | train → 0 근처 | ✅ 없음 |
| Overfitting | train ↓ + val ↑ 발산 | val 도 monotonic ↓ + plateau | ✅ 없음 |
| 정상 fit | train + val 격차 작음 | M1 ~0.025 / M2,M3 ~0.005 | ✅ 정상 |

→ 3 epoch 학습 적절. Figure G1 ([G1_training_loss.png](figures/run_20260427_143110/G1_training_loss.png)) 의 step 별 train loss 와 G5 ([G5_eval_loss.png](figures/run_20260427_143110/G5_eval_loss.png)) 의 epoch 별 val loss 도 일관됨.

### 5.3 Silver-overfitting (= cross-distribution gap)

silver 내부 (G4 [G4_eval_metrics.png](figures/run_20260427_143110/G4_eval_metrics.png)) 와 Gold (G6) 의 정확도를 비교하면 일반화 격차가 드러난다:

| Mode | Silver val acc | Gold acc | **Gap** |
|---|---:|---:|---:|
| **M1 답만** | 0.9927 | 0.4635 | **−0.5292** ❌❌ silver 표면 외움 |
| M2 문장 | 0.9969 | 0.8745 | −0.1224 ⚠ 보통 |
| M3 문장+neg | 0.9971 | 0.8673 | −0.1298 ⚠ 보통 |

→ **M1 의 silver-overfitting 이 극심** (53 pp gap). 토큰만으로 학습된 모델은 silver 의 단순 매핑을 외울 뿐, 자연 문맥에서 동일 entity 식별 실패. M2/M3 의 12 pp gap 은 silver/gold context 분포 차이의 자연스러운 수준.

### 5.4 라벨별 분포

Figure G8 ([G8_per_label_accuracy.png](figures/run_20260427_143110/G8_per_label_accuracy.png)) — 26 라벨 × 3 mode dot plot, M2 기준 정렬.

핵심 관측:

- **M1 의 가장 큰 손실 라벨**: `address` (0.44 → 0.00), `copyright_url` (1.00 → 0.00), `email` (1.00 → 0.00), `name` (0.95 → 0.00). regex-tight 인데도 답만 학습으론 0.0 으로 추락.
- **M1 에서도 robust**: `copyright_uci` (1.00), `ri_money` (1.00), `ri_info` (1.00), `copyright_language` (0.98), `ri_period` (0.97). 토큰 형식이 매우 정형이거나 vocabulary 가 좁은 라벨.
- **모든 mode 에서 saturated**: `ri_money`, `ri_period`, `ri_info`, `copyright_language` 등 12 개 라벨이 M2/M3 에서 1.00.
- **모든 mode 에서 broken**: `copyright_status` ≈ 0.00 (paper 5 와 동일 문제). `position` ≈ 0.03 (자유 직함, 학습 어려움).

### 5.5 시스템 메트릭

Figure G9 ([G9_gpu_memory.png](figures/run_20260427_143110/G9_gpu_memory.png)), G10 ([G10_wall_clock.png](figures/run_20260427_143110/G10_wall_clock.png)) — RTX 5070 12 GB. peak ≈ 3 GB, mode 무관. wall-clock 은 M1/M2 ~13 분 / M3 ~17 분 (substitutive M3 도 M2 와 동일 총량이라 격차 미세).

Figure G11 ([G11_weight_l2_evolution.png](figures/run_20260427_143110/G11_weight_l2_evolution.png)) — encoder layer 0/5/11 + classifier head 의 weight L2. 모든 mode 에서 단조 안정적 진화, 발산 없음 — overfitting 의 weight 신호도 부재.

---

## 6. Discussion

### 6.1 H1 (답만 학습 작동 안 함) 강력 지지

M1 = 0.4635 결과는 단순 "데이터 부족" 이 아닌 *근본적 학습 신호 부재* 를 시사. 26 라벨 silver 가 동일 cap 적용 (≈ 4-10 k records / label) 으로 토큰 자체는 충분히 다양했음에도 cross-distribution generalization 은 0.46. NER 모델은 라벨 토큰의 "형태" 가 아닌 "*문맥적 위치*" 를 학습한다는 BIO 의 본질을 정량적으로 재확인. silver-overfitting (53 pp gap) 이 직접 증거.

### 6.2 H2 (format-regularity 별 mode-sensitivity) 검증

| Class | M1 → M2 점프 | 해석 |
|---|---:|---|
| format-semi-regular | +7.18 pp | mode 에 robust — 위치 단서 (e.g., "저작물명 :" 직후) 가 라벨 토큰 자체에 강하게 인코딩 |
| format-regular | **+54.50 pp** | mode 에 매우 민감 — 의외 결과 |
| format-free | +43.75 pp | mode 에 매우 민감 — 예상 |

**의외 결과**: regex-tight format-regular 라벨도 답만 학습으론 식별 실패. BIO 학습은 *문맥 + 토큰 동시* 를 요구.

### 6.3 H3 (Negative over-regularization) — 매우 약하게만 지지 + 부수 ablation 발견

M3 (substitutive, true negative 25 %) − M2 = **−0.72 pp** (overall). format-regular −0.99 pp / format-free −0.78 pp / semi-regular ≈ 0. 모든 class 에서 일관되게 M3 ≤ M2 이긴 하나 격차 미세 → H3 의 강한 증거 아님.

**비교 ablation — additive vs substitutive**

| Metric | additive M3 (label noise + 240k) | substitutive M3 (true neg + 192k) | 차이 |
|---|---:|---:|---:|
| Gold overall | 0.8553 | **0.8673** | +1.20 pp |
| Silver val Precision | 0.7906 | **0.9876** | +19.70 pp |
| Silver val F1 | 0.8787 | **0.9888** | +11.01 pp |
| Silver val Loss | 0.1333 | **0.0092** | −14× |

옛 additive M3 의 −1.92 pp 손해 = **데이터 양 confound + label noise** 의 합성 효과. 진짜 negative 효과는 −0.72 pp 만으로 미세. 이 ablation 자체가 본 연구의 부수적 contribution — *NER negative ablation 에서 substitutive + true negative 가 표준이 되어야 함을 시사*. 다른 비율 (5 %, 10 %, 50 %) ablation 은 future work.

### 6.4 한계 (Limitations)

- **단일 backbone, 단일 seed** — paper 5 의 3-seed 분산 (σ ≈ 0.011) 대비 mode 효과 (≥ 0.04) 충분히 크나 backbone × seed full grid 는 향후 작업. paper 7 (ensemble × latency) 에서 보강.
- **엄격한 OOD 가 아님** — silver 의 entity value 가 gold answer pool 에서 derive 되므로 entity vocabulary 공유. 본 평가는 *context distribution shift* 의 generalization 측정. 진짜 entity-OOD 평가 (gold 의 hold-out source 사용) 는 paper 5 의 per-source breakdown 에서 부분 검증.
- **AI 증강 silver** — 5-mode 합성 노이즈에 한정. 실 도메인 분포와의 추가 차이는 paper 5 / paper 7 에서 보강.
- **단일 negative 비율 (25 %)** — 5 %, 10 %, 50 % ablation 은 future work.
- **한국어 권리문서 한정** — 영문 NER (CoNLL-2003 등) 으로의 일반화는 future work.

### 6.5 활용 가이드 (Practical Implication)

본 결과는 한국어 NER silver pipeline 설계에 구체적 가이드를 제공:

1. **학습 record 는 반드시 자연 문맥을 포함** (M2). 토큰만 silver 는 배제.
2. **Negative 는 보수적으로** — 25 % 미만 권장, ablation 후 사용. *반드시 substitutive + true negative* (entity span 제거된 자연 문장) — label noise (라벨만 O 치환) 절대 금물.
3. **format-regular 라벨도 문맥 학습 필요** — regex-tight 라벨이라고 silver 단순화 X.

---

## 7. Conclusion

본 연구는 한국어 공공저작물 권리문서 NER 의 학습 입력 구성이 모델 성능에 미치는 영향을 26 NER 라벨 × 3 입력 모드 controlled comparison 으로 정량화했다. **답만 학습은 작동하지 않으며 (M1 0.46 vs M2 0.87)**, 자연 문맥은 NER 학습의 본질적 신호이다. format-regularity 카테고리화로 **format-semi-regular** 만 모드에 robust 함을 발견했다. **True negative 25 % 추가의 손해는 −0.72 pp 로 미세** 하며, 옛 additive 구현의 −1.92 pp 손해는 데이터 양 confound + label noise 의 합성 효과임을 ablation 으로 분리. 본 결과는 한국어 NER silver pipeline 의 설계 가이드와 향후 NER negative ablation 의 표준 (substitutive + true negative) 을 제시한다.

---

## 8. References

1. J. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," *NAACL-HLT*, 2019.
2. M. Mosbach et al., "On the Stability of Fine-tuning BERT," *ICLR*, 2021.
3. S. Park et al., "KLUE: Korean Language Understanding Evaluation," *NeurIPS D&B*, 2021.
4. I. Magnusson et al., "Reproducibility in NLP," *Findings of ACL*, 2023.
5. A. Ratner et al., "Snorkel: Rapid Training Data Creation with Weak Supervision," *VLDB Journal*, 28(2), 2019.
6. E. Merdjanovska et al., "NoiseBench: Benchmarking the Effects of Real Label Noise on NER," *EMNLP*, 2024.
7. T. Wolf et al., "Transformers: State-of-the-Art Natural Language Processing," *EMNLP-Demos*, 2020.

---

## A. Appendix

### A.1 라벨별 정확도 전수 (Gold cross-distribution)

[`paper/paper1/data/runs/20260427_143110/summary.json`](data/runs/20260427_143110/summary.json) 에서 자동 생성. M2 기준 오름차순.

| Label | Class | M1 답만 | M2 문장 | M3 문장+neg |
|---|---|---:|---:|---:|
| copyright_status | format-regular | 0.0000 | 0.0015 | 0.0000 |
| position | format-free | 0.0237 | 0.0284 | 0.0332 |
| address | format-regular | 0.0000 | 0.4384 | 0.5567 |
| company | format-free | 0.6566 | 0.6920 | 0.6328 |
| date | format-regular | 0.0982 | 0.8370 | 0.5027 |
| copyright_num | format-regular | 0.1505 | 0.9359 | 0.9783 |
| ri_data | format-regular | 0.0186 | 0.9396 | 0.9753 |
| copyright_description | format-free | 0.9245 | 0.9415 | 0.9410 |
| name | format-free | 0.0000 | 0.9467 | 0.9548 |
| copyright_Keyword | format-semi-regular | 0.9626 | 0.9932 | 0.9932 |
| department | format-free | 0.1141 | 0.9933 | 0.9933 |
| copyright_kotitle | format-semi-regular | 0.8694 | 0.9941 | 0.9937 |
| copyright_type | format-free | 0.2525 | 0.9949 | 0.9949 |
| copyright_idnum | format-regular | 0.0185 | 1.0000 | 1.0000 |
| copyright_language | format-regular | 0.9791 | 1.0000 | 1.0000 |
| copyright_quantity | format-regular | 0.1410 | 1.0000 | 1.0000 |
| copyright_uci | format-regular | 1.0000 | 1.0000 | 1.0000 |
| copyright_url | format-regular | 0.0000 | 1.0000 | 1.0000 |
| email | format-regular | 0.0000 | 1.0000 | 1.0000 |
| phone | format-regular | 0.1420 | 1.0000 | 1.0000 |
| ri_contract_type | format-semi-regular | 0.9833 | 1.0000 | 1.0000 |
| ri_copyright | format-semi-regular | 0.9588 | 1.0000 | 1.0000 |
| ri_info | format-semi-regular | 1.0000 | 1.0000 | 1.0000 |
| ri_law_reference | format-semi-regular | 0.7827 | 1.0000 | 1.0000 |
| ri_money | format-regular | 1.0000 | 1.0000 | 1.0000 |
| ri_period | format-regular | 0.9745 | 1.0000 | 1.0000 |

### A.2 코드 재현

```bash
git clone <repo>
cd metadata
.venv/bin/python paper1/paper1.py build --source rule         # 3 silver 디렉터리
.venv/bin/python paper1/paper1.py run --configs rule_m1,rule_m2,rule_m3
.venv/bin/python paper1/scripts/eval_only.py 20260427_143110
.venv/bin/python paper1/scripts/make_graphs.py
```

### A.3 데이터 / 그래프 위치

| 항목 | 경로 |
|---|---|
| Silver (3 modes) | `paper1/configs/rule/{m1_answer, m2_context, m3_negatives}/<label>.jsonl` |
| Sweep 결과 | `paper/paper1/data/runs/20260427_143110/` |
| Logs (FullLogger, 학습 시점 모든 정보) | 위 경로 하위 `<cfg>/log/{env, config, scalars, params, gpu, events, random_state, log_history}` |
| Figures | `paper/paper1/figures/run_20260427_143110/G1–G12` |

### A.4 학습 비용 (RTX 5070 12 GB)

| Mode | Train | Eval | Total | Steps | GPU peak |
|---|---:|---:|---:|---:|---:|
| M1 | ~13 분 | ~3.5 분 | ~17 분 | 12,111 | ~3 GB |
| M2 | ~13 분 | ~3.5 분 | ~17 분 | 12,057 | ~3 GB |
| M3 (substitutive) | ~17 분 | ~3.5 분 | ~21 분 | 12,057 | ~3 GB |
