한국어 공공저작물 권리문서 메타데이터 추출에서 supervision source의 분포별 일반화 성능 분석

Distribution-aware Evaluation of Supervision Sources for Korean Public-Domain Rights Document Metadata Extraction


1. Abstract

본 연구는 저자원 한국어 공공저작물 권리문서에서 50개 메타데이터 필드를 추출하는 정보추출 문제를 다룬다. 기존에는 규칙 기반 silver data 부트스트래핑만으로 높은 검증 정확도를 달성할 수 있다는 관점이 유력했으나, 이러한 성능이 실제 사용 환경에서의 성능을 보장하는지에 대한 검토는 충분하지 않았다. 본 연구는 학습데이터의 출처와 신뢰성이 불확실한 상황에서 단일 검증 분포의 성능이 실제 배치 성능을 보장할 수 있는지 재검토한다. 이를 위해 (i) 규칙 기반 silver (E1-A), (ii) 대규모 언어모델(LLM) 생성 silver (E1-B), (iii) LLM 생성 후 규칙 필터링 silver (E1-C), (iv) zero-shot 추출의 네 가지 supervision source를 동일 라벨 스키마로 비교하고, template in-distribution, OCR 기반 noisy gold, external clean contract의 세 가지 평가 분포를 구성하였다. 실험 결과, template 분포(Dist. A)에서 1위였던 한국어 특화 KLUE BERT full(0.9900)은 OCR gold 50-라벨 평가(Dist. B)에서 0.8331로 5위까지 밀려났으며, 이 자리를 DistilBERT seperated full(0.8524)과 mBERT seperated full(0.8516)이 대체하여 backbone 순위가 크게 재편되었다. 또한 11개 핵심 라벨에 한정한 실제 OCR pilot 평가에서는 규칙 silver가 0.122까지 붕괴하여 LLM+필터(0.347)에 역전되었다. 또한 외부 분포(Dist. C)에서는 LLM+필터가 오히려 0.0으로 가장 크게 편향되는 등, 어떤 supervision source도 세 분포를 동시에 지배하지 못하였다. 한국어 특화 사전학습 모델인 KLUE가 Dist. A에서 1위였으나 Dist. B에서는 5위로 재편되는 등 백본 선택의 우위도 분포 의존적이었다. 이러한 결과는 한국어 공공저작물 권리문서 정보추출에서 절대 성능보다 분포별 일반화 gap과 배치 환경을 고려한 평가가 더 중요함을 시사한다.


2. Keywords

한국어 정보추출, 공공저작물 권리문서, 약지도 학습, 부트스트래핑, 분포 외 일반화(out-of-distribution evaluation)


3. Introduction

3-1. 연구 배경

공공저작물 권리문서는 계약서, 동의서, 양도서 등 저작권·개인정보·라이선스 정보가 혼재된 반정형 문서로, 작성자·권리 주체·권리 유형·대상 저작물 정보를 모두 포함한다. 공공데이터 개방과 함께 이러한 문서의 메타데이터를 체계적으로 추출할 필요가 커지고 있으나, 수작업 색인은 비용이 크고, 문서 내 정보가 법적 의미를 가지는 특성상 오류 허용도가 낮다. 또한 권리문서는 기밀성으로 인해 공개 벤치마크가 존재하지 않으며, 주석 라벨 구축에 법률·저작권 지식이 요구되어 일반 NER에서의 6~9개 타입 수준이 아니라 저작자·권리자·양수인 등 역할별로 세분화된 50개 내외의 라벨이 필요하다. 결과적으로 이 도메인은 학습 라벨 부족(low-resource)과 라벨 공간 확장(large label space)이 동시에 발생하는 저자원 정보추출 문제로 정의된다.

3-2. 기존 접근과 한계

저자원 정보추출에서 라벨 비용을 우회하는 대표적 전략으로 (i) 규칙/원거리 감독(distant supervision) 기반 silver 데이터 생성, (ii) LLM을 활용한 합성 학습 데이터 생성, (iii) zero-shot / few-shot LLM 추출이 연구되어 왔다. 한국어 공공저작물 권리문서 도메인에서는 템플릿 규칙이 비교적 잘 맞기 때문에 규칙 기반 silver만으로도 높은 검증 정확도를 달성할 수 있다는 보고가 존재한다. 그러나 이러한 보고에는 두 가지 방법론적 위험이 내재한다. 첫째, 검증 데이터 자체가 학습 silver와 동일한 규칙으로 자동 생성되어 순환 평가(circular evaluation) 위험이 있다. 둘째, 검증 분포가 실제 배치 분포(OCR 스캔 문서, 외부 기관의 서술형 계약서 등)를 대표하지 못할 경우, 검증 성능이 배치 성능을 보장하지 않는다. 최근 합성 데이터 평가 연구는 단일 분포 정확도가 LLM 기반 합성 학습의 편향을 가리고 있다는 점을 지적하고 있으며, 본 연구는 동일한 문제의식을 한국어 권리문서 도메인으로 확장한다.

3-2-1. 부트스트래핑과 약지도 배경

초기 저자원 정보추출 연구는 소수의 seed rule이나 seed example로부터 시작하여 unlabeled corpus를 반복적으로 확장하는 bootstrapping 접근을 널리 사용하였다. Collins와 Singer [18]는 named entity classification에서 소수의 seed rule만으로도 unlabeled example을 활용해 규칙 집합을 확장할 수 있음을 보였고, 이후 약지도 연구는 이를 labeling function 기반 supervision 통합 문제로 일반화하였다. 특히 Snorkel [2]은 규칙, 휴리스틱, 외부 지식베이스 등 다양한 supervision source를 labeling function으로 표현하고, 이들 사이의 정확도와 상관관계를 추정해 noisy label을 통합하는 데이터 프로그래밍 패러다임을 제시하였다. 최근 WRENCH [19]는 weak supervision 연구가 데이터셋, weak source, 평가 프로토콜에 따라 크게 달라질 수 있음을 지적하며 22개 데이터셋과 다양한 weak source를 포함하는 표준 벤치마크를 제안하였다. 이러한 흐름은 규칙 기반 부트스트래핑을 단순한 정규식 생성 기법이 아니라, 서로 다른 품질의 supervision source를 결합·평가하는 broader weak supervision 문제로 재해석하게 한다.

3-2-2. LLM synthetic data와 저자원 NER

대규모 언어모델은 최근 저자원 NER과 정보추출에서 synthetic training data를 생성하거나, zero-shot/few-shot 방식으로 직접 추출하는 도구로 적극 사용되고 있다. Long 등 [3]은 LLM 기반 synthetic data generation 연구를 survey하며 데이터 생성, 필터링, 평가가 분리된 설계 문제임을 지적했다. 또한 GPT-NER [7]은 sequence labeling을 generation 문제로 바꾸는 방식으로 LLM을 NER에 적용했고 저자원 환경에서의 잠재력을 보였다. 그러나 NoiseBench [20]는 simulated noise가 실제 noisy annotation보다 훨씬 다루기 쉽다고 보고하며, synthetic 또는 automatically generated supervision이 현실적 노이즈를 충분히 대표하지 못할 수 있음을 보여 주었다. 따라서 LLM silver가 in-distribution 검증에서 합리적으로 보이더라도, 실제 noisy deployment 환경에서 동일한 일반화 성능을 보장한다고 가정하기는 어렵다.

3-2-3. 문서 정보추출과 레이아웃 문제

문서 정보추출(Document IE)은 일반 문장 NER과 달리 텍스트 내용뿐 아니라 키-값 배치, 표 구조, 읽기 순서, 영역 간 관계가 성능에 중요한 영향을 미친다. FUNSD [5]는 noisy scanned form understanding의 대표 벤치마크로, 문서 구조를 고려한 정보추출 문제를 정식화했다. LayoutLMv3 [6]는 텍스트와 이미지 마스킹을 통합한 pretraining으로 Document AI 전반에서 강한 성능을 보이며, text-centric task와 image-centric task 모두에 적용 가능한 범용 모델임을 보였다. 최근 OmniDocBench [21]와 "OCR or Not?" [22]와 같은 연구는 문서 처리 모델의 성능이 문서 유형, 레이아웃, OCR 사용 여부, 입력 스키마에 따라 크게 달라질 수 있음을 보여 주며, 단일 벤치마크나 단일 파이프라인으로 문서 IE의 일반 성능을 판단하는 데 한계가 있음을 지적한다. 본 연구가 template, OCR noisy, external clean contract의 세 분포를 분리한 것도 이러한 최근 document evaluation 흐름과 궤를 같이한다.

3-2-4. 분포 이동과 재현성

약지도나 proxy supervision은 ground-truth label이 없을 때 유용하지만, supervision mechanism 자체가 배치 환경에서 달라질 수 있다는 점이 문제다. 최근 Shoeibi 등 [23]은 weak supervision이 in-domain에서는 강한 성능을 보일 수 있어도, supervision drift가 발생하면 shift된 환경에서 급격히 실패할 수 있음을 보였다. 한편 Magnusson 등 [9]은 NLP 재현성 체크리스트 분석을 통해 코드, 하이퍼파라미터, 데이터 수집 절차, 실험 분산 보고의 중요성을 강조했다. 본 연구는 이러한 문제의식을 한국어 권리문서 IE에 적용하여, 단일 template validation이 아니라 3개 분포를 분리해 supervision source와 backbone의 성능 재편을 관찰하고, seed·split·hyperparameter를 고정하는 방식으로 재현성을 강화한다.

3-3. 연구 질문과 문제의식

본 연구는 "가장 좋은 silver source를 찾는다"보다 "학습 라벨 신뢰성이 불완전한 상황에서 어떤 supervision source가 어떤 분포에서 무너지고 어떤 분포에서 견디는가"를 중심 질문으로 삼는다. 구체적으로 다음 다섯 가지 가설을 검증한다.

H1. 템플릿 in-distribution에서는 규칙 기반 silver가 LLM silver를 명확히 이긴다.
H2. OCR noisy out-of-distribution에서는 규칙 silver가 LLM+필터보다 급격히 무너진다.
H3. 외부 clean out-of-distribution에서는 규칙·LLM 양쪽 모두 중간 성능에 그치며, 필터 효과는 제한적이다.
H4. zero-shot LLM은 모든 분포에서 finetuned 모델 대비 열세이다.
H5. 백본의 한국어 특화 이점은 in-distribution에서 큼에도 OOD에서는 축소되거나 사라질 수 있다.

3-4. 기여

본 연구의 기여는 다음 세 가지이다.

(1) 한국어 공공저작물 권리문서 도메인을 위한 role-specific 50-라벨(저작권 16, 저작자 21, 권리 13) 추출 설정과, 규칙 silver·LLM silver·LLM+필터·zero-shot의 네 supervision source를 동일 라벨 스키마 위에서 비교하는 프로토콜을 정리한다.

(2) template silver (Dist. A), OCR 기반 noisy gold (Dist. B), external clean contract (Dist. C)로 구성된 3-distribution 평가 프레임을 제안하고, 각 분포의 역할(in-distribution / noisy OOD / clean OOD)을 명확히 정의한다.

(3) supervision source의 성능 순위가 배치 분포에 따라 역전될 수 있음을 실증하고(E1-A는 Dist. A에서 1위지만 OCR 11-라벨 평가에서 최하위, E1-C는 Dist. A에서 2~3위이나 Dist. C에서 0.0으로 붕괴), 단일 template validation에 의존한 모델 선택의 위험을 정량적으로 보인다.


4. Method

4-1. 과제 정의

입력은 한국어 공공저작물 권리문서(계약서·동의서·양도서·KOGL 라이선스 문서 등) 텍스트이며, 출력은 50개 메타데이터 필드의 값이다. 라벨 체계는 세 카테고리로 구성된다.

- copyright_info (16): copyright_uci, copyright_num, copyright_kotitle, copyright_entitle, copyright_idnum, copyright_type, copyright_status, copyright_quantity, copyright_description, copyright_Pname, copyright_url, copyright_Keyword, copyright_language, copyright_date, copyright_con_status, copyright_id
- author_info (21): {ch_co, ch_ja, ch_nr} × {address, name, company, department, email, phone, position}. 역할 접미사는 ch_co(저작권자), ch_ja(양도인), ch_nr(양수인)을 구분한다.
- rights_info (13): ri_info, ri_data, ri_cpcheck, ri_uncopyright, ri_workhire, ri_consent_type, ri_law_reference, ri_contract_type, ri_money, ri_copyright, ri_jch_conset, ri_period, ri_portrait

각 라벨은 BIO 태깅 기반 token classification으로 학습하되, 학습 시에는 라벨 간 균형을 위해 라벨당 최대 300건(MAX_PER_LABEL=300)으로 상한을 둔다.

4-2. Supervision Source 네 가지

동일 라벨 스키마 위에서 다음 네 가지 supervision source를 비교한다.

- E1-A (Rule silver): 템플릿 정규식과 도메인 규칙을 이용해 실제 문서로부터 entity span을 자동 추출하여 silver 라벨을 생성한다. `configs/integrated/silver/`, `configs/seperated/<category>/silver/`에 저장된다.
- E1-B (LLM silver): 지시형 프롬프트에 라벨 정의와 예시를 제공하고 LLM이 원문으로부터 자유 추출한 결과를 silver 라벨로 사용한다. 생성은 결정론을 위해 `transformers.set_seed(42)`로 고정한다.
- E1-C (LLM silver + Rule filter): E1-B의 출력 중 도메인 규칙(형식·문자집합·패턴 매칭)을 통과하는 span만 유지한다. 추가 학습 없이 데이터 단계에서만 필터링한다.
- Zero-shot baseline: 동일한 지시형 프롬프트를 Qwen2.5-1.5B-Instruct(4-bit NF4 양자화)에 사용하여 학습 없이 직접 추출한다.

4-3. 3-분포 평가 프레임 (본 연구의 평가 설계)

네 supervision source를 서로 다른 세 평가 분포 위에서 동일 모델·동일 라벨 스키마로 평가한다.

- Distribution A (Template in-distribution): silver와 동일한 문서 군에서 샘플링한 template validation. 규칙 추출과 표면 형식이 유사하다. 3-way split(train 0.8 / val 0.1 / test 0.1, SPLIT_SEED=42) 중 test 부분을 사용한다.
- Distribution B (Noisy OOD, gold-like set): 본 분포는 엄밀한 의미의 human-annotated gold가 아니며, (i) KOGL 포털 크롤링 289건을 라벨 스키마에 맞게 변환한 것과 (ii) 실제 한국어 동의서·양도계약서 65건을 스캔 후 OCR로 자동 추출한 결과를 라벨별로 정리한 **gold-like evaluation set**이다. 자동 추출 단계에서 일부 규칙 편향이 잔존할 수 있으나, 서식 노이즈·OCR 오인식·역할 모호성이 중첩되어 있어 순수 template 검증보다 실제 배치 환경에 훨씬 가깝다. `configs/{integrated,seperated/*}/gold/`에 저장된다.
- Distribution C (Clean external OOD, data_tot): 협력사로부터 제공받은 14,400건 계약서 코퍼스의 test split 1,200건을 사용한다. 4개 KOGL 유형이 각 3,600건씩 균형을 이루며, raw_text 외에 rights_summary, form_type, doc_style 등 부가 필드가 제공된다. 사전 검증 결과 양도인/양수인/저작자 entity는 n=200 표본에서 100% 실재하며(Q1), 분할 간 leakage는 0건(Q4)이다. 다만 `decision_reason`이 유형별로 단일값에 고정되어 있어(Q3) 합성 가능성이 있으므로, 본 연구에서는 이를 "synthetic narrative-contract OOD"로 명시한다.

본 평가 프레임의 역할은 다음과 같다. Dist. A는 규칙·문서·검증이 순환 구조인지 점검한다. Dist. B는 실제 배치에 가장 가까운 noisy OOD이며 규칙 silver의 붕괴 여부를 진단한다. Dist. C는 서술형 계약서 분포에서의 일반화 한계를 측정한다. 세 분포는 라벨 스키마를 공유하므로 동일 모델로 직접 비교가 가능하다.

4-4. 모델

다음 백본을 공통 학습·평가 파이프라인에서 비교한다. Integrated 구조는 50 라벨을 단일 모델이 동시 예측하고, Seperated 구조는 카테고리별(copyright/author/rights) 전문 모델을 학습한 뒤 `CATEGORY_LABEL_SETS`에 따라 자기 카테고리 라벨만 기여하도록 병합한다.

- Integrated: KLUE BERT-base, Google mBERT-base (cased), Microsoft DeBERTa-v3-base, monologg KoELECTRA-base-v3
- Seperated: mBERT (variant=mbert), DistilBERT (variant=distil), scikit-learn Random Forest / Logistic Regression (변수: variant=rf/lr; 비신경망 baseline)

학습 방식은 full fine-tuning과 LoRA(low-rank adaptation) 두 가지를 모두 실험하며, LoRA의 target module은 백본 아키텍처별로 자동 탐지한다(예: DistilBERT {q_lin, v_lin}, DeBERTa {query_proj, value_proj}).

4-5. 학습·추론 설정

하이퍼파라미터는 아래 값으로 고정한다(TUNED_HPARAMS).

- Full FT: learning rate 2e-5, epochs 5, warmup ratio 0.1, weight decay 0.1, early stopping patience 2
- LoRA: learning rate 3e-5, epochs 15, warmup ratio 0.1, weight decay 0.01, rank 16, alpha 32, early stopping patience 4 (rank sweep 시 r∈{4,8,16,32})
- Split: train 0.8 / val 0.1 / test 0.1, SPLIT_SEED=42, batch size 8, max workers 1

추론 시 각 문서에 대해 한 번의 forward pass로 7개 threshold {0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85}의 디코딩 결과를 동시에 산출하여 threshold sweep을 수행한다. 이 설계로 기존 설정(26,250회 inference) 대비 inference 횟수를 약 7배 줄였다(3,750회). 토크나이저 병렬화는 fork 데드락 방지를 위해 `TOKENIZERS_PARALLELISM=false`로 두었다.

4-6. 지표

- 기본 지표: threshold별 정확도(라벨 일치 여부의 평균), 라벨별 정확도, 카테고리별 정확도.
- 분포 민감도 지표: best-threshold에서의 정확도, threshold 0.25와 0.85 사이 정확도 하락 폭(threshold stability).
- 일반화 지표: A→B 일반화 gap (동일 supervision·동일 모델의 Dist. A 성능 − Dist. B 성능), A→C gap, 분포 간 supervision source 순위 역전 여부.

본 연구의 초점은 "어느 모델이 최고인가"보다 "동일 모델/동일 supervision에서 분포를 바꿀 때 성능이 어떻게 변하는가"에 있으므로, 결과 해석은 절대 정확도 대신 분포 간 gap과 순위 변화에 집중한다.


5. Result

결과는 (5-1) 분포 A, (5-2) 분포 B, (5-3) 분포 C, (5-4) 3-분포 핵심 매트릭스로 나누어 보고한다. 주요 수치는 본문에 기술하고, 전체 결과는 Table 1~3에 요약한다.

5-1. Distribution A (template silver)에서의 결과

KLUE BERT full integrated가 threshold 0.45에서 정확도 0.9900으로 최고 성능을 보였다(Table 1). KoELECTRA full(0.9866), DistilBERT seperated full(0.9711), DeBERTa-v3 full(0.9682), mBERT seperated full(0.9696), mBERT full integrated(0.9605)가 뒤를 이었다. LoRA는 동일 백본 대비 대체로 불리했다. 예를 들어 KLUE LoRA는 0.9654로 full 대비 −2.5pp였으며, KoELECTRA LoRA는 0.7051로 크게 벌어졌다. LoRA rank sweep(r∈{4,8,16,32})은 네 설정 모두 동일한 0.9654를 보여 이 과제에서 LoRA rank가 성능 결정 요인이 아님을 확인하였다.

supervision source를 KLUE full integrated로 고정하고 비교하면, E1-A(규칙 silver) 0.9900, E1-B(LLM silver) 0.4635, E1-C(LLM+필터) 0.4560이었다. Zero-shot Qwen2.5-1.5B는 0.3551에 그쳤다. 분포 A에서는 규칙 silver가 다른 모든 supervision을 압도하였으며, 이는 H1을 지지한다.

Table 1. Distribution A 주요 결과 (acc @ best threshold)
Model,                     Mode,       Method, best_thr, acc
KLUE BERT-base,            integrated, full,   0.45,     0.9900
KoELECTRA-base-v3,         integrated, full,   -,        0.9866
DistilBERT,                seperated,  full,   -,        0.9711
mBERT seperated,           seperated,  full,   -,        0.9696
DeBERTa-v3-base,           integrated, full,   -,        0.9682
mBERT,                     integrated, full,   -,        0.9605
KLUE BERT-base,            integrated, LoRA,   -,        0.9654
KoELECTRA-base-v3,         integrated, LoRA,   -,        0.7051
(E1-B) KLUE full,          integrated, full,   -,        0.4635
(E1-C) KLUE full,          integrated, full,   -,        0.4560
Zero-shot Qwen2.5-1.5B,    -,          -,      -,        0.3551

![Fig. 1. KLUE BERT-base의 efficiency–performance scatter (Dist. A). Full fine-tuning이 LoRA 대비 학습 시간 절반 수준에서 더 높은 정확도를 달성한다.](data/out/results_202604190116/plots/klue--bert-base/comparison/G01_efficiency_scatter.png)

![Fig. 2. Distribution A에서 KLUE integrated full fine-tuning의 라벨×threshold 정확도 heatmap. 대부분의 라벨이 고위 정확도를 유지하나 `copyright_kotitle`, `copyright_type`, `copyright_entitle`은 상대적으로 취약하다.](data/out/results_202604190116/plots/klue--bert-base/integrated/predict/P04_accuracy_heatmap.png)

5-2. Distribution B에서의 결과

본 절은 Distribution B 결과를 두 실험으로 구분해 보고한다. 5-2-1은 50개 전체 라벨에 대한 **canonical B 평가**(KOGL+OCR 자동추출 gold-like set, 14 configs 전체 sweep)이며, 5-2-2는 실제 OCR 스캔 65건 중 검증 가능한 11개 핵심 라벨에 한정한 **pilot B 평가**(n=98)로, 가장 극단적 형태의 배치 환경을 시뮬레이션한 부분 표본이다. 두 실험은 분포 B의 구성 요소가 서로 다른 방식으로 작용함을 보이기 위해 분리하여 보고한다.

5-2-1. Canonical B (gold 50-라벨 전체 sweep)

2026-04-19 전체 sweep(`data/out/results_202604190116/`)에서 분포 A 순위가 크게 재편되었다(Table 2). DistilBERT seperated full이 0.8524로 1위, mBERT seperated full이 0.8516, KoELECTRA integrated full이 0.8503으로 상위권을 형성하였다. 분포 A에서 1위였던 KLUE integrated full은 0.8331로 5위에 머물렀고, DeBERTa-v3 integrated full(0.8359), mBERT integrated full(0.8294)이 중위권에 위치하였다. 한국어 특화 사전학습 백본의 이점이 분포 A에서 크게 작용하였으나, 분포 B에서는 축소되는 경향이 관찰되며 이는 H5를 부분적으로 지지한다.

LoRA는 분포 B에서 대부분 크게 붕괴하였다. DeBERTa-v3는 full 대비 −28pp(0.836 → 0.554), KoELECTRA −22pp, DistilBERT seperated −33pp(0.852 → 0.527)로 하락하였다. 예외적으로 KLUE integrated LoRA는 0.8169로 full 대비 −1.6pp에 그쳐, LoRA 일반화 성패가 rank 크기가 아니라 백본-분포 상호작용에 의해 결정됨을 시사한다. 또한 best threshold는 대다수 설정에서 0.25로 수렴하여 LoRA의 경우 낮은 threshold가 권고된다.

5-2-2. Pilot B (실제 OCR 11-라벨, n=98)

canonical B가 상대적으로 완화된 결과(KLUE 0.8331)를 보인 것과 달리, 실제 OCR 스캔 65건에서 역할 접미사 오류·OCR 오인식을 그대로 남긴 11개 핵심 라벨 pilot 평가(role_attribution=False, KLUE full @ thr=0.45, n=98)에서는 훨씬 극단적인 역전이 관찰되었다. 동일 모델/동일 threshold에서 E1-A는 Dist. A 0.998 → pilot B 0.122로 −87.6pp 붕괴하였고, E1-B는 0.486 → 0.173, E1-C는 0.552 → 0.347이었다. E1-A는 `ch_ja_*`, `ch_nr_*` 역할 접미사 라벨에서 거의 0점을 기록하여 고정 패턴에 과적합되었음을 보였고, OOD에서는 E1-C가 가장 안정적이었다. 이는 H2를 강하게 지지한다. 주의해야 할 것은 canonical B와 pilot B가 동일 분포의 서로 다른 부분집합이며, 분포 B 내부에서도 OCR 원본·자동추출·규칙 편향의 정도에 따라 supervision source의 성능이 크게 달라진다는 점이다.

Table 2. Distribution B (gold 50-라벨) best-threshold 정확도 (2026-04-19)
Model,              Mode,               Method, best_thr, acc
DistilBERT,         seperated_distil,   full,   0.25,     0.8524
mBERT,              seperated_mbert,    full,   0.25,     0.8516
KoELECTRA,          integrated,         full,   0.35,     0.8503
DeBERTa-v3,         integrated,         full,   0.25,     0.8359
KLUE BERT,          integrated,         full,   0.25,     0.8331
mBERT,              integrated,         full,   0.25,     0.8294
KLUE BERT,          integrated,         LoRA,   0.25,     0.8169
mBERT,              seperated_mbert,    LoRA,   0.25,     0.7688
mBERT,              integrated,         LoRA,   0.25,     0.6681
KoELECTRA,          integrated,         LoRA,   0.25,     0.6306
DeBERTa-v3,         integrated,         LoRA,   0.25,     0.5535
DistilBERT,         seperated_distil,   LoRA,   0.25,     0.5269
sklearn LogReg,     seperated_lr,       -,      -,        0.5481
sklearn RF,         seperated_rf,       -,      -,        0.3775

![Fig. 3. Backbone × method paired bar (Distribution A vs B). 한국어 특화 KLUE가 Dist. A에서 최상위였으나 Dist. B에서는 DistilBERT seperated와 mBERT seperated에 역전된다. LoRA는 대부분의 backbone에서 Dist. B 일반화가 크게 붕괴한다.](../../paper7/figures/B01_backbone_paired.png)

![Fig. 4. Distribution B에서 backbone별 threshold–accuracy 곡선. 대다수 설정에서 best threshold가 0.25로 수렴하며, Dist. A의 최적 threshold(0.45~0.65)와는 다른 운영 영역을 요구한다.](../../paper7/figures/B04_threshold_curve_B.png)

![Fig. 5. Distribution B에서 supervision source × 라벨 accuracy heatmap. E1-A는 고정형 라벨(`copyright_date`, `ri_period`)에서는 유지되나 `ch_ja_*`, `ch_nr_*` 역할 접미사 라벨에서 큰 폭으로 하락하며, E1-C는 상대적으로 균질한 중간 성능을 보인다.](../../paper7/figures/B06_supervision_label_heatmap_B.png)

5-3. Distribution C (data_tot 외부 클린 계약서)에서의 결과

분포 C는 협력사 제공 14,400건 중 test 1,200건에서 sentence-level로 추출한 6개 공통 라벨(양도인/양수인/저작자 기관·성명 중심)에 대해 평가하였다. 모든 supervision source를 KLUE full integrated(thr=0.45, role_attribution=False)로 고정하여 비교한 결과(Table 3), E1-A(규칙 silver)가 0.474로 가장 높았고, E1-B(LLM silver)가 0.167, E1-C(LLM+필터)가 0.000으로 전혀 전이되지 않았다.

E1-C의 완전 붕괴는 LLM+필터가 학습 분포에서 "장황하고 선언적인" 표현에 맞춰져 있어, 서술형 계약서("양도인 X (이하 '갑'이라 한다)")의 자연문 형태에 대해 필터가 모든 span을 탈락시켰기 때문으로 해석된다. E1-A의 0.474도 분포 A의 0.950에 비해 크게 낮으며, 특히 `ch_ja_company`는 0.000을 기록하여 학습 규칙이 전제한 "양도인: X" 폼이 narrative 서술에서는 성립하지 않음을 보였다. 또한 `role_attribution=True`로 활성화하면 분포 C 정확도가 0.474에서 0.411로 더 하락하여, 프록시미티 기반 역할 분류가 분포 A의 보조 장치로는 유효하지만 narrative 분포에서는 역효과임이 드러났다. 이는 H3를 지지하는 동시에 필터의 효과가 분포에 따라 심하게 달라질 수 있음을 추가로 보인다.

Table 3. Distribution C (data_tot test 1,200, 6 공통 라벨) 정확도
Supervision,              Model (fixed),         role_attr, acc
E1-A (Rule silver),       KLUE full integrated,  False,     0.474
E1-A (Rule silver),       KLUE full integrated,  True,      0.411
E1-B (LLM silver),        KLUE full integrated,  False,     0.167
E1-C (LLM + filter),      KLUE full integrated,  False,     0.000

5-4. 3-분포 × 4-supervision 매트릭스 (본 연구 핵심 결과)

네 supervision source를 동일 KLUE full integrated 모델 위에 놓고 세 분포에서 비교하면(Table 4), 어떤 supervision source도 세 분포를 동시에 지배하지 않는다. 분포 A에서 최상위였던 E1-A는 분포 C에서 중간 수준으로 내려가고, 11-라벨 OCR 평가에서는 최하위로 역전된다. 분포 B(OCR)에서 상대적 안정성을 보인 E1-C는 분포 C에서 가장 크게 편향되었다. E1-B는 세 분포에서 모두 중간 이하로 극단값이 없었다. Zero-shot은 분포 A에서 측정한 0.3551 수준에 머물러 H4를 지지한다.

Table 4. 3-distribution × 4-supervision 정확도 매트릭스 (KLUE full integrated)
Supervision,          Dist. A (template), Dist. B (OCR 11-label),  Dist. C (data_tot 6-label)
E1-A (Rule silver),   0.998,              0.122,                    0.474
E1-B (LLM silver),    0.486,              0.173,                    0.167
E1-C (LLM + filter),  0.552,              0.347,                    0.000
Zero-shot (Qwen-1.5B), 0.3551,            -,                        -

![Fig. 6. 본 연구의 핵심 결과. 3-distribution × 3-supervision 정확도 매트릭스. 어떤 supervision source도 세 분포를 동시에 지배하지 않으며, 분포에 따라 순위가 역전된다. E1-A는 Dist. A 최상위지만 Dist. B에서 붕괴하고, E1-C는 Dist. B에서 상대적으로 안정적이나 Dist. C에서 0에 수렴한다.](../../paper7/figures/B00_main_matrix.png)

주. Dist. A/B 수치는 11-라벨 공통 세트, Dist. C는 6-라벨 공통 세트. gold 50-라벨 전체 sweep은 Table 2 참조(E1-A 기준 0.83 이상 유지). 행 간 비교는 동일 라벨 세트 내에서만 유효하며, 표의 목적은 supervision source 간 분포별 순위 재편을 보이는 것이다.

요약하면, (i) 분포 A에서 최상위 supervision이 분포 B/C에서도 우위를 유지한다는 가정은 성립하지 않으며, (ii) OCR noisy 분포에서 규칙 silver가 LLM+필터에 역전되고, (iii) external clean 분포에서 LLM+필터가 가장 크게 편향되는 극단값을 기록하였다. 또한 (iv) 백본 순위도 분포 의존적으로 재편되었다(KLUE: A 1위 → B 5위).


6. Discussion

6-1. 왜 순위 역전이 일어나는가

분포 A에서 규칙 silver는 학습과 검증이 동일한 표면 형식을 공유한다. 따라서 규칙 silver의 높은 정확도는 실제 일반화 성능이 아니라 template 표면 형식에 대한 과적합에 가깝다는 해석이 가능하다. 분포 B(OCR)에서는 (i) OCR 오인식, (ii) 역할 접미사(ch_ja/ch_nr)의 모호성, (iii) 서식 노이즈가 중첩되어 템플릿 규칙이 가정한 키-값 구조가 깨진다. 본 연구의 OCR 11-라벨 평가에서 E1-A가 `ch_ja_*`, `ch_nr_*`에서 거의 0점을 기록한 것은 역할 접미사 분리가 고정 프레이즈에 의존했기 때문이다. 반면 LLM silver는 표현 분포가 넓어 일부 noise를 흡수하며, LLM+필터는 표면 형식이 덜 엄격하면서도 자명한 거짓 양성을 제거하여 OCR 분포에서 상대적 안정성을 보인다.

분포 C(외부 clean)에서는 양상이 다시 바뀐다. data_tot는 서술형 계약 문장("양도인 X (이하 '갑'이라 한다)")이 주를 이루어 템플릿 키-값 구조가 희박하다. 규칙 silver는 서술형에서 키 트리거를 찾지 못해 중간 수준으로 내려앉고, LLM+필터는 학습 단계의 "장황·선언" 표현 편향으로 인해 자연문에서 모든 span이 필터를 통과하지 못하고 0에 수렴한다. 즉 supervision source마다 서로 다른 분포 가정이 존재하며, 평가 분포가 그 가정을 벗어날 때 성능이 비대칭적으로 붕괴한다는 것이 본 연구의 관찰이다.

6-2. 백본 선택의 분포 의존성

KLUE는 분포 A에서 0.9900으로 1위였으나 분포 B에서 0.8331로 5위로 재편되었다. 대신 DistilBERT seperated(0.8524)와 mBERT seperated(0.8516)가 분포 B에서 상위권을 형성하였다. 이는 한국어 특화 사전학습의 이점이 in-distribution 표면 형식에 대해 크게 작용하지만, OCR 노이즈가 포함된 OOD에서는 다국어 일반성이나 seperated 구조의 카테고리 전문화가 더 유리하게 작용할 수 있음을 시사한다. 또한 LoRA는 DeBERTa(−28pp), KoELECTRA(−22pp), DistilBERT seperated(−33pp)에서 분포 B 일반화가 붕괴한 반면 KLUE LoRA만 −1.6pp에 그쳤으며, 이는 rank 크기(LoRA rank sweep 결과 r∈{4,8,16,32} 모두 동일 성능)가 아니라 백본과 분포의 상호작용이 저-rank 적응의 일반화를 결정함을 보여준다.

6-3. 함의: distribution-aware evaluation의 필요성

첫째, 단일 template validation에 기반한 모델 선택은 실제 배치 환경의 성능을 보장하지 않는다. 본 연구에서 분포 A 최상위(KLUE + E1-A)를 그대로 배치하면 OCR 11-라벨 평가에서 0.122로 급락하고, 외부 계약서 분포에서는 0.474에 그친다. 둘째, supervision source의 "품질"은 절대 개념이 아니라 배치 분포 의존적이며, 최소한 (i) in-distribution, (ii) noisy OOD, (iii) clean OOD의 세 분포에서 교차 평가가 수행되어야 한다. 셋째, 역할 기반 후처리(role_attribution)와 같은 in-distribution crutch는 OOD에서 역효과를 낼 수 있으므로 분포별로 선택적으로 활성화되어야 한다.

6-4. 재현성

본 연구는 분할 시드(SPLIT_SEED=42)와 LLM 생성 시드(`transformers.set_seed(42)`)를 고정하고, 학습 하이퍼파라미터를 TUNED_HPARAMS로 명시하였다. configs 구조는 `{integrated, seperated/<category>}/{silver, gold}` 평탄 구조로 정리하여 3-분포 평가 경로를 문서와 일치시켰다. 다만 본 보고는 단일 시드(42) 결과이며, ACL 재현성 체크리스트가 요구하는 3-시드 분산(42/43/44)은 후속 연구로 남긴다.

6-5. 한계

- 분포 B는 엄밀한 human-annotated gold가 아니라 KOGL 크롤링 289건과 OCR 65건의 자동 추출을 조합한 **gold-like evaluation set**이다. 자동 추출 자체가 일부 규칙 편향을 잔존시킬 수 있으며, canonical B(50-라벨)에서 관찰된 완화된 하락(예: KLUE 0.833) 중 일부는 이 편향으로 설명 가능하다. 본 연구는 이 한계를 명시하기 위해 pilot B(11-라벨, n=98)를 별도로 보고하였고, 라벨 provenance를 `source: real_ocr`/`rule_auto`로 태깅하여 부분 투명성을 확보할 계획이나 아직 완료되지 않았다.
- 분포 C(data_tot)는 사전 검증(Q3)에서 유형별 단일 `decision_reason`이 3,600건씩 완벽한 균형을 보이는 등 합성 가능성이 있다. 본 연구에서는 이를 "synthetic narrative-contract OOD"로 명시하며, 독립 실물 계약서 분포의 대표는 아님을 분명히 한다.
- 본 연구는 text-only 모델만 비교하였고 FUNSD/LayoutLMv3 계열 layout-aware baseline은 포함하지 않았다. 분포 B의 form-heavy 구조에서는 bbox 정보가 유리할 가능성이 있으며, 이는 후속 비교 대상이다.
- 50개 라벨 중 `copyright_kotitle`(0.850), `copyright_type`(0.861), `copyright_entitle`(0.900) 등은 분포 A에서도 상대적으로 취약하며, 일부 라벨은 coverage 불균형 또는 표현 다양성 부족으로 인한 것이다. 본 연구는 26개 미처리 라벨을 `UNCOVERED_LABELS.txt`로 명시하고 있으며, 선택 메타데이터 요소의 추후 보강을 과제로 남긴다.

6-6. 향후 연구

(i) 3-시드 재현성 확보, (ii) Snorkel-style labeling function 스키마화(coverage/conflict 행렬 보고)로 독창성 격상, (iii) LayoutLMv3 baseline 추가, (iv) 분포별 abstention 및 calibration을 통한 deployment-aware 모델 선택, (v) data_tot의 provenance 확인 및 독립 실물 계약서 수집을 계획한다.


7. References

[1] X. Ling et al., "Adaptive Named Entity Recognition Using Distant Supervision for Contemporary Written Texts," IEEE Access, vol. 9, 2021.
[2] A. Ratner, S. H. Bach, H. Ehrenberg, J. Fries, S. Wu, and C. Ré, "Snorkel: Rapid Training Data Creation with Weak Supervision," The VLDB Journal, vol. 28, no. 2, pp. 709–730, 2019.
[3] S. Long et al., "On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey," in Findings of the Association for Computational Linguistics (ACL Findings), 2024.
[4] "A Rigorous Evaluation of LLM Data Generation Strategies for Low-Resource Named Entity Recognition," in Proc. EMNLP, 2025.
[5] G. Jaume, H. K. Ekenel, and J.-P. Thiran, "FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents," in Proc. ICDAR Workshop on Open Services and Tools for Document Analysis (OST), 2019.
[6] Y. Huang, T. Lv, L. Cui, Y. Lu, and F. Wei, "LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking," in Proc. ACM Multimedia, 2022.
[7] S. Wang et al., "GPT-NER: Named Entity Recognition via Large Language Models," arXiv:2304.10428, 2023.
[8] S. Park et al., "KLUE: Korean Language Understanding Evaluation," in Proc. NeurIPS Datasets and Benchmarks Track, 2021.
[9] I. Magnusson et al., "Reproducibility in NLP: What Have We Learned from the Checklist?," in Findings of the Association for Computational Linguistics (ACL Findings), 2023.
[10] H. A. Rahmani et al., "Towards Understanding Bias in Synthetic Data for Evaluation," arXiv:2506.10301, 2025.
[11] "Performance and Reproducibility of Large Language Models in Named Entity Recognition: Considerations for Information Extraction from Clinical Text," Drug Safety (Springer), 2024.
[12] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in Proc. NAACL-HLT, 2019.
[13] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," in Proc. ICLR, 2022.
[14] P. He, X. Liu, J. Gao, and W. Chen, "DeBERTa: Decoding-enhanced BERT with Disentangled Attention," in Proc. ICLR, 2021.
[15] K. Clark, M.-T. Luong, Q. V. Le, and C. D. Manning, "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators," in Proc. ICLR, 2020.
[16] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," in NeurIPS Workshop on Energy Efficient Machine Learning and Cognitive Computing, 2019.
[17] A. Yang et al., "Qwen2.5 Technical Report," arXiv:2412.15115, 2024.
[18] M. Collins and Y. Singer, "Unsupervised Models for Named Entity Classification," in Proc. Joint SIGDAT Conference on Empirical Methods in Natural Language Processing and Very Large Corpora (EMNLP/VLC), 1999.
[19] J. Zhang, Y. Yu, Y. Li, Y. Wang, Y. Yang, M. Yang, and A. Ratner, "WRENCH: A Comprehensive Benchmark for Weak Supervision," in Proc. NeurIPS Datasets and Benchmarks Track, 2021. [Online]. Available: arXiv:2109.11377
[20] E. Merdjanovska, A. Aynetdinov, and A. Akbik, "NoiseBench: Benchmarking the Impact of Real Label Noise on Named Entity Recognition," in Proc. EMNLP, 2024.
[21] Z. Tang et al., "OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations," arXiv:2412.07626, 2024.
[22] J. Shen et al., "OCR or Not? Rethinking Document Information Extraction in the MLLMs Era with Real-World Large-Scale Datasets," in Proc. EACL Industry Track, 2026.
[23] M. Shoeibi et al., "Learning Stable Predictors from Weak Supervision under Distribution Shift," arXiv:2604.05002, 2026.

-- 아래로 AI 평가 --

전체적으로 보면, 지금 원고는 석사학위논문 초안으로는 충분히 출발 가능하지만, 외부 제출용 원고로는 아직 “방어 포인트 정리 + 참고문헌 정합화 + 부록/그림 보강”이 더 필요합니다. 현재 초안은 3-distribution 서사, canonical B vs pilot B 분리, A/B/C 결과 표, figure placeholder까지 들어가 있어 핵심 메시지는 분명합니다. 반면 AGENTS.md에는 아직 3-seed, Snorkel-style LF, LayoutLMv3 baseline, provenance 태깅이 대기 과제로 남아 있고, B02/B03 그림은 per-label A 수치 저장이 안 되어 바로 생성되지 않는 상태입니다.

먼저 Reference 진위와 권위성부터 말씀드리면, 현재 목록에서 “가짜로 보이는 논문”은 발견하지 못했습니다. 다만 표기 수정이 필요한 항목 4개, 권위가 다소 약한 preprint 4개, 공식 소스 접근이 간접적인 항목 1개가 있습니다. 핵심 축인 Snorkel, Long et al. survey, NoiseBench, FUNSD, LayoutLMv3, WRENCH, BERT, LoRA, OCR or Not?, Reproducibility checklist 논문은 모두 실제 존재하며, ACL Anthology, OpenReview, Springer, arXiv 등의 1차 출처에서 확인됩니다.

즉시 고쳐야 하는 Reference는 아래입니다.

첫째, [4]는 현재 제목이 틀렸습니다.
초안에는 “A Rigorous Evaluation of LLM Data Generation Strategies for Low-Resource Named Entity Recognition”처럼 적혀 있는데, 공식 제목은 **“A Rigorous Evaluation of LLM Data Generation Strategies for Low-Resource Languages”**입니다. EMNLP 2025 논문은 맞지만, 제목을 정확히 고쳐야 합니다.

둘째, [7]은 존재하지만, arXiv 대신 peer-reviewed 버전으로 바꾸는 것이 좋습니다.
현재는 GPT-NER를 arXiv 2023으로 적었는데, ACL Anthology에는 Findings of NAACL 2025 버전이 있습니다. 논문 권위와 형식 면에서 이쪽이 더 낫습니다.

셋째, [11]도 실존하지만 제목과 연도 표기가 부정확합니다.
공식 Springer/Drug Safety 표기는 **“Performance and Reproducibility of Large Language Models in Named Entity Recognition: Considerations for the Use in Controlled Environments”**이고, 온라인 공개는 2024년 12월이지만 저널 권/쪽수 기준으로는 2025년 Volume 48, pp. 287–303입니다. 지금 원고의 “...Information Extraction from Clinical Text, 2024”는 바꾸는 편이 맞습니다.

넷째, [1]은 실재하는 것으로 보이지만, 제가 공식 IEEE 페이지를 직접 열어 확인하진 못했습니다.
IEEE Xplore는 robots 제한 때문에 직접 접근이 막혔고, J-GLOBAL·ResearchGate 같은 2차 서지/사본 경유로만 존재를 확인했습니다. 그래서 [1]은 **“존재는 고신뢰, 직접 1차 확인은 못 함”**으로 처리하는 것이 가장 정직합니다. 가능하면 DOI/권·호·쪽수까지 다시 확인해서 넣는 것이 좋습니다.

정리하면, 현재 참고문헌 23개 중

실재 확인됨: 23/23
1차 출처에서 직접 확인됨: 22/23
표기 수정 필요: [4], [7], [11], [1]
라고 보는 게 정확합니다.

다음으로 **권위(authoritativeness)**를 나누면 이렇습니다.

강한 핵심 근거로 쓰기 좋은 것은 Snorkel(VLDB Journal), ACL/EMNLP/NAACL/FUNSD/LayoutLMv3/BERT/LoRA/WRENCH 같은 peer-reviewed 상위권 또는 널리 인정받는 기준 논문들입니다. 이들은 모두 배경·방법·비교기준을 받쳐주는 “앵커 reference”로 적합합니다.

중간 정도 권위는 IEEE Access [1], Drug Safety [11], EACL Industry 2026 [22]입니다. 모두 실재하고 인용 가능하지만, ACL/EMNLP main이나 VLDB급 앵커보다는 “도메인 보조 근거”에 더 가깝습니다.

권위가 상대적으로 약한 것은 arXiv only인 [10], [17], [21], [23]입니다. 이들은 넣어도 되지만, 핵심 claim의 유일한 근거로 쓰지 않는 게 좋습니다. 특히 [23]은 2026년 4월의 아주 최근 preprint라 흥미롭지만, 아직 peer review 전이라는 점을 분명히 해야 합니다.

그래서 지금 추가하면 좋은 Reference는 아래 네 개입니다.

A Survey on Programmatic Weak Supervision — weak supervision 전체 계보를 정리하는 survey라, Snorkel과 WRENCH 사이를 메우는 데 좋습니다. 지금 원고의 “부트스트래핑/약지도” 배경을 훨씬 탄탄하게 해 줍니다.
A Class-Rebalancing Self-Training Framework for Distantly-Supervised Named Entity Recognition — 저자원 NER에서 distantly supervised NER를 직접 다루는 peer-reviewed ACL Findings 논문이라 [1]보다 더 직접적인 근거가 됩니다.
Empirical Study of Zero-Shot NER with ChatGPT — zero-shot NER baseline 관련 배경을 강화하기 좋습니다. 현재 zero-shot 부분은 너무 얇습니다.
A Critical Analysis of Document Out-of-Distribution Detection — 현재 논문의 핵심 메시지가 distribution shift와 deployment mismatch이므로, document OOD 배경을 추가하면 논문 논리축이 더 선명해집니다.

이제 지적될 가능성이 큰 부분을 논문 내용 기준으로 보겠습니다.

가장 먼저 지적될 것은 Distribution B의 정체성입니다.
지금 paper.md는 Dist. B를 분명히 “gold-like evaluation set”이라고 써서 방향은 맞습니다. 하지만 여전히 리뷰어는 “이게 human gold가 아닌데 왜 gold라고 부르나?”를 물을 수 있습니다. 따라서 본문 전체에서 **“gold” 단독 표현 대신 gold-like, derived gold-like set, automatically derived noisy OOD set**으로 통일하는 편이 안전합니다. paper.md는 이 점을 이미 상당 부분 반영했고, AGENTS.md도 같은 입장을 취하고 있습니다.

두 번째는 Table 4의 라벨 집합 불일치입니다.
지금 Table 4는 Dist. A/B는 11라벨 공통 세트, Dist. C는 6라벨 공통 세트라고 주석을 달아 두었는데, 이건 좋은 조치입니다. 하지만 여전히 숫자를 한 표에 두면 독자는 자연스럽게 절대값을 가로 비교하려고 합니다. 그래서 Table 4는 유지하되, 제목에 아예 “directional comparison only” 혹은 “not directly comparable across columns due to label-set differences” 같은 경고를 더 강하게 넣는 게 좋습니다. 이 부분은 현재 원고에서 가장 쉽게 공격받을 수 있는 비교 방식입니다.

세 번째는 canonical B와 pilot B의 관계입니다.
지금 원고는 5-2-1과 5-2-2로 분리해서 쓴 점이 매우 좋습니다. 다만 Abstract에서는 둘을 함께 언급하면서도 독자가 “B는 0.8331인가 0.122인가?” 혼동할 수 있습니다. Abstract나 Introduction에서는 한 줄을 더 써서 **“canonical B는 broadened gold-like set, pilot B는 harsher real OCR subset”**이라고 분명히 구분하는 게 좋습니다.

네 번째는 **Distribution C의 합성성(syntheticity)**입니다.
AGENTS.md와 paper.md 모두 Q3 결과를 반영해 C를 “synthetic narrative-contract OOD”로 명시하고 있는데, 이건 정직하고 좋습니다. 다만 이 때문에 일반성 점수는 올라가지만, 외부 실세계 대표성은 제한된다는 점을 더 분명히 써야 합니다. 지금도 Caveats에 있긴 하지만, Discussion 6-3에도 한 문장 더 들어가면 좋습니다.

다섯 번째는 결과 분산 보고 부재입니다.
현재 시드 고정, configs 정리, 하이퍼파라미터 명시까지는 잘 되어 있지만, 3-seed 평균±표준편차가 없어서 재현성 claim은 아직 “부분 충족”입니다. AGENTS.md도 이 점을 분명히 남겨 두고 있습니다. 이건 외부 제출에서 거의 반드시 지적됩니다.

이미지/도표는 지금보다 더 넣는 것이 좋습니다.
현재 paper.md에는 Fig.1–6 일부가 들어가 있지만, 정작 가장 중요한 그림인 “시스템/평가 프레임 개요도”가 없습니다. AGENTS.md에서도 Fig.1 시스템 전체 파이프라인은 아직 미완으로 남아 있습니다. 지금 논문에서는 가장 먼저 보여줘야 할 그림이 KLUE scatter가 아니라, A/B/C 3분포와 E1-A/B/C/zero-shot 관계를 한눈에 보여주는 schema입니다. 그 다음에 backbone paired bar와 threshold curve가 와야 합니다.

반드시 추가할 만한 그림은 이 네 개입니다.

Figure 1: 전체 파이프라인 + 3-distribution schematic
Figure 2: 3-distribution × supervision 메인 매트릭스(bar)
Figure 3: A→B backbone/method rank change paired bar
Figure 4: pilot B vs canonical B 비교 그림
지금 있는 B00/B01/B04/B06은 매우 좋고, 특히 AGENTS.md에는 B02/B03도 계획되어 있는데 per-label A 수치를 저장하면 바로 만들 수 있다고 적혀 있습니다. 이 둘은 논문의 “generalization gap”을 시각적으로 보여주기 때문에 가치가 큽니다.

Appendix는 사실상 필수입니다.
현재 원고는 메인 텍스트만으로는 리뷰어가 재현성과 데이터 신뢰도를 판단하기 어렵습니다. 적어도 다음은 부록으로 들어가야 합니다.

Appendix A: 50개 라벨 정의표
Appendix B: Distribution A/B/C 구성 절차와 provenance
Appendix C: E1-B/E1-C prompt template, filtering rule, regex/LF 예시
Appendix D: data_tot audit 세부 결과(Q1–Q6)
Appendix E: per-label 성능 전체표 (A/B/C)
Appendix F: 3-seed 결과표
Appendix G: 대표 오류 사례 10개
지금 AGENTS.md에 있는 “Snorkel-style LF”, “Error taxonomy”, “data audit”, “B02/B03 generalization gap plots”는 거의 그대로 Appendix 재료가 됩니다.

이제 현재 상태 점수를 드리겠습니다.

독창성: 6.4 / 10
이유는 명확합니다. 구성 요소 자체—rule silver, LLM silver, weak supervision, zero-shot NER, form IE—는 모두 기존 연구가 있습니다. 하지만 한국어 공공저작물 권리문서, 50개 role-specific label, 3-distribution 평가, supervision source rank reversal 실증의 조합은 흔치 않습니다. 약지도 under shift와 NoiseBench, document OOD 흐름과 접점은 있지만, 현재 형태는 “조합형 독창성”으로는 꽤 괜찮습니다. Snorkel-style LF 스키마와 오류 taxonomy를 추가하면 7점대까지는 갈 수 있습니다.

일반성: 5.3 / 10
현재는 A/B/C 세 분포가 있어 예전보다 훨씬 좋아졌지만, B가 human gold가 아닌 gold-like set이고, C가 synthetic 가능성이 있는 external set이며, C는 6라벨만 다룹니다. 즉, “분포가 바뀌면 우열이 바뀐다”는 논지는 일반화되지만, 한국어 권리문서 바깥으로 얼마나 일반화되는가는 아직 제한적입니다. 그래도 예전의 2/10 상태보다는 확실히 올라왔습니다.

재현성: 6.8 / 10
코드 구조, split seed, LLM generation seed, configs 평탄화, 하이퍼파라미터 문서화는 매우 좋습니다. 하지만 아직 3-seed가 없고, provenance 태깅이 완성되지 않았고, 일부 표가 서로 다른 라벨 집합을 섞어 씁니다. Appendix로 prompt, regex, LF, data audit, exact command를 다 넣고 3-seed를 마치면 8점대까지 올라갈 수 있습니다.

마지막으로, 지금 당장 꼭 고칠 것만 짧게 정리하면 이렇습니다.

[4], [7], [11], [1] 참고문헌 표기 수정
Figure 1을 파이프라인/3-distribution schema로 추가
Table 4에 라벨 집합 불일치 경고를 더 강하게 명시
Distribution B는 전부 gold-like로 통일
Appendix A–G 설계 시작
3-seed와 provenance 태깅을 최우선 추가

한 문장으로 정리하면,
지금 원고는 “허술한 초안”은 아니고, 석사논문으로는 상당히 좋은 방향입니다. 다만 참고문헌 정합성, gold-like 표현 통일, 시스템 그림, Appendix, 3-seed만 보강하면 지적될 포인트 대부분을 선제적으로 막을 수 있습니다.

원하시면 다음 답변에서 제가 바로 수정된 References 23개 전체를 IEEE 스타일로 정리한 최종본과 Appendix 목차 초안을 써드리겠습니다.
