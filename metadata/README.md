# Metadata Product Tree

`metadata/` 는 한국 공공저작물 권리문서에서 35-라벨 메타데이터를 추출하는
**프로덕트 파이프라인** 트리입니다. 연구 사료(논문 초안, sweep, 보관용 비교
결과)는 형제 디렉터리 `../paper/` 에 있습니다.

협업·핸드오프 컨벤션은 [AGENTS.md](AGENTS.md), 스키마 매핑(50-필드 xlsx 출력
규약)은 [NER_LLM_METADATA_CONNECTION.md](NER_LLM_METADATA_CONNECTION.md) 를
참고하세요.

## Quick Start

```bash
# Python 3.12 + CUDA 13.0 환경
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# end-to-end 실행 — OCR(Qwen3-VL) + regex + NER(xlm-roberta-base) + LLM placeholder
python main.py
```

`main.py` 는 `module.api.extract_metadata` 하나만 호출. 단일 진입점.

- 입력: `data/in/document/` 의 PDF/이미지 (txt 도 가능)
- OCR 결과: `data/out/ocr/result/*.txt`
- 메타데이터 (35-label JSON): `data/out/metadata/*_metadata.json`

OCR 캐시(`data/out/ocr/result`)가 입력 문서 수만큼 있으면 OCR 단계 자동 스킵.
강제 재실행: `FORCE_OCR=1 python main.py`.
`extract_metadata(input_text=...)` 또는 NER용 `input_path=...` 를 직접 넘기면 OCR을 건너뜁니다.
OCR 입력을 직접 지정할 때는 `in_path=...` 를 사용합니다.

## Pipeline

```
PDF/이미지/txt
     │
     ▼  Qwen3-VL-2B-Instruct (Apache-2.0, BF16, ~5GB VRAM)
OCR 평문 텍스트 (data/out/ocr/result/*.txt)
     │
     ├──▶ regex (9 strict-format 라벨)            module/extractor/regex.py
     ├──▶ NER  (17 free-form 라벨, xlm-roberta-base fine-tune) module/extractor/ner/
     ├──▶ post-process                            module/extractor/ner/postprocess.py::postprocess_metadata
     │     ├─ closed-vocab keyword 매칭 (ri_copyright, ri_info, copyright_type 등)
     │     ├─ form-cue line capture (성명:, 주소:, 전화번호:, 기관명: 등)
     │     ├─ length/한글 필터 (노이즈 컷)
     │     └─ ri_data ← date fallback
     └──▶ LLM (9 위임 라벨, 미구현 stub)            module/extractor/llm/llm.py
                       │
                       ▼  extract_metadata
            data/out/metadata/<rel>/<stem>_metadata.json
```

LLM 단계는 콜백 hook: `extract_metadata(..., llm_fn=<함수>)`. 시그니처
`(raw_text, partial_meta) -> {label: List[str]}`. `llm_fn=None` 이면 9 위임 라벨
모두 `["N/A"]` placeholder.

## Directory Layout

```
metadata/
├── main.py                       # 진입점 — extract_metadata 호출만
├── eval_compare.py               # (옛 실험) Regex/NER/Train 3-way 비교
├── eval/
│   └── (gold 라벨별 평가는 configs/integrated/gold/ 기준 — README 정확도 표 참조)
├── module/
│   ├── api.py                    # ★ extract_metadata public facade
│   ├── extractor/
│   │   ├── ocr/ocr.py            # Qwen3-VL-2B 래퍼 (PyMuPDF + transformers)
│   │   ├── regex.py              # 9-라벨 정규식 추출
│   │   ├── ner/                  # BERT 계열 token classification
│   │   │   ├── base.py           # 모델 경로/다운로드/라벨/디버그/predict 공통
│   │   │   ├── train.py          # ner_train()
│   │   │   ├── predict.py        # ner_predict() 오케스트레이터 + LLM callback hook
│   │   │   ├── token_cls.py      # TokenClassNER (HF Trainer + LoRA/full)
│   │   │   ├── postprocess.py    # closed-vocab/form-cue deterministic cleanup
│   │   │   └── full_logger.py    # step·layer 학습 로깅
│   │   └── llm/llm.py            # LLM 추출 (NotImplementedError stub)
│   └── parts/
│       ├── labels.py             # ★ 35-라벨 스키마 (REGEX/NER/LLM 3-way)
│       ├── paths.py              # project_root() / resolve_user_path()
│       ├── directory.py          # iter_document_files() 등
│       ├── io.py                 # ensure_outdir() / default_outfile()
│       ├── text.py               # 한국어 문장 분리·토큰화 (kiwipiepy + blingfire)
│       └── types.py              # Decision, Span dataclass
├── configs/
│   ├── labels.yaml               # OCR 모델 ID, NER threshold, 라벨 목록 (참조용)
│   └── integrated/
│       ├── gold/                 # 외부 gold 평가 데이터 (라벨별 jsonl)
│       └── silver/               # NER 학습 silver (654,377 BIO 레코드)
├── models/                       # 학습된 NER 어댑터 (7개 backbone)
│   └── <hf_id_with_dashes>/
│       ├── config.json / model.safetensors / tokenizer.* (베이스)
│       └── adapter/              # full fine-tune 산출물 (LoRA 변형도 지원)
├── model_downloaded/             # HuggingFace 다운로드 캐시
├── data/
│   ├── in/                       # 입력 문서 (PDF, 이미지, txt)
│   ├── out/
│   │   ├── ocr/result/           # OCR 평문 (NER 입력)
│   │   └── metadata/             # 최종 35-라벨 JSON (extract_metadata 출력)
│   └── validation/               # 수동 검수
├── debug/                        # 학습/추론 세션 로그 (gitignore 가능)
├── requirements.txt              # 의존성 (Python 3.12 기준)
├── requirements.lock.txt         # 핀 잠금 (선택)
├── keys.env                      # 외부 API 키 (gitignore 대상)
├── AGENTS.md                     # 협업 컨벤션
├── NER_LLM_METADATA_CONNECTION.md  # 35-라벨 → 50-필드 xlsx 매핑
└── README.md                     # 이 파일
```

## Schema — 35 라벨 (REGEX 9 + NER 17 + LLM 9)

`module/parts/labels.py` 가 source of truth. 세 집합 disjoint.

| Group | Count | Labels |
|---|---:|---|
| REGEX (`regex.py` PATTERNS) | 9 | `phone`, `email`, `copyright_url`, `copyright_uci`, `date`, `ri_money`, `copyright_num`, `copyright_idnum`, `copyright_quantity` |
| NER (`token_cls.py` BIO classifier) | 17 | copyright(6): `copyright_kotitle`, `copyright_status`, `copyright_description`, `copyright_Keyword`, `copyright_language`, `copyright_type`<br>author(5): `name`, `company`, `address`, `position`, `department`<br>rights(6): `ri_data`, `ri_period`, `ri_info`, `ri_contract_type`, `ri_copyright`, `ri_law_reference` |
| LLM-delegated (stub) | 9 | copyright(3): `copyright_id`, `copyright_Pname`, `copyright_con_status`<br>rights(6): `ri_cpcheck`, `ri_uncopyright`, `ri_workhire`, `ri_consent_type`, `ri_jch_conset`, `ri_portrait` |

xlsx 출력 시 7 role-free author 라벨이 21 role-prefixed 필드(`ch_co_*`,
`ch_ja_*`, `ch_nr_*`)로 확장 → 최종 50 필드. 자세한 매핑은
[NER_LLM_METADATA_CONNECTION.md](NER_LLM_METADATA_CONNECTION.md).

## Post-Process — line-capture 가 정확도의 핵심

NER 단독 결과는 한국 양식 (성명:, 주소:, 기관명:) 의 entity 를 자주 놓치고
경계 오류·1글자 노이즈가 많음. `postprocess_metadata` 가 3가지로 보정:

1. **closed-vocab 매칭** (`CLOSED_VOCAB`): ri_copyright, ri_contract_type, ri_info,
   copyright_type, copyright_status, copyright_language 는 정해진 키워드 검색.
   **vocab 우선, 없으면 NER 폴백** (기존엔 무조건 NER 대체 → 닫힌 목록에 없는
   정답을 전부 버려 gold 정확도가 급락했음). `copyright_status` 는 파일 확장자
   (`_extract_file_ext`)도 함께 인정.
2. **form-cue line-capture** (`FORM_CUE_PATTERNS`): "성명:", "주소:", "전화번호:",
   "기관명:" 등 cue 뒤 줄 끝까지 캡처. OCR 변형(콤마, %, 자릿수 잘림, `(주)` 누락)
   보존.
3. **gazetteer / lexicon 회수**: 지자체(`_extract_region_address`)→address,
   기관 접미사·접두어(`_extract_org_company`: …청/관/원/회/재단, 국립/한국/정부…)
   →company, 직위 lexicon + 크레딧(무대장치-이름)→position. NER 이 지자체·기관명을
   name 으로 오태깅하는 문제를 이름 정밀도 손실 없이 보정.
4. **휴리스틱 필터**: `name` 한글+2-20자, `address` 2-200자 + 부분문자열 중복 제거,
   `ri_period` 시간 cue 또는 날짜, `copyright_kotitle` 길이만(접미사 강제 폐지),
   `company` 한자 허용, `ri_data` ← date fallback.

증강 학습으로 모델 자체(raw)가 크게 개선됐고, 위 결정적 후처리가 남은 격차를 메워
gold 17/17 라벨 relaxed ≥0.90 을 달성한다.

## Models & Accuracy

### OCR — Qwen3-VL-2B-Instruct (Apache-2.0)

- 모델 ID: `Qwen/Qwen3-VL-2B-Instruct` (BF16, ~5GB VRAM)
- 12GB GPU (RTX 5070) 에서 안전, KV cache·image embedding 여유
- 한국어 직접 OCR (다국어 32+ 지원, Apache-2.0 상업 사용 OK)
- CPU 동작 차단 (`OCRDeviceError`) — production 속도 보장
- 설정: `configs/labels.yaml::ocr.qwen3vl` (model_id, render_zoom, max_new_tokens)
- 더 큰 GPU 면 `Qwen/Qwen3-VL-4B-Instruct` (BF16 ~9GB) 또는 8B BF16 권장

### NER — xlm-roberta-base 기본 (백본 토너먼트 채택, 2026-07)

`configs/integrated/silver/` (654,377 BIO 레코드) + 증강 silver(지자체→address,
"저작권자:<기관>"→company; 기존 silver 에서 파생, gold 누출 없음) 로 full fine-tune.

**백본 토너먼트 (6종, gold 라벨별 정확도 기준 — silver seqeval 아님).**
`mean_raw` = 후처리 없는 모델 자체 정확도, `mean_final` = regex+NER+후처리 제품 출력:

| HuggingFace ID | mean_raw | mean_final | name | address | company |
|---|---:|---:|---:|---:|---:|
| **FacebookAI/xlm-roberta-base (채택)** | 0.866 | 0.982 | 0.937 | **0.990** | 0.905 |
| klue/roberta-base | 0.866 | 0.980 | 0.893 | 0.941 | 0.803 |
| google-bert/bert-base-multilingual-cased | 0.862 | 0.979 | 0.920 | 0.951 | 0.840 |
| kakaobank/kf-deberta-base | 0.858 | 0.979 | 0.907 | 0.901 | 0.833 |
| jhu-clsp/mmBERT-base (2025) | 0.857 | 0.984 | 0.963 | 0.946 | 0.670 |
| microsoft/mdeberta-v3-base | 0.000 | 0.456 | — | — | — |

(mdeberta-v3-base 는 deberta-v2 sentencepiece + `is_split_into_words` word-align 파손으로
raw 0.0 — 이 토큰 정렬 파이프라인에 부적합.)

채택 근거: address(0.990 최고)·name 균형 최강, company 무난, 운영 안정(OOM 없음).
증강 학습으로 **address raw 0.60→0.99, company raw 0.60→0.74** — 규칙 의존을 벗고
모델 자체가 지자체=address / 기관=company 를 학습.

`main.py` 기본값: `configs/labels.yaml::ner.model_name = FacebookAI/xlm-roberta-base`
+ threshold 0.25. 가중치는 `models/FacebookAI--xlm-roberta-base/adapter/model.safetensors`
(full fine-tune). 다른 백본은 정리(제거)됨 — 필요 시 silver 로 재학습.

학습 속도: `LengthGroupedSampler`(길이 그룹핑) + `NER_MAX_LENGTH`(기본 512) env 로
동적 패딩 낭비 제거. silver 는 p50≈12·p99≈384 토큰이라 이 조합으로 학습 처리량
약 3배(3.6→11 it/s, batch 32). 대형 vocab/deberta 는 `NER_MAX_LENGTH=256` +
batch 축소 권장(disentangled attention O(n²) 메모리).

### 정확도 — gold 라벨별 실측 (`configs/integrated/gold/`)

라벨별 `{text, answer}` gold 로 예측을 1대1 대조. 지표는 relaxed(정답 엔티티가
예측에 포함) 기준. 채택된 xlm-roberta-base + 후처리 결과 **17/17 NER 라벨 ≥0.90**:

| 라벨 | relaxed | 라벨 | relaxed |
|---|---:|---|---:|
| name | 0.937 | copyright_status | 0.999 |
| address | 1.000 | copyright_type | 0.995 |
| company | 0.905 | copyright_language | 1.000 |
| department | 0.993 | copyright_kotitle | 0.956 |
| position | 0.986 | copyright_description | 0.995 |
| ri_data | 1.000 | copyright_Keyword | 0.974 |
| ri_period | 0.953 | ri_contract_type | 1.000 |
| ri_law_reference | 1.000 | ri_info | 1.000 |
| ri_copyright | 0.969 | | |

주의: 저장돼 있던 "eval_accuracy 0.98" 류는 **토큰 단위**(대부분 `O` 토큰)라
엔티티 정확도를 과대평가한다. 위 gold relaxed 가 실제 지표.

### LLM — 미구현

`module/extractor/llm/llm.py` 는 `NotImplementedError` stub. 9 위임 라벨은
`extract_metadata` 호출 시 `["N/A"]` placeholder. LLM 도입 시
`llm_fn=<함수>` 인자만 전달.

## Portability

- **Python**: 3.12 (3.11 호환 예상, 미테스트)
- **OS**: Linux. macOS/Windows 는 CUDA 13.0 PyTorch wheel 별도 인덱스 사용.
- **GPU**: CUDA 13.0 권장. OCR 은 CPU 대체 불가 (`OCRDeviceError`).
- **경로**: `module/parts/paths.py::project_root()` 가 `metadata/` 자동 감지 →
  사용자 홈 무관하게 동작. 모든 입출력은 프로젝트 루트 기준 상대 경로.
- **외부 API**: `keys.env::KCC_API_KEY` (공공데이터포털) 는 Gold 보강용으로만
  쓰이고 추출 파이프라인 실행에는 불필요.
- **OCR 첫 실행**: Qwen3-VL-2B (~5GB) HuggingFace 다운로드 필요. HF 익명 ~2-3
  MB/s. `HF_TOKEN` 환경변수 설정 시 속도 향상.

## Source Of Truth

| 항목 | 파일 |
|---|---|
| 35-라벨 스키마 (REGEX/NER/LLM 분할) | `module/parts/labels.py` |
| Regex 패턴 | `module/extractor/regex.py::PATTERNS` |
| OCR 모델·설정 | `configs/labels.yaml::ocr.qwen3vl` |
| NER threshold 기본값 | `module/extractor/ner/_runtime.py::DEFAULT_THRESHOLD` |
| 진입점 기본값 | `configs/labels.yaml::ner.model_name` (xlm-roberta-base + threshold 0.25) |
| Post-process 규칙 | `module/extractor/ner/postprocess.py::CLOSED_VOCAB`, `FORM_CUE_PATTERNS`, `postprocess_metadata` |
| 50-필드 xlsx 매핑 | `NER_LLM_METADATA_CONNECTION.md` |
| 정확도 평가 | `configs/integrated/gold/` 라벨별 gold (relaxed 지표) |

스키마/엔진/후처리 규칙이 바뀌면 `labels.py` → `api.py` → 본 README 순으로 함께
갱신.
