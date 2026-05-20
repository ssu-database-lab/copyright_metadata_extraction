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

# end-to-end 실행 — OCR(Qwen3-VL) + regex + NER(mBERT) + LLM placeholder
python main.py
```

`main.py` 는 `module.api.extract_metadata` 하나만 호출. 단일 진입점.

- 입력: `data/in/document/` 의 PDF/이미지 (txt 도 가능)
- OCR 결과: `data/out/ocr/result/*.txt`
- 메타데이터 (35-label JSON): `data/out/metadata/*_metadata.json`

OCR 캐시(`data/out/ocr/result`)가 입력 문서 수만큼 있으면 OCR 단계 자동 스킵.
강제 재실행: `FORCE_OCR=1 python main.py`.

## Pipeline

```
PDF/이미지/txt
     │
     ▼  Qwen3-VL-2B-Instruct (Apache-2.0, BF16, ~5GB VRAM)
OCR 평문 텍스트 (data/out/ocr/result/*.txt)
     │
     ├──▶ regex (9 strict-format 라벨)            module/extractor/regex.py
     ├──▶ NER  (17 free-form 라벨, mBERT fine-tune) module/extractor/ner/
     ├──▶ post-process                            module/api.py::postprocess_metadata
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
│   └── eval_audit.py             # line-cue ground truth 기반 전수 recall 평가
├── module/
│   ├── api.py                    # ★ extract_metadata + postprocess_metadata
│   ├── extractor/
│   │   ├── ocr/ocr.py            # Qwen3-VL-2B 래퍼 (PyMuPDF + transformers)
│   │   ├── regex.py              # 9-라벨 정규식 추출
│   │   ├── ner/                  # BERT 계열 token classification
│   │   │   ├── base.py           # 모델 경로/다운로드/라벨/디버그/predict 공통
│   │   │   ├── train.py          # ner_train()
│   │   │   ├── predict.py        # ner_predict() 오케스트레이터
│   │   │   ├── token_cls.py      # TokenClassNER (HF Trainer + LoRA/full)
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
│       └── silver/               # NER 학습 silver (267k BIO 레코드)
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

1. **closed-vocab 매칭** (`CLOSED_VOCAB`): NER 가 노이즈 심한 라벨
   (ri_copyright, ri_contract_type, ri_info, copyright_type, copyright_status,
   copyright_language) 은 정해진 키워드 직접 검색 → NER 대체
2. **form-cue line-capture** (`FORM_CUE_PATTERNS`): "성명:", "주소:",
   "전화번호:", "기관명:" 등 cue 뒤 줄 끝까지 그대로 캡처. OCR 변형
   (콤마 separator, % 문자, 자릿수 잘림, `(주)` paren 누락 등) 도 전부 보존
3. **휴리스틱 필터**: `name` 한글 1자 이상 + 2-20자, `address` 2-200자,
   `ri_period` 시간 cue 필수, `ri_data` ← date fallback 등

이 후처리가 100% recall 의 결정적 요인.

## Models & Accuracy

### OCR — Qwen3-VL-2B-Instruct (Apache-2.0)

- 모델 ID: `Qwen/Qwen3-VL-2B-Instruct` (BF16, ~5GB VRAM)
- 12GB GPU (RTX 5070) 에서 안전, KV cache·image embedding 여유
- 한국어 직접 OCR (다국어 32+ 지원, Apache-2.0 상업 사용 OK)
- CPU 동작 차단 (`OCRDeviceError`) — production 속도 보장
- 설정: `configs/labels.yaml::ocr.qwen3vl` (model_id, render_zoom, max_new_tokens)
- 더 큰 GPU 면 `Qwen/Qwen3-VL-4B-Instruct` (BF16 ~9GB) 또는 8B BF16 권장

### NER — 7개 backbone 학습 완료, mBERT 기본

`configs/integrated/silver/` (267,307 BIO 레코드, 26 만 + 노이즈) 로 fine-tune.
silver validation split (seed=42) best-epoch:

| Alias | HuggingFace ID | eval_accuracy | eval_F1 | 학습 |
|---|---|---:|---:|:---:|
| **mbert (기본)** | `google-bert/bert-base-multilingual-cased` | **0.9858** | **0.9506** | ✓ |
| klue | `klue/bert-base` | ~0.985 | ~0.95 | ✓ |
| koelectra | `monologg/koelectra-base-v3-discriminator` | 0.9845 | 0.9487 | ✓ |
| distilbert | `distilbert-base-multilingual-cased` | 0.9859 | 0.9547 | ✓ |
| xlmr-base | `FacebookAI/xlm-roberta-base` | 0.9861 | 0.9502 | ✓ |
| xlmr-large | `FacebookAI/xlm-roberta-large` | — | — | ✓ (어댑터만) |
| deberta | `microsoft/deberta-v3-base` | 0.9859 | 0.9608 | ✓ |

`main.py` 기본값: mBERT + threshold 0.25. 학습된 가중치는
`models/<id>/adapter/model.safetensors` (full fine-tune, 709MB).

### 실측 recall — 63 문서 전수 audit (`eval/eval_audit.py`)

line-cue ground truth (성명:, 주소:, 전화번호:, 기관명: 등) 기반 전수 측정:

| 라벨 | gt | match | recall |
|---|---:|---:|---:|
| address | 98 | 98 | **100%** |
| company | 40 | 40 | **100%** |
| date | 38 | 38 | **100%** |
| name | 113 | 113 | **100%** |
| phone | 309 | 309 | **100%** |
| ri_money | 2 | 2 | **100%** |
| **OVERALL** | 600 | 600 | **100%** |

모든 63개 doc 100%. 측정 가능한 라벨은 phone/email/date/ri_money/address/name/
company (line-cue 가 명확). description/status/info 등은 cue 가 없어 자동 측정
불가 — 수동 검수 영역.

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
| NER threshold 기본값 | `module/extractor/ner/base.py::DEFAULT_THRESHOLD` |
| 진입점 기본값 | `main.py` (mBERT + threshold 0.25) |
| Post-process 규칙 | `module/api.py::CLOSED_VOCAB`, `FORM_CUE_PATTERNS`, `postprocess_metadata` |
| 50-필드 xlsx 매핑 | `NER_LLM_METADATA_CONNECTION.md` |
| Recall 평가 | `eval/eval_audit.py` |

스키마/엔진/후처리 규칙이 바뀌면 `labels.py` → `api.py` → 본 README 순으로 함께
갱신.
