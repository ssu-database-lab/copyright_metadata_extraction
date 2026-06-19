# KCC 2026 Paper — 학습 입력 노이즈 구성에 따른 한국어 권리문서 NER 성능 분석

본 디렉토리는 KCC 2026 제출 논문 프로젝트를 `paper/` 저장소에서 분리해 옮긴 것이다.
주제는 BIO 학습 입력의 노이즈 구성(**M1 답만 / M2 문장 / M3 문장 + true negative**)이
KLUE-BERT NER 성능에 미치는 영향이다.

## 시작점 (먼저 볼 것)

- [STATUS.md](STATUS.md) — **협업자 핸드오프**: 지금 가능한 것 · PLANS 진행현황 · 블로커(다중 인코더 eval).
- [LABEL_TAXONOMY.md](LABEL_TAXONOMY.md) — 26라벨 Free/Regular/Semi 분류 (논문 Appendix-ready).
- [data/README.md](data/README.md) — 데이터 전수조사·출처 매핑.
- [PAPER_PLAN_AND_FINDINGS.md](PAPER_PLAN_AND_FINDINGS.md) · [ic_eeecs_short_paper_draft.md](ic_eeecs_short_paper_draft.md) — 기획·초안.

## 구성

| 경로 | 설명 | 옮긴 방식 |
|---|---|---|
| `paper1/` | 논문 프로젝트 본체 (코드 · configs · figures · `data/runs/` 학습 결과·체크포인트 58.5GB) | 복사 (원본은 `paper/` 저장소에 유지) |
| `paper_module/` | `paper1.py`가 import하는 공유 NER 학습/플롯 라이브러리 | 복사 (paper1·paper2가 공유하므로 이동이 아닌 복사) |
| `configs/integrated/silver/` | 빌드 소스 silver (26 라벨, ~469MB) | 복사·조립 (NER 17 ← `paper/configs/integrated/silver`(§24-14 v2), REGEX 9 ← `paper/configs/integrated/old/silver`). **git 제외**(.gitignore) |
| `configs/integrated/gold/` | OOD 평가 gold (36 라벨, 5.9MB) | 복사 (← `paper/configs/integrated/gold`). git 추적 |
| `paper1/configs/rule/{m1,m2,m3}/` | 빌드된 학습 입력 (26 라벨 × 3 mode) | 복사 (paper1/ 와 함께) |
| `data/` | 전수조사로 모은 NER/LLM 데이터 (다중 인코더 BIO·ground_truth·eval/benchmark 출력·예측·원문 코퍼스). 출처·git 정책은 [`data/README.md`](data/README.md) | 복사 (소형 추적 ~8MB, 대용량 원문·train txt 205MB 는 gitignore) |
| `제출용_...Klue-BERT 성능 분석.hwp` | 제출본 한글 문서 | 복사 |

원본 위치: `paper/old/cleanup_20260526/finished/paper1/` 및 `paper/old/cleanup_20260526/paper_module/`.
논문에 사용한 데이터(silver build-source·gold·빌드된 mode 데이터)는 모두 포함했고, 공유 데이터는 `paper/`·`metadata/` 의 원본을 복사했다.

## import 경로 · 실행 의존성

`paper1/paper1.py`는 다음 경로를 가정한다 (현재 구조에서 그대로 동작):

- `ROOT = kcc2026paper/` → `import paper_module ...`, 데이터는 `ROOT/configs/integrated/{silver,gold}` (위에서 복사 완료)
- `ROOT.parent/"metadata"` → `from module.parts.labels import ...` **및** `paper_module.core.ner.*` 가
  전이 의존하는 `module.extractor.ner.*` (= `copyright_metadata_extraction/metadata/module/`)

실행에 필요한 사항(분리 과정에서 정합):

- **Python 패키지**: `blingfire`, `kiwipiepy` (metadata NER 스택이 요구). `paper/.venv` 에 설치 완료.
- **`paper_module/core/ner/base.py` 셔임 패치**: 현행 metadata 가 `base.py` → `_runtime.py` 로 리팩터되어
  옛 surface(`DEFAULT_MODEL` 등)가 base 에서 빠졌으므로, `_runtime` 에서 재export 하도록 수정함
  (복사본만 수정, metadata/원본 paper 무손상).

## git 추적 참고

상위 `copyright_metadata_extraction` 저장소의 `.gitignore`에 의해 다음은 커밋되지 않는다(로컬에는 존재):

- 학습 가중치 `*.safetensors`, `*.bin` (`data/runs/*/model/`, ~59GB)
- figure `*.png`
- 제출본 `*.hwp`
- 빌드 소스 `configs/integrated/silver/` (~469MB; 디스크에는 존재, 위 표의 출처에서 재조립 가능)

따라서 커밋되는 것은 코드 · `configs/integrated/gold` · 빌드된 `paper1/configs/rule` · 결과 summary/csv/log 등 텍스트 산출물이다.
