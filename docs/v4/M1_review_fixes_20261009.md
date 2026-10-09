# M1 리뷰 반영 노트 — 평가기 스택 수정 (fixer, 2026-10-09)

패키지 `api/module/evaluation/v4/` · 규격 `M1_evaluator_spec.md` · 테스트 `venv/bin/python -m pytest api/module/evaluation/v4/tests -q`
리뷰 11건(BUG 1 · RISK 5 · NIT 5) 중 코드 수정 9건, 문서 기록 1건(τ 보정 근거), 범위 밖 1건(.pytest_cache).

## 1. 수정 내용

| # | 심각도 | 파일 | 무엇을 바꿨나 | 회귀 테스트 |
|---|---|---|---|---|
| 1 | BUG | `embed_judge.py` | 캐시 줄에 `device` 를 함께 기록하고, 모든 텍스트가 캐시에서 나와 모델을 올리지 않는 재실행에서는 캐시 줄의 `revision`/`device` 를 넘겨받아 `info()`(→ `settings.embed`)에 기록한다. 값이 하나로 모일 때만 채택, 섞여 있으면 경고 후 `null`. `--embed-revision` 고정값은 그대로 기록되고 다른 revision 줄은 무시. 적재한 모델의 revision 이 캐시 줄과 다르면 경고(벡터 혼합). `device` 요청값(`--device`)은 사실이 아니라 기록하지 않는다 — 모델을 올렸거나 캐시 줄이 말해 줄 때만. | `test_warm_cache_run_records_same_revision_and_device`, `test_cache_provenance_adoption_rules` |
| 2 | RISK | `gold23.py` | 원장 파일명 정규식 `^(?:\d+_)?(\d{5,8})\s+(.*?)(?:\s*\(\d+\))?$` — 장세영 전달 5건 형식(`00002_13387752 운수 자판기 (1)`)의 앞 순번·뒤 `(1)` 허용. `load_ledger()` 는 id 를 읽은 행이 0건이거나 비어 있지 않은 행의 절반 미만이면 **ValueError** (경고 후 전부 na 인 gold 를 조용히 만들지 않는다). | `test_ledger_work_id_forms`, `test_load_ledger_fails_instead_of_silent_all_na`, fixture 행 `00002_13387752 운수 자판기 (1)` |
| 3 | RISK | `gold23.py`, 규격 §1 #2·§3 | **file_name 관례를 하나로 정했다**: T-001 gold = 시스템에 실제로 주는 계약서 PDF 의 basename. `--file-name-from auto`(기본) = frozen ids csv 의 `contract_pdf` basename, 그 열이 없으면 `<원장 파일명>.pdf`; `ids` / `ledger` 로 한쪽 강제. ids 와 원장이 다른 세트 수를 경고로 알린다. 전달 PDF 이름 바꾸기(다른 선택지)는 `dataset/` 가 쓰기 범위 밖이라 하지 않았다. | `test_t001_file_name_follows_delivered_contract_pdf`, CLI `--file-name-from ids` |
| 4 | RISK | `evaluator.py` | 23키가 하나도 없는 예측(필드가 `metadata` 아래에 감싸인 API 응답 등) → `warnings.predictions_without_known_keys` + report §6 (채점은 그대로 전부 null = FN). 예측 디렉터리는 `.json` 을 대소문자 무시로 읽고(`X.JSON`), 그 외 파일은 `warnings.ignored_files` 로 알린다; `S1.json`+`S1.JSON` 은 `duplicate_predictions`. | `test_predictions_without_known_keys_and_case_insensitive_json` |
| 5 | RISK | `normalize.py`, 규격 §1 | `norm_rights` 는 표 조회 전에 뒤 괄호를 뗀다(`허용(방송권, 디지털음성송신권)` → 허용; §1 #8 하위 권리는 키가 아님). **체크 글리프(v/V/■/□)는 예측값으로 받지 않기로 정했다** — 원장/gold 표기일 뿐이고 추출기는 허용/불허 를 내야 한다(규격에 명시). | `test_norm_rights_forms`, `test_norm_rights_unparseable` |
| 6 | RISK | (문서) | τ 보정 근거 기록 — 아래 §2. 코드 변경 없음. | — |
| 7 | NIT | `matchers.py` | `work_title` 관리번호 접두는 **예측 쪽만** 뗀다(규격 §1 #1). implementer 노트 #8 의 양쪽 제거는 철회(원장 제목 중 5~8자리 접두 시작 0건 — 리뷰 검증). | `test_work_title_strips_id_prefix` (gold 접두 → 오답 확인) |
| 8 | NIT | `evaluate.py` | `logging.basicConfig` 는 `api` 패키지가 root 핸들러를 먼저 달아 무시되던 것 → `configure_logging()` 이 `api.module.evaluation.v4` 패키지 로거에 stderr 핸들러를 달고(`propagate=False`, 중복 출력 방지) INFO/DEBUG 를 보이게 한다. 핸들러는 항상 '지금의' `sys.stderr` 에 써서 테스트(capsys)에서도 닫힌 스트림을 잡지 않는다. CLI 의 Macro CI 는 `_ci_str()` 로 퍼센트 표기. | `test_cli_main_t001` (CI 형식 + stderr 로그) |
| 9 | NIT | `matchers.py`, `evaluator.py` | 'value' gold 가 정규화 후 비어 empty 로 다시 해석되면 `detail["state"]="empty"` 를 남기고 평가기가 그 상태로 ①/②/③ 표를 센다(채점과 상태 표 일치). | `TestEffectiveState`, `test_reinterpreted_empty_gold_counts_as_state_empty` |
| 10 | NIT | `normalize.py`, 규격 §1 | `norm_resolution(['1920','1080'])` 허용(숫자형 원소 2개 리스트); `norm_format('application/x-hwp')` → `HWP`(MIME `x-` 접두 제거); 날짜 14/15 는 순수 날짜만 받는다고 규격에 명시(코드 변경 없음). | `test_norm_resolution_forms`, `test_norm_format` |
| 11 | NIT | `.gitignore` / `.pytest_cache` | **범위 밖 — 수정 안 함.** `.gitignore` 는 `api/module/evaluation/v4/`·`docs/v4/` 밖이다. 리포 루트의 `.pytest_cache/`(리뷰어 실행이 만든 것)도 지우지 않았다. 권장: `.gitignore` 에 `.pytest_cache/` 추가(담당자), 그때까지 `-p no:cacheprovider` 로 실행. | — |

테스트: 165 → **197 passed** (추가 32건; 변경 1건 = 제목 접두 gold 쪽 제거 철회). 평가기 수준 회귀 `test_warm_embed_cache_rerun_is_byte_identical` 이 리뷰어의 따뜻한-캐시 시나리오(revision/device null)를 재현해 막는다. `.pytest_cache` 를 만들지 않으려면 `-p no:cacheprovider` 를 붙인다.

## 2. τ = 0.65 보정 근거 (리뷰어 측정, 코드 변경 없음)

- caption gold 5세트에서 OpenAI 초안(같은 그림, 독립 표현)과 사이트 캡션의 코사인: **0.6955**(일치, 공유 토큰 1), **0.6979**(불일치 — 공유 토큰 0으로 내용 확인에서 떨어짐), 0.6452, 0.6459, 0.5912.
  → 같은 저작물의 올바른 환언이 τ = 0.65 바로 근처에 몰려 있고, 경계 사례는 내용 토큰 확인이 결정한다.
- generated 트랙(pred == gold 텍스트)은 모두 코사인 정확히 1.0 — 자기 자신과의 비교라 보정 정보가 없다.
- implementer 스모크(`M1_implementer_A_notes.md`): 같은 장면 환언 0.914 · 다른 저작물 0.334 · 야간 바닷길 환언 0.745; §5d 불일치 p95 0.516.
- **해야 할 일(τ 확정 전):** 설명 파일럿 §5d 의 τ 스윕을 caption gold 가 있는 T-002 **679세트 전체**에서 돌리고, report.md 의 kind 별(caption vs generated) 표를 근거로 τ 를 정한다. 그때까지 규격의 0.65 는 **잠정값**(규격 §5 에 표시).

## 3. 결정 사항 요약 (규격에 반영)

1. **file_name(§1 #2)** — gold = 시스템에 주는 계약서 PDF 의 basename; 출처 우선순위 ids csv `contract_pdf` > 원장 파일명 + `.pdf`. 장세영 쪽 명명(`<원문인덱스> <제목>.pdf`)이 지켜지면 둘은 같다. 5건 임시 계약서로 시험할 때는 **그 파일을 가리키는 frozen ids csv** 를 만들면 gold 가 자동으로 전달 파일명을 따른다.
   - 실데이터 확인(2026-10-09, 현재 frozen `t001_ids.csv` 4,000세트 × 기존 원장): 원장 행이 있는 3,677세트 중 **61세트의 contract_pdf basename 이 `<원장 파일명>.pdf` 와 다르다** (예: `13300002 트랜스-매치스.pdf` vs 원장 `13300002 트랜스-매치스. 12_1845, 구출작전` — 제목의 구두점에서 PDF 생성기가 이름을 자른 듯). 디스크에는 ids 경로가 61/61건, 원장 이름 파일이 0/61건 존재 → 원장 전용 관례였다면 이 61세트의 file_name 은 구조적으로 오답이었다. `auto` 가 이를 바로잡고 경고로 건수를 알린다. 나머지 323세트는 원장 행 없음(계약서 미생성) → na.
2. **권리 값(§1 Rights values)** — 뒤 괄호 제거 후 8형식만; 글리프는 오답.
3. **날짜(§1 #14/15)** — 순수 날짜만(`부터`/`(계약일)` 붙으면 오답); `부터/까지` 는 #16 에서만.
4. **임베딩 provenance(§5)** — 캐시 줄 model/revision/device; 따뜻한 캐시 재실행도 같은 settings.embed.
5. **work_title(§1 #1)** — 접두 제거는 예측 쪽만.

## 4. 바꾸지 않은 것

- 23키가 없는 예측의 **자동 언래핑**(`metadata` 아래 필드 꺼내기)은 하지 않았다 — 스키마 추측이 되므로 경고만 낸다(리뷰어 제안 그대로).
- 캐시 revision 이 섞여 있을 때 **예외로 멈추지 않고 경고**한다(리뷰어 제안). 섞임을 피하려면 `--embed-revision` 고정.
- `gold23.configure_logging` 은 그대로 두었다(같은 패턴이지만 이번 리뷰 범위 밖).
