# M1 구현 노트 — 평가기 스택 (implementer A, 2026-10-09)

패키지 `api/module/evaluation/v4/` · 규격 `docs/v4/M1_evaluator_spec.md` · 테스트 `venv/bin/python -m pytest api/module/evaluation/v4/tests -q`

| 파일 | 규격 절 | 내용 |
|---|---|---|
| `normalize.py` | §4, §1 | norm_text/text_key, strip_id_prefix, norm_name/split_names, norm_tag/split_tags, norm_date, norm_period, norm_rights, norm_work_type, norm_format, norm_int_bytes, norm_resolution, norm_hash |
| `matchers.py` | §4, §1, §2 | scalar_counts(슬롯 채우기 관례), match_tag_lists(포함·1:1 탐욕), names_equal, match_key 디스패치 |
| `embed_judge.py` | §5 | EmbedJudge(KURE-v1, CLS 풀링, L2, fp16/CUDA, max_length 512, sha1 jsonl 캐시), content_tokens + 불용어 |
| `evaluator.py` | §6–§7 | load_gold/load_predictions, score_set, aggregate(키·셀·kind, Macro/Micro, 부트스트랩 CI), render_report, evaluate() |
| `evaluate.py` | §7 CLI | argparse CLI, `evaluate` 재노출 |
| `tests/` | §8 a–d | test_normalize(102), test_matchers(24), test_embed_judge(10), test_evaluator_e2e(11) = 147개; implementer B 의 test_gold23(18) 과 합쳐 165개 통과 (2026-10-09) |

`gold23.py`(§3, §8 e)는 implementer B 담당 — 여기서는 작성하지 않았다.

## 규격에서 정하지 않은 것 → 이렇게 정했다

1. **정규화 함수의 반환 규약** — None/빈 문자열/빈 리스트 → `None`(출력 없음), 해석 불가 → `ValueError`. 평가기는 ValueError 를 "값은 냈으나 틀림"으로 센다(gold value → FP+FN, gold empty → FP) 하고 `warnings.unparseable` 에 기록한다. **gold 쪽**이 해석 불가이면 그 판정은 제외하고 `warnings.gold_unparseable` 에 적는다(gold 를 고쳐야 한다).
2. **빈 문자열·빈 리스트 = null.** 문자열 `"null"`/`"none"` 도 권리 값에서는 null 로 본다.
3. **Macro 평균 포함 키** = 평가 건수(value+empty) > 0 **이고** 2TP+FP+FN > 0 인 키. 평가 건수는 있지만 판정이 하나도 없는 키(모두 empty + null 예측)는 F1 이 정의되지 않으므로 `excluded_keys` 에 `no_decisions` 로 나열하고 평균에서 뺀다. 판정이 전혀 없으면(2TP+FP+FN = 0) 정밀도·재현율·F1 모두 None(보고서 `-`); 한쪽 분모만 0 이면 그 값은 0.0 (sklearn 관례). F1 은 2TP/(2TP+FP+FN) 로 직접 계산한다.
4. **other_rights(type `na`)** 는 gold 상태와 무관하게 항상 제외한다(규격 §1 #13 "never FP"). 예측이 non-null 이면 `unscored_nonnull` 경고에 올린다.
5. **평가 대상 외 키 경고는 T-001 에만** 적용(17~23 non-null). T-002 의 1~16 은 §0 에서 "계약서 파싱값 복사 가능"이라고 했으므로 경고하지 않는다.
6. **리스트 매칭의 "longest first"** = gold 를 긴 것부터 처리하고, 후보 예측값 중 완전 일치 > 더 긴 예측값 > 사전순으로 고른다(결정적). 중복 태그는 `split_tags` 에서 제거되므로 TP 를 두 번 세지 않는다. gold 가 `value` 상태인데 정규화 후 빈 리스트이면 `empty` 로 취급한다.
7. **태그 문자열 분리** — `, | ; 개행 、` 와 `#` 경계에서만 끊고 **공백으로는 끊지 않는다**("경복궁 단청" 같은 두 단어 태그 보존).
8. **work_title 관리번호** — 규격은 예측 쪽만 떼라고 하지만 gold 에도 접두가 남아 있을 수 있어 양쪽 모두 뗀 뒤 비교한다(접두가 없으면 무변화). 비교 전 `casefold`(라틴 대소문자 무시). **[2026-10-09 리뷰 반영: 규격대로 예측 쪽만 뗀다 — 원장 제목 5,834건 중 5~8자리 접두로 시작하는 것이 0건이라 gold 쪽 제거는 불필요했고, 정당한 8자리 접두를 빠뜨린 예측을 맞다고 볼 위험만 있었다. `M1_review_fixes_20261009.md` 참조.]**
9. **file_name** — NFC + 공백 압축 후 **라틴 대소문자 무시**(`norm_text` 의 일반 규칙과 일치; `.PDF`/`.pdf`). 공백→`_` 치환은 그대로 오답이다.
10. **work_type 변형 매핑**은 보수적으로: 끝의 `저작물`·공백·괄호 제거 + {프로그램, 컴퓨터프로그램, 소프트웨어 → 컴퓨터프로그램; 동영상, 영상물 → 영상}. 그 외(예: '그림')는 해석 불가.
11. **file_format 정규형**: JPG, TIF, MP4 (JPEG/TIFF/MPEG4/MPEG-4 를 여기로). MIME(`image/jpeg`)·파일명(`a.JPG`)·`.jpg` 도 받는다.
12. **file_size** `"11553825.0"`(인덱스 표기)·`1234.0` 은 정수로 받고, KB/MB/소수는 해석 불가.
13. **설명 판정기 없이 `match_key` 를 부르면** 완전 일치(casefold)로 판정한다. 평가기는 T-002 에서 항상 EmbedJudge 를 만들지만 모델은 첫 `decide()` 때 적재한다 → T-001 은 torch 모델을 올리지 않는다.
14. **내용 토큰** = `[0-9A-Za-z가-힣]+` 최대 연속 구간(한글·라틴·숫자 혼합 구간은 하나의 토큰), 길이 ≥ 2, `norm_tag` 후 불용어 제외. 조사가 붙은 어절("바다를")은 "바다"와 다른 토큰이다 — 규격 그대로. 코사인은 반올림 없이 비교하고(τ 경계 포함 `>=`), 캐시 벡터도 float32 무손실로 저장해 재실행 때 같은 값이 나온다.
15. **임베딩 캐시**는 sha1(정규화 텍스트) 키 + `model`/`revision` 필드. 모델 id 가 다르거나 revision 이 둘 다 있고 다르면 그 줄은 무시한다.
16. **예측 누락 세트**는 빈 예측(전부 null)으로 채점한다 → gold value 인 키는 FN(리스트는 원소 수만큼), empty/na 는 0. `warnings.missing_predictions` 에 나열.
17. **부트스트랩**은 gold 세트 전체(예측 누락 포함)를 모집단으로 1,000회 복원추출(`numpy.random.default_rng(0)`), 키별 F1·Macro(포함 키의 nanmean)·Micro 의 2.5/97.5 백분위. 재표본에서 판정이 0 인 키는 그 재표본에서 건너뛴다. 셀별 지표에는 CI 를 붙이지 않는다(점추정만).
18. **`--cells`** 는 report.md 에 셀별 항목 표를 넣는 스위치다. JSON 에는 셀별 지표가 항상 들어가고, report.md 에는 셀별 Macro/Micro 요약표가 항상 들어간다.
19. **출력 결정성** — JSON 은 `sort_keys`, per_set.jsonl 은 set_id(문자열 정렬)×키 순서, 타임스탬프 없음. 반환 dict 에만 `outputs`(파일 경로)가 있고 파일에는 없다.
20. **gold 레코드의 `test` 가 요청 시험과 다르면** 제외하고 `gold_test_mismatch` 경고. `keys` 에 없는 키는 상태 na 로 본다.

## 환경 메모
- `venv` 에 pytest 가 없어 `pip install pytest`(9.1.1)로 추가했다 — 개발 도구만, 코드 의존성 아님.
- `python -m api.module.evaluation.v4.…` 는 `api/__init__.py` → torch/transformers 를 끌어와 첫 import 에 ~30초 걸린다(리포 구조상 v4 안에서는 피할 수 없음; 기능에는 영향 없음).
- 규격의 테스트 명령(`venv/bin/python -m pytest …`)은 리포 루트에 `.pytest_cache/` 를 만든다(.gitignore 에 없음). 커밋 전에 지우거나 `-p no:cacheprovider` 로 돌린다.
- 실제 모델 스모크(오프라인, CUDA fp16): 같은 장면 환언 0.914·다른 저작물 0.334·야간 바닷길 환언 0.745 — τ 0.65 와 §5d 의 분리(불일치 p95 0.516)와 부합. revision `8b418a58414668e75532ed045c22d9ca018ae2b2` 가 `info()` 에 기록된다.
- KURE-v1 스냅샷은 `~/.cache/huggingface/hub/models--nlpai-lab--KURE-v1/snapshots/8b418a58414668e75532ed045c22d9ca018ae2b2` — EmbedJudge 는 `local_files_only=True` 로 먼저 시도하므로 네트워크 없이 돈다.
