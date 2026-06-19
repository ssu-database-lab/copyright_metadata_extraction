# STATUS — kcc2026paper 협업자 핸드오프 (2026-06-19)

KCC2026(단일 KLUE-BERT) · IC-EEECS(다중 인코더) NER 논문 프로젝트. 본 디렉토리는 `paper/` 에서
분리해 최상위로 옮긴 것으로, 논문이 쓰는 데이터·코드·산출물을 자족적으로 담는다.
전체 구조·경로·데이터 출처는 [README.md](README.md), [data/README.md](data/README.md) 참조.

## 1. 지금 바로 가능한 것 (ready)

- **KCC2026 paper1 재현**: 데이터(silver 26라벨·gold 36라벨·빌드된 `rule/{m1,m2,m3}`)·코드·환경 모두 갖춰짐.
  ```bash
  cd kcc2026paper
  .venv/bin/python paper1/paper1.py analyze <stamp>   # 기존 run 집계 (CPU 가능)
  .venv/bin/python paper1/paper1.py run --configs rule_m1,rule_m2,rule_m3   # 학습 (GPU 권장)
  ```
  - **환경: `kcc2026paper/.venv`** (GPU 학습 전용, Python 3.12, torch 2.12+cu130 · transformers 5.11 · accelerate · blingfire · kiwipiepy 등). git 제외.
  - `paper_module/core/ner/base.py` 는 현행 metadata API(`_runtime`)에 맞춰 패치됨.
  - ⚠️ **GPU 블로커**: 현재 `nvidia-smi` 가 `NVML Driver/library version mismatch` (드라이버 580.167) 로 실패 → `torch.cuda.is_available()==False`.
    **재부팅**(또는 nvidia 커널모듈 재로드)으로 풀어야 GPU 학습 가능. venv·torch 빌드(cu130, RTX 5070/드라이버580→CUDA13)는 정상이라 재부팅 후 바로 동작.
  - 데이터 무결성 검증 완료: M1 193,752 · M2 192,927 · M3 192,927 (neg 25.0%), build_summary 대조 0 불일치.
  - 재현 기준 run = `paper1/data/runs/20260427_143110` (= paper1.md M1=0.4635·M2=0.8745·M3=0.8673). run 간 variance 존재.
  - **학습 추가 의존성**: `datasets`·`seqeval` (이 venv 에 설치 완료). transformers 5.11 호환을 위해 `paper1.py` `_Tee` 에 `isatty()` 추가 패치함.
  - ✅ **재현 검증 (2026-06-19, RTX 5070, 30.6분)**: 신규 run `20260619_224718` → M1 0.4910 / M2 0.8722 / M3 0.8498.
    기준 대비 Δ = +0.028 / −0.002 / −0.018 (모두 ±0.03 이내, run-to-run variance 범위). M1 붕괴·M2≈M3 핵심 결론 그대로 재현.
- **26-라벨 taxonomy**: [LABEL_TAXONOMY.md](LABEL_TAXONOMY.md) + [data/label_taxonomy.csv](data/label_taxonomy.csv)
  (Free/Regular/Semi-Regular, source of truth = `paper1.py:FORMAT_CLASS`).
- **다중 인코더 NER 데이터**: `data/ner_bio_encoders/<klue-roberta-large|mBERT|xlm-roberta-large>/{train,test,validation}.txt`,
  정답 `data/ground_truth/`, eval/benchmark 출력 `data/ner_eval_out/` — 재평가 입력으로 사용 가능.

## 2. PLANS 진행 현황 (PAPER_PLAN_AND_FINDINGS.md)

| PLANS 항목 | 상태 | 비고 |
|---|---|---|
| §8.3 26-라벨 enumeration + Free/Regular/Semi 분류 | ✅ **완료** | `LABEL_TAXONOMY.md` (코드에서 생성, 26=14/6/6 검증) |
| §5 데이터 수집·자족화 | ✅ **완료** | silver/gold/rule + `data/` 전수조사 수집 (README 참조) |
| 환경·import 정합 | ✅ **완료** | 패키지 설치 + base.py 셔임 패치, `paper1.py --help` 로드 확인 |
| §0/§11 #1 — 다중 인코더 eval 수정 → 깨끗한 M1/M2/M3 | ⛔ **블로커** | 아래 §3 |
| §8.2 실측 metric 표·per-label·유의성 | ⛔ 대기 | eval 수정 후 |
| §8.5 end-to-end ablation, §8.6 BRIDGE, §8.7 arbiter eval | ⛔ 범위 밖 | consolidation 파이프라인(api) 별도 |

## 3. 블로커 — 다중 인코더 eval (선배 액션 필요)

PLANS §0: 3개 인코더의 `eval_results_*.json` 이 P=R=F1=0.0 이고 셋 다 mBERT 의 `test_data_path` 를
가리켜 0점. **그 깨진 산출물·학습된 모델 체크포인트 원본은 이 저장소에 없다**(황성훈 GitHub push 대상).

필요한 것(주시면 바로 진행):
1. 3개 인코더의 **학습된 모델** (또는 재학습 가능한 HP/명령).
2. 각 모델의 **올바른 per-model test 파일** (현재 `data/ner_bio_encoders/<model>/test.txt` 가 후보).
3. 깨진 `eval_results_*.json` 원본(재조정용, 있으면).

→ 받는 즉시 (a) 0.0 원인(test_data_path 불일치) 확인, (b) 각 모델을 자기 test 로 재평가,
(c) 깨끗한 M1/M2/M3 표(Silver/Gold) 산출까지 진행.

## 4. 다음 액션 제안

- **선배**: 위 §3 의 모델·eval 산출물 push → 다중 인코더 재평가 unblock.
- **공통**: taxonomy(완료)를 IC-EEECS §III / 저널 Appendix A 에 반영, `[FILL]` 표는 재평가 수치로 채움.
