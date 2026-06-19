# kcc2026paper/data — 데이터 전수조사·수집 (2026-06-19)

`copyright_metadata_extraction` 저장소 전체를 전수조사하여, KCC2026(단일 KLUE-BERT)·
IC-EEECS(다중 인코더) NER 논문이 사용하는 데이터를 이 디렉토리로 모았다.
공유 데이터는 원본을 복사했고(원본은 출처에 유지), 거대 모델 트리(체크포인트 등)는 대상이 아니다.

학습/평가용 silver·gold 는 별도로 `../configs/integrated/{silver,gold}` 에 있다(상위 README 참조).

## 구성 (출처 매핑)

| 경로 | 내용 | 출처 | git |
|---|---|---|---|
| `ner_bio_encoders/<model>/{train,test,validation}.txt` | 다중 인코더 BIO 데이터 (klue-roberta-large · mBERT · xlm-roberta-large) + mBERT eval log | `api/src/ner/training/` | 추적 |
| `ground_truth/` | 실문서 정답(JSON) + `auto_extracted_ground_truth.json` | `api/data/ground_truth/`, `api/data/in/` | 추적 |
| `ner_eval_out/ner/` | 인코더별 추론 entity·summary | `api/data/out/ner/` | 추적 |
| `ner_eval_out/ner_results/` | NER 결과 | `api/data/out/ner_results/` | 추적 |
| `ner_eval_out/ner_benchmark/` | pure_bert / crf_bert × threshold fold 데이터 | `api/data/out/ner_benchmark/` | 추적 |
| `ner_eval_out/*.json,*.txt` | eval summary · confidence 실험 · 생성 샘플 | `api/data/out/` | 추적 |
| `ner_predictions/*.csv` | consent/contract 예측·학습 결과 | `ner/out/` | 추적 |
| `train_text/test/`, `train_text/real_document_train/` | 평가/학습 텍스트 (소형) | `api/data/in/` | 추적 |
| `train_text/{real_document_train,realistic_train}.txt` | 대용량 학습 텍스트 (각 ~20MB) | `api/data/in/` | **gitignore**(디스크 전용) |
| `documents_raw/{document,cleaned}` | 원문 문서 코퍼스 (계약서·동의서 등, ~167MB, 미디어 포함) | `api/data/in/` | **gitignore**(디스크 전용) |

## 제외(미수집) — 파생/대용량

- `api/data/out/pdf_convert` (562MB, PDF 변환 산출), `ner_visualization`(viz png), `ocr`(1.2MB) — 파생물이라 제외.
- 모델 체크포인트(`paper1/data/runs/*`, 59GB; `*/models/*`) — 가중치는 gitignore 정책.
- 전체 학습 코퍼스 `paper/data`(1.8GB)·`metadata/data`(1.4GB) — silver/gold 로 이미 대표됨.

## 참고 — §0 다중 인코더 eval 블로커

`PAPER_PLAN_AND_FINDINGS.md §0` 의 깨진 `eval_results_*.json`(0.0 F1·test_data_path 불일치)
원본은 이 저장소에 없다(황성훈 GitHub push 대상). 위 `ner_bio_encoders/`·`ner_eval_out/`·
`ground_truth/` 는 현재 저장소에 존재하는 다중 인코더 데이터이며, 깨끗한 재평가 시 입력으로 쓸 수 있다.
