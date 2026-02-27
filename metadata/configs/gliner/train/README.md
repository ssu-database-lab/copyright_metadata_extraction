# GLiNER 학습 데이터 (라벨별)

- **학습 데이터**: 이 디렉터리에 라벨별 `.jsonl` 저장. 라벨은 `configs/gliner/predict/` 와 동일한 이름.
- **어댑터 저장**: `configs/gliner/train/adapter/` (및 `train_state.json`). 예측 시 여기서 로드.
- **예측**: `configs/gliner/predict/` 참고. 예측 전 어댑터 확인 → 파일 변경 시 추가 학습, 없으면 그대로 → 예측.

## 파일 형식

- 파일명: `{라벨명}.jsonl` (예: `email.jsonl`, `company_name.jsonl`)
- 한 줄 = 한 문장. JSON: `tokens`, `labels`(BIO), (선택) `id`

```json
{"id": "doc1_sent0", "tokens": ["연락처", ":", "help@company.co.kr"], "labels": ["O", "O", "B-email"]}
```

## 사용 방법

1. 라벨별 `.jsonl` 파일을 이 디렉터리에 저장 (predict 에 있는 라벨명과 맞추기).
2. **auto (기본)**: 메타데이터 추출(predict) 시 학습 데이터가 바뀌었으면 자동으로 학습 후 predict, 바뀌지 않았으면 그대로 predict.
3. **수동 학습**: `python -m module.extractor.ner.train --train_dir configs/gliner/train`  
   (기본 --train_dir 가 이 경로이므로 생략 가능: `python -m module.extractor.ner.train`)

Weak supervision으로 생성할 때:

```bash
python scripts/generate_training_data_from_documents.py --output_dir configs/gliner/train
```
