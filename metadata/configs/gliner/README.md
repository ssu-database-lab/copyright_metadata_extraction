# GLiNER 설정 (configs/gliner)

**predict** 만 관리하면 됩니다. **train** 은 predict 에 있는 라벨 안에서만 자동으로 동작합니다 (auto).

## 디렉터리 구조

```
configs/gliner/
├── README.md
├── predict/          # 추론용: 라벨 설명 텍스트 (.txt, .md)
│   ├── address.txt
│   ├── company_name.txt
│   ├── email.txt
│   └── ...
└── train/            # 학습용: 라벨별 JSONL
    ├── README.md
    ├── email.jsonl
    ├── company_name.jsonl
    └── ...
```

---

## predict/ — 추론(Zero-shot)용 라벨 설명

- 각 라벨별로 `{라벨명}.txt` (또는 `.md`) 파일을 두면, 그 내용이 모델에 "이 라벨이 뭔지" 설명으로 전달됩니다.
- 예: `person_name.txt`, `address.txt`, `phone.txt` 등.
- 코드에서는 `configs/gliner/predict/` 를 우선으로 읽고, 없으면 `configs/gliner/` 루트, 그 다음 `configs/training/ner_labels/` 를 봅니다.

### Zero-shot에서 트레이닝 대신 할 수 있는 것

1. **라벨 설명 보강**
   - 한두 문장 정의 + 예시 1~2개가 있으면 충분합니다.
   - **도메인 용어를 넣기**: 저작권/계약서 문서라면 "대표이사", "저작자", "계약일", "저작물명" 같은 단어를 해당 라벨 설명에 넣어 두면 인식에 도움이 됩니다.
   - 무엇을 **제외**할지 짧게 적어 두어도 좋습니다 (예: address에 "이메일 주소 X").

2. **라벨별 threshold 조정** (트레이닝 없이 recall/precision 조정)
   - `configs/labels.yaml`의 `ner` 아래에 `threshold`, `threshold_per_label` 로 조정 가능.

3. **규칙과 병행**
   - 전화번호, 이메일, 날짜는 `regex_labels`로 먼저 처리하고, NER은 이름·주소·기관명 등에 집중하는 구성이 유리합니다.

4. **라벨 개수**
   - 꼭 필요한 것 위주로 두고, 비슷한 것은 하나로 묶는 편이 낫습니다.

---

## train/ — 학습용 데이터 (라벨별 JSONL, predict 안에서만)

- **라벨은 predict 에서만 정의**합니다. train 에 넣는 `{라벨명}.jsonl` 은 **predict/ 에 같은 이름의 .txt/.md 가 있는 라벨**만 병합됩니다.
- **auto**: predict 시 학습 데이터(`train/`)를 검사해 **달라진 점이 있으면 학습 후 predict**, 없으면 그냥 predict. `ner_extractor(..., auto=True)` (기본).
- `configs/labels.yaml` 의 `ner.labels: auto` 이면 라벨 목록도 **configs/gliner/predict/** 에서 자동 수집합니다.

---

## 요약 (auto)

- **라벨 소스**: `configs/labels.yaml` 에 `ner.labels: auto` 로 두면, **predict/** 에 있는 .txt/.md 파일명이 곧 라벨이 됩니다. predict 만 수정하면 추론·학습 모두 같은 라벨 집합으로 동작합니다.
- **추론**: `configs/gliner/predict/*.txt` 내용이 라벨 설명으로 사용됩니다.
- **학습**: `configs/gliner/train/*.jsonl` 중 **predict 에 있는 라벨과 같은 이름**만 병합됩니다. train 은 은연중에 predict 안에서만 동작합니다.
- **임계값**: `labels.yaml` 의 `ner.threshold`, `ner.threshold_per_label`.
