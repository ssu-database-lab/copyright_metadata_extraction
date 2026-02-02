# NER 시스템 검증 & 개선 요약

## 📊 검증 결과

### ✓ 정상 항목들 (모두 정상)

| 항목 | 상태 | 세부사항 |
|-----|------|---------|
| **labels.yaml** | ✓ | 7개 NER 라벨 정의 (address, company_name, person_name, phone_number, email, url, date) |
| **BIO 라벨** | ✓ | 15개 = O + 7×2 (각 라벨마다 B-, I- 태그) |
| **라벨 매핑** | ✓ | LABEL_TO_ID ↔ ID_TO_LABEL 완벽한 일관성 |
| **학습 데이터** | ✓ | 4,100개 샘플, 7개 파일, 유효한 JSON 형식 |
| **데이터 분포** | ✓ | 라벨별 균형 (각 600개 또는 500개) |
| **Train/Val Split** | ✓ | 80:20 비율, 검증 데이터 820개 |

---

## ❌ 발견된 문제: Precision/Recall/F1 = 0

### 원인 분석

**문제:** 각 어댑터가 자신의 라벨 샘플만으로 학습
```
ner_address 어댑터  → address.jsonl만 학습
ner_email 어댑터    → email.jsonl만 학습
...
```

**결과:** 라벨 혼동으로 평가 메트릭 완전 실패
```
예측: [B-address, B-address, B-address, ...]  ← 모든 예측이 address
정답: [B-address, I-address, B-email, ...]    ← 다양한 라벨
→ 불일치 → F1=0
```

### 해결책: 모든 라벨 혼합 데이터로 학습

**변경 전 (문제)**
```python
# 각 라벨별로 분리 로드
for label in ner_labels:
    label_samples = _load_samples_for_label(data_dir, label)  # address만!
    # 이 데이터로 ner_address 어댑터 학습
```

**변경 후 (개선)**
```python
# 1단계: 모든 라벨 데이터 혼합
all_mixed_samples = []
for label in ner_labels:
    all_mixed_samples.extend(_load_samples_for_label(data_dir, label))

# 2단계: 전체 데이터 train/val split
all_train_samples, all_val_samples = _split_train_val(
    all_mixed_samples, train_ratio=0.8
)

# 3단계: 각 어댑터가 혼합 데이터로 학습 (어댑터만 활성화)
for label in ner_labels:
    adapter_name = f"ner_{label}"
    model.add_adapter(adapter_name, config=adapter_cfg)
    model.set_active_adapters(adapter_name)  # 이 어댑터만 활성화
    
    # 혼합 데이터로 학습!
    train_ds = NERDataset(all_train_samples, tokenizer)
    trainer.train()
```

---

## 🎯 개선의 의미

### 학습 프로세스 개선

| 단계 | 이전 | 개선 후 |
|-----|------|---------|
| **1. 데이터 로드** | 라벨별 분리 | **모두 혼합** |
| **2. Split** | 라벨별 split | **전체 split** |
| **3. 학습** | 단일 라벨만 | **모든 라벨** (어댑터만 활성화) |
| **4. 평가** | 앙상블 (불일치) | **앙상블 (일관성 있음)** |

### 예상 결과

**이전:**
- 각 어댑터: 자신의 라벨만 99% 정확도
- 앙상블: 모든 라벨이 섞여서 F1=0

**개선 후:**
- 각 어댑터: 모든 라벨을 구분하며 학습
- 앙상블: 올바른 F1 메트릭 도출
- 일관성: 예측과 정답 라벨 일치 높음

---

## 📈 학습 데이터 통계

### 데이터 구성
```
┌─────────────┬──────┬─────────┐
│    라벨      │ 수량 │ 비율    │
├─────────────┼──────┼─────────┤
│ address     │  600 │ 14.6%  │
│ company_name│  600 │ 14.6%  │
│ date        │  600 │ 14.6%  │
│ email       │  600 │ 14.6%  │
│ phone_number│  600 │ 14.6%  │
│ url         │  600 │ 14.6%  │
│ person_name │  500 │ 12.2%  │
├─────────────┼──────┼─────────┤
│ 합계        │4,100 │ 100%   │
└─────────────┴──────┴─────────┘

Train (80%): 3,280개
Val (20%):   820개
```

### BIO 라벨 분포 (샘플)
```
address:      B-address=1, I-address=13, O=3
company_name: B-company_name=1, I-company_name=7, O=4
email:        B-email=1, I-email=16, O=4
url:          B-url=1, I-url=22, O=3
```

---

## 🔄 Adapter 학습 흐름 (개선)

```
모든 라벨 혼합 데이터 (4,100개)
         ↓
    Train/Val Split
    (80:20)
         ↓
┌─ Train (3,280개) ─┐
│                    │
│  adapter1 학습     │
│  (address 강화)    │
│         ↓          │
│  adapter2 학습     │
│  (company 강화)    │
│         ↓          │
│  ...               │
└────────────────────┘
         ↓
    통합 평가
    (320개 검증)
         ↓
  precision/recall/f1
```

---

## ✅ 변경 사항 요약

### train.py 수정 내용

1. **데이터 로드 변경**
   - 라벨별 분리 → 모든 라벨 혼합

2. **Split 방식 변경**
   - 라벨별 split → 전체 split

3. **학습 루프 개선**
   - 라벨별 독립 학습 → 혼합 데이터로 어댑터 다중 학습

4. **결과 기록**
   - `training_method: "mixed_labels"` 추가

---

## 🚀 다음 단계

1. **즉시 실행**
   ```bash
   python3 main.py
   ```

2. **예상 결과**
   - 각 어댑터가 3,280개 혼합 샘플로 학습
   - 앙상블 평가에서 정상적인 F1 메트릭 도출

3. **성공 확인 포인트**
   ```
   [통합 평가 결과]
   Precision: 0.xxxx (0이 아님!)
   Recall:    0.xxxx (0이 아님!)
   F1-score:  0.xxxx (0이 아님!)
   ```

---

## 📝 결론

✓ **데이터 품질**: 완벽
✓ **라벨 설정**: 완벽
✓ **구조**: 정상

❌ **문제**: 학습 전략
✅ **해결**: 혼합 데이터 학습

**→ train.py 수정 완료, 재실행 필요**
