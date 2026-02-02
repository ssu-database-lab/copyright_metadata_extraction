# Weak Supervision을 사용한 학습 데이터 자동 생성 가이드

## 개요

Weak Supervision은 정규식 패턴, 키워드, 규칙 등을 사용하여 실제 문서에서 자동으로 학습 데이터를 생성하는 방법입니다. 수동 라벨링 없이 대량의 학습 데이터를 생성할 수 있습니다.

## 방법 1: 문서 디렉토리에서 자동 생성

### 기본 사용법

```bash
# 기본 사용 (data/in/document에서 읽어서 configs/training/ner_labels에 저장)
python scripts/generate_training_data_from_documents.py

# 옵션 지정
python scripts/generate_training_data_from_documents.py \
    --input_dir data/in/document \
    --output_dir configs/training/ner_labels \
    --min_tokens 3 \
    --max_samples_per_file 100
```

### 동작 방식

1. **정규식 기반 라벨링**: `labels.yaml`의 `regex_labels` 패턴 사용
   - `phone_number`: 전화번호 패턴
   - `email`: 이메일 패턴
   - `url`: URL 패턴
   - `date`: 날짜 패턴

2. **키워드 기반 라벨링**: 특정 키워드 다음에 오는 단어를 엔티티로 간주
   - `person_name`: "대표", "사장", "이름" 등 다음의 단어
   - `company_name`: "회사", "기관", "법인" 등 다음의 단어
   - `address`: "주소", "소재지", "위치" 등 다음의 단어

3. **자동 저장**: 각 라벨별로 `ner_labels/{label}.jsonl` 파일에 추가

## 방법 2: 기존 추출 결과를 학습 데이터로 변환

### 스크립트 작성 예시

```python
# scripts/convert_extracted_to_training_data.py
from module import api
from module.extractor import text as text_module
from module.extractor.ner.weak_supervision import WeakSupervisionLabeler
from pathlib import Path
import json

# 1. 문서에서 메타데이터 추출
result = api.metadata_extract(file_path="data/in/document/sample.pdf")

# 2. 원본 텍스트와 추출된 메타데이터 사용
raw_text = result.get("raw_text", "")
struct = text_module.read_text(raw_text)
sentences = struct["sentences"]
tokens_list = struct["tokens"]

# 3. 추출된 Decision을 기반으로 라벨링
labeler = WeakSupervisionLabeler(output_dir="configs/training/ner_labels")

for sentence in sentences:
    sent_id = sentence.get("sent_id")
    if sent_id is None:
        continue
    
    sent_tokens = [t.get("text", "") for t in tokens_list 
                  if t.get("sent_id") == sent_id]
    
    # 해당 문장에서 추출된 Decision 찾기
    sent_decisions = [d for d in result.get("decisions", []) 
                     if d.sent_id == sent_id]
    
    if not sent_decisions:
        continue
    
    # Decision을 BIO 라벨로 변환
    labels = ["O"] * len(sent_tokens)
    for decision in sent_decisions:
        # decision.value가 sent_tokens에서 어디에 있는지 찾기
        value_tokens = decision.value.split()
        for i in range(len(sent_tokens) - len(value_tokens) + 1):
            if sent_tokens[i:i+len(value_tokens)] == value_tokens:
                labels[i] = f"B-{decision.label}"
                for j in range(i+1, i+len(value_tokens)):
                    labels[j] = f"I-{decision.label}"
                break
    
    # 파일에 저장
    labeler.append_jsonl(
        f"{decision.label}.jsonl",
        f"extracted_{sent_id}",
        sent_tokens,
        labels
    )
```

## 방법 3: 수동으로 데이터 추가

### 직접 jsonl 파일에 추가

각 라벨별 파일에 직접 추가할 수 있습니다:

```bash
# address.jsonl에 추가
echo '{"id":"manual_001","tokens":["서울시","강남구","역삼동","123"],"labels":["B-address","I-address","I-address","I-address"]}' >> configs/training/ner_labels/address.jsonl

# company_name.jsonl에 추가
echo '{"id":"manual_002","tokens":["주식회사","카카오"],"labels":["B-company_name","I-company_name"]}' >> configs/training/ner_labels/company_name.jsonl
```

## 방법 4: 배치 처리로 대량 생성

### 여러 문서를 한 번에 처리

```python
# scripts/batch_generate_training_data.py
from pathlib import Path
from scripts.generate_training_data_from_documents import generate_training_data_from_documents

# 여러 디렉토리 처리
input_dirs = [
    "data/in/document",
    "data/in/contracts",
    "data/in/forms",
]

for input_dir in input_dirs:
    if Path(input_dir).exists():
        print(f"\n처리 중: {input_dir}")
        stats = generate_training_data_from_documents(
            input_dir=input_dir,
            output_dir="configs/training/ner_labels",
            min_tokens=3,
            max_samples_per_file=50,  # 파일당 샘플 수 제한
        )
        print(f"생성된 샘플: {sum(stats.values())}개")
```

## 데이터 품질 개선 팁

### 1. 정규식 패턴 개선

`configs/labels.yaml`의 `regex_labels`를 더 정확하게 수정:

```yaml
regex_labels:
  phone: r'\d{2,3}-\d{3,4}-\d{4}|\d{3}-\d{4}-\d{4}|\(\d{2,3}\)\s*\d{3,4}-\d{4}|010-\d{4}-\d{4}'
  email: r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
  # ... 더 많은 패턴 추가
```

### 2. 키워드 확장

스크립트의 `keyword_patterns`에 더 많은 키워드 추가:

```python
keyword_patterns = {
    "person_name": ["대표", "사장", "이름", "대표이사", "대표자", "담당자", "작성자", "성명"],
    "company_name": ["회사", "기관", "법인", "주식회사", "(주)", "㈜", "기업", "단체", "조합"],
    "address": ["주소", "소재지", "위치", "본사", "사무소", "도로명", "지번", "주소지"],
}
```

### 3. 수동 검수 및 필터링

자동 생성된 데이터는 품질이 낮을 수 있으므로:

1. **샘플 수 제한**: `--max_samples_per_file`로 파일당 샘플 수 제한
2. **수동 검수**: 생성된 데이터 중 일부를 확인하고 잘못된 것 제거
3. **점진적 추가**: 처음에는 적은 수로 시작, 모델 학습 후 품질 확인

## 워크플로우 예시

```bash
# 1단계: 문서에서 자동 생성
python scripts/generate_training_data_from_documents.py \
    --input_dir data/in/document \
    --max_samples_per_file 20

# 2단계: 생성된 데이터 확인
wc -l configs/training/ner_labels/*.jsonl

# 3단계: 일부 샘플 확인 (선택사항)
head -5 configs/training/ner_labels/address.jsonl | python -m json.tool

# 4단계: 모델 학습
python main.py

# 5단계: 결과 확인 후 필요시 더 많은 데이터 추가
```

## 주의사항

1. **중복 제거**: 같은 샘플이 여러 번 추가될 수 있음 (현재는 중복 체크 없음)
2. **품질 관리**: 자동 생성된 데이터는 오류가 있을 수 있으므로 정기적으로 검수 필요
3. **라벨 충돌**: 한 문장에 여러 라벨이 있을 때 우선순위 고려 필요

## 고급: 커스텀 라벨링 함수 추가

`weak_supervision.py`의 `WeakSupervisionLabeler` 클래스를 확장하여 더 정교한 라벨링 함수를 추가할 수 있습니다:

```python
class CustomWeakSupervisionLabeler(WeakSupervisionLabeler):
    def label_with_context(self, tokens, context_keywords):
        """컨텍스트 키워드를 사용한 라벨링"""
        # 커스텀 로직 구현
        pass
```
