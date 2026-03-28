# 항목별 신뢰도 (Item Reliability) 및 증거연결 (Evidence Connection)

이 문서는 메타데이터 추출 시스템에서 각 항목(필드)별 신뢰도 계산과 증거 연결 메커니즘에 대해 설명합니다.

## 목차

1. [항목별 신뢰도 (Item Reliability)](#항목별-신뢰도-item-reliability)
2. [증거연결 (Evidence Connection)](#증거연결-evidence-connection)
3. [현재 구현 상태](#현재-구현-상태)
4. [신뢰도 계산 방법](#신뢰도-계산-방법)
5. [증거 생성 및 연결](#증거-생성-및-연결)
6. [개선 제안](#개선-제안)

---

## 항목별 신뢰도 (Item Reliability)

### 정의

**항목별 신뢰도**는 각 메타데이터 필드(항목)에 대해 추출된 값의 신뢰성을 0.0부터 1.0까지의 점수로 나타낸 것입니다. 이는 해당 필드의 값이 얼마나 정확하고 신뢰할 수 있는지를 나타냅니다.

### 신뢰도에 영향을 미치는 요소

1. **추출 소스 일치 여부**
   - LLM과 NER이 동일한 값 추출 → 높은 신뢰도 (0.9-1.0)
   - LLM과 NER이 다른 값 추출 → 중간 신뢰도 (0.7-0.9)
   - 한 소스만 추출 → 낮은 신뢰도 (0.5-0.8)

2. **OCR 텍스트 검증**
   - 추출된 값이 OCR 텍스트에 명확히 존재 → 신뢰도 증가
   - 추출된 값이 OCR 텍스트에 없거나 모호함 → 신뢰도 감소

3. **필드 타입 및 우선순위**
   - 필수 필드 (required fields) → 높은 우선순위
   - 선택 필드 (optional fields) → 낮은 우선순위
   - 필드 매핑 우선순위 점수 반영

4. **값의 형식 및 유효성**
   - 형식 검증 통과 (날짜, 전화번호 등) → 신뢰도 증가
   - 형식 검증 실패 → 신뢰도 감소

5. **모델 신뢰도**
   - LLM 모델의 기본 신뢰도
   - NER 모델의 예측 확률

---

## 증거연결 (Evidence Connection)

### 정의

**증거연결**은 추출된 각 메타데이터 항목이 원본 문서(OCR 텍스트)의 어느 부분에서 추출되었는지를 연결하고, 추출 결정의 근거를 제공하는 메커니즘입니다.

### 증거의 구성 요소

1. **OCR 텍스트 발췌 (OCR Excerpt)**
   - 추출된 값이 발견된 원본 텍스트 부분
   - 주변 맥락(context) 포함
   - 문자 위치 정보 (선택적)

2. **추출 소스 정보**
   - LLM 추출 값
   - NER 추출 값
   - 최종 선택된 값

3. **결정 근거 (Reasoning)**
   - 왜 해당 값이 선택되었는지 설명
   - 한국어로 작성된 설명

4. **신뢰도 점수**
   - 해당 항목의 신뢰도 점수
   - 결정 타입별 기본 신뢰도

---

## 현재 구현 상태

### 1. 신뢰도 계산 (`api/module/consolidator/consolidation_agent.py`)

#### A. 필드별 신뢰도 계산

```python
# 결정 타입별 기본 신뢰도
- AGREED (일치): 0.9-1.0
- CONFLICT (충돌 해결): 0.7-0.9
- LLM_ONLY (LLM만): 0.5-0.7
- NER_ONLY (NER만): 0.6-0.8
- MISSING (누락): 0.0
```

#### B. 전체 신뢰도 계산

```python
# 평균 신뢰도 계산
confidences = [d.get('confidence', 0.0) for d in decisions if d.get('confidence')]
overall_confidence = sum(confidences) / len(confidences) if confidences else 0.7
```

#### C. 필드 매핑 신뢰도 (`api/module/consolidator/field_mapper.py`)

```python
def _calculate_field_confidence(
    entity_text: str,
    entity_type: str,
    field_name: str,
    llm_metadata: Dict[str, Any],
    ocr_text: str,
    document_type: str
) -> float:
    confidence = 0.0
    
    # 1. 우선순위 점수 (0.0-1.0)
    if field_name in priorities:
        confidence += priorities[field_name] / 10.0
    
    # 2. 값 일치 여부 (0.0-0.4)
    if llm_value and values_match(entity_text, llm_value):
        confidence += 0.4  # 높은 부스트
    
    # 3. 필드 존재 여부 (0.0-0.1)
    if llm_value:
        confidence += 0.1
    
    return min(confidence, 1.0)
```

### 2. 증거 생성 (`api/module/consolidator/reasoning_generator.py`)

#### A. 증거 객체 구조

```python
evidence = {
    "field": "field_name",
    "llm_value": "value_from_llm",
    "ner_value": "value_from_ner",
    "final_value": "selected_value",
    "decision": "AGREED|CONFLICT|LLM_ONLY|NER_ONLY|MISSING",
    "confidence": 0.95,
    "reasoning": "LLM과 NER 모두 '집건에'를 추출하여 일치함",
    "ocr_excerpt": "... 계약자: 집건에 ..."
}
```

#### B. OCR 발췌 추출

```python
def _extract_ocr_excerpt(
    field_name: str,
    ocr_text: str,
    context_window: int = 50
) -> Optional[str]:
    # 필드명을 OCR 텍스트에서 찾기
    idx = ocr_text.lower().find(field_name.lower())
    
    # 주변 맥락 추출 (앞뒤 50자)
    start = max(0, idx - context_window)
    end = min(len(ocr_text), idx + len(field_name) + context_window)
    
    return ocr_text[start:end]
```

---

## 신뢰도 계산 방법

### 1. LLM 추출 신뢰도

#### A. 기본 신뢰도 계산 (`api/module/llm_extraction/models/base_extractor.py`)

```python
def _calculate_confidence(metadata: Dict[str, Any], schema: Dict[str, Any]) -> float:
    """완성도 기반 신뢰도 계산"""
    total_fields = len(schema.get('properties', {}))
    filled_fields = sum(1 for v in metadata.values() if v is not None and v != "")
    
    if total_fields == 0:
        return 1.0
    
    # 채워진 필드 비율 = 신뢰도
    return min(filled_fields / total_fields, 1.0)
```

**특징:**
- 필드 완성도 기반
- 채워진 필드가 많을수록 높은 신뢰도
- 최대 1.0

#### B. 체크박스 추출 신뢰도 (`api/module/llm_extraction/extractors/checkbox_extractor.py`)

```python
def calculate_confidence(checkbox_data: Dict) -> float:
    """체크박스 데이터 완성도 기반 신뢰도"""
    total_fields = sum(len(category) for category in checkbox_data.values())
    if total_fields == 0:
        return 0.0
    
    # 데이터 완성도 정규화
    return min(1.0, total_fields / 10.0)
```

### 2. NER 추출 신뢰도

#### A. 모델 예측 확률

```python
# BERT 모델의 토큰별 예측 확률
predictions = torch.softmax(outputs.logits, dim=-1)
confidence_scores = torch.max(predictions, dim=-1)[0]

# 엔티티별 평균 신뢰도
entity_confidence = mean(confidence_scores[entity_tokens])
```

#### B. 패턴 매칭 보정

```python
# 패턴 기반 감지 시 신뢰도 부스트
if detected_type and pattern_confidence > 0.7:
    combined_confidence = max(type_prob, pattern_confidence * type_prob)
    if combined_confidence > 0.1:
        confidence = max(confidence, combined_confidence)
```

### 3. 통합 신뢰도 (Consolidation)

#### A. 결정 타입별 신뢰도

| 결정 타입 | 신뢰도 범위 | 설명 |
|----------|------------|------|
| **AGREED** | 0.9-1.0 | LLM과 NER이 동일한 값 추출 |
| **CONFLICT** | 0.7-0.9 | 값이 다르지만 OCR 검증으로 해결 |
| **LLM_ONLY** | 0.5-0.7 | LLM만 추출 (추측 가능성) |
| **NER_ONLY** | 0.6-0.8 | NER만 추출 (패턴 기반) |
| **MISSING** | 0.0 | 둘 다 추출 실패 |

#### B. 통합 신뢰도 계산

```python
# 1. 각 필드의 신뢰도 수집
field_confidences = [decision['confidence'] for decision in decisions]

# 2. 가중 평균 계산 (필수 필드에 더 높은 가중치)
required_fields = schema.get('required', [])
weights = [2.0 if field in required_fields else 1.0 for field in fields]

weighted_sum = sum(conf * weight for conf, weight in zip(field_confidences, weights))
total_weight = sum(weights)

overall_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
```

---

## 증거 생성 및 연결

### 1. 증거 객체 생성 (`ReasoningGenerator`)

```python
def generate_evidence(
    field_name: str,
    llm_value: Any,
    ner_value: Any,
    final_value: Any,
    decision: str,
    ocr_text: str = "",
    confidence: float = 0.0
) -> Dict[str, Any]:
    """증거 객체 생성"""
    return {
        "field": field_name,
        "llm_value": llm_value,
        "ner_value": ner_value,
        "final_value": final_value,
        "decision": decision,
        "confidence": confidence,
        "reasoning": self._generate_reasoning(...),
        "ocr_excerpt": self._extract_ocr_excerpt(field_name, ocr_text)
    }
```

### 2. OCR 발췌 추출

#### A. 기본 방법 (현재 구현)

```python
def _extract_ocr_excerpt(field_name: str, ocr_text: str, context_window: int = 50):
    # 1. 필드명 검색
    idx = ocr_text.lower().find(field_name.lower())
    
    # 2. 주변 맥락 추출
    start = max(0, idx - context_window)
    end = min(len(ocr_text), idx + len(field_name) + context_window)
    
    # 3. 발췌 반환
    return ocr_text[start:end]
```

#### B. 개선된 방법 (제안)

```python
def _extract_ocr_excerpt_enhanced(
    field_name: str,
    extracted_value: str,
    ocr_text: str,
    context_window: int = 100
) -> Dict[str, Any]:
    """향상된 OCR 발췌 추출"""
    
    # 1. 필드명으로 검색
    field_matches = list(re.finditer(re.escape(field_name), ocr_text, re.IGNORECASE))
    
    # 2. 추출된 값으로 검색
    value_matches = list(re.finditer(re.escape(extracted_value), ocr_text, re.IGNORECASE))
    
    # 3. 가장 가까운 매치 찾기
    best_match = None
    min_distance = float('inf')
    
    for field_match in field_matches:
        for value_match in value_matches:
            distance = abs(field_match.start() - value_match.start())
            if distance < min_distance:
                min_distance = distance
                best_match = (field_match, value_match)
    
    if best_match:
        field_match, value_match = best_match
        # 두 매치를 포함하는 범위 계산
        start = min(field_match.start(), value_match.start()) - context_window
        end = max(field_match.end(), value_match.end()) + context_window
        
        return {
            "excerpt": ocr_text[max(0, start):min(len(ocr_text), end)],
            "field_position": field_match.span(),
            "value_position": value_match.span(),
            "distance": min_distance
        }
    
    return None
```

### 3. 결정 근거 생성

#### A. 결정 타입별 근거

```python
def _generate_reasoning(field_name, llm_value, ner_value, final_value, decision):
    if decision == "AGREED":
        return f"LLM과 NER 모두 '{final_value}' 값을 추출했습니다. 높은 신뢰도입니다."
    
    elif decision == "CONFLICT":
        return f"LLM은 '{llm_value}', NER은 '{ner_value}'를 추출했습니다. OCR 텍스트 검증 결과 '{final_value}'를 선택했습니다."
    
    elif decision == "LLM_ONLY":
        return f"NER에서 해당 필드를 찾을 수 없어 LLM 값 '{final_value}'를 사용했습니다."
    
    elif decision == "NER_ONLY":
        return f"LLM에서 해당 필드를 찾을 수 없어 NER 값 '{final_value}'를 사용했습니다."
    
    else:
        return f"필드 '{field_name}'에 대한 정보를 찾을 수 없습니다."
```

---

## 개선 제안

### 1. 항목별 신뢰도 개선

#### A. 다차원 신뢰도 계산

```python
def calculate_item_reliability(
    field_name: str,
    llm_value: Any,
    ner_value: Any,
    final_value: Any,
    decision: str,
    ocr_text: str,
    schema: Dict[str, Any]
) -> Dict[str, float]:
    """다차원 신뢰도 계산"""
    
    reliability = {
        "source_agreement": 0.0,      # 소스 일치도
        "ocr_verification": 0.0,      # OCR 검증도
        "format_validity": 0.0,       # 형식 유효성
        "field_priority": 0.0,        # 필드 우선순위
        "context_relevance": 0.0,     # 맥락 관련성
        "overall": 0.0                # 전체 신뢰도
    }
    
    # 1. 소스 일치도
    if decision == "AGREED":
        reliability["source_agreement"] = 1.0
    elif decision == "CONFLICT":
        reliability["source_agreement"] = 0.5
    elif decision in ["LLM_ONLY", "NER_ONLY"]:
        reliability["source_agreement"] = 0.3
    else:
        reliability["source_agreement"] = 0.0
    
    # 2. OCR 검증도
    if final_value and ocr_text:
        if final_value in ocr_text:
            reliability["ocr_verification"] = 1.0
        elif any(word in ocr_text for word in final_value.split()):
            reliability["ocr_verification"] = 0.7
        else:
            reliability["ocr_verification"] = 0.3
    
    # 3. 형식 유효성
    field_type = schema.get('properties', {}).get(field_name, {}).get('type')
    if field_type == 'date':
        if re.match(r'\d{4}-\d{2}-\d{2}', str(final_value)):
            reliability["format_validity"] = 1.0
    elif field_type == 'phone':
        if re.match(r'^[0-9\-]+$', str(final_value)):
            reliability["format_validity"] = 1.0
    
    # 4. 필드 우선순위
    required_fields = schema.get('required', [])
    if field_name in required_fields:
        reliability["field_priority"] = 1.0
    else:
        reliability["field_priority"] = 0.5
    
    # 5. 맥락 관련성 (향후 구현)
    reliability["context_relevance"] = 0.8  # 기본값
    
    # 6. 전체 신뢰도 (가중 평균)
    weights = {
        "source_agreement": 0.3,
        "ocr_verification": 0.3,
        "format_validity": 0.2,
        "field_priority": 0.1,
        "context_relevance": 0.1
    }
    
    reliability["overall"] = sum(
        reliability[key] * weights[key] 
        for key in weights.keys()
    )
    
    return reliability
```

#### B. 신뢰도 등급 시스템

```python
def get_reliability_grade(confidence: float) -> str:
    """신뢰도 등급 반환"""
    if confidence >= 0.9:
        return "매우 높음 (Very High)"
    elif confidence >= 0.7:
        return "높음 (High)"
    elif confidence >= 0.5:
        return "보통 (Medium)"
    elif confidence >= 0.3:
        return "낮음 (Low)"
    else:
        return "매우 낮음 (Very Low)"
```

### 2. 증거연결 개선

#### A. 다중 증거 연결

```python
def generate_evidence_chain(
    field_name: str,
    final_value: Any,
    ocr_text: str,
    llm_result: Dict,
    ner_result: Dict
) -> Dict[str, Any]:
    """증거 체인 생성"""
    
    evidence_chain = {
        "field": field_name,
        "final_value": final_value,
        "evidence_sources": [],
        "evidence_strength": 0.0
    }
    
    # 1. LLM 증거
    if llm_result.get('metadata', {}).get(field_name):
        llm_evidence = {
            "source": "LLM",
            "value": llm_result['metadata'][field_name],
            "confidence": llm_result.get('confidence', 0.0),
            "ocr_excerpt": extract_ocr_excerpt(field_name, ocr_text),
            "model_used": llm_result.get('model_used', 'unknown')
        }
        evidence_chain["evidence_sources"].append(llm_evidence)
    
    # 2. NER 증거
    ner_entities = ner_result.get('extracted_entities', [])
    for entity_text, entity_type in ner_entities:
        if entity_text == final_value:
            ner_evidence = {
                "source": "NER",
                "value": entity_text,
                "entity_type": entity_type,
                "confidence": 0.8,  # NER 기본 신뢰도
                "ocr_excerpt": extract_ocr_excerpt(entity_text, ocr_text),
                "model_used": ner_result.get('model_name', 'unknown')
            }
            evidence_chain["evidence_sources"].append(ner_evidence)
    
    # 3. OCR 직접 검색 증거
    if final_value in ocr_text:
        ocr_evidence = {
            "source": "OCR_DIRECT",
            "value": final_value,
            "confidence": 1.0,
            "ocr_excerpt": extract_ocr_excerpt(final_value, ocr_text),
            "position": ocr_text.find(final_value)
        }
        evidence_chain["evidence_sources"].append(ocr_evidence)
    
    # 4. 증거 강도 계산
    if len(evidence_chain["evidence_sources"]) >= 2:
        evidence_chain["evidence_strength"] = 0.9
    elif len(evidence_chain["evidence_sources"]) == 1:
        evidence_chain["evidence_strength"] = 0.6
    else:
        evidence_chain["evidence_strength"] = 0.0
    
    return evidence_chain
```

#### B. 증거 시각화

```python
def visualize_evidence(evidence_chain: Dict[str, Any]) -> str:
    """증거 시각화 (HTML/마크다운)"""
    
    html = f"""
    <div class="evidence-chain">
        <h3>필드: {evidence_chain['field']}</h3>
        <p>최종 값: <strong>{evidence_chain['final_value']}</strong></p>
        <p>증거 강도: {evidence_chain['evidence_strength']:.2f}</p>
        
        <h4>증거 소스:</h4>
        <ul>
    """
    
    for source in evidence_chain['evidence_sources']:
        html += f"""
        <li>
            <strong>{source['source']}</strong>: {source['value']}
            <br>신뢰도: {source['confidence']:.2f}
            <br>OCR 발췌: <code>{source['ocr_excerpt']}</code>
        </li>
        """
    
    html += """
        </ul>
    </div>
    """
    
    return html
```

### 3. 통합 출력 형식

```json
{
  "consolidated_metadata": {
    "rights_holder": "집건에",
    "user": "국립생태원",
    "work_title": "멸종위기 야생생물 대국민 온라인 홍보물"
  },
  "item_reliability": {
    "rights_holder": {
      "overall": 0.95,
      "source_agreement": 1.0,
      "ocr_verification": 1.0,
      "format_validity": 1.0,
      "field_priority": 1.0,
      "grade": "매우 높음"
    },
    "user": {
      "overall": 0.92,
      "source_agreement": 1.0,
      "ocr_verification": 0.9,
      "format_validity": 1.0,
      "field_priority": 1.0,
      "grade": "매우 높음"
    }
  },
  "evidence_connections": {
    "rights_holder": {
      "final_value": "집건에",
      "evidence_sources": [
        {
          "source": "LLM",
          "value": "집건에",
          "confidence": 0.95,
          "ocr_excerpt": "... 저작자 및 저작권 이용허락자 집건에 (이하 \"권리자\" ...",
          "position": 45
        },
        {
          "source": "NER",
          "value": "집건에",
          "entity_type": "COMPANY",
          "confidence": 0.88,
          "ocr_excerpt": "... 저작자 및 저작권 이용허락자 집건에 (이하 \"권리자\" ...",
          "position": 45
        }
      ],
      "evidence_strength": 0.95,
      "reasoning": "LLM과 NER 모두 '집건에'를 추출하여 일치함. OCR 텍스트에서도 확인됨."
    }
  },
  "validation_report": {
    "overall_confidence": 0.93,
    "total_fields": 15,
    "high_reliability_fields": 12,
    "medium_reliability_fields": 2,
    "low_reliability_fields": 1
  }
}
```

---

## 사용 예시

### 1. 신뢰도 기반 필터링

```python
def filter_by_reliability(metadata: Dict, min_reliability: float = 0.7):
    """신뢰도 기준으로 필터링"""
    filtered = {}
    for field, value in metadata.items():
        reliability = item_reliability.get(field, {}).get('overall', 0.0)
        if reliability >= min_reliability:
            filtered[field] = value
    return filtered
```

### 2. 증거 기반 검증

```python
def verify_with_evidence(field_name: str, value: Any, evidence_chain: Dict):
    """증거 체인으로 값 검증"""
    # 증거 소스가 2개 이상이면 높은 신뢰도
    if len(evidence_chain['evidence_sources']) >= 2:
        return True, "다중 소스 일치"
    
    # OCR 직접 검색으로 확인되면 신뢰
    for source in evidence_chain['evidence_sources']:
        if source['source'] == 'OCR_DIRECT':
            return True, "OCR 직접 확인"
    
    return False, "증거 부족"
```

---

## 요약

### 항목별 신뢰도 (Item Reliability)

- **목적**: 각 메타데이터 필드의 추출 신뢰성 평가
- **범위**: 0.0 (신뢰 불가) ~ 1.0 (매우 신뢰)
- **요소**: 소스 일치도, OCR 검증, 형식 유효성, 필드 우선순위
- **활용**: 필터링, 우선순위 정렬, 품질 평가

### 증거연결 (Evidence Connection)

- **목적**: 추출된 값의 출처와 근거 제공
- **구성**: OCR 발췌, 추출 소스, 결정 근거, 위치 정보
- **활용**: 검증, 디버깅, 사용자 설명, 감사(audit)

### 현재 구현

✅ **구현됨:**
- 기본 신뢰도 계산 (결정 타입 기반)
- OCR 발췌 추출
- 결정 근거 생성
- 증거 객체 생성

🔄 **개선 필요:**
- 다차원 신뢰도 계산
- 향상된 OCR 발췌 추출
- 증거 체인 생성
- 신뢰도 등급 시스템

---

이 문서는 메타데이터 추출 시스템의 신뢰도와 증거 메커니즘에 대한 종합적인 가이드입니다.

