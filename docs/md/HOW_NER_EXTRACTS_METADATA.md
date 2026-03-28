# How the NER Model Extracts Metadata

This document explains the complete flow of how the NER (Named Entity Recognition) model is used to extract metadata from documents in this project.

## Overview

The NER system uses a **dual extraction approach**:
1. **BERT-based model** with BIO tagging (primary method)
2. **Regex pattern matching** (backup/fallback method)

Both methods work together to extract entities from OCR text, which are then mapped to metadata fields and optionally consolidated with LLM extraction results.

---

## Complete Extraction Flow

### 1. **Entry Points**

The NER model can be called from multiple places:

#### A. Direct API Call (`api/api.py`)
```python
ner_result = ner_predict(
    input_path=str(ocr_dir),
    output_path=str(output_path),
    model_name=model_name,
    confidence_threshold=0.85
)
```

#### B. Web API Endpoint (`api/web/app.py`)
- **Endpoint**: `/api/llm-extract` (includes NER extraction)
- **Flow**: PDF → OCR → NER → LLM → Consolidation

#### C. Integrated Pipeline (`api/api.py::process_pdf_to_ner`)
- Complete pipeline: PDF → Images → OCR → NER

---

## 2. **NER Prediction Process** (`api/module/ner/ner_system.py`)

### Step 1: Model Loading

```python
def load_model_and_tokenizer(model_path: Path, verbose: bool = True):
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    # 2. Load label mapping (B-I-O tags)
    # Labels: O, B-NAME, I-NAME, B-PHONE, I-PHONE, etc.
    
    # 3. Load model (BERT-CRF or standard HuggingFace)
    model = AutoModelForTokenClassification.from_pretrained(str(model_path))
    
    # 4. Move to GPU if available
    model.to(device)
    model.eval()
```

**Model Path Resolution**:
- Checks `models/ner/{model_name}/`
- Falls back to `model_downloaded/{model_name}/`
- Downloads from Hugging Face if not found

### Step 2: Text Preprocessing

```python
def split_text_smartly(text: str, max_length: int = 512) -> List[str]:
    # Split text into sentences/chunks
    # Each chunk ≤ 512 tokens (BERT limit)
    # Preserves sentence boundaries
```

### Step 3: Dual Extraction

#### A. BERT-Based Extraction (`extract_entities_by_bio_tagging`)

**Process**:
1. **Tokenization**: Convert text to tokens
   ```python
   encoding = tokenizer(
       sentence,
       return_tensors="pt",
       truncation=True,
       max_length=512,
       return_offsets_mapping=True
   )
   ```

2. **Model Prediction**: Get token-level predictions
   ```python
   outputs = model(input_ids=input_ids, attention_mask=attention_mask)
   predictions = torch.argmax(outputs.logits, dim=2)
   ```

3. **BIO Tag Processing**:
   - **B-{TYPE}**: Beginning of entity (e.g., `B-NAME`)
   - **I-{TYPE}**: Inside entity (e.g., `I-NAME`)
   - **O**: Outside (not an entity)

4. **Entity Reconstruction**:
   ```python
   if tag.startswith('B-'):
       # Start new entity
       current_entity = {'label': tag[2:], 'text': token}
   elif tag.startswith('I-') and current_type == tag[2:]:
       # Continue entity
       current_entity['text'] += token
   elif tag == 'O':
       # End entity, save it
       entities.append(current_entity)
   ```

5. **Pattern-Based Correction**:
   - Uses regex patterns to validate/correct entity types
   - Examples:
     - Phone: `\d{2,3}-\d{3,4}-\d{4}`
     - Email: `@` symbol detection
     - Date: `\d{4}년`, `\d{1,2}월`, `\d{1,2}일`
     - Address: Contains `시`, `도`, `구`, `동`, `로`, `길`

#### B. Regex-Based Extraction (`extract_entities_by_regex`)

**Fallback method** when model is unavailable or for validation:

```python
# Phone patterns
phone_patterns = [
    r'(\d{2,3}-\d{3,4}-\d{4})',
    r'전화번호:\s*([0-9-]{10,15})',
    r'연락처:\s*([0-9-]{10,15})'
]

# Name patterns
name_patterns = [
    r'성명:\s*([가-힣]{2,4})',
    r'이름:\s*([가-힣]{2,4})',
    r'계약자:\s*([가-힣]{2,4})'
]

# Address patterns
address_patterns = [
    r'주소:\s*([가-힣0-9\s-]+(?:시|구|군|동|로|길)[가-힣0-9\s-]*)',
    r'([가-힣]+(?:시|도)\s+[가-힣]+(?:구|군)\s+[가-힣0-9\s-]*)'
]
```

### Step 4: Entity Validation

```python
def is_valid_entity(entity: str, entity_type: Optional[str] = None) -> bool:
    # Basic checks:
    # - Length: 2-50 characters
    # - No invalid characters (□, ☑, ○, ●)
    # - Not just digits
    # - Type-specific validation
```

**Type-Specific Validation**:
- **NAME**: Excludes role words like "양도자", "양수인"
- **PHONE**: Must have 7+ digits
- **COMPANY**: Must be 4+ characters (except known abbreviations)
- **ADDRESS**: Max 1 newline

### Step 5: Entity Deduplication

```python
# Remove duplicates (case-insensitive)
seen_entities = set()
for entity, label in all_entities:
    entity_lower = entity.lower().strip()
    if entity_lower not in seen_entities:
        seen_entities.add(entity_lower)
        final_entities.append((entity, label))
```

---

## 3. **Entity Types Supported**

The NER model recognizes **23 entity types**:

```python
ENTITY_TYPES = [
    "NAME",           # 이름
    "PHONE",          # 전화번호
    "ADDRESS",        # 주소
    "DATE",           # 날짜
    "COMPANY",        # 회사/기관명
    "EMAIL",          # 이메일
    "POSITION",       # 직책/직위
    "CONTRACT_TYPE",  # 계약서 유형
    "MONEY",          # 금액
    "PERIOD",         # 기간
    "ID_NUM",         # 신분증번호
    "CONSENT_TYPE",   # 동의서 유형
    "RIGHT_INFO",     # 권리정보
    "PROJECT_NAME",   # 사업명
    "LAW_REFERENCE",  # 법령 근거
    "TITLE",          # 제목
    "URL",            # URL정보
    "DESCRIPTION",    # 설명
    "TYPE",           # 유형
    "STATUS",         # 상태
    "DEPARTMENT",     # 부서정보
    "LANGUAGE",       # 언어
    "QUANTITY"        # 수량정보
]
```

---

## 4. **Integration with LLM Extraction**

### A. Field Mapping (`api/module/consolidator/field_mapper.py`)

NER entities are mapped to LLM metadata fields:

```python
# Example mappings:
"양도인성명" → ["rights_holder", "assignor_name", "data_subject"]
"양수인기관명" → ["user", "assignee_organization", "data_controller"]
"저작물명" → ["work_title", "title", "document_title"]
```

### B. Consolidation (`api/module/consolidator/consolidation_agent.py`)

The consolidation agent:
1. **Maps** NER entities to LLM fields
2. **Compares** LLM and NER values
3. **Merges** results using Qwen3-Next-80B model
4. **Validates** final metadata

**Decision Types**:
- `LLM_ONLY`: Only LLM extracted (confidence: 0.7-0.9)
- `NER_ONLY`: Only NER extracted (confidence: 0.6-0.8)
- `BOTH_MATCH`: Both match (confidence: 0.9-1.0)
- `BOTH_DIFFER`: Both differ, LLM preferred (confidence: 0.7-0.8)

---

## 5. **Output Format**

### JSON Structure

```json
{
  "file": "path/to/ocr_text.txt",
  "entities": {
    "NAME": ["김민수", "박영희"],
    "PHONE": ["010-1234-5678"],
    "ADDRESS": ["서울시 강남구 테헤란로 123"],
    "COMPANY": ["한국콘텐츠진흥원"],
    "DATE": ["2024년 1월 15일"]
  },
  "entity_count": 5,
  "entity_types": ["NAME", "PHONE", "ADDRESS", "COMPANY", "DATE"]
}
```

### Summary Statistics

```json
{
  "total_files_processed": 10,
  "total_entities_found": 45,
  "unique_entities": 40,
  "entity_types_count": {
    "NAME": 10,
    "PHONE": 8,
    "ADDRESS": 7,
    "COMPANY": 5,
    "DATE": 15
  },
  "processing_time": 12.5,
  "timestamp": "20241228_143022"
}
```

---

## 6. **Complete Example Flow**

### Input: OCR Text
```
저작물 저작재산권 양도 계약서

계약자: 김민수
전화번호: 010-1234-5678
이메일: minsu.kim@gmail.com
주소: 서울시 강남구 테헤란로 123

수탁기관: 한국콘텐츠진흥원
담당자: 박영희 부장
계약금: 5,000,000원
계약일: 2024년 1월 15일
```

### Step-by-Step Processing

1. **Text Splitting**: Split into sentences/chunks (≤512 tokens)

2. **Tokenization**: 
   ```
   ["저작물", "저작재산권", "양도", "계약서", ...]
   ```

3. **Model Prediction**:
   ```
   Token: "김민수" → B-NAME
   Token: "010" → B-PHONE
   Token: "-" → I-PHONE
   Token: "1234" → I-PHONE
   ...
   ```

4. **Entity Extraction**:
   ```python
   [
       ("김민수", "NAME"),
       ("010-1234-5678", "PHONE"),
       ("minsu.kim@gmail.com", "EMAIL"),
       ("서울시 강남구 테헤란로 123", "ADDRESS"),
       ("한국콘텐츠진흥원", "COMPANY"),
       ("박영희", "NAME"),
       ("부장", "POSITION"),
       ("5,000,000원", "MONEY"),
       ("2024년 1월 15일", "DATE")
   ]
   ```

5. **Validation**: Filter invalid entities

6. **Deduplication**: Remove duplicates

7. **Grouping**: Group by entity type

8. **Output**: Save to JSON files

---

## 7. **Key Features**

### A. Dual Extraction System
- **Primary**: BERT model with BIO tagging
- **Backup**: Regex patterns
- **Combined**: Results merged and deduplicated

### B. Pattern-Based Correction
- Validates entity types using regex
- Boosts confidence for pattern matches
- Corrects misclassified entities

### C. Smart Text Splitting
- Preserves sentence boundaries
- Handles long documents
- Maintains context

### D. Entity Validation
- Type-specific rules
- Length constraints
- Invalid character filtering
- Context-aware validation

### E. Model Flexibility
- Supports multiple models:
  - `klue/roberta-large` (default)
  - `google-bert/bert-base-multilingual-cased`
  - `xlm-roberta-large`
- Auto-downloads from Hugging Face
- Supports custom BERT-CRF models

---

## 8. **Performance Characteristics**

### Processing Speed
- **CPU**: ~0.5-1 second per document
- **GPU**: ~0.1-0.3 seconds per document
- **Batch Processing**: Parallel file processing

### Accuracy
- **BERT Model**: High accuracy for trained entities
- **Regex Fallback**: Good for structured data
- **Combined**: Best of both worlds

### Memory Usage
- **Model Loading**: ~500MB-2GB (depending on model)
- **Inference**: ~100-500MB per document
- **Batch Processing**: Scales with batch size

---

## 9. **Configuration**

### Model Selection
```python
# Available models
AVAILABLE_MODELS = {
    "klue-roberta-large": {
        "name": "klue/roberta-large",
        "description": "Korean RoBERTa Large"
    },
    "google-bert": {
        "name": "google-bert/bert-base-multilingual-cased",
        "description": "Multilingual BERT"
    },
    "xlm-roberta": {
        "name": "xlm-roberta-large",
        "description": "XLM-RoBERTa Large"
    }
}
```

### Confidence Threshold
- Default: `0.85`
- Lower = More entities (but more false positives)
- Higher = Fewer entities (but more precision)

---

## 10. **Integration Points**

### A. With OCR System
```python
# OCR → NER
ocr_result = ocr_google(image_dir, ocr_dir)
ner_result = ner_predict(ocr_dir, ner_dir, model_name="klue/roberta-large")
```

### B. With LLM Extraction
```python
# NER + LLM → Consolidation
llm_result = llm_extract_metadata(ocr_text, document_type)
ner_result = ner_predict(ocr_dir, ner_dir)
consolidated = consolidation_agent.consolidate(llm_result, ner_result, ocr_text)
```

### C. With Web API
```python
# POST /api/llm-extract
# Parameters:
# - ner_model: "klue-roberta-large"
# - consolidate: True
# Returns: Combined LLM + NER results
```

---

## Summary

The NER model extracts metadata through:

1. **Tokenization** of OCR text
2. **BERT-based prediction** with BIO tagging
3. **Regex pattern matching** as backup
4. **Entity validation** and filtering
5. **Deduplication** and grouping
6. **Field mapping** to LLM schema
7. **Consolidation** with LLM results (optional)

This dual approach ensures robust extraction even when the model is uncertain, and the integration with LLM extraction provides comprehensive metadata coverage.

