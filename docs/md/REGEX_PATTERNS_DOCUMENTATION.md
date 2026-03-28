# Regex Patterns Documentation

This document lists all regex patterns used by the NER model and LLM model for metadata extraction in the copyright metadata extraction project.

## Table of Contents
1. [NER Model Regex Patterns](#ner-model-regex-patterns)
2. [LLM Model Regex Patterns](#llm-model-regex-patterns)
3. [Validation & Utility Regex Patterns](#validation--utility-regex-patterns)

---

## NER Model Regex Patterns

### Contract Document Patterns (`ner/entity_extraction.py`, `ner/generate_training_data.py`, `ner/enhance_training_data.py`)

#### 저작물명 (Work Title)
- `r'저작물\s*명\s*[:：]?\s*([^\n,]+)'` - Basic pattern
- `r"저작물명\s*[:：]\s*([^\n]+)"` - With colon separator
- `r"○\s*저작물명\s*[:：]\s*([^\n]+)"` - With bullet point
- `r"저작물\s*[:：]\s*([^\n]+)"` - Simplified
- `r"저작물명\s*[:：]\s*([^\n\r,;]+?)(?:\s*[,;]\s*|\s*$|\s*[oO○●]\s*)"` - Enhanced with boundary detection
- `r"○\s*저작물명\s*[:：]\s*([^\n\r,;]+?)(?:\s*[,;]\s*|\s*$|\s*[oO○●]\s*)"` - Enhanced with bullet
- `r"저작물\s*[:：]\s*([^\n\r,;]+?)(?:\s*[,;]\s*|\s*$|\s*[oO○●]\s*)"` - Enhanced simplified
- `r"○\s*저작물\s*[:：]\s*([^\n\r,;]+?)(?:\s*[,;]\s*|\s*$|\s*[oO○●]\s*)"` - Enhanced with bullet and simplified

#### 대상 저작물 상세정보 (Target Work Details)
- `r'저작물\s*상세\s*정보\s*[:：]?\s*([^\n]+)'` - Basic pattern
- `r"대상\s*저작물\s*상세정보\s*[:：]\s*([^\n]+)"` - With colon separator
- `r"○\s*대상\s*저작물\s*상세정보\s*[:：]\s*([^\n]+)"` - With bullet point
- `r"대상\s*저작물\s*상세정보\s*[:：]\s*([^\n\r○●]+?)(?:\s*[□○●]\s*|\s*$)"` - Enhanced with boundary detection
- `r"○\s*대상\s*저작물\s*상세정보\s*[:：]\s*([^\n\r○●]+?)(?:\s*[□○●]\s*|\s*$)"` - Enhanced with bullet

#### 양수자 기관명 (Assignee Institution Name)
- `r'양수자\s*기관\s*명\s*[:：]?\s*([^\n,]+)'` - Basic pattern
- `r"양수자.*?기관명\s*[:：]\s*([^\n]+)"` - With non-greedy match
- `r"○\s*기관명\s*[:：]\s*([^\n]+)"` - With bullet point
- `r"기관명\s*[:：]\s*([^\n]+)"` - Simplified
- `r"양수자.*?기관명\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)"` - Enhanced with boundary detection
- `r"○\s*기관명\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)"` - Enhanced with bullet
- `r"기관명\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)"` - Enhanced simplified

#### 양수자 주소 (Assignee Address)
- `r'양수자\s*주소\s*[:：]?\s*([^\n]+)'` - Basic pattern
- `r"양수자.*?주소\s*[:：]\s*([^\n]+)"` - With non-greedy match
- `r"○\s*주소\s*[:：]\s*([^\n]+)"` - With bullet point
- `r"주소\s*[:：]\s*([^\n]+)"` - Simplified
- `r"양수자.*?주소\s*[:：]\s*([^\n\r○●]+?)(?:\s*[○●]\s*|\s*$)"` - Enhanced with boundary detection
- `r"○\s*주소\s*[:：]\s*([^\n\r○●]+?)(?:\s*[○●]\s*|\s*$)"` - Enhanced with bullet
- `r"주소\s*[:：]\s*([^\n\r○●]+?)(?:\s*[○●]\s*|\s*$)"` - Enhanced simplified

#### 양도자 기관(개인)명 (Assignor Institution/Individual Name)
- `r'양도자\s*기관\s*명\s*[:：]?\s*([^\n,]+)'` - Basic pattern
- `r"양도자.*?기관.*?명\s*[:：]\s*([^\n]+)"` - With non-greedy match
- `r"양도자.*?기관\s*[:：]\s*([^\n]+)"` - Simplified variant
- `r"양도자.*?기관.*?명\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)"` - Enhanced with boundary detection
- `r"양도자.*?기관\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)"` - Enhanced simplified
- `r"양도자\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)"` - Enhanced most simplified

#### 양도자 소속 (Assignor Affiliation)
- `r'양도자\s*소속\s*[:：]?\s*([^\n,]+)'` - Basic pattern
- `r"양도자.*?소속\s*[:：]\s*([^\n]+)"` - With non-greedy match
- `r"소속\s*[:：]\s*([^\n]+)"` - Simplified
- `r"양도자.*?소속\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)"` - Enhanced with boundary detection
- `r"소속\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)"` - Enhanced simplified

#### 양도자 대표주소 (Assignor Representative Address)
- `r'양도자\s*주소\s*[:：]?\s*([^\n]+)'` - Basic pattern
- `r"양도자.*?주소\s*[:：]\s*([^\n]+)"` - With non-greedy match
- `r"양도자.*?대표.*?주소\s*[:：]\s*([^\n]+)"` - With "대표" keyword
- `r"양도자.*?주소\s*[:：]\s*([^\n\r○●]+?)(?:\s*[○●]\s*|\s*$)"` - Enhanced with boundary detection
- `r"양도자.*?대표.*?주소\s*[:：]\s*([^\n\r○●]+?)(?:\s*[○●]\s*|\s*$)"` - Enhanced with "대표" keyword

#### 양도자 연락처 (Assignor Contact)
- `r'양도자\s*연락처\s*[:：]?\s*([\d\-]+)'` - Basic pattern (digits and hyphens only)
- `r"양도자.*?연락처\s*[:：]\s*([^\n]+)"` - With non-greedy match
- `r"양도자.*?전화\s*[:：]\s*([^\n]+)"` - With "전화" keyword
- `r"양도자.*?연락처\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)"` - Enhanced with boundary detection
- `r"양도자.*?전화\s*[:：]\s*([^\n\r,;○●]+?)(?:\s*[,;○●]\s*|\s*$)"` - Enhanced with "전화" keyword

#### 동의여부 (Consent Status)
- `r'동의\s*여부\s*[:：]?\s*([^\n,]+)'` - Basic pattern
- `r"동의함|동의|합의|승인"` - Direct match (no capture group)
- Used as fallback: checks if text contains "동의", "합의", "승인", or "체결"

#### 날짜 (Date)
- `r'날짜\s*[:：]?\s*([^\n]+)'` - Basic pattern
- `r"(\d{4}[\.\-\/년]\s*\d{1,2}[\.\-\/월]\s*\d{1,2}[일]?)"` - Date with Korean format (년, 월, 일)
- `r"(\d{4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2})"` - Date with dots (YYYY. MM. DD)

### Consent Document Patterns (`ner/entity_extraction.py`, `ner/generate_training_data.py`, `ner/enhance_training_data.py`)

#### 양수인 성명 (Assignee Name)
- `r'성\s*명\s*[:：]?\s*([^\s,]+)'` - Basic pattern
- `r"양도인\s*성명\s*[:：]\s*([^\s\n]+)"` - With "양도인" prefix
- `r"성명\s*[:：]\s*([^\s\n]+)"` - Simplified
- `r"양도인.*?성명.*?[:：]\s*([^\s\n]+)"` - With non-greedy match
- `r"양도인\s*성명\s*[:：]\s*([^\s\n\r(]+)"` - Enhanced excluding parentheses
- `r"성명\s*[:：]\s*([^\s\n\r(]+)"` - Enhanced simplified
- `r"양도인.*?성명.*?[:：]\s*([^\s\n\r(]+)"` - Enhanced with non-greedy match
- `r"양도자\s*본인은.*?([가-힣]{2,4})\s*\("` - Pattern for "양도자 본인은 [이름](" format
- `r"본인\s*([가-힣]{2,4})\s*\("` - Pattern for "본인 [이름](" format

#### 양도인 주소 (Assignor Address)
- `r'주\s*소\s*[:：]?\s*([^\n]+)'` - Basic pattern
- `r"양도인.*?주소\s*[:：]\s*([^\n]+)"` - With "양도인" prefix
- `r"주소\s*[:：]\s*([^\n]+)"` - Simplified
- `r"양도인.*?주소\s*[:：]\s*([^\n\r]+?)(?:\s*양수인|\s*전화|\s*$)"` - Enhanced with boundary detection
- `r"주소\s*[:：]\s*([^\n\r]+?)(?:\s*양수인|\s*전화|\s*$)"` - Enhanced simplified
- `r"주소.*?[:：]\s*([^\n\r]+?)(?:\s*양수인|\s*전화|\s*$)"` - Enhanced with non-greedy match

#### 양도인 전화번호 (Assignor Phone Number)
- `r'전화\s*번호\s*[:：]?\s*([\d\-]+)'` - Basic pattern (digits and hyphens only)
- `r"양도인.*?전화번호\s*[:：]\s*([^\s\n]+)"` - With "양도인" prefix
- `r"전화번호\s*[:：]\s*([^\s\n]+)"` - Simplified
- `r"전화\s*[:：]\s*([^\s\n]+)"` - Most simplified
- `r"(\d{3}-\d{4}-\d{4})"` - Korean phone format (XXX-XXXX-XXXX)
- `r"(\d{2,3}-\d{3,4}-\d{4})"` - Flexible Korean phone format
- `r"양도인.*?전화번호\s*[:：]\s*([0-9\-]+)"` - Enhanced with digits and hyphens only
- `r"전화번호\s*[:：]\s*([0-9\-]+)"` - Enhanced simplified
- `r"전화\s*[:：]\s*([0-9\-]+)"` - Enhanced most simplified

#### 양수인 기관명 (Assignee Institution Name)
- `r'기관\s*명\s*[:：]?\s*([^\n,]+)'` - Basic pattern
- `r"양수인\s*기관명\s*[:：]\s*([^\n]+)"` - With "양수인" prefix
- `r"기관명\s*[:：]\s*([^\n]+)"` - Simplified
- `r"양수인\s*기관명\s*[:：]\s*([^\n\r]+?)(?:\s*대표자|\s*$)"` - Enhanced with boundary detection
- `r"기관명\s*[:：]\s*([^\n\r]+?)(?:\s*대표자|\s*$)"` - Enhanced simplified

#### 양수인 대표자명 (Assignee Representative Name)
- `r'대표자\s*명\s*[:：]?\s*([^\n,]+)'` - Basic pattern
- `r"양수인.*?대표자명\s*[:：]\s*([^\s\n]+)"` - With "양수인" prefix
- `r"대표자명\s*[:：]\s*([^\s\n]+)"` - Simplified
- `r"대표자\s*[:：]\s*([^\s\n]+)"` - Most simplified
- `r"양수인.*?대표자명\s*[:：]\s*([^\s\n\r]+)"` - Enhanced
- `r"대표자명\s*[:：]\s*([^\s\n\r]+)"` - Enhanced simplified
- `r"대표자\s*[:：]\s*([^\s\n\r]+)"` - Enhanced most simplified

#### 양수인 대표자 주소 (Assignee Representative Address)
- `r'대표자\s*주소\s*[:：]?\s*([^\n]+)'` - Basic pattern
- `r"양수인.*?대표자.*?주소\s*[:：]\s*([^\n]+)"` - With "양수인" prefix
- `r"대표자.*?주소\s*[:：]\s*([^\n]+)"` - Simplified
- `r"양수인.*?대표자.*?주소\s*[:：]\s*([^\n\r]+?)(?:\s*대표자.*?연락처|\s*$)"` - Enhanced with boundary detection
- `r"대표자.*?주소\s*[:：]\s*([^\n\r]+?)(?:\s*대표자.*?연락처|\s*$)"` - Enhanced simplified

#### 양수인 대표자 연락처 (Assignee Representative Contact)
- `r'대표자\s*연락처\s*[:：]?\s*([\d\-]+)'` - Basic pattern (digits and hyphens only)
- `r"양수인.*?대표자.*?연락처\s*[:：]\s*([^\s\n]+)"` - With "양수인" prefix
- `r"대표자.*?연락처\s*[:：]\s*([^\s\n]+)"` - Simplified
- `r"양수인.*?대표자.*?연락처\s*[:：]\s*([0-9\-]+)"` - Enhanced with digits and hyphens only
- `r"대표자.*?연락처\s*[:：]\s*([0-9\-]+)"` - Enhanced simplified

#### 동의여부 (Consent Status)
- `r'동의\s*여부\s*[:：]?\s*([^\n,]+)'` - Basic pattern
- `r"동의함|동의|합의|승인"` - Direct match (no capture group)
- Used as fallback: checks if text contains "동의", "합의", or "승인"

#### 동의날짜 (Consent Date)
- `r'날짜\s*[:：]?\s*([^\n]+)'` - Basic pattern
- `r"(\d{4}[\.\-\/년]\s*\d{1,2}[\.\-\/월]\s*\d{1,2}[일]?)"` - Date with Korean format (년, 월, 일)
- `r"(\d{4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2})"` - Date with dots (YYYY. MM. DD)

### Text Cleaning Patterns (Used in NER preprocessing)
- `r'\s+'` - Multiple whitespace (used with `re.sub` to normalize spaces)

---

## LLM Model Regex Patterns

### Checkbox Extraction Patterns (`api/module/llm_extraction/extractors/checkbox_extractor.py`)

#### Checkbox State Detection
- `rf"{re.escape(checked_symbol)}\s*{re.escape(item_name)}"` - Pattern for checked checkbox (dynamically generated)
- `rf"{re.escape(unchecked_symbol)}\s*{re.escape(item_name)}"` - Pattern for unchecked checkbox (dynamically generated)

**Checkbox Symbols:**
- Checked: `📧`, `☑`, `✓`, `■`, `●`, `◼`, `◉`
- Unchecked: `☐`, `□`, `○`, `◯`, `◻`, `◦`

#### Work Display Extraction
- `r"저작물명\s*:\s*(.*?)(?=○|$)"` - Extract work names until next bullet or end (with DOTALL flag)
- `r"([^<]+)<([^>]+)>"` - Extract institution and work title from format: `[institution]<[title]>`
- `r"\d{4}년\s*<[^>]+>"` - Extract work names in format: `YYYY년 <title>`
- `r"상세정보\s*:\s*(.*?)(?=□|$)"` - Extract detailed info until checkbox or end (with DOTALL flag)

#### Copyright License Extraction
- `r"저작물의 개방을 통해 이용자가 자유롭게 이용할 수 있도록"` - License purpose pattern
- `r"국립극장"` - Licensing institution pattern

#### Public Nuri License Extraction
- `r"공공저작물의 자유이용 활성화"` - License purpose pattern
- `r"제[1-4]유형"` - Extract Nuri type (제1유형, 제2유형, 제3유형, 제4유형)
- `r"동일성유지권.*동의"` - Integrity right waiver pattern
- `r"변경.*가능"` - Modification allowed pattern
- `r"연구.*결과.*명예.*심각한.*훼손.*특별한.*사정.*없는.*한"` - Conditions pattern

### Payment Amount Extraction (`api/module/llm_extraction/extractors/document_extractors.py`)
- `r'\d+'` - Extract all digits from payment amount string

### Markdown Cleaning Patterns (`api/module/llm_extraction/models/cloud_extractor.py`, `api/module/ocr/alibaba_ocr.py`)

#### Code Block Removal
- `r'```[a-zA-Z]*\n?'` - Remove markdown code blocks with optional language tag
- `r'```\n?'` - Remove markdown code block closing tags

#### Markdown Formatting Removal
- `r'\*\*(.*?)\*\*'` - Remove bold formatting (replaced with `\1`)
- `r'\*(.*?)\*'` - Remove italic formatting (replaced with `\1`)
- `r'`([^`]*)`'` - Remove inline code formatting (replaced with `\1`)

#### Whitespace Normalization
- `r'\n\s*\n\s*\n'` - Replace multiple newlines with double newline (`\n\n`)

### JSON Extraction Patterns (`api/module/consolidator/consolidation_agent.py`)

#### JSON Block Extraction
- `r'```(?:json)?\s*(\{.*?\})\s*```'` - Extract JSON from markdown code blocks (with DOTALL flag)

#### JSON Object Extraction
- `r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'` - Extract complete JSON object using brace matching (with DOTALL flag)

---

## Validation & Utility Regex Patterns

### Date Validation (`api/module/consolidator/validation_engine.py`)
- `r'^\d{4}-\d{2}-\d{2}$'` - Validate date format YYYY-MM-DD (compiled pattern)

### Phone Number Validation (`api/module/consolidator/validation_engine.py`)
- `r'^[0-9\-]+$'` - Validate phone format (digits and hyphens only, compiled pattern)
- `r'[^0-9]'` - Extract digits only from phone number (used with `re.sub`)

### Email Validation (`api/module/consolidator/validation_engine.py`)
- `r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'` - Validate email format (compiled pattern)

### Date Comparison (`api/module/consolidator/field_mapper.py`)
- `r'\d{4}-\d{2}-\d{2}'` - Match date format for comparison

### Filename Sanitization (`api/web/app.py`)
- `r'(?u)[^-\w.]'` - Remove all characters except word characters, hyphens, and dots from filenames

---

## Summary Statistics

### NER Model Patterns
- **Contract Document**: 10 entity types with 30+ regex patterns
- **Consent Document**: 9 entity types with 25+ regex patterns
- **Total NER Patterns**: 55+ unique regex patterns

### LLM Model Patterns
- **Checkbox Extraction**: 2 dynamic pattern templates + 10+ specific patterns
- **Work Display**: 4 patterns
- **License Extraction**: 6 patterns
- **Markdown Cleaning**: 5 patterns
- **JSON Extraction**: 2 patterns
- **Total LLM Patterns**: 20+ unique regex patterns

### Validation Patterns
- **Date Validation**: 1 pattern
- **Phone Validation**: 2 patterns
- **Email Validation**: 1 pattern
- **Utility Patterns**: 2 patterns
- **Total Validation Patterns**: 6 patterns

### Grand Total
**Total Unique Regex Patterns in Project: 80+ patterns**

---

## Notes

1. **Pattern Evolution**: The project has both basic patterns (`ner/entity_extraction.py`) and enhanced patterns (`ner/enhance_training_data.py`) showing the evolution of pattern matching.

2. **Character Variations**: Many patterns account for:
   - Korean colon variations: `[:：]` (regular colon and full-width colon)
   - Bullet point variations: `○`, `●`, `□`, `☐`, etc.
   - Whitespace variations: `\s*` for flexible spacing

3. **Boundary Detection**: Enhanced patterns use lookahead assertions `(?=...)` to properly detect field boundaries and prevent over-matching.

4. **Dynamic Patterns**: LLM checkbox extraction uses `re.escape()` to dynamically generate patterns from checkbox symbols and item names.

5. **Flag Usage**: Most patterns use:
   - `re.IGNORECASE` - Case-insensitive matching
   - `re.DOTALL` - Allow `.` to match newlines
   - `re.MULTILINE` - Not commonly used

6. **Fallback Logic**: Some patterns (like 동의여부) have fallback logic that checks for keywords in text if regex matching fails.

