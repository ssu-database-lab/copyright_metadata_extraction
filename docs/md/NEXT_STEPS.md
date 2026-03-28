# Next Steps: Digital Content Schema Integration

## ✅ Completed

1. ✅ Created new `get_digital_content_schema()` with 42 fields
2. ✅ Updated schema detection logic (`get_schema_by_document_type()`)
3. ✅ Updated frontend to include "디지털 콘텐츠" document type option
4. ✅ Created comparison and implementation documentation

## 🔄 Next Steps (Priority Order)

### 1. **Update FieldMapper for Digital Content Schema** (HIGH PRIORITY)
   - **What**: Add mappings for new digital content fields to `field_mapper.py`
   - **Why**: Consolidation needs to map NER entities to the new schema fields
   - **Files**: `api/module/consolidator/field_mapper.py`
   - **Fields to add**:
     - `agency_name`, `site_name`, `board_name` → COMPANY entities
     - `created_date`, `registration_date`, `production_date` → DATE entities
     - `url` → URL entities
     - `phone` → PHONE entities (already exists but needs priority update)
     - `copyright_holder`, `co_author` → NAME/COMPANY entities
     - `work_title` → TITLE entities (already exists)
     - `description` → DESCRIPTION entities (already exists)

### 2. **Test the New Schema** (HIGH PRIORITY)
   - **What**: Test extraction with a real digital content document
   - **Why**: Verify the schema works correctly with actual data
   - **Steps**:
     1. Upload a digital content document via frontend
     2. Select "디지털 콘텐츠" as document type
     3. Verify all 42 fields are extracted correctly
     4. Check consolidation results

### 3. **Update Validation Engine** (MEDIUM PRIORITY)
   - **What**: Add validation rules for new digital content fields
   - **Why**: Ensure data quality (URL format, date formats, etc.)
   - **Fields needing validation**:
     - `url` - Must be valid URL format
     - `created_date`, `registration_date`, `production_date` - Must be YYYY-MM-DD
     - `kogl_type` - Must be one of: 제1유형, 제2유형, 제3유형, 제4유형
     - `video_count`, `photo_count`, `document_count`, `view_count` - Must be non-negative integers
     - `phone` - Must match phone number format

### 4. **Update Consolidation Schemas** (MEDIUM PRIORITY)
   - **What**: Ensure consolidation schema supports all new fields
   - **Why**: Consolidation needs to handle all field types
   - **File**: `api/module/consolidator/schemas/consolidation_schemas.py`
   - **Check**: Verify `get_consolidation_schema()` handles all 42 fields

### 5. **Create Test Script** (LOW PRIORITY)
   - **What**: Create a test script specifically for digital content schema
   - **Why**: Easy testing and validation
   - **File**: `api/module/llm_extraction/test_digital_content_schema.py`
   - **Include**: Sample data, field validation, extraction test

### 6. **Update Documentation** (LOW PRIORITY)
   - **What**: Update API documentation and user guides
   - **Why**: Help users understand the new schema
   - **Files**: 
     - API documentation
     - User guide
     - Schema reference

---

## Immediate Action Items

### Priority 1: Update FieldMapper

**File**: `api/module/consolidator/field_mapper.py`

**Changes needed**:

1. **Add new field mappings** to `_initialize_mappings()`:
   ```python
   'DATE': [
       # Existing fields...
       'created_date',      # NEW
       'registration_date', # NEW
       'production_date',   # NEW
       'valid_period'       # NEW
   ],
   
   'COMPANY': [
       # Existing fields...
       'agency_name',       # NEW
       'site_name',         # NEW
       'board_name'         # NEW
   ],
   
   'URL': [
       'url'                # NEW (already exists but ensure it's there)
   ],
   
   'PHONE': [
       # Existing fields...
       'phone'              # NEW (for digital content schema)
   ],
   
   'NAME': [
       # Existing fields...
       'copyright_holder',  # NEW
       'co_author'         # NEW
   ],
   ```

2. **Add priority scores** to `_initialize_priorities()`:
   ```python
   'DATE': {
       # Existing...
       'created_date': 9,
       'registration_date': 8,
       'production_date': 7
   },
   'COMPANY': {
       # Existing...
       'agency_name': 10,
       'site_name': 9,
       'board_name': 8
   },
   'PHONE': {
       # Existing...
       'phone': 8  # For digital content schema
   }
   ```

### Priority 2: Test End-to-End

1. **Prepare test document**: A sample digital content document (PDF/image)
2. **Run extraction**:
   ```bash
   # Via frontend: Select "디지털 콘텐츠" and upload
   # Or via API:
   curl -X POST "http://localhost:5000/api/llm-extract" \
     -F "file=@test_document.pdf" \
     -F "document_type=디지털 콘텐츠" \
     -F "model_name=alibaba-qwen-max"
   ```
3. **Verify results**: Check that all 42 fields are extracted
4. **Test consolidation**: Verify consolidation works with new fields

---

## Testing Checklist

- [ ] FieldMapper maps NER entities to new digital content fields
- [ ] LLM extracts all 42 fields correctly
- [ ] Consolidation merges LLM and NER results for digital content
- [ ] Frontend displays digital content metadata correctly
- [ ] Validation engine validates new field formats
- [ ] URL field accepts valid URLs
- [ ] Date fields accept YYYY-MM-DD format
- [ ] Count fields accept non-negative integers
- [ ] KOGL type field accepts valid values

---

## Estimated Time

- **FieldMapper update**: 30 minutes
- **Testing**: 1-2 hours
- **Validation updates**: 1 hour
- **Documentation**: 30 minutes

**Total**: ~3-4 hours

---

## Questions to Consider

1. **Field conflicts**: What if a document has both contract fields AND digital content fields? Should we support hybrid schemas?
2. **Required fields**: Currently only `work_title` is required. Should we add more required fields?
3. **Field types**: Some fields accept multiple types (string/array/null). Is this sufficient?
4. **Validation strictness**: How strict should validation be? (e.g., URL format, date format)

---

## Notes

- The new schema is designed to be flexible (most fields optional)
- Consolidation will work automatically once FieldMapper is updated
- Frontend already supports the new document type
- Schema detection is already implemented

