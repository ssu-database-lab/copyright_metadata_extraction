# Metadata Consolidation - Presentation Diagrams

## 1. High-Level Overview (Simple)

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCUMENT PROCESSING PIPELINE                │
└─────────────────────────────────────────────────────────────┘

    📄 Document (PDF/Image)
          │
          ▼
    ┌─────────────┐
    │  OCR Layer   │ → Extract Text
    └─────────────┘
          │
          ├──────────────────┐
          │                  │
          ▼                  ▼
    ┌──────────┐      ┌──────────┐
    │   LLM    │      │   NER    │
    │ Extractor│      │ Extractor│
    └──────────┘      └──────────┘
          │                  │
          ├──────────┬───────┤
          │          │       │
          ▼          ▼       ▼
    ┌─────────────────────────────┐
    │  Consolidation Agent        │
    │  (LLM Expert Judge)         │
    └─────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────┐
    │  Final Consolidated Metadata │
    │  + Validation Report         │
    │  + Evidence & Reasoning      │
    └─────────────────────────────┘
```

## 2. Consolidation Process (Step-by-Step)

```
STEP 1: FIELD MAPPING
─────────────────────────────────────────────────────────
NER Entities              LLM Fields
─────────────────────────────────────────────────────────
NAME: "집건에"      →      rights_holder: "집건에"
DATE: "2024-01-15"  →      signature_date: "2024-01-15"
PHONE: "010-..."    →      parties[].phone: "010-..."
COMPANY: "국립..."   →      user: "국립생태원"


STEP 2: COMPARISON
─────────────────────────────────────────────────────────
Field          LLM Value         NER Value        Status
─────────────────────────────────────────────────────────
signature_date "2024-01-15"      "2024-01-15"    ✓ AGREE
rights_holder  "집건에"            "집건에"          ✓ AGREE
user           "국립생태원..."     "국립생태원..."   ✓ AGREE
payment_amount 10000             null             ⚠ LLM ONLY


STEP 3: DECISION & REASONING
─────────────────────────────────────────────────────────
Field: signature_date
├─ LLM Value: "2024-01-15" (confidence: 0.95)
├─ NER Value: "2024-01-15" (confidence: 0.88)
├─ Decision: AGREED
├─ Final Value: "2024-01-15"
├─ Confidence: 1.0
└─ Reasoning: "Both sources extracted identical date"


STEP 4: OUTPUT
─────────────────────────────────────────────────────────
{
  "consolidated_metadata": { ... },
  "validation_report": {
    "confidence_score": 0.92,
    "total_fields": 15,
    "agreed_fields": 12,
    "conflicted_fields": 2
  }
}
```

## 3. Before vs After Comparison

```
┌─────────────────────────────────────────────────────────────┐
│                    BEFORE (Current System)                    │
└─────────────────────────────────────────────────────────────┘

    Document
       │
       ├─→ LLM Extraction → {metadata: {...}}
       │
       └─→ NER Extraction → {entities: [...]}
       │
       Result: Two separate outputs (no consolidation)

    Problems:
    ❌ No validation between sources
    ❌ No conflict resolution
    ❌ No evidence/reasoning
    ❌ Potential errors from hallucinations
    ❌ Users must manually compare


┌─────────────────────────────────────────────────────────────┐
│                     AFTER (With Consolidation)                │
└─────────────────────────────────────────────────────────────┘

    Document
       │
       ├─→ LLM Extraction → LLM Result
       │
       └─→ NER Extraction → NER Result
       │
       └─→ Consolidation Agent
              │
              ├─→ Field Mapping
              ├─→ Validation
              ├─→ Decision Making
              └─→ Reasoning Generation
       │
       Result: Single consolidated output

    Benefits:
    ✅ Intelligent merging & validation
    ✅ Automatic conflict resolution
    ✅ Evidence-based decisions
    ✅ Reduced errors & hallucinations
    ✅ Ready-to-use metadata
```

## 4. Decision Matrix

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DECISION MATRIX                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Scenario           │ LLM │ NER │ Decision        │ Confidence      │
├─────────────────────┼─────┼─────┼─────────────────┼────────────────┤
│ Both Agree          │  ✓  │  ✓  │ Use Value       │ High (0.9-1.0) │
│ Conflict            │  ✓  │  ✓  │ Best Source     │ Medium (0.7)   │
│ LLM Only            │  ✓  │  ✗  │ Use LLM         │ Low (0.5)      │
│ NER Only            │  ✗  │  ✓  │ Use NER         │ Medium (0.6)   │
│ Both Missing        │  ✗  │  ✗  │ Set to null     │ N/A            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 5. Component Architecture (Visual)

```
                    ┌─────────────────────────┐
                    │  Consolidation Agent    │
                    │   (LLM Orchestrator)    │
                    └─────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐    ┌──────────────┐
│ Field Mapper │      │  Validator   │    │  Reasoner    │
│              │      │              │    │              │
│ • Entity→Field│      │ • Format     │    │ • Evidence    │
│ • Context     │      │ • Logic      │    │ • Reasoning   │
│ • Matching    │      │ • Consistency │    │ • Confidence  │
└──────────────┘      └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   Consolidated Output   │
                    │                         │
                    │ • Final Metadata        │
                    │ • Validation Report     │
                    │ • Evidence & Reasoning  │
                    └─────────────────────────┘
```

## 6. Data Flow (Detailed)

```
INPUT PHASE
═══════════════════════════════════════════════════════════════

OCR Text:
"저작재산권 비독점적 이용허락 계약서
저작자 및 저작권 이용허락자 집건에..."


EXTRACTION PHASE
═══════════════════════════════════════════════════════════════

LLM Result:                        NER Result:
{                                  [
  "rights_holder": "집건에",         ("집건에", "NAME"),
  "user": "국립생태원...",          ("국립생태원", "COMPANY"),
  "signature_date": "2024-01-15",   ("2024-01-15", "DATE"),
  "confidence": 0.95                ("010-1234-5678", "PHONE")
}                                   ]


CONSOLIDATION PHASE
═══════════════════════════════════════════════════════════════

Field Mapping:
  NAME "집건에" → rights_holder
  DATE "2024-01-15" → signature_date
  COMPANY "국립생태원" → user
  PHONE "010-..." → parties[].phone

Comparison:
  ✓ rights_holder: AGREED
  ✓ signature_date: AGREED
  ✓ user: AGREED
  ✓ parties[].phone: ADDED FROM NER

Decision Making:
  - Use agreed values (high confidence)
  - Merge NER-only fields
  - Flag LLM-only fields (low confidence)

Reasoning Generation:
  - Extract OCR excerpts
  - Calculate confidence scores
  - Generate Korean explanations


OUTPUT PHASE
═══════════════════════════════════════════════════════════════

Consolidated Metadata:
{
  "rights_holder": "집건에",
  "user": "국립생태원 멸종위기종복원센터",
  "signature_date": "2024-01-15",
  "parties": [{
    "phone": "010-1234-5678"
  }]
}

Validation Report:
{
  "confidence_score": 0.92,
  "total_fields": 15,
  "agreed_fields": 12,
  "decisions": [
    {
      "field": "signature_date",
      "decision": "AGREED",
      "reasoning": "Both sources extracted identical date",
      "confidence": 1.0
    }
  ]
}
```

## 7. Use Cases & Benefits

```
┌─────────────────────────────────────────────────────────────┐
│                      USE CASES                               │
└─────────────────────────────────────────────────────────────┘

1. CONTRACT PROCESSING
   ───────────────────────────────────────────────
   Before: LLM might extract wrong date
   After:  Validated against NER → Correct date

2. CONSENT FORMS
   ───────────────────────────────────────────────
   Before: Missing phone numbers
   After:  NER fills in missing fields

3. COPYRIGHT DOCUMENTS
   ───────────────────────────────────────────────
   Before: Inconsistent entity names
   After:  Cross-validated → Consistent

4. AUDIT & COMPLIANCE
   ───────────────────────────────────────────────
   Before: No evidence trail
   After:  Full reasoning & evidence provided
```

## 8. Implementation Timeline

```
Week 1: Foundation
├─ Create module structure
├─ Field mapper (basic)
└─ Validation engine (basic)

Week 2: Core Logic
├─ Consolidation agent
├─ LLM prompt engineering
└─ Decision logic

Week 3: Reasoning
├─ Evidence generation
├─ Report formatting
└─ Confidence calculation

Week 4: Integration
├─ API endpoint integration
├─ Error handling
└─ Testing

Week 5: Production
├─ Performance optimization
├─ Documentation
└─ Deployment
```

## 9. Confidence Score Calculation

```
Formula:
─────────────────────────────────────────────────────────
Final Confidence = 
  (LLM_Confidence × LLM_Weight) 
  + (NER_Confidence × NER_Weight) 
  + Agreement_Bonus

Example:
─────────────────────────────────────────────────────────
Field: signature_date

LLM Confidence:    0.95
NER Confidence:    0.88
Agreement:         Yes (+0.1 bonus)

Calculation:
  0.95 × 0.6 + 0.88 × 0.4 + 0.1 = 0.57 + 0.352 + 0.1 = 1.022

Final Confidence:  min(1.022, 1.0) = 1.0 (High Confidence)
```

## 10. Error Handling Scenarios

```
Scenario 1: NER Extraction Fails
─────────────────────────────────────────────────────────
Action:    Use LLM result only
Warning:   "NER extraction failed, using LLM-only result"
Output:    metadata + warning flag

Scenario 2: LLM Extraction Fails
─────────────────────────────────────────────────────────
Action:    Use NER entities only
Warning:   "LLM extraction failed, using NER-only result"
Output:    NER-based metadata + warning flag

Scenario 3: Consolidation Fails
─────────────────────────────────────────────────────────
Action:    Return both results separately
Warning:   "Consolidation failed, returning raw results"
Output:    llm_metadata + ner_entities (separate)

Scenario 4: Both Fail
─────────────────────────────────────────────────────────
Action:    Return error with diagnostic info
Output:    Error response + detailed logs
```

## Presentation Tips

1. **Slide 1**: Use Diagram 1 (High-Level Overview)
2. **Slide 2**: Use Diagram 2 (Before vs After)
3. **Slide 3**: Use Diagram 3 (Step-by-Step Process)
4. **Slide 4**: Use Diagram 5 (Component Architecture)
5. **Slide 5**: Use Diagram 9 (Confidence Calculation)
6. **Slide 6**: Use Diagram 7 (Use Cases)

For technical audiences: Include all diagrams
For executive audiences: Use Diagrams 1, 2, 7, 9 only

