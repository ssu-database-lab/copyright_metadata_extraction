# Metadata Consolidation System - Architecture Diagrams

## 1. System Architecture Overview

```mermaid
graph TB
    subgraph "Input Layer"
        OCR[OCR Text Extraction]
        PDF[PDF/Image Documents]
    end
    
    subgraph "Extraction Layer"
        LLM[LLM Metadata Extractor]
        NER[NER Entity Extractor]
    end
    
    subgraph "Consolidation Layer"
        MAPPER[Field Mapper]
        VALIDATOR[Validation Engine]
        AGENT["Consolidation Agent<br/>🤖 Qwen3-Next-80B<br/>(Alibaba Cloud)"]
        REASONER[Reasoning Generator]
    end
    
    subgraph "Output Layer"
        CONSOLIDATED[Consolidated Metadata]
        REPORT[Validation Report]
        EVIDENCE[Evidence & Reasoning]
    end
    
    PDF --> OCR
    OCR --> LLM
    OCR --> NER
    
    LLM --> MAPPER
    NER --> MAPPER
    MAPPER --> VALIDATOR
    VALIDATOR --> AGENT
    AGENT --> REASONER
    
    REASONER --> CONSOLIDATED
    REASONER --> REPORT
    REASONER --> EVIDENCE
    
    style AGENT fill:#4299e1,stroke:#2b6cb0,stroke-width:3px,color:#fff
    style CONSOLIDATED fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
    style REPORT fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff
```

## 2. Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
    participant API as /api/llm-extract
    participant OCR as OCR Processor
    participant LLM as LLM Extractor
    participant NER as NER Extractor
    participant Consolidator as Consolidation Agent
    participant Output
    
    User->>API: Upload Document
    API->>OCR: Extract Text
    OCR-->>API: OCR Text
    
    par Parallel Extraction
        API->>LLM: Extract Metadata
        LLM-->>API: LLM Result
    and
        API->>NER: Extract Entities
        NER-->>API: NER Result
    end
    
    API->>Consolidator: Consolidate Results<br/>(LLM + NER + OCR)
    Note over Consolidator: Consolidation Agent Process:<br/>1. Field Mapping (No LLM)<br/>2. Validation (No LLM)
    Consolidator->>Qwen3: Call Qwen3-Next-80B API<br/>(Alibaba Cloud)
    Note over Qwen3: LLM compares & merges:<br/>- Compare LLM vs NER<br/>- Make decisions<br/>- Generate reasoning
    Qwen3-->>Consolidator: Consolidated Metadata<br/>+ Decisions + Reasoning
    Consolidator-->>API: Consolidated Result<br/>(Metadata + Report + Evidence)
    API->>Output: Return Metadata + Report
    Output-->>User: Final Response
```

## 3. Consolidation Process Flow

```mermaid
flowchart TD
    START([Start: LLM + NER Results])
    
    MAP[Field Mapping<br/>Map NER entities to LLM fields]
    COMPARE{Compare Values<br/>Field by Field}
    
    AGREE{Values<br/>Agree?}
    CONFLICT{Values<br/>Conflict?}
    MISSING{Field<br/>Missing?}
    
    VALIDATE[Validate Formats<br/>& Logic]
    
    DECIDE_AGREED[Use Agreed Value<br/>+ High Confidence]
    DECIDE_CONFLICT[Choose Best Source<br/>+ Conflict Resolution]
    DECIDE_MISSING[Use Available Source<br/>+ Warning Flag]
    
    REASON[Generate Reasoning<br/>& Evidence]
    
    MERGE[Merge into<br/>Consolidated Metadata]
    
    REPORT[Generate Validation<br/>Report]
    
    END([Output: Consolidated Result])
    
    START --> MAP
    MAP --> COMPARE
    COMPARE --> AGREE
    COMPARE --> CONFLICT
    COMPARE --> MISSING
    
    AGREE -->|Yes| VALIDATE
    CONFLICT -->|Yes| VALIDATE
    MISSING -->|Yes| DECIDE_MISSING
    
    VALIDATE --> DECIDE_AGREED
    VALIDATE --> DECIDE_CONFLICT
    
    DECIDE_AGREED --> REASON
    DECIDE_CONFLICT --> REASON
    DECIDE_MISSING --> REASON
    
    REASON --> MERGE
    MERGE --> REPORT
    REPORT --> END
    
    style START fill:#e0f2fe
    style END fill:#d1fae5
    style REASON fill:#fef3c7
    style VALIDATE fill:#fce7f3
```

## 4. Field Mapping Strategy

```mermaid
graph LR
    subgraph "NER Entities"
        NER_NAME[NAME: 집건에]
        NER_DATE[DATE: 2024-01-15]
        NER_PHONE[PHONE: 010-1234-5678]
        NER_COMPANY[COMPANY: 국립생태원]
    end
    
    subgraph "Mapping Logic"
        MAPPER[Field Mapper<br/>Context-aware Matching]
    end
    
    subgraph "LLM Fields"
        LLM_RIGHTS[rights_holder]
        LLM_USER[user]
        LLM_DATE[signature_date]
        LLM_PHONE[parties[].phone]
    end
    
    NER_NAME --> MAPPER
    NER_DATE --> MAPPER
    NER_PHONE --> MAPPER
    NER_COMPANY --> MAPPER
    
    MAPPER -->|Match by position| LLM_RIGHTS
    MAPPER -->|Match by position| LLM_USER
    MAPPER -->|Match by format| LLM_DATE
    MAPPER -->|Match by pattern| LLM_PHONE
    
    style MAPPER fill:#4299e1,color:#fff
```

## 5. Decision Logic Flow

```mermaid
flowchart TD
    INPUT[Field Comparison:<br/>LLM Value vs NER Value]
    
    CHECK1{Both<br/>Present?}
    CHECK2{Values<br/>Match?}
    CHECK3{Format<br/>Valid?}
    CHECK4{Confidence<br/>High?}
    
    DECISION1[Use Agreed Value<br/>Confidence: High]
    DECISION2[Use NER Value<br/>Reason: Present in OCR]
    DECISION3[Use LLM Value<br/>Reason: Higher Confidence]
    DECISION4[Use LLM Value<br/>Reason: Only Source Available]
    DECISION5[Use NER Value<br/>Reason: Format Validated]
    DECISION6[Flag for Review<br/>Reason: Low Confidence]
    
    REASONING[Generate Reasoning]
    
    INPUT --> CHECK1
    
    CHECK1 -->|Yes| CHECK2
    CHECK1 -->|No LLM| DECISION5
    CHECK1 -->|No NER| DECISION4
    
    CHECK2 -->|Yes| CHECK3
    CHECK2 -->|No| CHECK4
    
    CHECK3 -->|Yes| DECISION1
    CHECK3 -->|No| CHECK4
    
    CHECK4 -->|Yes| DECISION3
    CHECK4 -->|No| DECISION6
    
    DECISION2 --> CHECK4
    
    DECISION1 --> REASONING
    DECISION2 --> REASONING
    DECISION3 --> REASONING
    DECISION4 --> REASONING
    DECISION5 --> REASONING
    DECISION6 --> REASONING
    
    REASONING --> OUTPUT[Final Value + Evidence]
    
    style DECISION1 fill:#10b981,color:#fff
    style DECISION2 fill:#3b82f6,color:#fff
    style DECISION3 fill:#3b82f6,color:#fff
    style DECISION6 fill:#ef4444,color:#fff
```

## 6. Component Interaction

```mermaid
graph TB
    subgraph "Consolidation Agent Core"
        AGENT["Consolidation Agent<br/>🤖 Qwen3-Next-80B<br/>(Alibaba Cloud API)"]
    end
    
    subgraph "Supporting Modules"
        MAPPER[Field Mapper<br/>Entity → Field Mapping]
        VALIDATOR[Validation Engine<br/>Format & Logic Checks]
        REASONER[Reasoning Generator<br/>Evidence Creation]
    end
    
    subgraph "LLM Infrastructure"
        LLM_BASE[Base LLM Extractor<br/>Reuses existing infrastructure]
        PROMPT[Prompt Templates<br/>Consolidation-specific]
    end
    
    subgraph "Data Sources"
        LLM_RESULT[LLM Result<br/>Structured Metadata]
        NER_RESULT[NER Result<br/>Entity List]
        OCR_TEXT[OCR Text<br/>Source Document]
    end
    
    LLM_RESULT --> AGENT
    NER_RESULT --> AGENT
    OCR_TEXT --> AGENT
    
    AGENT --> MAPPER
    AGENT --> VALIDATOR
    AGENT --> REASONER
    AGENT --> LLM_BASE
    
    LLM_BASE --> PROMPT
    MAPPER --> AGENT
    VALIDATOR --> AGENT
    REASONER --> AGENT
    
    style AGENT fill:#4299e1,stroke:#2b6cb0,stroke-width:3px,color:#fff
    style LLM_BASE fill:#10b981,stroke:#059669,stroke-width:2px
```

## 7. Integration with Existing System

```mermaid
graph TB
    subgraph "Existing System"
        API[FastAPI Endpoint<br/>/api/llm-extract]
        EXISTING_LLM[LLM Extraction Module]
        EXISTING_NER[NER Extraction Module]
    end
    
    subgraph "New Consolidation Module"
        NEW_CONSOLIDATOR["Consolidation Agent<br/>🤖 Qwen3-Next-80B"]
        NEW_MAPPER[Field Mapper]
        NEW_VALIDATOR[Validation Engine]
    end
    
    subgraph "Output Enhancement"
        OLD_OUTPUT[Original Response<br/>LLM + NER separate]
        NEW_OUTPUT[Enhanced Response<br/>Consolidated Metadata + Report]
    end
    
    API --> EXISTING_LLM
    API --> EXISTING_NER
    
    EXISTING_LLM --> NEW_CONSOLIDATOR
    EXISTING_NER --> NEW_CONSOLIDATOR
    
    NEW_CONSOLIDATOR --> NEW_MAPPER
    NEW_CONSOLIDATOR --> NEW_VALIDATOR
    NEW_MAPPER --> NEW_CONSOLIDATOR
    NEW_VALIDATOR --> NEW_CONSOLIDATOR
    
    NEW_CONSOLIDATOR --> NEW_OUTPUT
    
    API --> OLD_OUTPUT
    API --> NEW_OUTPUT
    
    style NEW_CONSOLIDATOR fill:#4299e1,stroke:#2b6cb0,stroke-width:3px,color:#fff
    style NEW_OUTPUT fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff
```

## 8. Evidence Generation Process

```mermaid
flowchart LR
    subgraph "Input Sources"
        LLM_VAL[LLM Value]
        NER_VAL[NER Value]
        OCR_SRC[OCR Text]
        POS[Position Info]
    end
    
    subgraph "Evidence Builder"
        EXTRACT[Extract OCR Excerpt<br/>Around Position]
        COMPARE[Compare Values]
        CONFIDENCE[Calculate Confidence]
    end
    
    subgraph "Reasoning"
        REASON[Generate Reasoning<br/>in Korean]
        EVIDENCE[Create Evidence Object]
    end
    
    LLM_VAL --> EXTRACT
    NER_VAL --> COMPARE
    OCR_SRC --> EXTRACT
    POS --> EXTRACT
    
    EXTRACT --> REASON
    COMPARE --> REASON
    CONFIDENCE --> EVIDENCE
    
    REASON --> EVIDENCE
    
    EVIDENCE --> OUTPUT[Evidence Report]
    
    style EVIDENCE fill:#f59e0b,color:#fff
    style OUTPUT fill:#10b981,color:#fff
```

## 9. Error Handling & Fallback

```mermaid
flowchart TD
    START([Start Consolidation])
    
    TRY[Attempt Consolidation]
    
    SUCCESS{Success?}
    ERROR_TYPE{Error Type?}
    
    NER_FAIL[NER Failed]
    LLM_FAIL[LLM Failed]
    CONSOL_FAIL[Consolidation Failed]
    
    FALLBACK1[Use LLM Only<br/>+ Warning Flag]
    FALLBACK2[Use NER Only<br/>+ Warning Flag]
    FALLBACK3[Return Both Separately<br/>+ Warning Flag]
    
    LOG[Log Error for Analysis]
    RETURN[Return Result]
    
    START --> TRY
    TRY --> SUCCESS
    
    SUCCESS -->|Yes| RETURN
    SUCCESS -->|No| ERROR_TYPE
    
    ERROR_TYPE --> NER_FAIL
    ERROR_TYPE --> LLM_FAIL
    ERROR_TYPE --> CONSOL_FAIL
    
    NER_FAIL --> FALLBACK1
    LLM_FAIL --> FALLBACK2
    CONSOL_FAIL --> FALLBACK3
    
    FALLBACK1 --> LOG
    FALLBACK2 --> LOG
    FALLBACK3 --> LOG
    
    LOG --> RETURN
    
    style TRY fill:#4299e1,color:#fff
    style FALLBACK1 fill:#f59e0b,color:#fff
    style FALLBACK2 fill:#f59e0b,color:#fff
    style FALLBACK3 fill:#ef4444,color:#fff
```

## 10. Complete System Pipeline

```mermaid
graph TB
    subgraph "Phase 1: Input"
        DOC[Document Upload]
        OCR[OCR Processing]
    end
    
    subgraph "Phase 2: Extraction"
        LLM[LLM Metadata<br/>Extraction]
        NER[NER Entity<br/>Extraction]
    end
    
    subgraph "Phase 3: Consolidation"
        MAP[Field Mapping]
        VAL[Validation]
        DECIDE[Decision Making]
        REASON[Reasoning]
    end
    
    subgraph "Phase 4: Output"
        METADATA[Consolidated<br/>Metadata]
        REPORT[Validation<br/>Report]
        EVIDENCE[Evidence &<br/>Reasoning]
    end
    
    DOC --> OCR
    OCR --> LLM
    OCR --> NER
    
    LLM --> MAP
    NER --> MAP
    MAP --> VAL
    VAL --> DECIDE
    DECIDE --> REASON
    
    REASON --> METADATA
    REASON --> REPORT
    REASON --> EVIDENCE
    
    style OCR fill:#e0f2fe
    style LLM fill:#dbeafe
    style NER fill:#dbeafe
    style DECIDE fill:#4299e1,color:#fff
    style METADATA fill:#10b981,color:#fff
    style REPORT fill:#fef3c7
```

## How to Use These Diagrams

1. **For Presentations**: Copy the Mermaid code to [Mermaid Live Editor](https://mermaid.live/) or use in PowerPoint with a Mermaid plugin
2. **For Documentation**: Include directly in Markdown files (GitHub, GitLab support Mermaid)
3. **For Printing**: Export as PNG/SVG from Mermaid Live Editor
4. **Alternative**: Convert to draw.io format if needed

## Diagram Legend

- **Blue**: Core processing components
- **Green**: Output/results
- **Yellow**: Validation/warning states
- **Red**: Error/fallback scenarios
- **Purple**: Supporting infrastructure

