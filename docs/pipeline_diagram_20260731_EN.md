# System Pipeline Diagram

**Project** Copyright Metadata Extraction System · Soongsil University DB Lab
**Date** 2026-07-31 · **Status** As-built (current implementation)

---

## 1. Overall Pipeline

```mermaid
flowchart TD
    User(["User Browser / Muhayu HF Space"])
    User -->|"Upload: PDF, image, DOCX, HWP"| API

    subgraph SERVER["⑥ API / Web Server — FastAPI + SSE + Web UI"]
        API["PipelineOrchestrator<br/>stage orchestration · progress push"]
        API --> CONV["FileProcessor<br/>PDF → page images"]
        CONV --> ROUTER{"① Modality Router<br/>by file extension"}
    end

    ROUTER -->|document, text| DOCPATH
    ROUTER -->|image| IMGPATH
    ROUTER -->|"video, audio"| BLOCKED

    subgraph DOCPATH["Document Path"]
        OCR["② OCR<br/>UniversalOCRProcessor"]
        OCR --> PARA{" "}
        PARA --> LLM["③ LLM Extraction<br/>qwen3.5-122b-a10b"]
        PARA --> NER["④ NER — LOCAL CPU<br/>KLUE-RoBERTa-Large 1.3GB"]
    end

    subgraph IMGPATH["Image Work Path"]
        VLM["⑤ VLM Attribute Extraction"]
        VLM --> MAP["schema_mapping<br/>map_vlm_to_unified"]
    end

    BLOCKED["🚫 P3 Guard — Not Implemented<br/>video track planned"]

    LLM --> CONS
    NER --> CONS
    MAP --> CONS

    subgraph CONS["⑥ Consolidation — ConsolidationAgent"]
        FM["FieldMapper<br/>NER → LLM field mapping"]
        VE["ValidationEngine<br/>format · logic validation"]
        RG["ReasoningGenerator<br/>Korean evidence"]
        ARB["LLM Arbiter<br/>qwen3.5-122b-a10b<br/>fallback: qwen3.5-plus"]
        FM --> VE --> RG --> ARB
    end

    CONS --> INHERIT{"Contract metadata<br/>provided?"}
    INHERIT -->|Yes| MERGE["Contract → Work Inheritance<br/>28 rights fields · no LLM call"]
    INHERIT -->|No| OUT
    MERGE --> OUT

    OUT["Unified Metadata JSON — 67 fields<br/>per-field decision · confidence · evidence"]
    OUT -->|SSE stream| User
    OUT --> STORE[("results/{request_id}/<br/>llm_metadata.json<br/>consolidated_metadata.json")]

    style BLOCKED fill:#ffe0e0,stroke:#d33,stroke-width:2px
    style NER fill:#e0f0e0,stroke:#2a2
    style CONS fill:#f0f0ff,stroke:#66c
    style OUT fill:#fff8e0,stroke:#c90
```

---

## 1-B. Target Pipeline — with the video track enabled

Same flow as §1, but starting at file upload and with **video/audio routed through keyframe extraction into a VLM**. Image and video **share the same processing structure** (frames → one multi-image VLM call → schema mapping) but **use different models**.

> **Updated 2026-08-03** — the first edition assumed image and video converged on one model (Gemma). An 8-model × 5-video comparison found **`qwen3.5-omni-plus` (92.3%) clearly better than Gemma (76.9%) on video**, so the models are now separate. The structure is shared and only the model differs, so the code path is reused unchanged. Evidence: `docs/video_model_comparison_report_20260803_EN.md`

```mermaid
flowchart TD
    UPLOAD["📁 File Upload<br/>PDF · image · DOCX · HWP · video · audio"]
    UPLOAD --> FILE_PROCESSOR["File Processor<br/>PDF → page images"]
    FILE_PROCESSOR --> MODALITY_ROUTER{"Modality Router<br/>by file extension"}

    MODALITY_ROUTER -->|"document / text<br/>(contract 계약서 · text work 어문)"| OCR_STAGE
    MODALITY_ROUTER -->|"image work"| IMAGE_VLM
    MODALITY_ROUTER -->|"video / audio work"| KEYFRAME_EXTRACTION

    subgraph DOCUMENT_PATH["📄 Document Path — contract (계약서) OR text work (어문 저작물)"]
        OCR_STAGE["1️⃣ OCR<br/>Qwen3-VL-235B<br/>fallback: mistral → google → naver"]
        OCR_STAGE --> PARALLEL_SPLIT{"run in parallel"}
        PARALLEL_SPLIT --> LLM_EXTRACTION["2️⃣ LLM Metadata Extraction<br/>qwen3.5-122b-a10b"]
        PARALLEL_SPLIT --> NER_EXTRACTION["3️⃣ NER — LOCAL CPU<br/>KLUE-RoBERTa-Large"]
    end

    subgraph MEDIA_PATH["🎬 Media Work Path — shared structure · separate models"]
        KEYFRAME_EXTRACTION["4️⃣ Keyframe Extraction<br/>ffmpeg · uniform anchors + farthest-point<br/>3–16 frames by duration"]
        KEYFRAME_EXTRACTION --> VIDEO_VLM
        VIDEO_VLM["5️⃣-a Video VLM Attribute Extraction<br/>qwen3.5-omni-plus (92.3%)<br/>fallback: Qwen3-VL-235B → Gemma 4 31B"]
        IMAGE_VLM["5️⃣-b Image VLM Attribute Extraction<br/>Gemma 4 31B<br/>fallback: self-host → Qwen3-VL"]
        VIDEO_VLM --> SCHEMA_MAPPING
        IMAGE_VLM --> SCHEMA_MAPPING["6️⃣ Schema Mapping<br/>→ unified 67 fields<br/>+ ffprobe technical metadata"]
    end

    LLM_EXTRACTION --> CONSOLIDATION
    NER_EXTRACTION --> CONSOLIDATION
    SCHEMA_MAPPING --> CONSOLIDATION

    CONSOLIDATION["7️⃣ Consolidation — ConsolidationAgent<br/>FieldMapper → ValidationEngine →<br/>ReasoningGenerator → LLM Arbiter"]

    CONSOLIDATION --> INHERITANCE_CHECK{"Contract metadata<br/>provided?"}
    INHERITANCE_CHECK -->|"Yes"| INHERITANCE_MERGE["8️⃣ Contract → Work Inheritance<br/>28 rights fields · no LLM call"]
    INHERITANCE_CHECK -->|"No"| FINAL_OUTPUT
    INHERITANCE_MERGE --> FINAL_OUTPUT

    FINAL_OUTPUT["✅ Unified Metadata JSON — 67 fields<br/>per-field decision · confidence · evidence"]
    FINAL_OUTPUT --> RESULT_STORE[("results/{request_id}/")]

    style KEYFRAME_EXTRACTION fill:#e8f0ff,stroke:#4a7,stroke-width:2px
    style VIDEO_VLM fill:#e8f0ff,stroke:#4a7,stroke-width:2px
    style IMAGE_VLM fill:#e8f0ff,stroke:#4a7,stroke-width:2px
    style NER_EXTRACTION fill:#e0f0e0,stroke:#2a2
    style CONSOLIDATION fill:#f0f0ff,stroke:#66c
    style FINAL_OUTPUT fill:#fff8e0,stroke:#c90
```

**Stage reference**

| # | Stage | Runs on | Note |
|---|---|---|---|
| 1️⃣ | **OCR** | Alibaba (cloud) | Scanned document → Korean text |
| 2️⃣ | **LLM Metadata Extraction** | Alibaba (cloud) | Text → structured 67-field JSON |
| 3️⃣ | **NER** | **Local CPU** | Names, orgs, contacts — no external transmission |
| 4️⃣ | **Keyframe Extraction** | **Local (ffmpeg)** | Video only · zero API cost |
| 5️⃣-a | **Video VLM Attribute Extraction** | Alibaba (cloud) | qwen3.5-omni-plus · description, keywords |
| 5️⃣-b | **Image VLM Attribute Extraction** | OpenRouter (cloud) | Gemma 4 31B · description, work type, keywords |
| 6️⃣ | **Schema Mapping** | Local | VLM output + ffprobe → unified schema |
| 7️⃣ | **Consolidation** | Alibaba (cloud) | Arbitrates LLM vs NER, adds confidence + evidence |
| 8️⃣ | **Contract Inheritance** | **Local** | Pure merge, no LLM call |

> **Document Path serves two different roles**: a **contract/consent form (계약서·동의서)** — the rights document that *supplies* metadata — and a **text work (어문 저작물)** — a work that *receives* inherited rights. Both use the identical OCR → LLM ∥ NER → Consolidation chain; only their role in the inheritance step differs.

> **Key point**: image works go straight to the VLM; video works pass through keyframe extraction first, then enter a VLM. The two paths **share the call structure, prompt format, schema mapping and consolidation 100%** and differ **only in the model** (video `qwen3.5-omni-plus` / image Gemma 4 31B). Only the frame-extraction stage and the model routing are new.

---

## 2. Model Backends and External Dependencies

```mermaid
flowchart LR
    subgraph LOCAL["🖥️ Local Execution — no external transmission"]
        NER["④ NER<br/>KLUE-RoBERTa-Large<br/>CPU · 1.3GB"]
        SRV["⑥ FastAPI Server<br/>file conversion · orchestration"]
    end

    subgraph OCRCHAIN["② OCR — fallback chain"]
        direction TB
        O1["1. Alibaba<br/>qwen3-vl-235b-a22b-instruct"]
        O2["2. Mistral OCR"]
        O3["3. Google Vision"]
        O4["4. Naver CLOVA"]
        O1 -.->|on failure| O2 -.->|on failure| O3 -.->|on failure| O4
    end

    subgraph VLMCHAIN["⑤-b Image VLM — fallback chain"]
        direction TB
        V1["1. Gemma 4 31B<br/>OpenRouter"]
        V2["2. Gemma 4 31B<br/>self-hosted vLLM"]
        V3["3. Qwen3-VL-235B<br/>DashScope"]
        V1 -.->|on failure| V2 -.->|on failure| V3
    end

    subgraph VIDCHAIN["⑤-a Video VLM — fallback chain (added 2026-08-03)"]
        direction TB
        W1["1. qwen3.5-omni-plus<br/>DashScope · 92.3% · 5.0 s"]
        W2["2. Qwen3-VL-235B<br/>DashScope · 80.8% · 4.0 s"]
        W3["3. Gemma 4 31B<br/>OpenRouter · 76.9%"]
        W1 -.->|on failure| W2 -.->|on failure| W3
    end

    subgraph TEXTLLM["③⑥ Extraction / Consolidation"]
        T1["qwen3.5-122b-a10b<br/>DashScope"]
        T2["fallback: qwen3.5-plus"]
        T1 -.->|on failure| T2
    end

    subgraph REGION["🌍 Data Location"]
        SG["Singapore<br/>Alibaba DashScope"]
        US["United States +<br/>3rd-party providers<br/>OpenRouter"]
        OTHER["France / US<br/>Mistral, Google"]
    end

    OCRCHAIN --> SG
    TEXTLLM --> SG
    VLMCHAIN --> US
    VIDCHAIN --> SG
    VIDCHAIN -.->|"final fallback only"| US
    OCRCHAIN -.-> OTHER

    style LOCAL fill:#e0f0e0,stroke:#2a2,stroke-width:2px
    style REGION fill:#fff0f0,stroke:#d33
```

---

## 3. Paired Analysis Flow (Contract + Work)

The `/pair` endpoint analyses a contract and its works **concurrently**, then merges rights metadata once the contract completes.

```mermaid
sequenceDiagram
    participant U as User / Muhayu
    participant S as API Server
    participant C as Contract Pipeline
    participant W as Work Pipeline
    participant M as Inheritance Merge

    U->>S: Submit contract + N works (single form)
    par Contract analysis
        S->>C: POST /api/llm-extract (document_type=계약서)
        C->>C: OCR → LLM ∥ NER → Consolidation
        C-->>S: consolidated_metadata (rights fields)
    and Work analysis (concurrent, up to 3)
        S->>W: POST /api/llm-extract (no contract_metadata)
        W->>W: VLM → Consolidation
        W-->>S: work metadata (visual fields)
    end
    S->>M: POST /api/apply-inheritance
    Note over M: Pure merge — no LLM call<br/>28 inheritable rights fields<br/>Visual fields never overwritten
    M-->>U: Merged record<br/>CONTRACT_INHERITED (0.8) / CONTRACT_AMBIGUOUS (0.5)
```

---

## 4. Consolidation Decision Types

```mermaid
flowchart LR
    IN["LLM result + NER result"] --> D{"Per-field<br/>comparison"}
    D -->|"both agree"| A["AGREED<br/>confidence 0.9–1.0"]
    D -->|"values differ"| B["CONFLICT<br/>0.7–0.9<br/>LLM arbitrates"]
    D -->|"LLM only"| C["LLM_ONLY<br/>0.5–0.7"]
    D -->|"NER only"| E["NER_ONLY<br/>0.6–0.8"]
    D -->|"neither"| F["MISSING<br/>0.0"]
    F -.->|"contract provided"| G["CONTRACT_INHERITED 0.8<br/>CONTRACT_AMBIGUOUS 0.5"]

    style A fill:#e0f0e0
    style B fill:#fff0e0
    style F fill:#ffe0e0
    style G fill:#e0e8ff
```

---

## 5. API Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/llm-extract` | **Main** — full pipeline (document or image, auto-routed) |
| `POST /api/apply-inheritance` | Contract → work rights merge (no LLM call) |
| `POST /api/ocr-universal` | OCR only |
| `POST /api/ner-extract` | OCR + NER only |
| `GET /pair` | Paired inspection UI (contract + works, concurrent) |
| `GET /v2`, `GET /` | Single-file web UI |
| `GET /health` | Service status |

---

## 6. Current Status

| Component | Status |
|---|---|
| ② OCR | ✅ Operational — 4-provider fallback |
| ③ LLM extraction | ✅ Operational |
| ④ NER | ✅ Operational (local CPU, no external transmission) |
| ⑤ Image VLM | ✅ Operational — 3-tier fallback |
| ⑥ Consolidation | ✅ Operational |
| Contract inheritance | ✅ Operational — fills 19 of Muhayu's 20 work fields |
| **Video / audio track** | ❌ **Not implemented** (P3 guard) — see `video_track_implementation_plan_20260731_EN.md` |
| **RRN masking** | ❌ **Not implemented** — see `주민등록번호_마스킹_설계검토_20260731.md` |

*Korean version of this document: `docs/파이프라인_다이어그램_20260731.md`*
