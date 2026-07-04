# Clinical RAG Backend

A local-network retrieval-augmented generation (RAG) backend for clinical
patient-education workflows, with mandatory human review before any response is
delivered to a patient.

Designed for a predominantly Chinese-language preoperative setting with a mixed Chinese–English medical knowledge base.

## Features
- PDF ingestion pipeline: text extraction, section splitting, metadata
  annotation, and `segments.jsonl` generation.
- Hybrid retrieval:
  - Sparse lexical retrieval (TF-IDF).
  - Dense semantic retrieval with bge-m3, a multilingual encoder that
    natively supports both Chinese and English, indexed with FAISS.
  - Weighted fusion of dense and sparse scores with guideline-level prioritization.
- Threshold-gated external retrieval through an outbound-only secure gateway,
  triggered only when local retrieval is insufficient. Only a de-identified,
  abstracted query is transmitted, and results are filtered to a curated
  allowlist of vetted medical sources.
- Privacy: de-identification of Chinese identifiers (names, bed/ward/room
  numbers, phone/ID/date/time) before any query representation is used or transmitted.
- Mandatory human review prior to delivery, with audit logging and exposure
  confirmation.

## Repository layout
- `src/clinical_rag/` – core library (retrieval, privacy, policies, gateway, audit, review)
- `apps/api/` – FastAPI service
- `tools/` – ingestion (`ingest_pdf.py`) + indexing (`build_index.py`) utilities
- `docker/` – containerization
- `tests/` – invariant tests

## Notes
- Research use only; not a medical device. All responses require human review.
- Patient data is not included in this repository.

For LLM-assisted preoperative communication in prostate cancer patients.
