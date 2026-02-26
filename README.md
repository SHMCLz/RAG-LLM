# Clinical RAG Backend

A local-network retrieval-augmented generation (RAG) backend designed for clinical patient-education workflows with mandatory human review.


## Features
- PDF ingestion pipeline: text extraction, section splitting, metadata annotation, and `segments.jsonl` generation
- Hybrid retrieval: TF-IDF sparse retrieval (required) and optional dense embeddings + FAISS (optional)
- Optional outbound-only external retrieval gateway using de-identified, abstracted queries
- Mandatory human approval step prior to delivery
- Audit logging and exposure confirmation

## Repository layout
- `src/clinical_rag/` – core library (retrieval, privacy, policies, audit, review queue)
- `apps/api/` – FastAPI service
- `tools/` – ingestion + indexing utilities
- `docker/` – containerization
- `tests/` – invariant tests

# RAG-LLM
For LLM-Assisted Preoperative Communication in Prostate Cancer Patients

