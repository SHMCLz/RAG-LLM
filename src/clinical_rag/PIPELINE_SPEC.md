# Pipeline specification

This repository includes an end-to-end knowledge base pipeline:
1) PDF extraction (pdfplumber primary, PyPDF2 fallback)
2) Section splitting (heuristics using numbered headings)
3) Segment metadata annotation (source/type/year/section/page range/title/doi)
4) Output `segments.jsonl`
5) Build retrieval indices: TF-IDF (required) and optional dense index (FAISS)

## 1) Ingest PDF -> segments.jsonl
```bash
python tools/ingest_pdf.py --pdf "path/to/guideline.pdf" --out data/segments.jsonl --doc-type guideline --source CACA --pub-year 2024 --qc-json data/qc.json
```

## 2) Build index artifacts
```bash
python tools/build_index.py --segments data/segments.jsonl --index-dir runtime/index
python tools/build_index.py --segments data/segments.jsonl --index-dir runtime/index --dense
```
