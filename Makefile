.PHONY: install test lint run ingest index

install:
	pip install -r requirements.txt
	

test:
	pytest -q

lint:
	ruff check .

run:
	uvicorn RAG_bot.main_api:app --host 0.0.0.0 --port 8000

ingest:
	python tools/ingest_pdf.py --pdf "$(PDF)" --out data/segments.jsonl --doc-type guideline --source "$(SOURCE)" --pub-year $(YEAR) --qc-json data/qc.json

index:
	python tools/build_index.py --segments data/segments.jsonl --index-dir runtime/index
