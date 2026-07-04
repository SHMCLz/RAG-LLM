#!/usr/bin/env python3
"""PDF -> segments.jsonl

Implements the pipeline specification pipeline:
- PDF extraction via pdfplumber (primary) and PyPDF2 (fallback)
- Chapter/section-level splitting (heuristic by numbered headings)
- Metadata annotation (source/doc_type/pub_year/section title/page range)
- Output: JSON Lines segments file suitable for RAG indexing

Usage:
  python tools/ingest_pdf.py --pdf <path.pdf> --out data/segments.jsonl --doc-type guideline --source CACA --pub-year 2024
"""
from __future__ import annotations
import argparse, json, re, hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional

HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s*[\u2002\u2003\u00A0\s]*[—\-–]*\s*(.+?)\s*$")
# Common in guidelines: "1  Epidemiology", "2.1  PC screening"
ALT_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s*[\u2002\u2003\u00A0\s]+(.+?)\s*$")

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8', errors='ignore')).hexdigest()

def qc_sample_pages(pages: List[str], sample_n: int = 6) -> dict:
    # Simple proxy QC: percent non-empty chars on sampled pages.
    import random
    if not pages:
        return {"sample_n": 0, "nonempty_ratio": 0.0}
    idxs = list(range(len(pages)))
    random.seed(7)
    pick = idxs[:sample_n] if len(idxs) <= sample_n else random.sample(idxs, sample_n)
    ratios = []
    for i in pick:
        t = pages[i] or ""
        non_ws = sum(1 for ch in t if not ch.isspace())
        ratios.append(non_ws / max(1, len(t)))
    return {"sample_n": len(pick), "nonempty_ratio_mean": float(sum(ratios)/len(ratios)) if ratios else 0.0, "pages": [p+1 for p in pick]}

def extract_pages(pdf: Path) -> List[str]:

    pages: List[str] = []
    # Primary: pdfplumber
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(pdf)) as p:
            for pg in p.pages:
                t = pg.extract_text() or ""
                pages.append(t)
        if any(p.strip() for p in pages):
            return pages
    except Exception:
        pass

    # Fallback: PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore
        r = PdfReader(str(pdf))
        for pg in r.pages:
            pages.append(pg.extract_text() or "")
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")
    return pages

def normalize(text: str) -> str:
    t = text.replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    # Fix hyphenation across line breaks
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = re.sub(r"\s+\n", "\n", t)
    return t.strip()

def detect_title_and_doi(pages: List[str]) -> Tuple[Optional[str], Optional[str]]:
    first = pages[0] if pages else ""
    title = None
    doi = None
    # DOI
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", first, re.I)
    if m:
        doi = m.group(0)
    # Title line: heuristic: find a line with "guidelines" or "Guideline"
    for line in first.splitlines():
        l = line.strip()
        if not l:
            continue
        if "guideline" in l.lower() and len(l) > 20:
            title = l
            break
    # If not found, pick the longest non-author line in first page top
    if title is None:
        cand = [ln.strip() for ln in first.splitlines() if ln.strip()]
        cand = [c for c in cand if len(c) > 20 and "et al" not in c.lower()]
        if cand:
            title = max(cand, key=len)
    return title, doi

def find_headings(pages: List[str]) -> List[Tuple[int, str, str]]:
    """Return list of (page_index, section_id, section_title)."""
    out: List[Tuple[int, str, str]] = []
    for pi, raw in enumerate(pages):
        txt = normalize(raw)
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip page headers like "Page 3 of 34"
            if re.search(r"Page\s+\d+\s+of\s+\d+", line, re.I):
                continue
            m = HEADING_RE.match(line) or ALT_HEADING_RE.match(line)
            if m:
                sec = m.group(1)
                title = m.group(2).strip()
                # avoid false positives like TNM tables "T1"
                if len(title) < 3:
                    continue
                # typical guideline section numbers <= 2 dots
                if sec.count(".") <= 3:
                    out.append((pi, sec, title))
                break  # first heading per page is enough
    # de-duplicate consecutive identical headings
    dedup: List[Tuple[int, str, str]] = []
    for h in out:
        if dedup and dedup[-1][1] == h[1] and dedup[-1][2] == h[2]:
            continue
        dedup.append(h)
    return dedup

def build_sections(pages: List[str], headings: List[Tuple[int, str, str]]) -> List[Dict]:
    """Build sections by spanning from heading start to next heading."""
    if not headings:
        # whole doc as one
        full = "\n\n".join(normalize(p) for p in pages)
        return [{"section_id":"0", "section_title":"document", "page_start":1, "page_end":len(pages), "text":full}]

    # Build boundaries by page index
    bounds = headings + [(len(pages), "__END__", "")]
    sections: List[Dict] = []
    for i in range(len(headings)):
        p_start, sec_id, sec_title = headings[i]
        p_end = bounds[i+1][0]  # start page of next heading
        chunk_pages = pages[p_start:p_end] if p_end > p_start else pages[p_start:p_start+1]
        text = "\n\n".join(normalize(p) for p in chunk_pages)
        sections.append({
            "section_id": sec_id,
            "section_title": sec_title,
            "page_start": p_start+1,
            "page_end": p_end if p_end > p_start else p_start+1,
            "text": text
        })
    return sections

def chunk_text(text: str, max_chars: int = 2200, overlap: int = 200) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [c.strip() for c in chunks if c.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--doc-type", default="guideline")
    ap.add_argument("--source", default=None)
    ap.add_argument("--pub-year", type=int, default=None)
    ap.add_argument("--category", default=None)
    ap.add_argument("--max-chars", type=int, default=2200)
    ap.add_argument("--qc-json", default=None, help="write extraction QC summary to this path")
    ap.add_argument("--overlap", type=int, default=200)
    args = ap.parse_args()

    pdf = Path(args.pdf)
    pages = extract_pages(pdf)
    qc = qc_sample_pages(pages)

    title, doi = detect_title_and_doi(pages)
    headings = find_headings(pages)
    sections = build_sections(pages, headings)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with outp.open("w", encoding="utf-8") as f:
        for sec in sections:
            for j, c in enumerate(chunk_text(sec["text"], max_chars=args.max_chars, overlap=args.overlap)):
                seg_id = _sha1(f"{pdf.name}|{sec['section_id']}|{j}|{c[:80]}")
                rec = {
                    "seg_id": seg_id,
                    "text": c,
                    "source": args.source or pdf.stem,
                    "doc_type": args.doc_type,
                    "pub_year": args.pub_year,
                    "category": args.category,
                    "title": title,
                    "doi": doi,
                    "section_id": sec["section_id"],
                    "section_title": sec["section_title"],
                    "page_start": sec["page_start"],
                    "page_end": sec["page_end"],
                    "file_name": pdf.name,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1

    if args.qc_json:
        Path(args.qc_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.qc_json).write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"Wrote {n} segments -> {outp}")

if __name__ == "__main__":
    main()
