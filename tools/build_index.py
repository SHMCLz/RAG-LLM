#!/usr/bin/env python3
"""Build retrieval indices from segments.jsonl.

Outputs:
- TF-IDF artifacts (vectorizer + sparse matrix) to INDEX_DIR/tfidf.pkl
- Optional FAISS dense index to INDEX_DIR/faiss.index + INDEX_DIR/ids.json (if deps installed)

Usage:
  python tools/build_index.py --segments data/segments.jsonl --index-dir runtime/index
"""
from __future__ import annotations
import argparse, json, pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse  # type: ignore

def load_segments(path: Path) -> List[Dict]:
    segs: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segs.append(json.loads(line))
    return segs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--ngram-max", type=int, default=2)
    ap.add_argument("--dense", action="store_true", help="also build dense FAISS index if deps available")
    ap.add_argument("--dense-model", default="sentence-transformers/all-mpnet-base-v2")
    args = ap.parse_args()

    seg_path = Path(args.segments)
    idx_dir = Path(args.index_dir)
    idx_dir.mkdir(parents=True, exist_ok=True)

    segs = load_segments(seg_path)
    texts = [s["text"] for s in segs]
    seg_ids = [s["seg_id"] for s in segs]

    vec = TfidfVectorizer(min_df=args.min_df, ngram_range=(1, args.ngram_max))
    X = vec.fit_transform(texts)

    with (idx_dir / "tfidf.pkl").open("wb") as f:
        pickle.dump({"vectorizer": vec, "matrix": X, "seg_ids": seg_ids}, f)

    print(f"Saved TF-IDF -> {idx_dir/'tfidf.pkl'} (n={len(segs)})")

    if args.dense:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import faiss  # type: ignore
            model = SentenceTransformer(args.dense_model, device="cpu")
            embs = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)
            embs = np.asarray(embs, dtype="float32")
            index = faiss.IndexFlatIP(embs.shape[1])
            index.add(embs)
            faiss.write_index(index, str(idx_dir / "faiss.index"))
            (idx_dir / "ids.json").write_text(json.dumps(seg_ids), encoding="utf-8")
            print(f"Saved FAISS -> {idx_dir/'faiss.index'}")
        except Exception as e:
            print(f"Dense index skipped: {e}")

if __name__ == "__main__":
    main()
