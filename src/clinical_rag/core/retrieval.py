from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class LocalHybridRetriever:
    """Deployment-consistent retrieval core:
    - Sparse TF-IDF
    - Optional dense embeddings + FAISS IndexFlatIP if deps installed
    - Weighted fusion + guideline priority
    """
    def __init__(self, settings):
        self.settings = settings
        self._segments: List[Dict[str, Any]] = []
        self._vec = TfidfVectorizer(min_df=1, ngram_range=(1,2))
        self._X = None

        # dense optional
        self._dense = None
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self._dense_model = SentenceTransformer(settings.dense_embed_model, device="cpu")
            self._faiss = faiss.IndexFlatIP(768)
            self._dense_ids: List[str] = []
            self._dense = True
        except Exception:
            self._dense = None

    def load_segments(self, segments: List[Dict[str, Any]]) -> None:
        self._segments = segments
        texts = [s.get("text","") for s in segments]
        self._X = self._vec.fit_transform(texts)
        if self._dense:
            vecs = self._dense_model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
            vecs = np.asarray(vecs, dtype="float32")
            self._faiss.add(vecs)
            self._dense_ids = [s["seg_id"] for s in segments]

    def _sparse(self, query: str, k: int) -> List[Tuple[str, float]]:
        if self._X is None:
            return []
        q = self._vec.transform([query])
        scores = (self._X @ q.T).toarray().ravel()
        idx = np.argsort(-scores)[:k]
        return [(self._segments[i]["seg_id"], float(scores[i])) for i in idx.tolist()]

    def _dense_search(self, query: str, k: int) -> List[Tuple[str, float]]:
        if not self._dense:
            return []
        qv = self._dense_model.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32")
        scores, idx = self._faiss.search(qv, k)
        out = []
        for s, i in zip(scores[0].tolist(), idx[0].tolist()):
            if i == -1:
                continue
            out.append((self._dense_ids[i], float(s)))
        return out

    @staticmethod
    def _minmax(pairs: List[Tuple[str, float]]) -> dict[str, float]:
        if not pairs:
            return {}
        vals = [s for _, s in pairs]
        lo, hi = min(vals), max(vals)
        if abs(hi-lo) < 1e-12:
            return {i: 1.0 for i,_ in pairs}
        return {i: (s-lo)/(hi-lo) for i,s in pairs}

    def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], float]:
        dense = self._dense_search(query, self.settings.k_dense)
        sparse = self._sparse(query, self.settings.k_sparse)

        dn = self._minmax(dense)
        sn = self._minmax(sparse)
        fused: dict[str, float] = {}
        for sid, s in dn.items():
            fused[sid] = fused.get(sid, 0.0) + self.settings.w_dense * s
        for sid, s in sn.items():
            fused[sid] = fused.get(sid, 0.0) + self.settings.w_sparse * s

        # gather segments and apply guideline boost
        seg_by_id = {s["seg_id"]: s for s in self._segments}
        cand_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[: self.settings.k_fused*3]
        segs = [seg_by_id[i] for i in cand_ids if i in seg_by_id]
        for seg in segs:
            if seg.get("doc_type") == "guideline":
                fused[seg["seg_id"]] *= self.settings.guideline_boost

        segs.sort(key=lambda s: fused.get(s["seg_id"], 0.0), reverse=True)
        segs = segs[: self.settings.k_fused]
        top = float(fused.get(segs[0]["seg_id"], 0.0)) if segs else 0.0
        return segs, top
