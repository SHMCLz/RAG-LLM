from __future__ import annotations
import re
from typing import List, Tuple

UNSUITABLE_PATTERNS: List[Tuple[str, str]] = [
    ("hospital_regulations", r"\b(parking|refund|billing|complaint|appointment policy|rule|regulation)\b"),
    ("medical_law", r"\b(lawsuit|legal|malpractice|sue|court)\b"),
    ("physician_privacy", r"\b(doctor\s*(phone|address)|physician\s*(phone|address)|personal\s*info)\b"),
    ("disruptive", r"\b(bypass|fake|forge|avoid\s*paying)\b"),
]
def classify_unsuitable(raw_query: str) -> List[str]:
    q = raw_query.lower()
    hits: List[str] = []
    for label, pattern in UNSUITABLE_PATTERNS:
        if re.search(pattern, q):
            hits.append(label)
    return hits
