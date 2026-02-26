from __future__ import annotations
import re
from typing import List

_PATTERNS = [
    (re.compile(r"\b\d{11}\b"), "[PHONE]"),
    (re.compile(r"\b\d{17}[0-9Xx]\b"), "[ID]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"), "[DATE]"),
    (re.compile(r"\b\d{1,2}[:：]\d{2}\b"), "[TIME]"),
]

_STOP = set(["the","a","an","and","or","of","to","in","on","for","with","is","are"])

def deidentify(text: str) -> str:
    t = text
    for pat, rep in _PATTERNS:
        t = pat.sub(rep, t)
    t = re.sub(r"\b\d{6,}\b", "[NUMBER]", t)
    return t

def abstract_query(deid_text: str) -> str:
    q = re.sub(r"\s+", " ", deid_text.strip())[:500]
    tokens = [t for t in re.findall(r"[A-Za-z]+", q.lower()) if t not in _STOP]
    head = " ".join(tokens[:18])
    return f"intent:patient_education; key_terms:{head}"
