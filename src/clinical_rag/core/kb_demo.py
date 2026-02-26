from __future__ import annotations
from typing import List, Dict, Any

def demo_segments() -> List[Dict[str, Any]]:
    return [
        {"seg_id":"guideline_1","text":"Follow surgeon instructions after radical prostatectomy. Showering is typically allowed once the wound is sealed and dressings are removed.",
         "source":"guideline_demo","doc_type":"guideline","pub_year":2023,"category":"postop_care"},
        {"seg_id":"edu_1","text":"Seek medical attention for fever, increasing pain, redness, swelling, or wound discharge.",
         "source":"education_demo","doc_type":"education","pub_year":2022,"category":"warnings"},
    ]
