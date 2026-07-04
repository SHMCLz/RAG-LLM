from __future__ import annotations
import re
from typing import List, Tuple

# Queries deemed unsuitable for AI-generated responses are routed to a
# standardized referral message by the human reviewer. Patterns cover both
# English and Chinese, since patient queries are predominantly in Chinese.
# These rules are an assistive pre-screen only; the final determination is
# made by the human content reviewer.
UNSUITABLE_PATTERNS: List[Tuple[str, str]] = [
    (
        "hospital_regulations",
        r"(parking|refund|billing|complaint|appointment policy|rule|regulation"
        r"|停车|退费|报销|收费|投诉|预约(规定|政策)|规章|制度|探视|陪护规定)",
    ),
    (
        "medical_law",
        r"(lawsuit|legal|malpractice|sue|court"
        r"|起诉|诉讼|打官司|医疗事故|医疗纠纷|法律|赔偿|索赔|鉴定)",
    ),
    (
        "physician_privacy",
        r"(doctor\s*(phone|address)|physician\s*(phone|address)|personal\s*info"
        r"|医生(电话|手机|住址|地址|微信|私人)|大夫(电话|手机)|个人隐私|私人联系方式)",
    ),
    (
        "disruptive",
        r"(bypass|fake|forge|avoid\s*paying"
        r"|插队|加塞|走后门|托关系|红包|伪造|造假|逃费|逃避付费|绕过)",
    ),
]


def classify_unsuitable(raw_query: str) -> List[str]:
    q = (raw_query or "").lower()
    hits: List[str] = []
    for label, pattern in UNSUITABLE_PATTERNS:
        if re.search(pattern, q):
            hits.append(label)
    return hits
