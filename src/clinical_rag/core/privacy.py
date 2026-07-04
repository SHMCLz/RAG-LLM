from __future__ import annotations
import re
from typing import List

# ---------------------------------------------------------------------------
# De-identification for a predominantly Chinese-language clinical setting.
# Patient queries are mostly in Chinese; identifiers such as names, bed
# numbers, ward/room numbers, phone numbers, ID card numbers, dates and times
# are removed before any query representation is used or transmitted.
# ---------------------------------------------------------------------------

# Note: Chinese text has no word boundaries (\b) around digits, so numeric
# identifiers are matched without \b (ID first to avoid partial phone matches).
_PATTERNS = [
    # Chinese resident ID card (18 digits, last may be X) - match before phone
    (re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"), "[ID]"),
    # Chinese mobile phone (11 digits starting with 1) and generic 11-digit runs
    (re.compile(r"(?<!\d)1\d{10}(?!\d)"), "[PHONE]"),
    (re.compile(r"(?<!\d)\d{11}(?!\d)"), "[PHONE]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    # Dates: 2024-01-01 / 2024/1/1 and Chinese 2024年1月1日
    (re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"), "[DATE]"),
    (re.compile(r"\d{2,4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"), "[DATE]"),
    (re.compile(r"\d{1,2}\s*月\s*\d{1,2}\s*日"), "[DATE]"),
    # Times: 12:30 / 12：30 / 12点30
    (re.compile(r"\b\d{1,2}[:：]\d{2}\b"), "[TIME]"),
    (re.compile(r"\d{1,2}\s*点(\s*\d{1,2}\s*分)?"), "[TIME]"),
]

# Bed / ward / room identifiers, e.g. 3床 / 12号床 / 5病区 / 301房间 / 床号3
_BED_PATTERNS = [
    (re.compile(r"床号\s*\d+"), "[BED]"),
    (re.compile(r"\d+\s*号?\s*床"), "[BED]"),
    (re.compile(r"\d+\s*病区"), "[WARD]"),
    (re.compile(r"\d+\s*(?:房间|病房)"), "[ROOM]"),
]

# Chinese personal names, typically following a title/relation cue.
# We remove the name token after common cues to avoid over-stripping.
_NAME_CUE = re.compile(
    r"(我叫|我是|姓名[:：]?|患者[:：]?|病人[:：]?|名字[:：]?|叫做)\s*([\u4e00-\u9fff]{2,4})"
)
# Standalone surname+title (e.g. 张先生 / 李女士 / 王阿姨 / 刘叔叔)
_NAME_TITLE = re.compile(r"[\u4e00-\u9fff]{1,2}(先生|女士|小姐|阿姨|叔叔|大爷|大娘)")

# Chinese stopwords to drop when extracting key terms.
_STOP_ZH = set([
    "的", "了", "吗", "呢", "吧", "啊", "和", "与", "或", "是", "在", "我", "你",
    "他", "她", "它", "我们", "请问", "怎么", "如何", "什么", "可以", "需要", "会",
    "有", "没有", "这个", "那个", "一下", "一个", "还", "就", "都", "也", "被", "把",
])
_STOP_EN = set(["the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
                "with", "is", "are", "be", "can", "i", "you", "my", "me", "how",
                "what", "when", "please"])


def deidentify(text: str) -> str:
    """Remove direct and quasi-identifiers from a (mostly Chinese) query."""
    t = text or ""
    # Names first (before digits get masked so cues remain intact).
    t = _NAME_CUE.sub(lambda m: m.group(1) + "[NAME]", t)
    t = _NAME_TITLE.sub("[NAME]", t)
    # Bed / ward / room.
    for pat, rep in _BED_PATTERNS:
        t = pat.sub(rep, t)
    # Structured identifiers.
    for pat, rep in _PATTERNS:
        t = pat.sub(rep, t)
    # Any remaining long digit runs.
    t = re.sub(r"\b\d{6,}\b", "[NUMBER]", t)
    return t


def _tokenize_zh(text: str) -> List[str]:
    """Chinese word segmentation with jieba if available, else a char/bigram fallback."""
    try:
        import jieba  # type: ignore
        return [w.strip() for w in jieba.cut(text) if w.strip()]
    except Exception:
        # Fallback (jieba unavailable): split on non-Chinese/non-latin separators,
        # keeping contiguous Chinese runs and latin words as tokens. This avoids
        # the noisy sliding-window bigrams of a naive character split.
        return [t for t in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z]+", text) if t]


def abstract_query(deid_text: str) -> str:
    """Build a de-identified, abstracted key-term representation of the query.

    Supports Chinese (via segmentation) and English. Only key terms are kept;
    no raw sentence or identifier is retained.
    """
    q = re.sub(r"\s+", " ", (deid_text or "").strip())[:500]
    # Drop masked identifier placeholders from key terms.
    q_clean = re.sub(r"\[[A-Z]+\]", " ", q)

    tokens = _tokenize_zh(q_clean)
    key_terms: List[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in _STOP_EN or tok in _STOP_ZH:
            continue
        if re.fullmatch(r"[\W_]+", tok):
            continue
        if len(tok) == 1 and re.match(r"[\u4e00-\u9fff]", tok):
            # skip lone Chinese characters (low signal)
            continue
        key_terms.append(tok)
        if len(key_terms) >= 18:
            break

    head = " ".join(key_terms)
    return f"intent:patient_education; key_terms:{head}"
