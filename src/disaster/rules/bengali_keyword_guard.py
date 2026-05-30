import re

AFTERMATH_RE = re.compile(r"(দাবানলের|খরার|বন্যার|ঝড়ের|ভূমিকম্পের)\s*পর")

KEYWORD_TO_LABEL = {
    "দাবানল": "Wildfire",
    "খরা":   "Drought",
}

CONFIDENCE_THRESHOLD = 0.90


def keyword_override(context: str, text_argmax_label: str,
                     text_confidence: float):
    ctx = str(context)
    if AFTERMATH_RE.search(ctx):
        return None
    for kw, label in KEYWORD_TO_LABEL.items():
        if kw in ctx:
            if text_argmax_label == label and text_confidence > CONFIDENCE_THRESHOLD:
                return label
    return None
