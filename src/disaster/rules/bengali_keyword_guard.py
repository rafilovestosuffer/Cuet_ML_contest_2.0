"""Confidence-gated Bengali keyword override with an 'aftermath guard'.

Logic
-----
1. AFTERMATH GUARD: if the caption matches '[hazard] + por' (e.g. "after the
   wildfire / earthquake"), it describes the *aftermath / recovery*, not the
   live event -> DO NOT override.
2. Otherwise, the keywords daabaanal (wildfire) and khaaraa (drought) override
   the prediction ONLY IF:
     (a) the text-ensemble argmax already agrees with the keyword, AND
     (b) the text-ensemble confidence (softmax max) > 0.90.

This conservative triple guard changed only 3 predictions on the contest test
set, out of 182 (wildfire) + 158 (drought) raw keyword hits.
"""
import re

# "[hazard] + por" == after the <hazard>
AFTERMATH_RE = re.compile(r"(দাবানলের|খরার|বন্যার|ঝড়ের|ভূমিকম্পের)\s*পর")

KEYWORD_TO_LABEL = {
    "দাবানল": "Wildfire",   # daabaanal
    "খরা":   "Drought",     # khaaraa
}

CONFIDENCE_THRESHOLD = 0.90


def keyword_override(context: str, text_argmax_label: str,
                     text_confidence: float):
    """Return target label name if the guarded rule fires, else None."""
    ctx = str(context)
    if AFTERMATH_RE.search(ctx):
        return None  # aftermath guard: skip
    for kw, label in KEYWORD_TO_LABEL.items():
        if kw in ctx:
            if text_argmax_label == label and text_confidence > CONFIDENCE_THRESHOLD:
                return label
    return None
