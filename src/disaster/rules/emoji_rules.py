"""Deterministic emoji -> class overrides.

NOTE: On the contest test set these glyphs occur in ~1 row, so this rule is a
low-risk safeguard rather than a meaningful contributor. Kept for completeness
and transparency; every change is logged by the caller.
"""
from disaster.labels import LABEL2IDX

EMOJI_TO_LABEL = {
    "📍": "Non Disaster", "😍": "Non Disaster", "🥰": "Non Disaster",
    "📌": "Non Disaster", "✨": "Non Disaster",
    "🚨": "Landslides", "⚠": "Landslides",
    "♻": "Human Damage", "🚱": "Human Damage",
}


def emoji_override(context: str):
    """Return target label name if an emoji rule fires, else None."""
    for glyph, label in EMOJI_TO_LABEL.items():
        if glyph in str(context):
            return label
    return None
