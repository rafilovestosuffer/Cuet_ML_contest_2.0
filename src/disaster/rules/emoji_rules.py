EMOJI_TO_LABEL = {
    "📍": "Non Disaster", "😍": "Non Disaster", "🥰": "Non Disaster",
    "📌": "Non Disaster", "✨": "Non Disaster",
    "🚨": "Landslides", "⚠": "Landslides",
    "♻": "Human Damage", "🚱": "Human Damage",
}


def emoji_override(context: str):
    for glyph, label in EMOJI_TO_LABEL.items():
        if glyph in str(context):
            return label
    return None
