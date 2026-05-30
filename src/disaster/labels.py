"""Canonical label space. Order is FIXED — every OOF/test array uses it."""

LABELS = [
    "Drought", "Earthquake", "Flood", "Human Damage",
    "Landslides", "Non Disaster", "Tropical Storm", "Wildfire",
]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
IDX2LABEL = {i: l for i, l in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# The contest submission column is intentionally misspelled. Keep it.
SUBMISSION_LABEL_COLUMN = "categry"
