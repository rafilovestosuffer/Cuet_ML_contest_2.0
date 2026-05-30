LABELS = [
    "Drought", "Earthquake", "Flood", "Human Damage",
    "Landslides", "Non Disaster", "Tropical Storm", "Wildfire",
]
LABEL2IDX = {lab: i for i, lab in enumerate(LABELS)}
IDX2LABEL = {i: lab for i, lab in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

SUBMISSION_LABEL_COLUMN = "category"

SUBMISSION_LABEL_COLUMN_TYPO = "categry"
