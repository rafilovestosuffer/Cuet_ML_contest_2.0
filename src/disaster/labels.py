"""Canonical label space. Order is FIXED — every OOF/test array uses it."""

LABELS = [
    "Drought", "Earthquake", "Flood", "Human Damage",
    "Landslides", "Non Disaster", "Tropical Storm", "Wildfire",
]
LABEL2IDX = {lab: i for i, lab in enumerate(LABELS)}
IDX2LABEL = {i: lab for i, lab in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

# Submission label column.
#
# IMPORTANT — provenance note: the project build prompt (CLAUDE.md) states the
# contest column is misspelled `categry`. However, the original notebook
# (notebooks/Final_notebook.ipynb) detects the column dynamically from the
# contest's sample_submission.csv and resolves it to the correctly-spelled
# `category`. The two authoritative sources disagree.
#
# To be safe, `make_submission` auto-detects the column from sample_submission
# when a path is provided (the notebook's behaviour). This constant is only the
# fallback used when no sample_submission is available. Set it to whatever the
# *actual* sample_submission.csv uses for your contest instance.
SUBMISSION_LABEL_COLUMN = "category"

# The build prompt's stated (misspelled) alternative, kept for reference.
SUBMISSION_LABEL_COLUMN_TYPO = "categry"
