"""Build the submission CSV.

The submission label-column name is auto-detected from the contest's
sample_submission.csv when available (this is what the original notebook does
and is the safest behaviour — it adapts to whatever the grader expects, whether
that is `category` or the misspelled `categry`). When no sample_submission is
provided, falls back to `disaster.labels.SUBMISSION_LABEL_COLUMN`.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from disaster.labels import IDX2LABEL, SUBMISSION_LABEL_COLUMN


def detect_label_column(sample_submission_csv: str | Path) -> str:
    """Return the non-image_id column name from a sample submission file."""
    sample = pd.read_csv(sample_submission_csv, encoding="utf-8-sig")
    cols = [c for c in sample.columns if c.strip().lower() != "image_id"]
    if not cols:
        raise ValueError(f"No label column found in {sample_submission_csv}")
    return cols[0]


def make_submission(test_ids, pred_idx: np.ndarray,
                    out_csv: str | Path = "results/tables/submission.csv",
                    label_column: str | None = None,
                    sample_submission_csv: str | Path | None = None) -> pd.DataFrame:
    """Write a submission CSV.

    Parameters
    ----------
    test_ids : sequence of image ids (submission row order)
    pred_idx : (N,) integer class predictions
    label_column : explicit column name; overrides everything if given
    sample_submission_csv : if given (and label_column is None), the column name
        is auto-detected from this file (notebook behaviour)
    """
    if label_column is None:
        if sample_submission_csv is not None:
            label_column = detect_label_column(sample_submission_csv)
        else:
            label_column = SUBMISSION_LABEL_COLUMN

    labels = [IDX2LABEL[int(i)] for i in pred_idx]
    sub = pd.DataFrame({"image_id": list(test_ids), label_column: labels})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    return sub
