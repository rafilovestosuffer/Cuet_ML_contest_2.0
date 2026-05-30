"""Build the submission CSV. Preserves the contest's 'categry' column spelling."""
from pathlib import Path
import numpy as np
import pandas as pd

from disaster.labels import IDX2LABEL, SUBMISSION_LABEL_COLUMN


def make_submission(test_ids, pred_idx: np.ndarray,
                    out_csv: str | Path = "results/tables/submission.csv") -> pd.DataFrame:
    labels = [IDX2LABEL[int(i)] for i in pred_idx]
    sub = pd.DataFrame({"image_id": list(test_ids),
                        SUBMISSION_LABEL_COLUMN: labels})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    return sub
