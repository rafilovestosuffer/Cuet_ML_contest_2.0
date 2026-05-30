# RECONSTRUCTED REFERENCE IMPLEMENTATION — generated from specification.
# This is NOT the original training code and is NOT verified to reproduce the
# released OOF arrays. Reconcile with the author's original notebooks.

"""Pseudo-labeling for MuRIL (PL-MuRIL).

Recipe:
  1. Predict the test set with the trained MuRIL ensemble.
  2. Keep rows whose softmax-max ≥ 0.95 as hard pseudo-labels.
  3. Re-fine-tune MuRIL on gold + pseudo data → the `oof_pl_muril` branch.

`select_pseudo_labels` is pure array logic and is unit-testable; the re-finetune
loop reuses disaster.train.train_text and is reconstructed.
"""
import argparse

import numpy as np
import pandas as pd

PSEUDO_THRESHOLD = 0.95


def select_pseudo_labels(test_probs: np.ndarray, test_df: pd.DataFrame,
                         threshold: float = PSEUDO_THRESHOLD) -> pd.DataFrame:
    """Return a frame of confidently-predicted test rows as hard pseudo-labels.

    Output columns mirror the training frame: image_id, context, label (int).
    """
    conf = test_probs.max(1)
    hard = test_probs.argmax(1)
    keep = conf >= threshold
    out = test_df.loc[keep, ["image_id", "context"]].copy()
    out["label"] = hard[keep]
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pseudo_label_muril.yaml")
    ap.add_argument("--test-probs", default="artifacts/test_muril_large.npy")
    ap.add_argument("--test-csv", default="data/raw/Disaster_test.csv")
    args = ap.parse_args()

    test_probs = np.load(args.test_probs)
    test_df = pd.read_csv(args.test_csv, encoding="utf-8-sig")
    test_df.columns = [c.strip() for c in test_df.columns]
    pseudo = select_pseudo_labels(test_probs, test_df)
    print(f"Selected {len(pseudo)}/{len(test_df)} pseudo-labeled rows "
          f"(conf ≥ {PSEUDO_THRESHOLD}).")
    print("Re-finetune step is reconstructed — invoke train_text on the "
          "gold+pseudo concatenation with configs/pseudo_label_muril.yaml.")


if __name__ == "__main__":
    main()
