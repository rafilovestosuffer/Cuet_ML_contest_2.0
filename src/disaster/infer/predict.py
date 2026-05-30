"""Full inference pipeline: frozen test arrays → submission.csv.

Applies the same ensemble + calibration pipeline as reproduce_ensemble.py
but on the test probability arrays.

Usage (from repo root):
    PYTHONPATH=src python -m disaster.infer.predict \\
        --test-csv data/Test/test.csv \\
        --out results/tables/submission.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from disaster.ensemble.alpha_mix import apply_alpha, search_alpha
from disaster.ensemble.bias_calibration import apply_bias, fit_bias_bootstrap
from disaster.ensemble.blend import apply_blend, search_blend_weights
from disaster.infer.make_submission import make_submission
from disaster.labels import IDX2LABEL, LABELS
from disaster.rules.bengali_keyword_guard import keyword_override
from disaster.rules.emoji_rules import emoji_override

A = Path("artifacts")
BASE = [
    "banglabert_base", "banglabert_multi", "muril_large",
    "eva02_large", "fusion_eva_muril", "pl_muril",
]


def _load_oofs():
    return [np.load(A / f"oof_{n}.npy") for n in BASE]


def _load_tests():
    return [np.load(A / f"test_{n}.npy") for n in BASE]


def build_ensemble(oofs, y, n_restarts=50):
    """Fit blend weights + alpha + bias on OOF; return (w, alpha, bias)."""
    w, _ = search_blend_weights(oofs, y, n_restarts=n_restarts)
    blend_oof = apply_blend(oofs, w)
    stack_oof = np.load(A / "oof_stack.npy")
    alpha, _ = search_alpha(blend_oof, stack_oof, y)
    mix_oof = apply_alpha(blend_oof, stack_oof, alpha)
    bias, _ = fit_bias_bootstrap(mix_oof, y, n_boot=15)
    return w, alpha, bias


def apply_ensemble(tests, w, alpha, bias):
    """Apply fitted ensemble to test arrays."""
    blend_test = apply_blend(tests, w)
    stack_test = np.load(A / "test_stack.npy")
    mix_test = apply_alpha(blend_test, stack_test, alpha)
    return apply_bias(mix_test, bias)


def apply_rules(logits: np.ndarray, test_df: pd.DataFrame,
                corrections_csv: Path | None = None) -> np.ndarray:
    """Apply emoji and keyword overrides; log changes if corrections_csv given."""
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)

    pred = logits.argmax(1).copy()
    records = []

    # The keyword guard needs the text-ensemble confidence on the *test* rows:
    # mean softmax over the four text branches (BanglaBERT ×2, MuRIL, PL-MuRIL),
    # matching the notebook's `text_ens` definition.
    text_tests = [np.load(A / f"test_{n}.npy")
                  for n in ("banglabert_base", "banglabert_multi", "muril_large", "pl_muril")]
    text_mean = np.mean(text_tests, axis=0)  # (N, 8)

    for i, row in test_df.iterrows():
        ctx = str(row.get("context", ""))
        row_i = i if corrections_csv else list(test_df.index).index(i)

        # 1. Emoji override (highest priority)
        emo = emoji_override(ctx)
        if emo is not None:
            new_idx = list(LABELS).index(emo)
            if pred[row_i] != new_idx:
                records.append(dict(
                    row=row_i, image_id=row.get("image_id", row_i),
                    rule="emoji", keyword=emo,
                    old=IDX2LABEL[pred[row_i]], new=emo,
                    confidence=float(probs[row_i].max()),
                ))
                pred[row_i] = new_idx
            continue

        # 2. Keyword guard
        txt_argmax = IDX2LABEL[int(text_mean[row_i].argmax())]
        txt_conf = float(text_mean[row_i].max())
        kw = keyword_override(ctx, txt_argmax, txt_conf)
        if kw is not None:
            new_idx = list(LABELS).index(kw)
            if pred[row_i] != new_idx:
                records.append(dict(
                    row=row_i, image_id=row.get("image_id", row_i),
                    rule="keyword", keyword=kw,
                    old=IDX2LABEL[pred[row_i]], new=kw,
                    confidence=txt_conf,
                ))
                pred[row_i] = new_idx

    if corrections_csv is not None and records:
        corr_df = pd.DataFrame(records)
        Path(corrections_csv).parent.mkdir(parents=True, exist_ok=True)
        corr_df.to_csv(corrections_csv, index=False)
        print(f"  Rules applied {len(records)} overrides → {corrections_csv}")
    elif corrections_csv is not None:
        print("  Rules: 0 overrides applied.")

    return pred


def run(test_csv: str, out_csv: str = "results/tables/submission.csv",
        corrections_csv: str = "results/tables/applied_corrections.csv",
        sample_submission_csv: str | None = None):
    """Full pipeline: load OOFs → fit ensemble → apply to test → rules → CSV."""
    print("Loading OOF arrays …")
    y = pd.read_csv("data/folds/folds_canonical.csv",
                    encoding="utf-8-sig")["label"].values
    oofs = _load_oofs()
    tests = _load_tests()

    print("Fitting ensemble (blend + alpha + bias) on OOF …")
    w, alpha, bias = build_ensemble(oofs, y)

    print("Applying ensemble to test arrays …")
    logits = apply_ensemble(tests, w, alpha, bias)

    print("Loading test CSV for rule application …")
    test_df = pd.read_csv(test_csv, encoding="utf-8-sig")
    test_df.columns = [c.strip() for c in test_df.columns]
    test_df = test_df.reset_index(drop=True)

    print("Applying text rules …")
    pred = apply_rules(logits, test_df,
                       corrections_csv=Path(corrections_csv))

    sub = make_submission(test_df["image_id"], pred, out_csv,
                          sample_submission_csv=sample_submission_csv)
    label_col = [c for c in sub.columns if c != "image_id"][0]
    print(f"Submission written → {out_csv}  ({len(sub)} rows, label col '{label_col}')")
    print(f"Label distribution:\n{sub[label_col].value_counts().to_string()}")
    return sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True,
                        help="Path to test.csv (needs image_id + context columns)")
    parser.add_argument("--out", default="results/tables/submission.csv")
    parser.add_argument("--corrections",
                        default="results/tables/applied_corrections.csv")
    parser.add_argument("--sample-submission", default=None,
                        help="Auto-detect the label column from this file "
                             "(handles category vs categry)")
    args = parser.parse_args()
    run(args.test_csv, args.out, args.corrections, args.sample_submission)
