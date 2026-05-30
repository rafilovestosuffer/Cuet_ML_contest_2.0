#!/usr/bin/env python3
"""Reproduce the paper's ensemble macro-F1 from the frozen OOF arrays.

Run:
    PYTHONPATH=src python scripts/reproduce_ensemble.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

from disaster.labels import LABELS
from disaster.ensemble.blend import search_blend_weights, apply_blend
from disaster.ensemble.alpha_mix import search_alpha, apply_alpha
from disaster.ensemble.bias_calibration import (
    fit_bias_single, fit_bias_bootstrap, apply_bias)

A = "artifacts/"
BASE = ["oof_banglabert_base", "oof_banglabert_multi", "oof_muril_large",
        "oof_eva02_large", "oof_fusion_eva_muril", "oof_pl_muril"]


def f1(p, y):
    return f1_score(y, p.argmax(1), average="macro")


def main():
    y = pd.read_csv("data/folds/folds_canonical.csv",
                    encoding="utf-8-sig")["label"].values
    oofs = [np.load(A + n + ".npy") for n in BASE]
    stack = np.load(A + "oof_stack.npy")

    print("Individual OOF macro-F1:")
    for n, o in zip(BASE, oofs):
        print(f"  {n:26s} {f1(o, y):.5f}")
    print(f"  {'oof_stack':26s} {f1(stack, y):.5f}")

    w, s_blend = search_blend_weights(oofs, y, n_restarts=50)
    blend = apply_blend(oofs, w)
    print(f"\nDirichlet blend         : {s_blend:.5f}")
    print("  weights:", {n: round(float(wi), 3) for n, wi in zip(BASE, w)})

    a, s_mix = search_alpha(blend, stack, y)
    mix = apply_alpha(blend, stack, a)
    print(f"alpha-mix (a={a:.3f})     : {s_mix:.5f}")

    b1, f_single = fit_bias_single(mix, y)
    print(f"+ bias (single-shot)    : {f_single:.5f}")
    b2, f_stable = fit_bias_bootstrap(mix, y, n_boot=15)
    print(f"+ bias (bootstrap-stable): {f_stable:.5f}  <-- submitted")

    pred = apply_bias(mix, b2).argmax(1)
    print("\nPer-class report (bootstrap-stable):")
    print(classification_report(y, pred, target_names=LABELS, digits=4))


if __name__ == "__main__":
    main()
