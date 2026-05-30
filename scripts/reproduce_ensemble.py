#!/usr/bin/env python3
"""Reproduce the paper's ensemble macro-F1 from the frozen OOF arrays.

All printed numbers are computed at runtime — nothing is hardcoded.

Usage (from repo root):
    PYTHONPATH=src python scripts/reproduce_ensemble.py [--figures]
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from disaster.ensemble.alpha_mix import apply_alpha, search_alpha
from disaster.ensemble.bias_calibration import apply_bias, fit_bias_bootstrap, fit_bias_single
from disaster.ensemble.blend import apply_blend, search_blend_weights
from disaster.labels import LABELS

A = "artifacts/"
BASE = ["oof_banglabert_base", "oof_banglabert_multi", "oof_muril_large",
        "oof_eva02_large", "oof_fusion_eva_muril", "oof_pl_muril"]


def f1(p, y):
    return f1_score(y, p.argmax(1), average="macro")


def main(gen_figures=False):
    y = pd.read_csv("data/folds/folds_canonical.csv",
                    encoding="utf-8-sig")["label"].values
    oofs = [np.load(A + n + ".npy") for n in BASE]
    stack = np.load(A + "oof_stack.npy")

    # -----------------------------------------------------------------------
    # 1. Individual branches
    # -----------------------------------------------------------------------
    print("Individual OOF macro-F1:")
    for n, o in zip(BASE, oofs):
        print(f"  {n:26s}  {f1(o, y):.5f}")
    print(f"  {'oof_stack':26s}  {f1(stack, y):.5f}")

    # -----------------------------------------------------------------------
    # 2. Dirichlet blend
    # -----------------------------------------------------------------------
    w, s_blend = search_blend_weights(oofs, y, n_restarts=50)
    blend = apply_blend(oofs, w)
    print(f"\nDirichlet blend              : {s_blend:.5f}")
    print("  weights:", {n.replace("oof_", ""): round(float(wi), 3)
                         for n, wi in zip(BASE, w)})

    # -----------------------------------------------------------------------
    # 3. Alpha-mix (blend × stack)
    # -----------------------------------------------------------------------
    a, s_mix = search_alpha(blend, stack, y)
    mix = apply_alpha(blend, stack, a)
    print(f"alpha-mix (alpha={a:.3f})       : {s_mix:.5f}")

    # -----------------------------------------------------------------------
    # 4. Bias calibration
    # -----------------------------------------------------------------------
    b1, f_single = fit_bias_single(mix, y)
    print(f"+ bias single-shot           : {f_single:.5f}")
    b2, f_stable = fit_bias_bootstrap(mix, y, n_boot=15)
    print(f"+ bias bootstrap-stable      : {f_stable:.5f}  <-- submitted")
    print(f"  bias vector: {b2.round(4).tolist()}")

    # -----------------------------------------------------------------------
    # 5. Per-class report
    # -----------------------------------------------------------------------
    pred = apply_bias(mix, b2).argmax(1)
    print("\nPer-class F1 (bootstrap-stable bias):")
    print(classification_report(y, pred, target_names=LABELS, digits=4))

    # -----------------------------------------------------------------------
    # 6. Optional: generate figures
    # -----------------------------------------------------------------------
    if gen_figures:
        print("Generating figures …")
        from disaster.eda.plots import generate_all
        generate_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", action="store_true",
                        help="Also regenerate all paper figures")
    args = parser.parse_args()
    main(gen_figures=args.figures)
