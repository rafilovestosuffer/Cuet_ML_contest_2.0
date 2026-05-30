# RECONSTRUCTED REFERENCE IMPLEMENTATION — generated from specification.
# This is NOT the original training code and is NOT verified to reproduce the
# released OOF arrays. Reconcile with the author's original notebooks.
#
# EXCEPTION: `apply_spec` and `search_beta` below are TRANSCRIBED from the
# notebook (pure array math) and ARE runnable/verifiable on the frozen specialist
# OOF array.

"""3-class specialist on {Earthquake, Flood, Landslides} + power-fusion.

The specialist's 3-class posterior is fused multiplicatively into the 8-class
posterior:
    p_c  ←  p_c · (p_spec_c)^beta      for c in {Earthquake, Flood, Landslides}
then renormalized. `beta` is chosen by OOF grid search.
"""
import argparse

import numpy as np

# Class ids the specialist covers, in the order of its 3-class output.
SPEC_IDS = [1, 2, 4]  # Earthquake, Flood, Landslides
EPS = 1e-12


def apply_spec(p8: np.ndarray, p3: np.ndarray, beta: float) -> np.ndarray:
    """Multiplicative power-fusion of a 3-class specialist into 8-class probs.

    TRANSCRIBED from the notebook.
    """
    out = p8.copy()
    for li, gi in enumerate(SPEC_IDS):
        out[:, gi] *= (p3[:, li] ** beta)
    return out / out.sum(1, keepdims=True).clip(min=EPS)


def expand_spec_oof(spec_oof: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Scatter the (n_spec_rows, 3) specialist OOF back into (N, 3), zeros elsewhere.

    The specialist OOF only covers rows whose true label is in SPEC_IDS.
    """
    mask = np.isin(labels, SPEC_IDS)
    full = np.zeros((len(labels), 3), np.float32)
    full[mask] = spec_oof
    return full


def search_beta(p8: np.ndarray, spec_full: np.ndarray, y: np.ndarray,
                bias: np.ndarray | None = None,
                grid=np.linspace(0.3, 1.5, 7)):
    """Grid-search beta to maximize OOF macro-F1 (notebook grid: linspace(0.3,1.5,7))."""
    from sklearn.metrics import f1_score
    best_b, best_f = grid[0], -1.0
    for b in grid:
        corr = apply_spec(p8, spec_full, b)
        logits = np.log(corr + EPS) + (bias if bias is not None else 0.0)
        f = f1_score(y, logits.argmax(1), average="macro")
        if f > best_f:
            best_f, best_b = f, b
    return best_b, best_f


def main():
    ap = argparse.ArgumentParser(
        description="Demonstrate specialist power-fusion on frozen artifacts.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if args.config is None:
        # Verified demo: run beta search on committed artifacts.
        import pandas as pd
        y = pd.read_csv("data/folds/folds_canonical.csv",
                        encoding="utf-8-sig")["label"].values
        p8 = np.load("artifacts/oof_stack.npy")
        spec_full = expand_spec_oof(np.load("artifacts/oof_specialist.npy"), y)
        bias = np.load("artifacts/bias_s3e.npy")
        b, f = search_beta(p8, spec_full, y, bias=bias)
        print(f"Best beta={b:.2f}  OOF macro-F1={f:.5f}")
    else:
        raise NotImplementedError(
            "Specialist training loop is reconstructed; see module docstring. "
            "Run without --config for the verified power-fusion demo.")


if __name__ == "__main__":
    main()
