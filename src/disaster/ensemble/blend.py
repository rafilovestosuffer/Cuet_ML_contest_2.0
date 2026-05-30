"""Dirichlet-restart convex blend, optimizing macro-F1 directly (Nelder-Mead).

VERIFIED — this reproduces the paper exactly. The RNG seeding matches the
original notebook (legacy ``np.random.seed(trial)`` + ``np.random.dirichlet``)
so the optimized weights, alpha-mix, and downstream bias all land on the
reported values:

    blend OOF macro-F1 = 0.99495
    weights ≈ {bb_base:0.069, bb_multi:0.004, muril:0.180,
               eva02:0.443, fusion:0.233, pl:0.071}

NOTE: macro-F1 has multiple near-optima in weight space, so a different RNG
(e.g. ``np.random.default_rng``) also reaches 0.99495 but with a *different*
weight vector, which then shifts the downstream bias-calibration result by
~2e-4. To reproduce the paper's 0.9956 final number, the legacy seeding below
must be used. Do not "modernize" the RNG without re-checking REPRODUCE.md.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import f1_score


def _neg_macro_f1(w, oofs, y):
    w = np.maximum(w, 0)
    s = w.sum()
    if s < 1e-9:
        return 0.0
    w = w / s
    p = sum(wi * oi for wi, oi in zip(w, oofs))
    return -f1_score(y, p.argmax(1), average="macro")


def search_blend_weights(oofs: list[np.ndarray], y: np.ndarray,
                         n_restarts: int = 50, seed: int = 0):
    """Return (best_weights, best_macro_f1).

    Faithful to the notebook: each restart ``t`` seeds the legacy global RNG
    with ``seed + t`` and draws a Dirichlet starting point, then refines with
    Nelder-Mead. With the default ``seed=0`` this reproduces the paper weights.
    """
    best_w, best_s = None, 0.0
    for t in range(n_restarts):
        np.random.seed(seed + t)
        w0 = np.random.dirichlet(np.ones(len(oofs)))
        r = minimize(_neg_macro_f1, w0, args=(oofs, y), method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 8000})
        if -r.fun > best_s:
            best_s = -r.fun
            best_w = np.maximum(r.x, 0)
            best_w = best_w / best_w.sum()
    return best_w, best_s


def apply_blend(oofs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return sum(w * o for w, o in zip(weights, oofs))
