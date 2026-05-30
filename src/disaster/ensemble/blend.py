"""Dirichlet-restart convex blend, optimizing macro-F1 directly (Nelder-Mead)."""
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
    """Return (best_weights, best_macro_f1)."""
    best_w, best_s = None, 0.0
    for t in range(n_restarts):
        rng = np.random.default_rng(seed + t)
        w0 = rng.dirichlet(np.ones(len(oofs)))
        r = minimize(_neg_macro_f1, w0, args=(oofs, y), method="Nelder-Mead",
                     options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 8000})
        if -r.fun > best_s:
            best_s = -r.fun
            best_w = np.maximum(r.x, 0)
            best_w = best_w / best_w.sum()
    return best_w, best_s


def apply_blend(oofs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return sum(w * o for w, o in zip(weights, oofs))
