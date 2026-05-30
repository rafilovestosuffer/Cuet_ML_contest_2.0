"""Convex mix between the weighted blend and the stacking meta-learner."""
import numpy as np
from sklearn.metrics import f1_score


def search_alpha(blend_oof: np.ndarray, stack_oof: np.ndarray, y: np.ndarray,
                 grid: int = 201):
    """Return (best_alpha, best_macro_f1) for alpha*blend + (1-alpha)*stack."""
    best_a, best_f = 0.0, 0.0
    for a in np.linspace(0, 1, grid):
        f = f1_score(y, (a * blend_oof + (1 - a) * stack_oof).argmax(1),
                     average="macro")
        if f > best_f:
            best_f, best_a = f, a
    return best_a, best_f


def apply_alpha(blend: np.ndarray, stack: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * blend + (1 - alpha) * stack
