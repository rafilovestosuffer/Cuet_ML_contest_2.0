import numpy as np
from sklearn.metrics import f1_score

EPS = 1e-12


def _score(logp, bias, y):
    return f1_score(y, (logp + bias).argmax(1), average="macro")


def fit_bias_single(probs: np.ndarray, y: np.ndarray,
                    lo: float = -0.5, hi: float = 0.5, step: float = 0.01,
                    passes: int = 8):
    logp = np.log(probs + EPS)
    grid = np.arange(lo, hi + step / 2, step)
    bias = np.zeros(probs.shape[1])
    for _ in range(passes):
        for c in range(probs.shape[1]):
            orig = bias[c]
            best_b, best_s = orig, _score(logp, bias, y)
            for d in grid:
                bias[c] = orig + d
                v = _score(logp, bias, y)
                if v > best_s:
                    best_s, best_b = v, bias[c]
            bias[c] = best_b
    return bias, _score(logp, bias, y)


def fit_bias_bootstrap(probs: np.ndarray, y: np.ndarray, n_boot: int = 15,
                       passes: int = 4, seed: int = 0):
    n = len(y)
    biases = []
    for s in range(n_boot):
        rng = np.random.default_rng(seed + s)
        idx = rng.choice(n, n, replace=True)
        b, _ = fit_bias_single(probs[idx], y[idx], passes=passes)
        biases.append(b)
    bias = np.median(np.stack(biases), axis=0)
    return bias, _score(np.log(probs + EPS), bias, y)


def apply_bias(probs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (np.log(probs + EPS) + bias)
