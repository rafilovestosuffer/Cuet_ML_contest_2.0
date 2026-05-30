"""OOF arrays have the right shape and reproduce the reported macro-F1."""
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

A = "artifacts/"
EXPECTED = {
    "oof_muril_large": 0.97556,
    "oof_fusion_eva_muril": 0.99212,
    "oof_stack": 0.99351,
}


def _y():
    return pd.read_csv("data/folds/folds_canonical.csv",
                       encoding="utf-8-sig")["label"].values


def test_shapes():
    y = _y()
    for n in EXPECTED:
        a = np.load(A + n + ".npy")
        assert a.shape == (len(y), 8)


def test_individual_f1_matches_paper():
    y = _y()
    for n, exp in EXPECTED.items():
        got = f1_score(y, np.load(A + n + ".npy").argmax(1), average="macro")
        assert abs(got - exp) < 1e-3, f"{n}: {got:.5f} != {exp}"
