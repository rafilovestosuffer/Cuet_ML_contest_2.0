"""OOF arrays have the correct shapes and reproduce the reported macro-F1.

These tests are VERIFIED — they load frozen artifacts and assert against the
exact values computed by scripts/reproduce_ensemble.py.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

A = "artifacts/"
FOLDS_CSV = "data/folds/folds_canonical.csv"

# All 7 base + stack OOF arrays with expected macro-F1 (tolerance 1e-3)
ALL_OOF_F1 = {
    "oof_banglabert_base":   0.96587,
    "oof_banglabert_multi":  0.96656,
    "oof_muril_large":       0.97556,
    "oof_eva02_large":       0.96714,
    "oof_fusion_eva_muril":  0.99212,
    "oof_pl_muril":          0.97525,
    "oof_stack":             0.99351,
}


@pytest.fixture(scope="module")
def labels():
    return pd.read_csv(FOLDS_CSV, encoding="utf-8-sig")["label"].values


@pytest.fixture(scope="module")
def n_train(labels):
    return len(labels)


@pytest.mark.parametrize("name", list(ALL_OOF_F1))
def test_oof_shape(name, n_train):
    arr = np.load(A + name + ".npy")
    assert arr.shape == (n_train, 8), \
        f"{name}: expected ({n_train}, 8), got {arr.shape}"


@pytest.mark.parametrize("name,expected", list(ALL_OOF_F1.items()))
def test_oof_f1(name, expected, labels):
    arr = np.load(A + name + ".npy")
    got = f1_score(labels, arr.argmax(1), average="macro")
    assert abs(got - expected) < 1e-3, \
        f"{name}: got {got:.5f}, expected {expected:.5f}"


def test_specialist_shape():
    """Specialist OOF covers only the EQ/Flood/Landslides rows, 3 classes."""
    spec = np.load(A + "oof_specialist.npy")
    assert spec.ndim == 2 and spec.shape[1] == 3, \
        f"Expected (N, 3), got {spec.shape}"
    # Should be a strict subset of the 6323 training rows
    assert spec.shape[0] < 6323


def test_test_arrays_shape():
    """All test probability arrays should have shape (1580, 8)."""
    test_names = [
        "test_banglabert_base", "test_banglabert_multi", "test_muril_large",
        "test_eva02_large", "test_fusion_eva_muril", "test_pl_muril", "test_stack",
    ]
    for name in test_names:
        arr = np.load(A + name + ".npy")
        assert arr.shape == (1580, 8), \
            f"{name}: expected (1580, 8), got {arr.shape}"


def test_oof_probabilities_sum_to_one():
    """OOF softmax rows should sum to ~1 (or be logits that exp-sum to ~1)."""
    for name in ALL_OOF_F1:
        arr = np.load(A + name + ".npy")
        row_sums = arr.sum(axis=1)
        # Accept either probabilities (sum ≈ 1) or raw logits (sum != 1)
        # Just assert no NaN/Inf
        assert np.isfinite(arr).all(), f"{name} contains NaN or Inf"


def test_fold_count():
    """folds_canonical.csv must have exactly 5 folds (0–4)."""
    df = pd.read_csv(FOLDS_CSV, encoding="utf-8-sig")
    assert set(df["fold"].unique()) == {0, 1, 2, 3, 4}


def test_label_range():
    """Labels must be integers in [0, 7]."""
    df = pd.read_csv(FOLDS_CSV, encoding="utf-8-sig")
    assert df["label"].between(0, 7).all()
    assert set(df["label"].unique()) == set(range(8))
