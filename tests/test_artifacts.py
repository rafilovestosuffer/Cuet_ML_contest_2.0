import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

pytestmark = pytest.mark.needs_artifacts

A = "artifacts/"
FOLDS_CSV = "data/folds/folds_canonical.csv"

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
    spec = np.load(A + "oof_specialist.npy")
    assert spec.ndim == 2 and spec.shape[1] == 3, \
        f"Expected (N, 3), got {spec.shape}"
    assert spec.shape[0] < 6323


def test_test_arrays_shape():
    test_names = [
        "test_banglabert_base", "test_banglabert_multi", "test_muril_large",
        "test_eva02_large", "test_fusion_eva_muril", "test_pl_muril", "test_stack",
    ]
    for name in test_names:
        arr = np.load(A + name + ".npy")
        assert arr.shape == (1580, 8), \
            f"{name}: expected (1580, 8), got {arr.shape}"


def test_oof_probabilities_sum_to_one():
    for name in ALL_OOF_F1:
        arr = np.load(A + name + ".npy")
        assert np.isfinite(arr).all(), f"{name} contains NaN or Inf"


def test_fold_count():
    df = pd.read_csv(FOLDS_CSV, encoding="utf-8-sig")
    assert set(df["fold"].unique()) == {0, 1, 2, 3, 4}


def test_label_range():
    df = pd.read_csv(FOLDS_CSV, encoding="utf-8-sig")
    assert df["label"].between(0, 7).all()
    assert set(df["label"].unique()) == set(range(8))
