import pandas as pd

FOLDS = "data/folds/folds_canonical.csv"


def test_no_row_in_two_folds():
    df = pd.read_csv(FOLDS, encoding="utf-8-sig")
    assert df["fold"].between(0, 4).all()
    assert df["image_id"].is_unique


def test_stratification_preserved():
    df = pd.read_csv(FOLDS, encoding="utf-8-sig")
    overall = df["label"].value_counts(normalize=True).sort_index()
    for f in sorted(df["fold"].unique()):
        frac = df[df.fold == f]["label"].value_counts(normalize=True).sort_index()
        assert (abs(overall - frac) < 0.02).all(), f"fold {f} not stratified"
