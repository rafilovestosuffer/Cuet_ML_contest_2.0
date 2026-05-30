from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from disaster.labels import LABEL2IDX


def build_folds(train_csv: str | Path, n_splits: int = 5, seed: int = 42,
                out_csv: str | Path = "data/folds/folds_canonical.csv") -> pd.DataFrame:
    df = pd.read_csv(train_csv, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["label"] = df["category"].map(LABEL2IDX)
    assert df["label"].notna().all(), "Unmapped category found; check label spelling."
    df["label"] = df["label"].astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1
    for f, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = f
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df


def load_folds(path: str | Path = "data/folds/folds_canonical.csv") -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")
