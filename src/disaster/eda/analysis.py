"""Dataset exploratory analysis used to generate the paper's EDA section."""
import re
import pandas as pd

EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "]", flags=re.UNICODE)


def class_distribution(train_csv: str) -> pd.Series:
    df = pd.read_csv(train_csv, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df["category"].value_counts().sort_index()


def text_length_stats(train_csv: str) -> dict:
    df = pd.read_csv(train_csv, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    n = df["context"].astype(str).str.len()
    return dict(mean=float(n.mean()), median=float(n.median()),
                max=int(n.max()), min=int(n.min()))


def emoji_prevalence(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    has = df["context"].astype(str).apply(lambda x: bool(EMOJI_RE.search(x)))
    return dict(rows_with_emoji=int(has.sum()), total=int(len(df)),
                fraction=float(has.mean()))
