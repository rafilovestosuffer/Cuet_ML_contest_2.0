# RECONSTRUCTED REFERENCE IMPLEMENTATION — generated from specification.
# This is NOT the original training code and is NOT verified to reproduce the
# released OOF arrays. Reconcile with the author's original notebooks.

"""Train a text encoder branch (BanglaBERT / MuRIL) with 5-fold OOF.

Multi-seed support: pass repeated --seed flags (the notebook averaged BanglaBERT
over seeds 123 and 456 before saving a single OOF array).

Usage:
    PYTHONPATH=src python -m disaster.train.train_text \\
        --config configs/text_muril_large.yaml \\
        --train-csv data/raw/Disaster_train.csv
"""
import argparse

import pandas as pd

from disaster.config import load_config
from disaster.data.dataset import TextDataset
from disaster.data.folds import load_folds
from disaster.data.tokenization import get_tokenizer
from disaster.labels import LABEL2IDX
from disaster.models.text_encoder import TextClassifier
from disaster.train._common import save_outputs, train_kfold_oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-csv", default="data/raw/Disaster_train.csv")
    ap.add_argument("--test-csv", default=None)
    ap.add_argument("--seed", type=int, action="append", default=None)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = args.seed or cfg.get("seeds", [cfg.get("seed", 42)])
    tok = get_tokenizer(cfg["text_model"])

    df = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["label"] = df["category"].map(LABEL2IDX)
    folds = load_folds()[["image_id", "fold"]]
    df = df.merge(folds, on="image_id", how="left")
    df["fold"] = df["fold"].astype(int)

    test_df = None
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv, encoding="utf-8-sig")
        test_df.columns = [c.strip() for c in test_df.columns]

    def dataset_factory(frame, is_test):
        return TextDataset(frame, tok, cfg["max_len"], is_test=is_test)

    def batch_to_inputs(batch, dev):
        inputs = dict(input_ids=batch["input_ids"].to(dev),
                      attention_mask=batch["attention_mask"].to(dev))
        labels = batch["labels"].to(dev) if "labels" in batch else None
        return inputs, labels

    class _Net(TextClassifier):
        def forward(self, input_ids, attention_mask):
            logits, _ = super().forward(input_ids, attention_mask)
            return logits

    # Average OOF/test over seeds (the multi-seed recipe).
    oof_sum, test_sum = None, None
    for s in seeds:
        cfg["seed"] = s
        cfg["lr_backbone"] = cfg.get("lr_text", 8e-6)
        oof, test = train_kfold_oof(
            df, cfg, lambda: _Net(cfg["text_model"]),
            lambda m: [m.backbone], dataset_factory, batch_to_inputs,
            test_df=test_df)
        oof_sum = oof if oof_sum is None else oof_sum + oof
        if test is not None:
            test_sum = test if test_sum is None else test_sum + test

    oof_avg = oof_sum / len(seeds)
    test_avg = test_sum / len(seeds) if test_sum is not None else None
    save_outputs(oof_avg, test_avg, args.out, cfg.get("run_id", cfg["tag"]))


if __name__ == "__main__":
    main()
