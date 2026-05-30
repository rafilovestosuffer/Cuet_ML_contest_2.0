# RECONSTRUCTED REFERENCE IMPLEMENTATION — generated from specification.
# This is NOT the original training code and is NOT verified to reproduce the
# released OOF arrays. Reconcile with the author's original notebooks.

"""Train the cross-attention fusion model (JointMM) with 5-fold OOF.

Covers both backbone variants via the config's `img_model` / `img_dim`:
  configs/fusion_eva02_muril.yaml      (EVA-02-Large × MuRIL)
  configs/fusion_convnextv2_muril.yaml (ConvNeXt V2-base × MuRIL — notebook run)

Usage:
    PYTHONPATH=src python -m disaster.train.train_fusion \\
        --config configs/fusion_eva02_muril.yaml \\
        --train-csv data/raw/Disaster_train.csv --train-img data/Train \\
        --test-csv data/raw/Disaster_test.csv  --test-img data/Test
"""
import argparse

import pandas as pd

from disaster.config import load_config
from disaster.data.dataset import MMDataset, resolve_img
from disaster.data.folds import load_folds
from disaster.data.tokenization import get_tokenizer
from disaster.data.transforms import build_train_tfms, build_val_tfms
from disaster.labels import LABEL2IDX
from disaster.models.fusion import JointMM
from disaster.train._common import save_outputs, train_kfold_oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-csv", default="data/raw/Disaster_train.csv")
    ap.add_argument("--train-img", default="data/Train")
    ap.add_argument("--test-csv", default=None)
    ap.add_argument("--test-img", default="data/Test")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tok = get_tokenizer(cfg["text_model"])

    df = pd.read_csv(args.train_csv, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["label"] = df["category"].map(LABEL2IDX)
    folds = load_folds()[["image_id", "fold"]]
    df = df.merge(folds, on="image_id", how="left")
    df["fold"] = df["fold"].astype(int)
    df["img_path"] = df["image_id"].apply(lambda x: resolve_img(x, args.train_img))

    test_df = None
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv, encoding="utf-8-sig")
        test_df.columns = [c.strip() for c in test_df.columns]
        test_df["img_path"] = test_df["image_id"].apply(
            lambda x: resolve_img(x, args.test_img))

    train_tfm, val_tfm = build_train_tfms(), build_val_tfms()

    def dataset_factory(frame, is_test):
        return MMDataset(frame, tok, cfg["max_len"], cfg["img_size"],
                         val_tfm if is_test else train_tfm, is_test=is_test)

    def batch_to_inputs(batch, dev):
        inputs = dict(
            image=batch["image"].to(dev),
            input_ids=batch["input_ids"].to(dev),
            attention_mask=batch["attention_mask"].to(dev),
        )
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"].to(dev)
        labels = batch["labels"].to(dev) if "labels" in batch else None
        return inputs, labels

    oof, test = train_kfold_oof(
        df, cfg, lambda: JointMM(cfg), lambda m: [m.img_enc, m.text_enc],
        dataset_factory, batch_to_inputs, test_df=test_df)
    save_outputs(oof, test, args.out, cfg.get("run_id", cfg["tag"]))


if __name__ == "__main__":
    main()
