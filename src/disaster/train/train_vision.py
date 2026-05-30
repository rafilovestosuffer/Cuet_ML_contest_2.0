# RECONSTRUCTED REFERENCE IMPLEMENTATION — generated from specification.
# This is NOT the original training code and is NOT verified to reproduce the
# released OOF arrays. Reconcile with the author's original notebooks.

"""Train a vision encoder branch (EVA-02-Large / ConvNeXt V2-base) with 5-fold OOF.

Usage:
    PYTHONPATH=src python -m disaster.train.train_vision \\
        --config configs/vision_eva02_large.yaml \\
        --train-csv data/raw/Disaster_train.csv --train-img data/Train
"""
import argparse

import pandas as pd

from disaster.config import load_config
from disaster.data.dataset import ImageDataset, resolve_img
from disaster.data.folds import load_folds
from disaster.data.transforms import build_train_tfms, build_val_tfms
from disaster.labels import LABEL2IDX
from disaster.models.vision_encoder import VisionClassifier
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
    cfg["lr_backbone"] = cfg.get("lr_image", 1e-5)

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
        return ImageDataset(frame, cfg["img_size"],
                            val_tfm if is_test else train_tfm, is_test=is_test)

    def batch_to_inputs(batch, dev):
        inputs = dict(x=batch["image"].to(dev))
        labels = batch["labels"].to(dev) if "labels" in batch else None
        return inputs, labels

    class _Net(VisionClassifier):
        def forward(self, x):
            logits, _ = super().forward(x)
            return logits

    oof, test = train_kfold_oof(
        df, cfg, lambda: _Net(cfg["img_model"], drop_path_rate=cfg.get("drop_path", 0.10)),
        lambda m: [m.backbone], dataset_factory, batch_to_inputs, test_df=test_df)
    save_outputs(oof, test, args.out, cfg.get("run_id", cfg["tag"]))


if __name__ == "__main__":
    main()
